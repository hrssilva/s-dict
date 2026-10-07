# Dictionary architecture and implementation status

This crate owns the bidirectional mapping between RDF terms and stable
`TermId`s. It does not own graph/triple storage (`s-graph`'s job), RDF
parsing, named-graph semantics, or query-language algebra.

## The central idea

$$
\text{macro-dynamic organization} + \text{micro-static compression}
$$

Partition the lexicographically ordered term space into moderate-sized
**blocks**. Each block is a small, independently rebuildable structure
supporting rank (`term -> local rank`) and select (`local rank -> term`).
A small global router sends a query to the right block; a small reverse
directory sends a `TermId` to the right block. An insertion or deletion
touches at most one or two blocks, never the whole dictionary -- the same
principle `s-graph` already applies to its DFS-preorder radix trie.

A global minimal automaton over the whole term set would compress better
but destroys update locality (any change can touch the whole structure). A
fully dynamic trie preserves locality but gives up most of the sharing a
static, minimized structure gets from common prefixes/suffixes and
right-language equivalence. Blocks expose a continuum between those two
extremes, with block size `B` as the tuning knob.

## Per-block model

For a block `b` with sorted terms `(s_0, ..., s_{m-1})`:

- `F_b`: `s_i <-> i` -- the block's forward index (rank/select).
- `P_b[i] = f(s_i)`: the block-local external-id sequence.

Forward lookup: `term -> block -> rank_b(term) -> P_b[rank] -> TermId`.
Reverse lookup: `TermId -> block -> local rank -> select_b(rank) -> term`.

`F_b` is a minimal per-block finite-state map (a "micro-FST"), immutable
between rebuilds -- Stage 3, implemented via the off-the-shelf
[`fst`](https://docs.rs/fst) crate rather than a hand-rolled Daciuk-style
incremental automaton (see the Stage 3 section below for what that
concretely buys and costs).

## What's implemented (Stages 0, 1, 3, 5, plus one §13 baseline)

### Stage 0 -- `PlainDictionary` (`src/plain.rs`)

The design doc's own §13 "plain baseline": a `HashMap<Box<str>, TermId>`
forward index plus a dense `Vec<Option<Box<str>>>` reverse index, with
freed ids reused via a free-list stack. No blocks, no lexical ordering, no
compression. This is kept permanently, not thrown away once `Dictionary`
exists -- later stages are meant to be benchmarked against it.

### Stage 1 -- `Dictionary` (`src/dictionary.rs`, `src/block.rs`)

Terms partitioned into contiguous lexical blocks (`src/block.rs`'s
`Block`), each bounded to `[block_min, block_max]` terms. Blocks live in a
slab (`Vec<Option<Block>>`); a `BlockId` is a stable slot index that stays
valid across splits/merges of *other* blocks, avoiding a pointer-heavy
object graph (design doc §10).

Stage 1 as literally specified in the design doc ("introduce lexical
blocks and stable BlockIds, keep strings uncompressed") has no router yet
(that's nominally Stage 2) and no split/merge (§7 isn't pinned to a
specific stage). But a block collection isn't queryable at all without
some way to find the right block, and isn't bounded without some way to
keep blocks from growing forever -- both are intrinsic to lexical blocks
actually working, so this implementation pulls the simplest possible
version of each forward:

- **Router**: a sorted `Vec<(min_key, BlockId)>`, binary-searched via
  `partition_point`. This is explicitly one of the doc's listed router
  candidates ("another compact ordered predecessor structure"), just the
  simplest one -- not a radix/B-tree. Block 0 implicitly covers everything
  below its own `min_key` too, so no `-inf` sentinel key is needed.
- **Split**: on overflow (`len() > block_max`), median-split into two
  blocks and insert one new directory entry. `block_max >= 2 * block_min`
  is enforced so both halves always stay `>= block_min` after a
  single-term-triggered split.
- **Merge**: on underflow (`0 < len() < block_min`) or emptiness
  (`len() == 0`), absorb into (or drop into) a lexically adjacent
  neighbor, updating that neighbor's directory entry and every moved
  term's reverse-directory entry, then re-splitting if the merge pushed
  the combined block back over `block_max`.
- **Reverse mapping**: §5 **Variant A only** -- `TermId -> BlockId` as a
  dense `Vec<Option<BlockId>>`, then a linear scan of the owning block's
  local `ids` array to recover local rank. Variant B (`TermId -> (BlockId,
  local_rank)` directly) is an explicit Stage-5 evaluation in the doc and
  is not implemented; Variant A was chosen as the simpler starting point
  its own §5 recommends, and because it makes insertion/removal not have
  to touch any other term's reverse-directory entry.
- **Batch insertion** (`insert_batch`): sorts and dedupes the input once,
  groups it into contiguous runs by destination block (one router lookup
  per run, not per term), and merges each run into its block with one
  linear pass (`Block::merge_sorted`) instead of doing `Vec::insert` once
  per term. This is the doc's explicit §10/H5 requirement ("if 100 new
  terms target the same block, rebuild that block once"). An empty
  dictionary's first batch bootstraps directly into balanced
  `block_max`-sized chunks (evening out the remainder across the last
  couple of chunks) rather than going through the single-block bootstrap
  path term-by-term.

Both `PlainDictionary` and `Dictionary` implement the shared
`TermDictionary` trait (`src/lib.rs`), so later stages can be swapped in
and benchmarked against the same call sites (design doc §13).

### Stage 3 -- micro-FST blocks via the `fst` crate (`src/block.rs`)

Each block's `terms: Vec<Box<str>>` (Stage 1) is replaced by an
`fst::Map<Vec<u8>>` mapping each sorted term to its local rank as a `u64`,
plus the same `ids: Vec<TermId>` (`P_b`) as before. A cached `min_key:
Box<str>` field stores the block's smallest term redundantly, so routing
and directory maintenance never have to touch the FST (design doc §10:
separate hot metadata from the compressed cold payload).

The `fst` crate is a well-tested general byte-string FST library. An
earlier version of this document claimed it had "no native unranking" and
that getting real `select_b` (§2) would require hand-rolling a custom
automaton augmented with right-language cardinalities. **That was wrong**,
caught on review of the design doc's own §2/§3/§6 against what the crate
actually exposes at its lower `raw` layer:

- **`select_b` is native, not stream-skip.** `Map::as_fst()` exposes the
  underlying `raw::Fst`, which has `get_key(value: u64) -> Option<Vec<u8>>`
  (and a buffer-reusing `get_key_into`). Its implementation
  (`fst-0.4.7/src/raw/mod.rs`) walks from the root and, at each node,
  takes the *last* outgoing transition whose accumulated output value is
  `<= remaining value`, subtracts it, and descends -- exactly the
  right-language-cardinality-guided unranking algorithm §2 asks for
  ("augmented with right-language cardinalities sufficient to support...
  `select_b(i)=s_i`"), except the `fst` crate already ships it. Its own
  doc comment states the precondition plainly: "the values in this FST
  are not monotonically increasing when sorted lexicographically by key,
  then this routine has unspecified behavior." Since `Block::build` stores
  each key's *local rank* as its output value -- which is by construction
  monotonically increasing in lexicographic key order -- that precondition
  holds automatically, with no extra bookkeeping required. `Block::term_at`
  now calls this directly, at O(term length * branching factor per node)
  instead of the O(local_rank) stream-skip it used before.

  Measured, not assumed: an A/B on the same `perf_probe_default_block_bounds`
  probe (20,000 terms, default 64..=256 block bounds) gave **2.805µs/op**
  for the old stream-skip `term_at` vs. **384ns/op** for `get_key` --
  about **7.3x faster**, and it also cut the differential fuzz suite's
  wall time from ~1.06s to ~0.68s (that suite calls `get_term` heavily in
  its lockstep verification, so it was already exercising this path). No
  custom automaton was needed; the gap was in this crate's usage of the
  library, not in the library's capabilities.
- **No incremental mutation.** `fst::Map` is immutable once built (there's
  no insertion-point primitive for absent keys either, only
  present/absent). Every mutating `Dictionary` operation --
  `insert`, `remove`, `split`, `merge_with_neighbor`, and each
  `insert_batch` block-group -- decodes the affected block(s), mutates the
  result, and calls `Block::build` to replace the block wholesale. This is
  exactly the update procedure design doc §6 prescribes for FST-backed
  blocks ("enumerate/decode terms into reusable scratch memory... build
  new minimal FST... atomically replace"), and it's O(block size) either
  way -- the same complexity class as Stage 1's `Vec::insert` shifting,
  just via a decode+rebuild pass instead of a memmove. (Decoding and
  rebuilding used a `Vec<(Box<str>, TermId)>` scratch buffer originally;
  see the "Optimization pass" section below for why and how that changed.)

  Measured, not assumed: reusing that scratch buffer's outer `Vec` across
  calls did **not** produce a measurable improvement on its own. The
  `(block_min=2, block_max=8)` fuzz suite's wall time was unchanged
  (~1.06s before and after, same test, same machine). A temporary
  `#[ignore]`d probe (`tests/dictionary.rs`,
  `perf_probe_default_block_bounds`; run with
  `cargo test --release -- --ignored --nocapture`) doing 20,000 sequential
  inserts then removes at the *default* block bounds (64..=256) measured
  ~83µs/insert and ~69µs/remove, and an A/B with scratch reuse temporarily
  disabled on the `insert` hot path measured ~84µs/insert -- within noise
  of each other. So the outer `Vec`'s allocation churn was never the
  dominant cost at either block-size regime tried.

  A follow-up probe (`src/block.rs`, `perf_probe_block_rebuild_breakdown`,
  same `--ignored --nocapture` invocation) isolated *where* the ~83µs
  actually goes, at a fixed 160-entry block (the midpoint of the default
  64..=256 bounds), 5,000 iterations each. Numbers below are post-
  "Optimization pass" (see that section); `decode_into` was 11.95µs/op
  and the per-string clone floor 4.45µs/op before it:

  | Step | Cost |
  |---|---:|
  | `Block::decode_into` (existing block -> arena) | 7.25µs/op |
  | `Block::build` (fresh `MapBuilder` insert loop + finalize) | **59.59µs/op** |
  | pure `String` clone floor (no FST work at all) | 5.44µs/op |
  | decode + build (~ one `insert`'s rebuild cost) | 66.84µs/op |

  `Block::build` -- the `fst` crate's own per-key minimization work as
  `MapBuilder::insert` is called once per sorted key, plus finalization --
  is unambiguously the dominant cost, ~89% of the decode+build total, and
  decode+build alone already accounts for essentially all of the ~83µs
  measured end-to-end for a full `Dictionary::insert` (the remainder is
  routing/id-allocation/directory bookkeeping, all much smaller).
  `decode_into` costs about 1.3x the pure-allocation floor now (was 2.7x
  before the optimization pass), so most of its remaining cost is genuine
  FST-walking, not allocation -- but it's still a minor contributor next
  to `Block::build`. Per-entry allocation *was* a real, fixable cost (see
  "Optimization pass" below, ~40% off `decode_into` alone), but
  `Block::build`'s minimization work rules out the bigger question this
  section originally asked: it's specifically the general-purpose `fst`
  crate doing real automaton work on every one of a block's keys, on
  every single-term mutation, which by construction only gets cheaper by
  rebuilding less often (bigger batches/`insert_batch`, already available)
  or building cheaper (a specialized construction path for small bounded
  blocks -- the "custom automaton" possibility above), not by touching
  decode or allocation further.

One API consequence reaches all the way to the crate's public trait:
`TermDictionary::get_term` now returns `Option<String>` rather than
`Option<&str>`. `Block::term_at` (via `Fst::get_key`) and `decode_into`
(via a full stream) both hand back an owned `Vec<u8>`/key slice that has
to be copied into an owned `String`/`Box<str>` before the function that
produced it returns -- there's no `&self`-lifetime borrow to hand out
either way, whether the key comes from a bounded `get_key` descent or a
`Stream`'s internal buffer. `PlainDictionary::get_term` pays an
unnecessary allocation it didn't before, purely to keep both
implementations comparable under one trait (design doc §13); it could
still return a zero-copy borrow through its own inherent methods if that
regression ever matters enough to measure.

### Front-coded blocks -- the required §8/§13 baseline (`src/front_coded_block.rs`, `src/front_coded_dictionary.rs`)

§8 poses the actual question this architecture needs to answer, and it
isn't "can an FST compress RDF strings": *"how much additional
compression does a minimal finite-state representation obtain over much
simpler blocked front coding, once both are constrained to the same
update locality?"* Answering that requires a second compressed block
representation built the same way, which didn't exist until now --
`Dictionary` (FST) had nothing to be compared against except the fully
uncompressed `PlainDictionary`.

`FrontCodedDictionary` is the same macro structure as `Dictionary` --
identical router, slab, split/merge, `entries_scratch` reuse, batch
grouping, all duplicated rather than shared behind a generic block
abstraction (§16 anti-goal: "a highly polymorphic framework of
interchangeable node objects"; also, sharing machinery between the two
things being measured against each other risks a bug in the shared part
masking a real difference) -- with `FrontCodedBlock` in place of the
FST `Block`. Each sorted term is stored as a delta against its
predecessor, `(shared_prefix_len, suffix_bytes)`, in plain parallel
`Vec<u32>`/`Vec<u8>` arrays (§11: no bit-packing yet), with a full string
stored every `RESTART_INTERVAL = 16` entries (a single fixed constant,
not yet tuned) so random-access `rank`/`term_at` cost O(RESTART_INTERVAL)
via a restart-point binary search, not O(block size). `common_prefix_len`
clamps its raw byte-comparison result back to the nearest UTF-8 char
boundary before it's used as a slice index -- a real correctness hazard
this specific technique has and an automaton-based block doesn't, since
two independently-valid-UTF-8 strings' matching byte span isn't
guaranteed to end on a codepoint boundary. Verified by adding
`FrontCodedDictionary` as a third structure (alongside `Dictionary` and
the `PlainDictionary` oracle) to the existing differential fuzz suite,
all in lockstep -- it passed clean on the first run, including the
multibyte-UTF-8 regression test added specifically for the char-boundary
clamp.

Front coding captures the same shared-*prefix* redundancy an
automaton does, but not shared *suffixes*/equivalent right-languages,
and its rebuild is just recomputing prefix deltas -- no minimization
algorithm. Measured, not assumed (`tests/dictionary.rs`,
`perf_probe_space_and_rebuild_fst_vs_front_coded`, 9,000 namespace-
clustered RDF-like terms across 6 real namespaces, default 64..=256
block bounds, `cargo test --release -- --ignored --nocapture`):

| | FST (`Dictionary`) | Front-coded (`FrontCodedDictionary`) |
|---|---:|---:|
| Space | **13.84 bytes/term** | 24.07 bytes/term |
| Sequential insert (rebuild) | 91.08µs/op | **14.12µs/op** |
| `get_id` (forward, `rank_b`) | 344ns/op | **291ns/op** |
| `get_term` (reverse, `select_b`) | 484ns/op | **137ns/op** |

The rebuild/space tradeoff is the clean, decisive one the design doc
frames block size around: the FST is **~1.74x more space-efficient**,
front coding is **~6.4x faster to rebuild** -- a real number on each side
of the "update locality vs. global compression" tension §12 describes,
instead of an assumption. `size_bytes()` on both types implements the
same accounting (every live block's compressed payload + `ids` array,
plus the router `directory` and reverse `id_to_block` directory -- §14's
explicit requirement not to report compressed-payload size alone), so
the comparison is apples-to-apples; it does not account for `Vec`
allocator slack.

The lookup numbers are the more surprising result: front coding isn't
just cheaper to rebuild, it's also **faster at both lookup directions**
at this block size (`get_id` ~1.2x, `get_term` ~3.5x) -- despite `get_id`
walking a genuinely minimized automaton byte-by-byte and `get_term`
using the same right-language-cardinality-guided `Fst::get_key` descent
praised above. The design doc's own framing (§2/§3) might suggest an
automaton should win on lookup speed too, since that's much of its
stated appeal ("simultaneously serve as... a rank structure; a
select/unranking structure"); it doesn't, here. The likely reason:
`Fst::get_key`'s per-node transition scan and `Map::get`'s per-byte
`find_input` lookup both carry real per-step constant-factor overhead
(pointer-chasing through the compact node encoding) that a restart-point
decode -- at most `RESTART_INTERVAL=16` cheap `String::truncate` +
`push_str` calls, no node/transition indirection at all -- simply
doesn't pay. This is exactly why §11 says measure before assuming a
fancier structure is faster, not just smaller: at this specific `(block
size, RESTART_INTERVAL)` regime, it measurably isn't.

This is one data point at one block-size regime, one fixed
`RESTART_INTERVAL`, and one (still synthetic, if namespace-clustered)
term distribution -- not yet the systematic `B` sweep §12 asks for
(which should also sweep `RESTART_INTERVAL`, now that it visibly matters
to more than just space), and not yet run against a real RDF corpus
(§8's other requirement). Both remain open, see below.

### Optimization pass -- Data-Oriented Design (`src/entries_arena.rs`, plus `block.rs`/`front_coded_block.rs`/`dictionary.rs`/`front_coded_dictionary.rs`)

A DoD-focused pass targeting four concrete things: `get_term`'s allocation,
the scratch/rebuild path's *temporary* allocation, total memory size, and
right-sizing integer field widths. Each change is measured against the
same probes used throughout this document, not assumed.

- **`entries_scratch: Vec<(Box<str>, TermId)>` was itself a "pointer-heavy"
  structure** -- exactly what §10 warns against ("avoid representations
  resembling `State { Vec<Transition>, Box<State> }`... prefer compact
  indices and packed arrays"), just one level removed from the FST/DAWG
  case that section is actually about. Every block rebuild's
  `decode_into` allocated one `Box<str>` *per term in the block* (up to
  `block_max`, so up to 256 individual small heap allocations for a
  single-term insert), even though every one of those terms except the
  one actually changing was unchanged content being needlessly recopied
  into a fresh allocation.

  Replaced with [`EntriesArena`] (`src/entries_arena.rs`): a struct-of-
  arrays scratch buffer -- one contiguous `text: Vec<u8>` holding every
  term's bytes concatenated, `offsets: Vec<u32>` (CSR-style) marking
  where each one starts/ends, and `ids: Vec<TermId>`. `decode_into` now
  just pushes bytes onto `text` (amortized `Vec` growth, no per-entry
  allocation); rebuilding calls `Block::build`/`FrontCodedBlock::build`
  with a *borrowing* iterator over `(&str, TermId)` pairs straight out of
  the arena, so no owned string is ever materialized just to feed a
  rebuild. `Dictionary`/`FrontCodedDictionary` each keep two arenas
  (`scratch_a`, `scratch_b`, still `mem::take`n at the start of a
  mutation and handed back before any nested scratch use, same discipline
  as before) rather than one, because e.g. an insert needs "the block's
  existing content" and "that content merged with the new term" alive at
  once; a plain `merge_into(old, new, out)` free function (arena + sorted
  slice -> arena) is shared by both the single-insert path (`new` is a
  one-element slice) and each `insert_batch` group (`new` is the group),
  which also means batch grouping's `new_entries` no longer allocates a
  `Box<str>` per genuinely-new term either -- it borrows straight from the
  caller's `&[&str]` input. `bootstrap` (empty-dictionary initial batch)
  needed no arena at all once its ids were collected into a plain
  `Vec<TermId>` first: it builds directly from `unique_terms[start..end]`
  `&str` slices zipped with those ids.

  Measured (`src/block.rs`, `perf_probe_block_rebuild_breakdown`, same
  160-entry/5,000-iteration probe as above): `decode_into` dropped from
  11.95µs/op to **7.25µs/op**, about **40% faster**, bringing it to
  ~1.3x the pure-allocation floor (was ~2.7x) -- most of what's left is
  genuine FST-walking, not allocation. `Block::build` itself was
  unaffected (59.59µs/op, statistically the same as 58.79µs/op before),
  exactly as expected: its cost was never from entry materialization, it
  never touched the old `Box<str>`s beyond reading their bytes. Net
  effect on end-to-end `Dictionary::insert` is real but modest
  (~66.8µs/op vs ~70.7µs/op decode+build) precisely because `Block::build`
  dominates and this pass doesn't touch it -- consistent with everything
  measured earlier in this document.

- **`get_term_into`**: `Dictionary`, `FrontCodedDictionary`, and
  `PlainDictionary` each gained an inherent `get_term_into(&self, id,
  buf: &mut Vec<u8>) -> bool` alongside the trait's `get_term`, decoding
  into a caller-reused buffer instead of allocating a fresh `String`
  every call (§10's reusable-scratch principle applied to *reads*, not
  just rebuilds -- a hot loop materializing many terms pays one
  allocation total instead of one per term). `Block`/`FrontCodedBlock`
  gained the matching `term_at_into(local_rank, buf: &mut Vec<u8>)`
  underneath (using `Fst::get_key_into` and the front-coded restart-decode
  directly into the buffer, respectively); `term_at` is now a thin
  allocating wrapper over it, not a separate implementation.

  Measured, same-process A/B (`tests/dictionary.rs`,
  `perf_probe_default_block_bounds`, 20,000 terms, comparing `get_term`
  and `get_term_into` back-to-back in one run to avoid the cross-session
  noise noted below): **627ns/op for `get_term` vs. 533ns/op for
  `get_term_into`, ~15% faster.** Modest, because the FST descent itself
  dominates the cost and only the one small buffer allocation is being
  removed -- but real and reproducible within a single process.

  A note on measurement hygiene this pass ran into directly: an *absolute*
  cross-session comparison (628ns now vs. a previously-recorded 384ns for
  the same code path) looked like a ~60% regression at first. An A/B
  reverting the code to the literal old implementation, in the same
  session, reproduced the *same* ~630ns -- proving the "regression" was
  session-to-session environmental noise (CPU frequency scaling, system
  load) on this machine, not a code change. Absolute latency numbers in
  this document should be read as relative/same-session comparisons,
  not compared across sessions at face value; the `get_term_into` A/B
  above is trustworthy specifically because both halves ran in one
  process.

- **`FrontCodedBlock::prefix_lens: Vec<u32> -> Vec<u16>`.** A shared-
  prefix length is bounded by a single term's own byte length, never by
  total block content -- no realistic RDF term is within orders of
  magnitude of 65535 bytes, so `u16` loses nothing. (`suffix_offsets`
  stays `u32`: it's a *cumulative* byte offset across the whole block, so
  it needs headroom for `block_max` terms' worth of bytes, not one term's,
  and that can realistically exceed 65535 as `block_max` grows in a
  future `B` sweep.) Also switched `rank`'s hot binary-search-then-decode
  loop to compare raw bytes (`&[u8]`) instead of validated `str`s --
  `str`'s `Ord` is already byte-wise lexicographic, so re-validating UTF-8
  on every comparison was pure overhead never buying anything.

- **`BlockId`/`FrontCodedBlockId`: `u32` -> `NonZeroU32`** (storing slot +
  1, `BlockId::slot()` subtracting it back out). This doesn't change
  `BlockId` itself, but it changes `Option<BlockId>` -- via Rust's
  niche-filling optimization, `Option<NonZeroU32>` is the same size as
  `NonZeroU32` (4 bytes), whereas `Option<u32>` needs a separate
  discriminant (8 bytes, rounded to alignment). `id_to_block:
  Vec<Option<BlockId>>` is sized to the *total term count*, not the block
  count, so this is the single highest-leverage space change in this
  pass -- confirmed by a unit test (`block_id_niche_optimization_holds`)
  asserting `size_of::<Option<BlockId>>() == size_of::<BlockId>()`.

  Measured (`perf_probe_space_and_rebuild_fst_vs_front_coded`, same
  9,000-term namespace-clustered set as the front-coded-baseline
  comparison above): FST space dropped from 13.84 to **9.84 bytes/term**;
  front-coded dropped from 24.07 to **18.07 bytes/term**. Both match
  arithmetic exactly: FST's drop is `4 bytes x 9000 terms = 36,000 bytes`
  (the `NonZeroU32` change alone, `size_bytes()`'s `id_to_block` term);
  front-coded's drop is that same 36,000 bytes *plus* `2 bytes x 9000 =
  18,000 bytes` from the `prefix_lens` shrink, for 54,000 total -- exactly
  what both `size_bytes()` implementations report. Lookup numbers moved
  too: front-coded `get_id` dropped from 291ns/op to 231ns/op (the byte-
  comparison change); insert/`get_term` numbers on this dataset are noted
  but, per the measurement-hygiene point above, not cross-session-
  comparable to the earlier table without a same-session A/B, which
  wasn't rerun for every metric here.

None of this touched the *architectural* decisions from earlier sections
(block rebuild triggers, split/merge thresholds, the `fst` crate as F_b's
base representation) -- it's entirely about the DoD shape of scratch/
temporary data and field widths, and the correctness net (the 3-way
differential fuzz suite, extended with a `block_id_niche_optimization_holds`-
style assertion per type) passed clean on the first full rewrite. The one
architectural piece it left alone, Variant A vs. B reverse mapping (§5),
is the very next thing this document takes on.

### Stage 5 -- Variant A vs. Variant B reverse mapping, resolved (H3)

§5 poses this as an explicit open question -- don't decide from
asymptotics, benchmark both -- and by the time the optimization pass
above landed, the balance of costs had visibly shifted: `select_b`
(`term_at`/`term_at_into`) went from being the dominant cost in a
`get_term` call to, for front-coded blocks, one of the *cheaper* parts.
That made Variant A's O(block size) `local_rank_of` scan (linear over
`ids`, finding which local rank a `TermId` maps to before `select_b` can
even run) worth measuring in isolation rather than continuing to assume
it was noise.

Measured (`perf_probe_local_rank_of` in both `src/block.rs` and
`src/front_coded_block.rs`, 160-entry block, 800,000 total calls each):

| | `local_rank_of` alone | `local_rank_of` + `term_at_into` | scan's share |
|---|---:|---:|---:|
| FST (`Block`) | 33ns/op | 531ns/op | 6.3% |
| Front-coded (`FrontCodedBlock`) | 29ns/op | 74ns/op | **39.7%** |

For front-coded specifically, the scan is not noise -- it's two-fifths of
the whole reverse-lookup cost. That settled it: implemented **Variant B**,
`TermId -> (BlockId, local_rank)` stored directly (`id_to_block:
Vec<Option<(BlockId, u16)>>`; `local_rank` is `u16` since it's bounded by
`block_max`, which `with_block_bounds` now asserts fits in `u16::MAX`),
eliminating the scan for O(1) reverse lookup.

The cost §5 warns about is real and now paid on every rebuild, not just
the changed entry: since a block rebuild reassigns every surviving
entry's sorted position, `insert`/`remove`/`split`/`merge_with_neighbor`/
each `insert_batch` group all now call a shared `assign_ranks` helper
that walks the *entire* rebuilt block's entries (already available from
the same arena/iterator used to build it) and rewrites their
`id_to_block` entries -- O(block size) array writes per rebuild, where
Variant A only ever touched the one changed id (or, for split/merge, the
ids that physically moved to a different `BlockId`). The hypothesis
behind choosing B anyway: this extra write cost is array writes on top
of a rebuild that already costs tens of microseconds (`Block::build`'s
FST minimization, per the "Optimization pass" section), so it should be
comparatively invisible.

Measured, both directions:

- **Reads got faster**, cleanly and consistently. Same-process A/B
  (`perf_probe_default_block_bounds`, 20,000 terms): `get_term` 627ns ->
  588ns/op, `get_term_into` 533ns -> 494ns/op -- both dropped by the
  *same* ~39ns, exactly the scan cost this removed, not a coincidence.
  On the 9,000-term namespace-clustered set, front-coded `get_term`
  dropped from 108ns to **73ns/op** (a third of it gone, matching the
  39.7%-of-cost measurement above); FST `get_term` dropped from 837ns to
  782ns.
- **Space regressed, by exactly the predicted amount.** `Option<(BlockId,
  u16)>` is 8 bytes (confirmed by
  `id_to_block_entry_niche_optimization_holds`), not the 4 bytes
  `Option<BlockId>` got from the `NonZeroU32` niche trick -- that saved
  space is now spent storing `local_rank` instead. FST space went back
  from 9.84 to **13.84 bytes/term** and front-coded from 18.07 back to
  **22.07 bytes/term** -- both exactly reverting the `4 bytes x 9000 =
  36,000 byte` saving the niche optimization produced. This is precisely
  the tradeoff §5 names ("the reverse directory is smaller" is listed as
  a Variant A advantage) -- now quantified rather than assumed, and it
  came due *because* of the niche-optimization pass immediately before
  it: without `NonZeroU32`, Variant A's `id_to_block` was already paying
  8 bytes via the `Option<u32>` discriminant, so Variant B's 8 bytes
  would have looked "free" by comparison. Measuring space before *and*
  after both changes, in the right order, is what surfaced that this
  specific tradeoff is real and not an artifact of the optimization pass.
- **Insert cost moved too**, though this leg wasn't rerun as a clean
  same-session A/B the way `get_term` was: cross-session, FST insert
  went from ~88.0 to ~91.9µs/op and front-coded from ~14.4 to ~16.9µs/op
  on the namespace-clustered set. Consistent in *direction* with paying
  for `assign_ranks`'s O(block size) writes, but per the measurement-
  hygiene note above, this document treats it as a plausible-not-confirmed
  effect size until it's rerun as a same-session A/B -- still tiny next
  to the tens-of-microseconds rebuild cost either way.

`Block::local_rank_of`/`FrontCodedBlock::local_rank_of` (Variant A) are
kept, `#[cfg(test)]`-only, purely as the benchmark baseline this
comparison used and that a future `B`-sweep re-evaluation could reuse --
not called from production code anymore.

### Hardware counters for `get_term` vs. `get_term_into` (`examples/perf_get_term.rs`)

Everything measured so far used `std::time::Instant` wall-clock deltas.
Design doc §14 explicitly asks for more: "CPU cycles; instructions;
L1/L2/LLC misses where available; branch misses" -- not just "is it
faster" but *why*. `perf stat` gives that, but it measures a whole
process, and the setup cost (20,000 inserts, tens of microseconds each)
would otherwise swamp a `get_term`-sized signal. `examples/
perf_get_term.rs` handles this with a `setup_only` mode -- do the
identical 20,000-term build, run no loop -- whose counts get subtracted
from the `get_term`/`get_term_into` runs' totals before dividing by
operation count, canceling the shared setup cost out of the per-op
estimate. Reproduce with:

```sh
cargo build --release --example perf_get_term
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    -- ./target/release/examples/perf_get_term setup_only 400
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    -- ./target/release/examples/perf_get_term get_term 400
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    -- ./target/release/examples/perf_get_term get_term_into 400
```

Per-op deltas (8,000,000 calls each, `setup_only` subtracted), on the
same 20,000-term flat dataset as `perf_probe_default_block_bounds`:

| metric | `get_term` | `get_term_into` | reduction |
|---|---:|---:|---:|
| cycles | 2475.6 | 2004.4 | 19.0% |
| instructions | 8243.7 | 7281.4 | 11.7% |
| cache references | 5.17 | 1.55 | **69.9%** |
| cache misses | 0.112 | 0.059 | 47.0% |
| branches | 925.4 | 706.7 | 23.6% |
| branch misses | 1.88 | 2.79 | -48.7% (worse) |

The cycle counts corroborate the wall-clock numbers directly (2475.6
cycles at this machine's ~3.85GHz implied clock is ~643ns, matching the
588-627ns/op range `Instant` measured for `get_term` in the same
region of this document). But the *cache-references* column is the
actual explanation, not just confirmation: it drops by 70%, far more
than any other metric. `get_term` allocates a fresh `Vec<u8>`/`String`
on every single call -- new heap memory, cold for the allocator's
bookkeeping and for the data itself; `get_term_into` reuses the same
buffer across all 8,000,000 calls, keeping it resident and hot. That's
precisely what "reusable scratch memory" (§10) is supposed to buy, now
shown at the hardware-counter level rather than asserted from
first principles.

Not every metric moved in `get_term_into`'s favor: branch misses per op
are *higher* (1.88 -> 2.79, worse in absolute count, and worse as a rate
too -- 0.20% -> 0.40% of branches mispredicted), even though total
branches taken dropped 23.6%. The net cycle count still clearly favors
`get_term_into`, so this doesn't change the conclusion, but it's a real,
honestly-reported exception, not everything about the allocation-avoiding
path is strictly better on every axis -- consistent with this document's
running theme that measuring beats assuming, including when a change is
already known to be a net win.

#### Does it hold at bulk scale? (`examples/perf_decode_bulk.rs`)

The 20,000-term measurement above is dominated by single-lookup latency.
A fair objection: nanosecond-level deltas can still be worth chasing if
the real workload decodes *millions* of terms in one pass (a dictionary
dump/export, bulk materialization) -- so it's worth checking whether the
per-op advantage holds, shrinks, or grows at that scale, not just
assuming it scales linearly. `examples/perf_decode_bulk.rs` runs the
same `setup_only`/`get_term`/`get_term_into` comparison at 2,000,000
terms in a single pass (no repeats needed at this scale), built via
`insert_batch`'s bootstrap path rather than sequential `insert` --
sequential insert's ~90us/op rebuild cost would make multi-million-term
setup itself take minutes and would mix rebuild-cost variance into what
should be a clean decode-only measurement.

Per-op deltas (2,000,000 calls each, `setup_only` subtracted):

| metric | `get_term` | `get_term_into` | delta |
|---|---:|---:|---:|
| cycles | 2563.3 | 2119.7 | 17.3% |
| instructions | 8708.5 | 7824.4 | 10.2% |
| cache references | 5.76 | **-1.34** | 123% |
| cache misses | 3.61 | **-0.55** | 115% |
| branches | 955.4 | 763.5 | 20.1% |
| branch misses | 2.81 | 2.77 | 1.3% |

Two things change at this scale, both favorably:

- **`get_term_into`'s cache cost drops into the noise.** At 20,000 terms
  it was 1.55 cache-references/op (already small); at 2,000,000 it's
  *negative* -- statistically indistinguishable from zero, sometimes
  measuring slightly below the `setup_only` baseline itself. This is the
  mechanism working exactly as expected, taken to its logical conclusion:
  the one reused buffer amortizes its (already tiny) warm-up cost over
  far more calls, so per-op it essentially vanishes. `get_term`'s
  cache-reference cost, by contrast, doesn't amortize away -- it's paying
  for a genuinely fresh allocation on every single call regardless of how
  many calls there are, so it stays roughly constant (5.76 vs. 5.17
  before, a small increase plausibly from a larger heap at this scale).
- **The 20,000-term branch-misses anomaly (`get_term_into` measuring
  *worse*, -48.7%) mostly disappears at scale** (1.3% at 2,000,000
  terms). That's a useful data point on its own: the earlier result
  wasn't wrong, but a smaller sample was more exposed to noise/warm-up
  effects than the headline cycle-count numbers suggested. Bulk-scale
  measurement isn't just "the same conclusion with bigger numbers" --
  it can also correct which parts of a smaller measurement were solid
  versus noisy.

Translated to wall clock: `get_term` decoded 2,000,000 terms in 646.2ns/op,
`get_term_into` in 524.8ns/op -- **242.8ms saved over one 2-million-term
pass**, which is the concrete, scale-dependent answer to "does a
sub-100ns/op difference matter": at dump/export scale, yes, by hundreds
of milliseconds per pass, and that gap should widen (not shrink) as the
dataset grows further, per the cache-amortization argument above.

### Binary-search unranking: acting on the `perf record` finding

The measurements above characterize `get_term`/`get_term_into`'s cost;
they don't explain *where inside* `term_at_into` it goes. `perf stat`
can't answer that -- it only aggregates whole-process counters. `perf
record -g` (call-graph sampling) can, and reveals a concrete, fixable
finding rather than "the FST is just slow."

Isolating the recording from setup noise mattered here: at the scale of
a single decode pass, one-time bootstrap setup and the decode loop take
comparable wall time, so naively recording the whole process mixes
setup-time functions (`fst::raw::registry::Registry::new`, part of
`Block::build`'s minimization, not decoding) into what's supposed to be
a decode-only profile. `examples/perf_decode_bulk.rs` gained a `repeats`
argument so the loop can be made to dominate total runtime (e.g.
100,000 terms x 20 repeats -- one bootstrap, 2,000,000 decodes) rather
than trying to separate the two after the fact.

With a clean recording, 83.32% of self-time landed in `term_at_into`
(then implemented by calling `raw::Fst::get_key_into` directly), broken
down as:

| Sub-cost | Share of total runtime | What it is |
|---|---:|---|
| `Range<usize>::next()` | **41.5%** | Just stepping through candidate transitions |
| `transition`/`input` decode | ~24.2% | Actually decoding each transition's label/output/target |
| `unpack_delta`/`unpack_uint` | 2.4% | Varint-decoding the target address |
| `push` | 2.8% | Writing the decoded byte |
| loop overhead | 3.3% | `get_key_into`'s outer `while` |

`get_key_into`'s algorithm is `node.transitions().take_while(|t| t.out.value() <= value).last()` -- a **linear scan** through a node's transitions (up to its full branching factor), going through generic `Iterator`/`TakeWhile`/`fold`/`try_fold` machinery for every candidate examined. The single largest cost, 41.5% of total runtime, was that generic-iterator overhead itself -- not decoding transition data, just advancing through candidates one at a time.

The fix: `fst::raw::Node` also exposes `transition(&self, i: usize) ->
Transition` and `len(&self) -> usize` -- direct indexed access. And the
existing linear scan's own correctness already *requires* a node's
transitions to be sorted by non-decreasing accumulated output value
(that's the only way "take the last one `<= value`" is a well-defined
greedy-descent step at all) -- so a manual binary search over
`node.transition(i)` by index is valid using a precondition the
algorithm already relied on, not a new assumption. `src/block.rs` now
has its own `last_transition_at_most` (a `partition_point`-style binary
search) and `term_at_into` walks the FST directly via `Node::root`/
`Node::transition`/`Node::node`, bypassing `get_key_into` entirely.

Correctness risk here is real -- a subtly wrong binary-search comparison
would silently corrupt every reverse lookup -- so this got its own
targeted stress test (`term_at_and_rank_round_trip_with_wide_branching`,
3,000 entries with a digit suffix producing up to 10-way branching per
node, covering low/high/middle target indices within a node, not just
the always-one-choice case the other fixed-width test data happens to
produce) in addition to the full 3-way differential fuzz suite, which
already exercises `get_term` under randomized insert/remove sequences.
All passed clean.

Measured, before vs. after, at the same 2,000,000-term bulk-decode scale:

| metric (per op) | linear scan | binary search | change |
|---|---:|---:|---:|
| `get_term` cycles | 2563.3 | 2172.6 | -15.2% |
| `get_term` instructions | 8708.5 | 7483.1 | -14.1% |
| `get_term_into` cycles | 2119.7 | 1668.6 | -21.3% |
| `get_term_into` instructions | 7824.4 | 6594.7 | -15.7% |
| `get_term` wall clock | 646.1ns/op | 563.2ns/op | **-12.8%** |
| `get_term_into` wall clock | 524.8ns/op | 454.7ns/op | **-13.4%** |

Over the full 2,000,000-term pass: 165.8ms saved for `get_term`, 140.3ms
for `get_term_into` -- on top of the earlier `get_term_into`-vs-`get_term`
saving, not instead of it.

Not a clean win on every axis, reported honestly rather than only citing
the metrics that favor the change: branches per op went *up* (`get_term`
955.4 -> 1062.4, `get_term_into` 763.5 -> 876.0) and branch-misses got
meaningfully worse (`get_term` -62.0%, `get_term_into` -22.8% worse, in
both absolute count and as a fraction of branches taken), and
cache-references ticked up slightly too (binary search jumps to
arbitrary midpoints in a node's transition table; the old linear scan
walked it sequentially, which is more prefetcher-friendly even though it
does more total work). Binary search is less predictable per-branch than
a straight-line scan, and it doesn't have the sequential-access property
the old approach had either -- both real, structural costs of the
algorithm change. They just don't outweigh the reduction in *total* work
(far fewer transitions actually visited), so cycles, instructions, and
wall clock all still improved substantially. A follow-up `perf record`
after the change confirms the mechanism directly: the `next<usize>`
iterator-machinery hot spot is gone entirely from the profile: 71.91%
of self-time is still in `term_at_into`, but now split across
`transition`/`common_input` decode (~40%, legitimate work), the binary
search's own loop (~18%), byte-writing, and target-node construction --
i.e., what's left is mostly necessary work, not abstraction overhead.
That's a reasonable place to stop this specific optimization; further
gains would mean touching the FST's own byte-level decode routines
(`common_input`, `unpack_delta`) rather than how we call into them, a
larger step than this one.

#### Correction: the win above doesn't generalize to real RDF terms

The section above is left intact rather than quietly edited, because it's
a real measurement on the dataset it names -- but that dataset
(`http://example.org/term-{i:08}`, an 8-digit sequential numeric suffix)
turned out to be an unrepresentative stress test, and a follow-up
investigation using a *real* corpus found something materially
different. Credit: a parallel investigation built
`s-bench/src/bin/dict_profile.rs`, `s-dict`'s own profiling-harness
pattern (`perf_get_term.rs`/`perf_decode_bulk.rs`) pointed at the actual
W3C RDF test suite terms instead of synthetic keys, plus `perf
annotate` on `term_at_into`'s disassembly -- a level of detail this
document hadn't gone to.

The synthetic digit suffix forces up to 10-way branching at every one of
8 byte positions specifically *because* sequential integers spread
evenly across all 10 digits at each position -- that's what made the
linear-scan-vs-binary-search comparison count meaningful enough to
matter in the first place. Real IRIs don't look like that: long shared
literal path segments (namespace prefixes, common local-name structure)
dominate, and most nodes along a real term's decode path have low
branching factor -- often exactly one viable transition, sometimes a
handful, rarely anywhere near 10. On the real corpus (832 real terms,
5,000 repeats), `get_term`/`get_term_into`'s cycle and wall-clock numbers
came back statistically flat between the linear-scan and binary-search
versions: **there was rarely more than one or two candidates to search
among at a node in the first place**, so changing an O(k) scan to an
O(log k) search saves close to nothing when k itself is already small.

`perf annotate` on `term_at_into` confirms *why* directly, at the
instruction level: the dominant cost inside the function is not the
comparison/branch logic that decides which transition index to fetch at
all -- it's `node.transition(idx)` itself, specifically the shifts,
ANDs, and conditional moves needed to unpack *one* compactly bit-packed
transition record out of the FST's succinct encoding. That cost is paid
once per transition *decoded*, regardless of whether the index to decode
was found by a linear scan or a binary search -- the search algorithm
only changes how many candidates get examined before finding the right
one, not the cost of examining each one, and on real data that candidate
count was never large enough for the difference to matter.

The binary search change itself isn't wrong to keep -- O(log k) is never
worse than O(k), and on pathological high-branching-factor data (however
rare in practice) it remains a real backstop -- but it should not be
cited as *the* fix for real-world `get_term` cost, because on real data
it isn't. This is exactly the kind of thing §11's discipline exists to
catch, and this correction is itself evidence the discipline works: a
synthetic dataset chosen to be easy to reason about (sequential integers)
quietly encoded an assumption -- uniform, high branching factor -- that
doesn't hold for the actual target domain, and only a real-corpus
measurement caught it.

Reconfirmed on the real corpus, generalizing cleanly from the earlier
synthetic result: the `get_term` vs. `get_term_into` allocation-avoidance
gap is real and holds. Per-op deltas (`setup_only` subtracted):

| metric | `get_term` | `get_term_into` | ratio |
|---|---:|---:|---:|
| cycles | 2234 | 1795 | 1.24x |
| instructions | 7302 | 6472 | 1.13x |
| cache-references | 1.04 | 0.42 | **2.48x** |
| branches | 1090 | 907 | 1.20x |

The cache-references gap (2.48x) stands out from the others (~1.2x) --
the same signature as the earlier 20,000-term/2,000,000-term
measurements: `get_term`'s fresh per-call allocation touching cold heap
memory, `get_term_into`'s reused buffer staying resident. `perf report`
self-time confirms it's isolated cleanly: `get_term` spends 61.4% of
self-time in `term_at_into`, 19.9% in `fst::raw::node::Node::new` (node
construction/decode while walking), and ~13.5% spread across
malloc/realloc/free internals; `get_term_into` spends 73.6% in
`term_at_into`, 25.1% in `Node::new`, and allocator activity drops to
near zero -- the ~13.5% in `get_term` is entirely the avoidable
allocation, nothing else moved.

`Node::new` showing up as a distinct, substantial cost (~20-25% of
self-time, separate from `term_at_into`'s own transition-decode work) is
the new finding this investigation surfaced that the earlier one didn't:
every `get_term_into` call walks from the FST root and re-decodes every
node along the path, including nodes shared with the *previous* call if
the two requested ids happen to be lexically close (namespace-clustered
real data, §8, makes this likely for anything resembling bulk access).
Point lookups can't avoid this -- each call is independent by design --
but a caller resolving *many* ids at once could, by reusing the
existing streaming decode (`decode_into`/`EntriesArena`, already O(block
size) with zero per-entry allocation after the arena work above) instead
of re-walking from the root once per id.

Priorities out of this, in order:

1. **Already correct, keep using it as the default hot path:**
   `get_term_into`. The allocator overhead it avoids is real, confirmed
   again on real data, and there's already an API for it.
2. **Worth building:** a bulk/multi-id decode entry point using the
   existing block-level streaming machinery, for callers resolving many
   ids at once -- exactly a "dump the dictionary" workload. Implemented
   below as `get_terms_into`.
3. **Deprioritized based on this evidence:** further tuning the
   transition-search algorithm. The bottleneck is per-transition *decode*
   cost inside `fst`, not comparison count -- real, low-branching-factor
   data doesn't have much comparison count left to cut.
4. **Speculative, needs its own investigation if 1-2 aren't enough:**
   whether `fst`'s own encoding parameters (transition/output byte
   widths) could be tuned for cheaper per-transition decode. That's
   inside a third-party dependency's internals, a materially bigger step
   than anything else in this section.

### Bulk decode: `get_terms_into`

Acting on priority 2 above. `Dictionary::get_terms_into`/
`FrontCodedDictionary::get_terms_into`/`PlainDictionary::get_terms_into`
decode many `TermId`s in one call: group the requested ids by which
block owns them (`id_to_block` is already O(1) per id via Variant B),
then for each affected block, decode it *once* via the existing
`decode_into`/`EntriesArena` machinery and pull every requested term
straight out of the resulting arena (`EntriesArena::term_at`, a cheap
slice, no further FST walking) instead of one independent
root-to-leaf `get_term_into` walk per id.

This trades "decode only what's asked for" against "decode a whole
block once, however many of its entries are actually wanted." For bulk
materialization -- dumping most or all of a dictionary, or any workload
where requested ids cluster within blocks (namespace-clustered data
makes this the common case, §8) -- that's a clear win: shared-prefix
node decode that `get_term_into` would re-pay per call happens once per
block instead. For a handful of ids scattered across many different
blocks, it's a net loss (an entire block gets decoded to serve one id
from it) -- `get_term_into` remains the right choice there. The
docstring says so; this crate doesn't try to guess which case a caller
is in and switch automatically, matching §11's discipline of not
building adaptive machinery ahead of a measured need.

`PlainDictionary::get_terms_into` exists for API parity across all three
types (matching `get_term_into`'s precedent) even though the shared-
decode rationale doesn't apply to it -- there's no FST walk to amortize,
just a direct array lookup per id.

Measured, both the win and the honestly-expected loss (`examples/
perf_decode_bulk.rs`'s `get_terms_into`/`get_terms_into_scattered` modes,
2,000,000-term `Dictionary`, default 64..=256 block bounds, wall clock
via `time`, `setup_only` subtracted, averaged over 2 runs each):

| Scenario | `get_term_into` (repeated) | `get_terms_into` (bulk) | change |
|---|---:|---:|---:|
| Dump everything (2,000,000 ids, all blocks) | 850ms (425ns/op) | **222ms (111ns/op)** | **3.8x faster** |
| Scattered (6,667 ids, one per block, stride 300 > block_max) | ~3ms (too small to separate from noise at this scale) | 55ms (8.25us/op) | **~19x slower** |

Both land exactly where the design predicts. "Dump everything" is
`get_terms_into`'s target case: requested ids span nearly every block,
so each block's `decode_into` (already O(block size) with zero
per-entry allocation, from the arena work earlier) is paid once and
amortized across ~256 ids instead of once per id via `get_term_into`'s
independent root-to-leaf walks -- a **3.8x** wall-clock win, well past
what the per-call `Node::new`-amortization argument alone would suggest,
because it also collapses the router lookup and `id_to_block` access
that `get_term_into` still pays per call down to one grouping pass. The
scattered case is the flip side working as documented: with one
requested id per block (stride chosen specifically to guarantee this),
`get_terms_into` decodes an entire block to serve a single id from it,
6,667 times -- **~19x slower** than just calling `get_term_into` 6,667
times. This isn't a bug or a case to special-case away; it's `get_terms_into`'s
documented cost model working exactly as described, and it's why this
crate doesn't try to auto-detect which case a caller is in -- the two
measurements above are the actual data a caller (or a future adaptive
heuristic, if one ever proves worth building) would need.

## What's not implemented yet

In the design doc's own stage numbering (§17), this crate currently
implements Stage 0, a slightly pulled-forward Stage 1, and Stage 3 (ahead
of Stage 2, since a router was already needed to make Stage 1 usable at
all -- see the Stage 1 section above). Left for later, deliberately, per
§11's optimization discipline and §16's anti-goals -- do not build more
compression machinery before measuring whether it's needed:

- **Stage 2 proper**: a non-trivial router (Patricia/radix tree, or a
  compact B/B+-tree over separators) if the sorted-array router turns out
  to be a bottleneck at large block counts. Not assumed necessary yet
  (§4: "do not assume which router is best before measurement").
- **A custom automaton**, if a realistic workload ever shows the `fst`
  crate's full-rebuild-per-mutation cost (see Stage 3 above -- now a
  measured, explained ~59µs/op minimization cost, not a mystery) is a
  bottleneck a custom incremental-construction automaton (Daciuk et al.)
  could avoid by reusing untouched subtrees across rebuilds instead of
  rebuilding a block from scratch every time. Native rank/select (§2) is
  *not* a reason to do this anymore -- the `fst` crate already provides
  both, see Stage 3 above.
- **Stage 4**: systematic measurement of block size `B` across
  `{32, 64, 128, 256, 512, 1024, 2048, 4096, ...}` (§12), to test **H1**
  (does micro-static compression at moderate `B` approach global-FST
  compression, thanks to RDF namespace clustering?).
- **Stage 6**: physical FST layout optimization from profiling (contiguous
  state/transition arrays, separated hot/cold block metadata -- §10),
  to test **H4**. (Stage 5, Variant A vs. B reverse-mapping, is now done
  -- see "Stage 5" above; H3 resolved in Variant B's favor at this block
  size, with the tradeoff quantified, not just the direction.)
- **Stage 7 and beyond**: RDF-aware term decomposition (§9: term-kind
  prefixes, structural typed-literal encoding, language-tag dictionaries,
  blank-node encodings), FSST as an alternative block codec, compressed
  external-id permutations, SIMD-friendly layouts, the remaining §13
  baselines (front-coded blocks is now done, see above; still missing: a
  global static minimal FST, `m-Bonsai`/Dynamic-Path-Decomposed-Trie-style
  dynamic tries, global FST + compressed permutation), and the remaining
  §14 measurement suite (latency percentiles beyond median, allocation and
  peak-memory tracking during rebuilds -- bits/term and cache/branch
  misses now have real data points, see above, but not the systematic `B`
  sweep or a real RDF corpus).

None of this is assumed necessary; each is an explicit hypothesis (§15)
to test once there's a working Stage 0/1 baseline to measure against and
real RDF corpora to measure on, not something to build speculatively
ahead of that measurement.

## Interop

`TermId` is `u32`-width specifically so it can flow directly into
`s-graph`, whose triples are `(u32, u32, u32)` subject/predicate/object
ids (see `../s-graph/README.md`, "Getting real data in").

## Relevant literature

Condensed from the design conversation this crate was built from; see
that conversation for the full annotated list.

- Daciuk, Mihov, Watson, Watson. *Incremental Construction of Minimal
  Acyclic Finite-State Automata*. CL 26(1), 2000. Lexicographically
  ordered incremental construction -- the `fst` crate's `MapBuilder`
  follows this shape directly (it requires strictly increasing sorted
  keys), which is why every block rebuild already receives its terms
  pre-sorted from `decode_all`/the batch grouping pass rather than sorting
  again inside `Block::build`.
- Carrasco, Forcada. *Incremental Construction and Maintenance of Minimal
  Finite-State Automata*. CL 28(2), 2002. Addition/deletion while
  preserving minimality -- background for why this design deliberately
  avoids needing that (rebuild the block instead), and for what a future
  in-place-mutable per-block automaton would need if the full-rebuild cost
  (Stage 3 above) turns out to matter.
- Cognetta, Allauzen, Riley. *On the Compression of Lexicon Transducers*.
  FSMNLP 2019. Compact static FST encodings and right-language-count-based
  rank/select -- describes the general technique that turned out to
  already be present, in the form needed here, in the `fst` crate's
  `raw::Fst::get_key` (see Stage 3 above); the reference point if a more
  compact custom encoding is ever built to replace it.
- Brisaboa, Cerdeira-Pena, de Bernardo, Navarro. *Improved Compressed
  String Dictionaries*. 2019. Hierarchical front coding over URI/RDF
  literal collections -- the required Stage-7-era baseline (§13).
- Boncz, Neumann, Leis. *FSST: Fast Random Access String Compression*.
  PVLDB 13(11), 2020. Alternative block codec to compare against a
  micro-FST once one exists.
- Bender, Farach-Colton, Kuszmaul. *Cache-Oblivious String B-Trees*. PODS
  2006. Conceptual comparison for combining compressed variable-length
  strings, locality, and mutation.
- Grossi, Ottaviano. *Fast Compressed Tries through Path Decompositions*.
  Cache-conscious static trie comparison point.
- Kanda, Köppl, Tabei, Morita, Fuketa. *Dynamic Path-Decomposed Tries*.
  JEA 25, 2020. Genuinely dynamic compressed-dictionary baseline for §13.
- Poyias, Puglisi, Raman. *m-Bonsai: a Practical Compact Dynamic Trie*.
  Another §13 dynamic-trie baseline.
- Barbay, Navarro. *On Compressing Permutations and Adaptive Sorting*.
  Relevant if/when a global compressed lexical-rank-to-id permutation is
  evaluated as a §13 baseline.
- Acton. *Data-Oriented Design and C++*. CppCon 2014. Fabian. *Data-
  Oriented Design*. 2018. The methodology behind the slab/stable-handle
  choices in `Dictionary` and the cached `min_key` field on `Block`
  (§10): compact arrays and stable integer handles, hot routing metadata
  kept separate from the compressed cold FST payload.


### Prepared ingestion batches

`TermBatch` owns one caller-bounded text arena and reuses flat arrays of sorted
input indices and duplicate-group boundaries. Parser workers can prepare it
without accessing the dictionary. `Dictionary::insert_prepared` scatters IDs
back to original input positions; the existing `insert_batch` borrows caller
strings and uses the same insertion engine. Neither retains a dataset-wide
uncompressed term index.

The writer routes once per sorted destination run. For a run with at least
`max(8, block_len / 4)` distinct incoming terms, it decodes the old block once
and merges in lexical order, identifying duplicates and new IDs in that pass.
Smaller runs use point lookups, avoiding whole-block decoding for sparse
repeated terms. Rebuilt output still goes directly to final-sized blocks.
Scratch arenas are taken once per batch; empty replacements allocate nothing.
They store valid UTF-8 in a String with end offsets, so reading a term needs
no repeated UTF-8 scan. Bootstrap also reuses the temporary block-ID vector.

FST construction uses a two-way registry with `clamp(4 * entries, 64, 10000)`
rows instead of allocating the upstream fixed 10000 rows for every tiny block.
The vendored patch exposes only the existing row-count parameter; upstream
constructors keep their default. Evictions can reduce state deduplication and
increase encoded size but cannot alter key/value semantics. Constructor
parity tests cover disabled, tiny, adaptive-sized, and original caches.

### Bounded-batch sweep: does batching reduce rebuild frequency?

A `perf`-sampled profile outside this crate (during ingestion) attributed
roughly 40% of CPU cycles to block rebuilding and roughly 21% to batch-sort
preparation (already overlapping insertion). Before deciding whether the
next lever should be "rebuild less often" or "make each rebuild cheaper",
`examples/batch_sweep.rs` measures the first question directly: does a
larger caller-chosen flush threshold on `TermBatch` actually reduce how
often blocks get rebuilt, and by how much, against real cost (time,
retained allocation, RSS)?

Methodology: preload a fresh `Dictionary` (default block bounds) with
200,000 terms via one prepared batch (unmeasured setup), then stream
200,000 more terms -- disjoint from the preload by construction, cycling
through 7 differently-shaped synthetic namespaces (IRIs, a long literal
shape, blank nodes) rather than one sequential numeric suffix, since the
"Correction" section above found that shape manufactures artificial
branching factor; this sweep is much less sensitive to that than the
earlier binary-search measurement was, since it's measuring rebuild/
allocation bookkeeping, not FST transition-decode cost. Terms are pushed
into a `TermBatch` and flushed (`prepare` + `insert_prepared`) once a
configured cap is hit. `Dictionary::rebuild_stats()` (new: `blocks_rebuilt`,
`entries_decoded`, cumulative counters incremented at every `Block::build`/
`decode_into` call site) is snapshotted before and after the streaming
phase and diffed, so the one-time preload never pollutes the measurement.
Retained allocation is tracked with a counting `#[global_allocator]`
wrapping `System` (current live bytes, not peak -- peak is reported
separately, since a batch's transient decode/merge/build working set can
spike well above what it leaves behind); RSS comes from `/proc/self/status`
and is reported as a coarser, laggier cross-check (glibc doesn't always
return freed pages to the OS promptly).

Results (`cargo run --release --example batch_sweep`; `dec/new` and
`rblt/new` are entries-decoded and blocks-rebuilt per newly inserted term):

| batch cap | flushes | time | dec/new | rblt/new | retained | phase peak |
|---|---|---|---|---|---|---|
| terms<=1 | 200,000 | 42.5s | 191.60 | 1.0076 | 4.2MiB | 62.9MiB |
| terms<=16 | 12,500 | 17.5s | 83.79 | 0.4453 | 4.2MiB | 62.9MiB |
| terms<=64 | 3,125 | 4.5s | 20.83 | 0.1175 | 4.2MiB | 63.0MiB |
| terms<=256 | 782 | 1.35s | 5.54 | 0.0349 | 4.1MiB | 62.9MiB |
| terms<=1024 | 196 | 434ms | 1.19 | 0.0143 | 4.1MiB | 62.8MiB |
| terms<=4096 | 49 | 282ms | 0.46 | 0.0072 | 3.7MiB | 62.4MiB |
| terms<=16384 | 13 | 230ms | 0.19 | 0.0051 | 4.0MiB | 62.8MiB |
| terms<=65536 | 4 | 222ms | 0.11 | 0.0045 | 4.1MiB | 63.1MiB |
| bytes<=1KiB | 8,164 | 11.5s | 54.78 | 0.2937 | 4.2MiB | 62.9MiB |
| bytes<=64KiB | 133 | 410ms | 1.10 | 0.0103 | 3.6MiB | 62.4MiB |
| bytes<=1MiB | 9 | 223ms | 0.15 | 0.0048 | 4.0MiB | 62.8MiB |

(full output, including the byte-cap and combined sweeps, is reproducible
via the command above; not all rows are reproduced here).

**Batching reduces rebuild frequency dramatically, confirming H5, and the
effect is the dominant lever, not a marginal one.** Going from
`terms<=1` (equivalent to sequential `insert`: every new term is its own
flush) to `terms<=1024` cuts blocks rebuilt per new term by ~70x (1.008 ->
0.014) and entries decoded per new term by ~161x (191.6 -> 1.19), and wall
time by ~97x (42.5s -> 434ms) for the same 200,000 inserted terms into the
same starting dictionary. This is exactly what grouping same-destination
terms into one decode-merge-rebuild cycle (`insert_ordered`'s per-run
grouping, already in `src/dictionary.rs`) predicts: at `terms<=1`, nearly
every flush is a lone term hitting the sparse single-point-lookup path,
paying a full ~192-entry (the default block's steady-state size) decode +
rebuild for one new term; at `terms<=1024`, many new terms destined for
the same block get grouped into a single such cycle.

**Diminishing returns set in around a few times `block_max`** (default
256): `terms<=256` -> `terms<=1024` still roughly halves rebuild cost
(0.035 -> 0.014 rblt/new) and time (1.35s -> 434ms), but `terms<=4096` ->
`terms<=65536` (16x more) only takes rblt/new from 0.0072 to 0.0045 and
time from 282ms to 222ms -- most of a batch beyond a few thousand terms
is spread across many different destination blocks anyway, so raising the
cap further mostly just amortizes `prepare()`'s one sort per flush (item 4
of the report this sweep answers) rather than finding more same-block
grouping opportunity.

**Retained allocation and phase-peak allocation are flat across the whole
sweep** (~4.0-4.7MiB retained, ~62-64MiB phase peak, regardless of batch
cap) -- batching wins on time and rebuild count with no retained-memory
tradeoff at this scale; the final dictionary ends up the same size either
way, since `rebuild_batch_block`/`bootstrap`'s balanced chunking targets
the same final block sizes regardless of how the input was grouped to get
there. This is a real, reportable negative result specifically against
**item 3** of the report this sweep responds to (a measured 5.4% retained-
allocation increase) -- that increase is not explained by batch sizing
itself; it's a separate axis (almost certainly the FST registry-row-count/
compression tradeoff, which this sweep does not touch) and needs its own
measurement, not inference from this one. Byte-capped flushing (`bytes<=`)
tracks the term-capped results closely at comparable effective batch sizes
here, because this synthetic corpus's term-size variance (43 bytes/term
average, ~15-140 byte range) isn't extreme; a real corpus with a heavier
tail of long literals mixed with short IRIs would likely separate the two
cap strategies more, since a byte cap bounds peak batch memory that a
term-count cap alone wouldn't.

**Conclusion for prioritization**: the next implementation should
primarily target *rebuild frequency* (i.e., default to a substantially
larger `TermBatch` flush threshold than 1, on the order of a few times
`block_max`, in any ingestion pipeline built on this crate) rather than
*cheapening each rebuild* (report item 2, buffer reuse) -- at this
dictionary size and term shape, batching alone already captures ~97% of
the achievable wall-time reduction between the two extremes measured, and
the remaining ~3% (`terms<=1024` down to `terms<=65536`) is a much smaller
prize than the ~70x already on the table just by not flushing per-term.
Buffer reuse (item 2) and the registry/compression tradeoff (item 3)
remain worth pursuing but are now secondary, not primary, levers; item 4
(batch-sort cost) becomes the more visible remaining cost specifically
once a large flush threshold is adopted, matching the report's own note
that its priority rises "as the writer gets faster."

### Single-walk merge: removing the dense/sparse split

The dense/sparse branch inside `insert_ordered` (the bounded-batch sweep
above) turned out to have a real inefficiency, independent of batch
sizing: the sparse path did per-term `rank()` point lookups first, and
*if any of them missed*, fell through to a completely separate
`decode_into` walk of the whole block plus a second merge pass over the
result -- paying for both the lookups and the full decode whenever a run
turned out not to be all-duplicates. The dense path had a related, milder
version of the same problem: `decode_into` wrote every surviving entry
into a scratch arena, then a second loop read that arena back out while
merging -- every surviving entry written twice.

`Block::merge_into` (`block.rs`) replaces both: one method, one streaming
walk of the FST's own `Streamer`, merged directly against the (already
sorted, from `BatchOrder`) incoming run, writing the union straight into
the rebuilt block's content and resolving each incoming term's id (hit or
freshly allocated) as a side effect of the same walk. `insert_ordered` and
`TermDictionary::insert` (the single-term path) both now call this
unconditionally -- there is no dense/sparse heuristic left to tune or get
wrong. The cost: every touched block is now always fully walked, even
when nothing in the run turns out to be new, which the old sparse path
could sometimes avoid entirely. This crate's own workloads always
route real, new-term-containing runs to touched blocks in practice (see
above), so that tradeoff was accepted deliberately, not by default.

Wiring this into `Dictionary` needed one non-obvious borrow-checker
pattern: `insert_ordered` needs a `&Block` (to call `merge_into`) live at
the same time as `&mut self.free_term_ids`/`&mut self.next_term_id` (for
the id-allocation closure `merge_into` calls back into) -- which the
existing `self.block(id)` *method* can't give, since a method call borrows
`self` as a whole, opaque to the borrow checker even though it only reads
`self.blocks` internally. The fix is to index `self.blocks` directly at
the call site instead of going through that method; Rust's borrow checker
does support disjoint borrows of named fields accessed directly, just not
through an intervening method call. Same technique, same reason, applied
to the single-term `insert` path.

**Measurement caught a real methodology trap here, not just a result.**
`examples/batch_sweep.rs`'s wall-clock numbers showed this refactor as a
*regression* -- `terms<=1` went from ~34.8s (forced-`dense=true`, the
prior turn's experiment) to ~40.4s, reproducibly, across repeated runs in
the same session. Read at face value, that would mean single-walk is
worse despite doing strictly less work. An isolated microbenchmark
(`block::tests::perf_probe_merge_into_vs_decode_then_merge`, one new term
into a 160-entry block, 20,000 iterations, no `Dictionary`-level machinery
in the way) showed the opposite: `merge_into` at 13.8us/op vs. the old
two-step's 15.9us/op, ~13% *faster*, matching the algorithmic prediction
(one array write per surviving entry instead of two). Wall-clock and a
controlled microbenchmark disagreeing this sharply meant one of them was
wrong, so this got settled with `perf stat` hardware counters instead of
more wall-clock runs, per the user's steer -- a same-session, back-to-back
`git stash`-based A/B (`examples/perf_insert_single_flush.rs`, 50,000
preload + 50,000 single-term-flush inserts, `setup_only` subtracted) on
the *exact* pre-refactor vs. post-refactor code:

| metric (per insert, setup subtracted) | original dense/sparse | single-walk | delta |
|---|---|---|---|
| cycles | 693,908 | 687,914 | -0.9% |
| instructions | 1,645,215 | 1,623,677 | -1.3% |
| cache-misses | 250.4 | 221.1 | -11.7% |
| wall-clock (this run) | 184.5us | 222.4us | **+20.5%** |

Every CPU-work counter (cycles, instructions, cache-misses) says
single-walk is equal-or-better, matching the microbenchmark. Wall-clock,
measured in the same process, says the opposite. The only consistent
explanation is that the two runs executed at different effective clock
frequencies (DVFS/thermal state varying between consecutive `perf stat`
invocations) -- exactly the "cross-run wall-clock noise" hazard this
document's earlier measurement-hygiene notes already named, now caught
concretely rather than just asserted. **Conclusion: the single-walk
refactor is a real, small, unambiguous improvement by every hardware
counter; the wall-clock numbers from `batch_sweep` runs separated by other
work in between (as in the bounded-batch sweep above) should be treated
as directionally suggestive only, not trusted for a delta this size.**
`perf stat` cycles/instructions, not wall-clock, is now the standard for
any A/B this close going forward; wall-clock stays fine for the
order-of-magnitude effects the bounded-batch sweep found (70x is not
going to be a frequency-scaling artifact).

### Skipping unchanged reverse-map entries on rebuild

`assign_ranks` (`dictionary.rs`) rewrites `id_to_block` for every entry in
a rebuilt block, unconditionally -- Variant B's known cost (see "Stage 5"
above): a rebuild changes every surviving entry's local rank, not just the
one that triggered it, so the whole reverse mapping for that block needs
refreshing. Two things narrow that unconditionally:

- `Block::merge_into` (see "Single-walk merge" above) already knows, as a
  side effect of the streaming walk, the position of the first entry that
  differs from the block's pre-merge content -- everything before it kept
  its exact old position. It now returns this as `unchanged_prefix`
  (`MergeOutcome`), and `assign_ranks` takes a `skip` count so that prefix
  is never rewritten. Wired through `insert_ordered` (via
  `rebuild_batch_block`, which only applies the skip to chunk 0 -- any
  later chunk gets a fresh `BlockId` and renumbers from 0 regardless, so
  no old position carries over) and the single-term `insert` path.
  `remove` gets the equivalent for free: the removed entry's own
  `local_rank` *is* the skip count, no merge walk needed to find it.
- `split` and `merge_with_neighbor` don't go through `merge_into` at all,
  but the same fact applies more starkly: `split`'s left half keeps the
  original `BlockId` *and* occupies the exact same 0-based ranks it had
  before splitting (it's the block's own first half, in the same order) --
  its `assign_ranks` call was pure no-op writes and is now deleted
  outright, not just skipped. `merge_with_neighbor`'s `keep` block is the
  same story for its own entries (decoded first into the merge arena, so
  still at their original positions); only the absorbed neighbor's entries
  are actually new to that `BlockId`. The removed code's own comment
  claimed a full refresh was needed there -- it wasn't; this was a real,
  previously-unnoticed redundancy, not a new capability being added.

**Measured impact, honestly scoped**: an isolated microbenchmark
(`dictionary::tests::perf_probe_assign_ranks_skip`, a 256-entry block,
skip=0 vs. skip=128 -- the average case for one insert landing mid-block)
shows `assign_ranks` itself genuinely faster: 481ns/op unskipped vs.
283ns/op at the average skip, ~41%. But `perf stat` on the full
`examples/perf_insert_single_flush.rs` pipeline (50,000 preload + 50,000
single-flush inserts, same A/B method as the single-walk section above)
showed only a ~0.2% instructions-per-op reduction end to end
(1,623,524 -> 1,619,925), with cycles and cache-misses too noisy at that
size to read at all (the same DVFS-between-runs hazard as before). Both
numbers are true and consistent: `assign_ranks` is a small fraction of
total insert cost to begin with -- `Block::build`'s FST construction
dominates by orders of magnitude (see the "block rebuild breakdown" probe
in `block.rs`) -- so cutting it in half, even reliably, barely moves the
total. The `split`/`merge_with_neighbor` deletions are kept regardless:
they cost nothing (less code, not more) and are strictly correct, not a
speed bet. This is the accurate scope of "integrating the reverse map and
the blocks reduces wasted work" -- real, verified, and small, not the
large win the batching or single-walk changes were.

### Custom incremental micro-FST: stages 1-2 (bulk construction, proven correct)

`src/micro_fst.rs` is a from-scratch, self-contained minimal acyclic
automaton -- no dependency on `fst` or `crate::TermId` -- built toward
eventually replacing `Block`'s full-rebuild-per-mutation contract with
true incremental insertion (Daciuk, Mihov, Watson, Watson 2000, sorted-
input specialization; the transducer/output half per Mihov & Daciuk 1998;
deletion, when it's tackled, per Carrasco & Forcada 2002 -- full citations
and the rest of the literature considered, including why an unrolled
linked list was chosen for per-state transitions over double-array tries,
are in the approved plan this was built from). Staged deliberately, and
this session only covers the first two stages: bulk construction
(`Automaton::build_sorted`) and differential correctness testing against
the current `fst`-backed `Block` -- not incremental `insert` yet, and
nothing outside this module's own tests calls into it. `Block` and
`vendor/fst` are untouched; this changes no existing behavior.

Each state's outgoing transitions are stored as an unrolled linked list:
a `TransitionChunk` of up to 4 entries inline (no allocation), spilling
into a `Vec<TransitionChunk>` only for the rare wide node -- sized against
this project's own earlier finding that real RDF terms have low branching
factor at most nodes (the `s-bench` real-corpus profiling that produced
the binary-search "Correction" above), not assumed.

Minimization uses the standard register (a `HashMap` keyed by a state's
own transition set + finality, deduplicating equivalent states); rank
(the transition output value) isn't tracked incrementally during descent
-- it's derived once a state is confirmed, as the prefix sum over
lexicographically earlier sibling transitions of each sibling's subtree
word-count (`State::count`), itself a pure function of a state's own
(already-confirmed) subtree.

**A real bug, caught by the differential test, not by inspection.** The
first version of that output formula was `output(child) = sum of counts
of earlier siblings`, which passed simple cases (a single term; a term
that's nobody's prefix) but failed on real namespace-clustered data:
`automaton.rank("_:b10")` returned 1 where `Block::rank` (trusted, and
independently hand-verified) said 2. Root cause: a state that is *itself*
final and *also* has further transitions (e.g. the state after `"_:b1"`,
since `"_:b1"` is a complete term for `i=1` and also a prefix of
`"_:b10"`, `"_:b11"`, ...) accepts a word that's lexicographically smaller
than everything reachable through any of its children -- a prefix always
sorts before its own extensions -- so that state's own acceptance must
shift *every* child's output by one, not just contribute to the state's
own `count` for its parent's benefit. Fixed by seeding the running prefix
sum with `1` when the state is final, before folding in any child
(`finalize_outputs_and_count` in `micro_fst.rs`). Caught specifically
because the differential test used real, prefix-heavy, namespace-
clustered synthetic terms (8,000 of them, mixing IRIs, blank nodes, and
`urn:` terms) rather than only small hand-picked cases -- the exact kind
of case this project's "Correction" section already warned small/
artificial test data can hide.

### Custom incremental micro-FST: stage 3 (incremental insert, proven minimal)

`Automaton::insert(term)` adds one term to an already-finished, already-
minimal automaton -- a genuinely different problem from `build_sorted`'s
sorted-batch construction, not just "the same algorithm called once per
term". Bulk construction never needs to touch a state that's already
shared: sharing only happens *after* a state is confirmed and frozen. An
already-minimal automaton is full of shared states from the start, so
inserting into it safely requires copy-on-write: walk existing transitions
as far as `term` matches; clone every *non-root* state on that matched
path (any of them might be referenced by other terms, so none can be
mutated in place -- root is always safe, since nothing ever points back to
it, so it's exempt); extend the last clone with brand-new states for the
unmatched suffix; then re-confirm the whole modified path bottom-up
through the same register `build_sorted` populated, so a clone that
happens to now equal some other existing state gets merged away exactly
like any other duplicate, not as a special case.

**A second real bug, structurally different from the first, and again
only caught by testing at real scale with more than one insertion
order.** The initial implementation passed every hand-written small case
(idempotency, a term that's a prefix of another inserted either order,
inserting next to unrelated existing terms) but failed a differential test
against a fresh `build_sorted` at just 4 terms once shuffled into a
specific order. Root cause: `push_sorted` (the transition-list append used
while extending a path) assumed every new transition sorts after every
existing one at that node -- true for `build_sorted`, since sorted input
can never introduce a byte smaller than one already at a node once
construction has moved past it, but false in general for `insert`, which
accepts terms in *any* order. Inserting `"_:b0"` (byte `'_'`, 0x5F) after
`"urn:isbn:..."` (byte `'u'`, 0x75) had already been inserted left the
root's transitions as `['u', '_']` -- silently unsorted, which corrupted
every downstream computation depending on byte-sorted order (confirmation's
prefix-sum output calculation, `rank`, `term_at_into`) without ever
producing an out-of-range value or a panic, just wrong ranks that still
formed a valid permutation of `0..n` -- exactly the kind of bug a
narrower, single-seed test can pass by chance and a differential test
across several seeds reliably catches. Fixed by replacing the blind
append with `insert_transition`, which finds the correct sorted position
(rebuilding the chunk list from a small sorted `Vec`, cheap at this
corpus's measured branching factors) rather than assuming one.

Verified with `incremental_insert_matches_bulk_build_and_stays_minimal`:
500 unique ids (2,000 namespace-clustered terms) inserted one at a time,
across four different shuffles, each checked for (a) every rank/unrank
pair matching a trusted fresh `build_sorted`, (b) `reachable_state_count`
exactly equal to that fresh build's -- the actual minimality claim, not
just correct answers over a silently bloated structure -- and (c) a full
second pass re-inserting every term, confirming idempotency doesn't
regress minimality either. An ad hoc stress pass during development (not
kept, but worth recording) pushed this to 15,000 terms across five seeds
plus a double-insert pass with no failures, before paring back to the
scope above for the standing suite.

**Known, deliberate gap**: discarded clones and superseded originals are
left as unreachable garbage in the state slab rather than reclaimed --
correct (every read-path method only ever touches the reachable graph
from `root`) but a real, unaddressed memory-growth cost under sustained
incremental mutation. Slot reuse (mirroring `Dictionary`'s own
`free_block_slots` pattern) is future work, not attempted here, and
explicitly out of scope per the approved plan alongside deletion
(Carrasco & Forcada 2002) and wiring this into `Block` as a measured,
optional comparison implementation.

### Custom incremental micro-FST: measuring the actual hypothesis

The whole exercise was to test one hypothesis: does one insert into an
already-populated structure cost O(term length), independent of how many
other terms already exist, instead of `Block`'s O(block size) full
decode-and-rebuild -- the cost this session's own earlier measurements
(the single-walk merge refactor, the `assign_ranks` skip) confirmed
dominates every mutation by a wide margin. `micro_fst.rs`'s
`perf_probe_insert_cost_block` and `perf_probe_insert_cost_automaton` measure
exactly that, isolated from each other (two separate tests, not one
process running both loops, so `perf stat` can attribute hardware counters
to either side): 160 baseline terms (`block_max`'s midpoint, matching
`block::tests::perf_probe_block_rebuild_breakdown`'s own choice), one new
distinct term inserted per iteration, 20,000 iterations. `Block`'s timed
region is its whole real per-insert cost (decode, merge, rebuild --
nothing about a real `Dictionary::insert` call can skip any of it);
`Automaton`'s timed region is `insert` alone, with the per-iteration
`clone()` that resets it back to the 160-term baseline deliberately
excluded (a benchmark-harness artifact, not something a real caller
mutating one live automaton would ever pay).

Wall-clock (`cargo test --release -- --ignored --nocapture`): 34.8us/op
for `Block`, 8.5us/op for `Automaton::insert`, ~4.1x. Given this session
already found wall-clock-only comparisons misleading twice for effects far
smaller than this one, the same `perf stat` A/B discipline applied here
too rather than trusting a single large-looking number:

| metric (per op) | Block | Automaton::insert | ratio |
|---|---|---|---|
| cycles | 137,057 | 42,171 | 3.25x |
| instructions | 427,474 | 128,601 | 3.32x |
| cache-misses | 9.1 | 2.2 | 4.10x |
| branch-misses | 241.4 | 76.3 | 3.16x |

Every hardware counter agrees, independently, on roughly the same
magnitude (3.2-3.3x by cycles/instructions, the two least frequency-
sensitive metrics; a larger 4.1x on cache-misses, consistent with
`Block`'s rebuild touching all 160 entries' worth of FST bytes and
constructing a whole new compressed structure, versus `Automaton::insert`
touching only the one root-to-leaf path -- about 35 bytes here -- plus a
handful of register/`HashMap` operations). Wall-clock's 4.1x sits inside
that same range rather than contradicting it, unlike the two earlier
cases this session flagged -- this is what a real, unambiguous effect
looks like under the same measurement discipline that caught the fake
ones.

**What this does and doesn't establish.** The hypothesis behind building
this at all is confirmed, with real numbers, not just complexity-class
argument: incremental insertion into this automaton is a genuinely
different, and substantially cheaper, cost class than `Block`'s
full-rebuild contract, at a block size in the middle of this crate's
actual default bounds. It does *not* yet establish that swapping `Block`
for this would make `Dictionary::insert` 3x faster end to end -- `Block`
construction (`Block::build`'s own cost, not just the decode+merge around
it) was never the *only* cost in a real `Dictionary::insert` call
(routing, `assign_ranks`, batch grouping all still apply and are
unaffected by this), deletion isn't implemented on `Automaton` at all yet
(today's numbers only cover insert), and the known slot-reclamation gap
(discarded clones/originals left as unreachable garbage) is a real memory
cost under sustained mutation that this benchmark's single-shot
per-iteration `clone()`-and-discard pattern doesn't exercise. Those are
exactly the remaining items the approved plan already named before this
could be wired into `Dictionary` as an actual, measured comparison
implementation -- this result is the evidence that doing so would be
worth the remaining work, not a substitute for it.

### Custom incremental micro-FST: decode speed (the other half of the question)

The insert-cost result above only covers writes. Reads -- `rank` and
`term_at_into`, both directions -- went unmeasured, and there was a real
reason to expect either outcome: `Block::rank`/`term_at_into` walk `fst`'s
compact, bit-packed transition encoding, and this project's own earlier
profiling (the "Correction" section above) found that *bit-unpacking*
itself, not the search strategy around it, was the dominant per-transition
cost there. `Automaton`'s transitions are a plain unrolled array of
`(u8, u32, Option<StateId>)` -- no bit-unpacking at all -- so decode could
plausibly have gone either way, and needed its own measurement rather than
riding on the insert-cost result's coattails.

`perf_probe_decode_cost_block`/`_automaton` (same isolation-for-`perf
stat` pattern as the insert-cost probes; 2,000 namespace-clustered terms,
not a sequential digit suffix, for the same reason as always) measured it.
Wall-clock, `Block` is faster both directions: 189ns vs 273ns for `rank`
(~1.4x), 359ns vs 949ns for `term_at_into` (~2.6x). `perf stat`, blended
across both operations (one process runs both loops, so hardware counters
can't be split further than that without another round of test-splitting):

| metric (per op, rank+unrank blended) | Block | Automaton | ratio |
|---|---|---|---|
| cycles | 1,133.9 | 2,099.0 | 1.85x (worse) |
| instructions | 4,302.4 | 4,173.7 | 0.97x (fewer!) |
| branches | 619.2 | 509.1 | 0.82x (fewer!) |
| cache-misses | 0.5 | 0.5 | 0.99x (same) |
| branch-miss rate | 0.43% | 0.56% | 1.31x |

This is a real, `perf`-confirmed slowdown, not wall-clock noise -- but the
*shape* of the counters says it's an implementation-overhead problem, not
evidence that the unrolled-list structure is inherently worse for decode.
`Automaton` does fewer instructions and fewer branches than `Block`
per op, yet burns 85% more cycles doing it: cache-misses are essentially
identical (ruling out memory latency), so the likely cause is a genuine
instructions-per-cycle problem in `get`/`iter`/`term_at_into`'s current
form -- `std::iter::once(&self.inline).chain(self.overflow.iter())
.flat_map(...)` plus `.take_while(...).last()` and `.find(...)`, all
generic `Iterator` combinator machinery, matching exactly the shape of
overhead `block.rs`'s own `term_at_into` history already diagnosed once
before (`get_key_into`'s linear scan through generic-`Iterator` machinery
being ~41.5% of that function's runtime, well before bit-unpacking
entered the picture) and fixed by replacing it with a direct,
non-generic loop. `Automaton::term_at_into` still carries the doc comment
"correctness-first... once wired in and measured, not before" -- this is
that measurement, and it points at the same fix `Block` already needed:
replace the generic iterator chain with direct indexing over `inline`/
`overflow`, and only reach for binary search if a further profile shows
linear scan itself (not the iterator overhead around it) is the
remaining cost at the branching factors this corpus actually has.
Not done in this pass -- reported as found, per this project's practice
of keeping negative and unfinished results as visible as favorable ones.

**Fixed and remeasured.** `Transitions::get` and a new
`Transitions::last_output_at_most` (replacing `term_at_into`'s
`iter().take_while(..).last()`) now walk `inline`/`overflow` with direct
loops -- same answer, no `chain`/`flat_map`/`find`/`take_while` generic
`Iterator` machinery. `iter()` itself is kept, for the call sites that
genuinely want a full iterator (register-signature computation,
reachability walks, confirmation's per-target output pass) and aren't on
the hot decode path. Same `perf stat` A/B, same terms, same isolation:

| metric (per op, rank+unrank blended) | Block | Automaton | speedup |
|---|---|---|---|
| cycles | 1,201.4 | 574.3 | 2.09x |
| instructions | 4,239.7 | 1,258.7 | 3.37x |
| cache-misses | 0.5 | 0.4 | 1.32x |
| branches | 629.2 | 312.5 | 2.01x |

The direction fully reversed, not just narrowed: `Automaton` now does
3.37x *fewer* instructions (not the roughly-equal count from before the
fix) and takes 2.09x fewer cycles. That the instruction count itself
dropped this much, not just IPC improving, confirms the earlier
diagnosis: the generic iterator chain was genuinely doing more work, not
just executing the same work less efficiently. Wall-clock agrees in
direction and rough magnitude (`rank`: 340ns -> Block, 168ns ->
`Automaton`, ~2.0x; `term_at_into`: 612ns -> Block, 235ns -> `Automaton`,
~2.6x), consistent with the hardware counters rather than contradicting
them, same as the insert-cost result -- both reads and writes now
genuinely favor `Automaton` over `Block`, each confirmed the same way.

All differential/minimality tests (`matches_block_rank_and_unrank_on_
namespace_clustered_terms`, `incremental_insert_matches_bulk_build_and_
stays_minimal`, and the rest) still pass unchanged after this rewrite --
the fix only touched how an already-correct answer is computed, not what
answer is computed.

### Custom incremental micro-FST: closing the batch-insert gap

Everything measured so far about `Automaton` was single-term cost.
`Block::merge_into` earns its batching win (DESIGN.md's "Bounded-batch
sweep": ~70x fewer rebuilds going from per-term to batched flushing) by
merging many sorted incoming terms into one block in a single walk.
`Automaton::insert` had no equivalent -- calling it once per term in a
batch would re-walk from root and pay a fresh clone-on-write for every
term, even ones sharing a prefix *with each other* that the batch itself
just created. That gap was the stated reason not to consider `Automaton`
for `Dictionary`'s default yet, since this project's own data says
batching, not single-op cost, is the lever that matters most.

`Automaton::insert_sorted` closes it: one running `path` (the same
confirm-on-divergence shape `build_sorted` already uses) carried across
the whole sorted batch instead of reset per term, with a `batch_start_slot`
marker distinguishing states created during *this* call (safe to mutate
directly) from pre-existing ones (must be cloned before any write, same
as single `insert`).

**A suspicion raised, tested, and disproven before touching anything --
per explicit instruction, not by default.** Hand-tracing the design
surfaced a concern: `confirm_path` gets invoked on states that were only
*walked through* while matching an already-present term in the batch,
never actually modified -- and confirming touches the parent's transition
table. That looked unsafe on paper. Rather than "fix" it speculatively, a
naive implementation was written *without* any defense against this, and
two tests were built specifically to trigger it before any correction was
made: `insert_sorted_full_hit_before_a_diverging_sibling` (a hand-traceable
repro -- pre-existing `"xa"`/`"xb"` siblings under a shared parent, batch
`["xa" (hit), "xc" (new)]`, checking whether processing "xc" after the
"xa" hit corrupts the shared parent's existing `"xa"`/`"xb"` transitions)
and `insert_sorted_mixed_hits_and_new_across_rounds` (4 seeds x 8 rounds
x 40-term batches of mixed hits and new terms, against real
namespace-clustered branching, differentially checked against a fresh
`build_sorted` oracle after every round). Both passed against the naive,
unmodified implementation. The reasoning that explains why, worked out
*after* the tests confirmed there was nothing to fix: recomputing an
untouched state's outputs from its own (also unchanged) children
reproduces the same values it already had, and the register lookup that
follows then finds that state's own existing entry (or an equivalent one)
as canonical -- so the "unsafe" redirect is a no-op in effect. Wasted
work on those untouched entries, confirmed to be exactly that and nothing
more, not a correctness problem -- the naive version is what shipped as
`insert_sorted`, unmodified.

**Measured**, same isolation-for-`perf-stat` pattern as every other probe
in this section: a 2,000-term namespace-clustered baseline, 100 batches of
~500 new, sorted, disjoint terms each, comparing 500 sequential `insert`
calls per batch against one `insert_sorted` call:

| metric (per term) | sequential `insert` x N | `insert_sorted` | speedup |
|---|---|---|---|
| cycles | 38,706.8 | 2,395.7 | 16.16x |
| instructions | 103,966.4 | 6,142.6 | 16.93x |
| cache-misses | 233.4 | 5.8 | 40.55x |
| branches | 17,624.2 | 1,140.4 | 15.45x |
| wall-clock | 8.97us | 0.40us | ~22.4x |

A large, real, hardware-counter-confirmed win -- in the same territory as
`Block`'s own batching win, which was the whole point of closing this gap
before it could be a fair comparison. The especially large cache-miss
improvement (40x) makes sense structurally: sequential single-term
`insert` restarts from root and re-walks/re-clones shared prefixes once
per term, scattering memory accesses across the state slab call after
call, while `insert_sorted` processes prefix-sharing terms together,
keeping the working set hot. Wall-clock (~22.4x) is in the same range as
the hardware counters, not contradicting them.

**Still open, unchanged from before**: deletion isn't implemented (insert
only), no slot reclamation yet, and this still hasn't been measured wired
into an actual `Dictionary` at realistic scale with routing and the
reverse map in the loop -- this closes one specific, previously-named gap,
not the whole remaining list before a "default" decision.

### Custom incremental micro-FST: deletion

`Automaton::remove` closes the deletion gap (Carrasco & Forcada 2002's
territory, cited since Stage 1). Shape: walk the exact matching path (a
partial match, or a full match landing on a non-final state, means the
term isn't present -- no-op, same as `insert`'s idempotency); clone every
non-root state on that path, copy-on-write, for the same reason `insert`
does (any of them might be shared); clear `is_final` at the (now private)
leaf; then prune bottom-up -- a state with no transitions left and not
final has no reason to exist, so its incoming transition is *deleted*
from its parent outright (`Transitions::remove_transition`, new, the one
mutation here that isn't a redirect), and the check repeats one level up,
stopping at the first survivor (still final, or still has transitions).
Whatever survives is re-confirmed through the same register machinery as
every other mutation.

**Tested per the explicit instruction this was built under: write tests
that prove and measure a suspicion before fixing it, not after.** Five
targeted cases first (absent term is a no-op; removing the only term
empties the automaton down to just the root; removing a term that's a
prefix of a surviving longer term stops it being final without pruning
it away; removing one sibling under a shared parent leaves the other
sibling's transition untouched; remove-then-reinsert matches a fresh
build), then the real test:
`mixed_insert_batch_and_remove_across_rounds_matches_a_fresh_build` --
4 seeds x 12 rounds randomly choosing between single `insert`, batch
`insert_sorted`, and `remove` against real namespace-clustered branching
content, differentially checking every surviving rank *and* unrank *and*
`reachable_state_count` against a fresh `build_sorted` oracle over the
exact current live set after every single round. All pass.

### Custom incremental micro-FST: the slot-reclamation gap, measured

The known gap flagged since Stage 3 (discarded clones/originals left as
unreachable garbage in the state slab, never reclaimed) was measured, not
assumed, before any decision about fixing it -- per instruction, same
discipline as the deletion suspicion above. `state_slab_len()` (new,
`#[cfg(test)]`, exposes `states.len()`) against `reachable_state_count()`
under `measure_slot_reclamation_gap_under_sustained_churn`: sustained
one-out-one-in churn (remove one live term, insert one never-seen term,
keeping the live set size constant at 800) for 2,000 rounds:

| round | slab size | reachable | ratio |
|---|---|---|---|
| 0 | 907 | 91 | 1.0x |
| 200 | 12,731 | 182 | 70.0x |
| 1,000 | 62,271 | 241 | 258.4x |
| 2,000 | 126,990 | 244 | **520.5x** |

The result is unambiguous: the slab grows essentially linearly with
cumulative mutation count (~63 new slab entries per round, matching one
`remove` plus one `insert` each cloning/creating a handful of states)
while the reachable/live count stays flat, as it should for a fixed-size
sliding window. This is a real, severe, unbounded-growth cost under
sustained mutation -- confirmed with numbers, not a theoretical concern --
and would make `Automaton` unsuitable for any long-running workload
without slot reuse (mirroring `Dictionary`'s own `free_block_slots`
pattern) before it could be trusted as a default. Left unfixed in this
pass, per instruction: measuring it was the assignment, not fixing it yet.

### Custom incremental micro-FST: wiring into the dictionary (partial)

`src/automaton_block.rs`'s `AutomatonBlock` mirrors `Block`'s exact public
interface (`build`, `len`, `min_key`, `rank`, `id_at`, `term_at_into`,
`term_at`, `decode_into`, `merge_into`, and now `remove`) but backed by
`Automaton` instead of `fst::Map` -- the piece that actually makes the
automaton usable as a `TermId`-aware block, not just a standalone `term
<-> rank` structure. `Automaton` itself stays deliberately decoupled from
`TermId` (per its own module docs); `AutomatonBlock` is the thin layer
that pairs it with the external `ids: Vec<TermId>` array `Block` already
carries.

`merge_into`/`remove` here are a real, deliberate simplification, not an
oversight: rather than tracking an `unchanged_prefix` the way `Block`'s
own do (an O(changed suffix) reverse-index update -- see the "Skipping
unchanged reverse-map entries" section above), they rebuild the whole
`ids` array from a decode of the old content plus the automaton's own
mutation every time -- O(block size) regardless of how much actually
changed. This is a smaller first integration step: prove `Automaton` can
correctly serve the block role at all, before investing in matching
`Block`'s own incremental-update refinement here too. Tested
differentially against `Block` (`matches_block_on_build_rank_and_unrank`,
`decode_into_matches_original_content`) and for its own mutation
contracts (`merge_into_existing_terms_keep_their_id_new_ones_get_fresh_
ids`, `remove_drops_the_term_and_keeps_everything_else_correct`) -- ids
aren't compared byte-for-byte against `Block`'s own allocation order
(the two implementations allocate in different, equally valid orders),
but internal consistency (every surviving term keeps its id, every new
term gets one, round-trips hold) is.

**What this is not yet**: `AutomatonBlock` is not wired into `Dictionary`
itself. `Dictionary`'s router, split/merge, and batch-grouping machinery
(`dictionary.rs`, ~700 lines) all assume `Block` specifically; making
`Automaton` a real alternative there means either a parallel top-level
type (mirroring how `FrontCodedDictionary` sits alongside `Dictionary`)
or making `Dictionary` generic over its block type, both of which are a
substantially larger undertaking than this block-level piece --
duplicating or restructuring well-tested, currently-correct routing logic
carries real risk of introducing new bugs under the same kind of subtle
conditions this whole effort has already found twice (the sortedness
bug, the output-formula bug). Stopped here deliberately rather than
rush that next phase in the same pass; `AutomatonBlock` itself is
solid and tested, ready for whichever integration approach comes next.

### Custom incremental micro-FST: reference-counted slot reclamation

Fixes the measured gap above with reference counting, per explicit
direction, over the mark-and-compact alternative that was recommended
instead (lower risk, reuses the already-trusted reachability walk) --
the tradeoff named at the time (refcounting touches every mutation site,
real new surface area in code that had already produced two genuine bugs
this session) turned out to be exactly right: it produced a third.

**Design**: `refcounts: Vec<u32>` parallel to `states`, incremented by
`acquire` and decremented by `release`; `release` cascades (a dead
state's own children are released too -- a whole unreachable subtree can
hang off one redirected transition) and pushes the freed slot onto
`free_slots`, which `alloc_state` checks before growing the slab.
`redirect` (used everywhere a transition gets repointed, including
`confirm_path`'s own register-merge step) acquires the new target
*before* releasing the old one, specifically because the overwhelmingly
common case -- a freshly confirmed state registering as its own canonical
entry, i.e. `new_target == old_target` -- would otherwise see the
refcount bounce through zero and get wrongly cascade-freed a moment
before being re-acquired.

**Two real bugs, both caught by tests before being "fixed" on paper,
exactly per instruction:**

1. `insert_sorted`'s existing "was this state created during the current
   batch" check compared raw slot *numbers* against a snapshot taken at
   batch start. Slot reuse breaks that outright: a slot freed and reused
   *within* the same batch can land below that snapshot's number while
   being entirely private to the batch, misclassifying it as pre-existing
   and shared. Caught by reasoning before any code ran, not by a failing
   test -- fixed by comparing *generations* (a counter stamped on a slot
   every time `alloc_state` assigns it, fresh or reused) instead of slot
   numbers, which is correct regardless of reuse.
2. `clone_state` copies a state's transitions, which duplicates every
   reference they hold -- the clone is a *new*, independent referrer to
   each target, same as if each transition had just been created fresh.
   The first implementation never `acquire`d those targets on the
   clone's behalf. Result: releasing the *original* later (once its own
   clone superseded it) wrongly cascaded into freeing children the clone
   still legitimately pointed at -- a live, still-referenced subtree
   getting torn down out from under it. This one **was** caught by a
   test, immediately: 8 of 17 correctness tests failed the moment this
   landed, several with a direct panic (`redirect`'s "byte must already
   have a transition here" -- the freed child's transition had vanished
   from under a state that still needed it). Bisected with a minimal
   repro (`"xa"`/`"xb"`, passed) escalated to a shared-prefix-length
   sweep (`n=1,2` passed, `n=3` failed -- the bug needs at least 3 chained
   clone-loop iterations to manifest, which is why the trivial 1-byte
   case missed it and a real term like `"http://example.org/a"` vs
   `"http://example.org/b"` -- 20 shared bytes -- didn't). Fixed by
   having `clone_state` `acquire` every target the copied transitions
   point at, immediately after cloning.

**Verified the fix actually does what it's for**, not just that it
compiles and passes the existing suite: the same
`measure_slot_reclamation_gap_under_sustained_churn` probe that found the
500x+ growth now reports:

| round | slab | reachable | ratio |
|---|---|---|---|
| 0 | 94 | 91 | 1.03x |
| 1,000 | 242 | 241 | 1.00x |
| 2,000 | 253 | 246 | **1.03x** |

The slab stabilizes (253, not growing further) instead of climbing
linearly forever. One new targeted stress test
(`insert_sorted_after_churn_reuses_slots_correctly`) specifically forces
`free_slots` to be non-empty going into large batch inserts (interleaving
insert-then-immediately-remove churn rounds with 80-term batches, 3
seeds) -- exactly the generation-vs-slot-number scenario bug #1 was
about -- and passes.

**The cost, measured, not assumed away**: refcounting bookkeeping isn't
free. Re-running the earlier insert-cost probe: `Automaton::insert` is now
~11.5us/op (was ~8.5us/op before this), narrowing its wall-clock margin
over `Block` from ~4.1x to ~3.0x (the hardware-counter figure was ~3.3x
before this section; not re-measured with `perf stat` again here, since
the direction and rough size of the change is unambiguous at this margin
-- unlike the two cases earlier in this document where `perf stat` caught
wall-clock giving the *wrong sign*, this one only moved the number, not
its direction). Decode is unaffected (it never touches refcounts, so
`perf_probe_decode_cost_*`'s ~2.15-2.25x held exactly). Batch insert's
*relative* advantage held or slightly widened
(sequential single-term `insert` now pays the refcounting overhead once
per term, more exposed to it than `insert_sorted`, which shares more of
that cost across the batch) -- still on the order of the ~16-26x range
measured before. The headline results from every earlier section stand;
the margins are honestly a little smaller now, which is the expected
price of the correctness this section's fixes bought, not a surprise to
paper over.

### Custom incremental micro-FST: wired into the dictionary, for real this time

`AutomatonDictionary` (`src/automaton_dictionary.rs`) closes the gap the
earlier "wiring into the dictionary (partial)" section named: `Dictionary`'s
exact macro structure (lexical blocks, sorted-array router, split/merge
on overflow/underflow, `TermId -> (BlockId, local_rank)` reverse mapping),
`AutomatonBlock` in place of `Block`. This is the crate's fourth
`TermDictionary` implementation (after `PlainDictionary`, `Dictionary`,
`FrontCodedDictionary`) and the first point where the custom automaton is
measurable end to end, through the same trait every other implementation
shares, rather than only in isolated block-level microbenchmarks.
`Dictionary` and `Block` are untouched -- this crate's own "alongside,
promoted only if it wins" rule, unchanged since the plan that started this
whole effort.

Ported mechanically from `Dictionary` wherever `AutomatonBlock`'s
interface matches `Block`'s exactly (`route`, `bootstrap`,
`split_while_oversized`, `split`, `merge_with_neighbor`,
`remove_empty_block`, the `TermDictionary` methods), reusing
`Dictionary`'s own proven `mem::take`-scratch-arena pattern rather than
improvising a new one -- and it compiled clean, and passed its first real
test run clean, on the first attempt, for exactly that reason: the risky
part (getting `Automaton`/`AutomatonBlock` correct) was already done and
tested in the two sections above, and the borrow-checker patterns needed
here were already worked out by `Dictionary` itself, not new problems to
solve blind.

**Smaller in scope than `Dictionary`, deliberately, not an oversight**:
no `PreparedBatch`/`insert_prepared`, no `get_term_into`/`get_terms_into`,
no `RebuildStats`/`size_bytes`. `assign_ranks` here has no
`unchanged_prefix` skip either, since `AutomatonBlock::merge_into`/
`remove` don't report one (they rebuild their whole `ids` array
internally regardless, per their own documented simplification) -- there
is nothing to skip on this side, consistent with that choice rather than
fighting it. All real optimizations `Dictionary` accumulated over this
project's history; porting them is future work if this comparison proves
worthwhile, not a prerequisite for measuring it.

**Tested by extending the crate's existing 3-way fuzz differential
(`tests/dictionary.rs`) into a 4-way one**, rather than writing a new,
separate test harness: `AutomatonDictionary` now sits in the same
lockstep loop as `Dictionary`, `FrontCodedDictionary`, and the
`PlainDictionary` oracle, across the same randomized insert/remove
sequences and the same small-block-bounds configuration that forces
splits and merges every run. Every id, every `get_id`, every `get_term`,
every `len()` is checked against the oracle at every step, for all three
blocked implementations simultaneously -- not a new, separate test
philosophy, the same one `FrontCodedDictionary` was already held to,
now covering a third implementation in the same run. Plus a dedicated
round-trip test (`automaton_dictionary_round_trips_real_rdf_terms`,
small block bounds so the 7-term set actually forces a split) and an
`insert_batch` check. All pass.

One incidental finding while wiring this up: `Automaton::insert` (the
single-term path) turned out to be genuinely unreachable from non-test
code once `AutomatonBlock::merge_into` was written to route every
insertion -- single term included -- through `insert_sorted` uniformly,
rather than special-casing. Kept, not deleted: it's a distinct,
separately tested algorithm, not `insert_sorted` with a one-element
iterator in disguise, and a real candidate to reach for if a future
profile shows `insert_sorted`'s small extra bookkeeping costs anything
measurable for the single-term case specifically -- flagged with
`#[allow(dead_code)]` and a comment explaining exactly this, not a
silent workaround.

### Custom incremental micro-FST: measured end to end -- a mixed, honest result

Every measurement up to this point was either a correctness check or an
isolated block-level microbenchmark of `Automaton` alone (insert ~3x
faster than `Block`'s decode-and-rebuild, decode ~2-3.4x faster,
batch-insert 16-26x faster than sequential single inserts). None of that
had yet been checked against `AutomatonDictionary` as a whole
implementation, through the same `TermDictionary` trait every other
implementation shares, with routing and split/merge in the loop. New
example, `examples/perf_dictionary_automaton_vs_fst.rs`: five modes
(`bulk_insert`, `sequential_insert`, `get_id`, `get_term`, `remove`), each
runnable as its own process per dictionary kind for clean `perf stat`
attribution, `_setup`-only variants added for the four modes that
preload data (needed because `perf stat` times the whole process, not
just the `Instant`-wrapped operation -- see below for why that mattered).
Namespace-clustered terms throughout, 20,000 groups (80,000 terms),
`Dictionary`'s own default block bounds (64..=256) on both sides.

Wall-clock first:

| mode | fst (`Dictionary`) | automaton (`AutomatonDictionary`) | ratio |
|---|---|---|---|
| bulk_insert | 539 ns/op | 1.054 us/op | **0.51x** (slower) |
| sequential_insert | 62.78 us/op | 132.87 us/op | **0.47x** (slower) |
| get_id | 291 ns/op | 224 ns/op | 1.30x faster |
| get_term | 498 ns/op | 250 ns/op | 1.99x faster |
| remove | 28.30 us/op | 92.39 us/op | **0.31x** (slower) |

Three of five modes got *slower*, not faster -- the opposite of what
every prior block-level number implied. Rather than trust or dismiss
this on wall-clock alone (this project's own established rule, broken
twice already this session before it stuck), each headline number was
cross-checked with `perf stat` hardware counters, isolating setup from
the measured operation via the `_setup` variants:

- `sequential_insert`: whole-process `perf stat` (setup here is small
  relative to the 20,000-op timed loop, so no subtraction needed) showed
  **3.25x more cycles**, 2.76x more instructions, **10.8x more
  cache-misses**, 6.5x more branch-misses for automaton. Confirms the
  wall-clock slowdown; if anything understates it.
- `get_term`: whole-process counters were *not* clean here -- automaton's
  slower `bulk_insert`-shaped preload (80,000 terms) dominates the total
  and nearly cancels the lookup-loop win, which is exactly why the
  `_setup` subtraction exists. After subtracting `get_term_setup` from
  `get_term`: fst ~2148 cycles/op vs automaton ~1179 cycles/op (**1.82x
  fewer cycles**), instructions 6691 vs 2404/op (**2.78x fewer**).
  Confirms the wall-clock win, same direction, different magnitude.
- `get_id`: same subtraction method, fst ~1251 cycles/op vs automaton
  ~763 cycles/op (**1.64x fewer**). Confirms.
- `bulk_insert`: no preload to subtract (dictionary starts empty).
  Whole-process counters: **1.65x more cycles**, 1.80x more instructions
  for automaton. Confirms the wall-clock slowdown.

So the lookups (`get_id`, `get_term`) really are faster end to end, at
roughly the magnitude the block-level decode numbers predicted. But
`bulk_insert`, `sequential_insert`, and `remove` are genuinely,
hardware-counter-confirmed *slower* -- this is a real negative result,
not a measurement artifact.

**Why**, diagnosed the same way every other suspicion in this project was
handled: written down as a hypothesis, then checked with a targeted test
before being trusted. Two candidate causes, both already documented as
"simplifications" when they were built but never actually cost-measured
against `Block`'s equivalent path:

1. `AutomatonBlock::merge_into` rebuilds its whole `ids` array on every
   mutation via two full `HashMap<String, TermId>` passes over the
   block (one to snapshot old term->id, one to look surviving terms back
   up after the automaton mutates) -- each entry costs a fresh `String`
   allocation and a hash, on top of `Automaton::insert_sorted`'s own
   (fast) mutation. `Block::merge_into`, by contrast, does one single
   FST-stream walk that reads old entries and merges in new ones in the
   same pass, writing straight into an arena -- no `HashMap`, no
   per-entry `String` allocation.
2. `AutomatonDictionary::resync_after_mutation` then does a *second*
   full block decode (`decode_into`) plus a full `assign_ranks` pass over
   every entry, unconditionally, on top of whatever `AutomatonBlock`
   already did -- because `AutomatonBlock::merge_into`/`remove` don't
   report an `unchanged_prefix`, `assign_ranks` here has no skip and
   redoes the whole block's reverse mapping every time. `Dictionary`'s
   own `insert`/`insert_ordered` use `MergeOutcome::unchanged_prefix` to
   skip re-touching the reverse mapping for everything before the first
   change.

New isolated test, `automaton_block::tests::
perf_probe_merge_into_single_term_automaton_vs_block` (`#[ignore]`d,
wall-clock, block-level only -- no dictionary-level resync on top),
confirms cause (1) is real and already dominant on its own: 2,000 single-
term `merge_into` calls into one growing ~256+-entry block, 945 us/op for
`AutomatonBlock` vs 20.4 us/op for `Block` -- **~46x slower**, well before
`AutomatonDictionary`'s own extra resync pass gets a chance to add
anything on top. (The block in this probe grows unboundedly rather than
being capped at `block_max`, so the absolute ratio here is worse than
what shows up end to end where blocks stay near 256 -- but it isolates
that the block-level `ids`-rebuild cost alone, independent of the
dictionary layer, already outweighs `Automaton`'s own mutation advantage.)

**The honest conclusion**: the "insert 3x faster" and "batch-insert
16-26x faster" numbers earlier in this file were real, but measured a
narrower thing than they were being read as -- the bare `Automaton`
primitive's mutation cost in isolation, not the actual cost of using it
as a block backend the way `AutomatonBlock`/`AutomatonDictionary`
currently wrap it. The wrapping's own documented "simplification" (full
`ids` rebuild instead of `Block`'s single-walk incremental update) was a
real, unmeasured cost, and it's large enough to flip the sign on every
mutating operation end to end, even though the underlying automaton
really is cheaper to mutate. `bulk_insert`'s regression is a separate,
previously-unmeasured gap: `Automaton::build_sorted`'s from-scratch
construction speed was never checked against `fst`'s `MapBuilder`
directly (every earlier "faster" claim was about avoiding a rebuild, not
about how fast the rebuild itself is when one does happen), and it turns
out to be slower, plausibly because `fst`'s `MapBuilder` is a mature,
heavily optimized production implementation, and this project's own
register-based construction has had no equivalent optimization pass.

Net effect: `AutomatonDictionary` is **not** currently a strict win over
`Dictionary`, and should not be promoted to default status as it stands.
It is a strict win for read-heavy, mutation-light workloads (lookups
faster in both directions, confirmed by hardware counters, not just wall
clock), and a strict loss for anything insert/remove-heavy, also
confirmed. The path to actually cashing in the underlying automaton's
real mutation advantage runs through giving `AutomatonBlock::merge_into`/
`remove` an `unchanged_prefix`-tracked incremental `ids` update (closing
gap 1) and `AutomatonDictionary::resync_after_mutation` the same skip
`Dictionary` already has (closing gap 2) -- both flagged as scope
reductions when originally built, now confirmed by measurement, rather
than assumption, to actually matter. Not started; this section reports
the measurement this session was asked for, not a fix for what it found.

### Custom incremental micro-FST: closing the gap, for real this time

Both diagnosed causes fixed, each verified with the same isolated probe
and `perf stat` discipline used to find them -- not assumed fixed because
the change looked right.

**Fix 1: `AutomatonBlock::merge_into`/`remove` rewritten, no `HashMap`.**
`merge_into` is now a single sorted-merge walk of the block's existing
content against `incoming`, mirroring `Block::merge_into`'s algorithm
exactly -- read the old term at each rank, compare against the next
incoming term, decide old/new/matched, push into a freshly built `ids`
array, track `unchanged_prefix` the same way `Block::MergeOutcome` does.
No `HashMap<String, _>`, no per-entry `String` allocation. `remove` needs
no walk at all: removing one entry from a sorted sequence only shifts
everything after it, so `self.ids.remove(old_rank)` is exactly correct,
and `unchanged_prefix` is just `old_rank` -- the original implementation's
full HashMap-rebuild-on-removal was solving a problem that didn't exist.
`AutomatonDictionary::assign_ranks` gained the matching `skip` parameter
(mirroring `Dictionary`'s own), and turned out to need no decode at all
once written that way: it only ever consumed `(local_rank, id)` pairs, and
`AutomatonBlock::id_at` gives those directly, so `resync_after_mutation`'s
separate `decode_into` pass was removed entirely, not just skipped.

Measured on the isolated block-level probe (`perf_probe_merge_into_
single_term_automaton_vs_block`, no dictionary layer at all): **46x
slower than `Block::merge_into` -> ~3.3x slower**. Real progress, but the
probe's own before/after numbers made clear something structural was
still unaccounted for -- worth checking rather than declaring done at a
plausible-looking partial win.

**Fix 2: `Automaton::stream`, an incremental walk, replacing repeated
`term_at_into` calls.** The remaining ~3.3x traced to an architecture
mismatch, not a leftover inefficiency in the new merge walk: it read old
content via `term_at_into(rank, ..)` once per rank, and `term_at_into`
always restarts from the automaton's `root` -- paying the full path
length on *every* call, no matter how much of it was just walked for the
previous rank. `Block::merge_into`/`decode_into` use `fst::Map::stream()`,
which is a genuinely incremental walk: consecutive keys sharing a prefix
reuse that traversal, paying only for the divergence. This is a different
axis from the earlier, already-measured `term_at_into`-vs-`Block::
term_at_into` point-lookup comparison (`perf_probe_decode_cost_automaton`,
where `Automaton` already won) -- that probe compared two *non-incremental*
APIs against each other; `Block::merge_into`'s real code path was never a
fair comparison for `term_at_into` to be measured against in the first
place.

`Automaton::stream` (`micro_fst.rs`) is the equivalent incremental walk:
a preorder DFS over the trie with an explicit stack (one frame per state
on the current root-to-frontier path, each tracking its next child index
via a new direct-loop `Transitions::at`, not `Transitions::iter`'s
combinator chain -- consistent with this module's own earlier finding
that the combinator chain, not the scan itself, was the real cost).
`AutomatonBlock::merge_into` and `decode_into` were both switched to it.
Tested directly (`micro_fst::tests::stream_matches_term_at_into_for_every_
rank`, `stream_backtracks_correctly_across_shared_and_diverging_prefixes`),
not just relied on indirectly through the existing block-level tests,
which also all still pass unchanged (the streaming rewrite is a pure
internal implementation change; no public contract moved).

Measured on the same isolated block-level probe: **~3.3x slower ->
`AutomatonBlock` `47.4us/op` vs `Block` `14.4us/op`**, i.e. still nominally
behind at this micro-benchmark's specific (small, unboundedly-growing)
block-size regime, but the real test is the end-to-end numbers below,
which is where this was always meant to be judged.

**End-to-end re-measurement** (`examples/perf_dictionary_automaton_vs_fst.rs`,
same methodology as the original "measured end to end" section: 20,000
namespace-clustered term groups, `Dictionary`'s default block bounds,
`perf stat` cross-checking wall clock):

| mode | before fix | after fix | `perf stat` confirmation |
|---|---|---|---|
| `sequential_insert` | 0.47x (slower) | **2.9x faster** | 1.87x fewer cycles, 2x fewer instructions, 5.1x fewer cache-references |
| `remove` | 0.31x (slower) | **2.15x faster** | 2.07x fewer cycles, 2.2x fewer instructions, 5x fewer cache-references |
| `get_id` | 1.30x faster | 1.39x faster | unchanged (this path was never touched) |
| `get_term` | 1.99x faster | 2.01x faster | unchanged (this path was never touched) |
| `bulk_insert` | 0.51x (slower) | 0.51x (slower), unchanged | not addressed -- see below |

`sequential_insert` and `remove` both flipped from confirmed losses to
confirmed wins, at roughly 2-3x, and by a wide enough hardware-counter
margin (cache-references down 5x on both) that this isn't a borderline
call. `get_id`/`get_term` were untouched by this fix (their code paths
never went through `merge_into`/`remove`/`assign_ranks`) and remain
exactly the wins they were.

**`bulk_insert` is deliberately not addressed here.** It was already
diagnosed as a different problem in the original measurement section:
`Automaton::build_sorted`'s from-scratch construction speed against
`fst`'s `MapBuilder`, not a wrapping-overhead question at all -- nothing
this fix touched (`merge_into`, `remove`, `assign_ranks`, `stream`) sits
anywhere on `build_sorted`'s path. Confirmed unchanged by `perf stat`
(128.8M vs 220.7M cycles for 20,000 fresh terms, same ~1.7x gap as
before). A real, separate, still-open item.

**Net effect**: four of five measured operations are now confirmed wins
for `AutomatonDictionary` over `Dictionary` -- reads by ~1.4-2x, and now
mutations too, by ~2-3x. Only fresh bulk construction remains a loss.
This changes the promotion calculus from the previous section's "not a
strict win, don't promote" to "a strict win for every workload except
cold/bulk construction" -- still not a blanket recommendation to promote
to default (bulk construction matters for any workload that loads a
large corpus from scratch, and `AutomatonDictionary` still lacks
`insert_prepared`/`get_term_into`/`get_terms_into`/`RebuildStats`/
`size_bytes`), but a materially different, more favorable position than
before this fix, and a fix driven end to end by measurement at every
step: probe, diagnose, fix, re-probe, re-measure end to end, rather than
assuming a plausible-looking change worked.

### Custom incremental micro-FST: closing the bulk-construction gap

`bulk_insert` (`AutomatonBlock::build` -> `Automaton::build_sorted`, the
fresh-construction path `AutomatonDictionary::bootstrap` uses on an empty
dictionary) was the one operation the previous two fixes didn't touch --
neither `merge_into`/`remove`'s rewrite nor `Automaton::stream` sits
anywhere on `build_sorted`'s path. A first premise check mattered here:
the earlier "16-26x faster batch insert" number was about
`Automaton::insert_sorted` (a batch inserted into an *existing*
automaton) against N sequential `Automaton::insert` calls -- an entirely
different operation from `build_sorted` (fresh construction from
nothing), never previously measured against `fst`'s `MapBuilder` at all.
Checked by reading both `build_sorted` and `insert_sorted`'s code before
assuming one could substitute for the other: `insert_sorted` does
strictly *more* per-term work than `build_sorted` (a walk attempting to
match existing transitions, generation bookkeeping) with nothing to gain
from an empty starting automaton, so there was no shortcut there --
`build_sorted` needed its own, real profiling.

**Profiled with `perf record`** (a `profiling` Cargo profile already
existed with `debug = true, strip = false`, so this needed no new
tooling), not reasoned about from reading the code alone. Three real
findings, fixed one at a time, each re-measured before moving to the
next:

1. **`link_new` (used by every state `build_sorted`/`insert_sorted`
   create) unconditionally called `Transitions::insert_transition`** --
   the general, any-position rebuild-from-a-sorted-`Vec` path, needed by
   `Automaton::insert` (a single, order-independent incremental insert,
   which can genuinely diverge anywhere among a state's existing
   transitions) but *not* by `build_sorted`, whose own doc comment
   already noted "sorted input never branches backwards at a given
   node." Added `link_new_appending`, calling `Transitions::append_in_order`
   directly (no allocation, no rebuild) -- applied only to `build_sorted`'s
   call site, confirmed safe by the empty-start invariant (nothing exists
   yet to diverge before). *Not* applied to `insert_sorted`'s own
   extension loop: that one mutates a possibly non-empty existing
   automaton, where a new branch can genuinely need to land *between* two
   already-existing siblings (e.g. existing "cx"/"cz", inserting "cy") --
   a case `append_in_order` would silently place in the wrong position,
   and the existing test suite didn't happen to cover it. Checked by
   reasoning, not risked: left on the safe general path.
2. **`Signature::of`/`register: HashMap<Signature, StateId>` were the
   next-largest cost** -- `perf record` showed `SipHash` hashing
   (`std`'s default, deliberately slow for adversarial-input resistance)
   plus the `Vec` equality comparison it triggers as the two biggest
   remaining line items. `register` is a purely internal map over this
   module's own derived keys, never attacker-influenced, so DoS
   resistance buys nothing here -- switched to `FxHasher` (`micro_fst.rs`),
   the same simple multiply-rotate-xor hash `rustc`'s own compiler
   internals use for exactly this kind of hot internal map, via
   `HashMap<_, _, BuildHasherDefault<FxHasher>>` (a `type Register`
   alias, no new dependency).
3. **`Signature::of` and `finalize_outputs_and_count` were both still
   using `Transitions::iter()`'s combinator chain** (`Chain`, `FlatMap`,
   `Map`) to collect into a fresh `Vec` once per confirmed state -- the
   exact pattern this module's own earlier decode-speed investigation
   already found costly (see `Transitions::iter`'s doc comment), just not
   yet applied to these two call sites. Rewritten as direct loops over
   the already-added `Transitions::at`; `finalize_outputs_and_count` also
   lost its separate `targets: Vec<StateId>` entirely (folded into the
   same loop that builds `outputs`, since nothing else used `targets`).

Ratio on the isolated `bulk_insert` end-to-end measurement, `perf
stat`-confirmed at each step (instructions used as the primary metric for
the final numbers -- cycles proved noisier run to run on this machine,
consistent with this project's own earlier measurement-hygiene notes):

| stage | automaton cycles (approx) | ratio vs `fst` |
|---|---|---|
| before any bulk-insert fix | 220.7M | 1.71x slower |
| + `link_new_appending` | 172.4M | 1.37x slower |
| + `FxHasher` | 150.4M | 1.20x slower |
| + direct-loop `Signature`/`finalize_outputs_and_count` | ~154.8M (avg of 3 runs; cycles noisy) | ~1.13x slower (~1.10x by instructions, the more stable metric) |

**Not fully closed.** What's left, per a final `perf record` pass: the
single largest remaining cost is `BatchOrder::prepare`'s input sort
(`sort_unstable_by` on the incoming terms) -- confirmed via a `--call-graph
dwarf` trace, not assumed from the flat profile, which would have
wrongly blamed it on `Signature` comparison. This sort is identical code
shared by `Dictionary::insert_batch` and `AutomatonDictionary::insert_batch`
alike, so it isn't a gap between the two implementations at all -- both
sides pay it. The genuinely `Automaton`-specific remainder (register
lookups, `confirm_path`, state allocation) is now small enough that
closing it further would mean either an unsafe zero-allocation register
lookup (the `Signature` `Vec` still costs one real allocation per
confirmed state on a register miss; avoiding that on stable Rust without
a `hashbrown`-style `raw_entry`/`Equivalent` API isn't clean, and wasn't
attempted) or accepting the remaining ~1.1x as close enough. Left as-is
here, reported rather than pushed further, per this session's own
"measure, don't over-fit a diminishing-returns tail" instinct.

**Net effect across all five measured operations, now**: `bulk_insert`
~1.1x slower (down from ~1.95x when first measured, effectively closed);
`sequential_insert` **2.33x faster** (up from 2.9x measured right after
the `merge_into` fix -- further improved by this session's own shared
infrastructure, `confirm_path`/`Signature::of`/`FxHasher`, that
`insert_sorted` and `remove` also go through); `remove` **3.15x faster**
(up from 2.15x, same reason); `get_id` 1.33x faster; `get_term` 2.06x
faster -- all `perf stat`-confirmed. Four wins and one near-parity, from
a starting point of three outright losses out of five.

### Custom incremental micro-FST: an extensive analysis of what's left in `build_sorted`

A deeper follow-up investigation, specifically into *construction*
(`Automaton::build_sorted` / `Block::build`) rather than any mutation
path, using `perf record`/`perf stat` more rigorously than the previous
sections -- and finding that the previous section's own `bulk_insert`
numbers (1.1x slower) were themselves measuring something subtly wrong,
which this analysis had to correct before it could trust anything else.

**New isolated benchmark infrastructure**, because neither existing tool
was clean enough for this question: `examples/perf_micro_fst_build.rs`
(a standalone binary, `automaton`/`block` x `build`/`setup` modes, the
`setup` mode existing purely so `perf stat` can subtract term-generation/
sorting/chunking cost from the whole-process counters) plus
`src/lib.rs`'s `#[doc(hidden)] pub mod __internal_bench`, exposing thin
wrappers around the otherwise-`pub(crate)` `Automaton::build_sorted` and
`Block::build` so an external example binary can reach them. Chunked to
256-term pieces (`Dictionary`'s own `block_max`) to match
`AutomatonDictionary::bootstrap`'s real call shape.

**Finding 0, and the one that mattered most: the earlier `bulk_insert`
measurement's "1.1x slower" was contaminated by a shared cost, badly
enough to reverse the sign.** Isolated wall-clock on `build_sorted`
alone (no `BatchOrder::prepare` sort, no dictionary wrapping) first
*appeared* to show `Automaton::build_sorted` faster than `Block::build`
-- then a repeat run showed the opposite. Five-repetition `perf stat`
settled it with tight error bars (instructions: automaton 332.9M ±0.24%,
`Block` 314.8M ±0.89%; cycles noisier, ±3.7-4.4%, as this project's own
measurement-hygiene notes already warned): `Automaton::build_sorted`
alone is genuinely slower, ~1.06-1.16x depending on metric -- a real gap,
but far smaller than `bulk_insert`'s reported 1.1x once the shared sort
is correctly excluded on *both* sides. The lesson generalizes: even a
"whole-process, no dictionary layer" benchmark isn't automatically clean
if it does its own setup work inside the same process `perf` samples --
this crate's own `setup`/no-`setup` subtraction discipline needed
applying one level deeper than it previously had been.

**Finding 1: `perf record`'s flat profile was *also* lying, for the same
reason, and `--delay` fixed it properly.** A first `perf record` on the
isolated probe still showed `__memcmp_avx2_movbe` as the single largest
cost (~20-25%) -- but a `--call-graph fp` caller trace (`--call-graph
dwarf` couldn't unwind through glibc's assembly at all, no CFI there)
traced it entirely to the probe's *own* setup-phase `Vec<String>` sort,
not to construction. Confirmed by re-recording with `perf record -D
<ms>` (a startup delay long enough to skip past setup's own measured
duration entirely, timed from `setup` mode's own wall-clock): with setup
genuinely excluded from the recording window, `__memcmp_avx2_movbe`
disappears from `Automaton::build_sorted`'s top-20 entirely. Every
earlier profile in this whole multi-session effort that showed prominent
`memcmp` during a "bulk"/"build" operation was measuring the *input sort*,
not string comparison inside construction itself -- worth flagging as a
standing methodological trap for any future profiling of a batch/bulk
path in this crate.

**Finding 2: with the confound removed, the true `build_sorted` profile**
(construction only, `perf record -D`-isolated, self time):

| cost | automaton | note |
|---|---|---|
| `HashMap::get::<Signature>` (register lookup) | 14.52% | `FxHasher`-backed now (see the prior section) |
| `release` | 9.42% | refcount decrement / cascading free |
| `build_sorted` itself | 8.95% | main loop, `common_prefix_len`, path bookkeeping |
| `confirm_path` | 7.95% | per-term confirmation driver |
| `alloc_state` | 6.16% | state slab allocation/reuse |
| glibc allocator family (`_int_malloc`, `_int_free_*`, `cfree`, `unlink_chunk`, `malloc`, `realloc`, `_int_realloc`) | **~27.2% combined** | no single symbol dominates, but the *category* does |
| `finalize_outputs_and_count` | 3.30% | already rewritten to a direct loop (prior section) |
| `redirect` | 2.29% | repoint a transition to its canonical target |

And `Block::build`'s, same methodology, for a like-for-like comparison:

| cost | block | note |
|---|---|---|
| `UnfinishedNodes::find_common_prefix_and_set_output` | 14.04% | `fst`'s own construction-loop equivalent |
| `Builder::insert` | 10.38% | `fst`'s own per-term driver |
| glibc allocator family | **~22.5% combined** | comparable magnitude to automaton's -- not automaton-specific |
| `__memcmp_avx2_movbe` | 4.69% | genuinely inside `fst` here (registry/node equality), unlike the earlier false positive |
| `Registry::new` / `Registry::entry` | 4.13% / 3.66% | `fst`'s own register-deduplication -- the same algorithmic role as `Signature`/`register` |

**The two implementations' cost profiles are structurally similar**: both
spend a comparable share on allocator overhead (~22-27%, not something
distinguishing the two) and a comparable share on their own
register/dedup mechanism (`HashMap::get::<Signature>` at 14.52% vs.
`find_common_prefix_and_set_output` + `Registry::*` together at roughly
the same order). There is no longer one dominant, obviously-fixable
outlier on the automaton side -- this is now a "many small constant
factors" gap, not a "one big lever" gap.

**Finding 3, from `perf stat`'s deeper counters (`stalled-cycles-frontend`,
`L1-dcache-load-misses/loads`, delta-corrected via the `setup`/`build`
subtraction): the remaining ~1.30x cycle gap is frontend-stall-dominated,
not data-cache-bound.** Automaton pays ~2.85x more frontend stall
cycles than `Block::build` (6.99M vs. 2.45M) despite *better* L1 data
cache behavior: a 0.22% load-miss rate vs. `Block`'s 2.58% (automaton
does more total loads -- 63.8M vs. 52.8M -- but they hit cache far more
reliably). Ruled out as the sole explanation, not confirmed: a
same-crate-inlining-fragmentation hypothesis (`codegen-units=1`, no LTO,
tested via a `CARGO_PROFILE_RELEASE_*` env override, never touching the
committed `Cargo.toml`) changed *nothing* for either side (automaton
52.3M -> 52.85M cycles delta, `Block` 40.3M -> 39.7M -- both within
noise), meaning `alloc_state`/`release`/`redirect`/etc. are already
inlined adequately at the crate's current default `codegen-units=16`.
The frontend-stall cause itself is **not yet identified** -- reported as
an open question, not a solved one.

**Finding 4, a genuine surprise, reported as found rather than
cherry-picked: enabling LTO helps `Block::build` more than
`Automaton::build_sorted`, widening the relative gap.** Same env-override
experiment with `lto = "thin"` added: automaton's construction delta
dropped from 52.3M to 44.5M cycles (-15%), but `Block::build`'s dropped
from 40.3M to 29.5M (-27%) -- both genuinely faster in absolute terms,
but the *ratio* moved from 1.30x to 1.51x, the wrong direction for this
specific comparison. The likely explanation: `Block::build` calls into
the external `fst` crate on every operation, and without LTO none of
that ever got inlined across the crate boundary at all -- enabling it for
the first time unlocks a lot of previously-inaccessible headroom there.
`Automaton::build_sorted` is already same-crate, so it was never subject
to that specific limitation, and gained comparatively less. This is a
real, useful finding for the *project* (LTO looks like a genuine win
worth adopting for both implementations, and for `Dictionary` generally,
since it depends on `fst` too) but not a finding that closes automaton's
gap against `fst` specifically -- reported honestly rather than framed as
progress on the thing that was asked about.

**Not yet attempted, and the clearest remaining concrete lead**:
`Signature::of`'s `transitions: Vec<(u8, u32, StateId)>` and
`finalize_outputs_and_count`'s local `outputs: Vec<u32>` are each a
fresh heap allocation on *every confirmed state* -- exactly the ~27%
allocator-overhead category Finding 2 surfaced, and exactly the kind of
small, low-branching-factor-bounded (`CHUNK_CAP = 4`) data this crate's
own `Transitions` unrolled-list design already demonstrates the value of
avoiding heap allocation for. A `SmallVec<[T; 4]>`-style inline buffer
for both (a new small dependency, or a hand-rolled equivalent mirroring
`TransitionChunk`) is the most promising untried lever surfaced by this
analysis, but this session stopped at analysis for this specific item,
not implementation -- `Block::build` pays a comparable allocator tax
(~22.5%) via its own `Registry`, so this would need measuring against
`Block` again afterward to confirm it actually narrows the gap rather
than just making both sides faster in lockstep, the same trap Finding 4
already demonstrated is real.

### Custom incremental micro-FST: implementing the levers

Both levers the previous section surfaced were actually built and
measured -- one a confirmed negative result, reverted; one a confirmed
win, kept. Reported in that order because that's the order they were
tried, and the negative result is exactly as informative as a positive
one would have been.

**Lever 1: `SmallVec`-style inline buffers for `Signature::transitions`
and `finalize_outputs_and_count`'s `outputs` -- tried, measured slower,
reverted.** Built a hand-rolled `SmallVec<T, N>` (inline `[Option<T>; N]`
+ `Vec` overflow, no new dependency, matching `TransitionChunk`'s own
established shape), with a manual `PartialEq`/`Hash` giving exactly
`Vec<T>`'s own structural semantics (representation-independent: an
inline-only value and an overflowed value with the same logical content
compare equal and hash identically -- required, since `Signature` is a
`HashMap` key and two equal keys must always compare equal). Two direct
unit tests (`small_vec_push_get_len_across_the_inline_overflow_boundary`,
`small_vec_equality_and_hash_are_representation_independent`) confirmed
it correct before it went anywhere near `Signature`.

Wired in, then measured on the same isolated `perf_micro_fst_build`
`setup`/`build` delta this analysis already established as clean:
**instructions went *up*, not down** -- 146.9M -> 156.9M at `N=4`
(+6.8%), reproducible with `±0.00%` variance across repeated runs (not
noise). Tried `N=2` (156.7M, no real improvement) and `N=1` (160.9M,
*worse* -- a smaller inline capacity pushes more states into the
`overflow` `Vec` anyway, so it pays the abstraction's overhead *and*
still allocates often). No value of `N` tried came close to the plain
`Vec` baseline.

Why, reasoned through after the fact rather than before: `SmallVec` is a
bigger value to move around than a bare `Vec` (`[Option<(u8,u32,StateId)>;
N]` plus a `Vec` field, vs. a `Vec`'s three words), and every register
lookup constructs a `Signature`, compares it, and either drops it or
moves it into the map -- more bytes shuffled per lookup, and a
conditional branch (inline vs. overflow) on every element access via
`get`, where the old code's derived `Vec` comparison very likely compiled
to a tight, branchless loop (or even a real `memcmp`, for POD-ish `T`).
glibc's small-object allocator fast path, it turns out, was already
cheap enough that avoiding it wasn't worth what the avoidance itself
cost. Fully reverted (`Signature.transitions`/`finalize_outputs_and_count`'s
`outputs` are back to plain `Vec`, `set_outputs` back to `&[u32]`,
`SmallVec` and its tests deleted, not just unused) -- the code comments
at both sites now record this as a measured decision, not silence where
a reader might reasonably wonder why the "obvious" optimization from the
prior section never happened.

**Lever 2: thin LTO -- tried, measured faster, kept.** Added
`[profile.release] lto = "thin"` to `Cargo.toml`. Confirms the prior
section's Finding 4 was worth acting on. `codegen-units = 1` alone was
re-confirmed not worth adding on its own (no measured benefit, real
build-time cost) and left out.
*Correction, caught while measuring the next lever below*: this section
originally claimed the resulting isolated construction numbers "matched
the earlier experiment's numbers exactly" -- that was never actually
re-checked at the time, only asserted by analogy (the dictionary-level
`bulk_insert` numbers *were* checked and held up). They don't match:
`lto = "thin"` alone measures closer to `Block::build`'s *original*
no-LTO construction cost than to the earlier `codegen-units=1 +
lto=thin` combined experiment's ~27%-faster number -- `codegen-units=1`
apparently interacts with thin LTO's ability to inline across the `fst`
crate boundary (simplifying what ThinLTO has to merge) in a way it
doesn't deliver alone, an effect the original "codegen-units=1 changes
nothing" experiment (tested *without* LTO) didn't surface. Worth
revisiting -- `codegen-units=1` alone was correctly found not worth it,
but `codegen-units=1` *combined with* LTO was never isolated as its own
data point, and the combined-experiment's stronger number may well be
recoverable by adding it now. Not done in this pass; flagged, not fixed,
consistent with this whole investigation's own discipline about not
letting an unverified claim stand once noticed.

Checked at the full `AutomatonDictionary`-vs-`Dictionary` level too, not
just the isolated construction benchmark, since Finding 4 specifically
warned that a lever measured on one slice can shift a different slice's
ratio: `bulk_insert` (still automaton's one loss) moved from ~1.20x to
~1.16x slower -- unlike the isolated `build_sorted`-only benchmark, where
the same change widened the gap to 1.51x, the *dictionary-level* number
(diluted by `BatchOrder::prepare`'s shared sort and bootstrap overhead,
which LTO also speeds up on both sides) didn't get worse. `sequential_insert`
(2.32x fewer cycles), `get_id`, `get_term`, and `remove` (3.35x faster by
wall clock) all held steady or improved slightly. No regression found
anywhere `perf stat` was pointed at it. Safe for any future consumer of
this crate as a library dependency too: `[profile.*]` sections are only
read from the workspace root, never from a dependency's own
`Cargo.toml`, so this only affects builds where `s-dict` itself is what's
being built.

**Where this leaves the "implement the levers" instruction**: one lever
built, measured, and correctly discarded; one lever built, measured, and
kept. Net effect on the actual numbers that matter (the full
`AutomatonDictionary`-vs-`Dictionary` comparison) is a small further
improvement on top of the "closing the bulk-construction gap"
section's own results, not a dramatic one -- the extensive analysis
already found this specific investigation was past its biggest wins and
into "many small constant factors," and implementing what it found
confirmed that diagnosis rather than overturning it.

### Custom incremental micro-FST: a reused scratch buffer, not a `SmallVec`

A follow-up question after the `SmallVec` revert, prompted directly by
the user: instead of an inline-capacity type, what about a plain,
*reused* scratch buffer -- clear and refill the same `Vec` across calls
instead of allocating fresh each time? A materially different idea from
`SmallVec`, worth trying on its own merits rather than treated as
already answered by the previous negative result: `SmallVec` failed
because *its own abstraction* (a bigger struct to move, a branch on
every access) cost more than the allocation it avoided; a scratch buffer
adds no new abstraction at all -- still a plain `Vec`, just handed back
instead of dropped.

**Two different `Vec`s, two different situations**, worth being precise
about since only one of them is simple:

- `finalize_outputs_and_count`'s local `outputs: Vec<u32>` never
  escapes the function (read once by `set_outputs`, then unused) -- a
  trivial case: thread a `&mut Vec<u32>` in, `.clear()` it, reuse it.
  No complications.
- `Signature`'s `transitions: Vec<(u8, u32, StateId)>` is different: it
  becomes a `HashMap` key. On a register *hit* (the common case for a
  corpus with real structural sharing -- minimization's whole point),
  the `Signature` is dropped right after the lookup, and that allocation
  was pure waste all along. On a *miss*, the `Vec` has to be moved into
  the map and live there permanently -- nothing to reuse. The fix:
  `signature_of_into(state, scratch)` takes the buffer via `mem::take`
  (not a fresh allocation) to build the candidate `Signature`; the
  caller (`confirm_path`, and `release`, which has its own separate
  `Signature::of` call this investigation hadn't previously accounted
  for -- see below) gives the buffer back (`*scratch = sig.transitions`)
  on a hit, and simply lets a fresh one grow on the next miss. No
  `Borrow`/`raw_entry` trickery needed (the approach this project
  considered and set aside earlier as too fragile for stable Rust
  without `hashbrown`) -- ownership only ever moves one way, into the
  map, and never needs to be borrowed back out.

**Threading the buffers through turned out to touch more call sites
than expected.** `confirm_path` was the obvious one, but `release`
(freeing a superseded state's slot, purging its stale register entry)
has its *own* `Signature`-building call, reachable from `redirect`,
which is itself called from `confirm_path` *and* from `insert`'s and
`insert_sorted`'s own copy-on-write clone loops. Getting real reuse
meant threading `outputs_scratch: &mut Vec<u32>` /
`signature_scratch: &mut Vec<(u8, u32, StateId)>` from each top-level
entry point (`build_sorted`, `insert`, `insert_sorted`, `remove`, each
declaring its own pair once) down through `confirm_path`, `redirect`,
and `release`'s own recursive cascade -- six call sites, three function
signatures. `redirect` and `confirm_path` both crossed clippy's
`too_many_arguments` threshold (8 params) as a result; `#[allow]`ed with
a comment rather than force a context-struct refactor into an unrelated
change. All 84 tests pass, including the minimality regression test and
the 4-way fuzz differential -- the give-back logic (particularly
`release`'s, where `sig` is *never* moved into the map, since both
`HashMap::get` and `HashMap::remove` take the key by reference, so the
buffer can be reclaimed unconditionally, not just on the "not found"
branch) needed to be exactly right, and this is the check that it was.

**Measured on the same isolated `perf_micro_fst_build` benchmark,
instructions as the primary metric** -- cycles turned out unusually
noisy in this round specifically (single-run deltas swinging between
1.18x and 1.32x across repeated measurements of the *same* binary, both
sides moving together, clearly machine noise rather than a real change;
instructions stayed exactly reproducible, `±0.00%`, across every run):
automaton's isolated construction delta went from 146.9M instructions
(the "implementing the levers" section's post-LTO, pre-scratch-buffer
baseline) to **116.8M** -- and `Block::build`'s own delta, unchanged
code, measured at 118.6M in the same runs. **Automaton's construction
now uses very slightly *fewer* instructions than `Block::build`** (ratio
~0.985x), a reversal from every previous measurement in this whole
investigation, where automaton's construction cost was always the
slower side by some margin (1.06x-1.95x, depending on what else was
fixed at the time).

At the full `AutomatonDictionary`-vs-`Dictionary` `bulk_insert` level
(which also carries `BatchOrder::prepare`'s shared sort and dictionary
bootstrap overhead, diluting a pure-construction win somewhat, the same
effect noted for LTO above): instructions ratio improved from ~1.20x to
**1.041x** (automaton 288.98M vs. `fst` 277.47M for 80,000 terms) --
automaton's one remaining loss among the five measured dictionary
operations is now nearly closed, not just improved. `cycles` at this
level (automaton 134.26M vs. `fst` 124.06M, ratio 1.08x) agrees on
direction and magnitude, for once landing close to the instructions
ratio rather than diverging from it the way the pure-isolated-construction
benchmark's cycles did. All 84 tests still pass, clippy clean.

This is the strongest single result in the whole "optimizing
`build_sorted`" line of investigation, and it came from correctly
separating two questions the `SmallVec` attempt had conflated: *avoiding
an allocation* (worth doing, confirmed here) is not the same lever as
*avoiding it via an inline-capacity abstraction* (not worth it, per the
prior section) -- the same allocation, avoided a structurally simpler
way, paid off where the fancier version didn't.

### Custom incremental micro-FST: a fresh allocation-pattern audit

Requested explicitly: re-examine both algorithm and memory, `perf` as a
guide, hunting for two specific shapes -- (1) `A` copied to an
unnecessary intermediate `B`, tested, then moved to `C` (`B` never
needed to exist), and (2) a chain `T1(A) -> B1 -> T2(B1) -> B2 -> ... ->
C` that could be one composed transform straight from `A` to `C`. Found
via fresh `perf record` profiles of `sequential_insert` and `remove`
(both properly isolated from setup via `perf record -D`, this
investigation's own established discipline), not by re-reading code
first and guessing what to profile.

**Finding 1: `clone_state`'s `inherited_targets` -- the exact
combinator-chain anti-pattern already fixed at three other call sites
(`Signature::of`, `finalize_outputs_and_count`), missed at this one.**
`content.transitions.iter().map(|(_, _, t)| t).collect()` showed up as
a top-6 cost in `sequential_insert`'s flat profile (`Vec<StateId> as
SpecFromIterNested<..., clone_state::{closure#0}>::from_iter`, ~4.9%).
Same fix as before: a direct loop over `Transitions::at`. This one still
needs an owned `Vec` (the source `content` is moved into `alloc_state`
right after, so the targets must be captured first) -- not
type (1) or (2) above, just the same iterator-overhead lesson landing a
fourth time.

**Finding 2: `release`'s `children: Vec<StateId>` -- genuinely
type (1), an unnecessary `B`.** `states[slot].transitions.iter().map(..
).collect()` built an owned `Vec` of child ids purely so the subsequent
loop could call `release` on each one. But `release` only ever borrows
`states` *immutably* through its entire (recursive) call tree -- it
never mutates state content, only `refcounts`/`free_slots`/`register` --
so `states[slot]`'s transitions can't change out from under a direct
loop, and indexing `states[slot].transitions.at(i)` while also passing
`states` into the recursive call is just two simultaneous immutable
borrows, always legal. Rewritten to iterate directly with no
intermediate `Vec` at all: `A` (the live transitions) feeds `C` (the
recursive `release` calls) with no `B` in between.

**Finding 3, the largest single win of this audit: `min_key`/directory-key
recomputed on every mutation, needed on only a small fraction of them.**
`AutomatonBlock::merge_into` set `self.min_key = self.term_at(0)...`
unconditionally whenever anything was added -- a full root-restart
`term_at_into` walk (not the shared-prefix `stream()` path) plus a fresh
`String` allocation, on *every single insert*, even though `min_key`
only actually changes when rank 0 itself moved. The merge walk already
computes exactly that signal (`unchanged_prefix > 0` means rank 0 is
among the leading entries proven unmoved) and was discarding it right
after computing it. Same pattern, same fix, in three more places:
`AutomatonBlock::remove` (recomputed unless `old_rank == 0`) and all
three `AutomatonDictionary` call sites that copy a block's `min_key`
into the directory (`insert`, `insert_ordered`, `remove` -- consolidated
into one `refresh_directory_key` helper, since `Box<str>::from(&str)`
is a real allocation-and-copy every time too, and all three were paying
it unconditionally). Not exactly the "A/B/C" shape either -- more a
redundant-recomputation cousin of it, discarding a signal the algorithm
already had rather than threading data through an unnecessary buffer --
but the same underlying discipline (don't do work whose result you
already know) caught it.

**Measured**, `perf stat`, both wall-clock and hardware counters (fresh
5-rep runs, `perf_dictionary_automaton_vs_fst`):

| operation | before this audit | after | `perf stat` confirmation |
|---|---|---|---|
| `remove` | 3.15-3.35x faster than `fst` | **6.36x faster** (wall clock); **4.24x fewer cycles, 4.74x fewer instructions, 6.21x fewer cache-references** | yes |
| `sequential_insert` | 2.32-2.40x faster | **2.73x faster**, instructions down 15.1% from immediately before this audit | yes |
| `bulk_insert` | ~1.04-1.16x slower | unaffected (this audit's fixes are all on the mutate-existing-block path, not fresh construction) -- still ~1.04x slower, near parity | consistent |

`remove`'s jump is the standout: it exercises `release`'s cascading-free
path (Finding 2) and recomputes a block's `min_key` on every removal
(Finding 3) far more heavily than insert does, so both fixes compounded
specifically there. A post-fix `perf record` of both `remove` and
`sequential_insert` confirms the combinator-chain fingerprints are gone
from the flat profile entirely -- what remains (register `HashMap`
lookups, the merge walk itself, `confirm_path`/`alloc_state`/`clone_state`'s
now-fixed core loop) is legitimate algorithm cost, not a pattern this
audit's two shapes describe. All 99 tests pass (91 lib + 5 + 3 across
`tests/`), including the 4-way fuzz differential -- which would have
caught a wrong `min_key`/directory-key skip condition, since it checks
routing correctness after every mutation, not just decode correctness.

**Also found, not fixed here, explicitly out of scope**: this session's
review surfaced a substantial amount of code added outside this
conversation between turns -- `get_term_into`/`get_terms_into`/
`get_terms_scattered_into`/`get_terms_seeked_into`/`size_bytes` on
`AutomatonDictionary` (and matching `get_terms_scattered_into` additions
to `Dictionary`/`FrontCodedDictionary`), plus `Automaton::seek_stream`/
`size_bytes` and `AutomatonBlock::payload_bytes`/`decode_sorted_ranks_into`
in support. Read in full and checked against this audit's two anti-pattern
shapes specifically: no unnecessary intermediate buffer found in any of
the three bulk-decode strategies (`get_terms_seeked_into`/
`decode_sorted_ranks_into` in particular already writes straight from
the stream into the caller's `out`, no arena in between) -- this code
already appears to have internalized the same discipline. Its own
`s-bench`/`dict_bench`-based comparison (referenced in its doc comments,
external to this repo) is outside what this session can measure or
weigh in on.

### Custom incremental micro-FST: the memory-scaling problem at real corpus scale

An external benchmark (`s-bench`/`dict_bench`, outside this repo, using
real RDF data rather than this project's own synthetic generators) found
something this session's own measurements never surfaced: at 524,299
distinct terms, `AutomatonDictionary` retains 164.89 MiB (~330 bytes/term)
against `Dictionary`'s 15.94 MiB (~32 bytes/term, matching a compact FST's
expected footprint); at 4,579,699 terms, `Dictionary` stays flat (~32.7
bytes/term) but `AutomatonDictionary` grows to ~848 bytes/term -- a 2.6x
*increase* in per-term cost as the corpus grew 10x, not the flat or
sublinear scaling either implementation's design intends. Extrapolated to
the full ~33M-term corpus this benchmark's source data represents, the
automaton side would plausibly need tens of GB against `fst`'s ~1-1.3GB,
a real OOM risk on the 14GB machine this was tested on -- not attempted,
correctly, rather than risk it.

**First question: is this slot-reclamation garbage (the known,
already-addressed gap from earlier in this file), or the true minimal
structure itself?** The benchmark's own numbers already answered this --
growth showed up even in `size_bytes()` (both `Dictionary`'s and
`AutomatonDictionary`'s versions walk only *reachable* state, explicitly
excluding `register`/`refcounts`/`free_slots`/generation bookkeeping on
the automaton side, exactly the register-cleanup discipline this file's
"reference-counted slot reclamation" section already built and verified).
So this isn't stale garbage -- the true minimal automaton itself is
larger per term at scale than it should be.

**Reproduced in-repo, and the reproduction is itself the key finding.**
New `examples/perf_scale_memory.rs`, `AutomatonDictionary`/`Dictionary`'s
own `size_bytes()`, checkpointed across an increasing insert via
`insert_batch` (a realistic bulk-load shape). This project's own
namespace-clustered synthetic generator (highly repetitive: every term
shares one of four long, fixed namespace prefixes, differing only in a
short numeric suffix) reproduced *none* of this: 22.2 -> 26.3 bytes/term
across 52K -> 524K terms, `Dictionary`-relative ratio essentially flat at
~1.2x, the growth curve visibly *flattening*, not accelerating. That
result, by itself, located the cause: it isn't scale, it's **redundancy**.
A second generator (`hashlike`: a short shared prefix, `urn:uuid:`,
followed by 16 hex digits derived from `splitmix64(index)` -- no
cross-term structure beyond the last character or two, approximating
hash-based identifiers/blank node labels/UUIDs real RDF data has plenty
of) told the real story: at the *same* term count (100K), `Dictionary`
stayed at ~30.6 bytes/term (nearly identical to the clustered case --
`fst`'s byte-packed encoding is close to redundancy-invariant), while
`AutomatonDictionary` used **767-811 bytes/term -- ~25x worse than
`fst`**, against only ~1.2x worse for highly-redundant content. Real KG
corpora have a long-tail, Zipfian mix of highly-repetitive vocabulary
terms and low-redundancy identifiers/literals; as more data loads, the
weighted-average redundancy per block plausibly drops, which is
consistent with -- not just superficially similar to -- the external
benchmark's observed growth curve, even though this repo's own
`insert_batch` order/content differs from the real corpus load order.

**Root cause, confirmed by direct measurement, not inference from the
ratio alone**: `std::mem::size_of::<State>() == 72` bytes (verified via a
new diagnostic test, `micro_fst::tests::print_struct_sizes`) -- and
every one of those 72 bytes is paid **per state, regardless of actual
branching factor**, because `Transitions` holds one `TransitionChunk`
*by value*, inline, with `CHUNK_CAP = 4` fixed-size slots
(`bytes: [u8;4]`, `outputs: [u32;4]`, `targets: [Option<StateId>;4]`),
even for the extremely common case (the corpus's own low-branching-factor
premise this whole design leans on) of a state with exactly one
transition. For the `hashlike` case, 767-811 bytes/term / 72 bytes/state
implies roughly 10-11 essentially-unshareable states per term -- matching
a 16-hex-digit suffix where only the last character or two has enough
collision probability (1-in-16, 1-in-256) to ever get merged by
minimization, and every other position is, correctly, its own unique
state. `fst`'s underlying representation is a variable-length, byte-packed
encoding (a `Node`'s size scales with its actual transition count and
output magnitude, not a fixed Rust struct) -- it has no equivalent fixed
floor, which is exactly why it stays flat under low redundancy while this
representation doesn't.

**Tuning `CHUNK_CAP` down: tried, a real but incomplete and costly
mitigation, not shipped.** If most states only use 1 of 4 slots under low
redundancy, shrinking the inline capacity should shrink the fixed
per-state cost proportionally. Tested by direct measurement (temporarily
editing the constant, full test suite re-run clean at every value, wall
clock and `size_bytes()` both re-measured, then reverted -- the same
"try it, measure it" discipline as every other lever in this file, not
assumed from the size-formula alone):

| `CHUNK_CAP` | `size_of::<State>()` | hashlike bytes/term | clustered bytes/term | `remove` | `get_id` | `get_term` | `bulk_insert` |
|---|---|---|---|---|---|---|---|
| 4 (shipped) | 72 | 767-811 | 22.2-26.3 | 6.25us/op | 227ns/op | 251ns/op | 359ns/op |
| 2 | ~28 (not printed; derived) | 603-623 (**worse than CHUNK_CAP=1**) | 20.5-23.2 | 10.45us/op (1.67x slower) | 362ns/op (1.59x slower) | 371ns/op (1.48x slower) | 585ns/op (1.63x slower) |
| 1 | 48 | **524-554** (-32%) | **19.8-22.9** (-13%) | 10.66us/op (**1.70x slower**) | 351ns/op (**1.55x slower**) | 391ns/op (**1.56x slower**) | 672ns/op (**1.87x slower**) |

All values held up under the crate's full test suite (92+5+5 tests,
including the 4-way fuzz differential and the minimality regression
test) at every `CHUNK_CAP` tried -- correctness was never in question,
only the memory/speed tradeoff. Two things stand out:

1. **`CHUNK_CAP=2` is strictly worse than `CHUNK_CAP=1`** on memory
   (603-623 vs. 524-554 bytes/term for `hashlike`) while offering no
   real speed advantage over `CHUNK_CAP=1` either -- not a clean
   monotonic tradeoff curve. Overflow is chunked in `CHUNK_CAP`-sized
   pieces too (`overflow: Vec<TransitionChunk>`), so a state needing
   just one more transition than its inline capacity still allocates a
   whole extra `CHUNK_CAP`-sized chunk -- at `CHUNK_CAP=2`, a
   3-transition state (common near a block's root, where namespace
   diversity peaks) wastes half of that overflow chunk; at
   `CHUNK_CAP=1`, every state past 1 transition already expected to pay
   for an overflow allocation, so there's no equivalent waste to avoid.
2. **Even the best case (`CHUNK_CAP=1`) doesn't close the gap.** 524-554
   bytes/term is still ~17x worse than `fst`'s ~30 for the same
   low-redundancy content, for a 1.5-1.9x slowdown on every read/write
   operation measured. This is a mitigation, not a fix -- reverted, not
   shipped, since a 32% memory reduction that costs the exact
   performance advantage this whole custom-automaton effort was built to
   deliver isn't a net win on the numbers alone, and this session isn't
   the place to unilaterally decide that tradeoff for the project.

**Conclusion, stated plainly**: this is a genuine, now-diagnosed
architectural limitation, not a bug with a clean fix. A fixed-size Rust
struct per automaton state is fundamentally less memory-efficient than a
variable-length byte-packed encoding for low-redundancy content, and no
`CHUNK_CAP` value closes that gap without giving back most of the speed
advantage that's this whole implementation's reason to exist. Closing it
for real would need a structurally different state representation (a
genuinely variable-length/byte-packed encoding, or a different strategy
specifically for low-branching leaf runs) -- a bigger redesign than a
tuning knob, and a decision about whether `AutomatonDictionary` is
viable at all for corpora with a meaningful share of low-redundancy
content (which most real RDF data has) that's worth making deliberately,
not as a side effect of chasing one more lever. `examples/
perf_scale_memory.rs` and `micro_fst::tests::print_struct_sizes` are
kept, not reverted -- they're the reusable tools for whatever that
decision needs next, not single-use diagnostics.

### Custom incremental micro-FST: a byte-packed representation, prompted by `s-graph`

A design discussion, not a profiling session: the user pointed at
`../s-graph/examples/common/radixtree_dfs_interleaved_varskip_unrolled.rs`,
which stores a whole radix trie as one flat, byte-packed stream (no
per-node struct, no per-node allocation), split into independently
allocated fixed-capacity blocks specifically so single-record insertion
costs O(block capacity), not O(structure size). The one adaptation this
crate's problem needs that theirs doesn't: their structure is a plain
tree (one parent per node, so DFS-preorder addressing works); this
crate's automaton is a DAG (register-based minimization means one state
can have several parents), so a state needs an address independent of
any single path to it -- not a blocker, since `StateId` was already an
opaque handle, not a physical position, before any of this.

**Cheap validation first, per an explicit staging decision** (asked via
`AskUserQuestion`, "prototype size savings first" over "build the full
thing" or "flat Vec, add blocks only if needed"): `Automaton::
estimate_packed_bytes`, a pure measurement -- walks the *existing*
reachable structure and computes what a byte-packed, variable-length
record per state (`[flag_count: u8][per-transition: byte + LEB128
output + 4-byte address]`) would cost, without touching any storage,
addressing, or mutation code at all. Result, far bigger than anything
the `CHUNK_CAP`/`SmallVec` tuning found:

| scenario | current `size_bytes()` | packed estimate | reduction | vs. `fst` |
|---|---|---|---|---|
| hashlike (low redundancy) | 767-811 bytes/term | ~91-95 bytes/term | ~88% | 25x worse -> ~3x worse |
| clustered (high redundancy) | 26.1-26.3 bytes/term | ~14.5 bytes/term | ~45% | 1.2x worse -> automaton *wins* (~14.5 vs. `fst`'s ~30) |

**Read-only decode prototype, built to test the other half of the
tradeoff** (the estimate says nothing about speed): `PackedAutomaton`
(`micro_fst.rs`) -- `compile(&Automaton)` writes the same layout the
estimate assumed via a postorder walk (every child written before its
parent, so a parent's transitions can reference already-known
addresses; naturally preserves the source automaton's own
deduplication, since a state visited via a second parent just returns
its already-recorded address), then serves `rank`/`term_at_into` by
parsing records instead of indexing typed arrays. No insert, no remove,
no register, no refcounting -- deliberately out of scope; that's the
"unrolled blocks" half of the `s-graph`-prompted idea, not attempted
here.

Correctness before speed, as always: three new tests
(`packed_automaton_matches_automaton_rank_and_unrank`, 8,000
namespace-clustered terms; `..._on_branching_and_prefix_terms`, a
hand-traceable case with multi-child and final-with-children states;
`..._size_matches_the_earlier_estimate`, confirming `compile(..).
size_bytes()` exactly equals what `estimate_packed_bytes` predicted for
the same automaton) all pass before any timing was trusted.

**Decode speed: a real, `perf stat`-confirmed win, but it took finding
and fixing a specific problem to get there, not just building the
prototype and measuring once.** First pass, wall-clock: `rank` ~1.5-1.65x
faster on a 20,000-term hashlike corpus, but `term_at_into` showed *no*
improvement at all -- an asymmetry between two operations that walk the
exact same states, which shouldn't happen if the representation itself
were the only variable. A `perf record` profile explained it: `get`
(called from `rank`) had been inlined by LLVM by default; the
structurally near-identical `last_output_at_most` (called from
`term_at_into`) hadn't, paying a real function-call per visited state
that `rank`'s path didn't. Plain `#[inline]` (a hint) didn't change
LLVM's decision -- confirmed by re-profiling, not assumed fixed;
`#[inline(always)]` did, confirmed the same way. After that fix, both
operations show a real win, `perf stat`-confirmed on both the hashlike
and clustered corpora:

| metric | hashlike | clustered |
|---|---|---|
| cycles | 1.17x fewer | 1.17x fewer |
| instructions | 1.81x *more* | 1.81x *more* |
| cache-references | 1.97x fewer | 1.95x fewer |
| cache-misses | 1.44x fewer | 1.42x fewer |

More instructions (the parsing overhead a variable-length record pays
that direct array indexing doesn't) but far fewer cache references and
misses, netting out to fewer total cycles either way -- IPC roughly
doubles (1.4 -> 3.0 on the hashlike corpus), exactly the "more work per
cycle because the data actually fits near the CPU" signature this whole
line of investigation was betting on. This is the first time in this
entire memory-scaling investigation that a change has won on *both*
axes at once, rather than trading one for the other.

**What this does and doesn't establish.** Confirmed: the byte-packed
layout is both smaller and faster to *decode* than the current
`State`/`TransitionChunk` representation, for both tested redundancy
regimes. Not yet touched at all: whether it can be made *mutable*
(insert/remove) without giving back this win -- the entire reason
`s-graph`'s own design needs unrolled blocks in the first place, and the
next real step if this is worth pursuing further, not a small addition
to this prototype.

## the mutable unrolled-block version: a real insert, and a real problem it exposed

Extended `PackedAutomaton` in place (same type, not a new one) into the
mutable structure the decode-only prototype above deliberately deferred:
`blocks: Vec<Vec<u8>>` (fixed `BLOCK_CAPACITY = 4096` per block,
`s-graph`'s own default) addressed by a single virtual `u32`
(`block_idx * BLOCK_CAPACITY + offset`), a persistent `register:
HashMap<PackedSignature, u32>` mirroring the slab design's own register
but keyed by packed addresses instead of `StateId`, and `insert(&mut
self, term: &str)` mirroring `Automaton::insert`'s own algorithm exactly
(walk existing transitions as far as they match, "open" every matched
state into an editable `StagingFrame` by parsing its existing record,
extend with fresh empty frames for the unmatched suffix, confirm
bottom-up -- recomputing each frame's outputs via the same
prefix-sum-of-sibling-subtree-counts technique, deduping through the
register, and splicing the resolved `(byte, output, pos)` into the
parent frame).

One real simplification fell out of the DAG-vs-tree difference between
this and `s-graph`'s own structure, reasoned through before writing any
code: `s-graph`'s addressing is DFS-preorder position *in a tree*, which
is exactly why it needs shift-or-split-on-insert complexity (a node's
position can change as siblings are added). This crate's states are
addressed by explicit stored pointers (already true of `StateId` in the
slab design) and confirmation is always bottom-up -- a child's address
is permanent before its parent's record ever references it -- so a
confirmed record's address here never needs to move. That removed the
need for any shifting, splitting-to-make-room, or
reference-counting/redirect machinery at all: a superseded record is
just abandoned, unreachable bytes.

The record format needed one real addition the read-only version didn't:
an explicit `subtree_count` (LEB128 varint, right after the flag/count
byte) on every record, because `insert`'s output recomputation needs
each child's subtree size and there's no cheap way to recover that from
the read-only layout without re-walking whole subtrees.

**Correctness, confirmed before anything else.** Four new differential
tests: `packed_automaton_insert_matches_automaton_incremental_insert`
(interleaved, non-sorted insertion order -- `insert` has no
sorted-input assumption, unlike `Automaton::build_sorted`),
`..._insert_is_idempotent`, and
`..._insert_reaches_same_minimality_as_fresh_build` (this project's
existing minimality-regression pattern, applied here: after a
deterministically-reordered incremental build of 600 terms, the
*reachable* record count -- not `size_bytes()`, see below -- must equal
a fresh `Automaton::build_sorted` over the same final set; it does).
Building this surfaced and fixed an unrelated latent bug in an
*existing* test, not in `Automaton` itself:
`packed_automaton_size_matches_the_earlier_estimate` was feeding
`Automaton::build_sorted` unsorted, undeduplicated input, violating its
documented precondition. That produced a genuinely corrupted automaton
(reproduced directly: the root state held 900 transition entries for
only 3 distinct bytes -- 300 duplicate `'h'`/`'u'`/`'_'` entries each,
one extra per unsorted "run"), invisible before because the only
assertion was self-consistency between `compile` and `estimate`
(equally inflated on both sides) and because the *old* format's
`debug_assert!` on transition count compiles away entirely under
`cargo test --release`, this project's standard test invocation. Fixed
by sorting+deduplicating first, matching every other call site's
existing (correct) convention.

**Insert speed: real, `perf stat`-confirmed, and a genuine tradeoff, not
a second free win.** Same harness as `perf_probe_insert_cost_automaton`
(160-term baseline, matching `block_max`'s midpoint; 20,000 single-term
inserts against a freshly-cloned baseline each time, isolated from
`Block`'s and `Automaton`'s own probes so `perf stat` attributes cleanly
to one operation):

| implementation | time/op | cycles | instructions | cache-refs | cache-misses |
|---|---|---|---|---|---|
| `Block` (decode+merge+rebuild via `fst`) | 54.7us | 2,625M | 8,120M | 68.7M | 279K |
| `Automaton::insert` (mutable slab) | 8.83us | 508M | 1,441M | 7.80M | 60.3K |
| `PackedAutomaton::insert` (mutable unrolled blocks) | 14.3us | 788M | 2,493M | 2.63M | 69.9K |

`PackedAutomaton::insert` is ~3.3x faster than a full `fst` rebuild (the
thing blocks exist to avoid paying on every mutation) but ~1.55x
*slower* than the plain slab `Automaton::insert` it's meant to replace --
parsing each matched state's record back into a `StagingFrame` and
re-serializing a brand-new record for every state on the path (vs. the
slab's direct in-place pointer/field mutation) costs real cycles that
the decode path's win doesn't offset here. Fewer cache-references than
the slab version (2.63M vs 7.80M -- the packed layout's locality
advantage shows up on the write path too) but more cycles and
instructions overall: parsing/serialization cost dominates over cache
effects for this operation, the reverse of what decode showed.

**Memory under sustained inserts: measured, not assumed, and a real
regression -- the significant negative finding of this whole
extension.** The design reasoning above (see `PackedAutomaton`'s own
struct doc comment, corrected after this measurement) assumed
unreclaimed dead records were a small cost, bounded by how much churn
one `AutomatonBlock` sees between its own periodic full rebuilds
(`split`/`merge_with_neighbor`). That assumption was never measured
before being written down, and it's wrong: because inserting *any* term
changes every ancestor's `subtree_count` on the root-to-leaf path, the
register can essentially never reuse an old copy of any state on that
path -- every single insert writes a brand-new record for every state
from the new leaf up to and including the root. `measure_packed_
automaton_dead_bytes_under_sustained_inserts` (800-term namespace-
clustered baseline, insert-only churn, comparing `size_bytes()` against
a *fresh* `Automaton::build_sorted` over the same live set's
`estimate_packed_bytes()` at each checkpoint):

| inserts | live terms | size_bytes | fresh estimate | ratio |
|---|---|---|---|---|
| 0 | 800 | 1,338 | 1,338 | 1.00x |
| 144 | 944 | 48,841 | 1,814 | 26.9x |
| 192 | 992 | 67,006 | 1,835 | 36.5x |
| 200 | 1,000 | 70,142 | 1,883 | 37.3x |
| 2,000 | 2,800 | 690,235 | 2,105 | 327.9x |

192 inserts is comfortably *inside* one block's normal churn between
splits at this project's default `block_max` (256) -- not an extreme,
long-run stress case. The bloat is already 20-37x by then, and it keeps
climbing roughly linearly with insert count, not leveling off: this
representation, exactly as built, would make an `AutomatonBlock` using
it dramatically *larger* under real insert traffic than either the
current `fst`-backed `Block` or the existing mutable-slab `Automaton`
(which reclaims old slots via refcounting/`free_slots`, keeping its
slab-vs-reachable gap bounded -- see the sustained-churn measurement
earlier in this document). Confirmed as a real property of this
specific design (bottom-up confirmation touching the whole path, no
reclamation at all), not a bug in the implementation.

**Decode, re-measured -- the earlier "unambiguous win" claim above
didn't survive making the structure mutable, and shouldn't have been
assumed to without checking.** The mutable record format adds one field
the read-only prototype didn't have (`subtree_count`, a LEB128 varint
right after the flag/count byte, needed by `insert`'s output
recomputation) -- parsed on *every* state visited by `rank`/
`term_at_into` too, even though decode never uses it. Re-running the
same `perf_probe_decode_cost_packed_automaton(_hashlike)` probes against
this mutable version, `perf stat`-confirmed and reproduced twice:

| metric | hashlike | clustered |
|---|---|---|
| cycles | 1.40-1.46x *more* | 1.42x *more* |
| instructions | 2.67x more | 2.63x more |
| cache-references | 1.73-1.75x fewer | 1.05x fewer |
| cache-misses | roughly flat | roughly flat |

A `perf record --call-graph fp` profile confirmed this isn't an
inlining regression -- `header`/`read_leb128` are still fully inlined
into `get`/`last_output_at_most`, same as before -- it's genuine added
work: every state visit now pays for parsing a varint decode never
needed, on top of the one it already paid per transition for that
transition's own output. Cache-references are still lower than the
plain slab `Automaton` (the layout's locality advantage is real and
persists), but that's no longer enough to net out to fewer cycles the
way it did for the read-only prototype -- decode is unambiguously worse
now, not better, than both the read-only prototype *and* the plain slab
`Automaton` it was meant to beat. Storing `subtree_count` out-of-line
(e.g. only in a separate lookup structure `insert` consults, not inline
in every record `rank`/`term_at_into` must skip past) would remove this
specific cost, but that's a real redesign, not measured or attempted
here -- reported as a further open question, not a fix already in hand.

**Isolating the actual cause, instead of assuming it was `subtree_count`.**
A discussion of the addressing scheme (positional/renumbered vs.
permanent-pointer addressing, prompted by working through a hand-traced
insert example) raised the right question: does this crate's design
even need `s-graph`-style unrolled blocks at all? `s-graph` needs
block-relative addressing because its structure is a plain tree, where
a node's address *is* its DFS-preorder position, so a block boundary
can force a shift-or-split. This crate's automaton is a DAG addressed
by explicit, permanent `u32`s -- nothing here is ever positionally
addressed, so nothing ever needs to shift, which means unrolled blocks
were bought to solve a problem (bounded-cost single-record insertion
without touching the rest of the buffer) this design already had for
free by construction. What they cost instead was a second,
bounds-checked pointer hop (`blocks[block_index]`, to find the byte
array to then index into) in front of every single field read -- and a
throwaway variant (identical record format, `Vec<u8>` instead of
`Vec<Vec<u8>>`, deleted after use) isolated that this was the dominant
cause of the decode regression above, not `subtree_count`: with
`subtree_count` kept but blocking removed, the throwaway variant showed
925M cycles on the hashlike decode probe (*fewer* than the plain slab
`Automaton`'s ~984-996M) and 44.3M on clustered (~4% more than
`Automaton`'s ~42.5M) -- a small remaining `subtree_count` tax, and the
blocking tax gone.

**Rebuilding `PackedAutomaton` on a flat `Vec<u8>`, and re-measuring for
real (not the isolated throwaway).** `blocks: Vec<Vec<u8>>` was removed
in favor of a plain `bytes: Vec<u8>` (ordinary amortized-growth `Vec`
reallocation, no capacity cap, no `decompose`); every read/write path
updated accordingly; `insert`'s algorithm itself unchanged. Full
`perf stat`, `perf_probe_decode_cost_packed_automaton(_hashlike)` and
`perf_probe_insert_cost_packed_automaton`, reproduced (3 repeated runs,
hashlike decode: 1,214-1,222M cycles, tight agreement -- not noise):

| metric | Automaton (slab) | blocked (prior round) | flat (this round) |
|---|---|---|---|
| decode cycles, hashlike | ~984-996M | ~1,410M (1.43x) | ~1,217M (1.22x) |
| decode cycles, clustered | ~42.5M | ~60.3M (1.41x) | ~51.2M (1.20x) |
| insert cycles, N=160 | 518M | 788M (1.52x) | 667M (1.29x) |

Real, `perf stat`-confirmed improvement on *both* decode and insert
(roughly 14-15% fewer cycles than the blocked version, on every
metric) -- but **not** the near-parity-or-better result the isolated
throwaway suggested. There's an unexplained gap between that
microbenchmark (925M/44.3M) and this integrated result (~1,217M/51.2M)
that a `perf record --call-graph fp` check didn't resolve: inlining is
identical either way (`header`/`read_leb128` fully folded into
`get`/`last_output_at_most`/`term_at_into`, no separate symbols in
either version). The most likely remaining explanation is that thin-LTO
codegen decisions for these hot functions are sensitive to how much
other code shares the struct/call graph (the real `PackedAutomaton` has
`insert`/`confirm_frame`/`append_record`/etc. in the same `impl` block
and many more call sites than the throwaway's one isolated probe), but
that's not confirmed -- reported as an open discrepancy, not papered
over with the more flattering number.

Insert-memory bloat is, as expected, **unaffected** by this change --
still ~20-37x a fresh rebuild's size within 128-200 inserts
(re-verified, identical numbers to the blocked version) -- because that
cost comes entirely from every ancestor's `subtree_count` genuinely
changing on every insert, which no addressing/container scheme removes.

**Where this leaves the byte-packed idea, updated again.** Decode and
insert speed: both real, `perf stat`-confirmed improvements from
dropping unrolled blocks, but the mutable version still loses to the
plain slab `Automaton` on both (~1.2-1.3x more cycles), not a win.
Insert memory: still the significant, unaddressed open problem --
unreclaimed dead records under sustained churn make this representation
unsuitable for the write-heavy role `AutomatonBlock` would need it for.
Closing that gap would need either (a) periodic in-block compaction (a
`compile`-style rewrite of just the live reachable records, real added
complexity, cost not measured here), or (b) genuine reclamation of
variable-length dead records (the harder problem the original design
deferred). Neither is implemented. A separate, not-yet-attempted size
idea also came out of this discussion -- storing `subtree_count`
out-of-line (a side table `insert` consults, decode never touches) to
remove its small remaining tax, and/or a hybrid inline-vs-addressed
transition encoding (inline a child's data directly when it provably
has one parent, address it only when shared) to shrink the format
itself -- both real candidates for a further round, neither built or
measured yet. The read-only prototype's original result (byte-packing
wins on decode) still stands on its own; the mutable version, even
after this correction, has not recovered it. Not wired into
`AutomatonBlock`/`AutomatonDictionary`, `#[allow(dead_code)]`
throughout, same as before.

## a methodology correction: `perf stat` was counting `compile()` as "decode"

The previous section's flat-buffer numbers were themselves measured
with a real, previously-unnoticed flaw, caught by asking a good
question about *why* an earlier isolated throwaway benchmark had come
in so much faster than the real, integrated `PackedAutomaton`. Every
"decode" comparison against `PackedAutomaton` this whole investigation
has used `perf stat -- <test binary> <probe>` -- but `perf stat` counts
the *entire process*, not just the `Instant`-timed region a probe
reports in its own wall-clock printout. Every decode probe calls
`PackedAutomaton::compile(&automaton)` once, before the timed loop, on
the reasoning that "it's outside the timed region, so it's not being
measured" -- true for the wall-clock number printed to the console,
false for the `perf stat` cycle count wrapping the whole run.
`Automaton`'s own decode probes never pay an equivalent cost (there's
no separate compile step -- `build_sorted` *is* the queryable
structure), so this was a one-sided tax: every `PackedAutomaton`-vs-
`Automaton` decode comparison in this document was actually measuring
"compile + decode" against "decode alone."

**Isolating it.** Three throwaway setup-only probes (deleted after
use): "terms + `build_sorted`" alone (`Automaton`'s true setup, no
compile), "+ real `compile`" (with the `register: HashMap<
PackedSignature, u32>` bookkeeping `insert` needs), and "+ a no-register
compile" (a version of `compile` that skips `PackedSignature`
construction and the register lookup/insert entirely, just a plain
postorder write memoized by original `StateId`) -- to separate `compile`'s
own cost from decode's, and to check how much of `compile`'s cost is
specifically the register:

| probe (hashlike, 20,000 terms) | cycles |
|---|---|
| terms + `build_sorted` only | ~320-334M |
| + real `compile` (with register) | ~652-729M |
| + no-register compile | ~443M |

So `compile`'s own cost is roughly 320-400M cycles for this corpus, and
the register specifically accounts for roughly half of that (~440M
without it vs ~650-730M with it -- a real, substantial, but *one-time,
compile-time* cost, not a per-query decode cost; it amortizes across
however many `rank`/`term_at_into` calls happen before the next
rebuild, and is negligible next to `insert`'s own probe, whose baseline
is 160 terms, not 20,000).

**The corrected, decode-only picture**, subtracting the matching
setup-only baseline from each full probe's `perf stat` number:

| corpus | `Automaton` decode-only | `PackedAutomaton` (flat) decode-only | ratio |
|---|---|---|---|
| hashlike (20,000 terms, low redundancy) | ~668-675M cycles | ~602-640M cycles | packed ~1.05-1.12x *fewer* |
| clustered (2,000 terms, high redundancy) | ~40.2M cycles | ~49.1M cycles | packed ~1.22x *more* |

This changes the conclusion, and it's corpus-dependent, not uniform.
On the low-redundancy corpus, where `compile`'s cost is large relative
to the rest of the run (many states, little sharing), correcting for it
flips the result: decode is at parity with, and plausibly a little
faster than, the plain slab `Automaton` -- consistent with, not
contradicting, the original read-only prototype's confirmed win. On the
redundant corpus, `compile` is cheap to begin with (heavy sharing means
few states to build), so this correction barely moves the number, and
the ~1.2x decode regression there is real, not a measurement artifact.
Every earlier "decode regression" figure in this document for the
*blocked* version (now-deleted code, not re-measured under this
corrected methodology) is understood to have the same one-sided bias
and is likely somewhat overstated in magnitude, though not necessarily
wrong in direction -- not re-verified, since that representation is
already superseded and not worth resurrecting to re-check.

**Why the isolated throwaway looked so much better in the previous
section, concretely.** It used a `compile` with no register at all (no
`PackedSignature`, no `HashMap` bookkeeping), so its one-time setup cost
was cheaper by construction -- not because its *decode* logic was any
different from the real, integrated version's (the two are structurally
identical, and produce byte-identical output, confirmed via matching
`size_bytes()`). The earlier writeup's "unexplained gap...
possibly thin-LTO context sensitivity" guess was wrong; the real cause
was measurement scope, not codegen.

## wiring PackedAutomaton into a real dictionary -- and a surprising result

Everything above measured `PackedAutomaton` in isolation: a bare
automaton, built once via `compile` or mutated via a tight loop of
`insert` calls on one never-splitting structure. That's not what
`AutomatonBlock`/`AutomatonDictionary` actually do with it -- a real
dictionary splits a block every time it crosses `block_max` (256 by
default) and rebuilds both halves from scratch. The isolated
sustained-insert measurement (`measure_packed_automaton_dead_bytes_
under_sustained_inserts`, 20-37x bloat within 128-200 inserts) never
let that happen -- it grew one block, unsplit, for up to 2,000 rounds,
specifically to see what happens without the safety valve a real
dictionary always has. Whether that safety valve is enough to keep the
bloat in check in practice was an open question until measured, not
something to assume either way.

**Built:** `PackedAutomatonBlock`/`PackedAutomatonDictionary`
(`packed_automaton_block.rs`/`packed_automaton_dictionary.rs`), mirroring
`AutomatonBlock`/`AutomatonDictionary`'s exact macro structure (lexical
blocks, sorted-array router, split/merge, `TermId` reverse mapping), a
fourth `TermDictionary` implementation alongside `PlainDictionary`,
`Dictionary`, `AutomatonDictionary` (`FrontCodedDictionary` is a fifth,
unrelated one). Deliberately smaller in scope than `AutomatonDictionary`
itself (which is already smaller in scope than `Dictionary` -- see its
own module doc comment): no `PreparedBatch`, no bulk-decode variants
beyond `get_term_into`. Two real, honest simplifications forced by what
`PackedAutomaton` doesn't have yet: `decode_into`/`merge_into`'s
existing-content walk uses `term_at_into` once per rank (root-restart
every call) instead of an incremental `stream()` (which
`AutomatonBlock` has via `Automaton::stream` and this doesn't), and
`remove` -- since `PackedAutomaton` has no incremental removal at all
-- rebuilds the whole block from scratch over its remaining terms, the
same full-rebuild shape `Block`'s own fst-backed `remove` already has.
Both are called out as open costs, not hidden.

Correctness first, as always: added as a fifth dictionary to `tests/
dictionary.rs`'s existing 400-step randomized insert/remove fuzz
harness (alongside `Dictionary`, `FrontCodedDictionary`,
`AutomatonDictionary`, `PlainDictionary` as oracle) across three
seeds/block-size configurations, plus its own RDF round-trip test. All
pass, including the `remove`-heavy fuzz runs that specifically exercise
the full-rebuild path.

**Memory, at real scale (`examples/perf_scale_memory.rs`, now with a
`packed` kind alongside `automaton`/`fst`) -- the actual result this
whole prototype was for.** 300,000 terms, both redundancy regimes,
`bytes/term` at the final checkpoint:

| corpus | `fst` (`Dictionary`) | `Automaton` (`AutomatonDictionary`) | `PackedAutomaton` (this) |
|---|---|---|---|
| clustered (high redundancy) | 12.91 | 26.03 | **14.79** |
| hashlike (low redundancy) | 29.79 | 737.58 | **132.23** |

This is the real, positive result the earlier isolated test's alarm
didn't predict: under actual split/merge churn, `PackedAutomatonDictionary`
beats the plain slab `AutomatonDictionary` by **1.76x** on the redundant
corpus and **5.6x** on the low-redundancy one -- getting close to `fst`
parity on clustered content, and cutting most (not all) of the gap to
`fst` on hashlike content. This is exactly the corpus/scale shape of
the *original* external benchmark that started this entire
investigation (`AutomatonDictionary` growing to hundreds of bytes/term
at real scale) -- and the byte-packed representation, wired in for
real, substantially closes that gap. The isolated test wasn't wrong
about what it measured (dead bytes do accumulate exactly as described,
confirmed independently via direct byte inspection in conversation),
but it measured a scenario -- unbounded single-block growth, no
splitting ever -- a real dictionary never actually lets happen: a block
crosses `block_max` and gets rebuilt from scratch well before insert
churn compounds anywhere near that test's 128-2,000-round range.
`PackedAutomaton`'s own `estimate`-vs-real gap (visible via
`AutomatonDictionary`'s existing `packed_estimate` column, which this
same run prints for the slab side) stays modest throughout -- confirming
the dead-byte tax from churn-between-splits, while real, is nowhere
near what the worst-case isolated test implied once bounded by actual
split cadence.

**Insert speed, at the dictionary level -- a real, newly-visible cost,
separate from `PackedAutomaton::insert`'s own already-measured one.**
Wall-clock for the full 300,000-term `insert_batch` run:

| corpus | `Automaton` (`AutomatonDictionary`) | `PackedAutomaton` (this) |
|---|---|---|
| clustered | 0.39s | 4.86s (~12x slower) |
| hashlike | 4.92s | 6.57s (~1.3x slower) |

`PackedAutomaton::insert` itself was separately measured (this
document, single-automaton probes) at only ~1.29x the slab's cost, not
12x -- so this gap is coming from somewhere else, and the most likely
place is the simplification named above: `PackedAutomatonBlock::
merge_into` decodes a block's *entire existing content* via `term_at_into`
once per rank (root-restart every call) before it can even determine
which incoming terms are new, on every merge -- an O(block size *
depth) cost `AutomatonBlock` avoids entirely via `Automaton::stream`'s
incremental walk. That cost is *insensitive to content redundancy* (a
structural walk, not helped by shared states the way `insert` itself
is), which plausibly explains why the gap is so much larger for the
highly redundant clustered corpus specifically: the slab's own insert
gets *faster* from redundancy (more sharing, less real work), while
`merge_into`'s decode-every-time cost doesn't get any such benefit.
Not confirmed with a dedicated profile -- a real next step, not
concluded here from the wall-clock shape alone.

**Where this leaves the byte-packed idea, updated once more.** Memory:
a genuine, substantial, real-scale win over the plain slab automaton in
both regimes, closing most of the original external benchmark's gap to
`fst` -- the strongest positive result this whole investigation has
produced, and one the single-automaton, single-block measurements
alone couldn't have shown (it depends on real split cadence). Insert
speed at the dictionary level: a new, real cost, distinct from and
larger than the already-measured `PackedAutomaton::insert` regression,
traced (not yet confirmed) to the missing incremental block-decode
walk. Decode: unaffected by any of this, still the corpus-dependent
picture from the methodology-correction section above. Remove: full
block rebuild, an accepted, honest limitation. Net: this is the first
point in the whole investigation where the byte-packed representation
looks like a real net positive on the axis (memory, at scale) the
original problem was actually about -- with a now-quantified, not yet
closed, follow-on cost (dictionary-level insert speed) as the next
thing to fix if this is worth carrying further.

## a traditional (direct) build algorithm for PackedAutomaton

Every `PackedAutomaton` built so far -- in every measurement in this
document -- went through `compile(&Automaton::build_sorted(..))`: build
a full `Automaton` slab (every `State`, every `TransitionChunk`, the
slab's own register), then walk it postorder into packed bytes, then
throw the slab away. That's real, measured waste for a structure that's
never queried in slab form at all -- worth fixing directly rather than
accepting as the only way to get a `PackedAutomaton` from a term list.

Added `PackedAutomaton::build_sorted`, the same sorted-input
specialization of Daciuk/Mihov/Watson/Watson 2000 `Automaton::build_sorted`
already implements (rightmost path as a stack of unconfirmed frames,
confirm everything deeper than the common-prefix length with the
previous word), but writing packed records directly -- no slab at all.
It reuses `insert`'s own per-frame machinery unchanged (`StagingFrame`,
`confirm_frame`, `write_confirmed_state`): sorted construction only
changes how frames are opened and closed (append-only children, since a
diverging byte at any depth is always greater than any sibling already
there -- the same guarantee `Automaton::build_sorted`'s own
`link_new_appending` relies on, vs. `insert`'s general
`binary_search_by_key` splice), never how one frame gets confirmed once
ready.

**Correctness first.** Two new differential tests:
`packed_automaton_build_sorted_matches_compile_from_automaton` (2,000
clustered terms -- identical `size_bytes`, `reachable_state_count`, and
every `rank`/`term_at_into` answer against the old
compile-from-`Automaton` route) and
`packed_automaton_build_sorted_matches_incremental_insert` (320 terms,
`build_sorted` directly vs. one-at-a-time `insert` from empty -- same
reachable structure either way). Both pass, plus the full existing
suite (108 tests) unaffected.

**Build speed, `perf stat`-confirmed, isolated per-variant (not summed
across both in one probe):**

| corpus | via `Automaton::build_sorted` + `compile` | `build_sorted` (direct) | speedup |
|---|---|---|---|
| clustered (2,000 terms) | 813.7µs / 3.40M cycles | 378.1µs / 3.03M cycles | 2.15x wall, 1.12x cycles |
| hashlike (20,000 terms) | 257.8ms / 695.3M cycles | 144.4ms / 370.2M cycles | 1.79x wall, 1.88x cycles |

A real, substantial build-speed win in both regimes, exactly where
expected: skipping the intermediate slab construction entirely rather
than paying for it and discarding it.

**Wired in:** `PackedAutomatonBlock::build` (every block construction --
bootstrap, split, merge, and now also `remove`'s full-rebuild path) uses
`PackedAutomaton::build_sorted` directly instead of routing through
`Automaton`. Re-running `examples/perf_scale_memory.rs` at the same
300,000-term scale as the dictionary-level section above: memory is
byte-identical (14.79/132.23 bytes/term, unchanged, as the differential
tests already guaranteed), and hashlike `insert_batch` wall-clock
dropped from 6.57s to 4.90s (~1.34x faster) -- `bootstrap` (the initial
large batch, many `PackedAutomatonBlock::build` calls in a row) is a
real fraction of hashlike's total cost. Clustered `insert_batch`
wall-clock was essentially unchanged (4.86s vs. 4.85s), which is itself
informative: it confirms the earlier hypothesis that the dictionary-level
insert-speed gap is dominated by `merge_into`'s missing incremental
decode walk, not by block construction cost -- speeding up `build`
alone barely moves a number that construction was never the bottleneck
for. That gap remains open, unaffected by this change, and is still the
next thing to fix if dictionary-level insert speed matters for this
representation's viability.

## an incremental stream, to close the merge_into gap

`PackedAutomatonBlock::decode_into`/`merge_into` were built, deliberately,
using `term_at_into` once per rank (root-restart every call) because
`PackedAutomaton` had no incremental walk -- `AutomatonBlock` has one via
`Automaton::stream`. Added `PackedAutomaton::stream`/`PackedTermsStream`,
the same technique, over the packed byte buffer: each `PackedStreamFrame`
caches its own parse cursor (the byte offset of its next unvisited
transition), advanced by one transition per `next()` call, so a state's
transitions are each parsed exactly once total across a whole walk --
`Transitions::at`'s O(1) re-derivation (fine for a small typed array)
isn't available for a variable-length packed record, so the cursor cache
is what makes this genuinely amortized rather than a naive port.

**Correctness first**: two new differential tests
(`packed_stream_matches_term_at_into_for_every_rank`,
`packed_stream_backtracks_correctly_across_shared_and_diverging_prefixes`)
pass, plus the full suite (110 tests) including the `remove`-heavy fuzz
runs that exercise `decode_into`/`merge_into` continuously. Wired into
`decode_into`, `merge_into`'s existing-content walk, and `remove`'s
full-decode, replacing every `term_at_into`-per-rank loop in
`packed_automaton_block.rs`.

**Isolated at the block level** (same probe shape as `automaton_block.rs`'s
own `perf_probe_merge_into_single_term_automaton_vs_block`, now with a
`PackedAutomatonBlock` column): this alone *reversed* the block-level
comparison -- `PackedAutomatonBlock::merge_into` went from something not
directly measured before to 62.2-67.2µs/op, *faster* than
`AutomatonBlock::merge_into`'s 84.0µs/op (both still slower than
`Block`'s full fst rebuild at this specific, deliberately-unbounded,
never-split growing-block shape -- see that probe's own comment on why
it's not representative of a real, split-bounded workload).

**But the dictionary-level gap barely moved at first** (4.85s -> 4.56s
for the 300,000-term clustered `insert_batch` benchmark, far short of
what the block-level reversal predicted) -- a real, measured result that
didn't match the hypothesis, worth chasing rather than declaring victory
on the block-level number alone. A `perf record` profile of the actual
dictionary-scale run found why: **`hash_one::<&PackedSignature>` plus
`SipHash`'s own `write`, combined, close to 40% of all cycles** --
`PackedAutomaton`'s `register: HashMap<PackedSignature, u32>` was built
with plain `HashMap::new()`, silently defaulting to std's `SipHash`.
`Automaton`'s own slab register already solved exactly this problem,
years earlier in this project's history (`FxHasher`, a fast
non-cryptographic hash, documented at length in its own doc comment
specifically because a `perf record` profile of `build_sorted` found the
same cost dominant then) -- `PackedAutomaton`'s register was simply never
switched to it. Fixed by reusing `FxHasher` directly (`type
PackedRegister = HashMap<PackedSignature, u32,
BuildHasherDefault<FxHasher>>`), the exact type alias pattern `Register`
already established.

That alone dropped clustered to ~3.15s and hashlike to ~3.82s (both
`perf stat`-confirmed, not just wall-clock). Two more precedented,
low-risk fixes followed from continuing to profile rather than stopping
at the first win: `write_confirmed_state` built a fresh
`PackedSignature { transitions: children.to_vec(), .. }` on *every*
register lookup, the overwhelming majority of which are hits that
immediately discard that allocation -- exactly the pattern `Signature`/
`signature_of_into` already fixed for the slab register via a reused
scratch buffer (`sig_scratch`, reclaimed out of a dropped signature on a
hit instead of let it deallocate); and `compile`/`build_sorted` now
reserve the register's capacity upfront (`automaton.len()`/`terms.len()`
as a cheap-to-compute hint) instead of growing it one `insert` at a time.

**A methodology mistake caught mid-investigation, worth recording
plainly**: after applying the last two fixes, a re-profile of the
dictionary-scale run still showed `reserve_rehash` essentially unchanged
(~13% either way) -- initially read as "these two fixes didn't help."
The actual cause: `perf record` was pointed at `target/profiling/examples/
perf_scale_memory`, which had not been rebuilt since *before* the
scratch-buffer/capacity fixes landed (confirmed via `md5sum` once asked
directly whether the binary had actually been rebuilt) -- so that profile
was silently re-measuring old code. Rebuilding and re-profiling is the
only way any of these numbers are trustworthy; skipping that step
produced a specific, wrong conclusion, not just a stale one, and would
have gone unnoticed without the check.

**Final, verified-fresh numbers** (`perf stat`, 300,000 terms, matching
the earlier dictionary-level section's methodology exactly):

| corpus | `Automaton` (`AutomatonDictionary`) | `PackedAutomaton` (this) | ratio |
|---|---|---|---|
| clustered | 1.455B cycles (0.39s) | 9.070B cycles (3.86s) | packed 6.2x *more* |
| hashlike | 18.102B cycles (4.99s) | 12.563B cycles (4.64s) | packed 1.44x *fewer* |

Hashlike now genuinely beats the plain slab automaton at the dictionary
level -- a real, new result, not just parity. Clustered improved
substantially from where this section started (the original wall-clock
gap was ~12x; cycles now show ~6.2x) but the gap is **not closed**: a
`perf record` profile after all four fixes still shows `confirm_frame`
(~20%) and `reserve_rehash`/`cfree`/`malloc`/`HashMap::insert` (~12%,
~10%, ~4%, ~6% respectively) as the dominant costs, meaning real
allocation/rehashing churn remains, just reduced rather than eliminated.
Why the capacity-hint reservation doesn't fully close it is itself not
fully understood -- a direct check found a representative 256-term
clustered block only needs ~91 reachable states (`reachable_state_count()
/ len() ≈ 0.36`), so `terms.len()` as a capacity hint should
over-provision, not under-provision, for a single `build_sorted` call;
the remaining rehashing more likely comes from `merge_into`'s repeated
per-term `insert()` calls into an *already-built* block, which never
reserve anything themselves, but this hasn't been confirmed with a
dedicated profile isolating build from insert at realistic block sizes,
so it's reported as the leading hypothesis, not a settled conclusion.
Further work here (pre-reserving before a run of `insert` calls in
`merge_into`, or restructuring the register/signature scheme entirely)
is real, open, and not attempted in this pass.

## the real cause, found by asking two sharp questions instead of accepting the numbers

Two things about the section above didn't add up on inspection: (1) if
`FxHasher`, scratch-buffer reuse, and capacity reservation are all real
fixes, why does `Automaton`'s own register -- built the same way, same
hasher, comparable signature size -- not show the same `reserve_rehash`
cost in its own profile? (2) it makes no sense for *more* register hits
(clustered, heavily shared content) to make `PackedAutomaton` *worse*
relative to `Automaton` -- a hit is the cheap case (see the confirm-step
comparison two sections up: a hit just drops a temporary `Vec`, nothing
written). Chasing both found the real cause, and it isn't in the register
at all.

**Automaton's own register, profiled the same way** (`perf record` on
`automaton clustered` at the same 300,000-term scale): no `reserve_rehash`
in the top costs. The profile instead shows `HashMap::get` (11.4%, the
lookup itself), `insert_sorted` (8.3%), `release` (8.0%), `confirm_path`
(8.0%), `Transitions::at`/`TermsStream::next` (the decode-walk side).
Real hashing and lookup work, no resize churn -- a genuine difference
from `PackedAutomaton`'s profile, not a measurement artifact repeated on
both sides.

**A real benchmark asymmetry, found and fixed**: `examples/
perf_scale_memory.rs`'s `"automaton"` branch called `dict.
estimate_packed_bytes()` -- a full O(reachable states) diagnostic walk
across every block -- on *every checkpoint*, something the `"packed"`
and `"fst"` branches never did. This call predates `PackedAutomatonDictionary`
existing at all (it was a *hypothetical* size estimate, back when
nothing could measure the real thing); now that a real packed dictionary
exists to measure directly, the estimate is redundant, and it was
quietly inflating `Automaton`'s measured time in every earlier comparison
in this section. Removed. Re-measured: clustered barely changed (1.455B
-> 1.381B cycles, clustered's automaton has few states, cheap to walk)
but **hashlike dropped 27%** (18.102B -> 13.183B cycles -- hashlike's
automaton has vastly more states, per its own much worse bytes/term, so
walking all of them every checkpoint was real, substantial, uncounted
work). The "hashlike now beats Automaton" result two sections up was
correspondingly inflated by this asymmetry, not fabricated but
overstated -- see the corrected table below.

**The actual cause of "more hits is worse," found by reading
`Automaton::insert_sorted` next to `PackedAutomatonBlock::merge_into`
side by side**: `Automaton::insert_sorted` keeps `path: Vec<(StateId,
u8)>` *outside* its per-term loop and, for each incoming term, calls
`confirm_path(..., common_prefix_len(last_word, word) + 1, ...)` --
only the states *deeper* than the shared prefix with the *previous*
term get re-walked and re-confirmed; a long shared prefix across many
consecutive sorted terms is walked and confirmed once, not once per
term. `PackedAutomatonBlock::merge_into`, by contrast, still does `for
&term in &incoming { self.packed.insert(term); }` -- `PackedAutomaton::
insert` has no memory of the previous call, so it rebuilds `path: Vec<u32>
= vec![self.root]` and walks from the root fresh, every single term.
For a batch of terms sharing a long prefix (exactly what "more hits"
means), this redoes the *entire shared prefix's* walk-and-confirm work
once per term instead of once per batch -- an O(terms x shared-depth)
cost where `Automaton`'s own batch algorithm is closer to O(unique
states touched). This is why redundancy makes it *worse*, not better:
redundancy is exactly what a real batch algorithm turns into savings,
and what a term-at-a-time loop turns into repeated, wasted work instead.
None of the register-level fixes above (hasher, scratch buffer, capacity
hint) touch this -- they reduced the *cost per confirm*, not the *number
of redundant confirms*, which is the actual, larger problem.

**Corrected, fair comparison** (fixed benchmark, `perf stat`, 300,000
terms):

| corpus | `Automaton` | `PackedAutomaton` | ratio |
|---|---|---|---|
| clustered | 1.381B cycles (0.61s) | 9.070B cycles (3.86s) | packed 6.6x *more* |
| hashlike | 13.183B cycles (5.98s) | 12.563B cycles (4.64s) | packed 1.05x *fewer* (~parity) |

The hashlike "win" from the previous section shrinks from 1.44x to
essentially parity once the benchmark asymmetry is removed -- still not
a loss, but not the clear win previously reported either. Clustered's
gap got slightly *larger* (6.2x -> 6.6x) now that `Automaton`'s number
isn't artificially inflated. Given the root cause identified above is
algorithmic (missing batch sharing in `merge_into`), not something the
register-level fixes could reach, a real `PackedAutomaton::insert_sorted`
-- `insert_sorted`'s exact persistent-path, confirm-on-divergence
technique, targeting the packed representation the same way
`PackedAutomaton::build_sorted` already does for the from-scratch case
-- is the next real fix, not yet attempted. `PackedAutomatonBlock::
merge_into` would call it once per batch instead of looping single-term
`insert`, the same way `AutomatonBlock::merge_into` already calls
`insert_sorted` instead of looping `Automaton::insert`.

## PackedAutomaton::insert_sorted, and the gap actually closes

Built exactly that: `PackedAutomaton::insert_sorted`, `Automaton::
insert_sorted`'s persistent-rightmost-path technique targeting the
packed representation, the same way `build_sorted` already targets it
for from-scratch construction. `frames`/`path_bytes`/`last_word` are now
carried *across* the whole incoming batch; between consecutive terms,
`confirm_down_to` only re-confirms frames deeper than `common_prefix_len
(last_word, word) + 1`, and matching continues from `common` by reading
`frames.last().children` (a binary search) rather than restarting from
`self.root`. No clone-if-shared/generation bookkeeping is needed, unlike
the slab's own `insert_sorted`: every frame on the path is already a
private, in-memory `StagingFrame`, never a live reference other content
could alias -- the same reason plain `insert` never needed it either.

One real correctness fix fell out of building this: `confirm_down_to`
(shared with `build_sorted`) used to splice a resolved child into its
parent with a blind `push`, safe only because `build_sorted`'s frames
are always freshly created (a diverging byte is always greater than
every existing sibling). `insert_sorted` reuses the same helper to
re-confirm frames opened from *pre-existing* structure, where the byte
being spliced back may already have an entry -- a blind `push` there
would silently duplicate it. Fixed by switching to the same
`binary_search_by_key`-based update-or-insert `insert` already uses,
harmless for `build_sorted` (a no-op search that always misses at the
end) and correctness-critical for `insert_sorted`.

**Correctness first**: two new differential tests
(`packed_automaton_insert_sorted_matches_build_sorted`;
`packed_automaton_insert_sorted_matches_automaton_insert_sorted`, a
mixed batch of re-insertions and brand-new terms against an
already-populated baseline, checked against both `Automaton::
insert_sorted` and a fresh rebuild's `reachable_state_count`) pass,
alongside the full suite (112 tests) including the 400-step fuzz runs
that exercise `merge_into`'s new call path continuously.

Wired into `PackedAutomatonBlock::merge_into`, replacing the `for &term
in &incoming { self.packed.insert(term); }` loop with one
`self.packed.insert_sorted(incoming.iter().copied())` call.

**The result, `perf stat`-confirmed, reproduced**:

| corpus | `Automaton` | `PackedAutomaton` (naive loop) | `PackedAutomaton` (`insert_sorted`) |
|---|---|---|---|
| clustered | 1.381B cycles (0.61s) | 9.070B cycles (3.86s) | **0.84-0.87B cycles (0.39-0.41s)** |
| hashlike | 13.183B cycles (5.98s) | 12.563B cycles (4.64s) | **7.79-7.99B cycles (3.62-3.72s)** |

`PackedAutomatonDictionary` now beats `AutomatonDictionary` on insert
speed in **both** regimes -- clustered by ~1.6x fewer cycles (a 10.8x
improvement over the naive loop), hashlike by ~1.7x fewer cycles (a
1.6x improvement over the naive loop). This directly confirms the
diagnosis: the naive per-term loop's cost was dominated by redundant
re-confirmation of shared prefixes, not by anything inherent to the
register/hasher/allocation layer those earlier fixes targeted --
removing the redundancy at the algorithm level did far more than any of
those lower-level fixes individually, and this is the first point in
the whole investigation where the byte-packed representation wins on
speed as well as memory, in both tested regimes, not a tradeoff between
them.

A bonus, not the point of this change but worth recording: memory
improved too. Hashlike dropped from 132.2 to ~104.2 bytes/term (~21%
smaller); clustered was already good and stayed flat (~14.7-14.8).
Makes sense in hindsight -- the naive loop's redundant re-confirmation
of a shared prefix didn't just cost cycles, each redundant re-confirm
also wrote (or, on a lucky register hit, at least *attempted*) a fresh
record, so cutting the redundancy cuts dead-record accumulation too,
not just the CPU time spent producing it.

## decode_ranks_into, ported and wired through

`Automaton::decode_ranks_into` (a recursive descent that partitions a
sorted batch of requested ranks against each state's own transitions,
visiting a state only when some target actually falls in its range --
`AutomatonDictionary::get_terms_seeked_into`'s own primitive) had no
`PackedAutomaton` counterpart. Ported directly: `PackedAutomaton::
decode_ranks_into`/`decode_batch` mirror the slab algorithm exactly,
with one real adaptation -- `Transitions::at(idx)` re-derives any index
in O(1) for the slab's small typed array, but a packed record has no
such re-entry point, so a one-transition lookahead cache (`next`) parses
transition `idx + 1` once, when it's peeked for the sibling boundary,
and reuses that parse as `current` on the following iteration, keeping
this a single sequential pass over the record either way (same
reasoning as `PackedTermsStream`'s own cursor caching).

Two differential tests, mirroring `Automaton`'s own exactly (every rank
in one batch with decorrelated output positions; several widely
separated rank clusters plus a duplicate, the shape that broke an
earlier seek-then-crawl design) -- both pass. Wired all the way through:
`PackedAutomatonBlock::decode_sorted_ranks_into` and
`PackedAutomatonDictionary::get_terms_seeked_into`, mirroring
`AutomatonBlock`/`AutomatonDictionary`'s own exactly, with their own
differential tests. Full suite: 117 tests, clippy clean. Not yet
measured against `get_terms_into`/`get_terms_scattered_into`-equivalent
approaches the way `s-bench`'s `dict_bench` measures the slab's three
strategies against each other -- `PackedAutomatonDictionary` only has
this one bulk-decode path so far, so there's nothing to compare it
against yet within this crate.

## decode_ranks_into, measured

Four new perf probes (`perf_probe_decode_ranks_cost_{automaton,
packed_automaton}_{clustered,hashlike}`), a "seeked" query shape --
several small (5-rank) clusters scattered across the term set with
large, deliberately awkward gaps between them, the shape `decode_ranks_
into`/`get_terms_seeked_into` actually exist for, not "every rank in
order" (already covered separately by `stream`). First attempt used the
same `DECODE_PROBE_REPEATS = 20` every other decode probe uses and
produced untrustworthy numbers -- the actual decode work (200 targets,
20 reps) finished in ~0.2ms, far too small relative to process startup
for `perf stat` to attribute meaningfully; bumped to a dedicated
`DECODE_RANKS_PROBE_REPEATS = 2000` for these four probes specifically,
landing the measured work at 22-88ms, comparable to the other decode
probes' magnitude.

`Automaton`'s own numbers need no correction (no separate compile step);
`PackedAutomaton`'s do, same methodology as every other packed decode
comparison in this document -- subtracting `perf_probe_setup_only_
clustered`/`_hashlike`'s `build + compile` baseline from the full probe:

| corpus | `Automaton` (cycles) | `PackedAutomaton`, decode-only (cycles) | ratio |
|---|---|---|---|
| clustered | 89.87M | ~83.4M (86.7-87.1M full - 3.47M setup) | packed ~1.08x fewer |
| hashlike | 679.30M | ~304.9M (911-938M full - ~619M setup) | packed ~2.23x fewer |

`PackedAutomaton::decode_ranks_into` is faster than `Automaton`'s own in
both regimes -- roughly at parity for clustered, a real 2.2x win for
hashlike. Consistent with every other *decode-only* (compile excluded)
comparison already made in this document (`rank`/`term_at_into`, and now
this batch primitive): the byte-packed layout's decode advantage holds
across every decode operation measured so far, not just the two
originally checked.

## the real RDF finding: states/term itself grows, and it's boundary-sensitive

An external experiment wired both dictionaries into a real RDF parser
over a real large file, reproducing the original memory-scaling problem
this whole investigation started from -- and found `PackedAutomatonDictionary`
mitigates it (19-28% less retained memory than `AutomatonDictionary` at
both 524K and 4.58M distinct terms, slower growth: `size_bytes()` +69%
vs. automaton's +146% over that range) but doesn't fix it: both still
grow super-linearly while `Dictionary` (fst) stays flat.

**First lead, real but secondary.** `release()` (`src/micro_fst.rs:2379`)
explicitly removes a state's `Signature` from `Automaton`'s register the
moment its refcount hits zero -- the register only ever holds entries
for currently-reachable states, matching `size_bytes()`'s own scope.
`PackedAutomaton`'s register has no equivalent: `write_confirmed_state`
only ever `insert`s, by design (no refcounting at all, a deliberate
simplification from when `PackedAutomaton` was first built) -- so it
tracks *every* state signature ever confirmed for that automaton's
whole lifetime, live or dead, each one a separate heap-allocated
`Vec`, entirely invisible to `size_bytes()` (which only sums
`self.bytes.len()`). This explains why packed's retained-vs-`size_bytes()`
gap is much wider than automaton's (a ~68-point spread vs. an ~11-point
one, in the reported numbers) -- a real, previously unmeasured cost.
But it's secondary: it explains why *packed's* overhead is worse than
*automaton's*, not why *both* grow super-linearly while `fst` doesn't,
since `PackedAutomaton`'s bytes are a direct serialization of the same
minimal DAG `Automaton` itself computes.

**Reframing the question.** If packed's structure is `Automaton`'s own
DAG serialized, the shared root cause has to be upstream of both --
either the true *state count per term* is growing with corpus scale (a
property of minimization/content, which `fst` would share too, just
hidden by its much smaller per-state byte cost), or state count is flat
and something else is amplifying an invisible trend into a visible one.
`reachable_state_count()` (on `Automaton` and `PackedAutomaton`) already
existed for tests; un-gated from `#[cfg(test)]` and given real
counterparts: `Block::reachable_state_count()` (walks `fst::raw::Fst`'s
own node graph via `Node::transitions`/`CompiledAddr` -- public API, no
vendored-crate changes needed) and `Dictionary::reachable_state_count()`,
plus the matching `AutomatonBlock`/`AutomatonDictionary`/
`PackedAutomatonBlock`/`PackedAutomatonDictionary` wrappers. Wired into
`examples/perf_scale_memory.rs` as a `states`/`states/term` column
alongside the existing bytes ones.

**First result, at 2,000,000 clustered terms:** all three start
*exactly equal* (24,650 states at 200K terms) and diverge from there.
By 2M terms, `fst` has 256,937 states vs. automaton/packed's 366,962
for the identical term set -- **43% more**, and `PackedAutomaton` tracks
`Automaton` state-for-state throughout (as expected -- same DAG). So
it's real: state count itself is growing faster for our own
minimization than for `fst`'s, not just per-state byte cost.

**Isolating which content pattern is responsible**
(`namespace_clustered_terms_range` interleaves four different suffix
shapes): built a throwaway diagnostic driving each pattern alone
through `AutomatonDictionary`/`Dictionary` at increasing scale.
Decimal-digit suffixes diverge badly -- unpadded entity/property ids up
to **+56.7%** by 500K terms, fixed-width zero-padded isbn up to
**+30.1%** -- while hex blank-node ids (also digit-suffixed, also
unpadded) stay under **+3%**. `fst`'s states/term is essentially flat
for both decimal patterns (0.1794, 0.1031, unmoving) but *does* drift a
little for hex (+7%) -- the opposite direction from automaton's own
pattern. This points at base-10 digit branching specifically.

**Four decisive, isolating experiments**, using new `#[doc(hidden)]`
`__internal_bench` primitives (`build_automaton_states`,
`build_fst_states`, `incremental_insert_sorted_rounds_states` --
kept, low-cost, useful for any follow-on): all built and run on
consecutive, zero-padded 13-digit isbn-shaped strings, isolating one
candidate mechanism at a time.

1. *One global, unbounded automaton over 0..200,000* (no blocks at
   all): automaton = fst = 14 states. Zero divergence -- a single
   regular counting sequence compresses identically either way.
2. *2,000 independent, cleanly-aligned 256-term slices* (built
   directly, bypassing the dictionary/router entirely, at every offset
   from 0 to 512,000): automaton = fst at every single one. Zero
   divergence -- clean block boundaries alone don't cause it.
3. *Multi-round incremental `insert_sorted`* (10 separate rounds into
   one never-split automaton) vs. one-shot `build_sorted` over the
   identical final term set: 14 states either way. Zero divergence --
   incremental insertion doesn't drift away from minimality on its own.
4. *The real `AutomatonDictionary`/`Dictionary`, same 100,000-term
   corpus, `insert_batch` chunk size varied* (64 up to the full
   corpus in one call): divergence tracks chunk size almost exactly --
   **0%** when chunk size equals `block_max` (256) or the whole corpus
   arrives in one bootstrap call (both give the same 391, cleanly-cut
   blocks), up to **+83%** at chunk=4096 (585-736 blocks, each boundary
   chosen by `split()`'s repeated median-halving of an oversized
   block). `fst`'s own state count stays almost flat across every
   chunk size (6898-6955, aside from two much-smaller-chunk outliers);
   automaton's swings from 6,955 to 13,585 -- **nearly 2x** -- for the
   *same final term set*, cut differently.

**Conclusion, narrowed and now mechanistically explained**: this isn't
block partitioning, isn't incremental-insertion drift, and isn't
inherent to digit sequences in general -- it's specifically that
`Automaton`'s minimization is *sensitive to exactly where a block
boundary falls* for regular digit sequences, in a way `fst`'s is not.
`fst` gives close to the same total state count regardless of cut
position; ours varies sharply. `split()`'s median-halving of an
overflowing block (not `bootstrap`'s round, evenly-divided chunking) is
what produces the boundary positions that trigger it.

## why: output-pushing, not registry completeness

Reading `fst`'s own vendored source (`vendor/fst`) and the literature
both implementations draw from resolved the mechanism -- written up in
full, with code and paper citations, in `AUTOMATON_REFERENCE.md`. Short
version: `fst`'s registry (`vendor/fst/src/raw/registry.rs`) is a
*bounded, lossy* LRU cache -- `fst`'s own doc comment says outright
(`vendor/fst/src/raw/mod.rs:226-238`) that it does not guarantee a
minimal transducer, trading exactness for constant memory. `s-dict`'s
own `Register` (`micro_fst.rs:489`) is the opposite: an *unbounded,
exhaustive* `HashMap`, strictly more thorough at finding exact matches.
`fst` still wins on state count, because it's searching a *larger space
of what counts as a match*: `fst`'s `Builder` implements Mihov &
Maurel's output-pushing (`vendor/fst/src/raw/build.rs:406-453`, a
min-plus-semiring `Output::prefix`/`sub`/`cat` algebra,
`vendor/fst/src/raw/mod.rs:1276-1322`) -- a transition's output value is
*renegotiated*, not fixed, as later siblings arrive, so two subtrees
with the same shape but different absolute rank offsets become
byte-identical once the common offset is factored out. `s-dict`'s
`Automaton` computes each state's children's outputs as a one-shot
prefix sum at confirmation time (`finalize_outputs_and_count`,
`micro_fst.rs:2604-2630`) and never revisits them. An exhaustive search
over the narrower ("raw values must already match") space loses to a
lossy search over the wider ("match after renegotiation") space, because
the wider space contains merges the narrower one can never find no
matter how exhaustively it looks -- and a `split()`-produced boundary,
an arbitrary slice of a regular sequence with no reason to already have
aligned raw output values, is exactly the case that needs the
renegotiation to expose its shared structure.

Not yet done: isolating output-pushing's contribution from the
registry-boundedness difference (they differ simultaneously here), or
attempting to add output-pushing to `Automaton`/`PackedAutomaton` --
`AUTOMATON_REFERENCE.md`'s §5 has the specific next experiments this
would need, not attempted in this pass.

## measuring the register-pruning gap directly

`AUTOMATON_REFERENCE.md`/`FST_VS_PACKED_AUTOMATON.md` both asserted,
from reading the code rather than running it, that `Automaton::release()`
(`micro_fst.rs:2420`, `register.remove(&sig)` on refcount-zero) keeps
`register` trimmed to the reachable state set, while `PackedAutomaton`'s
`write_confirmed_state` (`micro_fst.rs:1751-1770`, `register.insert` at
line 1768, no corresponding `remove` anywhere in the type) has no such
mechanism. Added `register_len()` to both types (`#[cfg(test)]`,
`micro_fst.rs:1146` and `:1814`) and two measurement tests
(`measure_automaton_register_tracks_reachable_under_churn`,
`measure_packed_automaton_register_growth_under_sustained_inserts`,
both `#[ignore]`, `cargo test --release --lib <name> -- --ignored
--nocapture`) to check this directly instead of leaving it asserted.

**`Automaton`**, 2000 rounds of one-out-one-in churn (remove one live
term, insert one new one, holding the live set at 800 terms, same shape
as `measure_slot_reclamation_gap_under_sustained_churn`): `register_len`
tracked `reachable_state_count` within 0.994-0.996x for every reported
round, start to finish (round 0: register=90, reachable=91; round 2000:
register=245, reachable=246 -- off by exactly 1, the root, which is
never itself a register entry). The pruning claim holds, measured, not
just read.

**`PackedAutomaton`**, 2000 rounds of insert-only churn (no `remove`
exists on this type; same corpus and new-term pattern as
`measure_packed_automaton_dead_bytes_under_sustained_inserts`):
`reachable_state_count` stayed essentially flat (91 -> 118 -> 120 across
all 2000 rounds -- the inserted terms share a long common prefix, so
almost nothing new becomes reachable), while `register_len` climbed
without bound: 91 -> 62,632. The register/reachable ratio went from
1.0x at round 0 to **522x** at round 2000, climbing roughly linearly the
whole way (round 200: 55.3x; round 1000: 263.6x; round 2000: 521.9x).
Every insert leaves the previous round's now-superseded confirmed
states stranded in the register permanently -- measured, not inferred,
and a materially worse gap than the qualitative "only ever grows"
framing suggested before this was run.

## replacing PackedAutomaton's register with a bounded LRU cache

The unbounded exact `HashMap` register measured above is now gone.
`PackedRegister`/`PackedSignature` were replaced with `PackedRegistry`/
`PackedRegistryCell` (`src/micro_fst.rs`, just above `PackedAutomaton`'s
own struct definition) -- `fst`'s own `Registry`
(`vendor/fst/src/raw/registry.rs`) ported to this crate's target type
(`u32` byte offsets, not `CompiledAddr`): a fixed `table_size *
mru_size` array of cells, each holding its own clone of the signature
it was inserted with, hashed into a bucket (`FxHasher`, this crate's
own established choice, not `fst`'s FNV-1a), linearly scanned within
that bucket, promoted-on-hit, LRU-evicted-on-miss -- `fst`'s own
`RegistryCache::entry`'s exact algorithm. `table_size` is sized once,
from the same formula `Block::build` already uses for its own `fst`
registry (`entries.len() * 4, clamped [64, 10_000]`, `mru_size = 2`,
`src/block.rs:71`), and never resized afterward -- sound specifically
because `s-dict` blocks (and so a `PackedAutomaton`'s term count) are
already size-bounded by this crate's own split policy (`DEFAULT_BLOCK_
MAX = 256`, `src/dictionary.rs:33`), so a table sized from the initial
build stays proportioned across further incremental `insert`s rather
than needing to track growth. `write_confirmed_state` (`src/
micro_fst.rs`) simplified alongside this: `get`/`insert` now take
`children: &[(u8, u32, u32)]` directly, so a hit -- the overwhelming
majority of calls -- touches no allocation at all; `sig_scratch` (the
scratch-buffer-reuse workaround the old exact `HashMap` needed to avoid
allocating a fresh `Vec` on every lookup) is gone entirely, no longer
needed.

This is the same tradeoff `fst` has always made, now applied to
`PackedAutomaton` too: bounded, constant memory for the dedup
structure itself, in exchange for a documented, accepted loss of exact
minimality (`vendor/fst/src/raw/mod.rs:226-238`). Three existing tests
asserted exact structural equality across different construction
orders over the same term set (`packed_automaton_build_sorted_matches_
incremental_insert`, `packed_automaton_insert_sorted_matches_build_
sorted`, the former `packed_automaton_insert_reaches_same_minimality_
as_fresh_build`) -- a property this design deliberately no longer
guarantees, since a bounded LRU cache's eviction decisions (and so its
final structure) depend on access history, which different construction
orders genuinely differ on. Updated to assert what still must hold:
every `rank`/`term_at` answer stays correct regardless of construction
order, and a lossy cache's equivalence classes are a subset of an exact
one's, so `reachable_state_count` can only be `>=` a fresh minimal
build's, never less (`packed_automaton_insert_stays_correct_though_no_
longer_exactly_minimal` now asserts `>=` in place of `==`).

**Remeasured, same two corpora `measure_packed_automaton_register_
growth_under_sustained_inserts` (above) used**, 2000 rounds of
insert-only churn on an 800-term base set: `register_len` now plateaus
at exactly 6,400 (= `table_size` 3,200 x `mru_size` 2, `with_capacity_
hint(800)`'s own sizing) instead of climbing to 62,632 -- bounded, as
designed, confirmed rather than assumed:

```
round    0: register=91    reachable=91
round  200: register=4,662 reachable=118
round  600: register=6,324 reachable=120
round 1200: register=6,399 reachable=122
round 2000: register=6,400 reachable=124   (plateaued since ~round 800)
```

**Remeasured at dictionary scale, `examples/perf_scale_memory.rs`,
the exact two corpora/sizes the original RDF-report investigation
used.** Low-redundancy (hashlike, 300K terms) is where the register
growth actually showed up in `size_bytes()` before -- and it's fixed:

| terms | `PackedAutomaton` bytes/term | `PackedAutomaton` states/term | `fst` states/term |
|---|---|---|---|
| 30,000  | 104.54 | 10.803 | 10.786 |
| 90,000  | 104.83 | 10.540 | 10.566 |
| 150,000 | 103.83 | 10.398 | 10.334 |
| 210,000 | 109.45 | 10.208 | 10.192 |
| 300,000 | 105.33 | 10.147 | 10.081 |

(Full ten-checkpoint run, both `bytes/term` and `states/term`, in the
"measuring the register-pruning gap directly" section's sibling run
above this one -- `./target/release/examples/perf_scale_memory packed
hashlike 300000 10` / `... fst hashlike 300000 10`.) `bytes/term` is
now flat across the whole range (104.5 at 30K, 105.3 at 300K, no trend
-- bouncing within noise, not climbing) where this exact corpus and
scale previously showed real, recorded growth: 132.23 bytes/term at
300K hashlike terms before this fix ("Memory, at real scale" section
above, same `perf_scale_memory` harness), up from ~108 at 5K
(`FST_VS_PACKED_AUTOMATON.md`'s "Memory at scale" row). The
register-growth-driven bloat this whole investigation
chased is gone on this corpus. `states/term` now tracks `fst`'s own
curve closely (10.147 vs. 10.081 at 300K, both decreasing at
essentially the same rate) -- confirming the fix addressed exactly the
mechanism it targeted. The remaining `bytes`-level gap vs. `fst` (105
vs. ~30 bytes/term) is the separate,
already-documented coding-dimension difference (`fst`'s three adaptive
node layouts + variable-width packed ints vs. this prototype's one
uniform record format with fixed 4-byte addresses --
`FST_VS_PACKED_AUTOMATON.md`'s "On-disk / in-memory representation"),
not a dedup-structure question.

High-redundancy (clustered, 2M terms) tells the other half of the
story -- the part this fix was *not* expected to touch:

| terms | `PackedAutomaton` states/term | `fst` states/term |
|---|---|---|
| 200,000   | 0.123 | 0.123 |
| 1,000,000 | 0.176 | 0.127 |
| 2,000,000 | 0.183 | 0.128 |

`PackedAutomaton`'s `states/term` still grows on this corpus (0.123 ->
0.183, +49%) while `fst`'s stays essentially flat (0.123 -> 0.128) --
with the register now bounded and no longer the explanation, this is
clean, direct evidence (not just the earlier isolating experiments)
that the growth here is the *other*, separately-identified mechanism:
`fst`'s output-pushing searching a wider match space than
`PackedAutomaton`'s one-shot output computation
(`AUTOMATON_REFERENCE.md` §4). This is the isolating experiment
`AUTOMATON_REFERENCE.md` §5 asked for and hadn't been run yet --
registry-boundedness and output-pushing really are two independent
causes, now separated by direct measurement rather than argument:
fixing the first left the second's signature untouched.

**Speed, `perf stat`-confirmed** (`cycles:u`, 3 repeated runs each,
tight agreement): single-term insert
(`perf_probe_insert_cost_packed_automaton`) now costs ~707-712M cycles,
down slightly from the pre-fix 729.3M -- the bounded cache's O(`mru_
size`)=O(2) linear bucket scan is cheap, and there's no more incidental
`HashMap` resize/rehash cost at all, since the table is never resized.
Decode (`perf_probe_decode_cost_packed_automaton(_hashlike)`) is
unaffected, as expected -- decode never touches the register, only
`insert`/`compile`/`build_sorted` do.

## releasing PackedAutomaton's registry between mutations

Bounding the register fixed its *growth*, but every block still kept
its whole registry resident for its entire lifetime -- between merges,
when nothing reads it. Measured on a private real-RDF N-Triples corpus
(first ~5M triples, 4.58M distinct terms, default 64..=256 blocks,
`s-bench`'s `nt_dict_file_bench`), with a per-component heap breakdown
of `PackedAutomatonDictionary`:

| component | before | after |
|---|---|---|
| registry table (`table_size * 2` cells, 32 B each) | 944 MiB | 0 |
| registry cells' `transitions` `Vec`s | 504 MiB | 0 |
| `bytes` capacity (`size_bytes` counts only `len`) | 317 MiB (len 217) | = len |
| `id_to_block` capacity | 64 MiB (len ~35) | = len |
| **logical `size_bytes`** | 271 MiB | 282 MiB |
| **retained allocation** | 1,866 MiB (6.9x) | 287 MiB (1.02x) |

`with_capacity_hint` sizes the table at 8 cells/term, i.e. ~216 B/term
of fixed table plus ~115 B/term of cell `Vec`s -- ~5x `bytes` itself.

The fix: `register` is now `Option<PackedRegistry>`.
`PackedAutomaton::release_construction_state` drops it and
`shrink_to_fit`s `bytes`; `PackedAutomatonBlock` calls it after every
`build`, `merge_into` and `remove`. `write_confirmed_state` recreates
the registry on demand (sized from the root's `subtree_count`), so
`PackedAutomaton`'s own API and tests are unchanged. Dictionary-level
slack (`id_to_block`, directory, free lists, scratch arena) is trimmed
by an explicit `PackedAutomatonDictionary::shrink_to_fit` after a bulk
load, not per batch -- trimming a still-growing table per batch would
turn amortized growth into a realloc-and-copy each time.

Cost: a recreated registry starts empty, so new records can't dedup
against records written before the last release -- `bytes` grew 4%
(271 -> 282 MiB), the same class of lost sharing LRU eviction already
accepts. Insertion got slightly *faster* (26.3s -> 24.3s): no large
table to allocate-and-touch per block build. At 1M/10M triples the
retained/logical ratio is 1.03x/1.02x (was 9.2x/7.7x).

## real-RDF decomposition: where PackedAutomatonDictionary's bytes go

Same private real-RDF N-Triples corpus as the previous section (first
10M triples, 6.30M distinct terms, default 64..=256 blocks, one writer).
A scratch instrument took each block's own term set and built it five
ways -- the incremental automaton as the dictionary left it, a fresh
`PackedAutomaton::build_sorted`, an exact-minimal `Automaton::build_sorted`
(unbounded register), `Block::build` (`fst`), and re-emissions of the
fresh graph under candidate encodings (validated: the "current format"
re-emission reproduced the fresh build's byte count exactly).

| layer (bytes/term) | |
|---|---|
| total logical size | 53.7 |
| ids + reverse map + directory (not automaton) | 12.4 |
| dead records (`bytes.len` - reachable records) | 12.9 |
| incremental non-minimality (reachable - fresh build) | 3.0 |
| record encoding (fresh - fst-style re-emission) | 19.0 |
| fst-style encoding of the fresh automaton | 6.4 |
| *`Dictionary` (fst) total / automaton* | *18.8 / 6.7* |

**States/term** on identical term sets: fresh packed 2.722, `fst` 2.717,
exact 2.606, incremental 2.996. Output-pushing contributes nothing
measurable (AUTOMATON_REFERENCE.md §6); the bounded registry costs both
implementations ~4.5%; incremental merges cost packed a further ~10%.

**Shape**: 91% of fresh states have exactly one transition (unique term
tails), which is why per-state overhead dominates.

**Encoding re-emission** (fresh graphs, cumulative, bytes/term): current
25.45; no output on 1-transition states 22.98; no subtree count on
0/1-transition states 20.25; LEB128 backward-delta targets 10.85;
implicit NEXT target 8.49; label folded into header (`fst`'s common
inputs) 6.44. The fixed 4-byte target was the single largest cost.

**Block size** (global sorted terms re-chunked, fresh builds, fst-style
encoding / `fst`): 64: 7.65/8.18, 128: 6.74/7.01, 256: 6.14/6.31,
512: 5.69/5.83, 1024: 5.32/5.44, 4096: 4.81/4.88. Packed blocks average
175 terms (`Dictionary`'s: 214). Incremental insert makes 512-1024-term
blocks affordable for packed in a way they aren't for `fst`.

**Compaction by rebuilding from terms is too slow.** Variants of
`merge_into` (10M triples, one writer): rebuild every merge 227 MiB /
63.9s insertion; rebuild when a block exceeds 1.25x its post-build
bytes/term 240 MiB / 56.3s; 1.5x 254 MiB / 50.2s; 2x 269 MiB / 40.0s;
never 323 MiB / 33.9s (`fst`: 113 MiB / 50.9s). Rebuilding re-derives
the automaton; reclaiming dead records should copy reachable records
instead (still open).

## fst-style record encoding, and insert_prepared

Applied from the decomposition above (`micro_fst.rs`'s `PackedAutomaton`
doc comment has the exact layout): records are addressed by their last
byte (a header) and parsed downward, as in `fst`; the first transition's
output is implicit (`is_final`); targets are reversed-LEB128 deltas from
the field's own position; a NEXT header bit elides a single-transition
record's target when its child is the record immediately below; the
6-bit count escapes to one extra byte at 63 (the old 7-bit field had no
escape at all). **No subtree counts are stored**: `term_count` holds the
root's, and `StagingFrame` children carry counts derived top-down
(`open_frame`: consecutive output differences, the last child takes the
remainder), so `insert`/`insert_sorted` never read a count back from a
record. Label folding was not adopted (see below).

`PackedAutomatonDictionary::insert_prepared` was added (`Dictionary`'s
contract), so batches are sorted/deduplicated on parser workers, as for
`Dictionary`. Previously `nt_dict_file_bench` fell back to
`insert_batch` for packed, putting sorting inside its timed insertion.

Results, same corpus, one writer, logical size / insertion:

| triples (terms) | `Dictionary` (fst) | packed, before | packed, after |
|---|---|---|---|
| 1M (1.01M) | 15.1 MiB / 4.8s | 42.8 MiB / 3.6s | 23.5 MiB / 3.0s |
| 5M (4.58M) | 86.4 MiB / 30.6s | 282.3 MiB / 24.3s | 130.6 MiB / 20.7s |
| 10M (6.30M) | 112.8 MiB / 50.9s | 322.6 MiB / 40.8s | 160.5 MiB / 33.9s |

Dictionary-level lookups (5M triples, 200,000 ids spread over the whole
range, two runs each, agreeing within 2%): `get_term` 1,418 -> 1,159 ns,
`get_id` 1,805 -> 1,484 ns (`fst`: 1,790 / 1,610). Smaller records mean
fewer cache misses at this scale.

In-cache synthetic probes (`perf_probe_decode_cost_packed_automaton*`,
wall clock, 3 runs after a warm-up outlier): clustered `rank` ~165 ->
~110 ns/op, `term_at_into` ~152 -> ~139 ns/op; hashlike `rank` ~190 ->
~190, `term_at_into` ~171 -> ~215 ns/op (+26%) -- the varint parse cost
on wide states when nothing is cache-bound. Single-term `insert` probe
4.19 -> 4.36 us/op.

Open, in expected-value order: copy-compaction of dead records (now
smaller, since dead records shrank with live ones); a packed reverse map
(`Option<(BlockId, u16)>` is 8 bytes/term in both dictionaries); larger
blocks; fixed-width transitions on wide states (Lucene-style, for the
hashlike `term_at_into` regression); label folding (-2.1 bytes/term
fresh, at more branch cost per transition).

## copy-compaction: reclaiming dead records

`PackedAutomaton::compact` re-emits only the reachable records into a
fresh buffer (postorder, children in byte order, as `build_sorted`
writes them, so NEXT targets still apply), deduplicating *exactly* with
a temporary map keyed by `(is_final, rewritten children)`. Because
outputs are a function of each state's right language (AUTOMATON_REFERENCE.md
§6), this bottom-up exact dedup is Revuz-style minimization: the result
is the minimal automaton, tighter than either `build_sorted` or `fst`
(both lose ~4.5% to their bounded registries) --
`packed_automaton_compact_is_exactly_minimal_and_preserves_answers`
asserts equality with the unbounded-register slab build. Cost is
O(reachable records); nothing is retained afterwards.

Trigger sweep (same private real-RDF corpus, 10M triples, one writer;
ratio = block bytes/term vs. its value right after its last fresh build
or compaction, checked after each merge):

| in-load trigger | final compaction | logical size | insertion |
|---|---|---|---|
| never | no | 160.5 MiB | 34.6s |
| 2.0x | no | 138.1 MiB | 35.3s |
| 1.5x | no | 130.7 MiB | 37.3s |
| 1.25x | no | 124.2 MiB | 39.9s |
| 1.1x | no | 122.2 MiB | 45.0s |
| every merge | no | 121.2 MiB | 52.7s |
| never | yes | 120.9 MiB | 38.5s |
| 2.0x | yes | 120.9 MiB | 39.2s |

Final compaction alone reaches the floor in one O(live) pass (~4s here),
cheaper than any in-load ratio that gets close. Chosen: in-load 2.0x
(`COMPACT_GROWTH_RATIO`, bounding dead records during a long load for
~0.7s) plus compaction of every block in
`PackedAutomatonDictionary::shrink_to_fit`.

Result vs `Dictionary` (fst), same corpus (`nt_dict_file_bench`,
insertion includes `shrink_to_fit`):

| run | fst size / insertion | packed size / insertion |
|---|---|---|
| 1M triples, 1 writer | 15.11 MiB / 4.8s | 15.37 MiB / 3.6s |
| 5M triples, 1 writer | 86.41 MiB / 30.5s | 94.40 MiB / 24.3s |
| 10M triples, 1 writer | 112.75 MiB / 50.9s | 120.87 MiB / 38.5s |
| 10M triples, 12 parser workers (wall) | 75.62 MiB / 25.8s | 78.26 MiB / 25.0s |

Lookups after load (10M, 1 writer): `get_term` 1,086 ns vs 1,518;
`get_id` 1,445 vs 1,652. Peak RSS is higher for packed (193 vs 159 MiB
at 10M, 1 writer) -- dead records live until compaction. With 12
workers both dictionaries are writer-bound and packed's insertion lead
mostly disappears (compaction and merges on one thread).

Remaining gap to `fst` (~1.3 bytes/term at 10M): more, smaller blocks
(175 vs 214 terms average) and the unfolded labels (-2.1 bytes/term
fresh, measured). Next targets: reverse-map packing (8 bytes/term, both
dictionaries), larger blocks, label folding, and sharding the writer.

## packing the reverse map

`TermId -> (block, local rank)` was `Vec<Option<(BlockId, u16)>>`: 8
bytes per id after padding, the largest fixed per-term cost in both
`Dictionary` and `PackedAutomatonDictionary` (12.4 bytes/term of
non-automaton overhead in total). `ReverseMap` (`src/reverse_map.rs`)
packs `slot << rank_bits | rank` into one `u32`, `u32::MAX` meaning
unassigned/removed, with `rank_bits` sized for `2 * block_max` (9 bits
at the default 256 -> 23 bits, 8.4M block slots).

Why `2 * block_max`: ranks are briefly assigned to blocks over
`block_max` before their split -- by one term after a single insert, by
under `block_min <= block_max / 2` after a neighbor merge -- and
`Dictionary::split` relies on the left half's ranks already being
correct. `PackedAutomatonDictionary`'s batch path can overshoot by a
whole run, so its `assign_ranks` skips any block over `block_max`
(`split_while_oversized` always follows and assigns every chunk from
rank 0); `oversized_batch_run_is_split_before_ranks_are_assigned`
caught the case where an intermediate half was still oversized.
`ReverseMap::set` asserts both fields fit -- the previous
`local_rank as u16` would have truncated silently.

Same private real-RDF corpus, 10M triples, one writer: `Dictionary`
112.75 -> 88.71 MiB, packed 120.87 -> 96.83 MiB (exactly 4 bytes x
6.30M ids each); insertion and lookups unchanged within noise.

## block size and fill

Sweep after reverse-map packing (10M triples, one writer; bounds
`max/4..=max`):

| block_max | `Dictionary` size / insertion | packed (median halving) | packed (balanced chunks) |
|---|---|---|---|
| 256 | 88.71 MiB / 47.7s | 96.83 MiB / 39.3s | 95.08 MiB / 29.5s |
| 512 | 84.61 MiB / 49.8s | 92.13 MiB / 34.3s | |
| 1024 | 81.74 MiB / 50.2s | 88.44 MiB / 32.3s | 87.24 MiB / 26.3s |
| 2048 | 79.44 MiB / 49.3s | 85.69 MiB / 30.7s | 84.88 MiB / 25.8s |

**Fill**: packed split oversized blocks by repeated median halving,
leaving blocks anywhere between half and completely full (~40% more
blocks than `Dictionary` at every size). `Dictionary::rebuild_batch_block`
cuts a merged run into `ceil(len / block_max)` balanced chunks instead;
packed's `split_while_oversized` now does the same, giving the identical
block layout (29,400 / 7,103 / 3,510 blocks) and much faster insertion
-- one rebuild per chunk instead of one per halving level.

**Size**: larger blocks share more structure (fewer repeated prefix
paths, fewer directory keys) and make lookups faster in packed (`get_id`
1,444 -> 1,155 -> 1,054 ns at 256/1024/2048). The cost is single-term
mutation, which touches a whole block (2,000 absent terms inserted next
to existing ones, then removed):

| block_max | `Dictionary` insert / remove | packed insert / remove |
|---|---|---|
| 256 | 291 / 203 us | 93 / 180 us |
| 1024 | 1,090 / 848 us | 237 / 694 us |
| 2048 | 2,222 / 1,839 us | 431 / 1,424 us |

`Dictionary` rebuilds a block on every change, so its default stays
64..=256. Packed's default is now **256..=1024**: single insert stays
below `Dictionary`'s at its own default; `remove` (still a full block
rebuild for packed) is the price, ~3.4x `Dictionary`'s.

Result at each dictionary's default, same corpus:

| run | `Dictionary` size / insertion | packed size / insertion |
|---|---|---|
| 1M triples, 1 writer | 11.25 MiB / 4.8s | **9.78 MiB / 2.6s** |
| 5M triples, 1 writer | 68.94 MiB / 30.6s | 69.18 MiB / **16.4s** |
| 10M triples, 1 writer | 88.71 MiB / 50.5s | **87.24 MiB / 26.0s** |
| 10M triples, 12 parser workers (wall) | 58.84 MiB / 26.8s | **55.42 MiB / 18.2s** |

Lookups (10M, 1 writer): `get_term` 1,068 vs 1,551 ns, `get_id` 1,158 vs
1,643 ns. Single-term insert 237 vs 276 us; remove 710 vs 215 us. Peak
RSS 150 vs 134 MiB (dead records between compactions). Next: an
incremental packed `remove`, label folding (~-2 bytes/term fresh), and
sharding the writer.

## incremental single-term mutation

Starting point (10M triples, packed at 256..=1024): single-term insert
237 us, remove 710 us (`nt_dict_file_bench`'s probe: 2,000 absent terms
inserted next to existing ones across the id range, then removed). Four
causes, fixed in turn:

1. **`remove` rebuilt the block** (decode every term, `build_sorted`).
   `PackedAutomaton::remove` is now `insert`'s mirror: open the path's
   frames (counts derived top-down), clear the final flag, prune states
   left with no terms, re-confirm bottom-up. Dead records go to the usual
   growth-triggered compaction. 710 -> 31 us.
2. **`merge_into` streamed the whole block** to place even one term.
   `PackedAutomaton::lower_bound` (one descent: terms before `term`, and
   whether it's present) gives each incoming term's old rank; runs of at
   most half the block (`SMALL_RUN_FACTOR = 2`) use it. Bulk loads benefit
   too -- a 65,536-triple batch spread over thousands of blocks lands a
   few terms in each. Threshold sweep, bulk insertion: stream-only 27.7s,
   1/32: 26.4s, 1/16: 25.9s, 1/8: 23.5s, 1/4: 22.5s, 1/2: 22.3s.
3. **Each mutation recreated a block-sized registry** (`with_capacity_hint`
   on the block's term count: ~7,000 cells, ~230 KB, allocated and
   initialized per insert). A recreated registry can only dedup records
   written during the same mutation, so `reserve_registry` now sizes it to
   the run. Remove 31 -> 18 us; insert 149 -> 131 us.
4. **Full blocks split.** Balanced chunking filled bulk-loaded blocks to
   `block_max`, so 651 of 2,000 probe inserts (33%) split a block -- the
   classic B-tree bulk-load effect. `SPLIT_FILL_PERCENT = 90` chunks to 90%
   of `block_max`: 0 splits, insert 131 -> 21.5 us, logical size +0.4%
   (87.24 -> 87.56 MiB), bulk insertion slightly faster. 75% bought
   nothing further (23.3 us, +1.1%).

New tests: `packed_automaton_remove_and_lower_bound_match_model` (3,000
random inserts/removes against a sorted-set model, with compactions;
exact minimality at the end) and `mixed_mutations_match_model` (single
inserts, small and large batches, removes with neighbor merges, at tiny
block bounds).

Final, each dictionary at its default, same corpus:

| run | `Dictionary` (fst) | packed |
|---|---|---|
| 1M, 1 writer: size / insertion | 11.25 MiB / 4.9s | **9.83 MiB / 2.4s** |
| 5M, 1 writer | 68.94 MiB / 32.6s | 69.47 MiB / **13.8s** |
| 10M, 1 writer | 88.71 MiB / 47.7s | **87.56 MiB / 22.1s** |
| 10M, 1 writer: get_term / get_id | 1,550 / 1,703 ns | **1,089 / 1,269 ns** |
| 10M, 1 writer: single insert / remove | 296 / 219 us | **22.5 / 19.9 us** |
| 10M, 12 workers: wall time | 32.0s | **21.1s** |

(With a triple cap and 12 workers the writer stops at whichever batches
arrive first, so the two 12-worker runs held different term sets -- 4.40M
vs 4.29M -- and their sizes aren't comparable.) Peak RSS at 10M, 1 writer:
144 vs 134 MiB.

## label folding

The last measured encoding step (fresh-graph re-emission: -2.05
bytes/term). Single-transition, non-final records -- 91% of states,
mostly unique term tails -- now have their own header form, `1 N cccccc`:
N is NEXT, `cccccc` the label's code in a 63-entry table (0: label byte
stored below). All other records use `0 F nnnnnn` (final flag, count with
the existing escape) and no longer carry NEXT -- a single-transition
*final* state is rare (dozens in 6.3M terms), so nothing measurable was
lost.

Which 63 labels (same private real-RDF corpus, 10M triples, one writer):

| table | logical size | insertion |
|---|---|---|
| none (new header layout only) | 87.56 MiB | 21.9s |
| `fst`'s `COMMON_INPUTS` (general text; no `<>"#@`) | 78.01 MiB | 22.0s |
| curated RDF (`RDF_LABELS`: digits, a-z, `<>"/:.#-_@%^`, space, 14 capitals) | **77.02 MiB** | 22.3s |
| adaptive: top 63 bytes of the first batch's term text | 77.74 MiB | 22.6s |

The adaptive table lost because term-text byte frequency isn't
single-transition-label frequency. The curated table is a constant with
nothing derived from any dataset. (The public W3C `rdf-tests` corpus was
considered as a source and rejected: 159 KB, missing common bytes such as
`7`, `k`, `z`, and full of test-specific punctuation.)

Result vs `Dictionary` (fst) at each default, one writer:

| triples | `Dictionary` | packed |
|---|---|---|
| 1M | 11.25 MiB | **9.65 MiB** (-14%) |
| 5M | 68.94 MiB | **59.16 MiB** (-14%) |
| 10M | 88.71 MiB | **77.02 MiB** (-13%) |

Peak RSS at 10M: 131 MiB vs `Dictionary`'s 134. Lookups: `get_term`
1,116-1,156 ns over two runs (1,064-1,089 before folding; `Dictionary`
1,550), `get_id` 1,138-1,195 ns. Insertion and single-term mutation
unchanged (22.2s; ~20 / ~18 us).

Remaining: sharding the single writer by key range (both dictionaries
are writer-bound with 12 parser workers), and fixed-width transitions on
wide states for the in-cache hashlike `term_at_into` regression.

## sharding the writer

Every dictionary here has one writer. With 12 parser workers both were
writer-bound (`Dictionary` 48.3s, packed 22.3s wall for the first 10M
lines of the same private real-RDF corpus, versus 57.7s / 29.5s fully
sequential). `ShardedDictionary<D>` (`src/sharded.rs`) holds N independent
dictionaries, each owning one key range: `shard_of` is a binary search
over N-1 bounds, a global `TermId` carries the shard in its top bits
(`global_id`/`split_id`, asserting the local id fits), and it implements
`TermDictionary` like the others. `router()` gives parser workers an owned
copy of the routing while writer threads hold `shards_mut()`.
`s-bench`'s `nt_dict_file_bench` drives it: parsers route each term into
per-shard `TermBatch`es, and each writer thread inserts its shards' batches.

What it took to make that scale, in order:

1. **Comparable inputs.** A running triple cap with parallel workers stops
   at whichever batches arrive first, so configurations loaded different
   term sets. Capped runs now parse a byte prefix (the first N lines, found
   before timing): every configuration loads the same 6,274,377 terms.
2. **Balanced shards.** Bounds at quantiles of a 65,536-triple sample put
   49% of the terms in one of 4 shards. A sample includes a distinct term
   with probability proportional to its occurrences, so regions dense in
   once-used terms are under-weighted. Taking quantiles over the sample's
   singletons (Good-Turing) didn't fix it, nor did reshaping the sample
   (64x1024, 4096x16, 16384x4, 65536x1 triples: the same 1.82M-term first
   shard of 8). An unbiased distinct sample would need a full pre-pass.
   Instead: 32 fine shards per writer, dealt round-robin, which spreads any
   dense region over every writer. That also absorbed a second imbalance:
   at equal term counts, per-shard insertion cost varied ~7x (correlation
   with term count 0.69, with occurrences or bytes ~0) -- consistent with
   key regions arriving in random order touching every block each batch --
   and heavy shards cluster in key order. Busy time per writer at 8 shards
   per writer: 3-14s; at 32: within 15% of each other.
3. **Removing the bench's own contention.** The counting allocator updated
   one shared atomic on every allocation; with 8+ threads allocating, that
   cache line more than doubled writer time (6 parsers + 6 writers: 14.0s
   -> 6.6s wall when switched off). It now counts only bytes freed while
   the finished dictionary is dropped -- nothing during timed phases.
   (Earlier multi-threaded numbers carried this overhead for both
   dictionaries alike; single-threaded ones ~6%.)
4. **Bounded queues.** Per-writer queues of `2 x parsers x 32` sub-batches
   buffered ~8 parser batches per writer (1.3 GB peak RSS). One parser
   batch per writer: 0.50 GB peak, ~15% slower.

Result, first 10M lines (6.27M terms), wall time:

| parsers + writers | `Dictionary` (fst) | packed |
|---|---|---|
| 1 + 1 (sequential) | 57.7s | 29.5s |
| 12 + 1 | 48.3s | 22.3s |
| 8 + 4 | 15.6s | 7.5s |
| 6 + 6 | 13.6s (13.5s bounded queues) | 6.9s (7.4s) |
| 4 + 8 | 11.9s (12.8s) | 6.0s (6.9s) |

(6-core / 12-thread laptop CPU; all-core clocks are well below its
single-core boost, so per-thread work inflates as threads are added.)
Logical size is unchanged by sharding (packed 77.0 MiB, fst 88.6-88.8);
retained allocation +3.7 MiB for packed at 256 shards (per-shard tables).
Lookups unchanged except `get_id`'s routing step (packed 1,117 -> 1,270
ns); single-term insert 19 -> 25 us. Peak RSS with bounded queues:
packed 499 MiB, fst 464 MiB (the 12 + 1 runs: 599 / 580).

## block id encoding

Each block's `rank -> TermId` array was a plain `Vec<TermId>`: 4 bytes per
term, a third of packed's size. In a bulk load each batch assigns new ids
in sorted key order and a block covers a contiguous key range, so the
terms one batch adds to one block get consecutive ids that increase with
rank: a block's ids are a few such runs, interleaved. Measured on the full
private real-RDF corpus (6.88M terms, 8,847 blocks): 11.1 runs per block,
mean length 70, 2% singleton runs, 27% of blocks a single run, and 100% of
runs increasing with rank. Candidate encodings (bytes/term):

| encoding | bytes/term |
|---|---|
| raw `u32` | 4.00 |
| global bit-packing (23 bits) | 2.88 |
| FOR per block | 2.18 |
| FOR per 128 (BP128-style) / PFOR per 128 | 2.04 / 2.04 |
| delta + zigzag FOR per 128 | 1.91 |
| run table: `bases[run] + offset` | 1.38 |
| run table, single-run blocks implicit | **0.98** |
| run index only, offset by counting (not O(1)) | 0.39 |

The FOR family sees the interleaving as noise. `BlockIds`
(`src/block_ids.rs`) implements the 0.98 row: per block, the smallest of
`Implicit` (`id = base + rank`), `Runs` (bit-packed `(run, offset)` plus a
base table) and `For` (block minimum plus bit-packed offsets, the fallback
for single-term inserts or reused ids). Runs are found greedily in rank
order -- an id continues the open run expecting it, or starts one -- so
every run is monotone and offsets are exact for any id order; only the
size depends on structure. Access is O(1). Merges and removes decode,
edit and re-encode (both were already O(block)).

Full file, same corpus (`nt_dict_file_bench`, 36.2M triples, 6.88M terms):

| | `Dictionary` (fst) | packed, before | packed, now |
|---|---|---|---|
| logical size, sequential | 93.81 MiB (14.30 B/term) | 81.44 MiB | **61.76 MiB (9.41 B/term)** |
| logical size, 4 parsers + 8 writers | 94.48 MiB | 81.58 MiB | **63.58 MiB** |
| wall time, 4 + 8 | 24.5s | 15.4s | **13.4s** |
| insertion, sequential | 81.8s | 38.2-39.0s | **31.5s** |
| get_term / get_id, sequential | 1,513-2,191 / 1,734-2,296 ns | 995 / 1,111 ns | 986 / 1,110 ns |
| single-term insert / remove | 242-461 / 217-346 us | 19-25 / 17-18 us | 29-36 / 27-28 us |
| peak RSS, sequential | 143 MiB | 143 MiB | 129 MiB |

(`Dictionary`'s ranges span this and the previous full-file run; its code
didn't change, the spread is this laptop's thermal/page-cache state.)
Packed is now 34% smaller than `Dictionary`. The cost: single-term
mutations re-encode a block's ids (~+10 us). The sharded layout is ~2 MiB
larger (256 shards cut some runs, and each shard has its own id space).

## reverse map encoding

`ReverseMap` (`TermId -> (block slot, local rank)`, shared by `Dictionary`
and packed) was one `u32` per id: 4 bytes/term, packed's largest fixed
cost after block-id compression. The same batch structure applies: a
batch's ids for one block are consecutive with increasing ranks, so 64
consecutive ids fall in few blocks. Measured candidates on the full
private real-RDF corpus (6.88M ids; 98,186 id runs, 70 ids each -- exactly
`BlockIds`' runs seen from the other side):

| reverse map | bytes/term | id -> term |
|---|---|---|
| packed `u32` | 4.00 | O(1) |
| global bit-packing (14 slot + 10 rank bits) | 3.00 | O(1) |
| per-64-id chunk: slot table + rank FOR + live bit | 1.41 | O(1) |
| per-128 / per-256 chunks | 1.49 / 1.60 | O(1) |
| id-run range map + rank deltas | 0.67 | binary search + prefix sum |
| id-run range map, ranks via `BlockIds` select | 0.08 | O(block) scan |

Chosen: per-64-id chunks (O(1), local maintenance). Ranks shift on every
merge, so writes go through `assign` (sort by id, rewrite each touched
chunk once): the full load made 51,876 rank-assignment calls touching
1.11M chunks (~21 per call), so per-call batching -- not per-entry
rewrites -- is what keeps this cheap. Chunks live in one `u64` pool,
rewritten in place when they fit, else appended; garbage is compacted past
half the pool and in `shrink_to_fit`. Per-chunk FOR ranks also removed the
old `2 x block_max` rank-bit limit.

Full file (36.2M triples, 6.88M terms; `nt_dict_file_bench`):

| | `Dictionary` (fst) before / now | packed before / now |
|---|---|---|
| logical size, sequential | 93.81 / **79.41 MiB** | 61.76 / **46.38 MiB (7.07 B/term)** |
| logical size, 4 parsers + 8 writers | 94.48 / 80.20 MiB | 63.58 / 48.50 MiB |
| insertion, sequential | 81.8 / 81.5s | 31.5 / 35.1s |
| wall time, 4 + 8 (warm page cache) | 24.5 / 25.5s | 13.4 / 15.9s |
| get_term / get_id, sequential | 1,513-2,191 / 1,530-2,296 ns | 986 / 1,110 -> 1,034 / 1,125 ns |
| single-term insert / remove | 242-461 / 217-346 us -> 296 / 231 us | 29-36 / 27-28 -> 52-57 / 50 us |

Packed is now 42% smaller than `Dictionary` (46.4 vs 79.4 MiB); the
reverse map went from 4 to ~1.66 bytes/term (headers and 32-bit slot
table entries above the 1.41 estimate). Costs: each rank reassignment
re-encodes ~21 chunks (writer time +25% sharded, sequential insertion
+3.6s, single-term mutations +~23 us); lookups ~+5%. (One 4 + 8 run read
the file with a cold page cache: 30.3s wall, parsers' summed time 94.9s
instead of ~43s, writers unchanged; the warm rerun is reported.)

## static FST reference

To see what a static structure would cost, `s-bench`'s `static_fst_bench`
loads the full private real-RDF corpus into a `Dictionary`, reads every
term back, and builds single `fst` structures over all 6.88M terms with the
same vendored crate. A `fst::Map` from each term to the dictionary's own
(arbitrary) `TermId` takes 8.75 bytes/term and answers only term -> id; it
is immutable. Packed, with the same kind of ids, is now 7.07 bytes/term
while also answering id -> term and supporting insert and remove. The
same map with *sorted ranks* as values takes 4.22 bytes/term (an
`fst::Set`, 3.79): values that follow key order keep states shareable,
while arbitrary values, pushed toward the root, block suffix sharing.
That is the split packed already makes: its automata carry only local
rank outputs (~4.3 bytes/term, close to the rank-valued map), and the
arbitrary ids live beside them in `BlockIds` and the reverse map, encoded
from their run structure.

## ranks out of the reverse map

Ranks shift on every merge, but the block an id lives in changes only for
new ids, splits and neighbor merges. Storing ranks in the reverse map
meant rewriting them on every merge (51,876 calls, 1.11M chunk rewrites in
the full load). Packed now splits the two: `SlotMap` (`src/slot_map.rs`)
stores only `id -> block slot`, in the same 64-id chunk layout, written
only when an id changes block; `BlockIds::rank_of` answers `id -> rank`
from an inverse rebuilt whenever the block re-encodes its ids (every merge
already did). The inverse, per kind: `Implicit` `id - base`; `Runs` -- runs
relabelled in base order, binary search over the bases, then a rank sample
every 16 entries plus bit-packed deltas; `For` -- ranks in id order,
binary-searched. `Dictionary` (fst) keeps the rank-storing `ReverseMap`:
its blocks have no `BlockIds`.

Measured first (full private real-RDF corpus): slot-only map 0.49
bytes/term; per-run ranks as FOR per run 1.28, deltas + sample every 16
0.69, Elias-Fano 0.57 (select scan). Chosen: deltas + samples, ~1.2 total
vs 1.66 before.

Two things found on the way. The full-file sequential load briefly holds
a whole batch run in one block before splitting it, so a run can exceed
65,535 entries: the first descriptor layout (16-bit run length) panicked;
it is now 26 bits (`runs_longer_than_u16` covers it). And `get_term` got
~30% slower: the sample interval barely mattered (every 4/8/16: 1,392 /
1,398 / 1,437-1,456 ns at 10M lines), the extra dependent heap objects
did -- `BlockIds` had four boxed sections; putting them in one allocation
recovered ~5% (1,366-1,371 ns). The rest is inherent to computing the
rank: one more cache miss into the block's id data plus the search.

Full file (36.2M triples, 6.88M terms), packed before -> now (`Dictionary`
unchanged):

| | packed before | packed now | `Dictionary` (fst) |
|---|---|---|---|
| logical size, sequential | 46.38 MiB | **44.01 MiB (6.71 B/term)** | 79.41 MiB |
| logical size, 4 + 8 | 48.50 MiB | **46.62 MiB** | 80.16 MiB |
| insertion, sequential | 35.1s | **33.1s** | 79.7s |
| wall time, 4 + 8 | 15.9s | **14.0s** | 26.3s |
| single-term insert / remove | 52 / 50 us | **35 / 33 us** | 302 / 239 us |
| get_term, sequential / 4 + 8 | 1,034 / 1,097 ns | 1,299 / 1,438 ns | 1,490 / 1,596 ns |
| get_id, sequential / 4 + 8 | 1,125 / 1,272 ns | 1,199 / 1,301 ns | 1,616 / 2,051 ns |
| peak RSS, 4 + 8 | 441 MiB | 387 MiB | 461 MiB |
