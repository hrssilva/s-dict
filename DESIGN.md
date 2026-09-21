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
