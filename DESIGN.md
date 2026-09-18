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

## What's implemented (Stages 0, 1, 3, plus one §13 baseline)

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
(block rebuild triggers, split/merge thresholds, Variant A reverse
mapping, the `fst` crate as F_b's base representation) -- it's entirely
about the DoD shape of scratch/temporary data and field widths, and the
correctness net (the 3-way differential fuzz suite, extended with a
`block_id_niche_optimization_holds`-style assertion per type) passed
clean on the first full rewrite.

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
- **Stage 5**: Variant A vs. Variant B reverse-mapping evaluation (§5),
  to test **H3**.
- **Stage 6**: physical FST layout optimization from profiling (contiguous
  state/transition arrays, separated hot/cold block metadata -- §10),
  to test **H4**.
- **Stage 7 and beyond**: RDF-aware term decomposition (§9: term-kind
  prefixes, structural typed-literal encoding, language-tag dictionaries,
  blank-node encodings), FSST as an alternative block codec, compressed
  external-id permutations, SIMD-friendly layouts, the remaining §13
  baselines (front-coded blocks is now done, see above; still missing: a
  global static minimal FST, `m-Bonsai`/Dynamic-Path-Decomposed-Trie-style
  dynamic tries, global FST + compressed permutation), and the full
  measurement suite from §14 (latency percentiles, cache/branch misses,
  allocation and peak-memory tracking during rebuilds -- bits/term now has
  one real data point, see above, but not the systematic `B` sweep or a
  real RDF corpus).

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
