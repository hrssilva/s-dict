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

`F_b` is eventually meant to be a minimal per-block finite-state
dictionary (a "micro-FST"), immutable between rebuilds -- Stage 3 in the
roadmap below. **Not yet implemented**: today `F_b` is a plain sorted
`Vec<Box<str>>`, searched with binary search. This still gives exact
rank/select semantics; it just isn't compressed yet.

## What's implemented (Stages 0-1)

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

## What's not implemented yet

In the design doc's own stage numbering (§17), this crate currently
implements Stage 0 and (a slightly pulled-forward) Stage 1. Left for
later, deliberately, per §11's optimization discipline and §16's
anti-goals -- do not build compression machinery before measuring whether
it's needed:

- **Stage 2 proper**: a non-trivial router (Patricia/radix tree, or a
  compact B/B+-tree over separators) if the sorted-array router turns out
  to be a bottleneck at large block counts. Not assumed necessary yet
  (§4: "do not assume which router is best before measurement").
- **Stage 3**: replace each block's plain sorted `Vec<Box<str>>` with a
  minimal DAFSA/FST supporting rank and select -- the actual "micro-FST"
  the architecture is named for.
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
  external-id permutations, SIMD-friendly layouts, the full multi-baseline
  benchmark suite from §13 (front-coded blocks, a global static minimal
  FST, `m-Bonsai`/Dynamic-Path-Decomposed-Trie-style dynamic tries, global
  FST + compressed permutation), and the full measurement suite from §14
  (bits/term, latency percentiles, cache/branch misses, allocation and
  peak-memory tracking during rebuilds).

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
  ordered incremental construction -- directly relevant to Stage 3, since
  every block rebuild already receives its terms pre-sorted.
- Carrasco, Forcada. *Incremental Construction and Maintenance of Minimal
  Finite-State Automata*. CL 28(2), 2002. Addition/deletion while
  preserving minimality -- background for why this design deliberately
  avoids needing that (rebuild the block instead).
- Cognetta, Allauzen, Riley. *On the Compression of Lexicon Transducers*.
  FSMNLP 2019. Compact static FST encodings -- the Stage 3/6 target once
  compression is added.
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
  choices in `Dictionary` (§10): compact arrays and stable integer handles
  now, contiguous packed state/transition arrays once Stage 3 exists.
