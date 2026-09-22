# s-dict

A dynamic, incrementally mutable, bidirectional RDF term `<-> TermId`
dictionary. Terms are partitioned into contiguous lexical blocks, each an
independently rebuildable index; a small global router and reverse
directory send lookups to the right block, so an insertion or deletion
touches at most one or two blocks rather than the whole dictionary.

It's the sibling of [`s-graph`](../s-graph): `s-graph` indexes `(u32,
u32, u32)` triples but doesn't intern terms itself; `s-dict` is where real
IRIs/literals get dictionary-encoded into the `u32` ids `s-graph` expects.

## Status

This crate currently implements Stages 0, 1, and 3 of a larger design,
plus one of the required §13 comparison baselines (see [DESIGN.md](DESIGN.md)
for the full architecture, roadmap, and literature grounding):

- **`PlainDictionary`** -- a plain `HashMap`/`Vec` baseline. Kept
  permanently as the comparison point later stages get measured against,
  not a throwaway.
- **`Dictionary`** -- lexical blocks with a sorted-array router and
  split/merge on overflow/underflow, where each block's forward index is a
  minimal per-block FST (via the [`fst`](https://docs.rs/fst) crate) --
  the "micro-FST" the overall architecture is named for, with true
  O(term length) rank *and* select (via `fst`'s `raw::Fst::get_key`, not
  a linear scan). This is the crate's primary, general-purpose
  implementation.
- **`FrontCodedDictionary`** -- the same block/router/split-merge
  structure, but each block is shared-prefix delta-coded with restart
  points instead of an FST. Exists specifically to isolate what automaton
  minimization buys over blocking + prefix compression alone (design doc
  §8). Measured against `Dictionary` on 9,000 namespace-clustered terms:
  the FST is ~1.4x more space-efficient (13.84 vs 22.07 bytes/term), but
  front coding is faster to rebuild per insert and at both lookup
  directions at this block size -- see DESIGN.md for the full comparison
  and how to reproduce it.
- **`AutomatonDictionary`** -- the same block/router/split-merge structure
  again, `AutomatonBlock` (a from-scratch, incrementally mutable minimal
  automaton -- `src/micro_fst.rs`, no dependency on the `fst` crate) in
  place of `Block`. Exists to test whether a custom automaton that
  supports true incremental insert/remove (`perf`-confirmed ~3x faster
  single-term insert, ~2x faster decode, ~16-26x faster batch insert than
  `Block`'s full-rebuild-per-mutation contract -- see DESIGN.md) is worth
  the added complexity over reaching for the well-tested `fst` crate.
  Currently a smaller-scope comparison implementation (no prepared-batch
  or buffer-reusing decode APIs yet); `Dictionary` remains the crate's
  primary, general-purpose implementation unless and until this one earns
  that spot on the numbers -- see DESIGN.md's "custom incremental
  micro-FST" sections for the full history, including two real bugs this
  effort found and fixed by writing failing tests *before* trusting a fix,
  not after.

The first three implementations have also been through a
Data-Oriented-Design optimization pass (DESIGN.md's "Optimization pass"
section): a struct-of-arrays scratch buffer (`EntriesArena`) replacing a
`Vec<(Box<str>, TermId)>` that allocated per-term on every block rebuild,
a buffer-reusing `get_term_into` alongside the allocating `get_term`
(confirmed with `perf`-measured hardware counters on both synthetic *and*
real-corpus data -- see DESIGN.md), `u16` instead of `u32` for a field
genuinely bounded by a single term's length, and a `NonZeroU32`-backed
`BlockId` for a niche optimization. `AutomatonDictionary` shares the
structural pieces (`EntriesArena`, the `u16`/`NonZeroU32` reverse-mapping
layout) but not yet `get_term_into`/`get_terms_into` -- one of the scope
reductions named above. `Dictionary`/`FrontCodedDictionary`
also moved from design doc §5's Variant A reverse mapping to Variant B
(`TermId -> (BlockId, local_rank)`, no scan) after measuring both --
faster reverse lookups, more reverse-directory space, exactly the
tradeoff §5 names, now with real numbers on each side.

`Block::term_at_into` also walks the FST's `raw` API directly with a
binary search instead of calling the `fst` crate's own `get_key_into` (a
linear scan) -- a `perf record` call-graph profile on synthetic
high-branching-factor keys found that scan's generic-`Iterator`
overhead, not per-transition data decoding, was 41.5% of that function's
runtime. **A follow-up profile on real RDF terms (the W3C test corpus,
via a sibling `s-bench` profiling harness) found this doesn't generalize**:
real IRIs have low branching factor at most nodes (long shared literal
path segments dominate), so there's rarely more than one or two
candidates to search among, and `perf annotate` showed the true cost is
`node.transition(idx)`'s bit-unpacking itself, paid once per transition
regardless of how it was found. The binary search is kept (it's never
worse, and is a real backstop on pathological data) but isn't the actual
lever for real-world `get_term` cost -- see DESIGN.md's "Correction"
subsection for the full story, including why that matters more than the
result itself (a synthetic benchmark quietly encoded an assumption that
doesn't hold for real RDF terms, and only a real-corpus measurement
caught it). What *does* generalize to real data: the `get_term`/
`get_term_into` allocation gap, and a new finding -- `Node::new` (walking
to each node while decoding) is a distinct, substantial cost, motivating
`get_terms_into`: a bulk decode entry point that decodes each affected
block once (reusing the existing streaming machinery) instead of
re-walking the FST from the root once per id, for callers resolving many
ids at once (dictionary dumps, or any workload where requested ids
cluster within blocks). Measured on a 2,000,000-term dictionary:
**3.8x faster** than repeated `get_term_into` calls when dumping
(nearly) everything, and, just as expected and just as important to
report, **~19x slower** for ids scattered one-per-block -- `get_terms_into`
decodes a whole block to serve a single id from it in that case, exactly
the documented tradeoff, not a special case to hide. See DESIGN.md for
both numbers and how to reproduce them.

A non-trivial router, RDF-aware term decomposition, and the remaining §13
baselines (global static minimal FST, dynamic compact tries, global FST +
compressed permutation) are explicit future stages, not yet implemented --
see DESIGN.md's "What's not implemented yet".

## Install

```toml
[dependencies]
s-dict = { path = "../s-dict" }  # or a git dependency, once published
```

One dependency: [`fst`](https://docs.rs/fst), for each block's per-term
finite-state map.

## Quick start

```rust
use s_dict::{Dictionary, TermDictionary};

let mut dict = Dictionary::with_block_bounds(64, 256); // block_min, block_max

let type_id = dict.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
let class_id = dict.insert("http://www.w3.org/2000/01/rdf-schema#Class");

// Idempotent: re-inserting an existing term returns its existing id.
assert_eq!(dict.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), type_id);

// Bidirectional lookup. get_term returns an owned String: a term decoded
// from a block's FST can't be borrowed past the call (see DESIGN.md).
assert_eq!(dict.get_id("http://www.w3.org/2000/01/rdf-schema#Class"), Some(class_id));
assert_eq!(
    dict.get_term(class_id).as_deref(),
    Some("http://www.w3.org/2000/01/rdf-schema#Class")
);

// get_term_into reuses a caller buffer instead of allocating a fresh
// String per call -- for hot loops materializing many terms.
let mut buf = Vec::new();
dict.get_term_into(class_id, &mut buf);
assert_eq!(std::str::from_utf8(&buf).unwrap(), "http://www.w3.org/2000/01/rdf-schema#Class");

// get_terms_into decodes many ids in one call, one streaming pass per
// affected block instead of one FST walk per id -- for bulk lookups
// (dictionary dumps, or any workload where requested ids cluster within
// a block, e.g. namespace-clustered real RDF data).
let mut terms = Vec::new();
dict.get_terms_into(&[type_id, class_id], &mut terms);
assert_eq!(terms[0].as_deref(), Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"));

// Batch insertion groups new terms by destination block instead of
// rebuilding a block once per term.
let ids = dict.insert_batch(&["http://dbpedia.org/resource/Berlin", "http://dbpedia.org/resource/Paris"]);

dict.remove("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
assert_eq!(dict.get_id("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), None);
# let _ = ids;
```

For ingestion with parser workers, prepare a reusable owned batch before
passing it to the single dictionary writer. Preparation sorts indices and
stores duplicate positions in flat arrays. Returned IDs retain input order.

```rust
use s_dict::{Dictionary, TermBatch};
let mut batch = TermBatch::new();
batch.push("example");
batch.push("example");
batch.prepare(); // Can run on a parser worker, independently of the dictionary.
let mut dict = Dictionary::new();
let ids = dict.insert_prepared(batch.prepared().unwrap());
assert_eq!(ids[0], ids[1]);
batch.clear(); // Reuse text and metadata capacity for the next bounded batch.
```

`push` and `clear` invalidate preparation. The borrowed `PreparedBatch` view
prevents modification during insertion. Callers choose the batch size; the
library does not collect the entire input. Ordinary `insert_batch` remains
available with borrowed strings.

FST construction uses a block-sized state cache from a small vendored patch
to `fst` 0.4.7 (see `vendor/fst/LOCAL_CHANGES.md`). The encoding and lookups
are unchanged; reduced cache reuse can slightly increase compressed size.

`TermId` is `u32`-width, matching `s-graph`'s subject/predicate/object id
space directly.

`Dictionary::new()` uses default block bounds (`64..=256`); block size is
a first-class tunable parameter (`with_block_bounds`), not a fixed
constant -- see DESIGN.md §12 for why.

## Project layout

- `src/lib.rs` -- the `TermId` and `TermDictionary` public API.
- `src/plain.rs` -- `PlainDictionary` (Stage 0 / permanent plain baseline).
- `src/dictionary.rs` -- `Dictionary` (Stage 1: lexical blocks, router,
  split/merge).
- `src/block.rs` -- the per-block micro-FST (Stage 3): builds, decodes,
  and queries each block's `fst::Map`.
- `src/front_coded_dictionary.rs`, `src/front_coded_block.rs` --
  `FrontCodedDictionary`, the §13 front-coded-blocks baseline: the same
  macro structure as `Dictionary`, with shared-prefix delta coding
  instead of an FST.
- `src/entries_arena.rs` -- `EntriesArena`, the struct-of-arrays scratch
  buffer both `Dictionary` and `FrontCodedDictionary` decode into and
  rebuild from, instead of a `Vec<(Box<str>, TermId)>` that allocated
  per term on every block rebuild.
- `src/micro_fst.rs` -- `Automaton`, a from-scratch, incrementally
  mutable minimal acyclic finite-state automaton (build/insert/
  insert_sorted/remove/rank/term_at_into), no dependency on the `fst`
  crate; see DESIGN.md's "custom incremental micro-FST" sections for the
  literature grounding and every measurement.
- `src/automaton_block.rs`, `src/automaton_dictionary.rs` --
  `AutomatonDictionary`, the same macro structure as `Dictionary` again,
  `AutomatonBlock` (pairs `Automaton` with the external `TermId` array
  `Block` also carries) in place of `Block`.
- `tests/dictionary.rs` -- differential fuzz testing of `Dictionary`,
  `FrontCodedDictionary`, and `AutomatonDictionary` against
  `PlainDictionary` as an oracle, in lockstep, across several block-size
  configurations chosen to force splits and merges; plus `#[ignore]`d
  perf/space probes (run with `cargo test --release -- --ignored
  --nocapture`).
- `examples/perf_get_term.rs`, `examples/perf_decode_bulk.rs` -- dedicated
  `perf stat` targets isolating `get_term`/`get_term_into` from setup cost
  (via a `setup_only` baseline to subtract), for hardware-counter-level
  measurement (cycles, instructions, cache/branch misses) beyond what the
  wall-clock probes show -- the first at 20,000 terms/many repeats, the
  second in a single 2,000,000-term pass, checking whether the same
  per-op advantage holds (it does, and gets cleaner) at bulk-decode
  scale -- see DESIGN.md for how to run them and what they found.
- `examples/batch_sweep.rs` -- sweeps `TermBatch` flush thresholds (term
  count and byte size) against `Dictionary::rebuild_stats()`, wall time,
  retained allocation (via a counting global allocator), and RSS, to
  measure whether larger bounded batches reduce block-rebuild frequency
  and by how much -- see DESIGN.md's "Bounded-batch sweep" section.
- `examples/perf_insert_single_flush.rs` -- a focused `perf stat` target
  for single-term-flush insertion into a preloaded dictionary; hardware
  counters (cycles, instructions, cache-misses), not wall time, are what
  actually settled whether the single-walk merge refactor was a real win
  -- see DESIGN.md's "Single-walk merge" section for why wall-clock alone
  was misleading here.

## Testing

```sh
cargo test
cargo clippy --all-targets
```

## License

MIT -- see `LICENSE`.
