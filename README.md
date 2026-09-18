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
  the FST is ~1.4x more space-efficient (9.84 vs 18.07 bytes/term), but
  front coding is ~6x faster to rebuild per insert and faster at both
  lookup directions too at this block size -- see DESIGN.md for the full
  comparison, why the lookup result is the more surprising one, and how
  to reproduce it.

All three implementations have also been through a Data-Oriented-Design
optimization pass (DESIGN.md's "Optimization pass" section): a
struct-of-arrays scratch buffer (`EntriesArena`) replacing a
`Vec<(Box<str>, TermId)>` that allocated per-term on every block rebuild,
a buffer-reusing `get_term_into` alongside the allocating `get_term`,
`u16` instead of `u32` for a field genuinely bounded by a single term's
length, and a `NonZeroU32`-backed `BlockId` so `Option<BlockId>` costs 4
bytes instead of 8 via Rust's niche optimization -- each change measured
before and after, not assumed.

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

// Batch insertion groups new terms by destination block instead of
// rebuilding a block once per term.
let ids = dict.insert_batch(&["http://dbpedia.org/resource/Berlin", "http://dbpedia.org/resource/Paris"]);

dict.remove("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
assert_eq!(dict.get_id("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), None);
# let _ = ids;
```

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
- `tests/dictionary.rs` -- differential fuzz testing of `Dictionary` and
  `FrontCodedDictionary` against `PlainDictionary` as an oracle, in
  lockstep, across several block-size configurations chosen to force
  splits and merges; plus `#[ignore]`d perf/space probes (run with
  `cargo test --release -- --ignored --nocapture`).

## Testing

```sh
cargo test
cargo clippy --all-targets
```

## License

MIT -- see `LICENSE`.
