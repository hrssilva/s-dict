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

This crate currently implements the first two stages of a larger design
(see [DESIGN.md](DESIGN.md) for the full architecture, roadmap, and
literature grounding):

- **`PlainDictionary`** -- a plain `HashMap`/`Vec` baseline. Kept
  permanently as the comparison point later stages get measured against,
  not a throwaway.
- **`Dictionary`** -- lexical blocks with a sorted-array router and
  split/merge on overflow/underflow. This is the crate's primary,
  general-purpose implementation.

Per-block finite-state compression (the "micro-FST" the overall
architecture is named for), a non-trivial router, RDF-aware term
decomposition, and the full multi-baseline benchmark suite are explicit
future stages, not yet implemented -- see DESIGN.md's "What's not
implemented yet".

## Install

```toml
[dependencies]
s-dict = { path = "../s-dict" }  # or a git dependency, once published
```

No optional features and no dependencies at this stage.

## Quick start

```rust
use s_dict::{Dictionary, TermDictionary};

let mut dict = Dictionary::with_block_bounds(64, 256); // block_min, block_max

let type_id = dict.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
let class_id = dict.insert("http://www.w3.org/2000/01/rdf-schema#Class");

// Idempotent: re-inserting an existing term returns its existing id.
assert_eq!(dict.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), type_id);

// Bidirectional lookup.
assert_eq!(dict.get_id("http://www.w3.org/2000/01/rdf-schema#Class"), Some(class_id));
assert_eq!(dict.get_term(class_id), Some("http://www.w3.org/2000/01/rdf-schema#Class"));

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
- `src/block.rs` -- the per-block sorted-array structure both consume.
- `tests/dictionary.rs` -- differential fuzz testing of `Dictionary`
  against `PlainDictionary` as an oracle, across several block-size
  configurations chosen to force splits and merges.

## Testing

```sh
cargo test
cargo clippy --all-targets
```

## License

MIT -- see `LICENSE`.
