# s-dict

A dynamic, incrementally mutable, bidirectional RDF term `<-> TermId`
dictionary. Terms are partitioned into contiguous lexical blocks, each an
independently mutable index; a small router and an id range map send
lookups to the right block, so an insertion or deletion touches one or two
blocks rather than the whole dictionary.

It's the sibling of [`s-graph`](../s-graph): `s-graph` indexes `(u32,
u32, u32)` triples but doesn't intern terms itself; `s-dict` is where real
IRIs/literals get dictionary-encoded into the `u32` ids `s-graph` expects.

## Design

`PackedAutomatonDictionary` is the dictionary:

- **Blocks** (`src/packed_automaton_block.rs`) hold 256..=1024 terms by
  default. Each is a byte-packed minimal acyclic automaton over its sorted
  terms (`src/packed_automaton.rs`) whose outputs are ranks, so `term ->
  local rank` and `local rank -> term` are each one descent, beside a
  compressed `rank <-> TermId` permutation (`src/block_ids.rs`: a wavelet
  matrix over the runs of consecutive ids a bulk load produces).
- **Mutation is incremental**: a merge re-confirms only the paths it
  touches (append-only records, bottom-up dedup through a bounded
  registry); dead records are reclaimed by compaction once a block's bytes
  per term double.
- **End of a load** (`shrink_to_fit`): every block is compacted to its exact
  minimal automaton and path-compressed, with chain labels coded by an FSST
  symbol table trained on the dictionary (`src/fsst.rs`).
- **Routing**: a sorted directory of block minimum keys (`term -> block`)
  and a range map of id runs (`src/range_map.rs`, `id -> block`).
- **Batches** (`src/batch.rs`): `TermBatch::prepare` sorts and groups a
  batch with an MSD radix sort, off the writer thread.
- **Sharding** (`src/sharded.rs`): `ShardedDictionary` gives each of N
  writer threads its own key ranges.

See [DESIGN.md](DESIGN.md) for the history and every measurement behind
these choices, including the FST, front-coded and slab-automaton
dictionaries this design was measured against before they were removed
(they live in the git history; [FST_VS_PACKED_AUTOMATON.md](FST_VS_PACKED_AUTOMATON.md)
and [AUTOMATON_REFERENCE.md](AUTOMATON_REFERENCE.md) document the
comparison).

Measured on a private real-RDF corpus (36.2M triples, 6.88M distinct
terms; `s-bench`'s `nt_dict_file_bench`, 6-core laptop): 30.8 MiB logical
size (4.69 bytes/term), 21-22s of dictionary insertion with one writer,
12s wall time with 7 parser threads and 5 writers, ~1.25 us per lookup in
either direction, ~24 us per single-term insert or remove.

## Install

```toml
[dependencies]
s-dict = { path = "../s-dict" }  # or a git dependency, once published
```

No dependencies.

## Quick start

```rust
use s_dict::{PackedAutomatonDictionary, TermDictionary};

let mut dict = PackedAutomatonDictionary::new();

let type_id = dict.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
let class_id = dict.insert("http://www.w3.org/2000/01/rdf-schema#Class");

// Idempotent: re-inserting an existing term returns its existing id.
assert_eq!(dict.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), type_id);

// Bidirectional lookup. get_term returns an owned String: a term is
// decoded by walking its block's automaton.
assert_eq!(dict.get_id("http://www.w3.org/2000/01/rdf-schema#Class"), Some(class_id));
assert_eq!(dict.get_term(class_id).as_deref(), Some("http://www.w3.org/2000/01/rdf-schema#Class"));

// get_term_into reuses a caller buffer instead of allocating per call.
let mut buf = Vec::new();
dict.get_term_into(class_id, &mut buf);
assert_eq!(buf, b"http://www.w3.org/2000/01/rdf-schema#Class");

// Many ids at once: one shared descent per block.
let mut terms = Vec::new();
dict.get_terms_seeked_into(&[type_id, class_id], &mut terms);
assert_eq!(terms[0].as_deref(), Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"));

// Batch insertion groups new terms by destination block.
let ids = dict.insert_batch(&["http://dbpedia.org/resource/Berlin", "http://dbpedia.org/resource/Paris"]);
assert_eq!(ids.len(), 2);

dict.remove("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
assert_eq!(dict.get_id("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), None);

// After a bulk load: compact, path-compress and trim.
dict.shrink_to_fit();
```

For ingestion with parser workers, prepare a reusable owned batch before
passing it to the single dictionary writer. Returned ids keep input order.

```rust
use s_dict::{PackedAutomatonDictionary, TermBatch};
let mut batch = TermBatch::new();
batch.push("example");
batch.push("example");
batch.prepare(); // Can run on a parser worker, independently of the dictionary.
let mut dict = PackedAutomatonDictionary::new();
let ids = dict.insert_prepared(batch.prepared().unwrap());
assert_eq!(ids[0], ids[1]);
batch.clear(); // Reuse text and metadata capacity for the next bounded batch.
```

`push` and `clear` invalidate preparation; the borrowed `PreparedBatch`
view prevents modification during insertion.

`TermId` is `u32`, matching `s-graph`'s id space; ids are reused after
removal, and running out of ids panics rather than wrapping. Block size is
a tunable (`with_block_bounds`).

## Project layout

- `src/lib.rs` -- `TermId`, the `TermDictionary` trait, exports.
- `src/packed_automaton_dictionary.rs` -- the dictionary: router, id range
  map, split/merge, batches, end-of-load step.
- `src/packed_automaton_block.rs` -- one block, and the shared `Workspace`.
- `src/packed_automaton.rs` -- the byte-packed automaton: construction,
  incremental insert/remove, lookups, compaction, path compression.
- `src/block_ids.rs` -- the compressed `rank <-> TermId` permutation.
- `src/range_map.rs` -- `TermId -> block` as id runs.
- `src/fsst.rs` -- the chain-label symbol table.
- `src/batch.rs` -- `TermBatch`/`PreparedBatch` and the radix sort.
- `src/sharded.rs` -- `ShardedDictionary`.
- `src/entries_arena.rs` -- scratch for splits and neighbor merges.
- `tests/` -- differential tests against a plain `HashMap`/`Vec` model
  (`tests/common`), with block bounds small enough to force splits and
  merges.
- `examples/perf_scale_memory.rs`, `examples/perf_seeked_at_scale.rs` --
  synthetic scaling probes. Real-data benchmarks live in `s-bench`.

## Testing

```sh
cargo test
cargo clippy --all-targets
```

## License

MIT -- see `LICENSE`.
