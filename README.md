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

See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for what remains
before general production use (the id reuse contract, fuzzing/CI, API
stability).

## Install

```toml
[dependencies]
s-dict = { path = "../s-dict" }  # or a git dependency, once published
```

No dependencies by default. The optional `mmap` feature adds
[`memmap2`](https://crates.io/crates/memmap2) for
`PackedAutomatonDictionary::map`. `unsafe` code, each with its invariant
documented at the site:

- decoded terms become `String`s without re-validating UTF-8, a premise
  checked in debug builds (`decoded_term`; DESIGN.md's "premises, not
  re-checks") and established by validation for loaded files;
- block records are read through a pointer/length view of either an owned
  buffer or a shared image (`src/records.rs`), kept current by the one
  mutation path, so reads don't branch on where the bytes live;
- `map` (feature `mmap`), whose caller guarantees the file isn't modified
  while mapped.

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

For ingestion with parser workers, prepare a batch before passing it to the
single dictionary writer. Returned ids keep input order.

```rust
use s_dict::{PackedAutomatonDictionary, TermBatch};
let mut batch = TermBatch::new();
batch.push("example");
batch.push("example");
let prepared = batch.prepare(); // Can run on a parser worker, independently of the dictionary.
let mut dict = PackedAutomatonDictionary::new();
let ids = dict.insert_prepared(&prepared);
assert_eq!(ids[0], ids[1]);
let mut batch = prepared.into_batch();
batch.clear(); // Reuse text and metadata capacity for the next bounded batch.
```

`prepare` consumes the batch: a `PreparedBatch` can't change, so its
sorted order always matches its terms (`into_batch` to edit it again).

`TermId` is `u32`, matching `s-graph`'s id space; ids are reused after
removal. Block size is a tunable (`with_block_bounds`).

### Errors

Every insertion has a fallible form (`try_insert`, `try_insert_batch`,
`try_insert_prepared`, `TermBatch::try_push`) that refuses with a
`DictError`, checked before anything changes, so a refused operation leaves
the dictionary exactly as it was:

- `IdsExhausted`: more new terms than ids left below the id limit (`u32`
  by default, lower with `set_id_limit`). A batch is refused whole.
- `TermTooLarge`: a term over 128 MiB.
- `BatchTooLarge`: a `TermBatch` over 4 GiB of text.

The panicking forms (`insert`, `insert_batch`, `insert_prepared`, `push`)
panic exactly where these return errors. Nothing else depends on input
size: a block's records are `u32`-addressed, and every mutation is checked
against a provable bound on the bytes it can write, falling back to a
rebuild in bounded chunks where the incremental path can't be shown to fit
(`src/limits.rs`). `ShardedDictionary` gives each shard the id limit its
global ids leave it, and checks every shard before inserting a batch.

### Saving and loading

```rust
use s_dict::{PackedAutomatonDictionary, TermDictionary};
let mut dict = PackedAutomatonDictionary::new();
let id = dict.insert("http://example.org/a");
dict.shrink_to_fit(); // Compact form: no dead records in the file.

let mut bytes = Vec::new();
dict.save(&mut bytes).unwrap(); // Or `save_file(path)`: temp file, sync, rename.
let loaded = PackedAutomatonDictionary::load(&bytes).unwrap(); // Or `load_file(path)`.
assert_eq!(loaded.get_id("http://example.org/a"), Some(id));
```

`load` copies each block's records; `load_shared(Arc<[u8]>)` borrows them
from one shared buffer, and `map(path)` (feature `mmap`, `unsafe`: the file
must not change while mapped) from a memory-mapped file. A borrowed block
is copied out on its first mutation. On the reference corpus (6.88M terms)
a 29.9 MiB dictionary saves to a 28.0 MiB file in 0.3 s and loads in ~0.8 s
by any of the three, against ~42 s to rebuild. Every load validates the
whole file
-- checksum, then every block's automaton (bounds of every record, counts
behind every rank, UTF-8 of every term), block order, and that live and
free ids partition the id space -- and refuses anything inconsistent with a
`LoadError`, never a panic. The format is documented in
`src/packed_automaton_dictionary/persist.rs`.

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
- `src/packed_automaton_dictionary/persist.rs` -- the file format, save,
  and validating loads (`load`, `load_shared`, `map`).
- `src/records.rs` -- block records: owned, or borrowed from a shared image.
- `src/limits.rs` -- size limits and the write bounds behind them.
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
cargo test --features mmap
cargo clippy --all-targets --features mmap
# a longer corruption run against the file validation:
S_DICT_CORRUPTION_ROUNDS=300000 cargo test --release structurally
```

## License

MIT -- see `LICENSE`.
