# Production readiness

What stands between `s-dict` and use as a general production dependency,
in priority order. Items 1 and 2 are done; item 3 still blocks general
production use, and item 4 is needed before handling untrusted data at
scale.

Each item: current state, why it matters, a proposed approach, and when it
counts as done.

## 1. Persistence -- done

`save`/`save_file` write a versioned format (header with a checksum of
the header fields and payload; symbol table, free ids, block table, id
runs, automaton records); `load`/`load_file` copy records, `load_shared`
borrows them from one buffer, and `map` (feature `mmap`) from a
memory-mapped file, copying a block out on its first mutation. Every load
validates every section -- each block's automaton is parsed exactly as the
readers parse it with every address checked, its outputs checked against
subtree counts, its chain labels decoded, its language checked to be UTF-8
(one visit per record and decoder state), and its text against the block
limit; blocks in strictly increasing term order; live and free ids
partitioning `0..next id` -- so `decoded_term`'s premise holds for loaded
files too, and anything inconsistent is a `LoadError`, never a panic.

Tests: round trips (byte-identical re-save; identical behavior under
further mutation, including borrowed blocks), every truncation, single-byte
flips (all refused), and randomized structural corruption with the checksum
fixed up (refused, or accepted only as a dictionary whose reads and
mutations all behave): 200,000 rounds in release and 40,000 in debug
(overflow checks on) against the final format, without a panic. Still
open: coverage-guided fuzzing of the loader (item 4).

Measured on the reference file (6.88M terms, `s-bench`'s
`dict_persist_bench`): a 28.0 MiB file for a 29.9 MiB dictionary, saved in
0.3 s; `load_file`, `load_shared` and `map` each take ~0.8 s including full
validation, against ~42 s to rebuild from the N-Triples; lookups on loaded
dictionaries match the original's.

## 2. Fallible operations and atomic failure -- done

`try_insert`, `try_insert_batch`, `try_insert_prepared` and
`TermBatch::try_push` return `DictError` (`IdsExhausted`, `TermTooLarge`,
`BatchTooLarge`), checked before anything changes; the panicking forms wrap
them. The id limit is per dictionary (`set_id_limit`); `ShardedDictionary`
sets each shard's to what its global ids leave it, and checks every shard
(`check_insert_batch`) before inserting a batch, so a refusal on one shard
leaves all of them unchanged. Block records can no longer outgrow their
`u32` addresses: every merge and removal is checked against a provable
write bound (`src/limits.rs`) and rebuilds in chunks bounded by count and
text where the incremental path can't be shown to fit; chain records are
capped in length so expanding one is bounded; blocks split by text as well
as by count. Unit tests run with small limits so the rebuild paths are
exercised, and assert the bound on every write in debug builds.

## 3. Id reuse after `remove`

**State.** A removed term's id goes on a free list and is handed to the
next new term. Anything still holding the old id (an `s-graph` triple, a
cache) silently resolves it to an unrelated term.

**Why it matters.** It is a correctness hazard for every caller, and the
current API does not mention it.

**Approach.** Decide the contract. Options: (a) document that callers must
drop every reference to an id before removing its term; (b) never reuse
ids (simplest; ids grow monotonically, compaction of the id space becomes
an explicit operation); (c) generation counters in the id (costs id bits).
For a triple store, (b) with an explicit "renumber" step is usually the
least surprising.

**Done when.** The contract is chosen, documented on `remove` and
`TermId`, and tested (a removed id never resolves to a different term, or
the documented behavior holds).

## 4. Assurance depth

**State.** Deterministic differential tests against a model, a
brute-force minimality oracle, long-term tests, and a real-data run with
debug assertions. No coverage-guided fuzzing, no Miri run for the one
`unsafe` block (`decoded_term`), no CI. The compaction dedup table
(`DedupTable`) uses an unseeded hash with linear probing over signatures
partly determined by input terms; its worst case is bounded by one block's
records, so the risk is low.

**Why it matters.** The mutation, compaction and chain-expansion paths have
many states; deterministic tests cover chosen sequences only. Untrusted
input (item 1, or terms from untrusted sources) needs stronger evidence.

**Approach.** `cargo-fuzz` targets: random operation sequences checked
against the model (insert, batch, remove, `shrink_to_fit`, lookups), and
later the file loader. Run the test suite under Miri. CI running tests in
release and debug, clippy, doc tests and a short real-data smoke load with
debug assertions. Seed the dedup hash per process.

**Done when.** Fuzzers run for a fixed budget in CI without findings; Miri
passes; CI is green on every commit.

## 5. API stability

**State.** Version 0.1.0; the public API changed recently
(`PreparedBatch` became a type state, `PackedAutomatonBlockId` was
removed). The main type is named after its implementation
(`PackedAutomatonDictionary`). `Send`/`Sync` are derived automatically
(concurrent `&self` lookups are safe) but not documented or guaranteed.

**Why it matters.** Dependents need a stable surface and a stated
concurrency model.

**Approach.** Settle names (e.g. `Dictionary` for the main type), document
thread-safety (concurrent readers through `&self`; one writer; or
`ShardedDictionary` for parallel writers), add `#[must_use]` where
appropriate, review what `TermDictionary` should contain, then release
1.0 with semver from there.

**Done when.** The public API is reviewed and documented item by item, and
a semver-checking tool runs in CI.

## 6. Smaller items

- **Memory headroom during loads.** Sequential peak RSS is ~3x the final
  logical size (~94 MB for ~31 MB at 10M lines), and the sharded load
  reaches ~550 MB at the full file (bench queues and per-shard buffers
  included). Measure where the peak comes from (batches, transient
  oversized blocks, uncompacted blocks) and bound it.
- **Documentation.** `DESIGN.md` is a long history log. Users need a short
  reference: data model, guarantees, complexity of each operation, limits,
  and tuning (block bounds, batch sizes, sharding).

**Done when.** Peak memory during loads has a documented bound; a
reference document exists separately from the design history.
