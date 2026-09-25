# `fst` vs. `PackedAutomaton`: a complete difference map

A dimension-by-dimension comparison of the two per-block index
representations `s-dict` can use: the vendored [`fst`](https://docs.rs/fst)
crate (`Block`, `Dictionary`) and this project's own byte-packed
automaton (`PackedAutomaton`, `PackedAutomatonBlock`,
`PackedAutomatonDictionary`, all in `src/micro_fst.rs` +
`src/packed_automaton_*.rs`). Where a difference has already been
investigated in depth elsewhere, this document gives the short answer
and points to the source: `AUTOMATON_REFERENCE.md` (the algorithm/paper
survey) and `DESIGN.md` (every measurement, in full, with methodology).

## At a glance

| dimension | `fst` (`vendor/fst`) | `PackedAutomaton` (`src/micro_fst.rs`) |
|---|---|---|
| Base construction algorithm | Daciuk/Mihov/Watson/Watson 2000 | same |
| Output handling | Mihov & Maurel output-pushing (dynamic, min-plus algebra) | rank-as-output, static, one-shot |
| Minimality guarantee | **none** -- documented, deliberate | exact, within its narrower signature space |
| Dedup structure | bounded N-way set-associative LRU cache, no `HashMap` | unbounded exact `HashMap` (`FxHasher`) |
| State count, same input, different block cuts | ~flat | varies up to ~2x |
| On-disk node encoding | 3 adaptive layouts + bit-packed variable-width ints | 1 uniform layout, fixed 4-byte target addresses |
| Output value type | arbitrary `u64` (min-plus semiring) | `u32`, rank only |
| Mutability | **none** -- write-once, full rebuild on any change | incremental `insert`/`insert_sorted`; `remove` = full rebuild |
| Construction input | streams to any `io::Write`, bounded memory | in-memory `Vec<u8>` only |
| Dependency | external crate (vendored, locally patched) | none -- from scratch, same crate |
| Decode speed (clustered / hashlike, cycles) | 94.1M / 1,107.2M | **48.0M / 595.0M** |
| Insert speed (single term, cycles) | 2,600.5M (full rebuild) | **729.3M** (incremental) |
| Insert speed, batch, dictionary-scale | full rebuild every call | `insert_sorted`, shared-prefix batching |
| Memory at scale (clustered, 2M terms) | 12.92 bytes/term, flat | 14.7-14.8 bytes/term, flat-ish |
| Memory at scale (hashlike, 300K terms) | ~29.8 bytes/term, flat | ~104-132 bytes/term, growing |
| Dead/unreachable state reclamation | n/a -- always freshly rebuilt | none; accepted, measured cost |

## Algorithm and minimization

Both implement the same base incremental-construction algorithm
(Daciuk, Mihov, Watson, Watson 2000): a rightmost-path stack, confirmed
(register-deduplicated) from the deepest divergent state upward as
lexicographically sorted keys arrive. `fst`'s own doc comment cites this
paper directly (`vendor/fst/src/raw/mod.rs:256-258`) and notes it covers
only the *acceptor* case ("assuming all outputs are 0") -- the
output-bearing extension is where the two implementations diverge, and
it's the whole subject of `AUTOMATON_REFERENCE.md`. Short version:

- **`fst`** renegotiates output values as construction proceeds
  (`vendor/fst/src/raw/build.rs:406-453`, Mihov & Maurel's output-pushing,
  a min-plus semiring -- `Output::prefix`/`sub`/`cat`,
  `vendor/fst/src/raw/mod.rs:1276-1322`), so two subtrees with the same
  shape but different absolute rank offsets can still merge once the
  common offset is factored onto the incoming transition.
- **`PackedAutomaton`** computes each state's children's outputs once,
  as a prefix sum of sibling subtree sizes, at confirmation time
  (`finalize_outputs_and_count`, `src/micro_fst.rs:2604-2630`), and never
  revisits them. A state only merges with another if their *already
  assigned* `(byte, output, target)` triples are identical.

This is why `fst` achieves fewer states for the same term set at
scale, *despite* its own register being the lossier of the two (next
section) -- it's searching a strictly larger space of what counts as
"the same state." Full mechanism, with the isolating experiments that
confirmed it, in `AUTOMATON_REFERENCE.md` §4 and `DESIGN.md`'s "the real
RDF finding" section.

## Dedup structure: no `HashMap` in `fst` at all

`fst`'s `Registry` (`vendor/fst/src/raw/registry.rs`) is not a
`HashMap` -- it's a fixed-size `Vec<RegistryCell>`
(`table_size * mru_size` cells, allocated once, never resized), FNV-1a
hashed into a bucket, linear-scanned within that small bucket
(`mru_size = 2` here, via `Block::build`'s `with_registry_rows`,
`src/block.rs:71-75`), LRU-evicted on a miss. `fst`'s own doc comment
states plainly that this makes exact minimality unreachable for any
non-trivial input (`vendor/fst/src/raw/mod.rs:226-238`) -- a deliberate,
documented tradeoff for bounded, streaming-friendly construction memory.

`PackedAutomaton`'s `PackedRegister` (`src/micro_fst.rs:1275`) is a
plain `HashMap<PackedSignature, u32, BuildHasherDefault<FxHasher>>` --
unbounded, no eviction, so it always finds an exact match if one exists
within its (narrower) signature space. It also has no pruning
mechanism at all: unlike the slab `Automaton`'s own register (which
`release()`, `src/micro_fst.rs:2382`, keeps trimmed to exactly the
reachable state set), `PackedAutomaton`'s register only ever grows
across a block's incremental-mutation lifetime, entirely invisible to
`size_bytes()`. See `DESIGN.md`'s register-pruning finding for the
measured impact, and the prior chat turn on why porting `fst`'s bounded
cache into `PackedAutomaton` would be a real correctness tradeoff (loses
the exact-minimality guarantee), not a pure win, and hasn't been
attempted.

## On-disk / in-memory representation

`fst` adapts its per-node encoding to the node's own shape -- three
distinct physical layouts (`vendor/fst/src/raw/node.rs:278-292`:
`StateOneTransNext`, a single-transition, address-implicit fast path;
`StateOneTrans`; and the general `StateAnyTrans`), plus variable-width
packed integers for addresses and outputs, so a low-branching-factor
node (the common case for real terms, per this project's own earlier
measurements) costs close to the theoretical minimum bytes.
`PackedAutomaton` uses one uniform record format for every state
regardless of shape -- `[flag_count: u8][subtree_count: LEB128][per
transition: byte + LEB128 output + 4-byte target]` -- simpler, but not
adaptive: every target address costs a full 4 bytes even for a state
with one child pointing at an address that would fit in far fewer bits.
This is a real, unexplored difference in the *coding* dimension,
separate from the *minimization* dimension covered above -- not
measured against each other in isolation anywhere in this project yet.

## Mutability

`fst` has none. `Builder` (`vendor/fst/src/raw/build.rs:43-73`) requires
keys in strict sorted order, forbids duplicates, and streams its output
to any `io::Write` -- once `finish()`/`into_fst()` runs, the result is
immutable. Every mutation in this crate's `Block` (`src/block.rs`)
means: decode the whole block, apply the change, and call `Builder`
again from scratch (`Block::build`). There is no incremental insert or
remove at the `fst` level at all.

`PackedAutomaton` supports real incremental mutation: `insert` (single
term, `src/micro_fst.rs:1643`) and `insert_sorted` (persistent-path
batch insert mirroring the slab `Automaton`'s own, `src/micro_fst.rs:1570`)
both mutate in place without a full rebuild. `remove` has no incremental
path yet -- `PackedAutomatonBlock::remove` (`src/packed_automaton_block.rs`)
decodes the block and calls `PackedAutomaton::build_sorted` fresh, the
same full-rebuild shape `Block`'s own `remove` already has (an accepted
limitation, not a regression).

## Measured performance

All numbers below are `perf stat`-confirmed (cycles, not wall-clock
alone), with full methodology, corpus definitions, and repro commands
in `DESIGN.md`.

**Decode** (`rank`/`term_at_into`, `perf_probe_decode_cost_*` in
`src/micro_fst.rs`): `PackedAutomaton` beats `fst` in both tested
redundancy regimes -- 48.0M vs. 94.1M cycles on the clustered corpus,
595.0M vs. 1,107.2M on hashlike.

**Insert** (`perf_probe_insert_cost_*`): a single-term insert costs
`fst` 2,600.5M cycles (the full decode+merge+rebuild `Block::build`
pays every time) vs. `PackedAutomaton`'s 729.3M (incremental `insert`).
At the dictionary level, batched via `insert_batch` ->
`PackedAutomaton::insert_sorted`, `PackedAutomatonDictionary` beats
`AutomatonDictionary` (the slab-backed comparison implementation) on
both tested corpora too -- see `DESIGN.md`'s "the gap actually closes"
section.

**Memory at scale** (`examples/perf_scale_memory.rs`, 300K-2M terms):
`fst` stays flat regardless of corpus redundancy (~12.9 bytes/term
clustered, ~29.8-32.7 hashlike, across every tested scale).
`PackedAutomaton` stays close to flat for clustered content (~14.7-14.8
bytes/term) but grows with scale for low-redundancy content (~104-132
bytes/term at 300K hashlike terms, up from ~108 at 5K) -- better than
the slab `Automaton` by a substantial margin in both regimes, but not
flat the way `fst` is. Root cause: the minimization-power gap described
above (`AUTOMATON_REFERENCE.md`), not a `PackedAutomaton`-specific
implementation defect -- the slab `Automaton` (no packing at all) shows
the identical state-count growth curve, just at a larger fixed
per-state byte cost.

## Dependency footprint

`fst` is an external crate, vendored locally with one small local patch
(`vendor/fst/LOCAL_CHANGES.md` -- configurable builder registry row
count, used by `Block::build`'s `with_registry_rows` sizing).
`PackedAutomaton` has zero dependencies: built entirely from scratch in
this crate, no vendored or external minimization library.

## Where to look for more

- `AUTOMATON_REFERENCE.md` -- the literature survey and the exact
  mechanism behind the minimization-power gap, with paper and code
  citations on both sides.
- `DESIGN.md` -- every measurement referenced here, in full, with
  methodology, repro commands, and the complete history of how each
  finding was reached (including corrections along the way -- this
  project's own discipline is to report negative/mixed results as
  honestly as favorable ones).
