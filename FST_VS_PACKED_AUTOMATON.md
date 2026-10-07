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
| Minimality guarantee | **none** -- documented, deliberate | **none** -- same tradeoff, adopted deliberately (was exact until the dedup-structure fix below) |
| Dedup structure | bounded N-way set-associative LRU cache (`Registry`) | same algorithm, ported (`PackedRegistry`) -- was an unbounded exact `HashMap` (`FxHasher`) before, see below |
| State count, same input, different block cuts | ~flat | varies up to ~2x on synthetic data (see below) |
| States, fresh build, real RDF (identical term sets) | 2.717/term | **2.722/term** (+0.2%); incremental merges add ~10% -- see "Real RDF correction" |
| On-disk node encoding | 3 adaptive layouts + bit-packed variable-width ints | backward-read records: implicit first output, varint delta targets, NEXT flag, no stored counts (was 1 uniform layout with fixed 4-byte targets) |
| Output value type | arbitrary `u64` (min-plus semiring) | `u32`, rank only |
| Mutability | **none** -- write-once, full rebuild on any change | incremental `insert`/`insert_sorted`/`remove` |
| Construction input | streams to any `io::Write`, bounded memory | in-memory `Vec<u8>` only |
| Dependency | external crate (vendored, locally patched) | none -- from scratch, same crate |
| Decode speed (clustered / hashlike, cycles) | 94.1M / 1,107.2M | **48.0M / 595.0M** |
| Insert speed (single term, cycles) | 2,600.5M (full rebuild) | **~707-712M** (incremental) |
| Insert speed, batch, dictionary-scale | full rebuild every call | `insert_sorted`, shared-prefix batching |
| Memory at scale (clustered, 2M terms) | 12.92 bytes/term, flat | 14.7-14.8 bytes/term, flat-ish |
| Memory at scale (hashlike, 300K terms) | ~29.8 bytes/term, flat | ~104-109 bytes/term, flat (was ~104-132, growing, before the dedup-structure fix) |
| Dead/unreachable state reclamation | n/a -- always freshly rebuilt | `compact`: copy reachable records with exact dedup (yields the *minimal* automaton); on growth >2x per merge, and for every block in `shrink_to_fit` |
| Real RDF, 6.3M terms, each at its default blocks: logical size | 88.7 MiB (64..=256) | **77.0 MiB** (256..=1024, 90% split fill, folded labels; 322.6 before the record format, compaction, reverse-map packing, block and label changes) |
| Real RDF, full file (6.88M terms): logical size | 79.4 MiB | **33.9 MiB** (succinct id permutation, path compression, FSST-coded chain labels) |
| Real RDF, full file: insertion, 1 writer / wall time, 7 parsers + 5 writers | 79.7s / 28.7s | **33.7s / 14.3s** |
| Real RDF, 6.3M terms: insertion (prepared batches, 1 writer) | 47.7s | **22.2s** (incl. final compaction) |
| Real RDF, 6.3M terms: get_term / get_id (dictionary level) | 1,550 / 1,703 ns | **1,116-1,156 / 1,138-1,195 ns** |
| Real RDF, 6.3M terms: peak RSS (1 writer) | 134 MiB | **131 MiB** |
| Real RDF, 6.3M terms: wall time, 4 parsers + 8 writers (`ShardedDictionary`) | 12.8s | **6.9s** |
| Real RDF, 6.3M terms: single-term insert / remove | 296 / 219 us | **22.5 / 19.9 us** |

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
scale -- it's searching a strictly larger space of what counts as "the
same state," independent of how the dedup structure itself is built
(next section: both now use the same kind of structure). Full
mechanism, with the isolating experiments that confirmed it, in
`AUTOMATON_REFERENCE.md` §4 and `DESIGN.md`'s "the real RDF finding"
section; the clustered-corpus measurement at the end of the next
section is the direct confirmation that output-pushing, not the dedup
structure, is what's left causing this gap.

### Real RDF correction: output-pushing is not the gap

Measured on identical term sets (each block of a real-RDF dictionary,
6.3M terms), a *fresh* `PackedAutomaton::build_sorted` has 2.722
states/term against `fst`'s 2.717 (+0.2%) and an exact-minimal slab
build's 2.606 -- both bounded-registry implementations lose the same
~4.5% to LRU eviction, and nothing to output handling. That's
expected in hindsight: `PackedAutomaton`'s outputs are *local* prefix
sums of child subtree counts, a function of the state's right
language, so equivalent states always get identical outputs -- the
Lucchesi & Kowaltowski (1993) perfect-hashing construction, where a
minimal acceptor plus counts already is the minimal ranked structure.
The extra states measured at dictionary scale come from *incremental
merges* (`insert_sorted` into an existing block can't dedup against
records written before its registry was released: +10% reachable
states), which the earlier synthetic measurements didn't separate
from construction. Full numbers: DESIGN.md's "real-RDF decomposition".

## Dedup structure: both now use the same bounded LRU cache

`fst`'s `Registry` (`vendor/fst/src/raw/registry.rs`) is not a
`HashMap` -- it's a fixed-size `Vec<RegistryCell>`
(`table_size * mru_size` cells, allocated once, never resized), FNV-1a
hashed into a bucket, linear-scanned within that small bucket
(`mru_size = 2` here, via `Block::build`'s `with_registry_rows`,
`src/block.rs:71-75`), LRU-evicted on a miss. `fst`'s own doc comment
states plainly that this makes exact minimality unreachable for any
non-trivial input (`vendor/fst/src/raw/mod.rs:226-238`) -- a deliberate,
documented tradeoff for bounded, streaming-friendly construction memory.

`PackedAutomaton`'s dedup structure used to be different: a plain
`HashMap<PackedSignature, u32, FxHasher>` (`PackedRegister`) --
unbounded, no eviction, always finding an exact match if one exists.
Measured (not just read from the code), it also had no pruning
mechanism at all: unlike the slab `Automaton`'s own register (pruned by
`release()`, `register.remove(&sig)` on refcount-zero,
`src/micro_fst.rs:2420`), `write_confirmed_state` only ever called
`register.insert`, never `register.remove`. Driven through 2000 rounds
of sustained insert-only churn on an 800-term base set
(`measure_packed_automaton_register_growth_under_sustained_inserts`,
full methodology in `DESIGN.md`'s "measuring the register-pruning gap
directly"), `reachable_state_count` stayed essentially flat (91 -> 120)
while `register_len` climbed unboundedly (91 -> 62,632) -- a
register/reachable ratio from 1.0x to 522x, with no sign of leveling
off. Dead weight invisible to `size_bytes()`: every insert left the
previous round's now-superseded confirmed states stranded in the
register permanently. (For comparison, the same churn run against the
slab `Automaton`'s own register showed `register_len` tracking
`reachable_state_count` within 0.994-0.996x the entire time --
`release()`'s pruning does hold up, measured.)

**This has been replaced** with `PackedRegistry`/`PackedRegistryCell`
(`src/micro_fst.rs`) -- `fst`'s own `Registry` ported directly to this
crate's target type (`u32` byte offsets, not `CompiledAddr`): same
fixed `table_size * mru_size` cell array, same `FxHasher`-into-bucket
plus linear-scan-and-promote-or-evict algorithm, sized once from
`Block::build`'s own formula (`entries.len() * 4, clamped [64,
10_000]`, `mru_size = 2`) and never resized -- sound here specifically
because `s-dict` blocks are already size-bounded by the crate's own
split policy (`DEFAULT_BLOCK_MAX = 256`). Re-running the identical
churn test against the new structure: `register_len` now plateaus at
exactly 6,400 (`table_size` 3,200 x `mru_size` 2) instead of climbing
to 62,632 -- bounded, as designed. Full before/after numbers,
dictionary-scale remeasurement, and the three existing tests whose
exact-equality assertions had to change (a bounded LRU cache's
eviction decisions depend on access history, so different construction
orders over the same term set are no longer guaranteed to reach the
same structure -- only the same answers) are in `DESIGN.md`'s
"replacing PackedAutomaton's register with a bounded LRU cache."

At dictionary scale (`examples/perf_scale_memory.rs`, same corpora as
the "Measured performance" section below): on the low-redundancy
(hashlike) corpus, `bytes/term` is now flat (104.5 at 30K terms, 105.3
at 300K -- was 132.2 at 300K before this fix, growing). On the
high-redundancy (clustered) corpus, `PackedAutomaton`'s `states/term`
still grows (0.123 at 200K terms to 0.183 at 2M) while `fst`'s stays
flat (0.123 to 0.128) -- with the dedup structure now identical to
`fst`'s own, this is direct confirmation that the remaining gap is
output-pushing (previous section), not the dedup structure.
*(Superseded: `perf_scale_memory` inserts in several checkpoint
batches, so its blocks are built by incremental merges. On real RDF,
fresh builds match `fst`'s state count and the growth is explained by
merge non-minimality -- see "Real RDF correction" above.)*

## On-disk / in-memory representation

`fst` adapts its per-node encoding to the node's own shape -- three
distinct physical layouts (`vendor/fst/src/raw/node.rs:278-292`:
`StateOneTransNext`, a single-transition, address-implicit fast path;
`StateOneTrans`; and the general `StateAnyTrans`), plus variable-width
packed integers for addresses and outputs, so a low-branching-factor
node (the common case for real terms, per this project's own earlier
measurements) costs close to the theoretical minimum bytes.
`PackedAutomaton` originally used one uniform record format for
every state -- `[flag_count: u8][subtree_count: LEB128][per transition:
byte + LEB128 output + 4-byte target]`. On real RDF, 91% of states have
a single transition (unique term tails), and that format spent 8-9
bytes on each where `fst` spends 1-2: re-emitting fresh builds of real
blocks took 25.45 bytes/term in that format against 6.44 for an
fst-style encoding of the identical graphs (`fst` itself: 6.66). Measured contributions,
cumulative: no output on single-transition states (-2.5), no stored
subtree counts on 0/1-transition states (-2.7), varint delta targets
instead of fixed `u32` (-9.4, the largest), implicit NEXT target (-2.4),
label folded into the header via `fst`'s common-input table (-2.1).

The current format adopts all but the last: records are read backwards
from a header byte (like `fst`), the first transition's output is
implicit (`is_final`), targets are LEB128 deltas, a NEXT flag elides the
target of a single-transition record whose child is the record
immediately below, and **no** subtree counts are stored at all -- they're
derived top-down from the root's `term_count` and each parent's outputs
while descending (`open_frame`), which `insert`/`insert_sorted` already
do. All of it is compatible with append-only incremental mutation.
Label folding was adopted afterwards: a single-transition, non-final
record whose label is in a curated 63-byte RDF table keeps the label in
its header byte (`fst`'s `StateOneTrans*` common-input trick) --
-12% dictionary size on real RDF; see DESIGN.md's "label folding".
Fixed-width transitions for high-degree states (Lucene's approach, for
binary search) remain open.

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
both mutate in place without a full rebuild, and so does `remove`
(`insert`'s mirror: clear the final flag, prune emptied states, re-confirm
the path). `PackedAutomatonBlock` finds ranks with `lower_bound` (one
descent) for small runs instead of streaming the block, so single-term
mutations cost O(term length), not O(block size) -- DESIGN.md's
"incremental single-term mutation".

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
pays every time) vs. `PackedAutomaton`'s ~707-712M (incremental
`insert`, after the dedup-structure fix above -- a slight improvement
over the prior 729.3M, since the bounded cache's O(`mru_size`)=O(2)
bucket scan has no `HashMap` resize/rehash cost to pay). At the
dictionary level, batched via `insert_batch` ->
`PackedAutomaton::insert_sorted`, `PackedAutomatonDictionary` beats
`AutomatonDictionary` (the slab-backed comparison implementation) on
both tested corpora too -- see `DESIGN.md`'s "the gap actually closes"
section.

**Memory at scale** (`examples/perf_scale_memory.rs`, 300K-2M terms):
`fst` stays flat regardless of corpus redundancy (~12.9 bytes/term
clustered, ~29.8-32.7 hashlike, across every tested scale).
`PackedAutomaton` stays close to flat for clustered content (~14.7-14.8
bytes/term) -- but `states/term` on this same corpus still grows (0.123
at 200K terms to 0.183 at 2M, vs. `fst`'s flat 0.123-0.128): the
minimization-power gap described above (`AUTOMATON_REFERENCE.md`), now
confirmed (not just argued) to be the explanation, since the dedup
structure fix (above, "Dedup structure" section) removed the other
candidate cause and this growth persisted unchanged. On low-redundancy
(hashlike) content, `bytes/term` *was* growing with scale (~104-132
bytes/term at 300K terms, up from ~108 at 5K) -- but that one *was* a
`PackedAutomaton`-specific implementation defect (the unbounded dedup
structure, same section above), now fixed: `bytes/term` is flat
(~104.5-109.5) across the same range after the fix.

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
