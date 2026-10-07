# Minimal acyclic automaton/transducer construction: a reference survey

> **Historical.** The `fst`-backed `Dictionary`, the slab `Automaton` and their
> files were removed after this comparison (DESIGN.md's "review pass"); file and
> line references below are to the code as it was, in the git history.

Written to answer one specific question, empirically established but not
yet mechanistically explained in DESIGN.md ("the real RDF finding"
section): why does `s-dict`'s own `Automaton` need up to ~2x more states
than the vendored `fst` crate for the *identical* term set, with the gap
appearing specifically at block boundaries produced by `split()`'s
median-halving and vanishing at "clean" (bootstrap-aligned) boundaries?

This survey covers the literature both implementations draw from, `fst`'s
actual source (it's vendored at `vendor/fst`, so this is primary-source
reading, not secondhand), and `s-dict`'s own `Automaton`
(`src/micro_fst.rs`), then uses both to answer the question directly.

## 1. The literature

Both implementations are built on the same family of algorithms: register-
based incremental construction of a minimal acyclic deterministic
finite-state automaton (DAFSA) or transducer (DAFST), for lexicographically
sorted input. Three papers matter here, all already cited somewhere in
this project's own history (DESIGN.md, the original plan file) or in
`fst`'s own doc comment:

1. **Daciuk, Mihov, Watson, Watson, "Incremental Construction of Minimal
   Acyclic Finite-State Automata"** (Computational Linguistics 26(1),
   2000). The base algorithm both implementations use: maintain the
   rightmost path as a stack of unconfirmed states; when a new
   lexicographically-sorted key's common-prefix length with the previous
   key is known, confirm (register-deduplicate) every state deeper than
   the divergence point, deepest first, so a state's signature is only
   computed once its children are already canonical. `fst`'s own doc
   comment (`vendor/fst/src/raw/mod.rs:256-258`) cites this directly:
   *"Section 3 provides a decent overview of the algorithm used to
   construct transducers in this crate, assuming all outputs are 0."*
   That qualifier -- "assuming all outputs are 0" -- is the seam this
   whole investigation sits on: this paper describes the acceptor
   (unweighted) case. Both `fst` and `s-dict` need the *transducer*
   (output-bearing) case, which is a real extension, not a free
   corollary.

2. **Mihov & Maurel** (the paper `fst`'s doc comment calls *"the money
   shot, namely, it incorporates output values"* --
   `vendor/fst/src/raw/mod.rs:259-261`, "Direct Construction of Minimal
   Acyclic Subsequential Transducers"). This is the specific technique
   `fst` implements for output-bearing minimization: a *subsequential
   transducer* construction where output values are not fixed once
   written, but actively redistributed ("pushed") between a transition
   and the state it leads to, so that two structurally identical
   *suffixes* reachable via different *absolute* accumulated outputs can
   still be recognized as the same state once the common part of their
   output is factored out onto the incoming transition. This is the
   mechanism this document's answer turns on -- see §4.

3. **Mihov & Daciuk, "Direct Incremental Construction of Minimal
   Automata"** / Carrasco & Forcada's line of work on deletion --
   context for the broader design space (already in DESIGN.md's
   literature list from when `s-dict`'s own `Automaton` was first
   designed), not directly load-bearing for this specific question.

The key theoretical fact underlying all of this, stated plainly in
`fst`'s own doc comment (`vendor/fst/src/raw/mod.rs:226-233`): **exact
minimality requires memory proportional to automaton size.** There is no
way to guarantee a fully minimal result while bounding construction
memory to a constant. Every real implementation picks a point on that
tradeoff -- which is exactly where the two implementations in this
project diverge.

## 2. `fst`'s actual implementation (`vendor/fst`)

### 2.1 Construction shape (`src/raw/build.rs`)

`Builder` (`build.rs:43-73`) keeps an `UnfinishedNodes` stack
(`build.rs:76-84`), one `BuilderNodeUnfinished` per byte of the
currently-open path -- the Daciuk-style rightmost path. `insert_output`
(`build.rs:236-265`) computes the common-prefix length against the
previous key and calls `compile_from(prefix_len)` (`build.rs:267-280`),
which pops and compiles (registers or writes) every unfinished node
deeper than the shared prefix, exactly the confirm-on-divergence shape
from Daciuk et al. 2000.

### 2.2 Output-pushing (`build.rs:406-429`, `432-453`)

This is the part with no counterpart in `s-dict`'s own `Automaton`.
`find_common_prefix_and_set_output` walks the shared prefix with the
*previous* key and, at each position, does:

```rust
// build.rs:413-420 (elided)
let common_pre = t.out.prefix(out);   // = min(t.out, out)
let add_prefix = t.out.sub(common_pre);  // what t.out had "extra"
out = out.sub(common_pre);               // what's left to push further down
t.out = common_pre;                      // this transition keeps only the shared part
```

`Output` (`src/raw/mod.rs:1276-1322`) is a **min-plus semiring**:
`prefix` = `min`, `cat` = `+`, `sub` = `-`. This is the textbook
"onward"/output-pushing transformation for subsequential transducers:
whenever a transition's currently-stored output is *larger* than what
the new key needs at that point, the excess (`add_prefix`) is pushed
further down into the node via `add_output_prefix`
(`build.rs:443-453`, inside the `BuilderNodeUnfinished` impl starting at
`build.rs:432`), which adds it to the node's own `final_output` and
every one of its *already-attached* outgoing transitions. The upshot:
**a transition's output value is not fixed when first written -- it is
actively renegotiated as later, lexicographically-close siblings arrive**,
specifically to maximize how much of a node's later-compiled structure
can end up byte-identical to some other, already-registered node,
regardless of what absolute output value led to either one.

### 2.3 The registry is a bounded, lossy LRU cache, not an exhaustive map (`src/raw/registry.rs`)

`Registry::new(table_size, mru_size)` (`registry.rs:29-34`) allocates a
fixed `table_size * mru_size` array of cells -- an *N-way set-associative
cache* (`registry.rs:65-113` implements exactly this: FNV-1a hash into a
bucket, linear scan of that bucket's `mru_size` cells, promote-on-hit,
evict-LRU-on-miss). `Block::build` (`src/block.rs:71-75`, this project's
own call site) sizes it via `with_registry_rows`, `rows =
entries.len() * 4, clamped [64, 10_000]` -- for a 256-term block, ~1,024
rows x `mru_size = 2` = 2,048 cells. **A node that collides with two
more-recently-used nodes in its bucket is evicted and never reused,
even if it would have matched something inserted earlier.** `fst`'s own
doc comment says this outright (`src/raw/mod.rs:226-238`): *"For any
non-trivial sized set of keys, it is unlikely that this crate will
produce a minimal transducer... Guaranteeing a minimal transducer
requires working memory proportional to the number of states... a cache
of states... provides reasonably good compression ratios."* This is a
deliberate, acknowledged, *documented* non-minimality, traded for
`O(1)`-ish bounded memory during streaming construction.

## 3. `s-dict`'s own `Automaton` (`src/micro_fst.rs`)

### 3.1 Construction shape

`build_sorted` (`micro_fst.rs:544`) and `insert_sorted`
(`micro_fst.rs:760`) both use the same rightmost-path-as-stack technique,
confirmed via `confirm_path` (`micro_fst.rs:2545`) -- structurally the
same Daciuk et al. 2000 algorithm `fst` uses. This part matches.

### 3.2 Output computation: one-shot, never renegotiated

`finalize_outputs_and_count` (`micro_fst.rs:2604-2630`) computes a
state's children's outputs as a **prefix sum of sibling subtree
`count`s**, local to that one parent, the moment the state is confirmed:

```rust
// micro_fst.rs:2612-2625 (elided)
let mut running = u32::from(is_final);
while let Some((_, _, target)) = states[id.slot()].transitions.at(i) {
    outputs_scratch.push(running);
    running += states[target.slot()].count;
    i += 1;
}
states[id.slot()].transitions.set_outputs(outputs_scratch);
states[id.slot()].count = running;
```

`set_outputs` **replaces** the transition outputs at that state,
**once**. There is no mechanism anywhere in `micro_fst.rs` that revisits
an already-confirmed state's output values afterward -- no `prefix`/
`sub`/`cat`-style algebra, no equivalent of `add_output_prefix`. Once a
state is registered (`confirm_path`, `micro_fst.rs:2545`, via
`signature_of_into`, `micro_fst.rs:397`), its transitions' output values
are permanent.

### 3.3 The register is exhaustive, not bounded

`Register` (`micro_fst.rs:489`) is a plain `HashMap<Signature, StateId,
BuildHasherDefault<FxHasher>>` -- unbounded, no eviction, no cache rows.
Every distinct `Signature` (`micro_fst.rs:375`: `is_final` + the full
`(byte, output, target)` transition list) ever confirmed and still live
is found on lookup, guaranteed, no collisions lost to an LRU policy
(`release`, `micro_fst.rs:2382`, explicitly removes a state's entry when
it becomes unreachable, keeping the register exactly matched to
`reachable_state_count()` -- see DESIGN.md's earlier register-pruning
finding). **This register is strictly more thorough than `fst`'s bounded
cache** -- it will find every exact match that exists, with no chance of
losing one to eviction.

## 4. The comparison, and the answer

| | `fst` (`vendor/fst`) | `s-dict`'s `Automaton` |
|---|---|---|
| Base algorithm | Daciuk/Mihov/Watson/Watson 2000 | same |
| Output handling | Mihov & Maurel: dynamic, redistributed via min-plus output-pushing (`Output::prefix`/`sub`/`cat`) | static: prefix-sum computed once at confirmation, never revisited |
| Register | bounded LRU cache (`table_size x mru_size` cells, FNV-1a hashed, eviction on collision) | unbounded exact `HashMap`, no eviction |
| Minimality guarantee | **none** -- documented, deliberate | exact, for whatever signature space it searches |

The apparent paradox from DESIGN.md's last section -- `fst` achieves
*fewer* states than `s-dict`'s `Automaton` even though `fst`'s own
register is a lossy cache and `s-dict`'s is an exhaustive `HashMap` --
resolves directly from this table. `s-dict`'s register is strictly more
thorough at finding matches *within the signature space it searches*,
but that space is narrower: two states are only mergeable if their raw,
already-assigned `(byte, output, target)` triples are byte-identical.
`fst`'s output-pushing actively *enlarges* the space of what counts as
"the same state" by renegotiating output values between a transition and
its target, so that two subtrees with the same shape but different
absolute rank offsets become byte-identical *after* pushing, even though
they never were before. An exhaustive search over the narrower space
loses to a lossy search over the wider one, because the wider space
contains merges the narrower one can never find no matter how
exhaustively it looks.

This also explains the *specific* boundary-sensitivity finding from
DESIGN.md directly. A block whose term range happens to be "clean" (a
bootstrap-divided, evenly-sized chunk, or one that starts at a boundary
where the digit structure already lines up) needs little or no output
redistribution to expose its shared sub-structure -- `s-dict`'s
static, single-shot prefix sums already land on values that happen to
coincide with the same subtree occurring elsewhere. A block produced by
`split()`'s repeated median-halving of an overflowing block has no
reason to start or end at such a point -- its term range is an
arbitrary slice of a regular digit sequence, and recognizing that its
internal sub-structure matches sibling structure elsewhere in the same
block requires exactly the kind of "subtract the common offset, keep
the residual" renegotiation `s-dict`'s `Automaton` never performs.
`fst` does that renegotiation regardless of where the slice starts,
which is exactly why its state count stays close to flat (6,898-6,955
across most tested chunk sizes) while `s-dict`'s swings by nearly 2x for
the identical final term set, cut differently.

## 5. What this doesn't establish, and possible next steps

This explains the *mechanism* well enough to be confident it's the
right one -- it's consistent with every one of the four isolating
experiments in DESIGN.md, not just asserted by analogy. It does **not**
yet establish:

- ~~The exact quantitative contribution of output-pushing alone,
  isolated from the registry-boundedness difference~~ -- half done.
  `PackedAutomaton`'s own `PackedRegister` (unbounded exact `HashMap`)
  has been replaced with `PackedRegistry`, `fst`'s bounded LRU registry
  ported directly (`DESIGN.md`'s "replacing PackedAutomaton's register
  with a bounded LRU cache", `FST_VS_PACKED_AUTOMATON.md`'s "Dedup
  structure" section) -- output-pushing *not* added alongside it. Result:
  on the low-redundancy (hashlike) corpus, the growth this document's
  §4 attributed to output-pushing turned out to have been entangled
  with plain register-unboundedness all along -- fixing only the
  register flattened it completely (`bytes/term` 104-132, growing ->
  104.5-109.5, flat). On the high-redundancy (clustered) corpus, the
  growth persisted unchanged after the same fix (`states/term` still
  0.123 -> 0.183 vs. `fst`'s flat 0.123-0.128) -- direct, measured
  confirmation that output-pushing is the real cause of *that* gap,
  now cleanly isolated rather than argued by elimination across four
  indirect experiments. Still not done: the slab `Automaton`'s own
  `Register` was left exact and unbounded (this round only touched
  `PackedAutomaton`); porting the bounded registry there too, and
  separately, actually adding output-pushing to either type, remain
  open.
- Whether adding output-pushing to `s-dict`'s `Automaton` (a real,
  nontrivial change to `finalize_outputs_and_count`/`confirm_path`, and
  to `PackedAutomaton`'s equivalent, since rank-as-output is exactly
  the property `term_at_into`'s unranking already depends on and must
  keep working) is worth the complexity given the project's own
  established discipline (measure before building) -- not attempted or
  recommended here, only diagnosed.

## 6. Correction: on real RDF, output-pushing is not the cause

§4 attributes `fst`'s lower state count to output-pushing. Measured
directly on identical term sets (every block of a real-RDF dictionary,
6.3M terms; DESIGN.md's "real-RDF decomposition"):

| construction, same term sets | states/term |
|---|---|
| `PackedAutomaton::build_sorted` (fresh) | 2.722 |
| `fst` | 2.717 |
| exact minimal (`Automaton::build_sorted`, unbounded register) | 2.606 |
| `PackedAutomaton` built by incremental merges | 2.996 |

Fresh packed builds match `fst` to 0.2%; both lose ~4.5% to the
bounded LRU registry. The gap §4 set out to explain lives entirely in
the incremental row.

The theory agrees. §3.2's outputs are *local*: a state's transition
outputs are a prefix sum of its children's subtree counts, so they are
a function of the state's right language -- two equivalent states
always get identical outputs, and output handling can't block a merge.
This is Lucchesi & Kowaltowski's (1993) construction for minimal
perfect hashing: a minimal acyclic acceptor annotated with word counts
already supports ranking, with no larger transducer needed. Mihov &
Maurel's output-pushing matters for transducers whose outputs carry
information beyond the right language; for rank-as-output it brings
nothing `PackedAutomaton` doesn't already have.

What §4 measured instead: the dictionary-scale experiments it relies on
(`perf_scale_memory`, the boundary-sensitivity runs) build blocks
through several `insert_batch` calls, i.e. incremental merges, where
new records can't dedup against ones written before the merge's
registry existed. That mechanism also matches the boundary sensitivity:
how much a block is rebuilt incrementally depends on where batch and
split boundaries fall. §5's open question -- whether to add
output-pushing -- is answered: no; reducing merge non-minimality (or
compacting) is the lever.

Additional references for the encoding side (FST_VS_PACKED_AUTOMATON.md's
representation section): Daciuk, "Experiments with Automata Compression"
(CIAA 2000) and Daciuk & Weiss, "Smaller Representation of Finite State
Automata" (CIAA 2011 / TCS 2012) -- NEXT flags, label indexing in flag
bytes, variable-length pointers; Lucene's FST (`BIT_TARGET_NEXT`,
fixed-length arcs for binary search / direct addressing on wide nodes).
