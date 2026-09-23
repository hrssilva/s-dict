//! A from-scratch, minimal acyclic finite-state automaton (the "grammar"
//! over one block's sorted term set -- see DESIGN.md's "Custom incremental
//! micro-FST" notes), built for eventual incremental mutation instead of
//! the `fst` crate's full-rebuild-per-change contract. This module is
//! self-contained: no dependency on `fst` or on `crate::TermId`, so it can
//! be built and proven correct/minimal entirely on its own before anything
//! elsewhere in the crate depends on it (per the staged plan -- this file
//! covers bulk construction, differential testing against the current
//! `Block`, incremental single-term `insert`, and batch `insert_sorted`
//! (closing the gap `Block::merge_into` already had: merging many sorted
//! terms into one block in a single pass instead of paying per-term
//! overhead once per term). Deletion and wiring this into `Block` as a
//! measured comparison implementation are later, not-yet-started stages.
//!
//! Like `Block`, a term's rank in sorted order *is* its lookup output
//! value, monotonically non-decreasing across a state's transitions --
//! `rank`/`term_at_into` below are the same right-language-cardinality
//! technique `block.rs` already uses (Cognetta, Allauzen, Riley, "On the
//! Compression of Lexicon Transducers", FSMNLP 2019), just computed and
//! stored by us instead of by `fst::raw`.
//!
//! Construction follows Daciuk, Mihov, Watson, Watson, "Incremental
//! Construction of Minimal Acyclic Finite-State Automata" (Computational
//! Linguistics 26(1), 2000), sorted-input specialization: process terms in
//! lexicographic order, keep the current word's root-to-leaf path as a
//! stack of not-yet-frozen states, and "confirm" (register-deduplicate)
//! every state deeper than the next word's shared prefix before extending
//! the path further. The per-transition output values (this automaton's
//! transducer half, the hard part per Mihov & Daciuk's 1998 companion
//! paper) are derived, not tracked incrementally during descent: once a
//! state's full transition set is known (at confirmation time, since all
//! its children are already confirmed by then), each transition's output
//! is the prefix sum, over lexicographically earlier sibling transitions,
//! of the number of words accepted in that sibling's subtree -- and a
//! state's own such count is purely a function of its own (already-
//! confirmed) subtree, so it's well-defined and shareable the moment two
//! states are found equivalent, regardless of how many parents reach them.
//!
//! Wired in via `automaton_block::AutomatonBlock` and, from there,
//! `AutomatonDictionary` -- not called directly from anywhere outside
//! this crate, so several items here are still only reachable from
//! `#[cfg(test)]` code (`state_slab_len`, `reachable_state_count`) or
//! from `automaton_block.rs` specifically; nothing is unused by accident.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::num::NonZeroU32;

/// A stable handle for one automaton state: an index into `Automaton`'s
/// state slab. Mirrors `dictionary::BlockId`'s `NonZeroU32` niche pattern
/// so `Option<StateId>` (used per transition slot below, since a chunk
/// isn't always full) costs 4 bytes, not 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StateId(NonZeroU32);

impl StateId {
    fn from_slot(slot: u32) -> StateId {
        StateId(NonZeroU32::new(slot + 1).expect("slot + 1 does not overflow u32"))
    }

    fn slot(self) -> u32 {
        self.0.get() - 1
    }
}

/// How many transitions fit in one chunk without an extra allocation.
/// Real RDF terms have low branching factor at most nodes (established by
/// this project's own real-corpus profiling, see DESIGN.md's
/// "Correction" section) -- most states need exactly one chunk.
const CHUNK_CAP: usize = 4;

#[derive(Debug, Clone, Copy)]
struct TransitionChunk {
    len: u8,
    bytes: [u8; CHUNK_CAP],
    outputs: [u32; CHUNK_CAP],
    targets: [Option<StateId>; CHUNK_CAP],
}

impl Default for TransitionChunk {
    fn default() -> Self {
        TransitionChunk {
            len: 0,
            bytes: [0; CHUNK_CAP],
            outputs: [0; CHUNK_CAP],
            targets: [None; CHUNK_CAP],
        }
    }
}

/// One state's outgoing transitions, as an unrolled linked list: the
/// first `CHUNK_CAP` transitions live inline (no allocation at all, the
/// common case for this corpus); only a state with more than `CHUNK_CAP`
/// distinct next-bytes spills into `overflow`. Entries are always kept
/// sorted by byte (`insert_transition` maintains this even though
/// incremental `insert` can add a transition out of order; see its doc
/// comment) -- `iter`/`get`/`set_target_for_byte` all rely on that.
#[derive(Debug, Default, Clone)]
struct Transitions {
    inline: TransitionChunk,
    overflow: Vec<TransitionChunk>,
}

impl Transitions {
    /// Transitions in byte-sorted order: `(byte, output, target)`. Used
    /// where a full iterator genuinely helps (register-signature
    /// computation, reachability walks, confirmation's per-target output
    /// pass) -- `get`/`last_output_at_most` below are direct loops
    /// instead, on purpose: this project's own `perf`-measured decode
    /// comparison against `Block` found the generic `Iterator` combinator
    /// chain here (`chain`, `flat_map`, `take_while`, `find`), not
    /// genuinely more work, was costing ~85% more cycles for essentially
    /// the same instruction count -- see DESIGN.md's "decode speed"
    /// section for the counters.
    fn iter(&self) -> impl Iterator<Item = (u8, u32, StateId)> + '_ {
        std::iter::once(&self.inline)
            .chain(self.overflow.iter())
            .flat_map(|chunk| {
                (0..chunk.len as usize).map(move |i| {
                    (
                        chunk.bytes[i],
                        chunk.outputs[i],
                        chunk.targets[i].expect("populated slots always hold a target"),
                    )
                })
            })
    }

    /// `Some((output, target))` for the transition on `byte`, `None` if
    /// there isn't one. A direct loop over `inline` then `overflow`, not
    /// `iter().find(..)` -- see `iter`'s doc comment. Linear:
    /// `CHUNK_CAP`-sized in the common case, cache-resident; a future
    /// stage can binary-search within/across chunks if a profile shows
    /// the scan itself (not iterator overhead) matters at the branching
    /// factors this corpus actually has.
    fn get(&self, byte: u8) -> Option<(u32, StateId)> {
        for i in 0..self.inline.len as usize {
            if self.inline.bytes[i] == byte {
                return Some((
                    self.inline.outputs[i],
                    self.inline.targets[i].expect("populated slots always hold a target"),
                ));
            }
        }
        for chunk in &self.overflow {
            for i in 0..chunk.len as usize {
                if chunk.bytes[i] == byte {
                    return Some((
                        chunk.outputs[i],
                        chunk.targets[i].expect("populated slots always hold a target"),
                    ));
                }
            }
        }
        None
    }

    /// The `idx`-th transition in byte-sorted order, or `None` past the
    /// end -- what a state-by-state streaming walk needs to advance one
    /// child at a time without re-deriving its position from a byte
    /// value. A direct loop, not `iter().nth(idx)`; see `iter`'s doc
    /// comment for why that distinction has already been shown to matter
    /// here.
    fn at(&self, idx: usize) -> Option<(u8, u32, StateId)> {
        if idx < self.inline.len as usize {
            return Some((
                self.inline.bytes[idx],
                self.inline.outputs[idx],
                self.inline.targets[idx].expect("populated slots always hold a target"),
            ));
        }
        let mut remaining = idx - self.inline.len as usize;
        for chunk in &self.overflow {
            if remaining < chunk.len as usize {
                return Some((
                    chunk.bytes[remaining],
                    chunk.outputs[remaining],
                    chunk.targets[remaining].expect("populated slots always hold a target"),
                ));
            }
            remaining -= chunk.len as usize;
        }
        None
    }

    /// The last transition (in byte order) whose output is `<= value` --
    /// what `Automaton::term_at_into` needs once per state it visits
    /// while unranking. A direct loop, not `iter().take_while(..)
    /// .last()`: same answer (transitions are sorted by output, a
    /// consequence of being sorted by byte -- see the struct doc
    /// comment), no generic `Iterator` combinator overhead; see `iter`'s
    /// doc comment for why that overhead is the thing being avoided here,
    /// not the scan itself. Panics if no transition qualifies -- callers
    /// only ever pass a `value` known to be reachable from this state.
    fn last_output_at_most(&self, value: u32) -> (u8, u32, StateId) {
        let mut best: Option<(u8, u32, StateId)> = None;
        for i in 0..self.inline.len as usize {
            if self.inline.outputs[i] > value {
                return best.expect("value must be reachable via some transition at this state");
            }
            best = Some((
                self.inline.bytes[i],
                self.inline.outputs[i],
                self.inline.targets[i].expect("populated slots always hold a target"),
            ));
        }
        for chunk in &self.overflow {
            for i in 0..chunk.len as usize {
                if chunk.outputs[i] > value {
                    return best
                        .expect("value must be reachable via some transition at this state");
                }
                best = Some((
                    chunk.bytes[i],
                    chunk.outputs[i],
                    chunk.targets[i].expect("populated slots always hold a target"),
                ));
            }
        }
        best.expect("value must be reachable via some transition at this state")
    }

    /// `last_output_at_most`, also returning the winning transition's
    /// flat index (as `Transitions::at`/`StreamFrame::next_child`
    /// address it) -- needed only by `Automaton::seek_stream`, to build a
    /// `StreamFrame` resumable by `TermsStream::next` from a rank other
    /// than 0. Not used by `term_at_into`'s own hot path (which has no
    /// need for the index), so kept separate rather than changing
    /// `last_output_at_most`'s signature for every caller.
    fn last_output_at_most_with_index(&self, value: u32) -> (usize, u8, u32, StateId) {
        let mut best: Option<(usize, u8, u32, StateId)> = None;
        let mut idx = 0usize;
        for i in 0..self.inline.len as usize {
            if self.inline.outputs[i] > value {
                return best.expect("value must be reachable via some transition at this state");
            }
            best = Some((
                idx,
                self.inline.bytes[i],
                self.inline.outputs[i],
                self.inline.targets[i].expect("populated slots always hold a target"),
            ));
            idx += 1;
        }
        for chunk in &self.overflow {
            for i in 0..chunk.len as usize {
                if chunk.outputs[i] > value {
                    return best
                        .expect("value must be reachable via some transition at this state");
                }
                best = Some((
                    idx,
                    chunk.bytes[i],
                    chunk.outputs[i],
                    chunk.targets[i].expect("populated slots always hold a target"),
                ));
                idx += 1;
            }
        }
        best.expect("value must be reachable via some transition at this state")
    }

    /// Inserts a new transition for `byte` (which must not already
    /// exist), keeping transitions sorted by byte. `build_sorted` always
    /// happens to append at the end (sorted input never branches
    /// backwards at a given node), but incremental `insert` has no such
    /// guarantee -- a brand-new branch at an existing, already-populated
    /// node can land anywhere relative to its current transitions, so a
    /// blind append would silently break the byte-sorted-transitions
    /// invariant everything else here (confirmation's output computation,
    /// `rank`, `term_at_into`) depends on. Rebuilds from a small sorted
    /// `Vec` rather than shifting in place across chunks -- simple, and
    /// cheap given this corpus's measured low branching factor (see the
    /// module doc comment); outputs are left at 0, filled in later once
    /// this state is confirmed and its full sibling set is known.
    fn insert_transition(&mut self, byte: u8, target: StateId) {
        let mut entries: Vec<(u8, StateId)> = self.iter().map(|(b, _, t)| (b, t)).collect();
        let pos = entries.partition_point(|&(b, _)| b < byte);
        debug_assert!(
            entries.get(pos).is_none_or(|&(b, _)| b != byte),
            "transition for byte {byte} already exists"
        );
        entries.insert(pos, (byte, target));

        self.inline = TransitionChunk::default();
        self.overflow.clear();
        for (b, t) in entries {
            self.append_in_order(b, t);
        }
    }

    /// Appends assuming `byte` sorts after every existing entry -- only
    /// safe as a private helper for rebuilding from an already-sorted
    /// list (see `insert_transition`); never called with unsorted input.
    fn append_in_order(&mut self, byte: u8, target: StateId) {
        let chunk = if (self.inline.len as usize) < CHUNK_CAP {
            &mut self.inline
        } else {
            if self
                .overflow
                .last()
                .is_none_or(|c| c.len as usize == CHUNK_CAP)
            {
                self.overflow.push(TransitionChunk::default());
            }
            self.overflow.last_mut().expect("just ensured non-empty")
        };
        let i = chunk.len as usize;
        chunk.bytes[i] = byte;
        chunk.targets[i] = Some(target);
        chunk.outputs[i] = 0;
        chunk.len += 1;
    }

    fn is_empty(&self) -> bool {
        self.inline.len == 0
    }

    /// Removes the transition for `byte` (which must already exist).
    /// Used by `Automaton::remove` when pruning a state that's become
    /// dead (no longer final and, after this removal, no transitions
    /// left) back out of its parent entirely -- not a redirect, an actual
    /// deletion, unlike every other mutation here. Same
    /// rebuild-from-a-sorted-`Vec` approach as `insert_transition`, for
    /// the same reason (simple, cheap at this corpus's branching factors).
    fn remove_transition(&mut self, byte: u8) {
        let entries: Vec<(u8, StateId)> = self
            .iter()
            .filter(|&(b, _, _)| b != byte)
            .map(|(b, _, t)| (b, t))
            .collect();
        self.inline = TransitionChunk::default();
        self.overflow.clear();
        for (b, t) in entries {
            self.append_in_order(b, t);
        }
    }

    /// Overwrites the target of the transition for `byte` (which must
    /// already exist). Used both when confirmation finds an equivalent
    /// existing state and redirects the parent to it instead of the
    /// just-built duplicate, and when incremental insertion clones a
    /// shared state on the matched path and repoints its parent at the
    /// fresh clone.
    fn set_target_for_byte(&mut self, byte: u8, new_target: StateId) {
        for chunk in std::iter::once(&mut self.inline).chain(self.overflow.iter_mut()) {
            for i in 0..chunk.len as usize {
                if chunk.bytes[i] == byte {
                    chunk.targets[i] = Some(new_target);
                    return;
                }
            }
        }
        unreachable!("byte {byte} must already have a transition at this state");
    }

    /// Sets every transition's output, in iteration (byte-sorted) order.
    fn set_outputs(&mut self, outputs: &[u32]) {
        let mut idx = 0;
        for i in 0..self.inline.len as usize {
            self.inline.outputs[i] = outputs[idx];
            idx += 1;
        }
        for chunk in &mut self.overflow {
            for i in 0..chunk.len as usize {
                chunk.outputs[i] = outputs[idx];
                idx += 1;
            }
        }
        debug_assert_eq!(idx, outputs.len());
    }
}

#[derive(Debug, Default, Clone)]
struct State {
    transitions: Transitions,
    is_final: bool,
    /// Number of accepted words reachable from this state, itself
    /// included if final -- a pure function of this state's own subtree,
    /// so valid the moment it's confirmed and shared by every parent that
    /// reaches it afterward. See the module doc comment.
    count: u32,
}

/// The register-deduplication key: two states are equivalent (and so can
/// share one representation) exactly when they have the same finality and
/// the same transition set -- same bytes, same outputs, same *targets*
/// (which are themselves already-canonical `StateId`s by the time this is
/// computed, since children are always confirmed before their parent).
///
/// `transitions` is a plain `Vec`, not an inline/`SmallVec`-style buffer:
/// tried and measured (DESIGN.md's "implementing the levers" section) --
/// a hand-rolled inline-capacity `SmallVec` was built for exactly this
/// field, at every inline capacity tried (1, 2, 4). All of them made
/// `build_sorted` *slower*, not faster: `SmallVec`'s own overhead (a
/// bigger struct to move on every register lookup, a branch on every
/// indexed access to pick inline vs. overflow) outweighed the allocation
/// it avoided, and a smaller `N` only made it worse by pushing more
/// states into the overflow `Vec` anyway. Reverted.
///
/// A *reused scratch buffer* (not an inline-capacity type) is the
/// follow-up that actually helped -- see `signature_of_into`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Signature {
    is_final: bool,
    transitions: Vec<(u8, u32, StateId)>,
}

/// Builds `state`'s `Signature`, taking `scratch`'s buffer (`mem::take`,
/// not a fresh allocation) for `transitions`. Unlike
/// `finalize_outputs_and_count`'s `outputs_scratch` (which never leaves
/// the function that fills it), this buffer's fate depends on what the
/// caller does with the resulting `Signature`: on a register *hit*, the
/// `Signature` is about to be dropped, so the caller should give the
/// buffer back (`*scratch = sig.transitions`) before that happens, so
/// the next confirmation reuses its capacity instead of starting from
/// zero; on a *miss*, the `Signature` (transitions included) is moved
/// into `register` permanently -- there is nothing to give back, and
/// `scratch` is left however `mem::take` leaves it (empty, zero
/// capacity) until the next miss grows a fresh one. See `confirm_path`,
/// this function's only caller, for exactly that logic. Register hits
/// are the common case for a corpus with real structural sharing (that's
/// the entire point of minimization), so this should eliminate the
/// large majority of what was, before this, one allocation per confirmed
/// state regardless of hit or miss.
fn signature_of_into(state: &State, scratch: &mut Vec<(u8, u32, StateId)>) -> Signature {
    // A direct loop over `Transitions::at`, not `iter().collect()` --
    // see `finalize_outputs_and_count`'s doc comment (right next to this
    // one in a `perf record` profile of `build_sorted`) for why.
    scratch.clear();
    let mut i = 0usize;
    while let Some(t) = state.transitions.at(i) {
        scratch.push(t);
        i += 1;
    }
    Signature {
        is_final: state.is_final,
        transitions: std::mem::take(scratch),
    }
}

/// A fast, non-cryptographic hasher for `register` -- std's default
/// `HashMap` hasher (`SipHash`) is deliberately slow, traded for DoS
/// resistance against adversarial keys, which doesn't apply here: this
/// map is purely internal bookkeeping, its keys are `Signature`s this
/// module derives itself, never attacker-influenced. A `perf record`
/// profile of `build_sorted` (DESIGN.md's "closing the bulk-construction
/// gap" section) found `SipHash` hashing plus the `Vec` equality
/// comparison (`memcmp`) it triggers on every register lookup was,
/// combined, the single largest cost in bulk construction -- this is the
/// same simple multiply-xor-rotate hash `rustc`'s own compiler internals
/// use for exactly this class of problem (fast, well-distributed enough
/// for non-adversarial keys, nothing fancier needed). Not used anywhere
/// else in this crate: nothing else here is exposed to attacker-
/// controlled hash keys in a way that would make this an appropriate
/// default more broadly.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let (chunk, rest) = bytes.split_at(8);
            self.add(u64::from_ne_bytes(chunk.try_into().unwrap()));
            bytes = rest;
        }
        if bytes.len() >= 4 {
            let (chunk, rest) = bytes.split_at(4);
            self.add(u32::from_ne_bytes(chunk.try_into().unwrap()) as u64);
            bytes = rest;
        }
        if bytes.len() >= 2 {
            let (chunk, rest) = bytes.split_at(2);
            self.add(u16::from_ne_bytes(chunk.try_into().unwrap()) as u64);
            bytes = rest;
        }
        if let Some(&b) = bytes.first() {
            self.add(b as u64);
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }

    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

type Register = HashMap<Signature, StateId, BuildHasherDefault<FxHasher>>;

/// A minimal acyclic automaton over one block's sorted term set, built
/// from scratch (see module docs). `states` is a slab; states discarded
/// during register deduplication (bulk construction, or a clone made
/// stale by an incremental insert) are left as unreachable garbage rather
/// than reclaimed -- harmless for correctness (`len`/`rank`/`term_at_into`
/// never touch anything but the reachable graph from `root`), but a real,
/// currently unaddressed memory-growth cost for long-lived incremental
/// mutation; slot reuse (mirroring `Dictionary`'s own `free_block_slots`
/// pattern) is future work, not attempted here.
#[derive(Debug, Clone)]
pub(crate) struct Automaton {
    states: Vec<State>,
    root: StateId,
    /// Persists across `insert` calls, not just within one `build_sorted`
    /// -- incremental insertion needs to register-deduplicate new/cloned
    /// states against *every* state this automaton has ever confirmed,
    /// not just the ones from the current call.
    register: Register,
    /// Reference counts, parallel to `states` (indexed by slot): how many
    /// transitions, anywhere in the automaton, currently point at each
    /// state. Maintained by `acquire`/`release`/`redirect`/`link_new`
    /// below, never touched directly. `root`'s own count is never
    /// tracked (nothing is ever a transition *target* pointing back at
    /// it -- this automaton is acyclic and root is the unique entry
    /// point), so it never appears in `free_slots` either.
    refcounts: Vec<u32>,
    /// Slots whose refcount has dropped to zero -- dead, reusable. Popped
    /// by `alloc_state` before growing `states`/`refcounts`, closing the
    /// gap `state_slab_len` vs. `reachable_state_count` measured as a
    /// real, severe cost under sustained mutation (DESIGN.md's
    /// "slot-reclamation gap, measured" section) before this existed.
    free_slots: Vec<u32>,
    /// A monotonically increasing counter, stamped onto a slot (in
    /// `generations`, parallel to `states`) every time `alloc_state`
    /// assigns it -- fresh or reused. `insert_sorted` used to tell
    /// "created during this batch" from "pre-existing" by comparing raw
    /// slot *numbers* against a snapshot taken at batch start; slot
    /// *reuse* breaks that outright, since a slot freed and reused
    /// earlier in the same batch can land below that snapshot's number
    /// while being entirely private to this batch. Comparing generations
    /// instead of slot numbers is correct regardless of reuse.
    generations: Vec<u64>,
    next_generation: u64,
}

impl Automaton {
    /// Builds a minimal automaton from an already sorted, deduplicated
    /// sequence of terms. Mirrors `Block::build`'s contract but takes
    /// terms alone -- a term's rank in sorted order (0-based) is its
    /// lookup output value, purely positional, so there's nothing else to
    /// pass in; callers pairing ranks with external ids keep that mapping
    /// separately, exactly as `Block` already does with its own `ids`
    /// array.
    pub(crate) fn build_sorted<'a>(terms: impl ExactSizeIterator<Item = &'a str>) -> Automaton {
        let mut states: Vec<State> = vec![State::default()];
        let root = StateId::from_slot(0);
        let mut refcounts: Vec<u32> = vec![0];
        let mut free_slots: Vec<u32> = Vec::new();
        let mut generations: Vec<u64> = vec![0];
        let mut next_generation: u64 = 1;
        // path[i] = (state after i bytes of `last_word`, byte consumed to
        // reach it from path[i - 1]); path[0]'s byte is an unused
        // placeholder, since root has no incoming transition.
        let mut path: Vec<(StateId, u8)> = vec![(root, 0)];
        let mut last_word: Vec<u8> = Vec::new();
        let mut register: Register = Register::default();
        // Reused across every `confirm_path` call in this build, not
        // reallocated per confirmed state -- see `finalize_outputs_and_count`
        // and `signature_of_into`'s own doc comments.
        let mut outputs_scratch: Vec<u32> = Vec::new();
        let mut signature_scratch: Vec<(u8, u32, StateId)> = Vec::new();

        for term in terms {
            let word = term.as_bytes();
            let common = common_prefix_len(&last_word, word);

            confirm_path(
                &mut states,
                &mut path,
                &mut register,
                &mut refcounts,
                &mut free_slots,
                common + 1,
                &mut outputs_scratch,
                &mut signature_scratch,
            );

            let mut cur = path.last().expect("path always has at least the root").0;
            for &b in &word[common..] {
                let new_id = alloc_state(
                    &mut states,
                    &mut refcounts,
                    &mut generations,
                    &mut free_slots,
                    &mut next_generation,
                    State::default(),
                );
                link_new_appending(&mut states, &mut refcounts, cur, b, new_id);
                path.push((new_id, b));
                cur = new_id;
            }
            states[cur.slot() as usize].is_final = true;

            last_word.clear();
            last_word.extend_from_slice(word);
        }

        confirm_path(
            &mut states,
            &mut path,
            &mut register,
            &mut refcounts,
            &mut free_slots,
            0,
            &mut outputs_scratch,
            &mut signature_scratch,
        );

        Automaton {
            states,
            root,
            register,
            refcounts,
            free_slots,
            generations,
            next_generation,
        }
    }

    /// Inserts `term` if not already present, preserving minimality
    /// (Daciuk, Mihov, Watson, Watson 2000's technique, applied here to an
    /// already-finished automaton rather than during initial sorted
    /// construction -- see the module doc comment). Idempotent: a
    /// present term is a no-op, matching every other `insert` in this
    /// crate.
    ///
    /// Three phases: (1) walk existing transitions as far as `term`
    /// matches; (2) every *non-root* state on that matched path may be
    /// shared with other terms, so it's cloned rather than mutated in
    /// place, with each clone's immediate predecessor on the path
    /// repointed at it -- root itself is always safe to mutate directly,
    /// since nothing is ever a transition *target* pointing back at it
    /// (this automaton is acyclic and root is never anyone's child); (3)
    /// brand-new states extend the clone chain for `term`'s unmatched
    /// suffix. The whole modified path (clones and new states alike) is
    /// then re-confirmed bottom-up, exactly as `build_sorted` confirms
    /// each word's path -- a clone might, after its one new/redirected
    /// transition, happen to now equal some other already-registered
    /// state, in which case it's merged away just like any other
    /// duplicate; this is the same mechanism, not a special case of it.
    ///
    /// `AutomatonBlock::merge_into` currently routes every insertion,
    /// single term included, through `insert_sorted` instead of this --
    /// one call site regardless of batch size, simpler than special-
    /// casing. Kept and `#[allow(dead_code)]`-exempted here rather than
    /// deleted: it's a distinct, separately tested algorithm (the
    /// original single-insert path, not `insert_sorted` with a
    /// one-element iterator dressed up), and a real candidate if a future
    /// profile shows the batch path's small extra bookkeeping
    /// (`last_word` tracking, a `results` vec) costs anything measurable
    /// for the single-term case specifically.
    #[allow(dead_code)]
    pub(crate) fn insert(&mut self, term: &str) {
        let word = term.as_bytes();

        let mut path: Vec<(StateId, u8)> = vec![(self.root, 0)];
        let mut matched = 0;
        while matched < word.len() {
            let cur = path.last().expect("path always has at least the root").0;
            match self.states[cur.slot() as usize]
                .transitions
                .get(word[matched])
            {
                Some((_, target)) => {
                    path.push((target, word[matched]));
                    matched += 1;
                }
                None => break,
            }
        }

        if matched == word.len() {
            let leaf = path.last().expect("path always has at least the root").0;
            if self.states[leaf.slot() as usize].is_final {
                return;
            }
        }

        let mut outputs_scratch = Vec::new();
        let mut signature_scratch = Vec::new();

        for i in 1..path.len() {
            let (original, byte) = path[i];
            let cloned = clone_state(
                &mut self.states,
                &mut self.refcounts,
                &mut self.generations,
                &mut self.free_slots,
                &mut self.next_generation,
                original,
            );
            path[i].0 = cloned;
            let parent = path[i - 1].0;
            redirect(
                &mut self.states,
                &mut self.refcounts,
                &mut self.free_slots,
                &mut self.register,
                parent,
                byte,
                cloned,
                &mut signature_scratch,
            );
        }

        let mut cur = path.last().expect("path always has at least the root").0;
        for &b in &word[matched..] {
            let new_id = alloc_state(
                &mut self.states,
                &mut self.refcounts,
                &mut self.generations,
                &mut self.free_slots,
                &mut self.next_generation,
                State::default(),
            );
            link_new(&mut self.states, &mut self.refcounts, cur, b, new_id);
            path.push((new_id, b));
            cur = new_id;
        }
        self.states[cur.slot() as usize].is_final = true;

        confirm_path(
            &mut self.states,
            &mut path,
            &mut self.register,
            &mut self.refcounts,
            &mut self.free_slots,
            0,
            &mut outputs_scratch,
            &mut signature_scratch,
        );
    }

    /// Batch version of `insert`: processes an already-sorted,
    /// deduplicated stream of terms with a single running `path` (à la
    /// `build_sorted`) instead of one independent `insert` call per term,
    /// so states newly shared *within this batch* don't get cloned and
    /// re-confirmed once per term the way N separate `insert` calls
    /// would. `batch_start_slot` marks which states were created by this
    /// call (safe to mutate directly, already private) versus
    /// pre-existing ones (must still be cloned before any write). Returns
    /// whether each term (in the same order as `terms`) was newly added
    /// -- `false` for one already present, a no-op for that term.
    ///
    /// `confirm_path` gets called on states that were only walked through
    /// (matched against an already-present term) and never actually
    /// modified, which looks unsafe on paper -- confirming touches the
    /// parent's transition table. It isn't: recomputing an untouched
    /// state's outputs from its (also unchanged) children reproduces the
    /// same values, and the register lookup then finds that state's own
    /// existing entry (or an equivalent one) as canonical, so the
    /// "redirect" is a no-op in effect. Verified, not assumed --
    /// `insert_sorted_full_hit_before_a_diverging_sibling` and
    /// `insert_sorted_mixed_hits_and_new_across_rounds` target exactly
    /// this pattern (a full-hit term immediately before a diverging one
    /// sharing its prefix) at both a hand-traceable scale and a larger
    /// randomized, multi-round, multi-seed one, before this was trusted.
    /// It's wasted work on those untouched entries, not a correctness
    /// problem -- worth measuring before deciding it's worth avoiding.
    pub(crate) fn insert_sorted<'a>(&mut self, terms: impl Iterator<Item = &'a str>) -> Vec<bool> {
        // Compared against each visited state's *generation*, not its raw
        // slot number: a slot freed (refcount hits zero) and reused by
        // `alloc_state` earlier in this same batch would otherwise land
        // below this snapshot despite being entirely private to this
        // batch -- see the `generations` field's doc comment.
        let batch_start_generation = self.next_generation;
        let mut path: Vec<(StateId, u8)> = vec![(self.root, 0)];
        let mut last_word: Vec<u8> = Vec::new();
        let mut results = Vec::new();
        let mut outputs_scratch = Vec::new();
        let mut signature_scratch = Vec::new();

        for term in terms {
            let word = term.as_bytes();
            let common = common_prefix_len(&last_word, word);

            confirm_path(
                &mut self.states,
                &mut path,
                &mut self.register,
                &mut self.refcounts,
                &mut self.free_slots,
                common + 1,
                &mut outputs_scratch,
                &mut signature_scratch,
            );

            let mut matched = common;
            while matched < word.len() {
                let cur = path.last().expect("path always has at least the root").0;
                match self.states[cur.slot() as usize]
                    .transitions
                    .get(word[matched])
                {
                    Some((_, target)) => {
                        path.push((target, word[matched]));
                        matched += 1;
                    }
                    None => break,
                }
            }

            let is_new = if matched == word.len()
                && self.states[path.last().unwrap().0.slot() as usize].is_final
            {
                false
            } else {
                for i in 1..path.len() {
                    let (original, byte) = path[i];
                    if self.generations[original.slot() as usize] < batch_start_generation {
                        let cloned = clone_state(
                            &mut self.states,
                            &mut self.refcounts,
                            &mut self.generations,
                            &mut self.free_slots,
                            &mut self.next_generation,
                            original,
                        );
                        path[i].0 = cloned;
                        let parent = path[i - 1].0;
                        redirect(
                            &mut self.states,
                            &mut self.refcounts,
                            &mut self.free_slots,
                            &mut self.register,
                            parent,
                            byte,
                            cloned,
                            &mut signature_scratch,
                        );
                    }
                }
                let mut cur = path.last().unwrap().0;
                for &b in &word[matched..] {
                    let new_id = alloc_state(
                        &mut self.states,
                        &mut self.refcounts,
                        &mut self.generations,
                        &mut self.free_slots,
                        &mut self.next_generation,
                        State::default(),
                    );
                    link_new(&mut self.states, &mut self.refcounts, cur, b, new_id);
                    path.push((new_id, b));
                    cur = new_id;
                }
                self.states[cur.slot() as usize].is_final = true;
                true
            };
            results.push(is_new);

            last_word.clear();
            last_word.extend_from_slice(word);
        }

        confirm_path(
            &mut self.states,
            &mut path,
            &mut self.register,
            &mut self.refcounts,
            &mut self.free_slots,
            0,
            &mut outputs_scratch,
            &mut signature_scratch,
        );
        results
    }

    /// Removes `term` if present, preserving minimality (Carrasco &
    /// Forcada 2002's territory -- see the module doc comment's
    /// citations). Returns whether it was present. Walks the exact
    /// matching path (a partial match, or a full match landing on a
    /// non-final state, means the term isn't present -- no-op); clones
    /// every non-root state on that path, copy-on-write, same reasoning
    /// as `insert` (any of them might be shared with other terms); clears
    /// `is_final` at the (now private) leaf; then prunes bottom-up: a
    /// state with no transitions left and not final has no reason to
    /// exist, so its incoming transition is deleted from its parent
    /// outright (not redirected -- `Transitions::remove_transition`, the
    /// one mutation here that isn't a redirect), and the check repeats
    /// one level up, stopping at the first surviving (still-final or
    /// still-has-transitions) state. Whatever survives is re-confirmed
    /// through the same register-based machinery as every other mutation
    /// here -- a state that lost a transition might now coincide with
    /// some other already-registered state and get merged into it, same
    /// as any other confirmation.
    pub(crate) fn remove(&mut self, term: &str) -> bool {
        let word = term.as_bytes();

        let mut path: Vec<(StateId, u8)> = vec![(self.root, 0)];
        for &b in word {
            let cur = path.last().expect("path always has at least the root").0;
            match self.states[cur.slot() as usize].transitions.get(b) {
                Some((_, target)) => path.push((target, b)),
                None => return false,
            }
        }
        if !self.states[path.last().unwrap().0.slot() as usize].is_final {
            return false;
        }

        let mut outputs_scratch = Vec::new();
        let mut signature_scratch = Vec::new();

        for i in 1..path.len() {
            let (original, byte) = path[i];
            let cloned = clone_state(
                &mut self.states,
                &mut self.refcounts,
                &mut self.generations,
                &mut self.free_slots,
                &mut self.next_generation,
                original,
            );
            path[i].0 = cloned;
            let parent = path[i - 1].0;
            redirect(
                &mut self.states,
                &mut self.refcounts,
                &mut self.free_slots,
                &mut self.register,
                parent,
                byte,
                cloned,
                &mut signature_scratch,
            );
        }

        let leaf = path.last().unwrap().0;
        self.states[leaf.slot() as usize].is_final = false;

        while path.len() > 1 {
            let (id, byte) = *path.last().unwrap();
            let dead = !self.states[id.slot() as usize].is_final
                && self.states[id.slot() as usize].transitions.is_empty();
            if !dead {
                break;
            }
            path.pop();
            let parent = path.last().unwrap().0;
            self.states[parent.slot() as usize]
                .transitions
                .remove_transition(byte);
            // `id` had exactly one reference (its now-removed transition
            // from `parent` -- everything on `path` was already made
            // private by the clone loop above), and it has no
            // transitions of its own (`dead` just confirmed that), so
            // this can only free `id`'s own slot, no cascade -- but going
            // through `release` uniformly, rather than special-casing
            // that here, is what keeps this correct if that ever stops
            // being true.
            release(
                &self.states,
                &mut self.refcounts,
                &mut self.free_slots,
                &mut self.register,
                id,
                &mut signature_scratch,
            );
        }

        confirm_path(
            &mut self.states,
            &mut path,
            &mut self.register,
            &mut self.refcounts,
            &mut self.free_slots,
            0,
            &mut outputs_scratch,
            &mut signature_scratch,
        );
        true
    }

    /// `Some(rank)` if `term` is present -- the forward direction,
    /// mirroring `Block::rank`.
    pub(crate) fn rank(&self, term: &str) -> Option<u32> {
        let mut state = self.root;
        let mut value = 0u32;
        for &b in term.as_bytes() {
            let (output, target) = self.states[state.slot() as usize].transitions.get(b)?;
            value += output;
            state = target;
        }
        self.states[state.slot() as usize].is_final.then_some(value)
    }

    /// Decodes the term at `rank` into `buf` (cleared first) -- the
    /// reverse direction, mirroring `Block::term_at_into`. Delegates the
    /// "last transition whose output is `<= value`" search at each state
    /// to `Transitions::last_output_at_most`, a direct loop -- a `perf`
    /// comparison against `Block` (DESIGN.md's "decode speed" section)
    /// found the generic `Iterator` combinator chain this used to go
    /// through, not a linear scan itself, was the real cost. Binary
    /// search (matching `block.rs`'s own history) is still a candidate if
    /// a future profile shows the scan itself matters at the branching
    /// factors this corpus actually has -- not needed yet.
    pub(crate) fn term_at_into(&self, rank: u32, buf: &mut Vec<u8>) {
        buf.clear();
        let mut value = rank;
        let mut state = self.root;
        loop {
            let s = &self.states[state.slot() as usize];
            if value == 0 && s.is_final {
                return;
            }
            let (b, out, target) = s.transitions.last_output_at_most(value);
            value -= out;
            buf.push(b);
            state = target;
        }
    }

    /// A streaming walk of every term in sorted (rank) order -- the
    /// incremental counterpart to `term_at_into`, and the reason it
    /// exists: repeated `term_at_into(rank, ..)` calls always restart
    /// from `root`, paying the full path length on every single call,
    /// while `Block::decode_into`/`merge_into` use `fst::Map::stream()`,
    /// which reuses shared-prefix traversal state between consecutive
    /// keys. That asymmetry -- not `term_at_into` itself, already
    /// measured cheaper than `Block::term_at_into`'s own equivalent
    /// point-lookup in a fair, apples-to-apples comparison (see
    /// `perf_probe_decode_cost_automaton`) -- is what
    /// `AutomatonBlock::merge_into`'s single-pass rewrite was still
    /// paying for (DESIGN.md's "measured end to end" section): a
    /// per-state stack means a shared prefix between two consecutive
    /// terms is walked once, not once per term.
    pub(crate) fn stream(&self) -> TermsStream<'_> {
        TermsStream {
            automaton: self,
            buf: Vec::new(),
            stack: vec![StreamFrame {
                state: self.root,
                next_child: 0,
                yielded_final: false,
            }],
        }
    }

    /// Like `term_at_into` (a direct root-to-leaf descent to `rank`), but
    /// returns a positioned `TermsStream` instead of just the decoded
    /// bytes: `result.current()` is `rank`'s own term (same bytes
    /// `term_at_into` would have written), and calling `.next()` on the
    /// returned stream yields `rank + 1`, `rank + 2`, ... via ordinary
    /// incremental advancement -- reusing whatever prefix `rank`'s
    /// descent already established, the same shared-prefix win `stream()`
    /// gives two *consecutive* `next()` calls, but starting from an
    /// arbitrary rank instead of being stuck starting from 0.
    ///
    /// Exists to test a specific hypothesis about scattered bulk decode:
    /// `term_at_into` called once per id in a requested set always pays
    /// the full descent cost per call, even when consecutive requested
    /// ranks are close together and share most of their path -- this
    /// pays that full cost once, for the first id in a run, and lets
    /// `next()` pick up the shared-prefix reuse for the rest. Whether
    /// that actually wins in practice (traversal-reuse savings vs. the
    /// extra bookkeeping to build a resumable stack instead of a flat
    /// buffer) is exactly what needs measuring, not assumed -- see
    /// `s-bench`'s `dict_bench` for the real-data comparison.
    pub(crate) fn seek_stream(&self, rank: u32) -> TermsStream<'_> {
        let mut value = rank;
        let mut state = self.root;
        let mut buf = Vec::new();
        let mut stack = Vec::new();
        loop {
            let s = &self.states[state.slot() as usize];
            if value == 0 && s.is_final {
                stack.push(StreamFrame {
                    state,
                    next_child: 0,
                    yielded_final: true,
                });
                return TermsStream {
                    automaton: self,
                    buf,
                    stack,
                };
            }
            let (idx, b, out, target) = s.transitions.last_output_at_most_with_index(value);
            stack.push(StreamFrame {
                state,
                next_child: idx as u32 + 1,
                yielded_final: true,
            });
            value -= out;
            buf.push(b);
            state = target;
        }
    }

    /// The total number of terms this automaton accepts.
    pub(crate) fn len(&self) -> usize {
        self.states[self.root.slot() as usize].count as usize
    }

    /// The total size of the state slab, reachable garbage included --
    /// for measuring, not assuming, how much the known slot-reclamation
    /// gap (see the struct doc comment) actually costs under sustained
    /// mutation. Compare against `reachable_state_count`.
    #[cfg(test)]
    pub(crate) fn state_slab_len(&self) -> usize {
        self.states.len()
    }

    /// The number of *reachable* states -- the true minimality signal.
    /// `states.len()` alone would overcount, since bulk construction and
    /// incremental insertion both leave discarded duplicates as
    /// unreachable garbage in the slab (see the struct doc comment); this
    /// walks the actual live graph.
    #[cfg(test)]
    pub(crate) fn reachable_state_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                for (_, _, target) in self.states[id.slot() as usize].transitions.iter() {
                    stack.push(target);
                }
            }
        }
        seen.len()
    }

    /// Logical byte size of the *reachable* portion of this automaton's
    /// state graph -- the true minimal automaton, not the slab (which can
    /// hold register-deduplication garbage from bulk construction or
    /// incremental mutation; see the struct doc comment and
    /// `reachable_state_count`, whose walk this mirrors). The
    /// `Block::payload_bytes` analog: sums each reachable state's fixed
    /// cost (`State`, which holds its inline transition chunk by value,
    /// no allocation) plus any `overflow` chunks' heap bytes.
    ///
    /// Excludes `register`/`refcounts`/`free_slots`/`generations` --
    /// incremental-mutation bookkeeping this implementation carries that
    /// has no `Block`/`fst::Map` equivalent at all (an FST is rebuilt
    /// from scratch on every mutation, so it never needs any of this).
    /// So this is the fair, apples-to-apples number against
    /// `Block::payload_bytes` (both report "the compiled index
    /// structure's own size"), not this automaton's complete memory
    /// footprint under sustained mutation -- a caller wanting that should
    /// treat this as a lower bound. Does not account for allocator slack
    /// (`Vec` spare capacity) or heap allocation headers, the same
    /// simplification `Block::payload_bytes` makes.
    pub(crate) fn size_bytes(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![self.root];
        let mut total = 0usize;
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                let state = &self.states[id.slot() as usize];
                total += std::mem::size_of::<State>();
                total += state.transitions.overflow.len() * std::mem::size_of::<TransitionChunk>();
                for (_, _, target) in state.transitions.iter() {
                    stack.push(target);
                }
            }
        }
        total
    }
}

struct StreamFrame {
    state: StateId,
    /// Index into this state's `Transitions` of the next child to
    /// descend into (`Transitions::at`) -- an explicit position instead
    /// of a stored iterator so no per-state allocation is needed even
    /// with `Transitions::iter`'s combinator chain measured costly
    /// elsewhere (see `Transitions::iter`'s doc comment).
    next_child: u32,
    /// Whether this frame's own value (if `is_final`) has already been
    /// yielded -- a state is visited (and, if final, yielded) *before*
    /// its children, matching sorted order: any string through a final
    /// state is a proper extension of it, hence lexicographically
    /// greater.
    yielded_final: bool,
}

/// A streaming, incremental walk of every term in an [`Automaton`], in
/// sorted order -- see [`Automaton::stream`]'s doc comment for why this
/// exists alongside `term_at_into`. Not an `Iterator` (no borrowed-buffer
/// GAT in this edition), so `next` is a bespoke method returning a
/// reference into `self`'s own reused buffer, the same shape
/// `term_at_into` already uses.
pub(crate) struct TermsStream<'a> {
    automaton: &'a Automaton,
    buf: Vec<u8>,
    /// The current root-to-frontier path; `stack[i+1]` was reached by the
    /// byte at `buf[i]`, so `stack.len() - 1 == buf.len()` always (the
    /// root frame contributes no byte).
    stack: Vec<StreamFrame>,
}

impl<'a> TermsStream<'a> {
    /// The term this stream is currently positioned at -- for a stream
    /// from `Automaton::seek_stream(rank)`, that's `rank`'s own term
    /// (identical bytes to what `term_at_into(rank, ..)` would have
    /// written), before any `next()` call has advanced past it. Not
    /// meaningful on a fresh `stream()` (nothing decoded yet; call
    /// `next()` first).
    pub(crate) fn current(&self) -> &[u8] {
        &self.buf
    }

    /// Advances to the next term in sorted order, or `None` once
    /// exhausted. A preorder DFS over the trie using `stack` in place of
    /// recursion: visit a state's own value (if final) before its
    /// children, children in byte order (already guaranteed by
    /// `Transitions`' sortedness) -- the same traversal shape
    /// `fst::Map::stream()` performs internally, reimplemented here so
    /// consecutive calls share the cost of any common prefix instead of
    /// each re-walking it from `root`.
    pub(crate) fn next(&mut self) -> Option<&[u8]> {
        loop {
            let depth = self.stack.len();
            if depth == 0 {
                return None;
            }
            let state_id = self.stack[depth - 1].state;
            let already_yielded = self.stack[depth - 1].yielded_final;
            self.stack[depth - 1].yielded_final = true;
            let state = &self.automaton.states[state_id.slot() as usize];
            if !already_yielded && state.is_final {
                return Some(&self.buf);
            }
            let next_child = self.stack[depth - 1].next_child;
            match state.transitions.at(next_child as usize) {
                Some((b, _out, target)) => {
                    self.stack[depth - 1].next_child += 1;
                    self.buf.push(b);
                    self.stack.push(StreamFrame {
                        state: target,
                        next_child: 0,
                        yielded_final: false,
                    });
                }
                None => {
                    self.stack.pop();
                    if depth > 1 {
                        self.buf.pop();
                    }
                }
            }
        }
    }
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Allocates a fresh state slot holding `content`, reusing a dead one
/// from `free_slots` (refcount already zero -- see `release`) if one is
/// available, only growing `states`/`refcounts`/`generations` otherwise.
/// This is *the* slot-reclamation fix: without it, every clone/creation
/// grows the slab forever, which DESIGN.md's "slot-reclamation gap,
/// measured" section confirmed was a real, severe (500x+) cost under
/// sustained mutation, not a theoretical one.
fn alloc_state(
    states: &mut Vec<State>,
    refcounts: &mut Vec<u32>,
    generations: &mut Vec<u64>,
    free_slots: &mut Vec<u32>,
    next_generation: &mut u64,
    content: State,
) -> StateId {
    let generation = *next_generation;
    *next_generation += 1;
    if let Some(slot) = free_slots.pop() {
        states[slot as usize] = content;
        refcounts[slot as usize] = 0;
        generations[slot as usize] = generation;
        StateId::from_slot(slot)
    } else {
        states.push(content);
        refcounts.push(0);
        generations.push(generation);
        StateId::from_slot((states.len() - 1) as u32)
    }
}

/// Allocates a fresh state that starts as an exact copy of `original`'s
/// content (same transitions, same finality) -- used when incremental
/// insertion needs to change a state that might be shared with other
/// terms: the original is left untouched (still valid for whoever else
/// references it) and the clone becomes what this path points at instead.
///
/// Copying `original`'s transitions duplicates every reference they
/// hold: the clone is a *new*, independent referrer to each of those
/// targets, exactly as if each transition had just been created fresh
/// (`link_new`) -- so each target needs its own `acquire` too, not just
/// the clone's own slot needing to exist. Missing this is a real bug
/// this implementation had and a test caught immediately (three or more
/// chained clones deep -- DESIGN.md's reference-counting section): the
/// clone's *own* new reference to a child was never recorded, so
/// releasing the original later wrongly cascaded into treating that
/// child (and everything under it) as unreferenced too, even though the
/// clone still pointed at it.
fn clone_state(
    states: &mut Vec<State>,
    refcounts: &mut Vec<u32>,
    generations: &mut Vec<u64>,
    free_slots: &mut Vec<u32>,
    next_generation: &mut u64,
    original: StateId,
) -> StateId {
    let content = states[original.slot() as usize].clone();
    // A direct loop over `Transitions::at`, not `iter().map(..).collect()`
    // -- this call site was missed when `Signature::of`/
    // `finalize_outputs_and_count` got the same fix (see `Transitions::
    // iter`'s doc comment for why the combinator chain, not the scan
    // itself, is the cost); a fresh `perf record` of `sequential_insert`
    // found it as a top-6 cost (DESIGN.md's allocation-pattern audit).
    let mut inherited_targets: Vec<StateId> = Vec::new();
    let mut i = 0usize;
    while let Some((_, _, target)) = content.transitions.at(i) {
        inherited_targets.push(target);
        i += 1;
    }
    let cloned = alloc_state(
        states,
        refcounts,
        generations,
        free_slots,
        next_generation,
        content,
    );
    for target in inherited_targets {
        acquire(refcounts, target);
    }
    cloned
}

/// Records one more transition pointing at `id`.
fn acquire(refcounts: &mut [u32], id: StateId) {
    refcounts[id.slot() as usize] += 1;
}

/// Records one fewer transition pointing at `id`. If that was the last
/// one, `id` is dead: its own transitions no longer effectively exist
/// either (nothing reaches them *through* a dead state), so each of its
/// children is released too -- a real cascade, not just a single free,
/// since a whole now-unreachable subtree can hang off one redirected
/// transition. The freed slot goes on `free_slots` for `alloc_state` to
/// reuse -- and, critically, if `id` was ever a register entry's
/// canonical value, that entry is removed too: leaving it would let a
/// *future* lookup match `id`'s old signature and hand back a `StateId`
/// whose slot has since been reused for completely unrelated content, a
/// silent wrong-answer bug, not a crash, so this is checked unconditionally
/// rather than assumed safe to skip. Never called with `id == root`: root
/// is never a transition target (this automaton is acyclic, root is the
/// unique entry point), so its count is never tracked and can never reach
/// zero here.
fn release(
    states: &[State],
    refcounts: &mut Vec<u32>,
    free_slots: &mut Vec<u32>,
    register: &mut Register,
    id: StateId,
    signature_scratch: &mut Vec<(u8, u32, StateId)>,
) {
    let slot = id.slot() as usize;
    debug_assert!(
        refcounts[slot] > 0,
        "refcount underflow releasing a live state"
    );
    refcounts[slot] -= 1;
    if refcounts[slot] == 0 {
        let sig = signature_of_into(&states[slot], signature_scratch);
        if register.get(&sig) == Some(&id) {
            register.remove(&sig);
        }
        // Unlike `confirm_path`'s hit case, `sig` is never moved into
        // `register` here either way -- `get`/`remove` both take it by
        // reference -- so the buffer can always be reclaimed, not just
        // conditionally.
        *signature_scratch = sig.transitions;
        free_slots.push(slot as u32);
        // No `children: Vec<StateId>` collected first -- `states` is
        // only ever borrowed immutably through this whole call tree
        // (`release` never mutates state content, only `refcounts`/
        // `free_slots`/`register`), so `states[slot]`'s transitions
        // can't change out from under this loop; iterating them directly
        // while also passing `states` into the recursive call is just
        // two simultaneous immutable borrows, always allowed. The old
        // `Vec` was collecting into an intermediate that served no
        // purpose here (DESIGN.md's allocation-pattern audit).
        let mut i = 0usize;
        while let Some((_, _, child)) = states[slot].transitions.at(i) {
            release(
                states,
                refcounts,
                free_slots,
                register,
                child,
                signature_scratch,
            );
            i += 1;
        }
    }
}

/// Creates a brand-new transition from `parent` (which must not already
/// have one for `byte`) to `target`, and records the reference.
fn link_new(
    states: &mut [State],
    refcounts: &mut [u32],
    parent: StateId,
    byte: u8,
    target: StateId,
) {
    states[parent.slot() as usize]
        .transitions
        .insert_transition(byte, target);
    acquire(refcounts, target);
}

/// Same as `link_new`, but for callers that already guarantee `byte`
/// sorts after every transition `parent` currently has -- `build_sorted`
/// and `insert_sorted`'s own "extend with brand-new bytes" loops, both
/// processing already-sorted input, unlike `Automaton::insert` (a single,
/// order-independent incremental insert, which keeps using `link_new`;
/// see `Transitions::insert_transition`'s doc comment for why it can't
/// use this). Calls `Transitions::append_in_order` directly instead of
/// `insert_transition`'s full rebuild-from-a-sorted-`Vec` -- a `perf
/// record` profile of `build_sorted` (DESIGN.md's "closing the
/// bulk-construction gap" section) found that rebuild, paid on every
/// single state creation via `link_new`, was a dominant cost: an
/// allocation, plus `Transitions::iter()`'s combinator chain, already
/// found costly elsewhere in this module (see `iter`'s own doc comment),
/// for work that was always going to end in "append at the end" here.
fn link_new_appending(
    states: &mut [State],
    refcounts: &mut [u32],
    parent: StateId,
    byte: u8,
    target: StateId,
) {
    states[parent.slot() as usize]
        .transitions
        .append_in_order(byte, target);
    acquire(refcounts, target);
}

/// Repoints `parent`'s existing transition for `byte` (which must already
/// exist) from its current target to `new_target`, keeping reference
/// counts correct either way. Acquires `new_target` *before* releasing
/// the old one -- not the other way around -- specifically so the
/// extremely common case of `new_target` being the *same* state (a
/// freshly confirmed state registering as its own canonical entry, which
/// happens for essentially every genuinely new piece of structure) can't
/// see its refcount bounce through zero and get wrongly cascade-freed a
/// moment before being re-acquired.
///
/// 8 parameters, over clippy's default threshold: every one of these
/// low-level state-slab functions already travels with most of this
/// same set (`states`/`refcounts`/`free_slots`/`register`), and
/// `signature_scratch` (the reused buffer `release` needs -- see
/// `signature_of_into`'s doc comment) is one more of the same kind, not
/// a sign this function is doing too much. Bundling the slab fields into
/// a context struct would be a real, separate refactor touching every
/// function in this module, not something to fold into this change.
#[allow(clippy::too_many_arguments)]
fn redirect(
    states: &mut [State],
    refcounts: &mut Vec<u32>,
    free_slots: &mut Vec<u32>,
    register: &mut Register,
    parent: StateId,
    byte: u8,
    new_target: StateId,
    signature_scratch: &mut Vec<(u8, u32, StateId)>,
) {
    let old_target = states[parent.slot() as usize]
        .transitions
        .get(byte)
        .expect("byte must already have a transition here")
        .1;
    acquire(refcounts, new_target);
    states[parent.slot() as usize]
        .transitions
        .set_target_for_byte(byte, new_target);
    release(
        states,
        refcounts,
        free_slots,
        register,
        old_target,
        signature_scratch,
    );
}

/// Confirms (register-deduplicates, then computes outputs/count for)
/// every state on `path` from the end down to `keep_from`, deepest first
/// -- a state's signature depends on its children's *canonical* ids, so
/// children must be confirmed strictly before their parent. Truncates
/// `path` to length `max(keep_from, 1)` (the root, at index 0, is always
/// kept on the path and confirmed in place at the very end of
/// construction, but never removed or merged -- it's the unique entry
/// point, nothing "above" it to redirect). Redirects by the byte each
/// path entry was reached on, not by position -- required for
/// incremental `insert`, where the confirmed transition need not be the
/// last one at its parent (an existing state can have transitions for
/// several bytes; `build_sorted` only ever happens to touch the last one,
/// since it always extends the lexicographically newest path).
/// `outputs_scratch`/`signature_scratch`: reused buffers owned by the
/// top-level caller (`build_sorted`/`insert_sorted`/`insert`/`remove`),
/// threaded through so a whole operation's worth of confirmations shares
/// two allocations instead of paying for two per confirmed state -- see
/// `finalize_outputs_and_count` and `signature_of_into`'s own doc
/// comments for why each is safe to reuse this way.
///
/// 8 parameters, over clippy's default threshold -- see `redirect`'s own
/// doc comment for why that's not being treated as a signal to refactor
/// as part of this change.
#[allow(clippy::too_many_arguments)]
fn confirm_path(
    states: &mut [State],
    path: &mut Vec<(StateId, u8)>,
    register: &mut Register,
    refcounts: &mut Vec<u32>,
    free_slots: &mut Vec<u32>,
    keep_from: usize,
    outputs_scratch: &mut Vec<u32>,
    signature_scratch: &mut Vec<(u8, u32, StateId)>,
) {
    while path.len() > keep_from.max(1) {
        let (id, byte) = path.pop().expect("loop guard ensures at least one entry");
        let parent = path.last().expect("path always has at least the root").0;

        finalize_outputs_and_count(states, id, outputs_scratch);

        let sig = signature_of_into(&states[id.slot() as usize], signature_scratch);
        let canonical = match register.get(&sig) {
            Some(&existing) => {
                // Hit: `sig` is about to be dropped -- reclaim its
                // buffer for the next confirmation instead of letting it
                // deallocate (see `signature_of_into`'s doc comment).
                *signature_scratch = sig.transitions;
                existing // `id` released by `redirect` below if it loses out
            }
            None => {
                register.insert(sig, id);
                id
            }
        };
        redirect(
            states,
            refcounts,
            free_slots,
            register,
            parent,
            byte,
            canonical,
            signature_scratch,
        );
    }
    if keep_from == 0 {
        finalize_outputs_and_count(states, path[0].0, outputs_scratch);
    }
}

/// Computes and stores `id`'s outgoing transitions' output values (the
/// prefix-sum-of-sibling-subtree-counts technique, see the module doc
/// comment) and `id`'s own `count`, assuming every transition's target is
/// already confirmed (its `count` already final).
///
/// `outputs_scratch` is reused across calls (cleared, not reallocated):
/// this buffer never escapes the function (it's read back out by
/// `set_outputs` and then no longer needed), so unlike `Signature`'s own
/// buffer (see its doc comment) there's no ownership complication at all
/// -- a plain `mem::take`-free "the caller keeps handing back the same
/// `Vec`" scratch, the simplest case of the two `Vec`s a `perf record`
/// profile of `build_sorted` found allocating on every confirmed state
/// (DESIGN.md's "extensive analysis" / "scratch buffers" sections).
fn finalize_outputs_and_count(states: &mut [State], id: StateId, outputs_scratch: &mut Vec<u32>) {
    let is_final = states[id.slot() as usize].is_final;
    // A word accepted *at this state* is a prefix of every word reached
    // by descending into any child, and a prefix always sorts before its
    // own extensions -- so this state's own acceptance shifts every
    // child's output by one, not just this state's `count` (which needed
    // the same `+1`, so folding it into the running total up front gives
    // both for free).
    let mut running = u32::from(is_final);
    // A direct loop over `Transitions::at`, not `iter().map(..).collect()`
    // (also drops the separate `targets` Vec that used to sit between
    // them): this project's own earlier `perf`-measured decode comparison
    // already found the combinator chain, not the scan itself, was the
    // real cost here (see `Transitions::iter`'s doc comment) -- confirmed
    // again by a `perf record` profile of `build_sorted` naming this
    // exact call site (DESIGN.md's "closing the bulk-construction gap"
    // section).
    outputs_scratch.clear();
    let mut i = 0usize;
    while let Some((_, _, target)) = states[id.slot() as usize].transitions.at(i) {
        outputs_scratch.push(running);
        running += states[target.slot() as usize].count;
        i += 1;
    }
    states[id.slot() as usize]
        .transitions
        .set_outputs(outputs_scratch);
    states[id.slot() as usize].count = running;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TermId;
    use crate::block::Block;
    use crate::entries_arena::{EntriesArena, merge_into};
    use std::time::Instant;

    /// Targeted repro for a specific suspicion, not yet confirmed: does
    /// `confirm_path`, when a batch's running `path` includes a state
    /// that was only *walked through* (matched against an already-present
    /// term, never modified) rather than newly created/cloned, end up
    /// treating that untouched state as something needing confirmation --
    /// and if the register happens to redirect it somewhere, does that
    /// corrupt an unrelated sibling that shares the same parent?
    ///
    /// Setup: pre-existing "xa" and "xb" (siblings under a shared parent
    /// `P`, `P` reached only via `x` from root). Batch: `["xa", "xc"]`,
    /// sorted -- "xa" is a duplicate (full hit, no modification needed);
    /// "xc" is new and diverges from `P`. Processing "xa" first walks
    /// `root -> P -> FinalA` and pushes both onto the shared running
    /// `path` even though nothing about the walk needs changing; the
    /// question is whether processing "xc" next (which pops FinalA via
    /// `confirm_path`, then must clone `P` to add a `c` transition)
    /// leaves `P` and its *existing* `a`/`b` transitions intact.
    #[test]
    fn insert_sorted_full_hit_before_a_diverging_sibling() {
        let mut pre_existing = ["xa", "xb"].to_vec();
        pre_existing.sort_unstable();
        let mut automaton = Automaton::build_sorted(pre_existing.iter().copied());

        let results = automaton.insert_sorted(["xa", "xc"].into_iter());
        assert_eq!(results, vec![false, true], "xa is a duplicate, xc is new");

        let mut expected = ["xa", "xb", "xc"];
        expected.sort_unstable();
        let oracle = Automaton::build_sorted(expected.iter().copied());

        for (rank, term) in expected.iter().enumerate() {
            assert_eq!(
                automaton.rank(term),
                Some(rank as u32),
                "rank mismatch for {term:?} after batch insert"
            );
        }
        assert_eq!(automaton.len(), expected.len());
        assert_eq!(
            automaton.reachable_state_count(),
            oracle.reachable_state_count(),
            "batch insert left the automaton non-minimal"
        );
    }

    /// Broader, randomized version of the same suspicion: several rounds
    /// of sorted batches, each batch a mix of terms already present
    /// (full hits, interspersed) and genuinely new ones (diverging at
    /// varying depths), against pre-existing content with real branching
    /// (namespace-clustered, not a hand-picked 3-term case). If the
    /// single targeted repro above passed by luck rather than because the
    /// suspicion was simply wrong, a larger, multi-round, multi-seed
    /// version is what should surface it.
    #[test]
    fn insert_sorted_mixed_hits_and_new_across_rounds() {
        for seed in [1u64, 0xC0FFEE, 0xDEADBEEF, 999_999_937] {
            let pool = namespace_clustered_terms(600); // 2,400 distinct terms
            let mut live: Vec<String> = pool.iter().take(400).cloned().collect();
            live.sort_unstable();
            live.dedup();
            let mut automaton = Automaton::build_sorted(live.iter().map(String::as_str));

            let mut rng = seed;
            for round in 0..8 {
                // A mix of already-live terms (duplicates) and terms not
                // yet inserted (new), sorted and deduplicated, matching
                // what a real caller's prepared batch looks like.
                let mut batch: Vec<String> = Vec::new();
                for _ in 0..40 {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    batch.push(pool[(rng as usize) % pool.len()].clone());
                }
                batch.sort_unstable();
                batch.dedup();

                let results = automaton.insert_sorted(batch.iter().map(String::as_str));
                assert_eq!(results.len(), batch.len(), "seed {seed:#x} round {round}");
                for (term, &was_new) in batch.iter().zip(&results) {
                    assert_eq!(
                        was_new,
                        !live.contains(term),
                        "seed {seed:#x} round {round}: was_new mismatch for {term:?}"
                    );
                }

                for term in &batch {
                    if !live.contains(term) {
                        live.push(term.clone());
                    }
                }
                live.sort_unstable();
                live.dedup();

                let oracle = Automaton::build_sorted(live.iter().map(String::as_str));
                assert_eq!(
                    automaton.len(),
                    live.len(),
                    "seed {seed:#x} round {round}: len mismatch"
                );
                for (rank, term) in live.iter().enumerate() {
                    assert_eq!(
                        automaton.rank(term),
                        Some(rank as u32),
                        "seed {seed:#x} round {round}: rank mismatch for {term:?}"
                    );
                }
                assert_eq!(
                    automaton.reachable_state_count(),
                    oracle.reachable_state_count(),
                    "seed {seed:#x} round {round}: non-minimal after batch insert"
                );
            }
        }
    }

    #[test]
    fn remove_absent_term_is_a_no_op() {
        let mut automaton = Automaton::build_sorted(["xa", "xb"].into_iter());
        let before_states = automaton.reachable_state_count();
        assert!(!automaton.remove("xc")); // never existed
        assert!(!automaton.remove("x")); // a prefix, never itself inserted
        assert_eq!(automaton.reachable_state_count(), before_states);
        assert_eq!(automaton.rank("xa"), Some(0));
        assert_eq!(automaton.rank("xb"), Some(1));
    }

    #[test]
    fn remove_only_term_empties_the_automaton() {
        let mut automaton = Automaton::build_sorted(["xa"].into_iter());
        assert!(automaton.remove("xa"));
        assert_eq!(automaton.len(), 0);
        assert_eq!(automaton.rank("xa"), None);
        assert_eq!(automaton.reachable_state_count(), 1); // just the root
    }

    /// The leaf being removed might still be needed as a prefix of a
    /// longer surviving term -- it should stop being final without being
    /// pruned away.
    #[test]
    fn remove_term_that_is_a_prefix_of_a_surviving_term() {
        let mut automaton = Automaton::build_sorted(["xa", "xab"].into_iter());
        assert!(automaton.remove("xa"));
        assert_eq!(automaton.rank("xa"), None);
        assert_eq!(automaton.rank("xab"), Some(0));
        assert_eq!(automaton.len(), 1);

        let oracle = Automaton::build_sorted(["xab"].into_iter());
        assert_eq!(
            automaton.reachable_state_count(),
            oracle.reachable_state_count()
        );
    }

    /// A hand-traceable repro for the pattern the batch-insert suspicion
    /// was built around, applied to removal instead: does removing one
    /// sibling under a shared parent (`P`, reached from root via `x`,
    /// with children `a` -> FinalA and `b` -> FinalB) leave the *other*
    /// sibling's transition intact, or does pruning/cloning along the
    /// removed path accidentally disturb it?
    #[test]
    fn remove_one_sibling_leaves_the_other_intact() {
        let mut automaton = Automaton::build_sorted(["xa", "xb"].into_iter());
        assert!(automaton.remove("xa"));
        assert_eq!(automaton.rank("xa"), None);
        assert_eq!(automaton.rank("xb"), Some(0));
        assert_eq!(automaton.len(), 1);

        let oracle = Automaton::build_sorted(["xb"].into_iter());
        assert_eq!(
            automaton.reachable_state_count(),
            oracle.reachable_state_count()
        );
    }

    #[test]
    fn remove_then_reinsert_matches_a_fresh_build() {
        let mut automaton = Automaton::build_sorted(["xa", "xb", "xc"].into_iter());
        assert!(automaton.remove("xb"));
        automaton.insert("xb");

        let oracle = Automaton::build_sorted(["xa", "xb", "xc"].into_iter());
        for (rank, term) in ["xa", "xb", "xc"].into_iter().enumerate() {
            assert_eq!(automaton.rank(term), Some(rank as u32));
        }
        assert_eq!(
            automaton.reachable_state_count(),
            oracle.reachable_state_count(),
            "remove-then-reinsert left the automaton non-minimal"
        );
    }

    /// The broad, randomized check: many rounds mixing `insert`,
    /// `insert_sorted`, and `remove` against real namespace-clustered
    /// branching content, differentially verified (every surviving rank,
    /// and minimality) against a fresh `build_sorted` oracle over the
    /// exact current live set after every single round -- the same
    /// standard `insert_sorted_mixed_hits_and_new_across_rounds` was held
    /// to, now covering deletion and the interactions between all three
    /// mutating operations, not just insertion.
    #[test]
    fn mixed_insert_batch_and_remove_across_rounds_matches_a_fresh_build() {
        for seed in [1u64, 0xC0FFEE, 0xDEADBEEF, 999_999_937] {
            let pool = namespace_clustered_terms(600); // 2,400 distinct terms
            let mut live: Vec<String> = pool.iter().take(300).cloned().collect();
            live.sort_unstable();
            live.dedup();
            let mut automaton = Automaton::build_sorted(live.iter().map(String::as_str));

            let mut rng = seed;
            let mut next_rand = |bound: usize| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng as usize) % bound
            };

            for round in 0..12 {
                match round % 3 {
                    0 => {
                        // Single-term insert.
                        let term = pool[next_rand(pool.len())].clone();
                        automaton.insert(&term);
                        if !live.contains(&term) {
                            live.push(term);
                        }
                    }
                    1 => {
                        // Sorted batch insert, mixed hits and new.
                        let mut batch: Vec<String> = (0..30)
                            .map(|_| pool[next_rand(pool.len())].clone())
                            .collect();
                        batch.sort_unstable();
                        batch.dedup();
                        automaton.insert_sorted(batch.iter().map(String::as_str));
                        for term in batch {
                            if !live.contains(&term) {
                                live.push(term);
                            }
                        }
                    }
                    _ => {
                        // Removal -- only from what's actually live, so
                        // some calls hit real content, not just no-ops.
                        if !live.is_empty() {
                            let idx = next_rand(live.len());
                            let term = live.remove(idx);
                            assert!(
                                automaton.remove(&term),
                                "seed {seed:#x} round {round}: remove({term:?}) should have found it"
                            );
                        }
                    }
                }
                live.sort_unstable();
                live.dedup();

                let oracle = Automaton::build_sorted(live.iter().map(String::as_str));
                assert_eq!(
                    automaton.len(),
                    live.len(),
                    "seed {seed:#x} round {round}: len mismatch"
                );
                for (rank, term) in live.iter().enumerate() {
                    assert_eq!(
                        automaton.rank(term),
                        Some(rank as u32),
                        "seed {seed:#x} round {round}: rank mismatch for {term:?}"
                    );
                    let mut buf = Vec::new();
                    automaton.term_at_into(rank as u32, &mut buf);
                    assert_eq!(
                        std::str::from_utf8(&buf).unwrap(),
                        term.as_str(),
                        "seed {seed:#x} round {round}: unrank mismatch at rank {rank}"
                    );
                }
                assert_eq!(
                    automaton.reachable_state_count(),
                    oracle.reachable_state_count(),
                    "seed {seed:#x} round {round}: non-minimal"
                );
            }
        }
    }

    /// Targets the reference-counting/slot-reuse interaction directly,
    /// not just incidentally: `insert_sorted`'s "created during this
    /// batch" check compares *generations*, not raw slot numbers,
    /// specifically because a slot freed and reused mid-batch would
    /// otherwise land below the batch's starting slot count while being
    /// entirely private to it (see the `generations` field's doc
    /// comment). To actually exercise that, `free_slots` needs to be
    /// non-empty *going into* an `insert_sorted` call -- churn rounds
    /// (single insert immediately followed by a remove, guaranteed to
    /// free at least one slot) are interleaved with large batch inserts
    /// for exactly that reason, not for realism.
    #[test]
    fn insert_sorted_after_churn_reuses_slots_correctly() {
        for seed in [1u64, 0xC0FFEE, 0xDEADBEEF] {
            let pool = namespace_clustered_terms(500); // 2,000 distinct terms
            let mut live: Vec<String> = pool.iter().take(200).cloned().collect();
            live.sort_unstable();
            live.dedup();
            let mut automaton = Automaton::build_sorted(live.iter().map(String::as_str));

            let mut rng = seed;
            let mut next_rand = |bound: usize| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng as usize) % bound
            };

            for round in 0..10 {
                // Churn: insert one new term, immediately remove it --
                // guarantees at least one freed slot before the batch
                // below, without changing the live set.
                for _ in 0..20 {
                    let churn_term = format!(
                        "http://example.org/churn-{seed:x}-{round}-{}",
                        next_rand(1_000_000)
                    );
                    automaton.insert(&churn_term);
                    assert!(automaton.remove(&churn_term));
                }

                let mut batch: Vec<String> = (0..80)
                    .map(|_| pool[next_rand(pool.len())].clone())
                    .collect();
                batch.sort_unstable();
                batch.dedup();
                automaton.insert_sorted(batch.iter().map(String::as_str));
                for term in batch {
                    if !live.contains(&term) {
                        live.push(term);
                    }
                }
                live.sort_unstable();
                live.dedup();

                let oracle = Automaton::build_sorted(live.iter().map(String::as_str));
                assert_eq!(automaton.len(), live.len(), "seed {seed:#x} round {round}");
                for (rank, term) in live.iter().enumerate() {
                    assert_eq!(
                        automaton.rank(term),
                        Some(rank as u32),
                        "seed {seed:#x} round {round}: rank mismatch for {term:?}"
                    );
                }
                assert_eq!(
                    automaton.reachable_state_count(),
                    oracle.reachable_state_count(),
                    "seed {seed:#x} round {round}: non-minimal after post-churn batch insert"
                );
            }
        }
    }

    /// Measures, doesn't assume, the known slot-reclamation gap (the
    /// struct doc comment's "discarded clones/originals are left as
    /// unreachable garbage... a real, unaddressed memory-growth cost"):
    /// sustained one-out-one-in churn (remove one live term, insert one
    /// never-seen one, keeping the live set size constant) for many
    /// rounds, printing the state slab's total size against its
    /// reachable size after every round. If the gap is real, the slab
    /// should grow roughly with cumulative mutation count while the live
    /// term count -- and so the reachable state count -- stays flat.
    #[test]
    #[ignore = "manual measurement, not part of the regular suite"]
    fn measure_slot_reclamation_gap_under_sustained_churn() {
        let mut live = namespace_clustered_terms(200); // 800 terms
        live.sort_unstable();
        live.dedup();
        let mut automaton = Automaton::build_sorted(live.iter().map(String::as_str));

        let initial_slab = automaton.state_slab_len();
        let initial_reachable = automaton.reachable_state_count();
        eprintln!("round 0: slab={initial_slab} reachable={initial_reachable}");

        let mut rng = 0x5EED_u64;
        const ROUNDS: usize = 2000;
        const REPORT_EVERY: usize = 200;
        for round in 1..=ROUNDS {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let idx = (rng as usize) % live.len();
            let removed = live.swap_remove(idx);
            assert!(automaton.remove(&removed));

            let new_term = format!("http://example.org/churn-{round:08x}");
            automaton.insert(&new_term);
            live.push(new_term);

            if round % REPORT_EVERY == 0 {
                let slab = automaton.state_slab_len();
                let reachable = automaton.reachable_state_count();
                eprintln!(
                    "round {round}: slab={slab} reachable={reachable} \
                     ratio={:.2}x live_terms={}",
                    slab as f64 / reachable as f64,
                    live.len(),
                );
            }
        }

        let final_slab = automaton.state_slab_len();
        let final_reachable = automaton.reachable_state_count();
        eprintln!(
            "after {ROUNDS} churn rounds ({} total mutations): \
             slab={final_slab} reachable={final_reachable} ratio={:.2}x",
            ROUNDS * 2,
            final_slab as f64 / final_reachable as f64,
        );
    }

    fn build_both(mut terms: Vec<String>) -> (Automaton, Block, Vec<String>) {
        terms.sort_unstable();
        terms.dedup();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let block = Block::build(
            terms
                .iter()
                .enumerate()
                .map(|(i, t)| (t.as_str(), TermId(i as u32))),
        );
        (automaton, block, terms)
    }

    #[test]
    fn empty_input() {
        let (automaton, _block, _terms) = build_both(vec![]);
        assert_eq!(automaton.len(), 0);
        assert_eq!(automaton.rank("anything"), None);
        assert_eq!(automaton.reachable_state_count(), 1); // just the root
    }

    #[test]
    fn single_term() {
        let (automaton, block, terms) = build_both(vec!["http://example.org/a".to_string()]);
        assert_eq!(automaton.len(), 1);
        assert_eq!(automaton.rank(&terms[0]), Some(0));
        assert_eq!(
            automaton.rank(&terms[0]),
            block.rank(&terms[0]).map(|r| r as u32)
        );
        let mut buf = Vec::new();
        automaton.term_at_into(0, &mut buf);
        assert_eq!(std::str::from_utf8(&buf).unwrap(), terms[0]);
    }

    #[test]
    fn matches_block_rank_and_unrank_on_namespace_clustered_terms() {
        let terms: Vec<String> = (0..2000)
            .flat_map(|i| {
                [
                    format!("http://dbpedia.org/resource/Entity_{i}"),
                    format!("http://xmlns.com/foaf/0.1/property{i}"),
                    format!("urn:isbn:{i:013}"),
                    format!("_:b{i:x}"),
                ]
            })
            .collect();
        let (automaton, block, terms) = build_both(terms);

        assert_eq!(automaton.len(), terms.len());

        for (expected_rank, term) in terms.iter().enumerate() {
            let a_rank = automaton.rank(term);
            let b_rank = block.rank(term);
            assert_eq!(
                a_rank,
                Some(expected_rank as u32),
                "automaton rank mismatch for {term:?}"
            );
            assert_eq!(
                a_rank,
                b_rank.map(|r| r as u32),
                "automaton and Block disagree on rank for {term:?}"
            );

            let mut a_buf = Vec::new();
            automaton.term_at_into(expected_rank as u32, &mut a_buf);
            assert_eq!(std::str::from_utf8(&a_buf).unwrap(), term.as_str());
            assert_eq!(block.term_at(expected_rank), *term);
        }

        assert_eq!(automaton.rank("not-present-at-all"), None);
    }

    /// `Automaton::stream` (the incremental walk `AutomatonBlock::
    /// decode_into`/`merge_into` now use, in place of one `term_at_into`
    /// call per rank -- see their own module's doc comment for why) must
    /// visit exactly the same terms, in the same order, `term_at_into`
    /// would give one rank at a time. Direct test at this module's own
    /// level, not just relying on the indirect coverage `decode_into`'s
    /// and `merge_into`'s own tests already give it.
    #[test]
    fn stream_matches_term_at_into_for_every_rank() {
        let terms: Vec<String> = (0..500)
            .flat_map(|i| {
                [
                    format!("http://dbpedia.org/resource/Entity_{i}"),
                    format!("http://xmlns.com/foaf/0.1/property{i}"),
                    format!("urn:isbn:{i:013}"),
                    format!("_:b{i:x}"),
                ]
            })
            .collect();
        let (automaton, _block, terms) = build_both(terms);

        let mut stream = automaton.stream();
        let mut buf = Vec::new();
        for (rank, expected) in terms.iter().enumerate() {
            let streamed = stream
                .next()
                .unwrap_or_else(|| panic!("stream ended early at rank {rank}"));
            assert_eq!(std::str::from_utf8(streamed).unwrap(), expected.as_str());

            automaton.term_at_into(rank as u32, &mut buf);
            assert_eq!(
                &buf, streamed,
                "term_at_into disagrees with stream at rank {rank}"
            );
        }
        assert_eq!(stream.next(), None, "stream must end exactly at len()");
    }

    /// `seek_stream(rank)`'s own term must match `term_at_into(rank, ..)`
    /// exactly, and every subsequent `next()` call must match
    /// `term_at_into(rank + 1, ..)`, `term_at_into(rank + 2, ..)`, ...,
    /// for *every* starting rank, not just rank 0 (`stream()`'s only
    /// starting point) -- the direct correctness check for the resumable
    /// stack `seek_stream` hand-builds instead of walking there via
    /// `next()`.
    #[test]
    fn seek_stream_matches_term_at_into_from_every_starting_rank() {
        let terms: Vec<String> = (0..500)
            .flat_map(|i| {
                [
                    format!("http://dbpedia.org/resource/Entity_{i}"),
                    format!("http://xmlns.com/foaf/0.1/property{i}"),
                    format!("urn:isbn:{i:013}"),
                    format!("_:b{i:x}"),
                ]
            })
            .collect();
        let (automaton, _block, terms) = build_both(terms);
        let n = terms.len() as u32;

        for start in 0..n {
            let mut seeked = automaton.seek_stream(start);
            let mut expected_buf = Vec::new();
            automaton.term_at_into(start, &mut expected_buf);
            assert_eq!(
                seeked.current(),
                expected_buf.as_slice(),
                "seek_stream({start}).current() disagrees with term_at_into({start}, ..)"
            );

            for rank in (start + 1)..n {
                let next = seeked
                    .next()
                    .unwrap_or_else(|| panic!("seek_stream({start}) ended early at rank {rank}"));
                automaton.term_at_into(rank, &mut expected_buf);
                assert_eq!(
                    next,
                    expected_buf.as_slice(),
                    "seek_stream({start})'s next() disagrees with term_at_into({rank}, ..) after {} steps",
                    rank - start
                );
            }
            assert_eq!(
                seeked.next(),
                None,
                "seek_stream({start}) must end exactly at len()"
            );
        }
    }

    /// A shared prefix followed by divergent branches (the shape
    /// `Automaton::stream`'s stack-based backtracking has to get right)
    /// on an automaton small enough to reason about by hand.
    #[test]
    fn stream_backtracks_correctly_across_shared_and_diverging_prefixes() {
        let terms = ["a", "aa", "aab", "ab", "b"];
        let automaton = Automaton::build_sorted(terms.iter().copied());
        let mut stream = automaton.stream();
        let mut collected = Vec::new();
        while let Some(bytes) = stream.next() {
            collected.push(std::str::from_utf8(bytes).unwrap().to_string());
        }
        assert_eq!(collected, terms);
    }

    #[test]
    fn absent_terms_return_none() {
        let (automaton, block, terms) = build_both(vec![
            "http://example.org/a".to_string(),
            "http://example.org/ab".to_string(),
            "http://example.org/b".to_string(),
        ]);
        for absent in ["http://example.org/", "http://example.org/aa", "zzz", ""] {
            assert!(!terms.iter().any(|t| t == absent));
            assert_eq!(automaton.rank(absent), None);
            assert_eq!(block.rank(absent), None);
        }
    }

    /// The whole point of this exercise: the automaton must actually
    /// *minimize* (share equivalent states via the register), not just
    /// answer correctly with a bloated trie. True minimization goes
    /// further than sharing a literal common suffix: the states reached
    /// after "a", "b", "c", and "d" are themselves behaviorally identical
    /// (each accepts exactly "X" and nothing else), so a minimal DFA
    /// merges *those* too, not just the final "X"-accepting state -- a
    /// naive trie would have 9 states (root + 4 + 4); the minimal DAG has
    /// exactly 3: root, one shared "expects X next" state, one shared
    /// final state.
    #[test]
    fn shares_states_for_a_common_suffix() {
        let terms = [
            "aX".to_string(),
            "bX".to_string(),
            "cX".to_string(),
            "dX".to_string(),
        ];
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        assert_eq!(automaton.reachable_state_count(), 3);
        for (i, t) in terms.iter().enumerate() {
            assert_eq!(automaton.rank(t), Some(i as u32));
        }
    }

    /// Deterministic xorshift, matching this crate's other fuzz-style
    /// tests (e.g. `tests/prepared.rs`) -- no new dependency, reproducible.
    fn shuffled(mut terms: Vec<String>, mut seed: u64) -> Vec<String> {
        for i in (1..terms.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            terms.swap(i, (seed as usize) % (i + 1));
        }
        terms
    }

    fn namespace_clustered_terms(n: u32) -> Vec<String> {
        namespace_clustered_terms_range(0, n)
    }

    /// Same shape as `namespace_clustered_terms`, but starting from `start`
    /// instead of always from 0, covering `count` values of `i` -- for
    /// generating batches guaranteed disjoint from a baseline (or from
    /// each other), without needing to track which individual terms were
    /// already used.
    fn namespace_clustered_terms_range(start: u32, count: u32) -> Vec<String> {
        (start..start + count)
            .flat_map(|i| {
                [
                    format!("http://dbpedia.org/resource/Entity_{i}"),
                    format!("http://xmlns.com/foaf/0.1/property{i}"),
                    format!("urn:isbn:{i:013}"),
                    format!("_:b{i:x}"),
                ]
            })
            .collect()
    }

    /// The actual point of this whole exercise: build the same final term
    /// set two ways -- once via `build_sorted` (trusted, differentially
    /// tested against `Block` above), once by inserting the same terms
    /// one at a time, in an order that is *not* sorted -- and confirm
    /// both land on an automaton with the same reachable state count
    /// (true minimality survived incremental insertion, not just correct
    /// answers over a silently bloated structure) and agree on every
    /// rank/unrank pair. Several different shuffles of the same set, not
    /// just one: an earlier version of `insert` passed a single fixed
    /// seed here while silently corrupting transition order for *some*
    /// insertion sequences and not others (a state's transitions have to
    /// stay byte-sorted regardless of the order terms arrive in, and a
    /// blind append -- correct for `build_sorted`'s always-sorted input --
    /// broke that the first time a new branch needed to land before an
    /// existing one). Also re-inserts every term a second time in the
    /// same pass, checking idempotency doesn't regress minimality either.
    #[test]
    fn incremental_insert_matches_bulk_build_and_stays_minimal() {
        let mut terms = namespace_clustered_terms(500);
        terms.sort_unstable();
        terms.dedup();
        let bulk = Automaton::build_sorted(terms.iter().map(String::as_str));

        for seed in [0xC0FFEEu64, 1, 0xDEADBEEF, 999_999_937] {
            let insertion_order = shuffled(terms.clone(), seed);

            let mut incremental = Automaton::build_sorted(std::iter::empty());
            for t in &insertion_order {
                incremental.insert(t);
            }

            assert_eq!(incremental.len(), terms.len(), "seed {seed:#x}");
            assert_eq!(
                incremental.reachable_state_count(),
                bulk.reachable_state_count(),
                "seed {seed:#x}: incremental insertion left the automaton non-minimal"
            );

            for (expected_rank, term) in terms.iter().enumerate() {
                assert_eq!(
                    incremental.rank(term),
                    Some(expected_rank as u32),
                    "seed {seed:#x}: rank mismatch for {term:?}"
                );
                let mut buf = Vec::new();
                incremental.term_at_into(expected_rank as u32, &mut buf);
                assert_eq!(std::str::from_utf8(&buf).unwrap(), term.as_str());
            }
            assert_eq!(incremental.rank("not-present-at-all"), None);

            for t in &insertion_order {
                incremental.insert(t); // idempotent: must not disturb anything
            }
            assert_eq!(
                incremental.len(),
                terms.len(),
                "seed {seed:#x}: after re-insert"
            );
            assert_eq!(
                incremental.reachable_state_count(),
                bulk.reachable_state_count(),
                "seed {seed:#x}: re-insertion broke minimality"
            );
        }
    }

    #[test]
    fn insert_is_idempotent() {
        let mut automaton = Automaton::build_sorted(std::iter::empty());
        automaton.insert("http://example.org/a");
        automaton.insert("http://example.org/b");
        let before_states = automaton.reachable_state_count();
        let before_rank = automaton.rank("http://example.org/a");

        automaton.insert("http://example.org/a"); // already present

        assert_eq!(automaton.len(), 2);
        assert_eq!(automaton.reachable_state_count(), before_states);
        assert_eq!(automaton.rank("http://example.org/a"), before_rank);
    }

    /// A state that is itself a complete term *and* a prefix of another
    /// (in either insertion order) is exactly the case that exposed the
    /// output-formula bug this stage's predecessor found -- exercising it
    /// again here, through `insert` specifically (clone-on-write, not
    /// bulk construction), since the two code paths share the output
    /// logic but not the path-construction logic around it.
    #[test]
    fn insert_handles_prefix_relationships_either_direction() {
        let mut shorter_first = Automaton::build_sorted(std::iter::empty());
        shorter_first.insert("_:b1");
        shorter_first.insert("_:b10");
        assert_eq!(shorter_first.rank("_:b1"), Some(0));
        assert_eq!(shorter_first.rank("_:b10"), Some(1));

        let mut longer_first = Automaton::build_sorted(std::iter::empty());
        longer_first.insert("_:b10");
        longer_first.insert("_:b1");
        assert_eq!(longer_first.rank("_:b1"), Some(0));
        assert_eq!(longer_first.rank("_:b10"), Some(1));

        assert_eq!(
            shorter_first.reachable_state_count(),
            longer_first.reachable_state_count()
        );
    }

    /// Inserting into a block that already has other, unrelated content
    /// must not disturb the shared prefix states other terms depend on --
    /// the whole reason non-root states on the matched path get cloned
    /// instead of mutated in place.
    #[test]
    fn insert_does_not_disturb_unrelated_existing_terms() {
        let mut automaton =
            Automaton::build_sorted(["http://example.org/a", "http://example.org/b"].into_iter());
        automaton.insert("http://example.org/aa");

        let mut terms = [
            "http://example.org/a".to_string(),
            "http://example.org/aa".to_string(),
            "http://example.org/b".to_string(),
        ];
        terms.sort_unstable();
        for (rank, term) in terms.iter().enumerate() {
            assert_eq!(automaton.rank(term), Some(rank as u32));
        }

        let bulk = Automaton::build_sorted(terms.iter().map(String::as_str));
        assert_eq!(
            automaton.reachable_state_count(),
            bulk.reachable_state_count()
        );
    }

    const INSERT_PROBE_N: usize = 160; // block_max's midpoint, matching
    // block::tests::perf_probe_block_rebuild_breakdown's own choice.
    const INSERT_PROBE_ITERS: usize = 20_000;

    fn insert_probe_baseline_terms() -> Vec<(String, TermId)> {
        (0..INSERT_PROBE_N)
            .map(|i| (format!("http://example.org/term-{i:06}"), TermId(i as u32)))
            .collect()
    }

    fn insert_probe_new_terms() -> Vec<String> {
        (0..INSERT_PROBE_ITERS)
            .map(|i| format!("http://example.org/term-{i:06}-new"))
            .collect()
    }

    /// The measurement this whole exercise was for: does one insert into
    /// an already-populated structure actually cost O(term length),
    /// independent of how many other terms are already there, instead of
    /// `Block`'s O(block size) full decode-and-rebuild? Split into two
    /// separately-runnable probes (this one and the next), each doing
    /// only its own side's work, so `perf stat` can be run against each
    /// in isolation (`--exact` selects one test; a single process mixing
    /// both loops can't attribute hardware counters to either side) --
    /// this crate's own history this session found wall-clock-only
    /// comparisons misleading twice already for effects far smaller than
    /// this one, so the same discipline applies here, not skipped just
    /// because the wall-clock gap looks unambiguous. This probe's timed
    /// region is `Block`'s whole real per-insert cost (decode, merge,
    /// rebuild) -- nothing about a real `Dictionary::insert` call can
    /// skip any of that, unlike the `Automaton::insert` probe below,
    /// whose reset-to-baseline `clone()` is deliberately excluded from
    /// its own timed region for the matching reason.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_insert_cost_block() {
        let entries = insert_probe_baseline_terms();
        let block = Block::build(entries.iter().map(|(t, id)| (t.as_str(), *id)));
        let new_terms = insert_probe_new_terms();

        let mut arena = EntriesArena::new();
        let mut merged = EntriesArena::new();
        let start = Instant::now();
        for (i, new_term) in new_terms.iter().enumerate() {
            arena.clear();
            block.decode_into(&mut arena);
            merged.clear();
            merge_into(
                &arena,
                &[(new_term.as_str(), TermId((INSERT_PROBE_N + i) as u32))],
                &mut merged,
            );
            let rebuilt = Block::build(merged.iter());
            std::hint::black_box(&rebuilt);
        }
        let elapsed = start.elapsed();
        eprintln!(
            "Block (decode + merge + rebuild), N={INSERT_PROBE_N}, {INSERT_PROBE_ITERS} iters: \
             {elapsed:?} ({:?}/op)",
            elapsed / INSERT_PROBE_ITERS as u32,
        );
    }

    /// See `perf_probe_insert_cost_block`'s doc comment -- same baseline,
    /// same new terms, `Automaton::insert` in place of `Block`'s
    /// decode-merge-rebuild. The `clone()` resetting each iteration back
    /// to the shared N-term baseline is a benchmark-harness artifact (a
    /// repeatable starting condition), not something a real caller
    /// mutating one live automaton would ever pay, so it's outside the
    /// timed region -- excluding it is what makes this comparable to the
    /// other probe, not what inflates the result in `Automaton`'s favor.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_insert_cost_automaton() {
        let entries = insert_probe_baseline_terms();
        let automaton = Automaton::build_sorted(entries.iter().map(|(t, _)| t.as_str()));
        let new_terms = insert_probe_new_terms();

        let mut elapsed = std::time::Duration::ZERO;
        for new_term in &new_terms {
            let mut a = automaton.clone(); // reset to the baseline; not timed
            let start = Instant::now();
            a.insert(new_term);
            elapsed += start.elapsed();
            std::hint::black_box(&a);
        }
        eprintln!(
            "Automaton::insert, N={INSERT_PROBE_N}, {INSERT_PROBE_ITERS} iters: \
             {elapsed:?} ({:?}/op)",
            elapsed / INSERT_PROBE_ITERS as u32,
        );
    }

    /// Namespace-clustered, not sequential-digit-suffix terms: DESIGN.md's
    /// "Correction" section already found that a purely sequential
    /// numeric suffix manufactures artificially high branching factor
    /// that real IRIs don't have, and branching factor is exactly what
    /// decode cost (unlike insert cost) is sensitive to -- reusing that
    /// generator here instead of `insert_probe_baseline_terms`'s simpler
    /// pattern is deliberate, not an oversight.
    fn decode_probe_terms() -> Vec<String> {
        let mut terms = namespace_clustered_terms(500); // 2,000 terms
        terms.sort_unstable();
        terms.dedup();
        terms
    }

    const DECODE_PROBE_REPEATS: usize = 20;

    /// Decode (both directions) wasn't measured by the insert-cost probes
    /// above -- `Block::rank`/`term_at_into` walk `fst`'s compact,
    /// bit-packed transition encoding (the same encoding whose per-
    /// transition unpacking cost this project's own earlier profiling
    /// found dominant, see the "Correction" section), while
    /// `Automaton`'s transitions are a plain unrolled array of
    /// `(u8, u32, Option<StateId>)` -- no bit-unpacking at all. That could
    /// cut either way and needed its own measurement, not an assumption
    /// riding on the insert-cost result. Split into two probes (this one,
    /// `Block`; the next, `Automaton`) for the same `perf stat`
    /// attribution reason as the insert-cost probes.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_cost_block() {
        let terms = decode_probe_terms();
        let block = Block::build(
            terms
                .iter()
                .enumerate()
                .map(|(i, t)| (t.as_str(), TermId(i as u32))),
        );

        let start = Instant::now();
        for _ in 0..DECODE_PROBE_REPEATS {
            for t in &terms {
                std::hint::black_box(block.rank(t));
            }
        }
        let rank_elapsed = start.elapsed();

        let mut buf = Vec::new();
        let start = Instant::now();
        for _ in 0..DECODE_PROBE_REPEATS {
            for rank in 0..terms.len() {
                block.term_at_into(rank, &mut buf);
                std::hint::black_box(&buf);
            }
        }
        let unrank_elapsed = start.elapsed();

        let total_ops = (DECODE_PROBE_REPEATS * terms.len()) as u32;
        eprintln!(
            "Block decode, {} terms x {DECODE_PROBE_REPEATS} repeats ({total_ops} ops each):\n\
             rank:         {rank_elapsed:?} ({:?}/op)\n\
             term_at_into: {unrank_elapsed:?} ({:?}/op)",
            terms.len(),
            rank_elapsed / total_ops,
            unrank_elapsed / total_ops,
        );
    }

    /// See `perf_probe_decode_cost_block`'s doc comment.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_cost_automaton() {
        let terms = decode_probe_terms();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));

        let start = Instant::now();
        for _ in 0..DECODE_PROBE_REPEATS {
            for t in &terms {
                std::hint::black_box(automaton.rank(t));
            }
        }
        let rank_elapsed = start.elapsed();

        let mut buf = Vec::new();
        let start = Instant::now();
        for _ in 0..DECODE_PROBE_REPEATS {
            for rank in 0..terms.len() {
                automaton.term_at_into(rank as u32, &mut buf);
                std::hint::black_box(&buf);
            }
        }
        let unrank_elapsed = start.elapsed();

        let total_ops = (DECODE_PROBE_REPEATS * terms.len()) as u32;
        eprintln!(
            "Automaton decode, {} terms x {DECODE_PROBE_REPEATS} repeats ({total_ops} ops each):\n\
             rank:         {rank_elapsed:?} ({:?}/op)\n\
             term_at_into: {unrank_elapsed:?} ({:?}/op)",
            terms.len(),
            rank_elapsed / total_ops,
            unrank_elapsed / total_ops,
        );
    }

    const BATCH_PROBE_BASELINE_N: u32 = 500; // -> 2,000 namespace-clustered terms
    const BATCH_PROBE_BATCH_SIZE: usize = 500;
    const BATCH_PROBE_REPEATS: usize = 100;

    fn batch_probe_baseline() -> Automaton {
        let mut terms = namespace_clustered_terms(BATCH_PROBE_BASELINE_N);
        terms.sort_unstable();
        terms.dedup();
        Automaton::build_sorted(terms.iter().map(String::as_str))
    }

    /// One batch of new, sorted, namespace-clustered terms per repeat --
    /// disjoint from the baseline (a separate `i` range) and from every
    /// other repeat's batch, so `insert_sorted`'s "already present" path
    /// is never exercised here; this probe is about the cost of adding
    /// genuinely new content, matching what `Block`'s bounded-batch sweep
    /// (DESIGN.md) measured on the `fst`-backed side.
    fn batch_probe_batches() -> Vec<Vec<String>> {
        (0..BATCH_PROBE_REPEATS)
            .map(|r| {
                let start = BATCH_PROBE_BASELINE_N + (r as u32) * BATCH_PROBE_BATCH_SIZE as u32;
                let mut batch =
                    namespace_clustered_terms_range(start, BATCH_PROBE_BATCH_SIZE as u32 / 4);
                batch.sort_unstable();
                batch.dedup();
                batch
            })
            .collect()
    }

    /// Does batching within `Automaton` win the same way batching already
    /// won for `Block` (DESIGN.md's "Bounded-batch sweep": ~70x fewer
    /// rebuilds going from per-term to batched flushing)? Baseline:
    /// `BATCH_PROBE_REPEATS` sequential `insert` calls for the batch,
    /// each against a fresh clone of the same pre-populated automaton
    /// (the clone itself excluded from the timed region -- a
    /// benchmark-harness reset, not something a real caller pays, same
    /// rationale as the single-insert probes above).
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_batch_insert_cost_sequential() {
        let baseline = batch_probe_baseline();
        let batches = batch_probe_batches();

        let mut elapsed = std::time::Duration::ZERO;
        for batch in &batches {
            let mut a = baseline.clone(); // reset; not timed
            let start = Instant::now();
            for term in batch {
                a.insert(term);
            }
            elapsed += start.elapsed();
            std::hint::black_box(&a);
        }
        let total_terms: usize = batches.iter().map(Vec::len).sum();
        eprintln!(
            "Sequential insert, {} batches x ~{} terms ({total_terms} total): \
             {elapsed:?} ({:?}/term)",
            batches.len(),
            BATCH_PROBE_BATCH_SIZE,
            elapsed / total_terms as u32,
        );
    }

    /// See `perf_probe_batch_insert_cost_sequential`'s doc comment --
    /// same baseline, same batches, `insert_sorted` (one call per batch)
    /// in place of N sequential `insert` calls.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_batch_insert_cost_sorted() {
        let baseline = batch_probe_baseline();
        let batches = batch_probe_batches();

        let mut elapsed = std::time::Duration::ZERO;
        for batch in &batches {
            let mut a = baseline.clone(); // reset; not timed
            let start = Instant::now();
            a.insert_sorted(batch.iter().map(String::as_str));
            elapsed += start.elapsed();
            std::hint::black_box(&a);
        }
        let total_terms: usize = batches.iter().map(Vec::len).sum();
        eprintln!(
            "insert_sorted, {} batches x ~{} terms ({total_terms} total): \
             {elapsed:?} ({:?}/term)",
            batches.len(),
            BATCH_PROBE_BATCH_SIZE,
            elapsed / total_terms as u32,
        );
    }

    /// Chunks a large namespace-clustered corpus into ~`chunk_size`-term
    /// pieces, matching `AutomatonDictionary::bootstrap`'s own chunking
    /// of an initial batch into balanced, `<= block_max`-sized blocks --
    /// so this probe's per-call shape matches what `bulk_insert` actually
    /// does, not an arbitrary one-shot build.
    fn build_probe_chunks(total_terms: u32, chunk_size: usize) -> Vec<Vec<String>> {
        let mut terms = namespace_clustered_terms(total_terms / 4);
        terms.sort_unstable();
        terms.dedup();
        terms.chunks(chunk_size).map(<[String]>::to_vec).collect()
    }

    const BUILD_PROBE_TOTAL_TERMS: u32 = 80_000;
    const BUILD_PROBE_CHUNK_SIZE: usize = 256; // Dictionary's own default block_max

    /// Isolates `Automaton::build_sorted` alone -- no `BatchOrder::prepare`
    /// sort, no `AutomatonBlock`/`AutomatonDictionary` wrapping, no term
    /// generation inside the timed region -- unlike `bulk_insert`
    /// (`examples/perf_dictionary_automaton_vs_fst.rs`), whose own `perf
    /// record` profile turned out to be dominated by the *shared*
    /// `BatchOrder::prepare` sort (identical code on both the `fst` and
    /// automaton sides), not by construction itself. This is the clean
    /// signal for "building" specifically, chunked to match
    /// `bootstrap`'s real call shape (`BUILD_PROBE_CHUNK_SIZE`-term
    /// pieces, not one giant build). Run under `perf record`/`perf stat`
    /// with `--exact` to isolate this one test's samples.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_build_sorted_isolated() {
        let chunks = build_probe_chunks(BUILD_PROBE_TOTAL_TERMS, BUILD_PROBE_CHUNK_SIZE);
        let total_terms: usize = chunks.iter().map(Vec::len).sum();

        let start = Instant::now();
        for chunk in &chunks {
            let automaton = Automaton::build_sorted(chunk.iter().map(String::as_str));
            std::hint::black_box(&automaton);
        }
        let elapsed = start.elapsed();
        eprintln!(
            "Automaton::build_sorted, {} chunks x ~{} terms ({total_terms} total): \
             {elapsed:?} ({:?}/term)",
            chunks.len(),
            BUILD_PROBE_CHUNK_SIZE,
            elapsed / total_terms as u32,
        );
    }

    /// See `perf_probe_build_sorted_isolated`'s doc comment -- same
    /// chunks, `Block::build` in place of `Automaton::build_sorted`, for
    /// a like-for-like "where does *its* time go" comparison.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_block_build_isolated() {
        let chunks = build_probe_chunks(BUILD_PROBE_TOTAL_TERMS, BUILD_PROBE_CHUNK_SIZE);
        let total_terms: usize = chunks.iter().map(Vec::len).sum();

        let start = Instant::now();
        for chunk in &chunks {
            let entries: Vec<(&str, TermId)> = chunk
                .iter()
                .enumerate()
                .map(|(i, t)| (t.as_str(), TermId(i as u32)))
                .collect();
            let block = Block::build(entries.into_iter());
            std::hint::black_box(&block);
        }
        let elapsed = start.elapsed();
        eprintln!(
            "Block::build, {} chunks x ~{} terms ({total_terms} total): \
             {elapsed:?} ({:?}/term)",
            chunks.len(),
            BUILD_PROBE_CHUNK_SIZE,
            elapsed / total_terms as u32,
        );
    }
}
