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
use std::hash::{BuildHasherDefault, Hash};
use std::num::NonZeroU32;
use crate::packed_automaton::{FxHasher, common_prefix_len};
#[cfg(test)]
use crate::packed_automaton::PackedAutomaton;

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
///
/// A "single `Vec<TransitionChunk>`, no inline/overflow split" version
/// was tried here and reverted -- measured no real memory win (leaf
/// states got cheaper, but every 1+-transition state now needed a real
/// heap allocation for what used to be free, and those dominate) plus a
/// real speed loss (1.5-1.9x slower on `remove`/`get_id`/`get_term`/
/// `bulk_insert`). See DESIGN.md's memory-scaling investigation and the
/// design discussion that followed it -- a byte-packed, pooled-block
/// representation (not just "move the same fixed-size chunk to the
/// heap") is the direction that discussion is actually pursuing now;
/// this struct stays as shipped until/unless that's built and measured.
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

    /// Decodes a *batch* of `(output position, rank)` pairs -- sorted by
    /// rank ascending, not required to be deduplicated -- into
    /// `out[position]`, via one recursive descent instead of one
    /// `term_at_into` call per rank.
    ///
    /// This is the direct multi-key generalization of `term_at_into`'s
    /// own descent, not a heuristic layered on top of it: at each state,
    /// the remaining sorted targets are partitioned against that state's
    /// own transitions using the exact per-transition cumulative
    /// `output` values `last_output_at_most`/`term_at_into` already read
    /// -- no separate bookkeeping, no distance/gap comparison, no tuned
    /// constant. A state is visited only if at least one target's value
    /// falls in its range; a state with none does not get touched at
    /// all, at any cost. Two targets that happen to be structurally
    /// close (share a long common prefix in the automaton, which for
    /// sorted input usually but not always means "close in rank") pay
    /// for that shared path exactly once, discovered by the recursion
    /// itself as it walks both down the same branch -- not decided in
    /// advance by looking at how far apart their ranks are. Two targets
    /// that diverge immediately cost nothing extra for the parts of the
    /// tree neither of them touches.
    pub(crate) fn decode_ranks_into(&self, targets: &[(usize, u32)], out: &mut [Option<String>]) {
        let mut buf = Vec::new();
        self.decode_batch(self.root, targets, 0, &mut buf, out);
    }

    /// `targets` stays one contiguous slice of the *original* input for
    /// the whole recursion -- never copied or shifted -- with
    /// `value_offset` (the sum of every transition `output` consumed by
    /// an ancestor call) carried alongside instead, so a target's value
    /// *at this state* is always `stored_value - value_offset`. That
    /// keeps every recursive step to slice operations (`partition_point`,
    /// `split_at`) over the one backing array, no per-call allocation.
    fn decode_batch(
        &self,
        state_id: StateId,
        targets: &[(usize, u32)],
        value_offset: u32,
        buf: &mut Vec<u8>,
        out: &mut [Option<String>],
    ) {
        if targets.is_empty() {
            return;
        }
        let state = &self.states[state_id.slot() as usize];
        let mut remaining = targets;
        if state.is_final {
            // Every leading target still at value_offset (i.e. local
            // value 0) is this state's own term -- sorted ascending, so
            // these are always a prefix of `remaining`.
            let split = remaining.partition_point(|&(_, v)| v == value_offset);
            if split > 0 {
                let term =
                    std::str::from_utf8(buf).expect("keys were inserted as valid UTF-8 str bytes");
                for &(pos, _) in &remaining[..split] {
                    out[pos] = Some(term.to_string());
                }
                remaining = &remaining[split..];
            }
        }
        let mut idx = 0usize;
        while !remaining.is_empty() {
            let (byte, my_out, my_target) = state
                .transitions
                .at(idx)
                .expect("value must be reachable via some transition at this state");
            // Everything before the *next* transition's own output
            // belongs to this one -- `u32::MAX` if there is no next
            // transition, since the last one covers the rest by the
            // same invariant `last_output_at_most` relies on.
            let boundary = state
                .transitions
                .at(idx + 1)
                .map_or(u32::MAX, |(_, next_out, _)| value_offset + next_out);
            let split = remaining.partition_point(|&(_, v)| v < boundary);
            let (mine, rest) = remaining.split_at(split);
            if !mine.is_empty() {
                buf.push(byte);
                self.decode_batch(my_target, mine, value_offset + my_out, buf, out);
                buf.pop();
            }
            remaining = rest;
            idx += 1;
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

    /// The number of entries in `register` -- for measuring, not
    /// assuming, that `release`'s pruning (`register.remove(&sig)` on
    /// refcount-zero, `src/micro_fst.rs:2420`) actually keeps it bounded
    /// to the reachable state set, as opposed to `PackedAutomaton`'s own
    /// register, which has no equivalent pruning at all. Compare
    /// directly against `reachable_state_count`. Measured in
    /// `measure_automaton_register_tracks_reachable_under_churn`
    /// (`DESIGN.md`'s "measuring the register-pruning gap directly").
    #[cfg(test)]
    pub(crate) fn register_len(&self) -> usize {
        self.register.len()
    }

    /// The number of *reachable* states -- the true minimality signal.
    /// `states.len()` alone would overcount, since bulk construction and
    /// incremental insertion both leave discarded duplicates as
    /// unreachable garbage in the slab (see the struct doc comment); this
    /// walks the actual live graph. No longer test-only: used directly by
    /// `AutomatonBlock`/`AutomatonDictionary`'s own `reachable_state_count`
    /// for the states-per-term investigation (DESIGN.md).
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

    /// A pure measurement, not a real encoding: what would this
    /// automaton's *reachable* structure cost if every state were a
    /// byte-packed, variable-length record instead of a fixed-size
    /// `State`/`TransitionChunk`? Builds no new storage, changes no
    /// addressing, touches no mutation code -- the cheap way to check
    /// the byte-packing hypothesis (DESIGN.md's memory-scaling
    /// investigation, and the design discussion that followed it) before
    /// committing to rewriting `confirm_path`/`redirect`/`release`/
    /// `clone_state`/`alloc_state` around a real version of this layout.
    ///
    /// Per-state record: 1 byte (a count/flag byte -- `is_final` in the
    /// top bit, transition count in the rest, safe up to 127
    /// transitions, comfortably above this corpus's branching factor),
    /// then a LEB128 varint for this state's own subtree `count`
    /// (needed so a mutable packed representation can recompute a
    /// modified state's children's outputs without re-walking whole
    /// subtrees -- see `PackedAutomaton`'s record layout, which this
    /// mirrors exactly), then per transition: 1 byte (the edge byte) +
    /// a LEB128 varint for the output (bounded by block size, so almost
    /// always 1 byte at this corpus's default `block_max`) + a fixed
    /// `TARGET_ADDR_BYTES` address. The address width is assumed fixed
    /// here for simplicity -- a real implementation might vary-width it
    /// too (smaller within one block), which would make this a
    /// pessimistic estimate, not an optimistic one.
    pub(crate) fn estimate_packed_bytes(&self) -> usize {
        const TARGET_ADDR_BYTES: usize = 4;
        fn leb128_len(mut value: u32) -> usize {
            let mut len = 1;
            while value >= 0x80 {
                value >>= 7;
                len += 1;
            }
            len
        }

        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![self.root];
        let mut total = 0usize;
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                let state = &self.states[id.slot() as usize];
                total += 1; // flag/count byte
                total += leb128_len(state.count); // subtree count varint
                let mut i = 0usize;
                while let Some((_, output, target)) = state.transitions.at(i) {
                    total += 1; // edge byte
                    total += leb128_len(output);
                    total += TARGET_ADDR_BYTES;
                    stack.push(target);
                    i += 1;
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
    /// The packed equivalent of `automaton` (same term set).
    fn compile(automaton: &Automaton) -> PackedAutomaton {
        let mut terms: Vec<String> = Vec::new();
        let mut stream = automaton.stream();
        while let Some(t) = stream.next() {
            terms.push(String::from_utf8(t.to_vec()).expect("UTF-8 terms"));
        }
        PackedAutomaton::build_sorted(terms.iter().map(String::as_str))
    }

    use super::*;
    use crate::TermId;
    use crate::block::Block;
    use crate::entries_arena::{EntriesArena, merge_into};
    use std::time::Instant;

    /// Diagnostic for the memory-scaling investigation (DESIGN.md): what
    /// is the fixed per-state cost this representation pays regardless
    /// of actual branching factor, and how does it compare to a rough
    /// estimate of what `fst`'s byte-packed encoding needs per state.
    /// `#[ignore]`d: prints sizes, not a pass/fail assertion.
    #[test]
    #[ignore]
    fn print_struct_sizes() {
        eprintln!(
            "size_of::<StateId>()         = {}",
            std::mem::size_of::<StateId>()
        );
        eprintln!(
            "size_of::<TransitionChunk>() = {}",
            std::mem::size_of::<TransitionChunk>()
        );
        eprintln!(
            "size_of::<Transitions>()     = {}",
            std::mem::size_of::<Transitions>()
        );
        eprintln!(
            "size_of::<State>()           = {}",
            std::mem::size_of::<State>()
        );
        eprintln!("CHUNK_CAP                    = {CHUNK_CAP}");
    }

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

    /// `PackedAutomaton`'s counterpart to `measure_slot_reclamation_gap_
    /// under_sustained_churn`: it has no removal (see the struct doc
    /// comment) and no reclamation at all, ever, by design -- dead
    /// records from a superseded path just sit in `blocks`, unreachable,
    /// until whatever holds this structure decides to rebuild fresh
    /// (`AutomatonBlock`'s own periodic split/merge, not attempted by
    /// this prototype). This measures, not assumes, how fast that dead
    /// weight actually accumulates under sustained insert-only churn --
    /// `size_bytes()` (raw bytes written, dead included) against
    /// `estimate_packed_bytes()` on a *fresh* build of the same live set
    /// (what a from-scratch rebuild would cost), at increasing round
    /// counts.
    #[test]
    #[ignore = "manual measurement, not part of the regular suite"]
    fn measure_packed_automaton_dead_bytes_under_sustained_inserts() {
        let mut live = namespace_clustered_terms(200); // 800 terms
        live.sort_unstable();
        live.dedup();
        let automaton = Automaton::build_sorted(live.iter().map(String::as_str));
        let mut packed = compile(&automaton);

        eprintln!(
            "round 0: size_bytes={} fresh_estimate={}",
            packed.size_bytes(),
            automaton.estimate_packed_bytes(),
        );

        const ROUNDS: usize = 2000;
        const REPORT_EVERY: usize = 200;
        for round in 1..=ROUNDS {
            let new_term = format!("http://example.org/churn-{round:08x}");
            packed.insert(&new_term);
            live.push(new_term);

            if round % REPORT_EVERY == 0 {
                let mut sorted_live = live.clone();
                sorted_live.sort_unstable();
                let fresh = Automaton::build_sorted(sorted_live.iter().map(String::as_str));
                eprintln!(
                    "round {round}: size_bytes={} fresh_estimate={} \
                     ratio={:.2}x live_terms={}",
                    packed.size_bytes(),
                    fresh.estimate_packed_bytes(),
                    packed.size_bytes() as f64 / fresh.estimate_packed_bytes() as f64,
                    live.len(),
                );
            }
        }
    }

    /// Empirically grounds the "dedup structure" comparison against
    /// `fst` and `PackedAutomaton` (`AUTOMATON_REFERENCE.md` §3.3,
    /// `FST_VS_PACKED_AUTOMATON.md`'s "Dedup structure" section): both
    /// claim `release()` (`src/micro_fst.rs:2382`-ish) keeps `register`
    /// trimmed to exactly the reachable state set, but neither document
    /// ever measured `register.len()` directly -- only argued it from
    /// reading `release()`'s body. Same sustained one-out-one-in churn
    /// as `measure_slot_reclamation_gap_under_sustained_churn` (which
    /// measures the *slab*, not the register); this measures
    /// `register_len()` against `reachable_state_count()` instead. If
    /// the pruning claim holds, the two should stay equal (or very
    /// close -- a handful of states can be mid-flight unconfirmed on the
    /// rightmost path at any instant) every round, never drifting apart
    /// as cumulative mutation count grows.
    #[test]
    #[ignore = "manual measurement, not part of the regular suite"]
    fn measure_automaton_register_tracks_reachable_under_churn() {
        let mut live = namespace_clustered_terms(200); // 800 terms
        live.sort_unstable();
        live.dedup();
        let mut automaton = Automaton::build_sorted(live.iter().map(String::as_str));

        eprintln!(
            "round 0: register={} reachable={}",
            automaton.register_len(),
            automaton.reachable_state_count(),
        );

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
                let register = automaton.register_len();
                let reachable = automaton.reachable_state_count();
                eprintln!(
                    "round {round}: register={register} reachable={reachable} \
                     ratio={:.3}x live_terms={}",
                    register as f64 / reachable as f64,
                    live.len(),
                );
            }
        }

        let final_register = automaton.register_len();
        let final_reachable = automaton.reachable_state_count();
        eprintln!(
            "after {ROUNDS} churn rounds ({} total mutations): \
             register={final_register} reachable={final_reachable} ratio={:.3}x",
            ROUNDS * 2,
            final_register as f64 / final_reachable as f64,
        );
    }

    /// `PackedAutomaton`'s counterpart: `write_confirmed_state` only ever
    /// inserts into `PackedRegister`, never removes (the struct's own
    /// doc comment and `FST_VS_PACKED_AUTOMATON.md`'s "Dedup structure"
    /// section argue this from reading the code; this measures it).
    /// Same sustained insert-only churn as `measure_packed_automaton_
    /// dead_bytes_under_sustained_inserts` (which measures `size_bytes()`,
    /// not the register). If the no-pruning claim holds, `register_len()`
    /// should climb roughly linearly with cumulative insert count while
    /// `reachable_state_count()` stays flat (the live term count is held
    /// constant by `PackedAutomaton` having no `remove` at all here --
    /// every insert is a net-new term, so reachable count grows too, but
    /// far slower than register_len if old, superseded confirmed states
    /// are piling up).
    #[test]
    #[ignore = "manual measurement, not part of the regular suite"]
    fn measure_packed_automaton_register_growth_under_sustained_inserts() {
        let mut live = namespace_clustered_terms(200); // 800 terms
        live.sort_unstable();
        live.dedup();
        let automaton = Automaton::build_sorted(live.iter().map(String::as_str));
        let mut packed = compile(&automaton);

        eprintln!(
            "round 0: register={} reachable={}",
            packed.register_len(),
            packed.reachable_state_count(),
        );

        const ROUNDS: usize = 2000;
        const REPORT_EVERY: usize = 200;
        for round in 1..=ROUNDS {
            let new_term = format!("http://example.org/churn-{round:08x}");
            packed.insert(&new_term);
            live.push(new_term);

            if round % REPORT_EVERY == 0 {
                let register = packed.register_len();
                let reachable = packed.reachable_state_count();
                eprintln!(
                    "round {round}: register={register} reachable={reachable} \
                     ratio={:.3}x live_terms={}",
                    register as f64 / reachable as f64,
                    live.len(),
                );
            }
        }

        let final_register = packed.register_len();
        let final_reachable = packed.reachable_state_count();
        eprintln!(
            "after {ROUNDS} inserts: register={final_register} reachable={final_reachable} \
             ratio={:.3}x",
            final_register as f64 / final_reachable as f64,
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

    /// `PackedAutomaton::rank`/`term_at_into` must agree with
    /// `Automaton`'s own, for every term, across a namespace-clustered
    /// corpus (the same one the equivalent `Automaton`-vs-`Block` test
    /// above uses) -- the direct correctness check for the whole
    /// postorder-compile-then-parse scheme before trusting any decode
    /// speed measured against it.
    #[test]
    fn packed_automaton_matches_automaton_rank_and_unrank() {
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
        let (automaton, _block, terms) = build_both(terms);
        let packed = compile(&automaton);

        for (expected_rank, term) in terms.iter().enumerate() {
            let a_rank = automaton.rank(term);
            let p_rank = packed.rank(term);
            assert_eq!(
                p_rank,
                Some(expected_rank as u32),
                "packed rank mismatch for {term:?}"
            );
            assert_eq!(
                a_rank, p_rank,
                "automaton and packed disagree on rank for {term:?}"
            );

            let mut a_buf = Vec::new();
            automaton.term_at_into(expected_rank as u32, &mut a_buf);
            let mut p_buf = Vec::new();
            packed.term_at_into(expected_rank as u32, &mut p_buf);
            assert_eq!(
                a_buf, p_buf,
                "automaton and packed disagree on term_at_into({expected_rank})"
            );
            assert_eq!(std::str::from_utf8(&p_buf).unwrap(), term.as_str());
        }

        assert_eq!(packed.rank("not-present-at-all"), None);
        assert_eq!(packed.rank(""), None);
    }

    /// Sorted construction's `build_sorted` and `insert`'s general
    /// walk-match-extend algorithm are two different code paths that must
    /// agree on every answer. They are *not* required to reach the same
    /// `reachable_state_count` any more: `PackedRegistry` (see its own
    /// doc comment) is a bounded LRU cache, so its eviction decisions --
    /// and so its final structure -- depend on access history, which
    /// `build_sorted`'s direct confirmation order and `insert`'s
    /// one-at-a-time order genuinely differ on, even over the identical
    /// final term set. This is the same construction-order sensitivity
    /// `fst` itself has (`AUTOMATON_REFERENCE.md`); asserting exact
    /// structural equality here would be asserting the one property
    /// this design deliberately no longer guarantees.
    #[test]
    fn packed_automaton_build_sorted_matches_incremental_insert() {
        let mut terms = namespace_clustered_terms(80); // 320 terms
        terms.sort_unstable();
        terms.dedup();

        let built = PackedAutomaton::build_sorted(terms.iter().map(String::as_str));

        let mut inserted = PackedAutomaton::build_sorted(std::iter::empty());
        for term in &terms {
            inserted.insert(term);
        }

        for (expected_rank, term) in terms.iter().enumerate() {
            assert_eq!(built.rank(term), Some(expected_rank as u32));
            assert_eq!(built.rank(term), inserted.rank(term));
        }
    }

    /// `PackedAutomaton::insert_sorted`, starting from an empty automaton,
    /// must agree with `build_sorted` on every answer over the same term
    /// set. Not required to match its `reachable_state_count` --
    /// `packed_automaton_build_sorted_matches_incremental_insert`'s own
    /// doc comment explains why, and applies identically here.
    #[test]
    fn packed_automaton_insert_sorted_matches_build_sorted() {
        let mut terms = namespace_clustered_terms(80); // 320 terms
        terms.sort_unstable();
        terms.dedup();

        let built = PackedAutomaton::build_sorted(terms.iter().map(String::as_str));

        let mut inserted = PackedAutomaton::build_sorted(std::iter::empty());
        inserted.insert_sorted(terms.iter().map(String::as_str));

        for (expected_rank, term) in terms.iter().enumerate() {
            assert_eq!(inserted.rank(term), Some(expected_rank as u32));
            assert_eq!(built.rank(term), inserted.rank(term));
        }
    }

    /// `PackedAutomaton::insert_sorted` inserting a *mixed* batch (some
    /// terms already present, some new) into an already-populated
    /// automaton must match `Automaton::insert_sorted`'s own answer for
    /// every term, and reach the same minimal reachable structure a
    /// fresh build over the final term set would.
    #[test]
    fn packed_automaton_insert_sorted_matches_automaton_insert_sorted() {
        let mut baseline = namespace_clustered_terms(80); // 320 terms
        baseline.sort_unstable();
        baseline.dedup();

        let mut automaton = Automaton::build_sorted(baseline.iter().map(String::as_str));
        let mut packed = PackedAutomaton::build_sorted(baseline.iter().map(String::as_str));

        // A mixed batch: every 3rd existing term (re-insertion, a no-op)
        // plus a run of brand-new terms, sorted together.
        let mut incoming: Vec<String> = baseline.iter().step_by(3).cloned().collect();
        for i in 0..50 {
            incoming.push(format!("http://example.org/brand-new-{i:03}"));
        }
        incoming.sort_unstable();
        incoming.dedup();

        automaton.insert_sorted(incoming.iter().map(String::as_str));
        packed.insert_sorted(incoming.iter().map(String::as_str));

        let mut all_terms = baseline.clone();
        all_terms.extend(incoming.iter().cloned());
        all_terms.sort_unstable();
        all_terms.dedup();

        assert_eq!(
            packed.reachable_state_count(),
            automaton.reachable_state_count()
        );
        let fresh = Automaton::build_sorted(all_terms.iter().map(String::as_str));
        assert_eq!(
            packed.reachable_state_count(),
            fresh.reachable_state_count()
        );

        for (expected_rank, term) in all_terms.iter().enumerate() {
            assert_eq!(
                packed.rank(term),
                Some(expected_rank as u32),
                "packed rank mismatch for {term:?}"
            );
            assert_eq!(
                automaton.rank(term),
                packed.rank(term),
                "automaton and packed disagree on rank for {term:?}"
            );
        }
    }

    /// A small, hand-traceable case specifically exercising states with
    /// more than one transition and a state that's both final and has
    /// children (`"car"`, a prefix of `"cars"`) -- the shapes most
    /// likely to expose a postorder-ordering or flag-byte bug that a
    /// purely-leaf-heavy corpus might not.
    #[test]
    fn packed_automaton_matches_automaton_on_branching_and_prefix_terms() {
        let terms = ["car", "cars", "cat"];
        let automaton = Automaton::build_sorted(terms.iter().copied());
        let packed = compile(&automaton);

        for (rank, term) in terms.iter().enumerate() {
            assert_eq!(packed.rank(term), Some(rank as u32));
            let mut buf = Vec::new();
            packed.term_at_into(rank as u32, &mut buf);
            assert_eq!(std::str::from_utf8(&buf).unwrap(), *term);
        }
        assert_eq!(packed.rank("ca"), None);
        assert_eq!(packed.rank("carsx"), None);
    }

    /// `compile(..).size_bytes()` was once exactly
    /// `Automaton::estimate_packed_bytes` (same layout). The record format
    /// has since shrunk (DESIGN.md's "fst-style record encoding"), so the
    /// estimate is now an upper bound.
    /// Exercises the record format's edge cases: a root with more than
    /// `PACKED_COUNT_ESCAPE` transitions (header escape byte), long unique
    /// tails (NEXT chains), and targets far enough back to need
    /// multi-byte deltas -- via `build_sorted`, `insert_sorted` and
    /// `insert`, checked against the sorted term list.
    #[test]
    fn packed_automaton_wide_fanout_and_long_tails_round_trip() {
        let mut terms: Vec<String> = (1u8..=250)
            .map(|b| {
                let c = char::from_u32(u32::from(b)).unwrap_or('x');
                format!("{c}{}", "tail".repeat(usize::from(b % 40)))
            })
            .collect();
        terms.extend((0..300).map(|i| format!("p/{i:05}/{}", "z".repeat(i % 97))));
        terms.sort_unstable();
        terms.dedup();
        let (first, second): (Vec<_>, Vec<_>) =
            terms.iter().enumerate().partition(|(i, _)| i % 3 != 0);
        let first: Vec<&str> = first.into_iter().map(|(_, t)| t.as_str()).collect();
        let second: Vec<&str> = second.into_iter().map(|(_, t)| t.as_str()).collect();

        let built = PackedAutomaton::build_sorted(terms.iter().map(String::as_str));
        let mut batched = PackedAutomaton::build_sorted(first.iter().copied());
        batched.release_construction_state();
        batched.insert_sorted(second.iter().copied());
        let mut single = PackedAutomaton::build_sorted(std::iter::empty());
        for t in terms.iter().rev() {
            single.insert(t);
        }

        let mut chained = built.clone();
        chained.compact_and_chain();
        let mut coded = built.clone();
        let sample: Vec<&[u8]> = terms.iter().map(|t| t.as_bytes()).collect();
        coded.set_symbols(Some(std::sync::Arc::new(crate::fsst::SymbolTable::train(&sample))));
        coded.compact_and_chain();
        assert!(coded.chains_coded());
        for packed in [&built, &batched, &single, &chained, &coded] {
            let mut buf = Vec::new();
            for (rank, term) in terms.iter().enumerate() {
                assert_eq!(packed.rank(term), Some(rank as u32));
                packed.term_at_into(rank as u32, &mut buf);
                assert_eq!(buf, term.as_bytes());
            }
            let mut stream = packed.stream();
            for term in &terms {
                assert_eq!(stream.next(), Some(term.as_bytes()));
            }
            assert_eq!(stream.next(), None);
            let targets: Vec<(usize, u32)> = (0..terms.len()).map(|r| (r, r as u32)).collect();
            let mut out = vec![None; terms.len()];
            packed.decode_ranks_into(&targets, &mut out);
            for (term, decoded) in terms.iter().zip(&out) {
                assert_eq!(decoded.as_deref(), Some(term.as_str()));
            }
            assert_eq!(packed.rank("p/0000"), None);
        }
    }

    /// `compact` after sustained incremental merges: answers unchanged,
    /// dead records gone, and the reachable state count equals the exact
    /// minimal automaton's (bottom-up exact dedup is minimization).
    #[test]
    fn packed_automaton_compact_is_exactly_minimal_and_preserves_answers() {
        let mut terms = namespace_clustered_terms(1200);
        terms.extend((0..600).map(|i| format!("_:b{i}x{}", i * 7919 % 1000)));
        terms.sort_unstable();
        terms.dedup();
        let mut packed = PackedAutomaton::build_sorted(std::iter::empty());
        for round in 0..12 {
            let batch: Vec<&str> = terms
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 12 == round)
                .map(|(_, t)| t.as_str())
                .collect();
            packed.insert_sorted(batch.into_iter());
            packed.release_construction_state();
        }
        let before = packed.size_bytes();
        packed.compact();
        assert!(packed.size_bytes() < before);
        let exact = Automaton::build_sorted(terms.iter().map(String::as_str));
        assert_eq!(packed.reachable_state_count(), exact.reachable_state_count());
        let fresh = PackedAutomaton::build_sorted(terms.iter().map(String::as_str));
        assert!(packed.size_bytes() <= fresh.size_bytes());
        // Path compression keeps the same states (chains hold one per
        // label byte) and every answer below.
        packed.compact_and_chain();
        assert_eq!(packed.reachable_state_count(), exact.reachable_state_count());

        let mut buf = Vec::new();
        for (rank, term) in terms.iter().enumerate() {
            assert_eq!(packed.rank(term), Some(rank as u32));
            packed.term_at_into(rank as u32, &mut buf);
            assert_eq!(buf, term.as_bytes());
        }
        // Still mutable afterwards.
        packed.insert("zzz-after-compact");
        assert_eq!(packed.rank("zzz-after-compact"), Some(terms.len() as u32));
        packed.compact();
        assert_eq!(packed.rank("zzz-after-compact"), Some(terms.len() as u32));
    }

    /// `remove`/`insert`/`lower_bound` against a sorted-set model, under
    /// a deterministic pseudo-random mix of inserts and removes (shared
    /// prefixes, prefix-of-another-term cases, removing absent terms).
    #[test]
    fn packed_automaton_remove_and_lower_bound_match_model() {
        let pool: Vec<String> = (0..400)
            .map(|i| match i % 4 {
                0 => format!("http://ex.org/a/{}", i / 4),
                1 => format!("http://ex.org/a/{}x", i / 8),
                2 => format!("_:b{}", i % 37),
                _ => format!("\"{}\"@en", i * 31 % 97),
            })
            .collect();
        let mut model = std::collections::BTreeSet::<String>::new();
        let mut packed = PackedAutomaton::build_sorted(std::iter::empty());
        // Chaining compactions below code labels with this table.
        let sample: Vec<&[u8]> = pool.iter().map(|t| t.as_bytes()).collect();
        packed.set_symbols(Some(std::sync::Arc::new(crate::fsst::SymbolTable::train(&sample))));
        let mut state = 0x9E37_79B9_u32;
        for step in 0..3000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let term = &pool[state as usize % pool.len()];
            if state % 3 == 0 {
                assert_eq!(packed.remove(term), model.remove(term), "step {step} remove {term}");
            } else {
                packed.insert(term);
                model.insert(term.clone());
            }
            packed.release_construction_state();
            if step % 97 == 0 {
                // Alternate plain and chaining compaction; mutations after
                // a chaining one expand the chains first.
                if step % 194 == 0 {
                    packed.compact_and_chain();
                } else {
                    packed.compact();
                }
            }
            if step % 50 == 0 || step == 2999 {
                let sorted: Vec<&String> = model.iter().collect();
                let mut buf = Vec::new();
                for (rank, t) in sorted.iter().enumerate() {
                    assert_eq!(packed.rank(t), Some(rank as u32));
                    assert_eq!(packed.lower_bound(t), (rank as u32, true));
                    packed.term_at_into(rank as u32, &mut buf);
                    assert_eq!(buf, t.as_bytes());
                }
                for probe in pool.iter().map(|t| format!("{t}~")).chain(["".to_string(), "zzz".to_string(), "h".to_string()]) {
                    let below = sorted.iter().filter(|t| t.as_str() < probe.as_str()).count() as u32;
                    assert_eq!(packed.lower_bound(&probe), (below, model.contains(&probe)), "probe {probe:?}");
                }
                let streamed: Vec<Vec<u8>> = {
                    let mut st = packed.stream();
                    let mut v = Vec::new();
                    while let Some(t) = st.next() {
                        v.push(t.to_vec());
                    }
                    v
                };
                assert_eq!(streamed, sorted.iter().map(|t| t.as_bytes().to_vec()).collect::<Vec<_>>());
            }
        }
        packed.compact();
        let exact = Automaton::build_sorted(model.iter().map(String::as_str));
        assert_eq!(packed.reachable_state_count(), exact.reachable_state_count());
    }

    #[test]
    fn packed_automaton_size_matches_the_earlier_estimate() {
        let mut terms = namespace_clustered_terms(300);
        terms.sort_unstable();
        terms.dedup();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let packed = compile(&automaton);
        // The estimate models the original fixed-width layout; the current
        // one (implicit outputs/counts, delta/NEXT targets) is never larger.
        assert!(packed.size_bytes() <= automaton.estimate_packed_bytes());
    }

    /// `PackedAutomaton::insert`, built up one term at a time in
    /// non-sorted (interleaved-prefix) order, must reach the exact same
    /// `rank`/`term_at_into` answers as an `Automaton` built the same
    /// way -- the direct correctness check for the mutable version's
    /// walk-match/open-staging-frames/confirm-bottom-up algorithm,
    /// mirrored from `Automaton::insert` per the struct doc comment.
    #[test]
    fn packed_automaton_insert_matches_automaton_incremental_insert() {
        let terms = [
            "car", "cart", "cat", "cats", "dog", "do", "dogs", "carts", "ca",
        ];

        let mut automaton = Automaton::build_sorted(std::iter::empty());
        let mut packed = compile(&automaton);
        for term in terms {
            automaton.insert(term);
            packed.insert(term);
        }

        let mut sorted = terms.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        for (expected_rank, term) in sorted.iter().enumerate() {
            assert_eq!(
                packed.rank(term),
                Some(expected_rank as u32),
                "packed rank mismatch for {term:?}"
            );
            assert_eq!(
                automaton.rank(term),
                packed.rank(term),
                "automaton and packed disagree on rank for {term:?}"
            );

            let mut buf = Vec::new();
            packed.term_at_into(expected_rank as u32, &mut buf);
            assert_eq!(std::str::from_utf8(&buf).unwrap(), *term);
        }
        assert_eq!(packed.rank("not-present"), None);
    }

    /// Re-inserting an already-present term must be a true no-op: same
    /// reachable structure, same size, not just "doesn't crash" --
    /// `Automaton::insert`'s own documented early-return behavior,
    /// mirrored here.
    #[test]
    fn packed_automaton_insert_is_idempotent() {
        let mut terms = namespace_clustered_terms(50); // 200 distinct terms
        terms.sort_unstable();
        terms.dedup();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let mut packed = compile(&automaton);

        let before_size = packed.size_bytes();
        let before_reachable = packed.reachable_state_count();
        for term in &terms {
            packed.insert(term);
        }
        assert_eq!(packed.size_bytes(), before_size);
        assert_eq!(packed.reachable_state_count(), before_reachable);
    }

    /// The original design plan's minimality regression test (step 4:
    /// "after a scripted sequence of incremental inserts, the resulting
    /// automaton's state count must equal a fresh `build_sorted` over
    /// the same final term set") no longer holds exactly, now that
    /// `PackedRegistry` is a bounded LRU cache rather than an exact,
    /// unbounded one (its own doc comment has the full reasoning, and
    /// why this is the right tradeoff rather than a regression). What
    /// must still hold: every answer stays correct, and the lossy cache
    /// can only ever find *fewer or equal* merges than an exact one --
    /// its equivalence classes are a subset of an exact register's --
    /// so `reachable_state_count` can only be `>=` a fresh minimal
    /// build's, never less.
    #[test]
    fn packed_automaton_insert_stays_correct_though_no_longer_exactly_minimal() {
        let mut pool = namespace_clustered_terms(150); // 600 distinct terms
        pool.sort_unstable();
        pool.dedup();
        let mut shuffled = pool.clone();
        // A cheap, deterministic reordering (not a real shuffle) so
        // insertion order isn't just the same sorted order `build_sorted`
        // itself would see -- exercises genuine mid-structure splicing,
        // not only append-at-the-end growth.
        shuffled.sort_by_key(|t| {
            let mut hash = 0u64;
            for &b in t.as_bytes() {
                hash = hash.wrapping_mul(31).wrapping_add(u64::from(b));
            }
            hash
        });

        let mut packed = compile(&Automaton::build_sorted(std::iter::empty()));
        for term in &shuffled {
            packed.insert(term);
        }

        let fresh = Automaton::build_sorted(pool.iter().map(String::as_str));
        assert!(packed.reachable_state_count() >= fresh.reachable_state_count());

        for (expected_rank, term) in pool.iter().enumerate() {
            assert_eq!(packed.rank(term), Some(expected_rank as u32));
        }
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

    /// `PackedTermsStream`'s counterpart to `stream_matches_term_at_into_
    /// for_every_rank`: must visit exactly the same terms, in the same
    /// order, `PackedAutomaton::term_at_into` would give one rank at a
    /// time.
    #[test]
    fn packed_stream_matches_term_at_into_for_every_rank() {
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
        let packed = compile(&automaton);

        let mut stream = packed.stream();
        let mut buf = Vec::new();
        for (rank, expected) in terms.iter().enumerate() {
            let streamed = stream
                .next()
                .unwrap_or_else(|| panic!("stream ended early at rank {rank}"));
            assert_eq!(std::str::from_utf8(streamed).unwrap(), expected.as_str());

            packed.term_at_into(rank as u32, &mut buf);
            assert_eq!(
                &buf, streamed,
                "term_at_into disagrees with stream at rank {rank}"
            );
        }
        assert_eq!(stream.next(), None, "stream must end exactly at len()");
    }

    /// Same shared-prefix/diverging-branch shape as `stream_backtracks_
    /// correctly_across_shared_and_diverging_prefixes`, through
    /// `PackedTermsStream` instead -- specifically exercises a final
    /// state with children (`"a"`, a prefix of `"aa"`/`"aab"`) and a
    /// multi-level backtrack (`"aab"` -> `"ab"` -> `"b"`), the shapes
    /// most likely to expose an off-by-one in cursor/remaining
    /// bookkeeping that a purely-leaf-heavy corpus wouldn't.
    #[test]
    fn packed_stream_backtracks_correctly_across_shared_and_diverging_prefixes() {
        let terms = ["a", "aa", "aab", "ab", "b"];
        let automaton = Automaton::build_sorted(terms.iter().copied());
        let packed = compile(&automaton);
        let mut stream = packed.stream();
        let mut collected = Vec::new();
        while let Some(term) = stream.next() {
            collected.push(std::str::from_utf8(term).unwrap().to_string());
        }
        assert_eq!(collected, terms);
    }

    /// `decode_ranks_into` given *every* rank in one batch, sorted
    /// ascending by rank as required (its own precondition) but with
    /// output *positions* deliberately decorrelated from rank order --
    /// must match `term_at_into` called individually for each one. Since
    /// this queries the whole automaton, it exercises every state's own
    /// partition-by-transition logic at least once, not just a few
    /// hand-picked paths.
    #[test]
    fn decode_ranks_into_matches_term_at_into_for_every_rank_at_once() {
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
        let n = terms.len();

        // Ranks stay ascending (the actual precondition); only the output
        // position each one writes to is decorrelated from rank order.
        let targets: Vec<(usize, u32)> = (0..n as u32).map(|r| (n - 1 - r as usize, r)).collect();
        let mut out = vec![None; n];
        automaton.decode_ranks_into(&targets, &mut out);

        let mut buf = Vec::new();
        for (rank, expected) in terms.iter().enumerate() {
            automaton.term_at_into(rank as u32, &mut buf);
            assert_eq!(
                out[n - 1 - rank].as_deref(),
                Some(expected.as_str()),
                "mismatch at rank {rank}"
            );
            assert_eq!(std::str::from_utf8(&buf).unwrap(), expected.as_str());
        }
    }

    /// Several separate clusters of consecutive ranks, with large,
    /// deliberately un-requested gaps between them within the same
    /// automaton -- the exact shape that broke an earlier seek-then-crawl
    /// design (crawling through every unrequested rank in the gap). This
    /// checks only correctness; `dict_bench` covers the cost side.
    #[test]
    fn decode_ranks_into_handles_widely_separated_clusters_within_one_automaton() {
        let terms: Vec<String> = (0..2000)
            .map(|i| format!("http://example.org/entity-{i:05}"))
            .collect();
        let (automaton, _block, terms) = build_both(terms);
        let n = terms.len() as u32;

        let mut targets: Vec<(usize, u32)> = Vec::new();
        let mut pos = 0usize;
        let mut rank = 3u32;
        while rank < n {
            for r in rank..(rank + 5).min(n) {
                targets.push((pos, r));
                pos += 1;
            }
            rank += 331; // a large, deliberately awkward gap
        }
        // A duplicated rank, to check that's handled too -- inserted (not
        // appended) so `targets` stays sorted ascending by rank, its
        // actual precondition.
        let dup = targets[2];
        targets.push((pos, dup.1));
        pos += 1;
        targets.sort_unstable_by_key(|&(_, r)| r);

        let mut out = vec![None; pos];
        automaton.decode_ranks_into(&targets, &mut out);

        let mut buf = Vec::new();
        for &(out_pos, rank) in &targets {
            automaton.term_at_into(rank, &mut buf);
            let expected = std::str::from_utf8(&buf).unwrap();
            assert_eq!(
                out[out_pos].as_deref(),
                Some(expected),
                "mismatch for rank {rank} at output position {out_pos}"
            );
            assert_eq!(terms[rank as usize], expected);
        }
    }

    /// `PackedAutomaton::decode_ranks_into`'s counterpart to
    /// `decode_ranks_into_matches_term_at_into_for_every_rank_at_once`.
    #[test]
    fn packed_decode_ranks_into_matches_term_at_into_for_every_rank_at_once() {
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
        let packed = compile(&automaton);
        let n = terms.len();

        let targets: Vec<(usize, u32)> = (0..n as u32).map(|r| (n - 1 - r as usize, r)).collect();
        let mut out = vec![None; n];
        packed.decode_ranks_into(&targets, &mut out);

        let mut buf = Vec::new();
        for (rank, expected) in terms.iter().enumerate() {
            packed.term_at_into(rank as u32, &mut buf);
            assert_eq!(
                out[n - 1 - rank].as_deref(),
                Some(expected.as_str()),
                "mismatch at rank {rank}"
            );
            assert_eq!(std::str::from_utf8(&buf).unwrap(), expected.as_str());
        }
    }

    /// `PackedAutomaton::decode_ranks_into`'s counterpart to
    /// `decode_ranks_into_handles_widely_separated_clusters_within_one_automaton`.
    #[test]
    fn packed_decode_ranks_into_handles_widely_separated_clusters_within_one_automaton() {
        let terms: Vec<String> = (0..2000)
            .map(|i| format!("http://example.org/entity-{i:05}"))
            .collect();
        let (automaton, _block, terms) = build_both(terms);
        let packed = compile(&automaton);
        let n = terms.len() as u32;

        let mut targets: Vec<(usize, u32)> = Vec::new();
        let mut pos = 0usize;
        let mut rank = 3u32;
        while rank < n {
            for r in rank..(rank + 5).min(n) {
                targets.push((pos, r));
                pos += 1;
            }
            rank += 331;
        }
        let dup = targets[2];
        targets.push((pos, dup.1));
        pos += 1;
        targets.sort_unstable_by_key(|&(_, r)| r);

        let mut out = vec![None; pos];
        packed.decode_ranks_into(&targets, &mut out);

        let mut buf = Vec::new();
        for &(out_pos, rank) in &targets {
            packed.term_at_into(rank, &mut buf);
            let expected = std::str::from_utf8(&buf).unwrap();
            assert_eq!(
                out[out_pos].as_deref(),
                Some(expected),
                "mismatch for rank {rank} at output position {out_pos}"
            );
            assert_eq!(terms[rank as usize], expected);
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

    /// See `perf_probe_insert_cost_automaton`'s doc comment -- same
    /// baseline, same new terms, `PackedAutomaton::insert` in place of
    /// `Automaton::insert`. The one real question this probe exists to
    /// answer: does the mutable unrolled-block representation's per-
    /// insert cost (walk-match, parse each matched state's record back
    /// into a `StagingFrame`, append fresh records bottom-up) stay
    /// competitive with the plain slab's O(1)-per-state pointer
    /// operations, or does record parsing/re-serialization on every
    /// insert give back the decode-side win `perf_probe_decode_cost_
    /// packed_automaton` already confirmed.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_insert_cost_packed_automaton() {
        let entries = insert_probe_baseline_terms();
        let automaton = Automaton::build_sorted(entries.iter().map(|(t, _)| t.as_str()));
        let baseline = compile(&automaton);
        let new_terms = insert_probe_new_terms();

        let mut elapsed = std::time::Duration::ZERO;
        for new_term in &new_terms {
            let mut p = baseline.clone(); // reset to the baseline; not timed
            let start = Instant::now();
            p.insert(new_term);
            elapsed += start.elapsed();
            std::hint::black_box(&p);
        }
        eprintln!(
            "PackedAutomaton::insert, N={INSERT_PROBE_N}, {INSERT_PROBE_ITERS} iters: \
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

    /// See `perf_probe_decode_cost_block`'s doc comment -- same shape,
    /// `hashlike_decode_probe_terms` in place of `decode_probe_terms`,
    /// completing the three-way (`Block`/`Automaton`/`PackedAutomaton`)
    /// decode comparison on the low-redundancy corpus too (the automaton
    /// side already had one; `Block` didn't).
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_cost_block_hashlike() {
        let terms = hashlike_decode_probe_terms(20_000);
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
            "Block decode (hashlike), {} terms x {DECODE_PROBE_REPEATS} repeats ({total_ops} ops each):\n\
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

    /// See `perf_probe_decode_cost_automaton`'s doc comment -- same
    /// terms, `PackedAutomaton` in place of `Automaton`, isolated the
    /// same way for the same `perf stat` attribution reason.
    ///
    /// A methodology correction, worth stating plainly: an earlier
    /// version of this comment claimed the `compile` call above was
    /// "outside the timed region" and therefore not part of what gets
    /// measured. That's true of the *wall-clock* `Instant`-based timing
    /// this test prints, but `perf stat` wraps the *whole process*, so
    /// `compile`'s own cost (real and, for `insert`'s register, not
    /// small -- see below) was silently folded into every `perf stat`
    /// comparison run against this probe, something `Automaton`'s own
    /// probe never pays an equivalent of (it has no separate compile
    /// step). Isolating decode-only cycles properly needs subtracting a
    /// matching "terms + `build_sorted` (+ `compile`, for the packed
    /// side)" setup-only baseline from each full probe's `perf stat`
    /// numbers -- see DESIGN.md's methodology-correction section for
    /// the isolated numbers on both corpora, which changed the
    /// conclusion (this repeats the same `decode_probe_terms` shape the
    /// clustered isolation used, so see that section for the number
    /// specific to *this* probe, not the hashlike one).
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_cost_packed_automaton() {
        let terms = decode_probe_terms();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let packed = compile(&automaton);

        let start = Instant::now();
        for _ in 0..DECODE_PROBE_REPEATS {
            for t in &terms {
                std::hint::black_box(packed.rank(t));
            }
        }
        let rank_elapsed = start.elapsed();

        let mut buf = Vec::new();
        let start = Instant::now();
        for _ in 0..DECODE_PROBE_REPEATS {
            for rank in 0..terms.len() {
                packed.term_at_into(rank as u32, &mut buf);
                std::hint::black_box(&buf);
            }
        }
        let unrank_elapsed = start.elapsed();

        let total_ops = (DECODE_PROBE_REPEATS * terms.len()) as u32;
        eprintln!(
            "PackedAutomaton decode, {} terms x {DECODE_PROBE_REPEATS} repeats ({total_ops} ops each):\n\
             rank:         {rank_elapsed:?} ({:?}/op)\n\
             term_at_into: {unrank_elapsed:?} ({:?}/op)\n\
             size_bytes:   {}",
            terms.len(),
            rank_elapsed / total_ops,
            unrank_elapsed / total_ops,
            packed.size_bytes(),
        );
    }

    /// A cheap, deterministic 64-bit mix (splitmix64) -- see
    /// `examples/perf_scale_memory.rs`'s identical helper for the full
    /// rationale. Duplicated here rather than shared: that file is a
    /// separate binary crate, this is a `#[cfg(test)]` module in the
    /// library, and this one function isn't worth a shared dependency
    /// between them.
    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Low cross-term redundancy by design (see `examples/
    /// perf_scale_memory.rs`'s `hashlike_terms_range` for the full
    /// rationale) -- this is the regime `estimate_packed_bytes`/
    /// `PackedAutomaton::compile` found the largest *size* win in
    /// (DESIGN.md's memory-scaling investigation: ~88% smaller), and so
    /// the regime most likely to also show a real *decode-speed*
    /// difference from cache locality: the unpacked automaton's states
    /// here are scattered across roughly 10x more heap bytes per term
    /// than the packed encoding needs, unlike the highly-shared
    /// namespace-clustered corpus `decode_probe_terms` uses, where both
    /// representations are already small enough to mostly fit in cache
    /// regardless.
    fn hashlike_decode_probe_terms(n: u32) -> Vec<String> {
        let mut terms: Vec<String> = (0..n)
            .map(|i| format!("urn:uuid:{:016x}", splitmix64(i as u64)))
            .collect();
        terms.sort_unstable();
        terms.dedup();
        terms
    }

    /// See `perf_probe_decode_cost_automaton`'s doc comment for the
    /// isolation methodology -- same shape, `hashlike_decode_probe_terms`
    /// in place of `decode_probe_terms`, specifically to test the
    /// cache-locality hypothesis at the scale/redundancy regime most
    /// likely to show it (see `hashlike_decode_probe_terms`'s own doc
    /// comment).
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_cost_automaton_hashlike() {
        let terms = hashlike_decode_probe_terms(20_000);
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
            "Automaton decode (hashlike), {} terms x {DECODE_PROBE_REPEATS} repeats ({total_ops} ops each):\n\
             rank:         {rank_elapsed:?} ({:?}/op)\n\
             term_at_into: {unrank_elapsed:?} ({:?}/op)\n\
             size_bytes:   {}",
            terms.len(),
            rank_elapsed / total_ops,
            unrank_elapsed / total_ops,
            automaton.size_bytes(),
        );
    }

    /// See `perf_probe_decode_cost_automaton_hashlike`'s doc comment,
    /// and `perf_probe_decode_cost_packed_automaton`'s for the
    /// compile-cost-pollution methodology note -- reproduced here too.
    /// This is the probe where it mattered most: on this 20,000-term
    /// low-redundancy corpus, `compile`'s own (register-bookkeeping)
    /// cost is a large fraction of what whole-process `perf stat` had
    /// been reporting as "decode." See DESIGN.md for the isolated,
    /// decode-only numbers.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_cost_packed_automaton_hashlike() {
        let terms = hashlike_decode_probe_terms(20_000);
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let packed = compile(&automaton);

        let start = Instant::now();
        for _ in 0..DECODE_PROBE_REPEATS {
            for t in &terms {
                std::hint::black_box(packed.rank(t));
            }
        }
        let rank_elapsed = start.elapsed();

        let mut buf = Vec::new();
        let start = Instant::now();
        for _ in 0..DECODE_PROBE_REPEATS {
            for rank in 0..terms.len() {
                packed.term_at_into(rank as u32, &mut buf);
                std::hint::black_box(&buf);
            }
        }
        let unrank_elapsed = start.elapsed();

        let total_ops = (DECODE_PROBE_REPEATS * terms.len()) as u32;
        eprintln!(
            "PackedAutomaton decode (hashlike), {} terms x {DECODE_PROBE_REPEATS} repeats ({total_ops} ops each):\n\
             rank:         {rank_elapsed:?} ({:?}/op)\n\
             term_at_into: {unrank_elapsed:?} ({:?}/op)\n\
             size_bytes:   {}",
            terms.len(),
            rank_elapsed / total_ops,
            unrank_elapsed / total_ops,
            packed.size_bytes(),
        );
    }

    /// A "seeked" query shape, not "every rank": several small (5-rank)
    /// clusters scattered across the term set with large, deliberately
    /// awkward gaps between them -- the actual shape `decode_ranks_into`/
    /// `get_terms_seeked_into` exist for (a query touching only part of
    /// a block), as opposed to `stream`'s own "every rank in order"
    /// shape, already measured separately. Same generator as the
    /// correctness tests' "widely separated clusters" case, parameterized
    /// by corpus size instead of hardcoded.
    const DECODE_RANKS_PROBE_REPEATS: usize = 2000;

    fn scattered_rank_targets(n: u32) -> Vec<(usize, u32)> {
        let mut targets = Vec::new();
        let mut pos = 0usize;
        let mut rank = 3u32;
        let gap = (n / 40).max(50); // ~40 clusters spread across the term set
        while rank < n {
            for r in rank..(rank + 5).min(n) {
                targets.push((pos, r));
                pos += 1;
            }
            rank += gap;
        }
        targets
    }

    /// `decode_ranks_into`'s own decode-speed probe -- unlike
    /// `perf_probe_decode_cost_automaton`'s `rank`/`term_at_into`
    /// (independent, single-key operations), this measures the batch
    /// primitive `AutomatonDictionary::get_terms_seeked_into`/
    /// `PackedAutomatonDictionary::get_terms_seeked_into` actually call,
    /// not yet measured for speed anywhere before this (only
    /// correctness, in the differential tests). Isolated the same way as
    /// every other decode probe in this module.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_ranks_cost_automaton_clustered() {
        let terms = decode_probe_terms();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let targets = scattered_rank_targets(terms.len() as u32);
        let mut out = vec![None; targets.len()];

        let start = Instant::now();
        for _ in 0..DECODE_RANKS_PROBE_REPEATS {
            automaton.decode_ranks_into(&targets, &mut out);
            std::hint::black_box(&out);
        }
        let elapsed = start.elapsed();
        let total_ops = (DECODE_RANKS_PROBE_REPEATS * targets.len()) as u32;
        eprintln!(
            "Automaton decode_ranks_into (clustered), {} terms, {} targets x {DECODE_RANKS_PROBE_REPEATS} repeats ({total_ops} ops each): {elapsed:?} ({:?}/op)",
            terms.len(),
            targets.len(),
            elapsed / total_ops,
        );
    }

    /// See `perf_probe_decode_ranks_cost_automaton_clustered`'s doc
    /// comment -- same terms, same targets, `PackedAutomaton` in place
    /// of `Automaton`. `compile` is outside the wall-clock-timed region
    /// but not outside what `perf stat` counts (see the earlier decode
    /// probes' own methodology note) -- subtract `perf_probe_setup_only_
    /// clustered`'s `compile` baseline from any `perf stat` comparison
    /// against this probe.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_ranks_cost_packed_automaton_clustered() {
        let terms = decode_probe_terms();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let packed = compile(&automaton);
        let targets = scattered_rank_targets(terms.len() as u32);
        let mut out = vec![None; targets.len()];

        let start = Instant::now();
        for _ in 0..DECODE_RANKS_PROBE_REPEATS {
            packed.decode_ranks_into(&targets, &mut out);
            std::hint::black_box(&out);
        }
        let elapsed = start.elapsed();
        let total_ops = (DECODE_RANKS_PROBE_REPEATS * targets.len()) as u32;
        eprintln!(
            "PackedAutomaton decode_ranks_into (clustered), {} terms, {} targets x {DECODE_RANKS_PROBE_REPEATS} repeats ({total_ops} ops each): {elapsed:?} ({:?}/op)",
            terms.len(),
            targets.len(),
            elapsed / total_ops,
        );
    }

    /// See `perf_probe_decode_ranks_cost_automaton_clustered`'s doc
    /// comment -- hashlike corpus in place of clustered.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_ranks_cost_automaton_hashlike() {
        let terms = hashlike_decode_probe_terms(20_000);
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let targets = scattered_rank_targets(terms.len() as u32);
        let mut out = vec![None; targets.len()];

        let start = Instant::now();
        for _ in 0..DECODE_RANKS_PROBE_REPEATS {
            automaton.decode_ranks_into(&targets, &mut out);
            std::hint::black_box(&out);
        }
        let elapsed = start.elapsed();
        let total_ops = (DECODE_RANKS_PROBE_REPEATS * targets.len()) as u32;
        eprintln!(
            "Automaton decode_ranks_into (hashlike), {} terms, {} targets x {DECODE_RANKS_PROBE_REPEATS} repeats ({total_ops} ops each): {elapsed:?} ({:?}/op)",
            terms.len(),
            targets.len(),
            elapsed / total_ops,
        );
    }

    /// See `perf_probe_decode_ranks_cost_packed_automaton_clustered`'s
    /// doc comment (including the `compile`-subtraction note, using
    /// `perf_probe_setup_only_hashlike`'s baseline here instead) --
    /// hashlike corpus in place of clustered.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_decode_ranks_cost_packed_automaton_hashlike() {
        let terms = hashlike_decode_probe_terms(20_000);
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let packed = compile(&automaton);
        let targets = scattered_rank_targets(terms.len() as u32);
        let mut out = vec![None; targets.len()];

        let start = Instant::now();
        for _ in 0..DECODE_RANKS_PROBE_REPEATS {
            packed.decode_ranks_into(&targets, &mut out);
            std::hint::black_box(&out);
        }
        let elapsed = start.elapsed();
        let total_ops = (DECODE_RANKS_PROBE_REPEATS * targets.len()) as u32;
        eprintln!(
            "PackedAutomaton decode_ranks_into (hashlike), {} terms, {} targets x {DECODE_RANKS_PROBE_REPEATS} repeats ({total_ops} ops each): {elapsed:?} ({:?}/op)",
            terms.len(),
            targets.len(),
            elapsed / total_ops,
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

    /// Setup-only baselines (terms + `Automaton::build_sorted`, then
    /// separately `PackedAutomaton::compile`) for subtracting `compile`'s
    /// own cost out of the full decode probes' `perf stat` numbers -- see
    /// DESIGN.md's methodology-correction section for why that
    /// subtraction matters (`perf stat` wraps the whole process, so
    /// `compile`'s cost, unique to the packed side, otherwise gets
    /// counted as "decode").
    #[test]
    #[ignore = "manual isolation probe, not part of the regular suite"]
    fn perf_probe_setup_only_clustered() {
        let start = Instant::now();
        let terms = decode_probe_terms();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let build_elapsed = start.elapsed();
        let compile_start = Instant::now();
        let packed = compile(&automaton);
        let compile_elapsed = compile_start.elapsed();
        std::hint::black_box(&packed);
        eprintln!("clustered: build={build_elapsed:?} compile={compile_elapsed:?}");
    }

    #[test]
    #[ignore = "manual isolation probe, not part of the regular suite"]
    fn perf_probe_setup_only_hashlike() {
        let start = Instant::now();
        let terms = hashlike_decode_probe_terms(20_000);
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let build_elapsed = start.elapsed();
        let compile_start = Instant::now();
        let packed = compile(&automaton);
        let compile_elapsed = compile_start.elapsed();
        std::hint::black_box(&packed);
        eprintln!("hashlike: build={build_elapsed:?} compile={compile_elapsed:?}");
    }

    /// The build-speed comparison `PackedAutomaton::build_sorted` was
    /// added for: does building directly (no intermediate `Automaton`
    /// slab) actually cost less than the old `compile(&Automaton::
    /// build_sorted(..))` route? Four single-purpose probes (one variant
    /// each), not two combined ones, so `perf stat` attributes cycles to
    /// exactly one build path at a time -- correctness (`size_bytes`/
    /// rank/term_at_into parity) is already covered by
    /// `packed_automaton_build_sorted_matches_compile_from_automaton`,
    /// not re-checked here.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_build_via_automaton_clustered() {
        let terms = decode_probe_terms();
        let start = Instant::now();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let packed = compile(&automaton);
        let elapsed = start.elapsed();
        std::hint::black_box(&packed);
        eprintln!(
            "clustered, {} terms, via Automaton::build_sorted + compile: {elapsed:?}",
            terms.len(),
        );
    }

    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_build_direct_clustered() {
        let terms = decode_probe_terms();
        let start = Instant::now();
        let packed = PackedAutomaton::build_sorted(terms.iter().map(String::as_str));
        let elapsed = start.elapsed();
        std::hint::black_box(&packed);
        eprintln!(
            "clustered, {} terms, PackedAutomaton::build_sorted (direct): {elapsed:?}",
            terms.len(),
        );
    }

    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_build_via_automaton_hashlike() {
        let terms = hashlike_decode_probe_terms(20_000);
        let start = Instant::now();
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        let packed = compile(&automaton);
        let elapsed = start.elapsed();
        std::hint::black_box(&packed);
        eprintln!(
            "hashlike, {} terms, via Automaton::build_sorted + compile: {elapsed:?}",
            terms.len(),
        );
    }

    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_build_direct_hashlike() {
        let terms = hashlike_decode_probe_terms(20_000);
        let start = Instant::now();
        let packed = PackedAutomaton::build_sorted(terms.iter().map(String::as_str));
        let elapsed = start.elapsed();
        std::hint::black_box(&packed);
        eprintln!(
            "hashlike, {} terms, PackedAutomaton::build_sorted (direct): {elapsed:?}",
            terms.len(),
        );
    }
}
