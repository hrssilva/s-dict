//! A from-scratch, minimal acyclic finite-state automaton (the "grammar"
//! over one block's sorted term set -- see DESIGN.md's "Custom incremental
//! micro-FST" notes), built for eventual incremental mutation instead of
//! the `fst` crate's full-rebuild-per-change contract. This module is
//! self-contained: no dependency on `fst` or on `crate::TermId`, so it can
//! be built and proven correct/minimal entirely on its own before anything
//! elsewhere in the crate depends on it (per the staged plan -- this file
//! covers stages 1-3: bulk construction, differential testing against the
//! current `Block`, and incremental single-term `insert`. Deletion and
//! wiring this into `Block` as a measured comparison implementation are
//! later, not-yet-started stages.)
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
//! Deliberately standalone for now: nothing outside this module's own
//! tests calls into it yet (see the staged plan in DESIGN.md), so a
//! non-test build sees the whole module as dead code -- expected at this
//! stage, not a sign anything is unfinished by accident.

#![allow(dead_code)]

use std::collections::HashMap;
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Signature {
    is_final: bool,
    transitions: Vec<(u8, u32, StateId)>,
}

impl Signature {
    fn of(state: &State) -> Signature {
        Signature {
            is_final: state.is_final,
            transitions: state.transitions.iter().collect(),
        }
    }
}

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
    register: HashMap<Signature, StateId>,
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
        // path[i] = (state after i bytes of `last_word`, byte consumed to
        // reach it from path[i - 1]); path[0]'s byte is an unused
        // placeholder, since root has no incoming transition.
        let mut path: Vec<(StateId, u8)> = vec![(root, 0)];
        let mut last_word: Vec<u8> = Vec::new();
        let mut register: HashMap<Signature, StateId> = HashMap::new();

        for term in terms {
            let word = term.as_bytes();
            let common = common_prefix_len(&last_word, word);

            confirm_path(&mut states, &mut path, &mut register, common + 1);

            let mut cur = path.last().expect("path always has at least the root").0;
            for &b in &word[common..] {
                let new_id = StateId::from_slot(states.len() as u32);
                states.push(State::default());
                states[cur.slot() as usize]
                    .transitions
                    .insert_transition(b, new_id);
                path.push((new_id, b));
                cur = new_id;
            }
            states[cur.slot() as usize].is_final = true;

            last_word.clear();
            last_word.extend_from_slice(word);
        }

        confirm_path(&mut states, &mut path, &mut register, 0);

        Automaton {
            states,
            root,
            register,
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

        for i in 1..path.len() {
            let (original, byte) = path[i];
            let cloned = clone_state(&mut self.states, original);
            path[i].0 = cloned;
            self.states[path[i - 1].0.slot() as usize]
                .transitions
                .set_target_for_byte(byte, cloned);
        }

        let mut cur = path.last().expect("path always has at least the root").0;
        for &b in &word[matched..] {
            let new_id = StateId::from_slot(self.states.len() as u32);
            self.states.push(State::default());
            self.states[cur.slot() as usize]
                .transitions
                .insert_transition(b, new_id);
            path.push((new_id, b));
            cur = new_id;
        }
        self.states[cur.slot() as usize].is_final = true;

        confirm_path(&mut self.states, &mut path, &mut self.register, 0);
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

    /// The total number of terms this automaton accepts.
    pub(crate) fn len(&self) -> usize {
        self.states[self.root.slot() as usize].count as usize
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
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Allocates a fresh state that starts as an exact copy of `original`'s
/// content (same transitions, same finality) -- used when incremental
/// insertion needs to change a state that might be shared with other
/// terms: the original is left untouched (still valid for whoever else
/// references it) and the clone becomes what this path points at instead.
fn clone_state(states: &mut Vec<State>, original: StateId) -> StateId {
    states.push(states[original.slot() as usize].clone());
    StateId::from_slot((states.len() - 1) as u32)
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
fn confirm_path(
    states: &mut [State],
    path: &mut Vec<(StateId, u8)>,
    register: &mut HashMap<Signature, StateId>,
    keep_from: usize,
) {
    while path.len() > keep_from.max(1) {
        let (id, byte) = path.pop().expect("loop guard ensures at least one entry");
        let parent = path.last().expect("path always has at least the root").0;

        finalize_outputs_and_count(states, id);

        let sig = Signature::of(&states[id.slot() as usize]);
        let canonical = match register.get(&sig) {
            Some(&existing) => existing, // `id`'s slot becomes unreachable garbage; see struct docs
            None => {
                register.insert(sig, id);
                id
            }
        };
        states[parent.slot() as usize]
            .transitions
            .set_target_for_byte(byte, canonical);
    }
    if keep_from == 0 {
        finalize_outputs_and_count(states, path[0].0);
    }
}

/// Computes and stores `id`'s outgoing transitions' output values (the
/// prefix-sum-of-sibling-subtree-counts technique, see the module doc
/// comment) and `id`'s own `count`, assuming every transition's target is
/// already confirmed (its `count` already final).
fn finalize_outputs_and_count(states: &mut [State], id: StateId) {
    let targets: Vec<StateId> = states[id.slot() as usize]
        .transitions
        .iter()
        .map(|(_, _, target)| target)
        .collect();
    let is_final = states[id.slot() as usize].is_final;
    let mut outputs = Vec::with_capacity(targets.len());
    // A word accepted *at this state* is a prefix of every word reached
    // by descending into any child, and a prefix always sorts before its
    // own extensions -- so this state's own acceptance shifts every
    // child's output by one, not just this state's `count` (which needed
    // the same `+1`, so folding it into the running total up front gives
    // both for free).
    let mut running = u32::from(is_final);
    for target in targets {
        outputs.push(running);
        running += states[target.slot() as usize].count;
    }
    states[id.slot() as usize].transitions.set_outputs(&outputs);
    states[id.slot() as usize].count = running;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TermId;
    use crate::block::Block;
    use crate::entries_arena::{EntriesArena, merge_into};
    use std::time::Instant;

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
        (0..n)
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
}
