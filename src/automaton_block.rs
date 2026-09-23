//! `AutomatonBlock`: mirrors `Block`'s public interface (`block.rs`), but
//! backed by `micro_fst::Automaton` instead of `fst::Map` -- the piece
//! that makes the custom automaton usable wherever `Block` currently is,
//! without touching `Block`/`Dictionary` themselves (this project's own
//! "alongside, promoted only if it wins" rule, from the approved plan).
//!
//! `Automaton` itself only knows `term -> rank`, deliberately decoupled
//! from `TermId` (see its own module docs). `AutomatonBlock` is the thin
//! layer that pairs it with the external id array `Block` already
//! carries, exactly matching `Block`'s own `{ fst, ids, min_key }` shape.
//!
//! `merge_into`/`remove` originally rebuilt the whole `ids` array on every
//! mutation via a `HashMap<String, TermId>` (or two), one full decode per
//! pass -- a documented "smaller first integration step" simplification
//! at the time, but never cost-measured against `Block`'s own approach
//! until `examples/perf_dictionary_automaton_vs_fst.rs` found it made
//! every mutating `AutomatonDictionary` operation *slower* than
//! `Dictionary`, not faster (DESIGN.md's "measured end to end" section).
//! Fixed in two steps, each measured before moving to the next rather
//! than assumed sufficient:
//! 1. Replaced the `HashMap`/`String`-allocating rebuild with a single
//!    sorted-merge walk, mirroring `Block::merge_into`'s algorithm
//!    exactly (`remove` needs no walk at all: removing one entry from a
//!    sorted sequence is just `Vec::remove` at its old rank). Closed most
//!    of the gap (46x slower than `Block::merge_into`, block-level-only,
//!    down to ~3x) but not all of it.
//! 2. The remaining gap turned out to be structural, not incidental: the
//!    walk read old content via repeated `term_at_into(rank, ..)` calls,
//!    each restarting from the automaton's root, while `Block`'s own
//!    walk uses `fst::Map::stream()`, which reuses shared-prefix
//!    traversal state between consecutive keys. `Automaton::stream`
//!    (`micro_fst.rs`) is the equivalent incremental walk for this
//!    automaton; switching `merge_into` (and `decode_into`) to it closed
//!    the rest of the gap and turned it into a real win (see DESIGN.md).
//!
//! Both `merge_into`/`remove` report an `unchanged_prefix`/`skip` now
//! too, the same way `Block`'s `MergeOutcome` does, so `AutomatonDictionary`
//! can skip re-touching the reverse mapping for the part of the block
//! nothing moved in.
//!
//! Wired into `AutomatonDictionary` (`automaton_dictionary.rs`), which
//! gives this its real, end-to-end correctness signal (the 4-way fuzz
//! differential in `tests/dictionary.rs`) on top of the block-level
//! differential tests below.

use crate::TermId;
use crate::entries_arena::EntriesArena;
use crate::micro_fst::Automaton;

#[derive(Debug)]
pub(crate) struct AutomatonBlock {
    automaton: Automaton,
    ids: Vec<TermId>,
    min_key: Box<str>,
}

/// The result of [`AutomatonBlock::merge_into`]: how many `incoming`
/// terms were new, and how much of the block's leading content is
/// unchanged -- see [`crate::block::MergeOutcome`], which this mirrors
/// exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AutomatonMergeOutcome {
    pub(crate) added: usize,
    pub(crate) unchanged_prefix: usize,
}

impl AutomatonBlock {
    /// Builds an immutable block from an already sorted, deduplicated,
    /// exact-size sequence of `(term, id)` pairs -- same contract as
    /// `Block::build`.
    pub(crate) fn build<'a>(
        entries: impl ExactSizeIterator<Item = (&'a str, TermId)>,
    ) -> AutomatonBlock {
        let pairs: Vec<(&'a str, TermId)> = entries.collect();
        let min_key: Box<str> = pairs.first().map_or_else(Box::default, |&(t, _)| t.into());
        let ids: Vec<TermId> = pairs.iter().map(|&(_, id)| id).collect();
        let automaton = Automaton::build_sorted(pairs.iter().map(|&(t, _)| t));
        AutomatonBlock {
            automaton,
            ids,
            min_key,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn min_key(&self) -> &str {
        &self.min_key
    }

    /// `Some(local_rank)` if `term` is present.
    pub(crate) fn rank(&self, term: &str) -> Option<usize> {
        self.automaton.rank(term).map(|r| r as usize)
    }

    pub(crate) fn id_at(&self, local_rank: usize) -> TermId {
        self.ids[local_rank]
    }

    pub(crate) fn term_at_into(&self, local_rank: usize, buf: &mut Vec<u8>) {
        self.automaton.term_at_into(local_rank as u32, buf);
    }

    pub(crate) fn term_at(&self, local_rank: usize) -> String {
        let mut buf = Vec::new();
        self.term_at_into(local_rank, &mut buf);
        String::from_utf8(buf).expect("keys were inserted as valid UTF-8 str bytes")
    }

    /// `Block::payload_bytes`'s analog -- see `Automaton::size_bytes`'s
    /// own doc comment for exactly what's included/excluded.
    pub(crate) fn payload_bytes(&self) -> usize {
        self.automaton.size_bytes()
    }

    /// See `Automaton::estimate_packed_bytes` -- a measurement only, not
    /// a real encoding.
    pub(crate) fn estimate_packed_bytes(&self) -> usize {
        self.automaton.estimate_packed_bytes()
    }

    /// Decodes a *sorted* (ascending; may repeat) set of `(output
    /// position, local_rank)` pairs directly into `out[position]`, via
    /// `Automaton::decode_ranks_into` -- one recursive descent that
    /// visits each state on some target's path once, however many
    /// targets share it, instead of one independent root-to-leaf
    /// `term_at_into` call per rank. `positions` need not be sorted (the
    /// widening below sorts a small owned copy); see
    /// `Automaton::decode_ranks_into`'s own doc comment for how it
    /// decides what to share without any tuned distance/gap threshold.
    pub(crate) fn decode_sorted_ranks_into(
        &self,
        positions: &[(usize, u16)],
        out: &mut [Option<String>],
    ) {
        let mut targets: Vec<(usize, u32)> = positions
            .iter()
            .map(|&(pos, rank)| (pos, rank as u32))
            .collect();
        targets.sort_unstable_by_key(|&(_, rank)| rank);
        self.automaton.decode_ranks_into(&targets, out);
    }

    /// Appends every `(term, id)` pair in sorted order onto `out` --
    /// `Block::decode_into`'s contract, via `Automaton::stream` (a single
    /// incremental walk, sharing common-prefix traversal cost between
    /// consecutive entries) rather than one `term_at_into` call per rank.
    pub(crate) fn decode_into(&self, out: &mut EntriesArena) {
        let mut stream = self.automaton.stream();
        let mut rank = 0usize;
        while let Some(term_bytes) = stream.next() {
            let term = std::str::from_utf8(term_bytes)
                .expect("keys were inserted as valid UTF-8 str bytes");
            out.push(term, self.ids[rank]);
            rank += 1;
        }
    }

    /// Merges `incoming` (already sorted) into this block: existing terms
    /// keep their id, new ones get one from `alloc_id`; `resolved` is
    /// called once per incoming term, in order, with its final id -- same
    /// caller-facing contract as `Block::merge_into`, including its
    /// `unchanged_prefix` reporting now (see the module doc comment).
    ///
    /// A single sorted-merge walk of this block's existing content (via
    /// `Automaton::stream`, run against the *pre-mutation* automaton)
    /// against `incoming`, exactly `Block::merge_into`'s algorithm --
    /// only reached here read-only, so it must run and finish before
    /// `self.automaton.insert_sorted` below, which mutates in place and
    /// would otherwise invalidate the walk (and the ranks `self.ids` is
    /// still indexed by while it runs).
    pub(crate) fn merge_into<'a>(
        &mut self,
        incoming: impl Iterator<Item = &'a str>,
        mut alloc_id: impl FnMut() -> TermId,
        mut resolved: impl FnMut(TermId),
    ) -> AutomatonMergeOutcome {
        let incoming: Vec<&'a str> = incoming.collect();
        if incoming.is_empty() {
            return AutomatonMergeOutcome {
                added: 0,
                unchanged_prefix: self.ids.len(),
            };
        }

        let mut new_ids = Vec::with_capacity(self.ids.len() + incoming.len());
        let mut unchanged_prefix = None;
        let mut added = 0usize;
        let mut inc = incoming.iter().copied().peekable();
        let mut stream = self.automaton.stream();
        let mut old_rank = 0usize;
        while let Some(old_term_bytes) = stream.next() {
            let old_term = std::str::from_utf8(old_term_bytes)
                .expect("keys were inserted as valid UTF-8 str bytes");
            while let Some(&next) = inc.peek() {
                if next < old_term {
                    unchanged_prefix.get_or_insert(new_ids.len());
                    let id = alloc_id();
                    added += 1;
                    new_ids.push(id);
                    resolved(id);
                    inc.next();
                } else {
                    break;
                }
            }
            let matched = inc.peek() == Some(&old_term);
            let id = self.ids[old_rank];
            new_ids.push(id);
            if matched {
                inc.next();
                resolved(id);
            }
            old_rank += 1;
        }
        for _ in inc {
            unchanged_prefix.get_or_insert(new_ids.len());
            let id = alloc_id();
            added += 1;
            new_ids.push(id);
            resolved(id);
        }
        let unchanged_prefix = unchanged_prefix.unwrap_or(new_ids.len());

        if added > 0 {
            // Discards `insert_sorted`'s own new/existing flags: the merge
            // walk above already determined that from the same sortedness
            // guarantee `insert_sorted` itself relies on, so recomputing
            // it a second way would be redundant, not a cross-check
            // that's worth its own cost on every mutation.
            self.automaton.insert_sorted(incoming.iter().copied());
            self.ids = new_ids;
            // `unchanged_prefix > 0` means rank 0 -- exactly what
            // `min_key` caches -- is among the leading entries the merge
            // walk found unmoved, so `min_key` is still correct as-is.
            // Recomputing it unconditionally here (found via a fresh
            // `perf record` of `sequential_insert`, DESIGN.md's
            // allocation-pattern audit) paid for a full root-restart
            // `term_at_into` walk plus a fresh `String` allocation on
            // *every* mutation, the overwhelming majority of which don't
            // touch the block's minimum term at all.
            if unchanged_prefix == 0 {
                self.min_key = self.term_at(0).into_boxed_str();
            }
        }

        AutomatonMergeOutcome {
            added,
            unchanged_prefix,
        }
    }

    /// Removes `term` if present, returning its former id and the count of
    /// leading entries left unchanged (== the removed term's old rank,
    /// since removing one entry from a sorted sequence only shifts
    /// everything after it -- see [`AutomatonMergeOutcome::unchanged_prefix`]).
    /// No decode needed at all: unlike `merge_into`, where new entries can
    /// land anywhere, a single removal's effect on `ids`' order is fully
    /// known from `old_rank` alone.
    pub(crate) fn remove(&mut self, term: &str) -> Option<(TermId, usize)> {
        let old_rank = self.rank(term)?;
        let removed_id = self.ids[old_rank];

        let removed = self.automaton.remove(term);
        debug_assert!(removed);

        self.ids.remove(old_rank);
        // Only rank 0's own removal can change `min_key` -- removing
        // anything else leaves the old rank-0 entry exactly where it
        // was (same reasoning as `merge_into`'s own `unchanged_prefix`
        // check just above).
        if self.ids.is_empty() {
            self.min_key = Box::default();
        } else if old_rank == 0 {
            self.min_key = self.term_at(0).into_boxed_str();
        }

        Some((removed_id, old_rank))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;

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

    fn build_both(mut terms: Vec<String>) -> (AutomatonBlock, Block, Vec<String>) {
        terms.sort_unstable();
        terms.dedup();
        let entries: Vec<(&str, TermId)> = terms
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), TermId(i as u32)))
            .collect();
        let automaton_block = AutomatonBlock::build(entries.iter().copied());
        let block = Block::build(entries.into_iter());
        (automaton_block, block, terms)
    }

    #[test]
    fn matches_block_on_build_rank_and_unrank() {
        let (ab, block, terms) = build_both(namespace_clustered_terms(500));
        assert_eq!(ab.len(), block.len());
        assert_eq!(ab.min_key(), block.min_key());
        for (rank, term) in terms.iter().enumerate() {
            assert_eq!(ab.rank(term), Some(rank));
            assert_eq!(ab.rank(term), block.rank(term));
            assert_eq!(ab.id_at(rank), block.id_at(rank));
            assert_eq!(ab.term_at(rank), *term);
            assert_eq!(ab.term_at(rank), block.term_at(rank));
        }
        assert_eq!(ab.rank("not-present-at-all"), None);
    }

    #[test]
    fn decode_into_matches_original_content() {
        let (ab, _block, terms) = build_both(namespace_clustered_terms(300));
        let mut arena = EntriesArena::new();
        ab.decode_into(&mut arena);
        assert_eq!(arena.len(), terms.len());
        for (i, term) in terms.iter().enumerate() {
            assert_eq!(arena.term_at(i), term.as_str());
            assert_eq!(arena.id_at(i), ab.id_at(i));
        }
    }

    #[test]
    fn merge_into_existing_terms_keep_their_id_new_ones_get_fresh_ids() {
        let (mut ab, _block, terms) = build_both(namespace_clustered_terms(200));
        let old_ids: Vec<TermId> = (0..ab.len()).map(|r| ab.id_at(r)).collect();

        let mut next_id = 100_000u32;
        let mut resolved_ids = Vec::new();
        let mut incoming: Vec<String> = terms.iter().step_by(7).cloned().collect(); // duplicates
        incoming.push("http://example.org/brand-new-1".to_string());
        incoming.push("http://example.org/brand-new-2".to_string());
        incoming.sort_unstable();
        incoming.dedup();

        let outcome = ab.merge_into(
            incoming.iter().map(String::as_str),
            || {
                let id = TermId(next_id);
                next_id += 1;
                id
            },
            |id| resolved_ids.push(id),
        );
        assert_eq!(outcome.added, 2); // the two brand-new terms
        assert!(outcome.unchanged_prefix <= terms.len());
        assert_eq!(resolved_ids.len(), incoming.len());

        // Every originally-present term (whether re-submitted or not)
        // keeps its original id.
        for (term, &old_id) in terms.iter().zip(&old_ids) {
            assert_eq!(ab.rank(term).map(|r| ab.id_at(r)), Some(old_id));
        }
        // The new terms round-trip correctly and got ids from alloc_id.
        for term in [
            "http://example.org/brand-new-1",
            "http://example.org/brand-new-2",
        ] {
            let rank = ab.rank(term).expect("just inserted");
            assert!(ab.id_at(rank).0 >= 100_000);
        }
        assert_eq!(ab.len(), terms.len() + 2);
    }

    #[test]
    fn remove_drops_the_term_and_keeps_everything_else_correct() {
        let (mut ab, _block, terms) = build_both(namespace_clustered_terms(150));
        let target = terms[terms.len() / 2].clone();
        let target_rank = ab.rank(&target).unwrap();
        let target_id = ab.id_at(target_rank);

        let removed = ab.remove(&target);
        assert_eq!(removed, Some((target_id, target_rank)));
        assert_eq!(ab.rank(&target), None);
        assert_eq!(ab.len(), terms.len() - 1);
        assert_eq!(ab.remove(&target), None); // already gone

        for term in &terms {
            if term != &target {
                assert!(ab.rank(term).is_some(), "{term} should still be present");
            }
        }
    }

    /// Originally the diagnostic that found the cause of a surprising
    /// end-to-end result (`examples/perf_dictionary_automaton_vs_fst.rs`):
    /// `AutomatonDictionary::insert` measured *slower* than
    /// `Dictionary::insert`, even though the bare `Automaton::insert`
    /// primitive was already measured faster than `Block`'s
    /// decode-and-rebuild path. Isolating just `merge_into` (no
    /// dictionary-level routing/resync on top) showed the old
    /// `HashMap`-based `ids` rebuild was already ~46x slower than
    /// `Block::merge_into` at the block level alone -- now left in place
    /// as the before/after check for the single-sorted-merge-walk
    /// rewrite that replaced it (see the module doc comment). `#[ignore]`d:
    /// prints wall-clock timing, not a pass/fail assertion.
    #[test]
    #[ignore]
    fn perf_probe_merge_into_single_term_automaton_vs_block() {
        let preload = namespace_clustered_terms(64); // ~256 terms, block-sized
        let (mut ab, block, _terms) = build_both(preload);

        // Small N: this loop inserts into one growing block with no split,
        // so a large N would let the block grow far past realistic size
        // and dominate the average with late-stage cost -- fine for a
        // directional check, not meant as a bounded-block-size citation.
        let incoming = namespace_clustered_terms_range(1_000_000, 500);
        let mut next_id = 10_000_000u32;

        let start = std::time::Instant::now();
        for t in &incoming {
            ab.merge_into(
                std::iter::once(t.as_str()),
                || {
                    let id = TermId(next_id);
                    next_id += 1;
                    id
                },
                |_| {},
            );
        }
        let automaton_elapsed = start.elapsed();

        let mut arena = EntriesArena::new();
        let mut next_id = 20_000_000u32;
        let start = std::time::Instant::now();
        for t in &incoming {
            arena.clear();
            block.merge_into(
                std::iter::once(t.as_str()),
                &mut arena,
                || {
                    let id = TermId(next_id);
                    next_id += 1;
                    id
                },
                |_| {},
            );
        }
        let block_elapsed = start.elapsed();

        eprintln!(
            "merge_into single term (block-level only, no dictionary resync), {} ops:\n\
             AutomatonBlock: {automaton_elapsed:?} ({:?}/op)\n\
             Block:          {block_elapsed:?} ({:?}/op)",
            incoming.len(),
            automaton_elapsed / incoming.len() as u32,
            block_elapsed / incoming.len() as u32,
        );
    }

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
}
