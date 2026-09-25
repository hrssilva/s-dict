//! `PackedAutomatonBlock`: `AutomatonBlock`'s exact role (`automaton_block.rs`),
//! `micro_fst::PackedAutomaton` in place of `micro_fst::Automaton` -- the
//! piece that lets the byte-packed representation be measured end to end,
//! at real corpus scale, through the same block/dictionary macro structure
//! every other implementation in this crate shares, instead of only in
//! isolated decode/insert microbenchmarks (`micro_fst.rs`'s own
//! `perf_probe_*` tests) or the hand-computed examples worked through in
//! conversation. This is exactly the kind of comparison
//! `examples/perf_scale_memory.rs` already ran for `AutomatonDictionary`
//! against `Dictionary` and found the original memory-scaling problem
//! with -- wiring `PackedAutomaton` in the same way lets that same
//! benchmark answer whether the byte-packed representation's measured
//! decode parity and insert-memory bloat (DESIGN.md's "the mutable
//! unrolled-block version" and its follow-on corrections) hold up in a
//! real, mutating, splitting/merging dictionary, not just an isolated
//! automaton in a test.
//!
//! Deliberately smaller in scope than `AutomatonBlock` itself, which is
//! already smaller in scope than `Block` -- `PackedAutomaton` has no
//! incremental `remove`, so `remove` rebuilds the whole block from
//! scratch over its remaining terms, the same full-rebuild shape
//! `Block`'s own fst-backed `remove` already has, not a regression
//! introduced here. `decode_into`/`merge_into`'s existing-content walk
//! uses `PackedAutomaton::stream` (an incremental walk: a common prefix's
//! traversal cost is shared between consecutive entries instead of one
//! independent root descent per rank), and `merge_into`'s own mutation
//! uses `PackedAutomaton::insert_sorted` (a persistent-path batch insert,
//! `Automaton::insert_sorted`'s own technique targeting the packed
//! representation) rather than a loop of single-term `insert` calls --
//! see both methods' own doc comments and DESIGN.md's dictionary-level
//! insert-speed sections for the shapes these replaced and the real,
//! measured costs they were.

use crate::TermId;
use crate::entries_arena::EntriesArena;
use crate::micro_fst::PackedAutomaton;

#[derive(Debug)]
pub(crate) struct PackedAutomatonBlock {
    packed: PackedAutomaton,
    ids: Vec<TermId>,
    min_key: Box<str>,
}

/// See [`crate::automaton_block::AutomatonMergeOutcome`], which this
/// mirrors exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedMergeOutcome {
    pub(crate) added: usize,
    pub(crate) unchanged_prefix: usize,
}

impl PackedAutomatonBlock {
    /// Builds an immutable block from an already sorted, deduplicated,
    /// exact-size sequence of `(term, id)` pairs -- same contract as
    /// `Block::build`/`AutomatonBlock::build`. Via `PackedAutomaton::
    /// build_sorted`, the direct confirm-on-divergence build targeting
    /// the packed representation itself -- no intermediate `Automaton`
    /// slab constructed and discarded (see `PackedAutomaton::build_sorted`'s
    /// own doc comment for why that used to be real, measured waste).
    pub(crate) fn build<'a>(
        entries: impl ExactSizeIterator<Item = (&'a str, TermId)>,
    ) -> PackedAutomatonBlock {
        let pairs: Vec<(&'a str, TermId)> = entries.collect();
        let min_key: Box<str> = pairs.first().map_or_else(Box::default, |&(t, _)| t.into());
        let ids: Vec<TermId> = pairs.iter().map(|&(_, id)| id).collect();
        let packed = PackedAutomaton::build_sorted(pairs.iter().map(|&(t, _)| t));
        PackedAutomatonBlock {
            packed,
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

    pub(crate) fn rank(&self, term: &str) -> Option<usize> {
        self.packed.rank(term).map(|r| r as usize)
    }

    pub(crate) fn id_at(&self, local_rank: usize) -> TermId {
        self.ids[local_rank]
    }

    pub(crate) fn term_at_into(&self, local_rank: usize, buf: &mut Vec<u8>) {
        self.packed.term_at_into(local_rank as u32, buf);
    }

    pub(crate) fn term_at(&self, local_rank: usize) -> String {
        let mut buf = Vec::new();
        self.term_at_into(local_rank, &mut buf);
        String::from_utf8(buf).expect("keys were inserted as valid UTF-8 str bytes")
    }

    /// `Block::payload_bytes`'s analog -- see `PackedAutomaton::size_bytes`'s
    /// own doc comment for exactly what's included (and, under sustained
    /// inserts, what dead-record bloat this can carry -- see DESIGN.md).
    pub(crate) fn payload_bytes(&self) -> usize {
        self.packed.size_bytes()
    }

    /// See `PackedAutomaton::reachable_state_count` -- a measurement
    /// only, for the states-per-term investigation (DESIGN.md).
    pub(crate) fn reachable_state_count(&self) -> usize {
        self.packed.reachable_state_count()
    }

    /// Appends every `(term, id)` pair in sorted order onto `out` --
    /// `Block::decode_into`'s contract, via `PackedAutomaton::stream`
    /// (a single incremental walk, sharing common-prefix traversal cost
    /// between consecutive entries) rather than one `term_at_into` call
    /// per rank.
    pub(crate) fn decode_into(&self, out: &mut EntriesArena) {
        let mut stream = self.packed.stream();
        let mut rank = 0usize;
        while let Some(term_bytes) = stream.next() {
            let term = std::str::from_utf8(term_bytes)
                .expect("keys were inserted as valid UTF-8 str bytes");
            out.push(term, self.ids[rank]);
            rank += 1;
        }
    }

    /// Decodes a *sorted* (ascending; may repeat) set of `(output
    /// position, local_rank)` pairs directly into `out[position]`, via
    /// `PackedAutomaton::decode_ranks_into` -- one recursive descent that
    /// visits each state on some target's path once, however many
    /// targets share it, instead of one independent root-to-leaf
    /// `term_at_into` call per rank. `positions` need not be sorted (the
    /// widening below sorts a small owned copy); mirrors
    /// `AutomatonBlock::decode_sorted_ranks_into` exactly.
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
        self.packed.decode_ranks_into(&targets, out);
    }

    /// Merges `incoming` (already sorted) into this block: existing terms
    /// keep their id, new ones get one from `alloc_id`; `resolved` is
    /// called once per incoming term, in order, with its final id -- same
    /// contract as `AutomatonBlock::merge_into`, and now the same
    /// algorithm too: a single sorted-merge walk of this block's existing
    /// content via `PackedAutomaton::stream` (run against the
    /// *pre-mutation* automaton, so it must finish before `insert_sorted`
    /// below, which mutates in place) against `incoming`, followed by one
    /// `PackedAutomaton::insert_sorted` call over the whole batch --
    /// not a loop of single-term `insert` calls, which has no memory
    /// between terms and so re-walks/re-confirms a shared prefix once
    /// per term instead of once per batch (a real, measured regression
    /// for exactly the redundant content this representation should
    /// benefit most from -- see DESIGN.md's "the real cause" section).
    pub(crate) fn merge_into<'a>(
        &mut self,
        incoming: impl Iterator<Item = &'a str>,
        mut alloc_id: impl FnMut() -> TermId,
        mut resolved: impl FnMut(TermId),
    ) -> PackedMergeOutcome {
        let incoming: Vec<&'a str> = incoming.collect();
        if incoming.is_empty() {
            return PackedMergeOutcome {
                added: 0,
                unchanged_prefix: self.ids.len(),
            };
        }

        let mut new_ids = Vec::with_capacity(self.ids.len() + incoming.len());
        let mut unchanged_prefix = None;
        let mut added = 0usize;
        let mut inc = incoming.iter().copied().peekable();
        let mut stream = self.packed.stream();
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
            self.packed.insert_sorted(incoming.iter().copied());
            self.ids = new_ids;
            if unchanged_prefix == 0 {
                self.min_key = self.term_at(0).into_boxed_str();
            }
        }

        PackedMergeOutcome {
            added,
            unchanged_prefix,
        }
    }

    /// Removes `term` if present, returning its former id and the count
    /// of leading entries left unchanged (== the removed term's old rank,
    /// same reasoning as `AutomatonBlock::remove`). Unlike
    /// `AutomatonBlock::remove` (an O(1) `Vec::remove`, no decode),
    /// `PackedAutomaton` has no incremental remove at all -- this decodes
    /// every other entry and rebuilds the whole block from scratch, the
    /// same full-rebuild shape `Block`'s fst-backed `remove` already has.
    /// An honest, not-hidden cost: worth watching for in any timing
    /// comparison that includes removals.
    pub(crate) fn remove(&mut self, term: &str) -> Option<(TermId, usize)> {
        let old_rank = self.rank(term)?;
        let removed_id = self.ids[old_rank];

        let mut remaining_terms: Vec<String> = Vec::with_capacity(self.ids.len() - 1);
        let mut stream = self.packed.stream();
        let mut local_rank = 0usize;
        while let Some(term_bytes) = stream.next() {
            if local_rank != old_rank {
                remaining_terms.push(
                    std::str::from_utf8(term_bytes)
                        .expect("keys were inserted as valid UTF-8 str bytes")
                        .to_string(),
                );
            }
            local_rank += 1;
        }
        self.ids.remove(old_rank);
        self.packed = PackedAutomaton::build_sorted(remaining_terms.iter().map(String::as_str));

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

    fn build_both(mut terms: Vec<String>) -> (PackedAutomatonBlock, Block, Vec<String>) {
        terms.sort_unstable();
        terms.dedup();
        let entries: Vec<(&str, TermId)> = terms
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), TermId(i as u32)))
            .collect();
        let packed_block = PackedAutomatonBlock::build(entries.iter().copied());
        let block = Block::build(entries.into_iter());
        (packed_block, block, terms)
    }

    #[test]
    fn matches_block_on_build_rank_and_unrank() {
        let (pb, block, terms) = build_both(namespace_clustered_terms(500));
        assert_eq!(pb.len(), block.len());
        assert_eq!(pb.min_key(), block.min_key());
        for (rank, term) in terms.iter().enumerate() {
            assert_eq!(pb.rank(term), Some(rank));
            assert_eq!(pb.rank(term), block.rank(term));
            assert_eq!(pb.id_at(rank), block.id_at(rank));
            assert_eq!(pb.term_at(rank), *term);
            assert_eq!(pb.term_at(rank), block.term_at(rank));
        }
        assert_eq!(pb.rank("not-present-at-all"), None);
    }

    #[test]
    fn decode_into_matches_original_content() {
        let (pb, _block, terms) = build_both(namespace_clustered_terms(300));
        let mut arena = EntriesArena::new();
        pb.decode_into(&mut arena);
        assert_eq!(arena.len(), terms.len());
        for (i, term) in terms.iter().enumerate() {
            assert_eq!(arena.term_at(i), term.as_str());
            assert_eq!(arena.id_at(i), pb.id_at(i));
        }
    }

    #[test]
    fn merge_into_existing_terms_keep_their_id_new_ones_get_fresh_ids() {
        let (mut pb, _block, terms) = build_both(namespace_clustered_terms(200));
        let old_ids: Vec<TermId> = (0..pb.len()).map(|r| pb.id_at(r)).collect();

        let mut next_id = 100_000u32;
        let mut resolved_ids = Vec::new();
        let mut incoming: Vec<String> = terms.iter().step_by(7).cloned().collect(); // duplicates
        incoming.push("http://example.org/brand-new-1".to_string());
        incoming.push("http://example.org/brand-new-2".to_string());
        incoming.sort_unstable();
        incoming.dedup();

        let outcome = pb.merge_into(
            incoming.iter().map(String::as_str),
            || {
                let id = TermId(next_id);
                next_id += 1;
                id
            },
            |id| resolved_ids.push(id),
        );
        assert_eq!(outcome.added, 2);
        assert!(outcome.unchanged_prefix <= terms.len());
        assert_eq!(resolved_ids.len(), incoming.len());

        for (term, &old_id) in terms.iter().zip(&old_ids) {
            assert_eq!(pb.rank(term).map(|r| pb.id_at(r)), Some(old_id));
        }
        for term in [
            "http://example.org/brand-new-1",
            "http://example.org/brand-new-2",
        ] {
            let rank = pb.rank(term).expect("just inserted");
            assert!(pb.id_at(rank).0 >= 100_000);
        }
        assert_eq!(pb.len(), terms.len() + 2);
    }

    #[test]
    fn remove_drops_the_term_and_keeps_everything_else_correct() {
        let (mut pb, _block, terms) = build_both(namespace_clustered_terms(150));
        let target = terms[terms.len() / 2].clone();
        let target_rank = pb.rank(&target).unwrap();
        let target_id = pb.id_at(target_rank);

        let removed = pb.remove(&target);
        assert_eq!(removed, Some((target_id, target_rank)));
        assert_eq!(pb.rank(&target), None);
        assert_eq!(pb.len(), terms.len() - 1);
        assert_eq!(pb.remove(&target), None);

        for term in &terms {
            if term != &target {
                assert!(pb.rank(term).is_some(), "{term} should still be present");
            }
        }
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

    /// `automaton_block::tests::perf_probe_merge_into_single_term_automaton_vs_block`'s
    /// exact shape, `PackedAutomatonBlock` added as a third column --
    /// isolates `merge_into` itself (block-level only, no dictionary
    /// resync), decoupled from bootstrap/split noise, to directly
    /// attribute what `PackedAutomaton::stream` did and didn't close
    /// (DESIGN.md's dictionary-level insert-speed section).
    #[test]
    #[ignore]
    fn perf_probe_merge_into_single_term_packed_vs_automaton_vs_block() {
        let preload = namespace_clustered_terms(64); // ~256 terms, block-sized
        let (mut pb, block, _terms) = build_both(preload);

        let incoming = namespace_clustered_terms_range(1_000_000, 500);
        let mut next_id = 10_000_000u32;

        let start = std::time::Instant::now();
        for t in &incoming {
            pb.merge_into(
                std::iter::once(t.as_str()),
                || {
                    let id = TermId(next_id);
                    next_id += 1;
                    id
                },
                |_| {},
            );
        }
        let packed_elapsed = start.elapsed();

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
             PackedAutomatonBlock: {packed_elapsed:?} ({:?}/op)\n\
             Block:                {block_elapsed:?} ({:?}/op)",
            incoming.len(),
            packed_elapsed / incoming.len() as u32,
            block_elapsed / incoming.len() as u32,
        );
    }
}
