//! `PackedAutomatonBlock`: one lexical block of `PackedAutomatonDictionary`
//! -- a [`PackedAutomaton`] over the block's sorted terms (term <-> local
//! rank) beside a [`BlockIds`] (local rank <-> `TermId`).
//!
//! Mutations are incremental: a merge walks the block's existing terms
//! (streamed, or one `lower_bound` descent per incoming term for small
//! runs) to assign ids, then inserts only the new terms
//! (`PackedAutomaton::insert_sorted`); a removal rewrites only the removed
//! term's path. Dead records left behind are reclaimed by a compaction
//! once a block's bytes per term double (`COMPACT_GROWTH_RATIO`).
//! Construction state and scratch buffers live in a caller-owned
//! [`Workspace`], shared by every block of a dictionary.

use crate::TermId;
use crate::block_ids::BlockIds;
use crate::entries_arena::EntriesArena;
use crate::packed_automaton::{self, PackedAutomaton, decoded_term};

/// `merge_into` compacts a block (`PackedAutomaton::compact`) once its
/// bytes per term exceed this multiple of the value right after its last
/// fresh build or compaction. Dead records accumulate with every merge;
/// this bounds them during a load, cheaply, while
/// `PackedAutomatonDictionary::shrink_to_fit` compacts every block at the
/// end. See DESIGN.md's "copy-compaction" for the measured tradeoff.
const COMPACT_GROWTH_RATIO: f32 = 2.0;

/// `merge_into` switches from streaming the whole block to one
/// `lower_bound` descent per incoming term when the run is at most
/// `1 / SMALL_RUN_FACTOR` of the block. See DESIGN.md's "incremental
/// single-term mutation" for the measurement behind it.
const SMALL_RUN_FACTOR: usize = 2;

/// Scratch shared by every block of one dictionary (one writer): the
/// automaton's construction workspace, id buffers for re-encoding a
/// block's `BlockIds`, and the decoded-entries arena splits and neighbor
/// merges rebuild from.
#[derive(Debug, Default)]
pub(crate) struct Workspace {
    pub(crate) automaton: packed_automaton::Workspace,
    pub(crate) entries: EntriesArena,
    pub(crate) ids: Vec<TermId>,
    old_ids: Vec<TermId>,
    new_ids: Vec<TermId>,
    /// Indices (into a merge's incoming run) of the terms not yet present.
    new_terms: Vec<u32>,
    /// `PackedAutomaton::lower_bounds_sorted`'s descent stack.
    descent: Vec<(u32, u32, u32)>,
}

#[derive(Debug)]
pub(crate) struct PackedAutomatonBlock {
    packed: PackedAutomaton,
    /// Bytes per term right after the last fresh build or compaction --
    /// the baseline `merge_into` compares growth against.
    compact_bytes_per_term: f32,
    /// `rank -> TermId`, compressed.
    ids: BlockIds,
}

/// What a `merge_into` changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MergeOutcome {
    /// Terms that were not already present.
    pub(crate) added: usize,
    /// Leading ranks unchanged by the merge; 0 means the block's smallest
    /// term changed.
    pub(crate) unchanged_prefix: usize,
}

fn bytes_per_term(packed: &PackedAutomaton, terms: usize) -> f32 {
    packed.size_bytes() as f32 / terms.max(1) as f32
}

impl PackedAutomatonBlock {
    /// Builds a block from sorted, deduplicated `terms` and their `ids`
    /// (same length, same order).
    pub(crate) fn build<'a>(
        terms: impl ExactSizeIterator<Item = &'a [u8]>,
        ids: &[TermId],
        ws: &mut packed_automaton::Workspace,
    ) -> PackedAutomatonBlock {
        debug_assert_eq!(terms.len(), ids.len());
        let mut packed = PackedAutomaton::build_sorted(terms, ws);
        packed.shrink_to_fit();
        PackedAutomatonBlock {
            compact_bytes_per_term: bytes_per_term(&packed, ids.len()),
            packed,
            ids: BlockIds::new(ids),
        }
    }

    /// End of a load: compaction plus path compression for faster reads
    /// (`PackedAutomaton::compact_and_chain`), coding chain labels with
    /// `symbols` when given.
    pub(crate) fn finish(&mut self, symbols: Option<&std::sync::Arc<crate::fsst::SymbolTable>>, ws: &mut Workspace) {
        if let Some(table) = symbols {
            self.packed.set_symbols(Some(table.clone()));
        }
        self.packed.compact_and_chain(&mut ws.automaton);
        self.compact_bytes_per_term = bytes_per_term(&self.packed, self.ids.len());
    }

    /// Every chain label of this block (decoded): a training sample.
    pub(crate) fn chain_labels_into(&self, out: &mut Vec<Vec<u8>>) {
        self.packed.chain_labels_into(out);
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn rank(&self, term: &str) -> Option<usize> {
        self.packed.rank(term).map(|r| r as usize)
    }

    pub(crate) fn id_at(&self, local_rank: usize) -> TermId {
        self.ids.get(local_rank)
    }

    /// The local rank of `id`, if this block holds it.
    pub(crate) fn rank_of_id(&self, id: TermId) -> Option<usize> {
        self.ids.rank_of(id)
    }

    pub(crate) fn term_at_into(&self, local_rank: usize, buf: &mut Vec<u8>) {
        self.packed.term_at_into(local_rank as u32, buf);
    }

    pub(crate) fn term_at(&self, local_rank: usize) -> String {
        decoded_term(self.term_bytes(local_rank))
    }

    pub(crate) fn term_bytes(&self, local_rank: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        self.term_at_into(local_rank, &mut buf);
        buf
    }

    /// Gives this block's terms the ids `base..base + len()`, in rank order.
    pub(crate) fn set_id_run(&mut self, base: u32) {
        self.ids = BlockIds::run(self.len(), base);
    }

    /// Gives this block's terms the ids `ids` (indexed by local rank).
    pub(crate) fn set_ids(&mut self, ids: &[TermId]) {
        debug_assert_eq!(ids.len(), self.len(), "one id per term");
        self.ids = BlockIds::new(ids);
    }

    /// Appends every id in rank order to `out`.
    pub(crate) fn ids_into(&self, out: &mut Vec<TermId>) {
        self.ids.decode_into(out);
    }

    /// Logical size of the automaton's records (including dead records
    /// not yet compacted away).
    pub(crate) fn payload_bytes(&self) -> usize {
        self.packed.size_bytes()
    }

    /// Logical size of the compressed `rank -> TermId` mapping.
    pub(crate) fn ids_bytes(&self) -> usize {
        self.ids.size_bytes()
    }

    /// See `PackedAutomaton::reachable_state_count`.
    pub(crate) fn reachable_state_count(&self) -> usize {
        self.packed.reachable_state_count()
    }

    /// Appends every `(term, id)` pair in sorted order onto `out`, via one
    /// incremental walk (`PackedAutomaton::stream`); `ids` is scratch.
    pub(crate) fn decode_into(&self, out: &mut EntriesArena, ids: &mut Vec<TermId>) {
        ids.clear();
        self.ids.decode_into(ids);
        let mut stream = self.packed.stream();
        let mut rank = 0usize;
        while let Some(term) = stream.next() {
            out.push(term, ids[rank]);
            rank += 1;
        }
    }

    /// Decodes `targets` (`(output position, local rank)`, sorted by rank)
    /// into `out[position]`, via `PackedAutomaton::decode_ranks_into`: one
    /// descent visiting each state on some requested path once.
    pub(crate) fn decode_sorted_ranks_into(&self, targets: &[(usize, u32)], out: &mut [Option<String>]) {
        self.packed.decode_ranks_into(targets, out);
    }

    /// Merges `incoming` (sorted, deduplicated) into this block: existing
    /// terms keep their id, new ones get one from `alloc_id`; `resolved` is
    /// called once per incoming term, in order, with its final id. Ids are
    /// assigned by walking the existing terms (a stream, or a `lower_bound`
    /// per term for small runs); then only the new terms are inserted.
    pub(crate) fn merge_into(
        &mut self,
        incoming: &[&[u8]],
        alloc_id: impl FnMut() -> TermId,
        resolved: impl FnMut(TermId),
        ws: &mut Workspace,
    ) -> MergeOutcome {
        debug_assert!(incoming.windows(2).all(|w| w[0] < w[1]), "incoming must be sorted and distinct");
        if incoming.is_empty() {
            return MergeOutcome {
                added: 0,
                unchanged_prefix: self.ids.len(),
            };
        }
        ws.old_ids.clear();
        self.ids.decode_into(&mut ws.old_ids);
        ws.new_ids.clear();
        ws.new_terms.clear();
        let first_new = if incoming.len() * SMALL_RUN_FACTOR <= self.ids.len() {
            self.assign_by_descent(incoming, alloc_id, resolved, ws)
        } else {
            self.assign_by_stream(incoming, alloc_id, resolved, ws)
        };
        let added = ws.new_terms.len();
        if added > 0 {
            let terms = ws.new_terms.iter().map(|&i| incoming[i as usize]);
            self.packed.insert_sorted(terms, &mut ws.automaton);
            self.ids = BlockIds::new(&ws.new_ids);
            self.settle(ws);
        }
        MergeOutcome {
            added,
            unchanged_prefix: first_new.unwrap_or(self.ids.len()),
        }
    }

    /// `merge_into`'s id assignment for a run much smaller than the block:
    /// descents (sharing the run's common prefixes) find each incoming
    /// term's old rank, instead of streaming every term in the block -- the single-term
    /// insert and typical bulk-load case (a batch spread over thousands of
    /// blocks lands a few terms in each). Fills `new_ids`/`new_terms`;
    /// returns the first new term's rank, if any.
    fn assign_by_descent(
        &self,
        incoming: &[&[u8]],
        mut alloc_id: impl FnMut() -> TermId,
        mut resolved: impl FnMut(TermId),
        ws: &mut Workspace,
    ) -> Option<usize> {
        let (mut first_new, mut copied, mut i) = (None, 0usize, 0u32);
        let Workspace {
            old_ids,
            new_ids,
            new_terms,
            descent,
            ..
        } = ws;
        self.packed.lower_bounds_sorted(incoming.iter().copied(), descent, |rank, found| {
            let rank = rank as usize;
            new_ids.extend_from_slice(&old_ids[copied..rank]);
            copied = rank;
            if found {
                // Copied with the rest of the old ids at its own position.
                resolved(old_ids[rank]);
            } else {
                first_new.get_or_insert(new_ids.len());
                let id = alloc_id();
                new_ids.push(id);
                new_terms.push(i);
                resolved(id);
            }
            i += 1;
        });
        new_ids.extend_from_slice(&old_ids[copied..]);
        first_new
    }

    /// `merge_into`'s id assignment for a large run: one sorted-merge walk
    /// of the block's terms (`PackedAutomaton::stream`) against `incoming`.
    fn assign_by_stream(
        &self,
        incoming: &[&[u8]],
        mut alloc_id: impl FnMut() -> TermId,
        mut resolved: impl FnMut(TermId),
        ws: &mut Workspace,
    ) -> Option<usize> {
        let mut first_new = None;
        let mut next = 0usize;
        let mut stream = self.packed.stream();
        let mut old_rank = 0usize;
        loop {
            let old_term = stream.next();
            // New terms sorting before `old_term` (all remaining ones once
            // the block is exhausted).
            while let Some(&term) = incoming.get(next)
                && old_term.is_none_or(|old| term < old)
            {
                first_new.get_or_insert(ws.new_ids.len());
                let id = alloc_id();
                ws.new_ids.push(id);
                ws.new_terms.push(next as u32);
                resolved(id);
                next += 1;
            }
            let Some(old_term) = old_term else { break };
            let id = ws.old_ids[old_rank];
            ws.new_ids.push(id);
            if incoming.get(next).is_some_and(|&t| t == old_term) {
                resolved(id);
                next += 1;
            }
            old_rank += 1;
        }
        first_new
    }

    /// Removes `term` if present, returning its former id and rank (the
    /// count of leading entries left unchanged). Incremental: only the
    /// term's path is rewritten (`PackedAutomaton::remove`).
    pub(crate) fn remove(&mut self, term: &str, ws: &mut Workspace) -> Option<(TermId, usize)> {
        let rank = self.packed.remove(term, &mut ws.automaton)? as usize;
        ws.new_ids.clear();
        self.ids.decode_into(&mut ws.new_ids);
        let removed_id = ws.new_ids.remove(rank);
        self.ids = BlockIds::new(&ws.new_ids);
        self.settle(ws);
        Some((removed_id, rank))
    }

    /// After a mutation: compact once bytes per term exceed
    /// `COMPACT_GROWTH_RATIO` times the value right after the last fresh
    /// build or compaction; otherwise trim the records' growth capacity.
    fn settle(&mut self, ws: &mut Workspace) {
        if bytes_per_term(&self.packed, self.ids.len()) > COMPACT_GROWTH_RATIO * self.compact_bytes_per_term {
            self.packed.compact(&mut ws.automaton);
            self.compact_bytes_per_term = bytes_per_term(&self.packed, self.ids.len());
        } else {
            self.packed.shrink_to_fit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn namespace_clustered_terms(start: u32, count: u32) -> Vec<String> {
        let mut terms: Vec<String> = (start..start + count)
            .flat_map(|i| {
                [
                    format!("http://dbpedia.org/resource/Entity_{i}"),
                    format!("http://xmlns.com/foaf/0.1/property{i}"),
                    format!("urn:isbn:{i:013}"),
                    format!("_:b{i:x}"),
                ]
            })
            .collect();
        terms.sort_unstable();
        terms.dedup();
        terms
    }

    /// A block over `terms` with ids `0..len`.
    fn build(terms: &[String], ws: &mut Workspace) -> PackedAutomatonBlock {
        let ids: Vec<TermId> = (0..terms.len() as u32).map(TermId).collect();
        PackedAutomatonBlock::build(terms.iter().map(String::as_bytes), &ids, &mut ws.automaton)
    }

    #[test]
    fn build_rank_unrank_and_decode() {
        let mut ws = Workspace::default();
        let terms = namespace_clustered_terms(0, 300);
        let block = build(&terms, &mut ws);
        assert_eq!(block.len(), terms.len());
        for (rank, term) in terms.iter().enumerate() {
            assert_eq!(block.rank(term), Some(rank));
            assert_eq!(block.id_at(rank), TermId(rank as u32));
            assert_eq!(block.rank_of_id(TermId(rank as u32)), Some(rank));
            assert_eq!(block.term_at(rank), *term);
        }
        assert_eq!(block.rank("not-present-at-all"), None);
        let (mut arena, mut ids) = (EntriesArena::new(), Vec::new());
        block.decode_into(&mut arena, &mut ids);
        assert_eq!(arena.len(), terms.len());
        for (i, term) in terms.iter().enumerate() {
            assert_eq!(arena.term_at(i), term.as_bytes());
            assert_eq!(arena.id_at(i), TermId(i as u32));
        }
    }

    /// Both id-assignment paths (descent for small runs, stream for large
    /// ones): existing terms keep their id, new ones get fresh ids, and
    /// `resolved` reports every incoming term in order.
    #[test]
    fn merge_into_keeps_existing_ids_on_both_paths() {
        let terms = namespace_clustered_terms(0, 200);
        for step in [97, 2] {
            let mut ws = Workspace::default();
            let mut block = build(&terms, &mut ws);
            let mut incoming: Vec<String> = terms.iter().step_by(step).cloned().collect();
            incoming.extend(namespace_clustered_terms(10_000, 200 / step as u32 + 1));
            incoming.push(String::new()); // sorts first: a new smallest term
            incoming.push("~".repeat(3)); // sorts last: past every old term
            incoming.sort_unstable();
            incoming.dedup();
            let refs: Vec<&[u8]> = incoming.iter().map(String::as_bytes).collect();
            let mut next_id = 100_000u32;
            let mut resolved = Vec::new();
            let outcome = block.merge_into(
                &refs,
                || {
                    next_id += 1;
                    TermId(next_id)
                },
                |id| resolved.push(id),
                &mut ws,
            );
            assert_eq!(outcome.unchanged_prefix, 0, "the empty term is the new minimum");
            assert_eq!(resolved.len(), incoming.len());
            let new = incoming.iter().filter(|t| terms.binary_search(t).is_err()).count();
            assert_eq!(outcome.added, new);
            assert_eq!(block.len(), terms.len() + new);
            for (term, &id) in incoming.iter().zip(&resolved) {
                let rank = block.rank(term).expect("merged");
                assert_eq!(block.id_at(rank), id);
                match terms.binary_search(term) {
                    Ok(old) => assert_eq!(id, TermId(old as u32)),
                    Err(_) => assert!(id.0 > 100_000),
                }
            }
        }
    }

    #[test]
    fn remove_drops_the_term_and_keeps_everything_else() {
        let mut ws = Workspace::default();
        let terms = namespace_clustered_terms(0, 150);
        let mut block = build(&terms, &mut ws);
        let target = &terms[terms.len() / 2];
        let rank = block.rank(target).unwrap();
        assert_eq!(block.remove(target, &mut ws), Some((TermId(rank as u32), rank)));
        assert_eq!(block.rank(target), None);
        assert_eq!(block.remove(target, &mut ws), None);
        assert_eq!(block.len(), terms.len() - 1);
        for (old_rank, term) in terms.iter().enumerate().filter(|(_, t)| *t != target) {
            let rank = block.rank(term).expect("still present");
            assert_eq!(block.id_at(rank), TermId(old_rank as u32));
        }
    }
}
