//! `PackedAutomatonDictionary`: `AutomatonDictionary`'s exact macro
//! structure (`automaton_dictionary.rs` -- lexical blocks, sorted-array
//! router, split/merge on overflow/underflow, `TermId -> (BlockId,
//! local_rank)` reverse mapping), `PackedAutomatonBlock`
//! (`packed_automaton_block.rs`) in place of `AutomatonBlock` -- the piece
//! that lets the byte-packed automaton be measured end to end, through
//! the same `TermDictionary` trait and the same `examples/
//! perf_scale_memory.rs` harness that first found `AutomatonDictionary`'s
//! own memory-scaling problem, instead of only in isolated block-level or
//! bare-automaton microbenchmarks.
//!
//! Deliberately smaller in scope than `AutomatonDictionary` itself
//! (already smaller in scope than `Dictionary` -- see its own module doc
//! comment): no `get_terms_into`/`get_terms_scattered_into`.
//! `insert_prepared` *is* implemented, so worker threads can sort/dedup
//! batches off the writer thread exactly as for `Dictionary`. `get_terms_seeked_into` *is* implemented
//! (`PackedAutomatonBlock::decode_sorted_ranks_into`, via
//! `PackedAutomaton::decode_ranks_into` -- ported from `Automaton`'s own
//! once `PackedAutomaton` grew the same recursive-descent batch-decode
//! primitive); the other two are still out of scope, either needing
//! `PackedAutomatonBlock` capabilities it doesn't have (a shared,
//! per-id-independent decode that isn't just "stream the whole touched
//! block") or genuinely orthogonal to the question this type exists to
//! answer first (does the byte-packed representation's measured
//! decode/insert-speed and memory findings hold up end to end, at real
//! scale, under real split/merge churn) -- porting them is future work
//! if this comparison itself proves the representation worth pursuing
//! further, not a prerequisite for running it.

use std::num::NonZeroU32;

use crate::batch::{BatchOrder, PreparedBatch};
use crate::entries_arena::EntriesArena;
use crate::packed_automaton_block::PackedAutomatonBlock;
use crate::slot_map::SlotMap;
use crate::{TermDictionary, TermId};

/// A stable handle for a block: an index into `PackedAutomatonDictionary`'s
/// block slab. Mirrors `AutomatonBlockId`/`dictionary::BlockId` exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedAutomatonBlockId(NonZeroU32);

impl PackedAutomatonBlockId {
    fn from_slot(slot: u32) -> PackedAutomatonBlockId {
        PackedAutomatonBlockId(
            NonZeroU32::new(slot + 1)
                .expect("slot + 1 does not overflow u32 for realistic block counts"),
        )
    }

    fn slot(self) -> u32 {
        self.0.get() - 1
    }
}

/// Larger than `Dictionary`'s 64..=256: incremental merges make bigger
/// blocks cheap here (an `fst` block is rebuilt on every change). On real
/// RDF, 256..=1024 vs 64..=256 measured -8% logical size, -11% bulk
/// insertion and -17% `get_id`. Single-term insert/remove no longer scale
/// with block size (~20 us each; DESIGN.md's "incremental single-term
/// mutation"). DESIGN.md's "block size and fill".
const DEFAULT_BLOCK_MIN: usize = 256;

/// Target fill of the chunks `split_while_oversized` produces, as a
/// percentage of `block_max`. 100% leaves bulk-loaded blocks full, so the
/// first single-term inserts afterwards split a third of the time (651 of
/// 2,000 measured; 134 us average insert). 90% removed those splits
/// (21.5 us) for +0.4% size. DESIGN.md's "incremental single-term
/// mutation".
const SPLIT_FILL_PERCENT: usize = 90;

/// Chain-label symbol table training sample: up to this many blocks,
/// evenly spaced, stopping early at this many label bytes.
const SYMBOL_SAMPLE_BLOCKS: usize = 64;
const SYMBOL_SAMPLE_BYTES: usize = 1 << 20;
const DEFAULT_BLOCK_MAX: usize = 1024;

/// See the module doc comment. Same macro structure as `AutomatonDictionary`,
/// `PackedAutomatonBlock` in place of `AutomatonBlock`.
#[derive(Debug)]
pub struct PackedAutomatonDictionary {
    blocks: Vec<Option<PackedAutomatonBlock>>,
    free_block_slots: Vec<u32>,
    directory: Vec<(Box<str>, PackedAutomatonBlockId)>,
    /// `TermId -> block slot` only; ranks come from the block itself
    /// (`PackedAutomatonBlock::rank_of_id`). See `SlotMap`.
    id_to_block: SlotMap,
    free_term_ids: Vec<u32>,
    next_term_id: u32,
    block_min: usize,
    block_max: usize,
    len: usize,
    scratch: EntriesArena,
    /// Chain-label symbol table (`crate::fsst`), trained at the first
    /// `shrink_to_fit` and shared by every block afterwards.
    symbols: Option<std::sync::Arc<crate::fsst::SymbolTable>>,
}

impl Default for PackedAutomatonDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl PackedAutomatonDictionary {
    pub fn new() -> Self {
        Self::with_block_bounds(DEFAULT_BLOCK_MIN, DEFAULT_BLOCK_MAX)
    }

    pub fn with_block_bounds(block_min: usize, block_max: usize) -> Self {
        assert!(block_min >= 1, "block_min must be at least 1");
        assert!(
            block_max >= 2 * block_min,
            "block_max must be at least twice block_min, so a split always \
             leaves both halves >= block_min"
        );
        assert!(
            block_max <= u16::MAX as usize,
            "block_max must fit in u16: local ranks in the reverse directory are stored as u16"
        );
        PackedAutomatonDictionary {
            blocks: Vec::new(),
            free_block_slots: Vec::new(),
            directory: Vec::new(),
            id_to_block: SlotMap::new(),
            free_term_ids: Vec::new(),
            next_term_id: 0,
            block_min,
            block_max,
            len: 0,
            scratch: EntriesArena::new(),
            symbols: None,
        }
    }

    /// `Dictionary::insert_prepared`'s contract: insert a worker-prepared
    /// batch without sorting or deduplicating it again. Returns IDs in
    /// original input order, including duplicate occurrences.
    pub fn insert_prepared(&mut self, prepared: PreparedBatch<'_>) -> Vec<TermId> {
        let batch = prepared.batch;
        self.insert_ordered(&batch.order, |u| batch.term(batch.order.first(u)))
    }

    /// Compacts every block (`PackedAutomaton::compact`: drops dead records
    /// and minimizes exactly), then trims dictionary-level spare capacity
    /// -- the reverse id table, directory, slot/id free lists, and the
    /// split/merge scratch arena -- so retained heap tracks `size_bytes`.
    /// Call after a bulk load; not done per batch, since trimming a
    /// still-growing table would turn amortized growth into a
    /// realloc-and-copy per batch, and per-merge compaction is only
    /// triggered by growth (`COMPACT_GROWTH_RATIO`). Per-block construction
    /// state is already released after every mutation
    /// (`PackedAutomaton::release_construction_state`).
    pub fn shrink_to_fit(&mut self) {
        if self.symbols.is_none() {
            // Train the chain-label table once, on the chains of up to
            // `SYMBOL_SAMPLE_BLOCKS` evenly spaced blocks (chained raw for
            // the purpose, then re-chained coded below).
            let live: Vec<usize> = (0..self.blocks.len()).filter(|&i| self.blocks[i].is_some()).collect();
            let step = (live.len() / SYMBOL_SAMPLE_BLOCKS).max(1);
            let mut labels: Vec<Vec<u8>> = Vec::new();
            let mut sampled = 0usize;
            for &i in live.iter().step_by(step) {
                let block = self.blocks[i].as_mut().expect("live slot");
                block.finish(None);
                let before = labels.len();
                block.chain_labels_into(&mut labels);
                sampled += labels[before..].iter().map(Vec::len).sum::<usize>();
                if sampled >= SYMBOL_SAMPLE_BYTES {
                    break;
                }
            }
            if !labels.is_empty() {
                let sample: Vec<&[u8]> = labels.iter().map(Vec::as_slice).collect();
                self.symbols = Some(std::sync::Arc::new(crate::fsst::SymbolTable::train(&sample)));
            }
        }
        for block in self.blocks.iter_mut().flatten() {
            block.finish(self.symbols.as_ref());
        }
        self.blocks.shrink_to_fit();
        self.free_block_slots.shrink_to_fit();
        self.directory.shrink_to_fit();
        self.id_to_block.shrink_to_fit();
        self.free_term_ids.shrink_to_fit();
        self.scratch = EntriesArena::new();
    }

    /// The number of live blocks, for introspection/testing.
    pub fn block_count(&self) -> usize {
        self.directory.len()
    }

    /// `AutomatonDictionary::size_bytes`'s exact accounting.
    pub fn size_bytes(&self) -> usize {
        let blocks_bytes: usize = self
            .blocks
            .iter()
            .filter_map(|b| b.as_ref())
            .map(|b| b.payload_bytes() + b.ids_bytes())
            .sum();
        let directory_bytes: usize = self
            .directory
            .iter()
            .map(|(key, _)| key.len() + std::mem::size_of::<PackedAutomatonBlockId>())
            .sum();
        let id_to_block_bytes = self.id_to_block.size_bytes();
        let symbols_bytes = self.symbols.as_ref().map_or(0, |t| t.size_bytes());
        blocks_bytes + directory_bytes + id_to_block_bytes + symbols_bytes
    }

    /// Total reachable state count across every block's `PackedAutomaton`
    /// -- `Dictionary::reachable_state_count`'s counterpart, for the
    /// states-per-term investigation (DESIGN.md).
    pub fn reachable_state_count(&self) -> usize {
        self.blocks
            .iter()
            .filter_map(|b| b.as_ref())
            .map(PackedAutomatonBlock::reachable_state_count)
            .sum()
    }

    /// `Dictionary::get_term_into`'s exact contract.
    pub fn get_term_into(&self, id: TermId, buf: &mut Vec<u8>) -> bool {
        buf.clear();
        let Some((block_id, local_rank)) = self.reverse(id)
        else {
            return false;
        };
        self.block(block_id).term_at_into(local_rank as usize, buf);
        true
    }

    /// `AutomatonDictionary::get_terms_seeked_into`'s exact contract and
    /// algorithm: groups ids by block, sorts each block's requested
    /// ranks ascending, then decodes them via one `PackedAutomatonBlock::
    /// decode_sorted_ranks_into` call per block -- sharing each visited
    /// state's cost across every requested rank that passes through it,
    /// instead of one independent root descent per id. `ids` need not be
    /// sorted or deduplicated; `out` is cleared first and filled in the
    /// same order as `ids`, `None` marking an id not present.
    pub fn get_terms_seeked_into(&self, ids: &[TermId], out: &mut Vec<Option<String>>) {
        out.clear();
        out.resize(ids.len(), None);

        let mut by_block: std::collections::HashMap<PackedAutomatonBlockId, Vec<(usize, u16)>> =
            std::collections::HashMap::new();
        for (pos, &id) in ids.iter().enumerate() {
            if let Some((block_id, local_rank)) =
                self.reverse(id)
            {
                by_block
                    .entry(block_id)
                    .or_default()
                    .push((pos, local_rank));
            }
        }

        for (block_id, mut positions) in by_block {
            positions.sort_unstable_by_key(|&(_, rank)| rank);
            self.block(block_id)
                .decode_sorted_ranks_into(&positions, out);
        }
    }

    fn block(&self, id: PackedAutomatonBlockId) -> &PackedAutomatonBlock {
        self.blocks[id.slot() as usize]
            .as_ref()
            .expect("PackedAutomatonBlockId must reference a live block")
    }

    fn block_mut(&mut self, id: PackedAutomatonBlockId) -> &mut PackedAutomatonBlock {
        self.blocks[id.slot() as usize]
            .as_mut()
            .expect("PackedAutomatonBlockId must reference a live block")
    }

    fn alloc_block(&mut self, block: PackedAutomatonBlock) -> PackedAutomatonBlockId {
        match self.free_block_slots.pop() {
            Some(slot) => {
                self.blocks[slot as usize] = Some(block);
                PackedAutomatonBlockId::from_slot(slot)
            }
            None => {
                let slot = self.blocks.len() as u32;
                self.blocks.push(Some(block));
                PackedAutomatonBlockId::from_slot(slot)
            }
        }
    }

    fn alloc_term_id(&mut self) -> TermId {
        match self.free_term_ids.pop() {
            Some(slot) => TermId(slot),
            None => {
                let id = TermId(self.next_term_id);
                self.next_term_id += 1;
                id
            }
        }
    }

    /// `(block, local rank)` for a live id: the slot from `id_to_block`,
    /// the rank from the block's own id inverse. Ranks of a settled
    /// dictionary are `< block_max <= u16::MAX`.
    #[inline]
    fn reverse(&self, id: TermId) -> Option<(PackedAutomatonBlockId, u16)> {
        let slot = self.id_to_block.get(id.0)?;
        let rank = self.blocks.get(slot as usize)?.as_ref()?.rank_of_id(id)?;
        Some((PackedAutomatonBlockId::from_slot(slot), rank as u16))
    }

    /// Points every id of `block_id` at it -- after the block was built
    /// from ids that lived elsewhere (split chunks, neighbor merges). Rank
    /// changes need nothing here: ranks live in the block.
    fn assign_slots(&mut self, block_id: PackedAutomatonBlockId) {
        let block = self.blocks[block_id.slot() as usize]
            .as_ref()
            .expect("PackedAutomatonBlockId must reference a live block");
        self.id_to_block
            .assign(block_id.slot(), block.ids().into_iter().map(|id| id.0));
    }

    fn refresh_directory_key(
        &mut self,
        dir_idx: usize,
        block_id: PackedAutomatonBlockId,
        unchanged_prefix: usize,
    ) {
        if unchanged_prefix == 0 {
            self.directory[dir_idx].0 = self.block(block_id).min_key().into();
        }
    }

    fn route(&self, term: &str) -> Option<usize> {
        if self.directory.is_empty() {
            return None;
        }
        let idx = self
            .directory
            .partition_point(|(min_key, _)| min_key.as_ref() <= term);
        Some(idx.saturating_sub(1))
    }

    /// Splits the block at `dir_idx`, if it exceeds `block_max`, into
    /// balanced chunks of at most `SPLIT_FILL_PERCENT` of `block_max` --
    /// `Dictionary::rebuild_batch_block`'s shape -- rather than by repeated
    /// median halving, which leaves blocks anywhere from half to completely
    /// full after a large batch run (DESIGN.md's "block size and fill").
    /// Each chunk holds at least `block_min` terms.
    fn split_while_oversized(&mut self, dir_idx: usize) {
        let block_id = self.directory[dir_idx].1;
        let total = self.block(block_id).len();
        if total <= self.block_max {
            return;
        }
        let target = (self.block_max * SPLIT_FILL_PERCENT / 100).max(2 * self.block_min);
        let chunks = total.div_ceil(target);
        let base = total / chunks;
        let extra = total % chunks;

        let mut arena = std::mem::take(&mut self.scratch);
        arena.clear();
        self.block(block_id).decode_into(&mut arena);
        let mut new_directory = Vec::with_capacity(chunks - 1);
        let mut start = 0;
        for chunk in 0..chunks {
            let end = start + base + usize::from(chunk < extra);
            let block = PackedAutomatonBlock::build(arena.range(start, end));
            // The first chunk keeps the original id and min key.
            let chunk_id = if chunk == 0 {
                *self.block_mut(block_id) = block;
                block_id
            } else {
                let min_key: Box<str> = block.min_key().into();
                let id = self.alloc_block(block);
                new_directory.push((min_key, id));
                id
            };
            self.assign_slots(chunk_id);
            start = end;
        }
        self.scratch = arena;
        // Shift the router's tail once, rather than once per chunk.
        self.directory.splice(dir_idx + 1..dir_idx + 1, new_directory);
    }


    fn remove_empty_block(&mut self, dir_idx: usize) {
        let (_, block_id) = self.directory.remove(dir_idx);
        self.blocks[block_id.slot() as usize] = None;
        self.free_block_slots.push(block_id.slot());
    }

    fn merge_with_neighbor(&mut self, dir_idx: usize) {
        if self.directory.len() < 2 {
            return;
        }
        let (keep_idx, absorb_idx) = if dir_idx + 1 < self.directory.len() {
            (dir_idx, dir_idx + 1)
        } else {
            (dir_idx - 1, dir_idx)
        };

        let keep_block_id = self.directory[keep_idx].1;
        let absorb_block_id = self.directory[absorb_idx].1;

        let mut arena = std::mem::take(&mut self.scratch);
        arena.clear();
        self.block(keep_block_id).decode_into(&mut arena);
        self.block(absorb_block_id).decode_into(&mut arena);

        let rebuilt = PackedAutomatonBlock::build(arena.iter());
        *self.block_mut(keep_block_id) = rebuilt;
        self.assign_slots(keep_block_id);

        self.scratch = arena;

        self.blocks[absorb_block_id.slot() as usize] = None;
        self.free_block_slots.push(absorb_block_id.slot());
        self.directory.remove(absorb_idx);

        self.split_while_oversized(keep_idx);
    }

    fn bootstrap<'a>(
        &mut self,
        order: &BatchOrder,
        term: &impl Fn(usize) -> &'a str,
        result: &mut [TermId],
    ) {
        let total = order.unique_len();
        let num_chunks = total.div_ceil(self.block_max).max(1);
        let base = total / num_chunks;
        let extra = total % num_chunks;

        let mut start = 0;
        let mut chunk_ids = Vec::with_capacity(self.block_max.min(total));
        for chunk_idx in 0..num_chunks {
            let size = base + if chunk_idx < extra { 1 } else { 0 };
            let end = start + size;

            chunk_ids.clear();
            for offset in 0..(end - start) {
                let id = self.alloc_term_id();
                chunk_ids.push(id);
                for &pos in order.positions(start + offset) {
                    result[pos] = id;
                }
            }

            let block =
                PackedAutomatonBlock::build((start..end).map(term).zip(chunk_ids.iter().copied()));
            let min_key: Box<str> = block.min_key().into();
            let block_id = self.alloc_block(block);
            self.id_to_block.assign(block_id.slot(), chunk_ids.iter().map(|id| id.0));
            self.directory.push((min_key, block_id));
            self.len += end - start;
            start = end;
        }
    }

    fn insert_ordered<'a>(
        &mut self,
        order: &BatchOrder,
        term: impl Fn(usize) -> &'a str,
    ) -> Vec<TermId> {
        let mut result = vec![TermId(0); order.len()];
        if order.unique_len() == 0 {
            return result;
        }
        if self.directory.is_empty() {
            self.bootstrap(order, &term, &mut result);
            return result;
        }

        let mut u = 0;
        while u < order.unique_len() {
            let dir_idx = self.route(term(u)).expect("directory is non-empty");
            let mut run_end = u + 1;
            if let Some((next_min, _)) = self.directory.get(dir_idx + 1) {
                while run_end < order.unique_len() && term(run_end) < next_min.as_ref() {
                    run_end += 1;
                }
            } else {
                run_end = order.unique_len();
            }
            let block_id = self.directory[dir_idx].1;

            let mut k = u;
            let mut added_ids: Vec<u32> = Vec::new();
            let free_term_ids = &mut self.free_term_ids;
            let next_term_id = &mut self.next_term_id;
            let block_ref = self.blocks[block_id.slot() as usize]
                .as_mut()
                .expect("PackedAutomatonBlockId must reference a live block");
            let outcome = block_ref.merge_into(
                (u..run_end).map(&term),
                || {
                    let id = match free_term_ids.pop() {
                        Some(slot) => TermId(slot),
                        None => {
                            let id = TermId(*next_term_id);
                            *next_term_id += 1;
                            id
                        }
                    };
                    added_ids.push(id.0);
                    id
                },
                |id| {
                    for &pos in order.positions(k) {
                        result[pos] = id;
                    }
                    k += 1;
                },
            );

            if outcome.added > 0 {
                self.len += outcome.added;
                self.refresh_directory_key(dir_idx, block_id, outcome.unchanged_prefix);
                // Only the new ids changed block; existing ones' ranks
                // shifted, which the block's own inverse already reflects.
                self.id_to_block.assign(block_id.slot(), added_ids.into_iter());
                if self.block(block_id).len() > self.block_max {
                    self.split_while_oversized(dir_idx);
                }
            }
            u = run_end;
        }
        result
    }
}

impl TermDictionary for PackedAutomatonDictionary {
    fn get_id(&self, term: &str) -> Option<TermId> {
        let dir_idx = self.route(term)?;
        let block_id = self.directory[dir_idx].1;
        let local_rank = self.block(block_id).rank(term)?;
        Some(self.block(block_id).id_at(local_rank))
    }

    fn get_term(&self, id: TermId) -> Option<String> {
        let (block_id, local_rank) = self.reverse(id)?;
        Some(self.block(block_id).term_at(local_rank as usize))
    }

    fn insert(&mut self, term: &str) -> TermId {
        if self.directory.is_empty() {
            let id = self.alloc_term_id();
            let block = PackedAutomatonBlock::build(std::iter::once((term, id)));
            let block_id = self.alloc_block(block);
            self.directory.push((term.into(), block_id));
            self.id_to_block.set(id.0, block_id.slot());
            self.len += 1;
            return id;
        }

        let dir_idx = self.route(term).expect("directory is non-empty");
        let block_id = self.directory[dir_idx].1;

        if let Some(local_rank) = self.block(block_id).rank(term) {
            return self.block(block_id).id_at(local_rank);
        }

        let id = self.alloc_term_id();
        let block_ref = self.block_mut(block_id);
        let outcome = block_ref.merge_into(std::iter::once(term), || id, |_| {});

        self.len += 1;
        self.refresh_directory_key(dir_idx, block_id, outcome.unchanged_prefix);
        self.id_to_block.set(id.0, block_id.slot());

        if self.block(block_id).len() > self.block_max {
            self.split_while_oversized(dir_idx);
        }
        id
    }

    fn insert_batch(&mut self, terms: &[&str]) -> Vec<TermId> {
        let mut order = BatchOrder::default();
        order.prepare(terms.len(), |i| terms[i]);
        self.insert_ordered(&order, |u| terms[order.first(u)])
    }

    fn remove(&mut self, term: &str) -> Option<TermId> {
        let dir_idx = self.route(term)?;
        let block_id = self.directory[dir_idx].1;
        let block_ref = self.block_mut(block_id);
        let (id, unchanged_prefix) = block_ref.remove(term)?;

        self.id_to_block.clear(id.0);
        self.free_term_ids.push(id.0);
        self.len -= 1;

        let remaining = self.block(block_id).len();
        if remaining == 0 {
            self.remove_empty_block(dir_idx);
        } else {
            self.refresh_directory_key(dir_idx, block_id, unchanged_prefix);
            if remaining < self.block_min {
                self.merge_with_neighbor(dir_idx);
            }
        }
        Some(id)
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_term_into_matches_get_term() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(2, 8);
        let id = dict.insert("http://example.org/a");
        let mut buf = Vec::new();
        assert!(dict.get_term_into(id, &mut buf));
        assert_eq!(std::str::from_utf8(&buf).unwrap(), "http://example.org/a");

        let missing = TermId(9999);
        buf.clear();
        assert!(!dict.get_term_into(missing, &mut buf));
        assert!(buf.is_empty());
    }

    #[test]
    fn get_term_into_matches_get_term_across_many_terms_and_splits() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(4, 16);
        let terms: Vec<String> = (0..500)
            .map(|i| format!("http://example.org/entity-{i:04}"))
            .collect();
        let ids: Vec<TermId> = terms.iter().map(|t| dict.insert(t)).collect();
        let mut buf = Vec::new();
        for (term, id) in terms.iter().zip(&ids) {
            assert!(dict.get_term_into(*id, &mut buf));
            assert_eq!(std::str::from_utf8(&buf).unwrap(), term);
            assert_eq!(dict.get_term(*id).as_deref(), Some(term.as_str()));
        }
    }

    #[test]
    fn get_terms_seeked_into_matches_individual_lookups_across_blocks() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(2, 8);
        let terms: Vec<String> = (0..100).map(|i| format!("term-{i:04}")).collect();
        let ids: Vec<TermId> = terms.iter().map(|t| dict.insert(t)).collect();
        assert!(dict.block_count() > 1, "should span multiple blocks");

        // Duplicates (including a run of 3 identical ids), out-of-rank-order
        // within a block, and an absent id mixed in.
        let mut query: Vec<TermId> = vec![
            ids[50], ids[3], ids[99], ids[50], ids[0], ids[50], ids[4], ids[3],
        ];
        query.push(TermId(999_999)); // absent

        let mut out = Vec::new();
        dict.get_terms_seeked_into(&query, &mut out);
        assert_eq!(out.len(), query.len());
        for (i, &id) in query.iter().enumerate() {
            assert_eq!(out[i], dict.get_term(id), "mismatch at query position {i}");
        }
    }

    #[test]
    fn get_terms_seeked_into_matches_consecutive_ranks_within_one_block() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(16, 64);
        let terms: Vec<String> = (0..1000)
            .flat_map(|i| {
                [
                    format!("http://example.org/entity-{i:05}"),
                    format!("http://xmlns.com/foaf/0.1/property{i}"),
                ]
            })
            .collect();
        let ids: Vec<TermId> = terms.iter().map(|t| dict.insert(t)).collect();

        // A genuinely consecutive-rank cluster: every id landing in the
        // same block as `ids[0]` and its lexical neighbors.
        let mut query: Vec<TermId> = ids.iter().take(20).copied().collect();
        query.reverse(); // out of rank order on purpose

        let mut out = Vec::new();
        dict.get_terms_seeked_into(&query, &mut out);
        for (i, &id) in query.iter().enumerate() {
            assert_eq!(out[i], dict.get_term(id), "mismatch at query position {i}");
        }
    }

    #[test]
    fn get_terms_seeked_into_empty_input() {
        let dict = PackedAutomatonDictionary::with_block_bounds(2, 8);
        let mut out = vec![Some("stale".to_string())];
        dict.get_terms_seeked_into(&[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn size_bytes_grows_with_content_and_shrinks_on_remove() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(4, 16);
        let empty = dict.size_bytes();
        let terms: Vec<String> = (0..200)
            .map(|i| format!("http://example.org/entity-{i:04}"))
            .collect();
        for t in &terms {
            dict.insert(t);
        }
        let full = dict.size_bytes();
        assert!(full > empty, "size_bytes should grow as terms are inserted");
        for t in &terms {
            dict.remove(t);
        }
        let drained = dict.size_bytes();
        assert!(
            drained < full,
            "size_bytes should shrink as terms are removed"
        );
    }

    /// A later batch landing far more than `2 * block_max` terms in one
    /// block: ranks must be assigned only after splitting (the packed
    /// reverse map has no room for them before).
    #[test]
    fn oversized_batch_run_is_split_before_ranks_are_assigned() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(4, 16);
        let first: Vec<String> = (0..40).map(|i| format!("k{:05}", i * 100)).collect();
        let first_refs: Vec<&str> = first.iter().map(String::as_str).collect();
        dict.insert_batch(&first_refs);
        let second: Vec<String> = (0..4000).map(|i| format!("k{i:05}x")).collect();
        let second_refs: Vec<&str> = second.iter().map(String::as_str).collect();
        let ids = dict.insert_batch(&second_refs);
        for (t, id) in second_refs.iter().zip(&ids) {
            assert_eq!(dict.get_id(t), Some(*id));
            assert_eq!(dict.get_term(*id).as_deref(), Some(*t));
        }
        for t in &first_refs {
            let id = dict.get_id(t).expect("first batch term present");
            assert_eq!(dict.get_term(id).as_deref(), Some(*t));
        }
    }

    /// Mixed single inserts, small and large batches (both `merge_into`
    /// paths), and removes (incremental, with neighbor merges at these
    /// tiny bounds), checked against a model after every operation batch.
    #[test]
    fn mixed_mutations_match_model() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(8, 32);
        let mut model: std::collections::BTreeMap<String, TermId> = Default::default();
        let mut state = 0x2545_F491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for round in 0..400 {
            let op = next() % 10;
            if op < 4 {
                let t = format!("t{:04}", next() % 3000);
                let id = dict.insert(&t);
                assert_eq!(*model.entry(t).or_insert(id), id);
            } else if op < 7 {
                let n = if op == 6 { 300 } else { 1 + next() as usize % 6 };
                let batch: Vec<String> = (0..n).map(|_| format!("t{:04}", next() % 3000)).collect();
                let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
                let ids = dict.insert_batch(&refs);
                for (t, id) in batch.iter().zip(ids) {
                    assert_eq!(*model.entry(t.clone()).or_insert(id), id);
                }
            } else {
                let t = format!("t{:04}", next() % 3000);
                assert_eq!(dict.remove(&t), model.remove(&t), "round {round} remove {t}");
            }
            if round % 50 == 25 {
                // Chains every block; later mutations must expand them.
                dict.shrink_to_fit();
            }
            if round % 20 == 0 || round == 399 {
                assert_eq!(dict.len(), model.len());
                for (t, &id) in &model {
                    assert_eq!(dict.get_id(t), Some(id), "round {round} get_id {t}");
                    assert_eq!(dict.get_term(id).as_deref(), Some(t.as_str()));
                }
            }
        }
    }

    #[test]
    fn insert_prepared_matches_insert_batch() {
        let terms: Vec<String> = (0..2000).map(|i| format!("http://example.org/{}", i % 1500)).collect();
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let mut batch = crate::TermBatch::new();
        for t in &refs {
            batch.push(t);
        }
        batch.prepare();
        let mut prepared = PackedAutomatonDictionary::with_block_bounds(8, 32);
        let ids = prepared.insert_prepared(batch.prepared().expect("prepared"));
        let mut plain = PackedAutomatonDictionary::with_block_bounds(8, 32);
        assert_eq!(ids, plain.insert_batch(&refs));
        for (t, id) in refs.iter().zip(&ids) {
            assert_eq!(prepared.get_id(t), Some(*id));
        }
    }

    #[test]
    fn insert_batch_matches_sequential_insert() {
        let mut batch_dict = PackedAutomatonDictionary::with_block_bounds(4, 16);
        let mut seq_dict = PackedAutomatonDictionary::with_block_bounds(4, 16);
        let terms: Vec<String> = (0..300)
            .map(|i| format!("http://example.org/entity-{i:04}"))
            .collect();
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();

        batch_dict.insert_batch(&refs);
        for t in &refs {
            seq_dict.insert(t);
        }

        assert_eq!(batch_dict.len(), seq_dict.len());
        for t in &terms {
            let a = batch_dict.get_id(t);
            let b = seq_dict.get_id(t);
            assert!(a.is_some() && b.is_some());
            assert_eq!(
                batch_dict.get_term(a.unwrap()),
                seq_dict.get_term(b.unwrap())
            );
        }
    }
}
