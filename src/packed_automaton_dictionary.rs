//! `PackedAutomatonDictionary`: the crate's primary dictionary. Terms are
//! partitioned into contiguous lexical blocks (`PackedAutomatonBlock`,
//! each a `PackedAutomaton` beside its compressed ids); a sorted directory
//! of block minimum keys routes `term -> block`, and a range map
//! (`RangeMap`) routes `TermId -> block`, the block itself answering `id ->
//! local rank`. Blocks split above `block_max` terms and merge with a
//! neighbor below `block_min`, so a mutation touches one or two blocks.

use std::sync::Arc;

use crate::batch::{BatchOrder, PreparedBatch};
use crate::entries_arena::EntriesArena;
use crate::fsst::SymbolTable;
use crate::limits::{BLOCK_TEXT_LIMIT, MAX_TERM_BYTES};
use crate::packed_automaton_block::{PackedAutomatonBlock, Workspace};
use crate::range_map::RangeMap;
use crate::{DictError, TermDictionary, TermId};

/// Larger than an `fst` block's 64..=256: incremental merges make bigger
/// blocks cheap here. On real RDF, 256..=1024 vs 64..=256 measured -8%
/// logical size, -11% bulk insertion and -17% `get_id` (DESIGN.md's
/// "block size and fill").
const DEFAULT_BLOCK_MIN: usize = 256;
const DEFAULT_BLOCK_MAX: usize = 1024;

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

/// The default id limit: ids are below `u32::MAX`, so `id + 1` (an id
/// range's end) always fits.
const DEFAULT_ID_LIMIT: u32 = u32::MAX;

#[derive(Debug)]
pub struct PackedAutomatonDictionary {
    blocks: Vec<Option<PackedAutomatonBlock>>,
    free_block_slots: Vec<u32>,
    /// `(minimum key, block)`, sorted by key: the router. Keys are term
    /// bytes (byte order is `str` order).
    directory: Vec<(Box<[u8]>, u32)>,
    /// `TermId -> block slot` as ranges of consecutive ids; ranks come
    /// from the block itself (`PackedAutomatonBlock::rank_of_id`).
    id_to_block: RangeMap,
    free_term_ids: Vec<u32>,
    next_term_id: u32,
    /// Every id handed out is below this (`set_id_limit`).
    id_limit: u32,
    block_min: usize,
    block_max: usize,
    len: usize,
    /// Construction state and scratch shared by every block (one writer).
    workspace: Workspace,
    /// Ids of one merge or rebuilt block, for `id_to_block`.
    slot_ids: Vec<u32>,
    /// Chain-label symbol table (`crate::fsst`), trained at the first
    /// `shrink_to_fit` and shared by every block afterwards.
    symbols: Option<Arc<SymbolTable>>,
}

impl Default for PackedAutomatonDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// Hands out term ids: freed ones first, then fresh ones below
/// `id_limit`. Callers check `ids_available` before mutating anything
/// (`check_new_terms`); the assert only guards that premise, rather than
/// wrapping into live ids.
fn alloc_term_id(free_term_ids: &mut Vec<u32>, next_term_id: &mut u32, id_limit: u32) -> TermId {
    if let Some(id) = free_term_ids.pop() {
        return TermId(id);
    }
    let id = *next_term_id;
    assert!(id < id_limit, "id availability is checked before mutating");
    *next_term_id = id + 1;
    TermId(id)
}

/// Ends of the chunks a rebuild splits `total` sorted entries into:
/// `chunks` balanced ones by count, each further cut so it holds at most
/// `BLOCK_TEXT_LIMIT` bytes of text (or a single term), which keeps every
/// fresh build addressable (`crate::limits`). `len_at(i)`: entry `i`'s
/// term length.
fn chunk_ends(total: usize, chunks: usize, len_at: impl Fn(usize) -> usize) -> Vec<usize> {
    let chunks = chunks.clamp(1, total.max(1));
    let (base, extra) = (total / chunks, total % chunks);
    let mut ends = Vec::with_capacity(chunks);
    let mut start = 0;
    for chunk in 0..chunks {
        let end = start + base + usize::from(chunk < extra);
        let mut text = 0u64;
        for i in start..end {
            let len = len_at(i) as u64;
            if text > 0 && text + len > BLOCK_TEXT_LIMIT {
                ends.push(i);
                text = 0;
            }
            text += len;
        }
        ends.push(end);
        start = end;
    }
    ends
}

impl PackedAutomatonDictionary {
    pub fn new() -> Self {
        Self::with_block_bounds(DEFAULT_BLOCK_MIN, DEFAULT_BLOCK_MAX)
    }

    pub fn with_block_bounds(block_min: usize, block_max: usize) -> Self {
        assert!(block_min >= 1, "block_min must be at least 1");
        assert!(
            block_max >= 2 * block_min,
            "block_max must be at least twice block_min, so a split always leaves both halves >= block_min"
        );
        assert!(block_max <= u16::MAX as usize, "block_max must fit in u16: local ranks are u16");
        PackedAutomatonDictionary {
            blocks: Vec::new(),
            free_block_slots: Vec::new(),
            directory: Vec::new(),
            id_to_block: RangeMap::new(),
            free_term_ids: Vec::new(),
            next_term_id: 0,
            id_limit: DEFAULT_ID_LIMIT,
            block_min,
            block_max,
            len: 0,
            workspace: Workspace::default(),
            slot_ids: Vec::new(),
            symbols: None,
        }
    }

    /// Inserts a batch prepared off the writer thread (`TermBatch::prepare`
    /// sorted and grouped it). Returns ids in original input order,
    /// including duplicate occurrences. Panics where
    /// [`Self::try_insert_prepared`] returns an error.
    pub fn insert_prepared(&mut self, batch: &PreparedBatch) -> Vec<TermId> {
        self.try_insert_prepared(batch).unwrap_or_else(|e| panic!("{e}"))
    }

    /// [`Self::insert_prepared`], or an error (nothing inserted) if a term
    /// exceeds the term size limit or the batch needs more new ids than
    /// remain below the id limit.
    pub fn try_insert_prepared(&mut self, batch: &PreparedBatch) -> Result<Vec<TermId>, DictError> {
        let order = batch.order();
        self.try_insert_ordered(order, |u| batch.term(order.first(u)))
    }

    /// Ids still available to new terms: freed ids plus those below the id
    /// limit not yet handed out.
    pub fn ids_available(&self) -> u64 {
        self.free_term_ids.len() as u64 + u64::from(self.id_limit - self.next_term_id)
    }

    /// Checks, without changing anything, that inserting the distinct
    /// `uniques` would succeed: every term within the size limit, and ids
    /// for the new ones. Counts the new terms exactly only when the
    /// distinct count alone could exceed the ids available.
    fn check_new_terms(&self, uniques: &[&[u8]]) -> Result<(), DictError> {
        if let Some(len) = uniques.iter().map(|t| t.len()).find(|&len| len > MAX_TERM_BYTES) {
            return Err(DictError::TermTooLarge { len, max: MAX_TERM_BYTES });
        }
        let available = self.ids_available();
        if uniques.len() as u64 <= available {
            return Ok(());
        }
        let needed = uniques.iter().filter(|t| !self.contains_bytes(t)).count() as u64;
        if needed > available {
            return Err(DictError::IdsExhausted { needed, available });
        }
        Ok(())
    }

    fn contains_bytes(&self, term: &[u8]) -> bool {
        self.route(term).is_some_and(|dir_idx| self.block(self.directory[dir_idx].1).contains_bytes(term))
    }

    /// End of a load: compacts every block to its exact minimal automaton
    /// and path-compresses it (training the chain-label symbol table on
    /// the first call), then trims spare capacity and drops the
    /// construction workspace, so retained heap tracks `size_bytes`. Not
    /// done per batch: trimming still-growing tables would turn amortized
    /// growth into a copy per batch.
    pub fn shrink_to_fit(&mut self) {
        if self.symbols.is_none() {
            self.symbols = self.train_symbols();
        }
        for block in self.blocks.iter_mut().flatten() {
            block.finish(self.symbols.as_ref(), &mut self.workspace);
        }
        self.blocks.shrink_to_fit();
        self.free_block_slots.shrink_to_fit();
        self.directory.shrink_to_fit();
        self.id_to_block.shrink_to_fit();
        self.free_term_ids.shrink_to_fit();
        self.workspace = Workspace::default();
        self.slot_ids = Vec::new();
    }

    /// Trains the chain-label table on the chains of up to
    /// `SYMBOL_SAMPLE_BLOCKS` evenly spaced blocks (chained raw for the
    /// purpose, then re-chained coded by `shrink_to_fit`).
    fn train_symbols(&mut self) -> Option<Arc<SymbolTable>> {
        let live: Vec<usize> = (0..self.blocks.len()).filter(|&i| self.blocks[i].is_some()).collect();
        let step = (live.len() / SYMBOL_SAMPLE_BLOCKS).max(1);
        let mut labels: Vec<Vec<u8>> = Vec::new();
        let mut sampled = 0usize;
        for &i in live.iter().step_by(step) {
            let block = self.blocks[i].as_mut().expect("live slot");
            block.finish(None, &mut self.workspace);
            let before = labels.len();
            block.chain_labels_into(&mut labels);
            sampled += labels[before..].iter().map(Vec::len).sum::<usize>();
            if sampled >= SYMBOL_SAMPLE_BYTES {
                break;
            }
        }
        if labels.is_empty() {
            return None;
        }
        let sample: Vec<&[u8]> = labels.iter().map(Vec::as_slice).collect();
        Some(Arc::new(SymbolTable::train(&sample)))
    }

    /// The number of live blocks, for introspection/testing.
    pub fn block_count(&self) -> usize {
        self.directory.len()
    }

    /// Logical size: block records and ids, directory keys and handles,
    /// the id -> block map and the symbol table.
    pub fn size_bytes(&self) -> usize {
        let blocks_bytes: usize = self.blocks.iter().flatten().map(|b| b.payload_bytes() + b.ids_bytes()).sum();
        let directory_bytes: usize = self
            .directory
            .iter()
            .map(|(key, _)| key.len() + std::mem::size_of::<u32>())
            .sum();
        let symbols_bytes = self.symbols.as_ref().map_or(0, |t| t.size_bytes());
        blocks_bytes + directory_bytes + self.id_to_block.size_bytes() + symbols_bytes
    }

    /// `size_bytes` by component, as `(name, bytes)`: the block automata,
    /// the blocks' `rank <-> TermId` permutations, the directory, the
    /// `TermId -> block` map and the chain-label symbol table.
    pub fn size_breakdown(&self) -> Vec<(&'static str, usize)> {
        let automata = self.blocks.iter().flatten().map(PackedAutomatonBlock::payload_bytes).sum();
        let ids = self.blocks.iter().flatten().map(PackedAutomatonBlock::ids_bytes).sum();
        let directory = self.directory.iter().map(|(key, _)| key.len() + std::mem::size_of::<u32>()).sum();
        vec![
            ("block automata", automata),
            ("block id permutations", ids),
            ("directory", directory),
            ("id -> block map", self.id_to_block.size_bytes()),
            ("symbol table", self.symbols.as_ref().map_or(0, |t| t.size_bytes())),
        ]
    }

    /// Total reachable automaton states across every block, for the
    /// states-per-term investigation (DESIGN.md).
    pub fn reachable_state_count(&self) -> usize {
        self.blocks.iter().flatten().map(PackedAutomatonBlock::reachable_state_count).sum()
    }

    /// Terms per block, in block (term) order.
    pub fn block_sizes(&self) -> Vec<usize> {
        self.directory.iter().map(|&(_, slot)| self.block(slot).len()).collect()
    }

    /// Reassigns every id: blocks in `block_order` (a permutation of
    /// `0..block_count()`, positions in term order) get consecutive id
    /// ranges from 0, each block's terms in term order -- so the identity
    /// order makes every id its term's rank. Returns the mapping, indexed by
    /// old id (`u32::MAX` for ids not in use). Every block's ids become one
    /// run and the id -> block map one range per block, the cheapest form
    /// both have; freed ids are forgotten, and the next new term gets
    /// `len()`. Every id handed out before is invalidated: callers remap
    /// what they hold with the returned mapping.
    pub fn renumber(&mut self, block_order: &[usize]) -> Vec<u32> {
        assert_eq!(block_order.len(), self.directory.len(), "one position per block");
        let mut seen = vec![false; block_order.len()];
        for &pos in block_order {
            assert!(!std::mem::replace(&mut seen[pos], true), "block_order is a permutation");
        }
        let mut old_to_new = vec![u32::MAX; self.next_term_id as usize];
        let mut ids = std::mem::take(&mut self.workspace.ids);
        let mut runs = Vec::with_capacity(block_order.len());
        let mut next = 0u32;
        for &pos in block_order {
            let slot = self.directory[pos].1;
            let block = self.blocks[slot as usize].as_mut().expect("a live block slot");
            ids.clear();
            block.ids_into(&mut ids);
            for (rank, old) in ids.iter().enumerate() {
                old_to_new[old.0 as usize] = next + rank as u32;
            }
            block.set_id_run(next);
            let len = u32::try_from(ids.len()).expect("block length fits u32");
            runs.push((next, len, slot));
            next += len;
        }
        self.workspace.ids = ids;
        self.id_to_block = RangeMap::from_adjacent_runs(runs.into_iter());
        self.next_term_id = next;
        self.free_term_ids.clear();
        debug_assert_eq!(next as usize, self.len);
        old_to_new
    }

    /// Reassigns every id by group, then term order: `group_of(id)` (below
    /// `groups`) picks each live term's group; group 0's terms get the
    /// first ids, in term order, then group 1's, and so on. A block holds
    /// one id run per group it has terms in, and the id -> block map one
    /// range per (group, block). Otherwise as [`Self::renumber`]: returns
    /// the mapping by old id, forgets freed ids, the next new term gets
    /// `len()`, and every id handed out before is invalidated.
    pub fn renumber_grouped(&mut self, groups: usize, mut group_of: impl FnMut(TermId) -> usize) -> Vec<u32> {
        let mut ids = std::mem::take(&mut self.workspace.ids);
        self.ids_in_term_order(&mut ids);
        let group: Vec<u8> = ids
            .iter()
            .map(|&id| {
                let g = group_of(id);
                assert!(g < groups && groups <= 256, "group below `groups` (at most 256)");
                g as u8
            })
            .collect();
        let mut cursor = vec![0u32; groups];
        for &g in &group {
            cursor[g as usize] += 1;
        }
        let mut next = 0u32;
        for c in &mut cursor {
            let count = *c;
            *c = next;
            next += count;
        }
        let mut old_to_new = vec![u32::MAX; self.next_term_id as usize];
        let mut runs = Vec::new();
        let mut new_ids = Vec::new();
        let mut at = 0;
        for &(_, slot) in &self.directory {
            let block = self.blocks[slot as usize].as_mut().expect("a live block slot");
            let len = block.len();
            new_ids.clear();
            let first: Vec<u32> = cursor.clone();
            for (old, &g) in ids[at..at + len].iter().zip(&group[at..at + len]) {
                let new = cursor[g as usize];
                cursor[g as usize] += 1;
                old_to_new[old.0 as usize] = new;
                new_ids.push(TermId(new));
            }
            block.set_ids(&new_ids);
            for (g, &start) in first.iter().enumerate() {
                runs.push((start, cursor[g] - start, slot));
            }
            at += len;
        }
        runs.sort_unstable_by_key(|&(start, len, _)| (start, len));
        self.id_to_block = RangeMap::from_adjacent_runs(runs.into_iter().filter(|&(_, len, _)| len > 0));
        self.workspace.ids = ids;
        self.next_term_id = next;
        self.free_term_ids.clear();
        debug_assert_eq!(next as usize, self.len);
        old_to_new
    }

    /// Every live id, in its term's lexicographic order (`out` is cleared
    /// first): position = the term's rank. One pass over the blocks, no
    /// term decoding.
    pub fn ids_in_term_order(&self, out: &mut Vec<TermId>) {
        out.clear();
        out.reserve(self.len);
        for &(_, slot) in &self.directory {
            self.block(slot).ids_into(out);
        }
    }

    /// Writes the term for `id` into `buf` (cleared first), reusing its
    /// allocation; returns whether `id` is present.
    pub fn get_term_into(&self, id: TermId, buf: &mut Vec<u8>) -> bool {
        buf.clear();
        let Some((block_id, local_rank)) = self.reverse(id) else {
            return false;
        };
        self.block(block_id).term_at_into(local_rank as usize, buf);
        true
    }

    /// Resolves many ids at once: ids are grouped by block and each block
    /// decodes its requested ranks in one descent
    /// (`PackedAutomatonBlock::decode_sorted_ranks_into`), sharing every
    /// state visited by several. `ids` need not be sorted or deduplicated;
    /// `out` is cleared and filled in `ids` order, `None` for absent ids.
    pub fn get_terms_seeked_into(&self, ids: &[TermId], out: &mut Vec<Option<String>>) {
        out.clear();
        out.resize(ids.len(), None);
        let mut targets: Vec<(u32, usize, u32)> =
            ids.iter().enumerate().filter_map(|(pos, &id)| self.reverse(id).map(|(slot, rank)| (slot, pos, rank))).collect();
        targets.sort_unstable_by_key(|&(slot, _, rank)| (slot, rank));
        let mut ranks: Vec<(usize, u32)> = Vec::new();
        for group in targets.chunk_by(|a, b| a.0 == b.0) {
            ranks.clear();
            ranks.extend(group.iter().map(|&(_, pos, rank)| (pos, rank)));
            self.block(group[0].0).decode_sorted_ranks_into(&ranks, out);
        }
    }

    /// The live block in `slot` (directory entries and `id_to_block` only
    /// name live slots).
    fn block(&self, slot: u32) -> &PackedAutomatonBlock {
        self.blocks[slot as usize].as_ref().expect("a live block slot")
    }

    fn block_mut(&mut self, slot: u32) -> &mut PackedAutomatonBlock {
        self.blocks[slot as usize].as_mut().expect("a live block slot")
    }

    /// Stores `block` in a free slot (or a new one) and returns the slot.
    fn alloc_block(&mut self, block: PackedAutomatonBlock) -> u32 {
        if let Some(slot) = self.free_block_slots.pop() {
            self.blocks[slot as usize] = Some(block);
            return slot;
        }
        let slot = u32::try_from(self.blocks.len()).expect("block count fits u32");
        self.blocks.push(Some(block));
        slot
    }

    /// `(block slot, local rank)` of a live id. Only the id lookup can miss
    /// (an unknown or removed id). The rest are premises: `id_to_block`
    /// only points at live blocks holding the id (an emptied block's ids
    /// are cleared before the block is dropped, an absorbed block's ids are
    /// reassigned to the block absorbing them) -- `block` panics on a dead
    /// slot, the rank is checked in debug builds.
    #[inline]
    fn reverse(&self, id: TermId) -> Option<(u32, u32)> {
        let slot = self.id_to_block.get(id.0)?;
        let rank = self.block(slot).rank_of_id(id);
        debug_assert!(rank.is_some(), "id_to_block points {id:?} at a block without it");
        Some((slot, rank? as u32))
    }

    /// Points every id of `block_id` at it -- after the block was built
    /// from ids that lived elsewhere (split chunks, neighbor merges).
    fn assign_slots(&mut self, block_id: u32) {
        let mut ids = std::mem::take(&mut self.workspace.ids);
        ids.clear();
        self.block(block_id).ids_into(&mut ids);
        self.slot_ids.clear();
        self.slot_ids.extend(ids.iter().map(|id| id.0));
        self.id_to_block.assign(block_id, &mut self.slot_ids);
        self.workspace.ids = ids;
    }

    /// After a mutation of the block at `dir_idx`: refresh its directory
    /// key if its smallest term changed.
    fn refresh_directory_key(&mut self, dir_idx: usize, unchanged_prefix: usize) {
        if unchanged_prefix == 0 {
            let block_id = self.directory[dir_idx].1;
            self.directory[dir_idx].0 = self.block(block_id).term_bytes(0).into_boxed_slice();
        }
    }

    /// The directory index of the block whose key range holds `term`.
    fn route(&self, term: &[u8]) -> Option<usize> {
        if self.directory.is_empty() {
            return None;
        }
        Some(self.directory.partition_point(|(min_key, _)| min_key.as_ref() <= term).saturating_sub(1))
    }

    /// Splits the block at `dir_idx`, if it exceeds `block_max` terms or
    /// `BLOCK_TEXT_LIMIT` bytes of text, into balanced chunks of at most
    /// `SPLIT_FILL_PERCENT` of `block_max` (and at least `block_min`) --
    /// rather than by repeated median halving, which leaves blocks anywhere
    /// from half to completely full after a large batch run (DESIGN.md's
    /// "block size and fill") -- cut further by text (`chunk_ends`).
    fn split_while_oversized(&mut self, dir_idx: usize) {
        let block = self.block(self.directory[dir_idx].1);
        if block.len() <= self.block_max && block.text_bytes() <= BLOCK_TEXT_LIMIT {
            return;
        }
        let mut arena = std::mem::take(&mut self.workspace.entries);
        arena.clear();
        self.decode_block(self.directory[dir_idx].1, &mut arena);
        self.replace_with_chunks(dir_idx, &arena);
        self.workspace.entries = arena;
    }

    /// Replaces the block at `dir_idx` with fresh blocks over `arena`'s
    /// entries (sorted, non-empty, all routed to this block): one if they
    /// fit `block_max`, else balanced chunks of the split fill, all cut by
    /// text (`chunk_ends`). The first chunk keeps the block's slot.
    fn replace_with_chunks(&mut self, dir_idx: usize, arena: &EntriesArena) {
        debug_assert!(arena.len() > 0, "a block holds at least one term");
        let total = arena.len();
        let target = (self.block_max * SPLIT_FILL_PERCENT / 100).max(2 * self.block_min);
        let chunks = if total <= self.block_max { 1 } else { total.div_ceil(target) };
        let ends = chunk_ends(total, chunks, |i| arena.term_at(i).len());
        let block_id = self.directory[dir_idx].1;
        let mut new_directory = Vec::with_capacity(ends.len() - 1);
        let mut start = 0;
        for (chunk, &end) in ends.iter().enumerate() {
            let block = PackedAutomatonBlock::build(arena.terms(start..end), arena.ids(start..end), &mut self.workspace.automaton);
            let chunk_id = if chunk == 0 {
                *self.block_mut(block_id) = block;
                if self.directory[dir_idx].0.as_ref() != arena.term_at(0) {
                    self.directory[dir_idx].0 = Box::from(arena.term_at(0));
                }
                block_id
            } else {
                let id = self.alloc_block(block);
                new_directory.push((Box::from(arena.term_at(start)), id));
                id
            };
            self.assign_slots(chunk_id);
            start = end;
        }
        // Shift the router's tail once, rather than once per chunk.
        self.directory.splice(dir_idx + 1..dir_idx + 1, new_directory);
    }

    /// Merges the sorted, distinct `run` into the block at `dir_idx` by
    /// rebuilding it (`replace_with_chunks`): existing terms keep their
    /// id, new ones get one; `resolved` gets each run term's id in order.
    /// For runs an incremental merge can't be shown to fit
    /// (`PackedAutomatonBlock::fits_incremental`). Returns the terms added.
    fn merge_run_by_rebuild(&mut self, dir_idx: usize, run: &[&[u8]], mut resolved: impl FnMut(TermId)) -> usize {
        let mut old = std::mem::take(&mut self.workspace.entries);
        old.clear();
        self.decode_block(self.directory[dir_idx].1, &mut old);
        let mut merged = EntriesArena::new();
        let (mut i, mut added) = (0, 0);
        for &term in run {
            while i < old.len() && old.term_at(i) < term {
                merged.push(old.term_at(i), old.id_at(i));
                i += 1;
            }
            let id = if i < old.len() && old.term_at(i) == term {
                i += 1;
                old.id_at(i - 1)
            } else {
                added += 1;
                alloc_term_id(&mut self.free_term_ids, &mut self.next_term_id, self.id_limit)
            };
            merged.push(term, id);
            resolved(id);
        }
        for j in i..old.len() {
            merged.push(old.term_at(j), old.id_at(j));
        }
        self.workspace.entries = old;
        self.len += added;
        self.replace_with_chunks(dir_idx, &merged);
        added
    }

    /// Removes `term` from the block at `dir_idx` by rebuilding it, for a
    /// removal an incremental rewrite can't be shown to fit. Returns its
    /// id; the block is replaced, dropped if emptied.
    fn remove_by_rebuild(&mut self, dir_idx: usize, term: &[u8]) -> Option<TermId> {
        let mut old = std::mem::take(&mut self.workspace.entries);
        old.clear();
        self.decode_block(self.directory[dir_idx].1, &mut old);
        let found = (0..old.len()).find(|&i| old.term_at(i) == term);
        let removed = found.map(|at| {
            let mut rest = EntriesArena::new();
            for i in (0..old.len()).filter(|&i| i != at) {
                rest.push(old.term_at(i), old.id_at(i));
            }
            (old.id_at(at), rest)
        });
        self.workspace.entries = old;
        let (id, rest) = removed?;
        self.id_to_block.clear(id.0);
        self.free_term_ids.push(id.0);
        self.len -= 1;
        if rest.len() == 0 {
            self.remove_empty_block(dir_idx);
        } else {
            self.replace_with_chunks(dir_idx, &rest);
        }
        Some(id)
    }

    /// Appends `block_id`'s `(term, id)` entries to `arena`.
    fn decode_block(&mut self, block_id: u32, arena: &mut EntriesArena) {
        let mut ids = std::mem::take(&mut self.workspace.ids);
        self.block(block_id).decode_into(arena, &mut ids);
        self.workspace.ids = ids;
    }

    fn remove_empty_block(&mut self, dir_idx: usize) {
        let (_, block_id) = self.directory.remove(dir_idx);
        self.blocks[block_id as usize] = None;
        self.free_block_slots.push(block_id);
    }

    /// Merges the block at `dir_idx` with a neighbor (the next one, or the
    /// previous one for the last block), rebuilt as one block, or as split
    /// chunks if the two exceed `block_max` terms or the text limit.
    fn merge_with_neighbor(&mut self, dir_idx: usize) {
        if self.directory.len() < 2 {
            return;
        }
        let keep_idx = if dir_idx + 1 < self.directory.len() { dir_idx } else { dir_idx - 1 };
        let absorb_idx = keep_idx + 1;
        let keep_block_id = self.directory[keep_idx].1;
        let absorb_block_id = self.directory[absorb_idx].1;

        let mut arena = std::mem::take(&mut self.workspace.entries);
        arena.clear();
        self.decode_block(keep_block_id, &mut arena);
        self.decode_block(absorb_block_id, &mut arena);
        self.blocks[absorb_block_id as usize] = None;
        self.free_block_slots.push(absorb_block_id);
        self.directory.remove(absorb_idx);
        self.replace_with_chunks(keep_idx, &arena);
        self.workspace.entries = arena;
    }

    /// The first batch into an empty dictionary: balanced blocks of at
    /// most `block_max` terms, built directly.
    fn bootstrap(&mut self, order: &BatchOrder, uniques: &[&[u8]], result: &mut [TermId]) {
        let total = uniques.len();
        let ends = chunk_ends(total, total.div_ceil(self.block_max), |u| uniques[u].len());
        let mut chunk_ids = Vec::with_capacity(self.block_max.min(total));
        let mut start = 0;
        for end in ends {
            chunk_ids.clear();
            for u in start..end {
                let id = alloc_term_id(&mut self.free_term_ids, &mut self.next_term_id, self.id_limit);
                chunk_ids.push(id);
                for pos in order.positions(u) {
                    result[pos] = id;
                }
            }
            let block = PackedAutomatonBlock::build(uniques[start..end].iter().copied(), &chunk_ids, &mut self.workspace.automaton);
            let block_id = self.alloc_block(block);
            self.slot_ids.clear();
            self.slot_ids.extend(chunk_ids.iter().map(|id| id.0));
            self.id_to_block.assign(block_id, &mut self.slot_ids);
            self.directory.push((Box::from(uniques[start]), block_id));
            self.len += end - start;
            start = end;
        }
    }

    /// Inserts the distinct terms of `order` (sorted), routing each run of
    /// consecutive terms that falls in one block's key range to one
    /// `merge_into` call (or a rebuild, for a run an incremental merge
    /// can't be shown to fit). Returns ids in original input order, or an
    /// error, checked before anything changes (`check_new_terms`).
    fn try_insert_ordered<'a>(&mut self, order: &BatchOrder, term: impl Fn(usize) -> &'a [u8]) -> Result<Vec<TermId>, DictError> {
        let mut result = vec![TermId(0); order.len()];
        let uniques: Vec<&[u8]> = (0..order.unique_len()).map(term).collect();
        if uniques.is_empty() {
            return Ok(result);
        }
        self.check_new_terms(&uniques)?;
        if self.directory.is_empty() {
            self.bootstrap(order, &uniques, &mut result);
            return Ok(result);
        }

        let mut u = 0;
        while u < uniques.len() {
            let dir_idx = self.route(uniques[u]).expect("directory is non-empty");
            let run_end = match self.directory.get(dir_idx + 1) {
                Some((next_min, _)) => u + uniques[u..].partition_point(|t| *t < next_min.as_ref()),
                None => uniques.len(),
            };
            let block_id = self.directory[dir_idx].1;
            let run = &uniques[u..run_end];
            let run_text: u64 = run.iter().map(|t| t.len() as u64).sum();
            let block = self.block(block_id);
            if !block.fits_incremental(run_text, block.text_bytes() + run_text) {
                let mut k = u;
                self.merge_run_by_rebuild(dir_idx, run, |id| {
                    for pos in order.positions(k) {
                        result[pos] = id;
                    }
                    k += 1;
                });
                u = run_end;
                continue;
            }
            let PackedAutomatonDictionary {
                blocks,
                free_term_ids,
                next_term_id,
                id_limit,
                workspace,
                slot_ids,
                ..
            } = self;
            slot_ids.clear();
            let mut k = u;
            let block = blocks[block_id as usize].as_mut().expect("a live block slot");
            let outcome = block.merge_into(
                run,
                || {
                    let id = alloc_term_id(free_term_ids, next_term_id, *id_limit);
                    slot_ids.push(id.0);
                    id
                },
                |id| {
                    for pos in order.positions(k) {
                        result[pos] = id;
                    }
                    k += 1;
                },
                workspace,
            );
            if outcome.added > 0 {
                self.len += outcome.added;
                self.refresh_directory_key(dir_idx, outcome.unchanged_prefix);
                // Only the new ids changed block; existing ones' ranks
                // shifted, which the block's own inverse already reflects.
                self.id_to_block.assign(block_id, &mut self.slot_ids);
                self.split_while_oversized(dir_idx);
            }
            u = run_end;
        }
        Ok(result)
    }
}

impl TermDictionary for PackedAutomatonDictionary {
    fn get_id(&self, term: &str) -> Option<TermId> {
        let block = self.block(self.directory[self.route(term.as_bytes())?].1);
        Some(block.id_at(block.rank(term)?))
    }

    fn get_term(&self, id: TermId) -> Option<String> {
        let (block_id, local_rank) = self.reverse(id)?;
        Some(self.block(block_id).term_at(local_rank as usize))
    }

    fn try_insert(&mut self, term: &str) -> Result<TermId, DictError> {
        if let Some(id) = self.get_id(term) {
            return Ok(id);
        }
        let mut order = BatchOrder::default();
        order.prepare(1, |_| term.as_bytes());
        Ok(self.try_insert_ordered(&order, |_| term.as_bytes())?[0])
    }

    fn try_insert_batch(&mut self, terms: &[&str]) -> Result<Vec<TermId>, DictError> {
        let mut order = BatchOrder::default();
        order.prepare(terms.len(), |i| terms[i].as_bytes());
        self.try_insert_ordered(&order, |u| terms[order.first(u)].as_bytes())
    }

    fn check_insert_batch(&self, terms: &[&str]) -> Result<(), DictError> {
        let mut uniques: Vec<&[u8]> = terms.iter().map(|t| t.as_bytes()).collect();
        if uniques.len() as u64 > self.ids_available() {
            uniques.sort_unstable();
            uniques.dedup();
        }
        self.check_new_terms(&uniques)
    }

    fn remove(&mut self, term: &str) -> Option<TermId> {
        let dir_idx = self.route(term.as_bytes())?;
        let block_id = self.directory[dir_idx].1;
        let block = self.block(block_id);
        if !block.fits_incremental(term.len() as u64, block.text_bytes()) {
            return self.remove_by_rebuild(dir_idx, term.as_bytes());
        }
        let (id, unchanged_prefix) = self.blocks[block_id as usize]
            .as_mut()
            .expect("a live block slot")
            .remove(term, &mut self.workspace)?;
        self.id_to_block.clear(id.0);
        self.free_term_ids.push(id.0);
        self.len -= 1;

        let remaining = self.block(block_id).len();
        if remaining == 0 {
            self.remove_empty_block(dir_idx);
        } else {
            self.refresh_directory_key(dir_idx, unchanged_prefix);
            if remaining < self.block_min {
                self.merge_with_neighbor(dir_idx);
            }
        }
        Some(id)
    }

    /// Ids handed out from now on stay below `limit`. Panics if an id at or
    /// above it was already handed out.
    fn set_id_limit(&mut self, limit: u32) {
        assert!(self.next_term_id <= limit, "ids at or above the new limit are already in use");
        self.id_limit = limit;
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

    /// After `renumber`, every term keeps resolving under its new id, ids
    /// follow the requested block order, the id structures collapse to one
    /// run per block, and new terms take `len()`.
    #[test]
    fn renumber_reassigns_ids_by_block_order() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(4, 16);
        let terms: Vec<String> = (0..400).map(|i| format!("t{:05}", (i * 7919) % 1000)).collect();
        let mut old = Vec::new();
        for t in &terms {
            old.push(dict.insert(t));
        }
        for t in terms.iter().step_by(9) {
            dict.remove(t);
        }
        dict.shrink_to_fit();
        let live: Vec<(String, TermId)> = terms.iter().cloned().zip(old).filter(|(t, _)| dict.get_id(t).is_some()).collect();
        let blocks = dict.block_count();
        let order: Vec<usize> = (0..blocks).rev().collect();
        let sizes = dict.block_sizes();
        let map = dict.renumber(&order);
        assert_eq!(dict.size_breakdown()[1].1, 0, "every block's ids are one run");
        for (t, old_id) in &live {
            let new_id = TermId(map[old_id.0 as usize]);
            assert_eq!(dict.get_id(t), Some(new_id));
            assert_eq!(dict.get_term(new_id).as_deref(), Some(t.as_str()));
        }
        // Reversed block order: the last block (largest terms) gets id 0.
        let mut by_term: Vec<&(String, TermId)> = live.iter().collect();
        by_term.sort();
        let last_block_first = by_term[by_term.len() - sizes[blocks - 1]];
        assert_eq!(dict.get_id(&last_block_first.0), Some(TermId(0)));
        // New terms take the next id, then the dictionary keeps working.
        assert_eq!(dict.insert("t99999-new"), TermId(dict.len() as u32 - 1));
        let identity: Vec<usize> = (0..dict.block_count()).collect();
        let map = dict.renumber(&identity);
        let mut ranked: Vec<String> = live.iter().map(|(t, _)| t.clone()).chain(["t99999-new".to_string()]).collect();
        ranked.sort();
        for (rank, t) in ranked.iter().enumerate() {
            assert_eq!(dict.get_id(t), Some(TermId(rank as u32)), "identity order: id = rank");
        }
        assert_eq!(map.iter().filter(|&&m| m != u32::MAX).count(), dict.len());
    }

    /// After `renumber_grouped`, every term keeps resolving, each group's
    /// ids are a contiguous range in group order and term order inside it,
    /// mutations keep working, and new terms take `len()`.
    #[test]
    fn renumber_grouped_orders_ids_by_group_then_term() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(4, 16);
        let terms: Vec<String> = (0..400).map(|i| format!("t{:05}", (i * 7919) % 1000)).collect();
        let mut old = Vec::new();
        for t in &terms {
            old.push(dict.insert(t));
        }
        for t in terms.iter().step_by(9) {
            dict.remove(t);
        }
        let live: Vec<(String, TermId)> = terms.iter().cloned().zip(old).filter(|(t, _)| dict.get_id(t).is_some()).collect();
        // Group by the term's last digit modulo 3; group 1 deliberately empty.
        let group_of_term = |t: &str| match (t.as_bytes()[t.len() - 1] - b'0') % 3 {
            0 => 2,
            1 => 0,
            _ => 3,
        };
        let by_old_id: std::collections::HashMap<u32, usize> = live.iter().map(|(t, id)| (id.0, group_of_term(t))).collect();
        let map = dict.renumber_grouped(4, |id| by_old_id[&id.0]);
        let mut expected: Vec<(usize, &str)> = live.iter().map(|(t, _)| (group_of_term(t), t.as_str())).collect();
        expected.sort();
        for (new_id, &(_, t)) in expected.iter().enumerate() {
            assert_eq!(dict.get_id(t), Some(TermId(new_id as u32)), "group order, then term order");
            assert_eq!(dict.get_term(TermId(new_id as u32)).as_deref(), Some(t));
        }
        for (t, old_id) in &live {
            assert_eq!(map[old_id.0 as usize], dict.get_id(t).unwrap().0);
        }
        let new_id = dict.insert("t99999-new");
        assert_eq!(new_id, TermId(dict.len() as u32 - 1));
        let gone = expected[expected.len() / 2].1;
        assert_eq!(dict.remove(gone), Some(TermId(expected.len() as u32 / 2)));
        assert_eq!(dict.get_id(gone), None);
        assert_eq!(dict.get_term(new_id).as_deref(), Some("t99999-new"));
    }

    /// Every term and id of `dict` matches `model` (term -> id).
    fn assert_matches(dict: &PackedAutomatonDictionary, model: &std::collections::BTreeMap<String, TermId>) {
        assert_eq!(dict.len(), model.len());
        for (term, &id) in model {
            assert_eq!(dict.get_id(term), Some(id));
            assert_eq!(dict.get_term(id).as_deref(), Some(term.as_str()));
        }
    }

    /// Long terms (unit-test limits: 256 KiB terms, 16-state chains, 4 MiB
    /// of records per block) drive merges, removals and neighbor merges
    /// through the rebuild paths; `next_address` asserts the write bound on
    /// every write in debug builds.
    #[test]
    fn long_terms_stay_within_the_write_bounds() {
        use crate::limits::MAX_TERM_BYTES;
        let mut dict = PackedAutomatonDictionary::with_block_bounds(2, 8);
        let mut model = std::collections::BTreeMap::new();
        let mut x = 0x9E37_79B9u32;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let term = |n: u32, len: usize| {
            let mut t = format!("<http://example.org/{}/", n % 7);
            while t.len() < len {
                t.push_str(&format!("{n:x}é"));
            }
            t.truncate(t.char_indices().map(|(i, _)| i).take_while(|&i| i <= len).last().unwrap_or(0));
            t
        };
        for round in 0..30 {
            let lens = [10, 200, 5_000, 60_000, MAX_TERM_BYTES / 2, MAX_TERM_BYTES];
            let batch: Vec<String> = (0..6).map(|_| term(next() % 50, lens[(next() % 6) as usize])).collect();
            let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
            let ids = if round % 3 == 0 { refs.iter().map(|t| dict.insert(t)).collect() } else { dict.insert_batch(&refs) };
            for (t, id) in batch.iter().zip(ids) {
                assert_eq!(*model.entry(t.clone()).or_insert(id), id);
            }
            if round % 4 == 3 {
                let victims: Vec<String> = model.keys().step_by(3).cloned().collect();
                for t in victims {
                    assert_eq!(dict.remove(&t), model.remove(&t));
                }
            }
            if round % 10 == 9 {
                dict.shrink_to_fit();
            }
            assert_matches(&dict, &model);
        }
        assert!(dict.blocks.iter().flatten().all(|b| b.text_bytes() <= crate::limits::BLOCK_TEXT_LIMIT));
    }

    /// A term over the size limit is refused and changes nothing; one at
    /// the limit is accepted.
    #[test]
    fn oversized_terms_are_refused_without_changes() {
        use crate::limits::MAX_TERM_BYTES;
        let mut dict = PackedAutomatonDictionary::with_block_bounds(2, 8);
        let base: Vec<String> = (0..20).map(|i| format!("t{i:02}")).collect();
        let refs: Vec<&str> = base.iter().map(String::as_str).collect();
        dict.insert_batch(&refs);
        let (len, size) = (dict.len(), dict.size_bytes());
        let huge = "h".repeat(MAX_TERM_BYTES + 1);
        let too_large = DictError::TermTooLarge { len: MAX_TERM_BYTES + 1, max: MAX_TERM_BYTES };
        assert_eq!(dict.try_insert(&huge), Err(too_large.clone()));
        assert_eq!(dict.try_insert_batch(&["new", &huge]), Err(too_large.clone()));
        assert_eq!(dict.check_insert_batch(&["new", &huge]), Err(too_large));
        assert_eq!((dict.len(), dict.size_bytes(), dict.get_id("new")), (len, size, None));
        let at_limit = "h".repeat(MAX_TERM_BYTES);
        let id = dict.try_insert(&at_limit).unwrap();
        assert_eq!(dict.get_term(id).as_deref(), Some(at_limit.as_str()));
    }

    /// With an id limit, a batch needing more new ids than remain is
    /// refused whole; terms already present and freed ids don't count
    /// against it.
    #[test]
    fn id_limit_refuses_whole_batches() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(2, 8);
        dict.set_id_limit(10);
        let first: Vec<String> = (0..8).map(|i| format!("a{i}")).collect();
        let refs: Vec<&str> = first.iter().map(String::as_str).collect();
        dict.insert_batch(&refs);
        assert_eq!(dict.ids_available(), 2);
        let (len, size) = (dict.len(), dict.size_bytes());
        let err = dict.try_insert_batch(&["b0", "b1", "b2", "a0"]);
        assert_eq!(err, Err(DictError::IdsExhausted { needed: 3, available: 2 }));
        assert_eq!((dict.len(), dict.size_bytes(), dict.get_id("b0")), (len, size, None));
        // Present terms and duplicates need no ids.
        let ids = dict.try_insert_batch(&["a1", "b0", "a2", "b0", "b1", "a3"]).unwrap();
        assert_eq!(ids[1], ids[3]);
        assert_eq!(dict.ids_available(), 0);
        assert!(matches!(dict.try_insert("c"), Err(DictError::IdsExhausted { needed: 1, available: 0 })));
        assert_eq!(dict.try_insert("a5"), Ok(dict.get_id("a5").unwrap()));
        // A freed id is available again.
        let freed = dict.remove("a4").unwrap();
        assert_eq!(dict.try_insert("c"), Ok(freed));
        assert!(dict.blocks.iter().flatten().all(|b| b.len() > 0));
    }

    /// A sharded batch is checked on every shard before any inserts: one
    /// shard out of ids leaves the others unchanged.
    #[test]
    fn sharded_batches_are_atomic_across_shards() {
        use crate::ShardedDictionary;
        let mut dict = ShardedDictionary::new(vec!["m".to_string()], || PackedAutomatonDictionary::with_block_bounds(2, 8));
        // Two shards: shard 1 ids start at 1 << 31; leave it 3.
        dict.set_id_limit((1u32 << 31) + 3);
        let err = dict.try_insert_batch(&["a", "b", "x", "y", "z", "zz"]);
        assert_eq!(err, Err(DictError::IdsExhausted { needed: 4, available: 3 }));
        assert_eq!(dict.len(), 0);
        let ids = dict.try_insert_batch(&["a", "b", "x", "y", "z"]).unwrap();
        assert_eq!(ids[2].0, 1 << 31);
        assert_eq!(dict.len(), 5);
        assert!(dict.try_insert("zz").is_err());
        assert_eq!(dict.try_insert("c"), Ok(TermId(2)));
    }

    #[test]
    fn size_breakdown_sums_to_size_bytes() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(4, 16);
        for i in 0..500 {
            dict.insert(&format!("http://example.org/entity-{}", (i * 7919) % 1000));
        }
        dict.shrink_to_fit();
        assert_eq!(dict.size_breakdown().iter().map(|&(_, b)| b).sum::<usize>(), dict.size_bytes());
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
    fn ids_in_term_order_follow_term_order() {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(2, 8);
        let terms: Vec<String> = (0..300).map(|i| format!("t{:05}", (i * 7919) % 1000)).collect();
        for t in &terms {
            dict.insert(t);
        }
        for t in terms.iter().step_by(5) {
            dict.remove(t);
        }
        let mut ids = Vec::new();
        dict.ids_in_term_order(&mut ids);
        assert_eq!(ids.len(), dict.len());
        let decoded: Vec<String> = ids.iter().map(|&id| dict.get_term(id).expect("live id")).collect();
        assert!(decoded.windows(2).all(|w| w[0] < w[1]), "strictly increasing terms");
    }

    #[test]
    fn insert_prepared_matches_insert_batch() {
        let terms: Vec<String> = (0..2000).map(|i| format!("http://example.org/{}", i % 1500)).collect();
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let mut batch = crate::TermBatch::new();
        for t in &refs {
            batch.push(t);
        }
        let batch = batch.prepare();
        let mut prepared = PackedAutomatonDictionary::with_block_bounds(8, 32);
        let ids = prepared.insert_prepared(&batch);
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
