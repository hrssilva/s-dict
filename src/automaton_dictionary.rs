//! `AutomatonDictionary`: `Dictionary`'s exact macro structure (lexical
//! blocks, sorted-array router, split/merge on overflow/underflow,
//! `TermId -> (BlockId, local_rank)` reverse mapping) with `AutomatonBlock`
//! (`automaton_block.rs`) in place of `Block` -- the piece that actually
//! lets the custom automaton be measured end to end, through the same
//! `TermDictionary` trait every other implementation in this crate shares,
//! instead of only in isolated block-level microbenchmarks. This is the
//! comparison implementation the approved plan named from the start
//! ("alongside, promoted only if it wins"): `Dictionary` and `Block` stay
//! untouched, and this crate now has a fourth `TermDictionary`
//! implementation (after `PlainDictionary`, `Dictionary`,
//! `FrontCodedDictionary`), matching `FrontCodedDictionary`'s own role as
//! a from-scratch parallel structure, not a variant of `Dictionary`.
//!
//! Deliberately smaller in scope than `Dictionary` for this first pass,
//! not an oversight: no `PreparedBatch`/`insert_prepared` support, no
//! `get_term_into`/`get_terms_into` buffer-reusing decode, no
//! `RebuildStats`/`size_bytes` instrumentation. Those are all real
//! optimizations `Dictionary` accumulated over this project's history;
//! porting them is future work once this comparison itself proves
//! worthwhile, not a prerequisite for measuring it.
//!
//! `assign_ranks` *does* take an `unchanged_prefix`/`skip` now, mirroring
//! `Dictionary`'s own -- this used to not be worth having, back when
//! `AutomatonBlock::merge_into`/`remove` always rebuilt their whole `ids`
//! array regardless. That turned out to be a real, previously unmeasured
//! cost (see `AutomatonBlock`'s own module doc comment and DESIGN.md's
//! "measured end to end" section: it made every mutating operation
//! *slower* than `Dictionary`, not faster), so both were rewritten to
//! report a real `unchanged_prefix`, and `assign_ranks` here now skips
//! the same way `Dictionary`'s does -- and needs no decode to do it,
//! since it only ever needed `(local_rank, id)` pairs, which `id_at`
//! already gives directly.

use std::num::NonZeroU32;

use crate::automaton_block::AutomatonBlock;
use crate::batch::BatchOrder;
use crate::entries_arena::EntriesArena;
use crate::{TermDictionary, TermId};

/// A stable handle for a block: an index into `AutomatonDictionary`'s
/// block slab. Mirrors `dictionary::BlockId` exactly (see its own doc
/// comment for the niche-optimization rationale) -- a separate type, not
/// a reuse of `dictionary::BlockId`, so the two dictionaries' block slabs
/// can never be confused for each other at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AutomatonBlockId(NonZeroU32);

impl AutomatonBlockId {
    fn from_slot(slot: u32) -> AutomatonBlockId {
        AutomatonBlockId(
            NonZeroU32::new(slot + 1)
                .expect("slot + 1 does not overflow u32 for realistic block counts"),
        )
    }

    fn slot(self) -> u32 {
        self.0.get() - 1
    }
}

const DEFAULT_BLOCK_MIN: usize = 64;
const DEFAULT_BLOCK_MAX: usize = 256;

/// See the module doc comment. Same macro structure as `Dictionary`,
/// `AutomatonBlock` in place of `Block`.
#[derive(Debug)]
pub struct AutomatonDictionary {
    blocks: Vec<Option<AutomatonBlock>>,
    free_block_slots: Vec<u32>,
    directory: Vec<(Box<str>, AutomatonBlockId)>,
    id_to_block: Vec<Option<(AutomatonBlockId, u16)>>,
    free_term_ids: Vec<u32>,
    next_term_id: u32,
    block_min: usize,
    block_max: usize,
    len: usize,
    /// One reusable scratch arena, not two: unlike `Block::merge_into`,
    /// `AutomatonBlock::merge_into` mutates its block in place and needs
    /// no external "old vs. merged" arena pair -- this is only used for
    /// the decode-then-rebuild shape `split`/`merge_with_neighbor` (and
    /// the post-merge `assign_ranks` re-decode) still need. Taken via
    /// `mem::take` at the start of a mutation and handed back before any
    /// call that might itself want scratch space, same discipline as
    /// `Dictionary`'s own scratch arenas.
    scratch: EntriesArena,
}

impl Default for AutomatonDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl AutomatonDictionary {
    /// Creates an empty dictionary with the default block bounds.
    pub fn new() -> Self {
        Self::with_block_bounds(DEFAULT_BLOCK_MIN, DEFAULT_BLOCK_MAX)
    }

    /// Creates an empty dictionary bounding each block to
    /// `[block_min, block_max]` terms -- same contract as
    /// `Dictionary::with_block_bounds`, so the two can be compared at
    /// identical block sizes.
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
        AutomatonDictionary {
            blocks: Vec::new(),
            free_block_slots: Vec::new(),
            directory: Vec::new(),
            id_to_block: Vec::new(),
            free_term_ids: Vec::new(),
            next_term_id: 0,
            block_min,
            block_max,
            len: 0,
            scratch: EntriesArena::new(),
        }
    }

    /// The number of live blocks, for introspection/testing.
    pub fn block_count(&self) -> usize {
        self.directory.len()
    }

    /// `Dictionary::size_bytes`'s exact accounting (blocks' payload + ids
    /// array, the router, the reverse directory), `AutomatonBlockId` in
    /// place of `BlockId` -- see `AutomatonBlock::payload_bytes`/
    /// `Automaton::size_bytes` for what a block's own payload does and
    /// doesn't include.
    pub fn size_bytes(&self) -> usize {
        let blocks_bytes: usize = self
            .blocks
            .iter()
            .filter_map(|b| b.as_ref())
            .map(|b| b.payload_bytes() + b.len() * std::mem::size_of::<TermId>())
            .sum();
        let directory_bytes: usize = self
            .directory
            .iter()
            .map(|(key, _)| key.len() + std::mem::size_of::<AutomatonBlockId>())
            .sum();
        let id_to_block_bytes =
            self.id_to_block.len() * std::mem::size_of::<Option<(AutomatonBlockId, u16)>>();
        blocks_bytes + directory_bytes + id_to_block_bytes
    }

    /// `Dictionary::get_term_into`'s exact contract: decodes into a
    /// caller-reused buffer instead of allocating a fresh `String` every
    /// call. `buf` is cleared first; returns `false` (leaving `buf`
    /// empty) if `id` isn't present.
    pub fn get_term_into(&self, id: TermId, buf: &mut Vec<u8>) -> bool {
        buf.clear();
        let Some((block_id, local_rank)) = self.id_to_block.get(id.0 as usize).and_then(|e| *e)
        else {
            return false;
        };
        self.block(block_id).term_at_into(local_rank as usize, buf);
        true
    }

    /// `Dictionary::get_terms_into`'s exact contract: decodes many ids in
    /// one call, grouped by block and decoded once per affected block
    /// (via `AutomatonBlock::decode_into`'s `Automaton::stream`-based
    /// walk) instead of one independent root-to-leaf automaton walk per
    /// id. `ids` need not be sorted or deduplicated; `out` is cleared
    /// first and filled in the same order as `ids`, `None` marking an id
    /// not present.
    pub fn get_terms_into(&self, ids: &[TermId], out: &mut Vec<Option<String>>) {
        out.clear();
        out.resize(ids.len(), None);

        let mut by_block: std::collections::HashMap<AutomatonBlockId, Vec<(usize, u16)>> =
            std::collections::HashMap::new();
        for (pos, &id) in ids.iter().enumerate() {
            if let Some((block_id, local_rank)) =
                self.id_to_block.get(id.0 as usize).copied().flatten()
            {
                by_block
                    .entry(block_id)
                    .or_default()
                    .push((pos, local_rank));
            }
        }

        let mut arena = EntriesArena::new();
        for (block_id, positions) in by_block {
            arena.clear();
            self.block(block_id).decode_into(&mut arena);
            for (pos, local_rank) in positions {
                out[pos] = Some(arena.term_at(local_rank as usize).to_string());
            }
        }
    }

    /// `Dictionary::get_terms_scattered_into`'s exact contract and
    /// rationale: for a query that only touches a small fraction of each
    /// affected block, decode just the requested ranks directly via
    /// `AutomatonBlock::term_at_into` (an `O(term length)` walk from the
    /// automaton's root per call, independent of block size -- the same
    /// primitive `get_term_into` uses) instead of streaming each touched
    /// block's entire content. No block-grouping pass needed: `id_to_block`
    /// already gives `(block_id, local_rank)` in O(1) per id.
    pub fn get_terms_scattered_into(&self, ids: &[TermId], out: &mut Vec<Option<String>>) {
        out.clear();
        out.resize(ids.len(), None);
        let mut buf = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            let Some((block_id, local_rank)) =
                self.id_to_block.get(id.0 as usize).copied().flatten()
            else {
                continue;
            };
            self.block(block_id)
                .term_at_into(local_rank as usize, &mut buf);
            out[pos] = Some(
                std::str::from_utf8(&buf)
                    .expect("keys were inserted as valid UTF-8 str bytes")
                    .to_string(),
            );
        }
    }

    /// A third bulk-decode strategy, testing a specific hypothesis:
    /// unlike `get_terms_scattered_into` (independent `term_at_into` per
    /// id, always restarting from the automaton's root) or
    /// `get_terms_into` (streams a touched block's *entire* content),
    /// this groups ids by block, sorts each block's requested ranks
    /// ascending, then decodes them via one `AutomatonBlock::
    /// decode_sorted_ranks_into` call per block -- seeking to the first
    /// requested rank once, then advancing forward through the rest via
    /// `Automaton::seek_stream`'s resumable stack, reusing whatever
    /// prefix consecutive/nearby *requested* ranks share instead of
    /// paying a full root descent for each and without decoding entries
    /// nobody asked for. Whether this actually wins over the other two
    /// in practice depends on how much prefix sharing real requested
    /// ranks have -- see `s-bench`'s `dict_bench` for the measurement.
    pub fn get_terms_seeked_into(&self, ids: &[TermId], out: &mut Vec<Option<String>>) {
        out.clear();
        out.resize(ids.len(), None);

        let mut by_block: std::collections::HashMap<AutomatonBlockId, Vec<(usize, u16)>> =
            std::collections::HashMap::new();
        for (pos, &id) in ids.iter().enumerate() {
            if let Some((block_id, local_rank)) =
                self.id_to_block.get(id.0 as usize).copied().flatten()
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

    fn block(&self, id: AutomatonBlockId) -> &AutomatonBlock {
        self.blocks[id.slot() as usize]
            .as_ref()
            .expect("AutomatonBlockId must reference a live block")
    }

    fn block_mut(&mut self, id: AutomatonBlockId) -> &mut AutomatonBlock {
        self.blocks[id.slot() as usize]
            .as_mut()
            .expect("AutomatonBlockId must reference a live block")
    }

    fn alloc_block(&mut self, block: AutomatonBlock) -> AutomatonBlockId {
        match self.free_block_slots.pop() {
            Some(slot) => {
                self.blocks[slot as usize] = Some(block);
                AutomatonBlockId::from_slot(slot)
            }
            None => {
                let slot = self.blocks.len() as u32;
                self.blocks.push(Some(block));
                AutomatonBlockId::from_slot(slot)
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

    fn set_id_to_block(&mut self, id: TermId, block_id: AutomatonBlockId, local_rank: usize) {
        let idx = id.0 as usize;
        if idx >= self.id_to_block.len() {
            self.id_to_block.resize(idx + 1, None);
        }
        self.id_to_block[idx] = Some((block_id, local_rank as u16));
    }

    /// Reassigns `TermId -> (block_id, local_rank)` for every entry in
    /// `block_id`'s content from `skip` onward -- entries before `skip`
    /// are assumed already correct at their current position (see
    /// [`automaton_block::AutomatonMergeOutcome::unchanged_prefix`]).
    /// Reads ids straight from the block (`id_at`), not a decode:
    /// `assign_ranks` never needs term text, only `(local_rank, id)`
    /// pairs, so unlike the earlier decode-based version there is no
    /// decode to skip in the first place.
    fn assign_ranks(&mut self, block_id: AutomatonBlockId, skip: usize) {
        let len = self.block(block_id).len();
        for local_rank in skip..len {
            let id = self.block(block_id).id_at(local_rank);
            self.set_id_to_block(id, block_id, local_rank);
        }
    }

    /// Re-reads `block_id`'s (possibly unchanged) `min_key` into the
    /// directory, but only when `unchanged_prefix == 0` -- that's
    /// exactly the condition under which rank 0 (what `min_key` caches)
    /// could have moved (see `AutomatonBlock::merge_into`/`remove`'s own
    /// doc comments for the same check on the block's own `min_key`
    /// field). `unchanged_prefix > 0` means the directory's copy is
    /// already correct, and `Box<str>::from(&str)` is a real allocation
    /// and copy every time -- found doing this unconditionally on every
    /// mutation via a fresh `perf record` of `sequential_insert`
    /// (DESIGN.md's allocation-pattern audit), the same over-eager
    /// recomputation `AutomatonBlock` itself had.
    fn refresh_directory_key(
        &mut self,
        dir_idx: usize,
        block_id: AutomatonBlockId,
        unchanged_prefix: usize,
    ) {
        if unchanged_prefix == 0 {
            self.directory[dir_idx].0 = self.block(block_id).min_key().into();
        }
    }

    /// The directory index of the block that owns (or would own) `term`,
    /// or `None` if the dictionary has no blocks yet.
    fn route(&self, term: &str) -> Option<usize> {
        if self.directory.is_empty() {
            return None;
        }
        let idx = self
            .directory
            .partition_point(|(min_key, _)| min_key.as_ref() <= term);
        Some(idx.saturating_sub(1))
    }

    /// Repeatedly splits the block at `start_dir_idx` (and any newly
    /// created sibling) until every resulting block is within
    /// `block_max` -- identical shape to `Dictionary::split_while_oversized`.
    fn split_while_oversized(&mut self, start_dir_idx: usize) {
        let mut stack = vec![start_dir_idx];
        while let Some(dir_idx) = stack.pop() {
            let block_id = self.directory[dir_idx].1;
            if self.block(block_id).len() > self.block_max {
                self.split(dir_idx);
                stack.push(dir_idx);
                stack.push(dir_idx + 1);
            }
        }
    }

    /// Median-splits the block at `dir_idx` into two freshly built
    /// blocks, inserting a new directory entry for the right half.
    fn split(&mut self, dir_idx: usize) {
        let block_id = self.directory[dir_idx].1;
        let mut arena = std::mem::take(&mut self.scratch);
        arena.clear();
        self.block(block_id).decode_into(&mut arena);
        let mid = arena.len() / 2;

        let left_block = AutomatonBlock::build(arena.range(0, mid));
        let right_block = AutomatonBlock::build(arena.range(mid, arena.len()));
        let right_min: Box<str> = right_block.min_key().into();

        *self.block_mut(block_id) = left_block;
        self.assign_ranks(block_id, 0);

        let right_block_id = self.alloc_block(right_block);
        self.assign_ranks(right_block_id, 0);

        self.scratch = arena;

        self.directory
            .insert(dir_idx + 1, (right_min, right_block_id));
    }

    /// Removes the (already empty) directory entry and frees its block
    /// slot. Routing falls through to the predecessor block automatically.
    fn remove_empty_block(&mut self, dir_idx: usize) {
        let (_, block_id) = self.directory.remove(dir_idx);
        self.blocks[block_id.slot() as usize] = None;
        self.free_block_slots.push(block_id.slot());
    }

    /// Absorbs the underflowing block at `dir_idx` into a lexically
    /// adjacent neighbor, splitting the result back apart if it now
    /// exceeds `block_max` -- identical shape to
    /// `Dictionary::merge_with_neighbor`.
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

        let rebuilt = AutomatonBlock::build(arena.iter());
        *self.block_mut(keep_block_id) = rebuilt;
        self.assign_ranks(keep_block_id, 0);

        self.scratch = arena;

        self.blocks[absorb_block_id.slot() as usize] = None;
        self.free_block_slots.push(absorb_block_id.slot());
        self.directory.remove(absorb_idx);

        self.split_while_oversized(keep_idx);
    }

    /// Assigns fresh ids to an initial sorted, unique batch of terms and
    /// chunks them directly into balanced `<= block_max`-sized blocks --
    /// identical shape to `Dictionary::bootstrap`.
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
                AutomatonBlock::build((start..end).map(term).zip(chunk_ids.iter().copied()));
            let min_key: Box<str> = block.min_key().into();
            let block_id = self.alloc_block(block);
            for (local_rank, &id) in chunk_ids.iter().enumerate() {
                self.set_id_to_block(id, block_id, local_rank);
            }
            self.directory.push((min_key, block_id));
            self.len += end - start;
            start = end;
        }
    }

    /// Batch insertion, grouping incoming (already sorted, deduplicated)
    /// terms by destination block -- same run-grouping shape as
    /// `Dictionary::insert_ordered`, `AutomatonBlock::merge_into` (mutate
    /// in place) in place of `Block::merge_into` (produce a merged arena
    /// for the caller to rebuild from).
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
            let free_term_ids = &mut self.free_term_ids;
            let next_term_id = &mut self.next_term_id;
            let block_ref = self.blocks[block_id.slot() as usize]
                .as_mut()
                .expect("AutomatonBlockId must reference a live block");
            let outcome = block_ref.merge_into(
                (u..run_end).map(&term),
                || match free_term_ids.pop() {
                    Some(slot) => TermId(slot),
                    None => {
                        let id = TermId(*next_term_id);
                        *next_term_id += 1;
                        id
                    }
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
                self.assign_ranks(block_id, outcome.unchanged_prefix);
                if self.block(block_id).len() > self.block_max {
                    self.split_while_oversized(dir_idx);
                }
            }
            u = run_end;
        }
        result
    }
}

impl TermDictionary for AutomatonDictionary {
    fn get_id(&self, term: &str) -> Option<TermId> {
        let dir_idx = self.route(term)?;
        let block_id = self.directory[dir_idx].1;
        let local_rank = self.block(block_id).rank(term)?;
        Some(self.block(block_id).id_at(local_rank))
    }

    fn get_term(&self, id: TermId) -> Option<String> {
        let (block_id, local_rank) = self.id_to_block.get(id.0 as usize)?.as_ref().copied()?;
        Some(self.block(block_id).term_at(local_rank as usize))
    }

    fn insert(&mut self, term: &str) -> TermId {
        if self.directory.is_empty() {
            let id = self.alloc_term_id();
            let block = AutomatonBlock::build(std::iter::once((term, id)));
            let block_id = self.alloc_block(block);
            self.directory.push((term.into(), block_id));
            self.set_id_to_block(id, block_id, 0);
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
        self.assign_ranks(block_id, outcome.unchanged_prefix);

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

        self.id_to_block[id.0 as usize] = None;
        self.free_term_ids.push(id.0);
        self.len -= 1;

        let remaining = self.block(block_id).len();
        if remaining == 0 {
            self.remove_empty_block(dir_idx);
        } else {
            self.refresh_directory_key(dir_idx, block_id, unchanged_prefix);
            self.assign_ranks(block_id, unchanged_prefix);
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
        let mut dict = AutomatonDictionary::with_block_bounds(2, 8);
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
        let mut dict = AutomatonDictionary::with_block_bounds(4, 16);
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
    fn get_terms_into_matches_individual_lookups_across_blocks() {
        let mut dict = AutomatonDictionary::with_block_bounds(2, 8);
        let terms: Vec<String> = (0..100).map(|i| format!("term-{i:04}")).collect();
        let ids: Vec<TermId> = terms.iter().map(|t| dict.insert(t)).collect();
        assert!(dict.block_count() > 1, "should span multiple blocks");

        let mut query: Vec<TermId> = vec![ids[50], ids[3], ids[99], ids[50], ids[0]];
        query.push(TermId(999_999)); // absent

        let mut out = Vec::new();
        dict.get_terms_into(&query, &mut out);
        assert_eq!(out.len(), query.len());
        for (i, &id) in query.iter().enumerate() {
            assert_eq!(out[i], dict.get_term(id), "mismatch at query position {i}");
        }
    }

    #[test]
    fn get_terms_into_empty_input() {
        let dict = AutomatonDictionary::with_block_bounds(2, 8);
        let mut out = vec![Some("stale".to_string())];
        dict.get_terms_into(&[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn get_terms_scattered_into_matches_get_terms_into_and_get_term() {
        let mut dict = AutomatonDictionary::with_block_bounds(2, 8);
        let terms: Vec<String> = (0..100).map(|i| format!("term-{i:04}")).collect();
        let ids: Vec<TermId> = terms.iter().map(|t| dict.insert(t)).collect();
        assert!(dict.block_count() > 1, "should span multiple blocks");

        let mut query: Vec<TermId> = vec![ids[50], ids[3], ids[99], ids[50], ids[0]];
        query.push(TermId(999_999)); // absent

        let mut dense = Vec::new();
        dict.get_terms_into(&query, &mut dense);
        let mut scattered = Vec::new();
        dict.get_terms_scattered_into(&query, &mut scattered);
        assert_eq!(scattered, dense);
        for (i, &id) in query.iter().enumerate() {
            assert_eq!(
                scattered[i],
                dict.get_term(id),
                "mismatch at query position {i}"
            );
        }
    }

    #[test]
    fn get_terms_scattered_into_empty_input() {
        let dict = AutomatonDictionary::with_block_bounds(2, 8);
        let mut out = vec![Some("stale".to_string())];
        dict.get_terms_scattered_into(&[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn get_terms_seeked_into_matches_individual_lookups_across_blocks() {
        let mut dict = AutomatonDictionary::with_block_bounds(2, 8);
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
        let mut dict = AutomatonDictionary::with_block_bounds(16, 64);
        let terms: Vec<String> = (0..1000)
            .flat_map(|i| {
                [
                    format!("http://example.org/entity-{i:05}"),
                    format!("http://xmlns.com/foaf/0.1/property{i}"),
                ]
            })
            .collect();
        let mut sorted_terms = terms.clone();
        sorted_terms.sort_unstable();
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
        let dict = AutomatonDictionary::with_block_bounds(2, 8);
        let mut out = vec![Some("stale".to_string())];
        dict.get_terms_seeked_into(&[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn size_bytes_grows_with_content_and_shrinks_on_remove() {
        let mut dict = AutomatonDictionary::with_block_bounds(4, 16);
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
}
