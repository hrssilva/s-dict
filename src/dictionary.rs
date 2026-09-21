use std::num::NonZeroU32;

use crate::batch::{BatchOrder, PreparedBatch};
use crate::block::Block;
use crate::entries_arena::EntriesArena;
use crate::{TermDictionary, TermId};

/// A stable handle for a block: an index into `Dictionary`'s block slab.
/// Stays valid across splits/merges of *other* blocks (design doc §10
/// "stable integer handles... physical movement caused by compaction
/// should not invalidate logical references"). Wraps a `NonZeroU32`
/// storing the slot index plus one, rather than a plain `u32`, so that
/// `Option<BlockId>` -- as used throughout `id_to_block`, which is sized
/// to the total term count rather than the block count -- costs 4 bytes
/// via Rust's niche-filling optimization instead of 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(NonZeroU32);

impl BlockId {
    fn from_slot(slot: u32) -> BlockId {
        BlockId(
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

/// Cumulative counts of rebuild work, for measuring how batch shape affects
/// rebuild frequency and decode volume (design doc H2/H5). `blocks_rebuilt`
/// counts every [`Block::build`] call (splits and merge-driven rebuilds
/// each count their constituent blocks separately); `entries_decoded`
/// counts every entry read back out of an existing block via `decode_into`
/// across all of those rebuilds. Neither counts the one-time initial
/// `bootstrap` of an empty dictionary, since that has no "old" block to
/// decode -- callers comparing configurations should snapshot
/// [`Dictionary::rebuild_stats`] after their initial load and diff against
/// a later snapshot to isolate steady-state insertion cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebuildStats {
    pub blocks_rebuilt: u64,
    pub entries_decoded: u64,
}

/// Design-doc Stages 1 and 3: terms partitioned into contiguous lexical
/// blocks bounded to `[block_min, block_max]` terms, each block a minimal
/// per-block FST (the "micro-FST" -- see `DESIGN.md`), routed by a
/// sorted-array global router and split/merged in place on
/// overflow/underflow (§7).
///
/// Since a block's FST is immutable, every mutation (`insert`, `remove`,
/// `split`, `merge_with_neighbor`, batch grouping) follows the same shape:
/// decode the affected block(s) into an [`EntriesArena`] scratch buffer,
/// mutate that, and call [`Block::build`] to replace the block wholesale
/// (design doc §6). This is O(block size) either way, same as Stage 1's
/// `Vec::insert` shifting was -- just with a different constant factor,
/// which is exactly what H2 asks to measure.
#[derive(Debug)]
pub struct Dictionary {
    blocks: Vec<Option<Block>>,
    free_block_slots: Vec<u32>,
    /// Sorted by `min_key`; the global router (§4). `directory[i].0` is the
    /// smallest term stored in `directory[i].1`; block 0 implicitly covers
    /// everything below its own min_key too, so no `-inf` sentinel is
    /// needed.
    directory: Vec<(Box<str>, BlockId)>,
    /// `TermId -> (BlockId, local_rank)` (§5 Variant B, chosen over
    /// Variant A's `TermId -> BlockId` + linear scan after measuring
    /// both: see DESIGN.md's "H3" section). `local_rank` is `u16`, not
    /// `usize`/`u32` -- bounded by `block_max`, which `with_block_bounds`
    /// asserts stays under `u16::MAX`.
    id_to_block: Vec<Option<(BlockId, u16)>>,
    free_term_ids: Vec<u32>,
    next_term_id: u32,
    block_min: usize,
    block_max: usize,
    len: usize,
    /// Two reusable scratch arenas for decode-mutate-rebuild block
    /// updates (design doc §6/§10). Taken out via `mem::take` at the
    /// start of a mutation and handed back as soon as their content is no
    /// longer needed (and always before any call that might itself want
    /// scratch space, e.g. a cascading split) -- so their backing byte
    /// buffers are reused across the whole sequence of single-term
    /// inserts/removes instead of being freed and reallocated every time.
    /// Two, not one, because e.g. an insert needs both "the block's
    /// existing content" and "that content merged with the new term" live
    /// at once.
    scratch_a: EntriesArena,
    scratch_b: EntriesArena,
    stats: RebuildStats,
}

impl Default for Dictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl Dictionary {
    /// Creates an empty dictionary with the default block bounds.
    pub fn new() -> Self {
        Self::with_block_bounds(DEFAULT_BLOCK_MIN, DEFAULT_BLOCK_MAX)
    }

    /// Creates an empty dictionary bounding each block to
    /// `[block_min, block_max]` terms. Block size is a first-class
    /// experimental parameter (design doc §12): tune it here rather than
    /// hardcoding one value.
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
        Dictionary {
            blocks: Vec::new(),
            free_block_slots: Vec::new(),
            directory: Vec::new(),
            id_to_block: Vec::new(),
            free_term_ids: Vec::new(),
            next_term_id: 0,
            block_min,
            block_max,
            len: 0,
            scratch_a: EntriesArena::new(),
            scratch_b: EntriesArena::new(),
            stats: RebuildStats::default(),
        }
    }

    /// The number of live blocks, for introspection/testing.
    pub fn block_count(&self) -> usize {
        self.directory.len()
    }

    /// Cumulative rebuild/decode counts since this dictionary was created
    /// (see [`RebuildStats`]). Snapshot before and after a phase of
    /// interest and diff the two to isolate that phase's cost -- e.g. to
    /// exclude an initial bootstrap load from a steady-state insertion
    /// measurement.
    pub fn rebuild_stats(&self) -> RebuildStats {
        self.stats
    }

    /// Total logical byte size: every block's compressed FST payload plus
    /// its `ids` array, the global router (`directory`), and the reverse
    /// directory (`id_to_block`) -- design doc §14 explicitly requires not
    /// reporting compressed-block size alone while excluding router/
    /// directory/id-array overhead. Does not account for allocator slack
    /// (`Vec` spare capacity) or heap allocation headers.
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
            .map(|(key, _)| key.len() + std::mem::size_of::<BlockId>())
            .sum();
        let id_to_block_bytes =
            self.id_to_block.len() * std::mem::size_of::<Option<(BlockId, u16)>>();
        blocks_bytes + directory_bytes + id_to_block_bytes
    }

    /// Insert a worker-prepared batch without sorting or deduplicating it again.
    /// Returns IDs in original input order, including duplicate occurrences.
    /// Preparation owns only one caller-bounded batch; no global term copy or
    /// dataset-wide uncompressed index is retained.
    pub fn insert_prepared(&mut self, prepared: PreparedBatch<'_>) -> Vec<TermId> {
        let batch = prepared.batch;
        self.insert_ordered(&batch.order, |u| batch.term(batch.order.first(u)))
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

        // Take once per batch; empty replacement arena allocates nothing.
        let mut merged = std::mem::take(&mut self.scratch_b);
        let mut u = 0;
        while u < order.unique_len() {
            let dir_idx = self.route(term(u)).expect("directory is non-empty");
            let mut run_end = u + 1;
            // Sorted input stays in this destination until the next minimum.
            // No repeated binary search for each term in the run.
            if let Some((next_min, _)) = self.directory.get(dir_idx + 1) {
                while run_end < order.unique_len() && term(run_end) < next_min.as_ref() {
                    run_end += 1;
                }
            } else {
                run_end = order.unique_len();
            }
            let block_id = self.directory[dir_idx].1;

            merged.clear();
            // Direct field access (not the `self.block(id)` method) so this
            // borrows only `self.blocks`, leaving `self.free_term_ids`/
            // `self.next_term_id` free for the id-allocation closure below --
            // both run inside one FST walk now, not two passes needing `self`
            // shared in incompatible ways.
            let block_ref = self.blocks[block_id.slot() as usize]
                .as_ref()
                .expect("BlockId must reference a live block");
            self.stats.entries_decoded += block_ref.len() as u64;

            let mut k = u;
            let free_term_ids = &mut self.free_term_ids;
            let next_term_id = &mut self.next_term_id;
            let outcome = block_ref.merge_into(
                (u..run_end).map(&term),
                &mut merged,
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
                self.rebuild_batch_block(dir_idx, &merged, outcome.unchanged_prefix);
                self.len += outcome.added;
            }
            u = run_end;
        }
        self.scratch_b = merged;
        result
    }

    /// Like [`TermDictionary::get_term`], but decodes into a caller-
    /// provided, reused byte buffer instead of allocating a fresh
    /// `String` every call -- for hot loops materializing many terms
    /// (design doc §10 reusable scratch, applied to reads as well as
    /// rebuilds). `buf` is cleared first. Returns `false` (leaving `buf`
    /// empty) if `id` isn't present; otherwise `buf` holds the term's
    /// UTF-8 bytes, interpreted via `std::str::from_utf8` (no
    /// allocation) rather than returned as a `String`.
    pub fn get_term_into(&self, id: TermId, buf: &mut Vec<u8>) -> bool {
        buf.clear();
        let Some((block_id, local_rank)) = self.id_to_block.get(id.0 as usize).and_then(|e| *e)
        else {
            return false;
        };
        self.block(block_id).term_at_into(local_rank as usize, buf);
        true
    }

    /// Decodes many ids in one call, grouped by block and decoded once per
    /// affected block (via the existing `decode_into`/`EntriesArena`
    /// streaming machinery) instead of one independent root-to-leaf FST
    /// walk per id -- see DESIGN.md's "Bulk decode" section for why and
    /// when this wins over repeated `get_term_into` calls (bulk/clustered
    /// access) versus when it doesn't (a few ids scattered across many
    /// blocks, where this still decodes each whole affected block).
    ///
    /// `ids` need not be sorted or deduplicated. `out` is cleared first
    /// and filled in the same order as `ids`; `None` marks an id not
    /// present.
    pub fn get_terms_into(&self, ids: &[TermId], out: &mut Vec<Option<String>>) {
        out.clear();
        out.resize(ids.len(), None);

        let mut by_block: std::collections::HashMap<BlockId, Vec<(usize, u16)>> =
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

    fn block(&self, id: BlockId) -> &Block {
        self.blocks[id.slot() as usize]
            .as_ref()
            .expect("BlockId must reference a live block")
    }

    fn block_mut(&mut self, id: BlockId) -> &mut Block {
        self.blocks[id.slot() as usize]
            .as_mut()
            .expect("BlockId must reference a live block")
    }

    fn alloc_block(&mut self, block: Block) -> BlockId {
        match self.free_block_slots.pop() {
            Some(slot) => {
                self.blocks[slot as usize] = Some(block);
                BlockId::from_slot(slot)
            }
            None => {
                let slot = self.blocks.len() as u32;
                self.blocks.push(Some(block));
                BlockId::from_slot(slot)
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

    fn set_id_to_block(&mut self, id: TermId, block_id: BlockId, local_rank: usize) {
        let idx = id.0 as usize;
        if idx >= self.id_to_block.len() {
            self.id_to_block.resize(idx + 1, None);
        }
        self.id_to_block[idx] = Some((block_id, local_rank as u16));
    }

    /// Reassigns `TermId -> (block_id, local_rank)` for every entry in
    /// `entries` (their position in iteration order *is* their new local
    /// rank). Variant B (§5) pays this on every block rebuild -- every
    /// insert/remove/split/merge changes the sorted-position local rank
    /// of every surviving entry in the rebuilt block(s), not just the one
    /// that changed -- in exchange for O(1) reverse lookup instead of
    /// Variant A's O(block size) scan. Cost is O(block size) array
    /// writes, which measurement shows is negligible next to the
    /// decode+rebuild cost that already dominates every one of these
    /// call sites (see DESIGN.md's "H3" section).
    /// `skip` leading entries are assumed already correct in
    /// `id_to_block` at their current position (see
    /// [`crate::block::MergeOutcome::unchanged_prefix`] and the
    /// `split`/`merge_with_neighbor` call sites, which know a prefix is
    /// untouched without needing a merge at all) and are not rewritten.
    fn assign_ranks<'a>(
        &mut self,
        block_id: BlockId,
        entries: impl Iterator<Item = (&'a str, TermId)>,
        skip: usize,
    ) {
        for (local_rank, (_, id)) in entries.enumerate().skip(skip) {
            self.set_id_to_block(id, block_id, local_rank);
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

    /// Replace one batch destination with final-sized, balanced blocks.
    /// The merged arena contains only this destination's old and incoming
    /// terms. Build each final FST once, without constructing an oversized
    /// FST and decoding/rebuilding it through recursive splits.
    ///
    /// `unchanged_prefix` (from the [`crate::block::MergeOutcome`] that
    /// produced `merged`) is only meaningful for chunk 0: it keeps the
    /// original `BlockId` and the same 0-based local ranks the block had
    /// before this rebuild, so a leading run of entries there can still be
    /// exactly where they were. Every later chunk gets a fresh `BlockId`
    /// and renumbers from 0 regardless, so no prefix of it can match what
    /// any entry had before -- `assign_ranks` always runs unskipped there.
    fn rebuild_batch_block(
        &mut self,
        dir_idx: usize,
        merged: &EntriesArena,
        unchanged_prefix: usize,
    ) {
        let total = merged.len();
        debug_assert!(total > 0);
        let chunks = total.div_ceil(self.block_max);
        let base = total / chunks;
        let extra = total % chunks;
        // For an overflowing block, block_max >= 2 * block_min guarantees
        // every balanced chunk is also at least block_min terms long.
        debug_assert!(chunks == 1 || base >= self.block_min);

        let original_id = self.directory[dir_idx].1;
        let mut new_directory = Vec::with_capacity(chunks - 1);
        let mut start = 0;
        for chunk in 0..chunks {
            let end = start + base + usize::from(chunk < extra);
            let block = Block::build(merged.range(start, end));
            self.stats.blocks_rebuilt += 1;
            let min_key: Box<str> = block.min_key().into();
            let block_id = if chunk == 0 {
                *self.block_mut(original_id) = block;
                self.directory[dir_idx].0 = min_key;
                original_id
            } else {
                let id = self.alloc_block(block);
                new_directory.push((min_key, id));
                id
            };
            let skip = if chunk == 0 {
                unchanged_prefix.min(end)
            } else {
                0
            };
            self.assign_ranks(block_id, merged.range(start, end), skip);
            start = end;
        }
        // Shift the router's tail once, rather than once per split.
        self.directory
            .splice(dir_idx + 1..dir_idx + 1, new_directory);
    }

    /// Repeatedly splits the block at `start_dir_idx` (and any newly
    /// created sibling) until every resulting block is within `block_max`.
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

    /// Median-splits the block at `dir_idx` into two freshly built blocks,
    /// inserting a new directory entry for the right half (design doc §7).
    fn split(&mut self, dir_idx: usize) {
        let block_id = self.directory[dir_idx].1;
        let mut arena = std::mem::take(&mut self.scratch_a);
        arena.clear();
        self.block(block_id).decode_into(&mut arena);
        self.stats.entries_decoded += arena.len() as u64;
        let mid = arena.len() / 2;

        let left_block = Block::build(arena.range(0, mid));
        let right_block = Block::build(arena.range(mid, arena.len()));
        self.stats.blocks_rebuilt += 2;
        let right_min: Box<str> = right_block.min_key().into();

        // The left half keeps `block_id` *and* exactly the same 0-based
        // local ranks it already had (it's `arena`'s first `mid` entries,
        // in the same order, under the same id) -- id_to_block is already
        // correct for every one of them, so there's nothing to reassign.
        *self.block_mut(block_id) = left_block;

        let right_block_id = self.alloc_block(right_block);
        self.assign_ranks(right_block_id, arena.range(mid, arena.len()), 0);

        self.scratch_a = arena;

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
    /// adjacent neighbor (design doc §7), splitting the result back apart
    /// if it now exceeds `block_max`.
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
        let keep_len = self.block(keep_block_id).len();

        // Blocks partition sorted space contiguously and keep_idx < absorb_idx,
        // so `keep`'s entries are already all lexically smaller: decoding
        // both into the same arena back-to-back (no clear in between)
        // concatenates them in sorted order for free.
        let mut arena = std::mem::take(&mut self.scratch_a);
        arena.clear();
        self.block(keep_block_id).decode_into(&mut arena);
        self.block(absorb_block_id).decode_into(&mut arena);
        self.stats.entries_decoded += arena.len() as u64;

        let rebuilt = Block::build(arena.iter());
        self.stats.blocks_rebuilt += 1;
        *self.block_mut(keep_block_id) = rebuilt;
        // `keep`'s own entries stay at `keep_block_id`, in the same order,
        // at the same 0-based ranks they already had (they were decoded
        // first, into a fresh arena) -- only the absorbed half, appended
        // after them, is actually new at this BlockId and needs assigning.
        self.assign_ranks(keep_block_id, arena.iter(), keep_len);

        self.scratch_a = arena;

        self.blocks[absorb_block_id.slot() as usize] = None;
        self.free_block_slots.push(absorb_block_id.slot());
        self.directory.remove(absorb_idx);

        self.split_while_oversized(keep_idx);
    }

    /// Assigns fresh ids to an initial sorted, unique batch of terms and
    /// chunks them directly into balanced `<= block_max`-sized blocks
    /// (evening out the remainder rather than leaving a small trailing
    /// chunk), bypassing the single-term bootstrap path entirely.
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

            let block = Block::build((start..end).map(term).zip(chunk_ids.iter().copied()));
            self.stats.blocks_rebuilt += 1;
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
}

impl TermDictionary for Dictionary {
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
            let block = Block::build(std::iter::once((term, id)));
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

        let mut merged = std::mem::take(&mut self.scratch_b);
        merged.clear();
        // Same single-walk primitive as the batch path: one FST stream
        // finds the miss (already known from the `rank` check above, but
        // `merge_into` re-derives it as a side effect of producing the
        // rebuilt content) and allocates its id in the same pass, instead
        // of a separate decode-then-merge.
        let block_ref = self.blocks[block_id.slot() as usize]
            .as_ref()
            .expect("BlockId must reference a live block");
        self.stats.entries_decoded += block_ref.len() as u64;
        let free_term_ids = &mut self.free_term_ids;
        let next_term_id = &mut self.next_term_id;
        let mut id = None;
        let outcome = block_ref.merge_into(
            std::iter::once(term),
            &mut merged,
            || match free_term_ids.pop() {
                Some(slot) => TermId(slot),
                None => {
                    let new_id = TermId(*next_term_id);
                    *next_term_id += 1;
                    new_id
                }
            },
            |resolved_id| id = Some(resolved_id),
        );
        let id = id.expect("the single incoming term always resolves to exactly one id");

        let rebuilt = Block::build(merged.iter());
        self.stats.blocks_rebuilt += 1;
        self.directory[dir_idx].0 = rebuilt.min_key().into();
        *self.block_mut(block_id) = rebuilt;
        self.assign_ranks(block_id, merged.iter(), outcome.unchanged_prefix);

        self.scratch_b = merged;

        self.len += 1;
        self.split_while_oversized(dir_idx);
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
        let local_rank = self.block(block_id).rank(term)?;
        let id = self.block(block_id).id_at(local_rank);

        let mut old = std::mem::take(&mut self.scratch_a);
        old.clear();
        self.block(block_id).decode_into(&mut old);
        self.stats.entries_decoded += old.len() as u64;

        self.id_to_block[id.0 as usize] = None;
        self.free_term_ids.push(id.0);
        self.len -= 1;

        if old.len() == 1 {
            // The only entry was the one just removed.
            self.scratch_a = old;
            self.remove_empty_block(dir_idx);
        } else {
            let mut filtered = std::mem::take(&mut self.scratch_b);
            filtered.clear();
            for i in 0..old.len() {
                if i != local_rank {
                    filtered.push(old.term_at(i), old.id_at(i));
                }
            }

            let rebuilt = Block::build(filtered.iter());
            self.stats.blocks_rebuilt += 1;
            self.directory[dir_idx].0 = rebuilt.min_key().into();
            let remaining = rebuilt.len();
            *self.block_mut(block_id) = rebuilt;
            // Entries before the removed rank keep the same index they
            // already had (nothing before it moved); only entries from
            // the removed position onward shifted down by one.
            self.assign_ranks(block_id, filtered.iter(), local_rank);

            self.scratch_a = old;
            self.scratch_b = filtered;

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
    use std::time::Instant;

    /// Isolates `assign_ranks`'s own cost with vs. without the
    /// `unchanged_prefix` skip, on a fixed 256-entry block (the default
    /// `block_max`) with a fresh id-to-block insertion at the midpoint --
    /// the average case for a single insert into an already-full block,
    /// skipping about half the writes. Run alongside `perf stat` on
    /// `examples/perf_insert_single_flush.rs` for the full-pipeline
    /// picture: this isolates whether the skip is worth anything at all,
    /// separate from the much larger, noisier per-insert total that also
    /// includes `Block::build`'s FST construction.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_assign_ranks_skip() {
        const N: usize = 256;
        const ITERS: usize = 200_000;

        let entries: Vec<(String, TermId)> = (0..N)
            .map(|i| (format!("http://example.org/term-{i:06}"), TermId(i as u32)))
            .collect();
        let block_id = BlockId::from_slot(0);

        let mut dict = Dictionary::with_block_bounds(2, N);
        let start = Instant::now();
        for _ in 0..ITERS {
            dict.assign_ranks(block_id, entries.iter().map(|(t, id)| (t.as_str(), *id)), 0);
        }
        let no_skip_elapsed = start.elapsed();

        let mut dict = Dictionary::with_block_bounds(2, N);
        let start = Instant::now();
        for _ in 0..ITERS {
            dict.assign_ranks(
                block_id,
                entries.iter().map(|(t, id)| (t.as_str(), *id)),
                N / 2,
            );
        }
        let half_skip_elapsed = start.elapsed();

        eprintln!(
            "assign_ranks on a {N}-entry block, {ITERS} iters:\n\
             skip=0 (current behavior before this change):    {no_skip_elapsed:?} ({:?}/op)\n\
             skip=N/2 (average case for one mid-block insert): {half_skip_elapsed:?} ({:?}/op)",
            no_skip_elapsed / ITERS as u32,
            half_skip_elapsed / ITERS as u32,
        );
    }

    #[test]
    fn insert_is_idempotent() {
        let mut dict = Dictionary::with_block_bounds(2, 8);
        let a = dict.insert("http://example.org/a");
        let b = dict.insert("http://example.org/a");
        assert_eq!(a, b);
        assert_eq!(dict.len(), 1);
    }

    #[test]
    fn round_trips() {
        let mut dict = Dictionary::with_block_bounds(2, 8);
        let id = dict.insert("http://example.org/a");
        assert_eq!(dict.get_term(id).as_deref(), Some("http://example.org/a"));
        assert_eq!(dict.get_id("http://example.org/a"), Some(id));
    }

    #[test]
    fn get_term_into_matches_get_term() {
        let mut dict = Dictionary::with_block_bounds(2, 8);
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
    fn get_terms_into_matches_individual_lookups_across_blocks() {
        let mut dict = Dictionary::with_block_bounds(2, 8);
        let terms: Vec<String> = (0..100).map(|i| format!("term-{i:04}")).collect();
        let ids: Vec<TermId> = terms.iter().map(|t| dict.insert(t)).collect();
        assert!(dict.block_count() > 1, "should span multiple blocks");

        // Arbitrary order, with a duplicate and a missing id mixed in.
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
        let dict = Dictionary::with_block_bounds(2, 8);
        let mut out = vec![Some("stale".to_string())];
        dict.get_terms_into(&[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn remove_absent_is_none() {
        let mut dict = Dictionary::with_block_bounds(2, 8);
        assert_eq!(dict.remove("nope"), None);
    }

    #[test]
    fn empty_dictionary_round_trip() {
        let mut dict = Dictionary::with_block_bounds(2, 8);
        assert!(dict.is_empty());
        let id = dict.insert("a");
        assert_eq!(dict.remove("a"), Some(id));
        assert!(dict.is_empty());
        assert_eq!(dict.block_count(), 0);
        assert_eq!(dict.get_term(id), None);
    }

    #[test]
    fn split_and_merge_across_many_terms() {
        let mut dict = Dictionary::with_block_bounds(2, 8);
        let terms: Vec<String> = (0..100).map(|i| format!("term-{i:04}")).collect();
        let mut ids = Vec::new();
        for t in &terms {
            ids.push(dict.insert(t));
        }
        assert!(
            dict.block_count() > 1,
            "100 terms with block_max=8 should split"
        );
        for (t, &id) in terms.iter().zip(&ids) {
            assert_eq!(dict.get_id(t), Some(id));
            assert_eq!(dict.get_term(id).as_deref(), Some(t.as_str()));
        }
        for t in &terms {
            dict.remove(t);
        }
        assert!(dict.is_empty());
        assert_eq!(dict.block_count(), 0);
    }

    #[test]
    fn batch_insert_matches_sequential() {
        let mut sequential = Dictionary::with_block_bounds(2, 8);
        let mut batched = Dictionary::with_block_bounds(2, 8);
        let terms: Vec<String> = (0..50).map(|i| format!("t{i:03}")).collect();
        let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();

        for t in &term_refs {
            sequential.insert(t);
        }
        batched.insert_batch(&term_refs);

        assert_eq!(sequential.len(), batched.len());
        for t in &term_refs {
            let a = sequential.get_id(t);
            let b = batched.get_id(t);
            assert!(a.is_some() && b.is_some());
            assert_eq!(
                sequential.get_term(a.unwrap()),
                batched.get_term(b.unwrap())
            );
        }
    }

    #[test]
    fn batch_insert_with_duplicates_returns_consistent_ids() {
        let mut dict = Dictionary::with_block_bounds(2, 8);
        let terms = ["a", "b", "a", "c", "b"];
        let ids = dict.insert_batch(&terms);
        assert_eq!(ids[0], ids[2]); // both "a"
        assert_eq!(ids[1], ids[4]); // both "b"
        assert_eq!(dict.len(), 3);
    }

    #[test]
    fn large_initial_batch_stays_balanced() {
        let mut dict = Dictionary::with_block_bounds(2, 8);
        let terms: Vec<String> = (0..17).map(|i| format!("x{i:03}")).collect();
        let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        dict.insert_batch(&term_refs);
        // 17 terms, block_max=8 -> 3 chunks of ~6, none tiny.
        assert_eq!(dict.block_count(), 3);
        for t in &term_refs {
            assert!(dict.get_id(t).is_some());
        }
    }

    #[test]
    fn large_incremental_batch_builds_balanced_blocks_and_preserves_ids() {
        // Exercise both the minimum allowed max/min ratio and odd maxima,
        // including batches leaving a one-term remainder before balancing.
        for (min, max) in [(1, 2), (2, 4), (2, 5), (3, 8)] {
            for total in [max, max + 1, max * 8 + 1] {
                let mut dict = Dictionary::with_block_bounds(min, max);
                let terms: Vec<String> = (0..total).map(|i| format!("t{i:04}")).collect();
                let old_id = dict.insert(&terms[total / 2]);
                let original_block = dict.directory[0].1;
                let mut refs: Vec<&str> = terms.iter().rev().map(String::as_str).collect();
                refs.push(&terms[total / 2]);
                let ids = dict.insert_batch(&refs);
                assert_eq!(dict.len(), total);
                assert_eq!(dict.block_count(), total.div_ceil(max));
                assert_eq!(dict.directory[0].1, original_block);
                assert_eq!(dict.get_id(&terms[total / 2]), Some(old_id));
                for (&term, &id) in refs.iter().zip(&ids) {
                    assert_eq!(dict.get_id(term), Some(id));
                    assert_eq!(dict.get_term(id).as_deref(), Some(term));
                }
                let sizes: Vec<_> = dict
                    .directory
                    .iter()
                    .map(|(_, id)| dict.block(*id).len())
                    .collect();
                assert!(sizes.iter().all(|&n| n >= min && n <= max));
                assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
            }
        }
    }

    #[test]
    fn incremental_batches_route_across_expanding_blocks_and_reuse_slots() {
        let mut dict = Dictionary::with_block_bounds(2, 4);
        let seed = ["b", "d", "f", "h", "j", "l", "n", "p", "r", "t", "v", "x"];
        let seed_ids = dict.insert_batch(&seed);
        // Free a block slot and some term IDs before the bulk update.
        for term in &seed[4..] {
            dict.remove(term);
        }
        assert!(!dict.free_block_slots.is_empty());
        let terms: Vec<String> = ["a", "c", "e", "g", "i", "z"]
            .into_iter()
            .flat_map(|prefix| (0..17).map(move |i| format!("{prefix}{i:03}")))
            .collect();
        let mut refs: Vec<&str> = terms.iter().rev().map(String::as_str).collect();
        refs.extend_from_slice(&seed[..4]);
        refs.extend_from_slice(&["a000", "z016", "b"]);
        let ids = dict.insert_batch(&refs);
        assert_eq!(dict.len(), terms.len() + 4);
        for (&term, &id) in refs.iter().zip(&ids) {
            assert_eq!(dict.get_id(term), Some(id));
            assert_eq!(dict.get_term(id).as_deref(), Some(term));
        }
        for (&term, &id) in seed[..4].iter().zip(&seed_ids) {
            assert_eq!(dict.get_id(term), Some(id));
        }
        for pair in dict.directory.windows(2) {
            assert!(pair[0].0 < pair[1].0);
        }
        for (key, id) in &dict.directory {
            let block = dict.block(*id);
            assert_eq!(key.as_ref(), block.min_key());
            assert!((2..=4).contains(&block.len()));
        }
        assert_eq!(dict.insert_batch(&refs), ids);
        // Later ordinary mutations must still find and update all mappings.
        for term in terms
            .iter()
            .map(String::as_str)
            .chain(seed[..4].iter().copied())
        {
            let id = dict.get_id(term).unwrap();
            assert_eq!(dict.remove(term), Some(id));
            assert_eq!(dict.get_term(id), None);
        }
        assert!(dict.is_empty());
        let id = dict.insert("after");
        assert_eq!(dict.get_term(id).as_deref(), Some("after"));
    }

    #[test]
    fn block_id_niche_optimization_holds() {
        assert_eq!(
            std::mem::size_of::<Option<BlockId>>(),
            std::mem::size_of::<BlockId>()
        );
    }

    #[test]
    fn id_to_block_entry_niche_optimization_holds() {
        // The actual `id_to_block` element type: BlockId's NonZeroU32
        // niche should still let Option<(BlockId, u16)> avoid a separate
        // discriminant, landing at 8 bytes (4 + 2, padded) rather than 12.
        assert_eq!(std::mem::size_of::<Option<(BlockId, u16)>>(), 8);
    }

    #[test]
    fn get_id_and_get_term_are_consistent_after_many_splits_and_merges() {
        // Exercises Variant B's per-rebuild rank reassignment specifically:
        // insert enough to force several splits, remove enough to force
        // merges, and confirm every surviving term's id_to_block entry
        // still points at the right (block, local_rank) throughout.
        let mut dict = Dictionary::with_block_bounds(2, 8);
        let terms: Vec<String> = (0..200).map(|i| format!("k{i:04}")).collect();
        let mut ids = Vec::new();
        for t in &terms {
            ids.push(dict.insert(t));
        }
        for t in terms.iter().step_by(3) {
            dict.remove(t);
        }
        for (t, &id) in terms.iter().zip(&ids) {
            let still_present = dict.get_id(t).is_some();
            if still_present {
                assert_eq!(dict.get_id(t), Some(id));
                assert_eq!(dict.get_term(id).as_deref(), Some(t.as_str()));
            }
        }
    }
}
