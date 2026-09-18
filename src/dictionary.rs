use crate::block::Block;
use crate::{TermDictionary, TermId};

/// A stable handle for a block: an index into `Dictionary`'s block slab.
/// Stays valid across splits/merges of *other* blocks (design doc §10
/// "stable integer handles... physical movement caused by compaction
/// should not invalidate logical references").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(u32);

const DEFAULT_BLOCK_MIN: usize = 64;
const DEFAULT_BLOCK_MAX: usize = 256;

/// Design-doc Stages 1 and 3: terms partitioned into contiguous lexical
/// blocks bounded to `[block_min, block_max]` terms, each block a minimal
/// per-block FST (the "micro-FST" -- see `DESIGN.md`), routed by a
/// sorted-array global router and split/merged in place on
/// overflow/underflow (§7).
///
/// Since a block's FST is immutable, every mutation (`insert`, `remove`,
/// `split`, `merge_with_neighbor`, batch grouping) follows the same shape:
/// decode the affected block(s) to a plain sorted `Vec`, mutate that Vec,
/// and call [`Block::build`] to replace the block wholesale (design doc
/// §6). This is O(block size) either way, same as Stage 1's `Vec::insert`
/// shifting was -- just with a different constant factor, which is exactly
/// what H2 asks to measure.
#[derive(Debug)]
pub struct Dictionary {
    blocks: Vec<Option<Block>>,
    free_block_slots: Vec<u32>,
    /// Sorted by `min_key`; the global router (§4). `directory[i].0` is the
    /// smallest term stored in `directory[i].1`; block 0 implicitly covers
    /// everything below its own min_key too, so no `-inf` sentinel is
    /// needed.
    directory: Vec<(Box<str>, BlockId)>,
    /// `TermId -> BlockId` (§5 Variant A); local rank is recovered with a
    /// linear scan of the owning block's id array.
    id_to_block: Vec<Option<BlockId>>,
    free_term_ids: Vec<u32>,
    next_term_id: u32,
    block_min: usize,
    block_max: usize,
    len: usize,
    /// Reusable scratch buffer for decode-mutate-rebuild block updates
    /// (design doc §6/§10). Taken out via `mem::take` at the start of a
    /// mutation, cleared and handed back as soon as its content is no
    /// longer needed (and always before any call that might itself want
    /// the scratch buffer, e.g. a cascading split) -- so its backing
    /// allocation is reused across the whole sequence of single-term
    /// inserts/removes instead of being freed and reallocated every time.
    entries_scratch: Vec<(Box<str>, TermId)>,
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
            entries_scratch: Vec::new(),
        }
    }

    /// The number of live blocks, for introspection/testing.
    pub fn block_count(&self) -> usize {
        self.directory.len()
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
        let id_to_block_bytes = self.id_to_block.len() * std::mem::size_of::<Option<BlockId>>();
        blocks_bytes + directory_bytes + id_to_block_bytes
    }

    fn block(&self, id: BlockId) -> &Block {
        self.blocks[id.0 as usize]
            .as_ref()
            .expect("BlockId must reference a live block")
    }

    fn block_mut(&mut self, id: BlockId) -> &mut Block {
        self.blocks[id.0 as usize]
            .as_mut()
            .expect("BlockId must reference a live block")
    }

    fn alloc_block(&mut self, block: Block) -> BlockId {
        match self.free_block_slots.pop() {
            Some(slot) => {
                self.blocks[slot as usize] = Some(block);
                BlockId(slot)
            }
            None => {
                let id = BlockId(self.blocks.len() as u32);
                self.blocks.push(Some(block));
                id
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

    fn set_id_to_block(&mut self, id: TermId, block_id: BlockId) {
        let idx = id.0 as usize;
        if idx >= self.id_to_block.len() {
            self.id_to_block.resize(idx + 1, None);
        }
        self.id_to_block[idx] = Some(block_id);
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
        let mut entries = std::mem::take(&mut self.entries_scratch);
        self.block(block_id).decode_into(&mut entries);
        let mid = entries.len() / 2;

        let left_block = Block::build(&entries[..mid]);
        let right_block = Block::build(&entries[mid..]);
        let right_min: Box<str> = right_block.min_key().into();
        let moved_ids: Vec<TermId> = entries[mid..].iter().map(|(_, id)| *id).collect();

        entries.clear();
        self.entries_scratch = entries;

        *self.block_mut(block_id) = left_block;
        let right_block_id = self.alloc_block(right_block);
        for id in moved_ids {
            self.set_id_to_block(id, right_block_id);
        }
        self.directory
            .insert(dir_idx + 1, (right_min, right_block_id));
    }

    /// Removes the (already empty) directory entry and frees its block
    /// slot. Routing falls through to the predecessor block automatically.
    fn remove_empty_block(&mut self, dir_idx: usize) {
        let (_, block_id) = self.directory.remove(dir_idx);
        self.blocks[block_id.0 as usize] = None;
        self.free_block_slots.push(block_id.0);
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

        // Blocks partition sorted space contiguously and keep_idx < absorb_idx,
        // so `keep`'s entries are already all lexically smaller: a plain
        // concatenation stays sorted, no merge needed.
        let mut entries = std::mem::take(&mut self.entries_scratch);
        self.block(keep_block_id).decode_into(&mut entries);
        let absorbed_entries = self.block(absorb_block_id).decode_all();
        let absorbed_ids: Vec<TermId> = absorbed_entries.iter().map(|(_, id)| *id).collect();
        entries.extend(absorbed_entries);

        let rebuilt = Block::build(&entries);
        entries.clear();
        self.entries_scratch = entries;

        *self.block_mut(keep_block_id) = rebuilt;
        for id in absorbed_ids {
            self.set_id_to_block(id, keep_block_id);
        }

        self.blocks[absorb_block_id.0 as usize] = None;
        self.free_block_slots.push(absorb_block_id.0);
        self.directory.remove(absorb_idx);

        self.split_while_oversized(keep_idx);
    }

    /// Assigns fresh ids to an initial sorted, unique batch of terms and
    /// chunks them directly into balanced `<= block_max`-sized blocks
    /// (evening out the remainder rather than leaving a small trailing
    /// chunk), bypassing the single-term bootstrap path entirely.
    fn bootstrap(
        &mut self,
        unique_terms: &[&str],
        positions: &[Vec<usize>],
        result: &mut [TermId],
    ) {
        let total = unique_terms.len();
        let num_chunks = total.div_ceil(self.block_max).max(1);
        let base = total / num_chunks;
        let extra = total % num_chunks;

        let mut start = 0;
        for chunk_idx in 0..num_chunks {
            let size = base + if chunk_idx < extra { 1 } else { 0 };
            let end = start + size;

            let mut chunk_entries: Vec<(Box<str>, TermId)> = Vec::with_capacity(end - start);
            for (offset, &term) in unique_terms[start..end].iter().enumerate() {
                let id = self.alloc_term_id();
                chunk_entries.push((term.into(), id));
                for &pos in &positions[start + offset] {
                    result[pos] = id;
                }
            }

            let block = Block::build(&chunk_entries);
            let min_key: Box<str> = block.min_key().into();
            let block_id = self.alloc_block(block);
            for (_, id) in &chunk_entries {
                self.set_id_to_block(*id, block_id);
            }
            self.directory.push((min_key, block_id));
            self.len += end - start;
            start = end;
        }
    }
}

/// Merges two sorted, mutually disjoint `(term, id)` lists into one sorted
/// list in a single linear pass, instead of resorting the concatenation.
fn merge_sorted_entries(
    old: Vec<(Box<str>, TermId)>,
    new: Vec<(Box<str>, TermId)>,
) -> Vec<(Box<str>, TermId)> {
    let mut merged = Vec::with_capacity(old.len() + new.len());
    let mut old_iter = old.into_iter();
    let mut new_iter = new.into_iter();
    let mut old_next = old_iter.next();
    let mut new_next = new_iter.next();

    loop {
        match (&old_next, &new_next) {
            (Some((ot, _)), Some((nt, _))) => {
                if ot.as_ref() <= nt.as_ref() {
                    merged.push(old_next.take().unwrap());
                    old_next = old_iter.next();
                } else {
                    merged.push(new_next.take().unwrap());
                    new_next = new_iter.next();
                }
            }
            (Some(_), None) => {
                merged.push(old_next.take().unwrap());
                old_next = old_iter.next();
            }
            (None, Some(_)) => {
                merged.push(new_next.take().unwrap());
                new_next = new_iter.next();
            }
            (None, None) => break,
        }
    }

    merged
}

impl TermDictionary for Dictionary {
    fn get_id(&self, term: &str) -> Option<TermId> {
        let dir_idx = self.route(term)?;
        let block_id = self.directory[dir_idx].1;
        let local_rank = self.block(block_id).rank(term)?;
        Some(self.block(block_id).id_at(local_rank))
    }

    fn get_term(&self, id: TermId) -> Option<String> {
        let block_id = self.id_to_block.get(id.0 as usize)?.as_ref().copied()?;
        let block = self.block(block_id);
        let local_rank = block.local_rank_of(id)?;
        Some(block.term_at(local_rank))
    }

    fn insert(&mut self, term: &str) -> TermId {
        if self.directory.is_empty() {
            let id = self.alloc_term_id();
            let block = Block::build(&[(term.into(), id)]);
            let block_id = self.alloc_block(block);
            self.directory.push((term.into(), block_id));
            self.set_id_to_block(id, block_id);
            self.len += 1;
            return id;
        }

        let dir_idx = self.route(term).expect("directory is non-empty");
        let block_id = self.directory[dir_idx].1;

        if let Some(local_rank) = self.block(block_id).rank(term) {
            return self.block(block_id).id_at(local_rank);
        }

        let id = self.alloc_term_id();
        let mut entries = std::mem::take(&mut self.entries_scratch);
        self.block(block_id).decode_into(&mut entries);
        let insert_at = entries.partition_point(|(t, _)| t.as_ref() < term);
        entries.insert(insert_at, (term.into(), id));

        let rebuilt = Block::build(&entries);
        entries.clear();
        self.entries_scratch = entries;

        self.directory[dir_idx].0 = rebuilt.min_key().into();
        *self.block_mut(block_id) = rebuilt;
        self.set_id_to_block(id, block_id);
        self.len += 1;
        self.split_while_oversized(dir_idx);
        id
    }

    fn insert_batch(&mut self, terms: &[&str]) -> Vec<TermId> {
        if terms.is_empty() {
            return Vec::new();
        }

        // Group by destination up front instead of routing (and rebuilding
        // a block) once per term (design doc §10/H5).
        let mut order: Vec<usize> = (0..terms.len()).collect();
        order.sort_by(|&a, &b| terms[a].cmp(terms[b]));

        let mut unique_terms: Vec<&str> = Vec::new();
        let mut positions: Vec<Vec<usize>> = Vec::new();
        let mut i = 0;
        while i < order.len() {
            let term = terms[order[i]];
            let mut group = vec![order[i]];
            let mut j = i + 1;
            while j < order.len() && terms[order[j]] == term {
                group.push(order[j]);
                j += 1;
            }
            unique_terms.push(term);
            positions.push(group);
            i = j;
        }

        let mut result = vec![TermId(0); terms.len()];

        if self.directory.is_empty() {
            self.bootstrap(&unique_terms, &positions, &mut result);
            return result;
        }

        let mut u = 0;
        while u < unique_terms.len() {
            let dir_idx = self.route(unique_terms[u]).expect("directory is non-empty");
            let mut run_end = u + 1;
            while run_end < unique_terms.len() && self.route(unique_terms[run_end]) == Some(dir_idx)
            {
                run_end += 1;
            }

            let block_id = self.directory[dir_idx].1;
            let mut new_entries: Vec<(Box<str>, TermId)> = Vec::new();
            for k in u..run_end {
                let term = unique_terms[k];
                match self.block(block_id).rank(term) {
                    Some(local_rank) => {
                        let id = self.block(block_id).id_at(local_rank);
                        for &pos in &positions[k] {
                            result[pos] = id;
                        }
                    }
                    None => {
                        let id = self.alloc_term_id();
                        for &pos in &positions[k] {
                            result[pos] = id;
                        }
                        new_entries.push((term.into(), id));
                    }
                }
            }

            if !new_entries.is_empty() {
                let added = new_entries.len();
                let old_entries = self.block(block_id).decode_all();
                let merged = merge_sorted_entries(old_entries, new_entries);

                let rebuilt = Block::build(&merged);
                self.directory[dir_idx].0 = rebuilt.min_key().into();
                *self.block_mut(block_id) = rebuilt;
                self.len += added;
                self.split_while_oversized(dir_idx);
            }

            u = run_end;
        }

        result
    }

    fn remove(&mut self, term: &str) -> Option<TermId> {
        let dir_idx = self.route(term)?;
        let block_id = self.directory[dir_idx].1;
        let local_rank = self.block(block_id).rank(term)?;
        let id = self.block(block_id).id_at(local_rank);

        let mut entries = std::mem::take(&mut self.entries_scratch);
        self.block(block_id).decode_into(&mut entries);
        entries.remove(local_rank);

        self.id_to_block[id.0 as usize] = None;
        self.free_term_ids.push(id.0);
        self.len -= 1;

        if entries.is_empty() {
            entries.clear();
            self.entries_scratch = entries;
            self.remove_empty_block(dir_idx);
        } else {
            let rebuilt = Block::build(&entries);
            entries.clear();
            self.entries_scratch = entries;

            self.directory[dir_idx].0 = rebuilt.min_key().into();
            let remaining = rebuilt.len();
            *self.block_mut(block_id) = rebuilt;
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
}
