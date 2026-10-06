//! `TermId -> (block slot, local rank)`, packed into one `u32` per id.
//!
//! Replaces `Vec<Option<(BlockId, u16)>>` (8 bytes per id after padding)
//! in `Dictionary` and `PackedAutomatonDictionary`: the reverse map was
//! the largest fixed per-term cost in both (DESIGN.md's "packing the
//! reverse map"). The rank gets just enough bits for twice `block_max`
//! -- blocks briefly exceed `block_max` between a merge and its split
//! (by at most `block_min <= block_max / 2` in a neighbor merge, by one
//! term in a single insert), and `Dictionary::split` keeps the left
//! half's already-assigned ranks -- and the slot gets the rest.
//! `u32::MAX` marks an unassigned or removed id.

const NONE: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub(crate) struct ReverseMap {
    entries: Vec<u32>,
    rank_bits: u32,
}

impl ReverseMap {
    pub(crate) fn new(block_max: usize) -> ReverseMap {
        let max_rank = 2 * block_max - 1;
        let rank_bits = usize::BITS - max_rank.leading_zeros();
        assert!(rank_bits < 32, "block_max too large for a packed reverse map");
        ReverseMap {
            entries: Vec::new(),
            rank_bits,
        }
    }

    /// `(block slot, local rank)` for `id`, if assigned.
    #[inline]
    pub(crate) fn get(&self, id: u32) -> Option<(u32, usize)> {
        let entry = *self.entries.get(id as usize)?;
        (entry != NONE).then(|| {
            (entry >> self.rank_bits, (entry & ((1 << self.rank_bits) - 1)) as usize)
        })
    }

    pub(crate) fn set(&mut self, id: u32, slot: u32, rank: usize) {
        assert!(
            rank < 1 << self.rank_bits,
            "local rank {rank} exceeds the reverse map's {} rank bits",
            self.rank_bits
        );
        let entry = (u64::from(slot) << self.rank_bits) | rank as u64;
        assert!(entry < u64::from(NONE), "block slot {slot} exceeds the reverse map's slot bits");
        let idx = id as usize;
        if idx >= self.entries.len() {
            self.entries.resize(idx + 1, NONE);
        }
        self.entries[idx] = entry as u32;
    }

    pub(crate) fn clear(&mut self, id: u32) {
        if let Some(entry) = self.entries.get_mut(id as usize) {
            *entry = NONE;
        }
    }

    /// Logical size, as counted by the dictionaries' `size_bytes`.
    pub(crate) fn size_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<u32>()
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.entries.shrink_to_fit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_bounds() {
        let mut map = ReverseMap::new(256);
        assert_eq!(map.rank_bits, 9);
        map.set(5, 0, 0);
        map.set(2, (1 << 23) - 2, 511);
        assert_eq!(map.get(5), Some((0, 0)));
        assert_eq!(map.get(2), Some(((1 << 23) - 2, 511)));
        assert_eq!(map.get(3), None);
        assert_eq!(map.get(99), None);
        map.clear(5);
        assert_eq!(map.get(5), None);
        assert_eq!(map.size_bytes(), 6 * 4);
    }

    #[test]
    #[should_panic(expected = "rank bits")]
    fn oversized_rank_panics() {
        ReverseMap::new(256).set(0, 0, 1024);
    }
}
