//! `TermId -> block slot`, compressed in 64-id chunks: the
//! `PackedAutomatonDictionary` reverse map.
//!
//! Unlike [`crate::reverse_map::ReverseMap`] this stores no ranks: a block
//! answers `id -> rank` itself (`BlockIds::rank_of`), from an inverse it
//! rebuilds whenever it re-encodes its ids. Ranks shift on every merge;
//! the block an id lives in changes only for new ids, splits and neighbor
//! merges -- so this map is written far less often, and is smaller
//! (DESIGN.md's "ranks out of the reverse map").
//!
//! Chunk layout (`u64` words in one shared pool): word 0 the live bitmask,
//! word 1 `slot_count | index_bits << 8`, then `slot_count` slots two per
//! word, then 64 slot indices of `index_bits` each. Lookups are O(1);
//! writes rewrite whole chunks, so [`SlotMap::assign`] batches them. Pool
//! handling as in `ReverseMap`: in place when it fits, else appended;
//! garbage compacted past half the pool and in `shrink_to_fit`.

const CHUNK: usize = 64;
const NO_CHUNK: u32 = u32::MAX;

#[derive(Debug, Clone, Default)]
pub(crate) struct SlotMap {
    pool: Vec<u64>,
    offsets: Vec<u32>,
    caps: Vec<u8>,
    garbage: usize,
}

fn bits(v: u64) -> u32 {
    u64::BITS - v.leading_zeros()
}

impl SlotMap {
    pub(crate) fn new() -> SlotMap {
        SlotMap::default()
    }

    fn chunk_words(&self, chunk: usize) -> Option<&[u64]> {
        let off = *self.offsets.get(chunk)?;
        (off != NO_CHUNK).then(|| &self.pool[off as usize..off as usize + self.caps[chunk] as usize])
    }

    /// The block slot holding `id`, if assigned.
    #[inline]
    pub(crate) fn get(&self, id: u32) -> Option<u32> {
        let (chunk, i) = (id as usize / CHUNK, id as usize % CHUNK);
        let w = self.chunk_words(chunk)?;
        if w[0] >> i & 1 == 0 {
            return None;
        }
        let slot_count = (w[1] & 0xFF) as usize;
        let index_bits = (w[1] >> 8 & 0xFF) as usize;
        let index = if index_bits == 0 {
            0
        } else {
            let pos = i * index_bits;
            let entries = &w[2 + slot_count.div_ceil(2)..];
            let (word, shift) = (pos / 64, pos % 64);
            let mut v = entries[word] >> shift;
            if shift + index_bits > 64 {
                v |= entries[word + 1] << (64 - shift);
            }
            (v & ((1u64 << index_bits) - 1)) as usize
        };
        Some((w[2 + index / 2] >> (32 * (index % 2))) as u32)
    }

    fn decode(&self, chunk: usize) -> [Option<u32>; CHUNK] {
        let mut out = [None; CHUNK];
        if self.chunk_words(chunk).is_some() {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = self.get((chunk * CHUNK + i) as u32);
            }
        }
        out
    }

    fn encode(entries: &[Option<u32>; CHUNK]) -> Vec<u64> {
        let mut live = 0u64;
        let mut slots: Vec<u32> = Vec::new();
        for (i, e) in entries.iter().enumerate() {
            if let Some(slot) = *e {
                live |= 1 << i;
                if !slots.contains(&slot) {
                    slots.push(slot);
                }
            }
        }
        if live == 0 {
            return vec![0, 0];
        }
        let index_bits = if slots.len() == 1 { 0 } else { bits(slots.len() as u64 - 1) };
        let mut words = vec![live, slots.len() as u64 | u64::from(index_bits) << 8];
        for pair in slots.chunks(2) {
            words.push(u64::from(pair[0]) | pair.get(1).map_or(0, |&s| u64::from(s) << 32));
        }
        let base = words.len();
        words.resize(base + (CHUNK * index_bits as usize).div_ceil(64), 0);
        if index_bits > 0 {
            let width = index_bits as usize;
            for (i, e) in entries.iter().enumerate() {
                if let Some(slot) = *e {
                    let index = slots.iter().position(|&s| s == slot).expect("collected above") as u64;
                    let pos = i * width;
                    let (word, shift) = (pos / 64, pos % 64);
                    words[base + word] |= index << shift;
                    if shift + width > 64 {
                        words[base + word + 1] |= index >> (64 - shift);
                    }
                }
            }
        }
        words
    }

    fn used_words(words: &[u64]) -> usize {
        if words[0] == 0 {
            return 2;
        }
        let slot_count = (words[1] & 0xFF) as usize;
        let index_bits = (words[1] >> 8 & 0xFF) as usize;
        2 + slot_count.div_ceil(2) + (CHUNK * index_bits).div_ceil(64)
    }

    fn store(&mut self, chunk: usize, words: Vec<u64>) {
        if chunk >= self.offsets.len() {
            self.offsets.resize(chunk + 1, NO_CHUNK);
            self.caps.resize(chunk + 1, 0);
        }
        let (off, cap) = (self.offsets[chunk], self.caps[chunk] as usize);
        if off != NO_CHUNK && words.len() <= cap {
            let off = off as usize;
            self.pool[off..off + words.len()].copy_from_slice(&words);
            self.pool[off + words.len()..off + cap].fill(0);
            return;
        }
        if off != NO_CHUNK {
            self.garbage += cap;
        }
        assert!(self.pool.len() + words.len() < NO_CHUNK as usize, "slot map pool exceeds u32 offsets");
        self.offsets[chunk] = self.pool.len() as u32;
        self.caps[chunk] = words.len() as u8;
        self.pool.extend_from_slice(&words);
        if self.garbage > self.pool.len() / 2 {
            self.compact();
        }
    }

    /// Records `slot` for every id in `ids`, rewriting each touched chunk
    /// once.
    pub(crate) fn assign(&mut self, slot: u32, ids: impl Iterator<Item = u32>) {
        let mut ids: Vec<u32> = ids.collect();
        ids.sort_unstable();
        let mut start = 0;
        while start < ids.len() {
            let chunk = ids[start] as usize / CHUNK;
            let mut decoded = self.decode(chunk);
            let mut end = start;
            while end < ids.len() && ids[end] as usize / CHUNK == chunk {
                decoded[ids[end] as usize % CHUNK] = Some(slot);
                end += 1;
            }
            let words = Self::encode(&decoded);
            self.store(chunk, words);
            start = end;
        }
    }

    pub(crate) fn set(&mut self, id: u32, slot: u32) {
        self.assign(slot, std::iter::once(id));
    }

    pub(crate) fn clear(&mut self, id: u32) {
        let chunk = id as usize / CHUNK;
        if self.chunk_words(chunk).is_none() {
            return;
        }
        let mut decoded = self.decode(chunk);
        decoded[id as usize % CHUNK] = None;
        let words = Self::encode(&decoded);
        self.store(chunk, words);
    }

    fn compact(&mut self) {
        let mut pool = Vec::with_capacity(self.pool.len() - self.garbage);
        for chunk in 0..self.offsets.len() {
            let off = self.offsets[chunk];
            if off == NO_CHUNK {
                continue;
            }
            let old = &self.pool[off as usize..off as usize + self.caps[chunk] as usize];
            let used = Self::used_words(old);
            self.offsets[chunk] = pool.len() as u32;
            self.caps[chunk] = used as u8;
            pool.extend_from_slice(&old[..used]);
        }
        self.pool = pool;
        self.garbage = 0;
    }

    /// Logical size: each chunk's words in use plus per-chunk offset and
    /// capacity. O(chunks).
    pub(crate) fn size_bytes(&self) -> usize {
        let used: usize = (0..self.offsets.len())
            .filter_map(|chunk| self.chunk_words(chunk))
            .map(Self::used_words)
            .sum();
        used * 8 + self.offsets.len() * 5
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.compact();
        self.pool.shrink_to_fit();
        self.offsets.shrink_to_fit();
        self.caps.shrink_to_fit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn matches_model_under_batches_sets_and_clears() {
        let mut map = SlotMap::new();
        let mut model: HashMap<u32, u32> = HashMap::new();
        let mut x = 0x2545_F491u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for round in 0..3000 {
            match next() % 4 {
                0 | 1 => {
                    let slot = next() % 5000;
                    let mut id = next() % 20_000;
                    let ids: Vec<u32> = (0..1 + next() % 200)
                        .map(|_| {
                            id += 1 + next() % 3;
                            id
                        })
                        .collect();
                    for &id in &ids {
                        model.insert(id, slot);
                    }
                    map.assign(slot, ids.into_iter());
                }
                2 => {
                    let (id, slot) = (next() % 20_500, u32::MAX - 1 - next() % 3);
                    model.insert(id, slot);
                    map.set(id, slot);
                }
                _ => {
                    let id = next() % 20_500;
                    model.remove(&id);
                    map.clear(id);
                }
            }
            if round % 300 == 0 || round == 2999 {
                for id in 0..21_000u32 {
                    assert_eq!(map.get(id), model.get(&id).copied(), "round {round} id {id}");
                }
            }
        }
        let before = map.size_bytes();
        map.shrink_to_fit();
        assert_eq!(map.size_bytes(), before);
        for id in 0..21_000u32 {
            assert_eq!(map.get(id), model.get(&id).copied());
        }
    }
}
