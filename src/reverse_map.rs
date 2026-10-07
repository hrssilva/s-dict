//! `TermId -> (block slot, local rank)`, compressed in 64-id chunks.
//!
//! Shared by `Dictionary` and `PackedAutomatonDictionary`. A plain `u32`
//! per id (slot and rank bit-packed) was 4 bytes/term -- after block-id
//! compression, the largest fixed cost. Ids come in runs: a bulk-load batch
//! assigns consecutive ids, in key order, to the terms it adds to one block,
//! so 64 consecutive ids land in only a few blocks, and their ranks within
//! each block are close. Per chunk of 64 ids this stores the few distinct
//! slots once, and per id a slot index plus a frame-of-reference rank:
//! ~1.4 bytes/term measured on real RDF vs 3.0 for global bit-packing
//! (DESIGN.md's "reverse map encoding").
//!
//! Chunk layout (`u64` words in one shared pool):
//! - word 0: live bitmask (bit `i` = id `chunk * 64 + i` is assigned);
//! - word 1: `rank_base: u32 | slot_count << 32 | slot_bits << 40 |
//!   rank_bits << 48` (`slot_bits` = bits of a slot *index*);
//! - `slot_count` slots, two `u32` per word;
//! - 64 entries of `slot_bits + rank_bits` bits (dead positions are 0).
//!
//! Lookups decode one entry: O(1). Writes rewrite whole chunks, so callers
//! batch them ([`ReverseMap::assign`]: each touched chunk once per call).
//! A rewritten chunk goes back in place if it fits, else is appended to
//! the pool; garbage is compacted when it exceeds half the pool, and by
//! [`ReverseMap::shrink_to_fit`].

const CHUNK: usize = 64;
const NO_CHUNK: u32 = u32::MAX;

#[derive(Debug, Clone, Default)]
pub(crate) struct ReverseMap {
    pool: Vec<u64>,
    /// Per chunk: start in `pool` (`NO_CHUNK`: nothing assigned yet).
    offsets: Vec<u32>,
    /// Per chunk: words reserved at `offsets[i]` (>= words in use).
    caps: Vec<u8>,
    /// Pool words no longer referenced by any chunk.
    garbage: usize,
}

fn bits(v: u64) -> u32 {
    u64::BITS - v.leading_zeros()
}

/// One chunk, decoded: `(slot, rank)` per position, `None` if unassigned.
type Decoded = [Option<(u32, u32)>; CHUNK];

impl ReverseMap {
    pub(crate) fn new() -> ReverseMap {
        ReverseMap::default()
    }

    fn chunk_words(&self, chunk: usize) -> Option<&[u64]> {
        let off = *self.offsets.get(chunk)?;
        (off != NO_CHUNK).then(|| &self.pool[off as usize..off as usize + self.caps[chunk] as usize])
    }

    /// `(block slot, local rank)` for `id`, if assigned.
    #[inline]
    pub(crate) fn get(&self, id: u32) -> Option<(u32, usize)> {
        let (chunk, i) = (id as usize / CHUNK, id as usize % CHUNK);
        let w = self.chunk_words(chunk)?;
        if w[0] >> i & 1 == 0 {
            return None;
        }
        let header = w[1];
        let rank_base = header as u32;
        let slot_count = (header >> 32 & 0xFF) as usize;
        let slot_bits = (header >> 40 & 0xFF) as usize;
        let rank_bits = (header >> 48 & 0xFF) as usize;
        let entry_bits = slot_bits + rank_bits;
        let entry = read_bits(&w[2 + slot_count.div_ceil(2)..], i * entry_bits, entry_bits);
        let slot_index = (entry >> rank_bits) as usize;
        let rank = rank_base + (entry & ((1u64 << rank_bits) - 1)) as u32;
        let slot = (w[2 + slot_index / 2] >> (32 * (slot_index % 2))) as u32;
        Some((slot, rank as usize))
    }

    fn decode(&self, chunk: usize) -> Decoded {
        let mut out = [None; CHUNK];
        if let Some(w) = self.chunk_words(chunk) {
            let live = w[0];
            for (i, slot) in out.iter_mut().enumerate() {
                if live >> i & 1 == 1 {
                    *slot = self
                        .get((chunk * CHUNK + i) as u32)
                        .map(|(s, r)| (s, r as u32));
                }
            }
        }
        out
    }

    fn encode(entries: &Decoded) -> Vec<u64> {
        let mut live = 0u64;
        let mut slots: Vec<u32> = Vec::new();
        let (mut rmin, mut rmax) = (u32::MAX, 0u32);
        for (i, e) in entries.iter().enumerate() {
            if let Some((slot, rank)) = *e {
                live |= 1 << i;
                if !slots.contains(&slot) {
                    slots.push(slot);
                }
                rmin = rmin.min(rank);
                rmax = rmax.max(rank);
            }
        }
        if live == 0 {
            return vec![0, 0];
        }
        let slot_bits = bits(slots.len() as u64 - 1);
        let rank_bits = bits(u64::from(rmax - rmin));
        let entry_bits = (slot_bits + rank_bits) as usize;
        let mut words = vec![
            live,
            u64::from(rmin)
                | (slots.len() as u64) << 32
                | u64::from(slot_bits) << 40
                | u64::from(rank_bits) << 48,
        ];
        for pair in slots.chunks(2) {
            words.push(u64::from(pair[0]) | pair.get(1).map_or(0, |&s| u64::from(s) << 32));
        }
        let base = words.len();
        words.resize(base + (CHUNK * entry_bits).div_ceil(64), 0);
        for (i, e) in entries.iter().enumerate() {
            if let Some((slot, rank)) = *e {
                let index = slots.iter().position(|&s| s == slot).expect("collected above") as u64;
                write_bits(&mut words[base..], i * entry_bits, entry_bits, index << rank_bits | u64::from(rank - rmin));
            }
        }
        words
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
        assert!(self.pool.len() + words.len() < NO_CHUNK as usize, "reverse map pool exceeds u32 offsets");
        self.offsets[chunk] = self.pool.len() as u32;
        self.caps[chunk] = words.len() as u8;
        self.pool.extend_from_slice(&words);
        if self.garbage > self.pool.len() / 2 {
            self.compact();
        }
    }

    /// Assigns `(id, rank)` pairs, all in block `slot`, rewriting each
    /// touched chunk once. A repeated id keeps its last pair, as a
    /// sequence of `set` calls would (the sort is stable).
    pub(crate) fn assign(&mut self, slot: u32, entries: impl Iterator<Item = (u32, usize)>) {
        let mut entries: Vec<(u32, usize)> = entries.collect();
        entries.sort_by_key(|&(id, _)| id);
        let mut start = 0;
        while start < entries.len() {
            let chunk = entries[start].0 as usize / CHUNK;
            let mut decoded = self.decode(chunk);
            let mut end = start;
            while end < entries.len() && entries[end].0 as usize / CHUNK == chunk {
                let (id, rank) = entries[end];
                let rank = u32::try_from(rank).expect("local rank fits u32");
                decoded[id as usize % CHUNK] = Some((slot, rank));
                end += 1;
            }
            let words = Self::encode(&decoded);
            self.store(chunk, words);
            start = end;
        }
    }

    pub(crate) fn set(&mut self, id: u32, slot: u32, rank: usize) {
        self.assign(slot, std::iter::once((id, rank)));
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

    fn used_words(words: &[u64]) -> usize {
        let header = words[1];
        let slot_count = (header >> 32 & 0xFF) as usize;
        let entry_bits = (header >> 40 & 0xFF) as usize + (header >> 48 & 0xFF) as usize;
        if words[0] == 0 {
            2
        } else {
            2 + slot_count.div_ceil(2) + (CHUNK * entry_bits).div_ceil(64)
        }
    }

    /// Rewrites the pool with every chunk at its exact size.
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

    /// Logical size: each chunk's words in use (from its header; slack
    /// left by in-place rewrites and garbage excluded) plus per-chunk
    /// offset and capacity. O(chunks).
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

#[inline]
fn read_bits(words: &[u64], pos: usize, width: usize) -> u64 {
    if width == 0 {
        return 0;
    }
    let (word, shift) = (pos / 64, pos % 64);
    let mut v = words[word] >> shift;
    if shift + width > 64 {
        v |= words[word + 1] << (64 - shift);
    }
    v & ((1u64 << width) - 1)
}

fn write_bits(words: &mut [u64], pos: usize, width: usize, value: u64) {
    if width == 0 {
        return;
    }
    let (word, shift) = (pos / 64, pos % 64);
    words[word] |= value << shift;
    if shift + width > 64 {
        words[word + 1] |= value >> (64 - shift);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn matches_model_under_batches_sets_and_clears() {
        let mut map = ReverseMap::new();
        let mut model: HashMap<u32, (u32, usize)> = HashMap::new();
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
                    // A block's suffix: consecutive-ish ids, increasing ranks.
                    let slot = next() % 5000;
                    let start = next() % 20_000;
                    let n = 1 + next() as usize % 200;
                    // Increasing ids (gaps of 1-3) with increasing ranks, plus
                    // one repeated id: its last pair must win.
                    let mut id = start;
                    let mut entries: Vec<(u32, usize)> = (0..n)
                        .map(|k| {
                            id += 1 + next() % 3;
                            (id, k * 2 + next() as usize % 2)
                        })
                        .collect();
                    entries.push((entries[0].0, 7));
                    for &(id, rank) in &entries {
                        model.insert(id, (slot, rank));
                    }
                    map.assign(slot, entries.into_iter());
                }
                2 => {
                    let id = next() % 20_500;
                    let v = (u32::MAX - 1 - next() % 3, next() as usize % 100_000);
                    model.insert(id, v);
                    map.set(id, v.0, v.1);
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
        assert_eq!(map.garbage, 0);
        for id in 0..21_000u32 {
            assert_eq!(map.get(id), model.get(&id).copied());
        }
    }

    #[test]
    fn run_structured_ids_compress() {
        // 4 blocks, each receiving a run of 64 consecutive ids with
        // increasing ranks spread over ~1000.
        let mut map = ReverseMap::new();
        for b in 0..4u32 {
            map.assign(b, (0..64u32).map(|k| (b * 64 + k, (k * 15) as usize)));
        }
        map.shrink_to_fit();
        assert!(map.size_bytes() < 4 * 64 * 2, "{} bytes", map.size_bytes());
        assert_eq!(map.get(65), Some((1, 15)));
        assert_eq!(map.get(1000), None);
    }
}

