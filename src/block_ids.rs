//! `BlockIds`: a block's `rank -> TermId` array, compressed.
//!
//! In a bulk load each batch assigns new ids in sorted key order, and a
//! block covers a contiguous key range, so the terms one batch adds to one
//! block get a run of consecutive ids that also increase with rank. A
//! block's ids are a few such runs interleaved (on real RDF: ~11 runs per
//! block, mean length ~70; DESIGN.md's "block id encoding"). Plain FOR /
//! BP128 / PFOR see the interleaving as noise (~2 bytes/term); a table of
//! run bases with a per-entry `(run, offset)` sees the runs (~1 byte/term).
//!
//! Three encodings; `new` picks the smallest:
//!
//! - `Implicit`: a single run, so `id = base + rank` -- no per-entry data.
//! - `Runs`: `id = bases[run] + offset`, `(run, offset)` bit-packed.
//! - `For`: `id = bases[0] + offset`, offset bit-packed (frame of
//!   reference over the block's id range) -- the fallback when ids don't
//!   form few runs (single-term inserts, reused freed ids).
//!
//! Runs are found greedily in rank order: an id continues the open run
//! that expects it next, or starts a new one. Every run is therefore
//! monotone and offsets are exact for any id order -- only the size
//! depends on the ids' structure. All access is O(1).

use crate::TermId;

/// `Runs` inverse: an absolute rank every this many entries of a run,
/// deltas in between, so `rank_of` adds at most `RANK_SAMPLE - 1` deltas.
/// Measured on real RDF: 0.69 bytes/term vs 1.28 for FOR per run
/// (DESIGN.md's "ranks out of the reverse map").
const RANK_SAMPLE: usize = 16;

/// Greedy run detection gives up beyond this many runs (`For` is smaller
/// by then anyway: each run costs a 4-byte base).
const MAX_RUNS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Implicit,
    Runs,
    For,
}

#[derive(Debug, Clone)]
pub(crate) struct BlockIds {
    len: u32,
    kind: Kind,
    /// Offset bits; in `Runs`, the run index sits above them.
    off_bits: u8,
    entry_bits: u8,
    /// Bits of a rank in this block (`bits(len - 1)`).
    rank_bits: u8,
    /// Number of bases: run bases (`Runs`, in base order), or the single
    /// base (`Implicit`, `For`; 0 when `len == 0`).
    base_count: u32,
    /// Section starts in `data` (words): packed entries, run descriptors
    /// (`Runs`: `len | delta_bits << 26 | bit_offset << 32` per run;
    /// 26 length bits because a block can briefly hold a whole batch run,
    /// up to 3 x 65,536 terms, before `split_while_oversized`), and the
    /// inverse (`Runs`: per run a rank sample every `RANK_SAMPLE` entries,
    /// then rank deltas; `For`: ranks in id order).
    words_at: u32,
    meta_at: u32,
    inverse_at: u32,
    /// Bases (two `u32` per word), entries, descriptors and inverse in
    /// one allocation: a lookup touches one heap object, not four --
    /// measured as most of a ~300 ns `get_term` regression when they were
    /// separate boxes (DESIGN.md's "ranks out of the reverse map").
    data: Box<[u64]>,
}

fn bits(v: u64) -> u8 {
    (u64::BITS - v.leading_zeros()) as u8
}

fn words_for(len: usize, width: u8) -> usize {
    (len * width as usize).div_ceil(64)
}

fn pack(values: impl Iterator<Item = u64>, len: usize, width: u8) -> Box<[u64]> {
    let mut words = vec![0u64; words_for(len, width)];
    if width > 0 {
        for (i, v) in values.enumerate() {
            let pos = i * width as usize;
            let (word, shift) = (pos / 64, pos % 64);
            words[word] |= v << shift;
            if shift + width as usize > 64 {
                words[word + 1] |= v >> (64 - shift);
            }
        }
    }
    words.into_boxed_slice()
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

/// Per run (in base order): `ceil(len / RANK_SAMPLE)` rank samples of
/// `rank_bits`, then `len` deltas of the run's own width (the delta at a
/// sample position is unused). Returns `(run_meta, inverse)`.
fn encode_run_ranks(ranks: &[Vec<u64>], rank_bits: u8) -> (Box<[u64]>, Box<[u64]>) {
    let rank_bits = rank_bits as usize;
    let mut meta = Vec::with_capacity(ranks.len());
    let mut total = 0usize;
    let mut layout = Vec::with_capacity(ranks.len());
    for r in ranks {
        let delta_bits = r.windows(2).map(|w| bits(w[1] - w[0])).max().unwrap_or(0) as usize;
        layout.push((total, delta_bits));
        assert!(r.len() < 1 << 26 && total < 1 << 32, "block too large for run descriptors");
        meta.push(r.len() as u64 | (delta_bits as u64) << 26 | (total as u64) << 32);
        total += r.len().div_ceil(RANK_SAMPLE) * rank_bits + r.len() * delta_bits;
    }
    let mut inverse = vec![0u64; total.div_ceil(64)];
    for (r, &(start, delta_bits)) in ranks.iter().zip(&layout) {
        let deltas = start + r.len().div_ceil(RANK_SAMPLE) * rank_bits;
        for (t, &rank) in r.iter().enumerate() {
            if t % RANK_SAMPLE == 0 {
                write_bits(&mut inverse, start + (t / RANK_SAMPLE) * rank_bits, rank_bits, rank);
            } else {
                write_bits(&mut inverse, deltas + t * delta_bits, delta_bits, rank - r[t - 1]);
            }
        }
    }
    (meta.into_boxed_slice(), inverse.into_boxed_slice())
}

impl BlockIds {
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        len: usize,
        kind: Kind,
        bases: &[u32],
        words: &[u64],
        run_meta: &[u64],
        inverse: &[u64],
        off_bits: u8,
        entry_bits: u8,
        rank_bits: u8,
    ) -> BlockIds {
        let mut data: Vec<u64> = Vec::with_capacity(bases.len().div_ceil(2) + words.len() + run_meta.len() + inverse.len());
        for pair in bases.chunks(2) {
            data.push(u64::from(pair[0]) | pair.get(1).map_or(0, |&b| u64::from(b) << 32));
        }
        let words_at = data.len() as u32;
        data.extend_from_slice(words);
        let meta_at = data.len() as u32;
        data.extend_from_slice(run_meta);
        let inverse_at = data.len() as u32;
        data.extend_from_slice(inverse);
        BlockIds {
            len: len as u32,
            kind,
            off_bits,
            entry_bits,
            rank_bits,
            base_count: bases.len() as u32,
            words_at,
            meta_at,
            inverse_at,
            data: data.into_boxed_slice(),
        }
    }

    #[inline]
    fn base(&self, i: usize) -> u32 {
        (self.data[i / 2] >> (32 * (i % 2))) as u32
    }

    #[inline]
    fn words(&self) -> &[u64] {
        &self.data[self.words_at as usize..self.meta_at as usize]
    }

    #[inline]
    fn inverse(&self) -> &[u64] {
        &self.data[self.inverse_at as usize..]
    }

    pub(crate) fn new(ids: &[TermId]) -> BlockIds {
        let len = ids.len();
        if len == 0 {
            return BlockIds::assemble(0, Kind::Implicit, &[], &[], &[], &[], 0, 0, 0);
        }
        let min = ids.iter().map(|id| id.0).min().expect("non-empty");
        let max = ids.iter().map(|id| id.0).max().expect("non-empty");
        let for_bits = bits(u64::from(max - min));
        let for_size = 4 + 8 * words_for(len, for_bits);

        // Greedy runs: (base, next expected id).
        let mut runs: Vec<(u32, u64)> = Vec::new();
        let mut entries: Vec<(u32, u32)> = Vec::with_capacity(len);
        let mut last = 0usize;
        let mut max_offset = 0u32;
        for &TermId(id) in ids {
            // The previous entry's run is the likeliest to continue.
            let found = if runs.get(last).is_some_and(|r| r.1 == u64::from(id)) {
                Some(last)
            } else {
                runs.iter().position(|r| r.1 == u64::from(id))
            };
            let run = match found {
                Some(run) => run,
                None => {
                    if runs.len() == MAX_RUNS {
                        runs.clear();
                        break;
                    }
                    runs.push((id, u64::from(id)));
                    runs.len() - 1
                }
            };
            runs[run].1 += 1;
            let offset = id - runs[run].0;
            max_offset = max_offset.max(offset);
            entries.push((run as u32, offset));
            last = run;
        }

        let rank_bits = bits(len as u64 - 1);
        if runs.len() == 1 {
            return BlockIds::assemble(len, Kind::Implicit, &[runs[0].0], &[], &[], &[], 0, 0, rank_bits);
        }
        if !runs.is_empty() {
            let off_bits = bits(u64::from(max_offset));
            let entry_bits = bits(runs.len() as u64 - 1) + off_bits;
            let runs_size = 4 * runs.len() + 8 * words_for(len, entry_bits);
            if runs_size < for_size {
                // Relabel runs in base order, so `rank_of` can binary-search
                // the bases (runs are disjoint id ranges).
                let mut order: Vec<usize> = (0..runs.len()).collect();
                order.sort_unstable_by_key(|&r| runs[r].0);
                let mut label = vec![0u32; runs.len()];
                for (new, &old) in order.iter().enumerate() {
                    label[old] = new as u32;
                }
                let words = pack(
                    entries.iter().map(|&(run, off)| (u64::from(label[run as usize]) << off_bits) | u64::from(off)),
                    len,
                    entry_bits,
                );
                // Ranks of each run's entries, in rank order (= offset order).
                let mut ranks: Vec<Vec<u64>> = vec![Vec::new(); runs.len()];
                for (rank, &(run, _)) in entries.iter().enumerate() {
                    ranks[label[run as usize] as usize].push(rank as u64);
                }
                let (run_meta, inverse) = encode_run_ranks(&ranks, rank_bits);
                let bases: Vec<u32> = order.iter().map(|&r| runs[r].0).collect();
                return BlockIds::assemble(len, Kind::Runs, &bases, &words, &run_meta, &inverse, off_bits, entry_bits, rank_bits);
            }
        }
        // `For` inverse: ranks sorted by id, binary-searched in `rank_of`.
        let mut by_id: Vec<u32> = (0..len as u32).collect();
        by_id.sort_unstable_by_key(|&r| ids[r as usize].0);
        BlockIds::assemble(
            len,
            Kind::For,
            &[min],
            &pack(ids.iter().map(|id| u64::from(id.0 - min)), len, for_bits),
            &[],
            &pack(by_id.iter().map(|&r| u64::from(r)), len, rank_bits),
            for_bits,
            for_bits,
            rank_bits,
        )
    }

    /// The rank of `id` in this block, if present. `Implicit`: O(1).
    /// `Runs`: binary search over the run bases, then a sample plus at
    /// most `RANK_SAMPLE - 1` deltas. `For`: binary search over the ranks
    /// stored in id order.
    pub(crate) fn rank_of(&self, id: TermId) -> Option<usize> {
        let id = id.0;
        match self.kind {
            Kind::Implicit => {
                if self.base_count == 0 {
                    return None;
                }
                let base = self.base(0);
                (id >= base && id - base < self.len).then(|| (id - base) as usize)
            }
            Kind::Runs => {
                // partition_point over the bases: first base > id, minus one.
                let (mut lo, mut hi) = (0usize, self.base_count as usize);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if self.base(mid) <= id {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                let run = lo.checked_sub(1)?;
                let offset = (id - self.base(run)) as usize;
                let meta = self.data[self.meta_at as usize + run];
                let run_len = (meta & ((1 << 26) - 1)) as usize;
                if offset >= run_len {
                    return None;
                }
                let delta_bits = (meta >> 26 & 0x3F) as usize;
                let start = (meta >> 32) as usize;
                let rank_bits = self.rank_bits as usize;
                let sample = offset / RANK_SAMPLE;
                let mut rank = read_bits(self.inverse(), start + sample * rank_bits, rank_bits);
                let deltas = start + run_len.div_ceil(RANK_SAMPLE) * rank_bits;
                for t in sample * RANK_SAMPLE + 1..=offset {
                    rank += read_bits(self.inverse(), deltas + t * delta_bits, delta_bits);
                }
                Some(rank as usize)
            }
            Kind::For => {
                let rank_bits = self.rank_bits as usize;
                let at = |k: usize| read_bits(self.inverse(), k * rank_bits, rank_bits) as usize;
                let (mut lo, mut hi) = (0usize, self.len());
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let found = self.get(at(mid)).0;
                    if found == id {
                        return Some(at(mid));
                    }
                    if found < id {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                None
            }
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn read(&self, rank: usize) -> u64 {
        let width = self.entry_bits as usize;
        read_bits(self.words(), rank * width, width)
    }

    #[inline]
    pub(crate) fn get(&self, rank: usize) -> TermId {
        debug_assert!(rank < self.len());
        match self.kind {
            Kind::Implicit => TermId(self.base(0) + rank as u32),
            Kind::For => TermId(self.base(0) + self.read(rank) as u32),
            Kind::Runs => {
                let entry = self.read(rank);
                let offset = (entry & ((1u64 << self.off_bits) - 1)) as u32;
                TermId(self.base((entry >> self.off_bits) as usize) + offset)
            }
        }
    }

    pub(crate) fn to_vec(&self) -> Vec<TermId> {
        (0..self.len()).map(|r| self.get(r)).collect()
    }

    /// Logical size: the single data allocation.
    pub(crate) fn size_bytes(&self) -> usize {
        self.data.len() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[u32]) -> Vec<TermId> {
        v.iter().map(|&x| TermId(x)).collect()
    }

    fn round_trip(v: &[TermId]) -> BlockIds {
        let b = BlockIds::new(v);
        assert_eq!(b.len(), v.len());
        assert_eq!(b.to_vec(), v);
        for (rank, &id) in v.iter().enumerate() {
            assert_eq!(b.rank_of(id), Some(rank), "{:?} id {id:?}", b.kind);
        }
        // Absent ids: just outside each id, where not present.
        let present: std::collections::HashSet<u32> = v.iter().map(|id| id.0).collect();
        for &TermId(id) in v {
            for probe in [id.wrapping_sub(1), id.wrapping_add(1)] {
                if !present.contains(&probe) {
                    assert_eq!(b.rank_of(TermId(probe)), None, "{:?} probe {probe}", b.kind);
                }
            }
        }
        b
    }

    #[test]
    fn implicit_runs_and_for_round_trip() {
        assert_eq!(round_trip(&[]).size_bytes(), 0);
        let b = round_trip(&ids(&[100, 101, 102, 103]));
        assert_eq!(b.kind, Kind::Implicit);
        assert_eq!(b.size_bytes(), 8, "one base, in a single data word");
        // Three interleaved runs, each increasing with rank.
        let mut v = Vec::new();
        for i in 0..300u32 {
            v.push(match i % 3 {
                0 => 1_000_000 + i / 3,
                1 => 5_000 + i / 3,
                _ => 3_000_000 + i / 3,
            });
        }
        let b = round_trip(&ids(&v));
        assert_eq!(b.kind, Kind::Runs);
        // Forward: 2 run bits + 7 offset bits per entry plus 3 bases
        // (~356 bytes); inverse: per run 7 samples + 2-bit deltas plus a
        // descriptor (~125 bytes). Raw u32 both ways would be 2,400.
        assert!(b.size_bytes() < 600, "{}", b.size_bytes());
        // No runs at all: FOR over a narrow range.
        let v: Vec<u32> = (0..500u32).map(|i| 7_000 + (i * 7919) % 997).collect();
        let b = round_trip(&ids(&v));
        assert_eq!(b.kind, Kind::For);
        // Decreasing ids: each its own run, so FOR wins.
        let v: Vec<u32> = (0..400u32).rev().collect();
        assert_eq!(round_trip(&ids(&v)).kind, Kind::For);
    }

    /// A transiently oversized block (a whole batch run before its split)
    /// with runs longer than 65,535 -- the full-file load that found this.
    #[test]
    fn runs_longer_than_u16() {
        let mut v = Vec::new();
        for i in 0..70_000u32 {
            v.push(1_000_000 + i);
            v.push(5_000_000 + i);
        }
        assert_eq!(round_trip(&ids(&v)).kind, Kind::Runs);
    }

    #[test]
    fn extremes_and_more_runs_than_max() {
        round_trip(&ids(&[u32::MAX - 1, u32::MAX, 0, 1]));
        round_trip(&ids(&[u32::MAX]));
        let mut v = Vec::new();
        for r in 0..(MAX_RUNS as u32 + 10) {
            v.push(r * 1000);
            v.push(r * 1000 + 1);
        }
        round_trip(&ids(&v));
        // Pseudo-random ids over the whole range.
        let mut x = 0x9E37_79B9u32;
        let v: Vec<u32> = (0..1000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x
            })
            .collect();
        round_trip(&ids(&v));
    }
}
