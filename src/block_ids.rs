//! `BlockIds`: a block's `rank <-> TermId` permutation, compressed.
//!
//! In a bulk load each batch assigns new ids in sorted key order, and a
//! block covers a contiguous key range, so the terms one batch adds to one
//! block get a run of consecutive ids that also increase with rank. A
//! block's ids are a few such runs interleaved (on real RDF: ~11 runs per
//! block, mean length ~70; DESIGN.md's "block id encoding").
//!
//! Label each rank with its run (runs numbered in base order). Then
//! `id(rank) = base[label] + (occurrences of label before rank)` and,
//! inversely, `rank(id) = position of occurrence (id - base[run])` of
//! `run`'s label: one `rank` query and one `select` query on the label
//! sequence (Barbay & Navarro's compressed permutations by runs). The
//! label sequence is stored as a wavelet matrix -- `log2(runs)` bit
//! vectors with popcount rank samples -- so both directions come from
//! one structure, with no stored offsets or separate inverse (DESIGN.md's
//! "succinct id permutation").
//!
//! Three encodings; `new` picks the smallest:
//!
//! - `Implicit`: a single run, so `id = base + rank` -- no per-entry data.
//! - `Wavelet`: the run-label wavelet matrix above.
//! - `For`: `id = base + offset`, offset bit-packed (frame of reference),
//!   with ranks stored in id order for the inverse -- the fallback when
//!   ids don't form few runs (single-term inserts, reused freed ids).
//!
//! Runs are found greedily in rank order: an id continues the open run
//! that expects it next, or starts a new one. Every run is therefore
//! monotone, and both directions are exact for any id order -- only the
//! size depends on the ids' structure. Everything lives in one allocation.

use crate::TermId;

/// Greedy run detection gives up beyond this many runs (`For` is smaller
/// by then anyway).
const MAX_RUNS: usize = 256;

/// Bits covered by one rank sample of a wavelet-matrix level.
const SAMPLE_BITS: usize = 256;
const SAMPLE_WORDS: usize = SAMPLE_BITS / 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Implicit,
    Wavelet,
    For,
}

#[derive(Debug, Clone)]
pub(crate) struct BlockIds {
    len: u32,
    kind: Kind,
    /// `Wavelet`: levels (`bits(runs - 1)`). `For`: offset bits.
    width: u8,
    /// `For`: bits of a stored rank.
    rank_bits: u8,
    /// `Implicit`/`For`: the base id. `Wavelet`: the run count.
    base: u32,
    /// Section starts in `data` (words). `Wavelet`: run table
    /// (`base | len << 32` per run) at 0, then bottom-level starts (two
    /// `u32` per word), per-level zero counts (two per word), and the
    /// levels, each `sample words + bit words`. `For`: offsets at 0, ranks
    /// in id order at `second`.
    second: u32,
    third: u32,
    fourth: u32,
    data: Box<[u64]>,
}

fn bits(v: u64) -> u8 {
    (u64::BITS - v.leading_zeros()) as u8
}

fn words_for(len: usize, width: u8) -> usize {
    (len * width as usize).div_ceil(64)
}

fn pack(values: impl Iterator<Item = u64>, len: usize, width: u8) -> Vec<u64> {
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
    words
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

fn push_u32s(data: &mut Vec<u64>, values: &[u32]) {
    for pair in values.chunks(2) {
        data.push(u64::from(pair[0]) | pair.get(1).map_or(0, |&v| u64::from(v) << 32));
    }
}

/// Position of the `k`-th (0-based) set bit of `w`.
#[inline]
fn select_in_word(mut w: u64, k: u32) -> usize {
    for _ in 0..k {
        w &= w - 1;
    }
    w.trailing_zeros() as usize
}

/// Words of one wavelet-matrix level over `len` labels.
fn level_words(len: usize) -> (usize, usize) {
    let bit_words = len.div_ceil(64);
    let samples = bit_words.div_ceil(SAMPLE_WORDS) + 1;
    (samples.div_ceil(2), bit_words)
}

impl BlockIds {
    pub(crate) fn new(ids: &[TermId]) -> BlockIds {
        let len = ids.len();
        if len == 0 {
            return Self::implicit(0, 0);
        }
        let min = ids.iter().map(|id| id.0).min().expect("non-empty");
        let max = ids.iter().map(|id| id.0).max().expect("non-empty");

        // Greedy runs: (base, next expected id); label per rank.
        let mut runs: Vec<(u32, u64)> = Vec::new();
        let mut labels: Vec<u32> = Vec::with_capacity(len);
        let mut last = 0usize;
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
            labels.push(run as u32);
            last = run;
        }
        if runs.len() == 1 {
            return Self::implicit(len, runs[0].0);
        }

        let rank_bits = bits(len as u64 - 1);
        let for_bits = bits(u64::from(max - min));
        let for_size = 8 * (words_for(len, for_bits) + words_for(len, rank_bits));
        if !runs.is_empty() {
            let k = runs.len();
            let levels = bits(k as u64 - 1);
            let (sample_words, bit_words) = level_words(len);
            let wavelet_size = 8 * (k + k.div_ceil(2) + (levels as usize).div_ceil(2) + levels as usize * (sample_words + bit_words));
            if wavelet_size <= for_size {
                return Self::wavelet(&runs, &labels, levels);
            }
        }
        Self::frame_of_reference(ids, min, for_bits, rank_bits)
    }

    fn implicit(len: usize, base: u32) -> BlockIds {
        BlockIds {
            len: len as u32,
            kind: Kind::Implicit,
            width: 0,
            rank_bits: 0,
            base,
            second: 0,
            third: 0,
            fourth: 0,
            data: Box::new([]),
        }
    }

    fn frame_of_reference(ids: &[TermId], min: u32, for_bits: u8, rank_bits: u8) -> BlockIds {
        let len = ids.len();
        let mut by_id: Vec<u32> = (0..len as u32).collect();
        by_id.sort_unstable_by_key(|&r| ids[r as usize].0);
        let mut data = pack(ids.iter().map(|id| u64::from(id.0 - min)), len, for_bits);
        let second = data.len() as u32;
        data.extend(pack(by_id.iter().map(|&r| u64::from(r)), len, rank_bits));
        BlockIds {
            len: len as u32,
            kind: Kind::For,
            width: for_bits,
            rank_bits,
            base: min,
            second,
            third: 0,
            fourth: 0,
            data: data.into_boxed_slice(),
        }
    }

    /// `runs` in greedy order, `labels` per rank in that order.
    fn wavelet(runs: &[(u32, u64)], labels: &[u32], levels: u8) -> BlockIds {
        let len = labels.len();
        let k = runs.len();
        // Relabel runs in base order, so `rank_of` can binary-search bases.
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_unstable_by_key(|&r| runs[r].0);
        let mut relabel = vec![0u32; k];
        for (new, &old) in order.iter().enumerate() {
            relabel[old] = new as u32;
        }
        let mut cur: Vec<u32> = labels.iter().map(|&l| relabel[l as usize]).collect();
        let mut spare = vec![0u32; len];

        let mut data: Vec<u64> = Vec::new();
        for &old in &order {
            let (base, next) = runs[old];
            data.push(u64::from(base) | (next - u64::from(base)) << 32);
        }
        // Levels, top (most significant label bit) first.
        let (sample_words, bit_words) = level_words(len);
        let mut zeros: Vec<u32> = Vec::with_capacity(levels as usize);
        let mut level_data: Vec<u64> = Vec::with_capacity(levels as usize * (sample_words + bit_words));
        for l in 0..levels as usize {
            let shift = levels as usize - 1 - l;
            let mut words = vec![0u64; bit_words];
            for (i, &c) in cur.iter().enumerate() {
                if c >> shift & 1 == 1 {
                    words[i / 64] |= 1 << (i % 64);
                }
            }
            let mut samples: Vec<u32> = Vec::with_capacity(bit_words.div_ceil(SAMPLE_WORDS) + 1);
            let mut ones = 0u32;
            for (w, &word) in words.iter().enumerate() {
                if w % SAMPLE_WORDS == 0 {
                    samples.push(ones);
                }
                ones += word.count_ones();
            }
            samples.push(ones);
            samples.resize(bit_words.div_ceil(SAMPLE_WORDS) + 1, ones);
            let start = level_data.len();
            push_u32s(&mut level_data, &samples);
            level_data.resize(start + sample_words, 0);
            level_data.extend_from_slice(&words);
            zeros.push(len as u32 - ones);
            // Stable partition into the spare buffer: zeros first.
            let (mut zero, mut one) = (0usize, (len as u32 - ones) as usize);
            for &c in &cur {
                if c >> shift & 1 == 0 {
                    spare[zero] = c;
                    zero += 1;
                } else {
                    spare[one] = c;
                    one += 1;
                }
            }
            std::mem::swap(&mut cur, &mut spare);
        }
        // Bottom-level start of each label (equal labels are contiguous).
        let mut starts = vec![0u32; k];
        for (i, &c) in cur.iter().enumerate().rev() {
            starts[c as usize] = i as u32;
        }
        let second = data.len() as u32;
        push_u32s(&mut data, &starts);
        let third = data.len() as u32;
        push_u32s(&mut data, &zeros);
        let fourth = data.len() as u32;
        data.extend_from_slice(&level_data);
        BlockIds {
            len: len as u32,
            kind: Kind::Wavelet,
            width: levels,
            rank_bits: 0,
            base: k as u32,
            second,
            third,
            fourth,
            data: data.into_boxed_slice(),
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    fn u32_at(&self, section: u32, index: usize) -> u32 {
        (self.data[section as usize + index / 2] >> (32 * (index % 2))) as u32
    }

    /// `(sample section start, bit words start)` of wavelet level `l`.
    #[inline]
    fn level(&self, l: usize) -> (usize, usize) {
        let (sample_words, bit_words) = level_words(self.len());
        let at = self.fourth as usize + l * (sample_words + bit_words);
        (at, at + sample_words)
    }

    #[inline]
    fn sample(&self, samples_at: usize, g: usize) -> usize {
        (self.data[samples_at + g / 2] >> (32 * (g % 2))) as u32 as usize
    }

    /// Ones in level `l` before position `i`.
    #[inline]
    fn rank1(&self, l: usize, i: usize) -> usize {
        let (samples_at, words_at) = self.level(l);
        let g = i / SAMPLE_BITS;
        let mut r = self.sample(samples_at, g);
        let (first, word) = (g * SAMPLE_WORDS, i / 64);
        for w in first..word {
            r += self.data[words_at + w].count_ones() as usize;
        }
        if i % 64 != 0 {
            r += (self.data[words_at + word] & ((1u64 << (i % 64)) - 1)).count_ones() as usize;
        }
        r
    }

    /// Position in level `l` of the `j`-th (0-based) one (`ones`) or zero.
    fn select(&self, l: usize, j: usize, ones: bool) -> usize {
        let (samples_at, words_at) = self.level(l);
        let groups = self.len().div_ceil(64).div_ceil(SAMPLE_WORDS);
        let before = |g: usize| {
            let o = self.sample(samples_at, g);
            if ones { o } else { g * SAMPLE_BITS - o }
        };
        // Last group whose count before it is <= j.
        let (mut lo, mut hi) = (0usize, groups);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if before(mid) <= j {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let mut acc = before(lo);
        let mut w = lo * SAMPLE_WORDS;
        loop {
            let word = self.data[words_at + w];
            let word = if ones { word } else { !word };
            let c = word.count_ones() as usize;
            if acc + c > j {
                return w * 64 + select_in_word(word, (j - acc) as u32);
            }
            acc += c;
            w += 1;
        }
    }

    #[inline]
    pub(crate) fn get(&self, rank: usize) -> TermId {
        debug_assert!(rank < self.len());
        match self.kind {
            Kind::Implicit => TermId(self.base + rank as u32),
            Kind::For => TermId(self.base + read_bits(&self.data, rank * self.width as usize, self.width as usize) as u32),
            Kind::Wavelet => {
                let (mut i, mut c) = (rank, 0usize);
                for l in 0..self.width as usize {
                    let (_, words_at) = self.level(l);
                    let bit = (self.data[words_at + i / 64] >> (i % 64) & 1) as usize;
                    c = c << 1 | bit;
                    let ones = self.rank1(l, i);
                    i = if bit == 0 { i - ones } else { self.u32_at(self.third, l) as usize + ones };
                }
                let run = self.data[c];
                TermId(run as u32 + (i - self.u32_at(self.second, c) as usize) as u32)
            }
        }
    }

    /// The rank of `id` in this block, if present.
    pub(crate) fn rank_of(&self, id: TermId) -> Option<usize> {
        let id = id.0;
        match self.kind {
            Kind::Implicit => (id >= self.base && id - self.base < self.len).then(|| (id - self.base) as usize),
            Kind::Wavelet => {
                // Run whose base is the last <= id.
                let (mut lo, mut hi) = (0usize, self.base as usize);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if (self.data[mid] as u32) <= id {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                let c = lo.checked_sub(1)?;
                let run = self.data[c];
                let offset = (id - run as u32) as usize;
                if offset >= (run >> 32) as usize {
                    return None;
                }
                let levels = self.width as usize;
                let mut i = self.u32_at(self.second, c) as usize + offset;
                for l in (0..levels).rev() {
                    let bit = c >> (levels - 1 - l) & 1;
                    i = if bit == 0 {
                        self.select(l, i, false)
                    } else {
                        self.select(l, i - self.u32_at(self.third, l) as usize, true)
                    };
                }
                Some(i)
            }
            Kind::For => {
                let rank_bits = self.rank_bits as usize;
                let inverse = &self.data[self.second as usize..];
                let at = |k: usize| read_bits(inverse, k * rank_bits, rank_bits) as usize;
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

    /// Appends all ids in rank order to `out`. `Wavelet` decodes in bulk -- per-rank `get`
    /// would cost a `rank` query per level per entry, and merges call this
    /// on every block they touch: start from the bottom level (labels
    /// sorted, each contiguous), undo each level's stable partition
    /// bottom-up, then hand out each run's consecutive ids in rank order.
    pub(crate) fn decode_into(&self, out: &mut Vec<TermId>) {
        if self.kind != Kind::Wavelet {
            out.extend((0..self.len()).map(|r| self.get(r)));
            return;
        }
        let (n, k, levels) = (self.len(), self.base as usize, self.width as usize);
        let mut by_start: Vec<usize> = (0..k).collect();
        by_start.sort_unstable_by_key(|&c| self.u32_at(self.second, c));
        let mut cur: Vec<u32> = Vec::with_capacity(n);
        for &c in &by_start {
            let run_len = (self.data[c] >> 32) as usize;
            cur.extend(std::iter::repeat_n(c as u32, run_len));
        }
        let mut prev = vec![0u32; n];
        for l in (0..levels).rev() {
            let (_, words_at) = self.level(l);
            let (mut zero, mut one) = (0usize, self.u32_at(self.third, l) as usize);
            for (i, slot) in prev.iter_mut().enumerate() {
                if self.data[words_at + i / 64] >> (i % 64) & 1 == 0 {
                    *slot = cur[zero];
                    zero += 1;
                } else {
                    *slot = cur[one];
                    one += 1;
                }
            }
            std::mem::swap(&mut cur, &mut prev);
        }
        let mut next: Vec<u32> = (0..k).map(|c| self.data[c] as u32).collect();
        out.extend(cur.iter().map(|&c| {
            let id = next[c as usize];
            next[c as usize] += 1;
            TermId(id)
        }));
    }

    #[cfg(test)]
    fn to_vec(&self) -> Vec<TermId> {
        let mut out = Vec::new();
        self.decode_into(&mut out);
        out
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
        assert_eq!((0..v.len()).map(|r| b.get(r)).collect::<Vec<_>>(), v);
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
        assert_eq!(b.size_bytes(), 0, "base and length live in the struct");
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
        assert_eq!(b.kind, Kind::Wavelet);
        // Wavelet matrix: 2 levels of 300 bits plus rank samples, and 3
        // runs' base/length/start (~160 bytes). Raw u32 both ways: 2,400.
        assert!(b.size_bytes() < 200, "{}", b.size_bytes());
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
        assert_eq!(round_trip(&ids(&v)).kind, Kind::Wavelet);
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
