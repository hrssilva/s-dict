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

/// A block's `rank <-> TermId` permutation in the smallest of three
/// encodings (module doc).
#[derive(Debug, Clone)]
pub(crate) struct BlockIds(Repr);

#[derive(Debug, Clone)]
enum Repr {
    /// A single run: `id = base + rank`, no per-entry data.
    Implicit { len: u32, base: u32 },
    Wavelet(Wavelet),
    For(FrameOfReference),
}

/// Run labels as a wavelet matrix, in one allocation (`data`, in words):
/// the run table (`base | len << 32` per run, runs in base order), each
/// run's bottom-level start (two `u32` per word), each level's zero count
/// (two per word), then the levels, each `sample words + bit words`.
/// Section offsets follow from `runs` and `levels`; only the levels' one,
/// used on every query, is stored (deriving it per level cost ~2%).
#[derive(Debug, Clone)]
struct Wavelet {
    len: u32,
    /// Word offset of the first level: `zeros_at() + ceil(levels / 2)`.
    levels_at: u32,
    /// At most `MAX_RUNS`.
    runs: u16,
    /// `bits(runs - 1)`.
    levels: u8,
    data: Box<[u64]>,
}

/// The fallback: `id = base + offset`, `width`-bit offsets from word 0 of
/// `data`, then ranks in id order (`rank_bits` each) after them.
#[derive(Debug, Clone)]
struct FrameOfReference {
    len: u32,
    base: u32,
    width: u8,
    rank_bits: u8,
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
            return BlockIds(Repr::Implicit { len: 0, base: 0 });
        }
        let (min, max) = ids.iter().fold((u32::MAX, 0), |(lo, hi), id| (lo.min(id.0), hi.max(id.0)));

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
            return BlockIds(Repr::Implicit {
                len: len as u32,
                base: runs[0].0,
            });
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
                return BlockIds(Repr::Wavelet(Wavelet::new(&runs, &labels, levels)));
            }
        }
        BlockIds(Repr::For(FrameOfReference::new(ids, min, for_bits, rank_bits)))
    }

    /// `len` consecutive ids from `base`, in rank order.
    pub(crate) fn run(len: usize, base: u32) -> BlockIds {
        let len = u32::try_from(len).expect("block length fits u32");
        debug_assert!(base.checked_add(len).is_some(), "the run stays within u32");
        BlockIds(Repr::Implicit { len, base })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        match &self.0 {
            Repr::Implicit { len, .. } => *len as usize,
            Repr::Wavelet(w) => w.len as usize,
            Repr::For(f) => f.len as usize,
        }
    }

    #[inline]
    pub(crate) fn get(&self, rank: usize) -> TermId {
        debug_assert!(rank < self.len());
        match &self.0 {
            Repr::Implicit { base, .. } => TermId(base + rank as u32),
            Repr::Wavelet(w) => w.get(rank),
            Repr::For(f) => f.get(rank),
        }
    }

    /// The rank of `id` in this block, if present.
    #[inline]
    pub(crate) fn rank_of(&self, id: TermId) -> Option<usize> {
        match &self.0 {
            &Repr::Implicit { len, base } => (id.0 >= base && id.0 - base < len).then(|| (id.0 - base) as usize),
            Repr::Wavelet(w) => w.rank_of(id.0),
            Repr::For(f) => f.rank_of(id.0),
        }
    }

    /// Appends all ids in rank order to `out`.
    pub(crate) fn decode_into(&self, out: &mut Vec<TermId>) {
        match &self.0 {
            &Repr::Implicit { len, base } => out.extend((0..len).map(|rank| TermId(base + rank))),
            Repr::Wavelet(w) => w.decode_into(out),
            Repr::For(f) => out.extend((0..f.len as usize).map(|r| f.get(r))),
        }
    }

    #[cfg(test)]
    fn to_vec(&self) -> Vec<TermId> {
        let mut out = Vec::new();
        self.decode_into(&mut out);
        out
    }

    /// Logical size: the single data allocation.
    pub(crate) fn size_bytes(&self) -> usize {
        match &self.0 {
            Repr::Implicit { .. } => 0,
            Repr::Wavelet(w) => w.data.len() * 8,
            Repr::For(f) => f.data.len() * 8,
        }
    }
}

impl FrameOfReference {
    fn new(ids: &[TermId], min: u32, width: u8, rank_bits: u8) -> FrameOfReference {
        let len = ids.len();
        let mut by_id: Vec<u32> = (0..len as u32).collect();
        by_id.sort_unstable_by_key(|&r| ids[r as usize].0);
        let mut data = pack(ids.iter().map(|id| u64::from(id.0 - min)), len, width);
        data.extend(pack(by_id.iter().map(|&r| u64::from(r)), len, rank_bits));
        FrameOfReference {
            len: len as u32,
            base: min,
            width,
            rank_bits,
            data: data.into_boxed_slice(),
        }
    }

    #[inline]
    fn get(&self, rank: usize) -> TermId {
        TermId(self.base + read_bits(&self.data, rank * self.width as usize, self.width as usize) as u32)
    }

    /// Binary search over the ranks stored in id order.
    #[inline]
    fn rank_of(&self, id: u32) -> Option<usize> {
        let rank_bits = self.rank_bits as usize;
        let inverse = &self.data[words_for(self.len as usize, self.width)..];
        let at = |k: usize| read_bits(inverse, k * rank_bits, rank_bits) as usize;
        let (mut lo, mut hi) = (0usize, self.len as usize);
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

impl Wavelet {
    /// `runs` in greedy order, `labels` per rank in that order.
    fn new(runs: &[(u32, u64)], labels: &[u32], levels: u8) -> Wavelet {
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
            for (word, chunk) in words.iter_mut().zip(cur.chunks(64)) {
                *word = chunk.iter().enumerate().fold(0, |w, (j, &c)| w | u64::from(c >> shift & 1) << j);
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
            // Stable partition into the spare buffer, zeros first --
            // branch-free: label bits are effectively random.
            let (mut zero, mut one) = (0usize, (len as u32 - ones) as usize);
            for &c in &cur {
                let bit = (c >> shift & 1) as usize;
                spare[if bit == 0 { zero } else { one }] = c;
                zero += 1 - bit;
                one += bit;
            }
            std::mem::swap(&mut cur, &mut spare);
        }
        // Bottom-level start of each label (equal labels are contiguous).
        let mut starts = vec![0u32; k];
        for (i, &c) in cur.iter().enumerate().rev() {
            starts[c as usize] = i as u32;
        }
        push_u32s(&mut data, &starts);
        push_u32s(&mut data, &zeros);
        data.extend_from_slice(&level_data);
        let runs = u16::try_from(k).expect("at most MAX_RUNS runs");
        let levels_at = (k + k.div_ceil(2) + (levels as usize).div_ceil(2)) as u32;
        debug_assert_eq!(data.len(), levels_at as usize + levels as usize * (sample_words + bit_words));
        Wavelet {
            len: len as u32,
            levels_at,
            runs,
            levels,
            data: data.into_boxed_slice(),
        }
    }

    /// Word offset of the per-run bottom-level starts (after the run table).
    #[inline]
    fn starts_at(&self) -> usize {
        self.runs as usize
    }

    /// Word offset of the per-level zero counts.
    #[inline]
    fn zeros_at(&self) -> usize {
        self.starts_at() + (self.runs as usize).div_ceil(2)
    }


    #[inline]
    fn u32_at(&self, section: usize, index: usize) -> u32 {
        (self.data[section + index / 2] >> (32 * (index % 2))) as u32
    }

    /// Bottom-level start of run `c`.
    #[inline]
    fn start(&self, c: usize) -> usize {
        self.u32_at(self.starts_at(), c) as usize
    }

    /// Zeros in level `l`.
    #[inline]
    fn zeros(&self, l: usize) -> usize {
        self.u32_at(self.zeros_at(), l) as usize
    }

    /// `(sample section start, bit words start)` of level `l`.
    #[inline]
    fn level(&self, l: usize) -> (usize, usize) {
        let (sample_words, bit_words) = level_words(self.len as usize);
        let at = self.levels_at as usize + l * (sample_words + bit_words);
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
        if !i.is_multiple_of(64) {
            r += (self.data[words_at + word] & ((1u64 << (i % 64)) - 1)).count_ones() as usize;
        }
        r
    }

    /// Position in level `l` of the `j`-th (0-based) one (`ones`) or zero.
    fn select(&self, l: usize, j: usize, ones: bool) -> usize {
        let (samples_at, words_at) = self.level(l);
        let groups = (self.len as usize).div_ceil(64).div_ceil(SAMPLE_WORDS);
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

    /// `id(rank) = base[c] + (occurrences of c before rank)`: one top-down
    /// `rank` pass recovers the label `c` and its occurrence index.
    #[inline]
    fn get(&self, rank: usize) -> TermId {
        let (mut i, mut c) = (rank, 0usize);
        for l in 0..self.levels as usize {
            let (_, words_at) = self.level(l);
            let bit = (self.data[words_at + i / 64] >> (i % 64) & 1) as usize;
            c = c << 1 | bit;
            let ones = self.rank1(l, i);
            i = if bit == 0 { i - ones } else { self.zeros(l) + ones };
        }
        TermId(self.data[c] as u32 + (i - self.start(c)) as u32)
    }

    /// The run holding `id` (binary search over run bases), then one
    /// bottom-up `select` pass to its rank.
    #[inline]
    fn rank_of(&self, id: u32) -> Option<usize> {
        let (mut lo, mut hi) = (0usize, self.runs as usize);
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
        let levels = self.levels as usize;
        let mut i = self.start(c) + offset;
        for l in (0..levels).rev() {
            i = if c >> (levels - 1 - l) & 1 == 0 {
                self.select(l, i, false)
            } else {
                self.select(l, i - self.zeros(l), true)
            };
        }
        Some(i)
    }

    /// Bulk decode -- per-rank `get` would cost a `rank` query per level
    /// per entry, and merges call this on every block they touch: start
    /// from the bottom level (labels sorted, each contiguous), undo each
    /// level's stable partition bottom-up, then hand out each run's
    /// consecutive ids in rank order.
    fn decode_into(&self, out: &mut Vec<TermId>) {
        let (n, k, levels) = (self.len as usize, self.runs as usize, self.levels as usize);
        let mut by_start: Vec<usize> = (0..k).collect();
        by_start.sort_unstable_by_key(|&c| self.start(c));
        let mut cur: Vec<u32> = Vec::with_capacity(n);
        for &c in &by_start {
            let run_len = (self.data[c] >> 32) as usize;
            cur.extend(std::iter::repeat_n(c as u32, run_len));
        }
        let mut prev = vec![0u32; n];
        for l in (0..levels).rev() {
            let (_, words_at) = self.level(l);
            let words = &self.data[words_at..words_at + n.div_ceil(64)];
            let (mut zero, mut one) = (0usize, self.zeros(l));
            // Branch-free: label bits are effectively random.
            for (chunk, &word) in prev.chunks_mut(64).zip(words) {
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let bit = (word >> j & 1) as usize;
                    *slot = cur[if bit == 0 { zero } else { one }];
                    zero += 1 - bit;
                    one += bit;
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
            assert_eq!(b.rank_of(id), Some(rank), "{:?} id {id:?}", b.0);
        }
        // Absent ids: just outside each id, where not present.
        let present: std::collections::HashSet<u32> = v.iter().map(|id| id.0).collect();
        for &TermId(id) in v {
            for probe in [id.wrapping_sub(1), id.wrapping_add(1)] {
                if !present.contains(&probe) {
                    assert_eq!(b.rank_of(TermId(probe)), None, "{:?} probe {probe}", b.0);
                }
            }
        }
        b
    }

    #[test]
    fn implicit_runs_and_for_round_trip() {
        assert_eq!(round_trip(&[]).size_bytes(), 0);
        let b = round_trip(&ids(&[100, 101, 102, 103]));
        assert!(matches!(b.0, Repr::Implicit { .. }));
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
        assert!(matches!(b.0, Repr::Wavelet(_)));
        // Wavelet matrix: 2 levels of 300 bits plus rank samples, and 3
        // runs' base/length/start (~160 bytes). Raw u32 both ways: 2,400.
        assert!(b.size_bytes() < 200, "{}", b.size_bytes());
        // No runs at all: FOR over a narrow range.
        let v: Vec<u32> = (0..500u32).map(|i| 7_000 + (i * 7919) % 997).collect();
        let b = round_trip(&ids(&v));
        assert!(matches!(b.0, Repr::For(_)));
        // Decreasing ids: each its own run, so FOR wins.
        let v: Vec<u32> = (0..400u32).rev().collect();
        assert!(matches!(round_trip(&ids(&v)).0, Repr::For(_)));
    }

    /// One per block: no encoding's fields may grow every block.
    #[test]
    fn stays_small() {
        assert!(std::mem::size_of::<BlockIds>() <= 40, "{}", std::mem::size_of::<BlockIds>());
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
        assert!(matches!(round_trip(&ids(&v)).0, Repr::Wavelet(_)));
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
