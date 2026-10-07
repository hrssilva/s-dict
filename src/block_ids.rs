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
    /// Run bases (`Runs`), or the single base (`Implicit`, `For`; empty
    /// when `len == 0`).
    bases: Box<[u32]>,
    /// `len` bit-packed entries of `entry_bits` each; empty for `Implicit`.
    words: Box<[u64]>,
    /// Offset bits; in `Runs`, the run index sits above them.
    off_bits: u8,
    entry_bits: u8,
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

impl BlockIds {
    pub(crate) fn new(ids: &[TermId]) -> BlockIds {
        let len = ids.len();
        if len == 0 {
            return BlockIds {
                len: 0,
                kind: Kind::Implicit,
                bases: Box::new([]),
                words: Box::new([]),
                off_bits: 0,
                entry_bits: 0,
            };
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

        if runs.len() == 1 {
            return BlockIds {
                len: len as u32,
                kind: Kind::Implicit,
                bases: Box::new([runs[0].0]),
                words: Box::new([]),
                off_bits: 0,
                entry_bits: 0,
            };
        }
        if !runs.is_empty() {
            let off_bits = bits(u64::from(max_offset));
            let entry_bits = bits(runs.len() as u64 - 1) + off_bits;
            let runs_size = 4 * runs.len() + 8 * words_for(len, entry_bits);
            if runs_size < for_size {
                let words = pack(
                    entries.iter().map(|&(run, off)| (u64::from(run) << off_bits) | u64::from(off)),
                    len,
                    entry_bits,
                );
                return BlockIds {
                    len: len as u32,
                    kind: Kind::Runs,
                    bases: runs.iter().map(|r| r.0).collect(),
                    words,
                    off_bits,
                    entry_bits,
                };
            }
        }
        BlockIds {
            len: len as u32,
            kind: Kind::For,
            bases: Box::new([min]),
            words: pack(ids.iter().map(|id| u64::from(id.0 - min)), len, for_bits),
            off_bits: for_bits,
            entry_bits: for_bits,
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
        if width == 0 {
            return 0;
        }
        let pos = rank * width;
        let (word, shift) = (pos / 64, pos % 64);
        let mut v = self.words[word] >> shift;
        if shift + width > 64 {
            v |= self.words[word + 1] << (64 - shift);
        }
        v & ((1u64 << width) - 1)
    }

    #[inline]
    pub(crate) fn get(&self, rank: usize) -> TermId {
        debug_assert!(rank < self.len());
        match self.kind {
            Kind::Implicit => TermId(self.bases[0] + rank as u32),
            Kind::For => TermId(self.bases[0] + self.read(rank) as u32),
            Kind::Runs => {
                let entry = self.read(rank);
                let offset = (entry & ((1u64 << self.off_bits) - 1)) as u32;
                TermId(self.bases[(entry >> self.off_bits) as usize] + offset)
            }
        }
    }

    pub(crate) fn to_vec(&self) -> Vec<TermId> {
        (0..self.len()).map(|r| self.get(r)).collect()
    }

    /// Logical size: bases plus packed words.
    pub(crate) fn size_bytes(&self) -> usize {
        self.bases.len() * 4 + self.words.len() * 8
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
        b
    }

    #[test]
    fn implicit_runs_and_for_round_trip() {
        assert_eq!(round_trip(&[]).size_bytes(), 0);
        let b = round_trip(&ids(&[100, 101, 102, 103]));
        assert_eq!(b.kind, Kind::Implicit);
        assert_eq!(b.size_bytes(), 4);
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
        // 2 run bits + 7 offset bits per entry, plus 3 bases: ~356 bytes,
        // vs 1,200 raw and ~830 for FOR over this ~3M-wide range.
        assert!(b.size_bytes() < 400);
        // No runs at all: FOR over a narrow range.
        let v: Vec<u32> = (0..500u32).map(|i| 7_000 + (i * 7919) % 997).collect();
        let b = round_trip(&ids(&v));
        assert_eq!(b.kind, Kind::For);
        // Decreasing ids: each its own run, so FOR wins.
        let v: Vec<u32> = (0..400u32).rev().collect();
        assert_eq!(round_trip(&ids(&v)).kind, Kind::For);
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
