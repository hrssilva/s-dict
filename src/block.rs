use fst::{Map, MapBuilder, Streamer};

use crate::TermId;

/// A single lexical block: `fst` is `F_b`, a minimal per-block finite-state
/// map from each sorted term to its local rank (design doc §2-3, Stage 3),
/// and `ids` is the parallel `P_b` external-id sequence. `min_key` caches
/// the block's smallest term so routing/directory maintenance never has to
/// decode the FST just to compare against it (design doc §10: keep hot
/// metadata separate from the compressed cold payload).
///
/// The FST is immutable once built: there is no in-place insert or
/// remove. Every content change decodes the block (`decode_all`), mutates
/// the resulting plain `Vec`, and calls [`Block::build`] to replace the
/// whole block -- exactly the update procedure the design doc prescribes
/// for FST-backed blocks (§6).
#[derive(Debug)]
pub(crate) struct Block {
    fst: Map<Vec<u8>>,
    ids: Vec<TermId>,
    min_key: Box<str>,
}

impl Block {
    /// Builds an immutable block from already sorted, deduplicated
    /// `(term, id)` pairs. The only way to construct or change a block's
    /// content.
    pub(crate) fn build(sorted_entries: &[(Box<str>, TermId)]) -> Block {
        let mut builder =
            MapBuilder::new(Vec::new()).expect("building into an in-memory Vec<u8> cannot fail");
        let mut ids = Vec::with_capacity(sorted_entries.len());
        for (rank, (term, id)) in sorted_entries.iter().enumerate() {
            builder
                .insert(term.as_bytes(), rank as u64)
                .expect("entries are sorted and unique by construction");
            ids.push(*id);
        }
        let fst_bytes = builder
            .into_inner()
            .expect("finishing an in-memory Vec<u8> writer cannot fail");
        let fst = Map::new(fst_bytes).expect("bytes were just built by MapBuilder for this Map");
        let min_key = sorted_entries
            .first()
            .map(|(term, _)| term.clone())
            .unwrap_or_default();

        Block { fst, ids, min_key }
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// The smallest term in the block. Only meaningful for a non-empty
    /// block; callers rely on the invariant that no directory entry ever
    /// references an empty block.
    pub(crate) fn min_key(&self) -> &str {
        &self.min_key
    }

    /// `Some(local_rank)` if `term` is present.
    pub(crate) fn rank(&self, term: &str) -> Option<usize> {
        self.fst.get(term.as_bytes()).map(|rank| rank as usize)
    }

    pub(crate) fn id_at(&self, local_rank: usize) -> TermId {
        self.ids[local_rank]
    }

    pub(crate) fn local_rank_of(&self, id: TermId) -> Option<usize> {
        self.ids.iter().position(|&x| x == id)
    }

    /// Decodes the term at `local_rank`: true `select_b(local_rank)`
    /// (design doc §2/§5), not a linear scan. `raw::Fst::get_key` walks
    /// from the root and, at each node, takes the last transition whose
    /// output value is `<= remaining value`, subtracting as it descends --
    /// the "right-language cardinality" unranking algorithm §2 calls for,
    /// already built into the `fst` crate and exposed at the `raw` layer.
    /// It's correct here specifically because we store each key's *local
    /// rank* as its FST output value, which is monotonically increasing in
    /// lexicographic key order -- exactly `get_key`'s documented
    /// precondition. Cost is O(term length * branching factor per node),
    /// not O(local_rank)/O(block size).
    pub(crate) fn term_at(&self, local_rank: usize) -> String {
        let key_bytes = self
            .fst
            .as_fst()
            .get_key(local_rank as u64)
            .expect("local_rank must be < len(); ranks are exactly the FST's output values");
        String::from_utf8(key_bytes).expect("keys were inserted as valid UTF-8 str bytes")
    }

    /// Decodes every `(term, id)` pair in sorted order into `out`,
    /// clearing it first but keeping its allocation. O(block size); used
    /// to rebuild a block (insert/remove/split/merge), not for point
    /// queries. Takes a caller-owned buffer rather than returning a fresh
    /// `Vec` so hot call sites can pass in a reused scratch buffer instead
    /// of allocating on every mutation (design doc §6/§10: block rebuilds
    /// should use a reusable scratch arena, bounded by block size, rather
    /// than one-off heap allocations per rebuild).
    pub(crate) fn decode_into(&self, out: &mut Vec<(Box<str>, TermId)>) {
        out.clear();
        out.reserve(self.ids.len());
        let mut stream = self.fst.stream();
        let mut i = 0;
        while let Some((key_bytes, _rank)) = stream.next() {
            let term: Box<str> = std::str::from_utf8(key_bytes)
                .expect("keys were inserted as valid UTF-8 str bytes")
                .into();
            out.push((term, self.ids[i]));
            i += 1;
        }
    }

    /// Convenience wrapper over [`Block::decode_into`] for call sites that
    /// aren't hot enough to bother threading a reused scratch buffer
    /// through (e.g. the less-frequent "absorbed" side of a block merge).
    pub(crate) fn decode_all(&self) -> Vec<(Box<str>, TermId)> {
        let mut out = Vec::new();
        self.decode_into(&mut out);
        out
    }

    /// Logical byte size of this block's compressed representation (the
    /// raw FST bytes), for design doc §14 "bits per term" space
    /// measurements. Excludes `ids`/`min_key`, which
    /// [`crate::front_coded_block::FrontCodedBlock`] also carries
    /// identically, so they cancel out of an apples-to-apples comparison;
    /// callers wanting the whole block's footprint add those back in.
    pub(crate) fn payload_bytes(&self) -> usize {
        self.fst.as_fst().as_bytes().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn sample_entries(n: usize) -> Vec<(Box<str>, TermId)> {
        (0..n)
            .map(|i| {
                (
                    format!("http://example.org/term-{i:06}").into_boxed_str(),
                    TermId(i as u32),
                )
            })
            .collect()
    }

    /// Breaks the ~83us/insert measured in
    /// `tests/dictionary.rs::perf_probe_default_block_bounds` into its
    /// components: decoding an existing block, building a fresh one from
    /// scratch (what a single insert/remove pays), and a floor cost of
    /// just allocating N owned `Box<str>`s with no FST/automaton work at
    /// all, to see which part of the decode-mutate-rebuild cycle (design
    /// doc §6) actually dominates.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_block_rebuild_breakdown() {
        const N: usize = 160; // midpoint of the default block bounds, 64..=256
        const ITERS: usize = 5_000;

        let entries = sample_entries(N);
        let block = Block::build(&entries);

        let mut scratch = Vec::new();
        let start = Instant::now();
        for _ in 0..ITERS {
            block.decode_into(&mut scratch);
            std::hint::black_box(&scratch);
        }
        let decode_elapsed = start.elapsed();

        let start = Instant::now();
        for _ in 0..ITERS {
            let b = Block::build(&entries);
            std::hint::black_box(&b);
        }
        let build_elapsed = start.elapsed();

        let start = Instant::now();
        for _ in 0..ITERS {
            let cloned: Vec<(Box<str>, TermId)> =
                entries.iter().map(|(t, id)| (t.clone(), *id)).collect();
            std::hint::black_box(&cloned);
        }
        let clone_elapsed = start.elapsed();

        eprintln!(
            "block size {N}, {ITERS} iters:\n\
             decode_into:                 {decode_elapsed:?} ({:?}/op)\n\
             Block::build (full rebuild): {build_elapsed:?} ({:?}/op)\n\
             pure Box<str> clone floor:   {clone_elapsed:?} ({:?}/op)\n\
             decode+build (~one insert):  {:?}/op",
            decode_elapsed / ITERS as u32,
            build_elapsed / ITERS as u32,
            clone_elapsed / ITERS as u32,
            (decode_elapsed + build_elapsed) / ITERS as u32,
        );
    }
}
