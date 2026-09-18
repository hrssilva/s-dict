use fst::{Map, MapBuilder, Streamer};

use crate::TermId;
use crate::entries_arena::EntriesArena;

/// A single lexical block: `fst` is `F_b`, a minimal per-block finite-state
/// map from each sorted term to its local rank (design doc §2-3, Stage 3),
/// and `ids` is the parallel `P_b` external-id sequence. `min_key` caches
/// the block's smallest term so routing/directory maintenance never has to
/// decode the FST just to compare against it (design doc §10: keep hot
/// metadata separate from the compressed cold payload).
///
/// The FST is immutable once built: there is no in-place insert or
/// remove. Every content change decodes the block (`decode_into`), mutates
/// the result, and calls [`Block::build`] to replace the whole block --
/// exactly the update procedure the design doc prescribes for FST-backed
/// blocks (§6).
#[derive(Debug)]
pub(crate) struct Block {
    fst: Map<Vec<u8>>,
    ids: Vec<TermId>,
    min_key: Box<str>,
}

impl Block {
    /// Builds an immutable block from an already sorted, deduplicated,
    /// exact-size sequence of `(term, id)` pairs. The only way to
    /// construct or change a block's content. Takes a borrowing iterator
    /// (not `&[(Box<str>, TermId)]`) so callers never need to materialize
    /// an owned string per entry just to rebuild a block -- design doc
    /// §10's "avoid a pointer-heavy... prefer compact arrays" applied to
    /// the rebuild path itself, not just the final representation.
    pub(crate) fn build<'a>(entries: impl ExactSizeIterator<Item = (&'a str, TermId)>) -> Block {
        let mut builder =
            MapBuilder::new(Vec::new()).expect("building into an in-memory Vec<u8> cannot fail");
        let mut ids = Vec::with_capacity(entries.len());
        let mut min_key: Option<Box<str>> = None;
        for (rank, (term, id)) in entries.enumerate() {
            if rank == 0 {
                min_key = Some(term.into());
            }
            builder
                .insert(term.as_bytes(), rank as u64)
                .expect("entries are sorted and unique by construction");
            ids.push(id);
        }
        let fst_bytes = builder
            .into_inner()
            .expect("finishing an in-memory Vec<u8> writer cannot fail");
        let fst = Map::new(fst_bytes).expect("bytes were just built by MapBuilder for this Map");

        Block {
            fst,
            ids,
            min_key: min_key.unwrap_or_default(),
        }
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

    /// Decodes the term at `local_rank` into `buf` (cleared first, its
    /// allocation reused across calls): true `select_b(local_rank)`
    /// (design doc §2/§5), not a linear scan. `raw::Fst::get_key_into`
    /// walks from the root and, at each node, takes the last transition
    /// whose output value is `<= remaining value`, subtracting as it
    /// descends -- the "right-language cardinality" unranking algorithm
    /// §2 calls for, already built into the `fst` crate and exposed at
    /// the `raw` layer. It's correct here specifically because we store
    /// each key's *local rank* as its FST output value, which is
    /// monotonically increasing in lexicographic key order -- exactly
    /// `get_key`'s documented precondition. Cost is O(term length *
    /// branching factor per node), not O(local_rank)/O(block size).
    pub(crate) fn term_at_into(&self, local_rank: usize, buf: &mut Vec<u8>) {
        buf.clear();
        let found = self.fst.as_fst().get_key_into(local_rank as u64, buf);
        debug_assert!(
            found,
            "local_rank must be < len(); ranks are exactly the FST's output values"
        );
    }

    /// Convenience wrapper over [`Block::term_at_into`] that allocates a
    /// fresh `String`, for callers that don't have (or don't want to
    /// manage) a reusable buffer.
    pub(crate) fn term_at(&self, local_rank: usize) -> String {
        let mut buf = Vec::new();
        self.term_at_into(local_rank, &mut buf);
        String::from_utf8(buf).expect("keys were inserted as valid UTF-8 str bytes")
    }

    /// Appends every `(term, id)` pair in sorted order onto `out` (does
    /// *not* clear it first -- see [`EntriesArena`]'s append-only design;
    /// callers that want fresh content call `out.clear()` themselves).
    /// O(block size); used to rebuild a block (insert/remove/split/merge),
    /// not for point queries.
    pub(crate) fn decode_into(&self, out: &mut EntriesArena) {
        let mut stream = self.fst.stream();
        let mut i = 0;
        while let Some((key_bytes, _rank)) = stream.next() {
            let term = std::str::from_utf8(key_bytes)
                .expect("keys were inserted as valid UTF-8 str bytes");
            out.push(term, self.ids[i]);
            i += 1;
        }
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

    fn sample_entries(n: usize) -> Vec<(String, TermId)> {
        (0..n)
            .map(|i| (format!("http://example.org/term-{i:06}"), TermId(i as u32)))
            .collect()
    }

    fn build_from(entries: &[(String, TermId)]) -> Block {
        Block::build(entries.iter().map(|(t, id)| (t.as_str(), *id)))
    }

    #[test]
    fn build_and_decode_round_trip() {
        let entries = sample_entries(50);
        let block = build_from(&entries);
        assert_eq!(block.len(), 50);
        let mut arena = EntriesArena::new();
        block.decode_into(&mut arena);
        assert_eq!(arena.len(), 50);
        for (i, (t, id)) in entries.iter().enumerate() {
            assert_eq!(arena.term_at(i), t.as_str());
            assert_eq!(arena.id_at(i), *id);
            assert_eq!(block.term_at(i), *t);
            assert_eq!(block.rank(t), Some(i));
            assert_eq!(block.id_at(i), *id);
        }
    }

    #[test]
    fn decode_into_appends_without_clearing() {
        let a = build_from(&sample_entries(3));
        let mut arena = EntriesArena::new();
        arena.push("zzz-preexisting", TermId(999));
        a.decode_into(&mut arena);
        assert_eq!(arena.len(), 4);
        assert_eq!(arena.term_at(0), "zzz-preexisting");
    }

    /// Breaks the ~83us/insert measured in
    /// `tests/dictionary.rs::perf_probe_default_block_bounds` into its
    /// components: decoding an existing block, building a fresh one from
    /// scratch (what a single insert/remove pays), and a floor cost of
    /// just allocating N owned `String`s with no FST/automaton work at
    /// all, to see which part of the decode-mutate-rebuild cycle (design
    /// doc §6) actually dominates.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_block_rebuild_breakdown() {
        const N: usize = 160; // midpoint of the default block bounds, 64..=256
        const ITERS: usize = 5_000;

        let entries = sample_entries(N);
        let block = build_from(&entries);

        let mut arena = EntriesArena::new();
        let start = Instant::now();
        for _ in 0..ITERS {
            arena.clear();
            block.decode_into(&mut arena);
            std::hint::black_box(&arena);
        }
        let decode_elapsed = start.elapsed();

        let start = Instant::now();
        for _ in 0..ITERS {
            let b = build_from(&entries);
            std::hint::black_box(&b);
        }
        let build_elapsed = start.elapsed();

        let start = Instant::now();
        for _ in 0..ITERS {
            let cloned: Vec<(String, TermId)> =
                entries.iter().map(|(t, id)| (t.clone(), *id)).collect();
            std::hint::black_box(&cloned);
        }
        let clone_elapsed = start.elapsed();

        eprintln!(
            "block size {N}, {ITERS} iters:\n\
             decode_into:                 {decode_elapsed:?} ({:?}/op)\n\
             Block::build (full rebuild): {build_elapsed:?} ({:?}/op)\n\
             pure String clone floor:     {clone_elapsed:?} ({:?}/op)\n\
             decode+build (~one insert):  {:?}/op",
            decode_elapsed / ITERS as u32,
            build_elapsed / ITERS as u32,
            clone_elapsed / ITERS as u32,
            (decode_elapsed + build_elapsed) / ITERS as u32,
        );
    }
}
