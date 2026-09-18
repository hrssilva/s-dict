use fst::raw::Node;
use fst::{Map, MapBuilder, Streamer};

use crate::TermId;
use crate::entries_arena::EntriesArena;

/// Binary search for the last transition of `node` whose accumulated
/// output value is `<= value` -- `None` if even the first transition
/// exceeds it. Transitions are sorted by non-decreasing output value
/// (see [`Block::term_at_into`]'s doc comment for why that's guaranteed,
/// not assumed), so this is a valid `partition_point`-style search rather
/// than a linear scan.
fn last_transition_at_most(node: Node<'_>, value: u64) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = node.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if node.transition(mid).out.value() <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.checked_sub(1)
}

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

    /// The design doc §5 Variant A reverse lookup: an O(block size) linear
    /// scan of `ids`. `Dictionary` no longer calls this in production --
    /// it moved to Variant B, storing `(BlockId, local_rank)` directly in
    /// the reverse directory (see `dictionary.rs`) after measuring both.
    /// Kept, `#[cfg(test)]`-only, purely as the comparison baseline the
    /// `perf_probe_local_rank_of` benchmark measures against.
    #[cfg(test)]
    pub(crate) fn local_rank_of(&self, id: TermId) -> Option<usize> {
        self.ids.iter().position(|&x| x == id)
    }

    /// Decodes the term at `local_rank` into `buf` (cleared first, its
    /// allocation reused across calls): true `select_b(local_rank)`
    /// (design doc §2/§5), not a linear scan of the block. Walks the FST
    /// from the root and, at each node, finds the last transition whose
    /// accumulated output value is `<= remaining value`, subtracting as
    /// it descends -- the "right-language cardinality" unranking
    /// algorithm §2 calls for. It's correct here specifically because we
    /// store each key's *local rank* as its FST output value, which is
    /// monotonically increasing in lexicographic key order.
    ///
    /// This used to call `raw::Fst::get_key_into` directly, which finds
    /// that "last transition `<= value`" via
    /// `node.transitions().take_while(..).last()` -- a *linear* scan
    /// through the node's transitions. Profiling with `perf record` (see
    /// DESIGN.md) showed that scan's generic-`Iterator` machinery, not
    /// the per-transition data decode, was the single largest cost inside
    /// it: ~41.5% of this function's total runtime, because it's
    /// O(branching factor) per byte of the term rather than
    /// O(log branching factor). [`last_transition_at_most`] replaces it
    /// with a binary search over `Node::transition(i)` by index, using
    /// the same monotonic-output precondition `get_key_into` itself
    /// already relies on for correctness (transitions sorted by
    /// non-decreasing accumulated output is what makes "the last one
    /// `<= value`" a well-defined greedy-descent step at all) -- not a
    /// new assumption, just exploited directly instead of discovered by
    /// scanning. Cost is now O(term length * log(branching factor)).
    pub(crate) fn term_at_into(&self, local_rank: usize, buf: &mut Vec<u8>) {
        buf.clear();
        let mut value = local_rank as u64;
        let raw_fst = self.fst.as_fst();
        let mut node = raw_fst.root();
        while value != 0 || !node.is_final() {
            let idx = last_transition_at_most(node, value)
                .expect("local_rank must be < len(); ranks are exactly the FST's output values");
            let t = node.transition(idx);
            value -= t.out.value();
            buf.push(t.inp);
            node = raw_fst.node(t.addr);
        }
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

    /// Stress test for `last_transition_at_most`'s binary search
    /// specifically: `sample_entries`' fixed-width numeric suffix keeps
    /// every node's branching factor at exactly 1 (only one possible next
    /// byte at each position), which can't exercise a search over more
    /// than one candidate. This dataset's digit suffix creates real
    /// branching (up to 10 transitions per node at each digit position),
    /// covering low/high/middle target indices within a node's
    /// transitions, not just the always-only-choice case.
    #[test]
    fn term_at_and_rank_round_trip_with_wide_branching() {
        let entries: Vec<(String, TermId)> = (0..3000)
            .map(|i| {
                (
                    format!("http://example.org/entity-{i:05}", i = i),
                    TermId(i as u32),
                )
            })
            .collect();
        let block = build_from(&entries);
        assert_eq!(block.len(), 3000);
        for (i, (t, id)) in entries.iter().enumerate() {
            assert_eq!(block.term_at(i), *t, "term_at mismatch at rank {i}");
            assert_eq!(block.rank(t), Some(i), "rank mismatch for {t:?}");
            assert_eq!(block.id_at(i), *id);
        }
        // Explicit first/last-of-block edge cases.
        assert_eq!(block.term_at(0), entries[0].0);
        assert_eq!(
            block.term_at(entries.len() - 1),
            entries[entries.len() - 1].0
        );
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

    /// Isolates `local_rank_of`'s cost (§5 Variant A's linear scan over
    /// `ids`) from `term_at_into`'s (the part already optimized to native
    /// unranking). A `get_term` call pays both in sequence; this measures
    /// each separately and their sum, to judge whether the scan is worth
    /// replacing with Variant B (`TermId -> (BlockId, local_rank)`, no
    /// scan) now that decode itself is cheap.
    #[test]
    #[ignore = "manual perf probe, not part of the regular suite"]
    fn perf_probe_local_rank_of() {
        const N: usize = 160; // midpoint of the default block bounds, 64..=256
        const ITERS: usize = 5_000;

        let entries = sample_entries(N);
        let block = build_from(&entries);
        let ids: Vec<TermId> = entries.iter().map(|(_, id)| *id).collect();

        let start = Instant::now();
        for _ in 0..ITERS {
            for &id in &ids {
                std::hint::black_box(block.local_rank_of(id));
            }
        }
        let scan_elapsed = start.elapsed();

        let mut buf = Vec::new();
        let start = Instant::now();
        for _ in 0..ITERS {
            for &id in &ids {
                let rank = block.local_rank_of(id).unwrap();
                block.term_at_into(rank, &mut buf);
                std::hint::black_box(&buf);
            }
        }
        let full_elapsed = start.elapsed();

        let total_ops = (ITERS * N) as u32;
        eprintln!(
            "block size {N}, {total_ops} total calls:\n\
             local_rank_of alone:            {scan_elapsed:?} ({:?}/op)\n\
             local_rank_of + term_at_into:    {full_elapsed:?} ({:?}/op)\n\
             scan's share of the full cost:   {:.1}%",
            scan_elapsed / total_ops,
            full_elapsed / total_ops,
            100.0 * scan_elapsed.as_secs_f64() / full_elapsed.as_secs_f64(),
        );
    }
}
