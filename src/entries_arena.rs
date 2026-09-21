use crate::TermId;

/// A flat, struct-of-arrays scratch buffer for decode-mutate-rebuild block
/// operations: term bytes concatenated into one contiguous buffer, with
/// parallel offset and id arrays, instead of `Vec<(Box<str>, TermId)>`
/// (one heap allocation per entry). Design doc §10: "avoid representations
/// resembling `State { Vec<Transition>, Box<State> }`... prefer compact
/// indices and packed arrays," applied to the rebuild scratch path, not
/// just the final block representations -- decoding a block into this
/// arena and rebuilding from it costs zero *incremental* allocations once
/// its three backing `Vec`s have grown to their high-water mark, versus
/// one `Box<str>` allocation per entry before.
///
/// Append-only by design: `push` never overwrites, so decoding two blocks
/// back-to-back into the same arena (a block merge) concatenates them for
/// free. Callers that want fresh content call `clear()` first.
#[derive(Debug)]
pub(crate) struct EntriesArena {
    text: String,
    /// End offsets into `text`, one per ID; an empty arena needs no allocation.
    offsets: Vec<u32>,
    ids: Vec<TermId>,
}

impl Default for EntriesArena {
    fn default() -> Self {
        Self::new()
    }
}

impl EntriesArena {
    pub(crate) fn new() -> Self {
        EntriesArena {
            text: String::new(),
            offsets: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// Empties the arena, keeping its backing allocations.
    pub(crate) fn clear(&mut self) {
        self.text.clear();
        self.offsets.clear();
        self.ids.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn term_at(&self, i: usize) -> &str {
        let start = if i == 0 {
            0
        } else {
            self.offsets[i - 1] as usize
        };
        let end = self.offsets[i] as usize;
        &self.text[start..end]
    }

    pub(crate) fn id_at(&self, i: usize) -> TermId {
        self.ids[i]
    }

    /// Appends one entry to the end.
    pub(crate) fn push(&mut self, term: &str, id: TermId) {
        self.text.push_str(term);
        self.offsets
            .push(u32::try_from(self.text.len()).expect("scratch arena exceeds 32-bit offsets"));
        self.ids.push(id);
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = (&str, TermId)> {
        (0..self.len()).map(move |i| (self.term_at(i), self.id_at(i)))
    }

    /// The `[start, end)` sub-range as `(term, id)` pairs, for splitting a
    /// decoded block in two without copying into separate arenas.
    pub(crate) fn range(
        &self,
        start: usize,
        end: usize,
    ) -> impl ExactSizeIterator<Item = (&str, TermId)> {
        (start..end).map(move |i| (self.term_at(i), self.id_at(i)))
    }
}

/// Merges `old` (already sorted) with `new` (sorted, disjoint from `old`)
/// into `out` (cleared first) in one linear pass. Used for both a single
/// insert (`new` is a one-element slice) and a batch's per-block group
/// (`new` is the group), so the two share one merge implementation.
pub(crate) fn merge_into(old: &EntriesArena, new: &[(&str, TermId)], out: &mut EntriesArena) {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    while i < old.len() && j < new.len() {
        if old.term_at(i) <= new[j].0 {
            out.push(old.term_at(i), old.id_at(i));
            i += 1;
        } else {
            out.push(new[j].0, new[j].1);
            j += 1;
        }
    }
    while i < old.len() {
        out.push(old.term_at(i), old.id_at(i));
        i += 1;
    }
    for &(term, id) in &new[j..] {
        out.push(term, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taken_arena_leaves_allocation_free_default_and_retains_empty_unicode_terms() {
        let mut arena = EntriesArena::new();
        arena.push("", TermId(0));
        arena.push("é中", TermId(1));
        arena.push("", TermId(2));
        let mut taken = std::mem::take(&mut arena);
        assert_eq!(arena.text.capacity(), 0);
        assert_eq!(arena.offsets.capacity(), 0);
        assert_eq!(arena.ids.capacity(), 0);
        assert_eq!(taken.term_at(0), "");
        assert_eq!(taken.term_at(1), "é中");
        assert_eq!(taken.term_at(2), "");
        let capacity = taken.text.capacity();
        taken.clear();
        taken.push("中", TermId(3));
        assert_eq!(taken.term_at(0), "中");
        assert_eq!(taken.text.capacity(), capacity);
    }

    #[test]
    fn push_and_read_back() {
        let mut arena = EntriesArena::new();
        arena.push("a", TermId(0));
        arena.push("ab", TermId(1));
        arena.push("abc", TermId(2));
        assert_eq!(arena.len(), 3);
        assert_eq!(arena.term_at(0), "a");
        assert_eq!(arena.term_at(1), "ab");
        assert_eq!(arena.term_at(2), "abc");
        assert_eq!(arena.id_at(1), TermId(1));
        let collected: Vec<(&str, TermId)> = arena.iter().collect();
        assert_eq!(
            collected,
            vec![("a", TermId(0)), ("ab", TermId(1)), ("abc", TermId(2))]
        );
    }

    #[test]
    fn clear_resets_end_offsets() {
        let mut arena = EntriesArena::new();
        arena.push("x", TermId(0));
        arena.clear();
        assert_eq!(arena.len(), 0);
        arena.push("y", TermId(1));
        assert_eq!(arena.term_at(0), "y");
    }

    #[test]
    fn range_slices_correctly() {
        let mut arena = EntriesArena::new();
        for i in 0..5 {
            arena.push(&i.to_string(), TermId(i));
        }
        let mid: Vec<(&str, TermId)> = arena.range(1, 4).collect();
        assert_eq!(
            mid,
            vec![("1", TermId(1)), ("2", TermId(2)), ("3", TermId(3))]
        );
    }

    #[test]
    fn append_without_clear_concatenates() {
        let mut a = EntriesArena::new();
        a.push("a", TermId(0));
        let mut b = EntriesArena::new();
        b.push("b", TermId(1));
        // Simulate decoding two blocks back-to-back into one arena.
        a.push(b.term_at(0), b.id_at(0));
        assert_eq!(a.len(), 2);
        assert_eq!(a.term_at(0), "a");
        assert_eq!(a.term_at(1), "b");
    }

    #[test]
    fn merge_into_interleaves_correctly() {
        let mut old = EntriesArena::new();
        old.push("b", TermId(1));
        old.push("d", TermId(3));
        let new = [("a", TermId(0)), ("c", TermId(2)), ("e", TermId(4))];
        let mut out = EntriesArena::new();
        merge_into(&old, &new, &mut out);
        let collected: Vec<(&str, TermId)> = out.iter().collect();
        assert_eq!(
            collected,
            vec![
                ("a", TermId(0)),
                ("b", TermId(1)),
                ("c", TermId(2)),
                ("d", TermId(3)),
                ("e", TermId(4)),
            ]
        );
    }

    #[test]
    fn merge_into_reuses_and_overwrites_prior_content() {
        let mut old = EntriesArena::new();
        old.push("m", TermId(0));
        let mut out = EntriesArena::new();
        out.push("stale", TermId(99)); // must be cleared by merge_into
        merge_into(&old, &[("n", TermId(1))], &mut out);
        let collected: Vec<(&str, TermId)> = out.iter().collect();
        assert_eq!(collected, vec![("m", TermId(0)), ("n", TermId(1))]);
    }
}
