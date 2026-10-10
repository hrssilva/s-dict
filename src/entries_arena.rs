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
    /// Concatenated term bytes (decoded from blocks, so not re-validated as
    /// UTF-8: see `crate::packed_automaton::decoded_term`).
    text: Vec<u8>,
    /// End offsets into `text`, one per ID; an empty arena needs no
    /// allocation. `usize`: a rebuild may merge a large batch run into one
    /// block's entries (`crate::limits`).
    offsets: Vec<usize>,
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
            text: Vec::new(),
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

    pub(crate) fn term_at(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.offsets[i - 1] };
        let end = self.offsets[i];
        &self.text[start..end]
    }

    pub(crate) fn id_at(&self, i: usize) -> TermId {
        self.ids[i]
    }

    /// Appends one entry to the end.
    pub(crate) fn push(&mut self, term: &[u8], id: TermId) {
        self.text.extend_from_slice(term);
        self.offsets.push(self.text.len());
        self.ids.push(id);
    }

    /// The terms of entries `range`, for building a block from part of a
    /// decoded one.
    pub(crate) fn terms(&self, range: std::ops::Range<usize>) -> impl ExactSizeIterator<Item = &[u8]> {
        range.map(move |i| self.term_at(i))
    }

    /// The ids of entries `range`.
    pub(crate) fn ids(&self, range: std::ops::Range<usize>) -> &[TermId] {
        &self.ids[range]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taken_arena_leaves_allocation_free_default_and_retains_empty_unicode_terms() {
        let mut arena = EntriesArena::new();
        arena.push("".as_bytes(), TermId(0));
        arena.push("é中".as_bytes(), TermId(1));
        arena.push("".as_bytes(), TermId(2));
        let mut taken = std::mem::take(&mut arena);
        assert_eq!(arena.text.capacity(), 0);
        assert_eq!(arena.offsets.capacity(), 0);
        assert_eq!(arena.ids.capacity(), 0);
        assert_eq!(taken.term_at(0), "".as_bytes());
        assert_eq!(taken.term_at(1), "é中".as_bytes());
        assert_eq!(taken.term_at(2), "".as_bytes());
        let capacity = taken.text.capacity();
        taken.clear();
        taken.push("中".as_bytes(), TermId(3));
        assert_eq!(taken.term_at(0), "中".as_bytes());
        assert_eq!(taken.text.capacity(), capacity);
    }

    #[test]
    fn push_and_read_back() {
        let mut arena = EntriesArena::new();
        arena.push("a".as_bytes(), TermId(0));
        arena.push("ab".as_bytes(), TermId(1));
        arena.push("abc".as_bytes(), TermId(2));
        assert_eq!(arena.len(), 3);
        assert_eq!(arena.term_at(0), "a".as_bytes());
        assert_eq!(arena.term_at(1), "ab".as_bytes());
        assert_eq!(arena.term_at(2), "abc".as_bytes());
        assert_eq!(arena.id_at(1), TermId(1));
        assert_eq!(arena.terms(0..3).collect::<Vec<_>>(), [b"a" as &[u8], b"ab", b"abc"]);
        assert_eq!(arena.ids(0..3), [TermId(0), TermId(1), TermId(2)]);
    }

    #[test]
    fn clear_resets_end_offsets() {
        let mut arena = EntriesArena::new();
        arena.push("x".as_bytes(), TermId(0));
        arena.clear();
        assert_eq!(arena.len(), 0);
        arena.push("y".as_bytes(), TermId(1));
        assert_eq!(arena.term_at(0), "y".as_bytes());
    }

    #[test]
    fn range_slices_correctly() {
        let mut arena = EntriesArena::new();
        for i in 0..5 {
            arena.push(i.to_string().as_bytes(), TermId(i));
        }
        assert_eq!(arena.terms(1..4).collect::<Vec<_>>(), [b"1" as &[u8], b"2", b"3"]);
        assert_eq!(arena.ids(1..4), [TermId(1), TermId(2), TermId(3)]);
    }

    #[test]
    fn append_without_clear_concatenates() {
        let mut a = EntriesArena::new();
        a.push("a".as_bytes(), TermId(0));
        let mut b = EntriesArena::new();
        b.push("b".as_bytes(), TermId(1));
        // Simulate decoding two blocks back-to-back into one arena.
        a.push(b.term_at(0), b.id_at(0));
        assert_eq!(a.len(), 2);
        assert_eq!(a.term_at(0), "a".as_bytes());
        assert_eq!(a.term_at(1), "b".as_bytes());
    }

}
