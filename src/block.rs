use crate::TermId;

/// A single lexical block: `terms` is sorted ascending (this is `F_b`, the
/// block's forward index -- currently a plain binary-searched array, not
/// yet a micro-FST) and `ids` is the parallel `P_b` external-id sequence.
/// See `DESIGN.md` §2.
#[derive(Debug)]
pub(crate) struct Block {
    pub(crate) terms: Vec<Box<str>>,
    pub(crate) ids: Vec<TermId>,
}

impl Block {
    pub(crate) fn new() -> Self {
        Block {
            terms: Vec::new(),
            ids: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.terms.len()
    }

    pub(crate) fn min_key(&self) -> Option<&str> {
        self.terms.first().map(|s| s.as_ref())
    }

    /// `Ok(local_rank)` if `term` is present, `Err(insertion_point)` otherwise.
    pub(crate) fn rank(&self, term: &str) -> Result<usize, usize> {
        self.terms
            .binary_search_by(|probe| probe.as_ref().cmp(term))
    }

    pub(crate) fn select(&self, local_rank: usize) -> (&str, TermId) {
        (&self.terms[local_rank], self.ids[local_rank])
    }

    pub(crate) fn insert_at(&mut self, index: usize, term: &str, id: TermId) {
        self.terms.insert(index, term.into());
        self.ids.insert(index, id);
    }

    pub(crate) fn remove_at(&mut self, index: usize) -> (Box<str>, TermId) {
        (self.terms.remove(index), self.ids.remove(index))
    }

    pub(crate) fn local_rank_of(&self, id: TermId) -> Option<usize> {
        self.ids.iter().position(|&x| x == id)
    }

    /// Splits off `[mid, len)` into a new block, leaving `[0, mid)` in
    /// `self`. Both halves stay sorted since the cut is a plain index split.
    pub(crate) fn split_off(&mut self, mid: usize) -> Block {
        Block {
            terms: self.terms.split_off(mid),
            ids: self.ids.split_off(mid),
        }
    }

    /// Appends `other` onto the end of `self`, assuming every term in
    /// `other` sorts after every term in `self` (true for lexically
    /// adjacent blocks, which partition sorted space contiguously).
    pub(crate) fn append(&mut self, mut other: Block) {
        self.terms.append(&mut other.terms);
        self.ids.append(&mut other.ids);
    }

    /// Merges a sorted batch of new entries, known to be disjoint from
    /// `self`'s current content, into `self` in one linear pass. Rebuilds
    /// the block once instead of doing `new_entries.len()` separate
    /// `Vec::insert` shifts (design doc §10 batch grouping).
    pub(crate) fn merge_sorted(&mut self, new_entries: Vec<(Box<str>, TermId)>) {
        let old_terms = std::mem::take(&mut self.terms);
        let old_ids = std::mem::take(&mut self.ids);

        let mut terms = Vec::with_capacity(old_terms.len() + new_entries.len());
        let mut ids = Vec::with_capacity(old_ids.len() + new_entries.len());

        let mut old_iter = old_terms.into_iter().zip(old_ids);
        let mut new_iter = new_entries.into_iter();
        let mut old_next = old_iter.next();
        let mut new_next = new_iter.next();

        loop {
            match (&old_next, &new_next) {
                (Some((ot, _)), Some((nt, _))) => {
                    if ot.as_ref() <= nt.as_ref() {
                        let (t, i) = old_next.take().unwrap();
                        terms.push(t);
                        ids.push(i);
                        old_next = old_iter.next();
                    } else {
                        let (t, i) = new_next.take().unwrap();
                        terms.push(t);
                        ids.push(i);
                        new_next = new_iter.next();
                    }
                }
                (Some(_), None) => {
                    let (t, i) = old_next.take().unwrap();
                    terms.push(t);
                    ids.push(i);
                    old_next = old_iter.next();
                }
                (None, Some(_)) => {
                    let (t, i) = new_next.take().unwrap();
                    terms.push(t);
                    ids.push(i);
                    new_next = new_iter.next();
                }
                (None, None) => break,
            }
        }

        self.terms = terms;
        self.ids = ids;
    }
}
