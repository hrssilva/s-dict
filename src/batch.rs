//! Owned, reusable batches whose ordering can be prepared off the writer thread.
use std::ops::Range;

#[derive(Debug, Default)]
pub(crate) struct BatchOrder {
    order: Vec<usize>,
    starts: Vec<usize>,
}

impl BatchOrder {
    pub(crate) fn prepare<'a>(&mut self, len: usize, term: impl Fn(usize) -> &'a str) {
        self.order.clear();
        self.order.extend(0..len);
        self.order.sort_unstable_by(|&a, &b| term(a).cmp(term(b)));
        self.starts.clear();
        for i in 0..len {
            if i == 0 || term(self.order[i - 1]) != term(self.order[i]) {
                self.starts.push(i);
            }
        }
        self.starts.push(len);
    }

    pub(crate) fn len(&self) -> usize {
        self.order.len()
    }

    pub(crate) fn unique_len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    pub(crate) fn first(&self, group: usize) -> usize {
        self.order[self.starts[group]]
    }

    pub(crate) fn positions(&self, group: usize) -> &[usize] {
        &self.order[self.starts[group]..self.starts[group + 1]]
    }
}

/// Owned term text and flat ordering metadata for one caller-bounded batch.
/// `clear` retains capacity. Preparation never touches a dictionary and can
/// run on parser workers; moving the batch does not invalidate any references.
#[derive(Debug, Default)]
pub struct TermBatch {
    text: String,
    terms: Vec<Range<usize>>,
    pub(crate) order: BatchOrder,
    prepared: bool,
}

impl TermBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, term: &str) {
        let start = self.text.len();
        self.text.push_str(term);
        self.terms.push(start..self.text.len());
        self.prepared = false;
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.terms.clear();
        self.order.order.clear();
        self.order.starts.clear();
        self.prepared = false;
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn text_bytes(&self) -> usize {
        self.text.len()
    }

    pub(crate) fn term(&self, index: usize) -> &str {
        &self.text[self.terms[index].clone()]
    }

    /// Sort and group duplicate terms, retaining original input positions.
    pub fn prepare(&mut self) {
        if self.prepared {
            return;
        }
        let text = &self.text;
        let terms = &self.terms;
        self.order.prepare(terms.len(), |i| &text[terms[i].clone()]);
        self.prepared = true;
    }

    /// A read-only view that guarantees sorted metadata matches the term text.
    /// Returns `None` after any push/clear until `prepare` is called again.
    pub fn prepared(&self) -> Option<PreparedBatch<'_>> {
        self.prepared.then_some(PreparedBatch { batch: self })
    }
}

/// Validated by `TermBatch::prepare`; borrows and prevents mutation of its batch.
#[derive(Clone, Copy, Debug)]
pub struct PreparedBatch<'a> {
    pub(crate) batch: &'a TermBatch,
}
