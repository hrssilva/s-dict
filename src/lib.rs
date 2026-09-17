//! `s-dict`: a dynamic, incrementally mutable bidirectional RDF term
//! `<-> TermId` dictionary.
//!
//! This crate follows the "macro-dynamic organization + micro-static
//! compression" architecture described in `DESIGN.md`: the term space is
//! partitioned into contiguous lexical blocks, each block owning a small,
//! locally rebuilt forward/reverse index, while a small global router and
//! reverse directory route lookups to the right block. Updates rebuild at
//! most one or two blocks, never the whole dictionary.
//!
//! Currently implemented: **Stage 0** ([`PlainDictionary`], a plain
//! `HashMap`/`Vec` baseline -- also the permanent "plain baseline" other
//! stages are benchmarked against) and **Stage 1** ([`Dictionary`], lexical
//! blocks with a sorted-array router and split/merge). Per-block
//! finite-state compression (the "micro-FST" of the full design) is not
//! yet implemented; see `DESIGN.md` for the roadmap.

mod block;
mod dictionary;
mod plain;

pub use dictionary::{BlockId, Dictionary};
pub use plain::PlainDictionary;

/// A stable identifier for an RDF term, independent of lexical order.
///
/// `TermId`s are assigned on insertion and remain valid for as long as the
/// term is present in the dictionary. Removing a term may cause its id to
/// be reused by a later, unrelated insertion. `u32`-width to interoperate
/// directly with `s-graph`'s `u32` subject/predicate/object id space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TermId(pub u32);

/// The bidirectional operation surface shared by every dictionary
/// implementation in this crate, so stages can be benchmarked and swapped
/// in against the same call sites.
pub trait TermDictionary {
    /// Looks up the id of `term`, if present.
    fn get_id(&self, term: &str) -> Option<TermId>;

    /// Looks up the term for `id`, if present.
    fn get_term(&self, id: TermId) -> Option<&str>;

    /// Returns `true` if `term` is present.
    fn contains(&self, term: &str) -> bool {
        self.get_id(term).is_some()
    }

    /// Inserts `term`, returning its id. Idempotent: re-inserting a term
    /// already present returns its existing id unchanged.
    fn insert(&mut self, term: &str) -> TermId;

    /// Inserts many terms at once. The default implementation just calls
    /// [`insert`](Self::insert) in a loop; implementations that can batch
    /// work by destination (e.g. grouping by block) should override this.
    fn insert_batch(&mut self, terms: &[&str]) -> Vec<TermId> {
        terms.iter().map(|t| self.insert(t)).collect()
    }

    /// Removes `term` if present, returning its former id.
    fn remove(&mut self, term: &str) -> Option<TermId>;

    /// The number of terms currently in the dictionary.
    fn len(&self) -> usize;

    /// `true` if the dictionary has no terms.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
