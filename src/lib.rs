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
//! stages are benchmarked against), **Stages 1+3** ([`Dictionary`],
//! lexical blocks routed by a sorted-array router, split/merged on
//! overflow/underflow, each block's forward index a minimal per-block FST
//! via the `fst` crate), and the §13 "front-coded blocks" baseline
//! ([`FrontCodedDictionary`], the same block/router/split-merge structure
//! with shared-prefix delta coding instead of an FST, to isolate what
//! automaton minimization specifically buys over blocking alone). See
//! `DESIGN.md` for the roadmap and what's still not implemented.

mod block;
mod dictionary;
mod entries_arena;
mod front_coded_block;
mod front_coded_dictionary;
mod plain;

pub use dictionary::{BlockId, Dictionary};
pub use front_coded_dictionary::{FrontCodedBlockId, FrontCodedDictionary};
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

    /// Looks up the term for `id`, if present. Owned rather than borrowed:
    /// an FST-backed block (see [`Dictionary`]) can only decode a term by
    /// walking its automaton into a scratch buffer, which can't outlive
    /// the call, so the trait returns `String` uniformly rather than
    /// letting only some implementations borrow.
    fn get_term(&self, id: TermId) -> Option<String>;

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
