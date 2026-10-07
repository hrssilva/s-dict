//! `s-dict`: a dynamic, incrementally mutable bidirectional RDF term
//! `<-> TermId` dictionary.
//!
//! [`PackedAutomatonDictionary`] partitions the term space into contiguous
//! lexical blocks. Each block is a byte-packed minimal acyclic automaton
//! over its sorted terms (term <-> local rank in one descent) beside a
//! compressed `rank <-> TermId` permutation; a sorted directory routes terms
//! to blocks and a range map routes ids to blocks. Blocks split and merge
//! as they grow and shrink, so an update touches one or two blocks. Batches
//! can be sorted and grouped off the writer thread ([`TermBatch`]), and
//! [`ShardedDictionary`] runs one writer per key range. See `DESIGN.md` for
//! the design and every measurement behind it.

mod batch;
mod block_ids;
mod entries_arena;
mod fsst;
mod packed_automaton;
mod packed_automaton_block;
mod packed_automaton_dictionary;
mod range_map;
mod sharded;

pub use batch::{PreparedBatch, TermBatch};
pub use packed_automaton_dictionary::{PackedAutomatonBlockId, PackedAutomatonDictionary};
pub use sharded::{ShardRouter, ShardedDictionary};

/// A stable identifier for an RDF term, independent of lexical order.
///
/// `TermId`s are assigned on insertion and remain valid for as long as the
/// term is present in the dictionary. Removing a term may cause its id to
/// be reused by a later, unrelated insertion. `u32`-width to interoperate
/// directly with `s-graph`'s `u32` subject/predicate/object id space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TermId(pub u32);

/// The bidirectional operation surface of the dictionaries in this crate
/// (a [`PackedAutomatonDictionary`], or a [`ShardedDictionary`] of them).
pub trait TermDictionary {
    /// Looks up the id of `term`, if present.
    fn get_id(&self, term: &str) -> Option<TermId>;

    /// Looks up the term for `id`, if present. Owned rather than borrowed:
    /// a block decodes a term by walking its automaton into a buffer,
    /// which can't outlive the call.
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
