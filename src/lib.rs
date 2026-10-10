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
mod limits;
mod packed_automaton;
mod packed_automaton_block;
mod packed_automaton_dictionary;
mod range_map;
mod sharded;

pub use batch::{PreparedBatch, TermBatch};
pub use packed_automaton_dictionary::PackedAutomatonDictionary;
pub use sharded::{ShardRouter, ShardedDictionary};

/// Compiles and runs the README examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

/// A stable identifier for an RDF term, independent of lexical order.
///
/// `TermId`s are assigned on insertion and remain valid for as long as the
/// term is present in the dictionary. Removing a term may cause its id to
/// be reused by a later, unrelated insertion. `u32`-width to interoperate
/// directly with `s-graph`'s `u32` subject/predicate/object id space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TermId(pub u32);

/// Why a fallible operation (`try_*`) refused: checked before anything is
/// changed, so the dictionary is left exactly as it was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DictError {
    /// The operation needs `needed` new ids but only `available` remain
    /// below the dictionary's id limit.
    IdsExhausted { needed: u64, available: u64 },
    /// A term is `len` bytes long; at most `max` are supported.
    TermTooLarge { len: usize, max: usize },
    /// A batch would hold `len` bytes of text; at most `max` are supported.
    BatchTooLarge { len: usize, max: usize },
}

impl std::fmt::Display for DictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            DictError::IdsExhausted { needed, available } => {
                write!(f, "term ids exhausted: {needed} new ids needed, {available} available")
            }
            DictError::TermTooLarge { len, max } => write!(f, "term of {len} bytes exceeds the {max}-byte limit"),
            DictError::BatchTooLarge { len, max } => write!(f, "batch of {len} text bytes exceeds the {max}-byte limit"),
        }
    }
}

impl std::error::Error for DictError {}

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
    /// already present returns its existing id unchanged. On error nothing
    /// changes: see [`DictError`].
    fn try_insert(&mut self, term: &str) -> Result<TermId, DictError>;

    /// [`try_insert`](Self::try_insert), panicking on error.
    fn insert(&mut self, term: &str) -> TermId {
        self.try_insert(term).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Inserts many terms at once, returning ids in input order (duplicates
    /// share one). Atomic: on error nothing is inserted.
    fn try_insert_batch(&mut self, terms: &[&str]) -> Result<Vec<TermId>, DictError>;

    /// [`try_insert_batch`](Self::try_insert_batch), panicking on error.
    fn insert_batch(&mut self, terms: &[&str]) -> Vec<TermId> {
        self.try_insert_batch(terms).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Whether [`try_insert_batch`](Self::try_insert_batch) of `terms`
    /// would succeed right now, without changing anything (how
    /// [`ShardedDictionary`] keeps a batch over several shards atomic).
    fn check_insert_batch(&self, terms: &[&str]) -> Result<(), DictError>;

    /// Removes `term` if present, returning its former id.
    fn remove(&mut self, term: &str) -> Option<TermId>;

    /// The number of terms currently in the dictionary.
    fn len(&self) -> usize;

    /// From now on, ids handed out stay below `limit` (inserts that would
    /// need more report [`DictError::IdsExhausted`]). Panics if an id at
    /// or above `limit` was already handed out.
    fn set_id_limit(&mut self, limit: u32);

    /// `true` if the dictionary has no terms.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
