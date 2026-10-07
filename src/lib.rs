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

mod automaton_block;
mod automaton_dictionary;
mod batch;
mod block;
mod block_ids;
mod dictionary;
mod entries_arena;
mod front_coded_block;
mod front_coded_dictionary;
mod micro_fst;
mod packed_automaton_block;
mod packed_automaton_dictionary;
mod plain;
mod reverse_map;
mod sharded;

pub use automaton_dictionary::{AutomatonBlockId, AutomatonDictionary};
pub use batch::{PreparedBatch, TermBatch};
pub use dictionary::{BlockId, Dictionary, RebuildStats};
pub use front_coded_dictionary::{FrontCodedBlockId, FrontCodedDictionary};
pub use packed_automaton_dictionary::{PackedAutomatonBlockId, PackedAutomatonDictionary};
pub use plain::PlainDictionary;
pub use sharded::{ShardRouter, ShardedDictionary};

/// Not part of this crate's public API -- thin wrappers exposing
/// otherwise-`pub(crate)` construction primitives to `examples/
/// perf_micro_fst_build.rs`, which isolates `Automaton::build_sorted`
/// against `Block::build` specifically (DESIGN.md's "closing the
/// bulk-construction gap" analysis). `#[doc(hidden)]` since it exists
/// only for that one benchmark, not for external use.
#[doc(hidden)]
pub mod __internal_bench {
    use crate::TermId;
    use crate::block::Block;
    use crate::micro_fst::Automaton;

    pub fn build_automaton(terms: &[String]) -> usize {
        let automaton = Automaton::build_sorted(terms.iter().map(String::as_str));
        automaton.len()
    }

    pub fn build_block(terms: &[String]) -> usize {
        let entries: Vec<(&str, TermId)> = terms
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), TermId(i as u32)))
            .collect();
        Block::build(entries.into_iter()).len()
    }

    /// Reachable state count of an `Automaton` built directly from
    /// `terms` (already sorted, deduplicated) -- for `examples/
    /// perf_isolate_pattern.rs`'s block/dictionary-free comparison
    /// against `build_fst_states` (DESIGN.md's states-per-term
    /// investigation).
    pub fn build_automaton_states(terms: &[String]) -> usize {
        Automaton::build_sorted(terms.iter().map(String::as_str)).reachable_state_count()
    }

    /// `Block`'s (fst-backed) counterpart to `build_automaton_states`.
    pub fn build_fst_states(terms: &[String]) -> usize {
        let entries: Vec<(&str, TermId)> = terms
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), TermId(i as u32)))
            .collect();
        Block::build(entries.into_iter()).reachable_state_count()
    }

    /// Builds an automaton from an empty start, applying `terms` (already
    /// sorted) over `rounds` separate `Automaton::insert_sorted` calls
    /// (consecutive chunks) instead of one `build_sorted` -- for
    /// `examples/perf_isolate_pattern.rs`'s test of whether repeated
    /// incremental insertion drifts away from true minimality relative
    /// to `build_automaton_states`'s single-shot fresh build over the
    /// same final term set (DESIGN.md's states-per-term investigation).
    pub fn incremental_insert_sorted_rounds_states(terms: &[String], rounds: usize) -> usize {
        let mut automaton = Automaton::build_sorted(std::iter::empty());
        let chunk = terms.len().div_ceil(rounds.max(1));
        for round_terms in terms.chunks(chunk.max(1)) {
            automaton.insert_sorted(round_terms.iter().map(String::as_str));
        }
        automaton.reachable_state_count()
    }
}

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
