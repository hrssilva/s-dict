//! Size limits, and the write bounds that keep every block's records
//! addressable by `u32` (`PackedAutomaton::next_address`).
//!
//! A block's records are append-only `u32`-addressed bytes. Rather than
//! failing a mutation halfway once they outgrow 4 GiB, every mutation is
//! checked beforehand against a provable bound on the bytes it can write,
//! and takes a rebuild path (fresh builds of bounded chunks) when the
//! incremental one can't be shown to fit. The bounds:
//!
//! - A fresh build (`PackedAutomaton::build_sorted`) confirms at most one
//!   frame per input byte plus the root, and writes at most one record per
//!   frame: `2 + 11 x children` bytes (header, count escape, and per
//!   transition a byte, a LEB128 output and a LEB128 target delta), with
//!   children summing to at most the frames. So at most
//!   [`BUILD_BYTES_PER_TEXT_BYTE`] `x (text + 1)` bytes. Compaction and path
//!   compression rewrite into a fresh buffer within the same bound (one
//!   record per reachable per-byte state, chains no larger).
//! - An incremental merge or removal opens at most one existing state per
//!   input byte plus the root. Each opened state is rewritten once (at most
//!   256 transitions: `2 + 11 x 256` bytes) and may first expand one chain
//!   (at most [`MAX_CHAIN_STATES`] single-transition records of at most 2
//!   bytes, plus one target delta); new states cost as in a build. So at
//!   most [`INCREMENTAL_BYTES_PER_TEXT_BYTE`] `x (text + 1)` bytes.

/// Record addresses are below this (`u32` addresses).
#[cfg(not(test))]
pub(crate) const ADDRESS_LIMIT: u64 = u32::MAX as u64;
/// Small in unit tests, so the rebuild paths run on small inputs; the
/// writes are checked against it (`PackedAutomaton::next_address`).
#[cfg(test)]
pub(crate) const ADDRESS_LIMIT: u64 = 1 << 22;

/// Longest chain record, in states (`PackedAutomaton::chain_up` splits
/// longer runs into several chains).
#[cfg(not(test))]
pub(crate) const MAX_CHAIN_STATES: usize = 4096;
#[cfg(test)]
pub(crate) const MAX_CHAIN_STATES: usize = 16;

pub(crate) const BUILD_BYTES_PER_TEXT_BYTE: u64 = 13;

pub(crate) const INCREMENTAL_BYTES_PER_TEXT_BYTE: u64 = 2 + 11 * 256 + 2 * MAX_CHAIN_STATES as u64 + 5;

/// Most text one block may hold for compaction to stay addressable.
pub(crate) const COMPACT_TEXT_LIMIT: u64 = ADDRESS_LIMIT / BUILD_BYTES_PER_TEXT_BYTE - 1;

/// Blocks split once their text exceeds this (as well as `block_max`
/// terms), and rebuilt chunks hold at most this much text (or a single
/// term).
pub(crate) const BLOCK_TEXT_LIMIT: u64 = 1 << (ADDRESS_LIMIT.ilog2() - 4);

/// Longest term: a block holding it alone stays within
/// [`BLOCK_TEXT_LIMIT`].
pub(crate) const MAX_TERM_BYTES: usize = BLOCK_TEXT_LIMIT as usize;

/// Most text a [`crate::TermBatch`] holds (`u32` offsets).
#[cfg(not(test))]
pub(crate) const MAX_BATCH_TEXT_BYTES: usize = u32::MAX as usize;
#[cfg(test)]
pub(crate) const MAX_BATCH_TEXT_BYTES: usize = 1 << 16;

/// Whether an incremental mutation writing about `text` input bytes into
/// records currently `bytes` long, in a block holding `block_text` bytes of
/// terms afterwards, provably stays addressable (and compactable).
pub(crate) fn incremental_fits(bytes: usize, text: u64, block_text: u64) -> bool {
    block_text <= COMPACT_TEXT_LIMIT
        && (bytes as u64).saturating_add(INCREMENTAL_BYTES_PER_TEXT_BYTE.saturating_mul(text.saturating_add(1))) <= ADDRESS_LIMIT
}

const _: () = {
    assert!(BUILD_BYTES_PER_TEXT_BYTE * (BLOCK_TEXT_LIMIT + 1) <= ADDRESS_LIMIT);
    assert!(BLOCK_TEXT_LIMIT <= COMPACT_TEXT_LIMIT);
};
