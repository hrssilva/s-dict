//! `PackedAutomaton`: each block's term index -- a minimal acyclic
//! automaton over the block's sorted terms, stored as one append-only byte
//! buffer of fst-style records, with each state's outputs the ranks of its
//! terms (so rank and unrank are one descent). Built bottom-up from sorted
//! input (Daciuk, Mihov, Watson, Watson 2000), mutated in place by
//! re-confirming only the paths it touches, compacted to the exact minimal
//! automaton, and path-compressed with FSST-coded chain labels at the end
//! of a load. See the `PackedAutomaton` doc comment for the record layout,
//! and DESIGN.md for every measurement behind it.

use std::hash::Hasher;
use std::num::NonZeroU32;

/// A fast, non-cryptographic hasher for `register` -- std's default
/// `HashMap` hasher (`SipHash`) is deliberately slow, traded for DoS
/// resistance against adversarial keys, which doesn't apply here: this
/// map is purely internal bookkeeping, its keys are `Signature`s this
/// module derives itself, never attacker-influenced. A `perf record`
/// profile of `build_sorted` (DESIGN.md's "closing the bulk-construction
/// gap" section) found `SipHash` hashing plus the `Vec` equality
/// comparison (`memcmp`) it triggers on every register lookup was,
/// combined, the single largest cost in bulk construction -- this is the
/// same simple multiply-xor-rotate hash `rustc`'s own compiler internals
/// use for exactly this class of problem (fast, well-distributed enough
/// for non-adversarial keys, nothing fancier needed). Not used anywhere
/// else in this crate: nothing else here is exposed to attacker-
/// controlled hash keys in a way that would make this an appropriate
/// default more broadly.
#[derive(Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let (chunk, rest) = bytes.split_at(8);
            self.add(u64::from_ne_bytes(chunk.try_into().unwrap()));
            bytes = rest;
        }
        if bytes.len() >= 4 {
            let (chunk, rest) = bytes.split_at(4);
            self.add(u32::from_ne_bytes(chunk.try_into().unwrap()) as u64);
            bytes = rest;
        }
        if bytes.len() >= 2 {
            let (chunk, rest) = bytes.split_at(2);
            self.add(u16::from_ne_bytes(chunk.try_into().unwrap()) as u64);
            bytes = rest;
        }
        if let Some(&b) = bytes.first() {
            self.add(b as u64);
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }

    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

/// `PackedAutomaton`'s construction-time dedup structure: a **bounded,
/// lossy** 2-way set-associative LRU cache, `fst`'s own `Registry`
/// (`vendor/fst/src/raw/registry.rs`) over `u32` record addresses. It
/// replaced an unbounded exact `HashMap` that grew without bound under
/// sustained inserts (DESIGN.md's "replacing PackedAutomaton's register
/// with a bounded LRU cache"); like `fst`, it accepts some lost sharing
/// (eviction) in exchange for fixed memory.
///
/// A cell holds only a 32-bit tag of the signature hash and the record's
/// address: a tag hit is confirmed against the record already written at
/// that address (`PackedAutomaton::record_matches`, records are
/// append-only), the way `compact`'s `DedupTable` works. Storing a copy of
/// each signature (`fst`'s approach, and this type's previous one) cost a
/// heap `Vec` per cell and a dependent cache miss per probe. Buckets are a
/// power of two, so the bucket index is a mask, not a division (a 64-bit
/// `div` was a quarter of `confirm_frame`'s time).
#[derive(Clone, Debug, Default)]
struct PackedRegistry {
    /// At least `mask + 1` buckets (only that prefix is in use), most
    /// recently used cell first within a bucket.
    buckets: Vec<[RegistryCell; REGISTRY_WAYS]>,
    /// `buckets - 1`.
    mask: usize,
}

/// Cells per registry bucket (`fst`'s `mru_size` default).
const REGISTRY_WAYS: usize = 2;

#[derive(Clone, Copy, Debug, Default)]
struct RegistryCell {
    /// High half of the signature hash (the low bits pick the bucket).
    tag: u32,
    /// The record's address + 1 (`None`: empty cell; a zeroed cell is
    /// empty).
    target: Option<NonZeroU32>,
}

impl PackedRegistry {
    /// Empties the registry and sizes it for an operation confirming about
    /// `terms` terms: `Block::build`'s `fst` sizing (`4 x terms` rows, at
    /// least 64), rounded up to a power of two, at most 16,384 rows (256
    /// KiB). A recreated registry can only dedup records written during
    /// the operation, so sizing it to the whole automaton just zeroes cells
    /// nothing will use (DESIGN.md's "incremental single-term mutation").
    /// Reuses the allocation; only grows.
    fn reset(&mut self, terms: usize) {
        let buckets = terms.saturating_mul(4).clamp(64, 1 << 14).next_power_of_two();
        if self.buckets.len() < buckets {
            self.buckets = vec![Default::default(); buckets];
        } else {
            self.buckets[..buckets].fill(Default::default());
        }
        self.mask = buckets - 1;
    }

    #[inline]
    fn bucket(&mut self, hash: u64) -> &mut [RegistryCell; REGISTRY_WAYS] {
        &mut self.buckets[hash as usize & self.mask]
    }

    /// Addresses in `hash`'s bucket whose tag matches, most recent first.
    #[inline]
    fn candidates(&mut self, hash: u64) -> [Option<u32>; REGISTRY_WAYS] {
        let tag = (hash >> 32) as u32;
        self.bucket(hash)
            .map(|c| c.target.filter(|_| c.tag == tag).map(|t| t.get() - 1))
    }

    /// Marks way `way` of `hash`'s bucket most recently used.
    #[inline]
    fn promote(&mut self, hash: u64, way: usize) {
        self.bucket(hash)[..=way].rotate_right(1);
    }

    /// Inserts as most recently used, evicting the least recently used.
    #[inline]
    fn insert(&mut self, hash: u64, target: u32) {
        let bucket = self.bucket(hash);
        bucket.rotate_right(1);
        bucket[0] = RegistryCell {
            tag: (hash >> 32) as u32,
            target: Some(NonZeroU32::MIN.saturating_add(target)),
        };
    }

    /// Occupied cells -- always `<= cells.len()`. See
    /// `PackedAutomaton::register_len`.
    #[cfg(test)]
    fn len(&self) -> usize {
        let used = (self.mask + 1).min(self.buckets.len());
        self.buckets[..used].iter().flatten().filter(|c| c.target.is_some()).count()
    }
}


/// One state being incrementally (re)built by [`PackedAutomaton::insert`],
/// before it's confirmed and written -- the packed-representation
/// equivalent of the plain `State`/`Transitions` a matched-but-unconfirmed
/// state already was in the mutable slab design; kept in memory only for
/// the duration of one `insert` call, never itself part of the
/// persistent structure.
/// A record's transitions as `(byte, output, target)`.
type Edges = Vec<(u8, u32, u32)>;

#[derive(Debug)]
struct StagingFrame {
    is_final: bool,
    /// `(byte, child_subtree_count, target)`, kept sorted by byte
    /// throughout. Records don't store subtree counts; they're carried
    /// here instead, derived top-down (`PackedAutomaton::open_frame`:
    /// from the parent's count and its transitions' outputs) or known at
    /// confirmation (`confirm_frame` returns it). `confirm_frame` turns
    /// them into outputs -- the prefix-sum-of-sibling-subtree-counts
    /// technique `finalize_outputs_and_count` uses.
    children: Vec<(u8, u32, u32)>,
}

/// A byte-packed, incrementally mutable [`Automaton`] -- **not**
/// unrolled blocks. That was this type's first mutable design
/// (`blocks: Vec<Vec<u8>>`, prompted by `../s-graph`'s
/// `radixtree_dfs_interleaved_varskip_unrolled.rs`, specifically to
/// bound single-insert cost the way that file's own block-relative
/// addressing does for a flat byte stream), and it was a real, measured
/// mistake: `s-graph`'s structure is a plain tree, where DFS-preorder
/// position *is* a valid address, so a block boundary can force a
/// shift-or-split; this crate's automaton is a DAG (register-based
/// minimization means a state can have several parents) addressed by
/// explicit, permanent `u32`s instead -- nothing here is ever
/// positionally addressed, so nothing ever needs to shift, and unrolled
/// blocks were buying a property (bounded-cost single-record insertion
/// without touching the rest of the buffer) this design already had for
/// free by construction. What they cost instead: every field read went
/// through `blocks[block_index]` -- a second, bounds-checked pointer
/// hop in front of the actual data -- on top of what a single `Vec<u8>`
/// needs. Isolating that cost directly (a throwaway flat-buffer variant
/// with the same record format) found it was the dominant cause of a
/// measured decode regression, not the `subtree_count` field; `bytes:
/// Vec<u8>` below is that correction -- ordinary amortized-growth `Vec`
/// reallocation in place of block-capping, one pointer hop per field
/// read instead of two.
///
/// A second, separate methodology correction followed: `perf stat`
/// wraps the whole process, so every "decode" comparison against this
/// type had silently also been counting `compile`'s own cost (real,
/// and for a low-redundancy corpus not small -- see below) as if it
/// were decode cost, something `Automaton`'s probes never pay an
/// equivalent of. Once isolated (subtracting a matching "terms +
/// `build_sorted`, + `compile` for the packed side" setup-only
/// baseline from each full probe), the *true* decode-only picture is
/// corpus-dependent: on the low-redundancy (hashlike) corpus, decode is
/// roughly at parity with, possibly slightly faster than, the plain
/// slab `Automaton`; on the redundant (clustered) corpus, where
/// `compile` is cheap to begin with (little state to build) so this
/// correction barely moves the number, decode is still genuinely
/// slower (~1.2x more cycles). See DESIGN.md's methodology-correction
/// section for the numbers this is based on.
///
/// Confirmation is still always bottom-up (a child is written and given
/// its permanent address *before* its parent's record references it --
/// see [`PackedAutomaton::compile`]/[`PackedAutomaton::insert`]), so a
/// confirmed record's address still never needs to change after the
/// fact, and there's still no reference-counting or
/// redirect-into-an-already-written-record machinery: a state
/// superseded by an equivalent, already-registered one (or made
/// unreachable because its parent now points elsewhere) is simply dead,
/// unreachable bytes -- never explicitly reclaimed by this prototype.
/// That dead weight is a real, separately-measured cost under sustained
/// inserts (20-37x a fresh rebuild's size within 128-200 inserts, per
/// `measure_packed_automaton_dead_bytes_under_sustained_inserts`),
/// **unaffected by this flat-buffer change** -- every ancestor on an
/// insert path is rewritten (its outputs or its child's address change),
/// which no addressing scheme removes. Dead records are reclaimed by
/// [`PackedAutomaton::compact`], which callers trigger (see
/// `PackedAutomatonBlock::merge_into` and DESIGN.md's "copy-compaction").
///
/// Record layout (fst-style, DESIGN.md's "fst-style record encoding" and
/// "label folding"): a record's address is its *last* byte, the header,
/// and it is parsed downward from there (`fst`'s own convention), so every
/// field can be written while appending in postorder. Two header forms:
///
/// - `1 N cccccc` -- one transition, not final (91% of states on real
///   RDF). `cccccc` is the label's code in [`RDF_LABELS`] (0: the label
///   byte is stored below the header). N (NEXT): the target is the record
///   immediately below; otherwise a target delta follows.
/// - `0 F nnnnnn` -- everything else. F is `is_final`; `nnnnnn` the
///   transition count (63 = escape: one more byte below holds
///   `count - 63`). Then, per transition in byte order, reading downward:
///   `[byte: u8]`, `[output: reversed LEB128]` (omitted for the first
///   transition: always `is_final`), `[target delta: reversed LEB128]`.
///
/// Target deltas are `field_start - target`. No subtree counts are stored:
/// `term_count` holds the root's, and every other state's is derived
/// top-down while descending (`open_frame`).
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct PackedAutomaton {
    bytes: crate::records::Records,
    root: u32,
    /// The root's subtree count (= number of terms). The only count
    /// stored anywhere: every other state's count is derived top-down
    /// while descending (see `StagingFrame::children`).
    term_count: u32,
    /// Chain records may exist (path compression, see `append_chain`).
    /// Reads handle them anywhere; mutation expands just the chains on its
    /// path (`open_frame` -> `expand_chain`).
    has_chains: bool,
    /// Exactly as `compact_and_chain` left it: no mutation since.
    pristine: bool,
    /// Symbol table for chain labels (`crate::fsst`), shared by every
    /// block of a dictionary; set by the dictionary once trained.
    symbols: Option<std::sync::Arc<crate::fsst::SymbolTable>>,
    /// The current chains were written with `symbols` available.
    chains_coded: bool,
}

/// `PackedAutomaton::chain_step`'s outcome.
enum ChainStep {
    /// The label matched `len` query bytes; continue at the target.
    Through(usize, u32),
    /// The query sorts below every term through the chain.
    Below,
    /// The query sorts above every term through the chain.
    Above,
}

/// A non-empty chain record: label bytes at `start..start + len` (coded
/// with the automaton's symbol table if `coded`), then `target`.
#[derive(Clone, Copy)]
struct Chain {
    start: usize,
    len: usize,
    target: u32,
    coded: bool,
}

/// Reusable construction state for [`PackedAutomaton`]: the dedup
/// registry and every scratch buffer a build, mutation or compaction
/// needs. Any number of automata can share one (each operation resets the
/// registry for the automaton it works on), so a dictionary keeps a single
/// workspace: nothing construction-only stays resident per block, and
/// nothing is reallocated per mutation (DESIGN.md's "one construction
/// workspace").
#[derive(Debug, Default)]
pub(crate) struct Workspace {
    confirm: Confirmer,
    /// The rightmost path being built: one frame per state, root first.
    path: Vec<StagingFrame>,
    /// `path_bytes[i]` is the byte reaching `path[i + 1]`.
    path_bytes: Vec<u8>,
    last_word: Vec<u8>,
    /// `compact`'s old address -> new address map.
    moved: Vec<u32>,
    dedup: DedupTable,
}

/// The part of [`Workspace`] that confirming and opening frames needs, kept
/// separate so it can be borrowed alongside the path.
#[derive(Debug, Default)]
struct Confirmer {
    registry: PackedRegistry,
    /// Child lists for frames, reused instead of allocated per frame.
    frame_pool: Vec<Vec<(u8, u32, u32)>>,
    /// The record being confirmed, with outputs instead of child counts.
    children: Vec<(u8, u32, u32)>,
    /// A chain label being expanded.
    label: Vec<u8>,
}

impl Confirmer {
    /// An empty frame whose child list comes from the pool.
    #[inline]
    fn new_frame(&mut self) -> StagingFrame {
        let mut children = self.frame_pool.pop().unwrap_or_default();
        children.clear();
        StagingFrame {
            is_final: false,
            children,
        }
    }

    /// Returns a frame's child list to the pool (up to `RETAINED_FRAMES`).
    #[inline]
    fn recycle(&mut self, frame: StagingFrame) {
        if self.frame_pool.len() < RETAINED_FRAMES {
            self.frame_pool.push(frame.children);
        }
    }
}

/// Pooled frames kept between operations: a path holds one frame per term
/// byte, so the pool would otherwise grow to the longest term ever seen.
const RETAINED_FRAMES: usize = 4096;

/// Entries a workspace buffer keeps across operations; anything larger
/// (a very long term's path, a transiently oversized block's compaction
/// tables) is released once the operation ends.
const RETAINED_ENTRIES: usize = 1 << 16;

fn trim<T>(v: &mut Vec<T>) {
    if v.capacity() > RETAINED_ENTRIES {
        v.clear();
        v.shrink_to(RETAINED_ENTRIES);
    }
}

impl Workspace {
    /// Releases buffer capacity beyond what typical operations need.
    fn trim(&mut self) {
        trim(&mut self.path);
        trim(&mut self.path_bytes);
        trim(&mut self.last_word);
        trim(&mut self.confirm.label);
        trim(&mut self.moved);
        trim(&mut self.dedup.slots);
        trim(&mut self.dedup.spare);
    }
}

impl PackedAutomaton {
    fn empty() -> PackedAutomaton {
        PackedAutomaton {
            bytes: crate::records::Records::default(),
            root: 0,
            term_count: 0,
            has_chains: false,
            pristine: false,
            symbols: None,
            chains_coded: false,
        }
    }

    /// Moves the records out into a read-only automaton (same root, chain
    /// state and symbols), leaving `bytes` empty for a rewrite pass.
    fn take_records(&mut self) -> PackedAutomaton {
        PackedAutomaton {
            bytes: std::mem::take(&mut self.bytes),
            symbols: self.symbols.clone(),
            chains_coded: false,
            pristine: false,
            ..*self
        }
    }

    /// Builds the automaton of an already sorted, deduplicated sequence of
    /// terms (Daciuk, Mihov, Watson, Watson 2000, sorted input): the
    /// rightmost path is a stack of unconfirmed frames, and everything
    /// deeper than the next term's common prefix is confirmed bottom-up
    /// (written, or deduplicated through the registry) before the path is
    /// extended. Shares its loop with `insert_sorted` (`merge_sorted`),
    /// starting from an empty root instead of an opened one.
    pub(crate) fn build_sorted<'a>(
        terms: impl ExactSizeIterator<Item = &'a [u8]>,
        ws: &mut Workspace,
    ) -> PackedAutomaton {
        let mut automaton = PackedAutomaton::empty();
        let root = ws.confirm.new_frame();
        automaton.merge_sorted(root, terms, ws);
        automaton
    }

    /// Inserts an already sorted, deduplicated sequence of terms (already
    /// present ones are no-ops). The rightmost path persists across the
    /// whole sequence, so a prefix shared by consecutive terms is opened
    /// and re-confirmed once per call, not once per term (DESIGN.md's "the
    /// real cause"). Frames are private copies of the records they open
    /// (`open_frame`), and records are append-only, so nothing needs
    /// clone-if-shared bookkeeping: superseded records are just dead bytes
    /// until the next `compact`.
    pub(crate) fn insert_sorted<'a>(&mut self, terms: impl ExactSizeIterator<Item = &'a [u8]>, ws: &mut Workspace) {
        self.pristine = false;
        let root = self.open_frame(&mut ws.confirm, self.root, self.term_count);
        self.merge_sorted(root, terms, ws);
    }

    /// Inserts `term` if not already present: `insert_sorted` of one term.
    #[cfg(test)]
    pub(crate) fn insert(&mut self, term: &str, ws: &mut Workspace) {
        self.insert_sorted(std::iter::once(term.as_bytes()), ws);
    }

    /// The sorted-merge loop behind `build_sorted` and `insert_sorted`, from
    /// an already opened `root` frame. For each term: confirm the path
    /// below its common prefix with the previous term, follow existing
    /// transitions as far as they match (opening each state reached), then
    /// extend with new frames for the rest.
    fn merge_sorted<'a>(&mut self, root: StagingFrame, terms: impl ExactSizeIterator<Item = &'a [u8]>, ws: &mut Workspace) {
        let Workspace {
            confirm,
            path,
            path_bytes,
            last_word,
            ..
        } = ws;
        confirm.registry.reset(terms.len());
        path.clear();
        path_bytes.clear();
        last_word.clear();
        path.push(root);
        for (i, word) in terms.enumerate() {
            debug_assert!(i == 0 || last_word.as_slice() < word, "terms must be sorted and distinct");
            let common = common_prefix_len(last_word, word);
            self.confirm_down_to(confirm, path, path_bytes, common + 1);

            // The frame at the end of the path reflects this call's current
            // view (freshly opened, or updated by an earlier confirmation).
            let mut matched = common;
            while let Some(&byte) = word.get(matched) {
                let cur = path.last().expect("the path always holds the root");
                let Ok(idx) = cur.children.binary_search_by_key(&byte, |&(b, _, _)| b) else {
                    break;
                };
                let (_, child_count, target) = cur.children[idx];
                let frame = self.open_frame(confirm, target, child_count);
                path.push(frame);
                path_bytes.push(byte);
                matched += 1;
            }
            for &b in &word[matched..] {
                path.push(confirm.new_frame());
                path_bytes.push(b);
            }
            path.last_mut().expect("the path always holds the root").is_final = true;

            last_word.clear();
            last_word.extend_from_slice(word);
        }
        self.confirm_down_to(confirm, path, path_bytes, 0);
        ws.trim();
    }

    /// Pops and confirms every frame deeper than `keep_from` (down to
    /// `keep_from.max(1)` frames), splicing each one's `(byte, count,
    /// address)` into its parent; `keep_from == 0` also confirms the root
    /// and makes it this automaton's root. The splice is a sorted insert
    /// or an in-place update: an opened frame may already hold the byte
    /// (an existing transition whose target changed).
    fn confirm_down_to(&mut self, confirm: &mut Confirmer, path: &mut Vec<StagingFrame>, path_bytes: &mut Vec<u8>, keep_from: usize) {
        while path.len() > keep_from.max(1) {
            let frame = path.pop().expect("loop guard");
            let byte = path_bytes.pop().expect("one byte per non-root frame");
            let (pos, count) = self.confirm_frame(confirm, &frame);
            confirm.recycle(frame);
            let parent = path.last_mut().expect("the path always holds the root");
            match parent.children.binary_search_by_key(&byte, |&(b, _, _)| b) {
                Ok(idx) => parent.children[idx] = (byte, count, pos),
                Err(idx) => parent.children.insert(idx, (byte, count, pos)),
            }
        }
        if keep_from == 0 {
            let root = path.pop().expect("the path always holds the root");
            (self.root, self.term_count) = self.confirm_frame(confirm, &root);
            confirm.recycle(root);
        }
    }

    /// Removes `term` if present, returning its former rank: open the
    /// path's frames (counts derived top-down), clear the final flag, prune
    /// states left with no terms, and re-confirm the path bottom-up.
    /// Superseded records become dead bytes, reclaimed by [`Self::compact`].
    pub(crate) fn remove(&mut self, term: &str, ws: &mut Workspace) -> Option<u32> {
        let (rank, found) = self.lower_bound(term.as_bytes());
        if !found {
            return None;
        }
        self.pristine = false;
        let Workspace { confirm, path, .. } = ws;
        confirm.registry.reset(1);
        path.clear();
        let root = self.open_frame(confirm, self.root, self.term_count);
        path.push(root);
        for &byte in term.as_bytes() {
            let cur = path.last().expect("the path always holds the root");
            let idx = cur
                .children
                .binary_search_by_key(&byte, |&(b, _, _)| b)
                .expect("lower_bound found the term");
            let (_, child_count, target) = cur.children[idx];
            let frame = self.open_frame(confirm, target, child_count);
            path.push(frame);
        }
        path.last_mut().expect("the path always holds the root").is_final = false;

        for &byte in term.as_bytes().iter().rev() {
            let frame = path.pop().expect("one frame per byte plus the root");
            let resolved = (frame.is_final || !frame.children.is_empty()).then(|| self.confirm_frame(confirm, &frame));
            confirm.recycle(frame);
            let parent = path.last_mut().expect("the root is never popped here");
            let idx = parent
                .children
                .binary_search_by_key(&byte, |&(b, _, _)| b)
                .expect("descended through this transition");
            match resolved {
                Some((pos, count)) => parent.children[idx] = (byte, count, pos),
                None => {
                    parent.children.remove(idx);
                }
            }
        }
        let root = path.pop().expect("the path always holds the root");
        (self.root, self.term_count) = self.confirm_frame(confirm, &root);
        confirm.recycle(root);
        ws.trim();
        Some(rank)
    }

    /// `(number of terms sorting before term, whether term is present)` --
    /// one descent, with each state's subtree count derived top-down from
    /// `term_count` and the outputs, as in `open_frame`.
    pub(crate) fn lower_bound(&self, term: &[u8]) -> (u32, bool) {
        // Chains are rare (only after `compact_and_chain`), so they get
        // their own loop: handling them inline made the compiler spill
        // `value`/`count` in this hot loop (+60% of its time during loads).
        if self.has_chains {
            return self.lower_bound_chained(term);
        }
        let (mut pos, mut count, mut value) = (self.root, self.term_count, 0u32);
        for &byte in term {
            match self.step(pos, count, byte) {
                Ok((out, child_count, target)) => {
                    value += out;
                    count = child_count;
                    pos = target;
                }
                Err(skip) => return (value + skip, false),
            }
        }
        (value, self.is_final(pos))
    }

    /// `lower_bound` of each of `terms` (sorted ascending), in order, to
    /// `out`. The descent stack (`stack`: `(record, rank offset, count)`
    /// per byte consumed) is kept between consecutive terms and resumed at
    /// their common prefix, so a prefix the terms share is walked once per
    /// call -- the shape of a merge's run of new terms into one block.
    pub(crate) fn lower_bounds_sorted<'a>(
        &self,
        terms: impl Iterator<Item = &'a [u8]>,
        stack: &mut Vec<(u32, u32, u32)>,
        mut out: impl FnMut(u32, bool),
    ) {
        if self.has_chains {
            for term in terms {
                let (rank, found) = self.lower_bound(term);
                out(rank, found);
            }
            return;
        }
        stack.clear();
        stack.push((self.root, 0, self.term_count));
        let mut prev: &[u8] = &[];
        for word in terms {
            debug_assert!(prev <= word, "terms must be sorted");
            // A walk that stopped short of the common prefix resumes where
            // it stopped and stops there again, as it should.
            stack.truncate(common_prefix_len(prev, word) + 1);
            let (rank, found) = loop {
                let &(pos, value, count) = stack.last().expect("the stack always holds the root");
                let Some(&byte) = word.get(stack.len() - 1) else {
                    break (value, self.is_final(pos));
                };
                match self.step(pos, count, byte) {
                    Ok((offset, child_count, target)) => stack.push((target, value + offset, child_count)),
                    Err(skip) => break (value + skip, false),
                }
            };
            out(rank, found);
            prev = word;
        }
    }

    /// One `lower_bound` step at the record `pos` (subtree count `count`)
    /// on `byte`: `Ok((rank offset, child count, target))` through the
    /// matching transition, or `Err(terms sorting before byte here)`.
    #[inline(always)]
    fn step(&self, pos: u32, count: u32, byte: u8) -> Result<(u32, u32, u32), u32> {
        let (is_final, n, mode, mut cursor) = self.header(pos);
        // (output, target) of the matching transition; the output of the
        // first transition after it ends its subtree.
        let mut matched: Option<(u32, u32)> = None;
        for i in 0..n {
            let (b, out, target, after) = self.read_transition_at(cursor, i == 0, is_final, mode);
            cursor = after;
            if let Some((m_out, m_target)) = matched {
                return Ok((m_out, out - m_out, m_target));
            }
            if b == byte {
                matched = Some((out, target));
            } else if b > byte {
                return Err(out);
            }
        }
        match matched {
            Some((out, target)) => Ok((out, count - out, target)),
            None => Err(count),
        }
    }

    /// A chain step against `query`, out of line so the per-byte loops
    /// that call it stay tight (see `lower_bound`): `None` if `pos` is not
    /// a non-empty chain; `Through` if the whole label prefixes `query`;
    /// otherwise whether `query` sorts below or above every term through
    /// the chain.
    #[cold]
    #[inline(never)]
    fn chain_step(&self, pos: u32, query: &[u8]) -> Option<ChainStep> {
        let chain = self.chain_at(pos)?;
        Some(match self.chain_match(chain, query) {
            Ok(len) => ChainStep::Through(len, chain.target),
            Err((m, Some(label_byte))) if query[m] > label_byte => ChainStep::Above,
            Err(_) => ChainStep::Below,
        })
    }

    /// Appends a chain's label to `buf` and returns its target, or `None`
    /// if `pos` is not a non-empty chain. Out of line, as `chain_step`.
    #[cold]
    #[inline(never)]
    fn push_chain_label(&self, pos: u32, buf: &mut Vec<u8>) -> Option<u32> {
        let chain = self.chain_at(pos)?;
        self.chain_label(chain, buf);
        Some(chain.target)
    }

    #[inline(never)]
    fn lower_bound_chained(&self, t: &[u8]) -> (u32, bool) {
        let mut pos = self.root;
        let mut count = self.term_count;
        let mut value = 0u32;
        let mut at = 0usize;
        loop {
            if self.bytes[pos as usize] == 0 {
                // A chain's states are non-final with one child each: the
                // whole label either matches or decides the comparison.
                match self.chain_step(pos, &t[at..]) {
                    Some(ChainStep::Through(len, target)) => {
                        at += len;
                        pos = target;
                        continue;
                    }
                    Some(ChainStep::Above) => return (value + count, false),
                    Some(ChainStep::Below) => return (value, false),
                    None => {}
                }
            }
            let Some(&byte) = t.get(at) else {
                return (value, self.is_final(pos));
            };
            let (is_final, n, next, mut cursor) = self.header(pos);
            // (output, target) of the matching transition, and the output
            // of the first transition after it (its subtree's end).
            let mut matched: Option<(u32, u32)> = None;
            let mut end = count;
            for i in 0..n {
                let (b, out, target, after) = self.read_transition_at(cursor, i == 0, is_final, next);
                cursor = after;
                if let Some((_, _)) = matched {
                    end = out;
                    break;
                }
                if b == byte {
                    matched = Some((out, target));
                } else if b > byte {
                    return (value + out, false);
                }
            }
            let Some((out, target)) = matched else {
                return (value + count, false);
            };
            value += out;
            count = end - out;
            pos = target;
            at += 1;
        }
    }

    /// Computes `frame`'s outputs from its children's subtree counts (each
    /// output is the sum of the earlier siblings' counts, plus one if the
    /// state is final) and writes, or reuses, the record. Returns
    /// `(address, subtree count)`.
    fn confirm_frame(&mut self, confirm: &mut Confirmer, frame: &StagingFrame) -> (u32, u32) {
        let mut running = u32::from(frame.is_final);
        confirm.children.clear();
        for &(b, child_count, target) in &frame.children {
            confirm.children.push((b, running, target));
            running += child_count;
        }
        let pos = self.write_confirmed_state(&mut confirm.registry, frame.is_final, &confirm.children);
        (pos, running)
    }

    /// Opens the record at `pos` (whose subtree count is `count`) into an
    /// editable frame. Child counts are the differences between
    /// consecutive outputs; the last child gets the rest of `count`.
    fn open_frame(&mut self, confirm: &mut Confirmer, pos: u32, count: u32) -> StagingFrame {
        // A mutation path entering a chain expands just that chain.
        let pos = match self.chain_at(pos) {
            Some(chain) => self.expand_chain(&mut confirm.label, chain),
            None => pos,
        };
        let mut frame = confirm.new_frame();
        let (is_final, n, mode, mut cursor) = self.header(pos);
        frame.is_final = is_final;
        let children = &mut frame.children;
        for i in 0..n {
            let (b, out, target, after) = self.read_transition_at(cursor, i == 0, is_final, mode);
            cursor = after;
            children.push((b, out, target));
        }
        // Outputs -> child counts, ascending, so `children[i + 1].1` is
        // still that child's original output when read.
        for i in 0..n {
            let end = if i + 1 < n { children[i + 1].1 } else { count };
            children[i].1 = end - children[i].1;
        }
        frame
    }

    /// Writes `chain`'s states as single-transition records (last state
    /// first, so each points at the record just below it: NEXT) and
    /// returns the first one's address. The chain record itself becomes
    /// dead bytes, reclaimed by the next compaction. O(chain length): the
    /// per-chain alternative to expanding the whole block (`unchain`).
    fn expand_chain(&mut self, label: &mut Vec<u8>, chain: Chain) -> u32 {
        label.clear();
        self.chain_label(chain, label);
        let mut next = chain.target;
        for &b in label.iter().rev() {
            next = self.append_record(false, &[(b, 0, next)]);
        }
        next
    }

    /// Returns the address of a record with this signature: an equivalent
    /// one the registry remembers (confirmed against the written bytes),
    /// else a newly appended one.
    fn write_confirmed_state(&mut self, registry: &mut PackedRegistry, is_final: bool, children: &[(u8, u32, u32)]) -> u32 {
        let hash = signature_hash(is_final, children);
        for (way, candidate) in registry.candidates(hash).into_iter().enumerate() {
            if let Some(pos) = candidate
                && self.record_matches(pos, is_final, children)
            {
                registry.promote(hash, way);
                return pos;
            }
        }
        let pos = self.append_record(is_final, children);
        registry.insert(hash, pos);
        pos
    }

    /// Trims `bytes`' spare growth capacity, so a resident automaton's
    /// heap matches `size_bytes`.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.bytes.shrink_to_fit();
    }

    /// Rewrites `bytes` to hold only reachable records, merging records
    /// that become identical once their children are rewritten. Dead
    /// records (left behind by `insert_sorted`/`remove`) are dropped, and
    /// the exact dedup is bottom-up minimization (Revuz 1992): outputs are
    /// a function of each state's right language, so equivalent states
    /// have equal signatures once their children are canonical, and the
    /// result is the minimal automaton -- tighter than the bounded
    /// registry. Records are re-emitted in postorder, children in byte
    /// order, as `build_sorted` writes them, so NEXT targets apply.
    pub(crate) fn compact(&mut self, ws: &mut Workspace) {
        if self.pristine {
            // Already compacted, and nothing mutated since.
            return;
        }
        // Minimization runs on per-byte states: expand remaining chains.
        self.unchain();
        let old = self.take_records();
        let Workspace {
            confirm,
            moved,
            dedup,
            ..
        } = ws;
        moved.clear();
        moved.resize(old.bytes.len(), u32::MAX);
        dedup.clear();
        let pool = &mut confirm.frame_pool;
        let read = |pool: &mut Vec<Vec<(u8, u32, u32)>>, pos: u32| {
            let mut v = pool.pop().unwrap_or_default();
            v.clear();
            old.read_children_into(pos, &mut v);
            v
        };
        // (old address, its children, next child to visit)
        let root_children = read(pool, old.root);
        let mut stack: Vec<(u32, Edges, usize)> = vec![(old.root, root_children, 0)];
        while let Some((pos, children, next)) = stack.last_mut() {
            if let Some(&(_, _, child)) = children.get(*next) {
                *next += 1;
                if moved[child as usize] == u32::MAX {
                    let grandchildren = read(pool, child);
                    stack.push((child, grandchildren, 0));
                }
                continue;
            }
            let pos = *pos;
            let mut children = std::mem::take(children);
            stack.pop();
            for child in &mut children {
                child.2 = moved[child.2 as usize];
            }
            let is_final = old.is_final(pos);
            let hash = signature_hash(is_final, &children);
            let new_pos = match dedup.find(hash, |candidate| self.record_matches(candidate, is_final, &children)) {
                Some(existing) => existing,
                None => {
                    let new_pos = self.append_record(is_final, &children);
                    dedup.insert(hash, new_pos);
                    new_pos
                }
            };
            moved[pos as usize] = new_pos;
            pool.push(children);
        }
        self.root = moved[old.root as usize];
        self.bytes.shrink_to_fit();
        ws.trim();
    }

    /// `compact`, then path compression (`chain_up`) for read speed. Meant
    /// for the end of a load: a later mutation expands the chains on its
    /// path again (`expand_chain`), so chaining during a load -- e.g. on
    /// every growth-triggered compaction -- would only churn (measured:
    /// +20% insertion at 10M lines).
    pub(crate) fn compact_and_chain(&mut self, ws: &mut Workspace) {
        if self.pristine && (self.chains_coded || self.symbols.is_none()) {
            return;
        }
        // Not pristine, or chained before a symbol table existed: minimize
        // per-byte again, then chain (with the table).
        self.pristine = false;
        self.compact(ws);
        self.chain_up();
        self.bytes.shrink_to_fit();
    }

    /// `(byte, output, target)` per transition; a chain record's is its
    /// first label byte and its target (traversal helper for `chain_up`/
    /// `unchain`, which only need the record graph).
    fn record_edges(&self, pos: u32, out: &mut Vec<(u8, u32, u32)>) {
        out.clear();
        match self.chain_at(pos) {
            Some(chain) => {
                let mut label = Vec::new();
                self.chain_label(chain, &mut label);
                out.push((label[0], 0, chain.target));
            }
            None => self.read_children_into(pos, out),
        }
    }

    /// Postorder over the record graph from the root: `visit(pos, edges)`
    /// runs once per record, after all of its targets.
    fn postorder(&self, mut visit: impl FnMut(u32, &[(u8, u32, u32)])) {
        // Record addresses are dense small integers (< bytes.len()), so
        // per-record state lives in plain vectors indexed by address.
        let mut done = vec![false; self.bytes.len()];
        let mut pool: Vec<Vec<(u8, u32, u32)>> = Vec::new();
        let edges = |pool: &mut Vec<Vec<(u8, u32, u32)>>, pos: u32| {
            let mut v = pool.pop().unwrap_or_default();
            self.record_edges(pos, &mut v);
            v
        };
        let root_edges = edges(&mut pool, self.root);
        let mut stack: Vec<(u32, Edges, usize)> = vec![(self.root, root_edges, 0)];
        while let Some((pos, children, next)) = stack.last_mut() {
            if let Some(&(_, _, child)) = children.get(*next) {
                *next += 1;
                if !done[child as usize] {
                    let grandchildren = edges(&mut pool, child);
                    stack.push((child, grandchildren, 0));
                }
                continue;
            }
            let (pos, children) = (*pos, std::mem::take(children));
            stack.pop();
            if !done[pos as usize] {
                done[pos as usize] = true;
                visit(pos, &children);
            }
            pool.push(children);
        }
    }

    /// Second compaction pass (path compression): folds every run of
    /// single-transition, non-final states whose in-degree is 1 into one
    /// chain record (`append_chain`). Shared states are never absorbed, so
    /// the automaton stays exactly as minimal as `compact`'s first pass
    /// left it -- `reachable_state_count` counts a chain's states.
    fn chain_up(&mut self) {
        let old = self.take_records();
        let n = old.bytes.len();
        let mut chain_like = vec![false; n];
        let mut indeg = vec![0u32; n];
        let mut parent_chain_like = vec![false; n];
        old.postorder(|pos, edges| {
            let like = edges.len() == 1 && !old.is_final(pos);
            chain_like[pos as usize] = like;
            for &(_, _, t) in edges {
                indeg[t as usize] += 1;
                parent_chain_like[t as usize] = like;
            }
        });
        let absorbable =
            |pos: u32| chain_like[pos as usize] && indeg[pos as usize] == 1 && parent_chain_like[pos as usize];
        let mut moved = vec![u32::MAX; n];
        // An absorbed node's chain so far: its label (reversed) in `labels`
        // from `pending_start[pos]`, and the chain's final target. Labels
        // grow child-first, so one shared buffer suffices: a node's
        // pending label always ends the buffer when its parent takes it.
        let mut labels: Vec<u8> = Vec::new();
        let mut pending_start = vec![u32::MAX; n];
        let mut pending_target = vec![u32::MAX; n];
        let mut mapped: Vec<(u8, u32, u32)> = Vec::new();
        let symbols = self.symbols.clone();
        let mut coded: Vec<u8> = Vec::new();
        old.postorder(|pos, edges| {
            let p = pos as usize;
            if chain_like[p] {
                let (b, _, child) = edges[0];
                let c = child as usize;
                let (start, target) = if pending_start[c] != u32::MAX {
                    (pending_start[c] as usize, pending_target[c])
                } else {
                    (labels.len(), moved[c])
                };
                labels.push(b);
                if absorbable(pos) {
                    pending_start[p] = start as u32;
                    pending_target[p] = target;
                } else {
                    let label: Vec<u8> = labels[start..].iter().rev().copied().collect();
                    labels.truncate(start);
                    // At most `MAX_CHAIN_STATES` per chain record, last
                    // piece first (each points at the next): expanding one
                    // chain stays bounded (`crate::limits`).
                    let mut next = target;
                    for piece in label.rchunks(crate::limits::MAX_CHAIN_STATES) {
                        next = if piece.len() == 1 {
                            self.append_record(false, &[(piece[0], 0, next)])
                        } else {
                            coded.clear();
                            if let Some(table) = &symbols {
                                table.encode(piece, &mut coded);
                            }
                            if !coded.is_empty() && coded.len() < piece.len() {
                                self.append_chain(&coded, next, true)
                            } else {
                                self.append_chain(piece, next, false)
                            }
                        };
                    }
                    moved[p] = next;
                }
            } else {
                mapped.clear();
                mapped.extend(edges.iter().map(|&(b, out, t)| (b, out, moved[t as usize])));
                moved[p] = self.append_record(old.is_final(pos), &mapped);
            }
        });
        self.root = moved[old.root as usize];
        self.has_chains = true;
        self.pristine = true;
        self.chains_coded = symbols.is_some();
    }

    /// Expands every chain record back into single-transition records,
    /// for compaction's per-byte minimization. O(records); a no-op when no
    /// chain records exist.
    fn unchain(&mut self) {
        if !self.has_chains {
            return;
        }
        let old = self.take_records();
        let mut moved = vec![u32::MAX; old.bytes.len()];
        let mut mapped: Vec<(u8, u32, u32)> = Vec::new();
        old.postorder(|pos, edges| {
            let new_pos = match old.chain_at(pos) {
                Some(chain) => {
                    let mut label = Vec::new();
                    old.chain_label(chain, &mut label);
                    let mut next = moved[chain.target as usize];
                    for &b in label.iter().rev() {
                        next = self.append_record(false, &[(b, 0, next)]);
                    }
                    next
                }
                None => {
                    mapped.clear();
                    mapped.extend(edges.iter().map(|&(b, out, t)| (b, out, moved[t as usize])));
                    self.append_record(old.is_final(pos), &mapped)
                }
            };
            moved[pos as usize] = new_pos;
        });
        self.root = moved[old.root as usize];
        self.has_chains = false;
        self.pristine = false;
    }

    /// `read_children`, into a caller-provided buffer.
    fn read_children_into(&self, pos: u32, out: &mut Vec<(u8, u32, u32)>) {
        let (is_final, count, mode, mut cursor) = self.header(pos);
        for i in 0..count {
            let (b, output, target, after) = self.read_transition_at(cursor, i == 0, is_final, mode);
            cursor = after;
            out.push((b, output, target));
        }
    }

    /// Whether the record at `pos` has exactly this signature.
    fn record_matches(&self, pos: u32, is_final: bool, children: &[(u8, u32, u32)]) -> bool {
        let (f, count, mode, mut cursor) = self.header(pos);
        if f != is_final || count != children.len() {
            return false;
        }
        for (i, &expected) in children.iter().enumerate() {
            let (b, output, target, after) = self.read_transition_at(cursor, i == 0, is_final, mode);
            cursor = after;
            if (b, output, target) != expected {
                return false;
            }
        }
        true
    }

    /// The address the next appended byte gets. Record addresses are `u32`;
    /// callers check every mutation against the write bounds in
    /// `crate::limits` first (taking a rebuild path otherwise), so records
    /// never outgrow them. Still checked in release builds: a wrapped
    /// address would silently corrupt the block.
    #[inline]
    fn next_address(&self) -> u32 {
        debug_assert!(self.bytes.len() as u64 <= crate::limits::ADDRESS_LIMIT, "write bound exceeded");
        u32::try_from(self.bytes.len()).expect("a block's records exceed 4 GiB")
    }

    /// Appends one state's record to `bytes` -- ordinary amortized-growth
    /// `Vec` reallocation, no capping. This is the *only* place bytes
    /// are ever written: once appended, a record's address is permanent
    /// (see the struct doc comment for why that's sound here).
    fn append_record(&mut self, is_final: bool, children: &[(u8, u32, u32)]) -> u32 {
        debug_assert!(children.len() <= 256, "at most one transition per byte value");
        if children.is_empty() && !is_final {
            // The empty non-final state (only an empty automaton's root):
            // a chain record of length 0.
            return self.append_chain(&[], 0, false);
        }
        debug_assert!(
            children.first().is_none_or(|&(_, out, _)| out == u32::from(is_final)),
            "the first output is always is_final, so it is implicit"
        );
        if let [(byte, _, target)] = *children
            && !is_final
        {
            // Single-transition form: `1 N cccccc`.
            let next = !self.bytes.is_empty() && target as usize == self.bytes.len() - 1;
            if !next {
                let field = self.next_address();
                write_leb128_rev(&mut self.bytes.to_mut(), field - target);
            }
            let code = label_table().code(byte);
            if code == 0 {
                self.bytes.to_mut().push(byte);
            }
            self.bytes.to_mut().push(0x80 | (u8::from(next) << 6) | code);
            return self.next_address() - 1;
        }
        // General form: `0 F nnnnnn`, transitions written in reverse so
        // the record reads forward from its header downward.
        for (i, &(b, out, target)) in children.iter().enumerate().rev() {
            let field = self.next_address();
            write_leb128_rev(&mut self.bytes.to_mut(), field - target);
            if i > 0 {
                write_leb128_rev(&mut self.bytes.to_mut(), out);
            }
            self.bytes.to_mut().push(b);
        }
        if children.len() >= PACKED_COUNT_ESCAPE {
            self.bytes.to_mut().push((children.len() - PACKED_COUNT_ESCAPE) as u8);
        }
        self.bytes
            .to_mut()
            .push((u8::from(is_final) << 6) | children.len().min(PACKED_COUNT_ESCAPE) as u8);
        self.next_address() - 1
    }

    /// Total size of `bytes` -- directly comparable to `Automaton::
    /// estimate_packed_bytes`'s prediction for the same content, though
    /// only exactly equal right after a fresh `compile` (see the struct
    /// doc comment: `insert` can leave dead, unreachable bytes behind,
    /// which this still counts).
    pub(crate) fn size_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Occupied registry cells after the last operation that used `ws`
    /// (bounded by the registry's fixed size).
    #[cfg(test)]
    pub(crate) fn register_len(ws: &Workspace) -> usize {
        ws.confirm.registry.len()
    }

    /// The number of *reachable* records -- `Automaton::
    /// reachable_state_count`'s counterpart, walking live children
    /// instead of a slab. The true minimality signal for this
    /// representation: `insert` can leave dead, unreachable records
    /// behind in `blocks` (see the struct doc comment), so this is what
    /// a minimality regression test should compare against a fresh
    /// `Automaton::build_sorted`'s own `reachable_state_count` over the
    /// same final term set, not `size_bytes`/blocks length. No longer
    /// test-only: used directly by `PackedAutomatonBlock`/
    /// `PackedAutomatonDictionary`'s own `reachable_state_count` for the
    /// states-per-term investigation (DESIGN.md).
    /// States, not records: a chain record holds one state per label
    /// byte.
    pub(crate) fn reachable_state_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![self.root];
        let mut states = 0usize;
        while let Some(pos) = stack.pop() {
            if seen.insert(pos) {
                if let Some(chain) = self.chain_at(pos) {
                    states += self.chain_states(chain);
                    stack.push(chain.target);
                    continue;
                }
                states += 1;
                for &(_, _, target) in &self.read_children(pos) {
                    stack.push(target);
                }
            }
        }
        states
    }

    /// Mirrors `Automaton::rank`'s contract exactly.
    pub(crate) fn rank(&self, term: &str) -> Option<u32> {
        if self.has_chains {
            return self.rank_chained(term);
        }
        let mut pos = self.root;
        let mut value = 0u32;
        for &b in term.as_bytes() {
            let (output, target) = self.get(pos, b)?;
            value += output;
            pos = target;
        }
        self.is_final(pos).then_some(value)
    }

    #[inline(never)]
    fn rank_chained(&self, term: &str) -> Option<u32> {
        let t = term.as_bytes();
        let mut pos = self.root;
        let mut value = 0u32;
        let mut at = 0usize;
        loop {
            if self.bytes[pos as usize] == 0 {
                match self.chain_step(pos, &t[at..]) {
                    Some(ChainStep::Through(len, target)) => {
                        at += len;
                        pos = target;
                        continue;
                    }
                    Some(_) => return None,
                    None => {}
                }
            }
            let Some(&b) = t.get(at) else {
                return self.is_final(pos).then_some(value);
            };
            let (output, target) = self.get(pos, b)?;
            value += output;
            pos = target;
            at += 1;
        }
    }

    /// Mirrors `Automaton::term_at_into`'s contract exactly.
    pub(crate) fn term_at_into(&self, rank: u32, buf: &mut Vec<u8>) {
        if self.has_chains {
            return self.term_at_chained(rank, buf);
        }
        buf.clear();
        let mut value = rank;
        let mut pos = self.root;
        loop {
            if value == 0 && self.is_final(pos) {
                return;
            }
            let (b, out, target) = self.last_output_at_most(pos, value);
            value -= out;
            buf.push(b);
            pos = target;
        }
    }

    #[inline(never)]
    fn term_at_chained(&self, rank: u32, buf: &mut Vec<u8>) {
        buf.clear();
        let mut value = rank;
        let mut pos = self.root;
        loop {
            if self.bytes[pos as usize] == 0
                && let Some(target) = self.push_chain_label(pos, buf)
            {
                pos = target;
                continue;
            }
            if value == 0 && self.is_final(pos) {
                return;
            }
            let (b, out, target) = self.last_output_at_most(pos, value);
            value -= out;
            buf.push(b);
            pos = target;
        }
    }

    /// Decodes `targets` (`(output position, rank)`, sorted by rank; ranks
    /// may repeat) into `out[position]`. Iterative over an explicit path
    /// (a per-byte recursion overflowed the stack on long terms): each
    /// target pops back to the deepest state on the current path whose
    /// rank range holds it, then descends from there, so consecutive
    /// targets share their common prefix's walk.
    pub(crate) fn decode_ranks_into(&self, targets: &[(usize, u32)], out: &mut [Option<String>]) {
        // (record, first rank of its subtree, end rank, buffer length on entry)
        let mut path: Vec<(u32, u32, u32, usize)> = Vec::new();
        let mut buf: Vec<u8> = Vec::new();
        debug_assert!(targets.is_sorted_by_key(|&(_, rank)| rank), "targets must be sorted by rank");
        for &(position, rank) in targets {
            while let Some(&(_, lo, hi, len)) = path.last()
                && !(lo..hi).contains(&rank)
            {
                path.pop();
                buf.truncate(len);
            }
            if path.is_empty() {
                if rank >= self.term_count {
                    continue; // Not a rank of this automaton.
                }
                path.push((self.root, 0, self.term_count, 0));
            }
            loop {
                let &(pos, lo, hi, _) = path.last().expect("non-empty path");
                if let Some(chain) = self.chain_at(pos) {
                    let len = buf.len();
                    self.chain_label(chain, &mut buf);
                    path.push((chain.target, lo, hi, len));
                    continue;
                }
                let value = rank - lo;
                if value == 0 && self.is_final(pos) {
                    out[position] = Some(decoded_term(buf.clone()));
                    break;
                }
                let (byte, child_lo, child_hi, target) = self.child_holding(pos, value, hi - lo);
                let len = buf.len();
                buf.push(byte);
                path.push((target, lo + child_lo, lo + child_hi, len));
            }
        }
    }

    /// The transition of the record at `pos` (subtree count `count`) whose
    /// subtree holds local rank `value`: `(byte, first local rank, end,
    /// target)`, the end being the next transition's output (or `count`).
    fn child_holding(&self, pos: u32, value: u32, count: u32) -> (u8, u32, u32, u32) {
        let (is_final, n, mode, mut cursor) = self.header(pos);
        let mut best: Option<(u8, u32, u32)> = None;
        for i in 0..n {
            let (b, output, target, after) = self.read_transition_at(cursor, i == 0, is_final, mode);
            cursor = after;
            if output > value {
                let (b0, o0, t0) = best.expect("value must be reachable via some transition at this state");
                return (b0, o0, output, t0);
            }
            best = Some((b, output, target));
        }
        let (b, output, target) = best.expect("value must be reachable via some transition at this state");
        (b, output, count, target)
    }

    /// Appends a chain record: single-transition, non-final states in one
    /// record (path compression). Layout, reading downward from the `0x00`
    /// header: `len << 2 | coded << 1 | next` (reversed LEB128), the
    /// stored label (`len` bytes, forward order, so it is one contiguous
    /// slice; FSST-coded if `coded`), then the target delta unless NEXT
    /// (target = the record just below the label). Length 0 is the empty
    /// non-final state, with no target.
    fn append_chain(&mut self, label: &[u8], target: u32, coded: bool) -> u32 {
        let mut next = false;
        if !label.is_empty() {
            next = !self.bytes.is_empty() && target as usize == self.bytes.len() - 1;
            if !next {
                let field = self.next_address();
                write_leb128_rev(&mut self.bytes.to_mut(), field - target);
            }
            self.bytes.to_mut().extend_from_slice(label);
        }
        let len = u32::try_from(label.len()).expect("chain label fits u32");
        write_leb128_rev(&mut self.bytes.to_mut(), len << 2 | u32::from(coded) << 1 | u32::from(next));
        self.bytes.to_mut().push(0);
        self.next_address() - 1
    }

    /// The chain record at `pos`, if it is a non-empty one. Checks
    /// `has_chains` first: no automaton has chains during a load (only
    /// `compact_and_chain` creates them), and an unguarded check on every
    /// step of `lower_bound` measured as +54% of its time.
    #[inline(always)]
    fn chain_at(&self, pos: u32) -> Option<Chain> {
        if !self.has_chains || self.bytes[pos as usize] != 0 {
            return None;
        }
        let (packed_len, start) = read_leb128_rev(&self.bytes, pos as usize - 1);
        let len = (packed_len >> 2) as usize;
        if len == 0 {
            return None;
        }
        let label_start = start - len;
        let target = if packed_len & 1 == 1 {
            (label_start - 1) as u32
        } else {
            let (delta, field) = read_leb128_rev(&self.bytes, label_start - 1);
            field as u32 - delta
        };
        Some(Chain {
            start: label_start,
            len,
            target,
            coded: packed_len & 2 == 2,
        })
    }

    fn table(&self) -> &crate::fsst::SymbolTable {
        self.symbols.as_deref().expect("coded chain without a symbol table")
    }

    /// Appends `chain`'s decoded label to `out`; returns its length (the
    /// chain's number of states).
    #[inline]
    fn chain_label(&self, chain: Chain, out: &mut Vec<u8>) -> usize {
        let before = out.len();
        let stored = &self.bytes[chain.start..chain.start + chain.len];
        if chain.coded {
            self.table().decode(stored, out);
        } else {
            out.extend_from_slice(stored);
        }
        out.len() - before
    }

    /// `crate::fsst::SymbolTable::match_prefix`'s contract for `chain`.
    #[inline]
    fn chain_match(&self, chain: Chain, query: &[u8]) -> Result<usize, (usize, Option<u8>)> {
        let stored = &self.bytes[chain.start..chain.start + chain.len];
        if chain.coded {
            return self.table().match_prefix(stored, query);
        }
        let m = stored.iter().zip(query).take_while(|(a, b)| a == b).count();
        if m == stored.len() {
            Ok(m)
        } else {
            Err((m, query.get(m).map(|_| stored[m])))
        }
    }

    fn chain_states(&self, chain: Chain) -> usize {
        if chain.coded {
            self.table().decoded_len(&self.bytes[chain.start..chain.start + chain.len])
        } else {
            chain.len
        }
    }

    #[cfg(test)]
    pub(crate) fn chains_coded(&self) -> bool {
        self.chains_coded
    }

    /// Makes `symbols` this automaton's chain-label table (the next
    /// `compact_and_chain` codes chains with it).
    pub(crate) fn set_symbols(&mut self, symbols: Option<std::sync::Arc<crate::fsst::SymbolTable>>) {
        self.symbols = symbols;
    }

    /// Appends every chain label (decoded) -- a training sample.
    pub(crate) fn chain_labels_into(&self, out: &mut Vec<Vec<u8>>) {
        self.postorder(|pos, _| {
            if let Some(chain) = self.chain_at(pos) {
                let mut label = Vec::new();
                self.chain_label(chain, &mut label);
                out.push(label);
            }
        });
    }

    #[inline]
    fn is_final(&self, pos: u32) -> bool {
        self.bytes[pos as usize] & 0xC0 == 0x40
    }


    /// Follows chain records from `pos`, appending their labels to `buf`;
    /// returns the first non-chain record and the bytes appended.
    fn resolve_chains(&self, mut pos: u32, buf: &mut Vec<u8>) -> (u32, usize) {
        let mut pushed = 0;
        while let Some(chain) = self.chain_at(pos) {
            pushed += self.chain_label(chain, buf);
            pos = chain.target;
        }
        (pos, pushed)
    }

    /// `(is_final, transition_count, mode, cursor)` -- `cursor` is the
    /// offset of the first transition's first field below the header;
    /// transitions are parsed downward from there (see the struct doc
    /// comment).
    #[inline]
    fn header(&self, pos: u32) -> (bool, usize, RecordMode, usize) {
        let header = self.bytes[pos as usize];
        let mut cursor = (pos as usize).wrapping_sub(1);
        if header == 0 {
            // A chain record; callers resolve non-empty chains first
            // (`chain_at`), so this is the empty non-final state.
            debug_assert!(self.chain_at(pos).is_none(), "header() on a chain record");
            return (false, 0, RecordMode::GENERAL, cursor);
        }
        if header & 0x80 != 0 {
            let code = header & 0x3F;
            let mode = RecordMode {
                next: header & 0x40 != 0,
                label: (code != 0).then(|| label_table().byte(code)),
            };
            return (false, 1, mode, cursor);
        }
        let mut count = (header & 0x3F) as usize;
        if count == PACKED_COUNT_ESCAPE {
            count += self.bytes[cursor] as usize;
            cursor = cursor.wrapping_sub(1);
        }
        (header & 0x40 != 0, count, RecordMode::GENERAL, cursor)
    }

    /// Sequentially scans this state's transitions for `byte`, parsing
    /// each record in turn -- unlike the fixed-array scheme this
    /// replaces, a variable-length record can't jump straight to slot
    /// `i`, so this is O(branching factor) *parse work*, not just an
    /// O(branching factor) *scan* over already-typed fields. Whether
    /// that parse cost is offset by better cache locality is exactly
    /// what this prototype exists to measure, not assume.
    ///
    /// `#[inline]`: a `perf record` profile of the hashlike decode probe
    /// found this got inlined into `rank` by default, while
    /// `last_output_at_most` (same shape, called from `term_at_into`)
    /// didn't -- the resulting real function-call-per-visited-state cost
    /// was a plausible, testable explanation for `term_at_into` showing
    /// no speedup while `rank` showed ~1.5-1.65x, an asymmetry that
    /// looked structural at first but wasn't. Hinting both explicitly
    /// removes the asymmetry as a variable rather than leaving it to
    /// whatever LLVM's default heuristic happens to decide.
    #[inline]
    fn get(&self, pos: u32, byte: u8) -> Option<(u32, u32)> {
        let (is_final, count, next, mut cursor) = self.header(pos);
        for i in 0..count {
            let (b, output, target, after) = self.read_transition_at(cursor, i == 0, is_final, next);
            cursor = after;
            if b == byte {
                return Some((output, target));
            }
        }
        None
    }

    /// The last transition (in byte order) whose output is `<= value` --
    /// `Transitions::last_output_at_most`'s contract, over parsed
    /// records instead of a typed array. `#[inline(always)]`: plain
    /// `#[inline]` (a hint, not a command) measurably wasn't enough --
    /// a `perf record` profile still showed this as its own symbol
    /// afterward, unlike `get`, which LLVM folded into `rank` either
    /// way. See `get`'s doc comment for the fuller story.
    #[inline(always)]
    fn last_output_at_most(&self, pos: u32, value: u32) -> (u8, u32, u32) {
        let (is_final, count, next, mut cursor) = self.header(pos);
        let mut best: Option<(u8, u32, u32)> = None;
        for i in 0..count {
            let (b, output, target, after) = self.read_transition_at(cursor, i == 0, is_final, next);
            cursor = after;
            if output > value {
                return best.expect("value must be reachable via some transition at this state");
            }
            best = Some((b, output, target));
        }
        best.expect("value must be reachable via some transition at this state")
    }

    /// Every `(byte, output, target)` at `pos`, in byte-sorted order --
    /// `insert`'s own way of "opening" a matched state into an editable
    /// [`StagingFrame`].
    fn read_children(&self, pos: u32) -> Vec<(u8, u32, u32)> {
        let (is_final, count, next, mut cursor) = self.header(pos);
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let (b, output, target, after) = self.read_transition_at(cursor, i == 0, is_final, next);
            cursor = after;
            result.push((b, output, target));
        }
        result
    }

    /// Parses one transition at `cursor` (a byte offset into `bytes`,
    /// pointing at that transition's own leading `byte` field) --
    /// `(byte, output, target, cursor_after)`. The same inline parse
    /// `get`/`last_output_at_most`/`read_children` each already do in
    /// their own loops, factored out here specifically for
    /// `PackedTermsStream` (below), which needs the returned cursor to
    /// resume a state's transition scan next time without re-walking
    /// from its first transition -- not retrofitted into those other
    /// three call sites, deliberately: they're on the already-measured,
    /// delicate `rank`/`term_at_into` hot path (see `get`'s own doc
    /// comment on how sensitive that path proved to be to exactly this
    /// kind of restructuring), and this function existing is no reason
    /// to risk perturbing it again without a fresh measurement.
    ///
    /// `first`/`is_final`/`mode` come from the record's header: the first
    /// transition's output is implicit (`is_final`), a folded label isn't
    /// stored, and a NEXT record's one target is the record immediately
    /// below it.
    #[inline(always)]
    fn read_transition_at(
        &self,
        cursor: usize,
        first: bool,
        is_final: bool,
        mode: RecordMode,
    ) -> (u8, u32, u32, usize) {
        let (b, mut cursor) = match mode.label {
            Some(label) => (label, cursor),
            None => (self.bytes[cursor], cursor.wrapping_sub(1)),
        };
        let output = if first {
            u32::from(is_final)
        } else {
            let (value, start) = read_leb128_rev(&self.bytes, cursor);
            cursor = start.wrapping_sub(1);
            value
        };
        let target = if mode.next {
            cursor as u32
        } else {
            let (delta, start) = read_leb128_rev(&self.bytes, cursor);
            cursor = start.wrapping_sub(1);
            start as u32 - delta
        };
        (b, output, target, cursor)
    }

    /// `Automaton::stream`'s exact contract and reason for existing (see
    /// its own doc comment): a single incremental walk visiting every
    /// term in sorted order, sharing a common prefix's traversal cost
    /// between consecutive terms instead of paying a full root descent
    /// per term the way one `term_at_into` call per rank does. Built for
    /// `PackedAutomatonBlock::decode_into`/`merge_into`, which used that
    /// per-rank shape as an accepted simplification until now (see
    /// `packed_automaton_block.rs`'s module doc comment and DESIGN.md's
    /// dictionary-level insert-speed section: this is the traced, not
    /// yet closed, cause of `PackedAutomatonDictionary`'s
    /// `insert_batch` gap against `AutomatonDictionary`).
    ///
    /// Each [`PackedStreamFrame`] caches its own parse cursor (the byte
    /// offset of its *next* unvisited transition), advanced by one
    /// transition per `next()` call into it -- unlike `Transitions::at`
    /// (which `Automaton`'s own `StreamFrame` re-derives from `next_child`
    /// on every visit, cheap for a small typed array), a variable-length
    /// packed record has no such O(1) re-entry point, so re-deriving
    /// position `i` would mean re-parsing every transition before it
    /// again. Caching the cursor makes each state's own transitions
    /// still cost exactly one parse each, total, matching `Automaton::
    /// stream`'s own amortized cost rather than a naive port of its
    /// per-visit re-lookup.
    pub(crate) fn stream(&self) -> PackedTermsStream<'_> {
        let mut buf = Vec::new();
        let (root, pushed) = self.resolve_chains(self.root, &mut buf);
        let (is_final, count, next, cursor) = self.header(root);
        PackedTermsStream {
            packed: self,
            buf,
            stack: vec![PackedStreamFrame {
                pushed,
                is_final,
                next,
                first: true,
                yielded_final: false,
                remaining: count,
                cursor,
            }],
        }
    }
}

#[allow(dead_code)]
fn write_leb128(bytes: &mut Vec<u8>, mut value: u32) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            bytes.push(byte);
            break;
        }
        bytes.push(byte | 0x80);
    }
}

/// Hash of a record's signature: one FxHash round per transition (byte,
/// output and target packed into one word; overlapping bits only cost hash
/// quality, since hits are verified against the record), then a fold so
/// the low bits (registry bucket) depend on every word, not only the high
/// bits (registry tag).
fn signature_hash(is_final: bool, children: &[(u8, u32, u32)]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.add((children.len() as u64) << 1 | u64::from(is_final));
    for &(byte, output, target) in children {
        hasher.add(u64::from(target) << 32 ^ u64::from(output) << 8 ^ u64::from(byte));
    }
    let h = hasher.finish();
    (h ^ h >> 32).wrapping_mul(FX_SEED)
}

/// `compact`'s exact dedup: open addressing over `(signature hash, new
/// record address)`. A probe hit is confirmed against the record already
/// written (`record_matches`), so no signature is stored -- the previous
/// `HashMap<(bool, Vec<..>), u32>` allocated a `Vec` key per record.
#[derive(Debug, Default)]
struct DedupTable {
    /// The active table (a power of two, at least `MIN_SLOTS`) of `(hash,
    /// address + 1)`, `None` marking an empty slot; its allocation is kept
    /// across `clear`s.
    slots: Vec<(u64, Option<NonZeroU32>)>,
    /// The previous table while growing, kept for its allocation.
    spare: Vec<(u64, Option<NonZeroU32>)>,
    len: usize,
}

impl DedupTable {
    const MIN_SLOTS: usize = 256;

    /// Empties the table back to `MIN_SLOTS` slots. Only the active slots
    /// are reset: a compaction sized for one huge block must not make every
    /// later one pay to clear its table.
    fn clear(&mut self) {
        self.slots.clear();
        self.slots.resize(Self::MIN_SLOTS, (0, None));
        self.len = 0;
    }

    fn find(&self, hash: u64, mut matches: impl FnMut(u32) -> bool) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        loop {
            let (h, stored) = self.slots[i];
            let pos = stored?.get() - 1;
            if h == hash && matches(pos) {
                return Some(pos);
            }
            i = (i + 1) & mask;
        }
    }

    fn insert(&mut self, hash: u64, pos: u32) {
        if (self.len + 1) * 2 > self.slots.len() {
            let doubled = self.slots.len() * 2;
            std::mem::swap(&mut self.slots, &mut self.spare);
            self.slots.clear();
            self.slots.resize(doubled, (0, None));
            self.len = 0;
            for i in 0..self.spare.len() {
                if let (h, Some(stored)) = self.spare[i] {
                    self.place(h, stored);
                }
            }
        }
        self.place(hash, NonZeroU32::MIN.saturating_add(pos));
    }

    fn place(&mut self, hash: u64, stored: NonZeroU32) {
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        while self.slots[i].1.is_some() {
            i = (i + 1) & mask;
        }
        self.slots[i] = (hash, Some(stored));
        self.len += 1;
    }
}

/// What a record's header says about how to parse its transitions: NEXT
/// (single transition, target is the record immediately below) and a
/// label folded into the header (single transition, label not stored).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RecordMode {
    next: bool,
    label: Option<u8>,
}

impl RecordMode {
    const GENERAL: RecordMode = RecordMode {
        next: false,
        label: None,
    };
}

/// The 63 labels a single-transition record can fold into its header
/// (6-bit code `1..=63`; 0 = label stored separately) -- `fst`'s
/// `COMMON_INPUTS` idea (`vendor/fst/src/raw/common_inputs.rs`).
pub(crate) struct LabelTable {
    bytes: [u8; 64],
    codes: [u8; 256],
}

impl LabelTable {
    const fn new(labels: &[u8; 63]) -> LabelTable {
        let mut bytes = [0u8; 64];
        let mut codes = [0u8; 256];
        let mut i = 0;
        while i < 63 {
            bytes[i + 1] = labels[i];
            codes[labels[i] as usize] = (i + 1) as u8;
            i += 1;
        }
        LabelTable { bytes, codes }
    }

    #[inline]
    fn code(&self, byte: u8) -> u8 {
        self.codes[byte as usize]
    }

    #[inline]
    fn byte(&self, code: u8) -> u8 {
        self.bytes[code as usize]
    }
}

/// Curated for N-Triples terms: digits, lowercase, the IRI/literal
/// punctuation `<>"/:.#-_@%^`, space, and 14 common capitals. On real RDF
/// (DESIGN.md's "label folding") it beat both `fst`'s own table (tuned for
/// general text: no `<`, `>`, `"`, `#`, `@`) and a table adapted to the
/// first batch's byte frequencies (term bytes aren't distributed like the
/// labels of single-transition records, mostly unique tails).
static RDF_LABELS: LabelTable = LabelTable::new(
    b"0123456789abcdefghijklmnopqrstuvwxyz<>\"/:.#-_@%^ ABCDEFILMNPRST",
);

#[inline]
fn label_table() -> &'static LabelTable {
    &RDF_LABELS
}

/// `PackedAutomaton`'s header escape: a transition count `>= 63` stores
/// 63 in the header's 6-bit field plus `count - 63` in one extra byte.
const PACKED_COUNT_ESCAPE: usize = 63;

/// LEB128 with its bytes reversed, so a reader walking *downward* from
/// the highest byte sees the least-significant group first.
fn write_leb128_rev(bytes: &mut Vec<u8>, value: u32) {
    let start = bytes.len();
    write_leb128(bytes, value);
    bytes[start..].reverse();
}

/// Reads a [`write_leb128_rev`] value whose highest byte is at `end`;
/// returns `(value, lowest_byte_index)`.
#[inline(always)]
fn read_leb128_rev(bytes: &[u8], end: usize) -> (u32, usize) {
    let mut value = 0u32;
    let mut shift = 0;
    let mut i = end;
    loop {
        let byte = bytes[i];
        value |= u32::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return (value, i);
        }
        shift += 7;
        i -= 1;
    }
}

/// Returns `(value, bytes_consumed)`.
#[allow(dead_code)]
fn read_leb128(bytes: &[u8]) -> (u32, usize) {
    let mut value = 0u32;
    let mut shift = 0;
    let mut len = 0;
    loop {
        let byte = bytes[len];
        value |= u32::from(byte & 0x7F) << shift;
        len += 1;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (value, len)
}

/// One state's streaming position for [`PackedTermsStream`] --
/// [`StreamFrame`]'s counterpart, caching a parse cursor instead of an
/// index, per [`PackedAutomaton::stream`]'s own doc comment.
struct PackedStreamFrame {
    /// Bytes appended to the buffer when this frame was entered (its
    /// transition byte plus any chain labels resolved after it).
    pushed: usize,
    is_final: bool,
    /// The record's NEXT flag and folded label (see `read_transition_at`).
    next: RecordMode,
    /// No transition visited yet (the first one's output is implicit).
    first: bool,
    yielded_final: bool,
    /// Transitions not yet visited at this state.
    remaining: usize,
    /// Byte offset into `PackedTermsStream::packed`'s `bytes` of the
    /// next unvisited transition (meaningless once `remaining == 0`).
    cursor: usize,
}

/// [`TermsStream`]'s exact role, over a [`PackedAutomaton`] instead of an
/// [`Automaton`] -- see [`PackedAutomaton::stream`]'s doc comment.
#[allow(dead_code)]
pub(crate) struct PackedTermsStream<'a> {
    packed: &'a PackedAutomaton,
    buf: Vec<u8>,
    stack: Vec<PackedStreamFrame>,
}

#[allow(dead_code)]
impl<'a> PackedTermsStream<'a> {
    /// `TermsStream::next`'s exact contract: the next term in sorted
    /// order, or `None` once exhausted.
    pub(crate) fn next(&mut self) -> Option<&[u8]> {
        loop {
            let depth = self.stack.len();
            if depth == 0 {
                return None;
            }
            let already_yielded = self.stack[depth - 1].yielded_final;
            let is_final = self.stack[depth - 1].is_final;
            self.stack[depth - 1].yielded_final = true;
            if !already_yielded && is_final {
                return Some(&self.buf);
            }
            if self.stack[depth - 1].remaining == 0 {
                let pushed = self.stack.pop().expect("depth > 0").pushed;
                self.buf.truncate(self.buf.len() - pushed);
                continue;
            }
            let frame = &mut self.stack[depth - 1];
            let (b, _out, target, next_cursor) =
                self.packed
                    .read_transition_at(frame.cursor, frame.first, frame.is_final, frame.next);
            frame.first = false;
            frame.remaining -= 1;
            frame.cursor = next_cursor;
            self.buf.push(b);
            let (target, chained) = if self.packed.has_chains {
                self.packed.resolve_chains(target, &mut self.buf)
            } else {
                (target, 0)
            };
            let (child_is_final, child_count, child_next, child_cursor) = self.packed.header(target);
            self.stack.push(PackedStreamFrame {
                pushed: 1 + chained,
                is_final: child_is_final,
                next: child_next,
                first: true,
                yielded_final: false,
                remaining: child_count,
                cursor: child_cursor,
            });
        }
    }
}

/// A term decoded from an automaton, as a `String`, without re-validating
/// it. Premise: terms only ever enter a dictionary as `&str`
/// (`TermDictionary::insert`/`insert_batch`, `TermBatch::push`), and every
/// byte string an automaton yields is one of them -- construction,
/// incremental mutation, compaction and path compression all preserve the
/// accepted language exactly, and FSST-coded chain labels decode to the
/// bytes they encoded. Checked in debug builds.
pub(crate) fn decoded_term(bytes: Vec<u8>) -> String {
    debug_assert!(
        std::str::from_utf8(&bytes).is_ok(),
        "a decoded term is not UTF-8: the automaton no longer yields exactly its inserted terms"
    );
    // SAFETY: by the premise above, `bytes` is a complete term that was
    // inserted as a `&str`, hence valid UTF-8.
    unsafe { String::from_utf8_unchecked(bytes) }
}

/// `PackedAutomaton` flag bits as persisted (`crate::packed_automaton_dictionary::persist`).
pub(crate) const FLAG_HAS_CHAINS: u8 = 1;
pub(crate) const FLAG_PRISTINE: u8 = 2;
pub(crate) const FLAG_CHAINS_CODED: u8 = 4;
const KNOWN_FLAGS: u8 = FLAG_HAS_CHAINS | FLAG_PRISTINE | FLAG_CHAINS_CODED;

impl PackedAutomaton {
    /// `(records, root, term count, flags)`, for persistence.
    pub(crate) fn parts(&self) -> (&[u8], u32, u32, u8) {
        let flags = (u8::from(self.has_chains) * FLAG_HAS_CHAINS)
            | (u8::from(self.pristine) * FLAG_PRISTINE)
            | (u8::from(self.chains_coded) * FLAG_CHAINS_CODED);
        (&self.bytes, self.root, self.term_count, flags)
    }

    pub(crate) fn has_symbols(&self) -> bool {
        self.symbols.is_some()
    }

    /// An automaton from persisted parts, after checking that every read
    /// and mutation path will find what it relies on (see [`validate`]).
    /// Returns it with the total bytes of its terms.
    pub(crate) fn from_parts(
        bytes: crate::records::Records,
        root: u32,
        term_count: u32,
        flags: u8,
        symbols: Option<std::sync::Arc<crate::fsst::SymbolTable>>,
    ) -> Result<(PackedAutomaton, u64), &'static str> {
        if flags & !KNOWN_FLAGS != 0 {
            return Err("unknown automaton flags");
        }
        let automaton = PackedAutomaton {
            bytes,
            root,
            term_count,
            has_chains: flags & FLAG_HAS_CHAINS != 0,
            pristine: flags & FLAG_PRISTINE != 0,
            symbols,
            chains_coded: flags & FLAG_CHAINS_CODED != 0,
        };
        let text = validate(&automaton)?;
        Ok((automaton, text))
    }

    /// The smallest and largest terms (`term_count > 0`).
    pub(crate) fn first_and_last_terms(&self) -> (Vec<u8>, Vec<u8>) {
        let (mut first, mut last) = (Vec::new(), Vec::new());
        self.term_at_into(0, &mut first);
        self.term_at_into(self.term_count - 1, &mut last);
        (first, last)
    }
}

/// One record as [`validate`] parsed it: a state (`edges` are its
/// transitions) or a chain (`edges` holds its one target, `label` its
/// decoded label).
struct ParsedNode {
    pos: u32,
    is_final: bool,
    /// `label_start..label_end` in `Parsed::labels` for a chain.
    chain: Option<(u32, u32)>,
    edges: std::ops::Range<u32>,
}

#[derive(Default)]
struct Parsed {
    nodes: Vec<ParsedNode>,
    /// `(byte, output, target node)`; a chain's one edge has byte 0.
    edges: Vec<(u8, u32, u32)>,
    labels: Vec<u8>,
}

/// Reads a [`write_leb128_rev`] value ending at `end` as
/// [`read_leb128_rev`] does, refusing anything it would misread: an
/// out-of-range byte, or more than the 5 bytes (and 32 bits) of a `u32`.
fn checked_leb128_rev(bytes: &[u8], end: Option<usize>) -> Result<(u32, usize), &'static str> {
    let mut i = end.ok_or("field below the records")?;
    let mut value = 0u32;
    for k in 0..5 {
        let byte = *bytes.get(i).ok_or("field outside the records")?;
        if k == 4 && byte > 0x0F {
            return Err("LEB128 value exceeds u32");
        }
        value |= u32::from(byte & 0x7F) << (7 * k);
        if byte & 0x80 == 0 {
            return Ok((value, i));
        }
        i = i.checked_sub(1).ok_or("field below the records")?;
    }
    Err("LEB128 value exceeds u32")
}

/// A record as [`parse_checked`] reads it: `(is_final, chain target,
/// transitions)` -- a chain has a target (its label in the caller's
/// buffer) and no transitions.
type CheckedRecord = (bool, Option<u32>, Vec<(u8, u32, u32)>);

/// Parses the record at `pos` exactly as the readers do (`chain_at`,
/// `header`, `read_transition_at`), with every address and offset
/// checked; targets as addresses.
fn parse_checked(a: &PackedAutomaton, pos: u32, label: &mut Vec<u8>) -> Result<CheckedRecord, &'static str> {
    let bytes: &[u8] = &a.bytes;
    let p = pos as usize;
    let header = *bytes.get(p).ok_or("record outside the records")?;
    if a.has_chains && header == 0 {
        let (packed_len, start) = checked_leb128_rev(bytes, p.checked_sub(1))?;
        let len = (packed_len >> 2) as usize;
        if len != 0 {
            let label_start = start.checked_sub(len).ok_or("chain label below the records")?;
            let target = if packed_len & 1 == 1 {
                label_start.checked_sub(1).ok_or("chain target below the records")? as u32
            } else {
                let (delta, field) = checked_leb128_rev(bytes, label_start.checked_sub(1))?;
                (field as u32).checked_sub(delta).ok_or("chain target below the records")?
            };
            let stored = &bytes[label_start..start];
            label.clear();
            if packed_len & 2 == 2 {
                let table = a.symbols.as_deref().ok_or("coded chain without a symbol table")?;
                table.checked_decode(stored, label)?;
            } else {
                label.extend_from_slice(stored);
            }
            if label.is_empty() || label.len() > crate::limits::MAX_CHAIN_STATES {
                return Err("chain label length out of range");
            }
            return Ok((false, Some(target), Vec::new()));
        }
    }
    if header == 0 {
        // The empty non-final state (a length-0 chain record).
        return Ok((false, None, Vec::new()));
    }
    let mut cursor = p.checked_sub(1);
    let (is_final, count, mode) = if header & 0x80 != 0 {
        let code = header & 0x3F;
        let mode = RecordMode {
            next: header & 0x40 != 0,
            label: (code != 0).then(|| label_table().byte(code)),
        };
        (false, 1, mode)
    } else {
        let mut count = (header & 0x3F) as usize;
        if count == PACKED_COUNT_ESCAPE {
            let at = cursor.ok_or("count escape below the records")?;
            count += *bytes.get(at).ok_or("count escape outside the records")? as usize;
            cursor = at.checked_sub(1);
        }
        (header & 0x40 != 0, count, RecordMode::GENERAL)
    };
    let mut children = Vec::with_capacity(count);
    for i in 0..count {
        let b = match mode.label {
            Some(label) => label,
            None => {
                let at = cursor.ok_or("transition below the records")?;
                cursor = at.checked_sub(1);
                *bytes.get(at).ok_or("transition outside the records")?
            }
        };
        let output = if i == 0 {
            u32::from(is_final)
        } else {
            let (value, start) = checked_leb128_rev(bytes, cursor)?;
            cursor = start.checked_sub(1);
            value
        };
        let target = if mode.next {
            cursor.ok_or("NEXT target below the records")? as u32
        } else {
            let (delta, start) = checked_leb128_rev(bytes, cursor)?;
            cursor = start.checked_sub(1);
            (start as u32).checked_sub(delta).ok_or("transition target below the records")?
        };
        children.push((b, output, target));
    }
    Ok((is_final, None, children))
}

/// UTF-8 decoder states for checking an automaton's language: whether
/// every path to a final state spells valid UTF-8, one visit per
/// (record, state) pair rather than per term.
const UTF8_ACCEPT: u8 = 0;
const UTF8_REJECT: u8 = 8;

fn utf8_step(state: u8, b: u8) -> u8 {
    let cont = |next: u8, lo: u8, hi: u8| if (lo..=hi).contains(&b) { next } else { UTF8_REJECT };
    match state {
        UTF8_ACCEPT => match b {
            0x00..=0x7F => UTF8_ACCEPT,
            0xC2..=0xDF => 1,
            0xE1..=0xEC | 0xEE..=0xEF => 2,
            0xF1..=0xF3 => 3,
            0xE0 => 4,
            0xED => 5,
            0xF0 => 6,
            0xF4 => 7,
            _ => UTF8_REJECT,
        },
        // 1, 2, 3: that many continuation bytes still to come.
        1 => cont(UTF8_ACCEPT, 0x80, 0xBF),
        2 => cont(1, 0x80, 0xBF),
        3 => cont(2, 0x80, 0xBF),
        // The lead bytes whose second byte is restricted (no overlongs,
        // no surrogates, nothing above U+10FFFF).
        4 => cont(1, 0xA0, 0xBF),
        5 => cont(1, 0x80, 0x9F),
        6 => cont(2, 0x90, 0xBF),
        7 => cont(2, 0x80, 0x8F),
        _ => UTF8_REJECT,
    }
}

/// Checks loaded records against everything the read and mutation paths
/// assume of records they wrote themselves: every reachable record parses
/// within bounds (targets always point below their record, so traversal
/// ends); transition bytes strictly increase; each output is the count of
/// terms before its transition, so ranks and the counts derived top-down
/// are exact; no reachable state is empty; the root's count is
/// `term_count`; chain labels decode (valid codes) to at most
/// `MAX_CHAIN_STATES` bytes; and every term is valid UTF-8 (the premise of
/// `decoded_term`). Returns the total bytes of the terms, which must stay
/// within `BLOCK_TEXT_LIMIT` (so the write bounds of `crate::limits` hold).
fn validate(a: &PackedAutomaton) -> Result<u64, &'static str> {
    if a.bytes.len() as u64 > crate::limits::ADDRESS_LIMIT {
        return Err("records exceed the address limit");
    }
    if a.term_count == 0 {
        return Err("an automaton holds at least one term");
    }
    // Reachable records, parsed once: node index by address.
    let mut parsed = Parsed::default();
    let mut index: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut pending: Vec<(u32, u32)> = Vec::new(); // (edge index, target address)
    let mut stack = vec![a.root];
    index.insert(a.root, 0);
    let mut label = Vec::new();
    while let Some(pos) = stack.pop() {
        let (is_final, chain_target, children) = parse_checked(a, pos, &mut label)?;
        let edges_start = parsed.edges.len() as u32;
        let chain = match chain_target {
            Some(target) => {
                let start = parsed.labels.len() as u32;
                parsed.labels.extend_from_slice(&label);
                pending.push((parsed.edges.len() as u32, target));
                parsed.edges.push((0, 0, u32::MAX));
                Some((start, parsed.labels.len() as u32))
            }
            None => {
                if children.windows(2).any(|w| w[0].0 >= w[1].0) {
                    return Err("transition bytes not strictly increasing");
                }
                for (b, out, target) in children {
                    pending.push((parsed.edges.len() as u32, target));
                    parsed.edges.push((b, out, u32::MAX));
                }
                None
            }
        };
        parsed.nodes.push(ParsedNode {
            pos,
            is_final,
            chain,
            edges: edges_start..parsed.edges.len() as u32,
        });
        for &(edge, target) in &pending {
            let next = index.len() as u32;
            let node = *index.entry(target).or_insert_with(|| {
                stack.push(target);
                next
            });
            parsed.edges[edge as usize].2 = node;
        }
        pending.clear();
    }
    // Node indices follow first discovery; DFS discovery order differs
    // from `stack` pop order, so map discovery index -> parsed node.
    let mut by_index = vec![u32::MAX; index.len()];
    for (n, node) in parsed.nodes.iter().enumerate() {
        by_index[index[&node.pos] as usize] = n as u32;
    }
    for edge in &mut parsed.edges {
        edge.2 = by_index[edge.2 as usize];
    }
    let root = by_index[0] as usize;

    // Counts and text bottom-up: every target lies below its record, so
    // ascending address order visits targets first.
    let mut order: Vec<u32> = (0..parsed.nodes.len() as u32).collect();
    order.sort_unstable_by_key(|&n| parsed.nodes[n as usize].pos);
    let mut count = vec![0u64; parsed.nodes.len()];
    let mut text = vec![0u64; parsed.nodes.len()];
    for &n in &order {
        let node = &parsed.nodes[n as usize];
        let (mut c, mut t) = (u64::from(node.is_final), 0u64);
        for &(_, out, child) in &parsed.edges[node.edges.start as usize..node.edges.end as usize] {
            let child = child as usize;
            if parsed.nodes[child].pos >= node.pos {
                return Err("target not below its record");
            }
            let step = match node.chain {
                Some((ls, le)) => u64::from(le - ls),
                None => {
                    if u64::from(out) != c {
                        return Err("transition output differs from the terms before it");
                    }
                    1
                }
            };
            c += count[child];
            t = t.saturating_add(count[child].saturating_mul(step)).saturating_add(text[child]);
        }
        if c == 0 {
            return Err("reachable state with no terms");
        }
        if c > u64::from(u32::MAX) || t > crate::limits::BLOCK_TEXT_LIMIT {
            return Err("block exceeds the term count or text limit");
        }
        count[n as usize] = c;
        text[n as usize] = t;
    }
    if count[root] != u64::from(a.term_count) {
        return Err("root count differs from the term count");
    }

    // UTF-8: walk (node, decoder state) pairs once each.
    let mut seen = vec![0u16; parsed.nodes.len()];
    let mut walk = vec![(root as u32, UTF8_ACCEPT)];
    seen[root] = 1 << UTF8_ACCEPT;
    while let Some((n, state)) = walk.pop() {
        let node = &parsed.nodes[n as usize];
        if node.is_final && state != UTF8_ACCEPT {
            return Err("a term is not valid UTF-8");
        }
        for &(b, _, child) in &parsed.edges[node.edges.start as usize..node.edges.end as usize] {
            let next = match node.chain {
                Some((ls, le)) => parsed.labels[ls as usize..le as usize].iter().fold(state, |s, &b| utf8_step(s, b)),
                None => utf8_step(state, b),
            };
            if next == UTF8_REJECT {
                return Err("a term is not valid UTF-8");
            }
            let bit = 1u16 << next;
            if seen[child as usize] & bit == 0 {
                seen[child as usize] |= bit;
                walk.push((child, next));
            }
        }
    }
    Ok(text[root])
}

pub(crate) fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};
    use std::sync::Arc;

    fn sorted(mut terms: Vec<String>) -> Vec<String> {
        terms.sort_unstable();
        terms.dedup();
        terms
    }

    fn namespace_clustered_terms(start: u32, count: u32) -> Vec<String> {
        (start..start + count)
            .flat_map(|i| {
                [
                    format!("http://dbpedia.org/resource/Entity_{i}"),
                    format!("http://xmlns.com/foaf/0.1/property{i}"),
                    format!("urn:isbn:{i:013}"),
                    format!("_:b{i:x}"),
                ]
            })
            .collect()
    }

    /// Low-redundancy terms: hex digests behind a shared namespace.
    fn hashlike_terms(n: u64) -> Vec<String> {
        (0..n)
            .map(|i| {
                let mut x = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
                x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                format!("http://example.org/h/{:016x}", x ^ (x >> 31))
            })
            .collect()
    }

    /// Prefix relationships in both directions, the empty term, and a
    /// shared suffix across different stems.
    fn branching_terms() -> Vec<String> {
        ["", "a", "ab", "abc", "abd", "b", "ba", "xyz-tail", "uvw-tail", "zz", "z"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn build(terms: &[String], ws: &mut Workspace) -> PackedAutomaton {
        PackedAutomaton::build_sorted(terms.iter().map(String::as_bytes), ws)
    }

    /// States of the minimal acyclic automaton of `terms` (Myhill-Nerode):
    /// one per distinct right language `{s : p + s in terms}` over all
    /// prefixes `p`. Outputs (ranks) are a function of the right language,
    /// so this is also the minimal state count of the ranked automaton.
    fn minimal_state_count(terms: &[String]) -> usize {
        let set: BTreeSet<&[u8]> = terms.iter().map(|t| t.as_bytes()).collect();
        let mut languages: HashSet<Vec<&[u8]>> = HashSet::new();
        for t in &set {
            for cut in 0..=t.len() {
                let prefix = &t[..cut];
                let language: Vec<&[u8]> = set
                    .range(prefix..)
                    .take_while(|s| s.starts_with(prefix))
                    .map(|s| &s[cut..])
                    .collect();
                languages.insert(language);
            }
        }
        languages.len().max(1)
    }

    /// Every query answer against the sorted `terms`: rank, term_at_into,
    /// lower_bound (present and absent probes), the stream and a bulk
    /// decode of every rank.
    fn assert_answers(packed: &PackedAutomaton, terms: &[String]) {
        let mut buf = Vec::new();
        for (rank, term) in terms.iter().enumerate() {
            assert_eq!(packed.rank(term), Some(rank as u32), "rank of {term:?}");
            assert_eq!(packed.lower_bound(term.as_bytes()), (rank as u32, true));
            packed.term_at_into(rank as u32, &mut buf);
            assert_eq!(buf, term.as_bytes(), "term_at_into({rank})");
        }
        let probes = terms.iter().map(|t| format!("{t}~")).chain(["", "0", "h", "zzzz", "\u{7f}"].map(String::from));
        for probe in probes {
            let below = terms.partition_point(|t| t.as_str() < probe.as_str()) as u32;
            let present = terms.binary_search(&probe).is_ok();
            assert_eq!(packed.lower_bound(probe.as_bytes()), (below, present), "lower_bound({probe:?})");
            assert_eq!(packed.rank(&probe).is_some(), present);
        }
        // Batched descents over a sorted mix of present and absent terms
        // (prefixes, extensions, divergences) match one descent each.
        let mut queries: Vec<String> = terms.iter().flat_map(|t| [t.clone(), format!("{t}~"), format!("{t}\0")]).collect();
        queries.extend(terms.iter().map(|t| t.chars().take(t.chars().count() / 2).collect::<String>()));
        queries.extend(["", "0", "h", "zzzz"].map(String::from));
        let queries = sorted(queries);
        let mut batched = Vec::new();
        packed.lower_bounds_sorted(queries.iter().map(String::as_bytes), &mut Vec::new(), |rank, found| batched.push((rank, found)));
        let single: Vec<(u32, bool)> = queries.iter().map(|q| packed.lower_bound(q.as_bytes())).collect();
        assert_eq!(batched, single);
        let mut stream = packed.stream();
        for term in terms {
            assert_eq!(stream.next(), Some(term.as_bytes()));
        }
        assert_eq!(stream.next(), None);
        // Reversed positions: outputs land by position, not by rank.
        let targets: Vec<(usize, u32)> = (0..terms.len()).map(|r| (terms.len() - 1 - r, r as u32)).collect();
        let mut out = vec![None; terms.len()];
        packed.decode_ranks_into(&targets, &mut out);
        for (r, term) in terms.iter().enumerate() {
            assert_eq!(out[terms.len() - 1 - r].as_deref(), Some(term.as_str()));
        }
    }

    #[test]
    fn empty_and_single_term() {
        let mut ws = Workspace::default();
        let empty = build(&[], &mut ws);
        assert_eq!(empty.rank(""), None);
        assert_eq!(empty.lower_bound(b"a"), (0, false));
        assert_eq!(empty.stream().next(), None);
        let one = vec!["only".to_string()];
        assert_answers(&build(&one, &mut ws), &one);
        let blank = vec![String::new()];
        assert_answers(&build(&blank, &mut ws), &blank);
    }

    #[test]
    fn build_answers_queries_on_varied_term_shapes() {
        let mut ws = Workspace::default();
        for terms in [sorted(namespace_clustered_terms(0, 500)), sorted(hashlike_terms(2000)), sorted(branching_terms())] {
            let packed = build(&terms, &mut ws);
            assert_answers(&packed, &terms);
        }
    }

    /// `PACKED_COUNT_ESCAPE` transitions, long unique tails (NEXT runs) and
    /// targets far enough back to need multi-byte deltas, built every way:
    /// at once, in two sorted batches, one term at a time (descending),
    /// and compacted with raw and FSST-coded chains.
    #[test]
    fn wide_fanout_and_long_tails_every_construction_path() {
        let mut terms: Vec<String> = (1u8..=250)
            .map(|b| format!("{}{}", char::from_u32(u32::from(b)).unwrap_or('x'), "tail".repeat(usize::from(b % 40))))
            .collect();
        terms.extend((0..300).map(|i| format!("p/{i:05}/{}", "z".repeat(i % 97))));
        let terms = sorted(terms);
        let first: Vec<&str> = terms.iter().enumerate().filter(|(i, _)| i % 3 != 0).map(|(_, t)| t.as_str()).collect();
        let second: Vec<&str> = terms.iter().enumerate().filter(|(i, _)| i % 3 == 0).map(|(_, t)| t.as_str()).collect();
        let mut ws = Workspace::default();

        let built = build(&terms, &mut ws);
        let mut batched = PackedAutomaton::build_sorted(first.iter().map(|t| t.as_bytes()), &mut ws);
        batched.insert_sorted(second.iter().map(|t| t.as_bytes()), &mut ws);
        let mut single = build(&[], &mut ws);
        for t in terms.iter().rev() {
            single.insert(t, &mut ws);
        }
        let mut chained = built.clone();
        chained.compact_and_chain(&mut ws);
        let mut coded = built.clone();
        let sample: Vec<&[u8]> = terms.iter().map(|t| t.as_bytes()).collect();
        coded.set_symbols(Some(Arc::new(crate::fsst::SymbolTable::train(&sample))));
        coded.compact_and_chain(&mut ws);
        assert!(coded.chains_coded());
        for packed in [&built, &batched, &single, &chained, &coded] {
            assert_answers(packed, &terms);
        }
    }

    /// Re-inserting present terms changes nothing observable.
    #[test]
    fn insert_sorted_is_idempotent_and_mixes_hits_with_new_terms() {
        let mut ws = Workspace::default();
        let base = sorted(namespace_clustered_terms(0, 80));
        let mut packed = build(&base, &mut ws);
        let mut incoming: Vec<String> = base.iter().step_by(3).cloned().collect();
        incoming.extend((0..50).map(|i| format!("http://example.org/brand-new-{i:03}")));
        let incoming = sorted(incoming);
        packed.insert_sorted(incoming.iter().map(String::as_bytes), &mut ws);
        packed.insert_sorted(incoming.iter().map(String::as_bytes), &mut ws);
        let all = sorted(base.iter().chain(&incoming).cloned().collect());
        assert_answers(&packed, &all);
    }

    /// `compact` after incremental merges, removals and chaining reaches
    /// the exact minimal automaton (brute-force oracle), keeps every
    /// answer, and leaves the automaton mutable.
    #[test]
    fn compact_is_exactly_minimal() {
        let mut ws = Workspace::default();
        let terms = sorted(
            namespace_clustered_terms(0, 150)
                .into_iter()
                .chain((0..200).map(|i| format!("_:b{i}x{}", i * 7919 % 1000)))
                .chain(branching_terms())
                .collect(),
        );
        let fresh = build(&terms, &mut ws);
        assert!(fresh.reachable_state_count() >= minimal_state_count(&terms));
        let mut packed = build(&[], &mut ws);
        for round in 0..12 {
            let batch = terms.iter().enumerate().filter(|(i, _)| i % 12 == round).map(|(_, t)| t.as_bytes());
            packed.insert_sorted(batch.collect::<Vec<_>>().into_iter(), &mut ws);
        }
        packed.insert("zzz-removed-again", &mut ws);
        assert!(packed.remove("zzz-removed-again", &mut ws).is_some());
        let before = packed.size_bytes();
        packed.compact(&mut ws);
        assert!(packed.size_bytes() < before);
        assert!(packed.size_bytes() <= fresh.size_bytes());
        assert_eq!(packed.reachable_state_count(), minimal_state_count(&terms));
        // Chains hold one state per label byte: same count, same answers.
        packed.compact_and_chain(&mut ws);
        assert_eq!(packed.reachable_state_count(), minimal_state_count(&terms));
        assert_answers(&packed, &terms);
        packed.insert("zzz-after-compact", &mut ws);
        assert_eq!(packed.rank("zzz-after-compact"), Some(terms.len() as u32));
    }

    /// Random inserts and removals against a `BTreeSet` model, with plain
    /// and chaining compactions interleaved (mutations after a chaining
    /// compaction expand the chains on their path).
    #[test]
    fn churn_matches_model() {
        let pool: Vec<String> = (0..400)
            .map(|i| match i % 4 {
                0 => format!("http://ex.org/a/{}", i / 4),
                1 => format!("http://ex.org/a/{}x", i / 8),
                2 => format!("_:b{}", i % 37),
                _ => format!("\"{}\"@en", i * 31 % 97),
            })
            .collect();
        let mut ws = Workspace::default();
        let mut model = BTreeSet::<String>::new();
        let mut packed = build(&[], &mut ws);
        let sample: Vec<&[u8]> = pool.iter().map(|t| t.as_bytes()).collect();
        packed.set_symbols(Some(Arc::new(crate::fsst::SymbolTable::train(&sample))));
        let mut state = 0x9E37_79B9_u32;
        for step in 0..3000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let term = &pool[state as usize % pool.len()];
            if state.is_multiple_of(3) {
                let expected = model.range::<String, _>(..term.clone()).count() as u32;
                let removed = packed.remove(term, &mut ws);
                assert_eq!(removed, model.remove(term).then_some(expected), "step {step} remove {term}");
            } else {
                packed.insert(term, &mut ws);
                model.insert(term.clone());
            }
            match step % 97 {
                0 if step % 194 == 0 => packed.compact_and_chain(&mut ws),
                0 => packed.compact(&mut ws),
                _ => {}
            }
            if step % 100 == 0 || step == 2999 {
                assert_answers(&packed, &model.iter().cloned().collect::<Vec<_>>());
            }
        }
        packed.compact(&mut ws);
        assert_eq!(packed.reachable_state_count(), minimal_state_count(&model.into_iter().collect::<Vec<_>>()));
    }

    /// One workspace serves many automata: the registry is reset per
    /// operation, so nothing from one automaton leaks into another.
    #[test]
    fn shared_workspace_across_automata() {
        let mut ws = Workspace::default();
        let a_terms = sorted(namespace_clustered_terms(0, 300));
        let b_terms = sorted(hashlike_terms(300));
        let mut a = build(&a_terms, &mut ws);
        let mut b = build(&b_terms, &mut ws);
        let a_more = sorted(namespace_clustered_terms(300, 10));
        let b_more = sorted(hashlike_terms(310)[300..].to_vec());
        a.insert_sorted(a_more.iter().map(String::as_bytes), &mut ws);
        b.insert_sorted(b_more.iter().map(String::as_bytes), &mut ws);
        assert!(PackedAutomaton::register_len(&ws) <= 64 * REGISTRY_WAYS, "sized for a 10-term merge");
        assert_answers(&a, &sorted(a_terms.into_iter().chain(a_more).collect()));
        assert_answers(&b, &sorted(b_terms.into_iter().chain(b_more).collect()));
    }

    /// A term far longer than any stack frame budget: every path that
    /// walks it (build, lookups, stream, bulk decode, mutation,
    /// compaction, chaining) must be iterative, not recursive per byte.
    #[test]
    fn very_long_terms_do_not_recurse_per_byte() {
        let long: String = (0..400_000).map(|i| char::from(b'a' + (i * 7 % 26) as u8)).collect();
        let terms = sorted(vec![long.clone(), format!("{long}!"), "short".to_string()]);
        let mut ws = Workspace::default();
        let mut packed = build(&terms, &mut ws);
        assert_answers(&packed, &terms);
        packed.compact_and_chain(&mut ws);
        assert_answers(&packed, &terms);
        assert_eq!(packed.remove(&long, &mut ws), Some(0));
        packed.compact(&mut ws);
        assert_answers(&packed, &terms[1..]);
    }
}
