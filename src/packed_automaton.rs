//! `PackedAutomaton`: each block's term index -- a minimal acyclic
//! automaton over the block's sorted terms, stored as one append-only byte
//! buffer of fst-style records, with each state's outputs the ranks of its
//! terms (so rank and unrank are one descent). Built bottom-up from sorted
//! input (Daciuk, Mihov, Watson, Watson 2000), mutated in place by
//! re-confirming only the paths it touches, compacted to the exact minimal
//! automaton, and path-compressed with FSST-coded chain labels at the end
//! of a load. See the `PackedAutomaton` doc comment for the record layout,
//! and DESIGN.md for every measurement behind it.

use std::hash::{Hash, Hasher};

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
#[derive(Clone, Debug)]
struct PackedRegistry {
    /// `buckets * REGISTRY_WAYS` cells, most recently used first within a
    /// bucket; never resized.
    cells: Box<[RegistryCell]>,
    /// `buckets - 1`.
    mask: usize,
}

/// Cells per registry bucket (`fst`'s `mru_size` default).
const REGISTRY_WAYS: usize = 2;

#[derive(Clone, Copy, Debug, Default)]
struct RegistryCell {
    /// High half of the signature hash (the low bits pick the bucket).
    tag: u32,
    /// Record address + 1; 0 marks an empty cell, so a fresh table is all
    /// zeros (`alloc_zeroed`).
    target_plus_one: u32,
}

impl PackedRegistry {
    /// Sized from the number of terms a construction will confirm:
    /// `Block::build`'s `fst` sizing (`4 x terms` rows, at least 64),
    /// rounded up to a power of two, capped at 16,384 rows (256 KiB).
    fn with_capacity_hint(terms: usize) -> PackedRegistry {
        let buckets = terms.saturating_mul(4).clamp(64, 1 << 14).next_power_of_two();
        PackedRegistry {
            cells: vec![RegistryCell::default(); buckets * REGISTRY_WAYS].into_boxed_slice(),
            mask: buckets - 1,
        }
    }

    #[inline]
    fn bucket(&mut self, hash: u64) -> &mut [RegistryCell; REGISTRY_WAYS] {
        let start = (hash as usize & self.mask) * REGISTRY_WAYS;
        (&mut self.cells[start..start + REGISTRY_WAYS]).try_into().expect("bucket width")
    }

    /// Addresses in `hash`'s bucket whose tag matches, most recent first.
    #[inline]
    fn candidates(&mut self, hash: u64) -> [Option<u32>; REGISTRY_WAYS] {
        let tag = (hash >> 32) as u32;
        self.bucket(hash)
            .map(|c| (c.target_plus_one != 0 && c.tag == tag).then(|| c.target_plus_one - 1))
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
            target_plus_one: target + 1,
        };
    }

    /// Occupied cells -- always `<= cells.len()`. See
    /// `PackedAutomaton::register_len`.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.cells.iter().filter(|c| c.target_plus_one != 0).count()
    }
}


/// One state being incrementally (re)built by [`PackedAutomaton::insert`],
/// before it's confirmed and written -- the packed-representation
/// equivalent of the plain `State`/`Transitions` a matched-but-unconfirmed
/// state already was in the mutable slab design; kept in memory only for
/// the duration of one `insert` call, never itself part of the
/// persistent structure.
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
    bytes: Vec<u8>,
    root: u32,
    /// The root's subtree count (= number of terms). The only count
    /// stored anywhere: every other state's count is derived top-down
    /// while descending (see `StagingFrame::children`).
    term_count: u32,
    /// Construction-only state -- the dedup registry and reusable
    /// buffers -- boxed so a finished block (which has none) pays one
    /// pointer instead of ~90 inline bytes; see `Construction`. `None`
    /// after [`Self::release_construction_state`] or `compact`; recreated
    /// on demand ([`Self::cons`]).
    construction: Option<Box<Construction>>,
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

/// What only construction needs (`PackedAutomaton::construction`): the
/// dedup registry (`registry`; never read by any query -- long-lived
/// holders release it between mutations: kept resident, its fixed-size
/// table plus each cell's transitions `Vec` measured ~5x the automaton's
/// `bytes` on real RDF, DESIGN.md's "releasing PackedAutomaton's registry
/// between mutations"), and reusable buffers -- a per-confirm
/// child list and a pool of frame child lists, so the per-state work of
/// `insert_sorted`/`build_sorted`/`compact` doesn't allocate (malloc/free
/// was ~10% of a bulk load's profile).
#[derive(Clone, Debug, Default)]
struct Construction {
    register: Option<PackedRegistry>,
    scratch: Scratch,
}

#[derive(Clone, Debug, Default)]
struct Scratch {
    children: Vec<(u8, u32, u32)>,
    frames: Vec<Vec<(u8, u32, u32)>>,
}

#[allow(dead_code)]
impl PackedAutomaton {
    /// Builds a minimal packed automaton directly from an already sorted,
    /// deduplicated sequence of terms -- `Automaton::build_sorted`'s exact
    /// algorithm (Daciuk/Mihov/Watson/Watson 2000's sorted-input
    /// specialization: track the rightmost path as a stack of unconfirmed
    /// frames, confirm everything deeper than the common-prefix length
    /// whenever it's known a word won't extend that far again), targeting
    /// this byte-packed representation instead of the `State`/`Transitions`
    /// slab. Not a wrapper around `compile(&Automaton::build_sorted(..))`:
    /// that route pays for a full slab construction (every `State`,
    /// `TransitionChunk`, the slab's own register) purely to throw it away
    /// once compiled, real, measured waste for a structure never queried
    /// in slab form at all. This confirms and writes packed records
    /// directly, reusing the exact same per-frame machinery `insert` already
    /// has (`StagingFrame`, `confirm_frame`, `write_confirmed_state`) --
    /// sorted construction only changes *how* frames are opened and closed,
    /// never how a single frame gets confirmed once ready.
    ///
    /// One simplification sorted input allows that `insert`'s general
    /// `StagingFrame::children` splicing can't: a diverging byte at any
    /// depth is always lexicographically greater than every sibling
    /// already recorded there (the same guarantee `Automaton::
    /// build_sorted`'s own `link_new_appending` relies on, vs. `insert`'s
    /// `binary_search_by_key`), so a new child is always a plain `push`,
    /// never a sorted insert.
    pub(crate) fn build_sorted<'a>(
        terms: impl ExactSizeIterator<Item = &'a str>,
    ) -> PackedAutomaton {
        // See `compile`'s own comment on this capacity hint.
        let capacity_hint = terms.len();
        let mut result = PackedAutomaton {
            bytes: Vec::new(),
            root: 0,
            term_count: 0,
            construction: Some(Box::new(Construction {
                register: Some(PackedRegistry::with_capacity_hint(capacity_hint)),
                scratch: Scratch::default(),
            })),
            has_chains: false,
            pristine: false,
            symbols: None,
            chains_coded: false,
        };
        // frames[i] is the state reached after `path_bytes[..i]` of
        // `last_word`; frames[0] (root) has no incoming byte, so
        // `path_bytes[i - 1]` is the byte that reaches `frames[i]`.
        let mut frames: Vec<StagingFrame> = vec![StagingFrame {
            is_final: false,
            children: Vec::new(),
        }];
        let mut path_bytes: Vec<u8> = Vec::new();
        let mut last_word: Vec<u8> = Vec::new();

        for term in terms {
            let word = term.as_bytes();
            let common = common_prefix_len(&last_word, word);

            Self::confirm_down_to(&mut result, &mut frames, &mut path_bytes, common + 1);

            for &b in &word[common..] {
                frames.push(result.new_frame());
                path_bytes.push(b);
            }
            frames
                .last_mut()
                .expect("at least the root frame is present")
                .is_final = true;

            last_word.clear();
            last_word.extend_from_slice(word);
        }

        Self::confirm_down_to(&mut result, &mut frames, &mut path_bytes, 0);
        result
    }

    /// Pops and confirms every frame deeper than `keep_from` (i.e. until
    /// `frames.len() == keep_from.max(1)`), splicing each one's resolved
    /// `(byte, _, pos)` into its parent -- `Automaton::confirm_path`'s
    /// exact shape, over `StagingFrame`s instead of slab `State`s.
    /// `keep_from == 0` additionally confirms the root frame itself
    /// (which has no parent to splice into) and sets `result.root`.
    ///
    /// Splices via `binary_search_by_key` (update-in-place on a hit,
    /// sorted insert on a miss) rather than a plain `push`: safe for both
    /// callers. `build_sorted`'s own frames are always freshly created,
    /// so a diverging byte there is always greater than every sibling
    /// already recorded (a `push` would've been equivalent, just without
    /// the search) -- but `insert_sorted` (below) reuses this same
    /// helper to re-confirm frames *opened from pre-existing structure*,
    /// where the byte being spliced back may already have an entry (an
    /// existing transition whose target changed, not a new one), and a
    /// blind `push` there would silently create a duplicate.
    fn confirm_down_to(
        result: &mut PackedAutomaton,
        frames: &mut Vec<StagingFrame>,
        path_bytes: &mut Vec<u8>,
        keep_from: usize,
    ) {
        while frames.len() > keep_from.max(1) {
            let frame = frames.pop().expect("loop guard ensures at least one entry");
            let byte = path_bytes.pop().expect("one byte per non-root frame");
            let (pos, count) = result.confirm_frame(&frame);
            result.recycle_frame(frame);

            let parent = frames
                .last_mut()
                .expect("path always has at least the root");
            match parent.children.binary_search_by_key(&byte, |&(b, _, _)| b) {
                Ok(idx) => parent.children[idx] = (byte, count, pos),
                Err(idx) => parent.children.insert(idx, (byte, count, pos)),
            }
        }
        if keep_from == 0 {
            let root_frame = frames.pop().expect("root frame always present");
            (result.root, result.term_count) = result.confirm_frame(&root_frame);
            result.recycle_frame(root_frame);
        }
    }

    /// Incrementally inserts an already sorted, deduplicated sequence of
    /// terms into this (already populated) automaton -- `Automaton::
    /// insert_sorted`'s exact algorithm (a persistent rightmost-path
    /// stack, carried *across* the whole batch; between consecutive
    /// terms, only the frames deeper than their shared prefix get
    /// re-walked and re-confirmed) targeting the packed representation,
    /// the same way `build_sorted` targets it for from-scratch
    /// construction. This is the fix for a real, measured problem: the
    /// naive alternative -- calling `insert` once per incoming term --
    /// has no memory between calls, so it re-walks and re-confirms a
    /// shared prefix once *per term* instead of once *per batch*,
    /// exactly backwards for redundant content (see DESIGN.md's
    /// "the real cause" section: this is why more register hits made
    /// that naive loop *worse*, not better).
    ///
    /// No clone-if-shared/generation bookkeeping is needed here, unlike
    /// `Automaton::insert_sorted`'s own: every frame on the persistent
    /// path is already a private, in-memory `StagingFrame` (opened via
    /// `read_children`, or freshly pushed), never a live reference into
    /// `bytes` that other content could alias -- the same reason
    /// `insert` above needs none either. Callers that already know which
    /// terms are new (e.g. `PackedAutomatonBlock::merge_into`'s own
    /// sorted-merge walk) don't need this to report it back, so unlike
    /// the slab's own `insert_sorted` this returns nothing.
    pub(crate) fn insert_sorted<'a>(&mut self, terms: impl Iterator<Item = &'a str>) {
        self.pristine = false;
        let mut frames: Vec<StagingFrame> = vec![self.open_frame(self.root, self.term_count)];
        let mut path_bytes: Vec<u8> = Vec::new();
        let mut last_word: Vec<u8> = Vec::new();

        for term in terms {
            let word = term.as_bytes();
            let common = common_prefix_len(&last_word, word);

            Self::confirm_down_to(self, &mut frames, &mut path_bytes, common + 1);

            // Continue matching from where the shared-prefix-with-the-
            // previous-word left off: does more of `word` already exist
            // as a transition at the current (possibly just-truncated)
            // frame? `frames.last()`'s children always reflect this
            // batch's current view (either freshly read, or already
            // updated by an earlier iteration's `confirm_down_to`).
            let mut matched = common;
            while matched < word.len() {
                let byte = word[matched];
                let cur = frames.last().expect("frames always has at least the root");
                match cur.children.binary_search_by_key(&byte, |&(b, _, _)| b) {
                    Ok(idx) => {
                        let (_, child_count, target) = cur.children[idx];
                        frames.push(self.open_frame(target, child_count));
                        path_bytes.push(byte);
                        matched += 1;
                    }
                    Err(_) => break,
                }
            }

            let already_present = matched == word.len()
                && frames
                    .last()
                    .expect("at least the root frame is present")
                    .is_final;
            if !already_present {
                for &b in &word[matched..] {
                    frames.push(self.new_frame());
                    path_bytes.push(b);
                }
                frames
                    .last_mut()
                    .expect("at least one frame present")
                    .is_final = true;
            }

            last_word.clear();
            last_word.extend_from_slice(word);
        }

        Self::confirm_down_to(self, &mut frames, &mut path_bytes, 0);
    }

    /// Inserts `term` if not already present -- mirrors `Automaton::
    /// insert`'s own algorithm (walk existing transitions as far as
    /// possible, copy the matched-but-not-yet-reconfirmed states into an
    /// editable form, extend with brand-new states for the remaining
    /// suffix, then confirm everything bottom-up) but targets this
    /// append-only block structure instead of a `states: Vec<State>`
    /// slab -- see the struct doc comment for why that needs no
    /// reference-counting or redirect-into-an-existing-record step at
    /// all, unlike the mutable slab design's own `insert`.
    pub(crate) fn insert(&mut self, term: &str) {
        self.pristine = false;
        let word = term.as_bytes();

        // Walk existing transitions as far as they match, "opening" every
        // matched state into an editable frame holding its existing
        // children (read straight from its still-unmodified packed
        // record) -- opened while descending, since each child's subtree
        // count is derived from its parent's (`open_frame`).
        let mut staging: Vec<StagingFrame> = vec![self.open_frame(self.root, self.term_count)];
        let mut path_bytes: Vec<u8> = Vec::new();
        let mut matched = 0usize;
        while matched < word.len() {
            let cur = staging.last().expect("staging always has at least the root");
            match cur.children.binary_search_by_key(&word[matched], |&(b, _, _)| b) {
                Ok(idx) => {
                    let (_, child_count, target) = cur.children[idx];
                    let frame = self.open_frame(target, child_count);
                    staging.push(frame);
                    path_bytes.push(word[matched]);
                    matched += 1;
                }
                Err(_) => break,
            }
        }

        if matched == word.len() && staging.last().expect("non-empty staging").is_final {
            return; // already present
        }

        // Extend with brand-new states for the unmatched suffix.
        for &b in &word[matched..] {
            staging.push(StagingFrame {
                is_final: false,
                children: Vec::new(),
            });
            path_bytes.push(b);
        }
        staging
            .last_mut()
            .expect("at least the root frame is present")
            .is_final = true;

        // Confirm bottom-up: write each frame (or reuse an existing
        // equivalent record via the register), then splice its resolved
        // `(byte, count, pos)` into its parent's own children.
        while staging.len() > 1 {
            let frame = staging.pop().expect("staging.len() > 1 just checked");
            let byte = path_bytes.pop().expect("one byte per non-root frame");
            let (pos, count) = self.confirm_frame(&frame);

            let parent = staging.last_mut().expect("staging.len() > 1 just checked");
            match parent.children.binary_search_by_key(&byte, |&(b, _, _)| b) {
                Ok(idx) => parent.children[idx] = (byte, count, pos),
                Err(idx) => parent.children.insert(idx, (byte, count, pos)),
            }
        }

        let root_frame = staging.pop().expect("root frame always present");
        (self.root, self.term_count) = self.confirm_frame(&root_frame);
    }

    /// Removes `term` if present, returning whether it was -- `insert`'s
    /// mirror: open the path's frames (counts derived top-down), clear the
    /// final flag, prune states left with no terms, and re-confirm the
    /// path bottom-up. Superseded records become dead bytes, reclaimed by
    /// [`Self::compact`] like `insert`'s.
    pub(crate) fn remove(&mut self, term: &str) -> bool {
        self.pristine = false;
        let word = term.as_bytes();
        let mut staging: Vec<StagingFrame> = vec![self.open_frame(self.root, self.term_count)];
        for &byte in word {
            let cur = staging.last().expect("staging always has at least the root");
            let Ok(idx) = cur.children.binary_search_by_key(&byte, |&(b, _, _)| b) else {
                return false;
            };
            let (_, child_count, target) = cur.children[idx];
            let frame = self.open_frame(target, child_count);
            staging.push(frame);
        }
        let last = staging.last_mut().expect("staging always has at least the root");
        if !last.is_final {
            return false;
        }
        last.is_final = false;

        for &byte in word.iter().rev() {
            let frame = staging.pop().expect("one frame per byte plus the root");
            let resolved = if !frame.is_final && frame.children.is_empty() {
                None
            } else {
                Some(self.confirm_frame(&frame))
            };
            let parent = staging.last_mut().expect("the root is never popped here");
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
        let root_frame = staging.pop().expect("root frame always present");
        (self.root, self.term_count) = self.confirm_frame(&root_frame);
        true
    }

    /// `(number of terms sorting before term, whether term is present)` --
    /// one descent, with each state's subtree count derived top-down from
    /// `term_count` and the outputs, as in `open_frame`.
    pub(crate) fn lower_bound(&self, term: &str) -> (u32, bool) {
        // Chains are rare (only after `compact_and_chain`), so they get
        // their own loop: handling them inline made the compiler spill
        // `value`/`count` in this hot loop (+60% of its time during loads).
        if self.has_chains {
            return self.lower_bound_chained(term);
        }
        let mut pos = self.root;
        let mut count = self.term_count;
        let mut value = 0u32;
        for &byte in term.as_bytes() {
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
        }
        (value, self.is_final(pos))
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
    fn lower_bound_chained(&self, term: &str) -> (u32, bool) {
        let t = term.as_bytes();
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

    /// Computes `frame`'s outputs from its children's subtree counts
    /// (the prefix-sum-of-sibling-subtree-counts technique
    /// `finalize_outputs_and_count` uses) and writes (or reuses, via the
    /// register) the confirmed record. Returns `(address, subtree_count)`.
    fn confirm_frame(&mut self, frame: &StagingFrame) -> (u32, u32) {
        let mut running = u32::from(frame.is_final);
        let mut children = std::mem::take(&mut self.cons().scratch.children);
        children.clear();
        for &(b, child_count, target) in &frame.children {
            children.push((b, running, target));
            running += child_count;
        }
        let pos = self.write_confirmed_state(frame.is_final, &children);
        self.cons().scratch.children = children;
        (pos, running)
    }

    /// Opens the record at `pos` (whose subtree count is `count`) into an
    /// editable frame. Child counts are the differences between
    /// consecutive outputs; the last child gets the rest of `count`.
    fn open_frame(&mut self, pos: u32, count: u32) -> StagingFrame {
        // A mutation path entering a chain expands just that chain.
        let pos = match self.chain_at(pos) {
            Some(chain) => self.expand_chain(chain),
            None => pos,
        };
        let mut children = self.cons().scratch.frames.pop().unwrap_or_default();
        children.clear();
        let (is_final, n, mode, mut cursor) = self.header(pos);
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
        StagingFrame { is_final, children }
    }

    /// Writes `chain`'s states as single-transition records (last state
    /// first, so each points at the record just below it: NEXT) and
    /// returns the first one's address. The chain record itself becomes
    /// dead bytes, reclaimed by the next compaction. O(chain length): the
    /// per-chain alternative to expanding the whole block (`unchain`).
    fn expand_chain(&mut self, chain: Chain) -> u32 {
        let mut label = Vec::new();
        self.chain_label(chain, &mut label);
        let mut next = chain.target;
        for &b in label.iter().rev() {
            next = self.append_record(false, &[(b, 0, next)]);
        }
        next
    }

    /// Construction state, created on first use.
    #[inline]
    fn cons(&mut self) -> &mut Construction {
        self.construction.get_or_insert_with(Default::default)
    }

    /// An empty frame whose child list comes from the scratch pool.
    #[inline]
    fn new_frame(&mut self) -> StagingFrame {
        let mut children = self.cons().scratch.frames.pop().unwrap_or_default();
        children.clear();
        StagingFrame {
            is_final: false,
            children,
        }
    }

    /// Returns a confirmed frame's child list to the scratch pool.
    #[inline]
    fn recycle_frame(&mut self, frame: StagingFrame) {
        self.cons().scratch.frames.push(frame.children);
    }

    /// Returns the address of a record with this signature: an equivalent
    /// one the registry remembers (confirmed against the written bytes),
    /// else a newly appended one.
    fn write_confirmed_state(&mut self, is_final: bool, children: &[(u8, u32, u32)]) -> u32 {
        let hash = signature_hash(is_final, children);
        let candidates = self.registry().candidates(hash);
        for (way, candidate) in candidates.into_iter().enumerate() {
            if let Some(pos) = candidate
                && self.record_matches(pos, is_final, children)
            {
                self.registry().promote(hash, way);
                return pos;
            }
        }
        let pos = self.append_record(is_final, children);
        self.registry().insert(hash, pos);
        pos
    }

    /// The live registry, recreating it if released. Sized from
    /// `term_count`, the same hint
    /// `build_sorted` uses. A recreated registry starts empty, so records
    /// written before the release can't be deduplicated against -- the
    /// same kind of lost sharing the bounded LRU's eviction already
    /// accepts (measured: +4% `bytes` on real RDF, DESIGN.md).
    #[inline]
    fn registry(&mut self) -> &mut PackedRegistry {
        let hint = self.term_count as usize;
        let construction = self.cons();
        if construction.register.is_none() {
            construction.register = Some(PackedRegistry::with_capacity_hint(hint));
        }
        construction.register.as_mut().expect("just ensured")
    }

    /// Recreates a released registry sized for a mutation touching
    /// `terms` terms, instead of `registry`'s whole-automaton default. A
    /// recreated registry can only dedup records written during this
    /// mutation, so sizing it to the automaton just allocates and zeroes
    /// cells nothing will use -- measured as most of a single-term insert's
    /// cost in a 1024-term block (DESIGN.md's "incremental single-term
    /// mutation").
    pub(crate) fn reserve_registry(&mut self, terms: usize) {
        let construction = self.cons();
        if construction.register.is_none() {
            construction.register = Some(PackedRegistry::with_capacity_hint(terms));
        }
    }

    /// Drops everything only construction needs: frees the registry and
    /// trims `bytes`' spare growth capacity, so a resident automaton's
    /// heap matches `size_bytes`. Further `insert`/`insert_sorted` calls
    /// remain valid (see [`Self::registry`]).
    pub(crate) fn release_construction_state(&mut self) {
        self.construction = None;
        self.bytes.shrink_to_fit();
    }

    /// Rewrites `bytes` to hold only reachable records, merging records
    /// that become identical once their children are rewritten. Dead
    /// records (left behind by `insert`/`insert_sorted`, see the struct
    /// doc comment) are dropped, and the exact dedup is bottom-up
    /// minimization (Revuz 1992): outputs are a function of each state's
    /// right language, so equivalent states have equal signatures once
    /// their children are canonical, and the result is the minimal
    /// automaton -- tighter than `build_sorted`'s bounded registry. The
    /// dedup map is temporary; the registry is released, so this is
    /// O(reachable records) time with no retained construction state.
    /// Records are re-emitted in postorder, children in byte order, as
    /// `build_sorted` writes them, so NEXT targets apply.
    pub(crate) fn compact(&mut self) {
        if self.pristine {
            // Already compacted, and nothing mutated since.
            return;
        }
        // Minimization runs on per-byte states: expand remaining chains.
        self.unchain();
        let old = PackedAutomaton {
            bytes: std::mem::take(&mut self.bytes),
            root: self.root,
            term_count: self.term_count,
            construction: None,
            // A read-only copy of `self`'s records: same chain state.
            has_chains: self.has_chains,
            pristine: false,
            symbols: self.symbols.clone(),
            chains_coded: false,
        };
        let mut moved = vec![u32::MAX; old.bytes.len()];
        let mut canonical = DedupTable::new();
        let mut pool = self.construction.take().map(|c| c.scratch.frames).unwrap_or_default();
        let read = |pool: &mut Vec<Vec<(u8, u32, u32)>>, pos: u32| {
            let mut v = pool.pop().unwrap_or_default();
            v.clear();
            old.read_children_into(pos, &mut v);
            v
        };
        // (old address, its children, next child to visit)
        let root_children = read(&mut pool, old.root);
        let mut stack: Vec<(u32, Vec<(u8, u32, u32)>, usize)> = vec![(old.root, root_children, 0)];
        while let Some((pos, children, next)) = stack.last_mut() {
            if let Some(&(_, _, child)) = children.get(*next) {
                *next += 1;
                if moved[child as usize] == u32::MAX {
                    let grandchildren = read(&mut pool, child);
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
            let new_pos = match canonical.find(hash, |candidate| self.record_matches(candidate, is_final, &children)) {
                Some(existing) => existing,
                None => {
                    let new_pos = self.append_record(is_final, &children);
                    canonical.insert(hash, new_pos);
                    new_pos
                }
            };
            moved[pos as usize] = new_pos;
            pool.push(children);
        }
        self.root = moved[old.root as usize];
        self.bytes.shrink_to_fit();
        self.construction = None;
    }

    /// `compact`, then path compression (`chain_up`) for read speed. Meant
    /// for the end of a load: the next mutation expands the chains again
    /// (`unchain`), so chaining during a load -- e.g. on every
    /// growth-triggered compaction -- would only churn (measured: +20%
    /// insertion at 10M lines).
    pub(crate) fn compact_and_chain(&mut self) {
        if self.pristine && (self.chains_coded || self.symbols.is_none()) {
            return;
        }
        // Not pristine, or chained before a symbol table existed: minimize
        // per-byte again, then chain (with the table).
        self.pristine = false;
        self.compact();
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
        let mut stack: Vec<(u32, Vec<(u8, u32, u32)>, usize)> = vec![(self.root, root_edges, 0)];
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
        let old = PackedAutomaton {
            bytes: std::mem::take(&mut self.bytes),
            root: self.root,
            term_count: self.term_count,
            construction: None,
            // A read-only copy of `self`'s records: same chain state.
            has_chains: self.has_chains,
            pristine: false,
            symbols: self.symbols.clone(),
            chains_coded: false,
        };
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
                    moved[p] = if label.len() == 1 {
                        self.append_record(false, &[(label[0], 0, target)])
                    } else {
                        coded.clear();
                        if let Some(table) = &symbols {
                            table.encode(&label, &mut coded);
                        }
                        if !coded.is_empty() && coded.len() < label.len() {
                            self.append_chain(&coded, target, true)
                        } else {
                            self.append_chain(&label, target, false)
                        }
                    };
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
        let old = PackedAutomaton {
            bytes: std::mem::take(&mut self.bytes),
            root: self.root,
            term_count: self.term_count,
            construction: None,
            // A read-only copy of `self`'s records: same chain state.
            has_chains: self.has_chains,
            pristine: false,
            symbols: self.symbols.clone(),
            chains_coded: false,
        };
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
                let field = self.bytes.len() as u32;
                write_leb128_rev(&mut self.bytes, field - target);
            }
            let code = label_table().code(byte);
            if code == 0 {
                self.bytes.push(byte);
            }
            self.bytes.push(0x80 | (u8::from(next) << 6) | code);
            return (self.bytes.len() - 1) as u32;
        }
        // General form: `0 F nnnnnn`, transitions written in reverse so
        // the record reads forward from its header downward.
        for (i, &(b, out, target)) in children.iter().enumerate().rev() {
            let field = self.bytes.len() as u32;
            write_leb128_rev(&mut self.bytes, field - target);
            if i > 0 {
                write_leb128_rev(&mut self.bytes, out);
            }
            self.bytes.push(b);
        }
        if children.len() >= PACKED_COUNT_ESCAPE {
            self.bytes.push((children.len() - PACKED_COUNT_ESCAPE) as u8);
        }
        self.bytes
            .push((u8::from(is_final) << 6) | children.len().min(PACKED_COUNT_ESCAPE) as u8);
        (self.bytes.len() - 1) as u32
    }

    /// Total size of `bytes` -- directly comparable to `Automaton::
    /// estimate_packed_bytes`'s prediction for the same content, though
    /// only exactly equal right after a fresh `compile` (see the struct
    /// doc comment: `insert` can leave dead, unreachable bytes behind,
    /// which this still counts).
    pub(crate) fn size_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// The number of entries in `register` -- unlike `Automaton::
    /// register_len`, this has no pruning mechanism to measure the
    /// bound of: `write_confirmed_state` only ever inserts, never
    /// removes, so this tracks every distinct signature ever confirmed
    /// across this automaton's whole mutation history, live or dead.
    /// Compare directly against `reachable_state_count`. Measured in
    /// `measure_packed_automaton_register_growth_under_sustained_inserts`
    /// (`DESIGN.md`'s "measuring the register-pruning gap directly") --
    /// grows unboundedly (91 -> 62,632 over 2000 inserts) while
    /// `reachable_state_count` stays flat.
    #[cfg(test)]
    pub(crate) fn register_len(&self) -> usize {
        self.construction
            .as_ref()
            .and_then(|c| c.register.as_ref())
            .map_or(0, PackedRegistry::len)
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

    /// Mirrors `Automaton::decode_ranks_into`'s exact contract and
    /// algorithm -- see its own doc comment for the full reasoning (one
    /// recursive descent, partitioning the sorted target set against
    /// each state's transitions, visiting a state only if some target
    /// actually falls in its range). `targets` stays one contiguous
    /// slice of the original input throughout, `value_offset` carried
    /// alongside instead of shifting it -- identical to the slab
    /// version's own approach, no per-call allocation.
    pub(crate) fn decode_ranks_into(&self, targets: &[(usize, u32)], out: &mut [Option<String>]) {
        let mut buf = Vec::new();
        self.decode_batch(self.root, targets, 0, &mut buf, out);
    }

    /// The one adaptation this needs that the slab version doesn't:
    /// `Transitions::at(idx)` re-derives slot `idx` in O(1) (a small
    /// typed array), cheap to call for both the "current" transition and
    /// a one-ahead peek at "the next one" (for the boundary). A packed
    /// record has no such re-entry point -- peeking transition `idx + 1`
    /// means parsing forward from wherever `idx` ended, so `next` caches
    /// that peek across loop iterations (parsed once, when it's peeked;
    /// reused as `current` next iteration) instead of re-parsing it,
    /// keeping this one sequential pass over the record either way, the
    /// same reasoning `PackedTermsStream`'s own cursor caching uses.
    fn decode_batch(
        &self,
        pos: u32,
        targets: &[(usize, u32)],
        value_offset: u32,
        buf: &mut Vec<u8>,
        out: &mut [Option<String>],
    ) {
        if targets.is_empty() {
            return;
        }
        if let Some(chain) = self.chain_at(pos) {
            let len = self.chain_label(chain, buf);
            self.decode_batch(chain.target, targets, value_offset, buf, out);
            buf.truncate(buf.len() - len);
            return;
        }
        let (is_final, count, next_flag, mut cursor) = self.header(pos);
        let mut remaining = targets;
        if is_final {
            // Every leading target still at value_offset (i.e. local
            // value 0) is this state's own term -- sorted ascending, so
            // these are always a prefix of `remaining`.
            let split = remaining.partition_point(|&(_, v)| v == value_offset);
            if split > 0 {
                let term =
                    std::str::from_utf8(buf).expect("keys were inserted as valid UTF-8 str bytes");
                for &(p, _) in &remaining[..split] {
                    out[p] = Some(term.to_string());
                }
                remaining = &remaining[split..];
            }
        }

        let mut idx = 0usize;
        let mut next = (idx < count).then(|| {
            let (b, o, t, next_cursor) = self.read_transition_at(cursor, idx == 0, is_final, next_flag);
            cursor = next_cursor;
            (b, o, t)
        });
        while !remaining.is_empty() {
            let (byte, my_out, my_target) =
                next.expect("value must be reachable via some transition at this state");
            idx += 1;
            next = (idx < count).then(|| {
                let (b, o, t, next_cursor) = self.read_transition_at(cursor, idx == 0, is_final, next_flag);
                cursor = next_cursor;
                (b, o, t)
            });
            // Everything before the *next* transition's own output
            // belongs to this one -- `u32::MAX` if there is no next
            // transition, since the last one covers the rest by the
            // same invariant `last_output_at_most` relies on.
            let boundary = next.map_or(u32::MAX, |(_, next_out, _)| value_offset + next_out);
            let split = remaining.partition_point(|&(_, v)| v < boundary);
            let (mine, rest) = remaining.split_at(split);
            if !mine.is_empty() {
                buf.push(byte);
                self.decode_batch(my_target, mine, value_offset + my_out, buf, out);
                buf.pop();
            }
            remaining = rest;
        }
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
                let field = self.bytes.len() as u32;
                write_leb128_rev(&mut self.bytes, field - target);
            }
            self.bytes.extend_from_slice(label);
        }
        let len = u32::try_from(label.len()).expect("chain label fits u32");
        write_leb128_rev(&mut self.bytes, len << 2 | u32::from(coded) << 1 | u32::from(next));
        self.bytes.push(0);
        (self.bytes.len() - 1) as u32
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

fn signature_hash(is_final: bool, children: &[(u8, u32, u32)]) -> u64 {
    let mut hasher = FxHasher::default();
    is_final.hash(&mut hasher);
    children.hash(&mut hasher);
    hasher.finish()
}

/// `compact`'s exact dedup: open addressing over `(signature hash, new
/// record address)`. A probe hit is confirmed against the record already
/// written (`record_matches`), so no signature is stored -- the previous
/// `HashMap<(bool, Vec<..>), u32>` allocated a `Vec` key per record.
struct DedupTable {
    slots: Vec<(u64, u32)>,
    len: usize,
}

impl DedupTable {
    const EMPTY: u32 = u32::MAX;

    fn new() -> DedupTable {
        DedupTable {
            slots: vec![(0, Self::EMPTY); 256],
            len: 0,
        }
    }

    fn find(&self, hash: u64, mut matches: impl FnMut(u32) -> bool) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        loop {
            let (h, pos) = self.slots[i];
            if pos == Self::EMPTY {
                return None;
            }
            if h == hash && matches(pos) {
                return Some(pos);
            }
            i = (i + 1) & mask;
        }
    }

    fn insert(&mut self, hash: u64, pos: u32) {
        if (self.len + 1) * 2 > self.slots.len() {
            let grown = vec![(0, Self::EMPTY); self.slots.len() * 2];
            let old = std::mem::replace(&mut self.slots, grown);
            self.len = 0;
            for (h, p) in old {
                if p != Self::EMPTY {
                    self.insert(h, p);
                }
            }
        }
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        while self.slots[i].1 != Self::EMPTY {
            i = (i + 1) & mask;
        }
        self.slots[i] = (hash, pos);
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

pub(crate) fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}
