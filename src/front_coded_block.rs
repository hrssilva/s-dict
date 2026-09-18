use crate::TermId;

/// Every `RESTART_INTERVAL`-th entry is stored in full (no prefix sharing
/// with its predecessor) rather than delta-coded, bounding random-access
/// decode cost to O(RESTART_INTERVAL) instead of O(block size) -- the
/// "hierarchical front coding" the design doc cites (Brisaboa et al.,
/// *Improved Compressed String Dictionaries*) as the standard technique
/// for URI/RDF dictionaries. A single fixed constant, not yet tuned
/// (design doc §11: measure before introducing another adaptive knob).
const RESTART_INTERVAL: usize = 16;

/// The front-coded alternative to [`crate::Block`]'s micro-FST (design doc
/// §8/§13 "front-coded blocks" baseline): each sorted term is stored as a
/// delta against its lexicographic predecessor, `(shared_prefix_len,
/// suffix_bytes)`, except at restart points. This captures the same
/// shared-*prefix* redundancy a trie/automaton does, but not shared
/// *suffixes*/equivalent right-languages the way a minimized automaton
/// does -- the gap this baseline exists to measure.
///
/// Layout is plain parallel arrays (`Vec<u32>` offsets/lengths, one
/// concatenated `Vec<u8>` suffix buffer), not a bit-packed format --
/// design doc §11's "first implementation... even where tighter
/// representations are obvious."
#[derive(Debug)]
pub(crate) struct FrontCodedBlock {
    /// Concatenated suffix bytes for every entry (a restart entry's
    /// "suffix" is its full string).
    suffixes: Vec<u8>,
    /// CSR-style offsets into `suffixes`: entry `i`'s suffix is
    /// `suffixes[suffix_offsets[i]..suffix_offsets[i + 1]]`. Length is
    /// `ids.len() + 1`.
    suffix_offsets: Vec<u32>,
    /// `prefix_lens[i]` = bytes shared with entry `i - 1` (always 0 at a
    /// restart point).
    prefix_lens: Vec<u32>,
    ids: Vec<TermId>,
    min_key: Box<str>,
}

/// Byte-wise common prefix length, walked back to the nearest UTF-8 char
/// boundary. Bytes `0..raw_len` are identical between `a` and `b` by
/// construction, so a char-boundary check against either string gives the
/// same answer for any index `<= raw_len`.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut len = a
        .as_bytes()
        .iter()
        .zip(b.as_bytes())
        .take_while(|(x, y)| x == y)
        .count();
    while len > 0 && !a.is_char_boundary(len) {
        len -= 1;
    }
    len
}

impl FrontCodedBlock {
    /// Builds an immutable block from already sorted, deduplicated
    /// `(term, id)` pairs.
    pub(crate) fn build(sorted_entries: &[(Box<str>, TermId)]) -> FrontCodedBlock {
        let mut suffixes = Vec::new();
        let mut suffix_offsets = Vec::with_capacity(sorted_entries.len() + 1);
        let mut prefix_lens = Vec::with_capacity(sorted_entries.len());
        let mut ids = Vec::with_capacity(sorted_entries.len());
        suffix_offsets.push(0u32);

        let mut prev: &str = "";
        for (i, (term, id)) in sorted_entries.iter().enumerate() {
            let prefix_len = if i % RESTART_INTERVAL == 0 {
                0
            } else {
                common_prefix_len(prev, term)
            };
            suffixes.extend_from_slice(term[prefix_len..].as_bytes());
            suffix_offsets.push(suffixes.len() as u32);
            prefix_lens.push(prefix_len as u32);
            ids.push(*id);
            prev = term;
        }

        let min_key = sorted_entries
            .first()
            .map(|(term, _)| term.clone())
            .unwrap_or_default();
        FrontCodedBlock {
            suffixes,
            suffix_offsets,
            prefix_lens,
            ids,
            min_key,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn min_key(&self) -> &str {
        &self.min_key
    }

    fn suffix_str(&self, i: usize) -> &str {
        let start = self.suffix_offsets[i] as usize;
        let end = self.suffix_offsets[i + 1] as usize;
        std::str::from_utf8(&self.suffixes[start..end])
            .expect("suffix boundaries were computed from a valid UTF-8 str's char-boundary-safe prefix split")
    }

    /// `Some(local_rank)` if `term` is present: binary search over restart
    /// points by their (full, undelta'd) string, then decode forward at
    /// most `RESTART_INTERVAL` entries within that segment. O(log(block
    /// size / RESTART_INTERVAL) + RESTART_INTERVAL), not O(block size).
    pub(crate) fn rank(&self, term: &str) -> Option<usize> {
        if self.ids.is_empty() {
            return None;
        }
        let num_restarts = self.ids.len().div_ceil(RESTART_INTERVAL);

        let mut lo = 0usize;
        let mut hi = num_restarts;
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if self.suffix_str(mid * RESTART_INTERVAL) <= term {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let start = lo * RESTART_INTERVAL;
        let end = ((lo + 1) * RESTART_INTERVAL).min(self.ids.len());

        let mut current = self.suffix_str(start).to_string();
        if current == term {
            return Some(start);
        }
        for i in (start + 1)..end {
            current.truncate(self.prefix_lens[i] as usize);
            current.push_str(self.suffix_str(i));
            match current.as_str().cmp(term) {
                std::cmp::Ordering::Equal => return Some(i),
                std::cmp::Ordering::Greater => return None,
                std::cmp::Ordering::Less => {}
            }
        }
        None
    }

    pub(crate) fn id_at(&self, local_rank: usize) -> TermId {
        self.ids[local_rank]
    }

    pub(crate) fn local_rank_of(&self, id: TermId) -> Option<usize> {
        self.ids.iter().position(|&x| x == id)
    }

    /// `select_b(local_rank)`: decode forward from the nearest restart
    /// point. O(RESTART_INTERVAL), not O(block size) -- unlike a naive
    /// unbounded front-coded chain, this is exactly what restart points
    /// buy.
    pub(crate) fn term_at(&self, local_rank: usize) -> String {
        let restart = (local_rank / RESTART_INTERVAL) * RESTART_INTERVAL;
        let mut current = self.suffix_str(restart).to_string();
        for i in (restart + 1)..=local_rank {
            current.truncate(self.prefix_lens[i] as usize);
            current.push_str(self.suffix_str(i));
        }
        current
    }

    /// Decodes every `(term, id)` pair in sorted order into `out`,
    /// clearing it first but keeping its allocation (design doc §6/§10
    /// reusable scratch memory -- same contract as
    /// [`crate::Block::decode_into`]).
    pub(crate) fn decode_into(&self, out: &mut Vec<(Box<str>, TermId)>) {
        out.clear();
        out.reserve(self.ids.len());
        if self.ids.is_empty() {
            return;
        }
        let mut current = String::new();
        for i in 0..self.ids.len() {
            current.truncate(self.prefix_lens[i] as usize);
            current.push_str(self.suffix_str(i));
            out.push((current.as_str().into(), self.ids[i]));
        }
    }

    /// Convenience wrapper over [`FrontCodedBlock::decode_into`] for
    /// colder call sites (see [`crate::Block::decode_all`]).
    pub(crate) fn decode_all(&self) -> Vec<(Box<str>, TermId)> {
        let mut out = Vec::new();
        self.decode_into(&mut out);
        out
    }

    /// Logical byte size of this block's compressed representation
    /// (suffix bytes + the two `u32` index arrays), for design doc §14
    /// "bits per term" space measurements. Excludes `ids`/`min_key`,
    /// which [`crate::Block`] also carries identically, so they cancel
    /// out of an apples-to-apples comparison; callers wanting the whole
    /// block's footprint add those back in.
    pub(crate) fn payload_bytes(&self) -> usize {
        self.suffixes.len() + self.suffix_offsets.len() * 4 + self.prefix_lens.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(terms: &[&str]) -> Vec<(Box<str>, TermId)> {
        terms
            .iter()
            .enumerate()
            .map(|(i, t)| ((*t).into(), TermId(i as u32)))
            .collect()
    }

    #[test]
    fn round_trips_small_block() {
        let terms = ["a", "ab", "abc", "abd", "b"];
        let block = FrontCodedBlock::build(&entries(&terms));
        assert_eq!(block.len(), terms.len());
        for (i, t) in terms.iter().enumerate() {
            assert_eq!(block.rank(t), Some(i));
            assert_eq!(block.term_at(i), *t);
        }
        assert_eq!(block.rank("nope"), None);
    }

    #[test]
    fn round_trips_across_restart_boundary() {
        // More than one RESTART_INTERVAL so both restart and non-restart
        // entries, and decoding across a restart boundary, are exercised.
        let terms: Vec<String> = (0..50)
            .map(|i| format!("http://example.org/term-{i:04}"))
            .collect();
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let block = FrontCodedBlock::build(&entries(&refs));
        for (i, t) in refs.iter().enumerate() {
            assert_eq!(block.rank(t), Some(i), "rank mismatch for {t}");
            assert_eq!(block.term_at(i), *t, "term_at mismatch for rank {i}");
        }
        let decoded = block.decode_all();
        let decoded_terms: Vec<&str> = decoded.iter().map(|(t, _)| t.as_ref()).collect();
        assert_eq!(decoded_terms, refs);
    }

    #[test]
    fn handles_multibyte_utf8_shared_prefixes() {
        // Regression test for the char-boundary clamp in common_prefix_len:
        // "café..." vs "cafe..." share only up to the 'f', and "café1" vs
        // "café2" share the full multi-byte 'é'.
        let terms = ["cafe_a", "café1", "café2", "caff"];
        let mut sorted = terms.to_vec();
        sorted.sort();
        let block = FrontCodedBlock::build(&entries(&sorted));
        for (i, t) in sorted.iter().enumerate() {
            assert_eq!(block.rank(t), Some(i));
            assert_eq!(block.term_at(i), *t);
        }
    }

    #[test]
    fn empty_block() {
        let block = FrontCodedBlock::build(&[]);
        assert_eq!(block.len(), 0);
        assert_eq!(block.rank("anything"), None);
        assert_eq!(block.decode_all(), Vec::new());
    }
}
