use std::collections::HashMap;

use crate::{TermDictionary, TermId};

/// The Stage 0 / design-doc §13 "plain baseline": a `HashMap<Box<str>,
/// TermId>` forward index and a dense `Vec<Option<Box<str>>>` reverse
/// index, with freed ids reused via a free list. No blocks, no lexical
/// ordering, no compression. Kept permanently as the comparison point later
/// stages are measured against, not thrown away once `Dictionary` exists.
#[derive(Debug, Default)]
pub struct PlainDictionary {
    forward: HashMap<Box<str>, TermId>,
    reverse: Vec<Option<Box<str>>>,
    free_ids: Vec<u32>,
}

impl PlainDictionary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total logical byte size, on the same "compressed-payload +
    /// metadata" accounting as [`crate::Dictionary::size_bytes`] and
    /// [`crate::FrontCodedDictionary::size_bytes`] -- see their docs for
    /// exactly what's included, so all three are directly comparable.
    /// Unlike the blocked dictionaries, `PlainDictionary` genuinely stores
    /// each term's bytes twice (once as a `HashMap` key, once in the
    /// dense reverse `Vec`), which this reflects rather than hides -- a
    /// real, honest part of why it's expected to be the least
    /// space-efficient baseline. Does not account for `HashMap`/`Vec`
    /// allocator slack or bucket/control-byte overhead.
    pub fn size_bytes(&self) -> usize {
        let forward_bytes: usize = self
            .forward
            .keys()
            .map(|key| key.len() + std::mem::size_of::<TermId>())
            .sum();
        let reverse_bytes = self.reverse.len() * std::mem::size_of::<Option<Box<str>>>()
            + self
                .reverse
                .iter()
                .filter_map(|s| s.as_ref())
                .map(|s| s.len())
                .sum::<usize>();
        let free_ids_bytes = self.free_ids.len() * std::mem::size_of::<u32>();
        forward_bytes + reverse_bytes + free_ids_bytes
    }

    /// See [`crate::Dictionary::get_term_into`] -- identical contract.
    /// `PlainDictionary` already stores the term as an owned `Box<str>`,
    /// so this just copies its bytes into `buf` rather than decoding
    /// anything, but it's offered for the same reason: letting a hot loop
    /// reuse one buffer across many calls instead of allocating a
    /// `String` per call.
    pub fn get_term_into(&self, id: TermId, buf: &mut Vec<u8>) -> bool {
        buf.clear();
        let Some(Some(s)) = self.reverse.get(id.0 as usize) else {
            return false;
        };
        buf.extend_from_slice(s.as_bytes());
        true
    }

    /// See [`crate::Dictionary::get_terms_into`] -- same signature, for
    /// API parity across all three dictionary types. There's no FST walk
    /// to amortize here (`reverse` is a direct array, no block/decode
    /// concept at all), so this is just a per-id loop, not a real
    /// optimization the way it is for `Dictionary`/`FrontCodedDictionary`.
    pub fn get_terms_into(&self, ids: &[TermId], out: &mut Vec<Option<String>>) {
        out.clear();
        out.extend(ids.iter().map(|&id| {
            self.reverse
                .get(id.0 as usize)
                .and_then(|s| s.as_ref())
                .map(|s| s.to_string())
        }));
    }
}

impl TermDictionary for PlainDictionary {
    fn get_id(&self, term: &str) -> Option<TermId> {
        self.forward.get(term).copied()
    }

    fn get_term(&self, id: TermId) -> Option<String> {
        self.reverse
            .get(id.0 as usize)?
            .as_deref()
            .map(str::to_string)
    }

    fn insert(&mut self, term: &str) -> TermId {
        if let Some(&id) = self.forward.get(term) {
            return id;
        }
        let id = match self.free_ids.pop() {
            Some(slot) => TermId(slot),
            None => {
                let id = TermId(self.reverse.len() as u32);
                self.reverse.push(None);
                id
            }
        };
        self.reverse[id.0 as usize] = Some(term.into());
        self.forward.insert(term.into(), id);
        id
    }

    fn remove(&mut self, term: &str) -> Option<TermId> {
        let id = self.forward.remove(term)?;
        self.reverse[id.0 as usize] = None;
        self.free_ids.push(id.0);
        Some(id)
    }

    fn len(&self) -> usize {
        self.forward.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_is_idempotent() {
        let mut dict = PlainDictionary::new();
        let a = dict.insert("http://example.org/a");
        let b = dict.insert("http://example.org/a");
        assert_eq!(a, b);
        assert_eq!(dict.len(), 1);
    }

    #[test]
    fn round_trips() {
        let mut dict = PlainDictionary::new();
        let id = dict.insert("http://example.org/a");
        assert_eq!(dict.get_term(id).as_deref(), Some("http://example.org/a"));
        assert_eq!(dict.get_id("http://example.org/a"), Some(id));
    }

    #[test]
    fn remove_absent_is_none() {
        let mut dict = PlainDictionary::new();
        assert_eq!(dict.remove("nope"), None);
    }

    #[test]
    fn removed_id_is_reused() {
        let mut dict = PlainDictionary::new();
        let a = dict.insert("a");
        dict.remove("a");
        let b = dict.insert("b");
        assert_eq!(a, b);
        assert_eq!(dict.get_term(b).as_deref(), Some("b"));
    }

    #[test]
    fn empty_dictionary() {
        let dict = PlainDictionary::new();
        assert!(dict.is_empty());
        assert_eq!(dict.get_id("anything"), None);
    }

    #[test]
    fn size_bytes_grows_with_content_and_shrinks_on_remove() {
        let mut dict = PlainDictionary::new();
        assert_eq!(dict.size_bytes(), 0);

        dict.insert("http://example.org/a");
        let after_one = dict.size_bytes();
        assert!(after_one > 0);

        dict.insert("http://example.org/b");
        assert!(dict.size_bytes() > after_one);

        dict.remove("http://example.org/a");
        dict.remove("http://example.org/b");
        // The reverse Vec's slots (now None) still cost their Option<Box<str>>
        // size; only the string content and forward-map entries are freed.
        assert!(dict.size_bytes() < after_one);
    }

    #[test]
    fn get_term_into_matches_get_term() {
        let mut dict = PlainDictionary::new();
        let id = dict.insert("http://example.org/a");
        let mut buf = Vec::new();
        assert!(dict.get_term_into(id, &mut buf));
        assert_eq!(std::str::from_utf8(&buf).unwrap(), "http://example.org/a");

        let missing = TermId(9999);
        buf.clear();
        assert!(!dict.get_term_into(missing, &mut buf));
        assert!(buf.is_empty());
    }

    #[test]
    fn get_terms_into_matches_individual_lookups() {
        let mut dict = PlainDictionary::new();
        let a = dict.insert("a");
        let b = dict.insert("b");
        let query = [b, a, b, TermId(9999)];
        let mut out = Vec::new();
        dict.get_terms_into(&query, &mut out);
        assert_eq!(
            out,
            vec![
                Some("b".to_string()),
                Some("a".to_string()),
                Some("b".to_string()),
                None
            ]
        );
    }
}
