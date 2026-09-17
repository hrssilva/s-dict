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
}

impl TermDictionary for PlainDictionary {
    fn get_id(&self, term: &str) -> Option<TermId> {
        self.forward.get(term).copied()
    }

    fn get_term(&self, id: TermId) -> Option<&str> {
        self.reverse.get(id.0 as usize)?.as_deref()
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
        assert_eq!(dict.get_term(id), Some("http://example.org/a"));
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
        assert_eq!(dict.get_term(b), Some("b"));
    }

    #[test]
    fn empty_dictionary() {
        let dict = PlainDictionary::new();
        assert!(dict.is_empty());
        assert_eq!(dict.get_id("anything"), None);
    }
}
