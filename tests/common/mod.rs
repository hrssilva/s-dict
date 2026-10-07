//! A plain `HashMap`/`Vec` model of the dictionary contract, the oracle the
//! integration tests check `PackedAutomatonDictionary` against. Same id
//! assignment: freed ids are reused last-freed first, otherwise ids are
//! dense; a batch assigns new ids in term order.

use std::collections::HashMap;

use s_dict::{TermDictionary, TermId};

#[derive(Debug, Default)]
pub struct Model {
    forward: HashMap<String, TermId>,
    reverse: Vec<Option<String>>,
    free_ids: Vec<u32>,
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TermDictionary for Model {
    fn get_id(&self, term: &str) -> Option<TermId> {
        self.forward.get(term).copied()
    }

    fn get_term(&self, id: TermId) -> Option<String> {
        self.reverse.get(id.0 as usize)?.clone()
    }

    fn insert(&mut self, term: &str) -> TermId {
        if let Some(&id) = self.forward.get(term) {
            return id;
        }
        let id = match self.free_ids.pop() {
            Some(id) => TermId(id),
            None => {
                self.reverse.push(None);
                TermId(self.reverse.len() as u32 - 1)
            }
        };
        self.reverse[id.0 as usize] = Some(term.to_string());
        self.forward.insert(term.to_string(), id);
        id
    }

    /// New terms get ids in term order, as the dictionary assigns them.
    fn insert_batch(&mut self, terms: &[&str]) -> Vec<TermId> {
        let mut sorted = terms.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        for term in sorted {
            self.insert(term);
        }
        terms.iter().map(|t| self.forward[*t]).collect()
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
