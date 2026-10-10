//! A plain `HashMap`/`Vec` model of the dictionary contract, the oracle the
//! integration tests check `PackedAutomatonDictionary` against. Same id
//! assignment: freed ids are reused last-freed first, otherwise ids are
//! dense; a batch assigns new ids in term order.

use std::collections::HashMap;

use s_dict::{DictError, TermDictionary, TermId};

#[derive(Debug, Default)]
pub struct Model {
    forward: HashMap<String, TermId>,
    reverse: Vec<Option<String>>,
    free_ids: Vec<u32>,
    id_limit: Option<u32>,
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    fn available(&self) -> u64 {
        (self.free_ids.len() as u64).saturating_add(self.id_limit.map_or(u64::MAX, |l| u64::from(l) - self.reverse.len() as u64))
    }
}

impl TermDictionary for Model {
    fn get_id(&self, term: &str) -> Option<TermId> {
        self.forward.get(term).copied()
    }

    fn get_term(&self, id: TermId) -> Option<String> {
        self.reverse.get(id.0 as usize)?.clone()
    }

    fn try_insert(&mut self, term: &str) -> Result<TermId, DictError> {
        self.check_insert_batch(&[term])?;
        if let Some(&id) = self.forward.get(term) {
            return Ok(id);
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
        Ok(id)
    }

    /// New terms get ids in term order, as the dictionary assigns them.
    fn try_insert_batch(&mut self, terms: &[&str]) -> Result<Vec<TermId>, DictError> {
        self.check_insert_batch(terms)?;
        let mut sorted = terms.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        for term in sorted {
            self.try_insert(term)?;
        }
        Ok(terms.iter().map(|t| self.forward[*t]).collect())
    }

    /// Ids only (the model has no size limits).
    fn check_insert_batch(&self, terms: &[&str]) -> Result<(), DictError> {
        let mut new: Vec<&str> = terms.iter().copied().filter(|t| !self.forward.contains_key(*t)).collect();
        new.sort_unstable();
        new.dedup();
        let (needed, available) = (new.len() as u64, self.available());
        if needed > available {
            return Err(DictError::IdsExhausted { needed, available });
        }
        Ok(())
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

    fn set_id_limit(&mut self, limit: u32) {
        assert!(self.reverse.len() as u64 <= u64::from(limit));
        self.id_limit = Some(limit);
    }
}
