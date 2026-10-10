//! `ShardedDictionary`: N independent dictionaries, each owning one key
//! range, so N writer threads can build them concurrently.
//!
//! Every dictionary in this crate has a single writer; with many parser
//! workers that writer is the bottleneck (DESIGN.md's "sharding the
//! writer"). Ranges are fixed at construction (`bounds`, typically from a
//! sample via [`ShardedDictionary::bounds_from_sample`]), so routing is a
//! binary search over `shards - 1` keys and never needs coordination.
//! Each shard's own ids are dense from 0; a global [`TermId`] carries the
//! shard index in its top `shard_bits` bits.

use crate::{DictError, TermDictionary, TermId};

#[derive(Debug)]
pub struct ShardedDictionary<D> {
    shards: Vec<D>,
    /// `bounds[i]` is the smallest term routed to shard `i + 1`; sorted,
    /// strictly increasing, `shards.len() - 1` long.
    bounds: Vec<Box<str>>,
    shard_bits: u32,
}

impl<D: TermDictionary> ShardedDictionary<D> {
    /// One shard per range delimited by `bounds` (`bounds.len() + 1`
    /// shards), each built by `make` and limited to the local id space its
    /// global ids leave it (`TermDictionary::set_id_limit`), so a shard
    /// refuses an insert rather than handing out an id that doesn't fit.
    pub fn new(bounds: Vec<String>, mut make: impl FnMut() -> D) -> Self {
        assert!(
            bounds.windows(2).all(|w| w[0] < w[1]),
            "shard bounds must be strictly increasing"
        );
        let shards: Vec<D> = (0..=bounds.len()).map(|_| make()).collect();
        let shard_bits = usize::BITS - (shards.len() - 1).leading_zeros();
        assert!(shard_bits <= 16, "at most 65,536 shards");
        let mut dict = ShardedDictionary {
            shards,
            bounds: bounds.into_iter().map(String::into_boxed_str).collect(),
            shard_bits,
        };
        if shard_bits > 0 {
            dict.set_id_limit(u32::MAX);
        }
        dict
    }

    /// Each shard's sub-batch of `terms`: `(input positions, terms)`.
    fn per_shard<'a>(&self, terms: &[&'a str]) -> Vec<(Vec<usize>, Vec<&'a str>)> {
        let mut per_shard: Vec<(Vec<usize>, Vec<&str>)> = vec![Default::default(); self.shards.len()];
        for (pos, &term) in terms.iter().enumerate() {
            let (positions, sub) = &mut per_shard[self.shard_of(term)];
            positions.push(pos);
            sub.push(term);
        }
        per_shard
    }

    /// Up to `shards - 1` bounds that split the *distinct* terms of the
    /// data `sample` was drawn from into similar counts. `sample` is an
    /// occurrence sample (terms as they appear, repeats included). Its
    /// distinct terms are not a fair basis: a frequent term shows up many
    /// times but is one dictionary entry, while rare terms -- most of a
    /// dictionary -- are mostly unseen. Following Good-Turing, the terms
    /// seen exactly once estimate where the unseen distinct terms lie, so
    /// quantiles are taken over those singletons (over all distinct terms
    /// if there are fewer singletons than shards). Measured on real RDF:
    /// distinct-term quantiles put 49% of terms in one of 4 shards;
    /// singleton quantiles, see DESIGN.md's "sharding the writer".
    pub fn bounds_from_sample(sample: &mut Vec<&str>, shards: usize) -> Vec<String> {
        sample.sort_unstable();
        let mut singletons: Vec<&str> = Vec::new();
        let mut distinct: Vec<&str> = Vec::new();
        for run in sample.chunk_by(|a, b| a == b) {
            distinct.push(run[0]);
            if run.len() == 1 {
                singletons.push(run[0]);
            }
        }
        let basis = if singletons.len() >= shards { singletons } else { distinct };
        let mut bounds: Vec<String> = (1..shards)
            .filter_map(|k| basis.get(k * basis.len() / shards))
            .map(|t| t.to_string())
            .collect();
        bounds.dedup();
        bounds.retain(|b| !b.is_empty());
        bounds
    }

    /// The shard owning `term`.
    #[inline]
    pub fn shard_of(&self, term: &str) -> usize {
        shard_of(&self.bounds, term)
    }

    /// An owned copy of the routing, for producers (parser workers) that
    /// route terms while writer threads hold the shards mutably.
    pub fn router(&self) -> ShardRouter {
        ShardRouter {
            bounds: self.bounds.clone(),
        }
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn shards(&self) -> &[D] {
        &self.shards
    }

    /// For one writer thread per shard; ids a shard returns are local --
    /// convert with [`Self::global_id`].
    pub fn shards_mut(&mut self) -> &mut [D] {
        &mut self.shards
    }

    /// The global id of `shard`'s local id `local`. Panics if `local`
    /// doesn't fit the shard's local id space, which only a shard ignoring
    /// its id limit (`TermDictionary::set_id_limit`) hands out.
    #[inline]
    pub fn global_id(&self, shard: usize, local: TermId) -> TermId {
        if self.shard_bits == 0 {
            return local;
        }
        let local_bits = 32 - self.shard_bits;
        assert!(
            local.0 >> local_bits == 0,
            "shard {shard} handed out an id beyond its {local_bits}-bit id limit"
        );
        TermId(((shard as u32) << local_bits) | local.0)
    }

    /// `(shard, local id)` of a global id.
    #[inline]
    pub fn split_id(&self, id: TermId) -> (usize, TermId) {
        if self.shard_bits == 0 {
            return (0, id);
        }
        let local_bits = 32 - self.shard_bits;
        ((id.0 >> local_bits) as usize, TermId(id.0 & ((1 << local_bits) - 1)))
    }
}

/// [`ShardedDictionary::shard_of`], detached from the shards.
#[derive(Debug, Clone)]
pub struct ShardRouter {
    bounds: Vec<Box<str>>,
}

impl ShardRouter {
    #[inline]
    pub fn shard_of(&self, term: &str) -> usize {
        shard_of(&self.bounds, term)
    }

    pub fn shard_count(&self) -> usize {
        self.bounds.len() + 1
    }
}

#[inline]
fn shard_of(bounds: &[Box<str>], term: &str) -> usize {
    bounds.partition_point(|b| b.as_ref() <= term)
}

impl<D: TermDictionary> TermDictionary for ShardedDictionary<D> {
    fn get_id(&self, term: &str) -> Option<TermId> {
        let shard = self.shard_of(term);
        let local = self.shards[shard].get_id(term)?;
        Some(self.global_id(shard, local))
    }

    fn get_term(&self, id: TermId) -> Option<String> {
        let (shard, local) = self.split_id(id);
        self.shards.get(shard)?.get_term(local)
    }

    fn try_insert(&mut self, term: &str) -> Result<TermId, DictError> {
        let shard = self.shard_of(term);
        let local = self.shards[shard].try_insert(term)?;
        Ok(self.global_id(shard, local))
    }

    /// Routes each term to its shard, checks every shard's sub-batch
    /// (`check_insert_batch`, so a refusal leaves every shard unchanged),
    /// then inserts each with one `try_insert_batch` call. Ids in input
    /// order.
    fn try_insert_batch(&mut self, terms: &[&str]) -> Result<Vec<TermId>, DictError> {
        let per_shard = self.per_shard(terms);
        for (shard, (_, sub)) in per_shard.iter().enumerate() {
            if !sub.is_empty() {
                self.shards[shard].check_insert_batch(sub)?;
            }
        }
        let mut result = vec![TermId(0); terms.len()];
        for (shard, (positions, sub)) in per_shard.into_iter().enumerate() {
            if sub.is_empty() {
                continue;
            }
            let ids = self.shards[shard].try_insert_batch(&sub)?;
            for (pos, local) in positions.into_iter().zip(ids) {
                result[pos] = self.global_id(shard, local);
            }
        }
        Ok(result)
    }

    fn check_insert_batch(&self, terms: &[&str]) -> Result<(), DictError> {
        for (shard, (_, sub)) in self.per_shard(terms).iter().enumerate() {
            if !sub.is_empty() {
                self.shards[shard].check_insert_batch(sub)?;
            }
        }
        Ok(())
    }

    fn remove(&mut self, term: &str) -> Option<TermId> {
        let shard = self.shard_of(term);
        let local = self.shards[shard].remove(term)?;
        Some(self.global_id(shard, local))
    }

    fn len(&self) -> usize {
        self.shards.iter().map(TermDictionary::len).sum()
    }

    /// Global ids below `limit`: each shard gets the part of its local id
    /// space whose global ids stay below `limit`.
    fn set_id_limit(&mut self, limit: u32) {
        if self.shard_bits == 0 {
            self.shards[0].set_id_limit(limit);
            return;
        }
        let local_bits = 32 - self.shard_bits;
        for (i, shard) in self.shards.iter_mut().enumerate() {
            let base = (i as u64) << local_bits;
            let local_limit = (u64::from(limit).saturating_sub(base)).min(1 << local_bits);
            shard.set_id_limit(local_limit as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PackedAutomatonDictionary;

    #[test]
    fn routes_and_round_trips_across_shards() {
        let terms: Vec<String> = (0..3000).map(|i| format!("t{:05}", i * 7 % 3001)).collect();
        let mut sample: Vec<&str> = terms.iter().step_by(13).map(String::as_str).collect();
        let bounds = ShardedDictionary::<PackedAutomatonDictionary>::bounds_from_sample(&mut sample, 5);
        assert_eq!(bounds.len(), 4);
        let mut dict = ShardedDictionary::new(bounds, || PackedAutomatonDictionary::with_block_bounds(8, 32));
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let ids = dict.insert_batch(&refs[..2000]);
        let mut more: Vec<TermId> = refs[2000..].iter().map(|t| dict.insert(t)).collect();
        let mut all = ids;
        all.append(&mut more);
        assert_eq!(dict.len(), terms.len());
        let router = dict.router();
        assert!(refs.iter().all(|t| router.shard_of(t) == dict.shard_of(t)));
        assert!(dict.shards().iter().all(|s| s.len() > 0), "every shard used");
        let mut seen = std::collections::HashSet::new();
        for (t, &id) in refs.iter().zip(&all) {
            assert!(seen.insert(id), "global ids are unique");
            assert_eq!(dict.get_id(t), Some(id));
            assert_eq!(dict.get_term(id).as_deref(), Some(*t));
        }
        for t in &refs[..100] {
            assert!(dict.remove(t).is_some());
            assert_eq!(dict.get_id(t), None);
        }
        assert_eq!(dict.len(), terms.len() - 100);
        assert_eq!(dict.get_term(TermId(u32::MAX)), None);
    }

    #[test]
    fn single_shard_ids_are_unchanged() {
        let mut dict = ShardedDictionary::new(Vec::new(), PackedAutomatonDictionary::new);
        let id = dict.insert("a");
        assert_eq!(id, TermId(0));
        assert_eq!(dict.split_id(id), (0, TermId(0)));
    }
}
