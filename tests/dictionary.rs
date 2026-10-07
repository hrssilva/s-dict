//! Differential testing of `PackedAutomatonDictionary` against a plain
//! model (`common::Model`) across randomized insert/remove sequences, with
//! block bounds small enough to force splits and merges, interleaved with
//! end-of-load compaction and path compression (`shrink_to_fit`): the main
//! correctness signal for the router, split/merge and id-to-block logic.

mod common;

use common::Model;
use s_dict::{PackedAutomatonDictionary, ShardedDictionary, TermDictionary};

/// A small deterministic xorshift64* PRNG.
struct Rng(u64);

impl Rng {
    fn next_range(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % n
    }
}

/// RDF-like terms: IRIs from a few namespaces, language-tagged literals,
/// blank nodes, non-ASCII text and the empty term.
fn term_pool() -> Vec<String> {
    let namespaces = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://dbpedia.org/resource/",
        "http://dbpedia.org/ontology/",
    ];
    let locals = ["type", "Class", "label", "Alice", "Bob", "Person", "name", "knows", "Foo_Bar", "Baz", "subClassOf", "comment"];
    let mut pool: Vec<String> = namespaces.iter().flat_map(|ns| locals.iter().map(move |l| format!("{ns}{l}"))).collect();
    for i in 0..40 {
        pool.push(format!("\"literal value {i}\"@en"));
        pool.push(format!("\"literal value {i}\"@fr"));
        pool.push(format!("_:b{i}"));
        pool.push(format!("\"é中 {i}\""));
    }
    pool.push(String::new());
    pool.sort();
    pool.dedup();
    pool
}

fn assert_same<D: TermDictionary>(dict: &D, model: &Model, pool: &[String], context: &str) {
    assert_eq!(dict.len(), model.len(), "{context}: len");
    for term in pool {
        let id = model.get_id(term);
        assert_eq!(dict.get_id(term), id, "{context}: get_id({term:?})");
        if let Some(id) = id {
            assert_eq!(dict.get_term(id).as_deref(), Some(term.as_str()), "{context}: get_term({id:?})");
        }
    }
}

fn run_fuzz(seed: u64, block_min: usize, block_max: usize, steps: usize) {
    let pool = term_pool();
    let mut rng = Rng(seed);
    let mut model = Model::new();
    let mut dict = PackedAutomatonDictionary::with_block_bounds(block_min, block_max);
    for step in 0..steps {
        match rng.next_range(10) {
            0..=4 => {
                let term = &pool[rng.next_range(pool.len())];
                assert_eq!(dict.insert(term), model.insert(term), "step {step}: insert {term:?}");
            }
            5 | 6 => {
                let batch: Vec<&str> = (0..1 + rng.next_range(40)).map(|_| pool[rng.next_range(pool.len())].as_str()).collect();
                assert_eq!(dict.insert_batch(&batch), model.insert_batch(&batch), "step {step}: insert_batch");
            }
            _ => {
                let term = &pool[rng.next_range(pool.len())];
                assert_eq!(dict.remove(term), model.remove(term), "step {step}: remove {term:?}");
            }
        }
        if step % 97 == 50 {
            // Compaction and chaining; later mutations expand chains.
            dict.shrink_to_fit();
        }
        if step % 25 == 0 || step + 1 == steps {
            assert_same(&dict, &model, &pool, &format!("step {step}"));
        }
    }
}

#[test]
fn fuzz_small_blocks_force_splits_and_merges() {
    run_fuzz(0x1234_5678, 2, 4, 3000);
}

#[test]
fn fuzz_medium_blocks() {
    run_fuzz(0xDEAD_BEEF, 8, 32, 3000);
}

#[test]
fn fuzz_default_blocks() {
    let pool = term_pool();
    let mut dict = PackedAutomatonDictionary::new();
    let mut model = Model::new();
    let refs: Vec<&str> = pool.iter().map(String::as_str).collect();
    assert_eq!(dict.insert_batch(&refs), model.insert_batch(&refs));
    dict.shrink_to_fit();
    assert_same(&dict, &model, &pool, "default bounds");
}

/// Real-looking RDF terms round-trip, including bulk decode by id.
#[test]
fn round_trips_real_rdf_terms() {
    let terms = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
        "http://dbpedia.org/resource/Berlin",
        "\"Berlin\"@de",
        "\"Berlin\"@en",
        "_:b0",
        "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>",
    ];
    let mut dict = PackedAutomatonDictionary::with_block_bounds(1, 2);
    let ids: Vec<_> = terms.iter().map(|t| dict.insert(t)).collect();
    assert!(dict.block_count() > 1, "should span several blocks");
    let mut out = Vec::new();
    dict.get_terms_seeked_into(&ids, &mut out);
    for ((term, id), decoded) in terms.iter().zip(&ids).zip(&out) {
        assert_eq!(dict.get_id(term), Some(*id));
        assert_eq!(decoded.as_deref(), Some(*term));
    }
    assert_eq!(dict.remove(terms[2]), Some(ids[2]));
    assert_eq!(dict.get_id(terms[2]), None);
    assert_eq!(dict.get_term(ids[2]), None);
}

/// Shards route terms by key range and return globally unique ids.
#[test]
fn sharded_dictionary_round_trips() {
    let pool = term_pool();
    let mut sample: Vec<&str> = pool.iter().step_by(7).map(String::as_str).collect();
    let bounds = ShardedDictionary::<PackedAutomatonDictionary>::bounds_from_sample(&mut sample, 4);
    let mut dict = ShardedDictionary::new(bounds, || PackedAutomatonDictionary::with_block_bounds(2, 8));
    let refs: Vec<&str> = pool.iter().map(String::as_str).collect();
    let ids = dict.insert_batch(&refs);
    let mut seen = std::collections::HashSet::new();
    for (term, id) in refs.iter().zip(&ids) {
        assert!(seen.insert(*id), "ids are unique across shards");
        assert_eq!(dict.get_term(*id).as_deref(), Some(*term));
        assert_eq!(dict.get_id(term), Some(*id));
    }
}
