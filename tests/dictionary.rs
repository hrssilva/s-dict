//! Differential testing between `Dictionary` (Stage 1) and `PlainDictionary`
//! (Stage 0, the oracle) across randomized insert/remove sequences and
//! several block-bound configurations small enough to force splits and
//! merges during the run. This is the main correctness signal for the
//! router/split/merge logic in `Dictionary`, since `PlainDictionary` has no
//! such moving parts to get wrong.

use s_dict::{Dictionary, PlainDictionary, TermDictionary, TermId};

/// A small deterministic xorshift64* PRNG, avoiding a `rand` dependency for
/// this stage (design doc §11: minimize unnecessary structure).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// RDF-like terms per design doc §8: IRIs from a couple of namespaces,
/// language-tagged literals, and blank nodes.
fn term_pool() -> Vec<String> {
    let namespaces = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://dbpedia.org/resource/",
        "http://dbpedia.org/ontology/",
    ];
    let locals = [
        "type",
        "Class",
        "label",
        "Alice",
        "Bob",
        "Person",
        "name",
        "knows",
        "Foo_Bar",
        "Baz",
        "subClassOf",
        "comment",
    ];

    let mut pool = Vec::new();
    for ns in namespaces {
        for local in locals {
            pool.push(format!("{ns}{local}"));
        }
    }
    for i in 0..40 {
        pool.push(format!("\"literal value {i}\"@en"));
        pool.push(format!("\"literal value {i}\"@fr"));
        pool.push(format!("_:b{i}"));
    }
    pool.sort();
    pool.dedup();
    pool
}

fn run_fuzz(seed: u64, block_min: usize, block_max: usize, steps: usize) {
    let pool = term_pool();
    let mut rng = Lcg::new(seed);
    let mut plain = PlainDictionary::new();
    let mut blocked = Dictionary::with_block_bounds(block_min, block_max);
    let mut present: std::collections::HashSet<String> = std::collections::HashSet::new();

    for step in 0..steps {
        let term = &pool[rng.next_range(pool.len())];
        let will_insert = if present.contains(term) {
            // Occasionally re-insert an already-present term to exercise
            // idempotency, but mostly remove it.
            rng.next_range(4) == 0
        } else {
            true
        };

        if will_insert {
            let a = plain.insert(term);
            let b = blocked.insert(term);
            assert_eq!(a, b, "step {step}: id diverged inserting {term:?}");
            present.insert(term.clone());
        } else {
            let a = plain.remove(term);
            let b = blocked.remove(term);
            assert_eq!(a, b, "step {step}: id diverged removing {term:?}");
            present.remove(term.as_str());
        }

        // Full lockstep check: every term currently believed present must
        // round-trip identically through both dictionaries, and the two
        // dictionaries must agree on total size.
        assert_eq!(plain.len(), blocked.len(), "step {step}: length diverged");
        for t in &present {
            let a: Option<TermId> = plain.get_id(t);
            let b: Option<TermId> = blocked.get_id(t);
            assert_eq!(a, b, "step {step}: get_id diverged for {t:?}");
            assert_eq!(
                plain.get_term(a.unwrap()),
                blocked.get_term(b.unwrap()),
                "step {step}: get_term diverged for {t:?}"
            );
        }
    }
}

#[test]
fn fuzz_small_blocks_force_splits_and_merges() {
    run_fuzz(0x2545_F491_4F6C_DD1D, 2, 8, 400);
}

#[test]
fn fuzz_medium_blocks() {
    run_fuzz(0x9E37_79B9_7F4A_7C15, 8, 32, 400);
}

#[test]
fn fuzz_different_seed() {
    run_fuzz(0xBF58_476D_1CE4_E5B9, 2, 8, 400);
}

/// Sanity round-trip with real-looking RDF terms directly through
/// `Dictionary` (not just the fuzz oracle comparison), per design doc §8.
#[test]
fn dictionary_round_trips_real_rdf_terms() {
    let mut dict = Dictionary::with_block_bounds(4, 16);
    let terms = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#Class",
        "http://dbpedia.org/resource/Berlin",
        "http://dbpedia.org/ontology/populationTotal",
        "\"Berlin\"@en",
        "\"42\"^^http://www.w3.org/2001/XMLSchema#integer",
        "_:b0",
    ];

    let ids: Vec<TermId> = terms.iter().map(|t| dict.insert(t)).collect();
    for (term, &id) in terms.iter().zip(&ids) {
        assert_eq!(dict.get_id(term), Some(id));
        assert_eq!(dict.get_term(id), Some(*term));
    }

    // Remove one from the middle and confirm the rest still round-trip.
    let removed = dict.remove(terms[2]).unwrap();
    assert_eq!(dict.get_term(removed), None);
    assert_eq!(dict.get_id(terms[2]), None);
    for (term, &id) in terms.iter().zip(&ids) {
        if *term == terms[2] {
            continue;
        }
        assert_eq!(dict.get_id(term), Some(id));
    }
}
