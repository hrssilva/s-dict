//! Differential testing between `Dictionary` (Stage 1+3, FST blocks),
//! `FrontCodedDictionary` (the §8/§13 front-coded-blocks baseline), and
//! `PlainDictionary` (Stage 0, the oracle) across randomized insert/remove
//! sequences and several block-bound configurations small enough to force
//! splits and merges during the run. This is the main correctness signal
//! for the router/split/merge logic in both blocked dictionaries, since
//! `PlainDictionary` has no such moving parts to get wrong -- and, for
//! `FrontCodedDictionary` specifically, for the restart-point binary
//! search and UTF-8 char-boundary handling in its delta coding.

use s_dict::{Dictionary, FrontCodedDictionary, PlainDictionary, TermDictionary, TermId};

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
    let mut fst_blocked = Dictionary::with_block_bounds(block_min, block_max);
    let mut front_coded = FrontCodedDictionary::with_block_bounds(block_min, block_max);
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
            let b = fst_blocked.insert(term);
            let c = front_coded.insert(term);
            assert_eq!(a, b, "step {step}: id diverged inserting {term:?} (fst)");
            assert_eq!(
                a, c,
                "step {step}: id diverged inserting {term:?} (front-coded)"
            );
            present.insert(term.clone());
        } else {
            let a = plain.remove(term);
            let b = fst_blocked.remove(term);
            let c = front_coded.remove(term);
            assert_eq!(a, b, "step {step}: id diverged removing {term:?} (fst)");
            assert_eq!(
                a, c,
                "step {step}: id diverged removing {term:?} (front-coded)"
            );
            present.remove(term.as_str());
        }

        // Full lockstep check: every term currently believed present must
        // round-trip identically through all three dictionaries, and all
        // three must agree on total size.
        assert_eq!(
            plain.len(),
            fst_blocked.len(),
            "step {step}: length diverged (fst)"
        );
        assert_eq!(
            plain.len(),
            front_coded.len(),
            "step {step}: length diverged (front-coded)"
        );
        for t in &present {
            let a: Option<TermId> = plain.get_id(t);
            let b: Option<TermId> = fst_blocked.get_id(t);
            let c: Option<TermId> = front_coded.get_id(t);
            assert_eq!(a, b, "step {step}: get_id diverged for {t:?} (fst)");
            assert_eq!(a, c, "step {step}: get_id diverged for {t:?} (front-coded)");
            assert_eq!(
                plain.get_term(a.unwrap()).as_deref(),
                fst_blocked.get_term(b.unwrap()).as_deref(),
                "step {step}: get_term diverged for {t:?} (fst)"
            );
            assert_eq!(
                plain.get_term(a.unwrap()).as_deref(),
                front_coded.get_term(c.unwrap()).as_deref(),
                "step {step}: get_term diverged for {t:?} (front-coded)"
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
        assert_eq!(dict.get_term(id).as_deref(), Some(*term));
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

#[test]
#[ignore = "manual perf probe, not part of the regular suite"]
fn perf_probe_default_block_bounds() {
    use std::time::Instant;
    let terms: Vec<String> = (0..20_000)
        .map(|i| format!("http://example.org/term-{i:06}"))
        .collect();
    let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();

    let start = Instant::now();
    let mut dict = Dictionary::new(); // default block bounds 64..=256
    for t in &term_refs {
        dict.insert(t);
    }
    let insert_elapsed = start.elapsed();

    let ids: Vec<TermId> = term_refs.iter().map(|t| dict.get_id(t).unwrap()).collect();

    let start = Instant::now();
    for &id in &ids {
        std::hint::black_box(dict.get_term(id));
    }
    let get_term_elapsed = start.elapsed();

    let mut buf = Vec::new();
    let start = Instant::now();
    for &id in &ids {
        dict.get_term_into(id, &mut buf);
        std::hint::black_box(&buf);
    }
    let get_term_into_elapsed = start.elapsed();

    let start = Instant::now();
    for t in &term_refs {
        dict.remove(t);
    }
    let remove_elapsed = start.elapsed();

    eprintln!(
        "20000 sequential inserts: {insert_elapsed:?} ({:?}/op); \
         20000 get_term (allocates): {get_term_elapsed:?} ({:?}/op); \
         20000 get_term_into (reused buffer): {get_term_into_elapsed:?} ({:?}/op); \
         20000 sequential removes: {remove_elapsed:?} ({:?}/op)",
        insert_elapsed / 20_000,
        get_term_elapsed / 20_000,
        get_term_into_elapsed / 20_000,
        remove_elapsed / 20_000,
    );
}

/// The actual payoff measurement design doc §8/§13/§14 calls for: how much
/// space, rebuild-time, *and* forward/reverse lookup time the FST-backed
/// `Dictionary` costs or saves relative to the `FrontCodedDictionary`
/// baseline, on realistic namespace-clustered RDF-like terms (not the flat
/// `term-000001` style sequences the other perf probes use, which have no
/// shared-suffix structure for an automaton to exploit over front coding).
#[test]
#[ignore = "manual perf probe, not part of the regular suite"]
fn perf_probe_space_and_rebuild_fst_vs_front_coded() {
    use std::time::Instant;

    let namespaces = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://dbpedia.org/resource/",
        "http://dbpedia.org/ontology/",
        "http://xmlns.com/foaf/0.1/",
        "http://purl.org/dc/terms/",
    ];
    let mut terms: Vec<String> = Vec::new();
    for ns in namespaces {
        for i in 0..1500 {
            terms.push(format!("{ns}Entity{i:05}"));
        }
    }
    terms.sort();
    terms.dedup();
    let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();
    let n = term_refs.len();

    let start = Instant::now();
    let mut fst_dict = Dictionary::new();
    for t in &term_refs {
        fst_dict.insert(t);
    }
    let fst_insert = start.elapsed();
    let fst_size = fst_dict.size_bytes();
    let fst_blocks = fst_dict.block_count();

    let start = Instant::now();
    let mut fc_dict = FrontCodedDictionary::new();
    for t in &term_refs {
        fc_dict.insert(t);
    }
    let fc_insert = start.elapsed();
    let fc_size = fc_dict.size_bytes();
    let fc_blocks = fc_dict.block_count();

    // Forward decode: term -> TermId (rank_b via Map::get / the restart-point
    // binary search). The collected ids double as input for the reverse pass.
    let start = Instant::now();
    let fst_ids: Vec<TermId> = term_refs
        .iter()
        .map(|t| fst_dict.get_id(t).unwrap())
        .collect();
    let fst_get_id = start.elapsed();

    let start = Instant::now();
    let fc_ids: Vec<TermId> = term_refs
        .iter()
        .map(|t| fc_dict.get_id(t).unwrap())
        .collect();
    let fc_get_id = start.elapsed();

    // Reverse decode: TermId -> term (select_b via Fst::get_key vs. decode
    // forward from the nearest restart point).
    let start = Instant::now();
    for &id in &fst_ids {
        std::hint::black_box(fst_dict.get_term(id));
    }
    let fst_get_term = start.elapsed();

    let start = Instant::now();
    for &id in &fc_ids {
        std::hint::black_box(fc_dict.get_term(id));
    }
    let fc_get_term = start.elapsed();

    eprintln!(
        "n={n} terms, {fst_blocks} fst blocks / {fc_blocks} front-coded blocks (default 64..=256 bounds)\n\
         insert:          fst {fst_insert:?} ({:?}/op) vs front-coded {fc_insert:?} ({:?}/op)\n\
         get_id (rank_b):  fst {fst_get_id:?} ({:?}/op) vs front-coded {fc_get_id:?} ({:?}/op)\n\
         get_term (select_b): fst {fst_get_term:?} ({:?}/op) vs front-coded {fc_get_term:?} ({:?}/op)\n\
         size:   fst {fst_size} bytes ({:.2} bytes/term) vs front-coded {fc_size} bytes ({:.2} bytes/term)",
        fst_insert / n as u32,
        fc_insert / n as u32,
        fst_get_id / n as u32,
        fc_get_id / n as u32,
        fst_get_term / n as u32,
        fc_get_term / n as u32,
        fst_size as f64 / n as f64,
        fc_size as f64 / n as f64,
    );
}
