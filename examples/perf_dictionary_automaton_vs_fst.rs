//! End-to-end comparison: `AutomatonDictionary` vs `Dictionary`, through
//! the shared `TermDictionary` trait -- not an isolated block-level
//! microbenchmark like `perf_insert_single_flush.rs`/the `perf_probe_*`
//! tests in `src/micro_fst.rs`, but the whole implementation, routing and
//! split/merge included, at identical default block bounds (64..=256) so
//! the block backend is the only variable.
//!
//! Every prior measurement this project made was either a differential
//! correctness check or a block-level-only perf probe; this is the first
//! "does the block-level win survive contact with the router/split/merge
//! machinery" measurement, which is why it exists as its own example
//! rather than another `#[ignore]`d test.
//!
//! Usage: `cargo build --release --example perf_dictionary_automaton_vs_fst`,
//! then run each `(dict, mode)` pair as its own process invocation (this
//! crate's established `perf stat` isolation discipline -- one process
//! per side so hardware counters can be attributed cleanly):
//!
//! ```sh
//! ./target/release/examples/perf_dictionary_automaton_vs_fst fst bulk_insert
//! ./target/release/examples/perf_dictionary_automaton_vs_fst automaton bulk_insert
//! ./target/release/examples/perf_dictionary_automaton_vs_fst fst sequential_insert
//! ./target/release/examples/perf_dictionary_automaton_vs_fst automaton sequential_insert
//! ./target/release/examples/perf_dictionary_automaton_vs_fst fst get_id
//! ./target/release/examples/perf_dictionary_automaton_vs_fst automaton get_id
//! ./target/release/examples/perf_dictionary_automaton_vs_fst fst get_term
//! ./target/release/examples/perf_dictionary_automaton_vs_fst automaton get_term
//! ./target/release/examples/perf_dictionary_automaton_vs_fst fst remove
//! ./target/release/examples/perf_dictionary_automaton_vs_fst automaton remove
//! ```
//!
//! Each mode prints its own wall-clock timing (only the operation under
//! test is inside the `Instant` window; setup -- preloading via
//! `insert_batch` -- happens before the clock starts). For `perf stat`
//! hardware-counter verification, that's not enough on its own: `perf
//! stat` times the whole process, preload included, and preload cost is
//! not always small relative to the timed loop (`get_term`'s preload is
//! the full corpus, for instance). Every mode that preloads has a
//! matching `<mode>_setup` variant that runs only the preload and exits
//! -- run both, subtract the `_setup` counters from the real mode's, and
//! divide by the op count for a clean per-op hardware-counter estimate.
//! `bulk_insert` needs no `_setup` variant: the dictionary starts empty,
//! so the whole process already is the measured operation.
//!
//! `n` (third arg, default 50000) is the number of *distinct-namespace
//! groups*: each group expands to 4 terms (dbpedia/foaf/isbn/blank-node),
//! matching this project's established namespace-clustered term shape
//! (chosen over sequential-digit-suffix terms, which were found earlier
//! in this project to manufacture artificial branching factor).

use std::time::Instant;

use s_dict::{AutomatonDictionary, Dictionary, TermDictionary};

fn namespace_clustered_terms_range(start: u32, count: u32) -> Vec<String> {
    (start..start + count)
        .flat_map(|i| {
            [
                format!("http://dbpedia.org/resource/Entity_{i}"),
                format!("http://xmlns.com/foaf/0.1/property{i}"),
                format!("urn:isbn:{i:013}"),
                format!("_:b{i:x}"),
            ]
        })
        .collect()
}

fn namespace_clustered_terms(n: u32) -> Vec<String> {
    namespace_clustered_terms_range(0, n)
}

fn report(label: &str, ops: usize, elapsed: std::time::Duration) {
    let per_op = elapsed / ops.max(1) as u32;
    eprintln!("{label}: {ops} ops in {elapsed:?} ({per_op:?}/op)");
}

fn bulk_insert(dict: &mut impl TermDictionary, n: u32) {
    let terms = namespace_clustered_terms(n);
    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();

    let start = Instant::now();
    let ids = dict.insert_batch(&refs);
    let elapsed = start.elapsed();

    std::hint::black_box(&ids);
    assert_eq!(dict.len(), refs.len());
    report("bulk_insert", refs.len(), elapsed);
}

fn get_id(dict: &mut impl TermDictionary, n: u32, setup_only: bool) {
    let terms = namespace_clustered_terms(n);
    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
    dict.insert_batch(&refs); // unmeasured preload
    if setup_only {
        std::hint::black_box(&dict);
        return;
    }

    let start = Instant::now();
    for t in &refs {
        std::hint::black_box(dict.get_id(t));
    }
    let elapsed = start.elapsed();
    report("get_id", refs.len(), elapsed);
}

fn get_term(dict: &mut impl TermDictionary, n: u32, setup_only: bool) {
    let terms = namespace_clustered_terms(n);
    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
    let ids = dict.insert_batch(&refs); // unmeasured preload
    if setup_only {
        std::hint::black_box(&dict);
        return;
    }

    let start = Instant::now();
    for &id in &ids {
        std::hint::black_box(dict.get_term(id));
    }
    let elapsed = start.elapsed();
    report("get_term", ids.len(), elapsed);
}

fn remove(dict: &mut impl TermDictionary, n: u32, setup_only: bool) {
    let terms = namespace_clustered_terms(n);
    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
    dict.insert_batch(&refs); // unmeasured preload
    if setup_only {
        std::hint::black_box(&dict);
        return;
    }

    let start = Instant::now();
    for t in &refs {
        std::hint::black_box(dict.remove(t));
    }
    let elapsed = start.elapsed();
    assert_eq!(dict.len(), 0);
    report("remove", refs.len(), elapsed);
}

fn sequential_insert_impl(dict: &mut impl TermDictionary, n: u32, setup_only: bool) {
    let baseline = namespace_clustered_terms(2_000);
    let baseline_refs: Vec<&str> = baseline.iter().map(String::as_str).collect();
    dict.insert_batch(&baseline_refs); // unmeasured preload
    if setup_only {
        std::hint::black_box(&dict);
        return;
    }

    let incoming = namespace_clustered_terms_range(2_000, n);
    let start = Instant::now();
    for t in &incoming {
        std::hint::black_box(dict.insert(t));
    }
    let elapsed = start.elapsed();

    assert_eq!(dict.len(), baseline_refs.len() + incoming.len());
    report("sequential_insert", incoming.len(), elapsed);
}

/// `<mode>_setup` variants run only the unmeasured preload and exit --
/// for `perf stat`, which measures the whole process, this lets the
/// preload's own hardware-counter cost be captured separately and
/// subtracted, the same isolation discipline `perf_insert_single_flush.rs`
/// already established (`setup_only` vs. the real mode). Needed because
/// `get_id`/`get_term`/`remove`/`sequential_insert` all preload data
/// whose cost is excluded from this binary's own `Instant`-based timing
/// but NOT excluded from an external `perf stat -- ./binary <mode>`
/// invocation, which times the whole process.
fn run(mode: &str, n: u32, dict: &mut impl TermDictionary) {
    match mode {
        "bulk_insert" => bulk_insert(dict, n),
        "sequential_insert" => sequential_insert_impl(dict, n, false),
        "sequential_insert_setup" => sequential_insert_impl(dict, n, true),
        "get_id" => get_id(dict, n, false),
        "get_id_setup" => get_id(dict, n, true),
        "get_term" => get_term(dict, n, false),
        "get_term_setup" => get_term(dict, n, true),
        "remove" => remove(dict, n, false),
        "remove_setup" => remove(dict, n, true),
        other => panic!(
            "unknown mode {other:?}; expected bulk_insert, sequential_insert(_setup), \
             get_id(_setup), get_term(_setup), or remove(_setup)"
        ),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dict_kind = args.next().expect("first arg: fst or automaton");
    let mode = args
        .next()
        .expect("second arg: bulk_insert, sequential_insert, get_id, get_term, or remove");
    let n: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(50_000);

    match dict_kind.as_str() {
        "fst" => run(&mode, n, &mut Dictionary::new()),
        "automaton" => run(&mode, n, &mut AutomatonDictionary::new()),
        other => panic!("unknown dict kind {other:?}; expected fst or automaton"),
    }
}
