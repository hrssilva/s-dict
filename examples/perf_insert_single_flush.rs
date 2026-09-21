//! Focused `perf stat` target for the single-walk-vs-two-step question:
//! insert-one-new-term-at-a-time (`TermBatch` flushed at 1 term per
//! `insert_prepared` call, matching `batch_sweep`'s `terms<=1` config)
//! into an already-preloaded dictionary. Isolated from the rest of
//! `batch_sweep`'s many configs so a single `perf stat` run measures
//! exactly this path, and hardware counters (cycles, instructions) are
//! used instead of wall-clock time, since they're far less sensitive to
//! this machine's session-to-session scheduling/frequency-scaling noise
//! (established earlier in DESIGN.md's measurement-hygiene notes).
//!
//! Usage: `cargo build --profile profiling --example perf_insert_single_flush`, then
//!
//! ```sh
//! perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
//!     -- ./target/profiling/examples/perf_insert_single_flush setup_only 50000 50000
//! perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
//!     -- ./target/profiling/examples/perf_insert_single_flush insert 50000 50000
//! ```
//!
//! then subtract `setup_only`'s counts from `insert`'s and divide by the
//! incoming count for a per-op estimate.

use s_dict::{Dictionary, TermBatch, TermDictionary};

fn make_term(i: u64) -> String {
    match i % 7 {
        0 => format!("http://dbpedia.org/resource/Entity_{i:x}"),
        1 => format!("http://xmlns.com/foaf/0.1/property{i:x}"),
        2 => format!("urn:isbn:{i:013}"),
        3 => format!(
            "\"A longer literal value describing item {i} with padding text to vary length near typical RDF literal sizes\"@en"
        ),
        4 => format!("_:b{i:x}"),
        5 => format!("http://www.w3.org/1999/02/22-rdf-syntax-ns#_{i}"),
        _ => format!("https://schema.org/Thing/{i:x}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "insert".to_string());
    let preload_n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let incoming_n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50_000);

    let preload: Vec<String> = (0..preload_n as u64).map(make_term).collect();
    let incoming: Vec<String> = (preload_n as u64..(preload_n + incoming_n) as u64)
        .map(make_term)
        .collect();

    let mut dict = Dictionary::new();
    let mut batch = TermBatch::new();
    for t in &preload {
        batch.push(t);
    }
    batch.prepare();
    dict.insert_prepared(batch.prepared().unwrap());
    batch.clear();

    match mode.as_str() {
        "setup_only" => {
            std::hint::black_box(&dict);
        }
        "insert" => {
            for t in &incoming {
                batch.push(t);
                batch.prepare();
                let ids = dict.insert_prepared(batch.prepared().unwrap());
                batch.clear();
                std::hint::black_box(&ids);
            }
            assert_eq!(dict.len(), preload_n + incoming_n);
        }
        other => panic!("unknown mode {other:?}; expected setup_only or insert"),
    }
}
