//! A `perf`-profiling target for decoding *many* terms in one pass --
//! `get_term` vs. `get_term_into` at a scale where per-op nanosecond
//! differences compound into a real, measurable total (e.g. dumping or
//! exporting an entire dictionary's terms). Complements
//! `perf_get_term.rs`'s 20,000-term/many-repeats measurement with a
//! single-pass, millions-of-terms one.
//!
//! Builds via `insert_batch` (the bootstrap fast path), not sequential
//! `insert`: sequential insert's ~90us/op rebuild cost would make
//! multi-million-term setup itself take minutes and would mix rebuild-cost
//! variance into what's supposed to be a clean decode-only measurement.
//!
//! Usage: `cargo build --release --example perf_decode_bulk`, then e.g.
//!
//! ```sh
//! perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
//!     -- ./target/release/examples/perf_decode_bulk setup_only 2000000
//! perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
//!     -- ./target/release/examples/perf_decode_bulk get_term 2000000
//! perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
//!     -- ./target/release/examples/perf_decode_bulk get_term_into 2000000
//! ```
//!
//! then subtract `setup_only`'s counts from each of the other two totals
//! and divide by `count` for a per-op estimate, same as `perf_get_term.rs`.

use s_dict::{Dictionary, TermDictionary};

fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "get_term".to_string());
    let count: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    // Repeats the loop `repeats` times over the same `count` ids. Default 1
    // matches the documented single-pass 2,000,000-term measurement;
    // `perf record` call-graph profiling wants this raised (e.g. to 20-50
    // at a smaller `count`) so the loop's share of total process runtime
    // swamps the one-time bootstrap setup cost, which otherwise shows up
    // in the sampled profile too (setup and a single loop pass are
    // comparable in duration, both O(count)).
    let repeats: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let terms: Vec<String> = (0..count)
        .map(|i| format!("http://example.org/term-{i:08}"))
        .collect();
    let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();

    let mut dict = Dictionary::new(); // default block bounds 64..=256
    let ids = dict.insert_batch(&term_refs); // fast bootstrap path, not N rebuilds

    match mode.as_str() {
        "setup_only" => {
            std::hint::black_box(&dict);
        }
        "get_term" => {
            for _ in 0..repeats {
                for &id in &ids {
                    std::hint::black_box(dict.get_term(id));
                }
            }
        }
        "get_term_into" => {
            let mut buf = Vec::new();
            for _ in 0..repeats {
                for &id in &ids {
                    dict.get_term_into(id, &mut buf);
                    std::hint::black_box(&buf);
                }
            }
        }
        // "Dump everything" case: requested ids span (nearly) every block,
        // so get_terms_into's per-block decode is amortized over many ids.
        // Compare against the plain "get_term_into" mode above (same ids,
        // repeated single-id calls) as the baseline.
        "get_terms_into" => {
            let mut out = Vec::new();
            for _ in 0..repeats {
                dict.get_terms_into(&ids, &mut out);
                std::hint::black_box(&out);
            }
        }
        // Scattered case: one id sampled every 300 (> block_max=256), so
        // consecutive requests almost always land in different blocks --
        // the case get_terms_into is expected to lose in.
        "get_terms_into_scattered" => {
            let scattered: Vec<_> = ids.iter().step_by(300).copied().collect();
            eprintln!("scattered sample size: {}", scattered.len());
            let mut out = Vec::new();
            for _ in 0..repeats {
                dict.get_terms_into(&scattered, &mut out);
                std::hint::black_box(&out);
            }
        }
        "get_term_into_scattered" => {
            let scattered: Vec<_> = ids.iter().step_by(300).copied().collect();
            eprintln!("scattered sample size: {}", scattered.len());
            let mut buf = Vec::new();
            for _ in 0..repeats {
                for &id in &scattered {
                    dict.get_term_into(id, &mut buf);
                    std::hint::black_box(&buf);
                }
            }
        }
        other => panic!(
            "unknown mode {other:?}; expected setup_only, get_term, get_term_into, \
             get_terms_into, get_terms_into_scattered, or get_term_into_scattered"
        ),
    }
}
