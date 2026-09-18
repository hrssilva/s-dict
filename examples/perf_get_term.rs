//! A dedicated `perf`-profiling target for `get_term` vs. `get_term_into`,
//! separate from the crate's `#[ignore]`d wall-clock probes: `perf stat`
//! measures a whole process, so this binary's `setup_only` mode exists to
//! be subtracted from the `get_term`/`get_term_into` runs' totals,
//! canceling out the (otherwise dominant) insert-time setup cost from the
//! per-loop hardware-counter deltas.
//!
//! Usage: `cargo build --release --example perf_get_term`, then e.g.
//!
//! ```sh
//! perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
//!     -- ./target/release/examples/perf_get_term setup_only 400
//! perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
//!     -- ./target/release/examples/perf_get_term get_term 400
//! perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
//!     -- ./target/release/examples/perf_get_term get_term_into 400
//! ```
//!
//! then subtract `setup_only`'s counts from each of the other two totals
//! and divide by `repeats * 20_000` for a per-op estimate.

use s_dict::{Dictionary, TermDictionary, TermId};

fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "get_term".to_string());
    let repeats: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);

    let terms: Vec<String> = (0..20_000)
        .map(|i| format!("http://example.org/term-{i:06}"))
        .collect();
    let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();

    let mut dict = Dictionary::new(); // default block bounds 64..=256
    for t in &term_refs {
        dict.insert(t);
    }
    let ids: Vec<TermId> = term_refs.iter().map(|t| dict.get_id(t).unwrap()).collect();

    match mode.as_str() {
        "setup_only" => {
            // No loop: this run's totals are the "insert 20,000 terms"
            // baseline to subtract from the other two modes.
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
        other => panic!("unknown mode {other:?}; expected setup_only, get_term, or get_term_into"),
    }
}
