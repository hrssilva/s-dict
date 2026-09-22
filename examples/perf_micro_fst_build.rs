//! Isolates *construction* specifically -- `Automaton::build_sorted` vs
//! `Block::build` -- from everything this project's own measurement
//! history has already found to be a confound at higher levels:
//! `BatchOrder::prepare`'s sort (shared, identical code on both the
//! `fst` and automaton sides, see DESIGN.md's "closing the
//! bulk-construction gap" section) and, for a `perf record`/`perf stat`
//! whole-process capture specifically, this binary's own term-generation
//! and chunking setup. Every mode here has a `_setup` counterpart that
//! does only that setup and exits, so hardware counters can be isolated
//! by subtraction, exactly this crate's established `perf stat`
//! discipline (see `perf_insert_single_flush.rs`).
//!
//! Usage: `cargo build --release --example perf_micro_fst_build` (or
//! `--profile profiling` for a `perf record` call graph with symbols),
//! then for each `(kind, mode)` pair, in its own process:
//!
//! ```sh
//! ./target/release/examples/perf_micro_fst_build automaton build
//! ./target/release/examples/perf_micro_fst_build automaton setup
//! ./target/release/examples/perf_micro_fst_build block build
//! ./target/release/examples/perf_micro_fst_build block setup
//! ```
//!
//! Chunked to `chunk_size`-term pieces (default 256, `Dictionary`'s own
//! default `block_max`) to match `AutomatonDictionary::bootstrap`'s real
//! call shape -- one call per final block, not one giant build.

use std::time::Instant;

fn namespace_clustered_terms(n: u32) -> Vec<String> {
    (0..n)
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

fn build_chunks(total_terms: u32, chunk_size: usize) -> Vec<Vec<String>> {
    let mut terms = namespace_clustered_terms(total_terms / 4);
    terms.sort_unstable();
    terms.dedup();
    terms.chunks(chunk_size).map(<[String]>::to_vec).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let kind = args.next().expect("first arg: automaton or block");
    let mode = args.next().expect("second arg: build or setup");
    let total_terms: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(80_000);
    let chunk_size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);

    let chunks = build_chunks(total_terms, chunk_size);
    let total: usize = chunks.iter().map(Vec::len).sum();

    if mode == "setup" {
        std::hint::black_box(&chunks);
        eprintln!("setup only: {} chunks, {total} terms", chunks.len());
        return;
    }
    if mode != "build" {
        panic!("unknown mode {mode:?}; expected build or setup");
    }

    let start = Instant::now();
    match kind.as_str() {
        "automaton" => {
            for chunk in &chunks {
                let automaton = s_dict::__internal_bench::build_automaton(chunk);
                std::hint::black_box(&automaton);
            }
        }
        "block" => {
            for chunk in &chunks {
                let block = s_dict::__internal_bench::build_block(chunk);
                std::hint::black_box(&block);
            }
        }
        other => panic!("unknown kind {other:?}; expected automaton or block"),
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{kind} build, {} chunks x ~{chunk_size} terms ({total} total): {elapsed:?} ({:?}/term)",
        chunks.len(),
        elapsed / total as u32,
    );
}
