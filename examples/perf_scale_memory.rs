//! Reproduces (or rules out reproducing) the memory-scaling issue an
//! external benchmark found: `AutomatonDictionary::size_bytes()` growing
//! *super-linearly* with corpus size (~105 bytes/term at 524K terms ->
//! ~258 bytes/term at 4.58M terms), while `Dictionary::size_bytes()`
//! stays flat (~32 bytes/term at both scales). `size_bytes()` on both
//! sides already excludes incidental bookkeeping (register/refcounts on
//! the automaton side) -- if growth reproduces here too, it's in the
//! true minimal reachable structure itself, not slot-reclamation
//! garbage.
//!
//! Inserted via `insert_batch` (a realistic bulk-load shape, matching
//! how a real triple-store load would populate this), at increasing
//! checkpoints, printing `size_bytes() / len()` at each one so a trend
//! (or its absence) is visible directly, not just at two endpoints.
//!
//! Usage: `cargo run --release --example perf_scale_memory -- automaton|fst|packed [clustered|hashlike] [total] [checkpoints]`

use s_dict::{AutomatonDictionary, Dictionary, PackedAutomatonDictionary, TermDictionary};

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

/// A cheap, deterministic 64-bit mix (splitmix64) -- not for security,
/// just to turn a sequential index into a value with no cross-term
/// structure, unlike `namespace_clustered_terms_range`'s sequential
/// numeric suffixes (which share leading digits/padding with their
/// neighbors). Meant to approximate hash-based identifiers (blank node
/// labels, UUIDs) real RDF data sometimes has plenty of, where a term's
/// *suffix* shares almost nothing with its lexical neighbors even though
/// a short prefix still does.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Low cross-term redundancy by design: a short, fully-shared prefix
/// (`urn:uuid:`) followed by 16 hex digits derived from an
/// unrelated-looking hash of the index -- almost no shared structure
/// between consecutive (or any two) terms' suffixes, unlike sequential
/// integers. Isolates whether automaton's per-state overhead scales
/// worse than `fst`'s specifically under low redundancy.
fn hashlike_terms_range(start: u32, count: u32) -> Vec<String> {
    (start..start + count)
        .map(|i| {
            let h = splitmix64(i as u64);
            format!("urn:uuid:{h:016x}")
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let kind = args.next().expect("first arg: automaton, fst, or packed");
    let generator = args.next().unwrap_or_else(|| "clustered".to_string());
    let total: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let checkpoints: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let per_group = if generator == "clustered" { 4 } else { 1 };
    let total_groups = total / per_group;
    let chunk = (total_groups / checkpoints).max(1);

    let gen_range: Box<dyn Fn(u32, u32) -> Vec<String>> = match generator.as_str() {
        "clustered" => Box::new(namespace_clustered_terms_range),
        "hashlike" => Box::new(hashlike_terms_range),
        other => panic!("unknown generator {other:?}; expected clustered or hashlike"),
    };

    match kind.as_str() {
        "automaton" => {
            let mut dict = AutomatonDictionary::new();
            for cp in 1..=checkpoints {
                let terms = gen_range((cp - 1) * chunk, chunk);
                let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
                dict.insert_batch(&refs);
                let len = dict.len();
                let bytes = dict.size_bytes();
                let states = dict.reachable_state_count();
                eprintln!(
                    "{len:>10} terms  size_bytes={bytes:>14}  bytes/term={:.2}  \
                     states={states:>10}  states/term={:.3}  blocks={}",
                    bytes as f64 / len as f64,
                    states as f64 / len as f64,
                    dict.block_count(),
                );
            }
        }
        "fst" => {
            let mut dict = Dictionary::new();
            for cp in 1..=checkpoints {
                let terms = gen_range((cp - 1) * chunk, chunk);
                let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
                dict.insert_batch(&refs);
                let len = dict.len();
                let bytes = dict.size_bytes();
                let states = dict.reachable_state_count();
                eprintln!(
                    "{len:>10} terms  size_bytes={bytes:>14}  bytes/term={:.2}  \
                     states={states:>10}  states/term={:.3}",
                    bytes as f64 / len as f64,
                    states as f64 / len as f64,
                );
            }
        }
        "packed" => {
            let mut dict = PackedAutomatonDictionary::new();
            for cp in 1..=checkpoints {
                let terms = gen_range((cp - 1) * chunk, chunk);
                let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
                dict.insert_batch(&refs);
                let len = dict.len();
                let bytes = dict.size_bytes();
                let states = dict.reachable_state_count();
                eprintln!(
                    "{len:>10} terms  size_bytes={bytes:>14}  bytes/term={:.2}  \
                     states={states:>10}  states/term={:.3}  blocks={}",
                    bytes as f64 / len as f64,
                    states as f64 / len as f64,
                    dict.block_count(),
                );
            }
        }
        other => panic!("unknown kind {other:?}; expected automaton, fst, or packed"),
    }
}
