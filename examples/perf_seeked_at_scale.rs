//! Measures `PackedAutomatonDictionary::get_terms_seeked_into` end to end
//! on a synthetic corpus (the generators `perf_scale_memory.rs` uses),
//! queried with scattered clusters of consecutive-rank ids -- the shape the
//! batched decode exists for.
//!
//! Usage: `cargo run --release --example perf_seeked_at_scale -- [clustered|hashlike] [total] [num_clusters] [cluster_size]`

use s_dict::{PackedAutomatonDictionary, TermDictionary, TermId};

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

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn hashlike_terms_range(start: u32, count: u32) -> Vec<String> {
    (start..start + count)
        .map(|i| {
            let h = splitmix64(i as u64);
            format!("urn:uuid:{h:016x}")
        })
        .collect()
}

/// A small deterministic xorshift64* PRNG (matching `tests/dictionary.rs`'s
/// own `Lcg`), avoiding a `rand` dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }

    fn next_range(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// `num_clusters` scattered starting points in `sorted_ids` (already in
/// rank/lexical order), each contributing `cluster_size` *consecutive*
/// ids -- the "seeked" shape: mostly-one-block clusters, scattered
/// across the whole dictionary, not "every id" (already covered by a
/// full `decode_into`-style streamed dump) and not "one independent id
/// at a time" (already covered by `get_term_into`).
fn build_query(sorted_ids: &[TermId], num_clusters: usize, cluster_size: usize) -> Vec<TermId> {
    let mut rng = Lcg::new(0x2545_F491_4F6C_DD1D);
    let mut query = Vec::with_capacity(num_clusters * cluster_size);
    for _ in 0..num_clusters {
        let start = rng.next_range(sorted_ids.len() - cluster_size);
        query.extend_from_slice(&sorted_ids[start..start + cluster_size]);
    }
    query
}

const QUERY_REPEATS: usize = 10;

fn main() {
    let mut args = std::env::args().skip(1);
    let generator = args.next().unwrap_or_else(|| "clustered".to_string());
    let total: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(300_000);
    let num_clusters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let cluster_size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let per_group = if generator == "clustered" { 4 } else { 1 };
    let total_groups = total / per_group;

    let gen_range: Box<dyn Fn(u32, u32) -> Vec<String>> = match generator.as_str() {
        "clustered" => Box::new(namespace_clustered_terms_range),
        "hashlike" => Box::new(hashlike_terms_range),
        other => panic!("unknown generator {other:?}; expected clustered or hashlike"),
    };
    let terms = gen_range(0, total_groups);
    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();

    let mut dict = PackedAutomatonDictionary::new();
    let build_start = std::time::Instant::now();
    let ids = dict.insert_batch(&refs);
    let build_elapsed = build_start.elapsed();

    let mut sorted: Vec<(&str, TermId)> = refs.iter().copied().zip(ids).collect();
    sorted.sort_unstable_by_key(|&(t, _)| t);
    let sorted_ids: Vec<TermId> = sorted.iter().map(|&(_, id)| id).collect();
    let query = build_query(&sorted_ids, num_clusters, cluster_size);

    let mut out = Vec::new();
    let start = std::time::Instant::now();
    for _ in 0..QUERY_REPEATS {
        dict.get_terms_seeked_into(&query, &mut out);
        std::hint::black_box(&out);
    }
    let elapsed = start.elapsed();
    eprintln!(
        "PackedAutomatonDictionary: {} terms, {} blocks, build={build_elapsed:?}\n\
         get_terms_seeked_into: {} ids ({num_clusters} clusters x {cluster_size}) x {QUERY_REPEATS} repeats: \
         {elapsed:?} ({:?}/op)",
        dict.len(),
        dict.block_count(),
        query.len(),
        elapsed / (QUERY_REPEATS * query.len()) as u32,
    );
}
