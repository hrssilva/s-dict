//! Bounded-batch-size sweep with rebuild/decode counters, for the
//! "does batching reduce rebuild frequency, or just move the same rebuild
//! work around" question (DESIGN.md's "Prepared ingestion batches"
//! section; a `perf`-sampled profile outside this crate attributed ~40%
//! of CPU cycles during ingestion to block rebuilding).
//!
//! Each configuration: preload a fresh `Dictionary` with `PRELOAD` terms
//! via one big prepared batch (unmeasured setup), then stream `INCOMING`
//! *new*, disjoint terms through `TermBatch`, flushing (`prepare` +
//! `insert_prepared`) whenever the batch reaches `batch_limit` terms or
//! `byte_cap` bytes of text, whichever comes first. Measures, over the
//! streaming phase only (via `Dictionary::rebuild_stats` snapshots before
//! and after):
//!
//! - wall time
//! - blocks rebuilt and entries decoded, both total and per newly
//!   inserted term (the H5 batching-amortization signal)
//! - retained allocation growth (current live bytes, via a counting
//!   global allocator wrapping `System` -- not peak; peak is reported too
//!   but labeled separately, since a batch's transient decode/merge/build
//!   working set can spike well above what it leaves behind)
//! - RSS growth (`/proc/self/status` `VmRSS`; Linux only). RSS is a
//!   coarser, laggier signal than the allocator counters -- glibc doesn't
//!   always return freed pages to the OS promptly, so RSS can overstate
//!   what's actually retained. Reported alongside the allocator counters,
//!   not as a replacement for them.
//!
//! All configurations' results are collected into one `Vec<SweepResult>`
//! before anything is printed, so the final report is one aligned table,
//! not per-run prints that can't be compared against each other.
//!
//! Terms are synthetic (no real RDF corpus is vendored into this crate;
//! see the sibling `s-bench` harness for real-corpus profiling). Deliberately
//! *not* a single sequential numeric suffix pattern -- DESIGN.md's
//! "Correction" section found that shape manufactures artificial branching
//! factor that doesn't generalize. Terms here cycle through a handful of
//! differently-shaped namespaces/lengths instead. That said, this sweep
//! measures rebuild/allocation bookkeeping cost, not FST branching-factor-
//! sensitive decode cost, so it should be much less sensitive to exact
//! term shape than the earlier binary-search finding was.
//!
//! Usage: `cargo run --release --example batch_sweep [preload] [incoming]`
//! (defaults: 200,000 / 200,000).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;

use s_dict::{Dictionary, TermBatch, TermDictionary};

struct CountingAlloc;

static CURRENT_BYTES: AtomicI64 = AtomicI64::new(0);
static PHASE_PEAK_BYTES: AtomicI64 = AtomicI64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = CURRENT_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed)
                + layout.size() as i64;
            PHASE_PEAK_BYTES.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT_BYTES.fetch_sub(layout.size() as i64, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let delta = new_size as i64 - layout.size() as i64;
            let now = CURRENT_BYTES.fetch_add(delta, Ordering::Relaxed) + delta;
            PHASE_PEAK_BYTES.fetch_max(now, Ordering::Relaxed);
        }
        new_ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            let now = CURRENT_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed)
                + layout.size() as i64;
            PHASE_PEAK_BYTES.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn read_rss_kb() -> i64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0);
        }
    }
    0
}

/// Cycles through a handful of differently-shaped, differently-sized
/// namespaces instead of one sequential numeric pattern (see module docs).
/// Injective in `i`, so disjoint index ranges guarantee disjoint terms.
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

#[derive(Clone, Copy)]
enum Cap {
    /// Flush once the batch reaches this many terms.
    Terms(usize),
    /// Flush once the batch's text arena reaches this many bytes.
    Bytes(usize),
    /// Flush once either limit is reached, whichever comes first.
    Both(usize, usize),
}

impl Cap {
    fn should_flush(self, batch: &TermBatch) -> bool {
        match self {
            Cap::Terms(n) => batch.len() >= n,
            Cap::Bytes(b) => batch.text_bytes() >= b,
            Cap::Both(n, b) => batch.len() >= n || batch.text_bytes() >= b,
        }
    }

    fn label(self) -> String {
        match self {
            Cap::Terms(n) => format!("terms<={n}"),
            Cap::Bytes(b) => format!("bytes<={}", fmt_bytes(b as i64)),
            Cap::Both(n, b) => format!("terms<={n},bytes<={}", fmt_bytes(b as i64)),
        }
    }
}

struct SweepResult {
    label: String,
    flushes: u64,
    elapsed: std::time::Duration,
    blocks_rebuilt: u64,
    entries_decoded: u64,
    new_terms: u64,
    retained_bytes: i64,
    phase_peak_bytes: i64,
    rss_before_kb: i64,
    rss_after_kb: i64,
}

fn run_config(label: String, preload: &[String], incoming: &[String], cap: Cap) -> SweepResult {
    let mut dict = Dictionary::new();
    let mut batch = TermBatch::new();

    // Unmeasured setup: preload via one prepared batch (bootstrap fast path).
    for t in preload {
        batch.push(t);
    }
    batch.prepare();
    dict.insert_prepared(batch.prepared().unwrap());
    batch.clear();

    let stats_before = dict.rebuild_stats();
    PHASE_PEAK_BYTES.store(CURRENT_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
    let alloc_before = CURRENT_BYTES.load(Ordering::Relaxed);
    let rss_before_kb = read_rss_kb();
    let start = Instant::now();

    let mut flushes = 0u64;
    for t in incoming {
        batch.push(t);
        if cap.should_flush(&batch) {
            batch.prepare();
            dict.insert_prepared(batch.prepared().unwrap());
            batch.clear();
            flushes += 1;
        }
    }
    if !batch.is_empty() {
        batch.prepare();
        dict.insert_prepared(batch.prepared().unwrap());
        batch.clear();
        flushes += 1;
    }

    let elapsed = start.elapsed();
    let stats_after = dict.rebuild_stats();
    let alloc_after = CURRENT_BYTES.load(Ordering::Relaxed);
    let phase_peak_bytes = PHASE_PEAK_BYTES.load(Ordering::Relaxed);
    let rss_after_kb = read_rss_kb();

    let new_terms = dict.len() as u64 - preload.len() as u64;
    assert_eq!(
        new_terms,
        incoming.len() as u64,
        "incoming terms must be disjoint from preload for this to measure what it claims to"
    );

    SweepResult {
        label,
        flushes,
        elapsed,
        blocks_rebuilt: stats_after.blocks_rebuilt - stats_before.blocks_rebuilt,
        entries_decoded: stats_after.entries_decoded - stats_before.entries_decoded,
        new_terms,
        retained_bytes: alloc_after - alloc_before,
        phase_peak_bytes,
        rss_before_kb,
        rss_after_kb,
    }
}

fn fmt_bytes(b: i64) -> String {
    let sign = if b < 0 { "-" } else { "" };
    let b = b.unsigned_abs();
    if b >= 1 << 20 {
        format!("{sign}{:.1}MiB", b as f64 / (1 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{sign}{:.1}KiB", b as f64 / (1 << 10) as f64)
    } else {
        format!("{sign}{b}B")
    }
}

fn print_report(results: &[SweepResult]) {
    println!(
        "{:<24} {:>9} {:>10} {:>13} {:>13} {:>9} {:>9} {:>11} {:>11} {:>10} {:>10}",
        "config",
        "flushes",
        "time",
        "blocks_rblt",
        "entries_dec",
        "dec/new",
        "rblt/new",
        "retained",
        "phase_peak",
        "rss_before",
        "rss_after"
    );
    for r in results {
        println!(
            "{:<24} {:>9} {:>10.3?} {:>13} {:>13} {:>9.2} {:>9.4} {:>11} {:>11} {:>8}KiB {:>8}KiB",
            r.label,
            r.flushes,
            r.elapsed,
            r.blocks_rebuilt,
            r.entries_decoded,
            r.entries_decoded as f64 / r.new_terms.max(1) as f64,
            r.blocks_rebuilt as f64 / r.new_terms.max(1) as f64,
            fmt_bytes(r.retained_bytes),
            fmt_bytes(r.phase_peak_bytes),
            r.rss_before_kb,
            r.rss_after_kb,
        );
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let preload_n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let incoming_n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200_000);

    // Disjoint index ranges -> disjoint terms (make_term is injective), so
    // "new_terms" below is exactly incoming_n, not inflated by dedup.
    let preload: Vec<String> = (0..preload_n as u64).map(make_term).collect();
    let incoming: Vec<String> = (preload_n as u64..(preload_n + incoming_n) as u64)
        .map(make_term)
        .collect();

    let avg_term_bytes = incoming.iter().map(|t| t.len()).sum::<usize>() / incoming.len().max(1);
    println!(
        "preload={preload_n} incoming={incoming_n} avg_incoming_term_bytes={avg_term_bytes}\n"
    );

    let mut results = Vec::new();

    println!("-- sweep 1: batch term-count limit (byte cap disabled) --");
    for &n in &[1usize, 4, 16, 64, 256, 1024, 4096, 16384, 65536] {
        let cap = Cap::Terms(n);
        results.push(run_config(cap.label(), &preload, &incoming, cap));
    }

    println!("-- sweep 2: batch byte cap (term-count cap disabled) --");
    for &b in &[1usize << 10, 1 << 14, 1 << 16, 1 << 20, 1 << 24] {
        let cap = Cap::Bytes(b);
        results.push(run_config(cap.label(), &preload, &incoming, cap));
    }

    println!("-- sweep 3: combined (byte cap binds before a generous term cap) --");
    for &b in &[1usize << 12, 1 << 16] {
        let cap = Cap::Both(65536, b);
        results.push(run_config(cap.label(), &preload, &incoming, cap));
    }

    println!();
    print_report(&results);
}
