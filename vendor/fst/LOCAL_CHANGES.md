This is fst 0.4.7 from the Cargo registry, under its original MIT/Unlicense
terms (see LICENSE-MIT and UNLICENSE).

Local changes: raw::Builder::new_type_with_registry_rows and
MapBuilder::with_registry_rows expose the existing two-way registry row
count. Existing constructors retain their original 10,000-row default.
The FST format and query code are unchanged. Small caches may produce larger
FSTs due to evicted state reuse opportunities; they preserve key/value semantics.

The packaged data fixtures and benchmark directory are omitted. This copy
is used as a dependency; s-dict tests exercise the new constructor and compare
its results with the original constructor.

The FstRef return lifetime is explicit for current Rust compiler diagnostics.
