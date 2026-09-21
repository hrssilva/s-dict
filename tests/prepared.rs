use s_dict::{Dictionary, PlainDictionary, TermBatch, TermDictionary};

#[test]
fn prepared_batches_preserve_original_order_and_invalidate_on_mutation() {
    let mut batch = TermBatch::new();
    let mut dict = Dictionary::with_block_bounds(2, 8);
    assert!(batch.prepared().is_none());
    let terms = ["z", "é", "", "z", "中", "a", "é"];
    for term in terms {
        batch.push(term);
    }
    batch.prepare();
    batch.prepare();
    let ids = dict.insert_prepared(batch.prepared().unwrap());
    for (term, id) in terms.into_iter().zip(&ids) {
        assert_eq!(dict.get_id(term), Some(*id));
        assert_eq!(dict.get_term(*id).as_deref(), Some(term));
    }
    assert_eq!(ids[0], ids[3]);
    assert_eq!(ids[1], ids[6]);
    batch.push("later");
    assert!(batch.prepared().is_none());
    batch.prepare();
    assert_eq!(
        &dict.insert_prepared(batch.prepared().unwrap())[..terms.len()],
        &ids
    );
    batch.clear();
    assert!(batch.is_empty());
    assert_eq!(batch.text_bytes(), 0);
    assert!(batch.prepared().is_none());
    batch.prepare();
    assert!(dict.insert_prepared(batch.prepared().unwrap()).is_empty());
}

#[test]
fn mixed_prepared_regular_batches_and_removals_match_plain_dictionary() {
    for max in [4, 32, 256] {
        let mut dict = Dictionary::with_block_bounds(max / 2, max);
        let mut plain = PlainDictionary::new();
        let mut batch = TermBatch::new();
        let mut rng = 789123u64;
        let pool: Vec<_> = (0..1024).map(|i| format!("urn:é:term:{i:04}")).collect();
        for step in 0..200 {
            let mut refs = Vec::new();
            for _ in 0..(if step % 3 == 0 { 400 } else { 7 }) {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                refs.push(pool[rng as usize % pool.len()].as_str());
            }
            if step % 5 == 0 {
                for term in &refs {
                    assert_eq!(dict.remove(term), plain.remove(term));
                }
            } else {
                // The dictionary API assigns new IDs in lexical order.
                let mut sorted = refs.clone();
                sorted.sort_unstable();
                sorted.dedup();
                for term in sorted {
                    plain.insert(term);
                }
                let ids = if step % 2 == 0 {
                    dict.insert_batch(&refs)
                } else {
                    batch.clear();
                    for term in &refs {
                        batch.push(term);
                    }
                    batch.prepare();
                    dict.insert_prepared(batch.prepared().unwrap())
                };
                for (term, id) in refs.iter().zip(ids) {
                    assert_eq!(plain.get_id(term), Some(id));
                    assert_eq!(dict.get_term(id).as_deref(), Some(*term));
                }
            }
            assert_eq!(dict.len(), plain.len());
            for term in &pool {
                assert_eq!(dict.get_id(term), plain.get_id(term));
            }
        }
    }
}

#[test]
fn small_fst_caches_preserve_default_builder_semantics() {
    let mut terms: Vec<_> = (0..1024)
        .map(|i| format!("urn:{}:é:{:08x}", i % 17, i * 7919))
        .collect();
    terms.extend(["".to_owned(), "中".to_owned(), "a".repeat(4096)]);
    terms.sort_unstable();
    let mut baseline = fst::MapBuilder::new(Vec::new()).unwrap();
    for (i, term) in terms.iter().enumerate() {
        baseline.insert(term, i as u64).unwrap();
    }
    let baseline = fst::Map::new(baseline.into_inner().unwrap()).unwrap();
    for rows in [0, 1, 64, 1024, 10_000] {
        let mut builder = fst::MapBuilder::with_registry_rows(Vec::new(), rows).unwrap();
        for (i, term) in terms.iter().enumerate() {
            builder.insert(term, i as u64).unwrap();
        }
        let map = fst::Map::new(builder.into_inner().unwrap()).unwrap();
        assert_eq!(map.len(), baseline.len());
        for term in &terms {
            assert_eq!(map.get(term), baseline.get(term));
        }
        assert_eq!(map.get("absent"), None);
    }
}
