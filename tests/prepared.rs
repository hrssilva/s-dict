mod common;

use common::Model;
use s_dict::{PackedAutomatonDictionary, TermBatch, TermDictionary};

#[test]
fn prepared_batches_preserve_original_order_and_invalidate_on_mutation() {
    let mut batch = TermBatch::new();
    let mut dict = PackedAutomatonDictionary::with_block_bounds(2, 8);
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
    assert_eq!(&dict.insert_prepared(batch.prepared().unwrap())[..terms.len()], &ids);
    batch.clear();
    assert!(batch.is_empty());
    assert_eq!(batch.text_bytes(), 0);
    assert!(batch.prepared().is_none());
    batch.prepare();
    assert!(dict.insert_prepared(batch.prepared().unwrap()).is_empty());
}

#[test]
fn mixed_prepared_regular_batches_and_removals_match_model() {
    for max in [4, 32, 256] {
        let mut dict = PackedAutomatonDictionary::with_block_bounds(max / 2, max);
        let mut model = Model::new();
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
                    assert_eq!(dict.remove(term), model.remove(term));
                }
            } else {
                let expected = model.insert_batch(&refs);
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
                assert_eq!(ids, expected);
            }
            assert_eq!(dict.len(), model.len());
            for term in &pool {
                assert_eq!(dict.get_id(term), model.get_id(term));
            }
        }
    }
}
