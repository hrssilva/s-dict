mod common;

use common::Model;
use s_dict::{PackedAutomatonDictionary, TermBatch, TermDictionary};

/// Ids come back in input order (duplicates share one); a prepared batch
/// is changed only by going back to a `TermBatch` (`into_batch`), and
/// re-preparing it keeps earlier ids.
#[test]
fn prepared_batches_preserve_original_order_and_round_trip() {
    let mut dict = PackedAutomatonDictionary::with_block_bounds(2, 8);
    let terms = ["z", "é", "", "z", "中", "a", "é"];
    let mut batch = TermBatch::new();
    for term in terms {
        batch.push(term);
    }
    let prepared = batch.prepare();
    assert_eq!(prepared.len(), terms.len());
    let ids = dict.insert_prepared(&prepared);
    for (term, id) in terms.into_iter().zip(&ids) {
        assert_eq!(dict.get_id(term), Some(*id));
        assert_eq!(dict.get_term(*id).as_deref(), Some(term));
    }
    assert_eq!(ids[0], ids[3]);
    assert_eq!(ids[1], ids[6]);
    let mut batch = prepared.into_batch();
    batch.push("later");
    let prepared = batch.prepare();
    assert_eq!(&dict.insert_prepared(&prepared)[..terms.len()], &ids);
    let mut batch = prepared.into_batch();
    batch.clear();
    assert!(batch.is_empty());
    assert_eq!(batch.text_bytes(), 0);
    assert!(dict.insert_prepared(&batch.prepare()).is_empty());
    assert!(dict.insert_prepared(&Default::default()).is_empty());
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
                    let mut filling = std::mem::take(&mut batch);
                    filling.clear();
                    for term in &refs {
                        filling.push(term);
                    }
                    let prepared = filling.prepare();
                    let ids = dict.insert_prepared(&prepared);
                    batch = prepared.into_batch();
                    ids
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
