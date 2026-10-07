//! `TermId -> block slot` as ranges: the `PackedAutomatonDictionary`
//! reverse map.
//!
//! A batch assigns consecutive ids, in key order, to the terms it adds to
//! one block, so ids come in runs that live in one block (~70 ids per run
//! on real RDF; DESIGN.md's "global run -> block map"). This stores one
//! breakpoint per run instead of anything per id: `(first id, slot)`, the
//! value at `x` being the last breakpoint `<= x`, with explicit "none"
//! breakpoints over unassigned ids. Ranks are not stored at all -- a block
//! answers `id -> rank` itself (`BlockIds::rank_of`).
//!
//! Splits, neighbor merges and removals edit ranges in the middle, so the
//! breakpoints live in leaves of at most `LEAF_MAX` (a two-level sorted
//! structure, like a B-tree of height two): an edit moves at most one
//! leaf's entries, a lookup is two binary searches.

use std::num::NonZeroU32;

/// A breakpoint's value: the block slot, stored as `slot + 1` so that "no
/// block" is `None` (and `Option<Slot>` stays 4 bytes) rather than a
/// sentinel slot number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Slot(NonZeroU32);

impl Slot {
    #[inline]
    fn new(slot: u32) -> Slot {
        debug_assert!(slot < u32::MAX, "block slots stay below u32::MAX");
        Slot(NonZeroU32::MIN.saturating_add(slot))
    }

    #[inline]
    fn get(self) -> u32 {
        self.0.get() - 1
    }
}
const LEAF_MAX: usize = 512;

#[derive(Debug, Clone, Default)]
pub(crate) struct RangeMap {
    /// Breakpoints `(start, block or none)`, sorted; every leaf non-empty.
    leaves: Vec<Vec<(u32, Option<Slot>)>>,
    /// First start of each leaf.
    firsts: Vec<u32>,
}

impl RangeMap {
    pub(crate) fn new() -> RangeMap {
        RangeMap::default()
    }

    /// The value at `x`.
    #[inline]
    fn value_at(&self, x: u32) -> Option<Slot> {
        let leaf = self.firsts.partition_point(|&f| f <= x);
        if leaf == 0 {
            return None;
        }
        let entries = &self.leaves[leaf - 1];
        entries[entries.partition_point(|e| e.0 <= x) - 1].1
    }

    /// The block slot holding `id`, if assigned.
    #[inline]
    pub(crate) fn get(&self, id: u32) -> Option<u32> {
        let v = self.value_at(id);
        v.map(Slot::get)
    }

    /// Removes every breakpoint starting in `a..=b`.
    fn remove_points(&mut self, a: u32, b: u32) {
        let mut leaf = self.firsts.partition_point(|&f| f <= a).saturating_sub(1);
        while leaf < self.leaves.len() && self.firsts[leaf] <= b {
            let entries = &mut self.leaves[leaf];
            let lo = entries.partition_point(|e| e.0 < a);
            let hi = entries.partition_point(|e| e.0 <= b);
            entries.drain(lo..hi);
            if entries.is_empty() {
                self.leaves.remove(leaf);
                self.firsts.remove(leaf);
                continue;
            }
            self.firsts[leaf] = entries[0].0;
            leaf += 1;
        }
    }

    /// Inserts a breakpoint at `start` (none may exist there).
    fn insert_point(&mut self, start: u32, value: Option<Slot>) {
        if self.leaves.is_empty() {
            self.leaves.push(vec![(start, value)]);
            self.firsts.push(start);
            return;
        }
        let leaf = self.firsts.partition_point(|&f| f <= start).saturating_sub(1);
        let entries = &mut self.leaves[leaf];
        let i = entries.partition_point(|e| e.0 < start);
        entries.insert(i, (start, value));
        self.firsts[leaf] = entries[0].0;
        if entries.len() > LEAF_MAX {
            let right = entries.split_off(LEAF_MAX / 2);
            self.firsts.insert(leaf + 1, right[0].0);
            self.leaves.insert(leaf + 1, right);
        }
    }

    /// Sets ids `a..b` to `value`, merging with equal
    /// neighbors so each run stays one breakpoint.
    fn set_range(&mut self, a: u32, b: u32, value: Option<Slot>) {
        debug_assert!(a < b);
        let after = self.value_at(b);
        let before = if a == 0 { None } else { self.value_at(a - 1) };
        self.remove_points(a, b);
        if before != value {
            self.insert_point(a, value);
        }
        if after != value {
            self.insert_point(b, after);
        }
    }

    /// Records `slot` for every id in `ids` (any order; sorted and
    /// deduplicated in place), one range per run of consecutive ids.
    pub(crate) fn assign(&mut self, slot: u32, ids: &mut Vec<u32>) {
        let value = Some(Slot::new(slot));
        ids.sort_unstable();
        ids.dedup();
        let mut start = 0;
        while start < ids.len() {
            let mut end = start + 1;
            while end < ids.len() && ids[end] == ids[end - 1] + 1 {
                end += 1;
            }
            self.set_range(ids[start], ids[end - 1] + 1, value);
            start = end;
        }
    }

    #[cfg(test)]
    pub(crate) fn set(&mut self, id: u32, slot: u32) {
        self.set_range(id, id + 1, Some(Slot::new(slot)));
    }

    pub(crate) fn clear(&mut self, id: u32) {
        self.set_range(id, id + 1, None);
    }

    /// Logical size: 8 bytes per breakpoint plus per-leaf index entries.
    pub(crate) fn size_bytes(&self) -> usize {
        self.leaves.iter().map(Vec::len).sum::<usize>() * 8 + self.leaves.len() * 4
    }

    /// Repacks leaves full (to `LEAF_MAX / 2 .. LEAF_MAX` entries' worth of
    /// capacity, exactly sized).
    pub(crate) fn shrink_to_fit(&mut self) {
        let all: Vec<(u32, Option<Slot>)> = self.leaves.drain(..).flatten().collect();
        self.firsts.clear();
        for chunk in all.chunks(LEAF_MAX) {
            self.firsts.push(chunk[0].0);
            self.leaves.push(chunk.to_vec());
        }
        self.leaves.shrink_to_fit();
        self.firsts.shrink_to_fit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn matches_model_under_runs_sets_and_clears() {
        let mut map = RangeMap::new();
        let mut model: HashMap<u32, u32> = HashMap::new();
        let mut x = 0x2545_F491u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for round in 0..6000 {
            match next() % 5 {
                0 | 1 => {
                    let slot = next() % 50;
                    let start = next() % 30_000;
                    let mut ids: Vec<u32> = (0..1 + next() % 300).map(|k| start + k * (1 + u32::from(next() % 7 == 0))).collect();
                    for &id in &ids {
                        model.insert(id, slot);
                    }
                    map.assign(slot, &mut ids);
                }
                2 => {
                    let (id, slot) = (next() % 31_000, next() % 50);
                    model.insert(id, slot);
                    map.set(id, slot);
                }
                _ => {
                    let id = next() % 31_000;
                    model.remove(&id);
                    map.clear(id);
                }
            }
            if round % 600 == 0 || round == 5999 {
                for id in 0..31_500u32 {
                    assert_eq!(map.get(id), model.get(&id).copied(), "round {round} id {id}");
                }
                // Equal neighbors are merged: no two adjacent breakpoints
                // share a value.
                let all: Vec<(u32, Option<Slot>)> = map.leaves.iter().flatten().copied().collect();
                assert!(all.windows(2).all(|w| w[0].1 != w[1].1 && w[0].0 < w[1].0));
            }
        }
        map.shrink_to_fit();
        for id in 0..31_500u32 {
            assert_eq!(map.get(id), model.get(&id).copied());
        }
        assert_eq!(map.get(u32::MAX - 1), None);
    }
}
