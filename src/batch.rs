//! Owned, reusable batches whose ordering can be prepared off the writer
//! thread.

use crate::DictError;
use crate::limits::MAX_BATCH_TEXT_BYTES;

/// A batch's sorted, grouped order: `order` lists input indices in term
/// order, and `starts` delimits each group of identical terms (plus a final
/// end), so group `g` is `order[starts[g]..starts[g + 1]]`.
#[derive(Debug, Default)]
pub(crate) struct BatchOrder {
    order: Vec<u32>,
    starts: Vec<u32>,
    /// Ranges still to sort (scratch).
    work: Vec<Range>,
}

/// A range of the records still to sort, from byte `depth` of their terms
/// on (their first `depth` bytes are equal); `depth == ENDED` marks a
/// finished group of identical terms.
#[derive(Debug, Clone, Copy)]
struct Range {
    lo: u32,
    hi: u32,
    depth: u32,
}

/// One term being sorted: its current cached key and its input index (16
/// bytes: a batch holds one per term while it is prepared).
#[derive(Clone, Copy)]
struct Record {
    key: u64,
    index: u32,
}

const ENDED: u32 = u32::MAX;

/// Data bytes per cached key; the key's low byte holds their count.
const KEY_BYTES: usize = 7;

/// Ranges at most this long are finished by comparison sort.
const SMALL_RANGE: usize = 24;

/// Bytes `depth..depth + 7` of `term` (zero-padded) followed by how many
/// were present: keys order like the byte strings they cover, a shorter
/// string before any extension of it, and two keys tie only if both
/// strings continue past them or both ended identically.
#[inline]
fn key(term: &[u8], depth: usize) -> u64 {
    let rest = &term[depth.min(term.len())..];
    if let Some(word) = rest.first_chunk::<8>() {
        // One unaligned load; the 8th byte becomes the count.
        return u64::from_be_bytes(*word) & !0xFF | KEY_BYTES as u64;
    }
    let n = rest.len().min(KEY_BYTES);
    let mut bytes = [0u8; 8];
    bytes[..n].copy_from_slice(&rest[..n]);
    bytes[KEY_BYTES] = n as u8;
    u64::from_be_bytes(bytes)
}

impl BatchOrder {
    /// Sorts `0..len` by `term` bytes and groups identical terms.
    ///
    /// An MSD radix sort over cached keys (after Kärkkäinen and Rantala,
    /// "Engineering Radix Sort for Strings"): each pass computes the next 7
    /// bytes of every term in a range as one `u64` key and sorts the
    /// range's records by it; only ranges that still tie are refined
    /// further, so a long shared prefix (namespace IRIs) is read once per
    /// 7 bytes instead of once per comparison. Ties whose terms ended are
    /// the groups. Iterative, so duplicates of a very long term can't
    /// overflow the stack. Measured on real RDF batches (prepare alone): 159
    /// ns/term (154 with records holding term slices, at twice the memory),
    /// against 215 for a comparison sort plus a grouping pass;
    /// what remains is mostly cache misses fetching term bytes.
    pub(crate) fn prepare<'a>(&mut self, len: usize, term: impl Fn(usize) -> &'a [u8]) {
        let len32 = u32::try_from(len).expect("batch length fits u32");
        let term = |r: &Record| term(r.index as usize);
        let mut records: Vec<Record> = (0..len32).map(|index| Record { key: 0, index }).collect();
        self.starts.clear();
        self.work.clear();
        if len > 0 {
            self.work.push(Range { lo: 0, hi: len32, depth: 0 });
        }
        // Ranges are pushed right to left, so they pop (and emit groups)
        // in sorted order.
        while let Some(Range { lo, hi, depth }) = self.work.pop() {
            if depth == ENDED || hi - lo == 1 {
                self.starts.push(lo);
                continue;
            }
            let d = depth as usize;
            let range = &mut records[lo as usize..hi as usize];
            let first = self.work.len();
            if range.len() <= SMALL_RANGE {
                range.sort_unstable_by(|a, b| term(a)[d..].cmp(&term(b)[d..]));
                let mut group = 0;
                for i in 1..range.len() {
                    if term(&range[i - 1])[d..] != term(&range[i])[d..] {
                        self.work.push(Range { lo: lo + group as u32, hi: lo + i as u32, depth: ENDED });
                        group = i;
                    }
                }
                self.work.push(Range { lo: lo + group as u32, hi, depth: ENDED });
            } else {
                for r in range.iter_mut() {
                    r.key = key(term(r), d);
                }
                range.sort_unstable_by_key(|r| r.key);
                let mut start = 0usize;
                while start < range.len() {
                    let k = range[start].key;
                    let end = start + range[start..].partition_point(|r| r.key == k);
                    let continues = (k & 0xFF) as usize == KEY_BYTES;
                    self.work.push(Range {
                        lo: lo + start as u32,
                        hi: lo + end as u32,
                        depth: if continues { depth + KEY_BYTES as u32 } else { ENDED },
                    });
                    start = end;
                }
            }
            self.work[first..].reverse();
        }
        self.starts.push(len32);
        self.order.clear();
        self.order.extend(records.iter().map(|r| r.index));
    }

    pub(crate) fn len(&self) -> usize {
        self.order.len()
    }

    pub(crate) fn unique_len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// An input index of group `group`'s term.
    pub(crate) fn first(&self, group: usize) -> usize {
        self.order[self.starts[group] as usize] as usize
    }

    /// Every input index holding group `group`'s term.
    pub(crate) fn positions(&self, group: usize) -> impl Iterator<Item = usize> + '_ {
        self.order[self.starts[group] as usize..self.starts[group + 1] as usize].iter().map(|&i| i as usize)
    }

    fn clear(&mut self) {
        self.order.clear();
        self.starts.clear();
    }
}

/// Owned term text for one caller-bounded batch, being filled. `clear`
/// retains capacity. [`TermBatch::prepare`] turns it into a
/// [`PreparedBatch`] -- off the writer thread, if wanted: preparation never
/// touches a dictionary.
#[derive(Debug, Default)]
pub struct TermBatch {
    /// Concatenated term bytes; kept as bytes (each term was a `&str`, so
    /// slicing by its own offsets needs no char-boundary checks).
    text: Vec<u8>,
    /// `(start, end)` of each term in `text`.
    terms: Vec<(u32, u32)>,
    /// The sorted, grouped order; meaningful only in a `PreparedBatch`
    /// (kept here so its buffers are reused across batches).
    order: BatchOrder,
}

impl TermBatch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `term`, or refuses (the batch unchanged) if the batch's
    /// text would exceed its 4 GiB limit (`u32` offsets).
    pub fn try_push(&mut self, term: &str) -> Result<(), DictError> {
        let start = self.text.len();
        let end = start + term.len();
        if end > MAX_BATCH_TEXT_BYTES {
            return Err(DictError::BatchTooLarge { len: end, max: MAX_BATCH_TEXT_BYTES });
        }
        self.text.extend_from_slice(term.as_bytes());
        // Both fit: `MAX_BATCH_TEXT_BYTES` is at most `u32::MAX`.
        self.terms.push((start as u32, end as u32));
        Ok(())
    }

    /// [`Self::try_push`], panicking on error.
    pub fn push(&mut self, term: &str) {
        self.try_push(term).unwrap_or_else(|e| panic!("{e}"));
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.terms.clear();
        self.order.clear();
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn text_bytes(&self) -> usize {
        self.text.len()
    }

    /// The bytes of the term pushed `index`-th.
    fn term(&self, index: usize) -> &[u8] {
        let (start, end) = self.terms[index];
        &self.text[start as usize..end as usize]
    }

    /// Sorts and groups duplicate terms (keeping every input position) and
    /// returns the batch as one a dictionary inserts as is. The batch can't
    /// change while prepared, so its order always matches its terms
    /// ([`PreparedBatch::into_batch`] to change it again).
    pub fn prepare(mut self) -> PreparedBatch {
        let (text, terms) = (&self.text, &self.terms);
        self.order.prepare(terms.len(), |i| {
            let (start, end) = terms[i];
            &text[start as usize..end as usize]
        });
        PreparedBatch(self)
    }
}

/// A [`TermBatch`] whose terms are sorted and grouped: what
/// `PackedAutomatonDictionary::insert_prepared` takes. Only
/// [`TermBatch::prepare`] makes one.
#[derive(Debug)]
pub struct PreparedBatch(TermBatch);

impl Default for PreparedBatch {
    /// An empty prepared batch.
    fn default() -> Self {
        TermBatch::default().prepare()
    }
}

impl PreparedBatch {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn text_bytes(&self) -> usize {
        self.0.text_bytes()
    }

    /// Back to an editable batch (same terms, all capacity kept): `clear`
    /// it to reuse its buffers for the next batch.
    pub fn into_batch(self) -> TermBatch {
        self.0
    }

    pub(crate) fn order(&self) -> &BatchOrder {
        &self.0.order
    }

    /// The bytes of the term pushed `index`-th.
    pub(crate) fn term(&self, index: usize) -> &[u8] {
        self.0.term(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Groups, in order, as the terms they hold and their input positions.
    fn groups(order: &BatchOrder, terms: &[String]) -> Vec<(String, Vec<usize>)> {
        (0..order.unique_len())
            .map(|g| {
                let mut positions: Vec<usize> = order.positions(g).collect();
                positions.sort_unstable();
                (terms[order.first(g)].clone(), positions)
            })
            .collect()
    }

    fn reference(terms: &[String]) -> Vec<(String, Vec<usize>)> {
        let mut map = std::collections::BTreeMap::<String, Vec<usize>>::new();
        for (i, t) in terms.iter().enumerate() {
            map.entry(t.clone()).or_default().push(i);
        }
        map.into_iter().collect()
    }

    /// Text beyond the limit (small in unit tests) is refused, leaving the
    /// batch as it was.
    #[test]
    fn try_push_refuses_text_beyond_the_limit() {
        let mut batch = TermBatch::new();
        let chunk = "x".repeat(MAX_BATCH_TEXT_BYTES / 4);
        for _ in 0..4 {
            batch.try_push(&chunk).unwrap();
        }
        assert_eq!(batch.try_push("y"), Err(DictError::BatchTooLarge { len: MAX_BATCH_TEXT_BYTES + 1, max: MAX_BATCH_TEXT_BYTES }));
        assert_eq!((batch.len(), batch.text_bytes()), (4, MAX_BATCH_TEXT_BYTES));
        batch.try_push("").unwrap();
        assert_eq!(batch.prepare().len(), 5);
    }

    #[test]
    fn sorts_and_groups_like_a_btreemap() {
        let mut x = 0x2545_F491_u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        // Shared long prefixes, prefixes of each other, NUL and multibyte
        // bytes, empty terms, many duplicates, and lengths around the
        // 7-byte key boundary.
        let mut terms: Vec<String> = Vec::new();
        for _ in 0..5000 {
            let base = match next() % 6 {
                0 => format!("<http://example.org/ns/{}>", next() % 300),
                1 => format!("\"{}\"@en", "x".repeat((next() % 20) as usize)),
                2 => "a\0b".repeat((next() % 4) as usize),
                3 => format!("é中{}", next() % 50),
                4 => "1234567".repeat((next() % 3) as usize) + &"8".repeat((next() % 2) as usize),
                _ => String::new(),
            };
            terms.push(base);
        }
        let mut order = BatchOrder::default();
        order.prepare(terms.len(), |i| terms[i].as_bytes());
        assert_eq!(order.len(), terms.len());
        assert_eq!(groups(&order, &terms), reference(&terms));
        // Reuse, including small and empty batches.
        for n in [0, 1, 2, 30] {
            order.prepare(n, |i| terms[i].as_bytes());
            assert_eq!(groups(&order, &terms[..n]), reference(&terms[..n]));
        }
    }

    /// Duplicates of a very long term refine 7 bytes per pass; iterative,
    /// so this must not overflow the stack.
    #[test]
    fn duplicates_of_a_very_long_term() {
        let long = "q".repeat(2_000_000);
        let terms = vec![long.clone(), format!("{long}r"), long.clone(), "a".to_string()];
        let mut order = BatchOrder::default();
        order.prepare(terms.len(), |i| terms[i].as_bytes());
        assert_eq!(groups(&order, &terms), reference(&terms));
    }
}
