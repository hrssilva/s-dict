//! A static symbol table for chain labels, after FSST (Boncz, Neumann,
//! Leis, "FSST: Fast Random Access String Compression", VLDB 2020).
//!
//! Up to 255 symbols of 1-8 bytes, each replaced by a one-byte code; code
//! 255 escapes one literal byte. Training is FSST's greedy scheme: encode
//! a sample with the current table, count how often each symbol and each
//! adjacent pair of symbols occurs, and keep the 255 candidates (symbols,
//! and concatenated pairs up to 8 bytes) with the highest gain
//! (frequency x length); five generations. Encoding is greedy longest
//! match. Every string decodes on its own, so a chain record's label can
//! be compared or decoded without touching any other record.

/// Escapes one literal byte.
pub(crate) const ESCAPE: u8 = 255;
const MAX_SYMBOLS: usize = 255;
const GENERATIONS: usize = 5;

#[derive(Debug, Clone)]
pub(crate) struct SymbolTable {
    /// Symbol bytes (first `lens[code]` of each).
    symbols: Vec<[u8; 8]>,
    lens: Vec<u8>,
    /// Codes of the symbols starting with each byte, longest first.
    by_first: Vec<Vec<u8>>,
}

impl SymbolTable {
    fn from_symbols(mut syms: Vec<Vec<u8>>) -> SymbolTable {
        syms.truncate(MAX_SYMBOLS);
        let mut by_first = vec![Vec::new(); 256];
        let mut symbols = Vec::with_capacity(syms.len());
        let mut lens = Vec::with_capacity(syms.len());
        for (code, s) in syms.iter().enumerate() {
            let mut w = [0u8; 8];
            w[..s.len()].copy_from_slice(s);
            symbols.push(w);
            lens.push(s.len() as u8);
            by_first[s[0] as usize].push(code as u8);
        }
        for list in &mut by_first {
            list.sort_by_key(|&c| std::cmp::Reverse(lens[c as usize]));
        }
        SymbolTable {
            symbols,
            lens,
            by_first,
        }
    }

    #[inline]
    fn symbol(&self, code: u8) -> &[u8] {
        &self.symbols[code as usize][..self.lens[code as usize] as usize]
    }

    /// The longest symbol that prefixes `input` (non-empty), if any.
    #[inline]
    fn longest_match(&self, input: &[u8]) -> Option<(u8, usize)> {
        for &code in &self.by_first[input[0] as usize] {
            let s = self.symbol(code);
            if input.len() >= s.len() && &input[..s.len()] == s {
                return Some((code, s.len()));
            }
        }
        None
    }

    /// Trains a table on `sample` (the strings it will encode).
    pub(crate) fn train(sample: &[&[u8]]) -> SymbolTable {
        use std::collections::HashMap;
        // A unit (symbol or literal byte, or two joined) as (bytes, len).
        type Key = (u64, u8);
        fn key(bytes: &[u8]) -> Key {
            let mut w = [0u8; 8];
            w[..bytes.len()].copy_from_slice(bytes);
            (u64::from_le_bytes(w), bytes.len() as u8)
        }
        fn join(a: Key, b: Key) -> Key {
            (a.0 | b.0 << (8 * a.1 as u32), a.1 + b.1)
        }
        let mut table = SymbolTable::from_symbols(Vec::new());
        for _ in 0..GENERATIONS {
            let mut gains: HashMap<Key, u64> = HashMap::new();
            let mut pairs: HashMap<Key, u64> = HashMap::new();
            for &s in sample {
                let mut i = 0;
                let mut prev: Option<Key> = None;
                while i < s.len() {
                    let unit = match table.longest_match(&s[i..]) {
                        Some((code, len)) => {
                            i += len;
                            key(table.symbol(code))
                        }
                        None => {
                            i += 1;
                            key(&s[i - 1..i])
                        }
                    };
                    *gains.entry(unit).or_insert(0) += u64::from(unit.1);
                    if let Some(p) = prev
                        && p.1 + unit.1 <= 8
                    {
                        *pairs.entry(join(p, unit)).or_insert(0) += 1;
                    }
                    prev = Some(unit);
                }
            }
            for (k, count) in pairs {
                let gain = count * u64::from(k.1);
                let entry = gains.entry(k).or_insert(0);
                *entry = (*entry).max(gain);
            }
            let mut ranked: Vec<(Key, u64)> = gains.into_iter().collect();
            ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            table = SymbolTable::from_symbols(
                ranked
                    .into_iter()
                    .take(MAX_SYMBOLS)
                    .map(|((w, len), _)| w.to_le_bytes()[..len as usize].to_vec())
                    .collect(),
            );
        }
        table
    }

    /// Bytes of the table (symbols and lengths).
    pub(crate) fn size_bytes(&self) -> usize {
        self.symbols.len() * 9
    }

    /// Appends the coded form of `input` to `out`.
    pub(crate) fn encode(&self, input: &[u8], out: &mut Vec<u8>) {
        let mut i = 0;
        while i < input.len() {
            match self.longest_match(&input[i..]) {
                Some((code, len)) => {
                    out.push(code);
                    i += len;
                }
                None => {
                    out.push(ESCAPE);
                    out.push(input[i]);
                    i += 1;
                }
            }
        }
    }

    /// Appends the decoded form of `codes` to `out`.
    #[inline]
    pub(crate) fn decode(&self, codes: &[u8], out: &mut Vec<u8>) {
        // FSST's decoding trick: copy a symbol's whole 8-byte slot, then
        // keep only its length -- a fixed-size copy instead of a
        // variable-length one.
        out.reserve(codes.len() * 8);
        let mut i = 0;
        while i < codes.len() {
            let code = codes[i];
            if code == ESCAPE {
                out.push(codes[i + 1]);
                i += 2;
            } else {
                let len = out.len();
                out.extend_from_slice(&self.symbols[code as usize]);
                out.truncate(len + self.lens[code as usize] as usize);
                i += 1;
            }
        }
    }

    /// Decoded length of `codes`.
    pub(crate) fn decoded_len(&self, codes: &[u8]) -> usize {
        let (mut i, mut n) = (0, 0);
        while i < codes.len() {
            if codes[i] == ESCAPE {
                n += 1;
                i += 2;
            } else {
                n += self.lens[codes[i] as usize] as usize;
                i += 1;
            }
        }
        n
    }

    /// Compares the decoded `codes` with `query`, without decoding into a
    /// buffer: `Ok(decoded length)` if the whole label prefixes `query`,
    /// else `Err((matched bytes, label byte at the first mismatch or None
    /// if `query` ended first))`.
    #[inline]
    pub(crate) fn match_prefix(&self, codes: &[u8], query: &[u8]) -> Result<usize, (usize, Option<u8>)> {
        let (mut i, mut at) = (0, 0);
        while i < codes.len() {
            let code = codes[i];
            // Whole-symbol comparison as one masked 64-bit XOR when 8 query
            // bytes are available; only a mismatch (or the query's tail)
            // looks at single bytes.
            if code != ESCAPE && at + 8 <= query.len() {
                let len = self.lens[code as usize] as usize;
                let q = u64::from_le_bytes(query[at..at + 8].try_into().expect("8 bytes"));
                let sym = u64::from_le_bytes(self.symbols[code as usize]);
                let mask = if len == 8 { u64::MAX } else { (1u64 << (8 * len)) - 1 };
                if (q ^ sym) & mask == 0 {
                    at += len;
                    i += 1;
                    continue;
                }
            }
            let piece: &[u8] = if code == ESCAPE {
                i += 2;
                &codes[i - 1..i]
            } else {
                i += 1;
                self.symbol(code)
            };
            for &b in piece {
                match query.get(at) {
                    Some(&q) if q == b => at += 1,
                    Some(_) => return Err((at, Some(b))),
                    None => return Err((at, None)),
                }
            }
        }
        Ok(at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_compresses() {
        let strings: Vec<String> = (0..2000)
            .map(|i| format!("http://example.org/resource/{i}/label-{}", i % 7))
            .chain((0..200).map(|i| format!("\\u{{{i}}}é\\0{}", i * 37)))
            .collect();
        let sample: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let table = SymbolTable::train(&sample);
        let (mut raw, mut coded) = (0, 0);
        for s in &sample {
            let mut enc = Vec::new();
            table.encode(s, &mut enc);
            let mut dec = Vec::new();
            table.decode(&enc, &mut dec);
            assert_eq!(&dec, s);
            assert_eq!(table.decoded_len(&enc), s.len());
            assert_eq!(table.match_prefix(&enc, s), Ok(s.len()));
            let mut longer = s.to_vec();
            longer.push(b'z');
            assert_eq!(table.match_prefix(&enc, &longer), Ok(s.len()));
            if s.len() > 1 {
                assert_eq!(table.match_prefix(&enc, &s[..s.len() - 1]), Err((s.len() - 1, None)));
                let mut wrong = s.to_vec();
                wrong[s.len() / 2] ^= 1;
                assert_eq!(table.match_prefix(&enc, &wrong), Err((s.len() / 2, Some(s[s.len() / 2]))));
            }
            raw += s.len();
            coded += enc.len();
        }
        assert!(coded * 2 < raw, "expected > 2x on repetitive strings: {coded} vs {raw}");
        // Bytes never seen in training still round-trip (escaped).
        let mut enc = Vec::new();
        table.encode(&[0, 1, 255, 254], &mut enc);
        let mut dec = Vec::new();
        table.decode(&enc, &mut dec);
        assert_eq!(dec, [0, 1, 255, 254]);
    }
}
