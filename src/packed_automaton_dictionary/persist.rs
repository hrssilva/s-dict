//! Saving and loading a [`PackedAutomatonDictionary`].
//!
//! Format (version 1, little-endian):
//!
//! - Header, 64 bytes: magic `SDICTPA\0`, version (`u32`), header length
//!   (`u32`), payload length (`u64`), checksum (`u64`, of the rest of the
//!   header and the payload), term count (`u64`), next fresh id, id limit,
//!   `block_min`, `block_max`, block count, free id count (`u32` each).
//! - Payload: the symbol table (`u32` count, then per symbol a length byte
//!   and its bytes); the free ids (`u32` each, in reuse order); the block
//!   table, one 32-byte entry per block in term order (records offset
//!   `u64`, records length, root, term count, growth baseline as `f32`
//!   bits, all `u32`; automaton flags `u8`; has-symbols `u8`; 2 padding
//!   bytes; ids length `u32`); every block's ids (see [`write_ids`]);
//!   every block's automaton records, concatenated.
//!
//! Loading never trusts the file: besides the checksum (which catches
//! accidental corruption, not tampering: it is not cryptographic), every
//! section is checked against what the dictionary relies on -- see
//! `PackedAutomaton::from_parts` for the records, and
//! [`PackedAutomatonDictionary::load`] for the rest. A file that fails any
//! check is refused with an error, never a panic.

use std::io::{self, Write};
use std::sync::Arc;

use super::PackedAutomatonDictionary;
use crate::TermId;
use crate::fsst::SymbolTable;
use crate::packed_automaton::PackedAutomaton;
use crate::packed_automaton_block::{PackedAutomatonBlock, Workspace};
use crate::range_map::RangeMap;
use crate::records::{Image, Records};

const MAGIC: [u8; 8] = *b"SDICTPA\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 64;
const BLOCK_ENTRY_LEN: usize = 32;

/// Why a file couldn't be loaded.
#[derive(Debug)]
pub enum LoadError {
    /// Reading the file failed.
    Io(io::Error),
    /// Not a dictionary file, or one in a format version this build doesn't
    /// read.
    UnsupportedFormat(&'static str),
    /// The file is a dictionary file but fails a consistency check:
    /// truncated, corrupted, or not written by `save`.
    Invalid(&'static str),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "reading the dictionary file failed: {e}"),
            LoadError::UnsupportedFormat(why) => write!(f, "unsupported dictionary file: {why}"),
            LoadError::Invalid(why) => write!(f, "invalid dictionary file: {why}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        LoadError::Io(e)
    }
}

/// Header bytes after the checksum field: covered by the checksum.
const CHECKED_HEADER: std::ops::Range<usize> = 32..HEADER_LEN;

/// A 64-bit checksum of the header's data fields and the payload: a
/// multiply-rotate over 8-byte words (each step a bijection of the running
/// state, so changing any one word changes the result), then the length.
/// Catches accidental corruption; not a defense against a crafted file --
/// the structural checks are.
fn checksum(header: &[u8], payload: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = 0u64;
    let mut mix = |w: [u8; 8]| h = (h.rotate_left(23) ^ u64::from_le_bytes(w)).wrapping_mul(K);
    let header = &header[CHECKED_HEADER];
    for w in header.as_chunks::<8>().0.iter().chain(payload.as_chunks::<8>().0) {
        mix(*w);
    }
    let tail = payload.as_chunks::<8>().1;
    let mut last = [0u8; 8];
    last[..tail.len()].copy_from_slice(tail);
    mix(last);
    (h ^ payload.len() as u64).wrapping_mul(K)
}

fn bits(max: u32) -> u32 {
    32 - max.leading_zeros()
}

fn leb128_len(value: u32) -> usize {
    (bits(value).max(1) as usize).div_ceil(7)
}

/// Appends `values` (each below `1 << width`) packed LSB-first.
fn write_packed(out: &mut Vec<u8>, values: impl Iterator<Item = u32>, width: u32) {
    let (mut acc, mut filled) = (0u64, 0u32);
    for v in values {
        acc |= u64::from(v) << filled;
        filled += width;
        while filled >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            filled -= 8;
        }
    }
    if filled > 0 {
        out.push(acc as u8);
    }
}

/// Encoding tags of a block's ids.
const IDS_RUNS: u8 = 0;
const IDS_OFFSETS: u8 = 1;

/// Appends a block's ids (in rank order), in the smaller of two forms
/// (the information `BlockIds` keeps in memory, so the file stays as
/// compact): `IDS_RUNS` -- the ids as an interleaving of runs of
/// consecutive ids (LEB128 run count, then per run its first id and
/// length), plus each rank's run, bit-packed in `bits(runs - 1)` bits; or
/// `IDS_OFFSETS` -- the smallest id (`u32`), a width (`u8`) and every id's
/// offset from it, bit-packed.
fn write_ids(out: &mut Vec<u8>, ids: &[TermId]) {
    // Greedy runs: each id continues the run expecting it, else starts one.
    let mut expecting: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut labels: Vec<u32> = Vec::with_capacity(ids.len());
    for &TermId(id) in ids {
        let run = match expecting.remove(&id) {
            Some(run) => run,
            None => {
                runs.push((id, 0));
                runs.len() as u32 - 1
            }
        };
        runs[run as usize].1 += 1;
        labels.push(run);
        if let Some(next) = id.checked_add(1) {
            expecting.insert(next, run);
        }
    }
    let label_width = bits(runs.len().saturating_sub(1) as u32);
    let runs_size = leb128_len(runs.len() as u32)
        + runs.iter().map(|&(first, len)| leb128_len(first) + leb128_len(len)).sum::<usize>()
        + (ids.len() * label_width as usize).div_ceil(8);
    let min = ids.iter().map(|id| id.0).min().unwrap_or(0);
    let offset_width = bits(ids.iter().map(|id| id.0 - min).max().unwrap_or(0));
    let offsets_size = 5 + (ids.len() * offset_width as usize).div_ceil(8);
    if runs_size <= offsets_size {
        out.push(IDS_RUNS);
        write_leb128(out, runs.len() as u32);
        for (first, len) in runs {
            write_leb128(out, first);
            write_leb128(out, len);
        }
        write_packed(out, labels.into_iter(), label_width);
    } else {
        out.push(IDS_OFFSETS);
        out.extend_from_slice(&min.to_le_bytes());
        out.push(offset_width as u8);
        write_packed(out, ids.iter().map(|id| id.0 - min), offset_width);
    }
}

fn write_leb128(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// A bounds-checked cursor over a section of the file.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], LoadError> {
        let end = self.at.checked_add(n).filter(|&end| end <= self.bytes.len()).ok_or(LoadError::Invalid(what))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, LoadError> {
        Ok(self.take(1, what)?[0])
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, LoadError> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().expect("4 bytes")))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, LoadError> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().expect("8 bytes")))
    }

    fn leb128(&mut self, what: &'static str) -> Result<u32, LoadError> {
        let mut value = 0u32;
        for k in 0..5 {
            let byte = self.u8(what)?;
            if k == 4 && byte > 0x0F {
                return Err(LoadError::Invalid(what));
            }
            value |= u32::from(byte & 0x7F) << (7 * k);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(LoadError::Invalid(what))
    }

    fn is_empty(&self) -> bool {
        self.at == self.bytes.len()
    }

    /// `n` values of `width` bits, packed as [`write_packed`] writes them.
    fn packed(&mut self, n: usize, width: u32, what: &'static str) -> Result<Vec<u32>, LoadError> {
        if width > 32 {
            return Err(LoadError::Invalid(what));
        }
        let bytes = self.take(n.checked_mul(width as usize).ok_or(LoadError::Invalid(what))?.div_ceil(8), what)?;
        let mut values = Vec::with_capacity(n);
        let (mut acc, mut filled, mut next) = (0u64, 0u32, 0usize);
        let mask = if width == 32 { u64::from(u32::MAX) } else { (1u64 << width) - 1 };
        for _ in 0..n {
            while filled < width {
                acc |= u64::from(bytes[next]) << filled;
                next += 1;
                filled += 8;
            }
            values.push((acc & mask) as u32);
            acc >>= width;
            filled -= width;
        }
        Ok(values)
    }
}

/// Reads one block's ids ([`write_ids`]) into `ids`: exactly `count`, each
/// rank's run label naming an existing run, every run used exactly its
/// length.
fn read_ids(r: &mut Reader<'_>, count: usize, ids: &mut Vec<TermId>) -> Result<(), LoadError> {
    let invalid = || LoadError::Invalid("block ids");
    ids.clear();
    match r.u8("block ids")? {
        IDS_RUNS => {
            let run_count = r.leb128("block ids")? as usize;
            if run_count == 0 || run_count > count {
                return Err(invalid());
            }
            let mut runs = Vec::with_capacity(run_count);
            for _ in 0..run_count {
                let (first, len) = (r.leb128("block ids")?, r.leb128("block ids")?);
                first.checked_add(len).ok_or_else(invalid)?;
                runs.push((first, len, 0u32));
            }
            for label in r.packed(count, bits(run_count as u32 - 1), "block ids")? {
                let run = runs.get_mut(label as usize).ok_or_else(invalid)?;
                if run.2 == run.1 {
                    return Err(invalid());
                }
                ids.push(TermId(run.0 + run.2));
                run.2 += 1;
            }
            if runs.iter().any(|&(_, len, used)| used != len) {
                return Err(invalid());
            }
        }
        IDS_OFFSETS => {
            let min = r.u32("block ids")?;
            let width = u32::from(r.u8("block ids")?);
            for offset in r.packed(count, width, "block ids")? {
                ids.push(TermId(min.checked_add(offset).ok_or_else(invalid)?));
            }
        }
        _ => return Err(invalid()),
    }
    Ok(())
}

/// One block table entry.
struct BlockEntry {
    records_offset: u64,
    records_len: u32,
    root: u32,
    term_count: u32,
    compact_bytes_per_term: f32,
    flags: u8,
    has_symbols: bool,
    ids_len: u32,
}

impl PackedAutomatonDictionary {
    /// Writes the dictionary to `out` in the format of this module's doc
    /// comment. Blocks are written as they are: call
    /// [`shrink_to_fit`](Self::shrink_to_fit) first to drop dead records
    /// and write the compact, path-compressed form. Buffers the payload
    /// (the size of the file) to checksum it before the header.
    pub fn save(&self, out: &mut impl Write) -> io::Result<()> {
        let mut payload = Vec::new();
        let symbols: Vec<&[u8]> = self.symbols.as_deref().map(|t| t.symbols().collect()).unwrap_or_default();
        payload.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
        for symbol in &symbols {
            payload.push(symbol.len() as u8);
            payload.extend_from_slice(symbol);
        }
        for &id in &self.free_term_ids {
            payload.extend_from_slice(&id.to_le_bytes());
        }
        let mut table = Vec::with_capacity(self.directory.len() * BLOCK_ENTRY_LEN);
        let mut id_runs = Vec::new();
        let mut records: Vec<&[u8]> = Vec::with_capacity(self.directory.len());
        let mut records_offset = 0u64;
        let mut ids = Vec::new();
        for &(_, slot) in &self.directory {
            let block = self.block(slot);
            let (packed, compact_bytes_per_term) = block.persisted_parts();
            let (bytes, root, term_count, flags) = packed.parts();
            ids.clear();
            block.ids_into(&mut ids);
            let runs_start = id_runs.len();
            write_ids(&mut id_runs, &ids);
            table.extend_from_slice(&records_offset.to_le_bytes());
            table.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            table.extend_from_slice(&root.to_le_bytes());
            table.extend_from_slice(&term_count.to_le_bytes());
            table.extend_from_slice(&compact_bytes_per_term.to_bits().to_le_bytes());
            table.push(flags);
            table.push(u8::from(packed.has_symbols()));
            table.extend_from_slice(&[0, 0]);
            table.extend_from_slice(&((id_runs.len() - runs_start) as u32).to_le_bytes());
            records_offset += bytes.len() as u64;
            records.push(bytes);
        }
        payload.extend_from_slice(&table);
        payload.extend_from_slice(&id_runs);
        for bytes in records {
            payload.extend_from_slice(bytes);
        }

        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&(HEADER_LEN as u32).to_le_bytes());
        header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        header.extend_from_slice(&[0; 8]); // checksum, below
        header.extend_from_slice(&(self.len as u64).to_le_bytes());
        for value in [
            self.next_term_id,
            self.id_limit,
            self.block_min as u32,
            self.block_max as u32,
            self.directory.len() as u32,
            self.free_term_ids.len() as u32,
        ] {
            header.extend_from_slice(&value.to_le_bytes());
        }
        debug_assert_eq!(header.len(), HEADER_LEN);
        let sum = checksum(&header, &payload);
        header[24..32].copy_from_slice(&sum.to_le_bytes());
        out.write_all(&header)?;
        out.write_all(&payload)
    }

    /// Reads a dictionary [`save`](Self::save) wrote, copying its records.
    /// Every section is validated: anything inconsistent is refused with
    /// [`LoadError::Invalid`] (never a panic), so `bytes` need not be
    /// trusted. Checks, beyond the checksum and each block's automaton
    /// (`PackedAutomaton::from_parts`): block bounds as
    /// [`with_block_bounds`](Self::with_block_bounds) requires them; blocks
    /// non-empty, within `block_max` terms and in strictly increasing term
    /// order; the live ids and the free ids distinct and together exactly
    /// `0..next id`, below the id limit; the term count their sum.
    pub fn load(bytes: &[u8]) -> Result<PackedAutomatonDictionary, LoadError> {
        Self::load_with(bytes, |_, records| Records::from(records.to_vec()))
    }

    /// [`load`](Self::load) of a file.
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<PackedAutomatonDictionary, LoadError> {
        Self::load(&std::fs::read(path)?)
    }

    /// [`load`](Self::load) without copying records: every block borrows
    /// its records from `bytes` (kept alive by the dictionary) until its
    /// first mutation copies them out. Validated as `load` does.
    pub fn load_shared(bytes: Arc<[u8]>) -> Result<PackedAutomatonDictionary, LoadError> {
        Self::load_image(Arc::new(Image::Heap(bytes)))
    }

    /// [`load_shared`](Self::load_shared) of a memory-mapped file: blocks
    /// borrow their records from the mapping (pages read on demand, shared
    /// with other processes mapping the file), copying them out on their
    /// first mutation. Validated as [`load`](Self::load) does, which reads
    /// every reachable record once.
    ///
    /// # Safety
    ///
    /// The file must not be modified (by this or any other process) while
    /// the dictionary or anything loaded from it is alive: the mapping
    /// would change under validated records, which the dictionary decodes
    /// without re-checking (`decoded_term`'s UTF-8 premise).
    #[cfg(feature = "mmap")]
    pub unsafe fn map(path: impl AsRef<std::path::Path>) -> Result<PackedAutomatonDictionary, LoadError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the caller keeps the file unmodified (this function's
        // contract).
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Self::load_image(Arc::new(Image::Map(map)))
    }

    fn load_image(image: Arc<Image>) -> Result<PackedAutomatonDictionary, LoadError> {
        Self::load_with(image.bytes(), |range, _| Records::shared(&image, range))
    }

    /// Writes the dictionary to `path` atomically: to a temporary file
    /// next to it (`path` + `.tmp`), synced, then renamed over `path`.
    pub fn save_file(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let path = path.as_ref();
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let mut file = io::BufWriter::new(std::fs::File::create(&tmp)?);
        self.save(&mut file)?;
        file.into_inner().map_err(io::IntoInnerError::into_error)?.sync_all()?;
        std::fs::rename(&tmp, path)
    }

    /// [`load`](Self::load), building each block's records with `records`
    /// from that block's range of `bytes` (and the slice itself).
    fn load_with(
        bytes: &[u8],
        mut records: impl FnMut(std::ops::Range<usize>, &[u8]) -> Records,
    ) -> Result<PackedAutomatonDictionary, LoadError> {
        let invalid = LoadError::Invalid;
        let mut header = Reader { bytes, at: 0 };
        if header.take(MAGIC.len(), "truncated header").ok() != Some(&MAGIC[..]) {
            return Err(LoadError::UnsupportedFormat("not an s-dict dictionary file"));
        }
        let version = header.u32("truncated header")?;
        if version != VERSION {
            return Err(LoadError::UnsupportedFormat("unknown format version"));
        }
        if header.u32("truncated header")? as usize != HEADER_LEN {
            return Err(invalid("header length"));
        }
        let payload_len = header.u64("truncated header")?;
        let expected_checksum = header.u64("truncated header")?;
        let len = header.u64("truncated header")?;
        let next_term_id = header.u32("truncated header")?;
        let id_limit = header.u32("truncated header")?;
        let (block_min, block_max) = (header.u32("truncated header")? as usize, header.u32("truncated header")? as usize);
        let block_count = header.u32("truncated header")? as usize;
        let free_count = header.u32("truncated header")? as usize;
        let payload = &bytes[HEADER_LEN..];
        if payload.len() as u64 != payload_len {
            return Err(invalid("payload length differs from the header (truncated or extended file)"));
        }
        if checksum(&bytes[..HEADER_LEN], payload) != expected_checksum {
            return Err(invalid("checksum mismatch"));
        }
        if block_min < 1 || block_max < 2 * block_min || block_max > u16::MAX as usize {
            return Err(invalid("block bounds"));
        }
        if next_term_id > id_limit {
            return Err(invalid("id limit"));
        }

        let mut r = Reader { bytes: payload, at: 0 };
        let symbol_count = r.u32("symbol table")? as usize;
        let mut symbols = Vec::with_capacity(symbol_count.min(256));
        for _ in 0..symbol_count {
            let n = r.u8("symbol table")? as usize;
            symbols.push(r.take(n, "symbol table")?.to_vec());
        }
        let symbols = match symbol_count {
            0 => None,
            _ => Some(Arc::new(SymbolTable::from_persisted(symbols).map_err(invalid)?)),
        };
        let free_ids = r.take(free_count.checked_mul(4).ok_or(invalid("free ids"))?, "free ids")?;
        let free_term_ids: Vec<u32> = free_ids.as_chunks::<4>().0.iter().map(|&c| u32::from_le_bytes(c)).collect();
        let table = r.take(block_count.checked_mul(BLOCK_ENTRY_LEN).ok_or(invalid("block table"))?, "block table")?;
        let mut entries = Vec::with_capacity(block_count);
        let mut t = Reader { bytes: table, at: 0 };
        let mut ids_total = 0u64;
        for _ in 0..block_count {
            let entry = BlockEntry {
                records_offset: t.u64("block table")?,
                records_len: t.u32("block table")?,
                root: t.u32("block table")?,
                term_count: t.u32("block table")?,
                compact_bytes_per_term: f32::from_bits(t.u32("block table")?),
                flags: t.u8("block table")?,
                has_symbols: match t.u8("block table")? {
                    0 => false,
                    1 => true,
                    _ => return Err(invalid("block table")),
                },
                ids_len: {
                    t.take(2, "block table")?;
                    t.u32("block table")?
                },
            };
            ids_total += u64::from(entry.ids_len);
            entries.push(entry);
        }
        let id_runs = r.take(usize::try_from(ids_total).map_err(|_| invalid("id runs"))?, "id runs")?;
        let records_base = HEADER_LEN + r.at;
        let records_section = &payload[r.at..];

        // Every live id and every free id, each exactly once.
        let mut seen = vec![0u64; (next_term_id as usize).div_ceil(64)];
        let mut mark = |id: u32| -> Result<(), LoadError> {
            if id >= next_term_id {
                return Err(invalid("id beyond the next fresh id"));
            }
            let (word, bit) = (id as usize / 64, 1u64 << (id % 64));
            if seen[word] & bit != 0 {
                return Err(invalid("id used twice"));
            }
            seen[word] |= bit;
            Ok(())
        };
        for &id in &free_term_ids {
            mark(id)?;
        }

        let mut blocks = Vec::with_capacity(block_count);
        let mut directory: Vec<(Box<[u8]>, u32)> = Vec::with_capacity(block_count);
        let mut id_to_block = RangeMap::new();
        let mut runs = Reader { bytes: id_runs, at: 0 };
        let mut ids: Vec<TermId> = Vec::new();
        let mut slot_ids: Vec<u32> = Vec::new();
        let mut previous_last: Option<Vec<u8>> = None;
        let mut total_terms = 0u64;
        let mut expected_records_offset = 0u64;
        for (slot, entry) in entries.iter().enumerate() {
            if entry.term_count == 0 || entry.term_count as usize > block_max {
                return Err(invalid("block term count"));
            }
            if entry.records_offset != expected_records_offset {
                return Err(invalid("block records offset"));
            }
            let start = usize::try_from(entry.records_offset).map_err(|_| invalid("block records"))?;
            let block_records = start
                .checked_add(entry.records_len as usize)
                .and_then(|end| records_section.get(start..end))
                .ok_or(invalid("block records outside the file"))?;
            expected_records_offset += u64::from(entry.records_len);

            let mut block_ids = Reader {
                bytes: runs.take(entry.ids_len as usize, "block ids")?,
                at: 0,
            };
            read_ids(&mut block_ids, entry.term_count as usize, &mut ids)?;
            if !block_ids.is_empty() {
                return Err(invalid("block ids"));
            }
            for id in &ids {
                mark(id.0)?;
            }

            if entry.has_symbols && symbols.is_none() {
                return Err(invalid("block refers to a missing symbol table"));
            }
            let block_symbols = if entry.has_symbols { symbols.clone() } else { None };
            let range = records_base + start..records_base + start + block_records.len();
            let (packed, text_bytes) =
                PackedAutomaton::from_parts(records(range, block_records), entry.root, entry.term_count, entry.flags, block_symbols).map_err(invalid)?;
            let block = PackedAutomatonBlock::from_persisted(packed, text_bytes, &ids, entry.compact_bytes_per_term);
            let (first, last) = block.first_and_last_terms();
            if previous_last.as_ref().is_some_and(|prev| prev.as_slice() >= first.as_slice()) {
                return Err(invalid("blocks not in strictly increasing term order"));
            }
            previous_last = Some(last);
            slot_ids.clear();
            slot_ids.extend(ids.iter().map(|id| id.0));
            id_to_block.assign(slot as u32, &mut slot_ids);
            directory.push((first.into_boxed_slice(), slot as u32));
            blocks.push(Some(block));
            total_terms += u64::from(entry.term_count);
        }
        if !runs.is_empty() || expected_records_offset != records_section.len() as u64 {
            return Err(invalid("trailing bytes"));
        }
        if total_terms != len {
            return Err(invalid("term count differs from the blocks"));
        }
        let full_words = next_term_id as usize / 64;
        let tail_bits = next_term_id % 64;
        if seen[..full_words].iter().any(|&w| w != u64::MAX) || (tail_bits > 0 && seen[full_words] != (1u64 << tail_bits) - 1) {
            return Err(invalid("ids neither live nor free"));
        }
        Ok(PackedAutomatonDictionary {
            blocks,
            free_block_slots: Vec::new(),
            directory,
            id_to_block,
            free_term_ids,
            next_term_id,
            id_limit,
            block_min,
            block_max,
            len: len as usize,
            workspace: Workspace::default(),
            slot_ids: Vec::new(),
            symbols,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TermBatch, TermDictionary};

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A dictionary with several blocks, removals (free ids), long and
    /// non-ASCII terms, optionally finished (chains, symbol table).
    fn sample(seed: u64, finish: bool) -> PackedAutomatonDictionary {
        let mut rng = seed;
        let mut dict = PackedAutomatonDictionary::with_block_bounds(4, 16);
        let mut batch = TermBatch::new();
        for i in 0..600 {
            let n = xorshift(&mut rng) % 400;
            let term = match i % 5 {
                0 => format!("<http://example.org/resource/{n}>"),
                1 => format!("\"lit {n} é中\"@en"),
                2 => format!("_:b{n}"),
                3 => "x".repeat((n % 90) as usize),
                _ => format!("<http://example.org/{}/{}>", "long".repeat((n % 40) as usize), n),
            };
            batch.push(&term);
        }
        dict.insert_prepared(&batch.prepare());
        let terms: Vec<String> = (0..dict.next_term_id).filter_map(|id| dict.get_term(TermId(id))).collect();
        for t in terms.iter().step_by(7) {
            dict.remove(t);
        }
        if finish {
            dict.shrink_to_fit();
        }
        dict
    }

    fn saved(dict: &PackedAutomatonDictionary) -> Vec<u8> {
        let mut out = Vec::new();
        dict.save(&mut out).unwrap();
        out
    }

    /// Every id below the next fresh id resolves identically in both, and
    /// every term back.
    fn assert_same(a: &PackedAutomatonDictionary, b: &PackedAutomatonDictionary) {
        assert_eq!((a.len(), a.block_count(), a.ids_available()), (b.len(), b.block_count(), b.ids_available()));
        for id in 0..a.next_term_id.max(b.next_term_id) + 2 {
            let term = a.get_term(TermId(id));
            assert_eq!(term, b.get_term(TermId(id)));
            if let Some(t) = term {
                assert_eq!(b.get_id(&t), Some(TermId(id)));
            }
        }
    }

    /// Both id forms round-trip: one run, interleaved runs, scattered ids
    /// (offsets), extremes; a label naming a missing run is refused.
    #[test]
    fn block_ids_round_trip_in_both_forms() {
        let mut rng = 11u64;
        let scattered: Vec<u32> = (0..300).map(|_| (xorshift(&mut rng) % 1_000_000) as u32).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let cases: Vec<(Vec<u32>, u8)> = vec![
            ((7..1031).collect(), IDS_RUNS),
            ((0..500).flat_map(|i| [i, 10_000 + i, 50_000 + i]).collect(), IDS_RUNS),
            (scattered, IDS_OFFSETS),
            (vec![u32::MAX - 1], IDS_OFFSETS),
            (vec![3], IDS_RUNS),
            (vec![0, u32::MAX - 1, 5], IDS_RUNS),
        ];
        for (raw, form) in cases {
            let ids: Vec<TermId> = raw.iter().copied().map(TermId).collect();
            let mut out = Vec::new();
            write_ids(&mut out, &ids);
            assert_eq!(out[0], form, "form for {} ids", ids.len());
            let mut back = Vec::new();
            let mut r = Reader { bytes: &out, at: 0 };
            read_ids(&mut r, ids.len(), &mut back).unwrap();
            assert!(r.is_empty());
            assert_eq!(back, ids);
            // Runs carry their lengths; offsets can't tell, and an extra id
            // repeats one (refused later: every id is checked used once).
            if form == IDS_RUNS {
                assert!(read_ids(&mut Reader { bytes: &out, at: 0 }, ids.len() + 1, &mut back).is_err());
            }
        }
        // Two runs; then a label pointing at run 2 of 2.
        let mut out = vec![IDS_RUNS];
        write_leb128(&mut out, 2);
        for v in [0, 1, 10, 1] {
            write_leb128(&mut out, v);
        }
        out.push(0b10);
        assert!(read_ids(&mut Reader { bytes: &out, at: 0 }, 2, &mut Vec::new()).is_ok());
        *out.last_mut().unwrap() = 0b11;
        assert!(read_ids(&mut Reader { bytes: &out, at: 0 }, 2, &mut Vec::new()).is_err());
    }

    #[test]
    fn round_trips_and_keeps_mutating_identically() {
        for (seed, finish) in [(1, false), (2, true), (3, true)] {
            let mut original = sample(seed, finish);
            let bytes = saved(&original);
            let mut loaded = PackedAutomatonDictionary::load(&bytes).unwrap();
            assert_same(&original, &loaded);
            assert_eq!(saved(&loaded), bytes, "save is deterministic and load keeps everything");
            // Same ids for the same mutations afterwards: free ids, next
            // id and block layout carried over.
            let mut rng = seed + 100;
            for _ in 0..300 {
                let term = format!("<http://example.org/resource/{}>", xorshift(&mut rng) % 500);
                if xorshift(&mut rng).is_multiple_of(3) {
                    assert_eq!(original.remove(&term), loaded.remove(&term));
                } else {
                    assert_eq!(original.insert(&term), loaded.insert(&term));
                }
            }
            original.shrink_to_fit();
            loaded.shrink_to_fit();
            assert_same(&original, &loaded);
        }
        // Zero-copy: blocks borrow from the shared bytes until mutated,
        // then behave exactly as copied ones.
        let mut original = sample(8, true);
        let bytes = saved(&original);
        let mut shared = shared_load(&bytes);
        assert_same(&original, &shared);
        let mut rng = 108;
        for _ in 0..200 {
            let term = format!("\"lit {} é中\"@en", xorshift(&mut rng) % 500);
            if xorshift(&mut rng).is_multiple_of(3) {
                assert_eq!(original.remove(&term), shared.remove(&term));
            } else {
                assert_eq!(original.insert(&term), shared.insert(&term));
            }
        }
        assert_same(&original, &shared);
        // Empty, and renumbered.
        let empty = PackedAutomatonDictionary::new();
        assert_same(&empty, &PackedAutomatonDictionary::load(&saved(&empty)).unwrap());
        let mut renumbered = sample(4, true);
        let order: Vec<usize> = (0..renumbered.block_count()).collect();
        renumbered.renumber(&order);
        assert_same(&renumbered, &PackedAutomatonDictionary::load(&saved(&renumbered)).unwrap());
    }

    fn shared_load(bytes: &[u8]) -> PackedAutomatonDictionary {
        PackedAutomatonDictionary::load_shared(Arc::from(bytes)).unwrap()
    }

    #[test]
    fn save_file_and_load_file_round_trip() {
        let dict = sample(9, true);
        let path = std::env::temp_dir().join(format!("s-dict-test-{}.sdict", std::process::id()));
        dict.save_file(&path).unwrap();
        let loaded = PackedAutomatonDictionary::load_file(&path).unwrap();
        assert_same(&dict, &loaded);
        #[cfg(feature = "mmap")]
        {
            // SAFETY: the test owns the file and doesn't modify it while
            // the mapped dictionary lives.
            let mut mapped = unsafe { PackedAutomatonDictionary::map(&path) }.unwrap();
            assert_same(&dict, &mapped);
            let mut copy = loaded;
            exercise(&mut mapped);
            exercise(&mut copy);
            assert_same(&copy, &mapped);
        }
        std::fs::remove_file(&path).unwrap();
    }

    /// Exercises a loaded dictionary's read and mutation paths: a file that
    /// passed validation must behave (no panic, consistent answers).
    fn exercise(dict: &mut PackedAutomatonDictionary) {
        let mut terms = Vec::new();
        for id in 0..dict.next_term_id {
            if let Some(t) = dict.get_term(TermId(id)) {
                assert_eq!(dict.get_id(&t), Some(TermId(id)));
                terms.push(t);
            }
        }
        assert_eq!(terms.len(), dict.len());
        for t in terms.iter().step_by(3) {
            assert!(dict.remove(t).is_some());
        }
        for t in terms.iter().step_by(5) {
            dict.insert(&format!("{t}~"));
        }
        dict.shrink_to_fit();
    }

    fn fix_checksum(bytes: &mut [u8]) {
        let sum = checksum(&bytes[..HEADER_LEN], &bytes[HEADER_LEN..]);
        bytes[24..32].copy_from_slice(&sum.to_le_bytes());
    }

    #[test]
    fn truncated_and_corrupted_files_are_refused_without_panicking() {
        let bytes = saved(&sample(5, true));
        for len in (0..bytes.len()).step_by(7).chain([bytes.len() - 1]) {
            assert!(PackedAutomatonDictionary::load(&bytes[..len]).is_err(), "truncated to {len}");
        }
        let mut extended = bytes.clone();
        extended.push(0);
        assert!(PackedAutomatonDictionary::load(&extended).is_err());
        // Any single flipped byte: the checksum or a header check refuses.
        for at in (0..bytes.len()).step_by(3) {
            let mut corrupted = bytes.clone();
            corrupted[at] ^= 0x5A;
            assert!(PackedAutomatonDictionary::load(&corrupted).is_err(), "flip at {at}");
        }
    }

    /// Random edits with the checksum fixed up, so they reach the
    /// structural checks: refused, or accepted only as a dictionary that
    /// behaves (`exercise`). `S_DICT_CORRUPTION_ROUNDS` runs more rounds.
    #[test]
    fn structurally_corrupted_files_never_panic() {
        let rounds: usize = std::env::var("S_DICT_CORRUPTION_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);
        let mut rng = 0xDEC0DEu64 ^ rounds as u64;
        let originals = [saved(&sample(6, true)), saved(&sample(7, false))];
        let (mut refused, mut accepted) = (0, 0);
        for i in 0..rounds {
            let mut bytes = originals[i % 2].clone();
            for _ in 0..1 + xorshift(&mut rng) % 3 {
                // Mostly the payload; sometimes the header fields.
                let at = if xorshift(&mut rng).is_multiple_of(8) {
                    32 + (xorshift(&mut rng) as usize) % 32
                } else {
                    HEADER_LEN + (xorshift(&mut rng) as usize) % (bytes.len() - HEADER_LEN)
                };
                bytes[at] = match xorshift(&mut rng) % 4 {
                    0 => 0,
                    1 => 0xFF,
                    2 => bytes[at].wrapping_add(1),
                    _ => xorshift(&mut rng) as u8,
                };
            }
            fix_checksum(&mut bytes);
            let loaded = if i % 3 == 0 { PackedAutomatonDictionary::load_shared(Arc::from(&bytes[..])) } else { PackedAutomatonDictionary::load(&bytes) };
            match loaded {
                Err(_) => refused += 1,
                Ok(mut dict) => {
                    accepted += 1;
                    exercise(&mut dict);
                }
            }
        }
        eprintln!("corruption: {refused} refused, {accepted} accepted");
        assert!(refused * 4 > rounds * 3, "most edits are caught: {refused} refused, {accepted} accepted");
    }
}
