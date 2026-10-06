//! The `.snw` file: every column contiguous, fixed width, frame-of-reference
//! encoded to the narrowest unsigned lane that holds its range, 64-byte
//! aligned, and mmap'd -- so a scan reads the file's bytes directly.
//!
//! Layout (little endian):
//!   [0..8)   magic "SINEWDB1"
//!   [8..16)  catalog offset
//!   [16..24) catalog length
//!   data sections, each 64-byte aligned
//!   catalog (table/column/index descriptors referencing the sections)

mod build;
#[cfg(feature = "sqlite-import")]
pub mod import;

pub use build::{Builder, ColumnBuilder, ColumnValues, TableBuilder};

use crate::error::{Error, Result};
use crate::kernels::{Lanes, ZoneHit, EncPred};
use crate::value::Value;
use memmap2::Mmap;
use std::path::Path;

/// Rows per zone-map entry.
pub const ZONE_ROWS: usize = 1024;

pub(crate) const MAGIC: &[u8; 8] = b"SINEWDB1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Enc {
    /// `value = base + position`: a dense, sorted rowid. Nothing is stored.
    Dense = 0,
    U8 = 1,
    U16 = 2,
    U32 = 3,
    U64 = 4,
}

/// An integer column: `value = base.wrapping_add(enc as i64)`.
pub struct IntCol {
    pub enc: Enc,
    pub base: i64,
    /// Largest encoded value present (0 for an empty or all-NULL column).
    pub maxenc: u64,
    /// Encoded lanes; empty for `Dense`.
    pub data: Lanes<'static>,
    /// Per zone, `[min, max]` of the encoded non-NULL values; `min > max`
    /// marks a zone with none.
    pub zones: &'static [[u64; 2]],
    /// One byte per row, 0xFF = present, 0x00 = NULL; `None` when no NULLs.
    pub valid: Option<&'static [u8]>,
    /// Byte-sliced filter plane for wide columns: the top 8 significant bits of
    /// every encoding (`enc >> shift`), plus their histogram. A predicate is
    /// decided from this byte for every row whose byte differs from the
    /// bound's; only rows sharing it read the full lane.
    pub hi8: Option<Hi8>,
}

#[derive(Clone, Copy)]
pub struct Hi8 {
    pub plane: &'static [u8],
    pub shift: u32,
    pub hist: &'static [u32],
}

impl IntCol {
    #[inline(always)]
    pub fn enc_at(&self, pos: usize) -> u64 {
        match self.enc {
            Enc::Dense => pos as u64,
            _ => self.data.get(pos),
        }
    }

    #[inline(always)]
    pub fn decode(&self, enc: u64) -> i64 {
        self.base.wrapping_add(enc as i64)
    }

    #[inline(always)]
    pub fn is_null(&self, pos: usize) -> bool {
        matches!(self.valid, Some(v) if v[pos] == 0)
    }

    #[inline]
    pub fn value_at(&self, pos: usize) -> Option<i64> {
        if self.is_null(pos) { None } else { Some(self.decode(self.enc_at(pos))) }
    }

    /// Zone `z`'s verdict for `p`.
    #[inline]
    pub fn zone_hit(&self, z: usize, p: EncPred) -> ZoneHit {
        let [lo, hi] = self.zones[z];
        if lo > hi { ZoneHit::None } else { p.classify(lo, hi) }
    }

    /// Lowers `value <op> c` into the encoded domain. `None` means no non-NULL
    /// row can match; `Some(None)` means every non-NULL row matches.
    pub fn lower(&self, op: CmpOp, c: i64) -> Option<Option<EncPred>> {
        let e = c as i128 - self.base as i128;
        let max = self.maxenc as i128;
        // [lo, hi] in the encoded domain, clamped to what is present.
        let (lo, hi) = match op {
            CmpOp::Gt => (e + 1, max),
            CmpOp::Ge => (e, max),
            CmpOp::Lt => (0, e - 1),
            CmpOp::Le => (0, e),
            CmpOp::Eq => (e, e),
            CmpOp::Ne => {
                return if e < 0 || e > max { Some(None) } else { Some(Some(EncPred::Ne(e as u64))) };
            }
        };
        Self::interval(lo, hi, max)
    }

    /// The predicate for `lo <= enc <= hi`, choosing the cheapest form.
    pub(crate) fn interval(lo: i128, hi: i128, max: i128) -> Option<Option<EncPred>> {
        let lo = lo.max(0);
        let hi = hi.min(max);
        if lo > hi {
            return None;
        }
        if lo == 0 && hi == max {
            return Some(None);
        }
        let p = if lo == hi {
            EncPred::Eq(lo as u64)
        } else if hi == max {
            EncPred::Ge(lo as u64)
        } else if lo == 0 {
            EncPred::Le(hi as u64)
        } else {
            EncPred::Range { lo: lo as u64, span: (hi - lo) as u64 }
        };
        Some(Some(p))
    }

}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    /// The operator with its operands swapped (`c < x` is `x > c`).
    pub fn flip(self) -> CmpOp {
        match self {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::Le => CmpOp::Ge,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::Ge => CmpOp::Le,
            o => o,
        }
    }
}

pub enum ColData {
    Int(IntCol),
    Real { data: &'static [f64], valid: Option<&'static [u8]> },
    Text { offs: &'static [u32], bytes: &'static [u8], valid: Option<&'static [u8]> },
}

pub struct Column {
    pub name: String,
    pub decl: String,
    /// The implicit rowid of a table without an INTEGER PRIMARY KEY.
    pub hidden: bool,
    pub data: ColData,
}

impl Column {
    pub fn value(&self, pos: usize) -> Value {
        match &self.data {
            ColData::Int(c) => match c.value_at(pos) {
                Some(v) => Value::Int(v),
                None => Value::Null,
            },
            ColData::Real { data, valid } => {
                if matches!(valid, Some(v) if v[pos] == 0) { Value::Null } else { Value::Real(data[pos]) }
            }
            ColData::Text { offs, bytes, valid } => {
                if matches!(valid, Some(v) if v[pos] == 0) {
                    Value::Null
                } else {
                    let s = &bytes[offs[pos] as usize..offs[pos + 1] as usize];
                    Value::Text(String::from_utf8_lossy(s).into())
                }
            }
        }
    }

    pub fn int(&self) -> Option<&IntCol> {
        match &self.data {
            ColData::Int(c) => Some(c),
            _ => None,
        }
    }
}

/// A single-column secondary index: `(key, position)` sorted by key, keys in
/// the column's own encoding. NULLs are not indexed.
pub struct Index {
    pub name: String,
    pub col: usize,
    pub keys: Lanes<'static>,
    pub pos: &'static [u32],
}

impl Index {
    /// Positions `[lo, hi)` in the index whose key equals `k`.
    #[inline]
    pub fn equal_range(&self, k: u64) -> (usize, usize) {
        self.range(k, k)
    }

    /// Positions `[lo, hi)` in the index whose key lies in `[lo, hi]`.
    #[inline]
    pub fn range(&self, lo: u64, hi: u64) -> (usize, usize) {
        let a = lower_bound(self.keys, lo);
        let b = if hi == u64::MAX { self.keys.len() } else { lower_bound(self.keys, hi + 1) };
        (a, b.max(a))
    }
}

/// First index `i` with `keys[i] >= k`. Branch-free: the loop runs exactly
/// `ceil(log2(n))` times with a conditional move per step, so the CPU never
/// mispredicts and successive probes pipeline.
#[inline]
pub fn lower_bound(keys: Lanes, k: u64) -> usize {
    #[inline(always)]
    fn lb<T: Copy + Into<u64>>(a: &[T], k: u64) -> usize {
        let mut n = a.len();
        if n == 0 {
            return 0;
        }
        let mut base = 0usize;
        while n > 1 {
            let half = n / 2;
            // SAFETY: base + half < base + n <= a.len()
            let v: u64 = unsafe { (*a.get_unchecked(base + half - 1)).into() };
            base = if v < k { base + half } else { base };
            n -= half;
        }
        let v: u64 = unsafe { (*a.get_unchecked(base)).into() };
        base + (v < k) as usize
    }
    match keys {
        Lanes::U8(a) => lb(a, k),
        Lanes::U16(a) => lb(a, k),
        Lanes::U32(a) => lb(a, k),
        Lanes::U64(a) => lb(a, k),
    }
}

pub struct Table {
    pub name: String,
    pub nrows: usize,
    pub cols: Vec<Column>,
    /// The column holding the rowid (the INTEGER PRIMARY KEY, or the hidden one).
    pub rowid_col: usize,
    pub indexes: Vec<Index>,
}

impl Table {
    pub fn col_index(&self, name: &str) -> Option<usize> {
        if let Some(i) = self.cols.iter().position(|c| !c.hidden && c.name.eq_ignore_ascii_case(name)) {
            return Some(i);
        }
        if ["rowid", "_rowid_", "oid"].iter().any(|r| r.eq_ignore_ascii_case(name)) {
            return Some(self.rowid_col);
        }
        None
    }

    /// The index on column `col`, if any.
    pub fn index_on(&self, col: usize) -> Option<usize> {
        self.indexes.iter().position(|i| i.col == col)
    }

    /// The position of the row with this rowid.
    #[inline]
    pub fn rowid_pos(&self, rowid: i64) -> Option<usize> {
        let c = self.cols[self.rowid_col].int().expect("rowid is an integer column");
        let e = rowid as i128 - c.base as i128;
        if e < 0 || e > c.maxenc as i128 || self.nrows == 0 {
            return None;
        }
        match c.enc {
            Enc::Dense => Some(e as usize),
            _ => {
                let p = lower_bound(c.data, e as u64);
                (p < self.nrows && c.data.get(p) == e as u64).then_some(p)
            }
        }
    }

    pub fn nzones(&self) -> usize {
        self.nrows.div_ceil(ZONE_ROWS)
    }
}

pub struct Db {
    pub tables: Vec<Table>,
    // Every `'static` slice above points into this mapping; it lives exactly
    // as long as the tables because both are owned here and never handed out.
    _map: Mmap,
}

impl Db {
    pub fn open(path: impl AsRef<Path>) -> Result<Db> {
        let f = std::fs::File::open(path.as_ref())?;
        // SAFETY: the file is opened read-only and treated as immutable.
        let map = unsafe { Mmap::map(&f)? };
        let tables = build::read_catalog(&map)?;
        Ok(Db { tables, _map: map })
    }

    pub fn table(&self, name: &str) -> Option<usize> {
        self.tables.iter().position(|t| t.name.eq_ignore_ascii_case(name))
    }
}

pub(crate) fn format_err(m: impl Into<String>) -> Error {
    Error::Format(m.into())
}
