//! Writing a `.snw` file from column values, and reading its catalog back.

use super::*;
use std::io::{BufWriter, Seek, SeekFrom, Write};

#[cfg(target_endian = "big")]
compile_error!("the .snw format is little-endian and mapped in place");

pub enum ColumnValues {
    Int(Vec<Option<i64>>),
    Real(Vec<Option<f64>>),
    Text(Vec<Option<String>>),
}

impl ColumnValues {
    fn len(&self) -> usize {
        match self {
            ColumnValues::Int(v) => v.len(),
            ColumnValues::Real(v) => v.len(),
            ColumnValues::Text(v) => v.len(),
        }
    }
}

pub struct ColumnBuilder {
    pub name: String,
    pub decl: String,
    pub hidden: bool,
    pub values: ColumnValues,
}

pub struct TableBuilder {
    pub name: String,
    pub cols: Vec<ColumnBuilder>,
    /// Must be an `Int` column, non-NULL and strictly increasing: rows are
    /// stored in rowid order.
    pub rowid_col: usize,
    /// `(name, column)` single-column indexes.
    pub indexes: Vec<(String, usize)>,
}

#[derive(Default)]
pub struct Builder {
    pub tables: Vec<TableBuilder>,
}

struct Out {
    w: BufWriter<std::fs::File>,
    off: u64,
}

impl Out {
    fn section(&mut self, bytes: &[u8]) -> Result<(u64, u64)> {
        let pad = (64 - (self.off % 64)) % 64;
        self.w.write_all(&[0u8; 64][..pad as usize])?;
        self.off += pad;
        let at = self.off;
        self.w.write_all(bytes)?;
        self.off += bytes.len() as u64;
        Ok((at, bytes.len() as u64))
    }
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data integers/floats, little-endian target.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[derive(Default)]
struct Cat(Vec<u8>);
impl Cat {
    fn u8(&mut self, x: u8) {
        self.0.push(x);
    }
    fn u32(&mut self, x: u32) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    fn u64(&mut self, x: u64) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    fn i64(&mut self, x: i64) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
    fn sect(&mut self, s: (u64, u64)) {
        self.u64(s.0);
        self.u64(s.1);
    }
}

/// The encoded form of an integer column.
struct EncodedInt {
    enc: Enc,
    base: i64,
    maxenc: u64,
    lanes: Vec<u64>,
    zones: Vec<[u64; 2]>,
}

fn encode_int(v: &[Option<i64>], allow_dense: bool) -> EncodedInt {
    let n = v.len();
    let (mut min, mut max) = (i64::MAX, i64::MIN);
    for x in v.iter().flatten() {
        min = min.min(*x);
        max = max.max(*x);
    }
    if min > max {
        // empty or all NULL
        min = 0;
        max = 0;
    }
    let dense = allow_dense
        && n > 0
        && v.iter().enumerate().all(|(i, x)| matches!(x, Some(x) if (*x as i128) == min as i128 + i as i128));
    let range = (max as i128 - min as i128) as u128;
    let enc = if dense {
        Enc::Dense
    } else if range <= u8::MAX as u128 {
        Enc::U8
    } else if range <= u16::MAX as u128 {
        Enc::U16
    } else if range <= u32::MAX as u128 {
        Enc::U32
    } else {
        Enc::U64
    };
    let lanes: Vec<u64> = v.iter().map(|x| x.map_or(0, |x| x.wrapping_sub(min) as u64)).collect();
    let mut zones = Vec::with_capacity(n.div_ceil(ZONE_ROWS));
    for (z, chunk) in v.chunks(ZONE_ROWS).enumerate() {
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for (i, x) in chunk.iter().enumerate() {
            if x.is_some() {
                let e = lanes[z * ZONE_ROWS + i];
                lo = lo.min(e);
                hi = hi.max(e);
            }
        }
        zones.push([lo, hi]);
    }
    EncodedInt { enc, base: min, maxenc: range as u64, lanes, zones }
}

/// The byte-sliced plane of a column wider than a byte: `(shift, enc >> shift
/// per row, histogram of those bytes)`.
fn hi8_plane(e: &EncodedInt) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    if !matches!(e.enc, Enc::U16 | Enc::U32 | Enc::U64) {
        return None;
    }
    let bits = 64 - e.maxenc.leading_zeros();
    let shift = bits.saturating_sub(8);
    let plane: Vec<u8> = e.lanes.iter().map(|&x| (x >> shift) as u8).collect();
    let mut hist = vec![0u32; 256];
    for &b in &plane {
        hist[b as usize] += 1;
    }
    Some((shift, plane, hist))
}

fn lane_bytes(enc: Enc, lanes: &[u64]) -> Vec<u8> {
    match enc {
        Enc::Dense => Vec::new(),
        Enc::U8 => lanes.iter().map(|&x| x as u8).collect(),
        Enc::U16 => as_bytes(&lanes.iter().map(|&x| x as u16).collect::<Vec<_>>()).to_vec(),
        Enc::U32 => as_bytes(&lanes.iter().map(|&x| x as u32).collect::<Vec<_>>()).to_vec(),
        Enc::U64 => as_bytes(lanes).to_vec(),
    }
}

fn validity(present: impl Iterator<Item = bool>) -> Option<Vec<u8>> {
    let v: Vec<u8> = present.map(|p| if p { 0xFF } else { 0 }).collect();
    if v.iter().all(|&b| b != 0) { None } else { Some(v) }
}

impl Builder {
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let f = std::fs::File::create(path.as_ref())?;
        let mut out = Out { w: BufWriter::with_capacity(1 << 20, f), off: 0 };
        out.w.write_all(MAGIC)?;
        out.w.write_all(&[0u8; 16])?;
        out.off = 24;
        let mut cat = Cat::default();
        cat.u32(self.tables.len() as u32);
        for t in &self.tables {
            let nrows = t.cols.first().map_or(0, |c| c.values.len());
            if t.cols.iter().any(|c| c.values.len() != nrows) {
                return Err(format_err(format!("table {}: columns of different lengths", t.name)));
            }
            match t.cols.get(t.rowid_col).map(|c| &c.values) {
                Some(ColumnValues::Int(v)) => {
                    let ok = v.iter().all(|x| x.is_some())
                        && v.windows(2).all(|w| w[0].unwrap() < w[1].unwrap());
                    if !ok {
                        return Err(format_err(format!("table {}: rowids must be non-NULL and increasing", t.name)));
                    }
                }
                _ => return Err(format_err(format!("table {}: rowid column is not an integer column", t.name))),
            }
            cat.str(&t.name);
            cat.u64(nrows as u64);
            cat.u32(t.rowid_col as u32);
            cat.u32(t.cols.len() as u32);
            let mut encoded: Vec<Option<EncodedInt>> = Vec::new();
            for (ci, c) in t.cols.iter().enumerate() {
                cat.str(&c.name);
                cat.str(&c.decl);
                cat.u8(c.hidden as u8);
                match &c.values {
                    ColumnValues::Int(v) => {
                        cat.u8(0);
                        let valid = validity(v.iter().map(|x| x.is_some()));
                        cat.sect(out.section(valid.as_deref().unwrap_or(&[]))?);
                        let e = encode_int(v, ci == t.rowid_col);
                        cat.u8(e.enc as u8);
                        cat.i64(e.base);
                        cat.u64(e.maxenc);
                        cat.sect(out.section(&lane_bytes(e.enc, &e.lanes))?);
                        cat.sect(out.section(as_bytes(&e.zones))?);
                        match hi8_plane(&e) {
                            Some((shift, plane, hist)) => {
                                cat.u8(1);
                                cat.u8(shift as u8);
                                cat.sect(out.section(&plane)?);
                                cat.sect(out.section(as_bytes(&hist))?);
                            }
                            None => cat.u8(0),
                        }
                        encoded.push(Some(e));
                    }
                    ColumnValues::Real(v) => {
                        cat.u8(1);
                        let valid = validity(v.iter().map(|x| x.is_some()));
                        cat.sect(out.section(valid.as_deref().unwrap_or(&[]))?);
                        let d: Vec<f64> = v.iter().map(|x| x.unwrap_or(0.0)).collect();
                        cat.sect(out.section(as_bytes(&d))?);
                        encoded.push(None);
                    }
                    ColumnValues::Text(v) => {
                        cat.u8(2);
                        let valid = validity(v.iter().map(|x| x.is_some()));
                        cat.sect(out.section(valid.as_deref().unwrap_or(&[]))?);
                        let mut offs = Vec::with_capacity(v.len() + 1);
                        let mut bytes = Vec::new();
                        offs.push(0u32);
                        for s in v {
                            bytes.extend_from_slice(s.as_deref().unwrap_or("").as_bytes());
                            let o = u32::try_from(bytes.len())
                                .map_err(|_| format_err("text column larger than 4 GiB"))?;
                            offs.push(o);
                        }
                        cat.sect(out.section(as_bytes(&offs))?);
                        cat.sect(out.section(&bytes)?);
                        encoded.push(None);
                    }
                }
            }
            let idx: Vec<_> = t
                .indexes
                .iter()
                .filter(|(_, c)| matches!(encoded.get(*c), Some(Some(e)) if e.enc != Enc::Dense))
                .collect();
            cat.u32(idx.len() as u32);
            for (name, c) in idx {
                let e = encoded[*c].as_ref().unwrap();
                let ColumnValues::Int(v) = &t.cols[*c].values else { unreachable!() };
                let mut pairs: Vec<(u64, u32)> = (0..nrows)
                    .filter(|&i| v[i].is_some())
                    .map(|i| (e.lanes[i], i as u32))
                    .collect();
                pairs.sort_unstable();
                let keys: Vec<u64> = pairs.iter().map(|p| p.0).collect();
                let pos: Vec<u32> = pairs.iter().map(|p| p.1).collect();
                cat.str(name);
                cat.u32(*c as u32);
                cat.sect(out.section(&lane_bytes(e.enc, &keys))?);
                cat.sect(out.section(as_bytes(&pos))?);
            }
        }
        let cat_off = out.section(&cat.0)?;
        out.w.flush()?;
        let mut f = out.w.into_inner().map_err(|e| Error::Io(e.into_error()))?;
        f.seek(SeekFrom::Start(8))?;
        f.write_all(&cat_off.0.to_le_bytes())?;
        f.write_all(&cat_off.1.to_le_bytes())?;
        f.sync_all()?;
        Ok(())
    }
}

// ---- Reading.

struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.b.get(self.at..self.at + n).ok_or_else(|| format_err("truncated catalog"))?;
        self.at += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| format_err("bad string"))
    }
    fn sect(&mut self) -> Result<(u64, u64)> {
        Ok((self.u64()?, self.u64()?))
    }
}

/// A typed view of a section of the mapping.
fn view<T>(map: &Mmap, s: (u64, u64)) -> Result<&'static [T]> {
    let (off, len) = (s.0 as usize, s.1 as usize);
    let size = std::mem::size_of::<T>();
    if off.checked_add(len).is_none_or(|e| e > map.len()) || len % size != 0 {
        return Err(format_err("section out of bounds"));
    }
    if len == 0 {
        return Ok(&[]);
    }
    let p = unsafe { map.as_ptr().add(off) };
    if (p as usize) % std::mem::align_of::<T>() != 0 {
        return Err(format_err("misaligned section"));
    }
    // SAFETY: in bounds, aligned, plain-old-data; the mapping outlives every
    // table built from it (both are owned by `Db`), so 'static never escapes.
    Ok(unsafe { std::slice::from_raw_parts(p as *const T, len / size) })
}

fn lanes(map: &Mmap, enc: Enc, s: (u64, u64)) -> Result<Lanes<'static>> {
    Ok(match enc {
        Enc::Dense => Lanes::U64(&[]),
        Enc::U8 => Lanes::U8(view(map, s)?),
        Enc::U16 => Lanes::U16(view(map, s)?),
        Enc::U32 => Lanes::U32(view(map, s)?),
        Enc::U64 => Lanes::U64(view(map, s)?),
    })
}

fn opt_valid(map: &Mmap, s: (u64, u64), n: usize) -> Result<Option<&'static [u8]>> {
    if s.1 == 0 {
        return Ok(None);
    }
    let v: &[u8] = view(map, s)?;
    if v.len() != n {
        return Err(format_err("validity length mismatch"));
    }
    Ok(Some(v))
}

pub(crate) fn read_catalog(map: &Mmap) -> Result<Vec<Table>> {
    if map.len() < 24 || &map[..8] != MAGIC {
        return Err(format_err("not a sinew database"));
    }
    let cat_off = u64::from_le_bytes(map[8..16].try_into().unwrap()) as usize;
    let cat_len = u64::from_le_bytes(map[16..24].try_into().unwrap()) as usize;
    let cat = map
        .get(cat_off..cat_off.saturating_add(cat_len))
        .ok_or_else(|| format_err("catalog out of bounds"))?;
    let mut r = Rd { b: cat, at: 0 };
    let ntables = r.u32()?;
    let mut tables = Vec::new();
    for _ in 0..ntables {
        let name = r.str()?;
        let nrows = r.u64()? as usize;
        let rowid_col = r.u32()? as usize;
        let ncols = r.u32()? as usize;
        let nzones = nrows.div_ceil(ZONE_ROWS);
        let mut cols = Vec::with_capacity(ncols);
        for _ in 0..ncols {
            let cname = r.str()?;
            let decl = r.str()?;
            let hidden = r.u8()? != 0;
            let kind = r.u8()?;
            let valid = opt_valid(map, r.sect()?, nrows)?;
            let data = match kind {
                0 => {
                    let enc = match r.u8()? {
                        0 => Enc::Dense,
                        1 => Enc::U8,
                        2 => Enc::U16,
                        3 => Enc::U32,
                        4 => Enc::U64,
                        _ => return Err(format_err("bad encoding")),
                    };
                    let base = r.i64()?;
                    let maxenc = r.u64()?;
                    let data = lanes(map, enc, r.sect()?)?;
                    let zones: &[[u64; 2]] = view(map, r.sect()?)?;
                    if (enc != Enc::Dense && data.len() != nrows) || zones.len() != nzones {
                        return Err(format_err(format!("column {cname}: length mismatch")));
                    }
                    if enc == Enc::Dense && nrows > 0 && maxenc != nrows as u64 - 1 {
                        return Err(format_err(format!("column {cname}: bad dense range")));
                    }
                    let hi8 = if r.u8()? != 0 {
                        let shift = r.u8()? as u32;
                        let plane: &[u8] = view(map, r.sect()?)?;
                        let hist: &[u32] = view(map, r.sect()?)?;
                        if plane.len() != nrows || hist.len() != 256 || shift > 56 {
                            return Err(format_err(format!("column {cname}: bad hi8 plane")));
                        }
                        Some(Hi8 { plane, shift, hist })
                    } else {
                        None
                    };
                    ColData::Int(IntCol { enc, base, maxenc, data, zones, valid, hi8 })
                }
                1 => {
                    let data: &[f64] = view(map, r.sect()?)?;
                    if data.len() != nrows {
                        return Err(format_err(format!("column {cname}: length mismatch")));
                    }
                    ColData::Real { data, valid }
                }
                2 => {
                    let offs: &[u32] = view(map, r.sect()?)?;
                    let bytes: &[u8] = view(map, r.sect()?)?;
                    let ok = offs.len() == nrows + 1
                        && offs[0] == 0
                        && offs.windows(2).all(|w| w[0] <= w[1])
                        && offs[nrows] as usize <= bytes.len();
                    if !ok {
                        return Err(format_err(format!("column {cname}: bad text offsets")));
                    }
                    ColData::Text { offs, bytes, valid }
                }
                _ => return Err(format_err("bad column kind")),
            };
            cols.push(Column { name: cname, decl, hidden, data });
        }
        if !matches!(cols.get(rowid_col), Some(Column { data: ColData::Int(_), .. })) {
            return Err(format_err(format!("table {name}: bad rowid column")));
        }
        let nidx = r.u32()?;
        let mut indexes = Vec::new();
        for _ in 0..nidx {
            let iname = r.str()?;
            let col = r.u32()? as usize;
            let enc = match cols.get(col).map(|c| &c.data) {
                Some(ColData::Int(c)) => c.enc,
                _ => return Err(format_err(format!("index {iname}: bad column"))),
            };
            let keys = lanes(map, enc, r.sect()?)?;
            let pos: &[u32] = view(map, r.sect()?)?;
            if keys.len() != pos.len() || pos.iter().any(|&p| p as usize >= nrows) {
                return Err(format_err(format!("index {iname}: corrupt")));
            }
            indexes.push(Index { name: iname, col, keys, pos });
        }
        tables.push(Table { name, nrows, cols, rowid_col, indexes });
    }
    Ok(tables)
}
