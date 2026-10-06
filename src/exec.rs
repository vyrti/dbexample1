//! Execution. Each query first lowers its predicates into the encoded domain
//! of each column (once, with the bound values), then runs one of:
//!
//! - **scan aggregates** (`count`/`sum`/`avg`/`total`/`min`/`max`, no GROUP
//!   BY): zone maps skip or accept whole zones, a single predicate runs the
//!   fused count / sum kernels, several predicates build a selection mask per
//!   4096-row block; unfiltered `count(*)` is the row count and unfiltered
//!   `min`/`max` come from the zone maps;
//! - **direct-indexed GROUP BY** for u8/u16 keys: one banked accumulation pass
//!   per aggregate, no hashing, groups come out in key order;
//! - **top-N** for `ORDER BY` integer columns with a `LIMIT`: zones are visited
//!   best-first by their min/max, a SIMD threshold filter drops every row that
//!   cannot enter the heap, and the scan stops once no remaining zone can;
//! - a **generic** path (rowid/index access, joins, hash GROUP BY, sorting) for
//!   everything else in the grammar.

use crate::error::{Error, Result};
use crate::kernels::{self, EncPred, Lanes, ZoneHit};
use crate::plan::{self, Access, AggSpec, Col, GItem, Output, Plan, Pred, Probe};
use crate::sql::{AggFunc, Operand};
use crate::storage::{CmpOp, Db, Enc, Hi8, IntCol, Table, ZONE_ROWS};
use crate::value::Value;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

/// Rows per selection-mask block: 4 KiB of mask stays in L1 with the lanes.
const BLOCK: usize = 4096;

// ---------------------------------------------------------------- results

#[derive(Clone, Debug, PartialEq)]
pub struct QueryResult {
    pub ncols: usize,
    /// Row-major cells.
    pub cells: Vec<Value>,
}

impl QueryResult {
    pub fn rows(&self) -> impl Iterator<Item = &[Value]> {
        self.cells.chunks(self.ncols.max(1))
    }

    pub fn nrows(&self) -> usize {
        self.cells.len().checked_div(self.ncols).unwrap_or(0)
    }

    /// musql harness rendering: every cell type-tagged, `;` after each row.
    pub fn render(&self) -> String {
        let mut s = String::new();
        for r in self.rows() {
            for v in r {
                v.render_into(&mut s);
            }
            s.push(';');
        }
        s
    }
}

// ---------------------------------------------------------------- statements

#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

impl FxHasher {
    #[inline(always)]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut it = bytes.chunks_exact(8);
        for c in &mut it {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let r = it.remainder();
        if !r.is_empty() {
            let mut b = [0u8; 8];
            b[..r.len()].copy_from_slice(r);
            self.add(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.add(x);
    }
    #[inline]
    fn write_i64(&mut self, x: i64) {
        self.add(x as u64);
    }
    #[inline]
    fn write_u8(&mut self, x: u8) {
        self.add(x as u64);
    }
    #[inline]
    fn write_usize(&mut self, x: usize) {
        self.add(x as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type FxBuild = BuildHasherDefault<FxHasher>;

/// A prepared statement.
pub struct Stmt<'db> {
    db: &'db Db,
    plan: Arc<Plan>,
}

impl<'db> Stmt<'db> {
    pub fn columns(&self) -> &[String] {
        &self.plan.columns
    }

    pub fn query(&self, args: &[Value]) -> Result<QueryResult> {
        execute(self.db, &self.plan, args)
    }
}

/// A connection: prepares statements and caches their plans by SQL text, as
/// SQLite's `prepare_cached` and musql's per-session plan cache do.
pub struct Conn<'db> {
    db: &'db Db,
    cache: HashMap<String, Arc<Plan>, FxBuild>,
}

impl<'db> Conn<'db> {
    pub fn new(db: &'db Db) -> Self {
        Conn { db, cache: HashMap::default() }
    }

    pub fn prepare(&mut self, sql: &str) -> Result<Stmt<'db>> {
        let plan = match self.cache.get(sql) {
            Some(p) => p.clone(),
            None => {
                let p = Arc::new(plan::plan(self.db, sql)?);
                self.cache.insert(sql.to_string(), p.clone());
                p
            }
        };
        Ok(Stmt { db: self.db, plan })
    }

    /// Runs `sql` with `args`, planning it only the first time it is seen.
    pub fn query(&mut self, sql: &str, args: &[Value]) -> Result<QueryResult> {
        if let Some(p) = self.cache.get(sql) {
            return execute(self.db, p, args);
        }
        let p = Arc::new(plan::plan(self.db, sql)?);
        self.cache.insert(sql.to_string(), p.clone());
        execute(self.db, &p, args)
    }
}

// ---------------------------------------------------------------- predicates

/// A predicate lowered against bound values.
#[derive(Clone, Copy, Debug)]
enum RKind {
    Enc(EncPred),
    /// Matches every non-NULL row.
    NotNull,
}

#[derive(Clone, Copy, Debug)]
struct RPred {
    col: usize,
    kind: RKind,
}

/// One side's selection: a position range (rowid predicates on a dense rowid
/// become this) plus residual predicates.
struct Side {
    lo: usize,
    hi: usize,
    preds: Vec<RPred>,
    empty: bool,
}

#[inline]
fn arg(args: &[Value], i: usize) -> &Value {
    static NULL: Value = Value::Null;
    args.get(i).unwrap_or(&NULL)
}

fn operand_value<'a>(args: &'a [Value], o: &Operand, tmp: &'a mut Value) -> &'a Value {
    match o {
        Operand::Param(i) => arg(args, *i),
        Operand::Int(v) => {
            *tmp = Value::Int(*v);
            tmp
        }
    }
}

/// The integer interval `col <op> v` admits, or `None` if no row can match
/// (a NULL operand, or an equality with a non-integral REAL). The bool is set
/// when the predicate is `<>` with an in-range integer (returned as `Some(c)`).
enum Bound {
    Never,
    /// `[lo, hi]`
    Interval(i128, i128),
    NotEq(i64),
}

fn bound(op: CmpOp, v: &Value) -> Result<Bound> {
    const MIN: i128 = i64::MIN as i128 - 1;
    const MAX: i128 = i64::MAX as i128 + 1;
    Ok(match v {
        Value::Null => Bound::Never,
        Value::Int(c) => {
            let c = *c as i128;
            match op {
                CmpOp::Eq => Bound::Interval(c, c),
                CmpOp::Ne => Bound::NotEq(c as i64),
                CmpOp::Lt => Bound::Interval(MIN, c - 1),
                CmpOp::Le => Bound::Interval(MIN, c),
                CmpOp::Gt => Bound::Interval(c + 1, MAX),
                CmpOp::Ge => Bound::Interval(c, MAX),
            }
        }
        Value::Real(f) => {
            if f.is_nan() {
                return Ok(Bound::Never);
            }
            let fl = f.floor().clamp(MIN as f64, MAX as f64) as i128;
            let ce = f.ceil().clamp(MIN as f64, MAX as f64) as i128;
            let integral = f.fract() == 0.0 && f.abs() < 9.2e18;
            match op {
                CmpOp::Eq if integral => Bound::Interval(fl, fl),
                CmpOp::Eq => Bound::Never,
                CmpOp::Ne if integral => Bound::NotEq(fl as i64),
                CmpOp::Ne => Bound::Interval(MIN, MAX),
                CmpOp::Lt => Bound::Interval(MIN, ce - 1),
                CmpOp::Le => Bound::Interval(MIN, fl),
                CmpOp::Gt => Bound::Interval(fl + 1, MAX),
                CmpOp::Ge => Bound::Interval(ce, MAX),
            }
        }
        Value::Text(_) => return Err(Error::Bind("TEXT values compared with INTEGER columns".into())),
    })
}

fn resolve_side(t: &Table, preds: &[Pred], args: &[Value]) -> Result<Side> {
    let mut side = Side { lo: 0, hi: t.nrows, preds: Vec::new(), empty: false };
    // Per column: intersected interval and the <> values.
    let mut cols: Vec<(usize, i128, i128, Vec<i64>)> = Vec::new();
    for p in preds {
        let mut tmp = Value::Null;
        let v = operand_value(args, &p.rhs, &mut tmp);
        let e = match cols.iter().position(|c| c.0 == p.col.col) {
            Some(i) => i,
            None => {
                cols.push((p.col.col, i128::MIN, i128::MAX, Vec::new()));
                cols.len() - 1
            }
        };
        match bound(p.op, v)? {
            Bound::Never => {
                side.empty = true;
                return Ok(side);
            }
            Bound::Interval(lo, hi) => {
                cols[e].1 = cols[e].1.max(lo);
                cols[e].2 = cols[e].2.min(hi);
            }
            Bound::NotEq(c) => cols[e].3.push(c),
        }
    }
    for (ci, lo, hi, nes) in cols {
        let c = t.cols[ci].int().expect("planner admits integer columns only");
        let nullable = c.valid.is_some();
        let base = c.base as i128;
        let lowered = if lo == i128::MIN && hi == i128::MAX {
            Some(None)
        } else {
            IntCol::interval(lo.saturating_sub(base), hi.saturating_sub(base), c.maxenc as i128)
        };
        match lowered {
            None => {
                side.empty = true;
                return Ok(side);
            }
            Some(None) => {
                if nullable {
                    side.preds.push(RPred { col: ci, kind: RKind::NotNull });
                }
            }
            Some(Some(p)) => {
                if c.enc == Enc::Dense {
                    let (a, b) = match p {
                        EncPred::Ge(x) => (x, c.maxenc),
                        EncPred::Le(x) => (0, x),
                        EncPred::Range { lo, span } => (lo, lo + span),
                        EncPred::Eq(x) => (x, x),
                        EncPred::Ne(_) => unreachable!(),
                    };
                    side.lo = side.lo.max(a as usize);
                    side.hi = side.hi.min(b as usize + 1);
                } else {
                    side.preds.push(RPred { col: ci, kind: RKind::Enc(p) });
                }
            }
        }
        for v in nes {
            let e = v as i128 - base;
            if (0..=c.maxenc as i128).contains(&e) {
                side.preds.push(RPred { col: ci, kind: RKind::Enc(EncPred::Ne(e as u64)) });
            } else if nullable {
                side.preds.push(RPred { col: ci, kind: RKind::NotNull });
            }
        }
    }
    if side.lo >= side.hi {
        side.empty = true;
    }
    if side.preds.len() > 32 {
        return Err(Error::Unsupported("more than 32 predicates on one table".into()));
    }
    Ok(side)
}

/// The encoding of `v` in column `c`, if `v` is within its range.
#[inline]
fn enc_of(c: &IntCol, v: i64) -> Option<u64> {
    let e = v as i128 - c.base as i128;
    (0..=c.maxenc as i128).contains(&e).then_some(e as u64)
}

fn icol(t: &Table, c: usize) -> &IntCol {
    t.cols[c].int().expect("integer column")
}

/// Whether the row at `pos` satisfies every residual predicate of `side`.
#[inline]
fn test_pos(t: &Table, side: &Side, pos: usize) -> bool {
    if pos < side.lo || pos >= side.hi {
        return false;
    }
    side.preds.iter().all(|p| {
        let c = icol(t, p.col);
        if c.is_null(pos) {
            return false;
        }
        match p.kind {
            RKind::Enc(e) => e.test(c.enc_at(pos)),
            RKind::NotNull => true,
        }
    })
}

/// Writes (or ANDs) the selection mask of one predicate over rows `[s, e)`.
fn eval_mask(t: &Table, p: &RPred, s: usize, e: usize, out: &mut [u8], and: bool) {
    let c = icol(t, p.col);
    match p.kind {
        RKind::Enc(ep) if c.enc != Enc::Dense => {
            if and {
                let mut tmp = [0u8; BLOCK];
                let tm = &mut tmp[..e - s];
                mask_pred(c, ep, s, e, tm);
                kernels::mask_and(out, tm);
            } else {
                mask_pred(c, ep, s, e, out);
            }
        }
        RKind::Enc(ep) => {
            for (i, o) in out.iter_mut().enumerate() {
                let m = 0u8.wrapping_sub(ep.test((s + i) as u64) as u8);
                *o = if and { *o & m } else { m };
            }
        }
        RKind::NotNull => {
            if !and {
                out.fill(0xFF);
            }
        }
    }
    if let Some(v) = c.valid {
        kernels::mask_and(out, &v[s..e]);
    }
}

/// The mask of every predicate whose bit is set in `sig`, over `[s, e)`.
fn eval_masks(t: &Table, side: &Side, sig: u32, s: usize, e: usize, out: &mut [u8]) {
    let mut first = true;
    for (k, p) in side.preds.iter().enumerate() {
        if sig & (1 << k) != 0 {
            eval_mask(t, p, s, e, out, !first);
            first = false;
        }
    }
    if first {
        out.fill(0xFF);
    }
}

/// Calls `f(start, end, sig)` for maximal runs of zones inside the side's
/// range on which every predicate's zone verdict is the same, skipping zones
/// some predicate rules out. `sig` has bit k set when predicate k must be
/// evaluated row by row on the run; `sig == 0` means every row is selected.
fn for_each_run(
    t: &Table,
    side: &Side,
    mut f: impl FnMut(usize, usize, u32) -> Result<bool>,
) -> Result<()> {
    if side.empty {
        return Ok(());
    }
    let (lo, hi) = (side.lo, side.hi);
    let mut run: Option<(usize, usize, u32)> = None;
    let mut z = lo / ZONE_ROWS;
    while z * ZONE_ROWS < hi {
        let s = (z * ZONE_ROWS).max(lo);
        let e = ((z + 1) * ZONE_ROWS).min(hi);
        z += 1;
        let mut sig = 0u32;
        let mut skip = false;
        for (k, p) in side.preds.iter().enumerate() {
            let hit = match p.kind {
                RKind::Enc(ep) => icol(t, p.col).zone_hit(z - 1, ep),
                RKind::NotNull => ZoneHit::Some,
            };
            match hit {
                ZoneHit::None => {
                    skip = true;
                    break;
                }
                ZoneHit::Some => sig |= 1 << k,
                ZoneHit::All => {
                    // NULLs are 0 in the lanes and in the zone maps' eyes absent:
                    // a nullable column still needs its validity applied.
                    if icol(t, p.col).valid.is_some() {
                        sig |= 1 << k;
                    }
                }
            }
        }
        if skip {
            if let Some(r) = run.take()
                && !f(r.0, r.1, r.2)?
            {
                return Ok(());
            }
            continue;
        }
        match &mut run {
            Some(r) if r.2 == sig && r.1 == s => r.1 = e,
            _ => {
                if let Some(r) = run.replace((s, e, sig))
                    && !f(r.0, r.1, r.2)?
                {
                    return Ok(());
                }
            }
        }
    }
    if let Some(r) = run {
        f(r.0, r.1, r.2)?;
    }
    Ok(())
}


// ---------------------------------------------------------------- byte-sliced predicates

/// A predicate split over a column's hi8 plane: rows whose byte satisfies
/// `sure` match outright, rows whose byte is an `edge` value need the full
/// lane, every other row fails.
#[derive(Clone, Copy)]
struct Split {
    sure: Option<EncPred>,
    edge: [u64; 2],
    nedge: usize,
}

fn split(p: EncPred, shift: u32) -> Split {
    let h = |x: u64| x >> shift;
    let one = |sure, e| Split { sure, edge: [e, 0], nedge: 1 };
    match p {
        EncPred::Ge(t) => one((h(t) < 255).then(|| EncPred::Ge(h(t) + 1)), h(t)),
        EncPred::Le(t) => one((h(t) > 0).then(|| EncPred::Le(h(t) - 1)), h(t)),
        EncPred::Eq(c) => one(None, h(c)),
        EncPred::Ne(c) => one(Some(EncPred::Ne(h(c))), h(c)),
        EncPred::Range { lo, span } => {
            let (a, b) = (h(lo), h(lo + span));
            if a == b {
                one(None, a)
            } else {
                let sure = (b >= a + 2).then(|| EncPred::Range { lo: a + 1, span: b - a - 2 });
                Split { sure, edge: [a, b], nedge: 2 }
            }
        }
    }
}

/// The split of `p` over `c`'s hi8 plane, if it has one and the edge bytes
/// hold few enough rows (from the histogram) for the split to pay.
fn sliced(c: &IntCol, p: EncPred) -> Option<(Hi8, Split)> {
    let h = c.hi8?;
    let sp = split(p, h.shift);
    let edge_rows: u64 = sp.edge[..sp.nedge].iter().map(|&b| h.hist[b as usize] as u64).sum();
    (edge_rows * 16 <= h.plane.len() as u64).then_some((h, sp))
}

/// Appends the positions in `[s, e)` whose hi8 byte is an edge value.
fn edge_positions(h: &Hi8, sp: &Split, s: usize, e: usize, out: &mut Vec<u32>) {
    let lanes = Lanes::U8(&h.plane[s..e]);
    for &b in &sp.edge[..sp.nedge] {
        kernels::collect(lanes, EncPred::Eq(b), s as u32, out);
    }
}

/// Rows of `[s, e)` (NOT NULL column) satisfying `p`.
fn count_pred(c: &IntCol, s: usize, e: usize, p: EncPred) -> u64 {
    let Some((h, sp)) = sliced(c, p) else { return kernels::count(c.data.slice(s, e), p) };
    let mut n = sp.sure.map_or(0, |q| kernels::count(Lanes::U8(&h.plane[s..e]), q));
    let mut pos = Vec::new();
    edge_positions(&h, &sp, s, e, &mut pos);
    n += pos.iter().filter(|&&q| p.test(c.data.get(q as usize))).count() as u64;
    n
}

/// Rows of `[s, e)` satisfying `pa` on `a` and `pb` on `b` (NOT NULL columns).
/// Rows off every edge are decided by bytes in one fused pass; rows on an
/// edge of either column are checked exactly.
fn count2_pred(a: &IntCol, pa: EncPred, b: &IntCol, pb: EncPred, s: usize, e: usize) -> u64 {
    let (sa, sb) = (sliced(a, pa), sliced(b, pb));
    if sa.is_none() && sb.is_none() {
        return kernels::count2(a.data.slice(s, e), pa, b.data.slice(s, e), pb);
    }
    let side = |c: &IntCol, p: EncPred, sl: &Option<(Hi8, Split)>| match sl {
        Some((h, sp)) => (Lanes::U8(&h.plane[s..e]), sp.sure),
        None => (c.data.slice(s, e), Some(p)),
    };
    let (la, qa) = side(a, pa, &sa);
    let (lb, qb) = side(b, pb, &sb);
    let mut n = match (qa, qb) {
        (Some(qa), Some(qb)) => kernels::count2(la, qa, lb, qb),
        _ => 0,
    };
    let mut pos = Vec::new();
    for (h, sp) in sa.iter().chain(sb.iter()) {
        edge_positions(h, sp, s, e, &mut pos);
    }
    if sa.is_some() && sb.is_some() {
        pos.sort_unstable();
        pos.dedup();
    }
    n += pos
        .iter()
        .filter(|&&q| pa.test(a.data.get(q as usize)) && pb.test(b.data.get(q as usize)))
        .count() as u64;
    n
}

/// Writes the selection mask of `p` over `[s, e)` into `out`, via the hi8
/// plane when it pays.
fn mask_pred(c: &IntCol, p: EncPred, s: usize, e: usize, out: &mut [u8]) {
    let Some((h, sp)) = sliced(c, p) else { return kernels::mask(c.data.slice(s, e), p, out, false) };
    match sp.sure {
        Some(q) => kernels::mask(Lanes::U8(&h.plane[s..e]), q, out, false),
        None => out.fill(0),
    }
    let mut pos = Vec::new();
    edge_positions(&h, &sp, s, e, &mut pos);
    for q in pos {
        out[q as usize - s] = 0u8.wrapping_sub(p.test(c.data.get(q as usize)) as u8);
    }
}

// ---------------------------------------------------------------- execute

fn limit_offset(plan: &Plan, args: &[Value]) -> Result<(Option<usize>, usize)> {
    let get = |o: &Option<Operand>, what: &str| -> Result<Option<i64>> {
        let Some(o) = o else { return Ok(None) };
        let mut tmp = Value::Null;
        match operand_value(args, o, &mut tmp) {
            Value::Int(v) => Ok(Some(*v)),
            _ => Err(Error::Bind(format!("{what} must be an integer"))),
        }
    };
    let limit = get(&plan.limit, "LIMIT")?.and_then(|l| (l >= 0).then_some(l as usize));
    let offset = get(&plan.offset, "OFFSET")?.map_or(0, |o| o.max(0) as usize);
    Ok((limit, offset))
}

pub(crate) fn execute(db: &Db, plan: &Plan, args: &[Value]) -> Result<QueryResult> {
    let ncols = plan.columns.len();
    let t0 = &db.tables[plan.tables[0]];
    let side0 = resolve_side(t0, &plan.preds[0], args)?;
    let (limit, offset) = limit_offset(plan, args)?;

    if plan.join.is_none() && matches!(plan.access, Access::Scan) {
        match &plan.output {
            Output::Aggs(aggs) if scan_aggs_ok(t0, aggs) => {
                let cells = scan_aggs(t0, &side0, aggs)?;
                return Ok(finish_single(ncols, cells, limit, offset));
            }
            Output::Group { key, items, visible, order } if group_direct_ok(t0, *key, items) => {
                let rows = group_direct(t0, &side0, key.col, items)?;
                return Ok(finish_groups(ncols, rows, *visible, order, limit, offset));
            }
            Output::Rows { items, order } if !order.is_empty() => {
                if let Some(k) = limit.map(|l| l + offset)
                    && top_n_ok(t0, order, k)
                {
                    let pos = top_n(t0, &side0, order, k)?;
                    let mut cells = Vec::with_capacity(ncols * pos.len());
                    for &p in pos.iter().skip(offset) {
                        for c in items {
                            cells.push(t0.cols[c.col].value(p as usize));
                        }
                    }
                    return Ok(QueryResult { ncols, cells });
                }
            }
            _ => {}
        }
    }
    generic(db, plan, args, side0, limit, offset)
}

fn finish_single(ncols: usize, cells: Vec<Value>, limit: Option<usize>, offset: usize) -> QueryResult {
    if offset > 0 || limit == Some(0) {
        QueryResult { ncols, cells: Vec::new() }
    } else {
        QueryResult { ncols, cells }
    }
}

// ---------------------------------------------------------------- scan aggregates

fn scan_aggs_ok(t: &Table, aggs: &[AggSpec]) -> bool {
    aggs.iter().all(|a| a.arg.is_none_or(|c| t.cols[c.col].int().is_some()))
}

#[derive(Clone, Copy)]
struct IAcc {
    /// Non-NULL selected values.
    n: u64,
    /// Sum of their encodings.
    sum: u128,
    min: u64,
    max: u64,
}

impl Default for IAcc {
    fn default() -> Self {
        IAcc { n: 0, sum: 0, min: u64::MAX, max: 0 }
    }
}

fn needs_sum(f: AggFunc) -> bool {
    matches!(f, AggFunc::Sum | AggFunc::Avg | AggFunc::Total)
}

fn needs_minmax(f: AggFunc) -> bool {
    matches!(f, AggFunc::Min | AggFunc::Max)
}

/// Sum of `pos` over `[s, e)`: a dense column's encodings.
fn dense_sum(s: usize, e: usize) -> u128 {
    let (s, e) = (s as u128, e as u128);
    (s + e - 1) * (e - s) / 2
}

fn scan_aggs(t: &Table, side: &Side, aggs: &[AggSpec]) -> Result<Vec<Value>> {
    let mut rows = 0u64;
    let mut acc = vec![IAcc::default(); aggs.len()];
    let mut buf = [0u8; BLOCK];
    let mut tmp = [0u8; BLOCK];

    for_each_run(t, side, |s, e, sig| {
        if sig == 0 {
            rows += (e - s) as u64;
            for (a, st) in aggs.iter().zip(acc.iter_mut()) {
                let Some(col) = a.arg else { continue };
                let c = icol(t, col.col);
                st.n += match c.valid {
                    None => (e - s) as u64,
                    Some(v) => kernels::mask_count(&v[s..e]),
                };
                if needs_sum(a.func) {
                    st.sum += match (c.enc, c.valid) {
                        (Enc::Dense, _) => dense_sum(s, e),
                        (_, None) => kernels::sum_count(c.data.slice(s, e), EncPred::Ge(0)).1,
                        (_, Some(v)) => kernels::masked_sum(c.data.slice(s, e), &v[s..e]),
                    };
                }
                if needs_minmax(a.func) {
                    let (mn, mx) = run_minmax(c, t.nrows, s, e);
                    st.min = st.min.min(mn);
                    st.max = st.max.max(mx);
                }
            }
            return Ok(true);
        }
        // count(*) under exactly two predicates on NOT NULL stored columns:
        // the fused two-column kernel.
        if sig.count_ones() == 2 && aggs.iter().all(|a| a.arg.is_none()) {
            let i = sig.trailing_zeros() as usize;
            let j = (sig & (sig - 1)).trailing_zeros() as usize;
            let (p, q) = (side.preds[i], side.preds[j]);
            let ok = |r: &RPred| {
                let c = icol(t, r.col);
                matches!(r.kind, RKind::Enc(_)) && c.enc != Enc::Dense && c.valid.is_none()
            };
            if ok(&p) && ok(&q) {
                let (RKind::Enc(ep), RKind::Enc(eq)) = (p.kind, q.kind) else { unreachable!() };
                rows += count2_pred(icol(t, p.col), ep, icol(t, q.col), eq, s, e);
                return Ok(true);
            }
        }
        // One predicate on a NOT NULL stored column, and only aggregates of
        // that same column (or count(*)): the fused kernels, no mask.
        if sig.count_ones() == 1 {
            let p = side.preds[sig.trailing_zeros() as usize];
            let pc = icol(t, p.col);
            let fusable = matches!(p.kind, RKind::Enc(_))
                && pc.enc != Enc::Dense
                && pc.valid.is_none()
                && aggs.iter().all(|a| match a.arg {
                    None => true,
                    Some(c) => c.col == p.col && !needs_minmax(a.func),
                });
            if fusable {
                let RKind::Enc(ep) = p.kind else { unreachable!() };
                let lanes = pc.data.slice(s, e);
                let (n, sum) = if aggs.iter().any(|a| a.arg.is_some()) {
                    kernels::sum_count(lanes, ep)
                } else {
                    (count_pred(pc, s, e, ep), 0)
                };
                rows += n;
                for (a, st) in aggs.iter().zip(acc.iter_mut()) {
                    if a.arg.is_some() {
                        st.n += n;
                        st.sum += sum;
                    }
                }
                return Ok(true);
            }
        }
        let mut b = s;
        while b < e {
            let be = (b + BLOCK).min(e);
            let m = &mut buf[..be - b];
            eval_masks(t, side, sig, b, be, m);
            let sel = kernels::mask_count(m);
            rows += sel;
            if sel > 0 {
                for (a, st) in aggs.iter().zip(acc.iter_mut()) {
                    let Some(col) = a.arg else { continue };
                    let c = icol(t, col.col);
                    let mm: &[u8] = match c.valid {
                        None => m,
                        Some(v) => {
                            let tm = &mut tmp[..be - b];
                            tm.copy_from_slice(m);
                            kernels::mask_and(tm, &v[b..be]);
                            tm
                        }
                    };
                    st.n += if c.valid.is_none() { sel } else { kernels::mask_count(mm) };
                    if needs_sum(a.func) {
                        st.sum += if c.enc == Enc::Dense {
                            mm.iter().enumerate().filter(|x| *x.1 != 0).map(|(i, _)| (b + i) as u128).sum()
                        } else {
                            kernels::masked_sum(c.data.slice(b, be), mm)
                        };
                    }
                    if needs_minmax(a.func) {
                        for (i, &x) in mm.iter().enumerate() {
                            if x != 0 {
                                let v = c.enc_at(b + i);
                                st.min = st.min.min(v);
                                st.max = st.max.max(v);
                            }
                        }
                    }
                }
            }
            b = be;
        }
        Ok(true)
    })?;

    aggs.iter()
        .zip(&acc)
        .map(|(a, st)| {
            let Some(col) = a.arg else { return Ok(Value::Int(rows as i64)) };
            finish_int_agg(a.func, icol(t, col.col), st.n, st.sum, st.min, st.max)
        })
        .collect()
}

/// `(min, max)` encodings of the non-NULL values in `[s, e)`: whole zones from
/// the zone map, partial zones at the edges by scanning.
fn run_minmax(c: &IntCol, nrows: usize, s: usize, e: usize) -> (u64, u64) {
    let (mut mn, mut mx) = (u64::MAX, 0u64);
    let mut i = s;
    while i < e {
        let z = i / ZONE_ROWS;
        let zend = ((z + 1) * ZONE_ROWS).min(nrows);
        let ze = zend.min(e);
        if i == z * ZONE_ROWS && ze == zend {
            let [lo, hi] = c.zones[z];
            if lo <= hi {
                mn = mn.min(lo);
                mx = mx.max(hi);
            }
        } else {
            for p in i..ze {
                if !c.is_null(p) {
                    let v = c.enc_at(p);
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
            }
        }
        i = ze;
    }
    (mn, mx)
}

fn finish_int_agg(f: AggFunc, c: &IntCol, n: u64, sum: u128, min: u64, max: u64) -> Result<Value> {
    let total = || n as i128 * c.base as i128 + sum as i128;
    Ok(match f {
        AggFunc::Count => Value::Int(n as i64),
        AggFunc::Sum if n == 0 => Value::Null,
        AggFunc::Sum => {
            let v = total();
            if v < i64::MIN as i128 || v > i64::MAX as i128 {
                return Err(Error::IntegerOverflow);
            }
            Value::Int(v as i64)
        }
        AggFunc::Total => Value::Real(if n == 0 { 0.0 } else { total() as f64 }),
        AggFunc::Avg if n == 0 => Value::Null,
        AggFunc::Avg => Value::Real(total() as f64 / n as f64),
        AggFunc::Min if n == 0 => Value::Null,
        AggFunc::Min => Value::Int(c.decode(min)),
        AggFunc::Max if n == 0 => Value::Null,
        AggFunc::Max => Value::Int(c.decode(max)),
    })
}

// ---------------------------------------------------------------- direct GROUP BY

fn group_direct_ok(t: &Table, key: Col, items: &[GItem]) -> bool {
    let Some(k) = t.cols[key.col].int() else { return false };
    let key_ok = matches!(k.enc, Enc::U8 | Enc::U16) && k.valid.is_none();
    key_ok
        && items.iter().all(|i| match i {
            GItem::Key => true,
            GItem::Agg(a) => a.arg.is_none_or(|c| {
                t.cols[c.col]
                    .int()
                    .is_some_and(|c| c.valid.is_none() && matches!(c.enc, Enc::U8 | Enc::U16 | Enc::U32))
            }),
        })
}

/// Four accumulator banks per group, so a run of equal keys never makes one
/// read-modify-write wait on the previous one's store.
const BANKS: usize = 4;

trait GLane: Copy {
    fn u(self) -> usize;
    fn w(self) -> u64;
}
impl GLane for u8 {
    #[inline(always)]
    fn u(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn w(self) -> u64 {
        self as u64
    }
}
impl GLane for u16 {
    #[inline(always)]
    fn u(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn w(self) -> u64 {
        self as u64
    }
}
impl GLane for u32 {
    #[inline(always)]
    fn u(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn w(self) -> u64 {
        self as u64
    }
}

/// `acc[key*4 + bank] += value` (masked: values of unselected rows add 0).
#[inline(always)]
fn group_sum<K: GLane, V: GLane>(keys: &[K], vals: &[V], mask: Option<&[u8]>, acc: &mut [u64]) {
    let n = keys.len();
    let vals = &vals[..n];
    let main = n & !3;
    match mask {
        None => {
            let mut i = 0;
            while i < main {
                acc[keys[i].u() * BANKS] += vals[i].w();
                acc[keys[i + 1].u() * BANKS + 1] += vals[i + 1].w();
                acc[keys[i + 2].u() * BANKS + 2] += vals[i + 2].w();
                acc[keys[i + 3].u() * BANKS + 3] += vals[i + 3].w();
                i += 4;
            }
            for i in main..n {
                acc[keys[i].u() * BANKS] += vals[i].w();
            }
        }
        Some(m) => {
            for i in 0..n {
                acc[keys[i].u() * BANKS + (i & 3)] += vals[i].w() & 0u64.wrapping_sub((m[i] & 1) as u64);
            }
        }
    }
}

/// Count and sum in one pass: `acc[(key*4 + bank)*2] += 1`, `[+1] += value`,
/// both in the same cache line.
#[inline(always)]
fn group_count_sum<K: GLane, V: GLane>(keys: &[K], vals: &[V], mask: Option<&[u8]>, acc: &mut [u64]) {
    let n = keys.len();
    let vals = &vals[..n];
    let main = n & !3;
    match mask {
        None => {
            let mut i = 0;
            while i < main {
                let (g0, g1, g2, g3) = (
                    keys[i].u() * 2 * BANKS,
                    keys[i + 1].u() * 2 * BANKS + 2,
                    keys[i + 2].u() * 2 * BANKS + 4,
                    keys[i + 3].u() * 2 * BANKS + 6,
                );
                acc[g0] += 1;
                acc[g0 + 1] += vals[i].w();
                acc[g1] += 1;
                acc[g1 + 1] += vals[i + 1].w();
                acc[g2] += 1;
                acc[g2 + 1] += vals[i + 2].w();
                acc[g3] += 1;
                acc[g3 + 1] += vals[i + 3].w();
                i += 4;
            }
            for i in main..n {
                let g = keys[i].u() * 2 * BANKS;
                acc[g] += 1;
                acc[g + 1] += vals[i].w();
            }
        }
        Some(m) => {
            for i in 0..n {
                let g = keys[i].u() * 2 * BANKS + 2 * (i & 3);
                let sel = (m[i] & 1) as u64;
                acc[g] += sel;
                acc[g + 1] += vals[i].w() & 0u64.wrapping_sub(sel);
            }
        }
    }
}

/// Banked min/max: `mins[key*4 + bank]`.
#[inline(always)]
fn group_minmax<K: GLane, V: GLane>(
    keys: &[K],
    vals: &[V],
    mask: Option<&[u8]>,
    mins: &mut [u64],
    maxs: &mut [u64],
) {
    let vals = &vals[..keys.len()];
    match mask {
        None => {
            for (i, (k, v)) in keys.iter().zip(vals).enumerate() {
                let g = k.u() * BANKS + (i & 3);
                let v = v.w();
                mins[g] = mins[g].min(v);
                maxs[g] = maxs[g].max(v);
            }
        }
        Some(m) => {
            for (i, (k, v)) in keys.iter().zip(vals).enumerate() {
                if m[i] != 0 {
                    let g = k.u() * BANKS + (i & 3);
                    let v = v.w();
                    mins[g] = mins[g].min(v);
                    maxs[g] = maxs[g].max(v);
                }
            }
        }
    }
}

macro_rules! by_key {
    ($kl:expr, |$k:ident| $body:expr) => {
        match $kl {
            kernels::Lanes::U8($k) => $body,
            kernels::Lanes::U16($k) => $body,
            _ => unreachable!(),
        }
    };
}
macro_rules! by_val {
    ($vl:expr, |$v:ident| $body:expr) => {
        match $vl {
            kernels::Lanes::U8($v) => $body,
            kernels::Lanes::U16($v) => $body,
            kernels::Lanes::U32($v) => $body,
            _ => unreachable!(),
        }
    };
}

/// Rows `[key value, items...]` in ascending key order.
fn group_direct(t: &Table, side: &Side, key: usize, items: &[GItem]) -> Result<Vec<Vec<Value>>> {
    let kc = icol(t, key);
    // Table size: every value the key lane can hold, so no index can be out
    // of range and the compiler drops the bounds checks (u8: 256 groups).
    let g = match kc.data {
        kernels::Lanes::U8(_) => 256,
        _ => kc.maxenc as usize + 1,
    };
    // Distinct value columns: the first sum column fused with the count,
    // further ones banked separately, min/max plain.
    let mut sum_cols: Vec<usize> = Vec::new();
    let mut mm_cols: Vec<usize> = Vec::new();
    for it in items {
        if let GItem::Agg(a) = it
            && let Some(c) = a.arg
        {
            if needs_sum(a.func) && !sum_cols.contains(&c.col) {
                sum_cols.push(c.col);
            }
            if needs_minmax(a.func) && !mm_cols.contains(&c.col) {
                mm_cols.push(c.col);
            }
        }
    }
    // [(g*BANKS + b)*2] = count, [+1] = sum of sum_cols[0] (if any)
    let mut cs = vec![0u64; g * BANKS * 2];
    let mut sums = vec![vec![0u64; g * BANKS]; sum_cols.len().saturating_sub(1)];
    let mut mins = vec![vec![u64::MAX; g * BANKS]; mm_cols.len()];
    let mut maxs = vec![vec![0u64; g * BANKS]; mm_cols.len()];
    let mut buf = [0u8; BLOCK];

    let mut pass = |s: usize, e: usize, mask: Option<&[u8]>| {
        let kl = kc.data.slice(s, e);
        match sum_cols.first() {
            Some(&c0) => {
                let vl = icol(t, c0).data.slice(s, e);
                by_key!(kl, |k| by_val!(vl, |v| group_count_sum(k, v, mask, &mut cs)));
            }
            // no sum column: the keys stand in as values; the sums are unused
            None => by_key!(kl, |k| group_count_sum(k, k, mask, &mut cs)),
        }
        for (ci, &c) in sum_cols.iter().enumerate().skip(1) {
            let vl = icol(t, c).data.slice(s, e);
            by_key!(kl, |k| by_val!(vl, |v| group_sum(k, v, mask, &mut sums[ci - 1])));
        }
        for (ci, &c) in mm_cols.iter().enumerate() {
            let vl = icol(t, c).data.slice(s, e);
            by_key!(kl, |k| by_val!(vl, |v| group_minmax(k, v, mask, &mut mins[ci], &mut maxs[ci])));
        }
    };
    for_each_run(t, side, |s, e, sig| {
        if sig == 0 {
            // Blocked too, so each pass's key and value lanes are L1-resident.
            let mut b = s;
            while b < e {
                let be = (b + BLOCK * 4).min(e);
                pass(b, be, None);
                b = be;
            }
        } else {
            let mut b = s;
            while b < e {
                let be = (b + BLOCK).min(e);
                let m = &mut buf[..be - b];
                eval_masks(t, side, sig, b, be, m);
                pass(b, be, Some(m));
                b = be;
            }
        }
        Ok(true)
    })?;

    let mut rows = Vec::new();
    for gi in 0..g {
        let bank = |b: usize| (gi * BANKS + b) * 2;
        let cnt: u64 = (0..BANKS).map(|b| cs[bank(b)]).sum();
        if cnt == 0 {
            continue;
        }
        let mut row = Vec::with_capacity(items.len());
        for it in items {
            row.push(match it {
                GItem::Key => Value::Int(kc.decode(gi as u64)),
                GItem::Agg(a) => match a.arg {
                    None => Value::Int(cnt as i64),
                    Some(c) => {
                        let vc = icol(t, c.col);
                        let sum: u128 = match sum_cols.iter().position(|&x| x == c.col) {
                            Some(0) => (0..BANKS).map(|b| cs[bank(b) + 1] as u128).sum(),
                            Some(i) => sums[i - 1][gi * BANKS..gi * BANKS + BANKS].iter().map(|&x| x as u128).sum(),
                            None => 0,
                        };
                        let (mn, mx) = mm_cols.iter().position(|&x| x == c.col).map_or((0, 0), |i| {
                            let r = gi * BANKS..gi * BANKS + BANKS;
                            (mins[i][r.clone()].iter().copied().min().unwrap(), maxs[i][r].iter().copied().max().unwrap())
                        });
                        finish_int_agg(a.func, vc, cnt, sum, mn, mx)?
                    }
                },
            });
        }
        rows.push(row);
    }
    Ok(rows)
}

fn finish_groups(
    ncols: usize,
    mut rows: Vec<Vec<Value>>,
    visible: usize,
    order: &[(usize, bool)],
    limit: Option<usize>,
    offset: usize,
) -> QueryResult {
    // Rows arrive in ascending key order; ORDER BY re-sorts (stably).
    if !order.is_empty() {
        rows.sort_by(|a, b| {
            for &(i, desc) in order {
                let o = a[i].sql_cmp(&b[i]);
                let o = if desc { o.reverse() } else { o };
                if o != Ordering::Equal {
                    return o;
                }
            }
            Ordering::Equal
        });
    }
    let take = limit.unwrap_or(usize::MAX);
    let mut cells = Vec::with_capacity(ncols * rows.len().min(take));
    for mut r in rows.into_iter().skip(offset).take(take) {
        r.truncate(visible);
        cells.extend(r);
    }
    QueryResult { ncols, cells }
}

// ---------------------------------------------------------------- top-N

const MAX_ORDER_KEYS: usize = 4;
const MAX_TOP_N: usize = 1 << 16;

fn top_n_ok(t: &Table, order: &[(Col, bool)], k: usize) -> bool {
    k <= MAX_TOP_N
        && order.len() <= MAX_ORDER_KEYS
        && order.iter().all(|(c, _)| t.cols[c.col].int().is_some_and(|c| c.valid.is_none()))
}

/// Positions of the first `k` rows in ORDER BY order (ties by position).
fn top_n(t: &Table, side: &Side, order: &[(Col, bool)], k: usize) -> Result<Vec<u32>> {
    if k == 0 || side.empty {
        return Ok(Vec::new());
    }
    let cols: Vec<(&IntCol, bool)> = order.iter().map(|(c, d)| (icol(t, c.col), *d)).collect();
    // Order-preserving u64 per key: ascending encodings, or maxenc - enc.
    let key_of = |pos: usize| -> [u64; MAX_ORDER_KEYS] {
        let mut k = [0u64; MAX_ORDER_KEYS];
        for (j, (c, desc)) in cols.iter().enumerate() {
            let e = c.enc_at(pos);
            k[j] = if *desc { c.maxenc - e } else { e };
        }
        k
    };
    let (c1, desc1) = cols[0];

    // A dense first key (the rowid) is already in order: walk it.
    if c1.enc == Enc::Dense {
        let mut out = Vec::with_capacity(k);
        let mut buf = [0u8; BLOCK];
        let mut blocks: Vec<(usize, usize)> = Vec::new();
        let mut b = side.lo;
        while b < side.hi {
            let be = (b + BLOCK).min(side.hi);
            blocks.push((b, be));
            b = be;
        }
        if desc1 {
            blocks.reverse();
        }
        for (b, be) in blocks {
            let m = &mut buf[..be - b];
            eval_masks(t, side, if side.preds.is_empty() { 0 } else { u32::MAX >> (32 - side.preds.len()) }, b, be, m);
            let mut pos = Vec::new();
            kernels::mask_positions(m, b as u32, &mut pos);
            if desc1 {
                pos.reverse();
            }
            for p in pos {
                out.push(p);
                if out.len() == k {
                    return Ok(out);
                }
            }
        }
        return Ok(out);
    }

    // Zones best-first by the first key's zone bound.
    let mut zones: Vec<(u64, usize)> = Vec::new();
    let (zlo, zhi) = (side.lo / ZONE_ROWS, side.hi.div_ceil(ZONE_ROWS));
    'z: for z in zlo..zhi {
        let [mn, mx] = c1.zones[z];
        if mn > mx {
            continue;
        }
        for p in &side.preds {
            if let RKind::Enc(ep) = p.kind
                && icol(t, p.col).zone_hit(z, ep) == ZoneHit::None
            {
                continue 'z;
            }
        }
        // transformed best first-key value in the zone (smaller = better)
        let best = if desc1 { c1.maxenc - mx } else { mn };
        zones.push((best, z));
    }
    zones.sort_unstable();
    // The k-th best zone bound is itself a key some row has, and k distinct
    // rows (each zone's best) are at least that good -- so no row whose first
    // key is worse can be in the top k. With k <= #zones this bounds the scan
    // before reading a single row: only rows at or above it pass the SIMD
    // filter, and zones whose best is below it are never visited.
    // Only sound when every zone's best row qualifies: no predicates, and the
    // range covers whole zones (the whole table).
    let whole = side.preds.is_empty() && side.lo == 0 && side.hi == t.nrows;
    let zone_bound = (whole && k <= zones.len()).then(|| zones[k - 1].0);

    // Candidates accumulate in a buffer; whenever it reaches 2k a quickselect
    // cuts it back to the best k, and the k-th best becomes the threshold the
    // SIMD prefilter and the zone loop test against. O(n) overall, unlike a
    // heap's O(n log k) with expensive sifts for large k (OFFSET pagination).
    type Entry = ([u64; MAX_ORDER_KEYS], u32);
    let mut cands: Vec<Entry> = Vec::with_capacity(2 * k + ZONE_ROWS);
    let mut thr: Option<Entry> = None;
    let compact = |cands: &mut Vec<Entry>| -> Entry {
        if cands.len() > k {
            cands.select_nth_unstable(k - 1);
            cands.truncate(k);
        }
        *cands.iter().max().unwrap()
    };
    let mut buf = [0u8; ZONE_ROWS];
    let mut tbuf = [0u8; ZONE_ROWS];
    let mut cand: Vec<u32> = Vec::with_capacity(ZONE_ROWS);
    for (best, z) in zones {
        if let Some(th) = thr
            && best > th.0[0]
        {
            break; // no remaining zone holds a row that beats the k-th best
        }
        if zone_bound.is_some_and(|b| best > b) {
            break;
        }
        let s = (z * ZONE_ROWS).max(side.lo);
        let e = ((z + 1) * ZONE_ROWS).min(side.hi);
        // Rows whose first key is no worse than the threshold's (ties kept;
        // the later keys decide them).
        let bound = match (thr, zone_bound) {
            (Some(th), Some(b)) => Some(th.0[0].min(b)),
            (Some(th), None) => Some(th.0[0]),
            (None, b) => b,
        };
        let thr_pred = bound.map(|b| if desc1 { EncPred::Ge(c1.maxenc - b) } else { EncPred::Le(b) });
        let mut sig = 0u32;
        for (j, p) in side.preds.iter().enumerate() {
            let hit = match p.kind {
                RKind::Enc(ep) => icol(t, p.col).zone_hit(z, ep),
                RKind::NotNull => ZoneHit::Some,
            };
            if hit != ZoneHit::All || icol(t, p.col).valid.is_some() {
                sig |= 1 << j;
            }
        }
        cand.clear();
        let lanes = c1.data.slice(s, e);
        match (sig, thr_pred) {
            (0, None) => cand.extend(s as u32..e as u32),
            (0, Some(tp)) => kernels::collect(lanes, tp, s as u32, &mut cand),
            (_, tp) => {
                let m = &mut buf[..e - s];
                eval_masks(t, side, sig, s, e, m);
                if let Some(tp) = tp {
                    let tm = &mut tbuf[..e - s];
                    kernels::mask(lanes, tp, tm, false);
                    kernels::mask_and(m, tm);
                }
                kernels::mask_positions(m, s as u32, &mut cand);
            }
        }
        match thr {
            None => cands.extend(cand.iter().map(|&p| (key_of(p as usize), p))),
            Some(th) => {
                for &p in &cand {
                    let entry = (key_of(p as usize), p);
                    if entry < th {
                        cands.push(entry);
                    }
                }
            }
        }
        if cands.len() >= k && (thr.is_none() || cands.len() >= 2 * k) {
            thr = Some(compact(&mut cands));
        }
    }
    if cands.len() > k {
        cands.select_nth_unstable(k - 1);
        cands.truncate(k);
    }
    cands.sort_unstable();
    Ok(cands.into_iter().map(|(_, p)| p).collect())
}

// ---------------------------------------------------------------- generic path

/// Positions of the driving table, in position order, at most `cap`.
fn driver_positions(t: &Table, access: &Access, side: &Side, args: &[Value], cap: usize) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    if side.empty || cap == 0 {
        return Ok(out);
    }
    match access {
        Access::RowidEq(o) => {
            let mut tmp = Value::Null;
            if let Bound::Interval(lo, hi) = bound(CmpOp::Eq, operand_value(args, o, &mut tmp))?
                && lo == hi
                && let Some(p) = t.rowid_pos(lo as i64)
                && test_pos(t, side, p)
            {
                out.push(p as u32);
            }
        }
        Access::IndexEq { index, key } => {
            let ix = &t.indexes[*index];
            let c = icol(t, ix.col);
            let mut tmp = Value::Null;
            if let Bound::Interval(lo, hi) = bound(CmpOp::Eq, operand_value(args, key, &mut tmp))?
                && lo == hi
                && let Some(e) = enc_of(c, lo as i64)
            {
                let (a, b) = ix.equal_range(e);
                for &p in &ix.pos[a..b] {
                    if test_pos(t, side, p as usize) {
                        out.push(p);
                        if out.len() >= cap {
                            break;
                        }
                    }
                }
            }
        }
        Access::Scan => {
            let mut buf = [0u8; BLOCK];
            for_each_run(t, side, |s, e, sig| {
                if sig == 0 {
                    let e2 = e.min(s.saturating_add(cap - out.len()));
                    out.extend(s as u32..e2 as u32);
                } else {
                    let mut b = s;
                    while b < e && out.len() < cap {
                        let be = (b + BLOCK).min(e);
                        let m = &mut buf[..be - b];
                        eval_masks(t, side, sig, b, be, m);
                        kernels::mask_positions(m, b as u32, &mut out);
                        b = be;
                    }
                    out.truncate(cap);
                }
                Ok(out.len() < cap)
            })?;
        }
    }
    Ok(out)
}

/// Driver positions, or (driver, inner) pairs after the join.
enum Tuples {
    One(Vec<u32>),
    Two(Vec<(u32, u32)>),
}

impl Tuples {
    fn len(&self) -> usize {
        match self {
            Tuples::One(v) => v.len(),
            Tuples::Two(v) => v.len(),
        }
    }
}

struct Ctx<'a> {
    t: [&'a Table; 2],
    tuples: Tuples,
}

impl Ctx<'_> {
    #[inline]
    fn pos(&self, side: usize, i: usize) -> usize {
        match &self.tuples {
            Tuples::One(v) => v[i] as usize,
            Tuples::Two(v) => (if side == 0 { v[i].0 } else { v[i].1 }) as usize,
        }
    }

    #[inline]
    fn value(&self, c: Col, i: usize) -> Value {
        self.t[c.side].cols[c.col].value(self.pos(c.side, i))
    }
}

fn generic(
    db: &Db,
    plan: &Plan,
    args: &[Value],
    side0: Side,
    limit: Option<usize>,
    offset: usize,
) -> Result<QueryResult> {
    let ncols = plan.columns.len();
    let t0 = &db.tables[plan.tables[0]];
    let t1 = plan.tables.get(1).map_or(t0, |&i| &db.tables[i]);

    // Stop early when nothing downstream needs more rows than LIMIT+OFFSET.
    let streaming = matches!(&plan.output, Output::Rows { order, .. } if order.is_empty());
    let want = match (streaming, limit) {
        (true, Some(l)) => l.saturating_add(offset),
        _ => usize::MAX,
    };

    // count(*) over an index seek with no residual predicate: the range size.
    if let (Output::Aggs(aggs), Access::IndexEq { index, key }, None) = (&plan.output, &plan.access, &plan.join)
        && side0.preds.is_empty()
        && side0.lo == 0
        && side0.hi == t0.nrows
        && aggs.iter().all(|a| a.func == AggFunc::Count && a.arg.is_none())
    {
        let ix = &t0.indexes[*index];
        let mut tmp = Value::Null;
        let mut n = 0i64;
        if let Bound::Interval(lo, hi) = bound(CmpOp::Eq, operand_value(args, key, &mut tmp))?
            && lo == hi
            && let Some(e) = enc_of(icol(t0, ix.col), lo as i64)
        {
            let (a, b) = ix.equal_range(e);
            n = (b - a) as i64;
        }
        return Ok(finish_single(ncols, vec![Value::Int(n); aggs.len()], limit, offset));
    }

    let outer_cap = if plan.join.is_none() { want } else { usize::MAX };
    let outer = driver_positions(t0, &plan.access, &side0, args, outer_cap)?;
    let tuples = match &plan.join {
        None => Tuples::One(outer),
        Some(j) => {
            let side1 = resolve_side(t1, &plan.preds[1], args)?;
            let oc = icol(t0, j.outer);
            let ic = icol(t1, j.inner);
            let mut pairs = Vec::new();
            if !side1.empty {
                match j.probe {
                    Probe::Rowid => {
                        for &p in &outer {
                            if let Some(v) = oc.value_at(p as usize)
                                && let Some(ip) = t1.rowid_pos(v)
                                && test_pos(t1, &side1, ip)
                            {
                                pairs.push((p, ip as u32));
                            }
                        }
                    }
                    Probe::Index(ix) => {
                        let ix = &t1.indexes[ix];
                        for &p in &outer {
                            if let Some(v) = oc.value_at(p as usize)
                                && let Some(e) = enc_of(ic, v)
                            {
                                let (a, b) = ix.equal_range(e);
                                for &ip in &ix.pos[a..b] {
                                    if test_pos(t1, &side1, ip as usize) {
                                        pairs.push((p, ip));
                                    }
                                }
                            }
                        }
                    }
                    Probe::Hash => {
                        let inner = driver_positions(t1, &Access::Scan, &side1, args, usize::MAX)?;
                        let mut h: HashMap<i64, Vec<u32>, FxBuild> = HashMap::default();
                        for ip in inner {
                            if let Some(v) = ic.value_at(ip as usize) {
                                h.entry(v).or_default().push(ip);
                            }
                        }
                        for &p in &outer {
                            if let Some(v) = oc.value_at(p as usize)
                                && let Some(ips) = h.get(&v)
                            {
                                pairs.extend(ips.iter().map(|&ip| (p, ip)));
                            }
                        }
                    }
                }
            }
            if streaming {
                pairs.truncate(want);
            }
            Tuples::Two(pairs)
        }
    };
    let ctx = Ctx { t: [t0, t1], tuples };
    let n = ctx.tuples.len();

    match &plan.output {
        Output::Aggs(aggs) => {
            let mut accs = vec![GAcc::default(); aggs.len()];
            for i in 0..n {
                for (a, acc) in aggs.iter().zip(accs.iter_mut()) {
                    acc.add(a, &ctx, i);
                }
            }
            let cells = aggs.iter().zip(&accs).map(|(a, acc)| acc.finish(a, n)).collect::<Result<_>>()?;
            Ok(finish_single(ncols, cells, limit, offset))
        }
        Output::Group { key, items, visible, order } => {
            let mut groups: HashMap<HKey, (Value, u64, Vec<GAcc>), FxBuild> = HashMap::default();
            for i in 0..n {
                let kv = ctx.value(*key, i);
                let e = groups.entry(HKey::of(&kv)).or_insert_with(|| (kv, 0, vec![GAcc::default(); items.len()]));
                e.1 += 1;
                for (it, acc) in items.iter().zip(e.2.iter_mut()) {
                    if let GItem::Agg(a) = it {
                        acc.add(a, &ctx, i);
                    }
                }
            }
            let mut rows: Vec<(Value, Vec<Value>)> = Vec::with_capacity(groups.len());
            for (_, (kv, cnt, accs)) in groups {
                let mut row = Vec::with_capacity(items.len());
                for (it, acc) in items.iter().zip(&accs) {
                    row.push(match it {
                        GItem::Key => kv.clone(),
                        GItem::Agg(a) => acc.finish(a, cnt as usize)?,
                    });
                }
                rows.push((kv, row));
            }
            rows.sort_by(|a, b| a.0.sql_cmp(&b.0));
            let rows = rows.into_iter().map(|r| r.1).collect();
            Ok(finish_groups(ncols, rows, *visible, order, limit, offset))
        }
        Output::Rows { items, order } => {
            let mut idx: Vec<usize> = (0..n).collect();
            if !order.is_empty() {
                let keys: Vec<Vec<Value>> =
                    (0..n).map(|i| order.iter().map(|(c, _)| ctx.value(*c, i)).collect()).collect();
                idx.sort_by(|&a, &b| {
                    for (j, (_, desc)) in order.iter().enumerate() {
                        let o = keys[a][j].sql_cmp(&keys[b][j]);
                        let o = if *desc { o.reverse() } else { o };
                        if o != Ordering::Equal {
                            return o;
                        }
                    }
                    Ordering::Equal
                });
            }
            let take = limit.unwrap_or(usize::MAX);
            let mut cells = Vec::new();
            for &i in idx.iter().skip(offset).take(take) {
                for c in items {
                    cells.push(ctx.value(*c, i));
                }
            }
            Ok(QueryResult { ncols, cells })
        }
    }
}

/// Hashable group key (REALs by bit pattern; -0.0 and 0.0 are not merged).
#[derive(PartialEq, Eq, Hash)]
enum HKey {
    Null,
    Int(i64),
    Real(u64),
    Text(Box<str>),
}

impl HKey {
    fn of(v: &Value) -> HKey {
        match v {
            Value::Null => HKey::Null,
            Value::Int(i) => HKey::Int(*i),
            Value::Real(f) => HKey::Real(f.to_bits()),
            Value::Text(s) => HKey::Text(s.clone()),
        }
    }
}

/// Value-based aggregate state for the generic path.
#[derive(Clone, Default)]
struct GAcc {
    n: u64,
    isum: i128,
    fsum: f64,
    real: bool,
    min: Option<Value>,
    max: Option<Value>,
}

impl GAcc {
    #[inline]
    fn add(&mut self, a: &AggSpec, ctx: &Ctx, i: usize) {
        let Some(c) = a.arg else { return };
        let v = ctx.value(c, i);
        match &v {
            Value::Null => return,
            Value::Int(x) => self.isum += *x as i128,
            Value::Real(f) => {
                self.fsum += f;
                self.real = true;
            }
            Value::Text(_) => {}
        }
        self.n += 1;
        if needs_minmax(a.func) {
            if self.min.as_ref().is_none_or(|m| v.sql_cmp(m) == Ordering::Less) {
                self.min = Some(v.clone());
            }
            if self.max.as_ref().is_none_or(|m| v.sql_cmp(m) == Ordering::Greater) {
                self.max = Some(v);
            }
        }
    }

    fn finish(&self, a: &AggSpec, rows: usize) -> Result<Value> {
        let total = self.fsum + self.isum as f64;
        Ok(match a.func {
            AggFunc::Count if a.arg.is_none() => Value::Int(rows as i64),
            AggFunc::Count => Value::Int(self.n as i64),
            AggFunc::Sum if self.n == 0 => Value::Null,
            AggFunc::Sum if self.real => Value::Real(total),
            AggFunc::Sum => {
                if self.isum < i64::MIN as i128 || self.isum > i64::MAX as i128 {
                    return Err(Error::IntegerOverflow);
                }
                Value::Int(self.isum as i64)
            }
            AggFunc::Total => Value::Real(total),
            AggFunc::Avg if self.n == 0 => Value::Null,
            AggFunc::Avg => Value::Real(total / self.n as f64),
            AggFunc::Min => self.min.clone().unwrap_or(Value::Null),
            AggFunc::Max => self.max.clone().unwrap_or(Value::Null),
        })
    }
}
