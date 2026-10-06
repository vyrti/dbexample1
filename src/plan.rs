//! Planning: name resolution, access-path choice (rowid lookup, index seek,
//! scan), join strategy (rowid probe, index probe, hash), and the output shape
//! the executor dispatches on.

use crate::error::{Error, Result};
use crate::sql::{self, AggFunc, ColRef, Expr, Operand, OrderKey, Select, SelectItem};
use crate::storage::{CmpOp, ColData, Db};

/// A column of one of the (at most two) tables in the query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Col {
    /// 0 = driving table, 1 = joined (inner) table.
    pub side: usize,
    pub col: usize,
}

#[derive(Clone, Debug)]
pub struct Pred {
    pub col: Col,
    pub op: CmpOp,
    pub rhs: Operand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AggSpec {
    pub func: AggFunc,
    /// `None` is `COUNT(*)`.
    pub arg: Option<Col>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GItem {
    Key,
    Agg(AggSpec),
}

#[derive(Clone, Debug)]
pub enum Output {
    /// Aggregates with no GROUP BY: exactly one row.
    Aggs(Vec<AggSpec>),
    Group {
        key: Col,
        /// Visible items first, then hidden ones only ORDER BY needs.
        items: Vec<GItem>,
        visible: usize,
        /// `(item index, descending)`.
        order: Vec<(usize, bool)>,
    },
    Rows {
        items: Vec<Col>,
        order: Vec<(Col, bool)>,
    },
}

#[derive(Clone, Debug)]
pub enum Access {
    Scan,
    RowidEq(Operand),
    IndexEq { index: usize, key: Operand },
}

#[derive(Clone, Copy, Debug)]
pub enum Probe {
    Rowid,
    Index(usize),
    Hash,
}

#[derive(Clone, Debug)]
pub struct JoinSpec {
    /// Join column on the driving side.
    pub outer: usize,
    /// Join column on the inner side.
    pub inner: usize,
    pub probe: Probe,
}

#[derive(Debug)]
pub struct Plan {
    pub columns: Vec<String>,
    pub nparams: usize,
    /// Database table index per side.
    pub tables: Vec<usize>,
    pub access: Access,
    pub join: Option<JoinSpec>,
    /// Predicates per side (residual after `access`).
    pub preds: [Vec<Pred>; 2],
    pub output: Output,
    pub limit: Option<Operand>,
    pub offset: Option<Operand>,
}

struct Scope<'a> {
    db: &'a Db,
    /// (db table, name, alias) in FROM order.
    from: Vec<(usize, String, Option<String>)>,
}

impl Scope<'_> {
    /// Resolves to (FROM position, column).
    fn resolve(&self, c: &ColRef) -> Result<(usize, usize)> {
        let mut found = None;
        for (fi, (ti, name, alias)) in self.from.iter().enumerate() {
            if let Some(q) = &c.table {
                let matches = match alias {
                    Some(a) => a.eq_ignore_ascii_case(q),
                    None => name.eq_ignore_ascii_case(q),
                };
                if !matches {
                    continue;
                }
            }
            if let Some(ci) = self.db.tables[*ti].col_index(&c.col) {
                if found.is_some() {
                    return Err(Error::Semantic(format!("ambiguous column name: {}", c.col)));
                }
                found = Some((fi, ci));
            }
        }
        found.ok_or_else(|| match &c.table {
            Some(t) => Error::Semantic(format!("no such column: {t}.{}", c.col)),
            None => Error::Semantic(format!("no such column: {}", c.col)),
        })
    }
}

fn agg_name(f: AggFunc, arg: &Option<ColRef>) -> String {
    let fname = match f {
        AggFunc::Count => "count",
        AggFunc::Sum => "sum",
        AggFunc::Min => "min",
        AggFunc::Max => "max",
        AggFunc::Avg => "avg",
        AggFunc::Total => "total",
    };
    match arg {
        None => format!("{fname}(*)"),
        Some(c) => match &c.table {
            Some(t) => format!("{fname}({t}.{})", c.col),
            None => format!("{fname}({})", c.col),
        },
    }
}

pub fn plan(db: &Db, sql_text: &str) -> Result<Plan> {
    let s: Select = sql::parse(sql_text)?;
    if s.from.len() > 2 {
        return Err(Error::Unsupported("joins of more than two tables".into()));
    }
    let mut from = Vec::new();
    for t in &s.from {
        let ti = db.table(&t.name).ok_or_else(|| Error::Semantic(format!("no such table: {}", t.name)))?;
        from.push((ti, t.name.clone(), t.alias.clone()));
    }
    let scope = Scope { db, from };

    // ---- Join condition.
    let mut join_cols: Option<[usize; 2]> = None; // column per FROM position
    match (s.from.len(), s.joins.as_slice()) {
        (1, []) => {}
        (1, _) => return Err(Error::Unsupported("column-to-column comparisons in a single-table query".into())),
        (2, [(a, b)]) => {
            let (fa, ca) = scope.resolve(a)?;
            let (fb, cb) = scope.resolve(b)?;
            if fa == fb {
                return Err(Error::Unsupported("column-to-column comparisons within one table".into()));
            }
            let mut jc = [0; 2];
            jc[fa] = ca;
            jc[fb] = cb;
            for (fi, &c) in jc.iter().enumerate() {
                if db.tables[scope.from[fi].0].cols[c].int().is_none() {
                    return Err(Error::Unsupported("joins on non-integer columns".into()));
                }
            }
            join_cols = Some(jc);
        }
        (2, []) => return Err(Error::Unsupported("cross joins".into())),
        _ => return Err(Error::Unsupported("more than one join condition".into())),
    }

    // ---- Predicates, by FROM position.
    let mut fpreds: [Vec<(usize, CmpOp, Operand)>; 2] = [Vec::new(), Vec::new()];
    for c in &s.conds {
        let (fi, ci) = scope.resolve(&c.col)?;
        if db.tables[scope.from[fi].0].cols[ci].int().is_none() {
            return Err(Error::Unsupported(format!("comparisons on non-integer column {}", c.col.col)));
        }
        fpreds[fi].push((ci, c.op, c.rhs));
    }

    // ---- Driving side and access path.
    let access_for = |fi: usize| -> (u8, Access, Option<usize>) {
        let t = &db.tables[scope.from[fi].0];
        for (k, (ci, op, rhs)) in fpreds[fi].iter().enumerate() {
            if *op == CmpOp::Eq && *ci == t.rowid_col {
                return (0, Access::RowidEq(*rhs), Some(k));
            }
        }
        for (k, (ci, op, rhs)) in fpreds[fi].iter().enumerate() {
            if *op == CmpOp::Eq
                && let Some(ix) = t.index_on(*ci)
            {
                return (1, Access::IndexEq { index: ix, key: *rhs }, Some(k));
            }
        }
        (2, Access::Scan, None)
    };
    let probe_for = |fi: usize, col: usize| -> Probe {
        let t = &db.tables[scope.from[fi].0];
        if col == t.rowid_col {
            Probe::Rowid
        } else if let Some(ix) = t.index_on(col) {
            Probe::Index(ix)
        } else {
            Probe::Hash
        }
    };
    let driver = if s.from.len() == 1 {
        0
    } else {
        let jc = join_cols.unwrap();
        let score = |fi: usize| {
            let (a, _, _) = access_for(fi);
            let p = match probe_for(1 - fi, jc[1 - fi]) {
                Probe::Rowid => 0,
                Probe::Index(_) => 1,
                Probe::Hash => 2,
            };
            // Prefer the cheaper access; then a cheaper probe; then scanning
            // the larger table against a hash of the smaller.
            (a, p, std::cmp::Reverse(db.tables[scope.from[fi].0].nrows))
        };
        if score(1) < score(0) { 1 } else { 0 }
    };
    // side index for each FROM position
    let side_of = |fi: usize| if fi == driver { 0 } else { 1 };
    let (_, access, used) = access_for(driver);
    let mut preds: [Vec<Pred>; 2] = [Vec::new(), Vec::new()];
    for fi in 0..s.from.len() {
        for (k, (ci, op, rhs)) in fpreds[fi].iter().enumerate() {
            if fi == driver && Some(k) == used {
                continue;
            }
            preds[side_of(fi)].push(Pred { col: Col { side: side_of(fi), col: *ci }, op: *op, rhs: *rhs });
        }
    }
    let join = join_cols.map(|jc| {
        let inner = 1 - driver;
        JoinSpec { outer: jc[driver], inner: jc[inner], probe: probe_for(inner, jc[inner]) }
    });
    let tables: Vec<usize> = if s.from.len() == 1 {
        vec![scope.from[0].0]
    } else {
        vec![scope.from[driver].0, scope.from[1 - driver].0]
    };
    let to_col = |c: &ColRef| -> Result<Col> {
        let (fi, ci) = scope.resolve(c)?;
        Ok(Col { side: side_of(fi), col: ci })
    };
    let col_kind = |c: Col| &db.tables[tables[c.side]].cols[c.col].data;

    // ---- Select list.
    let mut columns = Vec::new();
    enum It {
        Col(Col),
        Agg(AggSpec),
    }
    let mut items: Vec<It> = Vec::new();
    let mut item_exprs: Vec<(Option<Expr>, Option<String>)> = Vec::new();
    for it in &s.items {
        match it {
            SelectItem::Star(q) => {
                let mut any = false;
                for (fi, (ti, name, alias)) in scope.from.iter().enumerate() {
                    if let Some(q) = q {
                        let m = alias.as_deref().unwrap_or(name);
                        if !m.eq_ignore_ascii_case(q) {
                            continue;
                        }
                    }
                    any = true;
                    for (ci, c) in db.tables[*ti].cols.iter().enumerate() {
                        if c.hidden {
                            continue;
                        }
                        columns.push(c.name.clone());
                        items.push(It::Col(Col { side: side_of(fi), col: ci }));
                        item_exprs.push((None, None));
                    }
                }
                if !any {
                    return Err(Error::Semantic(format!("no such table: {}", q.clone().unwrap_or_default())));
                }
            }
            SelectItem::Expr { expr, alias } => {
                match expr {
                    Expr::Col(c) => {
                        let col = to_col(c)?;
                        columns.push(alias.clone().unwrap_or_else(|| c.col.clone()));
                        items.push(It::Col(col));
                    }
                    Expr::Agg(f, arg) => {
                        let a = arg.as_ref().map(&to_col).transpose()?;
                        if let Some(a) = a
                            && matches!(f, AggFunc::Sum | AggFunc::Avg | AggFunc::Total)
                            && matches!(col_kind(a), ColData::Text { .. })
                        {
                            return Err(Error::Unsupported("sum/avg/total over TEXT".into()));
                        }
                        columns.push(alias.clone().unwrap_or_else(|| agg_name(*f, arg)));
                        items.push(It::Agg(AggSpec { func: *f, arg: a }));
                    }
                }
                item_exprs.push((Some(expr.clone()), alias.clone()));
            }
        }
    }

    // ORDER BY key -> visible item index, if it names one.
    let order_item = |k: &OrderKey| -> Result<Option<usize>> {
        match k {
            OrderKey::Position(n) => {
                if *n > items.len() {
                    return Err(Error::Semantic(format!("ORDER BY term out of range: {n}")));
                }
                Ok(Some(n - 1))
            }
            OrderKey::Expr(e) => {
                if let Expr::Col(ColRef { table: None, col }) = e
                    && let Some(i) =
                        item_exprs.iter().position(|(_, a)| a.as_deref().is_some_and(|a| a.eq_ignore_ascii_case(col)))
                {
                    return Ok(Some(i));
                }
                Ok(item_exprs.iter().position(|(x, _)| x.as_ref() == Some(e)))
            }
        }
    };

    let has_agg = items.iter().any(|i| matches!(i, It::Agg(_)));
    let output = if let Some(g) = &s.group_by {
        let key = to_col(g)?;
        let mut gitems = Vec::new();
        for it in &items {
            gitems.push(match it {
                It::Col(c) if *c == key => GItem::Key,
                It::Col(_) => return Err(Error::Unsupported("non-aggregate columns other than the GROUP BY key".into())),
                It::Agg(a) => GItem::Agg(*a),
            });
        }
        let visible = gitems.len();
        let mut order = Vec::new();
        for o in &s.order_by {
            let idx = match order_item(&o.key)? {
                Some(i) => i,
                None => {
                    let gi = match &o.key {
                        OrderKey::Expr(Expr::Col(c)) if to_col(c)? == key => GItem::Key,
                        OrderKey::Expr(Expr::Agg(f, a)) => {
                            GItem::Agg(AggSpec { func: *f, arg: a.as_ref().map(&to_col).transpose()? })
                        }
                        _ => return Err(Error::Unsupported("ORDER BY a non-grouped column".into())),
                    };
                    gitems.push(gi);
                    gitems.len() - 1
                }
            };
            order.push((idx, o.desc));
        }
        Output::Group { key, items: gitems, visible, order }
    } else if has_agg {
        let mut aggs = Vec::new();
        for it in &items {
            match it {
                It::Agg(a) => aggs.push(*a),
                It::Col(_) => return Err(Error::Unsupported("mixing aggregates and bare columns".into())),
            }
        }
        // A single row: ORDER BY is irrelevant once it resolves.
        for o in &s.order_by {
            order_item(&o.key)?;
        }
        Output::Aggs(aggs)
    } else {
        let cols: Vec<Col> = items.iter().map(|i| if let It::Col(c) = i { *c } else { unreachable!() }).collect();
        let mut order = Vec::new();
        for o in &s.order_by {
            let c = match order_item(&o.key)? {
                Some(i) => cols[i],
                None => match &o.key {
                    OrderKey::Expr(Expr::Col(c)) => to_col(c)?,
                    _ => return Err(Error::Semantic("aggregate in ORDER BY of a non-aggregate query".into())),
                },
            };
            order.push((c, o.desc));
        }
        Output::Rows { items: cols, order }
    };

    Ok(Plan { columns, nparams: s.nparams, tables, access, join, preds, output, limit: s.limit, offset: s.offset })
}
