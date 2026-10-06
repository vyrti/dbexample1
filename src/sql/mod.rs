//! Lexer and recursive-descent parser for the supported SELECT subset:
//!
//! ```text
//! SELECT item, ... FROM t [AS a] [[INNER] JOIN u [AS b] ON x = y [AND cond]... | , u]
//!   [WHERE cond AND ...] [GROUP BY col] [ORDER BY key [ASC|DESC], ...]
//!   [LIMIT n [OFFSET m]]
//! item := * | col | COUNT(*) | COUNT(col) | SUM(col) | MIN(col) | MAX(col)
//!         | AVG(col) | TOTAL(col)   [[AS] alias]
//! cond := operand op operand        (op: = == <> != < <= > >=; one side a column)
//!       | col BETWEEN operand AND operand
//! operand := col | ? | ?N | integer literal
//! key  := col | alias | position | an item's aggregate
//! ```

use crate::error::{Error, Result};
use crate::storage::CmpOp;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Int(i64),
    Str(String),
    Param(Option<usize>),
    Sym(&'static str),
}

fn lex(s: &str) -> Result<Vec<Tok>> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            b'-' if b.get(i + 1) == Some(&b'-') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'0'..=b'9' => {
                let st = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                if i < b.len() && (b[i] == b'.' || b[i] == b'e' || b[i] == b'E') {
                    return Err(Error::Unsupported("REAL literals".into()));
                }
                let v: i64 = s[st..i].parse().map_err(|_| Error::Unsupported("integer literal out of range".into()))?;
                out.push(Tok::Int(v));
            }
            b'?' => {
                i += 1;
                let st = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                let n = if i > st {
                    let n: usize = s[st..i].parse().map_err(|_| Error::Parse("bad parameter number".into()))?;
                    if n == 0 {
                        return Err(Error::Parse("?0 is not a valid parameter".into()));
                    }
                    Some(n)
                } else {
                    None
                };
                out.push(Tok::Param(n));
            }
            b'\'' => {
                let mut v = String::new();
                i += 1;
                loop {
                    match b.get(i) {
                        None => return Err(Error::Parse("unterminated string".into())),
                        Some(b'\'') if b.get(i + 1) == Some(&b'\'') => {
                            v.push('\'');
                            i += 2;
                        }
                        Some(b'\'') => {
                            i += 1;
                            break;
                        }
                        Some(_) => {
                            let ch = s[i..].chars().next().unwrap();
                            v.push(ch);
                            i += ch.len_utf8();
                        }
                    }
                }
                out.push(Tok::Str(v));
            }
            b'"' | b'`' | b'[' => {
                let close = if c == b'[' { b']' } else { c };
                let st = i + 1;
                let mut j = st;
                while j < b.len() && b[j] != close {
                    j += 1;
                }
                if j >= b.len() {
                    return Err(Error::Parse("unterminated identifier".into()));
                }
                out.push(Tok::Ident(s[st..j].to_string()));
                i = j + 1;
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let st = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'$') {
                    i += 1;
                }
                out.push(Tok::Ident(s[st..i].to_string()));
            }
            _ => {
                const SYMS: [&str; 16] =
                    ["<>", "!=", "<=", ">=", "==", "(", ")", ",", ".", "*", "=", "<", ">", ";", "-", "+"];
                let sym = SYMS
                    .iter()
                    .find(|sym| s[i..].starts_with(*sym))
                    .ok_or_else(|| Error::Parse(format!("unexpected character {:?}", c as char)))?;
                out.push(Tok::Sym(sym));
                i += sym.len();
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColRef {
    pub table: Option<String>,
    pub col: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AggFunc {
    Count,
    Sum,
    Min,
    Max,
    Avg,
    Total,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    Col(ColRef),
    /// `None` argument is `COUNT(*)`.
    Agg(AggFunc, Option<ColRef>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// 0-based parameter index.
    Param(usize),
    Int(i64),
}

#[derive(Clone, Debug)]
pub enum SelectItem {
    Star(Option<String>),
    Expr { expr: Expr, alias: Option<String> },
}

#[derive(Clone, Debug)]
pub struct TableRef {
    pub name: String,
    pub alias: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Cond {
    pub col: ColRef,
    pub op: CmpOp,
    pub rhs: Operand,
}

#[derive(Clone, Debug)]
pub enum OrderKey {
    Expr(Expr),
    /// 1-based result column.
    Position(usize),
}

#[derive(Clone, Debug)]
pub struct OrderItem {
    pub key: OrderKey,
    pub desc: bool,
}

#[derive(Clone, Debug)]
pub struct Select {
    pub items: Vec<SelectItem>,
    pub from: Vec<TableRef>,
    /// Column-equals-column conditions: join conditions.
    pub joins: Vec<(ColRef, ColRef)>,
    pub conds: Vec<Cond>,
    pub group_by: Option<ColRef>,
    pub order_by: Vec<OrderItem>,
    pub limit: Option<Operand>,
    pub offset: Option<Operand>,
    pub nparams: usize,
}

struct Parser {
    t: Vec<Tok>,
    i: usize,
    next_param: usize,
    max_param: usize,
}

const RESERVED: &[&str] = &[
    "select", "from", "where", "group", "order", "by", "limit", "offset", "join", "inner", "on", "and", "as",
    "asc", "desc", "between", "left", "cross", "having", "union", "or", "not",
];

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }

    fn kw(&self, k: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case(k))
    }

    fn eat_kw(&mut self, k: &str) -> bool {
        if self.kw(k) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn expect_kw(&mut self, k: &str) -> Result<()> {
        if self.eat_kw(k) { Ok(()) } else { Err(self.err(&format!("expected {}", k.to_uppercase()))) }
    }

    fn sym(&self, s: &str) -> bool {
        matches!(self.peek(), Some(Tok::Sym(x)) if *x == s)
    }

    fn eat_sym(&mut self, s: &str) -> bool {
        if self.sym(s) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn expect_sym(&mut self, s: &str) -> Result<()> {
        if self.eat_sym(s) { Ok(()) } else { Err(self.err(&format!("expected '{s}'"))) }
    }

    fn err(&self, m: &str) -> Error {
        Error::Parse(format!("{m} near {:?}", self.peek()))
    }

    fn ident(&mut self) -> Result<String> {
        match self.peek() {
            Some(Tok::Ident(s)) if !RESERVED.iter().any(|r| r.eq_ignore_ascii_case(s)) => {
                let s = s.clone();
                self.i += 1;
                Ok(s)
            }
            _ => Err(self.err("expected an identifier")),
        }
    }

    fn colref(&mut self) -> Result<ColRef> {
        let a = self.ident()?;
        if self.eat_sym(".") {
            let b = self.ident()?;
            Ok(ColRef { table: Some(a), col: b })
        } else {
            Ok(ColRef { table: None, col: a })
        }
    }

    fn expr(&mut self) -> Result<Expr> {
        if let Some(Tok::Ident(name)) = self.peek()
            && matches!(self.t.get(self.i + 1), Some(Tok::Sym("(")))
        {
            let f = match name.to_ascii_lowercase().as_str() {
                "count" => AggFunc::Count,
                "sum" => AggFunc::Sum,
                "min" => AggFunc::Min,
                "max" => AggFunc::Max,
                "avg" => AggFunc::Avg,
                "total" => AggFunc::Total,
                other => return Err(Error::Unsupported(format!("function {other}()"))),
            };
            self.i += 2;
            if self.kw("distinct") {
                return Err(Error::Unsupported("DISTINCT aggregates".into()));
            }
            let arg = if f == AggFunc::Count && self.eat_sym("*") { None } else { Some(self.colref()?) };
            self.expect_sym(")")?;
            return Ok(Expr::Agg(f, arg));
        }
        Ok(Expr::Col(self.colref()?))
    }

    fn param(&mut self, n: Option<usize>) -> Operand {
        let idx = match n {
            Some(n) => n - 1,
            None => {
                self.next_param += 1;
                self.next_param - 1
            }
        };
        self.next_param = self.next_param.max(idx + 1);
        self.max_param = self.max_param.max(idx + 1);
        Operand::Param(idx)
    }

    /// An operand that is not a column, or `None` if the next token starts a column.
    fn value_operand(&mut self) -> Result<Option<Operand>> {
        let neg = if self.sym("-") || self.sym("+") {
            let n = self.sym("-");
            self.i += 1;
            Some(n)
        } else {
            None
        };
        match self.peek().cloned() {
            Some(Tok::Int(v)) => {
                self.i += 1;
                Ok(Some(Operand::Int(if neg == Some(true) { -v } else { v })))
            }
            Some(Tok::Param(n)) if neg.is_none() => {
                self.i += 1;
                Ok(Some(self.param(n)))
            }
            Some(Tok::Str(_)) => Err(Error::Unsupported("TEXT comparisons".into())),
            _ if neg.is_some() => Err(self.err("expected a number")),
            _ => Ok(None),
        }
    }

    fn cmp_op(&mut self) -> Result<CmpOp> {
        let op = match self.peek() {
            Some(Tok::Sym("=")) | Some(Tok::Sym("==")) => CmpOp::Eq,
            Some(Tok::Sym("<>")) | Some(Tok::Sym("!=")) => CmpOp::Ne,
            Some(Tok::Sym("<")) => CmpOp::Lt,
            Some(Tok::Sym("<=")) => CmpOp::Le,
            Some(Tok::Sym(">")) => CmpOp::Gt,
            Some(Tok::Sym(">=")) => CmpOp::Ge,
            _ => return Err(self.err("expected a comparison operator")),
        };
        self.i += 1;
        Ok(op)
    }

    /// One conjunct; column-to-column equalities go to `joins`.
    fn cond(&mut self, conds: &mut Vec<Cond>, joins: &mut Vec<(ColRef, ColRef)>) -> Result<()> {
        if self.eat_sym("(") {
            self.conj(conds, joins)?;
            return self.expect_sym(")");
        }
        if let Some(lhs) = self.value_operand()? {
            let op = self.cmp_op()?;
            let col = self.colref()?;
            conds.push(Cond { col, op: op.flip(), rhs: lhs });
            return Ok(());
        }
        let col = self.colref()?;
        if self.eat_kw("between") {
            let lo = self.value_operand()?.ok_or_else(|| self.err("expected a value"))?;
            self.expect_kw("and")?;
            let hi = self.value_operand()?.ok_or_else(|| self.err("expected a value"))?;
            conds.push(Cond { col: col.clone(), op: CmpOp::Ge, rhs: lo });
            conds.push(Cond { col, op: CmpOp::Le, rhs: hi });
            return Ok(());
        }
        if self.kw("not") || self.kw("is") || self.kw("in") || self.kw("like") {
            return Err(Error::Unsupported(format!("{:?} predicates", self.peek())));
        }
        let op = self.cmp_op()?;
        if let Some(rhs) = self.value_operand()? {
            conds.push(Cond { col, op, rhs });
        } else {
            let other = self.colref()?;
            if op != CmpOp::Eq {
                return Err(Error::Unsupported("column-to-column comparisons other than =".into()));
            }
            joins.push((col, other));
        }
        Ok(())
    }

    fn conj(&mut self, conds: &mut Vec<Cond>, joins: &mut Vec<(ColRef, ColRef)>) -> Result<()> {
        self.cond(conds, joins)?;
        while self.eat_kw("and") {
            self.cond(conds, joins)?;
        }
        if self.kw("or") {
            return Err(Error::Unsupported("OR".into()));
        }
        Ok(())
    }

    fn table_ref(&mut self) -> Result<TableRef> {
        let name = self.ident()?;
        let alias = if self.eat_kw("as") {
            Some(self.ident()?)
        } else if let Some(Tok::Ident(s)) = self.peek()
            && !RESERVED.iter().any(|r| r.eq_ignore_ascii_case(s))
        {
            Some(self.ident()?)
        } else {
            None
        };
        Ok(TableRef { name, alias })
    }

    fn limit_operand(&mut self) -> Result<Operand> {
        match self.peek().cloned() {
            Some(Tok::Int(v)) => {
                self.i += 1;
                Ok(Operand::Int(v))
            }
            Some(Tok::Param(n)) => {
                self.i += 1;
                Ok(self.param(n))
            }
            _ => Err(self.err("expected a number")),
        }
    }

    fn select(&mut self) -> Result<Select> {
        self.expect_kw("select")?;
        if self.kw("distinct") {
            return Err(Error::Unsupported("SELECT DISTINCT".into()));
        }
        self.eat_kw("all");
        let mut items = Vec::new();
        loop {
            if self.eat_sym("*") {
                items.push(SelectItem::Star(None));
            } else if let (Some(Tok::Ident(t)), Some(Tok::Sym(".")), Some(Tok::Sym("*"))) =
                (self.peek().cloned(), self.t.get(self.i + 1), self.t.get(self.i + 2))
            {
                self.i += 3;
                items.push(SelectItem::Star(Some(t)));
            } else {
                let expr = self.expr()?;
                let alias = if self.eat_kw("as") {
                    Some(self.ident()?)
                } else if let Some(Tok::Ident(s)) = self.peek()
                    && !RESERVED.iter().any(|r| r.eq_ignore_ascii_case(s))
                {
                    Some(self.ident()?)
                } else {
                    None
                };
                items.push(SelectItem::Expr { expr, alias });
            }
            if !self.eat_sym(",") {
                break;
            }
        }
        self.expect_kw("from")?;
        let mut from = vec![self.table_ref()?];
        let mut conds = Vec::new();
        let mut joins = Vec::new();
        loop {
            if self.eat_sym(",") {
                from.push(self.table_ref()?);
            } else if self.kw("join") || self.kw("inner") {
                self.eat_kw("inner");
                self.expect_kw("join")?;
                from.push(self.table_ref()?);
                self.expect_kw("on")?;
                self.conj(&mut conds, &mut joins)?;
            } else if self.kw("left") || self.kw("cross") || self.kw("natural") {
                return Err(Error::Unsupported("outer, cross and natural joins".into()));
            } else {
                break;
            }
        }
        if self.eat_kw("where") {
            self.conj(&mut conds, &mut joins)?;
        }
        let mut group_by = None;
        if self.eat_kw("group") {
            self.expect_kw("by")?;
            group_by = Some(self.colref()?);
            if self.sym(",") {
                return Err(Error::Unsupported("GROUP BY on more than one column".into()));
            }
        }
        if self.kw("having") {
            return Err(Error::Unsupported("HAVING".into()));
        }
        let mut order_by = Vec::new();
        if self.eat_kw("order") {
            self.expect_kw("by")?;
            loop {
                let key = if let Some(Tok::Int(n)) = self.peek().cloned() {
                    self.i += 1;
                    if n < 1 {
                        return Err(Error::Semantic("ORDER BY term out of range".into()));
                    }
                    OrderKey::Position(n as usize)
                } else {
                    OrderKey::Expr(self.expr()?)
                };
                let desc = if self.eat_kw("desc") {
                    true
                } else {
                    self.eat_kw("asc");
                    false
                };
                order_by.push(OrderItem { key, desc });
                if !self.eat_sym(",") {
                    break;
                }
            }
        }
        let (mut limit, mut offset) = (None, None);
        if self.eat_kw("limit") {
            limit = Some(self.limit_operand()?);
            if self.eat_kw("offset") {
                offset = Some(self.limit_operand()?);
            } else if self.eat_sym(",") {
                // LIMIT offset, count
                offset = limit;
                limit = Some(self.limit_operand()?);
            }
        }
        self.eat_sym(";");
        if self.i != self.t.len() {
            return Err(self.err("unexpected trailing input"));
        }
        Ok(Select { items, from, joins, conds, group_by, order_by, limit, offset, nparams: self.max_param })
    }
}

pub fn parse(sql: &str) -> Result<Select> {
    let t = lex(sql)?;
    Parser { t, i: 0, next_param: 0, max_param: 0 }.select()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_benchmark_shapes() {
        for q in [
            "SELECT count(*) FROM t WHERE v > ?",
            "SELECT count(*) FROM t WHERE v > ? AND k <> ?",
            "SELECT sec FROM t WHERE id = ?",
            "SELECT count(*) FROM t WHERE sec = ?",
            "SELECT count(*) FROM t JOIN b ON t.bid = b.id WHERE t.sec = ?",
            "SELECT sum(v) FROM t WHERE v > ?",
            "SELECT k, count(*), sum(v) FROM t GROUP BY k ORDER BY k",
            "SELECT id, v FROM t ORDER BY v DESC, id DESC LIMIT 20",
            "SELECT count(*) FROM t",
            "SELECT min(v), max(v) FROM t",
            "SELECT count(*) FROM t WHERE v BETWEEN ? AND ?",
            "SELECT id, v FROM t ORDER BY v DESC, id DESC LIMIT 20 OFFSET 1000",
            "SELECT t.id, b.id FROM t JOIN b ON t.bid = b.id WHERE t.sec = ?",
        ] {
            parse(q).unwrap_or_else(|e| panic!("{q}: {e}"));
        }
        let s = parse("SELECT count(*) FROM t WHERE 5 < v AND k <> ?2").unwrap();
        assert_eq!(s.conds[0].op, CmpOp::Gt);
        assert_eq!(s.nparams, 2);
    }
}
