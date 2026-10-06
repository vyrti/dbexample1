use std::cmp::Ordering;
use std::fmt::Write;

/// A SQL value, with SQLite's storage classes (BLOB is not stored).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Real(f64),
    Text(Box<str>),
}

impl Value {
    /// SQLite's cross-class ordering: NULL < INTEGER/REAL < TEXT.
    pub fn sql_cmp(&self, other: &Value) -> Ordering {
        use Value::*;
        match (self, other) {
            (Null, Null) => Ordering::Equal,
            (Null, _) => Ordering::Less,
            (_, Null) => Ordering::Greater,
            (Int(a), Int(b)) => a.cmp(b),
            (Int(a), Real(b)) => (*a as f64).total_cmp(b),
            (Real(a), Int(b)) => a.total_cmp(&(*b as f64)),
            (Real(a), Real(b)) => a.total_cmp(b),
            (Text(a), Text(b)) => a.as_bytes().cmp(b.as_bytes()),
            (Text(_), _) => Ordering::Greater,
            (_, Text(_)) => Ordering::Less,
        }
    }

    /// Appends the type-tagged rendering musql's harness uses
    /// (`renderAnyCell`), so results from every engine compare byte for byte.
    pub fn render_into(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("N:|"),
            Value::Int(i) => {
                let _ = write!(out, "i:{i}|");
            }
            Value::Real(f) => {
                let _ = write!(out, "f:{f}|");
            }
            Value::Text(s) => {
                out.push_str("t:");
                out.push_str(s);
                out.push('|');
            }
        }
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}
