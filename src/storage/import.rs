//! Importing a SQLite database file.

use super::build::{Builder, ColumnBuilder, ColumnValues, TableBuilder};
use crate::error::{Error, Result};
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};
use std::path::Path;

/// Converts every rowid table of the SQLite database `src` into a sinew file
/// at `dst`, with every single-column index on an integer column.
pub fn import_sqlite(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<()> {
    let conn = Connection::open_with_flags(src, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut b = Builder::default();
    let tables: Vec<(String, String)> = conn
        .prepare("SELECT name, sql FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<std::result::Result<_, _>>()?;
    for (name, sql) in tables {
        if sql.to_ascii_uppercase().contains("WITHOUT ROWID") {
            return Err(Error::Unsupported(format!("table {name} is WITHOUT ROWID")));
        }
        b.tables.push(import_table(&conn, &name)?);
    }
    b.write(dst)
}

fn quote(id: &str) -> String {
    format!("\"{}\"", id.replace('"', "\"\""))
}

fn import_table(conn: &Connection, name: &str) -> Result<TableBuilder> {
    // (name, declared type, pk ordinal)
    let info: Vec<(String, String, i64)> = conn
        .prepare(&format!("PRAGMA table_info({})", quote(name)))?
        .query_map([], |r| Ok((r.get(1)?, r.get(2)?, r.get(5)?)))?
        .collect::<std::result::Result<_, _>>()?;
    let pks: Vec<usize> = (0..info.len()).filter(|&i| info[i].2 > 0).collect();
    let ipk = match pks.as_slice() {
        [i] if info[*i].1.eq_ignore_ascii_case("INTEGER") => Some(*i),
        _ => None,
    };

    // Read rowid plus every column, in rowid order.
    let cols: Vec<String> = info.iter().map(|c| quote(&c.0)).collect();
    let sql = format!("SELECT rowid, {} FROM {} ORDER BY rowid", cols.join(", "), quote(name));
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let ncols = info.len();
    let mut rowids: Vec<Option<i64>> = Vec::new();
    // Per column: raw values, typed lazily below.
    let mut raw: Vec<Vec<Raw>> = vec![Vec::new(); ncols];
    while let Some(r) = rows.next()? {
        rowids.push(Some(r.get(0)?));
        for (c, out) in raw.iter_mut().enumerate() {
            out.push(match r.get_ref(c + 1)? {
                ValueRef::Null => Raw::Null,
                ValueRef::Integer(i) => Raw::Int(i),
                ValueRef::Real(f) => Raw::Real(f),
                ValueRef::Text(t) => Raw::Text(String::from_utf8_lossy(t).into_owned()),
                ValueRef::Blob(_) => {
                    return Err(Error::Unsupported(format!("BLOB values in {name}.{}", info[c].0)));
                }
            });
        }
    }

    let mut out_cols = Vec::with_capacity(ncols + 1);
    for (c, vals) in raw.into_iter().enumerate() {
        let values = type_column(vals).ok_or_else(|| {
            Error::Unsupported(format!("column {name}.{} mixes storage classes", info[c].0))
        })?;
        out_cols.push(ColumnBuilder { name: info[c].0.clone(), decl: info[c].1.clone(), hidden: false, values });
    }
    let rowid_col = match ipk {
        Some(i) => i,
        None => {
            out_cols.push(ColumnBuilder {
                name: "rowid".into(),
                decl: "INTEGER".into(),
                hidden: true,
                values: ColumnValues::Int(rowids),
            });
            out_cols.len() - 1
        }
    };

    // Single-column, non-partial indexes on integer columns.
    let idx: Vec<(String, Option<String>)> = conn
        .prepare("SELECT name, sql FROM sqlite_schema WHERE type = 'index' AND tbl_name = ?1")?
        .query_map([name], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<std::result::Result<_, _>>()?;
    let mut indexes = Vec::new();
    for (iname, isql) in idx {
        if isql.as_deref().is_some_and(|s| s.to_ascii_uppercase().contains(" WHERE ")) {
            continue;
        }
        let icols: Vec<Option<String>> = conn
            .prepare(&format!("PRAGMA index_info({})", quote(&iname)))?
            .query_map([], |r| r.get(2))?
            .collect::<std::result::Result<_, _>>()?;
        if let [Some(cname)] = icols.as_slice()
            && let Some(ci) = info.iter().position(|c| c.0.eq_ignore_ascii_case(cname))
            && matches!(out_cols[ci].values, ColumnValues::Int(_))
        {
            indexes.push((iname, ci));
        }
    }
    Ok(TableBuilder { name: name.to_string(), cols: out_cols, rowid_col, indexes })
}

#[derive(Clone)]
enum Raw {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
}

/// One storage class per column (plus NULL). Anything else is declined rather
/// than coerced, since coercion would change what queries return.
fn type_column(vals: Vec<Raw>) -> Option<ColumnValues> {
    let (mut ints, mut reals, mut texts) = (false, false, false);
    for v in &vals {
        match v {
            Raw::Int(_) => ints = true,
            Raw::Real(_) => reals = true,
            Raw::Text(_) => texts = true,
            Raw::Null => {}
        }
    }
    if ints as u8 + reals as u8 + texts as u8 > 1 {
        return None;
    }
    Some(if texts {
        ColumnValues::Text(vals.into_iter().map(|v| if let Raw::Text(s) = v { Some(s) } else { None }).collect())
    } else if reals {
        ColumnValues::Real(vals.into_iter().map(|v| if let Raw::Real(f) = v { Some(f) } else { None }).collect())
    } else {
        ColumnValues::Int(vals.into_iter().map(|v| if let Raw::Int(i) = v { Some(i) } else { None }).collect())
    })
}
