//! Differential test: every query shape the engine supports, run against C
//! SQLite and sinew on the same data, under every kernel variant, compared on
//! the type-tagged rendering of the full result.

use rusqlite::types::ValueRef;
use sinew::kernels::{self, Variant};
use sinew::storage::import::import_sqlite;
use sinew::{Conn, Db, Value};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

const N: i64 = 5000;

fn seed(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
    let c = rusqlite::Connection::open(path).unwrap();
    c.execute_batch(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, sec INTEGER, k INTEGER, v INTEGER, bid INTEGER,
                         payload TEXT, n INTEGER, big INTEGER, s16 INTEGER, r REAL);
         CREATE TABLE b (id INTEGER PRIMARY KEY, label TEXT, w INTEGER);
         CREATE TABLE c (x INTEGER, y INTEGER);
         CREATE INDEX idx_t_sec ON t(sec);
         CREATE INDEX idx_t_n ON t(n);
         CREATE INDEX idx_b_w ON b(w);
         CREATE INDEX idx_c_x ON c(x);",
    )
    .unwrap();
    let mut rng = Rng(7);
    let tx = c.unchecked_transaction().unwrap();
    {
        let mut st = tx.prepare("INSERT INTO t VALUES (?,?,?,?,?,?,?,?,?,?)").unwrap();
        for i in 1..=N {
            // v is sorted-ish in the first half (zone maps can prune), random after
            let v = if i < N / 2 { i * 37 + rng.range(0, 30) } else { rng.range(0, 999_999) };
            let n: Option<i64> = if rng.next() % 7 == 0 { None } else { Some(rng.range(-500, 500)) };
            let big = if rng.next() % 2 == 0 { rng.range(i64::MIN / 4, i64::MAX / 4) } else { rng.range(-3, 3) };
            st.execute(rusqlite::params![
                i,
                rng.range(0, N),
                rng.range(0, 9),
                v,
                1 + rng.range(0, 3500),
                format!("row-{i}-payload"),
                n,
                big,
                rng.range(1000, 40_000),
                (i as f64) * 0.5,
            ])
            .unwrap();
        }
        let mut st = tx.prepare("INSERT INTO b VALUES (?,?,?)").unwrap();
        for i in 1..=3000 {
            if i % 11 == 0 {
                continue; // sparse rowids: not dense, binary-searched
            }
            st.execute(rusqlite::params![i, format!("label-{i}"), rng.range(0, 50)]).unwrap();
        }
        let mut st = tx.prepare("INSERT INTO c VALUES (?,?)").unwrap();
        for _ in 0..1500 {
            st.execute(rusqlite::params![rng.range(0, 100), rng.range(-1000, 1000)]).unwrap();
        }
    }
    tx.commit().unwrap();
    c.execute_batch("DELETE FROM c WHERE x % 5 = 0;").unwrap();
}

fn sqlite_render(c: &rusqlite::Connection, sql: &str, args: &[i64]) -> Result<String, String> {
    let mut st = c.prepare(sql).map_err(|e| e.to_string())?;
    let ncols = st.column_count();
    let mut rows = st.query(rusqlite::params_from_iter(args.iter())).map_err(|e| e.to_string())?;
    let mut out = String::new();
    while let Some(r) = rows.next().map_err(|e| e.to_string())? {
        for i in 0..ncols {
            let v = match r.get_ref(i).unwrap() {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(x) => Value::Int(x),
                ValueRef::Real(f) => Value::Real(f),
                ValueRef::Text(t) => Value::Text(String::from_utf8_lossy(t).into()),
                ValueRef::Blob(_) => panic!("blob"),
            };
            v.render_into(&mut out);
        }
        out.push(';');
    }
    Ok(out)
}

type Gen = fn(&mut Rng) -> Vec<i64>;

fn queries() -> Vec<(&'static str, Gen)> {
    vec![
        // the benchmark's nine
        ("SELECT count(*) FROM t WHERE v > ?", |r| vec![r.range(-10, 1_000_010)]),
        ("SELECT count(*) FROM t WHERE v > ? AND k <> ?", |r| vec![r.range(0, 1_000_000), r.range(-1, 10)]),
        ("SELECT sec FROM t WHERE id = ?", |r| vec![r.range(-2, N + 2)]),
        ("SELECT count(*) FROM t WHERE sec = ?", |r| vec![r.range(-1, N + 1)]),
        ("SELECT count(*) FROM t JOIN b ON t.bid = b.id WHERE t.sec = ?", |r| vec![r.range(0, N)]),
        ("SELECT sum(v) FROM t WHERE v > ?", |r| vec![r.range(-10, 1_000_010)]),
        ("SELECT k, count(*), sum(v) FROM t GROUP BY k ORDER BY k", |_| vec![]),
        ("SELECT id, v FROM t ORDER BY v DESC, id DESC LIMIT 20", |_| vec![]),
        ("SELECT count(*) FROM t", |_| vec![]),
        // predicate forms and merging
        ("SELECT count(*) FROM t WHERE v >= ? AND v < ?", |r| { let a = r.range(0, 1_000_000); vec![a, a + r.range(-5, 300_000)] }),
        ("SELECT count(*), sum(v), min(v), max(v) FROM t WHERE v BETWEEN ? AND ?", |r| { let a = r.range(0, 1_000_000); vec![a, a + r.range(0, 300_000)] }),
        ("SELECT count(*) FROM t WHERE k = ?", |r| vec![r.range(-1, 10)]),
        ("SELECT count(*) FROM t WHERE k <> ? AND k <> ?", |r| vec![r.range(0, 9), r.range(0, 9)]),
        ("SELECT count(*) FROM t WHERE ? < v", |r| vec![r.range(0, 1_000_000)]),
        ("SELECT count(*) FROM t WHERE v <= ?", |r| vec![r.range(-5, 1_000_000)]),
        ("SELECT count(*) FROM t WHERE s16 > ? AND v < ? AND k >= ?", |r| vec![r.range(900, 41_000), r.range(0, 1_000_000), r.range(0, 9)]),
        ("SELECT sum(s16), count(s16) FROM t WHERE s16 <= ?", |r| vec![r.range(900, 41_000)]),
        ("SELECT sum(k) FROM t WHERE k > ?", |r| vec![r.range(-1, 10)]),
        ("SELECT sum(v), avg(v), total(v) FROM t WHERE k = ?", |r| vec![r.range(0, 9)]),
        ("SELECT sum(v) FROM t WHERE v > ? AND v > ?", |r| vec![r.range(0, 1_000_000), r.range(0, 1_000_000)]),
        // NULLs, negatives, wide ranges
        ("SELECT count(*), count(n), sum(n), min(n), max(n) FROM t WHERE n > ?", |r| vec![r.range(-600, 600)]),
        ("SELECT count(*) FROM t WHERE n <> ?", |r| vec![r.range(-600, 600)]),
        ("SELECT count(n), sum(n), avg(n) FROM t", |_| vec![]),
        ("SELECT min(n), max(n), min(v), max(v), min(id), max(id) FROM t", |_| vec![]),
        ("SELECT count(*), min(big), max(big) FROM t WHERE big < ?", |r| vec![r.range(i64::MIN / 4, i64::MAX / 4)]),
        ("SELECT sum(big) FROM t WHERE big BETWEEN ? AND ?", |r| { let a = r.range(-3, 3); vec![a, a + r.range(0, 4)] }),
        ("SELECT count(*) FROM t WHERE n = ?", |r| vec![r.range(-500, 500)]),
        ("SELECT id, n FROM t WHERE n = ? ORDER BY id", |r| vec![r.range(-500, 500)]),
        // rowid ranges (dense rowid -> position range)
        ("SELECT count(*), sum(v) FROM t WHERE id > ? AND id <= ?", |r| { let a = r.range(-5, N); vec![a, a + r.range(0, 3000)] }),
        ("SELECT count(*) FROM t WHERE id <> ? AND v > ?", |r| vec![r.range(0, N), r.range(0, 1_000_000)]),
        ("SELECT sum(id), min(id), max(id) FROM t WHERE id >= ?", |r| vec![r.range(0, N)]),
        ("SELECT id, v FROM t WHERE id > ? ORDER BY id LIMIT 5", |r| vec![r.range(0, N)]),
        ("SELECT id FROM t WHERE id < ? ORDER BY id DESC LIMIT 7", |r| vec![r.range(0, N)]),
        // GROUP BY
        ("SELECT k, count(*), sum(v), min(v), max(v), avg(v) FROM t GROUP BY k", |_| vec![]),
        ("SELECT k, count(*) FROM t WHERE v > ? GROUP BY k ORDER BY k DESC", |r| vec![r.range(0, 1_000_000)]),
        ("SELECT k, sum(s16) FROM t WHERE id < ? GROUP BY k ORDER BY sum(s16) DESC, k", |r| vec![r.range(0, N)]),
        ("SELECT count(*), k FROM t GROUP BY k ORDER BY 1, 2 LIMIT 3 OFFSET 2", |_| vec![]),
        ("SELECT n, count(*) FROM t WHERE n < ? GROUP BY n ORDER BY n", |r| vec![r.range(-500, 500)]),
        ("SELECT s16, count(*) FROM t WHERE s16 > ? GROUP BY s16 ORDER BY count(*) DESC, s16 LIMIT 5", |r| vec![r.range(30_000, 40_000)]),
        ("SELECT w, count(*), max(id) FROM b GROUP BY w ORDER BY w", |_| vec![]),
        // top-N
        ("SELECT id, v FROM t ORDER BY v, id LIMIT 13", |_| vec![]),
        ("SELECT id, v FROM t WHERE k = ? ORDER BY v DESC, id DESC LIMIT 10", |r| vec![r.range(0, 9)]),
        ("SELECT id, v FROM t ORDER BY v DESC, id DESC LIMIT 20 OFFSET 1000", |_| vec![]),
        ("SELECT k, id FROM t ORDER BY k DESC, id LIMIT ?", |r| vec![r.range(0, 40)]),
        ("SELECT id, s16 FROM t WHERE v < ? ORDER BY s16, id DESC LIMIT 15", |r| vec![r.range(0, 1_000_000)]),
        ("SELECT id, big FROM t ORDER BY big DESC, id LIMIT 5", |_| vec![]),
        ("SELECT id, n FROM t ORDER BY n DESC, id LIMIT 9", |_| vec![]),
        // plain projections
        ("SELECT * FROM t WHERE id = ?", |r| vec![r.range(0, N)]),
        ("SELECT payload, r FROM t WHERE sec = ? ORDER BY id", |r| vec![r.range(0, N)]),
        ("SELECT id FROM t WHERE v > ? LIMIT 4", |r| vec![r.range(0, 1_000_000)]),
        ("SELECT id, k FROM t WHERE k = ? AND v < ? LIMIT 7 OFFSET 3", |r| vec![r.range(0, 9), r.range(0, 1_000_000)]),
        // joins: rowid probe (sparse rowids on b), index probe, hash
        ("SELECT t.id, b.id FROM t JOIN b ON t.bid = b.id WHERE t.sec = ?", |r| vec![r.range(0, N)]),
        ("SELECT count(*), sum(b.w) FROM t JOIN b ON t.bid = b.id WHERE t.k = ? AND b.w < ?", |r| vec![r.range(0, 9), r.range(0, 50)]),
        ("SELECT count(*) FROM b JOIN t ON b.id = t.bid WHERE b.w = ?", |r| vec![r.range(0, 50)]),
        ("SELECT b.label, t.id FROM t, b WHERE t.bid = b.id AND t.id = ?", |r| vec![r.range(0, N)]),
        ("SELECT count(*) FROM t JOIN c ON t.k = c.x WHERE t.id < ?", |r| vec![r.range(0, 200)]),
        ("SELECT c.y, t.id FROM c JOIN t ON c.y = t.n WHERE c.x = ? ORDER BY t.id, c.y", |r| vec![r.range(0, 100)]),
        ("SELECT b.w, count(*) FROM t JOIN b ON t.bid = b.id WHERE t.v > ? GROUP BY b.w ORDER BY b.w", |r| vec![r.range(0, 1_000_000)]),
        ("SELECT t.id, b.label FROM t JOIN b ON t.bid = b.id ORDER BY t.id DESC LIMIT 3", |_| vec![]),
        // hidden rowid, non-dense rowid lookups
        ("SELECT x, y FROM c WHERE rowid = ?", |r| vec![r.range(0, 1600)]),
        ("SELECT count(*), sum(y) FROM c WHERE x = ?", |r| vec![r.range(0, 100)]),
        ("SELECT label FROM b WHERE id = ?", |r| vec![r.range(0, 3001)]),
        ("SELECT count(*) FROM b WHERE id > ? AND id < ?", |r| { let a = r.range(0, 3000); vec![a, a + r.range(0, 500)] }),
        ("SELECT rowid, x FROM c ORDER BY rowid DESC LIMIT 3", |_| vec![]),
        // edge cases
        ("SELECT count(*) FROM t WHERE v > ? AND v < ?", |r| { let a = r.range(0, 1_000_000); vec![a, a] }),
        ("SELECT sum(v), min(v) FROM t WHERE v > 2000000", |_| vec![]),
        ("SELECT count(*) FROM t WHERE id = -1", |_| vec![]),
        ("SELECT k, count(*) FROM t WHERE v < 0 GROUP BY k", |_| vec![]),
        ("SELECT id FROM t ORDER BY v LIMIT 0", |_| vec![]),
        ("SELECT count(*) FROM t LIMIT 1 OFFSET 1", |_| vec![]),
    ]
}

#[test]
fn matches_sqlite() {
    let dir = std::env::temp_dir().join(format!("sinew-diff-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("t.db");
    let dst = dir.join("t.snw");
    seed(&src);
    import_sqlite(&src, &dst).unwrap();
    let db = Db::open(&dst).unwrap();
    let sq = rusqlite::Connection::open(&src).unwrap();

    let mut failures = Vec::new();
    let mut checked = 0;
    for variant in kernels::available() {
        kernels::set_variant(variant).unwrap();
        let mut conn = Conn::new(&db);
        let mut rng = Rng(99);
        for (q, argf) in queries() {
            for _ in 0..25 {
                let args = argf(&mut rng);
                let want = sqlite_render(&sq, q, &args);
                let vals: Vec<Value> = args.iter().map(|&a| Value::Int(a)).collect();
                let got = conn.query(q, &vals).map(|r| r.render()).map_err(|e| e.to_string());
                checked += 1;
                let same = match (&want, &got) {
                    (Ok(a), Ok(b)) => a == b,
                    (Err(_), Err(_)) => true,
                    _ => false,
                };
                if !same {
                    failures.push(format!(
                        "[{}] {q} {args:?}\n  sqlite: {}\n  sinew:  {}",
                        variant.name(),
                        trunc(&want),
                        trunc(&got)
                    ));
                }
            }
        }
    }
    kernels::set_variant(Variant::Portable).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{} of {checked} mismatches:\n{}", failures.len(), failures[..failures.len().min(15)].join("\n"));
}

fn trunc(r: &Result<String, String>) -> String {
    let s = match r {
        Ok(s) => s.clone(),
        Err(e) => format!("ERROR {e}"),
    };
    if s.len() > 300 { format!("{}...", &s[..300]) } else { s }
}
