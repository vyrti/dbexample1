//! Benchmark: sinew vs C SQLite vs Turso (in-process) vs musql (via the Go
//! driver in ../musql-bench), on identical data and identical bind values.
//!
//!   sinew-bench prepare --dir data [--rows 100000]
//!   (cd musql-bench && go run . -dir ../data > ../data/musql.json)
//!   sinew-bench run --dir data [--musql data/musql.json]
//!   sinew-bench kernels

use mimalloc::MiMalloc;
use rusqlite::types::ValueRef;
use sinew::kernels::{self, EncPred, Lanes, Variant};
use sinew::{Conn, Db, Value};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

type R<T> = Result<T, Box<dyn std::error::Error>>;

// ------------------------------------------------------------------ data

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn intn(&mut self, n: i64) -> i64 {
        (self.next() % n as u64) as i64
    }
}

const SEED: u64 = 0x5eed;
const NARGS: usize = 256;
const NCHECK: usize = 40;

struct Workload {
    name: &'static str,
    sql: &'static str,
    argf: fn(&mut Rng, i64) -> Vec<i64>,
    /// one of musql's nine headline workloads
    headline: bool,
}

fn workloads() -> Vec<Workload> {
    vec![
        Workload { name: "Filtered count, one predicate", sql: "SELECT count(*) FROM t WHERE v > ?", argf: |r, _| vec![r.intn(1_000_000)], headline: true },
        Workload { name: "Filtered count, two predicates", sql: "SELECT count(*) FROM t WHERE v > ? AND k <> ?", argf: |r, _| vec![r.intn(1_000_000), r.intn(10)], headline: true },
        Workload { name: "Rowid lookup", sql: "SELECT sec FROM t WHERE id = ?", argf: |r, n| vec![1 + r.intn(n)], headline: true },
        Workload { name: "Secondary-index equality", sql: "SELECT count(*) FROM t WHERE sec = ?", argf: |r, n| vec![r.intn(n)], headline: true },
        Workload { name: "Indexed equi-join", sql: "SELECT count(*) FROM t JOIN b ON t.bid = b.id WHERE t.sec = ?", argf: |r, n| vec![r.intn(n)], headline: true },
        Workload { name: "Sum over a filter", sql: "SELECT sum(v) FROM t WHERE v > ?", argf: |r, _| vec![r.intn(1_000_000)], headline: true },
        Workload { name: "Grouped aggregate", sql: "SELECT k, count(*), sum(v) FROM t GROUP BY k ORDER BY k", argf: |_, _| vec![], headline: true },
        Workload { name: "ORDER BY v DESC LIMIT 20", sql: "SELECT id, v FROM t ORDER BY v DESC, id DESC LIMIT 20", argf: |_, _| vec![], headline: true },
        Workload { name: "Whole-table count", sql: "SELECT count(*) FROM t", argf: |_, _| vec![], headline: true },
        // Shapes musql's harness lists as not served by its JIT.
        Workload { name: "min/max whole table", sql: "SELECT min(v), max(v) FROM t", argf: |_, _| vec![], headline: false },
        Workload { name: "Grouped min/max", sql: "SELECT k, min(v), max(v) FROM t GROUP BY k ORDER BY k", argf: |_, _| vec![], headline: false },
        Workload { name: "BETWEEN range", sql: "SELECT count(*) FROM t WHERE v BETWEEN ? AND ?", argf: |r, _| { let n = r.intn(500_000); vec![n, n + 250_000] }, headline: false },
        Workload { name: "Paginate OFFSET", sql: "SELECT id, v FROM t ORDER BY v DESC, id DESC LIMIT 20 OFFSET 1000", argf: |_, _| vec![], headline: false },
        Workload { name: "Join projecting rows", sql: "SELECT t.id, b.id FROM t JOIN b ON t.bid = b.id WHERE t.sec = ?", argf: |r, n| vec![r.intn(n)], headline: false },
    ]
}

fn prepare(dir: &Path, n: i64) -> R<()> {
    std::fs::create_dir_all(dir)?;
    let db = dir.join("bench.db");
    for f in ["bench.db", "bench.db-journal", "turso.db", "turso.db-wal", "bench.snw", "musql.musq"] {
        let _ = std::fs::remove_file(dir.join(f));
    }
    let t = Instant::now();
    let c = rusqlite::Connection::open(&db)?;
    c.execute_batch(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, sec INTEGER, k INTEGER, v INTEGER, bid INTEGER, payload TEXT);
         CREATE TABLE b (id INTEGER PRIMARY KEY, label TEXT);
         CREATE INDEX idx_t_sec ON t(sec);",
    )?;
    let mut rng = Rng(SEED);
    let tx = c.unchecked_transaction()?;
    {
        let mut st = tx.prepare("INSERT INTO b(id, label) VALUES (?, ?)")?;
        for i in 1..=n {
            st.execute(rusqlite::params![i, format!("label-{i}")])?;
        }
        let mut st = tx.prepare("INSERT INTO t(id, sec, k, v, bid, payload) VALUES (?, ?, ?, ?, ?, ?)")?;
        for i in 1..=n {
            st.execute(rusqlite::params![
                i,
                rng.intn(n),
                rng.intn(10),
                rng.intn(1_000_000),
                1 + rng.intn(n),
                format!("row-{i}-payload")
            ])?;
        }
    }
    tx.commit()?;
    c.execute_batch("VACUUM; ANALYZE;")?;
    drop(c);
    eprintln!("sqlite file: {:?} ({:.1} MB) in {:?}", db, mb(&db), t.elapsed());

    std::fs::copy(&db, dir.join("turso.db"))?;

    let t = Instant::now();
    sinew::storage::import::import_sqlite(&db, dir.join("bench.snw"))?;
    eprintln!("sinew file: {:.1} MB, imported in {:?}", mb(&dir.join("bench.snw")), t.elapsed());

    // Bind values, shared with the Go side.
    let mut out = String::new();
    let mut rng = Rng(SEED ^ 0xa5a5);
    writeln!(out, "R\t{n}")?;
    for w in workloads() {
        writeln!(out, "W\t{}\t{}", w.name, w.sql)?;
        for _ in 0..NARGS {
            let a = (w.argf)(&mut rng, n);
            writeln!(out, "A\t{}", a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))?;
        }
    }
    std::fs::write(dir.join("workloads.tsv"), out)?;
    Ok(())
}

fn mb(p: &Path) -> f64 {
    std::fs::metadata(p).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0)
}

struct Loaded {
    name: String,
    sql: String,
    args: Vec<Vec<i64>>,
    headline: bool,
}

fn load_workloads(dir: &Path) -> R<(i64, Vec<Loaded>)> {
    let text = std::fs::read_to_string(dir.join("workloads.tsv"))?;
    let heads: Vec<&str> = workloads().iter().filter(|w| w.headline).map(|w| w.name).collect();
    let mut rows = 0;
    let mut out: Vec<Loaded> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        match f[0] {
            "R" => rows = f[1].parse()?,
            "W" => out.push(Loaded {
                name: f[1].into(),
                sql: f[2].into(),
                args: Vec::new(),
                headline: heads.contains(&f[1]),
            }),
            "A" => out.last_mut().unwrap().args.push(if f[1].is_empty() {
                Vec::new()
            } else {
                f[1].split(',').map(|x| x.parse().unwrap()).collect()
            }),
            _ => {}
        }
    }
    Ok((rows, out))
}

// ------------------------------------------------------------------ timing

#[derive(Clone, Copy, Debug, Default)]
struct Stats {
    median: Duration,
    min: Duration,
    iters: usize,
}

/// The same procedure as ../musql-bench: three warm-up calls, iterations sized
/// from one timed call so a repetition takes ~`target`, `reps` repetitions
/// cycling through the bind values, median of the per-call means.
fn time_it(nargs: usize, target: Duration, reps: usize, mut run: impl FnMut(usize) -> R<()>) -> R<Stats> {
    for i in 0..3 {
        run(i % nargs)?;
    }
    let t = Instant::now();
    run(3 % nargs)?;
    let one = t.elapsed();
    let n = if one >= target {
        3
    } else {
        ((target.as_nanos() / one.as_nanos().max(1)) as usize).clamp(40, 2_000_000)
    };
    let mut means = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        for i in 0..n {
            run(i % nargs)?;
        }
        means.push(t.elapsed() / n as u32);
    }
    means.sort();
    Ok(Stats { median: means[means.len() / 2], min: means[0], iters: n })
}

// ------------------------------------------------------------------ engines

fn sqlite_rows(st: &mut rusqlite::CachedStatement, args: &[i64]) -> R<Vec<Value>> {
    let ncols = st.column_count();
    let mut rows = st.query(rusqlite::params_from_iter(args.iter()))?;
    let mut cells = Vec::new();
    while let Some(r) = rows.next()? {
        for i in 0..ncols {
            cells.push(match r.get_ref(i)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(x) => Value::Int(x),
                ValueRef::Real(f) => Value::Real(f),
                ValueRef::Text(t) => Value::Text(String::from_utf8_lossy(t).into()),
                ValueRef::Blob(_) => Value::Null,
            });
        }
    }
    Ok(cells)
}

fn render_cells(cells: &[Value], ncols: usize) -> String {
    let mut s = String::new();
    for row in cells.chunks(ncols.max(1)) {
        for v in row {
            v.render_into(&mut s);
        }
        s.push(';');
    }
    s
}

fn turso_rows(conn: &turso::Connection, sql: &str, args: &[i64]) -> R<(Vec<Value>, usize)> {
    pollster::block_on(async {
        let mut st = conn.prepare_cached(sql).await?;
        let params: Vec<turso::Value> = args.iter().map(|&a| turso::Value::Integer(a)).collect();
        let mut rows = st.query(params).await?;
        let ncols = rows.column_count();
        let mut cells = Vec::new();
        while let Some(r) = rows.next().await? {
            for i in 0..ncols {
                cells.push(match r.get_value(i)? {
                    turso::Value::Null => Value::Null,
                    turso::Value::Integer(x) => Value::Int(x),
                    turso::Value::Real(f) => Value::Real(f),
                    turso::Value::Text(t) => Value::Text(t.into()),
                    turso::Value::Blob(_) => Value::Null,
                });
            }
        }
        Ok((cells, ncols))
    })
}

#[derive(Default)]
struct Row {
    name: String,
    headline: bool,
    cells: BTreeMap<&'static str, Result<Stats, String>>,
}

struct MusqlRow {
    name: String,
    direct_ns: Option<f64>,
    direct_min_ns: Option<f64>,
    driver_ns: Option<f64>,
    error: String,
    renders: Vec<String>,
}

/// Reads musql-bench's TSV: `L label`, then per workload `W name direct
/// direct_min driver error` followed by one `C render` line per check value.
fn read_musql(path: &Path) -> R<(String, Vec<MusqlRow>)> {
    let text = std::fs::read_to_string(path)?;
    let mut label = "musql".to_string();
    let mut out: Vec<MusqlRow> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let num = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).filter(|x| *x > 0.0);
        match f[0] {
            "L" => label = f[1].to_string(),
            "W" => out.push(MusqlRow {
                name: f[1].to_string(),
                direct_ns: num(2),
                direct_min_ns: num(3),
                driver_ns: num(4),
                error: f.get(5).unwrap_or(&"").to_string(),
                renders: Vec::new(),
            }),
            "C" => {
                if let Some(w) = out.last_mut() {
                    w.renders.push(f.get(1).unwrap_or(&"").to_string());
                }
            }
            _ => {}
        }
    }
    Ok((label, out))
}

fn fmt_d(d: Duration) -> String {
    let ns = d.as_nanos() as f64;
    if ns < 1_000.0 {
        format!("{ns:.0} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.1} µs", ns / 1e3)
    } else {
        format!("{:.2} ms", ns / 1e6)
    }
}

fn ratio(other: Option<Duration>, ours: Duration) -> String {
    match other {
        None => "n/a".into(),
        Some(o) => {
            let x = o.as_nanos() as f64 / ours.as_nanos().max(1) as f64;
            if x >= 1.0 {
                if x >= 100.0 { format!("**{x:.0}×**") } else { format!("**{x:.1}×**") }
            } else {
                format!("{:.1}× slower", 1.0 / x)
            }
        }
    }
}

fn run(dir: &Path, musql: Option<PathBuf>, target: Duration, reps: usize, only: Option<String>) -> R<()> {
    let (rows, wls) = load_workloads(dir)?;
    let wls: Vec<Loaded> = wls
        .into_iter()
        .filter(|w| only.as_deref().is_none_or(|o| w.name.to_lowercase().contains(&o.to_lowercase())))
        .collect();

    let snw = Db::open(dir.join("bench.snw"))?;
    let sq = rusqlite::Connection::open_with_flags(dir.join("bench.db"), rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    // Give C SQLite its best single-process configuration: every page cached,
    // the file mmap'd, and the shared lock held across statements instead of
    // re-taken (and the cache revalidated) on each one.
    let sqlite_pragmas = std::env::var("SQLITE_PRAGMAS").unwrap_or_else(|_| {
        "PRAGMA cache_size = -1048576; PRAGMA mmap_size = 1073741824; PRAGMA locking_mode = EXCLUSIVE;".into()
    });
    sq.execute_batch(&sqlite_pragmas)?;
    sq.set_prepared_statement_cache_capacity(64);
    let turso_db = pollster::block_on(turso::Builder::new_local(dir.join("turso.db").to_str().unwrap()).build())?;
    let tconn = turso_db.connect()?;
    let _ = pollster::block_on(tconn.execute("PRAGMA cache_size = -1048576", ()));

    let musql = match musql {
        Some(p) => Some(read_musql(&p)?),
        None => None,
    };
    let musql_label = musql.as_ref().map(|m| m.0.clone()).unwrap_or_default();

    eprintln!(
        "{} rows; kernels: {} (available: {}); target {:?} x {} reps",
        rows,
        kernels::variant().name(),
        kernels::available().iter().map(|v| v.name()).collect::<Vec<_>>().join(", "),
        target,
        reps
    );

    let mut table: Vec<Row> = Vec::new();
    let mut failures = Vec::new();
    for w in &wls {
        eprint!("{:<34}", w.name);
        let mut row = Row { name: w.name.clone(), headline: w.headline, ..Default::default() };
        let vals: Vec<Vec<Value>> = w.args.iter().map(|a| a.iter().map(|&x| Value::Int(x)).collect()).collect();

        // Oracle renders.
        let mut st = sq.prepare_cached(&w.sql)?;
        let ncols = st.column_count();
        let oracle: Vec<String> =
            (0..NCHECK).map(|i| sqlite_rows(&mut st, &w.args[i]).map(|c| render_cells(&c, ncols))).collect::<R<_>>()?;
        drop(st);

        // ---- correctness: every engine on NCHECK bind values
        let mut conn = Conn::new(&snw);
        for i in 0..NCHECK {
            let got = conn.query(&w.sql, &vals[i]).map(|r| r.render());
            match got {
                Ok(g) if g == oracle[i] => {}
                Ok(g) => failures.push(format!("sinew {} {:?}: {} != {}", w.name, w.args[i], g, oracle[i])),
                Err(e) => failures.push(format!("sinew {} {:?}: error {e}", w.name, w.args[i])),
            }
        }
        let mut turso_ok = true;
        for i in 0..NCHECK {
            match turso_rows(&tconn, &w.sql, &w.args[i]) {
                Ok((c, n)) if render_cells(&c, n) == oracle[i] => {}
                Ok((c, n)) => {
                    turso_ok = false;
                    failures.push(format!("turso {} {:?}: {} != {}", w.name, w.args[i], render_cells(&c, n), oracle[i]));
                }
                Err(e) => {
                    turso_ok = false;
                    failures.push(format!("turso {} {:?}: error {e}", w.name, w.args[i]));
                    break;
                }
            }
        }
        if let Some((_, m)) = &musql
            && let Some(mw) = m.iter().find(|x| x.name == w.name)
        {
            for (i, r) in mw.renders.iter().enumerate().take(NCHECK) {
                if *r != oracle[i] {
                    failures.push(format!("musql {} {:?}: {} != {}", w.name, w.args[i], r, oracle[i]));
                }
            }
        }

        // ---- timing
        let n = w.args.len();
        let s = time_it(n, target, reps, |i| {
            black_box(conn.query(&w.sql, &vals[i])?);
            Ok(())
        });
        row.cells.insert("sinew", s.map_err(|e| e.to_string()));
        let stmt = conn.prepare(&w.sql)?;
        let s = time_it(n, target, reps, |i| {
            black_box(stmt.query(&vals[i])?);
            Ok(())
        });
        row.cells.insert("sinew-prepared", s.map_err(|e| e.to_string()));
        let s = time_it(n, target, reps, |i| {
            let mut st = sq.prepare_cached(&w.sql)?;
            black_box(sqlite_rows(&mut st, &w.args[i])?);
            Ok(())
        });
        row.cells.insert("sqlite", s.map_err(|e| e.to_string()));
        let s = if turso_ok {
            time_it(n, target, reps, |i| {
                black_box(turso_rows(&tconn, &w.sql, &w.args[i])?);
                Ok(())
            })
            .map_err(|e| e.to_string())
        } else {
            Err("wrong result".into())
        };
        row.cells.insert("turso", s);
        if let Some((_, m)) = &musql {
            let st = |ns: f64| Duration::from_nanos(ns as u64);
            match m.iter().find(|x| x.name == w.name) {
                Some(mw) => {
                    match mw.direct_ns {
                        Some(d) => row.cells.insert(
                            "musql",
                            Ok(Stats { median: st(d), min: mw.direct_min_ns.map_or(st(d), st), iters: 0 }),
                        ),
                        None => row.cells.insert("musql", Err(mw.error.clone())),
                    };
                    if let Some(drv) = mw.driver_ns {
                        row.cells.insert("musql-driver", Ok(Stats { median: st(drv), min: st(drv), iters: 0 }));
                    }
                }
                None => {
                    row.cells.insert("musql", Err("not run".into()));
                }
            }
        }
        eprintln!(
            " sinew {:>9}  sqlite {:>9}  turso {:>9}",
            row.cells["sinew"].as_ref().map(|s| fmt_d(s.median)).unwrap_or_else(|e| e.clone()),
            row.cells["sqlite"].as_ref().map(|s| fmt_d(s.median)).unwrap_or_else(|e| e.clone()),
            row.cells["turso"].as_ref().map(|s| fmt_d(s.median)).unwrap_or_else(|e| e.clone()),
        );
        table.push(row);
    }

    // ---- report
    let cpu = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let mut md = String::new();
    writeln!(md, "{rows} rows, {cpu}, sinew kernels: `{}`. Median of {reps} repetitions of ~{target:?} each.\n", kernels::variant().name())?;
    let has_musql = musql.is_some();
    let cell = |r: &Row, k: &str| match r.cells.get(k) {
        Some(Ok(s)) => fmt_d(s.median),
        Some(Err(e)) => e.chars().take(20).collect(),
        None => "–".into(),
    };
    let dur = |r: &Row, k: &str| r.cells.get(k).and_then(|c| c.as_ref().ok()).map(|s| s.median);
    for (title, head) in [("musql's nine headline workloads", true), ("Shapes musql's JIT does not serve", false)] {
        let rs: Vec<&Row> = table.iter().filter(|r| r.headline == head).collect();
        if rs.is_empty() {
            continue;
        }
        writeln!(md, "### {title}\n")?;
        if has_musql {
            writeln!(md, "| Query | sinew | {musql_label} | C SQLite | Turso | vs musql | vs SQLite | vs Turso |")?;
            writeln!(md, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")?;
        } else {
            writeln!(md, "| Query | sinew | C SQLite | Turso | vs SQLite | vs Turso |")?;
            writeln!(md, "| --- | ---: | ---: | ---: | ---: | ---: |")?;
        }
        for r in rs {
            let Some(ours) = dur(r, "sinew") else { continue };
            if has_musql {
                writeln!(
                    md,
                    "| {} | {} | {} | {} | {} | {} | {} | {} |",
                    r.name,
                    cell(r, "sinew"),
                    cell(r, "musql"),
                    cell(r, "sqlite"),
                    cell(r, "turso"),
                    ratio(dur(r, "musql"), ours),
                    ratio(dur(r, "sqlite"), ours),
                    ratio(dur(r, "turso"), ours)
                )?;
            } else {
                writeln!(
                    md,
                    "| {} | {} | {} | {} | {} | {} |",
                    r.name,
                    cell(r, "sinew"),
                    cell(r, "sqlite"),
                    cell(r, "turso"),
                    ratio(dur(r, "sqlite"), ours),
                    ratio(dur(r, "turso"), ours)
                )?;
            }
        }
        writeln!(md)?;
    }
    writeln!(md, "Per-call API cost (sinew SQL-text path vs prepared statement; musql direct engine vs `database/sql`):\n")?;
    writeln!(md, "| Query | sinew SQL text | sinew prepared | musql direct | musql database/sql |")?;
    writeln!(md, "| --- | ---: | ---: | ---: | ---: |")?;
    for r in &table {
        writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            r.name,
            cell(r, "sinew"),
            cell(r, "sinew-prepared"),
            cell(r, "musql"),
            cell(r, "musql-driver")
        )?;
    }
    println!("{md}");
    std::fs::write(dir.join("results.md"), &md)?;
    if failures.is_empty() {
        eprintln!("correctness: every engine matched C SQLite on {NCHECK} bind values per workload");
    } else {
        eprintln!("correctness: {} mismatches", failures.len());
        for f in failures.iter().take(30) {
            eprintln!("  {f}");
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ kernels

fn kernel_bench(n: usize) -> R<()> {
    let mut rng = Rng(1);
    let d32: Vec<u32> = (0..n).map(|_| rng.intn(1_000_000) as u32).collect();
    let d8: Vec<u8> = (0..n).map(|_| rng.intn(10) as u8).collect();
    let mut mask = vec![0u8; n];
    let t = 500_000u64;
    println!("kernel microbenchmarks, {n} rows (u32 = 4 B/row, u8 = 1 B/row), ns per call and GB/s of column read\n");
    println!("| kernel | {} |", kernels::available().iter().map(|v| v.name()).collect::<Vec<_>>().join(" | "));
    println!("|---|{}", "---:|".repeat(kernels::available().len()));
    type K<'a> = (&'a str, usize, Box<dyn Fn() -> u64 + 'a>);
    let ks: Vec<K> = vec![
        ("count u32 >= t", 4, Box::new(|| kernels::count(Lanes::U32(&d32), EncPred::Ge(t)))),
        ("count u32 in [lo,hi]", 4, Box::new(|| kernels::count(Lanes::U32(&d32), EncPred::Range { lo: 250_000, span: 500_000 }))),
        ("count u8 <> c", 1, Box::new(|| kernels::count(Lanes::U8(&d8), EncPred::Ne(3)))),
        ("sum+count u32 >= t", 4, Box::new(|| kernels::sum_count(Lanes::U32(&d32), EncPred::Ge(t)).0)),
    ];
    for (name, bpr, f) in &ks {
        let mut line = format!("| {name} |");
        for v in kernels::available() {
            kernels::set_variant(v)?;
            let s = time_it(1, Duration::from_millis(150), 5, |_| {
                black_box(f());
                Ok(())
            })?;
            let gbs = (n * bpr) as f64 / s.median.as_nanos() as f64;
            write!(line, " {} ({gbs:.0} GB/s) |", fmt_d(s.median))?;
        }
        println!("{line}");
    }
    // mask-producing kernel
    let mut line = "| mask u32 >= t |".to_string();
    for v in kernels::available() {
        kernels::set_variant(v)?;
        let s = time_it(1, Duration::from_millis(150), 5, |_| {
            kernels::mask(Lanes::U32(&d32), EncPred::Ge(t), &mut mask, false);
            black_box(&mask);
            Ok(())
        })?;
        write!(line, " {} ({:.0} GB/s) |", fmt_d(s.median), (n * 4) as f64 / s.median.as_nanos() as f64)?;
    }
    println!("{line}");
    Ok(())
}

// ------------------------------------------------------------------ main

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let dir = PathBuf::from(flag("--dir").unwrap_or_else(|| "data".into()));
    if let Some(k) = flag("--kernel") {
        kernels::set_variant(Variant::parse(&k).ok_or("unknown kernel variant")?)?;
    }
    match args.get(1).map(String::as_str) {
        Some("prepare") => prepare(&dir, flag("--rows").map_or(Ok(100_000), |s| s.parse())?),
        Some("run") => run(
            &dir,
            flag("--musql").map(PathBuf::from),
            Duration::from_millis(flag("--target-ms").map_or(Ok(200), |s| s.parse())?),
            flag("--reps").map_or(Ok(5), |s| s.parse())?,
            flag("--only"),
        ),
        Some("kernels") => {
            for n in [100_000usize, 10_000_000] {
                kernel_bench(n)?;
                println!();
            }
            Ok(())
        }
        _ => {
            eprintln!("usage: sinew-bench prepare|run|kernels [--dir DIR] [--rows N] [--musql FILE] [--kernel V]");
            std::process::exit(2);
        }
    }
}
