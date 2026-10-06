//! sinew: a columnar, SIMD-kernel SQL engine for a read-only SQLite subset.
//!
//! The design follows the reasoning in musql's `docs/format-design.md` and takes
//! it to its end: the row format sets the floor, so the format is columnar,
//! fixed-width and narrowed per column (frame-of-reference to u8/u16/u32/u64),
//! and every predicate is lowered into that encoded domain once per query, so
//! the scan kernels compare raw lanes with no decode at all.
//!
//! Layers:
//! - [`storage`]: the `.snw` file format (mmap'd), column encodings, zone maps,
//!   secondary indexes, and the SQLite importer.
//! - [`kernels`]: scan kernels in four variants -- portable Rust (left to LLVM's
//!   auto-vectorizer), NEON intrinsics, hand-written aarch64 assembly, and AVX2.
//! - [`sql`]: lexer and parser for the supported SELECT subset.
//! - [`plan`] / [`exec`]: planner and executor, with fast paths for filtered
//!   aggregates, small-domain GROUP BY, zone-ordered top-N, and index/rowid
//!   access, and a generic path for everything else in the grammar.

pub mod error;
pub mod exec;
pub mod kernels;
pub mod plan;
pub mod sql;
pub mod storage;
pub mod value;

pub use error::{Error, Result};
pub use exec::{Conn, QueryResult, Stmt};
pub use storage::Db;
pub use value::Value;
