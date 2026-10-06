//! Hand-written aarch64 assembly for the hottest loops: filtered count over
//! u32 and u8 lanes, and filtered sum+count over u32 lanes. Everything else
//! delegates to the NEON intrinsics.
//!
//! The loops are written for the Firestorm/Avalanche pipeline: four 128-bit
//! loads via two `ldp q` per 32 bytes, four SIMD ALUs kept busy by eight
//! independent compares per iteration, and mask pairs pre-added (`add`) so each
//! accumulator's dependency chain advances once per iteration.

use super::{EncPred, Lanes, neon};
use core::arch::aarch64::*;
use core::arch::asm;

// One compare per register: turns lane values in vR into an all-ones mask.
// `{t}` is the splatted bound; Range also uses `{s}` (the span).
macro_rules! ge32 { ($r:literal) => { concat!("cmhs v", $r, ".4s, v", $r, ".4s, {t:v}.4s\n") }; }
macro_rules! le32 { ($r:literal) => { concat!("cmhs v", $r, ".4s, {t:v}.4s, v", $r, ".4s\n") }; }
macro_rules! eq32 { ($r:literal) => { concat!("cmeq v", $r, ".4s, v", $r, ".4s, {t:v}.4s\n") }; }
macro_rules! rg32 { ($r:literal) => { concat!("sub v", $r, ".4s, v", $r, ".4s, {t:v}.4s\n",
                                              "cmhs v", $r, ".4s, {s:v}.4s, v", $r, ".4s\n") }; }
macro_rules! ge8 { ($r:literal) => { concat!("cmhs v", $r, ".16b, v", $r, ".16b, {t:v}.16b\n") }; }
macro_rules! le8 { ($r:literal) => { concat!("cmhs v", $r, ".16b, {t:v}.16b, v", $r, ".16b\n") }; }
macro_rules! eq8 { ($r:literal) => { concat!("cmeq v", $r, ".16b, v", $r, ".16b, {t:v}.16b\n") }; }
macro_rules! rg8 { ($r:literal) => { concat!("sub v", $r, ".16b, v", $r, ".16b, {t:v}.16b\n",
                                             "cmhs v", $r, ".16b, {s:v}.16b, v", $r, ".16b\n") }; }

// Sum kernels keep the values in v0-v3 and build masks in v4-v7.
macro_rules! mge32 { ($d:literal, $r:literal) => { concat!("cmhs v", $d, ".4s, v", $r, ".4s, {t:v}.4s\n") }; }
macro_rules! mle32 { ($d:literal, $r:literal) => { concat!("cmhs v", $d, ".4s, {t:v}.4s, v", $r, ".4s\n") }; }
macro_rules! meq32 { ($d:literal, $r:literal) => { concat!("cmeq v", $d, ".4s, v", $r, ".4s, {t:v}.4s\n") }; }
macro_rules! mrg32 { ($d:literal, $r:literal) => { concat!("sub v", $d, ".4s, v", $r, ".4s, {t:v}.4s\n",
                                                           "cmhs v", $d, ".4s, {s:v}.4s, v", $d, ".4s\n") }; }

/// Count over u32 lanes, 32 per iteration. Returns per-lane counts in 4 accs.
macro_rules! def_count_u32 {
    ($name:ident, $cmp:ident $(, $s:ident)?) => {
        #[inline(always)]
        unsafe fn $name(ptr: *const u32, iters: usize, t: uint32x4_t $(, $s: uint32x4_t)?) -> u64 {
            debug_assert!(iters > 0);
            unsafe {
                let (mut a0, mut a1, mut a2, mut a3) =
                    (vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0));
                asm!(
                    "2:",
                    "ldp q0, q1, [{p}]",
                    "ldp q2, q3, [{p}, #32]",
                    "ldp q4, q5, [{p}, #64]",
                    "ldp q6, q7, [{p}, #96]",
                    "add {p}, {p}, #128",
                    $cmp!("0"), $cmp!("1"), $cmp!("2"), $cmp!("3"),
                    $cmp!("4"), $cmp!("5"), $cmp!("6"), $cmp!("7"),
                    "add v0.4s, v0.4s, v4.4s",
                    "add v1.4s, v1.4s, v5.4s",
                    "add v2.4s, v2.4s, v6.4s",
                    "add v3.4s, v3.4s, v7.4s",
                    "sub {a0:v}.4s, {a0:v}.4s, v0.4s",
                    "sub {a1:v}.4s, {a1:v}.4s, v1.4s",
                    "sub {a2:v}.4s, {a2:v}.4s, v2.4s",
                    "sub {a3:v}.4s, {a3:v}.4s, v3.4s",
                    "subs {n}, {n}, #1",
                    "b.ne 2b",
                    p = inout(reg) ptr => _,
                    n = inout(reg) iters => _,
                    t = in(vreg) t,
                    $($s = in(vreg) $s,)?
                    a0 = inout(vreg) a0,
                    a1 = inout(vreg) a1,
                    a2 = inout(vreg) a2,
                    a3 = inout(vreg) a3,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                    options(nostack, readonly),
                );
                vaddlvq_u32(a0) + vaddlvq_u32(a1) + vaddlvq_u32(a2) + vaddlvq_u32(a3)
            }
        }
    };
}

def_count_u32!(count_u32_ge, ge32);
def_count_u32!(count_u32_le, le32);
def_count_u32!(count_u32_eq, eq32);
def_count_u32!(count_u32_rg, rg32, s);

/// Count over u8 lanes, 128 per iteration. Each accumulator lane gains at most
/// 2 per iteration, so callers pass at most 127 iterations.
macro_rules! def_count_u8 {
    ($name:ident, $cmp:ident $(, $s:ident)?) => {
        #[inline(always)]
        unsafe fn $name(ptr: *const u8, iters: usize, t: uint8x16_t $(, $s: uint8x16_t)?) -> u64 {
            debug_assert!(iters > 0 && iters <= 127);
            unsafe {
                let (mut a0, mut a1, mut a2, mut a3) =
                    (vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0));
                asm!(
                    "2:",
                    "ldp q0, q1, [{p}]",
                    "ldp q2, q3, [{p}, #32]",
                    "ldp q4, q5, [{p}, #64]",
                    "ldp q6, q7, [{p}, #96]",
                    "add {p}, {p}, #128",
                    $cmp!("0"), $cmp!("1"), $cmp!("2"), $cmp!("3"),
                    $cmp!("4"), $cmp!("5"), $cmp!("6"), $cmp!("7"),
                    "add v0.16b, v0.16b, v4.16b",
                    "add v1.16b, v1.16b, v5.16b",
                    "add v2.16b, v2.16b, v6.16b",
                    "add v3.16b, v3.16b, v7.16b",
                    "sub {a0:v}.16b, {a0:v}.16b, v0.16b",
                    "sub {a1:v}.16b, {a1:v}.16b, v1.16b",
                    "sub {a2:v}.16b, {a2:v}.16b, v2.16b",
                    "sub {a3:v}.16b, {a3:v}.16b, v3.16b",
                    "subs {n}, {n}, #1",
                    "b.ne 2b",
                    p = inout(reg) ptr => _,
                    n = inout(reg) iters => _,
                    t = in(vreg) t,
                    $($s = in(vreg) $s,)?
                    a0 = inout(vreg) a0,
                    a1 = inout(vreg) a1,
                    a2 = inout(vreg) a2,
                    a3 = inout(vreg) a3,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                    options(nostack, readonly),
                );
                vaddlvq_u8(a0) as u64 + vaddlvq_u8(a1) as u64 + vaddlvq_u8(a2) as u64 + vaddlvq_u8(a3) as u64
            }
        }
    };
}

def_count_u8!(count_u8_ge, ge8);
def_count_u8!(count_u8_le, le8);
def_count_u8!(count_u8_eq, eq8);
def_count_u8!(count_u8_rg, rg8, s);

/// Filtered (count, sum) over u32 lanes, 16 per iteration: mask, `and`, then
/// `uadalp` widens and accumulates the selected values into u64 lanes.
macro_rules! def_sum_u32 {
    ($name:ident, $cmp:ident $(, $s:ident)?) => {
        #[inline(always)]
        unsafe fn $name(ptr: *const u32, iters: usize, t: uint32x4_t $(, $s: uint32x4_t)?) -> (u64, u128) {
            debug_assert!(iters > 0);
            unsafe {
                let (mut c0, mut c1, mut c2, mut c3) =
                    (vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0));
                let (mut s0, mut s1, mut s2, mut s3) =
                    (vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0));
                asm!(
                    "2:",
                    "ldp q0, q1, [{p}]",
                    "ldp q2, q3, [{p}, #32]",
                    "add {p}, {p}, #64",
                    $cmp!("4", "0"), $cmp!("5", "1"), $cmp!("6", "2"), $cmp!("7", "3"),
                    "and v0.16b, v0.16b, v4.16b",
                    "and v1.16b, v1.16b, v5.16b",
                    "and v2.16b, v2.16b, v6.16b",
                    "and v3.16b, v3.16b, v7.16b",
                    "sub {c0:v}.4s, {c0:v}.4s, v4.4s",
                    "sub {c1:v}.4s, {c1:v}.4s, v5.4s",
                    "sub {c2:v}.4s, {c2:v}.4s, v6.4s",
                    "sub {c3:v}.4s, {c3:v}.4s, v7.4s",
                    "uadalp {s0:v}.2d, v0.4s",
                    "uadalp {s1:v}.2d, v1.4s",
                    "uadalp {s2:v}.2d, v2.4s",
                    "uadalp {s3:v}.2d, v3.4s",
                    "subs {n}, {n}, #1",
                    "b.ne 2b",
                    p = inout(reg) ptr => _,
                    n = inout(reg) iters => _,
                    t = in(vreg) t,
                    $($s = in(vreg) $s,)?
                    c0 = inout(vreg) c0,
                    c1 = inout(vreg) c1,
                    c2 = inout(vreg) c2,
                    c3 = inout(vreg) c3,
                    s0 = inout(vreg) s0,
                    s1 = inout(vreg) s1,
                    s2 = inout(vreg) s2,
                    s3 = inout(vreg) s3,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                    options(nostack, readonly),
                );
                let cnt = vaddlvq_u32(c0) + vaddlvq_u32(c1) + vaddlvq_u32(c2) + vaddlvq_u32(c3);
                let sum = vaddvq_u64(s0) as u128 + vaddvq_u64(s1) as u128
                    + vaddvq_u64(s2) as u128 + vaddvq_u64(s3) as u128;
                (cnt, sum)
            }
        }
    };
}

def_sum_u32!(sum_u32_ge, mge32);
def_sum_u32!(sum_u32_le, mle32);
def_sum_u32!(sum_u32_eq, meq32);
def_sum_u32!(sum_u32_rg, mrg32, s);

fn count_u32(s: &[u32], p: EncPred) -> u64 {
    let iters = s.len() / 32;
    let main = iters * 32;
    let mut total = 0;
    if iters > 0 {
        let ptr = s.as_ptr();
        total = unsafe {
            match p {
                EncPred::Ge(t) => count_u32_ge(ptr, iters, vdupq_n_u32(t as u32)),
                EncPred::Le(t) => count_u32_le(ptr, iters, vdupq_n_u32(t as u32)),
                EncPred::Eq(c) => count_u32_eq(ptr, iters, vdupq_n_u32(c as u32)),
                EncPred::Ne(c) => main as u64 - count_u32_eq(ptr, iters, vdupq_n_u32(c as u32)),
                EncPred::Range { lo, span } => {
                    count_u32_rg(ptr, iters, vdupq_n_u32(lo as u32), vdupq_n_u32(span as u32))
                }
            }
        };
    }
    total + s[main..].iter().filter(|&&x| p.test(x as u64)).count() as u64
}

fn count_u8(s: &[u8], p: EncPred) -> u64 {
    let iters = s.len() / 128;
    let main = iters * 128;
    let mut total = 0u64;
    let mut done = 0;
    let ptr = s.as_ptr();
    while done < iters {
        let k = (iters - done).min(127);
        let at = unsafe { ptr.add(done * 128) };
        total += unsafe {
            match p {
                EncPred::Ge(t) => count_u8_ge(at, k, vdupq_n_u8(t as u8)),
                EncPred::Le(t) => count_u8_le(at, k, vdupq_n_u8(t as u8)),
                EncPred::Eq(c) => count_u8_eq(at, k, vdupq_n_u8(c as u8)),
                EncPred::Ne(c) => (k * 128) as u64 - count_u8_eq(at, k, vdupq_n_u8(c as u8)),
                EncPred::Range { lo, span } => {
                    count_u8_rg(at, k, vdupq_n_u8(lo as u8), vdupq_n_u8(span as u8))
                }
            }
        };
        done += k;
    }
    total + s[main..].iter().filter(|&&x| p.test(x as u64)).count() as u64
}

fn sum_count_u32(s: &[u32], p: EncPred) -> (u64, u128) {
    let iters = s.len() / 16;
    let main = iters * 16;
    let (mut cnt, mut sum) = (0u64, 0u128);
    if iters > 0 {
        let ptr = s.as_ptr();
        (cnt, sum) = unsafe {
            match p {
                EncPred::Ge(t) => sum_u32_ge(ptr, iters, vdupq_n_u32(t as u32)),
                EncPred::Le(t) => sum_u32_le(ptr, iters, vdupq_n_u32(t as u32)),
                EncPred::Eq(c) => sum_u32_eq(ptr, iters, vdupq_n_u32(c as u32)),
                EncPred::Range { lo, span } => {
                    sum_u32_rg(ptr, iters, vdupq_n_u32(lo as u32), vdupq_n_u32(span as u32))
                }
                // Ne has no single-compare form here; the intrinsics handle it.
                EncPred::Ne(_) => return neon::sum_count(Lanes::U32(s), p),
            }
        };
    }
    for &x in &s[main..] {
        if p.test(x as u64) {
            cnt += 1;
            sum += x as u128;
        }
    }
    (cnt, sum)
}

pub fn count(d: Lanes, p: EncPred) -> u64 {
    match d {
        Lanes::U32(s) => count_u32(s, p),
        Lanes::U8(s) => count_u8(s, p),
        _ => unsafe { neon::count(d, p) },
    }
}

pub fn sum_count(d: Lanes, p: EncPred) -> (u64, u128) {
    match d {
        Lanes::U32(s) => sum_count_u32(s, p),
        _ => unsafe { neon::sum_count(d, p) },
    }
}
