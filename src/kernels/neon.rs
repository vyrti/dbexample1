//! NEON intrinsics kernels. NEON is baseline on aarch64, so no runtime
//! detection or `#[target_feature]` is needed.
//!
//! Masks are produced at lane width by the compare and narrowed to one byte per
//! row with `uzp1` (u16: 1 op per 16 rows, u32: 3 ops per 16 rows). Counts
//! accumulate by subtracting all-ones masks; sums by `uadalp` (pairwise
//! add-accumulate long). Accumulators are flushed before they can overflow.
//! u64 lanes (columns whose range exceeds 2^32) fall back to the portable code.

use super::{EncPred, Lanes, portable};
use core::arch::aarch64::*;

/// Binds `$f` to a closure computing the all-ones/zero lane mask of `$p`.
macro_rules! pred_v {
    ($p:expr, $T:ty, $dup:ident, $cge:ident, $cle:ident, $ceq:ident, $sub:ident, $not:ident,
     |$f:ident| $body:expr) => {
        match $p {
            EncPred::Ge(t) => {
                let tv = unsafe { $dup(t as $T) };
                let $f = move |x| unsafe { $cge(x, tv) };
                $body
            }
            EncPred::Le(t) => {
                let tv = unsafe { $dup(t as $T) };
                let $f = move |x| unsafe { $cle(x, tv) };
                $body
            }
            EncPred::Range { lo, span } => {
                let lv = unsafe { $dup(lo as $T) };
                let sv = unsafe { $dup(span as $T) };
                let $f = move |x| unsafe { $cle($sub(x, lv), sv) };
                $body
            }
            EncPred::Eq(c) => {
                let cv = unsafe { $dup(c as $T) };
                let $f = move |x| unsafe { $ceq(x, cv) };
                $body
            }
            EncPred::Ne(c) => {
                let cv = unsafe { $dup(c as $T) };
                let $f = move |x| unsafe { $not($ceq(x, cv)) };
                $body
            }
        }
    };
}

macro_rules! pred_u8 {
    ($p:expr, |$f:ident| $body:expr) => {
        pred_v!($p, u8, vdupq_n_u8, vcgeq_u8, vcleq_u8, vceqq_u8, vsubq_u8, vmvnq_u8, |$f| $body)
    };
}
macro_rules! pred_u16 {
    ($p:expr, |$f:ident| $body:expr) => {
        pred_v!($p, u16, vdupq_n_u16, vcgeq_u16, vcleq_u16, vceqq_u16, vsubq_u16, vmvnq_u16, |$f| $body)
    };
}
macro_rules! pred_u32 {
    ($p:expr, |$f:ident| $body:expr) => {
        pred_v!($p, u32, vdupq_n_u32, vcgeq_u32, vcleq_u32, vceqq_u32, vsubq_u32, vmvnq_u32, |$f| $body)
    };
}

// ---- 16 rows -> 16 mask bytes, per lane width.

#[inline(always)]
unsafe fn pack_u8(p: *const u8, f: &impl Fn(uint8x16_t) -> uint8x16_t) -> uint8x16_t {
    unsafe { f(vld1q_u8(p)) }
}

#[inline(always)]
unsafe fn pack_u16(p: *const u16, f: &impl Fn(uint16x8_t) -> uint16x8_t) -> uint8x16_t {
    unsafe {
        let a = f(vld1q_u16(p));
        let b = f(vld1q_u16(p.add(8)));
        vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b))
    }
}

#[inline(always)]
unsafe fn pack_u32(p: *const u32, f: &impl Fn(uint32x4_t) -> uint32x4_t) -> uint8x16_t {
    unsafe {
        let a = f(vld1q_u32(p));
        let b = f(vld1q_u32(p.add(4)));
        let c = f(vld1q_u32(p.add(8)));
        let d = f(vld1q_u32(p.add(12)));
        let ab = vuzp1q_u16(vreinterpretq_u16_u32(a), vreinterpretq_u16_u32(b));
        let cd = vuzp1q_u16(vreinterpretq_u16_u32(c), vreinterpretq_u16_u32(d));
        vuzp1q_u8(vreinterpretq_u8_u16(ab), vreinterpretq_u8_u16(cd))
    }
}

/// Runs `$body` with `$g` bound to "mask bytes of rows [i, i+16)" for `$d`.
macro_rules! with_groups {
    ($d:expr, $p:expr, |$g:ident, $n:ident| $body:expr) => {
        match $d {
            Lanes::U8(s) => pred_u8!($p, |f| {
                let ptr = s.as_ptr();
                let $n = s.len();
                let $g = |i: usize| unsafe { pack_u8(ptr.add(i), &f) };
                $body
            }),
            Lanes::U16(s) => pred_u16!($p, |f| {
                let ptr = s.as_ptr();
                let $n = s.len();
                let $g = |i: usize| unsafe { pack_u16(ptr.add(i), &f) };
                $body
            }),
            Lanes::U32(s) => pred_u32!($p, |f| {
                let ptr = s.as_ptr();
                let $n = s.len();
                let $g = |i: usize| unsafe { pack_u32(ptr.add(i), &f) };
                $body
            }),
            Lanes::U64(_) => unreachable!(),
        }
    };
}

/// Counts selected rows of the first `main` rows (a multiple of 64), four
/// independent u8 accumulators, flushed every 255 iterations.
#[inline(always)]
unsafe fn count_groups(main: usize, g: impl Fn(usize) -> uint8x16_t) -> u64 {
    unsafe {
        let mut total = 0u64;
        let mut i = 0;
        while i < main {
            let end = main.min(i + 255 * 64);
            let mut a0 = vdupq_n_u8(0);
            let mut a1 = vdupq_n_u8(0);
            let mut a2 = vdupq_n_u8(0);
            let mut a3 = vdupq_n_u8(0);
            while i < end {
                a0 = vsubq_u8(a0, g(i));
                a1 = vsubq_u8(a1, g(i + 16));
                a2 = vsubq_u8(a2, g(i + 32));
                a3 = vsubq_u8(a3, g(i + 48));
                i += 64;
            }
            total += vaddlvq_u8(a0) as u64
                + vaddlvq_u8(a1) as u64
                + vaddlvq_u8(a2) as u64
                + vaddlvq_u8(a3) as u64;
        }
        total
    }
}

pub unsafe fn count(d: Lanes, p: EncPred) -> u64 {
    if let Lanes::U64(_) = d {
        return portable::count(d, p);
    }
    let n = d.len();
    let main = n & !63;
    let head = with_groups!(d, p, |g, _n| unsafe { count_groups(main, g) });
    let tail = (main..n).filter(|&i| p.test(d.get(i))).count() as u64;
    head + tail
}

pub unsafe fn count2(a: Lanes, pa: EncPred, b: Lanes, pb: EncPred) -> u64 {
    if matches!(a, Lanes::U64(_)) || matches!(b, Lanes::U64(_)) {
        return portable::count2(a, pa, b, pb);
    }
    let n = a.len();
    let main = n & !63;
    let head = with_groups!(a, pa, |ga, _n| with_groups!(b, pb, |gb, _m| unsafe {
        count_groups(main, |i| vandq_u8(ga(i), gb(i)))
    }));
    let tail = (main..n).filter(|&i| pa.test(a.get(i)) && pb.test(b.get(i))).count() as u64;
    head + tail
}

pub unsafe fn mask(d: Lanes, p: EncPred, out: &mut [u8], and: bool) {
    if let Lanes::U64(_) = d {
        return portable::mask(d, p, out, and);
    }
    let n = d.len();
    let main = n & !15;
    let o = out.as_mut_ptr();
    with_groups!(d, p, |g, _n| unsafe {
        let mut i = 0;
        if and {
            while i < main {
                vst1q_u8(o.add(i), vandq_u8(vld1q_u8(o.add(i)), g(i)));
                i += 16;
            }
        } else {
            while i < main {
                vst1q_u8(o.add(i), g(i));
                i += 16;
            }
        }
    });
    for i in main..n {
        let m = 0u8.wrapping_sub(p.test(d.get(i)) as u8);
        if and {
            out[i] &= m;
        } else {
            out[i] = m;
        }
    }
}

/// Pushes the positions of set bytes in a 16-byte mask. `shrn #4` turns each
/// byte into a nibble of a u64; keeping one bit per nibble lets `b & (b-1)`
/// step through them.
#[inline(always)]
unsafe fn push_set(m: uint8x16_t, base: u32, out: &mut Vec<u32>) {
    unsafe {
        let nib = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(m))));
        let mut b = nib & 0x8888_8888_8888_8888;
        while b != 0 {
            out.push(base + (b.trailing_zeros() >> 2));
            b &= b - 1;
        }
    }
}

pub unsafe fn collect(d: Lanes, p: EncPred, base: u32, out: &mut Vec<u32>) {
    if let Lanes::U64(_) = d {
        return portable::collect(d, p, base, out);
    }
    let n = d.len();
    let main = n & !15;
    with_groups!(d, p, |g, _n| unsafe {
        let mut i = 0;
        while i < main {
            let m = g(i);
            // Most groups select nothing on the paths that use this (top-N
            // prefilter, sparse selections): one horizontal max rejects them.
            if vmaxvq_u8(m) != 0 {
                push_set(m, base + i as u32, out);
            }
            i += 16;
        }
    });
    for i in main..n {
        if p.test(d.get(i)) {
            out.push(base + i as u32);
        }
    }
}

pub unsafe fn mask_positions(m: &[u8], base: u32, out: &mut Vec<u32>) {
    let n = m.len();
    let main = n & !15;
    let ptr = m.as_ptr();
    let mut i = 0;
    unsafe {
        while i < main {
            let v = vld1q_u8(ptr.add(i));
            if vmaxvq_u8(v) != 0 {
                push_set(v, base + i as u32, out);
            }
            i += 16;
        }
    }
    for (j, &b) in m.iter().enumerate().skip(main) {
        if b != 0 {
            out.push(base + j as u32);
        }
    }
}

pub unsafe fn mask_count(m: &[u8]) -> u64 {
    let n = m.len();
    let main = n & !63;
    let ptr = m.as_ptr();
    let head = unsafe { count_groups(main, |i| vld1q_u8(ptr.add(i))) };
    head + m[main..].iter().filter(|&&b| b != 0).count() as u64
}

// ---- Sums. Width-specific: the values themselves have to be widened.

#[inline(always)]
unsafe fn sum_count_u8(s: &[u8], f: impl Fn(uint8x16_t) -> uint8x16_t) -> (u64, u128) {
    unsafe {
        let n = s.len();
        let main = n & !63;
        let ptr = s.as_ptr();
        let (mut cnt, mut sum) = (0u64, 0u64);
        let mut i = 0;
        while i < main {
            // u16 sum lanes gain <= 2*255 per step: flush every 128 steps.
            let end = main.min(i + 128 * 64);
            let (mut c0, mut c1, mut c2, mut c3) =
                (vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0));
            let (mut s0, mut s1, mut s2, mut s3) =
                (vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0));
            while i < end {
                let x0 = vld1q_u8(ptr.add(i));
                let x1 = vld1q_u8(ptr.add(i + 16));
                let x2 = vld1q_u8(ptr.add(i + 32));
                let x3 = vld1q_u8(ptr.add(i + 48));
                let (m0, m1, m2, m3) = (f(x0), f(x1), f(x2), f(x3));
                c0 = vsubq_u8(c0, m0);
                c1 = vsubq_u8(c1, m1);
                c2 = vsubq_u8(c2, m2);
                c3 = vsubq_u8(c3, m3);
                s0 = vpadalq_u8(s0, vandq_u8(x0, m0));
                s1 = vpadalq_u8(s1, vandq_u8(x1, m1));
                s2 = vpadalq_u8(s2, vandq_u8(x2, m2));
                s3 = vpadalq_u8(s3, vandq_u8(x3, m3));
                i += 64;
            }
            cnt += vaddlvq_u8(c0) as u64
                + vaddlvq_u8(c1) as u64
                + vaddlvq_u8(c2) as u64
                + vaddlvq_u8(c3) as u64;
            sum += vaddlvq_u16(s0) as u64
                + vaddlvq_u16(s1) as u64
                + vaddlvq_u16(s2) as u64
                + vaddlvq_u16(s3) as u64;
        }
        (cnt, sum as u128)
    }
}

#[inline(always)]
unsafe fn sum_count_u16(s: &[u16], f: impl Fn(uint16x8_t) -> uint16x8_t) -> (u64, u128) {
    unsafe {
        let n = s.len();
        let main = n & !31;
        let ptr = s.as_ptr();
        let (mut cnt, mut sum) = (0u64, 0u64);
        let mut i = 0;
        while i < main {
            // u32 sum lanes gain <= 2*65535 per step: flush every 32768 steps.
            let end = main.min(i + 32768 * 32);
            let (mut c0, mut c1, mut c2, mut c3) =
                (vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0));
            let (mut s0, mut s1, mut s2, mut s3) =
                (vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0));
            while i < end {
                let x0 = vld1q_u16(ptr.add(i));
                let x1 = vld1q_u16(ptr.add(i + 8));
                let x2 = vld1q_u16(ptr.add(i + 16));
                let x3 = vld1q_u16(ptr.add(i + 24));
                let (m0, m1, m2, m3) = (f(x0), f(x1), f(x2), f(x3));
                c0 = vsubq_u16(c0, m0);
                c1 = vsubq_u16(c1, m1);
                c2 = vsubq_u16(c2, m2);
                c3 = vsubq_u16(c3, m3);
                s0 = vpadalq_u16(s0, vandq_u16(x0, m0));
                s1 = vpadalq_u16(s1, vandq_u16(x1, m1));
                s2 = vpadalq_u16(s2, vandq_u16(x2, m2));
                s3 = vpadalq_u16(s3, vandq_u16(x3, m3));
                i += 32;
            }
            cnt += vaddlvq_u16(c0) as u64
                + vaddlvq_u16(c1) as u64
                + vaddlvq_u16(c2) as u64
                + vaddlvq_u16(c3) as u64;
            sum += vaddlvq_u32(s0) + vaddlvq_u32(s1) + vaddlvq_u32(s2) + vaddlvq_u32(s3);
        }
        (cnt, sum as u128)
    }
}

#[inline(always)]
unsafe fn sum_count_u32(s: &[u32], f: impl Fn(uint32x4_t) -> uint32x4_t) -> (u64, u128) {
    unsafe {
        let n = s.len();
        let main = n & !15;
        let ptr = s.as_ptr();
        let (mut c0, mut c1, mut c2, mut c3) =
            (vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0));
        let (mut s0, mut s1, mut s2, mut s3) =
            (vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0));
        let mut i = 0;
        // u32 count lanes gain 1 per step and u64 sum lanes < 2^33 per step:
        // no flush needed below 2^32 steps (2^36 rows).
        while i < main {
            let x0 = vld1q_u32(ptr.add(i));
            let x1 = vld1q_u32(ptr.add(i + 4));
            let x2 = vld1q_u32(ptr.add(i + 8));
            let x3 = vld1q_u32(ptr.add(i + 12));
            let (m0, m1, m2, m3) = (f(x0), f(x1), f(x2), f(x3));
            c0 = vsubq_u32(c0, m0);
            c1 = vsubq_u32(c1, m1);
            c2 = vsubq_u32(c2, m2);
            c3 = vsubq_u32(c3, m3);
            s0 = vpadalq_u32(s0, vandq_u32(x0, m0));
            s1 = vpadalq_u32(s1, vandq_u32(x1, m1));
            s2 = vpadalq_u32(s2, vandq_u32(x2, m2));
            s3 = vpadalq_u32(s3, vandq_u32(x3, m3));
            i += 16;
        }
        let cnt = vaddlvq_u32(c0) + vaddlvq_u32(c1) + vaddlvq_u32(c2) + vaddlvq_u32(c3);
        let sum = vaddvq_u64(s0) as u128
            + vaddvq_u64(s1) as u128
            + vaddvq_u64(s2) as u128
            + vaddvq_u64(s3) as u128;
        (cnt, sum)
    }
}

/// Scalar tail for sums: rows `[from, len)`.
fn sum_count_tail(d: Lanes, p: EncPred, from: usize) -> (u64, u128) {
    let (mut c, mut s) = (0u64, 0u128);
    for i in from..d.len() {
        let x = d.get(i);
        if p.test(x) {
            c += 1;
            s += x as u128;
        }
    }
    (c, s)
}

pub unsafe fn sum_count(d: Lanes, p: EncPred) -> (u64, u128) {
    let (head, main) = match d {
        Lanes::U8(s) => (pred_u8!(p, |f| unsafe { sum_count_u8(s, f) }), s.len() & !63),
        Lanes::U16(s) => (pred_u16!(p, |f| unsafe { sum_count_u16(s, f) }), s.len() & !31),
        Lanes::U32(s) => (pred_u32!(p, |f| unsafe { sum_count_u32(s, f) }), s.len() & !15),
        Lanes::U64(_) => return portable::sum_count(d, p),
    };
    let tail = sum_count_tail(d, p, main);
    (head.0 + tail.0, head.1 + tail.1)
}

pub unsafe fn masked_sum(d: Lanes, m: &[u8]) -> u128 {
    let mp = m.as_ptr();
    let n = d.len();
    let main = n & !15;
    let mut sum: u128 = 0;
    unsafe {
        match d {
            Lanes::U8(s) => {
                let ptr = s.as_ptr();
                let mut i = 0;
                while i < main {
                    let end = main.min(i + 128 * 16);
                    let mut acc = vdupq_n_u16(0);
                    while i < end {
                        let x = vandq_u8(vld1q_u8(ptr.add(i)), vld1q_u8(mp.add(i)));
                        acc = vpadalq_u8(acc, x);
                        i += 16;
                    }
                    sum += vaddlvq_u16(acc) as u128;
                }
            }
            Lanes::U16(s) => {
                let ptr = s.as_ptr();
                let mut i = 0;
                while i < main {
                    let end = main.min(i + 32768 * 16);
                    let (mut a0, mut a1) = (vdupq_n_u32(0), vdupq_n_u32(0));
                    while i < end {
                        let mb = vld1q_u8(mp.add(i));
                        let m0 = vreinterpretq_u16_u8(vzip1q_u8(mb, mb));
                        let m1 = vreinterpretq_u16_u8(vzip2q_u8(mb, mb));
                        a0 = vpadalq_u16(a0, vandq_u16(vld1q_u16(ptr.add(i)), m0));
                        a1 = vpadalq_u16(a1, vandq_u16(vld1q_u16(ptr.add(i + 8)), m1));
                        i += 16;
                    }
                    sum += (vaddlvq_u32(a0) + vaddlvq_u32(a1)) as u128;
                }
            }
            Lanes::U32(s) => {
                let ptr = s.as_ptr();
                let (mut a0, mut a1, mut a2, mut a3) =
                    (vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0), vdupq_n_u64(0));
                let mut i = 0;
                while i < main {
                    let mb = vld1q_u8(mp.add(i));
                    let lo = vzip1q_u8(mb, mb);
                    let hi = vzip2q_u8(mb, mb);
                    let m0 = vreinterpretq_u32_u8(vreinterpretq_u8_u16(vzip1q_u16(
                        vreinterpretq_u16_u8(lo),
                        vreinterpretq_u16_u8(lo),
                    )));
                    let m1 = vreinterpretq_u32_u8(vreinterpretq_u8_u16(vzip2q_u16(
                        vreinterpretq_u16_u8(lo),
                        vreinterpretq_u16_u8(lo),
                    )));
                    let m2 = vreinterpretq_u32_u8(vreinterpretq_u8_u16(vzip1q_u16(
                        vreinterpretq_u16_u8(hi),
                        vreinterpretq_u16_u8(hi),
                    )));
                    let m3 = vreinterpretq_u32_u8(vreinterpretq_u8_u16(vzip2q_u16(
                        vreinterpretq_u16_u8(hi),
                        vreinterpretq_u16_u8(hi),
                    )));
                    a0 = vpadalq_u32(a0, vandq_u32(vld1q_u32(ptr.add(i)), m0));
                    a1 = vpadalq_u32(a1, vandq_u32(vld1q_u32(ptr.add(i + 4)), m1));
                    a2 = vpadalq_u32(a2, vandq_u32(vld1q_u32(ptr.add(i + 8)), m2));
                    a3 = vpadalq_u32(a3, vandq_u32(vld1q_u32(ptr.add(i + 12)), m3));
                    i += 16;
                }
                sum = vaddvq_u64(a0) as u128
                    + vaddvq_u64(a1) as u128
                    + vaddvq_u64(a2) as u128
                    + vaddvq_u64(a3) as u128;
            }
            Lanes::U64(_) => return portable::masked_sum(d, m),
        }
    }
    for i in main..n {
        if m[i] != 0 {
            sum += d.get(i) as u128;
        }
    }
    sum
}
