//! Plain-Rust kernels: written in the branch-free, fixed-accumulator shape LLVM
//! vectorizes well, but with no intrinsics. They are the reference the SIMD
//! variants are tested against, and the baseline they are benchmarked against.

use super::{EncPred, Lanes};

/// Rows per inner chunk: small enough that a u32 count and a u64 sum of
/// up-to-32-bit lanes cannot overflow inside it.
const CHUNK: usize = 1 << 20;

pub(crate) trait Lane: Copy + Ord + 'static {
    const WIDE: bool;
    fn from_u64(x: u64) -> Self;
    fn to_u64(self) -> u64;
    fn wsub(self, o: Self) -> Self;
}

macro_rules! lane {
    ($t:ty, $wide:expr) => {
        impl Lane for $t {
            const WIDE: bool = $wide;
            #[inline(always)]
            fn from_u64(x: u64) -> Self {
                x as $t
            }
            #[inline(always)]
            fn to_u64(self) -> u64 {
                self as u64
            }
            #[inline(always)]
            fn wsub(self, o: Self) -> Self {
                self.wrapping_sub(o)
            }
        }
    };
}
lane!(u8, false);
lane!(u16, false);
lane!(u32, false);
lane!(u64, true);

/// Calls `$body` with `$f` bound to a closure evaluating `$p` on a lane of
/// type `$t`, monomorphizing the loop once per predicate form.
macro_rules! with_pred {
    ($t:ty, $p:expr, |$f:ident| $body:expr) => {
        match $p {
            EncPred::Ge(t) => {
                let t = <$t>::from_u64(t);
                let $f = move |x: $t| x >= t;
                $body
            }
            EncPred::Le(t) => {
                let t = <$t>::from_u64(t);
                let $f = move |x: $t| x <= t;
                $body
            }
            EncPred::Range { lo, span } => {
                let lo = <$t>::from_u64(lo);
                let span = <$t>::from_u64(span);
                let $f = move |x: $t| x.wsub(lo) <= span;
                $body
            }
            EncPred::Eq(c) => {
                let c = <$t>::from_u64(c);
                let $f = move |x: $t| x == c;
                $body
            }
            EncPred::Ne(c) => {
                let c = <$t>::from_u64(c);
                let $f = move |x: $t| x != c;
                $body
            }
        }
    };
}

macro_rules! per_lanes {
    ($d:expr, |$s:ident : $t:ident| $body:expr) => {
        match $d {
            Lanes::U8($s) => {
                type $t = u8;
                $body
            }
            Lanes::U16($s) => {
                type $t = u16;
                $body
            }
            Lanes::U32($s) => {
                type $t = u32;
                $body
            }
            Lanes::U64($s) => {
                type $t = u64;
                $body
            }
        }
    };
}

#[inline(always)]
fn count_with<T: Lane>(d: &[T], f: impl Fn(T) -> bool) -> u64 {
    let mut total = 0u64;
    for c in d.chunks(CHUNK) {
        let mut acc = 0u32;
        for &x in c {
            acc += f(x) as u32;
        }
        total += acc as u64;
    }
    total
}

pub fn count(d: Lanes, p: EncPred) -> u64 {
    per_lanes!(d, |s: T| with_pred!(T, p, |f| count_with(s, f)))
}

#[inline(always)]
fn sum_count_with<T: Lane>(d: &[T], f: impl Fn(T) -> bool) -> (u64, u128) {
    let mut cnt = 0u64;
    let mut sum = 0u128;
    for c in d.chunks(CHUNK) {
        let mut n = 0u32;
        if T::WIDE {
            let mut s = 0u128;
            for &x in c {
                let m = f(x);
                n += m as u32;
                s += (x.to_u64() & 0u64.wrapping_sub(m as u64)) as u128;
            }
            sum += s;
        } else {
            let mut s = 0u64;
            for &x in c {
                let m = f(x);
                n += m as u32;
                s += x.to_u64() & 0u64.wrapping_sub(m as u64);
            }
            sum += s as u128;
        }
        cnt += n as u64;
    }
    (cnt, sum)
}

pub fn sum_count(d: Lanes, p: EncPred) -> (u64, u128) {
    per_lanes!(d, |s: T| with_pred!(T, p, |f| sum_count_with(s, f)))
}

#[inline(always)]
fn mask_with<T: Lane>(d: &[T], out: &mut [u8], and: bool, f: impl Fn(T) -> bool) {
    if and {
        for (o, &x) in out.iter_mut().zip(d) {
            *o &= 0u8.wrapping_sub(f(x) as u8);
        }
    } else {
        for (o, &x) in out.iter_mut().zip(d) {
            *o = 0u8.wrapping_sub(f(x) as u8);
        }
    }
}

pub fn mask(d: Lanes, p: EncPred, out: &mut [u8], and: bool) {
    per_lanes!(d, |s: T| with_pred!(T, p, |f| mask_with(s, out, and, f)))
}

pub fn mask_count(m: &[u8]) -> u64 {
    let mut total = 0u64;
    for c in m.chunks(CHUNK) {
        let mut acc = 0u32;
        for &b in c {
            acc += (b & 1) as u32;
        }
        total += acc as u64;
    }
    total
}

#[inline(always)]
fn masked_sum_with<T: Lane>(d: &[T], m: &[u8]) -> u128 {
    let mut sum = 0u128;
    for (dc, mc) in d.chunks(CHUNK).zip(m.chunks(CHUNK)) {
        if T::WIDE {
            for (&x, &b) in dc.iter().zip(mc) {
                sum += (x.to_u64() & 0u64.wrapping_sub((b & 1) as u64)) as u128;
            }
        } else {
            let mut s = 0u64;
            for (&x, &b) in dc.iter().zip(mc) {
                s += x.to_u64() & 0u64.wrapping_sub((b & 1) as u64);
            }
            sum += s as u128;
        }
    }
    sum
}

pub fn masked_sum(d: Lanes, m: &[u8]) -> u128 {
    per_lanes!(d, |s: T| masked_sum_with::<T>(s, m))
}

#[inline(always)]
fn collect_with<T: Lane>(d: &[T], base: u32, out: &mut Vec<u32>, f: impl Fn(T) -> bool) {
    for (i, &x) in d.iter().enumerate() {
        if f(x) {
            out.push(base + i as u32);
        }
    }
}

pub fn collect(d: Lanes, p: EncPred, base: u32, out: &mut Vec<u32>) {
    per_lanes!(d, |s: T| with_pred!(T, p, |f| collect_with(s, base, out, f)))
}

pub fn mask_positions(m: &[u8], base: u32, out: &mut Vec<u32>) {
    for (i, &b) in m.iter().enumerate() {
        if b != 0 {
            out.push(base + i as u32);
        }
    }
}

pub fn count2(a: Lanes, pa: EncPred, b: Lanes, pb: EncPred) -> u64 {
    // Two mask passes over L1-sized blocks: as close to fused as plain Rust gets.
    let mut ma = [0u8; 4096];
    let mut total = 0;
    let mut i = 0;
    while i < a.len() {
        let e = (i + 4096).min(a.len());
        let m = &mut ma[..e - i];
        mask(a.slice(i, e), pa, m, false);
        mask(b.slice(i, e), pb, m, true);
        total += mask_count(m);
        i = e;
    }
    total
}
