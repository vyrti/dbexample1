//! Scan kernels over encoded (frame-of-reference, unsigned) integer lanes.
//!
//! Every integer column is stored as `value = base + enc` with `enc` an
//! unsigned u8/u16/u32/u64 lane, so every comparison a query can make on a
//! column is lowered -- once per query, not once per row -- into one of the
//! [`EncPred`] forms over raw lanes. The kernels never decode.
//!
//! Four implementations, selected at runtime (see [`set_variant`]):
//! - `Portable`: plain Rust loops, vectorized (or not) by LLVM alone.
//! - `Neon`: hand-written NEON intrinsics (aarch64).
//! - `Asm`: hand-written aarch64 assembly for the hottest loops, NEON
//!   intrinsics elsewhere.
//! - `Avx2`: AVX2 intrinsics (x86-64, detected at runtime).

use std::sync::atomic::{AtomicU8, Ordering};

pub mod portable;

#[cfg(target_arch = "aarch64")]
pub mod asm;
#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

/// A predicate over encoded lanes. Bounds are inclusive and always within the
/// lane's range, which is what makes `Range`'s single wrapping compare exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncPred {
    /// `enc >= t`
    Ge(u64),
    /// `enc <= t`
    Le(u64),
    /// `lo <= enc <= lo + span`, evaluated as `enc - lo <= span` (wrapping).
    Range { lo: u64, span: u64 },
    Eq(u64),
    Ne(u64),
}

impl EncPred {
    #[inline(always)]
    pub fn test(self, x: u64) -> bool {
        match self {
            EncPred::Ge(t) => x >= t,
            EncPred::Le(t) => x <= t,
            EncPred::Range { lo, span } => x.wrapping_sub(lo) <= span,
            EncPred::Eq(c) => x == c,
            EncPred::Ne(c) => x != c,
        }
    }

    /// What the predicate says about every row of a zone whose encoded values
    /// lie in `[zmin, zmax]`.
    #[inline]
    pub fn classify(self, zmin: u64, zmax: u64) -> ZoneHit {
        let (all, none) = match self {
            EncPred::Ge(t) => (zmin >= t, zmax < t),
            EncPred::Le(t) => (zmax <= t, zmin > t),
            EncPred::Range { lo, span } => {
                let hi = lo + span;
                (zmin >= lo && zmax <= hi, zmax < lo || zmin > hi)
            }
            EncPred::Eq(c) => (zmin == c && zmax == c, c < zmin || c > zmax),
            EncPred::Ne(c) => (c < zmin || c > zmax, zmin == c && zmax == c),
        };
        if none {
            ZoneHit::None
        } else if all {
            ZoneHit::All
        } else {
            ZoneHit::Some
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoneHit {
    None,
    Some,
    All,
}

/// A borrowed column of encoded lanes.
#[derive(Clone, Copy, Debug)]
pub enum Lanes<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
}

impl<'a> Lanes<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Lanes::U8(d) => d.len(),
            Lanes::U16(d) => d.len(),
            Lanes::U32(d) => d.len(),
            Lanes::U64(d) => d.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> u64 {
        match self {
            Lanes::U8(d) => d[i] as u64,
            Lanes::U16(d) => d[i] as u64,
            Lanes::U32(d) => d[i] as u64,
            Lanes::U64(d) => d[i],
        }
    }

    /// # Safety
    /// `i < self.len()`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, i: usize) -> u64 {
        unsafe {
            match self {
                Lanes::U8(d) => *d.get_unchecked(i) as u64,
                Lanes::U16(d) => *d.get_unchecked(i) as u64,
                Lanes::U32(d) => *d.get_unchecked(i) as u64,
                Lanes::U64(d) => *d.get_unchecked(i),
            }
        }
    }

    #[inline]
    pub fn slice(&self, from: usize, to: usize) -> Lanes<'a> {
        match self {
            Lanes::U8(d) => Lanes::U8(&d[from..to]),
            Lanes::U16(d) => Lanes::U16(&d[from..to]),
            Lanes::U32(d) => Lanes::U32(&d[from..to]),
            Lanes::U64(d) => Lanes::U64(&d[from..to]),
        }
    }

    pub fn width_bits(&self) -> u32 {
        match self {
            Lanes::U8(_) => 8,
            Lanes::U16(_) => 16,
            Lanes::U32(_) => 32,
            Lanes::U64(_) => 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Portable = 1,
    Neon = 2,
    Asm = 3,
    Avx2 = 4,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Variant::Portable => "portable",
            Variant::Neon => "neon",
            Variant::Asm => "asm",
            Variant::Avx2 => "avx2",
        }
    }

    pub fn parse(s: &str) -> Option<Variant> {
        match s.to_ascii_lowercase().as_str() {
            "portable" | "scalar" => Some(Variant::Portable),
            "neon" => Some(Variant::Neon),
            "asm" => Some(Variant::Asm),
            "avx2" => Some(Variant::Avx2),
            _ => None,
        }
    }

    fn from_u8(v: u8) -> Variant {
        match v {
            2 => Variant::Neon,
            3 => Variant::Asm,
            4 => Variant::Avx2,
            _ => Variant::Portable,
        }
    }
}

/// The variants this CPU can run, best first.
pub fn available() -> Vec<Variant> {
    #[allow(unused_mut)]
    let mut v = Vec::new();
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            v.push(Variant::Asm);
            v.push(Variant::Neon);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            v.push(Variant::Avx2);
        }
    }
    v.push(Variant::Portable);
    v
}

static VARIANT: AtomicU8 = AtomicU8::new(0);

/// The variant in use. Defaults to `SINEW_KERNEL` if set and available, else
/// the best available.
#[inline]
pub fn variant() -> Variant {
    let v = VARIANT.load(Ordering::Relaxed);
    if v != 0 {
        return Variant::from_u8(v);
    }
    init_variant()
}

#[cold]
fn init_variant() -> Variant {
    let avail = available();
    let chosen = std::env::var("SINEW_KERNEL")
        .ok()
        .and_then(|s| Variant::parse(&s))
        .filter(|v| avail.contains(v))
        .unwrap_or(avail[0]);
    VARIANT.store(chosen as u8, Ordering::Relaxed);
    chosen
}

/// Selects a kernel variant for the whole process.
pub fn set_variant(v: Variant) -> Result<(), String> {
    if !available().contains(&v) {
        return Err(format!("kernel variant {} is not available on this CPU", v.name()));
    }
    VARIANT.store(v as u8, Ordering::Relaxed);
    Ok(())
}

// ---- Dispatch. Each entry point picks the implementation once per call; the
// ---- branch is perfectly predicted and costs nothing next to a scan.

/// Number of lanes satisfying `p`.
#[inline]
pub fn count(d: Lanes, p: EncPred) -> u64 {
    match variant() {
        #[cfg(target_arch = "aarch64")]
        Variant::Asm => asm::count(d, p),
        #[cfg(target_arch = "aarch64")]
        Variant::Neon => unsafe { neon::count(d, p) },
        #[cfg(target_arch = "x86_64")]
        Variant::Avx2 => unsafe { avx2::count(d, p) },
        _ => portable::count(d, p),
    }
}

/// Number of rows satisfying both `pa` on `a` and `pb` on `b` (equal lengths),
/// fused: no mask is written.
#[inline]
pub fn count2(a: Lanes, pa: EncPred, b: Lanes, pb: EncPred) -> u64 {
    assert_eq!(a.len(), b.len());
    match variant() {
        #[cfg(target_arch = "aarch64")]
        Variant::Asm | Variant::Neon => unsafe { neon::count2(a, pa, b, pb) },
        #[cfg(target_arch = "x86_64")]
        Variant::Avx2 => unsafe { avx2::count2(a, pa, b, pb) },
        _ => portable::count2(a, pa, b, pb),
    }
}

/// `(count, sum of encoded values)` over lanes satisfying `p`.
#[inline]
pub fn sum_count(d: Lanes, p: EncPred) -> (u64, u128) {
    match variant() {
        #[cfg(target_arch = "aarch64")]
        Variant::Asm => asm::sum_count(d, p),
        #[cfg(target_arch = "aarch64")]
        Variant::Neon => unsafe { neon::sum_count(d, p) },
        #[cfg(target_arch = "x86_64")]
        Variant::Avx2 => unsafe { avx2::sum_count(d, p) },
        _ => portable::sum_count(d, p),
    }
}

/// Writes a selection byte mask (0xFF selected / 0x00 not) for `p`, or ANDs
/// it into `out` when `and` is set. `out.len() == d.len()`.
#[inline]
pub fn mask(d: Lanes, p: EncPred, out: &mut [u8], and: bool) {
    assert_eq!(out.len(), d.len());
    match variant() {
        #[cfg(target_arch = "aarch64")]
        Variant::Asm | Variant::Neon => unsafe { neon::mask(d, p, out, and) },
        #[cfg(target_arch = "x86_64")]
        Variant::Avx2 => unsafe { avx2::mask(d, p, out, and) },
        _ => portable::mask(d, p, out, and),
    }
}

/// `out[i] &= m[i]`.
#[inline]
pub fn mask_and(out: &mut [u8], m: &[u8]) {
    assert_eq!(out.len(), m.len());
    for (o, &x) in out.iter_mut().zip(m) {
        *o &= x;
    }
}

/// Number of selected rows in a byte mask.
#[inline]
pub fn mask_count(m: &[u8]) -> u64 {
    match variant() {
        #[cfg(target_arch = "aarch64")]
        Variant::Asm | Variant::Neon => unsafe { neon::mask_count(m) },
        #[cfg(target_arch = "x86_64")]
        Variant::Avx2 => unsafe { avx2::mask_count(m) },
        _ => portable::mask_count(m),
    }
}

/// Sum of encoded values of selected rows.
#[inline]
pub fn masked_sum(d: Lanes, m: &[u8]) -> u128 {
    assert_eq!(m.len(), d.len());
    match variant() {
        #[cfg(target_arch = "aarch64")]
        Variant::Asm | Variant::Neon => unsafe { neon::masked_sum(d, m) },
        #[cfg(target_arch = "x86_64")]
        Variant::Avx2 => unsafe { avx2::masked_sum(d, m) },
        _ => portable::masked_sum(d, m),
    }
}

/// Appends `base + i` for every lane `i` satisfying `p`.
#[inline]
pub fn collect(d: Lanes, p: EncPred, base: u32, out: &mut Vec<u32>) {
    match variant() {
        #[cfg(target_arch = "aarch64")]
        Variant::Asm | Variant::Neon => unsafe { neon::collect(d, p, base, out) },
        #[cfg(target_arch = "x86_64")]
        Variant::Avx2 => unsafe { avx2::collect(d, p, base, out) },
        _ => portable::collect(d, p, base, out),
    }
}

/// Appends `base + i` for every selected byte of `m`.
#[inline]
pub fn mask_positions(m: &[u8], base: u32, out: &mut Vec<u32>) {
    match variant() {
        #[cfg(target_arch = "aarch64")]
        Variant::Asm | Variant::Neon => unsafe { neon::mask_positions(m, base, out) },
        #[cfg(target_arch = "x86_64")]
        Variant::Avx2 => unsafe { avx2::mask_positions(m, base, out) },
        _ => portable::mask_positions(m, base, out),
    }
}

#[cfg(test)]
mod tests;
