//! Differential tests: every SIMD/asm variant against the portable reference,
//! over random data, every predicate form, every lane width, and lengths that
//! exercise every tail.

use super::*;

/// SplitMix64: deterministic, dependency-free.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

fn preds(rng: &mut Rng, max: u64) -> Vec<EncPred> {
    let mut v = vec![
        EncPred::Ge(0),
        EncPred::Ge(max),
        EncPred::Le(0),
        EncPred::Le(max),
        EncPred::Eq(0),
        EncPred::Ne(max),
        EncPred::Range { lo: 0, span: max },
    ];
    for _ in 0..6 {
        let a = rng.below(max.saturating_add(1));
        let b = rng.below(max.saturating_add(1));
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        v.push(EncPred::Ge(a));
        v.push(EncPred::Le(a));
        v.push(EncPred::Eq(a));
        v.push(EncPred::Ne(a));
        v.push(EncPred::Range { lo, span: hi - lo });
    }
    v
}

fn variants() -> Vec<Variant> {
    available().into_iter().filter(|v| *v != Variant::Portable).collect()
}

fn check_all(d: Lanes, max: u64, rng: &mut Rng) {
    let n = d.len();
    let ps = preds(rng, max);
    let mut m_ref = vec![0u8; n];
    let mut m_got = vec![0u8; n];
    let base_mask: Vec<u8> = (0..n).map(|_| if rng.below(2) == 0 { 0xFF } else { 0 }).collect();
    for &p in &ps {
        let want_count = portable::count(d, p);
        let want_sum = portable::sum_count(d, p);
        portable::mask(d, p, &mut m_ref, false);
        let want_msum = portable::masked_sum(d, &m_ref);
        let want_mcount = portable::mask_count(&m_ref);
        let mut want_pos = Vec::new();
        portable::collect(d, p, 7, &mut want_pos);
        let mut m_and_ref = base_mask.clone();
        portable::mask(d, p, &mut m_and_ref, true);

        // the reference agrees with the scalar definition
        let naive = (0..n).filter(|&i| p.test(d.get(i))).count() as u64;
        assert_eq!(want_count, naive, "portable count {p:?}");

        // pair with a second predicate over a reversed copy of the column
        let other: Vec<u64> = (0..n).map(|i| d.get(n - 1 - i)).collect();
        let p2 = ps[(want_count as usize) % ps.len()];
        let want2 = (0..n).filter(|&i| p.test(d.get(i)) && p2.test(other[i])).count() as u64;
        for v in variants() {
            set_variant(v).unwrap();
            let o = match d {
                Lanes::U8(_) => other.iter().map(|&x| x as u8).collect::<Vec<_>>(),
                _ => Vec::new(),
            };
            if let Lanes::U8(_) = d {
                assert_eq!(count2(d, p, Lanes::U8(&o), p2), want2, "{v:?} count2 {p:?} {p2:?} n={n}");
            }
            let o32: Vec<u32> = other.iter().map(|&x| x as u32).collect();
            if let Lanes::U32(_) = d {
                assert_eq!(count2(d, p, Lanes::U32(&o32), p2), want2, "{v:?} count2 {p:?} {p2:?} n={n}");
            }
            assert_eq!(count(d, p), want_count, "{v:?} count {p:?} n={n} w={}", d.width_bits());
            assert_eq!(sum_count(d, p), want_sum, "{v:?} sum_count {p:?} n={n} w={}", d.width_bits());
            mask(d, p, &mut m_got, false);
            assert_eq!(m_got, m_ref, "{v:?} mask {p:?} n={n} w={}", d.width_bits());
            let mut m_and = base_mask.clone();
            mask(d, p, &mut m_and, true);
            assert_eq!(m_and, m_and_ref, "{v:?} mask-and {p:?} n={n}");
            assert_eq!(masked_sum(d, &m_ref), want_msum, "{v:?} masked_sum {p:?} n={n} w={}", d.width_bits());
            assert_eq!(mask_count(&m_ref), want_mcount, "{v:?} mask_count n={n}");
            let mut pos = Vec::new();
            collect(d, p, 7, &mut pos);
            assert_eq!(pos, want_pos, "{v:?} collect {p:?} n={n} w={}", d.width_bits());
            let mut mpos = Vec::new();
            mask_positions(&m_ref, 7, &mut mpos);
            assert_eq!(mpos, want_pos, "{v:?} mask_positions {p:?} n={n}");
        }
        set_variant(Variant::Portable).unwrap();
    }
}

#[test]
fn kernels_match_reference() {
    let mut rng = Rng(42);
    let lens = [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 1000, 4096, 40_000];
    for &n in &lens {
        // full-range and narrow-range data for each width
        for narrow in [false, true] {
            let m8: u64 = if narrow { 9 } else { u8::MAX as u64 };
            let m16: u64 = if narrow { 300 } else { u16::MAX as u64 };
            let m32: u64 = if narrow { 999_999 } else { u32::MAX as u64 };
            let m64: u64 = if narrow { 1 << 40 } else { u64::MAX };
            let d8: Vec<u8> = (0..n).map(|_| rng.below(m8 + 1) as u8).collect();
            let d16: Vec<u16> = (0..n).map(|_| rng.below(m16 + 1) as u16).collect();
            let d32: Vec<u32> = (0..n).map(|_| rng.below(m32 + 1) as u32).collect();
            let d64: Vec<u64> = (0..n).map(|_| if m64 == u64::MAX { rng.next() } else { rng.below(m64 + 1) }).collect();
            check_all(Lanes::U8(&d8), m8, &mut rng);
            check_all(Lanes::U16(&d16), m16, &mut rng);
            check_all(Lanes::U32(&d32), m32, &mut rng);
            check_all(Lanes::U64(&d64), m64, &mut rng);
        }
    }
}

#[test]
fn u8_counts_flush_before_overflow() {
    // Every lane selected for far more than 255 iterations of every kernel.
    let d = vec![200u8; 300_000];
    let s16 = vec![60_000u16; 300_000];
    for v in available() {
        set_variant(v).unwrap();
        assert_eq!(count(Lanes::U8(&d), EncPred::Ge(1)), 300_000, "{v:?}");
        assert_eq!(sum_count(Lanes::U8(&d), EncPred::Ge(1)), (300_000, 200 * 300_000), "{v:?}");
        assert_eq!(sum_count(Lanes::U16(&s16), EncPred::Ge(1)), (300_000, 60_000 * 300_000), "{v:?}");
        let m = vec![0xFFu8; 300_000];
        assert_eq!(mask_count(&m), 300_000, "{v:?}");
        assert_eq!(masked_sum(Lanes::U8(&d), &m), 200 * 300_000, "{v:?}");
        assert_eq!(masked_sum(Lanes::U16(&s16), &m), 60_000 * 300_000, "{v:?}");
    }
}
