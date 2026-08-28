use super::{F128, F256Unreduced, ghash_reduce};
use core::arch::x86_64::*;

/// 64×64 carry-less product, returned as a 128-bit vector {lo, hi}.
///
/// # Safety
/// Caller must ensure `pclmulqdq` (and `sse4.1` for the lane extracts in
/// callers) is enabled — statically satisfied since every caller is itself
/// `#[target_feature(enable = "pclmulqdq,sse4.1")]`.
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn pmull(a: u64, b: u64) -> __m128i {
    let va = _mm_set_epi64x(0, a as i64);
    let vb = _mm_set_epi64x(0, b as i64);
    // IMM8 = 0x00: low qword of a × low qword of b.
    _mm_clmulepi64_si128::<0x00>(va, vb)
}

/// SSE2-only lane extract: shift the 128-bit register right by 8 bytes and
/// store the low 64 bits. Equivalent to `_mm_extract_epi64::<1>` from SSE4.1
/// but works under the broader `pclmulqdq,sse2` floor.
///
/// # Safety
/// Requires `sse2`; the function carries the attribute.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn hi64(v: __m128i) -> u64 {
    // srli by 8 bytes moves the high qword into the low qword position; we
    // then read that lane via a `_mm_cvtsi128_si64` (movq), which extracts the
    // low 64 bits of an __m128i and is an SSE2 baseline intrinsic.
    let shifted = _mm_srli_si128::<8>(v);
    _mm_cvtsi128_si64(shifted) as u64
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane0(v: __m128i) -> u64 {
    _mm_extract_epi64::<0>(v) as u64
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane1(v: __m128i) -> u64 {
    _mm_extract_epi64::<1>(v) as u64
}

/// `a · b` for a multiplier whose high limb is zero.
///
/// With `b.hi == 0` both limb products that involve `b.hi` vanish, so the
/// 256-bit product is exactly `a.lo·b_lo + (a.hi·b_lo)·x^64`. That is **2
/// CLMUL** plus the shift-only `ghash_reduce`, against the 5 CLMUL of the
/// general Karatsuba+Barrett path this crate's `Mul` uses on x86.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute. `b_lo` is the multiplier's low limb; its high limb must be 0.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_low_rhs(a: F128, b_lo: u64) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p0 = pmull(a.lo, b_lo);
        let q = pmull(a.hi, b_lo);
        super::ghash_reduce(lane0(p0), lane1(p0) ^ lane0(q), lane1(q), 0)
    }
}

/// Schoolbook 4 CLMUL — fully independent products, then scalar reduction.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_schoolbook(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let cr_lo = lane0(cross);
        let cr_hi = lane1(cross);

        ghash_reduce(
            lane0(p_ll),
            lane1(p_ll) ^ cr_lo,
            lane0(p_hh) ^ cr_hi,
            lane1(p_hh),
        )
    }
}

/// Binius-style: schoolbook 4 CLMUL + recursive 2-stage reduction (2 CLMUL).
/// Direct port of `aarch64::ghash_mul_binius`. `vextq_u64::<1>(zero, t)`
/// (= {0, t.lo}) becomes `_mm_slli_si128::<8>(t)` — an 8-byte left shift
/// that moves the low qword into the high lane and zeroes the low lane.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_binius(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let t0 = pmull(a.lo, b.lo);
        let t1a = pmull(a.lo, b.hi);
        let t1b = pmull(a.hi, b.lo);
        let t2 = pmull(a.hi, b.hi);
        let mut t1 = _mm_xor_si128(t1a, t1b);

        // First reduce: t1 = t1 + x^64 · t2 (mod p).
        let t2_shifted = _mm_slli_si128::<8>(t2); // {0, t2.lo}
        t1 = _mm_xor_si128(t1, t2_shifted);
        let t2_red = pmull(lane1(t2), 0x87);
        t1 = _mm_xor_si128(t1, t2_red);

        // Second reduce: t0 = t0 + x^64 · t1 (mod p).
        let t1_shifted = _mm_slli_si128::<8>(t1); // {0, t1.lo}
        let mut t0 = _mm_xor_si128(t0, t1_shifted);
        let t1_red = pmull(lane1(t1), 0x87);
        t0 = _mm_xor_si128(t0, t1_red);

        F128 {
            lo: lane0(t0),
            hi: lane1(t0),
        }
    }
}

/// Karatsuba 3 CLMUL product + binius 2-stage vector reduction (2 CLMUL,
/// only 2 lane extracts) = 5 CLMUL total, one fewer than the 6-CLMUL binius
/// schoolbook with the same fully-vector reduction shape. Field-identical.
///
/// # Safety
/// The caller must run on a CPU with the `pclmulqdq` and `sse4.1` target
/// features required by this function.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba_vec(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p0 = pmull(a.lo, b.lo);
        let p1 = pmull(a.hi, b.hi);
        let pm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);
        // cross = pm ^ p0 ^ p1 is the x^64 coefficient (binius's t1).
        let mut t1 = _mm_xor_si128(_mm_xor_si128(pm, p0), p1);
        let mut t0 = p0;

        // First reduce: t1 = t1 + x^64 · t2 (mod p), with t2 = p1.
        let t2_shifted = _mm_slli_si128::<8>(p1); // {0, p1.lo}
        t1 = _mm_xor_si128(t1, t2_shifted);
        let t2_red = pmull(lane1(p1), 0x87);
        t1 = _mm_xor_si128(t1, t2_red);

        // Second reduce: t0 = t0 + x^64 · t1 (mod p).
        let t1_shifted = _mm_slli_si128::<8>(t1); // {0, t1.lo}
        t0 = _mm_xor_si128(t0, t1_shifted);
        let t1_red = pmull(lane1(t1), 0x87);
        t0 = _mm_xor_si128(t0, t1_red);

        F128 {
            lo: lane0(t0),
            hi: lane1(t0),
        }
    }
}

/// Karatsuba 3 CLMUL — middle term depends on XOR of inputs. Port of
/// `aarch64::ghash_mul_karatsuba`.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p0 = pmull(a.lo, b.lo);
        let p1 = pmull(a.hi, b.hi);
        let pm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);

        let p0_lo = lane0(p0);
        let p0_hi = lane1(p0);
        let p1_lo = lane0(p1);
        let p1_hi = lane1(p1);
        let pm_lo = lane0(pm);
        let pm_hi = lane1(pm);

        let cross_lo = pm_lo ^ p0_lo ^ p1_lo;
        let cross_hi = pm_hi ^ p0_hi ^ p1_hi;

        ghash_reduce(p0_lo, p0_hi ^ cross_lo, p1_lo ^ cross_hi, p1_hi)
    }
}

/// Karatsuba 3 CLMUL + Barrett 2 CLMUL = 5 CLMUL total. Port of
/// `aarch64::ghash_mul_karatsuba_barrett`.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba_barrett(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let d0 = pmull(a.lo, b.lo);
        let d2 = pmull(a.hi, b.hi);
        let dm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);
        let d1 = _mm_xor_si128(_mm_xor_si128(dm, d0), d2);

        let d0_lo = lane0(d0);
        let d0_hi = lane1(d0);
        let d1_lo = lane0(d1);
        let d1_hi = lane1(d1);
        let d2_lo = lane0(d2);
        let d2_hi = lane1(d2);

        let lo_lo = d0_lo;
        let lo_hi = d0_hi ^ d1_lo;
        let hi_lo = d2_lo ^ d1_hi;
        let hi_hi = d2_hi;

        let r_hi = pmull(hi_hi, 0x87);
        let r_lo = pmull(hi_lo, 0x87);

        let r_lo_lo = lane0(r_lo);
        let r_lo_hi = lane1(r_lo);
        let r_hi_lo = lane0(r_hi);
        let r_hi_hi = lane1(r_hi);

        // hi_hi · 0x87 has degree ≤ 70, so r_hi_hi has at most 7 bits.
        let ov = r_hi_hi;
        let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);

        F128 {
            lo: lo_lo ^ r_lo_lo ^ corr,
            hi: lo_hi ^ r_lo_hi ^ r_hi_lo,
        }
    }
}

/// 256-bit unreduced schoolbook product, for XOR-accumulation then one
/// deferred `reduce()`. Port of `aarch64::ghash_mul_unreduced_neon`.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_unreduced_x86(a: F128, b: F128) -> F256Unreduced {
    // SAFETY: function carries the required target features.
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let cr_lo = lane0(cross);
        let cr_hi = lane1(cross);

        F256Unreduced {
            r0: lane0(p_ll),
            r1: lane1(p_ll) ^ cr_lo,
            r2: lane0(p_hh) ^ cr_hi,
            r3: lane1(p_hh),
        }
    }
}

// -----------------------------------------------------------------------
// PCLMULQDQ (SSE2-floor) 4x batched multiply.
//
// `PCLMULQDQ` is a 128-bit instruction: every call processes ONE 64×64
// carry-less product. To get 4 independent `F128` products out the other side
// we issue the schoolbook 4-CLMULs for product 0, then product 1, …, product
// 3 — i.e. a 4-way unroll in source order, but every product's CLMULs sit
// back-to-back so the OoO core sees 4×4 = 16 independent CLMULs in the
// instruction window. Westmere/Nehalem through Skylake have one
// `pclmulqdq`-capable pipe (port 5) and dispatch CLMULs at 1/cycle; the 4x
// unroll amortizes the per-product reduction (a 0x87 fold is one more CLMUL
// per stage) by overlapping it with the next product's wide multiply.
//
// SSE2 (not SSE4.1) is the floor: lane extracts use `_mm_srli_si128::<8>`
// instead of `_mm_extract_epi64::<1>`, so the kernel compiles on every
// `pclmulqdq`-capable core, including the Westmere/Clarkdale era where the
// field is otherwise forced to the software `portable` path.
//
// The 0x87 reduction is the same binius 2-stage fold used by
// `ghash_mul_binius` above — 4 product CLMULs + 2 reduction CLMULs per
// product, = 6 CLMUL per F128 · 4 products = 24 CLMULs in flight, field
// identical to the scalar `ghash_mul_binius` (cross-checked in the
// ntt module's tests).
// -----------------------------------------------------------------------

/// Issue one PCLMULQDQ of the low qword of `a` with the low qword of `b`,
/// returning the 128-bit product {lo, hi}. Identical to [`pmull`] but
/// callable from a kernel whose target feature is the broader
/// `pclmulqdq,sse2`.
#[inline]
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn pmull_lo_sse2(a: u64, b: u64) -> __m128i {
    let va = _mm_set_epi64x(0, a as i64);
    let vb = _mm_set_epi64x(0, b as i64);
    _mm_clmulepi64_si128::<0x00>(va, vb)
}

/// 4-way unrolled GF(2^128) multiply using only `pclmulqdq` and `sse2`.
///
/// Each of the 4 products runs the binius schoolbook + 2-stage `0x87`
/// reduction sequence (4 product CLMULs + 2 reduction CLMULs, plus the shift
/// and 0x87 fold CLMULs), so the kernel emits 24 PCLMULQDQ instructions
/// grouped by per-product dependency chains. The OoO core resolves the
/// inter-product ILP, and the per-product reduction sits behind the next
/// product's first CLMUL — net throughput ≈ 1 CLMUL/cycle, 4× the scalar
/// rate.
///
/// # Safety
/// Requires `pclmulqdq` and `sse2` (the function carries the attribute).
#[target_feature(enable = "pclmulqdq,sse2")]
pub unsafe fn ghash_mul_x4_pclmul_sse2(
    a: [F128; 4],
    b: [F128; 4],
) -> [F128; 4] {
    // SAFETY: function carries the required target features.
    unsafe {
        // Per-product: 4 product CLMULs (p0=ll, p1=hh, cross=lo·hi + hi·lo).
        // pclmul imm 0x00 = a.lo · b.lo; 0x11 = a.hi · b.hi.
        // We unroll all 4 products in this same block so the per-product
        // reductions and the next product's first CLMUL overlap in the issue
        // window.
        let p0a_ll = _mm_clmulepi64_si128::<0x00>(
            _mm_set_epi64x(a[0].hi as i64, a[0].lo as i64),
            _mm_set_epi64x(b[0].hi as i64, b[0].lo as i64),
        );
        let p0a_hh = _mm_clmulepi64_si128::<0x11>(
            _mm_set_epi64x(a[0].hi as i64, a[0].lo as i64),
            _mm_set_epi64x(b[0].hi as i64, b[0].lo as i64),
        );
        let p0a_lh = _mm_clmulepi64_si128::<0x10>(
            _mm_set_epi64x(a[0].hi as i64, a[0].lo as i64),
            _mm_set_epi64x(b[0].hi as i64, b[0].lo as i64),
        );
        let p0a_hl = _mm_clmulepi64_si128::<0x01>(
            _mm_set_epi64x(a[0].hi as i64, a[0].lo as i64),
            _mm_set_epi64x(b[0].hi as i64, b[0].lo as i64),
        );

        let p1a_ll = _mm_clmulepi64_si128::<0x00>(
            _mm_set_epi64x(a[1].hi as i64, a[1].lo as i64),
            _mm_set_epi64x(b[1].hi as i64, b[1].lo as i64),
        );
        let p1a_hh = _mm_clmulepi64_si128::<0x11>(
            _mm_set_epi64x(a[1].hi as i64, a[1].lo as i64),
            _mm_set_epi64x(b[1].hi as i64, b[1].lo as i64),
        );
        let p1a_lh = _mm_clmulepi64_si128::<0x10>(
            _mm_set_epi64x(a[1].hi as i64, a[1].lo as i64),
            _mm_set_epi64x(b[1].hi as i64, b[1].lo as i64),
        );
        let p1a_hl = _mm_clmulepi64_si128::<0x01>(
            _mm_set_epi64x(a[1].hi as i64, a[1].lo as i64),
            _mm_set_epi64x(b[1].hi as i64, b[1].lo as i64),
        );

        let p2a_ll = _mm_clmulepi64_si128::<0x00>(
            _mm_set_epi64x(a[2].hi as i64, a[2].lo as i64),
            _mm_set_epi64x(b[2].hi as i64, b[2].lo as i64),
        );
        let p2a_hh = _mm_clmulepi64_si128::<0x11>(
            _mm_set_epi64x(a[2].hi as i64, a[2].lo as i64),
            _mm_set_epi64x(b[2].hi as i64, b[2].lo as i64),
        );
        let p2a_lh = _mm_clmulepi64_si128::<0x10>(
            _mm_set_epi64x(a[2].hi as i64, a[2].lo as i64),
            _mm_set_epi64x(b[2].hi as i64, b[2].lo as i64),
        );
        let p2a_hl = _mm_clmulepi64_si128::<0x01>(
            _mm_set_epi64x(a[2].hi as i64, a[2].lo as i64),
            _mm_set_epi64x(b[2].hi as i64, b[2].lo as i64),
        );

        let p3a_ll = _mm_clmulepi64_si128::<0x00>(
            _mm_set_epi64x(a[3].hi as i64, a[3].lo as i64),
            _mm_set_epi64x(b[3].hi as i64, b[3].lo as i64),
        );
        let p3a_hh = _mm_clmulepi64_si128::<0x11>(
            _mm_set_epi64x(a[3].hi as i64, a[3].lo as i64),
            _mm_set_epi64x(b[3].hi as i64, b[3].lo as i64),
        );
        let p3a_lh = _mm_clmulepi64_si128::<0x10>(
            _mm_set_epi64x(a[3].hi as i64, a[3].lo as i64),
            _mm_set_epi64x(b[3].hi as i64, b[3].lo as i64),
        );
        let p3a_hl = _mm_clmulepi64_si128::<0x01>(
            _mm_set_epi64x(a[3].hi as i64, a[3].lo as i64),
            _mm_set_epi64x(b[3].hi as i64, b[3].lo as i64),
        );

        // Cross terms: t1 = p_lh ^ p_hl, lives at the x^64 coefficient.
        // _mm_xor_si128 is an SSE2 baseline op.
        let p0_t1 = _mm_xor_si128(p0a_lh, p0a_hl);
        let p1_t1 = _mm_xor_si128(p1a_lh, p1a_hl);
        let p2_t1 = _mm_xor_si128(p2a_lh, p2a_hl);
        let p3_t1 = _mm_xor_si128(p3a_lh, p3a_hl);

        // First reduce: t1 = t1 + x^64·t2 (mod p), where t2 = p_hh.
        // Shift t2.lo into t1.hi and fold t2.hi · 0x87 into t1.lo. The
        // shift is `_mm_slli_si128::<8>(t2) ^ zero_low_lane`; with an
        // existing zero in the low lane (we manufacture it via
        // `_mm_slli_si128::<8>(p_hh)` since the high qword of t2 shifts into
        // the high qword of the result, leaving the low qword zero).
        let zero_const = _mm_setzero_si128();

        // Product 0
        let p0_t2_shifted = _mm_slli_si128::<8>(p0a_hh);
        let p0_t1 = _mm_xor_si128(p0_t1, p0_t2_shifted);
        let p0_t2_hi = hi64(p0a_hh);
        let p0_t2_red = pmull_lo_sse2(p0_t2_hi, 0x87);
        let p0_t1 = _mm_xor_si128(p0_t1, p0_t2_red);

        // Product 1
        let p1_t2_shifted = _mm_slli_si128::<8>(p1a_hh);
        let p1_t1 = _mm_xor_si128(p1_t1, p1_t2_shifted);
        let p1_t2_hi = hi64(p1a_hh);
        let p1_t2_red = pmull_lo_sse2(p1_t2_hi, 0x87);
        let p1_t1 = _mm_xor_si128(p1_t1, p1_t2_red);

        // Product 2
        let p2_t2_shifted = _mm_slli_si128::<8>(p2a_hh);
        let p2_t1 = _mm_xor_si128(p2_t1, p2_t2_shifted);
        let p2_t2_hi = hi64(p2a_hh);
        let p2_t2_red = pmull_lo_sse2(p2_t2_hi, 0x87);
        let p2_t1 = _mm_xor_si128(p2_t1, p2_t2_red);

        // Product 3
        let p3_t2_shifted = _mm_slli_si128::<8>(p3a_hh);
        let p3_t1 = _mm_xor_si128(p3_t1, p3_t2_shifted);
        let p3_t2_hi = hi64(p3a_hh);
        let p3_t2_red = pmull_lo_sse2(p3_t2_hi, 0x87);
        let p3_t1 = _mm_xor_si128(p3_t1, p3_t2_red);

        // Silence "value assigned to p0a_ll is never read" — the LL product
        // is the lo limb of t0; the 2nd reduce consumes it.
        let _ = zero_const;
        let _ = p0a_ll;
        let _ = p1a_ll;
        let _ = p2a_ll;
        let _ = p3a_ll;

        // Second reduce: t0 = t0 + x^64·t1 (mod p).
        let p0_t1_shifted = _mm_slli_si128::<8>(p0_t1);
        let p0_t0 = _mm_xor_si128(p0a_ll, p0_t1_shifted);
        let p0_t1_hi = hi64(p0_t1);
        let p0_t1_red = pmull_lo_sse2(p0_t1_hi, 0x87);
        let p0_t0 = _mm_xor_si128(p0_t0, p0_t1_red);

        let p1_t1_shifted = _mm_slli_si128::<8>(p1_t1);
        let p1_t0 = _mm_xor_si128(p1a_ll, p1_t1_shifted);
        let p1_t1_hi = hi64(p1_t1);
        let p1_t1_red = pmull_lo_sse2(p1_t1_hi, 0x87);
        let p1_t0 = _mm_xor_si128(p1_t0, p1_t1_red);

        let p2_t1_shifted = _mm_slli_si128::<8>(p2_t1);
        let p2_t0 = _mm_xor_si128(p2a_ll, p2_t1_shifted);
        let p2_t1_hi = hi64(p2_t1);
        let p2_t1_red = pmull_lo_sse2(p2_t1_hi, 0x87);
        let p2_t0 = _mm_xor_si128(p2_t0, p2_t1_red);

        let p3_t1_shifted = _mm_slli_si128::<8>(p3_t1);
        let p3_t0 = _mm_xor_si128(p3a_ll, p3_t1_shifted);
        let p3_t1_hi = hi64(p3_t1);
        let p3_t1_red = pmull_lo_sse2(p3_t1_hi, 0x87);
        let p3_t0 = _mm_xor_si128(p3_t0, p3_t1_red);

        [
            F128 {
                lo: _mm_cvtsi128_si64(p0_t0) as u64,
                hi: hi64(p0_t0),
            },
            F128 {
                lo: _mm_cvtsi128_si64(p1_t0) as u64,
                hi: hi64(p1_t0),
            },
            F128 {
                lo: _mm_cvtsi128_si64(p2_t0) as u64,
                hi: hi64(p2_t0),
            },
            F128 {
                lo: _mm_cvtsi128_si64(p3_t0) as u64,
                hi: hi64(p3_t0),
            },
        ]
    }
}

/// 4-way unrolled GF(2^128) multiply, where the *right-hand* operand's
/// high limb is zero in every lane. This is the structural precondition for
/// the additive-NTT butterfly's `r · (a + b)` step: the `a + b` term has its
/// high limb cancelled to zero in the rank-1 update direction.
///
/// With `b.hi == 0` in every lane, the cross term `a.hi·b.lo ^ a.lo·b.hi`
/// collapses to `a.hi·b.lo`, and `a.hi·b.hi = 0` so the 2-stage reduction
/// shortens to a single `0x87` fold. Net 3 CLMUL per F128, 4 F128 per call
/// = 12 CLMULs — half the cost of [`ghash_mul_x4_pclmul_sse2`].
///
/// # Safety
/// Requires `pclmulqdq` and `sse2`. The caller must ensure every `b[i].hi == 0`.
#[target_feature(enable = "pclmulqdq,sse2")]
pub unsafe fn ghash_mul_x4_pclmul_sse2_low_rhs(
    a: [F128; 4],
    b: [F128; 4],
) -> [F128; 4] {
    debug_assert!(b.iter().all(|v| v.hi == 0), "b.hi must be 0 in every lane");
    // SAFETY: function carries the required target features.
    unsafe {
        // Only two product CLMULs per F128: p0 = a.lo · b.lo, p1 = a.hi · b.lo.
        let p0a_lo = _mm_clmulepi64_si128::<0x00>(
            _mm_set_epi64x(a[0].hi as i64, a[0].lo as i64),
            _mm_set_epi64x(0, b[0].lo as i64),
        );
        let p0a_hi = _mm_clmulepi64_si128::<0x01>(
            _mm_set_epi64x(a[0].hi as i64, a[0].lo as i64),
            _mm_set_epi64x(0, b[0].lo as i64),
        );

        let p1a_lo = _mm_clmulepi64_si128::<0x00>(
            _mm_set_epi64x(a[1].hi as i64, a[1].lo as i64),
            _mm_set_epi64x(0, b[1].lo as i64),
        );
        let p1a_hi = _mm_clmulepi64_si128::<0x01>(
            _mm_set_epi64x(a[1].hi as i64, a[1].lo as i64),
            _mm_set_epi64x(0, b[1].lo as i64),
        );

        let p2a_lo = _mm_clmulepi64_si128::<0x00>(
            _mm_set_epi64x(a[2].hi as i64, a[2].lo as i64),
            _mm_set_epi64x(0, b[2].lo as i64),
        );
        let p2a_hi = _mm_clmulepi64_si128::<0x01>(
            _mm_set_epi64x(a[2].hi as i64, a[2].lo as i64),
            _mm_set_epi64x(0, b[2].lo as i64),
        );

        let p3a_lo = _mm_clmulepi64_si128::<0x00>(
            _mm_set_epi64x(a[3].hi as i64, a[3].lo as i64),
            _mm_set_epi64x(0, b[3].lo as i64),
        );
        let p3a_hi = _mm_clmulepi64_si128::<0x01>(
            _mm_set_epi64x(a[3].hi as i64, a[3].lo as i64),
            _mm_set_epi64x(0, b[3].lo as i64),
        );

        // t1 = a.hi · b.lo (single product — b.hi = 0 kills a.lo · b.hi).
        // Single-stage reduction: t0 = t0 + x^64·t1 mod p.
        let p0_t1_shifted = _mm_slli_si128::<8>(p0a_hi);
        let p0_t0 = _mm_xor_si128(p0a_lo, p0_t1_shifted);
        let p0_t1_hi = hi64(p0a_hi);
        let p0_t1_red = pmull_lo_sse2(p0_t1_hi, 0x87);
        let p0_t0 = _mm_xor_si128(p0_t0, p0_t1_red);

        let p1_t1_shifted = _mm_slli_si128::<8>(p1a_hi);
        let p1_t0 = _mm_xor_si128(p1a_lo, p1_t1_shifted);
        let p1_t1_hi = hi64(p1a_hi);
        let p1_t1_red = pmull_lo_sse2(p1_t1_hi, 0x87);
        let p1_t0 = _mm_xor_si128(p1_t0, p1_t1_red);

        let p2_t1_shifted = _mm_slli_si128::<8>(p2a_hi);
        let p2_t0 = _mm_xor_si128(p2a_lo, p2_t1_shifted);
        let p2_t1_hi = hi64(p2a_hi);
        let p2_t1_red = pmull_lo_sse2(p2_t1_hi, 0x87);
        let p2_t0 = _mm_xor_si128(p2_t0, p2_t1_red);

        let p3_t1_shifted = _mm_slli_si128::<8>(p3a_hi);
        let p3_t0 = _mm_xor_si128(p3a_lo, p3_t1_shifted);
        let p3_t1_hi = hi64(p3a_hi);
        let p3_t1_red = pmull_lo_sse2(p3_t1_hi, 0x87);
        let p3_t0 = _mm_xor_si128(p3_t0, p3_t1_red);

        [
            F128 {
                lo: _mm_cvtsi128_si64(p0_t0) as u64,
                hi: hi64(p0_t0),
            },
            F128 {
                lo: _mm_cvtsi128_si64(p1_t0) as u64,
                hi: hi64(p1_t0),
            },
            F128 {
                lo: _mm_cvtsi128_si64(p2_t0) as u64,
                hi: hi64(p2_t0),
            },
            F128 {
                lo: _mm_cvtsi128_si64(p3_t0) as u64,
                hi: hi64(p3_t0),
            },
        ]
    }
}

/// Software prefetch hint for the F128 data feeding
/// [`ghash_mul_x4_pclmul_sse2`]. SSE2 baseline (`prefetcht0`).
///
/// # Safety
/// Requires `sse2`; the caller must ensure `p` is a valid readable pointer.
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn prefetch_f128(p: *const F128) {
    // _mm_prefetch is a `prefetcht0` (L1) under SSE2 — keeps the next
    // 4-element block warm in cache while the current block is multiplied.
    _mm_prefetch(p as *const i8, _MM_HINT_T0);
}

/// CPUID gate for the 4x batched PCLMULQDQ kernel. Returns `true` when
/// the running CPU advertises both `pclmulqdq` and `sse2`. Used by the
/// runtime-dispatch wrapper that picks between the 4x PCLMULQDQ kernel
/// and the per-element fallback. The static-feature build (`-C
/// target-cpu=native`, which is the harness default) folds this to a
/// constant `true`.
#[inline]
pub fn cpu_has_pclmulqdq_sse2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("pclmulqdq")
            && std::is_x86_feature_detected!("sse2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

// -----------------------------------------------------------------------
// AVX-512 + VPCLMULQDQ: 4 independent GF(2^128) multiplies per instruction.
//
// Lane-parallel port of `ghash_mul_binius` above — same 4 product CLMULs +
// two-stage `0x87` reduction, applied independently in each 128-bit lane of
// a `__m512i`. A `__m512i` holds 4 contiguous `F128` (lane i = {lo_i, hi_i});
// since `F128` is `repr(C, align(16))` little-endian, 4 elements load
// directly with `_mm512_loadu_si512` — no shuffles. The reduction is the
// same field element as the scalar `ghash_mul_binius` (cross-checked in the
// ntt module's tests), reached by the identical operation sequence.
// -----------------------------------------------------------------------

/// Per-lane reduction-poly low word: each 128-bit lane = {lo: 0x87, hi: 0}.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn ghash_poly_x4() -> __m512i {
    _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87)
}

/// Per-128-bit-lane reduce: returns `t0 + x^64 · t1` (mod p) in each lane.
/// Mirrors one stage of `ghash_mul_binius`'s recursive reduction:
/// `t0 ^= (t1 << 64)` then `t0 ^= t1.hi · 0x87` (clmul imm `0x01` = hi qword
/// of `t1` × lo qword of `poly`).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn gf2_128_reduce_x4(mut t0: __m512i, t1: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe {
        let poly = ghash_poly_x4();
        t0 = _mm512_xor_si512(t0, _mm512_bslli_epi128::<8>(t1));
        t0 = _mm512_xor_si512(t0, _mm512_clmulepi64_epi128::<0x01>(t1, poly));
        t0
    }
}

/// Reduce four lanes of an XOR-accumulated unreduced product triple
/// `(lo = Σ x.lo·y.lo, mid = Σ (x.hi·y.lo ^ x.lo·y.hi), hi = Σ x.hi·y.hi)`
/// — the same two-step fold [`ghash_mul_x4`] applies to a single product.
/// Reduction is F₂-linear, so this equals the XOR of the individually
/// reduced products, lane by lane (deferred reduction).
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` (statically satisfied by the
/// cfg gate and target-feature attribute).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_reduce_acc_x4(lo: __m512i, mid: __m512i, hi: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe {
        let t1 = gf2_128_reduce_x4(mid, hi);
        gf2_128_reduce_x4(lo, t1)
    }
}

/// 4 independent GF(2^128) products. `x` and `y` each hold 4 contiguous
/// `F128`; the result holds the 4 reduced products. Field-identical to
/// applying `ghash_mul_binius` to each lane.
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` are available (statically
/// satisfied by the cfg gate and target-feature attribute).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4(x: __m512i, y: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe {
        // Cross terms: x.hi·y.lo (imm 0x01) ^ x.lo·y.hi (imm 0x10), at x^64.
        let t1a = _mm512_clmulepi64_epi128::<0x01>(x, y);
        let t1b = _mm512_clmulepi64_epi128::<0x10>(x, y);
        let mut t1 = _mm512_xor_si512(t1a, t1b);
        // High product x.hi·y.hi (imm 0x11), folded into the cross.
        let t2 = _mm512_clmulepi64_epi128::<0x11>(x, y);
        t1 = gf2_128_reduce_x4(t1, t2);
        // Low product x.lo·y.lo (imm 0x00), then fold t1 down to the result.
        let t0 = _mm512_clmulepi64_epi128::<0x00>(x, y);
        gf2_128_reduce_x4(t0, t1)
    }
}

/// [`ghash_mul_x4`] specialized for a multiplier `x` whose high limb is zero
/// in **every** lane.
///
/// `x.hi = 0` kills the `0x01` (`x.hi·y.lo`) and `0x11` (`x.hi·y.hi`)
/// products, and the first reduction then folds a zero high operand, so it
/// disappears with them: **3 CLMUL instead of 6**, same reduced result.
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` (statically satisfied by the
/// cfg gate) and that every 128-bit lane of `x` has a zero high qword.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4_low_lhs(x: __m512i, y: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq and the zero-high-limb
    // precondition.
    unsafe {
        // Only the cross term x.lo·y.hi survives at x^64; x.hi·y.hi is zero,
        // so `gf2_128_reduce_x4(t1, 0) == t1` and is skipped entirely.
        let t1 = _mm512_clmulepi64_epi128::<0x10>(x, y);
        let t0 = _mm512_clmulepi64_epi128::<0x00>(x, y);
        gf2_128_reduce_x4(t0, t1)
    }
}

// -----------------------------------------------------------------------
// Split-twiddle ("x^64-companion") product: 5 CLMUL instead of 6.
//
// The additive-NTT butterflies all multiply a *variable* value `v` by a
// twiddle `t` that is CONSTANT for the whole row set, so any per-twiddle
// preprocessing is free. Write `v = v_lo + x^64·v_hi` (its two 64-bit limbs).
// Because reduction mod p is a ring homomorphism,
//
//     t·v  ≡  t·v_lo  +  (t·x^64 mod p)·v_hi   (mod p),
//
// i.e. with the companion constant `u = t·x^64 mod p` precomputed the product
// is a sum of two 128×64 products — degree ≤ 190, so it occupies only THREE
// 64-bit limbs instead of four. Schoolbook cost is unchanged (4 CLMUL), but
// the tail is one limb shorter, so folding it down needs a single `0x87`
// CLMUL rather than the incumbent's two-stage recursive reduction:
//
//     incumbent `ghash_mul_x4`: 4 product + 2 reduction CLMUL, 5 XOR, 2 VPSLLDQ
//     split form              : 4 product + 1 reduction CLMUL, 4 XOR, 1 VPSLLDQ
//
// On Sapphire Rapids VPCLMULQDQ-zmm and VPSLLDQ-zmm both issue only on port 5,
// so this drops the port-5 op count per 4-lane multiply from 8 to 6 (-25%);
// total zmm uops go 13 → 10. The result is the same field element, so proof
// bytes are unchanged.
// -----------------------------------------------------------------------

/// `t · x^64 mod p`, independently in each 128-bit lane — the companion
/// constant consumed by [`ghash_mul_x4_split`].
///
/// This is exactly one stage of the recursive reduction with a zero low half:
/// `0 + x^64·t mod p`.
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` (statically satisfied by the
/// cfg gate and target-feature attribute).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_shift64_x4(t: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe { gf2_128_reduce_x4(_mm512_setzero_si512(), t) }
}

/// 4 independent GF(2^128) products `v[i]·t` for a twiddle supplied in split
/// form: `t` and `t_x64 = t·x^64 mod p` (see [`ghash_shift64_x4`]). Both
/// twiddle operands are normally lane-broadcast loop constants.
///
/// Field-identical to [`ghash_mul_x4`]`(t, v)` — 5 CLMUL instead of 6.
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` (statically satisfied by the
/// cfg gate and target-feature attribute) and that `t_x64` really is
/// `t·x^64 mod p` in every lane.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4_split(v: __m512i, t: __m512i, t_x64: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq and the companion contract.
    unsafe {
        // Limb 0..1 of the 192-bit product: v.lo·t.lo ⊕ v.hi·t_x64.lo.
        let lo = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x00>(v, t),
            _mm512_clmulepi64_epi128::<0x01>(v, t_x64),
        );
        // Limb 1..2, weighted x^64: v.lo·t.hi ⊕ v.hi·t_x64.hi.
        let hi = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x10>(v, t),
            _mm512_clmulepi64_epi128::<0x11>(v, t_x64),
        );
        // One fold: lo + x^64·hi (mod p). The top limb is 64 bits wide, so the
        // single `0x87` CLMUL finishes it (degree ≤ 70 < 128).
        gf2_128_reduce_x4(lo, hi)
    }
}

// -----------------------------------------------------------------------
// Deferred-reduction 4-lane accumulator (port of binius `WideGhashProduct`,
// 4 lanes wide). Widen each product with 4 CLMULs but DON'T reduce; XOR many
// into the accumulator; reduce once at the end. Per 128-bit lane the
// unreduced product is `lo + mid·x^64 + hi·x^128` with `lo = x.lo·y.lo`,
// `hi = x.hi·y.hi`, `mid = x.hi·y.lo ⊕ x.lo·y.hi` — the same limb split the
// scalar `mul_unreduced`/`F256Unreduced` uses. `fold()` horizontally XORs
// the 4 lanes into one scalar `F256Unreduced`; since `ghash_reduce` is
// F2-linear, fold-then-reduce equals XOR-of-per-lane-reduce, which equals
// the scalar `Σ mul_unreduced` then `reduce`.
// -----------------------------------------------------------------------

/// Load 4 contiguous `F128` (lane i = `p[i]`) into a `__m512i`.
///
/// # Safety
/// `p` must point to 4 readable `F128`; `avx512f` available (cfg-gated).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_loadu(p: *const F128) -> __m512i {
    // SAFETY: caller guarantees 4 readable F128 at p.
    unsafe { _mm512_loadu_si512(p as *const __m512i) }
}

/// Pack 4 `F128` scalars into a `__m512i` (lane 0 = `a`, …, lane 3 = `d`).
///
/// # Safety
/// Requires `avx512f`, as guaranteed by the cfg gate.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_set(a: F128, b: F128, c: F128, d: F128) -> __m512i {
    // Pure register assembly; avx512f cfg-gated.
    _mm512_set_epi64(
        d.hi as i64,
        d.lo as i64,
        c.hi as i64,
        c.lo as i64,
        b.hi as i64,
        b.lo as i64,
        a.hi as i64,
        a.lo as i64,
    )
}

/// XOR the four 128-bit lanes of `v` into a single `__m128i`.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xor4_lanes(v: __m512i) -> __m128i {
    // Register-only lane extracts + XOR; avx512f cfg-gated.
    let l0 = _mm512_extracti32x4_epi32::<0>(v);
    let l1 = _mm512_extracti32x4_epi32::<1>(v);
    let l2 = _mm512_extracti32x4_epi32::<2>(v);
    let l3 = _mm512_extracti32x4_epi32::<3>(v);
    _mm_xor_si128(_mm_xor_si128(l0, l1), _mm_xor_si128(l2, l3))
}

/// 4-lane unreduced GF(2^128) product accumulator (deferred reduction).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[derive(Clone, Copy)]
pub struct WideGhashX4 {
    lo: __m512i,
    hi: __m512i,
    mid: __m512i,
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
impl WideGhashX4 {
    /// Empty accumulator.
    ///
    /// # Safety
    /// `avx512f` available (cfg-gated).
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn zero() -> Self {
        let z = _mm512_setzero_si512();
        Self {
            lo: z,
            hi: z,
            mid: z,
        }
    }

    /// XOR-accumulate the 4 unreduced products `x[i]·y[i]` into self.
    ///
    /// # Safety
    /// `avx512f` + `vpclmulqdq` available (cfg-gated).
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn mul_acc(&mut self, x: __m512i, y: __m512i) {
        // Register-only widen (4 CLMULs) + XOR-accumulate; cfg-gated.
        self.lo = _mm512_xor_si512(self.lo, _mm512_clmulepi64_epi128::<0x00>(x, y));
        self.hi = _mm512_xor_si512(self.hi, _mm512_clmulepi64_epi128::<0x11>(x, y));
        let m = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x01>(x, y),
            _mm512_clmulepi64_epi128::<0x10>(x, y),
        );
        self.mid = _mm512_xor_si512(self.mid, m);
    }

    /// Reduce each of the 4 lanes independently (no horizontal fold): the
    /// result holds the 4 reduced lane sums, field-identical to reducing every
    /// accumulated product separately and XORing per lane.
    ///
    /// # Safety
    /// `avx512f` + `vpclmulqdq` available (cfg-gated).
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn reduce_lanes(self) -> __m512i {
        // SAFETY: caller carries avx512f+vpclmulqdq.
        unsafe { ghash_reduce_acc_x4(self.lo, self.mid, self.hi) }
    }

    /// Horizontally XOR the 4 lanes and assemble a scalar `F256Unreduced`
    /// (NOT yet reduced, so it can be XORed with a scalar tail accumulator).
    ///
    /// # Safety
    /// `avx512f` + `sse4.1` available (cfg-gated + attr).
    #[inline]
    #[target_feature(enable = "avx512f,sse4.1")]
    pub unsafe fn fold(self) -> F256Unreduced {
        // SAFETY: caller carries avx512f+sse4.1.
        unsafe {
            let lo = xor4_lanes(self.lo);
            let hi = xor4_lanes(self.hi);
            let mid = xor4_lanes(self.mid);
            let lo_lo = _mm_extract_epi64::<0>(lo) as u64;
            let lo_hi = _mm_extract_epi64::<1>(lo) as u64;
            let hi_lo = _mm_extract_epi64::<0>(hi) as u64;
            let hi_hi = _mm_extract_epi64::<1>(hi) as u64;
            let mid_lo = _mm_extract_epi64::<0>(mid) as u64;
            let mid_hi = _mm_extract_epi64::<1>(mid) as u64;
            F256Unreduced {
                r0: lo_lo,
                r1: lo_hi ^ mid_lo,
                r2: hi_lo ^ mid_hi,
                r3: hi_hi,
            }
        }
    }
}
