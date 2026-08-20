use crate::field::F128;

/// Four-lane pair fold using AVX-512 lane deinterleaving and VPCLMULQDQ.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        // u64-element selectors: even 128-bit lanes -> {0,1,4,5,8,9,12,13},
        // odd -> {2,3,6,7,10,11,14,15} over concat(lo, hi).
        let idx_even = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let idx_odd = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let lanes = dst.len() & !3;
        let mut t = 0;
        while t < lanes {
            let s = 2 * (base + t);
            let lo = _mm512_loadu_si512(src.as_ptr().add(s) as *const __m512i);
            let hi = _mm512_loadu_si512(src.as_ptr().add(s + 4) as *const __m512i);
            let even = _mm512_permutex2var_epi64(lo, idx_even, hi);
            let odd = _mm512_permutex2var_epi64(lo, idx_odd, hi);
            let new = _mm512_xor_si512(even, ghash_mul_x4(r_bcast, _mm512_xor_si512(even, odd)));
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, new);
            t += 4;
        }
        portable_tail(src, base, dst, r, t);
    }
}

/// Four-lane `dst += scale * addend` for the lazy-OOD correction.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`; slices have equal length.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn add_scaled(dst: &mut [F128], addend: &[F128], scale: F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    debug_assert_eq!(dst.len(), addend.len());
    // SAFETY: caller supplies target features and equal slice lengths.
    unsafe {
        let scale_x4 =
            _mm512_broadcast_i32x4(_mm_set_epi64x(scale.hi as i64, scale.lo as i64));
        let lanes = dst.len() & !3;
        let mut i = 0usize;
        while i < lanes {
            let current = _mm512_loadu_si512(dst.as_ptr().add(i) as *const __m512i);
            let extra = _mm512_loadu_si512(addend.as_ptr().add(i) as *const __m512i);
            let corrected = _mm512_xor_si512(current, ghash_mul_x4(scale_x4, extra));
            _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut __m512i, corrected);
            i += 4;
        }
        while i < dst.len() {
            dst[i] += scale * addend[i];
            i += 1;
        }
    }
}

#[inline]
fn portable_tail(src: &[F128], base: usize, dst: &mut [F128], r: F128, mut t: usize) {
    // Char-2 one-mul tail (SIMD body already uses even + r*(even+odd)).
    while t < dst.len() {
        let s = 2 * (base + t);
        let even = src[s];
        dst[t] = even + r * (even + src[s + 1]);
        t += 1;
    }
}

/// Nested pair-fold of 4-tuples into `dst`, keeping the r0 mid in zmm.
///
/// For each slot `t`:
///   low  = a0 + r0·(a0+a1)
///   high = a2 + r0·(a2+a3)
///   dst[t] = low + r1·(low+high)
///
/// Four slots (16 source F128) per iteration. Same even/odd pairing and
/// `ghash_mul_x4(r, even XOR odd)` body as [`fold_pairs`], applied twice
/// in registers. Stores `dst` only — no mid buffer.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`. `src.len() == 4 * dst.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold4_nested(src: &[F128], dst: &mut [F128], r0: F128, r1: F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let r0_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r0.hi as i64, r0.lo as i64));
        let r1_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r1.hi as i64, r1.lo as i64));
        // Same even/odd 128-bit-lane selectors as `fold_pairs`.
        let idx_even = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let idx_odd = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let lanes = dst.len() & !3;
        let mut t = 0;
        while t < lanes {
            let s = 4 * t;
            let v0 = _mm512_loadu_si512(src.as_ptr().add(s) as *const __m512i);
            let v1 = _mm512_loadu_si512(src.as_ptr().add(s + 4) as *const __m512i);
            let v2 = _mm512_loadu_si512(src.as_ptr().add(s + 8) as *const __m512i);
            let v3 = _mm512_loadu_si512(src.as_ptr().add(s + 12) as *const __m512i);

            // Layer r0: adjacent pairs → [low0, high0, low1, high1] / [low2, …].
            let even01 = _mm512_permutex2var_epi64(v0, idx_even, v1);
            let odd01 = _mm512_permutex2var_epi64(v0, idx_odd, v1);
            let mid01 = _mm512_xor_si512(
                even01,
                ghash_mul_x4(r0_bcast, _mm512_xor_si512(even01, odd01)),
            );
            let even23 = _mm512_permutex2var_epi64(v2, idx_even, v3);
            let odd23 = _mm512_permutex2var_epi64(v2, idx_odd, v3);
            let mid23 = _mm512_xor_si512(
                even23,
                ghash_mul_x4(r0_bcast, _mm512_xor_si512(even23, odd23)),
            );

            // Layer r1: (low, high) pairs → [out0, out1, out2, out3].
            let low = _mm512_permutex2var_epi64(mid01, idx_even, mid23);
            let high = _mm512_permutex2var_epi64(mid01, idx_odd, mid23);
            let out = _mm512_xor_si512(low, ghash_mul_x4(r1_bcast, _mm512_xor_si512(low, high)));
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, out);
            t += 4;
        }
        while t < dst.len() {
            let a0 = src[4 * t];
            let a1 = src[4 * t + 1];
            let a2 = src[4 * t + 2];
            let a3 = src[4 * t + 3];
            let low = a0 + r0 * (a0 + a1);
            let high = a2 + r0 * (a2 + a3);
            dst[t] = low + r1 * (low + high);
            t += 1;
        }
    }
}

/// Sixteen-bank weighted fold with deferred reduction, four output slots per
/// pass: `dst[t] = Σ_{b<16} w[b] · src[16t + b]`.
///
/// The 16 slot-major loads of a 4-slot block are transposed (128-bit lanes)
/// into bank-major vectors, each multiplied by its broadcast weight into ONE
/// four-lane unreduced accumulator (`WideGhashX4::mul_acc`, 4 CLMUL per
/// vector), and reduced once per lane at the end — 18 vector CLMULs per four
/// outputs against 36 for the two nested pair-fold passes it replaces.
/// Field-identical (reduction is F₂-linear).
///
/// # Safety
/// Caller guarantees `avx512f` + `vpclmulqdq` and `src.len() == 16 * dst.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold16_banked(src: &[F128], dst: &mut [F128], w: &[F128; 16]) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;
    debug_assert_eq!(src.len(), 16 * dst.len());
    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let wb: [__m512i; 16] = core::array::from_fn(|b| {
            _mm512_broadcast_i32x4(_mm_set_epi64x(w[b].hi as i64, w[b].lo as i64))
        });
        // 4×4 transpose of 128-bit lanes: stage-1 index vectors interleave
        // lanes {0,1} / {2,3} of two inputs; stage 2 gathers lanes {0,1} /
        // {2,3} of the two stage-1 results.
        let s1_lo = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
        let s1_hi = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
        let s2_lo = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
        let s2_hi = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
        let quads = dst.len() & !3;
        let mut t = 0usize;
        while t < quads {
            let mut acc = WideGhashX4::zero();
            for g in 0..4 {
                // v_s = banks 4g..4g+3 of slot t+s.
                let base = 16 * t + 4 * g;
                let a0 = _mm512_loadu_si512(src.as_ptr().add(base) as *const __m512i);
                let a1 = _mm512_loadu_si512(src.as_ptr().add(base + 16) as *const __m512i);
                let a2 = _mm512_loadu_si512(src.as_ptr().add(base + 32) as *const __m512i);
                let a3 = _mm512_loadu_si512(src.as_ptr().add(base + 48) as *const __m512i);
                let t0 = _mm512_permutex2var_epi64(a0, s1_lo, a1); // [a0.L0 a1.L0 a0.L1 a1.L1]
                let t1 = _mm512_permutex2var_epi64(a0, s1_hi, a1); // [a0.L2 a1.L2 a0.L3 a1.L3]
                let t2 = _mm512_permutex2var_epi64(a2, s1_lo, a3);
                let t3 = _mm512_permutex2var_epi64(a2, s1_hi, a3);
                let u0 = _mm512_permutex2var_epi64(t0, s2_lo, t2); // bank 4g+0 over slots 0..4
                let u1 = _mm512_permutex2var_epi64(t0, s2_hi, t2); // bank 4g+1
                let u2 = _mm512_permutex2var_epi64(t1, s2_lo, t3); // bank 4g+2
                let u3 = _mm512_permutex2var_epi64(t1, s2_hi, t3); // bank 4g+3
                acc.mul_acc(u0, wb[4 * g]);
                acc.mul_acc(u1, wb[4 * g + 1]);
                acc.mul_acc(u2, wb[4 * g + 2]);
                acc.mul_acc(u3, wb[4 * g + 3]);
            }
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, acc.reduce_lanes());
            t += 4;
        }
        while t < dst.len() {
            let mut v = F128::ZERO;
            for b in 0..16 {
                v += w[b] * src[16 * t + b];
            }
            dst[t] = v;
            t += 1;
        }
    }
}

/// Sixty-four-bank weighted fold with one deferred reduction, four final
/// output slots per pass. This is the DirectFold8 analogue of
/// [`fold16_banked`]: 64 bank products accumulate into one `WideGhashX4`, so
/// each lane is reduced once rather than reducing four 16-bank mids and then
/// performing a reduced 4:1 fold.
///
/// # Safety
/// Caller guarantees `avx512f` + `vpclmulqdq` and `src.len() == 64 * dst.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold64_banked(src: &[F128], dst: &mut [F128], w: &[F128; 64]) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;
    debug_assert_eq!(src.len(), 64 * dst.len());
    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let s1_lo = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
        let s1_hi = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
        let s2_lo = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
        let s2_hi = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
        // Four independent four-slot accumulators break the 64-product
        // dependency chain and amortize every cached weight-vector load over
        // 16 final outputs. Twelve ZMM accumulator registers plus four live
        // weights leave room for the sequential transpose temporaries on SPR.
        let tiles = dst.len() & !15;
        let mut t = 0usize;
        while t < tiles {
            let mut acc0 = WideGhashX4::zero();
            let mut acc1 = WideGhashX4::zero();
            let mut acc2 = WideGhashX4::zero();
            let mut acc3 = WideGhashX4::zero();
            for g in 0..16 {
                // Broadcast four scalar weights once, then reuse them across
                // all four quads. Loading a pre-broadcast 4 KiB table would
                // spend 128 MiB of L1 traffic at the ranked shape; these four
                // 16-byte loads spend 32 MiB and need no table initialization.
                let w0 = _mm512_broadcast_i32x4(_mm_loadu_si128(
                    w.as_ptr().add(4 * g) as *const __m128i
                ));
                let w1 = _mm512_broadcast_i32x4(_mm_loadu_si128(
                    w.as_ptr().add(4 * g + 1) as *const __m128i
                ));
                let w2 = _mm512_broadcast_i32x4(_mm_loadu_si128(
                    w.as_ptr().add(4 * g + 2) as *const __m128i
                ));
                let w3 = _mm512_broadcast_i32x4(_mm_loadu_si128(
                    w.as_ptr().add(4 * g + 3) as *const __m128i
                ));
                macro_rules! accumulate_quad {
                    ($quad:expr, $acc:ident) => {{
                        let base = 64 * t + 256 * $quad + 4 * g;
                        let a0 =
                            _mm512_loadu_si512(src.as_ptr().add(base) as *const __m512i);
                        let a1 =
                            _mm512_loadu_si512(src.as_ptr().add(base + 64) as *const __m512i);
                        let a2 =
                            _mm512_loadu_si512(src.as_ptr().add(base + 128) as *const __m512i);
                        let a3 =
                            _mm512_loadu_si512(src.as_ptr().add(base + 192) as *const __m512i);
                        let p0 = _mm512_permutex2var_epi64(a0, s1_lo, a1);
                        let p1 = _mm512_permutex2var_epi64(a0, s1_hi, a1);
                        let p2 = _mm512_permutex2var_epi64(a2, s1_lo, a3);
                        let p3 = _mm512_permutex2var_epi64(a2, s1_hi, a3);
                        $acc.mul_acc(_mm512_permutex2var_epi64(p0, s2_lo, p2), w0);
                        $acc.mul_acc(_mm512_permutex2var_epi64(p0, s2_hi, p2), w1);
                        $acc.mul_acc(_mm512_permutex2var_epi64(p1, s2_lo, p3), w2);
                        $acc.mul_acc(_mm512_permutex2var_epi64(p1, s2_hi, p3), w3);
                    }};
                }
                accumulate_quad!(0, acc0);
                accumulate_quad!(1, acc1);
                accumulate_quad!(2, acc2);
                accumulate_quad!(3, acc3);
            }
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, acc0.reduce_lanes());
            _mm512_storeu_si512(
                dst.as_mut_ptr().add(t + 4) as *mut __m512i,
                acc1.reduce_lanes(),
            );
            _mm512_storeu_si512(
                dst.as_mut_ptr().add(t + 8) as *mut __m512i,
                acc2.reduce_lanes(),
            );
            _mm512_storeu_si512(
                dst.as_mut_ptr().add(t + 12) as *mut __m512i,
                acc3.reduce_lanes(),
            );
            t += 16;
        }
        while t < dst.len() {
            let mut v = F128::ZERO;
            for bank in 0..64 {
                v += w[bank] * src[64 * t + bank];
            }
            dst[t] = v;
            t += 1;
        }
    }
}
