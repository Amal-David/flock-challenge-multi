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
