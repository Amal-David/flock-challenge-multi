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

/// Fused pair-fold + sumcheck message accumulation for x86 AVX-512.
///
/// Folds `f_src[2*(f_base + t)..][..2]` and `b_src[2*(b_base + t)..][..2]`
/// at `r` (8 source F128 per iteration → 4 folded F128 per array), stores
/// the folded results to `f_dst`/`b_dst`, and accumulates the sumcheck
/// message terms `(u_0, u_2)` from the register-resident folded values —
/// avoiding the store-then-reload cycle of the two-pass `fold_pairs` +
/// `msg_reduce` approach. Bit-identical to the unfused sequence.
///
/// Message pairs are (k, k+1) for k = 0, 2, 4, …:
///   u_0 = Σ f_dst[k] · b_dst[k]
///   u_2 = Σ (f_dst[k] + f_dst[k+1]) · (b_dst[k] + b_dst[k+1])
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`. `f_dst.len() == b_dst.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_pairs_and_msg(
    f_src: &[F128],
    f_base: usize,
    f_dst: &mut [F128],
    b_src: &[F128],
    b_base: usize,
    b_dst: &mut [F128],
    r: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::{WideGhashX4, ghash_mul_x4};
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees target features and slice bounds.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let idx_even = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let idx_odd = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let perm_swap = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        // Process 8 F128 (4 pairs) per iteration — same granularity as
        // `msg_reduce_avx512` so the WideGhashX4 lanes stay saturated.
        let lanes = f_dst.len() & !7;
        let mut t = 0;
        let mut u0_acc = WideGhashX4::zero();
        let mut u2_acc = WideGhashX4::zero();

        while t < lanes {
            // --- f fold: 8 source → 4 folded (2 ZMM loads → 2 ZMM stores) ---
            let fs = 2 * (f_base + t);
            let f0 = _mm512_loadu_si512(f_src.as_ptr().add(fs) as *const __m512i);
            let f1 = _mm512_loadu_si512(f_src.as_ptr().add(fs + 4) as *const __m512i);
            let f2 = _mm512_loadu_si512(f_src.as_ptr().add(fs + 8) as *const __m512i);
            let f3 = _mm512_loadu_si512(f_src.as_ptr().add(fs + 12) as *const __m512i);

            let f_even01 = _mm512_permutex2var_epi64(f0, idx_even, f1);
            let f_odd01 = _mm512_permutex2var_epi64(f0, idx_odd, f1);
            let f_fold01 =
                _mm512_xor_si512(f_even01, ghash_mul_x4(r_bcast, _mm512_xor_si512(f_even01, f_odd01)));

            let f_even23 = _mm512_permutex2var_epi64(f2, idx_even, f3);
            let f_odd23 = _mm512_permutex2var_epi64(f2, idx_odd, f3);
            let f_fold23 =
                _mm512_xor_si512(f_even23, ghash_mul_x4(r_bcast, _mm512_xor_si512(f_even23, f_odd23)));

            // --- b fold: 8 source → 4 folded ---
            let bs = 2 * (b_base + t);
            let b0 = _mm512_loadu_si512(b_src.as_ptr().add(bs) as *const __m512i);
            let b1 = _mm512_loadu_si512(b_src.as_ptr().add(bs + 4) as *const __m512i);
            let b2 = _mm512_loadu_si512(b_src.as_ptr().add(bs + 8) as *const __m512i);
            let b3 = _mm512_loadu_si512(b_src.as_ptr().add(bs + 12) as *const __m512i);

            let b_even01 = _mm512_permutex2var_epi64(b0, idx_even, b1);
            let b_odd01 = _mm512_permutex2var_epi64(b0, idx_odd, b1);
            let b_fold01 =
                _mm512_xor_si512(b_even01, ghash_mul_x4(r_bcast, _mm512_xor_si512(b_even01, b_odd01)));

            let b_even23 = _mm512_permutex2var_epi64(b2, idx_even, b3);
            let b_odd23 = _mm512_permutex2var_epi64(b2, idx_odd, b3);
            let b_fold23 =
                _mm512_xor_si512(b_even23, ghash_mul_x4(r_bcast, _mm512_xor_si512(b_even23, b_odd23)));

            // Store folded results.
            _mm512_storeu_si512(f_dst.as_mut_ptr().add(t) as *mut __m512i, f_fold01);
            _mm512_storeu_si512(f_dst.as_mut_ptr().add(t + 4) as *mut __m512i, f_fold23);
            _mm512_storeu_si512(b_dst.as_mut_ptr().add(t) as *mut __m512i, b_fold01);
            _mm512_storeu_si512(b_dst.as_mut_ptr().add(t + 4) as *mut __m512i, b_fold23);

            // --- Message accumulation from register-resident folded values ---
            // u_0: products at even pair-positions t, t+2, t+4, t+6.
            let f_even = _mm512_permutex2var_epi64(f_fold01, idx_even, f_fold23);
            let b_even = _mm512_permutex2var_epi64(b_fold01, idx_even, b_fold23);
            u0_acc.mul_acc(f_even, b_even);

            // u_2: pair sums (t,t+1), (t+2,t+3), (t+4,t+5), (t+6,t+7).
            let f01s = _mm512_xor_si512(f_fold01, _mm512_permutexvar_epi64(perm_swap, f_fold01));
            let f23s = _mm512_xor_si512(f_fold23, _mm512_permutexvar_epi64(perm_swap, f_fold23));
            let f_sum = _mm512_permutex2var_epi64(f01s, idx_even, f23s);
            let b01s = _mm512_xor_si512(b_fold01, _mm512_permutexvar_epi64(perm_swap, b_fold01));
            let b23s = _mm512_xor_si512(b_fold23, _mm512_permutexvar_epi64(perm_swap, b_fold23));
            let b_sum = _mm512_permutex2var_epi64(b01s, idx_even, b23s);
            u2_acc.mul_acc(f_sum, b_sum);

            t += 8;
        }

        let mut u0 = u0_acc.fold().reduce();
        let mut u2 = u2_acc.fold().reduce();

        // Scalar tail: fold remaining pairs into dst.
        while t < f_dst.len() {
            let fs = 2 * (f_base + t);
            let f_even = f_src[fs];
            let f_fold = f_even + r * (f_even + f_src[fs + 1]);
            let bs = 2 * (b_base + t);
            let b_even = b_src[bs];
            let b_fold = b_even + r * (b_even + b_src[bs + 1]);
            f_dst[t] = f_fold;
            b_dst[t] = b_fold;
            t += 1;
        }

        // Message from scalar tail: same pairs as msg_reduce_avx512 tail.
        let mut k = 0;
        while k + 1 < f_dst.len() {
            let f0 = f_dst[k];
            let f1 = f_dst[k + 1];
            let b0 = b_dst[k];
            let b1 = b_dst[k + 1];
            u0 += f0 * b0;
            u2 += (f0 + f1) * (b0 + b1);
            k += 2;
        }

        (u0, u2)
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
