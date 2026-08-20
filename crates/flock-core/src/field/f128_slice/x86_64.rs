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

/// In-place DirectFold8 factor-state bind: adjacent-pair fold of `f` and `b`
/// with fused `(u0,u2)` accumulate. Same permute body as [`fold_pairs`] and
/// the same even/odd message layout as `msg_reduce_avx512`.
///
/// In-place is safe because output `t` depends only on source `2t..2t+2`,
/// and stores of `dst[0..t)` never overlap unread source `2t..`.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`. `f.len() == b.len()`, multiple of 4.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_two_and_msg_in_place(
    f: &mut Vec<F128>,
    b: &mut Vec<F128>,
    r: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4, WideGhashX4};
    use core::arch::x86_64::*;

    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len().is_multiple_of(4));
    let half = f.len() / 2;

    // SAFETY: caller guarantees features and even pair counts; loads of
    // `2t..2t+8` complete before stores to `t..t+4` overlap those addresses.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let idx_even = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let idx_odd = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let perm_swap = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);

        let fold4 = |ptr: *const F128, t: usize| -> __m512i {
            let s = 2 * t;
            let lo = _mm512_loadu_si512(ptr.add(s) as *const __m512i);
            let hi = _mm512_loadu_si512(ptr.add(s + 4) as *const __m512i);
            let even = _mm512_permutex2var_epi64(lo, idx_even, hi);
            let odd = _mm512_permutex2var_epi64(lo, idx_odd, hi);
            _mm512_xor_si512(even, ghash_mul_x4(r_bcast, _mm512_xor_si512(even, odd)))
        };

        let mut u0_acc = WideGhashX4::zero();
        let mut u2_acc = WideGhashX4::zero();
        let f_ptr = f.as_mut_ptr();
        let b_ptr = b.as_mut_ptr();
        let lanes = half & !7;
        let mut t = 0usize;
        while t < lanes {
            let f0 = fold4(f_ptr, t);
            let f1 = fold4(f_ptr, t + 4);
            let b0 = fold4(b_ptr, t);
            let b1 = fold4(b_ptr, t + 4);
            _mm512_storeu_si512(f_ptr.add(t) as *mut __m512i, f0);
            _mm512_storeu_si512(f_ptr.add(t + 4) as *mut __m512i, f1);
            _mm512_storeu_si512(b_ptr.add(t) as *mut __m512i, b0);
            _mm512_storeu_si512(b_ptr.add(t + 4) as *mut __m512i, b1);

            let f_even = _mm512_permutex2var_epi64(f0, idx_even, f1);
            let b_even = _mm512_permutex2var_epi64(b0, idx_even, b1);
            u0_acc.mul_acc(f_even, b_even);

            let f0s = _mm512_xor_si512(f0, _mm512_permutexvar_epi64(perm_swap, f0));
            let f1s = _mm512_xor_si512(f1, _mm512_permutexvar_epi64(perm_swap, f1));
            let f_sum = _mm512_permutex2var_epi64(f0s, idx_even, f1s);
            let b0s = _mm512_xor_si512(b0, _mm512_permutexvar_epi64(perm_swap, b0));
            let b1s = _mm512_xor_si512(b1, _mm512_permutexvar_epi64(perm_swap, b1));
            let b_sum = _mm512_permutex2var_epi64(b0s, idx_even, b1s);
            u2_acc.mul_acc(f_sum, b_sum);

            t += 8;
        }

        let mut u0 = u0_acc.fold().reduce();
        let mut u2 = u2_acc.fold().reduce();
        while t < half {
            let source = 2 * t;
            let f0 = *f_ptr.add(source) + r * (*f_ptr.add(source) + *f_ptr.add(source + 1));
            let f1 = *f_ptr.add(source + 2)
                + r * (*f_ptr.add(source + 2) + *f_ptr.add(source + 3));
            let b0 = *b_ptr.add(source) + r * (*b_ptr.add(source) + *b_ptr.add(source + 1));
            let b1 = *b_ptr.add(source + 2)
                + r * (*b_ptr.add(source + 2) + *b_ptr.add(source + 3));
            *f_ptr.add(t) = f0;
            *f_ptr.add(t + 1) = f1;
            *b_ptr.add(t) = b0;
            *b_ptr.add(t + 1) = b1;
            u0 += f0 * b0;
            u2 += (f0 + f1) * (b0 + b1);
            t += 2;
        }
        f.truncate(half);
        b.truncate(half);
        (u0, u2)
    }
}
