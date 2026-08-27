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

/// Lines of the sixteen-bank source asked for ahead of the quad the kernel
/// is folding, in F128 elements. `FLOCK_NO_FOLD16_PF=1` removes the hints
/// (they move no data of their own and change no value, so the fold is
/// byte-identical either way); `FLOCK_FOLD16_PF=<n>` overrides the distance.
const FOLD16_PF_AHEAD: usize = 512;

fn fold16_pf_ahead() -> usize {
    static D: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        if std::env::var_os("FLOCK_NO_FOLD16_PF").is_some() {
            return 0;
        }
        std::env::var("FLOCK_FOLD16_PF")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(FOLD16_PF_AHEAD)
    });
    *D
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
        let pf_ahead = fold16_pf_ahead();
        let pf_limit = src.len().saturating_sub(64);
        let mut t = 0usize;
        while t < quads {
            if pf_ahead != 0 {
                let ahead = 16 * t + pf_ahead;
                if ahead <= pf_limit {
                    let p = src.as_ptr().add(ahead).cast::<i8>();
                    let mut l = 0usize;
                    while l < 1024 {
                        _mm_prefetch::<_MM_HINT_T0>(p.add(l));
                        l += 64;
                    }
                }
            }
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

/// Direct 64-bank pad-57 fold, eight output slots per pass. The four even
/// slots consume every bank; the four odd slots consume banks 0..56 and never
/// load the structural-zero tail 57..63. Each high group keeps one even and
/// one odd [`WideGhashX4`] live so their sixteen low-bank weights are shared,
/// then the two high challenges are applied after the four group reductions.
/// The ranked block issues `255 * 124 = 31_620` default hints. The chunked
/// #1472 fallback resets lookahead eight times and issues `8 * 248 * 16 =
/// 31_744`, so the net delta is only 124 hints per block (31_744 hints, about
/// 1.94 MiB of hinted address footprint, over the whole proof); demand-load
/// and `mid4` traffic accounting is intentionally kept separate.
///
/// # Safety
/// Caller guarantees `avx512f` + `vpclmulqdq`, `src.len() == 64 * dst.len()`,
/// and `dst.len()` is a multiple of eight.
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold64_banked_pad57_pairs(
    src: &[F128],
    dst: &mut [F128],
    w: &[F128; 16],
    r4: F128,
    r5: F128,
) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4, WideGhashX4};
    use core::arch::x86_64::*;

    debug_assert_eq!(src.len(), 64 * dst.len());
    debug_assert!(dst.len().is_multiple_of(8));

    // SAFETY: the caller provides the feature and slice-geometry contracts.
    unsafe {
        // Resolve the diagnostics selector before any wide live range exists;
        // the LazyLock cold path therefore has no ZMM state to preserve.
        let pf_ahead = fold16_pf_ahead();
        let pf_limit = src.len().saturating_sub(512);
        let r4v = _mm512_broadcast_i32x4(_mm_set_epi64x(r4.hi as i64, r4.lo as i64));
        let r5v = _mm512_broadcast_i32x4(_mm_set_epi64x(r5.hi as i64, r5.lo as i64));
        let tr1_lo = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
        let tr1_hi = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
        let tr2_lo = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
        let tr2_hi = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
        let interleave_lo = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
        let interleave_hi = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);

        macro_rules! weight_vec {
            ($bank:expr) => {{
                // Keep one broadcast live at a time. A volatile 128-bit read
                // prevents loop-invariant hoisting of all sixteen ZMM weights
                // into spills while preserving the tiny read-only table value.
                let weight = core::ptr::read_volatile(
                    w.as_ptr().add($bank).cast::<__m128i>(),
                );
                _mm512_broadcast_i32x4(weight)
            }};
        }
        macro_rules! load_bank_quad {
            ($t:expr, $parity:expr, $high:expr, $quad:expr) => {{
                let base = 64 * ($t + $parity) + 16 * $high + 4 * $quad;
                let a0 = _mm512_loadu_si512(src.as_ptr().add(base) as *const __m512i);
                let a1 = _mm512_loadu_si512(src.as_ptr().add(base + 128) as *const __m512i);
                let a2 = _mm512_loadu_si512(src.as_ptr().add(base + 256) as *const __m512i);
                let a3 = _mm512_loadu_si512(src.as_ptr().add(base + 384) as *const __m512i);
                let p0 = _mm512_permutex2var_epi64(a0, tr1_lo, a1);
                let p1 = _mm512_permutex2var_epi64(a0, tr1_hi, a1);
                let p2 = _mm512_permutex2var_epi64(a2, tr1_lo, a3);
                let p3 = _mm512_permutex2var_epi64(a2, tr1_hi, a3);
                (
                    _mm512_permutex2var_epi64(p0, tr2_lo, p2),
                    _mm512_permutex2var_epi64(p0, tr2_hi, p2),
                    _mm512_permutex2var_epi64(p1, tr2_lo, p3),
                    _mm512_permutex2var_epi64(p1, tr2_hi, p3),
                )
            }};
        }
        macro_rules! full_pair_group {
            ($t:expr, $high:expr) => {{
                let mut even_acc = WideGhashX4::zero();
                let mut odd_acc = WideGhashX4::zero();
                let mut quad = 0usize;
                while quad < 4 {
                    let (e0, e1, e2, e3) = load_bank_quad!($t, 0, $high, quad);
                    let (o0, o1, o2, o3) = load_bank_quad!($t, 1, $high, quad);
                    let wb0 = weight_vec!(4 * quad);
                    even_acc.mul_acc(e0, wb0);
                    odd_acc.mul_acc(o0, wb0);
                    let wb1 = weight_vec!(4 * quad + 1);
                    even_acc.mul_acc(e1, wb1);
                    odd_acc.mul_acc(o1, wb1);
                    let wb2 = weight_vec!(4 * quad + 2);
                    even_acc.mul_acc(e2, wb2);
                    odd_acc.mul_acc(o2, wb2);
                    let wb3 = weight_vec!(4 * quad + 3);
                    even_acc.mul_acc(e3, wb3);
                    odd_acc.mul_acc(o3, wb3);
                    quad += 1;
                }
                (even_acc.reduce_lanes(), odd_acc.reduce_lanes())
            }};
        }
        macro_rules! bind_pair {
            ($lo:expr, $hi:expr, $r:expr) => {{
                _mm512_xor_si512($lo, ghash_mul_x4($r, _mm512_xor_si512($lo, $hi)))
            }};
        }
        macro_rules! prefetch_high_group {
            ($base:expr, $high:expr) => {{
                if pf_ahead != 0 && $base <= pf_limit {
                    let mut slot = 0usize;
                    while slot < 8 {
                        let group = $base + 64 * slot + 16 * $high;
                        let mut line = 0usize;
                        while line < 4 {
                            // Odd high-group line three is banks 60..63: the
                            // only wholly-dead cache line in the pad57 shape.
                            // A non-output-aligned diagnostics override has no
                            // provable future parity, so it keeps all hints.
                            let dead_line = pf_ahead.is_multiple_of(64)
                                && $high == 3
                                && line == 3
                                && (($base / 64 + slot) & 1 == 1);
                            if !dead_line {
                                _mm_prefetch::<_MM_HINT_T0>(
                                    src.as_ptr().add(group + 4 * line).cast::<i8>(),
                                );
                            }
                            line += 1;
                        }
                        slot += 1;
                    }
                }
            }};
        }

        let mut t = 0usize;
        while t < dst.len() {
            let pf_base = 64 * t + pf_ahead;
            prefetch_high_group!(pf_base, 0);
            let (even0, odd0) = full_pair_group!(t, 0);
            prefetch_high_group!(pf_base, 1);
            let (even1, odd1) = full_pair_group!(t, 1);
            let even_low = bind_pair!(even0, even1, r4v);
            let odd_low = bind_pair!(odd0, odd1, r4v);

            prefetch_high_group!(pf_base, 2);
            let (even2, odd2) = full_pair_group!(t, 2);

            // High group three: even slots still consume all sixteen banks.
            // Odd slots consume banks 48..55 through two ordinary transposes,
            // then bank 56 through four scalar 128-bit loads and inserts. No
            // load spans any odd bank in 57..63.
            prefetch_high_group!(pf_base, 3);
            let (even3, odd3) = {
                let mut even_acc = WideGhashX4::zero();
                let mut odd_acc = WideGhashX4::zero();
                let mut quad = 0usize;
                while quad < 2 {
                    let (e0, e1, e2, e3) = load_bank_quad!(t, 0, 3, quad);
                    let (o0, o1, o2, o3) = load_bank_quad!(t, 1, 3, quad);
                    let wb0 = weight_vec!(4 * quad);
                    even_acc.mul_acc(e0, wb0);
                    odd_acc.mul_acc(o0, wb0);
                    let wb1 = weight_vec!(4 * quad + 1);
                    even_acc.mul_acc(e1, wb1);
                    odd_acc.mul_acc(o1, wb1);
                    let wb2 = weight_vec!(4 * quad + 2);
                    even_acc.mul_acc(e2, wb2);
                    odd_acc.mul_acc(o2, wb2);
                    let wb3 = weight_vec!(4 * quad + 3);
                    even_acc.mul_acc(e3, wb3);
                    odd_acc.mul_acc(o3, wb3);
                    quad += 1;
                }

                let (e8, e9, e10, e11) = load_bank_quad!(t, 0, 3, 2);
                let odd56 = {
                    let base = 64 * (t + 1) + 56;
                    let load = |offset: usize| {
                        _mm_loadu_si128(src.as_ptr().add(base + offset) as *const __m128i)
                    };
                    let v = _mm512_castsi128_si512(load(0));
                    let v = _mm512_inserti32x4::<1>(v, load(128));
                    let v = _mm512_inserti32x4::<2>(v, load(256));
                    _mm512_inserti32x4::<3>(v, load(384))
                };
                let wb8 = weight_vec!(8);
                even_acc.mul_acc(e8, wb8);
                odd_acc.mul_acc(odd56, wb8);
                let wb9 = weight_vec!(9);
                even_acc.mul_acc(e9, wb9);
                let wb10 = weight_vec!(10);
                even_acc.mul_acc(e10, wb10);
                let wb11 = weight_vec!(11);
                even_acc.mul_acc(e11, wb11);

                let (e12, e13, e14, e15) = load_bank_quad!(t, 0, 3, 3);
                let wb12 = weight_vec!(12);
                even_acc.mul_acc(e12, wb12);
                let wb13 = weight_vec!(13);
                even_acc.mul_acc(e13, wb13);
                let wb14 = weight_vec!(14);
                even_acc.mul_acc(e14, wb14);
                let wb15 = weight_vec!(15);
                even_acc.mul_acc(e15, wb15);

                (even_acc.reduce_lanes(), odd_acc.reduce_lanes())
            };

            let even_high = bind_pair!(even2, even3, r4v);
            let odd_high = bind_pair!(odd2, odd3, r4v);
            let even = bind_pair!(even_low, even_high, r5v);
            let odd = bind_pair!(odd_low, odd_high, r5v);
            let out0 = _mm512_permutex2var_epi64(even, interleave_lo, odd);
            let out1 = _mm512_permutex2var_epi64(even, interleave_hi, odd);
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, out0);
            _mm512_storeu_si512(dst.as_mut_ptr().add(t + 4) as *mut __m512i, out1);
            t += 8;
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

/// Four-lane split-half bind: `lo[i] = lo[i] + r·(hi[i] + lo[i])`.
///
/// The two operand runs are separate contiguous slices (the top-bit split of
/// one table), so lanes pair up straight out of the loads — no permute.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`; `hi.len() >= lo.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn bind_split_half(lo: &mut [F128], hi: &[F128], r: F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    debug_assert!(hi.len() >= lo.len());
    // SAFETY: caller supplies target features and one `hi` per `lo` slot.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let lanes = lo.len() & !3;
        let mut i = 0usize;
        while i < lanes {
            let a = _mm512_loadu_si512(lo.as_ptr().add(i) as *const __m512i);
            let b = _mm512_loadu_si512(hi.as_ptr().add(i) as *const __m512i);
            let new = _mm512_xor_si512(a, ghash_mul_x4(r_bcast, _mm512_xor_si512(a, b)));
            _mm512_storeu_si512(lo.as_mut_ptr().add(i) as *mut __m512i, new);
            i += 4;
        }
        while i < lo.len() {
            lo[i] = lo[i] + r * (hi[i] + lo[i]);
            i += 1;
        }
    }
}

/// Four-lane split-half product-sumcheck message:
/// `(Σ chi[i]·zhi[i], Σ (chi[i]+clo[i])·(zhi[i]+zlo[i]))`.
///
/// Both sums use one deferred-reduction accumulator each (reduction is
/// F₂-linear, so one reduce per sum equals reducing every product), with an
/// unreduced scalar tail XORed in before the single reduce.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`; all four slices at least `n` long.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn msg_split_half(
    chi: &[F128],
    clo: &[F128],
    zhi: &[F128],
    zlo: &[F128],
    n: usize,
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use crate::field::gf2_128::F256Unreduced;
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees features and that every slice covers `n`.
    unsafe {
        let mut e1_wide = WideGhashX4::zero();
        let mut einf_wide = WideGhashX4::zero();
        let lanes = n & !3;
        let mut i = 0usize;
        while i < lanes {
            let ch = _mm512_loadu_si512(chi.as_ptr().add(i) as *const __m512i);
            let zh = _mm512_loadu_si512(zhi.as_ptr().add(i) as *const __m512i);
            e1_wide.mul_acc(ch, zh);
            let cl = _mm512_loadu_si512(clo.as_ptr().add(i) as *const __m512i);
            let zl = _mm512_loadu_si512(zlo.as_ptr().add(i) as *const __m512i);
            einf_wide.mul_acc(
                _mm512_xor_si512(ch, cl),
                _mm512_xor_si512(zh, zl),
            );
            i += 4;
        }
        let mut e1_acc = F256Unreduced::ZERO;
        let mut einf_acc = F256Unreduced::ZERO;
        while i < n {
            e1_acc ^= chi[i].mul_unreduced(zhi[i]);
            einf_acc ^= (chi[i] + clo[i]).mul_unreduced(zhi[i] + zlo[i]);
            i += 1;
        }
        e1_acc ^= e1_wide.fold();
        einf_acc ^= einf_wide.fold();
        (e1_acc.reduce(), einf_acc.reduce())
    }
}

/// Four-lane fused quarter bind + next-round message. Per slot `i`:
/// `lo = c0[i] + r·(c2[i]+c0[i])`, `hi = c1[i] + r·(c3[i]+c1[i])`, likewise
/// for `z`; `c0[i]/c1[i]/z0[i]/z1[i]` take the bound values, and the message
/// accumulates `hi·zhi` and `(hi+lo)·(zhi+zlo)`.
///
/// Quarters are separate contiguous runs, so no lane permute is needed; the
/// four binds share one broadcast `r`.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`; all eight slices at least `n` long.
#[target_feature(enable = "avx512f,vpclmulqdq")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn bind_both_and_msg_split(
    cq0: &mut [F128],
    cq1: &mut [F128],
    cq2: &[F128],
    cq3: &[F128],
    zq0: &mut [F128],
    zq1: &mut [F128],
    zq2: &[F128],
    zq3: &[F128],
    r: F128,
    n: usize,
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4, WideGhashX4};
    use crate::field::gf2_128::F256Unreduced;
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees features and that every slice covers `n`.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let mut e1_wide = WideGhashX4::zero();
        let mut einf_wide = WideGhashX4::zero();
        let lanes = n & !3;
        let mut i = 0usize;
        while i < lanes {
            let c0 = _mm512_loadu_si512(cq0.as_ptr().add(i) as *const __m512i);
            let c1 = _mm512_loadu_si512(cq1.as_ptr().add(i) as *const __m512i);
            let c2 = _mm512_loadu_si512(cq2.as_ptr().add(i) as *const __m512i);
            let c3 = _mm512_loadu_si512(cq3.as_ptr().add(i) as *const __m512i);
            let z0 = _mm512_loadu_si512(zq0.as_ptr().add(i) as *const __m512i);
            let z1 = _mm512_loadu_si512(zq1.as_ptr().add(i) as *const __m512i);
            let z2 = _mm512_loadu_si512(zq2.as_ptr().add(i) as *const __m512i);
            let z3 = _mm512_loadu_si512(zq3.as_ptr().add(i) as *const __m512i);

            let lo = _mm512_xor_si512(c0, ghash_mul_x4(r_bcast, _mm512_xor_si512(c2, c0)));
            let hi = _mm512_xor_si512(c1, ghash_mul_x4(r_bcast, _mm512_xor_si512(c3, c1)));
            let zlo = _mm512_xor_si512(z0, ghash_mul_x4(r_bcast, _mm512_xor_si512(z2, z0)));
            let zhi = _mm512_xor_si512(z1, ghash_mul_x4(r_bcast, _mm512_xor_si512(z3, z1)));

            _mm512_storeu_si512(cq0.as_mut_ptr().add(i) as *mut __m512i, lo);
            _mm512_storeu_si512(cq1.as_mut_ptr().add(i) as *mut __m512i, hi);
            _mm512_storeu_si512(zq0.as_mut_ptr().add(i) as *mut __m512i, zlo);
            _mm512_storeu_si512(zq1.as_mut_ptr().add(i) as *mut __m512i, zhi);

            e1_wide.mul_acc(hi, zhi);
            einf_wide.mul_acc(
                _mm512_xor_si512(hi, lo),
                _mm512_xor_si512(zhi, zlo),
            );
            i += 4;
        }
        let mut e1_acc = F256Unreduced::ZERO;
        let mut einf_acc = F256Unreduced::ZERO;
        while i < n {
            let lo = cq0[i] + r * (cq2[i] + cq0[i]);
            let hi = cq1[i] + r * (cq3[i] + cq1[i]);
            let zlo = zq0[i] + r * (zq2[i] + zq0[i]);
            let zhi = zq1[i] + r * (zq3[i] + zq1[i]);
            cq0[i] = lo;
            cq1[i] = hi;
            zq0[i] = zlo;
            zq1[i] = zhi;
            e1_acc ^= hi.mul_unreduced(zhi);
            einf_acc ^= (hi + lo).mul_unreduced(zhi + zlo);
            i += 1;
        }
        e1_acc ^= e1_wide.fold();
        einf_acc ^= einf_wide.fold();
        (e1_acc.reduce(), einf_acc.reduce())
    }
}
