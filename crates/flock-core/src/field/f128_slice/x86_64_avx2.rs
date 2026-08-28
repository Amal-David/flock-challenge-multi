use crate::field::F128;

// SAFETY: every public function in this module is `#[target_feature(enable =
// "avx2,pclmulqdq[,sse4.1]")]`, and the parent module's cfg gate
// (`#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature =
// "pclmulqdq"))]`) means these target features are statically required.
//
// "2-way interleaved" means we hold 2 F128s in each 256-bit ymm register —
// one F128 per 128-bit lane — and the two F128s of a pair fold live in
// adjacent lanes of the same ymm. Loading 2 contiguous F128 from src gives
// (even, odd) of one pair directly, and a second ymm gives the next pair;
// one ymm XOR + one ymm-wide 2-lane clmul + one ymm XOR-with-even produces 2
// folded outputs, stored in a single ymm.

/// Per-tile source size for the 8 KiB tiling. 8 KiB = 512 F128 = 256 pairs;
/// each tile writes 256 F128 to dst.
const FOLD_AVX2_TILE_F128: usize = 512;

/// Prefetch distance in tiles. 4 tiles ahead = 32 KiB hinted per outer step —
/// enough to overlap L2-fill latency for the next tile while still keeping
/// the working set in L1d. `FLOCK_NO_AVX2_FOLD_PF=1` disables; the
/// `FLOCK_AVX2_FOLD_PF=<n>` env var overrides the constant.
const FOLD_AVX2_PF_AHEAD: usize = 4;

fn fold_avx2_pf_ahead() -> usize {
    static D: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        if std::env::var_os("FLOCK_NO_AVX2_FOLD_PF").is_some() {
            return 0;
        }
        std::env::var("FLOCK_AVX2_FOLD_PF")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(FOLD_AVX2_PF_AHEAD)
    });
    *D
}

/// Lane-independent GF(2^128) carry-less product, binius-style.
///
/// 4 CLMUL schoolbook (lo·lo, lo·hi, hi·lo, hi·hi) + 2-stage 0x87 reduction
/// (t1 = t1 ⊕ (t2<<64) ⊕ clmul(t2.hi, 0x87); then t0 = t0 ⊕ (t1<<64) ⊕
/// clmul(t1.hi, 0x87)). Field-identical to `ghash_mul_binius` and
/// `ghash_mul_karatsuba_vec` (reduction commutes with the cross terms).
///
/// # Safety
/// Caller must ensure `pclmulqdq` and `sse4.1` are available.
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn ghash_mul_128(
    a: core::arch::x86_64::__m128i,
    b: core::arch::x86_64::__m128i,
) -> core::arch::x86_64::__m128i {
    // SAFETY: caller carries pclmulqdq + sse4.1.
    unsafe {
        use core::arch::x86_64::*;
        let t0 = _mm_clmulepi64_si128::<0x00>(a, b);
        let t1a = _mm_clmulepi64_si128::<0x10>(a, b);
        let t1b = _mm_clmulepi64_si128::<0x01>(a, b);
        let mut t1 = _mm_xor_si128(t1a, t1b);
        let t2 = _mm_clmulepi64_si128::<0x11>(a, b);
        let poly = _mm_set_epi64x(0, 0x87);
        // First reduce: t1 += (t2 << 64) ⊕ clmul(t2.hi, 0x87).
        let t2_shifted = _mm_slli_si128::<8>(t2);
        t1 = _mm_xor_si128(t1, t2_shifted);
        let t2_red = _mm_clmulepi64_si128::<0x01>(t2, poly);
        t1 = _mm_xor_si128(t1, t2_red);
        // Second reduce: t0 += (t1 << 64) ⊕ clmul(t1.hi, 0x87).
        let t1_shifted = _mm_slli_si128::<8>(t1);
        let mut t0 = _mm_xor_si128(t0, t1_shifted);
        let t1_red = _mm_clmulepi64_si128::<0x01>(t1, poly);
        t0 = _mm_xor_si128(t0, t1_red);
        t0
    }
}

/// 2-lane GF(2^128) product on ymm — calls the 128-bit lane helper twice
/// and re-packs. Independent of the (aarch64) 2-lane `ghash_mul_vec2_neon`
/// implementation; same shape (low/high 128-bit lane extract, two CLMUL
/// sequences, re-pack).
#[inline]
#[target_feature(enable = "avx2,pclmulqdq,sse4.1")]
unsafe fn ghash_mul_x2(
    x: core::arch::x86_64::__m256i,
    y: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256i {
    // SAFETY: caller carries avx2 + pclmulqdq + sse4.1.
    unsafe {
        use core::arch::x86_64::*;
        let x_lo = _mm256_castsi256_si128(x);
        let x_hi = _mm256_extracti128_si256::<1>(x);
        let y_lo = _mm256_castsi256_si128(y);
        let y_hi = _mm256_extracti128_si256::<1>(y);
        let p0 = ghash_mul_128(x_lo, y_lo);
        let p1 = ghash_mul_128(x_hi, y_hi);
        _mm256_set_m128i(p1, p0)
    }
}

/// 2-lane pair fold: `dst[t] = src[2j] + r·(src[2j] ⊕ src[2j+1])`,
/// processed two pairs at a time (one ymm load = 2 F128 = 1 pair, so two
/// ymm loads = 2 pairs = 2 outputs = one ymm store).
///
/// 8 KiB tile: each outer iteration consumes 512 source F128 (= 256 outputs)
/// and writes 256 destination F128. The whole tile fits in L1d on SPR
/// (32 KiB); the inner loop is a streaming read, so the prefetcher is given
/// `pf_ahead` tiles of lead time to overlap L2-fill latency.
///
/// # Safety
/// Caller guarantees the target features and that `src` contains both
/// elements for every output pair (i.e. `base + dst.len() <= src.len() / 2`).
#[target_feature(enable = "avx2,pclmulqdq,sse4.1")]
pub(super) unsafe fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    use core::arch::x86_64::*;
    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let r_x2 = _mm256_broadcastsi128_si256(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let pairs_per_tile = FOLD_AVX2_TILE_F128 / 2; // 256
        let n_out = dst.len();
        let pf_ahead = fold_avx2_pf_ahead();
        // Outer loop: one tile = `pairs_per_tile` outputs.
        let mut t0 = 0usize;
        while t0 < n_out {
            let tile_out = (n_out - t0).min(pairs_per_tile);
            // Prefetch the source for the tile `pf_ahead` steps ahead of
            // the current one. `_mm_prefetch` is a hint — the reduction is
            // byte-identical with or without it, and the kill switch lets
            // A/B-test the noise.
            if pf_ahead != 0 {
                let ahead_pair = t0 + pf_ahead * pairs_per_tile;
                if ahead_pair < n_out {
                    let ahead_src = 2 * (base + ahead_pair);
                    let remaining = (src.len().saturating_sub(ahead_src)) * 16;
                    let hint_bytes = (FOLD_AVX2_TILE_F128 * 16 * pf_ahead).min(remaining);
                    let mut off = 0usize;
                    while off < hint_bytes {
                        _mm_prefetch::<_MM_HINT_T0>(
                            src.as_ptr().add(ahead_src).cast::<i8>().add(off),
                        );
                        off += 64;
                    }
                }
            }
            let mut t = 0usize;
            // Inner loop: 2 pairs per ymm store.
            let even_pairs = tile_out & !1;
            while t < even_pairs {
                let s = 2 * (base + t0 + t);
                // pair0 = (even[0], odd[0]) of output t.
                let pair0 = _mm256_loadu_si256(src.as_ptr().add(s) as *const __m256i);
                // pair1 = (even[1], odd[1]) of output t+1.
                let pair1 = _mm256_loadu_si256(src.as_ptr().add(s + 2) as *const __m256i);
                // We need even = {even[0], even[1]} and odd = {odd[0], odd[1]}
                // in ymm lane order (low 128 = output 0, high 128 = output 1).
                // _mm256_permute2x128_si256::<0x20>(a, b) returns
                //   low 128 = a.low, high 128 = b.low
                // so even_y = permute2x128(pair0, pair1, 0x20) gives
                //   (e0, e1) in the two 128-bit lanes — exactly what we want.
                let even_y = _mm256_permute2x128_si256::<0x20>(pair0, pair1);
                // _mm256_permute2x128_si256::<0x31>(a, b) returns
                //   low 128 = a.high, high 128 = b.high
                // so odd_y = permute2x128(pair0, pair1, 0x31) gives (o0, o1).
                let odd_y = _mm256_permute2x128_si256::<0x31>(pair0, pair1);
                let diff_y = _mm256_xor_si256(even_y, odd_y);
                let prod_y = ghash_mul_x2(r_x2, diff_y);
                let out_y = _mm256_xor_si256(even_y, prod_y);
                _mm256_storeu_si256(dst.as_mut_ptr().add(t0 + t) as *mut __m256i, out_y);
                t += 2;
            }
            // Tile tail (1 leftover pair, only possible at the very end of
            // the last tile when `n_out` is odd).
            if t < tile_out {
                let s = 2 * (base + t0 + t);
                let pair0 = _mm256_loadu_si256(src.as_ptr().add(s) as *const __m256i);
                // For a single pair, even = pair0.low, odd = pair0.high.
                // We don't have a partner; replicate odd via lane-swap so
                // the ghash_mul_x2 body is still 2-wide (the high 128 of
                // the result is unused and discarded).
                let even_y = _mm256_permute2x128_si256::<0x20>(pair0, pair0);
                let odd_y = _mm256_permute2x128_si256::<0x31>(pair0, pair0);
                let diff_y = _mm256_xor_si256(even_y, odd_y);
                let prod_y = ghash_mul_x2(r_x2, diff_y);
                let out_y = _mm256_xor_si256(even_y, prod_y);
                // Store the low 128 only.
                let lo128 = _mm256_castsi256_si128(out_y);
                _mm_storeu_si128(dst.as_mut_ptr().add(t0 + t) as *mut __m128i, lo128);
            }
            t0 += tile_out;
        }
    }
}

/// 2-lane `dst += scale * addend` (lazy-OOD correction).
///
/// # Safety
/// Requires `avx2` + `pclmulqdq`; the caller guarantees `dst.len() ==
/// addend.len()`.
#[target_feature(enable = "avx2,pclmulqdq")]
pub(super) unsafe fn add_scaled(dst: &mut [F128], addend: &[F128], scale: F128) {
    use core::arch::x86_64::*;
    // SAFETY: caller supplies target features and equal slice lengths.
    unsafe {
        debug_assert_eq!(dst.len(), addend.len());
        let scale_x2 =
            _mm256_broadcastsi128_si256(_mm_set_epi64x(scale.hi as i64, scale.lo as i64));
        let pairs = dst.len() & !1;
        let mut i = 0usize;
        while i < pairs {
            let cur = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
            let ext = _mm256_loadu_si256(addend.as_ptr().add(i) as *const __m256i);
            let prod = ghash_mul_x2(scale_x2, ext);
            let updated = _mm256_xor_si256(cur, prod);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, updated);
            i += 2;
        }
        while i < dst.len() {
            dst[i] += scale * addend[i];
            i += 1;
        }
    }
}

/// 2-lane nested 4-tuple fold: two independent nested 4-tuple folds packed
/// into one ymm store. Each nested fold reads 4 contiguous F128
/// (a0,a1,a2,a3) and produces one F128; we compute two such outputs
/// independently and pack the two F128s into one ymm. The inner arithmetic
/// is 128-bit (one F128 per 128-bit lane) — the "2-way" part is the
/// doubled throughput from streaming two outputs per pass, not 2-wide
/// products. This is the form the AVX2 + PCLMULQDQ tier wants: each fold
/// body is exactly the binius 6-CLMUL `ghash_mul_128`, and the
/// characteristic-2 identity keeps each F128 product independent.
///
/// # Safety
/// Caller guarantees `avx2` + `pclmulqdq` + `sse4.1` and that
/// `src.len() == 4 * dst.len()`.
#[target_feature(enable = "avx2,pclmulqdq,sse4.1")]
pub(super) unsafe fn fold4_nested(src: &[F128], dst: &mut [F128], r0: F128, r1: F128) {
    use core::arch::x86_64::*;
    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let r0_bcast = _mm_set_epi64x(r0.hi as i64, r0.lo as i64);
        let r1_bcast = _mm_set_epi64x(r1.hi as i64, r1.lo as i64);
        // 4-tuple nested fold, one output: `low + r1·(low+high)` where
        // `low = a0 + r0·(a0+a1)` and `high = a2 + r0·(a2+a3)`. Inner
        // body is two `ghash_mul_128` (the r0 pair-folds) and one (the
        // r1 fold) — three binius products per output.
        #[inline(always)]
        unsafe fn fold4_one(
            a0: F128,
            a1: F128,
            a2: F128,
            a3: F128,
            r0_bcast: __m128i,
            r1_bcast: __m128i,
        ) -> F128 {
            // SAFETY: body uses only target-feature-gated intrinsics.
            unsafe {
                let e0 = _mm_set_epi64x(a0.hi as i64, a0.lo as i64);
                let o0 = _mm_set_epi64x(a1.hi as i64, a1.lo as i64);
                let e1 = _mm_set_epi64x(a2.hi as i64, a2.lo as i64);
                let o1 = _mm_set_epi64x(a3.hi as i64, a3.lo as i64);
                let d0 = _mm_xor_si128(e0, o0);
                let p0 = ghash_mul_128(r0_bcast, d0);
                let low = _mm_xor_si128(e0, p0);
                let d1 = _mm_xor_si128(e1, o1);
                let p1 = ghash_mul_128(r0_bcast, d1);
                let high = _mm_xor_si128(e1, p1);
                let d_out = _mm_xor_si128(low, high);
                let p_out = ghash_mul_128(r1_bcast, d_out);
                let out = _mm_xor_si128(low, p_out);
                F128 {
                    lo: _mm_extract_epi64::<0>(out) as u64,
                    hi: _mm_extract_epi64::<1>(out) as u64,
                }
            }
        }
        // 2 outputs per pass (8 src F128 → 2 dst F128 = one ymm store).
        let pairs = dst.len() & !1;
        let mut t = 0usize;
        while t < pairs {
            let s = 4 * t;
            let out0 = fold4_one(src[s], src[s + 1], src[s + 2], src[s + 3], r0_bcast, r1_bcast);
            let out1 = fold4_one(src[s + 4], src[s + 5], src[s + 6], src[s + 7], r0_bcast, r1_bcast);
            let out0_v = _mm_set_epi64x(out0.hi as i64, out0.lo as i64);
            let out1_v = _mm_set_epi64x(out1.hi as i64, out1.lo as i64);
            let packed = _mm256_set_m128i(out1_v, out0_v);
            _mm256_storeu_si256(dst.as_mut_ptr().add(t) as *mut __m256i, packed);
            t += 2;
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
