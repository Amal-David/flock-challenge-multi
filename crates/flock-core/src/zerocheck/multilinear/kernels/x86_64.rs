use crate::field::gf2_128::x86_64::{WideGhashX4, f128x4_loadu};
use crate::field::{F128, F256Unreduced};

/// Fold the four rows for one round-2 pair in parallel x86 SIMD registers.
/// Returns `[a0, a1, b0, b1]`.
///
/// The table lookups are data-dependent, so they remain four independent
/// aligned 128-bit loads per chunk. Keeping four XOR chains in flight exposes
/// their load-level parallelism; the caller then batches four returned pairs
/// into the AVX-512 GHASH message kernel.
///
/// # Safety
/// `table_data` must point to an 8 × 256 `F128` table and every row pointer
/// must expose 8 readable bytes.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[inline(always)]
pub(crate) unsafe fn fold_round2_pair_x86_unchecked_8(
    table_data: *const F128,
    a0_bytes: *const u8,
    a1_bytes: *const u8,
    b0_bytes: *const u8,
    b1_bytes: *const u8,
) -> [F128; 4] {
    use core::arch::x86_64::*;

    // SAFETY: the caller guarantees all table and row bounds. Every table
    // entry is 16-byte aligned because F128 has align(16).
    unsafe {
        let rows = [a0_bytes, a1_bytes, b0_bytes, b1_bytes];
        let mut acc = [_mm_setzero_si128(); 4];
        for chunk in 0..8 {
            let table_chunk = table_data.add(chunk * 256);
            for lane in 0..4 {
                let entry = table_chunk.add(*rows[lane].add(chunk) as usize);
                acc[lane] = _mm_xor_si128(acc[lane], _mm_load_si128(entry.cast::<__m128i>()));
            }
        }
        // F128 is exactly two u64 words and accepts every bit pattern.
        acc.map(|value| core::mem::transmute::<__m128i, F128>(value))
    }
}

/// x86 fused fold plus next-round message for one worker chunk.
///
/// Each four-message iteration folds eight `a` and `b` outputs, stores them
/// for the next round, and consumes the same ZMM registers for the current
/// message before they leave registers. This removes the immediate output
/// readback performed by the portable two-pass path.
///
/// # Safety
/// Input/output lengths must satisfy `input.len() == 2 * output.len()` and
/// `output.len() == 2 * eq_lo.len()`. AVX-512F and VPCLMULQDQ are cfg-gated.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(crate) unsafe fn fold_and_message_x86_avx512(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    debug_assert_eq!(a_in.len(), 2 * a_out.len());
    debug_assert_eq!(b_in.len(), 2 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());

    // Fold four adjacent output elements and return them in one ZMM.
    #[inline(always)]
    unsafe fn fold_x4(
        src: *const F128,
        r: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
    ) -> __m512i {
        use crate::field::gf2_128::x86_64::ghash_mul_x4;
        use core::arch::x86_64::*;

        // SAFETY: caller supplies eight readable F128 values at src.
        unsafe {
            let lo = _mm512_loadu_si512(src.cast::<__m512i>());
            let hi = _mm512_loadu_si512(src.add(4).cast::<__m512i>());
            let even = _mm512_permutex2var_epi64(lo, even_idx, hi);
            let odd = _mm512_permutex2var_epi64(lo, odd_idx, hi);
            _mm512_xor_si512(even, ghash_mul_x4(r, _mm512_xor_si512(even, odd)))
        }
    }

    // SAFETY: the function's length invariants bound all loads/stores and the
    // cfg gate supplies every intrinsic feature.
    unsafe {
        let r = _mm512_broadcast_i32x4(_mm_set_epi64x(r_fold.hi as i64, r_fold.lo as i64));
        // Select even/odd F128 lanes from two concatenated ZMM inputs. The same
        // selectors deinterleave fold inputs and gather message a0/a1 lanes.
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut p1_wide = WideGhashX4::zero();
        let mut pinf_wide = WideGhashX4::zero();
        let mut p1_tail = F256Unreduced::ZERO;
        let mut pinf_tail = F256Unreduced::ZERO;
        let mut x_lo = 0;

        while x_lo + 4 <= eq_lo.len() {
            let output = 2 * x_lo;
            let a_lo = fold_x4(a_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            let a_hi = fold_x4(a_in.as_ptr().add(2 * (output + 4)), r, even_idx, odd_idx);
            let b_lo = fold_x4(b_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            let b_hi = fold_x4(b_in.as_ptr().add(2 * (output + 4)), r, even_idx, odd_idx);

            _mm512_storeu_si512(a_out.as_mut_ptr().add(output).cast::<__m512i>(), a_lo);
            _mm512_storeu_si512(a_out.as_mut_ptr().add(output + 4).cast::<__m512i>(), a_hi);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output).cast::<__m512i>(), b_lo);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output + 4).cast::<__m512i>(), b_hi);

            let a0 = _mm512_permutex2var_epi64(a_lo, even_idx, a_hi);
            let a1 = _mm512_permutex2var_epi64(a_lo, odd_idx, a_hi);
            let b0 = _mm512_permutex2var_epi64(b_lo, even_idx, b_hi);
            let b1 = _mm512_permutex2var_epi64(b_lo, odd_idx, b_hi);
            let g1 = ghash_mul_x4(a1, b1);
            let g_inf = ghash_mul_x4(_mm512_xor_si512(a0, a1), _mm512_xor_si512(b0, b1));
            let eq = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
            p1_wide.mul_acc(eq, g1);
            pinf_wide.mul_acc(eq, g_inf);
            x_lo += 4;
        }

        // Power-of-two eq blocks leave either no tail or exactly two pairs.
        if x_lo < eq_lo.len() {
            debug_assert_eq!(eq_lo.len() - x_lo, 2);
            let output = 2 * x_lo;
            let a_folded = fold_x4(a_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            let b_folded = fold_x4(b_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            _mm512_storeu_si512(a_out.as_mut_ptr().add(output).cast::<__m512i>(), a_folded);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output).cast::<__m512i>(), b_folded);

            for lane in 0..2 {
                let o = output + 2 * lane;
                let a0 = a_out[o];
                let a1 = a_out[o + 1];
                let b0 = b_out[o];
                let b1 = b_out[o + 1];
                let eq = eq_lo[x_lo + lane];
                p1_tail ^= eq.mul_unreduced(a1 * b1);
                pinf_tail ^= eq.mul_unreduced((a0 + a1) * (b0 + b1));
            }
        }

        p1_tail ^= p1_wide.fold();
        pinf_tail ^= pinf_wide.fold();
        (p1_tail.reduce(), pinf_tail.reduce())
    }
}

/// x86 lookahead round-two sweep for one worker chunk: folds every pair of
/// this chunk into `a_chunk`/`b_chunk` (bit-identical to the incumbent
/// sweep) and returns the eight per-chunk sums
/// `[p1_even, pinf_even, p1_odd, pinf_odd, W0', W3', W4', W5']`, each reduced,
/// all accumulated on the group's shared odd-lane weight `w = eq_lo[2u+1]`
/// (see `uni_skip_fold_and_round_pair_optimized_packed_padded_lookahead`).
///
/// Per four groups (eight pairs): four reduced `w`-prescalings of the `a`
/// rows, then all eight products are one unreduced `WideGhashX4::mul_acc`
/// each — 56 CLMUL against the incumbent's 40 for the same eight pairs.
///
/// `WRITE = false` (the no-materialize sweep) skips every table store;
/// `a_chunk`/`b_chunk` may then be empty.
///
/// # Safety
/// `table_data` must point to the 8 × 256 `F128` fold table; `a_pkt`/`b_pkt`
/// must expose 8 readable bytes for every post-URM row
/// `row_base .. row_base + 2·eq_lo.len()`; if `WRITE`, `a_chunk.len() ==
/// b_chunk.len() == 2·eq_lo.len()`; `eq_lo.len()` is even. AVX-512F and
/// VPCLMULQDQ are cfg-gated.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn round2_lookahead_chunk_x86_avx512<const WRITE: bool>(
    table_data: *const F128,
    mats: Option<&[u64; 128]>,
    a_pkt: *const u8,
    b_pkt: *const u8,
    row_base: usize,
    a_chunk: &mut [F128],
    b_chunk: &mut [F128],
    eq_lo: &[F128],
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [F128; 8] {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    let lo_size = eq_lo.len();
    debug_assert!(!WRITE || a_chunk.len() == 2 * lo_size);
    debug_assert!(!WRITE || b_chunk.len() == 2 * lo_size);
    debug_assert!(lo_size.is_multiple_of(2));

    // SAFETY: the function's contract bounds every packed-row read, table
    // read and chunk write; the cfg gate supplies every intrinsic feature.
    unsafe {
        // Select the odd F128 lanes of eight consecutive eq_lo values.
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut acc = [WideGhashX4::zero(); 8];
        let mut tail = [F256Unreduced::ZERO; 8];
        let mut x_lo = 0;
        // GFNI batch fold: 32 consecutive pairs = 64 consecutive rows per
        // side prefolded in one bit-matrix batch (padded pairs fold zero
        // rows; the consume path below skips them exactly as before).
        let use_batch = cfg!(all(target_feature = "avx512vbmi", target_feature = "gfni"))
            && mats.is_some()
            && lo_size >= 32
            && lo_size.is_multiple_of(32);
        let mut fa = [F128::ZERO; 64];
        let mut fb = [F128::ZERO; 64];

        while x_lo + 8 <= lo_size {
            #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
            if use_batch && x_lo.is_multiple_of(32) {
                // Refill both caches for pairs x_lo..x_lo+32 (rows
                // row_base+2*x_lo .. +64 per side) — the same bytes the
                // gather path reads.
                let m = mats.unwrap();
                let g0 = row_base + 2 * x_lo;
                gfni_fold64_rows(a_pkt.add(g0 * 8), m, fa.as_mut_ptr());
                gfni_fold64_rows(b_pkt.add(g0 * 8), m, fb.as_mut_ptr());
            }
            // Batch path: this iteration's 16 folded rows sit CONTIGUOUSLY
            // in the fa/fb caches, and `a[k][lane] = row(4·lane + k)` is
            // exactly `transpose4` of four contiguous row-ZMMs — registers
            // and permutes instead of 32 scalar 16-byte stores re-read as
            // store-forwarding-blocked ZMM loads. The WRITE arm is a
            // straight register copy: chunk order equals row order, and
            // padded pairs' cached rows are already zero (zero raw rows
            // through zero-preserving fold tables), matching the explicit
            // zero stores of the scalar arm.
            let (a0, a1, a2, a3, b0, b1, b2, b3) = if use_batch && zc_regfold_enabled() {
                let r0 = 2 * (x_lo % 32);
                let fap = fa.as_ptr().add(r0).cast::<__m512i>();
                let fbp = fb.as_ptr().add(r0).cast::<__m512i>();
                let za = [
                    _mm512_loadu_si512(fap),
                    _mm512_loadu_si512(fap.add(1)),
                    _mm512_loadu_si512(fap.add(2)),
                    _mm512_loadu_si512(fap.add(3)),
                ];
                let zb = [
                    _mm512_loadu_si512(fbp),
                    _mm512_loadu_si512(fbp.add(1)),
                    _mm512_loadu_si512(fbp.add(2)),
                    _mm512_loadu_si512(fbp.add(3)),
                ];
                if WRITE {
                    let ac = a_chunk.as_mut_ptr().add(2 * x_lo).cast::<__m512i>();
                    let bc = b_chunk.as_mut_ptr().add(2 * x_lo).cast::<__m512i>();
                    for i in 0..4 {
                        _mm512_storeu_si512(ac.add(i), za[i]);
                        _mm512_storeu_si512(bc.add(i), zb[i]);
                    }
                }
                let [a0, a1, a2, a3] = transpose4_lanes(za[0], za[1], za[2], za[3]);
                let [b0, b1, b2, b3] = transpose4_lanes(zb[0], zb[1], zb[2], zb[3]);
                (a0, a1, a2, a3, b0, b1, b2, b3)
            } else {
                // a[k][lane]: row k (0..4) of group `lane` (0..4).
                let mut a = [[F128::ZERO; 4]; 4];
                let mut b = [[F128::ZERO; 4]; 4];
                for lane in 0..4 {
                    for half in 0..2 {
                        let pair = x_lo + 2 * lane + half;
                        let x0l = 2 * pair;
                        let x1l = x0l + 1;
                        if ((pair_idx_base + pair) & pair_in_block_mask) >= useful_pairs_inclusive
                        {
                            if WRITE {
                                a_chunk[x0l] = F128::ZERO;
                                a_chunk[x1l] = F128::ZERO;
                                b_chunk[x0l] = F128::ZERO;
                                b_chunk[x1l] = F128::ZERO;
                            }
                            continue;
                        }
                        let x0g = row_base + x0l;
                        let x1g = x0g + 1;
                        let folded = if use_batch {
                            let r = 2 * (pair % 32);
                            [fa[r], fa[r + 1], fb[r], fb[r + 1]]
                        } else {
                            fold_round2_pair_x86_unchecked_8(
                                table_data,
                                a_pkt.add(x0g * 8),
                                a_pkt.add(x1g * 8),
                                b_pkt.add(x0g * 8),
                                b_pkt.add(x1g * 8),
                            )
                        };
                        a[2 * half][lane] = folded[0];
                        a[2 * half + 1][lane] = folded[1];
                        b[2 * half][lane] = folded[2];
                        b[2 * half + 1][lane] = folded[3];
                        if WRITE {
                            a_chunk[x0l] = folded[0];
                            a_chunk[x1l] = folded[1];
                            b_chunk[x0l] = folded[2];
                            b_chunk[x1l] = folded[3];
                        }
                    }
                }
                (
                    f128x4_loadu(a[0].as_ptr()),
                    f128x4_loadu(a[1].as_ptr()),
                    f128x4_loadu(a[2].as_ptr()),
                    f128x4_loadu(a[3].as_ptr()),
                    f128x4_loadu(b[0].as_ptr()),
                    f128x4_loadu(b[1].as_ptr()),
                    f128x4_loadu(b[2].as_ptr()),
                    f128x4_loadu(b[3].as_ptr()),
                )
            };
            let e_lo = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
            let e_hi = f128x4_loadu(eq_lo.as_ptr().add(x_lo + 4));
            let w = _mm512_permutex2var_epi64(e_lo, odd_idx, e_hi);
            let a0w = ghash_mul_x4(w, a0);
            let a1w = ghash_mul_x4(w, a1);
            let a2w = ghash_mul_x4(w, a2);
            let a3w = ghash_mul_x4(w, a3);
            acc[0].mul_acc(a1w, b1);
            acc[1].mul_acc(_mm512_xor_si512(a0w, a1w), _mm512_xor_si512(b0, b1));
            acc[2].mul_acc(a3w, b3);
            acc[3].mul_acc(_mm512_xor_si512(a2w, a3w), _mm512_xor_si512(b2, b3));
            acc[4].mul_acc(a2w, b2);
            let e_aw = _mm512_xor_si512(a0w, a2w);
            let e_b = _mm512_xor_si512(b0, b2);
            let o_aw = _mm512_xor_si512(a1w, a3w);
            let o_b = _mm512_xor_si512(b1, b3);
            acc[5].mul_acc(e_aw, e_b);
            acc[6].mul_acc(o_aw, o_b);
            acc[7].mul_acc(_mm512_xor_si512(e_aw, o_aw), _mm512_xor_si512(e_b, o_b));
            x_lo += 8;
        }

        // Small instances (lo_size ∈ {2, 4}) leave whole groups for the
        // scalar path; identical arithmetic, one group at a time.
        while x_lo + 2 <= lo_size {
            let mut rows = [[F128::ZERO; 4]; 2];
            let mut any = false;
            for half in 0..2 {
                let pair = x_lo + half;
                let x0l = 2 * pair;
                let x1l = x0l + 1;
                if ((pair_idx_base + pair) & pair_in_block_mask) >= useful_pairs_inclusive {
                    if WRITE {
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                    }
                    continue;
                }
                any = true;
                let x0g = row_base + x0l;
                let x1g = x0g + 1;
                rows[half] = fold_round2_pair_x86_unchecked_8(
                    table_data,
                    a_pkt.add(x0g * 8),
                    a_pkt.add(x1g * 8),
                    b_pkt.add(x0g * 8),
                    b_pkt.add(x1g * 8),
                );
                if WRITE {
                    a_chunk[x0l] = rows[half][0];
                    a_chunk[x1l] = rows[half][1];
                    b_chunk[x0l] = rows[half][2];
                    b_chunk[x1l] = rows[half][3];
                }
            }
            if any {
                let [a0, a1, b0, b1] = rows[0];
                let [a2, a3, b2, b3] = rows[1];
                let wt = eq_lo[x_lo + 1];
                let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
                tail[0] ^= a1w.mul_unreduced(b1);
                tail[1] ^= (a0w + a1w).mul_unreduced(b0 + b1);
                tail[2] ^= a3w.mul_unreduced(b3);
                tail[3] ^= (a2w + a3w).mul_unreduced(b2 + b3);
                tail[4] ^= a2w.mul_unreduced(b2);
                let (e_aw, e_b) = (a0w + a2w, b0 + b2);
                let (o_aw, o_b) = (a1w + a3w, b1 + b3);
                tail[5] ^= e_aw.mul_unreduced(e_b);
                tail[6] ^= o_aw.mul_unreduced(o_b);
                tail[7] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
            }
            x_lo += 2;
        }

        let mut out = [F128::ZERO; 8];
        for i in 0..8 {
            tail[i] ^= acc[i].fold();
            out[i] = tail[i].reduce();
        }
        out
    }
}

/// x86 composed double fold (ρ₁ then ρ₂) plus round-four message for one
/// worker chunk. Every output group of four is materialized in registers,
/// stored once, and consumed for the message before it leaves registers.
///
/// # Safety
/// `a_in.len() == 4 · a_out.len()`, `b_in.len() == 4 · b_out.len()`,
/// `a_out.len() == 2 · eq_lo.len()`, `eq_lo.len()` even and ≥ 2. AVX-512F
/// and VPCLMULQDQ are cfg-gated.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(crate) unsafe fn fold2_and_message_x86_avx512(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho1: F128,
    rho2: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    debug_assert_eq!(a_in.len(), 4 * a_out.len());
    debug_assert_eq!(b_in.len(), 4 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());

    // Fold eight consecutive inputs at `src` into four outputs (one ZMM).
    #[inline(always)]
    unsafe fn fold_x4(
        src: *const F128,
        r: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
    ) -> __m512i {
        use core::arch::x86_64::*;
        // SAFETY: caller supplies eight readable F128 values at src.
        unsafe {
            let lo = _mm512_loadu_si512(src.cast::<__m512i>());
            let hi = _mm512_loadu_si512(src.add(4).cast::<__m512i>());
            fold_regs(lo, hi, r, even_idx, odd_idx)
        }
    }

    // Fold the eight values held in `lo ++ hi` (four consecutive pairs) into
    // four outputs: `even + r·(even + odd)`.
    #[inline(always)]
    unsafe fn fold_regs(
        lo: __m512i,
        hi: __m512i,
        r: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
    ) -> __m512i {
        use crate::field::gf2_128::x86_64::ghash_mul_x4;
        use core::arch::x86_64::*;
        // SAFETY: register-only; features cfg-gated.
        unsafe {
            let even = _mm512_permutex2var_epi64(lo, even_idx, hi);
            let odd = _mm512_permutex2var_epi64(lo, odd_idx, hi);
            _mm512_xor_si512(even, ghash_mul_x4(r, _mm512_xor_si512(even, odd)))
        }
    }

    // SAFETY: the function's length invariants bound all loads/stores and the
    // cfg gate supplies every intrinsic feature.
    unsafe {
        let r1 = _mm512_broadcast_i32x4(_mm_set_epi64x(rho1.hi as i64, rho1.lo as i64));
        let r2 = _mm512_broadcast_i32x4(_mm_set_epi64x(rho2.hi as i64, rho2.lo as i64));
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut p1_wide = WideGhashX4::zero();
        let mut pinf_wide = WideGhashX4::zero();
        let mut p1_tail = F256Unreduced::ZERO;
        let mut pinf_tail = F256Unreduced::ZERO;
        let mut x_lo = 0;

        while x_lo + 4 <= eq_lo.len() {
            let output = 2 * x_lo;
            let input = 4 * output;
            let a_src = a_in.as_ptr().add(input);
            let b_src = b_in.as_ptr().add(input);
            // Level 1 (ρ₁): 32 inputs → 16 values in four ZMMs.
            let ta0 = fold_x4(a_src, r1, even_idx, odd_idx);
            let ta1 = fold_x4(a_src.add(8), r1, even_idx, odd_idx);
            let ta2 = fold_x4(a_src.add(16), r1, even_idx, odd_idx);
            let ta3 = fold_x4(a_src.add(24), r1, even_idx, odd_idx);
            let tb0 = fold_x4(b_src, r1, even_idx, odd_idx);
            let tb1 = fold_x4(b_src.add(8), r1, even_idx, odd_idx);
            let tb2 = fold_x4(b_src.add(16), r1, even_idx, odd_idx);
            let tb3 = fold_x4(b_src.add(24), r1, even_idx, odd_idx);
            // Level 2 (ρ₂): 16 → 8 outputs in two ZMMs per array.
            let a_lo = fold_regs(ta0, ta1, r2, even_idx, odd_idx);
            let a_hi = fold_regs(ta2, ta3, r2, even_idx, odd_idx);
            let b_lo = fold_regs(tb0, tb1, r2, even_idx, odd_idx);
            let b_hi = fold_regs(tb2, tb3, r2, even_idx, odd_idx);

            _mm512_storeu_si512(a_out.as_mut_ptr().add(output).cast::<__m512i>(), a_lo);
            _mm512_storeu_si512(a_out.as_mut_ptr().add(output + 4).cast::<__m512i>(), a_hi);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output).cast::<__m512i>(), b_lo);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output + 4).cast::<__m512i>(), b_hi);

            let a0 = _mm512_permutex2var_epi64(a_lo, even_idx, a_hi);
            let a1 = _mm512_permutex2var_epi64(a_lo, odd_idx, a_hi);
            let b0 = _mm512_permutex2var_epi64(b_lo, even_idx, b_hi);
            let b1 = _mm512_permutex2var_epi64(b_lo, odd_idx, b_hi);
            let g1 = ghash_mul_x4(a1, b1);
            let g_inf = ghash_mul_x4(_mm512_xor_si512(a0, a1), _mm512_xor_si512(b0, b1));
            let eq = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
            p1_wide.mul_acc(eq, g1);
            pinf_wide.mul_acc(eq, g_inf);
            x_lo += 4;
        }

        // Power-of-two eq blocks leave either no tail or exactly two pairs.
        if x_lo < eq_lo.len() {
            debug_assert_eq!(eq_lo.len() - x_lo, 2);
            let output = 2 * x_lo;
            let input = 4 * output;
            let a_src = a_in.as_ptr().add(input);
            let b_src = b_in.as_ptr().add(input);
            let ta0 = fold_x4(a_src, r1, even_idx, odd_idx);
            let ta1 = fold_x4(a_src.add(8), r1, even_idx, odd_idx);
            let tb0 = fold_x4(b_src, r1, even_idx, odd_idx);
            let tb1 = fold_x4(b_src.add(8), r1, even_idx, odd_idx);
            let a_folded = fold_regs(ta0, ta1, r2, even_idx, odd_idx);
            let b_folded = fold_regs(tb0, tb1, r2, even_idx, odd_idx);
            _mm512_storeu_si512(a_out.as_mut_ptr().add(output).cast::<__m512i>(), a_folded);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output).cast::<__m512i>(), b_folded);

            for lane in 0..2 {
                let o = output + 2 * lane;
                let a0 = a_out[o];
                let a1 = a_out[o + 1];
                let b0 = b_out[o];
                let b1 = b_out[o + 1];
                let eq = eq_lo[x_lo + lane];
                p1_tail ^= eq.mul_unreduced(a1 * b1);
                pinf_tail ^= eq.mul_unreduced((a0 + a1) * (b0 + b1));
            }
        }

        p1_tail ^= p1_wide.fold();
        pinf_tail ^= pinf_wide.fold();
        (p1_tail.reduce(), pinf_tail.reduce())
    }
}

/// x86 cascade step for one worker chunk: composed double fold (ρ_a then
/// ρ_b), the round message split by pair parity, and the six next-round
/// aggregates — all on the group's shared odd-lane weight (see
/// `fold2_plain_and_round_pair_lookahead_into`). Returns
/// `[p1_even, pinf_even, p1_odd, pinf_odd, W0', W3', W4', W5']`, each reduced.
///
/// Per four groups (eight pairs, sixteen outputs, sixty-four inputs per
/// array): sixteen level-1 folds and eight level-2 folds materialize the four
/// output groups in four ZMMs, which are stored once and then transposed in
/// registers (eight `vshufi64x2`) so that row `k` of all four groups sits in
/// one ZMM; four reduced `w`-prescalings and eight unreduced products follow,
/// exactly as in the round-two sweep kernel.
///
/// # Safety
/// `a_in.len() == 4 · a_out.len()`, `b_in.len() == 4 · b_out.len()`,
/// `a_out.len() == 2 · eq_lo.len()`, `eq_lo.len()` even and ≥ 2. AVX-512F
/// and VPCLMULQDQ are cfg-gated.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(crate) unsafe fn fold2_and_message_lookahead_x86_avx512(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
) -> [F128; 8] {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    let lo_size = eq_lo.len();
    debug_assert_eq!(a_in.len(), 4 * a_out.len());
    debug_assert_eq!(b_in.len(), 4 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * lo_size);
    debug_assert!(lo_size.is_multiple_of(2));

    #[inline(always)]
    unsafe fn fold_regs(
        lo: __m512i,
        hi: __m512i,
        r: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
    ) -> __m512i {
        use crate::field::gf2_128::x86_64::ghash_mul_x4;
        use core::arch::x86_64::*;
        // SAFETY: register-only; features cfg-gated.
        unsafe {
            let even = _mm512_permutex2var_epi64(lo, even_idx, hi);
            let odd = _mm512_permutex2var_epi64(lo, odd_idx, hi);
            _mm512_xor_si512(even, ghash_mul_x4(r, _mm512_xor_si512(even, odd)))
        }
    }

    // Composed fold of sixteen consecutive inputs at `src` into one output
    // group of four (one ZMM).
    #[inline(always)]
    unsafe fn fold16_to_4(
        src: *const F128,
        ra: __m512i,
        rb: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
    ) -> __m512i {
        use core::arch::x86_64::*;
        // SAFETY: caller supplies sixteen readable F128 at src.
        unsafe {
            let i0 = _mm512_loadu_si512(src.cast::<__m512i>());
            let i1 = _mm512_loadu_si512(src.add(4).cast::<__m512i>());
            let i2 = _mm512_loadu_si512(src.add(8).cast::<__m512i>());
            let i3 = _mm512_loadu_si512(src.add(12).cast::<__m512i>());
            let t0 = fold_regs(i0, i1, ra, even_idx, odd_idx);
            let t1 = fold_regs(i2, i3, ra, even_idx, odd_idx);
            fold_regs(t0, t1, rb, even_idx, odd_idx)
        }
    }

    // 4×4 transpose of 128-bit lanes: rows `o0..o3` (one output group each)
    // → `[a0, a1, a2, a3]` with `ak` = row k of every group.
    #[inline(always)]
    unsafe fn transpose4(o0: __m512i, o1: __m512i, o2: __m512i, o3: __m512i) -> [__m512i; 4] {
        use core::arch::x86_64::*;
        // SAFETY: register-only; features cfg-gated.
        unsafe {
            let q0 = _mm512_shuffle_i64x2::<0x44>(o0, o1); // o0.0 o0.1 o1.0 o1.1
            let q1 = _mm512_shuffle_i64x2::<0xEE>(o0, o1); // o0.2 o0.3 o1.2 o1.3
            let q2 = _mm512_shuffle_i64x2::<0x44>(o2, o3);
            let q3 = _mm512_shuffle_i64x2::<0xEE>(o2, o3);
            [
                _mm512_shuffle_i64x2::<0x88>(q0, q2), // o0.0 o1.0 o2.0 o3.0
                _mm512_shuffle_i64x2::<0xDD>(q0, q2), // o0.1 o1.1 o2.1 o3.1
                _mm512_shuffle_i64x2::<0x88>(q1, q3), // o0.2 o1.2 o2.2 o3.2
                _mm512_shuffle_i64x2::<0xDD>(q1, q3), // o0.3 o1.3 o2.3 o3.3
            ]
        }
    }

    // SAFETY: the function's length invariants bound all loads/stores and the
    // cfg gate supplies every intrinsic feature.
    unsafe {
        let ra = _mm512_broadcast_i32x4(_mm_set_epi64x(rho_a.hi as i64, rho_a.lo as i64));
        let rb = _mm512_broadcast_i32x4(_mm_set_epi64x(rho_b.hi as i64, rho_b.lo as i64));
        let rho_ab = rho_a * rho_b;
        let rarb = _mm512_broadcast_i32x4(_mm_set_epi64x(rho_ab.hi as i64, rho_ab.lo as i64));
        let defer = zc_fold_defer_enabled();
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut acc = [WideGhashX4::zero(); 8];
        let mut tail = [F256Unreduced::ZERO; 8];
        let mut x_lo = 0;

        while x_lo + 8 <= lo_size {
            let output = 2 * x_lo;
            let input = 4 * output;
            let a_src = a_in.as_ptr().add(input);
            let b_src = b_in.as_ptr().add(input);
            let (oa0, oa1, oa2, oa3, ob0, ob1, ob2, ob3) = if defer {
                (
                    fold16_to_4_deferred(a_src, ra, rb, rarb),
                    fold16_to_4_deferred(a_src.add(16), ra, rb, rarb),
                    fold16_to_4_deferred(a_src.add(32), ra, rb, rarb),
                    fold16_to_4_deferred(a_src.add(48), ra, rb, rarb),
                    fold16_to_4_deferred(b_src, ra, rb, rarb),
                    fold16_to_4_deferred(b_src.add(16), ra, rb, rarb),
                    fold16_to_4_deferred(b_src.add(32), ra, rb, rarb),
                    fold16_to_4_deferred(b_src.add(48), ra, rb, rarb),
                )
            } else {
                (
                    fold16_to_4(a_src, ra, rb, even_idx, odd_idx),
                    fold16_to_4(a_src.add(16), ra, rb, even_idx, odd_idx),
                    fold16_to_4(a_src.add(32), ra, rb, even_idx, odd_idx),
                    fold16_to_4(a_src.add(48), ra, rb, even_idx, odd_idx),
                    fold16_to_4(b_src, ra, rb, even_idx, odd_idx),
                    fold16_to_4(b_src.add(16), ra, rb, even_idx, odd_idx),
                    fold16_to_4(b_src.add(32), ra, rb, even_idx, odd_idx),
                    fold16_to_4(b_src.add(48), ra, rb, even_idx, odd_idx),
                )
            };
            let ap = a_out.as_mut_ptr().add(output);
            let bp = b_out.as_mut_ptr().add(output);
            _mm512_storeu_si512(ap.cast::<__m512i>(), oa0);
            _mm512_storeu_si512(ap.add(4).cast::<__m512i>(), oa1);
            _mm512_storeu_si512(ap.add(8).cast::<__m512i>(), oa2);
            _mm512_storeu_si512(ap.add(12).cast::<__m512i>(), oa3);
            _mm512_storeu_si512(bp.cast::<__m512i>(), ob0);
            _mm512_storeu_si512(bp.add(4).cast::<__m512i>(), ob1);
            _mm512_storeu_si512(bp.add(8).cast::<__m512i>(), ob2);
            _mm512_storeu_si512(bp.add(12).cast::<__m512i>(), ob3);

            let [a0, a1, a2, a3] = transpose4(oa0, oa1, oa2, oa3);
            let [b0, b1, b2, b3] = transpose4(ob0, ob1, ob2, ob3);
            let e_lo = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
            let e_hi = f128x4_loadu(eq_lo.as_ptr().add(x_lo + 4));
            let w = _mm512_permutex2var_epi64(e_lo, odd_idx, e_hi);
            let a0w = ghash_mul_x4(w, a0);
            let a1w = ghash_mul_x4(w, a1);
            let a2w = ghash_mul_x4(w, a2);
            let a3w = ghash_mul_x4(w, a3);
            acc[0].mul_acc(a1w, b1);
            acc[1].mul_acc(_mm512_xor_si512(a0w, a1w), _mm512_xor_si512(b0, b1));
            acc[2].mul_acc(a3w, b3);
            acc[3].mul_acc(_mm512_xor_si512(a2w, a3w), _mm512_xor_si512(b2, b3));
            acc[4].mul_acc(a2w, b2);
            let e_aw = _mm512_xor_si512(a0w, a2w);
            let e_b = _mm512_xor_si512(b0, b2);
            let o_aw = _mm512_xor_si512(a1w, a3w);
            let o_b = _mm512_xor_si512(b1, b3);
            acc[5].mul_acc(e_aw, e_b);
            acc[6].mul_acc(o_aw, o_b);
            acc[7].mul_acc(_mm512_xor_si512(e_aw, o_aw), _mm512_xor_si512(e_b, o_b));
            x_lo += 8;
        }

        // Small instances (lo_size ∈ {2, 4, 6}) leave whole groups: fold one
        // group (sixteen inputs) at a time and finish it in scalar.
        while x_lo + 2 <= lo_size {
            let output = 2 * x_lo;
            let input = 4 * output;
            let oa = fold16_to_4(a_in.as_ptr().add(input), ra, rb, even_idx, odd_idx);
            let ob = fold16_to_4(b_in.as_ptr().add(input), ra, rb, even_idx, odd_idx);
            _mm512_storeu_si512(a_out.as_mut_ptr().add(output).cast::<__m512i>(), oa);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output).cast::<__m512i>(), ob);
            let (a0, a1, a2, a3) = (a_out[output], a_out[output + 1], a_out[output + 2], a_out[output + 3]);
            let (b0, b1, b2, b3) = (b_out[output], b_out[output + 1], b_out[output + 2], b_out[output + 3]);
            let wt = eq_lo[x_lo + 1];
            let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
            tail[0] ^= a1w.mul_unreduced(b1);
            tail[1] ^= (a0w + a1w).mul_unreduced(b0 + b1);
            tail[2] ^= a3w.mul_unreduced(b3);
            tail[3] ^= (a2w + a3w).mul_unreduced(b2 + b3);
            tail[4] ^= a2w.mul_unreduced(b2);
            let (e_aw, e_b) = (a0w + a2w, b0 + b2);
            let (o_aw, o_b) = (a1w + a3w, b1 + b3);
            tail[5] ^= e_aw.mul_unreduced(e_b);
            tail[6] ^= o_aw.mul_unreduced(o_b);
            tail[7] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
            x_lo += 2;
        }

        let mut out = [F128::ZERO; 8];
        for i in 0..8 {
            tail[i] ^= acc[i].fold();
            out[i] = tail[i].reduce();
        }
        out
    }
}

/// `FLOCK_NO_ZC_FOLD_NT=1` restores plain write-allocate stores for the
/// cascade fold outputs (exact same-binary A/B); the ranked worker's cleared
/// env never sets it.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(crate) fn zc_fold_nt_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_FOLD_NT").is_none());
    *ON
}

/// `FLOCK_NO_ZC_REGFOLD=1` restores the scalar staging arms (stack arrays
/// re-read as ZMM loads) in the round-2 and rounds-3+4 kernels (exact
/// same-binary A/B); the ranked worker's cleared env never sets it.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(crate) fn zc_regfold_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_REGFOLD").is_none());
    *ON
}

/// `FLOCK_NO_ZC_FOLD_DEFER=1` restores fully-reduced multiplies in the
/// composed (rho_a, rho_b) pair folds (exact same-binary A/B); the ranked
/// worker's cleared env never sets it.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(crate) fn zc_fold_defer_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_FOLD_DEFER").is_none());
    *ON
}

/// Deferred-reduction form of the sixteen-to-four composed pair fold. The
/// two fold levels expand (char 2) to
/// `out = x0 ^ ra*(x0^x1) ^ rb*(x0^x2) ^ (ra*rb)*(x0^x1^x2^x3)` per output
/// lane, with `[x0..x3] = transpose4` of the four input ZMMs — three
/// constant multiplies on independent operands, so the unreduced 256-bit
/// products XOR-accumulate and reduce ONCE per lane (reduction mod the
/// fixed irreducible is F2-linear): 14 CLMULs instead of 18, identical
/// shuffle count. Bit-identical to `fold16_to_4` by the same argument
/// `WideGhashX4` rests on.
///
/// # Safety
/// Sixteen readable F128 at `src`; avx512f + vpclmulqdq (module-gated,
/// restated here).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[inline(always)]
unsafe fn fold16_to_4_deferred(
    src: *const F128,
    ra: core::arch::x86_64::__m512i,
    rb: core::arch::x86_64::__m512i,
    rarb: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;
    // SAFETY: bounds per the contract; features per the cfg above.
    unsafe {
        let i0 = _mm512_loadu_si512(src.cast::<__m512i>());
        let i1 = _mm512_loadu_si512(src.add(4).cast::<__m512i>());
        let i2 = _mm512_loadu_si512(src.add(8).cast::<__m512i>());
        let i3 = _mm512_loadu_si512(src.add(12).cast::<__m512i>());
        let [x0, x1, x2, x3] = transpose4_lanes(i0, i1, i2, i3);
        let x01 = _mm512_xor_si512(x0, x1);
        let x02 = _mm512_xor_si512(x0, x2);
        let x0123 = _mm512_xor_si512(x01, _mm512_xor_si512(x2, x3));
        let mut acc = WideGhashX4::zero();
        acc.mul_acc(ra, x01);
        acc.mul_acc(rb, x02);
        acc.mul_acc(rarb, x0123);
        _mm512_xor_si512(x0, acc.reduce_lanes())
    }
}

/// Store one ZMM as four XMM non-temporal quarters. Large pool allocations
/// land 16 mod 64, so a 64-byte-aligned ZMM stream is unreachable; `F128`
/// is `repr(C, align(16))`, so every `Vec<F128>` base — and every F128
/// element offset from it — is 16-byte aligned by the allocation layout
/// (a language guarantee, not malloc folklore).
///
/// # Safety
/// `p` must be 16-byte aligned and cover 4 F128s; avx512f is required (the
/// module gate supplies it, and the explicit cfg keeps that visible here).
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn stream_zmm_as_xmm4(p: *mut F128, v: core::arch::x86_64::__m512i) {
    use core::arch::x86_64::*;
    // SAFETY: alignment per the contract; features per the cfg above.
    unsafe {
        let d = p as *mut __m128i;
        _mm_stream_si128(d, _mm512_extracti32x4_epi32::<0>(v));
        _mm_stream_si128(d.add(1), _mm512_extracti32x4_epi32::<1>(v));
        _mm_stream_si128(d.add(2), _mm512_extracti32x4_epi32::<2>(v));
        _mm_stream_si128(d.add(3), _mm512_extracti32x4_epi32::<3>(v));
    }
}

/// 4x4 transpose of 128-bit lanes at module level (the kernels' nested
/// copies stay for their local uses): rows -> columns of four ZMMs.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn transpose4_lanes(
    o0: core::arch::x86_64::__m512i,
    o1: core::arch::x86_64::__m512i,
    o2: core::arch::x86_64::__m512i,
    o3: core::arch::x86_64::__m512i,
) -> [core::arch::x86_64::__m512i; 4] {
    use core::arch::x86_64::*;
    // SAFETY: register-only; features per the cfg above.
    unsafe {
        let q0 = _mm512_shuffle_i64x2::<0x44>(o0, o1);
        let q1 = _mm512_shuffle_i64x2::<0xEE>(o0, o1);
        let q2 = _mm512_shuffle_i64x2::<0x44>(o2, o3);
        let q3 = _mm512_shuffle_i64x2::<0xEE>(o2, o3);
        [
            _mm512_shuffle_i64x2::<0x88>(q0, q2),
            _mm512_shuffle_i64x2::<0xDD>(q0, q2),
            _mm512_shuffle_i64x2::<0x88>(q1, q3),
            _mm512_shuffle_i64x2::<0xDD>(q1, q3),
        ]
    }
}

/// x86 no-materialize composed pass for one worker chunk: re-derives the
/// round-two folded rows from the packed witness through the same byte-table
/// gathers as the sweep (`fold_round2_pair_x86_unchecked_8`), folds them at
/// ρ₁ then ρ₂ in registers, stores each output group of four once, and
/// accumulates the parity-split round-four message plus the six round-five
/// aggregates exactly like `fold2_and_message_lookahead_x86_avx512`.
///
/// Composed output `x` (global) ← packed rows `4x..4x+4` = pairs `2x, 2x+1`;
/// a pair skipped by `round2_pair_skip` contributes zero rows (what the
/// materializing sweep wrote there), so the outputs are the sweep's tables
/// folded twice, bit for bit.
///
/// # Safety
/// `table_data` must point to the 8 × 256 `F128` fold table; `a_pkt`/`b_pkt`
/// must expose 8 readable bytes for every row `4·out_base ..
/// 4·(out_base + a_out.len())`; `a_out.len() == b_out.len() == 2·eq_lo.len()`,
/// `eq_lo.len()` even and ≥ 2. AVX-512F and VPCLMULQDQ are cfg-gated.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold2_from_packed_lookahead_x86_avx512(
    table_data: *const F128,
    mats: Option<&[u64; 128]>,
    a_pkt: *const u8,
    b_pkt: *const u8,
    out_base: usize,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho1: F128,
    rho2: F128,
    eq_lo: &[F128],
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    nt_out: bool,
) -> [F128; 8] {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    let lo_size = eq_lo.len();
    debug_assert_eq!(a_out.len(), 2 * lo_size);
    debug_assert_eq!(b_out.len(), 2 * lo_size);
    debug_assert!(lo_size.is_multiple_of(2));

    #[inline(always)]
    unsafe fn fold_regs(
        lo: __m512i,
        hi: __m512i,
        r: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
    ) -> __m512i {
        use crate::field::gf2_128::x86_64::ghash_mul_x4;
        use core::arch::x86_64::*;
        // SAFETY: register-only; features cfg-gated.
        unsafe {
            let even = _mm512_permutex2var_epi64(lo, even_idx, hi);
            let odd = _mm512_permutex2var_epi64(lo, odd_idx, hi);
            _mm512_xor_si512(even, ghash_mul_x4(r, _mm512_xor_si512(even, odd)))
        }
    }

    /// Sixteen contiguous F128 rows -> four outputs via the (rho_a, rho_b)
    /// pair folds — the register form of `group_from_packed`'s two levels
    /// (local twin of the cascade kernel's helper of the same name).
    #[inline(always)]
    unsafe fn fold16_to_4(
        src: *const F128,
        ra: __m512i,
        rb: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
    ) -> __m512i {
        use core::arch::x86_64::*;
        // SAFETY: caller supplies sixteen readable F128 at src.
        unsafe {
            let i0 = _mm512_loadu_si512(src.cast::<__m512i>());
            let i1 = _mm512_loadu_si512(src.add(4).cast::<__m512i>());
            let i2 = _mm512_loadu_si512(src.add(8).cast::<__m512i>());
            let i3 = _mm512_loadu_si512(src.add(12).cast::<__m512i>());
            let t0 = fold_regs(i0, i1, ra, even_idx, odd_idx);
            let t1 = fold_regs(i2, i3, ra, even_idx, odd_idx);
            fold_regs(t0, t1, rb, even_idx, odd_idx)
        }
    }

    #[inline(always)]
    unsafe fn transpose4(o0: __m512i, o1: __m512i, o2: __m512i, o3: __m512i) -> [__m512i; 4] {
        use core::arch::x86_64::*;
        // SAFETY: register-only; features cfg-gated.
        unsafe {
            let q0 = _mm512_shuffle_i64x2::<0x44>(o0, o1);
            let q1 = _mm512_shuffle_i64x2::<0xEE>(o0, o1);
            let q2 = _mm512_shuffle_i64x2::<0x44>(o2, o3);
            let q3 = _mm512_shuffle_i64x2::<0xEE>(o2, o3);
            [
                _mm512_shuffle_i64x2::<0x88>(q0, q2),
                _mm512_shuffle_i64x2::<0xDD>(q0, q2),
                _mm512_shuffle_i64x2::<0x88>(q1, q3),
                _mm512_shuffle_i64x2::<0xDD>(q1, q3),
            ]
        }
    }

    // One output group (four composed outputs = eight pairs = sixteen rows
    // per array) from packed rows: gathers → ρ₁ folds → ρ₂ fold. Returns
    // `(a_group, b_group)` as ZMMs.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn group_from_packed(
        table_data: *const F128,
        a_pkt: *const u8,
        b_pkt: *const u8,
        x0: usize,
        r1: __m512i,
        r2: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
        pair_in_block_mask: usize,
        useful_pairs_inclusive: usize,
        cache: Option<(&[F128; 64], &[F128; 64], usize)>,
    ) -> (__m512i, __m512i) {
        use core::arch::x86_64::*;
        // SAFETY: caller bounds rows 4·x0 .. 4·x0 + 16.
        unsafe {
            let mut ae = [F128::ZERO; 8];
            let mut ao = [F128::ZERO; 8];
            let mut be = [F128::ZERO; 8];
            let mut bo = [F128::ZERO; 8];
            for p in 0..8 {
                let pair = 2 * x0 + p;
                if (pair & pair_in_block_mask) >= useful_pairs_inclusive {
                    continue;
                }
                let r0 = 2 * pair;
                let folded = if let Some((fa, fb, cache_base)) = cache {
                    let i = r0 - cache_base;
                    [fa[i], fa[i + 1], fb[i], fb[i + 1]]
                } else {
                    fold_round2_pair_x86_unchecked_8(
                        table_data,
                        a_pkt.add(r0 * 8),
                        a_pkt.add((r0 + 1) * 8),
                        b_pkt.add(r0 * 8),
                        b_pkt.add((r0 + 1) * 8),
                    )
                };
                ae[p] = folded[0];
                ao[p] = folded[1];
                be[p] = folded[2];
                bo[p] = folded[3];
            }
            // Level 1 (ρ₁): eight pairs → eight values, four per ZMM.
            let ae_lo = f128x4_loadu(ae.as_ptr());
            let ao_lo = f128x4_loadu(ao.as_ptr());
            let ae_hi = f128x4_loadu(ae.as_ptr().add(4));
            let ao_hi = f128x4_loadu(ao.as_ptr().add(4));
            let ta_lo = _mm512_xor_si512(ae_lo, ghash_mul_x4(r1, _mm512_xor_si512(ae_lo, ao_lo)));
            let ta_hi = _mm512_xor_si512(ae_hi, ghash_mul_x4(r1, _mm512_xor_si512(ae_hi, ao_hi)));
            let be_lo = f128x4_loadu(be.as_ptr());
            let bo_lo = f128x4_loadu(bo.as_ptr());
            let be_hi = f128x4_loadu(be.as_ptr().add(4));
            let bo_hi = f128x4_loadu(bo.as_ptr().add(4));
            let tb_lo = _mm512_xor_si512(be_lo, ghash_mul_x4(r1, _mm512_xor_si512(be_lo, bo_lo)));
            let tb_hi = _mm512_xor_si512(be_hi, ghash_mul_x4(r1, _mm512_xor_si512(be_hi, bo_hi)));
            // Level 2 (ρ₂): eight → four outputs.
            (
                fold_regs(ta_lo, ta_hi, r2, even_idx, odd_idx),
                fold_regs(tb_lo, tb_hi, r2, even_idx, odd_idx),
            )
        }
    }

    // SAFETY: the function's contract bounds every packed-row read, table
    // read and output store; the cfg gate supplies every intrinsic feature.
    unsafe {
        let r1 = _mm512_broadcast_i32x4(_mm_set_epi64x(rho1.hi as i64, rho1.lo as i64));
        let r2 = _mm512_broadcast_i32x4(_mm_set_epi64x(rho2.hi as i64, rho2.lo as i64));
        let rho12 = rho1 * rho2;
        let r12 = _mm512_broadcast_i32x4(_mm_set_epi64x(rho12.hi as i64, rho12.lo as i64));
        let defer = zc_fold_defer_enabled();
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut acc = [WideGhashX4::zero(); 8];
        let mut tail = [F256Unreduced::ZERO; 8];
        let mut x_lo = 0;

        // GFNI batch fold: each iteration's four groups consume exactly 64
        // consecutive rows per side (rows 4·xg .. 4·xg+64) — one bit-matrix
        // batch per side per iteration.
        let use_batch =
            cfg!(all(target_feature = "avx512vbmi", target_feature = "gfni")) && mats.is_some();
        let mut fa = [F128::ZERO; 64];
        let mut fb = [F128::ZERO; 64];
        while x_lo + 8 <= lo_size {
            let ol = 2 * x_lo; // local output index of group 0
            let xg = out_base + ol;
            let cache = if use_batch {
                #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
                {
                    let m = mats.unwrap();
                    gfni_fold64_rows(a_pkt.add(4 * xg * 8), m, fa.as_mut_ptr());
                    gfni_fold64_rows(b_pkt.add(4 * xg * 8), m, fb.as_mut_ptr());
                }
                Some((&fa, &fb, 4 * xg))
            } else {
                None
            };
            let cache = cache.map(|(a, b, base)| (&*a, &*b, base));
            // Batch path: the 64 cached rows are CONTIGUOUS folded rows, and
            // group_from_packed's (ρ₁, ρ₂) two-level pair fold over them is
            // verbatim `fold16_to_4` — register loads + permutes instead of
            // 128 scalar 16-byte stores re-read as store-forwarding-blocked
            // ZMM loads. Padded pairs need no branch: their raw rows are
            // zero in memory and every fold table maps 0 → 0, so the cached
            // row is already the zero the scalar path wrote explicitly.
            let (oa0, ob0, oa1, ob1, oa2, ob2, oa3, ob3) =
                if let Some((fa, fb, cache_base)) = cache.filter(|_| zc_regfold_enabled()) {
                    debug_assert_eq!(cache_base, 4 * xg);
                    let _ = cache_base;
                    let ap = fa.as_ptr();
                    let bp2 = fb.as_ptr();
                    if defer {
                        (
                            fold16_to_4_deferred(ap, r1, r2, r12),
                            fold16_to_4_deferred(bp2, r1, r2, r12),
                            fold16_to_4_deferred(ap.add(16), r1, r2, r12),
                            fold16_to_4_deferred(bp2.add(16), r1, r2, r12),
                            fold16_to_4_deferred(ap.add(32), r1, r2, r12),
                            fold16_to_4_deferred(bp2.add(32), r1, r2, r12),
                            fold16_to_4_deferred(ap.add(48), r1, r2, r12),
                            fold16_to_4_deferred(bp2.add(48), r1, r2, r12),
                        )
                    } else {
                        (
                            fold16_to_4(ap, r1, r2, even_idx, odd_idx),
                            fold16_to_4(bp2, r1, r2, even_idx, odd_idx),
                            fold16_to_4(ap.add(16), r1, r2, even_idx, odd_idx),
                            fold16_to_4(bp2.add(16), r1, r2, even_idx, odd_idx),
                            fold16_to_4(ap.add(32), r1, r2, even_idx, odd_idx),
                            fold16_to_4(bp2.add(32), r1, r2, even_idx, odd_idx),
                            fold16_to_4(ap.add(48), r1, r2, even_idx, odd_idx),
                            fold16_to_4(bp2.add(48), r1, r2, even_idx, odd_idx),
                        )
                    }
                } else {
                    let (oa0, ob0) = group_from_packed(table_data, a_pkt, b_pkt, xg, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive, cache);
                    let (oa1, ob1) = group_from_packed(table_data, a_pkt, b_pkt, xg + 4, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive, cache);
                    let (oa2, ob2) = group_from_packed(table_data, a_pkt, b_pkt, xg + 8, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive, cache);
                    let (oa3, ob3) = group_from_packed(table_data, a_pkt, b_pkt, xg + 12, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive, cache);
                    (oa0, ob0, oa1, ob1, oa2, ob2, oa3, ob3)
                };
            let ap = a_out.as_mut_ptr().add(ol);
            let bp = b_out.as_mut_ptr().add(ol);
            // The round message below is computed from the same registers, so
            // the outputs are write-once here; their next reader is the NEXT
            // cascade level, after a Fiat–Shamir round trip — DRAM-cold at
            // the shapes the caller gates `nt_out` on. NT stores skip the
            // write-allocate RFO (~512 MiB/proof at the ranked shape).
            if nt_out {
                stream_zmm_as_xmm4(ap, oa0);
                stream_zmm_as_xmm4(ap.add(4), oa1);
                stream_zmm_as_xmm4(ap.add(8), oa2);
                stream_zmm_as_xmm4(ap.add(12), oa3);
                stream_zmm_as_xmm4(bp, ob0);
                stream_zmm_as_xmm4(bp.add(4), ob1);
                stream_zmm_as_xmm4(bp.add(8), ob2);
                stream_zmm_as_xmm4(bp.add(12), ob3);
            } else {
                _mm512_storeu_si512(ap.cast::<__m512i>(), oa0);
                _mm512_storeu_si512(ap.add(4).cast::<__m512i>(), oa1);
                _mm512_storeu_si512(ap.add(8).cast::<__m512i>(), oa2);
                _mm512_storeu_si512(ap.add(12).cast::<__m512i>(), oa3);
                _mm512_storeu_si512(bp.cast::<__m512i>(), ob0);
                _mm512_storeu_si512(bp.add(4).cast::<__m512i>(), ob1);
                _mm512_storeu_si512(bp.add(8).cast::<__m512i>(), ob2);
                _mm512_storeu_si512(bp.add(12).cast::<__m512i>(), ob3);
            }

            let [a0, a1, a2, a3] = transpose4(oa0, oa1, oa2, oa3);
            let [b0, b1, b2, b3] = transpose4(ob0, ob1, ob2, ob3);
            let e_lo = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
            let e_hi = f128x4_loadu(eq_lo.as_ptr().add(x_lo + 4));
            let w = _mm512_permutex2var_epi64(e_lo, odd_idx, e_hi);
            let a0w = ghash_mul_x4(w, a0);
            let a1w = ghash_mul_x4(w, a1);
            let a2w = ghash_mul_x4(w, a2);
            let a3w = ghash_mul_x4(w, a3);
            acc[0].mul_acc(a1w, b1);
            acc[1].mul_acc(_mm512_xor_si512(a0w, a1w), _mm512_xor_si512(b0, b1));
            acc[2].mul_acc(a3w, b3);
            acc[3].mul_acc(_mm512_xor_si512(a2w, a3w), _mm512_xor_si512(b2, b3));
            acc[4].mul_acc(a2w, b2);
            let e_aw = _mm512_xor_si512(a0w, a2w);
            let e_b = _mm512_xor_si512(b0, b2);
            let o_aw = _mm512_xor_si512(a1w, a3w);
            let o_b = _mm512_xor_si512(b1, b3);
            acc[5].mul_acc(e_aw, e_b);
            acc[6].mul_acc(o_aw, o_b);
            acc[7].mul_acc(_mm512_xor_si512(e_aw, o_aw), _mm512_xor_si512(e_b, o_b));
            x_lo += 8;
        }

        // Small instances leave whole groups: one group at a time, scalar finish.
        while x_lo + 2 <= lo_size {
            let ol = 2 * x_lo;
            let (oa, ob) = group_from_packed(table_data, a_pkt, b_pkt, out_base + ol, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive, None);
            _mm512_storeu_si512(a_out.as_mut_ptr().add(ol).cast::<__m512i>(), oa);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(ol).cast::<__m512i>(), ob);
            let (a0, a1, a2, a3) = (a_out[ol], a_out[ol + 1], a_out[ol + 2], a_out[ol + 3]);
            let (b0, b1, b2, b3) = (b_out[ol], b_out[ol + 1], b_out[ol + 2], b_out[ol + 3]);
            let wt = eq_lo[x_lo + 1];
            let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
            tail[0] ^= a1w.mul_unreduced(b1);
            tail[1] ^= (a0w + a1w).mul_unreduced(b0 + b1);
            tail[2] ^= a3w.mul_unreduced(b3);
            tail[3] ^= (a2w + a3w).mul_unreduced(b2 + b3);
            tail[4] ^= a2w.mul_unreduced(b2);
            let (e_aw, e_b) = (a0w + a2w, b0 + b2);
            let (o_aw, o_b) = (a1w + a3w, b1 + b3);
            tail[5] ^= e_aw.mul_unreduced(e_b);
            tail[6] ^= o_aw.mul_unreduced(o_b);
            tail[7] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
            x_lo += 2;
        }

        if nt_out {
            // Drain the WC buffers before this chunk's task returns; the
            // caller's rayon reduce is the next level's happens-before edge.
            _mm_sfence();
        }
        let mut out = [F128::ZERO; 8];
        for i in 0..8 {
            tail[i] ^= acc[i].fold();
            out[i] = tail[i].reduce();
        }
        out
    }
}

/// [`build_uni_skip_fold_mats`] over any 8×256-entry XOR-composed byte-table
/// block (the cascade K pass feeds its λ-scaled tables through this too —
/// scaling the basis scales every composed entry exactly, so the matrices
/// of a scaled table are just the scaled-basis matrices).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
pub(crate) fn build_row_fold_mats(data: &[F128]) -> [u64; 128] {
    debug_assert_eq!(data.len(), 8 * 256);
    let mut mats = [0u64; 128];
    for j in 0..8 {
        let basis: [F128; 8] = std::array::from_fn(|bit| data[j * 256 + (1 << bit)]);
        for k in 0..16 {
            let mut qword = 0u64;
            for i in 0..8 {
                let bit_index = 8 * k + i;
                let mut row = 0u8;
                for (b, basis_val) in basis.iter().enumerate() {
                    let bit = if bit_index < 64 {
                        (basis_val.lo >> bit_index) & 1
                    } else {
                        (basis_val.hi >> (bit_index - 64)) & 1
                    };
                    row |= (bit as u8) << b;
                }
                qword |= (row as u64) << (8 * (7 - i));
            }
            mats[j * 16 + k] = qword;
        }
    }
    mats
}

/// Fold 64 consecutive packed 8-byte rows through the univariate-skip byte
/// tables in one GFNI batch: `out[r] = Σ_j T_j[rows[8r + j]]`, bit-identical
/// to eight gathers per row (same XOR terms, reassociated).
///
/// Pipeline: per-ZMM 8×8 byte transpose (`vpermb`), 8×8 qword transpose
/// (`vpunpck` + two `vpermt2q` stages) to chunk-byte planes, 8 GFNI products
/// per output-byte plane folded with `vpternlogq`, then the inverse
/// transposes to reassemble row-major F128s. Zero table loads; the working
/// set is the 1 KiB matrix block.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512vbmi,gfni")]
pub(crate) unsafe fn gfni_fold64_rows(rows: *const u8, mats: &[u64; 128], out: *mut F128) {
    use core::arch::x86_64::*;
    // SAFETY: caller guarantees 512 readable bytes at `rows` and 64 writable
    // F128s at `out`.
    unsafe {
        let mut z = [_mm512_setzero_si512(); 8];
        for (i, slot) in z.iter_mut().enumerate() {
            *slot = _mm512_loadu_si512(rows.add(64 * i) as *const __m512i);
        }
        gfni_fold64_regs(z, mats, out);
    }
}

/// [`gfni_fold64_rows`] with the 64 rows already in registers (qword q of
/// `z[i]` = row 8i+q), for callers that assemble row batches with their own
/// permutations (the cascade K pass splits interleaved L1/L3 delta rows).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512vbmi,gfni")]
pub(crate) unsafe fn gfni_fold64_regs(
    z: [core::arch::x86_64::__m512i; 8],
    mats: &[u64; 128],
    out: *mut F128,
) {
    use core::arch::x86_64::*;
    // SAFETY (whole body): caller guarantees 64 writable F128s at `out`;
    // all shuffle indices are in range.
    unsafe {
        // 8×8 byte transpose inside each ZMM: result qword j = byte j of the
        // ZMM's eight rows (idx[8j + i] = 8i + j).
        #[rustfmt::skip]
        const BT: [i8; 64] = [
            0, 8, 16, 24, 32, 40, 48, 56,  1, 9, 17, 25, 33, 41, 49, 57,
            2, 10, 18, 26, 34, 42, 50, 58,  3, 11, 19, 27, 35, 43, 51, 59,
            4, 12, 20, 28, 36, 44, 52, 60,  5, 13, 21, 29, 37, 45, 53, 61,
            6, 14, 22, 30, 38, 46, 54, 62,  7, 15, 23, 31, 39, 47, 55, 63,
        ];
        let bt = _mm512_loadu_si512(BT.as_ptr() as *const __m512i);
        // vpermt2q index vectors for the two combining stages of the 8×8
        // qword transpose (and its inverse — the network is an involution).
        let s2_lo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        let s2_hi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
        let s3_lo = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        let s3_hi = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

        // qword_transpose(t0..t7) -> p0..p7 with p_j.qword[i] = t_i.qword[j].
        let qword_transpose = |t: [__m512i; 8]| -> [__m512i; 8] {
            let e01 = _mm512_unpacklo_epi64(t[0], t[1]);
            let o01 = _mm512_unpackhi_epi64(t[0], t[1]);
            let e23 = _mm512_unpacklo_epi64(t[2], t[3]);
            let o23 = _mm512_unpackhi_epi64(t[2], t[3]);
            let e45 = _mm512_unpacklo_epi64(t[4], t[5]);
            let o45 = _mm512_unpackhi_epi64(t[4], t[5]);
            let e67 = _mm512_unpacklo_epi64(t[6], t[7]);
            let o67 = _mm512_unpackhi_epi64(t[6], t[7]);
            // Halves for planes {0,2}, {4,6}, {1,3}, {5,7} over t0..t3 / t4..t7.
            let h02_a = _mm512_permutex2var_epi64(e01, s2_lo, e23);
            let h46_a = _mm512_permutex2var_epi64(e01, s2_hi, e23);
            let h13_a = _mm512_permutex2var_epi64(o01, s2_lo, o23);
            let h57_a = _mm512_permutex2var_epi64(o01, s2_hi, o23);
            let h02_b = _mm512_permutex2var_epi64(e45, s2_lo, e67);
            let h46_b = _mm512_permutex2var_epi64(e45, s2_hi, e67);
            let h13_b = _mm512_permutex2var_epi64(o45, s2_lo, o67);
            let h57_b = _mm512_permutex2var_epi64(o45, s2_hi, o67);
            [
                _mm512_permutex2var_epi64(h02_a, s3_lo, h02_b), // plane 0
                _mm512_permutex2var_epi64(h13_a, s3_lo, h13_b), // plane 1
                _mm512_permutex2var_epi64(h02_a, s3_hi, h02_b), // plane 2
                _mm512_permutex2var_epi64(h13_a, s3_hi, h13_b), // plane 3
                _mm512_permutex2var_epi64(h46_a, s3_lo, h46_b), // plane 4
                _mm512_permutex2var_epi64(h57_a, s3_lo, h57_b), // plane 5
                _mm512_permutex2var_epi64(h46_a, s3_hi, h46_b), // plane 6
                _mm512_permutex2var_epi64(h57_a, s3_hi, h57_b), // plane 7
            ]
        };

        // Input: 8 ZMMs of 8 rows each -> byte-transposed -> chunk planes.
        let mut t = [_mm512_setzero_si512(); 8];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = _mm512_permutexvar_epi8(bt, z[i]);
        }
        let p = qword_transpose(t);

        // Sixteen output-byte planes: eight GFNI products folded per plane.
        let mut acc = [_mm512_setzero_si512(); 16];
        for (k, slot) in acc.iter_mut().enumerate() {
            let g = |j: usize| {
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    p[j],
                    _mm512_set1_epi64(mats[j * 16 + k] as i64),
                )
            };
            let v1 = _mm512_ternarylogic_epi64::<0x96>(g(0), g(1), g(2));
            let v2 = _mm512_ternarylogic_epi64::<0x96>(g(3), g(4), g(5));
            let v3 = _mm512_ternarylogic_epi64::<0x96>(g(6), g(7), v1);
            *slot = _mm512_xor_si512(v2, v3);
        }

        // Reassemble: inverse qword transpose + inverse byte transpose per
        // half, then interleave lo/hi qwords into row-major F128s.
        let lo_half = qword_transpose(acc[..8].try_into().unwrap());
        let hi_half = qword_transpose(acc[8..].try_into().unwrap());
        let il_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        let il_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        for i in 0..8 {
            let lo = _mm512_permutexvar_epi8(bt, lo_half[i]);
            let hi = _mm512_permutexvar_epi8(bt, hi_half[i]);
            let out_ptr = (out as *mut u8).add(128 * i) as *mut __m512i;
            _mm512_storeu_si512(out_ptr, _mm512_permutex2var_epi64(lo, il_lo, hi));
            _mm512_storeu_si512(
                out_ptr.add(1),
                _mm512_permutex2var_epi64(lo, il_hi, hi),
            );
        }
    }
}
