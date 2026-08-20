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
    wtab: Option<&[F128]>,
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
        let wsplit = zc_wsplit_enabled();
        let mut tail = [F256Unreduced::ZERO; 8];
        let mut x_lo = 0;
        // GFNI batch fold: 32 consecutive pairs = 64 consecutive rows per
        // side prefolded in one bit-matrix batch (padded pairs fold zero
        // rows; the consume path below skips them exactly as before).
        let use_batch = cfg!(all(target_feature = "avx512vbmi", target_feature = "gfni"))
            && mats.is_some()
            && lo_size >= 32
            && lo_size.is_multiple_of(32);
        let mut fa_store = FoldCache([F128::ZERO; 64]);
        let mut fb_store = FoldCache([F128::ZERO; 64]);
        let fa = &mut fa_store.0;
        let fb = &mut fb_store.0;
        // Packed-row prefetch distance, resolved once per worker chunk
        // (never inside the refill / message loops).
        let pf_tiles = if zc_pkt_pf_far_enabled() {
            ZC_PKT_PF_TILES
        } else {
            1
        };

        // Residue-major refill (WRITE=false only — the chunk-store arm needs
        // row order): the prefold emits directly in the a_k/b_k register
        // grouping, deleting both consumer lane transposes per iteration.
        // Gated to shapes the 8-pair batch loop consumes exhaustively: the
        // scalar tail (and the regfold-off arm) index the cache BY ROW and
        // must never see the residue-major layout.
        #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
        let tr_emit = !WRITE
            && use_batch
            && lo_size.is_multiple_of(8)
            && zc_regfold_enabled()
            && zc_r2_tr_enabled();
        while x_lo + 8 <= lo_size {
            #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
            if use_batch && x_lo.is_multiple_of(32) {
                // Refill both caches for pairs x_lo..x_lo+32 (rows
                // row_base+2*x_lo .. +64 per side) — the same bytes the
                // gather path reads.
                let m = mats.unwrap();
                let g0 = row_base + 2 * x_lo;
                // `g0 = 2 · (pair_idx_base + x_lo)` is the tile's global row
                // start, so its block position decides the dead lines.
                let dead = super::super::prefold_dead_line_mask_gated(
                    g0,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                );
                if tr_emit {
                    gfni_fold64_rows_masked_tr(a_pkt.add(g0 * 8), m, fa.as_mut_ptr(), dead);
                    gfni_fold64_rows_masked_tr(b_pkt.add(g0 * 8), m, fb.as_mut_ptr(), dead);
                } else {
                    gfni_fold64_rows_masked(a_pkt.add(g0 * 8), m, fa.as_mut_ptr(), dead);
                    gfni_fold64_rows_masked(b_pkt.add(g0 * 8), m, fb.as_mut_ptr(), dead);
                }
                // The packed bursts `pf_tiles` refills ahead — a gap the
                // hardware prefetcher does not bridge across the strided
                // burst boundary. Pull both sides in now (prefetch is a
                // hint: `wrapping_add` keeps the past-the-end address on
                // the last tiles well defined, and the hint is dropped).
                if zc_pkt_pf_enabled() {
                    let next = (g0 + 64 * pf_tiles) * 8;
                    for l in 0..8 {
                        _mm_prefetch(
                            a_pkt.wrapping_add(next + 64 * l).cast::<i8>(),
                            core::arch::x86_64::_MM_HINT_T0,
                        );
                        _mm_prefetch(
                            b_pkt.wrapping_add(next + 64 * l).cast::<i8>(),
                            core::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
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
                if tr_emit {
                    // Residue-major cache: a_k for this window is the
                    // contiguous ZMM at F128 index 16k + r0/4 — the exact
                    // registers transpose4_lanes used to produce.
                    let g = r0 / 4;
                    let ld = |base: *const F128, k: usize| {
                        _mm512_loadu_si512(base.add(16 * k + g).cast::<__m512i>())
                    };
                    (
                        ld(fa.as_ptr(), 0),
                        ld(fa.as_ptr(), 1),
                        ld(fa.as_ptr(), 2),
                        ld(fa.as_ptr(), 3),
                        ld(fb.as_ptr(), 0),
                        ld(fb.as_ptr(), 1),
                        ld(fb.as_ptr(), 2),
                        ld(fb.as_ptr(), 3),
                    )
                } else {
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
                }
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
            let (a0w, a1w, a2w, a3w) = if let Some(wt) = wtab {
                // (w, w·x⁶⁴) precomputed once per pass: both are pure
                // functions of `x_lo` (the odd eq_lo lanes and their x⁶⁴
                // companions), yet the incumbent recomputed the companion —
                // a permute plus a CLMUL of pure latency — at the head of
                // the chain feeding all eight accumulates, every iteration.
                let wp = wt.as_ptr().add(x_lo) as *const __m512i;
                let w = _mm512_loadu_si512(wp);
                let w64 = _mm512_loadu_si512(wp.add(1));
                (
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a0, w, w64),
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a1, w, w64),
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a2, w, w64),
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a3, w, w64),
                )
            } else {
                let e_lo = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
                let e_hi = f128x4_loadu(eq_lo.as_ptr().add(x_lo + 4));
                let w = _mm512_permutex2var_epi64(e_lo, odd_idx, e_hi);
                if wsplit {
                    let w64 = crate::field::gf2_128::x86_64::ghash_shift64_x4(w);
                    (
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a0, w, w64),
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a1, w, w64),
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a2, w, w64),
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a3, w, w64),
                    )
                } else {
                    (
                        ghash_mul_x4(w, a0),
                        ghash_mul_x4(w, a1),
                        ghash_mul_x4(w, a2),
                        ghash_mul_x4(w, a3),
                    )
                }
            };
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
#[inline(never)]
unsafe fn fold2_and_message_lookahead_x86_avx512_leaf<const PRECOMPUTED_W: bool>(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
    wtab: &[F128],
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
        let wsplit = zc_wsplit_enabled();
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
            let (a0w, a1w, a2w, a3w) = if PRECOMPUTED_W {
                // (w, w·x⁶⁴) precomputed once per pass: both are pure
                // functions of `x_lo` (the odd eq_lo lanes and their x⁶⁴
                // companions), yet the incumbent recomputed the companion —
                // a permute plus a CLMUL of pure latency — at the head of
                // the chain feeding all eight accumulates, every iteration.
                let wp = wtab.as_ptr().add(x_lo) as *const __m512i;
                let w = _mm512_loadu_si512(wp);
                let w64 = _mm512_loadu_si512(wp.add(1));
                (
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a0, w, w64),
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a1, w, w64),
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a2, w, w64),
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a3, w, w64),
                )
            } else {
                let e_lo = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
                let e_hi = f128x4_loadu(eq_lo.as_ptr().add(x_lo + 4));
                let w = _mm512_permutex2var_epi64(e_lo, odd_idx, e_hi);
                if wsplit {
                    let w64 = crate::field::gf2_128::x86_64::ghash_shift64_x4(w);
                    (
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a0, w, w64),
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a1, w, w64),
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a2, w, w64),
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a3, w, w64),
                    )
                } else {
                    (
                        ghash_mul_x4(w, a0),
                        ghash_mul_x4(w, a1),
                        ghash_mul_x4(w, a2),
                        ghash_mul_x4(w, a3),
                    )
                }
            };
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

/// Select the precomputed-weight or on-demand-weight cascade leaf once per
/// worker chunk. The ranked path always supplies `Some`; keeping the choice
/// outside the `x_lo` loop lets LLVM give each leaf its own spill schedule.
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
    wtab: Option<&[F128]>,
) -> [F128; 8] {
    // SAFETY: both leaves have the public function's exact slice contract;
    // the borrowed weight slice remains live for the synchronous leaf call.
    unsafe {
        match wtab {
            Some(wt) => fold2_and_message_lookahead_x86_avx512_leaf::<true>(
                a_in, b_in, a_out, b_out, rho_a, rho_b, eq_lo, wt,
            ),
            None => fold2_and_message_lookahead_x86_avx512_leaf::<false>(
                a_in, b_in, a_out, b_out, rho_a, rho_b, eq_lo, eq_lo,
            ),
        }
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

/// `FLOCK_NO_ZC_WSPLIT=1` restores the fully-reduced `w·a_k` prescale
/// multiplies in the round-2 / rounds-3+4 message blocks (exact same-binary
/// A/B). The split form (`ghash_mul_x4_split` with `w·x^64` hoisted once per
/// iteration) is field-identical — reduction mod p is a ring homomorphism,
/// so `a·w = a.lo·w + a.hi·(w·x^64 mod p)` — and replaces the two-stage
/// recursive reduce on the chain feeding all eight accumulators with one:
/// 22 CLMUL + 1 shift per iteration instead of 24 + 8.
/// `FLOCK_NO_ZC_R2_TR_EMIT=1` restores the row-major prefold cache and the
/// consumer lane transposes in the round-2 message sweep (exact same-binary
/// A/B; the residue-major emit is a pure reordering of identical values).
pub(crate) fn zc_r2_tr_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R2_TR_EMIT").is_none());
    *ON
}

/// `FLOCK_NO_ZC_PKT_PF=1` disables the next-tile software prefetch of the
/// packed a/b bursts in the round-2 and rounds-3+4 fold kernels (exact
/// same-binary A/B; prefetch is architecturally invisible).
pub(crate) fn zc_pkt_pf_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_PKT_PF").is_none());
    *ON
}

/// Look-ahead, in 64-row tiles, for the packed-row prefetch in the round-2
/// and rounds-3+4 fold kernels. One tile is 512 bytes per side (64 rows ×
/// 8 packed bytes) and is exactly one refill of the `fa`/`fb` fold caches.
///
/// The incumbent value was 1 — the tile the very next refill demand-loads.
/// In the rounds-3+4 pass (`fold2_from_packed_lookahead_x86_avx512`) the
/// refill runs on EVERY iteration, so one tile of look-ahead is one loop
/// body: the prefetch is issued at 1d43xx, roughly half-way through a body
/// that retires ~1 700 uops per iteration (1 049 inline instructions plus
/// two `gfni_fold64_rows_masked_c4` calls), which leaves only the ~860
/// post-prefetch instructions — ~500 cycles — between the hint and the
/// demand load that needs it. That is the same order as a loaded DRAM miss
/// on this box, so the miss is still partly in flight when the next refill
/// asks for it.
///
/// Three tiles puts a full two extra iterations (~2 400 uops) behind the
/// hint while keeping every tile covered — tile `T+3` is requested at
/// iteration `T`, so each tile is still prefetched exactly once, just
/// earlier. Nothing is added and nothing moves: the same sixteen
/// `prefetcht0` instructions issue at the same point, only the address
/// changes, and a prefetch has no architectural effect, so the folded
/// values are bit-identical either way.
const ZC_PKT_PF_TILES: usize = 3;

/// `FLOCK_NO_ZC_PKT_PF_FAR=1` restores the incumbent one-tile look-ahead in
/// both packed-row prefetches (exact same-binary A/B).
pub(crate) fn zc_pkt_pf_far_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_PKT_PF_FAR").is_none());
    *ON
}

/// `FLOCK_NO_ZC_WTAB=1` disables the hoisted per-pass `(w, w·x⁶⁴)` pair
/// table in the zerocheck message sweeps (exact same-binary A/B; the table
/// holds the identical values the sweep otherwise derives per iteration).
pub(crate) fn zc_wtab_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_WTAB").is_none());
    *ON
}

pub(crate) fn zc_wsplit_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_WSPLIT").is_none());
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
    // SAFETY: alignment per the contract; features per the cfg above. At
    // 64-alignment (the allocator's recyclable class on this lineage) one
    // single-uop ZMM stream publishes the whole line.
    unsafe {
        if p as usize % 64 == 0 {
            _mm512_stream_si512(p as *mut __m512i, v);
            return;
        }
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
    cfold: Option<&CFoldMats>,
    wtab: Option<&[F128]>,
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
        let wsplit = zc_wsplit_enabled();
        let mut tail = [F256Unreduced::ZERO; 8];
        let mut x_lo = 0;

        // GFNI batch fold: each iteration's four groups consume exactly 64
        // consecutive rows per side (rows 4·xg .. 4·xg+64) — one bit-matrix
        // batch per side per iteration.
        let use_batch =
            cfg!(all(target_feature = "avx512vbmi", target_feature = "gfni")) && mats.is_some();
        // Composed-fold baking: the (ρ₁, ρ₂) pair fold is `out = Σ_k c_k ·
        // fold(row 4x+k)` with `c = (1+ρ₁)(1+ρ₂), ρ₁(1+ρ₂), (1+ρ₁)ρ₂, ρ₁ρ₂`.
        // Every `c_k` rides inside the batch's bit matrices, so the three
        // constant multiplies per output (14 CLMUL per sixteen rows) and the
        // lane transpose they fed both disappear; the composed output is one
        // XOR of four ZMM-strided reads of the residue-major fold.
        let use_c4 = use_batch && cfold.is_some() && zc_regfold_enabled();
        // Packed-row prefetch distance, resolved once per worker chunk
        // (never inside the refill loop).
        let pf_tiles = if zc_pkt_pf_far_enabled() {
            ZC_PKT_PF_TILES
        } else {
            1
        };
        let mut fa_store = FoldCache([F128::ZERO; 64]);
        let mut fb_store = FoldCache([F128::ZERO; 64]);
        let fa = &mut fa_store.0;
        let fb = &mut fb_store.0;
        while x_lo + 8 <= lo_size {
            let ol = 2 * x_lo; // local output index of group 0
            let xg = out_base + ol;
            let cache = if use_batch {
                #[cfg(all(target_feature = "avx512vbmi", target_feature = "gfni"))]
                {
                    // `4·xg` is the tile's global row start (output x ← rows
                    // 4x..4x+4), so its block position decides the dead lines.
                    let dead = super::super::prefold_dead_line_mask_gated(
                        4 * xg,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                    );
                    if use_c4 {
                        let c = cfold.unwrap();
                        gfni_fold64_rows_masked_c4(
                            a_pkt.add(4 * xg * 8),
                            c,
                            fa.as_mut_ptr(),
                            dead,
                        );
                        gfni_fold64_rows_masked_c4(
                            b_pkt.add(4 * xg * 8),
                            c,
                            fb.as_mut_ptr(),
                            dead,
                        );
                    } else {
                        let m = mats.unwrap();
                        gfni_fold64_rows_masked(a_pkt.add(4 * xg * 8), m, fa.as_mut_ptr(), dead);
                        gfni_fold64_rows_masked(b_pkt.add(4 * xg * 8), m, fb.as_mut_ptr(), dead);
                    }
                    // The 512-byte bursts `pf_tiles` refills ahead of the
                    // consumer — see the round-2 twin for the rationale.
                    // Same addresses on both refill arms.
                    if zc_pkt_pf_enabled() {
                        let next = (4 * xg + 64 * pf_tiles) * 8;
                        for l in 0..8 {
                            _mm_prefetch(
                                a_pkt.wrapping_add(next + 64 * l).cast::<i8>(),
                                core::arch::x86_64::_MM_HINT_T0,
                            );
                            _mm_prefetch(
                                b_pkt.wrapping_add(next + 64 * l).cast::<i8>(),
                                core::arch::x86_64::_MM_HINT_T0,
                            );
                        }
                    }
                }
                Some((&*fa, &*fb, 4 * xg))
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
            let (oa0, ob0, oa1, ob1, oa2, ob2, oa3, ob3) = if use_c4 {
                // `out[16·k + t] = c_k · fold(row 4·t + k)`, so group `t` is
                // the XOR of the four residue planes at the same lane offset
                // — output order already correct, no transpose.
                let xor4 = |p: *const F128, g: usize| -> __m512i {
                    let q = p.add(4 * g);
                    let v0 = _mm512_loadu_si512(q.cast::<__m512i>());
                    let v1 = _mm512_loadu_si512(q.add(16).cast::<__m512i>());
                    let v2 = _mm512_loadu_si512(q.add(32).cast::<__m512i>());
                    let v3 = _mm512_loadu_si512(q.add(48).cast::<__m512i>());
                    _mm512_xor_si512(
                        _mm512_ternarylogic_epi64::<0x96>(v0, v1, v2),
                        v3,
                    )
                };
                let ap = fa.as_ptr();
                let bp2 = fb.as_ptr();
                (
                    xor4(ap, 0),
                    xor4(bp2, 0),
                    xor4(ap, 1),
                    xor4(bp2, 1),
                    xor4(ap, 2),
                    xor4(bp2, 2),
                    xor4(ap, 3),
                    xor4(bp2, 3),
                )
            } else if let Some((fa, fb, cache_base)) = cache.filter(|_| zc_regfold_enabled()) {
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
            let (a0w, a1w, a2w, a3w) = if let Some(wt) = wtab {
                // (w, w·x⁶⁴) precomputed once per pass: both are pure
                // functions of `x_lo` (the odd eq_lo lanes and their x⁶⁴
                // companions), yet the incumbent recomputed the companion —
                // a permute plus a CLMUL of pure latency — at the head of
                // the chain feeding all eight accumulates, every iteration.
                let wp = wt.as_ptr().add(x_lo) as *const __m512i;
                let w = _mm512_loadu_si512(wp);
                let w64 = _mm512_loadu_si512(wp.add(1));
                (
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a0, w, w64),
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a1, w, w64),
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a2, w, w64),
                    crate::field::gf2_128::x86_64::ghash_mul_x4_split(a3, w, w64),
                )
            } else {
                let e_lo = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
                let e_hi = f128x4_loadu(eq_lo.as_ptr().add(x_lo + 4));
                let w = _mm512_permutex2var_epi64(e_lo, odd_idx, e_hi);
                if wsplit {
                    let w64 = crate::field::gf2_128::x86_64::ghash_shift64_x4(w);
                    (
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a0, w, w64),
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a1, w, w64),
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a2, w, w64),
                        crate::field::gf2_128::x86_64::ghash_mul_x4_split(a3, w, w64),
                    )
                } else {
                    (
                        ghash_mul_x4(w, a0),
                        ghash_mul_x4(w, a1),
                        ghash_mul_x4(w, a2),
                        ghash_mul_x4(w, a3),
                    )
                }
            };
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
    let cols: [F128; 64] = std::array::from_fn(|i| {
        let byte = i / 8;
        let bit = i % 8;
        data[byte * 256 + (1 << bit)]
    });
    build_row_fold_mats_from_cols(&cols)
}

/// Build the same eight-byte GFNI matrix block directly from the 64 basis
/// columns of the F2-linear map. This skips expanding those columns into
/// eight 256-entry subset-sum tables when the only consumer is GFNI.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
pub(crate) fn build_row_fold_mats_from_cols(cols: &[F128]) -> [u64; 128] {
    debug_assert_eq!(cols.len(), 64);
    let mut mats = [0u64; 128];
    for j in 0..8 {
        let basis = &cols[j * 8..j * 8 + 8];
        let lo_lanes: [u64; 8] = std::array::from_fn(|b| basis[b].lo);
        let hi_lanes: [u64; 8] = std::array::from_fn(|b| basis[b].hi);
        let mut lo_bytes = [0u8; 64];
        let mut hi_bytes = [0u8; 64];
        // This shared primitive is the proven 8-lane -> 64-column bit
        // transpose used by lincheck's GFNI matrix builder.
        crate::bits::transpose_8_u64s_to_64_bytes(&lo_lanes, &mut lo_bytes);
        crate::bits::transpose_8_u64s_to_64_bytes(&hi_lanes, &mut hi_bytes);
        for c in 0..8 {
            let lo: [u8; 8] = lo_bytes[c * 8..c * 8 + 8].try_into().unwrap();
            let hi: [u8; 8] = hi_bytes[c * 8..c * 8 + 8].try_into().unwrap();
            // GFNI stores output row i at byte 7-i.
            mats[j * 16 + c] = u64::from_le_bytes(lo).swap_bytes();
            mats[j * 16 + c + 8] = u64::from_le_bytes(hi).swap_bytes();
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
// Retained as the unpredicated reference the dead-line skip is proved
// byte-identical to (`gfni_masked_prefold_matches_unpredicated_kernel`); the
// hot paths call `gfni_fold64_rows_masked`.
#[cfg_attr(not(test), allow(dead_code))]
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

/// Fold two independent packed-u64 row maps into one batch of 64 F128s.
/// This is the DirectFold8 basis materializer's natural half-pair: low/high
/// halves of one claim, or the corresponding halves of a second claim.
/// Both input transposes feed one output-plane accumulation and one inverse
/// transpose. With `ADD`, the reassembled values are XORed with an existing
/// 64-row batch while still in registers.
///
/// # Safety
/// Each row pointer covers 512 bytes, `out` covers 64 F128s, and when `ADD`
/// is true `add` covers 64 F128s. AVX-512F/VBMI/GFNI are required.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512vbmi,gfni")]
pub(crate) unsafe fn gfni_fold64_two_maps<const ADD: bool>(
    rows0: *const u8,
    mats0: &[u64; 128],
    rows1: *const u8,
    mats1: &[u64; 128],
    out: *mut F128,
    add: *const F128,
) {
    use core::arch::x86_64::*;
    // SAFETY: caller supplies all pointer extents and target features.
    unsafe {
        #[rustfmt::skip]
        const BT: [i8; 64] = [
            0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57,
            2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
            4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
            6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
        ];
        let bt = _mm512_loadu_si512(BT.as_ptr() as *const __m512i);
        let s2_lo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        let s2_hi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
        let s3_lo = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        let s3_hi = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
        let qword_transpose = |t: [__m512i; 8]| -> [__m512i; 8] {
            let e01 = _mm512_unpacklo_epi64(t[0], t[1]);
            let o01 = _mm512_unpackhi_epi64(t[0], t[1]);
            let e23 = _mm512_unpacklo_epi64(t[2], t[3]);
            let o23 = _mm512_unpackhi_epi64(t[2], t[3]);
            let e45 = _mm512_unpacklo_epi64(t[4], t[5]);
            let o45 = _mm512_unpackhi_epi64(t[4], t[5]);
            let e67 = _mm512_unpacklo_epi64(t[6], t[7]);
            let o67 = _mm512_unpackhi_epi64(t[6], t[7]);
            let h02_a = _mm512_permutex2var_epi64(e01, s2_lo, e23);
            let h46_a = _mm512_permutex2var_epi64(e01, s2_hi, e23);
            let h13_a = _mm512_permutex2var_epi64(o01, s2_lo, o23);
            let h57_a = _mm512_permutex2var_epi64(o01, s2_hi, o23);
            let h02_b = _mm512_permutex2var_epi64(e45, s2_lo, e67);
            let h46_b = _mm512_permutex2var_epi64(e45, s2_hi, e67);
            let h13_b = _mm512_permutex2var_epi64(o45, s2_lo, o67);
            let h57_b = _mm512_permutex2var_epi64(o45, s2_hi, o67);
            [
                _mm512_permutex2var_epi64(h02_a, s3_lo, h02_b),
                _mm512_permutex2var_epi64(h13_a, s3_lo, h13_b),
                _mm512_permutex2var_epi64(h02_a, s3_hi, h02_b),
                _mm512_permutex2var_epi64(h13_a, s3_hi, h13_b),
                _mm512_permutex2var_epi64(h46_a, s3_lo, h46_b),
                _mm512_permutex2var_epi64(h57_a, s3_lo, h57_b),
                _mm512_permutex2var_epi64(h46_a, s3_hi, h46_b),
                _mm512_permutex2var_epi64(h57_a, s3_hi, h57_b),
            ]
        };
        let input_planes = |rows: *const u8| -> [__m512i; 8] {
            let z: [__m512i; 8] = core::array::from_fn(|i| {
                _mm512_loadu_si512(rows.add(64 * i) as *const __m512i)
            });
            let t = z.map(|v| _mm512_permutexvar_epi8(bt, v));
            qword_transpose(t)
        };
        let map_plane = |p: &[__m512i; 8], mats: &[u64; 128], k: usize| {
            let g = |j: usize| {
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    p[j],
                    _mm512_set1_epi64(mats[j * 16 + k] as i64),
                )
            };
            let v1 = _mm512_ternarylogic_epi64::<0x96>(g(0), g(1), g(2));
            let v2 = _mm512_ternarylogic_epi64::<0x96>(g(3), g(4), g(5));
            let v3 = _mm512_ternarylogic_epi64::<0x96>(g(6), g(7), v1);
            _mm512_xor_si512(v2, v3)
        };
        // Form the first map's 16 output-byte planes before loading the
        // second map. Keeping both eight-plane inputs live while creating
        // sixteen accumulators exceeds the SPR ZMM file once transpose
        // constants and temporaries are included, and LLVM then spills a
        // large fraction of the batch. Sequential accumulation has the same
        // GF(2) reassociation but caps the durable live set at 16 + 8 ZMMs.
        let (
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
        ) = {
            let p = input_planes(rows0);
            (
                map_plane(&p, mats0, 0),
                map_plane(&p, mats0, 1),
                map_plane(&p, mats0, 2),
                map_plane(&p, mats0, 3),
                map_plane(&p, mats0, 4),
                map_plane(&p, mats0, 5),
                map_plane(&p, mats0, 6),
                map_plane(&p, mats0, 7),
                map_plane(&p, mats0, 8),
                map_plane(&p, mats0, 9),
                map_plane(&p, mats0, 10),
                map_plane(&p, mats0, 11),
                map_plane(&p, mats0, 12),
                map_plane(&p, mats0, 13),
                map_plane(&p, mats0, 14),
                map_plane(&p, mats0, 15),
            )
        };
        let (
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
        ) = {
            let p = input_planes(rows1);
            (
                _mm512_xor_si512(a0, map_plane(&p, mats1, 0)),
                _mm512_xor_si512(a1, map_plane(&p, mats1, 1)),
                _mm512_xor_si512(a2, map_plane(&p, mats1, 2)),
                _mm512_xor_si512(a3, map_plane(&p, mats1, 3)),
                _mm512_xor_si512(a4, map_plane(&p, mats1, 4)),
                _mm512_xor_si512(a5, map_plane(&p, mats1, 5)),
                _mm512_xor_si512(a6, map_plane(&p, mats1, 6)),
                _mm512_xor_si512(a7, map_plane(&p, mats1, 7)),
                _mm512_xor_si512(a8, map_plane(&p, mats1, 8)),
                _mm512_xor_si512(a9, map_plane(&p, mats1, 9)),
                _mm512_xor_si512(a10, map_plane(&p, mats1, 10)),
                _mm512_xor_si512(a11, map_plane(&p, mats1, 11)),
                _mm512_xor_si512(a12, map_plane(&p, mats1, 12)),
                _mm512_xor_si512(a13, map_plane(&p, mats1, 13)),
                _mm512_xor_si512(a14, map_plane(&p, mats1, 14)),
                _mm512_xor_si512(a15, map_plane(&p, mats1, 15)),
            )
        };

        let lo_half = qword_transpose([a0, a1, a2, a3, a4, a5, a6, a7]);
        let hi_half = qword_transpose([a8, a9, a10, a11, a12, a13, a14, a15]);
        let il_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        let il_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        for i in 0..8 {
            let lo = _mm512_permutexvar_epi8(bt, lo_half[i]);
            let hi = _mm512_permutexvar_epi8(bt, hi_half[i]);
            let mut v0 = _mm512_permutex2var_epi64(lo, il_lo, hi);
            let mut v1 = _mm512_permutex2var_epi64(lo, il_hi, hi);
            if ADD {
                v0 = _mm512_xor_si512(
                    v0,
                    _mm512_loadu_si512(add.add(8 * i) as *const __m512i),
                );
                v1 = _mm512_xor_si512(
                    v1,
                    _mm512_loadu_si512(add.add(8 * i + 4) as *const __m512i),
                );
            }
            let out_ptr = out.add(8 * i) as *mut __m512i;
            _mm512_storeu_si512(out_ptr, v0);
            _mm512_storeu_si512(out_ptr.add(1), v1);
        }
    }
}

/// Fold four packed-u64 maps through two staged claim accumulations. Each
/// stage keeps only its two eight-plane inputs live and immediately stores
/// each completed output-byte plane. This replaces LLVM's accidental
/// 16-plane spills plus the inter-claim row-major temporary with one explicit
/// 16-ZMM plane buffer whose traffic is fixed and easy to audit.
///
/// # Safety
/// Each row pointer covers 512 bytes, `out` covers 64 F128s, `planes` covers
/// sixteen ZMMs, and the target features in the cfg are available.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512vbmi,gfni")]
pub(crate) unsafe fn gfni_fold64_four_maps_staged(
    rows0: *const u8,
    mats0: &[u64; 128],
    rows1: *const u8,
    mats1: &[u64; 128],
    rows2: *const u8,
    mats2: &[u64; 128],
    rows3: *const u8,
    mats3: &[u64; 128],
    out: *mut F128,
    planes: *mut core::arch::x86_64::__m512i,
) {
    use core::arch::x86_64::*;
    unsafe {
        #[rustfmt::skip]
        const BT: [i8; 64] = [
            0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57,
            2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
            4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
            6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
        ];
        let bt = _mm512_loadu_si512(BT.as_ptr().cast());
        let s2_lo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        let s2_hi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
        let s3_lo = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        let s3_hi = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
        let qword_transpose = |t: [__m512i; 8]| -> [__m512i; 8] {
            let e01 = _mm512_unpacklo_epi64(t[0], t[1]);
            let o01 = _mm512_unpackhi_epi64(t[0], t[1]);
            let e23 = _mm512_unpacklo_epi64(t[2], t[3]);
            let o23 = _mm512_unpackhi_epi64(t[2], t[3]);
            let e45 = _mm512_unpacklo_epi64(t[4], t[5]);
            let o45 = _mm512_unpackhi_epi64(t[4], t[5]);
            let e67 = _mm512_unpacklo_epi64(t[6], t[7]);
            let o67 = _mm512_unpackhi_epi64(t[6], t[7]);
            let h02_a = _mm512_permutex2var_epi64(e01, s2_lo, e23);
            let h46_a = _mm512_permutex2var_epi64(e01, s2_hi, e23);
            let h13_a = _mm512_permutex2var_epi64(o01, s2_lo, o23);
            let h57_a = _mm512_permutex2var_epi64(o01, s2_hi, o23);
            let h02_b = _mm512_permutex2var_epi64(e45, s2_lo, e67);
            let h46_b = _mm512_permutex2var_epi64(e45, s2_hi, e67);
            let h13_b = _mm512_permutex2var_epi64(o45, s2_lo, o67);
            let h57_b = _mm512_permutex2var_epi64(o45, s2_hi, o67);
            [
                _mm512_permutex2var_epi64(h02_a, s3_lo, h02_b),
                _mm512_permutex2var_epi64(h13_a, s3_lo, h13_b),
                _mm512_permutex2var_epi64(h02_a, s3_hi, h02_b),
                _mm512_permutex2var_epi64(h13_a, s3_hi, h13_b),
                _mm512_permutex2var_epi64(h46_a, s3_lo, h46_b),
                _mm512_permutex2var_epi64(h57_a, s3_lo, h57_b),
                _mm512_permutex2var_epi64(h46_a, s3_hi, h46_b),
                _mm512_permutex2var_epi64(h57_a, s3_hi, h57_b),
            ]
        };
        let input_planes = |rows: *const u8| -> [__m512i; 8] {
            let z: [__m512i; 8] = core::array::from_fn(|i| {
                _mm512_loadu_si512(rows.add(64 * i).cast())
            });
            qword_transpose(z.map(|v| _mm512_permutexvar_epi8(bt, v)))
        };
        let map_plane = |p: &[__m512i; 8], mats: &[u64; 128], k: usize| {
            let g = |j: usize| {
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    p[j],
                    _mm512_set1_epi64(mats[j * 16 + k] as i64),
                )
            };
            let v1 = _mm512_ternarylogic_epi64::<0x96>(g(0), g(1), g(2));
            let v2 = _mm512_ternarylogic_epi64::<0x96>(g(3), g(4), g(5));
            let v3 = _mm512_ternarylogic_epi64::<0x96>(g(6), g(7), v1);
            _mm512_xor_si512(v2, v3)
        };

        {
            let p0 = input_planes(rows0);
            let p1 = input_planes(rows1);
            for k in 0..16 {
                let value = _mm512_xor_si512(
                    map_plane(&p0, mats0, k),
                    map_plane(&p1, mats1, k),
                );
                _mm512_storeu_si512(planes.add(k), value);
            }
        }
        {
            let p2 = input_planes(rows2);
            let p3 = input_planes(rows3);
            for k in 0..16 {
                let value = _mm512_xor_si512(
                    map_plane(&p2, mats2, k),
                    map_plane(&p3, mats3, k),
                );
                let value = _mm512_xor_si512(value, _mm512_loadu_si512(planes.add(k)));
                _mm512_storeu_si512(planes.add(k), value);
            }
        }

        let acc: [__m512i; 16] = core::array::from_fn(|k| _mm512_loadu_si512(planes.add(k)));
        let lo_half = qword_transpose(acc[..8].try_into().unwrap());
        let hi_half = qword_transpose(acc[8..].try_into().unwrap());
        let il_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        let il_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        for i in 0..8 {
            let lo = _mm512_permutexvar_epi8(bt, lo_half[i]);
            let hi = _mm512_permutexvar_epi8(bt, hi_half[i]);
            let v0 = _mm512_permutex2var_epi64(lo, il_lo, hi);
            let v1 = _mm512_permutex2var_epi64(lo, il_hi, hi);
            let out_ptr = out.add(8 * i).cast::<__m512i>();
            _mm512_storeu_si512(out_ptr, v0);
            _mm512_storeu_si512(out_ptr.add(1), v1);
        }
    }
}

/// [`gfni_fold64_rows`] with the tile's provably-all-padding 64-byte lines
/// left unread: bit `i` of `dead_lines` replaces the load of rows
/// `8i ..= 8i+7` with a zero register, so that cache line is never fetched.
///
/// **Byte-identical to [`gfni_fold64_rows`] on the plans this tree builds.**
/// The GF(2) affine prefold is *per byte lane and per row*: after the two
/// transposes, plane `p[j]` holds row `n`'s chunk-`j` byte at byte position
/// `n`, and `_mm512_gf2p8affine_epi64_epi8::<0>` maps each byte independently
/// through an 8×8 GF(2) matrix with a zero constant term. So output row `n`
/// depends on input row `n` alone, and a zero input row yields
/// `M·0 = 0` in every one of the 16 output-byte planes — exactly what the
/// gather path produces for an all-zero row, since the byte tables are
/// XOR-composed from the set bits of the index and `T_j[0] = 0`. Two
/// consequences: substituting zeros can neither perturb any *other* row, nor
/// differ from folding a genuinely-zero row.
///
/// The caller obtains `dead_lines` from
/// `super::super::prefold_dead_line_mask_gated`, which only sets a bit when every
/// pair owning a row of that line is already skipped by the round-2 padding
/// predicate — so the substituted values are never read at all, and the
/// identity holds whatever the padding bytes contain.
///
/// # Safety
/// As [`gfni_fold64_rows`], except that the 64 bytes of a line whose
/// `dead_lines` bit is set need not be readable.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512vbmi,gfni")]
pub(crate) unsafe fn gfni_fold64_rows_masked(
    rows: *const u8,
    mats: &[u64; 128],
    out: *mut F128,
    dead_lines: u8,
) {
    use core::arch::x86_64::*;
    // SAFETY: caller guarantees 64 readable bytes at `rows.add(64 * i)` for
    // every line `i` not marked dead, and 64 writable F128s at `out`.
    unsafe {
        let mut z = [_mm512_setzero_si512(); 8];
        for (i, slot) in z.iter_mut().enumerate() {
            if dead_lines & (1u8 << i) == 0 {
                *slot = _mm512_loadu_si512(rows.add(64 * i) as *const __m512i);
            }
        }
        gfni_fold64_regs(z, mats, out);
    }
}

/// [`gfni_fold64_rows_masked`] emitting in **residue-major** order:
/// `out[16·k + t] = fold(row 4·t + k)` — the exact layout the round-2
/// message block consumes as its `a_k`/`b_k` register groups, so the
/// consumer's two 4×4 lane transposes (sixteen port-5 shuffles per
/// iteration, four iterations per refill) collapse into plain contiguous
/// loads. Achieved with the [`SIGMA_C4`] plane regroup the constant-baked
/// rounds-3+4 kernel already ships — eight `vpermb` per refill; the row
/// relabeling commutes with the per-row affine products, so the values are
/// identical, only the emit order changes.
///
/// # Safety
/// As [`gfni_fold64_rows_masked`].
#[cfg(all(target_feature = "avx512vbmi", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,avx512vbmi,gfni")]
pub(crate) unsafe fn gfni_fold64_rows_masked_tr(
    rows: *const u8,
    mats: &[u64; 128],
    out: *mut F128,
    dead_lines: u8,
) {
    use core::arch::x86_64::*;
    // SAFETY: as for the row-major form; SIGMA_C4 indices are in range.
    unsafe {
        let mut z = [_mm512_setzero_si512(); 8];
        for (i, slot) in z.iter_mut().enumerate() {
            if dead_lines & (1u8 << i) == 0 {
                *slot = _mm512_loadu_si512(rows.add(64 * i) as *const __m512i);
            }
        }
        gfni_fold64_regs_sigma(z, mats, out);
    }
}

/// Per-residue matrix set for [`gfni_fold64_rows_masked_c4`].
///
/// Entry `j * 16 + k` is the ZMM of eight 8×8 GF(2) matrices the batch feeds
/// `vgf2p8affineqb` for input chunk `j`, output byte `k`; qword `i` carries
/// 64-byte-aligned backing for the per-worker prefold caches `fa`/`fb`:
/// every refill store and every consume load of these buffers is ZMM-wide,
/// and `[F128; 64]`'s natural 16-byte alignment lets those 64-byte accesses
/// straddle cache lines (a split load/store each time the base is not
/// line-aligned).
#[repr(C, align(64))]
struct FoldCache([F128; 64]);

/// the matrix of residue `i / 2` (the layout [`SIGMA_C4`] produces).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[repr(C, align(64))]
pub(crate) struct CFoldMats(pub(crate) [[u64; 8]; 128]);

/// [`build_row_fold_mats`] for the table scaled by `c`.
///
/// The table is XOR-composed from its eight per-chunk basis entries, so
/// scaling the basis scales every composed entry — the matrices of `c ·
/// table` are exactly the matrices built from `c ·  basis`, and folding a row
/// through them yields `c · fold(row)` as an exact field identity.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[allow(dead_code)]
fn build_row_fold_mats_scaled(data: &[F128], c: F128) -> [u64; 128] {
    debug_assert_eq!(data.len(), 8 * 256);
    let mut mats = [0u64; 128];
    for j in 0..8 {
        let basis: [F128; 8] = std::array::from_fn(|bit| c * data[j * 256 + (1 << bit)]);
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

/// Build the four-residue matrix set for coefficients `coeffs[0..4]`.
///
/// `coeffs[k]` is the constant the composed fold applies to row `4x + k`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[allow(dead_code)]
pub(crate) fn build_cfold_mats(data: &[F128], coeffs: [F128; 4]) -> CFoldMats {
    let per: [[u64; 128]; 4] = std::array::from_fn(|c| build_row_fold_mats_scaled(data, coeffs[c]));
    let mut out = CFoldMats([[0u64; 8]; 128]);
    for idx in 0..128 {
        for i in 0..8 {
            // Qword `i` of the affine operand serves qword `i` of the plane,
            // whose eight rows all carry residue `i / 2` after `SIGMA_C4`.
            out.0[idx][i] = per[i / 2][idx];
        }
    }
    out
}

/// Row regrouping applied to each chunk plane before the affine products:
/// `plane'.byte[8·i + t] = plane.byte[32·(i & 1) + 4·t + (i >> 1)]`.
///
/// A plane's byte `r` is chunk `j` of row `r`, so this collects the sixteen
/// rows of one residue class `i >> 1` into qwords `2·(i>>1)` and `2·(i>>1)+1`
/// — the granularity at which `vgf2p8affineqb` can pick a matrix. It also
/// leaves the fold outputs in residue-major order, `out[16·k + t] = c_k ·
/// fold(row 4·t + k)`, so the composed output for group `t` is the XOR of
/// four ZMM-strided reads and needs no lane transpose.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[rustfmt::skip]
const SIGMA_C4: [i8; 64] = [
     0,  4,  8, 12, 16, 20, 24, 28,
    32, 36, 40, 44, 48, 52, 56, 60,
     1,  5,  9, 13, 17, 21, 25, 29,
    33, 37, 41, 45, 49, 53, 57, 61,
     2,  6, 10, 14, 18, 22, 26, 30,
    34, 38, 42, 46, 50, 54, 58, 62,
     3,  7, 11, 15, 19, 23, 27, 31,
    35, 39, 43, 47, 51, 55, 59, 63,
];

/// [`gfni_fold64_rows_masked`] that also applies the composed-fold constant
/// `coeffs[r & 3]` to every row `r` of the batch, for free, by folding the
/// constant into the bit matrices.
///
/// Writes the 64 scaled folds in **residue-major** order: `out[16·k + t]` is
/// `coeffs[k] · fold(row 4·t + k)`. The caller's composed output for group
/// `t` is therefore `out[t] ^ out[16 + t] ^ out[32 + t] ^ out[48 + t]`.
///
/// # Safety
/// As [`gfni_fold64_rows_masked`]: 64 readable bytes at `rows.add(64 * i)`
/// for every line not marked dead, and 64 writable `F128`s at `out`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512vbmi,gfni")]
pub(crate) unsafe fn gfni_fold64_rows_masked_c4(
    rows: *const u8,
    m: &CFoldMats,
    out: *mut F128,
    dead_lines: u8,
) {
    use core::arch::x86_64::*;
    // SAFETY (whole body): caller guarantees the row and output bounds; every
    // shuffle index is in range and the cfg gate supplies each intrinsic.
    unsafe {
        let mut z = [_mm512_setzero_si512(); 8];
        for (i, slot) in z.iter_mut().enumerate() {
            if dead_lines & (1u8 << i) == 0 {
                *slot = _mm512_loadu_si512(rows.add(64 * i) as *const __m512i);
            }
        }

        #[rustfmt::skip]
        const BT: [i8; 64] = [
            0, 8, 16, 24, 32, 40, 48, 56,  1, 9, 17, 25, 33, 41, 49, 57,
            2, 10, 18, 26, 34, 42, 50, 58,  3, 11, 19, 27, 35, 43, 51, 59,
            4, 12, 20, 28, 36, 44, 52, 60,  5, 13, 21, 29, 37, 45, 53, 61,
            6, 14, 22, 30, 38, 46, 54, 62,  7, 15, 23, 31, 39, 47, 55, 63,
        ];
        let bt = _mm512_loadu_si512(BT.as_ptr() as *const __m512i);
        let sigma = _mm512_loadu_si512(SIGMA_C4.as_ptr() as *const __m512i);
        let s2_lo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        let s2_hi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
        let s3_lo = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        let s3_hi = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

        let qword_transpose = |t: [__m512i; 8]| -> [__m512i; 8] {
            let e01 = _mm512_unpacklo_epi64(t[0], t[1]);
            let o01 = _mm512_unpackhi_epi64(t[0], t[1]);
            let e23 = _mm512_unpacklo_epi64(t[2], t[3]);
            let o23 = _mm512_unpackhi_epi64(t[2], t[3]);
            let e45 = _mm512_unpacklo_epi64(t[4], t[5]);
            let o45 = _mm512_unpackhi_epi64(t[4], t[5]);
            let e67 = _mm512_unpacklo_epi64(t[6], t[7]);
            let o67 = _mm512_unpackhi_epi64(t[6], t[7]);
            let h02_a = _mm512_permutex2var_epi64(e01, s2_lo, e23);
            let h46_a = _mm512_permutex2var_epi64(e01, s2_hi, e23);
            let h13_a = _mm512_permutex2var_epi64(o01, s2_lo, o23);
            let h57_a = _mm512_permutex2var_epi64(o01, s2_hi, o23);
            let h02_b = _mm512_permutex2var_epi64(e45, s2_lo, e67);
            let h46_b = _mm512_permutex2var_epi64(e45, s2_hi, e67);
            let h13_b = _mm512_permutex2var_epi64(o45, s2_lo, o67);
            let h57_b = _mm512_permutex2var_epi64(o45, s2_hi, o67);
            [
                _mm512_permutex2var_epi64(h02_a, s3_lo, h02_b),
                _mm512_permutex2var_epi64(h13_a, s3_lo, h13_b),
                _mm512_permutex2var_epi64(h02_a, s3_hi, h02_b),
                _mm512_permutex2var_epi64(h13_a, s3_hi, h13_b),
                _mm512_permutex2var_epi64(h46_a, s3_lo, h46_b),
                _mm512_permutex2var_epi64(h57_a, s3_lo, h57_b),
                _mm512_permutex2var_epi64(h46_a, s3_hi, h46_b),
                _mm512_permutex2var_epi64(h57_a, s3_hi, h57_b),
            ]
        };

        let mut t = [_mm512_setzero_si512(); 8];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = _mm512_permutexvar_epi8(bt, z[i]);
        }
        let p = qword_transpose(t);
        // Regroup each plane so all eight rows of a qword share a residue.
        let mut pc = [_mm512_setzero_si512(); 8];
        for (j, slot) in pc.iter_mut().enumerate() {
            *slot = _mm512_permutexvar_epi8(sigma, p[j]);
        }

        let mut acc = [_mm512_setzero_si512(); 16];
        for (k, slot) in acc.iter_mut().enumerate() {
            let g = |j: usize| {
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    pc[j],
                    _mm512_loadu_si512(m.0[j * 16 + k].as_ptr() as *const __m512i),
                )
            };
            let v1 = _mm512_ternarylogic_epi64::<0x96>(g(0), g(1), g(2));
            let v2 = _mm512_ternarylogic_epi64::<0x96>(g(3), g(4), g(5));
            let v3 = _mm512_ternarylogic_epi64::<0x96>(g(6), g(7), v1);
            *slot = _mm512_xor_si512(v2, v3);
        }

        let lo_half = qword_transpose(acc[..8].try_into().unwrap());
        let hi_half = qword_transpose(acc[8..].try_into().unwrap());
        let il_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        let il_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        for i in 0..8 {
            let lo = _mm512_permutexvar_epi8(bt, lo_half[i]);
            let hi = _mm512_permutexvar_epi8(bt, hi_half[i]);
            let out_ptr = (out as *mut u8).add(128 * i) as *mut __m512i;
            _mm512_storeu_si512(out_ptr, _mm512_permutex2var_epi64(lo, il_lo, hi));
            _mm512_storeu_si512(out_ptr.add(1), _mm512_permutex2var_epi64(lo, il_hi, hi));
        }
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
    // SAFETY: forwarded contract.
    unsafe { gfni_fold64_regs_impl::<false>(z, mats, out) }
}

/// [`gfni_fold64_regs`] with the [`SIGMA_C4`] residue-major relabeling —
/// see [`gfni_fold64_rows_masked_tr`].
///
/// # Safety
/// As [`gfni_fold64_regs`], plus `avx512vbmi`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512vbmi,gfni")]
unsafe fn gfni_fold64_regs_sigma(
    z: [core::arch::x86_64::__m512i; 8],
    mats: &[u64; 128],
    out: *mut F128,
) {
    // SAFETY: forwarded contract.
    unsafe { gfni_fold64_regs_impl::<true>(z, mats, out) }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,gfni")]
#[inline]
unsafe fn gfni_fold64_regs_impl<const SIGMA: bool>(
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
        let mut p = qword_transpose(t);
        // Residue-major relabeling: plane byte = row, and the affine products
        // treat every byte position uniformly, so permuting the plane bytes
        // relabels the rows for everything downstream — the row-major
        // reassembly then emits `out[16k + t] = fold(row 4t + k)`.
        #[cfg(all(target_feature = "avx512vbmi", target_feature = "vpclmulqdq"))]
        if SIGMA {
            let sigma = _mm512_loadu_si512(SIGMA_C4.as_ptr() as *const __m512i);
            for slot in p.iter_mut() {
                *slot = _mm512_permutexvar_epi8(sigma, *slot);
            }
        }

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
