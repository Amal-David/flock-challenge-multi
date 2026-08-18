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

        while x_lo + 8 <= lo_size {
            // a[k][lane]: row k (0..4) of group `lane` (0..4).
            let mut a = [[F128::ZERO; 4]; 4];
            let mut b = [[F128::ZERO; 4]; 4];
            for lane in 0..4 {
                for half in 0..2 {
                    let pair = x_lo + 2 * lane + half;
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
                    let x0g = row_base + x0l;
                    let x1g = x0g + 1;
                    let folded = fold_round2_pair_x86_unchecked_8(
                        table_data,
                        a_pkt.add(x0g * 8),
                        a_pkt.add(x1g * 8),
                        b_pkt.add(x0g * 8),
                        b_pkt.add(x1g * 8),
                    );
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
            let e_lo = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
            let e_hi = f128x4_loadu(eq_lo.as_ptr().add(x_lo + 4));
            let w = _mm512_permutex2var_epi64(e_lo, odd_idx, e_hi);
            let a0 = f128x4_loadu(a[0].as_ptr());
            let a1 = f128x4_loadu(a[1].as_ptr());
            let a2 = f128x4_loadu(a[2].as_ptr());
            let a3 = f128x4_loadu(a[3].as_ptr());
            let b0 = f128x4_loadu(b[0].as_ptr());
            let b1 = f128x4_loadu(b[1].as_ptr());
            let b2 = f128x4_loadu(b[2].as_ptr());
            let b3 = f128x4_loadu(b[3].as_ptr());
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
            let oa0 = fold16_to_4(a_src, ra, rb, even_idx, odd_idx);
            let oa1 = fold16_to_4(a_src.add(16), ra, rb, even_idx, odd_idx);
            let oa2 = fold16_to_4(a_src.add(32), ra, rb, even_idx, odd_idx);
            let oa3 = fold16_to_4(a_src.add(48), ra, rb, even_idx, odd_idx);
            let ob0 = fold16_to_4(b_src, ra, rb, even_idx, odd_idx);
            let ob1 = fold16_to_4(b_src.add(16), ra, rb, even_idx, odd_idx);
            let ob2 = fold16_to_4(b_src.add(32), ra, rb, even_idx, odd_idx);
            let ob3 = fold16_to_4(b_src.add(48), ra, rb, even_idx, odd_idx);
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
                let folded = fold_round2_pair_x86_unchecked_8(
                    table_data,
                    a_pkt.add(r0 * 8),
                    a_pkt.add((r0 + 1) * 8),
                    b_pkt.add(r0 * 8),
                    b_pkt.add((r0 + 1) * 8),
                );
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
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut acc = [WideGhashX4::zero(); 8];
        let mut tail = [F256Unreduced::ZERO; 8];
        let mut x_lo = 0;

        while x_lo + 8 <= lo_size {
            let ol = 2 * x_lo; // local output index of group 0
            let xg = out_base + ol;
            let (oa0, ob0) = group_from_packed(table_data, a_pkt, b_pkt, xg, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
            let (oa1, ob1) = group_from_packed(table_data, a_pkt, b_pkt, xg + 4, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
            let (oa2, ob2) = group_from_packed(table_data, a_pkt, b_pkt, xg + 8, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
            let (oa3, ob3) = group_from_packed(table_data, a_pkt, b_pkt, xg + 12, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
            let ap = a_out.as_mut_ptr().add(ol);
            let bp = b_out.as_mut_ptr().add(ol);
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

        // Small instances leave whole groups: one group at a time, scalar finish.
        while x_lo + 2 <= lo_size {
            let ol = 2 * x_lo;
            let (oa, ob) = group_from_packed(table_data, a_pkt, b_pkt, out_base + ol, r1, r2, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
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

        let mut out = [F128::ZERO; 8];
        for i in 0..8 {
            tail[i] ^= acc[i].fold();
            out[i] = tail[i].reduce();
        }
        out
    }
}

/// Accumulate one four-lane batch of eight-row groups into the 34 unreduced
/// three-challenge sums (see `la3_group_products` for the layout). `aw` are
/// the `a` rows prescaled by the group weight; `b` the raw `b` rows.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[inline(always)]
unsafe fn la3_group_products_x4(
    aw: &[core::arch::x86_64::__m512i; 8],
    b: &[core::arch::x86_64::__m512i; 8],
    acc: &mut [WideGhashX4; super::super::LA3_SUMS],
) {
    use core::arch::x86_64::*;
    // SAFETY: register-only arithmetic; features cfg-gated.
    unsafe {
        let x = |u: __m512i, v: __m512i| _mm512_xor_si512(u, v);
        for t in 0..4 {
            acc[2 * t].mul_acc(aw[2 * t + 1], b[2 * t + 1]);
            acc[2 * t + 1].mul_acc(x(aw[2 * t], aw[2 * t + 1]), x(b[2 * t], b[2 * t + 1]));
        }
        for h in 0..2 {
            let (a0, a1, a2, a3) = (aw[4 * h], aw[4 * h + 1], aw[4 * h + 2], aw[4 * h + 3]);
            let (b0, b1, b2, b3) = (b[4 * h], b[4 * h + 1], b[4 * h + 2], b[4 * h + 3]);
            acc[8 + 4 * h].mul_acc(a2, b2);
            let (ea, eb) = (x(a0, a2), x(b0, b2));
            let (oa, ob) = (x(a1, a3), x(b1, b3));
            acc[9 + 4 * h].mul_acc(ea, eb);
            acc[10 + 4 * h].mul_acc(oa, ob);
            acc[11 + 4 * h].mul_acc(x(ea, oa), x(eb, ob));
        }
        // Round four, point 1: c0 = a4, c1 = a4+a5, c2 = a4+a6, c3 = a4+a5+a6+a7.
        let ca = [aw[4], x(aw[4], aw[5]), x(aw[4], aw[6]), x(x(aw[4], aw[5]), x(aw[6], aw[7]))];
        let cb = [b[4], x(b[4], b[5]), x(b[4], b[6]), x(x(b[4], b[5]), x(b[6], b[7]))];
        // Point ∞ over e_j = a_j + a_{j+4}.
        let ea = [x(aw[0], aw[4]), x(aw[1], aw[5]), x(aw[2], aw[6]), x(aw[3], aw[7])];
        let eb = [x(b[0], b[4]), x(b[1], b[5]), x(b[2], b[6]), x(b[3], b[7])];
        let da = [ea[0], x(ea[0], ea[1]), x(ea[0], ea[2]), x(x(ea[0], ea[1]), x(ea[2], ea[3]))];
        let db = [eb[0], x(eb[0], eb[1]), x(eb[0], eb[2]), x(x(eb[0], eb[1]), x(eb[2], eb[3]))];
        let nine = |p: &[__m512i; 4], q: &[__m512i; 4], acc: &mut [WideGhashX4], base: usize| {
            acc[base].mul_acc(p[0], q[0]);
            acc[base + 1].mul_acc(p[1], q[1]);
            acc[base + 2].mul_acc(x(p[0], p[1]), x(q[0], q[1]));
            acc[base + 3].mul_acc(p[2], q[2]);
            acc[base + 4].mul_acc(p[3], q[3]);
            acc[base + 5].mul_acc(x(p[2], p[3]), x(q[2], q[3]));
            let (s0, s1) = (x(p[0], p[2]), x(p[1], p[3]));
            let (t0, t1) = (x(q[0], q[2]), x(q[1], q[3]));
            acc[base + 6].mul_acc(s0, t0);
            acc[base + 7].mul_acc(s1, t1);
            acc[base + 8].mul_acc(x(s0, s1), x(t0, t1));
        };
        nine(&ca, &cb, acc, 16);
        nine(&da, &db, acc, 25);
    }
}

/// x86 three-challenge sweep for one worker chunk (no stores). Returns the
/// 34 per-chunk sums of `la3_group_products`, each reduced.
///
/// # Safety
/// `table_data` must point to the 8 × 256 fold table; `a_pkt`/`b_pkt` must
/// expose 8 readable bytes for every row `row_base .. row_base + 2·eq_lo.len()`;
/// `eq_lo.len()` is a multiple of 4. AVX-512F and VPCLMULQDQ are cfg-gated.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn round2_lookahead3_chunk_x86_avx512(
    table_data: *const F128,
    a_pkt: *const u8,
    b_pkt: *const u8,
    row_base: usize,
    eq_lo: &[F128],
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [F128; super::super::LA3_SUMS] {
    use super::super::{LA3_SUMS, la3_group_products};
    use crate::field::gf2_128::x86_64::{f128x4_set, ghash_mul_x4};
    use core::arch::x86_64::*;

    let lo_size = eq_lo.len();
    debug_assert!(lo_size.is_multiple_of(4));

    // SAFETY: the function's contract bounds every packed-row and table read;
    // the cfg gate supplies every intrinsic feature.
    unsafe {
        let mut acc = [WideGhashX4::zero(); LA3_SUMS];
        let mut tail = [F256Unreduced::ZERO; LA3_SUMS];
        let mut x_lo = 0;

        while x_lo + 16 <= lo_size {
            // a[k][lane]: row k (0..8) of group `lane` (0..4).
            let mut a = [[F128::ZERO; 4]; 8];
            let mut b = [[F128::ZERO; 4]; 8];
            for lane in 0..4 {
                for t in 0..4 {
                    let pair = x_lo + 4 * lane + t;
                    if ((pair_idx_base + pair) & pair_in_block_mask) >= useful_pairs_inclusive {
                        continue;
                    }
                    let x0g = row_base + 2 * pair;
                    let folded = fold_round2_pair_x86_unchecked_8(
                        table_data,
                        a_pkt.add(x0g * 8),
                        a_pkt.add((x0g + 1) * 8),
                        b_pkt.add(x0g * 8),
                        b_pkt.add((x0g + 1) * 8),
                    );
                    a[2 * t][lane] = folded[0];
                    a[2 * t + 1][lane] = folded[1];
                    b[2 * t][lane] = folded[2];
                    b[2 * t + 1][lane] = folded[3];
                }
            }
            let w = f128x4_set(eq_lo[x_lo + 3], eq_lo[x_lo + 7], eq_lo[x_lo + 11], eq_lo[x_lo + 15]);
            let mut aw = [_mm512_setzero_si512(); 8];
            let mut bv = [_mm512_setzero_si512(); 8];
            for k in 0..8 {
                aw[k] = ghash_mul_x4(w, f128x4_loadu(a[k].as_ptr()));
                bv[k] = f128x4_loadu(b[k].as_ptr());
            }
            la3_group_products_x4(&aw, &bv, &mut acc);
            x_lo += 16;
        }

        // Small instances leave whole eight-row groups: scalar finish.
        while x_lo + 4 <= lo_size {
            let mut a = [F128::ZERO; 8];
            let mut b = [F128::ZERO; 8];
            let mut any = false;
            for t in 0..4 {
                let pair = x_lo + t;
                if ((pair_idx_base + pair) & pair_in_block_mask) >= useful_pairs_inclusive {
                    continue;
                }
                any = true;
                let x0g = row_base + 2 * pair;
                let folded = fold_round2_pair_x86_unchecked_8(
                    table_data,
                    a_pkt.add(x0g * 8),
                    a_pkt.add((x0g + 1) * 8),
                    b_pkt.add(x0g * 8),
                    b_pkt.add((x0g + 1) * 8),
                );
                a[2 * t] = folded[0];
                a[2 * t + 1] = folded[1];
                b[2 * t] = folded[2];
                b[2 * t + 1] = folded[3];
            }
            if any {
                let wt = eq_lo[x_lo + 3];
                let aw: [F128; 8] = core::array::from_fn(|k| wt * a[k]);
                la3_group_products(&aw, &b, &mut tail);
            }
            x_lo += 4;
        }

        let mut out = [F128::ZERO; LA3_SUMS];
        for i in 0..LA3_SUMS {
            tail[i] ^= acc[i].fold();
            out[i] = tail[i].reduce();
        }
        out
    }
}

/// x86 8:1 no-materialize composed pass for one worker chunk (rounds 3+4+5):
/// gathers → ρ₁ → ρ₂ → ρ₃ in registers, stores each output group once, and
/// accumulates the parity-split round-five message plus the six round-six
/// aggregates. Composed output `x` ← packed rows `8x..8x+8` (pairs `4x..4x+4`).
///
/// # Safety
/// `table_data` must point to the 8 × 256 fold table; `a_pkt`/`b_pkt` must
/// expose 8 readable bytes for every row `8·out_base .. 8·(out_base +
/// a_out.len())`; `a_out.len() == b_out.len() == 2·eq_lo.len()`, `eq_lo.len()`
/// even and ≥ 2. AVX-512F and VPCLMULQDQ are cfg-gated.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold3_from_packed_lookahead_x86_avx512(
    table_data: *const F128,
    a_pkt: *const u8,
    b_pkt: *const u8,
    out_base: usize,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho1: F128,
    rho2: F128,
    rho3: F128,
    eq_lo: &[F128],
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
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

    // One output group of four (sixteen pairs = thirty-two rows per array).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn group_from_packed3(
        table_data: *const F128,
        a_pkt: *const u8,
        b_pkt: *const u8,
        x0: usize,
        r1: __m512i,
        r2: __m512i,
        r3: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
        pair_in_block_mask: usize,
        useful_pairs_inclusive: usize,
    ) -> (__m512i, __m512i) {
        use core::arch::x86_64::*;
        // SAFETY: caller bounds rows 8·x0 .. 8·x0 + 32.
        unsafe {
            let mut ae = [F128::ZERO; 16];
            let mut ao = [F128::ZERO; 16];
            let mut be = [F128::ZERO; 16];
            let mut bo = [F128::ZERO; 16];
            for p in 0..16 {
                let pair = 4 * x0 + p;
                if (pair & pair_in_block_mask) >= useful_pairs_inclusive {
                    continue;
                }
                let r0 = 2 * pair;
                let folded = fold_round2_pair_x86_unchecked_8(
                    table_data,
                    a_pkt.add(r0 * 8),
                    a_pkt.add((r0 + 1) * 8),
                    b_pkt.add(r0 * 8),
                    b_pkt.add((r0 + 1) * 8),
                );
                ae[p] = folded[0];
                ao[p] = folded[1];
                be[p] = folded[2];
                bo[p] = folded[3];
            }
            let mut ta = [_mm512_setzero_si512(); 4];
            let mut tb = [_mm512_setzero_si512(); 4];
            for q in 0..4 {
                let e = f128x4_loadu(ae.as_ptr().add(4 * q));
                let o = f128x4_loadu(ao.as_ptr().add(4 * q));
                ta[q] = _mm512_xor_si512(e, ghash_mul_x4(r1, _mm512_xor_si512(e, o)));
                let e = f128x4_loadu(be.as_ptr().add(4 * q));
                let o = f128x4_loadu(bo.as_ptr().add(4 * q));
                tb[q] = _mm512_xor_si512(e, ghash_mul_x4(r1, _mm512_xor_si512(e, o)));
            }
            let ua0 = fold_regs(ta[0], ta[1], r2, even_idx, odd_idx);
            let ua1 = fold_regs(ta[2], ta[3], r2, even_idx, odd_idx);
            let ub0 = fold_regs(tb[0], tb[1], r2, even_idx, odd_idx);
            let ub1 = fold_regs(tb[2], tb[3], r2, even_idx, odd_idx);
            (
                fold_regs(ua0, ua1, r3, even_idx, odd_idx),
                fold_regs(ub0, ub1, r3, even_idx, odd_idx),
            )
        }
    }

    // SAFETY: the function's contract bounds every read/store; features cfg-gated.
    unsafe {
        let r1 = _mm512_broadcast_i32x4(_mm_set_epi64x(rho1.hi as i64, rho1.lo as i64));
        let r2 = _mm512_broadcast_i32x4(_mm_set_epi64x(rho2.hi as i64, rho2.lo as i64));
        let r3 = _mm512_broadcast_i32x4(_mm_set_epi64x(rho3.hi as i64, rho3.lo as i64));
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut acc = [WideGhashX4::zero(); 8];
        let mut tail = [F256Unreduced::ZERO; 8];
        let mut x_lo = 0;

        while x_lo + 8 <= lo_size {
            let ol = 2 * x_lo;
            let xg = out_base + ol;
            let (oa0, ob0) = group_from_packed3(table_data, a_pkt, b_pkt, xg, r1, r2, r3, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
            let (oa1, ob1) = group_from_packed3(table_data, a_pkt, b_pkt, xg + 4, r1, r2, r3, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
            let (oa2, ob2) = group_from_packed3(table_data, a_pkt, b_pkt, xg + 8, r1, r2, r3, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
            let (oa3, ob3) = group_from_packed3(table_data, a_pkt, b_pkt, xg + 12, r1, r2, r3, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
            let ap = a_out.as_mut_ptr().add(ol);
            let bp = b_out.as_mut_ptr().add(ol);
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

        while x_lo + 2 <= lo_size {
            let ol = 2 * x_lo;
            let (oa, ob) = group_from_packed3(table_data, a_pkt, b_pkt, out_base + ol, r1, r2, r3, even_idx, odd_idx, pair_in_block_mask, useful_pairs_inclusive);
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

        let mut out = [F128::ZERO; 8];
        for i in 0..8 {
            tail[i] ^= acc[i].fold();
            out[i] = tail[i].reduce();
        }
        out
    }
}
