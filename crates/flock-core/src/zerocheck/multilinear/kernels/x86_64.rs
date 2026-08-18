use crate::field::gf2_128::x86_64::{f128x4_loadu, WideGhashX4};
use crate::field::{F256Unreduced, F128};

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

fn r2_nibble_lut_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_R2_NIBBLE").is_none())
}

/// Dispatch: nibble LUT by default. `FLOCK_NO_R2_NIBBLE=1` restores the
/// scalar 256-entry XMM loads. Ranked env is cleared so the nibble path
/// always runs there.
#[inline(always)]
pub(crate) unsafe fn fold_round2_pair_x86(
    table_data: *const F128,
    nibble: &super::super::FoldNibbleLut,
    a0_bytes: *const u8,
    a1_bytes: *const u8,
    b0_bytes: *const u8,
    b1_bytes: *const u8,
) -> [F128; 4] {
    // SAFETY: caller matches fold_round2_pair_x86_unchecked_8; both callees
    // are cfg-gated avx512f+vpclmulqdq.
    unsafe {
        if r2_nibble_lut_enabled() {
            fold_round2_pair_x86_nibble(nibble, a0_bytes, a1_bytes, b0_bytes, b1_bytes)
        } else {
            fold_round2_pair_x86_unchecked_8(table_data, a0_bytes, a1_bytes, b0_bytes, b1_bytes)
        }
    }
}

/// F₂-linear nibble-LUT fold. `UniSkipFoldTable` is XOR-doubling subset
/// sums, so `T[byte] = T[n0] ⊕ T[n1 << 4]`. Four rows × one chunk are
/// looked up with two `vpermi2q` (lo/hi) instead of four data-dependent
/// 128-bit GPR-indexed loads.
///
/// Bit-identical to [`fold_round2_pair_x86_unchecked_8`] on linear tables.
#[inline(always)]
pub(crate) unsafe fn fold_round2_pair_x86_nibble(
    nibble: &super::super::FoldNibbleLut,
    a0_bytes: *const u8,
    a1_bytes: *const u8,
    b0_bytes: *const u8,
    b1_bytes: *const u8,
) -> [F128; 4] {
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn lookup8(idx: __m512i, table: *const u64) -> __m512i {
        unsafe {
            let a = _mm512_load_si512(table as *const __m512i);
            let b = _mm512_load_si512(table.add(8) as *const __m512i);
            _mm512_permutex2var_epi64(a, idx, b)
        }
    }

    // SAFETY: each row has 8 readable bytes; nibble indices are 0..=15
    // after AND-0xf so every vpermi2q stays in a 16-entry half. The LUT
    // is 64-byte aligned by FoldNibbleLut's repr.
    unsafe {
        let nibble_mask = _mm512_set1_epi32(0xf);
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();
        for chunk in 0..8 {
            let packed = _mm_set_epi32(
                *b1_bytes.add(chunk) as i32,
                *b0_bytes.add(chunk) as i32,
                *a1_bytes.add(chunk) as i32,
                *a0_bytes.add(chunk) as i32,
            );
            let row = _mm512_castsi128_si512(packed);
            let n0 = _mm512_and_si512(row, nibble_mask);
            let n1 = _mm512_and_si512(_mm512_srli_epi32::<4>(row), nibble_mask);
            let n0_8 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(n0));
            let n1_8 = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(n1));
            let los = _mm512_xor_si512(
                lookup8(n0_8, nibble.n0_lo[chunk].as_ptr()),
                lookup8(n1_8, nibble.n1_lo[chunk].as_ptr()),
            );
            let his = _mm512_xor_si512(
                lookup8(n0_8, nibble.n0_hi[chunk].as_ptr()),
                lookup8(n1_8, nibble.n1_hi[chunk].as_ptr()),
            );
            acc_lo = _mm256_xor_si256(acc_lo, _mm512_castsi512_si256(los));
            acc_hi = _mm256_xor_si256(acc_hi, _mm512_castsi512_si256(his));
        }
        let lo: [u64; 4] = core::mem::transmute(acc_lo);
        let hi: [u64; 4] = core::mem::transmute(acc_hi);
        // _mm_set_epi32 packed b1,b0,a1,a0 into lanes 3..0, then cvtepu32_epi64
        // puts them in qword lanes 0..3 as a0,a1,b0,b1.
        [
            F128::new(lo[0], hi[0]),
            F128::new(lo[1], hi[1]),
            F128::new(lo[2], hi[2]),
            F128::new(lo[3], hi[3]),
        ]
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
