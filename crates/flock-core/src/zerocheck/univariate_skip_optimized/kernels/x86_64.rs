#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
use super::super::{ELL, F128, N_MEDIUM};
#[cfg(target_feature = "gfni")]
use super::super::{F8, InvNttTableByteSingleGf8, N_CHUNKS};

/// AVX-512 (VBMI) 64-byte bit-transpose — direct port of the NEON two-stage
/// algorithm. `_mm512_permutexvar_epi8` does the byte-gather (NEON `vqtbl4q`)
/// in one instruction; the three masked bit-swap rounds (distances 7/14/28)
/// are identical to the NEON version, applied to all eight 64-bit lanes at once.
///
/// Replaces `bit_transpose_64bytes_scalar` (512 branchy bit ops/call) — which
/// profiling showed was ~85% of round1's time on x86.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
#[target_feature(enable = "avx512vbmi,avx512bw,avx512f")]
pub(crate) unsafe fn bit_transpose_64bytes_avx512(input: &[u8; 64], output: &mut [u8; 64]) {
    use core::arch::x86_64::*;
    // Gather index = NEON IDX0 ++ IDX1 ++ IDX2 ++ IDX3 (the 8×8 byte transpose).
    #[rustfmt::skip]
    const IDX: [i8; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56,  1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,  3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,  5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,  7, 15, 23, 31, 39, 47, 55, 63,
    ];
    unsafe {
        let inp = _mm512_loadu_si512(input.as_ptr() as *const __m512i);
        let idx = _mm512_loadu_si512(IDX.as_ptr() as *const __m512i);
        let mut y = _mm512_permutexvar_epi8(idx, inp); // y[i] = input[IDX[i]]

        let mask1 = _mm512_set1_epi64(0x00AA00AA00AA00AAu64 as i64);
        let mask2 = _mm512_set1_epi64(0x0000CCCC0000CCCCu64 as i64);
        let mask3 = _mm512_set1_epi64(0x00000000F0F0F0F0u64 as i64);

        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<7>(y)), mask1);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<7>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<14>(y)), mask2);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<14>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<28>(y)), mask3);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<28>(t)));

        _mm512_storeu_si512(output.as_mut_ptr() as *mut __m512i, y);
    }
}

/// SSE/GFNI x86 kernel. The inverse-NTT apply uses its best available x86 path,
/// writes two 64-byte columns, and this kernel multiplies them four XMM chunks
/// at a time. Kept as the fallback for GFNI CPUs without AVX-512.
#[inline]
#[allow(dead_code)] // unused in native AVX-512 builds; exercised by its oracle test
#[cfg(all(target_arch = "x86_64", target_feature = "gfni"))]
#[target_feature(enable = "gfni,sse2")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_sse(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    use core::arch::x86_64::*;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // SAFETY: function carries gfni+sse2; raw loads/stores stay within the
    // validated `a_col`/`b_col` (len ELL) and `out` ([u8; 64]) buffers.
    unsafe {
        // 4 byte-accumulators × 16 lanes = ELL = 64 lanes, reduced F_8 values.
        let mut acc = [_mm_setzero_si128(); 4];
        for k in 0..8usize {
            let chunk_off = byte_base_b + k * N_CHUNKS;
            inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
            inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
            let a_ptr = a_col.as_ptr() as *const u8;
            let b_ptr = b_col.as_ptr() as *const u8;
            let xk = _mm_set1_epi8((1u8 << k) as i8); // x^k as an F_8 byte; k=0 ⇒ 1
            for c in 0..4usize {
                let av = _mm_loadu_si128(a_ptr.add(c * 16) as *const __m128i);
                let bv = _mm_loadu_si128(b_ptr.add(c * 16) as *const __m128i);
                // y = (a·b) · x^k in F_8. For k=0, xk=1 ⇒ second mul is identity.
                let y = _mm_gf2p8mul_epi8(_mm_gf2p8mul_epi8(av, bv), xk);
                acc[c] = _mm_xor_si128(acc[c], y);
            }
        }
        let out_ptr = out.as_mut_ptr();
        for c in 0..4usize {
            _mm_storeu_si128(out_ptr.add(c * 16) as *mut __m128i, acc[c]);
        }
    }
}

/// Fused AVX-512/GFNI x86 kernel. Each inverse-NTT apply returns all 64 F_8
/// evaluations in one ZMM register; the product and x^k scaling stay 64-wide
/// and register-resident through the final XOR accumulation.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    use core::arch::x86_64::*;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // SAFETY: the caller's packed-input bounds guarantee 8 readable bytes at
    // every K-row offset. The table has the protocol-fixed ell=64/chunks=8
    // shape, and `out` is exactly one writable ZMM register.
    unsafe {
        let mut acc = _mm512_setzero_si512();
        for k in 0..8usize {
            let off = byte_base_b + k * N_CHUNKS;
            let av = inv_table.apply_x86_avx512_register_unchecked(a_packed.as_ptr().add(off));
            let bv = inv_table.apply_x86_avx512_register_unchecked(b_packed.as_ptr().add(off));
            let product = _mm512_gf2p8mul_epi8(av, bv);
            // x^0 is the multiplicative identity, so avoid one GFNI operation
            // for the first row.
            let scaled = if k == 0 {
                product
            } else {
                _mm512_gf2p8mul_epi8(product, _mm512_set1_epi8((1u8 << k) as i8))
            };
            acc = _mm512_xor_si512(acc, scaled);
        }
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
    }
}
/// partials. The C side is table-free (see `kernels::accumulate_c_banks`).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_convert_ab_x86_avx512(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; ELL],
) {
    use crate::field::gf2_128::x86_64::{f128x4_set, ghash_mul_x4};
    use core::arch::x86_64::*;
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert_eq!(ELL % 4, 0);

    // SAFETY: the fixed-size input/partial arrays contain every four-lane load
    // and store below. Convert indices are `b_med * 256 + u8`, bounded by the
    // 16*256-entry table. The cfg gate supplies both required target features.
    unsafe {
        let eq = f128x4_set(eq_lo_val, eq_lo_val, eq_lo_val, eq_lo_val);
        for lane in (0..ELL).step_by(4) {
            let mut cf_ab = [F128::ZERO; 4];
            for b_med in 0..n_b_med {
                let table_base = b_med * 256;
                for j in 0..4 {
                    let v_ab = chunk_ab_bytes[b_med][lane + j] as usize;
                    cf_ab[j] += convert[table_base + v_ab];
                }
            }

            let scaled_ab = ghash_mul_x4(f128x4_set(cf_ab[0], cf_ab[1], cf_ab[2], cf_ab[3]), eq);

            let ab_ptr = partial_ab.as_mut_ptr().add(lane) as *mut __m512i;
            _mm512_storeu_si512(
                ab_ptr,
                _mm512_xor_si512(_mm512_loadu_si512(ab_ptr), scaled_ab),
            );
        }
    }
}

/// AVX-512 DirectC drain. Mask construction works on 16 output lanes at once:
/// each C row is widened from bytes to dwords, then eight `vptestmd` masks
/// select the row's bit into the corresponding bank accumulator. Four F128s
/// are updated per ZMM using two L1-resident qword gathers (one per table).
///
/// This deliberately requires only the dispatch contract's AVX-512F. In
/// particular, it does not silently depend on AVX-512BW or VBMI even though
/// the ranked machine also provides those extensions.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_c_banks_x86_avx512(
    c_block: &[u8; 16 * ELL],
    n_b_med: usize,
    mask_tables: &[F128],
    partial_c: &mut [[F128; ELL]; 8],
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(ELL, 64);
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert_eq!(mask_tables.len(), 512);

    let (t_lo, t_hi) = mask_tables.split_at(256);

    // SAFETY: each row load reads 16 bytes from a 64-byte row of `c_block`;
    // n_b_med <= 16 bounds the row index. Each mask is at most 0xffff, and
    // its low/high bytes therefore index the two 256-entry table halves.
    // Gather indices address the two qwords of four selected F128 values.
    // Each partial load/store covers exactly four lanes in a 64-lane bank.
    unsafe {
        for lane_base in (0..ELL).step_by(16) {
            let mut masks = [_mm512_setzero_si512(); 8];
            let bank_bits = [
                _mm512_set1_epi32(1),
                _mm512_set1_epi32(2),
                _mm512_set1_epi32(4),
                _mm512_set1_epi32(8),
                _mm512_set1_epi32(16),
                _mm512_set1_epi32(32),
                _mm512_set1_epi32(64),
                _mm512_set1_epi32(128),
            ];

            for b_med in 0..n_b_med {
                let row_ptr = c_block.as_ptr().add(b_med * ELL + lane_base);
                let row_bytes = _mm_loadu_si128(row_ptr as *const __m128i);
                let row = _mm512_cvtepu8_epi32(row_bytes);
                let weight = _mm512_set1_epi32((1u32 << b_med) as i32);

                for (mask, bank_bit) in masks.iter_mut().zip(bank_bits) {
                    let selected = _mm512_test_epi32_mask(row, bank_bit);
                    *mask = _mm512_mask_or_epi32(*mask, selected, *mask, weight);
                }
            }

            for (bank, mask) in partial_c.iter_mut().zip(masks) {
                let mut mask_words = [0u32; 16];
                _mm512_storeu_si512(mask_words.as_mut_ptr() as *mut __m512i, mask);

                for lane_in_group in (0..16).step_by(4) {
                    let m0 = i64::from(mask_words[lane_in_group]);
                    let m1 = i64::from(mask_words[lane_in_group + 1]);
                    let m2 = i64::from(mask_words[lane_in_group + 2]);
                    let m3 = i64::from(mask_words[lane_in_group + 3]);

                    // F128 is two adjacent u64s. These qword indices preserve
                    // the in-memory [lo, hi] order for four gathered values.
                    let lo_indices = _mm512_set_epi64(
                        2 * (m3 & 0xff) + 1,
                        2 * (m3 & 0xff),
                        2 * (m2 & 0xff) + 1,
                        2 * (m2 & 0xff),
                        2 * (m1 & 0xff) + 1,
                        2 * (m1 & 0xff),
                        2 * (m0 & 0xff) + 1,
                        2 * (m0 & 0xff),
                    );
                    let hi_indices = _mm512_set_epi64(
                        2 * (m3 >> 8) + 1,
                        2 * (m3 >> 8),
                        2 * (m2 >> 8) + 1,
                        2 * (m2 >> 8),
                        2 * (m1 >> 8) + 1,
                        2 * (m1 >> 8),
                        2 * (m0 >> 8) + 1,
                        2 * (m0 >> 8),
                    );

                    let from_lo = _mm512_i64gather_epi64::<8>(
                        lo_indices,
                        t_lo.as_ptr() as *const i64,
                    );
                    let from_hi = _mm512_i64gather_epi64::<8>(
                        hi_indices,
                        t_hi.as_ptr() as *const i64,
                    );
                    let partial_ptr = bank
                        .as_mut_ptr()
                        .add(lane_base + lane_in_group)
                        as *mut __m512i;
                    let updated = _mm512_xor_si512(
                        _mm512_loadu_si512(partial_ptr),
                        _mm512_xor_si512(from_lo, from_hi),
                    );
                    _mm512_storeu_si512(partial_ptr, updated);
                }
            }
        }
    }
}

/// Legacy two-bank capture kernel retained for compatibility with the old
/// entry point. DirectC does not call it.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_convert_with_s_hat_v_x86_avx512(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    chunk_c_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; ELL],
    partial_c_0: &mut [F128; ELL],
    partial_c_1: &mut [F128; ELL],
) {
    for lane in 0..ELL {
        let mut converted_ab = F128::ZERO;
        let mut converted_c_0 = F128::ZERO;
        let mut converted_c_1 = F128::ZERO;
        for b_med in 0..n_b_med {
            let table_base = b_med * 256;
            let c = usize::from(chunk_c_bytes[b_med][lane]);
            converted_ab += convert[table_base + usize::from(chunk_ab_bytes[b_med][lane])];
            converted_c_0 += convert[table_base + (c & 0x55)];
            converted_c_1 += convert[table_base + (c & 0xaa)];
        }
        partial_ab[lane] += converted_ab * eq_lo_val;
        partial_c_0[lane] += converted_c_0 * eq_lo_val;
        partial_c_1[lane] += converted_c_1 * eq_lo_val;
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod tests {
    use super::super::accumulate_c_banks_scalar;
    use super::{accumulate_c_banks_x86_avx512, F128};

    #[test]
    fn accumulate_c_banks_avx512_matches_scalar() {
        let mut c_block = [0u8; 16 * 64];
        for (i, byte) in c_block.iter_mut().enumerate() {
            *byte = (i as u8)
                .wrapping_mul(0x9d)
                .rotate_left((i & 7) as u32)
                ^ (i >> 3) as u8;
        }

        let mut mask_tables = [F128::ZERO; 512];
        for (i, value) in mask_tables.iter_mut().enumerate() {
            let x = i as u64;
            *value = F128::new(
                x.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x0123_4567_89ab_cdef,
                x.rotate_left(29).wrapping_mul(0xd6e8_feb8_6659_fd93)
                    ^ 0xfedc_ba98_7654_3210,
            );
        }

        for n_b_med in 0..=16 {
            let mut scalar = [[F128::ZERO; 64]; 8];
            for (bank_index, bank) in scalar.iter_mut().enumerate() {
                for (lane, value) in bank.iter_mut().enumerate() {
                    *value = F128::new(
                        ((bank_index * 64 + lane) as u64).wrapping_mul(0xa24b_aed4_963e_e407),
                        ((lane * 8 + bank_index) as u64).wrapping_mul(0x9fb2_1c65_1e98_df25),
                    );
                }
            }
            let mut simd = scalar;

            accumulate_c_banks_scalar(&c_block, n_b_med, &mask_tables, &mut scalar);
            // SAFETY: this test is compiled only when the kernel's target
            // features are statically enabled; all buffers have exact sizes.
            unsafe {
                accumulate_c_banks_x86_avx512(
                    &c_block,
                    n_b_med,
                    &mask_tables,
                    &mut simd,
                );
            }
            assert_eq!(simd, scalar, "n_b_med={n_b_med}");
        }
    }
}
