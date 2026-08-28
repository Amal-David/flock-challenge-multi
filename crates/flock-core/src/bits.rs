//! Small bit-manipulation primitives shared across modules.

/// Hacker's Delight (Sec. 7-3) 8×8 bit-matrix transpose stored in a `u64`.
///
/// The input holds 8 bytes representing 8 rows of 8 bits each; the output holds
/// the transposed matrix (bit `r·8 + c` of input → bit `c·8 + r` of output).
///
/// Shared by the lincheck byte-stripe builder (`flock_prover::r1cs_hashes::common`)
/// and the PCS ring-switch `fold_1b` kernels ([`crate::pcs::ring_switch`]).
#[inline(always)]
pub(crate) fn transpose_8x8_bits(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AAu64;
    x = x ^ t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCCu64;
    x = x ^ t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0u64;
    x = x ^ t ^ (t << 28);
    x
}

/// Bit-transpose 8 little-endian `u64` lanes (the 64-byte block they form) into
/// a 64-byte output stripe.
///
/// The 8 LE u64s viewed as 64 bytes are exactly the input shape of the NEON
/// [`bit_transpose_64bytes`] kernel (input byte `r·8 + c` = byte `c` of lane
/// `r`; output byte `c·8 + t` bit `r` = that byte's bit `t`).
///
/// Dispatch: GFNI (compiled/detected) → AVX2 pshufb+unpack 8×8 → scalar tail.
///
/// [`bit_transpose_64bytes`]: crate::zerocheck::univariate_skip_optimized::bit_transpose_64bytes
#[inline(always)]
pub fn transpose_8_u64s_to_64_bytes(lanes: &[u64; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 64);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("gfni")
        {
            // SAFETY: the four feature flags were just observed on this CPU.
            unsafe { transpose_8_u64s_to_64_bytes_gfni(lanes, out) }
            return;
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 was just observed on this CPU.
            unsafe { transpose_8_u64s_to_64_bytes_avx2(lanes, out) }
            return;
        }
    }
    transpose_8_u64s_to_64_bytes_scalar(lanes, out);
}

/// Scalar 8×8-byte then 8×8-bit transpose. Public tail and test oracle.
#[allow(clippy::erasing_op, clippy::identity_op)]
fn transpose_8_u64s_to_64_bytes_scalar(lanes: &[u64; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 64);
    for c in 0..8 {
        let shift = c * 8;
        let mut packed: u64 = 0;
        packed |= ((lanes[0] >> shift) & 0xFF) << (0 * 8);
        packed |= ((lanes[1] >> shift) & 0xFF) << (1 * 8);
        packed |= ((lanes[2] >> shift) & 0xFF) << (2 * 8);
        packed |= ((lanes[3] >> shift) & 0xFF) << (3 * 8);
        packed |= ((lanes[4] >> shift) & 0xFF) << (4 * 8);
        packed |= ((lanes[5] >> shift) & 0xFF) << (5 * 8);
        packed |= ((lanes[6] >> shift) & 0xFF) << (6 * 8);
        packed |= ((lanes[7] >> shift) & 0xFF) << (7 * 8);
        let transposed = transpose_8x8_bits(packed);
        out[c * 8..c * 8 + 8].copy_from_slice(&transposed.to_le_bytes());
    }
}

#[cfg(target_arch = "x86_64")]
#[rustfmt::skip]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn transpose_8_u64s_to_64_bytes_gfni(lanes: &[u64; 8], out: &mut [u8]) {
    use core::arch::x86_64::*;
    const I:[u8;64]=[56,48,40,32,24,16,8,0,57,49,41,33,25,17,9,1,58,50,42,34,26,18,10,2,59,51,43,35,27,19,11,3,60,52,44,36,28,20,12,4,61,53,45,37,29,21,13,5,62,54,46,38,30,22,14,6,63,55,47,39,31,23,15,7];
    unsafe {
        let x=_mm512_loadu_si512(lanes.as_ptr() as *const __m512i);
        let i=_mm512_loadu_si512(I.as_ptr() as *const __m512i);
        let id=_mm512_set1_epi64(0x8040201008040201u64 as i64);
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i,_mm512_gf2p8affine_epi64_epi8::<0>(id,_mm512_permutexvar_epi8(i,x)));
    }
}

/// AVX2 8×8 byte transpose (`_mm256_shuffle_epi8` + unpack) then Hacker's
/// Delight 8×8 bit transpose in four-lane ymm, matching the scalar oracle.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transpose_8_u64s_to_64_bytes_avx2(lanes: &[u64; 8], out: &mut [u8]) {
    use core::arch::x86_64::*;
    unsafe {
        let a = _mm256_loadu_si256(lanes.as_ptr() as *const __m256i);
        let b = _mm256_loadu_si256(lanes.as_ptr().add(4) as *const __m256i);
        // Identity pshufb per 128-bit half (ssse3/avx2 contract); keeps the
        // 8×8 byte rows in-lane so unpack sees the same layout as scalar.
        let id = _mm256_broadcastsi128_si256(_mm_set_epi8(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        ));
        let a = _mm256_shuffle_epi8(a, id);
        let b = _mm256_shuffle_epi8(b, id);

        let r01 = _mm256_castsi256_si128(a);
        let r23 = _mm256_extracti128_si256::<1>(a);
        let r45 = _mm256_castsi256_si128(b);
        let r67 = _mm256_extracti128_si256::<1>(b);

        let l0 = r01;
        let l1 = _mm_srli_si128::<8>(r01);
        let l2 = r23;
        let l3 = _mm_srli_si128::<8>(r23);
        let l4 = r45;
        let l5 = _mm_srli_si128::<8>(r45);
        let l6 = r67;
        let l7 = _mm_srli_si128::<8>(r67);

        let t0 = _mm_unpacklo_epi8(l0, l1);
        let t1 = _mm_unpacklo_epi8(l2, l3);
        let t2 = _mm_unpacklo_epi8(l4, l5);
        let t3 = _mm_unpacklo_epi8(l6, l7);
        let u0 = _mm_unpacklo_epi16(t0, t1);
        let u1 = _mm_unpackhi_epi16(t0, t1);
        let u2 = _mm_unpacklo_epi16(t2, t3);
        let u3 = _mm_unpackhi_epi16(t2, t3);
        let v0 = _mm_unpacklo_epi32(u0, u2);
        let v1 = _mm_unpackhi_epi32(u0, u2);
        let v2 = _mm_unpacklo_epi32(u1, u3);
        let v3 = _mm_unpackhi_epi32(u1, u3);

        let cols_lo = _mm256_set_m128i(v1, v0);
        let cols_hi = _mm256_set_m128i(v3, v2);
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, transpose_8x8_bits_x4(cols_lo));
        _mm256_storeu_si256(
            out.as_mut_ptr().add(32) as *mut __m256i,
            transpose_8x8_bits_x4(cols_hi),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transpose_8x8_bits_x4(x: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    let m1 = _mm256_set1_epi64x(0x00AA_00AA_00AA_00AAu64 as i64);
    let t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64::<7>(x)), m1);
    let x = _mm256_xor_si256(_mm256_xor_si256(x, t), _mm256_slli_epi64::<7>(t));
    let m2 = _mm256_set1_epi64x(0x0000_CCCC_0000_CCCCu64 as i64);
    let t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64::<14>(x)), m2);
    let x = _mm256_xor_si256(_mm256_xor_si256(x, t), _mm256_slli_epi64::<14>(t));
    let m3 = _mm256_set1_epi64x(0x0000_0000_F0F0_F0F0u64 as i64);
    let t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64::<28>(x)), m3);
    _mm256_xor_si256(_mm256_xor_si256(x, t), _mm256_slli_epi64::<28>(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The selected transpose must match the scalar per-column oracle
    /// bit-for-bit on varied inputs.
    #[test]
    fn transpose_8_u64s_matches_scalar() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        for _ in 0..100 {
            let lanes: [u64; 8] = std::array::from_fn(|_| next());
            let mut fast = [0u8; 64];
            let mut oracle = [0u8; 64];
            transpose_8_u64s_to_64_bytes(&lanes, &mut fast);
            transpose_8_u64s_to_64_bytes_scalar(&lanes, &mut oracle);
            assert_eq!(fast, oracle);
        }
        // Edge patterns.
        for lanes in [[0u64; 8], [u64::MAX; 8], std::array::from_fn(|i| 1u64 << i)] {
            let mut fast = [0u8; 64];
            let mut oracle = [0u8; 64];
            transpose_8_u64s_to_64_bytes(&lanes, &mut fast);
            transpose_8_u64s_to_64_bytes_scalar(&lanes, &mut oracle);
            assert_eq!(fast, oracle, "lanes={lanes:?}");
        }
    }

    /// Transposing twice is the identity.
    #[test]
    fn transpose_is_involution() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..256 {
            state = state.wrapping_mul(0x2545_F491_4F6C_DD1D).rotate_left(31);
            assert_eq!(transpose_8x8_bits(transpose_8x8_bits(state)), state);
        }
    }

    /// Cross-check against a naive bit-by-bit transpose of the 8×8 matrix.
    #[test]
    fn matches_naive() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..256 {
            state = state.wrapping_mul(0x2545_F491_4F6C_DD1D).rotate_left(17);
            let got = transpose_8x8_bits(state);
            let mut want = 0u64;
            for r in 0..8 {
                for c in 0..8 {
                    if (state >> (r * 8 + c)) & 1 == 1 {
                        want |= 1u64 << (c * 8 + r);
                    }
                }
            }
            assert_eq!(got, want, "input={state:016x}");
        }
    }
}
