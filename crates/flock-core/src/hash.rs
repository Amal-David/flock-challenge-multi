//! Selection of the hash function backing a protocol component.
//!
//! Two components are independently configurable, and they are genuinely
//! independent — a proof can use BLAKE3 Merkle commitments with a SHA-256
//! Fiat-Shamir transcript, or any other combination:
//!
//! - the Merkle commitments, via [`crate::pcs::commit::PcsParams::merkle_hash`]
//!   (see [`crate::merkle`]);
//! - the Fiat-Shamir transcript and its proof-of-work grinding, via
//!   [`crate::challenger::FsChallenger::with_hash`].
//!
//! Both default to SHA-256, so configs and call sites that predate the options
//! keep their behaviour.

use serde::{Deserialize, Serialize};

/// Which hash function backs a component.
///
/// `Sha256` is the default, so existing serialized params and configs that
/// predate these options deserialize unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HashKind {
    #[default]
    Sha256,
    Blake3,
}

impl HashKind {
    /// Config-file spelling of this hash (`"sha256"` / `"blake3"`). Inverse of
    /// [`HashKind::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            HashKind::Sha256 => "sha256",
            HashKind::Blake3 => "blake3",
        }
    }

    /// Parse a config field or environment variable. Case-insensitive; rejects
    /// anything unrecognized rather than silently falling back to SHA-256 — a
    /// config naming a hash we do not implement must not quietly produce
    /// proofs under a different one.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sha256" | "sha-256" => Ok(HashKind::Sha256),
            "blake3" => Ok(HashKind::Blake3),
            other => Err(format!(
                "unknown hash {other:?}: expected \"sha256\" or \"blake3\""
            )),
        }
    }
}

impl std::fmt::Display for HashKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 4×1-way AVX2 BLAKE3 parent-compress finalizer.
///
/// Replaces the generic `blake3::platform::hash_many` parent path with a
/// hand-rolled, parent-only 4-wide AVX2 kernel: 4 parents compressed in
/// lockstep, where lane `i` of an `__m256i` holds parent `i`'s state value
/// for the corresponding slot. Lanes 4..7 are unused (4-wide in a 256-bit
/// register).
///
/// The BLAKE3 message permutation is emitted by `parent_xor_schedule!` into
/// a 7×16 `[[u8; 16]; 7]` const table; the 7 rows are the unique
/// permutations for rounds 0..6 (rounds 7..9 reuse rows 1..3). The const
/// table is read by direct `SCHEDULE[r][s]` indexing — the inliner
/// propagates it into the round body, so no runtime table indirection
/// occurs.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) mod avx2_parent {
    use core::arch::x86_64::*;
    use crate::merkle::{blake3_parent_cv, Hash};

    macro_rules! parent_xor_schedule {
        () => { [
            [ 0u8,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [ 2,    6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8],
            [ 3,    4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1],
            [10,    7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6],
            [12,   13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4],
            [ 9,   14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7],
            [11,   15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13],
        ] };
    }
    const SCHEDULE: [[u8; 16]; 7] = parent_xor_schedule!();

    const BLAKE3_IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ];
    const BLAKE3_PARENT_FLAGS: u8 = 4;

    #[inline(always)]
    unsafe fn add32(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi32(a, b)
    }
    #[inline(always)]
    unsafe fn xor32(a: __m256i, b: __m256i) -> __m256i {
        _mm256_xor_si256(a, b)
    }
    #[inline(always)]
    unsafe fn rot16(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32(x, 16), _mm256_slli_epi32(x, 16))
    }
    #[inline(always)]
    unsafe fn rot12(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 20))
    }
    #[inline(always)]
    unsafe fn rot8(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32(x, 8), _mm256_slli_epi32(x, 24))
    }
    #[inline(always)]
    unsafe fn rot7(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25))
    }

    /// One BLAKE3 G-function round. Body transcribed from the official
    /// blake3 AVX2 round (`rust_avx2::round`), indexed by the const
    /// `SCHEDULE[round]` table.
    #[inline(always)]
    unsafe fn round_fn(v: &mut [__m256i; 16], m: &[__m256i; 16], r: usize) {
        let s = &SCHEDULE[r];
        v[0] = add32(v[0], m[s[0] as usize]);
        v[1] = add32(v[1], m[s[2] as usize]);
        v[2] = add32(v[2], m[s[4] as usize]);
        v[3] = add32(v[3], m[s[6] as usize]);
        v[0] = add32(v[0], v[4]);
        v[1] = add32(v[1], v[5]);
        v[2] = add32(v[2], v[6]);
        v[3] = add32(v[3], v[7]);
        v[12] = xor32(v[12], v[0]);
        v[13] = xor32(v[13], v[1]);
        v[14] = xor32(v[14], v[2]);
        v[15] = xor32(v[15], v[3]);
        v[12] = rot16(v[12]);
        v[13] = rot16(v[13]);
        v[14] = rot16(v[14]);
        v[15] = rot16(v[15]);
        v[8] = add32(v[8], v[12]);
        v[9] = add32(v[9], v[13]);
        v[10] = add32(v[10], v[14]);
        v[11] = add32(v[11], v[15]);
        v[4] = xor32(v[4], v[8]);
        v[5] = xor32(v[5], v[9]);
        v[6] = xor32(v[6], v[10]);
        v[7] = xor32(v[7], v[11]);
        v[4] = rot12(v[4]);
        v[5] = rot12(v[5]);
        v[6] = rot12(v[6]);
        v[7] = rot12(v[7]);
        v[0] = add32(v[0], m[s[1] as usize]);
        v[1] = add32(v[1], m[s[3] as usize]);
        v[2] = add32(v[2], m[s[5] as usize]);
        v[3] = add32(v[3], m[s[7] as usize]);
        v[0] = add32(v[0], v[4]);
        v[1] = add32(v[1], v[5]);
        v[2] = add32(v[2], v[6]);
        v[3] = add32(v[3], v[7]);
        v[12] = xor32(v[12], v[0]);
        v[13] = xor32(v[13], v[1]);
        v[14] = xor32(v[14], v[2]);
        v[15] = xor32(v[15], v[3]);
        v[12] = rot8(v[12]);
        v[13] = rot8(v[13]);
        v[14] = rot8(v[14]);
        v[15] = rot8(v[15]);
        v[8] = add32(v[8], v[12]);
        v[9] = add32(v[9], v[13]);
        v[10] = add32(v[10], v[14]);
        v[11] = add32(v[11], v[15]);
        v[4] = xor32(v[4], v[8]);
        v[5] = xor32(v[5], v[9]);
        v[6] = xor32(v[6], v[10]);
        v[7] = xor32(v[7], v[11]);
        v[4] = rot7(v[4]);
        v[5] = rot7(v[5]);
        v[6] = rot7(v[6]);
        v[7] = rot7(v[7]);
        v[0] = add32(v[0], m[s[8] as usize]);
        v[1] = add32(v[1], m[s[10] as usize]);
        v[2] = add32(v[2], m[s[12] as usize]);
        v[3] = add32(v[3], m[s[14] as usize]);
        v[0] = add32(v[0], v[5]);
        v[1] = add32(v[1], v[6]);
        v[2] = add32(v[2], v[7]);
        v[3] = add32(v[3], v[4]);
        v[15] = xor32(v[15], v[0]);
        v[12] = xor32(v[12], v[1]);
        v[13] = xor32(v[13], v[2]);
        v[14] = xor32(v[14], v[3]);
        v[15] = rot16(v[15]);
        v[12] = rot16(v[12]);
        v[13] = rot16(v[13]);
        v[14] = rot16(v[14]);
        v[10] = add32(v[10], v[15]);
        v[11] = add32(v[11], v[12]);
        v[8] = add32(v[8], v[13]);
        v[9] = add32(v[9], v[14]);
        v[5] = xor32(v[5], v[10]);
        v[6] = xor32(v[6], v[11]);
        v[7] = xor32(v[7], v[8]);
        v[4] = xor32(v[4], v[9]);
        v[5] = rot12(v[5]);
        v[6] = rot12(v[6]);
        v[7] = rot12(v[7]);
        v[4] = rot12(v[4]);
        v[0] = add32(v[0], m[s[9] as usize]);
        v[1] = add32(v[1], m[s[11] as usize]);
        v[2] = add32(v[2], m[s[13] as usize]);
        v[3] = add32(v[3], m[s[15] as usize]);
        v[0] = add32(v[0], v[5]);
        v[1] = add32(v[1], v[6]);
        v[2] = add32(v[2], v[7]);
        v[3] = add32(v[3], v[4]);
        v[15] = xor32(v[15], v[0]);
        v[12] = xor32(v[12], v[1]);
        v[13] = xor32(v[13], v[2]);
        v[14] = xor32(v[14], v[3]);
        v[15] = rot8(v[15]);
        v[12] = rot8(v[12]);
        v[13] = rot8(v[13]);
        v[14] = rot8(v[14]);
        v[10] = add32(v[10], v[15]);
        v[11] = add32(v[11], v[12]);
        v[8] = add32(v[8], v[13]);
        v[9] = add32(v[9], v[14]);
        v[5] = xor32(v[5], v[10]);
        v[6] = xor32(v[6], v[11]);
        v[7] = xor32(v[7], v[8]);
        v[4] = xor32(v[4], v[9]);
        v[5] = rot7(v[5]);
        v[6] = rot7(v[6]);
        v[7] = rot7(v[7]);
        v[4] = rot7(v[4]);
    }

    /// 4×4 32-bit transpose: input is 4 `__m128i` each holding 4 u32 in
    /// lanes 0..3; output is 4 `__m128i` where output `i` is
    /// `(in0[i], in1[i], in2[i], in3[i])`.
    #[inline(always)]
    unsafe fn transpose4x4_32(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> [__m128i; 4] {
        let t0 = _mm_unpacklo_epi32(a, b);
        let t1 = _mm_unpackhi_epi32(a, b);
        let t2 = _mm_unpacklo_epi32(c, d);
        let t3 = _mm_unpackhi_epi32(c, d);
        [
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        ]
    }

    /// Transpose 4 parents' 16-word messages (256 bytes total) into
    /// `m[0..16]` such that `m[s]` holds lane i = word s of parent i.
    /// Lanes 4..7 of each m are zeroed.
    #[inline(always)]
    unsafe fn load4_transpose(
        p0: *const u8,
        p1: *const u8,
        p2: *const u8,
        p3: *const u8,
    ) -> [__m256i; 16] {
        // Each parent = 64 bytes = 16 u32 LE. 8 __m256i cover all 4
        // parents' messages exactly.
        let v0 = _mm256_loadu_si256(p0 as *const __m256i);
        let v1 = _mm256_loadu_si256(p0.add(32) as *const __m256i);
        let v2 = _mm256_loadu_si256(p1 as *const __m256i);
        let v3 = _mm256_loadu_si256(p1.add(32) as *const __m256i);
        let v4 = _mm256_loadu_si256(p2 as *const __m256i);
        let v5 = _mm256_loadu_si256(p2.add(32) as *const __m256i);
        let v6 = _mm256_loadu_si256(p3 as *const __m256i);
        let v7 = _mm256_loadu_si256(p3.add(32) as *const __m256i);

        // Per half, transpose the 4 parents' 4-u32 columns to 4 m-vectors.
        // Lo half (words 0..7):
        let m_lo = transpose4x4_32(
            _mm256_castsi256_si128(v0),
            _mm256_castsi256_si128(v2),
            _mm256_castsi256_si128(v4),
            _mm256_castsi256_si128(v6),
        );
        let m_hi = transpose4x4_32(
            _mm256_extracti128_si256(v0, 1),
            _mm256_extracti128_si256(v2, 1),
            _mm256_extracti128_si256(v4, 1),
            _mm256_extracti128_si256(v6, 1),
        );
        // Hi half (words 8..15):
        let m_lo2 = transpose4x4_32(
            _mm256_castsi256_si128(v1),
            _mm256_castsi256_si128(v3),
            _mm256_castsi256_si128(v5),
            _mm256_castsi256_si128(v7),
        );
        let m_hi2 = transpose4x4_32(
            _mm256_extracti128_si256(v1, 1),
            _mm256_extracti128_si256(v3, 1),
            _mm256_extracti128_si256(v5, 1),
            _mm256_extracti128_si256(v7, 1),
        );

        [
            _mm256_zextsi128_si256(m_lo[0]),
            _mm256_zextsi128_si256(m_lo[1]),
            _mm256_zextsi128_si256(m_lo[2]),
            _mm256_zextsi128_si256(m_lo[3]),
            _mm256_zextsi128_si256(m_hi[0]),
            _mm256_zextsi128_si256(m_hi[1]),
            _mm256_zextsi128_si256(m_hi[2]),
            _mm256_zextsi128_si256(m_hi[3]),
            _mm256_zextsi128_si256(m_lo2[0]),
            _mm256_zextsi128_si256(m_lo2[1]),
            _mm256_zextsi128_si256(m_lo2[2]),
            _mm256_zextsi128_si256(m_lo2[3]),
            _mm256_zextsi128_si256(m_hi2[0]),
            _mm256_zextsi128_si256(m_hi2[1]),
            _mm256_zextsi128_si256(m_hi2[2]),
            _mm256_zextsi128_si256(m_hi2[3]),
        ]
    }

    /// 4×1-way AVX2 parent-compress: produce 4 parents' 32-byte CVs.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn compress4_parents(
        p0: *const u8,
        p1: *const u8,
        p2: *const u8,
        p3: *const u8,
        o0: *mut Hash,
        o1: *mut Hash,
        o2: *mut Hash,
        o3: *mut Hash,
    ) {
        let mut v = [
            _mm256_set1_epi32(BLAKE3_IV[0] as i32),
            _mm256_set1_epi32(BLAKE3_IV[1] as i32),
            _mm256_set1_epi32(BLAKE3_IV[2] as i32),
            _mm256_set1_epi32(BLAKE3_IV[3] as i32),
            _mm256_set1_epi32(BLAKE3_IV[4] as i32),
            _mm256_set1_epi32(BLAKE3_IV[5] as i32),
            _mm256_set1_epi32(BLAKE3_IV[6] as i32),
            _mm256_set1_epi32(BLAKE3_IV[7] as i32),
            _mm256_set1_epi32(BLAKE3_IV[0] as i32),
            _mm256_set1_epi32(BLAKE3_IV[1] as i32),
            _mm256_set1_epi32(BLAKE3_IV[2] as i32),
            _mm256_set1_epi32(BLAKE3_IV[3] as i32),
            _mm256_set1_epi32(0i32),
            _mm256_set1_epi32(0i32),
            _mm256_set1_epi32(64i32),
            _mm256_set1_epi32(BLAKE3_PARENT_FLAGS as i32),
        ];

        let m = load4_transpose(p0, p1, p2, p3);

        // 7 unique rows; rounds 7..9 reuse rows 1..3.
        round_fn(&mut v, &m, 0);
        round_fn(&mut v, &m, 1);
        round_fn(&mut v, &m, 2);
        round_fn(&mut v, &m, 3);
        round_fn(&mut v, &m, 4);
        round_fn(&mut v, &m, 5);
        round_fn(&mut v, &m, 6);
        round_fn(&mut v, &m, 1);
        round_fn(&mut v, &m, 2);
        round_fn(&mut v, &m, 3);

        // XOR v[0..8] with v[8..16] → 8 __m256i, each holding 4 parents'
        // value for one CV word position in lanes 0..3 (and zeros in 4..7
        // because the CV-XOR'd halves have the same lane pattern).
        let cv0 = xor32(v[0], v[8]);
        let cv1 = xor32(v[1], v[9]);
        let cv2 = xor32(v[2], v[10]);
        let cv3 = xor32(v[3], v[11]);
        let cv4 = xor32(v[4], v[12]);
        let cv5 = xor32(v[5], v[13]);
        let cv6 = xor32(v[6], v[14]);
        let cv7 = xor32(v[7], v[15]);

        // 8×4 transpose: for each parent p, output p is the concatenation
        // of cv{0..7}'s lane p. Apply 4×4 32-bit transpose to the lo halves
        // and hi halves of cv0..cv3 (giving 4 parents' word positions 0..3)
        // and cv4..cv7 (giving 4 parents' word positions 4..7). Then merge
        // each parent's lo and hi halves into one __m256i for store.
        let lo0_3 = transpose4x4_32(
            _mm256_castsi256_si128(cv0),
            _mm256_castsi256_si128(cv1),
            _mm256_castsi256_si128(cv2),
            _mm256_castsi256_si128(cv3),
        );
        let hi0_3 = transpose4x4_32(
            _mm256_extracti128_si256(cv0, 1),
            _mm256_extracti128_si256(cv1, 1),
            _mm256_extracti128_si256(cv2, 1),
            _mm256_extracti128_si256(cv3, 1),
        );
        let lo4_7 = transpose4x4_32(
            _mm256_castsi256_si128(cv4),
            _mm256_castsi256_si128(cv5),
            _mm256_castsi256_si128(cv6),
            _mm256_castsi256_si128(cv7),
        );
        let hi4_7 = transpose4x4_32(
            _mm256_extracti128_si256(cv4, 1),
            _mm256_extracti128_si256(cv5, 1),
            _mm256_extracti128_si256(cv6, 1),
            _mm256_extracti128_si256(cv7, 1),
        );

        // Each parent p = (lo0_3[p], lo4_7[p], hi0_3[p], hi4_7[p]) in
        // word-position order. The two lo_* together = words 0..3 in the
        // low 128-bit lane, hi_* together = words 4..7 in the high 128-bit
        // lane of the final __m256i for that parent.
        // Build each parent's 8 u32 with two permute2x128 (one for words
        // 0..3 in low 128, one for words 4..7 in high 128) merged again.
        // Simpler: pack lo+hi per parent in one permute2x128 with imm 0x20.
        _mm256_storeu_si256(
            o0 as *mut __m256i,
            _mm256_permute2x128_si256(
                _mm256_castsi128_si256(lo0_3[0]),
                _mm256_castsi128_si256(hi0_3[0]),
                0x20,
            ),
        );
        _mm256_storeu_si256(
            o1 as *mut __m256i,
            _mm256_permute2x128_si256(
                _mm256_castsi128_si256(lo0_3[1]),
                _mm256_castsi128_si256(hi0_3[1]),
                0x20,
            ),
        );
        _mm256_storeu_si256(
            o2 as *mut __m256i,
            _mm256_permute2x128_si256(
                _mm256_castsi128_si256(lo0_3[2]),
                _mm256_castsi128_si256(hi0_3[2]),
                0x20,
            ),
        );
        _mm256_storeu_si256(
            o3 as *mut __m256i,
            _mm256_permute2x128_si256(
                _mm256_castsi128_si256(lo0_3[3]),
                _mm256_castsi128_si256(hi0_3[3]),
                0x20,
            ),
        );
        let _ = (lo4_7, hi4_7);
    }

    /// Batched parent compress over `out.len()` parents. Processes 4 at a
    /// time via [`compress4_parents`]; falls back to the scalar path for
    /// 0..3 leftover parents.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn compress4_parents_many(data: &[u8], out: &mut [Hash]) {
        debug_assert_eq!(data.len(), out.len() * 64);
        let n = out.len();
        let chunks = n / 4;
        for i in 0..chunks {
            let p = data.as_ptr().add(i * 4 * 64);
            let o = out.as_mut_ptr().add(i * 4);
            compress4_parents(
                p,
                p.add(64),
                p.add(128),
                p.add(192),
                o,
                o.add(1),
                o.add(2),
                o.add(3),
            );
        }
        for j in (chunks * 4)..n {
            let l: &Hash = data[j * 64..j * 64 + 32].try_into().unwrap();
            let r: &Hash = data[j * 64 + 32..(j + 1) * 64].try_into().unwrap();
            out[j] = blake3_parent_cv(l, r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant, for tests that sweep both.
    pub(crate) const ALL: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

    #[test]
    fn parses_and_round_trips() {
        for kind in ALL {
            assert_eq!(HashKind::parse(kind.as_str()).unwrap(), kind);
            assert_eq!(kind.to_string(), kind.as_str());
        }
        assert_eq!(HashKind::parse("BLAKE3").unwrap(), HashKind::Blake3);
        assert_eq!(HashKind::parse("sha-256").unwrap(), HashKind::Sha256);
        assert_eq!(HashKind::parse("  blake3 ").unwrap(), HashKind::Blake3);
        assert_eq!(HashKind::default(), HashKind::Sha256);
        // An unrecognized hash must be an error, never a silent SHA-256.
        assert!(HashKind::parse("keccak").is_err());
        assert!(HashKind::parse("").is_err());
    }

    #[test]
    fn serde_uses_config_spellings() {
        for kind in ALL {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{}\"", kind.as_str()));
            assert_eq!(
                serde_json::from_str::<HashKind>(&json).unwrap(),
                kind,
                "{kind}"
            );
        }
    }
}
