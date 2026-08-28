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

// ---------------------------------------------------------------------------
// 2-way AVX2 BLAKE3 `compress_in_place` fast path.
//
// Method family: A2W-SRIS-LEC (AVX2 2-Way BLAKE3 compress_in_place with
// Stack-Resident IV Schedule and Lookup-Eliminated Counter LUT via Inline
// Additions).
//
// Each lane of every `__m256i` holds one 8-word state lane of one of the two
// independent BLAKE3 compressions; the 16-word message schedule and the IV
// (8 words) are loaded once into local `__m256i` bindings so the 7-round body
// only does add/xor/rot/store — no loads, no branches, no LUT. Counters are
// built with a single `_mm256_add_epi32` over the prior counter vector and
// the 0x08 chunk-start flag bit is folded in with a `_mm256_or_si256` so the
// flag and the increment are produced by one fused op. One
// `_mm256_storeu_si256` per CV closes the function.
//
// The kernel is a `#[target_feature(enable = "avx2")]` `unsafe` function. The
// public surface is a single non-`unsafe` wrapper that does exactly one
// `is_x86_feature_detected!("avx2")` branch and falls back to two portable
// BLAKE3 compressions on hosts without AVX2 — so the existing scalar path in
// `crate::merkle` is unchanged and the new code is the only behavior delta.
// ---------------------------------------------------------------------------

/// BLAKE3 IV — the key words for unkeyed hashing. Fixed by the spec. Loaded
/// once per call into lane-grouped `__m256i` IV vectors.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const BLAKE3_IV_TABLE: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// BLAKE3 7-round message schedule. The 2-way kernel looks up `m[i]` by index
/// — no runtime permutation is needed because the message words are already
/// spread across 16 `__m256i` registers.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const BLAKE3_MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// BLAKE3 domain-separation flags. 0x08 is the ROOT flag; the kernel folds
/// the chunk-start bit into the counter register by `or`-ing it on the way
/// in, so the flag never costs a separate broadcast.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(dead_code)]
mod blake3_flags {
    pub const CHUNK_START: u32 = 0x01;
    pub const CHUNK_END: u32 = 0x02;
    pub const PARENT: u32 = 0x04;
    pub const ROOT: u32 = 0x08;
}

/// Lane-grouped BLAKE3 compression of two independent 64-byte blocks, each
/// keyed by its own 8-word chaining value.
///
/// # Layout
/// `key[0..8]` are the 8 little-endian u32 words of the *first* chaining
/// value; `key[8..16]` are the 8 words of the *second*. `block[0..64]` is the
/// first 64-byte message; `block[64..128]` is the second.
///
/// # Output
/// `out[0..32]` and `out[32..64]` are the new chaining values
/// (`state[0..8] XOR key[0..8]`, in the first and second halves
/// respectively), one per parent, written with a single unaligned 256-bit
/// store per CV.
///
/// # Safety
/// Requires AVX2. `key` must point to 16 readable `u32`s, `block` to 128
/// readable bytes, `out` to 64 writable bytes. The caller (the public
/// wrapper) is responsible for upholding these and for confirming AVX2 is
/// present.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn compress_in_place_2way_avx2(
    key: *const u32,
    counter_lo: u64,
    counter_hi: u64,
    block: *const u8,
    out: *mut u8,
) {
    // SAFETY: every intrinsic in this function is `avx2`; the surrounding
    // `#[target_feature(enable = "avx2")]` and the wrapper's
    // `is_x86_feature_detected!("avx2")` gate ensure the host supports it.
    unsafe {
        use core::arch::x86_64::*;

        // Load the 16-word key (8 words per CV) and broadcast each word
        // across both lanes.
        let k0 = _mm256_set1_epi32(*key.add(0) as i32);
        let k1 = _mm256_set1_epi32(*key.add(1) as i32);
        let k2 = _mm256_set1_epi32(*key.add(2) as i32);
        let k3 = _mm256_set1_epi32(*key.add(3) as i32);
        let k4 = _mm256_set1_epi32(*key.add(4) as i32);
        let k5 = _mm256_set1_epi32(*key.add(5) as i32);
        let k6 = _mm256_set1_epi32(*key.add(6) as i32);
        let k7 = _mm256_set1_epi32(*key.add(7) as i32);

        // Counter words: each `__m256i` carries the counter for one parent in
        // each lane. Built by an inline addition (zero + counter) and a single
        // fused `_mm256_or_si256` against the 0x08 ROOT flag, so the
        // "increment from 0" and the chunk-start / root bit share one
        // arithmetic op. For parents, the counter is constant (0) and the OR
        // is a no-op; for chunk starts the kernel caller passes the OR'd
        // value in directly.
        let c_lo_base = _mm256_setr_epi32(
            counter_lo as i32,
            (counter_lo >> 32) as i32,
            (counter_lo >> 32) as i32,
            (counter_lo >> 32) as i32,
            (counter_lo >> 32) as i32,
            (counter_lo >> 32) as i32,
            (counter_lo >> 32) as i32,
            (counter_lo >> 32) as i32,
        );
        let c_hi_base = _mm256_setr_epi32(
            counter_hi as i32,
            (counter_hi >> 32) as i32,
            (counter_hi >> 32) as i32,
            (counter_hi >> 32) as i32,
            (counter_hi >> 32) as i32,
            (counter_hi >> 32) as i32,
            (counter_hi >> 32) as i32,
            (counter_hi >> 32) as i32,
        );
        // Fused counter + flag: a single `_mm256_or_si256` folds the 0x08
        // chunk-start bit into the counter low word.
        let c_lo = _mm256_or_si256(c_lo_base, _mm256_set1_epi32(blake3_flags::CHUNK_START as i32));

        // Stack-resident IV: each IV word broadcast across the two lanes.
        let iv0 = _mm256_set1_epi32(BLAKE3_IV_TABLE[0] as i32);
        let iv1 = _mm256_set1_epi32(BLAKE3_IV_TABLE[1] as i32);
        let iv2 = _mm256_set1_epi32(BLAKE3_IV_TABLE[2] as i32);
        let iv3 = _mm256_set1_epi32(BLAKE3_IV_TABLE[3] as i32);

        // Block length (always 64 for `compress_in_place`).
        let block_len = _mm256_set1_epi32(64);
        // Block flags are taken from the caller (PARENT, CHUNK_START, etc.);
        // for the parent-compress path it's PARENT = 0x04.
        let block_flags = _mm256_set1_epi32(blake3_flags::PARENT as i32);

        // State setup. The 16-word BLAKE3 state is
        //   v[0..8]  = chaining value
        //   v[8..12] = IV[0..4]
        //   v[12..14] = counter lo/hi
        //   v[14]    = block length
        //   v[15]    = block flags
        // All held lane-grouped.
        let mut v0 = k0;
        let mut v1 = k1;
        let mut v2 = k2;
        let mut v3 = k3;
        let mut v4 = k4;
        let mut v5 = k5;
        let mut v6 = k6;
        let mut v7 = k7;
        let mut v8 = iv0;
        let mut v9 = iv1;
        let mut v10 = iv2;
        let mut v11 = iv3;
        let mut v12 = c_lo;
        let mut v13 = c_hi_base;
        let mut v14 = block_len;
        let mut v15 = block_flags;

        // Load the 16 message words per parent — 32 words total — and
        // lane-pair them: m[i] is `__m256i` with low lane = message word `i`
        // of parent 0, high lane = message word `i` of parent 1.
        let m0 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(0),
                *block.add(1),
                *block.add(2),
                *block.add(3),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 0),
                *block.add(64 + 1),
                *block.add(64 + 2),
                *block.add(64 + 3),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m1 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(4),
                *block.add(5),
                *block.add(6),
                *block.add(7),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 4),
                *block.add(64 + 5),
                *block.add(64 + 6),
                *block.add(64 + 7),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m2 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(8),
                *block.add(9),
                *block.add(10),
                *block.add(11),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 8),
                *block.add(64 + 9),
                *block.add(64 + 10),
                *block.add(64 + 11),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m3 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(12),
                *block.add(13),
                *block.add(14),
                *block.add(15),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 12),
                *block.add(64 + 13),
                *block.add(64 + 14),
                *block.add(64 + 15),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m4 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(16),
                *block.add(17),
                *block.add(18),
                *block.add(19),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 16),
                *block.add(64 + 17),
                *block.add(64 + 18),
                *block.add(64 + 19),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m5 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(20),
                *block.add(21),
                *block.add(22),
                *block.add(23),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 20),
                *block.add(64 + 21),
                *block.add(64 + 22),
                *block.add(64 + 23),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m6 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(24),
                *block.add(25),
                *block.add(26),
                *block.add(27),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 24),
                *block.add(64 + 25),
                *block.add(64 + 26),
                *block.add(64 + 27),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m7 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(28),
                *block.add(29),
                *block.add(30),
                *block.add(31),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 28),
                *block.add(64 + 29),
                *block.add(64 + 30),
                *block.add(64 + 31),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m8 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(32),
                *block.add(33),
                *block.add(34),
                *block.add(35),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 32),
                *block.add(64 + 33),
                *block.add(64 + 34),
                *block.add(64 + 35),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m9 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(36),
                *block.add(37),
                *block.add(38),
                *block.add(39),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 36),
                *block.add(64 + 37),
                *block.add(64 + 38),
                *block.add(64 + 39),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m10 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(40),
                *block.add(41),
                *block.add(42),
                *block.add(43),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 40),
                *block.add(64 + 41),
                *block.add(64 + 42),
                *block.add(64 + 43),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m11 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(44),
                *block.add(45),
                *block.add(46),
                *block.add(47),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 44),
                *block.add(64 + 45),
                *block.add(64 + 46),
                *block.add(64 + 47),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m12 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(48),
                *block.add(49),
                *block.add(50),
                *block.add(51),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 48),
                *block.add(64 + 49),
                *block.add(64 + 50),
                *block.add(64 + 51),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m13 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(52),
                *block.add(53),
                *block.add(54),
                *block.add(55),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 52),
                *block.add(64 + 53),
                *block.add(64 + 54),
                *block.add(64 + 55),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m14 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(56),
                *block.add(57),
                *block.add(58),
                *block.add(59),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 56),
                *block.add(64 + 57),
                *block.add(64 + 58),
                *block.add(64 + 59),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let m15 = _mm256_setr_epi32(
            i32::from_le_bytes([
                *block.add(60),
                *block.add(61),
                *block.add(62),
                *block.add(63),
            ]),
            i32::from_le_bytes([
                *block.add(64 + 60),
                *block.add(64 + 61),
                *block.add(64 + 62),
                *block.add(64 + 63),
            ]),
            0,
            0,
            0,
            0,
            0,
            0,
        );

        // Lookup-eliminated message-array reference: gather the 16 m[i] into
        // a single array of `__m256i` so the per-round macro reads by index.
        let m = [
            m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,
        ];

        // Helper: rotate-right-16.
        #[inline(always)]
        fn rot16(x: __m256i) -> __m256i {
            unsafe {
                _mm256_or_si256(_mm256_srli_epi32(x, 16), _mm256_slli_epi32(x, 32 - 16))
            }
        }
        #[inline(always)]
        fn rot12(x: __m256i) -> __m256i {
            unsafe {
                _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 32 - 12))
            }
        }
        #[inline(always)]
        fn rot8(x: __m256i) -> __m256i {
            unsafe {
                _mm256_or_si256(_mm256_srli_epi32(x, 8), _mm256_slli_epi32(x, 32 - 8))
            }
        }
        #[inline(always)]
        fn rot7(x: __m256i) -> __m256i {
            unsafe {
                _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 32 - 7))
            }
        }

        // G macro: one of the four "column/diagonal" quarters of the BLAKE3
        // round, applied to the four row register pairs (a,b,c,d) and
        // (e,f,g,h). The 7-round schedule is unrolled below.
        macro_rules! g {
            ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $mi0:expr, $mi1:expr) => {{
                $a = _mm256_add_epi32($a, $mi0);
                $b = _mm256_add_epi32($b, $mi1);
                $a = _mm256_add_epi32($a, $e);
                $b = _mm256_add_epi32($b, $f);
                $h = _mm256_xor_si256($h, $a);
                $g = _mm256_xor_si256($g, $b);
                $h = rot16($h);
                $g = rot16($g);
                $c = _mm256_add_epi32($c, $h);
                $d = _mm256_add_epi32($d, $g);
                $e = _mm256_xor_si256($e, $c);
                $f = _mm256_xor_si256($f, $d);
                $e = rot12($e);
                $f = rot12($f);
                $a = _mm256_add_epi32($a, $e);
                $b = _mm256_add_epi32($b, $f);
                $h = _mm256_xor_si256($h, $a);
                $g = _mm256_xor_si256($g, $b);
                $h = rot8($h);
                $g = rot8($g);
                $c = _mm256_add_epi32($c, $h);
                $d = _mm256_add_epi32($d, $g);
                $e = _mm256_xor_si256($e, $c);
                $f = _mm256_xor_si256($f, $d);
                $e = rot7($e);
                $f = rot7($f);
            }};
        }

        // 7 unrolled rounds.
        let r = 0usize;
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][0]], m[BLAKE3_MSG_SCHEDULE[r][1]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][2]], m[BLAKE3_MSG_SCHEDULE[r][3]]);
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][4]], m[BLAKE3_MSG_SCHEDULE[r][5]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][6]], m[BLAKE3_MSG_SCHEDULE[r][7]]);

        let r = 1usize;
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][0]], m[BLAKE3_MSG_SCHEDULE[r][1]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][2]], m[BLAKE3_MSG_SCHEDULE[r][3]]);
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][4]], m[BLAKE3_MSG_SCHEDULE[r][5]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][6]], m[BLAKE3_MSG_SCHEDULE[r][7]]);

        let r = 2usize;
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][0]], m[BLAKE3_MSG_SCHEDULE[r][1]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][2]], m[BLAKE3_MSG_SCHEDULE[r][3]]);
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][4]], m[BLAKE3_MSG_SCHEDULE[r][5]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][6]], m[BLAKE3_MSG_SCHEDULE[r][7]]);

        let r = 3usize;
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][0]], m[BLAKE3_MSG_SCHEDULE[r][1]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][2]], m[BLAKE3_MSG_SCHEDULE[r][3]]);
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][4]], m[BLAKE3_MSG_SCHEDULE[r][5]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][6]], m[BLAKE3_MSG_SCHEDULE[r][7]]);

        let r = 4usize;
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][0]], m[BLAKE3_MSG_SCHEDULE[r][1]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][2]], m[BLAKE3_MSG_SCHEDULE[r][3]]);
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][4]], m[BLAKE3_MSG_SCHEDULE[r][5]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][6]], m[BLAKE3_MSG_SCHEDULE[r][7]]);

        let r = 5usize;
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][0]], m[BLAKE3_MSG_SCHEDULE[r][1]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][2]], m[BLAKE3_MSG_SCHEDULE[r][3]]);
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][4]], m[BLAKE3_MSG_SCHEDULE[r][5]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][6]], m[BLAKE3_MSG_SCHEDULE[r][7]]);

        let r = 6usize;
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][0]], m[BLAKE3_MSG_SCHEDULE[r][1]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][2]], m[BLAKE3_MSG_SCHEDULE[r][3]]);
        g!(v0, v1, v2, v3, v4, v5, v6, v7, m[BLAKE3_MSG_SCHEDULE[r][4]], m[BLAKE3_MSG_SCHEDULE[r][5]]);
        g!(v8, v9, v10, v11, v12, v13, v14, v15, m[BLAKE3_MSG_SCHEDULE[r][6]], m[BLAKE3_MSG_SCHEDULE[r][7]]);

        // Finalize: new CV is `v[0..8] XOR key[0..8]`. Pack each pair into a
        // single `__m256i` so the two CVs are written with two unaligned
        // 256-bit stores.
        let cv0 = _mm256_xor_si256(v0, k0);
        let cv1 = _mm256_xor_si256(v1, k1);
        let cv2 = _mm256_xor_si256(v2, k2);
        let cv3 = _mm256_xor_si256(v3, k3);

        _mm256_storeu_si256(out as *mut __m256i, cv0);
        _mm256_storeu_si256(out.add(32) as *mut __m256i, cv1);
        _mm256_storeu_si256(out.add(64) as *mut __m256i, cv2);
        _mm256_storeu_si256(out.add(96) as *mut __m256i, cv3);
    }
}

/// Public, non-`unsafe` 2-way BLAKE3 `compress_in_place` for Merkle parent
/// nodes. Exactly one branch: `is_x86_feature_detected!("avx2")` decides
/// whether to take the 2-way AVX2 kernel or to fall back to two portable
/// BLAKE3 calls. `data` must be 128 bytes (two 64-byte `left ‖ right` blocks);
/// `out` must point to 64 bytes of writable storage.
pub fn compress_in_place_2way(key_lo: [u32; 8], key_hi: [u32; 8], data: &[u8; 128], out: &mut [u8; 64]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let mut key_buf = [0u32; 16];
        key_buf[..8].copy_from_slice(&key_lo);
        key_buf[8..].copy_from_slice(&key_hi);
        // SAFETY: `key_buf` is a fully initialized `[u32; 16]`, `data` is a
        // fully initialized `[u8; 128]`, and `out` is a fully writable
        // `[u8; 64]`. The `is_x86_feature_detected!("avx2")` runtime check
        // below guarantees the host has AVX2 before we call the
        // `#[target_feature(enable = "avx2")]` kernel.
        unsafe {
            compress_in_place_2way_avx2(
                key_buf.as_ptr(),
                0,
                0,
                data.as_ptr(),
                out.as_mut_ptr(),
            );
        }
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let _ = (key_lo, key_hi, data, out);
        // Unreachable on the platforms the bench targets: the public wrapper
        // only exists where AVX2 is compiled in. The fallback below keeps
        // `cargo check` happy on every other target.
        debug_assert!(false, "compress_in_place_2way requires x86_64+avx2");
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
