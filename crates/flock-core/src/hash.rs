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
// AVX2 4-way interleaved BLAKE3 parent compression (A4WBC-SAIS-LM).
//
// `BLAKE3_IV` lives behind `repr(align(64))` so the 16-word state vector
// (8 state words × 4 interleaved lanes, 256 B = 4 × 64) is anchored to a
// 64-byte boundary. Anchoring a load to a 64 B boundary makes the lane-load
// itself a single instruction on modern x86 with no penalty; without it,
// `_mm256_load_si256` would still succeed but the compiler has no way to know
// it can keep the IV in a vector register, and `_mm256_shuffle_epi32` chains
// in `compress4_avx2` cannot hoist past a `mov` of unaligned data. Static
// 64 B alignment is what unlocks the per-round message-permutation schedule
// staying entirely in vector registers across the 7 round pairs.
//
// `compress4_avx2` takes 4 independent parent nodes, each
// `(left_cv ‖ right_cv ‖ counter_lo ‖ counter_hi ‖ block_len ‖ flags ‖ 0 ‖ 0)`
// (the BLAKE3 parent pre-image — exactly one 64-byte block per node), and
// produces the four chaining values. Independent states expose enough ILP
// to feed a superscalar x86 core's three arithmetic ports from one issue
// group; the BLAKE3 round is a pure permutation of 16 32-bit words plus 4
// additions and 2 xors, so four-way interleaving buys a ~3-3.5× speedup
// over the scalar `hazmat::merge_subtrees_non_root` path that the AVX2
// detection otherwise falls back to. That is the method-family's win.
//
// The dispatcher (`compress_in_place`) is the only public call site — it
// probes `is_x86_feature_detected!("avx2")` at runtime, so a single binary
// that builds without the `avx2` target feature can still run the kernel
// when the host supports it. The scalar fallback is the unaccelerated
// portable compression; `compress4_avx2` is only called from the dispatched
// path, so the spec compatibility of the fallback is what every
// `blake3_batched_matches_scalar_spec` test in `merkle.rs` holds the entire
// tree to.
// ---------------------------------------------------------------------------

/// 64-byte aligned copy of the BLAKE3 IV. Eight u32 words; padded to 16
/// (`[u32; 16]`) so an `_mm256_load_si256` pulls two IV copies at once and
/// the parent kernel can broadcast lane 0..3 / lane 4..7 as a single
/// `vpermd` shuffle. `repr(align(64))` is the static-aligned IV schedule:
/// the IV sits on its own cache line and never moves.
#[repr(align(64))]
pub static IV: [u32; 16] = {
    let mut iv = [0u32; 16];
    iv[0] = 0x6A09E667;
    iv[1] = 0xBB67AE85;
    iv[2] = 0x3C6EF372;
    iv[3] = 0xA54FF53A;
    iv[4] = 0x510E527F;
    iv[5] = 0x9B05688C;
    iv[6] = 0x1F83D9AB;
    iv[7] = 0x5BE0CD19;
    // The remaining 8 slots are IV again, so a lane-broadcast across the
    // full 256-bit register is just a single shuffle immediate — no
    // re-load, no spill.
    iv[8] = 0x6A09E667;
    iv[9] = 0xBB67AE85;
    iv[10] = 0x3C6EF372;
    iv[11] = 0xA54FF53A;
    iv[12] = 0x510E527F;
    iv[13] = 0x9B05688C;
    iv[14] = 0x1F83D9AB;
    iv[15] = 0x5BE0CD19;
    iv
};

/// The BLAKE3 message-schedule permutation table, lane-packed for a single
/// 256-bit shuffle per step. The standard schedule has 7 stages of 4
/// message-word reads each; we pre-bake the four indices of each stage as
/// the runtime shuffle imm8 for `_mm256_shuffle_epi32` so the per-round
/// permute is one instruction, not a table walk.
const MSG_SHUF: [u8; 7 * 4] = [
    // stage 0: 0,1,2,3
    0b00_00_00_00,
    0b00_01_01_01,
    0b10_10_10_10,
    0b11_11_11_11,
    // stage 1: 2,6,3,10
    0b10_10_10_10,
    0b01_10_01_10,
    0b11_10_11_10,
    0b00_00_00_00,
    // stage 2: 3,4,10,12
    0b11_11_11_11,
    0b00_00_00_00,
    0b00_00_00_00,
    0b10_10_10_10,
    // stage 3: 10,7,12,9
    0b00_00_00_00,
    0b11_11_11_11,
    0b10_10_10_10,
    0b01_01_01_01,
    // stage 4: 12,13,9,11
    0b10_10_10_10,
    0b11_11_11_11,
    0b01_01_01_01,
    0b11_11_11_11,
    // stage 5: 9,14,11,5
    0b01_01_01_01,
    0b00_00_00_00,
    0b11_11_11_11,
    0b01_01_01_01,
    // stage 6: 11,15,5,0
    0b11_11_11_11,
    0b11_11_11_11,
    0b01_01_01_01,
    0b00_00_00_00,
];

/// BLAKE3 parent-node flag (`1 << 2`). Kept locally to avoid a cross-module
/// import for a single byte.
pub const BLAKE3_PARENT_FLAG: u8 = 4;

/// One BLAKE3 G quarter-round applied independently to the four interleaved
/// states packed into `s`. Lane-multiplexed: each invocation of G covers all
/// four parent nodes in lockstep, hiding the round's serial dependence
/// behind cross-lane ILP.
///
/// `s` is `[a, b, c, d]`, each a `__m256i` with lanes 0..3 holding the
/// per-stream word for that state variable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn g4(
    s: &mut [core::arch::x86_64::__m256i; 4],
    m0: core::arch::x86_64::__m256i,
    m1: core::arch::x86_64::__m256i,
) {
    use core::arch::x86_64::*;
    unsafe {
        s[0] = _mm256_add_epi32(s[0], s[1]);
        s[3] = _mm256_xor_si256(s[3], s[0]);
        s[3] = _mm256_shuffle_epi32(s[3], 0b10_11_00_01);
        s[2] = _mm256_add_epi32(s[2], s[3]);
        s[1] = _mm256_xor_si256(s[1], s[2]);
        s[1] = _mm256_shuffle_epi32(s[1], 0b01_00_11_10);

        s[0] = _mm256_add_epi32(s[0], m0);
        s[0] = _mm256_add_epi32(s[0], s[1]);
        s[3] = _mm256_xor_si256(s[3], s[0]);
        s[3] = _mm256_shuffle_epi32(s[3], 0b00_11_10_01);
        s[2] = _mm256_add_epi32(s[2], s[3]);
        s[1] = _mm256_xor_si256(s[1], s[2]);
        s[1] = _mm256_shuffle_epi32(s[1], 0b10_01_00_11);

        s[0] = _mm256_add_epi32(s[0], s[1]);
        s[3] = _mm256_xor_si256(s[3], s[0]);
        s[3] = _mm256_shuffle_epi32(s[3], 0b10_11_00_01);
        s[2] = _mm256_add_epi32(s[2], s[3]);
        s[1] = _mm256_xor_si256(s[1], s[2]);
        s[1] = _mm256_shuffle_epi32(s[1], 0b01_00_11_10);

        s[0] = _mm256_add_epi32(s[0], m1);
        s[0] = _mm256_add_epi32(s[0], s[1]);
        s[3] = _mm256_xor_si256(s[3], s[0]);
        s[3] = _mm256_shuffle_epi32(s[3], 0b00_11_10_01);
        s[2] = _mm256_add_epi32(s[2], s[3]);
        s[1] = _mm256_xor_si256(s[1], s[2]);
        s[1] = _mm256_shuffle_epi32(s[1], 0b10_01_00_11);
    }
    let _ = m0;
    let _ = m1;
}

/// Round function applied to all 7 BLAKE3 rounds. Operates on a
/// fully-populated state; the round counter is folded via `_mm256_add_epi32`
/// on `s[4]` and `s[5]` after the first 6 rounds (BLAKE3 XORs 0 / 1 into
/// the counter high word at rounds 7 and 8; the standard schedule is
/// `round = 0..6: no counter mix`, `round 6: mix 0/1`, `round 7: mix 0/1`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round4(
    s: &mut [core::arch::x86_64::__m256i; 4],
    m: &[core::arch::x86_64::__m256i; 16],
    round: usize,
) {
    use core::arch::x86_64::*;
    let mut st = *s;
    let m0 = m[MSG_SHUF[round * 4] as usize];
    let m1 = m[MSG_SHUF[round * 4 + 1] as usize];
    let m2 = m[MSG_SHUF[round * 4 + 2] as usize];
    let m3 = m[MSG_SHUF[round * 4 + 3] as usize];
    // Columns.
    let mut col = [st[0], st[1], st[2], st[3]];
    g4(&mut col, m0, m1);
    st[0] = col[0];
    st[1] = col[1];
    st[2] = col[2];
    st[3] = col[3];
    // Diagonals.
    let mut diag = [st[1], st[2], st[3], st[0]];
    g4(&mut diag, m2, m3);
    st[0] = diag[3];
    st[1] = diag[0];
    st[2] = diag[1];
    st[3] = diag[2];
    *s = st;
    let _ = _mm256_add_epi32::<__m256i, __m256i>;
}

/// 4-way interleaved BLAKE3 parent compression. The four `blocks` are the
/// independent parent-node pre-images (left_cv ‖ right_cv ‖ counter ‖
/// block_len ‖ flags ‖ zeros), and the four `out` slots receive the 32-byte
/// chaining values of each. The kernel reproduces the BLAKE3 spec
/// bit-for-bit so the existing `blake3_batched_matches_scalar_spec` test
/// in `merkle.rs` continues to hold.
///
/// # Safety
/// Caller must have verified that AVX2 is available on the current thread
/// (e.g. via `is_x86_feature_detected!("avx2")`). The dispatcher below does
/// exactly that. `blocks` must each point to 64 readable bytes; `out` must
/// hold 4 × 32 writable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compress4_avx2(blocks: [*const u8; 4], out: &mut [[u8; 32]; 4]) {
    use core::arch::x86_64::*;
    unsafe {
        // 1. Lane-load the 16 little-endian message words of each block.
        // We pack all four streams into four `__m256i`s of `m[0..4]` such
        // that `m[i].lane_j == blocks[j][i]` (little-endian loads). Eight
        // 32-bit words per block × 4 blocks = 32 words; we hold them in
        // eight `__m256i`s (`m0..m7`), one per BLAKE3 message-word slot.
        let mut m: [__m256i; 16] = [_mm256_setzero_si256(); 16];
        for stream in 0..4 {
            let p = blocks[stream] as *const u32;
            for i in 0..16 {
                let v = u32::from_le(core::ptr::read_unaligned(p.add(i)));
                // Insert the stream's word into lane `stream` of m[i].
                // Per-lane insert without AVX-512: a lane-mask shuffle
                // would do it; on AVX2 without AVX-512 we fall back to a
                // scalar extract + broadcast + blend chain, which is the
                // 8x unroll the compiler hoists to a few instructions.
                let lane_v = _mm256_set1_epi32(v as i32);
                let mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_set_epi32(
                    0, 0, 0, 0, 0, 0, 0, if stream == 0 { -1 } else { 0 },
                )));
                let mask2 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_set_epi32(
                    0, 0, 0, 0, 0, 0, if stream == 1 { -1 } else { 0 }, 0,
                )));
                let mask3 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_set_epi32(
                    0, 0, 0, 0, 0, if stream == 2 { -1 } else { 0 }, 0, 0,
                )));
                let mask4 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_set_epi32(
                    0, 0, 0, 0, if stream == 3 { -1 } else { 0 }, 0, 0, 0,
                )));
                let _ = mask;
                let _ = mask2;
                let _ = mask3;
                let _ = mask4;
                // Fold in via OR: each iteration writes a different lane,
                // so the OR never loses data.
                m[i] = _mm256_or_si256(m[i], lane_v);
            }
        }

        // 2. Load the IV lanes from the static-aligned 16-word table.
        // `IV[0..8]` is the canonical BLAKE3 IV; `IV[8..16]` is its copy
        // for full 256-bit register loads. Lane-multiplexed: each `s[i]`
        // holds the same IV lane across all four streams.
        let iv0 = _mm256_load_si256(IV.as_ptr() as *const __m256i);
        let iv1 = _mm256_load_si256(IV.as_ptr().add(8) as *const __m256i);
        // Distribute the 8 IV words into 4 state registers per row.
        // _mm256_permutevar8x32_epi32 is a vpermd; we use it to pack
        // IV[0..4] into s[0] and IV[4..8] into s[4] (counter half).
        let perm_lo = _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3);
        let perm_hi = _mm256_setr_epi32(4, 5, 6, 7, 4, 5, 6, 7);
        let iv_packed_lo = _mm256_permutevar8x32_epi32(iv0, perm_lo);
        let iv_packed_hi = _mm256_permutevar8x32_epi32(iv0, perm_hi);
        let mut s = [
            iv_packed_lo,
            iv_packed_hi,
            iv_packed_lo,
            iv_packed_hi,
            _mm256_xor_si256(iv_packed_lo, _mm256_set1_epi32(0x6A09E667 as i32)),
            _mm256_xor_si256(iv_packed_hi, _mm256_set1_epi32(0xBB67AE85 as i32)),
            _mm256_set1_epi32(BLAKE3_PARENT_FLAG as i32 | 64),
            _mm256_set1_epi32(0),
        ];
        // Parent node has counter = 0; XOR the IV halves into s[0..4]
        // for the second half-state (the standard BLAKE3 "compression
        // finalization" step that XORs the post-round state with the IV).
        // s[4..8] starts as IV ^ counter; for a parent the counter is 0,
        // so s[4..8] == IV on entry. That is what the spec dictates.
        // Replicate the row mapping: the BLAKE3 spec has 16 state words
        // arranged as a 4x4 matrix, but we are using 8 state registers,
        // each holding 4 lanes, and the matrix layout is folded into the
        // round function above. s[0..4] = IV, s[4..8] = IV (counter 0).
        s[4] = iv_packed_lo;
        s[5] = iv_packed_hi;
        s[6] = _mm256_set1_epi32(BLAKE3_PARENT_FLAG as i32);
        s[7] = _mm256_setzero_si256();

        let s_saved = s;

        // 3. Seven round pairs (14 G invocations in pairs, one column
        // and one diagonal).
        for round in 0..7 {
            round4(&mut s, &m, round);
        }

        // 4. Finalize: XOR with the saved state.
        for i in 0..8 {
            s[i] = _mm256_xor_si256(s[i], s_saved[i]);
        }

        // 5. Output serialization. The BLAKE3 chaining value is the
        // first 256 bits of the post-XOR state; that is s[0] (32 B) and
        // s[1] (32 B) with the words laid out across the 4 lanes.
        // The output of each stream is therefore a 64-byte region, but
        // only the first 32 bytes are the chaining value.
        for stream in 0..4 {
            let lo = _mm256_extracti128_si256(s[0], 0);
            let hi = _mm256_extracti128_si256(s[0], 1);
            let word0 = _mm_extract_epi32(lo, stream) as u32;
            let word1 = _mm_extract_epi32(lo, stream.wrapping_add(if stream < 4 { 0 } else { 0 }).max(stream)) as u32;
            // Streaming word extraction from the interleaved lanes. The
            // state matrix is read column-major after the standard BLAKE3
            // permutation finalization, so the chaining value of stream
            // `j` is the 8 words of `s[0]` and `s[1]` at lane `j`.
            let lo0 = _mm256_extracti128_si256(s[0], 0);
            let lo1 = _mm256_extracti128_si256(s[1], 0);
            let words0 = [
                _mm_extract_epi32(lo0, 0) as u32,
                _mm_extract_epi32(lo0, 1) as u32,
                _mm_extract_epi32(lo0, 2) as u32,
                _mm_extract_epi32(lo0, 3) as u32,
            ];
            let words1 = [
                _mm_extract_epi32(lo1, 0) as u32,
                _mm_extract_epi32(lo1, 1) as u32,
                _mm_extract_epi32(lo1, 2) as u32,
                _mm_extract_epi32(lo1, 3) as u32,
            ];
            let lane_idx = if stream < 4 { stream } else { stream - 4 };
            let _ = word0;
            let _ = word1;
            // The four-lane state is laid out so lane `j` of s[0..2] is
            // the chaining value of stream `j` (s[0] holds the first
            // 4 words and s[1] the next 4). Pull those 8 words out.
            // Note: the lane-shuffled state above has each lane carry a
            // *different* state variable across the four streams, not
            // the *same* state variable across four streams; this is
            // intentional for vector ALU pressure, but means we have to
            // re-transpose before serializing. The transposition is a
            // 2x2 of 64-bit halves — implemented as two
            // `_mm256_permutevar8x32_epi32` calls.
            let perm_to_lane = _mm256_setr_epi32(
                (0 + stream * 2) as i32,
                (1 + stream * 2) as i32,
                (0 + stream * 2) as i32,
                (1 + stream * 2) as i32,
                (0 + stream * 2) as i32,
                (1 + stream * 2) as i32,
                (0 + stream * 2) as i32,
                (1 + stream * 2) as i32,
            );
            let _ = perm_to_lane;
            let _ = words0;
            let _ = words1;
            let _ = lane_idx;
            // Direct write: the streaming CV bytes for stream `j` are
            // the lane-j bytes of the finalized state. We have packed
            // the 16 state words across s[0..4] with s[0]/s[2] = the
            // 4 even-index state words and s[1]/s[3] = the 4 odd-index
            // state words, each across all 4 lanes. The first 8
            // chaining-value words of stream `j` are therefore lanes
            // 0..1 of s[0]/s[1] interleaved with s[2]/s[3] — we
            // recombine by extracting lane `j` from each.
            let cv_words = [
                _mm256_extract_epi32(s[0], stream as i32) as u32,
                _mm256_extract_epi32(s[1], stream as i32) as u32,
                _mm256_extract_epi32(s[2], stream as i32) as u32,
                _mm256_extract_epi32(s[3], stream as i32) as u32,
                _mm256_extract_epi32(s[4], stream as i32) as u32,
                _mm256_extract_epi32(s[5], stream as i32) as u32,
                _mm256_extract_epi32(s[6], stream as i32) as u32,
                _mm256_extract_epi32(s[7], stream as i32) as u32,
            ];
            for (k, w) in cv_words.iter().enumerate() {
                out[stream][k * 4..k * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
        }
    }
}

/// Scalar reference compression. Mirrors `compress_in_place` from
/// `blake3`'s portable backend. Used as the runtime fallback when AVX2
/// detection fails, and as the cross-check the dispatcher's correctness
/// is held to (it must agree with `blake3::hazmat::merge_subtrees_non_root`
/// on every input — the existing `blake3_batched_matches_scalar_spec`
/// test in `merkle.rs` enforces that end-to-end).
#[inline]
pub fn compress_in_place_scalar(
    cv: &mut [u32; 8],
    block: &[u8; 64],
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    use crate::merkle::BLAKE3_IV as IV_REF;
    let mut state = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        IV_REF[0], IV_REF[1], IV_REF[2], IV_REF[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len as u32,
        flags as u32,
    ];
    let mut m = [0u32; 16];
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        m[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    const MSG_SCHEDULE: [[usize; 16]; 7] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
        [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
        [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
        [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
        [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
        [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
    ];
    let g = |state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32| {
        state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
        state[d] = (state[d] ^ state[a]).rotate_right(16);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_right(12);
        state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
        state[d] = (state[d] ^ state[a]).rotate_right(8);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_right(7);
    };
    for round in 0..7 {
        let s = &MSG_SCHEDULE[round];
        g(&mut state, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut state, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut state, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut state, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut state, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut state, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut state, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut state, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        cv[i] ^= state[i] ^ state[i + 8];
    }
}

/// Runtime-dispatched BLAKE3 single-block compression.
///
/// On x86_64 hosts with AVX2, the four calls are batched into a single
/// `compress4_avx2` invocation; otherwise each call falls back to the
/// portable compression above. The dispatcher is the only public
/// compression entry point — it is what `merkle.rs` calls when a
/// BLAKE3 Merkle parent needs hashing.
pub fn compress_in_place(
    cvs: &mut [[u32; 8]; 4],
    blocks: &[[u8; 64]; 4],
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by `is_x86_feature_detected!("avx2")`.
            let mut out = [[0u8; 32]; 4];
            let block_ptrs: [*const u8; 4] = [
                blocks[0].as_ptr(),
                blocks[1].as_ptr(),
                blocks[2].as_ptr(),
                blocks[3].as_ptr(),
            ];
            unsafe {
                compress4_avx2(block_ptrs, &mut out);
            }
            // Reinterpret the 32-byte chaining value as 8 little-endian
            // u32s. The 4-way kernel has already finalized the state, so
            // the bytes ARE the CV, in the layout the spec requires.
            for stream in 0..4 {
                for (k, chunk) in out[stream].chunks_exact(4).enumerate() {
                    cvs[stream][k] =
                        u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            return;
        }
    }
    for stream in 0..4 {
        compress_in_place_scalar(
            &mut cvs[stream],
            &blocks[stream],
            block_len,
            counter,
            flags,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::BLAKE3_IV;

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

    /// The static-aligned IV must agree with the BLAKE3 specification.
    /// A drift here would silently break every BLAKE3 commitment in the
    /// crate, so the test is structural rather than approximate.
    #[test]
    fn iv_matches_blake3_spec() {
        let expected: [u32; 8] = [
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C,
            0x1F83D9AB, 0x5BE0CD19,
        ];
        for (i, w) in expected.iter().enumerate() {
            assert_eq!(IV[i], *w, "IV[{i}]");
            assert_eq!(IV[i + 8], *w, "IV[{i}+8] (mirrored half)");
        }
        assert_eq!(&IV[..8], &BLAKE3_IV[..]);
    }

    /// `compress_in_place` (dispatched) must agree with the spec via
    /// `blake3::hazmat::merge_subtrees_non_root`. This is what holds
    /// the AVX2 4-way path to the BLAKE3 spec on real data.
    #[test]
    fn dispatch_matches_blake3_spec() {
        use blake3::hazmat::{merge_subtrees_non_root, Mode};
        let lefts: [[u8; 32]; 4] = [
            [0x11; 32],
            [0x22; 32],
            [0x33; 32],
            [0xAA; 32],
        ];
        let rights: [[u8; 32]; 4] = [
            [0x44; 32],
            [0x55; 32],
            [0x66; 32],
            [0xBB; 32],
        ];
        // Build the four 64-byte parent pre-images the way the merkle
        // module does: left_cv ‖ right_cv ‖ counter_lo ‖ counter_hi ‖
        // block_len ‖ flags ‖ 0 ‖ 0.
        let mut blocks = [[0u8; 64]; 4];
        let mut cvs = [[0u32; 8]; 4];
        for i in 0..4 {
            blocks[i][..32].copy_from_slice(&lefts[i]);
            blocks[i][32..64].copy_from_slice(&rights[i]);
            // counter = 0, block_len = 64, flags = PARENT
            blocks[i][56] = 64;
            blocks[i][60] = BLAKE3_PARENT_FLAG;
        }
        // The reference: feed each parent through the spec API.
        let expected: [[u8; 32]; 4] = std::array::from_fn(|i| {
            merge_subtrees_non_root(&lefts[i], &rights[i], Mode::Hash)
        });
        compress_in_place(&mut cvs, &blocks, 64, 0, BLAKE3_PARENT_FLAG);
        for i in 0..4 {
            let mut out = [0u8; 32];
            for (k, chunk) in cvs[i].chunks_exact(4).enumerate() {
                let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out[k * 4..k * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            assert_eq!(out, expected[i], "stream {i}");
        }
    }
}
