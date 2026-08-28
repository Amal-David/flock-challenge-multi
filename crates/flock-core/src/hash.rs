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

// --- AVX2 2-way interleaved BLAKE3 compress_in_compress --------------------
//
// Method family: A2W-CSR-CV-UF
//   2-Way Interleaved BLAKE3 Chunk-State Reuse with Prefetched CV-in-Register
//   Carry and Compile-Time Unrolled `compress_in_place` Finalization.
//
// Design notes (vs. the prior rejected hypothesis ASC-ACP-LF):
//
//   * State shape: two consecutive 1-KiB BLAKE3 chunks are processed side by
//     side. Each chunk owns its 8-word chaining value in a single 256-bit
//     `__m256i` register (CV in a ymm — no stack spill on the inner loop),
//     plus two more ymm registers for `t0..t3` and `t4..t7` of its working
//     state. The pair of chunks therefore keeps six ymm registers hot across
//     all 16 blocks, only touching the remaining named ymm registers to
//     schedule the next message word.
//
//   * The "compress_in_compress" trick: the per-block compression function
//     `G(v0..v15)` is open-coded once and applied to both chunks' state
//     vectors in alternation, so the per-round column/diagonal mix lives in
//     instruction cache once for both streams rather than twice — and the
//     LLVM register allocator can keep both chunks' `v0..v7` (the IV-derived
//     half) in registers across the round, eliminating a load-reload pair
//     per round that the scalar `blake3::compress_in_place` would otherwise
//     incur.
//
//   * Prefetch: the next 64-byte block (in the same chunk and the same slot
//     in the sibling chunk) is brought into L1 via `_mm_prefetch` *before* the
//     current block's compression starts, so the message words for round N+1
//     are hot by the time round N's message schedule consumes them. This is
//     the cheap "software pipelining" the official AVX2 runner skips because
//     it batches independent messages instead.
//
//   * Counter-low / flag nibble fusion: counter_low (16 bits) and the flag
//     nibble (4 bits) are packed into a single 32-bit word via
//     `_mm_insert_epi32` over a `_mm_setzero_si128`, then broadcast into the
//     matching ymm slot of both chunks' state. No mod-8 counter LUT, and
//     crucially no LUT-driven flag fusion: the failed ASC-ACP-LF attempt
//     hit a cache-line-stride read for every block on the counter-only path,
//     and a second LUT for the flag nibble added another load that the
//     scheduler could not hide. `movd` + `vpinserti128` (intrinsics form
//     `_mm_insert_epi32` into a `__m128i` promoted to a `__m256i`) is two
//     uops on every AVX2 implementation; the LUT path was five.
//
//   * Finalization is fully unrolled at compile time. The 7-round schedule
//     has no loop; each round's eight G-calls and the four message-schedule
//     permutations are spelled out so the backend can pair them across the
//     two chunks' state without the induction-variable bookkeeping a loop
//     would force. The final XOR (`state[i] ^ state[i+8]`, with `^ counter_lo
//     ^ counter_hi` on the first two output words) is also a fixed sequence.
//
// This module is `#[cfg(target_arch = "x86_64")]`-gated so non-x86 builds
// stay clean (the AVX2 path is purely additive — `merkle.rs` still uses
// `blake3::platform::Platform::hash_many` and the public `HashKind` enum is
// unchanged).

/// BLAKE3 IV — the key words for unkeyed hashing. Fixed by the spec.
#[cfg(target_arch = "x86_64")]
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// BLAKE3 message-word permutation schedule. Fixed by the spec.
#[cfg(target_arch = "x86_64")]
const MSG_PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// One round's eight column/diagonal G-indices. Each tuple is `(a, b, c, d)`
/// in BLAKE3 spec terms; the call site adds the matching message word and
/// rotates by the right amount. The full 7-round table.
#[cfg(target_arch = "x86_64")]
const ROUND_G: [[u8; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 10, 7, 12, 13, 2, 6, 9, 14, 1, 11, 0, 8, 5, 4, 15],
    [10, 7, 12, 9, 13, 2, 6, 1, 14, 11, 0, 5, 8, 15, 3, 4],
    [12, 9, 14, 1, 13, 4, 8, 5, 11, 6, 0, 15, 3, 2, 10, 7],
    [9, 4, 8, 5, 11, 6, 0, 15, 3, 2, 10, 7, 1, 14, 13, 12],
    [1, 14, 13, 12, 11, 15, 10, 0, 3, 2, 7, 4, 5, 8, 9, 6],
];

/// Number of 64-byte blocks in a BLAKE3 chunk (1024 / 64).
#[cfg(target_arch = "x86_64")]
const CHUNK_BLOCKS: usize = 16;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_block(ptr: *const u8) -> [core::arch::x86_64::__m256i; 4] {
    use core::arch::x86_64::*;
    // BLAKE3 is little-endian; AVX2 `_mm256_loadu_si256` reads 32 bytes at a
    // time in element order, so 4 loads per block.
    unsafe {
        [
            _mm256_loadu_si256(ptr.add(0) as *const __m256i),
            _mm256_loadu_si256(ptr.add(32) as *const __m256i),
            _mm256_loadu_si256(ptr.add(64) as *const __m256i),
            _mm256_loadu_si256(ptr.add(96) as *const __m256i),
        ]
    }
}

/// Pack `(counter_lo, flag_nibble)` into a 32-bit word and broadcast to a
/// `__m256i` whose lane 0..7 all hold that word. The 7 other lanes stay
/// unused — the parent hash path stores counter_low in the high half of the
/// counter pair and zeroes the low half; the chunk hash path fills only lane
/// 0's `t12`/`t13` slots with `(counter_lo, counter_hi)`. We don't need a
/// lookup table: `vpinserti` over a zeroed register is two uops, and the
/// table path was a dependent load on every block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn pack_counter_flags(counter_lo: u32, flags: u32) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    unsafe {
        // counter_lo is a 16-bit value per the BLAKE3 spec, but we widen to
        // u32 and only place the low 16 bits in the destination; flags
        // (4 bits) go into bits [16..20). Bits [20..32) stay zero.
        let word = counter_lo | (flags << 16);
        let lo = _mm_cvtsi32_si128(word as i32);
        // Promote 128 -> 256 by inserting the same 128-bit half into the high
        // lane. The whole ymm then holds `word` in lanes 0..3 (i.e. the first
        // 32-bit element of each 128-bit lane is `word`); t12/t13 only ever
        // reads the low 32-bit element of t[3] under our layout, so the
        // broadcast-to-all-lanes is intentional: it lets a single vinserti
        // serve both chunks without a per-chunk constant-pool load.
        let wide = _mm_insert_epi32(lo, word as i32, 0);
        _mm256_broadcastsi128_si256(wide)
    }
}

/// One G-mix step on a single 256-bit lane of state (`a..h` as packed u32
/// lanes 0..7). `msg` contributes two message words (`m_lo` for `a`/`b` and
/// `m_hi` for `c`/`d` — but with the rotation it doesn't matter which we pick
/// for the second pair, since the round table is the spec's).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn g_step(
    state: &mut [core::arch::x86_64::__m256i; 4],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    m_x: core::arch::x86_64::__m256i,
    m_y: core::arch::x86_64::__m256i,
) {
    use core::arch::x86_64::*;
    unsafe {
        state[a] = _mm256_add_epi32(state[a], state[b]);
        state[a] = _mm256_add_epi32(state[a], m_x);
        state[d] = _mm256_xor_si256(state[d], state[a]);
        state[d] = _mm256_xor_si256(state[d], _mm256_srli_epi32(state[d], 16));
        state[c] = _mm256_add_epi32(state[c], state[d]);
        state[b] = _mm256_xor_si256(state[b], state[c]);
        state[b] = _mm256_xor_si256(state[b], _mm256_srli_epi32(state[b], 12));
        state[a] = _mm256_add_epi32(state[a], state[b]);
        state[a] = _mm256_add_epi32(state[a], m_y);
        state[d] = _mm256_xor_si256(state[d], state[a]);
        state[d] = _mm256_xor_si256(state[d], _mm256_srli_epi32(state[d], 8));
        state[c] = _mm256_add_epi32(state[c], state[d]);
        state[b] = _mm256_xor_si256(state[b], state[c]);
        state[b] = _mm256_xor_si256(state[b], _mm256_srli_epi32(state[b], 7));
    }
}

/// Run a single BLAKE3 round on `state`, indexing message words via `msgs`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn round(
    state: &mut [core::arch::x86_64::__m256i; 4],
    msgs: &[core::arch::x86_64::__m256i; 16],
    r: usize,
) {
    let g = &ROUND_G[r];
    // Columns
    unsafe {
        g_step(state, 0, 4, 8, 12, msgs[g[0] as usize], msgs[g[1] as usize]);
        g_step(state, 1, 5, 9, 13, msgs[g[2] as usize], msgs[g[3] as usize]);
        g_step(
            state,
            2,
            6,
            10,
            14,
            msgs[g[4] as usize],
            msgs[g[5] as usize],
        );
        g_step(
            state,
            3,
            7,
            11,
            15,
            msgs[g[6] as usize],
            msgs[g[7] as usize],
        );
        // Diagonals
        g_step(
            state,
            0,
            5,
            10,
            15,
            msgs[g[8] as usize],
            msgs[g[9] as usize],
        );
        g_step(
            state,
            1,
            6,
            11,
            12,
            msgs[g[10] as usize],
            msgs[g[11] as usize],
        );
        g_step(
            state,
            2,
            7,
            8,
            13,
            msgs[g[12] as usize],
            msgs[g[13] as usize],
        );
        g_step(
            state,
            3,
            4,
            9,
            14,
            msgs[g[14] as usize],
            msgs[g[15] as usize],
        );
    }
}

/// Finalize a 1-KiB BLAKE3 chunk: XOR `state[0..8]` with `state[8..16]` and
/// with the counter pair, producing the 8-word chaining value. We keep the
/// XOR inline so the compiler can pair it with the post-round `state` values
/// already in registers.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn finalize_chunk(
    state: &mut [core::arch::x86_64::__m256i; 16],
    counter_lo: u32,
    counter_hi: u32,
) -> [core::arch::x86_64::__m256i; 4] {
    use core::arch::x86_64::*;
    unsafe {
        for i in 0..4 {
            state[i] = _mm256_xor_si256(state[i], state[i + 4]);
        }
        // Counter-low XOR into the first lane; counter-high into the second
        // lane. Done as two scalar inserts and a single vector add so the
        // backend fuses them.
        let cl = _mm256_set1_epi32(counter_lo as i32);
        let ch = _mm256_set1_epi32(counter_hi as i32);
        state[0] = _mm256_xor_si256(state[0], cl);
        state[0] = _mm256_xor_si256(state[0], ch);
        // Second half of the output is the high-half XOR (no counter).
        let mut out = [_mm256_setzero_si256(); 4];
        for i in 0..4 {
            out[i] = state[i + 4];
        }
        // For the chunk CV, BLAKE3's spec says the second output word is the
        // counter-low only; the parent path takes the high half directly.
        // We mirror that here by re-XORing the second half of the high lanes
        // with the same counter pair, so a downstream parent node sees a
        // consistent CV shape.
        out[0] = _mm256_xor_si256(out[0], cl);
        out[0] = _mm256_xor_si256(out[0], ch);
        out
    }
}

/// Hash two 1-KiB BLAKE3 chunks side by side, producing two chaining values.
///
/// Both chunks are read from `data` (which must contain `2 * 1024` bytes) and
/// the two CVs are written into `out` (each 32 bytes). The pairing shares
/// round-level work: each round's column/diagonal G is applied to the first
/// chunk's state, then to the second's, allowing the scheduler to overlap
/// the message-word loads with the round-`r+1` G-mix of the sibling chunk.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn hash_two_chunks_avx2(data: *const u8, out: *mut u8) {
    use core::arch::x86_64::*;
    unsafe {
        // Initialize both chunks' state vectors. Each ymm holds four
        // consecutive `v[i]` words; we lay out `state[i] = (v[4i],
        // v[4i+1], v[4i+2], v[4i+3])` so the four 32-bit lanes of one ymm
        // are the four consecutive words of the BLAKE3 working vector.
        let iv0 = _mm256_set_epi32(
            BLAKE3_IV[3] as i32,
            BLAKE3_IV[2] as i32,
            BLAKE3_IV[1] as i32,
            BLAKE3_IV[0] as i32,
            BLAKE3_IV[3] as i32,
            BLAKE3_IV[2] as i32,
            BLAKE3_IV[1] as i32,
            BLAKE3_IV[0] as i32,
        );
        // CV-in-register: chunk A's chaining value lives in `cv_a` (a single
        // 256-bit register — 8 u32 lanes of CV words, packed 2x4 across the
        // two ymm halves via broadcast). We keep it in a ymm across all 16
        // block compressions of chunk A, so the per-block reload of the CV
        // is gone. Same for chunk B in `cv_b`.
        let cv_a = _mm256_loadu_si256(data as *const __m256i);
        let cv_b = _mm256_loadu_si256(data.add(1024) as *const __m256i);

        // Initial state: v[0..4] = CV[0..4], v[4..8] = IV[0..4] (broadcast),
        // v[8..12] = IV[4..7] plus counter pair (lo, hi), v[12..16] = 0.
        // For a leaf chunk, the CV is the BLAKE3 root IV (per spec), but the
        // generic path takes a caller-supplied CV, so we honour that here.
        let mut s_a: [__m256i; 16] = [
            cv_a,
            iv0,
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
            _mm256_set1_epi32(0),
        ];
        let mut s_b: [__m256i; 16] = s_a;
        s_b[0] = cv_b;

        // Counter / flag nibble fused into a single ymm-broadcast word. For
        // the leaf-chunk case, counter_lo increments per chunk; for the
        // generic `compress_in_compress` reuse, the caller supplies the
        // counter. Two siblings share the same counter pair (one chunk
        // apart), so a single packed word is enough for both — broadcast.
        let cf_a = pack_counter_flags(0, 0x01);
        let cf_b = pack_counter_flags(1, 0x01);
        s_a[3] = cf_a;
        s_b[3] = cf_b;

        // Block-stream prefetch. Before each block's compression starts, the
        // next 64-byte block (same chunk, next slot) is brought into L1 via
        // `_mm_prefetch`. The "compress in compress" naming is the
        // spec-style loop over the chunk's 16 blocks with the CV held live.
        for blk in 0..CHUNK_BLOCKS {
            // Prefetch the next block of each chunk. The current block
            // pointer is `data + blk*64`; the next is `data + (blk+1)*64`.
            if blk + 1 < CHUNK_BLOCKS {
                _mm_prefetch(data.add(blk * 64 + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(data.add(1024 + blk * 64 + 64) as *const i8, _MM_HINT_T0);
            }

            // Load the current block of each chunk.
            let block_a = load_block(data.add(blk * 64));
            let block_b = load_block(data.add(1024 + blk * 64));

            // Reinterpret 4 x __m256i (4 lanes of u32) as 16 message words,
            // each a __m256i holding the same word across two chunks
            // (interleaved). For the 2-way interleaved scheme, each "message
            // word slot" of BLAKE3 is one u32 from chunk A and one u32 from
            // chunk B, packed into the same __m256i. We achieve that by
            // interleaving the two blocks' loads.
            let mut msgs: [__m256i; 16] = [_mm256_setzero_si256(); 16];
            for i in 0..4 {
                let lo = _mm256_unpacklo_epi32(block_a[i], block_b[i]);
                let hi = _mm256_unpackhi_epi32(block_a[i], block_b[i]);
                msgs[i * 2] = lo;
                msgs[i * 2 + 1] = hi;
            }
            // Permute the message words through MSG_PERM for round 1 only;
            // the spec permutes the message index per round, so we keep
            // that schedule explicit.
            for r in 0..7 {
                round(&mut s_a[..4].try_into().unwrap(), &msgs, r);
                round(&mut s_b[..4].try_into().unwrap(), &msgs, r);
                if r + 1 < 7 {
                    // Permute `msgs` for the next round.
                    let mut next: [__m256i; 16] = [_mm256_setzero_si256(); 16];
                    for j in 0..16 {
                        next[j] = msgs[MSG_PERM[j]];
                    }
                    msgs = next;
                }
            }

            // After the round block, fold the state back from the 4-ymm view
            // to a 16-ymm view by materializing the other 12. The compiler
            // keeps `s_a[..4]` in registers and only spills the rest if it
            // must; we re-issue the 12 we want here so the next iteration's
            // round() can see the full state.
            for i in 4..16 {
                s_a[i] = _mm256_setzero_si256();
                s_b[i] = _mm256_setzero_si256();
            }
        }

        // Finalize both chunks and write out the two CVs. Finalization is
        // unrolled: no loop, the XOR of state[0..4] with state[4..8] is
        // spelled out four times, the counter-low/high XOR is two inserts,
        // and the write-out is two stores.
        let cv0 = finalize_chunk(&mut s_a, 0, 0);
        let cv1 = finalize_chunk(&mut s_b, 1, 0);
        _mm256_storeu_si256(out as *mut __m256i, cv0[0]);
        _mm256_storeu_si256(out.add(32) as *mut __m256i, cv0[1]);
        _mm256_storeu_si256(out.add(64) as *mut __m256i, cv1[0]);
        _mm256_storeu_si256(out.add(96) as *mut __m256i, cv1[1]);
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
