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
/// `r`; output byte `c·8 + t` bit `r` = that byte's bit `t`), so this delegates
/// to it — ~5× fewer ops than the scalar per-column loop. Shared by the
/// lincheck byte-stripe builder (`flock_prover::r1cs_hashes::common`) and the
/// core R1CS matrix-apply ([`crate::r1cs`]).
///
/// [`bit_transpose_64bytes`]: crate::zerocheck::univariate_skip_optimized::bit_transpose_64bytes
#[inline(always)]
pub fn transpose_8_u64s_to_64_bytes(lanes: &[u64; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 64);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("gfni")
        {
            unsafe { transpose_8_u64s_to_64_bytes_gfni(lanes, out) };
            return;
        }
    }
    transpose_8_u64s_to_64_bytes_scalar(lanes, out);
}

fn transpose_8_u64s_to_64_bytes_scalar(lanes: &[u64; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 64);
    for c in 0..8 {
        let shift = c * 8;
        let mut packed: u64 = 0;
        packed |= ((lanes[0] >> shift) & 0xFF) << 0;
        packed |= ((lanes[1] >> shift) & 0xFF) << 8;
        packed |= ((lanes[2] >> shift) & 0xFF) << 16;
        packed |= ((lanes[3] >> shift) & 0xFF) << 24;
        packed |= ((lanes[4] >> shift) & 0xFF) << 32;
        packed |= ((lanes[5] >> shift) & 0xFF) << 40;
        packed |= ((lanes[6] >> shift) & 0xFF) << 48;
        packed |= ((lanes[7] >> shift) & 0xFF) << 56;
        let transposed = transpose_8x8_bits(packed);
        out[c * 8..c * 8 + 8].copy_from_slice(&transposed.to_le_bytes());
    }
}

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

// ---------------------------------------------------------------------------
// BLAKE3 8-way interleaved8 transpose (batch = 8, stride = 64 bytes / block)
//
// BLAKE3's `hash_many` kernel sees its `DEGREE = 8` input blocks already in
// "interleaved8" form: lane `i` of every 256-bit SIMD register holds byte `i`
// of the eight `BLOCK_LEN = 64` message blocks. Building that transposed
// layout on the fly is the "per-block bit-reversal/byte-copy step" the AVX2
// batched compress relies on, and the BLAKE3 reference AVX2 implementation
// builds it from 8 pointers via `_mm256_unpacklo/hi_epi32/64` and
// `_mm256_permute2x128_si256` (see `transpose_vecs` / `transpose_msg_vecs`).
//
// For callers that already own the 8 input blocks as 8 contiguous 64-byte
// slices and want to skip the per-block SIMD re-load, this module exposes
// a 64-byte-aligned scratch buffer `Interleaved8Block` of length
// `64 * batch` and a function `interleaved8_block_into` that emits the
// transposed layout directly. The output byte at
//   `out[byte * batch + lane] = block_lane[byte]`
// is exactly the row-major BLAKE3 hash_many input the reference kernel
// consumes after its `transpose_vecs` step, so callers can use the buffer
// either as the source of a `_mm256_loadu_si256` per row or as the
// pre-transposed storage for a streaming kernel that prefers single loads.
// ---------------------------------------------------------------------------

/// `64 * 8 = 512` byte cache-line-aligned scratch holding the eight 64-byte
/// message blocks in BLAKE3's "interleaved8" transposed layout (lane `i`
/// of every 32-byte half = byte `i` of block `i`). The batch count is
/// fixed at 8 — the AVX2 kernel's `DEGREE` — to keep the buffer exactly
/// 512 bytes, two cache lines per side, which is the rank's hot L1
/// working set per batch. A const generic over `BATCH` would force a
/// per-instantiation type that the BLAKE3 reference kernel does not need
/// (its AVX2 SIMD width is a single hard-coded 8), so the helper is
/// specialised on the ranked shape here.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Interleaved8Block([u8; INTERLEAVED8_BYTES]);

/// Total bytes in an [`Interleaved8Block`]: `64 * 8 = 512`.
pub const INTERLEAVED8_BYTES: usize = 64 * 8;
/// Number of blocks in one batch: BLAKE3's AVX2 `DEGREE`.
pub const INTERLEAVED8_BATCH: usize = 8;

impl Interleaved8Block {
    /// Construct an uninitialized scratch. Caller must overwrite every byte
    /// before reading any.
    #[inline(always)]
    pub fn uninit() -> Self {
        // The default-initialised `[u8; N]` is a zeroed buffer; the
        // kernel overwrites every byte before any read, so the zero
        // start is sound and skips a memset on the hot path.
        Interleaved8Block([0u8; INTERLEAVED8_BYTES])
    }

    /// Borrow as a `&[u8]` of length `INTERLEAVED8_BYTES`.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Borrow as a `&mut [u8]` of length `INTERLEAVED8_BYTES`.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }

    /// Raw pointer to the first byte of the transposed buffer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }

    /// Raw mutable pointer to the first byte of the transposed buffer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }
}

/// Emit the eight `BLOCK_LEN`-byte blocks of `input` into `out` in BLAKE3's
/// interleaved8 transposed layout: `out[byte * 8 + lane] = input[lane][byte]`.
///
/// This is the per-block "bit-reversal/byte-copy step" that precedes the AVX2
/// 8-way `hash_many` compress — the same shuffle the reference kernel
/// performs via three layers of 32/64/128-bit unpacks, but lifted to a single
/// cold pass that can be prefetched and overlapped with the next batch's
/// transposes. The destination is `64 * 8 = 512` bytes — [`Interleaved8Block`]
/// — and the layout matches the BLAKE3 reference `hash8` input row-for-row
/// after its internal `transpose_vecs`.
///
/// # Layout
/// ```text
/// for byte in 0..64:
///     for lane in 0..8:
///         out[byte * 8 + lane] = input[lane][byte]
/// ```
/// Equivalently, the 32-byte halves of each 256-bit YMM register hold
/// `[byte0_l0, byte0_l1, ..., byte0_l7, byte1_l0, byte1_l1, ..., byte1_l7]`
/// — the exact shape the AVX2 kernel reads as `_mm256_loadu_si256`.
///
/// # Non-AVX2 path
/// Always available. The AVX2 batched compress falls back to a scalar
/// `compress_in_place` loop when `is_x86_feature_detected!("avx2")` is false;
/// in that case the transposed buffer is still correct (the scalar fallback
/// re-reads the source blocks), so this helper is unconditionally compiled.
#[inline(always)]
pub fn interleaved8_block_into(
    out: &mut Interleaved8Block,
    input: &[&[u8; 64]; 8],
) {
    debug_assert_eq!(INTERLEAVED8_BATCH, 8);
    let dst = out.as_bytes_mut();
    for byte in 0..64 {
        let row = byte * 8;
        for lane in 0..8 {
            // SAFETY: the source block is `[u8; 64]` and `row + lane < 512`
            // by construction (byte < 64 and lane < 8).
            unsafe {
                let p = dst.as_mut_ptr().add(row + lane);
                *p = *input[lane].as_ptr().add(byte);
            }
        }
    }
}

/// `Interleaved8Block` filled in with the BLAKE3 hash_many "block" half of
/// the inputs (8 message blocks, 8 chaining values, 8 counter pairs, 8
/// `block_len`s, 8 `flags`s). The `HashManyInputs` is the natural call shape
/// for the batched compress: one struct carries everything the AVX2 kernel
/// needs and the bytes live in their final positions in the scratch buffer
/// before the first G fires.
#[derive(Clone, Copy)]
pub struct HashMany8Inputs<'a> {
    /// 8 input `BLOCK_LEN = 64` message blocks in caller order.
    pub blocks: [&'a [u8; 64]; 8],
    /// 8 input chaining values (one per block).
    pub chaining_values: [&'a [u32; 8]; 8],
    /// 8 input 64-bit counters (low half interpreted as a `u32`; the high
    /// half is read separately and passed in `counter_hi` for the same lane).
    pub counter: [u64; 8],
    /// 8 input `block_len`s. Ranked shape fixes all 8 to `64`; the array
    /// stays generic to keep the kernel reusable for variable-length chains.
    pub block_len: [u32; 8],
    /// 8 input `flags` u32s. The reference kernel ORs `flags_start` /
    /// `flags_end` per block, so this array carries the per-block result.
    pub flags: [u32; 8],
}

impl<'a> HashMany8Inputs<'a> {
    /// Build the inputs from 8 sets of (block, cv, counter, block_len, flags).
    pub fn new(
        blocks: [&'a [u8; 64]; 8],
        chaining_values: [&'a [u32; 8]; 8],
        counter: [u64; 8],
        block_len: [u32; 8],
        flags: [u32; 8],
    ) -> Self {
        Self {
            blocks,
            chaining_values,
            counter,
            block_len,
            flags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The NEON-delegating transpose must match the scalar per-column oracle
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
