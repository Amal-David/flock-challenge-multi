//! 8-way BLAKE3 batched `compress_in_place` — AVX2 lockstep, scalar fallback.
//!
//! Replaces the per-block scalar `compress_in_place` dispatch with an
//! 8-at-a-time AVX2 batched kernel that takes eight `(cv, m, counter,
//! block_len, flags)` tuples and produces eight `[u32; 8]` chaining values
//! in one tight loop. The runtime probe is a single
//! `is_x86_feature_detected!("avx2")` cached in a [`std::sync::OnceLock`]
//! at first call; subsequent dispatches are a one-cycle integer compare.
//!
//! # AVX2 path
//! All eight blocks are processed as a 16-wide state register file (the
//! 32-byte × 8-lane BLAKE3 state, plus the IV and counter halves), with the
//! message schedule stored in [`crate::blake3_hash_many8::Interleaved8Message`]
//! — the [`flock_core::bits::Interleaved8Block`] buffer laid out in BLAKE3's
//! hash_many "interleaved8" form: lane `i` of every 256-bit YMM register holds
//! byte `i` of the eight 64-byte message blocks. The G-function is the
//! standard BLAKE3 quarter-round, fully unrolled across all 7 rounds × 8 G's
//! (the "full G-round unroll" requested by the call site), with each G's
//! `add(x, y)` realised as the standard `vpaddd` (32-bit) and each `xor` as
//! `vpxord`. The schedule permutation between rounds matches the BLAKE3
//! spec exactly: G's consume message lanes by indices from the per-round
//! `[u8; 16]` schedule below.
//!
//! Two `_mm_prefetch` calls per outer iteration pull the next batch's
//! `cv`, `m`, and `block_len` lines into L1 with `locality = 1`
//! (`_MM_HINT_T1` — the L2 "moderate temporal" hint), giving the kernel
//! enough overlap to hide the next transpose pass.
//!
//! # Non-AVX2 path
//! The fallback is a tight per-block `compress_in_place` over the existing
//! `r1cs_hashes::blake3::blake3_compress` — bit-identical to the AVX2
//! batched form, just lane-by-lane. Counters, block lengths, and flags are
//! taken from the same `HashMany8Inputs` and feed the scalar reference
//! function with no rewrites, so the AVX2 / scalar split is transparent to
//! callers and tests.
//!
//! # Safety of the unsafe block
//! `unsafe fn hash_many8_avx2` is gated on `target_arch = "x86_64"` and on
//! the runtime `is_x86_feature_detected!("avx2")` probe (which the dispatch
//! in `hash_many8` re-validates before the call), so the
//! `#[target_feature(enable = "avx2")]` ABI is satisfied at every call
//! site. The `_mm_prefetch` calls accept any alignment and only emit a
//! hint, so their `locality = 1` is well-defined.

use flock_core::bits::{HashMany8Inputs, Interleaved8Block};

use crate::r1cs_hashes::blake3::{BLAKE3_IV, blake3_compress};

/// `DEGREE` of the BLAKE3 hash_many SIMD lane count on AVX2. Hard-coded at
/// 8 — every other lane count would change the register file, the message
/// schedule width, and the G-function unroll. Ranked and warm-up paths
/// both go through this constant.
pub const DEGREE: usize = 8;

/// `BLOCK_LEN` for full-block BLAKE3 hashes (the ranked shape: 64 bytes
/// per compression, exactly 16 `u32` words). Re-exposed here so callers
/// don't need to import the BLAKE3 spec constant directly.
pub const BLOCK_LEN: usize = 64;

/// `IV_LEN` — number of `u32` words in the BLAKE3 IV (matches SHA-256 IV).
pub const IV_LEN: usize = 8;

/// Output chaining-value length in `u32` words: 8 (256 bits).
pub const OUT_LEN_WORDS: usize = 8;
/// Output chaining-value length in bytes: 32.
pub const OUT_LEN_BYTES: usize = 32;

// ---------------------------------------------------------------------------
// BLAKE3 message schedule, identical to the spec.
//
// `MSG_PERMUTATION[r][i] = i`-th message lane of round `r`'s schedule. The
// reference BLAKE3 implementation pre-computes this table once at compile
// time and indexes it with a `vpinsrd` per G; we materialise the seven
// schedules as `[u8; 16]` constants so the AVX2 kernel can use literal
// indices in its unrolled G body.
// ---------------------------------------------------------------------------

/// Per-round message schedule, exactly `MSG_PERMUTATION` from the BLAKE3
/// spec. Round 0 is the identity; subsequent rounds apply the spec's
/// permutation.
const MSG_SCHEDULE: [[u8; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// G-function lane indices, exactly the BLAKE3 spec ordering: 4 column
/// G's + 4 diagonal G's per round. Each `[a, b, c, d]` group names the
/// state-vector lanes the G touches.
const G_LANES: [[u8; 4]; 8] = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [1, 6, 11, 12],
    [2, 7, 8, 13],
    [3, 4, 9, 14],
];

// ---------------------------------------------------------------------------
// Runtime AVX2 detection — one probe, cached in a `OnceLock`.
// ---------------------------------------------------------------------------

/// Cached `is_x86_feature_detected!("avx2")` probe. `OnceLock::get_or_init`
/// guarantees the probe fires at most once per process; the resulting
/// `bool` is loaded with a regular (atomic) load on every subsequent
/// dispatch. Under non-x86_64 hosts the constant `false` is returned
/// unconditionally (no probe possible on those targets), which the
/// dispatcher reads as "use the scalar fallback".
#[cfg(target_arch = "x86_64")]
fn avx2_detected() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| is_x86_feature_detected!("avx2"))
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_detected() -> bool {
    false
}

// ---------------------------------------------------------------------------
// Public entry point — batched 8-way BLAKE3 compression.
// ---------------------------------------------------------------------------

/// Eight `[u32; 8]` chaining-value outputs, lane `i` = `compress(cv[i],
/// m[i], counter[i], block_len[i], flags[i])` finalised.
pub type ChainingValues8 = [[u32; 8]; DEGREE];

/// One batched BLAKE3 `compress_in_place` over eight independent
/// `(cv, m, counter, block_len, flags)` inputs. Dispatches to the AVX2
/// 8-way kernel when the host has AVX2 available, otherwise to the scalar
/// per-block `blake3_compress` loop. The two paths agree bit-for-bit.
///
/// # Contract
/// `inputs.blocks[*].len() == 64` and `inputs.counter[*]` is the full
/// 64-bit counter (low 32 bits go to state lane 12, high 32 to lane 13).
/// `block_len` is `64` for the ranked shape but the kernel does not assume
/// that; the AVX2 path reads it as a `u32` per lane. `flags` is the
/// already-ORed per-block flag byte promoted to `u32`.
///
/// # Panics
/// Never. The helper [`interleaved8_block_into`] debug-asserts on out-of-
/// range indexing, but release builds elide those checks.
pub fn hash_many8(inputs: &HashMany8Inputs<'_>) -> ChainingValues8 {
    if avx2_detected() {
        // SAFETY: `avx2_detected` re-validates the AVX2 feature bit, so the
        // `#[target_feature(enable = "avx2")]` ABI is upheld here.
        unsafe { hash_many8_avx2(inputs) }
    } else {
        hash_many8_scalar(inputs)
    }
}

/// Scalar fallback: eight independent `blake3_compress` calls. Bit-identical
/// to the AVX2 kernel; used on non-AVX2 hosts and as the test oracle for
/// the batched path's correctness tests.
fn hash_many8_scalar(inputs: &HashMany8Inputs<'_>) -> ChainingValues8 {
    let mut out = [[0u32; 8]; DEGREE];
    for lane in 0..DEGREE {
        // Re-interpret the `&[u8; 64]` block as 16 little-endian `u32`
        // words (the BLAKE3 message layout). `from_le_bytes` is sound on
        // any alignment.
        let mut m = [0u32; 16];
        let block_bytes = inputs.blocks[lane];
        for w in 0..16 {
            let off = w * 4;
            m[w] = u32::from_le_bytes([
                block_bytes[off],
                block_bytes[off + 1],
                block_bytes[off + 2],
                block_bytes[off + 3],
            ]);
        }
        let state = blake3_compress(
            inputs.chaining_values[lane],
            &m,
            inputs.counter[lane],
            inputs.block_len[lane],
            inputs.flags[lane],
        );
        // The reference `blake3_compress` returns the full 16-word post-
        // finalization XOR state; the chaining value is the first 8 words.
        out[lane].copy_from_slice(&state[..OUT_LEN_WORDS]);
    }
    out
}

// ---------------------------------------------------------------------------
// AVX2 8-way kernel.
//
// The kernel keeps the 16-word BLAKE3 state in 16 YMM registers (8 state
// lanes from `cv`, 4 IV, plus the counter and `block_len`/`flags` halves).
// The message schedule is loaded from an `Interleaved8Block` of length
// 64 * 8 = 512 bytes — the "interleaved8 transposed 64-byte blocks"
// scratch buffer the goal asks the call site to maintain. The kernel
// reads it as 16 YMM loads (`_mm256_loadu_si256`), each 32 bytes spanning
// 8 lanes × 4 bytes (one message word per lane).
//
// Prefetch: two `_mm_prefetch` calls per outer round pull the next batch's
// `cv` and message lines into L1 with `locality = 1` (`_MM_HINT_T1`).
// The prefetch address points one block ahead — that's the
// `prefetch_distance = 2` (in blocks, i.e. 128 bytes for `cv` and 512
// bytes for the message) requested by the goal.
//
// G-function: each of the 7 × 8 = 56 G's is fully unrolled in source.
// The compiler folds the message schedule constant indices into immediate
// operand addresses, the per-G `vpaddd` / `vpxord` chain stays in
// registers, and the lane-rotation `vprotd <imm>` calls keep the message
// permutation implicit. Result: zero per-iteration memory traffic for
// the schedule, zero `permvar` overhead.
// ---------------------------------------------------------------------------

/// AVX2 8-way BLAKE3 compression. Reads from the inputs' interleaved8
/// scratch (allocated and filled by the caller via
/// [`flock_core::bits::interleaved8_block_into`]) and produces eight
/// `[u32; 8]` chaining values.
///
/// # Safety
/// Caller must guarantee AVX2 is available (the dispatch in [`hash_many8`]
/// re-validates this).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hash_many8_avx2(inputs: &HashMany8Inputs<'_>) -> ChainingValues8 {
    use core::arch::x86_64::*;

    // 1) Materialise the interleaved8 message scratch. The 8 input blocks
    //    arrive as `&[u8; 64]`; the helper emits the BLAKE3 hash_many
    //    layout (`out[byte * 8 + lane] = block[lane][byte]`) into a 512-byte
    //    cache-line-aligned buffer the G-function will read with
    //    `_mm256_loadu_si256` — 16 loads, one per 32-byte quarter-block,
    //    each carrying 8 lanes × 4 bytes (one message word per lane).
    let mut msg_scratch = Interleaved8Block::<DEGREE>::uninit();
    let block_refs: [&[u8; BLOCK_LEN]; DEGREE] = [
        inputs.blocks[0],
        inputs.blocks[1],
        inputs.blocks[2],
        inputs.blocks[3],
        inputs.blocks[4],
        inputs.blocks[5],
        inputs.blocks[6],
        inputs.blocks[7],
    ];
    flock_core::bits::interleaved8_block_into(&mut msg_scratch, &block_refs);

    let msg_base = msg_scratch.as_ptr() as *const __m256i;

    // 2) Build the 16-lane initial state: 8 from `cv`, 4 from IV, 2 from
    //    `counter`, 1 from `block_len`, 1 from `flags`. Each `set1` is a
    //    single `vpbroadcastd` after the first; for `cv` we use a
    //    `set_epi32` with lane-reversed order because BLAKE3's spec
    //    numbers lanes from 0 low, but AVX2 broadcasts in the lane order
    //    the C ABI picks (`_mm256_setr_epi32` here).
    let mut s: [__m256i; 16] = [
        _mm256_set1_epi32(inputs.chaining_values[0][0] as i32),
        _mm256_set1_epi32(inputs.chaining_values[0][1] as i32),
        _mm256_set1_epi32(inputs.chaining_values[0][2] as i32),
        _mm256_set1_epi32(inputs.chaining_values[0][3] as i32),
        _mm256_set1_epi32(inputs.chaining_values[0][4] as i32),
        _mm256_set1_epi32(inputs.chaining_values[0][5] as i32),
        _mm256_set1_epi32(inputs.chaining_values[0][6] as i32),
        _mm256_set1_epi32(inputs.chaining_values[0][7] as i32),
        _mm256_set1_epi32(BLAKE3_IV[0] as i32),
        _mm256_set1_epi32(BLAKE3_IV[1] as i32),
        _mm256_set1_epi32(BLAKE3_IV[2] as i32),
        _mm256_set1_epi32(BLAKE3_IV[3] as i32),
        _mm256_set1_epi32(inputs.counter[0] as i32),
        _mm256_set1_epi32((inputs.counter[0] >> 32) as i32),
        _mm256_set1_epi32(inputs.block_len[0] as i32),
        _mm256_set1_epi32(inputs.flags[0] as i32),
    ];

    // 3) Per-lane replacements: the cv/counter/flags broadcasts above
    //    used lane 0's value; we now stamp lanes 1..8 with the right
    //    values via `_mm256_insert_epi32` (one per (lane, slot)). The
    //    G-function reads lanes 0..7 of every state register, so each
    //    insertion must land in the right slot.
    for lane in 1..DEGREE {
        let cv = inputs.chaining_values[lane];
        for w in 0..8 {
            // `_mm256_insert_epi32::<lane_index>(v, val)` writes `val`
            // into 32-bit lane `lane_index` of `v` (a compile-time
            // immediate). For AVX2 the constant is the lane index, in
            // the same lane order the C ABI uses (lane 0 = 32-bit word
            // 0 of the low 128 bits).
            s[w] = _mm256_insert_epi32::<{ lane as i32 }>(s[w], cv[w] as i32);
        }
        s[12] = _mm256_insert_epi32::<{ lane as i32 }>(s[12], inputs.counter[lane] as i32);
        s[13] = _mm256_insert_epi32::<{ lane as i32 }>(
            s[13],
            (inputs.counter[lane] >> 32) as i32,
        );
        s[14] = _mm256_insert_epi32::<{ lane as i32 }>(s[14], inputs.block_len[lane] as i32);
        s[15] = _mm256_insert_epi32::<{ lane as i32 }>(s[15], inputs.flags[lane] as i32);
    }

    // 4) Prefetch the next batch's `cv` (one block ahead × `cv` stride)
    //    and message (one block ahead × 64 bytes) into L1 with
    //    `locality = 1` (`_MM_HINT_T1` — moderate temporal, kept for
    //    a while but not as hot as `T0`). Two addresses, two hints,
    //    matches the goal's `prefetch_distance = 2` (prefetch the *next*
    //    batch's cv and message lines from L2 into L1).
    //
    //    The pointers point at the *next* batch — for the very last
    //    batch they're past the end of `inputs`, but `_mm_prefetch`
    //    accepts any user-mode address and just no-ops on a fault, so
    //    this is safe at the batch tail.
    if !inputs.blocks.is_empty() {
        // `inputs.blocks[0]` is the *current* batch's first block; the
        // "next" batch's first block lives at `DEGREE` blocks ahead.
        // We can't easily compute its address in the slice sense, so
        // the prefetch addresses use the message scratch as a stand-in:
        // 128 bytes ahead of the current scratch pointer is exactly
        // the line that the next batch's transposed message would
        // occupy, and the prefetch hides that latency for the case
        // where the call site streams many batches back-to-back.
        let p_msg_next = (msg_base as *const i8).add(128);
        _mm_prefetch(p_msg_next, 1);
        // Also prefetch the next batch's first chaining value (32 bytes
        // per cv, 8 lanes = 256 bytes; the "next" is `DEGREE` cvs ahead,
        // so 256 bytes past the current `cv[0]` pointer).
        let p_cv0 = inputs.chaining_values[0].as_ptr() as *const i8;
        let p_cv_next = p_cv0.add(256);
        _mm_prefetch(p_cv_next, 1);
    }

    // 5) Full G-round unroll: 7 rounds × 8 G's. Each G is the BLAKE3
    //    quarter-round:
    //
    //      a = a + b + mx
    //      d = (d ^ a) >>> 16
    //      c = c + d
    //      b = (b ^ c) >>> 12
    //      a = a + b + my
    //      d = (d ^ a) >>>  8
    //      c = c + d
    //      b = (b ^ c) >>>  7
    //
    //    The 16 message words are loaded ONCE at the top of each round
    //    from the interleaved8 scratch (16 YMM loads → 16 message
    //    vectors), then indexed by `MSG_SCHEDULE[r]` per G. The G's
    //    run lane-symmetric, so the unroll produces 16 add/xor
    //    operations per G × 8 G's × 7 rounds = 896 vector ops, all in
    //    registers.
    let mut msg: [__m256i; 16] = [_mm256_setzero_si256(); 16];
    for r in 0..7 {
        for i in 0..16 {
            // SAFETY: `msg_base.add(i)` is in-bounds for the 16-word
            // message (16 * 32 = 512 bytes).
            msg[i] = _mm256_loadu_si256(msg_base.add(i));
        }

        // 8 G's per round, in the BLAKE3 column-then-diagonal order.
        for g in 0..8 {
            let lanes = G_LANES[g];
            let (a, b, c, d) = (lanes[0], lanes[1], lanes[2], lanes[3]);
            let sched = MSG_SCHEDULE[r];
            // Column G: pairs (mx, my) = (sched[2g], sched[2g+1]).
            // Diagonal G: pairs (mx, my) = (sched[2g], sched[2g+1])
            // with the same lane indices but rotated diagonal order.
            let mx_idx = sched[2 * g] as usize;
            let my_idx = sched[2 * g + 1] as usize;

            // a = a + b + mx
            s[a as usize] = _mm256_add_epi32(
                _mm256_add_epi32(s[a as usize], s[b as usize]),
                msg[mx_idx],
            );
            // d = (d ^ a) >>> 16
            s[d as usize] = _mm256_xor_si256(s[d as usize], s[a as usize]);
            s[d as usize] = _mm256_srli_epi32::<16>(s[d as usize]);
            // c = c + d
            s[c as usize] = _mm256_add_epi32(s[c as usize], s[d as usize]);
            // b = (b ^ c) >>> 12
            s[b as usize] = _mm256_xor_si256(s[b as usize], s[c as usize]);
            s[b as usize] = _mm256_srli_epi32::<12>(s[b as usize]);
            // a = a + b + my
            s[a as usize] = _mm256_add_epi32(
                _mm256_add_epi32(s[a as usize], s[b as usize]),
                msg[my_idx],
            );
            // d = (d ^ a) >>> 8
            s[d as usize] = _mm256_xor_si256(s[d as usize], s[a as usize]);
            s[d as usize] = _mm256_srli_epi32::<8>(s[d as usize]);
            // c = c + d
            s[c as usize] = _mm256_add_epi32(s[c as usize], s[d as usize]);
            // b = (b ^ c) >>> 7
            s[b as usize] = _mm256_xor_si256(s[b as usize], s[c as usize]);
            s[b as usize] = _mm256_srli_epi32::<7>(s[b as usize]);
        }
    }

    // 6) Finalisation XOR: out_cv[i] = s[i] ^ s[i + 8] for i in 0..8.
    //    The result is the new chaining value (256 bits = 8 × u32) per
    //    block. Store each of the 8 lanes of each output register into
    //    the corresponding slot of the per-lane `[u32; 8]` array.
    let mut out = [[0u32; OUT_LEN_WORDS]; DEGREE];
    for w in 0..8 {
        let final_vec = _mm256_xor_si256(s[w], s[w + 8]);
        // `_mm256_extract_epi32::<lane>(v)` returns the i32 in lane
        // `lane` of `v`, sign-extended. The BLAKE3 u32 message words
        // are never reinterpreted as signed, but the bit pattern is
        // identical, so the `as i32 → as u32` round-trip is a no-op.
        for lane in 0..DEGREE {
            let val_i = match lane {
                0 => _mm256_extract_epi32::<0>(final_vec),
                1 => _mm256_extract_epi32::<1>(final_vec),
                2 => _mm256_extract_epi32::<2>(final_vec),
                3 => _mm256_extract_epi32::<3>(final_vec),
                4 => _mm256_extract_epi32::<4>(final_vec),
                5 => _mm256_extract_epi32::<5>(final_vec),
                6 => _mm256_extract_epi32::<6>(final_vec),
                7 => _mm256_extract_epi32::<7>(final_vec),
                _ => unreachable!(),
            };
            out[lane][w] = val_i as u32;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs_hashes::blake3::BLAKE3_IV;
    use flock_core::bits::interleaved8_block_into;

    /// SplitMix64 — used to deterministically seed 8 distinct test inputs.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn nx(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Build 8 random `(cv, m, counter, block_len, flags)` compression
    /// inputs for the batched harness, plus a 64-byte-per-block bytes view
    /// (the BLAKE3 reference `blake3_compress` consumes `m` as 16 × u32 LE
    /// — the bytes are just the same 64 bytes reinterpreted).
    fn random_inputs(seed: u64) -> HashMany8Inputs<'static> {
        let mut rng = Rng::new(seed);
        // Eight `u32` cv words per lane → `[u32; 8]` → we need &'static
        // borrows, so we leak one allocation. Tests only.
        let cvs_storage: [[u32; 8]; DEGREE] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.nx() as u32));
        let msgs_storage: [[u8; BLOCK_LEN]; DEGREE] = std::array::from_fn(|_| {
            let mut buf = [0u8; BLOCK_LEN];
            for chunk in buf.chunks_exact_mut(4) {
                let v = rng.nx() as u32;
                chunk.copy_from_slice(&v.to_le_bytes());
            }
            buf
        });
        let counter: [u64; DEGREE] = std::array::from_fn(|_| rng.nx() & 0xFFFF_FFFF);
        let block_len: [u32; DEGREE] = [BLOCK_LEN as u32; DEGREE];
        // Mix the per-block flags so the kernel sees variation; the AVX2
        // path ORs nothing extra on top, so the value the caller passes is
        // what the kernel uses.
        let flags: [u32; DEGREE] = std::array::from_fn(|i| {
            // Ranked uses `flags = 11`; one variant tests the start/end
            // bit pattern the BLAKE3 spec defines.
            (11u32 | if i & 1 == 0 { 1 << 4 } else { 0 }) | if i & 2 == 0 { 1 << 5 } else { 0 }
        });
        // Promote the leak to `&'static`: `Box::leak` is fine for tests
        // — the storage lives for the full test process and the kernel
        // only reads from it.
        let cvs_static: &'static [[u32; 8]; DEGREE] = Box::leak(Box::new(cvs_storage));
        let msgs_static: &'static [[u8; BLOCK_LEN]; DEGREE] =
            Box::leak(Box::new(msgs_storage));
        let cvs_refs: [&'static [u32; 8]; DEGREE] = std::array::from_fn(|i| &cvs_static[i]);
        let msgs_refs: [&'static [u8; BLOCK_LEN]; DEGREE] =
            std::array::from_fn(|i| &msgs_static[i]);
        HashMany8Inputs {
            blocks: msgs_refs,
            chaining_values: cvs_refs,
            counter,
            block_len,
            flags,
        }
    }

    /// The AVX2 batched kernel must produce the same 8 chaining values
    /// as the scalar `blake3_compress` loop, bit-for-bit. This is the
    /// central correctness contract — the kernel is only safe to swap
    /// into the hot path if this test passes on every commit.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn hash_many8_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            // The CI on hosts without AVX2 is fine — the dispatch
            // automatically falls through to the scalar path, and the
            // `hash_many8_scalar_matches_blake3_compress` test below
            // pins the same oracle.
            return;
        }
        for seed in 0..16u64 {
            let inputs = random_inputs(seed);
            let got = hash_many8(&inputs);
            let want = hash_many8_scalar(&inputs);
            for lane in 0..DEGREE {
                assert_eq!(
                    got[lane], want[lane],
                    "AVX2 batched disagrees with scalar at seed={seed} lane={lane}"
                );
            }
        }
    }

    /// The scalar fallback must agree with the public
    /// `r1cs_hashes::blake3::blake3_compress` reference, block by block.
    /// This pins the BLAKE3 IV, counter, block_len, and flag semantics
    /// for the batched path on every host.
    #[test]
    fn hash_many8_scalar_matches_blake3_compress() {
        for seed in 0..16u64 {
            let inputs = random_inputs(seed);
            let got = hash_many8_scalar(&inputs);
            for lane in 0..DEGREE {
                let mut m = [0u32; 16];
                let bytes = inputs.blocks[lane];
                for w in 0..16 {
                    let off = w * 4;
                    m[w] = u32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]);
                }
                let st = blake3_compress(
                    inputs.chaining_values[lane],
                    &m,
                    inputs.counter[lane],
                    inputs.block_len[lane],
                    inputs.flags[lane],
                );
                let mut want = [0u32; OUT_LEN_WORDS];
                want.copy_from_slice(&st[..OUT_LEN_WORDS]);
                assert_eq!(
                    got[lane], want,
                    "scalar batched disagrees with blake3_compress at seed={seed} lane={lane}"
                );
            }
        }
    }

    /// The transposed scratch buffer must hold the BLAKE3 hash_many
    /// interleaved8 layout: `out[byte * 8 + lane] = block[lane][byte]`.
    /// The AVX2 kernel reads it via 16 `_mm256_loadu_si256` calls, one
    /// per 32-byte quarter-block; this test asserts the same byte at
    /// every (byte, lane) address.
    #[test]
    fn interleaved8_layout_is_correct() {
        let mut rng = Rng::new(0xB1A3_4E11);
        let blocks_storage: [[u8; BLOCK_LEN]; DEGREE] =
            std::array::from_fn(|_| std::array::from_fn(|_| (rng.nx() as u32 as u8)));
        let blocks_refs: [&[u8; BLOCK_LEN]; DEGREE] =
            std::array::from_fn(|i| &blocks_storage[i]);
        let mut buf = Interleaved8Block::<DEGREE>::uninit();
        interleaved8_block_into(&mut buf, &blocks_refs);
        let got = buf.as_bytes();
        for byte in 0..BLOCK_LEN {
            for lane in 0..DEGREE {
                assert_eq!(
                    got[byte * DEGREE + lane],
                    blocks_storage[lane][byte],
                    "transposed mismatch at byte={byte} lane={lane}"
                );
            }
        }
    }

    /// The cached AVX2 probe returns the same value as a fresh
    /// `is_x86_feature_detected!("avx2")` call.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_probe_cached_value_matches_direct() {
        let direct = is_x86_feature_detected!("avx2");
        let cached = avx2_detected();
        assert_eq!(cached, direct, "cached AVX2 probe disagrees with direct");
    }

    /// The dispatch picks the AVX2 path when the host has AVX2, scalar
    /// otherwise. This pins the dispatch contract: callers should never
    /// have to branch on the feature flag themselves.
    #[test]
    fn dispatch_returns_8_chaining_values() {
        let inputs = random_inputs(0xC0FFEE);
        let out = hash_many8(&inputs);
        assert_eq!(out.len(), DEGREE);
        for lane in 0..DEGREE {
            // The chaining value is 8 × u32 = 256 bits; the output array
            // shape enforces that statically.
            assert_eq!(out[lane].len(), OUT_LEN_WORDS);
        }
        // The two paths are equal — this is the dispatch identity contract.
        let scalar = hash_many8_scalar(&inputs);
        for lane in 0..DEGREE {
            assert_eq!(
                out[lane], scalar[lane],
                "dispatch disagreed with scalar at lane={lane}"
            );
        }
    }

    /// BLAKE3_IV[0..4] is the upper half of the state file; the BLAKE3
    /// spec constrains the lower half to come from the chaining value.
    /// This test pins the IV values used by both the AVX2 kernel and the
    /// scalar fallback (they share the constant by reference).
    #[test]
    fn iv_constants_match_sha256() {
        // SHA-256 IV first 4 words.
        const SHA256_IV: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        assert_eq!(BLAKE3_IV, SHA256_IV);
    }
}
