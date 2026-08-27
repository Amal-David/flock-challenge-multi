//! 8-wide AVX2 lockstep BLAKE3 witness builder (`__m256i`, 8×u32).
//!
//! Same G-function / carry-bit / packed-row stream as the 4-wide SSE kernel,
//! widened to one rayon group (8 compressions) per call. The z drain
//! optionally publishes through streaming stores (`z_nt`): z's next reader is
//! the commit encode, a later phase, so its write-allocate RFO is pure waste
//! (see the caller's gate).
//!
//! The a/b drains have two shapes, selected by `win_ab`:
//!  * `None` — the incumbent: temporal (`storeu`), because the caller re-reads
//!    a/b L1-hot for the round-1 window precompute in the same task.
//!  * `Some(..)` — FUSED: the very same `tr8` registers feed the main a/b
//!    buffers non-temporally AND a compact per-octa window buffer that the
//!    caller projects from instead. The main buffers are then never re-read,
//!    so their write-allocate RFO (1 GiB at the ranked shape) is deleted too.
//!
//! Ranked live path: `generate_witness_with_ab_packed_and_round1_inner_impl`
//! (`FLOCK_NO_WITGEN_LIVE_SIMD=1` restores the scalar 1-block loop).

use super::{
    ADDS_PER_G, BLAKE3_IV, CARRY_BITS_PER_ADD, Compression, G_STRIDE, GS_BASE, K, OUT_HI_BASE,
    USEFUL_BITS, WORD_BITS,
};
use core::arch::x86_64::*;
use flock_core::ntt::InvNttTableByteSingleGf8;
use flock_core::zerocheck::univariate_skip_optimized::{
    ROUND1_AB_OFF_WORDS, Round1AbTableImages, Round1AbWindowPlan,
    round1_ab_inner_window_from_offsets, round1_ab_inner_window_with_images,
    round1_ab_table_images,
};

const REC_C0: usize = 0;
const REC_C1: usize = CARRY_BITS_PER_ADD;
const REC_C2: usize = 2 * CARRY_BITS_PER_ADD;
const REC_C3: usize = 3 * CARRY_BITS_PER_ADD;
const REC_C4: usize = 4 * CARRY_BITS_PER_ADD;
const REC_C5: usize = 5 * CARRY_BITS_PER_ADD;
const REC_LIN0: usize = ADDS_PER_G * CARRY_BITS_PER_ADD;
const REC_LIN1: usize = REC_LIN0 + WORD_BITS;
const U32_PER_BLOCK: usize = K / 32;
const BYTES_PER_BLOCK: usize = K / 8;
const DUMP_CHUNKS: usize = U32_PER_BLOCK / 8;
/// Words a drain step publishes at once — sixteen, which is exactly one
/// 64-byte round-1 medium window per block.
const STEP_WORDS: usize = 16;
/// u32s of the streaming projection's staging pair: one step's sixteen words
/// for eight blocks, a side.
pub(crate) const STREAM_STAGE_WORDS: usize = 2 * 8 * STEP_WORDS;
// 62, not 61: an odd boundary leaves the z drain's paired NT loop with a
// lone 32-byte tail chunk — one masked, partially-written NT line per block
// (an ECC read-modify-write at the memory controller, 2^18 times per
// proof). Chunk 61 is entirely inside the zero tail (LAST_WORD = 481), so
// storing it is redundant-but-correct and the ragged tail branch vanishes.
const ELIDE_ZERO_CHUNK: usize = 62;
const ELIDE_B_TAIL_CHUNK: usize = 59;
// 60, not 59, for the FUSED (`dump_range_nt_win`) drain, for exactly the reason
// `ELIDE_ZERO_CHUNK` is 62: an odd first-elided chunk leaves chunk 58 as a lone
// 32-byte NT store inside the 64-byte line (58, 59) — a partially-filled
// write-combining buffer, i.e. a read-modify-write at the memory controller,
// once per block per prove. Chunks 60..64 are entirely inside b's fixed
// lin-id/out_hi ones + zero padding run (which starts at bit 15,089 < 256*60),
// so storing chunk 59 is redundant-but-correct and the ragged line vanishes.
// The temporal (`dump_range`) arm has no write-combining buffer to leave open,
// so it keeps the tighter 59.
const ELIDE_B_TAIL_CHUNK_WIN: usize = 60;
const ELIDE_B_PREFIX_CHUNKS: usize = 4;
const LAST_WORD: usize = (USEFUL_BITS - 1) / 32;
const _ELIDE_GEOMETRY: () = {
    assert!(8 * ELIDE_ZERO_CHUNK >= USEFUL_BITS.div_ceil(32));
    assert!(8 * ELIDE_ZERO_CHUNK < U32_PER_BLOCK);
    assert!(8 * ELIDE_B_PREFIX_CHUNKS <= 36);
    assert!(LAST_WORD == 481);
    // The fused drain walks chunk PAIRS from 0 while `dump_range_nt` walks
    // pairs from `g0`; the two agree on which chunks share a 64-byte line
    // only while every elided PREFIX is pair-aligned. (Elided tails need no
    // such property: a half-covered trailing pair degrades to the same
    // single-chunk stream in both.)
    assert!(DUMP_CHUNKS % 2 == 0);
    assert!(ELIDE_B_PREFIX_CHUNKS % 2 == 0);
    // The fused b tail is a pair-aligned SUBSET of the temporal one, so it
    // stays strictly inside the same content-independent constant run.
    assert!(ELIDE_B_TAIL_CHUNK_WIN >= ELIDE_B_TAIL_CHUNK);
    assert!(ELIDE_B_TAIL_CHUNK_WIN % 2 == 0);
    assert!(ELIDE_B_TAIL_CHUNK_WIN < DUMP_CHUNKS);
};

type V8 = __m256i;

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2"
))]
type V16 = __m512i;

/// Input form for one eight-compression witness kernel invocation.
///
/// The ranked speculative path carries the generator's closed form all the
/// way here. Producing the 25 draws directly as eight SIMD lanes avoids
/// materializing and immediately transposing an 896-byte `[Compression; 8]`.
/// Slice callers retain the original borrowed-input route verbatim.
pub(crate) enum OctaInputs<'a> {
    Blocks([&'a Compression; 8]),
    Closed { init: u64, base: usize },
}

struct PreparedInputs {
    cv: [V8; 8],
    message: [V8; 16],
    counter_lo: V8,
    counter_hi: V8,
    block_len: V8,
    flags: V8,
}

/// Closed-form inputs for the sixteen-lane kernel.  This deliberately stays
/// as one aligned memory object: `run_w16` borrows it out of line and loads
/// only the message/CV word needed by the current G, rather than extending 24
/// more ZMM live ranges across the compression state.
#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2"
))]
#[repr(C, align(64))]
struct PreparedInputs16 {
    cv: [V16; 8],
    message: [V16; 16],
    counter_lo: V16,
    counter_hi: V16,
    block_len: V16,
    flags: V16,
}

/// AVX2 fallback for low-half 64-bit multiplication. `_mm256_mul_epu32`
/// handles the low limbs; the two cross-products supply bits 32..63. The
/// high×high term is above bit 63 and is discarded, exactly like
/// `u64::wrapping_mul`.
#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn mullo_u64x4(a: __m256i, b: u64) -> __m256i {
    unsafe {
        let b_lo = _mm256_set1_epi64x(u64::from(b as u32) as i64);
        let b_hi = _mm256_set1_epi64x((b >> 32) as i64);
        let lo = _mm256_mul_epu32(a, b_lo);
        let a_hi = _mm256_srli_epi64::<32>(a);
        let cross_lo = _mm256_mul_epu32(a, b_hi);
        let cross_hi = _mm256_mul_epu32(a_hi, b_lo);
        let cross = _mm256_add_epi64(cross_lo, cross_hi);
        _mm256_add_epi64(lo, _mm256_slli_epi64::<32>(cross))
    }
}

#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn mix_u64x4(mut z: __m256i) -> __m256i {
    unsafe {
        z = _mm256_xor_si256(z, _mm256_srli_epi64::<30>(z));
        z = mullo_u64x4(z, 0xBF58_476D_1CE4_E5B9);
        z = _mm256_xor_si256(z, _mm256_srli_epi64::<27>(z));
        z = mullo_u64x4(z, 0x94D0_49BB_1331_11EB);
        _mm256_xor_si256(z, _mm256_srli_epi64::<31>(z))
    }
}

/// Pack the low u32 of eight u64 lanes into one `V8`, preserving block order.
#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn pack_low_u32x8(lo: __m256i, hi: __m256i) -> V8 {
    unsafe {
        let indices = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let lo = _mm256_permutevar8x32_epi32(lo, indices);
        let hi = _mm256_permutevar8x32_epi32(hi, indices);
        _mm256_permute2x128_si256::<0x20>(lo, hi)
    }
}

#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn next_generator_draw(states: &mut [__m256i; 2]) -> V8 {
    unsafe {
        let golden = _mm256_set1_epi64x(crate::seed_pipe::GOLDEN as i64);
        states[0] = _mm256_add_epi64(states[0], golden);
        states[1] = _mm256_add_epi64(states[1], golden);
        pack_low_u32x8(mix_u64x4(states[0]), mix_u64x4(states[1]))
    }
}

/// Generate the exact protected-wrapper inputs for blocks `base..base+8`
/// directly in the word-major form consumed by the witness kernel. AVX2
/// targets use two four-lane SplitMix states and the low-limb multiplication
/// above; Sapphire Rapids uses the eight-lane AVX-512DQ implementation below.
#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn prepare_closed_inputs(init: u64, base: usize) -> PreparedInputs {
    unsafe {
        let stride =
            crate::seed_pipe::GOLDEN.wrapping_mul(crate::seed_pipe::DRAWS_PER_BLOCK as u64);
        let first = init.wrapping_add((base as u64).wrapping_mul(stride));
        let mut states = [
            _mm256_setr_epi64x(
                first as i64,
                first.wrapping_add(stride) as i64,
                first.wrapping_add(stride.wrapping_mul(2)) as i64,
                first.wrapping_add(stride.wrapping_mul(3)) as i64,
            ),
            _mm256_setr_epi64x(
                first.wrapping_add(stride.wrapping_mul(4)) as i64,
                first.wrapping_add(stride.wrapping_mul(5)) as i64,
                first.wrapping_add(stride.wrapping_mul(6)) as i64,
                first.wrapping_add(stride.wrapping_mul(7)) as i64,
            ),
        ];
        let cv = std::array::from_fn(|_| next_generator_draw(&mut states));
        let message = std::array::from_fn(|_| next_generator_draw(&mut states));
        let counter_lo = next_generator_draw(&mut states);
        PreparedInputs {
            cv,
            message,
            counter_lo,
            counter_hi: _mm256_setzero_si256(),
            block_len: _mm256_set1_epi32(64),
            flags: _mm256_set1_epi32(11),
        }
    }
}

#[cfg(target_feature = "avx512dq")]
#[inline(always)]
unsafe fn mix_u64x8(mut z: __m512i) -> __m512i {
    unsafe {
        z = _mm512_xor_si512(z, _mm512_srli_epi64::<30>(z));
        z = _mm512_mullo_epi64(z, _mm512_set1_epi64(0xBF58_476D_1CE4_E5B9u64 as i64));
        z = _mm512_xor_si512(z, _mm512_srli_epi64::<27>(z));
        z = _mm512_mullo_epi64(z, _mm512_set1_epi64(0x94D0_49BB_1331_11EBu64 as i64));
        _mm512_xor_si512(z, _mm512_srli_epi64::<31>(z))
    }
}

#[cfg(target_feature = "avx512dq")]
#[inline(always)]
unsafe fn next_generator_draw(state: &mut __m512i) -> V8 {
    unsafe {
        *state = _mm512_add_epi64(*state, _mm512_set1_epi64(crate::seed_pipe::GOLDEN as i64));
        _mm512_cvtepi64_epi32(mix_u64x8(*state))
    }
}

#[cfg(target_feature = "avx512dq")]
#[inline(always)]
unsafe fn prepare_closed_inputs(init: u64, base: usize) -> PreparedInputs {
    unsafe {
        let stride =
            crate::seed_pipe::GOLDEN.wrapping_mul(crate::seed_pipe::DRAWS_PER_BLOCK as u64);
        let first = init.wrapping_add((base as u64).wrapping_mul(stride));
        let mut state = _mm512_setr_epi64(
            first as i64,
            first.wrapping_add(stride) as i64,
            first.wrapping_add(stride.wrapping_mul(2)) as i64,
            first.wrapping_add(stride.wrapping_mul(3)) as i64,
            first.wrapping_add(stride.wrapping_mul(4)) as i64,
            first.wrapping_add(stride.wrapping_mul(5)) as i64,
            first.wrapping_add(stride.wrapping_mul(6)) as i64,
            first.wrapping_add(stride.wrapping_mul(7)) as i64,
        );
        let cv = std::array::from_fn(|_| next_generator_draw(&mut state));
        let message = std::array::from_fn(|_| next_generator_draw(&mut state));
        let counter_lo = next_generator_draw(&mut state);
        PreparedInputs {
            cv,
            message,
            counter_lo,
            counter_hi: _mm256_setzero_si256(),
            block_len: _mm256_set1_epi32(64),
            flags: _mm256_set1_epi32(11),
        }
    }
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2"
))]
#[inline(always)]
unsafe fn join_v8(lo: V8, hi: V8) -> V16 {
    unsafe { _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi) }
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2"
))]
#[inline(always)]
unsafe fn low_v8(v: V16) -> V8 {
    unsafe { _mm512_castsi512_si256(v) }
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2"
))]
#[inline(always)]
unsafe fn high_v8(v: V16) -> V8 {
    unsafe { _mm512_extracti64x4_epi64::<1>(v) }
}

/// Form lanes 0..7 and 8..15 with the already-established closed generator,
/// then join each word.  No new generator arithmetic or draw ordering exists
/// in the W16 path.
#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2"
))]
#[inline(never)]
unsafe fn prepare_closed_inputs16(init: u64, base: usize) -> PreparedInputs16 {
    unsafe {
        let lo = prepare_closed_inputs(init, base);
        let hi = prepare_closed_inputs(init, base + 8);
        PreparedInputs16 {
            cv: core::array::from_fn(|i| join_v8(lo.cv[i], hi.cv[i])),
            message: core::array::from_fn(|i| join_v8(lo.message[i], hi.message[i])),
            counter_lo: join_v8(lo.counter_lo, hi.counter_lo),
            counter_hi: join_v8(lo.counter_hi, hi.counter_hi),
            block_len: join_v8(lo.block_len, hi.block_len),
            flags: join_v8(lo.flags, hi.flags),
        }
    }
}

#[inline(always)]
unsafe fn load_v8(p: *const u32) -> V8 {
    unsafe { _mm256_loadu_si256(p.cast::<__m256i>()) }
}

#[inline(always)]
unsafe fn store_v8(p: *mut u32, v: V8) {
    unsafe { _mm256_storeu_si256(p.cast::<__m256i>(), v) }
}

/// Widen one staged 64-byte window side (its two transposed V8 rows, still in
/// registers) to the pre-scaled `u16` offset half the split shift-reduce
/// kernel consumes: `byte * 64` per lane, two ZMM stores. Identical values to
/// the kernel's own prologue — the source is the register the staging store
/// was made from instead of a reload of that store.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn widen_off_half(lo: V8, hi: V8, op: *mut u16) {
    unsafe {
        let w = |v: V8| _mm512_slli_epi16::<6>(_mm512_cvtepu8_epi16(v));
        _mm512_store_si512(op.cast::<__m512i>(), w(lo));
        _mm512_store_si512(op.add(32).cast::<__m512i>(), w(hi));
    }
}

#[inline(always)]
fn dup_u32(x: u32) -> V8 {
    unsafe { _mm256_set1_epi32(x as i32) }
}

#[inline(always)]
fn xor_v8(a: V8, b: V8) -> V8 {
    unsafe { _mm256_xor_si256(a, b) }
}

/// Three-way XOR `a ^ b ^ c`. On AVX-512VL this folds the two `vpxord`s of the
/// carry-in chain into one `vpternlogd` with immediate `0x96` (the truth table
/// of `a ^ b ^ c`, order-independent, bit-identical to the paired XORs).
#[inline(always)]
fn xor3_v8(a: V8, b: V8, c: V8) -> V8 {
    #[cfg(target_feature = "avx512vl")]
    unsafe {
        _mm256_ternarylogic_epi32::<0x96>(a, b, c)
    }
    // AVX2-only hosts (no AVX-512VL): the paired XORs this folds. Bit-identical.
    #[cfg(not(target_feature = "avx512vl"))]
    unsafe {
        _mm256_xor_si256(_mm256_xor_si256(a, b), c)
    }
}

#[inline(always)]
fn or_v8(a: V8, b: V8) -> V8 {
    unsafe { _mm256_or_si256(a, b) }
}

#[inline(always)]
fn and_v8(a: V8, b: V8) -> V8 {
    unsafe { _mm256_and_si256(a, b) }
}

#[inline(always)]
fn add_v8(a: V8, b: V8) -> V8 {
    unsafe { _mm256_add_epi32(a, b) }
}

#[inline(always)]
fn shr_v8<const N: i32>(v: V8) -> V8 {
    unsafe { _mm256_srli_epi32::<N>(v) }
}

#[inline(always)]
fn shl_v8<const N: i32>(v: V8) -> V8 {
    unsafe { _mm256_slli_epi32::<N>(v) }
}

/// NEON `vsli` #N, 8 lanes: bits `N..32` from `b << N`, bits `0..N` keep `a`.
#[inline(always)]
fn vsli_v8<const N: i32>(a: V8, b: V8) -> V8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // One VPTERNLOGD (AVX512F, EVEX.256) folds the AND+OR of the AVX2
        // emulation: out = (b << N) | (a & mask) == ternlog<0xF8>(b<<N, a, mask).
        // Bit-identical; the carry-packing is the hot half of the witness
        // G-functions, and on the ranked runner the ternlog saves one op per
        // push (~16 per G, ~900 per 8-block call). The mask is a compile-time
        // constant here (N is literal), so it folds into the ternlog's memory
        // operand or a constant broadcast; no register pressure change.
        unsafe {
            let mask = _mm256_set1_epi32(((1u64 << N) - 1) as u32 as i32);
            _mm256_ternarylogic_epi32::<0xF8>(_mm256_slli_epi32::<N>(b), a, mask)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512vl")))]
    unsafe {
        let mask = _mm256_set1_epi32(((1u64 << N) - 1) as u32 as i32);
        _mm256_or_si256(_mm256_slli_epi32::<N>(b), _mm256_and_si256(a, mask))
    }
}

/// 8×8 u32 transpose. `r[i]` lane `j` becomes `out[j]` lane `i`.
#[inline(always)]
fn tr8(v0: V8, v1: V8, v2: V8, v3: V8, v4: V8, v5: V8, v6: V8, v7: V8) -> [V8; 8] {
    unsafe {
        let t0 = _mm256_unpacklo_epi32(v0, v1);
        let t1 = _mm256_unpackhi_epi32(v0, v1);
        let t2 = _mm256_unpacklo_epi32(v2, v3);
        let t3 = _mm256_unpackhi_epi32(v2, v3);
        let t4 = _mm256_unpacklo_epi32(v4, v5);
        let t5 = _mm256_unpackhi_epi32(v4, v5);
        let t6 = _mm256_unpacklo_epi32(v6, v7);
        let t7 = _mm256_unpackhi_epi32(v6, v7);

        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);
        let u4 = _mm256_unpacklo_epi64(t4, t6);
        let u5 = _mm256_unpackhi_epi64(t4, t6);
        let u6 = _mm256_unpacklo_epi64(t5, t7);
        let u7 = _mm256_unpackhi_epi64(t5, t7);

        [
            _mm256_permute2x128_si256::<0x20>(u0, u4),
            _mm256_permute2x128_si256::<0x20>(u1, u5),
            _mm256_permute2x128_si256::<0x20>(u2, u6),
            _mm256_permute2x128_si256::<0x20>(u3, u7),
            _mm256_permute2x128_si256::<0x31>(u0, u4),
            _mm256_permute2x128_si256::<0x31>(u1, u5),
            _mm256_permute2x128_si256::<0x31>(u2, u6),
            _mm256_permute2x128_si256::<0x31>(u3, u7),
        ]
    }
}

const RING_WORDS: usize = 32;
/// Words the pre-round prologue fills, starting at word 16.
const PROLOGUE_WORDS: usize = 20;
const _RING_GEOMETRY: () = {
    assert!(RING_WORDS >= 32);
    assert!(RING_WORDS <= U32_PER_BLOCK);
    assert!(RING_WORDS & (RING_WORDS - 1) == 0);
    // Every epoch boundary is a whole number of drain steps.
    assert!(RING_WORDS % STEP_WORDS == 0);
};

/// Streaming round-1 projection wired into the a/b drain: every 16-word drain
/// step is one 64-byte round-1 medium window per block, so the transform runs
/// off a `STREAM_STAGE_WORDS` staging pair as the words are produced instead
/// of off two full-block window buffers.
///
/// `stage` owns `STREAM_STAGE_WORDS` u32s (a side then b side, eight 16-word
/// block rows each) and is 64-byte aligned. `out` owns this octa's eight
/// `BYTES_PER_BLOCK` ab_inner blocks. Bit `j` of `live` selects block `j`.
pub(crate) struct StreamProj<'t> {
    pub(crate) stage: *mut u32,
    pub(crate) out: *mut u8,
    pub(crate) live: u32,
    pub(crate) inv_table: &'t InvNttTableByteSingleGf8,
    pub(crate) plan: Round1AbWindowPlan,
}

impl StreamProj<'_> {
    #[inline(always)]
    fn sides(&self) -> (*mut u32, *mut u32) {
        // SAFETY: the staging owns `STREAM_STAGE_WORDS` u32s.
        (self.stage, unsafe { self.stage.add(8 * STEP_WORDS) })
    }

    /// Transform the staged window `blk` (`0..2 · 16`) for every live block.
    ///
    /// # Safety
    /// Both staging sides hold the eight blocks' bytes for window `blk`.
    #[inline(never)]
    unsafe fn project(&self, blk: usize) {
        unsafe {
            let (plan, imgs) = self.window_prep(blk);
            self.project_blocks(blk, plan, imgs, None, None);
        }
    }

    /// The window's per-block invariants: its static-B eligibility and the
    /// table images the kernel addresses. Both are the same for all eight
    /// blocks, so a producer that walks the eight blocks in several pieces
    /// resolves them ONCE and hands them to every piece.
    #[inline(always)]
    fn window_prep(&self, blk: usize) -> (Round1AbWindowPlan, Round1AbTableImages) {
        let p = self.plan.for_window(blk);
        (p, round1_ab_table_images(self.inv_table, p))
    }

    /// [`Self::project`] with the window invariants already resolved, and with
    /// an optional set of drain rows to publish AS the eight blocks are
    /// transformed. `rows` is what makes the streaming stores spread: they are
    /// emitted three per block instead of twenty-four per step, and because
    /// the publisher lives inside the transform loop it costs no extra call
    /// and no second copy of the kernel body.
    ///
    /// # Safety
    /// As for [`Self::project`], and `plan`/`imgs` must be this window's
    /// [`Self::window_prep`]; `rows`, when present, must describe the drain
    /// step that produced this window's staging. `off`, when present, must
    /// point to the eight blocks' prebuilt offset arena
    /// (`8 * ROUND1_AB_OFF_WORDS` u16s, 64-byte aligned, block-major) for
    /// this window's staging bytes, and `plan.offsets_eligible(blk)` must
    /// hold.
    #[inline(never)]
    unsafe fn project_blocks(
        &self,
        blk: usize,
        plan: Round1AbWindowPlan,
        imgs: Round1AbTableImages,
        rows: Option<StepRows>,
        off: Option<*const u16>,
    ) {
        unsafe {
            let (sa, sb) = self.sides();
            let live = self.live;
            // Producer-fused offsets: the drain built the arena straight from
            // its transpose registers, so the staging is never reloaded here.
            // Publish scheduling identical to the loops below.
            if let Some(op) = off {
                debug_assert!(plan.offsets_eligible(blk));
                for j in 0..8usize {
                    if let Some(rows) = rows {
                        rows.publish(j, sa, sb);
                    }
                    if live & (1 << j) == 0 {
                        continue;
                    }
                    let out = &mut *self
                        .out
                        .add(j * BYTES_PER_BLOCK + blk * 64)
                        .cast::<[u8; 64]>();
                    round1_ab_inner_window_from_offsets(
                        &*op.add(j * ROUND1_AB_OFF_WORDS)
                            .cast::<[u16; ROUND1_AB_OFF_WORDS]>(),
                        out,
                        plan,
                        imgs,
                    );
                }
                return;
            }
            for j in 0..8usize {
                // ONE block transformed per THREE lines published: the drain's
                // streaming stores land ~1/8 of a window apart instead of all
                // together. Publishing block `j`'s rows before transforming it
                // is deliberate — the two touch disjoint memory, and this way
                // the eight-block loop below is untouched.
                if let Some(rows) = rows {
                    rows.publish(j, sa, sb);
                }
                if live & (1 << j) == 0 {
                    continue;
                }
                let a_win = &*sa.add(j * STEP_WORDS).cast::<[u8; 64]>();
                let b_win = &*sb.add(j * STEP_WORDS).cast::<[u8; 64]>();
                let out = &mut *self
                    .out
                    .add(j * BYTES_PER_BLOCK + blk * 64)
                    .cast::<[u8; 64]>();
                round1_ab_inner_window_with_images(
                    a_win,
                    b_win,
                    out,
                    blk,
                    self.inv_table,
                    plan,
                    imgs,
                );
            }
        }
    }
}

/// Rolling drain state shared by the A/B packed writers. The witness uses two
/// reusable `RING_WORDS`-word epochs instead of full 512-word stages; B's
/// writer flushes after A has written the epoch, and Z is derived from the
/// transposed A/B rows during that flush.
struct Drain8<'t> {
    ast: *mut V8,
    bs: *mut V8,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    win_ab: Option<(*mut u32, *mut u32)>,
    proj: Option<StreamProj<'t>>,
    elide: [bool; 3],
    z_nt: bool,
    wide_nt: bool,
    spread: bool,
}

/// Convert one low-aligned prior bit to the representation used by [`W8`].
/// Sapphire Rapids keeps pending bits at the high end of each lane so VBMI2
/// can join them to the next field with one `vpshldd`; the portable AVX2
/// fallback retains the incumbent low-aligned representation.
#[inline(always)]
fn packer_initial_bit(bit: V8) -> V8 {
    #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
    {
        shl_v8::<31>(bit)
    }
    #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
    {
        bit
    }
}

/// Recover low-aligned pending bits from the VBMI2 representation. `BACK` is
/// `32 - used`; the ranked stream finishes with `used = 17`, hence `BACK=15`.
#[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
#[inline(always)]
fn packer_high_to_low<const BACK: i32>(pending: V8) -> V8 {
    const {
        assert!(BACK > 0 && BACK < 32);
    }
    shr_v8::<BACK>(pending)
}

/// Lane-wise packed-word writer: 8 independent `PackedWordWriter`s.
///
/// On AVX-512VBMI2+VL targets `pending` is high-aligned: if `USED = u > 0`,
/// its pending low bits occupy bits `32-u..31`.  This makes
/// `vpshldd(v, pending, u)` equal `(v << u) | pending_low` in one instruction.
/// Bits below that high-aligned range are don't-care. Other targets keep the
/// incumbent low-aligned `pending` representation.
struct W8<'t, const FLUSH: bool> {
    pending: V8,
    stage: *mut V8,
    drain: *mut Drain8<'t>,
}

impl<'t, const FLUSH: bool> W8<'t, FLUSH> {
    #[inline(always)]
    fn at(stage: *mut V8, pending: V8, drain: *mut Drain8<'t>) -> Self {
        Self {
            pending,
            stage,
            drain,
        }
    }

    #[inline(always)]
    unsafe fn write_word<const WORD: usize>(&mut self, v: V8) {
        unsafe {
            store_v8(self.stage.add(WORD & (RING_WORDS - 1)) as *mut u32, v);
            if FLUSH && WORD % RING_WORDS == RING_WORDS - 1 {
                // Words 0..15 cannot be published until the final chaining
                // value is known.  The first rolling epoch therefore starts
                // at word 16; later epochs cover their complete 128 words.
                if WORD + 1 == RING_WORDS {
                    (*self.drain).drain_range(16, 16, RING_WORDS - 16);
                } else {
                    (*self.drain).drain_range(WORD + 1 - RING_WORDS, 0, RING_WORDS);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn push<const USED: i32, const WIDTH: i32, const BACK: i32, const WORD: usize>(
        &mut self,
        v: V8,
    ) {
        const {
            assert!(USED >= 0 && USED < 32);
            assert!(WIDTH == 31 || WIDTH == 32);
            assert!(BACK >= 1 && BACK < 32);
            assert!(WORD < U32_PER_BLOCK);
        }
        debug_assert!(USED + WIDTH <= 32 || BACK == 32 - USED);
        unsafe {
            #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
            {
                // Every u>0 push crosses a word boundary because WIDTH is 31
                // or 32.  VPSHLD supplies the old high-aligned pending bits
                // from its second source while shifting the new field left.
                if USED == 0 {
                    if WIDTH == 32 {
                        self.write_word::<WORD>(v);
                        self.pending = dup_u32(0);
                    } else {
                        // 31 pending low bits, moved to bits 1..31.
                        self.pending = shl_v8::<1>(v);
                    }
                } else {
                    let out = _mm256_shldi_epi32::<USED>(v, self.pending);
                    self.write_word::<WORD>(out);
                    // A crossed 31-bit field leaves u-1 low bits; high-aligning
                    // them is exactly v<<1.  A 32-bit field leaves u bits, so
                    // v itself already is the high-aligned representation.
                    self.pending = if WIDTH == 31 { shl_v8::<1>(v) } else { v };
                }
            }
            #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
            {
                if USED == 0 {
                    if WIDTH == 32 {
                        self.write_word::<WORD>(v);
                        self.pending = dup_u32(0);
                    } else {
                        self.pending = v;
                    }
                } else if USED + WIDTH < 32 {
                    self.pending = vsli_v8::<USED>(self.pending, v);
                } else {
                    let out = vsli_v8::<USED>(self.pending, v);
                    self.write_word::<WORD>(out);
                    if USED + WIDTH == 32 {
                        self.pending = dup_u32(0);
                    } else {
                        self.pending = shr_v8::<BACK>(v);
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn finish(&mut self) {
        const {
            assert!(USEFUL_BITS % 32 == 17);
        }
        unsafe {
            // The ranked stream finishes with 17 pending bits.  The VBMI2
            // representation holds those bits in 15..31; the incumbent
            // fallback is already low-aligned.
            #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
            let pending = packer_high_to_low::<15>(self.pending);
            #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
            let pending = self.pending;
            self.write_word::<LAST_WORD>(pending);
        }
    }
}

macro_rules! pushf8 {
    ($w:ident, $pos:expr, $width:literal, $v:expr) => {{
        $w.push::<{ ($pos % 32) as i32 }, $width, {
            let u = ($pos % 32) as i32;
            if u == 0 { 1 } else { 32 - u }
        }, { $pos / 32 }>($v);
    }};
}

#[inline(always)]
fn add_carry_parts_v8(x: V8, y: V8) -> (V8, V8, V8, V8) {
    // `cin = sum ^ x ^ y` is never consumed directly: the pushed parts are
    // `left = x ^ cin` and `right = y ^ cin`, and both collapse algebraically
    // (`left = sum ^ y`, `right = sum ^ x`). Computing them off `sum` removes
    // the carry-in XOR chain entirely — bit-identical outputs, one op less.
    let sum = add_v8(x, y);
    let left = xor_v8(sum, y);
    let right = xor_v8(sum, x);
    let carry = and_v8(left, right);
    (sum, left, right, carry)
}

#[inline(always)]
fn xor_rotr8<const N: i32, const M: i32>(x: V8, y: V8) -> V8 {
    debug_assert_eq!(N + M, 32);
    let v = xor_v8(x, y);
    ror_v8::<N>(v)
}

/// Rotate-right by N on AVX-512 (single `vprold`, EVEX.256) -- three ops on
/// AVX2. Bit-identical either way; the witness phase is ALU-bound so the
/// uop cut matters on the ranked runner. The EVEX form needs `avx512f`.
#[inline(always)]
fn ror_v8<const N: i32>(v: V8) -> V8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // SAFETY: target feature checked at compile time.
        unsafe { _mm256_ror_epi32::<N>(v) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        // N is a monomorphized literal (16/12/8/7 in the G-functions), so the
        // match folds to the single shift+or pair at compile time.
        match N {
            16 => or_v8(shr_v8::<16>(v), shl_v8::<16>(v)),
            12 => or_v8(shr_v8::<12>(v), shl_v8::<20>(v)),
            8 => or_v8(shr_v8::<8>(v), shl_v8::<24>(v)),
            7 => or_v8(shr_v8::<7>(v), shl_v8::<25>(v)),
            _ => panic!("unexpected rotate count {N}"),
        }
    }
}

/// Drain 8 consecutive stage words (`dump` chunk `g`) to eight row-major
/// 32-byte block runs. Temporal stores only.
#[inline(always)]
unsafe fn dump_range(stage: *const V8, dst: *mut u32, g0: usize, g1: usize) {
    unsafe {
        for g in g0..g1 {
            let w = 8 * g;
            let r0 = load_v8(stage.add(w) as *const u32);
            let r1 = load_v8(stage.add(w + 1) as *const u32);
            let r2 = load_v8(stage.add(w + 2) as *const u32);
            let r3 = load_v8(stage.add(w + 3) as *const u32);
            let r4 = load_v8(stage.add(w + 4) as *const u32);
            let r5 = load_v8(stage.add(w + 5) as *const u32);
            let r6 = load_v8(stage.add(w + 6) as *const u32);
            let r7 = load_v8(stage.add(w + 7) as *const u32);
            let t = tr8(r0, r1, r2, r3, r4, r5, r6, r7);
            store_v8(dst.add(w), t[0]);
            store_v8(dst.add(U32_PER_BLOCK + w), t[1]);
            store_v8(dst.add(2 * U32_PER_BLOCK + w), t[2]);
            store_v8(dst.add(3 * U32_PER_BLOCK + w), t[3]);
            store_v8(dst.add(4 * U32_PER_BLOCK + w), t[4]);
            store_v8(dst.add(5 * U32_PER_BLOCK + w), t[5]);
            store_v8(dst.add(6 * U32_PER_BLOCK + w), t[6]);
            store_v8(dst.add(7 * U32_PER_BLOCK + w), t[7]);
        }
    }
}

/// `FLOCK_NO_WIDE_NT=1` restores XMM-only streaming stores in [`dump_range_nt`].
fn wide_nt_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_WIDE_NT").is_none());
    *ON
}

/// `FLOCK_NO_SPREAD_NT=1` restores the bunched drain step (three eight-row
/// runs of streaming stores, then the whole window's projection).
fn spread_nt_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_SPREAD_NT").is_none());
    *ON
}

/// Non-temporal twin of [`dump_range`]: identical bytes. Recyclable-class
/// destinations are 64-aligned on this lineage, and `U32_PER_BLOCK = 512`
/// keeps every row start 64-aligned too, so a pair of 32-byte V8s is one
/// cache line. Publish that line with a single ZMM stream when `avx512f` is
/// compiled in, otherwise one YMM stream per V8. `FLOCK_NO_WIDE_NT=1` keeps
/// the historical two-XMM form. Chunks still drain in PAIRS so each line's
/// write-combining buffer closes as soon as it fills.
///
/// Caller contract: destinations are not read again until after an
/// `_mm_sfence()` on this thread (the witness task issues one per rayon
/// task; same-thread reads are self-consistent regardless).
#[inline(always)]
unsafe fn dump_range_nt(stage: *const V8, dst: *mut u32, g0: usize, g1: usize, wide_nt: bool) {
    unsafe {
        debug_assert_eq!(dst as usize % 16, 0);
        let mut g = g0;
        while g + 2 <= g1 {
            let w = 8 * g;
            let ta = tr8_chunk(stage, w);
            let tb = tr8_chunk(stage, w + 8);
            for r in 0..8 {
                stream_pair_v8(dst.add(r * U32_PER_BLOCK + w), ta[r], tb[r], wide_nt);
            }
            g += 2;
        }
        if g < g1 {
            let w = 8 * g;
            let t = tr8_chunk(stage, w);
            for r in 0..8 {
                stream_v8(dst.add(r * U32_PER_BLOCK + w), t[r], wide_nt);
            }
        }
    }
}

/// Transpose the eight stage words at `w` (one `dump` chunk) into eight
/// row-major 32-byte runs.
#[inline(always)]
unsafe fn tr8_chunk(stage: *const V8, w: usize) -> [V8; 8] {
    unsafe {
        tr8(
            load_v8(stage.add(w) as *const u32),
            load_v8(stage.add(w + 1) as *const u32),
            load_v8(stage.add(w + 2) as *const u32),
            load_v8(stage.add(w + 3) as *const u32),
            load_v8(stage.add(w + 4) as *const u32),
            load_v8(stage.add(w + 5) as *const u32),
            load_v8(stage.add(w + 6) as *const u32),
            load_v8(stage.add(w + 7) as *const u32),
        )
    }
}

/// Publish one 32-byte transposed run non-temporally.
///
/// # Safety
/// Caller guarantees 16-byte alignment of `p`; the YMM arm additionally
/// requires 32-byte alignment (true for every in-loop pointer when `dst` is
/// 32-aligned and `w` is a multiple of 8).
#[inline(always)]
unsafe fn stream_v8(p: *mut u32, v: V8, wide_nt: bool) {
    unsafe {
        if wide_nt && p as usize % 32 == 0 {
            _mm256_stream_si256(p.cast::<__m256i>(), v);
            return;
        }
        _mm_stream_si128(p.cast::<__m128i>(), _mm256_castsi256_si128(v));
        _mm_stream_si128(p.add(4).cast::<__m128i>(), _mm256_extracti128_si256::<1>(v));
    }
}

/// Publish a chunk PAIR — two consecutive 32-byte runs, i.e. one 64-byte
/// cache line when `p` is line-aligned — non-temporally, closing the line's
/// write-combining buffer in one shot where the ISA allows it.
///
/// # Safety
/// Same alignment contract as [`stream_v8`], for both `p` and `p.add(8)`.
#[inline(always)]
unsafe fn stream_pair_v8(p: *mut u32, va: V8, vb: V8, wide_nt: bool) {
    unsafe {
        #[cfg(target_feature = "avx512f")]
        if wide_nt && p as usize % 64 == 0 {
            let z = _mm512_castsi256_si512(va);
            let z = _mm512_inserti64x4::<1>(z, vb);
            _mm512_stream_si512(p.cast::<__m512i>(), z);
            return;
        }
        stream_v8(p, va, wide_nt);
        stream_v8(p.add(8), vb, wide_nt);
    }
}

/// Publish one row of a drain step to `p` (its low 32-byte run) and `p+8`
/// (its high run), under the same `nt`/liveness policy — and therefore with
/// the same bytes at the same addresses — as [`Drain8::publish_step`]'s body.
///
/// # Safety
/// As for [`stream_pair_v8`] and [`store_v8`].
#[inline(always)]
unsafe fn emit_pair(
    p: *mut u32,
    lo: V8,
    hi: V8,
    lo_live: bool,
    hi_live: bool,
    nt: bool,
    wide_nt: bool,
) {
    unsafe {
        match (nt, lo_live, hi_live) {
            (true, true, true) => stream_pair_v8(p, lo, hi, wide_nt),
            (true, true, false) => stream_v8(p, lo, wide_nt),
            (true, false, true) => stream_v8(p.add(8), hi, wide_nt),
            (false, true, true) => {
                store_v8(p, lo);
                store_v8(p.add(8), hi);
            }
            (false, true, false) => store_v8(p, lo),
            (false, false, true) => store_v8(p.add(8), hi),
            (_, false, false) => {}
        }
    }
}

/// One drain step's three destination rows, staged for publication BETWEEN
/// block transforms by [`StreamProj::project_blocks`].
///
/// Nothing here is kept in registers across the transform: a's and b's bytes
/// are already in the projection's own staging (they are its input), and z's
/// sixteen transposed rows are in the drain's frame, which is where the
/// bunched arm spills them anyway. The publisher therefore re-reads each row
/// from L1 and the spread costs no extra spill traffic.
#[derive(Clone, Copy)]
struct StepRows {
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    z_lo: *const V8,
    z_hi: *const V8,
    /// bit 0 z-lo, 1 z-hi, 2 a-lo, 3 a-hi, 4 b-lo, 5 b-hi live;
    /// bit 6 z non-temporal, bit 7 wide streaming stores.
    flags: u8,
}

impl StepRows {
    /// Publish block `j`'s three rows.
    ///
    /// # Safety
    /// `self` describes one live drain step and `j < 8`.
    #[inline(always)]
    unsafe fn publish(&self, j: usize, sa: *const u32, sb: *const u32) {
        unsafe {
            let f = self.flags;
            let wide = f & 0x80 != 0;
            let o = j * U32_PER_BLOCK;
            emit_pair(
                self.z.add(o),
                *self.z_lo.add(j),
                *self.z_hi.add(j),
                f & 1 != 0,
                f & 2 != 0,
                f & 0x40 != 0,
                wide,
            );
            let p = sa.add(j * STEP_WORDS);
            emit_pair(
                self.a.add(o),
                load_v8(p),
                load_v8(p.add(8)),
                f & 4 != 0,
                f & 8 != 0,
                true,
                wide,
            );
            let p = sb.add(j * STEP_WORDS);
            emit_pair(
                self.b.add(o),
                load_v8(p),
                load_v8(p.add(8)),
                f & 0x10 != 0,
                f & 0x20 != 0,
                true,
                wide,
            );
        }
    }
}

impl Drain8<'_> {
    /// Transpose one 16-word drain step of the current ring epoch and publish
    /// it. `carry`, when present, additionally receives all sixteen words of
    /// every block at `(base, row_stride)` — even where the recyclable main
    /// destination elides a constant range.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn publish_step(
        stage: *const V8,
        dst: *mut u32,
        carry: Option<(*mut u32, usize)>,
        abs_word: usize,
        ring_word: usize,
        g0: usize,
        g1: usize,
        nt: bool,
        wide_nt: bool,
    ) {
        unsafe {
            let lo_rows = tr8_chunk(stage, ring_word);
            let hi_rows = tr8_chunk(stage, ring_word + 8);
            let g = abs_word / 8;
            let lo_live = g >= g0 && g < g1;
            let hi_live = g + 1 >= g0 && g + 1 < g1;
            for r in 0..8 {
                if let Some((base, row_stride)) = carry {
                    let p = base.add(r * row_stride);
                    store_v8(p, lo_rows[r]);
                    store_v8(p.add(8), hi_rows[r]);
                }
                let p = dst.add(r * U32_PER_BLOCK + abs_word);
                match (nt, lo_live, hi_live) {
                    (true, true, true) => stream_pair_v8(p, lo_rows[r], hi_rows[r], wide_nt),
                    (true, true, false) => stream_v8(p, lo_rows[r], wide_nt),
                    (true, false, true) => stream_v8(p.add(8), hi_rows[r], wide_nt),
                    (false, true, true) => {
                        store_v8(p, lo_rows[r]);
                        store_v8(p.add(8), hi_rows[r]);
                    }
                    (false, true, false) => store_v8(p, lo_rows[r]),
                    (false, false, true) => store_v8(p.add(8), hi_rows[r]),
                    (_, false, false) => {}
                }
            }
        }
    }

    /// Publish one B step and derive the matching Z rows from the already
    /// transposed A rows. Every R1CS row satisfies `z = a & b`; `tr8` is a
    /// bit permutation, so the identity is unchanged after transposition.
    ///
    /// `a_rows` is the row-major carry written by the immediately preceding
    /// A `publish_step`. Keeping only B's sixteen results live preserves the
    /// paired 64-byte NT stores without holding all A and B rows in registers.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn publish_b_with_derived_z(
        b_stage: *const V8,
        b_dst: *mut u32,
        b_carry: Option<(*mut u32, usize)>,
        a_rows: (*const u32, usize),
        z_dst: *mut u32,
        abs_word: usize,
        ring_word: usize,
        z_g1: usize,
        b_g0: usize,
        b_g1: usize,
        b_nt: bool,
        z_nt: bool,
        wide_nt: bool,
    ) {
        unsafe {
            let b_lo_rows = tr8_chunk(b_stage, ring_word);
            let b_hi_rows = tr8_chunk(b_stage, ring_word + 8);
            let g = abs_word / 8;
            let z_lo_live = g < z_g1;
            let z_hi_live = g + 1 < z_g1;
            let b_lo_live = g >= b_g0 && g < b_g1;
            let b_hi_live = g + 1 >= b_g0 && g + 1 < b_g1;

            for r in 0..8 {
                let ap = a_rows.0.add(r * a_rows.1);
                let a_lo = load_v8(ap);
                let a_hi = load_v8(ap.add(8));
                let z_lo = and_v8(a_lo, b_lo_rows[r]);
                let z_hi = and_v8(a_hi, b_hi_rows[r]);

                if let Some((base, row_stride)) = b_carry {
                    let p = base.add(r * row_stride);
                    store_v8(p, b_lo_rows[r]);
                    store_v8(p.add(8), b_hi_rows[r]);
                }

                let bp = b_dst.add(r * U32_PER_BLOCK + abs_word);
                match (b_nt, b_lo_live, b_hi_live) {
                    (true, true, true) => stream_pair_v8(bp, b_lo_rows[r], b_hi_rows[r], wide_nt),
                    (true, true, false) => stream_v8(bp, b_lo_rows[r], wide_nt),
                    (true, false, true) => stream_v8(bp.add(8), b_hi_rows[r], wide_nt),
                    (false, true, true) => {
                        store_v8(bp, b_lo_rows[r]);
                        store_v8(bp.add(8), b_hi_rows[r]);
                    }
                    (false, true, false) => store_v8(bp, b_lo_rows[r]),
                    (false, false, true) => store_v8(bp.add(8), b_hi_rows[r]),
                    (_, false, false) => {}
                }

                let zp = z_dst.add(r * U32_PER_BLOCK + abs_word);
                match (z_nt, z_lo_live, z_hi_live) {
                    (true, true, true) => stream_pair_v8(zp, z_lo, z_hi, wide_nt),
                    (true, true, false) => stream_v8(zp, z_lo, wide_nt),
                    (true, false, true) => stream_v8(zp.add(8), z_hi, wide_nt),
                    (false, true, true) => {
                        store_v8(zp, z_lo);
                        store_v8(zp.add(8), z_hi);
                    }
                    (false, true, false) => store_v8(zp, z_lo),
                    (false, false, true) => store_v8(zp.add(8), z_hi),
                    (_, false, false) => {}
                }
            }
        }
    }

    #[inline(always)]
    fn ab_ranges(&self) -> (usize, usize, usize) {
        let a_g1 = if self.elide[1] {
            ELIDE_ZERO_CHUNK
        } else {
            DUMP_CHUNKS
        };
        let b_g0 = if self.elide[2] {
            ELIDE_B_PREFIX_CHUNKS
        } else {
            0
        };
        let b_g1 = if !self.elide[2] {
            DUMP_CHUNKS
        } else if self.win_ab.is_some() || self.proj.is_some() {
            ELIDE_B_TAIL_CHUNK_WIN
        } else {
            ELIDE_B_TAIL_CHUNK
        };
        (a_g1, b_g0, b_g1)
    }

    #[inline(never)]
    unsafe fn drain_range(&mut self, base_word: usize, ring_word: usize, words: usize) {
        unsafe {
            if let Some(proj) = &self.proj {
                // The live schedule. Everything below it is the bunched
                // incumbent, kept for `FLOCK_NO_SPREAD_NT=1`.
                if self.spread {
                    self.drain_range_spread(proj, base_word, ring_word, words);
                    return;
                }
                let z_g1 = if self.elide[0] {
                    ELIDE_ZERO_CHUNK
                } else {
                    DUMP_CHUNKS
                };
                let (a_g1, b_g0, b_g1) = self.ab_ranges();
                let (sa, sb) = proj.sides();
                for off in (0..words).step_by(STEP_WORDS) {
                    let abs_word = base_word + off;
                    let rw = ring_word + off;
                    Self::publish_step(
                        self.ast,
                        self.a,
                        Some((sa, STEP_WORDS)),
                        abs_word,
                        rw,
                        0,
                        a_g1,
                        true,
                        self.wide_nt,
                    );
                    Self::publish_b_with_derived_z(
                        self.bs,
                        self.b,
                        Some((sb, STEP_WORDS)),
                        (sa, STEP_WORDS),
                        self.z,
                        abs_word,
                        rw,
                        z_g1,
                        b_g0,
                        b_g1,
                        true,
                        self.z_nt,
                        self.wide_nt,
                    );
                    proj.project(abs_word / STEP_WORDS);
                }
                return;
            }
            self.drain_range_buffered(base_word, ring_word, words);
        }
    }

    /// Same bytes as the bunched arm of [`Self::drain_range`], published on a
    /// different SCHEDULE.
    ///
    /// The bunched form emits all twenty-four of a step's 64-byte streaming
    /// stores back to back — eight rows 2048 bytes apart, three destinations —
    /// and then runs ~1000 core-cycles of projection with none. Twenty-four
    /// lines in ~70 core-cycles is ~8x this core's sustained non-temporal
    /// write rate, so the write-combining buffers saturate, the streaming
    /// store at the head of the store buffer cannot retire, and allocation
    /// stalls behind it (`resource_stalls.sb`).
    ///
    /// Here nothing is published until the staging is complete, and then the
    /// window's eight block transforms publish their OWN three rows as they
    /// go. Identical addresses, identical bytes, three lines per block instead
    /// of twenty-four per step — and still exactly one call per step, so the
    /// spread costs no extra call or spill traffic.
    #[inline(never)]
    unsafe fn drain_range_spread(
        &self,
        proj: &StreamProj<'_>,
        base_word: usize,
        ring_word: usize,
        words: usize,
    ) {
        unsafe {
            let z_g1 = if self.elide[0] {
                ELIDE_ZERO_CHUNK
            } else {
                DUMP_CHUNKS
            };
            let (a_g1, b_g0, b_g1) = self.ab_ranges();
            let (sa, sb) = proj.sides();
            let mut base = 0u8;
            if self.z_nt {
                base |= 0x40;
            }
            if self.wide_nt {
                base |= 0x80;
            }
            for off in (0..words).step_by(STEP_WORDS) {
                let abs_word = base_word + off;
                let rw = ring_word + off;
                let blk = abs_word / STEP_WORDS;
                let (plan, imgs) = proj.window_prep(blk);
                // Fused offset arena: while the transposed rows are still in
                // registers, also widen them to the split kernel's pre-scaled
                // `u16` offsets. The kernel then consumes the arena and never
                // reloads the staging, and no consuming load executes in the
                // shadow of its own offset stores.
                let use_off = plan.offsets_eligible(blk);
                #[repr(align(64))]
                struct OffArena([u16; 8 * ROUND1_AB_OFF_WORDS]);
                let mut arena = core::mem::MaybeUninit::<OffArena>::uninit();
                let op = core::ptr::addr_of_mut!((*arena.as_mut_ptr()).0) as *mut u16;

                // Transpose the two ring sides first. a and b go straight
                // into the projection's staging — which is both its input AND
                // the bytes the main buffers get, so nothing is published yet.
                // Every packed row satisfies `z = a & b` and `tr8` is a bit
                // permutation, so the Z rows are derived from the transposed
                // A/B rows instead of transposing a third Z ring.
                let a_lo = tr8_chunk(self.ast, rw);
                let a_hi = tr8_chunk(self.ast, rw + 8);
                for r in 0..8 {
                    let p = sa.add(r * STEP_WORDS);
                    store_v8(p, a_lo[r]);
                    store_v8(p.add(8), a_hi[r]);
                    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
                    if use_off {
                        widen_off_half(a_lo[r], a_hi[r], op.add(r * ROUND1_AB_OFF_WORDS));
                    }
                }
                let b_lo = tr8_chunk(self.bs, rw);
                let b_hi = tr8_chunk(self.bs, rw + 8);
                for r in 0..8 {
                    let p = sb.add(r * STEP_WORDS);
                    store_v8(p, b_lo[r]);
                    store_v8(p.add(8), b_hi[r]);
                    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
                    if use_off {
                        widen_off_half(b_lo[r], b_hi[r], op.add(r * ROUND1_AB_OFF_WORDS + 64));
                    }
                }
                let z_lo: [V8; 8] = core::array::from_fn(|r| and_v8(a_lo[r], b_lo[r]));
                let z_hi: [V8; 8] = core::array::from_fn(|r| and_v8(a_hi[r], b_hi[r]));

                let g = abs_word / 8;
                let mut flags = base;
                if g < z_g1 {
                    flags |= 1;
                }
                if g + 1 < z_g1 {
                    flags |= 2;
                }
                if g < a_g1 {
                    flags |= 4;
                }
                if g + 1 < a_g1 {
                    flags |= 8;
                }
                if g >= b_g0 && g < b_g1 {
                    flags |= 0x10;
                }
                if g + 1 >= b_g0 && g + 1 < b_g1 {
                    flags |= 0x20;
                }
                let rows = StepRows {
                    z: self.z.add(abs_word),
                    a: self.a.add(abs_word),
                    b: self.b.add(abs_word),
                    z_lo: z_lo.as_ptr(),
                    z_hi: z_hi.as_ptr(),
                    flags,
                };

                proj.project_blocks(
                    blk,
                    plan,
                    imgs,
                    Some(rows),
                    if use_off {
                        Some(op as *const u16)
                    } else {
                        None
                    },
                );
            }
        }
    }

    /// The two non-streaming drains — the `win_ab` window fusion and the plain
    /// temporal publish. Out of line so the streaming drain above keeps a
    /// register allocation of its own.
    #[inline(never)]
    unsafe fn drain_range_buffered(&mut self, base_word: usize, ring_word: usize, words: usize) {
        unsafe {
            let z_g1 = if self.elide[0] {
                ELIDE_ZERO_CHUNK
            } else {
                DUMP_CHUNKS
            };
            let (a_g1, b_g0, b_g1) = self.ab_ranges();
            match self.win_ab {
                Some((win_a, win_b)) => {
                    for off in (0..words).step_by(STEP_WORDS) {
                        let abs_word = base_word + off;
                        let rw = ring_word + off;
                        Self::publish_step(
                            self.ast,
                            self.a,
                            Some((win_a.add(abs_word), U32_PER_BLOCK)),
                            abs_word,
                            rw,
                            0,
                            a_g1,
                            true,
                            self.wide_nt,
                        );
                        Self::publish_b_with_derived_z(
                            self.bs,
                            self.b,
                            Some((win_b.add(abs_word), U32_PER_BLOCK)),
                            (win_a.add(abs_word), U32_PER_BLOCK),
                            self.z,
                            abs_word,
                            rw,
                            z_g1,
                            b_g0,
                            b_g1,
                            true,
                            self.z_nt,
                            self.wide_nt,
                        );
                    }
                }
                None => {
                    let mut a_rows = core::mem::MaybeUninit::<[V8; STEP_WORDS]>::uninit();
                    let a_rows = a_rows.as_mut_ptr().cast::<u32>();
                    for off in (0..words).step_by(STEP_WORDS) {
                        let abs_word = base_word + off;
                        let rw = ring_word + off;
                        Self::publish_step(
                            self.ast,
                            self.a,
                            Some((a_rows, STEP_WORDS)),
                            abs_word,
                            rw,
                            0,
                            a_g1,
                            false,
                            self.wide_nt,
                        );
                        Self::publish_b_with_derived_z(
                            self.bs,
                            self.b,
                            None,
                            (a_rows, STEP_WORDS),
                            self.z,
                            abs_word,
                            rw,
                            z_g1,
                            b_g0,
                            b_g1,
                            false,
                            self.z_nt,
                            self.wide_nt,
                        );
                    }
                }
            }
        }
    }
}

/// Dual-destination twin of [`dump_range_nt`] for the a/b sides: one
/// transpose feeds BOTH
///  * `dst` — the main witness buffer, published NON-TEMPORALLY over the
///    un-elided chunk range `[g0, g1)`, in the same paired-chunk order (and
///    with the same bytes) `dump_range_nt(stage, dst, g0, g1)` would use; and
///  * `win` — a compact 8-block window buffer with the same row-major
///    geometry (`U32_PER_BLOCK` row stride), written TEMPORALLY over ALL of
///    `0..DUMP_CHUNKS`.
///
/// `win` deliberately ignores the elide range, so it always carries the FULL
/// `U32_PER_BLOCK` words per block. The round-1 window projection reads a
/// whole block; eliding a `dst` chunk is only legal because `dst` already
/// holds those exact constant bytes from a previous witgen (pool provenance
/// token), and those constants are *not* uniformly zero — b's elided prefix
/// is all-ones. Rebuilding every chunk into `win` rather than zero-filling
/// (or reading `dst` back) keeps the projection's input byte-identical to the
/// incumbent's for every elide setting, by construction.
///
/// # Safety
/// AVX2 required. `dst` owns 8 contiguous `U32_PER_BLOCK`-word blocks and is
/// 16-byte aligned; `win` owns 8 contiguous `U32_PER_BLOCK`-word blocks and
/// is disjoint from `dst` and from `stage`.
#[inline(always)]
unsafe fn dump_range_nt_win(
    stage: *const V8,
    dst: *mut u32,
    win: *mut u32,
    g0: usize,
    g1: usize,
    wide_nt: bool,
) {
    unsafe {
        debug_assert_eq!(dst as usize % 16, 0);
        let mut g = 0usize;
        while g < DUMP_CHUNKS {
            let w = 8 * g;
            let ta = tr8_chunk(stage, w);
            let tb = tr8_chunk(stage, w + 8);
            // Window first: plain stores to a 16 KiB L1-resident buffer, and
            // grouping them ahead of the streams keeps each row's write-
            // combining buffer open across consecutive NT stores.
            for r in 0..8 {
                let p = win.add(r * U32_PER_BLOCK + w);
                store_v8(p, ta[r]);
                store_v8(p.add(8), tb[r]);
            }
            let lo = g >= g0 && g < g1;
            let hi = g + 1 >= g0 && g + 1 < g1;
            if lo && hi {
                for r in 0..8 {
                    stream_pair_v8(dst.add(r * U32_PER_BLOCK + w), ta[r], tb[r], wide_nt);
                }
            } else if lo {
                for r in 0..8 {
                    stream_v8(dst.add(r * U32_PER_BLOCK + w), ta[r], wide_nt);
                }
            } else if hi {
                for r in 0..8 {
                    stream_v8(dst.add(r * U32_PER_BLOCK + w + 8), tb[r], wide_nt);
                }
            }
            g += 2;
        }
    }
}

#[inline(always)]
unsafe fn dump_elide(
    stage: *const V8,
    dst: *mut u32,
    elide_tail: bool,
    elide_prefix: bool,
    tail_chunk: usize,
    nt: bool,
    wide_nt: bool,
) {
    let g0 = if elide_prefix {
        ELIDE_B_PREFIX_CHUNKS
    } else {
        0
    };
    let g1 = if elide_tail { tail_chunk } else { DUMP_CHUNKS };
    unsafe {
        if nt {
            dump_range_nt(stage, dst, g0, g1, wide_nt)
        } else {
            dump_range(stage, dst, g0, g1)
        }
    }
}

/// [`dump_elide`]'s dual-destination form: same elide range selection, always
/// non-temporal into `dst`, always FULL into `win`. See [`dump_range_nt_win`].
#[inline(always)]
unsafe fn dump_elide_win(
    stage: *const V8,
    dst: *mut u32,
    win: *mut u32,
    elide_tail: bool,
    elide_prefix: bool,
    tail_chunk: usize,
    wide_nt: bool,
) {
    let g0 = if elide_prefix {
        ELIDE_B_PREFIX_CHUNKS
    } else {
        0
    };
    let g1 = if elide_tail { tail_chunk } else { DUMP_CHUNKS };
    unsafe { dump_range_nt_win(stage, dst, win, g0, g1, wide_nt) }
}

/// Build `(z, a, b)` for EIGHT compressions in u32-lane lockstep.
/// Bit-exact with two 4-wide quads and with the scalar driver ×8.
///
/// `win_ab = Some((win_a, win_b))` selects the FUSED a/b drain: the main a/b
/// buffers are published non-temporally and a full copy of this octa's 8
/// blocks lands in the two window buffers, which the caller projects from
/// instead of re-reading a/b. `None` is the incumbent temporal drain.
///
/// `proj = Some(..)` selects the STREAMING form of the same fusion: a/b are
/// published non-temporally exactly as under `win_ab`, and each drain step's
/// eight 64-byte round-1 medium windows are transformed straight into the
/// caller's ab_inner blocks out of a small staging pair, so no full-block
/// window buffer exists. `win_ab` and `proj` are mutually exclusive.
///
/// # Safety
/// Caller must have AVX2. `z`/`a`/`b` each own 8 contiguous 512-word blocks.
/// When `win_ab` is `Some`, both window pointers own 8 contiguous 512-word
/// blocks too, disjoint from each other and from `z`/`a`/`b`. When `proj` is
/// `Some`, its staging and `out` satisfy [`StreamProj`]'s contract. In every
/// non-temporal arm the caller must `_mm_sfence()` on this thread after its
/// last octa, before releasing a/b to another thread (same-thread reads are
/// self-consistent regardless).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn build_octa_witness_ab_stream_elide(
    inputs: OctaInputs<'_>,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    win_ab: Option<(*mut u32, *mut u32)>,
    proj: Option<StreamProj<'_>>,
    elide: [bool; 3],
    z_nt: bool,
) {
    unsafe {
        let prepared = match inputs {
            OctaInputs::Blocks(inputs) => {
                let ptrs = [
                    inputs[0].0.as_ptr(),
                    inputs[1].0.as_ptr(),
                    inputs[2].0.as_ptr(),
                    inputs[3].0.as_ptr(),
                    inputs[4].0.as_ptr(),
                    inputs[5].0.as_ptr(),
                    inputs[6].0.as_ptr(),
                    inputs[7].0.as_ptr(),
                ];
                let cv_rows = [
                    load_v8(ptrs[0]),
                    load_v8(ptrs[1]),
                    load_v8(ptrs[2]),
                    load_v8(ptrs[3]),
                    load_v8(ptrs[4]),
                    load_v8(ptrs[5]),
                    load_v8(ptrs[6]),
                    load_v8(ptrs[7]),
                ];
                let cv = tr8(
                    cv_rows[0], cv_rows[1], cv_rows[2], cv_rows[3], cv_rows[4], cv_rows[5],
                    cv_rows[6], cv_rows[7],
                );

                let mptrs = [
                    inputs[0].1.as_ptr(),
                    inputs[1].1.as_ptr(),
                    inputs[2].1.as_ptr(),
                    inputs[3].1.as_ptr(),
                    inputs[4].1.as_ptr(),
                    inputs[5].1.as_ptr(),
                    inputs[6].1.as_ptr(),
                    inputs[7].1.as_ptr(),
                ];
                let m_lo = tr8(
                    load_v8(mptrs[0]),
                    load_v8(mptrs[1]),
                    load_v8(mptrs[2]),
                    load_v8(mptrs[3]),
                    load_v8(mptrs[4]),
                    load_v8(mptrs[5]),
                    load_v8(mptrs[6]),
                    load_v8(mptrs[7]),
                );
                let m_hi = tr8(
                    load_v8(mptrs[0].add(8)),
                    load_v8(mptrs[1].add(8)),
                    load_v8(mptrs[2].add(8)),
                    load_v8(mptrs[3].add(8)),
                    load_v8(mptrs[4].add(8)),
                    load_v8(mptrs[5].add(8)),
                    load_v8(mptrs[6].add(8)),
                    load_v8(mptrs[7].add(8)),
                );
                let mut message = [dup_u32(0); 16];
                message[..8].copy_from_slice(&m_lo);
                message[8..].copy_from_slice(&m_hi);

                let mut tlo_a = [0u32; 8];
                let mut thi_a = [0u32; 8];
                let mut bl_a = [0u32; 8];
                let mut fl_a = [0u32; 8];
                for j in 0..8 {
                    tlo_a[j] = inputs[j].2 as u32;
                    thi_a[j] = (inputs[j].2 >> 32) as u32;
                    bl_a[j] = inputs[j].3;
                    fl_a[j] = inputs[j].4;
                }
                PreparedInputs {
                    cv,
                    message,
                    counter_lo: load_v8(tlo_a.as_ptr()),
                    counter_hi: load_v8(thi_a.as_ptr()),
                    block_len: load_v8(bl_a.as_ptr()),
                    flags: load_v8(fl_a.as_ptr()),
                }
            }
            OctaInputs::Closed { init, base } => prepare_closed_inputs(init, base),
        };
        let cv_v = prepared.cv;
        let m = prepared.message;
        let tlo = prepared.counter_lo;
        let thi = prepared.counter_hi;
        let blen = prepared.block_len;
        let flags = prepared.flags;

        let mut state: [V8; 16] = [
            cv_v[0],
            cv_v[1],
            cv_v[2],
            cv_v[3],
            cv_v[4],
            cv_v[5],
            cv_v[6],
            cv_v[7],
            dup_u32(BLAKE3_IV[0]),
            dup_u32(BLAKE3_IV[1]),
            dup_u32(BLAKE3_IV[2]),
            dup_u32(BLAKE3_IV[3]),
            tlo,
            thi,
            blen,
            flags,
        ];

        let zero = dup_u32(0);
        let mut ast = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let mut bs = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let ast = ast.as_mut_ptr().cast::<V8>();
        let bs = bs.as_mut_ptr().cast::<V8>();

        let mut drain = Drain8 {
            ast,
            bs,
            z,
            a,
            b,
            win_ab,
            proj,
            elide,
            z_nt,
            wide_nt: wide_nt_enabled(),
            spread: spread_nt_enabled(),
        };
        let maxv = dup_u32(u32::MAX);
        let one = dup_u32(1);
        let chain: [V8; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15], tlo, thi, blen, flags,
        ];
        // Words 16..35 are available before the rounds. Retain them in the
        // rolling epochs; the writer publishes each epoch when it completes
        // that epoch's last word.
        for k in 0..20usize {
            let v = if k == 0 {
                or_v8(one, shl_v8::<1>(chain[0]))
            } else {
                or_v8(shr_v8::<31>(chain[k - 1]), shl_v8::<1>(chain[k]))
            };
            let w = 16 + k;
            store_v8(ast.add(w & (RING_WORDS - 1)) as *mut u32, v);
            store_v8(bs.add(w & (RING_WORDS - 1)) as *mut u32, maxv);
        }

        // The round stream starts at word `PROLOGUE_WORDS + 16`, so a first
        // epoch that ends inside the prologue never reaches the writers'
        // epoch boundary — publish it here instead. Constant-folded away for
        // every ring long enough to reach the round stream.
        if RING_WORDS <= PROLOGUE_WORDS + 16 {
            drain.drain_range(16, 16, RING_WORDS - 16);
        }

        let pending_bit = packer_initial_bit(shr_v8::<31>(flags));
        let drain_ptr = &mut drain as *mut Drain8;
        let mut wa = W8::<false>::at(ast, pending_bit, drain_ptr);
        // B is pushed after A at every site; it alone triggers a band drain
        // once both rings contain the completed word. Z is derived there.
        let mut wb = W8::<true>::at(bs, packer_initial_bit(one), drain_ptr);

        macro_rules! g {
            ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
             $mx:literal, $my:literal) => {{
                let (t0, l0, r0, _) = add_carry_parts_v8(state[$la], state[$lb]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, l0);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C0, 31, r0);
                let (a1, l1, r1, _) = add_carry_parts_v8(t0, m[$mx]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, l1);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C1, 31, r1);
                let d1 = xor_rotr8::<16, 16>(state[$ld], a1);
                let (c1s, l2, r2, _) = add_carry_parts_v8(state[$lc], d1);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, l2);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C2, 31, r2);
                let b1 = xor_rotr8::<12, 20>(state[$lb], c1s);
                let (t1, l3, r3, _) = add_carry_parts_v8(a1, b1);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, l3);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C3, 31, r3);
                let (a2, l4, r4, _) = add_carry_parts_v8(t1, m[$my]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, l4);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C4, 31, r4);
                let d2 = xor_rotr8::<8, 24>(d1, a2);
                let (c2s, l5, r5, _) = add_carry_parts_v8(c1s, d2);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, l5);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C5, 31, r5);
                let bn = xor_rotr8::<7, 25>(b1, c2s);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, maxv);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, maxv);
                state[$la] = a2;
                state[$lb] = bn;
                state[$lc] = c2s;
                state[$ld] = d2;
            }};
        }
        macro_rules! round {
            ($gb:literal, $m0:literal, $m1:literal, $m2:literal, $m3:literal,
             $m4:literal, $m5:literal, $m6:literal, $m7:literal,
             $m8:literal, $m9:literal, $m10:literal, $m11:literal,
             $m12:literal, $m13:literal, $m14:literal, $m15:literal) => {{
                g!($gb, 0, 4, 8, 12, $m0, $m1);
                g!($gb + 1, 1, 5, 9, 13, $m2, $m3);
                g!($gb + 2, 2, 6, 10, 14, $m4, $m5);
                g!($gb + 3, 3, 7, 11, 15, $m6, $m7);
                g!($gb + 4, 0, 5, 10, 15, $m8, $m9);
                g!($gb + 5, 1, 6, 11, 12, $m10, $m11);
                g!($gb + 6, 2, 7, 8, 13, $m12, $m13);
                g!($gb + 7, 3, 4, 9, 14, $m14, $m15);
            }};
        }
        round!(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        round!(8, 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
        round!(16, 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
        round!(24, 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
        round!(32, 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
        round!(40, 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
        round!(48, 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

        const {
            assert!(OUT_HI_BASE % 32 == 17);
        }
        macro_rules! oh {
            ($w:literal) => {{
                let hv = xor_v8(state[$w + 8], cv_v[$w]);
                pushf8!(wa, OUT_HI_BASE + 32 * $w, 32, hv);
                pushf8!(wb, OUT_HI_BASE + 32 * $w, 32, maxv);
            }};
        }
        oh!(0);
        oh!(1);
        oh!(2);
        oh!(3);
        oh!(4);
        oh!(5);
        oh!(6);
        oh!(7);
        wa.finish();
        wb.finish();

        const ZF: usize = USEFUL_BITS.div_ceil(32);
        const {
            assert!(U32_PER_BLOCK - ZF == 30);
        }
        // finish() completed word 481. Complete the final rolling epoch with
        // the all-zero tail, then publish words 384..511 in one long sweep.
        for w in ZF..U32_PER_BLOCK {
            let i = w & (RING_WORDS - 1);
            store_v8(ast.add(i) as *mut u32, zero);
            store_v8(bs.add(i) as *mut u32, zero);
        }
        drain.drain_range(U32_PER_BLOCK - RING_WORDS, 0, RING_WORDS);

        // Band 0 is the one intentional deferral: words 0..7 are the input
        // CV, while words 8..15 depend on the final compression state. Build
        // the complete cache line now, then publish it through the exact same
        // elide/NT/window policy as every rolling band.
        for w in 0..8usize {
            let lo = xor_v8(state[w], state[w + 8]);
            store_v8(ast.add(w) as *mut u32, cv_v[w]);
            store_v8(bs.add(w) as *mut u32, maxv);
            store_v8(ast.add(8 + w) as *mut u32, lo);
            store_v8(bs.add(8 + w) as *mut u32, maxv);
        }
        drain.drain_range(0, 0, 16);
    }
}

/// Sixteen closed compressions share one AVX-512 arithmetic stream, but keep
/// the proven eight-block publication machinery.  Every completed V16 packed
/// word is split into two V8 rings; the low `Drain8` fully publishes/projects
/// before the high `Drain8` reuses the same projection staging allocation.
/// There is intentionally no 16x16 transpose or second staging pair.
#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2"
))]
mod w16_kernel {
    use super::*;

    #[inline(always)]
    fn dup(x: u32) -> V16 {
        unsafe { _mm512_set1_epi32(x as i32) }
    }

    #[inline(always)]
    fn xor(a: V16, b: V16) -> V16 {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn or(a: V16, b: V16) -> V16 {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn shl<const N: u32>(v: V16) -> V16 {
        unsafe { _mm512_slli_epi32::<N>(v) }
    }

    #[inline(always)]
    fn shr<const N: u32>(v: V16) -> V16 {
        unsafe { _mm512_srli_epi32::<N>(v) }
    }

    #[inline(always)]
    fn add_parts(x: V16, y: V16) -> (V16, V16, V16) {
        unsafe {
            let sum = _mm512_add_epi32(x, y);
            (sum, xor(sum, y), xor(sum, x))
        }
    }

    #[inline(always)]
    fn xor_rotr<const N: i32>(x: V16, y: V16) -> V16 {
        unsafe { _mm512_ror_epi32::<N>(xor(x, y)) }
    }

    #[inline(always)]
    unsafe fn store_split(lo: *mut V8, hi: *mut V8, word: usize, v: V16) {
        unsafe {
            store_v8(lo.add(word) as *mut u32, low_v8(v));
            store_v8(hi.add(word) as *mut u32, high_v8(v));
        }
    }

    struct Writer16<'t, const FLUSH: bool> {
        pending: V16,
        lo_stage: *mut V8,
        hi_stage: *mut V8,
        lo_drain: *mut Drain8<'t>,
        hi_drain: *mut Drain8<'t>,
    }

    impl<'t, const FLUSH: bool> Writer16<'t, FLUSH> {
        #[inline(always)]
        fn at(
            lo_stage: *mut V8,
            hi_stage: *mut V8,
            pending: V16,
            lo_drain: *mut Drain8<'t>,
            hi_drain: *mut Drain8<'t>,
        ) -> Self {
            Self { pending, lo_stage, hi_stage, lo_drain, hi_drain }
        }

        #[inline(always)]
        unsafe fn write_word<const WORD: usize>(&mut self, v: V16) {
            unsafe {
                let rw = WORD & (RING_WORDS - 1);
                store_split(self.lo_stage, self.hi_stage, rw, v);
                if FLUSH && WORD % RING_WORDS == RING_WORDS - 1 {
                    let (base, ring, words) = if WORD + 1 == RING_WORDS {
                        (16, 16, RING_WORDS - 16)
                    } else {
                        (WORD + 1 - RING_WORDS, 0, RING_WORDS)
                    };
                    (*self.lo_drain).drain_range(base, ring, words);
                    (*self.hi_drain).drain_range(base, ring, words);
                }
            }
        }

        #[inline(always)]
        unsafe fn push<const USED: i32, const WIDTH: i32, const WORD: usize>(&mut self, v: V16) {
            const {
                assert!(USED >= 0 && USED < 32);
                assert!(WIDTH == 31 || WIDTH == 32);
                assert!(WORD < U32_PER_BLOCK);
            }
            unsafe {
                if USED == 0 {
                    if WIDTH == 32 {
                        self.write_word::<WORD>(v);
                        self.pending = dup(0);
                    } else {
                        self.pending = shl::<1>(v);
                    }
                } else {
                    self.write_word::<WORD>(_mm512_shldi_epi32::<USED>(v, self.pending));
                    self.pending = if WIDTH == 31 { shl::<1>(v) } else { v };
                }
            }
        }

        #[inline(always)]
        unsafe fn finish(&mut self) {
            const { assert!(USEFUL_BITS % 32 == 17) };
            unsafe { self.write_word::<LAST_WORD>(shr::<15>(self.pending)) }
        }
    }

    macro_rules! pushf16 {
        ($w:ident, $pos:expr, $width:literal, $v:expr) => {{
            $w.push::<{ ($pos % 32) as i32 }, $width, { $pos / 32 }>($v);
        }};
    }

    #[inline(always)]
    unsafe fn chain_word(p: &PreparedInputs16, k: usize) -> V16 {
        unsafe {
            if k < 16 {
                *p.message.get_unchecked(k)
            } else {
                match k {
                    16 => p.counter_lo,
                    17 => p.counter_hi,
                    18 => p.block_len,
                    19 => p.flags,
                    _ => core::hint::unreachable_unchecked(),
                }
            }
        }
    }

    /// Kept out of line so `PreparedInputs16` remains a memory operand source
    /// instead of being cloned into 24 long-lived vector locals beside state.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run<'t>(
        p: &PreparedInputs16,
        z: *mut u32,
        a: *mut u32,
        b: *mut u32,
        proj_lo: StreamProj<'t>,
        proj_hi: StreamProj<'t>,
        elide: [bool; 3],
        z_nt: bool,
    ) {
        unsafe {
            let mut state: [V16; 16] = [
                p.cv[0], p.cv[1], p.cv[2], p.cv[3], p.cv[4], p.cv[5], p.cv[6], p.cv[7],
                dup(BLAKE3_IV[0]), dup(BLAKE3_IV[1]), dup(BLAKE3_IV[2]), dup(BLAKE3_IV[3]),
                p.counter_lo, p.counter_hi, p.block_len, p.flags,
            ];
            let mut ast_lo = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
            let mut ast_hi = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
            let mut bs_lo = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
            let mut bs_hi = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
            let ast_lo = ast_lo.as_mut_ptr().cast::<V8>();
            let ast_hi = ast_hi.as_mut_ptr().cast::<V8>();
            let bs_lo = bs_lo.as_mut_ptr().cast::<V8>();
            let bs_hi = bs_hi.as_mut_ptr().cast::<V8>();
            let half_words = 8 * U32_PER_BLOCK;
            let wide_nt = wide_nt_enabled();
            let spread = spread_nt_enabled();
            let mut drain_lo = Drain8 {
                ast: ast_lo, bs: bs_lo, z, a, b, win_ab: None, proj: Some(proj_lo),
                elide, z_nt, wide_nt, spread,
            };
            let mut drain_hi = Drain8 {
                ast: ast_hi, bs: bs_hi, z: z.add(half_words), a: a.add(half_words),
                b: b.add(half_words), win_ab: None, proj: Some(proj_hi),
                elide, z_nt, wide_nt, spread,
            };

            let maxv = dup(u32::MAX);
            let one = dup(1);
            let mut prev = dup(0);
            for k in 0..20usize {
                let cur = chain_word(p, k);
                let v = if k == 0 {
                    or(one, shl::<1>(cur))
                } else {
                    or(shr::<31>(prev), shl::<1>(cur))
                };
                let w = 16 + k;
                store_split(ast_lo, ast_hi, w & (RING_WORDS - 1), v);
                store_split(bs_lo, bs_hi, w & (RING_WORDS - 1), maxv);
                prev = cur;
            }
            if RING_WORDS <= PROLOGUE_WORDS + 16 {
                drain_lo.drain_range(16, 16, RING_WORDS - 16);
                drain_hi.drain_range(16, 16, RING_WORDS - 16);
            }

            let lo_drain = &mut drain_lo as *mut Drain8;
            let hi_drain = &mut drain_hi as *mut Drain8;
            let mut wa = Writer16::<false>::at(
                ast_lo, ast_hi, shl::<31>(shr::<31>(p.flags)), lo_drain, hi_drain,
            );
            let mut wb = Writer16::<true>::at(
                bs_lo, bs_hi, shl::<31>(one), lo_drain, hi_drain,
            );

            macro_rules! g {
                ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
                 $mx:literal, $my:literal) => {{
                    let (t0, l0, r0) = add_parts(state[$la], state[$lb]);
                    pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, l0);
                    pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C0, 31, r0);
                    let (a1, l1, r1) = add_parts(t0, p.message[$mx]);
                    pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, l1);
                    pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C1, 31, r1);
                    let d1 = xor_rotr::<16>(state[$ld], a1);
                    let (c1s, l2, r2) = add_parts(state[$lc], d1);
                    pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, l2);
                    pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C2, 31, r2);
                    let b1 = xor_rotr::<12>(state[$lb], c1s);
                    let (t1, l3, r3) = add_parts(a1, b1);
                    pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, l3);
                    pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C3, 31, r3);
                    let (a2, l4, r4) = add_parts(t1, p.message[$my]);
                    pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, l4);
                    pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C4, 31, r4);
                    let d2 = xor_rotr::<8>(d1, a2);
                    let (c2s, l5, r5) = add_parts(c1s, d2);
                    pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, l5);
                    pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C5, 31, r5);
                    let bn = xor_rotr::<7>(b1, c2s);
                    pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                    pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, maxv);
                    pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
                    pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, maxv);
                    state[$la] = a2; state[$lb] = bn; state[$lc] = c2s; state[$ld] = d2;
                }};
            }
            macro_rules! round {
                ($gb:literal, $m0:literal, $m1:literal, $m2:literal, $m3:literal,
                 $m4:literal, $m5:literal, $m6:literal, $m7:literal,
                 $m8:literal, $m9:literal, $m10:literal, $m11:literal,
                 $m12:literal, $m13:literal, $m14:literal, $m15:literal) => {{
                    g!($gb, 0, 4, 8, 12, $m0, $m1); g!($gb + 1, 1, 5, 9, 13, $m2, $m3);
                    g!($gb + 2, 2, 6, 10, 14, $m4, $m5); g!($gb + 3, 3, 7, 11, 15, $m6, $m7);
                    g!($gb + 4, 0, 5, 10, 15, $m8, $m9); g!($gb + 5, 1, 6, 11, 12, $m10, $m11);
                    g!($gb + 6, 2, 7, 8, 13, $m12, $m13); g!($gb + 7, 3, 4, 9, 14, $m14, $m15);
                }};
            }
            round!(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            round!(8, 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
            round!(16, 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
            round!(24, 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
            round!(32, 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
            round!(40, 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
            round!(48, 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

            macro_rules! oh {
                ($w:literal) => {{
                    pushf16!(wa, OUT_HI_BASE + 32 * $w, 32, xor(state[$w + 8], p.cv[$w]));
                    pushf16!(wb, OUT_HI_BASE + 32 * $w, 32, maxv);
                }};
            }
            oh!(0); oh!(1); oh!(2); oh!(3); oh!(4); oh!(5); oh!(6); oh!(7);
            wa.finish();
            wb.finish();

            const ZF: usize = USEFUL_BITS.div_ceil(32);
            for w in ZF..U32_PER_BLOCK {
                let i = w & (RING_WORDS - 1);
                store_split(ast_lo, ast_hi, i, dup(0));
                store_split(bs_lo, bs_hi, i, dup(0));
            }
            drain_lo.drain_range(U32_PER_BLOCK - RING_WORDS, 0, RING_WORDS);
            drain_hi.drain_range(U32_PER_BLOCK - RING_WORDS, 0, RING_WORDS);

            for w in 0..8usize {
                store_split(ast_lo, ast_hi, w, p.cv[w]);
                store_split(bs_lo, bs_hi, w, maxv);
                store_split(ast_lo, ast_hi, 8 + w, xor(state[w], state[w + 8]));
                store_split(bs_lo, bs_hi, 8 + w, maxv);
            }
            drain_lo.drain_range(0, 0, 16);
            drain_hi.drain_range(0, 0, 16);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn build_closed_witness_ab_stream_elide(
        init: u64,
        base: usize,
        z: *mut u32,
        a: *mut u32,
        b: *mut u32,
        proj_lo: StreamProj<'_>,
        proj_hi: StreamProj<'_>,
        elide: [bool; 3],
        z_nt: bool,
    ) {
        unsafe {
            let prepared = prepare_closed_inputs16(init, base);
            run(&prepared, z, a, b, proj_lo, proj_hi, elide, z_nt);
        }
    }
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl",
    target_feature = "avx512vbmi2"
))]
pub(crate) use w16_kernel::build_closed_witness_ab_stream_elide;

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn lanes(v: V8) -> [u32; 8] {
        let mut out = [0u32; 8];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), v) };
        out
    }

    /// Compile every W8 offset/width specialization used by the witness and
    /// compare its high-aligned VBMI2 state transition with an independent
    /// scalar bitstream append.  Bit-basis inputs include the deliberately
    /// dirty bit 31 of a width-31 field, so the oracle also proves that the
    /// next field, rather than an eager mask, discards that irrelevant bit.
    #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn check_vbmi2_push<const USED: i32, const WIDTH: i32, const BACK: i32>() {
        const {
            assert!(USED >= 0 && USED < 32);
            assert!(WIDTH == 31 || WIDTH == 32);
            assert!(BACK > 0 && BACK < 32);
        }
        let pending_mask = if USED == 0 {
            0
        } else {
            ((1u64 << USED) - 1) as u32
        };
        let field_mask = if WIDTH == 32 {
            u32::MAX
        } else {
            (1u32 << WIDTH) - 1
        };
        let mut pending_cases = vec![0, pending_mask];
        for bit in 0..USED {
            pending_cases.push(1u32 << bit);
        }
        pending_cases.sort_unstable();
        pending_cases.dedup();
        let mut field_cases = vec![0, field_mask, u32::MAX];
        for bit in 0..32 {
            field_cases.push(1u32 << bit);
        }
        field_cases.sort_unstable();
        field_cases.dedup();

        for &pending_low in &pending_cases {
            for &raw_v in &field_cases {
                let v = raw_v & field_mask;
                let acc = u64::from(pending_low) | (u64::from(v) << USED);
                let total = USED + WIDTH;
                let expected_out = (total >= 32).then_some(acc as u32);
                let next_used = total % 32;
                let expected_pending = if total >= 32 {
                    (acc >> 32) as u32
                } else {
                    acc as u32
                };
                let pending_hi = if USED == 0 {
                    0
                } else {
                    pending_low << (32 - USED)
                };

                let sentinel = 0xA5A5_5A5A;
                let mut stage = [dup_u32(sentinel); 1];
                let mut writer = W8::<false>::at(
                    stage.as_mut_ptr(),
                    dup_u32(pending_hi),
                    core::ptr::null_mut::<Drain8<'static>>(),
                );
                unsafe {
                    writer.push::<USED, WIDTH, BACK, 0>(dup_u32(raw_v));
                }

                let got_out = lanes(stage[0])[0];
                assert_eq!(
                    got_out,
                    expected_out.unwrap_or(sentinel),
                    "out: used={USED} width={WIDTH} pending={pending_low:#x} v={raw_v:#x}"
                );
                let got_hi = lanes(writer.pending)[0];
                if next_used != 0 {
                    assert_eq!(
                        got_hi >> (32 - next_used),
                        expected_pending,
                        "pending: used={USED} width={WIDTH} pending={pending_low:#x} v={raw_v:#x}"
                    );
                }
            }
        }
    }

    #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn check_vbmi2_finish<const USED: i32, const BACK: i32>() {
        const {
            assert!(USED > 0 && USED < 32);
            assert!(BACK == 32 - USED);
        }
        let mask = ((1u64 << USED) - 1) as u32;
        let mut cases = vec![0, mask];
        for bit in 0..USED {
            cases.push(1u32 << bit);
        }
        for pending_low in cases {
            let pending_hi = dup_u32(pending_low << BACK);
            let got = lanes(packer_high_to_low::<BACK>(pending_hi));
            assert_eq!(got, [pending_low; 8], "finish: used={USED}");
        }
    }

    #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
    #[test]
    fn vbmi2_high_aligned_w8_matches_scalar_all_offsets() {
        unsafe {
            macro_rules! check_offset {
                ($used:literal, $back:literal) => {{
                    check_vbmi2_push::<$used, 31, $back>();
                    check_vbmi2_push::<$used, 32, $back>();
                    check_vbmi2_finish::<$used, $back>();
                }};
            }
            check_vbmi2_push::<0, 31, 1>();
            check_vbmi2_push::<0, 32, 1>();
            check_offset!(1, 31);
            check_offset!(2, 30);
            check_offset!(3, 29);
            check_offset!(4, 28);
            check_offset!(5, 27);
            check_offset!(6, 26);
            check_offset!(7, 25);
            check_offset!(8, 24);
            check_offset!(9, 23);
            check_offset!(10, 22);
            check_offset!(11, 21);
            check_offset!(12, 20);
            check_offset!(13, 19);
            check_offset!(14, 18);
            check_offset!(15, 17);
            check_offset!(16, 16);
            check_offset!(17, 15);
            check_offset!(18, 14);
            check_offset!(19, 13);
            check_offset!(20, 12);
            check_offset!(21, 11);
            check_offset!(22, 10);
            check_offset!(23, 9);
            check_offset!(24, 8);
            check_offset!(25, 7);
            check_offset!(26, 6);
            check_offset!(27, 5);
            check_offset!(28, 4);
            check_offset!(29, 3);
            check_offset!(30, 2);
            check_offset!(31, 1);

            let flags = [0, 1, 0x7FFF_FFFF, 0x8000_0000, u32::MAX, 11, 64, 9];
            let flags_v = load_v8(flags.as_ptr());
            assert_eq!(
                lanes(packer_initial_bit(shr_v8::<31>(flags_v))),
                flags.map(|v| v & 0x8000_0000),
                "A prior flag must occupy bit 31"
            );
            assert_eq!(
                lanes(packer_initial_bit(dup_u32(1))),
                [0x8000_0000; 8],
                "B prior flag must occupy bit 31"
            );

            // Exercise the production finish itself at the ranked used=17.
            let pending_low = 0x1_5A5A;
            let mut stage = [dup_u32(0); RING_WORDS];
            let mut writer = W8::<false>::at(
                stage.as_mut_ptr(),
                dup_u32(pending_low << 15),
                core::ptr::null_mut::<Drain8<'static>>(),
            );
            writer.finish();
            assert_eq!(lanes(stage[LAST_WORD & (RING_WORDS - 1)]), [pending_low; 8]);
        }
    }

    #[test]
    fn closed_octa_inputs_match_scalar_generator() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe {
            for (init, base) in [
                (0u64, 0usize),
                (0x0123_4567_89AB_CDEF, 17),
                (u64::MAX - 31, (1 << 18) - 8),
            ] {
                let got = prepare_closed_inputs(init, base);
                let cv: [[u32; 8]; 8] = got.cv.map(|v| lanes(v));
                let message: [[u32; 8]; 16] = got.message.map(|v| lanes(v));
                let counter_lo = lanes(got.counter_lo);
                let counter_hi = lanes(got.counter_hi);
                let block_len = lanes(got.block_len);
                let flags = lanes(got.flags);
                for lane in 0..8 {
                    let expected = crate::seed_pipe::gen_block(init, base + lane);
                    for word in 0..8 {
                        assert_eq!(cv[word][lane], expected.0[word]);
                    }
                    for word in 0..16 {
                        assert_eq!(message[word][lane], expected.1[word]);
                    }
                    assert_eq!(counter_lo[lane], expected.2 as u32);
                    assert_eq!(counter_hi[lane], (expected.2 >> 32) as u32);
                    assert_eq!(block_len[lane], expected.3);
                    assert_eq!(flags[lane], expected.4);
                }
            }
        }
    }

    /// Independent algebra/codegen oracle for the production Z derivation.
    /// The direct path packs `a & b` and then transposes; the new path
    /// transposes A/B separately and ANDs corresponding row vectors.
    #[test]
    fn derived_z_after_tr8_matches_direct_z() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe {
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            for case in 0..66usize {
                let mut a = [[0u32; 8]; 8];
                let mut b = [[0u32; 8]; 8];
                for i in 0..8 {
                    for j in 0..8 {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        a[i][j] = if case == 0 { u32::MAX } else { state as u32 };
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        b[i][j] = if case == 1 { u32::MAX } else { state as u32 };
                    }
                }
                let av: [V8; 8] = core::array::from_fn(|i| load_v8(a[i].as_ptr()));
                let bv: [V8; 8] = core::array::from_fn(|i| load_v8(b[i].as_ptr()));
                let zv: [V8; 8] = core::array::from_fn(|i| and_v8(av[i], bv[i]));
                let direct = tr8(zv[0], zv[1], zv[2], zv[3], zv[4], zv[5], zv[6], zv[7]);
                let at = tr8(av[0], av[1], av[2], av[3], av[4], av[5], av[6], av[7]);
                let bt = tr8(bv[0], bv[1], bv[2], bv[3], bv[4], bv[5], bv[6], bv[7]);
                for row in 0..8 {
                    let derived = and_v8(at[row], bt[row]);
                    let mut direct_words = [0u32; 8];
                    let mut derived_words = [0u32; 8];
                    store_v8(direct_words.as_mut_ptr(), direct[row]);
                    store_v8(derived_words.as_mut_ptr(), derived);
                    assert_eq!(derived_words, direct_words, "case={case} row={row}");
                }
            }
        }
    }

    #[test]
    fn witgen8_tr8_is_8x8_u32_transpose() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe { tr8_check() }
    }

    unsafe fn tr8_check() {
        unsafe {
            let rows: [V8; 8] = core::array::from_fn(|i| {
                _mm256_setr_epi32(
                    (1000 * i) as i32,
                    (1000 * i + 1) as i32,
                    (1000 * i + 2) as i32,
                    (1000 * i + 3) as i32,
                    (1000 * i + 4) as i32,
                    (1000 * i + 5) as i32,
                    (1000 * i + 6) as i32,
                    (1000 * i + 7) as i32,
                )
            });
            let t = tr8(
                rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7],
            );
            for j in 0..8 {
                let mut buf = [0i32; 8];
                _mm256_storeu_si256(buf.as_mut_ptr().cast(), t[j]);
                for i in 0..8 {
                    assert_eq!(buf[i], (1000 * i + j) as i32, "t[{j}] lane {i}");
                }
            }
            let back = tr8(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
            for i in 0..8 {
                let mut a = [0i32; 8];
                let mut b = [0i32; 8];
                _mm256_storeu_si256(a.as_mut_ptr().cast(), rows[i]);
                _mm256_storeu_si256(b.as_mut_ptr().cast(), back[i]);
                assert_eq!(a, b, "tr8² row {i}");
            }
        }
    }
}
// witfire-7 draw marker 10514
// witfire-16 draw marker 2194
// fire51: fresh draw on new bar 3852475 (rival NTT promotion)

// vbmi2draw-1: independent official timing sample of promoted fbc3001 VBMI2 W8 packer; no executable change.
// rivaldraw-28 marker 76708 on 2758bc1
// rivaldraw-37 marker 33507 on ca47b5d
