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
    Round1AbWindowPlan, round1_ab_inner_window,
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

#[inline(always)]
unsafe fn load_v8(p: *const u32) -> V8 {
    unsafe { _mm256_loadu_si256(p.cast::<__m256i>()) }
}

#[inline(always)]
unsafe fn store_v8(p: *mut u32, v: V8) {
    unsafe { _mm256_storeu_si256(p.cast::<__m256i>(), v) }
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
            let (sa, sb) = self.sides();
            let plan = self.plan.for_block(blk);
            for j in 0..8usize {
                if self.live & (1 << j) == 0 {
                    continue;
                }
                let a_win = &*sa.add(j * STEP_WORDS).cast::<[u8; 64]>();
                let b_win = &*sb.add(j * STEP_WORDS).cast::<[u8; 64]>();
                let out = &mut *self
                    .out
                    .add(j * BYTES_PER_BLOCK + blk * 64)
                    .cast::<[u8; 64]>();
                round1_ab_inner_window(a_win, b_win, out, blk, self.inv_table, plan);
            }
        }
    }
}

/// Rolling drain state shared by the three packed writers. The witness uses
/// three reusable `RING_WORDS`-word epochs instead of three full 512-word
/// stages.
struct Drain8<'t> {
    zs: *mut V8,
    ast: *mut V8,
    bs: *mut V8,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    win_ab: Option<(*mut u32, *mut u32)>,
    proj: Option<StreamProj<'t>>,
    elide: [bool; 3],
    z_nt: bool,
}

/// Lane-wise packed-word writer: 8 independent `PackedWordWriter`s.
struct W8<'t> {
    pending: V8,
    stage: *mut V8,
    drain: *mut Drain8<'t>,
    flush: bool,
}

impl<'t> W8<'t> {
    #[inline(always)]
    fn at(stage: *mut V8, pending: V8, drain: *mut Drain8<'t>, flush: bool) -> Self {
        Self {
            pending,
            stage,
            drain,
            flush,
        }
    }

    #[inline(always)]
    unsafe fn write_word<const WORD: usize>(&mut self, v: V8) {
        unsafe {
            store_v8(self.stage.add(WORD & (RING_WORDS - 1)) as *mut u32, v);
            if self.flush && WORD % RING_WORDS == RING_WORDS - 1 {
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

    #[inline(always)]
    unsafe fn finish(&mut self) {
        unsafe {
            self.write_word::<LAST_WORD>(self.pending);
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
    let sum = add_v8(x, y);
    let cin = xor3_v8(sum, x, y);
    let left = xor_v8(x, cin);
    let right = xor_v8(y, cin);
    let carry = and_v8(left, right);
    (sum, left, right, carry)
}

#[inline(always)]
fn xor_rotr8<const N: i32, const M: i32>(x: V8, y: V8) -> V8 {
    debug_assert_eq!(N + M, 32);
    let v = xor_v8(x, y);
    or_v8(shr_v8::<N>(v), shl_v8::<M>(v))
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
unsafe fn dump_range_nt(stage: *const V8, dst: *mut u32, g0: usize, g1: usize) {
    unsafe {
        debug_assert_eq!(dst as usize % 16, 0);
        let mut g = g0;
        while g + 2 <= g1 {
            let w = 8 * g;
            let ta = tr8_chunk(stage, w);
            let tb = tr8_chunk(stage, w + 8);
            for r in 0..8 {
                stream_pair_v8(dst.add(r * U32_PER_BLOCK + w), ta[r], tb[r]);
            }
            g += 2;
        }
        if g < g1 {
            let w = 8 * g;
            let t = tr8_chunk(stage, w);
            for r in 0..8 {
                stream_v8(dst.add(r * U32_PER_BLOCK + w), t[r]);
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
unsafe fn stream_v8(p: *mut u32, v: V8) {
    unsafe {
        if wide_nt_enabled() && p as usize % 32 == 0 {
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
unsafe fn stream_pair_v8(p: *mut u32, va: V8, vb: V8) {
    unsafe {
        #[cfg(target_feature = "avx512f")]
        if wide_nt_enabled() && p as usize % 64 == 0 {
            let z = _mm512_castsi256_si512(va);
            let z = _mm512_inserti64x4::<1>(z, vb);
            _mm512_stream_si512(p.cast::<__m512i>(), z);
            return;
        }
        stream_v8(p, va);
        stream_v8(p.add(8), vb);
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
                    (true, true, true) => stream_pair_v8(p, lo_rows[r], hi_rows[r]),
                    (true, true, false) => stream_v8(p, lo_rows[r]),
                    (true, false, true) => stream_v8(p.add(8), hi_rows[r]),
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

    /// Transpose and publish `words` from the current ring epoch at absolute
    /// `base_word`. `win`, when present, always receives every word even when
    /// the recyclable main destination elides a constant range.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn publish_range(
        stage: *const V8,
        dst: *mut u32,
        win: Option<*mut u32>,
        base_word: usize,
        ring_word: usize,
        words: usize,
        g0: usize,
        g1: usize,
        nt: bool,
    ) {
        unsafe {
            debug_assert_eq!(base_word % 8, 0);
            debug_assert_eq!(ring_word % 8, 0);
            debug_assert_eq!(words % 16, 0);
            for off in (0..words).step_by(STEP_WORDS) {
                let abs_word = base_word + off;
                Self::publish_step(
                    stage,
                    dst,
                    win.map(|w| (w.add(abs_word), U32_PER_BLOCK)),
                    abs_word,
                    ring_word + off,
                    g0,
                    g1,
                    nt,
                );
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
            let z_g1 = if self.elide[0] {
                ELIDE_ZERO_CHUNK
            } else {
                DUMP_CHUNKS
            };
            let (a_g1, b_g0, b_g1) = self.ab_ranges();

            if let Some(proj) = &self.proj {
                let (sa, sb) = proj.sides();
                for off in (0..words).step_by(STEP_WORDS) {
                    let abs_word = base_word + off;
                    let rw = ring_word + off;
                    Self::publish_step(
                        self.zs, self.z, None, abs_word, rw, 0, z_g1, self.z_nt,
                    );
                    Self::publish_step(
                        self.ast,
                        self.a,
                        Some((sa, STEP_WORDS)),
                        abs_word,
                        rw,
                        0,
                        a_g1,
                        true,
                    );
                    Self::publish_step(
                        self.bs,
                        self.b,
                        Some((sb, STEP_WORDS)),
                        abs_word,
                        rw,
                        b_g0,
                        b_g1,
                        true,
                    );
                    proj.project(abs_word / STEP_WORDS);
                }
                return;
            }

            Self::publish_range(
                self.zs, self.z, None, base_word, ring_word, words, 0, z_g1, self.z_nt,
            );

            match self.win_ab {
                Some((win_a, win_b)) => {
                    Self::publish_range(
                        self.ast, self.a, Some(win_a), base_word, ring_word, words, 0, a_g1, true,
                    );
                    Self::publish_range(
                        self.bs, self.b, Some(win_b), base_word, ring_word, words, b_g0, b_g1, true,
                    );
                }
                None => {
                    Self::publish_range(
                        self.ast, self.a, None, base_word, ring_word, words, 0, a_g1, false,
                    );
                    Self::publish_range(
                        self.bs, self.b, None, base_word, ring_word, words, b_g0, b_g1, false,
                    );
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
                    stream_pair_v8(dst.add(r * U32_PER_BLOCK + w), ta[r], tb[r]);
                }
            } else if lo {
                for r in 0..8 {
                    stream_v8(dst.add(r * U32_PER_BLOCK + w), ta[r]);
                }
            } else if hi {
                for r in 0..8 {
                    stream_v8(dst.add(r * U32_PER_BLOCK + w + 8), tb[r]);
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
) {
    let g0 = if elide_prefix {
        ELIDE_B_PREFIX_CHUNKS
    } else {
        0
    };
    let g1 = if elide_tail { tail_chunk } else { DUMP_CHUNKS };
    unsafe {
        if nt {
            dump_range_nt(stage, dst, g0, g1)
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
) {
    let g0 = if elide_prefix {
        ELIDE_B_PREFIX_CHUNKS
    } else {
        0
    };
    let g1 = if elide_tail { tail_chunk } else { DUMP_CHUNKS };
    unsafe { dump_range_nt_win(stage, dst, win, g0, g1) }
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
    inputs: [&Compression; 8],
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    win_ab: Option<(*mut u32, *mut u32)>,
    proj: Option<StreamProj<'_>>,
    elide: [bool; 3],
    z_nt: bool,
) {
    unsafe {
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
        let cv_v = tr8(
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
        let mut m = [dup_u32(0); 16];
        m[..8].copy_from_slice(&m_lo);
        m[8..].copy_from_slice(&m_hi);

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
        let tlo = load_v8(tlo_a.as_ptr());
        let thi = load_v8(thi_a.as_ptr());
        let blen = load_v8(bl_a.as_ptr());
        let flags = load_v8(fl_a.as_ptr());

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
        let mut zs = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let mut ast = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let mut bs = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let zs = zs.as_mut_ptr().cast::<V8>();
        let ast = ast.as_mut_ptr().cast::<V8>();
        let bs = bs.as_mut_ptr().cast::<V8>();

        let mut drain = Drain8 {
            zs,
            ast,
            bs,
            z,
            a,
            b,
            win_ab,
            proj,
            elide,
            z_nt,
        };
        let maxv = dup_u32(u32::MAX);
        let one = dup_u32(1);
        let chain: [V8; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12],
            m[13], m[14], m[15], tlo, thi, blen, flags,
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
            store_v8(zs.add(w & (RING_WORDS - 1)) as *mut u32, v);
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

        let pending_bit = shr_v8::<31>(flags);
        let drain_ptr = &mut drain as *mut Drain8;
        let mut wz = W8::at(zs, pending_bit, drain_ptr, false);
        let mut wa = W8::at(ast, pending_bit, drain_ptr, false);
        // B is pushed after z and a at every site; it alone triggers a band
        // drain once all three rings contain the completed word.
        let mut wb = W8::at(bs, one, drain_ptr, true);

        macro_rules! g {
            ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
             $mx:literal, $my:literal) => {{
                let (t0, l0, r0, c0) = add_carry_parts_v8(state[$la], state[$lb]);
                pushf8!(wz, GS_BASE + G_STRIDE * $g + REC_C0, 31, c0);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, l0);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C0, 31, r0);
                let (a1, l1, r1, c1) = add_carry_parts_v8(t0, m[$mx]);
                pushf8!(wz, GS_BASE + G_STRIDE * $g + REC_C1, 31, c1);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, l1);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C1, 31, r1);
                let d1 = xor_rotr8::<16, 16>(state[$ld], a1);
                let (c1s, l2, r2, c2) = add_carry_parts_v8(state[$lc], d1);
                pushf8!(wz, GS_BASE + G_STRIDE * $g + REC_C2, 31, c2);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, l2);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C2, 31, r2);
                let b1 = xor_rotr8::<12, 20>(state[$lb], c1s);
                let (t1, l3, r3, c3) = add_carry_parts_v8(a1, b1);
                pushf8!(wz, GS_BASE + G_STRIDE * $g + REC_C3, 31, c3);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, l3);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C3, 31, r3);
                let (a2, l4, r4, c4) = add_carry_parts_v8(t1, m[$my]);
                pushf8!(wz, GS_BASE + G_STRIDE * $g + REC_C4, 31, c4);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, l4);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C4, 31, r4);
                let d2 = xor_rotr8::<8, 24>(d1, a2);
                let (c2s, l5, r5, c5) = add_carry_parts_v8(c1s, d2);
                pushf8!(wz, GS_BASE + G_STRIDE * $g + REC_C5, 31, c5);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, l5);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C5, 31, r5);
                let bn = xor_rotr8::<7, 25>(b1, c2s);
                pushf8!(wz, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, maxv);
                pushf8!(wz, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
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
                pushf8!(wz, OUT_HI_BASE + 32 * $w, 32, hv);
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
        wz.finish();
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
            store_v8(zs.add(i) as *mut u32, zero);
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
            store_v8(zs.add(w) as *mut u32, cv_v[w]);
            store_v8(ast.add(w) as *mut u32, cv_v[w]);
            store_v8(bs.add(w) as *mut u32, maxv);
            store_v8(zs.add(8 + w) as *mut u32, lo);
            store_v8(ast.add(8 + w) as *mut u32, lo);
            store_v8(bs.add(8 + w) as *mut u32, maxv);
        }
        drain.drain_range(0, 0, 16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
