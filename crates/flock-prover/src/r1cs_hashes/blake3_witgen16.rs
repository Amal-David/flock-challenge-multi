//! 16-wide Sapphire Rapids BLAKE3 witness arithmetic.
//!
//! State words, G arithmetic, carry witnesses, and packed-word assembly stay
//! in `__m512i` for all sixteen compressions. A packed word is split into two
//! V8 halves only when it enters the existing rolling ring, so the byte layout,
//! constant elision, NT publication, and round-1 streaming projection remain
//! exactly the proven [`super::Drain8`] contract.

use super::*;

type V16 = __m512i;

/// Input form for one sixteen-compression witness invocation.
pub(crate) enum HexaInputs<'a> {
    // The ranked release path is Closed-only. Keep arbitrary inputs solely in
    // the compiled SPR oracle so the release body cannot retain a cold gather
    // arm or an input discriminant that the dispatcher never selects.
    #[cfg(test)]
    Blocks([&'a Compression; 16]),
    Closed {
        init: u64,
        base: usize,
        _lifetime: core::marker::PhantomData<&'a Compression>,
    },
}

impl HexaInputs<'_> {
    #[inline(always)]
    pub(crate) fn closed(init: u64, base: usize) -> Self {
        Self::Closed {
            init,
            base,
            _lifetime: core::marker::PhantomData,
        }
    }
}

struct PreparedInputs16 {
    cv: [V16; 8],
    message: [V16; 16],
    counter_lo: V16,
    counter_hi: V16,
    block_len: V16,
    flags: V16,
}

#[inline(always)]
fn dup_u32x16(x: u32) -> V16 {
    unsafe { _mm512_set1_epi32(x as i32) }
}

#[inline(always)]
#[cfg(test)]
unsafe fn load_v16(p: *const u32) -> V16 {
    unsafe { _mm512_loadu_si512(p.cast::<__m512i>()) }
}

#[inline(always)]
fn join_v8(lo: V8, hi: V8) -> V16 {
    unsafe { _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi) }
}

#[inline(always)]
fn split_v16(v: V16) -> (V8, V8) {
    unsafe { (_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64::<1>(v)) }
}

#[inline(always)]
unsafe fn next_generator_draw16(states: &mut [__m512i; 2]) -> V16 {
    unsafe {
        let golden = _mm512_set1_epi64(crate::seed_pipe::GOLDEN as i64);
        states[0] = _mm512_add_epi64(states[0], golden);
        states[1] = _mm512_add_epi64(states[1], golden);
        let lo = _mm512_cvtepi64_epi32(mix_u64x8(states[0]));
        let hi = _mm512_cvtepi64_epi32(mix_u64x8(states[1]));
        join_v8(lo, hi)
    }
}

#[inline(always)]
unsafe fn prepare_closed_inputs16(init: u64, base: usize) -> PreparedInputs16 {
    unsafe {
        let stride =
            crate::seed_pipe::GOLDEN.wrapping_mul(crate::seed_pipe::DRAWS_PER_BLOCK as u64);
        let first = init.wrapping_add((base as u64).wrapping_mul(stride));
        let make_state = |lane_base: usize| {
            _mm512_setr_epi64(
                first.wrapping_add(stride.wrapping_mul((lane_base + 0) as u64)) as i64,
                first.wrapping_add(stride.wrapping_mul((lane_base + 1) as u64)) as i64,
                first.wrapping_add(stride.wrapping_mul((lane_base + 2) as u64)) as i64,
                first.wrapping_add(stride.wrapping_mul((lane_base + 3) as u64)) as i64,
                first.wrapping_add(stride.wrapping_mul((lane_base + 4) as u64)) as i64,
                first.wrapping_add(stride.wrapping_mul((lane_base + 5) as u64)) as i64,
                first.wrapping_add(stride.wrapping_mul((lane_base + 6) as u64)) as i64,
                first.wrapping_add(stride.wrapping_mul((lane_base + 7) as u64)) as i64,
            )
        };
        let mut states = [make_state(0), make_state(8)];
        let cv = std::array::from_fn(|_| next_generator_draw16(&mut states));
        let message = std::array::from_fn(|_| next_generator_draw16(&mut states));
        let counter_lo = next_generator_draw16(&mut states);
        PreparedInputs16 {
            cv,
            message,
            counter_lo,
            counter_hi: _mm512_setzero_si512(),
            block_len: _mm512_set1_epi32(64),
            flags: _mm512_set1_epi32(11),
        }
    }
}

#[inline(always)]
#[cfg(test)]
unsafe fn prepare_block_inputs16(inputs: [&Compression; 16]) -> PreparedInputs16 {
    unsafe {
        let cv = std::array::from_fn(|w| {
            let lanes: [u32; 16] = std::array::from_fn(|j| inputs[j].0[w]);
            load_v16(lanes.as_ptr())
        });
        let message = std::array::from_fn(|w| {
            let lanes: [u32; 16] = std::array::from_fn(|j| inputs[j].1[w]);
            load_v16(lanes.as_ptr())
        });
        let counter_lo = {
            let lanes: [u32; 16] = std::array::from_fn(|j| inputs[j].2 as u32);
            load_v16(lanes.as_ptr())
        };
        let counter_hi = {
            let lanes: [u32; 16] = std::array::from_fn(|j| (inputs[j].2 >> 32) as u32);
            load_v16(lanes.as_ptr())
        };
        let block_len = {
            let lanes: [u32; 16] = std::array::from_fn(|j| inputs[j].3);
            load_v16(lanes.as_ptr())
        };
        let flags = {
            let lanes: [u32; 16] = std::array::from_fn(|j| inputs[j].4);
            load_v16(lanes.as_ptr())
        };
        PreparedInputs16 {
            cv,
            message,
            counter_lo,
            counter_hi,
            block_len,
            flags,
        }
    }
}

#[inline(always)]
fn xor_v16(a: V16, b: V16) -> V16 {
    unsafe { _mm512_xor_si512(a, b) }
}

#[inline(always)]
fn or_v16(a: V16, b: V16) -> V16 {
    unsafe { _mm512_or_si512(a, b) }
}

#[inline(always)]
fn add_v16(a: V16, b: V16) -> V16 {
    unsafe { _mm512_add_epi32(a, b) }
}

#[inline(always)]
fn shr_v16<const N: u32>(v: V16) -> V16 {
    unsafe { _mm512_srli_epi32::<N>(v) }
}

#[inline(always)]
fn shl_v16<const N: u32>(v: V16) -> V16 {
    unsafe { _mm512_slli_epi32::<N>(v) }
}

#[inline(always)]
fn ror_v16<const N: i32>(v: V16) -> V16 {
    unsafe { _mm512_ror_epi32::<N>(v) }
}

#[inline(always)]
fn xor_rotr16<const N: i32, const M: i32>(x: V16, y: V16) -> V16 {
    debug_assert_eq!(N + M, 32);
    ror_v16::<N>(xor_v16(x, y))
}

#[inline(always)]
fn add_carry_parts_v16(x: V16, y: V16) -> (V16, V16, V16) {
    let sum = add_v16(x, y);
    (sum, xor_v16(sum, y), xor_v16(sum, x))
}

struct Drain16<'t> {
    halves: [Drain8<'t>; 2],
}

impl Drain16<'_> {
    #[inline(always)]
    unsafe fn drain_range(&mut self, base_word: usize, ring_word: usize, words: usize) {
        unsafe {
            self.halves[0].drain_range(base_word, ring_word, words);
            self.halves[1].drain_range(base_word, ring_word, words);
        }
    }
}

/// Lane-wise packed-word writer for sixteen blocks. Pending-bit arithmetic is
/// ZMM-wide; only a completed word is split for the two incumbent drain rings.
struct W16<'t> {
    pending: V16,
    stages: [*mut V8; 2],
    drain: *mut Drain16<'t>,
    flush: bool,
}

impl<'t> W16<'t> {
    #[inline(always)]
    fn at(stages: [*mut V8; 2], pending: V16, drain: *mut Drain16<'t>, flush: bool) -> Self {
        Self {
            pending,
            stages,
            drain,
            flush,
        }
    }

    #[inline(always)]
    unsafe fn write_word<const WORD: usize>(&mut self, v: V16) {
        unsafe {
            let (lo, hi) = split_v16(v);
            let i = WORD & (RING_WORDS - 1);
            store_v8(self.stages[0].add(i).cast::<u32>(), lo);
            store_v8(self.stages[1].add(i).cast::<u32>(), hi);
            if self.flush && WORD % RING_WORDS == RING_WORDS - 1 {
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
        v: V16,
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
                    self.pending = dup_u32x16(0);
                } else {
                    self.pending = shl_v16::<1>(v);
                }
            } else {
                let out = _mm512_shldi_epi32::<USED>(v, self.pending);
                self.write_word::<WORD>(out);
                self.pending = if WIDTH == 31 { shl_v16::<1>(v) } else { v };
            }
        }
    }

    #[inline(always)]
    unsafe fn finish(&mut self) {
        const {
            assert!(USEFUL_BITS % 32 == 17);
        }
        unsafe { self.write_word::<LAST_WORD>(shr_v16::<15>(self.pending)) }
    }
}

macro_rules! pushf16 {
    ($w:ident, $pos:expr, $width:literal, $v:expr) => {{
        $w.push::<{ ($pos % 32) as i32 }, $width, {
            let u = ($pos % 32) as i32;
            if u == 0 { 1 } else { 32 - u }
        }, { $pos / 32 }>($v);
    }};
}

/// Build `(z, a, b, ab_inner)` for sixteen blocks with one ZMM arithmetic
/// schedule. `proj` supplies the unchanged low/high eight-block streaming
/// projections; either half may be `None` for direct oracle tests.
///
/// # Safety
/// The crate must be compiled with AVX-512F, AVX-512VL, AVX-512DQ and
/// AVX-512VBMI2.
/// `z`/`a`/`b` each own sixteen contiguous 512-word blocks. Each projection,
/// when present, satisfies [`StreamProj`]'s contract. The caller retains the
/// incumbent same-thread `_mm_sfence()` publication contract.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) unsafe fn build_hexa_witness_ab_stream_elide(
    inputs: HexaInputs<'_>,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    proj: [Option<StreamProj<'_>>; 2],
    elide: [bool; 3],
    z_nt: bool,
) {
    unsafe {
        let prepared = match inputs {
            #[cfg(test)]
            HexaInputs::Blocks(inputs) => prepare_block_inputs16(inputs),
            HexaInputs::Closed { init, base, .. } => prepare_closed_inputs16(init, base),
        };
        let cv_v = prepared.cv;
        let m = prepared.message;
        let tlo = prepared.counter_lo;
        let thi = prepared.counter_hi;
        let blen = prepared.block_len;
        let flags = prepared.flags;

        let mut state: [V16; 16] = [
            cv_v[0],
            cv_v[1],
            cv_v[2],
            cv_v[3],
            cv_v[4],
            cv_v[5],
            cv_v[6],
            cv_v[7],
            dup_u32x16(BLAKE3_IV[0]),
            dup_u32x16(BLAKE3_IV[1]),
            dup_u32x16(BLAKE3_IV[2]),
            dup_u32x16(BLAKE3_IV[3]),
            tlo,
            thi,
            blen,
            flags,
        ];

        let mut ast_lo = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let mut ast_hi = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let mut bs_lo = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let mut bs_hi = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let ast = [
            ast_lo.as_mut_ptr().cast::<V8>(),
            ast_hi.as_mut_ptr().cast::<V8>(),
        ];
        let bs = [
            bs_lo.as_mut_ptr().cast::<V8>(),
            bs_hi.as_mut_ptr().cast::<V8>(),
        ];
        let [proj_lo, proj_hi] = proj;
        let block_half = 8 * U32_PER_BLOCK;
        let wide_nt = wide_nt_enabled();
        let spread = spread_nt_enabled();
        let mut drain = Drain16 {
            halves: [
                Drain8 {
                    ast: ast[0],
                    bs: bs[0],
                    z,
                    a,
                    b,
                    win_ab: None,
                    proj: proj_lo,
                    elide,
                    z_nt,
                    wide_nt,
                    spread,
                },
                Drain8 {
                    ast: ast[1],
                    bs: bs[1],
                    z: z.add(block_half),
                    a: a.add(block_half),
                    b: b.add(block_half),
                    win_ab: None,
                    proj: proj_hi,
                    elide,
                    z_nt,
                    wide_nt,
                    spread,
                },
            ],
        };

        let zero = dup_u32x16(0);
        let maxv = dup_u32x16(u32::MAX);
        let one = dup_u32x16(1);
        let chain: [V16; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15], tlo, thi, blen, flags,
        ];
        for k in 0..20usize {
            let v = if k == 0 {
                or_v16(one, shl_v16::<1>(chain[0]))
            } else {
                or_v16(shr_v16::<31>(chain[k - 1]), shl_v16::<1>(chain[k]))
            };
            let w = 16 + k;
            let (v_lo, v_hi) = split_v16(v);
            let (m_lo, m_hi) = split_v16(maxv);
            store_v8(ast[0].add(w & (RING_WORDS - 1)).cast::<u32>(), v_lo);
            store_v8(ast[1].add(w & (RING_WORDS - 1)).cast::<u32>(), v_hi);
            store_v8(bs[0].add(w & (RING_WORDS - 1)).cast::<u32>(), m_lo);
            store_v8(bs[1].add(w & (RING_WORDS - 1)).cast::<u32>(), m_hi);
        }
        if RING_WORDS <= PROLOGUE_WORDS + 16 {
            drain.drain_range(16, 16, RING_WORDS - 16);
        }

        let drain_ptr = &mut drain as *mut Drain16;
        let mut wa = W16::at(ast, shl_v16::<31>(shr_v16::<31>(flags)), drain_ptr, false);
        let mut wb = W16::at(bs, shl_v16::<31>(one), drain_ptr, true);

        macro_rules! g {
            ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
             $mx:literal, $my:literal) => {{
                let (t0, l0, r0) = add_carry_parts_v16(state[$la], state[$lb]);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, l0);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C0, 31, r0);
                let (a1, l1, r1) = add_carry_parts_v16(t0, m[$mx]);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, l1);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C1, 31, r1);
                let d1 = xor_rotr16::<16, 16>(state[$ld], a1);
                let (c1s, l2, r2) = add_carry_parts_v16(state[$lc], d1);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, l2);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C2, 31, r2);
                let b1 = xor_rotr16::<12, 20>(state[$lb], c1s);
                let (t1, l3, r3) = add_carry_parts_v16(a1, b1);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, l3);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C3, 31, r3);
                let (a2, l4, r4) = add_carry_parts_v16(t1, m[$my]);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, l4);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C4, 31, r4);
                let d2 = xor_rotr16::<8, 24>(d1, a2);
                let (c2s, l5, r5) = add_carry_parts_v16(c1s, d2);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, l5);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C5, 31, r5);
                let bn = xor_rotr16::<7, 25>(b1, c2s);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, maxv);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, maxv);
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
                let hv = xor_v16(state[$w + 8], cv_v[$w]);
                pushf16!(wa, OUT_HI_BASE + 32 * $w, 32, hv);
                pushf16!(wb, OUT_HI_BASE + 32 * $w, 32, maxv);
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
        let (zero_lo, zero_hi) = split_v16(zero);
        for w in ZF..U32_PER_BLOCK {
            let i = w & (RING_WORDS - 1);
            store_v8(ast[0].add(i).cast::<u32>(), zero_lo);
            store_v8(ast[1].add(i).cast::<u32>(), zero_hi);
            store_v8(bs[0].add(i).cast::<u32>(), zero_lo);
            store_v8(bs[1].add(i).cast::<u32>(), zero_hi);
        }
        drain.drain_range(U32_PER_BLOCK - RING_WORDS, 0, RING_WORDS);

        let (max_lo, max_hi) = split_v16(maxv);
        for w in 0..8usize {
            let lo = xor_v16(state[w], state[w + 8]);
            let (cv_lo, cv_hi) = split_v16(cv_v[w]);
            let (out_lo, out_hi) = split_v16(lo);
            store_v8(ast[0].add(w).cast::<u32>(), cv_lo);
            store_v8(ast[1].add(w).cast::<u32>(), cv_hi);
            store_v8(bs[0].add(w).cast::<u32>(), max_lo);
            store_v8(bs[1].add(w).cast::<u32>(), max_hi);
            store_v8(ast[0].add(8 + w).cast::<u32>(), out_lo);
            store_v8(ast[1].add(8 + w).cast::<u32>(), out_hi);
            store_v8(bs[0].add(8 + w).cast::<u32>(), max_lo);
            store_v8(bs[1].add(8 + w).cast::<u32>(), max_hi);
        }
        drain.drain_range(0, 0, 16);
    }
}
