//! 16-wide AVX-512 lockstep BLAKE3 witness builder (`__m512i`, 16×u32).
//!
//! Same G-function / carry-bit / packed-row stream as [`super::blake3_witgen8`],
//! widened to one rayon GROUP (16 compressions) per call. Dump still transposes
//! 8 packed words at a time so the 8-wide elide chunk geometry is unchanged:
//! each ZMM is split into two YMM halves, each half reuses the 8×8 transpose.
//! a/b drains stay temporal. z optionally publishes through the same paired
//! ZMM NT stores as the 8-wide path (`z_nt`).
//!
//! Ranked live path when compiled with `avx512f` unless `FLOCK_NO_WITGEN16=1`.

use super::{
    ADDS_PER_G, BLAKE3_IV, CARRY_BITS_PER_ADD, Compression, G_STRIDE, GS_BASE, K, OUT_HI_BASE,
    USEFUL_BITS, WORD_BITS,
};
use core::arch::x86_64::*;

const REC_C0: usize = 0;
const REC_C1: usize = CARRY_BITS_PER_ADD;
const REC_C2: usize = 2 * CARRY_BITS_PER_ADD;
const REC_C3: usize = 3 * CARRY_BITS_PER_ADD;
const REC_C4: usize = 4 * CARRY_BITS_PER_ADD;
const REC_C5: usize = 5 * CARRY_BITS_PER_ADD;
const REC_LIN0: usize = ADDS_PER_G * CARRY_BITS_PER_ADD;
const REC_LIN1: usize = REC_LIN0 + WORD_BITS;
const U32_PER_BLOCK: usize = K / 32;
const DUMP_CHUNKS: usize = U32_PER_BLOCK / 8;
const ELIDE_ZERO_CHUNK: usize = 61;
const ELIDE_B_TAIL_CHUNK: usize = 59;
const ELIDE_B_PREFIX_CHUNKS: usize = 4;
const LAST_WORD: usize = (USEFUL_BITS - 1) / 32;
const _ELIDE_GEOMETRY: () = {
    assert!(8 * ELIDE_ZERO_CHUNK >= USEFUL_BITS.div_ceil(32));
    assert!(8 * ELIDE_ZERO_CHUNK < U32_PER_BLOCK);
    assert!(8 * ELIDE_B_PREFIX_CHUNKS <= 36);
    assert!(LAST_WORD == 481);
};

type V8 = __m256i;
type V16 = __m512i;

pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_WITGEN16").is_none());
    *ON
}

#[inline(always)]
unsafe fn store_v8(p: *mut u32, v: V8) {
    unsafe { _mm256_storeu_si256(p.cast::<__m256i>(), v) }
}

#[inline(always)]
unsafe fn load_v16(p: *const u32) -> V16 {
    unsafe { _mm512_loadu_si512(p.cast::<__m512i>()) }
}

#[inline(always)]
unsafe fn store_v16(p: *mut u32, v: V16) {
    unsafe { _mm512_storeu_si512(p.cast::<__m512i>(), v) }
}

#[inline(always)]
fn dup_u32(x: u32) -> V16 {
    unsafe { _mm512_set1_epi32(x as i32) }
}

#[inline(always)]
fn xor_v16(a: V16, b: V16) -> V16 {
    unsafe { _mm512_xor_si512(a, b) }
}

/// Three-way XOR `a ^ b ^ c` via `vpternlogd` immediate `0x96`.
/// `FLOCK_NO_WITGEN_TERNLOG=1` restores the two-XOR fallback.
#[inline(always)]
fn xor3_v16(a: V16, b: V16, c: V16) -> V16 {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_WITGEN_TERNLOG").is_none());
    if *ON {
        unsafe { _mm512_ternarylogic_epi32::<0x96>(a, b, c) }
    } else {
        xor_v16(xor_v16(a, b), c)
    }
}

#[inline(always)]
fn or_v16(a: V16, b: V16) -> V16 {
    unsafe { _mm512_or_si512(a, b) }
}

#[inline(always)]
fn and_v16(a: V16, b: V16) -> V16 {
    unsafe { _mm512_and_si512(a, b) }
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

/// Lane-wise rotate-right. Bit-identical to `(x >> N) | (x << (32-N))`.
#[inline(always)]
fn ror_v16<const N: i32>(v: V16) -> V16 {
    unsafe { _mm512_ror_epi32::<N>(v) }
}

/// NEON `vsli` #N, 16 lanes: bits `N..32` from `b << N`, bits `0..N` keep `a`.
#[inline(always)]
fn vsli_v16<const N: u32>(a: V16, b: V16) -> V16 {
    unsafe {
        let mask = _mm512_set1_epi32(((1u64 << N) - 1) as u32 as i32);
        _mm512_or_si512(_mm512_slli_epi32::<N>(b), _mm512_and_si512(a, mask))
    }
}

/// 8×8 u32 transpose on YMM. Same permutation as the 8-wide builder.
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

#[inline(always)]
fn split_v16(v: V16) -> (V8, V8) {
    unsafe {
        (
            _mm512_castsi512_si256(v),
            _mm512_extracti64x4_epi64::<1>(v),
        )
    }
}

/// Lane-wise packed-word writer: 16 independent `PackedWordWriter`s.
struct W16 {
    pending: V16,
    stage: *mut V16,
}

impl W16 {
    #[inline(always)]
    fn at(stage: *mut V16, pending: V16) -> Self {
        Self { pending, stage }
    }

    #[inline(always)]
    unsafe fn push<const USED: u32, const WIDTH: i32, const BACK: u32, const WORD: usize>(
        &mut self,
        v: V16,
    ) {
        const {
            assert!(USED < 32);
            assert!(WIDTH == 31 || WIDTH == 32);
            assert!(BACK >= 1 && BACK < 32);
            assert!(WORD < U32_PER_BLOCK);
        }
        debug_assert!(USED + (WIDTH as u32) <= 32 || BACK == 32 - USED);
        unsafe {
            if USED == 0 {
                if WIDTH == 32 {
                    store_v16(self.stage.add(WORD) as *mut u32, v);
                    self.pending = dup_u32(0);
                } else {
                    self.pending = v;
                }
            } else if USED + (WIDTH as u32) < 32 {
                self.pending = vsli_v16::<USED>(self.pending, v);
            } else {
                let out = vsli_v16::<USED>(self.pending, v);
                store_v16(self.stage.add(WORD) as *mut u32, out);
                if USED + (WIDTH as u32) == 32 {
                    self.pending = dup_u32(0);
                } else {
                    self.pending = shr_v16::<BACK>(v);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn finish(&mut self) {
        unsafe {
            store_v16(self.stage.add(LAST_WORD) as *mut u32, self.pending);
        }
    }
}

macro_rules! pushf16 {
    ($w:ident, $pos:expr, $width:literal, $v:expr) => {{
        $w.push::<{ ($pos % 32) as u32 }, $width, {
            let u = ($pos % 32) as u32;
            if u == 0 { 1 } else { 32 - u }
        }, { $pos / 32 }>($v);
    }};
}

#[inline(always)]
fn add_carry_parts_v16(x: V16, y: V16) -> (V16, V16, V16, V16) {
    let sum = add_v16(x, y);
    let cin = xor3_v16(sum, x, y);
    let left = xor_v16(x, cin);
    let right = xor_v16(y, cin);
    let carry = and_v16(left, right);
    (sum, left, right, carry)
}

#[inline(always)]
fn xor_rotr16<const N: i32>(x: V16, y: V16) -> V16 {
    ror_v16::<N>(xor_v16(x, y))
}

/// Split one 8-word dump chunk of 16-lane stage into two 8×8 YMM transposes.
#[inline(always)]
unsafe fn transpose_chunk(stage: *const V16, g: usize) -> ([V8; 8], [V8; 8]) {
    unsafe {
        let w = 8 * g;
        let z0 = load_v16(stage.add(w) as *const u32);
        let z1 = load_v16(stage.add(w + 1) as *const u32);
        let z2 = load_v16(stage.add(w + 2) as *const u32);
        let z3 = load_v16(stage.add(w + 3) as *const u32);
        let z4 = load_v16(stage.add(w + 4) as *const u32);
        let z5 = load_v16(stage.add(w + 5) as *const u32);
        let z6 = load_v16(stage.add(w + 6) as *const u32);
        let z7 = load_v16(stage.add(w + 7) as *const u32);
        let (l0, h0) = split_v16(z0);
        let (l1, h1) = split_v16(z1);
        let (l2, h2) = split_v16(z2);
        let (l3, h3) = split_v16(z3);
        let (l4, h4) = split_v16(z4);
        let (l5, h5) = split_v16(z5);
        let (l6, h6) = split_v16(z6);
        let (l7, h7) = split_v16(z7);
        (
            tr8(l0, l1, l2, l3, l4, l5, l6, l7),
            tr8(h0, h1, h2, h3, h4, h5, h6, h7),
        )
    }
}

#[inline(always)]
unsafe fn dump_range(stage: *const V16, dst: *mut u32, g0: usize, g1: usize) {
    unsafe {
        for g in g0..g1 {
            let w = 8 * g;
            let (lo, hi) = transpose_chunk(stage, g);
            for r in 0..8 {
                store_v8(dst.add(r * U32_PER_BLOCK + w), lo[r]);
                store_v8(dst.add((r + 8) * U32_PER_BLOCK + w), hi[r]);
            }
        }
    }
}

fn wide_nt_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_WIDE_NT").is_none());
    *ON
}

#[inline(always)]
unsafe fn dump_range_nt(stage: *const V16, dst: *mut u32, g0: usize, g1: usize) {
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
    unsafe {
        debug_assert_eq!(dst as usize % 16, 0);
        let mut g = g0;
        while g + 2 <= g1 {
            let w = 8 * g;
            let (lo_a, hi_a) = transpose_chunk(stage, g);
            let (lo_b, hi_b) = transpose_chunk(stage, g + 1);
            for r in 0..8 {
                let p_lo = dst.add(r * U32_PER_BLOCK + w);
                let p_hi = dst.add((r + 8) * U32_PER_BLOCK + w);
                if wide_nt_enabled() && p_lo as usize % 64 == 0 {
                    let z = _mm512_castsi256_si512(lo_a[r]);
                    let z = _mm512_inserti64x4::<1>(z, lo_b[r]);
                    _mm512_stream_si512(p_lo.cast::<__m512i>(), z);
                } else {
                    stream_v8(p_lo, lo_a[r]);
                    stream_v8(p_lo.add(8), lo_b[r]);
                }
                if wide_nt_enabled() && p_hi as usize % 64 == 0 {
                    let z = _mm512_castsi256_si512(hi_a[r]);
                    let z = _mm512_inserti64x4::<1>(z, hi_b[r]);
                    _mm512_stream_si512(p_hi.cast::<__m512i>(), z);
                } else {
                    stream_v8(p_hi, hi_a[r]);
                    stream_v8(p_hi.add(8), hi_b[r]);
                }
            }
            g += 2;
        }
        if g < g1 {
            let w = 8 * g;
            let (lo, hi) = transpose_chunk(stage, g);
            for r in 0..8 {
                stream_v8(dst.add(r * U32_PER_BLOCK + w), lo[r]);
                stream_v8(dst.add((r + 8) * U32_PER_BLOCK + w), hi[r]);
            }
        }
    }
}

#[inline(always)]
unsafe fn dump_elide(
    stage: *const V16,
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

#[inline(always)]
fn load_word16(inputs: &[&Compression; 16], get: impl Fn(&Compression) -> u32) -> V16 {
    let mut tmp = [0u32; 16];
    for j in 0..16 {
        tmp[j] = get(inputs[j]);
    }
    unsafe { load_v16(tmp.as_ptr()) }
}

#[inline(always)]
fn load_cv_word(inputs: &[&Compression; 16], w: usize) -> V16 {
    let mut tmp = [0u32; 16];
    for j in 0..16 {
        tmp[j] = inputs[j].0[w];
    }
    unsafe { load_v16(tmp.as_ptr()) }
}

#[inline(always)]
fn load_msg_word(inputs: &[&Compression; 16], w: usize) -> V16 {
    let mut tmp = [0u32; 16];
    for j in 0..16 {
        tmp[j] = inputs[j].1[w];
    }
    unsafe { load_v16(tmp.as_ptr()) }
}

/// Build `(z, a, b)` for SIXTEEN compressions in u32-lane lockstep.
/// Bit-exact with two 8-wide octa calls and with the scalar driver ×16.
///
/// # Safety
/// Caller must have AVX-512F. `z`/`a`/`b` each own 16 contiguous 512-word blocks.
pub(crate) unsafe fn build_hexa_witness_ab_stream_elide(
    inputs: [&Compression; 16],
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    elide: [bool; 3],
    z_nt: bool,
) {
    unsafe {
        let cv_v = [
            load_cv_word(&inputs, 0),
            load_cv_word(&inputs, 1),
            load_cv_word(&inputs, 2),
            load_cv_word(&inputs, 3),
            load_cv_word(&inputs, 4),
            load_cv_word(&inputs, 5),
            load_cv_word(&inputs, 6),
            load_cv_word(&inputs, 7),
        ];
        let mut m = [dup_u32(0); 16];
        for w in 0..16 {
            m[w] = load_msg_word(&inputs, w);
        }
        let tlo = load_word16(&inputs, |c| c.2 as u32);
        let thi = load_word16(&inputs, |c| (c.2 >> 32) as u32);
        let blen = load_word16(&inputs, |c| c.3);
        let flags = load_word16(&inputs, |c| c.4);

        let mut state: [V16; 16] = [
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
        let mut zs = core::mem::MaybeUninit::<[V16; U32_PER_BLOCK]>::uninit();
        let mut ast = core::mem::MaybeUninit::<[V16; U32_PER_BLOCK]>::uninit();
        let mut bs = core::mem::MaybeUninit::<[V16; U32_PER_BLOCK]>::uninit();
        let zs = zs.as_mut_ptr().cast::<V16>();
        let ast = ast.as_mut_ptr().cast::<V16>();
        let bs = bs.as_mut_ptr().cast::<V16>();

        for w in 0..8usize {
            store_v16(zs.add(w) as *mut u32, cv_v[w]);
            store_v16(ast.add(w) as *mut u32, cv_v[w]);
        }
        let maxv = dup_u32(u32::MAX);
        for w in 0..36usize {
            store_v16(bs.add(w) as *mut u32, maxv);
        }
        let one = dup_u32(1);
        let chain: [V16; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12],
            m[13], m[14], m[15], tlo, thi, blen, flags,
        ];
        store_v16(
            zs.add(16) as *mut u32,
            or_v16(one, shl_v16::<1>(chain[0])),
        );
        for k in 1..20usize {
            let w = or_v16(shr_v16::<31>(chain[k - 1]), shl_v16::<1>(chain[k]));
            store_v16(zs.add(16 + k) as *mut u32, w);
        }
        for w in 16..36usize {
            let v = load_v16(zs.add(w) as *const u32);
            store_v16(ast.add(w) as *mut u32, v);
        }

        let pending_bit = shr_v16::<31>(flags);
        let mut wz = W16::at(zs, pending_bit);
        let mut wa = W16::at(ast, pending_bit);
        let mut wb = W16::at(bs, one);

        macro_rules! g {
            ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
             $mx:literal, $my:literal) => {{
                let (t0, l0, r0, c0) = add_carry_parts_v16(state[$la], state[$lb]);
                pushf16!(wz, GS_BASE + G_STRIDE * $g + REC_C0, 31, c0);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, l0);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C0, 31, r0);
                let (a1, l1, r1, c1) = add_carry_parts_v16(t0, m[$mx]);
                pushf16!(wz, GS_BASE + G_STRIDE * $g + REC_C1, 31, c1);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, l1);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C1, 31, r1);
                let d1 = xor_rotr16::<16>(state[$ld], a1);
                let (c1s, l2, r2, c2) = add_carry_parts_v16(state[$lc], d1);
                pushf16!(wz, GS_BASE + G_STRIDE * $g + REC_C2, 31, c2);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, l2);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C2, 31, r2);
                let b1 = xor_rotr16::<12>(state[$lb], c1s);
                let (t1, l3, r3, c3) = add_carry_parts_v16(a1, b1);
                pushf16!(wz, GS_BASE + G_STRIDE * $g + REC_C3, 31, c3);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, l3);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C3, 31, r3);
                let (a2, l4, r4, c4) = add_carry_parts_v16(t1, m[$my]);
                pushf16!(wz, GS_BASE + G_STRIDE * $g + REC_C4, 31, c4);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, l4);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C4, 31, r4);
                let d2 = xor_rotr16::<8>(d1, a2);
                let (c2s, l5, r5, c5) = add_carry_parts_v16(c1s, d2);
                pushf16!(wz, GS_BASE + G_STRIDE * $g + REC_C5, 31, c5);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, l5);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_C5, 31, r5);
                let bn = xor_rotr16::<7>(b1, c2s);
                pushf16!(wz, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf16!(wa, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf16!(wb, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, maxv);
                pushf16!(wz, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
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
                pushf16!(wz, OUT_HI_BASE + 32 * $w, 32, hv);
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
        wz.finish();
        wa.finish();
        wb.finish();

        const ZF: usize = USEFUL_BITS.div_ceil(32);
        const {
            assert!(U32_PER_BLOCK - ZF == 30);
        }
        for w in 0..30usize {
            store_v16(zs.add(ZF + w) as *mut u32, zero);
            store_v16(ast.add(ZF + w) as *mut u32, zero);
            store_v16(bs.add(ZF + w) as *mut u32, zero);
        }

        for w in 0..8usize {
            let lo = xor_v16(state[w], state[w + 8]);
            store_v16(zs.add(8 + w) as *mut u32, lo);
            store_v16(ast.add(8 + w) as *mut u32, lo);
        }

        dump_elide(zs, z, elide[0], false, ELIDE_ZERO_CHUNK, z_nt);
        dump_elide(ast, a, elide[1], false, ELIDE_ZERO_CHUNK, false);
        dump_elide(bs, b, elide[2], elide[2], ELIDE_B_TAIL_CHUNK, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witgen16_tr8_is_8x8_u32_transpose() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
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
        }
    }
}
