//! 8-wide AVX2 lockstep BLAKE3 witness builder (`__m256i`, 8×u32).
//!
//! Same G-function / carry-bit / packed-row stream as the 4-wide SSE kernel,
//! widened to one rayon group (8 compressions) per call. Drain stores are
//! temporal (`storeu`); NT / `_mm256_stream` publishes stay disabled.
//!
//! Ranked live path: `generate_witness_with_ab_packed_and_round1_inner_impl`
//! (`FLOCK_NO_WITGEN_LIVE_SIMD=1` restores the scalar 1-block loop).

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

/// Lane-wise packed-word writer: 8 independent `PackedWordWriter`s.
struct W8 {
    pending: V8,
    stage: *mut V8,
}

impl W8 {
    #[inline(always)]
    fn at(stage: *mut V8, pending: V8) -> Self {
        Self { pending, stage }
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
                    store_v8(self.stage.add(WORD) as *mut u32, v);
                    self.pending = dup_u32(0);
                } else {
                    self.pending = v;
                }
            } else if USED + WIDTH < 32 {
                self.pending = vsli_v8::<USED>(self.pending, v);
            } else {
                let out = vsli_v8::<USED>(self.pending, v);
                store_v8(self.stage.add(WORD) as *mut u32, out);
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
            store_v8(self.stage.add(LAST_WORD) as *mut u32, self.pending);
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
    // cin = sum ^ x ^ y. On SPR (avx512vl) that's one `vpternlogd` (imm 0x96)
    // instead of two `vpxor`. Algebra is unchanged: wrapping add still
    // defines the R1CS carry bits via (x^cin) & (y^cin).
    #[cfg(target_feature = "avx512vl")]
    let cin = unsafe { _mm256_ternarylogic_epi32(sum, x, y, 0x96) };
    #[cfg(not(target_feature = "avx512vl"))]
    let cin = xor_v8(xor_v8(sum, x), y);
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

#[inline(always)]
unsafe fn dump_elide(
    stage: *const V8,
    dst: *mut u32,
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
    unsafe { dump_range(stage, dst, g0, g1) }
}

/// Build `(z, a, b)` for EIGHT compressions in u32-lane lockstep.
/// Bit-exact with two 4-wide quads and with the scalar driver ×8.
///
/// # Safety
/// Caller must have AVX2. `z`/`a`/`b` each own 8 contiguous 512-word blocks.
pub(crate) unsafe fn build_octa_witness_ab_stream_elide(
    inputs: [&Compression; 8],
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    elide: [bool; 3],
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
        let mut zs = core::mem::MaybeUninit::<[V8; U32_PER_BLOCK]>::uninit();
        let mut ast = core::mem::MaybeUninit::<[V8; U32_PER_BLOCK]>::uninit();
        let mut bs = core::mem::MaybeUninit::<[V8; U32_PER_BLOCK]>::uninit();
        let zs = zs.as_mut_ptr().cast::<V8>();
        let ast = ast.as_mut_ptr().cast::<V8>();
        let bs = bs.as_mut_ptr().cast::<V8>();

        for w in 0..8usize {
            store_v8(zs.add(w) as *mut u32, cv_v[w]);
            store_v8(ast.add(w) as *mut u32, cv_v[w]);
        }
        let maxv = dup_u32(u32::MAX);
        for w in 0..36usize {
            store_v8(bs.add(w) as *mut u32, maxv);
        }
        let one = dup_u32(1);
        let chain: [V8; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12],
            m[13], m[14], m[15], tlo, thi, blen, flags,
        ];
        store_v8(
            zs.add(16) as *mut u32,
            or_v8(one, shl_v8::<1>(chain[0])),
        );
        for k in 1..20usize {
            let w = or_v8(shr_v8::<31>(chain[k - 1]), shl_v8::<1>(chain[k]));
            store_v8(zs.add(16 + k) as *mut u32, w);
        }
        for w in 16..36usize {
            let v = load_v8(zs.add(w) as *const u32);
            store_v8(ast.add(w) as *mut u32, v);
        }

        let pending_bit = shr_v8::<31>(flags);
        let mut wz = W8::at(zs, pending_bit);
        let mut wa = W8::at(ast, pending_bit);
        let mut wb = W8::at(bs, one);

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
        for w in 0..30usize {
            store_v8(zs.add(ZF + w) as *mut u32, zero);
            store_v8(ast.add(ZF + w) as *mut u32, zero);
            store_v8(bs.add(ZF + w) as *mut u32, zero);
        }

        for w in 0..8usize {
            let lo = xor_v8(state[w], state[w + 8]);
            store_v8(zs.add(8 + w) as *mut u32, lo);
            store_v8(ast.add(8 + w) as *mut u32, lo);
        }

        dump_elide(zs, z, elide[0], false, ELIDE_ZERO_CHUNK);
        dump_elide(ast, a, elide[1], false, ELIDE_ZERO_CHUNK);
        dump_elide(bs, b, elide[2], elide[2], ELIDE_B_TAIL_CHUNK);
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
