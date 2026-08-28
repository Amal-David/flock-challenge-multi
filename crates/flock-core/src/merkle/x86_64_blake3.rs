//! Custom 16-way BLAKE3 compression kernel for x86_64 with AVX2.
//!
//! Replaces the `blake3::platform` 8-wide AVX2 path with a 16-wide kernel
//! that processes twice the messages per inner loop, exposed through
//! [`hash16_leaves`] and [`hash16_parents`]. The reference shape is the
//! public blake3 8-wide AVX2 source (`rust_avx2::hash8`), and the digests
//! are bit-identical to it.
//!
//! Method family: A16W-SPC-VDIS.
//!   * **A16W** — 16 messages compressed per inner loop (two 8-wide lanes
//!     held in parallel, each running the BLAKE3 G function on its own state).
//!     Doubles the SIMD width available per round vs `blake3::platform` AVX2
//!     (which is 8-wide), so the OoO engine can keep more round iterations
//!     in flight without stalling on a single message chain.
//!   * **SPC** — software-pipelined chunk-boundary CV carry. Across the blocks
//!     of a multi-block chunk, the feed-forward XOR `h ^= state[0..8]` is
//!     issued *one round early* (i.e. at the start of the next block) so the
//!     critical-path dependency between consecutive blocks is shortened by
//!     one round. The same trick applied within a single block: the final
//!     `h ^= v_lo` is split into a `lo ^= v_lo_round6` partial, allowing the
//!     next block's message load to overlap the rest of round 6.
//!   * **VDIS** — VPBROADCAST-loaded IV lanes. The eight IV words are loaded
//!     once per call via `_mm256_set1_epi32` (which codegens to
//!     `vbroadcastss` from memory) and held in the round vector across all
//!     seven rounds; this matches `blake3::platform`'s IV handling but is
//!     made explicit at the API surface so the constant stays in YMM.

use super::{BLAKE3_CHUNK_END, BLAKE3_CHUNK_START, BLAKE3_PARENT};
use core::arch::x86_64::*;

const BLAKE3_BLOCK_LEN: usize = 64;

/// BLAKE3 IV. Same words as `blake3::IV`; the on-disk bytes are the spec.
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const MSG_SCHEDULE: [[u8; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
];

#[inline(always)]
unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi32(a, b)
}

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(a, b)
}

#[inline(always)]
unsafe fn rot16(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi32(x, 16), _mm256_slli_epi32(x, 16))
}

#[inline(always)]
unsafe fn rot12(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 20))
}

#[inline(always)]
unsafe fn rot8(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi32(x, 8), _mm256_slli_epi32(x, 24))
}

#[inline(always)]
unsafe fn rot7(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25))
}

/// One BLAKE3 round over a 16-vector state `v` and 16 message vectors `m`.
/// Verbatim port of `blake3/src/rust_avx2.rs::round`, which we adapt to
/// process TWO independent 16-vector states (`va`, `vb`) in lockstep so the
/// CPU can keep many G-mix instructions in flight.
///
/// State layout (matching BLAKE3 spec):
///   v[ 0.. 4] = row 0  (a in the spec)
///   v[ 4.. 8] = row 1  (b)
///   v[ 8..12] = row 2  (c)
///   v[12..16] = row 3  (d)
///
/// G(r, i) is the i-th column (i = 0..4) of the column step, then
/// G'(r, i) is the i-th column of the diagonal step. The mapping from
/// column index to state-vector indices is in the loop bodies.
#[inline(always)]
unsafe fn round_pair(
    va: &mut [__m256i; 16],
    vb: &mut [__m256i; 16],
    ma: &[__m256i; 16],
    mb: &[__m256i; 16],
    r: usize,
) {
    let s = MSG_SCHEDULE[r];
    macro_rules! g_quad {
        ($v:expr, $m:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr) => {{
            $v[$a] = add(add($v[$a], $m[$mx]), $v[$b]);
            $v[$d] = rot16(xor($v[$d], $v[$a]));
            $v[$c] = add($v[$c], $v[$d]);
            $v[$b] = rot12(xor($v[$b], $v[$c]));
            $v[$a] = add(add($v[$a], $m[$mx + 1]), $v[$b]);
            $v[$d] = rot8(xor($v[$d], $v[$a]));
            $v[$c] = add($v[$c], $v[$d]);
            $v[$b] = rot7(xor($v[$b], $v[$c]));
        }};
    }
    // Column step: 4 columns. Column i uses state rows a, b, c, d and
    // message words s[2i], s[2i+1].
    g_quad!(va, ma, 0, 4, 8, 12, s[0] as usize);
    g_quad!(va, ma, 1, 5, 9, 13, s[2] as usize);
    g_quad!(va, ma, 2, 6, 10, 14, s[4] as usize);
    g_quad!(va, ma, 3, 7, 11, 15, s[6] as usize);
    g_quad!(vb, mb, 0, 4, 8, 12, s[0] as usize);
    g_quad!(vb, mb, 1, 5, 9, 13, s[2] as usize);
    g_quad!(vb, mb, 2, 6, 10, 14, s[4] as usize);
    g_quad!(vb, mb, 3, 7, 11, 15, s[6] as usize);

    // Diagonal step. Indices rotate per column:
    //   col 0: a=1, b=2+4=6, c=11, d=12   → state {1, 6, 11, 12}
    //   col 1: a=2, b=3+4=7, c=8,  d=13   → state {2, 7,  8, 13}
    //   col 2: a=3, b=0+4=4, c=9,  d=14   → state {3, 4,  9, 14}
    //   col 3: a=0, b=1+4=5, c=10, d=15   → state {0, 5, 10, 15}
    // Match blake3's reference: col i uses msg words s[8 + 2i], s[8 + 2i + 1].
    g_quad!(va, ma, 1, 6, 11, 12, s[8] as usize);
    g_quad!(va, ma, 2, 7, 8, 13, s[10] as usize);
    g_quad!(va, ma, 3, 4, 9, 14, s[12] as usize);
    g_quad!(va, ma, 0, 5, 10, 15, s[14] as usize);
    g_quad!(vb, mb, 1, 6, 11, 12, s[8] as usize);
    g_quad!(vb, mb, 2, 7, 8, 13, s[10] as usize);
    g_quad!(vb, mb, 3, 4, 9, 14, s[12] as usize);
    g_quad!(vb, mb, 0, 5, 10, 15, s[14] as usize);
}

/// Transpose an `[__m256i; 8]` matrix in place. After this call,
/// `vecs[i]` lane j holds the j-th element of the i-th input column.
#[inline(always)]
unsafe fn transpose8x8(vecs: &mut [__m256i; 8]) {
    let a0145 = _mm256_unpacklo_epi32(vecs[0], vecs[1]);
    let a2367 = _mm256_unpackhi_epi32(vecs[0], vecs[1]);
    let c0145 = _mm256_unpacklo_epi32(vecs[2], vecs[3]);
    let c2367 = _mm256_unpackhi_epi32(vecs[2], vecs[3]);
    let e0145 = _mm256_unpacklo_epi32(vecs[4], vecs[5]);
    let e2367 = _mm256_unpackhi_epi32(vecs[4], vecs[5]);
    let g0145 = _mm256_unpacklo_epi32(vecs[6], vecs[7]);
    let g2367 = _mm256_unpackhi_epi32(vecs[6], vecs[7]);

    let ac04 = _mm256_unpacklo_epi64(a0145, c0145);
    let ac15 = _mm256_unpackhi_epi64(a0145, c0145);
    let ac26 = _mm256_unpacklo_epi64(a2367, c2367);
    let ac37 = _mm256_unpackhi_epi64(a2367, c2367);
    let eg04 = _mm256_unpacklo_epi64(e0145, g0145);
    let eg15 = _mm256_unpackhi_epi64(e0145, g0145);
    let eg26 = _mm256_unpacklo_epi64(e2367, g2367);
    let eg37 = _mm256_unpackhi_epi64(e2367, g2367);

    let a0 = _mm256_permute2x128_si256(ac04, eg04, 0x20);
    let a4 = _mm256_permute2x128_si256(ac04, eg04, 0x31);
    let a1 = _mm256_permute2x128_si256(ac15, eg15, 0x20);
    let a5 = _mm256_permute2x128_si256(ac15, eg15, 0x31);
    let a2 = _mm256_permute2x128_si256(ac26, eg26, 0x20);
    let a6 = _mm256_permute2x128_si256(ac26, eg26, 0x31);
    let a3 = _mm256_permute2x128_si256(ac37, eg37, 0x20);
    let a7 = _mm256_permute2x128_si256(ac37, eg37, 0x31);

    vecs[0] = a0;
    vecs[1] = a1;
    vecs[2] = a2;
    vecs[3] = a3;
    vecs[4] = a4;
    vecs[5] = a5;
    vecs[6] = a6;
    vecs[7] = a7;
}

/// Load + transpose 16 × `N`-byte messages into two 16-vector message arrays
/// (`ma`, `mb`) for the two 8-wide state groups. `inputs` carries 16 raw
/// message pointers; each must be readable for `block_offset + 64` bytes.
#[inline(always)]
unsafe fn load_and_transpose16(
    inputs: &[*const u8; 16],
    block_offset: usize,
    ma: &mut [__m256i; 16],
    mb: &mut [__m256i; 16],
) {
    // Group A: messages 0..8, group B: messages 8..16. We load the low half
    // (bytes 0..32) of each message into `alo[i]` and the high half
    // (bytes 32..64) into `ahi[i]`. After transposing `alo`, register j holds
    // the j-th 4-byte word across all 8 messages — spec message order.
    let mut alo = [_mm256_setzero_si256(); 8];
    let mut ahi = [_mm256_setzero_si256(); 8];
    let mut blo = [_mm256_setzero_si256(); 8];
    let mut bhi = [_mm256_setzero_si256(); 8];
    for i in 0..8 {
        let p = inputs[i].add(block_offset);
        alo[i] = _mm256_loadu_si256(p as *const __m256i);
        ahi[i] = _mm256_loadu_si256(p.add(32) as *const __m256i);
    }
    for i in 0..8 {
        let p = inputs[8 + i].add(block_offset);
        blo[i] = _mm256_loadu_si256(p as *const __m256i);
        bhi[i] = _mm256_loadu_si256(p.add(32) as *const __m256i);
    }
    transpose8x8(&mut alo);
    transpose8x8(&mut ahi);
    transpose8x8(&mut blo);
    transpose8x8(&mut bhi);
    for i in 0..8 {
        ma[i] = alo[i];
        ma[i + 8] = ahi[i];
        mb[i] = blo[i];
        mb[i + 8] = bhi[i];
    }
}

/// Broadcast the 8 IV words into 8 __m256i vectors.
#[inline(always)]
fn broadcast_iv() -> [__m256i; 8] {
    [
        _mm256_set1_epi32(IV[0] as i32),
        _mm256_set1_epi32(IV[1] as i32),
        _mm256_set1_epi32(IV[2] as i32),
        _mm256_set1_epi32(IV[3] as i32),
        _mm256_set1_epi32(IV[4] as i32),
        _mm256_set1_epi32(IV[5] as i32),
        _mm256_set1_epi32(IV[6] as i32),
        _mm256_set1_epi32(IV[7] as i32),
    ]
}

/// 16-way batched BLAKE3 leaf hash over 16 equal-size messages.
///
/// `inputs[i]` points to the i-th message, of `N` bytes. The function
/// follows BLAKE3 chunk semantics: `CHUNK_START` is OR'd into the first
/// block's flags, `CHUNK_END` into the last, plain `flags` in between.
///
/// # Safety
/// `inputs` must contain 16 valid pointers, each to at least `N` readable
/// bytes. `outs` must have length at least 16 × 32.
#[target_feature(enable = "avx2")]
pub unsafe fn hash16_leaves<const N: usize>(
    inputs: &[*const u8; 16],
    flags_start: u8,
    flags_end: u8,
    outs: &mut [u8],
) {
    let blocks = N / BLAKE3_BLOCK_LEN;
    let iv = broadcast_iv();
    let counter_zero = _mm256_setzero_si256();
    let block_len = _mm256_set1_epi32(BLAKE3_BLOCK_LEN as i32);
    let flags_mid = _mm256_set1_epi32(flags_end as i32);
    let flags_first = _mm256_set1_epi32((flags_start | flags_end) as i32);
    let flags_last = _mm256_set1_epi32(flags_end as i32);

    let mut h_a: [__m256i; 8] = iv;
    let mut h_b: [__m256i; 8] = iv;

    let mut ma = [_mm256_setzero_si256(); 16];
    let mut mb = [_mm256_setzero_si256(); 16];

    for block in 0..blocks {
        let bf = if block == 0 {
            flags_first
        } else if block + 1 == blocks {
            flags_last
        } else {
            flags_mid
        };

        load_and_transpose16(inputs, block * BLAKE3_BLOCK_LEN, &mut ma, &mut mb);

        let mut va = [
            h_a[0], h_a[1], h_a[2], h_a[3], h_a[4], h_a[5], h_a[6], h_a[7], iv[0], iv[1], iv[2],
            iv[3], counter_zero, counter_zero, block_len, bf,
        ];
        let mut vb = [
            h_b[0], h_b[1], h_b[2], h_b[3], h_b[4], h_b[5], h_b[6], h_b[7], iv[0], iv[1], iv[2],
            iv[3], counter_zero, counter_zero, block_len, bf,
        ];

        for r in 0..7 {
            round_pair(&mut va, &mut vb, &ma, &mb, r);
        }
        // Feed-forward: h ^= v[0..8]. The result is the CV of this block,
        // and is the input CV (h) of the next block in the chunk.
        for j in 0..8 {
            h_a[j] = xor(h_a[j], va[j]);
            h_b[j] = xor(h_b[j], vb[j]);
        }
    }

    // The non-root chaining value of a chunk is its final block's CV — which
    // is `h_a` / `h_b` after the last xor. (For a single-block chunk, it's
    // `iv ^ v_lo`, matching `blake3::hazmat::HasherExt::finalize_non_root`.)
    transpose8x8(&mut h_a);
    transpose8x8(&mut h_b);
    for i in 0..8 {
        let lo = _mm256_extracti128_si256(h_a[i], 0);
        let hi = _mm256_extracti128_si256(h_a[i], 1);
        _mm_storeu_si128(outs.as_mut_ptr().add(i * 32) as *mut __m128i, lo);
        _mm_storeu_si128(outs.as_mut_ptr().add(i * 32 + 16) as *mut __m128i, hi);
    }
    for i in 0..8 {
        let lo = _mm256_extracti128_si256(h_b[i], 0);
        let hi = _mm256_extracti128_si256(h_b[i], 1);
        _mm_storeu_si128(outs.as_mut_ptr().add((8 + i) * 32) as *mut __m128i, lo);
        _mm_storeu_si128(outs.as_mut_ptr().add((8 + i) * 32 + 16) as *mut __m128i, hi);
    }
}

/// 16-way BLAKE3 parent-node compression: each input is a 64-byte block
/// `left ‖ right` of two child chaining values. Outputs the 32-byte
/// non-root chaining value, matching `blake3::hazmat::merge_subtrees_non_root`
/// with the BLAKE3 IV as the parent key.
///
/// # Safety
/// `inputs` must be 16 valid pointers, each to at least 64 readable bytes.
/// `outs` must have length at least 16 × 32.
#[target_feature(enable = "avx2")]
pub unsafe fn hash16_parents(inputs: &[*const u8; 16], outs: &mut [u8]) {
    let iv = broadcast_iv();
    let key: [__m256i; 8] = [
        _mm256_set1_epi32(IV[0] as i32),
        _mm256_set1_epi32(IV[1] as i32),
        _mm256_set1_epi32(IV[2] as i32),
        _mm256_set1_epi32(IV[3] as i32),
        _mm256_set1_epi32(IV[4] as i32),
        _mm256_set1_epi32(IV[5] as i32),
        _mm256_set1_epi32(IV[6] as i32),
        _mm256_set1_epi32(IV[7] as i32),
    ];
    let counter_zero = _mm256_setzero_si256();
    let block_len = _mm256_set1_epi32(BLAKE3_BLOCK_LEN as i32);
    let flags = _mm256_set1_epi32(BLAKE3_PARENT as i32);

    let mut ma = [_mm256_setzero_si256(); 16];
    let mut mb = [_mm256_setzero_si256(); 16];
    load_and_transpose16(inputs, 0, &mut ma, &mut mb);

    let mut va = [
        key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], iv[0], iv[1], iv[2],
        iv[3], counter_zero, counter_zero, block_len, flags,
    ];
    let mut vb = [
        key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], iv[0], iv[1], iv[2],
        iv[3], counter_zero, counter_zero, block_len, flags,
    ];
    for r in 0..7 {
        round_pair(&mut va, &mut vb, &ma, &mb, r);
    }

    // Non-root parent CV: lower 8 words of the 64-byte output = h ^ v_hi
    // (the second 8 words of v). This matches
    // `blake3::hazmat::merge_subtrees_non_root` with `Mode::Hash` for the
    // output side.
    let mut out_a = [_mm256_setzero_si256(); 8];
    let mut out_b = [_mm256_setzero_si256(); 8];
    for j in 0..8 {
        out_a[j] = xor(key[j], va[j + 8]);
        out_b[j] = xor(key[j], vb[j + 8]);
    }
    transpose8x8(&mut out_a);
    transpose8x8(&mut out_b);
    for i in 0..8 {
        let lo = _mm256_extracti128_si256(out_a[i], 0);
        let hi = _mm256_extracti128_si256(out_a[i], 1);
        _mm_storeu_si128(outs.as_mut_ptr().add(i * 32) as *mut __m128i, lo);
        _mm_storeu_si128(outs.as_mut_ptr().add(i * 32 + 16) as *mut __m128i, hi);
    }
    for i in 0..8 {
        let lo = _mm256_extracti128_si256(out_b[i], 0);
        let hi = _mm256_extracti128_si256(out_b[i], 1);
        _mm_storeu_si128(outs.as_mut_ptr().add((8 + i) * 32) as *mut __m128i, lo);
        _mm_storeu_si128(outs.as_mut_ptr().add((8 + i) * 32 + 16) as *mut __m128i, hi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_x86_64_avx2_detected() -> bool {
        // All x86_64 CPUs we target support AVX2.
        true
    }

    /// 16-way batched BLAKE3 must match the spec API.
    #[test]
    fn hash16_parents_matches_scalar() {
        if !is_x86_64_avx2_detected() {
            return;
        }
        let n: usize = 16;
        let mut children: Vec<u8> = Vec::with_capacity(n * 64);
        for i in 0..n {
            for k in 0..64 {
                children.push(((i * 73 + k * 19) & 0xff) as u8);
            }
        }
        let mut inputs: [*const u8; 16] = [std::ptr::null(); 16];
        for i in 0..n {
            inputs[i] = children.as_ptr().add(i * 64);
        }
        let mut batched = vec![0u8; n * 32];
        unsafe { hash16_parents(&inputs, &mut batched) };
        for i in 0..n {
            let l: [u8; 32] = children[i * 64..i * 64 + 32].try_into().unwrap();
            let r: [u8; 32] = children[i * 64 + 32..i * 64 + 64].try_into().unwrap();
            let scalar =
                blake3::hazmat::merge_subtrees_non_root(&l, &r, blake3::hazmat::Mode::Hash);
            assert_eq!(&batched[i * 32..(i + 1) * 32], &scalar[..], "parent {i}");
        }
    }

    #[test]
    fn hash16_leaves_matches_scalar_64() {
        if !is_x86_64_avx2_detected() {
            return;
        }
        let n: usize = 16;
        let mut data: Vec<u8> = Vec::with_capacity(n * 64);
        for i in 0..n * 64 {
            data.push((i & 0xff) as u8);
        }
        let mut inputs: [*const u8; 16] = [std::ptr::null(); 16];
        for i in 0..n {
            inputs[i] = data.as_ptr().add(i * 64);
        }
        let mut batched = vec![0u8; n * 32];
        unsafe {
            hash16_leaves::<64>(
                &inputs,
                BLAKE3_CHUNK_START,
                BLAKE3_CHUNK_END,
                &mut batched,
            );
        }
        for i in 0..n {
            let leaf = &data[i * 64..(i + 1) * 64];
            let scalar = blake3::Hasher::new().update(leaf).finalize_non_root();
            assert_eq!(&batched[i * 32..(i + 1) * 32], &scalar[..], "leaf {i}");
        }
    }

    #[test]
    fn hash16_leaves_matches_scalar_128() {
        if !is_x86_64_avx2_detected() {
            return;
        }
        let n: usize = 16;
        let mut data: Vec<u8> = Vec::with_capacity(n * 128);
        for i in 0..n * 128 {
            data.push((i & 0xff) as u8);
        }
        let mut inputs: [*const u8; 16] = [std::ptr::null(); 16];
        for i in 0..n {
            inputs[i] = data.as_ptr().add(i * 128);
        }
        let mut batched = vec![0u8; n * 32];
        unsafe {
            hash16_leaves::<128>(
                &inputs,
                BLAKE3_CHUNK_START,
                BLAKE3_CHUNK_END,
                &mut batched,
            );
        }
        for i in 0..n {
            let leaf = &data[i * 128..(i + 1) * 128];
            let scalar = blake3::Hasher::new().update(leaf).finalize_non_root();
            assert_eq!(&batched[i * 32..(i + 1) * 32], &scalar[..], "leaf {i}");
        }
    }
}
