use super::super::{F128, build_sum_table};

const TILE_T: usize = 8;

/// Runtime gate for the block-major x86 gather kernel.
/// `FLOCK_NO_LINCHECK_QFORM=1` is the A/B kill (same name as the Apple q-form
/// switch). The ranked worker's cleared environment never sets it.
#[inline]
pub fn lincheck_x86_tile_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LINCHECK_QFORM").is_none());
    *ON
}

/// One little-endian u64 → 8 byte indices. Bit-identical to 8 `ldrb`.
#[inline(always)]
unsafe fn qform_bytes(ptr: *const u8) -> [usize; 8] {
    let packed = unsafe { std::ptr::read_unaligned(ptr as *const u64) };
    [
        (packed as usize) & 0xff,
        (packed as usize >> 8) & 0xff,
        (packed as usize >> 16) & 0xff,
        (packed as usize >> 24) & 0xff,
        (packed as usize >> 32) & 0xff,
        (packed as usize >> 40) & 0xff,
        (packed as usize >> 48) & 0xff,
        (packed as usize >> 56) & 0xff,
    ]
}

/// x86 single-matrix inner kernel — SSE2 mirror of
/// [`process_block_neon_single`]. Sweeps `TILE_T = 8` stripes for one
/// `BLOCK_K = 8` block of i_inner positions, keeping all 8 F128 accumulators in
/// xmm registers so the per-tile output is read/written once (vs once per
/// stripe in the untiled [`partial_fold_packed_z_fast_padded`] path).
///
/// Block-major callers pass `k = 128` (transposed row stride). Stripe callers
/// pass the matrix `k`. Same XOR-of-table algebra either way.
///
/// # Safety
/// - `tile_bytes_ptr` must point to at least `TILE_T * k` bytes.
/// - `tables_ptr` must point to at least `TILE_T * 256 * 16` bytes, 16-aligned.
/// - `out_ptr` must point to at least 8 F128 (128 bytes) of 16-aligned storage.
#[inline]
unsafe fn process_block_x86(
    tile_bytes_ptr: *const u8,
    k: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    use core::arch::x86_64::*;
    // SAFETY: caller upholds the pointer/length contract documented above; SSE2
    // is baseline on x86_64. Acc / table bases are F128-aligned; index bytes
    // may be unaligned (q-form `read_unaligned`).
    unsafe {
        let o = out_ptr as *mut u8;
        let mut a0 = _mm_load_si128(o as *const __m128i);
        let mut a1 = _mm_load_si128(o.add(16) as *const __m128i);
        let mut a2 = _mm_load_si128(o.add(32) as *const __m128i);
        let mut a3 = _mm_load_si128(o.add(48) as *const __m128i);
        let mut a4 = _mm_load_si128(o.add(64) as *const __m128i);
        let mut a5 = _mm_load_si128(o.add(80) as *const __m128i);
        let mut a6 = _mm_load_si128(o.add(96) as *const __m128i);
        let mut a7 = _mm_load_si128(o.add(112) as *const __m128i);
        for t in 0..TILE_T {
            let stripe_ptr = tile_bytes_ptr.add(t * k + bs);
            let ta = tables_ptr.add(t * 256 * 16);
            let [i0, i1, i2, i3, i4, i5, i6, i7] = qform_bytes(stripe_ptr);
            a0 = _mm_xor_si128(a0, _mm_load_si128(ta.add(i0 * 16) as *const __m128i));
            a1 = _mm_xor_si128(a1, _mm_load_si128(ta.add(i1 * 16) as *const __m128i));
            a2 = _mm_xor_si128(a2, _mm_load_si128(ta.add(i2 * 16) as *const __m128i));
            a3 = _mm_xor_si128(a3, _mm_load_si128(ta.add(i3 * 16) as *const __m128i));
            a4 = _mm_xor_si128(a4, _mm_load_si128(ta.add(i4 * 16) as *const __m128i));
            a5 = _mm_xor_si128(a5, _mm_load_si128(ta.add(i5 * 16) as *const __m128i));
            a6 = _mm_xor_si128(a6, _mm_load_si128(ta.add(i6 * 16) as *const __m128i));
            a7 = _mm_xor_si128(a7, _mm_load_si128(ta.add(i7 * 16) as *const __m128i));
        }
        _mm_store_si128(o as *mut __m128i, a0);
        _mm_store_si128(o.add(16) as *mut __m128i, a1);
        _mm_store_si128(o.add(32) as *mut __m128i, a2);
        _mm_store_si128(o.add(48) as *mut __m128i, a3);
        _mm_store_si128(o.add(64) as *mut __m128i, a4);
        _mm_store_si128(o.add(80) as *mut __m128i, a5);
        _mm_store_si128(o.add(96) as *mut __m128i, a6);
        _mm_store_si128(o.add(112) as *mut __m128i, a7);
    }
}

/// Block-major fold of one (tile, 128-column chunk). Full tiles only
/// (`tile_stripes == 8`); the caller scalar-finishes `chunk_bits % 8`.
/// Calls [`process_block_x86`] with `k = 128` (transposed row stride) — not
/// the stripe walker [`partial_fold_packed_z_x86_tiled_padded`].
pub fn fold_block_major_chunk_x86(
    transposed: &[u8],
    tables: &[F128],
    partial: &mut [F128],
    chunk_bits: usize,
) -> usize {
    assert_eq!(transposed.len(), TILE_T * 128);
    assert_eq!(tables.len(), TILE_T * 256);
    assert!(chunk_bits <= 128);
    assert!(partial.len() >= chunk_bits);

    let full_blocks = chunk_bits / 8;
    let tables_ptr = tables.as_ptr() as *const u8;
    for block_idx in 0..full_blocks {
        let bs = block_idx * 8;
        // SAFETY: full tile, `bs + 8 ≤ chunk_bits ≤ 128`; transposed is
        // 8×128; tables is 8×256 F128; `partial[bs..bs+8]` is 16-aligned.
        unsafe {
            process_block_x86(
                transposed.as_ptr(),
                128,
                bs,
                tables_ptr,
                partial.as_mut_ptr().add(bs),
            );
        }
    }
    full_blocks * 8
}

/// Gather 8×8 strided F128s and bit-transpose each stripe into 128 bytes.
///
/// Lane `(t, r)` is read at `src + (8·t + r) · stride`. Output byte `b` of
/// row `t` has bit `r` equal to bit `b` of that lane — bit-identical to
/// [`super::super::transpose_8_f128s_to_128_bytes`].
///
/// Dispatch: AVX-512 VBMI if detected, else AVX2 unpack-pyramid, else scalar.
/// No `vpgather`. No compile-time `cfg(avx512*)` without a fallback.
///
/// # Safety
/// - All 64 lane addresses must be in-bounds 16-aligned F128s.
/// - `out` must point to `8 × 128` writable bytes.
#[inline(never)]
pub unsafe fn gather_transpose_tile_x86(src: *const F128, stride: usize, out: *mut u8) {
    enum Kind {
        Avx512,
        Avx2,
        Scalar,
    }
    static KIND: std::sync::LazyLock<Kind> = std::sync::LazyLock::new(|| {
        if is_x86_feature_detected!("avx512vbmi") {
            Kind::Avx512
        } else if is_x86_feature_detected!("avx2") {
            Kind::Avx2
        } else {
            Kind::Scalar
        }
    });
    match *KIND {
        Kind::Avx512 => unsafe { gather_transpose_tile_avx512(src, stride, out) },
        Kind::Avx2 => unsafe { gather_transpose_tile_avx2(src, stride, out) },
        Kind::Scalar => unsafe { gather_transpose_tile_scalar(src, stride, out) },
    }
}

/// Scalar oracle / last-resort fallback. Same stores as the lincheck.rs body.
unsafe fn gather_transpose_tile_scalar(src: *const F128, stride: usize, out: *mut u8) {
    for t in 0..TILE_T {
        let lanes: [F128; 8] = std::array::from_fn(|r| unsafe { *src.add((8 * t + r) * stride) });
        let row = unsafe { std::slice::from_raw_parts_mut(out.add(t * 128), 128) };
        let lo: [u64; 8] = std::array::from_fn(|r| lanes[r].lo);
        let hi: [u64; 8] = std::array::from_fn(|r| lanes[r].hi);
        let (out_lo, out_hi) = row.split_at_mut(64);
        crate::bits::transpose_8_u64s_to_64_bytes(&lo, out_lo);
        crate::bits::transpose_8_u64s_to_64_bytes(&hi, out_hi);
    }
}

/// Hacker's Delight 8×8 bit-swap rounds on 4 u64 lanes in one ymm.
/// `target_feature` (not `inline(always)`: rustc 1.97 forbids both) so this
/// cannot be emitted on a non-AVX2 TU / SIGILL path.
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn swap_rounds_ymm(mut y: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    let m1 = _mm256_set1_epi64x(0x00AA_00AA_00AA_00AAu64 as i64);
    let m2 = _mm256_set1_epi64x(0x0000_CCCC_0000_CCCCu64 as i64);
    let m3 = _mm256_set1_epi64x(0x0000_0000_F0F0_F0F0u64 as i64);
    let t = _mm256_and_si256(_mm256_xor_si256(y, _mm256_srli_epi64::<7>(y)), m1);
    y = _mm256_xor_si256(y, _mm256_xor_si256(t, _mm256_slli_epi64::<7>(t)));
    let t = _mm256_and_si256(_mm256_xor_si256(y, _mm256_srli_epi64::<14>(y)), m2);
    y = _mm256_xor_si256(y, _mm256_xor_si256(t, _mm256_slli_epi64::<14>(t)));
    let t = _mm256_and_si256(_mm256_xor_si256(y, _mm256_srli_epi64::<28>(y)), m3);
    _mm256_xor_si256(y, _mm256_xor_si256(t, _mm256_slli_epi64::<28>(t)))
}

/// 8×8 byte transpose of eight u64 rows (low 8 bytes of each xmm) via the
/// unpack pyramid, then 3 swap rounds on ymm. Columns land as `col0‖col1`
/// matching NEON IDX0–3. AVX2 `vpshufb` is 16-byte-lane only — this is not
/// a fake 64-byte tbl.
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn transpose_8x8_bytes_swaps(
    r0: core::arch::x86_64::__m128i,
    r1: core::arch::x86_64::__m128i,
    r2: core::arch::x86_64::__m128i,
    r3: core::arch::x86_64::__m128i,
    r4: core::arch::x86_64::__m128i,
    r5: core::arch::x86_64::__m128i,
    r6: core::arch::x86_64::__m128i,
    r7: core::arch::x86_64::__m128i,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    use core::arch::x86_64::*;
    unsafe {
        let a01 = _mm_unpacklo_epi8(r0, r1);
        let a23 = _mm_unpacklo_epi8(r2, r3);
        let a45 = _mm_unpacklo_epi8(r4, r5);
        let a67 = _mm_unpacklo_epi8(r6, r7);
        let b0123_lo = _mm_unpacklo_epi16(a01, a23);
        let b0123_hi = _mm_unpackhi_epi16(a01, a23);
        let b4567_lo = _mm_unpacklo_epi16(a45, a67);
        let b4567_hi = _mm_unpackhi_epi16(a45, a67);
        let c01 = _mm_unpacklo_epi32(b0123_lo, b4567_lo);
        let c23 = _mm_unpackhi_epi32(b0123_lo, b4567_lo);
        let c45 = _mm_unpacklo_epi32(b0123_hi, b4567_hi);
        let c67 = _mm_unpackhi_epi32(b0123_hi, b4567_hi);
        let y0 = swap_rounds_ymm(_mm256_set_m128i(c23, c01));
        let y1 = swap_rounds_ymm(_mm256_set_m128i(c67, c45));
        (y0, y1)
    }
}

/// AVX2 gather+transpose. 8 aligned loads, punpck uzp, unpack-pyramid, 3 ymm
/// swap rounds. Prefetch +256 B is hint-only.
#[target_feature(enable = "avx2")]
unsafe fn gather_transpose_tile_avx2(src: *const F128, stride: usize, out: *mut u8) {
    use core::arch::x86_64::*;
    let stride_bytes = stride * 16;
    let mut p = src as *const u8;
    let mut row = out;
    for _t in 0..TILE_T {
        // SAFETY: caller guarantees 8 in-bounds 16-aligned F128s at
        // `p + r * stride_bytes`. Prefetch is a hint (may be past the
        // allocation; same +256 B lead as the NEON `prfm`).
        unsafe {
            let w0 = _mm_load_si128(p as *const __m128i);
            let w1 = _mm_load_si128(p.add(stride_bytes) as *const __m128i);
            let w2 = _mm_load_si128(p.add(2 * stride_bytes) as *const __m128i);
            let w3 = _mm_load_si128(p.add(3 * stride_bytes) as *const __m128i);
            let w4 = _mm_load_si128(p.add(4 * stride_bytes) as *const __m128i);
            let w5 = _mm_load_si128(p.add(5 * stride_bytes) as *const __m128i);
            let w6 = _mm_load_si128(p.add(6 * stride_bytes) as *const __m128i);
            let w7 = _mm_load_si128(p.add(7 * stride_bytes) as *const __m128i);
            for r in 0..8 {
                _mm_prefetch(
                    p.add(r * stride_bytes + 256) as *const i8,
                    _MM_HINT_T0,
                );
            }

            // uzp: lo = vuzp1q, hi = vuzp2q.
            let lo01 = _mm_unpacklo_epi64(w0, w1);
            let hi01 = _mm_unpackhi_epi64(w0, w1);
            let lo23 = _mm_unpacklo_epi64(w2, w3);
            let hi23 = _mm_unpackhi_epi64(w2, w3);
            let lo45 = _mm_unpacklo_epi64(w4, w5);
            let hi45 = _mm_unpackhi_epi64(w4, w5);
            let lo67 = _mm_unpacklo_epi64(w6, w7);
            let hi67 = _mm_unpackhi_epi64(w6, w7);

            // Split pairs so the pyramid sees 8 individual u64 rows.
            let l0 = lo01;
            let l1 = _mm_unpackhi_epi64(lo01, lo01);
            let l2 = lo23;
            let l3 = _mm_unpackhi_epi64(lo23, lo23);
            let l4 = lo45;
            let l5 = _mm_unpackhi_epi64(lo45, lo45);
            let l6 = lo67;
            let l7 = _mm_unpackhi_epi64(lo67, lo67);
            let (ylo0, ylo1) = transpose_8x8_bytes_swaps(l0, l1, l2, l3, l4, l5, l6, l7);

            let h0 = hi01;
            let h1 = _mm_unpackhi_epi64(hi01, hi01);
            let h2 = hi23;
            let h3 = _mm_unpackhi_epi64(hi23, hi23);
            let h4 = hi45;
            let h5 = _mm_unpackhi_epi64(hi45, hi45);
            let h6 = hi67;
            let h7 = _mm_unpackhi_epi64(hi67, hi67);
            let (yhi0, yhi1) = transpose_8x8_bytes_swaps(h0, h1, h2, h3, h4, h5, h6, h7);

            _mm256_storeu_si256(row as *mut __m256i, ylo0);
            _mm256_storeu_si256(row.add(32) as *mut __m256i, ylo1);
            _mm256_storeu_si256(row.add(64) as *mut __m256i, yhi0);
            _mm256_storeu_si256(row.add(96) as *mut __m256i, yhi1);
        }
        p = unsafe { p.add(8 * stride_bytes) };
        row = unsafe { row.add(128) };
    }
}

/// AVX-512 VBMI gather+transpose. Same 8 aligned loads + punpck uzp, then
/// `_mm512_permutexvar_epi8` (the live `bit_transpose_64bytes_avx512` tbl)
/// and 3 swap rounds on one zmm. Only called after runtime detect.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn gather_transpose_tile_avx512(src: *const F128, stride: usize, out: *mut u8) {
    use core::arch::x86_64::*;
    #[rustfmt::skip]
    const IDX: [i8; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56,  1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,  3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,  5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,  7, 15, 23, 31, 39, 47, 55, 63,
    ];
    let stride_bytes = stride * 16;
    let mut p = src as *const u8;
    let mut row = out;
    let idx = unsafe { _mm512_loadu_si512(IDX.as_ptr() as *const __m512i) };
    let mask1 = _mm512_set1_epi64(0x00AA_00AA_00AA_00AAu64 as i64);
    let mask2 = _mm512_set1_epi64(0x0000_CCCC_0000_CCCCu64 as i64);
    let mask3 = _mm512_set1_epi64(0x0000_0000_F0F0_F0F0u64 as i64);
    for _t in 0..TILE_T {
        unsafe {
            let w0 = _mm_load_si128(p as *const __m128i);
            let w1 = _mm_load_si128(p.add(stride_bytes) as *const __m128i);
            let w2 = _mm_load_si128(p.add(2 * stride_bytes) as *const __m128i);
            let w3 = _mm_load_si128(p.add(3 * stride_bytes) as *const __m128i);
            let w4 = _mm_load_si128(p.add(4 * stride_bytes) as *const __m128i);
            let w5 = _mm_load_si128(p.add(5 * stride_bytes) as *const __m128i);
            let w6 = _mm_load_si128(p.add(6 * stride_bytes) as *const __m128i);
            let w7 = _mm_load_si128(p.add(7 * stride_bytes) as *const __m128i);
            for r in 0..8 {
                _mm_prefetch(
                    p.add(r * stride_bytes + 256) as *const i8,
                    _MM_HINT_T0,
                );
            }

            let lo01 = _mm_unpacklo_epi64(w0, w1);
            let hi01 = _mm_unpackhi_epi64(w0, w1);
            let lo23 = _mm_unpacklo_epi64(w2, w3);
            let hi23 = _mm_unpackhi_epi64(w2, w3);
            let lo45 = _mm_unpacklo_epi64(w4, w5);
            let hi45 = _mm_unpackhi_epi64(w4, w5);
            let lo67 = _mm_unpacklo_epi64(w6, w7);
            let hi67 = _mm_unpackhi_epi64(w6, w7);

            let mut zlo = _mm512_castsi128_si512(lo01);
            zlo = _mm512_inserti32x4::<1>(zlo, lo23);
            zlo = _mm512_inserti32x4::<2>(zlo, lo45);
            zlo = _mm512_inserti32x4::<3>(zlo, lo67);
            let mut zhi = _mm512_castsi128_si512(hi01);
            zhi = _mm512_inserti32x4::<1>(zhi, hi23);
            zhi = _mm512_inserti32x4::<2>(zhi, hi45);
            zhi = _mm512_inserti32x4::<3>(zhi, hi67);

            zlo = bit_transpose_zmm(zlo, idx, mask1, mask2, mask3);
            zhi = bit_transpose_zmm(zhi, idx, mask1, mask2, mask3);
            _mm512_storeu_si512(row as *mut __m512i, zlo);
            _mm512_storeu_si512(row.add(64) as *mut __m512i, zhi);
        }
        p = unsafe { p.add(8 * stride_bytes) };
        row = unsafe { row.add(128) };
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn bit_transpose_zmm(
    inp: core::arch::x86_64::__m512i,
    idx: core::arch::x86_64::__m512i,
    mask1: core::arch::x86_64::__m512i,
    mask2: core::arch::x86_64::__m512i,
    mask3: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;
    let mut y = _mm512_permutexvar_epi8(idx, inp);
    let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<7>(y)), mask1);
    y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<7>(t)));
    let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<14>(y)), mask2);
    y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<14>(t)));
    let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<28>(y)), mask3);
    _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<28>(t)))
}

/// x86 tiled single-matrix partial fold — `TILE_T × BLOCK_K` register tile
/// mirror of `partial_fold_packed_z_neon_single_padded`. Replaces the untiled
/// full-k accumulator (streamed once per stripe ≈ `n_stripes × k` of acc
/// traffic) with per-tile register accumulators, cutting acc traffic ≈ `TILE_T`×
/// — the dominant cost of this bandwidth-bound fold on many-core hosts.
///
/// This is the **stripe** path (`partial_fold_packed_z_best`). Ranked lincheck
/// is block-major and must not call this function.
pub fn partial_fold_packed_z_x86_tiled_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 6, "need n_outer ≥ 64 for tile of 8 stripes");
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_blocks_full = k / BLOCK_K;
    // Boundary block past useful_bits holds 0 padding → table[0] = 0 contributes
    // nothing, so cover only blocks that touch useful bits.
    let n_blocks = useful_bits.div_ceil(BLOCK_K).min(n_blocks_full);

    let n_tiles = n_stripes / TILE_T;
    let tiles_per_chunk = (n_tiles / 256).max(1);
    let bytes_per_chunk = tiles_per_chunk * TILE_T * k;

    z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![F128::ZERO; k],
            |mut out, (chunk_idx, chunk_bytes)| {
                let tile_start = chunk_idx * tiles_per_chunk;
                // TILE_T × 256 F128 = 32 KB tables. L1 resident.
                let mut tables = vec![F128::ZERO; TILE_T * 256];
                let n_tiles_in_chunk = chunk_bytes.len() / (TILE_T * k);
                for tile_rel in 0..n_tiles_in_chunk {
                    let tile_idx = tile_start + tile_rel;
                    let stripe_base = tile_idx * TILE_T;
                    // SAFETY: tile_rel < n_tiles_in_chunk so the offset stays in bounds.
                    let tile_bytes_ptr = unsafe { chunk_bytes.as_ptr().add(tile_rel * TILE_T * k) };
                    for t in 0..TILE_T {
                        let byte_idx = stripe_base + t;
                        let eq_off = 8 * byte_idx;
                        build_sum_table(
                            &eq_outer[eq_off..eq_off + 8],
                            &mut tables[t * 256..(t + 1) * 256],
                        );
                    }
                    let tables_ptr = tables.as_ptr() as *const u8;
                    for block_idx in 0..n_blocks {
                        let bs = block_idx * BLOCK_K;
                        // SAFETY: bs + BLOCK_K ≤ k ≤ out.len(); tile_bytes_ptr
                        // covers TILE_T*k bytes; tables_ptr covers TILE_T*256 F128.
                        unsafe {
                            process_block_x86(
                                tile_bytes_ptr,
                                k,
                                bs,
                                tables_ptr,
                                out.as_mut_ptr().add(bs),
                            );
                        }
                    }
                }
                out
            },
        )
        .reduce(
            || vec![F128::ZERO; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    fn scalar_gather_row(src: &[F128], stride: usize, t: usize, b: usize) -> u8 {
        let mut want = 0u8;
        for r in 0..8 {
            let lane = src[(8 * t + r) * stride];
            let bit = if b < 64 {
                (lane.lo >> b) & 1
            } else {
                (lane.hi >> (b - 64)) & 1
            };
            want |= (bit as u8) << r;
        }
        want
    }

    fn check_gather(got: &[u8], src: &[F128], stride: usize) {
        for t in 0..TILE_T {
            for b in 0..128 {
                assert_eq!(
                    got[t * 128 + b],
                    scalar_gather_row(src, stride, t, b),
                    "stride={stride} t={t} b={b}"
                );
            }
        }
    }

    /// Dispatch + each compiled backend must match the scalar bit definition:
    /// output row t, byte b, bit r = bit b of lane (t, r).
    #[test]
    fn gather_transpose_tile_matches_scalar() {
        let mut rng = Rng(0x72A5_0002);
        for &stride in &[1usize, 2, 7, 128] {
            let src: Vec<F128> = (0..64 * stride).map(|_| rng.f128()).collect();
            let mut got = [0u8; TILE_T * 128];
            unsafe {
                gather_transpose_tile_x86(src.as_ptr(), stride, got.as_mut_ptr());
            }
            check_gather(&got, &src, stride);

            if is_x86_feature_detected!("avx2") {
                let mut avx2 = [0u8; TILE_T * 128];
                unsafe {
                    gather_transpose_tile_avx2(src.as_ptr(), stride, avx2.as_mut_ptr());
                }
                check_gather(&avx2, &src, stride);
            }
            if is_x86_feature_detected!("avx512vbmi") {
                let mut avx512 = [0u8; TILE_T * 128];
                unsafe {
                    gather_transpose_tile_avx512(src.as_ptr(), stride, avx512.as_mut_ptr());
                }
                check_gather(&avx512, &src, stride);
            }
        }
    }

    /// Fold leaf vs the scalar `acc += tables[t*256 + byte]` body, including
    /// a padded (non-multiple-of-8) `chunk_bits` tail and non-zero start accs.
    #[test]
    fn fold_block_major_chunk_matches_scalar() {
        let mut rng = Rng(0x1EAF_0001);
        for chunk_bits in [1usize, 7, 8, 15, 16, 97, 128] {
            let transposed: Vec<u8> = (0..TILE_T * 128)
                .map(|_| rng.next_u64() as u8)
                .collect();
            let tables: Vec<F128> = (0..TILE_T * 256).map(|_| rng.f128()).collect();
            let start: Vec<F128> = (0..chunk_bits).map(|_| rng.f128()).collect();

            let mut got = start.clone();
            let b_done = fold_block_major_chunk_x86(&transposed, &tables, &mut got, chunk_bits);
            assert_eq!(b_done, (chunk_bits / 8) * 8);

            let mut want = start;
            for b in 0..chunk_bits {
                let mut acc = want[b];
                for t in 0..TILE_T {
                    let byte = transposed[t * 128 + b] as usize;
                    acc += tables[t * 256 + byte];
                }
                want[b] = acc;
            }
            // SIMD covers `[0, b_done)`; scalar tail is the caller's job, so
            // only the folded prefix is compared here. Finish the tail the
            // same way lincheck.rs does so a padded block still matches.
            for b in b_done..chunk_bits {
                let mut acc = got[b];
                for t in 0..TILE_T {
                    let byte = transposed[t * 128 + b] as usize;
                    acc += tables[t * 256 + byte];
                }
                got[b] = acc;
            }
            assert_eq!(got, want, "chunk_bits={chunk_bits}");
        }
    }
}
