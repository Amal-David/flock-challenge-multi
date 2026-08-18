use super::super::{F128, build_sum_table};

/// x86 single-matrix inner kernel — SSE2 mirror of
/// [`process_block_neon_single`]. Sweeps `TILE_T = 8` stripes for one
/// `BLOCK_K = 8` block of i_inner positions, keeping all 8 F128 accumulators in
/// xmm registers so the per-tile output is read/written once (vs once per
/// stripe in the untiled [`partial_fold_packed_z_fast_padded`] path).
///
/// # Safety
/// - `tile_bytes_ptr` must point to at least `TILE_T * k` bytes.
/// - `tables_ptr` must point to at least `TILE_T * 256 * 16` bytes.
/// - `out_ptr` must point to at least 8 F128 (128 bytes) of mutable storage.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn process_block_x86(
    tile_bytes_ptr: *const u8,
    k: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    use core::arch::x86_64::*;
    const TILE_T: usize = 8;
    // SAFETY: caller upholds the pointer/length contract documented above; SSE2
    // is baseline on x86_64.
    unsafe {
        let o = out_ptr as *mut u8;
        let mut a0 = _mm_loadu_si128(o as *const __m128i);
        let mut a1 = _mm_loadu_si128(o.add(16) as *const __m128i);
        let mut a2 = _mm_loadu_si128(o.add(32) as *const __m128i);
        let mut a3 = _mm_loadu_si128(o.add(48) as *const __m128i);
        let mut a4 = _mm_loadu_si128(o.add(64) as *const __m128i);
        let mut a5 = _mm_loadu_si128(o.add(80) as *const __m128i);
        let mut a6 = _mm_loadu_si128(o.add(96) as *const __m128i);
        let mut a7 = _mm_loadu_si128(o.add(112) as *const __m128i);
        for t in 0..TILE_T {
            let stripe_ptr = tile_bytes_ptr.add(t * k + bs);
            let ta = tables_ptr.add(t * 256 * 16);
            let i0 = *stripe_ptr as usize;
            let i1 = *stripe_ptr.add(1) as usize;
            let i2 = *stripe_ptr.add(2) as usize;
            let i3 = *stripe_ptr.add(3) as usize;
            let i4 = *stripe_ptr.add(4) as usize;
            let i5 = *stripe_ptr.add(5) as usize;
            let i6 = *stripe_ptr.add(6) as usize;
            let i7 = *stripe_ptr.add(7) as usize;
            a0 = _mm_xor_si128(a0, _mm_loadu_si128(ta.add(i0 * 16) as *const __m128i));
            a1 = _mm_xor_si128(a1, _mm_loadu_si128(ta.add(i1 * 16) as *const __m128i));
            a2 = _mm_xor_si128(a2, _mm_loadu_si128(ta.add(i2 * 16) as *const __m128i));
            a3 = _mm_xor_si128(a3, _mm_loadu_si128(ta.add(i3 * 16) as *const __m128i));
            a4 = _mm_xor_si128(a4, _mm_loadu_si128(ta.add(i4 * 16) as *const __m128i));
            a5 = _mm_xor_si128(a5, _mm_loadu_si128(ta.add(i5 * 16) as *const __m128i));
            a6 = _mm_xor_si128(a6, _mm_loadu_si128(ta.add(i6 * 16) as *const __m128i));
            a7 = _mm_xor_si128(a7, _mm_loadu_si128(ta.add(i7 * 16) as *const __m128i));
        }
        _mm_storeu_si128(o as *mut __m128i, a0);
        _mm_storeu_si128(o.add(16) as *mut __m128i, a1);
        _mm_storeu_si128(o.add(32) as *mut __m128i, a2);
        _mm_storeu_si128(o.add(48) as *mut __m128i, a3);
        _mm_storeu_si128(o.add(64) as *mut __m128i, a4);
        _mm_storeu_si128(o.add(80) as *mut __m128i, a5);
        _mm_storeu_si128(o.add(96) as *mut __m128i, a6);
        _mm_storeu_si128(o.add(112) as *mut __m128i, a7);
    }
}

/// x86 tiled single-matrix partial fold — `TILE_T × BLOCK_K` register tile
/// mirror of `partial_fold_packed_z_neon_single_padded`. Replaces the untiled
/// full-k accumulator (streamed once per stripe ≈ `n_stripes × k` of acc
/// traffic) with per-tile register accumulators, cutting acc traffic ≈ `TILE_T`×
/// — the dominant cost of this bandwidth-bound fold on many-core hosts.
#[cfg(target_arch = "x86_64")]
pub fn partial_fold_packed_z_x86_tiled_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = 8;
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

/// Outer(tile)-partitioned sibling of [`partial_fold_packed_z_x86_tiled_padded`]
/// — builds each tile's sum-tables once instead of per worker/chunk.
#[cfg(target_arch = "x86_64")]
pub fn partial_fold_packed_z_x86_oblock_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = 8;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 6, "need n_outer >= 64 for tile of 8 stripes");
    assert!(k_log >= 3, "need k >= 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_tiles = n_stripes / TILE_T;

    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);
    if useful == 0 {
        return vec![F128::ZERO; k];
    }

    let p = rayon::current_num_threads().max(1);
    let tiles_per_worker = n_tiles.div_ceil(p);
    let n_workers = n_tiles.div_ceil(tiles_per_worker);

    let mut partials = vec![F128::ZERO; n_workers * k];
    partials
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(w, partial)| {
            let tile_lo = w * tiles_per_worker;
            let tile_hi = ((w + 1) * tiles_per_worker).min(n_tiles);
            let mut tables = vec![F128::ZERO; TILE_T * 256];
            for tile in tile_lo..tile_hi {
                let stripe_base = tile * TILE_T;
                for t in 0..TILE_T {
                    let eq_off = 8 * (stripe_base + t);
                    build_sum_table(
                        &eq_outer[eq_off..eq_off + 8],
                        &mut tables[t * 256..(t + 1) * 256],
                    );
                }
                let tables_ptr = tables.as_ptr() as *const u8;
                let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k) };
                let mut bs = 0usize;
                while bs < useful {
                    unsafe {
                        process_block_x86(
                            z_base,
                            k,
                            bs,
                            tables_ptr,
                            partial.as_mut_ptr().add(bs),
                        );
                    }
                    bs += BLOCK_K;
                }
            }
        });

    let (first, rest) = partials.split_at(k);
    let mut out = first.to_vec();
    for chunk in rest.chunks(k) {
        out.par_iter_mut()
            .zip(chunk.par_iter())
            .for_each(|(o, s)| *o += *s);
    }
    out
}

/// All-lane gather and bit-transpose for one full 8-stripe tile chunk on x86_64.
/// Lane (t, r) is read at `src + (8*t + r) * stride` (F128 elements).
/// Output layout: row `t` at `out + 128*t`.
/// # Safety
/// - `src` must expose all 64 lane reads in bounds.
/// - `out` must point to at least 8 * 128 = 1024 mutable bytes.
#[cfg(target_arch = "x86_64")]
pub unsafe fn gather_transpose_tile_x86(src: *const F128, stride: usize, out: *mut u8) {
    let stride_bytes = stride * 16;
    let mut p = src as *const u8;
    let mut row = out;
    for _t in 0..8 {
        let lanes: [F128; 8] = [
            (p as *const F128).read_unaligned(),
            (p.add(stride_bytes) as *const F128).read_unaligned(),
            (p.add(2 * stride_bytes) as *const F128).read_unaligned(),
            (p.add(3 * stride_bytes) as *const F128).read_unaligned(),
            (p.add(4 * stride_bytes) as *const F128).read_unaligned(),
            (p.add(5 * stride_bytes) as *const F128).read_unaligned(),
            (p.add(6 * stride_bytes) as *const F128).read_unaligned(),
            (p.add(7 * stride_bytes) as *const F128).read_unaligned(),
        ];
        super::super::transpose_8_f128s_to_128_bytes(&lanes, core::slice::from_raw_parts_mut(row, 128));
        p = p.add(8 * stride_bytes);
        row = row.add(128);
    }
}

/// Two-stream leaf for x86_64: fold 8 stripe tables into two adjacent 8-column accumulator blocks.
#[cfg(target_arch = "x86_64")]
pub fn fold_block_major_chunk_x86_x2(
    transposed: &[u8; 8 * 128],
    tables: &[F128],
    group: &mut [F128],
    chunk_bits: usize,
) -> usize {
    const TILE_T: usize = 8;
    assert_eq!(tables.len(), TILE_T * 256);
    let b_pairs = chunk_bits / 16;
    let b_done = b_pairs * 16;
    let tables_ptr = tables.as_ptr() as *const u8;
    for pair in 0..b_pairs {
        let b0 = pair * 16;
        let bytes0 = unsafe { transposed.as_ptr().add(b0) };
        let acc_ptr = unsafe { group.as_mut_ptr().add(b0) };
        unsafe {
            fold_block_major_bpair_x2_x86(bytes0, tables_ptr, acc_ptr);
        }
    }
    b_done
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fold_seg8_x86(
    idx_ptr: *const u8,
    table_ptr: *const u8,
    a0: &mut core::arch::x86_64::__m128i,
    a1: &mut core::arch::x86_64::__m128i,
    a2: &mut core::arch::x86_64::__m128i,
    a3: &mut core::arch::x86_64::__m128i,
    a4: &mut core::arch::x86_64::__m128i,
    a5: &mut core::arch::x86_64::__m128i,
    a6: &mut core::arch::x86_64::__m128i,
    a7: &mut core::arch::x86_64::__m128i,
) {
    use core::arch::x86_64::*;
    unsafe {
        let bits = (idx_ptr as *const u64).read_unaligned();
        let i0 = (bits & 0xff) as usize;
        let i1 = ((bits >> 8) & 0xff) as usize;
        let i2 = ((bits >> 16) & 0xff) as usize;
        let i3 = ((bits >> 24) & 0xff) as usize;
        let i4 = ((bits >> 32) & 0xff) as usize;
        let i5 = ((bits >> 40) & 0xff) as usize;
        let i6 = ((bits >> 48) & 0xff) as usize;
        let i7 = (bits >> 56) as usize;
        *a0 = _mm_xor_si128(*a0, _mm_loadu_si128(table_ptr.add(i0 << 4) as *const __m128i));
        *a1 = _mm_xor_si128(*a1, _mm_loadu_si128(table_ptr.add(i1 << 4) as *const __m128i));
        *a2 = _mm_xor_si128(*a2, _mm_loadu_si128(table_ptr.add(i2 << 4) as *const __m128i));
        *a3 = _mm_xor_si128(*a3, _mm_loadu_si128(table_ptr.add(i3 << 4) as *const __m128i));
        *a4 = _mm_xor_si128(*a4, _mm_loadu_si128(table_ptr.add(i4 << 4) as *const __m128i));
        *a5 = _mm_xor_si128(*a5, _mm_loadu_si128(table_ptr.add(i5 << 4) as *const __m128i));
        *a6 = _mm_xor_si128(*a6, _mm_loadu_si128(table_ptr.add(i6 << 4) as *const __m128i));
        *a7 = _mm_xor_si128(*a7, _mm_loadu_si128(table_ptr.add(i7 << 4) as *const __m128i));
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn fold_block_major_bpair_x2_x86(
    bytes0: *const u8,
    tables_ptr: *const u8,
    acc_ptr: *mut F128,
) {
    use core::arch::x86_64::*;
    unsafe {
        let o = acc_ptr as *mut u8;
        let mut a0 = _mm_loadu_si128(o as *const __m128i);
        let mut a1 = _mm_loadu_si128(o.add(16) as *const __m128i);
        let mut a2 = _mm_loadu_si128(o.add(32) as *const __m128i);
        let mut a3 = _mm_loadu_si128(o.add(48) as *const __m128i);
        let mut a4 = _mm_loadu_si128(o.add(64) as *const __m128i);
        let mut a5 = _mm_loadu_si128(o.add(80) as *const __m128i);
        let mut a6 = _mm_loadu_si128(o.add(96) as *const __m128i);
        let mut a7 = _mm_loadu_si128(o.add(112) as *const __m128i);

        let mut c0 = _mm_loadu_si128(o.add(128) as *const __m128i);
        let mut c1 = _mm_loadu_si128(o.add(144) as *const __m128i);
        let mut c2 = _mm_loadu_si128(o.add(160) as *const __m128i);
        let mut c3 = _mm_loadu_si128(o.add(176) as *const __m128i);
        let mut c4 = _mm_loadu_si128(o.add(192) as *const __m128i);
        let mut c5 = _mm_loadu_si128(o.add(208) as *const __m128i);
        let mut c6 = _mm_loadu_si128(o.add(224) as *const __m128i);
        let mut c7 = _mm_loadu_si128(o.add(240) as *const __m128i);

        for t in 0..8 {
            let idx_a = bytes0.add(t * 128);
            let idx_c = idx_a.add(8);
            let ta = tables_ptr.add(t * 256 * 16);
            fold_seg8_x86(
                idx_a, ta, &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
            );
            fold_seg8_x86(
                idx_c, ta, &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut c7,
            );
        }

        _mm_storeu_si128(o as *mut __m128i, a0);
        _mm_storeu_si128(o.add(16) as *mut __m128i, a1);
        _mm_storeu_si128(o.add(32) as *mut __m128i, a2);
        _mm_storeu_si128(o.add(48) as *mut __m128i, a3);
        _mm_storeu_si128(o.add(64) as *mut __m128i, a4);
        _mm_storeu_si128(o.add(80) as *mut __m128i, a5);
        _mm_storeu_si128(o.add(96) as *mut __m128i, a6);
        _mm_storeu_si128(o.add(112) as *mut __m128i, a7);

        _mm_storeu_si128(o.add(128) as *mut __m128i, c0);
        _mm_storeu_si128(o.add(144) as *mut __m128i, c1);
        _mm_storeu_si128(o.add(160) as *mut __m128i, c2);
        _mm_storeu_si128(o.add(176) as *mut __m128i, c3);
        _mm_storeu_si128(o.add(192) as *mut __m128i, c4);
        _mm_storeu_si128(o.add(208) as *mut __m128i, c5);
        _mm_storeu_si128(o.add(224) as *mut __m128i, c6);
        _mm_storeu_si128(o.add(240) as *mut __m128i, c7);
    }
}
