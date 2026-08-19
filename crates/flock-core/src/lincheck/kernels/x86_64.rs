use super::super::{F128, build_sum_table};

/// GFNI twin of [`partial_fold_packed_z_x86_tiled_padded`]: each stripe's
/// 256-entry sum table is F2-linear (`T[0] = 0`, XOR-composed from the eight
/// `eq_outer` values), so it IS sixteen 8×8 bit matrices, and
/// `VGF2P8AFFINEQB` folds 64 output columns per instruction with no table
/// loads and no table build. Accumulation is byte-plane-major (plane `k` of
/// 64 columns per ZMM, sixteen independent chains) and transposes back to
/// F128s once after the parallel reduce — the same XOR terms reassociated,
/// so the result is bit-identical.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
pub fn partial_fold_packed_z_x86_gfni_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 6, "need n_outer ≥ 64 for tile of 8 stripes");
    assert!(k_log >= 6, "GFNI fold needs 64-column blocks");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    // Columns past useful_bits hold zero padding: affine(0) = 0 contributes
    // nothing, so cover only the 64-column blocks that touch useful bits.
    let n_blocks64 = useful_bits.div_ceil(64).min(k / 64);

    let n_tiles = n_stripes / TILE_T;
    let tiles_per_chunk = (n_tiles / 256).max(1);
    let bytes_per_chunk = tiles_per_chunk * TILE_T * k;

    let planes = z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![0u8; k * 16],
            |mut out_planes, (chunk_idx, chunk_bytes)| {
                let tile_start = chunk_idx * tiles_per_chunk;
                let n_tiles_in_chunk = chunk_bytes.len() / (TILE_T * k);
                let mut mats = [0u64; TILE_T * 16];
                for tile_rel in 0..n_tiles_in_chunk {
                    let tile_idx = tile_start + tile_rel;
                    let stripe_base = tile_idx * TILE_T;
                    for t in 0..TILE_T {
                        let eq_off = 8 * (stripe_base + t);
                        fold_mats_from_basis(
                            &eq_outer[eq_off..eq_off + 8],
                            &mut mats[t * 16..(t + 1) * 16],
                        );
                    }
                    // SAFETY: tile_rel < n_tiles_in_chunk keeps the tile in
                    // bounds; the block loop stays within k columns and the
                    // plane buffer is k*16 bytes.
                    unsafe {
                        gfni_fold_tile(
                            chunk_bytes.as_ptr().add(tile_rel * TILE_T * k),
                            k,
                            n_blocks64,
                            &mats,
                            out_planes.as_mut_ptr(),
                        );
                    }
                }
                out_planes
            },
        )
        .reduce(
            || vec![0u8; k * 16],
            |mut a, b| {
                // Plane XOR merge — same sums, still plane-major.
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x ^= *y;
                }
                a
            },
        );

    // One transpose back to F128 columns at the very end.
    let mut out = vec![F128::ZERO; k];
    for b in 0..k / 64 {
        let base = b * 1024;
        for col in 0..64 {
            let mut lo = 0u64;
            let mut hi = 0u64;
            for byte in 0..8 {
                lo |= (planes[base + byte * 64 + col] as u64) << (8 * byte);
            }
            for byte in 8..16 {
                hi |= (planes[base + byte * 64 + col] as u64) << (8 * (byte - 8));
            }
            out[b * 64 + col] = F128 { lo, hi };
        }
    }
    out
}

/// The sixteen `VGF2P8AFFINEQB` matrices of one stripe's sum table, straight
/// from its eight `eq_outer` basis values (encoding: `out.bit[i] =
/// parity(byte[7-i] & in)`; input bit `j` ↔ stripe bit `j`, matching
/// `build_sum_table`'s `T[1 << j] = eq8[j]`).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
fn fold_mats_from_basis(eq8: &[F128], mats: &mut [u64]) {
    debug_assert_eq!(eq8.len(), 8);
    debug_assert_eq!(mats.len(), 16);
    for (byte_k, slot) in mats.iter_mut().enumerate() {
        let mut qword = 0u64;
        for i in 0..8 {
            let bit_index = 8 * byte_k + i;
            let mut row = 0u8;
            for (j, basis_val) in eq8.iter().enumerate() {
                let bit = if bit_index < 64 {
                    (basis_val.lo >> bit_index) & 1
                } else {
                    (basis_val.hi >> (bit_index - 64)) & 1
                };
                row |= (bit as u8) << j;
            }
            qword |= (row as u64) << (8 * (7 - i));
        }
        *slot = qword;
    }
}

/// One tile's GFNI sweep: for every 64-column block, sixteen byte-plane
/// accumulators fold the eight stripes' GFNI products (two per `vpternlogq`).
///
/// # Safety
/// - `tile_bytes_ptr` must point to at least `8 * k` bytes.
/// - `mats` holds the tile's 8×16 matrices.
/// - `out_planes_ptr` must point to at least `n_blocks64 * 1024` bytes.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,gfni")]
unsafe fn gfni_fold_tile(
    tile_bytes_ptr: *const u8,
    k: usize,
    n_blocks64: usize,
    mats: &[u64; 128],
    out_planes_ptr: *mut u8,
) {
    use core::arch::x86_64::*;
    // SAFETY: caller upholds the pointer/length contract above.
    unsafe {
        for block in 0..n_blocks64 {
            let bs = block * 64;
            let mut rows = [_mm512_setzero_si512(); 8];
            for (t, row) in rows.iter_mut().enumerate() {
                *row = _mm512_loadu_si512(tile_bytes_ptr.add(t * k + bs) as *const __m512i);
            }
            let planes = out_planes_ptr.add(block * 1024);
            for byte_k in 0..16 {
                let plane_ptr = planes.add(byte_k * 64) as *mut __m512i;
                let mut acc = _mm512_loadu_si512(plane_ptr as *const __m512i);
                for t in (0..8).step_by(2) {
                    let g0 = _mm512_gf2p8affine_epi64_epi8::<0>(
                        rows[t],
                        _mm512_set1_epi64(mats[t * 16 + byte_k] as i64),
                    );
                    let g1 = _mm512_gf2p8affine_epi64_epi8::<0>(
                        rows[t + 1],
                        _mm512_set1_epi64(mats[(t + 1) * 16 + byte_k] as i64),
                    );
                    acc = _mm512_ternarylogic_epi64::<0x96>(acc, g0, g1);
                }
                _mm512_storeu_si512(plane_ptr, acc);
            }
        }
    }
}

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

// ---------------------------------------------------------------------------
// Block-major fold: AVX-512 nibble-table accumulate.
//
// The ranked lincheck folds the block-major packed witness (2^18 blocks ×
// 128 F128) over the outer index. Per (tile of 8 stripes, 128-column chunk)
// the driver transposes each stripe's 8 lanes into 128 index bytes (byte b =
// the 8 blocks' bit b) and then, for every column b and stripe t, adds the
// 8-bit subset-sum `T_t[byte]` into `partial[b]`. Today that inner loop is
// scalar: 8 stripes × 128 columns = 1,024 byte-indexed 16-byte gathers per
// (tile, chunk), ~5·10^8 per ranked prove.
//
// This kernel keeps the same arithmetic but performs it eight columns wide:
// the 256-entry table is never built; instead the subset sum splits exactly
// into a low-nibble table `TL_t[l] = Σ_{r∈l} eq_t[r]` (r < 4) and a
// high-nibble table `TH_t[h] = Σ_{r∈h} eq_t[4+r]`, and `T_t[byte] =
// TL_t[byte & 15] + TH_t[byte >> 4]` (GF(2^128) addition is XOR, so the
// split is exact). Each 16-entry table lives in registers as two zmm of
// qwords (lo halves) plus two zmm (hi halves); `vpermi2q` looks up eight
// columns' entries at once from the eight index nibbles. Accumulation is SoA
// (lo/hi qword vectors) over the eight stripes and interleaved back to the
// AoS `partial` once per eight columns. Same set of XORs as the scalar loop,
// in a different association — exact in a characteristic-2 field.
// ---------------------------------------------------------------------------

/// Nibble sum tables for one stripe (8 outer weights): `[TL lo(16), TL hi(16),
/// TH lo(16), TH hi(16)]` as qwords, i.e. `TL_t[l]` = `(lo[l], hi[l])`.
pub(crate) type NibbleTables = [u64; 64];

/// Build the lo/hi-nibble subset-sum tables from eight outer weights.
#[inline]
pub(crate) fn build_nibble_tables(eq8: &[F128; 8], out: &mut NibbleTables) {
    let mut tl = [F128::ZERO; 16];
    let mut th = [F128::ZERO; 16];
    for i in 0..4 {
        let (el, eh) = (eq8[i], eq8[4 + i]);
        let len = 1usize << i;
        for j in 0..len {
            tl[len + j] = tl[j] + el;
            th[len + j] = th[j] + eh;
        }
    }
    for i in 0..16 {
        out[i] = tl[i].lo;
        out[16 + i] = tl[i].hi;
        out[32 + i] = th[i].lo;
        out[48 + i] = th[i].hi;
    }
}

/// AVX-512 accumulate for one (full 8-stripe tile, 128-column chunk):
/// `partial[b] += Σ_t T_t[transposed[t*128 + b]]` for `b < chunk_bits`.
///
/// `transposed`: 8 rows × 128 index bytes; `nib`: the 8 stripes' nibble
/// tables; `partial`: at least `chunk_bits` F128 (AoS). Columns
/// `chunk_bits..128` of `transposed` are ignored (masked out at the store).
///
/// # Safety
/// Requires AVX-512F/BW at runtime (guaranteed by the cfg gate that compiles
/// this function in). All loads/stores are bounds-checked by the asserts.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn fold_block_major_chunk_x86_avx512(
    transposed: &[u8],
    nib: &[NibbleTables],
    partial: &mut [F128],
    chunk_bits: usize,
) {
    use core::arch::x86_64::*;
    const STRIPES: usize = 8;
    debug_assert_eq!(transposed.len(), STRIPES * 128);
    debug_assert_eq!(nib.len(), STRIPES);
    debug_assert!(chunk_bits <= 128 && partial.len() >= chunk_bits);
    let n_groups = chunk_bits.div_ceil(8);
    // SoA accumulators for the 128 columns: [lo qwords; 16 groups][hi; 16].
    let mut acc = [_mm512_setzero_si512(); 32];
    unsafe {
        let nib_mask = _mm512_set1_epi64(0xF);
        for t in 0..STRIPES {
            let tp = nib.as_ptr().add(t) as *const u64;
            let tl_lo0 = _mm512_loadu_si512(tp as *const __m512i);
            let tl_lo1 = _mm512_loadu_si512(tp.add(8) as *const __m512i);
            let tl_hi0 = _mm512_loadu_si512(tp.add(16) as *const __m512i);
            let tl_hi1 = _mm512_loadu_si512(tp.add(24) as *const __m512i);
            let th_lo0 = _mm512_loadu_si512(tp.add(32) as *const __m512i);
            let th_lo1 = _mm512_loadu_si512(tp.add(40) as *const __m512i);
            let th_hi0 = _mm512_loadu_si512(tp.add(48) as *const __m512i);
            let th_hi1 = _mm512_loadu_si512(tp.add(56) as *const __m512i);
            let row = transposed.as_ptr().add(t * 128);
            for g in 0..n_groups {
                let idx8 = _mm_loadl_epi64(row.add(g * 8) as *const __m128i);
                let idx = _mm512_cvtepu8_epi64(idx8);
                let n0 = _mm512_and_si512(idx, nib_mask);
                let n1 = _mm512_srli_epi64::<4>(idx);
                let lo = _mm512_xor_si512(
                    _mm512_permutex2var_epi64(tl_lo0, n0, tl_lo1),
                    _mm512_permutex2var_epi64(th_lo0, n1, th_lo1),
                );
                let hi = _mm512_xor_si512(
                    _mm512_permutex2var_epi64(tl_hi0, n0, tl_hi1),
                    _mm512_permutex2var_epi64(th_hi0, n1, th_hi1),
                );
                acc[g] = _mm512_xor_si512(acc[g], lo);
                acc[16 + g] = _mm512_xor_si512(acc[16 + g], hi);
            }
        }
        // Interleave SoA → AoS and XOR into `partial` (F128 = lo || hi LE).
        let idx0 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
        let idx1 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
        let base = partial.as_mut_ptr() as *mut u64;
        for g in 0..n_groups {
            let aos0 = _mm512_permutex2var_epi64(acc[g], idx0, acc[16 + g]);
            let aos1 = _mm512_permutex2var_epi64(acc[g], idx1, acc[16 + g]);
            let cols = (chunk_bits - g * 8).min(8);
            let p = base.add(g * 16);
            if cols == 8 {
                let p0 = p as *mut __m512i;
                let p1 = p.add(8) as *mut __m512i;
                _mm512_storeu_si512(p0, _mm512_xor_si512(_mm512_loadu_si512(p0), aos0));
                _mm512_storeu_si512(p1, _mm512_xor_si512(_mm512_loadu_si512(p1), aos1));
            } else {
                // Tail group: 2 qwords per column; aos0 covers columns 0..4,
                // aos1 columns 4..8 of this group.
                let q = 2 * cols; // qwords to touch
                let m0: __mmask8 = if q >= 8 { 0xFF } else { ((1u16 << q) - 1) as u8 };
                let m1: __mmask8 = if q <= 8 { 0 } else { ((1u16 << (q - 8)) - 1) as u8 };
                let pi = p as *mut i64;
                let v0 = _mm512_maskz_loadu_epi64(m0, pi);
                _mm512_mask_storeu_epi64(pi, m0, _mm512_xor_si512(v0, aos0));
                if m1 != 0 {
                    let v1 = _mm512_maskz_loadu_epi64(m1, pi.add(8));
                    _mm512_mask_storeu_epi64(pi.add(8), m1, _mm512_xor_si512(v1, aos1));
                }
            }
        }
    }
}
