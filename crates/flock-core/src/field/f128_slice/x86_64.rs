use crate::field::F128;

#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn vpclmulqdq_runtime() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(u8::MAX);
    // Cache the result of the runtime detection: hot F128 paths call this on
    // every fold, so the one-shot check has to be cheap. `u8::MAX` is the
    // "not yet computed" sentinel; `0/1` are the cached verdicts.
    match CACHE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    let res = unsafe { detect_vpclmulqdq() };
    CACHE.store(res as u8, Ordering::Relaxed);
    res
}

/// Probe CPUID for AVX-512F + VPCLMULQDQ. Returns true iff both are usable on
/// this core. VPCLMULQDQ lives in leaf 7 sub-leaf 0 EBX bit 10; AVX-512F
/// lives in the same word at bit 16. We do not require OS support (XCR0
/// checks): on a kernel that hasn't enabled AVX-512 we'll fault on the first
/// `_mm512_*` use, but that is consistent with the rest of the x86_64
/// kernel set which assumes the runtime has already enabled AVX-512 state.
///
/// # Safety
/// Pure CPUID — no side effects, no memory access.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn detect_vpclmulqdq() -> bool {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__cpuid_count, CpuidResult};
    // SAFETY: SSE4.1 is universally available on x86_64; CPUID is side-effect
    // free and uses no memory.
    unsafe {
        let leaf7: CpuidResult = __cpuid_count(7, 0);
        const AVX512F_BIT: u32 = 1 << 16;
        const VPCLMULQDQ_BIT: u32 = 1 << 10;
        (leaf7.ebx & AVX512F_BIT != 0) && (leaf7.ebx & VPCLMULQDQ_BIT != 0)
    }
}

// `_mm_prefetch` and `_mm512_*` are SSE/AVX-512 intrinsics; keep this whole
// module compilable when only the lower x86 baseline is enabled.
#[allow(dead_code)]
const _: () = (); // anchor for cfg-gated attributes below

/// L1-cache-blocked iteration window for the pair-fold kernels, in output
/// F128 (2 source F128 per output F128). 256 dst F128 per block = 4 KiB of
/// destination + 8 KiB of source, with the next block's 8 KiB of source
/// prefetched ahead: 20 KiB total working set, comfortably fitting a 32 KiB
/// L1d on Sapphire Rapids / Ice Lake / Zen 4. `FLOCK_NO_FOLD_PAIRS_PF=1`
/// removes the prefetch (the kernel is byte-identical either way — the
/// hints move no data of their own), `FLOCK_FOLD_PAIRS_PF=<n>` overrides the
/// distance.
const FOLD_PAIRS_BLOCK: usize = 256;
const FOLD_PAIRS_PF_AHEAD: usize = 256;

#[inline]
fn fold_pairs_block_size() -> usize {
    FOLD_PAIRS_BLOCK
}

#[inline]
fn fold_pairs_pf_ahead() -> usize {
    static D: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        if std::env::var_os("FLOCK_NO_FOLD_PAIRS_PF").is_some() {
            return 0;
        }
        std::env::var("FLOCK_FOLD_PAIRS_PF")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(FOLD_PAIRS_PF_AHEAD)
    });
    *D
}

/// Issue a sequence of L1-prefetch hints 64 bytes apart starting at `p`.
/// `count * 64` bytes total; the hints move no data of their own.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn prefetch_l1_lines(p: *const i8, count: usize) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
    // SAFETY: caller guarantees the address range is mapped readable; hints
    // never fault.
    unsafe {
        let mut l = 0usize;
        while l < count {
            _mm_prefetch::<_MM_HINT_T0>(p.add(l));
            l = l.wrapping_add(64);
        }
    }
}

/// Four-lane pair fold with L1-cache-blocked iteration and an inner loop
/// unrolled by 8 (32 output F128 per iteration). Runtime-detected
/// VPCLMULQDQ + AVX-512F are required — see [`vpclmulqdq_runtime`].
///
/// Block size is [`FOLD_PAIRS_BLOCK`] source F128 (= 256 output F128). Each
/// iteration prefetches one block of source ahead of the current read
/// window. The inner body is a straight unroll-by-8 of the four-lane body
/// of [`fold_pairs`]; folding correctness is identical (the iteration
/// order over the source slice is the same, just with 8 quads per step).
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F + VPCLMULQDQ and that
/// `base + dst.len() * 2 <= src.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_pairs_cached(
    src: &[F128],
    base: usize,
    dst: &mut [F128],
    r: F128,
) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let r_x64 = ghash_shift64_x4(r_bcast);
        let total = dst.len();
        let block = fold_pairs_block_size().min(total);
        let pf_ahead = fold_pairs_pf_ahead();
        let src_base_ptr = src.as_ptr();
        let src_len = src.len();
        let mut done = 0usize;
        while done < total {
            // Prefetch the block of source that the *next* chunk will read;
            // the current chunk's loads overlap the previous chunk's
            // prefetch so the L1d demand stream stays fed. `pf_ahead` is
            // measured in dst F128 (= 2 src F128).
            if pf_ahead != 0 {
                let pf_src_idx = 2 * (base + done) + 2 * pf_ahead;
                if pf_src_idx + 64 <= src_len {
                    prefetch_l1_lines(
                        src_base_ptr.add(pf_src_idx).cast::<i8>(),
                        ((2 * pf_ahead) / 64).max(1),
                    );
                }
            }
            let chunk = block.min(total - done);
            let lanes = chunk & !3;
            let chunk_end = done + lanes;
            // Unroll-by-8: 8 quad-iterations per inner pass (32 dst slots,
            // 64 src F128). Each step is independent — the compiler is free
            // to pipeline CLMULs across them.
            let mut t = done;
            let unroll_end = done + (lanes & !31);
            while t < unroll_end {
                let s0 = 2 * (base + t);
                let s1 = s0 + 8;
                let s2 = s0 + 16;
                let s3 = s0 + 24;
                let s4 = s0 + 32;
                let s5 = s0 + 40;
                let s6 = s0 + 48;
                let s7 = s0 + 56;
                let lo0 = _mm512_loadu_si512(src.as_ptr().add(s0) as *const __m512i);
                let hi0 = _mm512_loadu_si512(src.as_ptr().add(s0 + 4) as *const __m512i);
                let even0 = _mm512_shuffle_i32x4::<0x88>(lo0, hi0);
                let odd0 = _mm512_shuffle_i32x4::<0xDD>(lo0, hi0);
                let v0 = _mm512_xor_si512(
                    even0,
                    ghash_mul_x4_split(_mm512_xor_si512(even0, odd0), r_bcast, r_x64),
                );
                let lo1 = _mm512_loadu_si512(src.as_ptr().add(s1) as *const __m512i);
                let hi1 = _mm512_loadu_si512(src.as_ptr().add(s1 + 4) as *const __m512i);
                let even1 = _mm512_shuffle_i32x4::<0x88>(lo1, hi1);
                let odd1 = _mm512_shuffle_i32x4::<0xDD>(lo1, hi1);
                let v1 = _mm512_xor_si512(
                    even1,
                    ghash_mul_x4_split(_mm512_xor_si512(even1, odd1), r_bcast, r_x64),
                );
                let lo2 = _mm512_loadu_si512(src.as_ptr().add(s2) as *const __m512i);
                let hi2 = _mm512_loadu_si512(src.as_ptr().add(s2 + 4) as *const __m512i);
                let even2 = _mm512_shuffle_i32x4::<0x88>(lo2, hi2);
                let odd2 = _mm512_shuffle_i32x4::<0xDD>(lo2, hi2);
                let v2 = _mm512_xor_si512(
                    even2,
                    ghash_mul_x4_split(_mm512_xor_si512(even2, odd2), r_bcast, r_x64),
                );
                let lo3 = _mm512_loadu_si512(src.as_ptr().add(s3) as *const __m512i);
                let hi3 = _mm512_loadu_si512(src.as_ptr().add(s3 + 4) as *const __m512i);
                let even3 = _mm512_shuffle_i32x4::<0x88>(lo3, hi3);
                let odd3 = _mm512_shuffle_i32x4::<0xDD>(lo3, hi3);
                let v3 = _mm512_xor_si512(
                    even3,
                    ghash_mul_x4_split(_mm512_xor_si512(even3, odd3), r_bcast, r_x64),
                );
                let lo4 = _mm512_loadu_si512(src.as_ptr().add(s4) as *const __m512i);
                let hi4 = _mm512_loadu_si512(src.as_ptr().add(s4 + 4) as *const __m512i);
                let even4 = _mm512_shuffle_i32x4::<0x88>(lo4, hi4);
                let odd4 = _mm512_shuffle_i32x4::<0xDD>(lo4, hi4);
                let v4 = _mm512_xor_si512(
                    even4,
                    ghash_mul_x4_split(_mm512_xor_si512(even4, odd4), r_bcast, r_x64),
                );
                let lo5 = _mm512_loadu_si512(src.as_ptr().add(s5) as *const __m512i);
                let hi5 = _mm512_loadu_si512(src.as_ptr().add(s5 + 4) as *const __m512i);
                let even5 = _mm512_shuffle_i32x4::<0x88>(lo5, hi5);
                let odd5 = _mm512_shuffle_i32x4::<0xDD>(lo5, hi5);
                let v5 = _mm512_xor_si512(
                    even5,
                    ghash_mul_x4_split(_mm512_xor_si512(even5, odd5), r_bcast, r_x64),
                );
                let lo6 = _mm512_loadu_si512(src.as_ptr().add(s6) as *const __m512i);
                let hi6 = _mm512_loadu_si512(src.as_ptr().add(s6 + 4) as *const __m512i);
                let even6 = _mm512_shuffle_i32x4::<0x88>(lo6, hi6);
                let odd6 = _mm512_shuffle_i32x4::<0xDD>(lo6, hi6);
                let v6 = _mm512_xor_si512(
                    even6,
                    ghash_mul_x4_split(_mm512_xor_si512(even6, odd6), r_bcast, r_x64),
                );
                let lo7 = _mm512_loadu_si512(src.as_ptr().add(s7) as *const __m512i);
                let hi7 = _mm512_loadu_si512(src.as_ptr().add(s7 + 4) as *const __m512i);
                let even7 = _mm512_shuffle_i32x4::<0x88>(lo7, hi7);
                let odd7 = _mm512_shuffle_i32x4::<0xDD>(lo7, hi7);
                let v7 = _mm512_xor_si512(
                    even7,
                    ghash_mul_x4_split(_mm512_xor_si512(even7, odd7), r_bcast, r_x64),
                );
                _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, v0);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 4) as *mut __m512i, v1);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 8) as *mut __m512i, v2);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 12) as *mut __m512i, v3);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 16) as *mut __m512i, v4);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 20) as *mut __m512i, v5);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 24) as *mut __m512i, v6);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 28) as *mut __m512i, v7);
                t += 32;
            }
            // Tail of the chunk (0..3 quads not fitting the unroll).
            while t < chunk_end {
                let s = 2 * (base + t);
                let lo = _mm512_loadu_si512(src.as_ptr().add(s) as *const __m512i);
                let hi = _mm512_loadu_si512(src.as_ptr().add(s + 4) as *const __m512i);
                let even = _mm512_shuffle_i32x4::<0x88>(lo, hi);
                let odd = _mm512_shuffle_i32x4::<0xDD>(lo, hi);
                let diff = _mm512_xor_si512(even, odd);
                let new = _mm512_xor_si512(even, ghash_mul_x4_split(diff, r_bcast, r_x64));
                _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, new);
                t += 4;
            }
            // Scalar tail of the chunk (0..3 dst slots).
            let mut tail = chunk_end;
            while tail < chunk {
                let s = 2 * (base + tail);
                let even = src[s];
                dst[tail] = even + r * (even + src[s + 1]);
                tail += 1;
            }
            done = done + chunk;
        }
    }
}

/// L1-cache-blocked, unrolled-by-8 version of [`fold_pairs_with_scaled_addend`].
/// Runtime-detected VPCLMULQDQ + AVX-512F required.
///
/// Same body shape as [`fold_pairs_cached`] but every quad additionally
/// applies a scaled addend fold: the per-slot work becomes
/// `src_fold + scale * addend_fold`. Folding correctness is preserved
/// (the in-chunk quad order is identical to the original).
///
/// # Safety
/// Caller must ensure AVX-512F + VPCLMULQDQ are available, the source
/// slice has `2 * (base + dst.len())` elements, and the addend slice
/// covers the same pair range.
#[target_feature(enable = "avx512f,vpclmulqdq")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn fold_pairs_with_scaled_addend_cached(
    src: &[F128],
    addend: &[F128],
    base: usize,
    dst: &mut [F128],
    r: F128,
    scale: F128,
) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees target features and source bounds.
    unsafe {
        let r_x4 = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let r_x64 = ghash_shift64_x4(r_x4);
        let scale_x4 = _mm512_broadcast_i32x4(_mm_set_epi64x(scale.hi as i64, scale.lo as i64));
        let scale_x64 = ghash_shift64_x4(scale_x4);
        let total = dst.len();
        let block = fold_pairs_block_size().min(total);
        let pf_ahead = fold_pairs_pf_ahead();
        let src_base_ptr = src.as_ptr();
        let addend_base_ptr = addend.as_ptr();
        let src_len = src.len();
        let addend_len = addend.len();
        let mut done = 0usize;
        while done < total {
            if pf_ahead != 0 {
                let pf_src_idx = 2 * (base + done) + 2 * pf_ahead;
                if pf_src_idx + 64 <= src_len {
                    prefetch_l1_lines(
                        src_base_ptr.add(pf_src_idx).cast::<i8>(),
                        ((2 * pf_ahead) / 64).max(1),
                    );
                }
                let pf_add_idx = 2 * (base + done) + 2 * pf_ahead;
                if pf_add_idx + 64 <= addend_len {
                    prefetch_l1_lines(
                        addend_base_ptr.add(pf_add_idx).cast::<i8>(),
                        ((2 * pf_ahead) / 64).max(1),
                    );
                }
            }
            let chunk = block.min(total - done);
            let lanes = chunk & !3;
            let chunk_end = done + lanes;
            let unroll_end = done + (lanes & !31);
            let mut t = done;
            while t < unroll_end {
                macro_rules! fold_pair_quad {
                    ($tt:expr) => {{
                        let s_local = 2 * (base + $tt);
                        let s_lo = _mm512_loadu_si512(
                            src.as_ptr().add(s_local) as *const __m512i
                        );
                        let s_hi = _mm512_loadu_si512(
                            src.as_ptr().add(s_local + 4) as *const __m512i
                        );
                        let s_even = _mm512_shuffle_i32x4::<0x88>(s_lo, s_hi);
                        let s_odd = _mm512_shuffle_i32x4::<0xDD>(s_lo, s_hi);
                        let s_folded = _mm512_xor_si512(
                            s_even,
                            ghash_mul_x4_split(
                                _mm512_xor_si512(s_even, s_odd),
                                r_x4,
                                r_x64,
                            ),
                        );
                        let a_lo = _mm512_loadu_si512(
                            addend.as_ptr().add(s_local) as *const __m512i
                        );
                        let a_hi = _mm512_loadu_si512(
                            addend.as_ptr().add(s_local + 4) as *const __m512i
                        );
                        let a_even = _mm512_shuffle_i32x4::<0x88>(a_lo, a_hi);
                        let a_odd = _mm512_shuffle_i32x4::<0xDD>(a_lo, a_hi);
                        let a_folded = _mm512_xor_si512(
                            a_even,
                            ghash_mul_x4_split(
                                _mm512_xor_si512(a_even, a_odd),
                                r_x4,
                                r_x64,
                            ),
                        );
                        _mm512_xor_si512(
                            s_folded,
                            ghash_mul_x4_split(a_folded, scale_x4, scale_x64),
                        )
                    }};
                }
                let v0 = fold_pair_quad!(t);
                let v1 = fold_pair_quad!(t + 4);
                let v2 = fold_pair_quad!(t + 8);
                let v3 = fold_pair_quad!(t + 12);
                let v4 = fold_pair_quad!(t + 16);
                let v5 = fold_pair_quad!(t + 20);
                let v6 = fold_pair_quad!(t + 24);
                let v7 = fold_pair_quad!(t + 28);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, v0);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 4) as *mut __m512i, v1);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 8) as *mut __m512i, v2);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 12) as *mut __m512i, v3);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 16) as *mut __m512i, v4);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 20) as *mut __m512i, v5);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 24) as *mut __m512i, v6);
                _mm512_storeu_si512(dst.as_mut_ptr().add(t + 28) as *mut __m512i, v7);
                t += 32;
            }
            while t < chunk_end {
                let s = 2 * (base + t);
                let s_lo = _mm512_loadu_si512(src.as_ptr().add(s) as *const __m512i);
                let s_hi = _mm512_loadu_si512(src.as_ptr().add(s + 4) as *const __m512i);
                let s_even = _mm512_shuffle_i32x4::<0x88>(s_lo, s_hi);
                let s_odd = _mm512_shuffle_i32x4::<0xDD>(s_lo, s_hi);
                let s_folded = _mm512_xor_si512(
                    s_even,
                    ghash_mul_x4_split(_mm512_xor_si512(s_even, s_odd), r_x4, r_x64),
                );
                let a_lo = _mm512_loadu_si512(addend.as_ptr().add(s) as *const __m512i);
                let a_hi = _mm512_loadu_si512(addend.as_ptr().add(s + 4) as *const __m512i);
                let a_even = _mm512_shuffle_i32x4::<0x88>(a_lo, a_hi);
                let a_odd = _mm512_shuffle_i32x4::<0xDD>(a_lo, a_hi);
                let a_folded = _mm512_xor_si512(
                    a_even,
                    ghash_mul_x4_split(_mm512_xor_si512(a_even, a_odd), r_x4, r_x64),
                );
                let out = _mm512_xor_si512(
                    s_folded,
                    ghash_mul_x4_split(a_folded, scale_x4, scale_x64),
                );
                _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, out);
                t += 4;
            }
            let mut tail = chunk_end;
            while tail < chunk {
                let index = 2 * (base + tail);
                let src_even = src[index];
                let addend_even = addend[index];
                let src_folded = src_even + r * (src_even + src[index + 1]);
                let addend_folded = addend_even + r * (addend_even + addend[index + 1]);
                dst[tail] = src_folded + scale * addend_folded;
                tail += 1;
            }
            done = done + chunk;
        }
    }
}

/// Four-lane pair fold using AVX-512 lane deinterleaving and VPCLMULQDQ.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let r_x64 = ghash_shift64_x4(r_bcast);
        let lanes = dst.len() & !3;
        let mut t = 0;
        while t < lanes {
            let s = 2 * (base + t);
            let lo = _mm512_loadu_si512(src.as_ptr().add(s) as *const __m512i);
            let hi = _mm512_loadu_si512(src.as_ptr().add(s + 4) as *const __m512i);
            let even = _mm512_shuffle_i32x4::<0x88>(lo, hi);
            let odd = _mm512_shuffle_i32x4::<0xDD>(lo, hi);
            let diff = _mm512_xor_si512(even, odd);
            let new = _mm512_xor_si512(even, ghash_mul_x4_split(diff, r_bcast, r_x64));
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, new);
            t += 4;
        }
        portable_tail(src, base, dst, r, t);
    }
}

/// Four-lane `dst += scale * addend` for the lazy-OOD correction.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`; slices have equal length.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn add_scaled(dst: &mut [F128], addend: &[F128], scale: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    debug_assert_eq!(dst.len(), addend.len());
    // SAFETY: caller supplies target features and equal slice lengths.
    unsafe {
        let scale_x4 = _mm512_broadcast_i32x4(_mm_set_epi64x(scale.hi as i64, scale.lo as i64));
        let scale_x64 = ghash_shift64_x4(scale_x4);
        let lanes = dst.len() & !3;
        let mut i = 0usize;
        while i < lanes {
            let current = _mm512_loadu_si512(dst.as_ptr().add(i) as *const __m512i);
            let extra = _mm512_loadu_si512(addend.as_ptr().add(i) as *const __m512i);
            let corrected =
                _mm512_xor_si512(current, ghash_mul_x4_split(extra, scale_x4, scale_x64));
            _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut __m512i, corrected);
            i += 4;
        }
        while i < dst.len() {
            dst[i] += scale * addend[i];
            i += 1;
        }
    }
}

/// Four-lane `dst = fold_r(src) + scale * fold_r(addend)` with all
/// intermediates kept in zmm and one final store.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`; the caller guarantees that both input
/// slices contain every pair selected by `base` and `dst.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_pairs_with_scaled_addend(
    src: &[F128],
    addend: &[F128],
    base: usize,
    dst: &mut [F128],
    r: F128,
    scale: F128,
) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    // SAFETY: caller supplies target features and source bounds.
    unsafe {
        let r_x4 = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let r_x64 = ghash_shift64_x4(r_x4);
        let scale_x4 = _mm512_broadcast_i32x4(_mm_set_epi64x(scale.hi as i64, scale.lo as i64));
        let scale_x64 = ghash_shift64_x4(scale_x4);
        let lanes = dst.len() & !3;
        let mut t = 0usize;
        while t < lanes {
            let index = 2 * (base + t);
            let src_lo = _mm512_loadu_si512(src.as_ptr().add(index) as *const __m512i);
            let src_hi = _mm512_loadu_si512(src.as_ptr().add(index + 4) as *const __m512i);
            let src_even = _mm512_shuffle_i32x4::<0x88>(src_lo, src_hi);
            let src_odd = _mm512_shuffle_i32x4::<0xDD>(src_lo, src_hi);
            let src_folded = _mm512_xor_si512(
                src_even,
                ghash_mul_x4_split(_mm512_xor_si512(src_even, src_odd), r_x4, r_x64),
            );
            let addend_lo = _mm512_loadu_si512(addend.as_ptr().add(index) as *const __m512i);
            let addend_hi = _mm512_loadu_si512(addend.as_ptr().add(index + 4) as *const __m512i);
            let addend_even = _mm512_shuffle_i32x4::<0x88>(addend_lo, addend_hi);
            let addend_odd = _mm512_shuffle_i32x4::<0xDD>(addend_lo, addend_hi);
            let addend_folded = _mm512_xor_si512(
                addend_even,
                ghash_mul_x4_split(_mm512_xor_si512(addend_even, addend_odd), r_x4, r_x64),
            );
            let output = _mm512_xor_si512(
                src_folded,
                ghash_mul_x4_split(addend_folded, scale_x4, scale_x64),
            );
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, output);
            t += 4;
        }
        while t < dst.len() {
            let index = 2 * (base + t);
            let src_even = src[index];
            let addend_even = addend[index];
            let src_folded = src_even + r * (src_even + src[index + 1]);
            let addend_folded = addend_even + r * (addend_even + addend[index + 1]);
            dst[t] = src_folded + scale * addend_folded;
            t += 1;
        }
    }
}

#[inline]
fn portable_tail(src: &[F128], base: usize, dst: &mut [F128], r: F128, mut t: usize) {
    // Char-2 one-mul tail (SIMD body already uses even + r*(even+odd)).
    while t < dst.len() {
        let s = 2 * (base + t);
        let even = src[s];
        dst[t] = even + r * (even + src[s + 1]);
        t += 1;
    }
}

/// Nested pair-fold of 4-tuples into `dst`, keeping the r0 mid in zmm.
///
/// For each slot `t`:
///   low  = a0 + r0·(a0+a1)
///   high = a2 + r0·(a2+a3)
///   dst[t] = low + r1·(low+high)
///
/// Four slots (16 source F128) per iteration. Same even/odd pairing and
/// `ghash_mul_x4(r, even XOR odd)` body as [`fold_pairs`], applied twice
/// in registers. Stores `dst` only — no mid buffer.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`. `src.len() == 4 * dst.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold4_nested(src: &[F128], dst: &mut [F128], r0: F128, r1: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let r0_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r0.hi as i64, r0.lo as i64));
        let r0_x64 = ghash_shift64_x4(r0_bcast);
        let r1_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r1.hi as i64, r1.lo as i64));
        let r1_x64 = ghash_shift64_x4(r1_bcast);
        let lanes = dst.len() & !3;
        let mut t = 0;
        while t < lanes {
            let s = 4 * t;
            let v0 = _mm512_loadu_si512(src.as_ptr().add(s) as *const __m512i);
            let v1 = _mm512_loadu_si512(src.as_ptr().add(s + 4) as *const __m512i);
            let v2 = _mm512_loadu_si512(src.as_ptr().add(s + 8) as *const __m512i);
            let v3 = _mm512_loadu_si512(src.as_ptr().add(s + 12) as *const __m512i);

            // Layer r0: adjacent pairs → [low0, high0, low1, high1] / [low2, …].
            let even01 = _mm512_shuffle_i32x4::<0x88>(v0, v1);
            let odd01 = _mm512_shuffle_i32x4::<0xDD>(v0, v1);
            let mid01 = _mm512_xor_si512(
                even01,
                ghash_mul_x4_split(_mm512_xor_si512(even01, odd01), r0_bcast, r0_x64),
            );
            let even23 = _mm512_shuffle_i32x4::<0x88>(v2, v3);
            let odd23 = _mm512_shuffle_i32x4::<0xDD>(v2, v3);
            let mid23 = _mm512_xor_si512(
                even23,
                ghash_mul_x4_split(_mm512_xor_si512(even23, odd23), r0_bcast, r0_x64),
            );

            // Layer r1: (low, high) pairs → [out0, out1, out2, out3].
            let low = _mm512_shuffle_i32x4::<0x88>(mid01, mid23);
            let high = _mm512_shuffle_i32x4::<0xDD>(mid01, mid23);
            let out = _mm512_xor_si512(
                low,
                ghash_mul_x4_split(_mm512_xor_si512(low, high), r1_bcast, r1_x64),
            );
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, out);
            t += 4;
        }
        while t < dst.len() {
            let a0 = src[4 * t];
            let a1 = src[4 * t + 1];
            let a2 = src[4 * t + 2];
            let a3 = src[4 * t + 3];
            let low = a0 + r0 * (a0 + a1);
            let high = a2 + r0 * (a2 + a3);
            dst[t] = low + r1 * (low + high);
            t += 1;
        }
    }
}

/// Lines of the sixteen-bank source asked for ahead of the quad the kernel
/// is folding, in F128 elements. `FLOCK_NO_FOLD16_PF=1` removes the hints
/// (they move no data of their own and change no value, so the fold is
/// byte-identical either way); `FLOCK_FOLD16_PF=<n>` overrides the distance.
const FOLD16_PF_AHEAD: usize = 512;

fn fold16_pf_ahead() -> usize {
    static D: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        if std::env::var_os("FLOCK_NO_FOLD16_PF").is_some() {
            return 0;
        }
        std::env::var("FLOCK_FOLD16_PF")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(FOLD16_PF_AHEAD)
    });
    *D
}

/// Sixteen-bank weighted fold with deferred reduction, four output slots per
/// pass: `dst[t] = Σ_{b<16} w[b] · src[16t + b]`.
///
/// The 16 slot-major loads of a 4-slot block are transposed (128-bit lanes)
/// into bank-major vectors, each multiplied by its broadcast weight into ONE
/// four-lane unreduced accumulator (`WideGhashX4::mul_acc`, 4 CLMUL per
/// vector), and reduced once per lane at the end — 18 vector CLMULs per four
/// outputs against 36 for the two nested pair-fold passes it replaces.
/// Field-identical (reduction is F₂-linear).
///
/// # Safety
/// Caller guarantees `avx512f` + `vpclmulqdq` and `src.len() == 16 * dst.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold16_banked(src: &[F128], dst: &mut [F128], w: &[F128; 16]) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;
    debug_assert_eq!(src.len(), 16 * dst.len());
    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let wb: [__m512i; 16] = core::array::from_fn(|b| {
            _mm512_broadcast_i32x4(_mm_set_epi64x(w[b].hi as i64, w[b].lo as i64))
        });
        // 4×4 transpose of 128-bit lanes: stage-1 index vectors interleave
        // lanes {0,1} / {2,3} of two inputs; stage 2 gathers lanes {0,1} /
        // {2,3} of the two stage-1 results.
        let s1_lo = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
        let s1_hi = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
        let s2_lo = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
        let s2_hi = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
        let quads = dst.len() & !3;
        let pf_ahead = fold16_pf_ahead();
        let pf_limit = src.len().saturating_sub(64);
        let mut t = 0usize;
        while t < quads {
            if pf_ahead != 0 {
                let ahead = 16 * t + pf_ahead;
                if ahead <= pf_limit {
                    let p = src.as_ptr().add(ahead).cast::<i8>();
                    let mut l = 0usize;
                    while l < 1024 {
                        _mm_prefetch::<_MM_HINT_T0>(p.add(l));
                        l += 64;
                    }
                }
            }
            let mut acc = WideGhashX4::zero();
            for g in 0..4 {
                // v_s = banks 4g..4g+3 of slot t+s.
                let base = 16 * t + 4 * g;
                let a0 = _mm512_loadu_si512(src.as_ptr().add(base) as *const __m512i);
                let a1 = _mm512_loadu_si512(src.as_ptr().add(base + 16) as *const __m512i);
                let a2 = _mm512_loadu_si512(src.as_ptr().add(base + 32) as *const __m512i);
                let a3 = _mm512_loadu_si512(src.as_ptr().add(base + 48) as *const __m512i);
                let t0 = _mm512_permutex2var_epi64(a0, s1_lo, a1); // [a0.L0 a1.L0 a0.L1 a1.L1]
                let t1 = _mm512_permutex2var_epi64(a0, s1_hi, a1); // [a0.L2 a1.L2 a0.L3 a1.L3]
                let t2 = _mm512_permutex2var_epi64(a2, s1_lo, a3);
                let t3 = _mm512_permutex2var_epi64(a2, s1_hi, a3);
                let u0 = _mm512_permutex2var_epi64(t0, s2_lo, t2); // bank 4g+0 over slots 0..4
                let u1 = _mm512_permutex2var_epi64(t0, s2_hi, t2); // bank 4g+1
                let u2 = _mm512_permutex2var_epi64(t1, s2_lo, t3); // bank 4g+2
                let u3 = _mm512_permutex2var_epi64(t1, s2_hi, t3); // bank 4g+3
                acc.mul_acc(u0, wb[4 * g]);
                acc.mul_acc(u1, wb[4 * g + 1]);
                acc.mul_acc(u2, wb[4 * g + 2]);
                acc.mul_acc(u3, wb[4 * g + 3]);
            }
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, acc.reduce_lanes());
            t += 4;
        }
        while t < dst.len() {
            let mut v = F128::ZERO;
            for b in 0..16 {
                v += w[b] * src[16 * t + b];
            }
            dst[t] = v;
            t += 1;
        }
    }
}

/// In-place DirectFold8 factor-state bind: adjacent-pair fold of `f` and `b`
/// with fused `(u0,u2)` accumulate. Same permute body as [`fold_pairs`] and
/// the same even/odd message layout as `msg_reduce_avx512`.
///
/// In-place is safe because output `t` depends only on source `2t..2t+2`,
/// and stores of `dst[0..t)` never overlap unread source `2t..`.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`. `f.len() == b.len()`, multiple of 4.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_two_and_msg_in_place(
    f: &mut Vec<F128>,
    b: &mut Vec<F128>,
    r: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::{WideGhashX4, ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len().is_multiple_of(4));
    let half = f.len() / 2;

    // SAFETY: caller guarantees features and even pair counts; loads of
    // `2t..2t+8` complete before stores to `t..t+4` overlap those addresses.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let r_x64 = ghash_shift64_x4(r_bcast);

        let fold4 = |ptr: *const F128, t: usize| -> __m512i {
            let s = 2 * t;
            let lo = _mm512_loadu_si512(ptr.add(s) as *const __m512i);
            let hi = _mm512_loadu_si512(ptr.add(s + 4) as *const __m512i);
            let even = _mm512_shuffle_i32x4::<0x88>(lo, hi);
            let odd = _mm512_shuffle_i32x4::<0xDD>(lo, hi);
            let diff = _mm512_xor_si512(even, odd);
            _mm512_xor_si512(even, ghash_mul_x4_split(diff, r_bcast, r_x64))
        };

        let mut u0_acc = WideGhashX4::zero();
        let mut u2_acc = WideGhashX4::zero();
        let f_ptr = f.as_mut_ptr();
        let b_ptr = b.as_mut_ptr();
        let lanes = half & !7;
        let mut t = 0usize;
        while t < lanes {
            let f0 = fold4(f_ptr, t);
            let f1 = fold4(f_ptr, t + 4);
            let b0 = fold4(b_ptr, t);
            let b1 = fold4(b_ptr, t + 4);
            _mm512_storeu_si512(f_ptr.add(t) as *mut __m512i, f0);
            _mm512_storeu_si512(f_ptr.add(t + 4) as *mut __m512i, f1);
            _mm512_storeu_si512(b_ptr.add(t) as *mut __m512i, b0);
            _mm512_storeu_si512(b_ptr.add(t + 4) as *mut __m512i, b1);

            let f_even = _mm512_shuffle_i32x4::<0x88>(f0, f1);
            let b_even = _mm512_shuffle_i32x4::<0x88>(b0, b1);
            u0_acc.mul_acc(f_even, b_even);

            let f0s = _mm512_xor_si512(f0, _mm512_shuffle_i32x4::<0xB1>(f0, f0));
            let f1s = _mm512_xor_si512(f1, _mm512_shuffle_i32x4::<0xB1>(f1, f1));
            let f_sum = _mm512_shuffle_i32x4::<0x88>(f0s, f1s);
            let b0s = _mm512_xor_si512(b0, _mm512_shuffle_i32x4::<0xB1>(b0, b0));
            let b1s = _mm512_xor_si512(b1, _mm512_shuffle_i32x4::<0xB1>(b1, b1));
            let b_sum = _mm512_shuffle_i32x4::<0x88>(b0s, b1s);
            u2_acc.mul_acc(f_sum, b_sum);

            t += 8;
        }

        let mut u0 = u0_acc.fold().reduce();
        let mut u2 = u2_acc.fold().reduce();
        while t < half {
            let source = 2 * t;
            let f0 = *f_ptr.add(source) + r * (*f_ptr.add(source) + *f_ptr.add(source + 1));
            let f1 = *f_ptr.add(source + 2) + r * (*f_ptr.add(source + 2) + *f_ptr.add(source + 3));
            let b0 = *b_ptr.add(source) + r * (*b_ptr.add(source) + *b_ptr.add(source + 1));
            let b1 = *b_ptr.add(source + 2) + r * (*b_ptr.add(source + 2) + *b_ptr.add(source + 3));
            *f_ptr.add(t) = f0;
            *f_ptr.add(t + 1) = f1;
            *b_ptr.add(t) = b0;
            *b_ptr.add(t + 1) = b1;
            u0 += f0 * b0;
            u2 += (f0 + f1) * (b0 + b1);
            t += 2;
        }
        f.truncate(half);
        b.truncate(half);
        (u0, u2)
    }
}

/// Four-lane split-half bind: `lo[i] = lo[i] + r·(hi[i] + lo[i])`.
///
/// The two operand runs are separate contiguous slices (the top-bit split of
/// one table), so lanes pair up straight out of the loads — no permute.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`; `hi.len() >= lo.len()`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn bind_split_half(lo: &mut [F128], hi: &[F128], r: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    debug_assert!(hi.len() >= lo.len());
    // SAFETY: caller supplies target features and one `hi` per `lo` slot.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let r_x64 = ghash_shift64_x4(r_bcast);
        let lanes = lo.len() & !3;
        let mut i = 0usize;
        while i < lanes {
            let a = _mm512_loadu_si512(lo.as_ptr().add(i) as *const __m512i);
            let b = _mm512_loadu_si512(hi.as_ptr().add(i) as *const __m512i);
            let new = _mm512_xor_si512(
                a,
                ghash_mul_x4_split(_mm512_xor_si512(a, b), r_bcast, r_x64),
            );
            _mm512_storeu_si512(lo.as_mut_ptr().add(i) as *mut __m512i, new);
            i += 4;
        }
        while i < lo.len() {
            lo[i] = lo[i] + r * (hi[i] + lo[i]);
            i += 1;
        }
    }
}

/// Four-lane split-half product-sumcheck message:
/// `(Σ chi[i]·zhi[i], Σ (chi[i]+clo[i])·(zhi[i]+zlo[i]))`.
///
/// Both sums use one deferred-reduction accumulator each (reduction is
/// F₂-linear, so one reduce per sum equals reducing every product), with an
/// unreduced scalar tail XORed in before the single reduce.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`; all four slices at least `n` long.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn msg_split_half(
    chi: &[F128],
    clo: &[F128],
    zhi: &[F128],
    zlo: &[F128],
    n: usize,
) -> (F128, F128) {
    use crate::field::gf2_128::F256Unreduced;
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees features and that every slice covers `n`.
    unsafe {
        let mut e1_wide = WideGhashX4::zero();
        let mut einf_wide = WideGhashX4::zero();
        let lanes = n & !3;
        let mut i = 0usize;
        while i < lanes {
            let ch = _mm512_loadu_si512(chi.as_ptr().add(i) as *const __m512i);
            let zh = _mm512_loadu_si512(zhi.as_ptr().add(i) as *const __m512i);
            e1_wide.mul_acc(ch, zh);
            let cl = _mm512_loadu_si512(clo.as_ptr().add(i) as *const __m512i);
            let zl = _mm512_loadu_si512(zlo.as_ptr().add(i) as *const __m512i);
            einf_wide.mul_acc(_mm512_xor_si512(ch, cl), _mm512_xor_si512(zh, zl));
            i += 4;
        }
        let mut e1_acc = F256Unreduced::ZERO;
        let mut einf_acc = F256Unreduced::ZERO;
        while i < n {
            e1_acc ^= chi[i].mul_unreduced(zhi[i]);
            einf_acc ^= (chi[i] + clo[i]).mul_unreduced(zhi[i] + zlo[i]);
            i += 1;
        }
        e1_acc ^= e1_wide.fold();
        einf_acc ^= einf_wide.fold();
        (e1_acc.reduce(), einf_acc.reduce())
    }
}

/// Four-lane fused quarter bind + next-round message. Per slot `i`:
/// `lo = c0[i] + r·(c2[i]+c0[i])`, `hi = c1[i] + r·(c3[i]+c1[i])`, likewise
/// for `z`; `c0[i]/c1[i]/z0[i]/z1[i]` take the bound values, and the message
/// accumulates `hi·zhi` and `(hi+lo)·(zhi+zlo)`.
///
/// Quarters are separate contiguous runs, so no lane permute is needed; the
/// four binds share one broadcast `r`.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`; all eight slices at least `n` long.
#[target_feature(enable = "avx512f,vpclmulqdq")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn bind_both_and_msg_split(
    cq0: &mut [F128],
    cq1: &mut [F128],
    cq2: &[F128],
    cq3: &[F128],
    zq0: &mut [F128],
    zq1: &mut [F128],
    zq2: &[F128],
    zq3: &[F128],
    r: F128,
    n: usize,
) -> (F128, F128) {
    use crate::field::gf2_128::F256Unreduced;
    use crate::field::gf2_128::x86_64::{WideGhashX4, ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees features and that every slice covers `n`.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let r_x64 = ghash_shift64_x4(r_bcast);
        let mut e1_wide = WideGhashX4::zero();
        let mut einf_wide = WideGhashX4::zero();
        let lanes = n & !3;
        let mut i = 0usize;
        while i < lanes {
            let c0 = _mm512_loadu_si512(cq0.as_ptr().add(i) as *const __m512i);
            let c1 = _mm512_loadu_si512(cq1.as_ptr().add(i) as *const __m512i);
            let c2 = _mm512_loadu_si512(cq2.as_ptr().add(i) as *const __m512i);
            let c3 = _mm512_loadu_si512(cq3.as_ptr().add(i) as *const __m512i);
            let z0 = _mm512_loadu_si512(zq0.as_ptr().add(i) as *const __m512i);
            let z1 = _mm512_loadu_si512(zq1.as_ptr().add(i) as *const __m512i);
            let z2 = _mm512_loadu_si512(zq2.as_ptr().add(i) as *const __m512i);
            let z3 = _mm512_loadu_si512(zq3.as_ptr().add(i) as *const __m512i);

            let lo = _mm512_xor_si512(
                c0,
                ghash_mul_x4_split(_mm512_xor_si512(c2, c0), r_bcast, r_x64),
            );
            let hi = _mm512_xor_si512(
                c1,
                ghash_mul_x4_split(_mm512_xor_si512(c3, c1), r_bcast, r_x64),
            );
            let zlo = _mm512_xor_si512(
                z0,
                ghash_mul_x4_split(_mm512_xor_si512(z2, z0), r_bcast, r_x64),
            );
            let zhi = _mm512_xor_si512(
                z1,
                ghash_mul_x4_split(_mm512_xor_si512(z3, z1), r_bcast, r_x64),
            );

            _mm512_storeu_si512(cq0.as_mut_ptr().add(i) as *mut __m512i, lo);
            _mm512_storeu_si512(cq1.as_mut_ptr().add(i) as *mut __m512i, hi);
            _mm512_storeu_si512(zq0.as_mut_ptr().add(i) as *mut __m512i, zlo);
            _mm512_storeu_si512(zq1.as_mut_ptr().add(i) as *mut __m512i, zhi);

            e1_wide.mul_acc(hi, zhi);
            einf_wide.mul_acc(_mm512_xor_si512(hi, lo), _mm512_xor_si512(zhi, zlo));
            i += 4;
        }
        let mut e1_acc = F256Unreduced::ZERO;
        let mut einf_acc = F256Unreduced::ZERO;
        while i < n {
            let lo = cq0[i] + r * (cq2[i] + cq0[i]);
            let hi = cq1[i] + r * (cq3[i] + cq1[i]);
            let zlo = zq0[i] + r * (zq2[i] + zq0[i]);
            let zhi = zq1[i] + r * (zq3[i] + zq1[i]);
            cq0[i] = lo;
            cq1[i] = hi;
            zq0[i] = zlo;
            zq1[i] = zhi;
            e1_acc ^= hi.mul_unreduced(zhi);
            einf_acc ^= (hi + lo).mul_unreduced(zhi + zlo);
            i += 1;
        }
        e1_acc ^= e1_wide.fold();
        einf_acc ^= einf_wide.fold();
        (e1_acc.reduce(), einf_acc.reduce())
    }
}
