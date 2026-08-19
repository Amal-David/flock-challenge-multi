#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
pub use aarch64::{
    fold_block_major_chunk_neon_x2, gather_transpose_tile_neon, lincheck_qform_enabled,
    partial_fold_packed_z_neon_iblock_padded, partial_fold_packed_z_neon_oblock_padded,
    partial_fold_packed_z_neon_single, partial_fold_packed_z_neon_single_padded,
};

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::partial_fold_packed_z_x86_tiled_padded;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
pub(crate) use x86_64::{build_nibble_tables, fold_block_major_chunk_x86_avx512};
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub(crate) use x86_64::{NibbleTables, build_nibble_tables as build_nibble_tables_portable};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
pub use x86_64::partial_fold_packed_z_x86_gfni_padded;

use super::F128;

/// Rayon grain for the product-sumcheck tables. 256 F128 = 4 KiB, four
/// cache lines of ZMM traffic, and a multiple of the 4-lane `ghash_mul_x4`
/// width. XOR-reduction across chunks is associative, so this is not a
/// scoring knob — it only amortizes dispatch.
pub(super) const SUMCHECK_WIDE_CHUNK: usize = 256;

/// `FLOCK_NO_LC_BIND_WIDE=1` restores the scalar product-sumcheck loops.
/// Runtime AVX-512F + VPCLMUL detect; other ISAs stay on the scalar body.
pub(super) fn sumcheck_wide_enabled() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return x86_64::sumcheck_wide_enabled();
    }
    #[allow(unreachable_code)]
    false
}

/// `(e1, einf) = (Σ χ·ζ_hi, Σ (χ⊕clo)·(ζ_hi⊕ζ_lo))` over equal-length halves.
pub(super) fn eval_round(clo: &[F128], chi: &[F128], zlo: &[F128], zhi: &[F128]) -> (F128, F128) {
    debug_assert_eq!(clo.len(), chi.len());
    debug_assert_eq!(zlo.len(), clo.len());
    debug_assert_eq!(zhi.len(), clo.len());
    #[cfg(target_arch = "x86_64")]
    if sumcheck_wide_enabled() {
        // SAFETY: runtime detect + kill-switch; per-lane fully-reduced mul.
        return unsafe { x86_64::eval_round_wide(clo, chi, zlo, zhi) };
    }
    let mut e1 = F128::ZERO;
    let mut einf = F128::ZERO;
    for i in 0..clo.len() {
        e1 += chi[i] * zhi[i];
        einf += (chi[i] + clo[i]) * (zhi[i] + zlo[i]);
    }
    (e1, einf)
}

/// In-place half-split bind: `lo[i] ← lo[i] + r·(hi[i] + lo[i])`.
pub(super) fn bind_halves(lo: &mut [F128], hi: &[F128], r: F128) {
    debug_assert_eq!(lo.len(), hi.len());
    #[cfg(target_arch = "x86_64")]
    if sumcheck_wide_enabled() {
        // SAFETY: runtime detect + kill-switch; same char-2 identity as fold_pairs.
        unsafe { x86_64::bind_halves_wide(lo, hi, r) };
        return;
    }
    for i in 0..lo.len() {
        lo[i] = lo[i] + r * (hi[i] + lo[i]);
    }
}

/// Fused quarter-split bind of `comb` and `z` plus the next-round message.
pub(super) fn bind_both_eval(
    c0: &mut [F128],
    c1: &mut [F128],
    c2: &[F128],
    c3: &[F128],
    z0: &mut [F128],
    z1: &mut [F128],
    z2: &[F128],
    z3: &[F128],
    r: F128,
) -> (F128, F128) {
    debug_assert_eq!(c0.len(), c1.len());
    debug_assert_eq!(c0.len(), c2.len());
    debug_assert_eq!(c0.len(), c3.len());
    debug_assert_eq!(c0.len(), z0.len());
    debug_assert_eq!(c0.len(), z1.len());
    debug_assert_eq!(c0.len(), z2.len());
    debug_assert_eq!(c0.len(), z3.len());
    #[cfg(target_arch = "x86_64")]
    if sumcheck_wide_enabled() {
        // SAFETY: runtime detect + kill-switch; bind then per-lane reduced mul.
        return unsafe { x86_64::bind_both_eval_wide(c0, c1, c2, c3, z0, z1, z2, z3, r) };
    }
    let n = c0.len();
    let mut e1 = F128::ZERO;
    let mut einf = F128::ZERO;
    for i in 0..n {
        let lo = c0[i] + r * (c2[i] + c0[i]);
        let hi = c1[i] + r * (c3[i] + c1[i]);
        let zlo = z0[i] + r * (z2[i] + z0[i]);
        let zhi = z1[i] + r * (z3[i] + z1[i]);
        c0[i] = lo;
        c1[i] = hi;
        z0[i] = zlo;
        z1[i] = zhi;
        e1 += hi * zhi;
        einf += (hi + lo) * (zhi + zlo);
    }
    (e1, einf)
}
