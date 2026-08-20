//! Architecture-selected kernels over contiguous [`F128`] slices.

use super::F128;

#[cfg(any(
    test,
    not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ),
        all(target_arch = "aarch64", target_feature = "aes")
    ))
))]
mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod x86_64;

/// Fold adjacent pairs from `src` into `dst`, starting at pair `base`.
///
/// Computes `dst[t] = src[2j] * (1 + r) + src[2j + 1] * r`, where
/// `j = base + t`. Portable / serial tails use the char-2 identity
/// `even + r*(even+odd)` (one mul). AVX-512 / NEON already used that form.
#[inline]
pub(crate) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    assert!(
        base <= src.len() / 2 && dst.len() <= src.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features and the
    // bounds check above guarantees both source elements for every output.
    unsafe {
        x86_64::fold_pairs(src, base, dst, r);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate guarantees PMULL support through the aes feature;
    // the bounds check above guarantees both source elements for every output.
    unsafe {
        aarch64::fold_pairs(src, base, dst, r);
    }

    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    portable::fold_pairs(src, base, dst, r);
}

/// Add one scaled field slice into another: `dst[i] += scale * addend[i]`.
///
/// The ranked lazy-OOD fold uses this after folding the incumbent basis and
/// before reducing the next-round message.  Keeping the operation here lets
/// the Sapphire Rapids build issue four independent VPCLMUL products at once;
/// other builds retain the exact scalar field operation.
#[inline]
pub(crate) fn add_scaled(dst: &mut [F128], addend: &[F128], scale: F128) {
    assert_eq!(dst.len(), addend.len(), "scaled addend length changed");

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features and the
    // length assertion guarantees one readable addend per destination slot.
    unsafe {
        x86_64::add_scaled(dst, addend, scale);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    for (value, &extra) in dst.iter_mut().zip(addend) {
        *value += scale * extra;
    }
}

/// Nested pair-fold of adjacent 4-tuples: `r0` then `r1`, even/odd pairing.
///
/// `dst[t] = low + r1·(low+high)` where
/// `low = a0 + r0·(a0+a1)`, `high = a2 + r0·(a2+a3)` and
/// `(a0,a1,a2,a3) = src[4t .. 4t+4]`. Writes `dst` only — the r0 mid stays
/// in registers on AVX-512. Portable / non-x86 is the scalar nested form.
#[inline]
pub(crate) fn fold4_nested(src: &[F128], dst: &mut [F128], r0: F128, r1: F128) {
    assert_eq!(
        src.len(),
        4 * dst.len(),
        "fold4 source must contain four elements for every destination slot"
    );

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features and the
    // bounds check above guarantees all four source elements per output.
    unsafe {
        x86_64::fold4_nested(src, dst, r0, r1);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    {
        for (t, value) in dst.iter_mut().enumerate() {
            let a0 = src[4 * t];
            let a1 = src[4 * t + 1];
            let a2 = src[4 * t + 2];
            let a3 = src[4 * t + 3];
            let low = a0 + r0 * (a0 + a1);
            let high = a2 + r0 * (a2 + a3);
            *value = low + r1 * (low + high);
        }
    }
}

#[cfg(test)]
mod tests {
    /// `fold16_banked` (deferred-reduction AVX-512 kernel on x86; scalar
    /// elsewhere) equals the straight reduced sum `Σ w[b]·src[16t+b]` at
    /// lengths that hit the four-slot vector body and the scalar tail.
    #[test]
    fn fold16_banked_matches_scalar_reduced_sum() {
        use super::*;
        let mut state = 0x5eed_f01d_16u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for n in [1usize, 3, 4, 5, 8, 13, 64, 257] {
            let src: Vec<F128> = (0..16 * n).map(|_| F128 { lo: next(), hi: next() }).collect();
            let w: [F128; 16] = core::array::from_fn(|_| F128 { lo: next(), hi: next() });
            let mut got = vec![F128::ZERO; n];
            fold16_banked(&src, &mut got, &w);
            for t in 0..n {
                let mut want = F128::ZERO;
                for b in 0..16 {
                    want += w[b] * src[16 * t + b];
                }
                assert_eq!(got[t], want, "n={n} t={t}");
            }
        }
        // Degenerate weights: one-hot, all-zero, all-one.
        let src: Vec<F128> = (0..16 * 8).map(|i| F128 { lo: i as u64 * 7 + 1, hi: (i as u64) << 40 }).collect();
        for b0 in 0..16 {
            let mut w = [F128::ZERO; 16];
            w[b0] = F128::ONE;
            let mut got = vec![F128::ZERO; 8];
            fold16_banked(&src, &mut got, &w);
            for t in 0..8 {
                assert_eq!(got[t], src[16 * t + b0]);
            }
        }
        let w = [F128::ONE; 16];
        let mut got = vec![F128::ZERO; 8];
        fold16_banked(&src, &mut got, &w);
        for t in 0..8 {
            let want = src[16 * t..16 * t + 16].iter().fold(F128::ZERO, |a, &b| a + b);
            assert_eq!(got[t], want);
        }
    }


    use super::*;

    #[test]
    fn selected_fold_matches_portable_with_offset_and_tail() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let src: Vec<F128> = (0..30)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let r = F128 {
            lo: next(),
            hi: next(),
        };
        let mut expected = vec![F128::ZERO; 9];
        let mut actual = vec![F128::ZERO; 9];

        portable::fold_pairs(&src, 3, &mut expected, r);
        fold_pairs(&src, 3, &mut actual, r);

        assert_eq!(actual, expected);
    }

    /// Portable one-mul leaf is bit-identical to the two-mul formula.
    #[test]
    fn portable_fold_pairs_matches_two_mul() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let src: Vec<F128> = (0..40)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let r = F128 {
            lo: next(),
            hi: next(),
        };
        let one_plus_r = F128::ONE + r;
        for &(base, n) in &[(0usize, 20usize), (3, 9), (1, 1), (5, 7)] {
            let mut got = vec![F128::ZERO; n];
            portable::fold_pairs(&src, base, &mut got, r);
            for t in 0..n {
                let s = 2 * (base + t);
                let expect = src[s] * one_plus_r + src[s + 1] * r;
                assert_eq!(got[t], expect, "base={base} t={t}");
            }
        }
    }

    /// Selected fold4_nested matches the scalar nested pair-fold, including a
    /// non-multiple-of-4 tail, and matches two `fold_pairs` (r0 then r1).
    #[test]
    fn selected_fold4_nested_matches_scalar_and_two_pass_pairs() {
        let mut state = 0xA5A5_C0DE_F00D_1234_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let src: Vec<F128> = (0..44)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let r0 = F128 {
            lo: next(),
            hi: next(),
        };
        let r1 = F128 {
            lo: next(),
            hi: next(),
        };
        for n in [1usize, 3, 4, 5, 7, 8, 11] {
            let mut got = vec![F128::ZERO; n];
            let mut portable_got = vec![F128::ZERO; n];
            fold4_nested(&src[..4 * n], &mut got, r0, r1);
            portable::fold4_nested(&src[..4 * n], &mut portable_got, r0, r1);
            assert_eq!(got, portable_got, "portable n={n}");
            for t in 0..n {
                let a0 = src[4 * t];
                let a1 = src[4 * t + 1];
                let a2 = src[4 * t + 2];
                let a3 = src[4 * t + 3];
                let low = a0 + r0 * (a0 + a1);
                let high = a2 + r0 * (a2 + a3);
                let expect = low + r1 * (low + high);
                assert_eq!(got[t], expect, "scalar n={n} t={t}");
            }
            // Two-pass fold_pairs on a tiny stack mid (test only) must agree.
            let mut mid = vec![F128::ZERO; 2 * n];
            let mut via_pairs = vec![F128::ZERO; n];
            fold_pairs(&src[..4 * n], 0, &mut mid, r0);
            fold_pairs(&mid, 0, &mut via_pairs, r1);
            assert_eq!(got, via_pairs, "two-pass pairs n={n}");
        }
    }
}

/// Sixteen-bank weighted fold: `dst[t] = Σ_{b<16} w[b] · src[16t + b]`.
///
/// AVX-512: deferred-reduction kernel (one reduce per output lane). Other
/// targets: the straightforward reduced form. Same field element either way.
#[inline]
pub(crate) fn fold16_banked(src: &[F128], dst: &mut [F128], w: &[F128; 16]) {
    assert_eq!(
        src.len(),
        16 * dst.len(),
        "fold16 source must contain sixteen elements for every destination slot"
    );

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features and the
    // bounds check above guarantees all sixteen source elements per output.
    unsafe {
        x86_64::fold16_banked(src, dst, w);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    {
        for (t, value) in dst.iter_mut().enumerate() {
            let mut v = F128::ZERO;
            for b in 0..16 {
                v += w[b] * src[16 * t + b];
            }
            *value = v;
        }
    }
}
