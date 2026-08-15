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

#[cfg(test)]
mod tests {
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
}
