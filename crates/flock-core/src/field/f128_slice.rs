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

/// Fused pair-fold + sumcheck message accumulation.
///
/// Folds `f_src` and `b_src` at `r` (halving), stores the results to
/// `f_dst`/`b_dst`, and returns the sumcheck message `(u_0, u_2)`
/// computed from the register-resident folded values — avoiding the
/// store-then-reload cycle of `fold_pairs` + separate message reduce.
///
/// On AVX-512 x86 this uses the fused kernel; everywhere else it falls
/// back to `fold_pairs` + scalar message reduce (bit-identical result).
#[inline]
pub(crate) fn fold_pairs_and_msg(
    f_src: &[F128],
    f_base: usize,
    f_dst: &mut [F128],
    b_src: &[F128],
    b_base: usize,
    b_dst: &mut [F128],
    r: F128,
) -> (F128, F128) {
    assert!(
        f_base <= f_src.len() / 2 && f_dst.len() <= f_src.len() / 2 - f_base,
        "fold_pairs_and_msg f source must contain both elements for every destination pair"
    );
    assert!(
        b_base <= b_src.len() / 2 && b_dst.len() <= b_src.len() / 2 - b_base,
        "fold_pairs_and_msg b source must contain both elements for every destination pair"
    );
    assert_eq!(f_dst.len(), b_dst.len());

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features and the
    // bounds checks above guarantee both source elements for every output.
    unsafe {
        return x86_64::fold_pairs_and_msg(f_src, f_base, f_dst, b_src, b_base, b_dst, r);
    }

    // Fallback: two-pass fold + scalar message reduce.
    fold_pairs(f_src, f_base, f_dst, r);
    fold_pairs(b_src, b_base, b_dst, r);
    let mut u0 = F128::ZERO;
    let mut u2 = F128::ZERO;
    let mut k = 0;
    while k + 1 < f_dst.len() {
        let f0 = f_dst[k];
        let f1 = f_dst[k + 1];
        let b0 = b_dst[k];
        let b1 = b_dst[k + 1];
        u0 += f0 * b0;
        u2 += (f0 + f1) * (b0 + b1);
        k += 2;
    }
    (u0, u2)
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
