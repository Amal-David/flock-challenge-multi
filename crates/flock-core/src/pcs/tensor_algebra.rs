// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Ported from binius64's `crates/math/src/tensor_algebra.rs`
// (https://github.com/binius-zk/binius64), specialized to `F = F_2`,
// `FE = F_{2^128}`.

//! Tensor algebra over `F_{2^128} ⊗_{F_2} F_{2^128}`.
//!
//! An element is a length-128 vector of `F128` (the "vertical-subring" elements
//! in DP24 nomenclature). Conceptually it's a 128×128 F_2 matrix, where row `i`
//! is `elems[i]` viewed via its bit-decomposition in the GHASH polynomial
//! basis (`bit_j(elems[i])` = coefficient of `γ^i ⊗ γ^j` in the tensor algebra).
//!
//! Used by the verifier's polylog `eval_rs_eq` (DP24 §1.3, Figure 3).

use crate::field::F128;
use core::ops::{Add, AddAssign};

/// The degree of `F_{2^128}` over `F_2`.
pub const DEGREE: usize = 128;

/// An element of `F_{2^128} ⊗_{F_2} F_{2^128}`, stored as 128 `F128` elements
/// (the vertical-subring decomposition).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorAlgebra {
    /// Length-128 vector. `elems[i]` is the coefficient of `γ^i` in the
    /// vertical basis decomposition.
    pub elems: Vec<F128>,
}

impl TensorAlgebra {
    /// All-zero element.
    pub fn zero() -> Self {
        Self {
            elems: vec![F128::ZERO; DEGREE],
        }
    }

    /// Multiplicative identity: `1 ⊗ 1`.
    pub fn one() -> Self {
        let mut elems = vec![F128::ZERO; DEGREE];
        elems[0] = F128::ONE;
        Self { elems }
    }

    /// Embed `x ∈ F_{2^128}` into the vertical subring: returns `1 ⊗ x`.
    pub fn from_vertical(x: F128) -> Self {
        let mut elems = vec![F128::ZERO; DEGREE];
        elems[0] = x;
        Self { elems }
    }

    /// Multiply by an element of the vertical subring: each `elems[i]` is
    /// scaled by `scalar` in `F_{2^128}`.
    pub fn scale_vertical(mut self, scalar: F128) -> Self {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        unsafe {
            use crate::field::gf2_128::x86_64::{
                ghash_mul_x4_low_rhs, ghash_mul_x4_split, ghash_shift64_x4,
            };
            use core::arch::x86_64::*;
            let s_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(scalar.hi as i64, scalar.lo as i64));
            if scalar.hi == 0 {
                for chunk in self.elems.chunks_exact_mut(16) {
                    let v0 = _mm512_loadu_si512(chunk.as_ptr() as *const __m512i);
                    let v1 = _mm512_loadu_si512(chunk.as_ptr().add(4) as *const __m512i);
                    let v2 = _mm512_loadu_si512(chunk.as_ptr().add(8) as *const __m512i);
                    let v3 = _mm512_loadu_si512(chunk.as_ptr().add(12) as *const __m512i);
                    let r0 = ghash_mul_x4_low_rhs(v0, s_bcast);
                    let r1 = ghash_mul_x4_low_rhs(v1, s_bcast);
                    let r2 = ghash_mul_x4_low_rhs(v2, s_bcast);
                    let r3 = ghash_mul_x4_low_rhs(v3, s_bcast);
                    _mm512_storeu_si512(chunk.as_mut_ptr() as *mut __m512i, r0);
                    _mm512_storeu_si512(chunk.as_mut_ptr().add(4) as *mut __m512i, r1);
                    _mm512_storeu_si512(chunk.as_mut_ptr().add(8) as *mut __m512i, r2);
                    _mm512_storeu_si512(chunk.as_mut_ptr().add(12) as *mut __m512i, r3);
                }
            } else {
                let s_x64 = ghash_shift64_x4(s_bcast);
                for chunk in self.elems.chunks_exact_mut(16) {
                    let v0 = _mm512_loadu_si512(chunk.as_ptr() as *const __m512i);
                    let v1 = _mm512_loadu_si512(chunk.as_ptr().add(4) as *const __m512i);
                    let v2 = _mm512_loadu_si512(chunk.as_ptr().add(8) as *const __m512i);
                    let v3 = _mm512_loadu_si512(chunk.as_ptr().add(12) as *const __m512i);
                    let r0 = ghash_mul_x4_split(v0, s_bcast, s_x64);
                    let r1 = ghash_mul_x4_split(v1, s_bcast, s_x64);
                    let r2 = ghash_mul_x4_split(v2, s_bcast, s_x64);
                    let r3 = ghash_mul_x4_split(v3, s_bcast, s_x64);
                    _mm512_storeu_si512(chunk.as_mut_ptr() as *mut __m512i, r0);
                    _mm512_storeu_si512(chunk.as_mut_ptr().add(4) as *mut __m512i, r1);
                    _mm512_storeu_si512(chunk.as_mut_ptr().add(8) as *mut __m512i, r2);
                    _mm512_storeu_si512(chunk.as_mut_ptr().add(12) as *mut __m512i, r3);
                }
            }
            return self;
        }
        #[allow(unreachable_code)]
        {
            for e in self.elems.iter_mut() {
                *e *= scalar;
            }
            self
        }
    }

    /// Multiply by an element of the horizontal subring. Implemented as
    /// `transpose ∘ scale_vertical ∘ transpose`.
    pub fn scale_horizontal(self, scalar: F128) -> Self {
        self.transpose().scale_vertical(scalar).transpose()
    }

    /// Transpose the tensor algebra element: swap vertical and horizontal
    /// subring roles. Concretely, after transpose, `bit_j(elems'[i]) =
    /// bit_i(elems[j])` for all `i, j ∈ [0, 128)`.
    pub fn transpose(mut self) -> Self {
        square_transpose(&mut self.elems);
        self
    }

    /// Fold the tensor algebra element to a single `F128` by scaling rows with
    /// `coeffs` (length 128) and summing.
    ///
    /// Computes `Σ_i coeffs[i] · transpose(self).elems[i]`.
    pub fn fold_vertical(self, coeffs: &[F128]) -> F128 {
        assert_eq!(
            coeffs.len(),
            DEGREE,
            "fold_vertical: coeffs.len() must be 128"
        );
        let transposed = self.transpose();
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq",
            target_feature = "sse4.1"
        ))]
        unsafe {
            use crate::field::gf2_128::x86_64::WideGhashX4;
            use core::arch::x86_64::*;
            let mut acc0 = WideGhashX4::zero();
            let mut acc1 = WideGhashX4::zero();
            for i in (0..DEGREE).step_by(8) {
                let e0 = _mm512_loadu_si512(transposed.elems.as_ptr().add(i) as *const __m512i);
                let c0 = _mm512_loadu_si512(coeffs.as_ptr().add(i) as *const __m512i);
                let e1 = _mm512_loadu_si512(transposed.elems.as_ptr().add(i + 4) as *const __m512i);
                let c1 = _mm512_loadu_si512(coeffs.as_ptr().add(i + 4) as *const __m512i);
                acc0.mul_acc(e0, c0);
                acc1.mul_acc(e1, c1);
            }
            let tail = acc0.fold() ^ acc1.fold();
            return tail.reduce();
        }
        #[allow(unreachable_code)]
        {
            let mut acc = F128::ZERO;
            for (e, c) in transposed.elems.iter().zip(coeffs.iter()) {
                acc += *e * *c;
            }
            acc
        }
    }
}

impl Add<&TensorAlgebra> for TensorAlgebra {
    type Output = TensorAlgebra;
    fn add(mut self, rhs: &TensorAlgebra) -> TensorAlgebra {
        self += rhs;
        self
    }
}

impl AddAssign<&TensorAlgebra> for TensorAlgebra {
    fn add_assign(&mut self, rhs: &TensorAlgebra) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use core::arch::x86_64::*;
            for i in (0..DEGREE).step_by(16) {
                let a0 = _mm512_loadu_si512(self.elems.as_ptr().add(i) as *const __m512i);
                let b0 = _mm512_loadu_si512(rhs.elems.as_ptr().add(i) as *const __m512i);
                let a1 = _mm512_loadu_si512(self.elems.as_ptr().add(i + 4) as *const __m512i);
                let b1 = _mm512_loadu_si512(rhs.elems.as_ptr().add(i + 4) as *const __m512i);
                let a2 = _mm512_loadu_si512(self.elems.as_ptr().add(i + 8) as *const __m512i);
                let b2 = _mm512_loadu_si512(rhs.elems.as_ptr().add(i + 8) as *const __m512i);
                let a3 = _mm512_loadu_si512(self.elems.as_ptr().add(i + 12) as *const __m512i);
                let b3 = _mm512_loadu_si512(rhs.elems.as_ptr().add(i + 12) as *const __m512i);
                _mm512_storeu_si512(self.elems.as_mut_ptr().add(i) as *mut __m512i, _mm512_xor_si512(a0, b0));
                _mm512_storeu_si512(self.elems.as_mut_ptr().add(i + 4) as *mut __m512i, _mm512_xor_si512(a1, b1));
                _mm512_storeu_si512(self.elems.as_mut_ptr().add(i + 8) as *mut __m512i, _mm512_xor_si512(a2, b2));
                _mm512_storeu_si512(self.elems.as_mut_ptr().add(i + 12) as *mut __m512i, _mm512_xor_si512(a3, b3));
            }
            return;
        }
        #[allow(unreachable_code)]
        {
            for (a, b) in self.elems.iter_mut().zip(rhs.elems.iter()) {
                *a = *a + *b;
            }
        }
    }
}

/// In-place 128×128 F_2 matrix transpose of the F128 coefficient table.
///
/// On input: `elems[i]` viewed as a 128-bit row; bit `j` is the F_2 coefficient
/// at position `(i, j)`.
/// On output: bit `j` of `elems[i]` becomes the old bit `i` of `elems[j]`.
///
#[inline]
fn transpose_64x64(src: &[u64], out: &mut [u64]) {
    debug_assert_eq!(src.len(), 64);
    debug_assert_eq!(out.len(), 64);
    let mut bytes = [0u8; 64];
    for chunk_r in 0..8 {
        let chunk: [u64; 8] = src[chunk_r * 8..chunk_r * 8 + 8].try_into().unwrap();
        crate::bits::transpose_8_u64s_to_64_bytes(&chunk, &mut bytes);
        for col in 0..64 {
            let byte_val = bytes[col] as u64;
            if chunk_r == 0 {
                out[col] = byte_val;
            } else {
                out[col] |= byte_val << (chunk_r * 8);
            }
        }
    }
}

/// Fast 128×128 bit matrix transpose using 4 vectorized 64×64 GFNI block transpositions.
fn square_transpose(elems: &mut [F128]) {
    assert_eq!(
        elems.len(),
        DEGREE,
        "square_transpose: input must be length 128"
    );

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    {
        let transposed = crate::pcs::ring_switch::tensor_algebra_transpose(elems);
        elems.copy_from_slice(&transposed);
        return;
    }

    #[allow(unreachable_code)]
    {
        let mut lo_lo = [0u64; 64];
        let mut hi_lo = [0u64; 64];
        let mut lo_hi = [0u64; 64];
        let mut hi_hi = [0u64; 64];

        let mut src_lo_0 = [0u64; 64];
        let mut src_lo_1 = [0u64; 64];
        let mut src_hi_0 = [0u64; 64];
        let mut src_hi_1 = [0u64; 64];

        for i in 0..64 {
            src_lo_0[i] = elems[i].lo;
            src_lo_1[i] = elems[64 + i].lo;
            src_hi_0[i] = elems[i].hi;
            src_hi_1[i] = elems[64 + i].hi;
        }

        transpose_64x64(&src_lo_0, &mut lo_lo);
        transpose_64x64(&src_lo_1, &mut lo_hi);
        transpose_64x64(&src_hi_0, &mut hi_lo);
        transpose_64x64(&src_hi_1, &mut hi_hi);

        for j in 0..64 {
            elems[j] = F128 { lo: lo_lo[j], hi: lo_hi[j] };
            elems[64 + j] = F128 { lo: hi_lo[j], hi: hi_hi[j] };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn nx(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.nx(),
                hi: self.nx(),
            }
        }
        fn ta(&mut self) -> TensorAlgebra {
            let elems = (0..DEGREE).map(|_| self.f128()).collect();
            TensorAlgebra { elems }
        }
    }

    #[test]
    fn transpose_involution() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..10 {
            let t = rng.ta();
            assert_eq!(t.clone().transpose().transpose(), t);
        }
    }

    #[test]
    fn transpose_bit_semantics() {
        // bit_j(elems[i]) on input becomes bit_i(elems[j]) on output.
        let mut rng = Rng::new(42);
        let original = rng.ta();
        let transposed = original.clone().transpose();

        fn bit(x: F128, b: usize) -> u64 {
            if b < 64 {
                (x.lo >> b) & 1
            } else {
                (x.hi >> (b - 64)) & 1
            }
        }

        for i in 0..DEGREE {
            for j in 0..DEGREE {
                let orig_ij = bit(original.elems[i], j);
                let trans_ji = bit(transposed.elems[j], i);
                assert_eq!(orig_ij, trans_ji, "transpose mismatch at (i={i}, j={j})");
            }
        }
    }

    #[test]
    fn from_vertical_scale_vertical() {
        // from_vertical(x).scale_vertical(y) should equal from_vertical(x*y).
        let mut rng = Rng::new(123);
        for _ in 0..10 {
            let x = rng.f128();
            let y = rng.f128();
            let lhs = TensorAlgebra::from_vertical(x).scale_vertical(y);
            let rhs = TensorAlgebra::from_vertical(x * y);
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn scale_horizontal_via_transpose() {
        // scale_horizontal(s) == transpose.scale_vertical(s).transpose by
        // construction, but verify the API works end-to-end.
        let mut rng = Rng::new(456);
        for _ in 0..10 {
            let t = rng.ta();
            let s = rng.f128();
            let via_api = t.clone().scale_horizontal(s);
            let manual = t.transpose().scale_vertical(s).transpose();
            assert_eq!(via_api, manual);
        }
    }

    #[test]
    fn add_is_xor_pairwise() {
        let mut rng = Rng::new(789);
        let a = rng.ta();
        let b = rng.ta();
        let sum = a.clone() + &b;
        for i in 0..DEGREE {
            let expected = F128 {
                lo: a.elems[i].lo ^ b.elems[i].lo,
                hi: a.elems[i].hi ^ b.elems[i].hi,
            };
            assert_eq!(sum.elems[i], expected);
        }
    }

    #[test]
    fn add_zero_is_identity() {
        let mut rng = Rng::new(1011);
        let t = rng.ta();
        let z = TensorAlgebra::zero();
        assert_eq!(t.clone() + &z, t);
    }

    #[test]
    fn one_from_vertical_one() {
        assert_eq!(
            TensorAlgebra::one(),
            TensorAlgebra::from_vertical(F128::ONE)
        );
    }

    #[test]
    fn scale_vertical_distributes_over_add() {
        // (a + b) * s == a*s + b*s
        let mut rng = Rng::new(1213);
        let a = rng.ta();
        let b = rng.ta();
        let s = rng.f128();
        let lhs = (a.clone() + &b).scale_vertical(s);
        let rhs = a.scale_vertical(s) + &b.scale_vertical(s);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn fold_vertical_with_zero_coeffs_is_zero() {
        let mut rng = Rng::new(1415);
        let t = rng.ta();
        let zeros = vec![F128::ZERO; DEGREE];
        assert_eq!(t.fold_vertical(&zeros), F128::ZERO);
    }
}
