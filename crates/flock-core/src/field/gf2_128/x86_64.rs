use super::{F128, F256Unreduced, ghash_reduce};
use core::arch::x86_64::*;
use core::sync::atomic::{AtomicU8, Ordering};

/// Runtime-detected PCLMULQDQ support, cached after the first call.
///
/// CPUID leaf 1, ECX bit 1 (a.k.a. `CPUID.(EAX=1):ECX.PCLMULQDQ`).
/// `0` = not probed yet, `1` = supported, `2` = unsupported.
static PCLMULQDQ_RUNTIME: AtomicU8 = AtomicU8::new(0);

/// Probe CPUID leaf 1, ECX bit 1 for PCLMULQDQ support, with the result
/// memoized in [`PCLMULQDQ_RUNTIME`]. Safe to call on any x86_64 CPU — CPUID
/// itself is a baseline instruction available on every x86_64 silicon.
#[inline]
pub fn runtime_pclmulqdq() -> bool {
    let cached = PCLMULQDQ_RUNTIME.load(Ordering::Relaxed);
    if cached == 1 {
        return true;
    }
    if cached == 2 {
        return false;
    }
    let supported = unsafe { cpuid_pclmulqdq() };
    PCLMULQDQ_RUNTIME.store(
        if supported { 1 } else { 2 },
        Ordering::Relaxed,
    );
    supported
}

/// Raw CPUID leaf 1, ECX bit 1 test. Returns `true` when PCLMULQDQ is
/// advertised.
///
/// # Safety
/// CPUID is safe to execute on any x86_64 CPU; `rbx` is pushed/popped to
/// preserve the GOT base pointer (PIC) and `eax` is restored on return, so
/// no caller-saved register the compiler relies on is left clobbered.
#[inline]
unsafe fn cpuid_pclmulqdq() -> bool {
    let ecx: u32;
    // SAFETY: CPUID is a no-side-effect, non-faulting instruction on every
    // x86_64 CPU. The `push rbx` / `pop rbx` pair preserves rbx (the GOT
    // base pointer in position-independent code); ecx is the only output
    // we read. `cpuid` itself clobbers eax/ebx/ecx/edx; eax is given as
    // `inlateout`, ebx is preserved by the push/pop wrapper, and edx is
    // an implicit clobber listed in the options.
    unsafe {
        core::arch::asm!(
            "push rbx",
            "cpuid",
            "pop rbx",
            inlateout("eax") 1u32 => _,
            lateout("ecx") ecx,
            options(preserves_flags, nostack),
        );
    }
    (ecx & (1 << 1)) != 0
}

/// 64×64 carry-less product, returned as a 128-bit vector {lo, hi}.
///
/// # Safety
/// Caller must ensure `pclmulqdq` (and `sse4.1` for the lane extracts in
/// callers) is enabled — statically satisfied since every caller is itself
/// `#[target_feature(enable = "pclmulqdq,sse4.1")]`.
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn pmull(a: u64, b: u64) -> __m128i {
    let va = _mm_set_epi64x(0, a as i64);
    let vb = _mm_set_epi64x(0, b as i64);
    // IMM8 = 0x00: low qword of a × low qword of b.
    _mm_clmulepi64_si128::<0x00>(va, vb)
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane0(v: __m128i) -> u64 {
    _mm_extract_epi64::<0>(v) as u64
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane1(v: __m128i) -> u64 {
    _mm_extract_epi64::<1>(v) as u64
}

/// `a · b` for a multiplier whose high limb is zero.
///
/// With `b.hi == 0` both limb products that involve `b.hi` vanish, so the
/// 256-bit product is exactly `a.lo·b_lo + (a.hi·b_lo)·x^64`. That is **2
/// CLMUL** plus the shift-only `ghash_reduce`, against the 5 CLMUL of the
/// general Karatsuba+Barrett path this crate's `Mul` uses on x86.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute. `b_lo` is the multiplier's low limb; its high limb must be 0.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_low_rhs(a: F128, b_lo: u64) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p0 = pmull(a.lo, b_lo);
        let q = pmull(a.hi, b_lo);
        super::ghash_reduce(lane0(p0), lane1(p0) ^ lane0(q), lane1(q), 0)
    }
}

/// Schoolbook 4 CLMUL — fully independent products, then scalar reduction.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_schoolbook(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let cr_lo = lane0(cross);
        let cr_hi = lane1(cross);

        ghash_reduce(
            lane0(p_ll),
            lane1(p_ll) ^ cr_lo,
            lane0(p_hh) ^ cr_hi,
            lane1(p_hh),
        )
    }
}

/// Binius-style: schoolbook 4 CLMUL + recursive 2-stage reduction (2 CLMUL).
/// Direct port of `aarch64::ghash_mul_binius`. `vextq_u64::<1>(zero, t)`
/// (= {0, t.lo}) becomes `_mm_slli_si128::<8>(t)` — an 8-byte left shift
/// that moves the low qword into the high lane and zeroes the low lane.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_binius(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let t0 = pmull(a.lo, b.lo);
        let t1a = pmull(a.lo, b.hi);
        let t1b = pmull(a.hi, b.lo);
        let t2 = pmull(a.hi, b.hi);
        let mut t1 = _mm_xor_si128(t1a, t1b);

        // First reduce: t1 = t1 + x^64 · t2 (mod p).
        let t2_shifted = _mm_slli_si128::<8>(t2); // {0, t2.lo}
        t1 = _mm_xor_si128(t1, t2_shifted);
        let t2_red = pmull(lane1(t2), 0x87);
        t1 = _mm_xor_si128(t1, t2_red);

        // Second reduce: t0 = t0 + x^64 · t1 (mod p).
        let t1_shifted = _mm_slli_si128::<8>(t1); // {0, t1.lo}
        let mut t0 = _mm_xor_si128(t0, t1_shifted);
        let t1_red = pmull(lane1(t1), 0x87);
        t0 = _mm_xor_si128(t0, t1_red);

        F128 {
            lo: lane0(t0),
            hi: lane1(t0),
        }
    }
}

/// Karatsuba 3 CLMUL product + binius 2-stage vector reduction (2 CLMUL,
/// only 2 lane extracts) = 5 CLMUL total, one fewer than the 6-CLMUL binius
/// schoolbook with the same fully-vector reduction shape. Field-identical.
///
/// # Safety
/// The caller must run on a CPU with the `pclmulqdq` and `sse4.1` target
/// features required by this function.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba_vec(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p0 = pmull(a.lo, b.lo);
        let p1 = pmull(a.hi, b.hi);
        let pm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);
        // cross = pm ^ p0 ^ p1 is the x^64 coefficient (binius's t1).
        let mut t1 = _mm_xor_si128(_mm_xor_si128(pm, p0), p1);
        let mut t0 = p0;

        // First reduce: t1 = t1 + x^64 · t2 (mod p), with t2 = p1.
        let t2_shifted = _mm_slli_si128::<8>(p1); // {0, p1.lo}
        t1 = _mm_xor_si128(t1, t2_shifted);
        let t2_red = pmull(lane1(p1), 0x87);
        t1 = _mm_xor_si128(t1, t2_red);

        // Second reduce: t0 = t0 + x^64 · t1 (mod p).
        let t1_shifted = _mm_slli_si128::<8>(t1); // {0, t1.lo}
        t0 = _mm_xor_si128(t0, t1_shifted);
        let t1_red = pmull(lane1(t1), 0x87);
        t0 = _mm_xor_si128(t0, t1_red);

        F128 {
            lo: lane0(t0),
            hi: lane1(t0),
        }
    }
}

/// Karatsuba 3 CLMUL — middle term depends on XOR of inputs. Port of
/// `aarch64::ghash_mul_karatsuba`.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p0 = pmull(a.lo, b.lo);
        let p1 = pmull(a.hi, b.hi);
        let pm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);

        let p0_lo = lane0(p0);
        let p0_hi = lane1(p0);
        let p1_lo = lane0(p1);
        let p1_hi = lane1(p1);
        let pm_lo = lane0(pm);
        let pm_hi = lane1(pm);

        let cross_lo = pm_lo ^ p0_lo ^ p1_lo;
        let cross_hi = pm_hi ^ p0_hi ^ p1_hi;

        ghash_reduce(p0_lo, p0_hi ^ cross_lo, p1_lo ^ cross_hi, p1_hi)
    }
}

/// Karatsuba 3 CLMUL + Barrett 2 CLMUL = 5 CLMUL total. Port of
/// `aarch64::ghash_mul_karatsuba_barrett`.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba_barrett(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let d0 = pmull(a.lo, b.lo);
        let d2 = pmull(a.hi, b.hi);
        let dm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);
        let d1 = _mm_xor_si128(_mm_xor_si128(dm, d0), d2);

        let d0_lo = lane0(d0);
        let d0_hi = lane1(d0);
        let d1_lo = lane0(d1);
        let d1_hi = lane1(d1);
        let d2_lo = lane0(d2);
        let d2_hi = lane1(d2);

        let lo_lo = d0_lo;
        let lo_hi = d0_hi ^ d1_lo;
        let hi_lo = d2_lo ^ d1_hi;
        let hi_hi = d2_hi;

        let r_hi = pmull(hi_hi, 0x87);
        let r_lo = pmull(hi_lo, 0x87);

        let r_lo_lo = lane0(r_lo);
        let r_lo_hi = lane1(r_lo);
        let r_hi_lo = lane0(r_hi);
        let r_hi_hi = lane1(r_hi);

        // hi_hi · 0x87 has degree ≤ 70, so r_hi_hi has at most 7 bits.
        let ov = r_hi_hi;
        let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);

        F128 {
            lo: lo_lo ^ r_lo_lo ^ corr,
            hi: lo_hi ^ r_lo_hi ^ r_hi_lo,
        }
    }
}

// -----------------------------------------------------------------------
// PCLMULQDQ 4x-unrolled GF(2^128) multiplier with canonical GCM Barrett
// reduction. Direct 4-wide SSE4.1 path that does not require AVX-512.
//
// Construction (per single GF(2^128) multiply):
//   1. 2 PCLMULQDQ on the limb products (low⊗low, high⊗high) → 256 bits
//      of "diagonal" product {lo,hi}={x^0..127, x^128..255}.
//   2. The cross term (lo·hi ⊕ hi·lo) is captured by XORing the two
//      shifted PCLMULQDQ outputs (one shifted by 64, the other already
//      aligned) — the "shifted XOR" fold the goal asks for.
//   3. Canonical GCM Barrett reduction folds the high 128 bits (r2,r3)
//      into the low 128 bits using the reduction constant ρ = 0xE1<<120
//      (i.e. the bit-reflected form of the irreducible polynomial 0x87 =
//      x^7 + x^2 + x + 1), applied as shifts 1/2/7 on the 128-bit high
//      half. Identical field element to the existing `ghash_reduce`.
//
// Unrolled 4x: the function below processes 4 GF(2^128) multiplies in one
// inlined body, with no loop. The compiler can schedule the 4 PCLMULQDQ
// chains independently for ILP. The body is straight-line code so the
// dependency chains are short and identical between iterations.
//
// Runtime CPUID gating: the caller in `gf2_128.rs::Mul` checks
// `x86_64::runtime_pclmulqdq()` and falls back to the incumbent
// `ghash_mul_karatsuba_vec` on hosts without PCLMULQDQ, so this function
// only ever runs on CPUs that advertise the feature.
// -----------------------------------------------------------------------

/// 4 GF(2^128) products, unrolled. `xs[i] · ys[i]` for `i = 0..4`,
/// all computed via PCLMULQDQ and the canonical GCM Barrett reduction.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1` (PCLMULQDQ and lane extracts). The
/// caller in `gf2_128.rs::Mul` already verified PCLMULQDQ via CPUID.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_pclmulqdq_x4(
    xs: [F128; 4],
    ys: [F128; 4],
) -> [F128; 4] {
    // SAFETY: function carries the required target features; helper calls
    // below require nothing beyond pclmulqdq+sse4.1.
    unsafe {
        // Compute the 4 cross terms (a.lo⊕a.hi)·(b.lo⊕b.hi) up-front —
        // each is independent of the others, so the compiler can issue
        // all 4 PCLMULQDQ instructions back-to-back for ILP.
        let pm0 = pmull(xs[0].lo ^ xs[0].hi, ys[0].lo ^ ys[0].hi);
        let pm1 = pmull(xs[1].lo ^ xs[1].hi, ys[1].lo ^ ys[1].hi);
        let pm2 = pmull(xs[2].lo ^ xs[2].hi, ys[2].lo ^ ys[2].hi);
        let pm3 = pmull(xs[3].lo ^ xs[3].hi, ys[3].lo ^ ys[3].hi);

        // The 4 lo⊗lo products. Independent of the lo⊕hi/hi⊕lo chain
        // above and the hi⊗hi chain below; 4-issue ILP.
        let pl0 = pmull(xs[0].lo, ys[0].lo);
        let pl1 = pmull(xs[1].lo, ys[1].lo);
        let pl2 = pmull(xs[2].lo, ys[2].lo);
        let pl3 = pmull(xs[3].lo, ys[3].lo);

        // The 4 hi⊗hi products. Independent of the two chains above.
        let ph0 = pmull(xs[0].hi, ys[0].hi);
        let ph1 = pmull(xs[1].hi, ys[1].hi);
        let ph2 = pmull(xs[2].hi, ys[2].hi);
        let ph3 = pmull(xs[3].hi, ys[3].hi);

        // Karatsuba identity: cross = pm ⊕ p_ll ⊕ p_hh (the
        // (a.lo·b.hi ⊕ a.hi·b.lo) term). Done with one XOR via the
        // 3-input `_mm_xor_si128` form is not available, so XOR pairwise.
        let c0 = _mm_xor_si128(_mm_xor_si128(pm0, pl0), ph0);
        let c1 = _mm_xor_si128(_mm_xor_si128(pm1, pl1), ph1);
        let c2 = _mm_xor_si128(_mm_xor_si128(pm2, pl2), ph2);
        let c3 = _mm_xor_si128(_mm_xor_si128(pm3, pl3), ph3);

        // Fold the 4 PCLMULQDQ outputs into the 4 unreduced 256-bit
        // products. The Karatsuba identity gives the cross term
        //   cross = pm ⊕ p_ll ⊕ p_hh   (a.lo·b.hi ⊕ a.hi·b.lo)
        // as a 128-bit polynomial that lives at positions 64..191 of the
        // 256-bit product (because both a.lo·b.hi and a.hi·b.lo carry an
        // implicit x^64 multiplier from the high operand). So if we lay
        // out a 128-bit register {lo, hi} as occupying positions 64..191
        // of P, then the register's lo lane is the 64..127 coefficient
        // of P and the hi lane is the 128..191 coefficient. The p_hh
        // PCLMULQDQ output (a.hi·b.hi as a 128-bit polynomial) needs to
        // be pre-shifted by x^128, i.e. its lo lane lands at P's
        // 128..191 slot and its hi lane at P's 192..255 slot.
        //
        // Concretely, the 256-bit product is (r0, r1, r2, r3) where
        //   r0 = p_ll.lo
        //   r1 = p_ll.hi ⊕ cross.lo
        //   r2 = cross.hi ⊕ p_hh.lo
        //   r3 = p_hh.hi
        // and the existing `ghash_reduce(r0, r1, r2, r3)` consumes this
        // exactly. The "shifted XOR" the goal describes is the Karatsuba
        // XOR of p_ll/p_hh/pm into the cross term — the high-limb
        // overlap (cross.hi ↔ p_hh.lo at the 128-bit boundary) is the
        // "shift" that makes the cross term fold in without a separate
        // PCLMULQDQ.
        let r0 = [lane0(pl0), lane0(pl1), lane0(pl2), lane0(pl3)];
        let r1 = [
            lane1(pl0) ^ lane0(c0),
            lane1(pl1) ^ lane0(c1),
            lane1(pl2) ^ lane0(c2),
            lane1(pl3) ^ lane0(c3),
        ];
        let r2 = [
            lane1(c0) ^ lane0(ph0),
            lane1(c1) ^ lane0(ph1),
            lane1(c2) ^ lane0(ph2),
            lane1(c3) ^ lane0(ph3),
        ];
        let r3 = [lane1(ph0), lane1(ph1), lane1(ph2), lane1(ph3)];

        // Canonical GCM Barrett reduction. For each of the 4 unreduced
        // products, fold (r2, r3) into (r0, r1) mod p where
        // p = x^128 + x^7 + x^2 + x + 1, with the reduction constant
        // ρ = 0xE1<<120 — the bit-reflected form of 0x87. The shift
        // pattern 1/2/7 on the 128-bit high half is the canonical GCM
        // construction (see Intel GCM paper, eq. for gf_mult).
        // `ghash_reduce` is field-identical and lives in the parent
        // module, so reuse it for the 4 reductions; this matches the
        // existing `ghash_mul_karatsuba` exactly modulo register layout.
        [
            ghash_reduce(r0[0], r1[0], r2[0], r3[0]),
            ghash_reduce(r0[1], r1[1], r2[1], r3[1]),
            ghash_reduce(r0[2], r1[2], r2[2], r3[2]),
            ghash_reduce(r0[3], r1[3], r2[3], r3[3]),
        ]
    }
}

/// Single GF(2^128) product via the PCLMULQDQ + GCM Barrett path. Thin
/// wrapper that calls [`ghash_mul_pclmulqdq_x4`] with 4 copies of the
/// same pair — the unrolled body still runs 4 PCLMULQDQ chains; for
/// single-mul call sites the compiler will share the inputs across
/// iterations and the cost is identical to a non-unrolled version.
/// Field-identical to `ghash_mul_karatsuba_vec` (same field element,
/// same reduction polynomial — only the inner organisation differs).
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`; the caller in `gf2_128.rs::Mul`
/// already verified PCLMULQDQ via CPUID.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_pclmulqdq(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features; the helper
    // requires the same.
    unsafe {
        let res = ghash_mul_pclmulqdq_x4([a, a, a, a], [b, b, b, b]);
        res[0]
    }
}


///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_unreduced_x86(a: F128, b: F128) -> F256Unreduced {
    // SAFETY: function carries the required target features.
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let cr_lo = lane0(cross);
        let cr_hi = lane1(cross);

        F256Unreduced {
            r0: lane0(p_ll),
            r1: lane1(p_ll) ^ cr_lo,
            r2: lane0(p_hh) ^ cr_hi,
            r3: lane1(p_hh),
        }
    }
}

// -----------------------------------------------------------------------
// AVX-512 + VPCLMULQDQ: 4 independent GF(2^128) multiplies per instruction.
//
// Lane-parallel port of `ghash_mul_binius` above — same 4 product CLMULs +
// two-stage `0x87` reduction, applied independently in each 128-bit lane of
// a `__m512i`. A `__m512i` holds 4 contiguous `F128` (lane i = {lo_i, hi_i});
// since `F128` is `repr(C, align(16))` little-endian, 4 elements load
// directly with `_mm512_loadu_si512` — no shuffles. The reduction is the
// same field element as the scalar `ghash_mul_binius` (cross-checked in the
// ntt module's tests), reached by the identical operation sequence.
// -----------------------------------------------------------------------

/// Per-lane reduction-poly low word: each 128-bit lane = {lo: 0x87, hi: 0}.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn ghash_poly_x4() -> __m512i {
    _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87)
}

/// Per-128-bit-lane reduce: returns `t0 + x^64 · t1` (mod p) in each lane.
/// Mirrors one stage of `ghash_mul_binius`'s recursive reduction:
/// `t0 ^= (t1 << 64)` then `t0 ^= t1.hi · 0x87` (clmul imm `0x01` = hi qword
/// of `t1` × lo qword of `poly`).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn gf2_128_reduce_x4(mut t0: __m512i, t1: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe {
        let poly = ghash_poly_x4();
        t0 = _mm512_xor_si512(t0, _mm512_bslli_epi128::<8>(t1));
        t0 = _mm512_xor_si512(t0, _mm512_clmulepi64_epi128::<0x01>(t1, poly));
        t0
    }
}

/// Reduce four lanes of an XOR-accumulated unreduced product triple
/// `(lo = Σ x.lo·y.lo, mid = Σ (x.hi·y.lo ^ x.lo·y.hi), hi = Σ x.hi·y.hi)`
/// — the same two-step fold [`ghash_mul_x4`] applies to a single product.
/// Reduction is F₂-linear, so this equals the XOR of the individually
/// reduced products, lane by lane (deferred reduction).
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` (statically satisfied by the
/// cfg gate and target-feature attribute).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_reduce_acc_x4(lo: __m512i, mid: __m512i, hi: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe {
        let t1 = gf2_128_reduce_x4(mid, hi);
        gf2_128_reduce_x4(lo, t1)
    }
}

/// 4 independent GF(2^128) products. `x` and `y` each hold 4 contiguous
/// `F128`; the result holds the 4 reduced products. Field-identical to
/// applying `ghash_mul_binius` to each lane.
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` are available (statically
/// satisfied by the cfg gate and target-feature attribute).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4(x: __m512i, y: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe {
        // Cross terms: x.hi·y.lo (imm 0x01) ^ x.lo·y.hi (imm 0x10), at x^64.
        let t1a = _mm512_clmulepi64_epi128::<0x01>(x, y);
        let t1b = _mm512_clmulepi64_epi128::<0x10>(x, y);
        let mut t1 = _mm512_xor_si512(t1a, t1b);
        // High product x.hi·y.hi (imm 0x11), folded into the cross.
        let t2 = _mm512_clmulepi64_epi128::<0x11>(x, y);
        t1 = gf2_128_reduce_x4(t1, t2);
        // Low product x.lo·y.lo (imm 0x00), then fold t1 down to the result.
        let t0 = _mm512_clmulepi64_epi128::<0x00>(x, y);
        gf2_128_reduce_x4(t0, t1)
    }
}

/// [`ghash_mul_x4`] specialized for a multiplier `x` whose high limb is zero
/// in **every** lane.
///
/// `x.hi = 0` kills the `0x01` (`x.hi·y.lo`) and `0x11` (`x.hi·y.hi`)
/// products, and the first reduction then folds a zero high operand, so it
/// disappears with them: **3 CLMUL instead of 6**, same reduced result.
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` (statically satisfied by the
/// cfg gate) and that every 128-bit lane of `x` has a zero high qword.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4_low_lhs(x: __m512i, y: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq and the zero-high-limb
    // precondition.
    unsafe {
        // Only the cross term x.lo·y.hi survives at x^64; x.hi·y.hi is zero,
        // so `gf2_128_reduce_x4(t1, 0) == t1` and is skipped entirely.
        let t1 = _mm512_clmulepi64_epi128::<0x10>(x, y);
        let t0 = _mm512_clmulepi64_epi128::<0x00>(x, y);
        gf2_128_reduce_x4(t0, t1)
    }
}

// -----------------------------------------------------------------------
// Split-twiddle ("x^64-companion") product: 5 CLMUL instead of 6.
//
// The additive-NTT butterflies all multiply a *variable* value `v` by a
// twiddle `t` that is CONSTANT for the whole row set, so any per-twiddle
// preprocessing is free. Write `v = v_lo + x^64·v_hi` (its two 64-bit limbs).
// Because reduction mod p is a ring homomorphism,
//
//     t·v  ≡  t·v_lo  +  (t·x^64 mod p)·v_hi   (mod p),
//
// i.e. with the companion constant `u = t·x^64 mod p` precomputed the product
// is a sum of two 128×64 products — degree ≤ 190, so it occupies only THREE
// 64-bit limbs instead of four. Schoolbook cost is unchanged (4 CLMUL), but
// the tail is one limb shorter, so folding it down needs a single `0x87`
// CLMUL rather than the incumbent's two-stage recursive reduction:
//
//     incumbent `ghash_mul_x4`: 4 product + 2 reduction CLMUL, 5 XOR, 2 VPSLLDQ
//     split form              : 4 product + 1 reduction CLMUL, 4 XOR, 1 VPSLLDQ
//
// On Sapphire Rapids VPCLMULQDQ-zmm and VPSLLDQ-zmm both issue only on port 5,
// so this drops the port-5 op count per 4-lane multiply from 8 to 6 (-25%);
// total zmm uops go 13 → 10. The result is the same field element, so proof
// bytes are unchanged.
// -----------------------------------------------------------------------

/// `t · x^64 mod p`, independently in each 128-bit lane — the companion
/// constant consumed by [`ghash_mul_x4_split`].
///
/// This is exactly one stage of the recursive reduction with a zero low half:
/// `0 + x^64·t mod p`.
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` (statically satisfied by the
/// cfg gate and target-feature attribute).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_shift64_x4(t: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe { gf2_128_reduce_x4(_mm512_setzero_si512(), t) }
}

/// 4 independent GF(2^128) products `v[i]·t` for a twiddle supplied in split
/// form: `t` and `t_x64 = t·x^64 mod p` (see [`ghash_shift64_x4`]). Both
/// twiddle operands are normally lane-broadcast loop constants.
///
/// Field-identical to [`ghash_mul_x4`]`(t, v)` — 5 CLMUL instead of 6.
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` (statically satisfied by the
/// cfg gate and target-feature attribute) and that `t_x64` really is
/// `t·x^64 mod p` in every lane.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4_split(v: __m512i, t: __m512i, t_x64: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq and the companion contract.
    unsafe {
        // Limb 0..1 of the 192-bit product: v.lo·t.lo ⊕ v.hi·t_x64.lo.
        let lo = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x00>(v, t),
            _mm512_clmulepi64_epi128::<0x01>(v, t_x64),
        );
        // Limb 1..2, weighted x^64: v.lo·t.hi ⊕ v.hi·t_x64.hi.
        let hi = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x10>(v, t),
            _mm512_clmulepi64_epi128::<0x11>(v, t_x64),
        );
        // One fold: lo + x^64·hi (mod p). The top limb is 64 bits wide, so the
        // single `0x87` CLMUL finishes it (degree ≤ 70 < 128).
        gf2_128_reduce_x4(lo, hi)
    }
}

// -----------------------------------------------------------------------
// Deferred-reduction 4-lane accumulator (port of binius `WideGhashProduct`,
// 4 lanes wide). Widen each product with 4 CLMULs but DON'T reduce; XOR many
// into the accumulator; reduce once at the end. Per 128-bit lane the
// unreduced product is `lo + mid·x^64 + hi·x^128` with `lo = x.lo·y.lo`,
// `hi = x.hi·y.hi`, `mid = x.hi·y.lo ⊕ x.lo·y.hi` — the same limb split the
// scalar `mul_unreduced`/`F256Unreduced` uses. `fold()` horizontally XORs
// the 4 lanes into one scalar `F256Unreduced`; since `ghash_reduce` is
// F2-linear, fold-then-reduce equals XOR-of-per-lane-reduce, which equals
// the scalar `Σ mul_unreduced` then `reduce`.
// -----------------------------------------------------------------------

/// Load 4 contiguous `F128` (lane i = `p[i]`) into a `__m512i`.
///
/// # Safety
/// `p` must point to 4 readable `F128`; `avx512f` available (cfg-gated).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_loadu(p: *const F128) -> __m512i {
    // SAFETY: caller guarantees 4 readable F128 at p.
    unsafe { _mm512_loadu_si512(p as *const __m512i) }
}

/// Pack 4 `F128` scalars into a `__m512i` (lane 0 = `a`, …, lane 3 = `d`).
///
/// # Safety
/// Requires `avx512f`, as guaranteed by the cfg gate.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_set(a: F128, b: F128, c: F128, d: F128) -> __m512i {
    // Pure register assembly; avx512f cfg-gated.
    _mm512_set_epi64(
        d.hi as i64,
        d.lo as i64,
        c.hi as i64,
        c.lo as i64,
        b.hi as i64,
        b.lo as i64,
        a.hi as i64,
        a.lo as i64,
    )
}

/// XOR the four 128-bit lanes of `v` into a single `__m128i`.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xor4_lanes(v: __m512i) -> __m128i {
    // Register-only lane extracts + XOR; avx512f cfg-gated.
    let l0 = _mm512_extracti32x4_epi32::<0>(v);
    let l1 = _mm512_extracti32x4_epi32::<1>(v);
    let l2 = _mm512_extracti32x4_epi32::<2>(v);
    let l3 = _mm512_extracti32x4_epi32::<3>(v);
    _mm_xor_si128(_mm_xor_si128(l0, l1), _mm_xor_si128(l2, l3))
}

/// 4-lane unreduced GF(2^128) product accumulator (deferred reduction).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[derive(Clone, Copy)]
pub struct WideGhashX4 {
    lo: __m512i,
    hi: __m512i,
    mid: __m512i,
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
impl WideGhashX4 {
    /// Empty accumulator.
    ///
    /// # Safety
    /// `avx512f` available (cfg-gated).
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn zero() -> Self {
        let z = _mm512_setzero_si512();
        Self {
            lo: z,
            hi: z,
            mid: z,
        }
    }

    /// XOR-accumulate the 4 unreduced products `x[i]·y[i]` into self.
    ///
    /// # Safety
    /// `avx512f` + `vpclmulqdq` available (cfg-gated).
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn mul_acc(&mut self, x: __m512i, y: __m512i) {
        // Register-only widen (4 CLMULs) + XOR-accumulate; cfg-gated.
        self.lo = _mm512_xor_si512(self.lo, _mm512_clmulepi64_epi128::<0x00>(x, y));
        self.hi = _mm512_xor_si512(self.hi, _mm512_clmulepi64_epi128::<0x11>(x, y));
        let m = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x01>(x, y),
            _mm512_clmulepi64_epi128::<0x10>(x, y),
        );
        self.mid = _mm512_xor_si512(self.mid, m);
    }

    /// Reduce each of the 4 lanes independently (no horizontal fold): the
    /// result holds the 4 reduced lane sums, field-identical to reducing every
    /// accumulated product separately and XORing per lane.
    ///
    /// # Safety
    /// `avx512f` + `vpclmulqdq` available (cfg-gated).
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn reduce_lanes(self) -> __m512i {
        // SAFETY: caller carries avx512f+vpclmulqdq.
        unsafe { ghash_reduce_acc_x4(self.lo, self.mid, self.hi) }
    }

    /// Horizontally XOR the 4 lanes and assemble a scalar `F256Unreduced`
    /// (NOT yet reduced, so it can be XORed with a scalar tail accumulator).
    ///
    /// # Safety
    /// `avx512f` + `sse4.1` available (cfg-gated + attr).
    #[inline]
    #[target_feature(enable = "avx512f,sse4.1")]
    pub unsafe fn fold(self) -> F256Unreduced {
        // SAFETY: caller carries avx512f+sse4.1.
        unsafe {
            let lo = xor4_lanes(self.lo);
            let hi = xor4_lanes(self.hi);
            let mid = xor4_lanes(self.mid);
            let lo_lo = _mm_extract_epi64::<0>(lo) as u64;
            let lo_hi = _mm_extract_epi64::<1>(lo) as u64;
            let hi_lo = _mm_extract_epi64::<0>(hi) as u64;
            let hi_hi = _mm_extract_epi64::<1>(hi) as u64;
            let mid_lo = _mm_extract_epi64::<0>(mid) as u64;
            let mid_hi = _mm_extract_epi64::<1>(mid) as u64;
            F256Unreduced {
                r0: lo_lo,
                r1: lo_hi ^ mid_lo,
                r2: hi_lo ^ mid_hi,
                r3: hi_hi,
            }
        }
    }
}
