use crate::field::F128;

/// `FLOCK_NO_NTT_MUL_DIET=1` restores the incumbent 6-CLMUL `ghash_mul_x4`
/// butterfly multiply inside the same binary, so a candidate/control pair
/// differs only in the twiddle-product form. Read once, outside every lane
/// loop.
#[inline]
fn mul_diet_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_NTT_MUL_DIET").is_some())
}

/// A butterfly twiddle broadcast into all four 128-bit lanes, in the split
/// form [`crate::field::gf2_128::x86_64::ghash_mul_x4_split`] consumes:
/// `.0 = t`, `.1 = t·x^64 mod p`. The companion limb is only materialised
/// when the diet multiply will actually read it; otherwise it aliases `.0`
/// and is dropped by DCE.
type TwX4 = (core::arch::x86_64::__m512i, core::arch::x86_64::__m512i);

/// Broadcast `value` to all lanes and, when the split product will be used,
/// derive its `x^64` companion. Every caller does this OUTSIDE its lane loop,
/// so the extra CLMUL is amortised over the whole row set.
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq`.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn tw_x4<const LOW: bool, const DIET: bool>(value: F128) -> TwX4 {
    use crate::field::gf2_128::x86_64::ghash_shift64_x4;
    use core::arch::x86_64::*;

    // SAFETY: caller carries the required target features.
    unsafe {
        let t = _mm512_broadcast_i32x4(_mm_set_epi64x(value.hi as i64, value.lo as i64));
        // The LOW kernel needs no companion: its 3-CLMUL form is already the
        // short-product shape the split form generalises.
        let companion = if DIET && !LOW { ghash_shift64_x4(t) } else { t };
        (t, companion)
    }
}

/// Broadcast-twiddle product. `LOW` asserts the twiddle's high limb is zero
/// in every lane (3 CLMUL); otherwise `DIET` picks the 5-CLMUL split form
/// over the incumbent 6-CLMUL `ghash_mul_x4`. Monomorphized, so the choice
/// costs no branch inside the lane loop.
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq`; when `LOW`, every 128-bit lane of `t.0`
/// must have a zero high qword; when `DIET && !LOW`, `t.1` must be
/// `t.0·x^64 mod p`.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn mul_x4<const LOW: bool, const DIET: bool>(
    t: TwX4,
    v: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4, ghash_mul_x4_low_lhs, ghash_mul_x4_split};
    // SAFETY: caller carries the features and the twiddle-form preconditions.
    unsafe {
        if LOW {
            ghash_mul_x4_low_lhs(t.0, v)
        } else if DIET {
            ghash_mul_x4_split(v, t.0, t.1)
        } else {
            ghash_mul_x4(t.0, v)
        }
    }
}

#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    // SAFETY: forwarded caller contract.
    unsafe { butterfly_row_pair_gen::<false>(top, bot, twiddle) }
}

/// Broadcast-twiddle row pair; `LOW` asserts the twiddle's high limb is zero.
///
/// # Safety
/// Requires `avx512f` + `vpclmulqdq`; when `LOW`, `twiddle.hi == 0`.
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_row_pair_gen<const LOW: bool>(
    top: &mut [F128],
    bot: &mut [F128],
    twiddle: F128,
) {
    // SAFETY: forwarded caller contract; the switch only picks the product
    // form, which is field-identical either way.
    unsafe {
        if mul_diet_disabled() {
            butterfly_row_pair_impl::<LOW, false>(top, bot, twiddle)
        } else {
            butterfly_row_pair_impl::<LOW, true>(top, bot, twiddle)
        }
    }
}

/// # Safety
/// Same contract as [`butterfly_row_pair_gen`].
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_row_pair_impl<const LOW: bool, const DIET: bool>(
    top: &mut [F128],
    bot: &mut [F128],
    twiddle: F128,
) {
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and equal slice lengths.
    unsafe {
        debug_assert!(!LOW || twiddle.hi == 0);
        let tw = tw_x4::<LOW, DIET>(twiddle);
        let lanes = top.len() & !3;
        let mut i = 0;
        while i < lanes {
            let top_lanes = _mm512_loadu_si512(top.as_ptr().add(i) as *const __m512i);
            let bot_lanes = _mm512_loadu_si512(bot.as_ptr().add(i) as *const __m512i);
            let new_top = _mm512_xor_si512(top_lanes, mul_x4::<LOW, DIET>(tw, bot_lanes));
            let new_bot = _mm512_xor_si512(bot_lanes, new_top);
            _mm512_storeu_si512(top.as_mut_ptr().add(i) as *mut __m512i, new_top);
            _mm512_storeu_si512(bot.as_mut_ptr().add(i) as *mut __m512i, new_bot);
            i += 4;
        }
        super::portable::butterfly_row_pair_gen::<LOW>(&mut top[i..], &mut bot[i..], twiddle);
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        butterfly_fused_2layer_gen::<false, false>(a, b, c, d, t_outer, t_inner_a, t_inner_b)
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer`], plus: when `OUTER_LOW`,
/// `t_outer.hi == 0`; when `INNER_LOW`, both inner twiddles have a zero high
/// limb.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_gen<const OUTER_LOW: bool, const INNER_LOW: bool>(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_2layer_impl::<OUTER_LOW, INNER_LOW, false>(
                a, b, c, d, t_outer, t_inner_a, t_inner_b,
            )
        } else {
            butterfly_fused_2layer_impl::<OUTER_LOW, INNER_LOW, true>(
                a, b, c, d, t_outer, t_inner_a, t_inner_b,
            )
        }
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer_gen`].
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_impl<
    const OUTER_LOW: bool,
    const INNER_LOW: bool,
    const DIET: bool,
>(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and equal slice lengths.
    unsafe {
        let outer = tw_x4::<OUTER_LOW, DIET>(t_outer);
        let inner_a = tw_x4::<INNER_LOW, DIET>(t_inner_a);
        let inner_b = tw_x4::<INNER_LOW, DIET>(t_inner_b);
        let lanes = a.len() & !3;
        let mut i = 0;
        while i < lanes {
            let mut va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let mut vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            let mut vc = _mm512_loadu_si512(c.as_ptr().add(i) as *const __m512i);
            let mut vd = _mm512_loadu_si512(d.as_ptr().add(i) as *const __m512i);

            let new_a = _mm512_xor_si512(va, mul_x4::<OUTER_LOW, DIET>(outer, vc));
            vc = _mm512_xor_si512(vc, new_a);
            va = new_a;
            let new_b = _mm512_xor_si512(vb, mul_x4::<OUTER_LOW, DIET>(outer, vd));
            vd = _mm512_xor_si512(vd, new_b);
            vb = new_b;

            let new_a = _mm512_xor_si512(va, mul_x4::<INNER_LOW, DIET>(inner_a, vb));
            vb = _mm512_xor_si512(vb, new_a);
            va = new_a;
            let new_c = _mm512_xor_si512(vc, mul_x4::<INNER_LOW, DIET>(inner_b, vd));
            vd = _mm512_xor_si512(vd, new_c);
            vc = new_c;

            _mm512_storeu_si512(a.as_mut_ptr().add(i) as *mut __m512i, va);
            _mm512_storeu_si512(b.as_mut_ptr().add(i) as *mut __m512i, vb);
            _mm512_storeu_si512(c.as_mut_ptr().add(i) as *mut __m512i, vc);
            _mm512_storeu_si512(d.as_mut_ptr().add(i) as *mut __m512i, vd);
            i += 4;
        }
        super::portable::butterfly_fused_2layer_gen::<OUTER_LOW, INNER_LOW>(
            &mut a[i..],
            &mut b[i..],
            &mut c[i..],
            &mut d[i..],
            t_outer,
            t_inner_a,
            t_inner_b,
        );
    }
}

/// Out-of-place fused two-layer forward butterfly (layers 1–2 seed).
/// Same algebra as [`butterfly_fused_2layer`], loads from `src` and stores
/// to `dst`. Source and destination must not overlap.
///
/// # Safety
/// Caller guarantees target features, valid non-aliasing src/dst rows, and
/// disjoint destination row groups across concurrent calls.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 3],
) {
    // SAFETY: forwarded caller contract; identical geometry on both sides.
    unsafe {
        butterfly_fused_2layer_row_from_geo(
            src, quarter, r, dst, quarter, r, num_ntts, twiddles,
        )
    }
}

/// [`butterfly_fused_2layer_row_from`] with independent source and
/// destination row geometry: source rows `(i·src_quarter + src_r)`,
/// destination rows `(i·dst_quarter + dst_r)`, `i ∈ 0..4`.
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    twiddles: &[F128; 3],
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_2layer_row_from_geo_impl::<false>(
                src, src_quarter, src_r, dst, dst_quarter, dst_r, num_ntts, twiddles,
            )
        } else {
            butterfly_fused_2layer_row_from_geo_impl::<true>(
                src, src_quarter, src_r, dst, dst_quarter, dst_r, num_ntts, twiddles,
            )
        }
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from_geo`].
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_row_from_geo_impl<const DIET: bool>(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    twiddles: &[F128; 3],
) {
    use core::arch::x86_64::*;

    let [t_outer, t_inner_a, t_inner_b] = *twiddles;
    // SAFETY: caller guarantees target features, pointer geometry, and
    // non-aliasing src/dst.
    unsafe {
        let outer = tw_x4::<false, DIET>(t_outer);
        let inner_a = tw_x4::<false, DIET>(t_inner_a);
        let inner_b = tw_x4::<false, DIET>(t_inner_b);
        let src_row = |i: usize| src.add((i * src_quarter + src_r) * num_ntts);
        let dst_row = |i: usize| dst.add((i * dst_quarter + dst_r) * num_ntts);
        let lanes = num_ntts & !3;
        let mut lane = 0;
        while lane < lanes {
            let mut va = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let mut vb = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let mut vc = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let mut vd = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);

            let new_a = _mm512_xor_si512(va, mul_x4::<false, DIET>(outer, vc));
            vc = _mm512_xor_si512(vc, new_a);
            va = new_a;
            let new_b = _mm512_xor_si512(vb, mul_x4::<false, DIET>(outer, vd));
            vd = _mm512_xor_si512(vd, new_b);
            vb = new_b;

            let new_a = _mm512_xor_si512(va, mul_x4::<false, DIET>(inner_a, vb));
            vb = _mm512_xor_si512(vb, new_a);
            va = new_a;
            let new_c = _mm512_xor_si512(vc, mul_x4::<false, DIET>(inner_b, vd));
            vd = _mm512_xor_si512(vd, new_c);
            vc = new_c;

            _mm512_storeu_si512(dst_row(0).add(lane) as *mut __m512i, va);
            _mm512_storeu_si512(dst_row(1).add(lane) as *mut __m512i, vb);
            _mm512_storeu_si512(dst_row(2).add(lane) as *mut __m512i, vc);
            _mm512_storeu_si512(dst_row(3).add(lane) as *mut __m512i, vd);
            lane += 4;
        }
        while lane < num_ntts {
            let mut a = *src_row(0).add(lane);
            let mut b = *src_row(1).add(lane);
            let mut c = *src_row(2).add(lane);
            let mut d = *src_row(3).add(lane);

            let new_a = a + c * t_outer;
            c += new_a;
            a = new_a;
            let new_b = b + d * t_outer;
            d += new_b;
            b = new_b;

            let new_a = a + b * t_inner_a;
            b += new_a;
            a = new_a;
            let new_c = c + d * t_inner_b;
            d += new_c;
            c = new_c;

            *dst_row(0).add(lane) = a;
            *dst_row(1).add(lane) = b;
            *dst_row(2).add(lane) = c;
            *dst_row(3).add(lane) = d;
            lane += 1;
        }
    }
}

/// Sparse sibling: layer-1 and left layer-2 twiddles are zero, so `a` is
/// unchanged. Dense-with-zeros of [`butterfly_fused_2layer_row_from`].
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    right_twiddle: F128,
) {
    // SAFETY: forwarded caller contract; identical geometry on both sides.
    unsafe {
        butterfly_fused_2layer_row_from_sparse_geo(
            src,
            quarter,
            r,
            dst,
            quarter,
            r,
            num_ntts,
            right_twiddle,
        )
    }
}

/// [`butterfly_fused_2layer_row_from_sparse`] with independent source and
/// destination row geometry (see [`butterfly_fused_2layer_row_from_geo`]).
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_2layer_row_from_sparse_geo_impl::<false>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                right_twiddle,
            )
        } else {
            butterfly_fused_2layer_row_from_sparse_geo_impl::<true>(
                src,
                src_quarter,
                src_r,
                dst,
                dst_quarter,
                dst_r,
                num_ntts,
                right_twiddle,
            )
        }
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from_sparse_geo`].
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_2layer_row_from_sparse_geo_impl<const DIET: bool>(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
) {
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees target features, pointer geometry, and
    // non-aliasing src/dst.
    unsafe {
        let inner_b = tw_x4::<false, DIET>(right_twiddle);
        let src_row = |i: usize| src.add((i * src_quarter + src_r) * num_ntts);
        let dst_row = |i: usize| dst.add((i * dst_quarter + dst_r) * num_ntts);
        let lanes = num_ntts & !3;
        let mut lane = 0;
        while lane < lanes {
            let va = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let mut vb = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let mut vc = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let mut vd = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);

            // t_outer = 0, t_inner_a = 0: a stays a.
            vc = _mm512_xor_si512(vc, va);
            vd = _mm512_xor_si512(vd, vb);
            vb = _mm512_xor_si512(vb, va);

            let new_c = _mm512_xor_si512(vc, mul_x4::<false, DIET>(inner_b, vd));
            vd = _mm512_xor_si512(vd, new_c);
            vc = new_c;

            _mm512_storeu_si512(dst_row(0).add(lane) as *mut __m512i, va);
            _mm512_storeu_si512(dst_row(1).add(lane) as *mut __m512i, vb);
            _mm512_storeu_si512(dst_row(2).add(lane) as *mut __m512i, vc);
            _mm512_storeu_si512(dst_row(3).add(lane) as *mut __m512i, vd);
            lane += 4;
        }
        while lane < num_ntts {
            let a = *src_row(0).add(lane);
            let mut b = *src_row(1).add(lane);
            let mut c = *src_row(2).add(lane);
            let mut d = *src_row(3).add(lane);

            c += a;
            d += b;
            b += a;
            let new_c = c + d * right_twiddle;
            d += new_c;
            c = new_c;

            *dst_row(0).add(lane) = a;
            *dst_row(1).add(lane) = b;
            *dst_row(2).add(lane) = c;
            *dst_row(3).add(lane) = d;
            lane += 1;
        }
    }
}

/// # Safety
/// The caller guarantees target features, pointer validity, and disjoint rows.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_4layer_row(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    active_lanes: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        if mul_diet_disabled() {
            butterfly_fused_4layer_row_impl::<false>(
                ptr,
                sixteenth,
                num_ntts,
                active_lanes,
                r,
                twiddles,
            )
        } else {
            butterfly_fused_4layer_row_impl::<true>(
                ptr,
                sixteenth,
                num_ntts,
                active_lanes,
                r,
                twiddles,
            )
        }
    }
}

/// # Safety
/// Same contract as [`butterfly_fused_4layer_row`].
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_4layer_row_impl<const DIET: bool>(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    active_lanes: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    use core::arch::x86_64::*;

    // SAFETY: caller provides target features and pointer geometry.
    unsafe {
        // Broadcast (and, under DIET, x^64-companion) every twiddle ONCE per
        // row group: 15 setup CLMULs against 32 butterflies × ⌊lanes/4⌋ lane
        // steps of savings.
        let zero = _mm512_setzero_si512();
        let mut tw = [(zero, zero); 15];
        for (slot, value) in tw.iter_mut().zip(twiddles.iter()) {
            *slot = tw_x4::<false, DIET>(*value);
        }
        let row = |i: usize| ptr.add((i * sixteenth + r) * num_ntts);
        let lanes = active_lanes & !3;
        let mut lane = 0;
        while lane < lanes {
            let mut values = [zero; 16];
            for (i, value) in values.iter_mut().enumerate() {
                *value = _mm512_loadu_si512(row(i).add(lane) as *const __m512i);
            }

            macro_rules! butterfly {
                ($u:expr, $v:expr, $twiddle:expr) => {{
                    let new_u =
                        _mm512_xor_si512(values[$u], mul_x4::<false, DIET>($twiddle, values[$v]));
                    values[$v] = _mm512_xor_si512(values[$v], new_u);
                    values[$u] = new_u;
                }};
            }

            let outer = tw[0];
            for i in 0..8 {
                butterfly!(i, i + 8, outer);
            }
            for s in 0..2 {
                let twiddle = tw[1 + s];
                for i in 0..4 {
                    butterfly!(8 * s + i, 8 * s + i + 4, twiddle);
                }
            }
            for s in 0..4 {
                let twiddle = tw[3 + s];
                for i in 0..2 {
                    butterfly!(4 * s + i, 4 * s + i + 2, twiddle);
                }
            }
            for s in 0..8 {
                let twiddle = tw[7 + s];
                butterfly!(2 * s, 2 * s + 1, twiddle);
            }

            for (i, value) in values.iter().enumerate() {
                _mm512_storeu_si512(row(i).add(lane) as *mut __m512i, *value);
            }
            lane += 4;
        }

        while lane < active_lanes {
            let mut values = [F128::ZERO; 16];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *row(i).add(lane);
            }
            super::portable::butterfly_fused_4layer(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *row(i).add(lane) = *value;
            }
            lane += 1;
        }
    }
}

/// Process one fused-three-layer group of eight CONSECUTIVE rows.
///
/// `ptr` addresses row 0; row `i` starts at `i · num_ntts`. Lanes
/// `0..dense_lanes` run the full 12-butterfly network; lanes
/// `dense_lanes..num_ntts` run the zero-odd-row specialization (see
/// [`super::portable::butterfly_fused_3layer_zero_odd`]).
///
/// # Safety
/// The caller guarantees `avx512f` + `vpclmulqdq`, that the eight rows are
/// valid and disjoint from any concurrently processed group,
/// `dense_lanes <= num_ntts`, and that rows 1, 3, 5 and 7 are zero on lanes
/// `dense_lanes..num_ntts`.
///
/// Kept out of line (like [`butterfly_fused_4layer_row`]) so the deep-pass
/// closure that calls it once per block does not inline two full
/// monomorphizations of a twelve-butterfly network into its own frame.
#[inline(never)]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_fused_3layer_rows(
    ptr: *mut F128,
    num_ntts: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    // The fused-three sweep is the LAST group of the deep region, so its two
    // inner layers are the domain's two deepest — and every twiddle of the
    // two deepest standard-basis layers has a zero high limb (an exact
    // property of the fixed basis, pinned by
    // `deepest_standard_twiddle_layers_have_zero_high_limb`). Six of the
    // seven twiddles then take the 3-CLMUL `LOW` product instead of the
    // 5-CLMUL split form, which is eight of every twelve butterflies in the
    // group. The predicate is still checked on the values themselves, so a
    // non-final fused-three group simply keeps the general form.
    // The group's outer layer is the domain's THIRD-deepest, where exactly
    // half the blocks carry a low twiddle, so slot 0 is worth testing per
    // call as well as slots 1..7.
    let on = !low_twiddle_fused3_disabled();
    let low_inner = on && twiddles[1..].iter().all(|t| t.hi == 0);
    let low_outer = on && twiddles[0].hi == 0;
    // SAFETY: forwarded caller contract; the two predicates prove the LOW
    // precondition for the slots they gate by inspecting the values.
    unsafe {
        match (mul_diet_disabled(), low_outer, low_inner) {
            (true, false, false) => butterfly_fused_3layer_rows_impl::<false, false, false>(
                ptr, num_ntts, dense_lanes, twiddles,
            ),
            (true, false, true) => butterfly_fused_3layer_rows_impl::<false, false, true>(
                ptr, num_ntts, dense_lanes, twiddles,
            ),
            (true, true, false) => butterfly_fused_3layer_rows_impl::<false, true, false>(
                ptr, num_ntts, dense_lanes, twiddles,
            ),
            (true, true, true) => butterfly_fused_3layer_rows_impl::<false, true, true>(
                ptr, num_ntts, dense_lanes, twiddles,
            ),
            (false, false, false) => butterfly_fused_3layer_rows_impl::<true, false, false>(
                ptr, num_ntts, dense_lanes, twiddles,
            ),
            (false, false, true) => butterfly_fused_3layer_rows_impl::<true, false, true>(
                ptr, num_ntts, dense_lanes, twiddles,
            ),
            (false, true, false) => butterfly_fused_3layer_rows_impl::<true, true, false>(
                ptr, num_ntts, dense_lanes, twiddles,
            ),
            (false, true, true) => butterfly_fused_3layer_rows_impl::<true, true, true>(
                ptr, num_ntts, dense_lanes, twiddles,
            ),
        }
    }
}

/// `FLOCK_NO_NTT_LOW_TWIDDLE_FUSED3=1` restores the general twiddle product
/// for the fused-three sweep's two inner layers inside the same binary, so a
/// candidate/control pair differs only in the product form. Read once,
/// outside every lane loop.
#[inline]
fn low_twiddle_fused3_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_NTT_LOW_TWIDDLE_FUSED3").is_some())
}

/// # Safety
/// Same contract as [`butterfly_fused_3layer_rows`].
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn butterfly_fused_3layer_rows_impl<
    const DIET: bool,
    const LOW_OUTER: bool,
    const LOW_INNER: bool,
>(
    ptr: *mut F128,
    num_ntts: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    use core::arch::x86_64::*;

    // SAFETY: caller provides target features and pointer geometry.
    unsafe {
        // Seven broadcasts (plus their x^64 companions under DIET) hoisted
        // out of the lane loop: 8 rows stay live in registers across all
        // three layers, so a row is loaded once and stored once for 12
        // butterflies instead of the fused-two + single-layer pair's two
        // loads and two stores for the same 12.
        let zero = _mm512_setzero_si512();
        let mut tw = [(zero, zero); 7];
        tw[0] = tw_x4::<LOW_OUTER, DIET>(twiddles[0]);
        for (slot, value) in tw[1..].iter_mut().zip(twiddles[1..].iter()) {
            *slot = tw_x4::<LOW_INNER, DIET>(*value);
        }
        let row = |i: usize| ptr.add(i * num_ntts);
        let mut lane = 0;

        macro_rules! butterfly {
            ($values:ident, $u:expr, $v:expr, $twiddle:expr, $low:expr) => {{
                let new_u =
                    _mm512_xor_si512($values[$u], mul_x4::<$low, DIET>($twiddle, $values[$v]));
                $values[$v] = _mm512_xor_si512($values[$v], new_u);
                $values[$u] = new_u;
            }};
        }
        // Two lane chunks per iteration: 16 live row registers + 14 twiddle
        // registers still fit the 32 zmm file, and doubling the independent
        // butterflies per layer (4 -> 8) covers the CLMUL latency and doubles
        // the outstanding row misses.
        macro_rules! butterfly2 {
            ($values:ident, $u:expr, $v:expr, $twiddle:expr, $low:expr) => {{
                for c in 0..2 {
                    let new_u = _mm512_xor_si512(
                        $values[c][$u],
                        mul_x4::<$low, DIET>($twiddle, $values[c][$v]),
                    );
                    $values[c][$v] = _mm512_xor_si512($values[c][$v], new_u);
                    $values[c][$u] = new_u;
                }
            }};
        }

        while lane + 8 <= dense_lanes {
            let mut values = [[zero; 8]; 2];
            for (c, chunk) in values.iter_mut().enumerate() {
                for (i, value) in chunk.iter_mut().enumerate() {
                    *value = _mm512_loadu_si512(row(i).add(lane + 4 * c) as *const __m512i);
                }
            }

            let outer = tw[0];
            for i in 0..4 {
                butterfly2!(values, i, i + 4, outer, LOW_OUTER);
            }
            for s in 0..2 {
                let twiddle = tw[1 + s];
                for i in 0..2 {
                    butterfly2!(values, 4 * s + i, 4 * s + i + 2, twiddle, LOW_INNER);
                }
            }
            for s in 0..4 {
                butterfly2!(values, 2 * s, 2 * s + 1, tw[3 + s], LOW_INNER);
            }

            for (c, chunk) in values.iter().enumerate() {
                for (i, value) in chunk.iter().enumerate() {
                    _mm512_storeu_si512(row(i).add(lane + 4 * c) as *mut __m512i, *value);
                }
            }
            lane += 8;
        }
        while lane + 4 <= dense_lanes {
            let mut values = [zero; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = _mm512_loadu_si512(row(i).add(lane) as *const __m512i);
            }

            let outer = tw[0];
            for i in 0..4 {
                butterfly!(values, i, i + 4, outer, LOW_OUTER);
            }
            for s in 0..2 {
                let twiddle = tw[1 + s];
                for i in 0..2 {
                    butterfly!(values, 4 * s + i, 4 * s + i + 2, twiddle, LOW_INNER);
                }
            }
            for s in 0..4 {
                butterfly!(values, 2 * s, 2 * s + 1, tw[3 + s], LOW_INNER);
            }

            for (i, value) in values.iter().enumerate() {
                _mm512_storeu_si512(row(i).add(lane) as *mut __m512i, *value);
            }
            lane += 4;
        }
        while lane < dense_lanes {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *row(i).add(lane);
            }
            super::portable::butterfly_fused_3layer(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *row(i).add(lane) = *value;
            }
            lane += 1;
        }

        // Published zero tail: rows 1, 3, 5, 7 are zero here, so only the
        // four even rows are read and the deepest layer is a copy.
        while lane + 4 <= num_ntts {
            let mut values = [zero; 8];
            for i in 0..4 {
                values[2 * i] = _mm512_loadu_si512(row(2 * i).add(lane) as *const __m512i);
            }
            let outer = tw[0];
            butterfly!(values, 0, 4, outer, LOW_OUTER);
            butterfly!(values, 2, 6, outer, LOW_OUTER);
            butterfly!(values, 0, 2, tw[1], LOW_INNER);
            butterfly!(values, 4, 6, tw[2], LOW_INNER);
            for i in 0..4 {
                let v = values[2 * i];
                _mm512_storeu_si512(row(2 * i).add(lane) as *mut __m512i, v);
                _mm512_storeu_si512(row(2 * i + 1).add(lane) as *mut __m512i, v);
            }
            lane += 4;
        }
        while lane < num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *row(i).add(lane);
            }
            super::portable::butterfly_fused_3layer_zero_odd(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *row(i).add(lane) = *value;
            }
            lane += 1;
        }
    }
}

#[cfg(test)]
mod diet_tests {
    use super::*;
    use crate::field::F128;

    fn rng(seed: u64) -> impl FnMut() -> F128 {
        let mut st = seed | 1;
        move || {
            let mut n = || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st
            };
            F128 { lo: n(), hi: n() }
        }
    }

    /// The fused-three sweep's low-inner specialization must be
    /// bit-identical to the general product form and to the portable
    /// reference, for every lane path (8-wide, 4-wide, scalar tail) and for
    /// both `DIET` arms.
    #[test]
    fn fused3_low_inner_matches_general() {
        let mut next = rng(0x0BAD_C0DE);
        for (num_ntts, dense_lanes) in [(64usize, 64usize), (67, 67), (64, 48), (67, 50)] {
            let mut base = vec![F128::ZERO; 8 * num_ntts];
            for v in base.iter_mut() {
                *v = next();
            }
            // Beyond `dense_lanes` the published tail is zero on odd rows —
            // the kernel's zero-odd contract for those lanes.
            for lane in dense_lanes..num_ntts {
                for i in (1..8).step_by(2) {
                    base[i * num_ntts + lane] = F128::ZERO;
                }
            }
            // Slot 0 is the group's outer layer (a general twiddle); slots
            // 1..7 are the two deepest layers, whose production twiddles
            // always have a zero high limb.
            let mut twiddles = [F128::ZERO; 7];
            twiddles[0] = next();
            for t in twiddles[1..].iter_mut() {
                let v = next();
                *t = F128 { lo: v.lo, hi: 0 };
            }

            let mut want = base.clone();
            for lane in 0..num_ntts {
                let mut values = [F128::ZERO; 8];
                for (i, value) in values.iter_mut().enumerate() {
                    *value = want[i * num_ntts + lane];
                }
                if lane < dense_lanes {
                    crate::ntt::additive_ntt_f128::kernels::portable::butterfly_fused_3layer(&mut values, &twiddles);
                } else {
                    crate::ntt::additive_ntt_f128::kernels::portable::butterfly_fused_3layer_zero_odd(&mut values, &twiddles);
                }
                for (i, value) in values.iter().enumerate() {
                    want[i * num_ntts + lane] = *value;
                }
            }

            for (diet, low) in [(false, false), (false, true), (true, false), (true, true)] {
                let mut got = base.clone();
                // SAFETY: this module compiles only with avx512f+vpclmulqdq;
                // the buffer is 8 rows of `num_ntts` lanes, and every
                // `twiddles[1..]` entry has a zero high limb as the LOW arm
                // requires.
                unsafe {
                    match (diet, low) {
                        (false, false) => butterfly_fused_3layer_rows_impl::<false, false, false>(
                            got.as_mut_ptr(), num_ntts, dense_lanes, &twiddles,
                        ),
                        (false, true) => butterfly_fused_3layer_rows_impl::<false, false, true>(
                            got.as_mut_ptr(), num_ntts, dense_lanes, &twiddles,
                        ),
                        (true, false) => butterfly_fused_3layer_rows_impl::<true, false, false>(
                            got.as_mut_ptr(), num_ntts, dense_lanes, &twiddles,
                        ),
                        (true, true) => butterfly_fused_3layer_rows_impl::<true, false, true>(
                            got.as_mut_ptr(), num_ntts, dense_lanes, &twiddles,
                        ),
                    }
                }
                assert_eq!(
                    got, want,
                    "num_ntts={num_ntts} dense={dense_lanes} diet={diet} low={low}"
                );
            }
        }
    }

    /// `ghash_shift64_x4` really is multiplication by `x^64`.
    #[test]
    fn shift64_matches_scalar() {
        use crate::field::gf2_128::x86_64::ghash_shift64_x4;
        use core::arch::x86_64::*;

        let mut next = rng(0x51F7_7EED);
        let x64 = F128 { lo: 0, hi: 1 };
        // SAFETY: this module only compiles with avx512f+vpclmulqdq statically
        // enabled (see the cfg gate on `mod x86_64`).
        unsafe {
            for _ in 0..512 {
                let t = next();
                let tv = _mm512_broadcast_i32x4(_mm_set_epi64x(t.hi as i64, t.lo as i64));
                let mut got = [0u64; 8];
                _mm512_storeu_si512(got.as_mut_ptr() as *mut __m512i, ghash_shift64_x4(tv));
                let want = t * x64;
                for i in 0..4 {
                    assert_eq!((got[2 * i], got[2 * i + 1]), (want.lo, want.hi), "t={t:?}");
                }
            }
        }
    }

    /// The 5-CLMUL split product must equal the incumbent 6-CLMUL product in
    /// every lane — and the scalar reference product — for arbitrary twiddles
    /// and values, including the degenerate zero/one/low twiddles the sparse
    /// seed layers carry.
    #[test]
    fn split_product_matches_incumbent() {
        use crate::field::gf2_128::x86_64::{ghash_mul_x4, ghash_mul_x4_split, ghash_shift64_x4};
        use core::arch::x86_64::*;

        let mut next = rng(0xD1E7_5EED);
        // SAFETY: module is statically gated on avx512f+vpclmulqdq.
        unsafe {
            let mut check = |t: F128, v: [F128; 4]| {
                let tv = _mm512_broadcast_i32x4(_mm_set_epi64x(t.hi as i64, t.lo as i64));
                let vv = _mm512_set_epi64(
                    v[3].hi as i64,
                    v[3].lo as i64,
                    v[2].hi as i64,
                    v[2].lo as i64,
                    v[1].hi as i64,
                    v[1].lo as i64,
                    v[0].hi as i64,
                    v[0].lo as i64,
                );
                let want = ghash_mul_x4(tv, vv);
                let got = ghash_mul_x4_split(vv, tv, ghash_shift64_x4(tv));
                let mut w = [0u64; 8];
                let mut g = [0u64; 8];
                _mm512_storeu_si512(w.as_mut_ptr() as *mut __m512i, want);
                _mm512_storeu_si512(g.as_mut_ptr() as *mut __m512i, got);
                assert_eq!(w, g, "t={t:?} v={v:?}");
                for (i, vi) in v.iter().enumerate() {
                    let p = *vi * t;
                    assert_eq!((g[2 * i], g[2 * i + 1]), (p.lo, p.hi), "lane {i} t={t:?}");
                }
            };
            for _ in 0..512 {
                let t = next();
                check(t, [next(), next(), next(), next()]);
            }
            let low = F128 {
                lo: next().lo,
                hi: 0,
            };
            let high = F128 {
                lo: 0,
                hi: next().hi,
            };
            for t in [
                F128::ZERO,
                F128::ONE,
                F128 { lo: 0, hi: 1 },
                F128 {
                    lo: u64::MAX,
                    hi: u64::MAX,
                },
                low,
                high,
            ] {
                check(
                    t,
                    [
                        F128::ZERO,
                        F128::ONE,
                        next(),
                        F128 {
                            lo: u64::MAX,
                            hi: u64::MAX,
                        },
                    ],
                );
            }
        }
    }

    /// Every AVX-512 butterfly kernel must be bit-identical with the diet
    /// multiply ON and OFF, over random twiddles, random data, and lane counts
    /// that exercise both the vector body and its scalar tail.
    #[test]
    fn kernels_match_incumbent() {
        let mut next = rng(0xFEED_FACE);

        for len in [1usize, 3, 4, 5, 7, 8, 16, 64] {
            // --- row pair, general and LOW twiddle ------------------------
            let top: Vec<F128> = (0..len).map(|_| next()).collect();
            let bot: Vec<F128> = (0..len).map(|_| next()).collect();
            for twiddle in [
                next(),
                F128 {
                    lo: next().lo,
                    hi: 0,
                },
            ] {
                let run_pair = |diet: bool| {
                    let (mut t, mut b) = (top.clone(), bot.clone());
                    // SAFETY: statically gated features; twiddle.hi == 0 is
                    // never asserted here (LOW = false in both arms).
                    unsafe {
                        if diet {
                            butterfly_row_pair_impl::<false, true>(&mut t, &mut b, twiddle);
                        } else {
                            butterfly_row_pair_impl::<false, false>(&mut t, &mut b, twiddle);
                        }
                    }
                    (t, b)
                };
                assert_eq!(run_pair(true), run_pair(false), "row_pair len={len}");
            }
            // The LOW monomorphization must also agree with the diet general
            // one on a zero-high-limb twiddle.
            let low_tw = F128 {
                lo: next().lo,
                hi: 0,
            };
            let run_low = |low: bool| {
                let (mut t, mut b) = (top.clone(), bot.clone());
                // SAFETY: statically gated features; low_tw.hi == 0.
                unsafe {
                    if low {
                        butterfly_row_pair_impl::<true, true>(&mut t, &mut b, low_tw);
                    } else {
                        butterfly_row_pair_impl::<false, true>(&mut t, &mut b, low_tw);
                    }
                }
                (t, b)
            };
            assert_eq!(run_low(true), run_low(false), "row_pair low len={len}");

            // --- fused two-layer, in place --------------------------------
            let (t_outer, t_a, t_b) = (next(), next(), next());
            let rows: Vec<Vec<F128>> =
                (0..4).map(|_| (0..len).map(|_| next()).collect()).collect();
            let run_fused2 = |diet: bool| {
                let mut r = rows.clone();
                let (a, rest) = r.split_at_mut(1);
                let (b, rest) = rest.split_at_mut(1);
                let (c, d) = rest.split_at_mut(1);
                // SAFETY: statically gated features; equal row lengths.
                unsafe {
                    if diet {
                        butterfly_fused_2layer_impl::<false, false, true>(
                            &mut a[0], &mut b[0], &mut c[0], &mut d[0], t_outer, t_a, t_b,
                        );
                    } else {
                        butterfly_fused_2layer_impl::<false, false, false>(
                            &mut a[0], &mut b[0], &mut c[0], &mut d[0], t_outer, t_a, t_b,
                        );
                    }
                }
                r
            };
            assert_eq!(run_fused2(true), run_fused2(false), "fused2 len={len}");

            // --- fused two-layer, out of place (dense + sparse) ------------
            let src: Vec<F128> = (0..4 * len).map(|_| next()).collect();
            let tw3 = [next(), next(), next()];
            let run_from = |diet: bool| {
                let mut dst = vec![F128::ZERO; 4 * len];
                // SAFETY: 4 rows of `len` lanes each, src/dst disjoint.
                unsafe {
                    if diet {
                        butterfly_fused_2layer_row_from_geo_impl::<true>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            &tw3,
                        );
                    } else {
                        butterfly_fused_2layer_row_from_geo_impl::<false>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            &tw3,
                        );
                    }
                }
                dst
            };
            assert_eq!(run_from(true), run_from(false), "row_from len={len}");

            let right = next();
            let run_sparse = |diet: bool| {
                let mut dst = vec![F128::ZERO; 4 * len];
                // SAFETY: 4 rows of `len` lanes each, src/dst disjoint.
                unsafe {
                    if diet {
                        butterfly_fused_2layer_row_from_sparse_geo_impl::<true>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            right,
                        );
                    } else {
                        butterfly_fused_2layer_row_from_sparse_geo_impl::<false>(
                            src.as_ptr(),
                            1,
                            0,
                            dst.as_mut_ptr(),
                            1,
                            0,
                            len,
                            right,
                        );
                    }
                }
                dst
            };
            assert_eq!(run_sparse(true), run_sparse(false), "sparse len={len}");

            // --- fused four-layer -----------------------------------------
            let mut tw15 = [F128::ZERO; 15];
            for t in tw15.iter_mut() {
                *t = next();
            }
            let base: Vec<F128> = (0..16 * len).map(|_| next()).collect();
            let run_fused4 = |diet: bool| {
                let mut buf = base.clone();
                // SAFETY: 16 rows of `len` lanes, sixteenth = 1, r = 0.
                unsafe {
                    if diet {
                        butterfly_fused_4layer_row_impl::<true>(
                            buf.as_mut_ptr(),
                            1,
                            len,
                            len,
                            0,
                            &tw15,
                        );
                    } else {
                        butterfly_fused_4layer_row_impl::<false>(
                            buf.as_mut_ptr(),
                            1,
                            len,
                            len,
                            0,
                            &tw15,
                        );
                    }
                }
                buf
            };
            assert_eq!(run_fused4(true), run_fused4(false), "fused4 len={len}");
        }
    }
}
