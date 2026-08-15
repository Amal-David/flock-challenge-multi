use crate::field::F128;

#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and equal slice lengths.
    unsafe {
        let twiddle_lanes =
            _mm512_broadcast_i32x4(_mm_set_epi64x(twiddle.hi as i64, twiddle.lo as i64));
        let lanes = top.len() & !3;
        let mut i = 0;
        while i < lanes {
            let top_lanes = _mm512_loadu_si512(top.as_ptr().add(i) as *const __m512i);
            let bot_lanes = _mm512_loadu_si512(bot.as_ptr().add(i) as *const __m512i);
            let new_top = _mm512_xor_si512(top_lanes, ghash_mul_x4(twiddle_lanes, bot_lanes));
            let new_bot = _mm512_xor_si512(bot_lanes, new_top);
            _mm512_storeu_si512(top.as_mut_ptr().add(i) as *mut __m512i, new_top);
            _mm512_storeu_si512(bot.as_mut_ptr().add(i) as *mut __m512i, new_bot);
            i += 4;
        }
        super::portable::butterfly_row_pair(&mut top[i..], &mut bot[i..], twiddle);
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
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and equal slice lengths.
    unsafe {
        let broadcast =
            |value: F128| _mm512_broadcast_i32x4(_mm_set_epi64x(value.hi as i64, value.lo as i64));
        let outer = broadcast(t_outer);
        let inner_a = broadcast(t_inner_a);
        let inner_b = broadcast(t_inner_b);
        let lanes = a.len() & !3;
        let mut i = 0;
        while i < lanes {
            let mut va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let mut vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            let mut vc = _mm512_loadu_si512(c.as_ptr().add(i) as *const __m512i);
            let mut vd = _mm512_loadu_si512(d.as_ptr().add(i) as *const __m512i);

            let new_a = _mm512_xor_si512(va, ghash_mul_x4(outer, vc));
            vc = _mm512_xor_si512(vc, new_a);
            va = new_a;
            let new_b = _mm512_xor_si512(vb, ghash_mul_x4(outer, vd));
            vd = _mm512_xor_si512(vd, new_b);
            vb = new_b;

            let new_a = _mm512_xor_si512(va, ghash_mul_x4(inner_a, vb));
            vb = _mm512_xor_si512(vb, new_a);
            va = new_a;
            let new_c = _mm512_xor_si512(vc, ghash_mul_x4(inner_b, vd));
            vd = _mm512_xor_si512(vd, new_c);
            vc = new_c;

            _mm512_storeu_si512(a.as_mut_ptr().add(i) as *mut __m512i, va);
            _mm512_storeu_si512(b.as_mut_ptr().add(i) as *mut __m512i, vb);
            _mm512_storeu_si512(c.as_mut_ptr().add(i) as *mut __m512i, vc);
            _mm512_storeu_si512(d.as_mut_ptr().add(i) as *mut __m512i, vd);
            i += 4;
        }
        super::portable::butterfly_fused_2layer(
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
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    let [t_outer, t_inner_a, t_inner_b] = *twiddles;
    // SAFETY: caller guarantees target features, pointer geometry, and
    // non-aliasing src/dst.
    unsafe {
        let broadcast =
            |value: F128| _mm512_broadcast_i32x4(_mm_set_epi64x(value.hi as i64, value.lo as i64));
        let outer = broadcast(t_outer);
        let inner_a = broadcast(t_inner_a);
        let inner_b = broadcast(t_inner_b);
        let src_row = |i: usize| src.add((i * quarter + r) * num_ntts);
        let dst_row = |i: usize| dst.add((i * quarter + r) * num_ntts);
        let lanes = num_ntts & !3;
        let mut lane = 0;
        while lane < lanes {
            let mut va = _mm512_loadu_si512(src_row(0).add(lane) as *const __m512i);
            let mut vb = _mm512_loadu_si512(src_row(1).add(lane) as *const __m512i);
            let mut vc = _mm512_loadu_si512(src_row(2).add(lane) as *const __m512i);
            let mut vd = _mm512_loadu_si512(src_row(3).add(lane) as *const __m512i);

            let new_a = _mm512_xor_si512(va, ghash_mul_x4(outer, vc));
            vc = _mm512_xor_si512(vc, new_a);
            va = new_a;
            let new_b = _mm512_xor_si512(vb, ghash_mul_x4(outer, vd));
            vd = _mm512_xor_si512(vd, new_b);
            vb = new_b;

            let new_a = _mm512_xor_si512(va, ghash_mul_x4(inner_a, vb));
            vb = _mm512_xor_si512(vb, new_a);
            va = new_a;
            let new_c = _mm512_xor_si512(vc, ghash_mul_x4(inner_b, vd));
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
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees target features, pointer geometry, and
    // non-aliasing src/dst.
    unsafe {
        let inner_b = _mm512_broadcast_i32x4(_mm_set_epi64x(
            right_twiddle.hi as i64,
            right_twiddle.lo as i64,
        ));
        let src_row = |i: usize| src.add((i * quarter + r) * num_ntts);
        let dst_row = |i: usize| dst.add((i * quarter + r) * num_ntts);
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

            let new_c = _mm512_xor_si512(vc, ghash_mul_x4(inner_b, vd));
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
    r: usize,
    twiddles: &[F128; 15],
) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    // SAFETY: caller provides target features and pointer geometry.
    unsafe {
        let broadcast =
            |value: F128| _mm512_broadcast_i32x4(_mm_set_epi64x(value.hi as i64, value.lo as i64));
        let row = |i: usize| ptr.add((i * sixteenth + r) * num_ntts);
        let lanes = num_ntts & !3;
        let mut lane = 0;
        while lane < lanes {
            let mut values = [_mm512_setzero_si512(); 16];
            for (i, value) in values.iter_mut().enumerate() {
                *value = _mm512_loadu_si512(row(i).add(lane) as *const __m512i);
            }

            macro_rules! butterfly {
                ($u:expr, $v:expr, $twiddle:expr) => {{
                    let new_u = _mm512_xor_si512(values[$u], ghash_mul_x4($twiddle, values[$v]));
                    values[$v] = _mm512_xor_si512(values[$v], new_u);
                    values[$u] = new_u;
                }};
            }

            let outer = broadcast(twiddles[0]);
            for i in 0..8 {
                butterfly!(i, i + 8, outer);
            }
            for s in 0..2 {
                let twiddle = broadcast(twiddles[1 + s]);
                for i in 0..4 {
                    butterfly!(8 * s + i, 8 * s + i + 4, twiddle);
                }
            }
            for s in 0..4 {
                let twiddle = broadcast(twiddles[3 + s]);
                for i in 0..2 {
                    butterfly!(4 * s + i, 4 * s + i + 2, twiddle);
                }
            }
            for s in 0..8 {
                let twiddle = broadcast(twiddles[7 + s]);
                butterfly!(2 * s, 2 * s + 1, twiddle);
            }

            for (i, value) in values.iter().enumerate() {
                _mm512_storeu_si512(row(i).add(lane) as *mut __m512i, *value);
            }
            lane += 4;
        }

        while lane < num_ntts {
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
