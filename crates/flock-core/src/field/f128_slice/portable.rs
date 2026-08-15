use crate::field::F128;

#[inline]
pub(super) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    // Char-2: even*(1+r) + odd*r = even + r*(even+odd). One mul per pair.
    for (t, value) in dst.iter_mut().enumerate() {
        let s = 2 * (base + t);
        let even = src[s];
        *value = even + r * (even + src[s + 1]);
    }
}
