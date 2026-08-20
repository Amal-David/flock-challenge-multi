//! Padding-aware **streaming** ab_inner producer.
//!
//! [`super::precompute_round1_ab_inner_windows`] — the seam the ranked BLAKE3
//! witness generator fuses into its octa dump — transforms all `1 << N_MEDIUM`
//! b_med sub-windows of every outer window. The *non*-streaming producer
//! ([`super::precompute_round1_ab_inner_packed_padded`]) has always consulted
//! [`super::build_b_med_counts`] and skipped the sub-windows that fall entirely
//! in the witness padding. This module gives the streaming seam the same skip.
//!
//! Why it is sound: round 1's reader bounds its ab_inner loads by exactly the
//! same counts —
//! `process_one_x_hi_with_precomputed_ab_fold4` computes
//! `n_b_med = b_med_counts[x_outer & within_outer_mask]` and only ever reads
//! `for b_med in 0..n_b_med`. Bytes above that bound are never loaded, so what
//! the producer leaves there cannot reach the transcript. (The padded producer
//! leaves zeros; the streaming one used to leave the real transform; both yield
//! the same proof.)
//!
//! At the ranked shape (`k_log = 14`, `useful_bits = 15409`) the counts are
//! `[16, 15]`: one 64-byte sub-window of every odd outer window — 1 of the 32
//! per BLAKE3 block — is pure padding.
//!
//! **Store-alignment contract.** Dropping stores out of a non-temporal stream
//! can leave a partially-filled write-combining buffer, which the memory
//! controller must resolve with a read-modify-write — losing more than the
//! dropped stores saved. The skipped tail is therefore only *dropped* when
//! doing so cannot split a line (`nt == 2`, i.e. the destination is 64-byte
//! aligned so the tail is whole aligned cache lines; or `nt == 0`, a temporal
//! stream with no WC buffer to strand). On the unaligned NT stream (`nt == 1`,
//! 16-mod-64, four XMM stores per block) the tail is instead written as
//! non-temporal zeros, which keeps every line full and still deletes the
//! transform work.

use super::{
    ELL, F8, InvNttTableByteSingleGf8, N_MEDIUM, PaddingSpec, build_b_med_counts, kernels,
    shift_reduce_windows_into_blocks,
};

const N_FULL: usize = 1 << N_MEDIUM;

const OUTER_BYTES: usize = (1 << N_MEDIUM) * 64;

/// Hoisted per-outer-window b_med counts. Built once per prove; the streaming
/// producer is called once per BLAKE3 block, far too often to rebuild the
/// count vector (or re-derive it from `useful_bits`) per call.
#[derive(Debug, Clone)]
pub struct BMedPadPlan {
    within_outer_mask: usize,
    counts: Vec<u8>,
    dense: bool,
}

impl BMedPadPlan {
    /// Derive the plan from the witness [`PaddingSpec`] — the same call the
    /// non-streaming producer and round 1's reader both make, so the three
    /// agree by construction rather than by a hardcoded window index.
    pub fn new(padding: &PaddingSpec) -> Self {
        let (within_outer_mask, counts) = build_b_med_counts(padding);
        let dense = counts.iter().all(|&c| c as usize == 1 << N_MEDIUM);
        Self {
            within_outer_mask,
            counts,
            dense,
        }
    }

    /// True when the counts prove nothing is skippable at this shape. Callers
    /// should fall back to the incumbent dense producer rather than pay the
    /// per-window plan lookups for no deletion.
    pub fn is_dense(&self) -> bool {
        self.dense
    }

    /// Sub-windows of global outer window `outer` that carry any useful bit.
    #[inline]
    pub fn n_b_med(&self, outer: usize) -> usize {
        self.counts[outer & self.within_outer_mask] as usize
    }

    /// The within-hash outer index — also the static-B kernel's plan hint.
    #[inline]
    pub fn within(&self, outer: usize) -> usize {
        outer & self.within_outer_mask
    }
}

/// Non-temporal zero fill for the skipped tail, matching the `nt` class the
/// transform stores use so the write-combining stream stays contiguous.
#[inline]
fn zero_tail(tail: &mut [u8], nt: u8) {
    debug_assert_eq!(tail.len() % 64, 0);
    #[cfg(target_arch = "x86_64")]
    if nt == 1 {
        use core::arch::x86_64::*;
        // SAFETY: `nt == 1` was derived from `out.as_ptr() % 64 % 16 == 0`, and
        // every offset below is a multiple of 16 from that base, so each store
        // is 16-byte aligned. The slice owns `tail.len()` bytes.
        unsafe {
            let z = _mm_setzero_si128();
            let p = tail.as_mut_ptr();
            for off in (0..tail.len()).step_by(16) {
                _mm_stream_si128(p.add(off).cast::<__m128i>(), z);
            }
        }
        return;
    }
    tail.fill(0);
}

/// [`super::shift_reduce_windows_into_blocks`] with the sub-window count as a
/// **const** parameter.
///
/// This exists purely for codegen. The dense producer passes the literal
/// `1 << N_MEDIUM`, so LLVM folds the trip count, fully unrolls the pair loop
/// and drops the odd-tail branch. Feeding the same function a count loaded
/// from the plan turns that into a real loop with a runtime bound — measured
/// on the ranked shape, losing the unroll costs more than the 3.1% of windows
/// the skip deletes (witness +0.4 ms instead of −1.6 ms). Dispatching on the
/// count and monomorphizing per value gives every window the incumbent's
/// codegen while still skipping the padding.
///
/// Bit-identical to [`super::shift_reduce_windows_into_blocks`]: same kernels,
/// same b_med order, same arguments.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn windows_into_blocks_const<const N: usize>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    out_outer: &mut [u8],
    a_col: &mut [F8],
    b_col: &mut [F8],
    bstatic: kernels::BstaticHint,
    nt: u8,
) {
    let mut b_med = 0;
    while b_med + 1 < N {
        let (blk0, rest) = out_outer[b_med * 64..].split_at_mut(64);
        let out0: &mut [u8; 64] = blk0.try_into().expect("one transformed b_med block");
        let out1: &mut [u8; 64] = (&mut rest[..64])
            .try_into()
            .expect("one transformed b_med block");
        kernels::shift_reduce_inner_ab_x2(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out0,
            out1,
            a_col,
            b_col,
            bstatic,
            nt,
        );
        b_med += 2;
    }
    if b_med < N {
        let dst: &mut [u8; 64] = (&mut out_outer[b_med * 64..(b_med + 1) * 64])
            .try_into()
            .expect("one transformed b_med block");
        kernels::shift_reduce_inner_ab(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            dst,
            a_col,
            b_col,
            bstatic,
            nt,
        );
    }
}

/// Dispatch one outer window to a monomorphized driver. `N_FULL` is the
/// overwhelmingly common count (every outer window but the one straddling the
/// end of the useful bits); `N_FULL - 1` is what a `useful_bits` that lands
/// inside the last sub-window produces, and is half of all windows at the
/// ranked `k_log = 14`. Anything else is rare enough to take the incumbent
/// runtime-count driver, which is correct at every count.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn windows_into_blocks_dispatch(
    n_b_med: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    out_outer: &mut [u8],
    a_col: &mut [F8],
    b_col: &mut [F8],
    bstatic: kernels::BstaticHint,
    nt: u8,
) {
    match n_b_med {
        N_FULL => windows_into_blocks_const::<N_FULL>(
            a_packed, b_packed, inv_table, chunk_byte_base, out_outer, a_col, b_col, bstatic, nt,
        ),
        n if n + 1 == N_FULL => windows_into_blocks_const::<{ N_FULL - 1 }>(
            a_packed, b_packed, inv_table, chunk_byte_base, out_outer, a_col, b_col, bstatic, nt,
        ),
        0 => {}
        n => shift_reduce_windows_into_blocks(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            n,
            out_outer,
            a_col,
            b_col,
            bstatic,
            nt,
        ),
    }
}

/// [`super::precompute_round1_ab_inner_windows`] with the all-padding b_med
/// tail of each outer window skipped per `plan`.
///
/// `outer_base` is the global outer-window index of `out[0]`, so the plan
/// lookup and the static-B parity hint match what the whole-buffer producer
/// would have used for these same bytes. Byte-identical to the dense producer
/// on every sub-window round 1 can read.
pub fn precompute_round1_ab_inner_windows_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    out: &mut [u8],
    inv_table: &InvNttTableByteSingleGf8,
    nt_out: bool,
    plan: &BMedPadPlan,
    outer_base: usize,
) {
    assert_eq!(a_packed.len(), b_packed.len());
    assert_eq!(a_packed.len(), out.len());
    assert_eq!(a_packed.len() % OUTER_BYTES, 0);
    assert_eq!(inv_table.k, super::K_SKIP);

    // One classification for the whole call: every window base is a multiple
    // of 64 from `out`, so they all share its residue.
    let nt: u8 = if nt_out {
        match out.as_ptr() as usize % 64 {
            0 => 2,
            r if r % 16 == 0 => 1,
            _ => 0,
        }
    } else {
        0
    };
    // See the store-alignment contract in the module docs.
    let drop_tail = nt != 1;

    let mut a_col = [F8::ZERO; ELL];
    let mut b_col = [F8::ZERO; ELL];
    let bstatic_ctx = kernels::prepare_bstatic(inv_table);
    for outer in 0..a_packed.len() / OUTER_BYTES {
        let base = outer * OUTER_BYTES;
        let g_outer = outer_base + outer;
        let n_b_med = plan.n_b_med(g_outer);
        windows_into_blocks_dispatch(
            n_b_med,
            a_packed,
            b_packed,
            inv_table,
            base,
            &mut out[base..base + OUTER_BYTES],
            &mut a_col,
            &mut b_col,
            bstatic_ctx.map(|p| (plan.within(g_outer), p)),
            nt,
        );
        if n_b_med < N_FULL && !drop_tail {
            zero_tail(&mut out[base + n_b_med * 64..base + OUTER_BYTES], nt);
        }
    }
}
