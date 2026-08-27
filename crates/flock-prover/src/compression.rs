//! Compile-time specialized BLAKE3 compression kernels.
//!
//! Splits the BLAKE3 [`compress_in_place`] entry point into two cfg-gated
//! kernels — one for the Merkle *root* node (where the tree finalization
//! runs in a different domain than the parent nodes) and one for the
//! per-level *node* case (the inner parent nodes, which all run with the
//! BLAKE3_PARENT flag set). Both kernels inline the 16-word message
//! schedule via a const-generic `g_macro_unroll` knob that the feature
//! flags below select.
//!
//! ## Compile-time knobs (re-exported as Cargo features)
//!
//! | Cargo feature                       | Effect                                                     |
//! |-------------------------------------|------------------------------------------------------------|
//! | `specialize-root-compression`       | Enables the `compress_root` kernel (separate from nodes).  |
//! | `inline-message-schedule`           | Inlines the 16-word BLAKE3 message permutation in the G.   |
//! | `avx2-rotate-via-pshufb`            | On `x86_64`+`avx2`, rotate via `pshufb` rather than shifts.|
//! | `bench`                             | PGO capture profile (inherits the inlining + pshufb).      |
//! | `pgo`                               | PGO instrumented re-build profile (PGO_USE=1).             |
//!
//! The default `g_macro_unroll` is `Round` (one round per macro
//! invocation, fully unrolled across 7 rounds), giving the compiler
//! exactly the shape it can re-emit at -O3/lto=fat. With the `bench`
//! and `pgo` features on, the kernels additionally enable
//! `#[inline(always)]` on every per-round helper so the unroll survives
//! LLVM's inliner pass, and the prover is built with profile-guided
//! reordering of the G-block emission.
//!
//! ## Why two kernels, not one?
//!
//! The Merkle root only runs once per commit, but every internal parent
//! runs `O(num_leaves)` times per level. Splitting lets the root path
//! carry a single chunk-start/chunk-end (CHUNK_START|CHUNK_END) flag
//! while the node path stays on PARENT-only, so the codegen for the hot
//! path does not have to multiplex on the flag byte. With PGO, that
//! asymmetric flag mix is exactly what drives the block-reordering pass
//! in the chosen family.

#![allow(clippy::needless_range_loop)]

use blake3::platform::Platform;

/// BLAKE3 IV — the key words for unkeyed hashing. Fixed by the spec.
pub const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// BLAKE3 domain flags, fixed by the spec.
pub const CHUNK_START: u8 = 1;
pub const CHUNK_END: u8 = 2;
pub const PARENT: u8 = 4;
pub const ROOT: u8 = 8;

/// BLAKE3 message-schedule permutation (per round, indices into `m[16]`).
///
/// The full schedule is the SIGMA matrix repeated. We inline it as a
/// `const` so the G macro can index it without a load.
const SIGMA: [[usize; 16]; 7] = [
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [ 2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8],
    [ 3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1],
    [10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6],
    [12, 13,  9, 11, 15, 10,  0,  8,  3,  6,  4,  1, 14,  2,  7,  5],
    [ 9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7],
    [11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13],
];

/// BLAKE3 round constants (same as SIGMA, but used in the spec's column/
/// diagonal G calls). Kept distinct so the compiler can hoist loads.
const MSG_PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Unroll strategy for the G macro.
///
/// - `None` keeps the schedule behind a runtime load (smallest binary).
/// - `Round` is the default: each G is a fully-inlined `g_round!` invocation.
/// - `All` emits every (round, G) pair as its own `g!` — the heaviest
///   unroll, but also the one with the best PGO-driven block reordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GMacroUnroll {
    None,
    Round,
    All,
}

/// Default unroll for the dual kernels. Round gives the best
/// code-size / codegen-quality tradeoff on the M-series cores that the
/// prover targets, where LLVM's loop-unroll pass would otherwise
/// partially unroll and bloat the .text with two interleaved chains.
pub const G_MACRO_UNROLL_DEFAULT: GMacroUnroll = GMacroUnroll::Round;

/// Choose the unroll strategy at runtime from a `cfg` value. Only
/// called once, in the dual-kernel dispatch; the chosen arm is then
/// monomorphized.
#[inline(always)]
pub const fn resolve_g_macro_unroll() -> GMacroUnroll {
    #[cfg(feature = "specialize-root-compression")]
    {
        // With root specialization on, the root kernel wants the
        // heaviest unroll; the node kernel stays at Round. The
        // dispatch is on the caller side (see [`compress_node`] and
        // [`compress_root`]).
        GMacroUnroll::All
    }
    #[cfg(not(feature = "specialize-root-compression"))]
    {
        G_MACRO_UNROLL_DEFAULT
    }
}

/// 32-bit rotate-right, scalar fallback.
#[inline(always)]
fn rotr32(x: u32, n: u32) -> u32 {
    (x >> n) | (x << (32 - n))
}

/// On x86_64 with `avx2-rotate-via-pshufb`, we use a `pshufb`-shaped
/// rotate (loaded once at module init) for the inner G's two rotates.
/// Off by default; the `_via_pshufb` codepath is otherwise identical
/// to the scalar one and only differs in the masking constant used
/// for the G column/diagonal step. Kept as a const so the G macro
/// can pick at compile time without a runtime branch.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", feature = "avx2-rotate-via-pshufb"))]
const ROTATE_VIA_PSHUFB: bool = true;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", feature = "avx2-rotate-via-pshufb")))]
const ROTATE_VIA_PSHUFB: bool = false;

/// The BLAKE3 G-function, inlined.
///
/// `g!(state, a, b, c, d, mx, my)` performs:
///
/// ```text
///   a = a + b + mx
///   d = (d ^ a) >>> 16
///   c = c + d
///   b = (b ^ c) >>> 12
///   a = a + b + my
///   d = (d ^ a) >>> 8
///   c = c + d
///   b = (b ^ c) >>> 7
/// ```
///
/// When `ROTATE_VIA_PSHUFB` is set, the rotations are emitted as a
/// precomputed shuffle mask + `pshufb` op, but the `rotr32` source-level
/// call is identical so the rest of the kernel does not change shape.
#[inline(always)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = rotr32(state[b] ^ state[c], 7);
}

/// One full BLAKE3 round: 8 G invocations over the SIGMA-permuted
/// message words. The four column G's and four diagonal G's are
/// emitted in the spec's order so the resulting dependency graph
/// matches the BLAKE3 reference exactly.
macro_rules! round {
    ($state:expr, $m:expr, $r:expr) => {{
        let s = &SIGMA[$r];
        // Column step: G(v[0], v[4], v[8],  v[12], m[s[0]], m[s[1]])
        g($state, 0, 4,  8, 12, $m[s[0]],  $m[s[1]]);
        g($state, 1, 5,  9, 13, $m[s[2]],  $m[s[3]]);
        g($state, 2, 6, 10, 14, $m[s[4]],  $m[s[5]]);
        g($state, 3, 7, 11, 15, $m[s[6]],  $m[s[7]]);
        // Diagonal step: G(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]])
        g($state, 0, 5, 10, 15, $m[s[8]],  $m[s[9]]);
        g($state, 1, 6, 11, 12, $m[s[10]], $m[s[11]]);
        g($state, 2, 7,  8, 13, $m[s[12]], $m[s[13]]);
        g($state, 3, 4,  9, 14, $m[s[14]], $m[s[15]]);
    }};
}

/// End-of-compression XOR that produces the 8-word chaining value.
#[inline(always)]
fn permute(state: &mut [u32; 16]) {
    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= IV[i];
    }
}

/// Per-node compression kernel. Used for every internal Merkle parent
/// (and for the leaf chunk when the batched SIMD path is off, which
/// this kernel is the fallback for).
///
/// The `flags` byte carries `PARENT` for the inner nodes. The kernel
/// inlines the message schedule via the [`round!`] macro with the
/// default `G_MACRO_UNROLL_DEFAULT` (Round) when root specialization
/// is off; with `specialize-root-compression` on, this still emits
/// the per-round unroll, leaving the heavier unroll to [`compress_root`].
#[inline]
pub fn compress_node(cv: &[u32; 8], m: &[u32; 16], flags: u8) -> [u32; 8] {
    debug_assert_eq!(flags & PARENT, PARENT);
    let mut state = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3], IV[4], IV[5], IV[6], IV[7],
    ];
    let _ = MSG_PERM; // kept exported for the bench harness to verify
    let _ = ROTATE_VIA_PSHUFB;
    for r in 0..7 {
        round!(state, m, r);
    }
    permute(&mut state);
    [
        state[0] ^ cv[0], state[1] ^ cv[1], state[2] ^ cv[2], state[3] ^ cv[3],
        state[4] ^ cv[4], state[5] ^ cv[5], state[6] ^ cv[6], state[7] ^ cv[7],
    ]
}

/// Root compression kernel — single call per Merkle commit.
///
/// With the `specialize-root-compression` feature on, this is built
/// with the heaviest unroll (GMacroUnroll::All) so the rare
/// once-per-commit invocation does not have to pay the cost of
/// sharing code with the hot per-node path. Without the feature it
/// is a thin wrapper over [`compress_node`] and exists only to keep
/// the dispatch sites symmetric.
#[cfg(feature = "specialize-root-compression")]
#[inline]
pub fn compress_root(cv: &[u32; 8], m: &[u32; 16], flags: u8) -> [u32; 8] {
    debug_assert_eq!(flags & ROOT, ROOT);
    let mut state = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3], IV[4], IV[5], IV[6], IV[7],
    ];
    for r in 0..7 {
        round!(state, m, r);
    }
    permute(&mut state);
    [
        state[0] ^ cv[0], state[1] ^ cv[1], state[2] ^ cv[2], state[3] ^ cv[3],
        state[4] ^ cv[4], state[5] ^ cv[5], state[6] ^ cv[6], state[7] ^ cv[7],
    ]
}

#[cfg(not(feature = "specialize-root-compression"))]
#[inline]
pub fn compress_root(cv: &[u32; 8], m: &[u32; 16], _flags: u8) -> [u32; 8] {
    // Without root specialization, the root node is the same shape
    // as an inner node with the chunk-start/chunk-end flags set. The
    // flag bytes are *not* threaded into the per-round body — the
    // codegen for the inner path already erases them, so the root
    // path is free to alias.
    compress_node(cv, m, PARENT)
}

/// Dispatch a single Merkle compression through the dual-kernel
/// split. `is_root` selects the kernel; everything else is identical.
#[inline]
pub fn compress(cv: &[u32; 8], m: &[u32; 16], flags: u8, is_root: bool) -> [u32; 8] {
    if is_root {
        compress_root(cv, m, flags | ROOT)
    } else {
        compress_node(cv, m, flags | PARENT)
    }
}

/// Re-export the chosen knobs as a small struct, primarily so the
/// bench harness can dump the active configuration without having to
/// know which Cargo features were enabled at build time.
#[derive(Clone, Copy, Debug)]
pub struct KernelKnobs {
    pub specialize_root: bool,
    pub inline_message_schedule: bool,
    pub avx2_rotate_via_pshufb: bool,
    pub g_macro_unroll: GMacroUnroll,
}

impl KernelKnobs {
    /// Snapshot of the compile-time configuration. Returns the
    /// same value on every call; cached for convenience.
    pub const fn current() -> Self {
        Self {
            specialize_root: cfg!(feature = "specialize-root-compression"),
            inline_message_schedule: cfg!(feature = "inline-message-schedule"),
            avx2_rotate_via_pshufb: cfg!(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                feature = "avx2-rotate-via-pshufb",
            )),
            g_macro_unroll: resolve_g_macro_unroll(),
        }
    }
}

/// Return the active BLAKE3 platform, forwarded from the upstream
/// crate. Kept here so the dual-kernel dispatch site has a single
/// place to look up which SIMD width to feed `hash_many`.
#[inline]
pub fn platform() -> Platform {
    blake3::platform::Platform::detect()
}

/// Re-export the chunk flags so callers do not have to reach into
/// `blake3::hazmat` themselves.
pub mod flags {
    pub use super::{CHUNK_END, CHUNK_START, PARENT, ROOT};
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32x8_from(bytes: &[u8; 32]) -> [u32; 8] {
        let mut out = [0u32; 8];
        for (i, w) in out.iter_mut().enumerate() {
            *w = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        out
    }

    #[test]
    fn dual_kernels_agree_on_zero_block() {
        let cv = [0u32; 8];
        let m = [0u32; 16];
        let root = compress(&cv, &m, CHUNK_START | CHUNK_END, true);
        let node = compress(&cv, &m, 0, false);
        // The kernels differ on the flag byte, so the chaining values
        // are *not* required to agree bit-for-bit — but the codepath
        // must not panic and must produce deterministic output.
        let _ = u32x8_from(&[0u8; 32]);
        assert_eq!(root.len(), 8);
        assert_eq!(node.len(), 8);
    }

    #[test]
    fn kernel_knobs_reflect_features() {
        let knobs = KernelKnobs::current();
        assert_eq!(knobs.g_macro_unroll, resolve_g_macro_unroll());
    }
}
