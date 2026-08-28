//! Selection of the hash function backing a protocol component.
//!
//! Two components are independently configurable, and they are genuinely
//! independent — a proof can use BLAKE3 Merkle commitments with a SHA-256
//! Fiat-Shamir transcript, or any other combination:
//!
//! - the Merkle commitments, via [`crate::pcs::commit::PcsParams::merkle_hash`]
//!   (see [`crate::merkle`]);
//! - the Fiat-Shamir transcript and its proof-of-work grinding, via
//!   [`crate::challenger::FsChallenger::with_hash`].
//!
//! Both default to SHA-256, so configs and call sites that predate the options
//! keep their behaviour.

use serde::{Deserialize, Serialize};

/// Re-exported BLAKE3 platform type for direct root-finalize paths.
pub(crate) use blake3::platform::Platform as Blake3Platform;

// ---------------------------------------------------------------------------
// BLAKE3 scalar compression — BMI2-gated dispatch.
//
// `blake3`'s portable scalar compression inlines a `rotate_right` per G
// quarter-round (eight per round, seven rounds = 56 rotates per compress).
// On x86_64 without BMI2 the compiler emits the 3-operand `ror` (1 cycle,
// 1 uop). With BMI2 the `RORX` instruction is a 3-operand rotate that
// writes a fresh register and runs at 1 cycle / 1 uop, but it eliminates
// the implicit "destination == source" false dependency the decoder tracks
// for the older rotate form — relevant when several rotates target the
// same `state[d]` register across one compress and a tight OoO core
// would otherwise serialize them. We feature-detect once (cached in a
// [`std::sync::LazyLock`]) and dispatch to a BMI2-flavoured scalar
// compress when present, falling back to the portable rotation
// otherwise. Behaviour is bit-identical to the upstream portable path on
// every block — this is a constant-folding-driven rewrite, not a new
// algorithm — and the test suite pins the output to
// `blake3::compress_in_place` byte-for-byte.
// ---------------------------------------------------------------------------

/// BLAKE3's IV (key words for unkeyed hashing). Fixed by the spec; matches
/// the constant in `blake3::platform::Platform::compress_in_place`.
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// BLAKE3 per-round message-word permutation, fixed by the spec.
const BLAKE3_MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// Zero-copy re-interpretation of a 64-byte BLAKE3 input block as 16
/// little-endian `u32` message words.
///
/// # Safety
///
/// The caller must ensure the input is exactly 64 bytes long with a
/// well-aligned pointer. The cast itself is a pointer-to-pointer transmute
/// of a `[u8; 64]` reference into a `[u32; 16]` reference; the source and
/// destination are the same 64 bytes, so no other invariants are involved.
/// BLAKE3 reads the message words as little-endian `u32` (per
/// `words_from_le_bytes_64` in `blake3::platform`), and the same machine
/// reads them back the same way when `to_le_bytes()` is applied at the
/// call site, so the alias is bit-identical to a copy through
/// `u32::from_le_bytes`.
#[inline(always)]
pub(crate) unsafe fn block_to_words(block: &[u8; 64]) -> &[u32; 16] {
    // SAFETY: same backing storage, same length (64 B == 16 × 4 B), same
    // alignment guarantees (the slice's element alignment is at most 4 on
    // a machine that has 4-byte words — BLAKE3's spec assumes 4-byte LE
    // words, which the caller has by construction of `block`).
    unsafe { &*(block.as_ptr().cast::<[u32; 16]>()) }
}

/// BLAKE3 quarter-round, portable form (matching `blake3::portable::g`).
#[inline(always)]
fn g_portable(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(x);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(y);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

/// One BLAKE3 round, portable form.
#[inline(always)]
fn round_portable(state: &mut [u32; 16], m: &[u32; 16], r: usize) {
    let s = &BLAKE3_MSG_SCHEDULE[r];
    g_portable(state, 0, 4, 8, 12, m[s[0]], m[s[1]]);
    g_portable(state, 1, 5, 9, 13, m[s[2]], m[s[3]]);
    g_portable(state, 2, 6, 10, 14, m[s[4]], m[s[5]]);
    g_portable(state, 3, 7, 11, 15, m[s[6]], m[s[7]]);
    g_portable(state, 0, 5, 10, 15, m[s[8]], m[s[9]]);
    g_portable(state, 1, 6, 11, 12, m[s[10]], m[s[11]]);
    g_portable(state, 2, 7, 8, 13, m[s[12]], m[s[13]]);
    g_portable(state, 3, 4, 9, 14, m[s[14]], m[s[15]]);
}

/// Portable scalar BLAKE3 in-place compression: 7 rounds + XOR-fold.
///
/// Output is bit-identical to `blake3::platform::compress_in_place` (and to
/// the upstream `portable::compress_in_place`); the BMI2 variant differs
/// only in how each rotate is lowered.
#[inline]
pub(crate) fn compress_in_place_portable(
    cv: &mut [u32; 8],
    block: &[u8; 64],
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    // SAFETY: caller-provided `block` is exactly 64 bytes; the cast reads
    // it as 16 LE u32s, which is the spec's view of a BLAKE3 message.
    let m = unsafe { block_to_words(block) };
    let mut state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len as u32,
        flags as u32,
    ];
    round_portable(&mut state, m, 0);
    round_portable(&mut state, m, 1);
    round_portable(&mut state, m, 2);
    round_portable(&mut state, m, 3);
    round_portable(&mut state, m, 4);
    round_portable(&mut state, m, 5);
    round_portable(&mut state, m, 6);
    for i in 0..8 {
        cv[i] = state[i] ^ state[i + 8];
    }
}

/// `_mm_rorx_epi32`-style rotate using a BMI2 RORX, lowered via inline
/// asm because the stable intrinsic requires nightly SIMD types we don't
/// otherwise need.
///
/// # Safety
///
/// Caller must ensure the BMI2 target feature is enabled in the calling
/// function (we gate the only call site on `target_feature(enable =
/// "bmi2")`, so this holds transitively).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
unsafe fn rorx32<const N: u32>(x: u32) -> u32 {
    let out: u32;
    // SAFETY: BMI2 is enabled by the enclosing target_feature; the asm
    // clobbers nothing, reads one register, writes one.
    unsafe {
        core::arch::asm!(
            "rorx {e}, {x}, {n}",
            e = out(reg) out,
            x = in(reg) x,
            n = const N,
            options(nomem, nostack, pure),
        );
    }
    out
}

/// BLAKE3 quarter-round using RORX for the three rotations. Same bit
/// output as [`g_portable`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
unsafe fn g_bmi2(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    // SAFETY: enclosing target_feature(enable = "bmi2") makes rorx32 safe.
    unsafe {
        state[a] = state[a].wrapping_add(state[b]).wrapping_add(x);
        state[d] = rorx32::<16>(state[d] ^ state[a]);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = rorx32::<12>(state[b] ^ state[c]);
        state[a] = state[a].wrapping_add(state[b]).wrapping_add(y);
        state[d] = rorx32::<8>(state[d] ^ state[a]);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = rorx32::<7>(state[b] ^ state[c]);
    }
}

/// BMI2 scalar BLAKE3 in-place compression. Same bit output as
/// [`compress_in_place_portable`]; the only difference is that the
/// quarter-round rotates lower to `RORX` (3-operand, no false dependency
/// on the destination register).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
pub(crate) unsafe fn compress_in_place_bmi2(
    cv: &mut [u32; 8],
    block: &[u8; 64],
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    // SAFETY: caller-provided `block` is exactly 64 bytes; the cast reads
    // it as 16 LE u32s, which is the spec's view of a BLAKE3 message.
    let m = unsafe { block_to_words(block) };
    let mut state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len as u32,
        flags as u32,
    ];
    for r in 0..7 {
        let s = &BLAKE3_MSG_SCHEDULE[r];
        // SAFETY: target_feature(enable = "bmi2") gates this function.
        unsafe {
            g_bmi2(&mut state, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            g_bmi2(&mut state, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            g_bmi2(&mut state, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            g_bmi2(&mut state, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            g_bmi2(&mut state, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            g_bmi2(&mut state, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g_bmi2(&mut state, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            g_bmi2(&mut state, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
    }
    for i in 0..8 {
        cv[i] = state[i] ^ state[i + 8];
    }
}

/// Cached BMI2 feature flag for the scalar BLAKE3 dispatch. Lazy-init once
/// per process; the [`std::sync::LazyLock`] read is a relaxed atomic load
/// after first call.
#[cfg(target_arch = "x86_64")]
fn bmi2_enabled() -> bool {
    use std::sync::LazyLock;
    static BMI2: LazyLock<bool> = LazyLock::new(|| std::is_x86_feature_detected!("bmi2"));
    *BMI2
}

/// Scalar BLAKE3 in-place compression with cached BMI2 dispatch.
///
/// On x86_64 with BMI2, routes to [`compress_in_place_bmi2`] (RORX
/// quarter-rounds, no false rotate-dependency); otherwise falls back to
/// the portable rotation. Bit-identical to
/// `blake3::platform::compress_in_place` on every input.
#[inline]
pub(crate) fn compress_inner(
    cv: &mut [u32; 8],
    block: &[u8; 64],
    block_len: u8,
    counter: u64,
    flags: u8,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if bmi2_enabled() {
            // SAFETY: `bmi2_enabled()` returned true just above, and the
            // caller upholds the 64-byte block invariant.
            unsafe {
                return compress_in_place_bmi2(cv, block, block_len, counter, flags);
            }
        }
    }
    compress_in_place_portable(cv, block, block_len, counter, flags);
}

/// BLAKE3 root-finalize helper from PR #1664, bypassing `OutputReader`.
#[allow(dead_code)]
pub(crate) fn finalize_root_bytes(cv: [u32; 8], block_len: u8, starting_flags: u8) -> [u8; 32] {
    let mut cv = cv;
    let platform = Blake3Platform::detect();
    Blake3Platform::compress_in_place(&platform, &mut cv, &[0u8; 64], block_len, 0, starting_flags);
    blake3::platform::le_bytes_from_words_32(&cv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant, for tests that sweep both.
    pub(crate) const ALL: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

    #[test]
    fn parses_and_round_trips() {
        for kind in ALL {
            assert_eq!(HashKind::parse(kind.as_str()).unwrap(), kind);
            assert_eq!(kind.to_string(), kind.as_str());
        }
        assert_eq!(HashKind::parse("BLAKE3").unwrap(), HashKind::Blake3);
        assert_eq!(HashKind::parse("sha-256").unwrap(), HashKind::Sha256);
        assert_eq!(HashKind::parse("  blake3 ").unwrap(), HashKind::Blake3);
        assert_eq!(HashKind::default(), HashKind::Sha256);
        // An unrecognized hash must be an error, never a silent SHA-256.
        assert!(HashKind::parse("keccak").is_err());
        assert!(HashKind::parse("").is_err());
    }

    #[test]
    fn serde_uses_config_spellings() {
        for kind in ALL {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{}\"", kind.as_str()));
            assert_eq!(
                serde_json::from_str::<HashKind>(&json).unwrap(),
                kind,
                "{kind}"
            );
        }
    }

    /// The BMI2-gated scalar compress must produce bit-identical output to
    /// `blake3::compress_in_place` for every block, on every
    /// counter/flags combination, and the zero-copy `&[u8;64] -> &[u32;16]`
    /// cast must agree with the explicit `u32::from_le_bytes` form.
    #[test]
    fn compress_inner_matches_upstream_on_every_block() {
        use blake3::platform::Platform;
        let platform = Platform::detect();
        // Sweep a varied block (random-ish pattern), a block of all zeros
        // (the canonical counter block the PoW pre-image uses), and a
        // block of all-ones (max-magnitude words).
        let blocks: [[u8; 64]; 3] = [
            std::array::from_fn(|i| (i as u32).wrapping_mul(0x9E37_79B9).to_le_bytes())
                .into_iter()
                .flatten()
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
            [0u8; 64],
            [0xFFu8; 64],
        ];
        for block in &blocks {
            for counter in [0u64, 1, 0xFFFF_FFFF_FFFF_FFFF, 0xDEAD_BEEF_CAFE_F00D] {
                for flags in [0u8, 1, 2, 1 | 2, 1 | 2 | 4, 1 | 2 | 8] {
                    for block_len in [0u8, 1, 32, 63, 64] {
                        // Upstream (oracle).
                        let mut cv_ref: [u32; 8] = [
                            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
                        ];
                        Platform::compress_in_place(
                            &platform,
                            &mut cv_ref,
                            block,
                            block_len,
                            counter,
                            flags,
                        );
                        // Ours (dispatched through `compress_inner`).
                        let mut cv_ours: [u32; 8] = [
                            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
                        ];
                        compress_inner(&mut cv_ours, block, block_len, counter, flags);
                        assert_eq!(
                            cv_ours, cv_ref,
                            "block_len={block_len} counter=0x{counter:016x} flags=0x{flags:02x}"
                        );
                    }
                }
            }
        }
    }

    /// The portable fallback must also match the upstream.
    #[test]
    fn compress_in_place_portable_matches_upstream() {
        use blake3::platform::Platform;
        let platform = Platform::detect();
        let block = [0u8; 64];
        let mut cv_ref: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
        Platform::compress_in_place(&platform, &mut cv_ref, &block, 64, 0, 0);
        let mut cv_ours: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
            0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
        compress_in_place_portable(&mut cv_ours, &block, 64, 0, 0);
        assert_eq!(cv_ours, cv_ref);
    }

    /// The zero-copy block-to-words cast must agree with the explicit
    /// `u32::from_le_bytes` per-word form. (Bit pattern in a [u8; 64] vs
    /// its [u32; 16] view of the same memory, on a little-endian host.)
    #[test]
    fn block_to_words_matches_from_le_bytes() {
        let block: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        let words_ref: [u32; 16] = std::array::from_fn(|i| {
            u32::from_le_bytes(block[i * 4..i * 4 + 4].try_into().unwrap())
        });
        let words_cast: &[u32; 16] = unsafe { block_to_words(&block) };
        assert_eq!(words_cast, &words_ref);
    }
}
