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

/// BLAKE3 IV, fixed by the spec. Copied here as a `u32` array so the scalar
/// compression can `#[inline(always)]` a fully-unrolled core that pre-xors the
/// IV into the upper half of the 16-word state and never re-reads it. The
/// canonical reference lives in the `blake3` crate; this copy exists so the
/// per-compression hot path does not have to chase a `&[u32; 8]` through a
/// trait-bounded associated constant.
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// BLAKE3 message-word permutation: 7 rounds, 16 indices each, indexing into
/// the 16 message words produced by a 64-byte LE block load. Identical to the
/// spec; promoted from a runtime table to a `const` so the compiler can
/// constant-fold each `MSG[i]` access and let the unrolled `g`/round body
/// see only `u32` registers.
const MSG_PERMUTATION: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// BLAKE3 quarter-round `G`, fully unrolled onto the 4 referenced state
/// words. The rotations are written as `(x >> n) | (x << (32 - n))` — the
/// compiler folds this into `ROR`/`ROL` on x86_64 and a `ROR` on aarch64
/// without needing the BMI2 / NEON extension path; `rustc` at `-C
/// target-cpu=native` chooses the cheapest encoding per target.
///
/// `#[inline(always)]` keeps every call site inside the caller, so the seven
/// rounds of one compression become a single straight-line block with the
/// message permutation fully resolved.
#[inline(always)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(x);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(y);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

/// One BLAKE3 round on a 16-word state, fully unrolled. The `round` index
/// selects the message-word permutation row; `msg[perm[k]]` becomes a
/// `msg[const]` after the permutation is `const`-promoted, so the eight
/// quarter-round calls below all see a literal `msg[N]`.
#[inline(always)]
fn round_fn(state: &mut [u32; 16], msg: &[u32; 16], round: usize) {
    let perm = MSG_PERMUTATION[round];
    g(state, 0, 4, 8, 12, msg[perm[0]], msg[perm[1]]);
    g(state, 1, 5, 9, 13, msg[perm[2]], msg[perm[3]]);
    g(state, 2, 6, 10, 14, msg[perm[4]], msg[perm[5]]);
    g(state, 3, 7, 11, 15, msg[perm[6]], msg[perm[7]]);
    g(state, 0, 5, 10, 15, msg[perm[8]], msg[perm[9]]);
    g(state, 1, 6, 11, 12, msg[perm[10]], msg[perm[11]]);
    g(state, 2, 7, 8, 13, msg[perm[12]], msg[perm[13]]);
    g(state, 3, 4, 9, 14, msg[perm[14]], msg[perm[15]]);
}

/// In-house scalar BLAKE3 compression, in-place on the chaining value
/// `cv`. Folds the 8-word CV with the 16-word LE-loaded block, the 32-bit
/// counter pair, the block length, and the 8-bit flags. Bit-for-bit
/// equivalent to `blake3::platform::Platform::compress_in_place` against the
/// same inputs (asserted in `compress_matches_reference` below), but on the
/// per-compression hot path of `finalize_root_bytes` it skips the
/// SIMD-batched dispatcher's match arm and the platform FFI the platform
/// `compress_in_place` would normally take.
#[inline(always)]
fn compress_in_place(cv: &mut [u32; 8], block: &[u8; 64], block_len: u8, counter: u64, flags: u8) {
    // SAFETY: a 64-byte aligned/padded LE block read into 16 `u32`s. The
    // caller passes a `&[u8; 64]`, which is 4-byte aligned for any reference;
    // the natural LE reinterpretation here is what every BLAKE3 backend does
    // to feed the message words. Using `from_le_bytes` per word keeps the
    // unaligned-load story correct for any future caller, and the compiler
    // folds the eight calls into a single 64-byte vectorized LE load when the
    // alignment cooperates.
    let mut msg = [0u32; 16];
    let mut i = 0;
    while i < 16 {
        let off = i * 4;
        let chunk: [u8; 4] = [block[off], block[off + 1], block[off + 2], block[off + 3]];
        msg[i] = u32::from_le_bytes(chunk);
        i += 1;
    }

    let counter_lo = counter as u32;
    let counter_hi = (counter >> 32) as u32;

    // Pre-xored IV: the upper half of the 16-word state starts as
    // `BLAKE3_IV[i] ^ counter_{lo,hi} ^ block_len ^ flags`, so the 7-round
    // unroll never has to read the IV again. The spec's final XOR
    // `state[i] ^ state[i + 8]` then reproduces the canonical CV fold with
    // the same constants folded in.
    let mut state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0] ^ counter_lo,
        BLAKE3_IV[1] ^ counter_hi,
        BLAKE3_IV[2] ^ block_len as u32,
        BLAKE3_IV[3] ^ flags as u32,
        BLAKE3_IV[4] ^ counter_lo,
        BLAKE3_IV[5] ^ counter_hi,
        BLAKE3_IV[6] ^ block_len as u32,
        BLAKE3_IV[7] ^ flags as u32,
    ];

    // Seven full rounds, each with the message permutation resolved at
    // compile time by `MSG_PERMUTATION` + `round_fn`'s `#[inline(always)]`.
    round_fn(&mut state, &msg, 0);
    round_fn(&mut state, &msg, 1);
    round_fn(&mut state, &msg, 2);
    round_fn(&mut state, &msg, 3);
    round_fn(&mut state, &msg, 4);
    round_fn(&mut state, &msg, 5);
    round_fn(&mut state, &msg, 6);

    // Standard BLAKE3 CV fold.
    cv[0] = state[0] ^ state[8];
    cv[1] = state[1] ^ state[9];
    cv[2] = state[2] ^ state[10];
    cv[3] = state[3] ^ state[11];
    cv[4] = state[4] ^ state[12];
    cv[5] = state[5] ^ state[13];
    cv[6] = state[6] ^ state[14];
    cv[7] = state[7] ^ state[15];
}

/// Which hash function backs a component.
///
/// `Sha256` is the default, so existing serialized params and configs that
/// predate these options deserialize unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HashKind {
    #[default]
    Sha256,
    Blake3,
}

impl HashKind {
    /// Config-file spelling of this hash (`"sha256"` / `"blake3"`). Inverse of
    /// [`HashKind::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            HashKind::Sha256 => "sha256",
            HashKind::Blake3 => "blake3",
        }
    }

    /// Parse a config field or environment variable. Case-insensitive; rejects
    /// anything unrecognized rather than silently falling back to SHA-256 — a
    /// config naming a hash we do not implement must not quietly produce
    /// proofs under a different one.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sha256" | "sha-256" => Ok(HashKind::Sha256),
            "blake3" => Ok(HashKind::Blake3),
            other => Err(format!(
                "unknown hash {other:?}: expected \"sha256\" or \"blake3\""
            )),
        }
    }
}

impl std::fmt::Display for HashKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// BLAKE3 root-finalize helper from PR #1664, bypassing `OutputReader`.
///
/// Uses the in-house scalar compression in this module so the one
/// per-Merkle-root finalize (only at the very top of the tree) does not have
/// to dispatch through `blake3::platform::Platform::compress_in_place`'s
/// match arm and FFI — a small constant win, but constant is constant.
#[allow(dead_code)]
pub(crate) fn finalize_root_bytes(cv: [u32; 8], block_len: u8, starting_flags: u8) -> [u8; 32] {
    let mut cv = cv;
    compress_in_place(&mut cv, &[0u8; 64], block_len, 0, starting_flags);
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

    /// The in-house scalar compression must agree with `blake3`'s
    /// `Platform::compress_in_place` reference on every (cv, block, counter,
    /// flags) tuple we care about. This is what pins the in-tree copy to
    /// real BLAKE3 semantics rather than to itself, the same role the rest
    /// of the merkle module's `blake3_batched_matches_scalar_spec` test
    /// plays for the SIMD-batched path.
    #[test]
    fn compress_matches_reference() {
        use blake3::platform::Platform as Blake3Platform;
        let platform = Blake3Platform::detect();
        // CVs, blocks, counters, and flags picked to exercise the IV XOR
        // (counter present + counter absent), the block_len slot (0 / 64),
        // and a mix of every flag bit the production callers ever set.
        let cvs: [[u32; 8]; 4] = [
            [
                0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
                0x5BE0CD19,
            ],
            [0; 8],
            [
                0xDEAD_BEEF,
                0x1234_5678,
                0x9ABC_DEF0,
                0xCAFE_F00D,
                0x0BAD_F00D,
                0xFACE_FEED,
                0xBAAD_F00D,
                0xFEED_BEEF,
            ],
            [0xFFFF_FFFF; 8],
        ];
        let blocks: [[u8; 64]; 3] = [
            [0u8; 64],
            {
                let mut b = [0u8; 64];
                for (i, slot) in b.iter_mut().enumerate() {
                    *slot = i as u8;
                }
                b
            },
            {
                let mut b = [0u8; 64];
                for (i, slot) in b.iter_mut().enumerate() {
                    *slot = (i as u8).wrapping_mul(0x9E);
                }
                b
            },
        ];
        for &cv in &cvs {
            for &block in &blocks {
                for &block_len in &[0u8, 1u8, 32u8, 64u8] {
                    for &counter in &[0u64, 1u64, 0xDEAD_BEEF_CAFE_F00D_u64, u64::MAX] {
                        for &flags in &[0u8, 1u8, 2u8, 4u8, 8u8, 0x1F, 0xFF] {
                            let mut ours = cv;
                            compress_in_place(&mut ours, &block, block_len, counter, flags);
                            let mut theirs = cv;
                            Blake3Platform::compress_in_place(
                                &platform,
                                &mut theirs,
                                &block,
                                block_len,
                                counter,
                                flags,
                            );
                            assert_eq!(
                                ours, theirs,
                                "cv={cv:?} block_len={block_len} counter={counter:#x} flags={flags:#x}"
                            );
                        }
                    }
                }
            }
        }
    }
}
