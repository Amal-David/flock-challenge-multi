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

/// Runtime check: is the AVX2 instruction set available on this CPU?
///
/// Used by the BLAKE3 Merkle parent-node compress helper in `crate::merkle`
/// to gate the `#[target_feature(enable = "avx2")]` 2-way interleaved
/// compress kernel. Falls back to the crate-default `blake3::platform`
/// batched path on CPUs without AVX2. `is_x86_feature_detected!` is a
/// one-time `cpuid` cached by libstd, so the branch is effectively free
/// after the first call. No-op (`false`) on non-x86_64 targets.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn avx2_detected() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn avx2_detected() -> bool {
    false
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
}
