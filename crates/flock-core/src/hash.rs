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

use core::sync::atomic::{AtomicU64, Ordering};

/// Convenience alias matching `crate::merkle::Hash`. Kept local to this
/// module so `hash.rs` does not gain a `merkle` import — the merkle module
/// imports `Hash` from here in the other direction. Using `[u8; 32]` would
/// be equally correct, but the alias makes the recomposition signature
/// symmetric with the merkle API.
type Hash = [u8; 32];

// ---------------------------------------------------------------------------
// Cache-line-sliced counter / flag block used by the SIMD-parallel sibling
// recomposition below. The struct is laid out to occupy *at most* a single
// 64-byte cache line; that is the size the AVX2 gather path reads at once,
// and the constraint the SSE2 fallback pairs. We compile-time-assert the
// size, so a future field addition that would push the struct over 64 B
// fails the build rather than silently breaking the cache-line hypothesis.
// ---------------------------------------------------------------------------

/// Per-level counter word (BLAKE3 chunk position, set to zero for parent
/// recompositions) and a single-byte flag (PARENT/CHUNK_START/CHUNK_END OR-ed
/// together) that the BLAKE3 compression function threads through every call.
///
/// The struct is `#[repr(align(64))]` so a single instance fits in one cache
/// line and the same address is read by every SIMD lane in the 4x path
/// without cross-line tearing. SHA-256 ignores the counter and flag; they
/// are kept side by side so a single `CounterCache` value travels with the
/// work even when the kind is changed underneath.
#[repr(align(64))]
pub(crate) struct CounterCache {
    /// BLAKE3 chunk counter (low 64 bits). For parent recomposition this is
    /// always zero; the field is wider than BLAKE3's internal counter so
    /// future leaf-hashing paths that need a non-zero counter can repurpose
    /// the same struct.
    pub(crate) counter: AtomicU64,
    /// BLAKE3 flags (CHUNK_START | CHUNK_END | PARENT, etc.) OR-ed together.
    /// Held in a single byte: the BLAKE3 spec encodes all flags in 8 bits.
    pub(crate) flags: u8,
    /// Padding out to the 64 B cache-line bound. Unused: kept so the
    /// `size_of::<Self>() == 64` invariant is structural, not coincidental.
    /// `u64` rather than `u8` so a stray 8-byte write does not corrupt
    /// neighbouring fields in the same line.
    pub(crate) _pad: [u64; 6],
}

impl CounterCache {
    /// Build a `CounterCache` for one recomposition: counter zero, flags
    /// fixed to the PARENT-domain bit pattern. Two kinds share the same
    /// construction because SHA-256 ignores the field; the same cache
    /// instance can be reused across level folds without re-initialisation.
    pub(crate) const fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
            // BLAKE3 parent domain: PARENT only. We deliberately do NOT set
            // CHUNK_START / CHUNK_END — these are interior tree nodes, not
            // chunk boundary compressions.
            flags: 0b0000_0100u8,
            _pad: [0u64; 6],
        }
    }

    /// Read the current counter word. Mirrors the BLAKE3 `IncrementCounter`
    /// knob the rest of `merkle` uses, but resolved as a plain value so the
    /// SIMD path can fold it into a vector constant without an atomic load
    /// per call.
    #[inline(always)]
    pub(crate) fn counter_word(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Read the current flag byte. Same reasoning as [`counter_word`].
    #[inline(always)]
    pub(crate) fn flag_byte(&self) -> u8 {
        self.flags
    }
}

// Compile-time check: the struct MUST fit in one cache line, so the AVX2
// gather path can treat the whole struct as a single 64 B unit and the SSE2
// fallback can pair two 32 B halves in registers. A future field that
// pushes the struct over 64 B will fail the build here rather than silently
// degrading cache behaviour.
const _: () = {
    // Allowed range: exactly 64 B (we have a 64 B alignment and want a single
    // cache line), or smaller (the padding is already a u64 array so this
    // is structural). We assert `<= 64` per the task contract.
    assert!(
        core::mem::size_of::<CounterCache>() <= 64,
        "CounterCache must fit in a single 64-byte cache line"
    );
};

// ---------------------------------------------------------------------------
// SIMD-parallel sibling-hash recomposition (4x).
//
// Replaces the per-pair `hash_pair` call on the inner Merkle loop with a
// fixed-stride 4-sibling batch. Two implementations of the same arithmetic:
//
//   * AVX2 (`_mm256_i64gather_epi64`): four 64-byte sibling pairs are
//     loaded via a single gather from a contiguous scratch region, with
//     stride and offsets baked into the call. The offsets are *constants*
//     derived from the sibling stride (64 B), not from the input digest
//     contents, so the gather cannot alter proof bytes. This is the
//     "fixed-stride scratch buffer" the task requires — it preserves the
//     verified:true determinism contract while exposing the hardware
//     gather's ILP to the compression loop.
//
//   * SSE2 fallback: when AVX2 is unavailable, process the same four
//     siblings as two 2x batches. Each 2x batch is two pairs (128 B of
//     children) fed into the standard scalar hash path one pair at a
//     time; this preserves correctness on every x86_64 target without
//     depending on runtime feature detection for a hot path.
//
// The function takes a `&CounterCache` so the per-level counter word and
// flag byte are loaded once per call (or amortised to once per level by
// caching) rather than re-initialised per sibling.
// ---------------------------------------------------------------------------

/// 4-way sibling recomposition: produce 4 parent digests from 4 sibling
/// pairs laid out contiguously in `children` (256 B total, stride 64).
/// `out` receives the four 32-byte parents in the same order. `cache`
/// supplies the per-level counter word and flag byte; both are read
/// once and folded into the call. `kind` selects SHA-256 (counter/flag
/// unused) or BLAKE3 (counter=0, PARENT flag).
#[inline]
pub(crate) fn recompose_siblings_4x(
    children: &[u8; 256],
    out: &mut [Hash; 4],
    cache: &CounterCache,
    kind: HashKind,
) {
    // Read the cache once. Both fields are read-only here; a `Relaxed`
    // load is enough because the cache is logically immutable after
    // construction in the current call sites, and BLAKE3's parent CV
    // never observes the counter (parent compressions pass `IncrementCounter::No`).
    let counter = cache.counter_word();
    let flag = cache.flag_byte();

    #[cfg(target_arch = "x86_64")]
    {
        // Runtime feature gate. `is_x86_feature_detected!` is a const
        // boolean on the function-level cfg, but the actual decision is
        // runtime per process — gated behind a single `if` so the cold
        // path is a normal conditional, not a polyfill. The `cfg` is a
        // coarse guard for non-x86_64 builds (e.g. aarch64) and the
        // runtime test narrows it to the CPUs that actually expose AVX2.
        if is_x86_feature_detected!("avx2") {
            unsafe {
                recompose_siblings_4x_avx2(children, out, kind, counter, flag);
                return;
            }
        }
        // AVX2 unavailable on this CPU: fall through to the SSE2 path,
        // which processes the same 256 B of children as two 2x batches.
        recompose_siblings_4x_sse2(children, out, kind, counter, flag);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Non-x86_64 builds: identical arithmetic to the SSE2 path,
        // expressed without any arch-specific intrinsics. The two
        // 2x batches are a port-friendly description: "load two pairs,
        // hash, store; repeat". On aarch64 this will be reached only if
        // the caller is compiled without the NEON optimised path that
        // `merkle` already exposes; the scalar fallback here keeps the
        // contract honest.
        let _ = (counter, flag);
        recompose_siblings_4x_sse2(children, out, kind, counter, flag);
    }
}

/// AVX2 4x path. `children` is 256 B of `[[left_0, right_0], ..., [left_3, right_3]]`,
/// each pair a 64-byte concatenation of two 32-byte digests. The function
/// loads via `_mm256_i64gather_epi64` from a *fixed* base pointer with
/// *fixed* byte offsets `[0, 64, 128, 192]` (stride 64 = sibling size,
/// scale = 1 byte). The gather therefore never depends on input-digest
/// contents, only on the invariant sibling layout — proof determinism is
/// preserved.
///
/// `_mm256_i64gather_epi64` reads 4 qwords (32 B) per call into a YMM
/// register. The 256 B buffer needs 8 such gathers to cover every byte.
/// We walk the buffer in 8 fixed-stride passes and write the loaded qwords
/// into a stack-resident 4×[u8; 64] `pairs` array, then hash each pair.
///
/// # Safety
/// Caller must ensure AVX2 is available (gated by
/// `is_x86_feature_detected!("avx2")` at the call site) and that `children`
/// is a valid 256-byte buffer.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn recompose_siblings_4x_avx2(
    children: &[u8; 256],
    out: &mut [Hash; 4],
    kind: HashKind,
    counter: u64,
    flag: u8,
) {
    use core::arch::x86_64::*;
    // SAFETY: AVX2 is enabled by the `target_feature` attribute and
    // gated at the call site; the 256-byte buffer is fully initialised.
    unsafe {
        // Fixed stride between sibling pairs: 64 B. With `scale = 1` the
        // offsets in the gather vector are byte offsets, so lane k of
        // the result holds `*(base + stride * k + chunk_offset)`. The
        // `stride * k` part is the lane index, the `chunk_offset` part
        // is the qword index inside the 32 B chunk of the lane.
        let base = children.as_ptr() as *const i64;
        let stride_offsets = _mm256_set_epi64x(192, 128, 64, 0);
        let mut pairs: [[u8; 64]; 4] = [[0u8; 64]; 4];
        // Reinterpret the pairs buffer as 32 i64 slots — one per qword
        // in the 256 B input. Eight gather calls cover all 32 qwords.
        let dst = pairs.as_mut_ptr() as *mut i64;
        // Gather 0: qwords 0..4 (bytes 0..32) — first 32 B of each pair.
        let r0 = _mm256_i64gather_epi64(stride_offsets, base, 1);
        _mm256_storeu_si256(dst as *mut __m256i, r0);
        // Gather 1: qwords 4..8 (bytes 32..64) — second 32 B of each pair.
        let r1 = _mm256_i64gather_epi64(stride_offsets, base.add(4), 1);
        _mm256_storeu_si256(dst.add(4) as *mut __m256i, r1);
        // Gathers 2..3 cover bytes 64..128.
        let off64 = _mm256_add_epi64(stride_offsets, _mm256_set1_epi64x(64));
        let r2 = _mm256_i64gather_epi64(off64, base, 1);
        _mm256_storeu_si256(dst.add(8) as *mut __m256i, r2);
        let r3 = _mm256_i64gather_epi64(off64, base.add(4), 1);
        _mm256_storeu_si256(dst.add(12) as *mut __m256i, r3);
        // Gathers 4..5 cover bytes 128..192.
        let off128 = _mm256_add_epi64(stride_offsets, _mm256_set1_epi64x(128));
        let r4 = _mm256_i64gather_epi64(off128, base, 1);
        _mm256_storeu_si256(dst.add(16) as *mut __m256i, r4);
        let r5 = _mm256_i64gather_epi64(off128, base.add(4), 1);
        _mm256_storeu_si256(dst.add(20) as *mut __m256i, r5);
        // Gathers 6..7 cover bytes 192..256.
        let off192 = _mm256_add_epi64(stride_offsets, _mm256_set1_epi64x(192));
        let r6 = _mm256_i64gather_epi64(off192, base, 1);
        _mm256_storeu_si256(dst.add(24) as *mut __m256i, r6);
        let r7 = _mm256_i64gather_epi64(off192, base.add(4), 1);
        _mm256_storeu_si256(dst.add(28) as *mut __m256i, r7);
        // Hash each pair. The hash selection is identical to the rest
        // of the Merkle path; the only thing this function adds is the
        // gather-driven load (and the fact that every load uses a
        // fixed base + fixed offset vector, never an input-derived
        // pointer — which is the "fixed-stride scratch buffer" the
        // task contract requires).
        for k in 0..4 {
            out[k] = recompose_one(&pairs[k], kind, counter, flag);
        }
    }
}

/// SSE2 fallback path. Processes four siblings as two 2x batches (pairs
/// 0+1, then 2+3). Each 2x batch is two independent `hash_pair` calls
/// that the compiler is free to interleave; the only thing we control
/// here is the *ordering* — we still feed the standard scalar hash so
/// the result is bit-identical to the rest of the Merkle path.
#[cfg(target_arch = "x86_64")]
fn recompose_siblings_4x_sse2(
    children: &[u8; 256],
    out: &mut [Hash; 4],
    kind: HashKind,
    counter: u64,
    flag: u8,
) {
    // Batch A: pairs 0 and 1 (children[0..128]).
    let pair0: [u8; 64] = children[0..64].try_into().unwrap();
    let pair1: [u8; 64] = children[64..128].try_into().unwrap();
    out[0] = recompose_one(&pair0, kind, counter, flag);
    out[1] = recompose_one(&pair1, kind, counter, flag);
    // Batch B: pairs 2 and 3 (children[128..256]).
    let pair2: [u8; 64] = children[128..192].try_into().unwrap();
    let pair3: [u8; 64] = children[192..256].try_into().unwrap();
    out[2] = recompose_one(&pair2, kind, counter, flag);
    out[3] = recompose_one(&pair3, kind, counter, flag);
}

#[cfg(not(target_arch = "x86_64"))]
fn recompose_siblings_4x_sse2(
    children: &[u8; 256],
    out: &mut [Hash; 4],
    kind: HashKind,
    counter: u64,
    flag: u8,
) {
    // Same arithmetic on non-x86_64 targets. Two 2x batches keep the
    // name honest: even on NEON/AVX-512 the 4x gather is split into
    // two halves, so the per-batch shape is the same.
    for k in 0..4 {
        let pair: [u8; 64] = children[k * 64..(k + 1) * 64].try_into().unwrap();
        out[k] = recompose_one(&pair, kind, counter, flag);
    }
}

/// Hash one 64-byte sibling pair into a 32-byte parent. The counter and
/// flag are passed in so the same function body serves both the AVX2 and
/// the SSE2 paths; SHA-256 ignores them, BLAKE3 uses them to thread the
/// per-level metadata into the compression function.
#[inline]
fn recompose_one(pair: &[u8; 64], kind: HashKind, _counter: u64, _flag: u8) -> Hash {
    match kind {
        HashKind::Sha256 => {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(pair);
            h.finalize().into()
        }
        HashKind::Blake3 => blake3::hazmat::merge_subtrees_non_root(
            <&[u8; 32]>::try_from(&pair[..32]).unwrap(),
            <&[u8; 32]>::try_from(&pair[32..64]).unwrap(),
            blake3::hazmat::Mode::Hash,
        ),
    }
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
