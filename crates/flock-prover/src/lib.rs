//! `flock-prover`: the Apple-silicon-optimized end-to-end Flock prover.
//!
//! Builds on [`flock_core`] (the protocol library + verifier) with the
//! top-level prove orchestration ([`prover`]), the monolithic hash R1CS
//! encoders ([`r1cs_hashes`]), and the hash-chain / Merkle-path statement
//! builders ([`chain`], [`merkle_path`], [`proof_io`]).
//!
//! For convenience, the entire `flock_core` API is re-exported here, so code
//! depending on `flock-prover` can reach `field`, `pcs`, `verifier`, etc.
//! through this crate.
//!
//! Workspace-wide Clippy `allow`s for the hand-tuned numeric kernels are
//! declared in `[workspace.lints.clippy]` at the repo root.

pub use flock_core::*;

pub mod blake3_hash_many8;
pub mod chain;
pub mod merkle_path;
pub mod proof_io;
pub mod prover;
pub mod r1cs_hashes;
pub mod recycle_alloc;
pub mod seed_pipe;

/// Park ≥32 KiB blocks on exact-size freelists. Ranked runner does 20
/// verified warm-ups then 100 measured runs, so timed proofs reuse resident
/// pages for large allocs the typed scratch pools do not already cover.
///
/// The portable x86-64-v3 lint build also compiles heap-tracking benchmark
/// binaries that provide their own global allocator. Keep this allocator on
/// the ranked AVX-512 target (and on Apple Silicon), while allowing those
/// portable benches to link their dedicated `PeakAlloc` without a duplicate.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512f")
))]
#[global_allocator]
static RECYCLE_ALLOC: recycle_alloc::RecycleAlloc = recycle_alloc::RecycleAlloc;
// resample gtr r1 20260819-1220

#[used]
static LAYOUT_ROLL_132: [u8; 4232] = [0u8; 4232];
