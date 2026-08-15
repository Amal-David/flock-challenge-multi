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

pub mod chain;
pub mod merkle_path;
pub mod proof_io;
pub mod prover;
pub mod r1cs_hashes;

// dispersion-resample marker hermes-x86-r1-1786797518-12057
// dispersion-resample marker hermes-x86-r2-1786797518-32723
// dispersion-resample marker hermes-x86-r3-1786797518-26018
// dispersion-resample marker hermes-x86-r4-1786797518-29141
// dispersion-resample marker hermes-x86-r5-1786797518-24902

// dispersion-resample marker hermes-x86-s2-1786798338-26580
// dispersion-resample marker hermes-x86-s3-1786798338-9623
// dispersion-resample marker hermes-x86-s4-1786798338-22226

// dispersion-resample marker hermes-x86-defend-1786799202-23090
// dispersion-resample marker hermes-x86-s5-1786799202-26239
// dispersion-resample marker hermes-x86-s6-1786799202-12330
// dispersion-resample marker hermes-x86-s7-1786799202-13151

// dispersion-resample marker hermes-x86-push-1786800040-20769
// dispersion-resample marker hermes-x86-push2-1786800040-8481
// dispersion-resample marker hermes-x86-push3-1786800040-20716
// dispersion-resample marker hermes-x86-push4-1786800040-6819
// dispersion-resample marker hermes-x86-push5-1786800040-23863
