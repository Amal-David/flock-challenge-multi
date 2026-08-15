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

// dispersion-resample marker hermes-x86-defend2-1786801413-5614
// dispersion-resample marker hermes-x86-defend3-1786801413-13170
// dispersion-resample marker hermes-x86-defend4-1786801413-1779

// dispersion-resample marker hermes-x86-extend-1786802291-18391
// dispersion-resample marker hermes-x86-extend2-1786802291-29261
// dispersion-resample marker hermes-x86-extend3-1786802291-27042
// dispersion-resample marker hermes-x86-extend4-1786802291-4513
// dispersion-resample marker hermes-x86-extend5-1786802291-19317

// dispersion-resample marker hermes-x86-push6-1786803203-10980
// dispersion-resample marker hermes-x86-push7-1786803203-23514
// dispersion-resample marker hermes-x86-push8-1786803203-23494
// dispersion-resample marker hermes-x86-push9-1786803203-4845
// dispersion-resample marker hermes-x86-push10-1786803203-30901

// dispersion-resample marker hermes-x86-streak-1786804045-6373
// dispersion-resample marker hermes-x86-streak2-1786804045-15142
// dispersion-resample marker hermes-x86-streak3-1786804045-3389
// dispersion-resample marker hermes-x86-streak4-1786804045-22364
// dispersion-resample marker hermes-x86-streak5-1786804045-7221

// dispersion-resample marker hermes-x86-s8-1786804868-10967
// dispersion-resample marker hermes-x86-s9-1786804868-18971
// dispersion-resample marker hermes-x86-s10-1786804868-23880
// dispersion-resample marker hermes-x86-s11-1786804868-32729
// dispersion-resample marker hermes-x86-s12-1786804868-13223

// dispersion-resample marker hermes-x86-b9-1786806411-7298
// dispersion-resample marker hermes-x86-b9b-1786806411-9268
// dispersion-resample marker hermes-x86-b9c-1786806411-11474
// dispersion-resample marker hermes-x86-b9d-1786806411-8995
// dispersion-resample marker hermes-x86-b9e-1786806411-7502

// dispersion-resample marker hermes-x86-b10-1786807305-9084
// dispersion-resample marker hermes-x86-b10b-1786807305-12581
// dispersion-resample marker hermes-x86-b10c-1786807305-25233
// dispersion-resample marker hermes-x86-b10d-1786807305-25108
// dispersion-resample marker hermes-x86-b10e-1786807305-8341
