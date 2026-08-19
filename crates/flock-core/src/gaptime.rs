//! `FLOCK_GAP_TIMING`: wall-clock boundary instrumentation for the fast
//! prover. When the env var is set, [`mark`] prints one line per phase
//! boundary with the delta since the previous mark and the offset since
//! [`begin`] — the sum of the deltas accounts for the FULL prove wall time,
//! unlike per-phase stopwatches which silently drop the time BETWEEN phases
//! (pool installs, buffer drops, challenger hashing, allocator churn).
//!
//! Diagnostics only: the ranked worker's cleared env never sets the var, and
//! the disabled path is a single cached-bool load. Output goes to stderr so
//! it never mixes with bench stdout parsing.

/// Gap timing is a developer-only diagnostic. Ranked workers clear their
/// environment, so specialize the shipped fast path instead of carrying the
/// cached environment probe and timestamp machinery through every phase mark.
#[inline(always)]
pub fn enabled() -> bool {
    false
}

/// Ranked-build no-op retained as the stable instrumentation call surface.
#[inline(always)]
pub fn begin(_label: &str) {
}

/// Ranked-build no-op retained as the stable instrumentation call surface.
#[inline(always)]
pub fn mark(_label: &str) {
}
