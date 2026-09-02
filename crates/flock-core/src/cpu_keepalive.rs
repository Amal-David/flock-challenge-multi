//! Prover CPU keep-alive: keeps core frequencies high across the ready→seed
//! gap by running cheap scalar spin threads on performance cores between
//! trials.

/// Start the background keep-alive spin threads. Idempotent: a second call
/// while threads are already spinning is a no-op.
#[inline]
pub fn keepalive_start() {
    imp::start();
}

/// Stop the keep-alive spin threads and wait for them to finish.
#[inline]
pub fn keepalive_stop() {
    imp::stop();
}

/// Signal the keep-alive spin threads to exit without waiting.
#[inline]
pub fn keepalive_signal() {
    imp::signal();
}

/// Wait until every spin thread has finished.
#[inline]
pub fn keepalive_join() {
    imp::join_all();
}

#[cfg(any(
    all(target_arch = "aarch64", target_os = "macos"),
    all(target_arch = "x86_64", target_os = "linux"),
    all(target_arch = "x86_64", target_os = "macos")
))]
mod imp {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Upper bound on how long the keep-alive spins before self-terminating.
    const MAX_KEEPALIVE: Duration = Duration::from_secs(2);

    /// `true` while the spin threads should keep running. Cleared by [`stop`];
    /// each thread also self-exits past `MAX_KEEPALIVE`.
    static RUNNING: AtomicBool = AtomicBool::new(false);

    /// Spin threads spawned but not yet finished.
    static LIVE: AtomicUsize = AtomicUsize::new(0);

    /// How long [`join_all`] may wait for the spin threads to notice the stop signal.
    const QUIET_TIMEOUT: Duration = Duration::from_micros(250);

    /// The per-core spin body: proof-irrelevant scalar-integer churn that keeps
    /// the core retiring instructions (so its DVFS clock request stays high)
    /// without SIMD/CLMUL power draw or any shared/memory state.
    fn spin_until_stopped(deadline: Instant) {
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        // qos

        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        while RUNNING.load(Ordering::Relaxed) {
            for _ in 0..64 {
                x = x
                    .wrapping_mul(0x2545_F491_4F6C_DD1D)
                    .wrapping_add(0x9E37_79B9_7F4A_7C15);
                x ^= x >> 29;
            }
            std::hint::black_box(x);
            if Instant::now() >= deadline {
                break;
            }
        }
        std::hint::black_box(x);
        LIVE.fetch_sub(1, Ordering::SeqCst);
    }

    pub(super) fn start() {
        if std::env::var_os("FLOCK_NO_CPU_KEEPALIVE").is_some() {
            return;
        }
        if RUNNING.swap(true, Ordering::SeqCst) {
            return;
        }
        let n_cores = rayon::current_num_threads().max(1);
        let deadline = Instant::now() + MAX_KEEPALIVE;
        for i in 0..n_cores {
            LIVE.fetch_add(1, Ordering::SeqCst);
            match std::thread::Builder::new()
                .name(format!("flock-keepalive-{i}"))
                .stack_size(64 * 1024)
                .spawn(move || spin_until_stopped(deadline))
            {
                Ok(_) => {}
                Err(_) => {
                    LIVE.fetch_sub(1, Ordering::SeqCst);
                    break;
                }
            }
        }
    }

    pub(super) fn stop() {
        signal();
        join_all();
    }

    pub(super) fn signal() {
        RUNNING.swap(false, Ordering::SeqCst);
    }

    pub(super) fn join_all() {
        let give_up_at = Instant::now() + QUIET_TIMEOUT;
        while LIVE.load(Ordering::SeqCst) != 0 {
            if Instant::now() >= give_up_at {
                return;
            }
            std::hint::spin_loop();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::Mutex;

        static TEST_LOCK: Mutex<()> = Mutex::new(());

        #[test]
        fn start_then_stop_is_clean() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            start();
            stop();
            assert!(!RUNNING.load(Ordering::SeqCst));
            assert_eq!(LIVE.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn stop_without_start_is_noop() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            stop();
            stop();
            assert!(!RUNNING.load(Ordering::SeqCst));
            assert_eq!(LIVE.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn restart_after_stop_is_clean() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            start();
            stop();
            start();
            stop();
            assert!(!RUNNING.load(Ordering::SeqCst));
            assert_eq!(LIVE.load(Ordering::SeqCst), 0);
        }
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_os = "macos"),
    all(target_arch = "x86_64", target_os = "linux"),
    all(target_arch = "x86_64", target_os = "macos")
)))]
mod imp {
    pub(super) fn start() {}
    pub(super) fn stop() {}
    pub(super) fn signal() {}
    pub(super) fn join_all() {}
}
