//! `flock-core`: the protocol library and verifier for Flock's R1CS-over-GF(2)
//! sumcheck/zerocheck PIOP with a multilinear PCS.
//!
//! This crate carries everything the verifier needs. It is portable — the NEON
//! kernels in `field`, `ntt`, `lincheck`, `zerocheck`, and `merkle` have scalar
//! fallbacks — though it is tuned for Apple silicon. The end-to-end prover, the
//! hash R1CS encoders, and the CLI live in the `flock-prover` crate built on
//! top of this one.
//!
//! Protocol flow:
//!   1. Prover commits to the witness z ∈ GF(2)^n via a multilinear PCS.
//!   2. Prover computes the row-witnesses a = A·z, b = B·z, c = C·z.
//!   3. Zerocheck PIOP reduces a·b ⊕ c = 0 to evaluation claims on (â, b̂, ĉ) at ρ.
//!   4. Lincheck PIOP reduces those to a single evaluation claim ẑ(ρ') = v.
//!   5. PCS opens ẑ at ρ'.
//!
//! Workspace-wide Clippy `allow`s for the hand-tuned numeric kernels are
//! declared in `[workspace.lints.clippy]` at the repo root.

pub mod bits;
pub mod challenger;
pub mod field;
pub mod gaptime;
pub mod gpu;
pub mod hash;
pub mod lincheck;
pub mod merkle;
pub mod ntt;
pub mod pcs;
pub mod permutation;
pub mod proof;
pub mod r1cs;
pub mod scratch;
pub mod verifier;
pub mod zerocheck;

/// Configure rayon's global thread pool to use only performance cores on
/// Apple silicon (excluding efficiency cores).
///
/// On M-series chips the 2 efficiency cores run at ~30-40% of perf-core
/// speed and become stragglers in compute-bound parallel work — the
/// work-stealing scheduler keeps assigning them tasks that hold up the perf
/// cores at synchronization barriers. Empirically, 8 threads beats 10 by
/// ~10-20% on `pcs::commit` and similar parallel-NTT workloads.
///
/// Call this **once** at program startup, before any other parallel flock
/// code runs (rayon's global pool is set on first use; if it's already
/// created, this call is a no-op).
///
/// Respects `RAYON_NUM_THREADS` — if that env var is set, this function
/// does nothing (so explicit user configuration always wins).
///
/// Returns the number of threads the pool was configured with, or `None`
/// if no change was made (either because the env var was set or because
/// rayon was already initialized).
pub fn init_perf_thread_pool() -> Option<usize> {
    if let Some(n) = topology_pool::try_build_global() {
        return Some(n);
    }
    if std::env::var("RAYON_NUM_THREADS").is_ok() {
        return None;
    }
    let n = perf_core_count();
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
    {
        Ok(()) => Some(n),
        Err(_) => None, // pool already built
    }
}

/// Topology-aware global rayon pool for the ranked x86 worker (Linux only).
///
/// The ranked runner exposes 16 vCPUs = 8 physical cores × 2 SMT threads and
/// the harness sets `RAYON_NUM_THREADS=16`, so rayon spins 16 workers that the
/// guest scheduler is free to migrate and to double up on one core while a
/// sibling core idles. This builds the global pool explicitly and pins worker
/// `i` to one logical CPU chosen from `/sys/devices/system/cpu/cpu*/topology/
/// thread_siblings_list`: mode `pin16` (default here) keeps the harness's
/// thread count but gives every worker a fixed logical CPU, alternating
/// physical cores first (worker 0 → core 0 sibling 0, worker 1 → core 1
/// sibling 0, … worker 8 → core 0 sibling 1, …), so a partially-idle pool
/// still spreads across cores; mode `phys8` uses one worker per physical
/// core. Selected at build time by `POOL_MODE`; `FLOCK_NO_TOPOLOGY_POOL=1`
/// disables it (rayon then builds its default pool from `RAYON_NUM_THREADS`).
/// Engages only when the machine has exactly 2 SMT threads per core and
/// `RAYON_NUM_THREADS` equals the logical CPU count (i.e. the ranked shape);
/// anything else falls through to the incumbent behaviour. Pure scheduling:
/// no arithmetic changes, proof bytes identical.
pub(crate) mod topology_pool {
    /// "pin16" | "phys8"
    pub(crate) const POOL_MODE: &str = "pin16";

    #[cfg(target_os = "linux")]
    fn sibling_groups() -> Option<Vec<Vec<usize>>> {
        let n = std::thread::available_parallelism().ok()?.get();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut seen = vec![false; n];
        for cpu in 0..n {
            if seen[cpu] {
                continue;
            }
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
            let list = std::fs::read_to_string(path).ok()?;
            let mut g = Vec::new();
            for part in list.trim().split(',') {
                if let Some((a, b)) = part.split_once('-') {
                    let (a, b): (usize, usize) = (a.parse().ok()?, b.parse().ok()?);
                    g.extend(a..=b);
                } else {
                    g.push(part.parse().ok()?);
                }
            }
            g.retain(|&c| c < n);
            g.sort_unstable();
            g.dedup();
            for &c in &g {
                seen[c] = true;
            }
            groups.push(g);
        }
        Some(groups)
    }

    #[cfg(target_os = "linux")]
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }

    #[cfg(target_os = "linux")]
    fn pin_to(cpu: usize) {
        let mut mask = [0u64; 16]; // 1024 CPUs
        if cpu >= 1024 {
            return;
        }
        mask[cpu / 64] |= 1u64 << (cpu % 64);
        // SAFETY: plain syscall on the calling thread with a valid mask buffer;
        // failure is ignored (pure hint).
        unsafe {
            let _ = sched_setaffinity(0, core::mem::size_of_val(&mask), mask.as_ptr());
        }
    }

    pub(crate) fn try_build_global() -> Option<usize> {
        #[cfg(not(target_os = "linux"))]
        {
            return None;
        }
        #[cfg(target_os = "linux")]
        {
            if std::env::var_os("FLOCK_NO_TOPOLOGY_POOL").is_some() {
                return None;
            }
            let want: usize = std::env::var("RAYON_NUM_THREADS").ok()?.parse().ok()?;
            let logical = std::thread::available_parallelism().ok()?.get();
            if want != logical {
                return None;
            }
            let groups = sibling_groups()?;
            let cores = groups.len();
            if cores == 0 || logical != 2 * cores || groups.iter().any(|g| g.len() != 2) {
                return None;
            }
            // Worker → CPU map: cores first (sibling 0 of each core), then the
            // second siblings; phys8 uses only the first half.
            let mut order: Vec<usize> = Vec::with_capacity(logical);
            for s in 0..2 {
                for g in &groups {
                    order.push(g[s]);
                }
            }
            let n = if POOL_MODE == "phys8" { cores } else { logical };
            let order = std::sync::Arc::new(order);
            let o2 = order.clone();
            match rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .start_handler(move |i| {
                    if let Some(&cpu) = o2.get(i) {
                        pin_to(cpu);
                    }
                })
                .build_global()
            {
                Ok(()) => Some(n),
                Err(_) => None,
            }
        }
    }
}


/// Best-effort `madvise(MADV_HUGEPAGE)` for multi-MB buffers. Ubuntu ships
/// THP in `madvise` mode, so without this hint the prover's 32 MB-1 GB
/// working buffers sit on 4 KiB pages through every strided sweep. Advising
/// at allocation time means the setup-phase prewarm faults the pool's pages
/// as 2 MiB pages once, before any timed proof. Raw syscall (no libc dep);
/// errors ignored. `FLOCK_NO_HUGEPAGES` is a local-diagnostics kill switch.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn advise_hugepages(ptr: *mut u8, bytes: usize) {
    const HUGE: usize = 1 << 21;
    if bytes < HUGE {
        return;
    }
    // Ablation: the advice is withheld unconditionally, so the prover's
    // multi-megabyte buffers stay on 4 KiB pages. The ranked environment is
    // cleared, so the kill switch below can never be set there and the
    // mechanism cannot be sampled through it; withholding the advice in the
    // source is the only way to measure what it is worth on this instance.
    // Every allocation is otherwise identical -- same sizes, same alignment,
    // same contents, same order -- so the difference is page size alone.
    return;
    #[allow(unreachable_code)]
    {
        static DISABLED: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_HUGEPAGES").is_some());
        if *DISABLED {
            return;
        }
    }
    const PAGE: usize = 4096;
    let start = (ptr as usize).next_multiple_of(PAGE);
    let end = ptr as usize + bytes;
    if end <= start {
        return;
    }
    const SYS_MADVISE: usize = 28;
    const MADV_HUGEPAGE: usize = 14;
    // SAFETY: the advised range lies within the just-allocated buffer, and
    // MADV_HUGEPAGE never alters contents or mapping validity.
    unsafe {
        let ret: isize;
        core::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MADVISE as isize => ret,
            in("rdi") start,
            in("rsi") end - start,
            in("rdx") MADV_HUGEPAGE,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
        let _ = ret;
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn advise_hugepages(_ptr: *mut u8, _bytes: usize) {}

/// Allocate a `Vec<T>` of length `n` whose contents are NOT zero-initialized.
/// Caller MUST write every slot before reading it.
///
/// Used to skip the eager zero-init of large ping-pong buffers in hot prover
/// paths (PCS open, Round-2 fold, NTT scratch, lincheck packing). At m=29 the
/// zero-fill of a fresh 128 MB `vec![T::default(); n]` runs sequentially on
/// the main thread (~22 ms), which caps the parallel speedup of those phases.
///
/// `T: Copy` ensures `T` has no Drop impl, so the leaked uninitialized
/// elements are a no-op on drop.
///
/// # Safety contract
///
/// Reading uninitialized memory is UB per Rust's memory model regardless of
/// whether all bit patterns are valid for `T`. Caller must ensure every slot
/// is written before any read.
// `uninit_vec` flags exactly this pattern; here it is the deliberate purpose of
// the function (the safety contract above is what makes it sound).
#[allow(clippy::uninit_vec)]
pub(crate) fn alloc_uninit_vec<T: Copy>(n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    // SAFETY:
    // - capacity == n was just allocated, so set_len(n) is in bounds.
    // - T: Copy implies !Drop, so leaking uninit elements is a no-op.
    // - Caller upholds write-before-read.
    unsafe {
        v.set_len(n);
    }
    advise_hugepages(v.as_mut_ptr().cast::<u8>(), n * core::mem::size_of::<T>());
    // A fresh allocation may land on the address of a buffer that was
    // released by a plain `drop` while a scratch provenance tag was still
    // armed for it. This address now belongs to a new object, so that tag is
    // void — drop it before the new buffer can inherit it at its own
    // release. Lock-free (a short atomic scan) when nothing is armed.
    crate::scratch::void_pending_tag(v.as_ptr().cast::<crate::field::F128>());
    v
}

/// Compatibility shim — same as `alloc_uninit_vec::<F128>(n)`.
pub(crate) fn alloc_uninit_f128_vec(n: usize) -> Vec<crate::field::F128> {
    alloc_uninit_vec::<crate::field::F128>(n)
}

/// Ranked default parallelizes a family of previously single-threaded
/// segments on the prover's Fiat–Shamir chain — pure per-element gathers,
/// bit-transposes, and eq-table builds that ran alone on one core while the
/// pool idled (ligerito query-phase gathers, zerocheck round-one table
/// preparation). Every gated site is byte-neutral: pure functions of
/// already-sampled values, order-preserving indexed parallel maps, and
/// exact-field factorizations — nothing the challenger observes changes.
/// Read once per process. `FLOCK_NO_SERIAL_PAR=1` restores every incumbent
/// sequential form (each site keeps it verbatim as its oracle).
pub(crate) fn serial_par_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_SERIAL_PAR").is_none());
    *ON
}

/// Cached [`perf_core_count`]. The uncached version may spawn `sysctl`; this
/// memoizes it so hot paths can cheaply ask "is the current rayon pool the
/// homogeneous P-core pool?" (i.e. `current_num_threads() <= this`).
#[cfg(target_arch = "aarch64")]
pub(crate) fn perf_core_count_cached() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(perf_core_count)
}

/// Best-effort count of **physical** performance cores used to size the
/// prover's thread pool. The hot phases are CLMUL-heavy and/or
/// memory-bandwidth-bound; SMT siblings share the core's execution ports and
/// add no DRAM bandwidth, so running 2 threads per physical core only adds
/// contention (on a 32C/64T Threadripper the prove is ~16% faster at 32 threads
/// than 64). On macOS, queries `hw.perflevel0.physicalcpu` (= P-core count on
/// Apple silicon, = physical CPU count on Intel). On Linux, `available_
/// parallelism()` counts SMT siblings, so derive physical cores from `/sys`
/// topology and clamp that host-wide count to the process's affinity/cgroup
/// availability. Elsewhere, falls back to `available_parallelism()`.
fn perf_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.physicalcpu"])
            .output()
            && let Ok(s) = std::str::from_utf8(&out.stdout)
            && let Ok(n) = s.trim().parse::<usize>()
            && n > 0
        {
            return n;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(n) = linux_physical_cores()
            && n > 0
        {
            let available = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            return n.min(available);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Count distinct physical cores via `/sys` topology: one entry per unique
/// `(physical_package_id, core_id)` over the online `cpuN` directories. Returns
/// `None` if the topology can't be read (caller falls back to logical count).
#[cfg(target_os = "linux")]
fn linux_physical_cores() -> Option<usize> {
    use std::collections::HashSet;
    let mut cores: HashSet<(String, String)> = HashSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let Some(rest) = name.strip_prefix("cpu") else {
            continue;
        };
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            continue; // skip "cpufreq", "cpuidle", etc.
        }
        let topo = path.join("topology");
        let core_id = std::fs::read_to_string(topo.join("core_id")).ok();
        let pkg = std::fs::read_to_string(topo.join("physical_package_id")).ok();
        if let (Some(c), Some(p)) = (core_id, pkg) {
            cores.insert((p.trim().to_owned(), c.trim().to_owned()));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}
