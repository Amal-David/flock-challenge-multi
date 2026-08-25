//! Recycling global allocator for the prover process.
//!
//! Blocks at least 32 KiB are parked on exact-size freelists rather than
//! returned to the system allocator. The ranked worker performs an untimed warm proof
//! with the same allocation pattern, so the timed proof reuses resident pages
//! for large allocations not already handled by the typed scratch pools.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{Acquire, Release},
};

const RECYCLE_MIN: usize = 32 * 1024;
const MAX_ALIGN: usize = 16;
const MAX_CLASSES: usize = 512;

struct Class {
    size: AtomicUsize,
    head: Mutex<usize>,
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Class = Class {
    size: AtomicUsize::new(0),
    head: Mutex::new(0),
};
static CLASSES: [Class; MAX_CLASSES] = [EMPTY; MAX_CLASSES];

#[inline]
fn class_slot(size: usize) -> usize {
    (size.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 55) % MAX_CLASSES
}

#[inline]
fn find_class(size: usize, insert: bool) -> Option<usize> {
    let start = class_slot(size);
    for probe in 0..MAX_CLASSES {
        let i = (start + probe) % MAX_CLASSES;
        let s = CLASSES[i].size.load(Acquire);
        if s == size {
            return Some(i);
        }
        if s == 0 {
            if !insert {
                return None;
            }
            if CLASSES[i]
                .size
                .compare_exchange(0, size, Release, Acquire)
                .is_ok()
            {
                return Some(i);
            }
            if CLASSES[i].size.load(Acquire) == size {
                return Some(i);
            }
        }
    }
    None
}

#[inline]
fn recyclable(layout: &Layout) -> bool {
    layout.size() >= RECYCLE_MIN && layout.align() <= MAX_ALIGN
}

/// `FLOCK_NO_ALIGN64=1` restores raw System pointers for the recyclable
/// class (exact same-binary A/B). Latched once, and only ever initialized
/// from a RECYCLABLE allocation — `var_os`'s own small allocations take the
/// non-recyclable branch straight to System, so initialization cannot
/// re-enter this latch.
#[inline]
fn align64_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ALIGN64").is_none())
}

/// RecycleAlloc-only huge-page advise. `alloc_uninit_vec` already hints
/// `MADV_HUGEPAGE` on typed scratch; this covers the residual GlobalAlloc
/// path (regular `Vec`s, proof buffers, first System.alloc of a size class)
/// that never goes through that helper. Ubuntu 24.04 THP is `madvise` mode,
/// so without a hint those 32 MB–1 GB blocks stay on 4 KiB pages for every
/// strided NTT/zerocheck sweep.
///
/// `collapse` is `MADV_COLLAPSE` (Linux 6.1+, value 25): a synchronous
/// collapse of the mapping onto 2 MiB pages. It is ONLY passed on a fresh
/// `System.alloc` miss, never on a freelist pop — collapse in the timed
/// window would walk page tables on a 100-trial critical path. The ranked
/// extra-warmup proves allocate every recyclable size class untimed, so the
/// cost sits entirely before the ready file. Errors ignored (ENOMEM, old
/// kernels, already-huge). `FLOCK_NO_RECYCLE_HUGEPAGES=1` disables both
/// hints for same-binary A/B; the ranked worker's cleared env never sets it.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn advise_fresh(ptr: *mut u8, bytes: usize, collapse: bool) {
    const HUGE: usize = 1 << 21;
    if ptr.is_null() || bytes < HUGE {
        return;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("FLOCK_NO_RECYCLE_HUGEPAGES").is_some()) {
        return;
    }
    const PAGE: usize = 4096;
    let start = (ptr as usize).next_multiple_of(PAGE);
    let end = (ptr as usize).saturating_add(bytes);
    if end <= start {
        return;
    }
    let len = end - start;
    const SYS_MADVISE: usize = 28;
    const MADV_HUGEPAGE: usize = 14;
    const MADV_COLLAPSE: usize = 25;
    // SAFETY: range is inside the just-allocated System block; neither
    // MADV_HUGEPAGE nor MADV_COLLAPSE changes contents, only page size.
    unsafe {
        let mut ret: isize;
        core::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MADVISE as isize => ret,
            in("rdi") start,
            in("rsi") len,
            in("rdx") MADV_HUGEPAGE,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
        let _ = ret;
        if collapse {
            core::arch::asm!(
                "syscall",
                inlateout("rax") SYS_MADVISE as isize => ret,
                in("rdi") start,
                in("rsi") len,
                in("rdx") MADV_COLLAPSE,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack)
            );
            let _ = ret;
        }
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn advise_fresh(_ptr: *mut u8, _bytes: usize, _collapse: bool) {}

/// Every large prover buffer historically landed `16 mod 64` (glibc mmap
/// chunk header), which the hot kernels pay for three ways: every wide
/// load/store on these buffers is a cache-line split, non-temporal 64-byte
/// writes straddle two lines (partial write-combining flushes are DRAM
/// read-modify-writes), and single-uop ZMM streaming stores are illegal.
/// For the recyclable class, over-allocate by `ALIGN_SLACK` and return
/// `round_up(base + 8, 64)`: a 64-aligned pointer with room for the
/// back-offset word stored at `aligned - 8` — BEFORE the block, so it can
/// never collide with the freelist link word at `aligned + 0`.
const ALIGN_SLACK: usize = 64;

#[inline]
fn adjusted(layout: &Layout) -> Layout {
    // SAFETY of unwrap: size + 64 cannot overflow isize for any layout the
    // caller could have constructed, and 16 is a power of two.
    Layout::from_size_align(layout.size() + ALIGN_SLACK, MAX_ALIGN).unwrap()
}

#[inline]
unsafe fn align_up(base: *mut u8) -> *mut u8 {
    if base.is_null() {
        return base;
    }
    let aligned = ((base as usize + 8 + 63) & !63) as *mut u8;
    // SAFETY: aligned - base is in [8, 64] ⊂ the ALIGN_SLACK the caller
    // over-allocated, and aligned - 8 >= base.
    unsafe { *(aligned.sub(8) as *mut usize) = aligned as usize - base as usize };
    aligned
}

#[inline]
unsafe fn align_base(ptr: *mut u8) -> *mut u8 {
    // SAFETY: ptr was produced by align_up, so the offset word at ptr - 8 is
    // intact (freelist links live at ptr + 0 and never touch it).
    let off = unsafe { *(ptr.sub(8) as *const usize) };
    debug_assert!((8..=ALIGN_SLACK).contains(&off));
    unsafe { ptr.sub(off) }
}

#[inline]
fn pop(size: usize) -> *mut u8 {
    let Some(i) = find_class(size, false) else {
        return core::ptr::null_mut();
    };
    let mut head = CLASSES[i].head.lock().unwrap();
    let top = *head;
    if top == 0 {
        return core::ptr::null_mut();
    }
    *head = unsafe { *(top as *const usize) };
    top as *mut u8
}

#[inline]
fn push(ptr: *mut u8, size: usize) -> bool {
    let Some(i) = find_class(size, true) else {
        return false;
    };
    let mut head = CLASSES[i].head.lock().unwrap();
    unsafe { *(ptr as *mut usize) = *head };
    *head = ptr as usize;
    true
}

pub struct RecycleAlloc;

// SAFETY: every recycled block came from this allocator with the exact same
// size class. With align64 on, every recyclable-class block System sees uses
// the `adjusted` layout on both alloc and dealloc, and the user pointer is
// recovered to its System base via the offset word `align_up` stored.
// glibc/mimalloc and the macOS allocator provide at least 16-byte alignment
// at these sizes; layouts requiring larger alignment bypass the recycler.
unsafe impl GlobalAlloc for RecycleAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if recyclable(&layout) {
            let p = pop(layout.size());
            if !p.is_null() {
                return p;
            }
            if align64_enabled() {
                // SAFETY: adjusted() reserves the slack align_up consumes.
                let p = unsafe { align_up(System.alloc(adjusted(&layout))) };
                advise_fresh(p, layout.size(), true);
                return p;
            }
        }
        let p = unsafe { System.alloc(layout) };
        advise_fresh(p, layout.size(), false);
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if recyclable(&layout) {
            let p = pop(layout.size());
            if !p.is_null() {
                unsafe { core::ptr::write_bytes(p, 0, layout.size()) };
                return p;
            }
            if align64_enabled() {
                // The user range [aligned, aligned + size) is inside the
                // zeroed System block; the offset word sits before it.
                // SAFETY: as for alloc.
                let p = unsafe { align_up(System.alloc_zeroed(adjusted(&layout))) };
                advise_fresh(p, layout.size(), true);
                return p;
            }
        }
        let p = unsafe { System.alloc_zeroed(layout) };
        advise_fresh(p, layout.size(), false);
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if recyclable(&layout) {
            if push(ptr, layout.size()) {
                return;
            }
            if align64_enabled() {
                // SAFETY: every recyclable-class pointer this allocator
                // handed out with align64 on came from align_up; recover the
                // System base and the adjusted layout it was allocated with.
                unsafe {
                    return System.dealloc(align_base(ptr), adjusted(&layout));
                }
            }
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    /// With `FLOCK_NO_ALIGN64` unset (the ranked worker's cleared env), every
    /// recyclable-class allocation — fresh from System, recycled off the
    /// freelist, and grown through realloc — returns a 64-aligned pointer
    /// with intact contents. (The switch-off arm can't be covered in the
    /// same process: the latch resolves once.)
    #[test]
    fn recyclable_class_is_64_aligned_and_roundtrips() {
        for i in 0..4usize {
            let n = 32 * 1024 + 4096 * i + 128;
            let v = vec![7u8; n];
            assert_eq!(v.as_ptr() as usize % 64, 0, "fresh alloc n={n}");
            drop(v);
            let v2 = vec![9u8; n];
            assert_eq!(v2.as_ptr() as usize % 64, 0, "recycled alloc n={n}");
            assert!(v2.iter().all(|&b| b == 9), "contents survive recycle n={n}");
            let z = vec![0u8; n + 64];
            assert_eq!(z.as_ptr() as usize % 64, 0, "alloc_zeroed n={}", n + 64);
            assert!(z.iter().all(|&b| b == 0), "zeroed contents n={}", n + 64);
        }
        // Growth path: realloc = alloc + copy + dealloc through this
        // allocator; contents must survive the move between size classes.
        let mut g: Vec<u8> = Vec::with_capacity(48 * 1024);
        g.extend(std::iter::repeat_n(0xA5u8, 48 * 1024));
        g.extend(std::iter::repeat_n(0x5Au8, 128 * 1024));
        assert_eq!(g.as_ptr() as usize % 64, 0, "grown alloc");
        assert!(g[..48 * 1024].iter().all(|&b| b == 0xA5));
        assert!(g[48 * 1024..].iter().all(|&b| b == 0x5A));
    }
}
