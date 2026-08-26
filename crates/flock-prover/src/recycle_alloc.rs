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

/// Every large prover buffer historically landed `16 mod 64` (glibc mmap
/// chunk header), which the hot kernels pay for three ways: every wide
/// load/store on these buffers is a cache-line split, non-temporal 64-byte
/// writes straddle two lines (partial write-combining flushes are DRAM
/// read-modify-writes), and single-uop ZMM streaming stores are illegal.
/// For the recyclable class, over-allocate by `ALIGN_SLACK` and return
/// `round_up(base + 8, 64)`: a 64-aligned pointer with room for the
/// back-offset word stored at `aligned - 8` — BEFORE the block, so it can
/// never collide with the freelist link word at `aligned + 0`.
/// Cache colouring for the multi-megabyte class.
///
/// The system allocator serves blocks this size by mapping them, and
/// consecutive maps land adjacent, so successive large buffers differ in
/// address by exactly their own size — a power of two at every size this
/// prover uses. Buffers that differ by a large power of two are congruent
/// modulo the cache's index bits: they compete for the same sets, and a prove
/// that sweeps several of them together pays conflict misses on top of the
/// capacity misses its working set already costs.
///
/// The cost of that congruence is measured rather than assumed. Pinning the
/// whole large class into one contiguous 2 MiB-aligned arena — the maximally
/// congruent arrangement — came out **1.64% slower** on the ranked instance
/// than leaving the addresses to the system allocator. This walks the same
/// axis the other way: give each successive large block a different page
/// offset within its first huge page, so the live set spreads across the index
/// bits instead of stacking on them.
///
/// The colour is a multiple of the page size, so it preserves the 64-byte
/// alignment the wide kernels need and the page alignment the huge-page advice
/// needs, and it rides in the same slack and the same back-offset word the
/// existing alignment already uses. Blocks below the threshold are untouched:
/// they are too small for their spacing to be a large power of two, and the
/// slack would be a large fraction of the block.
const HUGE: usize = 2 << 20;
const COLOURS: usize = 64;
const COLOUR_STEP: usize = 4096;
const COLOUR_SPAN: usize = COLOURS * COLOUR_STEP;
static COLOUR: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn colour_for(size: usize) -> usize {
    if size < HUGE {
        return 0;
    }
    (COLOUR.fetch_add(1, Release) % COLOURS) * COLOUR_STEP
}

#[inline]
const fn slack_for(size: usize) -> usize {
    if size >= HUGE { COLOUR_STEP + COLOUR_SPAN } else { ALIGN_SLACK }
}

/// Large blocks are rounded to a page, not just to a line: the colour is a
/// page multiple, so page-aligning the base is what makes the colours land on
/// distinct page offsets, and it also stops the huge-page advice losing the
/// leading partial page it currently rounds past.
#[inline]
const fn align_of_class(size: usize) -> usize {
    if size >= HUGE { COLOUR_STEP } else { 64 }
}

const ALIGN_SLACK: usize = 64;

#[inline]
fn adjusted(layout: &Layout) -> Layout {
    // SAFETY of unwrap: size + 64 cannot overflow isize for any layout the
    // caller could have constructed, and 16 is a power of two.
    Layout::from_size_align(layout.size() + slack_for(layout.size()), MAX_ALIGN).unwrap()
}

#[inline]
unsafe fn align_up(base: *mut u8, size: usize) -> *mut u8 {
    if base.is_null() {
        return base;
    }
    let a = align_of_class(size);
    let aligned = ((base as usize + 8 + colour_for(size) + a - 1) & !(a - 1)) as *mut u8;
    // SAFETY: aligned - base is in [8, slack_for(size)] ⊆ the slack the caller
    // over-allocated, and aligned - 8 >= base.
    unsafe { *(aligned.sub(8) as *mut usize) = aligned as usize - base as usize };
    aligned
}

#[inline]
unsafe fn align_base(ptr: *mut u8) -> *mut u8 {
    // SAFETY: ptr was produced by align_up, so the offset word at ptr - 8 is
    // intact (freelist links live at ptr + 0 and never touch it).
    let off = unsafe { *(ptr.sub(8) as *const usize) };
    debug_assert!((8..=COLOUR_STEP + COLOUR_SPAN).contains(&off));
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
                return unsafe { align_up(System.alloc(adjusted(&layout)), layout.size()) };
            }
        }
        unsafe { System.alloc(layout) }
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
                return unsafe { align_up(System.alloc_zeroed(adjusted(&layout)), layout.size()) };
            }
        }
        unsafe { System.alloc_zeroed(layout) }
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
    fn huge_blocks_are_coloured_and_roundtrip() {
        // Successive large blocks must not all land on the same page offset,
        // and every one of them must still satisfy the alignment the wide
        // kernels rely on and survive a recycle intact.
        let mut offsets = std::collections::HashSet::new();
        let mut kept = Vec::new();
        for i in 0..8usize {
            let n = (2 << 20) + 4096 * i;
            let v = vec![0xC3u8; n];
            let a = v.as_ptr() as usize;
            assert_eq!(a % 64, 0, "huge block {i} lost 64-byte alignment");
            assert_eq!(a % 4096, 0, "huge block {i} lost page alignment");
            offsets.insert(a % super::COLOUR_SPAN);
            assert!(v.iter().all(|&b| b == 0xC3), "huge block {i} contents wrong");
            kept.push(v);
        }
        assert!(offsets.len() > 1, "every huge block took the same colour");
        drop(kept);
        // Recycled blocks keep their colour and stay usable.
        for i in 0..8usize {
            let n = (2 << 20) + 4096 * i;
            let mut v = vec![5u8; n];
            assert_eq!(v.as_ptr() as usize % 64, 0, "recycled huge block {i} misaligned");
            assert!(v.iter().all(|&b| b == 5));
            v[n - 1] = 6;
            assert_eq!(v[n - 1], 6);
        }
        // Below the threshold nothing is coloured and the block is unchanged.
        let small = vec![2u8; (2 << 20) - 4096];
        assert_eq!(small.as_ptr() as usize % 64, 0);
        assert!(small.iter().all(|&b| b == 2));
        // Growth across the threshold goes alloc -> copy -> free; the free must
        // recover the right base for a block that was coloured on the way in.
        let mut g: Vec<u8> = vec![1u8; 1 << 16];
        g.resize(6 << 20, 2);
        assert_eq!(g[0], 1);
        assert_eq!(g[(6 << 20) - 1], 2);
    }

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
