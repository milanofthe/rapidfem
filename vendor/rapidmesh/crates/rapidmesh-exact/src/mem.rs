//! The heap in use and its peak, where a build counts them: a binary that
//! installs [`Counting`] as its global allocator (the Python binding with
//! its `mem` feature) makes every [`crate::log::stage`] record the peak of
//! the heap since the stage before it (`mem.<stage>`, MB) and the
//! allocations made since then (`allocs.<stage>`). Elsewhere nothing is
//! counted and nothing is recorded.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static USED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ON: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);

/// The system allocator, counting the bytes it hands out.
pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        USED.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            USED.fetch_sub(layout.size(), Ordering::Relaxed);
            grow(new_size);
        }
        p
    }
}

fn grow(bytes: usize) {
    ON.store(true, Ordering::Relaxed);
    CALLS.fetch_add(1, Ordering::Relaxed);
    let now = USED.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

/// The heap in use (bytes), `None` where nothing counts.
pub fn used() -> Option<usize> {
    ON.load(Ordering::Relaxed)
        .then(|| USED.load(Ordering::Relaxed))
}

/// The allocations (and reallocations) since the last call, `None` where
/// nothing counts.
pub fn take_calls() -> Option<usize> {
    ON.load(Ordering::Relaxed)
        .then(|| CALLS.swap(0, Ordering::Relaxed))
}

/// The heap's peak since the last call (bytes), `None` where nothing counts.
pub fn take_peak() -> Option<usize> {
    ON.load(Ordering::Relaxed).then(|| {
        let now = USED.load(Ordering::Relaxed);
        PEAK.swap(now, Ordering::Relaxed)
    })
}
