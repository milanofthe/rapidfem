//! Per-supernode storage of the left-looking factorizations: the per-index
//! cells of the emit state, the raw panel pointer for disjoint parallel
//! writes, and the scratch pool the node kernels borrow from.

use crate::memory::{HeapBytes, SCRATCH_KEEP};

/// Raw base pointer of a panel buffer, smuggled across rayon workers so each
/// task can write its own **disjoint row range** of a column-major panel. Safe
/// only because callers partition the rows so no two tasks touch the same cell.
pub(crate) struct PanelPtr<T>(pub *mut T);
impl<T> Clone for PanelPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for PanelPtr<T> {}
// SAFETY: the pointer is only dereferenced on disjoint, caller-partitioned cells.
unsafe impl<T> Send for PanelPtr<T> {}
unsafe impl<T> Sync for PanelPtr<T> {}
impl<T> PanelPtr<T> {
    /// Extract the raw pointer. Taking `self` by value forces a closure to
    /// capture the whole (Send+Sync) wrapper rather than disjoint-capturing the
    /// bare field.
    #[inline]
    pub fn get(self) -> *mut T {
        self.0
    }
}

/// A fixed-size array of independently written cells: each index is written by
/// exactly one owner (disjoint indices) and read only after a happens-before
/// barrier (subtree join / refcount Acquire-Release). Centralizes the
/// `Vec<UnsafeCell<V>>` pattern of the left-looking emit state.
pub(crate) struct Cells<V>(Vec<std::cell::UnsafeCell<V>>);

// SAFETY: disjoint-index writes; cross-thread visibility is the caller's
// barrier (see the type doc).
unsafe impl<V: Send> Sync for Cells<V> {}

impl<V: Default> Cells<V> {
    /// `n` default-initialized cells (for payloads without a cheap `Clone`).
    pub fn new_default(n: usize) -> Self {
        Cells(
            (0..n)
                .map(|_| std::cell::UnsafeCell::new(V::default()))
                .collect(),
        )
    }
}

impl<V: Clone> Cells<V> {
    pub fn new(n: usize, init: V) -> Self {
        Cells(
            (0..n)
                .map(|_| std::cell::UnsafeCell::new(init.clone()))
                .collect(),
        )
    }
}

impl<V> Cells<V> {
    /// SAFETY: `i` is this caller's exclusively owned index.
    #[inline]
    pub unsafe fn set(&self, i: usize, v: V) {
        *self.0[i].get() = v;
    }
    /// SAFETY: the write to `i` happened-before this read.
    #[inline]
    pub unsafe fn get(&self, i: usize) -> &V {
        &*self.0[i].get()
    }
    /// SAFETY: as [`set`](Self::set) - exclusive owner, e.g. for in-place take.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub unsafe fn get_mut(&self, i: usize) -> &mut V {
        &mut *self.0[i].get()
    }
}

impl<V> Cells<V> {
    /// The cells `r` as one slice (`UnsafeCell<V>` is laid out as `V`).
    ///
    /// # Safety
    /// Every write to `r` happened-before this read, and none follows while
    /// the slice lives.
    #[inline]
    pub unsafe fn slice(&self, r: std::ops::Range<usize>) -> &[V] {
        std::slice::from_raw_parts(self.0[r.clone()].as_ptr() as *const V, r.len())
    }
}

/// Scratch the node kernels borrow, one object per running kernel: taken on
/// entry and given back on exit, so the kernels of one factorization share
/// a handful of objects (about one per worker) whose buffers stay grown,
/// instead of allocating their own per supernode. An object heavier than
/// [`SCRATCH_KEEP`](crate::memory::SCRATCH_KEEP) is freed instead of kept.
pub(crate) struct ScratchPool<S>(std::sync::Mutex<Vec<S>>);

impl<S: Default + HeapBytes> ScratchPool<S> {
    pub fn new() -> Self {
        ScratchPool(std::sync::Mutex::new(Vec::new()))
    }

    /// A scratch object for the caller until the guard drops.
    pub fn take(&self) -> Lent<'_, S> {
        let s = self
            .0
            .lock()
            .ok()
            .and_then(|mut v| v.pop())
            .unwrap_or_default();
        Lent {
            pool: self,
            s: std::mem::ManuallyDrop::new(s),
        }
    }
}

/// A [`ScratchPool`] object on loan, returned when dropped.
pub(crate) struct Lent<'p, S: HeapBytes> {
    pool: &'p ScratchPool<S>,
    s: std::mem::ManuallyDrop<S>,
}

impl<S: HeapBytes> std::ops::Deref for Lent<'_, S> {
    type Target = S;
    fn deref(&self) -> &S {
        &self.s
    }
}

impl<S: HeapBytes> std::ops::DerefMut for Lent<'_, S> {
    fn deref_mut(&mut self) -> &mut S {
        &mut self.s
    }
}

impl<S: HeapBytes> Drop for Lent<'_, S> {
    fn drop(&mut self) {
        // SAFETY: the object is taken once, here, and never touched again.
        let s = unsafe { std::mem::ManuallyDrop::take(&mut self.s) };
        if s.heap_bytes() > SCRATCH_KEEP {
            return;
        }
        if let Ok(mut v) = self.pool.0.lock() {
            v.push(s);
        }
    }
}
