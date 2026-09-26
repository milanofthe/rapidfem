//! Per-supernode storage of the left-looking factorizations: the slot store
//! written once by a node's owner and read by its ancestors, the per-index
//! cells of the emit state, and the raw panel pointer for disjoint parallel
//! writes.

/// One `UnsafeCell` payload per supernode, written exactly once by the
/// supernode's owner and read only by nodes that are (transitively) its
/// assembly-tree ancestors - the single-writer-before-readers discipline the
/// left-looking schedule guarantees. `free` resets a slot once its last
/// consumer is done.
pub(crate) struct SlotStore<P> {
    slots: Vec<std::cell::UnsafeCell<P>>,
}

// SAFETY: single-writer-before-readers, disjoint indices (see the type doc).
unsafe impl<P: Send> Sync for SlotStore<P> {}

impl<P: Default> SlotStore<P> {
    pub fn new(nsuper: usize) -> Self {
        SlotStore {
            slots: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(P::default()))
                .collect(),
        }
    }

    /// SAFETY: `k` must be a fully-factored descendant of the current node
    /// (its owner's write happened-before this read).
    pub unsafe fn get(&self, k: usize) -> &P {
        &*self.slots[k].get()
    }

    /// SAFETY: only the owner of supernode `s` calls this, exactly once.
    pub unsafe fn set(&self, s: usize, p: P) {
        *self.slots[s].get() = p;
    }

    /// Move the panel out, leaving the default in its place. SAFETY: the
    /// owner of supernode `k`, after its last reader is done.
    pub unsafe fn take(&self, k: usize) -> P {
        std::mem::take(&mut *self.slots[k].get())
    }
}

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
