//! Diagnostic counters exposed alongside the permutation.

/// Diagnostic counters collected during AMF ordering.
///
/// Mirrors [`rslab_amd::AmdStats`] field-by-field; the AMF inner
/// loop populates the same `OrderDiagnostics` surface from the
/// shared `rslab-ordering-core` workspace.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AmfStats {
    /// Number of garbage-collection compactions fired.
    pub ncmpa: u32,
    /// Number of mark-array generation-counter resets.
    ///
    /// Currently always `0`: not wired to a backing counter. The reset
    /// it would count (`clear_flag`) only fires when the generation
    /// counter `wflg` reaches `wbig = i32::MAX - n`, which during
    /// elimination requires `n` on the order of tens of thousands, so
    /// the true count is `0` on every practically testable input.
    pub n_clear_flag: u32,
    /// Number of variables absorbed by mass elimination.
    pub n_mass_elim: u32,
    /// Number of supervariable merges detected.
    pub n_supervar_merge: u32,
    /// Number of variables placed into the dense-deferred bucket
    /// at initialization.
    pub n_dense_deferred: u32,
    /// Flop counter: divisions.
    pub ndiv: u64,
    /// Flop counter: LU multiply-subtracts.
    pub nms_lu: u64,
    /// Flop counter: LDLᵀ multiply-subtracts.
    pub nms_ldl: u64,
}
