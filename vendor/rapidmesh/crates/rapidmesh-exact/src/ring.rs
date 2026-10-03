//! The number abstraction shared by filter, exact, and oracle evaluation.

/// Exact ring operations over numbers constructed from f64 inputs.
///
/// Geometric expressions (determinants, implicit-point coordinates) are
/// written once against this trait and evaluated with [`crate::Interval`]
/// (fast conservative filter), [`crate::Expansion`] (exact fallback), or a
/// rational type in tests (oracle). Implementations must be exact in the
/// algebraic sense appropriate to the type: `Expansion` represents the value
/// exactly, `Interval` must always contain it.
pub trait Ring: Clone {
    /// Lifts an exact f64 input into the ring.
    fn from_f64(v: f64) -> Self;
    /// Sum.
    fn add(&self, other: &Self) -> Self;
    /// Difference.
    fn sub(&self, other: &Self) -> Self;
    /// Product.
    fn mul(&self, other: &Self) -> Self;
    /// Negation.
    fn neg(&self) -> Self;
}
