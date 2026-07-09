/// Inertia of a symmetric matrix: counts of positive, negative, zero eigenvalues.
///
/// This is a plain triple of counts. `total()` returns their sum, which equals
/// the dimension of whatever (sub)matrix the inertia describes. The type is
/// also used for sub-blocks - e.g. the 2×2 pivot classification in
/// `dense::factor` returns inertias with `total() == 2` - so the sum is the
/// described block's order, not necessarily the global matrix order `n`. The
/// counts are caller-supplied and not validated against any dimension; keeping
/// them consistent is the caller's responsibility (see `new`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

impl Inertia {
    /// Create a new Inertia from explicit counts. The counts are stored as
    /// given and are not validated against any matrix dimension; `total()`
    /// will return their sum.
    pub fn new(positive: usize, negative: usize, zero: usize) -> Self {
        Self {
            positive,
            negative,
            zero,
        }
    }

    /// Total dimension: positive + negative + zero.
    pub fn total(&self) -> usize {
        self.positive + self.negative + self.zero
    }
}

impl std::fmt::Display for Inertia {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.positive, self.negative, self.zero)
    }
}
