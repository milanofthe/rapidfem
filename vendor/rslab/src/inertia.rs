/// Inertia of a symmetric matrix: counts of positive, negative, zero eigenvalues.
///
/// A plain triple of counts; `total()` is their sum, the order of the
/// (sub)matrix the inertia describes.
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
