/// Errors returned by RSLAB's public API.
#[derive(Debug)]
pub enum RslabError {
    /// The matrix is numerically rank-deficient: a pivot was exactly or
    /// near-zero and `ZeroPivotAction::Fail` was specified. The factorization
    /// is incomplete.
    NumericallyRankDeficient,

    /// Input matrix dimensions are inconsistent or the matrix is not square.
    InvalidInput(String),

    /// The RHS vector length does not match the factored matrix dimension.
    DimensionMismatch { expected: usize, got: usize },

    /// An I/O or parse error occurred (e.g. reading a Matrix Market file).
    IoError(String),

    /// A supernode received more delayed pivots from its children at
    /// numeric time than the symbolic-analysis phase budgeted for.
    /// Mirrors MUMPS's `INFO(2)` workspace-overflow path: a predictable,
    /// recoverable failure that bounds worst-case front growth.
    /// See issue #55 and `dev/research/symbolic-delay-budget-2026-05-27.md`.
    DelayBudgetExceeded {
        supernode: usize,
        required: usize,
        capacity: usize,
    },

    /// The KLU path found a numerically singular column: no pivot candidate
    /// in the column's reach had a nonzero finite magnitude at factor time,
    /// or a frozen pivot came up zero during a numeric-only
    /// [`KluSolver::refactor`](crate::KluSolver::refactor). `column` is the
    /// **original** column index, so the caller can map it back to its model.
    /// After a refactor failure, re-factor with pivoting
    /// ([`KluSymbolic::factor`](crate::KluSymbolic::factor)).
    SingularBasis { column: usize },

    /// The matrix is structurally singular: no complete matching of columns
    /// onto rows with structural nonzeros exists, so the matrix is singular
    /// for *every* value assignment. Detected by the KLU path's maximum
    /// transversal (issue #15) before any numeric work.
    StructurallySingular,
}

impl std::fmt::Display for RslabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RslabError::NumericallyRankDeficient => {
                write!(f, "matrix is numerically rank-deficient")
            }
            RslabError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            RslabError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            RslabError::IoError(msg) => write!(f, "I/O error: {}", msg),
            RslabError::DelayBudgetExceeded {
                supernode,
                required,
                capacity,
            } => {
                write!(
                    f,
                    "delayed-pivot budget exceeded at supernode {}: \
                     required {} delayed columns, capacity {} (issue #55)",
                    supernode, required, capacity
                )
            }
            RslabError::SingularBasis { column } => {
                write!(f, "numerically singular at column {}", column)
            }
            RslabError::StructurallySingular => {
                write!(
                    f,
                    "matrix is structurally singular (incomplete column-row matching)"
                )
            }
        }
    }
}

impl std::error::Error for RslabError {}
