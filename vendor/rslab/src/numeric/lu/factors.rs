//! The LU factor types: the supernodal panel form of a numeric factorization
//! and the permutations, scalings and counters the solver keeps next to it.

use crate::numeric::supernodal::panel::PanelFactor;
use crate::scalar::Scalar;

/// Everything of `P_r^T A P = L U` except the panels: the permutations, the
/// equilibration and the outcome, `perm[e]` mapping factorization position
/// `e` to the original index.
pub(crate) struct LuPivots {
    pub n: usize,
    /// Column permutation: factorization position -> original column index
    /// (`P^T A P = L U`, the fill-reducing ordering).
    pub perm: Vec<usize>,
    /// Row permutation: factorization position -> original row index. Differs
    /// from `perm` when partial pivoting interchanged rows.
    pub perm_row: Vec<usize>,
    /// Two-sided equilibration: the factor is of `A_hat = diag(d_row)*A*diag(d_col)`.
    /// Solve applies `D_r` to the RHS and `D_c` to the result. Both length `n`.
    pub d_row: Vec<f64>,
    pub d_col: Vec<f64>,
    /// Parent of every supernode in the assembly tree (`usize::MAX` for a
    /// root); empty when unknown.
    pub supernode_parent: Vec<usize>,
    /// Number of statically perturbed pivots.
    pub n_perturbed: usize,
}

/// The numeric result of a sparse LU factorization: the unit lower `L` and
/// the transposed upper factor `U^T` (its diagonal in the panel) in
/// supernodal panel form (the storage the solves run on, written by the
/// drivers without a copy), the permutations, the scalings and the outcome.
pub(crate) struct LuNumeric<T> {
    /// `L` in panel form, in elimination order (unit diagonal implicit).
    pub l: PanelFactor<T>,
    /// `U^T` in panel form: column `c` of the panel is row `c` of `U`, the
    /// diagonal of `U` at the panel's diagonal.
    pub ut: PanelFactor<T>,
    /// `perm[e]` is the original column eliminated at position `e`.
    pub perm: Vec<usize>,
    /// `perm_row[e]` is the original row that became pivot row `e`.
    pub perm_row: Vec<usize>,
    /// Row scaling applied before the factorization.
    pub d_row: Vec<f64>,
    /// Column scaling applied before the factorization.
    pub d_col: Vec<f64>,
    /// Supernode tree over the factor's supernodes (`usize::MAX` for a root).
    pub supernode_parent: Vec<usize>,
    /// Pivots perturbed by the static regularization.
    pub n_perturbed: usize,
    /// Structural panel slots holding an exact zero (the symmetrized
    /// pattern, cancellation or `drop_tol`); the stored nonzeros are
    /// `l.nnz() + ut.nnz() - n_zeros`.
    pub n_zeros: usize,
}

impl<T: Scalar> LuNumeric<T> {
    /// Stored fill `nnz(L) + nnz(U)`: the structural panel entries minus the
    /// slots holding an exact zero.
    pub(crate) fn factor_nnz(&self) -> usize {
        self.l.nnz() + self.ut.nnz() - self.n_zeros
    }

    /// Bytes of the two panel factors.
    pub(crate) fn bytes(&self) -> usize {
        self.l.bytes() + self.ut.bytes()
    }

    /// Split into the two panel factors and the [`LuPivots`] the solver
    /// keeps next to them.
    pub(crate) fn into_parts(self) -> (PanelFactor<T>, PanelFactor<T>, LuPivots) {
        let pivots = LuPivots {
            n: self.l.n,
            perm: self.perm,
            perm_row: self.perm_row,
            d_row: self.d_row,
            d_col: self.d_col,
            supernode_parent: self.supernode_parent,
            n_perturbed: self.n_perturbed,
        };
        (self.l, self.ut, pivots)
    }
}
