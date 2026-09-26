//! The settings of the Krylov solvers.

/// Settings of [`gmres`](super::gmres), [`gmres_block`](super::gmres_block),
/// [`gmres_recycled`](super::gmres_recycled), [`cocg`](super::cocg) and
/// [`cocr`](super::cocr).
///
/// ```
/// use rslab::KrylovSettings;
/// let s = KrylovSettings::default().with_tol(1e-10).with_restart(60);
/// assert_eq!(s.restart, Some(60));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct KrylovSettings {
    /// Target of the relative residual `||b - A x|| / ||b||`, per column.
    /// Default `1e-8`.
    pub tol: f64,
    /// Iteration budget per solve. Default `400`.
    pub max_iter: usize,
    /// GMRES restart length (ignored by COCG and COCR). `None` (default)
    /// takes the largest length in `restart_min..=restart_max` whose Krylov
    /// bases fit [`basis_budget_bytes`](Self::basis_budget_bytes).
    pub restart: Option<usize>,
    /// Memory cap of the Krylov bases for the default restart length.
    /// Default 1 GiB.
    pub basis_budget_bytes: usize,
    /// Shortest default restart length. Default `20`.
    pub restart_min: usize,
    /// Longest default restart length. Default `80`.
    pub restart_max: usize,
    /// A new basis vector is orthogonalized a second time when the first
    /// pass keeps less than this fraction of its norm (the DGKS criterion).
    /// Default `1/sqrt(2)`.
    pub reorth_eta: f64,
    /// Rows per chunk of the parallel orthogonalization sums of block GMRES.
    /// The chunks are summed in a fixed order, so the result depends on this
    /// value but not on the thread count. Default `2048`.
    pub ortho_chunk: usize,
}

impl Default for KrylovSettings {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 400,
            restart: None,
            basis_budget_bytes: 1 << 30,
            restart_min: 20,
            restart_max: 80,
            reorth_eta: std::f64::consts::FRAC_1_SQRT_2,
            ortho_chunk: 2048,
        }
    }
}

impl KrylovSettings {
    /// Set the residual target.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the iteration budget.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fix the GMRES restart length.
    pub fn with_restart(mut self, restart: usize) -> Self {
        self.restart = Some(restart);
        self
    }

    /// The restart length for `n` unknowns, `columns` right-hand sides of
    /// `scalar_bytes` each and `bases` stored bases (two for flexible GMRES).
    pub(crate) fn restart_for(
        &self,
        n: usize,
        columns: usize,
        scalar_bytes: usize,
        bases: usize,
    ) -> usize {
        if let Some(r) = self.restart {
            return r.max(1);
        }
        let per_vector = n
            .saturating_mul(columns)
            .saturating_mul(scalar_bytes)
            .saturating_mul(bases);
        let max = self.restart_max.max(1);
        if per_vector == 0 {
            return max;
        }
        (self.basis_budget_bytes / per_vector)
            .saturating_sub(1)
            .clamp(self.restart_min.clamp(1, max), max)
    }
}

#[cfg(test)]
mod tests {
    use super::KrylovSettings;

    /// The default restart is the longest whose bases fit the budget,
    /// within the bounds; an explicit one is kept.
    #[test]
    fn the_default_restart_fits_the_basis_budget() {
        let s = KrylovSettings::default();
        assert_eq!(s.restart_for(1000, 1, 16, 2), 80);
        // 32 MB per basis vector: 33 fit into 1 GiB, one is the spare.
        assert_eq!(s.restart_for(500_000, 2, 16, 2), 32);
        assert_eq!(s.restart_for(50_000_000, 1, 16, 2), 20);
        assert_eq!(
            s.clone().with_restart(7).restart_for(50_000_000, 1, 16, 2),
            7
        );
    }
}
