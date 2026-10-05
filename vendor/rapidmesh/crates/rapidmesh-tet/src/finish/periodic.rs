//! The matching of the points of the two sides of a periodic pair (see
//! [`crate::params::PeriodicPair`] and `crate::surface::periodic`).

use rapidmesh_exact::vector::dist;
use rapidmesh_exact::vector::V3;
use rapidmesh_geom::grid::HashGrid;

/// Points by position, for matching the two sides of a pair.
pub(crate) struct PointIndex {
    grid: HashGrid<usize>,
}

impl PointIndex {
    pub fn new(tol: f64) -> PointIndex {
        PointIndex {
            grid: HashGrid::new(4.0 * tol),
        }
    }

    pub fn insert(&mut self, p: V3, id: usize) {
        self.grid.insert(p, id);
    }

    /// The first point within `tol` of `p` (`pos` gives the points).
    pub fn find(&self, p: V3, pos: &dyn Fn(usize) -> V3, tol: f64) -> Option<usize> {
        self.grid
            .around(self.grid.key(p), 1)
            .copied()
            .find(|&i| dist(pos(i), p) <= tol)
    }
}
