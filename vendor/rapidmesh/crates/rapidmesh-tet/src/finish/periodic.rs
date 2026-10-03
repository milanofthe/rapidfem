//! The matching of the points of the two sides of a periodic pair (see
//! [`crate::params::PeriodicPair`] and `crate::surface::periodic`).

use crate::finish::P3;
use rapidmesh_geom::grid::HashGrid;
use rapidmesh_geom::vec3::dist;

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

    pub fn insert(&mut self, p: P3, id: usize) {
        self.grid.insert(p, id);
    }

    /// The first point within `tol` of `p` (`pos` gives the points).
    pub fn find(&self, p: P3, pos: &dyn Fn(usize) -> P3, tol: f64) -> Option<usize> {
        self.grid
            .around(self.grid.key(p), 1)
            .copied()
            .find(|&i| dist(pos(i), p) <= tol)
    }
}
