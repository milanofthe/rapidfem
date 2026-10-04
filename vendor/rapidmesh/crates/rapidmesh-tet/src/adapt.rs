//! gmsh-style Dörfler-marking adaptive refinement for surface meshes: the
//! MARK -> REFINE half of the adaptive loop. Given per-triangle error
//! indicators, Dörfler-mark the bulk and turn the marked elements into point
//! size sources, which the mesher's grading + Ruppert refinement realise
//! sliver-free. SOLVE and ESTIMATE belong to the solver.

use crate::mesh::SurfaceMesh;
use rapidmesh_geom::vec3::{len, sub};

/// The knobs of Dörfler marking and the refinement it asks for.
#[derive(Clone, Copy, Debug)]
pub struct Dorfler {
    /// The share of the summed squared indicator the marked set reaches.
    pub theta: f64,
    /// A marked element's size is its local size over this.
    pub factor: f64,
    /// No marked size below this (0 off).
    pub h_min: f64,
}

impl Default for Dorfler {
    fn default() -> Dorfler {
        Dorfler {
            theta: 0.5,
            factor: 2.0,
            h_min: 0.0,
        }
    }
}

/// Dörfler (bulk) marking: the indices of the smallest element set whose summed
/// SQUARED indicator reaches `theta` of the total (`theta` in `(0, 1]`; 0.5 is
/// the common choice). The squared convention treats `eta` as an energy-norm
/// error contribution per element. Returns the marked indices in ascending order.
pub fn dorfler_mark(eta: &[f64], theta: f64) -> Vec<u32> {
    let e2: Vec<f64> = eta.iter().map(|e| e * e).collect();
    let total: f64 = e2.iter().sum();
    if total <= 0.0 || eta.is_empty() {
        return Vec::new();
    }
    // Indices by descending indicator; ties keep descending index order to match
    // a `argsort(e2)[::-1]` (ascending-stable, then reversed) convention.
    let mut order: Vec<u32> = (0..eta.len() as u32).collect();
    order.sort_by(|&a, &b| e2[b as usize].total_cmp(&e2[a as usize]).then(b.cmp(&a)));
    // Smallest prefix whose cumulative squared indicator reaches `theta * total`.
    let target = theta * total;
    let mut cum = 0.0;
    let mut k = order.len();
    for (i, &idx) in order.iter().enumerate() {
        cum += e2[idx as usize];
        if cum >= target {
            k = i + 1;
            break;
        }
    }
    let mut marked: Vec<u32> = order[..k].to_vec();
    marked.sort_unstable();
    marked
}

impl SurfaceMesh {
    /// Mean edge length per triangle (the local element size), parallel to `faces`.
    pub fn tri_local_h(&self) -> Vec<f64> {
        let p = &self.points;
        self.faces
            .iter()
            .map(|f| {
                let (a, b, c) = (p[f.tri[0]], p[f.tri[1]], p[f.tri[2]]);
                (len(sub(b, a)) + len(sub(c, b)) + len(sub(a, c))) / 3.0
            })
            .collect()
    }

    /// Dörfler-mark by `eta` and turn the marked elements into point size sources:
    /// returns the marked indices and parallel `(centroid, h)` arrays, with
    /// `h = local_h / factor` (clamped to `h_min` when `h_min > 0`). Feed the
    /// `(centroid, h)` pairs into a size field (`Geometry::add_size_points`)
    /// and remesh.
    pub fn dorfler_size_points(
        &self,
        eta: &[f64],
        theta: f64,
        factor: f64,
        h_min: f64,
    ) -> (Vec<u32>, Vec<[f64; 3]>, Vec<f64>) {
        let marked = dorfler_mark(eta, theta);
        let hloc = self.tri_local_h();
        let p = &self.points;
        let mut centroids = Vec::with_capacity(marked.len());
        let mut hs = Vec::with_capacity(marked.len());
        for &mi in &marked {
            let f = self.faces[mi as usize].tri;
            centroids.push([
                (p[f[0]][0] + p[f[1]][0] + p[f[2]][0]) / 3.0,
                (p[f[0]][1] + p[f[1]][1] + p[f[2]][1]) / 3.0,
                (p[f[0]][2] + p[f[1]][2] + p[f[2]][2]) / 3.0,
            ]);
            let mut h = hloc[mi as usize] / factor;
            if h_min > 0.0 {
                h = h.max(h_min);
            }
            hs.push(h);
        }
        (marked, centroids, hs)
    }
}

#[cfg(test)]
mod tests {
    use super::dorfler_mark;

    #[test]
    fn marks_the_bulk() {
        // One dominant element (10) and three small ones: theta=0.5 of the total
        // squared error (100 + 3 = 103, half = 51.5) is reached by element 0 alone.
        let eta = [10.0, 1.0, 1.0, 1.0];
        assert_eq!(dorfler_mark(&eta, 0.5), vec![0]);
        // theta=1.0 needs every element.
        assert_eq!(dorfler_mark(&eta, 1.0), vec![0, 1, 2, 3]);
        // All-zero indicators mark nothing.
        assert!(dorfler_mark(&[0.0, 0.0], 0.5).is_empty());
    }
}
