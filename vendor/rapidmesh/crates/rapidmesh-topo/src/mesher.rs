//! Source adapters for the rapidmesh mesher's own output types, behind the
//! `mesher` feature. The core stays std-only and builds for any external mesh;
//! this wires our `TetMesh` / `SurfaceMesh` into the same complex builders.

use crate::source::{TetSource, TriSource};
use rapidmesh_brep::Surface;
use rapidmesh_geom::SurfaceKind;
use rapidmesh_tet::{SurfaceFace, SurfaceMesh, TetMesh};

impl TetSource for TetMesh {
    fn n_verts(&self) -> usize {
        self.points.len()
    }
    fn n_tets(&self) -> usize {
        self.tets.len()
    }
    fn tet(&self, i: usize) -> [u32; 4] {
        let t = self.tets[i];
        [t[0] as u32, t[1] as u32, t[2] as u32, t[3] as u32]
    }
}

impl TriSource for SurfaceMesh {
    fn n_verts(&self) -> usize {
        self.points.len()
    }
    fn n_tris(&self) -> usize {
        self.faces.len()
    }
    fn tri(&self, i: usize) -> [u32; 3] {
        let t = self.faces[i].tri;
        [t[0] as u32, t[1] as u32, t[2] as u32]
    }
    fn tri_tag(&self, i: usize) -> i64 {
        self.faces[i].face_tag.0 as i64
    }
}

/// The boundary surface of a tet mesh as a triangle source (its tagged faces).
impl TriSource for TetMesh {
    fn n_verts(&self) -> usize {
        self.points.len()
    }
    fn n_tris(&self) -> usize {
        self.faces.len()
    }
    fn tri(&self, i: usize) -> [u32; 3] {
        let t = self.faces[i].tri;
        [t[0] as u32, t[1] as u32, t[2] as u32]
    }
    fn tri_tag(&self, i: usize) -> i64 {
        self.faces[i].face_tag.0 as i64
    }
}

// ---- analytic boundary geometry (what a `.msh` cannot carry) ----------------
//
// Where a boundary face lies on an analytic `SurfaceKind`, its *exact* normal
// and curvature are known in closed form (the B-rep carrier) -- the substrate
// for curved / higher-order elements. A `Plane` needs no override (the facet
// normal is already exact).

/// Exact unit normal direction of an analytic surface at a point on it. The
/// direction is geometric (outward of the carrier); callers align the sign to
/// the facet winding (see [`exact_face_normals`]). `None` for `Plane`.
pub fn surface_normal(kind: &SurfaceKind, p: [f64; 3]) -> Option<[f64; 3]> {
    Surface::curved(kind).map(|s| s.closest(p).1)
}

/// Principal curvatures `[k_max, k_min]` (magnitudes, 1/length) of an analytic
/// surface at a point. `None` for `Plane` (zero), a discrete patch and a cone
/// apex.
pub fn surface_curvature(kind: &SurfaceKind, p: [f64; 3]) -> Option<[f64; 2]> {
    Surface::curved(kind)?.principal_curvatures(p)
}

/// Exact per-face outward unit normals for the boundary faces on a curved
/// analytic surface, evaluated at the face centroid and sign-aligned to the
/// facet winding (so they match the mesh's outward orientation). `None` where the
/// face is planar (facet normal already exact) or the kind has no closed form.
/// Parallel to `faces`. Works for both `TetMesh` and `SurfaceMesh` (pass their
/// `points`, `faces`, `surfaces`).
pub fn exact_face_normals(
    points: &[[f64; 3]],
    faces: &[SurfaceFace],
    surfaces: &[SurfaceKind],
) -> Vec<Option<[f64; 3]>> {
    use crate::math::{add, cross, dot, scale, sub};
    faces
        .iter()
        .map(|f| {
            let kind = &surfaces[f.surface as usize];
            let [ia, ib, ic] = f.tri;
            let (a, b, c) = (points[ia], points[ib], points[ic]);
            let centroid = scale(add(add(a, b), c), 1.0 / 3.0);
            surface_normal(kind, centroid).map(|n| {
                let facet = cross(sub(b, a), sub(c, a));
                if dot(n, facet) < 0.0 {
                    scale(n, -1.0)
                } else {
                    n
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_normal_and_curvature() {
        let k = SurfaceKind::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 2.0,
        };
        let n = surface_normal(&k, [2.0, 0.0, 0.0]).unwrap();
        assert!((n[0] - 1.0).abs() < 1e-12 && n[1].abs() < 1e-12 && n[2].abs() < 1e-12);
        assert_eq!(surface_curvature(&k, [2.0, 0.0, 0.0]), Some([0.5, 0.5]));
    }

    #[test]
    fn cylinder_normal_radial() {
        let k = SurfaceKind::Cylinder {
            center: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            radius: 3.0,
        };
        let n = surface_normal(&k, [0.0, 3.0, 5.0]).unwrap();
        assert!(n[0].abs() < 1e-12 && (n[1] - 1.0).abs() < 1e-12 && n[2].abs() < 1e-12);
    }

    #[test]
    fn plane_has_no_override() {
        assert!(surface_normal(&SurfaceKind::Plane, [1.0, 2.0, 3.0]).is_none());
        assert!(surface_curvature(&SurfaceKind::Plane, [1.0, 2.0, 3.0]).is_none());
    }
}
