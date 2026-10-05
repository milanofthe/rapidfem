//! A DISCRETE surface carrier: a triangle-soup patch (an imported STL's smooth
//! region) queried by closest-point projection. The mesher needs only
//! `closest` and a consistent normal from a carrier; this provides both over an
//! AABB tree, so an imported mesh is remeshed against its own envelope instead
//! of keeping every input facet as a face of its own (every input edge would
//! then be an edge of the B-rep, and the import's slivers would stay).
//!
//! The carrier is tessellation-faithful (the envelope IS the facets), so the
//! remesh converges to the input surface, not to a smoothed version of it;
//! smooth-region grouping and feature (crease) edges are decided by the
//! importer's dihedral threshold, not here.

use crate::bvh::Bvh;
use rapidmesh_exact::vector::{
    bbox, centroid, closest_on_tri, cross, dist2, dot, normalize, sub, V3,
};

/// One smooth soup patch with a closest-point accelerator.
#[derive(Debug)]
pub struct DiscreteSurface {
    /// Patch vertices.
    pub points: Vec<V3>,
    /// Patch triangles (indices into `points`), consistently wound.
    pub tris: Vec<[u32; 3]>,
    /// Unit facet normals, parallel to `tris`.
    normals: Vec<V3>,
    /// Per-facet curvature radius estimate, parallel to `tris` (INFINITY on
    /// flats): the strongest bend over the facet's interior edges, radius =
    /// centroid distance / normal turning angle (the osculating-circle chord
    /// approximation). Feeds the same curvature-driven sizing as the
    /// analytic carriers.
    curv_r: Vec<f64>,
    /// The tree over the triangles.
    bvh: Bvh,
}

impl DiscreteSurface {
    /// Builds the patch accelerator. `tris` must be consistently wound (the
    /// normals give the outward side). The facets are kept in an order of
    /// their own (see [`crate::plc::canonical_mesh`]): a scan read with its
    /// triangles in another order is the same patch.
    pub fn new(points: Vec<V3>, tris: Vec<[u32; 3]>) -> DiscreteSurface {
        let (points, tris, _, _) = crate::plc::canonical_mesh(&points, &tris);
        let normals: Vec<V3> = tris
            .iter()
            .map(|t| {
                let (a, b, c) = (
                    points[t[0] as usize],
                    points[t[1] as usize],
                    points[t[2] as usize],
                );
                normalize(cross(sub(b, a), sub(c, a)))
            })
            .collect();
        let centroids: Vec<V3> = tris
            .iter()
            .map(|t| centroid(t.map(|v| points[v as usize])))
            .collect();
        // Per-facet curvature: for every interior edge (two owners inside
        // this smooth patch) the normals turn by theta over the centroid
        // distance d -- osculating radius ~ d / theta. A facet's radius is
        // its strongest bend; flats (and patch-boundary facets with no
        // interior edge) stay INFINITY. Crease edges never enter: the
        // importer splits patches there, so both owners are same-patch by
        // construction.
        let mut curv_r = vec![f64::INFINITY; tris.len()];
        {
            let mut by_edge: std::collections::HashMap<(u32, u32), [u32; 2]> =
                std::collections::HashMap::new();
            for (i, t) in tris.iter().enumerate() {
                for e in 0..3 {
                    let (a, b) = (t[e], t[(e + 1) % 3]);
                    let key = (a.min(b), a.max(b));
                    let s = by_edge.entry(key).or_insert([u32::MAX; 2]);
                    if s[0] == u32::MAX {
                        s[0] = i as u32;
                    } else {
                        s[1] = i as u32;
                    }
                }
            }
            for s in by_edge.values() {
                if s[1] == u32::MAX {
                    continue; // rim edge
                }
                let (i, j) = (s[0] as usize, s[1] as usize);
                let cosang = dot(normals[i], normals[j]).clamp(-1.0, 1.0);
                let theta = cosang.acos();
                if theta <= 1e-9 {
                    continue;
                }
                let d = dist2(centroids[i], centroids[j]).sqrt();
                let r = d / theta;
                curv_r[i] = curv_r[i].min(r);
                curv_r[j] = curv_r[j].min(r);
            }
            // Resolution floor: curvature below the input tessellation's own
            // edge length is not measurable -- it is normal NOISE of the
            // piecewise-linear envelope, and refining past the input's
            // information content buys no fidelity (measured unbounded:
            // spot x39 tets, fandisk x14). Radii clamp to a multiple of the
            // facet's longest edge (4 ~ the 1%-sagitta chord factor), so
            // genuinely tight input features (finely tessellated fillets,
            // ears) keep their refinement while flat-noise regions do not.
            for (i, t) in tris.iter().enumerate() {
                if !curv_r[i].is_finite() {
                    continue;
                }
                let mut lmax2 = 0.0f64;
                for e in 0..3 {
                    lmax2 = lmax2.max(dist2(
                        points[t[e] as usize],
                        points[t[(e + 1) % 3] as usize],
                    ));
                }
                curv_r[i] = curv_r[i].max(4.0 * lmax2.sqrt());
            }
        }
        let boxes = tris
            .iter()
            .map(|t| bbox(t.map(|v| points[v as usize])))
            .collect();
        DiscreteSurface {
            bvh: Bvh::build(boxes, &centroids),
            points,
            tris,
            normals,
            curv_r,
        }
    }

    /// Closest point on the patch and the (facet) normal there.
    pub fn closest(&self, p: V3) -> (V3, V3) {
        let (q, n, _) = self.closest_facet(p);
        (q, n)
    }

    /// [`DiscreteSurface::closest`] plus the footpoint's facet index.
    pub fn closest_facet(&self, p: V3) -> (V3, V3, usize) {
        self.closest_facet_near(p, usize::MAX)
    }

    /// [`DiscreteSurface::closest_facet`] searched from facet `hint`, one
    /// near the answer: its distance bounds the search from the start, so
    /// only the facets nearer than it are looked at.
    pub fn closest_facet_near(&self, p: V3, hint: usize) -> (V3, V3, usize) {
        let on = |ti: usize| {
            let t = self.tris[ti];
            closest_on_tri(
                p,
                self.points[t[0] as usize],
                self.points[t[1] as usize],
                self.points[t[2] as usize],
            )
        };
        let start = (hint < self.tris.len()).then(|| (hint, on(hint)));
        let limit = start.map_or(f64::INFINITY, |(_, q)| dist2(p, q));
        match self
            .bvh
            .nearest(p, limit, |ti| Some(dist2(p, on(ti as usize))))
            .map(|(ti, _)| (ti as usize, on(ti as usize)))
            .or(start)
        {
            Some((ti, q)) => (q, self.normals[ti], ti),
            None => (p, [0.0, 0.0, 1.0], 0),
        }
    }

    /// Curvature radius estimate at the footpoint of `p` (INFINITY on
    /// flats): the precomputed per-facet osculating radius, queried by the
    /// closest facet. Conservative in the same sense as the analytic
    /// carriers' curvature -- it feeds `h = r * sqrt(8 * tol)` sizing.
    pub fn curvature_radius(&self, p: V3) -> f64 {
        if self.curv_r.is_empty() {
            return f64::INFINITY;
        }
        self.curv_r[self.closest_facet(p).2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_on_a_quad_patch() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let s = DiscreteSurface::new(pts, tris);
        let (foot, n) = s.closest([0.3, 0.4, 0.7]);
        assert!((foot[0] - 0.3).abs() < 1e-12 && (foot[1] - 0.4).abs() < 1e-12);
        assert!(foot[2].abs() < 1e-12);
        assert!((n[2] - 1.0).abs() < 1e-12);
        // off the patch edge: clamps to the border
        let (foot, _) = s.closest([2.0, 0.5, 0.0]);
        assert!((foot[0] - 1.0).abs() < 1e-12);
    }
}
