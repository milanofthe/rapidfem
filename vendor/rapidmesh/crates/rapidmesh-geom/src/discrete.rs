//! A DISCRETE surface carrier: a triangle-soup patch (an imported STL's smooth
//! region) queried by closest-point projection. The refinement path needs only
//! `closest` + a consistent normal from a carrier -- this provides both over an
//! AABB tree, so an imported mesh can be REMESHED against its own envelope
//! instead of being frozen facet by facet (each input facet as its own plane
//! face makes every input edge a protected feature: the import is then pinned,
//! and its slivers are permanent).
//!
//! The carrier is tessellation-faithful (the envelope IS the facets), so the
//! remesh converges to the input surface, not to a smoothed version of it;
//! smooth-region grouping and feature (crease) edges are decided by the
//! importer's dihedral threshold, not here.

use crate::vec3::{cross, dot, normalize, sub, V3};

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
    /// Flat AABB tree (node = (lo, hi, left, right | leaf tri range)).
    nodes: Vec<Node>,
    /// Triangle order for leaf ranges.
    order: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct Node {
    lo: V3,
    hi: V3,
    /// Child indices, or `(start, !count)` leaf encoding when `right < 0`.
    left: i32,
    right: i32,
}

/// Closest point on triangle `(a, b, c)` to `p`.
fn closest_on_tri(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return std::array::from_fn(|k| a[k] + v * ab[k]);
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return std::array::from_fn(|k| a[k] + w * ac[k]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return std::array::from_fn(|k| b[k] + w * (c[k] - b[k]));
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    std::array::from_fn(|k| a[k] + ab[k] * v + ac[k] * w)
}

fn d2(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

fn box_d2(lo: V3, hi: V3, p: V3) -> f64 {
    let mut s = 0.0;
    for k in 0..3 {
        let d = (lo[k] - p[k]).max(0.0).max(p[k] - hi[k]);
        s += d * d;
    }
    s
}

impl DiscreteSurface {
    /// Builds the patch accelerator. `tris` must be consistently wound (the
    /// normals give the signed side of [`DiscreteSurface::signed_offset`]).
    pub fn new(points: Vec<V3>, tris: Vec<[u32; 3]>) -> DiscreteSurface {
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
            .map(|t| {
                std::array::from_fn(|k| {
                    (points[t[0] as usize][k] + points[t[1] as usize][k] + points[t[2] as usize][k])
                        / 3.0
                })
            })
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
                let d = d2(centroids[i], centroids[j]).sqrt();
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
                    lmax2 = lmax2.max(d2(points[t[e] as usize], points[t[(e + 1) % 3] as usize]));
                }
                curv_r[i] = curv_r[i].max(4.0 * lmax2.sqrt());
            }
        }
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let mut nodes = Vec::new();
        if !tris.is_empty() {
            build(
                &points,
                &tris,
                &centroids,
                &mut order,
                0,
                tris.len(),
                &mut nodes,
            );
        }
        DiscreteSurface {
            points,
            tris,
            normals,
            curv_r,
            nodes,
            order,
        }
    }

    /// Closest point on the patch and the (facet) normal there.
    pub fn closest(&self, p: V3) -> (V3, V3) {
        let (q, n, _) = self.closest_facet(p);
        (q, n)
    }

    /// [`DiscreteSurface::closest`] plus the footpoint's facet index.
    pub fn closest_facet(&self, p: V3) -> (V3, V3, usize) {
        if self.nodes.is_empty() {
            return (p, [0.0, 0.0, 1.0], 0);
        }
        let mut best = (f64::INFINITY, p, [0.0, 0.0, 1.0], 0usize);
        self.closest_rec(0, p, &mut best);
        (best.1, best.2, best.3)
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

    fn closest_rec(&self, ni: usize, p: V3, best: &mut (f64, V3, V3, usize)) {
        let n = self.nodes[ni];
        if box_d2(n.lo, n.hi, p) >= best.0 {
            return;
        }
        if n.right < 0 {
            let (start, count) = (n.left as usize, (!n.right) as usize);
            for &ti in &self.order[start..start + count] {
                let t = self.tris[ti as usize];
                let q = closest_on_tri(
                    p,
                    self.points[t[0] as usize],
                    self.points[t[1] as usize],
                    self.points[t[2] as usize],
                );
                let dd = d2(p, q);
                if dd < best.0 {
                    *best = (dd, q, self.normals[ti as usize], ti as usize);
                }
            }
            return;
        }
        let (l, r) = (n.left as usize, n.right as usize);
        let (dl, dr) = (
            box_d2(self.nodes[l].lo, self.nodes[l].hi, p),
            box_d2(self.nodes[r].lo, self.nodes[r].hi, p),
        );
        if dl <= dr {
            self.closest_rec(l, p, best);
            self.closest_rec(r, p, best);
        } else {
            self.closest_rec(r, p, best);
            self.closest_rec(l, p, best);
        }
    }

    /// Signed offset of `p`: distance along the facet normal at the footpoint
    /// (positive on the outward side). The sign flips exactly at the surface --
    /// the sphere-tracing / crossing-bisection contract of the refinement path.
    pub fn signed_offset(&self, p: V3) -> f64 {
        let (foot, n) = self.closest(p);
        dot(sub(p, foot), n)
    }
}

fn build(
    points: &[V3],
    tris: &[[u32; 3]],
    centroids: &[V3],
    order: &mut [u32],
    start: usize,
    end: usize,
    nodes: &mut Vec<Node>,
) -> i32 {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for &ti in &order[start..end] {
        for &v in &tris[ti as usize] {
            let p = points[v as usize];
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let idx = nodes.len() as i32;
    nodes.push(Node {
        lo,
        hi,
        left: 0,
        right: 0,
    });
    let count = end - start;
    if count <= 8 {
        nodes[idx as usize].left = start as i32;
        nodes[idx as usize].right = !(count as i32);
        return idx;
    }
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).partial_cmp(&(hi[b] - lo[b])).unwrap())
        .unwrap();
    order[start..end].sort_unstable_by(|&a, &b| {
        centroids[a as usize][axis]
            .partial_cmp(&centroids[b as usize][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + count / 2;
    let l = build(points, tris, centroids, order, start, mid, nodes);
    let r = build(points, tris, centroids, order, mid, end, nodes);
    nodes[idx as usize].left = l;
    nodes[idx as usize].right = r;
    idx
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
        assert!((s.signed_offset([0.3, 0.4, 0.7]) - 0.7).abs() < 1e-12);
        assert!((s.signed_offset([0.3, 0.4, -0.2]) + 0.2).abs() < 1e-12);
        // off the patch edge: clamps to the border
        let (foot, _) = s.closest([2.0, 0.5, 0.0]);
        assert!((foot[0] - 1.0).abs() < 1e-12);
    }
}
