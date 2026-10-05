//! Triangle complex: dimension-uniform topology (identical for planar MoM and
//! embedded surface meshes) plus coordinate-aware geometry.

use crate::convention::{canonical_edge, NONE, TRI_EDGE_LOCAL};
use crate::csr::Csr;
use crate::source::TriSource;
use std::collections::HashMap;

/// Derived connectivity of a triangle mesh. Pure topology -- no coordinates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TriTopology {
    pub n_verts: usize,
    /// The triangle -> vertex connectivity (the source elements, as `u32`), so the
    /// complex is self-contained and the geometry builders need only coordinates.
    pub tris: Vec<[u32; 3]>,
    /// Unique edges, canonical `(min, max)`.
    pub edges: Vec<[u32; 2]>,
    /// Global edge id per local edge of each triangle (`TRI_EDGE_LOCAL` order).
    pub tri_edges: Vec<[u32; 3]>,
    /// `+1` if the local edge runs min->max (matches canonical), else `-1`.
    pub tri_edge_sign: Vec<[i8; 3]>,
    /// The tag of every triangle (see [`TriSource::tri_tag`]).
    pub tri_tags: Vec<i64>,
    /// Edge -> every incident triangle, ascending: two on a manifold edge,
    /// one on a free edge, three or more at a junction of sheets.
    pub edge_tris_all: Csr,
    /// The first two triangles incident to each edge; `NONE` fills a free
    /// slot. At a junction the others are in `edge_tris_all`.
    pub edge_tris: Vec<[u32; 2]>,
    /// The tag of each incident triangle (parallel to `edge_tris`); `i64::MIN`
    /// for a free slot. Lets a MoM build pick interior-same-tag (RWG) or
    /// boundary/tag-change edges without re-walking the mesh.
    pub edge_tags: Vec<[i64; 2]>,
    /// Vertex -> incident triangles.
    pub vert_tris: Csr,
}

impl TriTopology {
    /// Build the complex in one O(n) pass.
    pub fn build(src: &impl TriSource) -> Self {
        let nt = src.n_tris();
        let mut edge_id: HashMap<[u32; 2], u32> = HashMap::new();
        let mut edges: Vec<[u32; 2]> = Vec::new();
        let mut tris = vec![[0u32; 3]; nt];
        let mut tri_edges = vec![[0u32; 3]; nt];
        let mut tri_edge_sign = vec![[0i8; 3]; nt];
        let mut vt_pairs: Vec<(u32, u32)> = Vec::with_capacity(nt * 3);

        for t in 0..nt {
            let tri = src.tri(t);
            tris[t] = tri;
            for &v in &tri {
                vt_pairs.push((v, t as u32));
            }
            for (k, &[la, lb]) in TRI_EDGE_LOCAL.iter().enumerate() {
                let (e, sign) = canonical_edge(tri[la], tri[lb]);
                let id = *edge_id.entry(e).or_insert_with(|| {
                    edges.push(e);
                    (edges.len() - 1) as u32
                });
                tri_edges[t][k] = id;
                tri_edge_sign[t][k] = sign;
            }
        }

        let ne = edges.len();
        let mut edge_tris = vec![[NONE; 2]; ne];
        let mut edge_tags = vec![[i64::MIN; 2]; ne];
        let mut cnt = vec![0u8; ne];
        for t in 0..nt {
            let tag = src.tri_tag(t);
            for k in 0..3 {
                let e = tri_edges[t][k] as usize;
                let c = cnt[e];
                if c < 2 {
                    edge_tris[e][c as usize] = t as u32;
                    edge_tags[e][c as usize] = tag;
                }
                cnt[e] = c.saturating_add(1);
            }
        }

        let tri_tags: Vec<i64> = (0..nt).map(|t| src.tri_tag(t)).collect();
        let et_pairs: Vec<(u32, u32)> = (0..nt)
            .flat_map(|t| tri_edges[t].map(|e| (e, t as u32)))
            .collect();
        let edge_tris_all = Csr::from_pairs(ne, &et_pairs);
        let vert_tris = Csr::from_pairs(src.n_verts(), &vt_pairs);
        TriTopology {
            n_verts: src.n_verts(),
            tris,
            edges,
            tri_edges,
            tri_edge_sign,
            tri_tags,
            edge_tris_all,
            edge_tris,
            edge_tags,
            vert_tris,
        }
    }

    /// The tags of every triangle at edge `e`, parallel to its row of
    /// `edge_tris_all`.
    fn tags_at(&self, e: usize, tags: &[i64]) -> Vec<i64> {
        self.edge_tris_all
            .row(e)
            .iter()
            .map(|&t| tags[t as usize])
            .collect()
    }

    /// RWG basis functions, as `[v0, v1, tri_plus, tri_minus]` (the edge's
    /// canonical vertices). An edge with `k` triangles of one tag carries
    /// `k - 1` functions, each from the lowest of them to another (the
    /// junction basis of a T of sheets); a manifold edge carries one. With
    /// `connect_tags` the triangles of all tags at an edge form one group
    /// (conductors of several tags joined there); without, a tag change
    /// separates them. `tags` holds a tag per triangle (`tri_tags`, or the
    /// caller's own grouping).
    pub fn rwg_edges(&self, tags: &[i64], connect_tags: bool) -> Vec<[u32; 4]> {
        let mut out = Vec::new();
        for e in 0..self.edges.len() {
            let ts = self.edge_tris_all.row(e);
            if ts.len() < 2 {
                continue;
            }
            let tg = self.tags_at(e, tags);
            let [a, b] = self.edges[e];
            let mut done = vec![false; ts.len()];
            for i in 0..ts.len() {
                if done[i] {
                    continue;
                }
                done[i] = true;
                for j in i + 1..ts.len() {
                    if !done[j] && (connect_tags || tg[j] == tg[i]) {
                        done[j] = true;
                        out.push([a, b, ts[i], ts[j]]);
                    }
                }
            }
        }
        out
    }

    /// RWG candidates without joining tags ([`TriTopology::rwg_edges`] on
    /// `tri_tags`).
    pub fn rwg_candidate_edges(&self) -> Vec<[u32; 4]> {
        self.rwg_edges(&self.tri_tags, false)
    }

    /// Conductor outline: edges with one triangle or with triangles of
    /// more than one tag, as `[v0, v1, tri]` (its lowest triangle). Pure
    /// topology.
    pub fn boundary_edges(&self) -> Vec<[u32; 3]> {
        let tags = &self.tri_tags;
        (0..self.edges.len())
            .filter(|&e| {
                let tg = self.tags_at(e, tags);
                tg.len() == 1 || tg.iter().any(|&t| t != tg[0])
            })
            .map(|e| {
                let [a, b] = self.edges[e];
                [a, b, self.edge_tris_all.row(e)[0]]
            })
            .collect()
    }
}

/// Per-element geometry of a triangle mesh. Coordinate-aware: planar (MoM) via
/// [`build_2d`](TriGeometry::build_2d), 3D-embedded surface via
/// [`build_3d`](TriGeometry::build_3d). All quantities are basis-free facts about
/// the mesh embedding (no reference element, no discretization).
#[derive(Debug, Clone, Default)]
pub struct TriGeometry {
    /// Unsigned triangle area.
    pub area: Vec<f64>,
    /// Triangle centroid (planar: `z = 0`).
    pub centroid: Vec<[f64; 3]>,
    /// Unit face normal. Planar: `[0, 0, +-1]` (sign of the signed area). 3D: the
    /// unit normal of the stored winding.
    pub normal: Vec<[f64; 3]>,
    /// Second area moment about the centroid `[int dx^2, int dx*dy, int dy^2]` (the multipole
    /// MoM moment). Populated by `build_2d` only; empty for 3D surfaces (the
    /// in-plane moment has no global frame there).
    pub inertia: Vec<[f64; 3]>,
    /// Per-edge length (parallel to `TriTopology::edges`).
    pub edge_len: Vec<f64>,
    /// Per-edge midpoint (planar: `z = 0`).
    pub edge_mid: Vec<[f64; 3]>,
    /// Per-triangle minimum interior angle in degrees (the element-quality field;
    /// a meshed surface has all `>=` the Ruppert bound). Parallel to `tris`.
    pub min_angle: Vec<f64>,
    /// Per-triangle gradients of the barycentric coordinates of its three
    /// vertices, tangent to the triangle (zero for a degenerate one).
    pub grad: Vec<[[f64; 3]; 3]>,
}

/// Barycentric gradients of triangle `p` with unit normal `n` and area `a`:
/// `grad lambda_i = n x (p[i+2] - p[i+1]) / 2a`.
fn bary_grad(p: [[f64; 3]; 3], n: [f64; 3], a: f64) -> [[f64; 3]; 3] {
    use rapidmesh_exact::vector::{cross, scale, sub};
    if !(a > 0.0) {
        return [[0.0; 3]; 3];
    }
    std::array::from_fn(|i| scale(cross(n, sub(p[(i + 2) % 3], p[(i + 1) % 3])), 0.5 / a))
}

impl TriGeometry {
    /// Planar (z = 0) geometry: area, centroid, +-z normal, second area moment,
    /// edge lengths/midpoints.
    pub fn build_2d(topo: &TriTopology, coords: &[[f64; 2]]) -> Self {
        let nt = topo.tris.len();
        let mut area = vec![0.0; nt];
        let mut centroid = vec![[0.0; 3]; nt];
        let mut normal = vec![[0.0; 3]; nt];
        let mut inertia = vec![[0.0; 3]; nt];
        let mut min_angle = vec![0.0; nt];
        let mut grad = vec![[[0.0; 3]; 3]; nt];
        for t in 0..nt {
            let [ia, ib, ic] = topo.tris[t];
            let (a, b, c) = (
                coords[ia as usize],
                coords[ib as usize],
                coords[ic as usize],
            );
            let signed = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
            area[t] = signed.abs();
            normal[t] = [0.0, 0.0, if signed >= 0.0 { 1.0 } else { -1.0 }];
            grad[t] = bary_grad(
                [[a[0], a[1], 0.0], [b[0], b[1], 0.0], [c[0], c[1], 0.0]],
                normal[t],
                area[t],
            );
            let (cx, cy) = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0);
            centroid[t] = [cx, cy, 0.0];
            let d = [
                [a[0] - cx, a[1] - cy],
                [b[0] - cx, b[1] - cy],
                [c[0] - cx, c[1] - cy],
            ];
            let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
            for p in &d {
                sxx += p[0] * p[0];
                sxy += p[0] * p[1];
                syy += p[1] * p[1];
            }
            let k = area[t] / 12.0;
            inertia[t] = [k * sxx, k * sxy, k * syy];
            min_angle[t] =
                tri_min_angle_deg([a[0], a[1], 0.0], [b[0], b[1], 0.0], [c[0], c[1], 0.0]);
        }
        let ne = topo.edges.len();
        let mut edge_len = vec![0.0; ne];
        let mut edge_mid = vec![[0.0; 3]; ne];
        for (e, &[ia, ib]) in topo.edges.iter().enumerate() {
            let (a, b) = (coords[ia as usize], coords[ib as usize]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            edge_len[e] = (dx * dx + dy * dy).sqrt();
            edge_mid[e] = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, 0.0];
        }
        TriGeometry {
            area,
            centroid,
            normal,
            inertia,
            edge_len,
            edge_mid,
            min_angle,
            grad,
        }
    }

    /// Surface (3D-embedded) geometry: area, centroid, unit normal, edge
    /// lengths/midpoints, min interior angle. `inertia` is left empty (see the
    /// field doc).
    pub fn build_3d(topo: &TriTopology, coords: &[[f64; 3]]) -> Self {
        use crate::edge_geom;
        use rapidmesh_exact::vector::{add, cross, len, normalize, scale, sub};
        let nt = topo.tris.len();
        let mut area = vec![0.0; nt];
        let mut centroid = vec![[0.0; 3]; nt];
        let mut normal = vec![[0.0; 3]; nt];
        let mut min_angle = vec![0.0; nt];
        let mut grad = vec![[[0.0; 3]; 3]; nt];
        for t in 0..nt {
            let [ia, ib, ic] = topo.tris[t];
            let (a, b, c) = (
                coords[ia as usize],
                coords[ib as usize],
                coords[ic as usize],
            );
            let n = cross(sub(b, a), sub(c, a));
            area[t] = 0.5 * len(n);
            normal[t] = normalize(n); // zero vector for a degenerate triangle
            grad[t] = bary_grad([a, b, c], normal[t], area[t]);
            centroid[t] = scale(add(add(a, b), c), 1.0 / 3.0);
            min_angle[t] = tri_min_angle_deg(a, b, c);
        }
        let (edge_len, edge_mid) = edge_geom(&topo.edges, coords);
        TriGeometry {
            area,
            centroid,
            normal,
            inertia: Vec::new(),
            edge_len,
            edge_mid,
            min_angle,
            grad,
        }
    }
}

/// Minimum interior angle (degrees) of triangle `(a, b, c)` in 3D.
fn tri_min_angle_deg(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    use rapidmesh_exact::vector::{dot, len, sub};
    let at = |u: [f64; 3], v: [f64; 3], w: [f64; 3]| {
        let (e1, e2) = (sub(v, u), sub(w, u));
        let cos = dot(e1, e2) / (len(e1) * len(e2) + 1e-30);
        cos.clamp(-1.0, 1.0).acos().to_degrees()
    };
    at(a, b, c).min(at(b, c, a)).min(at(c, a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Tris;
    use rapidmesh_exact::vector::{dot, sub};

    #[test]
    fn single_triangle() {
        let topo = TriTopology::build(&Tris::untagged(&[[0, 1, 2]], 3));
        assert_eq!(topo.edges.len(), 3);
        // every edge is a boundary edge: one incident triangle.
        for e in &topo.edge_tris {
            assert_eq!(e[0], 0);
            assert_eq!(e[1], NONE);
        }
        // each vertex touches the one triangle.
        for v in 0..3 {
            assert_eq!(topo.vert_tris.row(v), &[0]);
        }
    }

    #[test]
    fn shared_edge_is_interior() {
        // two triangles sharing edge (1,2).
        let topo = TriTopology::build(&Tris {
            tris: &[[0, 1, 2], [1, 3, 2]],
            tags: &[7, 9],
            n_verts: 4,
        });
        assert_eq!(topo.edges.len(), 5);
        // find the shared edge id (canonical (1,2)).
        let shared = topo.edges.iter().position(|&e| e == [1, 2]).unwrap();
        let mut tris = topo.edge_tris[shared];
        tris.sort_unstable();
        assert_eq!(tris, [0, 1]);
        // its two sides carry the two triangles' tags.
        let mut tags = topo.edge_tags[shared];
        tags.sort_unstable();
        assert_eq!(tags, [7, 9]);
    }

    #[test]
    fn geometry_2d_unit_right_triangle() {
        let topo = TriTopology::build(&Tris::untagged(&[[0, 1, 2]], 3));
        let g = TriGeometry::build_2d(&topo, &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert!((g.area[0] - 0.5).abs() < 1e-12);
        assert_eq!(g.normal[0], [0.0, 0.0, 1.0]); // CCW
        let c = g.centroid[0];
        assert!(
            (c[0] - 1.0 / 3.0).abs() < 1e-12 && (c[1] - 1.0 / 3.0).abs() < 1e-12 && c[2] == 0.0
        );
        // symmetric triangle: int dx^2 == int dy^2, cross moment negative.
        assert!((g.inertia[0][0] - g.inertia[0][2]).abs() < 1e-12);
        assert!(g.inertia[0][1] < 0.0);
    }

    /// The gradients of a tilted triangle: tangent, summing to zero, and
    /// `grad lambda_i * (p_j - p_k)` the difference of the Kronecker deltas.
    #[test]
    fn barycentric_gradients() {
        let p = [[0.1, 0.2, 0.3], [1.7, 0.4, -0.2], [0.5, 1.3, 0.9]];
        let topo = TriTopology::build(&Tris::untagged(&[[0, 1, 2]], 3));
        let g = TriGeometry::build_3d(&topo, &p);
        for i in 0..3 {
            assert!(dot(g.grad[0][i], g.normal[0]).abs() < 1e-12);
            for j in 0..3 {
                let want =
                    if i == j { 1.0 } else { 0.0 } - if i == (j + 1) % 3 { 1.0 } else { 0.0 };
                let got = dot(g.grad[0][i], sub(p[j], p[(j + 1) % 3]));
                assert!((got - want).abs() < 1e-12, "{i} {j}: {got}");
            }
        }
    }

    /// Three sheets meeting in one edge: two junction RWG functions from
    /// the lowest triangle, the edge no outline; with tags apart, only the
    /// pair of equal tags.
    #[test]
    fn junction_rwg() {
        let tris = [[0, 1, 2], [1, 0, 3], [0, 1, 4]];
        let topo = TriTopology::build(&Tris {
            tris: &tris,
            tags: &[5, 5, 6],
            n_verts: 5,
        });
        let e = topo.edges.iter().position(|&e| e == [0, 1]).unwrap();
        assert_eq!(topo.edge_tris_all.row(e), &[0, 1, 2]);
        let all = topo.rwg_edges(&topo.tri_tags, true);
        assert_eq!(all, vec![[0, 1, 0, 1], [0, 1, 0, 2]]);
        assert_eq!(topo.rwg_candidate_edges(), vec![[0, 1, 0, 1]]);
        assert!(topo.boundary_edges().iter().any(|b| b[..2] == [0, 1]));
        let same = TriTopology::build(&Tris::untagged(&tris, 5));
        assert!(!same.boundary_edges().iter().any(|b| b[..2] == [0, 1]));
    }

    #[test]
    fn geometry_3d_normal_and_area() {
        let topo = TriTopology::build(&Tris::untagged(&[[0, 1, 2]], 3));
        let g = TriGeometry::build_3d(&topo, &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        assert!((g.area[0] - 2.0).abs() < 1e-12);
        assert_eq!(g.normal[0], [0.0, 0.0, 1.0]);
        assert!(g.inertia.is_empty());
        assert_eq!(g.edge_len.len(), topo.edges.len());
    }
}
