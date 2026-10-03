//! Tetrahedral complex: full 0/1/2/3-cell incidence with the orientation signs
//! a vector-FEM (Nédélec) assembly needs, plus coordinate-aware geometry.

use crate::convention::{
    canonical_edge, face_perm, sort3_sign, NONE, TET_EDGE_LOCAL, TET_FACE_LOCAL, TRI_EDGE_LOCAL,
};
use crate::csr::Csr;
use crate::source::TetSource;
use std::collections::HashMap;

/// Derived connectivity of a tet mesh. Pure topology -- no coordinates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TetTopology {
    pub n_verts: usize,
    /// The tet -> vertex connectivity (the source elements, as `u32`), so the
    /// complex is self-contained and the geometry builders need only coordinates.
    pub tets: Vec<[u32; 4]>,
    /// Unique edges, canonical `(min, max)`.
    pub edges: Vec<[u32; 2]>,
    /// Unique faces, canonical (vertex ids ascending).
    pub faces: Vec<[u32; 3]>,
    /// Global edge id per local edge (`TET_EDGE_LOCAL` order).
    pub tet_edges: Vec<[u32; 6]>,
    /// `+1` if the local edge runs min->max (matches canonical), else `-1`.
    pub tet_edge_sign: Vec<[i8; 6]>,
    /// Global face id per local face (`TET_FACE_LOCAL` order).
    pub tet_faces: Vec<[u32; 4]>,
    /// Parity from the local outward order to the canonical face order. Two tets
    /// sharing a face carry opposite signs.
    pub tet_face_sign: Vec<[i8; 4]>,
    /// The permutation from the canonical face order to the local outward
    /// order, as an index into [`crate::convention::FACE_PERMS`].
    pub tet_face_perm: Vec<[u8; 4]>,
    /// The 3 edges of each face (`TRI_EDGE_LOCAL` order on the canonical face).
    pub face_edges: Vec<[u32; 3]>,
    /// The (<=2) tets incident to each face; `NONE` marks a boundary face.
    pub face_tets: Vec<[u32; 2]>,
    /// Vertex -> incident edges.
    pub vert_edges: Csr,
    /// Vertex -> incident tets.
    pub vert_tets: Csr,
}

impl TetTopology {
    /// Build the complex in one O(n) pass.
    pub fn build(src: &impl TetSource) -> Self {
        let nt = src.n_tets();
        let mut edge_id: HashMap<[u32; 2], u32> = HashMap::new();
        let mut edges: Vec<[u32; 2]> = Vec::new();
        let mut face_id: HashMap<[u32; 3], u32> = HashMap::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();
        let mut tet_edges = vec![[0u32; 6]; nt];
        let mut tet_edge_sign = vec![[0i8; 6]; nt];
        let mut tet_faces = vec![[0u32; 4]; nt];
        let mut tet_face_sign = vec![[0i8; 4]; nt];
        let mut tet_face_perm = vec![[0u8; 4]; nt];
        let mut tets = vec![[0u32; 4]; nt];
        let mut vt_pairs: Vec<(u32, u32)> = Vec::with_capacity(nt * 4);

        for t in 0..nt {
            let tet = src.tet(t);
            tets[t] = tet;
            for &v in &tet {
                vt_pairs.push((v, t as u32));
            }
            for (k, &[la, lb]) in TET_EDGE_LOCAL.iter().enumerate() {
                let (e, s) = canonical_edge(tet[la], tet[lb]);
                let id = *edge_id.entry(e).or_insert_with(|| {
                    edges.push(e);
                    (edges.len() - 1) as u32
                });
                tet_edges[t][k] = id;
                tet_edge_sign[t][k] = s;
            }
            for (k, &[a, b, c]) in TET_FACE_LOCAL.iter().enumerate() {
                let (sorted, s) = sort3_sign([tet[a], tet[b], tet[c]]);
                let id = *face_id.entry(sorted).or_insert_with(|| {
                    faces.push(sorted);
                    (faces.len() - 1) as u32
                });
                tet_faces[t][k] = id;
                tet_face_sign[t][k] = s;
                tet_face_perm[t][k] = face_perm([tet[a], tet[b], tet[c]]);
            }
        }

        // Face -> its 3 edges (every edge already exists as a tet edge).
        let nf = faces.len();
        let mut face_edges = vec![[0u32; 3]; nf];
        for (fi, &f) in faces.iter().enumerate() {
            for (k, &[la, lb]) in TRI_EDGE_LOCAL.iter().enumerate() {
                let (e, _) = canonical_edge(f[la], f[lb]);
                face_edges[fi][k] = *edge_id.get(&e).expect("face edge must be a tet edge");
            }
        }

        // Face -> incident tets (a volume face has <=2).
        let mut face_tets = vec![[NONE; 2]; nf];
        let mut fcnt = vec![0u8; nf];
        for t in 0..nt {
            for k in 0..4 {
                let f = tet_faces[t][k] as usize;
                let c = fcnt[f];
                if c < 2 {
                    face_tets[f][c as usize] = t as u32;
                }
                fcnt[f] = c.saturating_add(1);
            }
        }

        let mut ve_pairs: Vec<(u32, u32)> = Vec::with_capacity(edges.len() * 2);
        for (ei, &[a, b]) in edges.iter().enumerate() {
            ve_pairs.push((a, ei as u32));
            ve_pairs.push((b, ei as u32));
        }
        let vert_edges = Csr::from_pairs(src.n_verts(), &ve_pairs);
        let vert_tets = Csr::from_pairs(src.n_verts(), &vt_pairs);

        TetTopology {
            n_verts: src.n_verts(),
            tets,
            edges,
            faces,
            tet_edges,
            tet_edge_sign,
            tet_faces,
            tet_face_sign,
            tet_face_perm,
            face_edges,
            face_tets,
            vert_edges,
            vert_tets,
        }
    }
}

/// Per-element geometry of a tet mesh. All basis-free facts about the embedding.
/// `grad` holds grad lambda_i (the barycentric-coordinate gradients = the inverse-
/// transpose Jacobian) -- pure simplex calculus, the only per-element datum a
/// P1/Nédélec assembly needs.
#[derive(Debug, Clone, Default)]
pub struct TetGeometry {
    /// Unsigned tet volume.
    pub volume: Vec<f64>,
    /// grad lambda_i for the 4 local vertices (constant per tet). `sum_i grad lambda_i = 0`.
    pub grad: Vec<[[f64; 3]; 4]>,
    /// Per-edge length (parallel to `TetTopology::edges`).
    pub edge_len: Vec<f64>,
    /// Per-face area (parallel to `TetTopology::faces`).
    pub face_area: Vec<f64>,
    /// Per-face unit normal: boundary outward, interior `t0 -> t1` (oriented away
    /// from `face_tets[f][0]`'s opposite vertex -- outward for a boundary face).
    pub face_normal: Vec<[f64; 3]>,
    pub face_centroid: Vec<[f64; 3]>,
}

impl TetGeometry {
    pub fn build(topo: &TetTopology, coords: &[[f64; 3]]) -> Self {
        use crate::math::{add, cross, det3, dot, edge_geom, inv3, norm, scale, sub};

        let nt = topo.tets.len();
        let mut volume = vec![0.0; nt];
        let mut grad = vec![[[0.0; 3]; 4]; nt];
        for t in 0..nt {
            let [i0, i1, i2, i3] = topo.tets[t];
            let (p0, p1, p2, p3) = (
                coords[i0 as usize],
                coords[i1 as usize],
                coords[i2 as usize],
                coords[i3 as usize],
            );
            // T has columns (p1-p0, p2-p0, p3-p0); det(T) = 6 * signed volume.
            let (e1, e2, e3) = (sub(p1, p0), sub(p2, p0), sub(p3, p0));
            let m = [
                [e1[0], e2[0], e3[0]],
                [e1[1], e2[1], e3[1]],
                [e1[2], e2[2], e3[2]],
            ];
            volume[t] = det3(m).abs() / 6.0;
            if let Some(inv) = inv3(m) {
                // lambda_{1,2,3} = (T^{-1}(x - p0))_{0,1,2} -> grad lambda_i = rows of T^{-1}.
                let (g1, g2, g3) = (inv[0], inv[1], inv[2]);
                grad[t][1] = g1;
                grad[t][2] = g2;
                grad[t][3] = g3;
                grad[t][0] = [
                    -(g1[0] + g2[0] + g3[0]),
                    -(g1[1] + g2[1] + g3[1]),
                    -(g1[2] + g2[2] + g3[2]),
                ];
            }
        }

        let (edge_len, _) = edge_geom(&topo.edges, coords);

        let nf = topo.faces.len();
        let mut face_area = vec![0.0; nf];
        let mut face_normal = vec![[0.0; 3]; nf];
        let mut face_centroid = vec![[0.0; 3]; nf];
        for f in 0..nf {
            let [ia, ib, ic] = topo.faces[f];
            let (a, b, c) = (
                coords[ia as usize],
                coords[ib as usize],
                coords[ic as usize],
            );
            let n = cross(sub(b, a), sub(c, a));
            let len = norm(n);
            face_area[f] = 0.5 * len;
            let centroid = scale(add(add(a, b), c), 1.0 / 3.0);
            face_centroid[f] = centroid;
            let mut nn = if len > 0.0 {
                scale(n, 1.0 / len)
            } else {
                [0.0; 3]
            };
            // Orient away from face_tets[f][0]'s opposite vertex: outward for a
            // boundary face, t0 -> t1 for an interior one.
            let t0 = topo.face_tets[f][0];
            if t0 != NONE {
                let tet = topo.tets[t0 as usize];
                if let Some(&ov) = tet.iter().find(|&&v| v != ia && v != ib && v != ic) {
                    if dot(nn, sub(centroid, coords[ov as usize])) < 0.0 {
                        nn = scale(nn, -1.0);
                    }
                }
            }
            face_normal[f] = nn;
        }

        TetGeometry {
            volume,
            grad,
            edge_len,
            face_area,
            face_normal,
            face_centroid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Tets;

    #[test]
    fn single_tet() {
        let topo = TetTopology::build(&Tets {
            tets: &[[0, 1, 2, 3]],
            n_verts: 4,
        });
        assert_eq!(topo.edges.len(), 6);
        assert_eq!(topo.faces.len(), 4);
        // ascending vertex labels -> every local edge already runs min->max.
        assert_eq!(topo.tet_edge_sign[0], [1; 6]);
        // all four faces are on the boundary.
        for f in &topo.face_tets {
            assert_eq!(f[1], NONE);
        }
    }

    #[test]
    fn edge_sign_reversed() {
        // local edge 0 = (tet[0], tet[1]) = (3, 1) -> canonical (1,3), reversed.
        let topo = TetTopology::build(&Tets {
            tets: &[[3, 1, 2, 0]],
            n_verts: 4,
        });
        assert_eq!(topo.tet_edge_sign[0][0], -1);
    }

    #[test]
    fn shared_face_opposite_signs() {
        // two tets sharing face (1,2,3).
        let topo = TetTopology::build(&Tets {
            tets: &[[0, 1, 2, 3], [1, 2, 3, 4]],
            n_verts: 5,
        });
        let shared = topo.faces.iter().position(|&f| f == [1, 2, 3]).unwrap();
        let mut tets = topo.face_tets[shared];
        tets.sort_unstable();
        assert_eq!(tets, [0, 1]);
        // each tet lists the shared face; the two carry opposite orientation.
        let sign_in = |t: usize| {
            let k = topo.tet_faces[t]
                .iter()
                .position(|&f| f as usize == shared)
                .unwrap();
            topo.tet_face_sign[t][k]
        };
        assert_eq!(sign_in(0), -sign_in(1));
    }

    #[test]
    fn geometry_unit_tet() {
        let topo = TetTopology::build(&Tets {
            tets: &[[0, 1, 2, 3]],
            n_verts: 4,
        });
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let g = TetGeometry::build(&topo, &coords);
        assert!((g.volume[0] - 1.0 / 6.0).abs() < 1e-12);
        // grad lambda_0 = (-1,-1,-1).
        for k in 0..3 {
            assert!((g.grad[0][0][k] + 1.0).abs() < 1e-12);
        }
        // sum_i grad lambda_i = 0.
        let mut s = [0.0; 3];
        for i in 0..4 {
            for k in 0..3 {
                s[k] += g.grad[0][i][k];
            }
        }
        for k in 0..3 {
            assert!(s[k].abs() < 1e-12);
        }
        // all four faces are boundary -> unit, outward normals.
        let tetc = [0.25, 0.25, 0.25];
        for f in 0..topo.faces.len() {
            let n = g.face_normal[f];
            let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((nl - 1.0).abs() < 1e-12);
            let c = g.face_centroid[f];
            let out = n[0] * (c[0] - tetc[0]) + n[1] * (c[1] - tetc[1]) + n[2] * (c[2] - tetc[2]);
            assert!(out > 0.0, "face {f} normal not outward");
        }
    }

    #[test]
    fn geometry_grad_scaled_tet() {
        let topo = TetTopology::build(&Tets {
            tets: &[[0, 1, 2, 3]],
            n_verts: 4,
        });
        let coords = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let g = TetGeometry::build(&topo, &coords);
        assert!((g.volume[0] - 4.0).abs() < 1e-12); // (2*3*4)/6
        assert!((g.grad[0][1][0] - 0.5).abs() < 1e-12);
        assert!((g.grad[0][2][1] - 1.0 / 3.0).abs() < 1e-12);
        assert!((g.grad[0][3][2] - 0.25).abs() < 1e-12);
    }
}
