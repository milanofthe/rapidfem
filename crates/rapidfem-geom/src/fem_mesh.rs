// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The solver mesh of a rapidmesh mesh.
//!
//! Nodes come over as they are. rapidmesh orients its tets negatively
//! (`(b-a)·((c-a)×(d-a)) < 0`), the solver's element expects the gmsh
//! convention, positive, so two vertices of every tet are swapped. The
//! solver's own edge and face numbering is derived from them by
//! [`Mesh::from_tets`].
//! The groups that the model's tags refer to are unions of rapidmesh entities:
//! a face group is a set of B-rep faces (every mesh face classified onto one
//! of them joins the group), a volume group a set of regions. That is the role
//! gmsh physical groups played.

use rapidfem_core::mesh::Mesh;

/// A tagged union of B-rep faces or regions.
#[derive(Clone, Debug)]
pub struct Group {
    pub tag: i32,
    pub ids: Vec<u32>,
}

fn signed_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0])
}

/// The solver mesh of `mesh`, with `ftag_to_tri` from `face_groups` (B-rep
/// face ids) and `vtag_to_tet` from `volume_groups` (region ids). The tets of
/// `holes` (region ids) are left out, so their walls become boundary faces;
/// the nodes are renumbered over the tets that remain.
pub fn fem_mesh(mesh: &rapidmesh::Mesh, face_groups: &[Group], volume_groups: &[Group], holes: &[u32]) -> Mesh {
    let p = &mesh.points;
    let keep: Vec<usize> = (0..mesh.tets.len()).filter(|&t| !holes.contains(&mesh.tet_regions[t].0)).collect();
    let mut node_of = vec![usize::MAX; p.len()];
    let mut nodes = Vec::new();
    let mut tets: Vec<[usize; 4]> = Vec::with_capacity(keep.len());
    for &t in &keep {
        let mut q = mesh.tets[t];
        if signed_volume(p[q[0]], p[q[1]], p[q[2]], p[q[3]]) < 0.0 {
            q.swap(1, 2);
        }
        let mut local = [0usize; 4];
        for (k, &v) in q.iter().enumerate() {
            if node_of[v] == usize::MAX {
                node_of[v] = nodes.len();
                nodes.push(p[v]);
            }
            local[k] = node_of[v];
        }
        tets.push(local);
    }
    let mut out = Mesh::from_tets(nodes, tets);

    // B-rep face -> solver triangles, through the classified topology faces.
    let view = mesh.view();
    let mut by_patch: hashbrown::HashMap<u32, Vec<usize>> = hashbrown::HashMap::new();
    for (f, &patch) in view.class.face_patch.iter().enumerate() {
        if patch == rapidmesh::NONE {
            continue;
        }
        let [a, b, c] = view.topo.faces[f];
        let (a, b, c) = (node_of[a as usize], node_of[b as usize], node_of[c as usize]);
        if a == usize::MAX || b == usize::MAX || c == usize::MAX {
            continue; // a face inside a hole
        }
        let mut k = [a, b, c];
        k.sort_unstable();
        let key = (k[0], k[1], k[2]);
        if let Some(&tri) = out.inv_tris.get(&key) {
            by_patch.entry(patch).or_default().push(tri);
        }
    }
    for g in face_groups {
        let tris = out.ftag_to_tri.entry(g.tag).or_default();
        for id in &g.ids {
            tris.extend(by_patch.get(id).into_iter().flatten().copied());
        }
    }

    for g in volume_groups {
        let tets = out.vtag_to_tet.entry(g.tag).or_default();
        for (new, &t) in keep.iter().enumerate() {
            if g.ids.contains(&mesh.tet_regions[t].0) {
                tets.push(new);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh::shapes::Cuboid;
    use rapidmesh::{FaceFilter, Geometry, MeshOptions, Scope};

    /// A dielectric block in an air box: every tet lands in exactly one
    /// region group, and the air's -z face group holds the boundary
    /// triangles of that face and nothing else.
    #[test]
    fn groups_follow_regions_and_brep_faces() {
        let mut g = Geometry::new(Some(0.5));
        let air = g.add(Cuboid::new([4.0, 4.0, 2.0])).unwrap();
        let diel = g.add_solid(Cuboid::new([2.0, 2.0, 1.0]).at([1.0, 1.0, 0.5]), Some(0.3), false).unwrap();
        let bottom = g.resolve(&Scope::surf(Some(FaceFilter::normal([0.0, 0.0, -1.0])))).unwrap();
        let m = g.mesh(&MeshOptions::default()).unwrap();

        let groups = [
            Group { tag: 1, ids: vec![air.region] },
            Group { tag: 2, ids: vec![diel.region] },
        ];
        let faces = [Group { tag: 10, ids: bottom.clone() }];
        let fm = fem_mesh(&m, &faces, &groups, &[]);
        assert_eq!(fm.n_tets(), m.tets.len());
        for t in &fm.tets {
            let p = |i: usize| fm.nodes[t[i]];
            assert!(signed_volume(p(0), p(1), p(2), p(3)) > 0.0, "tet not positively oriented");
        }
        let n1 = fm.vtag_to_tet[&1].len();
        let n2 = fm.vtag_to_tet[&2].len();
        assert!(n1 > 0 && n2 > 0);
        assert_eq!(n1 + n2, fm.n_tets(), "every tet in exactly one group");

        let tris = &fm.ftag_to_tri[&10];
        assert!(!tris.is_empty());
        for &t in tris {
            // a -z face triangle: all three nodes at the lowest z of its face
            let z: Vec<f64> = fm.tris[t].iter().map(|&n| fm.nodes[n][2]).collect();
            assert!(z.iter().all(|&v| (v - z[0]).abs() < 1e-12), "not planar in z: {z:?}");
        }
        // the -z faces are the air's floor and the block's floor (an interface)
        let boundary = tris.iter().filter(|&&t| fm.tri_to_tet[t][1] == usize::MAX).count();
        assert!(boundary > 0);
    }
}


