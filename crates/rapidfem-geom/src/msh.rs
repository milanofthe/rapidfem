// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! A pre-built gmsh MSH volume mesh, taken as it is (no remeshing).
//!
//! rapidmesh reads the file: a volume entity becomes a region, a surface
//! entity a geometric face. The physical groups are the selections the model
//! is built on: a volume group is a set of regions, a surface group a set of
//! geometric faces, both by name. [`write_msh`] writes a solver mesh back
//! out with the solver's groups as the physical groups.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{self, Write};
use std::path::Path;

use rapidmesh_geom::RegionTag;
use rapidmesh_tet::TetMesh;
use rapidmesh_topo::export::Names;

use rapidfem_core::mesh::Mesh;

use crate::fem_mesh::{fem_mesh, Group};

/// A named physical group of the file: regions (`dim == 3`) or geometric
/// faces (`dim == 2`).
#[derive(Clone, Debug)]
pub struct NamedGroup {
    pub name: String,
    pub dim: u8,
    pub ids: Vec<u32>,
}

/// The mesh of an MSH file with its named groups.
pub struct MeshScene {
    pub mesh: rapidmesh::Mesh,
    pub groups: Vec<NamedGroup>,
}

impl MeshScene {
    pub fn load(path: impl AsRef<Path>) -> Result<MeshScene, String> {
        let mesh = rapidmesh::load_msh(path).map_err(|e| e.to_string())?;
        let mut groups: Vec<NamedGroup> = mesh
            .labels
            .region_groups()
            .into_iter()
            .map(|(name, ids)| NamedGroup { name, dim: 3, ids })
            .collect();
        // A surface's first physical group is its face tag: the group holds
        // every geometric face carrying the tag.
        let mut by_tag: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        for f in mesh.faces.iter().filter(|f| f.face_tag.0 != 0 && f.patch != rapidmesh::NONE) {
            by_tag.entry(f.face_tag.0).or_default().insert(f.patch);
        }
        for (tag, patches) in by_tag {
            let name = mesh.labels.tag_labels.get(&tag).cloned().unwrap_or_else(|| format!("group_2_{tag}"));
            groups.push(NamedGroup { name, dim: 2, ids: patches.into_iter().collect() });
        }
        for (name, ids) in &mesh.labels.face_names {
            groups.push(NamedGroup { name: name.clone(), dim: 2, ids: ids.clone() });
        }
        Ok(MeshScene { mesh, groups })
    }

    pub fn group(&self, name: &str) -> Result<&NamedGroup, String> {
        self.groups.iter().find(|g| g.name == name).ok_or_else(|| {
            let names: Vec<&str> = self.groups.iter().map(|g| g.name.as_str()).collect();
            format!("no physical group {name:?} in the mesh; available: {}", names.join(", "))
        })
    }

    /// `(xmin, ymin, zmin, xmax, ymax, zmax)` over the nodes of a group.
    pub fn bbox(&self, name: &str) -> Result<[f64; 6], String> {
        let g = self.group(name)?;
        let m = &self.mesh;
        let mut b = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let mut grow = |p: [f64; 3]| {
            for k in 0..3 {
                b[k] = b[k].min(p[k]);
                b[k + 3] = b[k + 3].max(p[k]);
            }
        };
        if g.dim == 3 {
            for (t, r) in m.tets.iter().zip(&m.tet_regions) {
                if g.ids.contains(&r.0) {
                    t.iter().for_each(|&v| grow(m.points[v]));
                }
            }
        } else {
            for f in m.faces.iter().filter(|f| g.ids.contains(&f.patch)) {
                f.tri.iter().for_each(|&v| grow(m.points[v]));
            }
        }
        Ok(b)
    }

    /// The solver mesh, each tag's group the union of the named groups.
    pub fn fem_mesh(
        &self,
        face_groups: &[(i32, Vec<String>)],
        volume_groups: &[(i32, Vec<String>)],
    ) -> Result<Mesh, String> {
        let resolve = |groups: &[(i32, Vec<String>)], dim: u8| -> Result<Vec<Group>, String> {
            groups
                .iter()
                .map(|(tag, names)| {
                    let mut ids = Vec::new();
                    for n in names {
                        let g = self.group(n)?;
                        if g.dim != dim {
                            return Err(format!("group {n:?} is {}-D, expected {dim}-D", g.dim));
                        }
                        ids.extend(&g.ids);
                    }
                    Ok(Group { tag: *tag, ids })
                })
                .collect()
        };
        Ok(fem_mesh(&self.mesh, &resolve(face_groups, 2)?, &resolve(volume_groups, 3)?, &[]))
    }
}

/// Writes `mesh` without the tets of `holes` as gmsh MSH 4.1: each volume
/// group (name, regions) and face group (name, B-rep faces) a physical
/// group. A region is one volume entity and carries one group; the names of
/// volume groups left without a region of their own (a region already taken
/// by an earlier group) are returned.
pub fn write_msh(
    mesh: &rapidmesh::Mesh,
    volume_groups: &[(String, Vec<u32>)],
    face_groups: &[(String, Vec<u32>)],
    holes: &[u32],
    w: &mut impl Write,
) -> io::Result<Vec<String>> {
    let keep: Vec<usize> = (0..mesh.tets.len()).filter(|&t| !holes.contains(&mesh.tet_regions[t].0)).collect();
    let open = |r: RegionTag| if holes.contains(&r.0) { RegionTag(0) } else { r };
    // faces inside a hole go
    let mut faces = Vec::with_capacity(mesh.faces.len());
    for f in &mesh.faces {
        let mut f = f.clone();
        f.regions = f.regions.map(open);
        if f.regions != [RegionTag(0); 2] {
            faces.push(f);
        }
    }
    let m = TetMesh {
        points: mesh.points.clone(),
        tets: keep.iter().map(|&t| mesh.tets[t]).collect(),
        tet_regions: keep.iter().map(|&t| mesh.tet_regions[t]).collect(),
        faces,
        surfaces: mesh.surfaces.clone(),
        surface_owners: mesh.surface_owners.clone(),
        plc_points: mesh.plc_points,
        point_class: mesh.point_class.clone(),
        curve_edges: mesh.curve_edges.clone(),
        periodic_points: mesh.periodic_points.clone(),
    };
    let mut region_groups = HashMap::new();
    let mut dropped = Vec::new();
    for (name, regions) in volume_groups {
        let free: Vec<u32> = regions.iter().copied().filter(|r| !region_groups.contains_key(r)).collect();
        match free.iter().min() {
            Some(&tag) => free.iter().for_each(|&r| {
                region_groups.insert(r, (tag, name.clone()));
            }),
            None => dropped.push(name.clone()),
        }
    }
    // Face groups take tags past every face tag the mesh carries.
    let first = mesh.faces.iter().map(|f| f.face_tag.0).max().unwrap_or(0) + 1;
    let groups = face_groups
        .iter()
        .enumerate()
        .map(|(k, (name, ids))| (2u8, first + k as u32, name.clone(), ids.clone()))
        .collect();
    let names = Names { region_groups, face_tags: HashMap::new(), groups };
    rapidmesh_topo::export::write_msh(&m, &names, w)?;
    Ok(dropped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh::shapes::Cuboid;
    use rapidmesh::{FaceFilter, Geometry, MeshOptions, Scope};

    /// A box written by rapidmesh with a named region and two named faces
    /// comes back with those groups, and the solver mesh carries them.
    #[test]
    fn named_groups_survive_a_round_trip() {
        let mut g = Geometry::new(Some(0.5));
        let air = g.add(Cuboid::new([2.0, 1.0, 3.0])).unwrap();
        g.label_solid(air, "air");
        let face = |n: [f64; 3]| Scope::surf(Some(FaceFilter::normal(n)));
        g.name(&face([0.0, 0.0, -1.0]), "port_in").unwrap();
        g.name(&face([0.0, 0.0, 1.0]), "port_out").unwrap();
        let m = g.mesh(&MeshOptions::default()).unwrap();
        let path = std::env::temp_dir().join(format!("rapidfem-msh-{}.msh", std::process::id()));
        m.write_msh(&path).unwrap();
        let s = MeshScene::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let names: BTreeSet<&str> = s.groups.iter().map(|g| g.name.as_str()).collect();
        for n in ["air", "port_in", "port_out"] {
            assert!(names.contains(n), "{n} missing from {names:?}");
        }
        assert_eq!(s.bbox("port_out").unwrap(), [0.0, 0.0, 3.0, 2.0, 1.0, 3.0]);
        let fm = s
            .fem_mesh(&[(2, vec!["port_in".into()])], &[(1, vec!["air".into()])])
            .unwrap();
        assert_eq!(fm.vtag_to_tet[&1].len(), fm.n_tets());
        let tris = &fm.ftag_to_tri[&2];
        assert!(!tris.is_empty());
        assert!(tris.iter().all(|&t| fm.tris[t].iter().all(|&n| fm.nodes[n][2] == 0.0)));
        assert!(s.fem_mesh(&[(2, vec!["air".into()])], &[]).is_err());
    }

    /// The solver's groups written out come back by name, the hole left out.
    #[test]
    fn write_msh_keeps_the_solver_groups() {
        use crate::geometry::{FaceOrigin, FaceSel, Geometry};
        let mut g = Geometry::new(Some(0.5));
        let air = g.add_solid(Cuboid::new([3.0, 3.0, 3.0]), None, false);
        let hole = g.add_solid(Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]), None, false);
        g.make_void(&[hole]);
        let m = g.mesh(&MeshOptions::default()).unwrap();
        let ra = g.region(air).unwrap().unwrap();
        let walls: Vec<u32> = g.resolve(&g.faces_of(hole).unwrap()).unwrap().iter().map(|f| f.id).collect();
        let top = FaceSel { origin: FaceOrigin::Solid { object: air, role: 1 }, side: None, across: None };
        let top: Vec<u32> = g.resolve(&[top]).unwrap().iter().map(|f| f.id).collect();
        let holes = g.hole_regions().unwrap();
        let path = std::env::temp_dir().join(format!("rapidfem-msh-w-{}.msh", std::process::id()));
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let dropped = write_msh(
            &m,
            &[("air".into(), vec![ra]), ("again".into(), vec![ra])],
            &[("pec".into(), walls), ("top".into(), top)],
            &holes,
            &mut f,
        )
        .unwrap();
        drop(f);
        assert_eq!(dropped, vec!["again".to_string()]);
        let s = MeshScene::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let direct = crate::fem_mesh::fem_mesh(&m, &[], &[], &holes);
        let fm = s.fem_mesh(&[(2, vec!["pec".into()]), (3, vec!["top".into()])], &[(1, vec!["air".into()])]).unwrap();
        assert_eq!(fm.n_tets(), direct.n_tets());
        assert_eq!(fm.vtag_to_tet[&1].len(), fm.n_tets());
        assert_eq!(s.bbox("pec").unwrap(), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        // the walls are the boundary now: one tet on each of their triangles
        assert!(fm.ftag_to_tri[&2].iter().all(|&t| fm.tri_to_tet[t][1] == usize::MAX));
        assert!(!fm.ftag_to_tri[&3].is_empty());
    }
}
