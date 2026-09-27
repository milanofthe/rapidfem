//! File output of a tet mesh: gmsh MSH 4.1 (ASCII) for solvers that read
//! gmsh meshes, VTK XML unstructured grid for viewing.
//!
//! The MSH entities are the geometry the mesh was made from: every B-rep
//! vertex, edge and face in use is a point, curve and surface entity (tag =
//! id + 1), every region a volume entity (tag = region). A node sits in the
//! block of what it is classified on, so a reader recovers the
//! classification; elements are grouped the same way (points on B-rep
//! vertices, lines on B-rep edges, triangles on B-rep faces, tets per
//! region). Physical groups: the regions (dimension 3), grouped and named
//! as the caller labels them, every nonzero face tag (dimension 2, tag =
//! face tag), and the caller's named groups of B-rep faces and edges.

use rapidmesh_geom::RegionTag;
use rapidmesh_tet::{CurveEdge, PointClass, SurfaceFace, SurfaceMesh, TetMesh};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{self, Write};

/// The physical groups: per region its group (tag and name; a label over
/// several regions is one group), per face tag its name. A region without
/// a group is a group of its own, `region_<r>`; an unnamed face tag is
/// `tag_<t>`.
#[derive(Debug, Clone, Default)]
pub struct Names {
    pub region_groups: HashMap<u32, (u32, String)>,
    pub face_tags: HashMap<u32, String>,
    /// Named groups of B-rep faces (dimension 2) or edges (dimension 1):
    /// (dimension, physical tag, name, B-rep ids). Their tags must not meet
    /// a face tag.
    pub groups: Vec<(u8, u32, String, Vec<u32>)>,
}

/// The parts of a mesh the files are written from: a tet mesh, or a
/// surface mesh (no tets).
struct Parts<'a> {
    points: &'a [[f64; 3]],
    point_class: &'a [PointClass],
    tets: &'a [[usize; 4]],
    tet_regions: &'a [RegionTag],
    faces: &'a [SurfaceFace],
    curve_edges: &'a [CurveEdge],
}

impl<'a> Parts<'a> {
    fn of_tets(m: &'a TetMesh) -> Parts<'a> {
        Parts {
            points: &m.points,
            point_class: &m.point_class,
            tets: &m.tets,
            tet_regions: &m.tet_regions,
            faces: &m.faces,
            curve_edges: &m.curve_edges,
        }
    }

    fn of_surface(m: &'a SurfaceMesh) -> Parts<'a> {
        Parts {
            points: &m.points,
            point_class: &m.point_class,
            tets: &[],
            tet_regions: &[],
            faces: &m.faces,
            curve_edges: &m.curve_edges,
        }
    }

    /// The entity of the cells at every point: the region of a tet at it,
    /// or (without tets) the B-rep face of a triangle at it; `None` for a
    /// point no cell uses.
    fn homes(&self) -> Vec<Option<(u8, u32)>> {
        let mut home = vec![None; self.points.len()];
        if self.tets.is_empty() {
            for f in self.faces.iter().filter(|f| f.patch != u32::MAX) {
                for &v in &f.tri {
                    home[v] = Some((2, f.patch + 1));
                }
            }
        } else {
            for (t, tet) in self.tets.iter().enumerate() {
                for &v in tet {
                    home[v] = Some((3, self.tet_regions[t].0));
                }
            }
        }
        home
    }
}

/// The (dimension, tag) of the entity a point is classified on, given the
/// entity of the cells at it.
fn node_entity(class: Option<&PointClass>, home: (u8, u32)) -> (u8, u32) {
    match class {
        Some(PointClass::Vertex(i)) => (0, i + 1),
        Some(PointClass::Edge(e)) if *e != u32::MAX => (1, e + 1),
        Some(PointClass::Face(f)) if *f != u32::MAX => (2, f + 1),
        _ => home,
    }
}

/// Writes a tet mesh as a gmsh MSH 4.1 ASCII file.
pub fn write_msh(mesh: &TetMesh, names: &Names, w: &mut impl Write) -> io::Result<()> {
    write_parts(&Parts::of_tets(mesh), names, w)
}

/// Writes a surface mesh as a gmsh MSH 4.1 ASCII file (points, lines and
/// triangles; face tags as physical groups).
pub fn write_surface_msh(mesh: &SurfaceMesh, names: &Names, w: &mut impl Write) -> io::Result<()> {
    write_parts(&Parts::of_surface(mesh), names, w)
}

fn write_parts(mesh: &Parts<'_>, names: &Names, w: &mut impl Write) -> io::Result<()> {
    let home = mesh.homes();

    // Elements per entity, in entity order.
    let mut blocks: BTreeMap<(u8, u32), (u8, Vec<Vec<usize>>)> = BTreeMap::new();
    for (v, class) in mesh.point_class.iter().enumerate() {
        if let (PointClass::Vertex(i), true) = (class, home[v].is_some()) {
            blocks
                .entry((0, i + 1))
                .or_insert((15, Vec::new()))
                .1
                .push(vec![v]);
        }
    }
    // Lines and triangles only where the cells are (a sheet reaching out of
    // every solid keeps its curves in the mesh, with no tets there).
    let inside = |vs: &[usize]| vs.iter().all(|&v| home[v].is_some());
    for ce in mesh.curve_edges {
        if ce.edge != u32::MAX && inside(&ce.v) {
            blocks
                .entry((1, ce.edge + 1))
                .or_insert((1, Vec::new()))
                .1
                .push(ce.v.to_vec());
        }
    }
    for sf in mesh.faces {
        if sf.patch != u32::MAX && inside(&sf.tri) {
            blocks
                .entry((2, sf.patch + 1))
                .or_insert((2, Vec::new()))
                .1
                .push(sf.tri.to_vec());
        }
    }
    for (t, tet) in mesh.tets.iter().enumerate() {
        blocks
            .entry((3, mesh.tet_regions[t].0))
            .or_insert((4, Vec::new()))
            .1
            .push(tet.to_vec());
    }

    // Nodes per entity.
    let mut nodes: BTreeMap<(u8, u32), Vec<usize>> = BTreeMap::new();
    for v in 0..mesh.points.len() {
        if let Some(h) = home[v] {
            nodes
                .entry(node_entity(mesh.point_class.get(v), h))
                .or_default()
                .push(v);
        }
    }

    // Entities: those with nodes or elements; box over both; physical tags.
    let mut entities: BTreeSet<(u8, u32)> = nodes.keys().copied().collect();
    entities.extend(blocks.keys().copied());
    let mut bbox: HashMap<(u8, u32), ([f64; 3], [f64; 3])> = HashMap::new();
    let mut grow = |e: (u8, u32), p: [f64; 3]| {
        let b = bbox.entry(e).or_insert(([f64::MAX; 3], [f64::MIN; 3]));
        for k in 0..3 {
            b.0[k] = b.0[k].min(p[k]);
            b.1[k] = b.1[k].max(p[k]);
        }
    };
    for (&e, vs) in &nodes {
        vs.iter().for_each(|&v| grow(e, mesh.points[v]));
    }
    for (&e, (_, els)) in &blocks {
        els.iter().flatten().for_each(|&v| grow(e, mesh.points[v]));
    }
    // A B-rep face's physical group is its face tag.
    let mut face_tag_of: HashMap<u32, u32> = HashMap::new();
    for sf in mesh.faces {
        if sf.patch != u32::MAX && sf.face_tag.0 != 0 {
            face_tag_of.insert(sf.patch + 1, sf.face_tag.0);
        }
    }
    // Entity tag = B-rep id + 1 for faces and edges.
    let physical = |e: (u8, u32)| -> Vec<u32> {
        let mut out: Vec<u32> = match e.0 {
            3 => vec![names.region_groups.get(&e.1).map_or(e.1, |g| g.0)],
            2 => face_tag_of.get(&e.1).copied().into_iter().collect(),
            _ => Vec::new(),
        };
        for (dim, tag, _, ids) in &names.groups {
            if *dim == e.0 && e.0 > 0 && ids.contains(&(e.1 - 1)) {
                out.push(*tag);
            }
        }
        out
    };

    writeln!(w, "$MeshFormat\n4.1 0 8\n$EndMeshFormat")?;

    let mut phys: BTreeSet<(u8, u32)> = BTreeSet::new();
    for &e in &entities {
        for p in physical(e) {
            phys.insert((e.0, p));
        }
    }
    writeln!(w, "$PhysicalNames\n{}", phys.len())?;
    for &(dim, tag) in &phys {
        let named = names
            .groups
            .iter()
            .find(|g| g.0 == dim && g.1 == tag)
            .map(|g| g.2.clone());
        let name = if let Some(n) = named {
            n
        } else if dim == 3 {
            names
                .region_groups
                .values()
                .find(|g| g.0 == tag)
                .map_or(format!("region_{tag}"), |g| g.1.clone())
        } else {
            names
                .face_tags
                .get(&tag)
                .cloned()
                .unwrap_or(format!("tag_{tag}"))
        };
        writeln!(w, "{dim} {tag} \"{name}\"")?;
    }
    writeln!(w, "$EndPhysicalNames")?;

    let count = |d: u8| entities.iter().filter(|e| e.0 == d).count();
    writeln!(w, "$Entities")?;
    writeln!(w, "{} {} {} {}", count(0), count(1), count(2), count(3))?;
    for &e in &entities {
        let (lo, hi) = bbox.get(&e).copied().unwrap_or(([0.0; 3], [0.0; 3]));
        let ps = physical(e);
        let tags = std::iter::once(ps.len().to_string())
            .chain(ps.iter().map(|p| p.to_string()))
            .collect::<Vec<_>>()
            .join(" ");
        if e.0 == 0 {
            writeln!(w, "{} {} {} {} {tags}", e.1, lo[0], lo[1], lo[2])?;
        } else {
            writeln!(
                w,
                "{} {} {} {} {} {} {} {tags} 0",
                e.1, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
            )?;
        }
    }
    writeln!(w, "$EndEntities")?;

    let n_nodes: usize = nodes.values().map(Vec::len).sum();
    let used = nodes.values().flatten();
    let (min_tag, max_tag) = used.fold((usize::MAX, 0), |(a, b), &v| (a.min(v + 1), b.max(v + 1)));
    writeln!(w, "$Nodes\n{} {n_nodes} {min_tag} {max_tag}", nodes.len())?;
    for (&(dim, tag), vs) in &nodes {
        writeln!(w, "{dim} {tag} 0 {}", vs.len())?;
        for &v in vs {
            writeln!(w, "{}", v + 1)?;
        }
        for &v in vs {
            let p = mesh.points[v];
            writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
        }
    }
    writeln!(w, "$EndNodes")?;

    let n_el: usize = blocks.values().map(|b| b.1.len()).sum();
    writeln!(w, "$Elements\n{} {n_el} 1 {n_el}", blocks.len())?;
    let mut tag = 0usize;
    for (&(dim, etag), (ty, els)) in &blocks {
        writeln!(w, "{dim} {etag} {ty} {}", els.len())?;
        for el in els {
            tag += 1;
            write!(w, "{tag}")?;
            for &v in el {
                write!(w, " {}", v + 1)?;
            }
            writeln!(w)?;
        }
    }
    writeln!(w, "$EndElements")
}

/// Writes a tet mesh as a VTK XML unstructured grid (ASCII): the tets and
/// the faces on the geometry, with cell data `region` (tets), `patch` and
/// `face_tag` (faces; -1 on the other kind).
pub fn write_vtu(mesh: &TetMesh, w: &mut impl Write) -> io::Result<()> {
    write_vtu_parts(&Parts::of_tets(mesh), w)
}

/// Writes a surface mesh as a VTK XML unstructured grid (ASCII), with cell
/// data `patch` and `face_tag`.
pub fn write_surface_vtu(mesh: &SurfaceMesh, w: &mut impl Write) -> io::Result<()> {
    write_vtu_parts(&Parts::of_surface(mesh), w)
}

fn write_vtu_parts(mesh: &Parts<'_>, w: &mut impl Write) -> io::Result<()> {
    let (nt, nf) = (mesh.tets.len(), mesh.faces.len());
    writeln!(w, "<?xml version=\"1.0\"?>")?;
    writeln!(
        w,
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(w, "<UnstructuredGrid>")?;
    writeln!(
        w,
        "<Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">",
        mesh.points.len(),
        nt + nf
    )?;
    writeln!(
        w,
        "<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    for p in mesh.points {
        writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
    }
    writeln!(w, "</DataArray></Points>")?;
    writeln!(w, "<Cells>")?;
    writeln!(
        w,
        "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">"
    )?;
    for t in mesh.tets {
        writeln!(w, "{} {} {} {}", t[0], t[1], t[2], t[3])?;
    }
    for f in mesh.faces {
        writeln!(w, "{} {} {}", f.tri[0], f.tri[1], f.tri[2])?;
    }
    writeln!(w, "</DataArray>")?;
    writeln!(
        w,
        "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">"
    )?;
    let offsets = (1..=nt)
        .map(|i| 4 * i)
        .chain((1..=nf).map(|i| 4 * nt + 3 * i));
    for o in offsets {
        writeln!(w, "{o}")?;
    }
    writeln!(w, "</DataArray>")?;
    writeln!(
        w,
        "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )?;
    for i in 0..nt + nf {
        writeln!(w, "{}", if i < nt { 10 } else { 5 })?;
    }
    writeln!(w, "</DataArray>")?;
    writeln!(w, "</Cells>")?;
    writeln!(w, "<CellData Scalars=\"region\">")?;
    let array =
        |w: &mut dyn Write, name: &str, vals: &mut dyn Iterator<Item = i64>| -> io::Result<()> {
            writeln!(
                w,
                "<DataArray type=\"Int64\" Name=\"{name}\" format=\"ascii\">"
            )?;
            for v in vals {
                writeln!(w, "{v}")?;
            }
            writeln!(w, "</DataArray>")
        };
    let neg = |n: usize| std::iter::repeat_n(-1i64, n);
    array(
        w,
        "region",
        &mut mesh.tet_regions.iter().map(|r| r.0 as i64).chain(neg(nf)),
    )?;
    let patch = |p: u32| if p == u32::MAX { -1 } else { p as i64 };
    array(
        w,
        "patch",
        &mut neg(nt).chain(mesh.faces.iter().map(|f| patch(f.patch))),
    )?;
    array(
        w,
        "face_tag",
        &mut neg(nt).chain(mesh.faces.iter().map(|f| f.face_tag.0 as i64)),
    )?;
    writeln!(w, "</CellData>")?;
    writeln!(w, "</Piece>\n</UnstructuredGrid>\n</VTKFile>")
}
