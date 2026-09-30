//! Reading a gmsh MSH file back into a [`Mesh`]: the volume mesh as it is,
//! with its physical groups as the names, no remeshing.
//!
//! MSH 4.1 and 2.2, ASCII. A volume entity becomes a region, labelled by the
//! name of its first physical group; a surface entity a geometric face
//! (its tag less one), whose first physical group is its face tag and any
//! further ones named face sets; a curve entity a geometric edge, its
//! physical groups named edge sets. Region interfaces and the boundary the
//! file has no triangles on become faces too, outside every geometric face.

use crate::mesh::{Labels, Mesh, Run, SolidInfo};
use crate::{Error, Result};
use rapidmesh_geom::plc::SHEET_OWNER;
use rapidmesh_geom::{FaceTag, RegionTag, SurfaceKind};
use rapidmesh_tet::{quality_stats, CurveEdge, PointClass, SurfaceFace, TetMesh};
use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;
use std::path::Path;

/// The mesh in the MSH file at `path`.
pub fn load_msh(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    let f = std::fs::File::open(path).map_err(Error::Io)?;
    read_msh(std::io::BufReader::new(f))
        .map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))
}

/// The mesh in an MSH stream.
pub fn read_msh(r: impl BufRead) -> Result<Mesh> {
    let raw = parse(r).map_err(Error::Invalid)?;
    build(raw).map_err(Error::Invalid)
}

/// What a file holds, by gmsh's own ids.
#[derive(Default)]
struct Raw {
    names: HashMap<(u8, u32), String>,
    /// Physical tags per entity `(dim, tag)`.
    entity_groups: HashMap<(u8, u32), Vec<u32>>,
    /// Node tag -> coordinates and the entity it lies on.
    nodes: Vec<(usize, [f64; 3], (u8, u32))>,
    /// Elements: gmsh type, entity `(dim, tag)`, node tags.
    elements: Vec<(u32, (u8, u32), Vec<usize>)>,
}

/// Whitespace-separated tokens of a section's lines, with quoted names kept.
struct Tokens<R: BufRead> {
    lines: std::io::Lines<R>,
    buf: std::collections::VecDeque<String>,
}

impl<R: BufRead> Tokens<R> {
    fn line(&mut self) -> std::result::Result<Option<String>, String> {
        self.buf.clear();
        match self.lines.next() {
            None => Ok(None),
            Some(l) => l.map(Some).map_err(|e| e.to_string()),
        }
    }
    fn next(&mut self) -> std::result::Result<String, String> {
        while self.buf.is_empty() {
            let l = self.line()?.ok_or("the file ends inside a section")?;
            self.buf = l.split_whitespace().map(str::to_string).collect();
        }
        Ok(self.buf.pop_front().unwrap())
    }
    fn num<T: std::str::FromStr>(&mut self) -> std::result::Result<T, String> {
        let t = self.next()?;
        t.parse().map_err(|_| format!("{t:?} is not a number here"))
    }
    /// Skips to the line `end`.
    fn skip_to(&mut self, end: &str) -> std::result::Result<(), String> {
        loop {
            match self.line()? {
                Some(l) if l.trim() == end => return Ok(()),
                Some(_) => {}
                None => return Err(format!("no {end}")),
            }
        }
    }
}

fn parse(r: impl BufRead) -> std::result::Result<Raw, String> {
    let mut t = Tokens {
        lines: r.lines(),
        buf: Default::default(),
    };
    let mut raw = Raw::default();
    let mut version = 0.0f64;
    // Per element entity in 2.2: its physical tag.
    let mut old_groups: HashMap<(u8, u32), Vec<u32>> = HashMap::new();
    while let Some(l) = t.line()? {
        match l.trim() {
            "$MeshFormat" => {
                version = t.num()?;
                let binary: u32 = t.num()?;
                if binary != 0 {
                    return Err("binary MSH files are not read; save as ASCII".into());
                }
                if !(version == 2.2 || (4.0..4.2).contains(&version)) {
                    return Err(format!("MSH version {version} is not read (2.2 or 4.1)"));
                }
                t.skip_to("$EndMeshFormat")?;
            }
            "$PhysicalNames" => {
                let n: usize = t.num()?;
                for _ in 0..n {
                    let dim: u8 = t.num()?;
                    let tag: u32 = t.num()?;
                    // The rest of the line is the quoted name.
                    let mut name = t.next()?;
                    let closed = |n: &str| n.len() > 1 && n.ends_with('"');
                    while !closed(&name) && !t.buf.is_empty() {
                        name.push(' ');
                        name.push_str(&t.next()?);
                    }
                    raw.names
                        .insert((dim, tag), name.trim_matches('"').to_string());
                }
                t.skip_to("$EndPhysicalNames")?;
            }
            "$Entities" => {
                let counts: [usize; 4] = [t.num()?, t.num()?, t.num()?, t.num()?];
                for (dim, &n) in counts.iter().enumerate() {
                    for _ in 0..n {
                        let tag: u32 = t.num()?;
                        let coords = if dim == 0 { 3 } else { 6 };
                        for _ in 0..coords {
                            t.next()?;
                        }
                        let np: usize = t.num()?;
                        let groups: Vec<u32> = (0..np)
                            .map(|_| t.num::<i64>().map(|p| p.unsigned_abs() as u32))
                            .collect::<std::result::Result<_, _>>()?;
                        raw.entity_groups.insert((dim as u8, tag), groups);
                        if dim > 0 {
                            let nb: usize = t.num()?;
                            for _ in 0..nb {
                                t.next()?;
                            }
                        }
                    }
                }
                t.skip_to("$EndEntities")?;
            }
            "$Nodes" if version >= 4.0 => {
                let blocks: usize = t.num()?;
                let _n: usize = t.num()?;
                t.next()?;
                t.next()?;
                for _ in 0..blocks {
                    let dim: u8 = t.num()?;
                    let tag: u32 = t.num()?;
                    let parametric: u32 = t.num()?;
                    let n: usize = t.num()?;
                    let tags: Vec<usize> = (0..n)
                        .map(|_| t.num())
                        .collect::<std::result::Result<_, _>>()?;
                    for tag_n in tags {
                        let p = [t.num()?, t.num()?, t.num()?];
                        if parametric != 0 {
                            for _ in 0..dim {
                                t.next()?;
                            }
                        }
                        raw.nodes.push((tag_n, p, (dim, tag)));
                    }
                }
                t.skip_to("$EndNodes")?;
            }
            "$Nodes" => {
                let n: usize = t.num()?;
                for _ in 0..n {
                    let tag: usize = t.num()?;
                    let p = [t.num()?, t.num()?, t.num()?];
                    raw.nodes.push((tag, p, (3, 0)));
                }
                t.skip_to("$EndNodes")?;
            }
            "$Elements" if version >= 4.0 => {
                let blocks: usize = t.num()?;
                t.next()?;
                t.next()?;
                t.next()?;
                for _ in 0..blocks {
                    let dim: u8 = t.num()?;
                    let tag: u32 = t.num()?;
                    let ty: u32 = t.num()?;
                    let n: usize = t.num()?;
                    let k = nodes_of(ty)?;
                    for _ in 0..n {
                        t.next()?;
                        let vs: Vec<usize> = (0..k)
                            .map(|_| t.num())
                            .collect::<std::result::Result<_, _>>()?;
                        raw.elements.push((ty, (dim, tag), vs));
                    }
                }
                t.skip_to("$EndElements")?;
            }
            "$Elements" => {
                let n: usize = t.num()?;
                for _ in 0..n {
                    t.next()?;
                    let ty: u32 = t.num()?;
                    let ntags: usize = t.num()?;
                    let tags: Vec<u32> = (0..ntags)
                        .map(|_| t.num())
                        .collect::<std::result::Result<_, _>>()?;
                    let k = nodes_of(ty)?;
                    let vs: Vec<usize> = (0..k)
                        .map(|_| t.num())
                        .collect::<std::result::Result<_, _>>()?;
                    let dim = dim_of(ty);
                    let entity = (dim, tags.get(1).copied().unwrap_or(0));
                    if let Some(&p) = tags.first() {
                        let g = old_groups.entry(entity).or_default();
                        if p != 0 && !g.contains(&p) {
                            g.push(p);
                        }
                    }
                    raw.elements.push((ty, entity, vs));
                }
                t.skip_to("$EndElements")?;
            }
            s if s.starts_with('$') && !s.starts_with("$End") => {
                let end = format!("$End{}", &s[1..]);
                t.skip_to(&end)?;
            }
            _ => {}
        }
    }
    if version == 0.0 {
        return Err("no $MeshFormat: not an MSH file".into());
    }
    if version == 2.2 {
        raw.entity_groups = old_groups;
    }
    Ok(raw)
}

/// Nodes per element of a gmsh type (the linear ones, which a volume mesh
/// needs).
fn nodes_of(ty: u32) -> std::result::Result<usize, String> {
    Ok(match ty {
        15 => 1,
        1 => 2,
        2 => 3,
        4 => 4,
        _ => {
            return Err(format!(
                "element type {ty} is not read (points, lines, triangles, tets)"
            ))
        }
    })
}

fn dim_of(ty: u32) -> u8 {
    match ty {
        15 => 0,
        1 => 1,
        2 => 2,
        _ => 3,
    }
}

fn build(raw: Raw) -> std::result::Result<Mesh, String> {
    // Points in node order.
    let mut nodes = raw.nodes;
    nodes.sort_by_key(|n| n.0);
    let index: HashMap<usize, usize> = nodes.iter().enumerate().map(|(i, n)| (n.0, i)).collect();
    let at = |tag: usize| -> std::result::Result<usize, String> {
        index
            .get(&tag)
            .copied()
            .ok_or(format!("element node {tag} is not in $Nodes"))
    };
    let points: Vec<[f64; 3]> = nodes.iter().map(|n| n.1).collect();
    let point_class: Vec<PointClass> = nodes
        .iter()
        .map(|n| match n.2 {
            (0, t) if t > 0 => PointClass::Vertex(t - 1),
            (1, t) if t > 0 => PointClass::Edge(t - 1),
            (2, t) if t > 0 => PointClass::Face(t - 1),
            _ => PointClass::Interior,
        })
        .collect();
    let groups = |e: (u8, u32)| raw.entity_groups.get(&e).cloned().unwrap_or_default();

    let (mut tets, mut tet_regions) = (Vec::new(), Vec::new());
    let (mut tris, mut lines) = (Vec::new(), Vec::new());
    for (ty, e, vs) in &raw.elements {
        let vs: Vec<usize> = vs
            .iter()
            .map(|&v| at(v))
            .collect::<std::result::Result<_, _>>()?;
        match ty {
            4 => {
                if e.1 == 0 {
                    return Err("a tet outside every volume entity".into());
                }
                // In the mesher's orientation (orient3d positive, the mirror
                // of gmsh's), whatever the file's.
                let [a, b, c, d] = [vs[0], vs[1], vs[2], vs[3]].map(|v| points[v]);
                let (u, w, z) = (
                    [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                    [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
                    [d[0] - a[0], d[1] - a[1], d[2] - a[2]],
                );
                let vol = u[0] * (w[1] * z[2] - w[2] * z[1]) - u[1] * (w[0] * z[2] - w[2] * z[0])
                    + u[2] * (w[0] * z[1] - w[1] * z[0]);
                tets.push(if vol > 0.0 {
                    [vs[0], vs[1], vs[3], vs[2]]
                } else {
                    [vs[0], vs[1], vs[2], vs[3]]
                });
                tet_regions.push(RegionTag(e.1));
            }
            2 => tris.push(([vs[0], vs[1], vs[2]], e.1)),
            1 => lines.push(CurveEdge {
                v: [vs[0], vs[1]],
                edge: e.1.saturating_sub(1),
            }),
            _ => {}
        }
    }
    if tets.is_empty() {
        return Err("the file has no tets".into());
    }

    // Every tet face by its sorted corners, with the tets on it.
    let key = |t: [usize; 3]| {
        let mut k = t;
        k.sort_unstable();
        k
    };
    let mut on: HashMap<[usize; 3], Vec<(usize, [usize; 3])>> = HashMap::new();
    for (i, t) in tets.iter().enumerate() {
        for skip in 0..4 {
            let f: Vec<usize> = (0..4).filter(|&k| k != skip).map(|k| t[k]).collect();
            // Wound to face out of the tet: away from the corner opposite.
            let (a, b, c, d) = (points[f[0]], points[f[1]], points[f[2]], points[t[skip]]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let toward = (0..3).map(|k| n[k] * (d[k] - a[k])).sum::<f64>();
            let out = if toward > 0.0 {
                [f[0], f[2], f[1]]
            } else {
                [f[0], f[1], f[2]]
            };
            on.entry(key(out)).or_default().push((i, out));
        }
    }
    // The regions in front of (the triangle's normal) and behind a face.
    let sides = |tri: [usize; 3]| -> [RegionTag; 2] {
        let mut s = [RegionTag(0); 2];
        for &(i, out) in on.get(&key(tri)).map(Vec::as_slice).unwrap_or(&[]) {
            // The tet lies behind its outward winding.
            let same = (0..3).any(|r| [out[r], out[(r + 1) % 3], out[(r + 2) % 3]] == tri);
            s[usize::from(!same)] = tet_regions[i];
        }
        [s[1], s[0]]
    };
    let mut faces = Vec::new();
    let mut given: std::collections::HashSet<[usize; 3]> = Default::default();
    for (tri, entity) in &tris {
        let gs = groups((2, *entity));
        faces.push(SurfaceFace {
            tri: *tri,
            face_tag: FaceTag(gs.first().copied().unwrap_or(0)),
            regions: sides(*tri),
            patch: entity.saturating_sub(1),
            surface: 0,
        });
        given.insert(key(*tri));
    }
    // Interfaces and boundary the file has no triangles on.
    let mut rest: Vec<_> = on
        .iter()
        .filter(|(k, ts)| {
            !given.contains(*k) && (ts.len() == 1 || tet_regions[ts[0].0] != tet_regions[ts[1].0])
        })
        .map(|(_, ts)| ts[0].1)
        .collect();
    rest.sort_unstable();
    for tri in rest {
        faces.push(SurfaceFace {
            tri,
            face_tag: FaceTag(0),
            regions: sides(tri),
            patch: u32::MAX,
            surface: 0,
        });
    }

    // Names: a region per volume entity, labelled by its first group; the
    // first group of a surface its face tag, the others named sets.
    let name = |dim: u8, tag: u32| raw.names.get(&(dim, tag)).cloned();
    let mut regions: Vec<u32> = tet_regions.iter().map(|r| r.0).collect();
    regions.sort_unstable();
    regions.dedup();
    let mut labels = Labels::default();
    for &r in &regions {
        let label = groups((3, r)).first().and_then(|&p| name(3, p));
        labels.solids.push(SolidInfo {
            region: r,
            label,
            roles: Vec::new(),
        });
    }
    let mut face_sets: BTreeMap<(u8, u32), Vec<u32>> = BTreeMap::new();
    let mut entities: Vec<&(u8, u32)> = raw.entity_groups.keys().collect();
    entities.sort_unstable();
    for &&(dim, tag) in &entities {
        let gs = groups((dim, tag));
        match dim {
            2 => {
                if let Some(&first) = gs.first() {
                    if let Some(n) = name(2, first) {
                        labels.tag_labels.insert(first, n);
                    }
                }
                for &p in gs.iter().skip(1) {
                    face_sets.entry((2, p)).or_default().push(tag - 1);
                }
            }
            1 => {
                for &p in &gs {
                    face_sets.entry((1, p)).or_default().push(tag - 1);
                }
            }
            _ => {}
        }
    }
    for ((dim, p), ids) in face_sets {
        let n = name(dim, p).unwrap_or(format!("group_{dim}_{p}"));
        let list = if dim == 2 {
            &mut labels.face_names
        } else {
            &mut labels.edge_names
        };
        list.push((n, ids));
    }

    let n = points.len();
    let inner = TetMesh {
        points,
        tets,
        tet_regions,
        faces,
        surfaces: vec![SurfaceKind::Plane],
        surface_owners: vec![SHEET_OWNER],
        plc_points: 0,
        point_size: vec![0.0; n],
        point_class,
        curve_edges: lines,
        periodic_points: Vec::new(),
        contact_faces: Vec::new(),
    };
    let quality = quality_stats(&inner);
    Ok(Mesh::new(inner, quality, labels, Run::default(), None))
}
