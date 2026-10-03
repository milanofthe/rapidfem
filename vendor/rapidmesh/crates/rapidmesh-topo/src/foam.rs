//! Finite volume output of a tet mesh: an OpenFOAM `polyMesh`, with the tets
//! as cells or with polyhedral cells (the median dual: one cell per vertex
//! and region), and the finite volume quality measures as OpenFOAM's
//! `checkMesh` reports them.
//!
//! A [`PolyMesh`] is cells bounded by polygonal faces, each face with its
//! owner, its neighbour (or patch, on the boundary) and its corners turned
//! so its normal points from owner to neighbour, out of the mesh on the
//! boundary. Writing puts the internal faces first in upper-triangular order
//! (by owner, then neighbour, owner the smaller cell), then the boundary
//! faces patch after patch.

use crate::math::{add, cross, dot, norm, scale, sub};
use crate::{TetTopology, NONE};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

type P = [f64; 3];

/// Cells bounded by polygonal faces.
#[derive(Debug, Clone, Default)]
pub struct PolyMesh {
    pub points: Vec<P>,
    /// The corners of every face, normal from owner to neighbour (out of the
    /// mesh on the boundary).
    pub faces: Vec<Vec<u32>>,
    pub owner: Vec<u32>,
    /// The neighbour of every face, [`NONE`] on the boundary.
    pub neighbour: Vec<u32>,
    /// The patch of every boundary face, [`NONE`] inside.
    pub patch: Vec<u32>,
    /// The zone (region) of every cell.
    pub cell_zone: Vec<u32>,
}

/// A named set of cells or faces (ids of the cells or faces of a
/// [`PolyMesh`]).
#[derive(Debug, Clone)]
pub struct FoamZone {
    pub name: String,
    pub ids: Vec<u32>,
}

/// The finite volume quality per face, as `checkMesh` measures it: the
/// non-orthogonality (degrees between a face's area vector and the line
/// from its owner's centre to its neighbour's) and the skewness (how far
/// the face centre lies from where that line crosses the face, over the
/// line's length). Boundary faces are 0 in both.
#[derive(Debug, Clone, Default)]
pub struct FvmQuality {
    pub non_orthogonality: Vec<f64>,
    pub skewness: Vec<f64>,
    pub max_non_orthogonality: f64,
    pub mean_non_orthogonality: f64,
    pub max_skewness: f64,
    /// Internal faces above `checkMesh`'s 70 degrees.
    pub severely_non_orthogonal: usize,
    /// The largest cell openness: the sum of a cell's area vectors (turned
    /// out of it) over the sum of their magnitudes, 0 for a closed cell
    /// (`checkMesh` fails above 1e-6).
    pub max_openness: f64,
}

fn mean(points: &[P], ids: &[u32]) -> P {
    let n = ids.len() as f64;
    ids.iter()
        .fold([0.0; 3], |c, &i| add(c, scale(points[i as usize], 1.0 / n)))
}

/// A face's centre and area vector as OpenFOAM computes them: triangles
/// fanned from the corners' mean, the centre their area-weighted centroid.
fn face_geometry(points: &[P], face: &[u32]) -> (P, P) {
    if face.len() == 3 {
        let [a, b, c] = [0, 1, 2].map(|k| points[face[k] as usize]);
        let s = scale(cross(sub(b, a), sub(c, a)), 0.5);
        return (scale(add(add(a, b), c), 1.0 / 3.0), s);
    }
    let m = mean(points, face);
    let (mut area, mut centre, mut weight) = ([0.0; 3], [0.0; 3], 0.0);
    for k in 0..face.len() {
        let (a, b) = (
            points[face[k] as usize],
            points[face[(k + 1) % face.len()] as usize],
        );
        let s = scale(cross(sub(a, m), sub(b, m)), 0.5);
        let w = norm(s);
        area = add(area, s);
        centre = add(centre, scale(add(add(a, b), m), w / 3.0));
        weight += w;
    }
    let centre = if weight > 0.0 {
        scale(centre, 1.0 / weight)
    } else {
        m
    };
    (centre, area)
}

impl PolyMesh {
    /// The number of cells.
    pub fn n_cells(&self) -> usize {
        self.cell_zone.len()
    }

    /// Every cell's centre and volume, by pyramids from the mean of its face
    /// centres to its faces.
    pub fn cells(&self) -> (Vec<P>, Vec<f64>) {
        let n = self.n_cells();
        let geo: Vec<(P, P)> = self
            .faces
            .iter()
            .map(|f| face_geometry(&self.points, f))
            .collect();
        let (mut guess, mut count) = (vec![[0.0; 3]; n], vec![0.0f64; n]);
        for (f, &(c, _)) in geo.iter().enumerate() {
            for cell in [self.owner[f], self.neighbour[f]] {
                if cell != NONE {
                    guess[cell as usize] = add(guess[cell as usize], c);
                    count[cell as usize] += 1.0;
                }
            }
        }
        for c in 0..n {
            guess[c] = scale(guess[c], 1.0 / count[c].max(1.0));
        }
        let (mut centre, mut volume) = (vec![[0.0; 3]; n], vec![0.0f64; n]);
        for (f, &(fc, s)) in geo.iter().enumerate() {
            // the face's area vector points out of the owner, into the neighbour
            for (cell, sign) in [(self.owner[f], 1.0), (self.neighbour[f], -1.0)] {
                if cell == NONE {
                    continue;
                }
                let c = cell as usize;
                let v = sign * dot(s, sub(fc, guess[c])) / 3.0;
                let at = add(scale(fc, 0.75), scale(guess[c], 0.25));
                centre[c] = add(centre[c], scale(at, v));
                volume[c] += v;
            }
        }
        for c in 0..n {
            centre[c] = if volume[c] != 0.0 {
                scale(centre[c], 1.0 / volume[c])
            } else {
                guess[c]
            };
        }
        (centre, volume)
    }

    /// The finite volume quality of every face.
    pub fn quality(&self) -> FvmQuality {
        let (cc, _) = self.cells();
        let nf = self.faces.len();
        let mut q = FvmQuality {
            non_orthogonality: vec![0.0; nf],
            skewness: vec![0.0; nf],
            ..Default::default()
        };
        let (mut sum_s, mut sum_a) = (vec![[0.0; 3]; self.n_cells()], vec![0.0f64; self.n_cells()]);
        for f in 0..nf {
            let (_, s) = face_geometry(&self.points, &self.faces[f]);
            for (cell, sign) in [(self.owner[f], 1.0), (self.neighbour[f], -1.0)] {
                if cell != NONE {
                    sum_s[cell as usize] = add(sum_s[cell as usize], scale(s, sign));
                    sum_a[cell as usize] += norm(s);
                }
            }
        }
        q.max_openness = (0..self.n_cells())
            .map(|c| norm(sum_s[c]) / sum_a[c].max(f64::MIN_POSITIVE))
            .fold(0.0, f64::max);
        let (mut sum, mut internal) = (0.0, 0usize);
        for f in 0..nf {
            if self.neighbour[f] == NONE {
                continue;
            }
            let (fc, s) = face_geometry(&self.points, &self.faces[f]);
            let (co, cn) = (cc[self.owner[f] as usize], cc[self.neighbour[f] as usize]);
            let d = sub(cn, co);
            let cos = (dot(d, s) / (norm(d) * norm(s)).max(f64::MIN_POSITIVE)).clamp(-1.0, 1.0);
            let angle = cos.acos().to_degrees();
            // where the line between the centres crosses the face, weighted
            // by their distances from the face plane
            let d_own = dot(sub(fc, co), s).abs();
            let d_nei = dot(sub(fc, cn), s).abs();
            let w = d_nei / (d_own + d_nei).max(f64::MIN_POSITIVE);
            let at = add(scale(co, w), scale(cn, 1.0 - w));
            let skew = norm(sub(fc, at)) / norm(d).max(f64::MIN_POSITIVE);
            q.non_orthogonality[f] = angle;
            q.skewness[f] = skew;
            q.max_non_orthogonality = q.max_non_orthogonality.max(angle);
            q.max_skewness = q.max_skewness.max(skew);
            q.severely_non_orthogonal += (angle > 70.0) as usize;
            sum += angle;
            internal += 1;
        }
        q.mean_non_orthogonality = sum / internal.max(1) as f64;
        q
    }

    /// The tets as cells, the tet faces as faces. `zone` is every tet's
    /// zone, `patch` every topology face's patch (read on the boundary only).
    pub fn from_tets(points: &[P], topo: &TetTopology, zone: &[u32], patch: &[u32]) -> PolyMesh {
        let cc: Vec<P> = topo.tets.iter().map(|t| mean(points, t)).collect();
        let mut m = PolyMesh {
            points: points.to_vec(),
            cell_zone: zone.to_vec(),
            ..Default::default()
        };
        for (f, &[t0, t1]) in topo.face_tets.iter().enumerate() {
            let own = if t1 == NONE { t0 } else { t0.min(t1) };
            let mut face = topo.faces[f].to_vec();
            let (fc, s) = face_geometry(points, &face);
            if dot(s, sub(fc, cc[own as usize])) < 0.0 {
                face.swap(1, 2);
            }
            m.faces.push(face);
            m.owner.push(own);
            m.neighbour.push(if t1 == NONE { NONE } else { t0.max(t1) });
            m.patch.push(if t1 == NONE { patch[f] } else { NONE });
        }
        m
    }

    /// The median dual of the tets: a cell per vertex and zone, the part of
    /// every tet nearest the vertex (each tet split into four by its
    /// centroid, its face centroids and its edge midpoints). Between two
    /// vertices of an edge the face runs through the centroids of the tets
    /// and faces around the edge; on the boundary and between zones each
    /// triangle gives each of its corners the quadrilateral of the corner,
    /// its two edge midpoints and the triangle's centroid. `zone` is every
    /// tet's zone, `patch` every topology face's patch (read on the boundary
    /// only).
    pub fn dual(points: &[P], topo: &TetTopology, zone: &[u32], patch: &[u32]) -> PolyMesh {
        let nf = topo.faces.len();
        // A face the dual stops at: on the boundary, or between zones.
        let breaks = |f: usize| {
            let [t0, t1] = topo.face_tets[f];
            t1 == NONE || zone[t0 as usize] != zone[t1 as usize]
        };
        // cells: one per vertex and zone, numbered in vertex order
        let mut cell_of: HashMap<(u32, u32), u32> = HashMap::new();
        let mut cell_zone: Vec<u32> = Vec::new();
        for v in 0..topo.n_verts {
            let mut zs: Vec<u32> = topo
                .vert_tets
                .row(v)
                .iter()
                .map(|&t| zone[t as usize])
                .collect();
            zs.sort_unstable();
            zs.dedup();
            for z in zs {
                cell_of.insert((v as u32, z), cell_zone.len() as u32);
                cell_zone.push(z);
            }
        }
        let mut dp = DualPoints {
            points,
            topo,
            pts: Vec::new(),
            index: HashMap::new(),
        };
        let mut m = PolyMesh::default();
        // the faces around every edge, and the tets on either side of each
        let mut edge_faces: Vec<Vec<u32>> = vec![Vec::new(); topo.edges.len()];
        for f in 0..nf {
            for &e in &topo.face_edges[f] {
                edge_faces[e as usize].push(f as u32);
            }
        }
        let tet_centre = |t: u32| mean(points, &topo.tets[t as usize]);
        // The face between the cells `a` and `b`: its points turned so the
        // normal runs along `along` from `a` to `b`, owner the smaller cell.
        let push = |m: &mut PolyMesh, mut face: Vec<u32>, pts: &[P], a: u32, b: u32, along: P| {
            let (_, s) = face_geometry(pts, &face);
            if dot(s, along) < 0.0 {
                face.reverse();
            }
            let (own, nei) = if a < b { (a, b) } else { (b, a) };
            if own != a {
                face.reverse();
            }
            m.faces.push(face);
            m.owner.push(own);
            m.neighbour.push(nei);
            m.patch.push(NONE);
        };
        for (e, &[u, v]) in topo.edges.iter().enumerate() {
            let around = &edge_faces[e];
            // tets around the edge, each with its two faces on the edge
            let mut tet_faces: HashMap<u32, Vec<u32>> = HashMap::new();
            for &f in around {
                for &t in topo.face_tets[f as usize].iter().filter(|&&t| t != NONE) {
                    tet_faces.entry(t).or_default().push(f);
                }
            }
            // walk the ring from each face the dual stops at (or from any
            // face where none does), tet by tet through the faces it does not
            // stop at
            let starts: Vec<u32> = {
                let b: Vec<u32> = around
                    .iter()
                    .copied()
                    .filter(|&f| breaks(f as usize))
                    .collect();
                if b.is_empty() {
                    around.first().copied().into_iter().collect()
                } else {
                    b
                }
            };
            let closed = around.iter().all(|&f| !breaks(f as usize));
            let mut seen: std::collections::HashSet<u32> = Default::default();
            for &start in &starts {
                for &t0 in topo.face_tets[start as usize]
                    .iter()
                    .filter(|&&t| t != NONE)
                {
                    if seen.contains(&t0) {
                        continue;
                    }
                    let z = zone[t0 as usize];
                    let mut ring: Vec<u32> = Vec::new();
                    if !closed {
                        ring.push(dp.get(At::Edge(e as u32)));
                    }
                    ring.push(dp.get(At::Face(start)));
                    let (mut t, mut from) = (t0, start);
                    loop {
                        seen.insert(t);
                        ring.push(dp.get(At::Tet(t)));
                        let next = tet_faces[&t]
                            .iter()
                            .copied()
                            .find(|&f| f != from)
                            .unwrap_or(from);
                        if next == start {
                            break;
                        }
                        ring.push(dp.get(At::Face(next)));
                        if breaks(next as usize) {
                            break;
                        }
                        let [a, b] = topo.face_tets[next as usize];
                        let other = if a == t { b } else { a };
                        if other == NONE || seen.contains(&other) {
                            break;
                        }
                        from = next;
                        t = other;
                    }
                    let (cu, cv) = (cell_of[&(u, z)], cell_of[&(v, z)]);
                    let along = sub(points[v as usize], points[u as usize]);
                    push(&mut m, ring, &dp.pts, cu, cv, along);
                }
            }
        }
        // the quadrilaterals of the triangles the dual stops at
        for f in 0..nf {
            if !breaks(f) {
                continue;
            }
            let [t0, t1] = topo.face_tets[f];
            let tri = topo.faces[f];
            let fc = mean(points, &tri);
            // the side of t0 the triangle faces away from
            let out = sub(fc, tet_centre(t0));
            for k in 0..3 {
                let (v, a, b) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
                let edge = |x: u32, y: u32| -> u32 {
                    topo.face_edges[f]
                        .iter()
                        .copied()
                        .find(|&e| {
                            let [p, q] = topo.edges[e as usize];
                            (p == x && q == y) || (p == y && q == x)
                        })
                        .expect("edge of the face")
                };
                let mut quad = vec![
                    dp.get(At::Vertex(v)),
                    dp.get(At::Edge(edge(v, a))),
                    dp.get(At::Face(f as u32)),
                    dp.get(At::Edge(edge(v, b))),
                ];
                let c0 = cell_of[&(v, zone[t0 as usize])];
                if t1 == NONE {
                    let (_, s) = face_geometry(&dp.pts, &quad);
                    if dot(s, out) < 0.0 {
                        quad.reverse();
                    }
                    m.faces.push(quad);
                    m.owner.push(c0);
                    m.neighbour.push(NONE);
                    m.patch.push(patch[f]);
                } else {
                    let c1 = cell_of[&(v, zone[t1 as usize])];
                    push(&mut m, quad, &dp.pts, c0, c1, out);
                }
            }
        }
        m.points = dp.pts;
        m.cell_zone = cell_zone;
        m
    }
}

/// A point of the median dual: a vertex, or the centroid of an edge, a
/// face or a tet.
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
enum At {
    Vertex(u32),
    Edge(u32),
    Face(u32),
    Tet(u32),
}

/// The dual's points, made as they are first used.
struct DualPoints<'a> {
    points: &'a [P],
    topo: &'a TetTopology,
    pts: Vec<P>,
    index: HashMap<At, u32>,
}

impl DualPoints<'_> {
    fn get(&mut self, at: At) -> u32 {
        if let Some(&i) = self.index.get(&at) {
            return i;
        }
        let (points, topo) = (self.points, self.topo);
        let p = match at {
            At::Vertex(v) => points[v as usize],
            At::Edge(e) => mean(points, &topo.edges[e as usize]),
            At::Face(f) => mean(points, &topo.faces[f as usize]),
            At::Tet(t) => mean(points, &topo.tets[t as usize]),
        };
        self.pts.push(p);
        let i = self.pts.len() as u32 - 1;
        self.index.insert(at, i);
        i
    }
}

/// The FoamFile header of a `constant/polyMesh` file.
fn header(w: &mut impl Write, class: &str, object: &str, note: Option<&str>) -> io::Result<()> {
    writeln!(w, "FoamFile\n{{")?;
    writeln!(w, "    format      ascii;")?;
    writeln!(w, "    class       {class};")?;
    if let Some(n) = note {
        writeln!(w, "    note        \"{n}\";")?;
    }
    writeln!(w, "    location    \"constant/polyMesh\";")?;
    writeln!(w, "    object      {object};")?;
    writeln!(w, "}}\n")
}

fn file(dir: &Path, name: &str) -> io::Result<io::BufWriter<std::fs::File>> {
    Ok(io::BufWriter::new(std::fs::File::create(dir.join(name))?))
}

/// Writes `points`, `faces`, `owner`, `neighbour`, `boundary` and the zones
/// of `mesh` into `dir` (the case's `constant/polyMesh`, created if
/// missing). `patches` names the patches by index; every boundary face must
/// be in one, empty patches are left out. Face zones name faces of `mesh`.
pub fn write_poly_mesh(
    dir: &Path,
    mesh: &PolyMesh,
    patches: &[String],
    cell_zones: &[FoamZone],
    face_zones: &[FoamZone],
) -> io::Result<()> {
    let invalid = |m: String| io::Error::new(io::ErrorKind::InvalidInput, m);
    let nf = mesh.faces.len();
    // (boundary, owner or patch, neighbour or owner, face): internal faces
    // first by owner and neighbour, then the boundary by patch and owner.
    let mut order: Vec<(bool, u32, u32, u32)> = Vec::with_capacity(nf);
    for f in 0..nf {
        if mesh.neighbour[f] == NONE {
            let p = mesh.patch[f];
            if p == NONE || p as usize >= patches.len() {
                return Err(invalid(format!("boundary face {f} in no patch")));
            }
            order.push((true, p, mesh.owner[f], f as u32));
        } else {
            if mesh.owner[f] >= mesh.neighbour[f] {
                return Err(invalid(format!("face {f}: owner not below neighbour")));
            }
            order.push((false, mesh.owner[f], mesh.neighbour[f], f as u32));
        }
    }
    order.sort_unstable();
    let internal = order.iter().filter(|o| !o.0).count();
    std::fs::create_dir_all(dir)?;
    let note = format!(
        "nPoints:{} nCells:{} nFaces:{} nInternalFaces:{}",
        mesh.points.len(),
        mesh.n_cells(),
        nf,
        internal
    );

    let mut w = file(dir, "points")?;
    header(&mut w, "vectorField", "points", None)?;
    writeln!(w, "{}\n(", mesh.points.len())?;
    for p in &mesh.points {
        writeln!(w, "({:?} {:?} {:?})", p[0], p[1], p[2])?;
    }
    writeln!(w, ")")?;
    w.flush()?;

    let mut faces = file(dir, "faces")?;
    let mut owner = file(dir, "owner")?;
    let mut neighbour = file(dir, "neighbour")?;
    header(&mut faces, "faceList", "faces", None)?;
    header(&mut owner, "labelList", "owner", Some(&note))?;
    header(&mut neighbour, "labelList", "neighbour", Some(&note))?;
    writeln!(faces, "{nf}\n(")?;
    writeln!(owner, "{nf}\n(")?;
    writeln!(neighbour, "{internal}\n(")?;
    for &(boundary, _, _, f) in &order {
        let face = &mesh.faces[f as usize];
        let ids: Vec<String> = face.iter().map(|v| v.to_string()).collect();
        writeln!(faces, "{}({})", face.len(), ids.join(" "))?;
        writeln!(owner, "{}", mesh.owner[f as usize])?;
        if !boundary {
            writeln!(neighbour, "{}", mesh.neighbour[f as usize])?;
        }
    }
    for w in [&mut faces, &mut owner, &mut neighbour] {
        writeln!(w, ")")?;
        w.flush()?;
    }

    let mut w = file(dir, "boundary")?;
    header(&mut w, "polyBoundaryMesh", "boundary", None)?;
    let used: Vec<(usize, usize)> = (0..patches.len())
        .map(|p| (p, order.iter().filter(|o| o.0 && o.1 == p as u32).count()))
        .filter(|&(_, n)| n > 0)
        .collect();
    writeln!(w, "{}\n(", used.len())?;
    let mut start = internal;
    for (p, n) in used {
        writeln!(w, "    {}\n    {{", patches[p])?;
        writeln!(w, "        type            patch;")?;
        writeln!(w, "        nFaces          {n};")?;
        writeln!(w, "        startFace       {start};")?;
        writeln!(w, "    }}")?;
        start += n;
    }
    writeln!(w, ")")?;
    w.flush()?;

    // Face zones name faces by their place in `faces`.
    let mut at = vec![0u32; nf];
    for (i, o) in order.iter().enumerate() {
        at[o.3 as usize] = i as u32;
    }
    let face_zones: Vec<FoamZone> = face_zones
        .iter()
        .map(|z| FoamZone {
            name: z.name.clone(),
            ids: z.ids.iter().map(|&f| at[f as usize]).collect(),
        })
        .collect();
    for (name, zones) in [("cellZones", cell_zones), ("faceZones", &face_zones[..])] {
        if zones.is_empty() {
            continue;
        }
        let mut w = file(dir, name)?;
        header(&mut w, "regIOobject", name, None)?;
        writeln!(w, "{}\n(", zones.len())?;
        let (kind, labels) = if name == "cellZones" {
            ("cellZone", "cellLabels")
        } else {
            ("faceZone", "faceLabels")
        };
        for z in zones {
            writeln!(w, "{}\n{{\n    type {kind};", z.name)?;
            writeln!(w, "    {labels} List<label> {}\n    (", z.ids.len())?;
            for id in &z.ids {
                writeln!(w, "        {id}")?;
            }
            writeln!(w, "    );")?;
            if name == "faceZones" {
                writeln!(w, "    flipMap List<bool> {} (", z.ids.len())?;
                for _ in &z.ids {
                    writeln!(w, "        0")?;
                }
                writeln!(w, "    );")?;
            }
            writeln!(w, "}}")?;
        }
        writeln!(w, ")")?;
        w.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tets;

    /// A cube split into six tets around its diagonal, the lower half zone
    /// 0, the upper zone 1 when `two`.
    fn cube(two: bool) -> (Vec<P>, TetTopology, Vec<u32>) {
        let points: Vec<P> = (0..8)
            .map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64])
            .collect();
        let tets = [
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [0, 4, 5, 7],
            [0, 5, 1, 7],
        ];
        let topo = TetTopology::build(&Tets {
            tets: &tets,
            n_verts: points.len(),
        });
        let zone = (0..6).map(|t| (two && t >= 3) as u32).collect();
        (points, topo, zone)
    }

    /// Every cell closed (its faces' area vectors, turned out of it, sum to
    /// zero) and the volumes summing to the cube's.
    fn closed(m: &PolyMesh) {
        let mut sum = vec![[0.0; 3]; m.n_cells()];
        for (f, face) in m.faces.iter().enumerate() {
            let (_, s) = face_geometry(&m.points, face);
            sum[m.owner[f] as usize] = add(sum[m.owner[f] as usize], s);
            if m.neighbour[f] != NONE {
                sum[m.neighbour[f] as usize] = sub(sum[m.neighbour[f] as usize], s);
            }
        }
        for s in sum {
            assert!(norm(s) < 1e-12, "{s:?}");
        }
        let (_, vol) = m.cells();
        assert!(vol.iter().all(|&v| v > 0.0), "{vol:?}");
        assert!((vol.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_tets_as_cells_are_closed() {
        let (p, topo, zone) = cube(false);
        let m = PolyMesh::from_tets(&p, &topo, &zone, &vec![0; topo.faces.len()]);
        assert_eq!(m.n_cells(), 6);
        closed(&m);
    }

    #[test]
    fn the_dual_has_a_cell_per_vertex_and_zone() {
        let (p, topo, zone) = cube(false);
        let m = PolyMesh::dual(&p, &topo, &zone, &vec![0; topo.faces.len()]);
        assert_eq!(m.n_cells(), 8);
        closed(&m);
        let (p, topo, zone) = cube(true);
        let m = PolyMesh::dual(&p, &topo, &zone, &vec![0; topo.faces.len()]);
        // the vertices on the interface between the zones (0 and 7 on the
        // diagonal, and 4, 2 on the splitting faces... each zone's own) get a
        // cell in each zone
        let both = (0..8u32)
            .filter(|&v| {
                let zs: std::collections::BTreeSet<u32> = topo
                    .vert_tets
                    .row(v as usize)
                    .iter()
                    .map(|&t| zone[t as usize])
                    .collect();
                zs.len() == 2
            })
            .count();
        assert_eq!(m.n_cells(), 8 + both);
        closed(&m);
    }

    #[test]
    fn the_poly_mesh_lists_internal_faces_first() {
        let (p, topo, zone) = cube(false);
        let m = PolyMesh::dual(&p, &topo, &zone, &vec![0; topo.faces.len()]);
        let dir = std::env::temp_dir().join(format!("rapidmesh_foam_{}", std::process::id()));
        let zones = [FoamZone {
            name: "fluid".into(),
            ids: (0..m.n_cells() as u32).collect(),
        }];
        write_poly_mesh(&dir, &m, &["walls".into()], &zones, &[]).unwrap();
        let read = |n: &str| std::fs::read_to_string(dir.join(n)).unwrap();
        let internal = m.neighbour.iter().filter(|&&n| n != NONE).count();
        assert!(read("neighbour").contains(&format!("{internal}\n(")));
        assert!(read("boundary").contains(&format!("startFace       {internal};")));
        assert!(read("cellZones").contains("fluid"));
        // a boundary face in no patch is refused
        assert!(write_poly_mesh(&dir, &m, &[], &[], &[]).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_regular_pair_has_low_non_orthogonality() {
        let (p, topo, zone) = cube(false);
        let q = PolyMesh::from_tets(&p, &topo, &zone, &vec![0; topo.faces.len()]).quality();
        assert!(
            q.max_non_orthogonality < 70.0,
            "{}",
            q.max_non_orthogonality
        );
        assert_eq!(q.severely_non_orthogonal, 0);
    }
}
