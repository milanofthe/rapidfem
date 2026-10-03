//! Finite element output beyond the linear tets: the second-order mesh
//! (tet10, its boundary as tri6) with its mid-edge nodes on the true
//! geometry, and the CalculiX / Abaqus input file.
//!
//! Node order is that of Abaqus C3D10 and VTK's quadratic tetra: the four
//! corners, then the mid-edge nodes of (0,1), (1,2), (2,0), (0,3), (1,3),
//! (2,3); a tri6 its corners, then (0,1), (1,2), (2,0).

use crate::mesh::Mesh;
use rapidmesh_brep::Surface;
use rapidmesh_geom::vec3::{cross, dot, sub};
use rapidmesh_geom::SurfaceKind;
use std::collections::{BTreeMap, HashMap};
use std::io::{self, Write};
use std::path::Path;

type P = [f64; 3];

pub use rapidmesh_topo::TET10_EDGES;

/// A second-order tet mesh.
#[derive(Debug, Clone)]
pub struct SecondOrder {
    /// The corners (the linear mesh's points, in its order), then the
    /// mid-edge nodes.
    pub points: Vec<P>,
    /// Ten nodes per tet, positively oriented (parallel to the linear tets).
    pub tets: Vec<[u32; 10]>,
    /// Six nodes per surface triangle (parallel to the linear mesh's faces).
    pub faces: Vec<[u32; 6]>,
    /// Per tet (parallel to `tets`): whether a mid-edge node of it lies off
    /// its chord, so its map from the reference tet is quadratic. Every
    /// other tet has its mid-edge nodes in the middle of its edges, an
    /// affine map as a linear tet: a solver with the geometry order apart
    /// from the order of its basis (curved boundary, straight medium) maps
    /// only these isoparametrically. A face of a curved tet that holds a
    /// curved edge belongs to curved tets only, so straight and curved tets
    /// meet on straight faces.
    pub curved_tets: Vec<bool>,
    /// Mid-edge nodes moved onto a curved surface or curve.
    pub curved: usize,
    /// Of those, the ones put back on their chord to keep a tet valid.
    pub straightened: usize,
}

fn mid(a: P, b: P) -> P {
    std::array::from_fn(|k| 0.5 * (a[k] + b[k]))
}

/// What a mid-edge node must lie on: a curved carrier, or the plane of a
/// flat face triangle (a `Plane` kind may gather several walls, so the
/// triangle's own plane is the one to keep to).
enum OnSurface {
    Curved(Surface),
    Plane(P, P),
}

impl OnSurface {
    fn project(&self, p: P) -> P {
        match self {
            OnSurface::Curved(s) => s.closest(p).0,
            OnSurface::Plane(o, n) => {
                let d = dot(sub(p, *o), *n);
                std::array::from_fn(|k| p[k] - d * n[k])
            }
        }
    }
}

/// The determinant of the Jacobian of a tet10 at barycentric `l`.
fn jacobian(x: &[P; 10], l: [f64; 4]) -> f64 {
    // dN/dL for the ten shape functions, then the chain rule with
    // L0 = 1 - xi - eta - zeta.
    let mut dl = [[0.0; 3]; 4]; // d x / d L_i
    for i in 0..4 {
        let c = 4.0 * l[i] - 1.0;
        for k in 0..3 {
            dl[i][k] += c * x[i][k];
        }
    }
    for (e, &[i, j]) in TET10_EDGES.iter().enumerate() {
        for k in 0..3 {
            dl[i][k] += 4.0 * l[j] * x[4 + e][k];
            dl[j][k] += 4.0 * l[i] * x[4 + e][k];
        }
    }
    let col = |m: usize| -> P { std::array::from_fn(|k| dl[m][k] - dl[0][k]) };
    let (a, b, c) = (col(1), col(2), col(3));
    dot(a, cross(b, c))
}

/// Whether a tet10 is valid: its Jacobian at the corners, the mid-edges,
/// the face centres and the centre at least `share` of the straight tet's.
fn valid(x: &[P; 10], share: f64) -> bool {
    let corners = [x[0], x[1], x[2], x[3]];
    let lin = dot(
        sub(corners[1], corners[0]),
        cross(sub(corners[2], corners[0]), sub(corners[3], corners[0])),
    );
    let mut samples: Vec<[f64; 4]> = Vec::with_capacity(15);
    for i in 0..4 {
        let mut l = [0.0; 4];
        l[i] = 1.0;
        samples.push(l);
    }
    for &[i, j] in &TET10_EDGES {
        let mut l = [0.0; 4];
        l[i] = 0.5;
        l[j] = 0.5;
        samples.push(l);
    }
    for i in 0..4 {
        let mut l = [1.0 / 3.0; 4];
        l[i] = 0.0;
        samples.push(l);
    }
    samples.push([0.25; 4]);
    samples.iter().all(|&l| jacobian(x, l) >= share * lin)
}

/// A tet's corners in the order Abaqus counts positive: the fourth on the
/// side the first three's normal points to.
fn positive(points: &[P], t: &[usize; 4]) -> [usize; 4] {
    let [a, b, c, d] = t.map(|v| points[v]);
    if dot(cross(sub(b, a), sub(c, a)), sub(d, a)) < 0.0 {
        [t[0], t[2], t[1], t[3]]
    } else {
        *t
    }
}

/// The smallest share of the straight tet's Jacobian a curved tet keeps.
const MIN_JACOBIAN: f64 = 0.2;

impl Mesh {
    /// The second-order mesh: a node in the middle of every edge, on the
    /// true geometry where the edge lies on a curved surface (its closest
    /// point) or on a curve (onto the curve: a rim between flat faces too,
    /// else onto the surfaces in turn).
    /// Where that leaves a tet's Jacobian below a fifth of the straight
    /// tet's anywhere it is sampled, its curved nodes go back on their
    /// chords.
    pub fn second_order(&self) -> SecondOrder {
        let m: &rapidmesh_tet::TetMesh = self;
        let mut points = m.points.clone();
        let key = |a: usize, b: usize| (a.min(b), a.max(b));
        // the surfaces each surface edge lies on
        let mut on: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (fi, f) in m.faces.iter().enumerate() {
            for k in 0..3 {
                on.entry(key(f.tri[k], f.tri[(k + 1) % 3]))
                    .or_default()
                    .push(fi);
            }
        }
        let carrier = |fi: usize| -> Option<OnSurface> {
            let f = &m.faces[fi];
            match &m.surfaces[f.surface as usize] {
                SurfaceKind::Discrete(_) | SurfaceKind::Facets => None,
                SurfaceKind::Plane { .. } => {
                    let [a, b, c] = f.tri.map(|v| m.points[v]);
                    let n = cross(sub(b, a), sub(c, a));
                    let l = dot(n, n).sqrt();
                    (l > 0.0).then(|| OnSurface::Plane(a, n.map(|x| x / l)))
                }
                kind => Surface::curved(kind).map(OnSurface::Curved),
            }
        };
        let mut node: HashMap<(usize, usize), u32> = HashMap::new();
        // per mid-edge node: its chord's middle, and whether it moved
        let mut chord: Vec<(u32, P)> = Vec::new();
        // The B-rep curve each mesh edge on one lies on, and the projection
        // onto the smooth ones.
        let on_curve: HashMap<(usize, usize), u32> = m
            .curve_edges
            .iter()
            .map(|ce| (key(ce.v[0], ce.v[1]), ce.edge))
            .collect();
        let project = self
            .model
            .as_deref()
            .map(|model| rapidmesh_tet::edge_projection(&model.brep));
        // The mid-edge node of the edge `k`: the middle of its chord, moved
        // onto the curve the edge lies on (a rim between flat faces too), or
        // onto every curved surface it lies on, in turn.
        let new_node = |k: (usize, usize), points: &mut Vec<P>, chord: &mut Vec<(u32, P)>| {
            let c = mid(m.points[k.0], m.points[k.1]);
            let curve = on_curve
                .get(&k)
                .and_then(|&e| project.as_ref().and_then(|pr| pr(e, c)));
            if let Some(p) = curve {
                let id = points.len() as u32;
                points.push(p);
                if p != c {
                    chord.push((id, c));
                }
                return id;
            }
            let mut p = c;
            let mut kinds: Vec<OnSurface> = Vec::new();
            let mut curved = false;
            for &fi in on.get(&k).into_iter().flatten() {
                if let Some(s) = carrier(fi) {
                    curved |= matches!(s, OnSurface::Curved(_));
                    kinds.push(s);
                }
            }
            if curved {
                for _ in 0..8 {
                    for s in &kinds {
                        p = s.project(p);
                    }
                }
            }
            let id = points.len() as u32;
            points.push(p);
            if p != c {
                chord.push((id, c));
            }
            id
        };
        let mut tets: Vec<[u32; 10]> = Vec::with_capacity(m.tets.len());
        for t in &m.tets {
            let t = positive(&m.points, t);
            let mut n10 = [0u32; 10];
            for i in 0..4 {
                n10[i] = t[i] as u32;
            }
            for (e, &[i, j]) in TET10_EDGES.iter().enumerate() {
                let k = key(t[i], t[j]);
                n10[4 + e] = *node
                    .entry(k)
                    .or_insert_with(|| new_node(k, &mut points, &mut chord));
            }
            tets.push(n10);
        }
        // A face edge no tet has (a sheet outside every region) takes its
        // node here, with no tet to keep valid.
        let faces: Vec<[u32; 6]> = m
            .faces
            .iter()
            .map(|f| {
                let [a, b, c] = f.tri;
                let mut at = |k: (usize, usize)| {
                    *node
                        .entry(k)
                        .or_insert_with(|| new_node(k, &mut points, &mut chord))
                };
                [
                    a as u32,
                    b as u32,
                    c as u32,
                    at(key(a, b)),
                    at(key(b, c)),
                    at(key(c, a)),
                ]
            })
            .collect();
        let curved = chord.len();
        let on_chord: HashMap<u32, P> = chord.into_iter().collect();
        let mut straightened = 0;
        for _ in 0..4 {
            let mut back: Vec<u32> = Vec::new();
            for t in &tets {
                let x: [P; 10] = t.map(|v| points[v as usize]);
                if !valid(&x, MIN_JACOBIAN) {
                    back.extend(
                        t[4..]
                            .iter()
                            .copied()
                            .filter(|v| on_chord.get(v).is_some_and(|&c| points[*v as usize] != c)),
                    );
                }
            }
            if back.is_empty() {
                break;
            }
            back.sort_unstable();
            back.dedup();
            straightened += back.len();
            for v in back {
                points[v as usize] = on_chord[&v];
            }
        }
        let off_chord = |v: &u32| on_chord.get(v).is_some_and(|&c| points[*v as usize] != c);
        let curved_tets = tets.iter().map(|t| t[4..].iter().any(off_chord)).collect();
        SecondOrder {
            points,
            tets,
            faces,
            curved_tets,
            curved,
            straightened,
        }
    }

    /// Writes the second-order mesh as a gmsh MSH 4.1 file (see
    /// [`Mesh::write_msh`]): lines with three nodes, triangles with six,
    /// tets with ten.
    pub fn write_msh_second_order(
        &self,
        so: &SecondOrder,
        path: impl AsRef<Path>,
    ) -> io::Result<()> {
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        let o = rapidmesh_topo::export::Order2 {
            points: &so.points,
            tets: &so.tets,
            faces: &so.faces,
        };
        rapidmesh_topo::export::write_msh_order2(self, &o, &self.labels.msh_names(true), &mut w)?;
        w.flush()
    }

    /// Writes a CalculiX / Abaqus input file of the linear mesh (C3D4).
    pub fn write_inp(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let m: &rapidmesh_tet::TetMesh = self;
        let tets: Vec<Vec<u32>> = m
            .tets
            .iter()
            .map(|t| positive(&m.points, t).iter().map(|&v| v as u32).collect())
            .collect();
        self.write_inp_of(path.as_ref(), &m.points, &tets, "C3D4")
    }

    /// Writes a CalculiX / Abaqus input file of the second-order mesh
    /// (C3D10).
    pub fn write_inp_second_order(
        &self,
        so: &SecondOrder,
        path: impl AsRef<Path>,
    ) -> io::Result<()> {
        let tets: Vec<Vec<u32>> = so.tets.iter().map(|t| t.to_vec()).collect();
        self.write_inp_of(path.as_ref(), &so.points, &tets, "C3D10")
    }

    /// The input file: nodes; elements per region group (`ELSET`); per
    /// named face set (named geometric faces, named sheet tags) its nodes
    /// (`NSET`) and its faces on the boundary as element faces (`SURFACE`).
    fn write_inp_of(
        &self,
        path: &Path,
        points: &[P],
        tets: &[Vec<u32>],
        kind: &str,
    ) -> io::Result<()> {
        let m: &rapidmesh_tet::TetMesh = self;
        let v = self.view();
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(
            w,
            "** rapidmesh: {} nodes, {} {kind} elements",
            points.len(),
            tets.len()
        )?;
        writeln!(w, "*NODE")?;
        for (i, p) in points.iter().enumerate() {
            writeln!(w, "{}, {:?}, {:?}, {:?}", i + 1, p[0], p[1], p[2])?;
        }
        let groups = self.labels.region_groups();
        let mut done = vec![false; tets.len()];
        for (name, rs) in &groups {
            let ids: Vec<usize> = (0..tets.len())
                .filter(|&t| rs.contains(&m.tet_regions[t].0))
                .collect();
            if ids.is_empty() {
                continue;
            }
            writeln!(w, "*ELEMENT, TYPE={kind}, ELSET={}", inp_name(name))?;
            for t in ids {
                done[t] = true;
                write_element(&mut w, t, &tets[t])?;
            }
        }
        if done.iter().any(|d| !d) {
            writeln!(w, "*ELEMENT, TYPE={kind}, ELSET=rest")?;
            for t in (0..tets.len()).filter(|&t| !done[t]) {
                write_element(&mut w, t, &tets[t])?;
            }
        }
        // named face sets over the topology's faces
        let nf = v.topo.faces.len();
        let mut named: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (name, ids) in &self.labels.face_names {
            named
                .entry(inp_name(name))
                .or_default()
                .extend((0..nf).filter(|&f| ids.contains(&v.class.face_patch[f])));
        }
        for (&tag, name) in &self.labels.tag_labels {
            named
                .entry(inp_name(name))
                .or_default()
                .extend((0..nf).filter(|&f| v.class.face_tag[f] == tag));
        }
        for (name, faces) in &named {
            // the nodes: corners, and the mid-edge nodes of a second-order mesh
            let mut nodes: Vec<u32> = Vec::new();
            let mut sides: Vec<(usize, usize)> = Vec::new();
            for &f in faces {
                let t = v.topo.face_tets[f][0] as usize;
                let corners = v.topo.faces[f];
                // the side of tet `t` it is: Abaqus numbers a tet's faces
                // by their corners (1,2,3), (1,4,2), (2,4,3), (3,4,1)
                let tet = &tets[t];
                let missing = (0..4).find(|&i| !corners.contains(&tet[i])).unwrap_or(3);
                let side = match missing {
                    3 => 1,
                    2 => 2,
                    0 => 3,
                    _ => 4,
                };
                for &i in &[[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]][side - 1] {
                    nodes.push(tet[i]);
                }
                if tet.len() == 10 {
                    for (e, &[i, j]) in TET10_EDGES.iter().enumerate() {
                        if i != missing && j != missing {
                            nodes.push(tet[4 + e]);
                        }
                    }
                }
                if v.topo.face_tets[f][1] == rapidmesh_topo::NONE {
                    sides.push((t, side));
                }
            }
            nodes.sort_unstable();
            nodes.dedup();
            writeln!(w, "*NSET, NSET={name}")?;
            for chunk in nodes.chunks(16) {
                let line: Vec<String> = chunk.iter().map(|n| (n + 1).to_string()).collect();
                writeln!(w, "{}", line.join(", "))?;
            }
            if !sides.is_empty() {
                writeln!(w, "*SURFACE, NAME={name}, TYPE=ELEMENT")?;
                for (t, s) in sides {
                    writeln!(w, "{}, S{s}", t + 1)?;
                }
            }
        }
        w.flush()
    }
}

impl SecondOrder {
    /// The volume of every tet, the Jacobian integrated exactly (it is a
    /// cubic; the 11-point Keast rule integrates degree 4).
    pub fn volumes(&self) -> Vec<f64> {
        // Keast: (weight, barycentric) on the reference tet of volume 1/6
        let a = 0.071_428_571_428_571_43;
        let b = 0.785_714_285_714_285_7;
        let c = 0.399_403_576_166_799_2;
        let d = 0.100_596_423_833_200_8;
        let mut rule: Vec<(f64, [f64; 4])> = vec![(-0.013_155_555_555_555_56, [0.25; 4])];
        for i in 0..4 {
            let mut l = [a; 4];
            l[i] = b;
            rule.push((0.007_622_222_222_222_222, l));
        }
        for &[i, j] in &TET10_EDGES {
            let mut l = [d; 4];
            l[i] = c;
            l[j] = c;
            rule.push((0.024_888_888_888_888_89, l));
        }
        self.tets
            .iter()
            .map(|t| {
                let x: [P; 10] = t.map(|v| self.points[v as usize]);
                rule.iter().map(|&(w, l)| w * jacobian(&x, l)).sum()
            })
            .collect()
    }

    /// Writes a VTK XML unstructured grid of the quadratic tets (cell type
    /// 24) with cell data `region`.
    pub fn write_vtu(&self, regions: &[u32], path: impl AsRef<Path>) -> io::Result<()> {
        use rapidmesh_topo::export::{write_vtu_grid, VTK_QUADRATIC_TET};
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        let cells = self
            .tets
            .iter()
            .map(|t| (t.iter().map(|&v| v as usize).collect(), VTK_QUADRATIC_TET));
        let regions: Vec<i64> = regions.iter().map(|&r| r as i64).collect();
        write_vtu_grid(&mut w, &self.points, cells, &[("region", &regions)])?;
        w.flush()
    }
}

fn write_element(w: &mut impl Write, t: usize, nodes: &[u32]) -> io::Result<()> {
    let ids: Vec<String> = nodes.iter().map(|n| (n + 1).to_string()).collect();
    writeln!(w, "{}, {}", t + 1, ids.join(", "))
}

/// A name as an Abaqus label: letters, digits and `_` kept, anything else
/// `_`, a leading digit behind a `_`.
fn inp_name(name: &str) -> String {
    let mut w: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if !w.starts_with(|c: char| c.is_ascii_alphabetic()) {
        w.insert(0, '_');
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight() -> [P; 10] {
        let c = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut x = [[0.0; 3]; 10];
        x[..4].copy_from_slice(&c);
        for (e, &[i, j]) in TET10_EDGES.iter().enumerate() {
            x[4 + e] = mid(c[i], c[j]);
        }
        x
    }

    #[test]
    fn a_straight_tet10_has_the_linear_jacobian() {
        let x = straight();
        for l in [[1.0, 0.0, 0.0, 0.0], [0.25; 4], [0.0, 0.5, 0.5, 0.0]] {
            assert!((jacobian(&x, l) - 1.0).abs() < 1e-12);
        }
        assert!(valid(&x, MIN_JACOBIAN));
    }

    #[test]
    fn a_mid_node_pulled_across_is_invalid() {
        let mut x = straight();
        // the node of edge (0,1) pulled far past the opposite corners
        x[4] = [0.5, 1.5, 1.5];
        assert!(!valid(&x, MIN_JACOBIAN));
        // pulled a little outward it stays valid
        let mut y = straight();
        y[4] = [0.5, -0.05, -0.05];
        assert!(valid(&y, MIN_JACOBIAN));
    }
}
