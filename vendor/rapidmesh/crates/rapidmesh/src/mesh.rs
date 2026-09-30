//! What the meshing calls return: the mesh, the labels it carries, the log
//! of the run, and what a solver reads off it (topology, named sets, files,
//! diagnostics).

use rapidmesh_brep::Model;
use rapidmesh_exact::clock::Instant;
use rapidmesh_exact::log::{Event, Level};
use rapidmesh_tet::diagnostics::{Defect, MeshDiagnostics};
use rapidmesh_tet::fidelity::Fidelity;
use rapidmesh_tet::{QualityStats, TetMesh};
use rapidmesh_topo::export::Names;
use rapidmesh_topo::{
    Classification, TetGeometry, TetTopology, TriClassification, TriGeometry, TriTopology, NONE,
};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::{self, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// A solid of the geometry: its region (0 for a void), its label and the
/// names of its faces by role (an empty name for a role without one).
#[derive(Clone, Debug, Serialize)]
pub struct SolidInfo {
    pub region: u32,
    pub label: Option<String>,
    #[serde(skip)]
    pub roles: Vec<String>,
}

/// The names a mesh carries from its geometry.
#[derive(Clone, Debug, Default)]
pub struct Labels {
    /// Per solid (insertion order, voids included).
    pub solids: Vec<SolidInfo>,
    /// Name per sheet tag.
    pub tag_labels: BTreeMap<u32, String>,
    /// Named geometric faces and edges: name and entity ids, in the order
    /// they were named.
    pub face_names: Vec<(String, Vec<u32>)>,
    pub edge_names: Vec<(String, Vec<u32>)>,
}

impl Labels {
    /// The regions per label, in the order of the first solid of each: a
    /// label over several solids is one group, an unlabelled region is
    /// `region_<r>`.
    pub fn region_groups(&self) -> Vec<(String, Vec<u32>)> {
        let mut out: Vec<(String, Vec<u32>)> = Vec::new();
        for s in self.solids.iter().filter(|s| s.region != 0) {
            let name = s
                .label
                .clone()
                .unwrap_or_else(|| format!("region_{}", s.region));
            match out.iter_mut().find(|(n, _)| *n == name) {
                Some((_, rs)) if !rs.contains(&s.region) => rs.push(s.region),
                Some(_) => {}
                None => out.push((name, vec![s.region])),
            }
        }
        out
    }

    /// The physical groups of the MSH file: the region groups (tag = their
    /// lowest region), the named sheet tags, and the named faces and edges
    /// with tags after the sheet tags.
    fn msh_names(&self, regions: bool) -> Names {
        let mut region_groups = HashMap::new();
        if regions {
            for (name, rs) in self.region_groups() {
                let tag = rs.iter().copied().min().unwrap_or(0);
                for r in rs {
                    region_groups.insert(r, (tag, name.clone()));
                }
            }
        }
        let mut tag = self.tag_labels.keys().copied().max().unwrap_or(0) + 1;
        let mut groups = Vec::new();
        for (dim, names) in [(2u8, &self.face_names), (1u8, &self.edge_names)] {
            for (name, ids) in names {
                groups.push((dim, tag, name.clone(), ids.clone()));
                tag += 1;
            }
        }
        Names {
            region_groups,
            face_tags: self
                .tag_labels
                .iter()
                .map(|(&t, n)| (t, n.clone()))
                .collect(),
            groups,
        }
    }
}

/// The record of one meshing run.
#[derive(Clone, Debug, Default)]
pub struct Run {
    /// Wall clock of the whole call.
    pub millis: u64,
    /// Seconds per stage, in pipeline order (`assemble.*`, `mesh.*`).
    pub timings: Vec<(String, f64)>,
    /// Named statistics (predicate calls, counts, quality).
    pub metrics: Vec<(String, f64)>,
    /// The events of the run, in order.
    pub log: Vec<Event>,
}

impl Run {
    /// Collects the log of the run started at `t0`.
    pub(crate) fn finish(t0: Instant) -> Run {
        let (timings, metrics, log) = rapidmesh_exact::log::take();
        Run {
            millis: t0.elapsed().as_millis() as u64,
            timings,
            metrics,
            log,
        }
    }

    /// The warnings and errors of the run.
    pub fn warnings(&self) -> impl Iterator<Item = &Event> {
        self.log
            .iter()
            .filter(|e| matches!(e.level, Level::Warn | Level::Error))
    }

    /// The log, one event per line.
    pub fn log_text(&self) -> String {
        let lines: Vec<String> = self
            .log
            .iter()
            .map(|e| {
                format!(
                    "[{:8.3}s {:>5} {}] {}",
                    e.at,
                    e.level.lower(),
                    e.stage,
                    e.message
                )
            })
            .collect();
        lines.join("\n")
    }
}

/// Named entity sets for boundary conditions, ports and materials, as
/// indices into the solver topology.
#[derive(Clone, Debug, Default)]
pub struct Sets {
    /// Tets per region group (volume meshes only).
    pub cells: BTreeMap<String, Vec<u32>>,
    /// Faces (triangles of a surface mesh) per named sheet tag and named
    /// geometric faces; a volume mesh adds `boundary`, every face with one
    /// tet.
    pub faces: BTreeMap<String, Vec<u32>>,
    /// Edges per named geometric edges.
    pub edges: BTreeMap<String, Vec<u32>>,
    /// Faces per geometric face, edges per geometric edge.
    pub patches: BTreeMap<u32, Vec<u32>>,
    pub curves: BTreeMap<u32, Vec<u32>>,
}

fn indices(n: usize, keep: impl Fn(usize) -> bool) -> Vec<u32> {
    (0..n).filter(|&i| keep(i)).map(|i| i as u32).collect()
}

fn group_by(ids: &[u32]) -> BTreeMap<u32, Vec<u32>> {
    let mut out: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (i, &p) in ids.iter().enumerate().filter(|(_, &p)| p != NONE) {
        out.entry(p).or_default().push(i as u32);
    }
    out
}

/// The named faces and edges, then the per-entity groups, over the faces'
/// geometric face `patch` and the edges' geometric edge `curve`.
fn named_sets(labels: &Labels, patch: &[u32], curve: &[u32], sets: &mut Sets) {
    for (name, ids) in &labels.face_names {
        let f = indices(patch.len(), |i| ids.contains(&patch[i]));
        sets.faces.insert(name.clone(), f);
    }
    for (name, ids) in &labels.edge_names {
        let e = indices(curve.len(), |i| ids.contains(&curve[i]));
        sets.edges.insert(name.clone(), e);
    }
    sets.patches = group_by(patch);
    sets.curves = group_by(curve);
}

/// Boundary fidelity and quality with located defects.
#[derive(Clone, Debug)]
pub struct Diagnostics {
    /// Quality (dihedral histogram, slivers) and conformity (watertight,
    /// non-manifold edges, surface deviation), region volumes.
    pub mesh: MeshDiagnostics,
    /// How faithfully the mesh reproduces its input (none without one).
    pub fidelity: Option<Fidelity>,
}

impl Diagnostics {
    /// Every located defect, of the mesh and of the fidelity check.
    pub fn defects(&self) -> impl Iterator<Item = &Defect> {
        let fid = self.fidelity.iter().flat_map(|f| f.defects.iter());
        self.mesh.defects.iter().chain(fid)
    }
}

/// The solver view of a tet mesh: topology with orientation, the geometric
/// entity of every face and edge, element geometry.
pub struct TetView {
    pub topo: TetTopology,
    pub geom: TetGeometry,
    pub class: Classification,
}

/// The solver view of a surface mesh.
pub struct TriView {
    pub topo: TriTopology,
    pub geom: TriGeometry,
    pub class: TriClassification,
}

#[derive(Serialize)]
struct ViewerFace {
    tri: [usize; 3],
    tag: u32,
    regions: [u32; 2],
    surface: u32,
}

#[derive(Serialize)]
struct ViewerStats {
    n_points: usize,
    n_tets: usize,
    min_dihedral_deg: f64,
    max_radius_edge: f64,
    max_edge: f64,
    millis: u64,
}

#[derive(Serialize)]
struct ViewerDefect {
    kind: &'static str,
    pos: [f64; 3],
    value: f64,
}

/// A mesh in the viewer schema (shared by the comparison viewer and the
/// showcase site).
#[derive(Serialize)]
struct Viewer<'a> {
    name: &'a str,
    mesher: &'static str,
    points: &'a [[f64; 3]],
    tets: &'a [[usize; 4]],
    tet_regions: Vec<u32>,
    faces: Vec<ViewerFace>,
    /// Owner solid per surface, -1 for a sheet.
    surface_owners: Vec<i64>,
    solids: &'a [SolidInfo],
    tag_labels: BTreeMap<String, &'a str>,
    edges: Vec<[usize; 2]>,
    stats: ViewerStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    defects: Option<Vec<ViewerDefect>>,
}

impl<'a> Viewer<'a> {
    fn new(
        name: &'a str,
        points: &'a [[f64; 3]],
        faces: &[rapidmesh_tet::SurfaceFace],
        owners: &[u32],
        labels: &'a Labels,
        stats: ViewerStats,
    ) -> Viewer<'a> {
        Viewer {
            name,
            mesher: "rapidmesh",
            points,
            tets: &[],
            tet_regions: Vec::new(),
            faces: faces
                .iter()
                .map(|f| ViewerFace {
                    tri: f.tri,
                    tag: f.face_tag.0,
                    regions: [f.regions[0].0, f.regions[1].0],
                    surface: f.surface,
                })
                .collect(),
            surface_owners: owners
                .iter()
                .map(|&o| if o == u32::MAX { -1 } else { o as i64 })
                .collect(),
            solids: &labels.solids,
            tag_labels: labels
                .tag_labels
                .iter()
                .map(|(t, n)| (t.to_string(), n.as_str()))
                .collect(),
            edges: Vec::new(),
            stats,
            defects: None,
        }
    }

    fn write(&self, name: &str, dir: &Path) -> io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("rapidmesh_{name}.json"));
        let mut w = io::BufWriter::new(std::fs::File::create(&path)?);
        serde_json::to_writer(&mut w, self)?;
        w.flush()?;
        crate::export::write_manifest(dir)?;
        Ok(path)
    }
}

/// Python's `{:.4g}`: four significant digits, exponent form outside
/// 1e-4..1e4.
fn fmt_g4(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let sci = format!("{x:.3e}");
    let (mant, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let trim = |s: String| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    if (-4..4).contains(&exp) {
        trim(format!("{:.*}", (3 - exp) as usize, x))
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mant.to_string()), exp.abs())
    }
}

/// A tetrahedral mesh with its labels and run record. Derefs to the
/// [`TetMesh`] (points, tets, regions, faces, classes).
pub struct Mesh {
    inner: TetMesh,
    /// Quality with the location of the worst tet and per region.
    pub quality: QualityStats,
    pub labels: Labels,
    pub run: Run,
    /// The model the mesh was made from, for the fidelity check.
    model: Option<Arc<Model>>,
    view: OnceLock<TetView>,
}

impl Deref for Mesh {
    type Target = TetMesh;
    fn deref(&self) -> &TetMesh {
        &self.inner
    }
}

impl Mesh {
    pub(crate) fn new(
        inner: TetMesh,
        quality: QualityStats,
        labels: Labels,
        run: Run,
        model: Option<Arc<Model>>,
    ) -> Mesh {
        Mesh {
            inner,
            quality,
            labels,
            run,
            model,
            view: OnceLock::new(),
        }
    }

    /// A mesh made elsewhere, without labels or input model.
    pub fn from_tets(inner: TetMesh) -> Mesh {
        let quality = rapidmesh_tet::quality_stats(&inner);
        Mesh::new(inner, quality, Labels::default(), Run::default(), None)
    }

    pub fn into_tet_mesh(self) -> TetMesh {
        self.inner
    }

    /// The solver view, built on first use.
    pub fn view(&self) -> &TetView {
        self.view.get_or_init(|| {
            let topo = TetTopology::build(&self.inner);
            let geom = TetGeometry::build(&topo, &self.inner.points);
            let class = Classification::build(&self.inner, &topo);
            TetView { topo, geom, class }
        })
    }

    /// The named sets: tets per region group, faces per named sheet tag,
    /// named geometric face and the `boundary`, edges per named geometric
    /// edge, and faces and edges per geometric face and edge.
    pub fn sets(&self) -> Sets {
        let v = self.view();
        let mut sets = Sets::default();
        for (name, rs) in self.labels.region_groups() {
            let regions = &self.inner.tet_regions;
            let cells = indices(regions.len(), |t| rs.contains(&regions[t].0));
            sets.cells.insert(name, cells);
        }
        let nf = v.topo.faces.len();
        for (&tag, name) in &self.labels.tag_labels {
            let f = indices(nf, |f| v.class.face_tag[f] == tag);
            sets.faces.insert(name.clone(), f);
        }
        let boundary = indices(nf, |f| v.topo.face_tets[f][1] == NONE);
        sets.faces.insert("boundary".into(), boundary);
        named_sets(
            &self.labels,
            &v.class.face_patch,
            &v.class.edge_curve,
            &mut sets,
        );
        sets
    }

    /// Writes a gmsh MSH 4.1 file: geometric vertices, edges and faces are
    /// point, curve and surface entities (tag = id + 1), regions volume
    /// entities; physical groups are the region groups, the named sheet
    /// tags and the named faces and edges.
    pub fn write_msh(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        self.write_msh_to(&mut w)?;
        w.flush()
    }

    pub fn write_msh_to(&self, w: &mut impl Write) -> io::Result<()> {
        rapidmesh_topo::export::write_msh(&self.inner, &self.labels.msh_names(true), w)
    }

    /// Writes a VTK XML unstructured grid: the tets and the geometric
    /// faces, with cell data `region`, `patch` and `face_tag`.
    pub fn write_vtu(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        rapidmesh_topo::export::write_vtu(&self.inner, &mut w)?;
        w.flush()
    }

    /// Quality, conformity and fidelity to the input, with located defects.
    pub fn diagnostics(&self) -> Diagnostics {
        let fidelity = self
            .model
            .as_ref()
            .filter(|_| !self.inner.tets.is_empty())
            .map(|m| rapidmesh_tet::fidelity::measure(&self.inner, m));
        Diagnostics {
            mesh: rapidmesh_tet::diagnostics::diagnose(&self.inner),
            fidelity,
        }
    }

    /// A report of the run: stage timings, where the quality is worst, per
    /// region quality and the warnings.
    pub fn report(&self) -> String {
        let q = &self.quality;
        let mut lines = vec![self.to_string(), String::new(), "timings (s):".into()];
        for (stage, secs) in &self.run.timings {
            lines.push(format!("  {stage:<22} {secs:8.3}"));
        }
        lines.push(String::new());
        lines.push("quality:".into());
        let loc = q.worst_location;
        lines.push(format!(
            "  min dihedral {:.2} deg in region {} near ({}, {}, {})",
            q.min_dihedral_deg,
            q.worst_region,
            fmt_g4(loc[0]),
            fmt_g4(loc[1]),
            fmt_g4(loc[2])
        ));
        lines.push(format!("  max radius/edge {:.2}", q.max_radius_edge));
        for &(region, min_dih, n) in &q.per_region {
            lines.push(format!(
                "  region {region:<3} min dihedral {min_dih:6.2} deg ({n} tets)"
            ));
        }
        let warn: Vec<&Event> = self.run.warnings().collect();
        if !warn.is_empty() {
            lines.push(String::new());
            lines.push("warnings:".into());
            for e in warn {
                lines.push(format!("  [{}] {}", e.stage, e.message));
            }
        }
        lines.join("\n")
    }

    fn viewer<'a>(&'a self, name: &'a str) -> Viewer<'a> {
        let q = &self.quality;
        let stats = ViewerStats {
            n_points: self.inner.points.len(),
            n_tets: self.inner.tets.len(),
            min_dihedral_deg: q.min_dihedral_deg,
            max_radius_edge: q.max_radius_edge,
            max_edge: q.max_edge,
            millis: self.run.millis,
        };
        let m = &self.inner;
        let defects = self
            .diagnostics()
            .defects()
            .map(|d| ViewerDefect {
                kind: d.kind.name(),
                pos: d.pos,
                value: d.value,
            })
            .collect();
        Viewer {
            tets: &m.tets,
            tet_regions: m.tet_regions.iter().map(|r| r.0).collect(),
            edges: m.feature_edges(),
            defects: Some(defects),
            ..Viewer::new(
                name,
                &m.points,
                &m.faces,
                &m.surface_owners,
                &self.labels,
                stats,
            )
        }
    }

    /// The mesh in the viewer JSON schema, with the located defects.
    pub fn viewer_json(&self, name: &str) -> String {
        serde_json::to_string(&self.viewer(name)).expect("serialize")
    }

    /// Writes `rapidmesh_<name>.json` into `dir` and refreshes the viewer
    /// manifest there.
    pub fn save_viewer_json(&self, name: &str, dir: impl AsRef<Path>) -> io::Result<PathBuf> {
        self.viewer(name).write(name, dir.as_ref())
    }
}

impl fmt::Display for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Mesh({} tets, {} points, min dihedral {:.1} deg, {} ms)",
            self.inner.tets.len(),
            self.inner.points.len(),
            self.quality.min_dihedral_deg,
            self.run.millis
        )
    }
}

/// A surface mesh with its labels and run record. Derefs to the
/// [`rapidmesh_tet::SurfaceMesh`].
pub struct SurfaceMesh {
    inner: rapidmesh_tet::SurfaceMesh,
    pub labels: Labels,
    pub run: Run,
    view: OnceLock<TriView>,
}

impl Deref for SurfaceMesh {
    type Target = rapidmesh_tet::SurfaceMesh;
    fn deref(&self) -> &rapidmesh_tet::SurfaceMesh {
        &self.inner
    }
}

impl SurfaceMesh {
    pub(crate) fn new(inner: rapidmesh_tet::SurfaceMesh, labels: Labels, run: Run) -> SurfaceMesh {
        SurfaceMesh {
            inner,
            labels,
            run,
            view: OnceLock::new(),
        }
    }

    pub fn into_surface_mesh(self) -> rapidmesh_tet::SurfaceMesh {
        self.inner
    }

    /// The solver view, built on first use.
    pub fn view(&self) -> &TriView {
        self.view.get_or_init(|| {
            let topo = TriTopology::build(&self.inner);
            let geom = TriGeometry::build_3d(&topo, &self.inner.points);
            let class = TriClassification::build(&self.inner, &topo);
            TriView { topo, geom, class }
        })
    }

    /// The named sets: triangles per named sheet tag and named geometric
    /// face, edges per named geometric edge, triangles and edges per
    /// geometric face and edge (no cells).
    pub fn sets(&self) -> Sets {
        let v = self.view();
        let mut sets = Sets::default();
        let faces = &self.inner.faces;
        for (&tag, name) in &self.labels.tag_labels {
            let f = indices(faces.len(), |i| faces[i].face_tag.0 == tag);
            sets.faces.insert(name.clone(), f);
        }
        named_sets(
            &self.labels,
            &v.class.tri_patch,
            &v.class.edge_curve,
            &mut sets,
        );
        sets
    }

    /// Writes a gmsh MSH 4.1 file (see [`Mesh::write_msh`]).
    pub fn write_msh(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        self.write_msh_to(&mut w)?;
        w.flush()
    }

    pub fn write_msh_to(&self, w: &mut impl Write) -> io::Result<()> {
        rapidmesh_topo::export::write_surface_msh(&self.inner, &self.labels.msh_names(false), w)
    }

    /// Writes a VTK XML unstructured grid with cell data `patch` and
    /// `face_tag`.
    pub fn write_vtu(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        rapidmesh_topo::export::write_surface_vtu(&self.inner, &mut w)?;
        w.flush()
    }

    /// RWG basis functions `[v0, v1, tri_plus, tri_minus]`: an edge with
    /// `k` triangles of one tag carries `k - 1`, from its lowest triangle to
    /// each other one; `connect_tags` joins triangles of different tags too.
    pub fn rwg_edges(&self, connect_tags: bool) -> Vec<[u32; 4]> {
        let t = &self.view().topo;
        t.rwg_edges(&t.tri_tags, connect_tags)
    }

    /// Conductor outline: edges with one triangle of a tag, `[v0, v1, tri]`.
    pub fn boundary_edges(&self) -> Vec<[u32; 3]> {
        self.view().topo.boundary_edges()
    }

    /// Boundary edges with both ends on the line `{axis = value}`, the
    /// first other coordinate in `[lo, hi]`.
    pub fn edges_on_line(
        &self,
        axis: usize,
        value: f64,
        lo: f64,
        hi: f64,
        tol: f64,
    ) -> Vec<[u32; 2]> {
        let other = (0..3).find(|&c| c != axis).unwrap_or(0);
        let on = |v: u32| {
            let p = self.inner.points[v as usize];
            (p[axis] - value).abs() <= tol && p[other] >= lo - tol && p[other] <= hi + tol
        };
        self.boundary_edges()
            .iter()
            .filter(|e| on(e[0]) && on(e[1]))
            .map(|e| [e[0], e[1]])
            .collect()
    }

    /// The surface mesh in the viewer JSON schema (no tets).
    pub fn viewer_json(&self, name: &str) -> String {
        let m = &self.inner;
        let stats = ViewerStats {
            n_points: m.points.len(),
            n_tets: 0,
            min_dihedral_deg: 0.0,
            max_radius_edge: 0.0,
            max_edge: 0.0,
            millis: self.run.millis,
        };
        let v = Viewer::new(
            name,
            &m.points,
            &m.faces,
            &m.surface_owners,
            &self.labels,
            stats,
        );
        serde_json::to_string(&v).expect("serialize")
    }
}

impl fmt::Display for SurfaceMesh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SurfaceMesh({} faces, {} points, {} ms)",
            self.inner.faces.len(),
            self.inner.points.len(),
            self.run.millis
        )
    }
}

#[cfg(test)]
mod tests {
    use super::fmt_g4;

    #[test]
    fn g4_matches_python() {
        for (x, s) in [
            (0.0, "0"),
            (1.0, "1"),
            (0.5, "0.5"),
            (1.23456, "1.235"),
            (12345.0, "1.234e+04"),
            (9999.5, "1e+04"),
            (0.000123456, "0.0001235"),
            (0.0000123, "1.23e-05"),
            (-2.5, "-2.5"),
            (100.0, "100"),
        ] {
            assert_eq!(fmt_g4(x), s, "{x}");
        }
    }
}
