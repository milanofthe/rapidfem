//! PLC types, region classification, and quality reporting for the tet mesher.
//!
//! `mesh_plc` / `mesh_plc_with` are the stable entry points; they delegate to
//! the CVT mesher in [`crate::cvt`]. The exact CSG arrangement (the `TaggedPlc`)
//! is produced upstream and meshed here into a conforming, region-tagged tet
//! mesh. This module owns the output types (`TetMesh`, `SurfaceFace`), the mesh
//! parameters (`MeshParams`), the coplanar-patch grouping reused for boundary
//! tagging, and the quality statistics. (Tet region classification now lives in
//! the central [`crate::domain::DomainTree`], by per-region ray-cast.)

use rapidmesh_geom::{FaceTag, RegionTag, SurfaceKind, TaggedPlc};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// Deterministic hashing: meshing decisions iterate these containers, and a
/// mesher must be reproducible run-to-run (std's RandomState is not).
type DState = BuildHasherDefault<rustc_hash::FxHasher>;
type DMap<K, V> = HashMap<K, V, DState>;

/// A boundary surface mesh: the conforming triangulation of the PLC patches,
/// produced by the surface path ([`crate::bottomup::surface_mesh`])
/// without the volume tetrahedralization. Same face schema as [`TetMesh`].
#[derive(Debug)]
pub struct SurfaceMesh {
    /// Surface vertices (PLC corners, graded edge points, patch interior).
    pub points: Vec<[f64; 3]>,
    /// The triangulation of every patch, tagged.
    pub faces: Vec<SurfaceFace>,
    /// The analytic surfaces referenced by [SurfaceFace::surface].
    pub surfaces: Vec<SurfaceKind>,
    /// Per-surface owner solid index, parallel to `surfaces`.
    pub surface_owners: Vec<u32>,
    /// The mesh edges on B-rep edges.
    pub curve_edges: Vec<CurveEdge>,
    /// What each point lies on, parallel to `points`.
    pub point_class: Vec<PointClass>,
}

/// A conforming surface face of the tet mesh, with its PLC tags.
#[derive(Debug, Clone)]
pub struct SurfaceFace {
    /// Global vertex indices.
    pub tri: [usize; 3],
    /// Face tag inherited from the PLC patch (sheets, ports).
    pub face_tag: FaceTag,
    /// Region tags on (front, back) of the source patch.
    pub regions: [RegionTag; 2],
    /// Identity of the source patch (faces of one patch are coplanar and may
    /// be re-tiled by the optimizer).
    pub patch: u32,
    /// Analytic surface this face approximates (index into
    /// [TetMesh::surfaces]); curved kinds let the optimizer move surface
    /// vertices on the true surface.
    pub surface: u32,
}

/// What a mesh vertex lies on: a B-rep vertex, edge or face (ids of the
/// B-rep the mesher builds from the PLC; face ids match
/// [`SurfaceFace::patch`]), or the interior of the region around it. The
/// mesher knows it for every vertex it places; solvers and later passes read
/// it instead of re-deriving it (curved elements project by it, boundary
/// conditions select by it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointClass {
    Vertex(u32),
    Edge(u32),
    Face(u32),
    Interior,
}

/// A mesh edge on a B-rep edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurveEdge {
    /// The two mesh points.
    pub v: [usize; 2],
    /// The B-rep edge it lies on.
    pub edge: u32,
}

/// A region-tagged conforming tetrahedral mesh.
#[derive(Debug)]
pub struct TetMesh {
    /// Mesh vertices (PLC vertices plus interior/Steiner points).
    pub points: Vec<[f64; 3]>,
    /// Positively oriented tets.
    pub tets: Vec<[usize; 4]>,
    /// Region of every tet.
    pub tet_regions: Vec<RegionTag>,
    /// The mesh faces tiling the PLC patches, with tags.
    pub faces: Vec<SurfaceFace>,
    /// The analytic surfaces referenced by [SurfaceFace::surface].
    pub surfaces: Vec<SurfaceKind>,
    /// Per-surface owner solid index (scene insertion order, voids included);
    /// `u32::MAX` for sheet surfaces. Parallel to `surfaces`.
    pub surface_owners: Vec<u32>,
    /// `points[..plc_points]` are the PLC's own vertices (the geometry);
    /// everything after is an interior point the mesher added.
    pub plc_points: usize,
    /// Local target edge length at each point (the sizing field `h(x)` at seed
    /// time), parallel to `points`. Lets the optimizer respect a GRADED size
    /// (e.g. curvature-fine at an airfoil nose) instead of one region-uniform
    /// floor, so it does not coarsen away fine, intentional detail. Empty or
    /// `INFINITY` means "no local target" (no size-driven coarsening there).
    pub point_size: Vec<f64>,
    /// What each point lies on, parallel to `points`.
    pub point_class: Vec<PointClass>,
    /// The mesh edges on B-rep curves (the mesher's protected curve
    /// segments), those inside one face included (an import's open crease).
    pub curve_edges: Vec<CurveEdge>,
    /// Periodic pairs: every point on a face `a` of a pair with its image
    /// on face `b`.
    pub periodic_points: Vec<[usize; 2]>,
    /// The faces (indices into `faces`) that close a filled contact wedge:
    /// off the geometry by design, as wide as the mesh is fine there.
    pub contact_faces: Vec<usize>,
}

impl TetMesh {
    /// Appends a point with its class (and no local size target).
    pub fn push_point(&mut self, p: [f64; 3], class: PointClass) {
        if self.point_class.len() == self.points.len() {
            self.point_class.push(class);
        }
        if self.point_size.len() == self.points.len() {
            self.point_size.push(f64::INFINITY);
        }
        self.points.push(p);
    }

    /// Feature (crease) edges of the final surface mesh, derived from the
    /// faces so they stay valid through optimizer rewrites. An edge is a
    /// feature edge iff it is not interior to one smooth surface group:
    /// boundary/non-manifold incidence (face count != 2), or the two faces
    /// differ in analytic surface, face tag, or region pair. Within ONE
    /// `Plane` surface entry that collects several non-coplanar walls (loft
    /// flanks, pipe segments), the planar patch id discriminates, so true
    /// geometric creases survive while the facet seams of curved analytic
    /// surfaces (cylinder barrel) stay smooth.
    ///
    /// The mesher's curve edges come on top, as long as they are still edges
    /// of the surface: they carry the features no label change marks.
    pub fn feature_edges(&self) -> Vec<[usize; 2]> {
        // group key per face: planes split by patch, curved by surface
        // (surface, smooth-id, face-tag, region-lo, region-hi)
        type FaceKey = (u32, u32, u32, u32, u32);
        let face_key = |sf: &SurfaceFace| -> FaceKey {
            let smooth = match self.surfaces[sf.surface as usize] {
                SurfaceKind::Plane { .. } | SurfaceKind::Facets => sf.patch,
                _ => u32::MAX,
            };
            let (r0, r1) = (
                sf.regions[0].0.min(sf.regions[1].0),
                sf.regions[0].0.max(sf.regions[1].0),
            );
            (sf.surface, smooth, sf.face_tag.0, r0, r1)
        };
        // edge -> (incidence count, first face key seen, mixed-key flag)
        let mut edges: DMap<(usize, usize), (u32, FaceKey, bool)> = DMap::default();
        for sf in &self.faces {
            let key = face_key(sf);
            for k in 0..3 {
                let (a, b) = (sf.tri[k], sf.tri[(k + 1) % 3]);
                let e = (a.min(b), a.max(b));
                let entry = edges.entry(e).or_insert((0, key, false));
                entry.0 += 1;
                if entry.1 != key {
                    entry.2 = true;
                }
            }
        }
        let mut out: Vec<[usize; 2]> = edges
            .iter()
            .filter(|(_, &(cnt, _, mixed))| cnt != 2 || mixed)
            .map(|(&(a, b), _)| [a, b])
            .collect();
        out.extend(
            self.curve_edges
                .iter()
                .map(|e| [e.v[0].min(e.v[1]), e.v[0].max(e.v[1])])
                .filter(|e| edges.contains_key(&(e[0], e[1]))),
        );
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Sizing and quality parameters for [`mesh_plc_with`].
#[derive(Debug, Clone)]
pub struct MeshParams {
    /// Target edge length, in the sense gmsh gives its mesh size: volume
    /// edges have a median near `maxh` and spread around it, curve samples
    /// are at most `maxh` apart. Curvature, feature and local sources refine
    /// below it.
    pub maxh: f64,
    /// Per-region target edge length, overriding maxh inside that region
    /// (Maxwell FEM sizes regions by local wavelength, h ~ lambda/sqrt(eps)).
    /// Interfaces and creases follow the finer adjacent region; transitions
    /// into coarser regions grade naturally.
    pub region_maxh: Vec<(u32, f64)>,
    /// Delaunay-refinement quality bound: tets with
    /// circumradius / shortest-edge above this get their circumcenter
    /// inserted. The provable refinement regime is >= 2.0.
    pub radius_edge_bound: f64,
    /// Refinement stops (best effort) once this many points exist.
    pub max_points: usize,
    /// Size grading: the target edge length may grow by at most this factor
    /// per unit distance from finer features (h(x) is Lipschitz with this
    /// constant). 0.5 grows neighbor elements by roughly 1.5x; INFINITY
    /// disables grading (sizes jump at region interfaces).
    pub grading: f64,
    /// Per-face-tag target edge length, overriding the adjacent regions'
    /// targets on those patches.
    pub face_maxh: Vec<(u32, f64)>,
    /// Per-solid SURFACE target edge length, keyed by the owner solid index
    /// in [TaggedPlc::surface_owners] (scene insertion order, voids
    /// included): refines the solid's boundary patches and grades into the
    /// surrounding volume. The only sizing handle that reaches a void's
    /// walls (a coax inner conductor has no region and no face tag).
    pub surface_maxh: Vec<(u32, f64)>,
    /// Point size sources `(position, h)`: the target shrinks to `h` at the
    /// point and recovers along the Lipschitz grading away from it
    /// (the hook for error-driven adaptive refinement).
    pub size_points: Vec<([f64; 3], f64)>,
    /// Relative chord (sagitta) tolerance for curved EDGES: a curve of radius `R`
    /// is sampled at `h = R*sqrt(8*tol_edge)`, so the chord deviates by at most
    /// `tol_edge * R`. Scale-invariant (constant segments per arc). Default 1e-2.
    pub tol_edge: f64,
    /// Relative chord (sagitta) tolerance for curved SURFACES, the 2D analogue of
    /// [`MeshParams::tol_edge`]: a facet on a surface of principal radius `R` is
    /// sized `h = R*sqrt(8*tol_surf)`. Default 1e-2. (There is no volume
    /// tolerance: the volume size follows from the surface.)
    pub tol_surf: f64,
    /// Maximum element edge length on EDGES (1-cells), combined with the global
    /// [`MeshParams::maxh`] as `min(maxh, cap_edge)`. `INFINITY` = no extra cap.
    pub cap_edge: f64,
    /// Maximum element edge length on SURFACES (2-cells); `min(maxh, cap_surf)`.
    pub cap_surf: f64,
    /// Maximum element edge length in the VOLUME (3-cells); `min(maxh, cap_vol)`.
    /// Per-region overrides come from [`MeshParams::region_maxh`].
    pub cap_vol: f64,
    /// Per-EDGE size override `(brep edge id, maxh)`: the hierarchical
    /// `g.region(..).surf(..).edge(..).maxh` resolves to entries here, overriding
    /// the global [`MeshParams::cap_edge`] on that specific edge.
    pub edge_maxh: Vec<(u32, f64)>,
    /// Per-EDGE deflection override `(brep edge id, tol)`, overriding
    /// [`MeshParams::tol_edge`] on that edge.
    pub edge_tol: Vec<(u32, f64)>,
    /// Per-FACE size override `(brep face id, maxh)`, overriding
    /// [`MeshParams::cap_surf`] on that face.
    pub surf_maxh: Vec<(u32, f64)>,
    /// Per-FACE deflection override `(brep face id, tol)`, overriding
    /// [`MeshParams::tol_surf`] on that face.
    pub surf_tol: Vec<(u32, f64)>,
    /// Minimum element edge length on SURFACES (2-cells) and their edges: a hard
    /// floor, the field is never refined below it (and the element budget cannot
    /// go under it). `0` = off.
    pub min_h_surf: f64,
    /// Minimum element edge length in the VOLUME (3-cells): a hard floor. `0` = off.
    pub min_h_vol: f64,
    /// Surface (2D) quality target: minimum triangle angle in DEGREES for the
    /// sizing-field-driven Ruppert/Chew refinement of free sheet faces. `0` = off
    /// (the volume path leaves this 0 to keep its frozen surface unchanged; the
    /// standalone surface mesher turns it on).
    pub surf_min_angle: f64,
    /// Surface triangle BUDGET (a cap, not a target): the count-driven Ruppert
    /// refines size-field-driven until h_min is reached, but stops early once the
    /// triangle count hits this budget (split across patches by area). `0` = off.
    pub surf_target_count: usize,
    /// Pairs of B-rep faces meshed with the same triangles: face `b` is
    /// face `a` shifted.
    pub periodic: Vec<crate::mesh3::PeriodicPair>,
    /// Elements across the thickness of each region: inside a region of
    /// thickness `t` (`2 V / S`, see
    /// [`TaggedPlc::region_thickness`](rapidmesh_geom::TaggedPlc::region_thickness))
    /// the size is at most `t / cells_across`, as if given per region, so a
    /// thin plate or wire gets proper tets through it. `0` = off: a stack of
    /// layers far thinner than the size then takes the layered mesh, flat
    /// tets through each layer.
    pub cells_across: f64,
}

impl Default for MeshParams {
    fn default() -> Self {
        MeshParams {
            maxh: f64::INFINITY,
            region_maxh: Vec::new(),
            radius_edge_bound: 2.0,
            max_points: 100_000,
            grading: 0.5,
            face_maxh: Vec::new(),
            surface_maxh: Vec::new(),
            size_points: Vec::new(),
            tol_edge: 1e-2,
            tol_surf: 1e-2,
            cap_edge: f64::INFINITY,
            cap_surf: f64::INFINITY,
            cap_vol: f64::INFINITY,
            edge_maxh: Vec::new(),
            edge_tol: Vec::new(),
            surf_maxh: Vec::new(),
            surf_tol: Vec::new(),
            min_h_surf: 0.0,
            min_h_vol: 0.0,
            surf_min_angle: 0.0,
            surf_target_count: 0,
            periodic: Vec::new(),
            cells_across: 0.0,
        }
    }
}

fn lookup(table: &[(u32, f64)], id: usize) -> Option<f64> {
    table
        .iter()
        .find(|&&(i, _)| i as usize == id)
        .map(|&(_, v)| v)
}

impl MeshParams {
    /// A copy with every size target scaled by `s`: lengths by `s`, chord
    /// tolerances by `s^2` (size is proportional to sqrt(tol)). The element-budget
    /// loop retunes the global scale with this while preserving the relative
    /// refinement; the cap methods read the scaled fields, so they scale too.
    pub fn scaled(&self, s: f64) -> MeshParams {
        let sv = |v: &[(u32, f64)], e: f64| -> Vec<(u32, f64)> {
            v.iter().map(|&(t, h)| (t, h * e)).collect()
        };
        MeshParams {
            maxh: self.maxh * s,
            region_maxh: sv(&self.region_maxh, s),
            radius_edge_bound: self.radius_edge_bound,
            max_points: self.max_points,
            grading: self.grading,
            face_maxh: sv(&self.face_maxh, s),
            surface_maxh: sv(&self.surface_maxh, s),
            size_points: self.size_points.iter().map(|&(p, h)| (p, h * s)).collect(),
            tol_edge: self.tol_edge * s * s,
            tol_surf: self.tol_surf * s * s,
            cap_edge: self.cap_edge * s,
            cap_surf: self.cap_surf * s,
            cap_vol: self.cap_vol * s,
            edge_maxh: sv(&self.edge_maxh, s),
            edge_tol: sv(&self.edge_tol, s * s),
            surf_maxh: sv(&self.surf_maxh, s),
            surf_tol: sv(&self.surf_tol, s * s),
            // Absolute floors: NOT scaled by the budget -- they bound the finest
            // element regardless of the count target.
            min_h_surf: self.min_h_surf,
            min_h_vol: self.min_h_vol,
            surf_min_angle: self.surf_min_angle,
            surf_target_count: self.surf_target_count,
            periodic: self.periodic.clone(),
            cells_across: self.cells_across,
        }
    }

    /// These parameters with the thickness bound of every region of `plc`
    /// (see [`MeshParams::cells_across`]) in the per-region sizes, the
    /// finer of the two winning. Applying it twice changes nothing.
    pub fn with_thickness_caps(&self, plc: &TaggedPlc) -> MeshParams {
        let mut p = self.clone();
        if !(self.cells_across > 0.0) {
            return p;
        }
        for (r, t) in plc.region_thickness() {
            let cap = t / self.cells_across;
            match p.region_maxh.iter_mut().find(|(rr, _)| *rr == r) {
                Some(e) => e.1 = e.1.min(cap),
                None => p.region_maxh.push((r, cap)),
            }
        }
        p
    }

    /// Effective edge-length cap on 1-cells: the global cap tightened by the
    /// per-dimension edge cap.
    pub fn edge_cap(&self) -> f64 {
        self.maxh.min(self.cap_edge)
    }
    /// Effective edge-length cap on 2-cells (surfaces).
    pub fn surf_cap(&self) -> f64 {
        self.maxh.min(self.cap_surf)
    }
    /// Effective edge-length cap in the 3-cell (volume).
    pub fn vol_cap(&self) -> f64 {
        self.maxh.min(self.cap_vol)
    }
    /// Size cap for brep edge `id`: its per-edge override, else the edge cap.
    /// The size floor of the refinement for a model of extent `extent`:
    /// clamps curvature-driven runaway at an eighth of the reference size,
    /// while explicit user targets (per edge, face, region or point) may be
    /// finer and win.
    pub fn h_floor(&self, extent: f64) -> f64 {
        let h_ref = if self.maxh.is_finite() {
            self.maxh
        } else {
            extent / crate::constants::DEFAULT_SUBDIV
        };
        let user_min = self
            .edge_maxh
            .iter()
            .chain(&self.surf_maxh)
            .chain(&self.face_maxh)
            .chain(&self.surface_maxh)
            .chain(&self.region_maxh)
            .map(|&(_, h)| h)
            .chain(self.size_points.iter().map(|&(_, h)| h))
            .fold(f64::INFINITY, f64::min);
        self.min_h_surf
            .max(self.min_h_vol)
            .max((h_ref / 8.0).min(user_min))
            .max(1e-12)
    }
    pub fn edge_maxh_for(&self, id: usize) -> f64 {
        lookup(&self.edge_maxh, id)
            .unwrap_or(f64::INFINITY)
            .min(self.edge_cap())
    }
    /// Deflection for brep edge `id`: its per-edge override, else `tol_edge`.
    pub fn edge_tol_for(&self, id: usize) -> f64 {
        lookup(&self.edge_tol, id).unwrap_or(self.tol_edge)
    }
    /// Size cap for brep face `id`: its per-face override, else the surface cap.
    pub fn surf_maxh_for(&self, id: usize) -> f64 {
        lookup(&self.surf_maxh, id)
            .unwrap_or(f64::INFINITY)
            .min(self.surf_cap())
    }
    /// Deflection for brep face `id`: its per-face override, else `tol_surf`.
    pub fn surf_tol_for(&self, id: usize) -> f64 {
        lookup(&self.surf_tol, id).unwrap_or(self.tol_surf)
    }
}

/// Circumcenter and circumradius of a tet, `None` if degenerate.
fn tet_circumcenter(p: [[f64; 3]; 4]) -> Option<([f64; 3], f64)> {
    // Rows 2(p_i - p_0), rhs |p_i|^2 - |p_0|^2.
    let row = |i: usize| -> [f64; 3] { std::array::from_fn(|k| 2.0 * (p[i][k] - p[0][k])) };
    let sq = |q: [f64; 3]| -> f64 { q.iter().map(|x| x * x).sum() };
    let (r1, r2, r3) = (row(1), row(2), row(3));
    let b = [
        sq(p[1]) - sq(p[0]),
        sq(p[2]) - sq(p[0]),
        sq(p[3]) - sq(p[0]),
    ];
    let det3 = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| -> f64 {
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
    };
    let d = det3(r1, r2, r3);
    let scale: f64 = [r1, r2, r3]
        .iter()
        .map(|r| r.iter().map(|x| x.abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    if d.abs() < 1e-12 * scale.powi(3) {
        return None;
    }
    let col = |j: usize| -> f64 {
        let mut m = [r1, r2, r3];
        for (i, row) in m.iter_mut().enumerate() {
            row[j] = b[i];
        }
        det3(m[0], m[1], m[2]) / d
    };
    let c = [col(0), col(1), col(2)];
    let r = (0..3).map(|k| (c[k] - p[0][k]).powi(2)).sum::<f64>().sqrt();
    Some((c, r))
}

/// Meshes a tagged PLC into a conforming, region-tagged tet mesh with default
/// sizing (an eighth of the bounding-box diagonal -- an UNBOUNDED size makes
/// the duplicate guards of the refinement path meaningless) and no quality
/// bound. Background (region 0) tets are dropped.
pub fn mesh_plc(plc: &TaggedPlc) -> TetMesh {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &plc.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = (0..3)
        .map(|k| hi[k] - lo[k])
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    mesh_plc_with(
        plc,
        &MeshParams {
            maxh: diag / 8.0,
            radius_edge_bound: f64::INFINITY,
            max_points: usize::MAX,
            ..Default::default()
        },
    )
}

/// Meshes a tagged PLC into a conforming, region-tagged tet mesh, refined to
/// the given sizing and quality targets (best effort under
/// `params.max_points`) by the restricted-Delaunay core (`mesh3`). Builds the
/// model for this one mesh; see [`mesh_model`] to share one.
pub fn mesh_plc_with(plc: &TaggedPlc, params: &MeshParams) -> TetMesh {
    mesh_model(&rapidmesh_brep::Model::new(plc.clone()), params)
}

/// Meshes a built model (its PLC and B-rep), the entry point for callers
/// that keep one model across several meshes and id lookups.
pub fn mesh_model(model: &rapidmesh_brep::Model, params: &MeshParams) -> TetMesh {
    crate::mesh3::brep::mesh_brep(model, params)
}

/// Quality summary of a tet mesh, with WHERE the worst element is and a
/// per-region breakdown.
#[derive(Debug, Clone)]
pub struct QualityStats {
    /// Number of tets.
    pub n_tets: usize,
    /// Smallest dihedral angle in degrees (sliver indicator; the load-bearing
    /// metric for Nedelec conditioning).
    pub min_dihedral_deg: f64,
    /// Number of slivers: tets with a min dihedral below
    /// [`crate::diagnostics::SLIVER_DEG`].
    pub n_slivers: usize,
    /// Largest circumradius / shortest-edge ratio.
    pub max_radius_edge: f64,
    /// Longest edge in the mesh.
    pub max_edge: f64,
    /// Index of the tet holding the smallest dihedral angle (`usize::MAX` for
    /// an empty mesh).
    pub worst_tet: usize,
    /// Centroid of the worst tet: where the worst sliver sits.
    pub worst_location: [f64; 3],
    /// Region tag of the worst tet.
    pub worst_region: u32,
    /// Per region, in ascending tag order: (region, min dihedral deg, tets).
    pub per_region: Vec<(u32, f64, usize)>,
}

use crate::diagnostics::tet_min_dihedral;

/// Computes quality statistics over all tets, tracking the worst element's
/// location/region and a per-region min-dihedral breakdown.
pub fn quality_stats(mesh: &TetMesh) -> QualityStats {
    use rayon::prelude::*;
    /// The statistics of a block of tets.
    #[derive(Clone)]
    struct Part {
        min_dihedral: f64,
        worst_tet: usize,
        n_slivers: usize,
        max_re: f64,
        max_edge2: f64,
        per_region: Vec<(f64, usize)>,
    }
    let nreg = mesh
        .tet_regions
        .iter()
        .map(|r| r.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let empty = Part {
        min_dihedral: f64::MAX,
        worst_tet: usize::MAX,
        n_slivers: 0,
        max_re: 0.0,
        max_edge2: 0.0,
        per_region: vec![(f64::MAX, 0); nreg],
    };
    const BLOCK: usize = 1 << 14;
    let parts: Vec<Part> = mesh
        .tets
        .par_chunks(BLOCK)
        .enumerate()
        .map(|(bi, tets)| {
            let mut q = empty.clone();
            for (k, t) in tets.iter().enumerate() {
                let ti = bi * BLOCK + k;
                let p: [[f64; 3]; 4] = std::array::from_fn(|k| mesh.points[t[k]]);
                let mut lmin2 = f64::MAX;
                for i in 0..4 {
                    for j in i + 1..4 {
                        let d2: f64 = (0..3).map(|k| (p[i][k] - p[j][k]).powi(2)).sum();
                        lmin2 = lmin2.min(d2);
                        q.max_edge2 = q.max_edge2.max(d2);
                    }
                }
                if let Some((_, r)) = tet_circumcenter(p) {
                    q.max_re = q.max_re.max(r / lmin2.sqrt());
                }
                let md = tet_min_dihedral(p);
                if md < crate::diagnostics::SLIVER_DEG {
                    q.n_slivers += 1;
                }
                if md < q.min_dihedral {
                    q.min_dihedral = md;
                    q.worst_tet = ti;
                }
                let e = &mut q.per_region[mesh.tet_regions[ti].0 as usize];
                e.0 = e.0.min(md);
                e.1 += 1;
            }
            q
        })
        .collect();
    // Blocks in order: the first of equal worst tets wins, as serially.
    let mut q = empty;
    for b in parts {
        if b.min_dihedral < q.min_dihedral {
            q.min_dihedral = b.min_dihedral;
            q.worst_tet = b.worst_tet;
        }
        q.n_slivers += b.n_slivers;
        q.max_re = q.max_re.max(b.max_re);
        q.max_edge2 = q.max_edge2.max(b.max_edge2);
        for (e, x) in q.per_region.iter_mut().zip(&b.per_region) {
            e.0 = e.0.min(x.0);
            e.1 += x.1;
        }
    }
    let Part {
        min_dihedral,
        worst_tet,
        n_slivers,
        max_re,
        max_edge2,
        per_region,
    } = q;
    let per_region: Vec<(u32, (f64, usize))> = per_region
        .into_iter()
        .enumerate()
        .filter(|(_, (_, n))| *n > 0)
        .map(|(r, x)| (r as u32, x))
        .collect();
    let worst_location = if worst_tet != usize::MAX {
        let t = mesh.tets[worst_tet];
        std::array::from_fn(|k| (0..4).map(|c| mesh.points[t[c]][k]).sum::<f64>() / 4.0)
    } else {
        [0.0; 3]
    };
    let worst_region = if worst_tet != usize::MAX {
        mesh.tet_regions[worst_tet].0
    } else {
        0
    };
    QualityStats {
        n_tets: mesh.tets.len(),
        min_dihedral_deg: min_dihedral,
        n_slivers,
        max_radius_edge: max_re,
        max_edge: max_edge2.sqrt(),
        worst_tet,
        worst_location,
        worst_region,
        per_region: per_region
            .into_iter()
            .map(|(r, (m, n))| (r, m, n))
            .collect(),
    }
}

/// Emits the headline VOLUME-mesh metrics through [`rapidmesh_exact::log`]: the
/// element + vertex counts, the quality summary, the sliver count (a `warn` when
/// any sliver survives), and -- for multi-region meshes -- a per-region quality
/// breakdown. Recorded both live (when the log level allows) and into
/// `mesh.stats`, so the important numbers are visible without re-deriving them.
pub fn log_metrics(q: &QualityStats, n_points: usize) {
    use rapidmesh_exact::log;
    log::stat("mesh.tets", q.n_tets as f64);
    log::stat("mesh.points", n_points as f64);
    log::stat("mesh.min_dihedral_deg", q.min_dihedral_deg);
    log::stat("mesh.max_radius_edge", q.max_radius_edge);
    log::stat("mesh.longest_edge", q.max_edge);
    log::stat("mesh.slivers", q.n_slivers as f64);
    log::info("metrics", format!("tets {}  points {}", q.n_tets, n_points));
    log::info(
        "metrics",
        format!(
            "min-dihedral {:.1} deg   max-radius-edge {:.2}   longest-edge {:.4}",
            q.min_dihedral_deg, q.max_radius_edge, q.max_edge
        ),
    );
    let pct = if q.n_tets > 0 {
        100.0 * q.n_slivers as f64 / q.n_tets as f64
    } else {
        0.0
    };
    let lvl = if q.n_slivers > 0 {
        log::Level::Warn
    } else {
        log::Level::Info
    };
    log::event(
        lvl,
        "metrics",
        format!(
            "slivers {} / {} ({pct:.2}%, below {:.0} deg)",
            q.n_slivers,
            q.n_tets,
            crate::diagnostics::SLIVER_DEG
        ),
    );
    if q.n_slivers > 0 {
        log::warn(
            "metrics",
            format!(
                "worst {:.1} deg in region {} near ({:.4}, {:.4}, {:.4})",
                q.min_dihedral_deg,
                q.worst_region,
                q.worst_location[0],
                q.worst_location[1],
                q.worst_location[2]
            ),
        );
    }
    if q.per_region.len() > 1 {
        let parts: Vec<String> = q
            .per_region
            .iter()
            .map(|(r, m, n)| format!("r{r}:{m:.1}deg/{n}"))
            .collect();
        log::info("metrics", format!("regions  {}", parts.join("  ")));
    }
}

/// Emits the headline SURFACE-mesh metrics through [`rapidmesh_exact::log`]:
/// triangle + vertex counts and the minimum interior angle, with a count of thin
/// (`< 15 deg`) triangles (a `warn` when any survive).
pub fn log_surface_metrics(mesh: &SurfaceMesh) {
    use rapidmesh_exact::log;
    // Per-triangle minimum interior angle (degrees). Inlined here (the only
    // in-crate user) so the MoM/quality accessors can live solely in the
    // downstream `rapidmesh_topo` topology layer.
    let dist = |u: [f64; 3], v: [f64; 3]| {
        ((u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2) + (u[2] - v[2]).powi(2)).sqrt()
    };
    let angle = |u: [f64; 3], v: [f64; 3], w: [f64; 3]| {
        let e1 = [v[0] - u[0], v[1] - u[1], v[2] - u[2]];
        let e2 = [w[0] - u[0], w[1] - u[1], w[2] - u[2]];
        let d = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (dist(u, v) * dist(u, w) + 1e-30);
        d.clamp(-1.0, 1.0).acos().to_degrees()
    };
    let angles: Vec<f64> = mesh
        .faces
        .iter()
        .map(|f| {
            let (a, b, c) = (
                mesh.points[f.tri[0]],
                mesh.points[f.tri[1]],
                mesh.points[f.tri[2]],
            );
            angle(a, b, c).min(angle(b, c, a)).min(angle(c, a, b))
        })
        .collect();
    let min_ang = angles.iter().copied().fold(f64::MAX, f64::min);
    let min_ang = if min_ang.is_finite() { min_ang } else { 0.0 };
    let n_bad = angles.iter().filter(|&&a| a < 15.0).count();
    log::stat("surface.faces", mesh.faces.len() as f64);
    log::stat("surface.points", mesh.points.len() as f64);
    log::stat("surface.min_angle_deg", min_ang);
    log::stat("surface.thin", n_bad as f64);
    log::info(
        "metrics",
        format!("faces {}  points {}", mesh.faces.len(), mesh.points.len()),
    );
    let lvl = if n_bad > 0 {
        log::Level::Warn
    } else {
        log::Level::Info
    };
    log::event(
        lvl,
        "metrics",
        format!("min-angle {min_ang:.1} deg   {n_bad} below 15 deg"),
    );
}
