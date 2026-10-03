//! Scene assembly: solids (material regions) and embedded sheets into one
//! conforming tagged PLC.
//!
//! All facets are arranged together; every arrangement sub-triangle is
//! classified per side against every solid by its exact barycenter; region
//! priority (later-added solid wins) resolves overlaps. Solid sub-facets
//! survive iff the regions on their two sides differ; sheet sub-facets always
//! survive and carry their face tag. Coincident survivors (a sheet lying on a
//! material interface, or two solids sharing a face) are merged into one
//! facet with combined tags. Vertices are exact until the final snap to f64.

use crate::faceted::{Faceted, SurfaceKind};
use crate::plc::{FaceTag, RegionTag, SurfaceRef, TaggedPlc, SHEET_OWNER};
use rapidmesh_csg::{
    arrange_facets, classify, winding_beside, Placement, PlanarInput, Tri, TriBoxes, VertexPool,
};
use rapidmesh_exact::{Point3, Prepared3};
// Deterministic (seedless) hashers: the weld/merge stages ITERATE these maps,
// and that order decides which coincident vertex wins -- std's RandomState would
// make the assembled PLC (and the whole mesh) vary run to run.
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// Relative (to the scene bounding-box diagonal) tolerance for welding f64
/// twins during scene assembly: exact constructions land within an ulp or two
/// of the input vertices they conceptually equal, and sub-tolerance twins
/// create crease chains the Delaunay can never hold apart. The same epsilon
/// gates the post-weld T-junction repair (so the two stages agree on what
/// "coincident" means).
const WELD_REL_TOL: f64 = 1e-12;

/// Points sharing a coordinate value that make it an axis-aligned plane for
/// the input snapping (a triangle's worth).
const SNAP_PLANE_POINTS: usize = 3;

/// T-junction repair rounds (each round splits every edge that currently has
/// an off-corner vertex on it) before the pass declares divergence. A handful
/// suffices for real geometry; this is a loud backstop, not a silent abandon.
const MAX_REPAIR_ROUNDS: usize = 64;

/// A point strictly inside the sub-triangle `v`, exactly on the plane of its
/// facet (`plane`, an explicit triangle there). The barycenter of implicit
/// vertices has a high degree, and its exact predicates stop being exact once
/// products of tiny inputs (a sin(pi) of 1e-16) underflow; a point of degree
/// one on the explicit plane triangle, at the f64 barycenter and certified
/// inside the sub-triangle, avoids that and is far cheaper. Falls back to the
/// barycenter when the certificate fails (a sliver below f64 resolution).
fn representative(v: [&Point3; 3], plane: &Tri) -> Point3 {
    let bary = || Point3::bary(v[0].clone(), v[1].clone(), v[2].clone());
    let Some(p) = v
        .iter()
        .map(|x| x.approx())
        .collect::<Option<Vec<[f64; 3]>>>()
    else {
        return bary();
    };
    let c: [f64; 3] = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0);
    let [a, b, t] = plane.v;
    let (axis, _) = plane.projection_axis();
    let (i, j) = match axis {
        rapidmesh_exact::Axis::X => (1, 2),
        rapidmesh_exact::Axis::Y => (2, 0),
        rapidmesh_exact::Axis::Z => (0, 1),
    };
    let (e1, e2, d) = (
        [b[i] - a[i], b[j] - a[j]],
        [t[i] - a[i], t[j] - a[j]],
        [c[i] - a[i], c[j] - a[j]],
    );
    let det = e1[0] * e2[1] - e1[1] * e2[0];
    if det == 0.0 {
        return bary();
    }
    let (u, w) = (
        (d[0] * e2[1] - d[1] * e2[0]) / det,
        (e1[0] * d[1] - e1[1] * d[0]) / det,
    );
    let q = Point3::pac(a, b, t, u, w);
    let o = |x: &Point3, y: &Point3, z: &Point3| rapidmesh_exact::orient2d(x, y, z, axis);
    let turn = o(v[0], v[1], v[2]);
    let inside = matches!(turn, Some(s) if s != rapidmesh_exact::Sign::Zero)
        && o(v[0], v[1], &q) == turn
        && o(v[1], v[2], &q) == turn
        && o(v[2], v[0], &q) == turn;
    if inside {
        q
    } else {
        bary()
    }
}

/// True if every vertex of the facet's loops and helper triangles lies exactly
/// on the plane of its first helper triangle.
fn exactly_planar(helpers: &[Tri], facet: &rapidmesh_csg::PlanarFacet) -> bool {
    let Some(t0) = helpers.first() else {
        return true;
    };
    let p = |v: [f64; 3]| Point3::Explicit(v);
    let (a, b, c) = (p(t0.v[0]), p(t0.v[1]), p(t0.v[2]));
    let on = |v: &[f64; 3]| {
        rapidmesh_exact::orient3d(&a, &b, &c, &p(*v)) == Some(rapidmesh_exact::Sign::Zero)
    };
    facet
        .outer
        .iter()
        .chain(facet.holes.iter().flatten())
        .all(on)
        && helpers.iter().flat_map(|t| t.v.iter()).all(on)
}

/// An input the scene could not assemble: the solid (by index, in the order
/// added) or the sheet (by face tag) whose facet the arrangement could not
/// triangulate, and what failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembleError {
    pub solid: Option<usize>,
    pub tag: u32,
    pub message: String,
}

impl std::fmt::Display for AssembleError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.solid {
            Some(k) => write!(f, "solid {k} does not assemble: {}", self.message),
            None => write!(f, "sheet {} does not assemble: {}", self.tag, self.message),
        }
    }
}

impl std::error::Error for AssembleError {}

/// A scene of material solids and embedded sheets.
#[derive(Clone, Default)]
pub struct Scene {
    solids: Vec<Faceted>,
    /// Region each solid resolves to (position = priority, value = tag;
    /// 0 marks a VOID: the carved volume belongs to the background and is
    /// not meshed, its walls survive as boundary patches).
    solid_regions: Vec<u32>,
    next_region: u32,
    sheets: Vec<(Faceted, FaceTag)>,
}

/// Source bookkeeping for one input facet.
struct Src {
    /// Index of the owning solid, `None` for sheet facets.
    solid: Option<usize>,
    /// Sheet face tag (0 for solid facets).
    tag: FaceTag,
    /// Global surface table index.
    surface: u32,
}

impl Scene {
    /// Empty scene.
    pub fn new() -> Scene {
        Scene::default()
    }

    /// Adds a closed, outward-oriented solid; returns its region tag.
    /// On overlap, the solid added later wins.
    pub fn add_solid(&mut self, f: Faceted) -> RegionTag {
        self.next_region += 1;
        self.solids.push(f);
        self.solid_regions.push(self.next_region);
        RegionTag(self.next_region)
    }

    /// Adds a closed, outward-oriented VOID: the volume is carved out of
    /// everything added before it (the cut boolean). A void resolves to the
    /// background region, so its interior is not meshed; its walls survive
    /// as boundary patches that face tags and boundary conditions can
    /// target.
    pub fn add_void(&mut self, f: Faceted) {
        self.solids.push(f);
        self.solid_regions.push(0);
    }

    /// The shape of solid (or void) `i`, in the order they were added.
    pub fn solid(&self, i: usize) -> Option<&Faceted> {
        self.solids.get(i)
    }

    /// Replaces the shape of solid (or void) `i`, keeping its region and
    /// priority.
    pub fn replace_solid(&mut self, i: usize, f: Faceted) {
        self.solids[i] = f;
    }

    /// The shape of sheet `i`, in the order they were added.
    pub fn sheet(&self, i: usize) -> Option<&Faceted> {
        self.sheets.get(i).map(|(f, _)| f)
    }

    /// Replaces the shape of sheet `i`, keeping its face tag.
    pub fn replace_sheet(&mut self, i: usize, f: Faceted) {
        self.sheets[i].0 = f;
    }

    /// Adds an embedded sheet with a face tag (use a nonzero tag).
    pub fn add_sheet(&mut self, f: Faceted, tag: FaceTag) {
        self.sheets.push((f, tag));
    }

    /// Unions solids by retagging every solid in region `from` to `into`: the
    /// boundary between them becomes a same-region internal face (dropped at
    /// assembly), so overlapping solids fuse into one material (a boolean union).
    pub fn merge_region(&mut self, into: RegionTag, from: RegionTag) {
        for r in &mut self.solid_regions {
            if *r == from.0 {
                *r = into.0;
            }
        }
    }

    /// Assembles the conforming tagged PLC; panics where
    /// [`Scene::try_assemble`] reports an error.
    pub fn assemble(&self) -> TaggedPlc {
        self.try_assemble().unwrap_or_else(|e| panic!("{e}"))
    }

    /// Assembles the conforming tagged PLC, or names the input the
    /// arrangement could not triangulate (a solid or a sheet, and what
    /// failed).
    pub fn try_assemble(&self) -> Result<TaggedPlc, AssembleError> {
        let mut plc = self.snapped().assemble_exact()?;
        // The rounded PLC, checked exactly: rounding and welding must not
        // have made triangles cross, fold or touch.
        let t = rapidmesh_exact::clock::Instant::now();
        plc.crossings = rapidmesh_csg::improper_pairs(&plc.vertices, &plc.triangles);
        rapidmesh_exact::log::stage("assemble.check", t.elapsed().as_secs_f64());
        if !plc.crossings.is_empty() {
            let name = |t: u32| {
                let s = plc.surface_refs[t as usize].0 as usize;
                let r = plc.region_tags[t as usize].map(|r| r.0);
                format!(
                    "triangle {t} (solid {}, role {}, regions {r:?})",
                    plc.surface_owners.get(s).copied().unwrap_or(u32::MAX),
                    plc.surface_roles.get(s).copied().unwrap_or(u32::MAX),
                )
            };
            let [a, b] = plc.crossings[0];
            rapidmesh_exact::log::warn(
                "assemble",
                format!(
                    "{} pairs of PLC triangles meet improperly, first {} and {}",
                    plc.crossings.len(),
                    name(a),
                    name(b)
                ),
            );
        }
        Ok(plc)
    }

    /// The scene with its axis-aligned planes snapped: coordinate values
    /// that at least [`SNAP_PLANE_POINTS`] points share, closer than the
    /// weld tolerance to another such value of the same axis, take the
    /// most frequent one (single points, e.g. on a curved surface, stay). The output welds
    /// features below that tolerance anyway; before the exact arrangement
    /// it turns float noise into exact coincidence, so faces meant to be
    /// coplanar are (the end of a barrel at -1.6 + 3.2 and the face of a
    /// flange at 1.42 + 0.18 differ by 4e-16, and the arrangement would
    /// keep the slab between them, whose faces the weld then stacks with
    /// contradicting regions). Triangles that collapse are dropped.
    fn snapped(&self) -> Scene {
        let shapes = || self.solids.iter().chain(self.sheets.iter().map(|(f, _)| f));
        // The distinct points (a corner shared by many triangles counts once).
        let mut seen: HashSet<[u64; 3]> = HashSet::default();
        let mut vals: [Vec<f64>; 3] = Default::default();
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for f in shapes() {
            f.for_each_point(|p| {
                if !seen.insert(p.map(f64::to_bits)) {
                    return;
                }
                for k in 0..3 {
                    vals[k].push(p[k]);
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            });
        }
        let diag = (0..3)
            .map(|k| (hi[k] - lo[k]).max(0.0).powi(2))
            .sum::<f64>()
            .sqrt();
        let tol = WELD_REL_TOL * diag.max(f64::MIN_POSITIVE);
        let maps: Vec<HashMap<u64, f64>> = vals
            .iter_mut()
            .map(|v| {
                v.sort_by(f64::total_cmp);
                let mut map: HashMap<u64, f64> = HashMap::default();
                let mut start = 0;
                while start < v.len() {
                    let mut end = start + 1;
                    while end < v.len() && v[end] - v[end - 1] <= tol {
                        end += 1;
                    }
                    // Runs of equal values in the group; a value shared by
                    // a plane's worth of points is an axis-aligned plane.
                    let group = &v[start..end];
                    let mut planes: Vec<(usize, f64)> = Vec::new();
                    let mut i = 0;
                    while i < group.len() {
                        let mut j = i;
                        while j < group.len() && group[j] == group[i] {
                            j += 1;
                        }
                        if j - i >= SNAP_PLANE_POINTS {
                            planes.push((j - i, group[i]));
                        }
                        i = j;
                    }
                    if planes.len() > 1 {
                        // The most frequent, the first of equally frequent.
                        let to = planes
                            .iter()
                            .fold((0, 0.0), |b, &p| if p.0 > b.0 { p } else { b })
                            .1;
                        for &(_, x) in &planes {
                            map.insert(x.to_bits(), to);
                        }
                    }
                    start = end;
                }
                map
            })
            .collect();
        if maps.iter().all(|m| m.is_empty()) {
            return Scene {
                solids: self.solids.clone(),
                solid_regions: self.solid_regions.clone(),
                next_region: self.next_region,
                sheets: self.sheets.clone(),
            };
        }
        let snap = |p: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|k| maps[k].get(&p[k].to_bits()).copied().unwrap_or(p[k]))
        };
        Scene {
            solids: self.solids.iter().map(|f| f.with_points(snap)).collect(),
            solid_regions: self.solid_regions.clone(),
            next_region: self.next_region,
            sheets: self
                .sheets
                .iter()
                .map(|(f, t)| (f.with_points(snap), *t))
                .collect(),
        }
    }

    /// [`Scene::assemble`] on exactly the input coordinates.
    fn assemble_exact(&self) -> Result<TaggedPlc, AssembleError> {
        // ------------------------------------------------------- flatten
        // Each input shape becomes a list of planar facets for the conformal
        // arrangement: every flat face (FlatFacet) is one boundary-polygon
        // facet carrying its helper triangulation; every remaining (curved)
        // triangle is a single-triangle facet. `rep_tri` is a representative
        // triangle per facet (coplanar with it, same outward normal) for the
        // boundary-coincidence classification.
        let mut facets: Vec<PlanarInput> = Vec::new();
        let mut src: Vec<Src> = Vec::new();
        let mut rep_tri: Vec<Tri> = Vec::new();
        let mut surfaces: Vec<SurfaceKind> = Vec::new();
        let mut surface_owners: Vec<u32> = Vec::new();
        let mut surface_roles: Vec<u32> = Vec::new();
        let mut owner_frames: Vec<crate::Frame> = Vec::new();
        let mut flatten = |f: &Faceted, solid: Option<usize>, tag: FaceTag| {
            if let Some(k) = solid {
                if owner_frames.len() <= k {
                    owner_frames.resize(k + 1, crate::Frame::IDENTITY);
                }
                owner_frames[k] = f.frame;
            }
            let base = surfaces.len() as u32;
            surfaces.extend(f.surfaces.iter().cloned());
            let owner = solid.map_or(SHEET_OWNER, |k| k as u32);
            surface_owners.extend(std::iter::repeat_n(owner, f.surfaces.len()));
            surface_roles.extend(0..f.surfaces.len() as u32);
            // Flat faces first, as boundary polygons with their helper tiling.
            // Only an EXACTLY planar one: a rotated or oblique face rounds off
            // its plane, and a polygon facet whose helper triangles disagree on
            // the plane gives cuts that no constraint recovery can hold. Its
            // triangles (each exactly planar) go in one by one instead, under
            // the same plane surface, so the B-rep still makes one face of them.
            let mut claimed = vec![false; f.tris.len()];
            for fl in &f.flats {
                let helpers: Vec<Tri> = f.tris[fl.tris.clone()].to_vec();
                if !exactly_planar(&helpers, &fl.facet) {
                    rapidmesh_exact::log::stat("assemble.nonplanar_flats", 1.0);
                    continue;
                }
                for i in fl.tris.clone() {
                    claimed[i] = true;
                }
                rep_tri.push(helpers[0]);
                facets.push(PlanarInput {
                    boundary: fl.facet.clone(),
                    helpers,
                });
                src.push(Src {
                    solid,
                    tag,
                    surface: base + fl.surface,
                });
            }
            // Remaining (curved) triangles as single-triangle facets.
            for (i, t) in f.tris.iter().enumerate() {
                if claimed[i] {
                    continue;
                }
                rep_tri.push(*t);
                facets.push(PlanarInput::tri(*t));
                src.push(Src {
                    solid,
                    tag,
                    surface: base + f.face_surface[i],
                });
            }
        };
        for (k, f) in self.solids.iter().enumerate() {
            flatten(f, Some(k), FaceTag(0));
        }
        for (f, tag) in &self.sheets {
            flatten(f, None, *tag);
        }

        let t0 = rapidmesh_exact::clock::Instant::now();
        let arr = arrange_facets(&facets).map_err(|e| AssembleError {
            solid: src[e.facet].solid,
            tag: src[e.facet].tag.0,
            message: e.message,
        })?;
        rapidmesh_exact::log::stage("assemble.arrange", t0.elapsed().as_secs_f64());
        rapidmesh_exact::log::stat("assemble.input_facets", facets.len() as f64);
        let t1 = rapidmesh_exact::clock::Instant::now();

        // Scene bounding box for ray targets, over every facet's geometry.
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for f in &facets {
            for t in &f.helpers {
                for v in &t.v {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k]);
                        hi[k] = hi[k].max(v[k]);
                    }
                }
            }
        }

        // --------------------------------------------- classify and keep
        let mut pool = VertexPool::default();
        let mut triangles: Vec<[u32; 3]> = Vec::new();
        let mut face_tags: Vec<FaceTag> = Vec::new();
        let mut surface_refs: Vec<SurfaceRef> = Vec::new();
        let mut region_tags: Vec<[RegionTag; 2]> = Vec::new();
        // Unordered vertex triple of already-emitted facets, for merging
        // coincident survivors.
        let mut emitted: HashMap<[u32; 3], usize> = HashMap::default();

        // Per-solid bounding boxes (exact: the input tessellations are
        // explicit f64), padded by a fat safety margin against the
        // representative point's approximation error (relative error
        // ~1e-15; the margin is a million times that). A representative
        // clearly outside a solid's padded box is outside the solid, so the
        // ray-parity classification of most (fragment, solid) pairs
        // collapses to three float comparisons.
        let margin = 1e-6 * (0..3).map(|k| hi[k] - lo[k]).fold(1.0_f64, f64::max);
        let solid_bbox: Vec<([f64; 3], [f64; 3])> = self
            .solids
            .iter()
            .map(|f| {
                let mut slo = [f64::MAX; 3];
                let mut shi = [f64::MIN; 3];
                for t in &f.tris {
                    for v in &t.v {
                        for k in 0..3 {
                            slo[k] = slo[k].min(v[k] - margin);
                            shi[k] = shi[k].max(v[k] + margin);
                        }
                    }
                }
                (slo, shi)
            })
            .collect();
        // Padded per-triangle boxes, once per solid (see TriBoxes).
        let solid_boxes: Vec<TriBoxes> = self
            .solids
            .iter()
            .map(|f| TriBoxes::build(&f.tris, margin))
            .collect();

        // Flat list of every sub-triangle (facet index, sub index): the
        // region resolution below is read-only and dominates assembly on
        // boolean-heavy scenes, so it runs in parallel; the cheap emission
        // (vertex pool, dedup) stays sequential in the SAME order, keeping
        // the output bit-identical to the serial pass.
        use rayon::prelude::*;
        let subs: Vec<(usize, usize)> = arr
            .facets
            .iter()
            .enumerate()
            .flat_map(|(fi, ft)| (0..ft.triangles.len()).map(move |si| (fi, si)))
            .collect();
        let regions: Vec<(RegionTag, RegionTag)> = subs
            .par_iter()
            .map(|&(fi, si)| {
                let ft = &arr.facets[fi];
                let s = &src[fi];
                let sub = &ft.triangles[si];
                let bary = representative(
                    [
                        &ft.vertices[sub[0]],
                        &ft.vertices[sub[1]],
                        &ft.vertices[sub[2]],
                    ],
                    &rep_tri[fi],
                );
                let rep = bary
                    .approx()
                    .expect("facet representative must be a valid point");
                let bary = Prepared3::new(bary);

                // Per-side region resolution, highest-priority solid first.
                let mut front: Option<u32> = None;
                let mut back: Option<u32> = None;
                for j in (0..self.solids.len()).rev() {
                    if front.is_some() && back.is_some() {
                        break;
                    }
                    let region = self.solid_regions[j];
                    if s.solid == Some(j) {
                        // Own boundary: the winding just beside it, so a
                        // facet inside an overlap of the solid's own shells
                        // has the solid on both sides.
                        let (wf, wb) = winding_beside(
                            &bary,
                            rep,
                            &rep_tri[fi],
                            &self.solids[j].tris,
                            &solid_boxes[j],
                            (lo, hi),
                        );
                        if wf > 0 {
                            front.get_or_insert(region);
                        }
                        if wb > 0 {
                            back.get_or_insert(region);
                        }
                        continue;
                    }
                    let (blo, bhi) = solid_bbox[j];
                    if (0..3).any(|k| rep[k] < blo[k] || rep[k] > bhi[k]) {
                        continue; // clearly outside solid j
                    }
                    let (in_front, in_back) = match classify(
                        &bary,
                        rep,
                        &rep_tri[fi],
                        &self.solids[j].tris,
                        &solid_boxes[j],
                        (lo, hi),
                    ) {
                        Placement::Inside => (true, true),
                        Placement::Outside => (false, false),
                        // Coincident facets: j's interior lies behind j's
                        // outward normal, i.e. behind ours iff the
                        // normals agree.
                        Placement::Boundary { same_normal } => (!same_normal, same_normal),
                    };
                    if in_front {
                        front.get_or_insert(region);
                    }
                    if in_back {
                        back.get_or_insert(region);
                    }
                }
                (RegionTag(front.unwrap_or(0)), RegionTag(back.unwrap_or(0)))
            })
            .collect();

        for (idx_sub, &(fi, si)) in subs.iter().enumerate() {
            let ft = &arr.facets[fi];
            let s = &src[fi];
            let sub = &ft.triangles[si];
            let (p0, p1, p2) = (
                &ft.vertices[sub[0]],
                &ft.vertices[sub[1]],
                &ft.vertices[sub[2]],
            );
            let (fr, br) = regions[idx_sub];

            // Solid facets survive only as region interfaces; sheets
            // always survive.
            if s.solid.is_some() && fr == br {
                continue;
            }

            let idx: [u32; 3] = [
                pool.insert(p0.clone()) as u32,
                pool.insert(p1.clone()) as u32,
                pool.insert(p2.clone()) as u32,
            ];
            let mut key = idx;
            key.sort_unstable();
            if let Some(&e) = emitted.get(&key) {
                // Coincident facet already emitted: merge tags. Solid
                // interfaces win the region pair (they are equal up to
                // orientation anyway); sheets contribute their face tag.
                face_tags[e] = face_tags[e].max(s.tag);
                continue;
            }
            emitted.insert(key, triangles.len());
            triangles.push(idx);
            face_tags.push(s.tag);
            surface_refs.push(SurfaceRef(s.surface));
            region_tags.push([fr, br]);
        }

        rapidmesh_exact::log::stage("assemble.classify_emit", t1.elapsed().as_secs_f64());
        // ------------------------------------------------- snap and emit
        // The PLC is pure f64 from here on. Exact arithmetic faithfully
        // preserves microscopic input asymmetries (e.g. cos and sin of the
        // same angle rounding differently make concentric-ring "radials"
        // not exactly collinear), so DISTINCT exact points can land on the
        // SAME f64 triple. Weld them, drop facets that collapse, and merge
        // facets that become coincident: zero-f64-area pieces carry no
        // region area, and duplicate indices would poison the mesher.
        let raw: Vec<[f64; 3]> = pool
            .verts
            .iter()
            .map(|p| p.approx().expect("valid point"))
            .collect();
        // Tolerance welding: exact constructions land within an ulp or two
        // of the input vertices they conceptually equal (cos and sin of the
        // same angle round differently, so concentric "radials" are not
        // exactly collinear and their crossings sit ~1e-16-relative off the
        // ring vertices). Sub-tolerance twins create twin crease chains the
        // Delaunay can never hold apart. Features below 1e-12 of the scene
        // diagonal are therefore welded, input vertices winning over
        // constructed points (inputs lie exactly on their planes).
        let diag = (0..3)
            .map(|k| (bhi_w(&raw, k) - blo_w(&raw, k)).powi(2))
            .sum::<f64>()
            .sqrt();
        fn blo_w(raw: &[[f64; 3]], k: usize) -> f64 {
            raw.iter().map(|q| q[k]).fold(f64::MAX, f64::min)
        }
        fn bhi_w(raw: &[[f64; 3]], k: usize) -> f64 {
            raw.iter().map(|q| q[k]).fold(f64::MIN, f64::max)
        }
        let tol = WELD_REL_TOL * diag.max(f64::MIN_POSITIVE);
        let cell = 2.0 * tol;
        let cell_of =
            |q: &[f64; 3]| -> [i64; 3] { std::array::from_fn(|k| (q[k] / cell).floor() as i64) };
        let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::default();
        let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(raw.len());
        let mut remap: Vec<u32> = vec![u32::MAX; raw.len()];
        let weld_pass = |explicit_only: bool,
                         grid: &mut HashMap<[i64; 3], Vec<u32>>,
                         vertices: &mut Vec<[f64; 3]>,
                         remap: &mut Vec<u32>| {
            for (i, q) in raw.iter().enumerate() {
                if remap[i] != u32::MAX {
                    continue;
                }
                if explicit_only && !matches!(pool.verts[i], Point3::Explicit(_)) {
                    continue;
                }
                let base = cell_of(q);
                let mut hit = None;
                'search: for dx in -1..=1i64 {
                    for dy in -1..=1i64 {
                        for dz in -1..=1i64 {
                            let key = [base[0] + dx, base[1] + dy, base[2] + dz];
                            if let Some(ids) = grid.get(&key) {
                                for &v in ids {
                                    let p = vertices[v as usize];
                                    let d2: f64 = (0..3).map(|k| (p[k] - q[k]).powi(2)).sum();
                                    if d2 <= tol * tol {
                                        hit = Some(v);
                                        break 'search;
                                    }
                                }
                            }
                        }
                    }
                }
                remap[i] = hit.unwrap_or_else(|| {
                    let v = vertices.len() as u32;
                    vertices.push(*q);
                    grid.entry(base).or_default().push(v);
                    v
                });
            }
        };
        weld_pass(true, &mut grid, &mut vertices, &mut remap);
        weld_pass(false, &mut grid, &mut vertices, &mut remap);
        let mut out_triangles: Vec<[u32; 3]> = Vec::with_capacity(triangles.len());
        let mut out_face_tags: Vec<FaceTag> = Vec::with_capacity(triangles.len());
        let mut out_surface_refs: Vec<SurfaceRef> = Vec::with_capacity(triangles.len());
        let mut out_region_tags: Vec<[RegionTag; 2]> = Vec::with_capacity(triangles.len());
        let mut emitted_snapped: HashMap<[u32; 3], usize> = HashMap::default();
        for (i, t) in triangles.iter().enumerate() {
            let m = t.map(|v| remap[v as usize]);
            if m[0] == m[1] || m[1] == m[2] || m[0] == m[2] {
                continue; // collapsed to an edge or point
            }
            let (a, b, c) = (
                vertices[m[0] as usize],
                vertices[m[1] as usize],
                vertices[m[2] as usize],
            );
            let u: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
            let v: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            if n.iter().all(|&x| x == 0.0) {
                continue; // exactly degenerate in f64
            }
            let mut key = m;
            key.sort_unstable();
            if let Some(&e) = emitted_snapped.get(&key) {
                // Coincident after welding: merge tags like the exact
                // coincident-survivor merge above.
                out_face_tags[e] = out_face_tags[e].max(face_tags[i]);
                continue;
            }
            emitted_snapped.insert(key, out_triangles.len());
            out_triangles.push(m);
            out_face_tags.push(face_tags[i]);
            out_surface_refs.push(surface_refs[i]);
            out_region_tags.push(region_tags[i]);
        }

        // ------------------------------------------- T-junction repair
        // Welding rounds DISTINCT exact crossings onto the same f64 vertex,
        // which can leave a vertex sitting in the interior of another facet's
        // boundary edge (an approximate T-junction): exactly coplanar with the
        // facet, but ~1e-9 off the edge's carrier LINE. The CDT recovery
        // downstream needs a combinatorially valid PLC (no vertex inside a
        // segment or facet), and adopts only EXACTLY collinear vertices. So we
        // make every such corner explicit here: split the straddled facet at
        // the vertex, turning the micro-kink into two exact straight segments
        // that meet at the (now shared) vertex. After this pass every
        // near-on-edge vertex is a genuine triangle corner of both incident
        // triangles, the input model the CDT assumes.
        repair_t_junctions(
            &vertices,
            &mut out_triangles,
            &mut out_face_tags,
            &mut out_surface_refs,
            &mut out_region_tags,
            tol,
        );
        let segments: Vec<[[f64; 3]; 2]> = self
            .solids
            .iter()
            .chain(self.sheets.iter().map(|(f, _)| f))
            .flat_map(|f| f.features.iter().copied())
            .collect();
        let features = feature_edges(&vertices, &out_triangles, &segments, tol);
        let declared: Vec<[f64; 3]> = self
            .solids
            .iter()
            .chain(self.sheets.iter().map(|(f, _)| f))
            .flat_map(|f| f.corners.iter().copied())
            .collect();
        let corners = corner_vertices(&vertices, &out_triangles, &declared, tol);
        rapidmesh_exact::log::stat("plc.vertices", vertices.len() as f64);
        rapidmesh_exact::log::stat("plc.triangles", out_triangles.len() as f64);
        rapidmesh_exact::log::stat("plc.features", features.len() as f64);

        Ok(TaggedPlc {
            vertices,
            triangles: out_triangles,
            face_tags: out_face_tags,
            surface_refs: out_surface_refs,
            region_tags: out_region_tags,
            surfaces,
            surface_owners,
            surface_roles,
            owner_frames,
            features,
            corners,
            crossings: Vec::new(),
        })
    }
}

/// The PLC vertices at the declared corner points: for each point the
/// nearest vertex of a triangle within `tol` (a corner the assembly cut
/// away has none). Sorted, without repeats.
fn corner_vertices(
    vertices: &[[f64; 3]],
    triangles: &[[u32; 3]],
    points: &[[f64; 3]],
    tol: f64,
) -> Vec<u32> {
    if points.is_empty() {
        return Vec::new();
    }
    let cell = tol.max(1e-300) * 4.0;
    let key = |p: [f64; 3]| p.map(|x| (x / cell).floor() as i64);
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::default();
    let mut used = vec![false; vertices.len()];
    for t in triangles {
        for &v in t {
            used[v as usize] = true;
        }
    }
    for (v, &p) in vertices.iter().enumerate() {
        if used[v] {
            grid.entry(key(p)).or_default().push(v as u32);
        }
    }
    let mut out = Vec::new();
    for &p in points {
        let k = key(p);
        let mut best: Option<(f64, u32)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &v in grid
                        .get(&[k[0] + dx, k[1] + dy, k[2] + dz])
                        .into_iter()
                        .flatten()
                    {
                        let q = vertices[v as usize];
                        let d =
                            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                                .sqrt();
                        if d <= tol && best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, v));
                        }
                    }
                }
            }
        }
        if let Some((_, v)) = best {
            out.push(v);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The PLC edges along each feature segment: from the vertex at one end,
/// step to the neighbour on the segment nearest ahead until the other end
/// (the arrangement may have split the segment). A segment the assembly cut
/// away, wholly or in part, is dropped.
fn feature_edges(
    vertices: &[[f64; 3]],
    triangles: &[[u32; 3]],
    segments: &[[[f64; 3]; 2]],
    tol: f64,
) -> Vec<[u32; 2]> {
    if segments.is_empty() {
        return Vec::new();
    }
    let mut next: HashMap<u32, Vec<u32>> = HashMap::default();
    for t in triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            next.entry(a).or_default().push(b);
            next.entry(b).or_default().push(a);
        }
    }
    for n in next.values_mut() {
        n.sort_unstable();
        n.dedup();
    }
    // Vertices by a grid of the weld tolerance, for the segment ends.
    let cell = tol.max(f64::MIN_POSITIVE);
    let key = |p: [f64; 3]| p.map(|x| (x / cell).floor() as i64);
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::default();
    for &v in next.keys() {
        grid.entry(key(vertices[v as usize])).or_default().push(v);
    }
    let at = |p: [f64; 3]| -> Option<u32> {
        let k = key(p);
        (0..27)
            .map(|n| [k[0] + n % 3 - 1, k[1] + n / 3 % 3 - 1, k[2] + n / 9 - 1])
            .filter_map(|c| grid.get(&c))
            .flatten()
            .copied()
            .find(|&v| (0..3).all(|i| (vertices[v as usize][i] - p[i]).abs() <= tol))
    };
    let mut out: Vec<[u32; 2]> = Vec::new();
    for &[p, q] in segments {
        let (Some(a), Some(b)) = (at(p), at(q)) else {
            continue;
        };
        let d: [f64; 3] = std::array::from_fn(|i| q[i] - p[i]);
        let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if a == b || len2 == 0.0 {
            continue;
        }
        let reach = tol + 1e-9 * len2.sqrt();
        // Parameter along the segment and distance off its line.
        let place = |v: u32| {
            let x = vertices[v as usize];
            let r: [f64; 3] = std::array::from_fn(|i| x[i] - p[i]);
            let t = (r[0] * d[0] + r[1] * d[1] + r[2] * d[2]) / len2;
            let off: f64 = (0..3)
                .map(|i| (r[i] - t * d[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            (t, off)
        };
        let mut path = Vec::new();
        let (mut cur, mut t_cur) = (a, 0.0);
        while cur != b {
            let step = next.get(&cur).and_then(|ns| {
                ns.iter()
                    .map(|&w| (w, place(w)))
                    .filter(|&(_, (t, off))| t > t_cur && t <= 1.0 + 1e-9 && off <= reach)
                    .min_by(|x, y| x.1 .0.total_cmp(&y.1 .0))
            });
            let Some((w, (t, _))) = step else {
                path.clear();
                break;
            };
            path.push([cur.min(w), cur.max(w)]);
            (cur, t_cur) = (w, t);
        }
        out.extend(path);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Makes approximate T-junctions explicit on the welded triangle soup: every
/// vertex sitting in the interior of a triangle edge (within the weld
/// tolerance of the OPEN segment) becomes a shared corner of both incident
/// triangles by splitting that edge across every triangle that carries it.
///
/// Why this exists: the CDT boundary recovery downstream assumes a
/// combinatorially valid PLC (no vertex in a segment or facet interior).
/// Welding distinct exact crossings onto one f64 vertex can violate that: a
/// vertex ends up exactly coplanar with a facet yet a hair off its boundary
/// edge's carrier line. Recovery cannot fuzzily adopt such a vertex without
/// kinking the boundary chain in-plane while the facet region stays the exact
/// straight triangle (which breaks face recovery's straddle-impossibility
/// argument). Splitting the facet here turns the micro-kink into two exact
/// straight segments meeting at the now shared vertex, so recovery only ever
/// sees exactly-collinear chain vertices.
///
/// Per-triangle attributes are duplicated onto the split children. The pass
/// iterates to a fixpoint (a split makes a new edge other vertices may sit
/// on) and is deterministic (candidates are sorted before they are applied).
fn repair_t_junctions(
    vertices: &[[f64; 3]],
    triangles: &mut Vec<[u32; 3]>,
    face_tags: &mut Vec<FaceTag>,
    surface_refs: &mut Vec<SurfaceRef>,
    region_tags: &mut Vec<[RegionTag; 2]>,
    tol: f64,
) {
    for round in 0.. {
        assert!(
            round < MAX_REPAIR_ROUNDS,
            "T-junction repair did not converge in {MAX_REPAIR_ROUNDS} rounds",
        );

        // Unique undirected edges of the current soup, in a spatial grid for
        // the vertex-near-edge search (so it is not O(V*E) on big scenes).
        let mut edge_set: HashSet<(u32, u32)> = HashSet::default();
        for t in triangles.iter() {
            for e in 0..3 {
                let (x, y) = (t[e], t[(e + 1) % 3]);
                edge_set.insert((x.min(y), x.max(y)));
            }
        }
        let mut edges: Vec<(u32, u32)> = edge_set.into_iter().collect();
        edges.sort_unstable();
        let grid = EdgeGrid::build(vertices, edges);

        // Candidate vertices per canonical edge `(a, b)` with `a < b`: every
        // vertex on the open segment that is not an endpoint. An edge can
        // carry many collinear vertices (a long edge crossed by a fence of
        // T-junctions); they are ALL subdivided in this one round, ordered
        // along the edge, so convergence does not depend on how many sit on a
        // single edge.
        let mut edge_verts: HashMap<(u32, u32), Vec<u32>> = HashMap::default();
        let mut buf: Vec<u32> = Vec::new();
        let mut seen: Vec<u32> = vec![0; grid.edges.len()];
        for v in 0..vertices.len() as u32 {
            grid.edges_near(vertices[v as usize], v + 1, &mut seen, &mut buf);
            for &ei in &buf {
                let (a, b) = grid.edges[ei as usize];
                if v == a || v == b {
                    continue;
                }
                if on_open_segment(
                    vertices[a as usize],
                    vertices[b as usize],
                    vertices[v as usize],
                    tol,
                ) {
                    edge_verts.entry((a, b)).or_default().push(v);
                }
            }
        }
        if edge_verts.is_empty() {
            break;
        }
        // Order each edge's vertices along a -> b (parameter, then index for
        // determinism) so the subdivided chain is monotone.
        for (&(a, b), vs) in edge_verts.iter_mut() {
            let (pa, pb) = (vertices[a as usize], vertices[b as usize]);
            let d: [f64; 3] = std::array::from_fn(|k| pb[k] - pa[k]);
            let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let param = |w: u32| -> f64 {
                let p = vertices[w as usize];
                (0..3).map(|k| (p[k] - pa[k]) * d[k]).sum::<f64>() / len2
            };
            vs.sort_by(|&x, &y| param(x).total_cmp(&param(y)).then(x.cmp(&y)));
            vs.dedup();
        }

        // Micro-edge prevention: only the ENDPOINTS of an edge protect their
        // neighborhood (on_open_segment excludes them); the subdivision
        // vertices themselves can sit arbitrarily close to EACH OTHER.
        // Splitting would bake micro edges into the soup whose endpoints
        // later reach the Delaunay as near-duplicate inserts and swallow
        // vertex stars. Weld each sub-tolerance cluster to its lowest
        // member (union-find over consecutive pairs), rewrite the soup, and
        // restart the round on the welded triangles.
        let mut remap: HashMap<u32, u32> = HashMap::default();
        fn find(m: &HashMap<u32, u32>, mut v: u32) -> u32 {
            while let Some(&r) = m.get(&v) {
                v = r;
            }
            v
        }
        for vs in edge_verts.values() {
            for w in vs.windows(2) {
                let (x, y) = (find(&remap, w[0]), find(&remap, w[1]));
                if x == y {
                    continue;
                }
                let d2: f64 = (0..3)
                    .map(|k| (vertices[x as usize][k] - vertices[y as usize][k]).powi(2))
                    .sum();
                if d2 < tol * tol {
                    remap.insert(x.max(y), x.min(y));
                }
            }
        }
        if !remap.is_empty() {
            let mut keep_i = 0usize;
            for ti in 0..triangles.len() {
                let mut t = triangles[ti];
                for v in &mut t {
                    *v = find(&remap, *v);
                }
                // Drop triangles degenerated by the weld.
                if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
                    continue;
                }
                triangles[keep_i] = t;
                face_tags[keep_i] = face_tags[ti];
                surface_refs[keep_i] = surface_refs[ti];
                region_tags[keep_i] = region_tags[ti];
                keep_i += 1;
            }
            triangles.truncate(keep_i);
            face_tags.truncate(keep_i);
            surface_refs.truncate(keep_i);
            region_tags.truncate(keep_i);
            continue;
        }

        // Build edge -> incident triangle indices for application.
        let mut edge_tris: HashMap<(u32, u32), Vec<usize>> = HashMap::default();
        for (ti, t) in triangles.iter().enumerate() {
            for e in 0..3 {
                let (x, y) = (t[e], t[(e + 1) % 3]);
                edge_tris.entry((x.min(y), x.max(y))).or_default().push(ti);
            }
        }

        // Apply edges in deterministic order. A triangle is split at most once
        // per round (its three edges may all be loaded, but a fan split bakes
        // in one edge at a time); an edge whose triangles are already consumed
        // is deferred to the next round, where its edge survives in a child
        // and is re-found. The first edge in order never conflicts, so every
        // round makes progress.
        //
        // A loaded edge can also cap a degenerate sliver: an incident triangle
        // (a, b, x) whose apex x is itself one of the on-edge vertices (the
        // three corners are then collinear within the tolerance, a near
        // zero-area sliver that survived the exact-zero cull by a last-ulp
        // wobble off the line). Such a sliver cannot be fanned (the fan would
        // be degenerate); it is DROPPED instead. The other (non-degenerate)
        // incident triangle's fan reproduces the chain a-v1-..-b, and the
        // sliver's two side edges (a, x) and (x, b) conform in LATER rounds:
        // their own incident triangles survive, and any remaining on-edge
        // vertices lie within tolerance of those sub-edges (they sit on the
        // same carrier line), so the fixpoint sweep re-finds and splits them
        // until the chains match. Watertightness therefore needs no
        // single-vertex restriction on the cap.
        let mut edges_sorted: Vec<(u32, u32)> = edge_verts.keys().copied().collect();
        edges_sorted.sort_unstable();
        let mut consumed: HashSet<usize> = HashSet::default();
        let mut children: HashMap<usize, Vec<[u32; 3]>> = HashMap::default();
        for e in &edges_sorted {
            let vs = &edge_verts[e];
            let Some(tlist) = edge_tris.get(e) else {
                continue;
            };
            if tlist.iter().any(|&ti| consumed.contains(&ti)) {
                continue; // a child carries this edge into the next round
            }
            // An edge may cap ONLY slivers: a degenerate flap on a tangent
            // seam (two barrels touching along a line) whose base edge no
            // real triangle holds — its side edges already belong to the
            // real surface triangles on both sides (the flap made them
            // non-manifold). Dropping every cap is then correct: the flap
            // has no area, and the base chain conforms through the side
            // edges' surviving triangles in later rounds, exactly like the
            // single-cap case.
            let is_sliver = |ti: usize| vs.iter().any(|w| triangles[ti].contains(w));
            for &ti in tlist {
                consumed.insert(ti);
                if is_sliver(ti) {
                    children.insert(ti, Vec::new()); // drop the degenerate cap
                } else {
                    children.insert(ti, split_tri_chain(triangles[ti], e.0, e.1, vs));
                }
            }
        }

        // Rebuild the parallel arrays, replacing each consumed triangle by its
        // children (attributes duplicated), in deterministic index order.
        let mut nt: Vec<[u32; 3]> = Vec::with_capacity(triangles.len());
        let mut nf: Vec<FaceTag> = Vec::with_capacity(triangles.len());
        let mut ns: Vec<SurfaceRef> = Vec::with_capacity(triangles.len());
        let mut nr: Vec<[RegionTag; 2]> = Vec::with_capacity(triangles.len());
        for ti in 0..triangles.len() {
            match children.get(&ti) {
                Some(kids) => {
                    for &k in kids {
                        nt.push(k);
                        nf.push(face_tags[ti]);
                        ns.push(surface_refs[ti]);
                        nr.push(region_tags[ti]);
                    }
                }
                None => {
                    nt.push(triangles[ti]);
                    nf.push(face_tags[ti]);
                    ns.push(surface_refs[ti]);
                    nr.push(region_tags[ti]);
                }
            }
        }
        *triangles = nt;
        *face_tags = nf;
        *surface_refs = ns;
        *region_tags = nr;
    }
}

/// Subdivides triangle `tri` along its edge `{a, b}` by the vertices `vs`
/// (ordered from `a` to `b`), preserving winding: with the edge running
/// `e0 -> e1` in the triangle's cyclic order and opposite corner `o`, the
/// chain `[e0, vs.., e1]` (reversed when the edge runs `b -> a`) fans out from
/// `o` into the triangles `(o, chain[i], chain[i + 1])`.
fn split_tri_chain(tri: [u32; 3], a: u32, b: u32, vs: &[u32]) -> Vec<[u32; 3]> {
    for i in 0..3 {
        let e0 = tri[i];
        let e1 = tri[(i + 1) % 3];
        let o = tri[(i + 2) % 3];
        let fwd = e0 == a && e1 == b;
        let rev = e0 == b && e1 == a;
        if fwd || rev {
            let mut chain = Vec::with_capacity(vs.len() + 2);
            chain.push(e0);
            if fwd {
                chain.extend_from_slice(vs);
            } else {
                chain.extend(vs.iter().rev().copied());
            }
            chain.push(e1);
            return chain.windows(2).map(|w| [o, w[0], w[1]]).collect();
        }
    }
    unreachable!("split edge {a},{b} not found in triangle {tri:?}");
}

/// True if `p` lies within `tol` of the OPEN segment `(a, b)`: its projection
/// parameter is strictly interior with a `tol/len` margin (so `p` is more than
/// `tol` from either endpoint, which is the welding's job, not ours) and its
/// perpendicular distance to the carrier line is at most `tol`.
fn on_open_segment(a: [f64; 3], b: [f64; 3], p: [f64; 3], tol: f64) -> bool {
    let d: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
    let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if len2 <= 0.0 {
        return false;
    }
    let pa: [f64; 3] = std::array::from_fn(|k| p[k] - a[k]);
    let t = (pa[0] * d[0] + pa[1] * d[1] + pa[2] * d[2]) / len2;
    let margin = tol / len2.sqrt();
    if !(t > margin && t < 1.0 - margin) {
        return false;
    }
    let cr = [
        pa[1] * d[2] - pa[2] * d[1],
        pa[2] * d[0] - pa[0] * d[2],
        pa[0] * d[1] - pa[1] * d[0],
    ];
    let perp2 = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]) / len2;
    perp2 <= tol * tol
}

/// Spatial grid over triangle edges for the vertex-near-edge search: an
/// edge is registered in the cells of samples along it at most half a cell
/// apart, so a long edge costs cells along its length rather than across
/// its bounding box (the fan edges of a large flat face next to fine
/// curved ones). A point within the weld tolerance of an edge lies within a
/// quarter cell plus the tolerance of a sample, so the 27 cells around the
/// point hold the edge. Cell size is the median edge length.
struct EdgeGrid {
    cell: f64,
    origin: [f64; 3],
    map: HashMap<[i64; 3], Vec<u32>>,
    edges: Vec<(u32, u32)>,
}

impl EdgeGrid {
    fn cell_of(&self, p: [f64; 3]) -> [i64; 3] {
        std::array::from_fn(|k| ((p[k] - self.origin[k]) / self.cell).floor() as i64)
    }

    fn build(verts: &[[f64; 3]], edges: Vec<(u32, u32)>) -> EdgeGrid {
        let len = |&(a, b): &(u32, u32)| -> f64 {
            let (pa, pb) = (verts[a as usize], verts[b as usize]);
            (0..3).map(|k| (pa[k] - pb[k]).powi(2)).sum::<f64>().sqrt()
        };
        let mut lens: Vec<f64> = edges.iter().map(len).collect();
        lens.sort_by(f64::total_cmp);
        let median = lens.get(lens.len() / 2).copied().unwrap_or(1.0);
        let cell = if median > 0.0 { median } else { 1.0 };
        let origin = verts.first().copied().unwrap_or([0.0; 3]);
        let mut g = EdgeGrid {
            cell,
            origin,
            map: HashMap::default(),
            edges,
        };
        let mut last: Option<[i64; 3]>;
        for ei in 0..g.edges.len() {
            let (a, b) = g.edges[ei];
            let (pa, pb) = (verts[a as usize], verts[b as usize]);
            let steps = (2.0 * len(&(a, b)) / g.cell).ceil().max(1.0) as usize;
            last = None;
            for i in 0..=steps {
                let t = i as f64 / steps as f64;
                let c = g.cell_of(std::array::from_fn(|k| pa[k] + t * (pb[k] - pa[k])));
                if last != Some(c) {
                    let v = g.map.entry(c).or_default();
                    if v.last() != Some(&(ei as u32)) {
                        v.push(ei as u32);
                    }
                    last = Some(c);
                }
            }
        }
        g
    }

    /// Edges that might pass within the weld tolerance of `p`, from the 27
    /// cells around it, each once: `seen` marks the edges already listed
    /// with `stamp`, a value distinct per query.
    fn edges_near(&self, p: [f64; 3], stamp: u32, seen: &mut [u32], out: &mut Vec<u32>) {
        out.clear();
        let base = self.cell_of(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.map.get(&[base[0] + dx, base[1] + dy, base[2] + dz]) {
                        for &e in v {
                            if seen[e as usize] != stamp {
                                seen[e as usize] = stamp;
                                out.push(e);
                            }
                        }
                    }
                }
            }
        }
    }
}
