//! The B-rep as a [`DomainOracle`], and the B-rep meshing entry point.
//!
//! The oracle is exact on the PLC: regions come from the exact parity
//! classifier, crossings from exact segment-facet tests, feature curves are
//! the B-rep edge chains, and patches are the B-rep faces. Everything the
//! refinement sees is one discrete model, so the consistency contract holds
//! by construction (an analytic carrier trimmed by facets cannot promise
//! that: between a chord and its arc the two disagree). The analytic shape
//! comes back afterwards, when [`snap`](super::snap) moves the boundary
//! vertices onto the carriers and edge curves.

use super::oracle::{Crossing, DomainOracle, FeatureCurve, Patch, SizeField, P3};
use super::refine::{mesh, Params};
use super::snap::Shape;
use super::{Complex, VertexKind};
use crate::brep_mesh::edge_curve;
use crate::conform::{MeshParams, PointClass, SurfaceFace, SurfaceMesh, TetMesh};
use crate::curve::{closest_arc, Curve, PolylineCurve};
use crate::domain::DomainTree;
use geometry_predicates::orient3d;
use rapidmesh_brep::index::FacetBvh;
use rapidmesh_brep::{Brep, Curve as BCurve, Surface};
use rapidmesh_csg::Tri;
use rapidmesh_geom::{RegionTag, TaggedPlc};
use std::sync::Arc;

fn centroid(plc: &TaggedPlc, fi: u32) -> P3 {
    let t = plc.triangles[fi as usize];
    std::array::from_fn(|k| {
        (plc.vertices[t[0] as usize][k]
            + plc.vertices[t[1] as usize][k]
            + plc.vertices[t[2] as usize][k])
            / 3.0
    })
}

fn dist(a: P3, b: P3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A B-rep over its PLC, answering the mesher's geometry questions exactly
/// on the PLC.
pub(crate) struct BrepOracle<'a> {
    plc: &'a TaggedPlc,
    pub(crate) brep: &'a Brep,
    domain: &'a DomainTree,
    /// The model's facet index (facet `i` = PLC triangle `i`).
    bvh: std::sync::Arc<FacetBvh>,
    /// B-rep face of each PLC facet (`u32::MAX` for none).
    facet_face: Vec<u32>,
    bbox: (P3, P3),
    patches: Vec<Patch>,
    corners: Vec<P3>,
    corner_patches: Vec<Vec<u32>>,
    curves: Vec<FeatureCurve>,
    /// The B-rep edge of each curve.
    pub(crate) curve_edge: Vec<u32>,
}

impl<'a> BrepOracle<'a> {
    pub(crate) fn new(
        plc: &'a TaggedPlc,
        brep: &'a Brep,
        domain: &'a DomainTree,
        params: &MeshParams,
    ) -> BrepOracle<'a> {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for p in &plc.vertices {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let extent = dist(lo, hi).max(1e-300);
        let mut facet_face = vec![u32::MAX; plc.triangles.len()];
        for (fid, f) in brep.faces.iter().enumerate() {
            for &ti in &f.facets {
                facet_face[ti as usize] = fid as u32;
            }
        }
        let bvh = domain.index().clone();
        let patches = brep
            .faces
            .iter()
            .map(|f| Patch {
                regions: [f.regions[0].0, f.regions[1].0],
                tag: f.face_tag.0,
            })
            .collect();

        let mut corners: Vec<P3> = brep.vertices.iter().map(|v| v.pos).collect();
        // Incidence from the B-rep: a corner lies on every face at its
        // vertex, a curve on every face with a co-edge along it.
        let mut corner_patches: Vec<Vec<u32>> = brep
            .vertices
            .iter()
            .map(|v| v.faces.iter().map(|f| f.0).collect())
            .collect();
        // A singular point inside a face (a cone apex) is a corner too: the
        // refinement only approaches it, so it must be pinned. It counts when
        // the face's PLC has a vertex there.
        for (fi, f) in brep.faces.iter().enumerate() {
            let Some(p) = brep.surface(f.surface).singular_point() else {
                continue;
            };
            let on_plc = f.facets.iter().any(|&ti| {
                plc.triangles[ti as usize]
                    .iter()
                    .any(|&v| dist(plc.vertices[v as usize], p) <= 1e-9 * extent)
            });
            if !on_plc {
                continue;
            }
            let k = match corners.iter().position(|&c| dist(c, p) <= 1e-9 * extent) {
                Some(k) => k,
                None => {
                    corners.push(p);
                    corner_patches.push(Vec::new());
                    corners.len() - 1
                }
            };
            if !corner_patches[k].contains(&(fi as u32)) {
                corner_patches[k].push(fi as u32);
            }
        }

        let mut curves = Vec::new();
        let mut curve_edge = Vec::new();
        for (ei, e) in brep.edges.iter().enumerate() {
            let Some(c) = PolylineCurve::new(&e.chain) else {
                continue;
            };
            let mut fs: Vec<u32> = e.coedges.iter().map(|&c| brep.coedge(c).face.0).collect();
            fs.sort_unstable();
            fs.dedup();
            curves.push(FeatureCurve {
                curve: Arc::new(c),
                ends: [Some(e.ends[0].0), Some(e.ends[1].0)],
                patches: fs,
                max_size: params.edge_maxh_for(ei),
                deflection: Some(params.edge_tol_for(ei)),
            });
            curve_edge.push(ei as u32);
        }
        BrepOracle {
            plc,
            brep,
            domain,
            bvh,
            facet_face,
            bbox: (lo, hi),
            patches,
            corners,
            corner_patches,
            curves,
            curve_edge,
        }
    }

    /// True if the segment `a b` provably misses every facet: it lies in the
    /// union of the boundary-free balls the sizing tree knows around its
    /// ends (a margin covers the rounding of the distances).
    fn clear(&self, a: P3, b: P3) -> bool {
        let reach = self.domain.clearance(a) + self.domain.clearance(b);
        reach * (1.0 - 1e-9) > dist(a, b)
    }

    /// The crossing of the open segment `(a, b)` with PLC facet `fi`, if it
    /// crosses strictly: an endpoint on the plane or a coplanar segment is a
    /// touch, not a crossing. A crossing through an edge or vertex counts
    /// for every facet there.
    fn crossing_with(&self, fi: u32, a: P3, b: P3) -> Option<Crossing> {
        let f = self.facet_face[fi as usize];
        if f == u32::MAX {
            return None;
        }
        let t = self.plc.triangles[fi as usize];
        let (p, q, r) = (
            self.plc.vertices[t[0] as usize],
            self.plc.vertices[t[1] as usize],
            self.plc.vertices[t[2] as usize],
        );
        let (oa, ob) = (orient3d(p, q, r, a), orient3d(p, q, r, b));
        if !((oa > 0.0 && ob < 0.0) || (oa < 0.0 && ob > 0.0)) {
            return None;
        }
        let (s0, s1, s2) = (
            orient3d(a, b, p, q),
            orient3d(a, b, q, r),
            orient3d(a, b, r, p),
        );
        let inside = (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0) || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0);
        if !inside {
            return None;
        }
        let tt = (oa / (oa - ob)).clamp(0.0, 1.0);
        Some(Crossing {
            t: tt,
            point: std::array::from_fn(|k| a[k] + tt * (b[k] - a[k])),
            patch: f,
        })
    }

    /// Singular points pinned as corners beyond the B-rep vertices.
    pub(crate) fn extra_corners(&self) -> usize {
        self.corners.len() - self.brep.vertices.len()
    }
}

impl DomainOracle for BrepOracle<'_> {
    fn bbox(&self) -> (P3, P3) {
        self.bbox
    }

    fn region(&self, p: P3) -> u32 {
        self.domain.region_at(p)
    }

    /// Exact transversal crossings with the PLC facets. A crossing through
    /// a facet edge or vertex is reported for every facet touching it (the
    /// tests are inclusive), so none is ever missed.
    fn crossings(&self, a: P3, b: P3, out: &mut Vec<Crossing>) {
        if self.clear(a, b) {
            return;
        }
        let start = out.len();
        let mut cand: Vec<u32> = Vec::new();
        self.bvh.facets_near_segment(a, b, 0.0, &mut cand);
        out.extend(
            cand.into_iter()
                .filter_map(|fi| self.crossing_with(fi, a, b)),
        );
        out[start..].sort_by(|x, y| x.t.total_cmp(&y.t).then(x.patch.cmp(&y.patch)));
    }

    fn clearance(&self, p: P3) -> f64 {
        self.domain.clearance(p)
    }

    fn nearest_crossing(&self, a: P3, b: P3, focus: P3) -> Option<Crossing> {
        if self.clear(a, b) {
            return None;
        }
        self.nearest_crossing_screened(a, b, focus)
    }

    fn nearest_crossing_screened(&self, a: P3, b: P3, focus: P3) -> Option<Crossing> {
        let key = |c: &Crossing| (dist(c.point, focus), c.t, c.patch);
        let mut best: Option<(f64, f64, u32, Crossing)> = None;
        let mut cand: Vec<u32> = Vec::new();
        self.bvh.facets_near_segment(a, b, 0.0, &mut cand);
        for fi in cand {
            let Some(c) = self.crossing_with(fi, a, b) else {
                continue;
            };
            let (d, t, p) = key(&c);
            if best.is_none_or(|(bd, bt, bp, _)| (d, t, p) < (bd, bt, bp)) {
                best = Some((d, t, p, c));
            }
        }
        best.map(|b| b.3)
    }

    fn patches(&self) -> &[Patch] {
        &self.patches
    }

    fn corners(&self) -> &[P3] {
        &self.corners
    }

    fn curves(&self) -> &[FeatureCurve] {
        &self.curves
    }

    fn corner_patches(&self, corner: u32) -> Vec<u32> {
        self.corner_patches[corner as usize].clone()
    }

    fn patch_within(&self, p: P3, r: f64, except: &[u32]) -> bool {
        let mut cand: Vec<u32> = Vec::new();
        self.bvh.facets_near_segment(p, p, r, &mut cand);
        cand.into_iter().any(|fi| {
            let f = self.facet_face[fi as usize];
            if f == u32::MAX || except.contains(&f) {
                return false;
            }
            let t = self.plc.triangles[fi as usize];
            let v = |i: u32| self.plc.vertices[i as usize];
            rapidmesh_brep::index::point_tri_dist2(p, &Tri::new(v(t[0]), v(t[1]), v(t[2]))) < r * r
        })
    }

    /// Farthest-point samples over the facet centroids of a face: 32 on a
    /// face without edges (a sphere, a torus), a few on every other one, so
    /// no face depends on its boundary curves alone (a circle's samples are
    /// coplanar and span no tet). A fixed stride over a structured
    /// tessellation can pick cocircular points, whose tets are degenerate.
    fn patch_seeds(&self, patch: u32) -> Vec<P3> {
        let face = &self.brep.faces[patch as usize];
        if face.facets.is_empty() {
            return Vec::new();
        }
        let has_edge = self.brep.coedges.iter().any(|c| c.face.0 == patch);
        let cents: Vec<P3> = face
            .facets
            .iter()
            .map(|&fi| centroid(self.plc, fi))
            .collect();
        let n = (if has_edge { 8 } else { 32 }).min(cents.len());
        let mut chosen = vec![0usize];
        let mut d: Vec<f64> = cents.iter().map(|&c| dist(c, cents[0])).collect();
        while chosen.len() < n {
            let (best, _) =
                d.iter().enumerate().fold(
                    (0, -1.0),
                    |acc, (i, &x)| if x > acc.1 { (i, x) } else { acc },
                );
            chosen.push(best);
            for (i, &c) in cents.iter().enumerate() {
                d[i] = d[i].min(dist(c, cents[best]));
            }
        }
        chosen.iter().map(|&i| cents[i]).collect()
    }
}

/// The analytic shape of a B-rep: face carriers and edge curves.
pub(crate) struct BrepShape<'a> {
    brep: &'a Brep,
    /// Per oracle curve: the analytic edge curve and a dense sample of it.
    curves: Vec<Option<(Box<dyn Curve>, Vec<(f64, P3)>)>>,
    /// Per oracle curve: whether it is smooth (not a polyline).
    smooth_curve: Vec<bool>,
}

impl<'a> BrepShape<'a> {
    pub(crate) fn new(brep: &'a Brep, oracle: &BrepOracle<'_>) -> BrepShape<'a> {
        let curves = oracle
            .curve_edge
            .iter()
            .map(|&ei| {
                let c = edge_curve(brep, &brep.edges[ei as usize])?;
                let len = c.length();
                let n = 4 * brep.edges[ei as usize].chain.len().max(16);
                let samples = (0..=n)
                    .map(|i| {
                        let s = len * i as f64 / n as f64;
                        (s, c.point_at(s))
                    })
                    .collect();
                Some((c, samples))
            })
            .collect();
        let smooth_curve = oracle
            .curve_edge
            .iter()
            .map(|&ei| !matches!(brep.edges[ei as usize].curve, BCurve::Polyline))
            .collect();
        BrepShape {
            brep,
            curves,
            smooth_curve,
        }
    }
}

impl Shape for BrepShape<'_> {
    fn smooth(&self, kind: VertexKind) -> bool {
        match kind {
            VertexKind::Patch(f) => self.brep.faces.get(f as usize).is_some_and(|face| {
                !matches!(self.brep.surface(face.surface), Surface::Discrete(_))
            }),
            VertexKind::Curve(c) => self.smooth_curve.get(c as usize).copied().unwrap_or(false),
            _ => false,
        }
    }

    fn project(&self, kind: VertexKind, p: P3) -> Option<P3> {
        match kind {
            VertexKind::Patch(f) => {
                let face = self.brep.faces.get(f as usize)?;
                Some(self.brep.surface(face.surface).closest(p).0)
            }
            VertexKind::Curve(c) => {
                let (curve, samples) = self.curves.get(c as usize)?.as_ref()?;
                Some(curve.point_at(closest_arc(curve.as_ref(), samples, p)))
            }
            VertexKind::Corner(_) | VertexKind::Volume => None,
        }
    }
}

/// The sizing tree as a [`SizeField`].
struct TreeSize<'a>(&'a DomainTree);

impl SizeField for TreeSize<'_> {
    fn size(&self, p: P3) -> f64 {
        self.0.h_at(p)
    }
}

/// Further snap and improve rounds while vertices stay off their shape.
const SNAP_ROUNDS: usize = 2;

/// Tets with a smaller dihedral (degrees) are improved locally after
/// snapping, over at most this many sweeps.
const IMPROVE_BELOW_DEG: f64 = 25.0;
const IMPROVE_PASSES: usize = 4;
/// The threshold for a layered mesh: its tets between this and
/// [`IMPROVE_BELOW_DEG`] are mostly flat by the thickness of a layer, which
/// no flip or move can change.
const LAYERED_IMPROVE_BELOW_DEG: f64 = 10.0;

/// Meshes a PLC through its B-rep with the restricted-Delaunay core.
pub fn mesh_brep(model: &rapidmesh_brep::Model, params: &MeshParams) -> TetMesh {
    use rapidmesh_exact::log as rmlog;
    let (plc, brep) = (&model.plc, &model.brep);
    let params = &params.with_thickness_caps(plc);
    let t0 = rapidmesh_exact::clock::Instant::now();
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &plc.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let extent = (0..3)
        .map(|k| hi[k] - lo[k])
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    let ts = rapidmesh_exact::clock::Instant::now();
    let domain = crate::cvt::build_sizing_domain(model, params);
    rmlog::stage("mesh3.setup.sizing", ts.elapsed().as_secs_f64());
    let ts = rapidmesh_exact::clock::Instant::now();
    let oracle = BrepOracle::new(plc, brep, &domain, params);
    rmlog::stage("mesh3.setup.oracle", ts.elapsed().as_secs_f64());
    rmlog::stage("mesh3.setup", t0.elapsed().as_secs_f64());

    let prm = Params {
        radius_edge: params.radius_edge_bound.max(1.0),
        curve_grading: if params.grading > 0.0 {
            params.grading
        } else {
            0.5
        },
        min_size: params.h_floor(extent),
        max_points: params.max_points,
        periodic: params.periodic.clone(),
        ..Params::default()
    };
    let t1 = rapidmesh_exact::clock::Instant::now();
    // Without the thickness bound, a domain of vertical extrusions (layer
    // stacks) is meshed as its plan extruded through its levels, flat tets
    // through each layer; everything else by refinement (so are periodic
    // domains, and every domain with the bound, which resolves each layer).
    let layered = if params.periodic.is_empty() && !(params.cells_across > 0.0) {
        super::layered::mesh(plc, &oracle, &domain)
    } else {
        None
    };
    let is_layered = layered.is_some();
    rmlog::stat("mesh3.layered", f64::from(u8::from(is_layered)));
    let (mut c, st) = match layered {
        Some(c) => (c, Default::default()),
        None => mesh(&oracle, &TreeSize(&domain), &prm),
    };
    rmlog::stage("mesh3.refine", t1.elapsed().as_secs_f64());
    rmlog::stat("mesh3.feature_points", st.feature_points as f64);
    rmlog::stat("mesh3.facet_insertions", st.facet_insertions as f64);
    rmlog::stat("mesh3.cell_insertions", st.cell_insertions as f64);
    rmlog::stat("mesh3.rejected", st.rejected as f64);
    rmlog::stat("mesh3.rejected_in_ball", st.rejected_in_ball as f64);
    rmlog::stat("mesh3.encroached", st.encroached as f64);
    rmlog::stat("mesh3.near_surface", st.near_surface as f64);
    rmlog::stat("mesh3.protection_rounds", st.protection_rounds as f64);
    rmlog::stat(
        "mesh3.protection_unresolved",
        st.protection_unresolved as f64,
    );
    rmlog::stat(
        "mesh3.protection_coincident",
        st.protection_coincident as f64,
    );
    rmlog::stat("mesh3.max_ball", st.max_ball);
    let t2 = rapidmesh_exact::clock::Instant::now();
    let shape = BrepShape::new(brep, &oracle);
    let (ss, im, left) = super::improve::finish(
        &mut c,
        &shape,
        if is_layered {
            LAYERED_IMPROVE_BELOW_DEG
        } else {
            IMPROVE_BELOW_DEG
        },
        IMPROVE_PASSES,
        SNAP_ROUNDS,
        &params
            .periodic
            .iter()
            .flat_map(|pp| [pp.a, pp.b])
            .collect::<Vec<u32>>(),
    );
    rmlog::stage("mesh3.improve", t2.elapsed().as_secs_f64());
    rmlog::stat("mesh3.periodic_rounds", st.periodic_rounds as f64);
    rmlog::stat("mesh3.periodic_left", st.periodic_left as f64);
    rmlog::stat("mesh3.periodic_missed", st.periodic_missed as f64);
    rmlog::stat("mesh3.snap_partial", ss.partial as f64);
    rmlog::stat("mesh3.snap_blocked", ss.blocked as f64);
    rmlog::stat("mesh3.snap_adopted", ss.adopted as f64);
    rmlog::stat("mesh3.snap_left", left as f64);
    rmlog::stat(
        "mesh3.improve_flips",
        (im.flips23 + im.flips32 + im.flips44) as f64,
    );
    rmlog::stat("mesh3.improve_moves", im.moves as f64);
    rmlog::stat("mesh3.improve_peeled", im.peeled as f64);
    rmlog::stat("mesh3.improve_planned", im.planned as f64);
    rmlog::stat("mesh3.improve_replanned", im.replanned as f64);
    rmlog::stat("mesh3.improve_bad_before", im.bad_before as f64);
    rmlog::stat("mesh3.improve_bad_after", im.bad_after as f64);
    // How the raw element size compares with the target, by where the tet
    // sits: touching a feature vertex, touching the surface, or interior.
    {
        let size = TreeSize(&domain);
        let mut buckets = [(0usize, 0usize); 3]; // (tets, below half target)
        for t in &c.tets {
            let p = t.map(|v| c.points[v as usize]);
            let mut l = 0.0;
            for a in 0..4 {
                for b in a + 1..4 {
                    l += dist(p[a], p[b]) / 6.0;
                }
            }
            let cen: P3 = std::array::from_fn(|k| 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k]));
            let kinds = t.map(|v| c.kinds[v as usize]);
            let b = if kinds
                .iter()
                .any(|k| matches!(k, VertexKind::Corner(_) | VertexKind::Curve(_)))
            {
                0
            } else if kinds.iter().any(|k| matches!(k, VertexKind::Patch(_))) {
                1
            } else {
                2
            };
            buckets[b].0 += 1;
            if l < 0.5 * size.size(cen).max(prm.min_size) {
                buckets[b].1 += 1;
            }
        }
        for (name, (n, small)) in ["feature", "surface", "interior"].iter().zip(buckets) {
            rmlog::stat(&format!("mesh3.tets.{name}"), n as f64);
            rmlog::stat(&format!("mesh3.tets.{name}_small"), small as f64);
        }
    }
    // The complex's own invariants, before any later stage touches it.
    let rep = super::verify::check(&c);
    for (name, n) in [
        ("inverted", rep.inverted),
        ("overfull_facets", rep.overfull_facets),
        ("missing_faces", rep.missing_faces),
        ("extra_faces", rep.extra_faces),
        ("wrong_sides", rep.wrong_sides),
        ("unpatched_faces", rep.unpatched_faces),
        ("open_or_nonmanifold_edges", rep.open_or_nonmanifold_edges),
        ("missing_feature_edges", rep.missing_feature_edges),
    ] {
        rmlog::stat(&format!("mesh3.verify.{name}"), n as f64);
    }
    for line in &st.protection_examples {
        rmlog::warn("mesh3", format!("protection at the floor: {line}"));
    }
    for line in describe_defects(&c, &rep) {
        rmlog::warn("mesh3", line);
    }
    let mut mesh = to_tet_mesh(&c, plc, brep, &domain, &oracle);
    mesh.periodic_points = periodic_points(&mesh, &params.periodic);
    rmlog::stat("refine.tets", mesh.tets.len() as f64);
    rmlog::stage("refine.total", t0.elapsed().as_secs_f64());
    mesh
}

/// One line per example offender of a report: where it is, what its
/// vertices are and which faces meet there.
fn describe_defects(c: &Complex, rep: &super::verify::Report) -> Vec<String> {
    let v = |i: u32| format!("{i} {:?} {:?}", c.kinds[i as usize], c.points[i as usize]);
    let mut out = Vec::new();
    for &(r, a, b, n) in &rep.bad_edges {
        let faces: Vec<String> = c
            .faces
            .iter()
            .filter(|f| f.tri.contains(&a) && f.tri.contains(&b) && f.regions.contains(&r))
            .map(|f| {
                let o = f
                    .tri
                    .iter()
                    .copied()
                    .find(|&x| x != a && x != b)
                    .unwrap_or(a);
                format!(
                    "[{:?} patch {} opp {:?}]",
                    f.regions, f.patch, c.kinds[o as usize]
                )
            })
            .collect();
        out.push(format!(
            "region {r} edge with {n} faces: {} -- {} faces {}",
            v(a),
            v(b),
            faces.join(" ")
        ));
    }
    for s in &rep.bad_segments {
        let (a, b) = (c.points[s[0] as usize], c.points[s[1] as usize]);
        out.push(format!(
            "missing feature edge: {} -- {} length {:.3e}",
            v(s[0]),
            v(s[1]),
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        ));
    }
    for f in &rep.bad_faces {
        let face = c.faces.iter().find(|x| x.tri == *f);
        out.push(format!(
            "unpatched face {:?}: {} | {} | {}",
            face.map(|x| x.regions),
            v(f[0]),
            v(f[1]),
            v(f[2])
        ));
    }
    out
}

/// The surface of a model: every B-rep face meshed by the restricted
/// Delaunay core alone (facet refinement, no cells, see
/// [`super::refine::Params::surface_only`]), curves and corners conforming
/// across faces through shared protecting balls, the vertices on their
/// analytic carriers. Each triangle is oriented like the PLC facet of its
/// face it lies on, with that facet's regions (front, back).
///
/// A triangle budget (`surf_target_count`, 0 = none) coarsens the sizes
/// until the count is within it: the mesh is the finer of the field and
/// the budget.
pub fn surface_mesh(model: &rapidmesh_brep::Model, params: &MeshParams) -> SurfaceMesh {
    let target = params.surf_target_count;
    let mut out = surface_once(model, params);
    let mut s = 1.0_f64;
    for _ in 0..BUDGET_ROUNDS {
        let n = out.faces.len();
        if target == 0 || n as f64 <= (1.0 + BUDGET_SLACK) * target as f64 {
            break;
        }
        // Triangles go with the inverse square of the size.
        s *= (n as f64 / target as f64).sqrt();
        out = surface_once(model, &params.scaled(s));
    }
    out
}

/// Remeshes at most this often to meet a triangle budget.
const BUDGET_ROUNDS: usize = 6;
/// A count this fraction over the budget meets it.
const BUDGET_SLACK: f64 = 0.06;

/// The surface size field: the domain's, floored by the surface minimum.
struct SurfaceSize<'a>(&'a DomainTree);

impl SizeField for SurfaceSize<'_> {
    fn size(&self, p: P3) -> f64 {
        self.0.h_at_surf(p)
    }
}

fn surface_once(model: &rapidmesh_brep::Model, params: &MeshParams) -> SurfaceMesh {
    use rapidmesh_exact::log as rmlog;
    let (plc, brep) = (&model.plc, &model.brep);
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &plc.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let extent = (0..3)
        .map(|k| hi[k] - lo[k])
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    let domain = crate::cvt::build_sizing_domain(model, params);
    let oracle = BrepOracle::new(plc, brep, &domain, params);
    let defaults = Params::default();
    let prm = Params {
        facet_angle_deg: if params.surf_min_angle > 0.0 {
            params.surf_min_angle
        } else {
            defaults.facet_angle_deg
        },
        curve_grading: if params.grading > 0.0 {
            params.grading
        } else {
            0.5
        },
        min_size: params.h_floor(extent),
        max_points: params.max_points,
        surface_only: true,
        ..defaults
    };
    let t1 = rapidmesh_exact::clock::Instant::now();
    let (mut c, _) = mesh(&oracle, &SurfaceSize(&domain), &prm);
    rmlog::stage("mesh3.surface", t1.elapsed().as_secs_f64());

    // Vertices onto their carriers (no tets to keep valid).
    let shape = BrepShape::new(brep, &oracle);
    let mut used = vec![false; c.points.len()];
    for f in &c.faces {
        for &v in &f.tri {
            used[v as usize] = true;
        }
    }
    for (v, p) in c.points.iter_mut().enumerate() {
        if used[v] {
            if let Some(q) = shape.project(c.kinds[v], *p) {
                *p = q;
            }
        }
    }

    // Compact to the used points; orient and label by the PLC facet of the
    // face nearest the triangle's centroid.
    let mut remap = vec![usize::MAX; c.points.len()];
    let mut points: Vec<P3> = Vec::new();
    let mut kinds: Vec<VertexKind> = Vec::new();
    for (v, &u) in used.iter().enumerate() {
        if u {
            remap[v] = points.len();
            points.push(c.points[v]);
            kinds.push(c.kinds[v]);
        }
    }
    let index = domain.index();
    let mut faces: Vec<SurfaceFace> = c
        .faces
        .iter()
        .map(|f| {
            let mut tri = f.tri.map(|v| remap[v as usize]);
            let bf = brep.faces.get(f.patch as usize);
            let (mut regions, face_tag, surface) = match bf {
                Some(bf) => (bf.regions, bf.face_tag, bf.plc_surface),
                None => (
                    [RegionTag(f.regions[0]), RegionTag(f.regions[1])],
                    rapidmesh_geom::FaceTag(0),
                    0,
                ),
            };
            let p = tri.map(|v| points[v]);
            let centroid: P3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0);
            let own = |fi: u32| oracle.facet_face[fi as usize] == f.patch;
            if let Some((fi, _)) = index.nearest_where(centroid, &own) {
                let t = plc.triangles[fi as usize].map(|v| plc.vertices[v as usize]);
                use rapidmesh_geom::vec3::{cross, dot, sub};
                let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                let m = cross(sub(t[1], t[0]), sub(t[2], t[0]));
                if dot(n, m) < 0.0 {
                    tri.swap(1, 2);
                }
                regions = plc.region_tags[fi as usize];
            }
            SurfaceFace {
                tri,
                face_tag,
                regions,
                patch: f.patch,
                surface,
            }
        })
        .collect();

    // Flips inside the patches, smoothing of the patch vertices on their
    // carriers and of the curve vertices along their curves; corners and
    // feature edges stay.
    let t2 = rapidmesh_exact::clock::Instant::now();
    let free: Vec<bool> = kinds
        .iter()
        .map(|k| matches!(k, VertexKind::Patch(_) | VertexKind::Curve(_)))
        .collect();
    let fixed: rustc_hash::FxHashSet<(usize, usize)> = c
        .feature_edges
        .iter()
        .map(|&([a, b], _)| (remap[a as usize], remap[b as usize]))
        .filter(|&(a, b)| a != usize::MAX && b != usize::MAX)
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    let mut tris: Vec<super::surfopt::Tri> = faces
        .iter()
        .map(|f| super::surfopt::Tri {
            v: f.tri,
            patch: f.patch,
        })
        .collect();
    super::surfopt::optimize(
        &mut points,
        &mut tris,
        &free,
        &fixed,
        &|v, x| shape.project(kinds[v], x),
        SURFACE_OPT_PASSES,
    );
    for (f, t) in faces.iter_mut().zip(&tris) {
        f.tri = t.v;
    }
    rmlog::stage("mesh3.surface_opt", t2.elapsed().as_secs_f64());
    let curve_edges = c
        .feature_edges
        .iter()
        .filter_map(|&([a, b], curve)| {
            let v = [remap[a as usize], remap[b as usize]];
            (v[0] != usize::MAX && v[1] != usize::MAX).then(|| crate::conform::CurveEdge {
                v,
                edge: oracle.curve_edge[curve as usize],
            })
        })
        .collect();
    SurfaceMesh {
        point_class: kinds
            .iter()
            .map(|&k| point_class(k, brep, &oracle))
            .collect(),
        points,
        faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        curve_edges,
    }
}

/// Rounds of flips and smoothing on a surface mesh.
const SURFACE_OPT_PASSES: usize = 4;

/// What a mesh vertex of `kind` lies on, in B-rep ids.
fn point_class(kind: VertexKind, brep: &Brep, oracle: &BrepOracle<'_>) -> PointClass {
    match kind {
        VertexKind::Corner(i) if (i as usize) < brep.vertices.len() => PointClass::Vertex(i),
        // A pinned singular point (a cone apex) lies on its face.
        VertexKind::Corner(i) => PointClass::Face(
            oracle.corner_patches[i as usize]
                .first()
                .copied()
                .unwrap_or(u32::MAX),
        ),
        VertexKind::Curve(ci) => PointClass::Edge(oracle.curve_edge[ci as usize]),
        VertexKind::Patch(p) => PointClass::Face(p),
        VertexKind::Volume => PointClass::Interior,
    }
}

/// The complex as a [`TetMesh`]: faces carry the B-rep face as patch, its
/// tag and carrier surface; corners come first among the points.
fn to_tet_mesh(
    c: &Complex,
    plc: &TaggedPlc,
    brep: &Brep,
    domain: &DomainTree,
    oracle: &BrepOracle<'_>,
) -> TetMesh {
    let point_class = c
        .kinds
        .iter()
        .map(|&k| point_class(k, brep, oracle))
        .collect();
    let faces = c
        .faces
        .iter()
        .map(|f| {
            let mut tri = f.tri.map(|v| v as usize);
            tri.sort_unstable();
            let (face_tag, surface) = match brep.faces.get(f.patch as usize) {
                Some(bf) => (bf.face_tag, bf.plc_surface),
                None => (rapidmesh_geom::FaceTag(0), 0),
            };
            let (a, b) = (f.regions[0], f.regions[1]);
            SurfaceFace {
                tri,
                face_tag,
                regions: [RegionTag(a.min(b)), RegionTag(a.max(b))],
                patch: f.patch,
                surface,
            }
        })
        .collect();
    TetMesh {
        points: c.points.clone(),
        tets: c.tets.iter().map(|t| t.map(|v| v as usize)).collect(),
        tet_regions: c.regions.iter().map(|&r| RegionTag(r)).collect(),
        faces,
        surfaces: plc.surfaces.clone(),
        surface_owners: plc.surface_owners.clone(),
        plc_points: brep.vertices.len() + oracle.extra_corners(),
        point_size: c.points.iter().map(|&p| domain.h_at(p)).collect(),
        point_class,
        curve_edges: c
            .feature_edges
            .iter()
            .map(|&(e, curve)| crate::conform::CurveEdge {
                v: e.map(|v| v as usize),
                edge: oracle.curve_edge[curve as usize],
            })
            .collect(),
        periodic_points: Vec::new(),
    }
}

/// Every point on a face `a` of a periodic pair with its image on face `b`
/// (within a millionth of the model size).
fn periodic_points(m: &TetMesh, pairs: &[super::periodic::PeriodicPair]) -> Vec<[usize; 2]> {
    use super::periodic::PointIndex;
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &m.points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let tol = 1e-6 * dist(lo, hi).max(1e-300);
    let pos = |v: usize| m.points[v];
    let mut out: Vec<[usize; 2]> = Vec::new();
    for pp in pairs {
        let on = |patch: u32| -> Vec<usize> {
            let mut v: Vec<usize> = m
                .faces
                .iter()
                .filter(|f| f.patch == patch)
                .flat_map(|f| f.tri)
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let mut index = PointIndex::new(tol);
        for v in on(pp.b) {
            index.insert(m.points[v], v);
        }
        for u in on(pp.a) {
            let p = m.points[u];
            let q = [p[0] + pp.shift[0], p[1] + pp.shift[1], p[2] + pp.shift[2]];
            if let Some(w) = index.find(q, &pos, tol) {
                out.push([u, w]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh3::snap::snap;
    use crate::mesh3::verify::check;
    use rapidmesh_geom::SurfaceKind;
    use rapidmesh_geom::{extrude_spline_profile, icosphere, solid_box, NurbsCurve, Scene};
    use std::collections::HashMap;

    fn scene(solids: Vec<rapidmesh_geom::Faceted>) -> TaggedPlc {
        let mut sc = rapidmesh_geom::Scene::new();
        for s in solids {
            sc.add_solid(s);
        }
        sc.assemble()
    }

    /// The oracle consistency contract on random segments through a B-rep.
    fn contract_holds(plc: &TaggedPlc) {
        let params = MeshParams::default();
        let model = rapidmesh_brep::Model::new(plc.clone());
        let brep = &model.brep;
        let domain = crate::cvt::build_sizing_domain(&model, &params);
        let o = BrepOracle::new(plc, brep, &domain, &params);
        assert!(
            o.facet_face.iter().all(|&f| f != u32::MAX),
            "facet without face"
        );
        let (lo, hi) = o.bbox();
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut cr = Vec::new();
        for _ in 0..500 {
            let mut pt =
                || -> P3 { std::array::from_fn(|k| lo[k] - 0.1 + (hi[k] - lo[k] + 0.2) * rnd()) };
            let (a, b) = (pt(), pt());
            cr.clear();
            o.crossings(a, b, &mut cr);
            // The near-first search finds what the full list says is nearest.
            let focus = pt();
            let want = cr.iter().copied().min_by(|x, y| {
                dist(x.point, focus)
                    .total_cmp(&dist(y.point, focus))
                    .then(x.t.total_cmp(&y.t))
                    .then(x.patch.cmp(&y.patch))
            });
            assert_eq!(o.nearest_crossing(a, b, focus), want, "a {a:?} b {b:?}");
            // Walk: the region flips across every interface crossing and is
            // constant between crossings.
            let mut ts: Vec<f64> = vec![0.0];
            ts.extend(cr.iter().map(|c| c.t));
            ts.push(1.0);
            let mut reg = o.region(a);
            for (k, c) in cr.iter().enumerate() {
                let p = o.patches()[c.patch as usize];
                let mt = 0.5 * (ts[k + 1] + ts[k + 2]);
                let m: P3 = std::array::from_fn(|j| a[j] + mt * (b[j] - a[j]));
                let next = o.region(m);
                let ok = if p.is_sheet() {
                    next == reg
                } else {
                    (p.regions[0] == reg && p.regions[1] == next)
                        || (p.regions[1] == reg && p.regions[0] == next)
                };
                assert!(
                    ok,
                    "a {a:?} b {b:?} crossing {c:?} {:?}: {reg} -> {next}",
                    p.regions
                );
                reg = next;
            }
            assert_eq!(o.region(b), reg, "a {a:?} b {b:?} {cr:?}");
        }
    }

    #[test]
    fn brep_oracles_keep_the_contract() {
        use rapidmesh_geom::{cylinder, solid_box, sphere};
        contract_holds(&scene(vec![solid_box([0.0; 3], [1.0, 0.8, 0.6])]));
        contract_holds(&scene(vec![sphere([0.5; 3], 0.4, 24, 12)]));
        contract_holds(&scene(vec![
            solid_box([0.0; 3], [1.0, 1.0, 1.0]),
            cylinder([0.5, 0.5, -0.2], [0.0, 0.0, 1.4], 0.25, 24),
        ]));
    }

    #[test]
    fn a_box_meshes_into_a_valid_complex() {
        let plc = scene(vec![rapidmesh_geom::solid_box([0.0; 3], [1.0, 0.8, 0.6])]);
        let params = MeshParams {
            maxh: 0.25,
            ..MeshParams::default()
        };
        let model = rapidmesh_brep::Model::new(plc.clone());
        let brep = &model.brep;
        let domain = crate::cvt::build_sizing_domain(&model, &params);
        let o = BrepOracle::new(&plc, brep, &domain, &params);
        let (c, st) = mesh(&o, &TreeSize(&domain), &Params::default());
        let r = check(&c);
        assert!(r.ok(), "{r:?} {st:?}");
        assert!((r.volume(1) - 0.48).abs() < 1e-9, "volume {}", r.volume(1));
    }

    #[test]
    fn a_sphere_snaps_onto_its_carrier() {
        let plc = scene(vec![rapidmesh_geom::sphere([0.0; 3], 1.0, 24, 12)]);
        let params = MeshParams {
            maxh: 0.3,
            ..MeshParams::default()
        };
        let model = rapidmesh_brep::Model::new(plc.clone());
        let brep = &model.brep;
        let domain = crate::cvt::build_sizing_domain(&model, &params);
        let o = BrepOracle::new(&plc, brep, &domain, &params);
        let (mut c, st) = mesh(&o, &TreeSize(&domain), &Params::default());
        let r = check(&c);
        assert!(r.ok(), "{r:?} {st:?}");
        let ss = snap(&mut c, &BrepShape::new(brep, &o));
        let r = check(&c);
        assert!(r.ok(), "after snap: {r:?} {ss:?}");
        let off = c
            .points
            .iter()
            .zip(&c.kinds)
            .filter(|(_, k)| matches!(k, VertexKind::Patch(_)))
            .map(|(p, _)| (dist(*p, [0.0; 3]) - 1.0).abs())
            .fold(0.0, f64::max);
        assert!(
            off < 1e-9 || ss.partial + ss.blocked > 0,
            "off {off} {ss:?}"
        );
        let exact = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(
            // Flat faces with edges near 0.34 inscribed in the unit sphere
            // cut about 4 % of its volume.
            (r.volume(1) - exact).abs() < 0.05 * exact,
            "volume {}",
            r.volume(1)
        );
    }

    /// A block with a round hole through it: the hole wall is a full barrel
    /// (no chart covers it without a seam), yet the surface closes up with
    /// the caps around its rims.
    #[test]
    fn surface_mesh_of_a_drilled_block_is_closed() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 2.0, 1.0]));
        scene.add_void(rapidmesh_geom::cylinder(
            [1.0, 1.0, -0.3],
            [0.0, 0.0, 1.6],
            0.4,
            32,
        ));
        let plc = scene.assemble();
        let sm = surface_mesh(
            &rapidmesh_brep::Model::new(plc.clone()),
            &MeshParams {
                maxh: 0.2,
                surf_min_angle: 20.0,
                ..Default::default()
            },
        );
        let mut count: HashMap<(usize, usize), usize> = HashMap::new();
        for f in &sm.faces {
            for k in 0..3 {
                *count
                    .entry({
                        let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
                        (a.min(b), a.max(b))
                    })
                    .or_insert(0) += 1;
            }
        }
        let open = count.values().filter(|&&c| c != 2).count();
        assert_eq!(open, 0, "{open} edges not shared by exactly two triangles");
    }

    #[test]
    fn surface_mesh_box_is_closed_manifold() {
        // The surface-only export of a closed box is a closed manifold surface:
        // every edge is shared by exactly two triangles, and it covers all six
        // faces (well over a dozen triangles at this size).
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let sm = surface_mesh(
            &rapidmesh_brep::Model::new(plc.clone()),
            &MeshParams {
                maxh: 0.8,
                ..Default::default()
            },
        );
        assert!(
            sm.faces.len() > 12,
            "box surface should be tessellated, got {}",
            sm.faces.len()
        );
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for f in &sm.faces {
            for e in 0..3 {
                let (a, b) = (f.tri[e], f.tri[(e + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        assert!(
            edges.values().all(|&c| c == 2),
            "closed manifold: every edge in exactly 2 faces"
        );
    }

    #[test]
    fn curved_surface_points_lie_on_sphere() {
        // Two overlapping spheres: every interior vertex of a Sphere face sits
        // EXACTLY on its sphere, and the boundary of each region is closed.
        let mut scene = Scene::new();
        scene.add_solid(icosphere([0.0, 0.0, 0.0], 1.0, 2));
        scene.add_solid(icosphere([1.2, 0.0, 0.0], 1.0, 2));
        let plc = scene.assemble();
        let sm = surface_mesh(
            &rapidmesh_brep::Model::new(plc.clone()),
            &MeshParams {
                maxh: 0.5,
                ..Default::default()
            },
        );

        // Interior points are projected EXACTLY onto the analytic sphere.
        // Points on the intersection curve (shared with the other sphere) sit
        // off it by at most the facet sagitta, so the max deviation stays
        // small. Verify both.
        let mut curved_faces = 0usize;
        let mut exact_on = 0usize;
        let mut max_dev = 0.0_f64;
        for f in &sm.faces {
            if let SurfaceKind::Sphere { center, radius } = sm.surfaces[f.surface as usize] {
                curved_faces += 1;
                for &v in &f.tri {
                    let p = sm.points[v];
                    let d = ((p[0] - center[0]).powi(2)
                        + (p[1] - center[1]).powi(2)
                        + (p[2] - center[2]).powi(2))
                    .sqrt();
                    let dev = (d - radius).abs();
                    max_dev = max_dev.max(dev);
                    if dev < 1e-9 {
                        exact_on += 1;
                    }
                }
            }
        }
        assert!(curved_faces > 0, "expected curved faces");
        assert!(exact_on > 0, "interior points lie exactly on the sphere");
        assert!(
            max_dev < 0.05,
            "no vertex grossly off the sphere, max_dev {max_dev}"
        );

        // Per-region closure: the boundary of each region is a closed 2-manifold
        // (every edge shared by exactly two of that region's faces). Edges on the
        // triple curve where three regions meet are manifold within each region
        // but carry three faces overall, which a global 2-manifold test rejects.
        let mut regions: Vec<u32> = sm
            .faces
            .iter()
            .flat_map(|f| [f.regions[0].0, f.regions[1].0])
            .collect();
        regions.sort_unstable();
        regions.dedup();
        for r in regions.into_iter().filter(|&r| r != 0) {
            let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
            for f in sm
                .faces
                .iter()
                .filter(|f| f.regions[0].0 == r || f.regions[1].0 == r)
            {
                for e in 0..3 {
                    let (a, b) = (f.tri[e], f.tri[(e + 1) % 3]);
                    *edges.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
            let bad = edges.values().filter(|&&c| c != 2).count();
            assert_eq!(bad, 0, "region {r} boundary not closed: {bad} edges");
        }
    }

    #[test]
    fn extruded_spline_surface_is_on_the_analytic_surface() {
        // A semicircle profile extruded into a half-cylinder (D-prism). The
        // curved wall is one Extruded surface; its interior points land
        // EXACTLY on the cylinder (radial distance == r).
        let r = 1.0;
        let w = 0.5_f64.sqrt();
        let profile = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
            vec![[r, 0.0], [r, r], [0.0, r], [-r, r], [-r, 0.0]],
            vec![1.0, w, 1.0, w, 1.0],
        );
        let solid = extrude_spline_profile(
            profile,
            24,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        );
        let mut scene = Scene::new();
        scene.add_solid(solid);
        let plc = scene.assemble();
        let sm = surface_mesh(
            &rapidmesh_brep::Model::new(plc.clone()),
            &MeshParams {
                maxh: 0.4,
                ..Default::default()
            },
        );

        let mut curved = 0usize;
        let mut exact_on = 0usize;
        let mut max_dev = 0.0_f64;
        for f in &sm.faces {
            if matches!(
                sm.surfaces[f.surface as usize],
                SurfaceKind::Extruded { .. }
            ) {
                curved += 1;
                for &vtx in &f.tri {
                    let p = sm.points[vtx];
                    let rad = (p[0] * p[0] + p[1] * p[1]).sqrt();
                    let dev = (rad - r).abs();
                    max_dev = max_dev.max(dev);
                    if dev < 1e-7 {
                        exact_on += 1;
                    }
                }
            }
        }
        assert!(curved > 0, "expected extruded curved faces");
        assert!(exact_on > 0, "interior points lie on the cylinder");
        assert!(
            max_dev < 0.02,
            "no curved vertex grossly off radius, max_dev {max_dev}"
        );

        // Per-region closure (single solid: region 1 boundary closed).
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for f in &sm.faces {
            for e in 0..3 {
                let (a, b) = (f.tri[e], f.tri[(e + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let bad = edges.values().filter(|&&c| c != 2).count();
        assert_eq!(bad, 0, "closed manifold, {bad} non-manifold edges");
    }
}
