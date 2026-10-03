//! Region / surface / edge topology with stable ids and per-entity geometry,
//! extracted from the [`Brep`]. This is the read model the hierarchical sizing
//! API (`g.region(...).surf(...).edge(...).maxh/.tol`) navigates: ids are indices
//! into the brep, whose entities go in the order of their origin (owner solid,
//! role of the surface in its shape, see `build::canonicalize`), so an id stays
//! when shapes are added after, the scene turns or a parameter changes that
//! leaves the faces as they are. Each entity carries the geometry
//! a selector needs (a face centroid/normal, an edge midpoint/length) plus the
//! incidence (region -> faces -> edges) so a scope can walk down the hierarchy.

use crate::{Brep, Curve};
use rapidmesh_geom::vec3::{cross, dot, len as norm, sub, V3};
use rapidmesh_geom::TaggedPlc;

/// The kind of an edge's curve, named for the selectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Line,
    Circle,
    Profile,
    Intersection,
    Polyline,
    Ellipse,
    Spline,
}

impl EdgeKind {
    /// Every kind, in the order of their names.
    pub const ALL: [EdgeKind; 7] = [
        EdgeKind::Line,
        EdgeKind::Circle,
        EdgeKind::Ellipse,
        EdgeKind::Spline,
        EdgeKind::Profile,
        EdgeKind::Intersection,
        EdgeKind::Polyline,
    ];

    /// The name of the kind: "line", "circle", "ellipse", "spline",
    /// "profile" (a swept profile's edge), "intersection" (of two curved
    /// surfaces) or "polyline" (no analytic curve).
    pub fn name(self) -> &'static str {
        match self {
            EdgeKind::Line => "line",
            EdgeKind::Circle => "circle",
            EdgeKind::Ellipse => "ellipse",
            EdgeKind::Spline => "spline",
            EdgeKind::Profile => "profile",
            EdgeKind::Intersection => "intersection",
            EdgeKind::Polyline => "polyline",
        }
    }

    /// The kind of a [`EdgeKind::name`].
    pub fn parse(name: &str) -> Option<EdgeKind> {
        EdgeKind::ALL.into_iter().find(|k| k.name() == name)
    }
}

/// One face of the boundary, with its sizing-relevant geometry and incidence.
#[derive(Debug, Clone)]
pub struct FaceTopo {
    /// Area-weighted centroid of the face's PLC facets.
    pub centroid: V3,
    /// Mean normal (front side, `regions[0]`): the area-weighted facet
    /// normals over the area. Unit on a planar face, shorter on a curved
    /// one, near zero on a closed barrel, so a normal selector keeps the
    /// faces that face its way.
    pub normal: V3,
    /// Total facet area.
    pub area: f64,
    /// Materials on the front (`+normal`) and back sides.
    pub regions: [u32; 2],
    /// Face tag (ports, PEC), 0 if untagged.
    pub face_tag: u32,
    /// Index into `plc.surfaces` (the analytic surface).
    pub surface: u32,
    /// Scene-solid owner.
    pub owner: u32,
    /// The role of its surface in the shape it came from (the order each
    /// primitive documents; a box: -z, +z, -y, +y, -x, +x).
    pub role: u32,
    /// Edge ids on this face: of its loops and inside it (sorted,
    /// deduplicated).
    pub edges: Vec<u32>,
    /// Axis-aligned bounding box `[min, max]` of its facets.
    pub bbox: [V3; 2],
}

/// One edge of the boundary, with its sizing-relevant geometry and incidence.
#[derive(Debug, Clone)]
pub struct EdgeTopo {
    /// Chain endpoints (the two corners).
    pub p0: V3,
    pub p1: V3,
    /// Midpoint of the chain (a stable point selector target).
    pub midpoint: V3,
    /// Arc length of the chain polyline.
    pub length: f64,
    /// Analytic curve kind.
    pub kind: EdgeKind,
    /// Face ids meeting along this edge (the radial cycle), sorted, deduplicated.
    pub faces: Vec<u32>,
    /// Axis-aligned bounding box `[min, max]` of its chain.
    pub bbox: [V3; 2],
}

/// The region / face / edge read model. Face and edge ids are indices into
/// `faces` / `edges`; region ids are the material tags.
#[derive(Debug, Clone, Default)]
pub struct Topology {
    /// Distinct meshed region tags (`> 0`), ascending.
    pub regions: Vec<u32>,
    /// Axis-aligned bounding box `[min, max]` of each region (of the faces
    /// bounding it), parallel to `regions`.
    pub region_bbox: Vec<[V3; 2]>,
    pub faces: Vec<FaceTopo>,
    pub edges: Vec<EdgeTopo>,
}

/// Face-selection criteria (`g.surf(id=, tag=, normal=, near=)`). A `None`
/// field is unconstrained; all present fields must hold (AND).
#[derive(Debug, Clone)]
pub struct FaceFilter {
    pub id: Option<u32>,
    pub tag: Option<u32>,
    /// The faces of this scene solid (its index, voids counted), from its
    /// surface `role` if set as well: a face by its origin, which a later
    /// change of the scene keeps.
    pub solid: Option<u32>,
    pub role: Option<u32>,
    /// Keep faces whose mean normal has a component >= `normal_tol` along
    /// this vector (a planar face: the cosine of the angle).
    pub normal: Option<V3>,
    pub normal_tol: f64,
    /// If set, keep only the single face whose centroid is closest to this point.
    pub near: Option<V3>,
}

/// Edge-selection criteria (`g.edge(id=, kind=, between=, near=)`).
#[derive(Debug, Clone, Default)]
pub struct EdgeFilter {
    pub id: Option<u32>,
    pub kind: Option<EdgeKind>,
    /// Keep edges whose incident faces span BOTH of these region tags.
    pub between: Option<(u32, u32)>,
    /// If set, keep only the single edge whose midpoint is closest to this point.
    pub near: Option<V3>,
}

/// Every face; `normal_tol` 0.9 for a normal set later.
impl Default for FaceFilter {
    fn default() -> FaceFilter {
        FaceFilter {
            id: None,
            tag: None,
            solid: None,
            role: None,
            normal: None,
            normal_tol: 0.9,
            near: None,
        }
    }
}

impl FaceFilter {
    /// The face with this id.
    pub fn id(id: u32) -> FaceFilter {
        FaceFilter {
            id: Some(id),
            ..Default::default()
        }
    }

    /// The faces with this face tag.
    pub fn tag(tag: u32) -> FaceFilter {
        FaceFilter {
            tag: Some(tag),
            ..Default::default()
        }
    }

    /// The faces whose mean normal points along `n` (cosine at least 0.9).
    pub fn normal(n: V3) -> FaceFilter {
        FaceFilter {
            normal: Some(n),
            ..Default::default()
        }
    }

    /// The faces of scene solid `solid` from its surface `role` (see
    /// [`FaceTopo::role`]).
    pub fn origin(solid: u32, role: u32) -> FaceFilter {
        FaceFilter {
            solid: Some(solid),
            role: Some(role),
            ..Default::default()
        }
    }

    /// The face whose centroid is nearest `p`.
    pub fn near(p: V3) -> FaceFilter {
        FaceFilter {
            near: Some(p),
            ..Default::default()
        }
    }
}

impl EdgeFilter {
    /// The edge with this id.
    pub fn id(id: u32) -> EdgeFilter {
        EdgeFilter {
            id: Some(id),
            ..Default::default()
        }
    }

    /// The edges of this kind.
    pub fn kind(kind: EdgeKind) -> EdgeFilter {
        EdgeFilter {
            kind: Some(kind),
            ..Default::default()
        }
    }

    /// The edges where regions `a` and `b` meet.
    pub fn between(a: u32, b: u32) -> EdgeFilter {
        EdgeFilter {
            between: Some((a, b)),
            ..Default::default()
        }
    }

    /// The edge whose midpoint is nearest `p`.
    pub fn near(p: V3) -> EdgeFilter {
        EdgeFilter {
            near: Some(p),
            ..Default::default()
        }
    }
}

fn d2(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

/// Reduce `ids` to the single entry whose `pos` is nearest `p` (the first on a
/// tie); a no-op if `p` is `None` or `ids` is empty.
fn keep_nearest(ids: &mut Vec<u32>, p: Option<V3>, pos: impl Fn(u32) -> V3) {
    if let Some(p) = p {
        if let Some(&best) = ids
            .iter()
            .min_by(|&&a, &&b| d2(pos(a), p).partial_cmp(&d2(pos(b), p)).unwrap())
        {
            *ids = vec![best];
        }
    }
}

impl Topology {
    /// Face ids bounding region `tag` (the region on either side of the face).
    pub fn region_faces(&self, tag: u32) -> Vec<u32> {
        self.faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.regions[0] == tag || f.regions[1] == tag)
            .map(|(i, _)| i as u32)
            .collect()
    }

    fn region_ok(f: &FaceTopo, region: Option<u32>) -> bool {
        match region {
            None => true,
            Some(r) => f.regions[0] == r || f.regions[1] == r,
        }
    }

    fn face_ok(id: u32, f: &FaceTopo, ff: &FaceFilter) -> bool {
        if matches!(ff.id, Some(i) if i != id) {
            return false;
        }
        if matches!(ff.tag, Some(t) if t != f.face_tag) {
            return false;
        }
        if matches!(ff.solid, Some(s) if s != f.owner) || matches!(ff.role, Some(r) if r != f.role)
        {
            return false;
        }
        if let Some(n) = ff.normal {
            let nl = norm(n).max(1e-30);
            let d = (f.normal[0] * n[0] + f.normal[1] * n[1] + f.normal[2] * n[2]) / nl;
            if d < ff.normal_tol {
                return false;
            }
        }
        true
    }

    fn edge_ok(&self, id: u32, e: &EdgeTopo, ef: &EdgeFilter) -> bool {
        if matches!(ef.id, Some(i) if i != id) {
            return false;
        }
        if matches!(ef.kind, Some(k) if k != e.kind) {
            return false;
        }
        if let Some((a, b)) = ef.between {
            let mut has_a = false;
            let mut has_b = false;
            for &fid in &e.faces {
                let r = self.faces[fid as usize].regions;
                has_a |= r[0] == a || r[1] == a;
                has_b |= r[0] == b || r[1] == b;
            }
            if !(has_a && has_b) {
                return false;
            }
        }
        true
    }

    /// Resolve a region-level scope: the region tags matching `want` (all if
    /// `None`). `want` is the region id/tag.
    pub fn resolve_regions(&self, want: Option<u32>) -> Vec<u32> {
        self.regions
            .iter()
            .copied()
            .filter(|&t| want.is_none_or(|w| t == w))
            .collect()
    }

    /// Resolve a surf-level scope to face ids: faces on `region` (if set) that
    /// pass `ff`, then the `near` reduction.
    pub fn resolve_faces(&self, region: Option<u32>, ff: &FaceFilter) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .faces
            .iter()
            .enumerate()
            .filter(|(i, f)| Self::region_ok(f, region) && Self::face_ok(*i as u32, f, ff))
            .map(|(i, _)| i as u32)
            .collect();
        keep_nearest(&mut ids, ff.near, |id| self.faces[id as usize].centroid);
        ids
    }

    /// Resolve an edge-level scope to edge ids: edges with at least one incident
    /// face on `region` (if set) and, when `face` is given, passing `face`; the
    /// edge itself must pass `ef`; then the `near` reduction.
    pub fn resolve_edges(
        &self,
        region: Option<u32>,
        face: Option<&FaceFilter>,
        ef: &EdgeFilter,
    ) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..self.edges.len() as u32)
            .filter(|&eid| {
                let e = &self.edges[eid as usize];
                if region.is_some()
                    && !e
                        .faces
                        .iter()
                        .any(|&fid| Self::region_ok(&self.faces[fid as usize], region))
                {
                    return false;
                }
                if let Some(ff) = face {
                    if !e
                        .faces
                        .iter()
                        .any(|&fid| Self::face_ok(fid, &self.faces[fid as usize], ff))
                    {
                        return false;
                    }
                }
                self.edge_ok(eid, e, ef)
            })
            .collect();
        keep_nearest(&mut ids, ef.near, |id| self.edges[id as usize].midpoint);
        ids
    }
}

/// Builds the [`Topology`] from the assembled `plc` (for facet geometry) and its
/// `brep` (for topology + analytic curves).
pub fn extract_topology(plc: &TaggedPlc, brep: &Brep) -> Topology {
    let vtx = |i: u32| plc.vertices[i as usize];

    // Faces: geometry from the PLC facets, edges from every co-edge.
    let mut faces: Vec<FaceTopo> = Vec::with_capacity(brep.faces.len());
    // Every edge on each face: of its loops and inside it.
    let mut face_edges: Vec<Vec<u32>> = vec![Vec::new(); brep.faces.len()];
    for c in &brep.coedges {
        face_edges[c.face.0 as usize].push(c.edge.0);
    }
    for es in &mut face_edges {
        es.sort_unstable();
        es.dedup();
    }
    for (fi, f) in brep.faces.iter().enumerate() {
        let (mut area, mut cen, mut nrm) = (0.0f64, [0.0; 3], [0.0; 3]);
        let mut bbox = EMPTY;
        for &ti in &f.facets {
            let t = plc.triangles[ti as usize];
            let (a, b, c) = (vtx(t[0]), vtx(t[1]), vtx(t[2]));
            for p in [a, b, c] {
                grow(&mut bbox, p);
            }
            let n = cross(sub(b, a), sub(c, a));
            let ar = 0.5 * norm(n);
            area += ar;
            let g = [
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ];
            for k in 0..3 {
                cen[k] += ar * g[k];
                nrm[k] += 0.5 * n[k]; // area-weighted (|n| = 2*area)
            }
        }
        if area > 0.0 {
            for k in 0..3 {
                cen[k] /= area;
            }
        }
        let a = area.max(1e-300);
        let normal = [nrm[0] / a, nrm[1] / a, nrm[2] / a];
        let edges: Vec<u32> = face_edges[fi].clone();
        faces.push(FaceTopo {
            centroid: cen,
            normal,
            area,
            regions: [f.regions[0].0, f.regions[1].0],
            face_tag: f.face_tag.0,
            surface: f.plc_surface,
            owner: f.owner,
            role: f.role,
            edges,
            bbox,
        });
    }

    // Edges: endpoints / length / kind from the chain + curve, faces from coedges.
    let mut edges: Vec<EdgeTopo> = Vec::with_capacity(brep.edges.len());
    for e in &brep.edges {
        let p0 = *e.chain.first().unwrap_or(&[0.0; 3]);
        let p1 = *e.chain.last().unwrap_or(&[0.0; 3]);
        let mut length = 0.0;
        for w in e.chain.windows(2) {
            length += norm(sub(w[1], w[0]));
        }
        // Midpoint at half the arc length along the chain.
        let mut mid = p0;
        let mut acc = 0.0;
        for w in e.chain.windows(2) {
            let seg = norm(sub(w[1], w[0]));
            if acc + seg >= 0.5 * length && seg > 0.0 {
                let t = (0.5 * length - acc) / seg;
                mid = [
                    w[0][0] + t * (w[1][0] - w[0][0]),
                    w[0][1] + t * (w[1][1] - w[0][1]),
                    w[0][2] + t * (w[1][2] - w[0][2]),
                ];
                break;
            }
            acc += seg;
        }
        let kind = match e.curve {
            Curve::Line { .. } => EdgeKind::Line,
            Curve::Circle { .. } => EdgeKind::Circle,
            Curve::Profile { .. } => EdgeKind::Profile,
            Curve::Ellipse { .. } => EdgeKind::Ellipse,
            Curve::Intersection { .. } => EdgeKind::Intersection,
            Curve::Nurbs { .. } => EdgeKind::Spline,
            Curve::Polyline => EdgeKind::Polyline,
        };
        let mut faces_of: Vec<u32> = e.coedges.iter().map(|&c| brep.coedge(c).face.0).collect();
        faces_of.sort_unstable();
        faces_of.dedup();
        let mut bbox = EMPTY;
        for &p in &e.chain {
            grow(&mut bbox, p);
        }
        edges.push(EdgeTopo {
            p0,
            p1,
            midpoint: mid,
            length,
            kind,
            faces: faces_of,
            bbox,
        });
    }

    // Regions: distinct meshed tags (> 0).
    let mut regions: Vec<u32> = faces
        .iter()
        .flat_map(|f| f.regions.into_iter())
        .filter(|&r| r != 0)
        .collect();
    regions.sort_unstable();
    regions.dedup();
    let region_bbox = regions
        .iter()
        .map(|&r| {
            let mut bbox = EMPTY;
            for f in faces.iter().filter(|f| f.regions.contains(&r)) {
                grow(&mut bbox, f.bbox[0]);
                grow(&mut bbox, f.bbox[1]);
            }
            bbox
        })
        .collect();

    Topology {
        regions,
        region_bbox,
        faces,
        edges,
    }
}

/// The empty box, which the first point grows to itself.
const EMPTY: [V3; 2] = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];

fn grow(b: &mut [V3; 2], p: V3) {
    for k in 0..3 {
        b[0][k] = b[0][k].min(p[k]);
        b[1][k] = b[1][k].max(p[k]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{solid_box, Scene};

    #[test]
    fn box_topology_is_one_region_six_faces_twelve_edges() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let brep = crate::build::from_plc(&plc);
        let topo = extract_topology(&plc, &brep);

        assert_eq!(topo.regions, vec![1], "one meshed region");
        assert_eq!(topo.faces.len(), 6, "six box faces");
        assert_eq!(topo.edges.len(), 12, "twelve box edges");
        // Every face is a quad: four boundary edges.
        for (i, f) in topo.faces.iter().enumerate() {
            assert_eq!(f.edges.len(), 4, "face {i} should have 4 edges");
            assert!(
                [6.0, 8.0, 12.0].iter().any(|&a| (f.area - a).abs() < 1e-9),
                "box face area {} unexpected",
                f.area
            );
            assert!(
                f.regions.contains(&1) && f.regions.contains(&0),
                "outer wall separates region 1 from void 0"
            );
        }
        // Every edge is straight (a box) and shared by exactly two faces.
        for (i, e) in topo.edges.iter().enumerate() {
            assert_eq!(e.kind, EdgeKind::Line, "box edge {i} is straight");
            assert_eq!(e.faces.len(), 2, "box edge {i} shared by two faces");
            assert!(
                [2.0, 3.0, 4.0].iter().any(|&l| (e.length - l).abs() < 1e-9),
                "box edge length {} unexpected",
                e.length
            );
        }
        // region 1 is bounded by all six faces.
        assert_eq!(topo.region_faces(1).len(), 6);
    }
}
