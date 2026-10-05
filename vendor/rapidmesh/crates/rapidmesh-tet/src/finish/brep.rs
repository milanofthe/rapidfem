//! The finish of a bottom-up mesh on its B-rep: the boundary snapped onto
//! the analytic carriers and edge curves ([`snap`](crate::finish::snap)), the worst
//! tets repaired and the surface relaxed ([`improve`](crate::finish::improve)), and
//! the result handed out as a [`TetMesh`] in B-rep terms.

use crate::mesh::{SurfaceFace, TetMesh};

use crate::curve::kinds::edge_curve;
use crate::curve::{Curve, CurveSamples, PolylineCurve};
use crate::finish::snap::Shape;
use crate::finish::{Complex, PointClass};
use crate::params::MeshParams;
use rapidmesh_brep::{Brep, Curve as BCurve};
use rapidmesh_exact::vector::V3;
use rapidmesh_exact::vector::{bbox, dist};
use rapidmesh_geom::Surface;
use rapidmesh_geom::{RegionTag, TaggedPlc};

/// The analytic shape of a B-rep: face carriers and edge curves.
pub(crate) struct BrepShape<'a> {
    brep: &'a Brep,
    /// Per B-rep edge: its curve and a dense sample of it (none for an
    /// edge whose chain makes no curve).
    curves: Vec<Option<(Box<dyn Curve>, CurveSamples)>>,
    /// Per B-rep edge: whether it has a smooth curve (not a polyline).
    smooth_curve: Vec<bool>,
}

impl<'a> BrepShape<'a> {
    pub(crate) fn new(brep: &'a Brep) -> BrepShape<'a> {
        let has_curve: Vec<bool> = brep
            .edges
            .iter()
            .map(|e| PolylineCurve::new(&e.chain).is_some())
            .collect();
        let curves = brep
            .edges
            .iter()
            .zip(&has_curve)
            .map(|(e, &has)| {
                if !has {
                    return None;
                }
                let c = edge_curve(brep, e)?;
                let len = c.length();
                let n = 4 * e.chain.len().max(16);
                let samples = (0..=n)
                    .map(|i| {
                        let s = len * i as f64 / n as f64;
                        (s, c.point_at(s))
                    })
                    .collect();
                Some((c, CurveSamples::new(samples)))
            })
            .collect();
        let smooth_curve = brep
            .edges
            .iter()
            .zip(&has_curve)
            .map(|(e, &has)| has && !matches!(e.curve, BCurve::Polyline))
            .collect();
        BrepShape {
            brep,
            curves,
            smooth_curve,
        }
    }
}

impl Shape for BrepShape<'_> {
    fn smooth(&self, kind: PointClass) -> bool {
        match kind {
            PointClass::Face(f) => self.brep.faces.get(f as usize).is_some_and(|face| {
                !matches!(self.brep.surface(face.surface), Surface::Discrete(_))
            }),
            PointClass::Edge(c) => self.smooth_curve.get(c as usize).copied().unwrap_or(false),
            _ => false,
        }
    }

    fn project(&self, kind: PointClass, p: V3) -> Option<V3> {
        match kind {
            PointClass::Face(f) => {
                let face = self.brep.faces.get(f as usize)?;
                Some(self.brep.surface(face.surface).closest(p).0)
            }
            PointClass::Edge(c) => {
                let (curve, samples) = self.curves.get(c as usize)?.as_ref()?;
                Some(curve.point_at(samples.closest_arc(curve.as_ref(), p)))
            }
            PointClass::Vertex(_) | PointClass::Interior => None,
        }
    }

    fn param(&self, kind: PointClass, p: V3) -> Option<[f64; 2]> {
        let PointClass::Face(f) = kind else {
            return None;
        };
        let s = self.brep.surface(self.brep.faces.get(f as usize)?.surface);
        s.searches().then(|| s.search_start(p))
    }

    fn project_from(&self, kind: PointClass, p: V3, uv: [f64; 2]) -> Option<(V3, [f64; 2])> {
        let PointClass::Face(f) = kind else {
            return self.project(kind, p).map(|q| (q, uv));
        };
        let (q, uv) = self
            .brep
            .surface(self.brep.faces.get(f as usize)?.surface)
            .closest_near(p, uv);
        Some((q, uv))
    }

    fn project_near_from(&self, kind: PointClass, p: V3, uv: [f64; 2]) -> Option<V3> {
        let PointClass::Face(f) = kind else {
            return self.project_from(kind, p, uv).map(|r| r.0);
        };
        let face = self.brep.faces.get(f as usize)?;
        Some(self.brep.surface(face.surface).toward(p, uv))
    }

    fn project_near(&self, kind: PointClass, p: V3) -> Option<V3> {
        let PointClass::Edge(c) = kind else {
            return self.project(kind, p);
        };
        // The nearest point of the polyline through the dense samples.
        let (_, samples) = self.curves.get(c as usize)?.as_ref()?;
        samples.nearest_on_polyline(p)
    }
}

/// The points of the smooth B-rep edges of `brep`: `p` onto edge `edge`'s
/// exact curve (`None` for a polyline or no curve). Built once, asked per
/// point (the second-order mesh puts the mid-edge nodes of curved edges
/// there).
pub fn edge_projection(brep: &Brep) -> impl Fn(u32, V3) -> Option<V3> + '_ {
    let shape = BrepShape::new(brep);
    move |edge, p| {
        let kind = PointClass::Edge(edge);
        if shape.smooth(kind) {
            shape.project(kind, p)
        } else {
            None
        }
    }
}

/// Further snap and improve rounds while vertices stay off their shape.
const SNAP_ROUNDS: usize = 2;

/// Tets with a smaller dihedral (degrees) are improved locally after
/// snapping, over at most this many sweeps.
const IMPROVE_BELOW_DEG: f64 = 25.0;
const IMPROVE_PASSES: usize = 8;

/// A raw bottom-up mesh snapped onto the shape of `model`, repaired and
/// relaxed, and returned as a [`TetMesh`].
pub(crate) fn finish_classified(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
    mut c: Complex,
) -> TetMesh {
    use rapidmesh_exact::log as rmlog;
    let (plc, brep) = (&model.plc, &model.brep);
    let t = rapidmesh_exact::clock::Instant::now();
    let shape = BrepShape::new(brep);
    rmlog::stage("finish.shape", t.elapsed().as_secs_f64());
    // The faces of periodic pairs stay as they are: each point is the
    // image of its partner's.
    let frozen: Vec<u32> = params.periodic.iter().flat_map(|pp| [pp.a, pp.b]).collect();
    let (im, left) = crate::finish::improve::finish(
        &mut c,
        &shape,
        IMPROVE_BELOW_DEG,
        IMPROVE_PASSES,
        SNAP_ROUNDS,
        &frozen,
    );
    rmlog::stage("finish.improve", t.elapsed().as_secs_f64());
    rmlog::heap("improve");
    rmlog::stat("finish.snap_left", left as f64);
    rmlog::stat(
        "finish.improve_flips",
        (im.flips23 + im.flips32 + im.flips44) as f64,
    );
    rmlog::stat("finish.improve_moves", im.moves as f64);
    rmlog::stat("finish.improve_bad_before", im.bad_before as f64);
    rmlog::stat("finish.improve_bad_after", im.bad_after as f64);
    // No verification here: `Mesh::diagnostics` checks conformity on
    // demand.
    let t = rapidmesh_exact::clock::Instant::now();
    let mut mesh = to_tet_mesh(&c, plc, brep);
    mesh.periodic_points = periodic_points(&mesh, &params.periodic);
    rmlog::stage("finish.output", t.elapsed().as_secs_f64());
    mesh
}

/// The complex as a [`TetMesh`]: faces carry the B-rep face as patch, its
/// tag and carrier surface; corners come first among the points.
fn to_tet_mesh(c: &Complex, plc: &TaggedPlc, brep: &Brep) -> TetMesh {
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
        plc_points: brep.vertices.len(),
        point_class: c.classes.clone(),
        curve_edges: c
            .edges
            .iter()
            .map(|&(e, edge)| crate::mesh::CurveEdge {
                v: e.map(|v| v as usize),
                edge,
            })
            .collect(),
        periodic_points: Vec::new(),
    }
}

/// Every point on a face `a` of a periodic pair with its image on face `b`
/// (within a millionth of the model size).
fn periodic_points(m: &TetMesh, pairs: &[crate::params::PeriodicPair]) -> Vec<[usize; 2]> {
    use crate::finish::periodic::PointIndex;
    let (lo, hi) = bbox(&m.points);
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
    use rapidmesh_geom::Surface;
    use rapidmesh_geom::{extrude_profile, icosphere, solid_box, ProfileEdge, Scene};
    use std::collections::HashMap;

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
        let sm = crate::mesher::surface_mesh(
            &rapidmesh_brep::Model::new(plc.clone()),
            &MeshParams {
                maxh: 0.2,
                surf_min_angle: 20.0,
                ..Default::default()
            },
        )
        .unwrap();
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
        let sm = crate::mesher::surface_mesh(
            &rapidmesh_brep::Model::new(plc.clone()),
            &MeshParams {
                maxh: 0.8,
                ..Default::default()
            },
        )
        .unwrap();
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
        let sm = crate::mesher::surface_mesh(
            &rapidmesh_brep::Model::new(plc.clone()),
            &MeshParams {
                maxh: 0.5,
                ..Default::default()
            },
        )
        .unwrap();

        // Interior points are projected EXACTLY onto the analytic sphere.
        // Points on the intersection curve (shared with the other sphere) sit
        // off it by at most the facet sagitta, so the max deviation stays
        // small. Verify both.
        let mut curved_faces = 0usize;
        let mut exact_on = 0usize;
        let mut max_dev = 0.0_f64;
        for f in &sm.faces {
            if let Some(Surface::Sphere { frame, radius }) = sm.surfaces[f.surface as usize] {
                let center = frame.o;
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
        // A spline through points of a semicircle, closed by its diameter
        // and extruded (a D-prism). The curved wall is one Extruded
        // surface; its points land exactly on it.
        let r = 1.0;
        let arc: Vec<[f64; 2]> = (1..8)
            .map(|k| {
                let a = std::f64::consts::PI * k as f64 / 8.0;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        let solid = extrude_profile(
            &[[r, 0.0], [-r, 0.0]],
            &[ProfileEdge::Spline(arc), ProfileEdge::Line],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            None,
            1e-2,
        )
        .unwrap();
        let mut scene = Scene::new();
        scene.add_solid(solid);
        let plc = scene.assemble();
        let sm = crate::mesher::surface_mesh(
            &rapidmesh_brep::Model::new(plc.clone()),
            &MeshParams {
                maxh: 0.4,
                ..Default::default()
            },
        )
        .unwrap();

        let mut curved = 0usize;
        let mut exact_on = 0usize;
        let mut max_dev = 0.0_f64;
        for f in &sm.faces {
            if let Some(s @ Surface::Extruded { .. }) = &sm.surfaces[f.surface as usize] {
                curved += 1;
                for &vtx in &f.tri {
                    let p = sm.points[vtx];
                    let q = s.closest(p).0;
                    let dev = rapidmesh_exact::vector::dist(p, q);
                    max_dev = max_dev.max(dev);
                    if dev < 1e-7 {
                        exact_on += 1;
                    }
                }
            }
        }
        assert!(curved > 0, "expected extruded curved faces");
        assert!(exact_on > 0, "interior points lie on the extruded spline");
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
