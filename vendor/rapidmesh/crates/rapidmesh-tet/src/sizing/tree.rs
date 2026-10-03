//! The sizing field of a mesh: one static octree over the domain, refined
//! to the local size h(x), which `h_at(p)` reads off its leaves.
//!
//! h(x) is the smallest of the region's cap and the graded sources: each
//! facet's target, each curved feature edge's target and each point source
//! plus `grading` times the distance to it. So h is Lipschitz and grows
//! smoothly from the fine boundary into the coarse interior.

use crate::params::MeshParams;
use crate::simplex::circumradius;
use crate::sizing::CurvatureLaw;
use rapidmesh_brep::index::{FacetBvh, Targets};
use rapidmesh_brep::Surface;
use rapidmesh_csg::classify::{ray_target, segment_crosses_triangle, RAY_TARGETS};
use rapidmesh_csg::Tri;
use rapidmesh_exact::{Point3, Prepared3};
use rapidmesh_geom::vec3::{dist, V3};
use rapidmesh_geom::{SurfaceKind, TaggedPlc};
use std::sync::Arc;

use crate::constants::DOMAIN_MAX_DEPTH as MAX_DEPTH;

/// A cell of the sizing tree.
enum Node {
    /// The size over the leaf's cell.
    Leaf(f64),
    Inner(Box<[Node; 8]>),
}

const SQRT3: f64 = 1.732_050_807_568_877_2;

/// The walls between regions: the model's facets with the region on either
/// side (an embedded sheet has the same region on both and bounds nothing).
struct Walls {
    index: Arc<FacetBvh>,
    tags: Vec<[u32; 2]>,
}

pub struct DomainTree {
    lo: V3,
    hi: V3,
    root: Node,
    /// The smallest size on surfaces (absolute), applied by
    /// [`DomainTree::h_at_surf`]; `0` is none.
    min_h_surf: f64,
}

fn child_box(center: V3, half: f64, oct: usize) -> (V3, f64) {
    let h = 0.5 * half;
    let c = std::array::from_fn(|k| center[k] + if oct & (1 << k) != 0 { h } else { -h });
    (c, h)
}

impl DomainTree {
    /// Builds the domain octree from the PLC and mesh parameters. `facet_surf`
    /// is an optional per-PLC-facet surface size target (the resolved per-FACE
    /// `surf_maxh`, mapped through the brep); empty means "no per-face override".
    /// `facet_law` likewise holds the curvature law per facet (a per-face
    /// `surf_tol`, the geometric error, else `tol_surf`; see
    /// [`CurvatureLaw`]); empty means `tol_surf` everywhere.
    /// Both feed the volume field too, so a finely sized face refines the
    /// volume behind it.
    pub fn build(
        plc: &TaggedPlc,
        index: Arc<FacetBvh>,
        params: &MeshParams,
        facet_surf: &[f64],
        facet_law: &[CurvatureLaw],
    ) -> DomainTree {
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for p in &plc.vertices {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let diag = (0..3).map(|k| hi[k] - lo[k]).fold(0.0_f64, f64::max);
        let center: V3 = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let half = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max) * 0.5 * 1.0001;
        let bbox = (lo, hi);

        let walls = Walls {
            index: index.clone(),
            tags: plc.region_tags.iter().map(|r| [r[0].0, r[1].0]).collect(),
        };

        // Sizing parameters.
        let maxh = if params.maxh.is_finite() && params.maxh > 0.0 {
            params.maxh
        } else {
            diag / 8.0
        };
        let grading = if params.grading > 0.0 {
            params.grading
        } else {
            0.5
        };

        let region_of = |p: V3| -> u32 { classify(&walls, bbox, p) };
        let region_cap = |r: u32| -> f64 {
            if r == 0 {
                return maxh;
            }
            params
                .region_maxh
                .iter()
                .find(|(rr, _)| *rr == r)
                .map(|&(_, h)| h)
                .unwrap_or(maxh)
        };

        // The SURFACE drives the interior grading. Each boundary facet carries a
        // target edge length `h_target`, the finest of: its face tag's
        // `face_maxh`, its owning solid's `surface_maxh`, the caps of its
        // adjacent regions, else the bulk `maxh`. The sizing field then grows
        // from these wall targets into the interior (Lipschitz with `grading`),
        // so a finely meshed face refines the volume behind it and coarsens away.
        let facet_centroid = |i: usize| -> V3 {
            let t = plc.triangles[i];
            std::array::from_fn(|k| {
                (plc.vertices[t[0] as usize][k]
                    + plc.vertices[t[1] as usize][k]
                    + plc.vertices[t[2] as usize][k])
                    / 3.0
            })
        };

        // Curvature/volume-error target of a curved facet: a facet edge `h` on a
        // surface of principal radius `R` deviates by sagitta ~ h^2/(8R), so
        // bounding the relative sagitta gives `h_curv = R * sqrt(8 * frac)`. This
        // refines the VOLUME near tightly curved boundaries (an airfoil nose), so
        // the surrounding region holds the fine on-surface nodes; the grading
        // term then coarsens away. A gentle curve (R large) leaves `maxh` intact.
        let law_of = |i: usize| {
            facet_law
                .get(i)
                .copied()
                .unwrap_or(CurvatureLaw::Chord(params.tol_surf))
        };
        let curvature_target = |i: usize| -> f64 {
            let kind = &plc.surfaces[plc.surface_refs[i].0 as usize];
            let r = Surface::curved(kind)
                .map_or(f64::INFINITY, |s| s.curvature_radius(facet_centroid(i)));
            law_of(i).surface(r)
        };

        // (cap, target) of a facet: the cap is the user's size there (face,
        // surface, region and global surface caps), the target adds the
        // curvature bound.
        let facet_target = |i: usize| -> (f64, f64) {
            let ft = plc.face_tags[i].0;
            let base = if let Some(&(_, h)) = params.face_maxh.iter().find(|(t, _)| *t == ft) {
                h.min(maxh)
            } else {
                let owner = plc.surface_owners[plc.surface_refs[i].0 as usize];
                if let Some(&(_, h)) = params.surface_maxh.iter().find(|(o, _)| *o == owner) {
                    h.min(maxh)
                } else {
                    let mut h = maxh;
                    for r in plc.region_tags[i] {
                        if r.0 != 0 {
                            h = h.min(region_cap(r.0));
                        }
                    }
                    h
                }
            };
            // Per-FACE override (resolved `surf_maxh`, finest wins), the GLOBAL
            // surface cap (`maxh_surf`), then curvature. `surf_cap()` defaults to
            // `maxh` (no-op) unless a global surface cap is set, so this keeps the
            // global `g.surf().maxh` consistent with the per-entity override: both
            // now refine the volume field behind the surface, not just the tiling.
            //
            // DISCRETE carriers: the estimated (tessellation-derived) curvature
            // may undercut the user's local target by at most 4x. The resolution
            // floor inside the estimate is not enough on its own -- scans are
            // tessellated far finer than any useful FEM resolution, so the
            // estimate otherwise overrides `maxh` unboundedly (measured on
            // cheburashka: h/7, ~1M tets, a 997k-candidate collapse pass).
            // Analytic curvature is exact and keeps its full authority.
            let ct = {
                let kind = &plc.surfaces[plc.surface_refs[i].0 as usize];
                let ct = curvature_target(i);
                if matches!(kind, rapidmesh_geom::SurfaceKind::Discrete(_)) {
                    ct.max(base * 0.25)
                } else {
                    ct
                }
            };
            let cap = base
                .min(facet_surf.get(i).copied().unwrap_or(f64::INFINITY))
                .min(params.surf_cap());
            (cap, cap.min(ct))
        };
        // Per facet: the target, and the cap where it undercuts `maxh` (where
        // the size may jump; INFINITY elsewhere).
        // In parallel: a curved carrier projects each centroid.
        let (targets, caps): (Vec<f64>, Vec<f64>) = {
            use rayon::prelude::*;
            (0..plc.triangles.len())
                .into_par_iter()
                .map(|i| {
                    let (cap, target) = facet_target(i);
                    (target, if cap < maxh { cap } else { f64::INFINITY })
                })
                .unzip()
        };

        // The finest volume target anywhere: the base spacing `s0` of the
        // sizing tree.
        let mut s0 = targets.iter().copied().fold(f64::MAX, f64::min);
        for &(_, sh) in &params.size_points {
            s0 = s0.min(sh);
        }
        // The global volume cap bounds it too, so the interior refines under
        // it, not only the band along the surface.
        s0 = s0.min(params.vol_cap());
        let spacing = if s0.is_finite() && s0 > 0.0 {
            s0
        } else {
            diag / 8.0
        };
        let min_half = (0.5 * spacing).max(1e-9 * diag.max(1.0));

        // The model's facet index with the targets: O(log F) nearest-facet
        // distance and graded-min.
        let targets = Targets::new(&index, targets);

        // The graded sources, each `target + grading * dist`: the facets
        // (caps and curvature, above), the curved feature edges (their
        // curvature, as segments: a degenerate triangle in a `FacetBvh`) and
        // the point sources.
        // The edges take the resolved bulk size, not `params.edge_cap()` raw: with
        // `maxh = INFINITY` (per-dimension caps only) the raw cap is infinite,
        // and the curvature baseline walk inside then never accumulates enough
        // arc length: on a CLOSED rim loop (a cylinder rim, where every vertex
        // has exactly two neighbours and no junction ever breaks the walk) it
        // circles forever.
        // With a geometric error an edge takes the finest law of its facets,
        // else the chord tolerance of the edges.
        let edge_law = |facets: &[u32], r: f64| -> f64 {
            if params.geom_error > 0.0 {
                facets
                    .iter()
                    .map(|&f| law_of(f as usize).curve(r))
                    .fold(f64::INFINITY, f64::min)
            } else {
                CurvatureLaw::Chord(params.tol_edge).curve(r)
            }
        };
        let edge_segments: Vec<(Tri, f64)> =
            edge_sizing_segments(plc, &edge_law, params.edge_cap().min(maxh));
        let edge_bvh = FacetBvh::build(&edge_segments.iter().map(|e| e.0).collect::<Vec<_>>());
        let edge_targets = Targets::new(&edge_bvh, edge_segments.iter().map(|e| e.1).collect());

        // Nearest-facet distance below `r` (see `Parent`).
        let dist_to_boundary = |p: V3, r: f64| -> f64 { index.nearest_dist_within(p, r) };
        // The size at `p` and its graded parts (faces, edges), each exact or
        // `None`. A graded part only counts where it undercuts the other
        // caps, so it is searched below them; and a parent's exact part plus
        // `grading * step` (triangle inequality) bounds it further.
        let h_of =
            |p: V3, region: u32, up: [Option<f64>; 2], step: f64| -> (f64, [Option<f64>; 2]) {
                let caps = region_cap(region).min(maxh).min(params.vol_cap());
                let mut points = f64::INFINITY;
                for (sp, sh) in &params.size_points {
                    points = points.min(sh + grading * dist(p, *sp));
                }
                let limit = caps.min(points);
                let bound =
                    |u: Option<f64>| widen(u.map_or(limit, |u| (u + grading * step).min(limit)));
                let (bf, be) = (bound(up[0]), bound(up[1]));
                let gf = index.graded_min_within(&targets, p, grading, bf);
                let ge = edge_bvh.graded_min_within(&edge_targets, p, grading, be);
                let h = caps.min(gf).min(ge).min(points);
                (h, [(gf < bf).then_some(gf), (ge < be).then_some(ge)])
            };

        // The caps are steps across interfaces, which a leaf's center value
        // smears over the whole leaf; the curvature targets vary smoothly.
        let caps = Targets::new(&index, caps);
        let boundary_min = |p: V3, r: f64| -> f64 { index.min_target_within(&caps, p, r) };
        let queries = NodeQueries {
            region_of: &region_of,
            dist_of: &dist_to_boundary,
            h_of: &h_of,
            boundary_min: &boundary_min,
            min_half,
        };
        let root = build_node(center, half, 0, &queries, None);
        DomainTree {
            lo,
            hi,
            root,
            min_h_surf: params.min_h_surf,
        }
    }

    /// The size at `p` (that of the nearest cell for a point outside the
    /// tree).
    pub fn h_at(&self, p: V3) -> f64 {
        let mut node = &self.root;
        let (mut c, mut h) = (
            std::array::from_fn(|k| 0.5 * (self.lo[k] + self.hi[k])),
            (0..3).map(|k| self.hi[k] - self.lo[k]).fold(0.0, f64::max) * 0.5 * 1.0001,
        );
        loop {
            match node {
                Node::Leaf(size) => return *size,
                Node::Inner(ch) => {
                    let mut oct = 0;
                    for k in 0..3 {
                        if p[k] >= c[k] {
                            oct |= 1 << k;
                        }
                    }
                    (c, h) = child_box(c, h, oct);
                    node = &ch[oct];
                }
            }
        }
    }

    /// The size at `p`, floored by the smallest size on surfaces.
    pub fn h_at_surf(&self, p: V3) -> f64 {
        self.h_at(p).max(self.min_h_surf)
    }
}

/// Sagitta-bounded sizing targets along curved feature edges. A feature edge
/// is a PLC edge between facets of different surfaces, at least one of them
/// curved (the rim of a cylinder at its cap, a plane cutting a sphere). Such a
/// curve may bend tighter than either surface (a small circle near a sphere's
/// pole), which the facet targets would miss. Its radius `R` at a vertex is
/// the circumradius of three points a fixed arc length apart along the edge
/// chain, projected onto the curve; each segment gets `h = R * sqrt(8 *
/// deflection)` (a segment is a degenerate triangle for the `FacetBvh`).
/// Where the edge bends no tighter than its surface, the facet target
/// already wins.
fn edge_sizing_segments(
    plc: &TaggedPlc,
    law: &dyn Fn(&[u32], f64) -> f64,
    maxh: f64,
) -> Vec<(Tri, f64)> {
    use rustc_hash::FxHashMap;
    let key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
    let is_curved = |sid: u32| {
        !matches!(
            plc.surfaces[sid as usize],
            SurfaceKind::Plane { .. } | SurfaceKind::Facets
        )
    };

    // Distinct analytic surfaces meeting along each undirected edge, and
    // the triangles on it.
    let mut edge_surf: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
    let mut edge_tris: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
    for (fi, t) in plc.triangles.iter().enumerate() {
        let s = plc.surface_refs[fi].0;
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let v = edge_surf.entry(key(a, b)).or_default();
            if !v.contains(&s) {
                v.push(s);
            }
            edge_tris.entry(key(a, b)).or_default().push(fi as u32);
        }
    }
    // Feature edges: two distinct surfaces meet, at least one curved, or a
    // sheet ends (one triangle on the edge: the rim of a disc or of a hole
    // in it bends within its plane). Sorted so the downstream segment list
    // (and its BVH) is order-deterministic.
    let mut feature: Vec<(u32, u32)> = edge_surf
        .iter()
        .filter(|(e, s)| {
            (s.len() >= 2 && s.iter().any(|&x| is_curved(x))) || edge_tris[*e].len() == 1
        })
        .map(|(&e, _)| e)
        .collect();
    feature.sort_unstable();

    // The carrier of every surface, a faceted one the plane of its first
    // facet. Needed so POCS can project onto a plane-cut edge, not only the
    // curved side.
    let mut carrier: FxHashMap<u32, Surface> = FxHashMap::default();
    for (fi, t) in plc.triangles.iter().enumerate() {
        let s = plc.surface_refs[fi].0;
        carrier.entry(s).or_insert_with(|| {
            let corners = t.map(|v| plc.vertices[v as usize]);
            Surface::from_kind(&plc.surfaces[s as usize], &corners)
        });
    }
    // Edge-curve neighbours of each feature vertex (its polyline link), and
    // the analytic surfaces meeting along each feature edge, sorted: two
    // links on different surfaces meet at a corner of the curves, not inside
    // one.
    let mut nbr: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for &(a, b) in &feature {
        nbr.entry(a).or_default().push(b);
        nbr.entry(b).or_default().push(a);
    }
    let curve_of = |a: u32, b: u32| -> Vec<u32> {
        let mut s = edge_surf[&key(a, b)].clone();
        s.sort_unstable();
        s
    };
    // Project a point onto the intersection of the analytic surfaces meeting at
    // the edge by alternating projection (POCS) onto BOTH sides -- a plane via its
    // recovered geometry, a curved surface via its closest point. This pulls the faceted
    // chain onto the true curve, so the osculating radius below reflects the REAL
    // curvature of the intersection (INFINITY for a plane-cut generator, the true
    // radius for a genuine curve) -- not the spurious tiny radius a faceted polyline
    // zigzag shows (the over-refinement that fanned out the borders).
    let pocs = |p: V3, sids: &[u32]| -> V3 {
        let mut q = p;
        for _ in 0..8 {
            for &s in sids {
                q = carrier[&s].closest(q).0;
            }
        }
        q
    };
    // Osculating radius at a vertex (only where it has exactly two neighbours -- a
    // smooth interior point; junctions/endpoints stay INFINITY). The radius is
    // sampled with a CONTROLLED step `eps` along the curve tangent, each sample
    // POCS-projected onto the curve, NOT from the raw faceted neighbours: the
    // arrangement places intersection vertices at irregular spacing (often two
    // almost coincident), whose 3-point circumradius is a spurious tiny value even
    // on a straight edge. The fixed-baseline analytic estimate gives the TRUE
    // curvature -- INFINITY on a straight intersection (e.g. plane-cut along a
    // cylinder generator), the real radius on a genuinely curved one.
    let eps = (0.35 * maxh).max(1e-9);
    // Walk the polyline link away from `v` (first step toward `first`) until the
    // accumulated arc length reaches `eps`, returning that on-curve vertex. A
    // controlled baseline of REAL curve points -- robust to the arrangement's
    // irregular vertex spacing AND, unlike a tangent-step + POCS, it does not
    // collapse on a curved-curved intersection (where projecting a stepped point
    // snaps back near `v`).
    let walk = |start: u32, first: u32, curve: &[u32]| -> u32 {
        let (mut prev, mut cur) = (start, first);
        let mut acc = dist(plc.vertices[start as usize], plc.vertices[first as usize]);
        while acc < eps {
            let next = match nbr.get(&cur) {
                Some(ns) if ns.len() == 2 => {
                    if ns[0] == prev {
                        ns[1]
                    } else {
                        ns[0]
                    }
                }
                _ => break, // junction / open end
            };
            // Full lap on a CLOSED loop: one circuit is the longest meaningful
            // baseline; without this, an oversized `eps` walks forever. A
            // corner where the curve ends ends the walk too.
            if next == start || curve_of(cur, next) != curve {
                break;
            }
            acc += dist(plc.vertices[cur as usize], plc.vertices[next as usize]);
            prev = cur;
            cur = next;
        }
        cur
    };
    // At a corner of two curves (a generator line meeting a rim arc) the
    // projection onto the surfaces of both would take all three points to
    // the corner and the radius to zero: a corner is an end of each curve.
    let vert_radius = |v: u32| -> f64 {
        match nbr.get(&v) {
            Some(ns) if ns.len() == 2 => {
                let sids = curve_of(v, ns[0]);
                if curve_of(v, ns[1]) != sids {
                    return f64::INFINITY;
                }
                let a = walk(v, ns[0], &sids);
                let b = walk(v, ns[1], &sids);
                circumradius(
                    pocs(plc.vertices[a as usize], &sids),
                    pocs(plc.vertices[v as usize], &sids),
                    pocs(plc.vertices[b as usize], &sids),
                )
            }
            _ => f64::INFINITY,
        }
    };

    let mut out: Vec<(Tri, f64)> = Vec::new();
    for &(a, b) in &feature {
        let r = vert_radius(a).min(vert_radius(b));
        if r.is_finite() {
            let va = plc.vertices[a as usize];
            let vb = plc.vertices[b as usize];
            // A degenerate tri (va, vb, va) is the segment va-vb for the BVH.
            out.push((Tri::new(va, vb, va), law(&edge_tris[&(a, b)], r)));
        }
    }
    out
}

/// The region containing `p`, by an exact ray cast against the walls.
fn classify(walls: &Walls, bbox: ([f64; 3], [f64; 3]), p: V3) -> u32 {
    // One exact parity cast against every wall the ray's box can meet: a
    // point lies in the region whose walls it crosses an odd number of
    // times (a wall bounds the regions on both of its sides).
    let diag = (0..3)
        .map(|k| bbox.1[k] - bbox.0[k])
        .fold(1.0_f64, f64::max);
    let pp = Prepared3::new(Point3::Explicit(p));
    let mut cand: Vec<u32> = Vec::new();
    let mut odd: Vec<u32> = Vec::new();
    'targets: for k in 0..RAY_TARGETS {
        let q = ray_target(bbox, p, k);
        cand.clear();
        walls
            .index
            .facets_near_segment(p, q, 1e-9 * diag, &mut cand);
        odd.clear();
        for &fi in &cand {
            let tags = walls.tags[fi as usize];
            if tags[0] == tags[1] {
                continue;
            }
            match segment_crosses_triangle(&pp, q, &walls.index.tris()[fi as usize]) {
                None => continue 'targets,
                Some(false) => {}
                Some(true) => {
                    for r in walls.tags[fi as usize] {
                        if r == 0 {
                            continue;
                        }
                        match odd.iter().position(|&x| x == r) {
                            Some(i) => {
                                odd.swap_remove(i);
                            }
                            None => odd.push(r),
                        }
                    }
                }
            }
        }
        return odd.iter().copied().min().unwrap_or(0);
    }
    panic!("no generic ray target found in {RAY_TARGETS} attempts");
}

/// Octree levels built in parallel (8^k subtrees).
const PAR_DEPTH: u32 = 2;

/// The queries the sizing tree is built from.
struct NodeQueries<'a> {
    region_of: &'a (dyn Fn(V3) -> u32 + Sync),
    /// Distance to the boundary, or the given reach if none is closer.
    dist_of: &'a (dyn Fn(V3, f64) -> f64 + Sync),
    /// Size and exact graded parts, from the region, the parent's exact
    /// graded parts and the step from its center.
    h_of: &'a (dyn Fn(V3, u32, [Option<f64>; 2], f64) -> (f64, [Option<f64>; 2]) + Sync),
    boundary_min: &'a (dyn Fn(V3, f64) -> f64 + Sync),
    min_half: f64,
}

/// What a cell hands its children: its center, and its boundary distance
/// and graded size parts where it knows them exactly (each bounds the
/// child's search by the triangle inequality), and its region when no
/// boundary enters it (then every child lies in that region).
#[derive(Clone, Copy)]
struct Parent {
    center: V3,
    dist: Option<f64>,
    graded: [Option<f64>; 2],
    region: Option<u32>,
}

/// `x` a hair larger, so a value equal to it in exact arithmetic lies
/// strictly below it after rounding.
fn widen(x: f64) -> f64 {
    x * (1.0 + 1e-9) + 1e-300
}

fn build_node(center: V3, half: f64, depth: u32, q: &NodeQueries, parent: Option<Parent>) -> Node {
    let step = parent.map_or(0.0, |p| dist(center, p.center));
    let up = |x: Option<f64>| x.map(|v| v + step);
    let region = parent
        .and_then(|p| p.region)
        .unwrap_or_else(|| (q.region_of)(center));
    // The boundary distance out to the cell's circumradius, exact below it:
    // all the uniform test below needs.
    let reach = widen(
        up(parent.and_then(|p| p.dist))
            .unwrap_or(f64::INFINITY)
            .min(half * SQRT3),
    );
    let d = (q.dist_of)(center, reach);
    let (h, graded) = (q.h_of)(center, region, parent.map_or([None; 2], |p| p.graded), step);
    // No boundary facet reaches into the cell if the center is farther from
    // the boundary than the cell circumradius (half * sqrt(3)).
    let uniform = d > half * SQRT3;
    // Subdivide while the cell is bigger than its target size (and not too deep
    // / too small). 2*half is the cell side.
    if 2.0 * half > h && depth < MAX_DEPTH && half > q.min_half {
        let me = Parent {
            center,
            dist: (d < reach).then_some(d),
            graded,
            region: uniform.then_some(region),
        };
        let child = |oct: usize| {
            let (cc, hh) = child_box(center, half, oct);
            build_node(cc, hh, depth + 1, q, Some(me))
        };
        let children: [Node; 8] = if depth < PAR_DEPTH {
            use rayon::prelude::*;
            let v: Vec<Node> = (0..8).into_par_iter().map(child).collect();
            match v.try_into() {
                Ok(a) => a,
                Err(_) => unreachable!("eight children"),
            }
        } else {
            std::array::from_fn(child)
        };
        Node::Inner(Box::new(children))
    } else {
        // A leaf the boundary passes through holds the finest boundary cap in
        // reach, not only its center's value: a point on an interface (a
        // feature curve, a surface point) then gets that interface's size
        // whichever leaf it falls in, instead of the coarser side's graded
        // value when the leaf center lies there.
        let h = if uniform {
            h
        } else {
            h.min((q.boundary_min)(center, half * SQRT3))
        };
        Node::Leaf(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{solid_box, Scene};

    fn cube_plc(s: f64) -> TaggedPlc {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [s, s, s]));
        scene.assemble()
    }

    #[test]
    fn grades_finer_near_size_point() {
        // A plain maxh box is uniform (correct: no feature drives a finer size);
        // grading appears around a refinement source. Put a size point at the
        // center and check h grows with distance from it (Lipschitz in grading).
        let plc = cube_plc(4.0);
        let t = DomainTree::build(
            &plc,
            rapidmesh_brep::Model::new(plc.clone()).index(),
            &MeshParams {
                maxh: 2.0,
                grading: 0.5,
                size_points: vec![([2.0, 2.0, 2.0], 0.1)],
                ..Default::default()
            },
            &[],
            &[],
        );
        let h_at_point = t.h_at([2.0, 2.0, 2.0]);
        let h_away = t.h_at([2.0, 2.0, 3.5]);
        assert!(
            h_at_point < h_away,
            "at point {h_at_point} finer than away {h_away}"
        );
        assert!(
            h_at_point <= 0.3,
            "near the size point ~0.1, got {h_at_point}"
        );
    }

    #[test]
    fn grades_by_region() {
        // A coarse box (maxh) with a finer interior cube (region_maxh) seeds the
        // fine region denser than the coarse one.
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [8.0, 8.0, 8.0]));
        let inner = scene.add_solid(solid_box([3.0, 3.0, 3.0], [5.0, 5.0, 5.0]));
        let plc = scene.assemble();
        let t = DomainTree::build(
            &plc,
            rapidmesh_brep::Model::new(plc.clone()).index(),
            &MeshParams {
                maxh: 4.0,
                region_maxh: vec![(inner.0, 1.0)],
                grading: 0.5,
                ..Default::default()
            },
            &[],
            &[],
        );
        // h is finer inside the small cube than out in the bulk.
        assert!(t.h_at([4.0, 4.0, 4.0]) < t.h_at([0.5, 0.5, 0.5]));
    }
}
