//! Restricted-Delaunay refinement against a [`DomainOracle`].
//!
//! The procedure of CGAL Mesh_3 (Boissonnat-Oudot, Cheng-Dey-Ramos), on the
//! kernel in [`crate::delaunay`]:
//!
//! 1. Protection: corners and samples on every feature curve, each the
//!    center of a protecting ball. The balls are settled geometrically before
//!    anything is inserted: consecutive balls on a curve overlap, other balls
//!    are disjoint (balls sharing a corner excepted), and no ball meets a
//!    patch its feature does not bound. They then enter the triangulation as
//!    weighted points (weight = radius squared) and never change: the
//!    regular triangulation keeps every curve segment as an edge, whatever
//!    the angles between the patches.
//! 2. Facets: a mesh facet is restricted when its dual (the segment between
//!    the orthocenters of its two tets) crosses a surface patch; the
//!    crossing is its surface center. A restricted facet is refined by
//!    inserting that center while it is too large, too flat, too far from
//!    the surface, or has a vertex off its patch. All facet work drains
//!    before any cell is touched. No point enters a protecting ball.
//! 3. Cells: a tet whose orthocenter lies in a region is refined there
//!    while too large or badly shaped. An orthocenter that would remove a
//!    restricted facet from inside its surface ball refines that facet
//!    instead, so the surface is never undersampled by volume points.
//! 4. Extraction: every tet takes the region at its orthocenter. Faces are
//!    the facets where the label changes, plus restricted facets of sheet
//!    patches inside their region.

use super::oracle::{Crossing, DomainOracle, FeatureCurve, Patch, SizeField, P3};
use super::{Complex, Face, VertexKind};
use crate::curve::distribute_floored;
use crate::tri::Triangulation;
use rapidmesh_brep::index::{FacetBvh, Targets};
use rapidmesh_csg::Tri;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Refinement criteria and limits. Sizes are relative to the size field
/// `h(x)` unless stated otherwise.
#[derive(Clone, Debug)]
pub struct Params {
    /// A restricted facet refines while its surface ball radius exceeds
    /// `facet_size * h`. With `cell_size`, calibrated so the median edge
    /// comes out near h (the meaning gmsh gives its size field).
    pub facet_size: f64,
    /// Minimum angle of a restricted facet, in degrees.
    pub facet_angle_deg: f64,
    /// A restricted facet refines while its surface center lies farther than
    /// `facet_distance * h` from the facet's plane (the approximation
    /// error).
    pub facet_distance: f64,
    /// A cell refines while its circumradius exceeds `cell_size * h`.
    pub cell_size: f64,
    /// A cell refines while circumradius over shortest edge exceeds this.
    pub radius_edge: f64,
    /// A cell refines while its smallest dihedral angle (degrees) is below
    /// this, at the best candidate near its orthocenter (0 = off: the
    /// optimizer handles slivers until a native cleanup replaces it).
    pub sliver_deg: f64,
    /// Chord deviation bound for sampling feature curves (relative to the
    /// local curvature radius).
    pub curve_deflection: f64,
    /// How fast the curve sampling may grade from fine to coarse.
    pub curve_grading: f64,
    /// Absolute lower bound on the size used anywhere (the refinement floor
    /// that guarantees termination near small input angles).
    pub min_size: f64,
    /// Stop inserting after this many points.
    pub max_points: usize,
    /// Refine the restricted facets only, no cells, and output the surface
    /// (every restricted facet with its patch) instead of the tets: the
    /// surface mesher of the same core.
    pub surface_only: bool,
    /// Patches meshed with the same triangles as their shifted partner.
    pub periodic: Vec<super::periodic::PeriodicPair>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            facet_size: 0.8,
            facet_angle_deg: 20.0,
            facet_distance: 0.1,
            cell_size: 0.9,
            radius_edge: 2.0,
            sliver_deg: 0.0,
            curve_deflection: 0.01,
            curve_grading: 0.5,
            min_size: 0.0,
            max_points: 2_000_000,
            surface_only: false,
            periodic: Vec::new(),
        }
    }
}

/// What the refinement did, for diagnostics.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub feature_points: usize,
    /// Protection rounds that shrank balls, and balls left violating a
    /// condition at the size floor.
    pub protection_rounds: usize,
    pub protection_unresolved: usize,
    /// Ball contacts left because the features lie closer than the floor.
    pub protection_coincident: usize,
    /// The largest protecting-ball radius.
    pub max_ball: f64,
    /// A few balls the last protection round left at the floor, and why.
    pub protection_examples: Vec<String>,
    pub facet_insertions: usize,
    pub cell_insertions: usize,
    pub rejected: usize,
    /// Boundary slivers removed by refining one of their facets.
    pub sliver_facets: usize,
    /// Cell points too close to the surface inserted anyway, their tet
    /// having no restricted facet to refine instead.
    pub near_surface: usize,
    /// Surface points inserted where a tet's circumcenter left its region.
    pub encroached: usize,
    /// Rejected surface points that fell inside a protecting ball.
    pub rejected_in_ball: usize,
    /// True when `max_points` stopped the refinement early.
    pub capped: bool,
    /// Periodic pairs: points whose image could not go on the partner,
    /// rounds of matching the two sides, facets left unmatched.
    pub periodic_missed: usize,
    pub periodic_rounds: usize,
    pub periodic_left: usize,
}

/// Faces of a tet opposite each corner, wound so the corner lies on the
/// positive side (the kernel's convention).
const FACE_LOCAL: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]];

/// A candidate closer than this fraction of the local size to an existing
/// vertex is dropped (the duplicate guard).
const DUP_FRAC: f64 = 0.05;
/// A cell point within this many surface duplicate guards of a patch is
/// not inserted (the guards of neighbouring points differ with the size).
const NEAR_SURFACE: f64 = 1.5;
/// Sliver candidates: the orthocenter and these directions, at
/// `SLIVER_SPREAD` times the orthoradius (icosahedron vertices).
const SLIVER_SPREAD: f64 = 0.3;
const SLIVER_CANDIDATES: [P3; 12] = {
    const A: f64 = 0.525_731_112_119_133_6;
    const B: f64 = 0.850_650_808_352_039_9;
    [
        [-A, B, 0.0],
        [A, B, 0.0],
        [-A, -B, 0.0],
        [A, -B, 0.0],
        [0.0, -A, B],
        [0.0, A, B],
        [0.0, -A, -B],
        [0.0, A, -B],
        [B, 0.0, -A],
        [B, 0.0, A],
        [-B, 0.0, -A],
        [-B, 0.0, A],
    ]
};
/// Probes for sliver candidates give up above this many cavity tets.
const SLIVER_CAVITY: usize = 512;

/// Protecting-ball radius as a fraction of the adjacent segment lengths.
const BALL_FRAC: f64 = 0.8;
/// Consecutive protecting balls must overlap: their radii sum to at least
/// this multiple of their distance.
const OVERLAP: f64 = 1.1;
/// Protection rounds before the remaining violations are accepted.
const PROTECT_ROUNDS: usize = 32;
/// Protecting balls shrink down to the local size over this. Corner balls
/// cover acute corners, so shrinking is not scale invariant; the floor
/// stops it at degenerate input (near-coincident features, as faceted CSG
/// leaves between tangent surfaces), which it leaves unresolved.
const PROTECT_DEPTH: f64 = 64.0;
/// A protecting ball of radius r marks features of scale `BALL_SIZE * r`
/// around it (the sample spacing it came from), graded by `BALL_GRADING`.
const BALL_SIZE: f64 = 1.25;
const BALL_GRADING: f64 = 0.5;
/// Balls filling a gap on a curve are spaced at this multiple of the larger
/// end ball's radius.
const FILL_STEP: f64 = 1.25;

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dist(a: P3, b: P3) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}
/// Distance from `p` to the segment `a b`.
fn seg_dist(p: P3, a: P3, b: P3) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 0.0 {
        (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    dist(p, std::array::from_fn(|k| a[k] + t * ab[k]))
}
fn mid(a: P3, b: P3) -> P3 {
    std::array::from_fn(|k| 0.5 * (a[k] + b[k]))
}
/// Orthocenter and power radius of a weighted tet (the center of the
/// sphere orthogonal to the four balls `|x - p_i|^2 = w_i`), `None` when
/// degenerate. With zero weights this is the circumcenter.
pub(crate) fn tet_orthocenter(p: [P3; 4], w: [f64; 4]) -> Option<(P3, f64)> {
    let (u, v, x) = (sub(p[1], p[0]), sub(p[2], p[0]), sub(p[3], p[0]));
    let det = dot(u, cross(v, x));
    let scale = [u, v, x]
        .iter()
        .map(|r| r.iter().map(|c| c.abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    if !(det.abs() > 1e-12 * scale * scale * scale) {
        return None;
    }
    let (uu, vv, xx) = (
        dot(u, u) - (w[1] - w[0]),
        dot(v, v) - (w[2] - w[0]),
        dot(x, x) - (w[3] - w[0]),
    );
    let (vx, xu, uv) = (cross(v, x), cross(x, u), cross(u, v));
    let f = 0.5 / det;
    let off: P3 = std::array::from_fn(|k| f * (uu * vx[k] + vv * xu[k] + xx * uv[k]));
    let c = [p[0][0] + off[0], p[0][1] + off[1], p[0][2] + off[2]];
    Some((c, (dot(off, off) - w[0]).max(0.0).sqrt()))
}

/// Circumcenter of a triangle, `None` when degenerate.
fn tri_circumcenter(a: P3, b: P3, c: P3) -> Option<P3> {
    let (u, v) = (sub(b, a), sub(c, a));
    let n = cross(u, v);
    let nn = dot(n, n);
    if !(nn > 0.0) {
        return None;
    }
    let t = cross(n, u);
    let s = cross(v, n);
    let (uu, vv) = (dot(u, u), dot(v, v));
    let off: P3 = std::array::from_fn(|k| (vv * t[k] + uu * s[k]) / (2.0 * nn));
    Some([a[0] + off[0], a[1] + off[1], a[2] + off[2]])
}

pub(crate) use crate::diagnostics::tet_min_dihedral;

/// Distance from `x` to the plane of triangle `a b c` (infinite when the
/// triangle is degenerate). For a surface center `x` on the facet's dual
/// line, which is perpendicular to the facet, this is its distance to the
/// facet's weighted circumcenter, Mesh_3's approximation error. The plain
/// circumcenter would be off the dual line wherever the facet has weighted
/// (protecting ball) vertices, and flag flat facets next to every feature.
fn plane_distance(a: P3, b: P3, c: P3, x: P3) -> f64 {
    let n = cross(sub(b, a), sub(c, a));
    let l = dot(n, n).sqrt();
    if !(l > 0.0) {
        return f64::INFINITY;
    }
    dot(n, sub(x, a)).abs() / l
}

/// Smallest angle of a triangle, in degrees.
fn tri_min_angle(a: P3, b: P3, c: P3) -> f64 {
    let ang = |u: P3, v: P3, w: P3| {
        let (e1, e2) = (sub(v, u), sub(w, u));
        let d = dot(e1, e2) / ((dot(e1, e1) * dot(e2, e2)).sqrt() + 1e-300);
        d.clamp(-1.0, 1.0).acos().to_degrees()
    };
    ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b))
}

/// A protected segment of feature curve `curve` between curve samples `va`
/// and `vb`.
#[derive(Clone, Copy, Debug)]
struct Seg {
    curve: u32,
    va: u32,
    vb: u32,
}

/// A restricted facet: its surface center, surface-ball radius and patch.
#[derive(Clone, Copy, Debug)]
struct Restricted {
    center: P3,
    radius: f64,
    patch: u32,
}

/// A uniform spatial hash for balls and segments.
struct Grid {
    cell: f64,
    map: FxHashMap<[i64; 3], Vec<u32>>,
}

impl Grid {
    fn new(cell: f64) -> Grid {
        Grid {
            cell: cell.max(1e-300),
            map: FxHashMap::default(),
        }
    }
    fn key(&self, p: P3) -> [i64; 3] {
        std::array::from_fn(|k| (p[k] / self.cell).floor() as i64)
    }
    /// Registers `id` in every cell the box `c +/- r` touches.
    fn insert(&mut self, c: P3, r: f64, id: u32) {
        let lo = self.key([c[0] - r, c[1] - r, c[2] - r]);
        let hi = self.key([c[0] + r, c[1] + r, c[2] + r]);
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let v = self.map.entry([x, y, z]).or_default();
                    if !v.contains(&id) {
                        v.push(id);
                    }
                }
            }
        }
    }
    fn at(&self, p: P3) -> &[u32] {
        self.map.get(&self.key(p)).map_or(&[], |v| v.as_slice())
    }
    /// Ids registered in any cell the box `c +/- r` touches (with repeats).
    fn near(&self, c: P3, r: f64, out: &mut Vec<u32>) {
        out.clear();
        let lo = self.key([c[0] - r, c[1] - r, c[2] - r]);
        let hi = self.key([c[0] + r, c[1] + r, c[2] + r]);
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    if let Some(v) = self.map.get(&[x, y, z]) {
                        out.extend_from_slice(v);
                    }
                }
            }
        }
    }
}

/// The protecting balls: corners first, then the curve samples, with
/// their radii and kinds, and per curve its chain of (sample, arc length).
struct Protection {
    pos: Vec<P3>,
    radius: Vec<f64>,
    kind: Vec<VertexKind>,
    chains: Vec<Vec<(u32, f64)>>,
    /// The largest radius each ball may keep.
    cap: Vec<f64>,
}

struct Mesher<'a, D: DomainOracle + ?Sized, S: SizeField + ?Sized> {
    dom: &'a D,
    size: &'a S,
    prm: &'a Params,
    db: Triangulation,
    kinds: Vec<VertexKind>,
    /// Patches incident to each corner (from the curves ending there).
    corner_patches: Vec<Vec<u32>>,
    /// Protecting-ball radius per vertex (0 = none), fixed after
    /// protection.
    ball: Vec<f64>,
    balls: Grid,
    /// The balls as graded scale sources (`BALL_SIZE` radii at the center,
    /// growing by `BALL_GRADING` with distance), for the duplicate guard.
    ball_size: (FacetBvh, Targets),
    /// The protected curve segments.
    segs: Vec<Seg>,
    /// Restricted status per slot face, the same on both slots of a facet:
    /// `UNSEEN` since its tets last changed, `PLAIN` when not restricted,
    /// else an index into `rpool`.
    rstate: Vec<[u32; 4]>,
    rpool: Vec<Restricted>,
    /// Label point per slot and the oracle's clearance there, computed on
    /// first use.
    lpoint: Vec<Option<(P3, f64)>>,
    facet_queue: VecDeque<(u32, [usize; 4])>,
    cell_queue: VecDeque<(u32, [usize; 4])>,
    far: f64,
    stats: Stats,
    probe_faces: Vec<[usize; 3]>,
    /// Protecting balls that keep a surface facet from refining (surface
    /// mode): the ball's vertex and the radius that frees the facet.
    blocked: Vec<(u32, f64)>,
    /// The periodic classes of patches, curves and corners.
    per: super::periodic::Periodic,
}

/// A facet's vertices, sorted: its key where it is met from both sides.
fn key3(f: [usize; 3]) -> [u32; 3] {
    let mut k = f.map(|v| v as u32);
    k.sort_unstable();
    k
}

/// A facet not examined since its tets last changed.
const UNSEEN: u32 = u32::MAX;
/// A facet examined and not restricted.
const PLAIN: u32 = u32::MAX - 1;

impl<'a, D: DomainOracle + ?Sized, S: SizeField + ?Sized> Mesher<'a, D, S> {
    /// The local target size: the size field above the global floor.
    fn h(&self, p: P3) -> f64 {
        self.size.size(p).max(self.prm.min_size).max(1e-300)
    }

    /// The scale of the duplicate guard: the target size, bounded near
    /// protecting balls by the feature size they encode, so the topology
    /// criterion can resolve a thin trace or an acute corner without the
    /// size criteria refining around it.
    fn guard_scale(&self, p: P3) -> f64 {
        let (bvh, targets) = &self.ball_size;
        self.h(p).min(bvh.graded_min(targets, p, BALL_GRADING))
    }

    fn pos(&self, v: usize) -> P3 {
        self.db.point(v)
    }

    /// Orthocenter and power radius of a tet (its circumcenter and
    /// circumradius away from protecting balls).
    fn cc(&self, t: [usize; 4]) -> Option<(P3, f64)> {
        tet_orthocenter(t.map(|v| self.pos(v)), t.map(|v| self.db.weight(v)))
    }

    /// The point a tet is labelled at: its circumcenter, or its centroid when
    /// it is degenerate (four coplanar points on a flat patch). Duals run
    /// between label points, so a label change across a facet always implies
    /// a crossing of its dual by the oracle's consistency contract.
    fn label_point(&self, t: [usize; 4]) -> P3 {
        match self.cc(t) {
            Some((c, _)) => c,
            None => {
                let p = t.map(|v| self.pos(v));
                std::array::from_fn(|k| 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k]))
            }
        }
    }

    /// The region a tet belongs to: the region at its label point. A
    /// degenerate tet (four points on one plane, as cocircular samples on a
    /// flat face leave) with all vertices on one boundary patch holds no
    /// volume of the domain and belongs outside; its centroid lies on the
    /// patch, where the region is a tie.
    fn label(&self, t: [usize; 4]) -> u32 {
        if self.cc(t).is_none() && self.on_boundary_patch(t) {
            return 0;
        }
        self.dom.region(self.label_point(t))
    }

    /// True if all four vertices lie on one patch with the outside on a side.
    fn on_boundary_patch(&self, t: [usize; 4]) -> bool {
        let cands: Vec<u32> = match self.kinds[t[0]] {
            VertexKind::Patch(p) => vec![p],
            VertexKind::Curve(c) => self.dom.curves()[c as usize].patches.clone(),
            VertexKind::Corner(c) => self.corner_patches[c as usize].clone(),
            VertexKind::Volume => Vec::new(),
        };
        cands.into_iter().any(|p| {
            self.dom.patches()[p as usize].regions.contains(&0)
                && t.iter().all(|&v| self.on_patch(v, p))
        })
    }

    /// True if vertex `v` lies on patch `patch` (directly, or on a feature
    /// curve or corner of it).
    fn on_patch(&self, v: usize, patch: u32) -> bool {
        match self.kinds[v] {
            VertexKind::Patch(p) => p == patch,
            VertexKind::Curve(c) => self.dom.curves()[c as usize].patches.contains(&patch),
            VertexKind::Corner(c) => self.corner_patches[c as usize].contains(&patch),
            VertexKind::Volume => false,
        }
    }

    // ------------------------------------------------------------ protection

    fn ball_owner(&self, p: P3) -> Option<u32> {
        self.balls.at(p).iter().copied().find(|&v| {
            let r = self.ball[v as usize];
            r > 0.0 && dist(p, self.pos(v as usize)) < r
        })
    }

    fn push_vertex(&mut self, kind: VertexKind) {
        self.kinds.push(kind);
        self.ball.push(0.0);
    }

    /// Settles the protecting balls on the corners and curves, then inserts
    /// them as weighted points: consecutive balls on a curve overlap, other
    /// balls are disjoint, and no ball meets a patch its feature does not
    /// bound, down to a floor relative to the local size.
    fn new(dom: &'a D, size: &'a S, prm: &'a Params) -> Self {
        let (lo, hi) = dom.bbox();
        let diag = dist(lo, hi).max(1e-12);
        let cell = {
            // Grid cell for balls and segments: the typical feature spacing.
            let mut h = f64::INFINITY;
            for i in 0..8 {
                let p: P3 = std::array::from_fn(|k| if i & (1 << k) != 0 { hi[k] } else { lo[k] });
                h = h.min(size.size(p));
            }
            h.max(prm.min_size).max(1e-6 * diag).min(0.1 * diag)
        };
        let pad: P3 = [0.05 * diag; 3];
        Mesher {
            dom,
            size,
            prm,
            db: Triangulation::enclosing(
                sub(lo, pad),
                [hi[0] + pad[0], hi[1] + pad[1], hi[2] + pad[2]],
            ),
            kinds: Vec::new(),
            corner_patches: vec![Vec::new(); dom.corners().len()],
            ball: Vec::new(),
            balls: Grid::new(cell),
            segs: Vec::new(),
            ball_size: {
                let bvh = FacetBvh::build(&[]);
                let targets = Targets::new(&bvh, Vec::new());
                (bvh, targets)
            },
            rstate: Vec::new(),
            rpool: Vec::new(),
            lpoint: Vec::new(),
            facet_queue: VecDeque::new(),
            cell_queue: VecDeque::new(),
            far: 4.0 * diag,
            stats: Stats::default(),
            probe_faces: Vec::new(),
            blocked: Vec::new(),
            per: Default::default(),
        }
    }

    fn log_protection(&self) {
        rapidmesh_exact::log::info(
            "mesh3.protect",
            format!(
                "{} balls, {} rounds, {} unresolved, {} coincident, largest {:.3e}",
                self.stats.feature_points,
                self.stats.protection_rounds,
                self.stats.protection_unresolved,
                self.stats.protection_coincident,
                self.stats.max_ball
            ),
        );
        for e in &self.stats.protection_examples {
            rapidmesh_exact::log::info("mesh3.protect", e.clone());
        }
    }

    /// The corners of the domain box padded by a quarter of its diagonal,
    /// as volume points: an open or flat surface alone spans no real tets,
    /// only tets with the enclosing vertices, whose facets the refinement
    /// does not see. No surface facet keeps them (a facet with a vertex off
    /// its patch refines).
    fn insert_box_corners(&mut self) {
        let (lo, hi) = self.dom.bbox();
        let pad = 0.25 * dist(lo, hi);
        for i in 0..8 {
            let c: P3 = std::array::from_fn(|k| {
                if i & (1 << k) != 0 {
                    hi[k] + pad
                } else {
                    lo[k] - pad
                }
            });
            self.insert_point(c, VertexKind::Volume);
        }
    }

    /// The patches through each corner: its own, and those of the curves
    /// ending there.
    fn init_corner_patches(&mut self) {
        let dom = self.dom;
        for (k, cp) in self.corner_patches.iter_mut().enumerate() {
            *cp = dom.corner_patches(k as u32);
        }
        for fc in dom.curves() {
            for &k in fc.ends.iter().flatten() {
                for &p in &fc.patches {
                    if !self.corner_patches[k as usize].contains(&p) {
                        self.corner_patches[k as usize].push(p);
                    }
                }
            }
        }
    }

    /// The protecting balls of the corners and curves: the samples, their
    /// radii and the chain of samples along every curve.
    fn protect(&mut self) -> Protection {
        let dom = self.dom;
        let nc = dom.corners().len();
        self.init_corner_patches();
        self.per = super::periodic::Periodic::build(dom, &self.corner_patches, &self.prm.periodic);

        // Nodes: the corners, then the samples of every curve; each curve
        // is a chain of (node, arc parameter), closed chains repeating
        // their first node at the end.
        let mut pos: Vec<P3> = dom.corners().to_vec();
        let mut kind: Vec<VertexKind> = (0..nc as u32).map(VertexKind::Corner).collect();
        let mut chains: Vec<Vec<(u32, f64)>> = Vec::with_capacity(dom.curves().len());
        for (ci, fc) in dom.curves().iter().enumerate() {
            let len = fc.curve.length();
            if !(len > 0.0) {
                chains.push(Vec::new());
                continue;
            }
            // The finest size along the curve, bounded by its own size.
            let finest = (0..=32)
                .map(|i| self.h(fc.curve.point_at(len * i as f64 / 32.0)))
                .fold(f64::INFINITY, f64::min)
                .min(fc.max_size);
            // A surface samples the curve after the size field along it; the
            // volume keeps the finest size all along (graded curves in the
            // volume trade straddlers for feature fidelity, see #94).
            let surface = self.prm.surface_only;
            let size = |s: f64| {
                if surface {
                    self.h(fc.curve.point_at(s)).min(fc.max_size)
                } else {
                    finest
                }
            };
            // An explicit per-curve bound finer than the global floor wins.
            let floor = self.prm.min_size.min(fc.max_size).max(1e-12 * len);
            let defl = fc.deflection.unwrap_or(self.prm.curve_deflection);
            // A curve in a periodic class after its first takes that one's
            // samples, shifted (and in its own direction).
            let image = self.per.curve[ci].filter(|im| (im.root as usize) < ci);
            let closed_curve = fc.ends[0].is_none() || fc.ends[0] == fc.ends[1];
            let inner: Vec<(f64, P3)> = if let Some(im) = image {
                let (root, shift) = (im.root, im.shift);
                let mut seen: FxHashSet<u32> = FxHashSet::default();
                let mut m: Vec<(f64, P3)> = chains[root as usize]
                    .iter()
                    .filter(|&&(n, _)| {
                        kind[n as usize] == VertexKind::Curve(root) && seen.insert(n)
                    })
                    .map(|&(n, s)| {
                        let arc = im.phase + im.dir * s;
                        let arc = if closed_curve {
                            arc.rem_euclid(len)
                        } else {
                            arc
                        };
                        (
                            arc,
                            [
                                pos[n as usize][0] + shift[0],
                                pos[n as usize][1] + shift[1],
                                pos[n as usize][2] + shift[2],
                            ],
                        )
                    })
                    .collect();
                // Closed curves anchored at corners that are no images of
                // each other: the image of the root's anchor is a sample
                // here (the root has one where this anchor's preimage is).
                if let (true, Some(kr), Some(km)) = (
                    closed_curve,
                    dom.curves()[root as usize].ends[0],
                    fc.ends[0],
                ) {
                    if self.per.corner[kr as usize] != self.per.corner[km as usize] {
                        let (a, own) = (pos[kr as usize], pos[km as usize]);
                        m.retain(|&(_, p)| dist(p, own) > 1e3 * self.per.tol);
                        m.push((
                            im.phase.rem_euclid(len),
                            [a[0] + shift[0], a[1] + shift[1], a[2] + shift[2]],
                        ));
                    }
                }
                m.sort_by(|a, b| a.0.total_cmp(&b.0));
                m
            } else {
                let ss = distribute_floored(&*fc.curve, defl, &size, self.prm.curve_grading, floor);
                // A curve without corners, or one starting and ending at the
                // same corner, closes on itself.
                let closed = fc.ends[0].is_none() || fc.ends[0] == fc.ends[1];
                let first = usize::from(fc.ends[0].is_some());
                let last = ss.len().saturating_sub(1);
                let mut arcs: Vec<f64> = ss[first.min(last)..last].to_vec();
                // A closed curve needs at least three samples to be a polygon.
                if closed && arcs.len() + first < 3 {
                    arcs = (first..3).map(|i| len * i as f64 / 3.0).collect();
                }
                let mut samples: Vec<(f64, P3)> = arcs
                    .into_iter()
                    .map(|s| (s, fc.curve.point_at(s)))
                    .collect();
                // A closed root of a periodic class: a sample exactly at the
                // preimage of every image's anchor that is no image of this
                // curve's anchor.
                if let (true, Some(kr)) = (closed, fc.ends[0]) {
                    for (m, im) in self.per.curve.iter().enumerate() {
                        let Some(im) = im else { continue };
                        if im.root as usize != ci || m == ci {
                            continue;
                        }
                        let Some(km) = dom.curves()[m].ends[0] else {
                            continue;
                        };
                        if self.per.corner[kr as usize] != self.per.corner[km as usize] {
                            let (a, t) = (pos[km as usize], im.shift);
                            samples.push((
                                (-im.phase * im.dir).rem_euclid(len),
                                [a[0] - t[0], a[1] - t[1], a[2] - t[2]],
                            ));
                        }
                    }
                    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
                    samples.dedup_by(|a, b| (a.0 - b.0).abs() <= 1e-9 * len);
                }
                samples
            };
            let mut chain: Vec<(u32, f64)> = Vec::with_capacity(inner.len() + 2);
            if let Some(k) = fc.ends[0] {
                chain.push((k, 0.0));
            }
            for (s, p) in inner {
                chain.push((pos.len() as u32, s));
                pos.push(p);
                kind.push(VertexKind::Curve(ci as u32));
            }
            match fc.ends[1] {
                Some(k) => chain.push((k, len)),
                None => {
                    if let Some(&(v0, _)) = chain.first() {
                        chain.push((v0, len));
                    }
                }
            }
            chains.push(chain);
        }

        // Radii are their own quantity (CGAL `Protect_edges_sizing_field`):
        // they start at `BALL_FRAC` of the shortest adjacent segment and
        // only shrink. A corner ball shrinks only when it meets a ball it is
        // not consecutive with, so the curve balls ring it and acute corners
        // settle instead of refining forever.
        let mut radius: Vec<f64> = vec![f64::INFINITY; pos.len()];
        for chain in &chains {
            for w in chain.windows(2) {
                let (a, b) = (w[0].0 as usize, w[1].0 as usize);
                if a != b {
                    let l = BALL_FRAC * dist(pos[a], pos[b]);
                    radius[a] = radius[a].min(l);
                    radius[b] = radius[b].min(l);
                }
            }
        }
        for (i, r) in radius.iter_mut().enumerate() {
            if !r.is_finite() {
                // A corner on no curve (an apex): a ball of its own size.
                *r = 0.5 * BALL_FRAC * self.h(pos[i]);
            }
        }
        let mut pr = Protection {
            cap: vec![f64::INFINITY; pos.len()],
            pos,
            radius,
            kind,
            chains,
        };
        self.settle(&mut pr);
        pr
    }

    /// Protection rounds until the balls satisfy the conditions: shrink
    /// balls that meet, fill the gaps between the balls of a curve. Balls
    /// stay below their cap (the surface refinement caps a ball holding a
    /// center it needs).
    fn settle(&mut self, pr: &mut Protection) {
        let dom = self.dom;
        let (lo, hi) = dom.bbox();
        let diag = dist(lo, hi).max(1e-300);
        let Protection {
            pos,
            radius,
            kind,
            chains,
            cap,
        } = pr;
        // The patches a node's ball may meet.
        let allowed_patches = |k: VertexKind| -> Vec<u32> {
            match k {
                VertexKind::Curve(c) => dom.curves()[c as usize].patches.clone(),
                VertexKind::Corner(c) => self.corner_patches[c as usize].clone(),
                _ => Vec::new(),
            }
        };
        let floor_at = |p: P3| (self.h(p) / PROTECT_DEPTH).max(1e-9 * diag);
        let mut grid_ids: Vec<u32> = Vec::new();
        let mut rounds = 0;
        let mut unresolved_last;
        let mut coincident_last;
        let mut examples: Vec<String> = Vec::new();
        let mut corner_log: Vec<String> = Vec::new();
        // Periodic images keep the same place and radius.
        let sync = |links: &[(u32, u32, Option<P3>)], pos: &mut [P3], r: &mut [f64]| {
            for &(a, b, shift) in links {
                if let Some(t) = shift {
                    let p = pos[a as usize];
                    pos[b as usize] = [p[0] + t[0], p[1] + t[1], p[2] + t[2]];
                }
                let m = r[a as usize].min(r[b as usize]);
                r[a as usize] = m;
            }
            for &(a, b, _) in links {
                r[b as usize] = r[a as usize];
            }
        };
        loop {
            let n = pos.len();
            let links = self.per.links(chains, kind, pos);
            sync(&links, pos, radius);
            let mut next: Vec<(u32, u32)> = Vec::new(); // consecutive pairs
            for chain in chains.iter() {
                for w in chain.windows(2) {
                    if w[0].0 != w[1].0 {
                        next.push((w[0].0.min(w[1].0), w[0].0.max(w[1].0)));
                    }
                }
            }
            next.sort_unstable();
            let consecutive = |i: u32, j: u32| next.binary_search(&(i.min(j), i.max(j))).is_ok();
            let mut chain_nbrs: Vec<Vec<u32>> = vec![Vec::new(); n];
            for &(a, b) in &next {
                chain_nbrs[a as usize].push(b);
                chain_nbrs[b as usize].push(a);
            }
            // Distance from ball `i`'s center to the curve through ball `j`
            // (the segments at `j`), which ignores the offset along it.
            let to_curve = |i: usize, j: usize| -> f64 {
                chain_nbrs[j]
                    .iter()
                    .map(|&k| seg_dist(pos[i], pos[j], pos[k as usize]))
                    .fold(dist(pos[i], pos[j]), f64::min)
            };
            // Shrink: two balls joined by an edge of their regular
            // triangulation that is not a curve segment must be disjoint;
            // both drop to their distance / 2.1. A ball the triangulation
            // declines (hidden, or hiding another) shrinks with every ball it
            // overlaps. A ball meeting a foreign patch halves.
            let mut shrunk = false;
            let mut unresolved = 0;
            examples.clear();
            let mut new_r = radius.clone();
            // Why each ball shrank this round: its partner ball, or a patch.
            let mut why: Vec<Option<(u32, bool)>> = vec![None; n];
            // Features closer than the floor can resolve count as one (a
            // sliver strip faceted CSG leaves between tangent surfaces):
            // their balls keep their size instead of collapsing to the floor.
            let mut coincident = 0usize;
            // Surface meshing triangulates every patch alone: balls meet
            // only in the triangulation of a patch they share.
            let surface = self.prm.surface_only;
            let share: Vec<Vec<u32>> = if surface {
                kind.iter().map(|&k| allowed_patches(k)).collect()
            } else {
                Vec::new()
            };
            let apart =
                |i: usize, j: usize| surface && !share[i].iter().any(|p| share[j].contains(p));
            let mut conflict =
                |i: usize, j: usize, new_r: &mut Vec<f64>, why: &mut Vec<Option<(u32, bool)>>| {
                    let d = dist(pos[i], pos[j]);
                    if i != j
                        && d < radius[i] + radius[j]
                        && !consecutive(i as u32, j as u32)
                        && !apart(i, j)
                    {
                        let f = floor_at(pos[i]).min(floor_at(pos[j]));
                        if d / 2.1 < f || to_curve(i, j).min(to_curve(j, i)) < 2.1 * f {
                            coincident += 1;
                            return;
                        }
                        for (k, o) in [(i, j), (j, i)] {
                            // A surface triangle between a ball and the
                            // curve of the other spans their distance: the
                            // ball stays below it (the width of a trace).
                            let rk = if surface {
                                to_curve(k, o) / 2.1
                            } else {
                                d / 2.1
                            };
                            if rk < new_r[k] {
                                new_r[k] = rk;
                                why[k] = Some((o as u32, false));
                            }
                        }
                    }
                };
            // Every ball the regular triangulation of all balls declines
            // (hidden, or hiding another) meets each ball it overlaps; in
            // the surface mode every ball does, since balls of one patch
            // can overlap without an edge of that triangulation between
            // them.
            let mut overlapping: Vec<usize> = Vec::new();
            if surface {
                overlapping.extend(0..n);
            } else {
                let mut rt = Triangulation::enclosing(
                    std::array::from_fn(|k| lo[k] - 0.05 * diag),
                    std::array::from_fn(|k| hi[k] + 0.05 * diag),
                );
                let mut node_of: Vec<u32> = Vec::with_capacity(n);
                for i in 0..n {
                    match rt.try_insert_weighted(pos[i], radius[i] * radius[i]) {
                        Some(_) => node_of.push(i as u32),
                        None => overlapping.push(i),
                    }
                }
                for t in rt.tets() {
                    for a in 0..4 {
                        for b in a + 1..4 {
                            let (i, j) = (node_of[t[a]] as usize, node_of[t[b]] as usize);
                            if i < j {
                                conflict(i, j, &mut new_r, &mut why);
                            }
                        }
                    }
                }
            }
            if !overlapping.is_empty() {
                let mut grid = Grid::new(self.balls.cell);
                for i in 0..n {
                    grid.insert(pos[i], radius[i], i as u32);
                }
                for &i in &overlapping {
                    grid.near(pos[i], radius[i], &mut grid_ids);
                    for &j in &grid_ids {
                        conflict(i, j as usize, &mut new_r, &mut why);
                    }
                }
            }
            // A ball meeting a patch other than its own halves (the surface
            // mode meets no other patch).
            for i in (0..n).filter(|_| !surface) {
                let allowed = allowed_patches(kind[i]);
                let mut r = new_r[i];
                let mut clear = false;
                for _ in 0..8 {
                    if !dom.patch_within(pos[i], r, &allowed) {
                        clear = true;
                        break;
                    }
                    r *= 0.5;
                }
                if r < new_r[i] {
                    if clear && r >= floor_at(pos[i]) {
                        new_r[i] = r;
                        why[i] = Some((u32::MAX, true));
                    } else {
                        // A foreign patch closer than the floor resolves.
                        coincident += 1;
                    }
                }
            }
            for i in 0..n {
                new_r[i] = new_r[i].min(cap[i]);
            }
            {
                let mut unused = pos.to_vec();
                let plain: Vec<(u32, u32, Option<P3>)> =
                    links.iter().map(|&(a, b, _)| (a, b, None)).collect();
                sync(&plain, &mut unused, &mut new_r);
            }
            for i in 0..n {
                let f = floor_at(pos[i]);
                if new_r[i] < radius[i] {
                    if matches!(kind[i], VertexKind::Corner(_)) && corner_log.len() < 8 {
                        corner_log.push(format!(
                            "round {rounds}: {:?} r {:.3e} -> {:.3e} by {}",
                            kind[i],
                            radius[i],
                            new_r[i],
                            match why[i] {
                                Some((_, true)) => "a foreign patch".to_string(),
                                Some((j, false)) => format!(
                                    "{:?} at {:?} r {:.3e}",
                                    kind[j as usize], pos[j as usize], radius[j as usize]
                                ),
                                None => "?".to_string(),
                            }
                        ));
                    }
                    let r = new_r[i].max(f);
                    if r < radius[i] {
                        radius[i] = r;
                        shrunk = true;
                    }
                    if new_r[i] < f {
                        unresolved += 1;
                        if examples.len() < 8 {
                            examples.push(match why[i] {
                                Some((_, true)) => format!(
                                    "{:?} at {:?} r {:.3e}: meets a foreign patch",
                                    kind[i], pos[i], new_r[i]
                                ),
                                Some((j, false)) => format!(
                                    "{:?} at {:?} r {:.3e}: meets {:?} at {:?} r {:.3e}",
                                    kind[i],
                                    pos[i],
                                    new_r[i],
                                    kind[j as usize],
                                    pos[j as usize],
                                    radius[j as usize]
                                ),
                                None => format!("{:?} at {:?}", kind[i], pos[i]),
                            });
                        }
                    }
                }
            }
            {
                let plain: Vec<(u32, u32, Option<P3>)> =
                    links.iter().map(|&(a, b, _)| (a, b, None)).collect();
                let mut unused = pos.to_vec();
                sync(&plain, &mut unused, radius);
            }
            // Fill: a segment whose end balls do not overlap gets evenly
            // spaced balls, sized after the larger end ball; later rounds
            // grade them down to a small end ball.
            let last_round = rounds >= PROTECT_ROUNDS;
            let mut filled = false;
            for (ci, chain) in chains.iter_mut().enumerate() {
                let fc = &dom.curves()[ci];
                let mut out: Vec<(u32, f64)> = Vec::with_capacity(chain.len());
                for w in chain.windows(2) {
                    out.push(w[0]);
                    let (a, b) = (w[0].0 as usize, w[1].0 as usize);
                    let l = dist(pos[a], pos[b]);
                    if radius[a] + radius[b] >= OVERLAP * l {
                        continue;
                    }
                    let f = floor_at(mid(pos[a], pos[b]));
                    if last_round || l < 2.0 * f {
                        unresolved += 1;
                        if examples.len() < 8 {
                            examples.push(format!(
                                "gap between {:?} at {:?} r {:.3e} and {:?} r {:.3e}, length {l:.3e}",
                                kind[a], pos[a], radius[a], kind[b], radius[b]
                            ));
                        }
                        continue;
                    }
                    // New balls go into the stretch neither end ball covers,
                    // never inside an end ball (a corner ball would shrink
                    // on them).
                    let (ra, rb) = (radius[a], radius[b]);
                    let (x0, x1) = (ra, l - rb);
                    if x0 >= x1 {
                        // The balls touch: the curve is covered, and any new
                        // center here would lie inside one of them.
                        continue;
                    }
                    let gap = x1 - x0;
                    let sp = (FILL_STEP * ra.max(rb)).max(f);
                    let k = ((gap / sp).ceil() as usize).max(1);
                    let s = gap / k as f64;
                    for m in 0..k {
                        let x = x0 + (m as f64 + 0.5) * s;
                        // As large as its neighbours' centers allow (so it
                        // overlaps them), at most the larger end ball.
                        let near = if k > 1 {
                            x.min(l - x).min(s)
                        } else {
                            x.min(l - x)
                        };
                        let r = (0.9 * near).min(ra.max(rb));
                        let arc = w[0].1 + (w[1].1 - w[0].1) * x / l;
                        out.push((pos.len() as u32, arc));
                        pos.push(fc.curve.point_at(arc));
                        kind.push(VertexKind::Curve(ci as u32));
                        radius.push(r);
                        cap.push(f64::INFINITY);
                    }
                    filled = true;
                }
                if let Some(&l) = chain.last() {
                    out.push(l);
                }
                *chain = out;
            }
            unresolved_last = unresolved;
            coincident_last = coincident;
            if !(shrunk || filled) || last_round {
                break;
            }
            rounds += 1;
        }
        let links = self.per.links(chains, kind, pos);
        sync(&links, pos, radius);
        self.stats.protection_rounds = rounds;
        self.stats.protection_unresolved = unresolved_last;
        self.stats.protection_coincident = coincident_last;
        examples.extend(corner_log);
        self.stats.protection_examples = examples;
        self.stats.max_ball = radius.iter().copied().fold(0.0, f64::max);
        self.stats.feature_points = pos.len();
    }

    /// Inserts the protected samples `keep` accepts, corners first, each
    /// weighted by its squared radius, with the curve segments between
    /// them. Returns the vertex of every sample (`None` when skipped or
    /// hidden).
    fn insert_protection(
        &mut self,
        pr: &Protection,
        keep: &dyn Fn(VertexKind) -> bool,
    ) -> Vec<Option<u32>> {
        let Protection {
            pos,
            radius,
            kind,
            chains,
            ..
        } = pr;
        let mut vid: Vec<Option<u32>> = vec![None; pos.len()];
        for i in 0..pos.len() {
            if !keep(kind[i]) {
                continue;
            }
            if let Some(v) = self.db.try_insert_weighted(pos[i], radius[i] * radius[i]) {
                self.push_vertex(kind[i]);
                self.ball[v] = radius[i];
                self.balls.insert(pos[i], radius[i], v as u32);
                vid[i] = Some(v as u32);
            }
        }
        for (ci, chain) in chains.iter().enumerate() {
            for w in chain.windows(2) {
                if let (Some(va), Some(vb)) = (vid[w[0].0 as usize], vid[w[1].0 as usize]) {
                    if va != vb {
                        self.segs.push(Seg {
                            curve: ci as u32,
                            va,
                            vb,
                        });
                    }
                }
            }
        }
        let sources: Vec<(Tri, f64)> = (0..self.db.len())
            .filter(|&v| self.ball[v] > 0.0)
            .map(|v| {
                let c = self.pos(v);
                (Tri::new(c, c, c), BALL_SIZE * self.ball[v])
            })
            .collect();
        let bvh = FacetBvh::build(&sources.iter().map(|s| s.0).collect::<Vec<_>>());
        let targets = Targets::new(&bvh, sources.iter().map(|s| s.1).collect());
        self.ball_size = (bvh, targets);
        vid
    }

    // ------------------------------------------------------------ insertion

    /// Sizes the per-slot caches to the triangulation.
    fn grow_slots(&mut self) {
        let n = self.db.slot_count();
        self.rstate.resize(n, [UNSEEN; 4]);
        self.lpoint.resize(n, None);
    }

    /// After an insertion: queue the new tets and forget what was cached
    /// about their slots, and about the old side of their base faces (every
    /// facet of a new tet has a new dual; the facets between new tets have
    /// new tets on both sides).
    fn after_insert(&mut self) {
        self.grow_slots();
        let created: Vec<u32> = self.db.last_created().to_vec();
        for slot in created {
            self.rstate[slot as usize] = [UNSEEN; 4];
            self.lpoint[slot as usize] = None;
            if let Some((o, of)) = self.db.neighbor_face(slot, 3) {
                self.rstate[o as usize][of] = UNSEEN;
            }
            if let Some(t) = self.db.tet_at(slot) {
                self.facet_queue.push_back((slot, t));
            }
        }
    }

    /// The label point of the tet in `slot` and the clearance there, cached
    /// per slot: a label point ends the duals of all four facets.
    fn label_at(&mut self, slot: u32, t: [usize; 4]) -> (P3, f64) {
        if let Some(pc) = self.lpoint[slot as usize] {
            return pc;
        }
        let p = self.label_point(t);
        let pc = (p, self.dom.clearance(p));
        self.lpoint[slot as usize] = Some(pc);
        pc
    }

    /// Inserts `p` (see [`Mesher::insert_one`]) and, on a periodic patch,
    /// its image on the partner patch. Returns true when the triangulation
    /// changed.
    fn insert_point(&mut self, p: P3, kind: VertexKind) -> bool {
        if !self.insert_one(p, kind) {
            return false;
        }
        if let VertexKind::Patch(a) = kind {
            if let Some((b, t)) = self.per.patch.get(a as usize).copied().flatten() {
                let x = [p[0] + t[0], p[1] + t[1], p[2] + t[2]];
                if !self.insert_one(x, VertexKind::Patch(b)) {
                    self.stats.periodic_missed += 1;
                }
            }
        }
        true
    }

    /// Inserts `p` unless the triangulation is full, `p` lies in a
    /// protecting ball or next to a vertex (the duplicate guard). The
    /// cavity is as large as the point's conflicts: a surface point next to
    /// a coarse region (a fan from one far vertex over a finely sampled
    /// face, concentric circles on one plane) empties thousands of tets.
    fn insert_one(&mut self, p: P3, kind: VertexKind) -> bool {
        if self.db.len() >= self.prm.max_points {
            self.stats.capped = true;
            return false;
        }
        // Protecting balls are fixed: a point inside one is dropped.
        if self.ball_owner(p).is_some() {
            self.stats.rejected += 1;
            self.stats.rejected_in_ball += 1;
            return false;
        }
        let g = DUP_FRAC * self.guard_scale(p);
        match self.db.insert_guarded(p, g * g, usize::MAX, None) {
            Some(_) => {
                self.push_vertex(kind);
                match kind {
                    VertexKind::Volume => self.stats.cell_insertions += 1,
                    _ => self.stats.facet_insertions += 1,
                }
                self.after_insert();
                true
            }
            None => {
                self.stats.rejected += 1;
                false
            }
        }
    }

    /// The facets of periodic patches without an image on the partner:
    /// their circumcenters, with their patch.
    fn periodic_mismatches(&mut self) -> Vec<(P3, u32)> {
        use super::periodic::PointIndex;
        let mut on: FxHashMap<u32, Vec<[usize; 3]>> = FxHashMap::default();
        let mut seen: FxHashSet<[u32; 3]> = FxHashSet::default();
        for (slot, t) in self.db.tets_with_slots() {
            for i in 0..4 {
                let Some(rf) = self.restricted(slot, t, i) else {
                    continue;
                };
                if self
                    .per
                    .patch
                    .get(rf.patch as usize)
                    .copied()
                    .flatten()
                    .is_none()
                {
                    continue;
                }
                let fv: [usize; 3] = std::array::from_fn(|k| t[FACE_LOCAL[i][k]]);
                if seen.insert(key3(fv)) {
                    on.entry(rf.patch).or_default().push(fv);
                }
            }
        }
        let tol = 1e3 * self.per.tol;
        let mut out = Vec::new();
        for pp in self.per.pairs.clone() {
            let empty = Vec::new();
            let (fa, fb) = (
                on.get(&pp.a).unwrap_or(&empty),
                on.get(&pp.b).unwrap_or(&empty),
            );
            let mut index = PointIndex::new(tol);
            for f in fb {
                for &v in f {
                    index.insert(self.pos(v), v);
                }
            }
            let keys_b: FxHashSet<[u32; 3]> = fb.iter().map(|&f| key3(f)).collect();
            let mut matched: FxHashSet<[u32; 3]> = FxHashSet::default();
            let pos = |v: usize| self.pos(v);
            for &f in fa {
                let image: Option<Vec<usize>> = f
                    .iter()
                    .map(|&v| {
                        let p = self.pos(v);
                        let q = [p[0] + pp.shift[0], p[1] + pp.shift[1], p[2] + pp.shift[2]];
                        index.find(q, &pos, tol)
                    })
                    .collect();
                let key = image.map(|w| key3([w[0], w[1], w[2]]));
                match key {
                    Some(k) if keys_b.contains(&k) => {
                        matched.insert(k);
                    }
                    _ => out.push((f, pp.a)),
                }
            }
            for &f in fb {
                if !matched.contains(&key3(f)) {
                    out.push((f, pp.b));
                }
            }
        }
        out.into_iter()
            .map(|(f, patch)| {
                let (a, b, c) = (self.pos(f[0]), self.pos(f[1]), self.pos(f[2]));
                let x = tri_circumcenter(a, b, c).unwrap_or([
                    (a[0] + b[0] + c[0]) / 3.0,
                    (a[1] + b[1] + c[1]) / 3.0,
                    (a[2] + b[2] + c[2]) / 3.0,
                ]);
                (x, patch)
            })
            .collect()
    }

    // ------------------------------------------------------------ facets

    /// The restricted status of the facet of `slot` opposite corner `i`.
    fn restricted(&mut self, slot: u32, t: [usize; 4], i: usize) -> Option<Restricted> {
        match self.rstate[slot as usize][i] {
            PLAIN => return None,
            UNSEEN => {}
            k => return Some(self.rpool[k as usize]),
        }
        let fv: [usize; 3] = std::array::from_fn(|k| t[FACE_LOCAL[i][k]]);
        let r = self.compute_restricted(slot, t, i, fv);
        let state = match r {
            None => PLAIN,
            Some(rf) => {
                self.rpool.push(rf);
                (self.rpool.len() - 1) as u32
            }
        };
        self.rstate[slot as usize][i] = state;
        if let Some((o, of)) = self.db.neighbor_face(slot, i) {
            self.rstate[o as usize][of] = state;
        }
        r
    }

    fn compute_restricted(
        &mut self,
        slot: u32,
        t: [usize; 4],
        i: usize,
        fv: [usize; 3],
    ) -> Option<Restricted> {
        let (a, b, c) = (self.pos(fv[0]), self.pos(fv[1]), self.pos(fv[2]));
        let (c1, r1) = self.label_at(slot, t);
        let across = self
            .db
            .neighbor_at(slot, i)
            .and_then(|nb| self.db.tet_at(nb).map(|nt| (nb, nt)));
        let (c2, r2) = match across {
            Some((nb, nt)) => self.label_at(nb, nt),
            None => {
                // Hull facet: the dual is a ray from c1 away from the tet.
                let mut n = cross(sub(b, a), sub(c, a));
                let l = dot(n, n).sqrt();
                if !(l > 0.0) {
                    return None;
                }
                n = n.map(|x| x / l);
                let opp = self.pos(t[i]);
                if dot(sub(opp, a), n) > 0.0 {
                    n = n.map(|x| -x);
                }
                let far = [
                    c1[0] + self.far * n[0],
                    c1[1] + self.far * n[1],
                    c1[2] + self.far * n[2],
                ];
                (far, 0.0)
            }
        };
        let fc = tri_circumcenter(a, b, c).unwrap_or([
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ]);
        // The dual, stretched by a relative 1e-9 at both ends: a label point
        // exactly on a surface (symmetric sampling puts circumcenters on
        // planes) then still sees the crossing it sits on, which a strict
        // sign-change test at the endpoint misses.
        let d = sub(c2, c1);
        let e = 1e-9;
        let a1: P3 = std::array::from_fn(|k| c1[k] - e * d[k]);
        let a2: P3 = std::array::from_fn(|k| c2[k] + e * d[k]);
        // The clear balls around the label points, shrunk by the stretch,
        // are clear around the ends: when they reach past the dual, it
        // crosses nothing.
        let slack = e * dot(d, d).sqrt();
        if ((r1 - slack) + (r2 - slack)) * (1.0 - 1e-9) > dist(a1, a2) {
            return None;
        }
        // Of several crossings (a thin feature crossed twice), the one
        // nearest the facet is its surface center.
        let x = self.dom.nearest_crossing_screened(a1, a2, fc)?;
        Some(Restricted {
            center: x.point,
            // The power distance to a facet vertex (equal for all three on
            // the dual).
            radius: (dot(sub(x.point, a), sub(x.point, a)) - self.db.weight(fv[0]))
                .max(0.0)
                .sqrt(),
            patch: x.patch,
        })
    }

    /// Examines the four facets of a tet; inserts at most one surface point.
    /// Returns true when the triangulation changed.
    fn refine_facets(&mut self, slot: u32, t: [usize; 4]) -> bool {
        for i in 0..4 {
            let Some(rf) = self.restricted(slot, t, i) else {
                continue;
            };
            let fv: [usize; 3] = std::array::from_fn(|k| t[FACE_LOCAL[i][k]]);
            let (a, b, c) = (self.pos(fv[0]), self.pos(fv[1]), self.pos(fv[2]));
            let h = self.h(rf.center);
            let big = rf.radius > self.prm.facet_size * h;
            // Shape, distance and topology only refine above a floor: at the
            // duplicate-guard scale they would churn without converging. A
            // surface alone has no cells to shape, so its facets refine
            // down to the feature size the protecting balls encode (a
            // short curve segment next to a long facet), as in Ruppert's
            // refinement.
            let scale = if self.prm.surface_only {
                self.guard_scale(rf.center)
            } else {
                h
            };
            let above = rf.radius > 0.3 * scale;
            let flat = above && tri_min_angle(a, b, c) < self.prm.facet_angle_deg;
            let far = above && plane_distance(a, b, c, rf.center) > self.prm.facet_distance * h;
            // Topology (Mesh_3's facet-vertices-on-surface): a restricted
            // facet with a vertex off its patch refines at any size; the
            // duplicate guard bounds it.
            let off = !fv.iter().all(|&v| self.on_patch(v, rf.patch));
            if big || flat || far || off {
                if self.insert_point(rf.center, VertexKind::Patch(rf.patch)) {
                    return true;
                }
                if self.prm.surface_only {
                    if let Some(v) = self.ball_owner(rf.center) {
                        let cap = 0.9 * dist(rf.center, self.pos(v as usize));
                        self.blocked.push((v, cap));
                    }
                }
            }
        }
        false
    }

    /// The ball a thin facet under the refinement floor skims, with the
    /// radius that frees the facet (half the distance to the other two
    /// vertices): the facet's shortest edge faces a vertex whose ball
    /// reaches nearly to the others, so the power distance of its surface
    /// center, and any point inserted there, stay below the floor. None
    /// for the angle between two curve segments at the vertex (an input
    /// angle no ball size changes).
    fn skimmed_ball(&self, fv: [usize; 3]) -> Option<(u32, f64)> {
        let p = fv.map(|v| self.pos(v));
        let k = (0..3).max_by(|&x, &y| self.ball[fv[x]].total_cmp(&self.ball[fv[y]]))?;
        let (v, q, r) = (fv[k], fv[(k + 1) % 3], fv[(k + 2) % 3]);
        let (dq, dr) = (dist(p[k], p[(k + 1) % 3]), dist(p[k], p[(k + 2) % 3]));
        let opposite = dist(p[(k + 1) % 3], p[(k + 2) % 3]);
        let cap = 0.5 * dq.min(dr);
        if !(self.ball[v] > cap) || opposite > dq.min(dr) {
            return None;
        }
        let seg = |a: usize, b: usize| {
            self.segs.iter().any(|s| {
                (s.va as usize, s.vb as usize) == (a, b) || (s.va as usize, s.vb as usize) == (b, a)
            })
        };
        if seg(v, q) && seg(v, r) {
            return None;
        }
        Some((v as u32, cap))
    }

    // ------------------------------------------------------------ cells

    /// Refines a tet in a region at its circumcenter when it is too large or
    /// badly shaped. Returns true when the triangulation changed.
    fn refine_cell(&mut self, slot: u32, t: [usize; 4]) -> bool {
        let Some((cc, r)) = self.cc(t) else {
            return false;
        };
        let p = t.map(|v| self.pos(v));
        // A tet in the domain whose circumcenter lies outside it would be
        // labelled outside and dropped. The boundary through its Delaunay
        // ball can be hidden from the facet refinement: the duals of its
        // facets cross it only beside a nearer boundary (a lid over a
        // finely sampled face, fanned to one vertex). Where the ball holds
        // a surface disc larger than a facet may be (the ball is empty, so
        // the crossing on the way to the center lies that far from every
        // vertex), the tet refines the boundary there, as a circumcenter
        // encroaching a subfacet does (Shewchuk).
        let region = self.dom.region(cc);
        if region == 0 {
            let g: P3 = std::array::from_fn(|k| 0.25 * (p[0][k] + p[1][k] + p[2][k] + p[3][k]));
            if self.dom.region(g) == 0 {
                return false;
            }
            let ok = self.dom.nearest_crossing(g, cc, g).is_some_and(|x| {
                r - dist(x.point, cc) > self.prm.facet_size * self.h(x.point)
                    && self.insert_point(x.point, VertexKind::Patch(x.patch))
            });
            self.stats.encroached += usize::from(ok);
            return ok;
        }
        let mut lmin = f64::INFINITY;
        for a in 0..4 {
            for b in a + 1..4 {
                lmin = lmin.min(dist(p[a], p[b]));
            }
        }
        let h = self.h(cc);
        let large_or_long =
            r > self.prm.cell_size * h || r / lmin.max(1e-300) > self.prm.radius_edge;
        let q = tet_min_dihedral(p);
        if !large_or_long && q >= self.prm.sliver_deg {
            return false;
        }
        // A sliver (small dihedral, fine size and radius-edge) is refined at
        // the candidate near its orthocenter whose new tets are best (Li and
        // Teng): the orthocenter itself often recreates a sliver.
        let cc = if large_or_long {
            cc
        } else {
            match self.sliver_point(cc, r, q, lmin) {
                Some(x) => x,
                // A sliver on the boundary (its vertices on the surface) has
                // no interior candidate that helps: refine its largest
                // restricted facet instead, which reshapes the surface there.
                None => {
                    let ok = self.refine_largest_facet(slot, t);
                    self.stats.sliver_facets += usize::from(ok);
                    return ok;
                }
            }
        };
        // A cell point closer to the surface than a surface point's
        // duplicate guard would get the surface points that belong there
        // rejected as its duplicates, and stay a vertex of a restricted
        // facet off the surface. Like a circumcenter encroaching a subfacet
        // (Shewchuk), it refines the tet's largest restricted facet instead.
        // A tet with none to refine (inside a layer thinner than the guard)
        // still takes its point: the alternative is no refinement at all,
        // and snapping adopts such a vertex if it ends up on the surface.
        if self
            .dom
            .patch_within(cc, NEAR_SURFACE * DUP_FRAC * self.guard_scale(cc), &[])
        {
            if self.refine_largest_facet(slot, t) {
                return true;
            }
            self.stats.near_surface += 1;
        }
        // A circumcenter inside the surface ball of a restricted facet it
        // would remove refines that facet instead.
        let mut blocker: Option<Restricted> = None;
        let (rstate, rpool) = (&self.rstate, &self.rpool);
        let mut keep = |s: u32, i: usize| -> bool {
            match rstate.get(s as usize).map_or(UNSEEN, |x| x[i]) {
                UNSEEN | PLAIN => true,
                k => {
                    let rf = rpool[k as usize];
                    if dist(cc, rf.center) < rf.radius {
                        blocker = Some(rf);
                        return false;
                    }
                    true
                }
            }
        };
        // `insert_point` needs `&mut self`; run the guarded insert through a
        // copy of the cache reference captured above.
        let inserted = {
            if self.db.len() >= self.prm.max_points {
                self.stats.capped = true;
                return false;
            }
            if self.ball_owner(cc).is_some() {
                return false;
            }
            // The guard follows the tet's own scale too: where the size
            // field grades from a fine surface to a coarse interior, h at the
            // orthocenter overstates the local spacing.
            let g = DUP_FRAC * h.min(lmin);
            self.db
                .insert_guarded(cc, g * g, usize::MAX, Some(&mut keep))
                .is_some()
        };
        if inserted {
            self.push_vertex(VertexKind::Volume);
            self.stats.cell_insertions += 1;
            self.after_insert();
            return true;
        }
        if let Some(rf) = blocker {
            return self.insert_point(rf.center, VertexKind::Patch(rf.patch));
        }
        self.stats.rejected += 1;
        false
    }

    /// Refines the restricted facet of `t` with the largest surface ball.
    fn refine_largest_facet(&mut self, slot: u32, t: [usize; 4]) -> bool {
        let best = (0..4)
            .filter_map(|i| self.restricted(slot, t, i))
            .max_by(|a, b| a.radius.total_cmp(&b.radius));
        best.is_some_and(|rf| self.insert_point(rf.center, VertexKind::Patch(rf.patch)))
    }

    /// The best of a few candidates around a sliver's orthocenter: the one
    /// whose new tets have the largest smallest dihedral, if that beats the
    /// sliver's own `q`. Candidates outside every region or inside a
    /// protecting ball are skipped.
    fn sliver_point(&mut self, cc: P3, r: f64, q: f64, lmin: f64) -> Option<P3> {
        let g = DUP_FRAC * self.h(cc).min(lmin);
        let mut best: Option<(f64, P3)> = None;
        let mut faces = std::mem::take(&mut self.probe_faces);
        for k in 0..=SLIVER_CANDIDATES.len() {
            let c: P3 = if k == 0 {
                cc
            } else {
                let d = SLIVER_CANDIDATES[k - 1];
                std::array::from_fn(|i| cc[i] + SLIVER_SPREAD * r * d[i])
            };
            if self.dom.region(c) == 0 || self.ball_owner(c).is_some() {
                continue;
            }
            faces.clear();
            if !self.db.probe(c, g * g, SLIVER_CAVITY, &mut faces) {
                continue;
            }
            let worst = faces
                .iter()
                .map(|f| tet_min_dihedral([self.pos(f[0]), self.pos(f[1]), self.pos(f[2]), c]))
                .fold(f64::INFINITY, f64::min);
            if best.is_none_or(|(b, _)| worst > b) {
                best = Some((worst, c));
            }
        }
        self.probe_faces = faces;
        let (bq, c) = best?;
        (bq > q).then_some(c)
    }

    // ------------------------------------------------------------ driver

    fn run(&mut self) {
        self.grow_slots();
        for (slot, t) in self.db.tets_with_slots() {
            self.facet_queue.push_back((slot, t));
        }
        loop {
            while let Some((slot, t)) = self.facet_queue.pop_front() {
                if self.db.tet_at(slot) != Some(t) {
                    continue;
                }
                if self.refine_facets(slot, t) {
                    if self.db.tet_at(slot) == Some(t) {
                        self.facet_queue.push_back((slot, t));
                    }
                } else {
                    self.cell_queue.push_back((slot, t));
                }
            }
            if self.prm.surface_only {
                break;
            }
            let mut advanced = false;
            while let Some((slot, t)) = self.cell_queue.pop_front() {
                if self.db.tet_at(slot) != Some(t) {
                    continue;
                }
                if self.refine_cell(slot, t) {
                    if self.db.tet_at(slot) == Some(t) {
                        self.cell_queue.push_back((slot, t));
                    }
                    advanced = true;
                    break;
                }
            }
            if !advanced && self.facet_queue.is_empty() {
                break;
            }
        }
    }

    // ------------------------------------------------------------ output

    fn extract(&mut self) -> Complex {
        use rayon::prelude::*;
        let all = self.db.tets_with_slots();
        // Label each real tet by the region at its circumcenter (centroid
        // for a degenerate tet).
        let labels: Vec<u32> = {
            let me = &*self;
            all.par_iter().map(|&(_, t)| me.label(t)).collect()
        };
        let label_of: FxHashMap<u32, u32> = all
            .iter()
            .zip(&labels)
            .map(|(&(s, _), &l)| (s, l))
            .collect();
        let mut out = Complex {
            points: (0..self.db.len()).map(|v| self.pos(v)).collect(),
            kinds: self.kinds.clone(),
            ..Complex::default()
        };
        let mut fe: Vec<([u32; 2], u32)> =
            self.segs.iter().map(|s| ([s.va, s.vb], s.curve)).collect();
        fe.sort_unstable_by_key(|e| (e.1, e.0));
        out.feature_edges = fe;
        let mut seen: FxHashSet<[u32; 3]> = FxHashSet::default();
        for (&(slot, t), &la) in all.iter().zip(&labels) {
            if la != 0 {
                out.tets.push(t.map(|v| v as u32));
                out.regions.push(la);
            }
            for i in 0..4 {
                let fv: [usize; 3] = std::array::from_fn(|k| t[FACE_LOCAL[i][k]]);
                let key = key3(fv);
                let lb = self
                    .db
                    .neighbor_at(slot, i)
                    .and_then(|nb| label_of.get(&nb).copied())
                    .unwrap_or(0);
                if la == lb {
                    // A sheet face: a restricted facet of a sheet patch of
                    // this region, emitted once.
                    if la == 0 || seen.contains(&key) {
                        continue;
                    }
                    if let Some(rf) = self.restricted(slot, t, i) {
                        let p = self.dom.patches()[rf.patch as usize];
                        if p.is_sheet() && p.regions[0] == la {
                            seen.insert(key);
                            out.faces.push(Face {
                                tri: fv.map(|v| v as u32),
                                regions: [la, la],
                                patch: rf.patch,
                            });
                        }
                    }
                    continue;
                }
                // A label change, emitted from the side of the tet whose
                // label is the front region (winding: this tet positive).
                if seen.contains(&key) {
                    continue;
                }
                seen.insert(key);
                let patch = self.interface_patch(slot, t, i, la, lb);
                out.faces.push(Face {
                    tri: fv.map(|v| v as u32),
                    regions: [la, lb],
                    patch,
                });
            }
        }
        out
    }

    /// The surface: every restricted facet once, with its patch and the
    /// patch's regions (the caller orients them); no tets.
    fn extract_surface(&mut self) -> Complex {
        let mut out = Complex {
            points: (0..self.db.len()).map(|v| self.pos(v)).collect(),
            kinds: self.kinds.clone(),
            ..Complex::default()
        };
        let mut fe: Vec<([u32; 2], u32)> =
            self.segs.iter().map(|s| ([s.va, s.vb], s.curve)).collect();
        fe.sort_unstable_by_key(|e| (e.1, e.0));
        out.feature_edges = fe;
        let mut seen: FxHashSet<[u32; 3]> = FxHashSet::default();
        for (slot, t) in self.db.tets_with_slots() {
            for i in 0..4 {
                let fv: [usize; 3] = std::array::from_fn(|k| t[FACE_LOCAL[i][k]]);
                let key = key3(fv);
                if seen.contains(&key) {
                    continue;
                }
                if let Some(rf) = self.restricted(slot, t, i) {
                    seen.insert(key);
                    out.faces.push(Face {
                        tri: fv.map(|v| v as u32),
                        regions: self.dom.patches()[rf.patch as usize].regions,
                        patch: rf.patch,
                    });
                }
            }
        }
        out
    }

    /// The patch separating labels `la` and `lb` across a facet: the
    /// restricted facet's patch when it separates them, else any crossing
    /// of the dual that does (`u32::MAX` if none).
    fn interface_patch(&mut self, slot: u32, t: [usize; 4], i: usize, la: u32, lb: u32) -> u32 {
        let matches = |d: &D, p: u32| {
            let r = d.patches()[p as usize].regions;
            (r[0] == la && r[1] == lb) || (r[0] == lb && r[1] == la)
        };
        if let Some(rf) = self.restricted(slot, t, i) {
            if matches(self.dom, rf.patch) {
                return rf.patch;
            }
        }
        // Oracle ties (a label point exactly on a surface) can hide the
        // crossing: take the lowest patch that all three vertices lie on and
        // that separates the two labels.
        let fv: [usize; 3] = std::array::from_fn(|k| t[FACE_LOCAL[i][k]]);
        (0..self.dom.patches().len() as u32)
            .find(|&p| matches(self.dom, p) && fv.iter().all(|&v| self.on_patch(v, p)))
            .unwrap_or(u32::MAX)
    }
}

/// Meshes the domain: restricted-Delaunay refinement, then extraction.
pub fn mesh<D: DomainOracle + ?Sized, S: SizeField + ?Sized>(
    dom: &D,
    size: &S,
    prm: &Params,
) -> (Complex, Stats) {
    if prm.surface_only {
        return mesh_surface(dom, size, prm);
    }
    let mut m = Mesher::new(dom, size, prm);
    let pr = m.protect();
    m.log_protection();
    m.insert_protection(&pr, &|_| true);
    for (pi, _) in dom.patches().iter().enumerate() {
        // The second patch of a periodic pair takes the seeds of the first.
        if prm.periodic.iter().any(|pp| pp.b as usize == pi) {
            continue;
        }
        for p in dom.patch_seeds(pi as u32) {
            m.insert_point(p, VertexKind::Patch(pi as u32));
        }
    }
    m.run();
    // Periodic pairs: equal point sets can still triangulate differently
    // where points lie on a circle; the circumcenter of a facet without
    // its image goes on both sides, until they match.
    if !prm.periodic.is_empty() {
        for _ in 0..PERIODIC_ROUNDS {
            let fixes = m.periodic_mismatches();
            m.stats.periodic_left = fixes.len();
            if fixes.is_empty() {
                break;
            }
            m.stats.periodic_rounds += 1;
            for (x, patch) in fixes {
                m.insert_point(x, VertexKind::Patch(patch));
            }
            m.run();
        }
        m.stats.periodic_left = m.periodic_mismatches().len();
    }
    let c = m.extract();
    (c, m.stats)
}

/// Rounds of matching the two sides of periodic pairs.
const PERIODIC_ROUNDS: usize = 8;

/// The surface alone, patch by patch: every patch is the restricted
/// Delaunay triangulation of its own samples, grown from the protected
/// samples of its curves and corners, which all patches share. Patches of
/// other faces never enter a patch's triangulation, so surfaces closer
/// than the size (stacked sheets, a thin wall) do not refine each other,
/// as they must in one triangulation of all of them; the shared samples
/// keep the patches conforming along their curves.
///
/// A surface center inside a protecting ball caps that ball below the
/// center's distance (Ruppert splitting an encroached segment): the
/// protection settles again, and the patches of the balls it changed
/// are meshed again.
fn mesh_surface<D: DomainOracle + ?Sized, S: SizeField + ?Sized>(
    dom: &D,
    size: &S,
    prm: &Params,
) -> (Complex, Stats) {
    use rayon::prelude::*;
    let mut m = Mesher::new(dom, size, prm);
    let mut pr = m.protect();
    m.log_protection();
    let corner_patches = m.corner_patches.clone();
    let patches_of = |k: VertexKind| -> &[u32] {
        match k {
            VertexKind::Corner(c) => &corner_patches[c as usize],
            VertexKind::Curve(c) => &dom.curves()[c as usize].patches,
            _ => &[],
        }
    };
    let np = dom.patches().len();
    let mut parts: Vec<Option<PatchMesh>> = (0..np).map(|_| None).collect();
    let mut dirty: Vec<u32> = (0..np as u32).collect();
    for round in 0..SURFACE_ROUNDS {
        let done: Vec<(u32, PatchMesh)> = dirty
            .par_iter()
            .map(|&patch| {
                let one = OnePatch { inner: dom, patch };
                let mut m = Mesher::new(&one, size, prm);
                m.init_corner_patches();
                // The box corners first, while their cavities are small.
                m.insert_box_corners();
                let vid = m.insert_protection(&pr, &|k| patches_of(k).contains(&patch));
                for p in dom.patch_seeds(patch) {
                    m.insert_point(p, VertexKind::Patch(patch));
                }
                m.run();
                let complex = m.extract_surface();
                // Facets left thin by a ball they skim.
                for f in &complex.faces {
                    let p = f.tri.map(|v| complex.points[v as usize]);
                    if tri_min_angle(p[0], p[1], p[2]) < prm.facet_angle_deg {
                        if let Some(b) = m.skimmed_ball(f.tri.map(|v| v as usize)) {
                            m.blocked.push(b);
                        }
                    }
                }
                let mut node = vec![u32::MAX; m.db.len()];
                for (i, v) in vid.iter().enumerate() {
                    if let Some(v) = v {
                        node[*v as usize] = i as u32;
                    }
                }
                let blocked = m
                    .blocked
                    .iter()
                    .map(|&(v, p)| (node[v as usize], p))
                    .filter(|b| b.0 != u32::MAX)
                    .collect();
                (
                    patch,
                    PatchMesh {
                        complex,
                        vid,
                        blocked,
                        stats: m.stats,
                    },
                )
            })
            .collect();
        for (patch, part) in done {
            for &(i, cap) in &part.blocked {
                let i = i as usize;
                pr.cap[i] = pr.cap[i].min(cap);
            }
            parts[patch as usize] = Some(part);
        }
        if round + 1 == SURFACE_ROUNDS {
            break;
        }
        let (n0, r0) = (pr.pos.len(), pr.radius.clone());
        m.settle(&mut pr);
        let mut changed: FxHashSet<u32> = FxHashSet::default();
        for i in 0..pr.pos.len() {
            if i >= n0 || pr.radius[i] != r0[i] {
                changed.extend(patches_of(pr.kind[i]));
            }
        }
        dirty = changed.into_iter().collect();
        dirty.sort_unstable();
        if dirty.is_empty() {
            break;
        }
    }

    // Merge: the protected samples once, then every patch's own points.
    let mut out = Complex {
        points: pr.pos.clone(),
        kinds: pr.kind.clone(),
        ..Complex::default()
    };
    let mut stats = m.stats;
    let mut edges: FxHashSet<([u32; 2], u32)> = FxHashSet::default();
    for part in parts.into_iter().flatten() {
        let c = &part.complex;
        let mut map: Vec<u32> = vec![u32::MAX; c.points.len()];
        for (i, v) in part.vid.iter().enumerate() {
            if let Some(v) = v {
                map[*v as usize] = i as u32;
            }
        }
        for f in &c.faces {
            for &v in &f.tri {
                let v = v as usize;
                if map[v] == u32::MAX {
                    map[v] = out.points.len() as u32;
                    out.points.push(c.points[v]);
                    out.kinds.push(c.kinds[v]);
                }
            }
        }
        out.faces.extend(c.faces.iter().map(|f| Face {
            tri: f.tri.map(|v| map[v as usize]),
            ..*f
        }));
        for &([a, b], curve) in &c.feature_edges {
            let (a, b) = (map[a as usize], map[b as usize]);
            if a != u32::MAX && b != u32::MAX {
                edges.insert(([a.min(b), a.max(b)], curve));
            }
        }
        stats.facet_insertions += part.stats.facet_insertions;
        stats.rejected += part.stats.rejected;
        stats.rejected_in_ball += part.stats.rejected_in_ball;
    }
    let mut fe: Vec<([u32; 2], u32)> = edges.into_iter().collect();
    fe.sort_unstable_by_key(|e| (e.1, e.0));
    out.feature_edges = fe;
    (out, stats)
}

/// Rounds of the surface mode: mesh the patches, cap the balls holding
/// centers off, settle the protection.
const SURFACE_ROUNDS: usize = 8;

/// One patch meshed alone: its complex, the vertex of every protected
/// sample in it, and the centers protecting balls held off (sample,
/// center).
struct PatchMesh {
    complex: Complex,
    vid: Vec<Option<u32>>,
    blocked: Vec<(u32, f64)>,
    stats: Stats,
}

/// A domain seen through one of its patches: crossings with the others
/// vanish, the patch ids, corners and curves stay those of the domain.
struct OnePatch<'a, D: ?Sized> {
    inner: &'a D,
    patch: u32,
}

impl<D: DomainOracle + ?Sized> DomainOracle for OnePatch<'_, D> {
    fn bbox(&self) -> (P3, P3) {
        self.inner.bbox()
    }
    fn region(&self, p: P3) -> u32 {
        self.inner.region(p)
    }
    fn crossings(&self, a: P3, b: P3, out: &mut Vec<Crossing>) {
        let start = out.len();
        self.inner.crossings(a, b, out);
        let mut k = start;
        for i in start..out.len() {
            if out[i].patch == self.patch {
                out[k] = out[i];
                k += 1;
            }
        }
        out.truncate(k);
    }
    fn patches(&self) -> &[Patch] {
        self.inner.patches()
    }
    fn corners(&self) -> &[P3] {
        self.inner.corners()
    }
    fn curves(&self) -> &[FeatureCurve] {
        self.inner.curves()
    }
    fn patch_seeds(&self, patch: u32) -> Vec<P3> {
        if patch == self.patch {
            self.inner.patch_seeds(patch)
        } else {
            Vec::new()
        }
    }
    fn corner_patches(&self, corner: u32) -> Vec<u32> {
        self.inner.corner_patches(corner)
    }
    fn patch_within(&self, p: P3, r: f64, except: &[u32]) -> bool {
        !except.contains(&self.patch) && self.inner.patch_within(p, r, except)
    }
}

#[cfg(test)]
mod tests {
    use super::super::oracle::domains::*;
    use super::super::oracle::Uniform;
    use super::super::verify::check;
    use super::*;

    #[test]
    fn a_ball_meshes_into_a_valid_complex() {
        let d = Balls::new([0.0; 3], &[1.0]);
        let (c, st) = mesh(&d, &Uniform(0.25), &Params::default());
        let r = check(&c);
        assert!(r.ok(), "{r:?} {st:?}");
        let v = r.volume(1);
        let exact = 4.0 / 3.0 * std::f64::consts::PI;
        assert!((v - exact).abs() < 0.05 * exact, "volume {v} vs {exact}");
    }

    #[test]
    fn nested_balls_conform_at_the_interface() {
        let d = Balls::new([0.0; 3], &[0.5, 1.0]);
        let (c, st) = mesh(&d, &Uniform(0.2), &Params::default());
        let r = check(&c);
        assert!(r.ok(), "{r:?} {st:?}");
        let pi = std::f64::consts::PI;
        let (v1, v2) = (4.0 / 3.0 * pi * 0.125, 4.0 / 3.0 * pi * (1.0 - 0.125));
        assert!(
            (r.volume(1) - v1).abs() < 0.08 * v1,
            "{} vs {v1}",
            r.volume(1)
        );
        assert!(
            (r.volume(2) - v2).abs() < 0.05 * v2,
            "{} vs {v2}",
            r.volume(2)
        );
    }

    /// The faces of `patch` as sorted position triples shifted by `t`,
    /// rounded to compare across the sides.
    fn face_keys(c: &Complex, patch: u32, t: P3) -> Vec<[[i64; 3]; 3]> {
        let q = |p: P3| std::array::from_fn(|k| ((p[k] + t[k]) * 1e9).round() as i64);
        let mut out: Vec<[[i64; 3]; 3]> = c
            .faces
            .iter()
            .filter(|f| f.patch == patch)
            .map(|f| {
                let mut k = f.tri.map(|v| q(c.points[v as usize]));
                k.sort_unstable();
                k
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// A box periodic in x and y: a valid complex whose opposite sides carry
    /// the same triangles.
    #[test]
    fn periodic_sides_carry_the_same_triangles() {
        use super::super::periodic::PeriodicPair;
        let (lx, ly) = (1.0, 0.7);
        let dom = Cube::new([0.0; 3], [lx, ly, 0.4]);
        let prm = Params {
            periodic: vec![
                PeriodicPair {
                    a: 0,
                    b: 1,
                    shift: [lx, 0.0, 0.0],
                },
                PeriodicPair {
                    a: 2,
                    b: 3,
                    shift: [0.0, ly, 0.0],
                },
            ],
            ..Params::default()
        };
        let (c, st) = mesh(&dom, &Uniform(0.05), &prm);
        let r = check(&c);
        assert!(r.ok(), "{r:?}");
        assert_eq!(st.periodic_left, 0, "{st:?}");
        assert_eq!(st.periodic_missed, 0, "{st:?}");
        for (a, b, t) in [(0, 1, [lx, 0.0, 0.0]), (2, 3, [0.0, ly, 0.0])] {
            let (ka, kb) = (face_keys(&c, a, t), face_keys(&c, b, [0.0; 3]));
            assert!(!ka.is_empty());
            assert_eq!(ka, kb, "patches {a} and {b}");
        }
    }

    #[test]
    fn a_cube_keeps_its_corners_and_volume() {
        let d = Cube::new([0.0; 3], [1.0, 1.0, 1.0]);
        let (c, st) = mesh(&d, &Uniform(0.3), &Params::default());
        let r = check(&c);
        assert!(r.ok(), "{r:?} {st:?}");
        assert!((r.volume(1) - 1.0).abs() < 1e-9, "volume {}", r.volume(1));
    }

    #[test]
    fn rigid_motions_keep_every_invariant() {
        use super::super::oracle::Moved;
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..12 {
            let axis = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
            let angle = 6.3 * rnd();
            let shift = [3.0 * rnd(), -2.0 * rnd(), rnd()];
            let h = 0.15 + 0.2 * rnd();
            let (c, st, want) = match case % 3 {
                0 => {
                    let d = Moved::new(Cube::new([0.0; 3], [1.0, 0.7, 0.4]), axis, angle, shift);
                    let (c, st) = mesh(&d, &Uniform(h), &Params::default());
                    (c, st, vec![(1, 0.28, 1e-9)])
                }
                1 => {
                    let d = Moved::new(Balls::new([0.0; 3], &[0.5, 1.0]), axis, angle, shift);
                    let (c, st) = mesh(&d, &Uniform(h), &Params::default());
                    let pi = std::f64::consts::PI;
                    (
                        c,
                        st,
                        vec![
                            (1, 4.0 / 3.0 * pi * 0.125, 0.12),
                            (2, 4.0 / 3.0 * pi * 0.875, 0.08),
                        ],
                    )
                }
                _ => {
                    let d = Moved::new(BallWithSheet::new([0.0; 3], 1.0, 0.6), axis, angle, shift);
                    let (c, st) = mesh(&d, &Uniform(h), &Params::default());
                    (c, st, vec![(1, 4.0 / 3.0 * std::f64::consts::PI, 0.06)])
                }
            };
            let r = check(&c);
            assert!(r.ok(), "case {case} h {h}: {r:?} {st:?}");
            for (reg, v, tol) in want {
                let got = r.volume(reg);
                assert!(
                    (got - v).abs() <= tol * v,
                    "case {case} region {reg}: {got} vs {v}"
                );
            }
        }
    }

    #[test]
    fn a_thin_shell_is_resolved() {
        // A shell far thinner than the target size: duals skip the shell,
        // and region boundaries pinch, until topology refinement resolves it.
        let d = Balls::new([0.0; 3], &[1.0, 1.03]);
        let (c, st) = mesh(&d, &Uniform(0.25), &Params::default());
        let r = check(&c);
        assert!(r.ok(), "{r:?} {st:?}");
        let pi = std::f64::consts::PI;
        let shell = 4.0 / 3.0 * pi * (1.03f64.powi(3) - 1.0);
        assert!(
            (r.volume(2) - shell).abs() < 0.15 * shell,
            "{} vs {shell}",
            r.volume(2)
        );
    }

    #[test]
    fn a_sheet_inside_a_ball_is_kept() {
        let d = BallWithSheet::new([0.0; 3], 1.0, 0.6);
        let (c, st) = mesh(&d, &Uniform(0.2), &Params::default());
        let r = check(&c);
        assert!(r.ok(), "{r:?} {st:?}");
        let sheet: f64 = c
            .faces
            .iter()
            .filter(|f| f.regions == [1, 1])
            .map(|f| {
                let p = |i: u32| c.points[i as usize];
                let n = cross(sub(p(f.tri[1]), p(f.tri[0])), sub(p(f.tri[2]), p(f.tri[0])));
                0.5 * dot(n, n).sqrt()
            })
            .sum();
        let disc = std::f64::consts::PI * 0.36;
        assert!(
            (sheet - disc).abs() < 0.05 * disc,
            "sheet area {sheet} vs {disc}"
        );
    }
}
