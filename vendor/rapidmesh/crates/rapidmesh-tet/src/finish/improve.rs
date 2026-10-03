//! Local improvement of a finished complex: the few tets below a dihedral
//! target are repaired by 2-3 and 3-2 flips and by moving their free
//! vertices, each change accepted only when it raises the worst dihedral
//! among the tets it touches.
//!
//! Every change keeps what the mesher guarantees: tets stay positively
//! oriented, no flip removes a face (boundary, interface or sheet) or a
//! feature edge, and flips stay inside one region. Volume vertices on no
//! face move freely, vertices on one patch move along its carrier without
//! turning a face, vertices inside a smooth curve along it; the rest stay.
//! With a shape, a flat tet on a boundary may also be peeled off (or across
//! an interface), which flips a surface edge. No change makes a surface face bridge its carrier. Work is
//! proportional to the bad tets, not to the mesh; a new tet takes the
//! place of a dead one, so the arrays keep the mesh's size.
//!
//! Last comes a relaxation of the surface for its triangles, whose shape
//! the tets on the boundary follow (gmsh's lead in the mean dihedral came
//! from there): vertices on one smooth patch go toward the centroid of
//! their faces, vertices inside a smooth curve toward the midpoint of
//! their neighbours, where the smallest angle around them rises, their
//! faces stray no further from the carriers, no tet at the target falls
//! below it, none below gets worse and the mean of their star does not
//! fall.

use crate::finish::snap::{adopt_face_vertices, Shape, FLOOR_DEG, HALVINGS};
use crate::finish::P3;
use crate::finish::{Complex, Face, PointClass};
use crate::simplex::{tet_min_dihedral, tri_min_angle};
use geometry_predicates::orient3d;
use rapidmesh_geom::vec3::{cross, dist, dot, len, sub};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// What [`finish`] did to the tets.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImproveStats {
    pub flips23: usize,
    pub flips32: usize,
    pub flips44: usize,
    pub moves: usize,
    /// Tets below the target before and after.
    pub bad_before: usize,
    pub bad_after: usize,
}

/// A change must raise the local worst dihedral by at least this (degrees).
const GAIN: f64 = 1e-3;

/// A surface face whose centroid is off its carrier by more than this share
/// of its longest edge bridges it; no change makes one.
const BRIDGE: f64 = 0.1;

/// Sweeps of the surface relaxation after the repair (a third gains next
/// to nothing on the gmsh compare pairs).
const RELAX_SWEEPS: usize = 2;

/// Vertex move candidates: these directions (icosahedron vertices) at these
/// fractions of the vertex's shortest edge.
const DIRS: [P3; 12] = {
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
const STEPS: [f64; 3] = [0.3, 0.15, 0.05];

/// Vertices with more tets than this stay where they are: a move costs
/// its candidates times its star, and so large a star (a fan over a
/// degenerate spot, far beyond the few dozen of a sound mesh) changes all
/// the time, so it would be tried again and again.
const MOVE_STAR_MAX: usize = 512;

/// A star this large is searched from another vertex where one has a
/// smaller star (see [`Improver::hub`]).
const LARGE_STAR: usize = 256;

/// Of the moves planned for the corners of a bad tet, the one whose star
/// keeps the largest worst dihedral: the first move that gains anything is
/// often a surface vertex nudged by a hair, while the free vertex would
/// lift the tet, and the same nudge then wins every sweep.
fn best_move(moves: [Option<Plan>; 4]) -> Plan {
    let worst = |p: &Plan| match p {
        Plan::Move { star, .. } => star.iter().map(|s| s.1).fold(f64::INFINITY, f64::min),
        _ => f64::NEG_INFINITY,
    };
    moves
        .into_iter()
        .flatten()
        .max_by(|a, b| worst(a).total_cmp(&worst(b)))
        .unwrap_or(Plan::Fail)
}

fn sorted3(f: [u32; 3]) -> [u32; 3] {
    let mut s = f;
    s.sort_unstable();
    s
}

/// Bad tets planned in parallel at a time, between these bounds after the
/// conflicts of the batch before. A plan reads only the stars of the tet's
/// vertices, so it stands as long as no change in the batch before it
/// touched one of them; otherwise the tet is planned again when its turn
/// comes. The result is the serial one, in any thread count and batch size.
const BATCH_MIN: usize = 32;
const BATCH_START: usize = 256;
const BATCH_MAX: usize = 1024;

#[derive(Clone, Copy)]
enum Op {
    Flip23,
    Flip32,
    Flip44,
}

/// The repair of a bad tet, planned without changing anything.
#[derive(Clone)]
enum Plan {
    /// Nothing to do (the tet is gone or good by now).
    Skip,
    /// The tets `old` replaced by `new` with their qualities.
    Replace {
        old: SmallVec<[u32; 4]>,
        new: SmallVec<[([u32; 4], f64); 4]>,
        kind: Op,
    },
    /// Vertex `v` moved to `x`, with the new qualities of its star.
    Move {
        v: u32,
        x: P3,
        star: Vec<(u32, f64)>,
    },
    /// A peel may apply: planned when applied, peel first.
    Peel,
    /// Nothing helps.
    Fail,
}

/// How a vertex may move.
#[derive(Clone, Copy, PartialEq)]
enum Mobility {
    Fixed,
    /// A volume vertex on no face.
    Free,
    /// A vertex on one patch: moves along its carrier.
    Surface,
    /// A vertex inside a smooth curve: moves along it.
    Curve,
}

struct Improver<'a> {
    c: &'a mut Complex,
    shape: Option<&'a dyn Shape>,
    /// Faces through each vertex (for the normal check of surface moves).
    vfaces: Vec<SmallVec<[u32; 8]>>,
    alive: Vec<bool>,
    q: Vec<f64>,
    vtets: Vec<SmallVec<[u32; 32]>>,
    /// Face index by sorted vertex triple (live faces only).
    faces: FxHashMap<[u32; 3], u32>,
    face_alive: Vec<bool>,
    features: FxHashSet<(u32, u32)>,
    /// The neighbours of each vertex inside a curve along it.
    along: FxHashMap<u32, SmallVec<[u32; 2]>>,
    /// The parameters of each patch vertex on a spline carrier: its
    /// projections search from there, not over the whole carrier.
    uv: Vec<Option<[f64; 2]>>,
    mobility: Vec<Mobility>,
    /// Patches whose faces stay as they are (periodic pairs).
    frozen: Vec<u32>,
    /// The sweep (from 1) in which each vertex last saw a change.
    changed: Vec<u32>,
    /// The vertices changed since the last sweep began (with repeats).
    touched: Vec<u32>,
    /// The batch in which each vertex last saw a change in its star, and
    /// the current batch (see [`Improver::repair`]).
    dirty: Vec<u32>,
    batch: u32,
    /// A counter of changes, the last one in each vertex's star, and the
    /// count at which each vertex last failed to move (0 never).
    event: u64,
    star_event: Vec<u64>,
    failed_at: Vec<u64>,
    batch_size: usize,
    /// The sweep that last took each tet as a candidate.
    seen: Vec<u32>,
    /// Dead tets whose places a new one takes.
    free: Vec<u32>,
    sweep: u32,
    stats_flips23: usize,
    stats_flips32: usize,
    stats_flips44: usize,
    stats_moves: usize,
}

impl Improver<'_> {
    fn p(&self, v: u32) -> P3 {
        self.c.points[v as usize]
    }

    fn quality(&self, t: [u32; 4]) -> f64 {
        tet_min_dihedral(t.map(|v| self.p(v)))
    }

    /// The tet's vertices in positive order, `None` if flat.
    fn positive(&self, t: [u32; 4]) -> Option<[u32; 4]> {
        let o = orient3d(self.p(t[0]), self.p(t[1]), self.p(t[2]), self.p(t[3]));
        if o > 0.0 {
            Some(t)
        } else if o < 0.0 {
            Some([t[0], t[1], t[3], t[2]])
        } else {
            None
        }
    }

    /// True if the segment `a b` passes through the interior of triangle
    /// `f`: the geometric condition of a valid 2-3 or 3-2 flip. With
    /// `on_edge` also through the inside of one of its edges: around an
    /// edge `a b` that is the case exactly when one of the three tets is
    /// flat (four corners on a plane, as the cocircular corners of a small
    /// face leave them), and the two new tets still fill the other two.
    /// The line through `a b` meeting `f` is not enough: three tets around
    /// `a b` with `a` and `b` on one side of `f` are the star of `b` inside
    /// the tet `a f` less the tet `b f`, which the flip would fold over.
    fn crosses(&self, a: u32, b: u32, f: [u32; 3], on_edge: bool) -> bool {
        let side = |x: u32| orient3d(self.p(f[0]), self.p(f[1]), self.p(f[2]), self.p(x));
        let (sa, sb) = (side(a), side(b));
        if !((sa > 0.0 && sb < 0.0) || (sa < 0.0 && sb > 0.0)) {
            return false;
        }
        let s: [f64; 3] = std::array::from_fn(|k| {
            orient3d(self.p(f[k]), self.p(f[(k + 1) % 3]), self.p(a), self.p(b))
        });
        let zeros = s.iter().filter(|&&x| x == 0.0).count();
        if zeros > usize::from(on_edge) {
            return false;
        }
        s.iter().all(|&x| x >= 0.0) || s.iter().all(|&x| x <= 0.0)
    }

    /// The vertex of `f` to search its tets from: the first, unless its
    /// star is large and another's smaller (a fan over a degenerate spot
    /// can hold a sizable part of the mesh).
    fn hub(&self, f: &[u32]) -> u32 {
        let len = |v: u32| self.vtets[v as usize].len();
        if len(f[0]) <= LARGE_STAR {
            return f[0];
        }
        f.iter().copied().min_by_key(|&v| len(v)).unwrap_or(f[0])
    }

    /// The live tets around the edge `(a, b)`.
    fn around(&self, a: u32, b: u32) -> impl Iterator<Item = u32> + '_ {
        let (from, to) = if self.hub(&[a, b]) == a {
            (a, b)
        } else {
            (b, a)
        };
        self.live(from)
            .filter(move |&u| self.c.tets[u as usize].contains(&to))
    }

    /// The live tet other than `t` on the triangle `f` of `t`.
    fn across(&self, t: u32, f: [u32; 3]) -> Option<u32> {
        self.live(self.hub(&f))
            .find(|&u| u != t && f.iter().all(|v| self.c.tets[u as usize].contains(v)))
    }

    fn live(&self, v: u32) -> impl Iterator<Item = u32> + '_ {
        self.vtets[v as usize]
            .iter()
            .copied()
            .filter(|&t| self.alive[t as usize])
    }

    /// Adds the tet `t` in the place of a dead one where there is one: the
    /// flips make and kill tets all the time, and the arrays stay as long
    /// as the mesh was.
    fn add(&mut self, t: [u32; 4], q: f64, region: u32) {
        let id = match self.free.pop() {
            Some(id) => {
                let i = id as usize;
                self.c.tets[i] = t;
                self.c.regions[i] = region;
                self.alive[i] = true;
                self.q[i] = q;
                id
            }
            None => {
                self.c.tets.push(t);
                self.c.regions.push(region);
                self.alive.push(true);
                self.q.push(q);
                self.c.tets.len() as u32 - 1
            }
        };
        for v in t {
            self.vtets[v as usize].push(id);
            self.touch(v);
        }
    }

    /// Marks `v` changed in this sweep and batch.
    fn touch(&mut self, v: u32) {
        self.changed[v as usize] = self.sweep;
        self.mark_star(v);
        self.touched.push(v);
    }

    /// Notes a change in the star of `w`: in this batch, and as the latest
    /// event, after which a failed move of `w` is worth trying again.
    fn mark_star(&mut self, w: u32) {
        self.dirty[w as usize] = self.batch;
        self.event += 1;
        self.star_event[w as usize] = self.event;
    }

    /// Kills the tet `t`: out of its corners' stars, its place free.
    fn kill(&mut self, t: u32) {
        self.alive[t as usize] = false;
        for v in self.c.tets[t as usize] {
            let star = &mut self.vtets[v as usize];
            if let Some(k) = star.iter().position(|&x| x == t) {
                star.swap_remove(k);
            }
        }
        self.free.push(t);
    }

    /// The replacement of the tets `old` by `new` (same region) when every
    /// new tet is positive and the worst dihedral rises.
    fn plan_replace(&self, old: &[u32], new: &[[u32; 4]], kind: Op) -> Option<Plan> {
        let before = old
            .iter()
            .map(|&t| self.q[t as usize])
            .fold(f64::INFINITY, f64::min);
        let mut fixed: SmallVec<[([u32; 4], f64); 4]> = SmallVec::new();
        let mut after = f64::INFINITY;
        for &t in new {
            let t = self.positive(t)?;
            let q = self.quality(t);
            after = after.min(q);
            fixed.push((t, q));
        }
        if !(after > before + GAIN) {
            return None;
        }
        Some(Plan::Replace {
            old: old.iter().copied().collect(),
            new: fixed,
            kind,
        })
    }

    fn apply_replace(&mut self, old: &[u32], new: &[([u32; 4], f64)], kind: Op) {
        let region = self.c.regions[old[0] as usize];
        for &t in old {
            self.kill(t);
        }
        for &(t, q) in new {
            self.add(t, q, region);
        }
        match kind {
            Op::Flip23 => self.stats_flips23 += 1,
            Op::Flip32 => self.stats_flips32 += 1,
            Op::Flip44 => self.stats_flips44 += 1,
        }
    }

    /// 2-3 flip across the facet of `t` opposite its vertex `i`.
    fn flip23(&self, t: u32, i: usize) -> Option<Plan> {
        let tv = self.c.tets[t as usize];
        let f = [tv[(i + 1) % 4], tv[(i + 2) % 4], tv[(i + 3) % 4]];
        if self.faces.contains_key(&sorted3(f)) {
            return None;
        }
        let u = self.across(t, f)?;
        if self.c.regions[u as usize] != self.c.regions[t as usize] {
            return None;
        }
        let a = tv[i];
        let &b = self.c.tets[u as usize].iter().find(|v| !f.contains(v))?;
        if !self.crosses(a, b, f, false) {
            return None;
        }
        let new = [[f[0], f[1], a, b], [f[1], f[2], a, b], [f[2], f[0], a, b]];
        self.plan_replace(&[t, u], &new, Op::Flip23)
    }

    /// 3-2 flip removing the edge `(a, b)` of `t`, if exactly three tets of
    /// one region surround it and none of their shared triangles is a face.
    fn flip32(&self, t: u32, a: u32, b: u32) -> Option<Plan> {
        if self.features.contains(&(a.min(b), a.max(b))) {
            return None;
        }
        let ring: Vec<u32> = self.around(a, b).collect();
        if ring.len() != 3 || !ring.contains(&t) {
            return None;
        }
        let region = self.c.regions[t as usize];
        if ring.iter().any(|&u| self.c.regions[u as usize] != region) {
            return None;
        }
        let mut others: Vec<u32> = Vec::with_capacity(3);
        for &u in &ring {
            for v in self.c.tets[u as usize] {
                if v != a && v != b && !others.contains(&v) {
                    others.push(v);
                }
            }
        }
        if others.len() != 3 {
            return None;
        }
        for &x in &others {
            if self.faces.contains_key(&sorted3([a, b, x])) {
                return None;
            }
        }
        let [p, q, r] = [others[0], others[1], others[2]];
        if !self.crosses(a, b, [p, q, r], true) {
            return None;
        }
        self.plan_replace(&ring, &[[p, q, r, a], [p, q, r, b]], Op::Flip32)
    }

    /// 4-4 flip exchanging the edge `(a, b)` of `t` for a diagonal of the
    /// quadrilateral around it, if exactly four tets of one region
    /// surround it and none of their shared triangles is a face: a 2-3
    /// flip creating the diagonal, then a 3-2 flip removing `(a, b)`. It
    /// removes a flat tet on four coplanar corners inside a region, which
    /// the 3-2 flip cannot reach.
    fn flip44(&self, t: u32, a: u32, b: u32) -> Option<Plan> {
        if self.features.contains(&(a.min(b), a.max(b))) {
            return None;
        }
        let ring: SmallVec<[u32; 4]> = self.around(a, b).collect();
        if ring.len() != 4 || !ring.contains(&t) {
            return None;
        }
        let region = self.c.regions[t as usize];
        if ring.iter().any(|&u| self.c.regions[u as usize] != region) {
            return None;
        }
        // The two corners of each ring tet off the edge, chained into the
        // quadrilateral p0 p1 p2 p3 around it.
        let pair = |u: u32| -> [u32; 2] {
            let mut o = self.c.tets[u as usize]
                .into_iter()
                .filter(|&v| v != a && v != b);
            [o.next().unwrap_or(a), o.next().unwrap_or(a)]
        };
        let pairs: SmallVec<[[u32; 2]; 4]> = ring.iter().map(|&u| pair(u)).collect();
        let mut quad = pairs[0].to_vec();
        let mut used = [true, false, false, false];
        while quad.len() < 4 {
            let last = *quad.last()?;
            let k = (0..4).find(|&k| !used[k] && pairs[k].contains(&last))?;
            used[k] = true;
            quad.push(if pairs[k][0] == last {
                pairs[k][1]
            } else {
                pairs[k][0]
            });
        }
        let closes =
            (0..4).any(|k| !used[k] && pairs[k].contains(&quad[3]) && pairs[k].contains(&quad[0]));
        if !closes || quad.iter().any(|&v| v == a || v == b) {
            return None;
        }
        for &x in &quad {
            if self.faces.contains_key(&sorted3([a, b, x])) {
                return None;
            }
        }
        for s in 0..2 {
            let [p0, p1, p2, p3] = [quad[s], quad[s + 1], quad[(s + 2) % 4], quad[(s + 3) % 4]];
            if self.faces.contains_key(&sorted3([p0, p2, p1]))
                || self.faces.contains_key(&sorted3([p0, p2, p3]))
            {
                continue;
            }
            // The 2-3 flip across (a, b, p1) creates p0 p2; the 3-2 flip
            // around (a, b) then needs a b through (p0, p2, p3).
            if !self.crosses(p0, p2, [a, b, p1], true) || !self.crosses(a, b, [p0, p2, p3], true) {
                continue;
            }
            let new = [
                [p0, p2, p1, a],
                [p0, p2, p1, b],
                [p0, p2, p3, a],
                [p0, p2, p3, b],
            ];
            if let Some(plan) = self.plan_replace(&ring, &new, Op::Flip44) {
                return Some(plan);
            }
        }
        None
    }

    /// Whether `t` has the two faces a peel needs (and there is a shape).
    fn may_peel(&self, t: u32) -> bool {
        if self.shape.is_none() {
            return false;
        }
        let tv = self.c.tets[t as usize];
        (0..4)
            .filter(|&i| {
                let f = [tv[(i + 1) % 4], tv[(i + 2) % 4], tv[(i + 3) % 4]];
                self.faces.contains_key(&sorted3(f))
            })
            .count()
            == 2
    }

    /// Peels a flat tet `abcd` off a boundary when `abc` and `abd` are its
    /// only faces, on one patch with the same region `s` behind both, and
    /// `ab` is no feature edge, and neither new face bridges the patch's
    /// carrier: `acd` and `bcd` become the faces instead,
    /// the surface edge `ab` flipping to `cd`. On the outer boundary
    /// (`s = 0`) the tet goes; on an interface it joins `s`, where its old
    /// faces are interior and ordinary flips can remove it. Only with a
    /// shape (the boundary already approximates it), since the regions
    /// trade the tet's volume.
    fn peel(&mut self, t: u32) -> bool {
        if !self.may_peel(t) {
            return false;
        }
        let tv = self.c.tets[t as usize];
        let region = self.c.regions[t as usize];
        // (vertex opposite, face index) of the tet's faces.
        let mut on: [(u32, u32); 2] = [(0, 0); 2];
        let mut n = 0;
        for i in 0..4 {
            let f = [tv[(i + 1) % 4], tv[(i + 2) % 4], tv[(i + 3) % 4]];
            if let Some(&fi) = self.faces.get(&sorted3(f)) {
                if n == 2 {
                    return false;
                }
                on[n] = (tv[i], fi);
                n += 1;
            }
        }
        if n != 2 {
            return false;
        }
        let (f0, f1) = (
            self.c.faces[on[0].1 as usize],
            self.c.faces[on[1].1 as usize],
        );
        // The region behind a face of `region` (none for a sheet).
        let behind = |f: &Face| match f.regions {
            [x, y] if x == region && y != region => Some(y),
            [x, y] if y == region && x != region => Some(x),
            _ => None,
        };
        let Some(s) = behind(&f0) else {
            return false;
        };
        if f0.patch != f1.patch
            || f0.patch == u32::MAX
            || behind(&f1) != Some(s)
            || self.frozen.contains(&f0.patch)
        {
            return false;
        }
        let (c, d) = (on[0].0, on[1].0);
        let mut ab = tv.iter().copied().filter(|&v| v != c && v != d);
        let (Some(a), Some(b)) = (ab.next(), ab.next()) else {
            return false;
        };
        if self.features.contains(&(a.min(b), a.max(b))) {
            return false;
        }
        // `cd` must be new to the surface, or it would carry four faces.
        let on_surface =
            |f: &u32| self.face_alive[*f as usize] && self.c.faces[*f as usize].tri.contains(&d);
        if self.vfaces[c as usize].iter().any(on_surface) {
            return false;
        }
        let normal = |tri: [u32; 3]| {
            let (p, q, r) = (self.p(tri[0]), self.p(tri[1]), self.p(tri[2]));
            cross(sub(q, p), sub(r, p))
        };
        // Each new face (without `away`) is wound like the old ones relative
        // to the mesh: `f0` has the mesh on the side of `c`, the tet's
        // vertex off it; a new face has it opposite `away`. It must also
        // face the same way as `f0`, or the surface would fold.
        let old = normal(f0.tri);
        let mesh_old = dot(old, sub(self.p(c), self.p(a))) > 0.0;
        let mut new = [f0; 2];
        for (k, (away, keep)) in [(b, a), (a, b)].into_iter().enumerate() {
            let mut tri = [keep, c, d];
            let mesh_new = dot(normal(tri), sub(self.p(away), self.p(keep))) < 0.0;
            if mesh_new != mesh_old {
                tri.swap(1, 2);
            }
            if !(dot(normal(tri), old) > 0.0) {
                return false;
            }
            new[k].tri = tri;
        }
        // No new face bridges the carrier (a chord across a sharp tip).
        let corners = |tri: [u32; 3]| tri.map(|w| self.p(w));
        if new
            .iter()
            .zip([f0, f1])
            .any(|(f, o)| self.bridge(f.tri, corners(f.tri), corners(o.tri), f0.patch))
        {
            return false;
        }
        if s == 0 {
            self.kill(t);
        } else {
            self.c.regions[t as usize] = s;
        }
        for (_, fi) in on {
            self.face_alive[fi as usize] = false;
            self.faces.remove(&sorted3(self.c.faces[fi as usize].tri));
        }
        for f in new {
            let id = self.c.faces.len() as u32;
            self.faces.insert(sorted3(f.tri), id);
            self.face_alive.push(true);
            for v in f.tri {
                self.vfaces[v as usize].push(id);
            }
            self.c.faces.push(f);
        }
        // Across an interface the peel only pays when the tet, now inside
        // `s`, flips away right there; otherwise everything is undone.
        if s != 0 && !self.flip_any(t) {
            self.c.regions[t as usize] = region;
            let added = self.c.faces.len() - 2;
            for fi in added..added + 2 {
                self.face_alive[fi] = false;
                self.faces.remove(&sorted3(self.c.faces[fi].tri));
            }
            for (_, fi) in on {
                self.face_alive[fi as usize] = true;
                self.faces
                    .insert(sorted3(self.c.faces[fi as usize].tri), fi);
            }
            return false;
        }
        for v in tv {
            self.touch(v);
        }
        true
    }

    /// A 2-3 flip across a facet of `t`, else a 3-2 flip removing an edge
    /// of it, whichever raises the local worst dihedral first.
    fn plan_flip(&self, t: u32) -> Option<Plan> {
        let tv = self.c.tets[t as usize];
        if let Some(p) = (0..4).find_map(|i| self.flip23(t, i)) {
            return Some(p);
        }
        for a in 0..4 {
            for b in a + 1..4 {
                if let Some(p) = self.flip32(t, tv[a], tv[b]) {
                    return Some(p);
                }
            }
        }
        for a in 0..4 {
            for b in a + 1..4 {
                if let Some(p) = self.flip44(t, tv[a], tv[b]) {
                    return Some(p);
                }
            }
        }
        None
    }

    /// The repair of the bad tet `t`: a flip, else a peel, else the move of
    /// one of its vertices that leaves the best star (see [`best_move`]).
    fn plan(&self, t: u32, target_deg: f64) -> Plan {
        if !self.alive[t as usize] || self.q[t as usize] >= target_deg {
            return Plan::Skip;
        }
        if let Some(p) = self.plan_flip(t) {
            return p;
        }
        if self.may_peel(t) {
            return Plan::Peel;
        }
        best_move(self.c.tets[t as usize].map(|v| self.plan_move(v)))
    }

    /// True when no vertex move of `v` helped at its last try and its star
    /// has not changed since: the try would fail the same way.
    fn failed_still(&self, v: u32) -> bool {
        let f = self.failed_at[v as usize];
        f != 0 && f >= self.star_event[v as usize]
    }

    /// [`Improver::plan`] for every tet of `chunk` at once, in parallel, the
    /// move of a vertex shared by several of them planned once.
    fn plan_batch(&self, chunk: &[u32], target_deg: f64) -> Vec<Plan> {
        use rayon::prelude::*;
        // Everything but the moves, which are `None` here.
        let first: Vec<Option<Plan>> = chunk
            .par_iter()
            .map(|&t| {
                if !self.alive[t as usize] || self.q[t as usize] >= target_deg {
                    return Some(Plan::Skip);
                }
                if let Some(p) = self.plan_flip(t) {
                    return Some(p);
                }
                self.may_peel(t).then_some(Plan::Peel)
            })
            .collect();
        let mut verts: Vec<u32> = chunk
            .iter()
            .zip(&first)
            .filter(|(_, p)| p.is_none())
            .flat_map(|(&t, _)| self.c.tets[t as usize])
            .collect();
        verts.sort_unstable();
        verts.dedup();
        let moves: FxHashMap<u32, Plan> = verts
            .par_iter()
            .filter_map(|&v| self.plan_move(v).map(|p| (v, p)))
            .collect();
        chunk
            .iter()
            .zip(first)
            .map(|(&t, p)| {
                p.unwrap_or_else(|| {
                    best_move(self.c.tets[t as usize].map(|v| moves.get(&v).cloned()))
                })
            })
            .collect()
    }

    /// Carries out the plan for `t`; true if something changed.
    fn apply(&mut self, t: u32, plan: Plan) -> bool {
        match plan {
            Plan::Skip => false,
            Plan::Fail => {
                // Planned on the stars as they are now: no move of a vertex
                // of `t` helps until one of those stars changes.
                self.event += 1;
                for v in self.c.tets[t as usize] {
                    self.failed_at[v as usize] = self.event;
                }
                false
            }
            Plan::Replace { old, new, kind } => {
                self.apply_replace(&old, &new, kind);
                true
            }
            Plan::Move { v, x, star } => {
                self.c.points[v as usize] = x;
                self.touch(v);
                for (t, q) in star {
                    self.q[t as usize] = q;
                    for w in self.c.tets[t as usize] {
                        self.mark_star(w);
                    }
                }
                self.stats_moves += 1;
                true
            }
            Plan::Peel => {
                let tv = self.c.tets[t as usize];
                self.peel(t) || tv.iter().any(|&v| self.smooth_vertex(v))
            }
        }
    }

    fn flip_any(&mut self, t: u32) -> bool {
        self.plan_flip(t).is_some_and(|p| self.apply(t, p))
    }

    /// Where a vertex would land for the candidate `x`: `x` for a free
    /// vertex, its projection onto the carrier for a surface vertex.
    fn place(&self, v: u32, x: P3) -> Option<P3> {
        match self.mobility[v as usize] {
            Mobility::Free => Some(x),
            Mobility::Surface => self.project_from(self.c.classes[v as usize], x, v),
            Mobility::Curve => self.shape?.project(self.c.classes[v as usize], x),
            Mobility::Fixed => None,
        }
    }

    /// [`Improver::place`] near enough to compare candidates (see
    /// [`Shape::project_near`] and [`Shape::project_near_from`]): still a
    /// point on the curve or carrier, so the chosen one stays as it is.
    fn place_near(&self, v: u32, x: P3) -> Option<P3> {
        let kind = self.c.classes[v as usize];
        match (self.mobility[v as usize], self.uv[v as usize]) {
            (Mobility::Curve, _) => self.shape?.project_near(kind, x),
            (Mobility::Surface, Some(uv)) => self.shape?.project_near_from(kind, x, uv),
            _ => self.place(v, x),
        }
    }

    /// Worst dihedral of the star of `v` with `v` at `x` if it beats
    /// `floor`; `None` when it does not, when a tet would not stay positive
    /// or when a face through `v` would turn. Stops at the first tet at or
    /// below `floor`, so a star sorted worst first rejects most candidates
    /// after a tet or two.
    fn star_quality(&self, v: u32, star: &[u32], x: P3, floor: f64) -> Option<f64> {
        let mut worst = f64::INFINITY;
        for &t in star {
            let tv = self.c.tets[t as usize];
            let pts = tv.map(|w| if w == v { x } else { self.p(w) });
            if !(orient3d(pts[0], pts[1], pts[2], pts[3]) > 0.0) {
                return None;
            }
            let q = tet_min_dihedral(pts);
            if q <= floor {
                return None;
            }
            worst = worst.min(q);
        }
        for &f in &self.vfaces[v as usize] {
            if !self.face_alive[f as usize] {
                continue;
            }
            let tri = self.c.faces[f as usize].tri;
            let at = |w: u32, moved: bool| if moved && w == v { x } else { self.p(w) };
            let n = |moved: bool| {
                let (a, b, c) = (at(tri[0], moved), at(tri[1], moved), at(tri[2], moved));
                let (u, w) = (sub(b, a), sub(c, a));
                cross(u, w)
            };
            let (n0, n1) = (n(false), n(true));
            if !(dot(n0, n1) > 0.0) {
                return None;
            }
        }
        Some(worst)
    }

    /// Whether `v` may go to `x`: every tet of its star stays positive, none
    /// at or above `target` drops below it and none below gets worse, their
    /// dihedrals sum to at least what they did (the mean does not fall), and
    /// no face through `v` turns.
    fn star_accepts(&self, v: u32, star: &[u32], x: P3, target: f64) -> bool {
        let (mut sum, mut sum0) = (0.0, 0.0);
        for &t in star {
            let tv = self.c.tets[t as usize];
            let pts = tv.map(|w| if w == v { x } else { self.p(w) });
            if !(orient3d(pts[0], pts[1], pts[2], pts[3]) > 0.0) {
                return false;
            }
            let (q, q0) = (tet_min_dihedral(pts), self.q[t as usize]);
            if q < q0.min(target) {
                return false;
            }
            sum += q;
            sum0 += q0;
        }
        sum >= sum0 && self.star_quality(v, &[], x, 0.0).is_some()
    }

    /// `x` onto the carrier of `kind`, searched from the parameters of
    /// vertex `near` where it has some there.
    fn project_from(&self, kind: PointClass, x: P3, near: u32) -> Option<P3> {
        let shape = self.shape?;
        match self.uv[near as usize] {
            Some(uv) if self.c.classes[near as usize] == kind => {
                shape.project_from(kind, x, uv).map(|r| r.0)
            }
            _ => shape.project(kind, x),
        }
    }

    /// `x` onto `patch`, searched from a corner of `tri` on it.
    fn project_on_patch(&self, tri: [u32; 3], x: P3, patch: u32) -> Option<P3> {
        let kind = PointClass::Face(patch);
        match tri
            .into_iter()
            .find(|&w| self.c.classes[w as usize] == kind)
        {
            Some(w) => self.project_from(kind, x, w),
            None => self.shape?.project(kind, x),
        }
    }

    fn commit_move(&mut self, v: u32, star: &[u32], x: P3) {
        if let (Some(shape), Some(uv)) = (self.shape, self.uv[v as usize]) {
            self.uv[v as usize] = shape
                .project_from(self.c.classes[v as usize], x, uv)
                .map(|r| r.1);
        }
        self.c.points[v as usize] = x;
        self.touch(v);
        for &t in star {
            let tv = self.c.tets[t as usize];
            self.q[t as usize] = self.quality(tv);
            for w in tv {
                self.mark_star(w);
            }
        }
    }

    /// The move of a vertex of a bad tet to the best nearby candidate, if
    /// its star's worst dihedral rises.
    fn plan_move(&self, v: u32) -> Option<Plan> {
        if self.mobility[v as usize] == Mobility::Fixed || self.failed_still(v) {
            return None;
        }
        if self.live(v).nth(MOVE_STAR_MAX).is_some() {
            return None;
        }
        let mut star: Vec<u32> = self.live(v).collect();
        if star.is_empty() {
            return None;
        }
        star.sort_by(|&a, &b| self.q[a as usize].total_cmp(&self.q[b as usize]));
        let before = self.q[star[0] as usize];
        let p0 = self.p(v);
        let lmin = self.shortest_edge(v, &star);
        // Along a curve only towards its neighbours on it, at the same
        // fractions of the shortest edge.
        let dirs: SmallVec<[P3; 12]> = match self.mobility[v as usize] {
            Mobility::Curve => self.along.get(&v).map_or(SmallVec::new(), |ns| {
                ns.iter()
                    .map(|&w| {
                        let d = sub(self.p(w), p0);
                        let l = dist(self.p(w), p0).max(f64::MIN_POSITIVE);
                        d.map(|x| x / l)
                    })
                    .collect()
            }),
            _ => DIRS.into_iter().collect(),
        };
        let mut best = (before, p0);
        for s in STEPS {
            for &d in &dirs {
                let Some(x) = self.place_near(v, std::array::from_fn(|k| p0[k] + s * lmin * d[k]))
                else {
                    continue;
                };
                if let Some(q) = self.star_quality(v, &star, x, best.0) {
                    best = (q, x);
                }
            }
            if best.0 > before + GAIN {
                break;
            }
        }
        if !(best.0 > before + GAIN) {
            return None;
        }
        let mut x = best.1;
        if self.mobility[v as usize] == Mobility::Curve {
            // The chosen place exactly on the curve, still a gain there.
            x = self.place(v, x)?;
            self.star_quality(v, &star, x, before + GAIN)?;
        }
        if self.mobility[v as usize] != Mobility::Free && self.strays(v, x) {
            return None;
        }
        let star = star
            .into_iter()
            .map(|t| {
                let tv = self.c.tets[t as usize];
                (
                    t,
                    tet_min_dihedral(tv.map(|w| if w == v { x } else { self.p(w) })),
                )
            })
            .collect();
        Some(Plan::Move { v, x, star })
    }

    /// Whether moving the surface vertex `v` to `x` makes a face through it
    /// a bridge (see [`Improver::bridge`]).
    fn strays(&self, v: u32, x: P3) -> bool {
        self.vfaces[v as usize]
            .iter()
            .filter(|&&f| self.face_alive[f as usize])
            .any(|&f| {
                let face = &self.c.faces[f as usize];
                let at = |w: u32| if w == v { x } else { self.p(w) };
                let old = face.tri.map(|w| self.p(w));
                self.bridge(face.tri, face.tri.map(at), old, face.patch)
            })
    }

    /// Whether the triangle `new` on `patch` strays from its carrier by
    /// more than [`BRIDGE`] of its longest edge where `old` did not: a
    /// chord across a sharp tip, which the diagnostics call a bridge.
    fn bridge(&self, tri: [u32; 3], new: [P3; 3], old: [P3; 3], patch: u32) -> bool {
        if self.shape.is_none() {
            return false;
        }
        let stray = |t: [P3; 3]| -> f64 {
            let m: P3 = std::array::from_fn(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0);
            let off = self
                .project_on_patch(tri, m, patch)
                .map_or(0.0, |y| dist(m, y));
            let longest = (0..3)
                .map(|k| dist(t[k], t[(k + 1) % 3]))
                .fold(0.0, f64::max);
            off / longest.max(f64::MIN_POSITIVE)
        };
        stray(new) > BRIDGE && stray(new) > stray(old)
    }

    fn smooth_vertex(&mut self, v: u32) -> bool {
        self.plan_move(v).is_some_and(|p| self.apply(v, p))
    }

    fn shortest_edge(&self, v: u32, star: &[u32]) -> f64 {
        let p0 = self.p(v);
        let mut lmin = f64::INFINITY;
        for &t in star {
            for w in self.c.tets[t as usize] {
                if w != v {
                    let d = sub(self.p(w), p0);
                    lmin = lmin.min(len(d));
                }
            }
        }
        lmin
    }
}

impl<'a> Improver<'a> {
    /// Indexes `c` for local changes. With a `shape`, vertices on one patch
    /// move along its carrier; without one only volume vertices move.
    fn new(c: &'a mut Complex, shape: Option<&'a dyn Shape>, frozen: &[u32]) -> Improver<'a> {
        let n = c.points.len();
        let mut vtets: Vec<SmallVec<[u32; 32]>> = vec![SmallVec::new(); n];
        for (ti, t) in c.tets.iter().enumerate() {
            for &v in t {
                vtets[v as usize].push(ti as u32);
            }
        }
        let faces: FxHashMap<[u32; 3], u32> = c
            .faces
            .iter()
            .enumerate()
            .map(|(i, f)| (sorted3(f.tri), i as u32))
            .collect();
        let features = c
            .edges
            .iter()
            .map(|(e, _)| (e[0].min(e[1]), e[0].max(e[1])))
            .collect();
        let mut along: FxHashMap<u32, SmallVec<[u32; 2]>> = FxHashMap::default();
        for &([a, b], curve) in &c.edges {
            for (u, w) in [(a, b), (b, a)] {
                if c.classes[u as usize] == PointClass::Edge(curve) {
                    along.entry(u).or_default().push(w);
                }
            }
        }
        let mut vfaces: Vec<SmallVec<[u32; 8]>> = vec![SmallVec::new(); n];
        for (fi, f) in c.faces.iter().enumerate() {
            for &v in &f.tri {
                vfaces[v as usize].push(fi as u32);
            }
        }
        let on_frozen = |v: usize| {
            vfaces[v]
                .iter()
                .any(|&f| frozen.contains(&c.faces[f as usize].patch))
        };
        let mobility: Vec<Mobility> = (0..n)
            .map(|v| match c.classes[v] {
                _ if on_frozen(v) => Mobility::Fixed,
                PointClass::Interior if vfaces[v].is_empty() => Mobility::Free,
                PointClass::Face(p)
                    if shape.is_some()
                        && vfaces[v].iter().all(|&f| c.faces[f as usize].patch == p) =>
                {
                    Mobility::Surface
                }
                PointClass::Edge(_) if shape.is_some_and(|s| s.smooth(c.classes[v])) => {
                    Mobility::Curve
                }
                _ => Mobility::Fixed,
            })
            .collect();
        // One search over the whole carrier per patch vertex; every later
        // projection searches from where the vertex is.
        let uv: Vec<Option<[f64; 2]>> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(|v| match (shape, c.classes[v]) {
                    (Some(s), k @ PointClass::Face(_)) => s.param(k, c.points[v]),
                    _ => None,
                })
                .collect()
        };
        let q: Vec<f64> = {
            use rayon::prelude::*;
            let pts = &c.points;
            c.tets
                .par_iter()
                .map(|t| tet_min_dihedral(t.map(|v| pts[v as usize])))
                .collect()
        };
        Improver {
            alive: vec![true; c.tets.len()],
            q,
            vtets,
            face_alive: vec![true; c.faces.len()],
            faces,
            features,
            along,
            uv,
            mobility,
            frozen: frozen.to_vec(),
            vfaces,
            shape,
            c,
            changed: vec![0; n],
            touched: Vec::new(),
            dirty: vec![0; n],
            batch: 0,
            event: 0,
            star_event: vec![0; n],
            failed_at: vec![0; n],
            batch_size: BATCH_START,
            seen: Vec::new(),
            free: Vec::new(),
            sweep: 0,
            stats_flips23: 0,
            stats_flips32: 0,
            stats_flips44: 0,
            stats_moves: 0,
        }
    }

    fn bad(&self, target_deg: f64) -> usize {
        (0..self.q.len())
            .filter(|&t| self.alive[t] && self.q[t] < target_deg)
            .count()
    }

    /// Repairs the tets below `target_deg` over at most `passes` sweeps,
    /// worst first. With `all` the first sweep takes every bad tet; every
    /// other sweep only those near a change since the sweep before (the
    /// others would fail again with the same neighbourhood).
    fn repair(&mut self, target_deg: f64, passes: usize, all: bool) {
        let start = self.sweep + 1;
        for sweep in start..start + passes as u32 {
            self.sweep = sweep;
            // The first sweep with `all` takes every tet, the others the
            // tets around a vertex changed since the sweep before.
            let mut bad: Vec<u32> = if all && sweep == start {
                self.touched.clear();
                (0..self.c.tets.len() as u32).collect()
            } else {
                let mut near: Vec<u32> = Vec::new();
                self.seen.resize(self.c.tets.len(), 0);
                let mut touched = std::mem::take(&mut self.touched);
                touched.sort_unstable();
                touched.dedup();
                for v in touched {
                    if self.changed[v as usize] + 1 < sweep {
                        continue;
                    }
                    for &t in &self.vtets[v as usize] {
                        let i = t as usize;
                        if self.seen[i] != sweep && self.alive[i] && self.q[i] < target_deg {
                            self.seen[i] = sweep;
                            near.push(t);
                        }
                    }
                }
                near.sort_unstable();
                near
            };
            bad.retain(|&t| self.alive[t as usize] && self.q[t as usize] < target_deg);
            if bad.is_empty() {
                break;
            }
            bad.sort_by(|&a, &b| self.q[a as usize].total_cmp(&self.q[b as usize]));
            let mut changed = false;
            let mut at = 0;
            while at < bad.len() {
                let chunk = &bad[at..(at + self.batch_size).min(bad.len())];
                at += chunk.len();
                let plans = self.plan_batch(chunk, target_deg);
                self.batch += 1;
                // Plans made again because a change before them in the batch
                // touched their neighbourhood.
                let mut conflicts = 0;
                for (&t, plan) in chunk.iter().zip(plans) {
                    let fresh = self.c.tets[t as usize]
                        .iter()
                        .all(|&v| self.dirty[v as usize] != self.batch);
                    let plan = if fresh {
                        plan
                    } else {
                        conflicts += 1;
                        self.plan(t, target_deg)
                    };
                    changed |= self.apply(t, plan);
                }
                // Clustered bad tets conflict: smaller batches then waste
                // fewer plans, scattered ones afford larger batches.
                if 4 * conflicts > chunk.len() {
                    self.batch_size = (self.batch_size / 2).max(BATCH_MIN);
                } else if 16 * conflicts < chunk.len() {
                    self.batch_size = (2 * self.batch_size).min(BATCH_MAX);
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// Where vertex `v` would go when relaxed: on one patch toward the
    /// area-weighted centroid of its faces, inside a curve toward the
    /// midpoint of its two neighbours along it, onto the carrier
    /// or the curve; taken where the smallest angle of the faces around it
    /// rises and its star accepts (see [`Improver::star_accepts`]). The full
    /// step, else a half or a quarter of it.
    fn plan_relax(&self, v: u32, target: f64) -> Option<P3> {
        let shape = self.shape?;
        let kind = self.c.classes[v as usize];
        let faces: SmallVec<[[u32; 3]; 8]> = self.vfaces[v as usize]
            .iter()
            .filter(|&&f| self.face_alive[f as usize])
            .map(|&f| self.c.faces[f as usize].tri)
            .collect();
        let on_frozen = self.vfaces[v as usize]
            .iter()
            .any(|&f| self.frozen.contains(&self.c.faces[f as usize].patch));
        if faces.len() < 3 || on_frozen || !shape.smooth(kind) {
            return None;
        }
        let p0 = self.p(v);
        let goal: P3 = match (self.mobility[v as usize], kind) {
            (Mobility::Surface, _) => {
                let (mut sum, mut area) = ([0.0; 3], 0.0);
                for t in &faces {
                    let [a, b, c] = t.map(|w| self.p(w));
                    let (u, w) = (sub(b, a), sub(c, a));
                    let n = cross(u, w);
                    let ar = len(n);
                    for k in 0..3 {
                        sum[k] += ar * (a[k] + b[k] + c[k]) / 3.0;
                    }
                    area += ar;
                }
                if !(area > 0.0) {
                    return None;
                }
                sum.map(|c| c / area)
            }
            (_, PointClass::Edge(_)) => match self.along.get(&v).map(|n| n.as_slice()) {
                Some(&[a, b]) => {
                    let (pa, pb) = (self.p(a), self.p(b));
                    std::array::from_fn(|k| 0.5 * (pa[k] + pb[k]))
                }
                _ => return None,
            },
            _ => return None,
        };
        let angle = |x: P3| -> f64 {
            faces
                .iter()
                .map(|t| {
                    let at = |w: u32| if w == v { x } else { self.p(w) };
                    tri_min_angle([at(t[0]), at(t[1]), at(t[2])])
                })
                .fold(f64::INFINITY, f64::min)
        };
        // How far the faces around v stray from their carriers: their
        // centroids' distance to them. A move may not raise it (a chord
        // across a thin part, a bridge).
        let patches: SmallVec<[u32; 8]> = self.vfaces[v as usize]
            .iter()
            .filter(|&&f| self.face_alive[f as usize])
            .map(|&f| self.c.faces[f as usize].patch)
            .collect();
        let stray = |x: P3| -> f64 {
            faces
                .iter()
                .zip(&patches)
                .map(|(t, &p)| {
                    let at = |w: u32| if w == v { x } else { self.p(w) };
                    let c: P3 =
                        std::array::from_fn(|k| (at(t[0])[k] + at(t[1])[k] + at(t[2])[k]) / 3.0);
                    self.project_on_patch(*t, c, p).map_or(0.0, |q| dist(c, q))
                })
                .fold(0.0, f64::max)
        };
        let before = angle(p0);
        let stray0 = stray(p0);
        let star: SmallVec<[u32; 32]> = self.live(v).collect();
        for frac in [1.0, 0.5, 0.25] {
            let x0: P3 = std::array::from_fn(|k| p0[k] + frac * (goal[k] - p0[k]));
            let x = self.project_from(kind, x0, v)?;
            if angle(x) > before + GAIN
                && stray(x) <= stray0
                && self.star_accepts(v, &star, x, target)
            {
                return Some(x);
            }
        }
        None
    }

    /// Relaxes the surface over at most `sweeps` sweeps (see
    /// [`Improver::plan_relax`]): each sweep plans every vertex in parallel
    /// and applies the plans in order, planning again a vertex whose star
    /// changed meanwhile; stops when a sweep moves nothing.
    fn relax_surface(&mut self, sweeps: usize, target: f64) -> usize {
        use rayon::prelude::*;
        if self.shape.is_none() {
            return 0;
        }
        let candidates: Vec<u32> = (0..self.c.points.len() as u32)
            .filter(|&v| !self.vfaces[v as usize].is_empty())
            .collect();
        let mut moved = 0;
        let mut stamp = vec![0u32; self.c.points.len()];
        for sweep in 1..=sweeps as u32 {
            let plans: Vec<(u32, Option<P3>)> = candidates
                .par_iter()
                .map(|&v| (v, self.plan_relax(v, target)))
                .collect();
            let mut any = false;
            for (v, plan) in plans {
                let Some(mut x) = plan else { continue };
                let star: Vec<u32> = self.live(v).collect();
                let stale = star.iter().any(|&t| {
                    self.c.tets[t as usize]
                        .iter()
                        .any(|&w| stamp[w as usize] == sweep)
                });
                if stale {
                    match self.plan_relax(v, target) {
                        Some(y) => x = y,
                        None => continue,
                    }
                }
                self.commit_move(v, &star, x);
                for &t in &star {
                    for w in self.c.tets[t as usize] {
                        stamp[w as usize] = sweep;
                    }
                }
                moved += 1;
                any = true;
            }
            if !any {
                break;
            }
        }
        moved
    }

    /// Moves the given vertices onto the shape, curve vertices before patch
    /// vertices, as far as every tet of the star stays positive and none
    /// drops below `FLOOR_DEG` (or below where its star already was): the
    /// full move, else the largest of a few halvings, else none. Returns
    /// the vertices left short of the shape.
    fn snap(&mut self, verts: &[u32]) -> Vec<u32> {
        use rayon::prelude::*;
        let targets: Vec<Option<P3>> = verts
            .par_iter()
            .map(|&v| self.project_from(self.c.classes[v as usize], self.p(v), v))
            .collect();
        let mut short = Vec::new();
        for pass in [0, 1] {
            for (&v, &q) in verts.iter().zip(&targets) {
                let on_curve = matches!(self.c.classes[v as usize], PointClass::Edge(_));
                if on_curve != (pass == 0) {
                    continue;
                }
                let Some(q) = q else { continue };
                let p = self.p(v);
                if p == q {
                    continue;
                }
                let star: Vec<u32> = self.live(v).collect();
                let floor = star
                    .iter()
                    .map(|&t| self.q[t as usize])
                    .fold(f64::INFINITY, f64::min)
                    .min(FLOOR_DEG);
                let mut f = 1.0;
                let mut moved = false;
                for _ in 0..=HALVINGS {
                    let x: P3 = std::array::from_fn(|k| p[k] + f * (q[k] - p[k]));
                    if self.star_min(v, &star, x).is_some_and(|w| w >= floor) {
                        self.commit_move(v, &star, x);
                        moved = true;
                        break;
                    }
                    f *= 0.5;
                }
                if !(moved && f == 1.0) {
                    short.push(v);
                }
            }
        }
        short
    }

    /// Smallest dihedral over the star of `v` with `v` at `x`, `None` when a
    /// tet would not stay positive.
    fn star_min(&self, v: u32, star: &[u32], x: P3) -> Option<f64> {
        let mut worst = f64::INFINITY;
        for &t in star {
            let pts = self.c.tets[t as usize].map(|w| if w == v { x } else { self.p(w) });
            if !(orient3d(pts[0], pts[1], pts[2], pts[3]) > 0.0) {
                return None;
            }
            worst = worst.min(tet_min_dihedral(pts));
        }
        Some(worst)
    }

    /// Drops the dead tets and faces; returns the tets left below the
    /// target and the change counts.
    fn compact(self, target_deg: f64) -> ImproveStats {
        let st = ImproveStats {
            flips23: self.stats_flips23,
            flips32: self.stats_flips32,
            flips44: self.stats_flips44,
            moves: self.stats_moves,
            bad_after: self.bad(target_deg),
            ..ImproveStats::default()
        };
        let Improver {
            c,
            alive,
            face_alive,
            ..
        } = self;
        let mut k = 0;
        c.faces.retain(|_| {
            k += 1;
            face_alive[k - 1]
        });
        let mut i = 0;
        c.tets.retain(|_| {
            i += 1;
            alive[i - 1]
        });
        let mut i = 0;
        c.regions.retain(|_| {
            i += 1;
            alive[i - 1]
        });
        st
    }
}

/// Snaps the boundary of `c` onto `shape` and repairs the tets below
/// `target_deg`; while vertices stay short of the shape and fewer of them
/// do each time, snaps those again and repairs around them, at most
/// `rounds` more times. One index serves all of it. The faces of the
/// `frozen` patches keep their triangles and vertices.
pub fn finish(
    c: &mut Complex,
    shape: &dyn Shape,
    target_deg: f64,
    passes: usize,
    rounds: usize,
    frozen: &[u32],
) -> (ImproveStats, usize) {
    let stage = rapidmesh_exact::log::stage;
    let t = rapidmesh_exact::clock::Instant::now();
    adopt_face_vertices(c);
    let verts: Vec<u32> = (0..c.points.len() as u32).collect();
    let mut im = Improver::new(c, Some(shape), frozen);
    stage("finish.index", t.elapsed().as_secs_f64());
    let t = rapidmesh_exact::clock::Instant::now();
    let mut short = im.snap(&verts);
    stage("finish.snap", t.elapsed().as_secs_f64());
    let t = rapidmesh_exact::clock::Instant::now();
    let before = im.bad(target_deg);
    im.repair(target_deg, passes, true);
    stage("finish.repair", t.elapsed().as_secs_f64());
    let t = rapidmesh_exact::clock::Instant::now();
    for _ in 0..rounds {
        if short.is_empty() {
            break;
        }
        let still = im.snap(&short);
        im.repair(target_deg, passes, false);
        let progress = still.len() < short.len();
        short = still;
        if !progress {
            break;
        }
    }
    stage("finish.resnap", t.elapsed().as_secs_f64());
    // Last, so nothing after it undoes what it keeps: the surface relaxed
    // for its triangles, no tet made worse than the target allows.
    let t = rapidmesh_exact::clock::Instant::now();
    let n = im.relax_surface(RELAX_SWEEPS, target_deg);
    rapidmesh_exact::log::stage("finish.relax", t.elapsed().as_secs_f64());
    rapidmesh_exact::log::stat("finish.relaxed", n as f64);
    let t = rapidmesh_exact::clock::Instant::now();
    let mut st = im.compact(target_deg);
    stage("finish.compact", t.elapsed().as_secs_f64());
    st.bad_before = before;
    (st, short.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complex(points: Vec<P3>, tets: Vec<[u32; 4]>) -> Complex {
        Complex {
            classes: vec![PointClass::Interior; points.len()],
            points,
            regions: vec![1; tets.len()],
            tets,
            ..Complex::default()
        }
    }

    /// The star of a vertex `b` inside the tet `a p q r`: three tets around
    /// `a b` and `b p q r`. The line through `a b` meets `p q r`, the edge
    /// does not; a 3-2 flip there would fold `a p q r` over `b p q r`.
    #[test]
    fn no_3_2_flip_around_an_edge_that_misses_the_triangle() {
        let (p, q, r, a, b) = (0, 1, 2, 3, 4);
        let mut c = complex(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, 1.0],
                [0.25, 0.25, 0.9],
            ],
            vec![[a, b, p, q], [a, b, q, r], [a, b, r, p], [b, p, q, r]],
        );
        let im = Improver::new(&mut c, None, &[]);
        for t in 0..3 {
            assert!(im.flip32(t, a, b).is_none(), "tet {t}");
        }
    }

    /// The tet across a triangle shares all three of its corners, also when
    /// the search starts from another corner than the first (a large star
    /// at the first).
    #[test]
    fn the_tet_across_a_triangle_holds_all_its_corners() {
        let (f0, f1, f2, a, e, c0, c1) = (0u32, 1, 2, 3, 4, 5, 6);
        let mut points: Vec<P3> = vec![[0.0; 3]; 7];
        let mut tets = vec![[a, f0, f1, f2], [f1, f2, c0, c1], [f0, f1, f2, e]];
        for k in 0..=LARGE_STAR as u32 {
            let v = points.len() as u32;
            points.extend([
                [k as f64, 1.0, 0.0],
                [k as f64, 2.0, 0.0],
                [k as f64, 3.0, 0.0],
            ]);
            tets.push([f0, v, v + 1, v + 2]);
        }
        let mut c = complex(points, tets);
        let im = Improver::new(&mut c, None, &[]);
        assert_eq!(im.across(0, [f0, f1, f2]), Some(2));
    }
}
