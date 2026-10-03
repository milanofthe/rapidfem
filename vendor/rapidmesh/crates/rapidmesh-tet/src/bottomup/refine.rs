//! Size-driven refinement of a region's constrained Delaunay
//! tetrahedralization.
//!
//! A tet whose circumradius exceeds the local size takes its circumcenter,
//! the worst first, inserted by Bowyer and Watson with the cavity kept on
//! this side of every constraint. The circumcenter is taken only where the
//! tet sees it (the segment from its centroid crosses no constraint): a
//! flat tet through a thin layer has its circumcenter far outside the layer
//! and stays as it is, so a layer is never refined down to its thickness
//! (the rule of TetGen and gmsh). A point too close to the cavity's corners
//! is not inserted either. There is no shape criterion: the improvement
//! that follows takes care of the shapes.

use super::predicates::{inside, orient, P3};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BinaryHeap;

const NONE: u32 = u32::MAX;

/// The vertices of face `i` of a positive tet, turned so the tet's vertex
/// `i` lies on their positive side.
const FACE: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]];

/// A tet is refined while its circumradius exceeds this multiple of the
/// size at its circumcentre (a regular tet of edge `h` has circumradius
/// `0.61 h`).
pub const RADIUS_OVER_SIZE: f64 = 0.9;

/// A circumcenter closer than this multiple of the size to a corner of its
/// cavity is not inserted.
pub const SPACING: f64 = 0.9;

/// A tet on the boundary alone flatter than this (smallest dihedral, in
/// degrees) takes a point inside, where that leaves no tet this flat.
pub const FLAT_DEG: f64 = 15.0;

/// A point inside a flat tet is not inserted closer than this multiple of
/// the tet's shortest edge to a corner of its cavity.
const FLAT_GAP: f64 = 0.2;

/// The refined tets of a region: `tets` over the region's points (global
/// ids) and the new points, numbered from `base` in the order of `points`.
pub struct Refined {
    pub points: Vec<P3>,
    pub tets: Vec<[u32; 4]>,
}

/// Refines the tets (global ids over `points`, positive) of a region whose
/// constraints are `faces`, to the size `size`, with new points numbered
/// from `base`; at most `budget` points are added.
#[allow(clippy::too_many_arguments)]
pub fn refine(
    points: &[P3],
    tets: &[[u32; 4]],
    faces: &[[u32; 3]],
    beyond: &[u32],
    size: &(dyn Fn(P3) -> f64 + Sync),
    base: u32,
    budget: usize,
) -> Refined {
    // Local vertices in global order, the keys of the perturbation; new
    // points follow with larger keys.
    let mut ids: Vec<u32> = tets.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let local: FxHashMap<u32, u32> = ids
        .iter()
        .enumerate()
        .map(|(i, &g)| (g, i as u32))
        .collect();
    let n0 = ids.len();
    let mut m = Mesh {
        pts: ids.iter().map(|&g| points[g as usize]).collect(),
        tets: tets.iter().map(|t| t.map(|v| local[&v])).collect(),
        nbr: Vec::new(),
        alive: vec![true; tets.len()],
        free: Vec::new(),
        constraint: faces
            .iter()
            .map(|f| {
                let mut k = f.map(|v| local[&v]);
                k.sort_unstable();
                k
            })
            .collect(),
        mark: vec![0; tets.len()],
        epoch: 0,
    };
    m.wire();
    // The region beyond each constraint (by its sorted local corners).
    let behind: FxHashMap<[u32; 3], u32> = faces
        .iter()
        .zip(beyond)
        .map(|(f, &r)| {
            let mut k = f.map(|v| local[&v]);
            k.sort_unstable();
            (k, r)
        })
        .collect();
    let mut heap: BinaryHeap<(Ordered, u32)> = BinaryHeap::new();
    for t in 0..m.tets.len() as u32 {
        if let Some(b) = m.badness(t, size) {
            heap.push((Ordered(b), t));
        }
    }
    let mut added = 0;
    while let Some((_, t)) = heap.pop() {
        if added >= budget {
            break;
        }
        if !m.alive[t as usize] {
            continue;
        }
        let Some(c) = m.circumcenter(t) else {
            continue;
        };
        let Some(home) = m.walk(t, c) else {
            continue;
        };
        let h = size(c);
        let Some(made) = m.insert(home, c, SPACING * h, 0.0) else {
            continue;
        };
        added += 1;
        for nt in made {
            if let Some(b) = m.badness(nt, size) {
                heap.push((Ordered(b), nt));
            }
        }
    }
    // A flat tet on the boundary alone (its four corners on the region's
    // faces) has no point the improvement may move: it takes one inside,
    // at its centroid, unless a new tet would be as flat (across a thin
    // layer, where the tets are flat by design). The new tets all hold that
    // point, so this ends.
    let flat: Vec<u32> = (0..m.tets.len() as u32)
        .filter(|&t| {
            m.alive[t as usize]
                && m.tets[t as usize].iter().all(|&v| (v as usize) < n0)
                && min_dihedral(m.tets[t as usize].map(|v| m.p(v))) < FLAT_DEG
        })
        .collect();
    for t in flat {
        if added >= budget {
            break;
        }
        if !m.alive[t as usize] {
            continue;
        }
        let p = m.tets[t as usize].map(|v| m.p(v));
        let g: P3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k] + p[3][k]) / 4.0);
        let shortest = (0..4)
            .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
            .map(|(i, j)| dist(p[i], p[j]))
            .fold(f64::INFINITY, f64::min);
        // A point off the tet's boundary faces into the region by part of
        // the size (under a ridge of two faces, or a cap of two triangles,
        // the centroid lies on them), else the centroid.
        let tv = m.tets[t as usize];
        // A flat tet between two other regions (a contact wedge) is left
        // to the contact fill.
        let mut others: Vec<u32> = FACE
            .iter()
            .filter_map(|f| {
                let mut k = f.map(|j| tv[j]);
                k.sort_unstable();
                behind.get(&k).copied()
            })
            .collect();
        others.sort_unstable();
        others.dedup();
        if others.len() > 1 {
            continue;
        }
        let mut inward = [0.0; 3];
        for (i, f) in FACE.iter().enumerate() {
            let mut k = f.map(|j| tv[j]);
            k.sort_unstable();
            if !m.constraint.contains(&k) {
                continue;
            }
            let q = f.map(|j| p[j]);
            let n = cross(sub(q[1], q[0]), sub(q[2], q[0]));
            // Turned towards the tet's vertex across the face.
            let s = if dot(n, sub(p[i], q[0])) > 0.0 {
                1.0
            } else {
                -1.0
            };
            let l = dot(n, n).sqrt().max(1e-300);
            for k in 0..3 {
                inward[k] += s * n[k] / l;
            }
        }
        let l = dot(inward, inward).sqrt();
        let h = size(g);
        let off = (l > 1e-12).then(|| inward.map(|x| x / l));
        // A new tet must be twice as good as this one (up to the bound):
        // the improvement then takes it further.
        let least = (2.0 * min_dihedral(p)).min(FLAT_DEG);
        let mut done = false;
        if let Some(n) = off {
            for share in [0.5, 0.3, 0.8] {
                let c: P3 = std::array::from_fn(|k| g[k] + share * h * n[k]);
                let Some(home) = m.walk(t, c) else { continue };
                if m.insert(home, c, FLAT_GAP * h, least).is_some() {
                    done = true;
                    break;
                }
            }
        }
        // Boundary faces on either side cancel (a tet across a thin
        // layer, flat by design): nothing goes in.
        if done || (off.is_some() && m.insert(t, g, FLAT_GAP * shortest, least).is_some()) {
            added += 1;
        }
    }
    let out_tets = m
        .tets
        .iter()
        .zip(&m.alive)
        .filter(|(_, &a)| a)
        .map(|(t, _)| {
            t.map(|v| {
                if (v as usize) < n0 {
                    ids[v as usize]
                } else {
                    base + (v as usize - n0) as u32
                }
            })
        })
        .collect();
    Refined {
        points: m.pts[n0..].to_vec(),
        tets: out_tets,
    }
}

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

/// The smallest dihedral angle of a tet, in degrees.
fn min_dihedral(p: [P3; 4]) -> f64 {
    let sub = |a: P3, b: P3| -> P3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] };
    let cross = |a: P3, b: P3| -> P3 {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: P3, b: P3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    // Outward normals of the faces opposite each vertex.
    let n: Vec<P3> = (0..4)
        .map(|i| {
            let f = FACE[i].map(|k| p[k]);
            let n = cross(sub(f[1], f[0]), sub(f[2], f[0]));
            let l = dot(n, n).sqrt().max(1e-300);
            n.map(|x| -x / l)
        })
        .collect();
    let mut least = 180.0f64;
    for i in 0..4 {
        for j in i + 1..4 {
            let c = (-dot(n[i], n[j])).clamp(-1.0, 1.0);
            least = least.min(c.acos().to_degrees());
        }
    }
    least
}

/// A float with a total order, for the heap.
struct Ordered(f64);

impl PartialEq for Ordered {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

struct Mesh {
    pts: Vec<P3>,
    tets: Vec<[u32; 4]>,
    /// Per tet and face: the neighbour as `tet << 2 | its face`.
    nbr: Vec<[u32; 4]>,
    alive: Vec<bool>,
    free: Vec<u32>,
    /// The constraint triangles, sorted vertex triples.
    constraint: FxHashSet<[u32; 3]>,
    mark: Vec<u32>,
    epoch: u32,
}

impl Mesh {
    fn p(&self, v: u32) -> P3 {
        self.pts[v as usize]
    }

    fn face(&self, t: u32, i: usize) -> [u32; 3] {
        let tv = self.tets[t as usize];
        FACE[i].map(|k| tv[k])
    }

    fn is_constraint(&self, f: [u32; 3]) -> bool {
        let mut k = f;
        k.sort_unstable();
        self.constraint.contains(&k)
    }

    /// Links the tets across their shared faces.
    fn wire(&mut self) {
        let mut open: FxHashMap<[u32; 3], u32> = FxHashMap::default();
        self.nbr = vec![[NONE; 4]; self.tets.len()];
        for t in 0..self.tets.len() as u32 {
            for i in 0..4 {
                let mut k = self.face(t, i);
                k.sort_unstable();
                let here = t << 2 | i as u32;
                match open.remove(&k) {
                    Some(there) => {
                        self.nbr[t as usize][i] = there;
                        self.nbr[(there >> 2) as usize][(there & 3) as usize] = here;
                    }
                    None => {
                        open.insert(k, here);
                    }
                }
            }
        }
    }

    /// The circumradius of `t` over the size at its centroid, when above
    /// the refinement threshold.
    fn badness(&self, t: u32, size: &(dyn Fn(P3) -> f64 + Sync)) -> Option<f64> {
        let p = self.tets[t as usize].map(|v| self.p(v));
        let c = circumcenter(p)?;
        // The size where the circumcentre would go: the same the spacing of
        // the insertion reads, so a tet too large is one whose circumcentre
        // clears every corner (the sphere is empty) and goes in.
        let r = dist(c, p[0]);
        let ratio = r / size(c).max(1e-300);
        (ratio > RADIUS_OVER_SIZE).then_some(ratio)
    }

    fn circumcenter(&self, t: u32) -> Option<P3> {
        circumcenter(self.tets[t as usize].map(|v| self.p(v)))
    }

    /// The tet holding `c`, reached from `t` along the segment from its
    /// centroid without crossing a constraint; none when a constraint is in
    /// the way (the tet does not see its circumcenter) or the walk strays.
    fn walk(&self, t: u32, c: P3) -> Option<u32> {
        let p = self.tets[t as usize].map(|v| self.p(v));
        let g: P3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k] + p[3][k]) / 4.0);
        let mut cur = t;
        let mut from = NONE;
        for _ in 0..4096 {
            let mut exit = None;
            for i in 0..4 {
                let nb = self.nbr[cur as usize][i];
                if nb != NONE && nb >> 2 == from {
                    continue;
                }
                let f = self.face(cur, i).map(|v| self.p(v));
                if orient(f[0], f[1], f[2], c) >= 0 {
                    continue;
                }
                // The segment leaves through this face when it passes
                // through the face's closed triangle.
                let o = [
                    orient(g, c, f[0], f[1]),
                    orient(g, c, f[1], f[2]),
                    orient(g, c, f[2], f[0]),
                ];
                if (o.iter().all(|&x| x >= 0) || o.iter().all(|&x| x <= 0))
                    && o.iter().any(|&x| x != 0)
                {
                    exit = Some(i);
                    break;
                }
            }
            let Some(i) = exit else {
                return Some(cur);
            };
            if self.is_constraint(self.face(cur, i)) {
                return None;
            }
            let nb = self.nbr[cur as usize][i];
            if nb == NONE {
                return None;
            }
            from = cur;
            cur = nb >> 2;
        }
        None
    }

    /// Inserts `c` into the cavity grown from `home` through the faces that
    /// are no constraints; none (and nothing changed) when a corner of the
    /// cavity lies within `gap` of it or the cavity is not star-shaped.
    fn insert(&mut self, home: u32, c: P3, gap: f64, least: f64) -> Option<Vec<u32>> {
        let p = self.pts.len() as u32;
        self.pts.push(c);
        self.epoch += 1;
        let epoch = self.epoch;
        let key = |t: [u32; 4]| [t[0], t[1], t[2], t[3], p];
        // A corner of the cavity crowding the point refuses it: found as the
        // cavity grows, not after (most refusals come early and cheap).
        let crowds =
            |pts: &[P3], t: [u32; 4]| t.iter().any(|&v| v != p && dist(pts[v as usize], c) < gap);
        if crowds(&self.pts, self.tets[home as usize]) {
            self.pts.pop();
            return None;
        }
        let mut cavity = vec![home];
        self.mark[home as usize] = epoch;
        let mut at = 0;
        while at < cavity.len() {
            let t = cavity[at];
            at += 1;
            for i in 0..4 {
                let nb = self.nbr[t as usize][i];
                if nb == NONE || self.mark[(nb >> 2) as usize] == epoch {
                    continue;
                }
                if self.is_constraint(self.face(t, i)) {
                    continue;
                }
                let n = nb >> 2;
                let tv = self.tets[n as usize];
                if inside(tv.map(|v| self.pts[v as usize]), c, key(tv)) {
                    if crowds(&self.pts, tv) {
                        self.pts.pop();
                        return None;
                    }
                    self.mark[n as usize] = epoch;
                    cavity.push(n);
                }
            }
        }
        // Every boundary face must see the new point strictly from inside,
        // and no corner may crowd it.
        let mut boundary: Vec<(u32, usize)> = Vec::new();
        for &t in &cavity {
            for i in 0..4 {
                let nb = self.nbr[t as usize][i];
                if nb != NONE && self.mark[(nb >> 2) as usize] == epoch {
                    continue;
                }
                let f = self.face(t, i).map(|v| self.p(v));
                if orient(f[0], f[1], f[2], c) <= 0
                    || f.iter().any(|&x| dist(x, c) < gap)
                    || (least > 0.0 && min_dihedral([f[0], f[1], f[2], c]) <= least)
                {
                    for &u in &cavity {
                        self.mark[u as usize] = 0;
                    }
                    self.pts.pop();
                    return None;
                }
                boundary.push((t, i));
            }
        }
        let mut links: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut made = Vec::with_capacity(boundary.len());
        for (t, i) in boundary {
            let outside = self.nbr[t as usize][i];
            let f = self.face(t, i);
            let nt = self.alloc([f[0], f[1], f[2], p]);
            made.push(nt);
            self.nbr[nt as usize][3] = outside;
            if outside != NONE {
                self.nbr[(outside >> 2) as usize][(outside & 3) as usize] = nt << 2 | 3;
            }
            for e in 0..3 {
                let (a, b) = (f[e], f[(e + 1) % 3]);
                let here = nt << 2 | ((e + 2) % 3) as u32;
                match links.remove(&(a.min(b), a.max(b))) {
                    Some(there) => {
                        self.nbr[nt as usize][(e + 2) % 3] = there;
                        self.nbr[(there >> 2) as usize][(there & 3) as usize] = here;
                    }
                    None => {
                        links.insert((a.min(b), a.max(b)), here);
                    }
                }
            }
        }
        for &t in &cavity {
            self.alive[t as usize] = false;
            self.free.push(t);
        }
        Some(made)
    }

    fn alloc(&mut self, t: [u32; 4]) -> u32 {
        match self.free.pop() {
            Some(id) => {
                self.tets[id as usize] = t;
                self.nbr[id as usize] = [NONE; 4];
                self.alive[id as usize] = true;
                id
            }
            None => {
                self.tets.push(t);
                self.nbr.push([NONE; 4]);
                self.alive.push(true);
                self.mark.push(0);
                (self.tets.len() - 1) as u32
            }
        }
    }
}

fn dist(a: P3, b: P3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The center of the sphere through four points (none when they are flat).
fn circumcenter(p: [P3; 4]) -> Option<P3> {
    let [a, b, c, d] = p;
    let sub = |x: P3, y: P3| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let dot = |x: P3, y: P3| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let cross = |x: P3, y: P3| {
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
    };
    let (u, v, w) = (sub(b, a), sub(c, a), sub(d, a));
    let det = 2.0 * dot(u, cross(v, w));
    if !(det.abs() > 0.0) {
        return None;
    }
    let (uu, vv, ww) = (dot(u, u), dot(v, v), dot(w, w));
    let (vw, wu, uv) = (cross(v, w), cross(w, u), cross(u, v));
    let o: P3 = std::array::from_fn(|k| a[k] + (uu * vw[k] + vv * wu[k] + ww * uv[k]) / det);
    o.iter().all(|x| x.is_finite()).then_some(o)
}
