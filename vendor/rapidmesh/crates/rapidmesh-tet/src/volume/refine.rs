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

use crate::predicates::{inside, orient};
use crate::simplex::TET_FACES;
use crate::simplex::{tet_circumsphere, tet_min_dihedral, Ordered};
use crate::volume::tets::{Tets, NONE};
use rapidmesh_exact::vector::V3;
use rapidmesh_exact::vector::{cross, dist, dot, sub};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BinaryHeap;

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
    pub points: Vec<V3>,
    pub tets: Vec<[u32; 4]>,
    /// Whether the budget stopped the refinement with tets still too large.
    pub capped: bool,
}

/// Refines the tets (global ids over `points`, positive) of a region whose
/// constraints are `faces`, to the size `size`, with new points numbered
/// from `base`; at most `budget` points are added.
#[allow(clippy::too_many_arguments)]
pub fn refine(
    points: &[V3],
    tets: &[[u32; 4]],
    faces: &[[u32; 3]],
    beyond: &[u32],
    size: &(dyn Fn(V3) -> f64 + Sync),
    base: u32,
    budget: usize,
) -> Refined {
    // Local vertices in global order; new points follow.
    let mut ids: Vec<u32> = tets.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let local: FxHashMap<u32, u32> = ids
        .iter()
        .enumerate()
        .map(|(i, &g)| (g, i as u32))
        .collect();
    let n0 = ids.len();
    let constraint: FxHashSet<[u32; 3]> = faces
        .iter()
        .map(|f| {
            let mut k = f.map(|v| local[&v]);
            k.sort_unstable();
            k
        })
        .collect();
    let t = Tets::wired(tets.iter().map(|t| t.map(|v| local[&v])).collect());
    // Per tet, which of its faces are constraints (bit `i` for face `i`).
    let cmask: Vec<u8> = (0..t.tets.len() as u32)
        .map(|ti| {
            (0..4).fold(0u8, |m, i| {
                let mut k = t.face(ti, i);
                k.sort_unstable();
                m | (u8::from(constraint.contains(&k)) << i)
            })
        })
        .collect();
    let mut m = Mesh {
        pts: ids.iter().map(|&g| points[g as usize]).collect(),
        t,
        cmask,
        cavity: Vec::new(),
        boundary: Vec::new(),
    };
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
    let mut heap: BinaryHeap<Bad> = BinaryHeap::new();
    for t in 0..m.t.tets.len() as u32 {
        if let Some(b) = m.badness(t, size) {
            heap.push(b);
        }
    }
    let mut added = 0;
    let mut capped = false;
    let mut made = Vec::new();
    while let Some(b) = heap.pop() {
        if added >= budget {
            capped = true;
            break;
        }
        // The tet as it was queued (its slot may hold another since).
        if !m.t.alive[b.t as usize] || m.t.tets[b.t as usize] != b.corners {
            continue;
        }
        let Some(home) = m.walk(b.t, b.c) else {
            continue;
        };
        if !m.insert(home, b.c, SPACING * b.h, 0.0, &mut made) {
            continue;
        }
        added += 1;
        for &nt in &made {
            if let Some(b) = m.badness(nt, size) {
                heap.push(b);
            }
        }
    }
    // A flat tet on the boundary alone (its four corners on the region's
    // faces) has no point the improvement may move: it takes one inside,
    // at its centroid, unless a new tet would be as flat (across a thin
    // layer, where the tets are flat by design). The new tets all hold that
    // point, so this ends.
    // A tet's corners in the order of their places, so what it takes is
    // the same however the mesh is numbered; the flat tets in that order.
    let placed = |m: &Mesh, t: u32| -> [u32; 4] {
        let mut tv = m.t.tets[t as usize];
        tv.sort_by_key(|&v| m.p(v).map(f64::to_bits));
        tv
    };
    let mut flat: Vec<u32> = (0..m.t.tets.len() as u32)
        .filter(|&t| {
            m.t.alive[t as usize]
                && m.t.tets[t as usize].iter().all(|&v| (v as usize) < n0)
                && tet_min_dihedral(m.t.tets[t as usize].map(|v| m.p(v))) < FLAT_DEG
        })
        .collect();
    flat.sort_by_cached_key(|&t| placed(&m, t).map(|v| m.p(v).map(f64::to_bits)));
    for t in flat {
        if added >= budget {
            break;
        }
        if !m.t.alive[t as usize] {
            continue;
        }
        let tv = placed(&m, t);
        let p = tv.map(|v| m.p(v));
        let g: V3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k] + p[3][k]) / 4.0);
        let shortest = (0..4)
            .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
            .map(|(i, j)| dist(p[i], p[j]))
            .fold(f64::INFINITY, f64::min);
        // A point off the tet's boundary faces into the region by part of
        // the size (under a ridge of two faces, or a cap of two triangles,
        // the centroid lies on them), else the centroid.
        // A flat tet between two other regions (where two bodies touch,
        // the region between them narrows to nothing) stays as it is:
        // refining a wedge that closes would not end, and its flat tets
        // keep the material of the region they lie in.
        let mut others: Vec<u32> = TET_FACES
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
        for (i, f) in TET_FACES.iter().enumerate() {
            let mut k = f.map(|j| tv[j]);
            k.sort_unstable();
            if !constraint.contains(&k) {
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
        let least = (2.0 * tet_min_dihedral(p)).min(FLAT_DEG);
        let mut done = false;
        if let Some(n) = off {
            for share in [0.5, 0.3, 0.8] {
                let c: V3 = std::array::from_fn(|k| g[k] + share * h * n[k]);
                let Some(home) = m.walk(t, c) else { continue };
                if m.insert(home, c, FLAT_GAP * h, least, &mut made) {
                    done = true;
                    break;
                }
            }
        }
        // Boundary faces on either side cancel (a tet across a thin
        // layer, flat by design): nothing goes in.
        if done || (off.is_some() && m.insert(t, g, FLAT_GAP * shortest, least, &mut made)) {
            added += 1;
        }
    }
    let out_tets =
        m.t.tets
            .iter()
            .zip(&m.t.alive)
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
        capped,
    }
}

/// A tet queued for its circumcenter: how much too large it is, the
/// circumcenter and the size there, and its corners when queued.
struct Bad {
    ratio: Ordered,
    t: u32,
    corners: [u32; 4],
    c: V3,
    h: f64,
}

impl PartialEq for Bad {
    fn eq(&self, other: &Bad) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Bad {}

impl PartialOrd for Bad {
    fn partial_cmp(&self, other: &Bad) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The worst first, ties by where the circumcenter lies (not by the tet's
/// number: the order of the insertions is the geometry's). Tets with one
/// circumsphere insert one point.
impl Ord for Bad {
    fn cmp(&self, other: &Bad) -> std::cmp::Ordering {
        let at = |b: &Bad| b.c.map(Ordered);
        (self.ratio, at(self)).cmp(&(other.ratio, at(other)))
    }
}

struct Mesh {
    pts: Vec<V3>,
    t: Tets,
    /// Per tet, which of its faces are constraints (bit `i` for face `i`).
    cmask: Vec<u8>,
    /// Buffers of an insertion, kept between them.
    cavity: Vec<u32>,
    boundary: Vec<(u32, usize)>,
}

impl Mesh {
    fn p(&self, v: u32) -> V3 {
        self.pts[v as usize]
    }

    fn face(&self, t: u32, i: usize) -> [u32; 3] {
        self.t.face(t, i)
    }

    /// Whether face `i` of tet `t` is a constraint.
    fn is_constraint(&self, t: u32, i: usize) -> bool {
        self.cmask[t as usize] >> i & 1 == 1
    }

    /// Tet `t` queued when its circumradius exceeds the refinement
    /// threshold times the size at its circumcenter.
    fn badness(&self, t: u32, size: &(dyn Fn(V3) -> f64 + Sync)) -> Option<Bad> {
        let corners = self.t.tets[t as usize];
        let p = corners.map(|v| self.p(v));
        let (c, radius) = tet_circumsphere(p)?;
        // The size where the circumcentre would go: the same the spacing of
        // the insertion reads, so a tet too large is one whose circumcentre
        // clears every corner (the sphere is empty) and goes in.
        let h = size(c);
        let ratio = radius / h.max(1e-300);
        (ratio > RADIUS_OVER_SIZE).then_some(Bad {
            ratio: Ordered(ratio),
            t,
            corners,
            c,
            h,
        })
    }

    /// The tet holding `c`, reached from `t` along the segment from its
    /// centroid without crossing a constraint; none when a constraint is in
    /// the way (the tet does not see its circumcenter) or the walk strays.
    fn walk(&self, t: u32, c: V3) -> Option<u32> {
        let p = self.t.tets[t as usize].map(|v| self.p(v));
        let g: V3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k] + p[3][k]) / 4.0);
        let mut cur = t;
        let mut from = NONE;
        for _ in 0..4096 {
            let mut exit = None;
            for i in 0..4 {
                let nb = self.t.nbr[cur as usize][i];
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
            if self.is_constraint(cur, i) {
                return None;
            }
            let nb = self.t.nbr[cur as usize][i];
            if nb == NONE {
                return None;
            }
            from = cur;
            cur = nb >> 2;
        }
        None
    }

    /// Inserts `c` into the cavity grown from `home` through the faces that
    /// are no constraints, the new tets into `made`; false (and nothing
    /// changed) when a corner of the cavity lies within `gap` of it, the
    /// cavity is not star-shaped or it holds both sides of a constraint.
    fn insert(&mut self, home: u32, c: V3, gap: f64, least: f64, made: &mut Vec<u32>) -> bool {
        let p = self.pts.len() as u32;
        self.pts.push(c);
        let epoch = self.t.next_epoch();
        // A corner of the cavity crowding the point refuses it: found as the
        // cavity grows, not after (most refusals come early and cheap).
        let crowds =
            |pts: &[V3], t: [u32; 4]| t.iter().any(|&v| v != p && dist(pts[v as usize], c) < gap);
        if crowds(&self.pts, self.t.tets[home as usize]) {
            self.pts.pop();
            return false;
        }
        let mut cavity = std::mem::take(&mut self.cavity);
        let mut boundary = std::mem::take(&mut self.boundary);
        cavity.clear();
        boundary.clear();
        let ok = self.grow(
            home,
            c,
            gap,
            least,
            epoch,
            &crowds,
            &mut cavity,
            &mut boundary,
        );
        if ok {
            made.clear();
            self.t.cone(&boundary, p, made);
            // The new tet on each boundary face keeps that face (its face
            // 3) and whether it is a constraint.
            for (&(t, i), &nt) in boundary.iter().zip(made.iter()) {
                let bit = self.cmask[t as usize] >> i & 1;
                if nt as usize >= self.cmask.len() {
                    self.cmask.resize(nt as usize + 1, 0);
                }
                self.cmask[nt as usize] = bit << 3;
            }
            self.t.kill(&cavity);
        } else {
            for &u in &cavity {
                self.t.mark[u as usize] = 0;
            }
            self.pts.pop();
        }
        self.cavity = cavity;
        self.boundary = boundary;
        ok
    }

    /// The cavity of `c` from `home` and its boundary faces;
    /// false where a corner crowds it, a boundary face does not see it
    /// strictly from inside, a new tet would be no better than `least`, or
    /// the cavity holds both sides of a constraint (it grew round the open
    /// rim of a sheet inside the region).
    #[allow(clippy::too_many_arguments)]
    fn grow(
        &mut self,
        home: u32,
        c: V3,
        gap: f64,
        least: f64,
        epoch: u32,
        crowds: &dyn Fn(&[V3], [u32; 4]) -> bool,
        cavity: &mut Vec<u32>,
        boundary: &mut Vec<(u32, usize)>,
    ) -> bool {
        cavity.push(home);
        self.t.mark[home as usize] = epoch;
        let mut at = 0;
        while at < cavity.len() {
            let t = cavity[at];
            at += 1;
            for i in 0..4 {
                let nb = self.t.nbr[t as usize][i];
                if nb == NONE || self.t.mark[(nb >> 2) as usize] == epoch {
                    continue;
                }
                if self.is_constraint(t, i) {
                    continue;
                }
                let n = nb >> 2;
                let tv = self.t.tets[n as usize];
                if inside(tv.map(|v| self.pts[v as usize]), c) {
                    if crowds(&self.pts, tv) {
                        return false;
                    }
                    self.t.mark[n as usize] = epoch;
                    cavity.push(n);
                }
            }
        }
        for &t in cavity.iter() {
            for i in 0..4 {
                let nb = self.t.nbr[t as usize][i];
                if nb != NONE && self.t.mark[(nb >> 2) as usize] == epoch {
                    if self.is_constraint(t, i) {
                        return false;
                    }
                    continue;
                }
                let f = self.face(t, i).map(|v| self.p(v));
                if orient(f[0], f[1], f[2], c) <= 0
                    || f.iter().any(|&x| dist(x, c) < gap)
                    || (least > 0.0 && tet_min_dihedral([f[0], f[1], f[2], c]) <= least)
                {
                    return false;
                }
                boundary.push((t, i));
            }
        }
        true
    }
}
