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

use crate::predicates::{inside, orient, P3};
use crate::simplex::{tet_circumcenter, tet_min_dihedral, Ordered};
use crate::volume::tets::{Tets, FACE, NONE};
use rapidmesh_geom::vec3::{cross, dist, dot, sub};
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
    let mut m = Mesh {
        pts: ids.iter().map(|&g| points[g as usize]).collect(),
        t: Tets::wired(tets.iter().map(|t| t.map(|v| local[&v])).collect()),
        constraint: faces
            .iter()
            .map(|f| {
                let mut k = f.map(|v| local[&v]);
                k.sort_unstable();
                k
            })
            .collect(),
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
    let mut heap: BinaryHeap<(Ordered, u32)> = BinaryHeap::new();
    for t in 0..m.t.tets.len() as u32 {
        if let Some(b) = m.badness(t, size) {
            heap.push((Ordered(b), t));
        }
    }
    let mut added = 0;
    while let Some((_, t)) = heap.pop() {
        if added >= budget {
            break;
        }
        if !m.t.alive[t as usize] {
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
    let flat: Vec<u32> = (0..m.t.tets.len() as u32)
        .filter(|&t| {
            m.t.alive[t as usize]
                && m.t.tets[t as usize].iter().all(|&v| (v as usize) < n0)
                && tet_min_dihedral(m.t.tets[t as usize].map(|v| m.p(v))) < FLAT_DEG
        })
        .collect();
    for t in flat {
        if added >= budget {
            break;
        }
        if !m.t.alive[t as usize] {
            continue;
        }
        let p = m.t.tets[t as usize].map(|v| m.p(v));
        let g: P3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k] + p[3][k]) / 4.0);
        let shortest = (0..4)
            .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
            .map(|(i, j)| dist(p[i], p[j]))
            .fold(f64::INFINITY, f64::min);
        // A point off the tet's boundary faces into the region by part of
        // the size (under a ridge of two faces, or a cap of two triangles,
        // the centroid lies on them), else the centroid.
        let tv = m.t.tets[t as usize];
        // A flat tet between two other regions (where two bodies touch,
        // the region between them narrows to nothing) stays as it is:
        // refining a wedge that closes would not end, and its flat tets
        // keep the material of the region they lie in.
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
        let least = (2.0 * tet_min_dihedral(p)).min(FLAT_DEG);
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
    }
}

struct Mesh {
    pts: Vec<P3>,
    t: Tets,
    /// The constraint triangles, sorted vertex triples.
    constraint: FxHashSet<[u32; 3]>,
}

impl Mesh {
    fn p(&self, v: u32) -> P3 {
        self.pts[v as usize]
    }

    fn face(&self, t: u32, i: usize) -> [u32; 3] {
        self.t.face(t, i)
    }

    fn is_constraint(&self, f: [u32; 3]) -> bool {
        let mut k = f;
        k.sort_unstable();
        self.constraint.contains(&k)
    }

    /// The circumradius of `t` over the size at its centroid, when above
    /// the refinement threshold.
    fn badness(&self, t: u32, size: &(dyn Fn(P3) -> f64 + Sync)) -> Option<f64> {
        let p = self.t.tets[t as usize].map(|v| self.p(v));
        let c = tet_circumcenter(p)?;
        // The size where the circumcentre would go: the same the spacing of
        // the insertion reads, so a tet too large is one whose circumcentre
        // clears every corner (the sphere is empty) and goes in.
        let r = dist(c, p[0]);
        let ratio = r / size(c).max(1e-300);
        (ratio > RADIUS_OVER_SIZE).then_some(ratio)
    }

    fn circumcenter(&self, t: u32) -> Option<P3> {
        tet_circumcenter(self.t.tets[t as usize].map(|v| self.p(v)))
    }

    /// The tet holding `c`, reached from `t` along the segment from its
    /// centroid without crossing a constraint; none when a constraint is in
    /// the way (the tet does not see its circumcenter) or the walk strays.
    fn walk(&self, t: u32, c: P3) -> Option<u32> {
        let p = self.t.tets[t as usize].map(|v| self.p(v));
        let g: P3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k] + p[3][k]) / 4.0);
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
            if self.is_constraint(self.face(cur, i)) {
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
    /// are no constraints; none (and nothing changed) when a corner of the
    /// cavity lies within `gap` of it, the cavity is not star-shaped or it
    /// holds both sides of a constraint.
    fn insert(&mut self, home: u32, c: P3, gap: f64, least: f64) -> Option<Vec<u32>> {
        let p = self.pts.len() as u32;
        self.pts.push(c);
        let epoch = self.t.next_epoch();
        // A corner of the cavity crowding the point refuses it: found as the
        // cavity grows, not after (most refusals come early and cheap).
        let crowds =
            |pts: &[P3], t: [u32; 4]| t.iter().any(|&v| v != p && dist(pts[v as usize], c) < gap);
        if crowds(&self.pts, self.t.tets[home as usize]) {
            self.pts.pop();
            return None;
        }
        let mut cavity = vec![home];
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
                if self.is_constraint(self.face(t, i)) {
                    continue;
                }
                let n = nb >> 2;
                let tv = self.t.tets[n as usize];
                if inside(tv.map(|v| self.pts[v as usize]), c) {
                    if crowds(&self.pts, tv) {
                        self.pts.pop();
                        return None;
                    }
                    self.t.mark[n as usize] = epoch;
                    cavity.push(n);
                }
            }
        }
        // Every boundary face must see the new point strictly from inside,
        // and no corner may crowd it. A constraint between two tets of the
        // cavity (it grew round the open rim of a sheet inside the region,
        // onto both sides of a face) would vanish: no point then.
        let mut boundary: Vec<(u32, usize)> = Vec::new();
        for &t in &cavity {
            for i in 0..4 {
                let nb = self.t.nbr[t as usize][i];
                if nb != NONE && self.t.mark[(nb >> 2) as usize] == epoch {
                    if self.is_constraint(self.face(t, i)) {
                        for &u in &cavity {
                            self.t.mark[u as usize] = 0;
                        }
                        self.pts.pop();
                        return None;
                    }
                    continue;
                }
                let f = self.face(t, i).map(|v| self.p(v));
                if orient(f[0], f[1], f[2], c) <= 0
                    || f.iter().any(|&x| dist(x, c) < gap)
                    || (least > 0.0 && tet_min_dihedral([f[0], f[1], f[2], c]) <= least)
                {
                    for &u in &cavity {
                        self.t.mark[u as usize] = 0;
                    }
                    self.pts.pop();
                    return None;
                }
                boundary.push((t, i));
            }
        }
        let mut made = Vec::with_capacity(boundary.len());
        self.t.cone(&boundary, p, &mut made);
        self.t.kill(&cavity);
        Some(made)
    }
}
