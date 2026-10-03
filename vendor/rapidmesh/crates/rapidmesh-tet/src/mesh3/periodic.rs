//! Periodic pairs of patches: a patch `b` that is patch `a` shifted by a
//! vector, to be meshed with the same triangles.
//!
//! The mesher keeps the two sides equal in three ways. The curves and
//! corners that the shifts carry onto each other form classes: the lowest
//! curve of a class is sampled, every other one takes its samples shifted,
//! and a protecting ball in a class shrinks with all its images. Every
//! point the refinement puts on a periodic patch is put on its partner
//! too. At the end the restricted facets of both sides are compared; where
//! they differ (four points on a circle, whose two diagonals are equally
//! Delaunay), the circumcenter goes on both sides and the refinement goes
//! on.

use super::oracle::{DomainOracle, P3};
use super::VertexKind;
use rustc_hash::FxHashMap;

/// Patch `b` is patch `a` moved by `shift`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicPair {
    pub a: u32,
    pub b: u32,
    pub shift: P3,
}

fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dist(a: P3, b: P3) -> f64 {
    let d = sub(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

fn union(p: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(p, a), find(p, b));
    if ra != rb {
        p[ra.max(rb)] = ra.min(rb);
    }
}

/// A curve as the image of the root curve of its class: the point at arc
/// length `s` of the root, moved by `shift`, is this curve's point at
/// `phase + dir * s` (modulo the length on a closed curve).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CurveImage {
    pub root: u32,
    pub dir: f64,
    pub phase: f64,
    pub shift: P3,
}

/// The arc length on curve `d` of length `len` nearest to `p`.
fn nearest_arc(d: &dyn crate::curve::Curve, len: f64, p: P3) -> f64 {
    const N: usize = 512;
    let samples: Vec<(f64, P3)> = (0..=N)
        .map(|i| {
            let s = len * i as f64 / N as f64;
            (s, d.point_at(s))
        })
        .collect();
    crate::curve::closest_arc(d, &samples, p).rem_euclid(len)
}

/// The classes the periodic pairs make of a domain's patches, curves and
/// corners.
#[derive(Clone, Debug, Default)]
pub(crate) struct Periodic {
    pub pairs: Vec<PeriodicPair>,
    /// Per patch: its partner and the shift onto it.
    pub patch: Vec<Option<(u32, P3)>>,
    /// Per curve in a class: how it is an image of the lowest curve of the
    /// class (`None` for a curve alone).
    pub curve: Vec<Option<CurveImage>>,
    /// Per corner: the lowest corner of its class.
    pub corner: Vec<u32>,
    /// Points closer than this are one.
    pub tol: f64,
}

impl Periodic {
    /// The classes of `pairs` over `dom`; `corner_patches` holds the
    /// patches through every corner.
    pub fn build<D: DomainOracle + ?Sized>(
        dom: &D,
        corner_patches: &[Vec<u32>],
        pairs: &[PeriodicPair],
    ) -> Periodic {
        let np = dom.patches().len();
        let (nc, nk) = (dom.curves().len(), dom.corners().len());
        let (lo, hi) = dom.bbox();
        let tol = 1e-9 * dist(lo, hi).max(1.0);
        let mut per = Periodic {
            pairs: pairs.to_vec(),
            patch: vec![None; np],
            curve: vec![None; nc],
            corner: (0..nk as u32).collect(),
            tol,
        };
        if pairs.is_empty() {
            return per;
        }
        let mut kp: Vec<usize> = (0..nk).collect();
        let mut cp: Vec<usize> = (0..nc).collect();
        // Relative direction of each curve to its class root, filled by a
        // walk over the matches below.
        // (curve, its image, direction, phase, length, closed): the point
        // at `s` of the first is the image's point at `phase + dir * s`.
        let mut matches: Vec<(usize, usize, f64, f64, f64, bool)> = Vec::new();
        for pp in pairs {
            let neg = pp.shift.map(|x| -x);
            per.patch[pp.a as usize] = Some((pp.b, pp.shift));
            per.patch[pp.b as usize] = Some((pp.a, neg));
            let corners = dom.corners();
            for k in 0..nk {
                if !corner_patches[k].contains(&pp.a) {
                    continue;
                }
                let target = add(corners[k], pp.shift);
                if let Some(j) = (0..nk)
                    .find(|&j| corner_patches[j].contains(&pp.b) && dist(corners[j], target) <= tol)
                {
                    union(&mut kp, k, j);
                }
            }
            let curves = dom.curves();
            for (c, fc) in curves.iter().enumerate() {
                if !fc.patches.contains(&pp.a) {
                    continue;
                }
                let lc = fc.curve.length();
                let closed = fc.ends[0].is_none() || fc.ends[0] == fc.ends[1];
                let at = |f: f64| add(fc.curve.point_at(f * lc), pp.shift);
                for (d, fd) in curves.iter().enumerate() {
                    if !fd.patches.contains(&pp.b) {
                        continue;
                    }
                    let ld = fd.curve.length();
                    if (lc - ld).abs() > tol {
                        continue;
                    }
                    // Where the image of the start lies on `d` (anywhere on a
                    // closed curve, an end on an open one), and which way
                    // round: the quarter points tell.
                    let phases: Vec<f64> = if closed {
                        vec![nearest_arc(&*fd.curve, ld, at(0.0))]
                    } else {
                        vec![0.0, ld]
                    };
                    let on = |arc: f64| {
                        let arc = if closed { arc.rem_euclid(ld) } else { arc };
                        fd.curve.point_at(arc)
                    };
                    let found = phases.iter().find_map(|&phase| {
                        [1.0, -1.0].into_iter().find_map(|dir: f64| {
                            let fits = [0.0, 0.25, 0.5, 0.75, 1.0]
                                .iter()
                                .all(|&f| dist(on(phase + dir * f * lc), at(f)) <= tol);
                            fits.then_some((dir, phase))
                        })
                    });
                    if let Some((dir, phase)) = found {
                        matches.push((c, d, dir, phase, ld, closed));
                        union(&mut cp, c, d);
                        break;
                    }
                }
            }
        }
        for k in 0..nk {
            per.corner[k] = find(&mut kp, k) as u32;
        }
        // Direction and phase of every curve relative to its root: a walk
        // from the roots over the matches.
        let mut rel: Vec<Option<(f64, f64)>> = vec![None; nc];
        for c in 0..nc {
            if find(&mut cp, c) == c {
                rel[c] = Some((1.0, 0.0));
            }
        }
        let wrap = |x: f64, len: f64, closed: bool| if closed { x.rem_euclid(len) } else { x };
        let mut changed = true;
        while changed {
            changed = false;
            for &(c, d, dir, phase, len, closed) in &matches {
                match (rel[c], rel[d]) {
                    (Some((dc, pc)), None) => {
                        rel[d] = Some((dir * dc, wrap(phase + dir * pc, len, closed)));
                        changed = true;
                    }
                    (None, Some((dd, pd))) => {
                        rel[c] = Some((dir * dd, wrap(dir * (pd - phase), len, closed)));
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
        let curves = dom.curves();
        for c in 0..nc {
            let root = find(&mut cp, c);
            if root == c && !matches.iter().any(|m| m.0 == c || m.1 == c) {
                continue;
            }
            let (dir, phase) = rel[c].unwrap_or((1.0, 0.0));
            let shift = sub(
                curves[c].curve.point_at(phase),
                curves[root].curve.point_at(0.0),
            );
            per.curve[c] = Some(CurveImage {
                root: root as u32,
                dir,
                phase,
                shift,
            });
        }
        per
    }
}

impl Periodic {
    /// The protecting balls that are images of each other: per curve after
    /// the first of its class, (node of the first, its image, the shift
    /// onto it), matched by position; per corner after the first of its
    /// class, (that corner, this one, no shift).
    pub fn links(
        &self,
        chains: &[Vec<(u32, f64)>],
        kind: &[VertexKind],
        pos: &[P3],
    ) -> Vec<(u32, u32, Option<P3>)> {
        let mut out = Vec::new();
        for (k, &root) in self.corner.iter().enumerate() {
            if root as usize != k {
                out.push((root, k as u32, None));
            }
        }
        // Every node of a chain, its corners included: a closed curve can
        // be anchored where its image has a plain sample.
        let nodes = |c: u32| -> Vec<u32> {
            let mut v: Vec<u32> = chains[c as usize].iter().map(|&(n, _)| n).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        for (c, link) in self.curve.iter().enumerate() {
            let Some(CurveImage { root, shift, .. }) = *link else {
                continue;
            };
            if root as usize == c {
                continue;
            }
            let rn = nodes(root);
            let tol = 1e3 * self.tol;
            for m in nodes(c as u32) {
                // Corners of one class are linked as corners, and a corner
                // never moves.
                let corner = |n: u32| match kind[n as usize] {
                    VertexKind::Corner(k) => Some(self.corner[k as usize]),
                    _ => None,
                };
                if let Some(&r) = rn
                    .iter()
                    .find(|&&r| dist(add(pos[r as usize], shift), pos[m as usize]) <= tol)
                {
                    match (corner(r), corner(m)) {
                        (Some(a), Some(b)) if a == b => {}
                        (None, None) => out.push((r, m, Some(shift))),
                        // A corner against a plain sample: equal radii,
                        // the corner stays where it is.
                        (Some(_), None) => out.push((r, m, Some(shift))),
                        _ => out.push((r, m, None)),
                    }
                }
            }
        }
        out
    }
}

/// Points by position, for matching the two sides of a pair.
pub(crate) struct PointIndex {
    cell: f64,
    map: FxHashMap<[i64; 3], Vec<usize>>,
}

impl PointIndex {
    pub fn new(tol: f64) -> PointIndex {
        PointIndex {
            cell: 4.0 * tol,
            map: FxHashMap::default(),
        }
    }

    fn key(&self, p: P3) -> [i64; 3] {
        p.map(|x| (x / self.cell).floor() as i64)
    }

    pub fn insert(&mut self, p: P3, id: usize) {
        let k = self.key(p);
        self.map.entry(k).or_default().push(id);
    }

    /// The point at `p` within `tol` (by `pos`), if any.
    pub fn find(&self, p: P3, pos: &dyn Fn(usize) -> P3, tol: f64) -> Option<usize> {
        let k = self.key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let c = [k[0] + dx, k[1] + dy, k[2] + dz];
                    if let Some(ids) = self.map.get(&c) {
                        if let Some(&i) = ids.iter().find(|&&i| dist(pos(i), p) <= tol) {
                            return Some(i);
                        }
                    }
                }
            }
        }
        None
    }
}
