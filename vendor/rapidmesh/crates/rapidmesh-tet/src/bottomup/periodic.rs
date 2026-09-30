//! Periodic faces bottom-up: a face `b` that is face `a` moved by a shift
//! takes `a`'s mesh moved by that shift.
//!
//! The edges the shifts carry onto each other form classes. The lowest
//! edge of a class is sampled; every other one takes the root's samples
//! mapped by its direction and phase (the point at arc `s` of the root,
//! moved, is the member's point at `phase + dir * s`). A split of a member
//! is a split of the root, and so of the whole class.

use crate::curve::{closest_arc, Curve, PolylineCurve};
use crate::mesh3::periodic::PeriodicPair;
use rapidmesh_brep::Brep;

type P3 = [f64; 3];

/// How an edge is the image of the root of its class.
#[derive(Clone, Copy, Debug)]
struct Image {
    root: usize,
    dir: f64,
    phase: f64,
    closed: bool,
}

/// The classes of the edges under the periodic pairs.
#[derive(Default)]
pub(crate) struct Classes {
    image: Vec<Option<Image>>,
    /// Per face: the face it copies and the shift from that face onto it.
    pub copy_of: Vec<Option<(usize, P3)>>,
}

fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

fn dist(a: P3, b: P3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

impl Classes {
    /// The classes of `pairs` over the edges of `brep` sampled along
    /// `curves`, the edges of each face given by `face_edges`; points
    /// closer than `tol` are one.
    pub(crate) fn new(
        brep: &Brep,
        curves: &[Option<PolylineCurve>],
        face_edges: &[Vec<usize>],
        pairs: &[PeriodicPair],
        tol: f64,
    ) -> Classes {
        let ne = brep.edges.len();
        let mut out = Classes {
            image: vec![None; ne],
            copy_of: vec![None; brep.faces.len()],
        };
        if pairs.is_empty() {
            return out;
        }
        let closed = |e: usize| brep.edges[e].ends[0] == brep.edges[e].ends[1];
        let mut parent: Vec<usize> = (0..ne).collect();
        // (edge, its image, direction, phase): the point at `s` of the
        // first, moved, is the image's point at `phase + dir * s`.
        let mut matches: Vec<(usize, usize, f64, f64)> = Vec::new();
        for pp in pairs {
            out.copy_of[pp.b as usize] = Some((pp.a as usize, pp.shift));
            for &c in &face_edges[pp.a as usize] {
                let Some(cc) = &curves[c] else { continue };
                let lc = cc.length();
                let at = |f: f64| add(cc.point_at(f * lc), pp.shift);
                for &d in &face_edges[pp.b as usize] {
                    let Some(cd) = &curves[d] else { continue };
                    let ld = cd.length();
                    if (lc - ld).abs() > tol || closed(c) != closed(d) {
                        continue;
                    }
                    let phases: Vec<f64> = if closed(d) {
                        let samples: Vec<(f64, P3)> = (0..=256)
                            .map(|i| {
                                let s = ld * i as f64 / 256.0;
                                (s, cd.point_at(s))
                            })
                            .collect();
                        vec![closest_arc(cd, &samples, at(0.0)).rem_euclid(ld)]
                    } else {
                        vec![0.0, ld]
                    };
                    let on =
                        |arc: f64| cd.point_at(if closed(d) { arc.rem_euclid(ld) } else { arc });
                    let found = phases.iter().find_map(|&phase| {
                        [1.0, -1.0].into_iter().find_map(|dir: f64| {
                            [0.0, 0.25, 0.5, 0.75, 1.0]
                                .iter()
                                .all(|&f| dist(on(phase + dir * f * lc), at(f)) <= tol)
                                .then_some((dir, phase))
                        })
                    });
                    if let Some((dir, phase)) = found {
                        matches.push((c, d, dir, phase));
                        let (rc, rd) = (find(&mut parent, c), find(&mut parent, d));
                        if rc != rd {
                            parent[rc.max(rd)] = rc.min(rd);
                        }
                        break;
                    }
                }
            }
        }
        // Direction and phase of every edge of a class to its root.
        let mut rel: Vec<Option<(f64, f64)>> = vec![None; ne];
        for (e, r) in rel.iter_mut().enumerate() {
            if find(&mut parent, e) == e {
                *r = Some((1.0, 0.0));
            }
        }
        let len = |e: usize| curves[e].as_ref().map_or(0.0, |c| c.length());
        let wrap = |x: f64, e: usize| if closed(e) { x.rem_euclid(len(e)) } else { x };
        let mut changed = true;
        while changed {
            changed = false;
            for &(c, d, dir, phase) in &matches {
                match (rel[c], rel[d]) {
                    (Some((dc, pc)), None) => {
                        rel[d] = Some((dir * dc, wrap(phase + dir * pc, d)));
                        changed = true;
                    }
                    (None, Some((dd, pd))) => {
                        rel[c] = Some((dir * dd, wrap(dir * (pd - phase), c)));
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
        for e in 0..ne {
            let root = find(&mut parent, e);
            if root == e && !matches.iter().any(|m| m.0 == e || m.1 == e) {
                continue;
            }
            let (dir, phase) = rel[e].unwrap_or((1.0, 0.0));
            out.image[e] = Some(Image {
                root,
                dir,
                phase,
                closed: closed(e),
            });
        }
        out
    }

    /// The root of edge `e`'s class and the root's arc for `e`'s arc `s`
    /// (`e` itself for an edge alone).
    pub(crate) fn to_root(&self, e: usize, s: f64, len: f64) -> (usize, f64) {
        match self.image[e] {
            Some(im) => {
                let r = im.dir * (s - im.phase);
                (im.root, if im.closed { r.rem_euclid(len) } else { r })
            }
            None => (e, s),
        }
    }

    /// Every edge of `root`'s class, the root first.
    pub(crate) fn members(&self, root: usize) -> Vec<usize> {
        let mut out = vec![root];
        out.extend(
            self.image
                .iter()
                .enumerate()
                .filter(|(e, im)| *e != root && im.is_some_and(|im| im.root == root))
                .map(|(e, _)| e),
        );
        out
    }

    /// Whether any edge is in a class.
    pub(crate) fn any(&self) -> bool {
        self.image.iter().any(|x| x.is_some())
    }

    /// The samples of every member of a class from its root's: each
    /// member's own corner (on a closed edge, where its image is no corner
    /// of the root) joins the root's samples first. `spaced` sorts and
    /// thins an edge's samples (arc lengths strictly between its ends).
    pub(crate) fn sync(
        &self,
        arcs: &mut [Vec<f64>],
        curves: &[Option<PolylineCurve>],
        spaced: &dyn Fn(&mut Vec<f64>, f64),
    ) {
        let len = |e: usize| curves[e].as_ref().map_or(0.0, |c| c.length());
        for (e, im) in self.image.iter().enumerate() {
            let Some(im) = im else { continue };
            if im.root != e && im.closed {
                let (r, s) = self.to_root(e, 0.0, len(e));
                arcs[r].push(s);
            }
        }
        for (e, im) in self.image.iter().enumerate() {
            if im.is_some_and(|im| im.root == e) {
                spaced(&mut arcs[e], len(e));
            }
        }
        for (e, im) in self.image.iter().enumerate() {
            let Some(im) = im else { continue };
            if im.root == e {
                continue;
            }
            let l = len(e);
            let mut mapped: Vec<f64> = arcs[im.root]
                .iter()
                .map(|&s| {
                    let x = im.phase + im.dir * s;
                    if im.closed {
                        x.rem_euclid(l)
                    } else {
                        x
                    }
                })
                .collect();
            spaced(&mut mapped, l);
            arcs[e] = mapped;
        }
    }
}
