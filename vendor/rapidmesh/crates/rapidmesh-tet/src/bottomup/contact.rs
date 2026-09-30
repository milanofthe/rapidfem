//! Contact wedges: where two bodies touch (along a line or at a point),
//! the region between them narrows to nothing, and no mesh fills that wedge
//! but with flat tets that span it from one body to the other. Those tets
//! take the material of one of the two bodies: the contact becomes a small
//! fillet of material as wide as the mesh is fine there, and the region
//! around no longer pinches to an edge of four faces.
//!
//! A tet is taken when its four corners lie on the two bodies (none inside
//! the region, so the gap there holds no point of its own) and it reaches
//! a contact through such tets: a point on both bodies where their faces
//! open by less than [`WEDGE_DEG`] (a trace on a substrate meets it at a
//! right angle and keeps its edges). A thin gap that never closes (the
//! dielectric between two plates) touches no contact and stays as it is.

use crate::conform::PointClass;
use crate::mesh3::{Complex, Face};
use rapidmesh_brep::Brep;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

type P3 = [f64; 3];

/// The widest opening, in degrees, between the faces of two bodies at a
/// point they share, for a contact whose wedge is filled.
pub const WEDGE_DEG: f64 = 30.0;

/// The faces of a tet, each opposite its vertex of the same index.
const FACE: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

/// Fills the contact wedges of `c`, whose points lie on the B-rep as
/// `classes` says; returns the number of tets taken and the faces (indices
/// into `c.faces`) that close the wedges.
pub(crate) fn fill(c: &mut Complex, brep: &Brep, classes: &[PointClass]) -> (usize, Vec<usize>) {
    // The bodies (regions but the outside) each point lies on.
    let bodies: Vec<SmallVec<[u32; 4]>> = (0..c.points.len())
        .map(|v| {
            let mut out: SmallVec<[u32; 4]> = match classes.get(v) {
                Some(PointClass::Vertex(i)) => brep.vertices[*i as usize]
                    .faces
                    .iter()
                    .flat_map(|f| brep.faces[f.0 as usize].regions.map(|r| r.0))
                    .collect(),
                Some(PointClass::Edge(e)) => brep.edges[*e as usize]
                    .coedges
                    .iter()
                    .flat_map(|ce| {
                        let f = brep.coedge(*ce).face;
                        brep.faces[f.0 as usize].regions.map(|r| r.0)
                    })
                    .collect(),
                Some(PointClass::Face(f)) => brep.faces[*f as usize]
                    .regions
                    .map(|r| r.0)
                    .into_iter()
                    .collect(),
                _ => SmallVec::new(),
            };
            out.retain(|r| *r != 0);
            out.sort_unstable();
            out.dedup();
            out
        })
        .collect();
    // A tet spanning two bodies has every corner on a body but its own
    // region and a corner on two of them: without one, nothing to fill (the
    // common case, found without the maps below).
    let candidate = |ti: usize| {
        let r = c.regions[ti];
        let on = |v: u32| bodies[v as usize].iter().filter(|&&b| b != r).count();
        c.tets[ti].iter().all(|&v| on(v) > 0) && c.tets[ti].iter().any(|&v| on(v) > 1)
    };
    {
        use rayon::prelude::*;
        if !(0..c.tets.len()).into_par_iter().any(candidate) {
            return (0, Vec::new());
        }
    }
    // Per point and pair of labels, the summed normals of the faces there,
    // turned into the first label.
    let mut facing: FxHashMap<(u32, u32, u32), P3> = FxHashMap::default();
    for f in &c.faces {
        let [a, b] = f.regions;
        if a == b {
            continue;
        }
        let q = f.tri.map(|v| c.points[v as usize]);
        let n = cross(sub(q[1], q[0]), sub(q[2], q[0]));
        for &v in &f.tri {
            for (into, from, s) in [(a, b, 1.0), (b, a, -1.0)] {
                let e = facing.entry((v, into, from)).or_insert([0.0; 3]);
                for k in 0..3 {
                    e[k] += s * n[k];
                }
            }
        }
    }
    let cos_wedge = (180.0 - WEDGE_DEG).to_radians().cos();
    // Whether bodies `a` and `b` meet at `v` in a wedge of region `r`.
    let wedge = |v: u32, r: u32, a: u32, b: u32| -> bool {
        let (Some(na), Some(nb)) = (facing.get(&(v, r, a)), facing.get(&(v, r, b))) else {
            return false;
        };
        let (la, lb) = (dot(*na, *na).sqrt(), dot(*nb, *nb).sqrt());
        la > 0.0 && lb > 0.0 && dot(*na, *nb) / (la * lb) < cos_wedge
    };
    let key = |t: [u32; 3]| {
        let mut k = t;
        k.sort_unstable();
        k
    };
    // The (at most two) tets on each face among those with every corner on
    // another body (the only ones a wedge takes); `NO` for none.
    const NO: usize = usize::MAX;
    let on_bodies = |ti: usize| {
        let r = c.regions[ti];
        c.tets[ti]
            .iter()
            .all(|&v| bodies[v as usize].iter().any(|&b| b != r))
    };
    let mut across: FxHashMap<[u32; 3], [usize; 2]> = FxHashMap::default();
    for (ti, t) in c.tets.iter().enumerate() {
        if !on_bodies(ti) {
            continue;
        }
        for f in FACE {
            let e = across.entry(key(f.map(|i| t[i]))).or_insert([NO; 2]);
            e[usize::from(e[0] != NO)] = ti;
        }
    }
    // The two bodies a tet of region `r` spans, with all its corners on
    // them and not all on one; and whether a corner lies on both.
    let spans = |ti: usize| -> Option<([u32; 2], bool)> {
        let r = c.regions[ti];
        let on: Vec<SmallVec<[u32; 4]>> = c.tets[ti]
            .iter()
            .map(|&v| {
                bodies[v as usize]
                    .iter()
                    .copied()
                    .filter(|&b| b != r)
                    .collect()
            })
            .collect();
        if on.iter().any(|b| b.is_empty()) {
            return None;
        }
        let common = on[0].iter().any(|b| on[1..].iter().all(|o| o.contains(b)));
        if common {
            return None;
        }
        let mut all: Vec<u32> = on.iter().flatten().copied().collect();
        all.sort_unstable();
        all.dedup();
        let pair = [all[0], all[1]];
        let contact = c.tets[ti].iter().zip(&on).any(|(&v, b)| {
            b.contains(&pair[0]) && b.contains(&pair[1]) && wedge(v, r, pair[0], pair[1])
        });
        Some((pair, contact))
    };
    let mut label: Vec<u32> = c.regions.clone();
    let mut taken: Vec<usize> = Vec::new();
    let mut seen: FxHashSet<usize> = FxHashSet::default();
    for start in 0..c.tets.len() {
        if !candidate(start) {
            continue;
        }
        let Some((pair, true)) = spans(start) else {
            continue;
        };
        if !seen.insert(start) {
            continue;
        }
        let region = c.regions[start];
        let mut queue = vec![start];
        while let Some(t) = queue.pop() {
            label[t] = pair[0];
            taken.push(t);
            let tv = c.tets[t];
            for f in FACE {
                for n in across
                    .get(&key(f.map(|i| tv[i])))
                    .copied()
                    .unwrap_or([NO; 2])
                {
                    if n != NO
                        && n != t
                        && c.regions[n] == region
                        && !seen.contains(&n)
                        && spans(n).is_some_and(|(p, _)| p == pair)
                    {
                        seen.insert(n);
                        queue.push(n);
                    }
                }
            }
        }
    }
    if taken.is_empty() {
        return (0, Vec::new());
    }
    // Every tet at a corner of a taken one, by corner: the other side of
    // each face of a taken tet is among them.
    let mut at_taken: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
    for &t in &taken {
        for v in c.tets[t] {
            at_taken.entry(v).or_default();
        }
    }
    for (ti, t) in c.tets.iter().enumerate() {
        for v in t {
            if let Some(ts) = at_taken.get_mut(v) {
                ts.push(ti);
            }
        }
    }
    let other_side = |t: usize, tri: [u32; 3]| -> Option<usize> {
        at_taken[&tri[0]]
            .iter()
            .copied()
            .find(|&n| n != t && tri.iter().all(|v| c.tets[n].contains(v)))
    };
    // The faces of the taken tets, afresh: none between equal labels, the
    // old face turned to its new labels, or a new one on the patch of the
    // body the tet joined.
    let mut index: FxHashMap<[u32; 3], usize> = FxHashMap::default();
    for (i, f) in c.faces.iter().enumerate() {
        index.insert(key(f.tri), i);
    }
    let p = |v: u32| c.points[v as usize];
    let mut drop: FxHashSet<usize> = FxHashSet::default();
    let mut added: Vec<Face> = Vec::new();
    // Whether each added face is new, through the wedge (else an old face
    // on a body, now between other labels).
    let mut closing: Vec<bool> = Vec::new();
    let mut done: FxHashSet<[u32; 3]> = FxHashSet::default();
    for &t in &taken {
        let tv = c.tets[t];
        for (i, f) in FACE.iter().enumerate() {
            let tri = f.map(|k| tv[k]);
            let k = key(tri);
            if !done.insert(k) {
                continue;
            }
            let other = other_side(t, tri);
            let (mine, theirs) = (label[t], other.map_or(0, |n| label[n]));
            let old = index.get(&k).copied();
            if let Some(o) = old {
                if c.faces[o].regions[0] == c.faces[o].regions[1] {
                    continue;
                }
                drop.insert(o);
            }
            if mine == theirs {
                continue;
            }
            // Wound so the normal points into regions[0].
            let n = cross(sub(p(tri[1]), p(tri[0])), sub(p(tri[2]), p(tri[0])));
            let into_mine = dot(n, sub(p(tv[i]), p(tri[0]))) > 0.0;
            let regions = if into_mine {
                [mine, theirs]
            } else {
                [theirs, mine]
            };
            let patch = old.map(|o| c.faces[o].patch).unwrap_or_else(|| {
                // A patch between the same two labels at a corner.
                tri.iter()
                    .find_map(|&v| match classes.get(v as usize) {
                        Some(PointClass::Face(f)) => {
                            let r = brep.faces[*f as usize].regions.map(|r| r.0);
                            (r.contains(&mine) && r.contains(&theirs)).then_some(*f)
                        }
                        _ => None,
                    })
                    .unwrap_or(u32::MAX)
            });
            added.push(Face {
                tri,
                regions,
                patch,
            });
            closing.push(old.is_none());
        }
    }
    let mut k = 0;
    c.faces.retain(|_| {
        k += 1;
        !drop.contains(&(k - 1))
    });
    let from = c.faces.len();
    c.faces.extend(added);
    c.regions = label;
    let faces = (from..c.faces.len())
        .filter(|&i| closing[i - from])
        .collect();
    (taken.len(), faces)
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
