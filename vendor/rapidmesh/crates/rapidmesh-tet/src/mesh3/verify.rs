//! Invariants of a [`Complex`], checked from scratch on the output alone.
//!
//! Tests assert an all-zero [`Report`]; the corpus diagnostics can read the
//! same counts. Orientation and sidedness use exact predicates, so a report
//! never depends on rounding.

use super::Complex;
use geometry_predicates::orient3d;
use rustc_hash::FxHashMap;

/// What [`check`] found. Every count is zero for a valid complex.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// Tets checked.
    pub tets: usize,
    /// Tets not positively oriented.
    pub inverted: usize,
    /// Mesh facets shared by more than two tets.
    pub overfull_facets: usize,
    /// Facets where the label changes but no face exists.
    pub missing_faces: usize,
    /// Faces on no mesh facet, duplicated, or where the label does not
    /// change and the face is not a sheet face of that label.
    pub extra_faces: usize,
    /// Faces whose region pair or orientation disagrees with the adjacent
    /// tets' labels.
    pub wrong_sides: usize,
    /// Faces without a patch.
    pub unpatched_faces: usize,
    /// Edges of a region's boundary with a face count other than two.
    pub open_or_nonmanifold_edges: usize,
    /// Protected curve segments that are not edges of the mesh.
    pub missing_feature_edges: usize,
    /// Volume per region, ascending by region.
    pub volumes: Vec<(u32, f64)>,
    /// The first few offending boundary edges `(region, a, b, faces)`.
    pub bad_edges: Vec<(u32, u32, u32, u32)>,
    /// The first few unpatched faces.
    pub bad_faces: Vec<[u32; 3]>,
    /// The first few missing feature edges.
    pub bad_segments: Vec<[u32; 2]>,
}

/// How many offenders of each kind a report lists.
const EXAMPLES: usize = 8;

impl Report {
    /// True when every invariant holds.
    pub fn ok(&self) -> bool {
        self.inverted == 0
            && self.overfull_facets == 0
            && self.missing_faces == 0
            && self.extra_faces == 0
            && self.wrong_sides == 0
            && self.unpatched_faces == 0
            && self.open_or_nonmanifold_edges == 0
            && self.missing_feature_edges == 0
    }

    /// Volume of `region` (0 when absent).
    pub fn volume(&self, region: u32) -> f64 {
        self.volumes
            .iter()
            .find(|v| v.0 == region)
            .map_or(0.0, |v| v.1)
    }
}

/// Faces of a tet opposite each corner, wound so the corner lies on the
/// positive side (the kernel's convention).
const FACE_LOCAL: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]];

fn sorted3(f: [u32; 3]) -> [u32; 3] {
    let mut s = f;
    s.sort_unstable();
    s
}

/// Tets per block of the parallel sums: blocks summed in a fixed order
/// make the volumes independent of the thread count.
const BLOCK: usize = 1 << 16;

/// Checks every invariant of the complex.
pub fn check(c: &Complex) -> Report {
    use rayon::prelude::*;
    let p = |i: u32| c.points[i as usize];
    let mut rep = Report {
        tets: c.tets.len(),
        ..Report::default()
    };

    // Orientation and volumes, per region.
    let nreg = c
        .regions
        .iter()
        .copied()
        .max()
        .map_or(0, |r| r as usize + 1);
    let blocks: Vec<(usize, Vec<f64>, Vec<bool>)> = c
        .tets
        .par_chunks(BLOCK)
        .zip(c.regions.par_chunks(BLOCK))
        .map(|(tets, regions)| {
            let (mut inverted, mut vol, mut seen) = (0, vec![0.0; nreg], vec![false; nreg]);
            for (t, &r) in tets.iter().zip(regions) {
                let (a, b, cc, d) = (p(t[0]), p(t[1]), p(t[2]), p(t[3]));
                if !(orient3d(a, b, cc, d) > 0.0) {
                    inverted += 1;
                }
                let e = |x: [f64; 3]| [x[0] - a[0], x[1] - a[1], x[2] - a[2]];
                let (u, v, w) = (e(b), e(cc), e(d));
                let det = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                    + u[2] * (v[0] * w[1] - v[1] * w[0]);
                vol[r as usize] += det.abs() / 6.0;
                seen[r as usize] = true;
            }
            (inverted, vol, seen)
        })
        .collect();
    let mut vol = vec![0.0; nreg];
    let mut seen = vec![false; nreg];
    for (inverted, v, s) in &blocks {
        rep.inverted += inverted;
        for r in 0..nreg {
            vol[r] += v[r];
            seen[r] |= s[r];
        }
    }
    rep.volumes = (0..nreg)
        .filter(|&r| seen[r])
        .map(|r| (r as u32, vol[r]))
        .collect();

    // Facet incidences (facet, tet, local corner), sorted so the tets of one
    // facet are adjacent, in ascending tet order.
    // Written in place: a parallel collect would hold its pieces and the
    // whole at once.
    let mut incidences: Vec<([u32; 3], u32, u8)> = vec![([0; 3], 0, 0); 4 * c.tets.len()];
    incidences
        .par_chunks_mut(4)
        .zip(c.tets.par_iter())
        .enumerate()
        .for_each(|(ti, (out, t))| {
            for (i, fl) in FACE_LOCAL.iter().enumerate() {
                out[i] = (sorted3([t[fl[0]], t[fl[1]], t[fl[2]]]), ti as u32, i as u8);
            }
        });
    incidences.par_sort_unstable();
    let facets = || incidences.chunk_by(|x, y| x.0 == y.0);
    rep.overfull_facets = facets().filter(|g| g.len() > 2).count();

    // Faces by vertex set.
    let mut faces: FxHashMap<[u32; 3], Vec<usize>> = FxHashMap::default();
    for (fi, f) in c.faces.iter().enumerate() {
        faces.entry(sorted3(f.tri)).or_default().push(fi);
        if f.patch == u32::MAX {
            rep.unpatched_faces += 1;
            if rep.bad_faces.len() < EXAMPLES {
                rep.bad_faces.push(f.tri);
            }
        }
    }

    // The label on each side of every facet, and the matching faces.
    for inc in facets() {
        let key = inc[0].0;
        let label = |k: usize| inc.get(k).map_or(0, |&(_, t, _)| c.regions[t as usize]);
        let (l0, l1) = (label(0), label(1));
        let fs = faces.get(&key).map_or(&[][..], |v| v.as_slice());
        if l0 != l1 {
            if fs.is_empty() {
                rep.missing_faces += 1;
                continue;
            }
            if fs.len() > 1 {
                rep.extra_faces += fs.len() - 1;
            }
        } else if fs.is_empty() {
            continue;
        } else {
            // Same label on both sides: only one sheet face of that label.
            let f = &c.faces[fs[0]];
            if f.regions != [l0, l0] || fs.len() > 1 {
                rep.extra_faces += fs.len();
                continue;
            }
        }
        // Sidedness: the tet labelled regions[0] lies on the positive side of
        // the face's winding.
        let f = &c.faces[fs[0]];
        let (a, b, cc) = (p(f.tri[0]), p(f.tri[1]), p(f.tri[2]));
        let mut ok = true;
        for &(_, t, i) in inc.iter() {
            let opp = p(c.tets[t as usize][i as usize]);
            let side = orient3d(a, b, cc, opp);
            let lab = c.regions[t as usize];
            let expect = if l0 == l1 {
                true // a sheet: both sides carry the same region
            } else if side > 0.0 {
                lab == f.regions[0]
            } else {
                lab == f.regions[1]
            };
            ok &= expect;
        }
        // A hull facet: the outside (label 0) is the other side.
        if inc.len() == 1 && l0 != l1 {
            let (_, t, i) = inc[0];
            let opp = p(c.tets[t as usize][i as usize]);
            let side = orient3d(a, b, cc, opp);
            let outside = if side > 0.0 {
                f.regions[1]
            } else {
                f.regions[0]
            };
            ok &= outside == 0;
        }
        if !ok {
            rep.wrong_sides += 1;
        }
    }
    // Faces on no mesh facet.
    for (key, fs) in &faces {
        if incidences.binary_search_by(|x| x.0.cmp(key)).is_err() {
            rep.extra_faces += fs.len();
        }
    }

    // Each region's boundary must be closed and edge-manifold: every edge of
    // its (non-sheet) faces used by exactly two of them.
    let mut edge_use: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
    for f in &c.faces {
        if f.regions[0] == f.regions[1] {
            continue;
        }
        for &r in &f.regions {
            if r == 0 {
                continue;
            }
            for k in 0..3 {
                let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
                *edge_use.entry((r, a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
    }
    rep.open_or_nonmanifold_edges = edge_use.values().filter(|&&n| n != 2).count();
    let mut bad: Vec<(u32, u32, u32, u32)> = edge_use
        .iter()
        .filter(|(_, &n)| n != 2)
        .map(|(&(r, a, b), &n)| (r, a, b, n))
        .collect();
    bad.sort_unstable();
    bad.truncate(EXAMPLES);
    rep.bad_edges = bad;

    // Protected segments must be mesh edges: the tets tick off the segments
    // they carry.
    let wanted: rustc_hash::FxHashSet<(u32, u32)> = c
        .feature_edges
        .iter()
        .map(|(e, _)| (e[0].min(e[1]), e[0].max(e[1])))
        .collect();
    let found: rustc_hash::FxHashSet<(u32, u32)> = c
        .tets
        .par_iter()
        .fold(rustc_hash::FxHashSet::default, |mut found, t| {
            for a in 0..4 {
                for b in a + 1..4 {
                    let e = (t[a].min(t[b]), t[a].max(t[b]));
                    if wanted.contains(&e) {
                        found.insert(e);
                    }
                }
            }
            found
        })
        .reduce(rustc_hash::FxHashSet::default, |mut x, y| {
            x.extend(y);
            x
        });
    let missing: Vec<[u32; 2]> = c
        .feature_edges
        .iter()
        .filter(|(e, _)| !found.contains(&(e[0].min(e[1]), e[0].max(e[1]))))
        .map(|(e, _)| *e)
        .collect();
    rep.missing_feature_edges = missing.len();
    rep.bad_segments = missing.into_iter().take(EXAMPLES).collect();
    rep
}

#[cfg(test)]
mod tests {
    use super::super::{Complex, Face, VertexKind};
    use super::*;

    /// Two tets sharing a facet, labelled 1 and 2, with the interface face
    /// and the six hull faces.
    fn two_tets() -> Complex {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        // Orient both positively.
        let mut t0 = [0u32, 1, 2, 3];
        let mut t1 = [0u32, 1, 2, 4];
        for t in [&mut t0, &mut t1] {
            let q = |i: u32| points[i as usize];
            if orient3d(q(t[0]), q(t[1]), q(t[2]), q(t[3])) < 0.0 {
                t.swap(2, 3);
            }
        }
        let mut c = Complex {
            points,
            kinds: vec![VertexKind::Volume; 5],
            tets: vec![t0, t1],
            regions: vec![1, 2],
            faces: Vec::new(),
            feature_edges: vec![([0, 1], 0)],
        };
        // Faces: every facet where the label changes, wound with the
        // regions[0] tet on the positive side.
        for (ti, t) in c.tets.clone().iter().enumerate() {
            for fl in FACE_LOCAL {
                let tri = [t[fl[0]], t[fl[1]], t[fl[2]]];
                let key = sorted3(tri);
                let shared = key == sorted3([0, 1, 2]);
                if shared && ti == 1 {
                    continue; // the interface once
                }
                let other = if shared { 2 } else { 0 };
                c.faces.push(Face {
                    tri,
                    regions: [c.regions[ti], other],
                    patch: 0,
                });
            }
        }
        c
    }

    #[test]
    fn a_valid_complex_passes() {
        let c = two_tets();
        let r = check(&c);
        assert!(r.ok(), "{r:?}");
        assert!((r.volume(1) - 1.0 / 6.0).abs() < 1e-15);
        assert!((r.volume(2) - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn broken_complexes_are_caught() {
        let mut c = two_tets();
        c.faces.pop();
        assert_eq!(check(&c).missing_faces, 1);

        let mut c = two_tets();
        c.faces[0].regions.swap(0, 1);
        assert_eq!(check(&c).wrong_sides, 1);

        let mut c = two_tets();
        let t = c.tets[0];
        c.tets[0] = [t[1], t[0], t[2], t[3]];
        assert_eq!(check(&c).inverted, 1);

        let mut c = two_tets();
        c.regions[1] = 1; // interface face now separates equal labels
        assert!(check(&c).extra_faces >= 1);

        let mut c = two_tets();
        c.faces[0].patch = u32::MAX;
        assert_eq!(check(&c).unpatched_faces, 1);

        let mut c = two_tets();
        c.feature_edges.push(([3, 4], 0)); // not an edge of either tet
        assert_eq!(check(&c).missing_feature_edges, 1);
    }
}
