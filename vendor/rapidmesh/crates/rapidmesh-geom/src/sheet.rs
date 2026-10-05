//! Exact booleans of flat sheets lying in one plane.
//!
//! Both sheets go into the plane's own frame (its coordinates themselves
//! where the plane is square to an axis, so every input point comes back
//! bit for bit) and through the conformal planar arrangement the scene
//! uses, which cuts each against the other exactly. A piece of one sheet
//! the other covers is a sub-triangle both have, so a piece is kept or left
//! by whether the other sheet has it. The result is a flat sheet again: a
//! flat facet per connected piece with its outer and hole loops, the input
//! corners and the crossings of the outlines as corners, and the curves the
//! inputs declare (a disc's rim stays a circle wherever a piece of it is
//! left).

use crate::faceted::Faceted;
use crate::surface::Surface;
use rapidmesh_csg::{arrange_facets, BoolOp, PlanarFacet, PlanarInput, Tri, VertexPool};
use rapidmesh_exact::vector::{cross, dot, newell, normalize, sub, V3};
use rapidmesh_exact::Point3;
use rustc_hash::{FxHashMap, FxHashSet};

/// How far off the target's plane, relative to the sheets' extent, a point
/// of a sheet in a tilted plane may lie and still be in it.
const IN_PLANE_REL: f64 = 1e-9;

/// The exact boolean `op` of the flat sheets `a` and `b`, which lie in one
/// plane. The result has `a`'s plane, frame and winding.
pub fn sheet_boolean(a: &Faceted, b: &Faceted, op: BoolOp) -> Result<Faceted, String> {
    let carrier = flat_plane(a, "the target")?;
    flat_plane(b, "a tool")?;
    let frame = PlaneFrame::of(a, b)?;
    // Each input point by where it lands in the frame, to come back as it
    // was.
    let mut original: FxHashMap<[u64; 2], V3> = FxHashMap::default();
    let mut lift = |p: V3| -> [f64; 3] {
        let q = frame.to2d(p);
        original.entry(q.map(bits)).or_insert(p);
        [q[0], q[1], 0.0]
    };
    let mut input: Vec<PlanarInput> = Vec::new();
    for f in [a, b] {
        for fl in &f.flats {
            input.push(PlanarInput {
                boundary: PlanarFacet::with_holes(
                    fl.facet.outer.iter().map(|&p| lift(p)).collect(),
                    fl.facet
                        .holes
                        .iter()
                        .map(|h| h.iter().map(|&p| lift(p)).collect())
                        .collect(),
                ),
                helpers: f.tris[fl.tris.clone()]
                    .iter()
                    .map(|t| Tri::new(lift(t.v[0]), lift(t.v[1]), lift(t.v[2])))
                    .collect(),
            });
        }
    }
    let arr = arrange_facets(&input).map_err(|e| format!("sheet boolean: {e:?}"))?;
    let mut pool = VertexPool::default();
    let mut subs: [Vec<[u32; 3]>; 2] = [Vec::new(), Vec::new()];
    for (k, ft) in arr.facets.iter().enumerate() {
        let side = usize::from(k >= a.flats.len());
        for t in &ft.triangles {
            subs[side].push(t.map(|i| pool.insert(ft.vertices[i].clone()) as u32));
        }
    }
    let key = |t: &[u32; 3]| {
        let mut k = *t;
        k.sort_unstable();
        k
    };
    let has: [FxHashSet<[u32; 3]>; 2] = [
        subs[0].iter().map(key).collect(),
        subs[1].iter().map(key).collect(),
    ];
    let kept: Vec<[u32; 3]> = match op {
        BoolOp::Difference => subs[0]
            .iter()
            .filter(|t| !has[1].contains(&key(t)))
            .copied()
            .collect(),
        BoolOp::Intersection => subs[0]
            .iter()
            .filter(|t| has[1].contains(&key(t)))
            .copied()
            .collect(),
        BoolOp::Union => subs[0]
            .iter()
            .chain(subs[1].iter().filter(|t| !has[0].contains(&key(t))))
            .copied()
            .collect(),
    };
    // The points: an input point as it was, a crossing in the frame.
    let at2: Vec<[f64; 2]> = pool
        .verts
        .iter()
        .map(|p| {
            let q = p.approx().expect("arrangement points are valid");
            [q[0], q[1]]
        })
        .collect();
    let input_point: Vec<bool> = pool
        .verts
        .iter()
        .zip(&at2)
        .map(|(p, q)| matches!(p, Point3::Explicit(_)) && original.contains_key(&q.map(bits)))
        .collect();
    let at3: Vec<V3> = at2
        .iter()
        .map(|q| {
            original
                .get(&q.map(bits))
                .copied()
                .unwrap_or_else(|| frame.to3d(*q))
        })
        .collect();
    // Every piece wound as `a` is in the frame.
    let turn = |t: &[u32; 3]| {
        let [p, q, r] = t.map(|i| at2[i as usize]);
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    };
    let want = a.tris.iter().map(|t| {
        let [p, q, r] = t.v.map(|v| frame.to2d(v));
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    });
    let positive = want.sum::<f64>() > 0.0;
    let tris: Vec<[u32; 3]> = kept
        .into_iter()
        .filter(|t| {
            at3[t[0] as usize] != at3[t[1] as usize]
                && at3[t[1] as usize] != at3[t[2] as usize]
                && at3[t[2] as usize] != at3[t[0] as usize]
        })
        .map(|t| {
            if (turn(&t) > 0.0) == positive {
                t
            } else {
                [t[0], t[2], t[1]]
            }
        })
        .collect();
    let mut out = Faceted::new();
    out.frame = a.frame;
    out.curves = a.curves.iter().chain(&b.curves).cloned().collect();
    out.features = a.features.iter().chain(&b.features).copied().collect();
    let s = out.add_surface(carrier);
    let corners: FxHashSet<[u64; 3]> = a
        .corners
        .iter()
        .chain(&b.corners)
        .map(|p| p.map(bits))
        .collect();
    let mut on_outline: FxHashSet<u32> = FxHashSet::default();
    for piece in pieces(&tris, &at2, positive)? {
        let to3 = |lp: &[u32]| lp.iter().map(|&i| at3[i as usize]).collect::<Vec<V3>>();
        for lp in std::iter::once(&piece.outer).chain(&piece.holes) {
            on_outline.extend(lp.iter().copied());
        }
        let helpers: Vec<Tri> = piece
            .tris
            .iter()
            .map(|t| {
                let [p, q, r] = t.map(|i| at3[i as usize]);
                Tri::new(p, q, r)
            })
            .collect();
        out.push_flat(
            PlanarFacet::with_holes(
                to3(&piece.outer),
                piece.holes.iter().map(|h| to3(h)).collect(),
            ),
            &helpers,
            s,
        );
    }
    // Corners: the inputs' on the outline left, and the crossings of the
    // outlines, where one sheet's edge leaves for the other's.
    let mut on: Vec<u32> = on_outline.into_iter().collect();
    on.sort_unstable();
    out.corners = on
        .into_iter()
        .map(|i| i as usize)
        .filter(|&i| !input_point[i] || corners.contains(&at3[i].map(bits)))
        .map(|i| at3[i])
        .collect();
    Ok(out)
}

/// `+ 0.0` folds -0.0 into 0.0, which it equals.
fn bits(x: f64) -> u64 {
    (x + 0.0).to_bits()
}

/// The plane of the flat sheet `f` (`what` names it in the error).
fn flat_plane(f: &Faceted, what: &str) -> Result<Option<Surface>, String> {
    let covered: usize = f.flats.iter().map(|fl| fl.tris.len()).sum();
    if f.flats.is_empty() || covered != f.tris.len() {
        return Err(format!("{what} is no flat sheet"));
    }
    let s = f.surfaces[f.flats[0].surface as usize].clone();
    if !s.as_ref().is_some_and(Surface::is_plane) {
        return Err(format!("{what} is no flat sheet"));
    }
    Ok(s)
}

/// The 2D frame of the plane two sheets lie in.
enum PlaneFrame {
    /// The plane square to axis `k` at that coordinate: the other two
    /// coordinates, in turn.
    Axis { k: usize, at: f64 },
    /// A tilted plane: an origin and two axes in it.
    Tilted { o: V3, u: V3, v: V3 },
}

impl PlaneFrame {
    fn of(a: &Faceted, b: &Faceted) -> Result<PlaneFrame, String> {
        let points: Vec<V3> = [a, b]
            .iter()
            .flat_map(|f| f.tris.iter().flat_map(|t| t.v))
            .collect();
        let first = points[0];
        if let Some(k) = (0..3).find(|&k| points.iter().all(|p| bits(p[k]) == bits(first[k]))) {
            return Ok(PlaneFrame::Axis { k, at: first[k] });
        }
        let n = normalize(newell_of(a));
        let o = a.tris[0].v[0];
        let edge = a.tris[0]
            .v
            .iter()
            .map(|&p| sub(p, o))
            .fold([0.0; 3], |m, d| if dot(d, d) > dot(m, m) { d } else { m });
        let u = normalize(sub(edge, scale(n, dot(edge, n))));
        let v = cross(n, u);
        let (lo, hi) = rapidmesh_exact::vector::bbox(points.iter());
        let extent = dot(sub(hi, lo), sub(hi, lo)).sqrt();
        if let Some(p) = points
            .iter()
            .find(|&&p| dot(sub(p, o), n).abs() > IN_PLANE_REL * extent)
        {
            return Err(format!(
                "the sheets are not in one plane (a point at {p:?} is off it)"
            ));
        }
        Ok(PlaneFrame::Tilted { o, u, v })
    }

    fn to2d(&self, p: V3) -> [f64; 2] {
        match *self {
            PlaneFrame::Axis { k, .. } => [p[(k + 1) % 3], p[(k + 2) % 3]],
            PlaneFrame::Tilted { o, u, v } => {
                let d = sub(p, o);
                [dot(d, u), dot(d, v)]
            }
        }
    }

    fn to3d(&self, q: [f64; 2]) -> V3 {
        match *self {
            PlaneFrame::Axis { k, at } => {
                let mut p = [0.0; 3];
                p[k] = at;
                p[(k + 1) % 3] = q[0];
                p[(k + 2) % 3] = q[1];
                p
            }
            PlaneFrame::Tilted { o, u, v } => {
                std::array::from_fn(|i| o[i] + q[0] * u[i] + q[1] * v[i])
            }
        }
    }
}

fn scale(a: V3, s: f64) -> V3 {
    a.map(|x| x * s)
}

/// The area vector of `f`'s triangles.
fn newell_of(f: &Faceted) -> V3 {
    f.tris.iter().fold([0.0; 3], |s, t| {
        let n = newell(&t.v);
        [s[0] + n[0], s[1] + n[1], s[2] + n[2]]
    })
}

/// A connected piece of a sheet: its triangles, its outer loop and its
/// holes, as point indices.
struct Piece {
    tris: Vec<[u32; 3]>,
    outer: Vec<u32>,
    holes: Vec<Vec<u32>>,
}

/// The pieces of the triangles `tris` (wound alike, counterclockwise in
/// the frame where `positive`) over the points `at`: connected across
/// their edges, each with its loops traced along the edges only one
/// triangle has. Where loops touch at a point, a loop turns off along the
/// first edge clockwise from the one it came in by, so the region stays on
/// one side of it and the loops stay simple.
fn pieces(tris: &[[u32; 3]], at: &[[f64; 2]], positive: bool) -> Result<Vec<Piece>, String> {
    let mut parent: Vec<usize> = (0..tris.len()).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    let mut by_edge: FxHashMap<(u32, u32), usize> = FxHashMap::default();
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            by_edge.insert((t[k], t[(k + 1) % 3]), ti);
        }
    }
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            if let Some(&o) = by_edge.get(&(t[(k + 1) % 3], t[k])) {
                let (x, y) = (find(&mut parent, ti), find(&mut parent, o));
                parent[x.max(y)] = x.min(y);
            }
        }
    }
    // The outline: edges with no twin, from each point.
    let mut from: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    let mut owner: FxHashMap<(u32, u32), usize> = FxHashMap::default();
    let mut halves: Vec<(u32, u32)> = Vec::new();
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (p, q) = (t[k], t[(k + 1) % 3]);
            if !by_edge.contains_key(&(q, p)) {
                from.entry(p).or_default().push(q);
                owner.insert((p, q), ti);
                halves.push((p, q));
            }
        }
    }
    let angle = |p: u32, q: u32| {
        let (a, b) = (at[p as usize], at[q as usize]);
        let s = if positive { 1.0 } else { -1.0 };
        (s * (b[1] - a[1])).atan2(b[0] - a[0])
    };
    let mut used: FxHashSet<(u32, u32)> = FxHashSet::default();
    let mut loops: Vec<(usize, Vec<u32>)> = Vec::new();
    for &(p0, q0) in &halves {
        if used.contains(&(p0, q0)) {
            continue;
        }
        let mut lp = vec![p0];
        let (mut p, mut q) = (p0, q0);
        loop {
            used.insert((p, q));
            if q == p0 {
                break;
            }
            lp.push(q);
            let back = angle(q, p);
            let next = from[&q]
                .iter()
                .copied()
                .filter(|&r| !used.contains(&(q, r)))
                .min_by(|&r, &s| {
                    let cw = |r: u32| (back - angle(q, r)).rem_euclid(std::f64::consts::TAU);
                    cw(r).total_cmp(&cw(s))
                })
                .ok_or("a sheet boolean left an open outline")?;
            (p, q) = (q, next);
        }
        loops.push((find(&mut parent, owner[&(p0, q0)]), lp));
    }
    let area = |lp: &[u32]| {
        let s: f64 = (0..lp.len())
            .map(|i| {
                let (a, b) = (at[lp[i] as usize], at[lp[(i + 1) % lp.len()] as usize]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum();
        if positive {
            s
        } else {
            -s
        }
    };
    let root: Vec<usize> = (0..tris.len()).map(|i| find(&mut parent, i)).collect();
    let mut roots = root.clone();
    roots.sort_unstable();
    roots.dedup();
    roots
        .into_iter()
        .map(|r| {
            let mine: Vec<&Vec<u32>> = loops
                .iter()
                .filter(|(c, _)| *c == r)
                .map(|(_, l)| l)
                .collect();
            let outer: Vec<&&Vec<u32>> = mine.iter().filter(|l| area(l) > 0.0).collect();
            let [outer] = outer[..] else {
                return Err("a sheet boolean left a piece touching itself at a point".to_string());
            };
            Ok(Piece {
                tris: (0..tris.len())
                    .filter(|&t| root[t] == r)
                    .map(|t| tris[t])
                    .collect(),
                outer: (*outer).clone(),
                holes: mine
                    .iter()
                    .filter(|l| area(l) <= 0.0)
                    .map(|l| (*l).clone())
                    .collect(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim::{sheet_disk, sheet_rect};
    use rapidmesh_exact::vector::Affine;

    fn area(f: &Faceted) -> f64 {
        f.tris
            .iter()
            .map(|t| 0.5 * dot(newell(&t.v), newell(&t.v)).sqrt())
            .sum()
    }

    fn plate() -> Faceted {
        sheet_rect([0.0, 0.0, 1.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0])
    }

    #[test]
    fn a_disc_out_of_a_plate_leaves_a_round_hole() {
        let disc = sheet_disk([2.0, 2.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 48);
        let f = sheet_boolean(&plate(), &disc, BoolOp::Difference).unwrap();
        assert!((area(&f) - (16.0 - area(&disc))).abs() < 1e-12);
        assert_eq!(f.flats.len(), 1);
        assert_eq!(f.flats[0].facet.holes.len(), 1);
        assert_eq!(f.flats[0].facet.holes[0].len(), 48);
        // The rim is the disc's circle, its points as they were.
        assert_eq!(f.curves.len(), 1);
        let ring: FxHashSet<[u64; 3]> = disc
            .tris
            .iter()
            .flat_map(|t| t.v)
            .map(|p| p.map(bits))
            .collect();
        assert!(f.flats[0].facet.holes[0]
            .iter()
            .all(|p| ring.contains(&p.map(bits))));
        // Wound as the plate is.
        assert!(dot(newell_of(&f), newell_of(&plate())) > 0.0);
    }

    #[test]
    fn a_strip_across_cuts_a_plate_in_two() {
        let strip = sheet_rect([1.5, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 6.0, 0.0]);
        let f = sheet_boolean(&plate(), &strip, BoolOp::Difference).unwrap();
        assert_eq!(f.flats.len(), 2);
        assert!((area(&f) - 12.0).abs() < 1e-12);
        // The plate's corners and the four crossings of the outlines.
        assert_eq!(f.corners.len(), 8);
        for x in [1.5, 2.5] {
            for y in [0.0, 4.0] {
                assert!(f.corners.contains(&[x, y, 1.0]));
            }
        }
        let common = sheet_boolean(&plate(), &strip, BoolOp::Intersection).unwrap();
        assert!((area(&common) - 4.0).abs() < 1e-12);
        let union = sheet_boolean(&plate(), &strip, BoolOp::Union).unwrap();
        assert!((area(&union) - 18.0).abs() < 1e-12);
        assert_eq!(union.flats.len(), 1);
    }

    #[test]
    fn a_tilted_plane_takes_its_booleans_too() {
        let m = Affine::rotation([0.0; 3], [1.0, 1.0, 0.0], 0.7).unwrap();
        let disc = sheet_disk([2.0, 2.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 32);
        let f = sheet_boolean(
            &plate().transformed(&m),
            &disc.transformed(&m),
            BoolOp::Difference,
        )
        .unwrap();
        assert!((area(&f) - (16.0 - area(&disc))).abs() < 1e-9);
        assert_eq!(f.flats[0].facet.holes.len(), 1);
    }

    #[test]
    fn sheets_in_two_planes_are_refused() {
        let other = sheet_rect([0.0, 0.0, 1.5], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]);
        assert!(sheet_boolean(&plate(), &other, BoolOp::Difference).is_err());
    }
}
