//! Exact point-vs-solid classification by the winding number: the signed
//! crossings of a ray with the solid's oriented surface. Unlike parity it
//! stays right for a solid whose shells overlap (an import of several
//! shells, a sweep crossing itself): a point in the overlap has winding 2.
//!
//! The representative point is implicit (a sub-triangle barycenter); the ray
//! is the segment from it to an explicit target far outside the solid.
//! Degenerate configurations (segment through an edge/vertex, target on a
//! plane) are detected exactly and resolved by retrying with a different
//! target — the set of bad targets is measure-zero, so a deterministic
//! pseudo-random target sequence escapes after a try or two.

use crate::tri::Tri;
use rapidmesh_exact::{orient2d, orient3d, orient3d_explicit, Point3, Prepared3, Sign};

/// Where a point lies relative to a closed solid surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Strictly inside the solid.
    Inside,
    /// Strictly outside the solid.
    Outside,
    /// Exactly on the surface; `same_normal` compares the queried facet's
    /// orientation with the coincident solid facet's orientation.
    Boundary {
        /// True if the coincident facets face the same way.
        same_normal: bool,
    },
}

/// Does the segment (p, q) cross the interior of `tri`?
///
/// `None` means a degenerate configuration that requires a different target
/// `q` (segment through an edge/vertex of the triangle, or `q` on its plane).
/// A `p` on this triangle's plane gives no crossing: the caller either knows
/// `p` is off the surface, or counts the triangles through `p` on its own.
pub fn segment_crosses_triangle(p: &Prepared3, q: [f64; 3], tri: &Tri) -> Option<bool> {
    Some(signed_crossing(p, q, tri)? != 0)
}

/// Side of the outward-oriented `tri` that `d` lies behind: [`orient3d`] is
/// positive for a point on the side opposite the normal of a
/// counter-clockwise triangle, the inside of a solid.
const BEHIND: Sign = Sign::Positive;

/// The crossing of the segment (p, q) with `tri`: +1 leaving the solid
/// through it (from behind to in front), -1 entering, 0 none; `None` as in
/// [`segment_crosses_triangle`].
fn signed_crossing(p: &Prepared3, q: [f64; 3], tri: &Tri) -> Option<i64> {
    let [a, b, c] = tri.v;
    let sp = orient3d_explicit(a, b, c, p).expect("valid");
    if sp == Sign::Zero {
        return Some(0);
    }
    let pq = Point3::Explicit(q);
    let sq = orient3d(&tri.point(0), &tri.point(1), &tri.point(2), &pq).expect("valid");
    if sq == Sign::Zero {
        return None;
    }
    if sp == sq {
        return Some(0);
    }
    // The sides of the segment's edges, each orient3d(p, q, u, v) negated
    // (it equals -orient3d(q, u, v, p)): only their agreement counts.
    let s1 = orient3d_explicit(q, a, b, p).expect("valid");
    let s2 = orient3d_explicit(q, b, c, p).expect("valid");
    let s3 = orient3d_explicit(q, c, a, p).expect("valid");
    if s1 == Sign::Zero || s2 == Sign::Zero || s3 == Sign::Zero {
        return None;
    }
    if !(s1 == s2 && s2 == s3) {
        return Some(0);
    }
    Some(if sp == BEHIND { 1 } else { -1 })
}

/// Per-triangle bounding boxes of a solid's tessellation, padded by a fat
/// safety margin against the f64 approximation error of implicit query
/// points (relative error ~1e-15; callers pad with ~1e-6 of the scene
/// diagonal). Built once per solid, they collapse the linear exact-predicate
/// scans of [`on_solid_boundary`] and [`point_inside_solid`] to the few
/// triangles actually near the query point or ray.
pub struct TriBoxes {
    boxes: Vec<([f64; 3], [f64; 3])>,
    /// A uniform grid over the boxes: `cells[c]` lists the triangles whose
    /// box overlaps cell `c`, so a query reads only its cells.
    lo: [f64; 3],
    size: [f64; 3],
    dims: [usize; 3],
    cells: Vec<Vec<u32>>,
}

impl TriBoxes {
    /// Boxes of `tris`, each padded by `pad` on every side.
    pub fn build(tris: &[Tri], pad: f64) -> TriBoxes {
        let boxes: Vec<([f64; 3], [f64; 3])> = tris
            .iter()
            .map(|t| {
                let mut lo = [f64::MAX; 3];
                let mut hi = [f64::MIN; 3];
                for v in &t.v {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k] - pad);
                        hi[k] = hi[k].max(v[k] + pad);
                    }
                }
                (lo, hi)
            })
            .collect();
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for (blo, bhi) in &boxes {
            for k in 0..3 {
                lo[k] = lo[k].min(blo[k]);
                hi[k] = hi[k].max(bhi[k]);
            }
        }
        // About one triangle per cell, at most 64 cells a side.
        let n = (boxes.len() as f64).cbrt().ceil().clamp(1.0, 64.0) as usize;
        let dims = if boxes.is_empty() { [1; 3] } else { [n; 3] };
        let size: [f64; 3] =
            std::array::from_fn(|k| ((hi[k] - lo[k]) / dims[k] as f64).max(f64::MIN_POSITIVE));
        let mut grid = TriBoxes {
            boxes,
            lo,
            size,
            dims,
            cells: vec![Vec::new(); dims[0] * dims[1] * dims[2]],
        };
        for i in 0..grid.boxes.len() {
            let (blo, bhi) = grid.boxes[i];
            let (a, b) = (grid.cell_of(blo), grid.cell_of(bhi));
            for x in a[0]..=b[0] {
                for y in a[1]..=b[1] {
                    for z in a[2]..=b[2] {
                        let c = grid.index([x, y, z]);
                        grid.cells[c].push(i as u32);
                    }
                }
            }
        }
        grid
    }

    fn cell_of(&self, p: [f64; 3]) -> [usize; 3] {
        std::array::from_fn(|k| {
            let f = ((p[k] - self.lo[k]) / self.size[k]).floor();
            (f.max(0.0) as usize).min(self.dims[k] - 1)
        })
    }

    fn index(&self, c: [usize; 3]) -> usize {
        (c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]
    }

    fn contains(&self, i: usize, p: [f64; 3]) -> bool {
        let (lo, hi) = self.boxes[i];
        (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k])
    }

    fn overlaps(&self, i: usize, lo: [f64; 3], hi: [f64; 3]) -> bool {
        let (blo, bhi) = self.boxes[i];
        (0..3).all(|k| blo[k] <= hi[k] && bhi[k] >= lo[k])
    }

    /// The triangles whose box contains `p`, ascending.
    fn at(&self, p: [f64; 3]) -> Vec<usize> {
        if self.boxes.is_empty()
            || (0..3).any(|k| {
                p[k] < self.lo[k] || p[k] > self.lo[k] + self.size[k] * self.dims[k] as f64
            })
        {
            return Vec::new();
        }
        let mut out: Vec<usize> = self.cells[self.index(self.cell_of(p))]
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| self.contains(i, p))
            .collect();
        out.sort_unstable();
        out
    }

    /// The triangles whose box overlaps the box `[lo, hi]`, ascending.
    fn near(&self, lo: [f64; 3], hi: [f64; 3]) -> Vec<usize> {
        if self.boxes.is_empty() {
            return Vec::new();
        }
        let (a, b) = (self.cell_of(lo), self.cell_of(hi));
        let mut out: Vec<usize> = Vec::new();
        for x in a[0]..=b[0] {
            for y in a[1]..=b[1] {
                for z in a[2]..=b[2] {
                    out.extend(
                        self.cells[self.index([x, y, z])]
                            .iter()
                            .map(|&i| i as usize)
                            .filter(|&i| self.overlaps(i, lo, hi)),
                    );
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// The solid facet whose (closed) area contains `p`, if `p` lies exactly on
/// the solid's surface. `rep` is the f64 approximation of `p` (the padded
/// boxes absorb its error).
pub fn on_solid_boundary(
    p: &Prepared3,
    rep: [f64; 3],
    solid: &[Tri],
    boxes: &TriBoxes,
) -> Option<usize> {
    boxes.at(rep).into_iter().find(|&i| {
        let t = &solid[i];
        orient3d_explicit(t.v[0], t.v[1], t.v[2], p) == Some(Sign::Zero) && {
            let (axis, orientation) = t.projection_axis();
            t.contains_coplanar(p.point(), axis, orientation)
        }
    })
}

/// True if two coplanar triangles face the same way.
pub fn coplanar_same_normal(t1: &Tri, t2: &Tri) -> bool {
    let (axis, s1) = t1.projection_axis();
    let s2 = orient2d(&t2.point(0), &t2.point(1), &t2.point(2), axis)
        .expect("explicit points are always valid");
    debug_assert_ne!(
        s2,
        Sign::Zero,
        "triangles must be coplanar and non-degenerate"
    );
    s1 == s2
}

/// Is `p` (not on the surface) inside the closed solid (winding above 0)?
///
/// `rep` is the f64 approximation of `p`; `bbox` is the solid's (or scene's)
/// bounding box; targets are placed outside it. The ray runs mostly along +x
/// (with a pseudo-random lateral jitter that dodges degenerate hits), so its
/// bounding box stays a narrow column and the padded triangle boxes reject
/// almost every exact segment test.
pub fn point_inside_solid(
    p: &Prepared3,
    rep: [f64; 3],
    solid: &[Tri],
    boxes: &TriBoxes,
    bbox: ([f64; 3], [f64; 3]),
) -> bool {
    winding(p, rep, bbox, solid, boxes, |_| true).0 > 0
}

/// Ray targets tried before a cast gives up.
pub const RAY_TARGETS: u64 = 32;

/// The `k`-th target of a ray cast from `rep`: beyond the bounding box
/// in +x, jittered deterministically in y and z by up to 1 % of its diagonal
/// (a new `k` escapes a degenerate configuration of the previous one).
pub fn ray_target(bbox: ([f64; 3], [f64; 3]), rep: [f64; 3], k: u64) -> [f64; 3] {
    let (lo, hi) = bbox;
    let diag = (0..3).map(|k| hi[k] - lo[k]).fold(1.0_f64, f64::max);
    let mut s = (k + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut frac = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let qx = hi[0] + diag * (0.5 + frac());
    let qy = rep[1] + diag * 0.02 * (frac() - 0.5);
    let qz = rep[2] + diag * 0.02 * (frac() - 0.5);
    [qx, qy, qz]
}

/// The winding number around `p` along the first generic ray: the signed
/// crossings with the triangles of `solid` near it, and the target used. `accept(q)` may reject a target
/// (one on a plane through `p` the caller needs a side of). A target is
/// dropped whenever any candidate is degenerate for it, so the order of the
/// candidates is irrelevant.
fn winding(
    p: &Prepared3,
    rep: [f64; 3],
    bbox: ([f64; 3], [f64; 3]),
    solid: &[Tri],
    boxes: &TriBoxes,
    accept: impl Fn(&Point3) -> bool,
) -> (i64, Point3) {
    'targets: for k in 0..RAY_TARGETS {
        let [qx, qy, qz] = ray_target(bbox, rep, k);
        let q = Point3::explicit(qx, qy, qz);
        if !accept(&q) {
            continue;
        }
        let seg_lo: [f64; 3] = std::array::from_fn(|d| rep[d].min([qx, qy, qz][d]));
        let seg_hi: [f64; 3] = std::array::from_fn(|d| rep[d].max([qx, qy, qz][d]));
        let mut w = 0i64;
        for i in boxes.near(seg_lo, seg_hi) {
            match signed_crossing(p, [qx, qy, qz], &solid[i]) {
                None => continue 'targets,
                Some(c) => w += c,
            }
        }
        return (w, q);
    }
    panic!("no generic ray target found in {RAY_TARGETS} attempts");
}

/// The winding numbers of the solid just in front of and just behind `own`,
/// for `p` inside the facet `own` lies on (a facet of this very solid). A ray
/// into one side counts that side; stepping across `p` adds +1 for every
/// solid triangle through `p` facing like `own` and -1 for one facing the
/// other way (coincident shells).
pub fn winding_beside(
    p: &Prepared3,
    rep: [f64; 3],
    own: &Tri,
    solid: &[Tri],
    boxes: &TriBoxes,
    bbox: ([f64; 3], [f64; 3]),
) -> (i64, i64) {
    let (a, b, c) = (own.point(0), own.point(1), own.point(2));
    let side = |q: &Point3| orient3d(&a, &b, &c, q).expect("valid");
    let (w, q) = winding(p, rep, bbox, solid, boxes, |q| side(q) != Sign::Zero);
    let step: i64 = boxes
        .at(rep)
        .into_iter()
        .filter(|&i| {
            let t = &solid[i];
            orient3d_explicit(t.v[0], t.v[1], t.v[2], p) == Some(Sign::Zero) && {
                let (axis, orientation) = t.projection_axis();
                t.contains_coplanar(p.point(), axis, orientation)
            }
        })
        .map(|i| {
            if coplanar_same_normal(own, &solid[i]) {
                1
            } else {
                -1
            }
        })
        .sum();
    if side(&q) == BEHIND {
        (w - step, w)
    } else {
        (w, w + step)
    }
}

/// Full placement of `p` (interior representative of a facet of `own`, with
/// f64 approximation `rep`) relative to the closed solid `other`.
pub fn classify(
    p: &Prepared3,
    rep: [f64; 3],
    own_facet: &Tri,
    other: &[Tri],
    boxes: &TriBoxes,
    bbox: ([f64; 3], [f64; 3]),
) -> Placement {
    match on_solid_boundary(p, rep, other, boxes) {
        Some(j) => Placement::Boundary {
            same_normal: coplanar_same_normal(own_facet, &other[j]),
        },
        None => {
            if point_inside_solid(p, rep, other, boxes, bbox) {
                Placement::Inside
            } else {
                Placement::Outside
            }
        }
    }
}
