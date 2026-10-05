//! The element kernels of the mesher, one of each: the face table of a tet,
//! circumcenters and circumradii, the smallest dihedral of a tet and the
//! smallest angle of a triangle, and a float ordered for the heaps.

use rapidmesh_exact::vector::{cross, dist, dot, len, sub, V3};
use std::cmp::Ordering;

/// The vertices of face `i` of a positive tet, turned so the tet's vertex
/// `i` lies on their positive side.
pub(crate) const TET_FACES: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]];

/// Circumradius of triangle `(a, b, c)` in 3-space: `|ab||bc||ca| / (4 * area)`,
/// computed from the cross-product area. Returns `INFINITY` for a degenerate
/// (collinear) triangle.
pub(crate) fn circumradius(a: V3, b: V3, c: V3) -> f64 {
    let (ab, bc, ca) = (dist(a, b), dist(b, c), dist(c, a));
    // 2 * area, from the cross product of two edges.
    let area2 = len(cross(sub(b, a), sub(c, a)));
    if area2 <= 1e-300 {
        f64::INFINITY
    } else {
        ab * bc * ca / (2.0 * area2)
    }
}

/// The center of the sphere through the four corners of a tet (none when
/// they are flat).
pub(crate) fn tet_circumcenter(p: [V3; 4]) -> Option<V3> {
    tet_circumsphere(p).map(|s| s.0)
}

/// The center and radius of the sphere through the four corners of a tet
/// (none when they are flat), from the corners in the order of their
/// places: a tet numbered any way gets them to the bit.
pub(crate) fn tet_circumsphere(p: [V3; 4]) -> Option<(V3, f64)> {
    let mut p = p;
    p.sort_by(|x, y| {
        x[0].total_cmp(&y[0])
            .then(x[1].total_cmp(&y[1]))
            .then(x[2].total_cmp(&y[2]))
    });
    let [a, b, c, d] = p;
    let (u, v, w) = (sub(b, a), sub(c, a), sub(d, a));
    let det = 2.0 * dot(u, cross(v, w));
    if !(det.abs() > 0.0) {
        return None;
    }
    let (uu, vv, ww) = (dot(u, u), dot(v, v), dot(w, w));
    let (vw, wu, uv) = (cross(v, w), cross(w, u), cross(u, v));
    let o: V3 = std::array::from_fn(|k| a[k] + (uu * vw[k] + vv * wu[k] + ww * uv[k]) / det);
    o.iter().all(|x| x.is_finite()).then_some((o, dist(o, a)))
}

/// Circumradius over shortest edge of a tet (none when it is flat).
pub(crate) fn radius_edge(p: [V3; 4]) -> Option<f64> {
    let c = tet_circumcenter(p)?;
    let mut lmin = f64::MAX;
    for i in 0..4 {
        for j in i + 1..4 {
            lmin = lmin.min(dist(p[i], p[j]));
        }
    }
    (lmin > 0.0).then(|| dist(c, p[0]) / lmin)
}

/// The volume of a tet (unsigned).
pub(crate) fn tet_volume(p: [V3; 4]) -> f64 {
    let [a, b, c, d] = p;
    dot(sub(b, a), cross(sub(c, a), sub(d, a))).abs() / 6.0
}

/// A tet whose six times volume is below this share of its longest edge
/// cubed is flat (its corners in one plane up to rounding).
const FLAT_VOLUME: f64 = 1e-12;

/// Smallest dihedral angle of a tet, in degrees (0 for a flat or
/// degenerate tet): at each edge the angle between the two faces through
/// it, from their outward normals (four cross products for the six edges),
/// the arccosine of the largest cosine, so one `acos` per tet.
pub(crate) fn tet_min_dihedral(p: [V3; 4]) -> f64 {
    min_dihedral(p, false)
}

/// [`tet_min_dihedral`] of a tet positive by `orient3d` (its fourth corner
/// below the plane of the first three), 0 for one that is not: its volume
/// beyond the rounding of a flat one has the sign (the exact orientation
/// then has it too).
pub(crate) fn positive_min_dihedral(p: [V3; 4]) -> f64 {
    min_dihedral(p, true)
}

fn min_dihedral(p: [V3; 4], positive: bool) -> f64 {
    // Flat to rounding: the normals' sides below are the signs of rounding
    // noise there, and could make a flat tet look a good one.
    let l2 = (0..4)
        .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
        .map(|(i, j)| dot(sub(p[i], p[j]), sub(p[i], p[j])))
        .fold(0.0f64, f64::max);
    let l3 = l2 * l2.sqrt();
    let vol6 = dot(cross(sub(p[1], p[0]), sub(p[2], p[0])), sub(p[3], p[0]));
    if !(vol6.abs() > FLAT_VOLUME * l3) || (positive && vol6 > 0.0) {
        return 0.0;
    }
    // The normal of the face opposite each corner, turned away from it.
    let mut n = [[0.0; 3]; 4];
    let mut len2 = [0.0; 4];
    for k in 0..4 {
        let (a, b, c) = (p[(k + 1) % 4], p[(k + 2) % 4], p[(k + 3) % 4]);
        let m = cross(sub(b, a), sub(c, a));
        let s = dot(m, sub(p[k], a));
        let l = dot(m, m);
        if !(s != 0.0 && l > 0.0) {
            return 0.0;
        }
        n[k] = if s > 0.0 { m.map(|x| -x) } else { m };
        len2[k] = l;
    }
    // The edge off corners k and l lies on the faces opposite them.
    let mut cos = -1.0f64;
    for (k, l) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        cos = cos.max(-dot(n[k], n[l]) / (len2[k] * len2[l]).sqrt());
    }
    cos.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Smallest angle of a triangle, in degrees (0 when degenerate).
pub(crate) fn tri_min_angle(p: [V3; 3]) -> f64 {
    let mut m = f64::INFINITY;
    for k in 0..3 {
        let (u, v) = (sub(p[(k + 1) % 3], p[k]), sub(p[(k + 2) % 3], p[k]));
        let d = (dot(u, u) * dot(v, v)).sqrt();
        if !(d > 0.0) {
            return 0.0;
        }
        m = m.min((dot(u, v) / d).clamp(-1.0, 1.0).acos().to_degrees());
    }
    m
}

/// A float with a total order, for the heaps.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ordered(pub f64);

impl PartialEq for Ordered {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The corners of a wall's rectangle, flat but for rounding (the tet of
    /// a chip layout's side wall): a dihedral angle of 0, not a good tet.
    #[test]
    fn a_tet_flat_to_rounding_has_no_angle() {
        let p = [
            [39.546225, -41.063945, 5.625],
            [40.305085, -40.305085, 4.365],
            [40.305085, -40.305085, 5.625],
            [39.546225, -41.063945, 4.365],
        ];
        assert_eq!(tet_min_dihedral(p), 0.0);
        let good = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert!(tet_min_dihedral(good) > 50.0);
    }

    #[test]
    fn dihedral_of_known_tets() {
        let r = 1.0 / 3f64.sqrt();
        let regular = [[r, r, r], [r, -r, -r], [-r, r, -r], [-r, -r, r]];
        assert!((tet_min_dihedral(regular) - (1.0f64 / 3.0).acos().to_degrees()).abs() < 1e-9);
        let corner = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let want = (1.0 / 3f64.sqrt()).acos().to_degrees();
        assert!((tet_min_dihedral(corner) - want).abs() < 1e-9);
        let flipped = [corner[1], corner[0], corner[2], corner[3]];
        assert!((tet_min_dihedral(flipped) - want).abs() < 1e-9);
        let flat = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert_eq!(tet_min_dihedral(flat), 0.0);
    }

    /// The face-normal form agrees with the angle between the half-planes
    /// at each edge, on random and on flat tets.
    #[test]
    fn dihedral_matches_the_edge_form() {
        fn by_edges(p: [V3; 4]) -> f64 {
            let mut cos = -1.0f64;
            for (i, j, k, l) in [
                (0, 1, 2, 3),
                (0, 2, 1, 3),
                (0, 3, 1, 2),
                (1, 2, 0, 3),
                (1, 3, 0, 2),
                (2, 3, 0, 1),
            ] {
                let e = sub(p[j], p[i]);
                let (n1, n2) = (cross(e, sub(p[k], p[i])), cross(e, sub(p[l], p[i])));
                cos = cos.max(dot(n1, n2) / (dot(n1, n1) * dot(n2, n2)).sqrt());
            }
            cos.clamp(-1.0, 1.0).acos().to_degrees()
        }
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut r = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for i in 0..20_000 {
            let mut p: [V3; 4] = std::array::from_fn(|_| [r(), r(), r()]);
            if i % 4 == 0 {
                // Nearly flat: the fourth corner close to the plane of the others.
                let t = [r().abs(), r().abs()];
                p[3] = std::array::from_fn(|k| {
                    p[0][k] + t[0] * (p[1][k] - p[0][k]) + t[1] * (p[2][k] - p[0][k])
                });
                p[3][2] += 1e-3 * r();
            }
            let (a, b) = (tet_min_dihedral(p), by_edges(p));
            assert!((a - b).abs() < 1e-6 * (1.0 + b), "{p:?}: {a} vs {b}");
        }
    }
}
