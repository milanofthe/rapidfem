//! Spheres projected stereographically: from a point of the sphere onto
//! the plane through its centre square to that point. The projection keeps
//! angles and stretches lengths by one factor per place, so a face of a
//! sphere is meshed in one plane with its sizes stretched by that factor,
//! and its triangles keep their shapes on the sphere.
//!
//! The point projected from (the pole) lies off the face, as far from it as
//! the candidates go: opposite the face's mean direction, or in a hole of
//! it. A loop round the pole becomes the outer loop in the plane, which the
//! even-odd inside test takes as it comes.

use rapidmesh_brep::{Model, Surface};
use rapidmesh_geom::vec3::{add, cross, dot, perp, unit};

type P2 = [f64; 2];
type P3 = [f64; 3];

/// The pole stays at least this far (in degrees seen from the centre) from
/// every facet of the face.
const POLE_CLEARANCE_DEG: f64 = 15.0;

/// A sphere seen from its pole `n`, with `e1`, `e2` spanning the plane.
pub struct Stereo {
    c: P3,
    r: f64,
    e1: P3,
    e2: P3,
    n: P3,
}

impl Stereo {
    /// The projection for face `fi` on a sphere whose loops pass through
    /// `rings` (points), or none for a face that is no part of a sphere, has
    /// no loop, or leaves no room for the pole.
    pub fn of(model: &Model, fi: usize, rings: &[Vec<P3>]) -> Option<Stereo> {
        let (plc, brep) = (&model.plc, &model.brep);
        let face = &brep.faces[fi];
        let Surface::Sphere { center, radius, .. } = *brep.surface(face.surface) else {
            return None;
        };
        if rings.is_empty() {
            return None;
        }
        let dirs: Vec<P3> = face
            .facets
            .iter()
            .filter_map(|&t| {
                let p = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
                let m: P3 =
                    std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0 - center[k]);
                unit(m)
            })
            .collect();
        if dirs.is_empty() {
            return None;
        }
        let mut candidates: Vec<P3> = Vec::new();
        if let Some(m) = unit(dirs.iter().fold([0.0; 3], |s, d| add(s, *d))) {
            candidates.push(m.map(|x| -x));
        }
        for ring in rings {
            let m: P3 = std::array::from_fn(|k| {
                ring.iter().map(|p| p[k]).sum::<f64>() / ring.len().max(1) as f64 - center[k]
            });
            if let Some(m) = unit(m) {
                candidates.push(m);
                candidates.push(m.map(|x| -x));
            }
        }
        // The candidate farthest from the face (the least cosine to it).
        let (n, closest) = candidates
            .into_iter()
            .map(|n| {
                let c = dirs
                    .iter()
                    .map(|d| dot(*d, n))
                    .fold(f64::NEG_INFINITY, f64::max);
                (n, c)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        if closest > POLE_CLEARANCE_DEG.to_radians().cos() {
            return None;
        }
        let e1 = unit(perp(n))?;
        let e2 = cross(n, e1);
        Some(Stereo {
            c: center,
            r: radius,
            e1,
            e2,
            n,
        })
    }

    /// The chart point of a point of the sphere.
    pub fn to_chart(&self, p: P3) -> P2 {
        let d = [p[0] - self.c[0], p[1] - self.c[1], p[2] - self.c[2]];
        let u = unit(d).unwrap_or(self.n.map(|x| -x));
        let (x, y, z) = (dot(u, self.e1), dot(u, self.e2), dot(u, self.n));
        let s = 2.0 * self.r / (1.0 - z).max(1e-300);
        [x * s, y * s]
    }

    /// The point of the sphere at chart point `q`.
    pub fn lift(&self, q: P2) -> P3 {
        let w = [q[0] / (2.0 * self.r), q[1] / (2.0 * self.r)];
        let s = w[0] * w[0] + w[1] * w[1];
        let (x, y, z) = (
            2.0 * w[0] / (1.0 + s),
            2.0 * w[1] / (1.0 + s),
            (s - 1.0) / (1.0 + s),
        );
        std::array::from_fn(|k| {
            self.c[k] + self.r * (x * self.e1[k] + y * self.e2[k] + z * self.n[k])
        })
    }

    /// The factor a size on the sphere is scaled by in the chart at `q`
    /// (the chart's length of a unit length on the sphere).
    pub fn shrink(&self, q: P2) -> f64 {
        1.0 + (q[0] * q[0] + q[1] * q[1]) / (4.0 * self.r * self.r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_and_lift_invert_each_other() {
        let s = Stereo {
            c: [1.0, 2.0, 3.0],
            r: 2.0,
            e1: [1.0, 0.0, 0.0],
            e2: [0.0, 1.0, 0.0],
            n: [0.0, 0.0, 1.0],
        };
        for q in [[0.0, 0.0], [1.5, -0.5], [-3.0, 7.0]] {
            let p = s.lift(q);
            let d = ((p[0] - 1.0).powi(2) + (p[1] - 2.0).powi(2) + (p[2] - 3.0).powi(2)).sqrt();
            assert!((d - 2.0).abs() < 1e-12);
            let back = s.to_chart(p);
            assert!((back[0] - q[0]).abs() < 1e-9 && (back[1] - q[1]).abs() < 1e-9);
        }
        // Unit scale at the point opposite the pole; a short step in the
        // chart is that much shorter on the sphere.
        assert!((s.shrink([0.0, 0.0]) - 1.0).abs() < 1e-15);
        let (q, h) = ([1.5, -0.5], 1e-6);
        let a = s.lift(q);
        let b = s.lift([q[0] + h, q[1]]);
        let on_sphere =
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        assert!((h / on_sphere - s.shrink(q)).abs() < 1e-5);
    }
}
