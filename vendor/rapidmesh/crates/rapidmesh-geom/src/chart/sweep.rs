//! Swept surfaces unrolled: an extrusion by arc length along its profile
//! and height along its axis, a tube round an open path by angle round
//! the path and arc length along it (parallel-transport frames).

use crate::TubePath;
use rapidmesh_exact::vector::{
    add, angle_about, cross, dot, perp, scale, sub, turn, unit, wrap_pm, V3,
};
use std::sync::Arc;

/// The profile of an extrusion by arc length: an extrusion is developable,
/// unrolled by arc length along its profile and height along its axis
/// without distortion. A closed profile goes once round (angle `2 pi s /
/// L`), an open one half round, so its ends stay apart.
pub(super) struct Profile {
    /// Parameters of the profile and the arc lengths there, increasing.
    t: Vec<f64>,
    s: Vec<f64>,
    pub(super) closed: bool,
}

impl Profile {
    const SAMPLES: usize = 1024;

    pub(super) fn of(curve: &crate::NurbsCurve) -> Option<Profile> {
        let (t0, t1) = curve.domain();
        if !(t1 > t0) {
            return None;
        }
        let t: Vec<f64> = (0..=Self::SAMPLES)
            .map(|i| t0 + (t1 - t0) * i as f64 / Self::SAMPLES as f64)
            .collect();
        let mut s = vec![0.0];
        for w in t.windows(2) {
            let (a, b) = (curve.eval(w[0]), curve.eval(w[1]));
            s.push(s[s.len() - 1] + (b[0] - a[0]).hypot(b[1] - a[1]));
        }
        let len = s[s.len() - 1];
        if !(len > 0.0) {
            return None;
        }
        let (a, b) = (curve.eval(t0), curve.eval(t1));
        let closed = (b[0] - a[0]).hypot(b[1] - a[1]) <= 1e-9 * len;
        Some(Profile { t, s, closed })
    }

    pub(super) fn len(&self) -> f64 {
        self.s[self.s.len() - 1]
    }

    /// Linear interpolation from one table to the other.
    fn map(from: &[f64], to: &[f64], x: f64) -> f64 {
        let i = from.partition_point(|&v| v <= x).clamp(1, from.len() - 1);
        let (a, b) = (from[i - 1], from[i]);
        let w = if b > a {
            ((x - a) / (b - a)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        to[i - 1] + w * (to[i] - to[i - 1])
    }

    /// The arc length at parameter `t`, on round a closed profile.
    pub(super) fn arc(&self, t: f64) -> f64 {
        if !self.closed {
            return Self::map(&self.t, &self.s, t);
        }
        let (lo, hi) = (self.t[0], self.t[self.t.len() - 1]);
        let turns = ((t - lo) / (hi - lo)).floor();
        Self::map(&self.t, &self.s, t - turns * (hi - lo)) + turns * self.len()
    }

    pub(super) fn param(&self, s: f64) -> f64 {
        let s = if self.closed {
            s.rem_euclid(self.len())
        } else {
            s
        };
        Self::map(&self.s, &self.t, s)
    }
}

/// An open tube path with a frame at each node that turns with it
/// (parallel transport): arc length and angle round the path, continuous
/// across its kinks.
pub(super) struct Frames {
    path: Arc<TubePath>,
    /// The arc length at each node.
    at: Vec<f64>,
    /// The normal of the plane through each node that halves the turn there
    /// (the tangent at the ends): a point between two such planes belongs to
    /// the segment between, at the share of its distances to them.
    plane: Vec<V3>,
    /// The frame in that plane at each node.
    frame: Vec<(V3, V3)>,
}

impl Frames {
    pub(super) fn of(path: &Arc<TubePath>) -> Option<Frames> {
        let p = &path.pts;
        let m = p.len() - 1;
        let tangent: Vec<V3> = p.windows(2).filter_map(|w| unit(sub(w[1], w[0]))).collect();
        if tangent.len() != m {
            return None;
        }
        let mut at = vec![0.0];
        for w in p.windows(2) {
            at.push(at[at.len() - 1] + dot(sub(w[1], w[0]), sub(w[1], w[0])).sqrt());
        }
        // A closed path turns the face into a torus: no seam of one chart.
        if dot(sub(p[m], p[0]), sub(p[m], p[0])).sqrt() <= 1e-9 * at[m] {
            return None;
        }
        let plane: Vec<V3> = (0..=m)
            .map(|i| match i {
                0 => tangent[0],
                _ if i == m => tangent[m - 1],
                _ => unit(add(tangent[i - 1], tangent[i])).unwrap_or(tangent[i]),
            })
            .collect();
        let mut frame = Vec::with_capacity(m + 1);
        let mut n = unit(perp(plane[0]))?;
        for i in 0..=m {
            if i > 0 {
                n = turn(n, plane[i - 1], plane[i]);
            }
            let x = unit(sub(n, scale(plane[i], dot(n, plane[i]))))?;
            n = x;
            frame.push((x, cross(plane[i], x)));
        }
        Some(Frames {
            path: path.clone(),
            at,
            plane,
            frame,
        })
    }

    /// Angle round the path and arc length along it of `q`.
    pub(super) fn angle_length(&self, q: V3) -> (f64, f64) {
        let p = &self.path.pts;
        let m = p.len() - 1;
        let mut i = self.path.closest_segment(q);
        let side = |k: usize| dot(sub(q, p[k]), self.plane[k]);
        for _ in 0..m {
            if i > 0 && side(i) < 0.0 {
                i -= 1;
            } else if i + 1 < m && side(i + 1) > 0.0 {
                i += 1;
            } else {
                break;
            }
        }
        let (d0, d1) = (side(i), side(i + 1));
        let t = if d0 - d1 > 0.0 { d0 / (d0 - d1) } else { 0.5 };
        // Past the path's ends the ends' segments go on.
        let t = match (i == 0, i + 1 == m) {
            (true, true) => t,
            (true, false) => t.min(1.0),
            (false, true) => t.max(0.0),
            (false, false) => t.clamp(0.0, 1.0),
        };
        let angle = |k: usize| {
            let w = sub(q, p[k]);
            let (x, y) = self.frame[k];
            angle_about(w, x, y)
        };
        let (a0, a1) = (angle(i), angle(i + 1));
        let a = a0 + t.clamp(0.0, 1.0) * wrap_pm(a1 - a0);
        (a, self.at[i] + t * (self.at[i + 1] - self.at[i]))
    }

    /// The point at angle `a` and arc length `s` at `radius` from the path
    /// (near the tube; its closest point is on it).
    pub(super) fn point(&self, a: f64, s: f64, radius: f64) -> V3 {
        let p = &self.path.pts;
        let m = p.len() - 1;
        let i = self.at.partition_point(|&x| x <= s).clamp(1, m) - 1;
        let t = (s - self.at[i]) / (self.at[i + 1] - self.at[i]).max(1e-300);
        let axis: V3 = std::array::from_fn(|k| p[i][k] + t * (p[i + 1][k] - p[i][k]));
        let dir = |k: usize| {
            let (x, y) = self.frame[k];
            add(scale(x, a.cos()), scale(y, a.sin()))
        };
        let tc = t.clamp(0.0, 1.0);
        let d = add(scale(dir(i), 1.0 - tc), scale(dir(i + 1), tc));
        let d = unit(d).unwrap_or(dir(i));
        add(axis, scale(d, radius))
    }
}
