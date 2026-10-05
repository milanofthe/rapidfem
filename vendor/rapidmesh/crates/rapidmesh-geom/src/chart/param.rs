//! B-spline faces charted in their own parameters, made isotropic on
//! average: the parameters mapped by the root of the face's mean first
//! fundamental form, so a step in the chart is about as long on the face
//! whichever way it goes. Where the face's metric strays from that mean by
//! more than [`MAX_STRETCH`] in some direction, each parameter is first
//! mapped so the face's two speeds keep in step: half the log of their
//! ratio split into a part of each parameter, the map's slope the
//! exponential of its part. A speed varying along its own parameter
//! (crowded knots, the sections of a loft) becomes a length along it, one
//! varying across (a fan, a strip tapering) a conformal map as for a
//! surface of revolution; the size then varies over the chart as the
//! face's scale does. Where the metric still strays (parameter lines that
//! cross at a slant varying over the face), the face has no such chart.

use crate::Surface;
use rapidmesh_exact::vector::{dot, V2, V3};

/// The largest ratio of the lengths a unit step in two directions of the
/// chart takes on the face: the height field chart of a face tilted by 60
/// degrees keeps to the same.
pub(super) const MAX_STRETCH: f64 = 2.0;

/// The samples the metric is measured at, at most.
const SAMPLES: usize = 64;

/// The stations of a parameter's map, and the lines across the face its
/// part of the speeds' ratio is averaged over at each.
const STATIONS: usize = 32;
const ACROSS: usize = 8;

pub(super) struct Param {
    /// Each parameter's map onto the length along it.
    warp: [Warp; 2],
    /// The chart of the mapped parameters `q = a (w0(u), w1(v))` and its
    /// inverse.
    a: [[f64; 2]; 2],
    inv: [[f64; 2]; 2],
    /// The parameter that closes and its period.
    pub(super) periodic: Option<(usize, f64)>,
    /// The start of the parameter domain.
    lo: [f64; 2],
}

/// A parameter mapped by a given slope: from the domain's start, piecewise
/// linear between stations; on past the ends by its end slopes, or by
/// whole periods on a closed parameter. Or the parameter itself, less the
/// domain's start.
struct Warp {
    lo: f64,
    step: f64,
    /// The mapped value at each station; none for the parameter itself.
    at: Vec<f64>,
    /// The domain's length.
    span: f64,
    closed: bool,
}

impl Warp {
    fn plain(lo: f64, hi: f64) -> Warp {
        Warp {
            lo,
            step: (hi - lo) / STATIONS as f64,
            at: Vec::new(),
            span: hi - lo,
            closed: false,
        }
    }

    fn new(lo: f64, hi: f64, closed: bool, slope: impl Fn(f64) -> f64) -> Warp {
        let step = (hi - lo) / STATIONS as f64;
        let speeds: Vec<f64> = (0..=STATIONS)
            .map(|i| slope(lo + i as f64 * step))
            .collect();
        // A line where the face pinches to nothing still moves the map.
        let floor = 1e-3 * speeds.iter().fold(0.0f64, |m, &x| m.max(x));
        let mut at = vec![0.0];
        for w in speeds.windows(2) {
            at.push(at[at.len() - 1] + 0.5 * step * (w[0].max(floor) + w[1].max(floor)));
        }
        Warp {
            lo,
            step,
            at,
            span: hi - lo,
            closed,
        }
    }

    fn total(&self) -> f64 {
        match self.at.last() {
            Some(&t) => t,
            None => self.span,
        }
    }

    /// The station interval holding station coordinate `x`, the end ones
    /// reaching on.
    fn interval(x: f64) -> usize {
        (x.floor().max(0.0) as usize).min(STATIONS - 1)
    }

    fn map(&self, u: f64) -> f64 {
        if self.at.is_empty() {
            return u - self.lo;
        }
        let mut x = (u - self.lo) / self.step;
        let mut base = 0.0;
        if self.closed {
            let turns = (x / STATIONS as f64).floor();
            x -= turns * STATIONS as f64;
            base = turns * self.total();
        }
        let i = Self::interval(x);
        base + self.at[i] + (x - i as f64) * (self.at[i + 1] - self.at[i])
    }

    /// The slope of the map at `u`, per parameter.
    fn slope(&self, u: f64) -> f64 {
        if self.at.is_empty() {
            return 1.0;
        }
        let mut x = (u - self.lo) / self.step;
        if self.closed {
            x = x.rem_euclid(STATIONS as f64);
        }
        let i = Self::interval(x);
        (self.at[i + 1] - self.at[i]) / self.step
    }

    fn unmap(&self, t: f64) -> f64 {
        if self.at.is_empty() {
            return self.lo + t;
        }
        let (mut t, mut base) = (t, self.lo);
        if self.closed {
            let turns = (t / self.total()).floor();
            t -= turns * self.total();
            base += turns * STATIONS as f64 * self.step;
        }
        let i = self.at[1..STATIONS].partition_point(|&a| a <= t);
        let d = (self.at[i + 1] - self.at[i]).max(f64::MIN_POSITIVE);
        base + self.step * (i as f64 + (t - self.at[i]) / d)
    }
}

/// The first fundamental form `[E, F, G]` of `surface` at `uv`.
fn metric(surface: &Surface, uv: V2) -> [f64; 3] {
    let d = surface.ders(uv);
    [dot(d[1], d[1]), dot(d[1], d[2]), dot(d[2], d[2])]
}

impl Param {
    /// The chart of a B-spline face through `samples`, or none for a
    /// surface with a pole, closed both ways, or stretched past
    /// [`MAX_STRETCH`] over the samples.
    pub(super) fn of(surface: &Surface, samples: &[V3]) -> Option<Param> {
        let Surface::Nurbs(n) = surface else {
            return None;
        };
        if !surface.poles().is_empty() {
            return None;
        }
        let periodic = match surface.periods() {
            [Some(p), None] => Some((0, p)),
            [None, Some(p)] => Some((1, p)),
            [None, None] => None,
            _ => return None,
        };
        let step = samples.len().div_ceil(SAMPLES).max(1);
        let uvs: Vec<V2> = samples
            .iter()
            .step_by(step)
            .map(|&p| surface.param(p))
            .collect();
        if uvs.is_empty() {
            return None;
        }
        let (ud, vd) = n.domain();
        let lo = [ud[0], vd[0]];
        let plain = [Warp::plain(ud[0], ud[1]), Warp::plain(vd[0], vd[1])];
        if let Some(p) = Param::fit(surface, &uvs, plain, periodic, lo) {
            return Some(p);
        }
        // Half the log of the speeds' ratio, as a part of u and a part of
        // v: each part its mean over lines across the face's box, less half
        // the mean over the box.
        let (mut blo, mut bhi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for uv in &uvs {
            for k in 0..2 {
                blo[k] = blo[k].min(uv[k]);
                bhi[k] = bhi[k].max(uv[k]);
            }
        }
        let line =
            |k: usize, j: usize| blo[k] + (bhi[k] - blo[k]) * (j as f64 + 0.5) / ACROSS as f64;
        let half_log = |uv: V2| {
            let [e, _, g] = metric(surface, uv);
            0.5 * (e.max(f64::MIN_POSITIVE) / g.max(f64::MIN_POSITIVE)).ln()
        };
        let part = |k: usize, x: f64| {
            (0..ACROSS)
                .map(|j| {
                    let mut uv = [x; 2];
                    uv[1 - k] = line(1 - k, j);
                    half_log(uv)
                })
                .sum::<f64>()
                / ACROSS as f64
        };
        let mean = (0..ACROSS).map(|i| part(0, line(0, i))).sum::<f64>() / ACROSS as f64;
        let periods = surface.periods();
        let lengths = [
            Warp::new(ud[0], ud[1], periods[0].is_some(), |u| {
                (part(0, u) - 0.5 * mean).exp()
            }),
            Warp::new(vd[0], vd[1], periods[1].is_some(), |v| {
                (0.5 * mean - part(1, v)).exp()
            }),
        ];
        Param::fit(surface, &uvs, lengths, periodic, lo)
    }

    /// The chart of the face's parameters at `uvs` mapped by `warp`, or
    /// none where the metric there strays past [`MAX_STRETCH`].
    fn fit(
        surface: &Surface,
        uvs: &[V2],
        warp: [Warp; 2],
        periodic: Option<(usize, f64)>,
        lo: V2,
    ) -> Option<Param> {
        // The metric in the mapped parameters at each sample.
        let forms: Vec<[f64; 3]> = uvs
            .iter()
            .map(|&uv| mapped(&metric(surface, uv), &warp, uv))
            .collect();
        let k = forms.len() as f64;
        let mean = forms.iter().fold([0.0; 3], |m, f| {
            [m[0] + f[0] / k, m[1] + f[1] / k, m[2] + f[2] / k]
        });
        let a = sqrt_spd(mean)?;
        let inv = inverse(a)?;
        // The metric in the chart at each sample: its two principal
        // stretches within MAX_STRETCH of each other.
        for f in &forms {
            let [r0, r1, r2] = pulled(f, &inv);
            let (tr, det) = (r0 + r2, r0 * r2 - r1 * r1);
            let disc = (0.25 * tr * tr - det).max(0.0).sqrt();
            let (big, small) = (0.5 * tr + disc, 0.5 * tr - disc);
            if !(small > 0.0 && big <= MAX_STRETCH * MAX_STRETCH * small) {
                return None;
            }
        }
        Some(Param {
            warp,
            a,
            inv,
            periodic,
            lo,
        })
    }

    /// The chart point of parameters `uv` (relative to the domain's start).
    pub(super) fn to_chart(&self, uv: V2) -> V2 {
        mul(&self.a, [self.warp[0].map(uv[0]), self.warp[1].map(uv[1])])
    }

    /// The parameters at chart point `q`, the closed one moved on by
    /// `shift` and into its domain.
    pub(super) fn params(&self, q: V2, shift: f64) -> V2 {
        let r = mul(&self.inv, q);
        let mut uv = [self.warp[0].unmap(r[0]), self.warp[1].unmap(r[1])];
        if let Some((k, p)) = self.periodic {
            uv[k] = self.lo[k] + (uv[k] + shift - self.lo[k]).rem_euclid(p);
        }
        uv
    }

    /// The chart length of a whole period of the closed parameter.
    pub(super) fn period_length(&self) -> f64 {
        match self.periodic {
            Some((k, _)) => self.warp[k].total() * self.a[0][k].hypot(self.a[1][k]),
            None => f64::INFINITY,
        }
    }

    /// Chart length per length on the face at parameters `uv`: by the
    /// geometric mean of the face's two principal stretches there.
    pub(super) fn shrink(&self, surface: &Surface, uv: V2) -> f64 {
        let [r0, r1, r2] = pulled(&mapped(&metric(surface, uv), &self.warp, uv), &self.inv);
        (r0 * r2 - r1 * r1).max(1e-300).powf(-0.25)
    }
}

/// The metric `[E, F, G]` at `uv` in the parameters mapped by `warp`.
fn mapped(f: &[f64; 3], warp: &[Warp; 2], uv: V2) -> [f64; 3] {
    let d = [warp[0].slope(uv[0]), warp[1].slope(uv[1])];
    [
        f[0] / (d[0] * d[0]),
        f[1] / (d[0] * d[1]),
        f[2] / (d[1] * d[1]),
    ]
}

/// The metric `[E, F, G]` pulled back through `inv`: `inv^T M inv`.
fn pulled(f: &[f64; 3], inv: &[[f64; 2]; 2]) -> [f64; 3] {
    let m = [[f[0], f[1]], [f[1], f[2]]];
    let mut r = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    r[i][j] += inv[k][i] * m[k][l] * inv[l][j];
                }
            }
        }
    }
    [r[0][0], r[0][1], r[1][1]]
}

/// The symmetric root of the positive definite `[[a, b], [b, c]]`.
fn sqrt_spd([a, b, c]: [f64; 3]) -> Option<[[f64; 2]; 2]> {
    let det = a * c - b * b;
    if !(det > 0.0 && a > 0.0) {
        return None;
    }
    let s = det.sqrt();
    let t = (a + c + 2.0 * s).sqrt();
    Some([[(a + s) / t, b / t], [b / t, (c + s) / t]])
}

fn inverse(m: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    (det.abs() > 0.0).then(|| {
        [
            [m[1][1] / det, -m[0][1] / det],
            [-m[1][0] / det, m[0][0] / det],
        ]
    })
}

fn mul(m: &[[f64; 2]; 2], x: V2) -> V2 {
    [
        m[0][0] * x[0] + m[0][1] * x[1],
        m[1][0] * x[0] + m[1][1] * x[1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_length_map_goes_back_and_runs_on_by_periods() {
        // Speed 1 + u on [0, 2]: the map is u + u^2 / 2 at the stations.
        let w = Warp::new(0.0, 2.0, false, |u| 1.0 + u);
        assert!((w.total() - 4.0).abs() < 1e-12);
        for u in [-0.5, 0.0, 0.3, 1.0, 1.77, 2.0, 2.4] {
            assert!((w.unmap(w.map(u)) - u).abs() < 1e-12, "{u}");
        }
        let c = Warp::new(1.0, 3.0, true, |u| 2.0 + (u * 3.0).sin());
        for u in [-3.2, 0.0, 1.0, 2.5, 3.0, 4.9, 7.3] {
            assert!((c.unmap(c.map(u)) - u).abs() < 1e-9, "{u}");
            assert!((c.map(u + 2.0) - c.map(u) - c.total()).abs() < 1e-9);
        }
        // A constant speed maps linearly.
        let l = Warp::new(-1.0, 1.0, false, |_| 3.0);
        assert!((l.map(0.25) - 3.75).abs() < 1e-12 && (l.slope(0.7) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_face_whose_speed_varies_along_a_parameter_has_a_chart() {
        // A flat quadratic sheet on u in [2, 5], v in [-1, 3], its controls
        // crowding at one end: along u the speed varies tenfold.
        use crate::NurbsSurface;
        let ctrl = vec![
            [0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.2, 4.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
        ];
        let n = NurbsSurface::new(
            [2, 1],
            [
                vec![2.0, 2.0, 2.0, 5.0, 5.0, 5.0],
                vec![-1.0, -1.0, 3.0, 3.0],
            ],
            [3, 2],
            ctrl,
            vec![1.0; 6],
        );
        let s = Surface::Nurbs(std::sync::Arc::new(n));
        let uvs: Vec<V2> = (0..8)
            .flat_map(|i| {
                (0..8).map(move |j| {
                    [
                        2.0 + 3.0 * (i as f64 + 0.5) / 8.0,
                        -1.0 + 4.0 * (j as f64 + 0.5) / 8.0,
                    ]
                })
            })
            .collect();
        let samples: Vec<V3> = uvs.iter().map(|&uv| s.eval(uv)).collect();
        let pm = Param::of(&s, &samples).expect("a chart");
        for &uv in &uvs {
            let back = pm.params(pm.to_chart(uv), 0.0);
            assert!((back[0] - uv[0]).abs() < 1e-9 && (back[1] - uv[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn the_root_squares_back() {
        let m = [3.0, 0.7, 1.5];
        let r = sqrt_spd(m).unwrap();
        let sq = [
            r[0][0] * r[0][0] + r[0][1] * r[1][0],
            r[0][0] * r[0][1] + r[0][1] * r[1][1],
            r[1][0] * r[0][1] + r[1][1] * r[1][1],
        ];
        for k in 0..3 {
            assert!((sq[k] - m[k]).abs() < 1e-12);
        }
        // Pulled back through its own inverse root, a metric is the identity.
        let p = pulled(&m, &inverse(r).unwrap());
        assert!((p[0] - 1.0).abs() < 1e-12 && p[1].abs() < 1e-12 && (p[2] - 1.0).abs() < 1e-12);
    }
}
