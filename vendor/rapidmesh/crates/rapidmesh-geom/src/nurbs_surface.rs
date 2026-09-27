//! Tensor-product rational B-spline (NURBS) surface.
//!
//! The missing geometry primitive for consuming general CAD/STEP geometry: a
//! trimmed NURBS surface is what a boolean of free-form bodies produces, and what
//! the B-rep layer must be able to carry as a first-class [`crate::SurfaceKind`]
//! sibling. This is the surface analogue of [`crate::nurbs::NurbsCurve`]: the same
//! clamped knot vectors and rational weights, in two parameter directions.
//!
//! Evaluation is the tensor product of the per-direction B-spline basis (Piegl &
//! Tiller A2.2) with the rational quotient on the homogeneous control net; the
//! derivatives up to second order use the basis derivatives (A2.3) and the
//! rational quotient rule (A4.4). The inverse map (`(x,y,z) -> (u,v)`) is a
//! safeguarded Newton projection started per knot patch, and the patches are
//! pruned by the convex hull of their control points.

use crate::nurbs::{basis_funs, ders_basis, find_span, MAX_DEGREE};
use crate::vec3::{cross, dot, scale, sub, V3};
/// A tensor-product rational B-spline surface `S(u,v)`.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsSurface {
    /// Polynomial degrees `(p, q)` in the `u` and `v` directions.
    pub degree: [usize; 2],
    /// Clamped, non-decreasing knot vectors `(U, V)`; `U.len() == n_u + p + 1`.
    pub knots: [Vec<f64>; 2],
    /// Control-point counts `(n_u, n_v)` per direction.
    pub n: [usize; 2],
    /// Control net, row-major: control `(i, j)` is `ctrl[i * n_v + j]`.
    pub ctrl: Vec<V3>,
    /// Rational weights, parallel to `ctrl` (all 1.0 = a plain B-spline).
    pub weights: Vec<f64>,
}

impl NurbsSurface {
    /// Builds a surface, validating the knot/control/weight sizes; panics on
    /// a malformed input (see [`NurbsSurface::try_new`]).
    pub fn new(
        degree: [usize; 2],
        knots: [Vec<f64>; 2],
        n: [usize; 2],
        ctrl: Vec<V3>,
        weights: Vec<f64>,
    ) -> NurbsSurface {
        NurbsSurface::try_new(degree, knots, n, ctrl, weights).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Builds a surface, or says what is malformed: degrees in
    /// `1..=MAX_DEGREE`, `n + degree + 1` clamped non-decreasing knots per
    /// direction, an `n[0] x n[1]` net and one positive weight per control.
    pub fn try_new(
        degree: [usize; 2],
        knots: [Vec<f64>; 2],
        n: [usize; 2],
        ctrl: Vec<V3>,
        weights: Vec<f64>,
    ) -> Result<NurbsSurface, String> {
        for d in 0..2 {
            let (p, k) = (degree[d], &knots[d]);
            if !(1..=MAX_DEGREE).contains(&p) {
                return Err(format!("degree {p} is not in 1..={MAX_DEGREE}"));
            }
            if n[d] <= p {
                return Err(format!("{} controls need degree < {}", n[d], n[d]));
            }
            if k.len() != n[d] + p + 1 {
                return Err(format!("{} knots, need {}", k.len(), n[d] + p + 1));
            }
            if k.windows(2).any(|w| !(w[0] <= w[1])) || !(k[p] < k[n[d]]) {
                return Err("knots must be non-decreasing with a non-empty domain".into());
            }
        }
        if ctrl.len() != n[0] * n[1] {
            return Err(format!("{} controls, need {}", ctrl.len(), n[0] * n[1]));
        }
        if weights.len() != ctrl.len() || !weights.iter().all(|&w| w > 0.0) {
            return Err("one positive weight per control point".into());
        }
        Ok(NurbsSurface {
            degree,
            knots,
            n,
            ctrl,
            weights,
        })
    }

    /// A clamped uniform knot vector for `n` controls of degree `p`.
    pub fn clamped_knots(n: usize, p: usize) -> Vec<f64> {
        let inner = n - p;
        (0..n + p + 1)
            .map(|i| (i.saturating_sub(p)).min(inner) as f64 / inner as f64)
            .collect()
    }

    /// Parameter domain `([u_min, u_max], [v_min, v_max])` (the clamped end knots).
    pub fn domain(&self) -> ([f64; 2], [f64; 2]) {
        let ud = [self.knots[0][self.degree[0]], self.knots[0][self.n[0]]];
        let vd = [self.knots[1][self.degree[1]], self.knots[1][self.n[1]]];
        (ud, vd)
    }

    /// Surface point `S(u, v)`.
    pub fn eval(&self, u: f64, v: f64) -> V3 {
        let (ud, vd) = self.domain();
        let u = u.clamp(ud[0], ud[1]);
        let v = v.clamp(vd[0], vd[1]);
        let (pu, pv) = (self.degree[0], self.degree[1]);
        let su = find_span(&self.knots[0], self.n[0] - 1, pu, u);
        let sv = find_span(&self.knots[1], self.n[1] - 1, pv, v);
        let bu = basis_funs(su, u, pu, &self.knots[0]);
        let bv = basis_funs(sv, v, pv, &self.knots[1]);
        let mut num = [0.0f64; 3];
        let mut den = 0.0f64;
        for i in 0..=pu {
            let ci = su - pu + i;
            for j in 0..=pv {
                let cj = sv - pv + j;
                let idx = ci * self.n[1] + cj;
                let w = self.weights[idx] * bu[i] * bv[j];
                den += w;
                for k in 0..3 {
                    num[k] += w * self.ctrl[idx][k];
                }
            }
        }
        std::array::from_fn(|k| num[k] / den)
    }

    /// `[S, S_u, S_v, S_uu, S_uv, S_vv]` at `(u, v)` (clamped to the domain).
    pub fn ders2(&self, u: f64, v: f64) -> [V3; 6] {
        let (ud, vd) = self.domain();
        let u = u.clamp(ud[0], ud[1]);
        let v = v.clamp(vd[0], vd[1]);
        let (pu, pv) = (self.degree[0], self.degree[1]);
        let su = find_span(&self.knots[0], self.n[0] - 1, pu, u);
        let sv = find_span(&self.knots[1], self.n[1] - 1, pv, v);
        let nu = ders_basis(&self.knots[0], pu, su, u);
        let nv = ders_basis(&self.knots[1], pv, sv, v);
        // Homogeneous derivatives a[k][l] = d^k/du^k d^l/dv^l of (w x, w).
        let mut a = [[[0.0f64; 4]; 3]; 3];
        for i in 0..=pu {
            for j in 0..=pv {
                let idx = (su - pu + i) * self.n[1] + sv - pv + j;
                let w = self.weights[idx];
                let c = self.ctrl[idx];
                let pw = [c[0] * w, c[1] * w, c[2] * w, w];
                for k in 0..3 {
                    for l in 0..3 - k {
                        let b = nu[k][i] * nv[l][j];
                        for (x, &y) in a[k][l].iter_mut().zip(&pw) {
                            *x += b * y;
                        }
                    }
                }
            }
        }
        let w = |k: usize, l: usize| a[k][l][3];
        let h = |k: usize, l: usize| -> V3 { [a[k][l][0], a[k][l][1], a[k][l][2]] };
        let inv = 1.0 / w(0, 0);
        let s = scale(h(0, 0), inv);
        let s_u = scale(sub(h(1, 0), scale(s, w(1, 0))), inv);
        let s_v = scale(sub(h(0, 1), scale(s, w(0, 1))), inv);
        let s_uu = scale(
            sub(sub(h(2, 0), scale(s_u, 2.0 * w(1, 0))), scale(s, w(2, 0))),
            inv,
        );
        let s_vv = scale(
            sub(sub(h(0, 2), scale(s_v, 2.0 * w(0, 1))), scale(s, w(0, 2))),
            inv,
        );
        let s_uv = scale(
            sub(
                sub(sub(h(1, 1), scale(s_v, w(1, 0))), scale(s_u, w(0, 1))),
                scale(s, w(1, 1)),
            ),
            inv,
        );
        [s, s_u, s_v, s_uu, s_uv, s_vv]
    }

    /// Unit normal `S_u x S_v` at `(u, v)`. Where it degenerates (a pole, a
    /// collapsed edge) the normal of a point nudged toward the domain centre
    /// stands in, the limit the surface approaches there.
    pub fn normal(&self, u: f64, v: f64) -> V3 {
        let (ud, vd) = self.domain();
        let centre = [0.5 * (ud[0] + ud[1]), 0.5 * (vd[0] + vd[1])];
        for t in [0.0, 1e-9, 1e-6, 1e-3] {
            let (uu, vv) = (u + t * (centre[0] - u), v + t * (centre[1] - v));
            let d = self.ders2(uu, vv);
            let n = cross(d[1], d[2]);
            let l = dot(n, n).sqrt();
            let scale_ref = dot(d[1], d[1]).sqrt() * dot(d[2], d[2]).sqrt();
            if l > 1e-12 * scale_ref && l > 0.0 {
                return scale(n, 1.0 / l);
            }
        }
        [0.0, 0.0, 1.0]
    }

    /// Principal curvatures `[k_max, k_min]` (magnitudes) at `(u, v)` from
    /// the first and second fundamental forms. Where the parametrization
    /// degenerates (a pole) a point nudged toward the domain centre stands in,
    /// as for the normal; none if that fails too.
    pub fn principal_curvatures(&self, u: f64, v: f64) -> Option<[f64; 2]> {
        let (ud, vd) = self.domain();
        let centre = [0.5 * (ud[0] + ud[1]), 0.5 * (vd[0] + vd[1])];
        [0.0, 1e-9, 1e-6, 1e-3]
            .into_iter()
            .find_map(|t| self.curvatures_at(u + t * (centre[0] - u), v + t * (centre[1] - v)))
    }

    fn curvatures_at(&self, u: f64, v: f64) -> Option<[f64; 2]> {
        let [_, s_u, s_v, s_uu, s_uv, s_vv] = self.ders2(u, v);
        let (e, f, g) = (dot(s_u, s_u), dot(s_u, s_v), dot(s_v, s_v));
        let det = e * g - f * f;
        if !(det > 1e-24 * (e * g).max(f64::MIN_POSITIVE)) {
            return None;
        }
        let n = scale(cross(s_u, s_v), 1.0 / det.sqrt());
        let (l, m, nn) = (dot(s_uu, n), dot(s_uv, n), dot(s_vv, n));
        let mean = (e * nn + g * l - 2.0 * f * m) / (2.0 * det);
        let gauss = (l * nn - m * m) / det;
        let disc = (mean * mean - gauss).max(0.0).sqrt();
        let (k1, k2) = ((mean + disc).abs(), (mean - disc).abs());
        Some([k1.max(k2), k1.min(k2)])
    }

    /// The parameters of the point of the surface nearest `q`.
    pub fn closest_param(&self, q: V3) -> [f64; 2] {
        let (pu, pv) = (self.degree[0], self.degree[1]);
        let spans = |d: usize| -> Vec<usize> {
            let (k, p) = (&self.knots[d], self.degree[d]);
            (p..self.n[d]).filter(|&s| k[s] < k[s + 1]).collect()
        };
        // A lower bound of the distance to a knot patch: the box of its
        // control points, which holds the patch (positive weights).
        let bound = |su: usize, sv: usize| -> f64 {
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for i in su - pu..=su {
                for c in &self.ctrl[i * self.n[1] + sv - pv..=i * self.n[1] + sv] {
                    for k in 0..3 {
                        lo[k] = lo[k].min(c[k]);
                        hi[k] = hi[k].max(c[k]);
                    }
                }
            }
            (0..3)
                .map(|k| (lo[k] - q[k]).max(q[k] - hi[k]).max(0.0).powi(2))
                .sum()
        };
        let mut patches: Vec<(f64, usize, usize)> = spans(0)
            .iter()
            .flat_map(|&su| spans(1).into_iter().map(move |sv| (su, sv)))
            .map(|(su, sv)| (bound(su, sv), su, sv))
            .collect();
        patches.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (ud, vd) = self.domain();
        let mut best = ([ud[0], vd[0]], f64::INFINITY);
        for (b, su, sv) in patches {
            if b >= best.1 {
                break;
            }
            let c = self.closest_in_patch(su, sv, q);
            if c.1 < best.1 {
                best = c;
            }
        }
        best.0
    }

    /// Nearest parameters to `q` within the knot patch `(su, sv)`, with the
    /// squared distance: the best of a sample grid, then Newton steps that
    /// are halved until the distance drops.
    fn closest_in_patch(&self, su: usize, sv: usize, q: V3) -> ([f64; 2], f64) {
        const SAMPLES: usize = 4;
        let a = [self.knots[0][su], self.knots[1][sv]];
        let b = [self.knots[0][su + 1], self.knots[1][sv + 1]];
        let d2 = |x: V3| dot(sub(x, q), sub(x, q));
        let mut best = (a, f64::INFINITY);
        for i in 0..=SAMPLES {
            for j in 0..=SAMPLES {
                let t = [
                    a[0] + (b[0] - a[0]) * i as f64 / SAMPLES as f64,
                    a[1] + (b[1] - a[1]) * j as f64 / SAMPLES as f64,
                ];
                let d = d2(self.eval(t[0], t[1]));
                if d < best.1 {
                    best = (t, d);
                }
            }
        }
        let (mut t, mut dist) = best;
        for _ in 0..32 {
            let [s, s_u, s_v, s_uu, s_uv, s_vv] = self.ders2(t[0], t[1]);
            let r = sub(s, q);
            let g = [dot(r, s_u), dot(r, s_v)];
            let h = [
                dot(s_u, s_u) + dot(r, s_uu),
                dot(s_u, s_v) + dot(r, s_uv),
                dot(s_v, s_v) + dot(r, s_vv),
            ];
            let det = h[0] * h[2] - h[1] * h[1];
            // Newton where the Hessian is positive definite, else a scaled
            // gradient step.
            let step = if h[0] > 0.0 && det > 0.0 {
                [
                    (h[2] * g[0] - h[1] * g[1]) / det,
                    (h[0] * g[1] - h[1] * g[0]) / det,
                ]
            } else {
                let m = dot(s_u, s_u).max(dot(s_v, s_v)).max(f64::MIN_POSITIVE);
                [g[0] / m, g[1] / m]
            };
            let mut lambda = 1.0;
            let mut moved = false;
            for _ in 0..16 {
                let c = [
                    (t[0] - lambda * step[0]).clamp(a[0], b[0]),
                    (t[1] - lambda * step[1]).clamp(a[1], b[1]),
                ];
                let d = d2(self.eval(c[0], c[1]));
                if d < dist {
                    moved = (c[0] - t[0]).abs() > 1e-15 * (b[0] - a[0])
                        || (c[1] - t[1]).abs() > 1e-15 * (b[1] - a[1]);
                    t = c;
                    dist = d;
                    break;
                }
                lambda *= 0.5;
            }
            if !moved {
                break;
            }
        }
        (t, dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bilinear (degree 1x1) patch over a 2x2 net: eval interpolates the corners
    /// and the center is the corner average.
    #[test]
    fn bilinear_patch_evaluates() {
        let ctrl = vec![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let s = NurbsSurface::new(
            [1, 1],
            [vec![0.0, 0.0, 1.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]],
            [2, 2],
            ctrl,
            vec![1.0; 4],
        );
        let (ud, vd) = s.domain();
        assert_eq!(ud, [0.0, 1.0]);
        assert_eq!(vd, [0.0, 1.0]);
        // corners
        assert_eq!(s.eval(0.0, 0.0), [0.0, 0.0, 0.0]);
        assert_eq!(s.eval(1.0, 1.0), [1.0, 1.0, 0.0]);
        // center = average of the four corners
        let c = s.eval(0.5, 0.5);
        assert!((c[0] - 0.5).abs() < 1e-12);
        assert!((c[1] - 0.5).abs() < 1e-12);
        assert!((c[2] - 0.5).abs() < 1e-12);
    }

    /// A degree-2 row stays planar in z when all control z are equal (partition of
    /// unity), confirming the rational tensor sum.
    #[test]
    fn quadratic_partition_of_unity() {
        let ctrl: Vec<V3> = (0..3)
            .flat_map(|i| (0..3).map(move |j| [i as f64, j as f64, 2.0]))
            .collect();
        let s = NurbsSurface::new(
            [2, 2],
            [
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            [3, 3],
            ctrl,
            vec![1.0; 9],
        );
        for &(u, v) in &[(0.3, 0.7), (0.5, 0.5), (0.9, 0.1)] {
            assert!((s.eval(u, v)[2] - 2.0).abs() < 1e-12, "z must stay 2.0");
        }
    }

    /// A rational bicubic over a wavy 5x4 net, two interior knots in u.
    fn wavy() -> NurbsSurface {
        let (nu, nv) = (5, 4);
        let ctrl: Vec<V3> = (0..nu)
            .flat_map(|i| {
                (0..nv).map(move |j| {
                    let (x, y) = (i as f64, j as f64 * 1.3);
                    [x, y, 0.4 * (x * 1.7).sin() * (y * 0.9).cos()]
                })
            })
            .collect();
        let weights = (0..nu * nv)
            .map(|k| 1.0 + 0.3 * ((k * 7 % 5) as f64) / 4.0)
            .collect();
        NurbsSurface::new(
            [3, 3],
            [
                vec![0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [nu, nv],
            ctrl,
            weights,
        )
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let s = wavy();
        let h = 1e-5;
        for &(u, v) in &[(0.2, 0.3), (0.55, 0.8), (0.9, 0.5)] {
            let d = s.ders2(u, v);
            let fd = |f: &dyn Fn(f64, f64) -> V3, du: f64, dv: f64| -> V3 {
                scale(sub(f(u + du, v + dv), f(u - du, v - dv)), 0.5 / h)
            };
            let p = |a: f64, b: f64| s.eval(a, b);
            let pu = |a: f64, b: f64| s.ders2(a, b)[1];
            let pv = |a: f64, b: f64| s.ders2(a, b)[2];
            let want = [
                fd(&p, h, 0.0),
                fd(&p, 0.0, h),
                fd(&pu, h, 0.0),
                fd(&pu, 0.0, h),
                fd(&pv, 0.0, h),
            ];
            for (k, w) in want.iter().enumerate() {
                let e = sub(d[k + 1], *w);
                assert!(
                    dot(e, e).sqrt() < 1e-5 * (1.0 + dot(*w, *w).sqrt()),
                    "derivative {k} at ({u}, {v})"
                );
            }
        }
    }

    #[test]
    fn footpoint_matches_a_dense_scan() {
        let s = wavy();
        for q in [
            [1.3, 2.0, 0.8],
            [3.7, 0.4, -0.5],
            [-0.5, 5.0, 0.3],
            [2.0, 1.9, 0.05],
        ] {
            let t = s.closest_param(q);
            let e = sub(s.eval(t[0], t[1]), q);
            let d = dot(e, e).sqrt();
            let n = 300;
            let mut scan = f64::INFINITY;
            for i in 0..=n {
                for j in 0..=n {
                    let e = sub(s.eval(i as f64 / n as f64, j as f64 / n as f64), q);
                    scan = scan.min(dot(e, e).sqrt());
                }
            }
            assert!(d <= scan + 1e-12, "{q:?}: {d} vs scan {scan}");
            assert!(scan - d < 1e-3, "{q:?}: {d} vs scan {scan}");
        }
    }

    #[test]
    fn rational_quarter_cylinder_is_exact() {
        // A quarter circle of radius 2 (rational quadratic) swept along z.
        let r = 2.0;
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let arc = [[r, 0.0], [r, r], [0.0, r]];
        let ctrl: Vec<V3> = arc
            .iter()
            .flat_map(|c| [[c[0], c[1], 0.0], [c[0], c[1], 3.0]])
            .collect();
        let s = NurbsSurface::new(
            [2, 1],
            [vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]],
            [3, 2],
            ctrl,
            vec![1.0, 1.0, w, w, 1.0, 1.0],
        );
        for &(u, v) in &[(0.1, 0.2), (0.5, 0.5), (0.8, 0.9)] {
            let p = s.eval(u, v);
            assert!((p[0].hypot(p[1]) - r).abs() < 1e-12, "on the barrel");
            let n = s.normal(u, v);
            let radial = [p[0] / r, p[1] / r, 0.0];
            assert!((dot(n, radial).abs() - 1.0).abs() < 1e-12, "radial normal");
            let k = s.principal_curvatures(u, v).unwrap();
            assert!((k[0] - 1.0 / r).abs() < 1e-9 && k[1].abs() < 1e-9, "{k:?}");
        }
        // A point off the barrel lands on its radial footpoint.
        let q = [2.5, 1.5, 1.2];
        let t = s.closest_param(q);
        let f = s.eval(t[0], t[1]);
        let rho = q[0].hypot(q[1]);
        let want = [q[0] * r / rho, q[1] * r / rho, 1.2];
        assert!(
            dot(sub(f, want), sub(f, want)).sqrt() < 1e-9,
            "{f:?} vs {want:?}"
        );
    }
}
