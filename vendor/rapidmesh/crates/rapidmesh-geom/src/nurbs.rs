//! Rational B-spline (NURBS) curves in 2D, the profile primitive for swept and
//! extruded spline surfaces (e.g. an airfoil section). A curve is a degree, a
//! clamped knot vector, control points and rational weights; evaluation and the
//! first two derivatives use the standard basis-function-derivative algorithms
//! (Piegl & Tiller, "The NURBS Book", A2.1/A2.3), so curvature is exact and
//! drives the curved-surface sizing bias.
//!
//! Weights all 1.0 give an ordinary (non-rational) B-spline; conic sections
//! (the exact circle/ellipse) need the rational form, so the type carries
//! weights throughout and de-homogenizes with the quotient rule.

type P2 = [f64; 2];

/// Highest supported degree. Basis evaluation runs on fixed-size stack
/// arrays, so evaluating a curve or surface never allocates.
pub const MAX_DEGREE: usize = 9;
pub(crate) const W: usize = MAX_DEGREE + 1;

/// The knot span index containing `u` (Piegl & Tiller A2.1); `n` is the last
/// control index, `p` the degree.
pub(crate) fn find_span(knots: &[f64], n: usize, p: usize, u: f64) -> usize {
    if u >= knots[n + 1] {
        return n;
    }
    if u <= knots[p] {
        return p;
    }
    let (mut low, mut high) = (p, n + 1);
    let mut mid = (low + high) / 2;
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

/// Nonzero basis functions `N_{span-p+j, p}(u)`, `j` in `0..=p` (Piegl &
/// Tiller A2.2).
pub(crate) fn basis_funs(span: usize, u: f64, p: usize, knots: &[f64]) -> [f64; W] {
    let mut n = [0.0f64; W];
    let mut left = [0.0f64; W];
    let mut right = [0.0f64; W];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// Nonzero basis functions and their first two derivatives at `u`
/// of degree `p` (Piegl & Tiller A2.3). Returns `ders[k][j]`, `k` in `0..=2`, `j` in
/// `0..=degree` for control point `span - degree + j`.
pub(crate) fn ders_basis(knots: &[f64], p: usize, span: usize, u: f64) -> [[f64; W]; 3] {
    let nd = 2;
    let u_ = knots;
    let mut ndu = [[0.0f64; W]; W];
    let mut left = [0.0f64; W];
    let mut right = [0.0f64; W];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = u - u_[span + 1 - j];
        right[j] = u_[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    let mut ders = [[0.0f64; W]; 3];
    for j in 0..=p {
        ders[0][j] = ndu[j][p];
    }
    // derivatives (orders above the degree are identically zero)
    let kmax = nd.min(p);
    let mut a = [[0.0f64; W]; 2];
    for r in 0..=p {
        let (mut s1, mut s2) = (0usize, 1usize);
        a[0][0] = 1.0;
        for k in 1..=kmax {
            let mut d = 0.0;
            let rk = r as isize - k as isize;
            let pk = p as isize - k as isize;
            if r >= k {
                a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                d = a[s2][0] * ndu[rk as usize][pk as usize];
            }
            let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
            let j2 = if (r as isize) - 1 <= pk { k - 1 } else { p - r };
            for j in j1..=j2 {
                a[s2][j] =
                    (a[s1][j] - a[s1][j - 1]) / ndu[(pk + 1) as usize][(rk + j as isize) as usize];
                d += a[s2][j] * ndu[(rk + j as isize) as usize][pk as usize];
            }
            if r <= pk as usize {
                a[s2][k] = -a[s1][k - 1] / ndu[(pk + 1) as usize][r];
                d += a[s2][k] * ndu[r][pk as usize];
            }
            ders[k][r] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }
    // multiply by the factorial factors p!/(p-k)! (orders above the degree
    // stay zero, so only scale up to kmax)
    let mut r = p as f64;
    for k in 1..=kmax {
        for j in 0..=p {
            ders[k][j] *= r;
        }
        r *= (p - k) as f64;
    }
    ders
}

/// Solves `A x = b` (two RHS columns) by Gaussian elimination with partial
/// pivoting. `A` is small and dense (the interpolation collocation matrix).
fn solve_banded(a: &mut [Vec<f64>], b: &mut [[f64; 2]]) -> Vec<[f64; 2]> {
    let n = a.len();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in (col + 1)..n {
            let f = a[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r][0] -= f * b[col][0];
            b[r][1] -= f * b[col][1];
        }
    }
    let mut x = vec![[0.0; 2]; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s[0] -= a[col][c] * x[c][0];
            s[1] -= a[col][c] * x[c][1];
        }
        x[col][0] = s[0] / a[col][col];
        x[col][1] = s[1] / a[col][col];
    }
    x
}

/// A 2D rational B-spline curve.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsCurve {
    /// Polynomial degree `p` (order `p + 1`).
    pub degree: usize,
    /// Clamped, non-decreasing knot vector, length `ctrl.len() + degree + 1`.
    pub knots: Vec<f64>,
    /// Control points.
    pub ctrl: Vec<P2>,
    /// Rational weights, parallel to `ctrl` (all 1.0 = a plain B-spline).
    pub weights: Vec<f64>,
}

impl NurbsCurve {
    /// The same curve traversed the other way, over the same domain.
    pub fn reversed(&self) -> NurbsCurve {
        let (lo, hi) = (self.knots[0], self.knots[self.knots.len() - 1]);
        NurbsCurve {
            degree: self.degree,
            knots: self.knots.iter().rev().map(|&k| lo + hi - k).collect(),
            ctrl: self.ctrl.iter().rev().copied().collect(),
            weights: self.weights.iter().rev().copied().collect(),
        }
    }

    /// The curve scaled by `s` about the origin (exact: NURBS are affine
    /// invariant).
    pub fn scaled(&self, s: f64) -> NurbsCurve {
        NurbsCurve {
            ctrl: self.ctrl.iter().map(|c| [c[0] * s, c[1] * s]).collect(),
            ..self.clone()
        }
    }

    /// Builds a curve, validating the knot/control/weight sizes.
    pub fn new(degree: usize, knots: Vec<f64>, ctrl: Vec<P2>, weights: Vec<f64>) -> NurbsCurve {
        assert!(
            (1..=MAX_DEGREE).contains(&degree),
            "degree must be in 1..={MAX_DEGREE}"
        );
        assert!(ctrl.len() > degree, "need at least degree+1 control points");
        assert_eq!(
            knots.len(),
            ctrl.len() + degree + 1,
            "knot count must be n+p+2"
        );
        assert_eq!(weights.len(), ctrl.len(), "one weight per control point");
        assert!(weights.iter().all(|&w| w > 0.0), "weights must be positive");
        NurbsCurve {
            degree,
            knots,
            ctrl,
            weights,
        }
    }

    /// A uniform clamped cubic (degree 3) interpolating the given control points
    /// directly (control polygon, not interpolation): a convenience for tests
    /// and simple profiles. Non-rational.
    pub fn clamped_uniform(degree: usize, ctrl: Vec<P2>) -> NurbsCurve {
        let n = ctrl.len();
        let p = degree;
        // clamped: p+1 zeros, interior 1..(n-p), p+1 ones (count n+p+1)
        let interior = n.saturating_sub(p + 1);
        let mut knots = vec![0.0; p + 1];
        for i in 1..=interior {
            knots.push(i as f64 / (interior + 1) as f64);
        }
        knots.extend(std::iter::repeat_n(1.0, p + 1));
        let weights = vec![1.0; n];
        NurbsCurve::new(degree, knots, ctrl, weights)
    }

    /// Cubic B-spline INTERPOLATING the given points (the curve passes through
    /// each, unlike `clamped_uniform` which uses them as control points). Gives
    /// the FAITHFUL curvature of the sampled shape -- no control-polygon
    /// artifacts -- so a curvature sizing field refines only the genuinely
    /// curved regions (Piegl & Tiller, global interpolation A9.1: chord-length
    /// parameters, averaged knots, solve `N P = Q`). Needs >= 4 points.
    pub fn interpolate(points: &[P2]) -> NurbsCurve {
        let n = points.len() - 1;
        assert!(n >= 3, "cubic interpolation needs >= 4 points");
        let p = 3;
        let dd = |a: P2, b: P2| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        // Chord-length parameters.
        let total: f64 = (1..=n).map(|k| dd(points[k], points[k - 1])).sum();
        let mut u = vec![0.0; n + 1];
        for k in 1..n {
            u[k] = u[k - 1] + dd(points[k], points[k - 1]) / total;
        }
        u[n] = 1.0;
        // Averaged knot vector.
        let mut knots = vec![0.0; n + p + 2];
        for j in (n + 1)..(n + p + 2) {
            knots[j] = 1.0;
        }
        for j in 1..=(n - p) {
            knots[j + p] = (j..j + p).map(|i| u[i]).sum::<f64>() / p as f64;
        }
        // Collocation matrix A[k][i] = N_{i,p}(u_k); solve A P = points per coord.
        let scratch = NurbsCurve::new(p, knots.clone(), points.to_vec(), vec![1.0; n + 1]);
        let mut a = vec![vec![0.0f64; n + 1]; n + 1];
        for k in 0..=n {
            let span = scratch.find_span(u[k]);
            let basis = basis_funs(span, u[k], p, &scratch.knots);
            for j in 0..=p {
                a[k][span - p + j] = basis[j];
            }
        }
        let mut rhs: Vec<[f64; 2]> = points.to_vec();
        let ctrl = solve_banded(&mut a, &mut rhs);
        NurbsCurve::new(p, knots, ctrl, vec![1.0; n + 1])
    }

    /// Parameter domain `[u_min, u_max]` (the clamped end knots).
    pub fn domain(&self) -> (f64, f64) {
        (self.knots[self.degree], self.knots[self.ctrl.len()])
    }

    /// The knot span index containing `u`.
    fn find_span(&self, u: f64) -> usize {
        find_span(&self.knots, self.ctrl.len() - 1, self.degree, u)
    }

    /// Point and first two derivatives `(C, C', C'')` at parameter `u`, via the
    /// rational quotient rule on the homogeneous numerator.
    pub fn ders2(&self, u: f64) -> (P2, P2, P2) {
        let p = self.degree;
        let (lo, hi) = self.domain();
        let u = u.clamp(lo, hi);
        let span = self.find_span(u);
        let nd = ders_basis(&self.knots, self.degree, span, u);
        // homogeneous A^(k) = (x, y, w) for k = 0,1,2
        let mut a = [[0.0f64; 3]; 3];
        for j in 0..=p {
            let i = span - p + j;
            let w = self.weights[i];
            let pw = [self.ctrl[i][0] * w, self.ctrl[i][1] * w, w];
            for (ak, ndk) in a.iter_mut().zip(&nd) {
                for c in 0..3 {
                    ak[c] += ndk[j] * pw[c];
                }
            }
        }
        let (w0, w1, w2) = (a[0][2], a[1][2], a[2][2]);
        let c0 = [a[0][0] / w0, a[0][1] / w0];
        let c1 = [(a[1][0] - w1 * c0[0]) / w0, (a[1][1] - w1 * c0[1]) / w0];
        let c2 = [
            (a[2][0] - 2.0 * w1 * c1[0] - w2 * c0[0]) / w0,
            (a[2][1] - 2.0 * w1 * c1[1] - w2 * c0[1]) / w0,
        ];
        (c0, c1, c2)
    }

    /// Curve point at `u`.
    pub fn eval(&self, u: f64) -> P2 {
        let p = self.degree;
        let (lo, hi) = self.domain();
        let u = u.clamp(lo, hi);
        let span = self.find_span(u);
        let b = basis_funs(span, u, p, &self.knots);
        let (mut x, mut y, mut w) = (0.0, 0.0, 0.0);
        for (j, bj) in b.iter().enumerate().take(p + 1) {
            let i = span - p + j;
            let bw = bj * self.weights[i];
            x += bw * self.ctrl[i][0];
            y += bw * self.ctrl[i][1];
            w += bw;
        }
        [x / w, y / w]
    }

    /// Parameter of the curve point nearest to `q`. A knot span lies in the
    /// convex hull of its `degree + 1` control points, so the distance to
    /// their bounding box bounds the span from below: the nearest span is
    /// searched first, then only spans that could still beat it. Inside a
    /// span a few samples seed Newton on `(C(t) - q) . C'(t) = 0`.
    pub fn closest_param(&self, q: P2) -> f64 {
        let p = self.degree;
        let spans = (p..self.ctrl.len()).filter(|&s| self.knots[s] < self.knots[s + 1]);
        let bound = |s: usize| {
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for c in &self.ctrl[s - p..=s] {
                for k in 0..2 {
                    lo[k] = lo[k].min(c[k]);
                    hi[k] = hi[k].max(c[k]);
                }
            }
            let d: [f64; 2] = std::array::from_fn(|k| (lo[k] - q[k]).max(q[k] - hi[k]).max(0.0));
            d[0] * d[0] + d[1] * d[1]
        };
        let Some(first) = spans.clone().min_by(|&a, &b| bound(a).total_cmp(&bound(b))) else {
            return self.domain().0;
        };
        let mut best = self.closest_in_span(first, q);
        for s in spans {
            if s != first && bound(s) < best.1 {
                let c = self.closest_in_span(s, q);
                if c.1 < best.1 {
                    best = c;
                }
            }
        }
        best.0
    }

    /// Nearest parameter to `q` within knot span `s`, with its squared
    /// distance.
    fn closest_in_span(&self, s: usize, q: P2) -> (f64, f64) {
        const SAMPLES: usize = 4;
        let (a, b) = (self.knots[s], self.knots[s + 1]);
        let d2 = |c: P2| (c[0] - q[0]).powi(2) + (c[1] - q[1]).powi(2);
        let mut best = (a, f64::INFINITY);
        for i in 0..=SAMPLES {
            let t = a + (b - a) * i as f64 / SAMPLES as f64;
            let d = d2(self.eval(t));
            if d < best.1 {
                best = (t, d);
            }
        }
        let mut t = best.0;
        for _ in 0..16 {
            let (c, d1, dd) = self.ders2(t);
            let r = [c[0] - q[0], c[1] - q[1]];
            let g = r[0] * d1[0] + r[1] * d1[1];
            let gp = d1[0] * d1[0] + d1[1] * d1[1] + r[0] * dd[0] + r[1] * dd[1];
            if !(gp > 1e-15) {
                break;
            }
            let step = g / gp;
            t = (t - step).clamp(a, b);
            if step.abs() < 1e-13 * (b - a).max(1.0) {
                break;
            }
        }
        let d = d2(self.eval(t));
        if d < best.1 {
            best = (t, d);
        }
        best
    }

    /// Signed-magnitude curvature `kappa = |x' x y''| / |x'|^3` at `u`.
    pub fn curvature(&self, u: f64) -> f64 {
        let (_, c1, c2) = self.ders2(u);
        let cross = c1[0] * c2[1] - c1[1] * c2[0];
        let speed = (c1[0] * c1[0] + c1[1] * c1[1]).sqrt();
        if speed < 1e-15 {
            0.0
        } else {
            cross.abs() / speed.powi(3)
        }
    }

    /// Arc length over `[u0, u1]` by composite Gauss-Legendre (3-point per
    /// sub-interval), used to build the distance-faithful chart parameter.
    pub fn arc_length(&self, u0: f64, u1: f64, subdivisions: usize) -> f64 {
        // 3-point Gauss-Legendre nodes/weights on [-1, 1].
        const X: [f64; 3] = [-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4];
        const W: [f64; 3] = [
            0.555_555_555_555_555_6,
            0.888_888_888_888_889,
            0.555_555_555_555_555_6,
        ];
        let n = subdivisions.max(1);
        let h = (u1 - u0) / n as f64;
        let mut total = 0.0;
        for i in 0..n {
            let a = u0 + i as f64 * h;
            let mid = a + 0.5 * h;
            for k in 0..3 {
                let u = mid + 0.5 * h * X[k];
                let (_, d, _) = self.ders2(u);
                total += W[k] * (d[0] * d[0] + d[1] * d[1]).sqrt();
            }
        }
        0.5 * h * total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(p: P2) -> f64 {
        (p[0] * p[0] + p[1] * p[1]).sqrt()
    }

    #[test]
    fn rational_quarter_circle_is_exact() {
        // Standard rational quadratic quarter unit circle.
        let w = (0.5_f64).sqrt(); // cos(45 deg)
        let c = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![1.0, w, 1.0],
        );
        for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let p = c.eval(t);
            assert!(
                (norm(p) - 1.0).abs() < 1e-12,
                "on unit circle at {t}: |p|={}",
                norm(p)
            );
        }
        // curvature of the unit circle is 1 everywhere.
        for &t in &[0.1, 0.5, 0.9] {
            assert!(
                (c.curvature(t) - 1.0).abs() < 1e-9,
                "kappa at {t} = {}",
                c.curvature(t)
            );
        }
        // arc length of the quarter circle is pi/2.
        let l = c.arc_length(0.0, 1.0, 64);
        assert!((l - std::f64::consts::FRAC_PI_2).abs() < 1e-6, "arc {l}");
    }

    #[test]
    fn straight_segment_has_zero_curvature() {
        let c = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![[0.0, 0.0], [3.0, 4.0]],
            vec![1.0, 1.0],
        );
        assert!(c.curvature(0.5) < 1e-12);
        assert!((c.arc_length(0.0, 1.0, 4) - 5.0).abs() < 1e-12);
        let p = c.eval(0.5);
        assert!((p[0] - 1.5).abs() < 1e-12 && (p[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn parabola_curvature_matches_closed_form() {
        // Quadratic Bezier for y = x^2 near x=0 is not a parabola arc-by-arc, so
        // use the control polygon of a true parabola segment: a degree-2 Bezier
        // [(-1,1),(0,-1),(1,1)] traces y = 2x^2 - 1? Verify against the analytic
        // curvature of the traced quadratic at the apex (t=0.5 -> x=0).
        let c = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![[-1.0, 1.0], [0.0, -1.0], [1.0, 1.0]],
            vec![1.0, 1.0, 1.0],
        );
        // At t=0.5 the Bezier apex: B=(0, 0), B'=(2,0), B''=(4,4)? curvature =
        // |x'y''-y'x''|/|x'|^3. Compute numerically-independent reference.
        let (_, d1, d2) = c.ders2(0.5);
        let kappa =
            (d1[0] * d2[1] - d1[1] * d2[0]).abs() / (d1[0] * d1[0] + d1[1] * d1[1]).powf(1.5);
        assert!((c.curvature(0.5) - kappa).abs() < 1e-12);
        assert!(kappa > 0.0);
    }

    /// The span-bounded search finds the same nearest point as a dense scan,
    /// for points near, far and on the curve.
    #[test]
    fn closest_param_matches_a_dense_scan() {
        let pts: Vec<P2> = (0..40)
            .map(|i| {
                let t = i as f64 / 39.0 * std::f64::consts::TAU;
                [t.cos() * (1.0 + 0.3 * (3.0 * t).sin()), 0.6 * t.sin()]
            })
            .collect();
        let c = NurbsCurve::interpolate(&pts);
        let (lo, hi) = c.domain();
        let d = |t: f64, q: P2| {
            let x = c.eval(t);
            ((x[0] - q[0]).powi(2) + (x[1] - q[1]).powi(2)).sqrt()
        };
        for i in 0..200 {
            let a = i as f64 * 0.731;
            let r = [0.0, 0.5, 1.0, 1.3, 3.0][i % 5];
            let q = [r * a.cos(), r * a.sin()];
            let dense = (0..=20000)
                .map(|k| d(lo + (hi - lo) * k as f64 / 20000.0, q))
                .fold(f64::INFINITY, f64::min);
            let got = d(c.closest_param(q), q);
            assert!(got <= dense + 1e-9, "q {q:?}: {got} vs dense {dense}");
        }
    }
}
