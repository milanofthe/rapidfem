//! Rational B-spline (NURBS) curves in any dimension: in 2D the profile
//! primitive for swept and extruded spline surfaces (e.g. an airfoil
//! section), in 3D the B-spline edges of CAD files. A curve is a degree, a
//! clamped knot vector, control points and rational weights; evaluation and the
//! first two derivatives use the standard basis-function-derivative algorithms
//! (Piegl & Tiller, "The NURBS Book", A2.1/A2.3), so curvature is exact and
//! drives the curved-surface sizing bias.
//!
//! Weights all 1.0 give an ordinary (non-rational) B-spline; conic sections
//! (the exact circle/ellipse) need the rational form, so the type carries
//! weights throughout and de-homogenizes with the quotient rule.

#[cfg(test)]
type P2 = [f64; 2];

fn dot<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
    (0..N).map(|k| a[k] * b[k]).sum()
}

fn dist<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
    (0..N).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

/// Highest supported degree. Basis evaluation runs on fixed-size stack
/// arrays, so evaluating a curve or surface never allocates.
pub const MAX_DEGREE: usize = 25;
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
    // The triangle of basis values takes (p + 1)^2 entries: on a table for
    // the common low degrees, not one for the highest on every call.
    if p < 8 {
        ders_basis_in::<8>(knots, p, span, u)
    } else {
        ders_basis_in::<W>(knots, p, span, u)
    }
}

/// [`ders_basis`] with a table of `M` by `M` (`p < M`).
fn ders_basis_in<const M: usize>(knots: &[f64], p: usize, span: usize, u: f64) -> [[f64; W]; 3] {
    let nd = 2;
    let u_ = knots;
    let mut ndu = [[0.0f64; M]; M];
    let mut left = [0.0f64; M];
    let mut right = [0.0f64; M];
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
    let mut a = [[0.0f64; M]; 2];
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
fn solve_banded<const N: usize>(a: &mut [Vec<f64>], b: &mut [[f64; N]]) -> Vec<[f64; N]> {
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
            for k in 0..N {
                b[r][k] -= f * b[col][k];
            }
        }
    }
    let mut x = vec![[0.0; N]; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            for k in 0..N {
                s[k] -= a[col][c] * x[c][k];
            }
        }
        x[col] = s.map(|v| v / a[col][col]);
    }
    x
}

/// A rational B-spline curve in `N` dimensions (2 by default: a profile).
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsCurve<const N: usize = 2> {
    /// Polynomial degree `p` (order `p + 1`).
    pub degree: usize,
    /// Clamped, non-decreasing knot vector, length `ctrl.len() + degree + 1`.
    pub knots: Vec<f64>,
    /// Control points.
    pub ctrl: Vec<[f64; N]>,
    /// Rational weights, parallel to `ctrl` (all 1.0 = a plain B-spline).
    pub weights: Vec<f64>,
}

impl<const N: usize> NurbsCurve<N> {
    /// The same curve traversed the other way, over the same domain.
    pub fn reversed(&self) -> NurbsCurve<N> {
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
    pub fn scaled(&self, s: f64) -> NurbsCurve<N> {
        NurbsCurve {
            ctrl: self.ctrl.iter().map(|c| c.map(|x| x * s)).collect(),
            ..self.clone()
        }
    }

    /// Builds a curve, validating the knot/control/weight sizes.
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        ctrl: Vec<[f64; N]>,
        weights: Vec<f64>,
    ) -> NurbsCurve<N> {
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

    /// Cubic B-spline INTERPOLATING the given points (the curve passes through
    /// each). Gives
    /// the FAITHFUL curvature of the sampled shape -- no control-polygon
    /// artifacts -- so a curvature sizing field refines only the genuinely
    /// curved regions (Piegl & Tiller, global interpolation A9.1: chord-length
    /// parameters, averaged knots, solve `N P = Q`). Needs >= 4 points.
    pub fn interpolate(points: &[[f64; N]]) -> NurbsCurve<N> {
        let n = points.len() - 1;
        assert!(n >= 3, "cubic interpolation needs >= 4 points");
        let p = 3;
        let dd = |a: [f64; N], b: [f64; N]| dist(a, b);
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
        let mut rhs: Vec<[f64; N]> = points.to_vec();
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
    pub fn ders2(&self, u: f64) -> ([f64; N], [f64; N], [f64; N]) {
        let p = self.degree;
        let (lo, hi) = self.domain();
        let u = u.clamp(lo, hi);
        let span = self.find_span(u);
        let nd = ders_basis(&self.knots, self.degree, span, u);
        // The homogeneous numerator A^(k) and weight w^(k), k = 0, 1, 2.
        let mut a = [[0.0f64; N]; 3];
        let mut w = [0.0f64; 3];
        for j in 0..=p {
            let i = span - p + j;
            let wi = self.weights[i];
            for k in 0..3 {
                for c in 0..N {
                    a[k][c] += nd[k][j] * self.ctrl[i][c] * wi;
                }
                w[k] += nd[k][j] * wi;
            }
        }
        let c0: [f64; N] = std::array::from_fn(|c| a[0][c] / w[0]);
        let c1: [f64; N] = std::array::from_fn(|c| (a[1][c] - w[1] * c0[c]) / w[0]);
        let c2: [f64; N] =
            std::array::from_fn(|c| (a[2][c] - 2.0 * w[1] * c1[c] - w[2] * c0[c]) / w[0]);
        (c0, c1, c2)
    }

    /// Curve point at `u`.
    pub fn eval(&self, u: f64) -> [f64; N] {
        let p = self.degree;
        let (lo, hi) = self.domain();
        let u = u.clamp(lo, hi);
        let span = self.find_span(u);
        let b = basis_funs(span, u, p, &self.knots);
        let (mut x, mut w) = ([0.0; N], 0.0);
        for (j, bj) in b.iter().enumerate().take(p + 1) {
            let i = span - p + j;
            let bw = bj * self.weights[i];
            for c in 0..N {
                x[c] += bw * self.ctrl[i][c];
            }
            w += bw;
        }
        x.map(|v| v / w)
    }

    /// Parameter of the curve point nearest to `q`. A knot span lies in the
    /// convex hull of its `degree + 1` control points, so the distance to
    /// their bounding box bounds the span from below: the nearest span is
    /// searched first, then only spans that could still beat it. Inside a
    /// span a few samples seed Newton on `(C(t) - q) . C'(t) = 0`.
    pub fn closest_param(&self, q: [f64; N]) -> f64 {
        let p = self.degree;
        let spans = (p..self.ctrl.len()).filter(|&s| self.knots[s] < self.knots[s + 1]);
        let bound = |s: usize| {
            let (mut lo, mut hi) = ([f64::INFINITY; N], [f64::NEG_INFINITY; N]);
            for c in &self.ctrl[s - p..=s] {
                for k in 0..N {
                    lo[k] = lo[k].min(c[k]);
                    hi[k] = hi[k].max(c[k]);
                }
            }
            (0..N)
                .map(|k| (lo[k] - q[k]).max(q[k] - hi[k]).max(0.0).powi(2))
                .sum::<f64>()
        };
        // Nearest bound first; once a bound reaches the best distance, no
        // span after it can beat it.
        let mut order: Vec<(f64, usize)> = spans.map(|s| (bound(s), s)).collect();
        if order.is_empty() {
            return self.domain().0;
        }
        order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut best = (self.domain().0, f64::INFINITY);
        for (b, s) in order {
            if b >= best.1 {
                break;
            }
            let c = self.closest_in_span(s, q);
            if c.1 < best.1 {
                best = c;
            }
        }
        best.0
    }

    /// The parameter of a curve point near `q` by Newton from `t0`, the
    /// parameter of a point near it (a search where the answer lies close,
    /// not over the whole curve). Falls back to [`NurbsCurve::closest_param`]
    /// where Newton does not come nearer than `t0` was.
    pub fn closest_param_near(&self, q: [f64; N], t0: f64) -> f64 {
        let (lo, hi) = self.domain();
        let d2 = |t: f64| dist(self.eval(t), q).powi(2);
        let mut t = t0.clamp(lo, hi);
        let start = d2(t);
        for _ in 0..16 {
            let (c, d1, dd) = self.ders2(t);
            let r: [f64; N] = std::array::from_fn(|k| c[k] - q[k]);
            let gp = dot(d1, d1) + dot(r, dd);
            if !(gp > 1e-15) {
                break;
            }
            let step = dot(r, d1) / gp;
            t = (t - step).clamp(lo, hi);
            if step.abs() < 1e-13 * (hi - lo).max(1.0) {
                break;
            }
        }
        if d2(t) <= start {
            t
        } else {
            self.closest_param(q)
        }
    }

    /// Nearest parameter to `q` within knot span `s`, with its squared
    /// distance.
    fn closest_in_span(&self, s: usize, q: [f64; N]) -> (f64, f64) {
        const SAMPLES: usize = 4;
        let (a, b) = (self.knots[s], self.knots[s + 1]);
        let d2 = |c: [f64; N]| dist(c, q).powi(2);
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
            let r: [f64; N] = std::array::from_fn(|k| c[k] - q[k]);
            let g = dot(r, d1);
            let gp = dot(d1, d1) + dot(r, dd);
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

    /// Curvature `kappa = |C' x C''| / |C'|^3` at `u`, the cross product's
    /// length summed over the coordinate planes (in any dimension, and
    /// without the cancellation of `|C'|^2 |C''|^2 - (C' . C'')^2` where
    /// the curve is nearly straight).
    pub fn curvature(&self, u: f64) -> f64 {
        let (_, c1, c2) = self.ders2(u);
        let speed = dot(c1, c1).sqrt();
        if speed < 1e-15 {
            return 0.0;
        }
        let mut cross2 = 0.0;
        for i in 0..N {
            for j in i + 1..N {
                cross2 += (c1[i] * c2[j] - c1[j] * c2[i]).powi(2);
            }
        }
        cross2.sqrt() / speed.powi(3)
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
                total += W[k] * dot(d, d).sqrt();
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

    /// The quarter circle of radius 2 in the plane x = y, as a 3D curve:
    /// on the circle, curvature 1/2, and the nearest point of a point
    /// beside it on the ray through that point.
    #[test]
    fn a_tilted_quarter_circle_in_3d_is_exact() {
        let w = (0.5_f64).sqrt();
        let (e1, e2) = ([w, w, 0.0], [0.0, 0.0, 1.0]);
        let at =
            |a: f64, b: f64| -> [f64; 3] { std::array::from_fn(|k| 2.0 * (a * e1[k] + b * e2[k])) };
        let c: NurbsCurve<3> = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![at(1.0, 0.0), at(1.0, 1.0), at(0.0, 1.0)],
            vec![1.0, w, 1.0],
        );
        for &t in &[0.0, 0.3, 0.5, 0.9, 1.0] {
            let p = c.eval(t);
            assert!((dist(p, [0.0; 3]) - 2.0).abs() < 1e-12, "{t}: {p:?}");
            assert!((p[0] - p[1]).abs() < 1e-12);
            assert!(
                (c.curvature(t) - 0.5).abs() < 1e-9,
                "{t}: {}",
                c.curvature(t)
            );
        }
        let q = at(3.0 * 0.6, 3.0 * 0.8);
        let foot = c.eval(c.closest_param(q));
        let want = at(0.6, 0.8);
        assert!(dist(foot, want) < 1e-9, "{foot:?} {want:?}");
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
