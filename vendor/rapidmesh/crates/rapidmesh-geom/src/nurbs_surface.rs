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
    /// The sample grid of every knot patch, built on the first projection.
    samples: Samples,
}

/// A piece of the surface over the parameters `a..b`, with a grid of
/// samples (row by row) and the box that holds it: the box of the samples
/// grown by `sag`, twice the most the piece strays from the bilinear cells
/// of its samples.
struct Patch {
    a: [f64; 2],
    b: [f64; 2],
    points: Vec<V3>,
    lo: V3,
    hi: V3,
    sag: f64,
}

/// The pieces a projection searches, made on its first call: the knot
/// patches, each quartered until it is flat against its box. The pieces
/// follow the surface however it is parametrized: a CAD loft whose
/// parameter lines wind round the part (its control net spread far beyond
/// it) comes apart into pieces as local as those of a plain surface. A copy
/// of the surface starts without (its control points may move).
#[derive(Default)]
struct Samples(std::sync::OnceLock<Vec<Patch>>);

impl Clone for Samples {
    fn clone(&self) -> Samples {
        Samples::default()
    }
}

impl PartialEq for Samples {
    fn eq(&self, _: &Samples) -> bool {
        true
    }
}

impl std::fmt::Debug for Samples {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Samples")
    }
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
            samples: Samples::default(),
        })
    }

    /// A clamped uniform knot vector for `n` controls of degree `p`.
    pub fn clamped_knots(n: usize, p: usize) -> Vec<f64> {
        let inner = n - p;
        (0..n + p + 1)
            .map(|i| (i.saturating_sub(p)).min(inner) as f64 / inner as f64)
            .collect()
    }

    /// Clamped knots of `degree` with `spans` equal spans over `domain`.
    pub fn uniform_knots(domain: [f64; 2], degree: usize, spans: usize) -> Vec<f64> {
        NurbsSurface::clamped_knots(spans + degree, degree)
            .into_iter()
            .map(|k| domain[0] + k * (domain[1] - domain[0]))
            .collect()
    }

    /// The polynomial B-spline surface of `degree` on `knots` (clamped,
    /// per direction) fitted to `f` by least squares on a grid of samples:
    /// along `u` first, then along `v`, each by the normal equations of its
    /// banded basis. With it the largest distance from `f` on a grid twice
    /// as fine.
    pub fn fit(
        f: &(dyn Fn(f64, f64) -> V3 + Sync),
        degree: [usize; 2],
        knots: [Vec<f64>; 2],
    ) -> (NurbsSurface, f64) {
        let n = [0, 1].map(|d| knots[d].len() - degree[d] - 1);
        // Samples evenly in each knot span, `per` of them per degree and
        // span, so every basis function is sampled however uneven the
        // knots.
        let params = |d: usize, per: usize| -> Vec<f64> {
            let k = &knots[d];
            let steps = per * (degree[d] + 1);
            let mut ts: Vec<f64> = (degree[d]..n[d])
                .filter(|&s| k[s] < k[s + 1])
                .flat_map(|s| {
                    (0..steps).map(move |i| k[s] + (k[s + 1] - k[s]) * i as f64 / steps as f64)
                })
                .collect();
            ts.push(k[n[d]]);
            ts
        };
        let (pu, pv) = (params(0, 2), params(1, 2));
        let (cu, cv) = (params(0, 4), params(1, 4));
        let m = [pu.len(), pv.len()];
        // The basis at each sample, as (first control index, values).
        let basis = |d: usize, ts: &[f64]| -> Vec<(usize, Vec<f64>)> {
            ts.iter()
                .map(|&t| {
                    let span = find_span(&knots[d], n[d] - 1, degree[d], t);
                    let b = basis_funs(span, t, degree[d], &knots[d]);
                    (span - degree[d], b[..=degree[d]].to_vec())
                })
                .collect()
        };
        let (bu, bv) = (basis(0, &pu), basis(1, &pv));
        let q: Vec<V3> = pu
            .iter()
            .flat_map(|&u| pv.iter().map(move |&v| (u, v)))
            .map(|(u, v)| f(u, v))
            .collect();
        // Stage one: each sample column along u onto n[0] controls.
        let mut r = vec![[0.0; 3]; n[0] * m[1]];
        for j in 0..m[1] {
            let col: Vec<V3> = (0..m[0]).map(|i| q[i * m[1] + j]).collect();
            for (i, c) in least_squares(&bu, n[0], &col).into_iter().enumerate() {
                r[i * m[1] + j] = c;
            }
        }
        // Stage two: each row along v onto n[1] controls.
        let mut ctrl = vec![[0.0; 3]; n[0] * n[1]];
        for i in 0..n[0] {
            let row: Vec<V3> = (0..m[1]).map(|j| r[i * m[1] + j]).collect();
            for (j, c) in least_squares(&bv, n[1], &row).into_iter().enumerate() {
                ctrl[i * n[1] + j] = c;
            }
        }
        let s = NurbsSurface::new(degree, knots, n, ctrl, vec![1.0; n[0] * n[1]]);
        let mut err = 0.0f64;
        for &u in &cu {
            for &v in &cv {
                let d = sub(s.eval(u, v), f(u, v));
                err = err.max(dot(d, d).sqrt());
            }
        }
        (s, err)
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

    /// The parameters of a surface point near `q` by Newton from `start`,
    /// the parameters of a point near it (a search where the answer lies
    /// close, not over the whole surface).
    pub fn closest_param_near(&self, q: V3, start: [f64; 2]) -> [f64; 2] {
        let r = sub(self.eval(start[0], start[1]), q);
        self.descend(start, dot(r, r), q).0
    }

    /// The parameters of a surface point near `q` by one Gauss-Newton step
    /// from `start` (within the domain): for a candidate place a small
    /// offset away, where a point on the surface matters and not that it is
    /// the nearest to `q` ([`NurbsSurface::closest_param_near`] for that).
    pub fn step_toward(&self, q: V3, start: [f64; 2]) -> [f64; 2] {
        let [s, s_u, s_v, ..] = self.ders2(start[0], start[1]);
        let r = sub(q, s);
        let (a, b, c) = (dot(s_u, s_u), dot(s_u, s_v), dot(s_v, s_v));
        let (g0, g1) = (dot(r, s_u), dot(r, s_v));
        let det = a * c - b * b;
        if !(det > 0.0) {
            return start;
        }
        let (ud, vd) = self.domain();
        [
            (start[0] + (c * g0 - b * g1) / det).clamp(ud[0], ud[1]),
            (start[1] + (a * g1 - b * g0) / det).clamp(vd[0], vd[1]),
        ]
    }

    /// The parameters of the point of the surface nearest `q`: a Newton
    /// descent from the nearest sample of each piece that can hold a
    /// nearer point than found, the nearest piece first.
    pub fn closest_param(&self, q: V3) -> [f64; 2] {
        let pieces = self.pieces();
        let (ud, vd) = self.domain();
        let mut best = ([ud[0], vd[0]], f64::INFINITY);
        let first = pieces
            .iter()
            .enumerate()
            .map(|(i, piece)| (i, box_d2(piece, q)))
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .map(|(i, _)| i);
        if let Some(first) = first {
            self.descend_from(&pieces[first], q, &mut best);
            for (i, piece) in pieces.iter().enumerate() {
                if i != first && box_d2(piece, q) < best.1 {
                    self.descend_from(piece, q, &mut best);
                }
            }
        }
        best.0
    }

    /// A Newton descent toward `q` from the nearest sample of `piece`, into
    /// `best` (parameters, squared distance) if it comes nearer. It runs
    /// over the whole surface, so it reaches a nearest point beside the
    /// piece.
    fn descend_from(&self, piece: &Patch, q: V3, best: &mut ([f64; 2], f64)) {
        let cells = self.cells();
        let nj = cells[1] + 1;
        let d2 = |x: V3| dot(sub(x, q), sub(x, q));
        let Some((k, d)) = piece
            .points
            .iter()
            .map(|&p| d2(p))
            .enumerate()
            .min_by(|x, y| x.1.total_cmp(&y.1))
        else {
            return;
        };
        let t = grid_point(piece.a, piece.b, cells, [k / nj, k % nj]);
        let c = self.descend(t, d, q);
        if c.1 < best.1 {
            *best = c;
        }
    }

    /// Cells of the sample grid of a piece per direction: one per degree,
    /// four at least.
    fn cells(&self) -> [usize; 2] {
        self.degree.map(|p| p.max(4))
    }

    /// The pieces of the surface (see [`Samples`]).
    fn pieces(&self) -> &[Patch] {
        /// A piece bends little enough when its sag is at most this share
        /// of its shorter side.
        const FLAT: f64 = 0.1;
        /// The longest a piece is against its breadth.
        const ASPECT: f64 = 2.0;
        /// Halvings of a knot patch at most.
        const DEPTH: usize = 12;
        self.samples.0.get_or_init(|| {
            let spans = |d: usize| -> Vec<usize> {
                let (k, p) = (&self.knots[d], self.degree[d]);
                (p..self.n[d]).filter(|&s| k[s] < k[s + 1]).collect()
            };
            let mut todo = Vec::new();
            for su in spans(0) {
                for sv in spans(1) {
                    let a = [self.knots[0][su], self.knots[1][sv]];
                    let b = [self.knots[0][su + 1], self.knots[1][sv + 1]];
                    todo.push((a, b, 0));
                }
            }
            let mut out = Vec::new();
            while let Some((a, b, depth)) = todo.pop() {
                let (piece, sides) = self.piece(a, b);
                let (short, long) = (sides[0].min(sides[1]), sides[0].max(sides[1]));
                if depth < DEPTH && (piece.sag > FLAT * short || long > ASPECT * short) {
                    // Halved across its longer side.
                    let d = usize::from(sides[1] > sides[0]);
                    let m = 0.5 * (a[d] + b[d]);
                    let (mut b0, mut a1) = (b, a);
                    b0[d] = m;
                    a1[d] = m;
                    todo.push((a, b0, depth + 1));
                    todo.push((a1, b, depth + 1));
                } else {
                    out.push(piece);
                }
            }
            out
        })
    }

    /// The piece over `a..b`, sampled and bounded, with its sides: the
    /// longest line of its samples along `u` and along `v`.
    fn piece(&self, a: [f64; 2], b: [f64; 2]) -> (Patch, [f64; 2]) {
        let cells = self.cells();
        let nj = cells[1] + 1;
        let at = |ij: [f64; 2]| {
            self.eval(
                a[0] + (b[0] - a[0]) * ij[0] / cells[0] as f64,
                a[1] + (b[1] - a[1]) * ij[1] / cells[1] as f64,
            )
        };
        let mut points = Vec::with_capacity((cells[0] + 1) * nj);
        for i in 0..=cells[0] {
            for j in 0..=cells[1] {
                points.push(at([i as f64, j as f64]));
            }
        }
        // The most the piece strays from its bilinear cells: at the middle
        // of each cell and of each of its sides.
        let point = |i: usize, j: usize| points[i * nj + j];
        let off = |x: [f64; 2], corners: &[V3]| -> f64 {
            let n = corners.len() as f64;
            let m: V3 = std::array::from_fn(|k| corners.iter().map(|c| c[k]).sum::<f64>() / n);
            let d = sub(at(x), m);
            dot(d, d).sqrt()
        };
        let mut sag = 0.0f64;
        for i in 0..=cells[0] {
            for j in 0..=cells[1] {
                let (x, y) = (i as f64, j as f64);
                if i < cells[0] {
                    sag = sag.max(off([x + 0.5, y], &[point(i, j), point(i + 1, j)]));
                }
                if j < cells[1] {
                    sag = sag.max(off([x, y + 0.5], &[point(i, j), point(i, j + 1)]));
                }
                if i < cells[0] && j < cells[1] {
                    let c = [
                        point(i, j),
                        point(i + 1, j),
                        point(i, j + 1),
                        point(i + 1, j + 1),
                    ];
                    sag = sag.max(off([x + 0.5, y + 0.5], &c));
                }
            }
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for p in &points {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let length = |p: V3, r: V3| dot(sub(p, r), sub(p, r)).sqrt();
        let along_u = (0..=cells[1])
            .map(|j| {
                (0..cells[0])
                    .map(|i| length(point(i, j), point(i + 1, j)))
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let along_v = (0..=cells[0])
            .map(|i| {
                (0..cells[1])
                    .map(|j| length(point(i, j), point(i, j + 1)))
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let piece = Patch {
            a,
            b,
            points,
            lo,
            hi,
            sag: 2.0 * sag,
        };
        (piece, [along_u, along_v])
    }

    /// Newton steps toward the point nearest `q` from `t` (at squared
    /// distance `dist`) within the domain, halved until the distance drops,
    /// until a step is small: one more is taken if it helps, the next would
    /// be smaller than the rounding of the first.
    fn descend(&self, mut t: [f64; 2], mut dist: f64, q: V3) -> ([f64; 2], f64) {
        let (ud, vd) = self.domain();
        let (lo, hi) = ([ud[0], vd[0]], [ud[1], vd[1]]);
        let small = [1e-9 * (hi[0] - lo[0]), 1e-9 * (hi[1] - lo[1])];
        let d2 = |x: V3| dot(sub(x, q), sub(x, q));
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
            let mut step = if h[0] > 0.0 && det > 0.0 {
                [
                    (h[2] * g[0] - h[1] * g[1]) / det,
                    (h[0] * g[1] - h[1] * g[0]) / det,
                ]
            } else {
                let m = dot(s_u, s_u).max(dot(s_v, s_v)).max(f64::MIN_POSITIVE);
                [g[0] / m, g[1] / m]
            };
            // On a side of the domain and heading out: along the side, by
            // Newton in the other parameter alone.
            let out = [0, 1]
                .map(|k| (t[k] <= lo[k] && step[k] > 0.0) || (t[k] >= hi[k] && step[k] < 0.0));
            if out[0] || out[1] {
                let curv = [h[0], h[2]];
                let tangent = [dot(s_u, s_u), dot(s_v, s_v)];
                for k in 0..2 {
                    step[k] = if out[k] {
                        0.0
                    } else if curv[k] > 0.0 {
                        g[k] / curv[k]
                    } else {
                        g[k] / tangent[k].max(f64::MIN_POSITIVE)
                    };
                }
            }
            let mut lambda = 1.0;
            let mut moved = false;
            for _ in 0..16 {
                let c = [
                    (t[0] - lambda * step[0]).clamp(lo[0], hi[0]),
                    (t[1] - lambda * step[1]).clamp(lo[1], hi[1]),
                ];
                let small = (c[0] - t[0]).abs() <= small[0] && (c[1] - t[1]).abs() <= small[1];
                let d = d2(self.eval(c[0], c[1]));
                if d < dist {
                    t = c;
                    dist = d;
                    moved = !small;
                    break;
                }
                // A small step that does not help is rounding: there.
                if small {
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

/// Point `ij` of a grid of `cells` over `a..b`.
fn grid_point(a: [f64; 2], b: [f64; 2], cells: [usize; 2], ij: [usize; 2]) -> [f64; 2] {
    [
        a[0] + (b[0] - a[0]) * ij[0] as f64 / cells[0] as f64,
        a[1] + (b[1] - a[1]) * ij[1] as f64 / cells[1] as f64,
    ]
}

/// The squared distance from `q` to the box `lo..hi` grown by `grow`.
fn outside_d2(lo: V3, hi: V3, grow: f64, q: V3) -> f64 {
    (0..3)
        .map(|k| ((lo[k] - grow - q[k]).max(q[k] - hi[k] - grow).max(0.0)).powi(2))
        .sum()
}

/// The squared distance from `q` to the box of `piece`.
fn box_d2(piece: &Patch, q: V3) -> f64 {
    outside_d2(piece.lo, piece.hi, piece.sag, q)
}

/// The `n` coefficients whose B-spline (basis per sample in `basis`)
/// fits `y` best in the least-squares sense: the normal equations, solved
/// by Cholesky.
fn least_squares(basis: &[(usize, Vec<f64>)], n: usize, y: &[V3]) -> Vec<V3> {
    let mut a = vec![0.0; n * n];
    let mut rhs = vec![[0.0; 3]; n];
    for ((first, b), yv) in basis.iter().zip(y) {
        for (k, &bk) in b.iter().enumerate() {
            let i = first + k;
            for c in 0..3 {
                rhs[i][c] += bk * yv[c];
            }
            for (l, &bl) in b.iter().enumerate() {
                a[i * n + first + l] += bk * bl;
            }
        }
    }
    // Cholesky in place: a = L L^T.
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        let d = d.max(1e-300).sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    let mut x = rhs;
    for i in 0..n {
        for k in 0..i {
            let l = a[i * n + k];
            for c in 0..3 {
                x[i][c] -= l * x[k][c];
            }
        }
        let d = a[i * n + i];
        x[i] = x[i].map(|v| v / d);
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            let l = a[k * n + i];
            for c in 0..3 {
                x[i][c] -= l * x[k][c];
            }
        }
        let d = a[i * n + i];
        x[i] = x[i].map(|v| v / d);
    }
    x
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

    /// A fit of a smooth surface converges: halving the spans takes the
    /// error down by far more than half.
    #[test]
    fn a_quintic_fit_converges_on_a_smooth_surface() {
        let f = |u: f64, v: f64| [u, v, (3.0 * u).sin() * (2.0 * v).cos()];
        let fit = |spans| {
            let knots = [
                NurbsSurface::uniform_knots([0.0, 2.0], 5, spans),
                NurbsSurface::uniform_knots([0.0, 1.0], 5, spans),
            ];
            NurbsSurface::fit(&f, [5, 5], knots)
        };
        let ((_, e4), (s8, e8)) = (fit(4), fit(8));
        assert!(e8 < e4 / 20.0, "{e4} {e8}");
        assert!(e8 < 1e-5, "{e8}");
        let p = s8.eval(1.3, 0.4);
        let q = f(1.3, 0.4);
        assert!((0..3).all(|k| (p[k] - q[k]).abs() < 1e-5));
    }

    /// The projection onto a band whose parameter lines wind round it (its
    /// control net spread far beyond it, as a CAD loft has it) comes no
    /// farther than the nearest of a dense grid of samples.
    #[test]
    fn the_projection_onto_a_winding_band_finds_the_nearest_point() {
        let f = |u: f64, v: f64| {
            let (t, r) = (std::f64::consts::TAU * u + 3.0 * v, 10.0 - 4.0 * v);
            [r * t.cos(), r * t.sin(), 36.0 * v]
        };
        let knots = [
            NurbsSurface::uniform_knots([0.0, 1.0], 5, 16),
            NurbsSurface::uniform_knots([0.0, 1.0], 3, 1),
        ];
        let (s, _) = NurbsSurface::fit(&f, [5, 3], knots);
        let mut seed = 12345u64;
        let mut r = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 1_000_000) as f64 / 1e6
        };
        let d = |p: V3, q: V3| dot(sub(p, q), sub(p, q)).sqrt();
        for _ in 0..40 {
            let p = s.eval(r(), r());
            let q = [
                p[0] + 2.0 * r() - 1.0,
                p[1] + 2.0 * r() - 1.0,
                p[2] + 2.0 * r() - 1.0,
            ];
            let uv = s.closest_param(q);
            let got = d(s.eval(uv[0], uv[1]), q);
            let mut dense = f64::INFINITY;
            for i in 0..=300 {
                for j in 0..=60 {
                    dense = dense.min(d(s.eval(i as f64 / 300.0, j as f64 / 60.0), q));
                }
            }
            assert!(got <= dense + 1e-9, "{q:?}: {got} against {dense}");
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
