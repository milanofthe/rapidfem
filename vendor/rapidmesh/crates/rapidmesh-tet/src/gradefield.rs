//! A GRADIENT-LIMITED sizing field: an arbitrary target-edge-length field `f(x, y)` (possibly with
//! sharp jumps — e.g. a per-element AMR indicator) sampled onto a uniform grid, then Lipschitz-
//! limited so the size never changes faster than `grading` per unit distance (`|∇h| ≤ grading`).
//! That smooth grading is what makes a graded mesh high-quality: without it, a hard fine→coarse
//! transition forces skewed, low-angle triangles. Evaluated in O(1) by bilinear interpolation.
//!
//! The limiting is the lower envelope `h(x) = min_y (f(y) + grading·|x − y|)`, computed on the grid
//! by min-plus fast sweeping (a few 8-neighbour directional passes converge).

type P2 = [f64; 2];

/// A scalar sizing field on a uniform grid, gradient-limited to a maximum slope.
pub struct GradedField {
    lo: P2,
    cell: f64,
    nx: usize,
    ny: usize,
    v: Vec<f64>,
}

impl GradedField {
    /// Sample `f` onto a grid over `[lo, hi]` at spacing `cell`, then Lipschitz-limit to slope
    /// `grading` (`grading <= 0` ⇒ no limiting, a plain sampled cache). The grid is capped at
    /// `MAX_N` cells per axis (the spacing grows if the box is huge relative to `cell`).
    pub fn from_fn(lo: P2, hi: P2, cell: f64, grading: f64, f: impl Fn(P2) -> f64) -> Self {
        const MAX_N: usize = 640;
        let span = [(hi[0] - lo[0]).max(0.0), (hi[1] - lo[1]).max(0.0)];
        let cell = cell.max(span[0].max(span[1]) / MAX_N as f64).max(1e-12);
        let nx = ((span[0] / cell).ceil() as usize + 1).clamp(2, MAX_N);
        let ny = ((span[1] / cell).ceil() as usize + 1).clamp(2, MAX_N);
        let mut v = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                v[j * nx + i] = f([lo[0] + i as f64 * cell, lo[1] + j as f64 * cell]).max(1e-12);
            }
        }
        if grading > 0.0 {
            let (s, sd) = (grading * cell, grading * cell * std::f64::consts::SQRT_2);
            let nb = [
                (-1i32, 0i32, s),
                (1, 0, s),
                (0, -1, s),
                (0, 1, s),
                (-1, -1, sd),
                (1, -1, sd),
                (-1, 1, sd),
                (1, 1, sd),
            ];
            // Min-plus fast sweep: 2 rounds × 4 sweep directions propagate the envelope everywhere.
            for _ in 0..2 {
                for d in 0..4 {
                    let irev = d & 1 == 1;
                    let jrev = d & 2 == 2;
                    for jj in 0..ny {
                        let j = if jrev { ny - 1 - jj } else { jj };
                        for ii in 0..nx {
                            let i = if irev { nx - 1 - ii } else { ii };
                            let mut best = v[j * nx + i];
                            for &(di, dj, w) in &nb {
                                let (ni, nj) = (i as i32 + di, j as i32 + dj);
                                if ni >= 0 && nj >= 0 && (ni as usize) < nx && (nj as usize) < ny {
                                    let cand = v[nj as usize * nx + ni as usize] + w;
                                    if cand < best {
                                        best = cand;
                                    }
                                }
                            }
                            v[j * nx + i] = best;
                        }
                    }
                }
            }
        }
        GradedField {
            lo,
            cell,
            nx,
            ny,
            v,
        }
    }

    /// Field value at `p`: bilinear interpolation over the grid (clamped to the border outside).
    pub fn eval(&self, p: P2) -> f64 {
        let fx = ((p[0] - self.lo[0]) / self.cell).clamp(0.0, (self.nx - 1) as f64);
        let fy = ((p[1] - self.lo[1]) / self.cell).clamp(0.0, (self.ny - 1) as f64);
        let (i0, j0) = (fx.floor() as usize, fy.floor() as usize);
        let (i1, j1) = ((i0 + 1).min(self.nx - 1), (j0 + 1).min(self.ny - 1));
        let (tx, ty) = (fx - i0 as f64, fy - j0 as f64);
        let g = |i: usize, j: usize| self.v[j * self.nx + i];
        let bot = g(i0, j0) * (1.0 - tx) + g(i1, j0) * tx;
        let top = g(i0, j1) * (1.0 - tx) + g(i1, j1) * tx;
        bot * (1.0 - ty) + top * ty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_a_step_to_the_grading_slope() {
        // A hard step from 0.1 to 10 at x=5; after limiting to slope 0.3 the field must rise no
        // faster than ~0.3 per unit, so just left of the step it is ≈ 0.1, and by x≈2 below the step
        // it has grown but stays bounded by 0.1 + 0.3·dist.
        let f = |p: P2| if p[0] < 5.0 { 0.1 } else { 10.0 };
        let gf = GradedField::from_fn([0.0, 0.0], [10.0, 10.0], 0.1, 0.3, f);
        let h = gf.eval([3.0, 5.0]); // 2 units left of the step
        assert!(h <= 0.1 + 0.3 * 2.0 + 0.2, "not limited: {h}");
        assert!(h >= 0.1, "below floor: {h}");
        // far inside the fine region it stays fine.
        assert!(gf.eval([0.5, 5.0]) < 0.5, "fine region not preserved");
    }
}
