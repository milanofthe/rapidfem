//! Variable-distance inward offset of a planar region, and the local width it is built on.
//!
//! A mesher that wants thin elements along a boundary needs two things a polygon does not
//! carry by itself: how WIDE the shape is at each boundary point, and where a curve at a
//! given distance inside the boundary runs. Both are here, as methods on [`Region2D`]:
//!
//! - [`Region2D::local_width`] — the width of the shape at a boundary point, from the
//!   largest inscribed ball tangent there;
//! - [`Region2D::offset_chains`] — the inward offsets of the boundary at a distance that
//!   VARIES along it, ready to hand back as [`Region2D::constraints`].
//!
//! The offset is not a buffer: the distance is a function of the local width, so a trace that
//! narrows carries a row that narrows with it, and a wide pad next to a narrow one keeps its
//! own pitch. That is what makes the result usable as an anisotropic boundary layer.

use crate::bundle::{Mesh2DOptions, Region2D};

/// The grading of offset chains where the caller sets none: rows grow
/// away from the boundary by at most this slope.
pub const OFFSET_GRADING: f64 = 0.3;

/// `(a, b, outward unit normal)` of one boundary segment.
type Segment = ([f64; 2], [f64; 2], [f64; 2]);

/// Shoelace signed area (positive = counter-clockwise).
fn signed_area(ring: &[[f64; 2]]) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        * 0.5
}

/// Duplicate and repeated closing vertices removed: a zero-length segment has no normal, and
/// layout rings routinely repeat their first point at the end.
fn clean(ring: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(ring.len());
    for &q in ring {
        if out
            .last()
            .is_none_or(|p: &[f64; 2]| (p[0] - q[0]).abs() > 1e-15 || (p[1] - q[1]).abs() > 1e-15)
        {
            out.push(q);
        }
    }
    while out.len() > 1
        && (out[0][0] - out[out.len() - 1][0]).abs() <= 1e-15
        && (out[0][1] - out[out.len() - 1][1]).abs() <= 1e-15
    {
        out.pop();
    }
    out
}

/// The boundary of a region as directed segments with their OUTWARD normals.
struct Boundary {
    segs: Vec<Segment>,
}

impl Boundary {
    fn new(rings: &[Vec<[f64; 2]>]) -> Boundary {
        let mut segs = Vec::new();
        for (ri, ring) in rings.iter().enumerate() {
            let n = ring.len();
            if n < 3 {
                continue;
            }
            // The interior lies left of a CCW outer ring and right of a CCW hole ring.
            let left = (signed_area(ring) > 0.0) == (ri == 0);
            let sgn = if left { 1.0 } else { -1.0 };
            for i in 0..n {
                let (a, b) = (ring[i], ring[(i + 1) % n]);
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let len = dx.hypot(dy);
                if len > 0.0 {
                    segs.push((a, b, [dy / len * sgn, -dx / len * sgn]));
                }
            }
        }
        Boundary { segs }
    }

    /// Distance from `p` to the nearest point of the boundary.
    fn distance(&self, p: [f64; 2]) -> f64 {
        let mut best = f64::INFINITY;
        for &(a, b, _) in &self.segs {
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let l2 = ex * ex + ey * ey;
            let t = if l2 > 0.0 {
                (((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            best = best.min((p[0] - a[0] - t * ex).hypot(p[1] - a[1] - t * ey));
        }
        best
    }

    /// Twice the radius of the largest ball tangent to the boundary at `p` that is stopped by
    /// the wall FACING `p`. See [`Region2D::local_width`].
    fn width_at(&self, p: [f64; 2], n: [f64; 2]) -> f64 {
        let mut r = f64::INFINITY;
        for &(a, b, m) in &self.segs {
            if m[0] * n[0] + m[1] * n[1] <= 1e-9 {
                continue;
            }
            let (ax, ay) = (a[0] - p[0], a[1] - p[1]);
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let (na, ne) = (n[0] * ax + n[1] * ay, n[0] * ex + n[1] * ey);
            let (ee, ae, aa) = (ex * ex + ey * ey, ax * ex + ay * ey, ax * ax + ay * ay);
            // d/dt of |u|^2 / (2 n.u) with u = a - p + t e, cleared of its denominator:
            //   t^2 |e|^2 (n.e) + 2 t |e|^2 (n.a) + 2 (a.e)(n.a) - (n.e)|a|^2 = 0
            let (qa, qb, qc) = (ee * ne, 2.0 * ee * na, 2.0 * ae * na - ne * aa);
            let mut ts = [0.0, 1.0, f64::NAN, f64::NAN];
            if qa.abs() > 1e-300 {
                let disc = qb * qb - 4.0 * qa * qc;
                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    ts[2] = (-qb + sq) / (2.0 * qa);
                    ts[3] = (-qb - sq) / (2.0 * qa);
                }
            } else if qb.abs() > 1e-300 {
                ts[2] = -qc / qb;
            }
            for t in ts {
                if !(0.0..=1.0).contains(&t) {
                    continue;
                }
                let (ux, uy) = (ax + t * ex, ay + t * ey);
                let den = 2.0 * (n[0] * ux + n[1] * uy);
                let num = ux * ux + uy * uy;
                // `den <= 0`: the point lies on or behind the tangent line, no ball through it.
                if den > 1e-300 && num > 0.0 {
                    r = r.min(num / den);
                }
            }
        }
        2.0 * r
    }
}

/// The local width of regions, as a field over the plane: the width at
/// samples along each boundary (segment middles, long segments in pieces
/// about as long as the width there), read at the sample nearest a point.
pub(crate) struct WidthField {
    samples: Vec<([f64; 2], f64)>,
    cell: f64,
    grid: std::collections::HashMap<(i64, i64), Vec<usize>>,
}

impl WidthField {
    /// The field of `patches`, each its outer ring and its holes.
    pub(crate) fn new(patches: &[(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)]) -> WidthField {
        let mut samples: Vec<([f64; 2], f64)> = Vec::new();
        for (outer, holes) in patches {
            let rings: Vec<Vec<[f64; 2]>> = std::iter::once(outer)
                .chain(holes.iter())
                .map(|r| clean(r))
                .collect();
            let b = Boundary::new(&rings);
            for &(a, c, m) in &b.segs {
                let inward = [-m[0], -m[1]];
                let len = (c[0] - a[0]).hypot(c[1] - a[1]);
                let at = |t: f64| [a[0] + t * (c[0] - a[0]), a[1] + t * (c[1] - a[1])];
                let w = b.width_at(at(0.5), inward);
                let pieces = if w.is_finite() && w > 0.0 {
                    ((len / w).ceil() as usize).clamp(1, 64)
                } else {
                    1
                };
                for k in 0..pieces {
                    let q = at((k as f64 + 0.5) / pieces as f64);
                    samples.push((q, b.width_at(q, inward)));
                }
            }
        }
        let mut widths: Vec<f64> = samples
            .iter()
            .map(|s| s.1)
            .filter(|w| w.is_finite())
            .collect();
        widths.sort_by(f64::total_cmp);
        let cell = widths
            .get(widths.len() / 2)
            .copied()
            .unwrap_or(1.0)
            .max(1e-12);
        let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = Default::default();
        let key = |p: [f64; 2]| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64);
        for (i, s) in samples.iter().enumerate() {
            grid.entry(key(s.0)).or_default().push(i);
        }
        WidthField {
            samples,
            cell,
            grid,
        }
    }

    /// The width at `p`: the least width of the samples within their own
    /// width of it (so a trace's side governs up to its ends, not the end
    /// wall that faces the far end), else the width of the nearest sample
    /// (INFINITY without any).
    pub(crate) fn at(&self, p: [f64; 2]) -> f64 {
        let c = (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
        );
        let mut near = f64::INFINITY;
        let mut nearest = (f64::INFINITY, f64::INFINITY);
        for r in 0..4096i64 {
            for x in c.0 - r..=c.0 + r {
                for y in c.1 - r..=c.1 + r {
                    if (x - c.0).abs().max((y - c.1).abs()) != r {
                        continue;
                    }
                    for &i in self.grid.get(&(x, y)).into_iter().flatten() {
                        let (q, w) = self.samples[i];
                        let d = (q[0] - p[0]).hypot(q[1] - p[1]);
                        if d <= w {
                            near = near.min(w);
                        }
                        if d < nearest.0 {
                            nearest = (d, w);
                        }
                    }
                }
            }
            // Past the typical width (a cell) and the nearest sample, no
            // farther sample reaches `p` through its width in practice.
            if r >= 2 && nearest.0 <= r as f64 * self.cell {
                break;
            }
        }
        if near.is_finite() {
            near
        } else {
            nearest.1
        }
    }
}

/// Where the two offset lines meeting at vertex `i` cross, each at its own distance.
///
/// Parallel neighbours give the offset point itself; a sharp convex corner whose mitre runs
/// away is capped to a bevel at twice the larger distance, so a spike does not throw a point
/// across the shape.
fn offset_vertex(
    ring: &[[f64; 2]],
    i: usize,
    d_prev: f64,
    d_cur: f64,
    left: bool,
) -> Option<[f64; 2]> {
    let n = ring.len();
    let sgn = if left { 1.0 } else { -1.0 };
    let line = |j: usize, d: f64| -> Option<([f64; 2], [f64; 2])> {
        let (a, b) = (ring[j], ring[(j + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = dx.hypot(dy);
        if len <= 0.0 {
            return None;
        }
        let nrm = [-dy / len * sgn, dx / len * sgn];
        Some(([a[0] + nrm[0] * d, a[1] + nrm[1] * d], [dx / len, dy / len]))
    };
    let (p0, u0) = line((i + n - 1) % n, d_prev)?;
    let (p1, u1) = line(i, d_cur)?;
    let det = u0[0] * u1[1] - u0[1] * u1[0];
    let corner = ring[i];
    let q = if det.abs() < 1e-9 {
        p1
    } else {
        let (rx, ry) = (p1[0] - p0[0], p1[1] - p0[1]);
        let t = (rx * u1[1] - ry * u1[0]) / det;
        [p0[0] + u0[0] * t, p0[1] + u0[1] * t]
    };
    let m = (q[0] - corner[0]).hypot(q[1] - corner[1]);
    let cap = 2.0 * d_prev.max(d_cur);
    if m > cap && m > 0.0 {
        let s = cap / m;
        return Some([
            corner[0] + (q[0] - corner[0]) * s,
            corner[1] + (q[1] - corner[1]) * s,
        ]);
    }
    Some(q)
}

/// The maximal runs of surviving points as chains: ONE closed ring when every point survived,
/// else one OPEN polyline per run (a run of a single point carries no segment and is dropped).
fn runs_to_chains(pts: &[Option<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    if pts.iter().all(|p| p.is_some()) {
        let mut ring: Vec<[f64; 2]> = pts.iter().map(|p| p.unwrap()).collect();
        ring.push(ring[0]);
        return vec![ring];
    }
    let start = match pts.iter().position(|p| p.is_none()) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let mut chains = Vec::new();
    let mut cur: Vec<[f64; 2]> = Vec::new();
    for step in 1..=n {
        match pts[(start + step) % n] {
            Some(q) => cur.push(q),
            None => {
                if cur.len() >= 2 {
                    chains.push(std::mem::take(&mut cur));
                } else {
                    cur.clear();
                }
            }
        }
    }
    if cur.len() >= 2 {
        chains.push(cur);
    }
    chains
}

impl Region2D {
    /// The width of the region at the boundary point `p` whose INWARD unit normal is `inward`:
    /// twice the radius of the largest ball that is tangent to the boundary at `p` and touches
    /// the wall ACROSS from it. `INFINITY` when nothing faces `p`.
    ///
    /// For any boundary point `q` the ball tangent at `p` that also passes through `q` has
    /// radius `|q-p|^2 / (2 n.(q-p))`, so the largest inscribed one is the smallest of those
    /// over the boundary — the distance from `p` to the medial axis, doubled. Two details
    /// decide whether the number is the local WIDTH or something else:
    ///
    /// * only segments whose outward normal has a positive component along `inward` count.
    ///   They are the wall facing `p`. Without that filter the END of a strip stops the ball of
    ///   every point near it and the width collapses into every convex corner, which is a
    ///   property of the corner and not of the shape: 2 units from its end, a 20-unit-wide
    ///   strip is still 20 wide.
    /// * the minimum runs over each SEGMENT, not over its endpoints. A ball is routinely
    ///   stopped by the middle of an edge, and sampling vertices alone reads straight past it.
    ///
    /// This is the canonical input for an automatic sizing field: `local_width / k` asks for
    /// `k` elements across the shape wherever it happens to be.
    pub fn local_width(&self, p: [f64; 2], inward: [f64; 2]) -> f64 {
        // The direction only: any length of `inward` measures the same.
        let n = inward[0].hypot(inward[1]);
        let inward = if n > 0.0 {
            [inward[0] / n, inward[1] / n]
        } else {
            inward
        };
        let rings: Vec<Vec<[f64; 2]>> = std::iter::once(&self.outer)
            .chain(self.holes.iter())
            .map(|r| clean(r))
            .collect();
        Boundary::new(&rings).width_at(p, inward)
    }

    /// Inward offsets of the boundary at `scales` multiples of a distance that VARIES along it,
    /// as chains fit to hand back through [`Region2D::constraints`].
    ///
    /// `pitch` maps the local width (see [`Region2D::local_width`]) to the offset distance
    /// there, so `|w| w / 16.0` lays the first row a sixteenth of the local width inside the
    /// boundary — narrow where the shape is narrow, wide where it is wide. `scales` are the
    /// multiples to emit, e.g. `[1.0, 3.0, 7.0]` for rows of pitch `d`, `2d`, `4d` growing away
    /// from the boundary. From [`Mesh2DOptions`] it reads `grading` and `minh`, and nothing else.
    ///
    /// It is a plain variable-distance offset, built the way one is built:
    ///
    /// 1. the pitch is measured continuously along the boundary, with enough samples per
    ///    segment to follow a width that changes along it;
    /// 2. the pitch field is floored at `minh` and made `grading`-Lipschitz along the boundary,
    ///    CORNERS INCLUDED — the two sides of a corner are at zero distance from each other, so
    ///    they end up equal. Without that a 20-wide edge meeting its 600-wide end would have to
    ///    jump by a factor of 30 across one vertex, and a mitre across such a jump lands deep
    ///    inside the shape;
    /// 3. each row offsets the segments along their own normals and the vertices at the mitre
    ///    of the two adjacent offset lines (bevelled at twice the pitch);
    /// 4. a point survives while its distance to the boundary still IS its offset. Where it is
    ///    less, the point has crossed the medial axis and the chain BREAKS there, so the row
    ///    stops at a neck instead of the whole ring being lost;
    /// 5. a row is emitted only where it FITS: the same row comes in from the facing wall, so
    ///    the two together may not take more than the shape is wide. Without that the rows of a
    ///    narrow trace meet on its centre line and leave a sliver only skewed elements can fill;
    /// 6. an interior sample whose pitch is the straight interpolation of its neighbours is a
    ///    point ON the chain and buys nothing but elements, so it is dropped. A uniform edge
    ///    falls back to its two corner points this way.
    ///
    /// The `minh` floor is the point of taking [`Mesh2DOptions`] rather than the two numbers:
    /// a chain is a CONSTRAINT, the mesher must reproduce it, so a row finer than the floor
    /// cannot be repaired downstream. A shape whose local width collapses somewhere — a 0.2
    /// waist in a 20-wide trace — would otherwise ask for a row four orders of magnitude below
    /// the base cell.
    pub fn offset_chains(
        &self,
        pitch: impl Fn(f64) -> f64,
        scales: &[f64],
        opts: &Mesh2DOptions,
    ) -> Vec<Vec<[f64; 2]>> {
        let mut out = Vec::new();
        if scales.is_empty() {
            return out;
        }
        let rings: Vec<Vec<[f64; 2]>> = std::iter::once(&self.outer)
            .chain(self.holes.iter())
            .map(|r| clean(r))
            .collect();
        let boundary = Boundary::new(&rings);
        let grade = opts.grading.max(1e-3);
        // The OUTERMOST row decides which samples the corner wedges swallow, and that decision
        // is then the same for every row. Rows built from different sample sets are rows of
        // different shape: the inner one follows the graded pitch while the outer one, having
        // lost its samples, runs straight from mitre to mitre, and at the end of a strip the
        // two cross. Nested rows cannot cross, since at a shared sample the offsets only grow.
        let reach = scales.iter().cloned().fold(0.0, f64::max);

        for (ri, ring) in rings.iter().enumerate() {
            let n = ring.len();
            if n < 3 {
                continue;
            }
            let left = (signed_area(ring) > 0.0) == (ri == 0);
            let sgn = if left { 1.0 } else { -1.0 };
            let geom = |i: usize| -> ([f64; 2], [f64; 2], f64, [f64; 2]) {
                let (a, b) = (ring[i], ring[(i + 1) % n]);
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let len = dx.hypot(dy);
                (a, b, len, [-dy / len * sgn, dx / len * sgn])
            };
            // ── the pitch field ────────────────────────────────────────────────────────────
            // Nodes in order around the ring: both ends of every segment plus enough interior
            // samples to follow a width that changes along it. A vertex appears twice, once per
            // side, at the same arc position — which is what makes the smoothing tie the two
            // sides of a corner together.
            let at_point = |q: [f64; 2], nrm: [f64; 2]| -> (f64, f64) {
                let w = boundary.width_at(q, nrm);
                (pitch(w).max(opts.minh), w)
            };
            let mut node: Vec<(f64, f64)> = Vec::new(); // (arc position, pitch)
            let mut node_w: Vec<f64> = Vec::new();
            let mut seg_t: Vec<Vec<f64>> = Vec::with_capacity(n);
            let mut arc = 0.0;
            for i in 0..n {
                let (a, b, len, nrm) = geom(i);
                if !(len > 0.0) {
                    seg_t.push(Vec::new());
                    continue;
                }
                let at = |t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
                let d_mid = at_point(at(0.5), nrm).0;
                // One sample per few pitches, so a width that changes along the segment is
                // seen; capped, since the chain is a constraint for the mesher, not the mesh.
                let m = if d_mid.is_finite() && d_mid > 0.0 {
                    ((len / (4.0 * d_mid)).ceil() as usize).clamp(1, 8)
                } else {
                    1
                };
                let mut ts = Vec::with_capacity(m + 2);
                ts.push(0.0);
                for j in 0..m {
                    ts.push((j as f64 + 0.5) / m as f64);
                }
                ts.push(1.0);
                for &t in &ts {
                    let (d, w) = at_point(at(t), nrm);
                    node.push((arc + t * len, d));
                    node_w.push(w);
                }
                seg_t.push(ts);
                arc += len;
            }
            if node.iter().all(|&(_, d)| !d.is_finite()) {
                continue;
            }
            // `grade`-Lipschitz along the boundary, twice around so the wrap settles.
            let total = arc;
            let k = node.len();
            for _ in 0..2 {
                for j in 1..=k {
                    let (i0, i1) = ((j - 1) % k, j % k);
                    let ds = (node[i1].0 - node[i0].0).rem_euclid(total);
                    let cap = node[i0].1 + grade * ds;
                    if node[i1].1 > cap {
                        node[i1].1 = cap;
                    }
                }
                for j in (0..k).rev() {
                    let (i0, i1) = ((j + 1) % k, j);
                    let ds = (node[i1].0 - node[i0].0).rem_euclid(total);
                    let cap = node[i0].1 + grade * ds;
                    if node[i1].1 > cap {
                        node[i1].1 = cap;
                    }
                }
            }
            let mut seg_d: Vec<Vec<f64>> = Vec::with_capacity(n);
            let mut seg_w: Vec<Vec<f64>> = Vec::with_capacity(n);
            let mut at = 0usize;
            for ts in &seg_t {
                seg_d.push(node[at..at + ts.len()].iter().map(|&(_, d)| d).collect());
                seg_w.push(node_w[at..at + ts.len()].to_vec());
                at += ts.len();
            }
            // Interior samples that lie on the straight line between their neighbours buy
            // nothing but elements: along a segment the offset direction is fixed, so such a
            // sample is a point ON the chain.
            for i in 0..n {
                let (ts, ds, ws) = (&mut seg_t[i], &mut seg_d[i], &mut seg_w[i]);
                if ts.len() < 3 {
                    continue;
                }
                let tol = 0.05 * ds.iter().cloned().fold(f64::INFINITY, f64::min);
                let mut keep = vec![true; ts.len()];
                let mut lo = 0usize;
                for j in 1..ts.len() - 1 {
                    let hi = j + 1;
                    let f = (ts[j] - ts[lo]) / (ts[hi] - ts[lo]);
                    if (ds[lo] + (ds[hi] - ds[lo]) * f - ds[j]).abs() > tol {
                        lo = j;
                    } else {
                        keep[j] = false;
                    }
                }
                let mut c = 0;
                ts.retain(|_| {
                    c += 1;
                    keep[c - 1]
                });
                c = 0;
                ds.retain(|_| {
                    c += 1;
                    keep[c - 1]
                });
                c = 0;
                ws.retain(|_| {
                    c += 1;
                    keep[c - 1]
                });
            }

            // ── the rows ───────────────────────────────────────────────────────────────────
            let mut prev_scale = 0.0;
            for &scale in scales {
                let row_width = (scale - prev_scale).max(0.0);
                prev_scale = scale;
                if !(scale > 0.0) {
                    continue;
                }
                let mut pts: Vec<Option<[f64; 2]>> = Vec::new();
                let mut any = false;
                let keep = |q: [f64; 2], d: f64, w: f64| -> Option<[f64; 2]> {
                    let fits = !w.is_finite() || 2.0 * scale + row_width <= w / d;
                    (d.is_finite() && d > 0.0 && fits && boundary.distance(q) >= 0.9 * scale * d)
                        .then_some(q)
                };
                for i in 0..n {
                    let (a, b, len, nrm) = geom(i);
                    let ts = &seg_t[i];
                    if ts.is_empty() {
                        continue;
                    }
                    let (ds, ws) = (&seg_d[i], &seg_w[i]);
                    let prev = (i + n - 1) % n;
                    let d_in = *seg_d[prev].last().unwrap_or(&ds[0]);
                    let d_out = ds[0];
                    let w_v = seg_w[prev].last().unwrap_or(&ws[0]).min(ws[0]);
                    let q = offset_vertex(ring, i, d_in * scale, d_out * scale, left)
                        .and_then(|q| keep(q, d_in.min(d_out), w_v));
                    any |= q.is_some();
                    pts.push(q);
                    let inner = ts.len().saturating_sub(2);
                    for ((&t, &d), &w) in ts.iter().zip(ds).zip(ws).skip(1).take(inner) {
                        // Inside a corner's wedge the mitre IS the offset; a sample there would
                        // measure its distance against the neighbouring wall and tear the chain.
                        if t.min(1.0 - t) * len < 1.2 * d * reach {
                            continue;
                        }
                        let q = [
                            a[0] + (b[0] - a[0]) * t + nrm[0] * d * scale,
                            a[1] + (b[1] - a[1]) * t + nrm[1] * d * scale,
                        ];
                        let q = keep(q, d, w);
                        any |= q.is_some();
                        pts.push(q);
                    }
                }
                if !any {
                    break;
                }
                out.extend(runs_to_chains(&pts));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: f64, h: f64) -> Region2D {
        Region2D::new(vec![[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], 0)
    }

    /// The width a strip reads along each of its four edges. The END of a strip must read the
    /// strip's WIDTH, not its length: the facing wall is the other long side, and the short
    /// ends face each other only in the sense that a ball between them would have to cross the
    /// whole shape. Without that the band around a strip collapses into its corners.
    #[test]
    fn local_width_reads_across_the_shape_not_along_it() {
        let r = rect(600.0, 20.0);
        assert!((r.local_width([300.0, 0.0], [0.0, 1.0]) - 20.0).abs() < 1e-9);
        assert!((r.local_width([300.0, 20.0], [0.0, -1.0]) - 20.0).abs() < 1e-9);
        // the end wall faces the far end, 600 away
        assert!((r.local_width([0.0, 10.0], [1.0, 0.0]) - 600.0).abs() < 1e-9);
        // A ball tangent in the middle of a segment, not at a vertex: the notch's inner edge
        // stops it at the notch depth even though no vertex of the notch is nearest.
        let notched = Region2D::new(
            vec![
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 40.0],
                [60.0, 40.0],
                [60.0, 30.0],
                [40.0, 30.0],
                [40.0, 40.0],
                [0.0, 40.0],
            ],
            0,
        );
        assert!((notched.local_width([50.0, 0.0], [0.0, 1.0]) - 30.0).abs() < 1e-9);
    }

    /// Only the direction of `inward` counts, not its length.
    #[test]
    fn local_width_takes_any_length_of_the_direction() {
        let r = rect(600.0, 20.0);
        for scale in [0.01, 1.0, 7.5] {
            let w = r.local_width([300.0, 0.0], [0.0, scale]);
            assert!((w - 20.0).abs() < 1e-9, "scale {scale}: {w}");
        }
    }

    /// Two rows around a uniform strip: closed rings at the pitch and three times it, each
    /// point exactly its own offset away from the boundary.
    #[test]
    fn offset_chains_ring_a_uniform_strip() {
        let r = rect(600.0, 20.0);
        let opts = Mesh2DOptions::default();
        let chains = r.offset_chains(|w| w / 8.0, &[1.0, 3.0], &opts);
        assert_eq!(chains.len(), 2, "two closed rows: {chains:?}");
        for (ch, scale) in chains.iter().zip([1.0, 3.0]) {
            assert_eq!(ch[0], ch[ch.len() - 1], "closed");
            let d = scale * 20.0 / 8.0;
            let b = Boundary::new(std::slice::from_ref(&r.outer));
            for &p in &ch[..ch.len() - 1] {
                // the long sides carry the uniform pitch; the ends bulge under the grading
                if p[0] > 3.0 * d && p[0] < 600.0 - 3.0 * d {
                    assert!(
                        (b.distance(p) - d).abs() < 1e-9,
                        "{p:?} sits {} from the boundary, not {d}",
                        b.distance(p)
                    );
                }
            }
        }
    }

    /// A row is laid only where it fits between the two facing walls. The ladder 1, 3, 7 on a
    /// shape eight pitches wide has no room for the third row: it would meet the row coming in
    /// from the other side on the centre line.
    #[test]
    fn a_row_that_would_meet_its_mirror_is_not_laid() {
        let r = rect(600.0, 20.0);
        let opts = Mesh2DOptions::default();
        let two = r.offset_chains(|w| w / 8.0, &[1.0, 3.0], &opts).len();
        let three = r.offset_chains(|w| w / 8.0, &[1.0, 3.0, 7.0], &opts).len();
        assert_eq!(two, 2);
        assert_eq!(three, 2, "the third row does not fit and is dropped");
    }

    /// The pitch follows the local width, so a waist asks for a fine row — and `minh` is the
    /// floor that stops it, because a chain is a constraint the mesher must reproduce and
    /// nothing downstream can coarsen it again.
    #[test]
    fn minh_floors_the_pitch() {
        let waist = Region2D::new(
            vec![
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 9.9],
                [200.0, 0.0],
                [300.0, 0.0],
                [300.0, 20.0],
                [200.0, 20.0],
                [100.0, 10.1],
                [0.0, 20.0],
            ],
            0,
        );
        let b = Boundary::new(std::slice::from_ref(&waist.outer));
        let closest = |minh: f64| -> f64 {
            let opts = Mesh2DOptions {
                minh,
                ..Default::default()
            };
            waist
                .offset_chains(|w| w / 16.0, &[1.0, 3.0, 7.0], &opts)
                .iter()
                .flatten()
                .map(|&p| b.distance(p))
                .fold(f64::INFINITY, f64::min)
        };
        assert!(closest(0.0) < 0.05, "the waist can ask: {:e}", closest(0.0));
        assert!(
            closest(0.5) >= 0.45,
            "no point below the floor: {:e}",
            closest(0.5)
        );
    }

    /// A uniform edge needs no interior samples: its row is a straight segment between the two
    /// corner mitres, and every point in between would only cost elements.
    #[test]
    fn a_uniform_edge_keeps_only_its_corners() {
        let chains =
            rect(600.0, 20.0).offset_chains(|w| w / 8.0, &[1.0], &Mesh2DOptions::default());
        assert_eq!(chains.len(), 1);
        // four corners plus the two ends bulging under the grading, closed
        assert!(
            chains[0].len() <= 8,
            "a uniform ring should stay small: {} points",
            chains[0].len()
        );
    }
}
