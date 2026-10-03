//! What the volume mesher asks of the geometry, as two traits.
//!
//! [`DomainOracle`] answers the three questions a restricted-Delaunay mesher
//! needs: which region contains a point, where a segment crosses the surface
//! (and which patch it crosses), and which corners and curves are features to
//! protect. [`SizeField`] gives the target edge length at a point. The mesher
//! sees nothing else of the geometry, so it can be tested on the analytic
//! domains in [`domains`] without any CSG, and a B-rep, an implicit surface
//! or a CAD kernel plug in by implementing the trait.
//!
//! Consistency contract: along any segment, `region` changes exactly at the
//! crossings `crossings` reports with a patch whose two regions differ (a
//! sheet, whose regions are equal, is reported but changes nothing). The
//! mesher's conformity rests on it: a mesh facet between two differently
//! labelled cells is then always a restricted facet of some interface patch.

use crate::curve::Curve;
use std::sync::Arc;

fn dist(a: P3, b: P3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A point in space.
pub type P3 = [f64; 3];

/// A surface patch: one face of the domain's boundary or an embedded sheet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Patch {
    /// The regions on the two sides; equal for a sheet inside one region.
    /// Region 0 is the outside.
    pub regions: [u32; 2],
    /// Caller tag (ports, boundary conditions), carried to the mesh faces.
    pub tag: u32,
}

impl Patch {
    /// True for an embedded sheet (the same region on both sides).
    pub fn is_sheet(&self) -> bool {
        self.regions[0] == self.regions[1]
    }
}

/// A crossing of a segment with a surface patch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crossing {
    /// Parameter along the segment, in `(0, 1)`.
    pub t: f64,
    /// The crossing point, on the patch.
    pub point: P3,
    /// The patch crossed.
    pub patch: u32,
}

/// A feature curve (a 1-dimensional feature: an edge where patches meet, or
/// the free boundary of a sheet), protected by the mesher.
pub struct FeatureCurve {
    /// The curve, parametrized by arc length.
    pub curve: Arc<dyn Curve>,
    /// Corner indices at the curve's start and end (`None` for a closed curve).
    pub ends: [Option<u32>; 2],
    /// The patches meeting along the curve.
    pub patches: Vec<u32>,
    /// Upper bound on the sample spacing along this curve, on top of the
    /// size field (`INFINITY` for none).
    pub max_size: f64,
    /// Chord deviation bound for this curve, overriding
    /// `Params::curve_deflection`.
    pub deflection: Option<f64>,
}

impl FeatureCurve {
    /// A curve sampled by the size field and the default deflection.
    pub fn new(curve: Arc<dyn Curve>, ends: [Option<u32>; 2], patches: Vec<u32>) -> FeatureCurve {
        FeatureCurve {
            curve,
            ends,
            patches,
            max_size: f64::INFINITY,
            deflection: None,
        }
    }
}

/// The geometry the volume mesher meshes.
pub trait DomainOracle: Sync {
    /// A box containing the whole domain.
    fn bbox(&self) -> (P3, P3);
    /// The region containing `p` (0 = outside every region).
    fn region(&self, p: P3) -> u32;
    /// The crossings of the open segment `(a, b)` with the surface patches,
    /// appended to `out` in increasing `t`. Tangential touches are not
    /// crossings.
    fn crossings(&self, a: P3, b: P3, out: &mut Vec<Crossing>);
    /// A radius around `p` that no patch enters (0 when unknown): a segment
    /// whose ends' clearances together reach past its length crosses none.
    fn clearance(&self, _p: P3) -> f64 {
        0.0
    }
    /// [`DomainOracle::nearest_crossing`] for a caller that screened the
    /// segment by [`DomainOracle::clearance`] already.
    fn nearest_crossing_screened(&self, a: P3, b: P3, focus: P3) -> Option<Crossing> {
        self.nearest_crossing(a, b, focus)
    }
    /// The crossing of the open segment `(a, b)` nearest to `focus` (ties
    /// to the smaller `t`, then the smaller patch), if any: the surface
    /// center of a restricted facet whose circumcenter is `focus`.
    fn nearest_crossing(&self, a: P3, b: P3, focus: P3) -> Option<Crossing> {
        let mut cr = Vec::new();
        self.crossings(a, b, &mut cr);
        cr.into_iter().min_by(|x, y| {
            dist(x.point, focus)
                .total_cmp(&dist(y.point, focus))
                .then(x.t.total_cmp(&y.t))
                .then(x.patch.cmp(&y.patch))
        })
    }
    /// The surface patches; patch ids index this slice.
    fn patches(&self) -> &[Patch];
    /// The corners (0-dimensional features).
    fn corners(&self) -> &[P3];
    /// The feature curves (1-dimensional features).
    fn curves(&self) -> &[FeatureCurve];
    /// Well spread points on a patch, the first samples the refinement grows
    /// from. A patch without curves (a sphere, a torus) needs them; bounded
    /// patches may return a few too (their curve samples can be coplanar).
    fn patch_seeds(&self, patch: u32) -> Vec<P3>;
    /// Patches through a corner beyond those of the curves ending there (a
    /// cone apex inside its patch has no curve).
    fn corner_patches(&self, _corner: u32) -> Vec<u32> {
        Vec::new()
    }
    /// True if a patch other than those in `except` passes within `r` of
    /// `p`: protection shrinks a ball that would. Domains whose features
    /// stay far from every other patch may keep the default.
    fn patch_within(&self, _p: P3, _r: f64, _except: &[u32]) -> bool {
        false
    }
}

/// The target edge length of the mesh at a point.
pub trait SizeField: Sync {
    /// Target edge length at `p` (positive, finite).
    fn size(&self, p: P3) -> f64;
}

/// A constant size field.
pub struct Uniform(pub f64);

impl SizeField for Uniform {
    fn size(&self, _: P3) -> f64 {
        self.0
    }
}

/// Analytic domains for tests and experiments: closed-form regions,
/// crossings and features, consistent by construction.
pub mod domains {
    use super::*;
    use crate::curve::PolylineCurve;

    fn sub(a: P3, b: P3) -> P3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn dot(a: P3, b: P3) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn lerp(a: P3, b: P3, t: f64) -> P3 {
        [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
        ]
    }

    /// Parameters in `(0, 1)` where segment `a -> b` crosses the sphere
    /// `|x - c| = r` transversally, ascending.
    fn sphere_ts(a: P3, b: P3, c: P3, r: f64) -> Vec<f64> {
        let d = sub(b, a);
        let m = sub(a, c);
        let aa = dot(d, d);
        if aa == 0.0 {
            return Vec::new();
        }
        let bb = dot(m, d);
        let cc = dot(m, m) - r * r;
        let disc = bb * bb - aa * cc;
        if disc <= 0.0 {
            return Vec::new();
        }
        let s = disc.sqrt();
        let mut out: Vec<f64> = [(-bb - s) / aa, (-bb + s) / aa]
            .into_iter()
            .filter(|&t| t > 0.0 && t < 1.0)
            .collect();
        out.dedup();
        out
    }

    /// Spread points on a sphere (Fibonacci lattice).
    fn sphere_points(c: P3, r: f64, n: usize) -> Vec<P3> {
        let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        (0..n)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let rho = (1.0 - z * z).sqrt();
                let th = golden * i as f64;
                [
                    c[0] + r * rho * th.cos(),
                    c[1] + r * rho * th.sin(),
                    c[2] + r * z,
                ]
            })
            .collect()
    }

    /// A circle as a closed polyline curve (fine enough to be exact for
    /// sampling purposes).
    fn circle_curve(c: P3, r: f64, n: usize) -> Arc<dyn Curve> {
        let pts: Vec<P3> = (0..=n)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                [c[0] + r * a.cos(), c[1] + r * a.sin(), c[2]]
            })
            .collect();
        Arc::new(PolylineCurve::new(&pts).expect("circle samples"))
    }

    /// Concentric balls: ball `i` (radius `radii[i]`, ascending) minus ball
    /// `i - 1` is region `i + 1`. One ball is the simplest closed domain;
    /// two give an interface patch between two regions.
    pub struct Balls {
        pub center: P3,
        pub radii: Vec<f64>,
        patches: Vec<Patch>,
    }

    impl Balls {
        pub fn new(center: P3, radii: &[f64]) -> Balls {
            let n = radii.len() as u32;
            // Patch i is sphere i: region i + 1 inside, region i + 2 outside
            // (0 beyond the last).
            let patches = (0..n)
                .map(|i| Patch {
                    regions: [i + 1, if i + 1 == n { 0 } else { i + 2 }],
                    tag: i,
                })
                .collect();
            Balls {
                center,
                radii: radii.to_vec(),
                patches,
            }
        }
    }

    impl DomainOracle for Balls {
        fn bbox(&self) -> (P3, P3) {
            let r = self.radii.last().copied().unwrap_or(0.0);
            let c = self.center;
            (
                [c[0] - r, c[1] - r, c[2] - r],
                [c[0] + r, c[1] + r, c[2] + r],
            )
        }
        fn region(&self, p: P3) -> u32 {
            let d = sub(p, self.center);
            let r = dot(d, d).sqrt();
            self.radii
                .iter()
                .position(|&ri| r < ri)
                .map_or(0, |i| i as u32 + 1)
        }
        fn crossings(&self, a: P3, b: P3, out: &mut Vec<Crossing>) {
            let start = out.len();
            for (i, &r) in self.radii.iter().enumerate() {
                for t in sphere_ts(a, b, self.center, r) {
                    out.push(Crossing {
                        t,
                        point: lerp(a, b, t),
                        patch: i as u32,
                    });
                }
            }
            out[start..].sort_by(|x, y| x.t.total_cmp(&y.t));
        }
        fn patches(&self) -> &[Patch] {
            &self.patches
        }
        fn corners(&self) -> &[P3] {
            &[]
        }
        fn curves(&self) -> &[FeatureCurve] {
            &[]
        }
        fn patch_seeds(&self, patch: u32) -> Vec<P3> {
            sphere_points(self.center, self.radii[patch as usize], 32)
        }
        fn patch_within(&self, p: P3, r: f64, except: &[u32]) -> bool {
            let d = sub(p, self.center);
            let l = dot(d, d).sqrt();
            self.radii
                .iter()
                .enumerate()
                .any(|(i, &ri)| !except.contains(&(i as u32)) && (l - ri).abs() < r)
        }
    }

    /// An axis-aligned box: six planar patches, eight corners, twelve
    /// straight feature curves.
    pub struct Cube {
        pub lo: P3,
        pub hi: P3,
        patches: Vec<Patch>,
        corners: Vec<P3>,
        curves: Vec<FeatureCurve>,
    }

    impl Cube {
        pub fn new(lo: P3, hi: P3) -> Cube {
            // Patch 2k is the face x_k = lo, 2k + 1 is x_k = hi.
            let patches = (0..6)
                .map(|i| Patch {
                    regions: [1, 0],
                    tag: i,
                })
                .collect();
            let corner = |m: usize| -> P3 {
                std::array::from_fn(|k| if m & (1 << k) != 0 { hi[k] } else { lo[k] })
            };
            let corners: Vec<P3> = (0..8).map(corner).collect();
            let mut curves = Vec::new();
            for m in 0..8usize {
                for k in 0..3 {
                    if m & (1 << k) == 0 {
                        let n = m | (1 << k);
                        let (a, b) = (corner(m), corner(n));
                        // The two faces meeting along the edge: the other two
                        // axes at this edge's side.
                        let patches: Vec<u32> = (0..3)
                            .filter(|&j| j != k)
                            .map(|j| (2 * j + usize::from(m & (1 << j) != 0)) as u32)
                            .collect();
                        curves.push(FeatureCurve::new(
                            Arc::new(PolylineCurve::new(&[a, b]).expect("edge")),
                            [Some(m as u32), Some(n as u32)],
                            patches,
                        ));
                    }
                }
            }
            Cube {
                lo,
                hi,
                patches,
                corners,
                curves,
            }
        }
    }

    impl DomainOracle for Cube {
        fn bbox(&self) -> (P3, P3) {
            (self.lo, self.hi)
        }
        fn region(&self, p: P3) -> u32 {
            u32::from((0..3).all(|k| p[k] > self.lo[k] && p[k] < self.hi[k]))
        }
        fn crossings(&self, a: P3, b: P3, out: &mut Vec<Crossing>) {
            let start = out.len();
            for k in 0..3 {
                for (side, plane) in [(0usize, self.lo[k]), (1, self.hi[k])] {
                    let (da, db) = (a[k] - plane, b[k] - plane);
                    if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                        let t = da / (da - db);
                        let mut x = lerp(a, b, t);
                        x[k] = plane;
                        let inside = (0..3)
                            .filter(|&j| j != k)
                            .all(|j| x[j] >= self.lo[j] && x[j] <= self.hi[j]);
                        if inside {
                            out.push(Crossing {
                                t,
                                point: x,
                                patch: (2 * k + side) as u32,
                            });
                        }
                    }
                }
            }
            out[start..].sort_by(|x, y| x.t.total_cmp(&y.t));
        }
        fn patches(&self) -> &[Patch] {
            &self.patches
        }
        fn corners(&self) -> &[P3] {
            &self.corners
        }
        fn curves(&self) -> &[FeatureCurve] {
            &self.curves
        }
        fn patch_seeds(&self, _: u32) -> Vec<P3> {
            Vec::new()
        }
        fn patch_within(&self, p: P3, r: f64, except: &[u32]) -> bool {
            (0..3).any(|k| {
                [(0usize, self.lo[k]), (1, self.hi[k])]
                    .iter()
                    .any(|&(side, plane)| {
                        if except.contains(&((2 * k + side) as u32)) {
                            return false;
                        }
                        // Distance to the face rectangle.
                        let d2: f64 = (0..3)
                            .map(|j| {
                                let e = if j == k {
                                    p[j] - plane
                                } else {
                                    p[j] - p[j].clamp(self.lo[j], self.hi[j])
                                };
                                e * e
                            })
                            .sum();
                        d2 < r * r
                    })
            })
        }
    }

    /// A ball (region 1) with a flat disc sheet through its center: the
    /// disc `z = center.z, |xy| < disc_r` (disc_r < ball_r) has region 1 on
    /// both sides, and its rim circle is a feature curve.
    pub struct BallWithSheet {
        pub center: P3,
        pub ball_r: f64,
        pub disc_r: f64,
        patches: Vec<Patch>,
        curves: Vec<FeatureCurve>,
    }

    impl BallWithSheet {
        pub fn new(center: P3, ball_r: f64, disc_r: f64) -> BallWithSheet {
            BallWithSheet {
                center,
                ball_r,
                disc_r,
                patches: vec![
                    Patch {
                        regions: [1, 0],
                        tag: 0,
                    },
                    Patch {
                        regions: [1, 1],
                        tag: 7,
                    },
                ],
                curves: vec![FeatureCurve::new(
                    circle_curve(center, disc_r, 512),
                    [None, None],
                    vec![1],
                )],
            }
        }
    }

    impl DomainOracle for BallWithSheet {
        fn bbox(&self) -> (P3, P3) {
            let (c, r) = (self.center, self.ball_r);
            (
                [c[0] - r, c[1] - r, c[2] - r],
                [c[0] + r, c[1] + r, c[2] + r],
            )
        }
        fn region(&self, p: P3) -> u32 {
            let d = sub(p, self.center);
            u32::from(dot(d, d) < self.ball_r * self.ball_r)
        }
        fn crossings(&self, a: P3, b: P3, out: &mut Vec<Crossing>) {
            let start = out.len();
            for t in sphere_ts(a, b, self.center, self.ball_r) {
                out.push(Crossing {
                    t,
                    point: lerp(a, b, t),
                    patch: 0,
                });
            }
            let z = self.center[2];
            let (da, db) = (a[2] - z, b[2] - z);
            if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                let t = da / (da - db);
                let mut x = lerp(a, b, t);
                x[2] = z;
                let (dx, dy) = (x[0] - self.center[0], x[1] - self.center[1]);
                if dx * dx + dy * dy < self.disc_r * self.disc_r {
                    out.push(Crossing {
                        t,
                        point: x,
                        patch: 1,
                    });
                }
            }
            out[start..].sort_by(|x, y| x.t.total_cmp(&y.t));
        }
        fn patches(&self) -> &[Patch] {
            &self.patches
        }
        fn corners(&self) -> &[P3] {
            &[]
        }
        fn curves(&self) -> &[FeatureCurve] {
            &self.curves
        }
        fn patch_seeds(&self, patch: u32) -> Vec<P3> {
            match patch {
                0 => sphere_points(self.center, self.ball_r, 32),
                _ => Vec::new(),
            }
        }
    }
}

/// A rigid motion of another domain: `x = R x_inner + t` with `R` a
/// rotation. Every query maps back into the inner domain, so a mesher's
/// output on a rotated domain can be compared with the unrotated one.
pub struct Moved<D: DomainOracle> {
    pub inner: D,
    /// Row-major rotation matrix.
    pub rot: [[f64; 3]; 3],
    pub shift: P3,
    corners: Vec<P3>,
    curves: Vec<FeatureCurve>,
}

struct MovedCurve {
    curve: Arc<dyn Curve>,
    rot: [[f64; 3]; 3],
    shift: P3,
}

impl Curve for MovedCurve {
    fn length(&self) -> f64 {
        self.curve.length()
    }
    fn point_at(&self, s: f64) -> P3 {
        apply(&self.rot, self.shift, self.curve.point_at(s))
    }
    fn radius_at(&self, s: f64) -> f64 {
        self.curve.radius_at(s)
    }
    fn ders_at(&self, s: f64) -> [P3; 3] {
        let [p, t, k] = self.curve.ders_at(s);
        let turn = |v: P3| apply(&self.rot, [0.0; 3], v);
        [apply(&self.rot, self.shift, p), turn(t), turn(k)]
    }
}

fn apply(r: &[[f64; 3]; 3], t: P3, q: P3) -> P3 {
    std::array::from_fn(|i| r[i][0] * q[0] + r[i][1] * q[1] + r[i][2] * q[2] + t[i])
}

fn unapply(r: &[[f64; 3]; 3], t: P3, x: P3) -> P3 {
    let d = [x[0] - t[0], x[1] - t[1], x[2] - t[2]];
    std::array::from_fn(|j| r[0][j] * d[0] + r[1][j] * d[1] + r[2][j] * d[2])
}

impl<D: DomainOracle> Moved<D> {
    /// The domain rotated by `angle` radians about `axis` (normalized here),
    /// then shifted.
    pub fn new(inner: D, axis: P3, angle: f64, shift: P3) -> Moved<D> {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
        let (c, s) = (angle.cos(), angle.sin());
        let t = 1.0 - c;
        let rot = [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ];
        let corners = inner
            .corners()
            .iter()
            .map(|&q| apply(&rot, shift, q))
            .collect();
        let curves = inner
            .curves()
            .iter()
            .map(|fc| FeatureCurve {
                curve: Arc::new(MovedCurve {
                    curve: fc.curve.clone(),
                    rot,
                    shift,
                }),
                ends: fc.ends,
                patches: fc.patches.clone(),
                max_size: fc.max_size,
                deflection: fc.deflection,
            })
            .collect();
        Moved {
            inner,
            rot,
            shift,
            corners,
            curves,
        }
    }
}

impl<D: DomainOracle> DomainOracle for Moved<D> {
    fn bbox(&self) -> (P3, P3) {
        let (lo, hi) = self.inner.bbox();
        let (mut a, mut b) = ([f64::MAX; 3], [f64::MIN; 3]);
        for m in 0..8 {
            let q: P3 = std::array::from_fn(|k| if m & (1 << k) != 0 { hi[k] } else { lo[k] });
            let x = apply(&self.rot, self.shift, q);
            for k in 0..3 {
                a[k] = a[k].min(x[k]);
                b[k] = b[k].max(x[k]);
            }
        }
        (a, b)
    }
    fn region(&self, p: P3) -> u32 {
        self.inner.region(unapply(&self.rot, self.shift, p))
    }
    fn crossings(&self, a: P3, b: P3, out: &mut Vec<Crossing>) {
        let start = out.len();
        self.inner.crossings(
            unapply(&self.rot, self.shift, a),
            unapply(&self.rot, self.shift, b),
            out,
        );
        for c in &mut out[start..] {
            c.point = apply(&self.rot, self.shift, c.point);
        }
    }
    fn patches(&self) -> &[Patch] {
        self.inner.patches()
    }
    fn corners(&self) -> &[P3] {
        &self.corners
    }
    fn curves(&self) -> &[FeatureCurve] {
        &self.curves
    }
    fn patch_seeds(&self, patch: u32) -> Vec<P3> {
        self.inner
            .patch_seeds(patch)
            .into_iter()
            .map(|q| apply(&self.rot, self.shift, q))
            .collect()
    }
    fn corner_patches(&self, corner: u32) -> Vec<u32> {
        self.inner.corner_patches(corner)
    }
    fn patch_within(&self, p: P3, r: f64, except: &[u32]) -> bool {
        self.inner
            .patch_within(unapply(&self.rot, self.shift, p), r, except)
    }
}

#[cfg(test)]
mod tests {
    use super::domains::*;
    use super::*;

    /// The consistency contract on random segments: walking a segment, the
    /// region flips exactly at the interface crossings.
    fn check_consistent(d: &dyn DomainOracle, seed: u64) {
        let (lo, hi) = d.bbox();
        let mut s = seed;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let pt = |rnd: &mut dyn FnMut() -> f64| -> P3 {
            std::array::from_fn(|k| lo[k] - 0.2 * (hi[k] - lo[k]) + 1.4 * (hi[k] - lo[k]) * rnd())
        };
        let mut cr = Vec::new();
        for _ in 0..2000 {
            let (a, b) = (pt(&mut rnd), pt(&mut rnd));
            cr.clear();
            d.crossings(a, b, &mut cr);
            let mut region = d.region(a);
            for c in &cr {
                assert!(c.t > 0.0 && c.t < 1.0);
                let p = d.patches()[c.patch as usize];
                if !p.is_sheet() {
                    assert!(
                        p.regions.contains(&region),
                        "crossing patch {} {:?} from region {region}",
                        c.patch,
                        p.regions
                    );
                    region = if p.regions[0] == region {
                        p.regions[1]
                    } else {
                        p.regions[0]
                    };
                }
            }
            assert_eq!(region, d.region(b), "segment {a:?} -> {b:?}");
        }
    }

    #[test]
    fn analytic_domains_keep_the_contract() {
        check_consistent(&Balls::new([0.1, -0.2, 0.3], &[1.0]), 1);
        check_consistent(&Balls::new([0.0; 3], &[0.5, 1.0, 1.3]), 2);
        check_consistent(&Cube::new([0.0; 3], [1.0, 2.0, 0.5]), 3);
        check_consistent(&BallWithSheet::new([0.0; 3], 1.0, 0.6), 4);
    }

    #[test]
    fn cube_features_are_complete() {
        let c = Cube::new([0.0; 3], [1.0; 3]);
        assert_eq!(c.corners().len(), 8);
        assert_eq!(c.curves().len(), 12);
        for fc in c.curves() {
            assert_eq!(fc.patches.len(), 2);
            assert!((fc.curve.length() - 1.0).abs() < 1e-12);
        }
    }
}
