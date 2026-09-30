//! Charts: the plane a face is meshed in, and the way back onto the face.
//!
//! A planar face is its own chart. A curved face whose normals stay within
//! [`MAX_TILT_DEG`] of their mean is a height field over the plane through
//! its centre square to that mean: it is meshed in that plane, with sizes
//! shrunk by the tilt so the lifted triangles keep theirs, and a point is
//! lifted through the face's own PLC facets (the one it falls in, by its
//! barycentric coordinates) and then onto the face's carrier.

use rapidmesh_brep::{Model, Surface};
use rustc_hash::FxHashMap;

type P2 = [f64; 2];
type P3 = [f64; 3];

/// The largest angle between a facet's normal and the mean normal of a
/// curved face charted as a height field.
pub const MAX_TILT_DEG: f64 = 60.0;

/// A chart of one face: the plane `o + x u + y v` with normal `n`, and for a
/// curved face its facets projected into it.
pub struct Chart<'a> {
    pub o: P3,
    pub u: P3,
    pub v: P3,
    pub n: P3,
    /// The chart's normal turned to the face's front (set by the caller
    /// that knows the front; `n` until then).
    pub front: P3,
    curved: Option<Curved<'a>>,
}

struct Curved<'a> {
    surface: &'a Surface,
    /// The facets: corners in 3D and in the chart, and the cosine of the
    /// tilt of each.
    tris: Vec<([P3; 3], [P2; 3], f64)>,
    /// Facets by grid cell of the chart.
    cell: f64,
    lo: P2,
    grid: FxHashMap<(i64, i64), Vec<u32>>,
}

impl<'a> Chart<'a> {
    /// The chart of face `fi`, or none when the face is too curved for one.
    pub fn of(model: &'a Model, fi: usize) -> Option<Chart<'a>> {
        let (plc, brep) = (&model.plc, &model.brep);
        let face = &brep.faces[fi];
        let surface = brep.surface(face.surface);
        if let Surface::Plane { o, u, v, normal } = *surface {
            return Some(Chart {
                o,
                u,
                v,
                n: normal,
                front: normal,
                curved: None,
            });
        }
        let corners: Vec<[P3; 3]> = face
            .facets
            .iter()
            .map(|&t| plc.triangles[t as usize].map(|i| plc.vertices[i as usize]))
            .collect();
        let normals: Vec<P3> = corners
            .iter()
            .map(|p| cross(sub(p[1], p[0]), sub(p[2], p[0])))
            .collect();
        let sum = normals.iter().fold([0.0; 3], |s, x| add(s, *x));
        let n = unit(sum)?;
        let tilts: Vec<f64> = normals
            .iter()
            .map(|x| unit(*x).map_or(1.0, |x| dot(x, n)))
            .collect();
        let min_cos = MAX_TILT_DEG.to_radians().cos();
        if tilts.iter().any(|&c| c < min_cos) {
            return None;
        }
        // A height field: no two facets overlap in the plane.
        let u = unit(perp(n))?;
        let v = cross(n, u);
        let mut flat = Flat::new(&corners, u, v);
        if !corners.iter().all(|p| flat.try_add(*p)) {
            return None;
        }
        Chart::of_facets(surface, corners, n)
    }

    /// The height field chart of `corners` (facets of a curved carrier)
    /// over the plane square to `n`, which they must all face.
    pub fn of_facets(surface: &'a Surface, corners: Vec<[P3; 3]>, n: P3) -> Option<Chart<'a>> {
        let tilts: Vec<f64> = corners
            .iter()
            .map(|p| unit(cross(sub(p[1], p[0]), sub(p[2], p[0]))).map_or(1.0, |x| dot(x, n)))
            .collect();
        let count = (3 * corners.len()).max(1) as f64;
        let o: P3 = corners
            .iter()
            .flatten()
            .fold([0.0; 3], |s, x| add(s, *x))
            .map(|x| x / count);
        let u = unit(perp(n))?;
        let v = cross(n, u);
        let to2 = |p: P3| [dot(sub(p, o), u), dot(sub(p, o), v)];
        let tris: Vec<([P3; 3], [P2; 3], f64)> = corners
            .iter()
            .zip(&tilts)
            .map(|(p, &c)| (*p, p.map(to2), c))
            .collect();
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for (_, q, _) in &tris {
            for x in q {
                for k in 0..2 {
                    lo[k] = lo[k].min(x[k]);
                    hi[k] = hi[k].max(x[k]);
                }
            }
        }
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-300);
        let cell = (span / (tris.len() as f64).sqrt().max(1.0)).max(1e-12 * span);
        let key = |x: f64, k: usize| ((x - lo[k]) / cell).floor() as i64;
        let mut grid: FxHashMap<(i64, i64), Vec<u32>> = FxHashMap::default();
        for (ti, (_, q, _)) in tris.iter().enumerate() {
            let (a, b) = (
                [
                    q[0][0].min(q[1][0]).min(q[2][0]),
                    q[0][1].min(q[1][1]).min(q[2][1]),
                ],
                [
                    q[0][0].max(q[1][0]).max(q[2][0]),
                    q[0][1].max(q[1][1]).max(q[2][1]),
                ],
            );
            for gx in key(a[0], 0)..=key(b[0], 0) {
                for gy in key(a[1], 1)..=key(b[1], 1) {
                    grid.entry((gx, gy)).or_default().push(ti as u32);
                }
            }
        }
        Some(Chart {
            o,
            u,
            v,
            n,
            front: n,
            curved: Some(Curved {
                surface,
                tris,
                cell,
                lo,
                grid,
            }),
        })
    }

    /// Whether `q` falls in one of the charted facets (a planar face's
    /// chart covers everything).
    pub fn contains(&self, q: P2) -> bool {
        match &self.curved {
            None => true,
            Some(c) => {
                let (_, _, inside) = self.locate_full(c, q);
                inside
            }
        }
    }

    /// Whether the face is curved (not its own chart).
    pub fn is_curved(&self) -> bool {
        self.curved.is_some()
    }

    /// A point in the chart.
    pub fn to2(&self, p: P3) -> P2 {
        let d = sub(p, self.o);
        [dot(d, self.u), dot(d, self.v)]
    }

    /// The point of the chart plane at `p`.
    pub fn plane_point(&self, p: P2) -> P3 {
        std::array::from_fn(|k| self.o[k] + p[0] * self.u[k] + p[1] * self.v[k])
    }

    /// The facet under `p` and its barycentric coordinates (the nearest
    /// facet, clamped, where `p` falls between them).
    fn locate(&self, c: &Curved<'_>, p: P2) -> (usize, [f64; 3]) {
        let (t, l, _) = self.locate_full(c, p);
        (t, l)
    }

    /// [`Chart::locate`], and whether `p` falls in the facet (closed).
    fn locate_full(&self, c: &Curved<'_>, p: P2) -> (usize, [f64; 3], bool) {
        let key = |x: f64, k: usize| ((x - c.lo[k]) / c.cell).floor() as i64;
        let bary = |q: [P2; 3]| bary(q, p);
        let (gx, gy) = (key(p[0], 0), key(p[1], 1));
        let mut best = (usize::MAX, [1.0, 0.0, 0.0], f64::NEG_INFINITY);
        for r in 0..=2 {
            for x in gx - r..=gx + r {
                for y in gy - r..=gy + r {
                    let Some(ts) = c.grid.get(&(x, y)) else {
                        continue;
                    };
                    for &t in ts {
                        let l = bary(c.tris[t as usize].1);
                        let worst = l.iter().copied().fold(f64::INFINITY, f64::min);
                        if worst > best.2 {
                            best = (t as usize, l, worst);
                        }
                    }
                }
            }
            if best.2 >= 0.0 {
                break;
            }
        }
        if best.0 == usize::MAX {
            // Far outside every facet: the nearest by centre.
            let t = (0..c.tris.len())
                .min_by(|&a, &b| {
                    let d = |t: usize| {
                        let q = c.tris[t].1;
                        let m = [
                            (q[0][0] + q[1][0] + q[2][0]) / 3.0,
                            (q[0][1] + q[1][1] + q[2][1]) / 3.0,
                        ];
                        (m[0] - p[0]).powi(2) + (m[1] - p[1]).powi(2)
                    };
                    d(a).total_cmp(&d(b))
                })
                .unwrap_or(0);
            best = (t, bary(c.tris[t].1), 0.0);
        }
        let inside = best.2 >= -1e-9;
        let l = best.1.map(|x| x.max(0.0));
        let s = (l[0] + l[1] + l[2]).max(1e-300);
        (best.0, l.map(|x| x / s), inside)
    }

    /// The face's point at `p`.
    pub fn lift(&self, p: P2) -> P3 {
        let Some(c) = &self.curved else {
            return self.plane_point(p);
        };
        c.surface.closest(self.on_facets(p)).0
    }

    /// The point of the charted facets at `p` (the chart plane's for a
    /// planar face).
    ///
    /// Past the facets (the carrier bulges past its facets, and the cuts of
    /// an atlas are on the carrier) the plane of the nearest facet goes on:
    /// clamped to it, every point beside a facet would land on its edge.
    pub fn on_facets(&self, p: P2) -> P3 {
        let Some(c) = &self.curved else {
            return self.plane_point(p);
        };
        let (t, _) = self.locate(c, p);
        let (q, q2) = (c.tris[t].0, c.tris[t].1);
        let l = bary(q2, p);
        std::array::from_fn(|k| l[0] * q[0][k] + l[1] * q[1][k] + l[2] * q[2][k])
    }

    /// The factor a size is scaled by in the chart at `p`: the cosine of the
    /// tilt of the face there (1 for a planar face).
    pub fn shrink(&self, p: P2) -> f64 {
        let Some(c) = &self.curved else {
            return 1.0;
        };
        let (t, _) = self.locate(c, p);
        c.tris[t].2.clamp(0.25, 1.0)
    }
}

/// Triangles projected into a plane (`u`, `v`), none overlapping another:
/// what keeps a chart a height field while it grows.
pub(crate) struct Flat {
    u: P3,
    v: P3,
    cell: f64,
    tris: Vec<[P2; 3]>,
    grid: FxHashMap<(i64, i64), Vec<u32>>,
}

impl Flat {
    /// An empty plane for triangles the size of `like`.
    pub(crate) fn new(like: &[[P3; 3]], u: P3, v: P3) -> Flat {
        let mean = like
            .iter()
            .map(|p| dot(sub(p[1], p[0]), sub(p[1], p[0])).sqrt())
            .sum::<f64>()
            / like.len().max(1) as f64;
        Flat {
            u,
            v,
            cell: mean.max(1e-300),
            tris: Vec::new(),
            grid: FxHashMap::default(),
        }
    }

    /// Adds triangle `p` unless its projection overlaps one already in.
    pub(crate) fn try_add(&mut self, p: [P3; 3]) -> bool {
        let q = p.map(|x| [dot(x, self.u), dot(x, self.v)]);
        let (lo, hi) = (
            [
                q[0][0].min(q[1][0]).min(q[2][0]),
                q[0][1].min(q[1][1]).min(q[2][1]),
            ],
            [
                q[0][0].max(q[1][0]).max(q[2][0]),
                q[0][1].max(q[1][1]).max(q[2][1]),
            ],
        );
        let key = |x: f64| (x / self.cell).floor() as i64;
        let cells: Vec<(i64, i64)> = (key(lo[0])..=key(hi[0]))
            .flat_map(|x| (key(lo[1])..=key(hi[1])).map(move |y| (x, y)))
            .collect();
        for c in &cells {
            if let Some(ts) = self.grid.get(c) {
                if ts.iter().any(|&t| overlap(&self.tris[t as usize], &q)) {
                    return false;
                }
            }
        }
        let id = self.tris.len() as u32;
        self.tris.push(q);
        for c in cells {
            self.grid.entry(c).or_default().push(id);
        }
        true
    }
}

/// Whether the insides of two triangles in the plane overlap (touching
/// along an edge or at a corner is no overlap).
fn overlap(a: &[P2; 3], b: &[P2; 3]) -> bool {
    let scale = a
        .iter()
        .chain(b)
        .flat_map(|p| p.iter().map(|x| x.abs()))
        .fold(0.0, f64::max)
        .max(1e-300);
    let eps = 1e-10 * scale * scale;
    let side = |p: P2, q: P2, r: P2| (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]);
    // Separated by the line of an edge of one: the other lies on its far
    // side (or on it).
    let apart = |a: &[P2; 3], b: &[P2; 3]| {
        let s = side(a[0], a[1], a[2]).signum();
        (0..3).any(|k| {
            let (p, q) = (a[k], a[(k + 1) % 3]);
            b.iter().all(|&r| s * side(p, q, r) <= eps)
        })
    };
    !(apart(a, b) || apart(b, a))
}

/// The barycentric coordinates of `p` in the triangle `q` (unclamped).
fn bary(q: [P2; 3], p: P2) -> [f64; 3] {
    let d = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
    if d.abs() < 1e-300 {
        return [1.0, 0.0, 0.0];
    }
    let l1 = ((p[0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (p[1] - q[0][1])) / d;
    let l2 = ((q[1][0] - q[0][0]) * (p[1] - q[0][1]) - (p[0] - q[0][0]) * (q[1][1] - q[0][1])) / d;
    [1.0 - l1 - l2, l1, l2]
}

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(a: P3) -> Option<P3> {
    let l = dot(a, a).sqrt();
    (l > 0.0 && l.is_finite()).then(|| a.map(|x| x / l))
}

/// A vector square to `n`.
fn perp(n: P3) -> P3 {
    let k = (0..3)
        .min_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
        .unwrap_or(0);
    let mut e = [0.0; 3];
    e[k] = 1.0;
    cross(n, e)
}
