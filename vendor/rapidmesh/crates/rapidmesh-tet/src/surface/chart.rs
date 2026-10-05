//! Height field charts: a curved face (or a piece of one) whose normals
//! stay within [`MAX_TILT_DEG`] of their mean is a height field over the
//! plane through
//! its centre square to that mean: it is meshed in that plane, with sizes
//! shrunk by the tilt so the lifted triangles keep theirs, and a point is
//! lifted through the face's own PLC facets (the one it falls in, by its
//! barycentric coordinates) and then onto the face's carrier.

use rapidmesh_brep::Model;
use rapidmesh_exact::vector::{add, centroid, cross, dot, perp, sub, unit};
use rapidmesh_exact::vector::{V2, V3};
use rapidmesh_geom::grid::HashGrid;
use rapidmesh_geom::Surface;

/// The largest angle between a facet's normal and the mean normal of a
/// curved face charted as a height field.
pub const MAX_TILT_DEG: f64 = 60.0;

/// A height field chart: the plane `o + x u + y v` with normal `n`, and
/// the facets projected into it.
pub struct FacetChart<'a> {
    pub o: V3,
    pub u: V3,
    pub v: V3,
    pub n: V3,
    /// The chart's normal turned to the face's front (set by the caller
    /// that knows the front; `n` until then).
    pub front: V3,
    curved: Curved<'a>,
}

struct Curved<'a> {
    surface: &'a Surface,
    /// The facets: corners in 3D and in the chart, and the cosine of the
    /// tilt of each.
    tris: Vec<([V3; 3], [V2; 3], f64)>,
    /// Facets by grid cell of the chart.
    grid: HashGrid<u32, 2>,
}

impl<'a> FacetChart<'a> {
    /// The chart of face `fi`, or none when the face is too curved for one.
    pub fn of(model: &'a Model, fi: usize) -> Option<FacetChart<'a>> {
        let (plc, brep) = (&model.plc, &model.brep);
        let face = &brep.faces[fi];
        let surface = brep.surface(face.surface);
        let corners: Vec<[V3; 3]> = face
            .facets
            .iter()
            .map(|&t| plc.triangles[t as usize].map(|i| plc.vertices[i as usize]))
            .collect();
        let normals: Vec<V3> = corners
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
        FacetChart::of_facets(surface, corners, n)
    }

    /// The height field chart of `corners` (facets of a curved carrier)
    /// over the plane square to `n`, which they must all face.
    fn of_facets(surface: &'a Surface, corners: Vec<[V3; 3]>, n: V3) -> Option<FacetChart<'a>> {
        let tilts: Vec<f64> = corners
            .iter()
            .map(|p| unit(cross(sub(p[1], p[0]), sub(p[2], p[0]))).map_or(1.0, |x| dot(x, n)))
            .collect();
        let o: V3 = centroid(corners.iter().flatten());
        let u = unit(perp(n))?;
        let v = cross(n, u);
        let to2 = |p: V3| [dot(sub(p, o), u), dot(sub(p, o), v)];
        let tris: Vec<([V3; 3], [V2; 3], f64)> = corners
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
        let mut grid = HashGrid::with_origin(lo, cell);
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
            grid.insert_box(a, b, ti as u32);
        }
        Some(FacetChart {
            o,
            u,
            v,
            n,
            front: n,
            curved: Curved {
                surface,
                tris,
                grid,
            },
        })
    }

    /// A point in the chart.
    pub fn to2(&self, p: V3) -> V2 {
        let d = sub(p, self.o);
        [dot(d, self.u), dot(d, self.v)]
    }

    /// The facet under `p` and its barycentric coordinates (the nearest
    /// facet, clamped, where `p` falls between them).
    fn locate(&self, c: &Curved<'_>, p: V2) -> (usize, [f64; 3]) {
        let bary = |q: [V2; 3]| bary(q, p);
        let home = c.grid.key(p);
        let mut best = (usize::MAX, [1.0, 0.0, 0.0], f64::NEG_INFINITY);
        for r in 0..=2 {
            for &t in c.grid.around(home, r) {
                let l = bary(c.tris[t as usize].1);
                let worst = l.iter().copied().fold(f64::INFINITY, f64::min);
                if worst > best.2 {
                    best = (t as usize, l, worst);
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
        let l = best.1.map(|x| x.max(0.0));
        let s = (l[0] + l[1] + l[2]).max(1e-300);
        (best.0, l.map(|x| x / s))
    }

    /// The face's point at `p`.
    pub fn lift(&self, p: V2) -> V3 {
        self.curved.surface.closest(self.on_facets(p)).0
    }

    /// The point of the charted facets at `p`.
    ///
    /// Past the facets (the carrier bulges past its facets) the plane of the
    /// nearest facet goes on:
    /// clamped to it, every point beside a facet would land on its edge.
    pub fn on_facets(&self, p: V2) -> V3 {
        let c = &self.curved;
        let (t, _) = self.locate(c, p);
        let (q, q2) = (c.tris[t].0, c.tris[t].1);
        let l = bary(q2, p);
        std::array::from_fn(|k| l[0] * q[0][k] + l[1] * q[1][k] + l[2] * q[2][k])
    }

    /// The factor a size is scaled by in the chart at `p`: the cosine of the
    /// tilt of the face there.
    pub fn shrink(&self, p: V2) -> f64 {
        let c = &self.curved;
        let (t, _) = self.locate(c, p);
        c.tris[t].2.clamp(0.25, 1.0)
    }
}

/// Triangles projected into a plane (`u`, `v`), none overlapping another:
/// what keeps a chart a height field while it grows.
pub(crate) struct Flat {
    u: V3,
    v: V3,
    tris: Vec<[V2; 3]>,
    grid: HashGrid<u32, 2>,
}

impl Flat {
    /// An empty plane for triangles the size of `like`.
    pub(crate) fn new(like: &[[V3; 3]], u: V3, v: V3) -> Flat {
        let mean = like
            .iter()
            .map(|p| dot(sub(p[1], p[0]), sub(p[1], p[0])).sqrt())
            .sum::<f64>()
            / like.len().max(1) as f64;
        Flat {
            u,
            v,
            tris: Vec::new(),
            grid: HashGrid::new(mean.max(1e-300)),
        }
    }

    /// Adds triangle `p` unless its projection overlaps one already in.
    pub(crate) fn try_add(&mut self, p: [V3; 3]) -> bool {
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
        if self
            .grid
            .in_box(lo, hi)
            .any(|&t| overlap(&self.tris[t as usize], &q))
        {
            return false;
        }
        let id = self.tris.len() as u32;
        self.tris.push(q);
        self.grid.insert_box(lo, hi, id);
        true
    }
}

/// Whether the insides of two triangles in the plane overlap (touching
/// along an edge or at a corner is no overlap).
fn overlap(a: &[V2; 3], b: &[V2; 3]) -> bool {
    let scale = a
        .iter()
        .chain(b)
        .flat_map(|p| p.iter().map(|x| x.abs()))
        .fold(0.0, f64::max)
        .max(1e-300);
    let eps = 1e-10 * scale * scale;
    let side = |p: V2, q: V2, r: V2| (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]);
    // Separated by the line of an edge of one: the other lies on its far
    // side (or on it).
    let apart = |a: &[V2; 3], b: &[V2; 3]| {
        let s = side(a[0], a[1], a[2]).signum();
        (0..3).any(|k| {
            let (p, q) = (a[k], a[(k + 1) % 3]);
            b.iter().all(|&r| s * side(p, q, r) <= eps)
        })
    };
    !(apart(a, b) || apart(b, a))
}

/// The barycentric coordinates of `p` in the triangle `q` (unclamped).
fn bary(q: [V2; 3], p: V2) -> [f64; 3] {
    let d = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
    if d.abs() < 1e-300 {
        return [1.0, 0.0, 0.0];
    }
    let l1 = ((p[0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (p[1] - q[0][1])) / d;
    let l2 = ((q[1][0] - q[0][0]) * (p[1] - q[0][1]) - (p[0] - q[0][0]) * (q[1][1] - q[0][1])) / d;
    [1.0 - l1 - l2, l1, l2]
}
