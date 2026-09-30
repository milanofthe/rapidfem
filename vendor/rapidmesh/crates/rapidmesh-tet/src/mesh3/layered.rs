//! The layered path: a domain made of vertical extrusions (every facet
//! horizontal or vertical, as boxes, prisms and vertical cylinders give) is
//! meshed as one 2D mesh of its plan, extruded through its levels.
//!
//! Restricted Delaunay refinement has to sample a layer at its thickness,
//! which for the metal of a chip or board stack (a micron under a size of
//! several) multiplies the mesh by the aspect ratio. Here a layer is one or a
//! few slabs of prisms whatever its thickness: flat tets across thin layers,
//! as a surface mesh filled by constrained Delaunay would give, with regions
//! exact by construction.
//!
//! The plan is the footprint of every vertical facet, split where footprints
//! cross or come close (within a model tolerance: layouts carry nanometre
//! steps under a size of microns); each piece knows the heights of the walls
//! over it. Its 2D mesh conforms to every piece, and each slab between two
//! levels takes the region of each connected set of triangles between the
//! walls standing in it. A plan point keeps only the levels where a triangle
//! at it changes region, plus enough that no column grows taller than the
//! size, so the air beside a thin metal layer is not cut by it. Each column
//! over a plan triangle is swept into tets corner by corner in an order that
//! two corners decide alone, so neighbouring columns agree on their shared
//! side. The result is an ordinary [`Complex`].

use super::brep::BrepOracle;
use super::oracle::{DomainOracle, P3};
use super::{Complex, Face, VertexKind};
use crate::constants::{LAYERED_SNAP, LAYERED_THIN};
use crate::domain::DomainTree;
use geometry_predicates::{orient2d, orient3d};
use rapidmesh_geom::TaggedPlc;
use rustc_hash::{FxHashMap, FxHashSet};

type P2 = [f64; 2];

/// The heights a wall spans over a plan piece, and whether it is a sheet.
type Span = ([f64; 2], bool);

/// The footprint of a vertical facet and the heights it spans; a sheet's
/// stands in one region.
struct Wall {
    a: P2,
    b: P2,
    z: [f64; 2],
    sheet: bool,
}

/// A horizontal sheet facet: its plan triangle and its height.
struct Flat {
    t: [P2; 3],
    z: f64,
}

/// Levels (heights of the horizontal facets and of the ends of vertical
/// sheets), walls and horizontal sheet facets of a layered PLC, or why it
/// is not one (a facet neither horizontal nor vertical). The outline of each
/// horizontal sheet is a wall of no height, so the plan conforms to it.
/// Horizontal and vertical hold up to `tol`, and levels closer than `tol`
/// are one: stacks built by summing thicknesses carry rounding noise.
fn layers(plc: &TaggedPlc, tol: f64) -> Result<(Vec<f64>, Vec<Wall>, Vec<Flat>), &'static str> {
    let mut levels: Vec<f64> = Vec::new();
    let mut walls: Vec<Wall> = Vec::new();
    let mut flats: Vec<Flat> = Vec::new();
    // Edges of the horizontal sheet facets, per sheet face tag.
    let mut flat_edges: FxHashMap<(u32, u32, u32), usize> = FxHashMap::default();
    for (fi, (t, rt)) in plc.triangles.iter().zip(&plc.region_tags).enumerate() {
        let sheet = rt[0] == rt[1];
        let p = t.map(|i| plc.vertices[i as usize]);
        let zs = p.map(|x| x[2]);
        let zlo = zs.iter().copied().fold(f64::INFINITY, f64::min);
        let zhi = zs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let q = p.map(|x| [x[0], x[1]]);
        if zhi - zlo <= tol {
            levels.push(zlo);
            if sheet {
                flats.push(Flat { t: q, z: zlo });
                let tag = plc.face_tags[fi].0;
                for e in 0..3 {
                    let (a, b) = (t[e], t[(e + 1) % 3]);
                    *flat_edges.entry((a.min(b), a.max(b), tag)).or_default() += 1;
                }
            }
            continue;
        }
        // The two footprint points farthest apart span the segment; the
        // third lies on it.
        let d = |i: usize, j: usize| (q[i][0] - q[j][0]).powi(2) + (q[i][1] - q[j][1]).powi(2);
        let (i, j) = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .max_by(|x, y| d(x.0, x.1).total_cmp(&d(y.0, y.1)))
            .expect("three pairs");
        let span = d(i, j).sqrt();
        if !(span > tol) {
            // A vertical sliver over a point: no wall in the plan.
            continue;
        }
        if orient2d(q[0], q[1], q[2]).abs() > tol * span {
            return Err("a facet neither horizontal nor vertical");
        }
        if sheet {
            levels.extend([zlo, zhi]);
        }
        walls.push(Wall {
            a: q[i],
            b: q[j],
            z: [zlo, zhi],
            sheet,
        });
    }
    let mut outline: Vec<&(u32, u32, u32)> = flat_edges
        .iter()
        .filter(|(_, &n)| n == 1)
        .map(|(k, _)| k)
        .collect();
    outline.sort_unstable();
    for &(a, b, _) in outline {
        let (pa, pb) = (plc.vertices[a as usize], plc.vertices[b as usize]);
        walls.push(Wall {
            a: [pa[0], pa[1]],
            b: [pb[0], pb[1]],
            z: [pa[2], pa[2]],
            sheet: true,
        });
    }
    if walls.is_empty() || levels.is_empty() {
        return Err("no walls or no levels");
    }
    levels.sort_by(f64::total_cmp);
    levels.dedup_by(|b, a| *b - *a <= tol);
    Ok((levels, walls, flats))
}

/// Points welded at a tolerance, through a hash grid of that cell size.
struct Weld {
    tol: f64,
    pts: Vec<P2>,
    grid: FxHashMap<(i64, i64), Vec<usize>>,
}

impl Weld {
    fn id(&mut self, p: P2) -> usize {
        let key = |x: f64| (x / self.tol).floor() as i64;
        let (kx, ky) = (key(p[0]), key(p[1]));
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(ids) = self.grid.get(&(kx + dx, ky + dy)) {
                    for &i in ids {
                        let q = self.pts[i];
                        if (q[0] - p[0]).abs() <= self.tol && (q[1] - p[1]).abs() <= self.tol {
                            return i;
                        }
                    }
                }
            }
        }
        self.pts.push(p);
        self.grid
            .entry((kx, ky))
            .or_default()
            .push(self.pts.len() - 1);
        self.pts.len() - 1
    }
}

/// The plan: every footprint split where footprints cross or come within
/// `tol` of each other, as pieces between points welded at `tol`, each with
/// the spans of the walls over it.
fn plan(walls: &[Wall], tol: f64) -> (Vec<P2>, FxHashMap<(usize, usize), Vec<Span>>) {
    // One segment per distinct footprint, with every wall's heights.
    let mut foot: FxHashMap<[u64; 4], usize> = FxHashMap::default();
    let mut segs: Vec<(P2, P2, Vec<Span>)> = Vec::new();
    for w in walls {
        let (a, b) = if (w.a[0], w.a[1]) <= (w.b[0], w.b[1]) {
            (w.a, w.b)
        } else {
            (w.b, w.a)
        };
        let key = [
            a[0].to_bits(),
            a[1].to_bits(),
            b[0].to_bits(),
            b[1].to_bits(),
        ];
        let i = *foot.entry(key).or_insert_with(|| {
            segs.push((a, b, Vec::new()));
            segs.len() - 1
        });
        segs[i].2.push((w.z, w.sheet));
    }
    // Split parameters of each segment, from crossings and touches, found
    // through a grid over the segment boxes.
    let mut lo = [f64::MAX; 2];
    let mut hi = [f64::MIN; 2];
    for (a, b, _) in &segs {
        for p in [a, b] {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let cell = ((hi[0] - lo[0]).max(hi[1] - lo[1]) / (segs.len() as f64).sqrt().max(1.0)).max(tol);
    // Boxes padded by the tolerance, so near misses share a cell.
    let key = |x: f64, k: usize| ((x - lo[k]) / cell).floor() as i64;
    let mut grid: FxHashMap<(i64, i64), Vec<usize>> = FxHashMap::default();
    for (i, (a, b, _)) in segs.iter().enumerate() {
        for gx in key(a[0].min(b[0]) - tol, 0)..=key(a[0].max(b[0]) + tol, 0) {
            for gy in key(a[1].min(b[1]) - tol, 1)..=key(a[1].max(b[1]) + tol, 1) {
                grid.entry((gx, gy)).or_default().push(i);
            }
        }
    }
    let param = |a: P2, b: P2, p: P2| {
        let d = [b[0] - a[0], b[1] - a[1]];
        ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1])
    };
    let mut cuts: Vec<Vec<P2>> = vec![Vec::new(); segs.len()];
    let mut seen: FxHashSet<(usize, usize)> = FxHashSet::default();
    for cellv in grid.values() {
        for (x, &i) in cellv.iter().enumerate() {
            for &j in &cellv[x + 1..] {
                if !seen.insert((i.min(j), i.max(j))) {
                    continue;
                }
                let ((a, b, _), (c, d, _)) = (&segs[i], &segs[j]);
                let (o1, o2) = (orient2d(*a, *b, *c), orient2d(*a, *b, *d));
                let (o3, o4) = (orient2d(*c, *d, *a), orient2d(*c, *d, *b));
                if o1 * o2 < 0.0 && o3 * o4 < 0.0 {
                    // A proper crossing.
                    let t = o3 / (o3 - o4);
                    let x = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                    cuts[i].push(x);
                    cuts[j].push(x);
                    continue;
                }
                // An end of one on or within the tolerance of the other
                // (touches, near misses, overlaps): cut the other at its foot.
                for (s, t, p, o) in [
                    (i, (a, b), *c, o1),
                    (i, (a, b), *d, o2),
                    (j, (c, d), *a, o3),
                    (j, (c, d), *b, o4),
                ] {
                    let (ta, tb) = (*t.0, *t.1);
                    let len = ((tb[0] - ta[0]).powi(2) + (tb[1] - ta[1]).powi(2)).sqrt();
                    let u = param(ta, tb, p);
                    if o.abs() <= tol * len && u > 0.0 && u < 1.0 {
                        cuts[s].push([ta[0] + u * (tb[0] - ta[0]), ta[1] + u * (tb[1] - ta[1])]);
                    }
                }
            }
        }
    }
    let mut weld = Weld {
        tol,
        pts: Vec::new(),
        grid: FxHashMap::default(),
    };
    let mut pieces: FxHashMap<(usize, usize), Vec<Span>> = FxHashMap::default();
    for (i, (a, b, zs)) in segs.iter().enumerate() {
        let mut ps: Vec<(f64, P2)> = cuts[i].iter().map(|&p| (param(*a, *b, p), p)).collect();
        ps.push((0.0, *a));
        ps.push((1.0, *b));
        ps.sort_by(|x, y| x.0.total_cmp(&y.0));
        let ids: Vec<usize> = ps.iter().map(|&(_, p)| weld.id(p)).collect();
        for w in ids.windows(2) {
            if w[0] != w[1] {
                pieces
                    .entry((w[0].min(w[1]), w[0].max(w[1])))
                    .or_default()
                    .extend_from_slice(zs);
            }
        }
    }
    (weld.pts, pieces)
}

/// A small union-find.
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// The layered mesh of `plc`, or `None` (with the reason logged) when it is
/// not layered.
pub(crate) fn mesh(
    plc: &TaggedPlc,
    oracle: &BrepOracle<'_>,
    domain: &DomainTree,
) -> Option<Complex> {
    layered(plc, oracle, domain)
        .map_err(|why| rapidmesh_exact::log::info("mesh3.layered", format!("not layered: {why}")))
        .ok()
}

fn layered(
    plc: &TaggedPlc,
    oracle: &BrepOracle<'_>,
    domain: &DomainTree,
) -> Result<Complex, &'static str> {
    let (lo, hi) = oracle.bbox();
    let extent = (0..3)
        .map(|k| hi[k] - lo[k])
        .fold(0.0, f64::max)
        .max(1e-300);
    // The model tolerance: layouts carry steps and gaps of a nanometre under
    // a size of microns, which would only give needles. Closer than this,
    // levels, walls and plan points are one.
    let tol = (LAYERED_SNAP * domain.finest()).max(1e-9 * extent);
    let (levels, walls, flats) = layers(plc, tol)?;
    if levels.len() < 2 {
        return Err("a single level");
    }
    let mids: Vec<f64> = levels.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    // The plan size: the finest size over the heights of the slabs.
    let target = |p: P2| -> f64 {
        mids.iter()
            .map(|&z| domain.h_at([p[0], p[1], z]))
            .fold(f64::INFINITY, f64::min)
    };

    // ---- the plan and its 2D mesh
    let (pts, pieces) = plan(&walls, tol);
    // Only a stack with a layer thinner than a fraction of the size takes
    // this path: refinement meshes thicker layers with better tets.
    let thinness = levels
        .windows(2)
        .zip(&mids)
        .map(|(w, &z)| {
            let mut hs: Vec<f64> = pts.iter().map(|p| domain.h_at([p[0], p[1], z])).collect();
            hs.sort_by(f64::total_cmp);
            (w[1] - w[0]) / hs[hs.len() / 2]
        })
        .fold(f64::INFINITY, f64::min);
    rapidmesh_exact::log::stat("mesh3.layered.thinness", thinness);
    if !(thinness < LAYERED_THIN) {
        return Err("no layer thin against the size");
    }
    let mut boundary: Vec<P2> = pts.clone();
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut heights: FxHashMap<(usize, usize), Vec<Span>> = FxHashMap::default();
    let mut keys: Vec<&(usize, usize)> = pieces.keys().collect();
    keys.sort_unstable();
    for &(u, v) in keys {
        let zs = &pieces[&(u, v)];
        let (a, b) = (pts[u], pts[v]);
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let h = target([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]);
        let n = ((len / h).ceil() as usize).max(1);
        let mut prev = u;
        for k in 1..=n {
            let next = if k == n {
                v
            } else {
                let t = k as f64 / n as f64;
                boundary.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
                boundary.len() - 1
            };
            segments.push((prev, next));
            heights.insert((prev.min(next), prev.max(next)), zs.clone());
            prev = next;
        }
    }
    let inside = |p: P2| p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1];
    let (p2, mut tris) = crate::surf2d::mesh_constrained(
        boundary,
        segments,
        target,
        inside,
        domain.finest(),
        28.0,
        0,
        4,
        12,
        true,
        |_, _| {},
    );
    if tris.is_empty() {
        return Err("an empty plan mesh");
    }
    for t in &mut tris {
        if orient2d(p2[t[0]], p2[t[1]], p2[t[2]]) < 0.0 {
            t.swap(1, 2);
        }
    }
    let n2 = p2.len();

    // ---- sheets: the heights each plan triangle carries a horizontal sheet
    // at, found through a grid of the triangle centroids, and the spans of
    // the vertical sheets over each plan edge
    let centroid = |t: &[usize; 3]| -> P2 {
        std::array::from_fn(|k| (p2[t[0]][k] + p2[t[1]][k] + p2[t[2]][k]) / 3.0)
    };
    let mut flat_at: Vec<Vec<f64>> = vec![Vec::new(); tris.len()];
    if !flats.is_empty() {
        let cell = ((hi[0] - lo[0]).max(hi[1] - lo[1]) / (tris.len() as f64).sqrt()).max(tol);
        let key = |x: f64, k: usize| ((x - lo[k]) / cell).floor() as i64;
        let mut grid: FxHashMap<(i64, i64), Vec<usize>> = FxHashMap::default();
        for (ti, t) in tris.iter().enumerate() {
            let c = centroid(t);
            grid.entry((key(c[0], 0), key(c[1], 1)))
                .or_default()
                .push(ti);
        }
        for f in &flats {
            let o = orient2d(f.t[0], f.t[1], f.t[2]).signum();
            let (flo, fhi) = (0..3).fold(([f64::MAX; 2], [f64::MIN; 2]), |(l, h), i| {
                (
                    std::array::from_fn(|k| l[k].min(f.t[i][k])),
                    std::array::from_fn(|k| h[k].max(f.t[i][k])),
                )
            });
            for gx in key(flo[0], 0)..=key(fhi[0], 0) {
                for gy in key(flo[1], 1)..=key(fhi[1], 1) {
                    for &ti in grid.get(&(gx, gy)).into_iter().flatten() {
                        let c = centroid(&tris[ti]);
                        let within =
                            (0..3).all(|e| orient2d(f.t[e], f.t[(e + 1) % 3], c) * o >= 0.0);
                        if within {
                            flat_at[ti].push(f.z);
                        }
                    }
                }
            }
        }
    }
    let sheet_sides: FxHashMap<(usize, usize), Vec<[f64; 2]>> = heights
        .iter()
        .filter_map(|(&key, zs)| {
            let spans: Vec<[f64; 2]> = zs
                .iter()
                .filter(|&&(z, sheet)| sheet && z[1] - z[0] > tol)
                .map(|&(z, _)| z)
                .collect();
            (!spans.is_empty()).then_some((key, spans))
        })
        .collect();

    // ---- regions per slab: triangles joined across edges without a wall
    let mut edge_tris: FxHashMap<(usize, usize), Vec<usize>> = FxHashMap::default();
    for (ti, t) in tris.iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            edge_tris.entry((a.min(b), a.max(b))).or_default().push(ti);
        }
    }
    let region_of: Vec<Vec<u32>> = mids
        .iter()
        .map(|&zm| {
            let mut parent: Vec<usize> = (0..tris.len()).collect();
            for (key, ts) in &edge_tris {
                if ts.len() != 2 {
                    continue;
                }
                let walled = heights.get(key).is_some_and(|zs| {
                    zs.iter()
                        .any(|&(z, sheet)| !sheet && z[0] <= zm && zm <= z[1])
                });
                if !walled {
                    let (a, b) = (find(&mut parent, ts[0]), find(&mut parent, ts[1]));
                    parent[a] = b;
                }
            }
            let mut of_root: FxHashMap<usize, u32> = FxHashMap::default();
            (0..tris.len())
                .map(|ti| {
                    let r = find(&mut parent, ti);
                    *of_root.entry(r).or_insert_with(|| {
                        let t = tris[r];
                        let c: P2 = std::array::from_fn(|k| {
                            (p2[t[0]][k] + p2[t[1]][k] + p2[t[2]][k]) / 3.0
                        });
                        oracle.region([c[0], c[1], zm])
                    })
                })
                .collect()
        })
        .collect();

    // ---- levels: every slab split into sub-slabs at the finest size over
    // it; each plan point keeps what its own size needs (below)
    let hv: Vec<Vec<f64>> = p2
        .iter()
        .map(|p| mids.iter().map(|&z| domain.h_at([p[0], p[1], z])).collect())
        .collect();
    let mut zs: Vec<f64> = vec![levels[0]];
    let mut slab_of: Vec<usize> = Vec::new();
    for (k, w) in levels.windows(2).enumerate() {
        let hz = hv
            .iter()
            .map(|h| h[k])
            .fold(f64::INFINITY, f64::min)
            .max(1e-300);
        let n = (((w[1] - w[0]) / hz).round() as usize).max(1);
        for s in 1..=n {
            zs.push(if s == n {
                w[1]
            } else {
                w[0] + (w[1] - w[0]) * s as f64 / n as f64
            });
            slab_of.push(k);
        }
    }

    // ---- the levels each plan point keeps: where a triangle at it changes
    // region, and enough that no column grows taller than the size
    let last = zs.len() - 1;
    let mut point_tris: Vec<Vec<usize>> = vec![Vec::new(); n2];
    for (ti, t) in tris.iter().enumerate() {
        for &v in t {
            point_tris[v].push(ti);
        }
    }
    let changes = |ti: usize, k: usize| region_of[slab_of[k - 1]][ti] != region_of[slab_of[k]][ti];
    // The sheets as level indices, and the levels each plan point keeps for
    // them: the ends of the vertical sheets beside it, the height of each
    // horizontal sheet over a triangle at it.
    let level_of = |z: f64| -> usize {
        let i = zs.partition_point(|&x| x < z);
        [i.saturating_sub(1), i.min(last)]
            .into_iter()
            .min_by(|&a, &b| (zs[a] - z).abs().total_cmp(&(zs[b] - z).abs()))
            .expect("two candidates")
    };
    let side_levels: FxHashMap<(usize, usize), Vec<[usize; 2]>> = sheet_sides
        .iter()
        .map(|(&key, spans)| (key, spans.iter().map(|z| z.map(level_of)).collect()))
        .collect();
    let mut flat_faces: FxHashSet<([usize; 3], usize)> = FxHashSet::default();
    let mut must: Vec<Vec<usize>> = vec![Vec::new(); n2];
    for (&(u, v), ks) in &side_levels {
        for k in ks {
            must[u].extend(k);
            must[v].extend(k);
        }
    }
    for (ti, t) in tris.iter().enumerate() {
        for &z in &flat_at[ti] {
            let k = level_of(z);
            let mut key = *t;
            key.sort_unstable();
            flat_faces.insert((key, k));
            for &v in t {
                must[v].push(k);
            }
        }
    }
    for m in &mut must {
        m.sort_unstable();
        m.dedup();
    }
    let kept: Vec<Vec<usize>> = (0..n2)
        .map(|v| {
            let mut ks = vec![0];
            for k in 1..last {
                let below = zs[ks[ks.len() - 1]];
                if point_tris[v].iter().any(|&t| changes(t, k))
                    || must[v].binary_search(&k).is_ok()
                    || zs[k + 1] - below > hv[v][slab_of[k]] * (1.0 + 1e-9)
                {
                    ks.push(k);
                }
            }
            ks.push(last);
            ks
        })
        .collect();
    let mut points: Vec<P3> = Vec::new();
    let mut at: Vec<(usize, usize)> = Vec::new();
    let columns: Vec<Vec<(usize, u32)>> = kept
        .iter()
        .enumerate()
        .map(|(v, ks)| {
            ks.iter()
                .map(|&k| {
                    points.push([p2[v][0], p2[v][1], zs[k]]);
                    at.push((v, k));
                    (k, points.len() as u32 - 1)
                })
                .collect()
        })
        .collect();

    let stat = rapidmesh_exact::log::stat;
    stat("mesh3.layered.plan_points", n2 as f64);
    stat("mesh3.layered.plan_triangles", tris.len() as f64);
    stat("mesh3.layered.levels", zs.len() as f64);
    stat("mesh3.layered.nodes", points.len() as f64);

    // ---- columns to tets
    let mut tets: Vec<[u32; 4]> = Vec::new();
    let mut regions: Vec<u32> = Vec::new();
    for (ti, t) in tris.iter().enumerate() {
        let col = t.map(|v| (v, columns[v].as_slice()));
        for (tet, k) in sweep(col) {
            let r = region_of[slab_of[k - 1]][ti];
            if r == 0 {
                continue;
            }
            let p = tet.map(|x| points[x as usize]);
            tets.push(if orient3d(p[0], p[1], p[2], p[3]) < 0.0 {
                [tet[0], tet[1], tet[3], tet[2]]
            } else {
                tet
            });
            regions.push(r);
        }
    }
    // A face inside a region lies on a sheet: on the side of a plan edge
    // within the levels of a vertical sheet over it, or on a plan triangle
    // at the level of a horizontal sheet over it.
    let on_sheet = |f: [(usize, usize); 3]| -> bool {
        let mut vs = f.map(|x| x.0);
        vs.sort_unstable();
        if vs[0] != vs[1] && vs[1] != vs[2] {
            return f[0].1 == f[1].1 && f[1].1 == f[2].1 && flat_faces.contains(&(vs, f[0].1));
        }
        let (u, v) = (vs[0], if vs[0] == vs[1] { vs[2] } else { vs[1] });
        side_levels.get(&(u, v)).is_some_and(|spans| {
            spans
                .iter()
                .any(|k| f.iter().all(|x| k[0] <= x.1 && x.1 <= k[1]))
        })
    };
    Ok(complete(
        points, at, tets, regions, on_sheet, oracle, extent, tol,
    ))
}

/// The tets of the column over a plan triangle, each with the level of its
/// top node. `col` holds per corner its plan point and its nodes bottom to
/// top as (level, node). A front triangle sweeps up the column, advancing one
/// corner at a time to its next node (the lowest next level first, the lower
/// plan point among equals), and each advance is one tet. The order in which
/// two corners advance depends on those two corners alone, so the columns on
/// either side of a plan edge split their shared side the same way.
fn sweep(col: [(usize, &[(usize, u32)]); 3]) -> Vec<([u32; 4], usize)> {
    let mut at = [0usize; 3];
    let mut out = Vec::new();
    loop {
        let next = (0..3)
            .filter(|&i| at[i] + 1 < col[i].1.len())
            .min_by_key(|&i| (col[i].1[at[i] + 1].0, col[i].0));
        let Some(i) = next else {
            return out;
        };
        let front = [0, 1, 2].map(|c| col[c].1[at[c]].1);
        at[i] += 1;
        let (k, node) = col[i].1[at[i]];
        out.push(([front[0], front[1], front[2], node], k));
    }
}

/// Faces, feature edges and vertex kinds of the tets, and the unused
/// points dropped. `at` holds each point's plan point and level, which
/// `on_sheet` takes per face corner to tell a sheet face inside a region.
#[allow(clippy::too_many_arguments)]
fn complete(
    points: Vec<P3>,
    at: Vec<(usize, usize)>,
    tets: Vec<[u32; 4]>,
    regions: Vec<u32>,
    on_sheet: impl Fn([(usize, usize); 3]) -> bool,
    oracle: &BrepOracle<'_>,
    extent: f64,
    tol: f64,
) -> Complex {
    // Drop the points no tet uses.
    let mut used = vec![u32::MAX; points.len()];
    let mut kept: Vec<P3> = Vec::new();
    for t in &tets {
        for &v in t {
            if used[v as usize] == u32::MAX {
                used[v as usize] = kept.len() as u32;
                kept.push(points[v as usize]);
            }
        }
    }
    let tets: Vec<[u32; 4]> = tets.iter().map(|t| t.map(|v| used[v as usize])).collect();
    let mut at_kept = vec![(0, 0); kept.len()];
    for (v, &u) in used.iter().enumerate() {
        if u != u32::MAX {
            at_kept[u as usize] = at[v];
        }
    }

    // Faces: tet faces where the region changes, wound into the tet's side.
    const FACE: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]];
    let mut first: FxHashMap<[u32; 3], ([u32; 3], u32)> = FxHashMap::default();
    let mut pairs: Vec<([u32; 3], u32, u32)> = Vec::new();
    for (t, &r) in tets.iter().zip(&regions) {
        for fl in FACE {
            let f = [t[fl[0]], t[fl[1]], t[fl[2]]];
            let mut key = f;
            key.sort_unstable();
            match first.remove(&key) {
                Some((g, s)) => {
                    if s != r || on_sheet(g.map(|v| at_kept[v as usize])) {
                        pairs.push((g, s, r));
                    }
                }
                None => {
                    first.insert(key, (f, r));
                }
            }
        }
    }
    pairs.extend(first.into_values().map(|(f, r)| (f, r, 0)));
    pairs.sort_unstable();
    // The probe for a face's patch reaches past the snap of the plan.
    let eps = 2.0 * tol + 1e-7 * extent;
    let faces: Vec<Face> = pairs
        .into_iter()
        .map(|(tri, a, b)| {
            let p = tri.map(|v| kept[v as usize]);
            let (u, w) = (sub(p[1], p[0]), sub(p[2], p[0]));
            let n = cross(u, w);
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-300);
            let c: P3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0);
            let s0: P3 = std::array::from_fn(|k| c[k] - eps * n[k] / l);
            let s1: P3 = std::array::from_fn(|k| c[k] + eps * n[k] / l);
            let patch = oracle
                .nearest_crossing(s0, s1, c)
                .map_or(u32::MAX, |x| x.patch);
            Face {
                tri,
                regions: [a, b],
                patch,
            }
        })
        .collect();

    // Feature edges: face edges where patches meet, on a B-rep edge.
    let mut edge_patches: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
    for f in &faces {
        for e in 0..3 {
            let (a, b) = (f.tri[e], f.tri[(e + 1) % 3]);
            edge_patches
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(f.patch);
        }
    }
    let curve_of_edge: FxHashMap<u32, u32> = oracle
        .curve_edge
        .iter()
        .enumerate()
        .map(|(ci, &e)| (e, ci as u32))
        .collect();
    let chain_segs: Vec<(P3, P3, u32)> = oracle
        .brep
        .edges
        .iter()
        .enumerate()
        .filter_map(|(e, edge)| curve_of_edge.get(&(e as u32)).map(|&ci| (edge, ci)))
        .flat_map(|(edge, ci)| edge.chain.windows(2).map(move |w| (w[0], w[1], ci)))
        .collect();
    let reach = (2.0 * tol).max(1e-6 * extent);
    let on_chain = |m: P3| -> Option<u32> {
        chain_segs.iter().find_map(|&(a, b, ci)| {
            let d = sub(b, a);
            let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let t = (dot(sub(m, a), d) / dd).clamp(0.0, 1.0);
            let x: P3 = std::array::from_fn(|k| a[k] + t * d[k] - m[k]);
            (dot(x, x) <= reach * reach).then_some(ci)
        })
    };
    let mut feature_edges: Vec<([u32; 2], u32)> = Vec::new();
    let mut keys: Vec<&(u32, u32)> = edge_patches.keys().collect();
    keys.sort_unstable();
    for &(a, b) in keys {
        let ps = &edge_patches[&(a, b)];
        if ps.iter().all(|&p| p == ps[0]) && ps.len() == 2 {
            continue;
        }
        let m: P3 = std::array::from_fn(|k| 0.5 * (kept[a as usize][k] + kept[b as usize][k]));
        if let Some(ci) = on_chain(m) {
            feature_edges.push(([a, b], ci));
        }
    }

    // Vertex kinds: corners, then curves, then patches. A corner is the
    // point within the snap of it, found through a grid of that cell size.
    let cell = |p: P3| p.map(|x| (x / reach).floor() as i64);
    let mut grid: FxHashMap<[i64; 3], Vec<u32>> = FxHashMap::default();
    for (v, &p) in kept.iter().enumerate() {
        grid.entry(cell(p)).or_default().push(v as u32);
    }
    let mut corner_of: FxHashMap<u32, u32> = FxHashMap::default();
    for (i, &c) in oracle.corners().iter().enumerate() {
        let k = cell(c);
        let near = (0..27)
            .map(|n| [k[0] + n % 3 - 1, k[1] + n / 3 % 3 - 1, k[2] + n / 9 - 1])
            .filter_map(|key| grid.get(&key))
            .flatten()
            .map(|&v| (v, sub(kept[v as usize], c)))
            .filter(|(_, d)| dot(*d, *d) <= reach * reach)
            .min_by(|x, y| dot(x.1, x.1).total_cmp(&dot(y.1, y.1)));
        if let Some((v, _)) = near {
            corner_of.insert(v, i as u32);
        }
    }
    let mut kinds = vec![VertexKind::Volume; kept.len()];
    for f in &faces {
        for &v in &f.tri {
            if kinds[v as usize] == VertexKind::Volume && f.patch != u32::MAX {
                kinds[v as usize] = VertexKind::Patch(f.patch);
            }
        }
    }
    for &(e, ci) in &feature_edges {
        for v in e {
            kinds[v as usize] = VertexKind::Curve(ci);
        }
    }
    for (&v, &i) in &corner_of {
        kinds[v as usize] = VertexKind::Corner(i);
    }
    Complex {
        points: kept,
        kinds,
        tets,
        regions,
        faces,
        feature_edges,
    }
}

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Columns with different levels per corner: the tets fill each column
    /// with positive volume, and two columns sharing a plan edge split
    /// their common side into the same triangles.
    #[test]
    fn column_sweeps_agree_across_shared_sides() {
        let plan: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let zs = [0.0, 0.3, 0.5, 0.9, 1.0];
        let kept: [&[usize]; 4] = [&[0, 2, 4], &[0, 1, 2, 3, 4], &[0, 4], &[0, 3, 4]];
        let mut points: Vec<P3> = Vec::new();
        let cols: Vec<Vec<(usize, u32)>> = (0..4)
            .map(|v| {
                kept[v]
                    .iter()
                    .map(|&k| {
                        points.push([plan[v][0], plan[v][1], zs[k]]);
                        (k, points.len() as u32 - 1)
                    })
                    .collect()
            })
            .collect();
        let mut sides: Vec<Vec<[u32; 3]>> = Vec::new();
        for tri in [[0usize, 1, 2], [1, 3, 2]] {
            let col = tri.map(|v| (v, cols[v].as_slice()));
            let mut vol = 0.0;
            let mut side = Vec::new();
            for (t, _) in sweep(col) {
                let p = t.map(|x| points[x as usize]);
                let o = orient3d(p[0], p[1], p[2], p[3]);
                assert!(o != 0.0, "flat tet");
                vol += o.abs() / 6.0;
                // Faces on the side over plan edge 1-2.
                let on = |x: u32| cols[1].iter().chain(&cols[2]).any(|c| c.1 == x);
                for f in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
                    let mut g = f.map(|i| t[i]);
                    if g.iter().all(|&x| on(x)) {
                        g.sort_unstable();
                        side.push(g);
                    }
                }
            }
            assert!((vol - 0.5).abs() < 1e-12, "column volume {vol}");
            side.sort_unstable();
            sides.push(side);
        }
        assert_eq!(sides[0], sides[1], "the shared side splits differently");
    }
}
