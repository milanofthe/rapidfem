//! The constrained Delaunay tetrahedralization of one region, by gift
//! wrapping from its boundary.
//!
//! The front holds the open faces, each oriented with the unmeshed side
//! positive: first the boundary triangles turned into the region, and a
//! sheet inside it from both sides. Each open face takes the vertex that
//! comes first in the order of the (perturbed) spheres through the face
//! among the vertices whose tet would meet the front only in shared
//! corners. When the region's edge segments are strongly Delaunay (the
//! surface stage splits them until they are), the constrained Delaunay
//! tetrahedralization exists, the vertex so chosen is its apex, and the
//! wrapping never needs another point (Shewchuk). Nothing outside the
//! region is ever built, so there is nothing to carve.

use super::predicates::{inside, orient, P3};
use rustc_hash::FxHashMap;

/// Why a region has no constrained Delaunay tetrahedralization here.
#[derive(Debug, Clone, PartialEq)]
pub enum CdtError {
    /// An open face without a valid apex (global ids).
    Stuck { face: [u32; 3] },
}

impl std::fmt::Display for CdtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CdtError::Stuck { face } => write!(f, "no apex for the open face {face:?}"),
        }
    }
}

impl std::error::Error for CdtError {}

/// The tets (global ids, positive) filling the region bounded by `faces`:
/// triangles over `points` (global ids), each oriented with the region on
/// its positive side, a sheet inside the region given in both
/// orientations.
pub fn tetrahedralize(points: &[P3], faces: &[[u32; 3]]) -> Result<Filled, CdtError> {
    tetrahedralize_on(points, faces, None)
}

/// [`tetrahedralize`] on the Delaunay tetrahedralization `dt` of the
/// region's points where it was made on exactly these (in global order).
pub fn tetrahedralize_on(
    points: &[P3],
    faces: &[[u32; 3]],
    dt: Option<super::region::Kept>,
) -> Result<Filled, CdtError> {
    // Local vertices in global order: the keys of the perturbation.
    let mut ids: Vec<u32> = faces.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let local: FxHashMap<u32, u32> = ids
        .iter()
        .enumerate()
        .map(|(i, &g)| (g, i as u32))
        .collect();
    let pts: Vec<P3> = ids.iter().map(|&g| points[g as usize]).collect();
    let dt = match dt {
        Some(k) if k.pts == pts => k.dt,
        _ => super::delaunay::Delaunay::new(&pts),
    };
    let (dts, nbrs) = dt.tets_and_neighbours();
    let conflict = conflicts(&pts, &dts, &nbrs, faces, &local);
    let (cavity_of, cavity_corners) = cavities(&dts, &nbrs, &conflict, pts.len());
    // The tet of the Delaunay tetrahedralization on the positive side of
    // each of its faces.
    let mut dt_tet: FxHashMap<[u32; 3], (u32, u32)> = FxHashMap::default();
    for (ti, t) in dts.iter().enumerate() {
        for (i, fl) in [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]]
            .iter()
            .enumerate()
        {
            dt_tet.insert(canon(fl.map(|k| t[k])), (t[i], ti as u32));
        }
    }
    let n0 = pts.len();
    let mut w = Wrap {
        pts: pts.clone(),
        dt_tet,
        conflict,
        cavity_of,
        cavity_corners,
        apexes: dt.apexes(),
        grid: Grid::new(&pts),
        front: FxHashMap::default(),
        cells: FxHashMap::default(),
        stack: Vec::new(),
        searched: 0,
        tested: std::cell::Cell::new(0),
        limit: SEARCH_BUDGET * pts.len() + SEARCH_BASE,
    };
    drop(dt);
    for f in faces {
        w.push(f.map(|g| local[&g]));
    }
    let mut tets: Vec<[u32; 4]> = Vec::new();
    while let Some(f) = w.stack.pop() {
        let global = |v: u32| {
            if (v as usize) < n0 {
                ids[v as usize]
            } else {
                STEINER + (v - n0 as u32)
            }
        };
        // A wrapping that tests far more tets than it makes is lost (a
        // boundary short of the Delaunay condition where no point helps):
        // it stops, and the caller meshes otherwise.
        if tets.len() > 64 * n0 + 64 || w.tested.get() > w.limit {
            return Err(CdtError::Stuck {
                face: f.map(global),
            });
        }
        if !w.front.contains_key(&canon(f)) {
            continue;
        }
        // Where the wrapping finds no apex (a curved face at a sharp angle
        // left short of the Delaunay condition), a point of its own just in
        // front of the face lets it go on.
        let Some(q) = w.apex(f).or_else(|| {
            (w.pts.len() - n0 < MAX_STEINER.max(n0 / 16))
                .then(|| w.steiner(f))
                .flatten()
        }) else {
            return Err(CdtError::Stuck {
                face: f.map(&global),
            });
        };
        #[cfg(test)]
        if std::env::var("RAPIDMESH_TRACE_CDT").is_ok() {
            trace_cdt(
                &w.pts,
                faces,
                &local,
                [f[0], f[1], f[2], q],
                f,
                tets.len(),
                &w,
            );
        }
        w.front.remove(&canon(f));
        let [a, b, c] = f;
        // The tet a b c q is positive, and `g` are its faces through q with
        // the tet on their positive side. Such a face closes the open face
        // it equals (the tet fills that face's unmeshed side), else it opens
        // turned away from the tet.
        for g in [[a, q, b], [b, q, c], [c, q, a]] {
            if w.front.remove(&canon(g)).is_none() {
                w.push([g[0], g[2], g[1]]);
            }
        }
        tets.push([a, b, c, q].map(global));
    }
    rapidmesh_exact::log::stat("bottomup.cdt_searched", w.searched as f64);
    Ok(Filled {
        tets,
        steiner: w.pts[n0..].to_vec(),
    })
}

/// The tets of a region: over the global ids of its boundary points, and
/// from [`STEINER`] on over the points the wrapping added (in order).
pub struct Filled {
    pub tets: Vec<[u32; 4]>,
    pub steiner: Vec<P3>,
}

/// The first id of a point the wrapping added, before the caller numbers
/// it after every other.
pub const STEINER: u32 = 1 << 30;

/// Front entries the validity tests may look at per point of the region,
/// and in all besides.
const SEARCH_BUDGET: usize = 4096;
const SEARCH_BASE: usize = 1 << 16;

/// Points the wrapping may add to a region of few points.
const MAX_STEINER: usize = 64;

/// Reports the first tet with a vertex in its sphere that its centroid
/// sees across no constraint (debugging aid).
#[cfg(test)]
fn trace_cdt(
    pts: &[P3],
    faces: &[[u32; 3]],
    local: &FxHashMap<u32, u32>,
    t: [u32; 4],
    f: [u32; 3],
    n: usize,
    w: &Wrap,
) {
    let pt = t.map(|v| pts[v as usize]);
    let cen0: P3 = std::array::from_fn(|k| (pt[0][k] + pt[1][k] + pt[2][k] + pt[3][k]) / 4.0);
    let cons: Vec<[u32; 3]> = faces.iter().map(|g| g.map(|x| local[&x])).collect();
    let samples: Vec<P3> = std::iter::once(cen0)
        .chain((0..4).map(|i| std::array::from_fn(|k| 0.8 * pt[i][k] + 0.2 * cen0[k])))
        .collect();
    for v in 0..pts.len() as u32 {
        if t.contains(&v) || !inside(pt, pts[v as usize], [t[0], t[1], t[2], t[3], v]) {
            continue;
        }
        let x = pts[v as usize];
        let blocked_from = |cen: P3| {
            cons.iter().any(|g| {
                if g.contains(&v) {
                    return false;
                }
                let [g0, g1, g2] = g.map(|u| pts[u as usize]);
                let (s0, s1) = (orient(g0, g1, g2, cen), orient(g0, g1, g2, x));
                if s0 == 0 || s1 == 0 || s0 == s1 {
                    return false;
                }
                let o = [
                    orient(cen, x, g0, g1),
                    orient(cen, x, g1, g2),
                    orient(cen, x, g2, g0),
                ];
                !(o.iter().any(|&k| k > 0) && o.iter().any(|&k| k < 0))
            })
        };
        if samples.iter().any(|&c| !blocked_from(c)) {
            if let Some(&q) = w.apexes.get(&canon(f)) {
                for g in [[f[0], q, f[1]], [f[1], q, f[2]], [f[2], q, f[0]]] {
                    let away = canon([g[0], g[2], g[1]]);
                    if w.front.contains_key(&away) && !w.front.contains_key(&canon(g)) {
                        eprintln!("TRACE   dt apex {q}: new face {g:?} on the meshed side of an open face");
                    }
                }
                let tq = [f[0], f[1], f[2], q];
                let ptq = tq.map(|u| pts[u as usize]);
                for g in w.front.values() {
                    if canon(*g) == canon(f) {
                        continue;
                    }
                    let mut why = Vec::new();
                    for e in [[f[0], q], [f[1], q], [f[2], q]] {
                        if seg_meets_tri(pts, e, *g) {
                            why.push(format!("edge {e:?}"));
                        }
                    }
                    for k in 0..3 {
                        let e = [g[k], g[(k + 1) % 3]];
                        for h in [[f[0], f[1], q], [f[1], f[2], q], [f[2], f[0], q]] {
                            if seg_meets_tri(pts, e, h) {
                                why.push(format!("front edge {e:?} vs {h:?}"));
                            }
                        }
                    }
                    for &u in g {
                        if !tq.contains(&u) && strictly_inside(ptq, pts[u as usize]) {
                            why.push(format!("vertex {u} inside"));
                        }
                    }
                    if !why.is_empty() {
                        eprintln!(
                            "TRACE   dt apex {q} {:?} blocked by {g:?} {:?}: {why:?}",
                            pts[q as usize],
                            g.map(|u| pts[u as usize])
                        );
                    }
                }
            }
            eprintln!(
                "TRACE tet {n}: face {f:?} {:?} dt apex {:?} took {} {:?}; vertex {v} {x:?} in the sphere, visible",
                f.map(|u| pts[u as usize]),
                w.apexes.get(&canon(f)),
                t[3],
                pt[3]
            );
            return;
        }
    }
}

/// Per Delaunay tet, whether it meets a constraint (a triangle of `faces`)
/// other than in shared vertices and edges. Only constraints that are no
/// Delaunay face can: each is followed through the tets around it.
fn conflicts(
    pts: &[P3],
    dts: &[[u32; 4]],
    nbrs: &[[u32; 4]],
    faces: &[[u32; 3]],
    local: &FxHashMap<u32, u32>,
) -> Vec<bool> {
    let mut dt_faces: rustc_hash::FxHashSet<[u32; 3]> = rustc_hash::FxHashSet::default();
    let mut vertex_tets: Vec<Vec<u32>> = vec![Vec::new(); pts.len()];
    for (ti, t) in dts.iter().enumerate() {
        for i in 0..4 {
            let mut f = [t[(i + 1) % 4], t[(i + 2) % 4], t[(i + 3) % 4]];
            f.sort_unstable();
            dt_faces.insert(f);
            vertex_tets[t[i] as usize].push(ti as u32);
        }
    }
    let mut conflict = vec![false; dts.len()];
    let mut seen: FxHashMap<u32, u32> = FxHashMap::default();
    for (gi, g) in faces.iter().enumerate() {
        let g = g.map(|x| local[&x]);
        let mut key = g;
        key.sort_unstable();
        if dt_faces.contains(&key) {
            continue;
        }
        let mut queue: Vec<u32> = g
            .iter()
            .flat_map(|&v| vertex_tets[v as usize].iter().copied())
            .collect();
        while let Some(t) = queue.pop() {
            if seen.insert(t, gi as u32) == Some(gi as u32) {
                continue;
            }
            if tet_meets_tri(pts, dts[t as usize], g) {
                conflict[t as usize] = true;
                queue.extend(nbrs[t as usize].iter().copied().filter(|&n| n != u32::MAX));
            }
        }
    }
    conflict
}

/// The cavities of the conflicting tets (connected through their faces):
/// per vertex, the cavities it is a corner of, and per cavity its corners.
/// A tet of the constrained tetrahedralization inside a cavity has its
/// corners among the cavity's.
fn cavities(
    dts: &[[u32; 4]],
    nbrs: &[[u32; 4]],
    conflict: &[bool],
    n: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut comp = vec![u32::MAX; dts.len()];
    let mut corners: Vec<Vec<u32>> = Vec::new();
    for start in 0..dts.len() {
        if !conflict[start] || comp[start] != u32::MAX {
            continue;
        }
        let id = corners.len() as u32;
        let mut vs: Vec<u32> = Vec::new();
        let mut queue = vec![start as u32];
        comp[start] = id;
        while let Some(t) = queue.pop() {
            vs.extend(dts[t as usize]);
            for &nb in &nbrs[t as usize] {
                if nb != u32::MAX && conflict[nb as usize] && comp[nb as usize] == u32::MAX {
                    comp[nb as usize] = id;
                    queue.push(nb);
                }
            }
        }
        vs.sort_unstable();
        vs.dedup();
        corners.push(vs);
    }
    let mut of_vertex: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (c, vs) in corners.iter().enumerate() {
        for &v in vs {
            of_vertex[v as usize].push(c as u32);
        }
    }
    (of_vertex, corners)
}

/// Whether the tet `t` and the triangle `g` (vertex ids over `pts`) share a
/// point other than in common vertices and edges.
fn tet_meets_tri(pts: &[P3], t: [u32; 4], g: [u32; 3]) -> bool {
    let pt = t.map(|v| pts[v as usize]);
    let edges = [
        [t[0], t[1]],
        [t[0], t[2]],
        [t[0], t[3]],
        [t[1], t[2]],
        [t[1], t[3]],
        [t[2], t[3]],
    ];
    let faces = [
        [t[0], t[1], t[2]],
        [t[0], t[3], t[1]],
        [t[0], t[2], t[3]],
        [t[1], t[3], t[2]],
    ];
    let shares_face = faces.iter().any(|f| {
        let mut a = *f;
        let mut b = g;
        a.sort_unstable();
        b.sort_unstable();
        a == b
    });
    if shares_face {
        return false;
    }
    edges.iter().any(|&e| seg_meets_tri(pts, e, g))
        || (0..3).any(|k| {
            let e = [g[k], g[(k + 1) % 3]];
            faces.iter().any(|&h| seg_meets_tri(pts, e, h))
        })
        || g.iter()
            .any(|&v| !t.contains(&v) && strictly_inside(pt, pts[v as usize]))
}

/// The corners of a cavity searched as a whole for an apex.
const SMALL_CAVITY: usize = 256;

/// A directed triangle rotated to start at its smallest vertex.
fn canon(f: [u32; 3]) -> [u32; 3] {
    let i = (0..3).min_by_key(|&i| f[i]).unwrap_or(0);
    [f[i], f[(i + 1) % 3], f[(i + 2) % 3]]
}

/// A uniform grid over the region's points, about two points a cell.
struct Grid {
    lo: P3,
    cell: f64,
    dims: [i64; 3],
    points: FxHashMap<[i64; 3], Vec<u32>>,
}

impl Grid {
    fn new(pts: &[P3]) -> Grid {
        let (lo, hi) = bbox(pts);
        let ext: P3 = std::array::from_fn(|k| (hi[k] - lo[k]).max(0.0));
        let span = ext.iter().copied().fold(0.0, f64::max).max(1e-300);
        // Cells by the point density over the box, flat boxes counted by
        // their largest faces.
        let vol = ext.iter().map(|&x| x.max(1e-3 * span)).product::<f64>();
        let cell = (2.0 * vol / pts.len().max(1) as f64)
            .cbrt()
            .max(1e-9 * span);
        let dims = ext.map(|x| (x / cell).floor() as i64 + 1);
        let mut g = Grid {
            lo,
            cell,
            dims,
            points: FxHashMap::default(),
        };
        for (i, &p) in pts.iter().enumerate() {
            g.points.entry(g.key(p)).or_default().push(i as u32);
        }
        g
    }

    fn key(&self, p: P3) -> [i64; 3] {
        std::array::from_fn(|k| {
            (((p[k] - self.lo[k]) / self.cell).floor() as i64).clamp(0, self.dims[k] - 1)
        })
    }

    /// The cells over the box `lo..hi`.
    fn cells(&self, lo: P3, hi: P3) -> impl Iterator<Item = [i64; 3]> {
        let (a, b) = (self.key(lo), self.key(hi));
        (a[0]..=b[0]).flat_map(move |x| {
            (a[1]..=b[1]).flat_map(move |y| (a[2]..=b[2]).map(move |z| [x, y, z]))
        })
    }
}

struct Wrap {
    pts: Vec<P3>,
    /// The Delaunay apex and tet on the positive side of each Delaunay face,
    /// and the tets that meet a constraint missing from the tetrahedralization.
    dt_tet: FxHashMap<[u32; 3], (u32, u32)>,
    conflict: Vec<bool>,
    /// The cavities each vertex is a corner of, and each cavity's corners.
    cavity_of: Vec<Vec<u32>>,
    cavity_corners: Vec<Vec<u32>>,
    /// The apex of each face of the Delaunay tetrahedralization on its
    /// positive side.
    apexes: FxHashMap<[u32; 3], u32>,
    grid: Grid,
    /// Open faces by their canonical rotation.
    front: FxHashMap<[u32; 3], [u32; 3]>,
    /// Open faces by grid cell (stale entries skipped).
    cells: FxHashMap<[i64; 3], Vec<[u32; 3]>>,
    stack: Vec<[u32; 3]>,
    /// Faces that needed the search beyond the Delaunay apex.
    searched: usize,
    /// Front entries looked at by the validity tests so far, the work of
    /// the search.
    tested: std::cell::Cell<usize>,
    /// The work the wrapping may spend before it gives up.
    limit: usize,
}

impl Wrap {
    fn p(&self, v: u32) -> P3 {
        self.pts[v as usize]
    }

    fn push(&mut self, f: [u32; 3]) {
        let key = canon(f);
        self.front.insert(key, f);
        let (lo, hi) = bbox(&f.map(|v| self.p(v)));
        let cells: Vec<[i64; 3]> = self.grid.cells(lo, hi).collect();
        for c in cells {
            self.cells.entry(c).or_default().push(key);
        }
        self.stack.push(f);
    }

    /// The apex of the open face `f`: first in sphere order among the
    /// vertices on its positive side whose tet is valid. The Delaunay apex
    /// has an empty sphere, so it is first whenever it is valid; else the
    /// search takes the vertices in the sphere of the first valid one.
    fn apex(&mut self, f: [u32; 3]) -> Option<u32> {
        // A Delaunay tet meeting no constraint is constrained Delaunay (its
        // sphere is empty), hence the one tet of the tetrahedralization on
        // `f`: no test against the front.
        if let Some(&(q, t)) = self.dt_tet.get(&canon(f)) {
            if !self.conflict[t as usize] {
                return Some(q);
            }
        }
        if let Some(&q) = self.apexes.get(&canon(f)) {
            if self.valid(f, q) {
                return Some(q);
            }
        }
        self.searched += 1;
        // In the cavities at its corners: the apex is a corner of the cavity
        // its unmeshed side lies in.
        // Only a small cavity pays: a large one (a slab whose Delaunay
        // tetrahedralization runs through its holes) goes to the search.
        let small = f
            .iter()
            .flat_map(|&v| self.cavity_of[v as usize].iter())
            .all(|&c| self.cavity_corners[c as usize].len() <= SMALL_CAVITY);
        let mut near: Vec<u32> = if small {
            f.iter()
                .flat_map(|&v| self.cavity_of[v as usize].iter())
                .flat_map(|&c| self.cavity_corners[c as usize].iter().copied())
                .collect()
        } else {
            Vec::new()
        };
        if !near.is_empty() {
            let [a, b, c] = f.map(|v| self.p(v));
            near.sort_unstable();
            near.dedup();
            near.retain(|&v| !f.contains(&v) && orient(a, b, c, self.p(v)) > 0);
            self.sort(f, &mut near);
            if let Some(q) = near.into_iter().find(|&q| self.valid(f, q)) {
                return Some(q);
            }
        }
        let [a, b, c] = f.map(|v| self.p(v));
        let positive = |v: u32| !f.contains(&v) && orient(a, b, c, self.p(v)) > 0;
        // A first valid vertex, from the nearest cells outward.
        let centre: P3 = std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0);
        let home = self.grid.key(centre);
        let reach = self.grid.dims.iter().copied().max().unwrap_or(1);
        let mut first = None;
        'rings: for r in 0..=reach {
            // The cells of a ring are work too, empty or not.
            let side = (2 * r + 1) as usize;
            let inner = (2 * r - 1).max(0) as usize;
            self.tested
                .set(self.tested.get() + side * side * side - inner * inner * inner);
            if self.tested.get() > self.limit {
                return None;
            }
            let mut ring: Vec<u32> = Vec::new();
            for x in home[0] - r..=home[0] + r {
                for y in home[1] - r..=home[1] + r {
                    for z in home[2] - r..=home[2] + r {
                        let d = (x - home[0])
                            .abs()
                            .max((y - home[1]).abs())
                            .max((z - home[2]).abs());
                        if d != r {
                            continue;
                        }
                        if let Some(vs) = self.grid.points.get(&[x, y, z]) {
                            ring.extend(vs.iter().copied().filter(|&v| positive(v)));
                        }
                    }
                }
            }
            self.sort(f, &mut ring);
            for q in ring {
                if self.valid(f, q) {
                    first = Some(q);
                    break 'rings;
                }
            }
        }
        let q0 = first?;
        // Every vertex before it in sphere order lies in its sphere.
        let t0 = [a, b, c, self.p(q0)];
        let (lo, hi) = sphere_box(t0);
        let mut cand: Vec<u32> = self
            .grid
            .cells(lo, hi)
            .filter_map(|k| self.grid.points.get(&k))
            .flatten()
            .copied()
            .filter(|&v| v != q0 && positive(v) && inside(t0, self.p(v), [f[0], f[1], f[2], q0, v]))
            .collect();
        cand.push(q0);
        cand.sort_unstable();
        cand.dedup();
        self.sort(f, &mut cand);
        cand.into_iter().find(|&q| self.valid(f, q))
    }

    /// A new point just in front of the open face `f` (on its unmeshed
    /// side, nearer each try) whose tet with `f` is valid.
    fn steiner(&mut self, f: [u32; 3]) -> Option<u32> {
        let [a, b, c] = f.map(|v| self.p(v));
        let n = cross(sub(b, a), sub(c, a));
        let l = dot(n, n).sqrt();
        if !(l > 0.0) {
            return None;
        }
        // The positive side lies against the triangle's right-hand normal.
        let n = n.map(|x| -x / l);
        let cen: P3 = std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0);
        let short = [dist2(a, b), dist2(b, c), dist2(c, a)]
            .into_iter()
            .fold(f64::INFINITY, f64::min)
            .sqrt();
        let q = self.pts.len() as u32;
        for k in 0..24 {
            let d = 0.5 * short * 0.5f64.powi(k);
            let p: P3 = std::array::from_fn(|i| cen[i] + d * n[i]);
            if orient(a, b, c, p) <= 0 {
                continue;
            }
            self.pts.push(p);
            if self.valid(f, q) {
                let key = self.grid.key(p);
                self.grid.points.entry(key).or_default().push(q);
                self.cavity_of.push(Vec::new());
                return Some(q);
            }
            self.pts.pop();
        }
        None
    }

    /// Sorts vertices on the positive side of `f` by sphere order: `q`
    /// before `p` when `q` lies inside the sphere through `f` and `p`.
    fn sort(&self, f: [u32; 3], vs: &mut [u32]) {
        let [a, b, c] = f.map(|v| self.p(v));
        vs.sort_by(|&x, &y| {
            if x == y {
                std::cmp::Ordering::Equal
            } else if inside([a, b, c, self.p(y)], self.p(x), [f[0], f[1], f[2], y, x]) {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });
    }

    /// Whether the tet of `f` and `q` meets the front only in shared
    /// corners: no new edge meets an open face, no open edge meets a new
    /// face, and no front vertex lies inside.
    fn valid(&self, f: [u32; 3], q: u32) -> bool {
        let t = [f[0], f[1], f[2], q];
        let pt = t.map(|v| self.p(v));
        let (lo, hi) = bbox(&pt);
        let new_edges = [[f[0], q], [f[1], q], [f[2], q]];
        let new_faces = [[f[0], f[1], q], [f[1], f[2], q], [f[2], f[0], q]];
        let fkey = canon(f);
        // A new face on an open face that turns its unmeshed side away
        // puts the tet on that face's meshed side (the tests below see only
        // shared corners there). A sheet is open from both sides; then the
        // side facing the tet is the one it fills.
        for g in [[f[0], q, f[1]], [f[1], q, f[2]], [f[2], q, f[0]]] {
            let away = canon([g[0], g[2], g[1]]);
            if self.front.contains_key(&away) && !self.front.contains_key(&canon(g)) {
                return false;
            }
        }
        let mut seen: rustc_hash::FxHashSet<[u32; 3]> = rustc_hash::FxHashSet::default();
        for cell in self.grid.cells(lo, hi) {
            let Some(keys) = self.cells.get(&cell) else {
                continue;
            };
            self.tested.set(self.tested.get() + keys.len());
            for key in keys {
                // Closed faces stay listed in their cells: skipped first.
                let Some(&g) = self.front.get(key) else {
                    continue;
                };
                if *key == fkey || !seen.insert(*key) {
                    continue;
                }
                let pg = g.map(|v| self.p(v));
                let (glo, ghi) = bbox(&pg);
                if (0..3).any(|k| ghi[k] < lo[k] || glo[k] > hi[k]) {
                    continue;
                }
                // On or below the plane of the base `f`: the tet lies above it,
                // so the face could meet it only inside `f`, which an open face
                // of a valid front never does (its neighbours in the plane of
                // a slab are most of what a flat tet's box holds).
                if pg.iter().all(|&v| orient(pt[0], pt[1], pt[2], v) <= 0) {
                    continue;
                }
                // Separated by a plane of the tet (the face strictly outside
                // it) or by the face's own plane (the tet strictly on one
                // side): no contact at all.
                let outside = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
                    .iter()
                    .any(|fl| {
                        let (x, y, z) = (pt[fl[0]], pt[fl[1]], pt[fl[2]]);
                        pg.iter().all(|&v| orient(x, y, z, v) < 0)
                    });
                if outside {
                    continue;
                }
                let side = pt.map(|v| orient(pg[0], pg[1], pg[2], v));
                if side.iter().all(|&x| x > 0) || side.iter().all(|&x| x < 0) {
                    continue;
                }
                for e in new_edges {
                    if seg_meets_tri(&self.pts, e, g) {
                        return false;
                    }
                }
                for k in 0..3 {
                    let e = [g[k], g[(k + 1) % 3]];
                    for h in new_faces {
                        if seg_meets_tri(&self.pts, e, h) {
                            return false;
                        }
                    }
                }
                for &v in &g {
                    if !t.contains(&v) && strictly_inside(pt, self.p(v)) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// A box around the sphere through the four points, padded for the
/// rounding of its centre (the order it bounds is exact).
fn sphere_box(t: [P3; 4]) -> (P3, P3) {
    let [a, b, c, d] = t;
    let (u, v, w) = (sub(b, a), sub(c, a), sub(d, a));
    let det = 2.0 * dot(u, cross(v, w));
    let (uu, vv, ww) = (dot(u, u), dot(v, v), dot(w, w));
    let num: P3 =
        std::array::from_fn(|k| uu * cross(v, w)[k] + vv * cross(w, u)[k] + ww * cross(u, v)[k]);
    if !(det.abs() > 0.0) {
        return ([f64::NEG_INFINITY; 3], [f64::INFINITY; 3]);
    }
    let o: P3 = std::array::from_fn(|k| a[k] + num[k] / det);
    let r = dot(sub(o, a), sub(o, a)).sqrt();
    let pad = r * 1e-6 + 1e-12 * (dot(u, u) + dot(v, v) + dot(w, w)).sqrt();
    if !(r.is_finite()) {
        return ([f64::NEG_INFINITY; 3], [f64::INFINITY; 3]);
    }
    (o.map(|x| x - r - pad), o.map(|x| x + r + pad))
}

fn dist2(a: P3, b: P3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
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

fn bbox(p: &[P3]) -> (P3, P3) {
    p.iter()
        .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), x| {
            (
                std::array::from_fn(|k| lo[k].min(x[k])),
                std::array::from_fn(|k| hi[k].max(x[k])),
            )
        })
}

/// Whether `x` lies strictly inside the positive tet `t`.
fn strictly_inside(t: [P3; 4], x: P3) -> bool {
    orient(t[0], t[1], t[2], x) > 0
        && orient(t[0], t[3], t[1], x) > 0
        && orient(t[0], t[2], t[3], x) > 0
        && orient(t[1], t[3], t[2], x) > 0
}

/// Whether the closed segment `e` and the closed triangle `t` (vertex ids
/// over `pts`) share a point other than a common vertex.
fn seg_meets_tri(pts: &[P3], e: [u32; 2], t: [u32; 3]) -> bool {
    let mut buf = [0u32; 2];
    let mut n = 0;
    for v in e {
        if t.contains(&v) {
            buf[n] = v;
            n += 1;
        }
    }
    let shared = &buf[..n];
    if shared.len() == 2 {
        return false; // an edge of the triangle
    }
    let p = |v: u32| pts[v as usize];
    let (e0, e1) = (p(e[0]), p(e[1]));
    let [t0, t1, t2] = t.map(p);
    // Apart where their boxes are (exact: closed boxes that do not overlap
    // hold no common point).
    if (0..3).any(|k| {
        e0[k].max(e1[k]) < t0[k].min(t1[k]).min(t2[k])
            || e0[k].min(e1[k]) > t0[k].max(t1[k]).max(t2[k])
    }) {
        return false;
    }
    let s0 = orient(t0, t1, t2, e0);
    let s1 = orient(t0, t1, t2, e1);
    if s0 != 0 && s0 == s1 {
        return false;
    }
    if s0 == 0 && s1 == 0 {
        return coplanar_seg_meets_tri(pts, e, t, shared);
    }
    // One point where the segment's line meets the plane. An end on the
    // plane is that point; a shared end there is the common vertex.
    if (s0 == 0 && shared.contains(&e[0])) || (s1 == 0 && shared.contains(&e[1])) {
        return false;
    }
    if s0 != 0 && s1 != 0 && !shared.is_empty() {
        // Strictly across the plane, the crossing is no triangle vertex
        // unless the segment runs through one; a shared vertex is an end,
        // which lies off the plane: impossible, so nothing shared counts.
    }
    // The crossing lies in the closed triangle when the segment turns the
    // same way (or not at all) around each of its edges.
    let o = [
        orient(e0, e1, t0, t1),
        orient(e0, e1, t1, t2),
        orient(e0, e1, t2, t0),
    ];
    let (pos, neg) = (o.iter().any(|&x| x > 0), o.iter().any(|&x| x < 0));
    if pos && neg {
        return false;
    }
    if !pos && !neg {
        // The segment lies in the plane of... impossible here: e0 or e1 is
        // off the plane, so the orientations cannot all vanish.
        return true;
    }
    // Inside or on the boundary. On the boundary through a shared vertex
    // only is no meeting: that needs the crossing at the vertex, which an
    // end off the plane cannot give (handled above for ends on it).
    true
}

/// [`seg_meets_tri`] for a segment in the plane of the triangle.
fn coplanar_seg_meets_tri(pts: &[P3], e: [u32; 2], t: [u32; 3], shared: &[u32]) -> bool {
    let p = |v: u32| pts[v as usize];
    let [t0, t1, t2] = t.map(p);
    // Drop the axis the triangle's normal points along most.
    let u = [t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2]];
    let v = [t2[0] - t0[0], t2[1] - t0[1], t2[2] - t0[2]];
    let n = [
        (u[1] * v[2] - u[2] * v[1]).abs(),
        (u[2] * v[0] - u[0] * v[2]).abs(),
        (u[0] * v[1] - u[1] * v[0]).abs(),
    ];
    let drop = if n[0] >= n[1] && n[0] >= n[2] {
        0
    } else if n[1] >= n[2] {
        1
    } else {
        2
    };
    let q = |x: P3| -> [f64; 2] {
        match drop {
            0 => [x[1], x[2]],
            1 => [x[2], x[0]],
            _ => [x[0], x[1]],
        }
    };
    let o2 = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| -> i8 {
        let d = geometry_predicates::orient2d(a, b, c);
        if d > 0.0 {
            1
        } else if d < 0.0 {
            -1
        } else {
            0
        }
    };
    let (a, b) = (q(p(e[0])), q(p(e[1])));
    let tri = [q(t0), q(t1), q(t2)];
    let turn = o2(tri[0], tri[1], tri[2]);
    let inside_closed = |x: [f64; 2]| (0..3).all(|k| o2(tri[k], tri[(k + 1) % 3], x) * turn >= 0);
    // An end that is no shared vertex inside the closed triangle.
    for (k, &x) in [a, b].iter().enumerate() {
        if !shared.contains(&e[k]) && inside_closed(x) {
            return true;
        }
    }
    // The segment crossing or touching a triangle edge away from shared
    // vertices.
    for k in 0..3 {
        let (c, d) = (tri[k], tri[(k + 1) % 3]);
        let (vc, vd) = (t[k], t[(k + 1) % 3]);
        let (d1, d2) = (o2(a, b, c), o2(a, b, d));
        let (d3, d4) = (o2(c, d, a), o2(c, d, b));
        if d1 * d2 < 0 && d3 * d4 < 0 {
            return true;
        }
        // Touches: an end on the edge, or an edge end on the segment,
        // unless that point is a shared vertex.
        let on = |x: [f64; 2], s: [f64; 2], t: [f64; 2]| {
            o2(s, t, x) == 0
                && x[0] >= s[0].min(t[0])
                && x[0] <= s[0].max(t[0])
                && x[1] >= s[1].min(t[1])
                && x[1] <= s[1].max(t[1])
        };
        if (on(a, c, d) && !shared.contains(&e[0]))
            || (on(b, c, d) && !shared.contains(&e[1]))
            || (on(c, a, b) && !shared.contains(&vc))
            || (on(d, a, b) && !shared.contains(&vd))
        {
            return true;
        }
    }
    // Through the interior from a shared vertex: the segment leaves the
    // shared corner into the triangle.
    if let Some(&s) = shared.first() {
        let far = if e[0] == s { b } else { a };
        let k = t.iter().position(|&x| x == s).unwrap_or(0);
        let (c, d) = (tri[(k + 1) % 3], tri[(k + 2) % 3]);
        let corner = tri[k];
        if o2(corner, c, far) * turn > 0 && o2(d, corner, far) * turn > 0 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::super::surface::{boundary, Boundary};
    use super::*;
    use crate::conform::MeshParams;
    use rapidmesh_brep::Model;
    use rapidmesh_geom::{solid_box, Scene};

    fn region_faces(model: &Model, b: &Boundary, r: u32) -> Vec<[u32; 3]> {
        super::super::region_faces(&model.brep, b, r)
    }

    fn volume(points: &[P3], tets: &[[u32; 4]]) -> f64 {
        tets.iter()
            .map(|t| {
                let [a, b, c, d] = t.map(|v| points[v as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
                (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                    + u[2] * (v[0] * w[1] - v[1] * w[0]))
                    / -6.0
            })
            .sum()
    }

    fn fill(scene: Scene, maxh: f64) -> (Model, Boundary, Vec<(u32, Vec<[u32; 4]>)>) {
        let model = Model::new(scene.assemble());
        let params = MeshParams {
            maxh,
            ..Default::default()
        };
        let domain = crate::cvt::build_sizing_domain(&model, &params);
        let b = boundary(&model, &domain, &params).unwrap();
        let mut regions: Vec<u32> = model
            .brep
            .faces
            .iter()
            .flat_map(|f| f.regions.map(|r| r.0))
            .filter(|&r| r != 0)
            .collect();
        regions.sort_unstable();
        regions.dedup();
        let tets = regions
            .iter()
            .map(|&r| {
                let faces = region_faces(&model, &b, r);
                let filled = tetrahedralize(&b.points, &faces).unwrap();
                assert!(filled.steiner.is_empty(), "region {r} needed points");
                (r, filled.tets)
            })
            .collect();
        (model, b, tets)
    }

    /// Every tet positive, the volume exact, and every boundary triangle a
    /// face of exactly one tet of its region.
    fn check(model: &Model, b: &Boundary, tets: &[(u32, Vec<[u32; 4]>)], volumes: &[f64]) {
        for ((r, ts), &want) in tets.iter().zip(volumes) {
            for t in ts {
                let p = t.map(|v| b.points[v as usize]);
                assert_eq!(orient(p[0], p[1], p[2], p[3]), 1, "region {r}");
            }
            let v = volume(&b.points, ts);
            assert!((v - want).abs() < 1e-9 * want, "region {r}: {v} vs {want}");
            let mut faces: FxHashMap<[u32; 3], usize> = FxHashMap::default();
            for t in ts {
                for f in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
                    let mut k = f.map(|i| t[i]);
                    k.sort_unstable();
                    *faces.entry(k).or_default() += 1;
                }
            }
            for mut t in region_faces(model, b, *r) {
                t.sort_unstable();
                assert!(
                    faces.get(&t).is_some_and(|&n| n >= 1),
                    "region {r}: face {t:?} missing"
                );
            }
        }
    }

    #[test]
    fn a_box_fills_exactly() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 1.0]));
        let (model, b, tets) = fill(scene, 0.4);
        check(&model, &b, &tets, &[6.0]);
    }

    /// A thin stack with a box inside the thick layer: the thick layer is
    /// no convex region, the thin one is flat against the size.
    #[test]
    fn a_thin_stack_with_an_inner_box_fills_exactly() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [100.0, 80.0, 40.0]));
        scene.add_solid(solid_box([0.0, 0.0, 40.0], [100.0, 80.0, 40.2]));
        scene.add_solid(solid_box([30.0, 20.0, 10.0], [60.0, 40.0, 20.0]));
        let (model, b, tets) = fill(scene, 16.0);
        let inner = 30.0 * 20.0 * 10.0;
        let mut vols: Vec<(u32, f64)> = tets
            .iter()
            .map(|(r, ts)| (*r, volume(&b.points, ts)))
            .collect();
        vols.sort_by(|x, y| x.1.total_cmp(&y.1));
        let want = [100.0 * 80.0 * 0.2, inner, 100.0 * 80.0 * 40.0 - inner];
        let by_region: Vec<f64> = tets
            .iter()
            .map(|(r, _)| {
                let v = vols.iter().find(|x| x.0 == *r).unwrap().1;
                *want
                    .iter()
                    .min_by(|a, b| (*a - v).abs().total_cmp(&(*b - v).abs()))
                    .unwrap()
            })
            .collect();
        check(&model, &b, &tets, &by_region);
    }

    /// Sheets inside a region (floating, and standing on a wall) and a
    /// void through its top: the sheets are faces of tets from both sides.
    #[test]
    fn sheets_and_a_hole_fill_exactly() {
        use rapidmesh_geom::{sheet_polygon, sheet_rect, FaceTag};
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [10.0, 10.0, 4.0]));
        scene.add_void(solid_box([4.0, 4.0, 2.0], [6.0, 6.0, 5.0]));
        let l = [
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [1.0, 3.0],
        ];
        scene.add_sheet(
            sheet_polygon(&l, &[], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            FaceTag(7),
        );
        scene.add_sheet(
            sheet_rect([0.0, 7.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 2.0]),
            FaceTag(8),
        );
        let (model, b, tets) = fill(scene, 0.7);
        check(&model, &b, &tets, &[400.0 - 8.0]);
    }

    /// An L shaped prism: no convex region, its reentrant edge inside.
    #[test]
    fn an_l_prism_fills_exactly() {
        use rapidmesh_geom::extrude_polygon;
        let l = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [1.0, 1.0],
            [1.0, 3.0],
            [0.0, 3.0],
        ];
        let mut scene = Scene::new();
        scene.add_solid(extrude_polygon(
            &l,
            &[],
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ));
        let (model, b, tets) = fill(scene, 0.3);
        check(&model, &b, &tets, &[12.0]);
    }

    /// Two sheets crossing inside a box: four half sheets on one edge.
    #[test]
    fn crossing_sheets_fill_exactly() {
        use rapidmesh_geom::{sheet_rect, FaceTag};
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]));
        scene.add_sheet(
            sheet_rect([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]),
            FaceTag(1),
        );
        scene.add_sheet(
            sheet_rect([-1.0, 0.0, -1.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]),
            FaceTag(2),
        );
        let (model, b, tets) = fill(scene, 0.25);
        check(&model, &b, &tets, &[27.0]);
    }

    /// A region read from `/tmp/region.txt` (as RAPIDMESH_DUMP_REGION
    /// writes it), for tracing a failure found on a model; with
    /// RAPIDMESH_TRACE_CDT set it reports the first tet that is not
    /// constrained Delaunay.
    #[test]
    #[ignore]
    fn region_from_file() {
        let txt = std::fs::read_to_string("/tmp/region.txt").unwrap();
        let mut lines = txt.lines();
        let head: Vec<usize> = lines
            .next()
            .unwrap()
            .split(' ')
            .map(|x| x.parse().unwrap())
            .collect();
        let pts: Vec<P3> = (0..head[0])
            .map(|_| {
                let v: Vec<f64> = lines
                    .next()
                    .unwrap()
                    .split(' ')
                    .map(|x| x.parse().unwrap())
                    .collect();
                [v[0], v[1], v[2]]
            })
            .collect();
        let faces: Vec<[u32; 3]> = (0..head[1])
            .map(|_| {
                let v: Vec<u32> = lines
                    .next()
                    .unwrap()
                    .split(' ')
                    .map(|x| x.parse().unwrap())
                    .collect();
                [v[0], v[1], v[2]]
            })
            .collect();
        let r = tetrahedralize(&pts, &faces);
        eprintln!(
            "result: {:?}",
            r.as_ref().map(|f| (f.tets.len(), f.steiner.len()))
        );
    }
}
