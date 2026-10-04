//! Discrete faces (scans and other imported soups) meshed on their own
//! facets instead of in a chart.
//!
//! The facets are taken as they are; the samples of the face's edges go
//! into their outline and the edges inside it, and the PLC points between
//! samples are collapsed away, so the outline runs from sample to sample as
//! the neighbouring faces meet it. The rest is remeshed to the size by
//! splits, collapses, flips and tangential smoothing (Botsch and Kobbelt),
//! every new point projected onto the facets that face its way. Each step is
//! local and keeps the mesh a manifold with the same outline: however thin
//! the face gets, no part of it can come to lie over another, which a chart
//! of pieces cannot promise.

use crate::surface::Slot;
use rapidmesh_geom::grid::HashGrid;
use rapidmesh_geom::vec3::{bbox, closest_on_tri, cross, dist, dot, sub};
use rustc_hash::{FxHashMap, FxHashSet};

type P3 = [f64; 3];

/// Rounds of splits, collapses, flips and smoothing.
const ROUNDS: usize = 8;

/// A change keeps each triangle it touches within this angle of its old
/// normal (cosine).
const KEEP_COS: f64 = 0.5;

/// A PLC point on the outline is collapsed away while its triangles turn
/// no more than this (cosine): the outline straightens between samples.
const OUTLINE_COS: f64 = 0.0;

/// A triangle faces within this (cosine) of the facets it lies on.
const FIT_COS: f64 = 0.5;

/// Facet edges bending more than this (degrees) are creases the mesh keeps
/// (the fidelity check's threshold): a point on one moves only along it.
const SHARP_DEG: f64 = 30.0;

/// A facet edge bending more than this (degrees) is a crease however large
/// its facets: no tessellation of a smooth surface bends so much.
const RIDGE_DEG: f64 = 60.0;

/// Ridges this many diagonals of a face's box long make it a ridged face.
const RIDGE_SPAN: f64 = 2.0;

/// A facet edge is a crease where its bend implies a radius below this
/// share of the size.
const CREASE_RADIUS: f64 = 0.5;

/// A change may move the surface by this share of the size at most.
const DEVIATION: f64 = 0.1;

/// The face's mesh: a slot per point, the face's own points, triangles over
/// the points, wound to the face's front.
pub(crate) struct Remeshed {
    pub slots: Vec<Slot>,
    pub own: Vec<P3>,
    pub tris: Vec<[usize; 3]>,
}

/// Remeshes the face whose `facets` (wound to its front) carry the loops
/// `rings` and the edges inside it `inner` (global ids of `points`; their
/// PLC chains `chains`), with the `corners` it touches and the `required`
/// points it must take, to the size `size`; `spacing` of the size keeps a
/// required point off the others. Its triangles lie on the `carrier` where
/// it has one, else on the facets.
#[allow(clippy::too_many_arguments)]
pub(crate) fn remesh(
    facets: &[[P3; 3]],
    rings: &[Vec<u32>],
    inner: &[Vec<u32>],
    chains: &[&[P3]],
    corners: &[u32],
    required: &[P3],
    points: &[P3],
    size: &dyn Fn(P3) -> f64,
    spacing: f64,
    carrier: Option<&rapidmesh_brep::Surface>,
) -> Result<Remeshed, &'static str> {
    if facets.is_empty() {
        return Err("no facets");
    }
    let mut m = Mesh::of(facets, carrier);
    // The edges of the outline (on one facet) and those inside the face
    // along its edges (the samples of `inner` lie on them) are locked.
    let mut count: FxHashMap<(u32, u32), usize> = FxHashMap::default();
    for t in &m.tris {
        for k in 0..3 {
            *count.entry(key(t[k], t[(k + 1) % 3])).or_default() += 1;
        }
    }
    for (&e, &n) in &count {
        if n == 1 {
            m.locked.insert(e);
        }
    }
    m.lock_chains(chains);
    // The creases of the facets: edges that bend steeply, or sharply over
    // a distance short against the size (a ridge the mesh must keep). A
    // gentle bend spread over facets as large as the size (a coarse
    // tessellation of a smooth surface) is left to the size field.
    let cos_sharp = SHARP_DEG.to_radians().cos();
    let cos_ridge = RIDGE_DEG.to_radians().cos();
    let mut across: FxHashMap<(u32, u32), Vec<usize>> = FxHashMap::default();
    for (i, t) in m.tris.iter().enumerate() {
        for k in 0..3 {
            across.entry(key(t[k], t[(k + 1) % 3])).or_default().push(i);
        }
    }
    let centroid = |t: [u32; 3]| -> P3 {
        let q = t.map(|v| m.pt(v));
        std::array::from_fn(|k| (q[0][k] + q[1][k] + q[2][k]) / 3.0)
    };
    let mut creases = Vec::new();
    for (e, ts) in &across {
        if let [x, y] = ts[..] {
            let (nx, ny) = (
                unit_or_z(m.tri_normal(m.tris[x])),
                unit_or_z(m.tri_normal(m.tris[y])),
            );
            let c = dot(nx, ny);
            if m.locked.contains(e) || c >= cos_sharp {
                continue;
            }
            let (cx, cy) = (centroid(m.tris[x]), centroid(m.tris[y]));
            let radius = dist(cx, cy) / c.clamp(-1.0, 1.0).acos();
            let mid: P3 = std::array::from_fn(|k| 0.5 * (cx[k] + cy[k]));
            if c < cos_ridge || radius < CREASE_RADIUS * size(mid) {
                creases.push(*e);
            }
        }
    }
    m.sharp.extend(creases);

    // ---- the samples into the locked edges
    let mut fixed: Vec<(u32, P3)> = Vec::new();
    for &g in rings
        .iter()
        .flatten()
        .chain(inner.iter().flatten())
        .chain(corners)
    {
        fixed.push((g, points[g as usize]));
    }
    fixed.sort_unstable_by_key(|x| x.0);
    fixed.dedup_by_key(|x| x.0);
    m.embed(&fixed)?;
    // ---- the PLC points between samples collapsed away
    m.straighten()?;
    // ---- the required points, where they are clear of the fixed ones
    let mut own: Vec<P3> = Vec::new();
    if !required.is_empty() {
        let mut near = point_grid(&m);
        for &p in required {
            let clear = spacing * size(p);
            if m.near_fixed(p, clear, &near) {
                continue;
            }
            if let Some(q) = m.insert(p, Slot::Own(own.len() as u32), &mut near) {
                own.push(q);
            }
        }
    }
    // ---- remeshed to the size
    for _ in 0..ROUNDS {
        m.split_long(size);
        m.collapse_short(size);
        m.flip_valence(size);
        m.smooth();
    }
    m.flip_delaunay(size);
    m.clean(size, 10.0);
    Ok(m.out(own))
}

/// The points of a mesh in a grid of cells about an edge long.
/// The points of the mesh in a grid of the median edge.
type Points = HashGrid<u32>;

fn point_grid(m: &Mesh) -> Points {
    let edges = m.edges();
    let mut lens: Vec<f64> = edges.iter().map(|&(a, b)| dist(m.pt(a), m.pt(b))).collect();
    lens.sort_by(f64::total_cmp);
    let cell = lens.get(lens.len() / 2).copied().unwrap_or(1.0).max(1e-300);
    let mut out = HashGrid::new(cell);
    for v in 0..m.p.len() as u32 {
        if !m.star[v as usize].is_empty() {
            out.insert(m.pt(v), v);
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mobility {
    Free,
    Along,
    Pinned,
}

/// Whether `facets` (wound alike) run along ridges: edges bending more
/// than [`RIDGE_DEG`] as long together as [`RIDGE_SPAN`] diagonals of their
/// box (a loft round the corners of a polygon: about 3.5), not the noisy
/// spikes of a scan (below 1).
pub(crate) fn ridged(facets: &[[P3; 3]]) -> bool {
    let cos_ridge = RIDGE_DEG.to_radians().cos();
    let bits = |p: P3| p.map(f64::to_bits);
    let mut at: FxHashMap<([u64; 3], [u64; 3]), P3> = FxHashMap::default();
    let mut ridge = 0.0;
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for t in facets {
        let n = unit_or_z(normal(t[0], t[1], t[2]));
        for k in 0..3 {
            for i in 0..3 {
                lo[i] = lo[i].min(t[k][i]);
                hi[i] = hi[i].max(t[k][i]);
            }
            let (a, b) = (bits(t[k]), bits(t[(k + 1) % 3]));
            let e = if a < b { (a, b) } else { (b, a) };
            if let Some(m) = at.insert(e, n) {
                if dot(m, n) < cos_ridge {
                    ridge += dist(t[k], t[(k + 1) % 3]);
                }
            }
        }
    }
    ridge >= RIDGE_SPAN * dist(lo, hi)
}

fn key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// The facets, for the point on them nearest a place, among those facing
/// its way (a thin part's other side faces away).
struct Reference {
    tris: Vec<[P3; 3]>,
    normals: Vec<P3>,
    grid: HashGrid<u32>,
    /// The true surface where the face has one (a B-spline band): it, not
    /// the facets, is what a triangle must lie on and a point goes to, so
    /// the facets' resolution does not matter; with the sign that turns its
    /// normal to the face's front.
    carrier: Option<(rapidmesh_brep::Surface, f64)>,
}

impl Reference {
    fn new(facets: &[[P3; 3]], carrier: Option<&rapidmesh_brep::Surface>) -> Reference {
        let mean = facets
            .iter()
            .map(|t| dist(t[0], t[1]) + dist(t[1], t[2]) + dist(t[2], t[0]))
            .sum::<f64>()
            / (3 * facets.len()) as f64;
        let cell = (2.0 * mean).max(1e-300);
        let mut grid = HashGrid::new(cell);
        for (i, t) in facets.iter().enumerate() {
            let (lo, hi) = bbox(t);
            grid.insert_box(lo, hi, i as u32);
        }
        Reference {
            tris: facets.to_vec(),
            normals: facets
                .iter()
                .map(|t| unit_or_z(normal(t[0], t[1], t[2])))
                .collect(),
            grid,
            carrier: carrier.map(|s| {
                // The facets face the front: the surface's normal agrees
                // with them or opposes them throughout.
                let agree: f64 = facets
                    .iter()
                    .map(|t| {
                        let c = std::array::from_fn(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0);
                        dot(s.closest(c).1, normal(t[0], t[1], t[2]))
                    })
                    .sum();
                (s.clone(), if agree < 0.0 { -1.0 } else { 1.0 })
            }),
        }
    }

    /// Whether triangle `t` lies on the facets as it faces: some facet within
    /// a quarter of its longest edge from its centroid faces within 60
    /// degrees of it. A triangle folded over, or across a thin part to its
    /// other side, finds none.
    fn fits(&self, t: [P3; 3]) -> bool {
        let n = normal(t[0], t[1], t[2]);
        let l = dot(n, n).sqrt();
        if !(l > 0.0) {
            return false;
        }
        let n = n.map(|x| x / l);
        let c: P3 = std::array::from_fn(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0);
        let reach = 0.25 * dist(t[0], t[1]).max(dist(t[1], t[2])).max(dist(t[2], t[0]));
        if let Some((s, sign)) = &self.carrier {
            let (q, m) = s.closest(c);
            return dist(c, q) <= reach && sign * dot(m, n) >= FIT_COS;
        }
        let cc = self.grid.key(c);
        let r = (reach / self.grid.cell()).ceil() as i64 + 1;
        // Ring by ring from the centroid's cell: the facet under it fits
        // almost always, found first.
        (0..=r).flat_map(|ring| self.grid.ring(cc, ring)).any(|&f| {
            if dot(self.normals[f as usize], n) < FIT_COS {
                return false;
            }
            let [a, b, e] = self.tris[f as usize];
            dist(c, closest_on_tri(c, a, b, e)) <= reach
        })
    }

    /// The point on the facets facing `n` nearest `p`, searched a few
    /// cells around it; `p` itself when none is near.
    fn project(&self, p: P3, n: P3) -> P3 {
        if let Some((s, _)) = &self.carrier {
            return s.closest(p).0;
        }
        let c = self.grid.key(p);
        let mut best = (f64::INFINITY, p);
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        for r in 0..=2i64 {
            for &t in self.grid.ring(c, r) {
                if !seen.insert(t) || dot(self.normals[t as usize], n) <= 0.0 {
                    continue;
                }
                let [a, b, cc] = self.tris[t as usize];
                let q = closest_on_tri(p, a, b, cc);
                let d = dist(p, q);
                if d < best.0 {
                    best = (d, q);
                }
            }
            // Nothing beyond ring `r` is nearer than `r` cells.
            if best.0 <= r as f64 * self.grid.cell() {
                break;
            }
        }
        best.1
    }
}

/// A triangle mesh under local changes: points, what is fixed of them,
/// triangles (dead ones kept as holes), the triangles at each point and the
/// locked edges.
struct Mesh {
    refs: Reference,
    p: Vec<P3>,
    slot: Vec<Option<Slot>>,
    tris: Vec<[u32; 3]>,
    alive: Vec<bool>,
    star: Vec<Vec<u32>>,
    locked: FxHashSet<(u32, u32)>,
    /// Creases: split and collapsed only along themselves, never flipped.
    sharp: FxHashSet<(u32, u32)>,
    /// Whether each point is on a locked edge (set once the outline runs
    /// from sample to sample; the locked edges change no more after).
    rim: Vec<bool>,
}

impl Mesh {
    fn of(facets: &[[P3; 3]], carrier: Option<&rapidmesh_brep::Surface>) -> Mesh {
        let mut id: FxHashMap<[u64; 3], u32> = FxHashMap::default();
        let mut p: Vec<P3> = Vec::new();
        let mut tris = Vec::with_capacity(facets.len());
        for t in facets {
            let v = t.map(|x| {
                *id.entry(x.map(f64::to_bits)).or_insert_with(|| {
                    p.push(x);
                    (p.len() - 1) as u32
                })
            });
            if v[0] != v[1] && v[1] != v[2] && v[2] != v[0] {
                tris.push(v);
            }
        }
        let mut star = vec![Vec::new(); p.len()];
        for (i, t) in tris.iter().enumerate() {
            for &v in t {
                star[v as usize].push(i as u32);
            }
        }
        Mesh {
            refs: Reference::new(facets, carrier),
            slot: vec![None; p.len()],
            alive: vec![true; tris.len()],
            p,
            tris,
            star,
            locked: FxHashSet::default(),
            sharp: FxHashSet::default(),
            rim: Vec::new(),
        }
    }

    fn pt(&self, v: u32) -> P3 {
        self.p[v as usize]
    }

    fn tri_normal(&self, t: [u32; 3]) -> P3 {
        normal(self.pt(t[0]), self.pt(t[1]), self.pt(t[2]))
    }

    /// The triangles on edge `a b`.
    fn on_edge(&self, a: u32, b: u32) -> Vec<u32> {
        self.star[a as usize]
            .iter()
            .copied()
            .filter(|&t| self.tris[t as usize].contains(&b))
            .collect()
    }

    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut out: Vec<u32> = self.star[v as usize]
            .iter()
            .flat_map(|&t| self.tris[t as usize])
            .filter(|&w| w != v)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn on_outline(&self, v: u32) -> bool {
        if let Some(&r) = self.rim.get(v as usize) {
            return r;
        }
        self.neighbours(v)
            .iter()
            .any(|&w| self.locked.contains(&key(v, w)))
    }

    /// The vertex normal: the area-weighted normals of its triangles.
    fn vertex_normal(&self, v: u32) -> P3 {
        let mut n = [0.0; 3];
        for &t in &self.star[v as usize] {
            let m = self.tri_normal(self.tris[t as usize]);
            for k in 0..3 {
                n[k] += m[k];
            }
        }
        unit_or_z(n)
    }

    /// The neighbours of `v` across creases.
    fn creased(&self, v: u32) -> Vec<u32> {
        self.neighbours(v)
            .into_iter()
            .filter(|&w| self.sharp.contains(&key(v, w)))
            .collect()
    }

    /// How a point may move: freely, along its creases (collapsed only
    /// into a neighbour across one, where the shape stays within the
    /// deviation), or not at all (fixed or on the outline).
    fn mobility(&self, v: u32) -> Mobility {
        if self.slot[v as usize].is_some() || self.on_outline(v) {
            return Mobility::Pinned;
        }
        if self.creased(v).is_empty() {
            Mobility::Free
        } else {
            Mobility::Along
        }
    }

    fn kill(&mut self, t: u32) {
        self.alive[t as usize] = false;
        for v in self.tris[t as usize] {
            self.star[v as usize].retain(|&x| x != t);
        }
    }

    fn add(&mut self, t: [u32; 3]) -> u32 {
        let id = self.tris.len() as u32;
        self.tris.push(t);
        self.alive.push(true);
        for v in t {
            self.star[v as usize].push(id);
        }
        id
    }

    fn add_point(&mut self, q: P3, slot: Option<Slot>) -> u32 {
        self.p.push(q);
        self.slot.push(slot);
        self.star.push(Vec::new());
        if !self.rim.is_empty() {
            self.rim.push(false);
        }
        (self.p.len() - 1) as u32
    }

    /// Locks the facet edges along the PLC chains of the edges inside.
    fn lock_chains(&mut self, chains: &[&[P3]]) {
        let id: FxHashMap<[u64; 3], u32> = self
            .p
            .iter()
            .enumerate()
            .map(|(i, x)| (x.map(f64::to_bits), i as u32))
            .collect();
        for chain in chains {
            for w in chain.windows(2) {
                let (Some(&a), Some(&b)) = (
                    id.get(&w[0].map(f64::to_bits)),
                    id.get(&w[1].map(f64::to_bits)),
                ) else {
                    continue;
                };
                if !self.on_edge(a, b).is_empty() {
                    self.locked.insert(key(a, b));
                }
            }
        }
    }

    /// Puts each fixed point into the mesh: onto the point at its place, or
    /// into the locked edge it lies on.
    fn embed(&mut self, fixed: &[(u32, P3)]) -> Result<(), &'static str> {
        let id: FxHashMap<[u64; 3], u32> = self
            .p
            .iter()
            .enumerate()
            .map(|(i, x)| (x.map(f64::to_bits), i as u32))
            .collect();
        let edges: Vec<(u32, u32)> = self.locked.iter().copied().collect();
        let mean = edges
            .iter()
            .map(|&(a, b)| dist(self.pt(a), self.pt(b)))
            .sum::<f64>()
            / edges.len().max(1) as f64;
        let cell = (2.0 * mean).max(1e-300);
        let mut grid: HashGrid<usize> = HashGrid::new(cell);
        for (i, &(a, b)) in edges.iter().enumerate() {
            let (lo, hi) = bbox([self.pt(a), self.pt(b)]);
            grid.insert_box(lo, hi, i);
        }
        // The points on each locked edge, by their place along it.
        let mut along: FxHashMap<usize, Vec<(f64, u32, P3)>> = FxHashMap::default();
        for &(g, q) in fixed {
            if let Some(&v) = id.get(&q.map(f64::to_bits)) {
                self.slot[v as usize] = Some(Slot::Global(g));
                continue;
            }
            let mut best: Option<(f64, usize)> = None;
            for &i in grid.around(grid.key(q), 1) {
                let (a, b) = edges[i];
                let d = seg_dist(q, self.pt(a), self.pt(b));
                if best.is_none_or(|x| d < x.0) {
                    best = Some((d, i));
                }
            }
            let Some((d, i)) = best else {
                return Err("an edge sample off the facets");
            };
            let (a, b) = edges[i];
            let (pa, pb) = (self.pt(a), self.pt(b));
            if d > 1e-6 * dist(pa, pb).max(1e-300) {
                return Err("an edge sample off the facets");
            }
            let t = dot(sub(q, pa), sub(pb, pa)) / dot(sub(pb, pa), sub(pb, pa)).max(1e-300);
            along.entry(i).or_default().push((t, g, q));
        }
        for (i, mut on) in along {
            on.sort_by(|x, y| x.0.total_cmp(&y.0));
            let (mut a, b) = edges[i];
            // Split from `a` towards `b`: each point cuts the rest of the edge.
            for (_, g, q) in on {
                let m = self.split(a, b, q, Some(Slot::Global(g)));
                a = m;
            }
        }
        Ok(())
    }

    /// Splits edge `a b` at `q` (a new point with `slot`); a locked edge
    /// stays locked in its halves.
    fn split(&mut self, a: u32, b: u32, q: P3, slot: Option<Slot>) -> u32 {
        let m = self.add_point(q, slot);
        for t in self.on_edge(a, b) {
            let tv = self.tris[t as usize];
            // The triangle turned so it runs x -> y -> z with {x, y} = {a, b}.
            let k = (0..3)
                .find(|&k| {
                    let (x, y) = (tv[k], tv[(k + 1) % 3]);
                    key(x, y) == key(a, b)
                })
                .unwrap_or(0);
            let (x, y, z) = (tv[k], tv[(k + 1) % 3], tv[(k + 2) % 3]);
            self.kill(t);
            self.add([x, m, z]);
            self.add([m, y, z]);
        }
        if self.locked.remove(&key(a, b)) {
            self.locked.insert(key(a, m));
            self.locked.insert(key(m, b));
        }
        if self.sharp.remove(&key(a, b)) {
            self.sharp.insert(key(a, m));
            self.sharp.insert(key(m, b));
        }
        m
    }

    /// Whether `a` may collapse into `b`: the link condition holds, no
    /// triangle that stays turns further than `min_cos` from its normal or
    /// gets longer edges than `max_len`, and `a` stays within `tol` of them.
    fn collapse_ok(&self, a: u32, b: u32, min_cos: f64, max_len: f64, tol: f64) -> bool {
        let on = self.on_edge(a, b);
        if on.is_empty() {
            return false;
        }
        let apex: FxHashSet<u32> = on
            .iter()
            .flat_map(|&t| self.tris[t as usize])
            .filter(|&v| v != a && v != b)
            .collect();
        let nb: FxHashSet<u32> = self.neighbours(b).into_iter().collect();
        let common: FxHashSet<u32> = self
            .neighbours(a)
            .into_iter()
            .filter(|v| nb.contains(v))
            .collect();
        if common != apex {
            return false;
        }
        let (pa, pb) = (self.pt(a), self.pt(b));
        let mut near = f64::INFINITY;
        for &t in &self.star[a as usize] {
            let tv = self.tris[t as usize];
            if tv.contains(&b) {
                continue;
            }
            let old = self.tri_normal(tv);
            let new_t = tv.map(|v| if v == a { b } else { v });
            let new = self.tri_normal(new_t);
            let (lo, ln) = (dot(old, old).sqrt(), dot(new, new).sqrt());
            if !(ln > 1e-12 * lo) || dot(old, new) < min_cos * lo * ln {
                return false;
            }
            if new_t
                .iter()
                .any(|&v| v != b && dist(self.pt(v), pb) > max_len)
            {
                return false;
            }
            let q = new_t.map(|v| self.pt(v));
            // Straightening the outline (no bound) must go through.
            if tol.is_finite() && !self.refs.fits(q) {
                return false;
            }
            near = near.min(dist(pa, closest_on_tri(pa, q[0], q[1], q[2])));
        }
        near <= tol || tol.is_infinite()
    }

    /// Collapses `a` into `b`.
    fn collapse(&mut self, a: u32, b: u32) {
        let around = self.neighbours(a);
        for t in self.on_edge(a, b) {
            self.kill(t);
        }
        let ts = std::mem::take(&mut self.star[a as usize]);
        for t in ts {
            let tv = &mut self.tris[t as usize];
            for v in tv.iter_mut() {
                if *v == a {
                    *v = b;
                }
            }
            self.star[b as usize].push(t);
        }
        let moved: Vec<(u32, u32)> = around
            .iter()
            .map(|&w| key(a, w))
            .filter(|e| self.locked.contains(e))
            .collect();
        for e in moved {
            self.locked.remove(&e);
            let w = if e.0 == a { e.1 } else { e.0 };
            if w != b {
                self.locked.insert(key(b, w));
            }
        }
        let moved: Vec<(u32, u32)> = around
            .iter()
            .map(|&w| key(a, w))
            .filter(|e| self.sharp.contains(e))
            .collect();
        for e in moved {
            self.sharp.remove(&e);
            let w = if e.0 == a { e.1 } else { e.0 };
            if w != b {
                self.sharp.insert(key(b, w));
            }
        }
    }

    /// Whether the edge `a b` may flip, and the triangles and far corners.
    /// The new diagonal must lie within `tol` of the old one.
    fn flip_of(&self, a: u32, b: u32, min_cos: f64, tol: f64) -> Option<([u32; 2], u32, u32)> {
        if self.locked.contains(&key(a, b)) || self.sharp.contains(&key(a, b)) {
            return None;
        }
        let on = self.on_edge(a, b);
        if on.len() != 2 {
            return None;
        }
        // t1 runs a -> b -> c, t2 runs b -> a -> d.
        let run = |t: u32, x: u32, y: u32| -> Option<u32> {
            let tv = self.tris[t as usize];
            (0..3)
                .find(|&k| tv[k] == x && tv[(k + 1) % 3] == y)
                .map(|k| tv[(k + 2) % 3])
        };
        let (t1, t2, c, d) = match (run(on[0], a, b), run(on[1], b, a)) {
            (Some(c), Some(d)) => (on[0], on[1], c, d),
            _ => match (run(on[1], a, b), run(on[0], b, a)) {
                (Some(c), Some(d)) => (on[1], on[0], c, d),
                _ => return None,
            },
        };
        if c == d || self.neighbours(c).contains(&d) {
            return None;
        }
        let old = [self.tri_normal([a, b, c]), self.tri_normal([b, a, d])];
        let mean = unit_or_z([
            old[0][0] + old[1][0],
            old[0][1] + old[1][1],
            old[0][2] + old[1][2],
        ]);
        for t in [[c, a, d], [d, b, c]] {
            let n = self.tri_normal(t);
            let l = dot(n, n).sqrt();
            if !(l > 0.0)
                || dot(n, mean) < min_cos * l
                || (tol.is_finite() && !self.refs.fits(t.map(|v| self.pt(v))))
            {
                return None;
            }
        }
        if tol.is_finite() {
            // The distance between the lines of the two diagonals.
            let (pa, pb, pc, pd) = (self.pt(a), self.pt(b), self.pt(c), self.pt(d));
            let w = normal([0.0; 3], sub(pb, pa), sub(pd, pc));
            let l = dot(w, w).sqrt();
            if l > 0.0 && dot(sub(pc, pa), w).abs() / l > tol {
                return None;
            }
        }
        Some(([t1, t2], c, d))
    }

    fn flip(&mut self, a: u32, b: u32, ts: [u32; 2], c: u32, d: u32) {
        self.kill(ts[0]);
        self.kill(ts[1]);
        self.add([c, a, d]);
        self.add([d, b, c]);
    }

    /// Takes the PLC points of the outline and of the edges inside that are
    /// no samples out, each collapsed along a locked edge into a neighbour;
    /// where neither way works, the edges at it flip away first.
    fn straighten(&mut self) -> Result<(), &'static str> {
        for _ in 0..16 {
            let doomed: Vec<u32> = (0..self.p.len() as u32)
                .filter(|&v| {
                    self.slot[v as usize].is_none()
                        && !self.star[v as usize].is_empty()
                        && self.on_outline(v)
                })
                .collect();
            if doomed.is_empty() {
                let mut rim = vec![false; self.p.len()];
                for &(a, b) in &self.locked {
                    rim[a as usize] = true;
                    rim[b as usize] = true;
                }
                self.rim = rim;
                return Ok(());
            }
            let mut stuck = Vec::new();
            for v in doomed {
                if self.star[v as usize].is_empty() {
                    continue;
                }
                let along: Vec<u32> = self
                    .neighbours(v)
                    .into_iter()
                    .filter(|&w| self.locked.contains(&key(v, w)))
                    .collect();
                match along
                    .iter()
                    .copied()
                    .find(|&w| self.collapse_ok(v, w, OUTLINE_COS, f64::INFINITY, f64::INFINITY))
                {
                    Some(w) => self.collapse(v, w),
                    None => stuck.push(v),
                }
            }
            // Around a point that would not go, its free edges flip away
            // where they can, so fewer triangles hang on it.
            for v in stuck {
                for w in self.neighbours(v) {
                    if let Some((ts, c, d)) = self.flip_of(v, w, OUTLINE_COS, f64::INFINITY) {
                        self.flip(v, w, ts, c, d);
                    }
                }
            }
        }
        Err("an outline point that would not collapse")
    }

    /// Whether a fixed point lies within `r` of `p`.
    fn near_fixed(&self, p: P3, r: f64, near: &Points) -> bool {
        let rings = (r / near.cell()).ceil() as i64;
        near.around(near.key(p), rings).any(|&v| {
            let v = v as usize;
            self.slot[v].is_some() && !self.star[v].is_empty() && dist(self.p[v], p) <= r
        })
    }

    /// Puts `p` into the triangle nearest it (onto the facets), or into its
    /// edge where it lies close to one; where the point went, or none.
    fn insert(&mut self, p: P3, slot: Slot, near: &mut Points) -> Option<P3> {
        let mut cand: Vec<u32> = near
            .around(near.key(p), 2)
            .flat_map(|&v| self.star[v as usize].iter().copied())
            .collect();
        cand.sort_unstable();
        cand.dedup();
        let best = cand
            .into_iter()
            .map(|t| t as usize)
            .filter(|&t| self.alive[t])
            .map(|t| {
                let tv = self.tris[t].map(|v| self.pt(v));
                (dist(p, closest_on_tri(p, tv[0], tv[1], tv[2])), t)
            })
            .min_by(|x, y| x.0.total_cmp(&y.0));
        let (_, t) = best?;
        let tv = self.tris[t];
        let q = self.refs.project(p, unit_or_z(self.tri_normal(tv)));
        // Near an edge (within a fifth of the triangle's height over it),
        // the edge splits there.
        let old = self.tri_normal(tv);
        let area2 = dot(old, old).sqrt();
        for k in 0..3 {
            let (a, b) = (tv[k], tv[(k + 1) % 3]);
            let n = normal(self.pt(a), self.pt(b), q);
            let height = 2.0 * area2 / (2.0 * dist(self.pt(a), self.pt(b))).max(1e-300);
            let off = dot(n, old) / area2.max(1e-300) / dist(self.pt(a), self.pt(b)).max(1e-300);
            if off < 0.2 * height {
                if self.locked.contains(&key(a, b)) || self.on_edge(a, b).len() != 2 {
                    return None;
                }
                let (pa, pb) = (self.pt(a), self.pt(b));
                let d = sub(pb, pa);
                let s = (dot(sub(q, pa), d) / dot(d, d).max(1e-300)).clamp(0.1, 0.9);
                let at: P3 = std::array::from_fn(|i| pa[i] + s * d[i]);
                let at = self.refs.project(at, unit_or_z(old));
                // Each triangle on the edge splits in two that keep its
                // side and lie on the facets.
                let fits = self.on_edge(a, b).iter().all(|&t| {
                    let tv = self.tris[t as usize];
                    let n = self.tri_normal(tv);
                    (0..3).all(|k| {
                        let (x, y) = (tv[k], tv[(k + 1) % 3]);
                        if key(x, y) != key(a, b) {
                            return true;
                        }
                        let z = tv[(k + 2) % 3];
                        [[self.pt(x), at, self.pt(z)], [at, self.pt(y), self.pt(z)]]
                            .iter()
                            .all(|q| dot(normal(q[0], q[1], q[2]), n) > 0.0 && self.refs.fits(*q))
                    })
                });
                if !fits {
                    return None;
                }
                let m = self.split(a, b, at, Some(slot));
                near.insert(at, m);
                return Some(at);
            }
        }
        let fits = (0..3).all(|k| {
            let q3 = [self.pt(tv[k]), self.pt(tv[(k + 1) % 3]), q];
            self.refs.fits(q3)
        });
        if !fits {
            return None;
        }
        let m = self.add_point(q, Some(slot));
        near.insert(q, m);
        self.kill(t as u32);
        for k in 0..3 {
            self.add([tv[k], tv[(k + 1) % 3], m]);
        }
        Some(q)
    }

    /// The triangles with an angle below `deg`: the longest edge of each
    /// flipped, else its shortest collapsed, where the shape stays within
    /// the deviation (a crease may flip or collapse too, then).
    fn clean(&mut self, size: &dyn Fn(P3) -> f64, deg: f64) {
        let cos_min = deg.to_radians().cos();
        for _ in 0..4 {
            let mut any = false;
            for t in 0..self.tris.len() {
                if !self.alive[t] {
                    continue;
                }
                let tv = self.tris[t];
                let q = tv.map(|v| self.pt(v));
                let thin = (0..3).any(|k| {
                    let (u, w) = (sub(q[(k + 1) % 3], q[k]), sub(q[(k + 2) % 3], q[k]));
                    dot(u, w) > cos_min * (dot(u, u) * dot(w, w)).sqrt()
                });
                if !thin {
                    continue;
                }
                let mut edges: Vec<(f64, u32, u32)> = (0..3)
                    .map(|k| (dist(q[k], q[(k + 1) % 3]), tv[k], tv[(k + 1) % 3]))
                    .collect();
                edges.sort_by(|x, y| x.0.total_cmp(&y.0));
                let h = size(q[0]);
                let (_, a, b) = edges[2];
                let crease = self.sharp.remove(&key(a, b));
                let flip = self.flip_of(a, b, KEEP_COS, DEVIATION * h);
                if crease {
                    self.sharp.insert(key(a, b));
                }
                // Only a flip that lifts the smaller angle of the pair: no
                // flip back and forth where every point is fixed.
                let flip = flip.filter(|&(_, c, d)| {
                    let before = self.min_angle([a, b, c]).min(self.min_angle([b, a, d]));
                    let after = self.min_angle([c, a, d]).min(self.min_angle([d, b, c]));
                    after > before + 1e-9
                });
                if let Some((ts, c, d)) = flip {
                    self.sharp.remove(&key(a, b));
                    self.flip(a, b, ts, c, d);
                    any = true;
                    continue;
                }
                let (_, a, b) = edges[0];
                for (x, y) in [(a, b), (b, a)] {
                    if self.mobility(x) != Mobility::Pinned
                        && self.collapse_ok(x, y, KEEP_COS, 2.0 * h, DEVIATION * h)
                    {
                        self.collapse(x, y);
                        any = true;
                        break;
                    }
                }
            }
            if !any {
                break;
            }
        }
    }

    /// The smallest angle of triangle `t` (degrees).
    fn min_angle(&self, t: [u32; 3]) -> f64 {
        crate::simplex::tri_min_angle(t.map(|v| self.pt(v)))
    }

    /// The edges of the living triangles, each once.
    fn edges(&self) -> Vec<(u32, u32)> {
        let mut out: Vec<(u32, u32)> = self
            .tris
            .iter()
            .zip(&self.alive)
            .filter(|(_, &a)| a)
            .flat_map(|(t, _)| (0..3).map(move |k| key(t[k], t[(k + 1) % 3])))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn split_long(&mut self, size: &dyn Fn(P3) -> f64) {
        for (a, b) in self.edges() {
            if self.locked.contains(&(a, b)) {
                continue;
            }
            let (pa, pb) = (self.pt(a), self.pt(b));
            let mid: P3 = std::array::from_fn(|k| 0.5 * (pa[k] + pb[k]));
            if dist(pa, pb) <= 4.0 / 3.0 * size(mid) {
                continue;
            }
            let on = self.on_edge(a, b);
            if on.is_empty() {
                continue;
            }
            let mut n = [0.0; 3];
            for &t in &on {
                let m = self.tri_normal(self.tris[t as usize]);
                for k in 0..3 {
                    n[k] += m[k];
                }
            }
            // A crease splits on its chord: the crease runs there.
            let q = if self.sharp.contains(&(a, b)) {
                mid
            } else {
                self.refs.project(mid, unit_or_z(n))
            };
            self.split(a, b, q, None);
        }
    }

    fn collapse_short(&mut self, size: &dyn Fn(P3) -> f64) {
        for (a, b) in self.edges() {
            if self.locked.contains(&(a, b)) || self.star[a as usize].is_empty() {
                continue;
            }
            if self.star[b as usize].is_empty() {
                continue;
            }
            let (pa, pb) = (self.pt(a), self.pt(b));
            let mid: P3 = std::array::from_fn(|k| 0.5 * (pa[k] + pb[k]));
            let h = size(mid);
            if dist(pa, pb) >= 0.8 * h {
                continue;
            }
            // A free point goes into the other, a point on a crease into
            // its neighbour along it; one on the outline, fixed or at a
            // corner of creases stays.
            let crease = self.sharp.contains(&(a, b));
            let goes = |v: u32| match self.mobility(v) {
                Mobility::Free => !crease,
                Mobility::Along => crease,
                Mobility::Pinned => false,
            };
            let way = if goes(a) {
                Some((a, b))
            } else if goes(b) {
                Some((b, a))
            } else {
                None
            };
            let Some((x, y)) = way else { continue };
            if self.collapse_ok(x, y, KEEP_COS, 4.0 / 3.0 * h, DEVIATION * h) {
                self.collapse(x, y);
            }
        }
    }

    fn flip_valence(&mut self, size: &dyn Fn(P3) -> f64) {
        let valence = |m: &Mesh, v: u32| m.neighbours(v).len() as i64;
        let target = |m: &Mesh, v: u32| if m.on_outline(v) { 4 } else { 6 };
        for (a, b) in self.edges() {
            let h = size(self.pt(a));
            let Some((ts, c, d)) = self.flip_of(a, b, KEEP_COS, DEVIATION * h) else {
                continue;
            };
            let dev = |v: u32, change: i64| (valence(self, v) + change - target(self, v)).pow(2);
            let before = dev(a, 0) + dev(b, 0) + dev(c, 0) + dev(d, 0);
            let after = dev(a, -1) + dev(b, -1) + dev(c, 1) + dev(d, 1);
            if after < before {
                self.flip(a, b, ts, c, d);
            }
        }
    }

    fn smooth(&mut self) {
        for v in 0..self.p.len() as u32 {
            if self.star[v as usize].is_empty() || self.mobility(v) != Mobility::Free {
                continue;
            }
            // The area-weighted centroid of the triangles around, moved in
            // the tangent plane.
            let (mut c, mut w) = ([0.0; 3], 0.0);
            for &t in &self.star[v as usize] {
                let tv = self.tris[t as usize].map(|x| self.pt(x));
                let a = dot(normal(tv[0], tv[1], tv[2]), normal(tv[0], tv[1], tv[2])).sqrt();
                for k in 0..3 {
                    c[k] += a * (tv[0][k] + tv[1][k] + tv[2][k]) / 3.0;
                }
                w += a;
            }
            if !(w > 0.0) {
                continue;
            }
            let p = self.pt(v);
            let n = self.vertex_normal(v);
            let d: P3 = std::array::from_fn(|k| c[k] / w - p[k]);
            let dn = dot(d, n);
            let step: P3 = std::array::from_fn(|k| d[k] - dn * n[k]);
            let olds: Vec<(u32, P3)> = self.star[v as usize]
                .iter()
                .map(|&t| (t, self.tri_normal(self.tris[t as usize])))
                .collect();
            for f in [1.0, 0.5, 0.25] {
                let q = self
                    .refs
                    .project(std::array::from_fn(|k| p[k] + f * step[k]), n);
                self.p[v as usize] = q;
                let ok = olds.iter().all(|&(t, old)| {
                    let tv = self.tris[t as usize];
                    let new = self.tri_normal(tv);
                    let (lo, ln) = (dot(old, old).sqrt(), dot(new, new).sqrt());
                    ln > 1e-12 * lo
                        && dot(old, new) >= KEEP_COS * lo * ln
                        && self.refs.fits(tv.map(|v| self.pt(v)))
                });
                if ok {
                    break;
                }
                self.p[v as usize] = p;
            }
        }
    }

    /// Flips each free edge whose opposite angles sum past a half turn, so
    /// the triangles are Delaunay on the face where the flips keep it.
    fn flip_delaunay(&mut self, size: &dyn Fn(P3) -> f64) {
        for _ in 0..8 {
            let mut any = false;
            for (a, b) in self.edges() {
                let h = size(self.pt(a));
                let Some((ts, c, d)) = self.flip_of(a, b, 0.9, DEVIATION * h) else {
                    continue;
                };
                let angle = |x: u32, y: u32, z: u32| {
                    let (u, w) = (sub(self.pt(y), self.pt(x)), sub(self.pt(z), self.pt(x)));
                    (dot(u, w) / (dot(u, u) * dot(w, w)).sqrt().max(1e-300))
                        .clamp(-1.0, 1.0)
                        .acos()
                };
                if angle(c, a, b) + angle(d, a, b) > std::f64::consts::PI + 1e-9 {
                    self.flip(a, b, ts, c, d);
                    any = true;
                }
            }
            if !any {
                break;
            }
        }
    }

    /// The mesh as slots, own points (`own` first, then the free points)
    /// and triangles over the points in use.
    fn out(&self, mut own: Vec<P3>) -> Remeshed {
        let mut local = vec![usize::MAX; self.p.len()];
        let mut slots = Vec::new();
        let mut tris = Vec::new();
        for (t, &a) in self.tris.iter().zip(&self.alive) {
            if !a {
                continue;
            }
            let lt = t.map(|v| {
                let i = v as usize;
                if local[i] == usize::MAX {
                    local[i] = slots.len();
                    slots.push(match self.slot[i] {
                        Some(s) => s,
                        None => {
                            own.push(self.p[i]);
                            Slot::Own((own.len() - 1) as u32)
                        }
                    });
                }
                local[i]
            });
            tris.push(lt);
        }
        Remeshed { slots, own, tris }
    }
}

fn normal(a: P3, b: P3, c: P3) -> P3 {
    let (u, v) = (sub(b, a), sub(c, a));
    cross(u, v)
}

/// The unit vector along `a`, +z for a zero one.
fn unit_or_z(a: P3) -> P3 {
    rapidmesh_geom::vec3::unit(a).unwrap_or([0.0, 0.0, 1.0])
}

fn seg_dist(p: P3, a: P3, b: P3) -> f64 {
    let d = sub(b, a);
    let t = (dot(sub(p, a), d) / dot(d, d).max(1e-300)).clamp(0.0, 1.0);
    dist(p, std::array::from_fn(|k| a[k] + t * d[k]))
}
