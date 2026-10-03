//! Exact constrained triangulation of a single facet.
//!
//! Given a facet (input triangle), a set of points and a set of constraint
//! segments on it (all exact, possibly implicit), produces a triangulation of
//! the facet whose vertex set is the input set plus all constraint-constraint
//! crossing points, and whose edge set contains every (sub-divided) constraint
//! segment. Everything runs on exact predicates over the facet's 2D
//! projection; no coordinate is ever rounded.
//!
//! Algorithm: classic incremental insertion (interior 1→3 split, on-edge
//! 2→4 split) followed by flip-based constraint edge recovery. Constraints
//! are pre-split at their mutual crossing points (constructed exactly from
//! constraint provenance) and at every vertex lying on them, so recovery only
//! ever handles segments with empty interiors — the regime where flip
//! recovery provably terminates. The triangulation is constrained, not
//! Delaunay; a CDT upgrade (indirect incircle) can be layered on later
//! without changing this module's contract.

use crate::constraint::Constraint;
use crate::tri::Tri;
use rapidmesh_exact::proj::{self, Projected};
use rapidmesh_exact::{cmp_along, collinear, lex_cmp, strictly_between, Axis, Point3, Sign};

/// Result of triangulating a facet.
#[derive(Debug)]
pub struct FacetTriangulation {
    /// Vertex pool. Indices 0..3 are the facet corners.
    pub vertices: Vec<Point3>,
    /// Sub-triangles as vertex indices, all oriented like the facet
    /// (their 2D orientation in `axis` equals `orientation`).
    pub triangles: Vec<[usize; 3]>,
    /// The projection axis used.
    pub axis: Axis,
    /// The facet's 2D orientation in that projection.
    pub orientation: Sign,
}

impl FacetTriangulation {
    /// True if the (undirected) edge {u, v} is present.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.triangles.iter().any(|t| {
            (0..3).any(|e| {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                (a == u && b == v) || (a == v && b == u)
            })
        })
    }
}

/// The vertex pool of a facet triangulation: every point also prepared for
/// the 2D predicates of the facet's projection.
struct Pool {
    axis: Axis,
    points: Vec<Point3>,
    prep: Vec<Projected>,
    /// The constraint lines each point lies on by construction (a constraint
    /// endpoint on its line, a crossing point on both): three points sharing
    /// one are collinear with no arithmetic.
    lines: Vec<Vec<u32>>,
}

impl Pool {
    fn new(axis: Axis, seed: Vec<Point3>) -> Pool {
        let prep = seed
            .iter()
            .map(|p| Projected::new(p.clone(), axis))
            .collect();
        let lines = vec![Vec::new(); seed.len()];
        Pool {
            axis,
            points: seed,
            prep,
            lines,
        }
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    /// Adds `p` unless an exactly coincident point exists; returns its index.
    fn add(&mut self, p: Point3) -> usize {
        // The points of a facet share its plane, so apart projections rule
        // coincidence out without the exact test.
        let pp = Projected::new(p.clone(), self.axis);
        let same = |i: usize| {
            self.points[i] == p || (self.prep[i].may_coincide(&pp) && self.points[i].coincides(&p))
        };
        if let Some(i) = (0..self.points.len()).find(|&i| same(i)) {
            return i;
        }
        self.prep.push(pp);
        self.lines.push(Vec::new());
        self.points.push(p);
        self.points.len() - 1
    }

    /// Records that point `i` lies on constraint line `line`.
    fn on_line(&mut self, i: usize, line: u32) {
        if !self.lines[i].contains(&line) {
            self.lines[i].push(line);
        }
    }

    /// True if `a`, `b` and `c` lie on one constraint line by construction.
    fn same_line(&self, a: usize, b: usize, c: usize) -> bool {
        self.lines[a]
            .iter()
            .any(|l| self.lines[b].contains(l) && self.lines[c].contains(l))
    }

    fn orient(&self, a: usize, b: usize, c: usize) -> Sign {
        // A repeated point, or three points on one constraint line:
        // degenerate by construction, no arithmetic (an implicit point's
        // zero would take the full exact stage).
        if a == b || b == c || a == c || self.same_line(a, b, c) {
            return Sign::Zero;
        }
        proj::orient2d(&self.prep[a], &self.prep[b], &self.prep[c])
            .expect("all triangulation points are valid")
    }

    fn incircle(&self, a: usize, b: usize, c: usize, d: usize) -> Sign {
        proj::incircle2d(&self.prep[a], &self.prep[b], &self.prep[c], &self.prep[d])
            .expect("all triangulation points are valid")
    }

    /// [`Pool::incircle`] where the filters certify it, else `None`.
    fn incircle_filtered(&self, a: usize, b: usize, c: usize, d: usize) -> Option<Sign> {
        proj::incircle2d_filtered(&self.prep[a], &self.prep[b], &self.prep[c], &self.prep[d])
    }
}

/// Exact constrained triangulation of `facet` with the given points and
/// constraint segments. All points and constraint endpoints must lie on the
/// (closed) facet; constraint crossing points are constructed exactly from
/// constraint provenance. An input that breaks this (a point off the facet,
/// a constraint leaving it) is an error that says what failed.
pub fn triangulate_facet(
    facet: &Tri,
    points: &[Point3],
    constraints: &[Constraint],
) -> Result<FacetTriangulation, String> {
    let (axis, orientation) = facet.projection_axis();
    let seed_pool: Vec<Point3> = (0..3).map(|i| facet.point(i)).collect();
    let seed_tris = vec![[0usize, 1, 2]];
    triangulate_seeded(
        axis,
        orientation,
        facet.v,
        seed_pool,
        seed_tris,
        points,
        constraints,
        true,
    )
}

/// Exact constrained triangulation of a planar facet given a SEED triangulation
/// (a valid tiling of the facet by triangles, vertices `seed_pool`, faces
/// `seed_tris`), the facet's projection (`axis`, `orientation`), three
/// non-collinear plane points `plane3` (for `PlaneCut` TPI provenance), and the
/// points/constraints dropped on it. Generalizes [`triangulate_facet`] from a
/// single input triangle to an arbitrary planar polygon (with holes): the seed
/// boundary edges (loops, including holes) have no opposite triangle and are
/// preserved, while seed-internal edges are flipped toward the constrained
/// Delaunay triangulation — so a fan/ear seed leaves no artificial interior
/// structure behind. `canonical` makes that the exact constrained Delaunay
/// triangulation, a pure function of the geometry, which a facet with a
/// coincident coplanar partner needs (the overlap must triangulate the same
/// from both); otherwise near-cocircular cases the filters cannot decide
/// keep their diagonal, sparing the exact incircle stage.
#[allow(clippy::too_many_arguments)]
pub fn triangulate_seeded(
    axis: Axis,
    orientation: Sign,
    plane3: [[f64; 3]; 3],
    seed_pool: Vec<Point3>,
    seed_tris: Vec<[usize; 3]>,
    points: &[Point3],
    constraints: &[Constraint],
    canonical: bool,
) -> Result<FacetTriangulation, String> {
    // ------------------------------------------------------ vertex pool
    let mut pool = Pool::new(axis, seed_pool);
    let seed_len = pool.len();
    for p in points {
        pool.add(p.clone());
    }
    // The supporting line of each constraint, one id per distinct provenance
    // (keyed by the exact coordinates that define it).
    let mut line_ids: rustc_hash::FxHashMap<Vec<u64>, u32> = Default::default();
    let line_of: Vec<u32> = constraints
        .iter()
        .map(|c| {
            let key: Vec<u64> = match &c.line {
                crate::constraint::ConstraintLine::PlaneCut(t) => {
                    t.iter().flatten().map(|x| x.to_bits()).collect()
                }
                crate::constraint::ConstraintLine::Edge(u, v) => std::iter::once(u64::MAX)
                    .chain(u.iter().chain(v).map(|x| x.to_bits()))
                    .collect(),
            };
            let n = line_ids.len() as u32;
            *line_ids.entry(key).or_insert(n)
        })
        .collect();
    // Pool indices of every constraint's endpoints, on their line.
    let ends: Vec<(usize, usize)> = constraints
        .iter()
        .zip(&line_of)
        .map(|(c, &l)| {
            let (a, b) = (pool.add(c.a.clone()), pool.add(c.b.clone()));
            pool.on_line(a, l);
            pool.on_line(b, l);
            (a, b)
        })
        .collect();
    let constraint_ids: Vec<(usize, usize, u32)> = ends
        .iter()
        .zip(&line_of)
        .filter(|((a, b), _)| a != b)
        .map(|(&(a, b), &l)| (a, b, l))
        .collect();

    // Pre-split: exact crossing points of strictly crossing constraint pairs.
    for (i, ci) in constraints.iter().enumerate() {
        for (j, cj) in constraints.iter().enumerate().skip(i + 1) {
            // Constraints on one line never cross strictly.
            if line_of[i] == line_of[j] {
                continue;
            }
            let ((ia, ib), (ja, jb)) = (ends[i], ends[j]);
            let si_a = pool.orient(ia, ib, ja);
            let si_b = pool.orient(ia, ib, jb);
            if si_a.combine(si_b) != Sign::Negative {
                continue;
            }
            let sj_a = pool.orient(ja, jb, ia);
            let sj_b = pool.orient(ja, jb, ib);
            if sj_a.combine(sj_b) != Sign::Negative {
                continue;
            }
            let x = ci
                .line_intersection(cj, plane3)
                .expect("strictly crossing constraints have intersecting lines");
            debug_assert!(x.is_valid());
            let k = pool.add(x);
            pool.on_line(k, line_of[i]);
            pool.on_line(k, line_of[j]);
        }
    }

    // ------------------------------------------------- point insertion
    let mut tris = Tris::new(seed_tris);
    for k in seed_len..pool.len() {
        insert_vertex(&mut tris, &pool, orientation, k)?;
    }

    // -------------------------------------------- constraint recovery
    // Cached f64 positions for the segment bounding-box prefilter below.
    let approx: Vec<[f64; 3]> = pool
        .points
        .iter()
        .map(|p| p.approx().expect("valid"))
        .collect();
    let mut chain_edges: Vec<(usize, usize)> = Vec::new();
    for &(ia, ib, line) in &constraint_ids {
        // Padded segment bounding box (f64): a vertex strictly on the
        // segment lies within it (its approx is within an ulp of the exact
        // position; the pad is a million times that). On boolean scenes most
        // pool vertices belong to OTHER constraints far from this one, so the
        // cheap box test rejects them before the exact collinear/between
        // predicates run -- the difference between O(C*P) exact tests and a
        // handful per segment.
        let (pa, pb) = (approx[ia], approx[ib]);
        let seg_len = (0..3).map(|k| (pa[k] - pb[k]).powi(2)).sum::<f64>().sqrt();
        let pad = 1e-9 * seg_len.max(f64::MIN_POSITIVE);
        let slo: [f64; 3] = std::array::from_fn(|k| pa[k].min(pb[k]) - pad);
        let shi: [f64; 3] = std::array::from_fn(|k| pa[k].max(pb[k]) + pad);
        // All pool vertices strictly inside the segment split it into a chain.
        let mut on_seg: Vec<usize> = (0..pool.len())
            .filter(|&k| {
                k != ia
                    && k != ib
                    && (0..3).all(|d| approx[k][d] >= slo[d] && approx[k][d] <= shi[d])
                    && (pool.lines[k].contains(&line)
                        || collinear(&pool.points[ia], &pool.points[ib], &pool.points[k])
                            .expect("valid"))
                    && strictly_between(&pool.points[ia], &pool.points[ib], &pool.points[k])
                        .expect("valid")
            })
            .collect();
        on_seg.sort_by(|&p, &q| {
            let pts = &pool.points;
            match cmp_along(&pts[ia], &pts[ib], &pts[p], &pts[q]).expect("valid") {
                Sign::Negative => std::cmp::Ordering::Greater,
                Sign::Zero => std::cmp::Ordering::Equal,
                Sign::Positive => std::cmp::Ordering::Less,
            }
        });
        let mut prev = ia;
        for &k in on_seg.iter().chain(std::iter::once(&ib)) {
            chain_edges.push((prev, k));
            prev = k;
        }
    }
    for &(u, v) in &chain_edges {
        recover_edge(&mut tris, &pool, u, v)?;
    }
    // Every chain edge must now be present (recovery of one constraint can
    // never flip away another: constraints are non-crossing after pre-split).
    for &(u, v) in &chain_edges {
        if tris.edge(u, v).is_none() && tris.edge(v, u).is_none() {
            return Err(format!("constraint edge {u}-{v} is missing after recovery"));
        }
    }

    // Canonical (constrained Delaunay) pass: makes the triangulation a pure
    // function of the geometry, so coincident coplanar facets of different
    // inputs triangulate their overlap identically and can be matched
    // triangle-by-triangle downstream.
    let constrained: rustc_hash::FxHashSet<(usize, usize)> = chain_edges
        .iter()
        .map(|&(u, v)| (u.min(v), u.max(v)))
        .collect();
    delaunay_pass(&mut tris, &pool, orientation, &constrained, canonical)?;

    Ok(FacetTriangulation {
        vertices: pool.points,
        triangles: tris.t,
        axis,
        orientation,
    })
}

/// No neighbour (a boundary edge of the seed) or no triangle.
const NO: usize = usize::MAX;

/// The triangles of a facet with their neighbours: `nb[t][e]` is the
/// triangle across edge `e` of `t` (from `t[e]` to `t[e + 1]`), where the
/// same edge runs the other way. `vt[v]` is some triangle containing `v`.
/// Triangles are only ever rewritten in place or appended, never removed.
struct Tris {
    t: Vec<[usize; 3]>,
    nb: Vec<[usize; 3]>,
    vt: Vec<usize>,
    /// Walk start of the next point location.
    last: usize,
}

impl Tris {
    fn new(seed: Vec<[usize; 3]>) -> Tris {
        let mut nb = vec![[NO; 3]; seed.len()];
        let mut open: rustc_hash::FxHashMap<(usize, usize), (usize, usize)> =
            rustc_hash::FxHashMap::default();
        for (ti, t) in seed.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                match open.remove(&(b, a)) {
                    Some((u, f)) => {
                        nb[ti][e] = u;
                        nb[u][f] = ti;
                    }
                    None => {
                        open.insert((a, b), (ti, e));
                    }
                }
            }
        }
        let mut tris = Tris {
            t: Vec::with_capacity(seed.len()),
            nb: Vec::new(),
            vt: Vec::new(),
            last: 0,
        };
        for t in seed {
            let ti = tris.t.len();
            tris.t.push(t);
            tris.touch(ti);
        }
        tris.nb = nb;
        tris
    }

    fn len(&self) -> usize {
        self.t.len()
    }

    /// Points the hints of the vertices of `ti` at it.
    fn touch(&mut self, ti: usize) {
        for v in self.t[ti] {
            if v >= self.vt.len() {
                self.vt.resize(v + 1, NO);
            }
            self.vt[v] = ti;
        }
    }

    fn put(&mut self, ti: usize, tri: [usize; 3], nb: [usize; 3]) {
        if ti == self.t.len() {
            self.t.push(tri);
            self.nb.push(nb);
        } else {
            self.t[ti] = tri;
            self.nb[ti] = nb;
        }
        self.touch(ti);
    }

    /// Makes the triangle across edge `e` of `ti` point back at `ti`.
    fn backlink(&mut self, ti: usize, e: usize) {
        let n = self.nb[ti][e];
        if n == NO {
            return;
        }
        let (a, b) = (self.t[ti][e], self.t[ti][(e + 1) % 3]);
        let f = (0..3)
            .find(|&f| self.t[n][f] == b && self.t[n][(f + 1) % 3] == a)
            .expect("neighbours share their edge");
        self.nb[n][f] = ti;
    }

    fn slot(&self, ti: usize, v: usize) -> usize {
        (0..3)
            .find(|&i| self.t[ti][i] == v)
            .expect("vertex of the triangle")
    }

    /// Every triangle around vertex `x`, by rotation through the neighbours
    /// (both ways where the star is cut by the boundary).
    fn star(&self, x: usize) -> Vec<usize> {
        let s = self.vt.get(x).copied().unwrap_or(NO);
        if s == NO {
            return Vec::new();
        }
        let mut out = vec![s];
        // One way: across the edge entering x.
        let mut ti = s;
        loop {
            let i = self.slot(ti, x);
            let n = self.nb[ti][(i + 2) % 3];
            if n == NO {
                break;
            }
            if n == s {
                return out;
            }
            out.push(n);
            ti = n;
        }
        // The boundary cut the star: the other way from the start.
        let mut ti = s;
        loop {
            let i = self.slot(ti, x);
            let n = self.nb[ti][i];
            if n == NO {
                break;
            }
            out.push(n);
            ti = n;
        }
        out
    }

    /// The triangle holding the directed edge `x -> y`, with the edge index.
    fn edge(&self, x: usize, y: usize) -> Option<(usize, usize)> {
        self.star(x).into_iter().find_map(|ti| {
            let i = self.slot(ti, x);
            (self.t[ti][(i + 1) % 3] == y).then_some((ti, i))
        })
    }

    /// 1 -> 3 split of `ti` at `k`.
    fn split_face(&mut self, ti: usize, k: usize) {
        let [i, j, l] = self.t[ti];
        let [n0, n1, n2] = self.nb[ti];
        let (t1, t2) = (self.len(), self.len() + 1);
        self.put(ti, [i, j, k], [n0, t1, t2]);
        self.put(t1, [j, l, k], [n1, t2, ti]);
        self.put(t2, [l, i, k], [n2, ti, t1]);
        self.backlink(t1, 0);
        self.backlink(t2, 0);
    }

    /// Splits edge `e` of `ti` (and of the triangle across it) at `k`.
    fn split_edge(&mut self, ti: usize, e: usize, k: usize) {
        let (x, y, c) = (
            self.t[ti][e],
            self.t[ti][(e + 1) % 3],
            self.t[ti][(e + 2) % 3],
        );
        let (a_yc, a_cx) = (self.nb[ti][(e + 1) % 3], self.nb[ti][(e + 2) % 3]);
        let tj = self.nb[ti][e];
        let ta = self.len();
        if tj == NO {
            self.put(ti, [x, k, c], [NO, ta, a_cx]);
            self.put(ta, [k, y, c], [NO, a_yc, ti]);
            self.backlink(ta, 1);
            return;
        }
        let f = (0..3)
            .find(|&f| self.t[tj][f] == y && self.t[tj][(f + 1) % 3] == x)
            .expect("neighbours share their edge");
        let d = self.t[tj][(f + 2) % 3];
        let (b_xd, b_dy) = (self.nb[tj][(f + 1) % 3], self.nb[tj][(f + 2) % 3]);
        let tb = ta + 1;
        self.put(ti, [x, k, c], [tb, ta, a_cx]);
        self.put(ta, [k, y, c], [tj, a_yc, ti]);
        self.put(tj, [y, k, d], [ta, tb, b_dy]);
        self.put(tb, [k, x, d], [ti, b_xd, tj]);
        self.backlink(ta, 1);
        self.backlink(tb, 1);
    }

    /// Flips the edge `x -> y` of `li` (opposite `c`) with the triangle
    /// `ri` across it (opposite `d`): `li` becomes `(c, x, d)`, `ri`
    /// becomes `(d, y, c)`.
    fn flip(&mut self, li: usize, ri: usize, x: usize, y: usize) {
        let e = (0..3)
            .find(|&e| self.t[li][e] == x && self.t[li][(e + 1) % 3] == y)
            .expect("edge of the left triangle");
        let f = (0..3)
            .find(|&f| self.t[ri][f] == y && self.t[ri][(f + 1) % 3] == x)
            .expect("edge of the right triangle");
        let (c, d) = (self.t[li][(e + 2) % 3], self.t[ri][(f + 2) % 3]);
        let (l_yc, l_cx) = (self.nb[li][(e + 1) % 3], self.nb[li][(e + 2) % 3]);
        let (r_xd, r_dy) = (self.nb[ri][(f + 1) % 3], self.nb[ri][(f + 2) % 3]);
        self.put(li, [c, x, d], [l_cx, r_xd, ri]);
        self.put(ri, [d, y, c], [r_dy, l_yc, li]);
        self.backlink(li, 1);
        self.backlink(ri, 1);
    }
}

/// Inserts pool vertex `k` into the triangulation (interior 1→3 split or
/// on-edge 2→4 split), located by a walk. Panics if the vertex lies outside
/// the facet.
fn insert_vertex(tris: &mut Tris, pool: &Pool, orientation: Sign, k: usize) -> Result<(), String> {
    let outside = orientation.flip();
    let located = locate(tris, pool, outside, k);
    let Some((ti, s)) = located else {
        let axis = pool.axis;
        let dump = |i: usize| {
            pool.points[i]
                .approx()
                .map(|p| format!("{p:?}"))
                .unwrap_or_default()
        };
        return Err(format!(
            "vertex {k} lies outside the facet\n  vertex: {} axis {axis:?} orientation {orientation:?}\n  \
             seed corners: {} | {} | {} ({} tris)",
            dump(k), dump(0), dump(1), dump(2), tris.len(),
        ));
    };
    match s.iter().filter(|&&x| x == Sign::Zero).count() {
        0 => tris.split_face(ti, k),
        1 => {
            let e = (0..3).find(|&e| s[e] == Sign::Zero).expect("one zero");
            tris.split_edge(ti, e, k);
        }
        _ => return Err(format!("vertex {k} coincides with a corner")),
    }
    tris.last = ti;
    Ok(())
}

/// The triangle containing `k` (closed), with the signs of `k` against its
/// three edges: a visibility walk from the last insertion, a scan when the
/// walk does not settle.
fn locate(tris: &Tris, pool: &Pool, outside: Sign, k: usize) -> Option<(usize, [Sign; 3])> {
    let signs = |ti: usize| -> [Sign; 3] {
        let t = tris.t[ti];
        std::array::from_fn(|e| pool.orient(t[e], t[(e + 1) % 3], k))
    };
    let mut ti = tris.last.min(tris.len() - 1);
    let mut prev = NO;
    for _ in 0..tris.len() + 8 {
        let s = signs(ti);
        let next = (0..3)
            .filter(|&e| s[e] == outside)
            .map(|e| tris.nb[ti][e])
            .find(|&n| n != prev);
        match next {
            None if s.iter().all(|&x| x != outside) => return Some((ti, s)),
            Some(n) if n != NO => {
                prev = ti;
                ti = n;
            }
            // Out across a boundary edge, or only back the way we came.
            _ => break,
        }
    }
    (0..tris.len()).find_map(|ti| {
        let s = signs(ti);
        s.iter().all(|&x| x != outside).then_some((ti, s))
    })
}

/// Flips non-constrained edges to the constrained Delaunay triangulation,
/// with a deterministic geometric tie-break for cocircular quads (prefer the
/// diagonal containing the lexicographically smallest of the four vertices).
/// The result is unique given the vertex set and constraints — the property
/// that makes coincident facets of different inputs match exactly.
fn delaunay_pass(
    tris: &mut Tris,
    pool: &Pool,
    orientation: Sign,
    constrained: &rustc_hash::FxHashSet<(usize, usize)>,
    canonical: bool,
) -> Result<(), String> {
    let o2d = |a: usize, b: usize, c: usize| pool.orient(a, b, c);
    let mut queue: std::collections::VecDeque<(usize, usize)> = tris
        .t
        .iter()
        .flat_map(|t| (0..3).map(move |e| (t[e], t[(e + 1) % 3])))
        .filter(|&(x, y)| x < y)
        .collect();
    let cap = 1000 + 64 * tris.len() * tris.len();
    let mut steps = 0usize;
    while let Some((x, y)) = queue.pop_front() {
        steps += 1;
        if steps > cap {
            return Err("the Delaunay pass did not converge".into());
        }
        if constrained.contains(&(x.min(y), x.max(y))) {
            continue;
        }
        let (Some((li, e)), Some((ri, f))) = (tris.edge(x, y), tris.edge(y, x)) else {
            continue; // boundary edge or already flipped away
        };
        let c = tris.t[li][(e + 2) % 3];
        let d = tris.t[ri][(f + 2) % 3];
        // In-circle in the triangle's own handedness: (x, y, c) has the
        // facet orientation, so "d inside" carries that sign. Without the
        // canonical form, a case only the exact stage could decide
        // (near-cocircular) keeps its diagonal: both are equally good.
        let s = if canonical {
            pool.incircle(x, y, c, d)
        } else {
            match pool.incircle_filtered(x, y, c, d) {
                Some(s) => s,
                None => continue,
            }
        };
        let flip = if s == orientation {
            true
        } else if s == Sign::Zero {
            // Cocircular: prefer the diagonal owning the lex-smallest vertex.
            let min_of = |a: usize, b: usize| -> usize {
                match lex_cmp(&pool.points[a], &pool.points[b]).expect("valid") {
                    std::cmp::Ordering::Greater => b,
                    _ => a,
                }
            };
            let overall = min_of(min_of(x, y), min_of(c, d));
            overall == c || overall == d
        } else {
            false
        };
        if !flip {
            continue;
        }
        // Quad must be strictly convex (guards collinear/degenerate ties).
        if o2d(c, d, x).combine(o2d(c, d, y)) != Sign::Negative {
            continue;
        }
        tris.flip(li, ri, x, y);
        for &(p, q) in &[(c, x), (x, d), (d, y), (y, c)] {
            queue.push_back((p.min(q), p.max(q)));
        }
    }
    Ok(())
}

/// Restores the edge {u, v} (whose open interior contains no vertices) by
/// flipping edges that cross it — Sloan-style FIFO processing.
///
/// The edges crossing the segment are collected once by walking along it
/// from `u`; an edge whose surrounding quad is not strictly convex is
/// deferred to the back of the queue (some other flip will unlock it), and a
/// flip's new diagonal is re-enqueued only if it still crosses the segment.
/// Always flipping the first flippable edge found by rescanning would
/// instead oscillate: a valid flip's inverse is immediately valid again.
fn recover_edge(tris: &mut Tris, pool: &Pool, u: usize, v: usize) -> Result<(), String> {
    let o2d = |a: usize, b: usize, c: usize| pool.orient(a, b, c);
    let crosses = |x: usize, y: usize| -> bool {
        o2d(u, v, x).combine(o2d(u, v, y)) == Sign::Negative
            && o2d(x, y, u).combine(o2d(x, y, v)) == Sign::Negative
    };
    if tris.edge(u, v).is_some() || tris.edge(v, u).is_some() {
        return Ok(());
    }
    // The first crossed edge: the one opposite `u` in some triangle of its
    // star; then from triangle to triangle across the crossed edges.
    let mut queue: std::collections::VecDeque<(usize, usize)> = std::collections::VecDeque::new();
    let mut cur: Option<(usize, usize, usize)> = tris.star(u).into_iter().find_map(|ti| {
        let i = tris.slot(ti, u);
        let (a, b) = (tris.t[ti][(i + 1) % 3], tris.t[ti][(i + 2) % 3]);
        crosses(a, b).then_some((ti, a, b))
    });
    while let Some((ti, a, b)) = cur.take() {
        queue.push_back((a.min(b), a.max(b)));
        let e = (0..3)
            .find(|&e| tris.t[ti][e] == a && tris.t[ti][(e + 1) % 3] == b)
            .expect("edge of the triangle");
        let n = tris.nb[ti][e];
        if n == NO {
            return Err(format!("constraint {u}-{v} leaves the facet"));
        }
        let w = (0..3)
            .map(|i| tris.t[n][i])
            .find(|&w| w != a && w != b)
            .expect("third vertex");
        if w == v {
            break;
        }
        // Across (b -> a) into `n`, the segment leaves through (a, w) or
        // (w, b); its edges run b -> a -> w -> b.
        cur = if crosses(a, w) {
            Some((n, a, w))
        } else {
            Some((n, w, b))
        };
    }

    // Each deferred edge gets unlocked by some other flip; if a full round
    // of deferrals makes no progress, the input violated the contract.
    let mut deferred_in_a_row = 0;
    while let Some((x, y)) = queue.pop_front() {
        let (Some((li, e)), Some((ri, f))) = (tris.edge(x, y), tris.edge(y, x)) else {
            // The edge was flipped away while queued (as a new diagonal that
            // itself got flipped); nothing to do.
            continue;
        };
        let c = tris.t[li][(e + 2) % 3];
        let d = tris.t[ri][(f + 2) % 3];
        if o2d(c, d, x).combine(o2d(c, d, y)) != Sign::Negative {
            // Quad not strictly convex: defer.
            deferred_in_a_row += 1;
            if deferred_in_a_row > queue.len() {
                return Err(format!("edge recovery stalled for constraint {u}-{v}"));
            }
            queue.push_back((x, y));
            continue;
        }
        deferred_in_a_row = 0;
        tris.flip(li, ri, x, y);
        if crosses(c.min(d), c.max(d)) {
            queue.push_back((c.min(d), c.max(d)));
        }
    }

    if tris.edge(u, v).is_none() && tris.edge(v, u).is_none() {
        return Err(format!("constraint edge {u}-{v} is absent after recovery"));
    }
    Ok(())
}
