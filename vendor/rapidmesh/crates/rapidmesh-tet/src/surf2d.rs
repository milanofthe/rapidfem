//! 2D Delaunay + Lloyd relaxation for the surface (2D) stage of the hierarchy.
//!
//! Built on the EXACT robust predicates (`orient2d`/`incircle2d` from
//! rapidmesh-exact, the Shewchuk-style kernel): all triangulation decisions are
//! exact, so the relaxation is robust on degenerate (cocircular/collinear) grid
//! inputs. Only the centroid WEIGHTS are float (a non-decision quantity).
//!
//! Used by the surface stage: each planar patch is filled by scattering interior
//! points and relaxing them with the patch boundary (1D edge points) held fixed.
//! The triangulation here serves the relaxation; the conforming surface is read
//! back from the 3D mesh downstream.

use rapidmesh_exact::{incircle2d, orient2d, Axis, Point3, Sign};

type P2 = [f64; 2];

fn p3(p: P2) -> Point3 {
    Point3::explicit(p[0], p[1], 0.0)
}

/// Orientation of (a, b, c) in the xy plane (exact).
fn orient(a: P2, b: P2, c: P2) -> Sign {
    orient2d(&p3(a), &p3(b), &p3(c), Axis::Z).expect("explicit points are valid")
}

/// True iff `d` is strictly inside the circumcircle of CCW triangle (a, b, c)
/// (exact; cocircular -> false, a consistent choice yielding a valid mesh).
fn in_circumcircle(a: P2, b: P2, c: P2, d: P2) -> bool {
    incircle2d(&p3(a), &p3(b), &p3(c), &p3(d), Axis::Z) == Some(Sign::Positive)
}

/// The triangle reordered to CCW.
fn ccw(t: [usize; 3], pts: &[P2]) -> [usize; 3] {
    if orient(pts[t[0]], pts[t[1]], pts[t[2]]) == Sign::Negative {
        [t[0], t[2], t[1]]
    } else {
        t
    }
}

fn dist2(a: P2, b: P2) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

const NONE2: usize = usize::MAX;

/// A triangle whose circumradius exceeds this multiple of the target is split
/// for size. The target is the typical edge length, as a mesh size is in gmsh:
/// an equilateral triangle splits from an edge of 1.39 targets on, so the
/// split halves land around it and the median edge comes out near the target
/// (at 0.6, the bound of an edge of one target, it came out at 0.76).
const SIZE_SPLIT_RADIUS: f64 = 0.8;

/// Walks from triangle `start` to the (CCW) triangle containing `p`, stepping
/// across any edge that `p` lies strictly to the right of. Falls back to a
/// linear scan if the walk does not converge (degenerate connectivity).
fn locate2(
    start: usize,
    p: P2,
    tris: &[[usize; 3]],
    nbr: &[[usize; 3]],
    alive: &[bool],
    pts: &[P2],
) -> usize {
    let mut t = start;
    for _ in 0..tris.len() * 2 + 16 {
        let tv = tris[t];
        let mut step = NONE2;
        for e in 0..3 {
            let (a, b) = (tv[e], tv[(e + 1) % 3]);
            // CCW triangle: its interior is left of each directed edge a->b, so
            // `p` strictly right (orient negative) means it lies across edge e.
            if orient(pts[a], pts[b], p) == Sign::Negative && nbr[t][e] != NONE2 {
                step = nbr[t][e];
                break;
            }
        }
        if step == NONE2 {
            return t;
        }
        t = step;
    }
    (0..tris.len())
        .find(|&t| {
            alive[t]
                && (0..3).all(|e| {
                    let (a, b) = (tris[t][e], tris[t][(e + 1) % 3]);
                    orient(pts[a], pts[b], p) != Sign::Negative
                })
        })
        .unwrap_or(start)
}

/// Links the directed p-edge `(u, v)` at edge slot `es` of triangle `slot` to
/// the neighbouring new triangle that owns the reverse edge `(v, u)`.
fn link_pedge(
    map: &mut rustc_hash::FxHashMap<(usize, usize), (usize, usize)>,
    nbr: &mut [[usize; 3]],
    slot: usize,
    es: usize,
    u: usize,
    v: usize,
) {
    if let Some((other, oes)) = map.remove(&(v, u)) {
        nbr[slot][es] = other;
        nbr[other][oes] = slot;
    } else {
        map.insert((u, v), (slot, es));
    }
}

/// A persistent 2D triangulation with triangle adjacency: the Bowyer-Watson
/// Delaunay of `n` real points plus a covering super-triangle (its three
/// vertices are indices `n, n+1, n+2`, kept so the exterior is represented and
/// the convex hull has neighbours). The same structure backs the unconstrained
/// `delaunay2` (relaxation) and the constrained CDT (face triangulation): the
/// super-triangle lets a constraint walk and the exterior flood-fill terminate.
/// Triangles are CCW index triples; exact predicates throughout.
pub struct Cdt {
    pts: Vec<P2>,
    /// Number of real points; super-triangle vertices are `n, n+1, n+2`.
    pub n: usize,
    tris: Vec<[usize; 3]>,
    nbr: Vec<[usize; 3]>,
    alive: Vec<bool>,
    /// Some alive triangle containing each vertex (`NONE2` if never inserted,
    /// e.g. a dropped duplicate). Refreshed on every triangle write, so the
    /// star walk (`star`) replaces the full-mesh scans that made constraint
    /// forcing O(segments * triangles).
    vert_hint: Vec<usize>,
}

impl Cdt {
    /// Builds the Delaunay triangulation of `points` (super-triangle retained).
    pub fn new(points: &[P2]) -> Cdt {
        let n = points.len();
        let mut lo = points.first().copied().unwrap_or([0.0, 0.0]);
        let mut hi = lo;
        for p in points {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let d = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-12);
        let mid = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
        let big = 1000.0 * d;
        let mut pts: Vec<P2> = points.to_vec();
        let (s0, s1, s2) = (n, n + 1, n + 2);
        pts.push([mid[0] - big, mid[1] - big]);
        pts.push([mid[0] + big, mid[1] - big]);
        pts.push([mid[0], mid[1] + big]);

        let mut tris: Vec<[usize; 3]> = vec![ccw([s0, s1, s2], &pts)];
        let mut nbr: Vec<[usize; 3]> = vec![[NONE2; 3]];
        let mut alive: Vec<bool> = vec![true];
        let mut vert_hint: Vec<usize> = vec![NONE2; n + 3];
        for v in [s0, s1, s2] {
            vert_hint[v] = 0;
        }
        let mut free: Vec<usize> = Vec::new();
        let mut mark: Vec<u32> = vec![0];
        let mut epoch = 0u32;
        let mut last = 0usize;
        let mut edge_map: rustc_hash::FxHashMap<(usize, usize), (usize, usize)> =
            rustc_hash::FxHashMap::default();

        for i in 0..n {
            let p = pts[i];
            let start = locate2(last, p, &tris, &nbr, &alive, &pts);
            let tv = tris[start];
            if !in_circumcircle(pts[tv[0]], pts[tv[1]], pts[tv[2]], p) {
                continue; // cocircular: leave the mesh as is (consistent choice)
            }
            epoch += 1;
            mark[start] = epoch;
            let mut cavity = vec![start];
            let mut stack = vec![start];
            // Boundary edges (a, b, external triangle) found during the flood-fill.
            let mut boundary: Vec<(usize, usize, usize)> = Vec::new();
            while let Some(t) = stack.pop() {
                let tv = tris[t];
                for e in 0..3 {
                    let nb = nbr[t][e];
                    let bad = nb != NONE2 && {
                        let v = tris[nb];
                        in_circumcircle(pts[v[0]], pts[v[1]], pts[v[2]], p)
                    };
                    if bad {
                        if mark[nb] != epoch {
                            mark[nb] = epoch;
                            cavity.push(nb);
                            stack.push(nb);
                        }
                    } else {
                        boundary.push((tv[e], tv[(e + 1) % 3], nb));
                    }
                }
            }
            for &t in &cavity {
                alive[t] = false;
                free.push(t);
            }
            // Fan p to each boundary edge; link to the external triangle across that
            // edge and (via edge_map) to the adjacent new triangles along the p-edges.
            edge_map.clear();
            let mut last_new = start;
            for (a, b, x) in boundary {
                let slot = match free.pop() {
                    Some(s) => {
                        tris[s] = [a, b, i];
                        nbr[s] = [NONE2; 3];
                        alive[s] = true;
                        s
                    }
                    None => {
                        tris.push([a, b, i]);
                        nbr.push([NONE2; 3]);
                        alive.push(true);
                        mark.push(0);
                        tris.len() - 1
                    }
                };
                // edge 0 = (a,b) faces the external triangle x (which holds (b,a)).
                nbr[slot][0] = x;
                if x != NONE2 {
                    for e in 0..3 {
                        if tris[x][e] == b && tris[x][(e + 1) % 3] == a {
                            nbr[x][e] = slot;
                        }
                    }
                }
                // edge 1 = (b,i), edge 2 = (i,a): internal cavity edges.
                link_pedge(&mut edge_map, &mut nbr, slot, 1, b, i);
                link_pedge(&mut edge_map, &mut nbr, slot, 2, i, a);
                // Every cavity-boundary vertex (and p) reappears in the fan,
                // so refreshing hints here keeps them valid for all vertices.
                vert_hint[a] = slot;
                vert_hint[b] = slot;
                vert_hint[i] = slot;
                last_new = slot;
            }
            last = last_new;
        }
        Cdt {
            pts,
            n,
            tris,
            nbr,
            alive,
            vert_hint,
        }
    }

    /// All alive triangles containing `v`, walked through the neighbour
    /// pointers from the vertex hint (`None` if the vertex never entered the
    /// triangulation, e.g. a dropped exact duplicate). The star of a vertex is
    /// edge-connected, so the local walk enumerates it completely -- O(degree)
    /// instead of the full-mesh scan.
    fn star(&self, v: usize, out: &mut Vec<usize>) -> bool {
        out.clear();
        let seed = self.vert_hint[v];
        if seed == NONE2 || !self.alive[seed] || !self.tris[seed].contains(&v) {
            debug_assert!(seed == NONE2, "stale vertex hint for {v}");
            return false;
        }
        out.push(seed);
        let mut head = 0;
        while head < out.len() {
            let t = out[head];
            head += 1;
            let i = self.tris[t]
                .iter()
                .position(|&w| w == v)
                .expect("star triangle holds v");
            // The two edges incident to v: slot i = (v, next), slot i+2 = (prev, v).
            for e in [i, (i + 2) % 3] {
                let nb = self.nbr[t][e];
                if nb != NONE2 && !out.contains(&nb) {
                    debug_assert!(self.alive[nb] && self.tris[nb].contains(&v));
                    out.push(nb);
                }
            }
        }
        true
    }

    /// Builds the CONSTRAINED Delaunay of `points`: every segment forced as a
    /// mesh edge (Sloan 1993), then the Delaunay property restored on the free
    /// edges. Returns the LIVE structure plus the recorded constraint set, so
    /// incremental consumers (the mesh smoother) can move points and re-restore
    /// locally instead of re-triangulating from scratch.
    pub fn new_constrained(
        points: &[P2],
        segments: &[(usize, usize)],
    ) -> (Cdt, DSet<(usize, usize)>) {
        let mut cdt = Cdt::new(points);
        let mut constraints: DSet<(usize, usize)> = DSet::default();
        for &(a, b) in segments {
            cdt.force_edge(a, b, &mut constraints);
        }
        cdt.restore_delaunay(&constraints);
        (cdt, constraints)
    }

    /// [`Cdt::triangles`] filtered by the region membership of the centroid:
    /// the conforming triangulation of the face (exterior + holes dropped).
    pub fn kept_triangles(&self, inside: impl Fn(P2) -> bool) -> Vec<[usize; 3]> {
        self.triangles()
            .into_iter()
            .filter(|t| {
                let c = [
                    (self.pts[t[0]][0] + self.pts[t[1]][0] + self.pts[t[2]][0]) / 3.0,
                    (self.pts[t[0]][1] + self.pts[t[1]][1] + self.pts[t[2]][1]) / 3.0,
                ];
                inside(c)
            })
            .collect()
    }

    /// Position of vertex `i` (real or super).
    pub fn point(&self, i: usize) -> P2 {
        self.pts[i]
    }

    /// Moves vertex `i` in place. The caller guarantees the move keeps every
    /// incident alive triangle positively oriented (guarded moves); Delaunayness
    /// is repaired afterwards via [`Cdt::restore`].
    pub fn set_point(&mut self, i: usize, p: P2) {
        self.pts[i] = p;
    }

    /// Restores the Delaunay property on non-constraint edges by local flips
    /// (Lawson) -- the incremental repair after guarded point moves.
    pub fn restore(&mut self, constraints: &DSet<(usize, usize)>) {
        self.restore_delaunay(constraints);
    }

    /// Alive triangles whose three vertices are all real (super-triangle and its
    /// fan dropped).
    pub fn triangles(&self) -> Vec<[usize; 3]> {
        self.tris
            .iter()
            .enumerate()
            .filter(|&(t, _)| self.alive[t] && self.tris[t].iter().all(|&v| v < self.n))
            .map(|(_, t)| *t)
            .collect()
    }

    /// Local edge index `i` of triangle `t` for which `(tris[t][i], tris[t][i+1]) == (a, b)`.
    fn edge_slot(&self, t: usize, a: usize, b: usize) -> Option<usize> {
        (0..3).find(|&e| self.tris[t][e] == a && self.tris[t][(e + 1) % 3] == b)
    }

    /// Repoints the neighbour pointer of `tri` that currently references slot
    /// `old` to `new` (a no-op for the exterior `NONE2`).
    fn relink(&mut self, tri: usize, old: usize, new: usize) {
        if tri == NONE2 {
            return;
        }
        for e in 0..3 {
            if self.nbr[tri][e] == old {
                self.nbr[tri][e] = new;
            }
        }
    }

    /// Flips the diagonal of the convex quad sharing edge `e=(p,q)` of triangle
    /// `t` (shared with `n = nbr[t][e]`): replaces the diagonal `(p,q)` by
    /// `(x,y)` where `x,y` are the two apexes, reusing slots `t` and `n`. The
    /// quad is `x,p,y,q` in CCW order, so the new CCW triangles are `(x,p,y)` and
    /// `(x,y,q)`. Caller guarantees the flip is legal (\code{flippable}).
    fn flip(&mut self, t: usize, e: usize) {
        let nn = self.nbr[t][e];
        let p = self.tris[t][e];
        let q = self.tris[t][(e + 1) % 3];
        let x = self.tris[t][(e + 2) % 3];
        let f = self
            .edge_slot(nn, q, p)
            .expect("shared edge is reversed in the neighbour");
        let y = self.tris[nn][(f + 2) % 3];
        // Outer neighbours of the quad (read before overwriting).
        let n_qx = self.nbr[t][(e + 1) % 3]; // across (q,x)
        let n_xp = self.nbr[t][(e + 2) % 3]; // across (x,p)
        let n_py = self.nbr[nn][(f + 1) % 3]; // across (p,y)
        let n_yq = self.nbr[nn][(f + 2) % 3]; // across (y,q)
        self.tris[t] = [x, p, y];
        self.nbr[t] = [n_xp, n_py, nn];
        self.tris[nn] = [x, y, q];
        self.nbr[nn] = [t, n_yq, n_qx];
        // Reverse links: (x,p) and (y,q) keep their owners; (q,x) moves t->nn,
        // (p,y) moves nn->t.
        self.relink(n_qx, t, nn);
        self.relink(n_py, nn, t);
        // Hints: p lives only in t now, q only in nn; x and y are in both.
        self.vert_hint[p] = t;
        self.vert_hint[x] = t;
        self.vert_hint[y] = nn;
        self.vert_hint[q] = nn;
    }

    /// Is edge `e=(p,q)` of triangle `t` (interior, with neighbour) flippable,
    /// i.e. is the union quad strictly convex so the opposite diagonal `(x,y)`
    /// lies inside it? True iff `p` and `q` fall on opposite sides of line
    /// `(x,y)` (the apexes), with no three of the four corners collinear.
    fn flippable(&self, t: usize, e: usize) -> bool {
        let nn = self.nbr[t][e];
        if nn == NONE2 {
            return false;
        }
        let p = self.tris[t][e];
        let q = self.tris[t][(e + 1) % 3];
        let x = self.tris[t][(e + 2) % 3];
        let f = match self.edge_slot(nn, q, p) {
            Some(f) => f,
            None => return false,
        };
        let y = self.tris[nn][(f + 2) % 3];
        let sp = orient(self.pts[x], self.pts[y], self.pts[p]);
        let sq = orient(self.pts[x], self.pts[y], self.pts[q]);
        sp != Sign::Zero && sq != Sign::Zero && sp != sq
    }

    /// Does open segment `(a,b)` properly cross open segment `(c,d)` (interiors
    /// intersect at one point)? Both orientation pairs must have opposite signs
    /// (derivation in \code{report/derivations/cdt2d.py}).
    fn proper_cross(&self, a: usize, b: usize, c: usize, d: usize) -> bool {
        let (pa, pb, pc, pd) = (self.pts[a], self.pts[b], self.pts[c], self.pts[d]);
        let s1 = orient(pa, pb, pc);
        let s2 = orient(pa, pb, pd);
        let s3 = orient(pc, pd, pa);
        let s4 = orient(pc, pd, pb);
        s1 != Sign::Zero
            && s2 != Sign::Zero
            && s1 != s2
            && s3 != Sign::Zero
            && s4 != Sign::Zero
            && s3 != s4
    }

    /// True iff the (undirected) edge `(a,b)` is already an edge of some alive
    /// triangle. O(degree) via the star of `a`.
    fn has_edge(&self, a: usize, b: usize) -> bool {
        let mut star = Vec::new();
        if !self.star(a, &mut star) {
            return false; // `a` never entered the triangulation
        }
        star.iter().any(|&t| self.tris[t].contains(&b))
    }

    /// Collects the interior edges `(t, e)` that segment `(a,b)` properly crosses,
    /// by walking triangles from `a` toward `b`. Returns `None` if a third vertex
    /// lies exactly on the segment (the caller then splits the constraint there).
    fn crossing_edges(&self, a: usize, b: usize) -> Option<Vec<(usize, usize)>> {
        // Start triangle: the one in `a`'s star whose opposite edge is crossed
        // (O(degree), not a full-mesh scan).
        let mut astar = Vec::new();
        if !self.star(a, &mut astar) {
            return None;
        }
        let mut start = None;
        for &t in &astar {
            let k = self.tris[t]
                .iter()
                .position(|&w| w == a)
                .expect("star triangle holds a");
            let (c, d) = (self.tris[t][(k + 1) % 3], self.tris[t][(k + 2) % 3]);
            if self.proper_cross(a, b, c, d) {
                start = Some((t, (k + 1) % 3)); // edge (c,d) = slot k+1
                break;
            }
        }
        let (mut t, mut e) = start?;
        let mut out = Vec::new();
        loop {
            out.push((t, e));
            let nn = self.nbr[t][e];
            if nn == NONE2 {
                return Some(out); // hit the hull/super-triangle (degenerate input)
            }
            let tv = self.tris[nn];
            if tv.contains(&b) {
                return Some(out);
            }
            // Entry edge in nn is the reverse of (tris[t][e], tris[t][e+1]); the
            // apex is the third vertex. The segment exits through one of the two
            // other edges, whichever it properly crosses.
            let p = self.tris[t][e];
            let q = self.tris[t][(e + 1) % 3];
            let f = self.edge_slot(nn, q, p)?;
            let r = tv[(f + 2) % 3]; // apex of nn
                                     // Edge (p, r) is slot (f+2)%3 ? In nn=(q,p,r) with entry slot f=(q,p):
                                     // slot f+1 = (p,r), slot f+2 = (r,q).
            if self.proper_cross(a, b, p, r) {
                e = (f + 1) % 3;
            } else if self.proper_cross(a, b, r, q) {
                e = (f + 2) % 3;
            } else {
                return None; // segment passes through apex r: split there
            }
            t = nn;
        }
    }

    /// Forces the edge `(a,b)` to appear in the triangulation (Sloan 1993):
    /// repeatedly flip the edges the segment crosses until none remain, then
    /// records `(a,b)` as a constraint. Splits at an on-segment vertex if the
    /// walk reports one. Idempotent if the edge already exists.
    fn force_edge(&mut self, a: usize, b: usize, constraints: &mut DSet<(usize, usize)>) {
        if a == b {
            return;
        }
        constraints.insert((a.min(b), a.max(b)));
        if self.has_edge(a, b) {
            return;
        }
        let crossing = match self.crossing_edges(a, b) {
            Some(c) => c,
            None => {
                // The segment runs through some vertex r; find it and split.
                if let Some(r) = self.on_segment_vertex(a, b) {
                    self.force_edge(a, r, constraints);
                    self.force_edge(r, b, constraints);
                }
                return;
            }
        };
        let mut queue: std::collections::VecDeque<(usize, usize)> = crossing.into_iter().collect();
        let mut guard = 0usize;
        let cap = queue.len() * 50 + 100;
        while let Some((t, e)) = queue.pop_front() {
            guard += 1;
            if guard > cap {
                break; // safety: degenerate input, leave as-is
            }
            // The edge may have been renumbered by an earlier flip; re-find it by
            // its endpoints if they are still both present, else skip.
            if !self.alive[t] || self.nbr[t][e] == NONE2 {
                continue;
            }
            let (p, q) = (self.tris[t][e], self.tris[t][(e + 1) % 3]);
            if !self.proper_cross(a, b, p, q) {
                continue; // no longer crosses (already resolved)
            }
            if !self.flippable(t, e) {
                queue.push_back((t, e)); // try again once neighbours have flipped
                continue;
            }
            self.flip(t, e);
            // After the flip, slots t and the old neighbour hold the new diagonal
            // (x,y). If that diagonal still crosses (a,b), re-queue it.
            for &slot in &[t, self.nbr[t][2]] {
                if slot == NONE2 || !self.alive[slot] {
                    continue;
                }
                for ee in 0..3 {
                    let (u, v) = (self.tris[slot][ee], self.tris[slot][(ee + 1) % 3]);
                    if self.proper_cross(a, b, u, v) {
                        queue.push_back((slot, ee));
                    }
                }
            }
        }
    }

    /// A real vertex lying exactly on open segment `(a,b)`, if any (used to split
    /// a constraint that runs through a point).
    fn on_segment_vertex(&self, a: usize, b: usize) -> Option<usize> {
        let (pa, pb) = (self.pts[a], self.pts[b]);
        (0..self.n).find(|&v| {
            v != a && v != b && orient(pa, pb, self.pts[v]) == Sign::Zero && {
                // strictly between a and b
                let (px, py) = (self.pts[v][0], self.pts[v][1]);
                let t = if (pb[0] - pa[0]).abs() > (pb[1] - pa[1]).abs() {
                    (px - pa[0]) / (pb[0] - pa[0])
                } else {
                    (py - pa[1]) / (pb[1] - pa[1])
                };
                t > 0.0 && t < 1.0
            }
        })
    }

    /// Restores the Delaunay property on non-constraint edges after constraint
    /// insertion: flips any locally non-Delaunay, flippable, non-constraint edge.
    /// This yields the *constrained* Delaunay triangulation (Delaunay except
    /// where a constraint forbids the flip).
    fn restore_delaunay(&mut self, constraints: &DSet<(usize, usize)>) {
        let mut changed = true;
        let mut guard = 0usize;
        let cap = self.tris.len() * 40 + 200;
        while changed && guard < cap {
            changed = false;
            for t in 0..self.tris.len() {
                if !self.alive[t] {
                    continue;
                }
                for e in 0..3 {
                    let (u, v) = (self.tris[t][e], self.tris[t][(e + 1) % 3]);
                    if constraints.contains(&(u.min(v), u.max(v))) {
                        continue;
                    }
                    let nn = self.nbr[t][e];
                    if nn == NONE2 || !self.alive[nn] {
                        continue;
                    }
                    let f = match self.edge_slot(nn, v, u) {
                        Some(f) => f,
                        None => continue,
                    };
                    let apex = self.tris[nn][(f + 2) % 3];
                    let x = self.tris[t][(e + 2) % 3];
                    // Only consider real apexes (skip the super-triangle fan).
                    if !in_circumcircle(self.pts[u], self.pts[v], self.pts[x], self.pts[apex]) {
                        continue;
                    }
                    if self.flippable(t, e) {
                        self.flip(t, e);
                        changed = true;
                    }
                }
                guard += 1;
            }
        }
    }
}

/// Incremental 2D Delaunay triangulation of `points` (super-triangle removed),
/// CCW triples into `points`. Backs the relaxation passes (`cvt_fill`).
pub fn delaunay2(points: &[P2]) -> Vec<[usize; 3]> {
    if points.len() < 3 {
        return Vec::new();
    }
    Cdt::new(points).triangles()
}

/// Deterministic hashing for the constraint set.
type DSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// Constrained Delaunay triangulation of a planar face: the Delaunay of
/// `points` with every segment of `segments` forced as a mesh edge (Sloan
/// 1993), then the exterior and hole triangles removed by an `inside` test on
/// the triangle centroid. `segments` are index pairs into `points` tracing the
/// boundary chains (outer loop and any hole loops, each already subdivided by
/// its edge points). The result is a conforming triangulation of the (possibly
/// non-convex, holed) face: exactly the 2D analogue of the boundary-constrained
/// volume (\cref{prop:watertight}).
pub fn triangulate_constrained(
    points: &[P2],
    segments: &[(usize, usize)],
    inside: impl Fn(P2) -> bool,
) -> Vec<[usize; 3]> {
    if points.len() < 3 {
        return Vec::new();
    }
    let (cdt, _constraints) = Cdt::new_constrained(points, segments);
    cdt.kept_triangles(inside)
}

/// Interior angles (degrees) at each vertex of triangle (a, b, c).
fn tri_angles(a: P2, b: P2, c: P2) -> [f64; 3] {
    let ang = |u: P2, v: P2, w: P2| {
        let (e1, e2) = ([v[0] - u[0], v[1] - u[1]], [w[0] - u[0], w[1] - u[1]]);
        let n = (e1[0] * e1[0] + e1[1] * e1[1]).sqrt() * (e2[0] * e2[0] + e2[1] * e2[1]).sqrt();
        ((e1[0] * e2[0] + e1[1] * e2[1]) / (n + 1e-30))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    };
    [ang(a, b, c), ang(b, c, a), ang(c, a, b)]
}

/// Circumcenter of (a, b, c); `None` if (near-)degenerate.
fn circumcenter(a: P2, b: P2, c: P2) -> Option<P2> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-30 {
        return None;
    }
    let (a2, b2, c2) = (
        a[0] * a[0] + a[1] * a[1],
        b[0] * b[0] + b[1] * b[1],
        c[0] * c[0] + c[1] * c[1],
    );
    Some([
        (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
        (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d,
    ])
}

/// Sizing-field-driven Delaunay (Ruppert/Chew) refinement of a constrained
/// triangulation: repeatedly insert the circumcentre of any triangle whose
/// minimum angle is below `min_angle_deg` OR whose circumradius exceeds the local
/// `target` size (so the result is graded to the field). A circumcentre that
/// encroaches a REFINABLE boundary segment splits that segment at its midpoint
/// instead (Ruppert's segment protection), so free sheet boundaries stay
/// conforming while the interior gains a guaranteed angle bound -- no slivers.
/// Boundary midpoints are appended to `boundary`/`segments`/`refinable`; interior
/// points to `interior`. Terminates for input angles >= ~60 deg.
/// `target_count > 0` is a triangle BUDGET (a cap, not a target): the angle bound
/// is always met (no slivers) and the field is resolved down to its `min_h_surf`
/// floor as usual, but once the triangle count reaches `target_count` no further
/// size-driven splits are made -- the budget caps the refinement at
/// `min(field-resolved, target_count)`, spending its last splits on the most
/// oversized triangles first.
#[allow(clippy::too_many_arguments)]
pub fn refine_quality_with(
    boundary: &mut Vec<P2>,
    segments: &mut Vec<(usize, usize)>,
    refinable: &mut Vec<bool>,
    interior: &mut Vec<P2>,
    target: impl Fn(P2) -> f64,
    inside: impl Fn(P2) -> bool,
    min_angle_deg: f64,
    target_count: usize,
    // Hard cap on refinement passes. Each pass re-triangulates the whole patch,
    // so this bounds the cost: the surface stage runs the full set, a display
    // mesh only a few (the boundary conformity + angle bound are met early; the
    // rest is marginal interior density).
    max_passes: usize,
    // Called once per refinement pass with the current `(points, triangles)`
    // BEFORE that pass's inserts -- the coarse-to-fine intermediate states. A
    // no-op in the default `refine_quality`; the landing page collects them to
    // animate the mesher live.
    mut on_pass: impl FnMut(&[P2], &[[usize; 3]]),
) {
    let diam2 = |b: &[P2], u: usize, v: usize| 0.25 * dist2(b[u], b[v]);
    let mid = |b: &[P2], u: usize, v: usize| [0.5 * (b[u][0] + b[v][0]), 0.5 * (b[u][1] + b[v][1])];
    // Per-pass stage timing (RAPIDMESH_2D_TRACE): rebuild / criteria / apply.
    static TRACE2D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let trace = *TRACE2D.get_or_init(|| std::env::var_os("RAPIDMESH_2D_TRACE").is_some());
    for pass in 0..max_passes {
        let t0 = rapidmesh_exact::clock::Instant::now();
        let mut all = boundary.clone();
        all.extend_from_slice(interior);
        let tris = triangulate_constrained(&all, segments, &inside);
        let t_tri = t0.elapsed();
        on_pass(&all, &tris);
        let mut split: Vec<usize> = Vec::new();
        let mut inserts: Vec<P2> = Vec::new();
        let t1 = rapidmesh_exact::clock::Instant::now();

        // Edge -> the opposite apex of each incident triangle. A constrained
        // segment can only be encroached by a vertex VISIBLE to it, i.e. an apex
        // of one of its two triangles -- so we test those O(1) vertices instead of
        // every mesh vertex. (The all-vertices scan was O(segments * points) per
        // round, the refinement's quadratic cost.)
        let mut apex: rustc_hash::FxHashMap<(usize, usize), [usize; 2]> =
            rustc_hash::FxHashMap::default();
        for t in &tris {
            for e in 0..3 {
                let (a, b, c) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
                let slot = apex.entry((a.min(b), a.max(b))).or_insert([usize::MAX; 2]);
                if slot[0] == usize::MAX {
                    slot[0] = c;
                } else {
                    slot[1] = c;
                }
            }
        }

        // (1) protect segments: split any refinable segment whose diametral circle
        // contains a visible vertex. Done BEFORE circumcentres to keep termination.
        for (si, &(u, v)) in segments.iter().enumerate() {
            if !refinable[si] {
                continue;
            }
            let (m, h2) = (mid(boundary, u, v), diam2(boundary, u, v));
            if let Some(aps) = apex.get(&(u.min(v), u.max(v))) {
                if aps
                    .iter()
                    .any(|&k| k != usize::MAX && k != u && k != v && dist2(all[k], m) < h2 - 1e-12)
                {
                    split.push(si);
                }
            }
        }

        // (2) otherwise, drive on bad/oversized triangles. Angle-violating
        // triangles are always refined (mandatory, priority +inf); field-oversized
        // ones carry their relative oversize as priority. In budget mode only the
        // angle fixes plus enough of the WORST oversized to approach `target_count`
        // are taken -- so the refinement stops at min(field-resolved, budget).
        if split.is_empty() {
            let cur = tris.len();
            let mut cand: Vec<(f64, P2)> = Vec::new();
            // Segment lookup grid for the circumcentre encroachment test below:
            // every segment registers the cells its DIAMETRAL DISK overlaps, so a
            // candidate reads exactly one cell and tests only nearby segments --
            // the per-candidate all-segments scan was O(triangles * segments) per
            // pass, the refinement's remaining quadratic hot spot.
            let seg_cell = {
                let mean: f64 = segments
                    .iter()
                    .map(|&(u, v)| dist2(boundary[u], boundary[v]).sqrt())
                    .sum::<f64>()
                    / segments.len().max(1) as f64;
                mean.max(1e-12)
            };
            let skey = |p: P2| {
                (
                    (p[0] / seg_cell).floor() as i64,
                    (p[1] / seg_cell).floor() as i64,
                )
            };
            let mut seg_grid: rustc_hash::FxHashMap<(i64, i64), Vec<u32>> =
                rustc_hash::FxHashMap::default();
            for (si, &(u, v)) in segments.iter().enumerate() {
                let m = mid(boundary, u, v);
                let r = diam2(boundary, u, v).sqrt();
                let (lo, hi) = (skey([m[0] - r, m[1] - r]), skey([m[0] + r, m[1] + r]));
                for cx in lo.0..=hi.0 {
                    for cy in lo.1..=hi.1 {
                        seg_grid.entry((cx, cy)).or_default().push(si as u32);
                    }
                }
            }
            for t in &tris {
                let p = [all[t[0]], all[t[1]], all[t[2]]];
                let a = tri_angles(p[0], p[1], p[2]);
                let cc = match circumcenter(p[0], p[1], p[2]) {
                    Some(c) => c,
                    None => continue,
                };
                let amin = a[0].min(a[1]).min(a[2]);
                let tg = target(cc).max(1e-12);
                let ratio = dist2(p[0], cc) / (tg * tg);
                let angle_bad = amin < min_angle_deg;
                let size_bad = ratio > SIZE_SPLIT_RADIUS * SIZE_SPLIT_RADIUS
                    && (target_count == 0 || cur < target_count);
                if !(angle_bad || size_bad) {
                    continue;
                }
                // Encroachment: a circumcentre inside a boundary segment's
                // diametral circle. A refinable segment is split; a PROTECTED one
                // makes us drop the insert -- jamming a point against a fixed edge
                // is exactly what seeds the boundary slivers, so we leave the
                // (mildly bad) triangle rather than make it worse. A disk
                // containing `cc` registered `cc`'s cell, so one lookup is exact.
                let mut split_seg = None;
                let mut hits_protected = false;
                if let Some(list) = seg_grid.get(&skey(cc)) {
                    for &si in list {
                        let (u, v) = segments[si as usize];
                        if dist2(cc, mid(boundary, u, v)) < diam2(boundary, u, v) {
                            if refinable[si as usize] {
                                split_seg = Some(si as usize);
                                break;
                            }
                            hits_protected = true;
                        }
                    }
                }
                if let Some(si) = split_seg {
                    split.push(si);
                    continue;
                }
                if hits_protected {
                    continue;
                }
                if inside(cc) {
                    cand.push((if angle_bad { f64::INFINITY } else { ratio }, cc));
                }
            }
            if split.is_empty() {
                if target_count > 0 {
                    cand.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
                    let n_angle = cand.iter().take_while(|c| c.0.is_infinite()).count();
                    let deficit = target_count.saturating_sub(cur);
                    // each accepted insert adds ~2 triangles; take the angle fixes
                    // plus enough of the worst oversized to reach the budget.
                    let want = n_angle.max(deficit.div_ceil(2)).min(cand.len());
                    inserts.extend(cand[..want].iter().map(|c| c.1));
                } else {
                    inserts.extend(cand.iter().map(|c| c.1));
                }
            }
        }

        split.sort_unstable();
        split.dedup();
        if split.is_empty() && inserts.is_empty() {
            break;
        }
        if !split.is_empty() {
            let mut ns = Vec::with_capacity(segments.len() + split.len());
            let mut nr = Vec::with_capacity(refinable.len() + split.len());
            for (i, &(u, v)) in segments.iter().enumerate() {
                if split.binary_search(&i).is_ok() {
                    let m = mid(boundary, u, v);
                    let mi = boundary.len();
                    boundary.push(m);
                    ns.push((u, mi));
                    nr.push(true);
                    ns.push((mi, v));
                    nr.push(true);
                } else {
                    ns.push((u, v));
                    nr.push(refinable[i]);
                }
            }
            *segments = ns;
            *refinable = nr;
        }
        // Spacing guard via a uniform hash grid: reject an insert within
        // 0.5*target of an existing vertex. O(1) per insert instead of
        // O(points) (the loop's main quadratic cost besides the rebuild). It is
        // approximate at strong gradients -- a missed neighbour only yields a
        // slightly denser spot, never a quality violation.
        if !inserts.is_empty() {
            let gc = inserts
                .iter()
                .map(|&c| 0.5 * target(c))
                .fold(f64::INFINITY, f64::min)
                .max(1e-9);
            let key = |p: P2| ((p[0] / gc).floor() as i64, (p[1] / gc).floor() as i64);
            let mut grid: rustc_hash::FxHashMap<(i64, i64), Vec<P2>> =
                rustc_hash::FxHashMap::default();
            for &q in boundary.iter().chain(interior.iter()) {
                grid.entry(key(q)).or_default().push(q);
            }
            for c in inserts {
                let r = 0.5 * target(c);
                let r2 = r * r;
                let (cx, cy) = key(c);
                let rc = ((r / gc).ceil() as i64).min(6);
                let mut ok = true;
                'scan: for dx in -rc..=rc {
                    for dy in -rc..=rc {
                        if let Some(v) = grid.get(&(cx + dx, cy + dy)) {
                            if v.iter().any(|&q| dist2(c, q) < r2) {
                                ok = false;
                                break 'scan;
                            }
                        }
                    }
                }
                if ok {
                    grid.entry(key(c)).or_default().push(c);
                    interior.push(c);
                }
            }
        }
        if trace {
            eprintln!(
                "2D pass {pass}: tris {} segs {} pts {}  rebuild {:.3}s criteria+apply {:.3}s",
                tris.len(),
                segments.len(),
                all.len(),
                t_tri.as_secs_f64(),
                t1.elapsed().as_secs_f64()
            );
        }
    }
    // The Ruppert refinement met the angle bound but left the elements uneven;
    // relax the whole mesh (interior freely, the outline sliding) to convergence for
    // near-equilateral, gmsh-grade shape.
    let t_sm = rapidmesh_exact::clock::Instant::now();
    smooth_mesh(
        boundary,
        segments,
        interior,
        &inside,
        &target,
        min_angle_deg,
        50,
    );
    if trace {
        eprintln!("2D smooth: {:.3}s", t_sm.elapsed().as_secs_f64());
    }
}

/// Sizing-field-driven Ruppert/Chew refinement (see [`refine_quality_with`])
/// without per-pass snapshots -- the default path.
#[allow(clippy::too_many_arguments)]
pub fn refine_quality(
    boundary: &mut Vec<P2>,
    segments: &mut Vec<(usize, usize)>,
    refinable: &mut Vec<bool>,
    interior: &mut Vec<P2>,
    target: impl Fn(P2) -> f64,
    inside: impl Fn(P2) -> bool,
    min_angle_deg: f64,
    target_count: usize,
) {
    refine_quality_with(
        boundary,
        segments,
        refinable,
        interior,
        target,
        inside,
        min_angle_deg,
        target_count,
        60,
        |_, _| {},
    );
}

/// Holistic ODT mesh optimisation (SOTA element quality): after Ruppert meets the
/// angle bound, relax EVERY movable vertex toward its area-weighted optimal-Delaunay
/// target until convergence. Interior vertices move freely; a BOUNDARY vertex whose
/// two outline segments are collinear (an edge-interior point) SLIDES along that
/// edge, while corners (a segment-angle vertex) stay pinned — so the conductor
/// outline is preserved exactly, yet the edge triangles even out to near-equilateral
/// the way a full-mesh smoother (gmsh) does, instead of staying skewed against a
/// frozen boundary. Gauss-Seidel sweep (each move sees the latest positions),
/// per-move guarded: a move is applied only if every incident triangle stays
/// in-domain, un-flipped, and at/above the angle bound — the no-sliver guarantee and
/// the triangle count are preserved. Stops early once the largest move is below a
/// thousandth of the local edge length.
fn smooth_mesh(
    boundary: &mut [P2],
    segments: &[(usize, usize)],
    interior: &mut [P2],
    inside: impl Fn(P2) -> bool,
    target: impl Fn(P2) -> f64,
    min_angle_deg: f64,
    max_iters: usize,
) {
    let nb = boundary.len();
    if nb == 0 {
        return;
    }
    let sarea2 =
        |a: P2, b: P2, c: P2| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);

    // Contour adjacency of each boundary vertex (its two neighbours along the resampled
    // outline) -> a slide tangent if the outline is locally straight there, else pinned.
    let mut nbr = vec![[usize::MAX; 2]; nb];
    // Constraint degree: a vertex where a constraint CHAIN meets the outline (or two chains
    // meet) has three or more constraint neighbours and stays pinned — sliding it along the
    // outline would tilt the chain off the line it must hold.
    let mut deg = vec![0usize; nb];
    for &(u, v) in segments {
        if u < nb && v < nb {
            for (a, b) in [(u, v), (v, u)] {
                deg[a] += 1;
                if nbr[a][0] == usize::MAX {
                    nbr[a][0] = b;
                } else if nbr[a][1] == usize::MAX && nbr[a][0] != b {
                    nbr[a][1] = b;
                }
            }
        }
    }
    let tangent: Vec<Option<[f64; 2]>> = (0..nb)
        .map(|i| {
            let [n0, n1] = nbr[i];
            if n0 == usize::MAX || n1 == usize::MAX || deg[i] != 2 {
                return None;
            }
            let (a, p, b) = (boundary[n0], boundary[i], boundary[n1]);
            let d0 = [p[0] - a[0], p[1] - a[1]];
            let d1 = [b[0] - p[0], b[1] - p[1]];
            let l0 = (d0[0] * d0[0] + d0[1] * d0[1]).sqrt();
            let l1 = (d1[0] * d1[0] + d1[1] * d1[1]).sqrt();
            if l0 < 1e-300 || l1 < 1e-300 {
                return None;
            }
            // |sin∠| between the two outline segments: ~0 ⇒ edge-interior ⇒ slide; else corner.
            let cross = (d0[0] * d1[1] - d0[1] * d1[0]).abs() / (l0 * l1);
            if cross > 1e-3 {
                return None;
            }
            let t = [b[0] - a[0], b[1] - a[1]];
            let lt = (t[0] * t[0] + t[1] * t[1]).sqrt();
            Some([t[0] / lt, t[1] / lt])
        })
        .collect();

    // ONE live constrained triangulation for the whole smoothing: guarded
    // point moves keep every incident triangle exactly CCW, and a local
    // Lawson pass (`restore`) repairs Delaunayness after each sweep -- the
    // full re-triangulation per iteration (constraint forcing included) was
    // the smoother's dominant, superlinear cost.
    let mut all: Vec<P2> = boundary.to_vec();
    all.extend_from_slice(interior);
    let (mut cdt, constraints) = Cdt::new_constrained(&all, segments);
    let mut star: Vec<usize> = Vec::new();

    const RELAX: f64 = 0.9;
    for _ in 0..max_iters {
        let tris = cdt.kept_triangles(&inside);
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); all.len()];
        for (ti, t) in tris.iter().enumerate() {
            for &i in t {
                incident[i].push(ti);
            }
        }
        let mut max_rel = 0.0f64;
        // Gauss-Seidel: each vertex relaxes against the latest positions of its
        // neighbours (read live from the triangulation).
        for i in 0..all.len() {
            let tang = if i < nb {
                match tangent[i] {
                    Some(t) => Some(t),
                    None => continue, // pinned corner / open contour
                }
            } else {
                None
            };
            if incident[i].is_empty() {
                continue;
            }
            // ODT optimum x* = (Σ|T|·(sum of the OTHER two verts)) / (2 Σ|T|), plus the
            // local edge scale for the convergence test.
            let (mut num, mut den, mut hmin) = ([0.0f64; 2], 0.0f64, f64::INFINITY);
            for &ti in &incident[i] {
                let t = tris[ti];
                let (a, b, c) = (cdt.point(t[0]), cdt.point(t[1]), cdt.point(t[2]));
                // DENSITY-weighted ODT: weight by area · ρ, ρ = 1/h² from the sizing field at the
                // triangle centroid. Plain area weighting equidistributes to UNIFORM size and so
                // fights a graded field at the fine↔coarse transitions (skewed elements); weighting
                // by the target density makes the optimum follow the field → smooth, well-shaped
                // graded transitions. ρ is a smooth scalar, so the no-flip / angle guard still holds.
                let cen = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0];
                let h = target(cen).max(1e-12);
                let w = 0.5 * sarea2(a, b, c).abs() / (h * h);
                num[0] += w * (a[0] + b[0] + c[0] - cdt.point(i)[0]);
                num[1] += w * (a[1] + b[1] + c[1] - cdt.point(i)[1]);
                den += w;
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    hmin = hmin.min((u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2));
                }
            }
            if den <= 0.0 {
                continue;
            }
            let cur = cdt.point(i);
            let opt = [num[0] / (2.0 * den), num[1] / (2.0 * den)];
            let mut delta = [opt[0] - cur[0], opt[1] - cur[1]];
            if let Some(t) = tang {
                let dp = delta[0] * t[0] + delta[1] * t[1]; // keep only the along-edge component
                delta = [dp * t[0], dp * t[1]];
            }
            let cand = [cur[0] + RELAX * delta[0], cur[1] + RELAX * delta[1]];
            if i >= nb && !inside(cand) {
                continue;
            }
            // Quality gate on the KEPT triangles (as before), plus the exact
            // CCW guard over the FULL star (exterior/hole triangles included):
            // the live structure's flip invariants need every alive triangle
            // to stay positively oriented, not just the in-domain ones.
            let ok = incident[i].iter().all(|&ti| {
                let t = tris[ti];
                let q: [P2; 3] =
                    std::array::from_fn(|j| if t[j] == i { cand } else { cdt.point(t[j]) });
                let a = tri_angles(q[0], q[1], q[2]);
                a[0].min(a[1]).min(a[2]) >= min_angle_deg
            }) && {
                cdt.star(i, &mut star)
                    && star.iter().all(|&ti| {
                        let t = cdt.tris[ti];
                        let q: [P2; 3] =
                            std::array::from_fn(|j| if t[j] == i { cand } else { cdt.point(t[j]) });
                        orient(q[0], q[1], q[2]) == Sign::Positive
                    })
            };
            if ok {
                let mv2 = (cand[0] - cur[0]).powi(2) + (cand[1] - cur[1]).powi(2);
                if hmin.is_finite() && hmin > 0.0 {
                    max_rel = max_rel.max(mv2 / hmin);
                }
                cdt.set_point(i, cand);
                all[i] = cand;
            }
        }
        boundary.copy_from_slice(&all[..nb]);
        interior.copy_from_slice(&all[nb..]);
        if max_rel < 1e-6 {
            break; // converged: largest move < 0.1 % of the local edge length
        }
        // Local Delaunay repair (flips only) before the next sweep.
        cdt.restore(&constraints);
    }
}

/// Fills a planar region with Lloyd-relaxed interior points at a GRADED local
/// `target` spacing (`target(q)` is the desired edge length at `q`). `step` is
/// the finest target on the patch, the grid step of the initial scatter; the
/// per-point separation is the LOCAL `0.5 * target`, so the density grades:
/// dense where `target` is small, sparse where it is large. `boundary` is the
/// set of FIXED boundary points (graded 1D edge points and corners); `inside`
/// decides patch membership (exact, supplied by the caller). Interior points are
/// scattered on a grid in `[lo, hi]`, kept inside and clear of the boundary by
/// the local radius, then moved toward the area-weighted centroid of their
/// incident triangles with a local separation guard (no collapse / sliver seed).
#[allow(clippy::too_many_arguments)]
pub fn cvt_fill(
    boundary: &[P2],
    lo: P2,
    hi: P2,
    step: f64,
    target: impl Fn(P2) -> f64,
    iters: usize,
    inside: impl Fn(P2) -> bool,
    density: bool,
) -> Vec<P2> {
    if !(step.is_finite() && step > 0.0) {
        return Vec::new();
    }
    let sep2 = |q: P2| (0.5 * target(q)).powi(2);
    let nb = boundary.len();
    let nx = (((hi[0] - lo[0]) / step).ceil() as usize).max(1);
    let ny = (((hi[1] - lo[1]) / step).ceil() as usize).max(1);
    // Greedy graded scatter: keep a grid node only if it clears the boundary and
    // every already-kept interior point by its OWN local radius. A spatial hash
    // (cell ~ the finest radius) makes each clearance test O(1) instead of O(n) --
    // the scatter and the relaxation below were the fill's O(n^2) bottleneck.
    let gc = (0.5 * step).max(1e-9);
    let key = |p: P2| ((p[0] / gc).floor() as i64, (p[1] / gc).floor() as i64);
    let mut bgrid: rustc_hash::FxHashMap<(i64, i64), Vec<P2>> = rustc_hash::FxHashMap::default();
    for &b in boundary {
        bgrid.entry(key(b)).or_default().push(b);
    }
    let mut interior: Vec<P2> = Vec::new();
    let mut igrid: rustc_hash::FxHashMap<(i64, i64), Vec<P2>> = rustc_hash::FxHashMap::default();
    for i in 1..nx {
        for j in 1..ny {
            let q = [lo[0] + i as f64 * step, lo[1] + j as f64 * step];
            if !inside(q) {
                continue;
            }
            let r2 = sep2(q);
            if grid_clear(&bgrid, q, r2, gc, None) && grid_clear(&igrid, q, r2, gc, None) {
                igrid.entry(key(q)).or_default().push(q);
                interior.push(q);
            }
        }
    }

    for _ in 0..iters {
        if interior.is_empty() {
            break;
        }
        let mut all: Vec<P2> = boundary.to_vec();
        all.extend_from_slice(&interior);
        let tris = delaunay2(&all);
        let mut num = vec![[0.0f64; 2]; all.len()];
        let mut den = vec![0.0f64; all.len()];
        for t in &tris {
            let p = [all[t[0]], all[t[1]], all[t[2]]];
            // Float area as a relaxation WEIGHT (not a decision).
            let area = 0.5
                * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                    - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))
                    .abs();
            let c = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            ];
            // DENSITY-WEIGHTED CVT (2D, adaptive mode): weight by area * rho,
            // rho = 1/target^2 (spacing ~ target), so a graded surface field
            // relaxes into a smooth gradient. Gated: it shifts the surface point
            // distribution, which the exact-volume fixtures (1e-9) cannot absorb;
            // plain area weighting is the uniform CVT.
            let w = if density {
                let h = target(c).max(1e-12);
                area / (h * h)
            } else {
                area
            };
            for &v in t {
                num[v][0] += w * c[0];
                num[v][1] += w * c[1];
                den[v] += w;
            }
        }
        // Rebuild the interior hash for this pass, then keep it live as points
        // move (so later moves see earlier ones, as the O(n) scan did).
        igrid.clear();
        for &p in &interior {
            igrid.entry(key(p)).or_default().push(p);
        }
        for k in 0..interior.len() {
            let v = nb + k;
            if den[v] == 0.0 {
                continue;
            }
            let tgt = [num[v][0] / den[v], num[v][1] / den[v]];
            if !inside(tgt) {
                continue;
            }
            let r2 = sep2(tgt);
            let old = interior[k];
            if grid_clear(&bgrid, tgt, r2, gc, None) && grid_clear(&igrid, tgt, r2, gc, Some(old)) {
                if let Some(cell) = igrid.get_mut(&key(old)) {
                    if let Some(pos) = cell.iter().position(|&p| p == old) {
                        cell.swap_remove(pos);
                    }
                }
                igrid.entry(key(tgt)).or_default().push(tgt);
                interior[k] = tgt;
            }
        }
    }
    interior
}

/// True if `q` is at least its own radius (`sqrt(r2)`) from every point in the
/// spatial hash `grid` (cell size `gc`), optionally skipping one point `skip`
/// (the query's own current position during relaxation). The cell window is
/// sized to the radius, so it is exact -- no neighbour within range is missed.
fn grid_clear(
    grid: &rustc_hash::FxHashMap<(i64, i64), Vec<P2>>,
    q: P2,
    r2: f64,
    gc: f64,
    skip: Option<P2>,
) -> bool {
    let rc = (r2.sqrt() / gc).ceil() as i64 + 1;
    let cx = (q[0] / gc).floor() as i64;
    let cy = (q[1] / gc).floor() as i64;
    for dx in -rc..=rc {
        for dy in -rc..=rc {
            if let Some(v) = grid.get(&(cx + dx, cy + dy)) {
                for &p in v {
                    if Some(p) == skip {
                        continue;
                    }
                    if dist2(q, p) < r2 {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Even-odd point-in-contours test over ROW-BUCKETED loop edges. Same
/// crossing rule and per-edge arithmetic as the classic even-odd scan (so the
/// answers are bit-identical), but a query only touches the edges whose
/// y-interval intersects its row -- the linear all-loops scan per `inside`
/// query was THE canvas-scale hot spot (every triangle filter, CVT candidate
/// and smoothing gate pays one query; with mesh_layers packing every patch
/// into one canvas, each query scanned the whole layout's outline).
struct PipRows {
    y0: f64,
    cell: f64,
    rows: Vec<Vec<(P2, P2)>>,
}

impl PipRows {
    fn build(loops: &[Vec<P2>]) -> PipRows {
        let (mut ylo, mut yhi, mut len_sum, mut n_edges) =
            (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize);
        for lp in loops {
            let n = lp.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let (a, b) = (lp[i], lp[(i + 1) % n]);
                ylo = ylo.min(a[1]);
                yhi = yhi.max(a[1]);
                len_sum += dist2(a, b).sqrt();
                n_edges += 1;
            }
        }
        if n_edges == 0 || !(yhi > ylo) {
            return PipRows {
                y0: 0.0,
                cell: 1.0,
                rows: Vec::new(),
            };
        }
        // Row height ~ the mean edge length: short (h-sized) boundary edges land
        // in one or two rows; the row count stays bounded for huge canvases.
        let cell = (len_sum / n_edges as f64)
            .max((yhi - ylo) / 4096.0)
            .max(1e-12);
        let nrows = (((yhi - ylo) / cell).ceil() as usize + 1).max(1);
        let mut rows: Vec<Vec<(P2, P2)>> = vec![Vec::new(); nrows];
        let row_of =
            |y: f64| (((y - ylo) / cell).floor() as i64).clamp(0, nrows as i64 - 1) as usize;
        for lp in loops {
            let n = lp.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let (a, b) = (lp[i], lp[(i + 1) % n]);
                let (r0, r1) = (row_of(a[1].min(b[1])), row_of(a[1].max(b[1])));
                for row in rows.iter_mut().take(r1 + 1).skip(r0) {
                    row.push((a, b));
                }
            }
        }
        PipRows {
            y0: ylo,
            cell,
            rows,
        }
    }

    /// Even-odd membership of `p` (inside the outer loop, outside the holes).
    fn inside(&self, p: P2) -> bool {
        if self.rows.is_empty() {
            return false;
        }
        let r = ((p[1] - self.y0) / self.cell).floor();
        if r < 0.0 || r >= self.rows.len() as f64 {
            return false; // outside the loops' y-range entirely
        }
        let mut inside = false;
        for &(a, b) in &self.rows[r as usize] {
            // Identical crossing rule to the classic scan (half-open in y).
            if ((a[1] > p[1]) != (b[1] > p[1]))
                && (p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0])
            {
                inside = !inside;
            }
        }
        inside
    }
}

/// Squared distance from point `p` to segment `a`-`b`.
fn pt_seg_dist2(p: P2, a: P2, b: P2) -> f64 {
    let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
    let (wx, wy) = (p[0] - a[0], p[1] - a[1]);
    let c1 = vx * wx + vy * wy;
    if c1 <= 0.0 {
        return wx * wx + wy * wy;
    }
    let c2 = vx * vx + vy * vy;
    if c2 <= c1 {
        return (p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2);
    }
    let t = c1 / c2;
    (p[0] - (a[0] + t * vx)).powi(2) + (p[1] - (a[1] + t * vy)).powi(2)
}

/// Knobs for [`mesh_polygon`] -- everything but the contour loops, the sizing field,
/// and the per-pass callback. `step` seeds the CVT grid (about the finest target edge
/// length); `target_count` is the triangle BUDGET (`0` = field-driven, else the mesh
/// is capped at `min(field-resolved, budget)`); `min_angle_deg` is the Ruppert quality
/// bound; `cvt_iters`/`max_passes` bound the work.
#[derive(Clone, Copy, Debug)]
pub struct PolyMeshParams {
    pub step: f64,
    pub min_angle_deg: f64,
    pub target_count: usize,
    pub cvt_iters: usize,
    pub max_passes: usize,
}

impl Default for PolyMeshParams {
    fn default() -> Self {
        PolyMeshParams {
            step: 0.0,
            min_angle_deg: 28.0,
            target_count: 0,
            cvt_iters: 4,
            max_passes: 12,
        }
    }
}

/// Mesh a polygon-with-holes in ONE call -- the 2D entry point: contour loops
/// and a sizing field in, a sliver-free graded triangle mesh out. This is the
/// same path the surface stage runs per planar patch: a graded CVT seed
/// ([`cvt_fill`]) fills the interior at the field density, Ruppert
/// ([`refine_quality`]) enforces the angle bound, then the constrained
/// triangulation is read back.
///
/// `loops` are the boundary contours -- the outer boundary followed by the holes
/// (orientation irrelevant; membership is even-odd). `target` is the desired edge
/// length at any point (uniform or graded). `on_pass(points, triangles)` fires once
/// per Ruppert pass with the current intermediate mesh (a no-op `|_, _| {}` for the
/// non-animated path). Returns `(points, triangles)`.
///
/// THE single 2D loops entry point: every knob (the CVT seed `step`, the Ruppert
/// quality bound, the triangle BUDGET, the work bounds) rides in [`PolyMeshParams`].
pub fn mesh_polygon(
    loops: &[Vec<P2>],
    target: impl Fn(P2) -> f64,
    params: &PolyMeshParams,
    on_pass: impl FnMut(&[P2], &[[usize; 3]]),
) -> (Vec<P2>, Vec<[usize; 3]>) {
    mesh_polygon_with_chains(loops, &[], target, params, on_pass)
}

/// [`mesh_polygon`] with additional CONSTRAINT CHAINS: open polylines (or closed ones given
/// with a repeated end point) whose segments are honoured as element edges like the contour
/// segments — protected from splitting, seeds cleared from them — but which play NO part in
/// the membership (even-odd over `loops` only). This is how a planar layer is meshed
/// ALIGNED to the outlines of the conductors on the layers above and below it: the outlines,
/// clipped to the layer, come in as chains, and every cell boundary the neighbouring
/// conductor induces in the charge is an element edge. A chain must lie inside the region
/// (chain end points on a contour must be contour vertices).
pub fn mesh_polygon_with_chains(
    loops: &[Vec<P2>],
    chains: &[Vec<P2>],
    target: impl Fn(P2) -> f64,
    params: &PolyMeshParams,
    on_pass: impl FnMut(&[P2], &[[usize; 3]]),
) -> (Vec<P2>, Vec<[usize; 3]>) {
    let mut boundary: Vec<P2> = Vec::new();
    let mut segments: Vec<(usize, usize)> = Vec::new();
    for lp in loops {
        let n = lp.len();
        if n < 3 {
            continue;
        }
        let base = boundary.len();
        boundary.extend_from_slice(lp);
        for i in 0..n {
            segments.push((base + i, base + (i + 1) % n));
        }
    }
    // Chains: their points are welded onto EXISTING boundary points where they coincide (a
    // chain end on a contour vertex), so no duplicate vertex enters the triangulation.
    for ch in chains {
        if ch.len() < 2 {
            continue;
        }
        let mut idx: Vec<usize> = Vec::with_capacity(ch.len());
        for &q in ch {
            let found = boundary
                .iter()
                .position(|&b| (b[0] - q[0]).abs() <= 1e-12 && (b[1] - q[1]).abs() <= 1e-12);
            idx.push(match found {
                Some(i) => i,
                None => {
                    boundary.push(q);
                    boundary.len() - 1
                }
            });
        }
        for w in idx.windows(2) {
            if w[0] != w[1] {
                segments.push((w[0], w[1]));
            }
        }
    }
    let pip = PipRows::build(loops);
    mesh_constrained(
        boundary,
        segments,
        target,
        |p| pip.inside(p),
        params.step,
        params.min_angle_deg,
        params.target_count,
        params.cvt_iters,
        params.max_passes,
        on_pass,
    )
}

/// THE 2D meshing core, shared by EVERY 2D mesh in the project (the surface
/// stage's planar patches and the landing page): a fixed boundary (`boundary`
/// points + constraint `segments`), a sizing `target`, and an `inside`
/// predicate in -> a graded, sliver-free triangulation out.
///
/// A graded CVT seed ([`cvt_fill`]) fills the interior at the field; seeds that
/// would hide under a contour edge are dropped (edge clearance); then Ruppert
/// ([`refine_quality_with`]) refines the interior with the boundary PROTECTED --
/// the caller pre-samples the contours, so re-splitting them would only chase the
/// seed points off the boundary into thin spikes. `target_count` caps the element
/// count (0 = field-driven); `step` seeds the CVT grid; `max_passes`/`cvt_iters`
/// bound the work. Returns `(points, triangles)` with the boundary first
/// (unchanged) then the interior.
#[allow(clippy::too_many_arguments)]
pub fn mesh_constrained(
    mut boundary: Vec<P2>,
    mut segments: Vec<(usize, usize)>,
    target: impl Fn(P2) -> f64,
    inside: impl Fn(P2) -> bool,
    step: f64,
    min_angle_deg: f64,
    target_count: usize,
    cvt_iters: usize,
    max_passes: usize,
    on_pass: impl FnMut(&[P2], &[[usize; 3]]),
) -> (Vec<P2>, Vec<[usize; 3]>) {
    if boundary.len() < 3 {
        return (boundary, Vec::new());
    }
    let (mut lo, mut hi) = (boundary[0], boundary[0]);
    for &p in &boundary {
        lo[0] = lo[0].min(p[0]);
        lo[1] = lo[1].min(p[1]);
        hi[0] = hi[0].max(p[0]);
        hi[1] = hi[1].max(p[1]);
    }
    let mut interior = cvt_fill(&boundary, lo, hi, step, &target, cvt_iters, &inside, true);
    // cvt_fill clears boundary POINTS but not boundary EDGES, so a seed can land
    // just under a contour segment and form a flat boundary triangle. Drop seeds
    // closer than ~half the local size to any segment. Segments live in a
    // uniform grid over their own AABBs; each seed scans only the cells its
    // clearance ball overlaps (the all-segments scan was O(seeds * segments)).
    {
        let seg_cell = {
            let mean: f64 = segments
                .iter()
                .map(|&(u, v)| dist2(boundary[u], boundary[v]).sqrt())
                .sum::<f64>()
                / segments.len().max(1) as f64;
            mean.max(1e-12)
        };
        let skey = |p: P2| {
            (
                (p[0] / seg_cell).floor() as i64,
                (p[1] / seg_cell).floor() as i64,
            )
        };
        let mut seg_grid: rustc_hash::FxHashMap<(i64, i64), Vec<u32>> =
            rustc_hash::FxHashMap::default();
        for (si, &(u, v)) in segments.iter().enumerate() {
            let (a, b) = (boundary[u], boundary[v]);
            let lo = skey([a[0].min(b[0]), a[1].min(b[1])]);
            let hi = skey([a[0].max(b[0]), a[1].max(b[1])]);
            for cx in lo.0..=hi.0 {
                for cy in lo.1..=hi.1 {
                    seg_grid.entry((cx, cy)).or_default().push(si as u32);
                }
            }
        }
        interior.retain(|&p| {
            let r = 0.5 * target(p);
            let r2 = r * r;
            let (clo, chi) = (skey([p[0] - r, p[1] - r]), skey([p[0] + r, p[1] + r]));
            for cx in clo.0..=chi.0 {
                for cy in clo.1..=chi.1 {
                    if let Some(list) = seg_grid.get(&(cx, cy)) {
                        for &si in list {
                            let (u, v) = segments[si as usize];
                            if pt_seg_dist2(p, boundary[u], boundary[v]) < r2 {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        });
    }
    // The boundary is PROTECTED (caller pre-samples it): Ruppert refines the
    // interior but never re-splits a contour edge, so no boundary spikes.
    let mut refin = vec![false; segments.len()];
    refine_quality_with(
        &mut boundary,
        &mut segments,
        &mut refin,
        &mut interior,
        &target,
        &inside,
        min_angle_deg,
        target_count,
        max_passes,
        on_pass,
    );
    let mut all = boundary;
    all.extend(interior);
    let tris = triangulate_constrained(&all, &segments, &inside);
    (all, tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delaunay2_grid_triangle_count() {
        let mut pts = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                pts.push([i as f64, j as f64]);
            }
        }
        // 25 points, 16 on the hull -> 2*25 - 2 - 16 = 32 triangles.
        assert_eq!(delaunay2(&pts).len(), 32);
    }

    /// Even-odd point-in-polygon over one or more loops (test helper).
    fn pip(p: P2, loops: &[Vec<usize>], pts: &[P2]) -> bool {
        let mut inside = false;
        for lp in loops {
            let m = lp.len();
            for k in 0..m {
                let a = pts[lp[k]];
                let b = pts[lp[(k + 1) % m]];
                if (a[1] > p[1]) != (b[1] > p[1]) {
                    let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
                    if p[0] < x {
                        inside = !inside;
                    }
                }
            }
        }
        inside
    }

    fn loops_to_segs(loops: &[Vec<usize>]) -> Vec<(usize, usize)> {
        let mut s = Vec::new();
        for lp in loops {
            let m = lp.len();
            for k in 0..m {
                s.push((lp[k], lp[(k + 1) % m]));
            }
        }
        s
    }

    fn tri_area(t: [usize; 3], pts: &[P2]) -> f64 {
        let (a, b, c) = (pts[t[0]], pts[t[1]], pts[t[2]]);
        0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
    }

    #[test]
    fn constrained_l_polygon_fills_only_the_l() {
        // Reflex (non-convex) L: an unconstrained Delaunay would triangulate the
        // convex hull (area 4); the constrained one must cover exactly the L
        // (area 3) and contain every boundary edge.
        let pts: Vec<P2> = vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let loops = vec![vec![0, 1, 2, 3, 4, 5]];
        let segs = loops_to_segs(&loops);
        let tris = triangulate_constrained(&pts, &segs, |p| pip(p, &loops, &pts));
        let area: f64 = tris.iter().map(|&t| tri_area(t, &pts)).sum();
        assert!((area - 3.0).abs() < 1e-9, "L area should be 3, got {area}");
        // every boundary edge is an edge of some kept triangle
        for k in 0..6 {
            let (a, b) = (loops[0][k], loops[0][(k + 1) % 6]);
            let present = tris.iter().any(|t| {
                (0..3).any(|e| {
                    let (u, v) = (t[e], t[(e + 1) % 3]);
                    (u == a && v == b) || (u == b && v == a)
                })
            });
            assert!(present, "boundary edge ({a},{b}) missing");
        }
    }

    #[test]
    fn constrained_square_with_hole_leaves_the_hole_empty() {
        // Outer square [0,4]^2 (CCW) with a square hole [1,3]^2 (CW): the meshed
        // region is the annulus, area 16 - 4 = 12, and no triangle centroid may
        // fall in the hole.
        let pts: Vec<P2> = vec![
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0], // outer
            [1.0, 1.0],
            [1.0, 3.0],
            [3.0, 3.0],
            [3.0, 1.0], // hole (CW)
        ];
        let loops = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let segs = loops_to_segs(&loops);
        let tris = triangulate_constrained(&pts, &segs, |p| pip(p, &loops, &pts));
        let area: f64 = tris.iter().map(|&t| tri_area(t, &pts)).sum();
        assert!(
            (area - 12.0).abs() < 1e-9,
            "annulus area should be 12, got {area}"
        );
        for &t in &tris {
            let c = [
                (pts[t[0]][0] + pts[t[1]][0] + pts[t[2]][0]) / 3.0,
                (pts[t[0]][1] + pts[t[1]][1] + pts[t[2]][1]) / 3.0,
            ];
            let in_hole = c[0] > 1.0 && c[0] < 3.0 && c[1] > 1.0 && c[1] < 3.0;
            assert!(!in_hole, "triangle centroid {c:?} lies in the hole");
        }
    }

    #[test]
    fn cvt_fill_square_well_separated() {
        // Unit square boundary at spacing 0.2.
        let m = 5;
        let mut boundary = Vec::new();
        for i in 0..m {
            boundary.push([i as f64 / m as f64, 0.0]);
            boundary.push([1.0, i as f64 / m as f64]);
            boundary.push([1.0 - i as f64 / m as f64, 1.0]);
            boundary.push([0.0, 1.0 - i as f64 / m as f64]);
        }
        let sq = |p: P2| p[0] > 0.0 && p[0] < 1.0 && p[1] > 0.0 && p[1] < 1.0;
        let interior = cvt_fill(
            &boundary,
            [0.0, 0.0],
            [1.0, 1.0],
            0.2,
            |_| 0.2,
            12,
            sq,
            true,
        );
        assert!(!interior.is_empty());
        let mut all = boundary.clone();
        all.extend_from_slice(&interior);
        let mut min_sep2 = f64::MAX;
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                min_sep2 = min_sep2.min(dist2(all[i], all[j]));
            }
        }
        assert!(
            min_sep2.sqrt() >= 0.5 * 0.2,
            "points too close: {}",
            min_sep2.sqrt()
        );
    }
}
