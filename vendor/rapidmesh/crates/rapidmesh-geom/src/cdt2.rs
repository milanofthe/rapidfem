//! Exact 2D Delaunay and constrained Delaunay triangulation: incremental
//! Bowyer-Watson with a super-triangle, constraints forced by Sloan's edge
//! flips, exact `orient2d`/`incircle2d` throughout. It backs the planar mesher
//! (triangulation, refinement and relaxation of faces in their charts) and
//! the tessellation of imported faces in their parameters.

use rapidmesh_exact::{incircle2d, orient2d, Axis, Point3, Sign};

type P2 = [f64; 2];

fn p3(p: P2) -> Point3 {
    Point3::explicit(p[0], p[1], 0.0)
}

/// Orientation of (a, b, c) in the xy plane (exact).
pub fn orient(a: P2, b: P2, c: P2) -> Sign {
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

const NONE2: usize = usize::MAX;

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
    pub fn star(&self, v: usize, out: &mut Vec<usize>) -> bool {
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

    /// Vertices of triangle `t` (CCW).
    pub fn triangle(&self, t: usize) -> [usize; 3] {
        self.tris[t]
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
pub type DSet<T> =
    std::collections::HashSet<T, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

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
