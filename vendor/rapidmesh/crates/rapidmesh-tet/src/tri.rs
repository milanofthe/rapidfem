//! The regular (weighted Delaunay) tetrahedralization at the core of the
//! mesher, built incrementally (Bowyer and Watson) over explicit points.
//!
//! Tets live in a slab of slots with a free list. Each face stores its
//! neighbour as `slot << 2 | face`, so crossing a face and finding the way
//! back are both O(1). Point location is a visibility walk from the last
//! touched tet (a scan backs up degenerate walks); the cavity grows through
//! the neighbours and is repaired to a star shape around the new point;
//! membership is tracked with epoch marks on tets and vertices, so the hot
//! path neither hashes nor allocates beyond amortized scratch growth.
//!
//! Predicates are exact: Shewchuk's adaptive orient3d and insphere, and the
//! filtered power test once weights appear. Internal vertices 0..4 are the
//! corners of an enclosing super-tet; public indices count inserted points
//! from 0.

use rapidmesh_exact::Sign;

type P3 = [f64; 3];

/// No neighbour (the super-tet hull), no slot, no vertex.
pub const NONE: u32 = u32::MAX;

fn orient(pts: &[P3], a: u32, b: u32, c: u32, d: u32) -> Sign {
    Sign::of_f64(geometry_predicates::orient3d(
        pts[a as usize],
        pts[b as usize],
        pts[c as usize],
        pts[d as usize],
    ))
}

fn insphere(pts: &[P3], t: [u32; 4], p: u32) -> Sign {
    Sign::of_f64(geometry_predicates::insphere(
        pts[t[0] as usize],
        pts[t[1] as usize],
        pts[t[2] as usize],
        pts[t[3] as usize],
        pts[p as usize],
    ))
}

/// Face of a positively oriented tet opposite vertex `i`, wound so the
/// opposite vertex lies on its positive side.
pub fn face(t: [u32; 4], i: usize) -> [u32; 3] {
    match i {
        0 => [t[1], t[3], t[2]],
        1 => [t[0], t[2], t[3]],
        2 => [t[0], t[3], t[1]],
        _ => [t[0], t[1], t[2]],
    }
}

fn pack(slot: u32, face: usize) -> u32 {
    slot << 2 | face as u32
}

/// The slot and face of a packed neighbour.
pub fn unpack(nb: u32) -> (u32, usize) {
    (nb >> 2, (nb & 3) as usize)
}

/// An incremental regular tetrahedralization.
pub struct Triangulation {
    /// The super-tet interior: per-axis lower bounds and the upper bound on
    /// the coordinate sum (its four face planes).
    domain: (P3, f64),
    pts: Vec<P3>,
    /// Weights per internal vertex (0 for unweighted points).
    wts: Vec<f64>,
    /// True once any point carries a weight: conflicts then use the power
    /// test (with all five weights zero it is still the insphere test).
    weighted: bool,
    /// Per internal vertex, some alive slot containing it (refreshed by
    /// every refill: a vertex on a cavity boundary reappears in the new
    /// tets, so hints never go stale).
    hint: Vec<u32>,
    /// Per internal vertex, the epoch of the last cavity boundary it was on.
    vmark: Vec<u32>,
    tets: Vec<[u32; 4]>,
    /// `nbr[t][i]`: the neighbour across the face opposite vertex `i`,
    /// packed as `slot << 2 | face` (NONE at the super-tet hull).
    nbr: Vec<[u32; 4]>,
    alive: Vec<bool>,
    free: Vec<u32>,
    /// Per slot, the epoch of the last cavity containing it.
    mark: Vec<u32>,
    epoch: u32,
    /// Walk start.
    last: u32,
    cavity: Vec<u32>,
    /// Cavity boundary faces as (cavity slot, face).
    boundary: Vec<(u32, u8)>,
    new_tets: Vec<u32>,
    /// Edge of a boundary face -> the new tet face through it and the new
    /// point, for wiring the new tets to each other.
    links: rustc_hash::FxHashMap<(u32, u32), u32>,
}

impl Triangulation {
    /// A triangulation whose super-tet comfortably encloses the box; every
    /// inserted point must lie inside it.
    pub fn enclosing(lo: P3, hi: P3) -> Triangulation {
        let c: P3 = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let d = (0..3).map(|k| hi[k] - lo[k]).fold(1.0_f64, f64::max);
        let big = 64.0 * d;
        let pts = vec![
            [c[0] - big, c[1] - big, c[2] - big],
            [c[0] + 3.0 * big, c[1] - big, c[2] - big],
            [c[0] - big, c[1] + 3.0 * big, c[2] - big],
            [c[0] - big, c[1] - big, c[2] + 3.0 * big],
        ];
        let mut seed = [0u32, 1, 2, 3];
        if orient(&pts, 0, 1, 2, 3) == Sign::Negative {
            seed.swap(2, 3);
        }
        Triangulation {
            domain: (
                std::array::from_fn(|k| c[k] - big),
                c[0] + c[1] + c[2] + big,
            ),
            pts,
            wts: vec![0.0; 4],
            weighted: false,
            hint: vec![0; 4],
            vmark: vec![0; 4],
            tets: vec![seed],
            nbr: vec![[NONE; 4]],
            alive: vec![true],
            free: Vec::new(),
            mark: vec![0],
            epoch: 0,
            last: 0,
            cavity: Vec::new(),
            boundary: Vec::new(),
            new_tets: Vec::new(),
            links: rustc_hash::FxHashMap::default(),
        }
    }

    /// Number of inserted points.
    pub fn len(&self) -> usize {
        self.pts.len() - 4
    }

    /// True if no point was inserted yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Position of an inserted point.
    pub fn point(&self, v: usize) -> P3 {
        self.pts[v + 4]
    }

    /// Weight of an inserted point (0 when unweighted).
    pub fn weight(&self, v: usize) -> f64 {
        self.wts[v + 4]
    }

    /// Number of slots ever allocated (alive or free).
    pub fn slot_count(&self) -> usize {
        self.tets.len()
    }

    /// The tet in `slot` as public vertex indices; `None` when the slot is
    /// dead or the tet has a super-tet corner.
    pub fn tet_at(&self, slot: u32) -> Option<[usize; 4]> {
        let t = self.tets[slot as usize];
        if !self.alive[slot as usize] || t.iter().any(|&v| v < 4) {
            return None;
        }
        Some(t.map(|v| (v - 4) as usize))
    }

    /// Every real tet with its slot.
    pub fn tets_with_slots(&self) -> Vec<(u32, [usize; 4])> {
        (0..self.tets.len() as u32)
            .filter_map(|s| self.tet_at(s).map(|t| (s, t)))
            .collect()
    }

    /// Every real tet, as public vertex indices.
    pub fn tets(&self) -> Vec<[usize; 4]> {
        (0..self.tets.len() as u32)
            .filter_map(|s| self.tet_at(s))
            .collect()
    }

    /// The slot across the face of `slot` opposite its vertex `i` (`None`
    /// at the super-tet hull).
    pub fn neighbor_at(&self, slot: u32, i: usize) -> Option<u32> {
        let nb = self.nbr[slot as usize][i];
        (nb != NONE).then(|| unpack(nb).0)
    }

    /// The slot and face across the face of `slot` opposite its vertex `i`
    /// (`None` at the super-tet hull).
    pub fn neighbor_face(&self, slot: u32, i: usize) -> Option<(u32, usize)> {
        let nb = self.nbr[slot as usize][i];
        (nb != NONE).then(|| unpack(nb))
    }

    /// Vertices of a (possibly dead) slot, `None` for super-tet corners.
    pub fn verts_of_slot(&self, slot: u32) -> [Option<usize>; 4] {
        self.tets[slot as usize].map(|v| (v >= 4).then(|| (v - 4) as usize))
    }

    /// The slots the last successful insert created.
    pub fn last_created(&self) -> &[u32] {
        &self.new_tets
    }

    fn inside(&self, p: P3) -> bool {
        (0..3).all(|k| p[k] > self.domain.0[k]) && p[0] + p[1] + p[2] < self.domain.1
    }

    fn locate_scan(&self, p: u32) -> u32 {
        (0..self.tets.len() as u32)
            .find(|&s| {
                let t = self.tets[s as usize];
                self.alive[s as usize]
                    && (0..4).all(|i| {
                        let f = face(t, i);
                        orient(&self.pts, f[0], f[1], f[2], p) != Sign::Negative
                    })
            })
            .expect("point must lie inside the super-tet")
    }

    /// Visibility walk from the last touched tet; a scan when the walk
    /// degenerates.
    fn locate(&self, p: u32) -> u32 {
        let mut cur = self.last;
        if !self.alive[cur as usize] {
            return self.locate_scan(p);
        }
        let mut prev = NONE;
        let mut steps = 0usize;
        'walk: loop {
            steps += 1;
            if steps > self.tets.len() + 64 {
                return self.locate_scan(p);
            }
            let t = self.tets[cur as usize];
            let mut back = false;
            for i in 0..4 {
                let f = face(t, i);
                if orient(&self.pts, f[0], f[1], f[2], p) == Sign::Negative {
                    let nb = self.nbr[cur as usize][i];
                    if nb == NONE {
                        return self.locate_scan(p);
                    }
                    let (next, _) = unpack(nb);
                    if next != prev {
                        prev = cur;
                        cur = next;
                        continue 'walk;
                    }
                    back = true;
                }
            }
            // Only the way back is negative: degenerate ping-pong.
            return if back { self.locate_scan(p) } else { cur };
        }
    }

    fn alloc(&mut self, t: [u32; 4]) -> u32 {
        if let Some(s) = self.free.pop() {
            self.tets[s as usize] = t;
            self.nbr[s as usize] = [NONE; 4];
            self.alive[s as usize] = true;
            self.mark[s as usize] = 0;
            s
        } else {
            self.tets.push(t);
            self.nbr.push([NONE; 4]);
            self.alive.push(true);
            self.mark.push(0);
            (self.tets.len() - 1) as u32
        }
    }

    /// The conflict test of `p` against tet `t`: the strict insphere test,
    /// or with weights the power test. Positive means `p` removes the tet.
    fn conflict(&self, t: [u32; 4], p: u32) -> Sign {
        let w = |i: u32| self.wts[i as usize];
        if !self.weighted || (w(p) == 0.0 && t.iter().all(|&i| w(i) == 0.0)) {
            return insphere(&self.pts, t, p);
        }
        let q = |i: u32| self.pts[i as usize];
        rapidmesh_exact::power_test3d(
            q(t[0]),
            w(t[0]),
            q(t[1]),
            w(t[1]),
            q(t[2]),
            w(t[2]),
            q(t[3]),
            w(t[3]),
            q(p),
            w(p),
        )
    }

    fn push_point(&mut self, p: P3, w: f64) -> u32 {
        let v = self.pts.len() as u32;
        self.pts.push(p);
        self.wts.push(w);
        self.hint.push(NONE);
        self.vmark.push(0);
        v
    }

    fn pop_point(&mut self) {
        self.pts.pop();
        self.wts.pop();
        self.hint.pop();
        self.vmark.pop();
    }

    /// Grows the cavity of the internal vertex `p` into `self.cavity` and
    /// its boundary into `self.boundary`. `false` (cavity undefined) when a
    /// cavity vertex lies closer to `p` than `min_dist2` allows (the nearest
    /// neighbour of `p` is always among them), when the cavity would exceed
    /// `max_cavity` tets, when `p` is hidden by the weights, or when a cavity
    /// vertex would lose its whole star (a near-duplicate of `p`).
    fn cavity_of(&mut self, p: u32, min_dist2: f64, max_cavity: usize) -> bool {
        let start = self.locate(p);
        let x = self.pts[p as usize];
        let too_close = |pts: &[P3], t: [u32; 4]| {
            t.iter().any(|&v| {
                v >= 4 && {
                    let q = pts[v as usize];
                    (0..3).map(|k| (x[k] - q[k]).powi(2)).sum::<f64>() < min_dist2
                }
            })
        };
        if too_close(&self.pts, self.tets[start as usize]) {
            return false;
        }
        // Even the tet containing a hidden point is not in conflict with it.
        if self.weighted && self.conflict(self.tets[start as usize], p) != Sign::Positive {
            return false;
        }
        self.epoch += 1;
        let epoch = self.epoch;
        self.cavity.clear();
        self.cavity.push(start);
        self.mark[start as usize] = epoch;
        let mut head = 0;
        while head < self.cavity.len() {
            let t = self.cavity[head];
            head += 1;
            for i in 0..4 {
                let nb = self.nbr[t as usize][i];
                if nb == NONE {
                    continue;
                }
                let (n, _) = unpack(nb);
                if self.mark[n as usize] == epoch {
                    continue;
                }
                if self.conflict(self.tets[n as usize], p) == Sign::Positive {
                    if too_close(&self.pts, self.tets[n as usize])
                        || self.cavity.len() >= max_cavity
                    {
                        return false;
                    }
                    self.mark[n as usize] = epoch;
                    self.cavity.push(n);
                }
            }
        }
        // Star-shape repair: absorb neighbours across faces that do not see
        // `p` strictly (on-face points, cospherical clusters).
        loop {
            let mut grew = false;
            let mut idx = 0;
            while idx < self.cavity.len() {
                let t = self.cavity[idx];
                idx += 1;
                let tv = self.tets[t as usize];
                for i in 0..4 {
                    let nb = self.nbr[t as usize][i];
                    if nb != NONE && self.mark[unpack(nb).0 as usize] == epoch {
                        continue;
                    }
                    let f = face(tv, i);
                    if orient(&self.pts, f[0], f[1], f[2], p) != Sign::Positive {
                        assert!(nb != NONE, "cavity reached the super-tet hull");
                        let (n, _) = unpack(nb);
                        if too_close(&self.pts, self.tets[n as usize]) {
                            return false;
                        }
                        self.mark[n as usize] = epoch;
                        self.cavity.push(n);
                        grew = true;
                    }
                }
            }
            if !grew {
                break;
            }
        }
        self.boundary.clear();
        for ci in 0..self.cavity.len() {
            let t = self.cavity[ci];
            for i in 0..4 {
                let nb = self.nbr[t as usize][i];
                if nb == NONE || self.mark[unpack(nb).0 as usize] != epoch {
                    self.boundary.push((t, i as u8));
                    for w in face(self.tets[t as usize], i) {
                        self.vmark[w as usize] = epoch;
                    }
                }
            }
        }
        // Every cavity vertex must stay on the boundary, or it would detach.
        self.cavity.iter().all(|&t| {
            self.tets[t as usize]
                .iter()
                .all(|&w| self.vmark[w as usize] == epoch)
        })
    }

    /// Cones `p` to the cavity boundary, wires the new tets and retires the
    /// cavity. Returns the public index of `p`.
    fn refill(&mut self, p: u32) -> usize {
        self.links.clear();
        self.new_tets.clear();
        for bi in 0..self.boundary.len() {
            let (t, fi) = self.boundary[bi];
            let outside = self.nbr[t as usize][fi as usize];
            let f = face(self.tets[t as usize], fi as usize);
            let nt = self.alloc([f[0], f[1], f[2], p]);
            self.new_tets.push(nt);
            // Across the base face (opposite `p`, local 3).
            self.nbr[nt as usize][3] = outside;
            if outside != NONE {
                let (o, of) = unpack(outside);
                self.nbr[o as usize][of] = pack(nt, 3);
            }
            // The other faces contain `p` and one base edge.
            for e in 0..3 {
                let (a, b) = (f[e], f[(e + 1) % 3]);
                // Opposite the third base vertex, at local (e + 2) % 3.
                let here = pack(nt, (e + 2) % 3);
                match self.links.entry((a.min(b), a.max(b))) {
                    std::collections::hash_map::Entry::Occupied(o) => {
                        let there = o.remove();
                        let (ot, of) = unpack(there);
                        self.nbr[nt as usize][(e + 2) % 3] = there;
                        self.nbr[ot as usize][of] = here;
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        v.insert(here);
                    }
                }
            }
        }
        for ci in 0..self.cavity.len() {
            let t = self.cavity[ci];
            self.alive[t as usize] = false;
            self.free.push(t);
        }
        for &nt in &self.new_tets {
            for v in self.tets[nt as usize] {
                self.hint[v as usize] = nt;
            }
        }
        self.last = *self.new_tets.last().expect("a cavity has boundary faces");
        (p - 4) as usize
    }

    /// Inserts a point; panics when it duplicates a vertex.
    pub fn insert(&mut self, point: P3) -> usize {
        self.try_insert(point)
            .expect("point duplicates a vertex (it would detach its star)")
    }

    /// Inserts a point, `None` (nothing changed) when it would detach an
    /// existing vertex's star (a near-duplicate).
    pub fn try_insert(&mut self, point: P3) -> Option<usize> {
        self.try_insert_weighted(point, 0.0)
    }

    /// Inserts a weighted point: the triangulation stays the regular
    /// triangulation of all points (unweighted ones weigh 0). `None`, with
    /// nothing changed, when the point is hidden or would hide a vertex.
    pub fn try_insert_weighted(&mut self, point: P3, weight: f64) -> Option<usize> {
        let p = self.push_point(point, weight);
        let was = self.weighted;
        self.weighted |= weight != 0.0;
        if !self.cavity_of(p, -1.0, usize::MAX) {
            self.pop_point();
            self.weighted = was;
            return None;
        }
        Some(self.refill(p))
    }

    /// Inserts `point` unless a vertex lies closer than `sqrt(min_dist2)`,
    /// its cavity exceeds `max_cavity` tets, or `keep` rejects one of the
    /// interior faces the insertion would remove (as a slot and the face
    /// opposite its vertex; faces touching the super-tet are not offered).
    /// `None` leaves the triangulation untouched.
    pub fn insert_guarded(
        &mut self,
        point: P3,
        min_dist2: f64,
        max_cavity: usize,
        keep: Option<&mut dyn FnMut(u32, usize) -> bool>,
    ) -> Option<usize> {
        if !self.inside(point) {
            return None;
        }
        let p = self.push_point(point, 0.0);
        if !self.cavity_of(p, min_dist2, max_cavity) {
            self.pop_point();
            return None;
        }
        if let Some(keep) = keep {
            let epoch = self.epoch;
            for ci in 0..self.cavity.len() {
                let t = self.cavity[ci];
                let tv = self.tets[t as usize];
                for i in 0..4 {
                    // Interior: the neighbour is in the cavity as well.
                    let nb = self.nbr[t as usize][i];
                    if nb == NONE || self.mark[unpack(nb).0 as usize] != epoch {
                        continue;
                    }
                    if face(tv, i).iter().any(|&v| v < 4) {
                        continue;
                    }
                    if !keep(t, i) {
                        self.pop_point();
                        return None;
                    }
                }
            }
        }
        Some(self.refill(p))
    }

    /// The faces `point` would be coned to, appended to `out` (public
    /// indices, wound so `(face, point)` is positive; faces on the super-tet
    /// skipped), without inserting. `false`, with `out` untouched, when an
    /// insert with the same guards would be declined.
    pub fn probe(
        &mut self,
        point: P3,
        min_dist2: f64,
        max_cavity: usize,
        out: &mut Vec<[usize; 3]>,
    ) -> bool {
        if !self.inside(point) {
            return false;
        }
        let p = self.push_point(point, 0.0);
        let ok = self.cavity_of(p, min_dist2, max_cavity);
        if ok {
            for &(t, fi) in &self.boundary {
                let f = face(self.tets[t as usize], fi as usize);
                if f.iter().all(|&v| v >= 4) {
                    out.push(f.map(|v| (v - 4) as usize));
                }
            }
        }
        self.pop_point();
        ok
    }
}

/// Exact Delaunay tetrahedralization of `points` (no duplicates; coplanar
/// input yields no tets), as positively oriented index quadruples.
pub fn tetrahedralize(points: &[P3]) -> Vec<[usize; 4]> {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let mut t = Triangulation::enclosing(lo, hi);
    for &p in points {
        t.insert(p);
    }
    t.tets()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn sorted(mut t: Vec<[usize; 4]>) -> Vec<[usize; 4]> {
        for x in &mut t {
            x.sort_unstable();
        }
        t.sort_unstable();
        t
    }

    /// Every neighbour link is mutual, points at the same face, and the
    /// two tets share exactly that face's vertices.
    fn check_links(tr: &Triangulation) {
        for s in 0..tr.tets.len() {
            if !tr.alive[s] {
                continue;
            }
            for i in 0..4 {
                let nb = tr.nbr[s][i];
                if nb == NONE {
                    continue;
                }
                let (o, of) = unpack(nb);
                assert!(tr.alive[o as usize], "link to a dead slot");
                assert_eq!(tr.nbr[o as usize][of], pack(s as u32, i), "link not mutual");
                let mut a = face(tr.tets[s], i);
                let mut b = face(tr.tets[o as usize], of);
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b, "linked faces differ");
            }
        }
    }

    #[test]
    fn zero_weights_give_the_delaunay_triangulation() {
        let mut r = rng(7);
        let pts: Vec<P3> = (0..300).map(|_| [r(), r(), r()]).collect();
        let mut a = Triangulation::enclosing([0.0; 3], [1.0; 3]);
        let mut b = Triangulation::enclosing([0.0; 3], [1.0; 3]);
        // One weight far away puts `b` into weighted mode without touching
        // the unit cube's triangulation.
        a.try_insert([10.0, 10.0, 10.0]).unwrap();
        b.try_insert_weighted([10.0, 10.0, 10.0], 1e-6).unwrap();
        for &p in &pts {
            a.try_insert(p).unwrap();
            b.try_insert(p).unwrap();
        }
        assert_eq!(sorted(a.tets()), sorted(b.tets()));
        check_links(&b);
    }

    #[test]
    fn weighted_triangulations_are_regular() {
        let mut r = rng(11);
        let mut tr = Triangulation::enclosing([0.0; 3], [1.0; 3]);
        let mut live: Vec<usize> = Vec::new();
        let mut hidden = 0;
        for i in 0..400 {
            let p = [r(), r(), r()];
            let w = if i % 3 == 0 { (0.2 * r()).powi(2) } else { 0.0 };
            match tr.try_insert_weighted(p, w) {
                Some(v) => live.push(v),
                None => hidden += 1,
            }
        }
        assert!(hidden > 0 && live.len() > 300, "hidden {hidden}");
        check_links(&tr);
        let tets = tr.tets();
        for t in &tets {
            for &v in &live {
                if t.contains(&v) {
                    continue;
                }
                let s = rapidmesh_exact::power_test3d(
                    tr.point(t[0]),
                    tr.weight(t[0]),
                    tr.point(t[1]),
                    tr.weight(t[1]),
                    tr.point(t[2]),
                    tr.weight(t[2]),
                    tr.point(t[3]),
                    tr.weight(t[3]),
                    tr.point(v),
                    tr.weight(v),
                );
                assert_ne!(s, Sign::Positive, "tet {t:?} not regular at {v}");
            }
        }
        let mut used = vec![false; tr.len()];
        for t in &tets {
            for &v in t {
                used[v] = true;
            }
        }
        for &v in &live {
            assert!(used[v], "vertex {v} lost");
        }
    }

    #[test]
    fn probing_predicts_the_insert_and_changes_nothing() {
        let mut r = rng(3);
        let mut tr = Triangulation::enclosing([0.0; 3], [1.0; 3]);
        for _ in 0..200 {
            tr.try_insert([r(), r(), r()]).unwrap();
        }
        for _ in 0..50 {
            let p = [r(), r(), r()];
            let before = tr.tets();
            let mut faces = Vec::new();
            assert!(tr.probe(p, -1.0, usize::MAX, &mut faces));
            assert_eq!(tr.tets(), before, "probe changed the triangulation");
            let v = tr.try_insert(p).unwrap();
            let want = sorted(faces.iter().map(|f| [f[0], f[1], f[2], v]).collect());
            let got = sorted(tr.tets().into_iter().filter(|t| t.contains(&v)).collect());
            assert_eq!(got, want);
        }
        check_links(&tr);
    }

    #[test]
    fn guards_decline_without_a_trace() {
        let mut r = rng(5);
        let mut tr = Triangulation::enclosing([0.0; 3], [1.0; 3]);
        for _ in 0..100 {
            tr.try_insert([r(), r(), r()]).unwrap();
        }
        let before = tr.tets();
        let near = {
            let q = tr.point(17);
            [q[0] + 1e-6, q[1], q[2]]
        };
        assert!(tr.insert_guarded(near, 1e-8, usize::MAX, None).is_none());
        let mut veto = |_: u32, _: usize| false;
        assert!(tr
            .insert_guarded([0.5, 0.5, 0.5], -1.0, usize::MAX, Some(&mut veto))
            .is_none());
        assert!(tr.insert_guarded([0.5, 0.5, 0.5], -1.0, 3, None).is_none());
        assert_eq!(tr.tets(), before);
        assert_eq!(tr.len(), 100);
        assert!(tr
            .insert_guarded([0.5, 0.5, 0.5], -1.0, usize::MAX, None)
            .is_some());
        check_links(&tr);
    }
}
