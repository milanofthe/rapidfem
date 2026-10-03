//! The Delaunay tetrahedralization of a point set under the symbolic
//! perturbation of [`crate::predicates`]: unique, whatever the ties, so the
//! constrained stage and its segment check agree on every cospherical
//! configuration.
//!
//! Bowyer and Watson inside an enclosing tet: the tets whose perturbed
//! sphere holds the new point form its cavity, grown through the
//! neighbours; a boundary face the point does not see strictly from inside
//! takes the tet behind it too, so the cavity stays star-shaped.

use crate::predicates::{inside, orient, P3};
use crate::volume::tets::{Tets, FACE, NONE};
use rapidmesh_geom::vec3::bbox;

/// Steps of a point location walk per tet before it gives up and scans
/// (a stochastic walk ends long before; the scan is the guarantee).
const WALK_STEPS_PER_TET: usize = 4;

/// A Delaunay tetrahedralization. Vertices `0..4` are the corners of the
/// enclosing tet; point `i` of the input is vertex `i + 4`.
pub struct Delaunay {
    pts: Vec<P3>,
    t: Tets,
    last: u32,
    /// Buffers of an insertion, kept between them: the cavity, its
    /// boundary faces and the new tets.
    cavity: Vec<u32>,
    boundary: Vec<(u32, usize)>,
    made: Vec<u32>,
    /// The box of the first points; later ones must lie in it.
    lo: P3,
    hi: P3,
}

impl Delaunay {
    /// The tetrahedralization of `points` (distinct).
    pub fn new(points: &[P3]) -> Delaunay {
        let (lo, hi) = bbox(points);
        let c: P3 = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let r = (0..3).map(|k| hi[k] - lo[k]).fold(1e-300, f64::max) * 50.0;
        let mut d = Delaunay {
            pts: vec![
                [c[0] - r, c[1] - r, c[2] - r],
                [c[0] + 3.0 * r, c[1] - r, c[2] - r],
                [c[0] - r, c[1] + 3.0 * r, c[2] - r],
                [c[0] - r, c[1] - r, c[2] + 3.0 * r],
            ],
            t: Tets::default(),
            last: 0,
            cavity: Vec::new(),
            boundary: Vec::new(),
            made: Vec::new(),
            lo,
            hi,
        };
        let mut t0 = [0, 1, 2, 3];
        if orient(d.pts[0], d.pts[1], d.pts[2], d.pts[3]) < 0 {
            t0.swap(2, 3);
        }
        d.t.alloc(t0);
        // Insert along a space-filling order, so each walk is short.
        let mut order: Vec<usize> = (0..points.len()).collect();
        order.sort_by_key(|&i| morton_key(points[i], lo, hi));
        d.pts.extend_from_slice(points);
        for i in order {
            d.insert(i as u32 + 4);
        }
        d
    }

    /// The tetrahedralization of its points without the input points
    /// `gone` and with `points` added after the rest: the one made afresh
    /// (the perturbation is a property of the points). Returns, for each
    /// point of the new numbering before the added ones, its index before;
    /// `None` (and the tetrahedralization unusable) where an added point
    /// falls outside the box the enclosing tet was made for, or a removal
    /// cannot close its hole, so the caller makes it afresh.
    pub fn update(&mut self, gone: &[u32], points: &[P3]) -> Option<Vec<u32>> {
        let inside_box = |p: &P3| (0..3).all(|k| p[k] >= self.lo[k] && p[k] <= self.hi[k]);
        if !points.iter().all(inside_box) {
            return None;
        }
        for &g in gone {
            if !self.remove(g + 4) {
                return None;
            }
        }
        let kept = self.compact(gone);
        let base = self.pts.len();
        let mut order: Vec<usize> = (0..points.len()).collect();
        let (lo, hi) = (self.lo, self.hi);
        order.sort_by_key(|&i| morton_key(points[i], lo, hi));
        self.pts.extend_from_slice(points);
        for i in order {
            self.insert((base + i) as u32);
        }
        Some(kept)
    }

    /// Removes vertex `p`: its star goes, and the hole is filled with the
    /// tets of the tetrahedralization of its link whose spheres held `p`,
    /// which are the tets the points without `p` have there (Devillers).
    /// That holds where the star is Delaunay; an insertion that took a tet
    /// behind a face its point lies on (to keep the cavity star-shaped)
    /// can leave one that is not, and then the fill does not close the
    /// hole: `false`, nothing changed.
    fn remove(&mut self, p: u32) -> bool {
        let x = self.pts[p as usize];
        let start = self.locate(p);
        if !self.t.tets[start as usize].contains(&p) {
            return false;
        }
        // The star, through the faces at p; its faces opposite p bound the hole.
        let mut star = vec![start];
        let epoch = self.t.next_epoch();
        self.t.mark[start as usize] = epoch;
        let mut at = 0;
        let mut hole: rustc_hash::FxHashMap<[u32; 3], u32> = rustc_hash::FxHashMap::default();
        while at < star.len() {
            let t = star[at];
            at += 1;
            for i in 0..4 {
                let nb = self.t.nbr[t as usize][i];
                if self.t.tets[t as usize][i] == p {
                    let mut f = self.face(t, i);
                    f.sort_unstable();
                    hole.insert(f, nb);
                    continue;
                }
                let n = nb >> 2;
                if nb != NONE && self.t.mark[n as usize] != epoch {
                    self.t.mark[n as usize] = epoch;
                    star.push(n);
                }
            }
        }
        // The link and its tetrahedralization; the tets in conflict with p.
        let mut link: Vec<u32> = hole.keys().flatten().copied().collect();
        link.sort_unstable();
        link.dedup();
        let local = Delaunay::new(
            &link
                .iter()
                .map(|&v| self.pts[v as usize])
                .collect::<Vec<_>>(),
        );
        let fill: Vec<[u32; 4]> = local
            .tets()
            .into_iter()
            .map(|t| t.map(|v| link[v as usize]))
            .filter(|t| inside(t.map(|v| self.pts[v as usize]), x))
            .collect();
        if !closes(&fill, &hole) {
            return false;
        }
        self.t.kill(&star);
        // The new tets, each face glued to the tet across the hole's side or
        // to the new one sharing it.
        if let Some(last) = self.t.fill(fill, &hole) {
            self.last = last;
        }
        true
    }

    /// Renumbers the input points without those in `gone` (removed),
    /// keeping their order; returns the old index of each.
    fn compact(&mut self, gone: &[u32]) -> Vec<u32> {
        let gone: rustc_hash::FxHashSet<u32> = gone.iter().copied().collect();
        let n = self.pts.len() - 4;
        let mut new_of = vec![NONE; n];
        let mut kept = Vec::with_capacity(n - gone.len().min(n));
        for i in 0..n as u32 {
            if !gone.contains(&i) {
                new_of[i as usize] = kept.len() as u32;
                kept.push(i);
            }
        }
        let corners: Vec<P3> = self.pts[..4].to_vec();
        self.pts = corners
            .into_iter()
            .chain(kept.iter().map(|&i| self.pts[i as usize + 4]))
            .collect();
        for (t, &alive) in self.t.tets.iter_mut().zip(&self.t.alive) {
            if alive {
                *t = t.map(|v| {
                    if v < 4 {
                        v
                    } else {
                        new_of[(v - 4) as usize] + 4
                    }
                });
            }
        }
        kept
    }

    /// The tets without a corner of the enclosing tet, as input point
    /// indices, positively oriented.
    pub fn tets(&self) -> Vec<[u32; 4]> {
        self.t
            .tets
            .iter()
            .zip(&self.t.alive)
            .filter(|(t, &a)| a && t.iter().all(|&v| v >= 4))
            .map(|(t, _)| t.map(|v| v - 4))
            .collect()
    }

    /// The tets without a corner of the enclosing tet (input indices,
    /// positive) and, per tet and face (opposite its vertex `i`), the index
    /// of the tet across it (`u32::MAX` for none or an outer one).
    pub fn tets_and_neighbours(&self) -> (Vec<[u32; 4]>, Vec<[u32; 4]>) {
        let mut index = vec![u32::MAX; self.t.tets.len()];
        let mut out = Vec::new();
        for (t, (tv, &alive)) in self.t.tets.iter().zip(&self.t.alive).enumerate() {
            if alive && tv.iter().all(|&v| v >= 4) {
                index[t] = out.len() as u32;
                out.push(tv.map(|v| v - 4));
            }
        }
        let nbrs = self
            .t
            .tets
            .iter()
            .zip(&self.t.alive)
            .enumerate()
            .filter(|(t, _)| index[*t] != u32::MAX)
            .map(|(t, _)| {
                self.t.nbr[t].map(|nb| {
                    if nb == NONE {
                        u32::MAX
                    } else {
                        index[(nb >> 2) as usize]
                    }
                })
            })
            .collect();
        (out, nbrs)
    }

    /// Every face without a corner of the enclosing tet, as a directed
    /// triangle (input indices, rotated to start at its smallest) with the
    /// vertex of the tet on its positive side.
    pub fn apexes(&self) -> rustc_hash::FxHashMap<[u32; 3], u32> {
        let mut out = rustc_hash::FxHashMap::default();
        for (t, &alive) in self.t.tets.iter().zip(&self.t.alive) {
            if !alive {
                continue;
            }
            for (i, f) in FACE.iter().enumerate() {
                let tri = f.map(|k| t[k]);
                if t[i] < 4 || tri.iter().any(|&v| v < 4) {
                    continue;
                }
                let tri = tri.map(|v| v - 4);
                let r = (0..3).min_by_key(|&k| tri[k]).unwrap_or(0);
                out.insert([tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]], t[i] - 4);
            }
        }
        out
    }

    fn keyed(&self, t: [u32; 4], p: u32) -> bool {
        let q = |v: u32| self.pts[v as usize];
        inside(t.map(q), q(p))
    }

    fn face(&self, t: u32, i: usize) -> [u32; 3] {
        self.t.face(t, i)
    }

    /// The tet containing vertex `p` (closed), by a visibility walk.
    /// The tet holding point `p`, by a visibility walk from the last one
    /// made: across a face `p` lies beyond. The face tried first is drawn
    /// at random each step (a stochastic walk, which ends with probability
    /// one in a Delaunay tetrahedralization; a fixed order can cycle around
    /// degenerate tets for ever). The draws are seeded by `p`, so the result
    /// does not change between runs; past `WALK_STEPS_PER_TET` steps per tet
    /// a scan of all tets finds it instead.
    fn locate(&self, p: u32) -> u32 {
        let x = self.pts[p as usize];
        let mut t = self.last;
        let mut rng = (p as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let cap = WALK_STEPS_PER_TET * self.t.alive.len().max(16);
        'walk: for _ in 0..cap {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let first = (rng >> 62) as usize;
            for j in 0..4 {
                let i = (j + first) % 4;
                let f = self.face(t, i).map(|v| self.pts[v as usize]);
                if orient(f[0], f[1], f[2], x) < 0 {
                    let nb = self.t.nbr[t as usize][i];
                    if nb == NONE {
                        return t;
                    }
                    t = nb >> 2;
                    continue 'walk;
                }
            }
            return t;
        }
        self.scan(x).unwrap_or(t)
    }

    /// A live tet holding `x` (on its faces included), by trying them all.
    fn scan(&self, x: P3) -> Option<u32> {
        (0..self.t.alive.len() as u32).find(|&t| {
            self.t.alive[t as usize]
                && (0..4).all(|i| {
                    let f = self.face(t, i).map(|v| self.pts[v as usize]);
                    orient(f[0], f[1], f[2], x) >= 0
                })
        })
    }

    fn insert(&mut self, p: u32) {
        let start = self.locate(p);
        let epoch = self.t.next_epoch();
        let mut cavity = std::mem::take(&mut self.cavity);
        cavity.clear();
        cavity.push(start);
        self.t.mark[start as usize] = epoch;
        let x = self.pts[p as usize];
        let mut at = 0;
        while at < cavity.len() {
            let t = cavity[at];
            at += 1;
            for i in 0..4 {
                let nb = self.t.nbr[t as usize][i];
                if nb == NONE {
                    continue;
                }
                let n = nb >> 2;
                if self.t.mark[n as usize] == epoch {
                    continue;
                }
                let f = self.face(t, i).map(|v| self.pts[v as usize]);
                // `p` must see the face strictly from inside the cavity
                // (orientation negative seen from `t`'s face frame means
                // outside); otherwise the tet behind joins.
                let sees = orient(f[0], f[1], f[2], x) > 0;
                if !sees || self.keyed(self.t.tets[n as usize], p) {
                    self.t.mark[n as usize] = epoch;
                    cavity.push(n);
                }
            }
        }
        // Cone p to the boundary faces.
        let mut boundary = std::mem::take(&mut self.boundary);
        boundary.clear();
        for &t in &cavity {
            for i in 0..4 {
                let nb = self.t.nbr[t as usize][i];
                if nb == NONE || self.t.mark[(nb >> 2) as usize] != epoch {
                    boundary.push((t, i));
                }
            }
        }
        let mut made = std::mem::take(&mut self.made);
        made.clear();
        self.t.cone(&boundary, p, &mut made);
        self.t.kill(&cavity);
        debug_assert!(!made.is_empty(), "a cavity has faces");
        self.last = made[made.len() - 1];
        self.cavity = cavity;
        self.boundary = boundary;
        self.made = made;
    }
}

/// Whether the tets `fill` close the hole bounded by the faces `hole`
/// (sorted): each of those taken once, every other face of theirs twice.
fn closes(fill: &[[u32; 4]], hole: &rustc_hash::FxHashMap<[u32; 3], u32>) -> bool {
    let mut count: rustc_hash::FxHashMap<[u32; 3], u32> = rustc_hash::FxHashMap::default();
    for t in fill {
        for f in FACE {
            let mut f = f.map(|k| t[k]);
            f.sort_unstable();
            *count.entry(f).or_default() += 1;
        }
    }
    hole.keys().all(|f| count.get(f) == Some(&1))
        && count
            .iter()
            .all(|(f, &n)| n == 2 || (n == 1 && hole.contains_key(f)))
}

/// The Morton code of `p` on a 1024 grid over the box `lo..hi`.
fn morton_key(p: P3, lo: P3, hi: P3) -> u64 {
    let q = |x: f64, k: usize| {
        (((x - lo[k]) / (hi[k] - lo[k]).max(1e-300)) * 1023.0).clamp(0.0, 1023.0) as u64
    };
    morton(q(p[0], 0), q(p[1], 1), q(p[2], 2))
}

fn morton(x: u64, y: u64, z: u64) -> u64 {
    let spread = |mut v: u64| {
        v &= 0x3ff;
        v = (v | v << 16) & 0x030000ff;
        v = (v | v << 8) & 0x0300f00f;
        v = (v | v << 4) & 0x030c30c3;
        (v | v << 2) & 0x09249249
    };
    spread(x) | spread(y) << 1 | spread(z) << 2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tet as its corners' coordinates, sorted: a tetrahedralization
    /// as a property of its points, whatever their numbering.
    fn shape(d: &Delaunay) -> Vec<[[u64; 3]; 4]> {
        let n = d.pts.len() - 4;
        let pts = &d.pts[4..4 + n];
        let mut out: Vec<[[u64; 3]; 4]> = d
            .tets()
            .into_iter()
            .map(|t| {
                let mut c = t.map(|v| pts[v as usize].map(f64::to_bits));
                c.sort_unstable();
                c
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// A fill closes a hole when it takes each of the hole's faces once and
    /// pairs every other face of its own; a missing or an extra tet does not.
    #[test]
    fn a_fill_closes_its_hole_or_says_so() {
        // The hole of a removed apex over the square 0 1 2 3: four faces.
        let hole: rustc_hash::FxHashMap<[u32; 3], u32> = [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [0, 3, 4],
            [0, 1, 2],
            [0, 2, 3],
        ]
        .into_iter()
        .map(|f| (f, NONE))
        .collect();
        let fill = [[0, 1, 2, 4], [0, 2, 3, 4]];
        assert!(closes(&fill, &hole));
        assert!(!closes(&fill[..1], &hole));
        assert!(!closes(&[fill[0], fill[1], [0, 1, 3, 4]], &hole));
    }

    /// Removing points and adding others makes the tetrahedralization made
    /// afresh, on a grid full of cospherical ties.
    #[test]
    fn an_update_is_the_tetrahedralization_made_afresh() {
        let n = 5;
        let mut grid: Vec<P3> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    grid.push([i as f64, j as f64, k as f64]);
                }
            }
        }
        let mut d = Delaunay::new(&grid);
        let gone: Vec<u32> = (0..grid.len() as u32).filter(|i| i % 7 == 3).collect();
        let added: Vec<P3> = (0..20)
            .map(|i| {
                let t = i as f64;
                [
                    0.5 + (t * 0.37) % 3.0,
                    0.5 + (t * 0.61) % 3.0,
                    0.25 + (t * 0.83) % 3.5,
                ]
            })
            .collect();
        let kept = d.update(&gone, &added).expect("inside the box");
        let rest: Vec<P3> = kept
            .iter()
            .map(|&i| grid[i as usize])
            .chain(added.iter().copied())
            .collect();
        assert_eq!(shape(&d), shape(&Delaunay::new(&rest)));
    }

    /// The volume of positive tets (Shewchuk's orientation: the triple
    /// product of `b - a`, `c - a`, `d - a` is negative).
    fn volume(pts: &[P3], tets: &[[u32; 4]]) -> f64 {
        tets.iter()
            .map(|t| {
                let [a, b, c, d] = t.map(|v| pts[v as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
                (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
                    + u[2] * (v[0] * w[1] - v[1] * w[0]))
                    / -6.0
            })
            .sum()
    }

    /// A grid of points, all of it cospherical in groups: the tets fill its
    /// box exactly, every one positive, and every empty-sphere test between
    /// a tet and a vertex holds under the perturbation.
    #[test]
    fn random_points_fill_their_hull() {
        let mut s = 7u64;
        let mut r = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut pts: Vec<P3> = (0..8)
            .map(|i| [(i & 1) as f64, (i >> 1 & 1) as f64, (i >> 2 & 1) as f64])
            .collect();
        pts.extend((0..200).map(|_| [0.05 + 0.9 * r(), 0.05 + 0.9 * r(), 0.05 + 0.9 * r()]));
        let tets = Delaunay::new(&pts).tets();
        assert!(
            (volume(&pts, &tets) - 1.0).abs() < 1e-9,
            "{}",
            volume(&pts, &tets)
        );
    }

    #[test]
    fn a_degenerate_grid_is_tetrahedralized_exactly() {
        let n = 4;
        let pts: Vec<P3> = (0..n * n * n)
            .map(|i| [(i % n) as f64, (i / n % n) as f64, (i / n / n) as f64])
            .collect();
        let d = Delaunay::new(&pts);
        let tets = d.tets();
        for t in &tets {
            let q = t.map(|v| pts[v as usize]);
            assert_eq!(orient(q[0], q[1], q[2], q[3]), 1);
        }
        let side = (n - 1) as f64;
        let mut bad = 0;
        for t in &tets {
            for v in 0..pts.len() as u32 {
                if t.contains(&v) {
                    continue;
                }
                let q = t.map(|w| pts[w as usize]);
                if inside(q, pts[v as usize]) {
                    bad += 1;
                }
            }
        }
        assert!(
            (volume(&pts, &tets) - side * side * side).abs() < 1e-9,
            "volume {} of {}, {} tets, {bad} spheres not empty",
            volume(&pts, &tets),
            side * side * side,
            tets.len()
        );
        for t in &tets {
            for v in 0..pts.len() as u32 {
                if t.contains(&v) {
                    continue;
                }
                let q = t.map(|w| pts[w as usize]);
                assert!(!inside(q, pts[v as usize]));
            }
        }
    }
}
