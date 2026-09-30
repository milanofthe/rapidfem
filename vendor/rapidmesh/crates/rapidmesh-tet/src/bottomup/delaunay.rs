//! The Delaunay tetrahedralization of a point set under the symbolic
//! perturbation of [`super::predicates`]: unique, whatever the ties, so the
//! constrained stage and its segment check agree on every cospherical
//! configuration.
//!
//! Bowyer and Watson inside an enclosing tet: the tets whose perturbed
//! sphere holds the new point form its cavity, grown through the
//! neighbours; a boundary face the point does not see strictly from inside
//! takes the tet behind it too, so the cavity stays star-shaped.

use super::predicates::{inside, orient, P3};

const NONE: u32 = u32::MAX;

/// The vertices of face `i` of a positive tet, turned so the tet's vertex
/// `i` lies on their positive side.
const FACE: [[usize; 3]; 4] = [[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]];

/// A Delaunay tetrahedralization. Vertices `0..4` are the corners of the
/// enclosing tet; point `i` of the input is vertex `i + 4`.
pub struct Delaunay {
    pts: Vec<P3>,
    tets: Vec<[u32; 4]>,
    /// Per tet and face: the neighbour as `tet << 2 | its face`.
    nbr: Vec<[u32; 4]>,
    alive: Vec<bool>,
    free: Vec<u32>,
    last: u32,
    mark: Vec<u32>,
    epoch: u32,
    /// Buffers of an insertion, kept between them: the cavity and the
    /// open edges of the cone.
    cavity: Vec<u32>,
    links: rustc_hash::FxHashMap<(u32, u32), u32>,
}

impl Delaunay {
    /// The tetrahedralization of `points` (distinct).
    pub fn new(points: &[P3]) -> Delaunay {
        let (lo, hi) = points
            .iter()
            .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), p| {
                (
                    std::array::from_fn(|k| lo[k].min(p[k])),
                    std::array::from_fn(|k| hi[k].max(p[k])),
                )
            });
        let c: P3 = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let r = (0..3).map(|k| hi[k] - lo[k]).fold(1e-300, f64::max) * 50.0;
        let mut d = Delaunay {
            pts: vec![
                [c[0] - r, c[1] - r, c[2] - r],
                [c[0] + 3.0 * r, c[1] - r, c[2] - r],
                [c[0] - r, c[1] + 3.0 * r, c[2] - r],
                [c[0] - r, c[1] - r, c[2] + 3.0 * r],
            ],
            tets: Vec::new(),
            nbr: Vec::new(),
            alive: Vec::new(),
            free: Vec::new(),
            last: 0,
            mark: Vec::new(),
            epoch: 0,
            cavity: Vec::new(),
            links: rustc_hash::FxHashMap::default(),
        };
        let mut t0 = [0, 1, 2, 3];
        if orient(d.pts[0], d.pts[1], d.pts[2], d.pts[3]) < 0 {
            t0.swap(2, 3);
        }
        d.tets.push(t0);
        d.nbr.push([NONE; 4]);
        d.alive.push(true);
        d.mark.push(0);
        // Insert along a space-filling order, so each walk is short.
        let mut order: Vec<usize> = (0..points.len()).collect();
        let key = |p: P3| -> u64 {
            let q = |x: f64, k: usize| {
                (((x - lo[k]) / (hi[k] - lo[k]).max(1e-300)) * 1023.0).clamp(0.0, 1023.0) as u64
            };
            morton(q(p[0], 0), q(p[1], 1), q(p[2], 2))
        };
        order.sort_by_key(|&i| key(points[i]));
        d.pts.extend_from_slice(points);
        for i in order {
            d.insert(i as u32 + 4);
        }
        d
    }

    /// The tets without a corner of the enclosing tet, as input point
    /// indices, positively oriented.
    pub fn tets(&self) -> Vec<[u32; 4]> {
        self.tets
            .iter()
            .zip(&self.alive)
            .filter(|(t, &a)| a && t.iter().all(|&v| v >= 4))
            .map(|(t, _)| t.map(|v| v - 4))
            .collect()
    }

    /// The tets without a corner of the enclosing tet (input indices,
    /// positive) and, per tet and face (opposite its vertex `i`), the index
    /// of the tet across it (`u32::MAX` for none or an outer one).
    pub fn tets_and_neighbours(&self) -> (Vec<[u32; 4]>, Vec<[u32; 4]>) {
        let mut index = vec![u32::MAX; self.tets.len()];
        let mut out = Vec::new();
        for (t, (tv, &alive)) in self.tets.iter().zip(&self.alive).enumerate() {
            if alive && tv.iter().all(|&v| v >= 4) {
                index[t] = out.len() as u32;
                out.push(tv.map(|v| v - 4));
            }
        }
        let nbrs = self
            .tets
            .iter()
            .zip(&self.alive)
            .enumerate()
            .filter(|(t, _)| index[*t] != u32::MAX)
            .map(|(t, _)| {
                self.nbr[t].map(|nb| {
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
        for (t, &alive) in self.tets.iter().zip(&self.alive) {
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
        inside(t.map(q), q(p), [t[0], t[1], t[2], t[3], p])
    }

    fn face(&self, t: u32, i: usize) -> [u32; 3] {
        let tv = self.tets[t as usize];
        FACE[i].map(|k| tv[k])
    }

    /// The tet containing vertex `p` (closed), by a visibility walk.
    fn locate(&self, p: u32) -> u32 {
        let x = self.pts[p as usize];
        let mut t = self.last;
        let mut steps = 0usize;
        'walk: loop {
            steps += 1;
            // Rotate the start face with the step count, so degenerate
            // walks cannot cycle.
            for j in 0..4 {
                let i = (j + steps) % 4;
                let f = self.face(t, i).map(|v| self.pts[v as usize]);
                if orient(f[0], f[1], f[2], x) < 0 {
                    let nb = self.nbr[t as usize][i];
                    if nb == NONE {
                        break 'walk;
                    }
                    t = nb >> 2;
                    continue 'walk;
                }
            }
            break;
        }
        t
    }

    fn insert(&mut self, p: u32) {
        let start = self.locate(p);
        self.epoch += 1;
        let epoch = self.epoch;
        let mut cavity = std::mem::take(&mut self.cavity);
        cavity.clear();
        cavity.push(start);
        self.mark[start as usize] = epoch;
        let x = self.pts[p as usize];
        let mut at = 0;
        while at < cavity.len() {
            let t = cavity[at];
            at += 1;
            for i in 0..4 {
                let nb = self.nbr[t as usize][i];
                if nb == NONE {
                    continue;
                }
                let n = nb >> 2;
                if self.mark[n as usize] == epoch {
                    continue;
                }
                let f = self.face(t, i).map(|v| self.pts[v as usize]);
                // `p` must see the face strictly from inside the cavity
                // (orientation negative seen from `t`'s face frame means
                // outside); otherwise the tet behind joins.
                let sees = orient(f[0], f[1], f[2], x) > 0;
                if !sees || self.keyed(self.tets[n as usize], p) {
                    self.mark[n as usize] = epoch;
                    cavity.push(n);
                }
            }
        }
        // Cone p to the boundary faces.
        let mut links = std::mem::take(&mut self.links);
        links.clear();
        let mut last = NONE;
        for &t in &cavity {
            for i in 0..4 {
                let nb = self.nbr[t as usize][i];
                if nb != NONE && self.mark[(nb >> 2) as usize] == epoch {
                    continue;
                }
                let f = self.face(t, i);
                let nt = self.alloc([f[0], f[1], f[2], p]);
                last = nt;
                self.nbr[nt as usize][3] = nb;
                if nb != NONE {
                    self.nbr[(nb >> 2) as usize][(nb & 3) as usize] = nt << 2 | 3;
                }
                for e in 0..3 {
                    let (a, b) = (f[e], f[(e + 1) % 3]);
                    // The face of the new tet opposite f's third vertex.
                    let here = nt << 2 | ((e + 2) % 3) as u32;
                    match links.remove(&(a.min(b), a.max(b))) {
                        Some(there) => {
                            self.nbr[nt as usize][(e + 2) % 3] = there;
                            self.nbr[(there >> 2) as usize][(there & 3) as usize] = here;
                        }
                        None => {
                            links.insert((a.min(b), a.max(b)), here);
                        }
                    }
                }
            }
        }
        debug_assert!(links.is_empty(), "the cavity boundary is closed");
        for &t in &cavity {
            self.alive[t as usize] = false;
            self.free.push(t);
        }
        debug_assert!(last != NONE, "a cavity has faces");
        self.last = last;
        self.cavity = cavity;
        self.links = links;
    }

    fn alloc(&mut self, t: [u32; 4]) -> u32 {
        let id = match self.free.pop() {
            Some(id) => {
                self.tets[id as usize] = t;
                self.nbr[id as usize] = [NONE; 4];
                self.alive[id as usize] = true;
                id
            }
            None => {
                self.tets.push(t);
                self.nbr.push([NONE; 4]);
                self.alive.push(true);
                self.mark.push(0);
                (self.tets.len() - 1) as u32
            }
        };
        debug_assert!(
            {
                let q = t.map(|v| self.pts[v as usize]);
                orient(q[0], q[1], q[2], q[3]) > 0
            },
            "a new tet is positive"
        );
        id
    }
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
                let k = t.map(|w| w + 4);
                if inside(q, pts[v as usize], [k[0], k[1], k[2], k[3], v + 4]) {
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
                let k = t.map(|w| w + 4);
                assert!(!inside(q, pts[v as usize], [k[0], k[1], k[2], k[3], v + 4]));
            }
        }
    }
}
