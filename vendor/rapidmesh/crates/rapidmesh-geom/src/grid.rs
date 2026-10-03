//! A uniform grid of hashed cells in `D` dimensions: items binned by the
//! cell of a point or by every cell of a box, looked up around a cell, over
//! a box, or ring by ring for the nearest. The one grid of every crate above
//! `rapidmesh-csg`.

use rustc_hash::FxHashMap;

/// Items in hashed cells of side `cell`, the cells counted from `origin`.
#[derive(Debug, Clone)]
pub struct HashGrid<T, const D: usize = 3> {
    origin: [f64; D],
    cell: f64,
    map: FxHashMap<[i64; D], Vec<T>>,
    /// The box of the cells in use, for the ring search to end.
    lo: [i64; D],
    hi: [i64; D],
}

impl<T, const D: usize> HashGrid<T, D> {
    /// An empty grid of cells of side `cell` from the origin.
    pub fn new(cell: f64) -> Self {
        Self::with_origin([0.0; D], cell)
    }

    /// An empty grid of cells of side `cell` counted from `origin`.
    pub fn with_origin(origin: [f64; D], cell: f64) -> Self {
        HashGrid {
            origin,
            cell,
            map: FxHashMap::default(),
            lo: [i64::MAX; D],
            hi: [i64::MIN; D],
        }
    }

    /// The side of a cell.
    pub fn cell(&self) -> f64 {
        self.cell
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The cell holding `p`.
    pub fn key(&self, p: [f64; D]) -> [i64; D] {
        std::array::from_fn(|k| ((p[k] - self.origin[k]) / self.cell).floor() as i64)
    }

    /// The items of each cell in use, cell by cell in no particular order.
    pub fn bins(&self) -> impl Iterator<Item = &[T]> + '_ {
        self.map.values().map(Vec::as_slice)
    }

    /// The items of cell `key`, a new one to fill where there are none.
    pub fn at_mut(&mut self, key: [i64; D]) -> &mut Vec<T> {
        for k in 0..D {
            self.lo[k] = self.lo[k].min(key[k]);
            self.hi[k] = self.hi[k].max(key[k]);
        }
        self.map.entry(key).or_default()
    }

    /// The items of cell `key`.
    pub fn at(&self, key: [i64; D]) -> &[T] {
        self.map.get(&key).map_or(&[], Vec::as_slice)
    }

    /// Adds `item` in the cell of `p`.
    pub fn insert(&mut self, p: [f64; D], item: T) {
        let k = self.key(p);
        self.at_mut(k).push(item);
    }

    /// The items of the cells `key` plus at most `r` in every direction,
    /// cell by cell (the first coordinate slowest).
    pub fn around(&self, key: [i64; D], r: i64) -> impl Iterator<Item = &T> + '_ {
        let lo = key.map(|x| x - r);
        let hi = key.map(|x| x + r);
        self.cells(lo, hi)
    }

    /// The items of the cells exactly `r` cells from `key` (Chebyshev),
    /// in the order of [`keys`].
    pub fn ring(&self, key: [i64; D], r: i64) -> impl Iterator<Item = &T> + '_ {
        shell(key, r).flat_map(move |k| self.at(k))
    }

    /// The items of every cell the box `lo..hi` meets, cell by cell (an
    /// item binned by a box may come more than once).
    pub fn in_box(&self, lo: [f64; D], hi: [f64; D]) -> impl Iterator<Item = &T> + '_ {
        self.cells(self.key(lo), self.key(hi))
    }

    /// The items of the cells `lo..=hi`.
    pub fn cells(&self, lo: [i64; D], hi: [i64; D]) -> impl Iterator<Item = &T> + '_ {
        keys(lo, hi).flat_map(move |k| self.at(k))
    }

    /// The item nearest `p` by `d2`, a squared distance at least that from
    /// `p` to the cell of the item less `slack`: the cells ring by ring
    /// around `p`'s until no farther ring can hold a nearer item. Ties go to
    /// the first found.
    pub fn nearest(&self, p: [f64; D], slack: f64, d2: impl Fn(&T) -> f64) -> Option<(&T, f64)> {
        if self.map.is_empty() {
            return None;
        }
        let home = self.key(p);
        // Rings past this one hold no cell in use.
        let last = (0..D)
            .map(|k| (home[k] - self.lo[k]).max(self.hi[k] - home[k]))
            .max()
            .unwrap_or(0)
            .max(0);
        // Rings before this one hold no cell in use either.
        let first = (0..D)
            .map(|k| (self.lo[k] - home[k]).max(home[k] - self.hi[k]))
            .max()
            .unwrap_or(0)
            .max(0);
        let mut best: Option<(&T, f64)> = None;
        for r in first..=last {
            for item in self.ring(home, r) {
                let d = d2(item);
                if best.is_none_or(|(_, b)| d < b) {
                    best = Some((item, d));
                }
            }
            // A ring farther out lies at least `r` cells away.
            if let Some((_, b)) = best {
                let reach = (r as f64 * self.cell - slack).max(0.0);
                if b <= reach * reach && reach > 0.0 {
                    break;
                }
            }
        }
        best
    }
}

impl<T: Clone, const D: usize> HashGrid<T, D> {
    /// Adds `item` in every cell the box `lo..hi` meets.
    pub fn insert_box(&mut self, lo: [f64; D], hi: [f64; D], item: T) {
        for k in keys(self.key(lo), self.key(hi)) {
            self.at_mut(k).push(item.clone());
        }
    }
}

/// The keys `lo..=hi`, the first coordinate slowest.
pub fn keys<const D: usize>(lo: [i64; D], hi: [i64; D]) -> impl Iterator<Item = [i64; D]> {
    let empty = (0..D).any(|k| lo[k] > hi[k]);
    let mut next = (!empty).then_some(lo);
    std::iter::from_fn(move || {
        let cur = next?;
        let mut k = cur;
        let mut d = D;
        next = loop {
            if d == 0 {
                break None;
            }
            d -= 1;
            if k[d] < hi[d] {
                k[d] += 1;
                break Some(k);
            }
            k[d] = lo[d];
        };
        Some(cur)
    })
}

/// The keys at Chebyshev distance exactly `r` from `c`, in the order of
/// [`keys`].
fn shell<const D: usize>(c: [i64; D], r: i64) -> impl Iterator<Item = [i64; D]> {
    keys(c.map(|x| x - r), c.map(|x| x + r))
        .filter(move |k| (0..D).map(|i| (k[i] - c[i]).abs()).max().unwrap_or(0) == r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_run_the_first_coordinate_slowest() {
        let k: Vec<[i64; 2]> = keys([0, 0], [1, 2]).collect();
        assert_eq!(k, vec![[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]);
        assert_eq!(keys([1, 0], [0, 0]).count(), 0);
        assert_eq!(shell([0, 0, 0], 1).count(), 26);
        assert_eq!(shell([5, 5, 5], 0).collect::<Vec<_>>(), vec![[5, 5, 5]]);
    }

    /// The ring search finds what a scan finds, and a box query every item
    /// a box meets.
    #[test]
    fn nearest_and_boxes_match_a_scan() {
        let pts: Vec<[f64; 3]> = (0..400)
            .map(|i| {
                let t = i as f64;
                let r = (t / 400.0).powi(2) * 7.0;
                [r * (t * 0.9).cos(), r * (t * 1.7).sin(), (t * 0.23) % 1.5]
            })
            .collect();
        let mut g: HashGrid<usize> = HashGrid::with_origin([-1.0, -2.0, 0.5], 0.37);
        for (i, &p) in pts.iter().enumerate() {
            g.insert(p, i);
        }
        let d2 = |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
        for q in [[0.1, 0.2, 0.3], [9.0, -9.0, 4.0], [-3.0, 1.0, 0.0]] {
            let (&i, d) = g.nearest(q, 0.0, |&i| d2(pts[i], q)).unwrap();
            let best = pts.iter().map(|&p| d2(p, q)).fold(f64::INFINITY, f64::min);
            assert_eq!(d, best);
            assert_eq!(d2(pts[i], q), best);
        }
        let (lo, hi) = ([-1.0, -1.0, 0.0], [2.0, 0.5, 1.0]);
        let mut found: Vec<usize> = g
            .in_box(lo, hi)
            .copied()
            .filter(|&i| (0..3).all(|k| pts[i][k] >= lo[k] && pts[i][k] <= hi[k]))
            .collect();
        found.sort_unstable();
        let scan: Vec<usize> = (0..pts.len())
            .filter(|&i| (0..3).all(|k| pts[i][k] >= lo[k] && pts[i][k] <= hi[k]))
            .collect();
        assert_eq!(found, scan);
    }
}
