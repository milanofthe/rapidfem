//! One bounding-volume hierarchy over a triangle set, for `O(log F)`
//! nearest-facet distance, segment candidates and graded distance fields.
//!
//! The model builds it once over its PLC facets ([`crate::Model::index`]);
//! the sizing field, the region query and the mesher all query that one
//! index. Per-facet values (size targets) are not part of the geometry:
//! a [`Targets`] pairs values with the tree's per-node minima, so one tree
//! serves every set of targets.
//!
//! Median/SAH splits on facet centroids: each facet sits in exactly one leaf,
//! internal nodes carry the subtree box, so the queries prune by a node
//! lower bound (branch and bound).

use rapidmesh_csg::Tri;
use rapidmesh_exact::vector::{box_d2, closest_on_tri, dist2, V3};
use rapidmesh_geom::bvh::Bvh;

/// Squared distance from point `p` to triangle `t`.
pub fn point_tri_dist2(p: V3, t: &Tri) -> f64 {
    dist2(p, closest_on_tri(p, t.v[0], t.v[1], t.v[2]))
}

pub struct FacetBvh {
    tris: Vec<Tri>,
    bvh: Bvh,
}

impl std::fmt::Debug for FacetBvh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FacetBvh({} facets, {} nodes)",
            self.tris.len(),
            self.bvh.nodes().len()
        )
    }
}

/// Values per facet of a [`FacetBvh`] (size targets) with their minimum per
/// tree node, the lower bound of the graded queries.
pub struct Targets {
    values: Vec<f64>,
    node_min: Vec<f64>,
}

impl Targets {
    /// `values[i]` for facet `i` of `bvh`.
    pub fn new(bvh: &FacetBvh, values: Vec<f64>) -> Targets {
        assert_eq!(values.len(), bvh.tris.len(), "one value per facet");
        let nodes = bvh.bvh.nodes();
        let mut node_min = vec![f64::INFINITY; nodes.len()];
        // Children come after their parent in the flat array, so a pass
        // from the back sees every child before its parent.
        for ni in (0..nodes.len()).rev() {
            let n = &nodes[ni];
            node_min[ni] = if n.count > 0 {
                bvh.bvh
                    .leaf(n)
                    .iter()
                    .map(|&fi| values[fi as usize])
                    .fold(f64::INFINITY, f64::min)
            } else {
                node_min[n.left as usize].min(node_min[n.right as usize])
            };
        }
        Targets { values, node_min }
    }
}

impl FacetBvh {
    /// The facets, in build order.
    pub fn tris(&self) -> &[Tri] {
        &self.tris
    }

    /// Builds a BVH over `tris` (facet `i` is `tris[i]`). Empty input is
    /// queryable (`nearest_dist` returns INFINITY).
    pub fn build(tris: &[Tri]) -> FacetBvh {
        let tris = tris.to_vec();
        let centroids: Vec<V3> = tris
            .iter()
            .map(|t| std::array::from_fn(|k| (t.v[0][k] + t.v[1][k] + t.v[2][k]) / 3.0))
            .collect();
        let boxes: Vec<(V3, V3)> = tris
            .iter()
            .map(|t| {
                (
                    std::array::from_fn(|k| t.v[0][k].min(t.v[1][k]).min(t.v[2][k])),
                    std::array::from_fn(|k| t.v[0][k].max(t.v[1][k]).max(t.v[2][k])),
                )
            })
            .collect();
        FacetBvh {
            bvh: Bvh::build(boxes, &centroids),
            tris,
        }
    }

    /// Distance from `p` to the nearest facet (INFINITY if empty).
    pub fn nearest_dist(&self, p: V3) -> f64 {
        self.nearest(p).map_or(f64::INFINITY, |(_, d)| d)
    }

    /// The distance from `p` to the nearest facet, or `r` if none is
    /// closer: the search skips everything beyond `r`, so a small `r` is
    /// cheap. Exact whenever the result is below `r`.
    pub fn nearest_dist_within(&self, p: V3, r: f64) -> f64 {
        if !(r < f64::INFINITY) {
            return self.nearest_dist(p).min(r);
        }
        self.bvh
            .nearest(p, r * r, |fi| {
                Some(point_tri_dist2(p, &self.tris[fi as usize]))
            })
            .map_or(r, |(_, d2)| d2.sqrt())
    }

    /// The nearest facet to `p` (its index in the build input) and its
    /// distance, `None` if empty.
    pub fn nearest(&self, p: V3) -> Option<(u32, f64)> {
        self.nearest_where(p, &|_| true)
    }

    /// The nearest facet to `p` among those `keep` accepts (by index in the
    /// build input) and its distance, `None` if there is none.
    pub fn nearest_where(&self, p: V3, keep: &dyn Fn(u32) -> bool) -> Option<(u32, f64)> {
        self.bvh
            .nearest(p, f64::INFINITY, |fi| {
                keep(fi).then(|| point_tri_dist2(p, &self.tris[fi as usize]))
            })
            .map(|(fi, d2)| (fi, d2.sqrt()))
    }

    /// Facet indices whose bounding box, grown by `pad`, meets the segment
    /// `a -> b`, appended to `out` (unsorted, no duplicates). A superset of
    /// the facets within `pad` of the segment: the candidate set for
    /// segment-surface crossings.
    pub fn facets_near_segment(&self, a: V3, b: V3, pad: f64, out: &mut Vec<u32>) {
        self.bvh.along_segment(a, b, pad, out);
    }

    /// The finest target among the facets within `r` of `p` (INFINITY if
    /// none).
    pub fn min_target_within(&self, targets: &Targets, p: V3, r: f64) -> f64 {
        if !(r >= 0.0) {
            return f64::INFINITY;
        }
        let r2 = r * r;
        let nodes = self.bvh.nodes();
        self.bvh
            .minimize(
                f64::INFINITY,
                |ni| {
                    if box_d2(nodes[ni].lo, nodes[ni].hi, p) > r2 {
                        f64::INFINITY
                    } else {
                        targets.node_min[ni]
                    }
                },
                |fi, best| {
                    let t = targets.values[fi as usize];
                    if t < best && point_tri_dist2(p, &self.tris[fi as usize]) <= r2 {
                        t
                    } else {
                        f64::INFINITY
                    }
                },
            )
            .1
    }

    /// `min over facets ( target + grading * dist(p, facet) )`, the graded
    /// distance field that grows the sizing field from the fine wall
    /// targets, or `bound` if it is not below: the search skips every
    /// subtree that cannot go below `bound`. Exact whenever the result is
    /// below `bound`.
    pub fn graded_min_within(&self, targets: &Targets, p: V3, grading: f64, bound: f64) -> f64 {
        let nodes = self.bvh.nodes();
        // Lower bound for anything in a subtree: the finest target plus the
        // graded distance to the subtree box.
        self.bvh
            .minimize(
                bound,
                |ni| targets.node_min[ni] + grading * box_d2(nodes[ni].lo, nodes[ni].hi, p).sqrt(),
                |fi, _| {
                    targets.values[fi as usize]
                        + grading * point_tri_dist2(p, &self.tris[fi as usize]).sqrt()
                },
            )
            .1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targeted(f: &[(Tri, f64)]) -> (FacetBvh, Targets) {
        let bvh = FacetBvh::build(&f.iter().map(|x| x.0).collect::<Vec<_>>());
        let tg = Targets::new(&bvh, f.iter().map(|x| x.1).collect());
        (bvh, tg)
    }

    fn tri(a: V3, b: V3, c: V3, target: f64) -> (Tri, f64) {
        (Tri::new(a, b, c), target)
    }

    fn brute_nearest(facets: &[(Tri, f64)], p: V3) -> f64 {
        facets
            .iter()
            .map(|(t, _)| point_tri_dist2(p, t))
            .fold(f64::MAX, f64::min)
            .sqrt()
    }

    fn brute_graded(facets: &[(Tri, f64)], p: V3, g: f64) -> f64 {
        facets
            .iter()
            .map(|(t, tg)| tg + g * point_tri_dist2(p, t).sqrt())
            .fold(f64::MAX, f64::min)
    }

    fn box_facets() -> Vec<(Tri, f64)> {
        // Two triangles per face of the unit cube, target varying per face.
        let mut f = Vec::new();
        let c = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.1),
            ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0.1),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], 0.5),
            ([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], 0.5),
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], 0.3),
            ([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], 0.3),
        ];
        for (a, b, cc, t) in c {
            f.push(tri(a, b, cc, t));
        }
        f
    }

    #[test]
    fn nearest_matches_brute() {
        let f = box_facets();
        let (bvh, _) = targeted(&f);
        for p in [
            [0.5, 0.5, 0.5],
            [0.1, 0.2, 0.9],
            [-1.0, 0.5, 0.5],
            [0.5, 0.5, 2.0],
        ] {
            let got = bvh.nearest_dist(p);
            let want = brute_nearest(&f, p);
            assert!(
                (got - want).abs() < 1e-12,
                "nearest at {p:?}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn graded_min_matches_brute() {
        let f = box_facets();
        let (bvh, tg) = targeted(&f);
        let g = 0.5;
        for p in [
            [0.5, 0.5, 0.5],
            [0.1, 0.2, 0.9],
            [0.5, 0.5, 0.05],
            [0.95, 0.5, 0.5],
        ] {
            let got = bvh.graded_min_within(&tg, p, g, f64::INFINITY);
            let want = brute_graded(&f, p, g);
            assert!(
                (got - want).abs() < 1e-12,
                "graded at {p:?}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn deep_tree_matches_brute() {
        // Many facets force a multi-level tree, so the flat-array child indices
        // (right != left+1) are exercised: a grid of small triangles at varied
        // depths with varied targets.
        let mut f = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                let (x, y) = (i as f64 * 0.5, j as f64 * 0.5);
                let z = 0.1 * (i + j) as f64;
                let target = 0.05 + 0.02 * ((i * 7 + j) % 5) as f64;
                f.push(tri(
                    [x, y, z],
                    [x + 0.4, y, z],
                    [x, y + 0.4, z + 0.2],
                    target,
                ));
            }
        }
        let (bvh, tg) = targeted(&f);
        for p in [
            [1.3, 1.7, 0.5],
            [-2.0, 0.5, 1.0],
            [3.1, 3.2, 0.0],
            [0.25, 0.25, 5.0],
            [1.0, 2.0, -1.0],
        ] {
            assert!(
                (bvh.nearest_dist(p) - brute_nearest(&f, p)).abs() < 1e-9,
                "nearest at {p:?}"
            );
            assert!(
                (bvh.graded_min_within(&tg, p, 0.5, f64::INFINITY) - brute_graded(&f, p, 0.5))
                    .abs()
                    < 1e-9,
                "graded at {p:?}"
            );
        }
    }

    #[test]
    fn segment_candidates_cover_every_facet_near_the_segment() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut f = Vec::new();
        for _ in 0..300 {
            let a: V3 = std::array::from_fn(|_| 4.0 * rnd());
            let b: V3 = std::array::from_fn(|k| a[k] + 0.3 * (rnd() - 0.5));
            let c: V3 = std::array::from_fn(|k| a[k] + 0.3 * (rnd() - 0.5));
            f.push(tri(a, b, c, 0.1));
        }
        let (bvh, _) = targeted(&f);
        let mut out = Vec::new();
        for _ in 0..500 {
            let a: V3 = std::array::from_fn(|_| 5.0 * rnd() - 0.5);
            let b: V3 = std::array::from_fn(|k| a[k] + 2.0 * (rnd() - 0.5));
            let pad = 0.1 * rnd();
            out.clear();
            bvh.facets_near_segment(a, b, pad, &mut out);
            let mut sorted = out.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), out.len(), "duplicates");
            // Every facet within `pad` of a dense sample of the segment is a
            // candidate.
            for (fi, (t, _)) in f.iter().enumerate() {
                let near = (0..=200).any(|i| {
                    let u = i as f64 / 200.0;
                    let p: V3 = std::array::from_fn(|k| a[k] + u * (b[k] - a[k]));
                    point_tri_dist2(p, t).sqrt() < pad
                });
                if near {
                    assert!(out.contains(&(fi as u32)), "facet {fi} missed");
                }
            }
        }
    }

    #[test]
    fn empty_is_safe() {
        let bvh = FacetBvh::build(&[]);
        assert_eq!(bvh.nearest_dist([0.0; 3]), f64::INFINITY);
    }
}
