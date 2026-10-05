//! The meshing topology: the B-rep as the mesh at its size can keep it.
//!
//! A face smaller than a cell (a crease split of a scan cuts noise into
//! such faces) cannot be kept: its edges would force samples closer than
//! the size and chords that cross at sharp corners. Such a face joins the
//! neighbour it shares the most boundary with, where both separate the same
//! two regions and carry the same tag; the edges between them are no edges
//! of the mesh any more. The joined faces are meshed as one composite face
//! on their facets, and each triangle goes back to the face of the facet it
//! lies on, so the B-rep stays the source of every tag and carrier.
//!
//! Only faces whose facets are their carrier join so far: discrete faces,
//! and planes into a composite that holds a discrete face (a scan's single
//! flat facet between creases); planes alone never join.

use rapidmesh_brep::{Brep, Model};
use rapidmesh_exact::vector::V3;
use rapidmesh_geom::Surface;
use rustc_hash::FxHashMap;

/// A face whose facets span less than this share of the size at its centre
/// joins a neighbour.
const JOIN_SIZE: f64 = 1.0;

/// Faces joined into composites and the edges between them.
#[derive(Default)]
pub(crate) struct Composites {
    /// Per face: the face its composite is meshed as (itself when alone).
    pub root: Vec<usize>,
    /// Per edge: whether it lies inside a composite (on its faces alone).
    pub internal: Vec<bool>,
}

/// The root of `x` in the forest `p` (halving the path on the way).
pub(crate) fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

impl Composites {
    /// Every face alone.
    pub(crate) fn alone(brep: &Brep) -> Composites {
        Composites {
            root: (0..brep.faces.len()).collect(),
            internal: vec![false; brep.edges.len()],
        }
    }

    /// The composites of `model` at the size `size`; faces `kept` never
    /// join (a size of their own, a periodic side).
    pub(crate) fn new(
        model: &Model,
        size: &dyn Fn(V3) -> f64,
        kept: &dyn Fn(usize) -> bool,
    ) -> Composites {
        let (plc, brep) = (&model.plc, &model.brep);
        let nf = brep.faces.len();
        let mut out = Composites {
            root: (0..nf).collect(),
            internal: vec![false; brep.edges.len()],
        };
        // Faces whose facets are their carrier: discrete ones, and planes
        // (a scan's single flat facet between creases is one).
        let joinable = |f: usize| {
            !brep.faces[f].facets.is_empty()
                && matches!(
                    brep.surface(brep.faces[f].surface),
                    Surface::Discrete(_) | Surface::Plane { .. }
                )
        };
        // Each face's span (the diagonal of its facets' box) and centre.
        let spans: Vec<(f64, V3)> = brep
            .faces
            .iter()
            .map(|f| {
                let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
                for &t in &f.facets {
                    for &v in &plc.triangles[t as usize] {
                        let p = plc.vertices[v as usize];
                        for k in 0..3 {
                            lo[k] = lo[k].min(p[k]);
                            hi[k] = hi[k].max(p[k]);
                        }
                    }
                }
                let d = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f64>().sqrt();
                (d, std::array::from_fn(|k| 0.5 * (lo[k] + hi[k])))
            })
            .collect();
        let small = |f: usize| {
            let (d, c) = spans[f];
            joinable(f) && !kept(f) && d < JOIN_SIZE * size(c)
        };
        // Per pair of faces: the length of the edges they share.
        let mut shared: FxHashMap<(usize, usize), f64> = FxHashMap::default();
        for e in &brep.edges {
            let len: f64 = e
                .chain
                .windows(2)
                .map(|w| {
                    (0..3)
                        .map(|k| (w[1][k] - w[0][k]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum();
            let fs: Vec<usize> = e
                .coedges
                .iter()
                .map(|&c| brep.coedge(c).face.0 as usize)
                .collect();
            for (i, &a) in fs.iter().enumerate() {
                for &b in &fs[i + 1..] {
                    if a != b {
                        *shared.entry((a.min(b), a.max(b))).or_default() += len;
                    }
                }
            }
        }
        let alike = |a: usize, b: usize| {
            let (fa, fb) = (&brep.faces[a], &brep.faces[b]);
            fa.regions == fb.regions && fa.face_tag == fb.face_tag && joinable(b) && !kept(b)
        };
        // Smallest first, each into the neighbour it shares the most with.
        let mut order: Vec<usize> = (0..nf).filter(|&f| small(f)).collect();
        order.sort_by(|&a, &b| spans[a].0.total_cmp(&spans[b].0));
        let mut parent: Vec<usize> = (0..nf).collect();
        // Whether each composite (by its root) holds a discrete face: planes
        // join only such a one (a scan's flat facet its discrete patch; two
        // planes of a CSG or CAD part meet at a true edge and stay apart).
        let mut discrete: Vec<bool> = (0..nf)
            .map(|f| matches!(brep.surface(brep.faces[f].surface), Surface::Discrete(_)))
            .collect();
        for f in order {
            let best = shared
                .iter()
                .filter(|(&(a, b), _)| a == f || b == f)
                .map(|(&(a, b), &l)| (if a == f { b } else { a }, l))
                .filter(|&(g, _)| {
                    let (rf, rg) = (find(&mut parent, f), find(&mut parent, g));
                    alike(f, g) && rf != rg && (discrete[rf] || discrete[rg])
                })
                .max_by(|x, y| x.1.total_cmp(&y.1).then(y.0.cmp(&x.0)));
            if let Some((g, _)) = best {
                let (rf, rg) = (find(&mut parent, f), find(&mut parent, g));
                // The larger side stays the root.
                let (keep, join) = if spans[rf].0 >= spans[rg].0 {
                    (rf, rg)
                } else {
                    (rg, rf)
                };
                parent[join] = keep;
                discrete[keep] = discrete[keep] || discrete[join];
            }
        }
        for f in 0..nf {
            out.root[f] = find(&mut parent, f);
        }
        for (ei, e) in brep.edges.iter().enumerate() {
            let roots: Vec<usize> = e
                .coedges
                .iter()
                .map(|&c| out.root[brep.coedge(c).face.0 as usize])
                .collect();
            let joined = e.coedges.iter().any(|&c| {
                out.root[brep.coedge(c).face.0 as usize] != brep.coedge(c).face.0 as usize
            });
            out.internal[ei] = joined && roots.len() >= 2 && roots.iter().all(|&r| r == roots[0]);
        }
        out
    }

    /// Whether any face joined another.
    pub(crate) fn any(&self) -> bool {
        self.root.iter().enumerate().any(|(f, &r)| r != f)
    }

    /// The faces of the composite of `root`, `root` first.
    pub(crate) fn members(&self, root: usize) -> Vec<usize> {
        let mut out = vec![root];
        out.extend(
            self.root
                .iter()
                .enumerate()
                .filter(|&(f, &r)| r == root && f != root)
                .map(|(f, _)| f),
        );
        out
    }

    /// The outline of the composite of `root` as loops of co-edges (ids),
    /// each run end to end, and the co-edges inside it that stay edges (a
    /// crease inside one of its faces).
    pub(crate) fn outline(&self, brep: &Brep, root: usize) -> (Vec<Vec<u32>>, Vec<u32>) {
        let members = self.members(root);
        let mut on_loop: Vec<u32> = Vec::new();
        for &f in &members {
            for lp in &brep.faces[f].loops {
                on_loop.extend(lp.coedges.iter().map(|c| c.0));
            }
        }
        let keep = |c: u32| !self.internal[brep.coedges[c as usize].edge.0 as usize];
        let inner: Vec<u32> = brep
            .coedges
            .iter()
            .enumerate()
            .filter(|(ci, c)| {
                members.contains(&(c.face.0 as usize))
                    && !on_loop.contains(&(*ci as u32))
                    && keep(*ci as u32)
            })
            .map(|(ci, _)| ci as u32)
            .collect();
        let ends = |c: u32| {
            let ce = &brep.coedges[c as usize];
            let e = &brep.edges[ce.edge.0 as usize];
            let (a, b) = (e.ends[0].0, e.ends[1].0);
            if ce.forward {
                (a, b)
            } else {
                (b, a)
            }
        };
        // The co-edges left, chained from end to start.
        let mut left: Vec<u32> = on_loop.into_iter().filter(|&c| keep(c)).collect();
        let mut loops = Vec::new();
        while let Some(first) = left.pop() {
            let start = ends(first).0;
            let mut ring = vec![first];
            let mut at = ends(first).1;
            while at != start {
                let Some(k) = left.iter().position(|&c| ends(c).0 == at) else {
                    break;
                };
                let c = left.swap_remove(k);
                at = ends(c).1;
                ring.push(c);
            }
            loops.push(ring);
        }
        (loops, inner)
    }
}
