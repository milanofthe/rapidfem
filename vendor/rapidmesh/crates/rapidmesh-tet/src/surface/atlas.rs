//! Atlases: a curved face no single chart covers (a sphere, a torus, a
//! tube, an imported patch) is meshed in pieces, each a height field over
//! its own plane.
//!
//! The face's PLC facets are grown into pieces whose normals stay within
//! [`PIECE_TILT_DEG`] of their seed's. The PLC edges between pieces chain
//! into cuts, which run from a branch point, the face's outline or an edge
//! inside it to the next (or close on themselves). A cut is sampled once by
//! the size and its points (on the carrier) are the face's own, shared by
//! the pieces on either side; an end on the outline takes the outline's
//! nearest sample, so nothing is left hanging. Each piece is then an
//! ordinary domain in its chart: its outline segments (by the PLC edge
//! nearest each), its cuts, and what its facets cover as the inside.

use crate::surface::chart::{Chart, Flat};
use crate::surface::{Domain2, Slot};
use rapidmesh_brep::Model;
use rapidmesh_geom::grid::HashGrid;
use rapidmesh_geom::vec3::{bbox, cross, dist2, dot, perp, sub, unit};
use rustc_hash::{FxHashMap, FxHashSet};

type P3 = [f64; 3];

/// The largest angle between a facet's normal and its piece's seed.
pub const PIECE_TILT_DEG: f64 = 50.0;

/// A piece narrower than this multiple of the size (twice its area over
/// its perimeter) goes to a neighbour.
const MERGE_WIDTH: f64 = 0.5;

/// Where cut ends from different points snap to one sample, an end farther
/// from it than this share of the size takes a sample of its own.
const SNAP_SHARE: f64 = 1e-6;

/// A loop of the outline shorter than this multiple of the size lies in
/// one piece.
const SMALL_LOOP: f64 = 4.0;

/// The largest angle between a facet's normal and the seed of the piece it
/// goes to when its own piece is merged.
const MERGE_TILT_DEG: f64 = 60.0;

/// Why a face has no atlas: a reason it cannot be charted, or the samples
/// its edges need first (edge, arc length): where a cut ends on an edge,
/// the edge takes a sample, so the pieces meet the outline at their own
/// points and in their own order.
#[derive(Debug)]
pub(crate) enum AtlasError {
    Curved(&'static str),
    Refine(Vec<(u32, f64)>),
}

/// A piece of an atlas: its chart and its domain there.
pub(crate) struct Piece<'a> {
    pub(crate) chart: Chart<'a>,
    pub(crate) domain: Domain2,
    /// The points the face must take that fall on this piece.
    pub(crate) required: Vec<P3>,
}

/// The atlas of face `fi` whose loops are `rings`, whose edges inside are
/// `inner` (sample ids) and whose other fixed points are `corners`: the
/// face's own points the pieces share (their domains start with these), and
/// the pieces.
#[allow(clippy::too_many_arguments)]
pub(crate) fn atlas<'a>(
    model: &'a Model,
    fi: usize,
    points: &[P3],
    rings: &[Vec<u32>],
    inner: &[Vec<u32>],
    inner_edges: &[u32],
    edges: &[Vec<u32>],
    corners: &[u32],
    required: &[P3],
    size: &dyn Fn(P3) -> f64,
) -> Result<(Vec<P3>, Vec<Piece<'a>>), AtlasError> {
    let (plc, brep) = (&model.plc, &model.brep);
    let face = &brep.faces[fi];
    let surface = brep.surface(face.surface);
    let ff = Facets::of(model, fi)?;
    let nf = ff.ids.len();
    let pos = |v: u32| plc.vertices[v as usize];
    let corner3 = |k: usize| ff.all[k];
    let (piece, seeds) = grow_pieces(&ff, rings, points, size);
    let np = seeds.len();
    let cuts_found = cut_chains(&ff, model, fi, &piece);

    let (mut shared, measured, edge_at) = measure_cuts(
        &ff,
        model,
        fi,
        &piece,
        &cuts_found,
        points,
        rings,
        inner,
        edges,
        size,
    )?;
    let edge_of = &edge_at;
    let base = shared.len();
    let mut factor = vec![1usize; measured.len()];
    let mut cuts = sample_cuts(&measured, base, &factor, &mut shared, surface, size);

    let Assigned {
        outline,
        crease_bound,
        crease_inside,
    } = assign_outline(
        &ff,
        &piece,
        np,
        points,
        rings,
        inner,
        inner_edges,
        &cuts,
        &cuts_found,
    )?;
    let centroid = |k: usize| -> P3 {
        let p = corner3(k);
        std::array::from_fn(|i| (p[0][i] + p[1][i] + p[2][i]) / 3.0)
    };
    let facet_grid = near_grid(&(0..nf).map(centroid).collect::<Vec<_>>());
    let nearest_piece = |p: P3| -> usize {
        facet_grid
            .nearest(p, 0.0, |&k| dist2(centroid(k), p))
            .map_or(0, |(&k, _)| piece[k])
    };
    let mut of_piece: Vec<Vec<usize>> = vec![Vec::new(); np];
    for k in 0..nf {
        of_piece[piece[k]].push(k);
    }
    let mut corners_of: Vec<Vec<u32>> = vec![Vec::new(); np];
    for &g in corners {
        corners_of[nearest_piece(points[g as usize])].push(g);
    }
    let mut required_of: Vec<Vec<P3>> = vec![Vec::new(); np];
    for &p in required {
        required_of[nearest_piece(p)].push(p);
    }
    let mut cuts_of: Vec<Vec<usize>> = vec![Vec::new(); np];
    for (i, (sides, _)) in cuts.iter().enumerate() {
        cuts_of[sides[0]].push(i);
        if sides[1] != sides[0] {
            cuts_of[sides[1]].push(i);
        }
    }

    // ---- a chart and a domain per piece
    loop {
        let mut pieces = Vec::with_capacity(np);
        for c in 0..np {
            let corners3: Vec<[P3; 3]> = of_piece[c].iter().map(|&k| corner3(k)).collect();
            let Some(chart) = Chart::of_facets(surface, corners3, seeds[c]) else {
                return Err(AtlasError::Curved("a piece without a chart"));
            };
            let mut d = Domain2::default();
            for &p in &shared {
                d.add_own(p);
            }
            let at = |g: u32| (Slot::Global(g), chart.to2(points[g as usize]));
            for &(a, b) in outline[c].iter().chain(&crease_bound[c]) {
                d.add_chain(&[at(a), at(b)]);
            }
            for &i in &cuts_of[c] {
                let mapped: Vec<_> = cuts[i].1.iter().map(|&(s, p)| (s, chart.to2(p))).collect();
                d.add_chain(&mapped);
            }
            for &g in &corners_of[c] {
                let (s, q) = at(g);
                d.add_point(s, q);
            }
            d.close_loops();
            if d.loops.is_empty() {
                return Err(AtlasError::Curved("a piece whose outline closes no loop"));
            }
            // The creases inside, and those that end inside it, constrain.
            for &(a, b) in crease_inside[c].iter().chain(&crease_bound[c]) {
                d.add_chain(&[at(a), at(b)]);
            }
            pieces.push(Piece {
                chart,
                domain: d,
                required: required_of[c].clone(),
            });
        }
        // The cuts with chords that cross in a piece's chart.
        let mut of_chord: FxHashMap<(Slot, Slot), usize> = FxHashMap::default();
        for (i, (_, run)) in cuts.iter().enumerate() {
            for w in run.windows(2) {
                of_chord.insert((w[0].0.min(w[1].0), w[0].0.max(w[1].0)), i);
            }
        }
        // A crossing chord at a cut's end snapped off its PLC point asks
        // the edge there for a sample of its own; any other takes more
        // samples along its cut.
        let mut finer = false;
        let mut needed: Vec<(u32, f64)> = Vec::new();
        for p in &pieces {
            let d = &p.domain;
            let segments: Vec<(usize, usize)> = d.segments.iter().copied().collect();
            for k in crate::surface::crossings(&d.pts, &segments) {
                let (a, b) = segments[k];
                let (sa, sb) = (d.slots[a], d.slots[b]);
                let Some(&i) = of_chord.get(&(sa.min(sb), sa.max(sb))) else {
                    continue;
                };
                let run = &cuts[i].1;
                let end = [0, 1].into_iter().find(|&j| {
                    let s = if j == 0 {
                        run[0].0
                    } else {
                        run[run.len() - 1].0
                    };
                    matches!(s, Slot::Global(_)) && (s == sa || s == sb)
                });
                let off = end.and_then(|j| {
                    let v = measured[i].ids[j];
                    let &(e, arc) = edge_of.get(&v)?;
                    let g = if j == 0 {
                        run[0].1
                    } else {
                        run[run.len() - 1].1
                    };
                    (dist2(g, pos(v)).sqrt() > SNAP_SHARE * size(pos(v))).then_some((e, arc))
                });
                if let Some(x) = off {
                    needed.push(x);
                } else if run.len() < measured[i].ps.len() {
                    factor[i] *= 2;
                    finer = true;
                }
            }
        }
        if !needed.is_empty() {
            rapidmesh_exact::log::debug(
                "surface.refine",
                format!("face {fi}: {} cut ends off their samples", needed.len()),
            );
            return Err(AtlasError::Refine(needed));
        }
        if !finer {
            return Ok((shared, pieces));
        }
        cuts = sample_cuts(&measured, base, &factor, &mut shared, surface, size);
    }
}

/// The PLC facets of a face, their corners, their unit normals (turned to
/// the face's front) and the facets at each PLC edge.
struct Facets<'m> {
    plc: &'m rapidmesh_geom::TaggedPlc,
    ids: Vec<[u32; 3]>,
    all: Vec<[P3; 3]>,
    normal: Vec<P3>,
    edge_f: FxHashMap<(u32, u32), Vec<usize>>,
}

impl<'m> Facets<'m> {
    fn of(model: &'m Model, fi: usize) -> Result<Facets<'m>, AtlasError> {
        let (plc, brep) = (&model.plc, &model.brep);
        let face = &brep.faces[fi];
        let ids: Vec<[u32; 3]> = face
            .facets
            .iter()
            .map(|&t| plc.triangles[t as usize])
            .collect();
        if ids.is_empty() {
            return Err(AtlasError::Curved("no facets"));
        }
        let pos = |v: u32| plc.vertices[v as usize];
        let all: Vec<[P3; 3]> = ids.iter().map(|t| t.map(pos)).collect();
        let normal = all
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                let n = if plc.region_tags[face.facets[k] as usize] == face.regions {
                    n
                } else {
                    n.map(|x| -x)
                };
                unit(n).unwrap_or([0.0, 0.0, 1.0])
            })
            .collect();
        let mut edge_f: FxHashMap<(u32, u32), Vec<usize>> = FxHashMap::default();
        for (k, t) in ids.iter().enumerate() {
            for e in 0..3 {
                edge_f.entry(key(t[e], t[(e + 1) % 3])).or_default().push(k);
            }
        }
        Ok(Facets {
            plc,
            ids,
            all,
            normal,
            edge_f,
        })
    }

    fn pos(&self, v: u32) -> P3 {
        self.plc.vertices[v as usize]
    }
}

fn key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// The pieces of the facets: grown from seeds while they face their seed
/// closely and cover none of the piece in its plane, then the narrow ones
/// and those round a small loop of the outline merged into a neighbour.
/// Returns the piece of each facet and the seed normal of each piece.
fn grow_pieces(
    ff: &Facets,
    rings: &[Vec<u32>],
    points: &[P3],
    size: &dyn Fn(P3) -> f64,
) -> (Vec<usize>, Vec<P3>) {
    let (facets, normal, edge_f, all) = (&ff.ids, &ff.normal, &ff.edge_f, &ff.all);
    let nf = facets.len();
    let pos = |v: u32| ff.pos(v);
    let corner3 = |k: usize| all[k];
    // ---- pieces: facets grown from a seed, breadth first, while they face
    // it closely and cover none of the piece in its plane (a helix's top
    // faces up turn after turn)
    let cos_t = PIECE_TILT_DEG.to_radians().cos();
    let mut piece = vec![usize::MAX; nf];
    let mut seeds: Vec<P3> = Vec::new();
    for s0 in 0..nf {
        if piece[s0] != usize::MAX {
            continue;
        }
        let c = seeds.len();
        let n = normal[s0];
        seeds.push(n);
        let u = unit(perp(n)).unwrap_or([1.0, 0.0, 0.0]);
        let mut flat = Flat::new(all, u, cross(n, u));
        flat.try_add(all[s0]);
        piece[s0] = c;
        let mut queue = std::collections::VecDeque::from([s0]);
        while let Some(k) = queue.pop_front() {
            let t = facets[k];
            for e in 0..3 {
                for &nb in &edge_f[&key(t[e], t[(e + 1) % 3])] {
                    if piece[nb] == usize::MAX
                        && dot(normal[nb], n) >= cos_t
                        && flat.try_add(all[nb])
                    {
                        piece[nb] = c;
                        queue.push_back(nb);
                    }
                }
            }
        }
    }
    // ---- a piece narrower than the size (a strip left between two others)
    // would have its cuts closer than their samples: it goes to the
    // neighbour it shares the most boundary with, where that neighbour
    // still covers it as a height field
    let mut of: Vec<Vec<usize>> = vec![Vec::new(); seeds.len()];
    for k in 0..nf {
        of[piece[k]].push(k);
    }
    let area = |k: usize| {
        let p = corner3(k);
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        0.5 * dot(n, n).sqrt()
    };
    let mut order: Vec<usize> = (0..seeds.len()).collect();
    order.sort_by(|&a, &b| {
        let s = |p: usize| of[p].iter().map(|&k| area(k)).sum::<f64>();
        s(a).total_cmp(&s(b))
    });
    let cos_merge = MERGE_TILT_DEG.to_radians().cos();
    for p in order {
        if of[p].is_empty() {
            continue;
        }
        let mut shared: FxHashMap<usize, f64> = FxHashMap::default();
        let mut perimeter = 0.0;
        for &k in &of[p] {
            let t = facets[k];
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let across = edge_f[&key(a, b)].iter().find(|&&f| f != k).copied();
                let l = dist2(pos(a), pos(b)).sqrt();
                match across {
                    Some(f) if piece[f] == p => {}
                    Some(f) => {
                        perimeter += l;
                        *shared.entry(piece[f]).or_default() += l;
                    }
                    None => perimeter += l,
                }
            }
        }
        let a: f64 = of[p].iter().map(|&k| area(k)).sum();
        let c = centroid_of(&of[p], &corner3);
        if 2.0 * a / perimeter.max(1e-300) >= MERGE_WIDTH * size(c) {
            continue;
        }
        let mut near: Vec<(usize, f64)> = shared.into_iter().collect();
        near.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
        for (q, _) in near {
            let n = seeds[q];
            if of[p].iter().any(|&k| dot(normal[k], n) < cos_merge) {
                continue;
            }
            let u = unit(perp(n)).unwrap_or([1.0, 0.0, 0.0]);
            let mut flat = Flat::new(all, u, cross(n, u));
            if !of[q].iter().chain(&of[p]).all(|&k| flat.try_add(all[k])) {
                continue;
            }
            let moved = std::mem::take(&mut of[p]);
            for &k in &moved {
                piece[k] = q;
            }
            of[q].extend(moved);
            break;
        }
    }
    // A loop of the outline shorter than a few sizes (a small hole) lies
    // in one piece: several pieces round it would meet it closer than its
    // samples. They go to the largest of them where it covers them.
    for ring in rings {
        let pts: Vec<P3> = ring.iter().map(|&g| points[g as usize]).collect();
        if pts.len() < 2 {
            continue;
        }
        let len: f64 = (0..pts.len())
            .map(|k| dist2(pts[k], pts[(k + 1) % pts.len()]).sqrt())
            .sum();
        let c: P3 =
            std::array::from_fn(|k| pts.iter().map(|p| p[k]).sum::<f64>() / pts.len() as f64);
        let h = size(c);
        if len >= SMALL_LOOP * h {
            continue;
        }
        let reach = pts.iter().map(|&p| dist2(p, c)).fold(0.0, f64::max).sqrt() + 0.5 * h;
        // The pieces with a facet at the outline near the loop.
        let mut near: Vec<usize> = (0..nf)
            .filter(|&k| {
                let t = facets[k];
                (0..3).any(|e| edge_f[&key(t[e], t[(e + 1) % 3])].len() == 1)
                    && corner3(k).iter().any(|&p| dist2(p, c) <= reach * reach)
            })
            .map(|k| piece[k])
            .collect();
        near.sort_unstable();
        near.dedup();
        if near.len() < 2 {
            continue;
        }
        let Some(&q) = near.iter().max_by_key(|&&p| of[p].len()) else {
            continue;
        };
        let n = seeds[q];
        let others: Vec<usize> = near.iter().copied().filter(|&p| p != q).collect();
        let tilted = others
            .iter()
            .flat_map(|&p| of[p].iter())
            .any(|&k| dot(normal[k], n) < cos_merge);
        if tilted {
            continue;
        }
        let u = unit(perp(n)).unwrap_or([1.0, 0.0, 0.0]);
        let mut flat = Flat::new(all, u, cross(n, u));
        let fits = of[q]
            .iter()
            .chain(others.iter().flat_map(|&p| of[p].iter()))
            .all(|&k| flat.try_add(all[k]));
        if !fits {
            continue;
        }
        for p in others {
            let moved = std::mem::take(&mut of[p]);
            for &k in &moved {
                piece[k] = q;
            }
            of[q].extend(moved);
        }
    }
    // The pieces left, numbered afresh.
    let mut renumber = vec![usize::MAX; seeds.len()];
    let mut kept: Vec<P3> = Vec::new();
    for (p, fs) in of.iter().enumerate() {
        if !fs.is_empty() {
            renumber[p] = kept.len();
            kept.push(seeds[p]);
        }
    }
    for x in piece.iter_mut() {
        *x = renumber[*x];
    }
    (piece, kept)
}

/// The PLC edges of a face's outline, its creases (the PLC edges of the
/// B-rep edges inside it, per edge), the PLC points by their coordinates,
/// and the cuts between pieces chained from break to break (a branch
/// point, the outline or a crease) or round a closed loop.
struct Cuts {
    /// The outline's PLC edges with the facet at each.
    boundary: Vec<((u32, u32), usize)>,
    vid: FxHashMap<[u64; 3], u32>,
    crease_of: FxHashMap<u32, Vec<(u32, u32)>>,
    on_outline: FxHashSet<u32>,
    on_crease: FxHashSet<u32>,
    chains: Vec<Vec<u32>>,
}

fn cut_chains(ff: &Facets, model: &Model, fi: usize, piece: &[usize]) -> Cuts {
    let brep = &model.brep;
    let face = &brep.faces[fi];
    let (facets, edge_f) = (&ff.ids, &ff.edge_f);
    let pos = |v: u32| ff.pos(v);
    let boundary: Vec<((u32, u32), usize)> = edge_f
        .iter()
        .filter(|(_, fs)| fs.len() == 1)
        .map(|(&e, fs)| (e, fs[0]))
        .collect();
    let bits = |p: P3| p.map(f64::to_bits);
    let vid: FxHashMap<[u64; 3], u32> = facets
        .iter()
        .flatten()
        .map(|&v| (bits(pos(v)), v))
        .collect();
    let mut crease: FxHashSet<(u32, u32)> = FxHashSet::default();
    let mut crease_of: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
    let in_loop: FxHashSet<u32> = face
        .loops
        .iter()
        .flat_map(|l| l.coedges.iter().map(|c| c.0))
        .collect();
    for (ci, c) in brep.coedges.iter().enumerate() {
        if c.face.0 as usize != fi || in_loop.contains(&(ci as u32)) {
            continue;
        }
        let chain = &brep.edges[c.edge.0 as usize].chain;
        for w in chain.windows(2) {
            if let (Some(&a), Some(&b)) = (vid.get(&bits(w[0])), vid.get(&bits(w[1]))) {
                crease.insert(key(a, b));
                crease_of.entry(c.edge.0).or_default().push(key(a, b));
            }
        }
    }
    let cut: Vec<(u32, u32)> = edge_f
        .iter()
        .filter(|(e, fs)| fs.len() == 2 && piece[fs[0]] != piece[fs[1]] && !crease.contains(e))
        .map(|(&e, _)| e)
        .collect();

    // ---- the cuts chained and sampled
    let mut adj: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for &(a, b) in &cut {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    let on_outline: FxHashSet<u32> = boundary.iter().flat_map(|(e, _)| [e.0, e.1]).collect();
    let on_crease: FxHashSet<u32> = crease.iter().flat_map(|e| [e.0, e.1]).collect();
    let is_break = |v: u32| adj[&v].len() != 2 || on_outline.contains(&v) || on_crease.contains(&v);
    let mut chains: Vec<Vec<u32>> = Vec::new();
    let mut done: FxHashSet<(u32, u32)> = FxHashSet::default();
    let walk = |start: u32, next: u32, done: &mut FxHashSet<(u32, u32)>| -> Vec<u32> {
        let mut chain = vec![start];
        let (mut prev, mut cur) = (start, next);
        loop {
            done.insert(key(prev, cur));
            chain.push(cur);
            if cur == start || is_break(cur) {
                break;
            }
            let Some(&nx) = adj[&cur]
                .iter()
                .find(|&&x| x != prev && !done.contains(&key(cur, x)))
            else {
                break;
            };
            prev = cur;
            cur = nx;
        }
        chain
    };
    let mut starts: Vec<u32> = adj.keys().copied().filter(|&v| is_break(v)).collect();
    starts.sort_unstable();
    for s in starts {
        for &n in &adj[&s].clone() {
            if !done.contains(&key(s, n)) {
                chains.push(walk(s, n, &mut done));
            }
        }
    }
    for &(a, b) in &cut {
        if !done.contains(&key(a, b)) {
            chains.push(walk(a, b, &mut done));
        }
    }
    Cuts {
        boundary,
        vid,
        crease_of,
        on_outline,
        on_crease,
        chains,
    }
}

/// A cut: the pieces either side and its points (slot, place).
struct Chain {
    sides: [usize; 2],
    /// The PLC points it runs between.
    ids: [u32; 2],
    ps: Vec<P3>,
    cum: Vec<f64>,
    ends: [(Slot, P3); 2],
}

/// The cuts measured along their PLC points, each end on the outline or a
/// crease snapped to the nearest sample of the B-rep edge it ends on, the
/// others a point of the face's own. Returns the face's own points, the
/// cuts, and per PLC point on an edge of the face that edge and the arc
/// length there; the samples the edges need first where cut ends from
/// different points snapped to one sample.
#[allow(clippy::too_many_arguments)]
fn measure_cuts(
    ff: &Facets,
    model: &Model,
    fi: usize,
    piece: &[usize],
    found: &Cuts,
    points: &[P3],
    rings: &[Vec<u32>],
    inner: &[Vec<u32>],
    edges: &[Vec<u32>],
    size: &dyn Fn(P3) -> f64,
) -> Result<(Vec<P3>, Vec<Chain>, FxHashMap<u32, (u32, f64)>), AtlasError> {
    let brep = &model.brep;
    let surface = brep.surface(brep.faces[fi].surface);
    let edge_f = &ff.edge_f;
    let pos = |v: u32| ff.pos(v);
    let bits = |p: P3| p.map(f64::to_bits);
    let Cuts {
        vid,
        on_outline,
        on_crease,
        chains,
        ..
    } = found;
    // A cut ends on the outline or an edge inside at the nearest sample of
    // the B-rep edge it ends on (a sample of another edge nearby would
    // hang its piece on the wrong side of that edge).
    // Per PLC vertex on an edge of the face: the edge and its arc length
    // there.
    let mut edge_at: FxHashMap<u32, (u32, f64)> = FxHashMap::default();
    for c in brep.coedges.iter().filter(|c| c.face.0 as usize == fi) {
        let chain = &brep.edges[c.edge.0 as usize].chain;
        let mut arc = 0.0;
        for (i, p) in chain.iter().enumerate() {
            if i > 0 {
                arc += dist2(chain[i - 1], *p).sqrt();
            }
            if let Some(&v) = vid.get(&bits(*p)) {
                edge_at.entry(v).or_insert((c.edge.0, arc));
            }
        }
    }
    let anchors: Vec<u32> = rings
        .iter()
        .flatten()
        .chain(inner.iter().flatten())
        .copied()
        .collect();
    let anchor_grid = near_grid(
        &anchors
            .iter()
            .map(|&g| points[g as usize])
            .collect::<Vec<_>>(),
    );
    let nearest_anchor = |v: u32| -> Option<u32> {
        let p = pos(v);
        let by = |g: &&u32| dist2(points[**g as usize], p);
        match edge_at.get(&v) {
            Some(&(e, _)) => edges[e as usize]
                .iter()
                .min_by(|a, b| by(a).total_cmp(&by(b)))
                .copied(),
            None => anchor_grid
                .nearest(p, anchor_grid.cell(), |&i| {
                    dist2(points[anchors[i] as usize], p)
                })
                .map(|(&i, _)| anchors[i]),
        }
    };
    let mut shared: Vec<P3> = Vec::new();
    let mut junction: FxHashMap<u32, u32> = FxHashMap::default();
    // The ends that snapped to each sample: two apart on one sample would
    // meet the outline out of their order.
    let mut snapped: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    let mut ends = |v: u32, shared: &mut Vec<P3>| -> (Slot, P3) {
        if on_outline.contains(&v) || on_crease.contains(&v) {
            if let Some(g) = nearest_anchor(v) {
                let at = snapped.entry(g).or_default();
                if !at.contains(&v) {
                    at.push(v);
                }
                return (Slot::Global(g), points[g as usize]);
            }
        }
        let k = *junction.entry(v).or_insert_with(|| {
            shared.push(surface.closest(pos(v)).0);
            (shared.len() - 1) as u32
        });
        (Slot::Own(k), shared[k as usize])
    };
    let mut measured: Vec<Chain> = Vec::new();
    for ch in chains {
        let fs = &edge_f[&key(ch[0], ch[1])];
        let ps: Vec<P3> = ch.iter().map(|&v| pos(v)).collect();
        let mut cum = vec![0.0];
        for w in ps.windows(2) {
            let l = cum[cum.len() - 1] + dist2(w[0], w[1]).sqrt();
            cum.push(l);
        }
        if !(cum[cum.len() - 1] > 0.0) {
            continue;
        }
        let ends = [
            ends(ch[0], &mut shared),
            ends(ch[ch.len() - 1], &mut shared),
        ];
        measured.push(Chain {
            sides: [piece[fs[0]], piece[fs[1]]],
            ids: [ch[0], ch[ch.len() - 1]],
            ps,
            cum,
            ends,
        });
    }
    // Ends of cuts from different points that snapped to one sample ask
    // their edges for samples of their own (each where it lies off that
    // sample).
    let edge_of = &edge_at;
    let samples_needed: Vec<(u32, f64)> = snapped
        .iter()
        .filter(|(_, vs)| vs.len() > 1)
        .flat_map(|(&g, vs)| {
            vs.iter().filter_map(move |&v| {
                let &(e, arc) = edge_of.get(&v)?;
                let off = dist2(points[g as usize], pos(v)).sqrt();
                (off > SNAP_SHARE * size(pos(v))).then_some((e, arc))
            })
        })
        .collect();
    if !samples_needed.is_empty() {
        rapidmesh_exact::log::debug(
            "surface.refine",
            format!(
                "face {fi}: {} cut ends snapped together",
                samples_needed.len()
            ),
        );
        return Err(AtlasError::Refine(samples_needed));
    }
    Ok((shared, measured, edge_at))
}

/// A cut as sampled: the pieces either side and its points (slot, place).
type Cut = ([usize; 2], Vec<(Slot, P3)>);

/// The cuts sampled, each a sample per size times its `factor` (a cut whose
/// chords cross in a piece's chart, a piece narrower there than the size,
/// takes more, up to its PLC points, whose chain crosses nothing); the
/// face's own points from `base` on are those of the cuts.
fn sample_cuts(
    measured: &[Chain],
    base: usize,
    factor: &[usize],
    shared: &mut Vec<P3>,
    surface: &rapidmesh_brep::Surface,
    size: &dyn Fn(P3) -> f64,
) -> Vec<Cut> {
    // Cuts between the same two ends bound a piece between them: each
    // takes a point between, so the piece is no two-gon; so does a cut
    // between two points of the outline (a piece of one facet at the
    // outline lies between it and one outline segment). A cut that closes
    // on itself (round an island, or out from the outline and back to the
    // same sample) takes two.
    let pair = |c: &Chain| {
        let (a, b) = (c.ends[0].0, c.ends[1].0);
        (a.min(b), a.max(b))
    };
    let mut between: FxHashMap<(Slot, Slot), usize> = FxHashMap::default();
    for c in measured {
        *between.entry(pair(c)).or_default() += 1;
    }
    shared.truncate(base);
    let mut cuts: Vec<Cut> = Vec::new();
    for (c, &f) in measured.iter().zip(factor) {
        let (ps, cum) = (&c.ps, &c.cum);
        let len = cum[cum.len() - 1];
        let mid = ps[ps.len() / 2];
        let least = if c.ends[0].0 == c.ends[1].0 {
            3
        } else if between[&pair(c)] > 1
            || matches!(c.ends, [(Slot::Global(_), _), (Slot::Global(_), _)])
        {
            2
        } else {
            1
        };
        let n = ((len / size(mid).max(1e-300)).ceil() as usize).max(least);
        let at = |s: f64| -> P3 {
            let i = cum.partition_point(|&x| x <= s).clamp(1, cum.len() - 1);
            let t = ((s - cum[i - 1]) / (cum[i] - cum[i - 1]).max(1e-300)).clamp(0.0, 1.0);
            std::array::from_fn(|k| ps[i - 1][k] + t * (ps[i][k] - ps[i - 1][k]))
        };
        let inner: Vec<P3> = if f > 1 && n * f >= ps.len() - 1 && ps.len() > least {
            ps[1..ps.len() - 1].to_vec()
        } else {
            let n = n * f;
            (1..n)
                .map(|i| surface.closest(at(len * i as f64 / n as f64)).0)
                .collect()
        };
        let mut run = vec![c.ends[0]];
        for p in inner {
            shared.push(p);
            run.push((Slot::Own((shared.len() - 1) as u32), p));
        }
        run.push(c.ends[1]);
        cuts.push((c.sides, run));
    }
    cuts
}

/// The outline's segments and the creases inside, by piece: the segments
/// bounding each piece, and the crease segments on its border or inside it.
struct Assigned {
    outline: Vec<Vec<(u32, u32)>>,
    crease_bound: Vec<Vec<(u32, u32)>>,
    crease_inside: Vec<Vec<(u32, u32)>>,
}

#[allow(clippy::too_many_arguments)]
fn assign_outline(
    ff: &Facets,
    piece: &[usize],
    np: usize,
    points: &[P3],
    rings: &[Vec<u32>],
    inner: &[Vec<u32>],
    inner_edges: &[u32],
    cuts: &[Cut],
    found: &Cuts,
) -> Result<Assigned, AtlasError> {
    let (facets, edge_f) = (&ff.ids, &ff.edge_f);
    let pos = |v: u32| ff.pos(v);
    let bits = |p: P3| p.map(f64::to_bits);
    let Cuts {
        boundary,
        vid,
        crease_of,
        ..
    } = found;
    let seg_dist2 = |p: P3, a: P3, b: P3| -> f64 {
        let d = sub(b, a);
        let t = (dot(sub(p, a), d) / dot(d, d).max(1e-300)).clamp(0.0, 1.0);
        dist2(p, std::array::from_fn(|k| a[k] + t * d[k]))
    };
    let edge_mid = |e: (u32, u32)| -> P3 {
        let (a, b) = (pos(e.0), pos(e.1));
        std::array::from_fn(|k| 0.5 * (a[k] + b[k]))
    };
    let boundary_grid = near_grid(&boundary.iter().map(|x| edge_mid(x.0)).collect::<Vec<_>>());
    let piece_of_segment = |g1: u32, g2: u32, edges: &[((u32, u32), usize)]| -> Option<usize> {
        let m: P3 =
            std::array::from_fn(|k| 0.5 * (points[g1 as usize][k] + points[g2 as usize][k]));
        boundary_grid
            .nearest(m, boundary_grid.cell(), |&i| {
                seg_dist2(m, pos(edges[i].0 .0), pos(edges[i].0 .1))
            })
            .map(|(&i, _)| piece[edges[i].1])
    };
    // A loop of the outline changes piece exactly where a cut ends on it:
    // each run between two such ends goes to one piece, the one most of its
    // segments lie by.
    // The pieces either side of the cuts (and the edges inside between
    // two pieces) ending at each outline point: a run between two such
    // ends belongs to the piece on both, and only
    // where that leaves a choice (or no cut ends on the loop) do its
    // segments vote by the facets they lie by.
    let mut cut_ends: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
    for (sides, run) in cuts {
        for s in [run[0].0, run[run.len() - 1].0] {
            if let Slot::Global(g) = s {
                cut_ends.entry(g).or_default().extend(sides);
            }
        }
    }
    // Each segment of an edge inside the face goes to the pieces whose
    // facets meet at the nearest of that edge's creases: it bounds them
    // where they differ, and lies inside the one piece where they do not.
    let mut crease_bound: Vec<Vec<(u32, u32)>> = vec![Vec::new(); np];
    let mut crease_inside: Vec<Vec<(u32, u32)>> = vec![Vec::new(); np];
    let no_crease: Vec<(u32, u32)> = Vec::new();
    // The pieces either side where such an edge ends (on the outline, a
    // place the pieces change as at a cut's end).
    let mut crease_ends: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
    let sides_by = |edge: u32, a: u32, b: u32| -> Option<(usize, usize)> {
        let m: P3 = std::array::from_fn(|k| 0.5 * (points[a as usize][k] + points[b as usize][k]));
        crease_of
            .get(&edge)
            .unwrap_or(&no_crease)
            .iter()
            .map(|e| (seg_dist2(m, pos(e.0), pos(e.1)), e))
            .min_by(|x, y| x.0.total_cmp(&y.0))
            .map(|(_, e)| {
                let fs = &edge_f[e];
                let (a, b) = (piece[fs[0]], piece[fs[fs.len() - 1]]);
                (a.min(b), a.max(b))
            })
    };
    // As on the outline, the pieces change along an edge inside only where
    // a cut ends on it: each run between such ends lies between the two
    // pieces most of its segments lie between.
    for (c, &edge) in inner.iter().zip(inner_edges) {
        let n = c.len();
        let mut breaks: Vec<usize> = (1..n.saturating_sub(1))
            .filter(|&k| cut_ends.contains_key(&c[k]))
            .collect();
        breaks.insert(0, 0);
        breaks.push(n - 1);
        for w in breaks.windows(2) {
            let segs: Vec<(u32, u32)> = (w[0]..w[1])
                .map(|k| (c[k], c[k + 1]))
                .filter(|(a, b)| a != b)
                .collect();
            let mut votes: FxHashMap<(usize, usize), usize> = FxHashMap::default();
            for &(a, b) in &segs {
                if let Some(pair) = sides_by(edge, a, b) {
                    *votes.entry(pair).or_default() += 1;
                }
            }
            let Some((pl, pr)) = votes
                .into_iter()
                .max_by_key(|&(pair, k)| (k, std::cmp::Reverse(pair)))
                .map(|x| x.0)
            else {
                continue;
            };
            if pl == pr {
                crease_inside[pl].extend(&segs);
            } else {
                crease_bound[pl].extend(&segs);
                crease_bound[pr].extend(&segs);
                for k in [w[0], w[1]] {
                    if k == 0 || k == n - 1 {
                        crease_ends.entry(c[k]).or_default().extend([pl, pr]);
                    }
                }
            }
        }
    }
    for (g, ps) in crease_ends {
        cut_ends.entry(g).or_default().extend(ps);
    }
    // Where loops of the outline touch (a point in more than one of them,
    // or twice in one), a piece may pass from one loop to the other with no
    // cut ending there: such a point is a place the pieces change too, with
    // the pieces of the facets at it.
    let mut seen_on: FxHashMap<u32, usize> = FxHashMap::default();
    for r in rings {
        for &g in r {
            *seen_on.entry(g).or_default() += 1;
        }
    }
    let mut at_vertex: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
    for (k, t) in facets.iter().enumerate() {
        for &v in t {
            at_vertex.entry(v).or_default().push(k);
        }
    }
    for (&g, &n) in &seen_on {
        if n < 2 {
            continue;
        }
        if let Some(v) = vid.get(&bits(points[g as usize])) {
            let ps: Vec<usize> = at_vertex
                .get(v)
                .map(|ks| ks.iter().map(|&k| piece[k]).collect())
                .unwrap_or_default();
            cut_ends.entry(g).or_default().extend(ps);
        }
    }
    let mut outline: Vec<Vec<(u32, u32)>> = vec![Vec::new(); np];
    for r in rings {
        let n = r.len();
        let breaks: Vec<usize> = (0..n).filter(|&k| cut_ends.contains_key(&r[k])).collect();
        let runs: Vec<(usize, usize)> = if breaks.is_empty() {
            vec![(0, n)]
        } else {
            (0..breaks.len())
                .map(|i| {
                    let (a, b) = (breaks[i], breaks[(i + 1) % breaks.len()]);
                    (a, if b > a { b - a } else { b + n - a })
                })
                .collect()
        };
        for (start, len) in runs {
            let segs: Vec<(u32, u32)> = (0..len)
                .map(|k| (r[(start + k) % n], r[(start + k + 1) % n]))
                .filter(|(a, b)| a != b)
                .collect();
            let mut candidates: Vec<usize> =
                match (cut_ends.get(&r[start]), cut_ends.get(&r[(start + len) % n])) {
                    (Some(x), Some(y)) => x.iter().copied().filter(|p| y.contains(p)).collect(),
                    _ => Vec::new(),
                };
            candidates.sort_unstable();
            candidates.dedup();
            if candidates.is_empty() {
                candidates = (0..np).collect();
            }
            let p = if let [p] = candidates[..] {
                p
            } else {
                let mut votes = vec![0usize; np];
                for &(a, b) in &segs {
                    let Some(p) = piece_of_segment(a, b, boundary) else {
                        return Err(AtlasError::Curved("an outline without facets"));
                    };
                    votes[p] += 1;
                }
                candidates
                    .iter()
                    .copied()
                    .max_by_key(|&p| votes[p])
                    .unwrap_or(0)
            };
            outline[p].extend(segs);
        }
    }
    Ok(Assigned {
        outline,
        crease_bound,
        crease_inside,
    })
}

/// The centroid of facets `ks`.
fn centroid_of(ks: &[usize], corner3: &dyn Fn(usize) -> [P3; 3]) -> P3 {
    let mut c = [0.0; 3];
    for &k in ks {
        for p in corner3(k) {
            for i in 0..3 {
                c[i] += p[i];
            }
        }
    }
    c.map(|x| x / (3 * ks.len().max(1)) as f64)
}

/// The points in a grid of about one a cell, for the nearest by any
/// distance that grows with the distance to the point (a segment's by its
/// middle, which may undercut it by up to a cell).
fn near_grid(pts: &[P3]) -> HashGrid<usize> {
    let (lo, hi) = bbox(pts);
    let span = (0..3)
        .map(|k| hi[k] - lo[k])
        .fold(0.0, f64::max)
        .max(1e-300);
    let cell = (span / (pts.len() as f64).cbrt().max(1.0)).max(1e-12 * span);
    let mut g = HashGrid::with_origin(lo, cell);
    for (i, &p) in pts.iter().enumerate() {
        g.insert(p, i);
    }
    g
}
