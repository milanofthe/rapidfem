//! Virtual cuts: a model too large for one worker is meshed in blocks.
//!
//! Planes cut the model's box, one cell at a time, until each cell holds
//! about a block's share of the tets. A plane goes where no corner of the
//! model lies within the size of it along its axis (so it meets every face
//! across, never along or close to one) and as near the middle of the
//! cell's work as that allows. The cut model is the model's arrangement
//! with the planes as sheets, and each piece of a region in one cell is a
//! region of its own: the surface stage meshes the cut faces like any
//! face, so the blocks conform, and every block runs the Delaunay, the
//! constrained and the refinement stages on its own, in parallel.
//!
//! Before the improvement everything is named in the model's own terms
//! again: a block by its region, a piece of a face or an edge by the face
//! or edge it is a piece of, and a point on a cut by what it lies on in the
//! model (a face, an edge, or nothing: the volume). The cut faces leave
//! the mesh, and their points move freely like any volume point.

use crate::mesh::PointClass;

use crate::params::MeshParams;
use crate::sizing::tree::DomainTree;
use rapidmesh_brep::Model;
use rapidmesh_geom::vec3::{bbox, cross, dot, sub};
use rapidmesh_geom::{sheet_polygon, FaceTag, RegionTag, Scene};
use rustc_hash::FxHashMap;

type P3 = [f64; 3];

/// The face tag of the cut sheets.
pub const CUT_TAG: FaceTag = FaceTag(u32::MAX);

/// A model of fewer tets stays whole.
const CUT_FROM: f64 = 500_000.0;

/// The tets of a block: enough blocks for every worker to take several in
/// turn (a large one last would keep the others waiting), within these
/// bounds (each cut adds its faces to mesh).
const BLOCKS_PER_WORKER: f64 = 4.0;
const BLOCK_MIN: f64 = 100_000.0;
const BLOCK_MAX: f64 = 250_000.0;

/// Tets per cube of the size (a regular mesh of edge length `h` has about
/// `6 sqrt 2 / h^3` per unit volume).
const TETS_PER_CUBE: f64 = 8.5;

/// A cut keeps at least this many sizes from every corner of the model
/// along its axis (each corner's size), and a place with up to [`ROOM`]
/// sizes is taken over one with less.
const CLEAR: f64 = 1.0;
const ROOM: f64 = 4.0;

/// Size samples of a cell: slices along the axis, and per slice this many
/// across in each other direction.
const SLICES: usize = 32;
const ACROSS: usize = 8;

/// A cell of the cuts, as a node of their tree.
enum Node {
    Leaf(u32),
    Split {
        axis: usize,
        at: f64,
        kids: [u32; 2],
    },
}

/// One cut: the plane `x[axis] = at` over the cell box `lo..hi`.
struct Cut {
    axis: usize,
    at: f64,
    lo: P3,
    hi: P3,
}

/// Where the cuts go.
pub struct Plan {
    nodes: Vec<Node>,
    cuts: Vec<Cut>,
    /// The tolerance within which a point lies on a cut.
    tol: f64,
}

impl Plan {
    /// The cuts of `model` at the sizes of `domain` into blocks of about
    /// `block` tets (none where the whole is less than two), by default
    /// as many as the workers can share evenly (none below [`CUT_FROM`]).
    pub fn new(model: &Model, domain: &DomainTree, block: Option<f64>) -> Plan {
        let pts = &model.plc.vertices;
        let (lo, hi) = bbox(pts);
        let diag = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f64>().sqrt();
        let mut plan = Plan {
            nodes: vec![Node::Leaf(0)],
            cuts: Vec::new(),
            tol: 1e-9 * diag.max(1e-300),
        };
        if pts.is_empty() {
            return plan;
        }
        // The cuts reach out of the model, so each meets its outer faces
        // across.
        let pad = 0.01 * diag;
        let (lo, hi) = (lo.map(|x| x - pad), hi.map(|x| x + pad));
        let block = match block {
            Some(b) => b,
            None => {
                let total: f64 = work(domain, lo, hi, 0).iter().sum();
                if total < CUT_FROM {
                    return plan;
                }
                let workers = rayon::current_num_threads() as f64;
                (total / (BLOCKS_PER_WORKER * workers)).clamp(BLOCK_MIN, BLOCK_MAX)
            }
        };
        let sizes: Vec<f64> = {
            use rayon::prelude::*;
            pts.par_iter().map(|&p| domain.h_at(p).min(1e300)).collect()
        };
        let mut cells = 0u32;
        let mut stack: Vec<(u32, P3, P3, Vec<u32>)> =
            vec![(0, lo, hi, (0..pts.len() as u32).collect())];
        while let Some((node, lo, hi, inside)) = stack.pop() {
            match split(domain, pts, &sizes, &inside, lo, hi, block, &plan.cuts) {
                Some((axis, at)) => {
                    let (below, above): (Vec<u32>, Vec<u32>) =
                        inside.iter().partition(|&&v| pts[v as usize][axis] < at);
                    let kids = [plan.nodes.len() as u32, plan.nodes.len() as u32 + 1];
                    plan.nodes.push(Node::Leaf(0));
                    plan.nodes.push(Node::Leaf(0));
                    plan.nodes[node as usize] = Node::Split { axis, at, kids };
                    rapidmesh_exact::log::debug(
                        "mesher.blocks",
                        format!("cut x{axis} = {at} over {lo:?}..{hi:?}"),
                    );
                    plan.cuts.push(Cut { axis, at, lo, hi });
                    let (mut top, mut bottom) = (hi, lo);
                    top[axis] = at;
                    bottom[axis] = at;
                    stack.push((kids[0], lo, top, below));
                    stack.push((kids[1], bottom, hi, above));
                }
                None => {
                    plan.nodes[node as usize] = Node::Leaf(cells);
                    cells += 1;
                }
            }
        }
        plan
    }

    /// The number of cells.
    pub fn cells(&self) -> usize {
        self.cuts.len() + 1
    }

    /// Whether the model stays whole.
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }

    /// Whether `p` lies within `reach` of a cut.
    pub fn near(&self, p: P3, reach: f64) -> bool {
        self.cuts.iter().any(|c| {
            (p[c.axis] - c.at).abs() <= reach
                && (0..3)
                    .all(|k| k == c.axis || (p[k] >= c.lo[k] - reach && p[k] <= c.hi[k] + reach))
        })
    }

    /// The cell of `p`; on the cut `(axis, at)` the one above it or below.
    fn cell(&self, p: P3, on: Option<(usize, f64, bool)>) -> u32 {
        let mut n = 0;
        loop {
            match self.nodes[n] {
                Node::Leaf(c) => return c,
                Node::Split { axis, at, kids } => {
                    let above = match on {
                        Some((a, x, side)) if a == axis && (x - at).abs() <= self.tol => side,
                        _ => p[axis] > at,
                    };
                    n = kids[above as usize] as usize;
                }
            }
        }
    }
}

/// The cut of the cell `lo..hi` holding the corners `inside` (`sizes` the
/// size at each corner of the model), if it holds more than two `block`s'
/// work: on an axis not much shorter than the longest, between the
/// quartiles of its work, where it stays farthest from every corner in
/// that corner's size (up to [`ROOM`] sizes), nearest the median among
/// equals; none where no place keeps [`CLEAR`] sizes. A place within a
/// size of a cut already made on the same axis (in another cell) moves onto
/// it where it keeps clear there too: cells side by side cut in one plane,
/// not in two a hair apart, whose step along their common face would leave
/// slivers between the blocks.
#[allow(clippy::too_many_arguments)]
fn split(
    domain: &DomainTree,
    pts: &[P3],
    sizes: &[f64],
    inside: &[u32],
    lo: P3,
    hi: P3,
    block: f64,
    made: &[Cut],
) -> Option<(usize, f64)> {
    let longest = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
    let coarsest = inside
        .iter()
        .map(|&i| sizes[i as usize])
        .fold(0.0, f64::max);
    // The best place so far: its room (sizes), its distance from the
    // median (a share of the cell), the place and its axis.
    let mut best: Option<(f64, f64, f64, usize)> = None;
    for axis in 0..3 {
        if hi[axis] - lo[axis] < 0.5 * longest {
            continue;
        }
        let width = (hi[axis] - lo[axis]) / SLICES as f64;
        let work = work(domain, lo, hi, axis);
        let total: f64 = work.iter().sum();
        if total < 2.0 * block {
            return None;
        }
        // The place below which a share of the work lies.
        let quantile = |q: f64| {
            let mut acc = 0.0;
            for (s, &w) in work.iter().enumerate() {
                if acc + w >= q * total && w > 0.0 {
                    return lo[axis] + (s as f64 + (q * total - acc) / w) * width;
                }
                acc += w;
            }
            hi[axis]
        };
        let (from, mid, to) = (quantile(0.25), quantile(0.5), quantile(0.75));
        let mut corners: Vec<(f64, f64)> = inside
            .iter()
            .map(|&i| (pts[i as usize][axis], sizes[i as usize]))
            .collect();
        corners.sort_by(|a, b| a.0.total_cmp(&b.0));
        // The room at `x`: the least distance to a corner in its size; only
        // corners within ROOM of the coarsest size can bound it below ROOM.
        let reach = ROOM * coarsest;
        let room = |x: f64| -> f64 {
            let start = corners.partition_point(|c| c.0 < x - reach);
            corners[start..]
                .iter()
                .take_while(|c| c.0 <= x + reach)
                .map(|&(c, h)| (x - c).abs() / h)
                .fold(ROOM, f64::min)
        };
        // The candidates: the ends and the median, and between each two
        // corners the place as far from both in their sizes.
        let mut candidates = vec![from, mid, to];
        candidates.extend(corners.windows(2).filter_map(|w| {
            let ((a, ha), (b, hb)) = (w[0], w[1]);
            let x = (a * hb + b * ha) / (ha + hb);
            (x > from && x < to).then_some(x)
        }));
        let planes: Vec<f64> = made
            .iter()
            .filter(|c| c.axis == axis && c.at > lo[axis] && c.at < hi[axis])
            .map(|c| c.at)
            .collect();
        for x in candidates {
            let x = planes
                .iter()
                .copied()
                .filter(|&c| (c - x).abs() <= coarsest && room(c) >= CLEAR)
                .min_by(|a, b| (a - x).abs().total_cmp(&(b - x).abs()))
                .unwrap_or(x);
            let r = room(x);
            if r < CLEAR {
                continue;
            }
            let d = (x - mid).abs() / (hi[axis] - lo[axis]);
            if best.is_none_or(|(r0, d0, _, _)| r > r0 || (r == r0 && d < d0)) {
                best = Some((r, d, x, axis));
            }
        }
    }
    best.map(|(_, _, x, axis)| (axis, x))
}

/// The estimated tets of each of [`SLICES`] slices of the box `lo..hi`
/// along `axis`, from the size at [`ACROSS`] squared points in each.
fn work(domain: &DomainTree, lo: P3, hi: P3, axis: usize) -> Vec<f64> {
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let width = (hi[axis] - lo[axis]) / SLICES as f64;
    let volume: f64 = (0..3).map(|k| hi[k] - lo[k]).product();
    let sample = volume / (SLICES * ACROSS * ACROSS) as f64;
    (0..SLICES)
        .map(|s| {
            let mut w = 0.0;
            for a in 0..ACROSS {
                for b in 0..ACROSS {
                    let mut p = [0.0; 3];
                    p[axis] = lo[axis] + (s as f64 + 0.5) * width;
                    p[u] = lo[u] + (a as f64 + 0.5) / ACROSS as f64 * (hi[u] - lo[u]);
                    p[v] = lo[v] + (b as f64 + 0.5) / ACROSS as f64 * (hi[v] - lo[v]);
                    let h = domain.h_at(p);
                    if h.is_finite() && h > 0.0 {
                        w += TETS_PER_CUBE * sample / (h * h * h);
                    }
                }
            }
            w
        })
        .collect()
}

/// The model cut into blocks, and how its entities are named in the
/// model it was cut from (the source).
pub struct Blocks {
    pub model: Model,
    /// Per region of the cut model: the source region (0 stays 0).
    region: Vec<u32>,
    /// Per face: the source face it is a piece of, `None` on a cut.
    face: Vec<Option<u32>>,
    /// Per edge and per vertex: what it lies on in the source.
    edge: Vec<PointClass>,
    vertex: Vec<PointClass>,
}

impl Blocks {
    /// `source` (the model of `scene`) cut as `plan` says; `None` where the
    /// cut model does not arrange or loses a corner of the source.
    pub fn new(scene: &Scene, source: &Model, plan: &Plan) -> Option<Blocks> {
        let mut scene = scene.clone();
        for c in &plan.cuts {
            // Every coordinate of the sheet is one of its cell's bounds,
            // exactly: a cut ending on the one before it (on its cell's
            // side) then meets it, not a rounding short or beyond.
            let (u, v) = ((c.axis + 1) % 3, (c.axis + 2) % 3);
            let (mut base, mut eu, mut ev) = ([0.0; 3], [0.0; 3], [0.0; 3]);
            base[c.axis] = c.at;
            eu[u] = 1.0;
            ev[v] = 1.0;
            let outline = [
                [c.lo[u], c.lo[v]],
                [c.hi[u], c.lo[v]],
                [c.hi[u], c.hi[v]],
                [c.lo[u], c.hi[v]],
            ];
            let mut sheet = sheet_polygon(&outline, &[], base, eu, ev);
            // Its corners are no corners of the model: where they lie inside
            // it (on the cut before), nothing ends there.
            sheet.corners.clear();
            scene.add_sheet(sheet, CUT_TAG);
        }
        let mut plc = match scene.try_assemble() {
            Ok(plc) => plc,
            Err(e) => {
                rapidmesh_exact::log::warn("mesher.blocks", format!("no cut model: {e}"));
                return None;
            }
        };
        // Each side of each triangle: its region in the cell it faces.
        let mut blocks: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut region = vec![0u32];
        let mut keep = vec![true; plc.triangles.len()];
        for (ti, t) in plc.triangles.iter().enumerate() {
            let [a, b, c] = t.map(|v| plc.vertices[v as usize]);
            let centre: P3 = std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0);
            let [front, back] = plc.region_tags[ti].map(|r| r.0);
            let cells = if plc.face_tags[ti] == CUT_TAG {
                if front == 0 && back == 0 {
                    keep[ti] = false;
                    continue;
                }
                let n = cross(sub(b, a), sub(c, a));
                let axis = (0..3)
                    .max_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
                    .unwrap_or(0);
                // The normal points into the front.
                let up = n[axis] > 0.0;
                [
                    plan.cell(centre, Some((axis, centre[axis], up))),
                    plan.cell(centre, Some((axis, centre[axis], !up))),
                ]
            } else {
                let c = plan.cell(centre, None);
                [c, c]
            };
            let mut block = |r: u32, cell: u32| -> u32 {
                if r == 0 {
                    return 0;
                }
                *blocks.entry((r, cell)).or_insert_with(|| {
                    region.push(r);
                    region.len() as u32 - 1
                })
            };
            plc.region_tags[ti] = [
                RegionTag(block(front, cells[0])),
                RegionTag(block(back, cells[1])),
            ];
        }
        let mut k = 0;
        let mut kept = |_: &_| {
            k += 1;
            keep[k - 1]
        };
        plc.triangles.retain(&mut kept);
        k = 0;
        plc.face_tags.retain(|_| {
            k += 1;
            keep[k - 1]
        });
        k = 0;
        plc.surface_refs.retain(|_| {
            k += 1;
            keep[k - 1]
        });
        k = 0;
        plc.region_tags.retain(|_| {
            k += 1;
            keep[k - 1]
        });
        let mut used = vec![false; plc.vertices.len()];
        for t in &plc.triangles {
            for &v in t {
                used[v as usize] = true;
            }
        }
        plc.features
            .retain(|e| used[e[0] as usize] && used[e[1] as usize]);
        plc.corners.retain(|&v| used[v as usize]);
        let model = Model::new(plc);
        let blocks = Blocks::name(model, source, region);
        if blocks.is_none() {
            rapidmesh_exact::log::warn("mesher.blocks", "the cut model lost a corner");
        }
        blocks
    }

    /// Names the entities of the cut `model` in the source's terms.
    fn name(model: Model, source: &Model, region: Vec<u32>) -> Option<Blocks> {
        let (brep, src) = (&model.brep, &source.brep);
        // Each face by the source face under one of its facets, found among
        // the source facets on the same carrier.
        let mut face_of_tri = vec![u32::MAX; source.plc.triangles.len()];
        for (fi, f) in src.faces.iter().enumerate() {
            for &t in &f.facets {
                face_of_tri[t as usize] = fi as u32;
            }
        }
        let index = source.index();
        let face: Vec<Option<u32>> = brep
            .faces
            .iter()
            .map(|f| {
                if f.face_tag == CUT_TAG {
                    return None;
                }
                let t = *f.facets.first()?;
                let tv = model.plc.triangles[t as usize].map(|v| model.plc.vertices[v as usize]);
                let centre: P3 = std::array::from_fn(|k| (tv[0][k] + tv[1][k] + tv[2][k]) / 3.0);
                let on = model.plc.surface_refs[t as usize];
                index
                    .nearest_where(centre, &|s| source.plc.surface_refs[s as usize] == on)
                    .map(|(s, _)| face_of_tri[s as usize])
                    .filter(|&f| f != u32::MAX)
            })
            .collect();
        // Each edge: a piece of a source edge where it meets no cut, else on
        // the source face it crosses (or inside a region).
        let mut edges_of_face: Vec<Vec<u32>> = vec![Vec::new(); src.faces.len()];
        for (ei, e) in src.edges.iter().enumerate() {
            for &c in &e.coedges {
                edges_of_face[src.coedge(c).face.0 as usize].push(ei as u32);
            }
        }
        let edge: Vec<PointClass> = brep
            .edges
            .iter()
            .map(|e| {
                let faces: Vec<Option<u32>> = e
                    .coedges
                    .iter()
                    .map(|&c| face[brep.coedge(c).face.0 as usize])
                    .collect();
                let real: Vec<u32> = faces.iter().flatten().copied().collect();
                if faces.iter().any(|f| f.is_none()) {
                    return real
                        .first()
                        .map_or(PointClass::Interior, |&f| PointClass::Face(f));
                }
                let Some(&f) = real.first() else {
                    return PointClass::Interior;
                };
                let mid = midpoint(&e.chain);
                edges_of_face[f as usize]
                    .iter()
                    .map(|&se| (se, polyline_dist(&src.edges[se as usize].chain, mid)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map_or(PointClass::Face(f), |(se, _)| PointClass::Edge(se))
            })
            .collect();
        // Each vertex: a source corner at the same place, else what its
        // edges lie on (an edge before a face before the volume).
        let corner: FxHashMap<[u64; 3], u32> = src
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| (v.pos.map(f64::to_bits), i as u32))
            .collect();
        let mut vertex: Vec<PointClass> = brep
            .vertices
            .iter()
            .map(|v| match corner.get(&v.pos.map(f64::to_bits)) {
                Some(&i) => PointClass::Vertex(i),
                None => PointClass::Interior,
            })
            .collect();
        let rank = |c: PointClass| match c {
            PointClass::Vertex(_) => 3,
            PointClass::Edge(_) => 2,
            PointClass::Face(_) => 1,
            PointClass::Interior => 0,
        };
        for (ei, e) in brep.edges.iter().enumerate() {
            for end in e.ends {
                let v = &mut vertex[end.0 as usize];
                if rank(edge[ei]) > rank(*v) {
                    *v = edge[ei];
                }
            }
        }
        // Every corner of the source is one of the cut model's.
        let mut seen = vec![false; src.vertices.len()];
        for v in &vertex {
            if let PointClass::Vertex(i) = *v {
                seen[i as usize] = true;
            }
        }
        if seen.iter().any(|&s| !s)
            || face
                .iter()
                .zip(&brep.faces)
                .any(|(f, bf)| f.is_none() && bf.face_tag != CUT_TAG)
        {
            return None;
        }
        Some(Blocks {
            model,
            region,
            face,
            edge,
            vertex,
        })
    }

    /// The number of blocks.
    pub fn count(&self) -> usize {
        self.region.len() - 1
    }

    /// The source region of region `r` of the cut model.
    pub fn region(&self, r: u32) -> u32 {
        self.region[r as usize]
    }

    /// Whether face `f` of the cut model is a cut or a piece of a face a
    /// cut splits: one a cut can be blamed for.
    pub fn at_cut(&self, f: u32) -> bool {
        match self.face(f) {
            None => true,
            Some(src) => self.face.iter().filter(|&&x| x == Some(src)).count() > 1,
        }
    }

    /// The source face of face `f` of the cut model, `None` on a cut.
    pub fn face(&self, f: u32) -> Option<u32> {
        self.face[f as usize]
    }

    /// The source edge of edge `e` of the cut model, where it is a piece
    /// of one.
    pub fn edge(&self, e: u32) -> Option<u32> {
        match self.edge[e as usize] {
            PointClass::Edge(s) => Some(s),
            _ => None,
        }
    }

    /// What a point of class `c` in the cut model lies on in the source.
    pub fn class(&self, c: PointClass) -> PointClass {
        match c {
            PointClass::Vertex(v) => self.vertex[v as usize],
            PointClass::Edge(e) => self.edge[e as usize],
            PointClass::Face(f) => {
                self.face[f as usize].map_or(PointClass::Interior, PointClass::Face)
            }
            PointClass::Interior => PointClass::Interior,
        }
    }

    /// The source vertex at vertex `v` of the cut model, if there is one.
    pub fn corner(&self, v: u32) -> Option<u32> {
        match self.vertex[v as usize] {
            PointClass::Vertex(s) => Some(s),
            _ => None,
        }
    }

    /// `params`, its per-face and per-edge settings on the pieces.
    pub fn params(&self, params: &MeshParams) -> MeshParams {
        let mut p = params.clone();
        let pieces = |of: &[(u32, f64)], map: &dyn Fn(usize) -> Option<u32>, n: usize| {
            (0..n)
                .filter_map(|i| {
                    let s = map(i)?;
                    of.iter().find(|x| x.0 == s).map(|x| (i as u32, x.1))
                })
                .collect::<Vec<_>>()
        };
        let (nf, ne) = (self.face.len(), self.edge.len());
        p.surf_maxh = pieces(&params.surf_maxh, &|i| self.face[i], nf);
        p.surf_tol = pieces(&params.surf_tol, &|i| self.face[i], nf);
        p.edge_maxh = pieces(&params.edge_maxh, &|i| self.edge(i as u32), ne);
        p.edge_tol = pieces(&params.edge_tol, &|i| self.edge(i as u32), ne);
        p
    }
}

/// The point halfway along a polyline.
fn midpoint(chain: &[P3]) -> P3 {
    let len = |w: &[P3]| dot(sub(w[1], w[0]), sub(w[1], w[0])).sqrt();
    let total: f64 = chain.windows(2).map(len).sum();
    let mut left = 0.5 * total;
    for w in chain.windows(2) {
        let l = len(w);
        if l >= left && l > 0.0 {
            let t = left / l;
            return std::array::from_fn(|k| w[0][k] + t * (w[1][k] - w[0][k]));
        }
        left -= l;
    }
    chain.first().copied().unwrap_or([0.0; 3])
}

/// The distance from `p` to a polyline.
fn polyline_dist(chain: &[P3], p: P3) -> f64 {
    chain
        .windows(2)
        .map(|w| {
            let d = sub(w[1], w[0]);
            let t = (dot(sub(p, w[0]), d) / dot(d, d).max(1e-300)).clamp(0.0, 1.0);
            let q: P3 = std::array::from_fn(|k| w[0][k] + t * d[k]);
            dot(sub(p, q), sub(p, q)).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::TetMesh;
    use rapidmesh_geom::solid_box;

    fn volumes(m: &TetMesh) -> FxHashMap<u32, f64> {
        let mut out = FxHashMap::default();
        for (t, r) in m.tets.iter().zip(&m.tet_regions) {
            let [a, b, c, d] = t.map(|v| m.points[v]);
            *out.entry(r.0).or_default() += dot(sub(b, a), cross(sub(c, a), sub(d, a))).abs() / 6.0;
        }
        out
    }

    fn areas(m: &TetMesh) -> FxHashMap<u32, f64> {
        let mut out = FxHashMap::default();
        for f in &m.faces {
            let [a, b, c] = f.tri.map(|v| m.points[v]);
            let n = cross(sub(b, a), sub(c, a));
            *out.entry(f.patch).or_default() += 0.5 * dot(n, n).sqrt();
        }
        out
    }

    fn same(x: &FxHashMap<u32, f64>, y: &FxHashMap<u32, f64>) -> bool {
        x.len() == y.len()
            && x.iter().all(|(k, a)| {
                y.get(k)
                    .is_some_and(|b| (a - b).abs() <= 1e-9 * a.abs().max(1.0))
            })
    }

    /// A model meshed in blocks is the model meshed whole: the same volume
    /// in each region and area on each face, positive tets, no cut face
    /// left, and its corners first and in order.
    #[test]
    fn blocks_mesh_the_model_they_cut() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0; 3], [4.0, 2.0, 2.0]));
        scene.add_solid(solid_box([1.3, 0.6, 0.7], [2.9, 1.4, 1.3]));
        let model = Model::try_of_scene(&scene).unwrap();
        let params = MeshParams {
            maxh: 0.3,
            ..MeshParams::default()
        };
        let domain = crate::sizing::build_sizing_domain(&model, &params);
        let block = 800.0;
        let plan = Plan::new(&model, &domain, Some(block));
        assert!(plan.cells() >= 4);
        assert!(Blocks::new(&scene, &model, &plan).is_some_and(|b| b.count() >= 8));
        let whole = crate::mesher::mesh_in_blocks(&model, None, &params).unwrap();
        let cut =
            crate::mesher::mesh_in_blocks(&model, Some((&scene, Some(block))), &params).unwrap();
        assert!(same(&volumes(&whole), &volumes(&cut)));
        assert!(same(&areas(&whole), &areas(&cut)));
        for t in &cut.tets {
            let [a, b, c, d] = t.map(|v| cut.points[v]);
            assert!(geometry_predicates::orient3d(a, b, c, d) > 0.0);
        }
        assert!(cut
            .faces
            .iter()
            .all(|f| (f.patch as usize) < model.brep.faces.len()));
        for (v, corner) in model.brep.vertices.iter().enumerate() {
            assert_eq!(cut.points[v], corner.pos);
            assert_eq!(cut.point_class[v], PointClass::Vertex(v as u32));
        }
    }
}
