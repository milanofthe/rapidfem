//! Builds a [`Brep`] from the exact CSG output (`TaggedPlc`).
//!
//! The tagged triangle soup is the source of truth for TOPOLOGY (which surfaces
//! meet, region labels, exact vertex positions). This step RECONSTRUCTS the
//! boundary representation from it -- groups triangles into faces, chains their
//! boundary edges into B-rep edges, recovers an analytic curve per edge, orders
//! the loops, and radially links faces. Nothing is snapped: positions, regions
//! and incidence come unchanged from the arrangement; only analytic curves are
//! added on top.
//!
//! Both the CSG path and the STEP-import path converge on `TaggedPlc`, so this
//! one function covers both.

use crate::{
    Brep, CoEdge, CoEdgeId, Curve, Edge, EdgeId, Face, FaceId, Loop, Surface, SurfaceId, Vertex,
    VertexId,
};
use rapidmesh_geom::vec3::{add, cross, dist, dot, normalize as norm, scale, sub, V3};
use rapidmesh_geom::{NurbsCurve, SurfaceKind, TaggedPlc};
use std::sync::Arc;
// Deterministic (seedless) hashers: from_plc's map ITERATION order sets the
// B-rep edge / face / vertex order, which flows into the surface point order and
// the mesh -- std's RandomState would make the whole mesh vary run to run.
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// Turn cosine below which a degree-2 vertex along a face without an exact
/// carrier is still a corner (45 deg), matching the mesher's feature-edge
/// splitter.
const CORNER_COS: f64 = 0.707;

fn key2(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn uf_find(rep: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while rep[r] != r {
        r = rep[r];
    }
    let mut c = x;
    while rep[c] != c {
        let nx = rep[c];
        rep[c] = r;
        c = nx;
    }
    r
}
fn uf_union(rep: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (uf_find(rep, a), uf_find(rep, b));
    if ra != rb {
        rep[ra.max(rb)] = ra.min(rb);
    }
}

/// Build a B-rep from a tagged PLC (pure function; no CSG state, no snapping).
pub fn from_plc(plc: &TaggedPlc) -> Brep {
    let pos: &[V3] = &plc.vertices;
    let tri = |i: usize| {
        let t = plc.triangles[i];
        [t[0] as usize, t[1] as usize, t[2] as usize]
    };
    let n_tri = plc.triangles.len();
    let mut diag = 0.0f64;
    {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for p in pos {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        for k in 0..3 {
            diag = diag.max(hi[k] - lo[k]);
        }
    }
    let tol = 1e-9 * diag.max(1.0);

    // ---- B1: group triangles into faces (key + connected component) ----------
    // Key = (analytic surface, unordered region pair, face tag). Within a key,
    // triangles connected through a shared edge are ONE face (two disjoint
    // patches of the same surface stay separate).
    let tkey = |i: usize| -> (u32, u32, u32, u32) {
        let r = plc.region_tags[i];
        (
            plc.surface_refs[i].0,
            r[0].0.min(r[1].0),
            r[0].0.max(r[1].0),
            plc.face_tags[i].0,
        )
    };
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
    for i in 0..n_tri {
        let c = tri(i);
        for e in 0..3 {
            edge_tris
                .entry(key2(c[e], c[(e + 1) % 3]))
                .or_default()
                .push(i);
        }
    }
    let mut frep: Vec<usize> = (0..n_tri).collect();
    for tris in edge_tris.values() {
        for a in 0..tris.len() {
            for b in (a + 1)..tris.len() {
                if tkey(tris[a]) == tkey(tris[b]) {
                    uf_union(&mut frep, tris[a], tris[b]);
                }
            }
        }
    }
    // Component representative -> face id; build the Face records.
    let mut face_of_rep: HashMap<usize, usize> = HashMap::default();
    let mut faces: Vec<Face> = Vec::new();
    let mut tri_face: Vec<usize> = vec![usize::MAX; n_tri];
    for i in 0..n_tri {
        let r = uf_find(&mut frep, i);
        let fid = *face_of_rep.entry(r).or_insert_with(|| {
            let rt = plc.region_tags[i];
            let sid = plc.surface_refs[i].0;
            faces.push(Face {
                surface: SurfaceId(sid),
                loops: Vec::new(),
                regions: rt,
                face_tag: plc.face_tags[i],
                plc_surface: sid,
                owner: plc.surface_owners[sid as usize],
                role: 0,
                facets: Vec::new(),
            });
            faces.len() - 1
        });
        tri_face[i] = fid;
        faces[fid].facets.push(i as u32);
    }

    absorb_thin_faces(plc, &mut faces, &mut tri_face, &edge_tris, tol);

    // ---- boundary edges per face, and the radial face set per edge -----------
    // A face's boundary edge is used by exactly one of its triangles (interior
    // edges by two). The set of faces sharing a boundary edge is its radial set.
    let mut bedge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
    {
        // count (face, edge) uses
        let mut fe_count: HashMap<(usize, (usize, usize)), usize> = HashMap::default();
        for i in 0..n_tri {
            let c = tri(i);
            let f = tri_face[i];
            for e in 0..3 {
                *fe_count.entry((f, key2(c[e], c[(e + 1) % 3]))).or_insert(0) += 1;
            }
        }
        for ((f, e), cnt) in fe_count {
            if cnt == 1 {
                bedge_faces.entry(e).or_default().push(f);
            }
        }
    }
    for v in bedge_faces.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    // Feature edges inside a face (an import's open crease the face wraps
    // around): the boundary of no face, yet B-rep edges, with the faces
    // around them. They take part in no loop.
    let mut inner: HashSet<(usize, usize)> = HashSet::default();
    for f in &plc.features {
        let e = key2(f[0] as usize, f[1] as usize);
        if bedge_faces.contains_key(&e) {
            continue;
        }
        if let Some(ts) = edge_tris.get(&e) {
            let mut fs: Vec<usize> = ts.iter().map(|&t| tri_face[t]).collect();
            fs.sort_unstable();
            fs.dedup();
            bedge_faces.insert(e, fs);
            inner.insert(e);
        }
    }

    // ---- B3: chain boundary edges into B-rep edges, split at corners ---------
    // The boundary graph: vertices linked by boundary edges. Walk maximal chains
    // that keep the SAME radial face set, splitting at junctions (degree != 2),
    // at a change of the face set, and at sharp turns (> 45 deg).
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::default();
    for &(a, b) in bedge_faces.keys() {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    let fset = |a: usize, b: usize| -> &Vec<usize> { &bedge_faces[&key2(a, b)] };
    let carriers: Vec<Option<Surface>> = faces.iter().map(|f| face_carrier(f, plc, tol)).collect();
    // Two carriers touching tangentially at `p`: the direction of their
    // intersection is not defined there, and a turn of the chain is the
    // faceting crossing itself (the zigzag along a capsule seam).
    let tangent_at = |fs: &[usize], p: V3| -> bool {
        let [a, b] = fs else {
            return false;
        };
        match (&carriers[*a], &carriers[*b]) {
            (Some(sa), Some(sb)) => dot(sa.closest(p).1, sb.closest(p).1).abs() > TANGENT_COS,
            _ => false,
        }
    };
    // A corner is where the chain branches or ends, where the faces along
    // it change, or where an input shape declares one. Along faces with
    // exact carriers nothing else is: a turn of a cut between two carriers
    // is its sampling. Only where a face has none (an import, a faceted wall
    // off its plane) is a sharp turn the one hint left.
    let declared: HashSet<usize> = plc.corners.iter().map(|&v| v as usize).collect();
    let is_corner = |v: usize, adj: &HashMap<usize, Vec<usize>>| -> bool {
        let ns = &adj[&v];
        if ns.len() != 2 {
            return true;
        }
        let fs = fset(v, ns[0]);
        if fs != fset(v, ns[1]) || declared.contains(&v) {
            return true;
        }
        if fs.iter().all(|&f| carriers[f].is_some()) || tangent_at(fs, pos[v]) {
            return false;
        }
        let d0 = norm(sub(pos[v], pos[ns[0]]));
        let d1 = norm(sub(pos[ns[1]], pos[v]));
        dot(d0, d1) < CORNER_COS
    };
    let walk = |c0: usize,
                start: usize,
                adj: &HashMap<usize, Vec<usize>>,
                done: &mut HashSet<(usize, usize)>|
     -> Vec<usize> {
        let mut chain = vec![c0];
        let (mut prev, mut cur) = (c0, start);
        loop {
            chain.push(cur);
            done.insert(key2(prev, cur));
            if is_corner(cur, adj) || cur == c0 {
                break;
            }
            let ns = &adj[&cur];
            let nxt = if ns[0] == prev { ns[1] } else { ns[0] };
            prev = cur;
            cur = nxt;
            if chain.len() > adj.len() + 2 {
                break;
            }
        }
        chain
    };
    let mut chains: Vec<Vec<usize>> = Vec::new();
    let mut done: HashSet<(usize, usize)> = HashSet::default();
    let mut corners: Vec<usize> = adj
        .keys()
        .copied()
        .filter(|&v| is_corner(v, &adj))
        .collect();
    corners.sort_unstable();
    for &c0 in &corners {
        for &start in &adj[&c0].clone() {
            if !done.contains(&key2(c0, start)) {
                chains.push(walk(c0, start, &adj, &mut done));
            }
        }
    }
    // Corner-less loops (a smooth rim): anchor at the lowest-index vertex.
    let mut keys: Vec<usize> = adj.keys().copied().collect();
    keys.sort_unstable();
    for &a in &keys {
        for &b in &adj[&a].clone() {
            if !done.contains(&key2(a, b)) {
                let mut ch = walk(a, b, &adj, &mut done);
                if ch.last() != Some(&a) {
                    ch.push(a);
                }
                chains.push(ch);
            }
        }
    }

    // ---- B2: vertices = unique chain endpoints --------------------------------
    let mut vid: HashMap<usize, VertexId> = HashMap::default();
    let mut vertices: Vec<Vertex> = Vec::new();
    let corner_id = |plc_v: usize,
                     vid: &mut HashMap<usize, VertexId>,
                     vertices: &mut Vec<Vertex>|
     -> VertexId {
        *vid.entry(plc_v).or_insert_with(|| {
            vertices.push(Vertex { pos: pos[plc_v] });
            VertexId((vertices.len() - 1) as u32)
        })
    };

    // ---- B3 cont.: build Edge records (curve recovery), keep radial faces ----
    let mut edges: Vec<Edge> = Vec::new();
    let mut edge_faces: Vec<Vec<FaceId>> = Vec::new();
    let mut is_inner: Vec<bool> = Vec::new();

    for ch in &chains {
        let a = ch[0];
        let b = *ch.last().unwrap();
        let va = corner_id(a, &mut vid, &mut vertices);
        let vb = corner_id(b, &mut vid, &mut vertices);
        let chain_pts: Vec<V3> = ch.iter().map(|&v| pos[v]).collect();
        // radial faces: the face set of the chain's segments (constant by the
        // same-face-set split, so the first segment suffices).
        let mut rad: Vec<FaceId> = fset(ch[0], ch[1])
            .iter()
            .map(|&f| FaceId(f as u32))
            .collect();
        rad.sort_unstable();
        let curve = recover_curve(&chain_pts, &rad, &faces, plc, tol);
        edges.push(Edge {
            ends: [va, vb],
            chain: chain_pts,
            curve,
            coedges: Vec::new(),
        });
        edge_faces.push(rad);
        is_inner.push(inner.contains(&key2(ch[0], ch[1])));
    }

    // ---- B4/B5: per face, build its self-contained surface, order loops, and
    // make one co-edge per (edge, loop-direction). A plane gets its frame from
    // an originating facet; every other kind is self-contained from its
    // parameters.
    let mut face_edges: Vec<Vec<usize>> = vec![Vec::new(); faces.len()];
    for (ei, ef) in edge_faces.iter().enumerate() {
        if is_inner[ei] {
            continue;
        }
        for f in ef {
            face_edges[f.0 as usize].push(ei);
        }
    }
    let mut surfaces: Vec<Surface> = Vec::new();
    let mut coedges: Vec<CoEdge> = Vec::new();
    for fid in 0..faces.len() {
        let signed = order_loops(&face_edges[fid], &edges);
        // Frame for the surface. For a plane this must be EXACT (so on-plane
        // carriers stay bit-exact on the PLC plane -> exact region volumes): use an
        // originating facet triangle (exact PLC vertices, cross-product normal),
        // not the float edge points. Other kinds ignore the frame (self-contained).
        let frame_pts: Vec<V3> = if let Some(&tfi) = faces[fid].facets.first() {
            let t = plc.triangles[tfi as usize];
            vec![
                plc.vertices[t[0] as usize],
                plc.vertices[t[1] as usize],
                plc.vertices[t[2] as usize],
            ]
        } else {
            signed
                .first()
                .map(|lp| loop_points(lp, &edges))
                .unwrap_or_default()
        };
        let mut kind = plc.surfaces[faces[fid].surface.0 as usize].clone();
        // A `Plane` kind whose facets are NOT coplanar is a faceted CURVED
        // face without an analytic recovery -- loft mantles, swept tubes,
        // helix coils all tag their whole side wall as one Plane surface. A
        // plane fit through such a face is a garbage carrier (the refinement
        // core projects and classifies against it, shredding the mesh into
        // fragments). Carry it as a DISCRETE patch of its own facets instead:
        // the same closest-point oracle that remeshes STL imports.
        if matches!(kind, SurfaceKind::Plane) && !face_facets_coplanar(&faces[fid], plc, tol) {
            let mut vmap: HashMap<usize, u32> = HashMap::default();
            let mut dpoints: Vec<V3> = Vec::new();
            let mut dtris: Vec<[u32; 3]> = Vec::new();
            for &tfi in &faces[fid].facets {
                let t = plc.triangles[tfi as usize];
                let ids: [u32; 3] = std::array::from_fn(|k| {
                    *vmap.entry(t[k] as usize).or_insert_with(|| {
                        dpoints.push(plc.vertices[t[k] as usize]);
                        (dpoints.len() - 1) as u32
                    })
                });
                dtris.push(ids);
            }
            kind = SurfaceKind::Discrete(std::sync::Arc::new(
                rapidmesh_geom::DiscreteSurface::new(dpoints, dtris),
            ));
        }
        let sid = SurfaceId(surfaces.len() as u32);
        // One surface per face, in face order: `Curve::Intersection` (built in
        // recover_curve, before this loop) references faces' surfaces by this
        // identity, so it must hold.
        debug_assert_eq!(sid.0 as usize, fid, "surface id must equal face id");
        surfaces.push(Surface::from_kind(&kind, &frame_pts));
        faces[fid].surface = sid;
        let mut loops_out: Vec<Loop> = Vec::new();
        for sl in &signed {
            let mut lp = Loop::default();
            for &(ei, fwd) in sl {
                let cid = CoEdgeId(coedges.len() as u32);
                coedges.push(CoEdge {
                    edge: EdgeId(ei as u32),
                    face: FaceId(fid as u32),
                    forward: fwd,
                });
                edges[ei].coedges.push(cid);
                lp.coedges.push(cid);
            }
            loops_out.push(lp);
        }
        faces[fid].loops = loops_out;
    }
    // Inner edges: one co-edge per face around them, outside every loop.
    for (ei, ef) in edge_faces.iter().enumerate() {
        if !is_inner[ei] {
            continue;
        }
        for &f in ef {
            let cid = CoEdgeId(coedges.len() as u32);
            coedges.push(CoEdge {
                edge: EdgeId(ei as u32),
                face: f,
                forward: true,
            });
            edges[ei].coedges.push(cid);
        }
    }

    let mut b = Brep {
        vertices,
        edges,
        coedges,
        faces,
        surfaces,
    };
    canonicalize(&mut b, plc);
    b
}

/// Puts the entities in the order of their origin, so an id follows the
/// geometry rather than the order the build met it in: faces by owner
/// solid, role of their surface and face tag; edges by their faces;
/// vertices by their edges; co-edges by edge and face. Entities alike in
/// that (the parts of one surface, two edges between the same faces) go by
/// where they lie in the frame of their solid, which turning or moving the
/// solid keeps. An id then stays when other shapes are added after, when
/// the scene is turned, and when a parameter changes that leaves the faces
/// as they are.
fn canonicalize(b: &mut Brep, plc: &TaggedPlc) {
    fn inverse(order: &[usize]) -> Vec<u32> {
        let mut inv = vec![0u32; order.len()];
        for (n, &o) in order.iter().enumerate() {
            inv[o] = n as u32;
        }
        inv
    }
    let role = |s: u32| plc.surface_roles.get(s as usize).copied().unwrap_or(s);
    for f in &mut b.faces {
        f.role = role(f.plc_surface);
    }
    // Where a point lies in the frame of solid `owner`, on a grid of a
    // billionth of the model, so rounding does not reorder.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &plc.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let grid = 1e-9
        * (0..3)
            .map(|k| hi[k] - lo[k])
            .fold(0.0, f64::max)
            .max(1e-300);
    let local = |owner: u32, p: V3| -> [i64; 3] {
        let q = match plc.owner_frames.get(owner as usize) {
            Some(fr) => fr.to_local(p),
            None => p,
        };
        q.map(|x| (x / grid).round() as i64)
    };
    let centroid = |f: &Face| -> V3 {
        let (mut c, mut w) = ([0.0; 3], 0.0);
        for &t in &f.facets {
            let v = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
            let n = cross(sub(v[1], v[0]), sub(v[2], v[0]));
            let a = 0.5 * dot(n, n).sqrt();
            for k in 0..3 {
                c[k] += a * (v[0][k] + v[1][k] + v[2][k]) / 3.0;
            }
            w += a;
        }
        c.map(|x| x / w.max(1e-300))
    };

    // Faces, and their surfaces with them.
    let mut order: Vec<usize> = (0..b.faces.len()).collect();
    let fkeys: Vec<(u32, u32, u32, [i64; 3])> = b
        .faces
        .iter()
        .map(|f| (f.owner, f.role, f.face_tag.0, local(f.owner, centroid(f))))
        .collect();
    order.sort_by(|&x, &y| fkeys[x].cmp(&fkeys[y]));
    let fnew = inverse(&order);
    let mut faces: Vec<Face> = order.iter().map(|&i| b.faces[i].clone()).collect();
    let mut sorder: Vec<usize> = Vec::new();
    let mut snew = vec![u32::MAX; b.surfaces.len()];
    for f in &faces {
        let s = f.surface.0 as usize;
        if snew[s] == u32::MAX {
            snew[s] = sorder.len() as u32;
            sorder.push(s);
        }
    }
    for (s, n) in snew.iter_mut().enumerate() {
        if *n == u32::MAX {
            *n = sorder.len() as u32;
            sorder.push(s);
        }
    }
    let surfaces: Vec<Surface> = sorder.iter().map(|&s| b.surfaces[s].clone()).collect();

    // Edges by the faces around them, then where they lie in the frame of
    // the first face's solid.
    let owner_of = |f: u32| faces[f as usize].owner;
    let edge_key = |e: &Edge| {
        let mut fs: Vec<u32> = e
            .coedges
            .iter()
            .map(|c| fnew[b.coedges[c.0 as usize].face.0 as usize])
            .collect();
        fs.sort_unstable();
        fs.dedup();
        let n = e.chain.len().max(1) as f64;
        let mid: V3 = std::array::from_fn(|k| e.chain.iter().map(|p| p[k]).sum::<f64>() / n);
        let at = local(fs.first().map_or(u32::MAX, |&f| owner_of(f)), mid);
        (fs, at)
    };
    let mut order: Vec<usize> = (0..b.edges.len()).collect();
    let keys: Vec<(Vec<u32>, [i64; 3])> = b.edges.iter().map(edge_key).collect();
    order.sort_by(|&x, &y| keys[x].cmp(&keys[y]));
    let enew = inverse(&order);

    // Vertices by the edges ending at them.
    let mut ends: Vec<Vec<u32>> = vec![Vec::new(); b.vertices.len()];
    for (e, edge) in b.edges.iter().enumerate() {
        for v in edge.ends {
            ends[v.0 as usize].push(enew[e]);
        }
    }
    for l in &mut ends {
        l.sort_unstable();
    }
    let vowner: Vec<u32> = ends
        .iter()
        .map(|es| {
            es.iter()
                .filter_map(|&e| keys[order[e as usize]].0.first().copied())
                .min()
                .map_or(u32::MAX, owner_of)
        })
        .collect();
    let vkeys: Vec<(&Vec<u32>, [i64; 3])> = (0..b.vertices.len())
        .map(|v| (&ends[v], local(vowner[v], b.vertices[v].pos)))
        .collect();
    let mut vorder: Vec<usize> = (0..b.vertices.len()).collect();
    vorder.sort_by(|&x, &y| vkeys[x].cmp(&vkeys[y]));
    let vnew = inverse(&vorder);

    // Co-edges by edge and face.
    let mut corder: Vec<usize> = (0..b.coedges.len()).collect();
    corder.sort_by_key(|&c| {
        let ce = &b.coedges[c];
        (
            enew[ce.edge.0 as usize],
            fnew[ce.face.0 as usize],
            !ce.forward,
            c,
        )
    });
    let cnew = inverse(&corder);

    for f in &mut faces {
        f.surface = SurfaceId(snew[f.surface.0 as usize]);
        for l in &mut f.loops {
            for c in &mut l.coedges {
                *c = CoEdgeId(cnew[c.0 as usize]);
            }
        }
    }
    let edges: Vec<Edge> = order
        .iter()
        .map(|&e| {
            let mut edge = b.edges[e].clone();
            edge.ends = edge.ends.map(|v| VertexId(vnew[v.0 as usize]));
            for c in &mut edge.coedges {
                *c = CoEdgeId(cnew[c.0 as usize]);
            }
            edge.coedges.sort_unstable_by_key(|c| c.0);
            if let Curve::Intersection { a, b: s } = &mut edge.curve {
                *a = SurfaceId(snew[a.0 as usize]);
                *s = SurfaceId(snew[s.0 as usize]);
            }
            edge
        })
        .collect();
    let coedges: Vec<CoEdge> = corder
        .iter()
        .map(|&c| {
            let ce = &b.coedges[c];
            CoEdge {
                edge: EdgeId(enew[ce.edge.0 as usize]),
                face: FaceId(fnew[ce.face.0 as usize]),
                forward: ce.forward,
            }
        })
        .collect();
    let vertices: Vec<Vertex> = vorder.iter().map(|&v| b.vertices[v].clone()).collect();
    *b = Brep {
        vertices,
        edges,
        coedges,
        faces,
        surfaces,
    };
}

/// Ordered 3D points along a signed-edge loop (chains concatenated, reversed where
/// the loop runs backward), used to fit a planar face's chart frame.
fn loop_points(sl: &[(usize, bool)], edges: &[Edge]) -> Vec<V3> {
    let mut pts: Vec<V3> = Vec::new();
    for &(ei, fwd) in sl {
        let ch = &edges[ei].chain;
        let seq: Vec<V3> = if fwd {
            ch.clone()
        } else {
            ch.iter().rev().cloned().collect()
        };
        for p in seq {
            if pts.last().map(|&q| dist(q, p) > 1e-12).unwrap_or(true) {
                pts.push(p);
            }
        }
    }
    pts
}

/// Orders a face's edges into oriented loops by walking shared endpoints, as
/// sequences of `(edge index, forward)`. `forward` is true when the loop
/// traverses the edge from `ends[0]` to `ends[1]`. The largest-perimeter loop is
/// placed first (the outer boundary; the rest are holes).
fn order_loops(eids: &[usize], edges: &[Edge]) -> Vec<Vec<(usize, bool)>> {
    let mut adj: HashMap<u32, Vec<usize>> = HashMap::default();
    for &ei in eids {
        let [a, b] = edges[ei].ends;
        adj.entry(a.0).or_default().push(ei);
        if b.0 != a.0 {
            adj.entry(b.0).or_default().push(ei);
        }
    }
    let mut used = vec![false; edges.len()];
    let mut loops: Vec<(f64, Vec<(usize, bool)>)> = Vec::new();
    for &start in eids {
        if used[start] {
            continue;
        }
        let mut seq: Vec<(usize, bool)> = Vec::new();
        let mut perim = 0.0f64;
        let mut cur = start;
        let mut at = edges[start].ends[0].0; // walk leaving from ends[0]
        loop {
            if used[cur] {
                break;
            }
            used[cur] = true;
            let [a, b] = edges[cur].ends;
            let forward = at == a.0;
            let next_v = if forward { b.0 } else { a.0 };
            seq.push((cur, forward));
            perim += arc_len(&edges[cur].chain);
            let nxt = adj
                .get(&next_v)
                .and_then(|inc| inc.iter().copied().find(|&e| !used[e]));
            match nxt {
                Some(e) => {
                    cur = e;
                    at = next_v;
                }
                None => break,
            }
        }
        loops.push((perim, seq));
    }
    loops.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    loops.into_iter().map(|(_, l)| l).collect()
}

fn arc_len(chain: &[V3]) -> f64 {
    chain.windows(2).map(|w| dist(w[0], w[1])).sum()
}

/// Recovers the analytic curve of an edge from its vertex chain and the surfaces
/// of its radial faces. Handles the forms our scenes use; everything else falls
/// back to the faceted polyline (`Curve::Polyline`).
fn recover_curve(chain: &[V3], rad: &[FaceId], faces: &[Face], plc: &TaggedPlc, tol: f64) -> Curve {
    if chain.len() < 2 {
        return Curve::Polyline;
    }
    let (p0, pn) = (chain[0], chain[chain.len() - 1]);
    let len = arc_len(chain);

    // Straight: every chain point lies on the segment p0..pn.
    if len > 0.0 && dist(p0, pn) > tol {
        let dir = norm(sub(pn, p0));
        let straight = chain.iter().all(|&p| {
            let t = dot(sub(p, p0), dir);
            let foot: V3 = std::array::from_fn(|k| p0[k] + dir[k] * t);
            dist(p, foot) < tol.max(1e-7 * len)
        });
        if straight {
            return Curve::Line { p0, dir };
        }
    }

    // Circle: two coaxial surfaces of revolution meet in circles square to
    // their axis, from the carriers alone (see `revolution_circle`).
    if let Some((center, axis, radius, x)) = revolution_circle(chain, rad, faces, plc) {
        return Curve::Circle {
            center,
            axis,
            radius,
            x,
        };
    }

    // Oblique plane section of a cylinder: an exact ELLIPSE (the axis-perpendicular
    // case is the circle above). Derived from the cylinder's parameters and the
    // EXACT facet plane of the adjacent planar face -- chain-independent, so the
    // recovered ellipse lies exactly on both carriers (a fit to the faceted chain
    // would sit a sagitta inside, the straddler-sliver mechanism).
    let planes: Vec<(V3, V3)> = rad
        .iter()
        .filter(|f| {
            matches!(
                plc.surfaces[faces[f.0 as usize].plc_surface as usize],
                SurfaceKind::Plane
            )
        })
        .filter_map(|f| exact_face_plane(&faces[f.0 as usize], plc))
        .collect();
    for f in rad {
        let kind = &plc.surfaces[faces[f.0 as usize].plc_surface as usize];
        if let SurfaceKind::Cylinder {
            center,
            axis,
            radius,
        } = kind
        {
            for &(po, pn) in &planes {
                if let Some(e) =
                    plane_cylinder_ellipse(chain, po, pn, *center, norm(*axis), *radius, tol)
                {
                    return e;
                }
            }
        }
    }

    // On an extruded surface at constant height: the analytic profile curve.
    for f in rad {
        let sid = faces[f.0 as usize].surface;
        if let SurfaceKind::Extruded {
            profile,
            base,
            udir,
            vdir,
            axis,
        } = &plc.surfaces[sid.0 as usize]
        {
            let (u, v, a) = (norm(*udir), norm(*vdir), norm(*axis));
            let z0 = dot(sub(p0, *base), a);
            // constant extrusion height along the whole chain -> a profile edge
            let const_h = chain
                .iter()
                .all(|&p| (dot(sub(p, *base), a) - z0).abs() < tol.max(1e-7));
            if const_h {
                let foot = |p: V3| -> f64 {
                    let rel = sub(p, *base);
                    profile.closest_param([dot(rel, u), dot(rel, v)])
                };
                return Curve::Profile {
                    profile: profile.clone(),
                    base: *base,
                    u,
                    v,
                    axis: a,
                    t: [foot(p0), foot(pn)],
                    z: z0,
                };
            }
        }
    }

    // Two distinct analytic carriers, no closed form matched: the edge is their
    // intersection curve. The mesher densifies the chain and pulls every sample
    // onto BOTH surfaces (alternating projection), so the edge follows the true
    // curve instead of the coarse arrangement chain. Prefer two curved carriers,
    // else curved + plane; two planes intersect in a line (handled above).
    {
        let mut carriers: Vec<(bool, u32, FaceId)> = rad
            .iter()
            .map(|&f| {
                let sid = faces[f.0 as usize].plc_surface;
                (
                    matches!(plc.surfaces[sid as usize], SurfaceKind::Plane),
                    sid,
                    f,
                )
            })
            .collect();
        carriers.sort_unstable_by_key(|&(is_plane, sid, _)| (is_plane, sid));
        carriers.dedup_by_key(|c| c.1);
        if carriers.len() >= 2 && !carriers[0].0 {
            // NB: Curve::Intersection stores brep SurfaceIds; from_plc assigns one
            // surface per face IN FACE ORDER, so SurfaceId(fid) is that face's
            // surface (asserted in from_plc).
            return Curve::Intersection {
                a: SurfaceId(carriers[0].2 .0),
                b: SurfaceId(carriers[1].2 .0),
            };
        }
    }

    // No carrier pair gives the curve (an import's crease, a loft rim): the
    // edge is its chain, which is the geometry there.
    Curve::Polyline
}

/// The analytic carrier of a face; none for a discrete patch or a curved
/// wall tagged plane.
fn face_carrier(f: &Face, plc: &TaggedPlc, tol: f64) -> Option<Surface> {
    let kind = &plc.surfaces[f.surface.0 as usize];
    if matches!(kind, SurfaceKind::Discrete(_)) {
        return None;
    }
    if matches!(kind, SurfaceKind::Plane) && !face_facets_coplanar(f, plc, tol) {
        return None;
    }
    let t = plc.triangles[*f.facets.first()? as usize];
    let frame: Vec<V3> = t.iter().map(|&v| plc.vertices[v as usize]).collect();
    Some(Surface::from_kind(kind, &frame))
}

/// Carriers whose normals are this close to parallel (cos 10 deg) touch
/// tangentially.
const TANGENT_COS: f64 = 0.985;

/// A face at most this factor of the faceting errors of its neighbours wide
/// (mean width, twice its area over its perimeter) is an artifact of them.
const THIN_FACE: f64 = 2.0;

/// B1b: absorbs the thin faces the CSG of tangent faceted surfaces leaves
/// (#47). Where two carriers touch tangentially, their tessellations cross
/// each other along the contact: the union keeps strips between the two
/// polygons, a sagitta wide, as faces of their own (the cap plane of a
/// cylinder between its rim and the section of a sphere on it). A face no
/// wider than the faceting errors of its neighbours explain goes to one of
/// them, the one whose shared boundary lies worse on the other's carrier:
/// the boundary that stays lies on both carriers (the cylinder rim, which
/// is on the sphere too). Faces of the same kind that become adjacent
/// merge. Region pairs are kept; a plane takes only coplanar facets.
fn absorb_thin_faces(
    plc: &TaggedPlc,
    faces: &mut Vec<Face>,
    tri_face: &mut [usize],
    edge_tris: &HashMap<(usize, usize), Vec<usize>>,
    tol: f64,
) {
    let nf = faces.len();
    let pos = &plc.vertices;
    let tri_pts = |t: u32| plc.triangles[t as usize].map(|v| pos[v as usize]);
    // Each face's carrier (none for a discrete patch or a curved wall tagged
    // plane) and its faceting error: the largest distance of a facet
    // centroid to the carrier.
    let carrier: Vec<Option<Surface>> = faces.iter().map(|f| face_carrier(f, plc, tol)).collect();
    let off = |f: usize, p: V3| -> f64 {
        carrier[f]
            .as_ref()
            .map_or(f64::INFINITY, |s| dist(s.closest(p).0, p))
    };
    let facet_err = |f: usize, t: u32| -> f64 {
        if carrier[f].is_none() {
            return 0.0;
        }
        let q = tri_pts(t);
        off(f, scale(add(add(q[0], q[1]), q[2]), 1.0 / 3.0))
    };
    let err: Vec<f64> = (0..nf)
        .map(|f| {
            faces[f]
                .facets
                .iter()
                .map(|&t| facet_err(f, t))
                .fold(0.0, f64::max)
        })
        .collect();
    // Area, perimeter, and per neighbour the shared length, the shared
    // vertices and the faceting error of its facets along the boundary.
    let mut area = vec![0.0f64; nf];
    let mut perim = vec![0.0f64; nf];
    let mut shared: Vec<HashMap<usize, (f64, Vec<usize>, f64)>> = vec![HashMap::default(); nf];
    for (t, &f) in tri_face.iter().enumerate() {
        let q = tri_pts(t as u32);
        area[f] += 0.5
            * dot(
                cross(sub(q[1], q[0]), sub(q[2], q[0])),
                cross(sub(q[1], q[0]), sub(q[2], q[0])),
            )
            .sqrt();
        let c = plc.triangles[t];
        for e in 0..3 {
            let (a, b) = (c[e] as usize, c[(e + 1) % 3] as usize);
            let others: Vec<(usize, u32)> = edge_tris[&key2(a, b)]
                .iter()
                .map(|&u| (tri_face[u], u as u32))
                .filter(|&(g, _)| g != f)
                .collect();
            if others.is_empty() {
                continue;
            }
            let l = dist(pos[a], pos[b]);
            perim[f] += l;
            for (g, u) in others {
                let e = facet_err(g, u);
                let s = shared[f].entry(g).or_insert((0.0, Vec::new(), 0.0));
                s.0 += l;
                s.1.extend([a, b]);
                s.2 = s.2.max(e);
            }
        }
    }
    let pair = |f: &Face| {
        let r = f.regions;
        (r[0].0.min(r[1].0), r[0].0.max(r[1].0))
    };
    let mut order: Vec<usize> = (0..nf).filter(|&f| perim[f] > 0.0).collect();
    let width = |f: usize| 2.0 * area[f] / perim[f];
    order.sort_by(|&a, &b| width(a).total_cmp(&width(b)).then(a.cmp(&b)));
    let mut rep: Vec<usize> = (0..nf).collect();
    for f in order {
        let mut nbrs: Vec<(usize, f64)> = shared[f].iter().map(|(&g, s)| (g, s.0)).collect();
        nbrs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        nbrs.truncate(2);
        // The faceting errors there: of the face, and of its neighbours'
        // facets along it (a leading edge far away does not count).
        let explained: f64 = err[f] + nbrs.iter().map(|&(g, _)| shared[f][&g].2).sum::<f64>();
        if nbrs.is_empty() || !(width(f) <= THIN_FACE * explained) {
            continue;
        }
        // The boundary with neighbour g, measured on the carrier of the
        // other neighbour.
        let mismatch = |g: usize| -> f64 {
            let Some(&(o, _)) = nbrs.iter().find(|n| n.0 != g) else {
                return 0.0;
            };
            shared[f][&g]
                .1
                .iter()
                .map(|&v| off(o, pos[v]))
                .fold(0.0, f64::max)
        };
        let fits = |g: usize| -> bool {
            if pair(&faces[g]) != pair(&faces[f]) || carrier[g].is_none() {
                return false;
            }
            let plane = matches!(
                plc.surfaces[faces[g].surface.0 as usize],
                SurfaceKind::Plane
            );
            !plane
                || faces[f]
                    .facets
                    .iter()
                    .all(|&t| tri_pts(t).iter().all(|&p| off(g, p) <= tol))
        };
        let target = nbrs
            .iter()
            .map(|&(g, _)| g)
            .filter(|&g| fits(g))
            .max_by(|&a, &b| mismatch(a).total_cmp(&mismatch(b)).then(b.cmp(&a)));
        if let Some(g) = target {
            let (rf, rg) = (uf_find(&mut rep, f), uf_find(&mut rep, g));
            if rf != rg {
                rep[rf] = rg;
            }
        }
    }
    // Faces of one kind that the absorbed ones joined merge as well.
    let key = |f: &Face| (f.surface.0, pair(f), f.face_tag.0);
    for f in 0..nf {
        for &g in shared[f].keys() {
            let (rf, rg) = (uf_find(&mut rep, f), uf_find(&mut rep, g));
            if rf != rg && key(&faces[rf]) == key(&faces[rg]) {
                rep[rf] = rg;
            }
        }
    }
    if (0..nf).all(|f| rep[f] == f) {
        return;
    }
    // Rebuild the faces from the representatives, in order.
    let mut new_id: HashMap<usize, usize> = HashMap::default();
    let mut out: Vec<Face> = Vec::new();
    for f in 0..nf {
        let r = uf_find(&mut rep, f);
        if let std::collections::hash_map::Entry::Vacant(e) = new_id.entry(r) {
            e.insert(out.len());
            let mut face = faces[r].clone();
            face.facets.clear();
            out.push(face);
        }
    }
    for (t, tf) in tri_face.iter_mut().enumerate() {
        let r = uf_find(&mut rep, *tf);
        *tf = new_id[&r];
        out[*tf].facets.push(t as u32);
    }
    *faces = out;
}

/// True if every facet vertex of the face lies on the plane of its FIRST
/// facet (within `tol`): the gate that separates a real planar face from a
/// faceted curved side wall mis-tagged as `Plane`.
fn face_facets_coplanar(face: &Face, plc: &TaggedPlc, tol: f64) -> bool {
    let Some((o, n)) = exact_face_plane(face, plc) else {
        return true;
    };
    face.facets.iter().all(|&tfi| {
        let t = plc.triangles[tfi as usize];
        (0..3).all(|k| dot(sub(plc.vertices[t[k] as usize], o), n).abs() <= tol)
    })
}

/// The EXACT carrier plane of a planar face `(origin, unit normal)`, from its
/// first originating PLC facet (exact vertices, cross-product normal) -- not a
/// Newell fit to float edge points.
fn exact_face_plane(face: &Face, plc: &TaggedPlc) -> Option<(V3, V3)> {
    let &tfi = face.facets.first()?;
    let t = plc.triangles[tfi as usize];
    let (a, b, c) = (
        plc.vertices[t[0] as usize],
        plc.vertices[t[1] as usize],
        plc.vertices[t[2] as usize],
    );
    let n = cross(sub(b, a), sub(c, a));
    if dot(n, n) < 1e-24 {
        return None;
    }
    Some((a, norm(n)))
}

/// How close to square (or to parallel) a plane must meet an axis for the
/// cut to count as a circle (or as lines): a cosine within this of 1 (of 0);
/// likewise how close to an axis a centre must lie, relative to the edge.
/// The carriers agree to rounding when they are meant to.
const SQUARE_TOL: f64 = 1e-9;

/// How far apart two meridians may pass and still touch: a sphere on a
/// cylinder of its radius, faceted apart by rounding of their parameters.
const TOUCH_TOL: f64 = 1e-6;

/// A carrier's meridian: its section with a half-plane `(r, z)` of an axis
/// it is a surface of revolution about.
#[derive(Clone, Debug)]
enum Meridian {
    /// The points `p + t d` (unit `d`).
    Line { p: [f64; 2], d: [f64; 2] },
    /// The circle about `c` of radius `r`.
    Circle { c: [f64; 2], r: f64 },
    /// A profile `(r, z)` shifted by `z0` along the axis, its `z` scaled by
    /// `up` (-1 where the axis runs the other way).
    Profile {
        curve: Arc<NurbsCurve>,
        z0: f64,
        up: f64,
    },
}

impl Meridian {
    /// The point of the meridian nearest `q` and the unit normal there.
    fn foot(&self, q: [f64; 2]) -> ([f64; 2], [f64; 2]) {
        match *self {
            Meridian::Line { p, d } => {
                let t = (q[0] - p[0]) * d[0] + (q[1] - p[1]) * d[1];
                ([p[0] + t * d[0], p[1] + t * d[1]], [-d[1], d[0]])
            }
            Meridian::Circle { c, r } => {
                let (x, y) = (q[0] - c[0], q[1] - c[1]);
                let l = x.hypot(y);
                let u = if l > 0.0 { [x / l, y / l] } else { [1.0, 0.0] };
                ([c[0] + r * u[0], c[1] + r * u[1]], u)
            }
            Meridian::Profile { ref curve, z0, up } => {
                let t = curve.closest_param([q[0], (q[1] - z0) * up]);
                let (c, d, _) = curve.ders2(t);
                let l = d[0].hypot(d[1]).max(f64::MIN_POSITIVE);
                ([c[0], z0 + up * c[1]], [-d[1] * up / l, d[0] / l])
            }
        }
    }
}

/// The axis a carrier fixes (a point on it and its unit direction). A plane
/// and a sphere fix none: any axis along the normal, through the centre.
fn own_axis(k: &SurfaceKind) -> Option<(V3, V3)> {
    match k {
        SurfaceKind::Cylinder { center, axis, .. } | SurfaceKind::Torus { center, axis, .. } => {
            Some((*center, norm(*axis)))
        }
        SurfaceKind::Cone { apex, axis, .. } => Some((*apex, norm(*axis))),
        SurfaceKind::Revolved { origin, axis, .. } => Some((*origin, norm(*axis))),
        _ => None,
    }
}

/// The meridian of carrier `k` (with `plane`, the exact geometry of a plane
/// face) about the axis `(o, a)`, if it is a surface of revolution about it.
/// `size` scales the tolerance for a centre on the axis.
fn meridian(k: &SurfaceKind, plane: Option<(V3, V3)>, o: V3, a: V3, size: f64) -> Option<Meridian> {
    let on_axis = |c: V3| {
        let d = sub(c, o);
        let off = sub(d, scale(a, dot(d, a)));
        dot(off, off).sqrt() <= SQUARE_TOL * size
    };
    let along = |b: V3| dot(norm(b), a).abs() >= 1.0 - SQUARE_TOL;
    let z = |c: V3| dot(sub(c, o), a);
    match k {
        SurfaceKind::Plane => {
            let (po, pn) = plane?;
            along(pn).then_some(Meridian::Line {
                p: [0.0, z(po)],
                d: [1.0, 0.0],
            })
        }
        SurfaceKind::Sphere { center, radius } => on_axis(*center).then_some(Meridian::Circle {
            c: [0.0, z(*center)],
            r: *radius,
        }),
        SurfaceKind::Cylinder {
            center,
            axis,
            radius,
        } => (along(*axis) && on_axis(*center)).then_some(Meridian::Line {
            p: [*radius, 0.0],
            d: [0.0, 1.0],
        }),
        SurfaceKind::Cone {
            apex,
            axis,
            tan_half_angle,
        } => (along(*axis) && on_axis(*apex)).then(|| {
            // The generator leaves the apex into the nappe, whichever way
            // the axis runs.
            let up = dot(norm(*axis), a).signum();
            let l = tan_half_angle.hypot(1.0);
            Meridian::Line {
                p: [0.0, z(*apex)],
                d: [tan_half_angle / l, up / l],
            }
        }),
        SurfaceKind::Torus {
            center,
            axis,
            major_radius,
            minor_radius,
        } => (along(*axis) && on_axis(*center)).then_some(Meridian::Circle {
            c: [*major_radius, z(*center)],
            r: *minor_radius,
        }),
        SurfaceKind::Revolved {
            profile,
            origin,
            axis,
            ..
        } => (along(*axis) && on_axis(*origin)).then(|| Meridian::Profile {
            curve: profile.clone(),
            z0: z(*origin),
            up: dot(norm(*axis), a).signum(),
        }),
        _ => None,
    }
}

/// Where two meridians meet nearest `q`: Newton on their tangent lines where
/// they cross, the point of contact where they touch (within `TOUCH_TOL` of
/// `size`), none where they miss or coincide.
fn meridians_meet(m1: &Meridian, m2: &Meridian, mut q: [f64; 2], size: f64) -> Option<[f64; 2]> {
    for _ in 0..64 {
        let ((f1, n1), (f2, n2)) = (m1.foot(q), m2.foot(q));
        let det = n1[0] * n2[1] - n1[1] * n2[0];
        if det.abs() < 1e-9 {
            break;
        }
        let (b1, b2) = (n1[0] * f1[0] + n1[1] * f1[1], n2[0] * f2[0] + n2[1] * f2[1]);
        let next = [
            (b1 * n2[1] - b2 * n1[1]) / det,
            (n1[0] * b2 - n2[0] * b1) / det,
        ];
        let step = (next[0] - q[0]).hypot(next[1] - q[1]);
        q = next;
        if step <= 1e-15 * size {
            return Some(q);
        }
    }
    // Tangent meridians: the point of contact, from the closest approach.
    let d2 = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    match (m1, m2) {
        (Meridian::Line { .. }, &Meridian::Circle { c, r })
        | (&Meridian::Circle { c, r }, Meridian::Line { .. }) => {
            let line = if matches!(m1, Meridian::Line { .. }) {
                m1
            } else {
                m2
            };
            let f = line.foot(c).0;
            ((d2(f, c) - r).abs() <= TOUCH_TOL * size).then_some(f)
        }
        (&Meridian::Circle { c: c1, r: r1 }, &Meridian::Circle { c: c2, r: r2 }) => {
            let d = d2(c1, c2);
            (d > 0.0)
                .then(|| {
                    let u = [(c2[0] - c1[0]) / d, (c2[1] - c1[1]) / d];
                    [1.0, -1.0]
                        .map(|s| [c1[0] + s * r1 * u[0], c1[1] + s * r1 * u[1]])
                        .into_iter()
                        .min_by(|a, b| (d2(*a, c2) - r2).abs().total_cmp(&(d2(*b, c2) - r2).abs()))
                })
                .flatten()
                .filter(|p| (d2(*p, c2) - r2).abs() <= TOUCH_TOL * size)
        }
        _ => None,
    }
}

/// The circle two radial carriers of the chain meet in: two coaxial surfaces
/// of revolution (plane square to the axis, cylinder, cone, sphere centred
/// on it, torus) meet in circles square to the axis, at the radius and height
/// where their meridians cross or touch; the one nearest the chain. A plane
/// and a sphere take the axis the other carrier fixes, two spheres the line
/// of their centres. `(center, unit axis, radius, unit x toward the chain
/// start)`.
fn revolution_circle(
    chain: &[V3],
    rad: &[FaceId],
    faces: &[Face],
    plc: &TaggedPlc,
) -> Option<(V3, V3, f64, V3)> {
    let size = arc_len(chain).max(f64::MIN_POSITIVE);
    let kind = |f: &FaceId| &plc.surfaces[faces[f.0 as usize].plc_surface as usize];
    let plane = |f: &FaceId| {
        matches!(kind(f), SurfaceKind::Plane)
            .then(|| exact_face_plane(&faces[f.0 as usize], plc))
            .flatten()
    };
    for (i, f) in rad.iter().enumerate() {
        for g in &rad[i + 1..] {
            let (kf, kg) = (kind(f), kind(g));
            let axis = own_axis(kf)
                .or_else(|| own_axis(kg))
                .or_else(|| match (kf, kg) {
                    (SurfaceKind::Sphere { center, .. }, SurfaceKind::Plane)
                    | (SurfaceKind::Plane, SurfaceKind::Sphere { center, .. }) => {
                        let (_, n) = plane(f).or_else(|| plane(g))?;
                        Some((*center, n))
                    }
                    (
                        SurfaceKind::Sphere { center: c1, .. },
                        SurfaceKind::Sphere { center: c2, .. },
                    ) => {
                        let d = sub(*c2, *c1);
                        (dot(d, d) > 0.0).then(|| (*c1, norm(d)))
                    }
                    _ => None,
                });
            let Some((o, a)) = axis else { continue };
            let (Some(mf), Some(mg)) = (
                meridian(kf, plane(f), o, a, size),
                meridian(kg, plane(g), o, a, size),
            ) else {
                continue;
            };
            // Start from the chain's mean radius and height.
            let n = chain.len() as f64;
            let q0 = chain.iter().fold([0.0, 0.0], |acc, &p| {
                let d = sub(p, o);
                let h = dot(d, a);
                let r = sub(d, scale(a, h));
                [acc[0] + dot(r, r).sqrt() / n, acc[1] + h / n]
            });
            let Some([r, h]) = meridians_meet(&mf, &mg, q0, size) else {
                continue;
            };
            if !(r > 1e-12 * size) {
                continue;
            }
            let center = add(o, scale(a, h));
            let d = sub(chain[0], center);
            let x = norm(sub(d, scale(a, dot(d, a))));
            return Some((center, a, r, x));
        }
    }
    None
}

/// The exact ellipse of an oblique plane∩cylinder section. Plane `(po, pn)`,
/// cylinder `(center c, unit axis ca, radius r)`: the section is an ellipse
/// with center on the cylinder axis, semi-minor `r` along `ca x pn`,
/// semi-major `r/|ca·pn|` along the axis' in-plane projection. `None` when
/// near-perpendicular (a circle, handled elsewhere) or near-parallel (no
/// bounded section).
fn plane_cylinder_ellipse(
    _chain: &[V3],
    po: V3,
    pn: V3,
    c: V3,
    ca: V3,
    r: f64,
    _tol: f64,
) -> Option<Curve> {
    let cosphi = dot(ca, pn);
    // Square cut (a circle) or a plane along the axis (lines): not ours.
    if cosphi.abs() >= 1.0 - SQUARE_TOL || cosphi.abs() <= SQUARE_TOL {
        return None;
    }
    // Ellipse center: the cylinder axis pierced through the plane.
    let t = dot(sub(po, c), pn) / cosphi;
    let center = add(c, scale(ca, t));
    let minor_dir = norm(cross(ca, pn));
    let major_dir = norm(cross(pn, minor_dir));
    let (a, b) = (r / cosphi.abs(), r);
    Some(Curve::Ellipse {
        center,
        major: major_dir,
        minor: minor_dir,
        a,
        b,
    })
}
