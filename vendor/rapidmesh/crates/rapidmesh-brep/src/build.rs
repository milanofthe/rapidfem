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
    Brep, CoEdge, CoEdgeId, Curve, Edge, EdgeId, Face, FaceId, Loop, SurfaceId, Vertex, VertexId,
};
use rapidmesh_exact::vector::{
    add, bbox, cross, dist, dot, normalize as norm, scale, segment_dist2, sub, tri_normal, V3,
};
use rapidmesh_geom::{FaceTag, Surface, TaggedPlc};
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
        let (lo, hi) = bbox(pos);
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
    // Faces an edge runs across (two of their triangles on it): no boundary
    // of theirs, yet the edge lies on them.
    let mut across: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
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
            } else {
                across.entry(e).or_default().push(f);
            }
        }
    }
    for v in bedge_faces.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    // Feature edges inside a face (an import's open crease the face wraps
    // around), and edges where faces cross each other without either ending
    // (two sheets through each other): the boundary of no face, yet B-rep
    // edges, with the faces around them. They take part in no loop.
    let mut inner: HashSet<(usize, usize)> = HashSet::default();
    for (e, fs) in &across {
        if fs.len() >= 2 && !bedge_faces.contains_key(e) {
            let mut fs = fs.clone();
            fs.sort_unstable();
            fs.dedup();
            bedge_faces.insert(*e, fs);
            inner.insert(*e);
        }
    }
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
            vertices.push(Vertex {
                pos: pos[plc_v],
                faces: Vec::new(),
            });
            VertexId((vertices.len() - 1) as u32)
        })
    };

    // The curves the shapes declare, each with the box of its points.
    let declared: Vec<(&rapidmesh_geom::EdgeCurve, V3, V3)> = plc
        .curves
        .iter()
        .map(|c| {
            let (lo, hi) = bbox(&c.points);
            (c, lo, hi)
        })
        .collect();

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
        let curve = declared_curve(&chain_pts, &declared, tol)
            .unwrap_or_else(|| recover_curve(&chain_pts, &rad, &faces, plc, tol));
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
        // Where a plane's frame starts and points: an originating facet
        // (exact PLC vertices), so chart coordinates stay small. The normal
        // is the plane's own; other kinds ignore the frame.
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
        // A face whose facets are its carrier (a loft mantle, a swept wall)
        // is a DISCRETE patch of its own facets, the same closest-point
        // carrier an STL import gets; one that happens to be flat is
        // the plane of its facets. A plane whose facets leave it (which the
        // geometry should never produce) is carried by its facets too.
        let given = plc.surfaces[faces[fid].surface.0 as usize].as_ref();
        let flat = match given {
            None if faces[fid].facets.is_empty() => Some(Surface::plane_of(&frame_pts)),
            None | Some(Surface::Plane(_)) => Some(facets_plane(&faces[fid], given, plc, tol)),
            Some(_) => None,
        };
        let surface = match flat {
            None => given.expect("a curved carrier").clone(),
            Some(Some(plane)) => plane,
            Some(None) => {
                if given.is_some() {
                    rapidmesh_exact::log::debug(
                        "brep.plane",
                        format!("face {fid} leaves its plane, carried by its facets"),
                    );
                }
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
                Surface::Discrete(Arc::new(rapidmesh_geom::DiscreteSurface::new(
                    dpoints, dtris,
                )))
            }
        };
        let sid = SurfaceId(surfaces.len() as u32);
        // One surface per face, in face order: `Curve::Intersection` (built in
        // recover_curve, before this loop) references faces' surfaces by this
        // identity, so it must hold.
        debug_assert_eq!(sid.0 as usize, fid, "surface id must equal face id");
        surfaces.push(surface.fitted(&frame_pts));
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
    // Inner edges, and faces an edge runs across: one co-edge per face
    // around them, outside every loop.
    for (ei, ch) in chains.iter().enumerate() {
        let mut inside: Vec<FaceId> = if is_inner[ei] {
            edge_faces[ei].clone()
        } else {
            Vec::new()
        };
        for w in ch.windows(2) {
            if let Some(fs) = across.get(&key2(w[0], w[1])) {
                inside.extend(fs.iter().map(|&f| FaceId(f as u32)));
            }
        }
        inside.sort_unstable();
        inside.dedup();
        for f in inside {
            if edges[ei]
                .coedges
                .iter()
                .any(|c| coedges[c.0 as usize].face == f)
            {
                continue;
            }
            let cid = CoEdgeId(coedges.len() as u32);
            coedges.push(CoEdge {
                edge: EdgeId(ei as u32),
                face: f,
                forward: true,
            });
            edges[ei].coedges.push(cid);
        }
    }
    // Every face with a triangle at a corner.
    for t in 0..n_tri {
        for v in tri(t) {
            if let Some(&id) = vid.get(&v) {
                let f = FaceId(tri_face[t] as u32);
                let fs = &mut vertices[id.0 as usize].faces;
                if !fs.contains(&f) {
                    fs.push(f);
                }
            }
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
    let (lo, hi) = bbox(&plc.vertices);
    let grid = 1e-9
        * (0..3)
            .map(|k| hi[k] - lo[k])
            .fold(0.0, f64::max)
            .max(1e-300);
    let back: Vec<Option<rapidmesh_exact::vector::Affine>> =
        plc.owner_frames.iter().map(|f| f.inverse()).collect();
    let local = |owner: u32, p: V3| -> [i64; 3] {
        let q = match back.get(owner as usize) {
            Some(Some(m)) => m.point(p),
            _ => p,
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
    let vertices: Vec<Vertex> = vorder
        .iter()
        .map(|&v| {
            let mut vx = b.vertices[v].clone();
            for f in &mut vx.faces {
                *f = FaceId(fnew[f.0 as usize]);
            }
            vx.faces.sort_unstable_by_key(|f| f.0);
            vx
        })
        .collect();
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

/// The piece of the declared curve the chain lies on (every point within
/// `tol` of the curve's points' polyline); `None` where none holds it or the
/// chain does not run along it monotonically (a piece across the seam of a
/// closed B-spline).
fn declared_curve(
    chain: &[V3],
    declared: &[(&rapidmesh_geom::EdgeCurve, V3, V3)],
    tol: f64,
) -> Option<Curve> {
    if chain.len() < 2 {
        return None;
    }
    let on_polyline = |pts: &[V3], q: V3| {
        pts.windows(2)
            .any(|w| segment_dist2(q, w[0], w[1]) <= tol * tol)
    };
    let (c, _, _) = declared.iter().find(|(c, lo, hi)| {
        chain
            .iter()
            .all(|q| (0..3).all(|k| q[k] >= lo[k] - tol && q[k] <= hi[k] + tol))
            && chain.iter().all(|&q| on_polyline(&c.points, q))
    })?;
    piece(c.curve.clone(), chain, tol)
}

/// The piece of `curve` the chain runs along (see
/// [`rapidmesh_geom::Curve::piece`]).
fn piece(curve: rapidmesh_geom::Curve<3>, chain: &[V3], tol: f64) -> Option<Curve> {
    let t = curve.piece(chain, tol)?;
    Some(Curve::Piece { curve, t })
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
            if let Some(c) = piece(rapidmesh_geom::Curve::Line { p: p0, d: dir }, chain, tol) {
                return c;
            }
        }
    }

    // Circle: two coaxial surfaces of revolution meet in circles square to
    // their axis, from the carriers alone (see `revolution_circle`).
    if let Some((center, axis, radius, x)) = revolution_circle(chain, rad, faces, plc) {
        if let Some(c) = rapidmesh_geom::Curve::circle(center, axis, x, radius)
            .and_then(|circle| piece(circle, chain, tol))
        {
            return c;
        }
    }

    // Oblique plane section of a cylinder: an exact ELLIPSE (the axis-perpendicular
    // case is the circle above). Derived from the cylinder's parameters and the
    // EXACT facet plane of the adjacent planar face -- chain-independent, so the
    // recovered ellipse lies exactly on both carriers (a fit to the faceted chain
    // would sit a sagitta inside, the straddler-sliver mechanism).
    let planes: Vec<(V3, V3)> = rad
        .iter()
        .filter_map(
            |f| match &plc.surfaces[faces[f.0 as usize].plc_surface as usize] {
                Some(Surface::Plane(p)) => Some((p.o, p.z)),
                _ => None,
            },
        )
        .collect();
    for f in rad {
        let kind = &plc.surfaces[faces[f.0 as usize].plc_surface as usize];
        if let Some(Surface::Cylinder { frame, radius }) = kind {
            for &(po, pn) in &planes {
                if let Some(e) = plane_cylinder_ellipse(po, pn, frame.o, frame.z, *radius)
                    .and_then(|ellipse| piece(ellipse, chain, tol))
                {
                    return e;
                }
            }
        }
    }

    // On an extruded surface at constant height: the analytic profile curve.
    for f in rad {
        let sid = faces[f.0 as usize].surface;
        if let Some(Surface::Extruded { frame, profile }) = &plc.surfaces[sid.0 as usize] {
            let z0 = dot(sub(p0, frame.o), frame.z);
            // constant extrusion height along the whole chain -> a profile edge
            let const_h = chain
                .iter()
                .all(|&p| (dot(sub(p, frame.o), frame.z) - z0).abs() < tol.max(1e-7));
            let lifted = || rapidmesh_geom::Curve::Nurbs(profile.clone()).lifted(frame, z0);
            if let Some(c) = const_h.then(lifted).and_then(|c| piece(c, chain, tol)) {
                return c;
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
                    matches!(plc.surfaces[sid as usize], None | Some(Surface::Plane(_))),
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

/// The analytic carrier of a face; none for a discrete patch or a face
/// whose facets are not flat but carry it.
fn face_carrier(f: &Face, plc: &TaggedPlc, tol: f64) -> Option<Surface> {
    let kind = plc.surfaces[f.surface.0 as usize].as_ref();
    let t = plc.triangles[*f.facets.first()? as usize];
    let frame: Vec<V3> = t.iter().map(|&v| plc.vertices[v as usize]).collect();
    match kind {
        Some(Surface::Discrete(_)) => None,
        None | Some(Surface::Plane(_)) => Some(facets_plane(f, kind, plc, tol)?.fitted(&frame)),
        Some(k) => Some(k.fitted(&frame)),
    }
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
    // Per facet, once and in parallel: a projection is the costly part.
    let facet_err: Vec<f64> = {
        use rayon::prelude::*;
        (0..tri_face.len())
            .into_par_iter()
            .map(|t| {
                let f = tri_face[t];
                if carrier[f].is_none() {
                    return 0.0;
                }
                let q = tri_pts(t as u32);
                off(f, scale(add(add(q[0], q[1]), q[2]), 1.0 / 3.0))
            })
            .collect()
    };
    let err: Vec<f64> = (0..nf)
        .map(|f| {
            faces[f]
                .facets
                .iter()
                .map(|&t| facet_err[t as usize])
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
                let e = facet_err[u as usize];
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
        // A face a block cut bounds is a piece the cut made of a face, as
        // thin as it lies, not a strip between tangent tessellations: it
        // keeps its own carrier.
        if shared[f].keys().any(|&g| faces[g].face_tag == FaceTag::CUT) {
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
            // On g's carrier: a plane takes coplanar facets only, a curved
            // carrier those its faceting errors explain (the strip a
            // sagitta wide), not a face of another surface beside it.
            let plane = matches!(
                plc.surfaces[faces[g].surface.0 as usize],
                None | Some(Surface::Plane(_))
            );
            let reach = if plane { tol } else { tol.max(explained) };
            let worst = faces[f]
                .facets
                .iter()
                .flat_map(|&t| tri_pts(t))
                .map(|p| off(g, p))
                .fold(0.0f64, f64::max);
            if worst > reach {
                rapidmesh_exact::log::debug(
                    "brep.absorb",
                    format!("face {f} kept from face {g}: {worst:.3e} off its carrier, {reach:.3e} explained"),
                );
            }
            worst <= reach
        };
        let target = nbrs
            .iter()
            .map(|&(g, _)| g)
            .filter(|&g| fits(g))
            .max_by(|&a, &b| mismatch(a).total_cmp(&mismatch(b)).then(b.cmp(&a)));
        if target.is_none() {
            rapidmesh_exact::log::debug(
                "brep.absorb",
                format!(
                    "face {f} (width {:.3e}, faceting errors {:.3e}) taken by no neighbour of {:?}",
                    width(f),
                    explained,
                    nbrs.iter()
                        .map(|&(g, _)| (
                            g,
                            pair(&faces[g]) == pair(&faces[f]),
                            carrier[g].is_some()
                        ))
                        .collect::<Vec<_>>()
                ),
            );
        }
        if let Some(g) = target {
            rapidmesh_exact::log::debug(
                "brep.absorb",
                format!(
                    "face {f} ({} facets, width {:.3e}, faceting errors {:.3e}) into face {g}",
                    faces[f].facets.len(),
                    width(f),
                    explained
                ),
            );
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

/// The plane `(point, unit normal)` all facets of a face lie on (within
/// `tol`): a plane's own, or for faceted kind the plane of its first proper
/// facet. None where they leave it.
fn facets_plane(face: &Face, kind: Option<&Surface>, plc: &TaggedPlc, tol: f64) -> Option<Surface> {
    let corners = |tfi: &u32| plc.triangles[*tfi as usize].map(|v| plc.vertices[v as usize]);
    let plane = match kind {
        Some(p @ Surface::Plane(_)) => p.clone(),
        _ => face.facets.iter().map(corners).find_map(|[a, b, c]| {
            let n = tri_normal(a, b, c);
            (dot(n, n) >= 1e-24).then(|| Surface::plane(a, n)).flatten()
        })?,
    };
    let f = *plane.frame()?;
    face.facets
        .iter()
        .flat_map(corners)
        .all(|p| dot(sub(p, f.o), f.z).abs() <= tol)
        .then_some(plane)
}

/// How close to square (or to parallel) a plane must meet an axis for the
/// cut to count as a circle (or as lines): a cosine within this of 1 (of 0);
/// likewise how close to an axis a centre must lie, relative to the edge.
/// The carriers agree to rounding when they are meant to.
const SQUARE_TOL: f64 = 1e-9;

/// How far apart two meridians may pass and still touch: a sphere on a
/// cylinder of its radius, faceted apart by rounding of their parameters.
const TOUCH_TOL: f64 = 1e-6;

/// The point of the meridian `m` nearest `q` and the unit normal there.
fn foot(m: &rapidmesh_geom::Curve<2>, q: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    let (c, d, _) = m.ders(m.param(q));
    let l = d[0].hypot(d[1]).max(f64::MIN_POSITIVE);
    (c, [-d[1] / l, d[0] / l])
}

/// Where two meridians meet nearest `q`: Newton on their tangent lines where
/// they cross, the point of contact where they touch (within `TOUCH_TOL` of
/// `size`), none where they miss or coincide.
fn meridians_meet(
    m1: &rapidmesh_geom::Curve<2>,
    m2: &rapidmesh_geom::Curve<2>,
    mut q: [f64; 2],
    size: f64,
) -> Option<[f64; 2]> {
    for _ in 0..64 {
        let ((f1, n1), (f2, n2)) = (foot(m1, q), foot(m2, q));
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
    let line = |m: &rapidmesh_geom::Curve<2>| matches!(m, rapidmesh_geom::Curve::Line { .. });
    let touch = |p: [f64; 2], c: [f64; 2], r: f64| (d2(p, c) - r).abs() <= TOUCH_TOL * size;
    match (m1.as_circle(), m2.as_circle()) {
        (Some((c, r)), None) if line(m2) => Some(foot(m2, c).0).filter(|&f| touch(f, c, r)),
        (None, Some((c, r))) if line(m1) => Some(foot(m1, c).0).filter(|&f| touch(f, c, r)),
        (Some((c1, r1)), Some((c2, r2))) => {
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
                .filter(|&p| touch(p, c2, r2))
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
    let kind = |f: &FaceId| plc.surfaces[faces[f.0 as usize].plc_surface as usize].as_ref();
    for (i, f) in rad.iter().enumerate() {
        for g in &rad[i + 1..] {
            let (Some(kf), Some(kg)) = (kind(f), kind(g)) else {
                continue;
            };
            let axis = kf.axis().or_else(|| kg.axis()).or_else(|| match (kf, kg) {
                (Surface::Sphere { frame: s, .. }, Surface::Plane(p))
                | (Surface::Plane(p), Surface::Sphere { frame: s, .. }) => Some((s.o, p.z)),
                (Surface::Sphere { frame: s1, .. }, Surface::Sphere { frame: s2, .. }) => {
                    let d = sub(s2.o, s1.o);
                    (dot(d, d) > 0.0).then(|| (s1.o, norm(d)))
                }
                _ => None,
            });
            let Some((o, a)) = axis else { continue };
            let (Some(mf), Some(mg)) = (
                kf.meridian(o, a, SQUARE_TOL * size),
                kg.meridian(o, a, SQUARE_TOL * size),
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

/// The exact ellipse of an oblique plane-cylinder section. Plane `(po, pn)`,
/// cylinder `(center c, unit axis ca, radius r)`: the section is an ellipse
/// with center on the cylinder axis, semi-minor `r` along `ca x pn`,
/// semi-major `r/|ca*pn|` along the axis' in-plane projection. `None` when
/// near-perpendicular (a circle, handled elsewhere) or near-parallel (no
/// bounded section).
fn plane_cylinder_ellipse(
    po: V3,
    pn: V3,
    c: V3,
    ca: V3,
    r: f64,
) -> Option<rapidmesh_geom::Curve<3>> {
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
    Some(rapidmesh_geom::Curve::Ellipse {
        c: center,
        p: scale(major_dir, r / cosphi.abs()),
        q: scale(minor_dir, r),
    })
}
