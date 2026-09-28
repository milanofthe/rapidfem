//! Surface mesh import: STL (binary and ASCII) and Wavefront OBJ.
//!
//! Imported facets are grouped into SMOOTH REGIONS at crease edges (facet
//! normals turning by more than the crease threshold): each region becomes one
//! [`SurfaceKind::Discrete`] carrier, so the mesher REMESHES the import against
//! its own envelope -- creases survive as B-rep feature edges, smooth areas are
//! free to resample. (One plane per facet, the old behaviour, froze the whole
//! import: every input edge a protected feature, every input sliver permanent.)
//! Exactly degenerate (collinear) facets are dropped on import; duplicated
//! facets are rejected. [`validate_closed`] checks the watertight,
//! consistently-oriented 2-manifold invariant that [`crate::Scene`] solids
//! require.

use crate::faceted::{Faceted, SurfaceKind};
use rapidmesh_csg::Tri;
use rapidmesh_exact::collinear;
use std::collections::HashMap;
use std::io::Read as _;
use std::path::Path;

/// Import failure.
#[derive(Debug)]
pub enum ImportError {
    /// I/O failure.
    Io(std::io::Error),
    /// Malformed file content (message describes the location).
    Parse(String),
    /// Structural defect found by [`validate_closed`].
    NotClosed(String),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::Io(e) => write!(f, "io error: {e}"),
            ImportError::Parse(m) => write!(f, "parse error: {m}"),
            ImportError::NotClosed(m) => write!(f, "surface not closed: {m}"),
        }
    }
}

impl std::error::Error for ImportError {}

impl From<std::io::Error> for ImportError {
    fn from(e: std::io::Error) -> ImportError {
        ImportError::Io(e)
    }
}

/// Default crease threshold (degrees): facet normals turning by more than this
/// across a shared edge mark a feature edge; smoother turns stay inside one
/// discrete region.
pub const CREASE_DEG: f64 = 40.0;

/// Builds a [`Faceted`] from raw triangles: drops exactly degenerate facets,
/// groups the rest into smooth regions at crease edges (`crease_deg`), and
/// gives every region ONE carrier: a plane where its facets lie in one (to
/// the tolerance the B-rep checks planes with), else a
/// [`SurfaceKind::Discrete`] patch.
pub(crate) fn faceted_from_tris_creased(tris: Vec<Tri>, crease_deg: f64) -> Faceted {
    let tris: Vec<Tri> = tris
        .into_iter()
        .filter(|t| collinear(&t.point(0), &t.point(1), &t.point(2)) != Some(true))
        .collect();
    let n = tris.len();

    // Shared vertex indexing (exact bit match -- STL repeats vertices per facet).
    let mut vid: HashMap<[u64; 3], u32> = HashMap::new();
    let mut points: Vec<[f64; 3]> = Vec::new();
    let key = |p: [f64; 3]| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
    let mut conn: Vec<[u32; 3]> = Vec::with_capacity(n);
    for t in &tris {
        let idx: [u32; 3] = std::array::from_fn(|k| {
            let p = t.v[k];
            *vid.entry(key(p)).or_insert_with(|| {
                points.push(p);
                (points.len() - 1) as u32
            })
        });
        conn.push(idx);
    }

    // Facet normals + edge adjacency.
    let normal = |c: &[u32; 3]| -> [f64; 3] {
        let (a, b, cc) = (
            points[c[0] as usize],
            points[c[1] as usize],
            points[c[2] as usize],
        );
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let w = [cc[0] - a[0], cc[1] - a[1], cc[2] - a[2]];
        let mut nr = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        let l = (nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]).sqrt();
        if l > 0.0 {
            for x in &mut nr {
                *x /= l;
            }
        }
        nr
    };
    let normals: Vec<[f64; 3]> = conn.iter().map(normal).collect();
    let mut by_edge: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (i, c) in conn.iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (c[e], c[(e + 1) % 3]);
            by_edge
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(i as u32);
        }
    }

    // Union-find over facets: smooth (sub-crease) shared edges merge regions.
    let cos_crease = crease_deg.to_radians().cos();
    let mut rep: Vec<u32> = (0..n as u32).collect();
    fn find(rep: &mut [u32], mut x: u32) -> u32 {
        while rep[x as usize] != x {
            rep[x as usize] = rep[rep[x as usize] as usize];
            x = rep[x as usize];
        }
        x
    }
    let mut crease: Vec<((u32, u32), [u32; 2])> = Vec::new();
    for (&(a, b), owners) in &by_edge {
        if owners.len() != 2 {
            continue; // boundary / non-manifold: always a region boundary
        }
        let (i, j) = (owners[0] as usize, owners[1] as usize);
        let d = normals[i][0] * normals[j][0]
            + normals[i][1] * normals[j][1]
            + normals[i][2] * normals[j][2];
        if d >= cos_crease {
            let (ri, rj) = (find(&mut rep, i as u32), find(&mut rep, j as u32));
            if ri != rj {
                rep[ri.max(rj) as usize] = ri.min(rj);
            }
        } else {
            crease.push(((a, b), [owners[0], owners[1]]));
        }
    }

    // NOISE-ROBUST crease filtering: real features form long chains or
    // closed loops (a CAD model's whole crease network is typically ONE
    // connected component -- fandisk: 710 edges, 1 component), while scan
    // noise shatters into short OPEN fragments (cow: 94 of 137 components
    // have <= 3 edges; armadillo: 500 of 573). A raw dihedral threshold
    // turns every fragment into a feature curve with protecting balls --
    // measured at 4.8 HOURS on the 5.8k-facet cow. Open components shorter
    // than `NOISE_CHAIN_MIN` edges are therefore treated as smooth; closed
    // loops of any size stay (a tiny loop can be a genuine small feature).
    const NOISE_CHAIN_MIN: usize = 8;
    let mut noise: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    // Connected components of crease edges (through shared vertices).
    let mut erep: Vec<u32> = (0..crease.len() as u32).collect();
    if !crease.is_empty() {
        crease.sort_unstable(); // deterministic component ids
                                // Connected components of crease edges via shared vertices, plus
                                // per-vertex degree (open component <=> some vertex has degree 1).
        let mut vfirst: HashMap<u32, u32> = HashMap::new();
        let mut vdeg: HashMap<u32, u32> = HashMap::new();
        for (ei, &((a, b), _)) in crease.iter().enumerate() {
            for v in [a, b] {
                *vdeg.entry(v).or_insert(0) += 1;
                match vfirst.entry(v) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(ei as u32);
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        let (ri, rj) = (find(&mut erep, ei as u32), find(&mut erep, *e.get()));
                        if ri != rj {
                            erep[ri.max(rj) as usize] = ri.min(rj);
                        }
                    }
                }
            }
        }
        let mut comp_size: HashMap<u32, usize> = HashMap::new();
        let mut comp_open: HashMap<u32, bool> = HashMap::new();
        for ei in 0..crease.len() as u32 {
            let r = find(&mut erep, ei);
            *comp_size.entry(r).or_insert(0) += 1;
            let ((a, b), _) = crease[ei as usize];
            if vdeg[&a] == 1 || vdeg[&b] == 1 {
                comp_open.insert(r, true);
            }
        }
        for ei in 0..crease.len() as u32 {
            let r = find(&mut erep, ei);
            if comp_size[&r] < NOISE_CHAIN_MIN && comp_open.get(&r).copied().unwrap_or(false) {
                noise.insert(crease[ei as usize].0);
                let [i, j] = crease[ei as usize].1;
                let (ri, rj) = (find(&mut rep, i), find(&mut rep, j));
                if ri != rj {
                    rep[ri.max(rj) as usize] = ri.min(rj);
                }
            }
        }
    }

    // One Discrete carrier per region (locally reindexed patch).
    let mut region_facets: HashMap<u32, Vec<u32>> = HashMap::new();
    for i in 0..n as u32 {
        let r = find(&mut rep, i);
        region_facets.entry(r).or_default().push(i);
    }
    let mut regions: Vec<Vec<u32>> = region_facets.into_values().collect();
    regions.sort_by_key(|m| m[0]); // deterministic surface order

    let mut f = Faceted::new();
    // Creases with one region on both sides go along as explicit features
    // when their crease component also bounds regions: the feature network
    // runs on past an open end the region wraps around (fandisk). A component
    // wholly inside one smooth region is a floating fragment, scan noise.
    let mut bounds: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for (ei, &(_, [i, j])) in crease.iter().enumerate() {
        if find(&mut rep, i) != find(&mut rep, j) {
            bounds.insert(find(&mut erep, ei as u32));
        }
    }
    for (ei, &((a, b), [i, j])) in crease.iter().enumerate() {
        if find(&mut rep, i) == find(&mut rep, j)
            && !noise.contains(&(a, b))
            && bounds.contains(&find(&mut erep, ei as u32))
        {
            f.features.push([points[a as usize], points[b as usize]]);
        }
    }
    // The tolerance of a plane: as the B-rep's, relative to the extent.
    let (lo, hi) = points
        .iter()
        .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), p| {
            (
                std::array::from_fn(|k| lo[k].min(p[k])),
                std::array::from_fn(|k| hi[k].max(p[k])),
            )
        });
    let diag = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f64>().sqrt();
    let flat_tol = 1e-9 * diag.max(1.0);
    for members in regions {
        // Flat: every vertex on the plane of the region's first facet.
        let first = conn[members[0] as usize];
        let o = points[first[0] as usize];
        let n = normals[members[0] as usize];
        let flat = members.iter().all(|&fi| {
            conn[fi as usize].iter().all(|&v| {
                let p = points[v as usize];
                ((p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1] + (p[2] - o[2]) * n[2]).abs()
                    <= flat_tol
            })
        });
        if flat {
            let s = f.add_surface(SurfaceKind::Plane);
            for &fi in &members {
                f.push_tri(tris[fi as usize], s);
            }
            continue;
        }
        let mut l_vid: HashMap<u32, u32> = HashMap::new();
        let mut l_pts: Vec<[f64; 3]> = Vec::new();
        let mut l_tris: Vec<[u32; 3]> = Vec::new();
        for &fi in &members {
            let c = conn[fi as usize];
            let lt: [u32; 3] = std::array::from_fn(|k| {
                *l_vid.entry(c[k]).or_insert_with(|| {
                    l_pts.push(points[c[k] as usize]);
                    (l_pts.len() - 1) as u32
                })
            });
            l_tris.push(lt);
        }
        let s = f.add_surface(SurfaceKind::Discrete(std::sync::Arc::new(
            crate::discrete::DiscreteSurface::new(l_pts, l_tris),
        )));
        for &fi in &members {
            f.push_tri(tris[fi as usize], s);
        }
    }
    f
}

// ----------------------------------------------------------------- STL

/// Reads an STL file (binary or ASCII, auto-detected) into a [`Faceted`].
/// Facet normals in the file are ignored; orientation comes from the vertex
/// winding (the STL convention requires both to agree).
pub fn import_stl(path: &Path) -> Result<Faceted, ImportError> {
    import_stl_creased(path, CREASE_DEG)
}

/// [`import_stl`] with an explicit crease threshold (degrees) for the
/// feature-edge detection that splits the soup into smooth Discrete regions.
pub fn import_stl_creased(path: &Path, crease_deg: f64) -> Result<Faceted, ImportError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    let tris = if stl_is_binary(&bytes) {
        parse_stl_binary(&bytes)?
    } else {
        parse_stl_ascii(&bytes)?
    };
    Ok(faceted_from_tris_creased(tris, crease_deg))
}

/// Binary detection: the 80-byte header is free-form (may even start with
/// "solid"), so the reliable test is the binary length invariant
/// `84 + 50 * n_triangles`.
fn stl_is_binary(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    bytes.len() == 84 + 50 * n
}

fn parse_stl_binary(bytes: &[u8]) -> Result<Vec<Tri>, ImportError> {
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let mut tris = Vec::with_capacity(n);
    let f32_at = |off: usize| -> f64 {
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]) as f64
    };
    for i in 0..n {
        let base = 84 + 50 * i;
        // 12 bytes facet normal (skipped), then 3 vertices of 12 bytes.
        let v: [[f64; 3]; 3] =
            std::array::from_fn(|j| std::array::from_fn(|k| f32_at(base + 12 + 12 * j + 4 * k)));
        if v.iter().flatten().any(|x| !x.is_finite()) {
            return Err(ImportError::Parse(format!(
                "non-finite vertex in facet {i}"
            )));
        }
        tris.push(Tri::new(v[0], v[1], v[2]));
    }
    Ok(tris)
}

fn parse_stl_ascii(bytes: &[u8]) -> Result<Vec<Tri>, ImportError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| ImportError::Parse("ascii stl is not valid utf-8".to_string()))?;
    let mut tris = Vec::new();
    let mut verts: Vec<[f64; 3]> = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("vertex") => {
                let mut v = [0.0f64; 3];
                for x in &mut v {
                    *x = it
                        .next()
                        .and_then(|s| s.parse::<f64>().ok())
                        .filter(|x| x.is_finite())
                        .ok_or_else(|| {
                            ImportError::Parse(format!("bad vertex on line {}", ln + 1))
                        })?;
                }
                verts.push(v);
            }
            Some("endloop") => {
                if verts.len() != 3 {
                    return Err(ImportError::Parse(format!(
                        "facet with {} vertices on line {}",
                        verts.len(),
                        ln + 1
                    )));
                }
                tris.push(Tri::new(verts[0], verts[1], verts[2]));
                verts.clear();
            }
            _ => {}
        }
    }
    if tris.is_empty() {
        return Err(ImportError::Parse("no facets found".to_string()));
    }
    Ok(tris)
}

// ----------------------------------------------------------------- OBJ

/// Reads a Wavefront OBJ file into a [`Faceted`]. Only `v` and `f` records
/// are interpreted; faces with more than three corners are fan-triangulated;
/// `f` indices may be 1-based or negative (relative), with optional
/// `/texture/normal` suffixes.
pub fn import_obj(path: &Path) -> Result<Faceted, ImportError> {
    import_obj_creased(path, CREASE_DEG)
}

/// [`import_obj`] with an explicit crease threshold (degrees), see
/// [`import_stl_creased`].
pub fn import_obj_creased(path: &Path, crease_deg: f64) -> Result<Faceted, ImportError> {
    let text = std::fs::read_to_string(path)?;
    let mut verts: Vec<[f64; 3]> = Vec::new();
    let mut tris: Vec<Tri> = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let mut v = [0.0f64; 3];
                for x in &mut v {
                    *x = it
                        .next()
                        .and_then(|s| s.parse::<f64>().ok())
                        .filter(|x| x.is_finite())
                        .ok_or_else(|| {
                            ImportError::Parse(format!("bad vertex on line {}", ln + 1))
                        })?;
                }
                verts.push(v);
            }
            Some("f") => {
                let mut idx: Vec<usize> = Vec::new();
                for tok in it {
                    let first = tok.split('/').next().unwrap_or("");
                    let i: i64 = first.parse().map_err(|_| {
                        ImportError::Parse(format!("bad face index on line {}", ln + 1))
                    })?;
                    let resolved = if i > 0 {
                        i as usize - 1
                    } else if i < 0 {
                        let r = verts.len() as i64 + i;
                        if r < 0 {
                            return Err(ImportError::Parse(format!(
                                "face index out of range on line {}",
                                ln + 1
                            )));
                        }
                        r as usize
                    } else {
                        return Err(ImportError::Parse(format!(
                            "face index 0 on line {}",
                            ln + 1
                        )));
                    };
                    if resolved >= verts.len() {
                        return Err(ImportError::Parse(format!(
                            "face index out of range on line {}",
                            ln + 1
                        )));
                    }
                    idx.push(resolved);
                }
                if idx.len() < 3 {
                    return Err(ImportError::Parse(format!(
                        "face with {} corners on line {}",
                        idx.len(),
                        ln + 1
                    )));
                }
                for j in 1..idx.len() - 1 {
                    tris.push(Tri::new(verts[idx[0]], verts[idx[j]], verts[idx[j + 1]]));
                }
            }
            _ => {}
        }
    }
    if tris.is_empty() {
        return Err(ImportError::Parse("no faces found".to_string()));
    }
    Ok(faceted_from_tris_creased(tris, crease_deg))
}

// ---------------------------------------------------------- validation

/// Smallest triangle height (altitude) of the shape relative to its
/// bounding-box diagonal. Near-degenerate facets put neighboring facet
/// planes within float rounding of each other, which exact conforming
/// meshing cannot tile; inputs below roughly `1e-3` need mesh repair and
/// should be screened before meshing.
pub fn min_height_ratio(f: &Faceted) -> f64 {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    let mut min_h = f64::MAX;
    for t in &f.tris {
        for v in &t.v {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        let [a, b, c] = t.v;
        let u: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
        let w: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
        let n = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        let area2 = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let e = |p: [f64; 3], q: [f64; 3]| -> f64 {
            (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>().sqrt()
        };
        let lmax = e(a, b).max(e(b, c)).max(e(c, a));
        if lmax > 0.0 {
            min_h = min_h.min(area2 / lmax);
        }
    }
    let diag = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f64>().sqrt();
    if diag > 0.0 {
        min_h / diag
    } else {
        0.0
    }
}

/// Checks the closed-solid invariant [`crate::Scene::add_solid`] requires:
/// after welding bit-identical vertices, every undirected edge must be shared
/// by exactly two facets with opposite directions (watertight, consistently
/// oriented 2-manifold), and no facet may appear twice.
pub fn validate_closed(f: &Faceted) -> Result<(), ImportError> {
    let mut vid: HashMap<[u64; 3], u32> = HashMap::new();
    let mut key = |p: [f64; 3]| -> u32 {
        let bits: [u64; 3] = std::array::from_fn(|k| {
            // Weld +0.0 and -0.0; all other coordinates by exact bits.
            let x = if p[k] == 0.0 { 0.0 } else { p[k] };
            x.to_bits()
        });
        let next = vid.len() as u32;
        *vid.entry(bits).or_insert(next)
    };
    // Per undirected edge: net winding count (+1 forward, -1 backward) and
    // total incidence count.
    let mut edges: HashMap<(u32, u32), (i64, u64)> = HashMap::new();
    let mut seen_facets: HashMap<[u32; 3], usize> = HashMap::new();
    for (fi, t) in f.tris.iter().enumerate() {
        let v: [u32; 3] = std::array::from_fn(|i| key(t.v[i]));
        if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
            return Err(ImportError::NotClosed(format!(
                "facet {fi} has repeated vertices after welding"
            )));
        }
        let mut sorted = v;
        sorted.sort_unstable();
        if let Some(&prev) = seen_facets.get(&sorted) {
            return Err(ImportError::NotClosed(format!(
                "facet {fi} duplicates facet {prev}"
            )));
        }
        seen_facets.insert(sorted, fi);
        for e in 0..3 {
            let (a, b) = (v[e], v[(e + 1) % 3]);
            let entry = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
            entry.0 += if a < b { 1 } else { -1 };
            entry.1 += 1;
        }
    }
    for (&(a, b), &(net, count)) in &edges {
        if count != 2 || net != 0 {
            return Err(ImportError::NotClosed(format!(
                "edge ({a}, {b}) has {count} incident facets (net winding {net}), expected 2 with opposite orientation"
            )));
        }
    }
    Ok(())
}
