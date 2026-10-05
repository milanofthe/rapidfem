//! Primitive shapes: solids (closed, outward-oriented) and sheets (open
//! surfaces, e.g. PEC traces and ports, embeddable in volumes).
//!
//! Every builder returns a [`Faceted`] with per-triangle analytic surface
//! back-references. Solid builders guarantee watertightness and outward
//! orientation; sheet builders guarantee consistent winding.

use crate::curve::Curve;
use crate::faceted::{EdgeCurve, Faceted};
use crate::nurbs::NurbsCurve;
use crate::nurbs_surface::NurbsSurface;
use crate::polygon::{polygon_orientation, triangulate_polygon};
use crate::surface::Surface;
use rapidmesh_csg::{PlanarFacet, Tri};
use rapidmesh_exact::vector::Frame;
use rapidmesh_exact::vector::{add, cross, dot, len, ortho_unit, scale, unit};
use rapidmesh_exact::Sign;
use std::sync::Arc;

/// Two unit vectors square to `axis` and to each other, the first its
/// [`ortho_unit`].
fn orthonormal_basis(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let a = unit(axis).expect("axis must be nonzero");
    let e1 = ortho_unit(a);
    (e1, cross(a, e1))
}

/// Embeds a local 2D point into the (base, u, v) frame.
fn embed(base: [f64; 3], u: [f64; 3], v: [f64; 3], p: [f64; 2]) -> [f64; 3] {
    [
        base[0] + p[0] * u[0] + p[1] * v[0],
        base[1] + p[0] * u[1] + p[1] * v[1],
        base[2] + p[0] * u[2] + p[1] * v[2],
    ]
}

// ------------------------------------------------------------------ solids

/// Axis-aligned box. Six `Plane` face groups in the order
/// -z, +z, -y, +y, -x, +x.
pub fn solid_box(min: [f64; 3], max: [f64; 3]) -> Faceted {
    assert!(
        (0..3).all(|k| min[k] < max[k]),
        "box must have positive extent"
    );
    // Corner index bits: bit0 = x, bit1 = y, bit2 = z.
    let c: [[f64; 3]; 8] = std::array::from_fn(|i| {
        [
            if i & 1 == 0 { min[0] } else { max[0] },
            if i & 2 == 0 { min[1] } else { max[1] },
            if i & 4 == 0 { min[2] } else { max[2] },
        ]
    });
    let quads: [[usize; 4]; 6] = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    let normals: [[f64; 3]; 6] = [
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ];
    let mut f = Faceted::new();
    for (q, normal) in quads.into_iter().zip(normals) {
        let s = f.add_surface(Surface::plane(c[q[0]], normal));
        let loop4 = vec![c[q[0]], c[q[1]], c[q[2]], c[q[3]]];
        let tris = [
            Tri::new(c[q[0]], c[q[1]], c[q[2]]),
            Tri::new(c[q[0]], c[q[2]], c[q[3]]),
        ];
        f.push_flat(PlanarFacet::new(loop4), &tris, s);
    }
    f
}

/// Circular cylinder from `base_center` along the full height vector `axis`,
/// its barrel one row of full-height quads (see [`frustum`]).
pub fn cylinder(base_center: [f64; 3], axis: [f64; 3], radius: f64, segments: usize) -> Faceted {
    frustum(base_center, axis, radius, radius, segments, 1)
}

/// The azimuthal facet count a curved primitive of `radius` needs so its faceted
/// tessellation tracks the analytic surface at a mesh of target size `maxh`: fine
/// enough that the chord stays under the target (1.5x oversampled, like the edge
/// sampling) AND the sagitta stays under the surface tolerance `tol`. Curved
/// primitives are partly meshed FROM these facets (closed faces, the arrangement
/// of intersection curves), so a fixed count too coarse for the requested `maxh`
/// leaves the boundary off the true surface -- the straddler root cause. The CORE
/// derives this so every embedder (not just the Python layer) gets it.
pub(crate) fn facet_count(radius: f64, maxh: Option<f64>, tol: f64) -> usize {
    use std::f64::consts::PI;
    let n_tol = (PI / (1.0 - tol.clamp(1e-9, 0.5)).acos()).ceil() as usize;
    let n_maxh = match maxh {
        Some(h) if h.is_finite() && h > 0.0 => (1.5 * 2.0 * PI * radius / h).ceil() as usize,
        _ => 0,
    };
    n_tol.max(n_maxh).max(12)
}

/// The geodesic (icosphere) subdivision level whose edge length matches the
/// `facet_count` chord for a sphere of `radius` at target `maxh` (with an
/// explicit azimuthal floor of `min_segments`). A geodesic tessellation is
/// isotropic and POLE-FREE -- unlike a UV sphere, whose latitude rings cluster
/// at the poles and seed slivers there -- so it is the right facet distribution
/// for a curved primitive that gets refined to track `maxh`.
pub fn facet_subdivisions(radius: f64, maxh: Option<f64>, tol: f64, min_segments: usize) -> usize {
    use std::f64::consts::TAU;
    // icosphere level-0 edge ~= 1.0515 R; each level halves it. Match it to the
    // facet_count azimuthal chord (2*pi*R / n): level = log2(1.0515 n / (2 pi)).
    let n = facet_count(radius, maxh, tol).max(min_segments) as f64;
    ((1.0515 * n / TAU).log2().ceil() as i64).clamp(1, 6) as usize
}

/// Prism: a planar polygon (with holes) in the (base, u, v) frame, extruded
/// along `h` (out of the frame's plane): the [`sheet_polygon`] swept by
/// [`extrude_sheet`]. Surface groups: bottom, top, a plane per wall.
pub fn extrude_polygon(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    base: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
    h: [f64; 3],
) -> Faceted {
    extrude_sheet(&sheet_polygon(outer, holes, base, u, v), h)
        .expect("a polygon extrudes out of its plane")
}

/// Points of the NACA 0012 airfoil section (chord along +x, span `chord`),
/// from the upper trailing edge round the leading edge to the lower one,
/// `n_per_side + 1` a side, closer where it bends (cosine spacing in x):
/// a spline through them is the section. The standard 4-digit thickness
/// law with the `-0.1015` term leaves a small trailing edge thickness, so
/// the section is open, closed by a straight blunt trailing edge.
pub fn naca0012_points(chord: f64, n_per_side: usize) -> Vec<[f64; 2]> {
    assert!(n_per_side >= 4);
    let yt = |x: f64| {
        0.6 * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x
            - 0.1015 * x * x * x * x)
    };
    // INTERPOLATE the analytic NACA points (not use them as control points), so
    // the curve carries the TRUE NACA curvature: high only at the round leading
    // edge (sqrt(x) tangent), low along the gently sloping body -> a curvature
    // sizing field refines just the genuine features, not the whole thin shape.
    // Cosine x-spacing samples densely where it bends (LE/TE), the open profile
    // running upper TE -> LE -> lower TE.
    let mut pts: Vec<[f64; 2]> = Vec::new();
    for i in 0..=n_per_side {
        let x = 0.5 * (1.0 + (std::f64::consts::PI * i as f64 / n_per_side as f64).cos()); // 1..0
        pts.push([x * chord, yt(x) * chord]);
    }
    for i in 1..=n_per_side {
        let x = 0.5 * (1.0 - (std::f64::consts::PI * i as f64 / n_per_side as f64).cos()); // 0..1
        pts.push([x * chord, -yt(x) * chord]);
    }
    pts
}

/// UV torus around `center` with the major circle normal to `axis`.
/// `segments_major` divides the major circle, `segments_minor` the tube.
pub fn torus(
    center: [f64; 3],
    axis: [f64; 3],
    major_radius: f64,
    minor_radius: f64,
    segments_major: usize,
    segments_minor: usize,
) -> Faceted {
    assert!(
        segments_major >= 3 && segments_minor >= 3,
        "torus needs >= 3 segments per ring"
    );
    assert!(
        major_radius > minor_radius && minor_radius > 0.0,
        "torus needs 0 < minor_radius < major_radius"
    );
    let (e1, e2) = orthonormal_basis(axis);
    let a_hat = scale(axis, 1.0 / len(axis));
    let mut f = Faceted::new();
    let s = f.add_surface(Surface::torus(center, axis, major_radius, minor_radius));
    let pt = |i: usize, j: usize| -> [f64; 3] {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / segments_major as f64;
        let phi = 2.0 * std::f64::consts::PI * j as f64 / segments_minor as f64;
        let radial = add(scale(e1, theta.cos()), scale(e2, theta.sin()));
        add(
            center,
            add(
                scale(radial, major_radius + minor_radius * phi.cos()),
                scale(a_hat, minor_radius * phi.sin()),
            ),
        )
    };
    for i in 0..segments_major {
        let i2 = (i + 1) % segments_major;
        for j in 0..segments_minor {
            let j2 = (j + 1) % segments_minor;
            let (a, b, c, d) = (pt(i, j), pt(i2, j), pt(i2, j2), pt(i, j2));
            f.push_tri(Tri::new(a, b, c), s);
            f.push_tri(Tri::new(a, c, d), s);
        }
    }
    f
}

/// Wedge: a box `dx x dy x dz` at `position` whose top edge is shortened to
/// `top_x` along x (0 gives a triangular prism). The trapezoid profile lives
/// in the xz-plane and extrudes along +y.
pub fn wedge(position: [f64; 3], dx: f64, dy: f64, dz: f64, top_x: f64) -> Faceted {
    assert!(
        dx > 0.0 && dy > 0.0 && dz > 0.0,
        "wedge must have positive extent"
    );
    assert!((0.0..=dx).contains(&top_x), "top_x must lie in [0, dx]");
    // Profile points as (z, x) pairs in the (u = z-hat, v = x-hat) frame:
    // (u x v) . h = (z x x) . y = +1, right-handed.
    let profile: Vec<[f64; 2]> = if top_x > 0.0 {
        vec![[0.0, 0.0], [0.0, dx], [dz, top_x], [dz, 0.0]]
    } else {
        vec![[0.0, 0.0], [0.0, dx], [dz, 0.0]]
    };
    extrude_polygon(
        &profile,
        &[],
        position,
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, dy, 0.0],
    )
}

/// Signed volume of a closed faceted shape (divergence theorem); used to
/// normalize outward orientation of sweep/loft results.
fn signed_volume(f: &Faceted) -> f64 {
    let mut v6 = 0.0;
    for t in &f.tris {
        let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
        v6 += dot(a, cross(b, c));
    }
    v6 / 6.0
}

/// Solid from an externally supplied triangle soup (an imported STL surface,
/// a marching-cubes iso-surface, etc.). The input must describe a closed,
/// non-self-intersecting surface; the winding is normalized to outward via the
/// signed volume (every triangle flips together when the soup is inward). The
/// triangles are grouped as an import is, at creases of
/// [`crate::import::CREASE_DEG`]: a flat group gets a plane carrier, a curved
/// one its own facets as a discrete carrier, and the creases become edges.
pub fn mesh_solid(verts: &[[f64; 3]], tris: &[[u32; 3]]) -> Faceted {
    assert!(!tris.is_empty(), "mesh_solid needs at least one triangle");
    // Outward first, so the carriers see the right side.
    let vol: f64 = tris
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|v| verts[v as usize]);
            dot(a, cross(b, c))
        })
        .sum();
    assert!(vol.abs() > 0.0, "degenerate mesh_solid (zero volume)");
    let soup: Vec<Tri> = tris
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|v| verts[v as usize]);
            if vol < 0.0 {
                Tri::new(a, c, b)
            } else {
                Tri::new(a, b, c)
            }
        })
        .collect();
    crate::import::faceted_from_tris(soup, crate::import::CREASE_DEG)
}

/// Circular frustum from `base_center` along the full height vector `axis`,
/// radii `r_base` and `r_top`: a cylinder where they are equal, a cone where
/// `r_top == 0` (its top row an apex fan, the one rotationally symmetric
/// spot, as gmsh's apex). The barrel is a structured grid of `rows` height
/// levels with a linear taper (one row: full-height quads; more: cells
/// about square, near-equilateral triangles as gmsh and tetgen distribute
/// them). Surface groups: barrel ([`Surface::Cylinder`] or
/// [`Surface::Cone`]), top cap (absent for a cone), bottom cap; flat fans
/// (the mesher remeshes them).
pub fn frustum(
    base_center: [f64; 3],
    axis: [f64; 3],
    r_base: f64,
    r_top: f64,
    segments: usize,
    rows: usize,
) -> Faceted {
    assert!(
        segments >= 3 && rows >= 1,
        "need >= 3 segments and >= 1 row"
    );
    assert!(
        r_base > 0.0 && r_top >= 0.0,
        "radii must be positive (top may be 0)"
    );
    let (e1, e2) = orthonormal_basis(axis);
    let ring = |center: [f64; 3], r: f64| -> Vec<[f64; 3]> {
        (0..segments)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
                add(center, add(scale(e1, r * a.cos()), scale(e2, r * a.sin())))
            })
            .collect()
    };
    let radius_at = |k: usize| r_base + (r_top - r_base) * (k as f64 / rows as f64);
    let center_at = |k: usize| add(base_center, scale(axis, k as f64 / rows as f64));
    let levels: Vec<Vec<[f64; 3]>> = (0..=rows)
        .map(|k| ring(center_at(k), radius_at(k)))
        .collect();

    let mut f = Faceted::new();
    let barrel_kind = if r_top == r_base {
        Surface::cylinder(base_center, axis, r_base)
    } else {
        let factor = r_base / (r_base - r_top);
        let apex = add(base_center, scale(axis, factor));
        Surface::cone(
            apex,
            scale(axis, -factor),
            (r_base / (len(axis) * factor).abs()).atan(),
        )
    };
    let barrel = f.add_surface(barrel_kind);
    let top_center = add(base_center, axis);

    for k in 0..rows {
        let (lo, hi) = (&levels[k], &levels[k + 1]);
        let top_apex = r_top == 0.0 && k == rows - 1;
        for i in 0..segments {
            let j = (i + 1) % segments;
            if top_apex {
                f.push_tri(Tri::new(lo[i], lo[j], top_center), barrel);
            } else {
                f.push_tri(Tri::new(lo[i], lo[j], hi[j]), barrel);
                f.push_tri(Tri::new(lo[i], hi[j], hi[i]), barrel);
            }
        }
    }
    if r_top > 0.0 {
        let top = &levels[rows];
        let cap_t = f.add_surface(Surface::plane(top_center, axis));
        let top_tris: Vec<Tri> = (0..segments)
            .map(|i| Tri::new(top_center, top[i], top[(i + 1) % segments]))
            .collect();
        f.push_flat(PlanarFacet::new(top.clone()), &top_tris, cap_t);
    }
    let bot = &levels[0];
    let cap_b = f.add_surface(Surface::plane(base_center, scale(axis, -1.0)));
    let bot_tris: Vec<Tri> = (0..segments)
        .map(|i| Tri::new(base_center, bot[(i + 1) % segments], bot[i]))
        .collect();
    let bot_loop: Vec<[f64; 3]> = bot.iter().rev().copied().collect();
    f.push_flat(PlanarFacet::new(bot_loop), &bot_tris, cap_b);
    f
}

/// Geodesic sphere (subdivided icosahedron projected onto the analytic
/// sphere). Unlike a UV tessellation (latitude rings clustering at the
/// poles, rotationally-symmetric points), the icosphere distributes
/// near-equilateral triangles isotropically over the hull, matching how gmsh
/// and tetgen tessellate a sphere. `subdivisions` controls density: the face
/// count is `20 * 4^subdivisions` (0 -> 20, 2 -> 320, 3 -> 1280). Carries the
/// exact [`Surface::Sphere`](crate::Surface::Sphere) so mesh vertices still snap onto the true
/// sphere.
pub fn icosphere(center: [f64; 3], radius: f64, subdivisions: usize) -> Faceted {
    assert!(radius > 0.0);
    // Regular icosahedron (golden-ratio rectangles).
    let t = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut verts: Vec<[f64; 3]> = vec![
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    // Loop-subdivide: each triangle -> four, sharing edge midpoints (a cache
    // keyed by the sorted endpoint pair keeps the mesh watertight).
    let mut cache: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for _ in 0..subdivisions {
        cache.clear();
        let mut next = Vec::with_capacity(faces.len() * 4);
        for tri in &faces {
            let mid = |a: usize,
                       b: usize,
                       verts: &mut Vec<[f64; 3]>,
                       cache: &mut std::collections::HashMap<(usize, usize), usize>|
             -> usize {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = cache.get(&key) {
                    return m;
                }
                let (va, vb) = (verts[a], verts[b]);
                verts.push([
                    (va[0] + vb[0]) * 0.5,
                    (va[1] + vb[1]) * 0.5,
                    (va[2] + vb[2]) * 0.5,
                ]);
                let idx = verts.len() - 1;
                cache.insert(key, idx);
                idx
            };
            let a = mid(tri[0], tri[1], &mut verts, &mut cache);
            let b = mid(tri[1], tri[2], &mut verts, &mut cache);
            let c = mid(tri[2], tri[0], &mut verts, &mut cache);
            next.push([tri[0], a, c]);
            next.push([tri[1], b, a]);
            next.push([tri[2], c, b]);
            next.push([a, b, c]);
        }
        faces = next;
    }
    // Project every vertex onto the sphere and emit.
    let proj = |v: [f64; 3]| -> [f64; 3] {
        let n = len(v);
        add(center, scale(v, radius / n))
    };
    let mut f = Faceted::new();
    let s = f.add_surface(Surface::sphere(center, radius));
    for tri in &faces {
        f.push_tri(
            Tri::new(
                proj(verts[tri[0]]),
                proj(verts[tri[1]]),
                proj(verts[tri[2]]),
            ),
            s,
        );
    }
    // Icosahedron faces are wound outward; keep that (flip all if degenerate
    // winding slipped through, as the other solid builders do).
    if signed_volume(&f) < 0.0 {
        for t in &mut f.tris {
            t.v.swap(1, 2);
        }
    }
    f
}

/// Tube swept along an open polyline path with a circular cross-section
/// (the base of sweeps and helices). One ring per path
/// node, oriented normal to the local tangent bisector, with a
/// parallel-transported frame (no twist). The path must not double back on
/// itself, and `radius` must stay below the local curvature radius or the
/// tube self-intersects (the caller controls sampling density). Facets carry
/// no analytic surface (fidelity snapping off; sample the path finely).
pub fn pipe(path: &[[f64; 3]], radius: f64, segments: usize) -> Faceted {
    assert!(path.len() >= 2, "pipe path needs at least 2 points");
    assert!(radius > 0.0 && segments >= 3);
    let n = path.len();
    let seg_dir = |i: usize| -> [f64; 3] {
        let d: [f64; 3] = std::array::from_fn(|k| path[i + 1][k] - path[i][k]);
        let l = len(d);
        assert!(l > 0.0, "pipe path points must be distinct");
        scale(d, 1.0 / l)
    };
    // Node normals: tangent bisectors (segment tangents at the ends).
    let node_normal = |i: usize| -> [f64; 3] {
        let t = if i == 0 {
            seg_dir(0)
        } else if i == n - 1 {
            seg_dir(n - 2)
        } else {
            let s = add(seg_dir(i - 1), seg_dir(i));
            let l = len(s);
            assert!(l > 1e-9, "pipe path doubles back on itself");
            scale(s, 1.0 / l)
        };
        t
    };
    // Parallel transport the ring frame from node to node (rotation taking
    // one bisector to the next; no roll accumulates).
    let mut frames: Vec<([f64; 3], [f64; 3])> = Vec::with_capacity(n);
    let n0 = node_normal(0);
    frames.push(orthonormal_basis(n0));
    for i in 1..n {
        let (prev_n, cur_n) = (node_normal(i - 1), node_normal(i));
        let (u, v) = frames[i - 1];
        let c = dot(prev_n, cur_n).clamp(-1.0, 1.0);
        let axis = cross(prev_n, cur_n);
        let s = len(axis);
        if s < 1e-12 {
            frames.push((u, v));
            continue;
        }
        let k = scale(axis, 1.0 / s);
        let rot = |p: [f64; 3]| -> [f64; 3] {
            // Rodrigues: p c + (k x p) s + k (k . p)(1 - c).
            let kxp = cross(k, p);
            let kdp = dot(k, p);
            std::array::from_fn(|m| p[m] * c + kxp[m] * s + k[m] * kdp * (1.0 - c))
        };
        frames.push((rot(u), rot(v)));
    }
    let ring = |i: usize| -> Vec<[f64; 3]> {
        let (u, v) = frames[i];
        (0..segments)
            .map(|j| {
                let t = 2.0 * std::f64::consts::PI * j as f64 / segments as f64;
                add(
                    path[i],
                    add(scale(u, radius * t.cos()), scale(v, radius * t.sin())),
                )
            })
            .collect()
    };
    let mut f = Faceted::new();
    // The barrel's analytic carrier: constant-radius tube about the path.
    let s = f.add_surface(Surface::Tube {
        path: std::sync::Arc::new(crate::tube::TubePath::new(path.to_vec())),
        radius,
    });
    let rings: Vec<Vec<[f64; 3]>> = (0..n).map(ring).collect();
    for i in 0..n - 1 {
        for j in 0..segments {
            let j2 = (j + 1) % segments;
            let (a, b) = (rings[i][j], rings[i][j2]);
            let (d, c) = (rings[i + 1][j], rings[i + 1][j2]);
            f.push_tri(Tri::new(a, b, c), s);
            f.push_tri(Tri::new(a, c, d), s);
        }
    }
    // End caps lie in planes normal to the end tangents, so they are flat.
    let cap0_tris: Vec<Tri> = (0..segments)
        .map(|j| Tri::new(path[0], rings[0][(j + 1) % segments], rings[0][j]))
        .collect();
    let cap0_loop: Vec<[f64; 3]> = rings[0].iter().rev().copied().collect();
    let cap0 = f.add_surface(Surface::plane_of(&cap0_loop));
    f.push_flat(PlanarFacet::new(cap0_loop), &cap0_tris, cap0);
    let cap1 = f.add_surface(Surface::plane_of(&rings[n - 1]));
    let cap1_tris: Vec<Tri> = (0..segments)
        .map(|j| {
            Tri::new(
                path[n - 1],
                rings[n - 1][j],
                rings[n - 1][(j + 1) % segments],
            )
        })
        .collect();
    f.push_flat(PlanarFacet::new(rings[n - 1].clone()), &cap1_tris, cap1);
    debug_assert!(signed_volume(&f) > 0.0, "pipe orientation");
    f
}

/// Helical tube around the +z axis through `base`: `radius` of the helix,
/// `pitch` advance per turn, circular wire cross-section. Composition of a
/// sampled helix path and [`pipe`].
pub fn helix(
    base: [f64; 3],
    radius: f64,
    pitch: f64,
    turns: f64,
    wire_radius: f64,
    points_per_turn: usize,
    segments: usize,
) -> Faceted {
    assert!(turns > 0.0 && points_per_turn >= 8);
    let n = ((turns * points_per_turn as f64).ceil() as usize).max(2);
    let path: Vec<[f64; 3]> = (0..=n)
        .map(|i| {
            let t = turns * i as f64 / n as f64;
            let ang = 2.0 * std::f64::consts::PI * t;
            [
                base[0] + radius * ang.cos(),
                base[1] + radius * ang.sin(),
                base[2] + pitch * t,
            ]
        })
        .collect();
    pipe(&path, wire_radius, segments)
}

/// Ruled loft between two planar profiles with the SAME vertex count,
/// corresponded by index (horn tapers). Caps are fanned from
/// the profile centroids, so each profile must be star-shaped with respect
/// to its centroid (convex profiles always are). Each side carries the
/// ruled surface between its two profile edges: a plane where its four
/// corners lie in one, else the bilinear patch through them. Output
/// orientation is normalized to outward via the signed volume.
pub fn loft(profile_a: &[[f64; 3]], profile_b: &[[f64; 3]]) -> Faceted {
    assert!(
        profile_a.len() == profile_b.len() && profile_a.len() >= 3,
        "loft profiles need the same vertex count (>= 3)"
    );
    let n = profile_a.len();
    let centroid = |pts: &[[f64; 3]]| -> [f64; 3] {
        let mut c = [0.0; 3];
        for p in pts {
            for k in 0..3 {
                c[k] += p[k];
            }
        }
        scale(c, 1.0 / pts.len() as f64)
    };
    let (ca, cb) = (centroid(profile_a), centroid(profile_b));
    let mut f = Faceted::new();
    let reversed: Vec<[f64; 3]> = profile_a.iter().rev().copied().collect();
    let cap_a = f.add_surface(Surface::plane_through(&reversed));
    for i in 0..n {
        let j = (i + 1) % n;
        f.push_tri(Tri::new(ca, profile_a[j], profile_a[i]), cap_a);
    }
    let cap_b = f.add_surface(Surface::plane_through(profile_b));
    for i in 0..n {
        let j = (i + 1) % n;
        f.push_tri(Tri::new(cb, profile_b[i], profile_b[j]), cap_b);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        let (a, b) = (profile_a[i], profile_a[j]);
        let (d, c) = (profile_b[i], profile_b[j]);
        let ruled = Surface::plane_through(&[a, b, c, d]).unwrap_or_else(|| {
            let knots = vec![0.0, 0.0, 1.0, 1.0];
            Surface::Nurbs(Arc::new(NurbsSurface::new(
                [1, 1],
                [knots.clone(), knots],
                [2, 2],
                vec![a, d, b, c],
                vec![1.0; 4],
            )))
        });
        let side = f.add_surface(Some(ruled));
        f.push_tri(Tri::new(a, b, c), side);
        f.push_tri(Tri::new(a, c, d), side);
    }
    let vol = signed_volume(&f);
    assert!(vol.abs() > 0.0, "degenerate loft");
    if vol < 0.0 {
        for t in &mut f.tris {
            t.v.swap(1, 2);
        }
    }
    f
}

// ------------------------------------------------------------------ sheets

/// Parallelogram sheet spanned by `u`, `v` at `corner`.
pub fn sheet_rect(corner: [f64; 3], u: [f64; 3], v: [f64; 3]) -> Faceted {
    let mut f = Faceted::new();
    let s = f.add_surface(Surface::plane(corner, cross(u, v)));
    let b = add(corner, u);
    let c = add(add(corner, u), v);
    let d = add(corner, v);
    let tris = [Tri::new(corner, b, c), Tri::new(corner, c, d)];
    f.push_flat(PlanarFacet::new(vec![corner, b, c, d]), &tris, s);
    f.corners = vec![corner, b, c, d];
    f
}

/// The least turn (degrees) at a vertex of a polygon sheet that makes it a
/// corner of the sheet.
const POLYGON_CORNER_DEG: f64 = 20.0;

/// Planar polygon sheet (with holes) in the (base, u, v) frame.
pub fn sheet_polygon(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    base: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
) -> Faceted {
    // The rings run as the triangles do: the outline counterclockwise, the
    // holes clockwise.
    let turned = |pts: &[[f64; 2]], want: Sign| -> Vec<[f64; 2]> {
        let o = polygon_orientation(pts);
        assert_ne!(o, Sign::Zero, "degenerate polygon ring");
        if o == want {
            pts.to_vec()
        } else {
            pts.iter().rev().copied().collect()
        }
    };
    let outer = &turned(outer, Sign::Positive)[..];
    let holes: Vec<Vec<[f64; 2]>> = holes.iter().map(|h| turned(h, Sign::Negative)).collect();
    let holes = &holes[..];
    let mut f = Faceted::new();
    let s = f.add_surface(Surface::plane(base, cross(u, v)));
    let tris: Vec<Tri> = triangulate_polygon(outer, holes)
        .iter()
        .map(|t| {
            Tri::new(
                embed(base, u, v, t[0]),
                embed(base, u, v, t[1]),
                embed(base, u, v, t[2]),
            )
        })
        .collect();
    let outer_loop: Vec<[f64; 3]> = outer.iter().map(|&p| embed(base, u, v, p)).collect();
    let hole_loops: Vec<Vec<[f64; 3]>> = holes
        .iter()
        .map(|h| h.iter().map(|&p| embed(base, u, v, p)).collect())
        .collect();
    // A vertex where the ring turns by `POLYGON_CORNER_DEG` or more is a
    // corner; finer turns sample a smooth outline (a circle given as a
    // polygon), which stays one curve.
    let cos_max = POLYGON_CORNER_DEG.to_radians().cos();
    for ring in std::iter::once(outer).chain(holes.iter().map(|h| h.as_slice())) {
        let n = ring.len();
        for k in 0..n {
            let (a, b, c) = (ring[(k + n - 1) % n], ring[k], ring[(k + 1) % n]);
            let (d0, d1) = ([b[0] - a[0], b[1] - a[1]], [c[0] - b[0], c[1] - b[1]]);
            let l = (d0[0].hypot(d0[1]) * d1[0].hypot(d1[1])).max(f64::MIN_POSITIVE);
            let turns = (d0[0] * d1[0] + d0[1] * d1[1]) / l < cos_max;
            if turns && polygon_orientation(&[a, b, c]) != Sign::Zero {
                f.corners.push(embed(base, u, v, b));
            }
        }
    }
    f.push_flat(PlanarFacet::with_holes(outer_loop, hole_loops), &tris, s);
    f
}

/// Elliptic disk sheet at `center` spanned by the radius vectors `e1`, `e2`.
pub fn sheet_disk(center: [f64; 3], e1: [f64; 3], e2: [f64; 3], segments: usize) -> Faceted {
    assert!(segments >= 3);
    let mut f = Faceted::new();
    let s = f.add_surface(Surface::plane(center, cross(e1, e2)));
    let ring: Vec<[f64; 3]> = (0..segments)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
            add(center, add(scale(e1, t.cos()), scale(e2, t.sin())))
        })
        .collect();
    let tris: Vec<Tri> = (0..segments)
        .map(|i| Tri::new(center, ring[i], ring[(i + 1) % segments]))
        .collect();
    // The rim is the exact ellipse (a circle where the radius vectors are
    // square and of one length) the ring samples, closed on its first
    // point: the B-rep edge takes it as its carrier, so the mesh samples it
    // by its curvature, not by the ring.
    let mut points = ring.clone();
    points.push(ring[0]);
    f.curves.push(EdgeCurve {
        curve: Curve::Ellipse {
            c: center,
            p: e1,
            q: e2,
        },
        points,
    });
    f.push_flat(PlanarFacet::new(ring), &tris, s);
    f
}

/// Sheet over a NURBS surface: a `segments[0] x segments[1]` grid over its
/// parameter domain, carried by the exact surface so the mesher snaps to it.
/// Triangles collapsed by a degenerate edge (a pole) are dropped; the four
/// corners of the domain are declared corners.
pub fn sheet_nurbs(surface: &NurbsSurface, segments: [usize; 2]) -> Faceted {
    assert!(segments[0] >= 1 && segments[1] >= 1);
    let (ud, vd) = surface.domain();
    let at = |i: usize, j: usize| {
        let u = ud[0] + (ud[1] - ud[0]) * i as f64 / segments[0] as f64;
        let v = vd[0] + (vd[1] - vd[0]) * j as f64 / segments[1] as f64;
        surface.eval(u, v)
    };
    let mut f = Faceted::new();
    let s = f.add_surface(Surface::Nurbs(Arc::new(surface.clone())));
    let area2 = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
        let n = cross(
            rapidmesh_exact::vector::sub(b, a),
            rapidmesh_exact::vector::sub(c, a),
        );
        dot(n, n)
    };
    for i in 0..segments[0] {
        for j in 0..segments[1] {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            for t in [[a, b, c], [a, c, d]] {
                if area2(t[0], t[1], t[2]) > 0.0 {
                    f.push_tri(Tri::new(t[0], t[1], t[2]), s);
                }
            }
        }
    }
    let mut corners: Vec<[f64; 3]> = Vec::new();
    for p in [
        at(0, 0),
        at(segments[0], 0),
        at(segments[0], segments[1]),
        at(0, segments[1]),
    ] {
        if !corners.contains(&p) {
            corners.push(p);
        }
    }
    f.corners = corners;
    f
}

/// The solid a flat sheet sweeps along `w` (not parallel to it): the sheet's
/// own faces as the bottom (its vertices kept exactly), the same moved by
/// `w` as the top, and a wall along every boundary edge. A wall along a
/// circle the sheet declares (a disc's rim, a round hole cut by one) is on
/// the cylinder the circle sweeps, which needs `w` along its axis; every
/// other wall is a plane of its own. Surfaces: bottom, top, then the walls;
/// the sheet's corners are corners at both ends.
pub fn extrude_sheet(sheet: &Faceted, w: [f64; 3]) -> Result<Faceted, String> {
    if sheet.flats.is_empty()
        || sheet.tris.len() != sheet.flats.iter().map(|f| f.tris.len()).sum::<usize>()
    {
        return Err("only a flat sheet extrudes".into());
    }
    let wl = dot(w, w).sqrt();
    if !(wl > 0.0 && wl.is_finite()) {
        return Err("an extrusion needs a nonzero vector".into());
    }
    let mut f = Faceted::new();
    f.frame = sheet.frame;
    // The sheet's plane: its facets' area vector, the side the sweep goes.
    let n0 = sheet.tris.iter().fold([0.0; 3], |acc, t| {
        add(
            acc,
            cross(
                rapidmesh_exact::vector::sub(t.v[1], t.v[0]),
                rapidmesh_exact::vector::sub(t.v[2], t.v[0]),
            ),
        )
    });
    let sign = if dot(n0, w) > 0.0 { 1.0 } else { -1.0 };
    let p0 = sheet.tris[0].v[0];
    let bottom = f.add_surface(Surface::plane(p0, scale(n0, -sign)));
    let top = f.add_surface(Surface::plane(add(p0, w), scale(n0, sign)));
    let up = |p: [f64; 3]| add(p, w);
    // The declared curve a wall runs along: its ends and its middle on the
    // curve's points (a crossing a boolean cut into a rim lies on a chord).
    let (lo, hi) = rapidmesh_exact::vector::bbox(sheet.tris.iter().flat_map(|t| t.v.iter()));
    let tol = 1e-9
        * dot(
            rapidmesh_exact::vector::sub(hi, lo),
            rapidmesh_exact::vector::sub(hi, lo),
        )
        .sqrt();
    let curve_of = |a: [f64; 3], b: [f64; 3]| {
        let mid = scale(add(a, b), 0.5);
        sheet.curves.iter().position(|c| {
            [a, b, mid].iter().all(|&q| {
                c.points
                    .windows(2)
                    .any(|s| rapidmesh_exact::vector::segment_dist2(q, s[0], s[1]) <= tol * tol)
            })
        })
    };
    // The cylinder each declared circle sweeps, made at its first wall.
    let mut swept: Vec<Option<u32>> = vec![None; sheet.curves.len()];
    for fl in &sheet.flats {
        let tris = &sheet.tris[fl.tris.clone()];
        let n = tris.iter().fold([0.0; 3], |acc, t| {
            add(
                acc,
                cross(
                    rapidmesh_exact::vector::sub(t.v[1], t.v[0]),
                    rapidmesh_exact::vector::sub(t.v[2], t.v[0]),
                ),
            )
        });
        let along = dot(n, w);
        if !(along.abs() > 1e-9 * dot(n, n).sqrt() * wl) {
            return Err("an extrusion vector must leave the sheet's plane".into());
        }
        // The loops wind about n; the sweep goes with n or against it.
        let with = along > 0.0;
        let flip = |t: [[f64; 3]; 3], out: bool| if out { t } else { [t[0], t[2], t[1]] };
        let cap = |pts: &[[f64; 3]]| pts.to_vec();
        let down: Vec<Tri> = tris
            .iter()
            .map(|t| {
                let v = flip(t.v, !with);
                Tri::new(v[0], v[1], v[2])
            })
            .collect();
        let raised: Vec<Tri> = tris
            .iter()
            .map(|t| {
                let v = flip(t.v.map(up), with);
                Tri::new(v[0], v[1], v[2])
            })
            .collect();
        let lower = PlanarFacet::with_holes(cap(&fl.facet.outer), fl.facet.holes.clone());
        let upper = PlanarFacet::with_holes(
            fl.facet.outer.iter().map(|&p| up(p)).collect(),
            fl.facet
                .holes
                .iter()
                .map(|h| h.iter().map(|&p| up(p)).collect())
                .collect(),
        );
        f.push_flat(lower, &down, bottom);
        f.push_flat(upper, &raised, top);
        for lp in std::iter::once(&fl.facet.outer).chain(&fl.facet.holes) {
            for k in 0..lp.len() {
                let (a, b) = (lp[k], lp[(k + 1) % lp.len()]);
                let (ah, bh) = (up(a), up(b));
                let rim = match curve_of(a, b) {
                    Some(ci) => Some(match swept[ci] {
                        Some(s) => s,
                        None => {
                            let c = &sheet.curves[ci].curve;
                            let (Some((center, r)), Some(axis)) = (c.as_circle(), c.axis()) else {
                                return Err(format!("a {} rim does not extrude", c.name()));
                            };
                            let sin = len(cross(axis, w)) / wl;
                            if sin > 1e-9 {
                                return Err("a circular rim extrudes along its axis only (an oblique sweep is an elliptic cylinder)".into());
                            }
                            let s = f.add_surface(Surface::cylinder(center, axis, r));
                            swept[ci] = Some(s);
                            s
                        }
                    }),
                    None => None,
                };
                let s = rim.unwrap_or_else(|| {
                    f.add_surface(Surface::plane(
                        a,
                        scale(
                            cross(rapidmesh_exact::vector::sub(b, a), w),
                            if with { 1.0 } else { -1.0 },
                        ),
                    ))
                });
                let tris = [[a, b, bh], [a, bh, ah]].map(|t| {
                    let t = flip(t, with);
                    Tri::new(t[0], t[1], t[2])
                });
                // A wall of its own plane is a flat facet; one on the rim's
                // curved carrier is its two triangles.
                if rim.is_some() {
                    tris.into_iter().for_each(|t| f.push_tri(t, s));
                } else {
                    let quad = if with { [a, b, bh, ah] } else { [a, ah, bh, b] };
                    f.push_flat(PlanarFacet::new(quad.to_vec()), &tris, s);
                }
            }
        }
    }
    f.corners = sheet.corners.iter().flat_map(|&c| [c, up(c)]).collect();
    let vol = signed_volume(&f);
    if !(vol > 0.0) {
        return Err(format!("the extruded solid has no volume ({vol})"));
    }
    Ok(f)
}

// ---------------------------------------------------------------- revolve

/// An edge of a revolve profile, from its vertex to the next one.
#[derive(Clone, Debug, PartialEq)]
pub enum ProfileEdge {
    /// Straight.
    Line,
    /// A circular arc of bulge `tan(angle / 4)`, the swept angle positive
    /// counterclockwise in `(r, z)` (the DXF convention).
    Arc(f64),
    /// A cubic spline through these interior points.
    Spline(Vec<[f64; 2]>),
}

/// What an edge of a profile is, sampled: the line between its ends, a
/// circle (centre, radius) or a spline.
enum Piece {
    Line([f64; 2], [f64; 2]),
    Arc([f64; 2], f64),
    Spline(Arc<NurbsCurve>),
}

/// The closed profile `verts` / `edges` sampled: per edge its samples (its
/// start vertex first, not its end) and what it is. Arcs take the facet
/// count of their radius at `maxh` and `tol`, splines 8 samples per point.
fn profile_pieces(
    verts: &[[f64; 2]],
    edges: &[ProfileEdge],
    maxh: Option<f64>,
    tol: f64,
) -> Result<(Vec<Vec<[f64; 2]>>, Vec<Piece>), String> {
    use std::f64::consts::TAU;
    let n = verts.len();
    let mut samples = Vec::with_capacity(n);
    let mut pieces = Vec::with_capacity(n);
    for (i, e) in edges.iter().enumerate() {
        let (p, q) = (verts[i], verts[(i + 1) % n]);
        if p == q {
            return Err(format!("profile edge {i} has no length"));
        }
        match e {
            ProfileEdge::Line | ProfileEdge::Arc(0.0) => {
                samples.push(vec![p]);
                pieces.push(Piece::Line(p, q));
            }
            ProfileEdge::Arc(b) => {
                if !(b.abs() < 1e6) {
                    return Err(format!("profile edge {i}: bulge {b} is not finite"));
                }
                let (dr, dz) = (q[0] - p[0], q[1] - p[1]);
                let chord = dr.hypot(dz);
                let sweep = 4.0 * b.atan();
                let rho = 0.5 * chord / (0.5 * sweep).sin().abs();
                let h = 0.5 * chord / (0.5 * sweep).tan();
                let c = [
                    0.5 * (p[0] + q[0]) - dz / chord * h,
                    0.5 * (p[1] + q[1]) + dr / chord * h,
                ];
                let a0 = (p[1] - c[1]).atan2(p[0] - c[0]);
                let m = ((facet_count(rho, maxh, tol) as f64 * sweep.abs() / TAU).ceil() as usize)
                    .max(1);
                samples.push(
                    (0..m)
                        .map(|k| {
                            let t = a0 + sweep * k as f64 / m as f64;
                            if k == 0 {
                                p
                            } else {
                                [c[0] + rho * t.cos(), c[1] + rho * t.sin()]
                            }
                        })
                        .collect(),
                );
                pieces.push(Piece::Arc(c, rho));
            }
            ProfileEdge::Spline(inner) => {
                if inner.len() < 2 {
                    return Err(format!(
                        "profile edge {i}: a spline needs at least 2 interior points"
                    ));
                }
                let pts: Vec<[f64; 2]> = std::iter::once(p)
                    .chain(inner.iter().copied())
                    .chain(std::iter::once(q))
                    .collect();
                let curve = NurbsCurve::interpolate(&pts);
                let (lo, hi) = curve.domain();
                let m = (8 * (inner.len() + 1)).clamp(8, 256);
                samples.push(
                    (0..m)
                        .map(|k| {
                            if k == 0 {
                                p
                            } else {
                                curve.eval(lo + (hi - lo) * k as f64 / m as f64)
                            }
                        })
                        .collect(),
                );
                pieces.push(Piece::Spline(Arc::new(curve)));
            }
        }
    }
    Ok((samples, pieces))
}

/// Prism of the closed profile `verts` / `edges` in the orthonormal frame
/// `(base; u, v)`, swept along `h` (square to the frame, `(u x v) . h > 0`).
/// Every edge is a wall with its exact carrier: a plane for a line, a
/// cylinder for an arc, an [`Surface::Extruded`](crate::Surface::Extruded) for a spline; then the
/// bottom and top caps, in that order after the walls.
#[allow(clippy::too_many_arguments)]
pub fn extrude_profile(
    verts: &[[f64; 2]],
    edges: &[ProfileEdge],
    base: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
    h: [f64; 3],
    maxh: Option<f64>,
    tol: f64,
) -> Result<Faceted, String> {
    let n = verts.len();
    if n < 2 || edges.len() != n {
        return Err("a profile needs at least 2 vertices, one edge each".into());
    }
    let hl = dot(h, h).sqrt();
    let unit = |w: [f64; 3]| (dot(w, w) - 1.0).abs() < 1e-12;
    let square = |a: [f64; 3], b: [f64; 3]| dot(a, b).abs() < 1e-12 * hl.max(1.0);
    if !(unit(u) && unit(v) && square(u, v) && square(u, h) && square(v, h) && hl > 0.0)
        || dot(cross(u, v), h) <= 0.0
    {
        return Err("an extrusion needs an orthonormal frame and a sweep along u x v".into());
    }
    let axis = scale(h, 1.0 / hl);
    let (samples, pieces) = profile_pieces(verts, edges, maxh, tol)?;
    let ring: Vec<[f64; 2]> = samples.concat();
    let ccw = match polygon_orientation(&ring) {
        Sign::Positive => true,
        Sign::Negative => false,
        Sign::Zero => return Err("the profile encloses no area".into()),
    };
    let at = |p: [f64; 2]| embed(base, u, v, p);
    let mut f = Faceted::new();
    for (i, piece) in pieces.into_iter().enumerate() {
        let kind = match piece {
            // Its outward side: counterclockwise, (b - a) x h.
            Piece::Line(p, q) => Surface::plane(
                at(p),
                scale(
                    cross(rapidmesh_exact::vector::sub(at(q), at(p)), h),
                    if ccw { 1.0 } else { -1.0 },
                ),
            ),
            Piece::Arc(c, rho) => Some(Surface::cylinder(at(c), axis, rho)),
            Piece::Spline(curve) => Some(Surface::Extruded {
                frame: Frame {
                    o: base,
                    x: u,
                    y: v,
                    z: axis,
                },
                profile: curve,
            }),
        };
        let s = f.add_surface(kind);
        let next = samples[(i + 1) % n][0];
        let pts: Vec<[f64; 2]> = samples[i].iter().copied().chain([next]).collect();
        for w in pts.windows(2) {
            // Counterclockwise, (a, b, b + h, a + h) faces out.
            let (a, b) = (at(w[0]), at(w[1]));
            let (ah, bh) = (add(a, h), add(b, h));
            for t in [[a, b, bh], [a, bh, ah]] {
                let t = if ccw { t } else { [t[0], t[2], t[1]] };
                f.push_tri(Tri::new(t[0], t[1], t[2]), s);
            }
        }
    }
    let loop2: Vec<[f64; 2]> = if ccw {
        ring.clone()
    } else {
        ring.iter().rev().copied().collect()
    };
    let tris = triangulate_polygon(&loop2, &[]);
    // The bottom faces against h, the top along it.
    for (shift, up) in [([0.0; 3], false), (h, true)] {
        let s = f.add_surface(Surface::plane(
            add(at(loop2[0]), shift),
            scale(cross(u, v), if up { 1.0 } else { -1.0 }),
        ));
        let mut lp: Vec<[f64; 3]> = loop2.iter().map(|&p| add(at(p), shift)).collect();
        let ts: Vec<Tri> = tris
            .iter()
            .map(|t| {
                let [p0, p1, p2] = t.map(|q| add(at(q), shift));
                if up {
                    Tri::new(p0, p1, p2)
                } else {
                    Tri::new(p0, p2, p1)
                }
            })
            .collect();
        if !up {
            lp.reverse();
        }
        f.push_flat(PlanarFacet::new(lp), &ts, s);
    }
    Ok(f)
}

/// Solid of revolution: the closed profile `verts` in the half-plane
/// `(r, z)` of the axis through `origin` along `axis`, edge `i` running from
/// vertex `i` to the next, turned by `angle` radians (a full turn at `TAU`).
/// Every edge is its own surface, with its exact carrier: a straight edge
/// a plane, cylinder or cone, an arc a sphere or torus, a spline a
/// [`Surface::Revolved`](crate::Surface::Revolved); an edge on the axis has no faces. A part turn
/// adds the start and end caps as the two last surfaces. The facet counts
/// follow [`facet_count`] for the target size `maxh` and tolerance `tol`,
/// at least `min_segments` around a full turn.
#[allow(clippy::too_many_arguments)]
pub fn revolve(
    verts: &[[f64; 2]],
    edges: &[ProfileEdge],
    origin: [f64; 3],
    axis: [f64; 3],
    angle: f64,
    maxh: Option<f64>,
    tol: f64,
    min_segments: usize,
) -> Result<Faceted, String> {
    use std::f64::consts::TAU;
    let n = verts.len();
    if n < 2 || edges.len() != n {
        return Err("a revolve profile needs at least 2 vertices, one edge each".into());
    }
    if verts.iter().any(|v| !(v[0] >= 0.0) || !v[1].is_finite()) {
        return Err("a revolve profile must stay in r >= 0".into());
    }
    if !(angle > 0.0 && angle <= TAU) {
        return Err(format!("revolve angle {angle} is not in (0, 2 pi]"));
    }
    let len = dot(axis, axis).sqrt();
    if !(len > 0.0) {
        return Err("revolve axis must be nonzero".into());
    }
    let a = scale(axis, 1.0 / len);
    let pick = if a[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let x = {
        let c = cross(a, pick);
        scale(c, 1.0 / dot(c, c).sqrt())
    };
    // As many turns as the widest radius needs at the target size.
    let rmax = profile_pieces(verts, edges, maxh, tol)?
        .0
        .concat()
        .iter()
        .map(|v| v[0])
        .fold(0.0, f64::max);
    let full = angle >= TAU;
    let turns = ((facet_count(rmax, maxh, tol).max(min_segments) as f64 * angle / TAU).ceil()
        as usize)
        .max(if full { 3 } else { 1 });
    let angles: Vec<f64> = (0..turns + usize::from(!full))
        .map(|k| angle * k as f64 / turns as f64)
        .collect();
    revolve_at(verts, edges, origin, a, x, &angles, full, maxh, tol)
}

/// [`revolve`] at the given `angles` (ascending radians from the unit
/// direction `x`, square to the unit `axis`): a full turn through them all
/// (`full`, the last back to the first), else from the first to the last,
/// capped there.
#[allow(clippy::too_many_arguments)]
pub fn revolve_at(
    verts: &[[f64; 2]],
    edges: &[ProfileEdge],
    origin: [f64; 3],
    a: [f64; 3],
    x: [f64; 3],
    angles: &[f64],
    full: bool,
    maxh: Option<f64>,
    tol: f64,
) -> Result<Faceted, String> {
    let n = verts.len();
    if n < 2 || edges.len() != n {
        return Err("a revolve profile needs at least 2 vertices, one edge each".into());
    }
    if angles.len() < if full { 3 } else { 2 } || angles.windows(2).any(|w| !(w[0] < w[1])) {
        return Err("revolve angles must ascend, at least 3 for a full turn".into());
    }
    let y = cross(a, x);
    let size = verts
        .iter()
        .flat_map(|v| verts.iter().map(move |w| (v[0] - w[0]).hypot(v[1] - w[1])))
        .fold(0.0, f64::max);

    // The samples of every edge (its start vertex, not its end) and carrier.
    let (samples, pieces) = profile_pieces(verts, edges, maxh, tol)?;
    let on_axis = |z: f64| add(origin, scale(a, z));
    let kinds: Vec<Option<Surface>> = pieces
        .into_iter()
        .map(|piece| match piece {
            Piece::Line(p, q) => {
                if p[0] == 0.0 && q[0] == 0.0 {
                    None
                } else if p[1] == q[1] {
                    // Across the axis at its height; the profile's interior
                    // lies left of it, its outside right: (dz, -dr) in (r, z).
                    Surface::plane(on_axis(p[1]), scale(a, p[0] - q[0]))
                } else if p[0] == q[0] {
                    Some(Surface::cylinder(origin, a, p[0]))
                } else {
                    // The apex where the line meets the axis; the cone
                    // opens the way r grows.
                    let slope = (q[0] - p[0]) / (q[1] - p[1]);
                    Some(Surface::cone(
                        on_axis(p[1] - p[0] / slope),
                        scale(a, slope.signum()),
                        (slope.abs()).atan(),
                    ))
                }
            }
            // A circle about a point of the axis turns into a sphere; off
            // the axis into a torus (the same about -r as about r).
            Piece::Arc(c, rho) => Some(if c[0].abs() <= 1e-12 * size {
                Surface::sphere(on_axis(c[1]), rho)
            } else {
                Surface::torus(on_axis(c[1]), a, c[0].abs(), rho)
            }),
            Piece::Spline(curve) => Some(Surface::Revolved {
                frame: Frame::new(origin, a, Some(x)).expect("a revolution has an axis"),
                profile: curve,
            }),
        })
        .collect();
    let ring: Vec<[f64; 2]> = samples.concat();
    if ring.iter().any(|v| v[0] < 0.0) {
        return Err("a revolve profile must stay in r >= 0".into());
    }
    let rmax = ring.iter().map(|v| v[0]).fold(0.0, f64::max);
    if !(rmax > 0.0) {
        return Err("a revolve profile must leave the axis".into());
    }
    // The solid on the left of the profile: counterclockwise in (r, z).
    let ccw = match polygon_orientation(&ring) {
        Sign::Positive => true,
        Sign::Negative => false,
        Sign::Zero => return Err("the revolve profile encloses no area".into()),
    };
    let turns = if full { angles.len() } else { angles.len() - 1 };
    let point = |v: [f64; 2], k: usize| -> [f64; 3] {
        if v[0] == 0.0 {
            return on_axis(v[1]);
        }
        let t = angles[if full { k % turns } else { k }];
        let dir = add(scale(x, t.cos()), scale(y, t.sin()));
        add(on_axis(v[1]), scale(dir, v[0]))
    };
    let area2 = |t: &[[f64; 3]; 3]| {
        let c = cross(
            rapidmesh_exact::vector::sub(t[1], t[0]),
            rapidmesh_exact::vector::sub(t[2], t[0]),
        );
        dot(c, c)
    };
    let mut f = Faceted::new();
    let push = |f: &mut Faceted, t: [[f64; 3]; 3], s: u32| {
        let t = if ccw { t } else { [t[0], t[2], t[1]] };
        if area2(&t) > 0.0 {
            f.push_tri(Tri::new(t[0], t[1], t[2]), s);
        }
    };
    for (i, kind) in kinds.into_iter().enumerate() {
        // The spline carrier runs the profile's way; turn it so its normal
        // points out.
        let kind = match kind {
            Some(Surface::Revolved { frame, profile }) if !ccw => Some(Surface::Revolved {
                frame,
                profile: Arc::new(profile.reversed()),
            }),
            k => k,
        };
        // An edge on the axis keeps its surface slot (roles follow the
        // edges) but has no faces.
        let s = f.add_surface(kind.clone());
        if kind.is_none() {
            continue;
        }
        let next = samples[(i + 1) % n][0];
        let pts: Vec<[f64; 2]> = samples[i].iter().copied().chain([next]).collect();
        for w in pts.windows(2) {
            for k in 0..turns {
                let (p0, p1) = (point(w[0], k), point(w[0], k + 1));
                let (q0, q1) = (point(w[1], k), point(w[1], k + 1));
                push(&mut f, [p0, q1, q0], s);
                push(&mut f, [p0, p1, q1], s);
            }
        }
    }
    if !full {
        // Start cap: counterclockwise in (r, z) faces along x cross axis,
        // out of the solid at theta = 0; the end cap the other way.
        let loop2: Vec<[f64; 2]> = if ccw {
            ring.clone()
        } else {
            ring.iter().rev().copied().collect()
        };
        let tris = triangulate_polygon(&loop2, &[]);
        for (k, outward) in [(0, true), (turns, false)] {
            // Through the axis at the cap's angle, facing x cross axis
            // turned there, the end cap the other way.
            let t = angles[k];
            let dir = add(scale(x, t.cos()), scale(y, t.sin()));
            let s = f.add_surface(Surface::plane(
                origin,
                scale(cross(dir, a), if outward { 1.0 } else { -1.0 }),
            ));
            let mut lp: Vec<[f64; 3]> = loop2.iter().map(|&v| point(v, k)).collect();
            let ts: Vec<Tri> = tris
                .iter()
                .map(|t| {
                    let (p0, p1, p2) = (point(t[0], k), point(t[1], k), point(t[2], k));
                    if outward {
                        Tri::new(p0, p1, p2)
                    } else {
                        Tri::new(p0, p2, p1)
                    }
                })
                .collect();
            if !outward {
                lp.reverse();
            }
            f.push_flat(PlanarFacet::new(lp), &ts, s);
        }
    }
    let vol = signed_volume(&f);
    if !(vol > 0.0) {
        return Err(format!("the revolved solid has no volume ({vol})"));
    }
    Ok(f)
}
