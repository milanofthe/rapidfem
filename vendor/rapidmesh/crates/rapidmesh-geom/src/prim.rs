//! Primitive shapes: solids (closed, outward-oriented) and sheets (open
//! surfaces, e.g. PEC traces and ports, embeddable in volumes).
//!
//! Every builder returns a [`Faceted`] with per-triangle analytic surface
//! back-references. Solid builders guarantee watertightness and outward
//! orientation; sheet builders guarantee consistent winding.

use crate::faceted::{Faceted, SurfaceKind};
use crate::nurbs::NurbsCurve;
use crate::nurbs_surface::NurbsSurface;
use crate::polygon::{polygon_orientation, triangulate_polygon};
use rapidmesh_csg::{PlanarFacet, Tri};
use rapidmesh_exact::Sign;
use std::sync::Arc;

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// Two unit vectors orthogonal to `axis` (and to each other).
fn orthonormal_basis(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let n = norm(axis);
    assert!(n > 0.0, "axis must be nonzero");
    let a = scale(axis, 1.0 / n);
    // Pick the coordinate axis least aligned with `a`.
    let helper = if a[0].abs() <= a[1].abs() && a[0].abs() <= a[2].abs() {
        [1.0, 0.0, 0.0]
    } else if a[1].abs() <= a[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let e1 = cross3(a, helper);
    let e1 = scale(e1, 1.0 / norm(e1));
    let e2 = cross3(a, e1);
    (e1, e2)
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
    let mut f = Faceted::new();
    for q in quads {
        let s = f.add_surface(SurfaceKind::Plane);
        let loop4 = vec![c[q[0]], c[q[1]], c[q[2]], c[q[3]]];
        let tris = [
            Tri::new(c[q[0]], c[q[1]], c[q[2]]),
            Tri::new(c[q[0]], c[q[2]], c[q[3]]),
        ];
        f.push_flat(PlanarFacet::new(loop4), &tris, s);
    }
    f
}

/// Circular frustum from `base_center` along the full height vector `axis`,
/// radii `r_base` and `r_top` (`r_top == 0` gives a cone). Surface groups:
/// barrel (`Cylinder`/`Cone`), bottom cap, top cap (absent for a cone).
pub fn frustum(
    base_center: [f64; 3],
    axis: [f64; 3],
    r_base: f64,
    r_top: f64,
    segments: usize,
) -> Faceted {
    assert!(segments >= 3, "need at least 3 segments");
    assert!(
        r_base > 0.0 && r_top >= 0.0,
        "radii must be positive (top may be 0)"
    );
    let (e1, e2) = orthonormal_basis(axis);
    let top_center = add(base_center, axis);
    let ring = |center: [f64; 3], r: f64| -> Vec<[f64; 3]> {
        (0..segments)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
                add(center, add(scale(e1, r * t.cos()), scale(e2, r * t.sin())))
            })
            .collect()
    };
    let bottom = ring(base_center, r_base);

    let mut f = Faceted::new();
    let barrel_kind = if r_top == r_base {
        SurfaceKind::Cylinder {
            center: base_center,
            axis,
            radius: r_base,
        }
    } else {
        // Apex where the barrel lines meet.
        let factor = r_base / (r_base - r_top);
        let apex = add(base_center, scale(axis, factor));
        SurfaceKind::Cone {
            apex,
            axis: scale(axis, -factor),
            tan_half_angle: r_base / (norm(axis) * factor).abs(),
        }
    };
    let barrel = f.add_surface(barrel_kind);

    if r_top == 0.0 {
        for i in 0..segments {
            let j = (i + 1) % segments;
            f.push_tri(Tri::new(bottom[i], bottom[j], top_center), barrel);
        }
    } else {
        let top = ring(top_center, r_top);
        for i in 0..segments {
            let j = (i + 1) % segments;
            f.push_tri(Tri::new(bottom[i], bottom[j], top[j]), barrel);
            f.push_tri(Tri::new(bottom[i], top[j], top[i]), barrel);
        }
        // Top cap: ring CCW around +axis matches the outward (+axis) normal.
        let cap = f.add_surface(SurfaceKind::Plane);
        let cap_tris: Vec<Tri> = (0..segments)
            .map(|i| Tri::new(top_center, top[i], top[(i + 1) % segments]))
            .collect();
        f.push_flat(PlanarFacet::new(top.clone()), &cap_tris, cap);
    }
    // Bottom cap: outward normal is -axis, so the boundary runs the bottom
    // ring in reverse (clockwise around +axis), matching the fan winding.
    let cap = f.add_surface(SurfaceKind::Plane);
    let cap_tris: Vec<Tri> = (0..segments)
        .map(|i| Tri::new(base_center, bottom[(i + 1) % segments], bottom[i]))
        .collect();
    let bottom_loop: Vec<[f64; 3]> = bottom.iter().rev().copied().collect();
    f.push_flat(PlanarFacet::new(bottom_loop), &cap_tris, cap);
    f
}

/// Circular cylinder from `base_center` along the full height vector `axis`.
pub fn cylinder(base_center: [f64; 3], axis: [f64; 3], radius: f64, segments: usize) -> Faceted {
    frustum(base_center, axis, radius, radius, segments)
}

/// The azimuthal facet count a curved primitive of `radius` needs so its faceted
/// tessellation tracks the analytic surface at a mesh of target size `maxh`: fine
/// enough that the chord stays under the target (1.5x oversampled, like the edge
/// sampling) AND the sagitta stays under the surface tolerance `tol`. Curved
/// primitives are partly meshed FROM these facets (closed faces, the arrangement
/// of intersection curves), so a fixed count too coarse for the requested `maxh`
/// leaves the boundary off the true surface -- the straddler root cause. The CORE
/// derives this so every embedder (not just the Python layer) gets it.
pub fn facet_count(radius: f64, maxh: Option<f64>, tol: f64) -> usize {
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

/// UV sphere: `segments` longitudes (>= 3), `rings` latitude bands (>= 2).
pub fn sphere(center: [f64; 3], radius: f64, segments: usize, rings: usize) -> Faceted {
    assert!(
        segments >= 3 && rings >= 2,
        "sphere needs >= 3 segments, >= 2 rings"
    );
    assert!(radius > 0.0);
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Sphere { center, radius });
    let pt = |theta: f64, phi: f64| -> [f64; 3] {
        add(
            center,
            [
                radius * theta.sin() * phi.cos(),
                radius * theta.sin() * phi.sin(),
                radius * theta.cos(),
            ],
        )
    };
    let north = add(center, [0.0, 0.0, radius]);
    let south = add(center, [0.0, 0.0, -radius]);
    let row: Vec<Vec<[f64; 3]>> = (1..rings)
        .map(|k| {
            let theta = std::f64::consts::PI * k as f64 / rings as f64;
            (0..segments)
                .map(|i| {
                    pt(
                        theta,
                        2.0 * std::f64::consts::PI * i as f64 / segments as f64,
                    )
                })
                .collect()
        })
        .collect();
    for i in 0..segments {
        let j = (i + 1) % segments;
        f.push_tri(Tri::new(north, row[0][i], row[0][j]), s);
        let last = &row[rings - 2];
        f.push_tri(Tri::new(south, last[j], last[i]), s);
    }
    for k in 0..rings - 2 {
        for i in 0..segments {
            let j = (i + 1) % segments;
            let (a, b) = (row[k][i], row[k][j]);
            let (d, c) = (row[k + 1][i], row[k + 1][j]);
            f.push_tri(Tri::new(a, c, b), s);
            f.push_tri(Tri::new(a, d, c), s);
        }
    }
    f
}

/// Prism: a planar polygon (with holes) in the (base, u, v) frame, extruded
/// along `h`. The frame must be right-handed with respect to the extrusion:
/// (u x v) . h > 0. Surface groups: bottom, top, one side group per ring.
pub fn extrude_polygon(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    base: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
    h: [f64; 3],
) -> Faceted {
    assert!(
        dot3(cross3(u, v), h) > 0.0,
        "extrusion frame must be right-handed: (u x v) . h > 0"
    );
    // Normalize ring orientations: outer counterclockwise, holes clockwise
    // (so that the shared wall formula yields outward normals everywhere).
    let normalize = |pts: &[[f64; 2]], want: Sign| -> Vec<[f64; 2]> {
        let o = polygon_orientation(pts);
        assert_ne!(o, Sign::Zero, "degenerate polygon ring");
        if o == want {
            pts.to_vec()
        } else {
            pts.iter().rev().copied().collect()
        }
    };
    let outer_ccw = normalize(outer, Sign::Positive);
    let holes_cw: Vec<Vec<[f64; 2]>> = holes
        .iter()
        .map(|hole| normalize(hole, Sign::Negative))
        .collect();

    let cap = triangulate_polygon(&outer_ccw, &holes_cw);
    let mut f = Faceted::new();
    let top_base = add(base, h);
    let embed_loop = |loop2: &[[f64; 2]], origin: [f64; 3]| -> Vec<[f64; 3]> {
        loop2.iter().map(|&p| embed(origin, u, v, p)).collect()
    };

    // Bottom cap: counterclockwise in (u, v) faces along +(u x v); the
    // outward bottom normal is the opposite, so reverse the winding (both the
    // fan triangles and the boundary loops).
    let bottom = f.add_surface(SurfaceKind::Plane);
    let bottom_tris: Vec<Tri> = cap
        .iter()
        .map(|t| {
            Tri::new(
                embed(base, u, v, t[0]),
                embed(base, u, v, t[2]),
                embed(base, u, v, t[1]),
            )
        })
        .collect();
    let bottom_outer: Vec<[f64; 3]> = embed_loop(&outer_ccw, base).into_iter().rev().collect();
    let bottom_holes: Vec<Vec<[f64; 3]>> = holes_cw
        .iter()
        .map(|h| embed_loop(h, base).into_iter().rev().collect())
        .collect();
    f.push_flat(
        PlanarFacet::with_holes(bottom_outer, bottom_holes),
        &bottom_tris,
        bottom,
    );

    let top = f.add_surface(SurfaceKind::Plane);
    let top_tris: Vec<Tri> = cap
        .iter()
        .map(|t| {
            Tri::new(
                embed(top_base, u, v, t[0]),
                embed(top_base, u, v, t[1]),
                embed(top_base, u, v, t[2]),
            )
        })
        .collect();
    let top_outer = embed_loop(&outer_ccw, top_base);
    let top_holes: Vec<Vec<[f64; 3]>> = holes_cw.iter().map(|h| embed_loop(h, top_base)).collect();
    f.push_flat(
        PlanarFacet::with_holes(top_outer, top_holes),
        &top_tris,
        top,
    );

    // Walls: region lies left of every (normalized) ring edge, so the quad
    // (a, b, b+h, a+h) faces outward. Each wall is a DISTINCT plane, so each gets
    // its own analytic surface (a shared one would fit a single bogus plane
    // through non-coplanar walls and break the per-face parameter meshing).
    for ring in std::iter::once(&outer_ccw).chain(holes_cw.iter()) {
        for i in 0..ring.len() {
            let j = (i + 1) % ring.len();
            let a = embed(base, u, v, ring[i]);
            let b = embed(base, u, v, ring[j]);
            let (at, bt) = (add(a, h), add(b, h));
            let side = f.add_surface(SurfaceKind::Plane);
            let tris = [Tri::new(a, b, bt), Tri::new(a, bt, at)];
            f.push_flat(PlanarFacet::new(vec![a, b, bt, at]), &tris, side);
        }
    }
    f
}

/// Extrudes an OPEN 2D profile curve along `h` into a closed solid: the swept
/// curve becomes one [`SurfaceKind::Extruded`] wall (the analytic curved face),
/// the chord that closes the open profile becomes a flat wall, and the two
/// cross-section caps are flat. `n_seg` facets sample the curve along its
/// parameter domain. `(u, v, h)` must be right-handed; the profile lives in the
/// `(u, v)` plane. A profile whose endpoints coincide (a closed loop) gets no
/// closing wall (the whole side is curved).
pub fn extrude_spline_profile(
    profile: NurbsCurve,
    n_seg: usize,
    base: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
    h: [f64; 3],
) -> Faceted {
    assert!(
        dot3(cross3(u, v), h) > 0.0,
        "extrusion frame must be right-handed"
    );
    assert!(n_seg >= 2, "need at least 2 segments");
    let (lo, hi) = profile.domain();
    let mut pts2: Vec<[f64; 2]> = (0..=n_seg)
        .map(|i| profile.eval(lo + (hi - lo) * i as f64 / n_seg as f64))
        .collect();
    // The cross-section is the sampled polyline closed end->start. Want CCW so
    // walls face outward; reverse the SAMPLES if the loop is CW (the curve in
    // the surface metadata is unaffected -- projection snaps to nearest point).
    if polygon_orientation(&pts2) == Sign::Negative {
        pts2.reverse();
    }
    let cap = triangulate_polygon(&pts2, &[]);
    let top_base = add(base, h);
    let hl = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
    let axis = [h[0] / hl, h[1] / hl, h[2] / hl];

    let mut f = Faceted::new();
    // Bottom cap: reverse winding for the outward -(uxv) normal.
    let bottom = f.add_surface(SurfaceKind::Plane);
    let bottom_tris: Vec<Tri> = cap
        .iter()
        .map(|t| {
            Tri::new(
                embed(base, u, v, t[0]),
                embed(base, u, v, t[2]),
                embed(base, u, v, t[1]),
            )
        })
        .collect();
    let bottom_loop: Vec<[f64; 3]> = pts2.iter().rev().map(|&p| embed(base, u, v, p)).collect();
    f.push_flat(PlanarFacet::new(bottom_loop), &bottom_tris, bottom);

    let top = f.add_surface(SurfaceKind::Plane);
    let top_tris: Vec<Tri> = cap
        .iter()
        .map(|t| {
            Tri::new(
                embed(top_base, u, v, t[0]),
                embed(top_base, u, v, t[1]),
                embed(top_base, u, v, t[2]),
            )
        })
        .collect();
    let top_loop: Vec<[f64; 3]> = pts2.iter().map(|&p| embed(top_base, u, v, p)).collect();
    f.push_flat(PlanarFacet::new(top_loop), &top_tris, top);

    // Curved wall: the swept profile, one Extruded surface (first-class tris).
    let ext = f.add_surface(SurfaceKind::Extruded {
        profile: Arc::new(profile),
        base,
        udir: u,
        vdir: v,
        axis,
    });
    for i in 0..n_seg {
        let a = embed(base, u, v, pts2[i]);
        let b = embed(base, u, v, pts2[i + 1]);
        let (at, bt) = (add(a, h), add(b, h));
        f.push_tri(Tri::new(a, b, bt), ext);
        f.push_tri(Tri::new(a, bt, at), ext);
    }
    // Closing flat wall (chord end->start) for an open profile.
    let (q0, qn) = (pts2[0], pts2[n_seg]);
    if (q0[0] - qn[0]).hypot(q0[1] - qn[1]) > 1e-12 {
        let flat = f.add_surface(SurfaceKind::Plane);
        let a = embed(base, u, v, qn);
        let b = embed(base, u, v, q0);
        let (at, bt) = (add(a, h), add(b, h));
        let tris = [Tri::new(a, b, bt), Tri::new(a, bt, at)];
        f.push_flat(PlanarFacet::new(vec![a, b, bt, at]), &tris, flat);
    }
    f
}

/// The NACA 0012 airfoil section as a 2D profile curve (chord along +x, span
/// `chord`), an OPEN B-spline from the (blunt) upper trailing edge around the
/// leading edge to the lower trailing edge. Cosine x-spacing concentrates
/// control points at the high-curvature leading edge. The standard 4-digit
/// thickness law with the `-0.1015` term leaves a small finite trailing-edge
/// thickness, so the profile is open (a blunt TE), which charts without a seam.
pub fn naca0012_profile(chord: f64, n_per_side: usize) -> NurbsCurve {
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
    NurbsCurve::interpolate(&pts)
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
    let a_hat = scale(axis, 1.0 / norm(axis));
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Torus {
        center,
        axis,
        major_radius,
        minor_radius,
    });
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
        v6 += dot3(a, cross3(b, c));
    }
    v6 / 6.0
}

/// Solid from an externally supplied triangle soup (an imported STL surface,
/// a marching-cubes iso-surface, etc.). The input must describe a closed,
/// non-self-intersecting surface; the winding is normalized to outward via the
/// signed volume (every triangle flips together when the soup is inward). Each
/// triangle is its own first-class facet on a single shared [`SurfaceKind::Plane`]
/// surface (no `FlatFacet` grouping, so the conformal arrangement treats the
/// many small facets independently rather than as one coplanar face). Fidelity
/// snapping is off, as for any faceted import: the triangles are the surface.
pub fn mesh_solid(verts: &[[f64; 3]], tris: &[[u32; 3]]) -> Faceted {
    assert!(!tris.is_empty(), "mesh_solid needs at least one triangle");
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Plane);
    for t in tris {
        let (a, b, c) = (
            verts[t[0] as usize],
            verts[t[1] as usize],
            verts[t[2] as usize],
        );
        f.push_tri(Tri::new(a, b, c), s);
    }
    let vol = signed_volume(&f);
    assert!(vol.abs() > 0.0, "degenerate mesh_solid (zero volume)");
    if vol < 0.0 {
        for t in &mut f.tris {
            t.v.swap(1, 2);
        }
    }
    f
}

/// Cylinder with an isotropic barrel: instead of [`cylinder`]'s single ring of
/// full-height quads, the side is a structured grid of `rows` height levels so
/// the cells are roughly square (near-equilateral triangles), matching gmsh /
/// tetgen's even surface-point distribution. Flat fan caps (the conformal
/// mesher refines those isotropically anyway). Carries the analytic
/// [`SurfaceKind::Cylinder`] for vertex snapping.
pub fn cylinder_iso(
    base_center: [f64; 3],
    axis: [f64; 3],
    radius: f64,
    segments: usize,
    rows: usize,
) -> Faceted {
    assert!(
        segments >= 3 && rows >= 1,
        "need >= 3 segments and >= 1 row"
    );
    assert!(radius > 0.0);
    let (e1, e2) = orthonormal_basis(axis);
    let ring = |center: [f64; 3]| -> Vec<[f64; 3]> {
        (0..segments)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
                add(
                    center,
                    add(scale(e1, radius * a.cos()), scale(e2, radius * a.sin())),
                )
            })
            .collect()
    };
    // Height levels 0..=rows along the axis.
    let levels: Vec<Vec<[f64; 3]>> = (0..=rows)
        .map(|k| ring(add(base_center, scale(axis, k as f64 / rows as f64))))
        .collect();

    let mut f = Faceted::new();
    let barrel = f.add_surface(SurfaceKind::Cylinder {
        center: base_center,
        axis,
        radius,
    });
    for k in 0..rows {
        let (lo, hi) = (&levels[k], &levels[k + 1]);
        for i in 0..segments {
            let j = (i + 1) % segments;
            // outward winding (matches frustum's barrel)
            f.push_tri(Tri::new(lo[i], lo[j], hi[j]), barrel);
            f.push_tri(Tri::new(lo[i], hi[j], hi[i]), barrel);
        }
    }
    // Top cap: ring CCW around +axis -> outward (+axis) normal.
    let top = &levels[rows];
    let top_center = add(base_center, axis);
    let cap_t = f.add_surface(SurfaceKind::Plane);
    let top_tris: Vec<Tri> = (0..segments)
        .map(|i| Tri::new(top_center, top[i], top[(i + 1) % segments]))
        .collect();
    f.push_flat(PlanarFacet::new(top.clone()), &top_tris, cap_t);
    // Bottom cap: outward normal -axis -> reverse the ring.
    let bot = &levels[0];
    let cap_b = f.add_surface(SurfaceKind::Plane);
    let bot_tris: Vec<Tri> = (0..segments)
        .map(|i| Tri::new(base_center, bot[(i + 1) % segments], bot[i]))
        .collect();
    let bot_loop: Vec<[f64; 3]> = bot.iter().rev().copied().collect();
    f.push_flat(PlanarFacet::new(bot_loop), &bot_tris, cap_b);
    f
}

/// Frustum / cone with an isotropic barrel (the [`frustum`] analog of
/// [`cylinder_iso`]): the side is a structured grid of `rows` height levels
/// with a linear radius taper, so the cells stay roughly square instead of
/// full-height strips. `r_top == 0.0` gives a true cone (the top row collapses
/// to an apex fan, the one unavoidable rotationally-symmetric spot, like
/// gmsh's apex). Flat fan caps. Carries the analytic [`SurfaceKind::Cylinder`]
/// (equal radii) or [`SurfaceKind::Cone`] for vertex snapping.
pub fn frustum_iso(
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
        SurfaceKind::Cylinder {
            center: base_center,
            axis,
            radius: r_base,
        }
    } else {
        let factor = r_base / (r_base - r_top);
        let apex = add(base_center, scale(axis, factor));
        SurfaceKind::Cone {
            apex,
            axis: scale(axis, -factor),
            tan_half_angle: r_base / (norm(axis) * factor).abs(),
        }
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
        let cap_t = f.add_surface(SurfaceKind::Plane);
        let top_tris: Vec<Tri> = (0..segments)
            .map(|i| Tri::new(top_center, top[i], top[(i + 1) % segments]))
            .collect();
        f.push_flat(PlanarFacet::new(top.clone()), &top_tris, cap_t);
    }
    let bot = &levels[0];
    let cap_b = f.add_surface(SurfaceKind::Plane);
    let bot_tris: Vec<Tri> = (0..segments)
        .map(|i| Tri::new(base_center, bot[(i + 1) % segments], bot[i]))
        .collect();
    let bot_loop: Vec<[f64; 3]> = bot.iter().rev().copied().collect();
    f.push_flat(PlanarFacet::new(bot_loop), &bot_tris, cap_b);
    f
}

/// Geodesic sphere (subdivided icosahedron projected onto the analytic
/// sphere). Unlike [`sphere`]'s UV tessellation (latitude rings clustering at
/// the poles, rotationally-symmetric points), the icosphere distributes
/// near-equilateral triangles isotropically over the hull, matching how gmsh
/// and tetgen tessellate a sphere. `subdivisions` controls density: the face
/// count is `20 * 4^subdivisions` (0 -> 20, 2 -> 320, 3 -> 1280). Carries the
/// exact [`SurfaceKind::Sphere`] so mesh vertices still snap onto the true
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
        let n = norm(v);
        add(center, scale(v, radius / n))
    };
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Sphere { center, radius });
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
        let l = norm(d);
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
            let l = norm(s);
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
        let c = dot3(prev_n, cur_n).clamp(-1.0, 1.0);
        let axis = cross3(prev_n, cur_n);
        let s = norm(axis);
        if s < 1e-12 {
            frames.push((u, v));
            continue;
        }
        let k = scale(axis, 1.0 / s);
        let rot = |p: [f64; 3]| -> [f64; 3] {
            // Rodrigues: p c + (k x p) s + k (k . p)(1 - c).
            let kxp = cross3(k, p);
            let kdp = dot3(k, p);
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
    let s = f.add_surface(SurfaceKind::Tube {
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
    let cap0 = f.add_surface(SurfaceKind::Plane);
    let cap0_tris: Vec<Tri> = (0..segments)
        .map(|j| Tri::new(path[0], rings[0][(j + 1) % segments], rings[0][j]))
        .collect();
    let cap0_loop: Vec<[f64; 3]> = rings[0].iter().rev().copied().collect();
    f.push_flat(PlanarFacet::new(cap0_loop), &cap0_tris, cap0);
    let cap1 = f.add_surface(SurfaceKind::Plane);
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
/// to its centroid (convex profiles always are). Output orientation is
/// normalized to outward via the signed volume.
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
    let cap_a = f.add_surface(SurfaceKind::Plane);
    for i in 0..n {
        let j = (i + 1) % n;
        f.push_tri(Tri::new(ca, profile_a[j], profile_a[i]), cap_a);
    }
    let cap_b = f.add_surface(SurfaceKind::Plane);
    for i in 0..n {
        let j = (i + 1) % n;
        f.push_tri(Tri::new(cb, profile_b[i], profile_b[j]), cap_b);
    }
    let side = f.add_surface(SurfaceKind::Plane);
    for i in 0..n {
        let j = (i + 1) % n;
        let (a, b) = (profile_a[i], profile_a[j]);
        let (d, c) = (profile_b[i], profile_b[j]);
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
    let s = f.add_surface(SurfaceKind::Plane);
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
pub const POLYGON_CORNER_DEG: f64 = 20.0;

/// Planar polygon sheet (with holes) in the (base, u, v) frame.
pub fn sheet_polygon(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    base: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
) -> Faceted {
    let mut f = Faceted::new();
    let s = f.add_surface(SurfaceKind::Plane);
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
    let s = f.add_surface(SurfaceKind::Plane);
    let ring: Vec<[f64; 3]> = (0..segments)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
            add(center, add(scale(e1, t.cos()), scale(e2, t.sin())))
        })
        .collect();
    let tris: Vec<Tri> = (0..segments)
        .map(|i| Tri::new(center, ring[i], ring[(i + 1) % segments]))
        .collect();
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
    let s = f.add_surface(SurfaceKind::Nurbs(Arc::new(surface.clone())));
    let area2 = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
        let n = cross3(crate::vec3::sub(b, a), crate::vec3::sub(c, a));
        dot3(n, n)
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

/// Solid of revolution: the closed profile `verts` in the half-plane
/// `(r, z)` of the axis through `origin` along `axis`, edge `i` running from
/// vertex `i` to the next, turned by `angle` radians (a full turn at `TAU`).
/// Every edge is its own surface, with its exact carrier: a straight edge
/// a plane, cylinder or cone, an arc a sphere or torus, a spline a
/// [`SurfaceKind::Revolved`]; an edge on the axis has no faces. A part turn
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
    let len = dot3(axis, axis).sqrt();
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
        let c = cross3(a, pick);
        scale(c, 1.0 / dot3(c, c).sqrt())
    };
    let y = cross3(a, x);
    let size = verts
        .iter()
        .flat_map(|v| verts.iter().map(move |w| (v[0] - w[0]).hypot(v[1] - w[1])))
        .fold(0.0, f64::max);

    // The samples of every edge (its start vertex, not its end) and carrier.
    let mut samples: Vec<Vec<[f64; 2]>> = Vec::with_capacity(n);
    let mut kinds: Vec<Option<SurfaceKind>> = Vec::with_capacity(n);
    let on_axis = |z: f64| add(origin, scale(a, z));
    for (i, e) in edges.iter().enumerate() {
        let (p, q) = (verts[i], verts[(i + 1) % n]);
        if p == q {
            return Err(format!("profile edge {i} has no length"));
        }
        let line = |kinds: &mut Vec<Option<SurfaceKind>>| {
            let kind = if p[0] == 0.0 && q[0] == 0.0 {
                None
            } else if p[1] == q[1] {
                Some(SurfaceKind::Plane)
            } else if p[0] == q[0] {
                Some(SurfaceKind::Cylinder {
                    center: origin,
                    axis: a,
                    radius: p[0],
                })
            } else {
                // The apex where the line meets the axis; the cone opens
                // the way r grows.
                let slope = (q[0] - p[0]) / (q[1] - p[1]);
                Some(SurfaceKind::Cone {
                    apex: on_axis(p[1] - p[0] / slope),
                    axis: scale(a, slope.signum()),
                    tan_half_angle: slope.abs(),
                })
            };
            kinds.push(kind);
        };
        match e {
            ProfileEdge::Line => {
                line(&mut kinds);
                samples.push(vec![p]);
            }
            ProfileEdge::Arc(b) if *b == 0.0 => {
                line(&mut kinds);
                samples.push(vec![p]);
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
                // A circle about a point of the axis turns into a sphere;
                // off the axis into a torus (the same about -r as about r).
                kinds.push(Some(if c[0].abs() <= 1e-12 * size {
                    SurfaceKind::Sphere {
                        center: on_axis(c[1]),
                        radius: rho,
                    }
                } else {
                    SurfaceKind::Torus {
                        center: on_axis(c[1]),
                        axis: a,
                        major_radius: c[0].abs(),
                        minor_radius: rho,
                    }
                }));
            }
            ProfileEdge::Spline(inner) => {
                if inner.len() < 2 {
                    return Err(format!(
                        "profile edge {i}: a spline needs at least 2 interior points"
                    ));
                }
                if inner.iter().any(|v| !(v[0] >= 0.0)) {
                    return Err("a revolve profile must stay in r >= 0".into());
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
                kinds.push(Some(SurfaceKind::Revolved {
                    profile: Arc::new(curve),
                    origin,
                    axis: a,
                    x,
                }));
            }
        }
    }
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
    let full = angle >= TAU;
    let turns = ((facet_count(rmax, maxh, tol).max(min_segments) as f64 * angle / TAU).ceil()
        as usize)
        .max(if full { 3 } else { 1 });
    let point = |v: [f64; 2], k: usize| -> [f64; 3] {
        if v[0] == 0.0 {
            return on_axis(v[1]);
        }
        let k = if full { k % turns } else { k };
        let t = angle * k as f64 / turns as f64;
        let dir = add(scale(x, t.cos()), scale(y, t.sin()));
        add(on_axis(v[1]), scale(dir, v[0]))
    };
    let area2 = |t: &[[f64; 3]; 3]| {
        let c = cross3(crate::vec3::sub(t[1], t[0]), crate::vec3::sub(t[2], t[0]));
        dot3(c, c)
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
            Some(SurfaceKind::Revolved {
                profile,
                origin,
                axis,
                x,
            }) if !ccw => Some(SurfaceKind::Revolved {
                profile: Arc::new(profile.reversed()),
                origin,
                axis,
                x,
            }),
            k => k,
        };
        // An edge on the axis keeps its surface slot (roles follow the
        // edges) but has no faces.
        let s = f.add_surface(kind.clone().unwrap_or(SurfaceKind::Plane));
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
            let s = f.add_surface(SurfaceKind::Plane);
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
