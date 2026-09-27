//! Exact triangulation of planar polygons, including holes.
//!
//! Reuses the csg machinery: the polygon edges become constraints of one big
//! container facet, the constrained triangulation subdivides it, and each
//! sub-triangle is kept iff its exact barycenter has odd crossing parity with
//! the polygon edges (even-odd rule, which handles holes for free). All
//! decisions are exact predicates; for simple non-crossing input every output
//! vertex is an input vertex.

use rapidmesh_csg::{triangulate_facet, Constraint, ConstraintLine, Tri};
use rapidmesh_exact::{orient2d, Axis, Expansion, Point3, Sign};

/// Exact orientation of a simple planar polygon (shoelace sign, evaluated in
/// expansion arithmetic): `Positive` = counterclockwise.
pub fn polygon_orientation(pts: &[[f64; 2]]) -> Sign {
    let mut acc = Expansion::from_f64(0.0);
    for i in 0..pts.len() {
        let j = (i + 1) % pts.len();
        let cross = Expansion::from_f64(pts[i][0])
            .mul(&Expansion::from_f64(pts[j][1]))
            .add(
                &Expansion::from_f64(pts[j][0])
                    .mul(&Expansion::from_f64(pts[i][1]))
                    .neg(),
            );
        acc = acc.add(&cross);
    }
    acc.sign()
}

/// Even-odd test for a (possibly implicit) point in the z = 0 plane against
/// a set of polygon edges, by exact segment crossing parity.
fn point_in_edges_z0(p: &Point3, edges: &[([f64; 3], [f64; 3])], hi: [f64; 2]) -> bool {
    'targets: for k in 0..32u64 {
        let mut s = (k + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut frac = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let q = Point3::explicit(hi[0] + 1.0 + frac(), hi[1] + 1.0 + frac(), 0.0);
        let mut crossings = 0usize;
        for (a, b) in edges {
            let pa = Point3::Explicit(*a);
            let pb = Point3::Explicit(*b);
            let sp = orient2d(&pa, &pb, p, Axis::Z).expect("valid");
            if sp == Sign::Zero {
                // p exactly on the edge line: the segment (p, q) meets this
                // edge's line only at p, and p is never on an open edge (it
                // is a sub-triangle barycenter): no crossing.
                continue;
            }
            let sq = orient2d(&pa, &pb, &q, Axis::Z).expect("valid");
            let sa = orient2d(p, &q, &pa, Axis::Z).expect("valid");
            let sb = orient2d(p, &q, &pb, Axis::Z).expect("valid");
            if sq == Sign::Zero || sa == Sign::Zero || sb == Sign::Zero {
                continue 'targets;
            }
            if sp != sq && sa != sb {
                crossings += 1;
            }
        }
        return crossings % 2 == 1;
    }
    panic!("no generic parity target found in 32 attempts");
}

/// Exact triangulation of a planar polygon with holes, in local 2D
/// coordinates.
///
/// `outer` and each hole must be simple polygons (no self-crossings); holes
/// must lie inside the outer boundary and not cross each other (touching at
/// points/edges is fine — classification is by even-odd parity). Returns the
/// input-coordinate triangles, counterclockwise.
pub fn triangulate_polygon(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>]) -> Vec<[[f64; 2]; 3]> {
    assert!(outer.len() >= 3, "polygon needs at least 3 vertices");

    // Bounding box and container facet (comfortably larger than the input).
    let mut lo = outer[0];
    let mut hi = outer[0];
    for p in outer.iter().chain(holes.iter().flatten()) {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let d = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0);
    let container = Tri::new(
        [lo[0] - d, lo[1] - d, 0.0],
        [hi[0] + 3.0 * d, lo[1] - d, 0.0],
        [lo[0] - d, hi[1] + 3.0 * d, 0.0],
    );

    // All boundary edges (outer + holes) as constraints.
    let mut edges: Vec<([f64; 3], [f64; 3])> = Vec::new();
    let mut ring = |pts: &[[f64; 2]]| {
        for i in 0..pts.len() {
            let j = (i + 1) % pts.len();
            edges.push(([pts[i][0], pts[i][1], 0.0], [pts[j][0], pts[j][1], 0.0]));
        }
    };
    ring(outer);
    for h in holes {
        ring(h);
    }
    let constraints: Vec<Constraint> = edges
        .iter()
        .map(|&(u, v)| Constraint {
            a: Point3::Explicit(u),
            b: Point3::Explicit(v),
            line: ConstraintLine::Edge(u, v),
        })
        .collect();

    let ft = triangulate_facet(&container, &[], &constraints);

    // Keep sub-triangles whose barycenter is inside by even-odd parity.
    let mut out = Vec::new();
    for t in &ft.triangles {
        let (p0, p1, p2) = (&ft.vertices[t[0]], &ft.vertices[t[1]], &ft.vertices[t[2]]);
        let bary = Point3::bary(p0.clone(), p1.clone(), p2.clone());
        if !point_in_edges_z0(&bary, &edges, hi) {
            continue;
        }
        let tri: [[f64; 2]; 3] = std::array::from_fn(|k| {
            let v = ft.vertices[t[k]]
                .as_explicit()
                .expect("simple polygon input produces only explicit vertices");
            [v[0], v[1]]
        });
        out.push(tri);
    }
    // The container is counterclockwise in the z-projection, so kept
    // sub-triangles are too.
    debug_assert_eq!(ft.orientation, Sign::Positive);
    out
}
