//! 2D Delaunay + Lloyd relaxation for the surface (2D) stage of the hierarchy.
//!
//! Built on the EXACT robust predicates (`orient2d`/`incircle2d` from
//! rapidmesh-exact, the Shewchuk-style kernel): all triangulation decisions are
//! exact, so the relaxation is robust on degenerate (cocircular/collinear) grid
//! inputs. Only the centroid WEIGHTS are float (a non-decision quantity).
//!
//! Used by the surface stage: each planar patch is filled by scattering interior
//! points and relaxing them with the patch boundary (1D edge points) held fixed.
//! The triangulation here serves the relaxation; the conforming surface is read
//! back from the 3D mesh downstream.

use rapidmesh_exact::Sign;
use rapidmesh_geom::cdt2::orient;
pub use rapidmesh_geom::cdt2::{delaunay2, triangulate_constrained, Cdt};

type P2 = [f64; 2];

fn dist2(a: P2, b: P2) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

/// A triangle whose circumradius exceeds this multiple of the target is split
/// for size. The target is the typical edge length, as a mesh size is in gmsh:
/// an equilateral triangle splits from an edge of 1.39 targets on, so the
/// split halves land around it and the median edge comes out near the target
/// (at 0.6, the bound of an edge of one target, it came out at 0.76).
const SIZE_SPLIT_RADIUS: f64 = 0.8;

/// The cosine of the smallest interior angle of triangle (a, b, c): the
/// largest of the three cosines (an angle is below `deg` exactly where its
/// cosine is above `cos(deg)`, with no `acos` per angle).
fn smallest_angle_cos(a: P2, b: P2, c: P2) -> f64 {
    let cos = |u: P2, v: P2, w: P2| {
        let (e1, e2) = ([v[0] - u[0], v[1] - u[1]], [w[0] - u[0], w[1] - u[1]]);
        let n = (e1[0] * e1[0] + e1[1] * e1[1]).sqrt() * (e2[0] * e2[0] + e2[1] * e2[1]).sqrt();
        ((e1[0] * e2[0] + e1[1] * e2[1]) / (n + 1e-30)).clamp(-1.0, 1.0)
    };
    cos(a, b, c).max(cos(b, c, a)).max(cos(c, a, b))
}

/// Circumcenter of (a, b, c); `None` if (near-)degenerate.
fn circumcenter(a: P2, b: P2, c: P2) -> Option<P2> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-30 {
        return None;
    }
    let (a2, b2, c2) = (
        a[0] * a[0] + a[1] * a[1],
        b[0] * b[0] + b[1] * b[1],
        c[0] * c[0] + c[1] * c[1],
    );
    Some([
        (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
        (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d,
    ])
}

/// Sizing-field-driven Delaunay (Ruppert/Chew) refinement of a constrained
/// triangulation: repeatedly insert the circumcentre of any triangle whose
/// minimum angle is below `min_angle_deg` OR whose circumradius exceeds the local
/// `target` size (so the result is graded to the field), the interior gaining a
/// guaranteed angle bound -- no slivers. Interior points are appended to `interior`; the boundary segments are
/// protected (a circumcentre that encroaches one is not inserted).
#[allow(clippy::too_many_arguments)]
fn refine_quality(
    boundary: &mut [P2],
    segments: &[(usize, usize)],
    interior: &mut Vec<P2>,
    target: impl Fn(P2) -> f64,
    inside: impl Fn(P2) -> bool,
    min_angle_deg: f64,
    // Hard cap on refinement passes. Each pass re-triangulates the whole patch,
    // so this bounds the cost: the surface stage runs the full set, a display
    // mesh only a few (the boundary conformity + angle bound are met early; the
    // rest is marginal interior density).
    max_passes: usize,
    // Whether the final smoothing may slide a boundary point along a straight
    // stretch of the outline; not where the outline's points are shared with
    // other meshes and must stay where they are.
    slide: impl Fn(P2) -> bool,
) {
    let diam2 = |b: &[P2], u: usize, v: usize| 0.25 * dist2(b[u], b[v]);
    let mid = |b: &[P2], u: usize, v: usize| [0.5 * (b[u][0] + b[v][0]), 0.5 * (b[u][1] + b[v][1])];
    let cos_min = min_angle_deg.to_radians().cos();
    let mut good: rustc_hash::FxHashSet<[[u64; 2]; 3]> = rustc_hash::FxHashSet::default();
    for _ in 0..max_passes {
        let mut all = boundary.to_vec();
        all.extend_from_slice(interior);
        let tris = triangulate_constrained(&all, segments, &inside);
        let mut inserts: Vec<P2> = Vec::new();

        // Bad triangles: below the angle bound, or larger than the field.
        {
            let mut cand: Vec<P2> = Vec::new();
            // Segment lookup grid for the circumcentre encroachment test below:
            // every segment registers the cells its DIAMETRAL DISK overlaps, so a
            // candidate reads exactly one cell and tests only nearby segments --
            // the per-candidate all-segments scan was O(triangles * segments) per
            // pass, the refinement's remaining quadratic hot spot.
            let seg_cell = {
                let mean: f64 = segments
                    .iter()
                    .map(|&(u, v)| dist2(boundary[u], boundary[v]).sqrt())
                    .sum::<f64>()
                    / segments.len().max(1) as f64;
                mean.max(1e-12)
            };
            let skey = |p: P2| {
                (
                    (p[0] / seg_cell).floor() as i64,
                    (p[1] / seg_cell).floor() as i64,
                )
            };
            let mut seg_grid: rustc_hash::FxHashMap<(i64, i64), Vec<u32>> =
                rustc_hash::FxHashMap::default();
            for (si, &(u, v)) in segments.iter().enumerate() {
                let m = mid(boundary, u, v);
                let r = diam2(boundary, u, v).sqrt();
                let (lo, hi) = (skey([m[0] - r, m[1] - r]), skey([m[0] + r, m[1] + r]));
                for cx in lo.0..=hi.0 {
                    for cy in lo.1..=hi.1 {
                        seg_grid.entry((cx, cy)).or_default().push(si as u32);
                    }
                }
            }
            for t in &tris {
                let p = [all[t[0]], all[t[1]], all[t[2]]];
                // A triangle found good in an earlier pass is good still: its
                // angles and its size at its circumcentre are its own, and the
                // budget only ever runs out.
                let key = {
                    let mut k = p.map(|q| [q[0].to_bits(), q[1].to_bits()]);
                    k.sort_unstable();
                    k
                };
                if good.contains(&key) {
                    continue;
                }
                let cc = match circumcenter(p[0], p[1], p[2]) {
                    Some(c) => c,
                    None => continue,
                };
                let tg = target(cc).max(1e-12);
                let ratio = dist2(p[0], cc) / (tg * tg);
                let angle_bad = smallest_angle_cos(p[0], p[1], p[2]) > cos_min;
                let size_bad = ratio > SIZE_SPLIT_RADIUS * SIZE_SPLIT_RADIUS;
                if !(angle_bad || size_bad) {
                    good.insert(key);
                    continue;
                }
                // Encroachment: a circumcentre inside a boundary segment's
                // diametral circle is dropped -- jamming a point against a
                // fixed edge is exactly what seeds the boundary slivers, so we
                // leave the (mildly bad) triangle rather than make it worse. A
                // disk containing `cc` registered `cc`'s cell, so one lookup is
                // exact.
                let encroaches = seg_grid.get(&skey(cc)).is_some_and(|list| {
                    list.iter().any(|&si| {
                        let (u, v) = segments[si as usize];
                        dist2(cc, mid(boundary, u, v)) < diam2(boundary, u, v)
                    })
                });
                if !encroaches && inside(cc) {
                    cand.push(cc);
                }
            }
            inserts.extend(cand);
        }
        if inserts.is_empty() {
            break;
        }
        // Spacing guard via a uniform hash grid: reject an insert within
        // 0.5*target of an existing vertex. O(1) per insert instead of
        // O(points) (the loop's main quadratic cost besides the rebuild). It is
        // approximate at strong gradients -- a missed neighbour only yields a
        // slightly denser spot, never a quality violation.
        if !inserts.is_empty() {
            let gc = inserts
                .iter()
                .map(|&c| 0.5 * target(c))
                .fold(f64::INFINITY, f64::min)
                .max(1e-9);
            let key = |p: P2| ((p[0] / gc).floor() as i64, (p[1] / gc).floor() as i64);
            let mut grid: rustc_hash::FxHashMap<(i64, i64), Vec<P2>> =
                rustc_hash::FxHashMap::default();
            for &q in boundary.iter().chain(interior.iter()) {
                grid.entry(key(q)).or_default().push(q);
            }
            for c in inserts {
                let r = 0.5 * target(c);
                let r2 = r * r;
                let (cx, cy) = key(c);
                let rc = ((r / gc).ceil() as i64).min(6);
                let mut ok = true;
                'scan: for dx in -rc..=rc {
                    for dy in -rc..=rc {
                        if let Some(v) = grid.get(&(cx + dx, cy + dy)) {
                            if v.iter().any(|&q| dist2(c, q) < r2) {
                                ok = false;
                                break 'scan;
                            }
                        }
                    }
                }
                if ok {
                    grid.entry(key(c)).or_default().push(c);
                    interior.push(c);
                }
            }
        }
    }
    // The Ruppert refinement met the angle bound but left the elements uneven;
    // relax the whole mesh (interior freely, the outline sliding) to convergence for
    // near-equilateral, gmsh-grade shape.
    smooth_mesh(
        boundary,
        segments,
        interior,
        &inside,
        &target,
        min_angle_deg,
        12,
        slide,
    );
}

/// Holistic ODT mesh optimisation (SOTA element quality): after Ruppert meets the
/// angle bound, relax EVERY movable vertex toward its area-weighted optimal-Delaunay
/// target until convergence. Interior vertices move freely; a BOUNDARY vertex whose
/// two outline segments are collinear (an edge-interior point) SLIDES along that
/// edge, while corners (a segment-angle vertex) stay pinned — so the conductor
/// outline is preserved exactly, yet the edge triangles even out to near-equilateral
/// the way a full-mesh smoother (gmsh) does, instead of staying skewed against a
/// frozen boundary. Gauss-Seidel sweep (each move sees the latest positions),
/// per-move guarded: a move is applied only if every incident triangle stays
/// in-domain, un-flipped, and at/above the angle bound — the no-sliver guarantee and
/// the triangle count are preserved. Stops early once the largest move is below a
/// thousandth of the local edge length.
fn smooth_mesh(
    boundary: &mut [P2],
    segments: &[(usize, usize)],
    interior: &mut [P2],
    inside: impl Fn(P2) -> bool,
    target: impl Fn(P2) -> f64,
    min_angle_deg: f64,
    max_iters: usize,
    slide: impl Fn(P2) -> bool,
) {
    let nb = boundary.len();
    if nb == 0 {
        return;
    }
    let sarea2 =
        |a: P2, b: P2, c: P2| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);

    // Contour adjacency of each boundary vertex (its two neighbours along the resampled
    // outline) -> a slide tangent if the outline is locally straight there, else pinned.
    let mut nbr = vec![[usize::MAX; 2]; nb];
    // Constraint degree: a vertex where a constraint CHAIN meets the outline (or two chains
    // meet) has three or more constraint neighbours and stays pinned — sliding it along the
    // outline would tilt the chain off the line it must hold.
    let mut deg = vec![0usize; nb];
    for &(u, v) in segments {
        if u < nb && v < nb {
            for (a, b) in [(u, v), (v, u)] {
                deg[a] += 1;
                if nbr[a][0] == usize::MAX {
                    nbr[a][0] = b;
                } else if nbr[a][1] == usize::MAX && nbr[a][0] != b {
                    nbr[a][1] = b;
                }
            }
        }
    }
    let tangent: Vec<Option<[f64; 2]>> = (0..nb)
        .map(|i| {
            let [n0, n1] = nbr[i];
            if n0 == usize::MAX || n1 == usize::MAX || deg[i] != 2 || !slide(boundary[i]) {
                return None;
            }
            let (a, p, b) = (boundary[n0], boundary[i], boundary[n1]);
            let d0 = [p[0] - a[0], p[1] - a[1]];
            let d1 = [b[0] - p[0], b[1] - p[1]];
            let l0 = (d0[0] * d0[0] + d0[1] * d0[1]).sqrt();
            let l1 = (d1[0] * d1[0] + d1[1] * d1[1]).sqrt();
            if l0 < 1e-300 || l1 < 1e-300 {
                return None;
            }
            // |sin∠| between the two outline segments: ~0 ⇒ edge-interior ⇒ slide; else corner.
            let cross = (d0[0] * d1[1] - d0[1] * d1[0]).abs() / (l0 * l1);
            if cross > 1e-3 {
                return None;
            }
            let t = [b[0] - a[0], b[1] - a[1]];
            let lt = (t[0] * t[0] + t[1] * t[1]).sqrt();
            Some([t[0] / lt, t[1] / lt])
        })
        .collect();

    // ONE live constrained triangulation for the whole smoothing: guarded
    // point moves keep every incident triangle exactly CCW, and a local
    // Lawson pass (`restore`) repairs Delaunayness after each sweep -- the
    // full re-triangulation per iteration (constraint forcing included) was
    // the smoother's dominant, superlinear cost.
    let mut all: Vec<P2> = boundary.to_vec();
    all.extend_from_slice(interior);
    let (mut cdt, constraints) = Cdt::new_constrained(&all, segments);
    let mut star: Vec<usize> = Vec::new();

    const RELAX: f64 = 0.9;
    for _ in 0..max_iters {
        let tris = cdt.kept_triangles(&inside);
        // The target at every point, once a sweep (a triangle weighs by the
        // mean of its corners'): the field is smooth and the moves short.
        let hv: Vec<f64> = (0..all.len()).map(|i| target(cdt.point(i))).collect();
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); all.len()];
        for (ti, t) in tris.iter().enumerate() {
            for &i in t {
                incident[i].push(ti);
            }
        }
        let mut max_rel = 0.0f64;
        // Gauss-Seidel: each vertex relaxes against the latest positions of its
        // neighbours (read live from the triangulation).
        for i in 0..all.len() {
            let tang = if i < nb {
                match tangent[i] {
                    Some(t) => Some(t),
                    None => continue, // pinned corner / open contour
                }
            } else {
                None
            };
            if incident[i].is_empty() {
                continue;
            }
            // ODT optimum x* = (Σ|T|·(sum of the OTHER two verts)) / (2 Σ|T|), plus the
            // local edge scale for the convergence test.
            let (mut num, mut den, mut hmin) = ([0.0f64; 2], 0.0f64, f64::INFINITY);
            for &ti in &incident[i] {
                let t = tris[ti];
                let (a, b, c) = (cdt.point(t[0]), cdt.point(t[1]), cdt.point(t[2]));
                // DENSITY-weighted ODT: weight by area · ρ, ρ = 1/h² from the sizing field at the
                // triangle centroid. Plain area weighting equidistributes to UNIFORM size and so
                // fights a graded field at the fine↔coarse transitions (skewed elements); weighting
                // by the target density makes the optimum follow the field → smooth, well-shaped
                // graded transitions. ρ is a smooth scalar, so the no-flip / angle guard still holds.
                let h = ((hv[t[0]] + hv[t[1]] + hv[t[2]]) / 3.0).max(1e-12);
                let w = 0.5 * sarea2(a, b, c).abs() / (h * h);
                num[0] += w * (a[0] + b[0] + c[0] - cdt.point(i)[0]);
                num[1] += w * (a[1] + b[1] + c[1] - cdt.point(i)[1]);
                den += w;
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    hmin = hmin.min((u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2));
                }
            }
            if den <= 0.0 {
                continue;
            }
            let cur = cdt.point(i);
            let opt = [num[0] / (2.0 * den), num[1] / (2.0 * den)];
            let mut delta = [opt[0] - cur[0], opt[1] - cur[1]];
            if let Some(t) = tang {
                let dp = delta[0] * t[0] + delta[1] * t[1]; // keep only the along-edge component
                delta = [dp * t[0], dp * t[1]];
            }
            let cand = [cur[0] + RELAX * delta[0], cur[1] + RELAX * delta[1]];
            if i >= nb && !inside(cand) {
                continue;
            }
            // Quality gate on the KEPT triangles (as before), plus the exact
            // CCW guard over the FULL star (exterior/hole triangles included):
            // the live structure's flip invariants need every alive triangle
            // to stay positively oriented, not just the in-domain ones.
            let bound = min_angle_deg.to_radians().cos();
            let worst = |at: P2| {
                incident[i]
                    .iter()
                    .map(|&ti| {
                        let t = tris[ti];
                        let q: [P2; 3] =
                            std::array::from_fn(|j| if t[j] == i { at } else { cdt.point(t[j]) });
                        smallest_angle_cos(q[0], q[1], q[2])
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            let ok = worst(cand) <= bound && {
                cdt.star(i, &mut star)
                    && star.iter().all(|&ti| {
                        let t = cdt.triangle(ti);
                        let q: [P2; 3] =
                            std::array::from_fn(|j| if t[j] == i { cand } else { cdt.point(t[j]) });
                        orient(q[0], q[1], q[2]) == Sign::Positive
                    })
            };
            if ok {
                let mv2 = (cand[0] - cur[0]).powi(2) + (cand[1] - cur[1]).powi(2);
                if hmin.is_finite() && hmin > 0.0 {
                    max_rel = max_rel.max(mv2 / hmin);
                }
                cdt.set_point(i, cand);
                all[i] = cand;
            }
        }
        boundary.copy_from_slice(&all[..nb]);
        interior.copy_from_slice(&all[nb..]);
        if max_rel < 1e-4 {
            break; // converged: largest move < 1 % of the local edge length
        }
        // Local Delaunay repair (flips only) before the next sweep.
        cdt.restore(&constraints);
    }
}

/// Fills a planar region with Lloyd-relaxed interior points at a GRADED local
/// `target` spacing (`target(q)` is the desired edge length at `q`). `step` is
/// the finest target on the patch, the grid step of the initial scatter; the
/// per-point separation is the LOCAL `0.5 * target`, so the density grades:
/// dense where `target` is small, sparse where it is large. `boundary` is the
/// set of FIXED boundary points (graded 1D edge points and corners); `inside`
/// decides patch membership (exact, supplied by the caller). Interior points are
/// scattered on a grid in `[lo, hi]`, kept inside and clear of the boundary by
/// the local radius, then moved toward the area-weighted centroid of their
/// incident triangles with a local separation guard (no collapse / sliver seed).
#[allow(clippy::too_many_arguments)]
pub fn cvt_fill(
    boundary: &[P2],
    lo: P2,
    hi: P2,
    step: f64,
    target: impl Fn(P2) -> f64,
    iters: usize,
    inside: impl Fn(P2) -> bool,
    density: bool,
) -> Vec<P2> {
    if !(step.is_finite() && step > 0.0) {
        return Vec::new();
    }
    let nb = boundary.len();
    let nx = (((hi[0] - lo[0]) / step).ceil() as usize).max(1);
    let ny = (((hi[1] - lo[1]) / step).ceil() as usize).max(1);
    // Greedy graded scatter: keep a grid node only if it clears the boundary and
    // every already-kept interior point by its OWN local radius. The nodes are
    // those of the grid at `step`, thinned by a quadtree where the target is
    // coarser (a block of 2^k nodes a side whose size stays under the local
    // radius, half the target, offers its first node only), so a fine spot in
    // a large patch costs its own nodes, not the patch's; they are taken in
    // grid order.
    let gc = (0.5 * step).max(1e-9);
    let extent = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    let mut bgrid = LevelGrid::new(gc, extent);
    for &b in boundary {
        bgrid.insert(b);
    }
    let node = |i: usize, j: usize| [lo[0] + i as f64 * step, lo[1] + j as f64 * step];
    // The nodes offered, with the target of the block that offers each.
    let mut nodes: Vec<(usize, usize, f64)> = Vec::new();
    let mut side = 1usize;
    while side < nx.max(ny) {
        side *= 2;
    }
    let mut stack = vec![(0usize, 0usize, side)];
    while let Some((i0, j0, k)) = stack.pop() {
        if i0 >= nx || j0 >= ny {
            continue;
        }
        let half = k / 2;
        let centre = node(i0 + half, j0 + half);
        let t = target(centre);
        if k > 1 && k as f64 * step > 0.5 * t {
            for (di, dj) in [(0, 0), (half, 0), (0, half), (half, half)] {
                stack.push((i0 + di, j0 + dj, half));
            }
        } else if i0 >= 1 && j0 >= 1 {
            nodes.push((i0, j0, t));
        }
    }
    nodes.sort_unstable_by_key(|a| (a.0, a.1));
    let mut interior: Vec<P2> = Vec::new();
    let mut igrid = LevelGrid::new(gc, extent);
    for (i, j, t) in nodes {
        let q = node(i, j);
        if !inside(q) {
            continue;
        }
        let r2 = (0.5 * t).powi(2);
        if bgrid.clear(q, r2, None) && igrid.clear(q, r2, None) {
            igrid.insert(q);
            interior.push(q);
        }
    }

    // The target at every point, the boundary's once, the interior's each
    // pass: a triangle weighs by the mean of its corners', a point moves
    // clear of the others by its own.
    let hb: Vec<f64> = if density {
        boundary.iter().map(|&b| target(b)).collect()
    } else {
        Vec::new()
    };
    for _ in 0..iters {
        if interior.is_empty() {
            break;
        }
        let mut all: Vec<P2> = boundary.to_vec();
        all.extend_from_slice(&interior);
        let hi: Vec<f64> = interior.iter().map(|&q| target(q)).collect();
        let h_at = |v: usize| if v < nb { hb[v] } else { hi[v - nb] };
        let tris = delaunay2(&all);
        let mut num = vec![[0.0f64; 2]; all.len()];
        let mut den = vec![0.0f64; all.len()];
        for t in &tris {
            let p = [all[t[0]], all[t[1]], all[t[2]]];
            // Float area as a relaxation WEIGHT (not a decision).
            let area = 0.5
                * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                    - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))
                    .abs();
            let c = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            ];
            // DENSITY-WEIGHTED CVT (2D, adaptive mode): weight by area * rho,
            // rho = 1/target^2 (spacing ~ target), so a graded surface field
            // relaxes into a smooth gradient. Gated: it shifts the surface point
            // distribution, which the exact-volume fixtures (1e-9) cannot absorb;
            // plain area weighting is the uniform CVT.
            let w = if density {
                let h = ((h_at(t[0]) + h_at(t[1]) + h_at(t[2])) / 3.0).max(1e-12);
                area / (h * h)
            } else {
                area
            };
            for &v in t {
                num[v][0] += w * c[0];
                num[v][1] += w * c[1];
                den[v] += w;
            }
        }
        // Rebuild the interior hash for this pass, then keep it live as points
        // move (so later moves see earlier ones, as the O(n) scan did).
        igrid = LevelGrid::new(gc, extent);
        let mut id: Vec<u32> = interior.iter().map(|&p| igrid.insert(p)).collect();
        for k in 0..interior.len() {
            let v = nb + k;
            if den[v] == 0.0 {
                continue;
            }
            let tgt = [num[v][0] / den[v], num[v][1] / den[v]];
            if !inside(tgt) {
                continue;
            }
            let r2 = (0.5 * hi[k]).powi(2);
            if bgrid.clear(tgt, r2, None) && igrid.clear(tgt, r2, Some(id[k])) {
                igrid.remove(id[k]);
                id[k] = igrid.insert(tgt);
                interior[k] = tgt;
            }
        }
    }
    interior
}

/// Points in hash grids of cells `gc`, `2 gc`, `4 gc`, ...: a clearance
/// test looks in the grid whose cells match its radius, so a large radius
/// among fine points scans a few cells, not the square of its reach in the
/// finest.
struct LevelGrid {
    gc: f64,
    /// Every point inserted, and whether it is still in.
    pts: Vec<P2>,
    alive: Vec<bool>,
    levels: Vec<rustc_hash::FxHashMap<(i64, i64), Vec<u32>>>,
}

impl LevelGrid {
    /// Grids from cells `gc` up to cells as large as `extent`.
    fn new(gc: f64, extent: f64) -> LevelGrid {
        let mut n = 1;
        while n < 40 && gc * ((1u64 << (n - 1)) as f64) < extent {
            n += 1;
        }
        LevelGrid {
            gc,
            pts: Vec::new(),
            alive: Vec::new(),
            levels: (0..n).map(|_| rustc_hash::FxHashMap::default()).collect(),
        }
    }

    fn key(&self, p: P2, level: usize) -> (i64, i64) {
        let c = self.gc * (1u64 << level) as f64;
        ((p[0] / c).floor() as i64, (p[1] / c).floor() as i64)
    }

    /// Adds `p`; returns its id.
    fn insert(&mut self, p: P2) -> u32 {
        let id = self.pts.len() as u32;
        self.pts.push(p);
        self.alive.push(true);
        for l in 0..self.levels.len() {
            let k = self.key(p, l);
            self.levels[l].entry(k).or_default().push(id);
        }
        id
    }

    /// Takes the point `id` out (it stays in the cells, skipped).
    fn remove(&mut self, id: u32) {
        self.alive[id as usize] = false;
    }

    /// True if `q` is at least its own radius (`sqrt(r2)`) from every point,
    /// optionally skipping the point `skip` (the query's own during
    /// relaxation). Exact: no point within range is missed.
    fn clear(&self, q: P2, r2: f64, skip: Option<u32>) -> bool {
        let r = r2.sqrt();
        let mut level = 0;
        while level + 1 < self.levels.len() && self.gc * ((1u64 << level) as f64) < r {
            level += 1;
        }
        let c = self.gc * (1u64 << level) as f64;
        let rc = (r / c).ceil() as i64 + 1;
        let (cx, cy) = self.key(q, level);
        for dx in -rc..=rc {
            for dy in -rc..=rc {
                if let Some(v) = self.levels[level].get(&(cx + dx, cy + dy)) {
                    for &i in v {
                        if Some(i) == skip || !self.alive[i as usize] {
                            continue;
                        }
                        if dist2(q, self.pts[i as usize]) < r2 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

/// Even-odd point-in-contours test over ROW-BUCKETED loop edges. Same
/// crossing rule and per-edge arithmetic as the classic even-odd scan (so the
/// answers are bit-identical), but a query only touches the edges whose
/// y-interval intersects its row -- the linear all-loops scan per `inside`
/// query was the hot spot on large faces (every triangle filter, CVT
/// candidate and smoothing gate pays one query).
pub(crate) struct PipRows {
    y0: f64,
    cell: f64,
    rows: Vec<Vec<(P2, P2)>>,
}

impl PipRows {
    pub(crate) fn build(loops: &[Vec<P2>]) -> PipRows {
        let (mut ylo, mut yhi, mut len_sum, mut n_edges) =
            (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize);
        for lp in loops {
            let n = lp.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let (a, b) = (lp[i], lp[(i + 1) % n]);
                ylo = ylo.min(a[1]);
                yhi = yhi.max(a[1]);
                len_sum += dist2(a, b).sqrt();
                n_edges += 1;
            }
        }
        if n_edges == 0 || !(yhi > ylo) {
            return PipRows {
                y0: 0.0,
                cell: 1.0,
                rows: Vec::new(),
            };
        }
        // Row height ~ the mean edge length: short (h-sized) boundary edges land
        // in one or two rows; the row count stays bounded for huge canvases.
        let cell = (len_sum / n_edges as f64)
            .max((yhi - ylo) / 4096.0)
            .max(1e-12);
        let nrows = (((yhi - ylo) / cell).ceil() as usize + 1).max(1);
        let mut rows: Vec<Vec<(P2, P2)>> = vec![Vec::new(); nrows];
        let row_of =
            |y: f64| (((y - ylo) / cell).floor() as i64).clamp(0, nrows as i64 - 1) as usize;
        for lp in loops {
            let n = lp.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let (a, b) = (lp[i], lp[(i + 1) % n]);
                let (r0, r1) = (row_of(a[1].min(b[1])), row_of(a[1].max(b[1])));
                for row in rows.iter_mut().take(r1 + 1).skip(r0) {
                    row.push((a, b));
                }
            }
        }
        PipRows {
            y0: ylo,
            cell,
            rows,
        }
    }

    /// Even-odd membership of `p` (inside the outer loop, outside the holes).
    pub(crate) fn inside(&self, p: P2) -> bool {
        if self.rows.is_empty() {
            return false;
        }
        let r = ((p[1] - self.y0) / self.cell).floor();
        if r < 0.0 || r >= self.rows.len() as f64 {
            return false; // outside the loops' y-range entirely
        }
        let mut inside = false;
        for &(a, b) in &self.rows[r as usize] {
            // Identical crossing rule to the classic scan (half-open in y).
            if ((a[1] > p[1]) != (b[1] > p[1]))
                && (p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0])
            {
                inside = !inside;
            }
        }
        inside
    }
}

/// Squared distance from point `p` to segment `a`-`b`.
fn pt_seg_dist2(p: P2, a: P2, b: P2) -> f64 {
    let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
    let (wx, wy) = (p[0] - a[0], p[1] - a[1]);
    let c1 = vx * wx + vy * wy;
    if c1 <= 0.0 {
        return wx * wx + wy * wy;
    }
    let c2 = vx * vx + vy * vy;
    if c2 <= c1 {
        return (p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2);
    }
    let t = c1 / c2;
    (p[0] - (a[0] + t * vx)).powi(2) + (p[1] - (a[1] + t * vy)).powi(2)
}

/// A planar patch meshed: its boundary (points and constraint `segments`), a
/// sizing `target` and an `inside` predicate in, a graded, sliver-free
/// triangulation out.
///
/// A graded CVT seed ([`cvt_fill`]) fills the interior at the field; seeds that
/// would hide under a contour edge are dropped (edge clearance); then Ruppert
/// refines the interior with the boundary PROTECTED -- the caller pre-samples
/// the contours, so re-splitting them would only chase the seed points off the
/// boundary into thin spikes. `step` seeds the CVT grid; `max_passes` and
/// `cvt_iters` bound the work. `slide` tells the boundary points the final
/// smoothing may move along straight stretches of the outline; the others come
/// back exactly as given. Returns `(points, triangles)`, the boundary first (in
/// its order), then the interior.
#[allow(clippy::too_many_arguments)]
pub fn mesh_constrained(
    mut boundary: Vec<P2>,
    segments: Vec<(usize, usize)>,
    target: impl Fn(P2) -> f64,
    inside: impl Fn(P2) -> bool,
    step: f64,
    min_angle_deg: f64,
    cvt_iters: usize,
    max_passes: usize,
    slide: impl Fn(P2) -> bool,
) -> (Vec<P2>, Vec<[usize; 3]>) {
    if boundary.len() < 3 {
        return (boundary, Vec::new());
    }
    let (mut lo, mut hi) = (boundary[0], boundary[0]);
    for &p in &boundary {
        lo[0] = lo[0].min(p[0]);
        lo[1] = lo[1].min(p[1]);
        hi[0] = hi[0].max(p[0]);
        hi[1] = hi[1].max(p[1]);
    }
    let mut interior = cvt_fill(&boundary, lo, hi, step, &target, cvt_iters, &inside, true);
    // cvt_fill clears boundary POINTS but not boundary EDGES, so a seed can land
    // just under a contour segment and form a flat boundary triangle. Drop seeds
    // closer than ~half the local size to any segment. Segments live in a
    // uniform grid over their own AABBs; each seed scans only the cells its
    // clearance ball overlaps (the all-segments scan was O(seeds * segments)).
    {
        let seg_cell = {
            let mean: f64 = segments
                .iter()
                .map(|&(u, v)| dist2(boundary[u], boundary[v]).sqrt())
                .sum::<f64>()
                / segments.len().max(1) as f64;
            mean.max(1e-12)
        };
        let skey = |p: P2| {
            (
                (p[0] / seg_cell).floor() as i64,
                (p[1] / seg_cell).floor() as i64,
            )
        };
        let mut seg_grid: rustc_hash::FxHashMap<(i64, i64), Vec<u32>> =
            rustc_hash::FxHashMap::default();
        for (si, &(u, v)) in segments.iter().enumerate() {
            let (a, b) = (boundary[u], boundary[v]);
            let lo = skey([a[0].min(b[0]), a[1].min(b[1])]);
            let hi = skey([a[0].max(b[0]), a[1].max(b[1])]);
            for cx in lo.0..=hi.0 {
                for cy in lo.1..=hi.1 {
                    seg_grid.entry((cx, cy)).or_default().push(si as u32);
                }
            }
        }
        interior.retain(|&p| {
            let r = 0.5 * target(p);
            let r2 = r * r;
            let (clo, chi) = (skey([p[0] - r, p[1] - r]), skey([p[0] + r, p[1] + r]));
            for cx in clo.0..=chi.0 {
                for cy in clo.1..=chi.1 {
                    if let Some(list) = seg_grid.get(&(cx, cy)) {
                        for &si in list {
                            let (u, v) = segments[si as usize];
                            if pt_seg_dist2(p, boundary[u], boundary[v]) < r2 {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        });
    }
    // The boundary is PROTECTED where the caller says (it pre-samples it): Ruppert refines
    // the interior but never re-splits such a contour edge, so no boundary spikes.
    refine_quality(
        &mut boundary,
        &segments,
        &mut interior,
        &target,
        &inside,
        min_angle_deg,
        max_passes,
        slide,
    );
    let mut all = boundary;
    all.extend(interior);
    let tris = triangulate_constrained(&all, &segments, &inside);
    (all, tris)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delaunay2_grid_triangle_count() {
        let mut pts = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                pts.push([i as f64, j as f64]);
            }
        }
        // 25 points, 16 on the hull -> 2*25 - 2 - 16 = 32 triangles.
        assert_eq!(delaunay2(&pts).len(), 32);
    }

    /// Even-odd point-in-polygon over one or more loops (test helper).
    fn pip(p: P2, loops: &[Vec<usize>], pts: &[P2]) -> bool {
        let mut inside = false;
        for lp in loops {
            let m = lp.len();
            for k in 0..m {
                let a = pts[lp[k]];
                let b = pts[lp[(k + 1) % m]];
                if (a[1] > p[1]) != (b[1] > p[1]) {
                    let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
                    if p[0] < x {
                        inside = !inside;
                    }
                }
            }
        }
        inside
    }

    fn loops_to_segs(loops: &[Vec<usize>]) -> Vec<(usize, usize)> {
        let mut s = Vec::new();
        for lp in loops {
            let m = lp.len();
            for k in 0..m {
                s.push((lp[k], lp[(k + 1) % m]));
            }
        }
        s
    }

    fn tri_area(t: [usize; 3], pts: &[P2]) -> f64 {
        let (a, b, c) = (pts[t[0]], pts[t[1]], pts[t[2]]);
        0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
    }

    #[test]
    fn constrained_l_polygon_fills_only_the_l() {
        // Reflex (non-convex) L: an unconstrained Delaunay would triangulate the
        // convex hull (area 4); the constrained one must cover exactly the L
        // (area 3) and contain every boundary edge.
        let pts: Vec<P2> = vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let loops = vec![vec![0, 1, 2, 3, 4, 5]];
        let segs = loops_to_segs(&loops);
        let tris = triangulate_constrained(&pts, &segs, |p| pip(p, &loops, &pts));
        let area: f64 = tris.iter().map(|&t| tri_area(t, &pts)).sum();
        assert!((area - 3.0).abs() < 1e-9, "L area should be 3, got {area}");
        // every boundary edge is an edge of some kept triangle
        for k in 0..6 {
            let (a, b) = (loops[0][k], loops[0][(k + 1) % 6]);
            let present = tris.iter().any(|t| {
                (0..3).any(|e| {
                    let (u, v) = (t[e], t[(e + 1) % 3]);
                    (u == a && v == b) || (u == b && v == a)
                })
            });
            assert!(present, "boundary edge ({a},{b}) missing");
        }
    }

    #[test]
    fn constrained_square_with_hole_leaves_the_hole_empty() {
        // Outer square [0,4]^2 (CCW) with a square hole [1,3]^2 (CW): the meshed
        // region is the annulus, area 16 - 4 = 12, and no triangle centroid may
        // fall in the hole.
        let pts: Vec<P2> = vec![
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0], // outer
            [1.0, 1.0],
            [1.0, 3.0],
            [3.0, 3.0],
            [3.0, 1.0], // hole (CW)
        ];
        let loops = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let segs = loops_to_segs(&loops);
        let tris = triangulate_constrained(&pts, &segs, |p| pip(p, &loops, &pts));
        let area: f64 = tris.iter().map(|&t| tri_area(t, &pts)).sum();
        assert!(
            (area - 12.0).abs() < 1e-9,
            "annulus area should be 12, got {area}"
        );
        for &t in &tris {
            let c = [
                (pts[t[0]][0] + pts[t[1]][0] + pts[t[2]][0]) / 3.0,
                (pts[t[0]][1] + pts[t[1]][1] + pts[t[2]][1]) / 3.0,
            ];
            let in_hole = c[0] > 1.0 && c[0] < 3.0 && c[1] > 1.0 && c[1] < 3.0;
            assert!(!in_hole, "triangle centroid {c:?} lies in the hole");
        }
    }

    #[test]
    fn cvt_fill_square_well_separated() {
        // Unit square boundary at spacing 0.2.
        let m = 5;
        let mut boundary = Vec::new();
        for i in 0..m {
            boundary.push([i as f64 / m as f64, 0.0]);
            boundary.push([1.0, i as f64 / m as f64]);
            boundary.push([1.0 - i as f64 / m as f64, 1.0]);
            boundary.push([0.0, 1.0 - i as f64 / m as f64]);
        }
        let sq = |p: P2| p[0] > 0.0 && p[0] < 1.0 && p[1] > 0.0 && p[1] < 1.0;
        let interior = cvt_fill(
            &boundary,
            [0.0, 0.0],
            [1.0, 1.0],
            0.2,
            |_| 0.2,
            12,
            sq,
            true,
        );
        assert!(!interior.is_empty());
        let mut all = boundary.to_vec();
        all.extend_from_slice(&interior);
        let mut min_sep2 = f64::MAX;
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                min_sep2 = min_sep2.min(dist2(all[i], all[j]));
            }
        }
        assert!(
            min_sep2.sqrt() >= 0.5 * 0.2,
            "points too close: {}",
            min_sep2.sqrt()
        );
    }
}
