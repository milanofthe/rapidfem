//! The size field of a mesh ([`tree::DomainTree`], built with every
//! override by [`build_sizing_domain`]), the element budgets that scale it
//! ([`budget`], [`budgeted`]) and the refinement toward a smallest dihedral
//! angle ([`angled`]).

pub(crate) mod tree;

use crate::mesh::TetMesh;

use crate::params::MeshParams;
use crate::simplex::tet_min_dihedral;
use crate::sizing::tree::DomainTree;
use rapidmesh_exact::vector::{centroid, dist};

/// The mesh of `model` by `mesher`, under a tet budget where one is given:
/// the sizes scale over a few meshes until the count lands near it.
pub fn budgeted<E>(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
    target_elements: Option<usize>,
    mesher: &dyn Fn(&MeshParams) -> Result<TetMesh, E>,
) -> Result<(TetMesh, MeshParams), E> {
    // the thickness bounds join the per-region sizes
    let params = &params.with_thickness_caps(&model.plc);
    let target = target_elements.unwrap_or(0);
    budget(params, target, Budget::Near(3.0), mesher, |m| m.tets.len())
}

/// How the size follows the curvature: the size at a radius of curvature.
///
/// A chord tolerance keeps the deviation a share of the radius (constant
/// segments per turn). A gap bounds what the elements leave between them and
/// the true geometry, on average, as lost area per unit length of a curve or
/// lost volume per unit area of a surface; with the elements' order the size
/// follows from it (sympy and quadrature on circles and spheres, mid-edge
/// nodes of a quadratic element on the curve):
///
/// | lost per unit | flat | quadratic |
/// |---|---|---|
/// | curve | `h^2 / (12 R)` | `h^4 / (960 R^3)` |
/// | surface (sphere) | `h^2 / (8 R)` | `h^4 / (182 R^3)` |
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CurvatureLaw {
    /// The chord deviates by at most this share of the radius.
    Chord(f64),
    /// At most this gap, for elements of this order.
    Gap { g: f64, order: u8 },
}

/// The largest turn (radians) a quadratic element follows: a quarter of
/// that, eight elements round a full turn at least. Beyond it the error
/// leaves the asymptotic law, and the curved tets on the boundary turn
/// invalid and are put back on their chords.
const QUADRATIC_TURN: f64 = std::f64::consts::FRAC_PI_4;

impl CurvatureLaw {
    /// The size along a curve of radius `r`.
    pub fn curve(&self, r: f64) -> f64 {
        match *self {
            CurvatureLaw::Chord(tol) => r * (8.0 * tol.max(1e-12)).sqrt(),
            CurvatureLaw::Gap { g, order: 1 } => (12.0 * g * r).sqrt(),
            CurvatureLaw::Gap { g, .. } => {
                (960.0 * g * r * r * r).powf(0.25).min(QUADRATIC_TURN * r)
            }
        }
    }

    /// The size on a surface of smallest principal radius `r`.
    pub fn surface(&self, r: f64) -> f64 {
        match *self {
            CurvatureLaw::Chord(tol) => r * (8.0 * tol.max(1e-12)).sqrt(),
            CurvatureLaw::Gap { g, order: 1 } => (8.0 * g * r).sqrt(),
            CurvatureLaw::Gap { g, .. } => {
                (182.0 * g * r * r * r).powf(0.25).min(QUADRATIC_TURN * r)
            }
        }
    }
}

/// The share of the error's gap the sizes plan for: the finish slides
/// points along curves and carriers for the tets' quality, so the samples end
/// up spaced unevenly (a disc rim of even 0.039 at 0.013 to 0.1), and the
/// longer elements lose more than the shorter save; half the gap keeps the
/// measured error below the one asked for.
const GAP_SHARE: f64 = 0.5;

/// The curvature law of every B-rep face and edge. A per-entity tolerance
/// is a chord tolerance; else with a geometric error a gap: that error times
/// the thickness of what the face bounds (a region's volume over its
/// surface, the thinnest region on either side; a sheet's area over its
/// perimeter), an edge the smallest of its faces'; else the global chord
/// tolerances.
pub(crate) fn curvature_laws(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
) -> (Vec<CurvatureLaw>, Vec<CurvatureLaw>) {
    let (plc, brep) = (&model.plc, &model.brep);
    let explicit_face = |fi: usize| params.surf_tol.iter().any(|&(i, _)| i as usize == fi);
    let explicit_edge = |ei: usize| params.edge_tol.iter().any(|&(i, _)| i as usize == ei);
    if !(params.geom_error > 0.0) {
        let faces = (0..brep.faces.len())
            .map(|fi| CurvatureLaw::Chord(params.surf_tol_for(fi)))
            .collect();
        let edges = (0..brep.edges.len())
            .map(|ei| CurvatureLaw::Chord(params.edge_tol_for(ei)))
            .collect();
        return (faces, edges);
    }
    let order = params.order.clamp(1, 2);
    // Volume over surface per region (half its thickness `2 V / S`).
    let depth: std::collections::HashMap<u32, f64> = plc
        .region_thickness()
        .into_iter()
        .map(|(r, t)| (r, 0.5 * t))
        .collect();
    // Area and perimeter per face, for the sheets.
    let mut perimeter = vec![0.0; brep.faces.len()];
    for c in &brep.coedges {
        let chain = &brep.edges[c.edge.0 as usize].chain;
        perimeter[c.face.0 as usize] += chain.windows(2).map(|w| dist(w[0], w[1])).sum::<f64>();
    }
    let area = |fi: usize| -> f64 {
        brep.faces[fi]
            .facets
            .iter()
            .map(|&t| {
                let [a, b, c] = plc.triangles[t as usize].map(|v| plc.vertices[v as usize]);
                0.5 * rapidmesh_exact::vector::len(rapidmesh_exact::vector::cross(
                    rapidmesh_exact::vector::sub(b, a),
                    rapidmesh_exact::vector::sub(c, a),
                ))
            })
            .sum()
    };
    let faces: Vec<CurvatureLaw> = brep
        .faces
        .iter()
        .enumerate()
        .map(|(fi, f)| {
            if explicit_face(fi) {
                return CurvatureLaw::Chord(params.surf_tol_for(fi));
            }
            let [a, b] = f.regions.map(|r| r.0);
            let thick = if a == b {
                area(fi) / perimeter[fi].max(f64::MIN_POSITIVE)
            } else {
                [a, b]
                    .iter()
                    .filter_map(|r| depth.get(r).copied())
                    .fold(f64::INFINITY, f64::min)
            };
            CurvatureLaw::Gap {
                g: GAP_SHARE * params.geom_error * thick,
                order,
            }
        })
        .collect();
    let mut edges: Vec<CurvatureLaw> = (0..brep.edges.len())
        .map(|ei| CurvatureLaw::Chord(params.edge_tol_for(ei)))
        .collect();
    let mut gap = vec![f64::INFINITY; brep.edges.len()];
    for c in &brep.coedges {
        if let CurvatureLaw::Gap { g, .. } = faces[c.face.0 as usize] {
            let e = &mut gap[c.edge.0 as usize];
            *e = e.min(g);
        }
    }
    for (ei, law) in edges.iter_mut().enumerate() {
        if !explicit_edge(ei) && gap[ei].is_finite() {
            *law = CurvatureLaw::Gap { g: gap[ei], order };
        }
    }
    (faces, edges)
}

/// Rounds of refinement toward a smallest dihedral angle at most.
const ANGLE_ROUNDS: usize = 6;
/// Rounds in a row without a better mesh before the refinement stops.
const ANGLE_PATIENCE: usize = 2;
/// A tet below the angle becomes a size source at its centroid of this
/// share of its mean edge.
const ANGLE_SHRINK: f64 = 0.5;

/// [`budgeted`], refined where tets stay below `min_angle` (degrees): each
/// round puts a size source at every such tet, half its mean edge, and
/// meshes again. Most of them are flat tets through a layer or around a
/// feature far below the size, which the finer size there resolves; the
/// refinement stays where they are, not in the whole region. The best mesh
/// (fewest tets below the angle, then the larger smallest one) is kept, and
/// the rounds stop when none are left, after two rounds without a better
/// mesh (a wedge sharper than the angle stays one) or when one fails (sizes
/// the geometry cannot follow there). Tets left below the angle are a warning.
pub fn angled<E>(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
    target_elements: Option<usize>,
    min_angle: Option<f64>,
    mesher: &dyn Fn(&MeshParams) -> Result<TetMesh, E>,
) -> Result<(TetMesh, MeshParams), E> {
    let first = budgeted(model, params, target_elements, mesher)?;
    let Some(angle) = min_angle.filter(|a| *a > 0.0) else {
        return Ok(first);
    };
    // The tets below the angle as size sources, with the smallest angle.
    let below = |m: &TetMesh| {
        let mut worst = (f64::INFINITY, [0.0; 3]);
        let sources: Vec<([f64; 3], f64)> = m
            .tets
            .iter()
            .filter_map(|t| {
                let p = t.map(|v| m.points[v]);
                let q = tet_min_dihedral(p);
                let c = centroid(p);
                if q < worst.0 {
                    worst = (q, c);
                }
                (q < angle).then(|| {
                    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
                    let mean = edges.iter().map(|&(a, b)| dist(p[a], p[b])).sum::<f64>() / 6.0;
                    (c, ANGLE_SHRINK * mean)
                })
            })
            .collect();
        (sources, worst)
    };
    let (mut sources, mut worst) = below(&first.0);
    let mut best = ((sources.len(), -worst.0), first, worst);
    let mut p = params.clone();
    let mut idle = 0;
    for round in 1..=ANGLE_ROUNDS {
        if sources.is_empty() {
            break;
        }
        // A place an earlier round refined and still bad takes half the
        // size it was given there, not half its tets' edges (which the
        // grading around it keeps larger): it resolves in fewer rounds.
        let again: Vec<([f64; 3], f64)> = sources
            .iter()
            .map(|&(c, h)| {
                let before = p
                    .size_points
                    .iter()
                    .filter(|(q, hq)| dist(c, *q) <= 2.0 * hq.max(h))
                    .map(|&(_, hq)| hq)
                    .fold(f64::INFINITY, f64::min);
                (c, h.min(ANGLE_SHRINK * before))
            })
            .collect();
        p.size_points.extend(again);
        let Ok(next) = budgeted(model, &p, target_elements, mesher) else {
            rapidmesh_exact::log::info(
                "mesher.min_angle",
                format!("round {round} failed: the sizes there are finer than the geometry allows"),
            );
            break;
        };
        (sources, worst) = below(&next.0);
        let key = (sources.len(), -worst.0);
        rapidmesh_exact::log::info(
            "mesher.min_angle",
            format!(
                "round {round}: {} tets below {angle} deg, smallest {:.2}",
                key.0, worst.0
            ),
        );
        if key < best.0 {
            best = (key, next, worst);
            idle = 0;
        } else {
            idle += 1;
            if idle == ANGLE_PATIENCE {
                break;
            }
        }
    }
    let ((left, _), mesh, (smallest, at)) = best;
    if left > 0 {
        rapidmesh_exact::log::warn(
            "mesher.min_angle",
            format!(
                "{left} tets stay below {angle} deg, the smallest {smallest:.2} deg at \
                 ({:.6}, {:.6}, {:.6})",
                at[0], at[1], at[2]
            ),
        );
    }
    Ok(mesh)
}

/// How a mesh meets an element budget.
#[derive(Clone, Copy)]
pub(crate) enum Budget {
    /// The count lands near the budget; the count grows with the size to
    /// this power (3 for tets).
    Near(f64),
    /// The count is at most a little over the budget (a cap); the power as
    /// for `Near` (2 for triangles).
    Cap(f64),
}

/// Meshes at most this often to meet a budget.
const BUDGET_ROUNDS: usize = 6;
/// A count this fraction off the budget meets it.
const BUDGET_SLACK: f64 = 0.06;

/// The mesh by `mesh` under a budget of `target` elements (`count` of
/// them; 0 none): the sizes scale by one global factor over a few meshes
/// until the count meets it. Returns the mesh and the parameters it was
/// made with.
pub(crate) fn budget<M, E>(
    params: &MeshParams,
    target: usize,
    how: Budget,
    mesh: impl Fn(&MeshParams) -> Result<M, E>,
    count: impl Fn(&M) -> usize,
) -> Result<(M, MeshParams), E> {
    let mut p = params.clone();
    let mut m = mesh(&p)?;
    if target == 0 {
        return Ok((m, p));
    }
    let mut s = 1.0_f64;
    for _ in 1..BUDGET_ROUNDS {
        let n = count(&m).max(1) as f64;
        let ratio = n / target as f64;
        let (met, power) = match how {
            Budget::Near(d) => ((ratio - 1.0).abs() < BUDGET_SLACK, d),
            Budget::Cap(d) => (ratio <= 1.0 + BUDGET_SLACK, d),
        };
        if met {
            break;
        }
        s *= ratio.powf(1.0 / power);
        p = params.scaled(s);
        m = mesh(&p)?;
    }
    Ok((m, p))
}

/// Builds the domain sizing octree with the per-entity overrides applied: per-face
/// `surf_maxh` -> per-facet volume target (`facet_surf`), and per-edge `edge_maxh`
/// -> point sources sampled along the brep edge chain (so the field stays fine
/// along a refined edge). Shared by the volume meshes and the surface-only
/// export ([`crate::mesher::surface_mesh`]), so BOTH honor the same sizing knobs
/// (per-entity AND global caps, which `DomainTree::build` composes).
pub(crate) fn build_sizing_domain(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
) -> DomainTree {
    let (plc, brep) = (&model.plc, &model.brep);
    // Per-face `surf_maxh` -> per-facet volume target, per-face curvature
    // law (chord tolerance or geometric error) -> per-facet law.
    let (face_laws, _) = curvature_laws(model, params);
    let mut facet_surf = vec![f64::INFINITY; plc.triangles.len()];
    // A facet of several faces takes the finest of their laws.
    let mut facet_law: Vec<Option<CurvatureLaw>> = vec![None; plc.triangles.len()];
    for (fid, f) in brep.faces.iter().enumerate() {
        let h = params.surf_maxh_for(fid);
        let law = face_laws[fid];
        for &ti in &f.facets {
            facet_surf[ti as usize] = facet_surf[ti as usize].min(h);
            let slot = &mut facet_law[ti as usize];
            if slot.is_none_or(|l| law.surface(1.0) < l.surface(1.0)) {
                *slot = Some(law);
            }
        }
    }
    let facet_law: Vec<CurvatureLaw> = facet_law
        .into_iter()
        .map(|l| l.unwrap_or(CurvatureLaw::Chord(params.tol_surf)))
        .collect();
    // The segments of the edges given a tolerance of their own, with it.
    let mut edge_tol: rustc_hash::FxHashMap<[[u64; 3]; 2], f64> = Default::default();
    for &(ei, tol) in &params.edge_tol {
        if let Some(e) = brep.edges.get(ei as usize) {
            for w in e.chain.windows(2) {
                edge_tol.insert(tree::segment_key([w[0], w[1]]), tol);
            }
        }
    }
    if params.edge_maxh.is_empty() {
        return DomainTree::build(
            plc,
            model.index(),
            params,
            &facet_surf,
            &facet_law,
            &edge_tol,
        );
    }
    // Per-edge `edge_maxh` -> point sources along the brep edge chain. Only clones
    // the params when an edge override is actually present.
    let mut pa = params.clone();
    for (eid, e) in brep.edges.iter().enumerate() {
        let Some(&(_, h)) = params.edge_maxh.iter().find(|&&(i, _)| i as usize == eid) else {
            continue;
        };
        for w in e.chain.windows(2) {
            let n = ((dist(w[0], w[1]) / h).ceil() as usize).max(1);
            for k in 0..n {
                let t = k as f64 / n as f64;
                pa.size_points.push((
                    std::array::from_fn(|c| w[0][c] + t * (w[1][c] - w[0][c])),
                    h,
                ));
            }
        }
        if let Some(&last) = e.chain.last() {
            pa.size_points.push((last, h));
        }
    }
    DomainTree::build(plc, model.index(), &pa, &facet_surf, &facet_law, &edge_tol)
}
