//! The count-driven volume entry point ([`mesh_budgeted`]) and the sizing
//! domain every mesh of a model is built with ([`build_sizing_domain`]).
//!
//! The volume engine itself is the restricted-Delaunay core ([`crate::mesh3`],
//! reached via [`crate::conform::mesh_model`]); this module wraps it with the
//! element-budget retune loop. The surface export runs on the same core
//! ([`crate::mesh3::brep::surface_mesh`]).

use crate::conform::{MeshParams, TetMesh};
use crate::domain::DomainTree;
use rapidmesh_geom::vec3::dist;

/// Mesh `plc` to an optional element budget, with the optional quality pass.
///
/// `optimize_passes`: `Some(n)` runs [`crate::optimize::optimize`] (whose size
/// targets mirror the params, so the quality pass respects the mesher's sizing)
/// for `n` passes after each remesh; `None` skips it.
///
/// `target_elements`: `Some(target)` retunes the GLOBAL size scale over a few
/// remeshes so the FINAL tet count (after optimize, which can shrink it ~25%)
/// lands within 6% of `target` -- the tet count scales as `scale^-3`, so each
/// step multiplies the scale by `(n/target)^(1/3)`. The relative refinement
/// (curvature + size points) keeps its shape throughout. `None` meshes once.
///
/// Returns the mesh and the (possibly budget-scaled) params it was built with.
/// This is the count-driven volume entry point; the surface analogue is the
/// `surf_target_count` budget of [`crate::mesh3::brep::surface_mesh`].
pub fn mesh_budgeted(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
    target_elements: Option<usize>,
    optimize_passes: Option<usize>,
) -> (TetMesh, MeshParams) {
    // The volume backend is the restricted-Delaunay REFINEMENT core
    // (analytic carriers, protecting balls, manifold sweeps); the budget
    // loop and the optimize pass wrap it unchanged.
    let infallible: Result<_, std::convert::Infallible> =
        budgeted(model, params, target_elements, optimize_passes, &|p| {
            Ok(crate::conform::mesh_model(model, p))
        });
    match infallible {
        Ok(x) => x,
        Err(never) => match never {},
    }
}

/// [`mesh_budgeted`] around any mesher of `model`: the tet budget scales the
/// sizes over a few meshes, the optimizer runs on each.
pub fn budgeted<E>(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
    target_elements: Option<usize>,
    optimize_passes: Option<usize>,
    mesher: &dyn Fn(&MeshParams) -> Result<TetMesh, E>,
) -> Result<(TetMesh, MeshParams), E> {
    // The thickness bounds join the per-region sizes here already, so the
    // optimizer keeps them too.
    let params = &params.with_thickness_caps(&model.plc);
    let mesh_once = |p: &MeshParams| -> Result<TetMesh, E> {
        let mut m = mesher(p)?;
        if let Some(passes) = optimize_passes {
            let opt = crate::optimize::OptimizeParams {
                passes,
                maxh: p.maxh,
                region_maxh: p.region_maxh.clone(),
                face_maxh: p.face_maxh.clone(),
                surface_maxh: p.surface_maxh.clone(),
            };
            crate::optimize::optimize(&mut m, &opt);
        }
        Ok(m)
    };
    match target_elements {
        Some(target) if target > 0 => {
            let mut s = 1.0_f64;
            let mut out: Option<(TetMesh, MeshParams)> = None;
            for _ in 0..6 {
                let p = params.scaled(s);
                let m = mesh_once(&p)?;
                let n = m.tets.len().max(1);
                let rel = (n as f64 - target as f64).abs() / target as f64;
                out = Some((m, p));
                if rel < 0.06 {
                    break;
                }
                s *= (n as f64 / target as f64).powf(1.0 / 3.0);
            }
            Ok(out.expect("budget loop runs at least once"))
        }
        _ => {
            let m = mesh_once(params)?;
            Ok((m, params.clone()))
        }
    }
}

/// Builds the domain sizing octree with the per-entity overrides applied: per-face
/// `surf_maxh` -> per-facet volume target (`facet_surf`), and per-edge `edge_maxh`
/// -> point sources sampled along the brep edge chain (so the field stays fine
/// along a refined edge). Shared by the volume path (`mesh_refine`) and the
/// surface-only export (`surface_mesh`), so BOTH honor the same sizing knobs
/// (per-entity AND global caps, which `DomainTree::build` composes).
pub(crate) fn build_sizing_domain(
    model: &rapidmesh_brep::Model,
    params: &MeshParams,
) -> DomainTree {
    let (plc, brep) = (&model.plc, &model.brep);
    // Per-face `surf_maxh` -> per-facet volume target, per-face `surf_tol`
    // -> per-facet chord tolerance.
    let mut facet_surf = vec![f64::INFINITY; plc.triangles.len()];
    let mut facet_tol = vec![params.tol_surf; plc.triangles.len()];
    for (fid, f) in brep.faces.iter().enumerate() {
        let h = params.surf_maxh_for(fid);
        let tol = params.surf_tol_for(fid);
        for &ti in &f.facets {
            facet_surf[ti as usize] = facet_surf[ti as usize].min(h);
            facet_tol[ti as usize] = facet_tol[ti as usize].min(tol);
        }
    }
    if params.edge_maxh.is_empty() {
        return DomainTree::build(plc, model.index(), params, &facet_surf, &facet_tol);
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
    DomainTree::build(plc, model.index(), &pa, &facet_surf, &facet_tol)
}
