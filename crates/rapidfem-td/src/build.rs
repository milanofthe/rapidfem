// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The time-domain operator of a [`Model`] on a mesh.
//!
//! Maps the backend-independent model onto the DG operator:
//!
//! - materials: diagonal `ε`, `μ` and conductivity per tet; a Debye material
//!   runs at `ε∞` with its relaxation on the auxiliary-polarisation block;
//! - PML regions: graded impedance-matched absorbing layers (the TD
//!   equivalent of the frequency-domain coordinate stretch);
//! - ports in the order rectangular, coax, Floquet, numerical wave, then the
//!   absorbing (ABC) faces, so the modal-port indices stay contiguous;
//! - PEC faces (boundary PEC is the default anyway, internal plates are
//!   retagged) and periodic pairs.
//!
//! Lumped ports have no time-domain counterpart (a uniform delta-gap only
//! carries a clean mode on a genuine parallel-plate gap) and are rejected.
//! Surface impedances, lumped elements, PMC and user-defined ports are
//! frequency-domain only; they are skipped with a warning.

use crate::dispersive::DebyeMaterial;
use crate::rhs::{ElemMaterial, MaxwellOperator, PecSpec, PeriodicSpec, PortSpec};
use crate::waveguide::FloquetPolarisation;
use rapidfem_core::mesh::Mesh;
use rapidfem_core::model::{FaceSpec, Model, WaveKind};

/// Round-trip loss `ν_max·t` of a matched absorbing layer. The loss rate
/// ramps quadratically with depth (`ν_max·frac²`), so a slab of thickness `t`
/// attenuates a round trip by about `exp(−2·ν_max·t/3)`; 24 gives 1e-7 at any
/// thickness.
pub const ABSORBER_LOSS_BUDGET: f64 = 24.0;

/// Build the operator of `model` on `mesh`: DG order `order`, flux blend
/// `flux_alpha` (1 upwind, 0 central), `c` the speed of light in the mesh's
/// length unit (the operator runs with time in length units).
pub fn operator_from_model(
    mesh: &Mesh,
    model: &Model,
    order: usize,
    flux_alpha: f64,
    c: f64,
) -> Result<MaxwellOperator, String> {
    let tets_of = |tag: i32| mesh.vtag_to_tet.get(&tag).map(|v| v.as_slice()).unwrap_or(&[]);

    // Materials, then the dispersive tets at ε∞.
    let mut materials = vec![ElemMaterial::VACUUM; mesh.n_tets()];
    let mut disp_elems: Vec<(usize, DebyeMaterial)> = Vec::new();
    for m in &model.materials {
        let eps = match (m.debye, m.er_diag) {
            (Some(d), _) => [d.er_inf; 3],
            (None, Some(e)) => e,
            (None, None) => [m.er; 3],
        };
        let mu = m.ur_diag.unwrap_or([m.ur; 3]);
        for &t in tets_of(m.volume_tag) {
            materials[t] = ElemMaterial { eps, mu, sigma: m.conductivity, sigma_m: 0.0 };
            if let Some(d) = m.debye {
                // Relaxation time in operator time (length units).
                let debye = DebyeMaterial { eps_inf: d.er_inf, eps_static: d.er_static, tau: c * d.tau_s };
                disp_elems.push((t, debye));
            }
        }
        if m.drude.is_some() {
            eprintln!(
                "  TD: Drude dispersion on volume tag {} is not modelled (only Debye has an ADE)",
                m.volume_tag
            );
        }
    }

    // PML regions as graded matched absorbers: σ = ν ε, σ* = ν μ keeps the
    // layer reflectionless at its interface.
    for p in &model.pml {
        let axis = (0..3)
            .max_by(|&a, &b| p.direction[a].abs().partial_cmp(&p.direction[b].abs()).unwrap())
            .unwrap();
        let is_low = p.direction[axis] < 0.0;
        let nu_max = if p.thickness > 0.0 { ABSORBER_LOSS_BUDGET / p.thickness } else { 0.0 };
        for &t in tets_of(p.volume_tag) {
            let centroid = mesh.tets[t].iter().map(|&n| mesh.nodes[n][axis]).sum::<f64>() / 4.0;
            let depth = if is_low { p.inner_face - centroid } else { centroid - p.inner_face };
            if depth <= 0.0 {
                continue;
            }
            let frac = (depth / p.thickness).clamp(0.0, 1.0);
            let nu = nu_max * frac * frac;
            let m = &mut materials[t];
            m.sigma = nu * m.eps[0];
            m.sigma_m = nu * m.mu[0];
        }
    }

    // Ports, grouped by kind in the order the port indices assume.
    let missing = |what: &str, tag: i32| format!("{what} face tag {tag} has no triangles");
    let mut port_specs: Vec<PortSpec> = Vec::new();
    for f in &model.faces {
        if let FaceSpec::Rectangular { tag, mode, .. } = f {
            let spec = PortSpec::from_mesh_tag_with_z0(mesh, *tag, (mode[0], mode[1]), None, 1.0)
                .ok_or_else(|| missing("rectangular port", *tag))?;
            port_specs.push(spec);
        }
    }
    for f in &model.faces {
        if let FaceSpec::Coax { tag, origin, .. } = f {
            let spec = PortSpec::coax_from_mesh_tag(mesh, *tag, *origin)
                .ok_or_else(|| missing("coax port", *tag))?;
            port_specs.push(spec);
        }
    }
    for f in &model.faces {
        if let FaceSpec::Floquet { tag, scan_theta_deg, scan_phi_deg, mode_nr, .. } = f {
            let polarisation = match mode_nr {
                1 => FloquetPolarisation::Te,
                2 => FloquetPolarisation::Tm,
                _ => return Err(format!("floquet port tag {tag}: mode_nr must be 1 (TE) or 2 (TM)")),
            };
            let spec = PortSpec::floquet_from_mesh_tag(
                mesh,
                *tag,
                polarisation,
                scan_theta_deg.to_radians(),
                scan_phi_deg.to_radians(),
                None,
            )
            .ok_or_else(|| missing("floquet port", *tag))?;
            port_specs.push(spec);
        }
    }
    // Numerical wave ports read the per-tet ε and mark every node on a PEC
    // face as an internal conductor of the cross-section.
    let eps_per_tet: Vec<f64> = materials.iter().map(|m| m.eps[0]).collect();
    let mut pec_nodes = vec![false; mesh.n_nodes()];
    for &tag in &model.pec_tags {
        for &t in mesh.ftag_to_tri.get(&tag).map(|v| v.as_slice()).unwrap_or(&[]) {
            for &n in &mesh.tris[t] {
                pec_nodes[n] = true;
            }
        }
    }
    for f in &model.faces {
        if let FaceSpec::WaveNumerical { tag, f0, mode_index, kind, .. } = f {
            // A vector solve at k0 = 2π f0 / c; without f0 the scalar TE/TM
            // solve of a homogeneous guide.
            let k0 = match (kind, f0) {
                (WaveKind::Vector, Some(f)) => 2.0 * std::f64::consts::PI * f / c,
                _ => -1.0,
            };
            let spec = PortSpec::wave_from_mesh_tag(
                mesh,
                *tag,
                *kind != WaveKind::Tm,
                *mode_index,
                Some(&eps_per_tet),
                k0,
                Some(&pec_nodes),
            )
            .ok_or_else(|| {
                format!(
                    "wave port face tag {tag}: no triangles, or the cross-section \
                     eigensolve found fewer than {} mode(s)",
                    mode_index + 1
                )
            })?;
            port_specs.push(spec);
        }
    }
    for f in &model.faces {
        match f {
            FaceSpec::Abc { tag } => {
                let spec = PortSpec::absorbing_from_mesh_tag(mesh, *tag)
                    .ok_or_else(|| missing("ABC", *tag))?;
                port_specs.push(spec);
            }
            FaceSpec::Lumped { .. } => {
                return Err("LumpedPort is not supported by the time-domain backend. Use a \
                            modal port (RectWaveguidePort, CoaxPort) for waveguide / TEM \
                            geometries, or a WavePort (2D cross-section eigensolve) for \
                            microstrip-class lines. The frequency-domain backend still \
                            supports LumpedPort."
                    .to_string());
            }
            FaceSpec::SurfaceImpedance { tag, .. }
            | FaceSpec::LumpedElement { tag, .. }
            | FaceSpec::UserDefined { tag, .. }
            | FaceSpec::Pmc { tag } => {
                eprintln!("  TD: face tag {tag} carries a frequency-domain-only condition, ignored");
            }
            _ => {}
        }
    }

    let mut periodic_specs = Vec::new();
    for &(a, b) in &model.periodic {
        let spec = PeriodicSpec::from_mesh_tags(mesh, a, b)
            .ok_or_else(|| format!("periodic pair ({a}, {b}): a face tag has no triangles"))?;
        periodic_specs.push(spec);
    }
    let mut pec_specs = Vec::new();
    for &tag in &model.pec_tags {
        pec_specs.push(PecSpec::from_mesh_tag(mesh, tag).ok_or_else(|| missing("PEC", tag))?);
    }

    Ok(MaxwellOperator::new_full(
        mesh,
        order,
        flux_alpha,
        &materials,
        &port_specs,
        &disp_elems,
        &periodic_specs,
        &pec_specs,
    ))
}
