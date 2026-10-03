// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! High-level Simulation API: owns a mesh, the model placed on it and the
//! analysis settings, and exposes sweep, eigenmode and far-field. The entry
//! point of the Python bindings.
//!
//! Construction is split from execution so that callers can inspect/modify the
//! pre-built ports and materials before solving.

use num_complex::Complex64 as C64;

use crate::basis::NedelecBasis;
use rapidfem_core::model::{FaceSpec, Model, WaveKind};
use crate::order::OrderPolicy;
use crate::constants::{EPS0, MU0};
use crate::eigenmode::Eigenmode;
use crate::farfield::RadiationPattern;
use crate::interp;
use crate::materials::{self, Material, PmlRegion};
use crate::mesh::Mesh;
use crate::port::Port;
use crate::sparam::{sparam_voltage_surface, sparam_waveport};
use crate::waveguide::{
    cs_from_origin_zaxis, detect_rect_port, lumped_port_dims, AbsorbingBoundary, CoaxPort,
    FloquetPort, LumpedElement, LumpedPort, NumericalWavePort, RectWaveguide, SurfaceImpedance,
    UserDefinedPort,
};
use rapidfem_core::port_eigen::{solve_port_face, ModeKind, PortSolve};
use rapidfem_core::geom::{dot, sub};

/// Result of a frequency sweep.
pub struct SweepResult {
    pub frequencies: Vec<f64>,
    /// S-parameters: `[freq_idx][port_obs][port_exc]`. Only driven ports.
    pub sparams: Vec<Vec<Vec<C64>>>,
    /// FEM E-field solutions: `[freq_idx][port_exc][dof]`.
    pub solutions: Vec<Vec<Vec<C64>>>,
    /// Reference impedance of each driven port, `[freq_idx][port]` (ohm):
    /// the mode impedance of a modal port, the `z0` of a lumped one. The
    /// S-parameters are referenced to these, see [`crate::network`].
    pub port_impedances: Vec<Vec<f64>>,
    /// Number of driven ports (matches the inner dimension of `sparams`).
    pub n_driven: usize,
    /// Total wall-clock for the sweep (s).
    pub solve_time_s: f64,
}

/// Frequency-independent context for per-frequency S-parameter extraction.
/// Built once per sweep (the tet-locating grid and material weights do not
/// depend on frequency), then reused for every frequency so the extraction
/// can run inside the per-frequency streaming callback.
struct SParamCtx {
    grid: interp::TetGrid,
    eps_tet: Vec<f64>,
    mur_tet: Vec<f64>,
    driven_indices: Vec<usize>,
}

/// Analysis settings of a frequency-domain run.
#[derive(Clone, Debug)]
pub struct FdSettings {
    /// Sweep points in Hz, in evaluation order.
    pub frequencies: Vec<f64>,
    /// How the per-cell element orders are chosen.
    pub order: OrderPolicy,
    /// `(target frequency in Hz, number of modes)` of an eigenmode analysis.
    pub eigenmode: Option<(f64, usize)>,
}

/// Simulation context: a mesh, the model placed on it, the settings, and the
/// pre-built BC objects.
pub struct Simulation {
    pub mesh: Mesh,
    pub basis: NedelecBasis,
    pub model: Model,
    pub settings: FdSettings,
    pub ports: Vec<Box<dyn Port>>,
    pub port_tris: Vec<Vec<usize>>,
    pub pec_tris: Vec<usize>,
    pub materials: Vec<Material>,
    pub pml_regions: Vec<PmlRegion>,
}

impl Simulation {
    /// Build a `Simulation` from an owned mesh, its model and the settings.
    /// All BC objects (ports, PEC, materials, PML, lumped integration lines)
    /// are constructed up-front.
    pub fn new(mesh: Mesh, model: Model, settings: FdSettings) -> Result<Self, String> {
        let mut mesh = mesh;
        // Lever ④: non-dimensionalize the geometry to O(1) coordinates so the
        // assembly and its absolute tolerances are unit-/scale-invariant. The
        // transform is reversed at the output boundary (field/coord rescaling).
        // `RAPIDFEM_NO_NORMALIZE` keeps physical units (l0 = 1), used to prove
        // the normalized path is bit-identical on the solver's outputs.
        if std::env::var_os("RAPIDFEM_NO_NORMALIZE").is_none() {
            let l0 = mesh.normalize_characteristic_length();
            eprintln!("  Geometry normalized: L0 = {:.6e} m (mean edge length)", l0);
        }
        // Materials before ports so `wave_numerical` can consult per-tet ε_r
        // when running the vector-hybrid mode solve on the port face, and before
        // the basis, because the order policy reads them.
        let materials = build_materials(&mesh, &model);

        let orders = match settings.order {
            OrderPolicy::Uniform(p) => crate::order::OrderMap::uniform(&mesh, p),
            OrderPolicy::Adaptive { theta } => {
                // Choose the orders at the HIGHEST frequency of the sweep. The
                // wavelength is shortest there, so k·h is largest and the policy
                // reduces the fewest cells: the order map that is adequate at the
                // top of the band is adequate across it. (One map for the whole
                // sweep is also what lets the symbolic factorisation be reused.)
                let f_max = settings.frequencies.iter().copied().fold(0.0_f64, f64::max);
                let (er, ur) =
                    crate::materials::material_tensors(mesh.n_tets(), Some(&materials), f_max, true, None);
                let k = crate::order::cell_wavenumbers(&mesh, &er, &ur, f_max);
                let om = crate::order::wavelength_policy(&mesh, &k, theta);
                let n1 = om.cell.iter().filter(|&&p| p == 1).count();
                eprintln!(
                    "  Order policy (θ = {}, f = {:.4e} Hz): {} of {} cells at order 1",
                    theta,
                    f_max,
                    n1,
                    mesh.n_tets()
                );
                om
            }
        };

        let full_dofs = crate::order::OrderMap::uniform(&mesh, 2).n_dofs();
        let basis = NedelecBasis::with_orders(&mesh, orders);
        eprintln!(
            "RapidFEM - {} tets, {} DOFs{}",
            mesh.n_tets(),
            basis.n_field,
            if basis.n_field < full_dofs {
                format!(
                    " ({:.0}% fewer than uniform order 2)",
                    100.0 * (1.0 - basis.n_field as f64 / full_dofs as f64)
                )
            } else {
                String::new()
            }
        );
        let f_min = settings.frequencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let (ports, port_tris) = build_ports(&mesh, &model, &materials, f_min)?;
        let pec_tris = build_pec_tris(&mesh, &model);
        let pml_regions = build_pml_regions(&mesh, &model);

        Ok(Simulation {
            mesh,
            basis,
            model,
            settings,
            ports,
            port_tris,
            pec_tris,
            materials,
            pml_regions,
        })
    }

    fn ports_dyn(&self) -> Vec<&dyn Port> {
        self.ports.iter().map(|b| b.as_ref()).collect()
    }

    fn port_tris_slices(&self) -> Vec<&[usize]> {
        self.port_tris.iter().map(|v| v.as_slice()).collect()
    }

    fn frequencies(&self) -> Vec<f64> {
        self.settings.frequencies.clone()
    }

    fn materials_opt(&self) -> Option<&[Material]> {
        if self.materials.is_empty() {
            None
        } else {
            Some(self.materials.as_slice())
        }
    }

    fn pml_opt(&self) -> Option<&[PmlRegion]> {
        if self.pml_regions.is_empty() {
            None
        } else {
            Some(self.pml_regions.as_slice())
        }
    }

    /// Run a frequency sweep and extract S-parameters.
    ///
    /// `on_freq`, if given, is invoked after each frequency's solve with
    /// `(freq_idx, freq_hz, s_matrix)` where `s_matrix[obs][exc]` is the
    /// S-parameter block for that frequency, and returns `false` to stop the
    /// sweep early (e.g. on a user interrupt). This lets a UI stream partial
    /// results as the sweep progresses; the returned `SweepResult` covers only
    /// the frequencies actually solved.
    pub fn run_sweep(
        &self,
        on_freq: Option<&dyn Fn(usize, f64, &[Vec<C64>]) -> bool>,
    ) -> Result<SweepResult, String> {
        let mut frequencies = self.frequencies();
        let port_dyn = self.ports_dyn();
        let port_tri_refs = self.port_tris_slices();
        let n_driven = port_dyn.iter().filter(|p| p.is_driven()).count();
        // Build the frequency-independent extraction context once.
        let ctx = self.sparam_ctx(&port_dyn);

        // S-parameters are accumulated per frequency inside the solve callback
        // (so the streaming `on_freq` sees the same matrix that lands in the
        // result), instead of a separate batch pass afterwards. The callback
        // returns whether the sweep should continue (false = interrupt).
        let mut all_sparams: Vec<Vec<Vec<C64>>> = Vec::with_capacity(frequencies.len());
        let t0 = web_time::Instant::now();
        let results;
        {
            let mut on_solve = |fi: usize, freq: f64, sr: &crate::assembly::SolveResult| -> bool {
                let s = self.extract_sparams_one(&ctx, &port_dyn, &port_tri_refs, freq, sr, n_driven);
                let keep_going = match on_freq {
                    Some(cb) => cb(fi, freq, &s),
                    None => true,
                };
                all_sparams.push(s);
                keep_going
            };
            results = crate::assembly::frequency_sweep(
                &self.mesh,
                &self.basis,
                &port_dyn,
                &port_tri_refs,
                &self.pec_tris,
                &frequencies,
                self.materials_opt(),
                self.pml_opt(),
                Some(&mut on_solve),
            )?;
        }
        let solve_time_s = t0.elapsed().as_secs_f64();

        // An early interrupt leaves fewer solved frequencies than requested;
        // truncate so frequencies / sparams / solutions stay the same length.
        frequencies.truncate(results.len());

        let solutions: Vec<Vec<Vec<C64>>> = results
            .into_iter()
            .map(|r| r.solutions.into_iter().collect())
            .collect();
        let port_impedances = frequencies.iter().map(|&f| self.port_impedances(f)).collect();

        Ok(SweepResult {
            frequencies,
            sparams: all_sparams,
            solutions,
            port_impedances,
            n_driven,
            solve_time_s,
        })
    }

    /// Build the frequency-independent S-parameter extraction context (tet
    /// grid + per-tet material weights + driven-port indices). Reused for
    /// every frequency by :meth:`extract_sparams_one`.
    fn sparam_ctx(&self, port_dyn: &[&dyn Port]) -> SParamCtx {
        let driven_indices: Vec<usize> = (0..port_dyn.len())
            .filter(|&i| port_dyn[i].is_driven())
            .collect();

        // The tet-locating grid depends only on the mesh, not the frequency.
        let grid = interp::TetGrid::new(&self.mesh);

        // Local wave-admittance weight √(εᵣ/μᵣ) per tet, for the power-overlap
        // S-parameter (the TEM weight `1/√(μᵣ/εᵣ)`). Constant
        // across a homogeneous port (cancels in the ratio); varies across an
        // inhomogeneous quasi-TEM cross-section, where it is what keeps the
        // extraction unitary. Material scalars are frequency-flat here.
        let n_tets = self.mesh.n_tets();
        let eps_tet = per_tet(&self.materials, n_tets, 1.0, er_scalar);
        let mur_tet = per_tet(&self.materials, n_tets, 1.0, mur_scalar);
        SParamCtx { grid, eps_tet, mur_tet, driven_indices }
    }

    /// Extract the S-parameter block for a single frequency from its solved
    /// field, using the prebuilt `ctx`. Pulled out of the old batch
    /// `extract_sparams` so it can run inside the per-frequency callback.
    fn extract_sparams_one(
        &self,
        ctx: &SParamCtx,
        port_dyn: &[&dyn Port],
        port_tri_refs: &[&[usize]],
        freq: f64,
        freq_result: &crate::assembly::SolveResult,
        n_driven: usize,
    ) -> Vec<Vec<C64>> {
        let weight = |x: f64, y: f64, z: f64| -> f64 {
            match ctx.grid.find_containing_tet(&self.mesh, x, y, z) {
                Some(tet) => (ctx.eps_tet[tet] / ctx.mur_tet[tet]).sqrt(),
                None => 1.0,
            }
        };

        let exc = crate::excitation::Excitation::new(freq, self.mesh.l0);
        let mut freq_s = vec![vec![C64::new(0.0, 0.0); n_driven]; n_driven];

        for (exc_idx, sol) in freq_result.solutions.iter().enumerate() {
            let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
                match ctx.grid.find_containing_tet(&self.mesh, x, y, z) {
                    Some(tet) => interp::eval_field_in_tet(&self.mesh, &self.basis, sol, tet, x, y, z),
                    None => (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)),
                }
            };
            for (obs_idx, &obs_pi) in ctx.driven_indices.iter().enumerate() {
                let active = obs_idx == exc_idx;
                let obs_tris: Vec<[usize; 3]> = port_tri_refs[obs_pi]
                    .iter()
                    .map(|&ti| self.mesh.tris[ti])
                    .collect();
                let s = if let Some((dir, height, v_inc)) = port_dyn[obs_pi].lumped() {
                    // Area-averaged mode projection V = (l/A)∫E·l̂ dS, robust
                    // for tall / non-TEM ports (derivations/lumped_port/).
                    sparam_voltage_surface(
                        &self.mesh.nodes, &obs_tris, dir, height, v_inc, active, &fieldf, 4,
                    )
                } else {
                    sparam_waveport(&self.mesh.nodes, &obs_tris, port_dyn[obs_pi], &exc, active, &fieldf, &weight, 4)
                };
                freq_s[obs_idx][exc_idx] = s;
            }
        }
        freq_s
    }

    /// Run an eigenmode analysis (requires `settings.eigenmode`).
    pub fn run_eigenmode(&self) -> Result<Vec<Eigenmode>, String> {
        let (target, n_modes) = self.settings.eigenmode.ok_or("no eigenmode target set")?;
        crate::eigenmode::solve_eigenmode(
            &self.mesh,
            &self.basis,
            &self.pec_tris,
            self.materials_opt(),
            target,
            n_modes,
        )
    }

    /// The fraction of the incident power at port `port` the structure
    /// accepts, 1 − Σ_i |S_i,port|² (what a lossless antenna radiates; the
    /// realized-gain offset of the far field). `None` without S data.
    pub fn accepted_fraction(&self, result: &SweepResult, freq_idx: usize, port: usize) -> Option<f64> {
        let s = result.sparams.get(freq_idx)?;
        let column: Vec<f64> = s.iter().filter_map(|row| row.get(port)).map(|s| s.norm_sqr()).collect();
        if column.is_empty() {
            return None;
        }
        Some((1.0 - column.iter().sum::<f64>()).clamp(0.0, 1.0))
    }

    /// The reference impedance (ohm) of each driven port at `freq` Hz, in
    /// S-matrix order: the mode impedance of a modal port, the `z0` of a
    /// lumped one.
    pub fn port_impedances(&self, freq: f64) -> Vec<f64> {
        let exc = crate::excitation::Excitation::new(freq, self.mesh.l0);
        self.ports.iter().filter(|p| p.is_driven()).map(|p| p.z_mode(&exc)).collect()
    }

    /// The centroid of every tet in physical coordinates (m).
    pub fn tet_centroids(&self) -> Vec<[f64; 3]> {
        let l0 = self.mesh.l0;
        self.mesh
            .tets
            .iter()
            .map(|tet| {
                let mut c = [0.0; 3];
                for &n in tet {
                    for k in 0..3 {
                        c[k] += self.mesh.nodes[n][k] * l0;
                    }
                }
                c.map(|x| x / 4.0)
            })
            .collect()
    }

    /// Monk-style residual a-posteriori error indicator per tet for a given
    /// `(freq_idx, port_idx)` solution. Returns the full estimate (η per tet,
    /// volume and face contributions, total, marked subset from Dörfler at
    /// `theta`). Diagnostic: it drives user-side refinement, nothing re-meshes.
    pub fn element_errors_at(
        &self,
        result: &SweepResult,
        freq_idx: usize,
        port_idx: usize,
        theta: f64,
    ) -> Option<crate::error_estimator::ErrorEstimate> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(port_idx))?;
        let freq = *result.frequencies.get(freq_idx)?;
        let k0 = crate::excitation::Excitation::new(freq, self.mesh.l0).k0;
        let (er_tensors, _) = materials::material_tensors(self.mesh.n_tets(), self.materials_opt(), freq, true, None);
        Some(crate::error_estimator::estimate_error(
            &self.mesh, &self.basis, solution, k0, &er_tensors, theta,
        ))
    }

    /// Interpolate the FEM E-field at each mesh node for a given (freq_idx, port_idx).
    /// Returns a flat `Vec<C64>` of length `3 * n_nodes` (interleaved Ex, Ey, Ez per node).
    /// Used by the Python pyvista exporter.
    pub fn field_at_nodes(&self, result: &SweepResult, freq_idx: usize, port_idx: usize) -> Option<Vec<C64>> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(port_idx))?;
        Some(self.eval_dofs_at_nodes(solution))
    }

    /// Same shape as `field_at_nodes` but for an eigenmode's DOF vector.
    /// The mode field is a free-field eigenfunction (not normalised to a
    /// driving port), visualisation libraries typically rescale to a
    /// peak magnitude. Returns `None` if the mode's DOF vector is empty
    /// (defensive, `run_eigenmode` never produces empty modes).
    pub fn eigenmode_field_at_nodes(&self, mode: &Eigenmode) -> Option<Vec<C64>> {
        if mode.field.is_empty() {
            return None;
        }
        Some(self.eval_dofs_at_nodes(&mode.field))
    }

    /// The physical E field (V/m) of the DOF vector `solution` at each node,
    /// shared by `field_at_nodes` and `eigenmode_field_at_nodes`.
    fn eval_dofs_at_nodes(&self, solution: &[C64]) -> Vec<C64> {
        // Lever ④: the field reconstructed on the L₀-normalized mesh is L₀·E_phys
        // (the Nédélec basis is scale-invariant), so divide by L₀ for physical
        // V/m. A no-op when l0 = 1.
        let inv_l0 = C64::from(1.0 / self.mesh.l0);
        self.per_node(|t, p| {
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(&self.mesh, &self.basis, solution, t, p[0], p[1], p[2]);
            [ex * inv_l0, ey * inv_l0, ez * inv_l0]
        })
    }

    /// Per-tet loss-equivalent conductivity at angular frequency `omega`:
    ///
    ///     σ_eff = ω · ε₀ · εᵣ · tan(δ) + σ_bulk
    ///
    /// The first term turns dielectric losses (loss tangent) into an
    /// equivalent current density so substrates like Rogers, which carry
    /// tan_δ but no bulk σ, still light up in the J channel. The second
    /// term is the ordinary Ohmic conductivity. Together this matches the
    /// total imaginary permittivity the solver uses for power dissipation.
    fn per_tet_sigma_eff(&self, omega: f64) -> Vec<f64> {
        per_tet(&self.materials, self.mesh.n_tets(), 0.0, |m| omega * EPS0 * m.er * m.tand + m.cond)
    }

    /// `f(tet, node position)` at every mesh node, evaluated in the first tet
    /// holding the node (the same tet for every sampler, so they agree at
    /// material interfaces); zero for a node no tet holds. Flat `[x, y, z]`
    /// per node.
    fn per_node(&self, f: impl Fn(usize, [f64; 3]) -> [C64; 3]) -> Vec<C64> {
        let mut node_to_tet = vec![usize::MAX; self.mesh.n_nodes()];
        for (itet, tet) in self.mesh.tets.iter().enumerate() {
            for &ni in tet {
                if node_to_tet[ni] == usize::MAX {
                    node_to_tet[ni] = itet;
                }
            }
        }
        node_to_tet
            .iter()
            .zip(&self.mesh.nodes)
            .flat_map(|(&t, &p)| if t == usize::MAX { [C64::new(0.0, 0.0); 3] } else { f(t, p) })
            .collect()
    }

    /// Loss-equivalent current density J = σ_eff · E at each mesh node, in
    /// (A/m²). `σ_eff = ω·ε₀·εᵣ·tan(δ) + σ_bulk` covers both Ohmic and
    /// dielectric losses, so this is the actual dissipative current, not
    /// just the bulk-conduction component. Zero in lossless regions.
    /// Returns `Vec<C64>` of length `3 · n_nodes` (interleaved Jx, Jy, Jz).
    pub fn current_density_at_nodes(&self, result: &SweepResult, freq_idx: usize, port_idx: usize) -> Option<Vec<C64>> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(port_idx))?;
        let freq = *result.frequencies.get(freq_idx)?;
        let omega = crate::excitation::Excitation::new(freq, self.mesh.l0).omega;
        let sigma = self.per_tet_sigma_eff(omega);
        let zero = C64::new(0.0, 0.0);
        Some(self.per_node(|t, p| {
            if sigma[t] == 0.0 {
                return [zero; 3];
            }
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(&self.mesh, &self.basis, solution, t, p[0], p[1], p[2]);
            // J = σ·E_phys; E reconstructed on the normalized mesh is L₀·E_phys.
            let s = C64::from(sigma[t] / self.mesh.l0);
            [ex * s, ey * s, ez * s]
        }))
    }

    /// Magnetic field H = ∇×E / (-jωμ₀μ_r) at each mesh node, in (A/m)
    /// (time dependence e^{jωt}: ∇×E = -jωμH).
    /// Returns `Vec<C64>` of length `3 · n_nodes` (interleaved Hx, Hy, Hz).
    /// Uses the analytic Nédélec-2 curl evaluated at the node position.
    pub fn h_field_at_nodes(&self, result: &SweepResult, freq_idx: usize, port_idx: usize) -> Option<Vec<C64>> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(port_idx))?;
        let freq = *result.frequencies.get(freq_idx)?;
        let omega = crate::excitation::Excitation::new(freq, self.mesh.l0).omega;
        let mur = per_tet(&self.materials, self.mesh.n_tets(), 1.0, mur_scalar);
        let j = C64::new(0.0, 1.0);
        Some(self.per_node(|t, p| {
            let curl = crate::interp::eval_curl_in_tet(&self.mesh, &self.basis, solution, t, p[0], p[1], p[2]);
            // ∇×E on the normalized mesh is L₀²·(∇×E)_phys (one L₀ from the basis
            // field, one from the normalized ∇), so H = ∇×E/(-jωμ) needs /L₀².
            let denom = -j * C64::from(omega * MU0 * mur[t] * self.mesh.l0 * self.mesh.l0);
            [curl[0] / denom, curl[1] / denom, curl[2] / denom]
        }))
    }

    /// The far-field pattern at `(freq_idx, exc_port_idx)`. The Huygens
    /// surface is `model.far_field_tag` if one is marked, else the outer
    /// boundary of the domain (open problems, with an ABC on it); `None`
    /// without either. When every outer face but the ABC lies in one plane
    /// (a ground or symmetry plane), that plane is infinite: the surface is
    /// the ABC, closed by its image (see [`crate::farfield`]).
    pub fn compute_farfield(
        &self,
        result: &SweepResult,
        freq_idx: usize,
        exc_port_idx: usize,
        n_theta: usize,
        n_phi: usize,
    ) -> Option<RadiationPattern> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(exc_port_idx))?;
        let (surface, image) = self.huygens_surface()?;
        if surface.is_empty() {
            return None;
        }
        Some(crate::farfield::compute_farfield(
            &self.mesh,
            &self.basis,
            solution,
            &surface,
            image,
            result.frequencies[freq_idx],
            n_theta,
            n_phi,
            4,
            self.accepted_fraction(result, freq_idx, exc_port_idx),
        ))
    }

    /// The triangles of the far-field surface and the image plane closing
    /// it, see [`Self::compute_farfield`]. The candidate surface is the
    /// marked one, else the outer boundary (with an ABC on it). Its pieces
    /// on the domain's conducting boundary (outer faces that are no ABC:
    /// PEC, PMC, the untagged PEC default) close it: in one plane they are
    /// that plane's image, otherwise they stay on the surface (n̂ × E = 0
    /// on PEC leaves only their electric currents).
    fn huygens_surface(&self) -> Option<(Vec<usize>, Option<crate::farfield::ImagePlane>)> {
        let tagged = |pick: fn(&FaceSpec) -> Option<i32>| -> std::collections::HashSet<usize> {
            self.model.faces.iter().filter_map(pick).flat_map(|t| self.mesh.tris_for_tag(t).to_vec()).collect()
        };
        let abc = tagged(|f| if let FaceSpec::Abc { tag } = f { Some(*tag) } else { None });
        let pmc = tagged(|f| if let FaceSpec::Pmc { tag } = f { Some(*tag) } else { None });
        let outer = |t: usize| self.mesh.tri_to_tet[t][1] == usize::MAX;
        let candidate: Vec<usize> = match self.model.far_field_tag {
            Some(tag) => self.mesh.tris_for_tag(tag).to_vec(),
            None if !abc.is_empty() => (0..self.mesh.n_tris()).filter(|&t| outer(t)).collect(),
            None => return None,
        };
        let (closing, open): (Vec<usize>, Vec<usize>) =
            candidate.iter().copied().partition(|&t| outer(t) && !abc.contains(&t));
        match self.plane_of(&closing) {
            Some((point, normal)) => {
                let pec = !closing.iter().all(|t| pmc.contains(t));
                Some((open, Some(crate::farfield::ImagePlane { point, normal, pec })))
            }
            None => Some((candidate, None)),
        }
    }

    /// The plane all of `tris` (boundary triangles) lie in, with the normal
    /// into the domain; `None` if they are empty or not coplanar.
    fn plane_of(&self, tris: &[usize]) -> Option<([f64; 3], [f64; 3])> {
        let m = &self.mesh;
        let &first = tris.first()?;
        let a = m.nodes[m.tris[first][0]];
        // the inward normal; the nodes are O(1) (normalized mesh)
        let n = m.tri_inward_normal(first)?;
        let tol = 1e-9;
        let on_plane = |p: [f64; 3]| dot(sub(p, a), n).abs() < tol;
        tris.iter().all(|&t| m.tris[t].iter().all(|&i| on_plane(m.nodes[i]))).then_some((a, n))
    }
}

// ============================================================================
// Construction helpers, extracted from main.rs's prior orchestration
// ============================================================================

/// The Robin-type boundaries of the model. `f_min`, the lowest frequency of
/// the sweep, bounds the reach of the surface impedances' edge correction.
fn build_ports(
    mesh: &Mesh,
    model: &Model,
    materials: &[Material],
    f_min: f64,
) -> Result<(Vec<Box<dyn Port>>, Vec<Vec<usize>>), String> {
    let mut ports: Vec<Box<dyn Port>> = Vec::new();
    let mut port_tris: Vec<Vec<usize>> = Vec::new();
    // PMC walls a wave port meets (the symmetry plane of a half model) are
    // natural on the port rim too.
    let pmc_tags: Vec<i32> = model
        .faces
        .iter()
        .filter_map(|f| if let FaceSpec::Pmc { tag } = f { Some(*tag) } else { None })
        .collect();

    for pc in &model.faces {
        let tag = pc.tag();
        if let FaceSpec::Pmc { .. } = pc {
            eprintln!("  PMC: tag={}, {} triangles (natural BC)", tag, mesh.tris_for_tag(tag).len());
            continue;
        }
        let tri_ids = mesh.tris_for_tag(tag).to_vec();
        if tri_ids.is_empty() {
            eprintln!("  WARNING: tag {} has no triangles, skipping {:?}", tag, std::mem::discriminant(pc));
            continue;
        }
        let port_num = ports.len() + 1;
        let port: Box<dyn Port> = match pc {
            FaceSpec::Rectangular { tag, width, height, mode, er, power } => {
                let (cs, det_w, det_h) = detect_rect_port(mesh, &tri_ids);
                // The model's dims are physical lengths; the mesh (and det_*) are in
                // L₀ units, so normalize config dims to match (lever ④).
                let w = if *width > 0.0 { *width / mesh.l0 } else { det_w };
                let h = if *height > 0.0 { *height / mesh.l0 } else { det_h };
                let port = RectWaveguide {
                    port_number: port_num,
                    power: *power,
                    mode: (mode[0], mode[1]),
                    er: *er,
                    dims: (w, h),
                    cs,
                };
                eprintln!("  Port {}: rectangular, tag={}, TE{}{}, dims=({:.2}mm, {:.2}mm), er={:.1}",
                    port_num, tag, mode[0], mode[1], w * 1e3, h * 1e3, er);
                Box::new(port)
            }
            FaceSpec::Coax { tag, ri, ro, origin, z_axis, er, power } => {
                let (cs_detected, _, _) = detect_rect_port(mesh, &tri_ids);
                // The model's origin is a physical coordinate; normalize to L₀ units.
                let org = origin
                    .map(|o| [o[0] / mesh.l0, o[1] / mesh.l0, o[2] / mesh.l0])
                    .unwrap_or(cs_detected.origin);
                let zax = z_axis.unwrap_or(cs_detected.zax);
                let cs = cs_from_origin_zaxis(org, zax);
                let port = CoaxPort {
                    port_number: port_num,
                    power: *power, er: *er,
                    ri: *ri / mesh.l0, ro: *ro / mesh.l0, cs,
                };
                eprintln!("  Port {}: coax, tag={}, Ri={:.3}mm, Ro={:.3}mm, er={:.2}, Z0={:.2}Ohm",
                    port_num, tag, ri * 1e3, ro * 1e3, er, port.port_z());
                Box::new(port)
            }
            FaceSpec::Lumped { tag, z0, l, c, direction, width, height, power } => {
                let (det_w, det_h) = lumped_port_dims(mesh, &tri_ids, direction);
                let w = if *width > 0.0 { *width / mesh.l0 } else { det_w };
                let h = if *height > 0.0 { *height / mesh.l0 } else { det_h };
                let port = LumpedPort {
                    port_number: port_num,
                    power: *power,
                    termination: LumpedElement { r: *z0, l: *l, c: *c, width: w, height: h },
                    direction: *direction,
                };
                eprintln!("  Port {}: lumped, tag={}, Z0={:.0}Ohm, dir=({:.1},{:.1},{:.1})",
                    port_num, tag, z0, direction[0], direction[1], direction[2]);
                Box::new(port)
            }
            FaceSpec::UserDefined { tag, e_field, power } => {
                let port = UserDefinedPort { port_number: port_num, e_field: *e_field };
                eprintln!("  Port {}: user_defined, tag={}, E=({:.3},{:.3},{:.3}), P={:.2}W",
                    port_num, tag, e_field[0], e_field[1], e_field[2], power);
                Box::new(port)
            }
            FaceSpec::Floquet { tag, scan_theta_deg, scan_phi_deg, mode_nr, power } => {
                // Only normal incidence is supported in the FD solver: oblique
                // scan needs periodic side-wall BCs and a complex mode field
                // (issue #14). Reject θ≠0 rather than silently returning wrong
                // S-parameters.
                if scan_theta_deg.abs() >= 1e-9 {
                    return Err(format!(
                        "FloquetPort: oblique scan (θ={scan_theta_deg:.3}°) is not supported in \
                         the frequency-domain solver, it needs periodic side-wall boundary \
                         conditions and a complex mode field (issue #14); only normal \
                         incidence (θ=0) is valid"
                    ));
                }
                let (cs_detected, det_w, det_h) = detect_rect_port(mesh, &tri_ids);
                let area = det_w * det_h;
                let port = FloquetPort {
                    port_number: port_num,
                    power: *power, area,
                    scan_theta: scan_theta_deg.to_radians(),
                    scan_phi: scan_phi_deg.to_radians(),
                    mode_nr: *mode_nr,
                    cs: cs_detected,
                };
                eprintln!("  Port {}: floquet, tag={}, mode={} ({}), theta={:.1}deg, phi={:.1}deg, A={:.2}mm^2",
                    port_num, tag, mode_nr,
                    if *mode_nr == 1 { "TE/S" } else { "TM/P" },
                    scan_theta_deg, scan_phi_deg, area * 1e6);
                Box::new(port)
            }
            FaceSpec::Pmc { .. } => unreachable!("PMC faces are skipped above"),
            FaceSpec::LumpedElement { tag, r, l, c, width, height, direction } => {
                let (det_w, det_h) = lumped_port_dims(mesh, &tri_ids, direction);
                // surf_z uses w/h as a ratio (scale-invariant); normalize both
                // anyway so the values stay consistent with the L₀-unit mesh.
                let w = if *width > 0.0 { *width / mesh.l0 } else { det_w };
                let h = if *height > 0.0 { *height / mesh.l0 } else { det_h };
                let bc = LumpedElement { r: *r, l: *l, c: *c, width: w, height: h };
                eprintln!("  LumpedElement: tag={}, R={:.2}Ohm, L={:.2e}H, C={:?}F, w={:.2}mm, h={:.2}mm",
                    tag, r, l, c, w * 1e3, h * 1e3);
                Box::new(bc)
            }
            FaceSpec::SurfaceImpedance { tag, conductivity, mur, er, thickness, two_sided, sheet, zs } => {
                let mut bc = match zs {
                    Some(zs) => SurfaceImpedance::from_zs(C64::new(zs[0], zs[1])),
                    None => SurfaceImpedance::from_conductivity(*conductivity),
                };
                bc.mur = *mur; bc.er = *er; bc.thickness = *thickness; bc.two_sided = *two_sided; bc.sheet = *sheet;
                if zs.is_none() && !*sheet && f_min.is_finite() && f_min > 0.0 {
                    let delta = bc.skin_depth(&crate::excitation::Excitation::new(f_min, mesh.l0));
                    bc.edges = crate::sibc_edge::EdgeProfile::build(mesh, &tri_ids, crate::sibc_edge::REACH * delta / mesh.l0);
                }
                eprintln!("  SurfaceImpedance: tag={}, sigma={:.2e}S/m, ur={:.2}, er={:.2}, t={:?}, two_sided={}, sheet={}",
                    tag, conductivity, mur, er, thickness, two_sided, sheet);
                Box::new(bc)
            }
            FaceSpec::Abc { tag } => {
                let abc = AbsorbingBoundary;
                eprintln!("  ABC: tag={}", tag);
                Box::new(abc)
            }
            FaceSpec::WaveNumerical { tag, f0, mode_index, kind, pec_tags, power } => {
                let f0 = f0.ok_or_else(|| format!(
                    "WavePort on tag {tag}: the frequency-domain backend needs f0 \
                     (the operating frequency of the 2D mode eigensolve)"))?;
                let pn = build_wave_numerical(
                    mesh, materials, &tri_ids, f0, *mode_index, *kind,
                    pec_tags, &pmc_tags, *power, port_num,
                );
                let Some(port) = pn else {
                    eprintln!("  WARNING: tag {}: wave_numerical eigensolve failed, skipping", tag);
                    continue;
                };
                eprintln!(
                    "  Port {}: wave_numerical, tag={}, f0={:.3}GHz, kind={:?}, mode_idx={}, n_eff={:.3}",
                    port_num, tag, f0 * 1e-9, kind, mode_index, port.n_eff,
                );
                Box::new(port)
            }
        };
        port_tris.push(tri_ids);
        ports.push(port);
    }

    Ok((ports, port_tris))
}

fn build_pec_tris(mesh: &Mesh, model: &Model) -> Vec<usize> {
    use std::collections::HashSet;
    let mut pec: HashSet<usize> = HashSet::new();
    for &tag in &model.pec_tags {
        pec.extend(mesh.tris_for_tag(tag).iter().copied());
    }

    // Default boundary condition: every EXTERIOR boundary face (one adjacent
    // tet) that carries no explicit port / BC becomes PEC (tangential E = 0).
    // A magnetic wall, the bare natural BC of the curl-curl form, is opt-in
    // via an explicit PMC. This makes a closed metal box the default and
    // removes the footgun where an untagged outer wall silently leaks (acts as
    // a magnetic wall). Interior faces (two adjacent tets) are never touched.
    let mut assigned: HashSet<usize> = pec.clone();
    for pc in &model.faces {
        assigned.extend(mesh.tris_for_tag(pc.tag()).iter().copied());
        if let FaceSpec::WaveNumerical { pec_tags, .. } = pc {
            for &t in pec_tags {
                assigned.extend(mesh.tris_for_tag(t).iter().copied());
            }
        }
    }
    for t in 0..mesh.n_tris() {
        if mesh.tri_to_tet[t][1] == usize::MAX && !assigned.contains(&t) {
            pec.insert(t);
        }
    }

    pec.into_iter().collect()
}

fn build_materials(mesh: &Mesh, model: &Model) -> Vec<Material> {
    model.materials.iter().map(|mc| {
        let tet_indices = mesh
            .vtag_to_tet
            .get(&mc.volume_tag).cloned()
            .unwrap_or_default();
        if tet_indices.is_empty() {
            eprintln!("  WARNING: volume tag {} has no tets", mc.volume_tag);
        } else {
            eprintln!("  Material: tag={}, er={:.2}, ur={:.2}, tand={:.4}, cond={:.2e}, {} tets",
                mc.volume_tag, mc.er, mc.ur, mc.tand, mc.conductivity, tet_indices.len());
        }
        let dispersion = if let Some(d) = &mc.debye {
            materials::Dispersion::Debye {
                er_inf: d.er_inf, er_static: d.er_static, tau_s: d.tau_s,
            }
        } else if let Some(d) = &mc.drude {
            materials::Dispersion::Drude {
                er_inf: d.er_inf, plasma_freq_hz: d.plasma_freq_hz, damping_freq_hz: d.damping_freq_hz,
            }
        } else {
            materials::Dispersion::None
        };
        if dispersion.is_dispersive() {
            eprintln!("    (dispersive: er(f) recomputed per frequency)");
        }
        Material {
            er: mc.er, ur: mc.ur, tand: mc.tand, cond: mc.conductivity,
            cond_diag: mc.cond_diag,
            tet_indices,
            er_diag: mc.er_diag,
            ur_diag: mc.ur_diag,
            dispersion,
        }
    }).collect()
}

fn build_pml_regions(mesh: &Mesh, model: &Model) -> Vec<PmlRegion> {
    model.pml.iter().map(|pc| {
        let tet_indices = mesh
            .vtag_to_tet
            .get(&pc.volume_tag).cloned()
            .unwrap_or_default();
        if tet_indices.is_empty() {
            eprintln!("  WARNING: PML volume tag {} has no tets", pc.volume_tag);
        } else {
            eprintln!("  PML: tag={}, dir=({:.0},{:.0},{:.0}), inner={:.3}m, t={:.3}m, n={:.1}, delta_max={:.1}, {} tets",
                pc.volume_tag, pc.direction[0], pc.direction[1], pc.direction[2],
                pc.inner_face, pc.thickness, pc.exponent, pc.delta_max, tet_indices.len());
        }
        PmlRegion {
            tet_indices,
            er_base: pc.er_base,
            ur_base: pc.ur_base,
            direction: pc.direction,
            // `inner_face` is a coordinate and `thickness` a length; the stretch
            // profile is evaluated against the lever-④ normalized node
            // coordinates, so both must be divided by L0 to stay consistent.
            // `u = (coord − inner_face)/thickness` is a length ratio, so the
            // corrected stretch is scale-invariant.
            inner_face: pc.inner_face / mesh.l0,
            thickness: pc.thickness / mesh.l0,
            exponent: pc.exponent,
            delta_max: pc.delta_max,
        }
    }).collect()
}

/// One scalar per tet from its material, `default` where none applies.
fn per_tet(materials: &[Material], n_tets: usize, default: f64, f: impl Fn(&Material) -> f64) -> Vec<f64> {
    let mut out = vec![default; n_tets];
    for mat in materials {
        let v = f(mat);
        for &ti in &mat.tet_indices {
            out[ti] = v;
        }
    }
    out
}

/// A material's scalar εr: the mean of a diagonal tensor.
fn er_scalar(m: &Material) -> f64 {
    m.er_diag.map_or(m.er, |[a, b, c]| (a + b + c) / 3.0)
}

/// A material's scalar μr: the mean of a diagonal tensor.
fn mur_scalar(m: &Material) -> f64 {
    m.ur_diag.map_or(m.ur, |[a, b, c]| (a + b + c) / 3.0)
}

/// Run the 2D port-face eigensolve and wrap the dominant mode as a
/// `NumericalWavePort`. Picks scalar TE/TM or full-vector hybrid based on
/// `mode_kind`. Returns `None` if the solve fails or yields fewer than
/// `mode_index + 1` modes.
fn build_wave_numerical(
    mesh: &Mesh,
    materials: &[Material],
    tri_ids: &[usize],
    f0: f64,
    mode_index: usize,
    kind: WaveKind,
    pec_tags: &[i32],
    pmc_tags: &[i32],
    power: f64,
    port_num: usize,
) -> Option<NumericalWavePort> {
    let pec = (!pec_tags.is_empty()).then(|| mesh.nodes_on_tags(pec_tags));
    let pmc = (!pmc_tags.is_empty()).then(|| mesh.nodes_on_tags(pmc_tags));
    let k0 = crate::excitation::Excitation::new(f0, mesh.l0).k0;
    let solve = match kind {
        WaveKind::Te => PortSolve::Scalar(ModeKind::Te),
        WaveKind::Tm => PortSolve::Scalar(ModeKind::Tm),
        WaveKind::Vector => PortSolve::Vector,
    };
    let eps_per_tet = per_tet(materials, mesh.n_tets(), 1.0, er_scalar);
    let (nm, n_eff) = solve_port_face(
        mesh, tri_ids, solve, mode_index, &eps_per_tet, k0, pec.as_deref(), pmc.as_deref(),
    )?;
    let is_vector = solve == PortSolve::Vector;
    let face_tris: Vec<[usize; 3]> = tri_ids.iter().map(|&t| mesh.tris[t]).collect();
    Some(NumericalWavePort::new(
        port_num,
        power,
        nm,
        n_eff,
        is_vector,
        &mesh.nodes,
        &face_tris,
    ))
}
