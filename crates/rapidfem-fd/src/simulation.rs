// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! High-level Simulation API, owns a mesh + parsed config and exposes callable
//! methods for sweep, eigenmode, and far-field. The single entry point used by
//! both the CLI (main.rs), the Python bindings (PyO3), and the WASM wrapper.
//!
//! Construction is split from execution so that callers can inspect/modify the
//! pre-built ports and materials before solving.

use num_complex::Complex64 as C64;

use crate::basis::Nedelec2Basis;
use crate::config::{Config, PortConfig};
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
use rapidfem_core::port_eigen::{
    solve_modes, solve_vector_modes, ModeKind, NumericalMode, PortMesh2D,
};

/// Result of a frequency sweep.
pub struct SweepResult {
    pub frequencies: Vec<f64>,
    /// S-parameters: `[freq_idx][port_obs][port_exc]`. Only driven ports.
    pub sparams: Vec<Vec<Vec<C64>>>,
    /// FEM E-field solutions: `[freq_idx][port_exc][dof]`.
    pub solutions: Vec<Vec<Vec<C64>>>,
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

/// Simulation context: a mesh + parsed config + pre-built BC objects.
pub struct Simulation {
    pub mesh: Mesh,
    pub basis: Nedelec2Basis,
    pub config: Config,
    pub ports: Vec<Box<dyn Port>>,
    pub port_tris: Vec<Vec<usize>>,
    pub pec_tris: Vec<usize>,
    pub materials: Vec<Material>,
    pub pml_regions: Vec<PmlRegion>,
}

impl Simulation {
    /// Build a `Simulation` from in-memory mesh bytes and a TOML config string.
    /// Boundary-friendly entry point (no std::fs use), suitable for Python / WASM bindings.
    pub fn from_bytes(mesh_bytes: &[u8], config_toml: &str) -> Result<Self, String> {
        let mesh = crate::mesh_io::parse_mesh_bytes(mesh_bytes)?;
        let config = crate::config::parse_config(config_toml)?;
        Ok(Self::new(mesh, config))
    }

    /// Build a `Simulation` from an owned mesh and a parsed config. All BC objects
    /// (ports, PEC, materials, PML, lumped integration lines) are constructed up-front.
    pub fn new(mesh: Mesh, config: Config) -> Self {
        let mut mesh = mesh;
        // Lever ④: non-dimensionalize the geometry to O(1) coordinates so the
        // assembly and its absolute tolerances are unit-/scale-invariant. The
        // transform is reversed at the output boundary (field/coord rescaling).
        // `RAPIDFEM_NO_NORMALIZE` keeps physical units (l0 = 1) — used to prove
        // the normalized path is bit-identical on the solver's outputs.
        if std::env::var_os("RAPIDFEM_NO_NORMALIZE").is_none() {
            let l0 = mesh.normalize_characteristic_length();
            eprintln!("  Geometry normalized: L0 = {:.6e} m (mean edge length)", l0);
        }
        let basis = Nedelec2Basis::new(&mesh);
        eprintln!(
            "RapidFEM - {} tets, {} DOFs",
            mesh.n_tets(),
            basis.n_field
        );

        // Materials before ports so `wave_numerical` can consult per-tet ε_r
        // when running the vector-hybrid mode solve on the port face.
        let materials = build_materials(&mesh, &config);
        let (ports, port_tris) = build_ports(&mesh, &config, &materials);
        let pec_tris = build_pec_tris(&mesh, &config);
        let pml_regions = build_pml_regions(&mesh, &config);

        Simulation {
            mesh,
            basis,
            config,
            ports,
            port_tris,
            pec_tris,
            materials,
            pml_regions,
        }
    }

    fn ports_dyn(&self) -> Vec<&dyn Port> {
        self.ports.iter().map(|b| b.as_ref()).collect()
    }

    fn port_tris_slices(&self) -> Vec<&[usize]> {
        self.port_tris.iter().map(|v| v.as_slice()).collect()
    }

    fn frequencies(&self) -> Vec<f64> {
        self.config.frequency.frequencies()
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

    /// For a single frequency's solution vector, return |E| (V/m) averaged
    /// at every mesh node by sampling the Nedelec-2 basis in each tet that
    /// contains the node and averaging the resulting magnitudes.
    /// Returned `Vec<f32>` has length `mesh.n_nodes()`.
    pub fn nodal_field_magnitudes(&self, solution: &[C64]) -> Vec<f32> {
        let n_nodes = self.mesh.n_nodes();
        let mut sum = vec![0.0f64; n_nodes];
        let mut count = vec![0u32; n_nodes];
        // Sample at each tet's centroid; assign that magnitude to each of its
        // 4 vertices. Cheap, gives a smooth nodal field via averaging.
        for ti in 0..self.mesh.n_tets() {
            let tet = &self.mesh.tets[ti];
            let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
            for k in 0..4 {
                let p = self.mesh.nodes[tet[k]];
                cx += p[0]; cy += p[1]; cz += p[2];
            }
            cx /= 4.0; cy /= 4.0; cz /= 4.0;
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(
                &self.mesh, &self.basis, solution, ti, cx, cy, cz,
            );
            // |E| on the normalized mesh is L₀·|E_phys| → divide by L₀.
            let mag = (ex.norm_sqr() + ey.norm_sqr() + ez.norm_sqr()).sqrt() / self.mesh.l0;
            for k in 0..4 {
                sum[tet[k]] += mag;
                count[tet[k]] += 1;
            }
        }
        sum.into_iter()
            .zip(count.into_iter())
            .map(|(s, c)| if c == 0 { 0.0 } else { (s / c as f64) as f32 })
            .collect()
    }

    /// Per-node phasor terms `(A, B, C)` for animated `|E(t)|²` rendering:
    ///
    ///   A = |Re(E)|² = Re_x² + Re_y² + Re_z²
    ///   B = |Im(E)|² = Im_x² + Im_y² + Im_z²
    ///   C = Re(E) · Im(E)   (real dot product)
    ///
    /// Then `|E(x,t)|² = A·cos²(ωt) + B·sin²(ωt) − 2·C·sin(ωt)·cos(ωt)`,
    /// which lets the viewer's shader modulate one uniform (phase) and
    /// render a propagating wave without any new field evaluations.
    pub fn nodal_field_phasor_terms(&self, solution: &[C64]) -> Vec<[f32; 3]> {
        let n_nodes = self.mesh.n_nodes();
        let mut sum = vec![[0.0f64; 3]; n_nodes];
        let mut count = vec![0u32; n_nodes];
        for ti in 0..self.mesh.n_tets() {
            let tet = &self.mesh.tets[ti];
            let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
            for k in 0..4 {
                let p = self.mesh.nodes[tet[k]];
                cx += p[0]; cy += p[1]; cz += p[2];
            }
            cx /= 4.0; cy /= 4.0; cz /= 4.0;
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(
                &self.mesh, &self.basis, solution, ti, cx, cy, cz,
            );
            // These are quadratic in E (|E|²), and E on the normalized mesh is
            // L₀·E_phys, so divide by L₀² for physical |E|² units.
            let inv_l02 = 1.0 / (self.mesh.l0 * self.mesh.l0);
            let a = (ex.re * ex.re + ey.re * ey.re + ez.re * ez.re) * inv_l02;
            let b = (ex.im * ex.im + ey.im * ey.im + ez.im * ez.im) * inv_l02;
            let c = (ex.re * ex.im + ey.re * ey.im + ez.re * ez.im) * inv_l02;
            for k in 0..4 {
                sum[tet[k]][0] += a;
                sum[tet[k]][1] += b;
                sum[tet[k]][2] += c;
                count[tet[k]] += 1;
            }
        }
        sum.into_iter().zip(count.into_iter()).map(|(s, c)| {
            if c == 0 { [0.0, 0.0, 0.0] }
            else {
                let inv = 1.0 / c as f64;
                [(s[0] * inv) as f32, (s[1] * inv) as f32, (s[2] * inv) as f32]
            }
        }).collect()
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
            results = crate::assembly::frequency_sweep_with_pml(
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

        Ok(SweepResult {
            frequencies,
            sparams: all_sparams,
            solutions,
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
        let eps_tet = per_tet_eps_scalar(&self.materials, n_tets);
        let mut mur_tet = vec![1.0_f64; n_tets];
        for mat in &self.materials {
            let ur = match mat.ur_diag {
                Some([a, b, c]) => (a + b + c) / 3.0,
                None => mat.ur,
            };
            for &ti in &mat.tet_indices { mur_tet[ti] = ur; }
        }
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
                let s = if let (true, Some((dir, _z0, v_inc)), Some(height)) = (
                    port_dyn[obs_pi].is_lumped(),
                    port_dyn[obs_pi].lumped_voltage_params(),
                    port_dyn[obs_pi].port_height(),
                ) {
                    // Area-averaged mode projection V = (l/A)∫E·l̂ dS — robust
                    // for tall / non-TEM ports (derivations/lumped_port/).
                    let obs_tris: Vec<[usize; 3]> = port_tri_refs[obs_pi]
                        .iter()
                        .map(|&ti| self.mesh.tris[ti])
                        .collect();
                    sparam_voltage_surface(
                        &self.mesh.nodes, &obs_tris, dir, height, v_inc, active, &fieldf, 4,
                    )
                } else {
                    let obs_tris: Vec<[usize; 3]> = port_tri_refs[obs_pi]
                        .iter()
                        .map(|&ti| self.mesh.tris[ti])
                        .collect();
                    sparam_waveport(&self.mesh.nodes, &obs_tris, port_dyn[obs_pi], &exc, active, &fieldf, &weight, 4)
                };
                freq_s[obs_idx][exc_idx] = s;
            }
        }
        freq_s
    }

    /// Run an eigenmode analysis (requires `config.eigenmode` to be set).
    pub fn run_eigenmode(&self) -> Result<Vec<Eigenmode>, String> {
        let eig = self.config.eigenmode.as_ref()
            .ok_or("config.eigenmode not set")?;
        crate::eigenmode::solve_eigenmode(
            &self.mesh,
            &self.basis,
            &self.pec_tris,
            self.materials_opt(),
            eig.target_frequency,
            eig.n_modes,
        )
    }

    /// Radiation efficiency η = 1 − Σ |S_i1|² for the first driven port at the given freq.
    /// Used as the gain-offset for far-field. Returns None if no driven ports / no S data.
    pub fn radiation_efficiency(&self, result: &SweepResult, freq_idx: usize) -> Option<f64> {
        let s = result.sparams.get(freq_idx)?;
        if s.is_empty() || s[0].is_empty() {
            return None;
        }
        let s11_sum_sq: f64 = s.iter().filter_map(|row| row.first()).map(|s| s.norm_sqr()).sum();
        Some((1.0 - s11_sum_sq).clamp(0.0, 1.0))
    }

    /// Monk-style residual a-posteriori error indicator per tet for a given
    /// `(freq_idx, port_idx)` solution. Returns the full estimate (η per tet,
    /// volume and face contributions, total, marked subset from Dörfler at
    /// `theta`). Intended for diagnostics, the same indicator drives the
    /// adaptive loop in `--adaptive` sweeps.
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
        let n_tets = self.mesh.n_tets();
        let (er_tensors, _) = if self.materials.is_empty() {
            let id: [[C64; 3]; 3] = [
                [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
                [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
            ];
            (vec![id; n_tets], vec![id; n_tets])
        } else {
            materials::build_material_tensors(n_tets, &self.materials, freq)
        };
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

    /// Common interior, node → first-adjacent tet → barycentric eval.
    /// Both ``field_at_nodes`` and ``eigenmode_field_at_nodes`` route here
    /// so the per-node node→tet table is built the same way in both paths.
    fn eval_dofs_at_nodes(&self, solution: &[C64]) -> Vec<C64> {
        let n_nodes = self.mesh.n_nodes();

        // Node → adjacent tet (first one wins, matches vtk_export behaviour).
        let mut node_to_tet = vec![usize::MAX; n_nodes];
        for (itet, tet) in self.mesh.tets.iter().enumerate() {
            for &ni in tet {
                if node_to_tet[ni] == usize::MAX {
                    node_to_tet[ni] = itet;
                }
            }
        }

        // Lever ④: the field reconstructed on the L₀-normalized mesh is L₀·E_phys
        // (the Nédélec basis is scale-invariant), so divide by L₀ for physical
        // V/m. A no-op when l0 = 1.
        let inv_l0 = C64::from(1.0 / self.mesh.l0);
        let mut out: Vec<C64> = Vec::with_capacity(3 * n_nodes);
        for ni in 0..n_nodes {
            let tet_idx = node_to_tet[ni];
            if tet_idx == usize::MAX {
                out.extend_from_slice(&[C64::new(0.0, 0.0); 3]);
                continue;
            }
            let p = self.mesh.nodes[ni];
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(
                &self.mesh, &self.basis, solution, tet_idx, p[0], p[1], p[2],
            );
            out.push(ex * inv_l0);
            out.push(ey * inv_l0);
            out.push(ez * inv_l0);
        }
        out
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
        let mut sigma = vec![0.0f64; self.mesh.n_tets()];
        let w_eps0 = omega * EPS0;
        for mat in &self.materials {
            if mat.cond == 0.0 && mat.tand == 0.0 { continue; }
            let s = w_eps0 * mat.er * mat.tand + mat.cond;
            for &ti in &mat.tet_indices {
                sigma[ti] = s;
            }
        }
        sigma
    }

    /// Per-tet relative permeability μ_r, default 1.0 where no material applies.
    fn per_tet_mur(&self) -> Vec<f64> {
        let mut mur = vec![1.0f64; self.mesh.n_tets()];
        for mat in &self.materials {
            for &ti in &mat.tet_indices {
                mur[ti] = mat.ur;
            }
        }
        mur
    }

    /// Build the node → adjacent-tet map (first tet wins). Shared by all the
    /// per-node samplers below so they pick the same tet at material interfaces.
    fn node_to_tet_map(&self) -> Vec<usize> {
        let n_nodes = self.mesh.n_nodes();
        let mut node_to_tet = vec![usize::MAX; n_nodes];
        for (itet, tet) in self.mesh.tets.iter().enumerate() {
            for &ni in tet {
                if node_to_tet[ni] == usize::MAX {
                    node_to_tet[ni] = itet;
                }
            }
        }
        node_to_tet
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
        let n_nodes = self.mesh.n_nodes();
        let node_to_tet = self.node_to_tet_map();
        let mut out: Vec<C64> = Vec::with_capacity(3 * n_nodes);
        for ni in 0..n_nodes {
            let tet_idx = node_to_tet[ni];
            if tet_idx == usize::MAX || sigma[tet_idx] == 0.0 {
                out.extend_from_slice(&[C64::new(0.0, 0.0); 3]);
                continue;
            }
            let p = self.mesh.nodes[ni];
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(
                &self.mesh, &self.basis, solution, tet_idx, p[0], p[1], p[2],
            );
            // J = σ·E_phys; E reconstructed on the normalized mesh is L₀·E_phys.
            let s = C64::from(sigma[tet_idx] / self.mesh.l0);
            out.push(ex * s);
            out.push(ey * s);
            out.push(ez * s);
        }
        Some(out)
    }

    /// Magnetic field H = ∇×E / (jωμ₀μ_r) at each mesh node, in (A/m).
    /// Returns `Vec<C64>` of length `3 · n_nodes` (interleaved Hx, Hy, Hz).
    /// Uses the analytic Nédélec-2 curl evaluated at the node position.
    pub fn h_field_at_nodes(&self, result: &SweepResult, freq_idx: usize, port_idx: usize) -> Option<Vec<C64>> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(port_idx))?;
        let freq = *result.frequencies.get(freq_idx)?;
        let omega = crate::excitation::Excitation::new(freq, self.mesh.l0).omega;
        let mur = self.per_tet_mur();
        let n_nodes = self.mesh.n_nodes();
        let node_to_tet = self.node_to_tet_map();
        let j = C64::new(0.0, 1.0);
        let mut out: Vec<C64> = Vec::with_capacity(3 * n_nodes);
        for ni in 0..n_nodes {
            let tet_idx = node_to_tet[ni];
            if tet_idx == usize::MAX {
                out.extend_from_slice(&[C64::new(0.0, 0.0); 3]);
                continue;
            }
            let p = self.mesh.nodes[ni];
            let curl = crate::interp::eval_curl_in_tet(
                &self.mesh, &self.basis, solution, tet_idx, p[0], p[1], p[2],
            );
            // ∇×E on the normalized mesh is L₀²·(∇×E)_phys (one L₀ from the basis
            // field, one from the normalized ∇), so H = ∇×E/(jωμ) needs /L₀².
            let denom = j * C64::from(omega * MU0 * mur[tet_idx] * self.mesh.l0 * self.mesh.l0);
            out.push(curl[0] / denom);
            out.push(curl[1] / denom);
            out.push(curl[2] / denom);
        }
        Some(out)
    }

    /// Compute the far-field at a given (freq_idx, exc_port_idx). NFFT surface = config.output.nfft_tag
    /// (auto-detected ABC tag if not specified). PEC surfaces from config.pec.tags are included to close
    /// the integration boundary.
    pub fn compute_farfield(
        &self,
        result: &SweepResult,
        freq_idx: usize,
        exc_port_idx: usize,
        n_theta: usize,
        n_phi: usize,
    ) -> Option<RadiationPattern> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(exc_port_idx))?;
        let nfft_tag = self.config.output.nfft_tag.unwrap_or_else(|| {
            for pc in &self.config.ports {
                if let PortConfig::Abc { tag, .. } = pc {
                    return *tag;
                }
            }
            2
        });
        let pec_nfft: Vec<usize> = self
            .config
            .pec
            .tags
            .iter()
            .flat_map(|&t| self.mesh.tris_for_tag(t).to_vec())
            .collect();
        // A face can carry both the NFFT tag and a PEC tag (e.g. a ground
        // plane the user also marked as part of the Huygens surface). Drop
        // those tris from the NFFT set so they aren't integrated twice; the
        // PEC pass already covers them (with M_s = 0, the correct treatment
        // of tangential E on a conductor).
        let pec_set: std::collections::HashSet<usize> = pec_nfft.iter().copied().collect();
        let nfft_tris: Vec<usize> = self
            .mesh
            .tris_for_tag(nfft_tag)
            .iter()
            .copied()
            .filter(|t| !pec_set.contains(t))
            .collect();
        if nfft_tris.is_empty() {
            return None;
        }
        let efficiency = self.radiation_efficiency(result, freq_idx);

        Some(crate::farfield::compute_farfield_full(
            &self.mesh,
            &self.basis,
            solution,
            &nfft_tris,
            &pec_nfft,
            result.frequencies[freq_idx],
            n_theta,
            n_phi,
            4,
            efficiency,
        ))
    }
}

// ============================================================================
// Construction helpers, extracted from main.rs's prior orchestration
// ============================================================================

fn build_ports(
    mesh: &Mesh,
    config: &Config,
    materials: &[Material],
) -> (Vec<Box<dyn Port>>, Vec<Vec<usize>>) {
    let mut ports: Vec<Box<dyn Port>> = Vec::new();
    let mut port_tris: Vec<Vec<usize>> = Vec::new();

    for pc in &config.ports {
        match pc {
            PortConfig::Rectangular { tag, width, height, mode, er, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping port", tag);
                    continue;
                }
                let (cs, det_w, det_h) = detect_rect_port(mesh, &tri_ids);
                // Config dims are physical lengths; the mesh (and det_*) are in
                // L₀ units, so normalize config dims to match (lever ④).
                let w = if *width > 0.0 { *width / mesh.l0 } else { det_w };
                let h = if *height > 0.0 { *height / mesh.l0 } else { det_h };
                let port_num = ports.len() + 1;
                let port = RectWaveguide {
                    port_number: port_num,
                    power: *power,
                    mode: (mode[0], mode[1]),
                    er: *er,
                    polarization: 1.0,
                    dims: (w, h),
                    cs,
                };
                eprintln!("  Port {}: rectangular, tag={}, TE{}{}, dims=({:.2}mm, {:.2}mm), er={:.1}",
                    port_num, tag, mode[0], mode[1], w * 1e3, h * 1e3, er);
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::Coax { tag, ri, ro, origin, z_axis, er, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping CoaxPort", tag);
                    continue;
                }
                let (cs_detected, _, _) = detect_rect_port(mesh, &tri_ids);
                // Config origin is a physical coordinate; normalize to L₀ units.
                let org = origin
                    .map(|o| [o[0] / mesh.l0, o[1] / mesh.l0, o[2] / mesh.l0])
                    .unwrap_or(cs_detected.origin);
                let zax = z_axis.unwrap_or(cs_detected.zax);
                let cs = cs_from_origin_zaxis(org, zax);
                let port_num = ports.len() + 1;
                let port = CoaxPort {
                    port_number: port_num,
                    power: *power, er: *er,
                    ri: *ri / mesh.l0, ro: *ro / mesh.l0, cs,
                };
                eprintln!("  Port {}: coax, tag={}, Ri={:.3}mm, Ro={:.3}mm, er={:.2}, Z0={:.2}Ohm",
                    port_num, tag, ri * 1e3, ro * 1e3, er, port.port_z());
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::Lumped { tag, z0, l, c, direction, width, height, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping port", tag);
                    continue;
                }
                let port_num = ports.len() + 1;
                let (det_w, det_h) = lumped_port_dims(mesh, &tri_ids, direction);
                let w = if *width > 0.0 { *width / mesh.l0 } else { det_w };
                let h = if *height > 0.0 { *height / mesh.l0 } else { det_h };
                let port = LumpedPort {
                    port_number: port_num,
                    power: *power,
                    z0: *z0,
                    l: *l,
                    c: *c,
                    width: w,
                    height: h,
                    direction: *direction,
                };
                eprintln!("  Port {}: lumped, tag={}, Z0={:.0}Ohm, dir=({:.1},{:.1},{:.1})",
                    port_num, tag, z0, direction[0], direction[1], direction[2]);
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::UserDefined { tag, e_field, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping UserDefined", tag);
                    continue;
                }
                let port_num = ports.len() + 1;
                let port = UserDefinedPort::from_constant(port_num, *power, *e_field);
                eprintln!("  Port {}: user_defined, tag={}, E=({:.3},{:.3},{:.3}), P={:.2}W",
                    port_num, tag, e_field[0], e_field[1], e_field[2], power);
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::Floquet { tag, scan_theta_deg, scan_phi_deg, mode_nr, er, power } => {
                // Only normal incidence is supported in the FD solver: oblique
                // scan needs periodic side-wall BCs and a complex mode field
                // (issue #14). Reject θ≠0 rather than silently returning wrong
                // S-parameters.
                assert!(
                    scan_theta_deg.abs() < 1e-9,
                    "FloquetPort: oblique scan (θ={:.3}°) is not yet supported in \
                     the frequency-domain solver — it requires periodic side-wall \
                     boundary conditions and a complex mode field (see issue #14). \
                     Only normal incidence (θ=0) is valid.",
                    scan_theta_deg
                );
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping FloquetPort", tag);
                    continue;
                }
                let (cs_detected, det_w, det_h) = detect_rect_port(mesh, &tri_ids);
                let area = det_w * det_h;
                let port_num = ports.len() + 1;
                let port = FloquetPort {
                    port_number: port_num,
                    power: *power, er: *er, area,
                    scan_theta: scan_theta_deg.to_radians(),
                    scan_phi: scan_phi_deg.to_radians(),
                    mode_nr: *mode_nr,
                    cs: cs_detected,
                };
                eprintln!("  Port {}: floquet, tag={}, mode={} ({}), theta={:.1}deg, phi={:.1}deg, A={:.2}mm^2",
                    port_num, tag, mode_nr,
                    if *mode_nr == 1 { "TE/S" } else { "TM/P" },
                    scan_theta_deg, scan_phi_deg, area * 1e6);
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::Pmc { tag } => {
                let tri_ids = mesh.tris_for_tag(*tag);
                eprintln!("  PMC: tag={}, {} triangles (natural BC)", tag, tri_ids.len());
            }
            PortConfig::LumpedElement { tag, r, l, c, width, height, direction } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping LumpedElement", tag);
                    continue;
                }
                let (det_w, det_h) = lumped_port_dims(mesh, &tri_ids, direction);
                // surf_z uses w/h as a ratio (scale-invariant); normalize both
                // anyway so the values stay consistent with the L₀-unit mesh.
                let w = if *width > 0.0 { *width / mesh.l0 } else { det_w };
                let h = if *height > 0.0 { *height / mesh.l0 } else { det_h };
                let bc = LumpedElement { r: *r, l: *l, c: *c, width: w, height: h };
                eprintln!("  LumpedElement: tag={}, R={:.2}Ohm, L={:.2e}H, C={:?}F, w={:.2}mm, h={:.2}mm",
                    tag, r, l, c, w * 1e3, h * 1e3);
                port_tris.push(tri_ids);
                ports.push(Box::new(bc));
            }
            PortConfig::SurfaceImpedance { tag, conductivity, mur, er, thickness, zs } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping SurfaceImpedance", tag);
                    continue;
                }
                let bc = if let Some(zs_arr) = zs {
                    let mut s = SurfaceImpedance::from_zs(C64::new(zs_arr[0], zs_arr[1]));
                    s.mur = *mur; s.er = *er; s.thickness = *thickness;
                    s
                } else {
                    let mut s = SurfaceImpedance::from_conductivity(*conductivity);
                    s.mur = *mur; s.er = *er; s.thickness = *thickness;
                    s
                };
                eprintln!("  SurfaceImpedance: tag={}, sigma={:.2e}S/m, ur={:.2}, er={:.2}, t={:?}",
                    tag, conductivity, mur, er, thickness);
                port_tris.push(tri_ids);
                ports.push(Box::new(bc));
            }
            PortConfig::Abc { tag } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping ABC", tag);
                    continue;
                }
                let abc = AbsorbingBoundary::new();
                eprintln!("  ABC: tag={}", tag);
                port_tris.push(tri_ids);
                ports.push(Box::new(abc));
            }
            PortConfig::WaveNumerical { tag, f0, mode_index, mode_kind, pec_tags, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping WaveNumerical", tag);
                    continue;
                }
                let port_num = ports.len() + 1;
                let pn = build_wave_numerical(
                    mesh, materials, &tri_ids, *f0, *mode_index, mode_kind,
                    pec_tags, *power, port_num,
                );
                match pn {
                    Some(port) => {
                        eprintln!(
                            "  Port {}: wave_numerical, tag={}, f0={:.3}GHz, kind={}, mode_idx={}, n_eff={:.3}",
                            port_num, tag, f0 * 1e-9, mode_kind, mode_index, port.n_eff,
                        );
                        port_tris.push(tri_ids);
                        ports.push(Box::new(port));
                    }
                    None => {
                        eprintln!("  WARNING: tag {}: wave_numerical eigensolve failed, skipping", tag);
                    }
                }
            }
        }
    }

    (ports, port_tris)
}

fn build_pec_tris(mesh: &Mesh, config: &Config) -> Vec<usize> {
    use std::collections::HashSet;
    let mut pec: HashSet<usize> = HashSet::new();
    for &tag in &config.pec.tags {
        pec.extend(mesh.tris_for_tag(tag).iter().copied());
    }

    // Default boundary condition: every EXTERIOR boundary face (one adjacent
    // tet) that carries no explicit port / BC becomes PEC (tangential E = 0).
    // A magnetic wall — the bare natural BC of the curl-curl form — is opt-in
    // via an explicit PMC. This makes a closed metal box the default and
    // removes the footgun where an untagged outer wall silently leaks (acts as
    // a magnetic wall). Interior faces (two adjacent tets) are never touched.
    let mut assigned: HashSet<usize> = pec.clone();
    for pc in &config.ports {
        assigned.extend(mesh.tris_for_tag(pc.tag()).iter().copied());
        if let PortConfig::WaveNumerical { pec_tags, .. } = pc {
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

fn build_materials(mesh: &Mesh, config: &Config) -> Vec<Material> {
    config.materials.iter().map(|mc| {
        let tet_indices = mesh
            .vtag_to_tet
            .get(&mc.volume_tag)
            .map(|v| v.clone())
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
            tet_indices,
            er_diag: mc.er_diag,
            ur_diag: mc.ur_diag,
            dispersion,
        }
    }).collect()
}

fn build_pml_regions(mesh: &Mesh, config: &Config) -> Vec<PmlRegion> {
    config.pml.iter().map(|pc| {
        let tet_indices = mesh
            .vtag_to_tet
            .get(&pc.volume_tag)
            .map(|v| v.clone())
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

/// Build a per-tet scalar relative permittivity from the materials list. Used
/// by `build_wave_numerical` to feed the vector eigensolver. Anisotropic
/// materials collapse to the mean of their diagonal, sufficient for the
/// shift-invert pivot in `solve_vector_modes`.
fn per_tet_eps_scalar(materials: &[Material], n_tets: usize) -> Vec<f64> {
    let mut eps = vec![1.0_f64; n_tets];
    for mat in materials {
        let er_scalar = match mat.er_diag {
            Some([a, b, c]) => (a + b + c) / 3.0,
            None => mat.er,
        };
        for &ti in &mat.tet_indices {
            eps[ti] = er_scalar;
        }
    }
    eps
}

/// Inward face normal toward the adjacent tet centroid (same construction as
/// CoaxPort / TD's wave_from_mesh_tag). Returns `None` if the face has no
/// adjacent tet (which would mean a boundary tri with no interior side).
fn face_inward_normal(mesh: &Mesh, t0: usize) -> Option<[f64; 3]> {
    let [v0, v1, v2] = mesh.tris[t0];
    let p0 = mesh.nodes[v0];
    let p1 = mesh.nodes[v1];
    let p2 = mesh.nodes[v2];
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let mut nrm = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (nrm[0].powi(2) + nrm[1].powi(2) + nrm[2].powi(2)).sqrt();
    if len < 1e-30 {
        return None;
    }
    for c in nrm.iter_mut() { *c /= len; }
    let tet = mesh.tri_to_tet[t0].iter().copied().find(|&x| x != usize::MAX)?;
    let mut centroid = [0.0_f64; 3];
    for &nd in &mesh.tets[tet] {
        for k in 0..3 { centroid[k] += mesh.nodes[nd][k] / 4.0; }
    }
    let inward = [centroid[0] - p0[0], centroid[1] - p0[1], centroid[2] - p0[2]];
    let dot = nrm[0] * inward[0] + nrm[1] * inward[1] + nrm[2] * inward[2];
    if dot < 0.0 {
        for c in nrm.iter_mut() { *c = -*c; }
    }
    Some(nrm)
}

/// Build a per-global-node boolean mask: `true` for nodes that lie on any of
/// the listed PEC physical groups. Used by `PortMesh2D::from_face` to mark
/// internal-conductor (e.g. microstrip trace) nodes as PEC inside the
/// cross-section eigensolve.
fn build_internal_pec_mask(mesh: &Mesh, pec_tags: &[i32]) -> Vec<bool> {
    let mut mask = vec![false; mesh.nodes.len()];
    for &tag in pec_tags {
        for &ti in mesh.tris_for_tag(tag) {
            for &v in &mesh.tris[ti] {
                mask[v] = true;
            }
        }
    }
    mask
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
    mode_kind: &str,
    pec_tags: &[i32],
    power: f64,
    port_num: usize,
) -> Option<NumericalWavePort> {
    let t0 = *tri_ids.first()?;
    let nrm = face_inward_normal(mesh, t0)?;
    let pec_mask = build_internal_pec_mask(mesh, pec_tags);
    let pec_opt = if pec_tags.is_empty() { None } else { Some(pec_mask.as_slice()) };

    let face_tris: Vec<[usize; 3]> = tri_ids.iter().map(|&t| mesh.tris[t]).collect();
    let pm = PortMesh2D::from_face(&mesh.nodes, &face_tris, nrm, pec_opt);

    let k0 = crate::excitation::Excitation::new(f0, mesh.l0).k0;

    // Per-face εr (size = face_tris.len()) is always needed: the vector
    // path uses it as the eigensolve weight, and the unified amplitude
    // normalisation in NumericalWavePort uses it for the Poynting-flux
    // integral. Scalar paths get a uniform-1.0 vector, they sit on
    // homogeneous-fill cross-sections by construction.
    let eps_per_tet = per_tet_eps_scalar(materials, mesh.n_tets());
    let eps_face: Vec<f64> = tri_ids
        .iter()
        .map(|&t| {
            mesh.tri_to_tet[t]
                .iter()
                .copied()
                .find(|&x| x != usize::MAX)
                .map(|e| eps_per_tet[e])
                .unwrap_or(1.0)
        })
        .collect();

    let kind_lc = mode_kind.to_lowercase();
    let (nm, n_eff, is_vector): (NumericalMode, f64, bool) = match kind_lc.as_str() {
        "te" => {
            let modes = solve_modes(&pm, ModeKind::Te, mode_index + 1);
            let mode = modes.get(mode_index)?;
            let kc = mode.k_c;
            let beta = (k0 * k0 - kc * kc).max(0.0).sqrt();
            let n_eff = if k0 > 0.0 { beta / k0 } else { 0.0 };
            (NumericalMode::from_scalar(pm, mode, ModeKind::Te), n_eff, false)
        }
        "tm" => {
            let modes = solve_modes(&pm, ModeKind::Tm, mode_index + 1);
            let mode = modes.get(mode_index)?;
            let kc = mode.k_c;
            let beta = (k0 * k0 - kc * kc).max(0.0).sqrt();
            let n_eff = if k0 > 0.0 { beta / k0 } else { 0.0 };
            (NumericalMode::from_scalar(pm, mode, ModeKind::Tm), n_eff, false)
        }
        "auto" | "vector" | "hybrid" => {
            let n_pec_nodes = pm.on_pec.iter().filter(|&&b| b).count();
            let n_boundary_nodes = pm.on_boundary.iter().filter(|&&b| b).count();
            eprintln!(
                "  wave_numerical[vector] tag={}: {} face tris, {} nodes, \
                 {} boundary + {} internal PEC, eps=[{:.2},{:.2}], k0={:.3}/m",
                tri_ids.len(), pm.tris.len(), pm.nodes.len(),
                n_boundary_nodes, n_pec_nodes,
                eps_face.iter().cloned().fold(f64::INFINITY, f64::min),
                eps_face.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                k0,
            );
            let modes = solve_vector_modes(&pm, &eps_face, k0, mode_index + 1);
            if modes.is_empty() {
                eprintln!("    -> solve_vector_modes returned 0 modes");
                return None;
            }
            eprintln!("    -> got {} mode(s), n_eff = {:?}",
                modes.len(),
                modes.iter().map(|m| m.n_eff).collect::<Vec<_>>(),
            );
            let mode = modes.get(mode_index)?;
            let n_eff = mode.n_eff;
            (NumericalMode::from_vector(pm, mode), n_eff, true)
        }
        other => {
            eprintln!(
                "  WARNING: wave_numerical: unknown mode_kind {:?}, expected one of \
                 'auto'/'vector'/'hybrid'/'te'/'tm'",
                other,
            );
            return None;
        }
    };

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
