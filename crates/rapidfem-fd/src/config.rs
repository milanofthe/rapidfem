// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! TOML configuration file parsing for CLI.

use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub mesh: MeshConfig,
    #[serde(default)]
    pub frequency: FrequencyConfig,
    #[serde(default)]
    pub ports: Vec<PortConfig>,
    #[serde(default)]
    pub materials: Vec<MaterialConfig>,
    pub pec: PecConfig,
    #[serde(default)]
    pub solver: SolverConfig,
    #[serde(default)]
    pub adaptive: Option<AdaptiveConfig>,
    #[serde(default)]
    pub eigenmode: Option<EigenmodeConfig>,
    #[serde(default)]
    pub pml: Vec<PmlConfig>,
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Deserialize)]
pub struct AdaptiveConfig {
    #[serde(default = "default_theta")]
    pub theta: f64,
    #[serde(default = "default_refinement_ratio")]
    pub refinement_ratio: f64,
}

fn default_theta() -> f64 { 0.5 }
fn default_refinement_ratio() -> f64 { 0.5 }

#[derive(Deserialize)]
pub struct EigenmodeConfig {
    pub target_frequency: f64,
    #[serde(default = "default_n_modes")]
    pub n_modes: usize,
}

fn default_n_modes() -> usize { 6 }

#[derive(Deserialize)]
pub struct MeshConfig {
    pub file: String,
}

#[derive(Deserialize, Default)]
pub struct FrequencyConfig {
    #[serde(default)]
    pub values: Vec<f64>,
    #[serde(default)]
    pub range: Vec<f64>,
}

impl FrequencyConfig {
    pub fn frequencies(&self) -> Vec<f64> {
        if !self.values.is_empty() {
            self.values.clone()
        } else if self.range.len() == 3 {
            let (start, stop, n) = (self.range[0], self.range[1], self.range[2] as usize);
            if n <= 1 { return vec![start]; }
            (0..n).map(|i| start + (stop - start) * i as f64 / (n - 1) as f64).collect()
        } else {
            vec![10.0e9]
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum PortConfig {
    #[serde(rename = "rectangular")]
    Rectangular {
        tag: i32,
        #[serde(default)]
        width: f64,
        #[serde(default)]
        height: f64,
        #[serde(default = "default_mode")]
        mode: [usize; 2],
        #[serde(default = "default_one")]
        er: f64,
        #[serde(default = "default_one")]
        power: f64,
    },
    /// Floquet plane-wave port for unit-cell simulations. NOTE: a full Floquet sim also
    /// requires periodic BCs on side walls (not yet implemented); see `FloquetPort` doc.
    /// Mode = 1 (TE/S-pol) or 2 (TM/P-pol). At normal incidence (θ=0) the implementation
    /// is exact; at oblique angles the transverse phase factor is dropped (approximation).
    #[serde(rename = "floquet")]
    Floquet {
        tag: i32,
        /// Scan θ in degrees from port normal. Default 0 (normal incidence).
        #[serde(default)]
        scan_theta_deg: f64,
        /// Scan φ in degrees. Default 0.
        #[serde(default)]
        scan_phi_deg: f64,
        /// Polarization mode: 1 = TE (S), 2 = TM (P)
        #[serde(default = "default_floquet_mode")]
        mode_nr: u32,
        #[serde(default = "default_one")]
        er: f64,
        #[serde(default = "default_one")]
        power: f64,
    },
    /// User-defined port with a uniform constant E vector mode (covers the parallel-plate
    /// TEM case directly). For more elaborate spatial modes, instantiate `UserDefinedPort`
    /// programmatically via the Rust API rather than TOML.
    #[serde(rename = "user_defined")]
    UserDefined {
        tag: i32,
        /// Constant E-field vector across the port face (V/m, but normalized, magnitude is set by `power`)
        e_field: [f64; 3],
        #[serde(default = "default_one")]
        power: f64,
    },
    #[serde(rename = "coax")]
    Coax {
        tag: i32,
        /// Inner conductor radius (m)
        ri: f64,
        /// Outer conductor radius (m)
        ro: f64,
        /// Coax center on the port face (m). If omitted, auto-detected from the face centroid.
        #[serde(default)]
        origin: Option<[f64; 3]>,
        /// Axial direction (propagation direction, normal to port face). Auto-detected if omitted.
        #[serde(default)]
        z_axis: Option<[f64; 3]>,
        /// Dielectric inside the coax (default 1.0 = air)
        #[serde(default = "default_one")]
        er: f64,
        #[serde(default = "default_one")]
        power: f64,
    },
    #[serde(rename = "lumped")]
    Lumped {
        tag: i32,
        /// Reference resistance R (Ω); S-parameters normalise to this.
        #[serde(default = "default_z0")]
        z0: f64,
        /// Series inductance L (H) of the port termination; 0 ⇒ none.
        #[serde(default)]
        l: f64,
        /// Series capacitance C (F) of the port termination; None ⇒ none.
        #[serde(default)]
        c: Option<f64>,
        direction: [f64; 3],
        #[serde(default)]
        width: f64,
        #[serde(default)]
        height: f64,
        #[serde(default = "default_one")]
        power: f64,
    },
    #[serde(rename = "abc")]
    Abc {
        tag: i32,
    },
    /// Perfect magnetic conductor, natural BC (n × H = 0). No-op during assembly.
    /// Useful when users want to mark a surface explicitly as PMC for documentation.
    #[serde(rename = "pmc")]
    Pmc { tag: i32 },
    /// Lumped element (R, L, C in series) load on a surface. width/height define the
    /// surface-impedance scaling: surfZ = (R + jωL + 1/(jωC)) * width/height.
    #[serde(rename = "lumped_element")]
    LumpedElement {
        tag: i32,
        #[serde(default)]
        r: f64,
        #[serde(default)]
        l: f64,
        #[serde(default)]
        c: Option<f64>,
        /// Width (orthogonal to field direction); auto-detected if 0
        #[serde(default)]
        width: f64,
        /// Height (along field direction); auto-detected if 0
        #[serde(default)]
        height: f64,
        /// Field direction unit vector, used for auto width/height detection
        #[serde(default = "default_z_dir")]
        direction: [f64; 3],
    },
    /// Numerical wave port: 2D mode eigensolve on the port-face triangulation.
    ///
    /// The mode profile is computed once at the user-supplied operating
    /// frequency `f0` by `rapidfem_core::port_eigen::{solve_modes, solve_vector_modes}`
    /// and cached. Three modes:
    /// - default (`mode_kind = "auto"` or omitted): full-vector hybrid for an
    ///   inhomogeneous cross-section (substrate + air), quasi-TEM microstrip.
    /// - `mode_kind = "te"` / `"tm"`: scalar Helmholtz on a homogeneously
    ///   filled hollow guide (rectangular / ridged / arbitrary shape).
    ///
    /// `mode_index` selects the n-th mode ordered by descending n_eff (vector
    /// solve) or descending cutoff (scalar solve). `pec_tags` lists internal
    /// PEC surfaces (a microstrip trace cutting through the port face); nodes
    /// on those tags get the `tangential E = 0` constraint.
    ///
    /// Dispersion approximation: β(k0) = n_eff(f0) · k0 (vector path), β(k0) =
    /// sqrt(k0² - k_c²) (scalar path). For broadband sweeps far from f0 the
    /// vector path's β drifts; re-solving per frequency is a follow-up.
    #[serde(rename = "wave_numerical")]
    WaveNumerical {
        tag: i32,
        /// Operating frequency for the mode solve (Hz).
        f0: f64,
        /// Mode index, 0 = fundamental (descending n_eff / cutoff).
        #[serde(default)]
        mode_index: usize,
        /// "auto" / "vector", full hybrid solve (default). "te" / "tm",
        /// scalar Helmholtz on the homogeneously-filled cross-section.
        #[serde(default = "default_wave_kind")]
        mode_kind: String,
        /// Physical-group tags whose nodes are internal PEC (e.g. a
        /// microstrip trace dividing the port face). Outer-boundary PEC is
        /// inferred automatically.
        #[serde(default)]
        pec_tags: Vec<i32>,
        #[serde(default = "default_one")]
        power: f64,
    },
    /// Surface impedance (lossy conductor wall). Either supply σ (S/m) for skin-depth-based
    /// impedance, or `zs` (real+imag Ω/sq) for a frequency-independent constant.
    #[serde(rename = "surface_impedance")]
    SurfaceImpedance {
        tag: i32,
        #[serde(default)]
        conductivity: f64,
        #[serde(default = "default_one")]
        mur: f64,
        #[serde(default = "default_one")]
        er: f64,
        #[serde(default)]
        thickness: Option<f64>,
        /// Explicit surface impedance [re, im] in Ω/sq (overrides conductivity if present).
        #[serde(default)]
        zs: Option<[f64; 2]>,
    },
}

impl PortConfig {
    /// Physical-group tag of the face this port / boundary condition occupies.
    /// Every variant carries the face tag as its first field; used to know
    /// which boundary faces are "assigned" when defaulting the rest to PEC.
    pub fn tag(&self) -> i32 {
        match self {
            PortConfig::Rectangular { tag, .. }
            | PortConfig::Floquet { tag, .. }
            | PortConfig::UserDefined { tag, .. }
            | PortConfig::Coax { tag, .. }
            | PortConfig::Lumped { tag, .. }
            | PortConfig::Abc { tag }
            | PortConfig::Pmc { tag }
            | PortConfig::LumpedElement { tag, .. }
            | PortConfig::WaveNumerical { tag, .. }
            | PortConfig::SurfaceImpedance { tag, .. } => *tag,
        }
    }
}

/// Perfectly Matched Layer region. Tets in `volume_tag` get coordinate-stretched
/// anisotropic ε/μ tensors evaluated at each tet centroid. `direction` selects which
/// face the layer absorbs (e.g. [1,0,0] for the +x boundary, [0,0,-1] for -z).
/// `inner_face` is the coordinate of the PML's inner boundary along that axis;
/// `thickness` extends outward from there.
#[derive(Deserialize)]
pub struct PmlConfig {
    pub volume_tag: i32,
    pub direction: [f64; 3],
    pub inner_face: f64,
    pub thickness: f64,
    #[serde(default = "default_one")]
    pub er_base: f64,
    #[serde(default = "default_one")]
    pub ur_base: f64,
    #[serde(default = "default_pml_exponent")]
    pub exponent: f64,
    #[serde(default = "default_pml_delta_max")]
    pub delta_max: f64,
}

fn default_pml_exponent() -> f64 { 1.5 }
fn default_pml_delta_max() -> f64 { 8.0 }

#[derive(Deserialize)]
pub struct MaterialConfig {
    pub volume_tag: i32,
    #[serde(default = "default_one")]
    pub er: f64,
    #[serde(default = "default_one")]
    pub ur: f64,
    #[serde(default)]
    pub tand: f64,
    #[serde(default)]
    pub conductivity: f64,
    /// Optional diagonal εr anisotropy [εxx, εyy, εzz]; overrides scalar `er`.
    #[serde(default)]
    pub er_diag: Option<[f64; 3]>,
    /// Optional diagonal μr anisotropy [μxx, μyy, μzz]; overrides scalar `ur`.
    #[serde(default)]
    pub ur_diag: Option<[f64; 3]>,
    /// Debye dispersion: ε(ω) = ε∞ + (εs - ε∞)/(1 + jωτ).
    #[serde(default)]
    pub debye: Option<DebyeConfig>,
    /// Drude dispersion: ε(ω) = ε∞ - ωp²/(ω² + jγω).
    #[serde(default)]
    pub drude: Option<DrudeConfig>,
}

#[derive(Deserialize)]
pub struct DebyeConfig {
    /// ε∞ (high-frequency permittivity)
    pub er_inf: f64,
    /// εs (static / low-frequency permittivity)
    pub er_static: f64,
    /// Relaxation time τ in seconds
    pub tau_s: f64,
}

#[derive(Deserialize)]
pub struct DrudeConfig {
    /// ε∞ (background permittivity, often 1)
    #[serde(default = "default_one")]
    pub er_inf: f64,
    /// Plasma frequency in Hz (NOT angular)
    pub plasma_freq_hz: f64,
    /// Collision/damping frequency in Hz (NOT angular)
    pub damping_freq_hz: f64,
}

#[derive(Deserialize)]
pub struct PecConfig {
    pub tags: Vec<i32>,
}

#[derive(Deserialize, Default)]
pub struct SolverConfig {
    #[serde(default = "default_solver_prefer")]
    pub prefer: String,
}

#[derive(Deserialize, Default)]
pub struct OutputConfig {
    #[serde(default)]
    pub touchstone: Option<String>,
    #[serde(default = "default_z0")]
    pub z0: f64,
    #[serde(default)]
    pub vtk: Option<String>,
    #[serde(default)]
    pub farfield: Option<String>,
    /// Physical tag of the NFFT surface (defaults to ABC tag)
    #[serde(default)]
    pub nfft_tag: Option<i32>,
    /// Optional CSV path for group delay τ_g = -dφ/dω of S-parameters.
    #[serde(default)]
    pub group_delay: Option<String>,
}

fn default_mode() -> [usize; 2] { [1, 0] }
fn default_one() -> f64 { 1.0 }
fn default_z0() -> f64 { 50.0 }
fn default_solver_prefer() -> String { "auto".to_string() }
fn default_z_dir() -> [f64; 3] { [0.0, 0.0, 1.0] }
fn default_floquet_mode() -> u32 { 1 }
fn default_wave_kind() -> String { "auto".to_string() }

pub fn load_config(path: &str) -> Result<Config, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path, e))?;
    parse_config(&content).map_err(|e| format!("{}: {}", path, e))
}

/// Parse a config from a TOML string. Used by Python and WASM bindings to avoid file I/O.
pub fn parse_config(toml_str: &str) -> Result<Config, String> {
    toml::from_str(toml_str).map_err(|e| format!("Cannot parse TOML: {}", e))
}
