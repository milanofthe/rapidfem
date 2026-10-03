// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The simulation model: which material fills which volume and which boundary
//! condition or port sits on which face, independent of the backend.
//!
//! The frequency-domain `Simulation` and the time-domain `MaxwellOperator`
//! are both built from a [`Model`] and a mesh. Volumes and faces are named by
//! their mesh tags (`Mesh::vtag_to_tet`, `Mesh::ftag_to_tri`), which is the
//! seam to the mesher: whatever produces the mesh also produces the tags.

/// Everything placed on the mesh.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub materials: Vec<MaterialSpec>,
    /// Ports and face boundary conditions other than PEC, in declaration
    /// order (the order fixes the port indices).
    pub faces: Vec<FaceSpec>,
    /// Faces that are perfect electric conductors. Boundary faces without any
    /// assignment are PEC as well.
    pub pec_tags: Vec<i32>,
    pub pml: Vec<PmlSpec>,
    /// Near-to-far-field surface (defaults to the ABC faces).
    pub far_field_tag: Option<i32>,
    /// Periodic face pairs `(a, b)`: `b` is `a` shifted, meshed alike.
    pub periodic: Vec<(i32, i32)>,
}

/// Bulk material of the tets of `volume_tag`.
#[derive(Clone, Debug)]
pub struct MaterialSpec {
    pub volume_tag: i32,
    pub er: f64,
    pub ur: f64,
    pub tand: f64,
    pub conductivity: f64,
    /// Diagonal conductivity `[σxx, σyy, σzz]` (S/m), overrides `conductivity`
    /// (homogenised via arrays).
    pub cond_diag: Option<[f64; 3]>,
    /// Diagonal permittivity, overrides `er`.
    pub er_diag: Option<[f64; 3]>,
    /// Diagonal permeability, overrides `ur`.
    pub ur_diag: Option<[f64; 3]>,
    pub debye: Option<Debye>,
    pub drude: Option<Drude>,
}

impl MaterialSpec {
    /// Vacuum on `volume_tag`, to be refined field by field.
    pub fn vacuum(volume_tag: i32) -> Self {
        MaterialSpec {
            volume_tag,
            er: 1.0,
            ur: 1.0,
            tand: 0.0,
            conductivity: 0.0,
            cond_diag: None,
            er_diag: None,
            ur_diag: None,
            debye: None,
            drude: None,
        }
    }
}

/// Debye relaxation: ε(ω) = ε∞ + (εs − ε∞)/(1 + jωτ).
#[derive(Clone, Copy, Debug)]
pub struct Debye {
    pub er_inf: f64,
    pub er_static: f64,
    /// Relaxation time τ in seconds.
    pub tau_s: f64,
}

/// Drude dispersion: ε(ω) = ε∞ − ωp²/(ω² + jγω).
#[derive(Clone, Copy, Debug)]
pub struct Drude {
    pub er_inf: f64,
    /// Plasma frequency in Hz (not angular).
    pub plasma_freq_hz: f64,
    /// Collision frequency in Hz (not angular).
    pub damping_freq_hz: f64,
}

/// Which cross-section solve a numerical wave port runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveKind {
    /// Full-vector hybrid solve (inhomogeneous cross-sections, quasi-TEM).
    Vector,
    /// Scalar Helmholtz, TE (`H_z`, Neumann walls).
    Te,
    /// Scalar Helmholtz, TM (`E_z`, Dirichlet walls).
    Tm,
}

impl WaveKind {
    /// The solve a wave port's `mode_kind` asks for: "te" and "tm" as named,
    /// "auto" the vector solve at `f0`, or without `f0` (the time domain,
    /// which has no operating frequency) the scalar TE solve.
    pub fn pick(mode_kind: &str, f0: Option<f64>) -> Result<WaveKind, String> {
        match mode_kind {
            "te" => Ok(WaveKind::Te),
            "tm" => Ok(WaveKind::Tm),
            "auto" if f0.is_none() => Ok(WaveKind::Te),
            "auto" => Ok(WaveKind::Vector),
            other => Err(format!("WavePort mode_kind must be one of ('auto', 'te', 'tm'), got '{other}'")),
        }
    }
}

/// A port or a boundary condition on the faces of `tag`.
#[derive(Clone, Debug)]
pub enum FaceSpec {
    /// Analytic `TE_mn` mode of a rectangular guide. `width`/`height` 0 are
    /// fitted to the face.
    Rectangular { tag: i32, width: f64, height: f64, mode: [usize; 2], er: f64, power: f64 },
    /// Plane-wave port of a unit cell. `mode_nr` 1 = TE (s), 2 = TM (p).
    /// Exact at normal incidence; at oblique scan the transverse phase factor
    /// is dropped.
    Floquet { tag: i32, scan_theta_deg: f64, scan_phi_deg: f64, mode_nr: u32, power: f64 },
    /// Uniform constant tangential field across the face.
    UserDefined { tag: i32, e_field: [f64; 3], power: f64 },
    /// Analytic coaxial TEM mode. `origin`/`z_axis` are fitted when `None`.
    Coax {
        tag: i32,
        ri: f64,
        ro: f64,
        origin: Option<[f64; 3]>,
        z_axis: Option<[f64; 3]>,
        er: f64,
        power: f64,
    },
    /// Lumped port: reference resistance `z0`, optional series L and C of
    /// the termination, voltage along `direction`.
    Lumped {
        tag: i32,
        z0: f64,
        l: f64,
        c: Option<f64>,
        direction: [f64; 3],
        width: f64,
        height: f64,
        power: f64,
    },
    /// First-order absorbing boundary.
    Abc { tag: i32 },
    /// Perfect magnetic conductor (the natural boundary condition).
    Pmc { tag: i32 },
    /// Series RLC element as a surface impedance
    /// `(R + jωL + 1/(jωC)) · width/height`.
    LumpedElement {
        tag: i32,
        r: f64,
        l: f64,
        c: Option<f64>,
        width: f64,
        height: f64,
        direction: [f64; 3],
    },
    /// Numerically solved cross-section mode. `f0` is the frequency of the
    /// vector solve (the frequency domain needs it); `pec_tags` mark internal
    /// conductors cutting the face.
    WaveNumerical {
        tag: i32,
        f0: Option<f64>,
        mode_index: usize,
        kind: WaveKind,
        pec_tags: Vec<i32>,
        power: f64,
    },
    /// Surface impedance of a lossy conductor, from `conductivity` or an
    /// explicit `zs = [re, im]` (Ω/sq). See
    /// `rapidfem_fd::waveguide::SurfaceImpedance` for `two_sided` and `sheet`.
    SurfaceImpedance {
        tag: i32,
        conductivity: f64,
        mur: f64,
        er: f64,
        thickness: Option<f64>,
        two_sided: bool,
        sheet: bool,
        zs: Option<[f64; 2]>,
    },
}

impl FaceSpec {
    /// Mesh tag of the faces this entry occupies.
    pub fn tag(&self) -> i32 {
        match self {
            FaceSpec::Rectangular { tag, .. }
            | FaceSpec::Floquet { tag, .. }
            | FaceSpec::UserDefined { tag, .. }
            | FaceSpec::Coax { tag, .. }
            | FaceSpec::Lumped { tag, .. }
            | FaceSpec::Abc { tag }
            | FaceSpec::Pmc { tag }
            | FaceSpec::LumpedElement { tag, .. }
            | FaceSpec::WaveNumerical { tag, .. }
            | FaceSpec::SurfaceImpedance { tag, .. } => *tag,
        }
    }

    /// Puts the entry on the faces of `new` (an entry is described before
    /// the mesh tags are handed out).
    pub fn set_tag(&mut self, new: i32) {
        match self {
            FaceSpec::Rectangular { tag, .. }
            | FaceSpec::Floquet { tag, .. }
            | FaceSpec::UserDefined { tag, .. }
            | FaceSpec::Coax { tag, .. }
            | FaceSpec::Lumped { tag, .. }
            | FaceSpec::Abc { tag }
            | FaceSpec::Pmc { tag }
            | FaceSpec::LumpedElement { tag, .. }
            | FaceSpec::WaveNumerical { tag, .. }
            | FaceSpec::SurfaceImpedance { tag, .. } => *tag = new,
        }
    }

    /// Whether the entry is a driven port.
    pub fn is_port(&self) -> bool {
        matches!(
            self,
            FaceSpec::Rectangular { .. }
                | FaceSpec::Floquet { .. }
                | FaceSpec::UserDefined { .. }
                | FaceSpec::Coax { .. }
                | FaceSpec::Lumped { .. }
                | FaceSpec::WaveNumerical { .. }
        )
    }
}

/// Perfectly matched layer on the tets of `volume_tag`, absorbing towards
/// `direction` from the coordinate `inner_face` over `thickness`.
#[derive(Clone, Debug)]
pub struct PmlSpec {
    pub volume_tag: i32,
    pub direction: [f64; 3],
    pub inner_face: f64,
    pub thickness: f64,
    pub er_base: f64,
    pub ur_base: f64,
    pub exponent: f64,
    pub delta_max: f64,
}
