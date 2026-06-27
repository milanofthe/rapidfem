// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Port boundary conditions and their analytic modal fields.
//!
//! Each port supplies a modal field profile, a propagation/mode impedance and
//! a Robin γ-coefficient for the boundary term. The closed forms are standard
//! microwave theory (Pozar, *Microwave Engineering*; Collin, *Field Theory of
//! Guided Waves*): rectangular-waveguide TE/TM modes, the coaxial and parallel
//! plate TEM modes, a Floquet plane-wave port, a Leontovich surface-impedance
//! BC, lumped ports/elements, and an absorbing boundary. `NumericalWavePort`
//! (general cross-sections) is solved numerically in `port_eigen`.

use num_complex::Complex64 as C64;
use rapidfem_core::port_eigen::NumericalMode;
use crate::constants::*;
use crate::excitation::Excitation;

/// Orthonormal port coordinate frame (origin + x̂,ŷ,ẑ rows).
///
/// Stores origin, xax (xhat), yax (yhat), zax (zhat), _basis, _basis_inv.
pub struct CoordinateSystem {
    pub origin: [f64; 3],
    pub xax: [f64; 3],
    pub yax: [f64; 3],
    pub zax: [f64; 3],
    /// _basis: rows are xax, yax, zax
    pub basis: [[f64; 3]; 3],
    /// _basis_inv: inverse of basis (for global→local transform)
    pub basis_inv: [[f64; 3]; 3],
}

impl CoordinateSystem {
    /// Build CS from origin and 3 orthonormal axes.
    pub fn new(origin: [f64; 3], xax: [f64; 3], yax: [f64; 3], zax: [f64; 3]) -> Self {
        let basis = [xax, yax, zax];
        // For orthonormal basis, inverse = transpose
        let basis_inv = [
            [xax[0], yax[0], zax[0]],
            [xax[1], yax[1], zax[1]],
            [xax[2], yax[2], zax[2]],
        ];
        CoordinateSystem { origin, xax, yax, zax, basis, basis_inv }
    }

    /// Map a global point into the local frame.
    pub fn in_local_cs(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let b = &self.basis_inv;
        let xg = x - self.origin[0];
        let yg = y - self.origin[1];
        let zg = z - self.origin[2];
        (
            b[0][0]*xg + b[0][1]*yg + b[0][2]*zg,
            b[1][0]*xg + b[1][1]*yg + b[1][2]*zg,
            b[2][0]*xg + b[2][1]*yg + b[2][2]*zg,
        )
    }

    /// Rotate local vector components back into the global frame.
    pub fn in_global_basis(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        (
            self.xax[0]*x + self.yax[0]*y + self.zax[0]*z,
            self.xax[1]*x + self.yax[1]*y + self.zax[1]*z,
            self.xax[2]*x + self.yax[2]*y + self.zax[2]*z,
        )
    }
}

/// Rectangular-waveguide port (TE/TM modes).
pub struct RectWaveguide {
    pub port_number: usize,
    pub power: f64,
    pub mode: (usize, usize),
    pub er: f64,
    pub polarization: f64,
    pub dims: (f64, f64),  // (width, height)
    pub cs: CoordinateSystem,
}

impl RectWaveguide {
    /// Modal field amplitude for the requested power.
    /// amplitude = sqrt(power * 4 * Z0 / (width * height))
    pub fn get_amplitude(&self, _exc: &Excitation) -> f64 {
        let zte = Z0;
        (self.power * 4.0 * zte / (self.dims.0 * self.dims.1)).sqrt()
    }

    /// Propagation constant β = √(k₀²εᵣ − k_c²).
    /// beta = sqrt(er*k0^2 - (pi*m/width)^2 - (pi*n/height)^2)
    pub fn get_beta(&self, exc: &Excitation) -> f64 {
        let (width, height) = self.dims;
        let (m, n) = self.mode;
        (self.er * exc.k0 * exc.k0
            - (PI * m as f64 / width).powi(2)
            - (PI * n as f64 / height).powi(2)).sqrt()
    }

    /// Robin γ-coefficient for the port boundary term.
    /// gamma = 1j * beta
    pub fn get_gamma(&self, exc: &Excitation) -> C64 {
        C64::new(0.0, self.get_beta(exc))
    }

    /// Mode wave impedance (TE: ωμ/β). Uses the length-coupled ω̃ = κ·c₀ so the
    /// impedance is invariant under the L₀ scaling (ω̃ and β both carry one L₀).
    /// Zmode = ω̃ * MU0 / beta
    pub fn z_mode(&self, exc: &Excitation) -> f64 {
        exc.omega_scaled() * MU0 / self.get_beta(exc)
    }

    /// Mode admittance-like weighting factor.
    /// qmode = sqrt(Zmode / Z0)
    pub fn qmode(&self, exc: &Excitation) -> f64 {
        (self.z_mode(exc) / Z0).sqrt()
    }

    /// Transverse modal E-field at a local cross-section point.
    /// Returns (Ex, Ey, Ez) in LOCAL coordinates.
    ///
    /// Ev = polarization * amplitude * cos(pi*m*x/width) * cos(pi*n*y/height)
    /// Eh = polarization * amplitude * sin(pi*m*x/width) * sin(pi*n*y/height)
    /// Ex = Eh, Ey = Ev, Ez = 0
    /// Result scaled by qmode.
    pub fn port_mode_3d(&self, x_local: f64, y_local: f64, exc: &Excitation) -> (f64, f64, f64) {
        let (width, height) = self.dims;
        let (m, n) = self.mode;
        let a = self.get_amplitude(exc);
        let ev = self.polarization * a
            * (PI * m as f64 * x_local / width).cos()
            * (PI * n as f64 * y_local / height).cos();
        let eh = self.polarization * a
            * (PI * m as f64 * x_local / width).sin()
            * (PI * n as f64 * y_local / height).sin();
        let ex = eh;
        let ey = ev;
        let ez = 0.0;
        let q = self.qmode(exc);
        (q * ex, q * ey, q * ez)
    }

    /// Modal E-field at a global point (rotated into global components).
    /// Returns (Ex, Ey, Ez) in GLOBAL coordinates.
    pub fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> (f64, f64, f64) {
        let (xl, yl, _zl) = self.cs.in_local_cs(x, y, z);
        let (ex, ey, ez) = self.port_mode_3d(xl, yl, exc);
        self.cs.in_global_basis(ex, ey, ez)
    }

    /// Incident-wave source term for the port excitation vector.
    /// Returns -2j * beta * port_mode_3d_global(...) as complex [3] vector.
    pub fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> [C64; 3] {
        let (ex, ey, ez) = self.port_mode_3d_global(x, y, z, exc);
        let factor = C64::new(0.0, -2.0 * self.get_beta(exc));
        [factor * C64::from(ex), factor * C64::from(ey), factor * C64::from(ez)]
    }
}

/// First-order Sommerfeld absorbing boundary: γ = j·k₀·neff (neff = 1 for air).
/// A plain matched-impedance sheet, dissipative by construction.
pub struct AbsorbingBoundary {
    pub neff: f64,
}

impl AbsorbingBoundary {
    pub fn new() -> Self {
        AbsorbingBoundary { neff: 1.0 }
    }

    /// γ(k0) = j·k₀·neff.
    pub fn get_gamma(&self, exc: &Excitation) -> C64 {
        C64::new(0.0, exc.k0 * self.neff)
    }
}

impl Default for AbsorbingBoundary {
    fn default() -> Self {
        Self::new()
    }
}

/// Floquet plane-wave port.
///
/// Models an incident plane wave at scan angles (θ, φ). Robin BC with γ = j·k₀·cos(θ).
/// Two polarization modes: TE (S-pol, mode_nr=1) and TM (P-pol, mode_nr=2).
///
/// LIMITATION: This implementation uses real-valued port_mode_3d_global, which is exact
/// at normal incidence (θ=0) where the mode field has no transverse phase variation.
/// For oblique incidence the field has a phase factor exp(-j(kx·x + ky·y)) which currently
/// does not flow through the S-parameter extraction trait. Full oblique support requires
/// extending the Port trait to a complex-valued mode field.
///
/// LIMITATION: A full Floquet simulation also requires periodic boundary conditions on the
/// side walls of the unit cell with phase factor exp(-jk₀·u·Δv). That's a separate feature
/// (master/slave DOF elimination in the assembly layer) and is not yet implemented (issue #14);
/// without it, FloquetPort is only useful for problems whose side walls are physically PEC/PMC,
/// and `build_ports` rejects oblique scan (θ≠0).
pub struct FloquetPort {
    pub port_number: usize,
    pub power: f64,
    pub er: f64,
    /// Port-face area (m²), used for amplitude normalization.
    pub area: f64,
    /// Scan angle θ measured from port normal (radians). 0 = normal incidence.
    pub scan_theta: f64,
    /// Scan angle φ in the port plane (radians).
    pub scan_phi: f64,
    /// 1 = TE (S-pol), 2 = TM (P-pol)
    pub mode_nr: u32,
    /// Local CS: origin = port-face anchor, z = outward normal.
    pub cs: CoordinateSystem,
}

impl FloquetPort {
    pub fn beta(&self, exc: &Excitation) -> f64 { exc.k0 * self.scan_theta.cos() }

    pub fn get_gamma(&self, exc: &Excitation) -> C64 {
        C64::new(0.0, self.beta(exc))
    }

    pub fn amplitude(&self, _exc: &Excitation) -> f64 {
        // E0 = sqrt(2·Z0·P / (A · cos θ)).
        (2.0 * Z0 * self.power / (self.area * self.scan_theta.cos())).sqrt()
    }

    pub fn port_mode_3d_global(&self, _x: f64, _y: f64, _z: f64, exc: &Excitation) -> (f64, f64, f64) {
        let (s, p): (f64, f64) = match self.mode_nr {
            1 => (1.0, 0.0),  // TE
            _ => (0.0, 1.0),  // TM
        };
        let cos_p = self.scan_phi.cos();
        let sin_p = self.scan_phi.sin();
        let cos_t = self.scan_theta.cos();
        let sin_t = self.scan_theta.sin();
        let e0 = self.amplitude(exc);

        // Uniform transverse mode field, valid only at normal incidence (θ=0),
        // which `build_ports` enforces. Oblique scan additionally needs the
        // transverse Bloch phase exp(-j·k_t·r) and periodic side-wall BCs — see
        // issue #14.
        let ex_l = e0 * (-s * sin_p - p * cos_t * cos_p);
        let ey_l = e0 * (s * cos_p - p * cos_t * sin_p);
        let ez_l = e0 * (-p * sin_t);
        self.cs.in_global_basis(ex_l, ey_l, ez_l)
    }

    pub fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> [C64; 3] {
        let (mx, my, mz) = self.port_mode_3d_global(x, y, z, exc);
        let factor = C64::new(0.0, -2.0 * self.beta(exc));
        [factor * C64::from(mx), factor * C64::from(my), factor * C64::from(mz)]
    }
}

/// User-defined port: a caller-supplied uniform transverse field.
///
/// The user supplies the port's E-field mode as a closure. A common case (constant E
/// uniform across the face, e.g. parallel-plate TEM) is exposed via `from_constant`.
/// gamma = j·β, get_uinc = -2j·β·mode_field. β defaults to k₀ but can be overridden.
pub struct UserDefinedPort {
    pub port_number: usize,
    pub power: f64,
    /// Mode field function: (k0, x, y, z) -> (Ex, Ey, Ez). Stored as a boxed closure.
    pub mode_fn: Box<dyn Fn(f64, f64, f64, f64) -> (f64, f64, f64) + Send + Sync>,
    /// Optional kz(k0) override; defaults to k0 (TEM-like)
    pub beta_fn: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
}

impl UserDefinedPort {
    /// Construct a port with a uniform constant E vector across the face.
    pub fn from_constant(port_number: usize, power: f64, e_vec: [f64; 3]) -> Self {
        let mode_fn: Box<dyn Fn(f64, f64, f64, f64) -> (f64, f64, f64) + Send + Sync> =
            Box::new(move |_k0, _x, _y, _z| (e_vec[0], e_vec[1], e_vec[2]));
        UserDefinedPort { port_number, power, mode_fn, beta_fn: None }
    }

    pub fn beta(&self, exc: &Excitation) -> f64 {
        match &self.beta_fn {
            Some(f) => f(exc.k0),
            None => exc.k0,
        }
    }

    pub fn get_gamma(&self, exc: &Excitation) -> C64 {
        C64::new(0.0, self.beta(exc))
    }

    pub fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> (f64, f64, f64) {
        (self.mode_fn)(exc.k0, x, y, z)
    }

    pub fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> [C64; 3] {
        let (mx, my, mz) = self.port_mode_3d_global(x, y, z, exc);
        let factor = C64::new(0.0, -2.0 * self.beta(exc));
        [factor * C64::from(mx), factor * C64::from(my), factor * C64::from(mz)]
    }
}

/// Coaxial port (TEM mode): radial E ∝ 1/ρ between inner and outer conductors.
///
/// Mode field is the analytic TEM coaxial wave: E_ρ = V₀ / (ρ · ln(Ro/Ri)).
/// V₀ = √(2·pZ₀·P), pZ₀ = (η/2π)·ln(Ro/Ri), η = Z₀/√εr, β = k₀√εr.
pub struct CoaxPort {
    pub port_number: usize,
    pub power: f64,
    pub er: f64,
    /// Inner conductor radius (m)
    pub ri: f64,
    /// Outer conductor radius (m)
    pub ro: f64,
    /// Local coordinate system. Origin = coax center on the port face,
    /// z-axis = propagation direction (out of plane).
    pub cs: CoordinateSystem,
}

impl CoaxPort {
    pub fn beta(&self, exc: &Excitation) -> f64 { exc.k0 * self.er.sqrt() }

    pub fn get_gamma(&self, exc: &Excitation) -> C64 {
        C64::new(0.0, self.beta(exc))
    }

    /// Characteristic impedance of the coaxial line (Ω).
    pub fn port_z(&self) -> f64 {
        let eta = Z0 / self.er.sqrt();
        (eta / (2.0 * std::f64::consts::PI)) * (self.ro / self.ri).ln()
    }

    /// V₀ for the requested power: P = |V₀|²/(2·Z₀_port).
    pub fn v0(&self) -> f64 {
        (2.0 * self.port_z() * self.power).sqrt()
    }

    pub fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, _exc: &Excitation) -> (f64, f64, f64) {
        // Local cylindrical: rho = sqrt(xl² + yl²), phi = atan2(yl, xl).
        // The mode formula is mathematically valid for any ρ>0; the gmsh-meshed annulus
        // confines us to ρ ∈ [Ri, Ro], so we don't gate on the radii (mesh-imperfect quadrature
        // points slightly outside the geometric bounds would otherwise be cut, costing power).
        let (xl, yl, _) = self.cs.in_local_cs(x, y, z);
        let rho = (xl * xl + yl * yl).sqrt();
        if rho < SINGULAR_EPS {
            return (0.0, 0.0, 0.0);
        }
        let phi = yl.atan2(xl);
        let e_rho = self.v0() / (rho * (self.ro / self.ri).ln());
        let ex_l = e_rho * phi.cos();
        let ey_l = e_rho * phi.sin();
        self.cs.in_global_basis(ex_l, ey_l, 0.0)
    }

    pub fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> [C64; 3] {
        let (mx, my, mz) = self.port_mode_3d_global(x, y, z, exc);
        let factor = C64::new(0.0, -2.0 * self.beta(exc));
        [factor * C64::from(mx), factor * C64::from(my), factor * C64::from(mz)]
    }
}

/// Construct an orthonormal coordinate system from origin + z-axis. The x and y axes
/// are chosen via Gram-Schmidt against an arbitrary reference (avoiding parallel cases).
pub fn cs_from_origin_zaxis(origin: [f64; 3], z_axis: [f64; 3]) -> CoordinateSystem {
    let zn = (z_axis[0].powi(2) + z_axis[1].powi(2) + z_axis[2].powi(2)).sqrt();
    let zhat = [z_axis[0] / zn, z_axis[1] / zn, z_axis[2] / zn];
    // Choose an x-axis reference that's not parallel to z
    let xref = if zhat[0].abs() < AXIS_REF_PARALLEL_MAX {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let dot = xref[0] * zhat[0] + xref[1] * zhat[1] + xref[2] * zhat[2];
    let xraw = [xref[0] - dot * zhat[0], xref[1] - dot * zhat[1], xref[2] - dot * zhat[2]];
    let xn = (xraw[0].powi(2) + xraw[1].powi(2) + xraw[2].powi(2)).sqrt();
    let xhat = [xraw[0] / xn, xraw[1] / xn, xraw[2] / xn];
    let yhat = [
        zhat[1] * xhat[2] - zhat[2] * xhat[1],
        zhat[2] * xhat[0] - zhat[0] * xhat[2],
        zhat[0] * xhat[1] - zhat[1] * xhat[0],
    ];
    CoordinateSystem::new(origin, xhat, yhat, zhat)
}

/// Surface impedance boundary condition (lossy conductor wall).
/// Leontovich surface-impedance boundary (lossy-metal sheet).
///
/// Robin BC with γ = j·k₀·Z₀/R, where R = surface resistivity.
/// Supports either user-supplied surface impedance or computation from σ via skin depth.
pub struct SurfaceImpedance {
    /// Surface conductivity in S/m (used to compute skin-depth R)
    pub sigma: f64,
    /// Relative permeability of the conductor
    pub mur: f64,
    /// Relative permittivity of the conductor (usually 1)
    pub er: f64,
    /// Optional finite layer thickness (m); if None, treated as semi-infinite
    pub thickness: Option<f64>,
    /// Optional explicit surface impedance Zs (Ω/sq); overrides σ-based calc when Some
    pub zs: Option<C64>,
}

impl SurfaceImpedance {
    pub fn from_conductivity(sigma: f64) -> Self {
        SurfaceImpedance { sigma, mur: 1.0, er: 1.0, thickness: None, zs: None }
    }

    pub fn from_zs(zs: C64) -> Self {
        SurfaceImpedance { sigma: 0.0, mur: 1.0, er: 1.0, thickness: None, zs: Some(zs) }
    }

    /// Robin γ-coefficient from the surface impedance Zs: γ = j·k₀·Z₀/Zs.
    pub fn get_gamma(&self, exc: &Excitation) -> C64 {
        let r = self.surface_impedance(exc);
        // γ = j*k0*Z0 / R
        C64::new(0.0, exc.k0 * Z0) / r
    }

    fn surface_impedance(&self, exc: &Excitation) -> C64 {
        if let Some(zs) = self.zs {
            return zs;
        }
        let w0 = exc.omega;
        let eps = crate::constants::EPS0 * self.er;
        let mu = crate::constants::MU0 * self.mur;
        let rho = 1.0 / self.sigma;
        // Skin depth: δ = sqrt( 2ρ/(ωμ) * (sqrt(1 + (ωερ)²) + ρωε) )
        let we = w0 * eps;
        let inner = (1.0 + (we * rho).powi(2)).sqrt() + rho * we;
        let d_skin = (2.0 * rho / (w0 * mu) * inner).sqrt();
        // R = (1 + j) ρ / δ
        let mut r = C64::new(1.0, 1.0) * C64::from(rho / d_skin);
        if let Some(t) = self.thickness {
            // Finite thickness scaler: R / tanh(γ_m * t), γ_m = j ω √(με_c)
            let eps_c = C64::new(eps, -self.sigma / w0);
            let mu_c = C64::new(mu, 0.0);
            let gamma_m = C64::new(0.0, w0) * (mu_c * eps_c).sqrt();
            r = r / (gamma_m * C64::from(t)).tanh();
        }
        r
    }
}

/// Lumped element (series R, L, C) applied as a surface impedance.
///
/// Robin BC with γ = j·k₀·Z₀/(Z(ω)·width/height) where Z(ω) = R + jωL + 1/(jωC).
/// Distinct from LumpedPort: there's no excitation, just a passive impedance load.
pub struct LumpedElement {
    /// Series resistance (Ω)
    pub r: f64,
    /// Series inductance (H)
    pub l: f64,
    /// Series capacitance (F); None means no capacitor (open replaced with infinite impedance)
    pub c: Option<f64>,
    /// Width (orthogonal to the field direction across the element gap)
    pub width: f64,
    /// Height (along the field direction across the element gap)
    pub height: f64,
}

impl LumpedElement {
    pub fn impedance(&self, exc: &Excitation) -> C64 {
        let omega = exc.omega;
        let mut z = C64::new(self.r, omega * self.l);
        if let Some(c) = self.c {
            if c > 0.0 {
                z += C64::new(0.0, -1.0 / (omega * c));
            }
        }
        z
    }

    pub fn surf_z(&self, exc: &Excitation) -> C64 {
        self.impedance(exc) * C64::from(self.width / self.height)
    }

    pub fn get_gamma(&self, exc: &Excitation) -> C64 {
        C64::new(0.0, exc.k0 * Z0) / self.surf_z(exc)
    }
}

/// Lumped port: a uniform-field gap source terminated in a series R-L-C.
///
/// Termination impedance Z(ω) = R + jωL + 1/(jωC) (R = `z0`); the Robin BC uses
/// the full RLC sheet impedance, while the incident-wave source and the
/// S-parameter reference use the resistive part R (the standard real reference
/// impedance). With L = 0 and no C this is the pure-R reference port.
/// Derivation: derivations/lumped_port/.
pub struct LumpedPort {
    pub port_number: usize,
    pub power: f64,
    /// Reference resistance R (Ω) — S-parameters and incident power normalise to this.
    pub z0: f64,
    /// Series inductance L (H); 0 ⇒ no inductor.
    pub l: f64,
    /// Series capacitance C (F); None ⇒ no capacitor.
    pub c: Option<f64>,
    pub width: f64,
    pub height: f64,
    /// E-field direction unit vector in global coordinates
    pub direction: [f64; 3],
}

impl LumpedPort {
    /// Series RLC termination impedance Z(ω) = R + jωL + 1/(jωC), R = z0.
    pub fn impedance(&self, exc: &Excitation) -> C64 {
        let omega = exc.omega;
        let mut z = C64::new(self.z0, omega * self.l);
        if let Some(c) = self.c {
            if c > 0.0 {
                z += C64::new(0.0, -1.0 / (omega * c));
            }
        }
        z
    }

    /// Sheet impedance for the Robin BC (full RLC): Z(ω) · width/height.
    pub fn surf_z(&self, exc: &Excitation) -> C64 {
        self.impedance(exc) * C64::from(self.width / self.height)
    }

    /// Real reference sheet impedance for the matched source: R · width/height.
    pub fn surf_z_ref(&self) -> f64 {
        self.z0 * self.width / self.height
    }

    /// Incident drive voltage referenced to R: sqrt(2 · power · R).
    pub fn voltage(&self) -> f64 {
        (2.0 * self.power * self.z0).sqrt()
    }

    /// Robin γ-coefficient for the port boundary term: j·k0·η0 / Zs(ω) (full RLC).
    pub fn get_gamma(&self, exc: &Excitation) -> C64 {
        C64::new(0.0, exc.k0 * Z0) / self.surf_z(exc)
    }

    /// Uniform modal field along the gap direction.
    pub fn port_mode_3d_global(&self, _x: f64, _y: f64, _z: f64, _exc: &Excitation) -> (f64, f64, f64) {
        (self.direction[0], self.direction[1], self.direction[2])
    }

    /// Incident source term −2·γ_R·E_inc (matched feed referenced to R):
    /// −2j·k0 · (voltage/height) · (η0 / Zs_ref) · mode_field, Zs_ref = R·w/h.
    pub fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> [C64; 3] {
        let emag = C64::new(0.0, -2.0 * exc.k0)
            * C64::from(self.voltage() / self.height * (Z0 / self.surf_z_ref()));
        let (ex, ey, ez) = self.port_mode_3d_global(x, y, z, exc);
        [emag * C64::from(ex), emag * C64::from(ey), emag * C64::from(ez)]
    }
}

/// Auto-detect port coordinate system and dimensions from mesh face triangles.
/// Detect an axis-aligned rectangular port and its local basis from its tris.
///
/// Returns (CoordinateSystem, width, height) where width ≥ height.
pub fn detect_rect_port(
    mesh: &crate::mesh::Mesh,
    tri_ids: &[usize],
) -> (CoordinateSystem, f64, f64) {
    let nodes = &mesh.nodes;
    let tris = &mesh.tris;

    // 1. Compute centroid (origin)
    let mut center = [0.0f64; 3];
    let mut count = 0.0;
    let mut all_verts = std::collections::HashSet::new();
    for &ti in tri_ids {
        for &vi in &tris[ti] {
            if all_verts.insert(vi) {
                for k in 0..3 { center[k] += nodes[vi][k]; }
                count += 1.0;
            }
        }
    }
    for k in 0..3 { center[k] /= count; }

    // 2. Compute face normal from first triangle
    let first_tri = tris[tri_ids[0]];
    let v0 = nodes[first_tri[0]];
    let v1 = nodes[first_tri[1]];
    let v2 = nodes[first_tri[2]];
    let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
    let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
    let cr = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    let nn = (cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]).sqrt();
    let mut normal = [cr[0]/nn, cr[1]/nn, cr[2]/nn];

    // 3. Orient outward using adjacent tet centroid
    let adj_tet = mesh.tri_to_tet[tri_ids[0]][0];
    let tet = &mesh.tets[adj_tet];
    let tc = [
        (nodes[tet[0]][0]+nodes[tet[1]][0]+nodes[tet[2]][0]+nodes[tet[3]][0])/4.0,
        (nodes[tet[0]][1]+nodes[tet[1]][1]+nodes[tet[2]][1]+nodes[tet[3]][1])/4.0,
        (nodes[tet[0]][2]+nodes[tet[1]][2]+nodes[tet[2]][2]+nodes[tet[3]][2])/4.0,
    ];
    let to_tet = [tc[0]-center[0], tc[1]-center[1], tc[2]-center[2]];
    if normal[0]*to_tet[0]+normal[1]*to_tet[1]+normal[2]*to_tet[2] > 0.0 {
        normal = [-normal[0], -normal[1], -normal[2]];
    }

    // 4. Find face extents along each axis
    let mut min_c = [f64::INFINITY; 3];
    let mut max_c = [f64::NEG_INFINITY; 3];
    for &vi in &all_verts {
        for k in 0..3 {
            min_c[k] = min_c[k].min(nodes[vi][k]);
            max_c[k] = max_c[k].max(nodes[vi][k]);
        }
    }
    let ext = [max_c[0]-min_c[0], max_c[1]-min_c[1], max_c[2]-min_c[2]];

    // 5. Determine axes: normal axis has smallest extent, xhat = largest extent, yhat = remaining
    let normal_axis = if normal[0].abs() > normal[1].abs() && normal[0].abs() > normal[2].abs() { 0 }
        else if normal[1].abs() > normal[2].abs() { 1 } else { 2 };

    let mut face_axes: Vec<(usize, f64)> = (0..3)
        .filter(|&k| k != normal_axis)
        .map(|k| (k, ext[k]))
        .collect();
    face_axes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let broad_axis = face_axes[0].0;
    let narrow_axis = face_axes[1].0;
    let width = face_axes[0].1;
    let height = face_axes[1].1;

    // 6. Build orthonormal CS
    let zhat = normal;
    let mut xhat = [0.0; 3];
    xhat[broad_axis] = 1.0;
    // Orthogonalize xhat against zhat (Gram-Schmidt)
    let dot_xz = xhat[0]*zhat[0]+xhat[1]*zhat[1]+xhat[2]*zhat[2];
    let xraw = [xhat[0]-dot_xz*zhat[0], xhat[1]-dot_xz*zhat[1], xhat[2]-dot_xz*zhat[2]];
    let xn = (xraw[0]*xraw[0]+xraw[1]*xraw[1]+xraw[2]*xraw[2]).sqrt();
    let xhat_f = [xraw[0]/xn, xraw[1]/xn, xraw[2]/xn];
    let yhat = [zhat[1]*xhat_f[2]-zhat[2]*xhat_f[1],
                zhat[2]*xhat_f[0]-zhat[0]*xhat_f[2],
                zhat[0]*xhat_f[1]-zhat[1]*xhat_f[0]];

    let _ = narrow_axis;
    (CoordinateSystem::new(center, xhat_f, yhat, zhat), width, height)
}

/// Return (width, height) for a lumped port:
/// height = extent along `direction` (the axis the potential is imposed along),
/// width = extent orthogonal to it in the port plane. surfZ = Z0 · width / height.
pub fn lumped_port_dims(
    mesh: &crate::mesh::Mesh,
    tri_ids: &[usize],
    direction: &[f64; 3],
) -> (f64, f64) {
    let mut verts = std::collections::HashSet::new();
    let mut min_c = [f64::INFINITY; 3];
    let mut max_c = [f64::NEG_INFINITY; 3];
    for &ti in tri_ids {
        for &vi in &mesh.tris[ti] {
            if verts.insert(vi) {
                for k in 0..3 {
                    min_c[k] = min_c[k].min(mesh.nodes[vi][k]);
                    max_c[k] = max_c[k].max(mesh.nodes[vi][k]);
                }
            }
        }
    }

    // Height: extent along direction
    let mut min_proj = f64::INFINITY;
    let mut max_proj = f64::NEG_INFINITY;
    for &vi in &verts {
        let p = mesh.nodes[vi];
        let proj = p[0]*direction[0] + p[1]*direction[1] + p[2]*direction[2];
        min_proj = min_proj.min(proj);
        max_proj = max_proj.max(proj);
    }
    let height = max_proj - min_proj;

    // Width: largest in-plane extent orthogonal to direction.
    // For axis-aligned ports we use AABB extents minus the height axis and the normal axis
    // (= zero-extent axis). The remaining axis with non-zero extent gives width.
    let ext = [max_c[0]-min_c[0], max_c[1]-min_c[1], max_c[2]-min_c[2]];
    let dir_axis = if direction[0].abs() > direction[1].abs() && direction[0].abs() > direction[2].abs() { 0 }
        else if direction[1].abs() > direction[2].abs() { 1 } else { 2 };
    let mut width = 0.0f64;
    for k in 0..3 {
        if k != dir_axis && ext[k] > width {
            width = ext[k];
        }
    }
    (width, height)
}


// =============================================================================
// NumericalWavePort: 2D mode eigensolve on the port-face triangulation.
//
// Wraps a `NumericalMode` (scalar TE/TM Helmholtz OR full-vector hybrid for
// inhomogeneous cross-sections) and exposes it via the `Port` trait. The
// mode profile is unit-peak-normalised by `NumericalMode`; we rescale to the
// power-normalised amplitude via the precomputed integral
//   norm = sqrt( ∫ |e_t^unit|^2 dA )
// at construction time. Then
//   port_mode_3d_global(x, k0) = e_profile(x) * amplitude(k0) / norm
// with amplitude(k0) chosen so that 0.5/Z_mode * A^2 * norm^2 = power.
// =============================================================================

/// Single quadrature point on a triangle: (weight, L1, L2, L3) in barycentric.
type Tri4Tuple = (f64, f64, f64, f64);

/// Sum a function over the triangle surface mesh using gmsh-style P1 quadrature.
/// Inline replica of `sparam::surface_integral` for real-valued scalars, we
/// only need this at port construction to integrate |e_t^unit|^2, which is
/// real, so importing the complex version would be churn.
fn integrate_scalar_over_tris(
    nodes: &[[f64; 3]],
    triangles: &[[usize; 3]],
    dpts: &[Tri4Tuple],
    f: &dyn Fn(f64, f64, f64) -> f64,
) -> f64 {
    let mut total = 0.0;
    for tri in triangles {
        let v1 = nodes[tri[0]];
        let v2 = nodes[tri[1]];
        let v3 = nodes[tri[2]];
        let e1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
        let e2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]];
        let cr = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let area = 0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
        let mut tri_sum = 0.0;
        for &(w, l1, l2, l3) in dpts {
            let x = v1[0] * l1 + v2[0] * l2 + v3[0] * l3;
            let y = v1[1] * l1 + v2[1] * l2 + v3[1] * l3;
            let z = v1[2] * l1 + v2[2] * l2 + v3[2] * l3;
            tri_sum += w * f(x, y, z);
        }
        total += tri_sum * area;
    }
    total
}

/// FD-side wrapper of `NumericalMode` that satisfies the `Port` trait.
///
/// One construction per port: the eigensolve runs at `f0` (the operating
/// frequency the user passed), and the resulting `(e_t, h_t)` mode profile is
/// cached. During the sweep, `get_uinc` / `port_mode_3d_global` re-scale the
/// (real) profile by an amplitude that depends on the live `k0`:
///   amplitude(k0) = sqrt(2 · z_mode(k0) · power) / mode_l2_norm
/// where `mode_l2_norm = sqrt(∫|e_t^unit|^2 dA)` is precomputed.
pub struct NumericalWavePort {
    pub port_number: usize,
    pub power: f64,
    pub mode: NumericalMode,
    /// √(∫ |e_t^unit|^2 dA) over the port face, precomputed at construction.
    pub mode_l2_norm: f64,
    /// Effective index frozen at the eigensolve frequency f0. Hybrid quasi-
    /// TEM only, scalar modes ignore this and derive β from the cutoff.
    pub n_eff: f64,
    /// `true` if `mode` came from `solve_vector_modes` (β = n_eff·k0); `false`
    /// if from `solve_modes` (β = sqrt(k0² − k_c²)).
    pub is_vector: bool,
}

impl NumericalWavePort {
    /// Modal propagation constant β at the live operating wavenumber k0.
    /// Vector path: linear dispersion frozen at f0. Scalar path: closed-form
    /// from the cutoff.
    pub fn get_beta(&self, exc: &Excitation) -> f64 {
        if self.is_vector {
            self.n_eff * exc.k0
        } else {
            let kc = self.mode.cutoff();
            (exc.k0 * exc.k0 - kc * kc).max(0.0).sqrt()
        }
    }

    pub fn get_gamma(&self, exc: &Excitation) -> C64 {
        C64::new(0.0, self.get_beta(exc))
    }

    /// Modal wave impedance in SI ohms, the field-ratio `|E_t|/|H_t|` of
    /// the propagating mode. Scalar TE: `Z_0/√(1−(k_c/ω)²)`; scalar TM:
    /// `Z_0·√(1−(k_c/ω)²)`; vector hybrid (quasi-TEM): `Z_0/n_eff`. Used
    /// only by power-flux-based S-param paths (`sparam_field_power`,
    /// `sparam_mode_power`); the mode-projection path (`sparam_waveport`)
    /// is amplitude-invariant.
    pub fn z_mode(&self, exc: &Excitation) -> f64 {
        // Length-coupled ω̃ = κ·c₀ keeps the cutoff ratio ωc/ω̃ invariant under L₀.
        self.mode.te_impedance(exc.omega_scaled()) * Z0
    }

    /// Power-normalised amplitude scaling for the unit-peak mode profile.
    ///
    /// The eigenmode's transverse power flux (Poynting integral) is, for
    /// any propagating mode written as `exp(-jβz)`:
    ///
    /// ``S_z = (β / 2ωμ_0) · |E_t|²  =  (n_eff / 2 Z_0) · |E_t|²``
    ///
    /// independent of TE / TM / hybrid character, homogeneous or
    /// inhomogeneous fill. Setting the unit-peak-normalised mode to
    /// carry the user's `power` watts gives
    ///
    /// ``amp = √(2 Z_0 · power / n_eff(k0)) / mode_l2_norm``
    ///
    /// where `mode_l2_norm² = ∫ |E_t^unit|² dA`. Frequency dependence
    /// enters only via `n_eff(k0)` (scalar TE/TM dispersion); the vector
    /// path freezes `n_eff` at the eigensolve frequency `f_0`.
    fn amplitude(&self, exc: &Excitation) -> f64 {
        if self.mode_l2_norm <= 0.0 {
            return 0.0;
        }
        let beta = self.get_beta(exc);
        if beta <= 0.0 {
            return 0.0;
        }
        let n_eff_now = beta / exc.k0;
        (2.0 * Z0 * self.power / n_eff_now).sqrt() / self.mode_l2_norm
    }

    /// Mode profile at a global point, scaled so the incident wave carries
    /// `self.power` watts through the port face.
    pub fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> (f64, f64, f64) {
        let e = self.mode.e_profile([x, y, z]);
        let a = self.amplitude(exc);
        (a * e[0], a * e[1], a * e[2])
    }

    /// Incident-wave RHS contribution, same convention as RectWaveguide:
    /// uinc = −2j·β · port_mode_3d_global.
    pub fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> [C64; 3] {
        let (ex, ey, ez) = self.port_mode_3d_global(x, y, z, exc);
        let factor = C64::new(0.0, -2.0 * self.get_beta(exc));
        [factor * C64::from(ex), factor * C64::from(ey), factor * C64::from(ez)]
    }

    /// Construct from an already-solved `NumericalMode`.
    ///
    /// Pre-computes `mode_l2_norm = √(∫ |E_t^unit|² dA)` over the port
    /// face. Combined with the eigenmode's `n_eff` it gives the
    /// Poynting-flux-per-unit-amplitude `S_z = (n_eff/2 Z_0) · |E_t|²`
    /// that `amplitude(k0)` inverts to land the unit-peak mode at the
    /// user-requested `power`.
    pub fn new(
        port_number: usize,
        power: f64,
        mode: NumericalMode,
        n_eff: f64,
        is_vector: bool,
        face_nodes: &[[f64; 3]],
        face_tris: &[[usize; 3]],
    ) -> Self {
        let dpts: Vec<Tri4Tuple> = crate::quadrature::gaus_quad_tri(4)
            .into_iter()
            .map(|q| (q[0], q[1], q[2], q[3]))
            .collect();
        let l2_sq = integrate_scalar_over_tris(face_nodes, face_tris, &dpts, &|x, y, z| {
            let e = mode.e_profile([x, y, z]);
            e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
        });
        let mode_l2_norm = l2_sq.max(0.0).sqrt();
        NumericalWavePort {
            port_number,
            power,
            mode,
            mode_l2_norm,
            n_eff,
            is_vector,
        }
    }
}
