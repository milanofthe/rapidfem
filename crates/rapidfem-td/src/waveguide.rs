// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Analytic waveguide port modes.
//!
//! A rectangular waveguide of width `a` and height `b` supports `TE_mn`
//! modes with closed-form transverse field profiles. For a time-domain
//! port these profiles shape the incident excitation and the modal
//! extraction; the guide dispersion itself emerges from the PEC walls
//! during propagation, so only the transverse profile and the cutoff are
//! needed here. A coaxial line carries the dispersionless TEM mode, a
//! radial `E_ρ ∝ 1/ρ` field between the two conductors. Everything is in
//! the solver's normalised units (`c = ε₀ = μ₀ = 1`).

use crate::constants::{COAX_RADIUS_FLOOR, Field};
/// Pi in the operator's working precision (`Field`).
const PI: Field = std::f64::consts::PI as Field;

#[inline]
fn dot(a: [Field; 3], b: [Field; 3]) -> Field {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [Field; 3], b: [Field; 3]) -> [Field; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A rectangular-waveguide port: its in-plane coordinate frame, cross
/// section, and `TE_mn` mode.
///
/// `u_hat`, `v_hat`, `w_hat` form a right-handed frame (`û × v̂ = ŵ`) with
/// `w_hat` the **inward** normal, pointing into the simulation domain.
#[derive(Clone, Debug)]
pub struct RectPort {
    /// A corner of the port rectangle (global coords), the `(u,v)=(0,0)` point.
    pub origin: [Field; 3],
    /// Unit vector along the width `a` (global).
    pub u_hat: [Field; 3],
    /// Unit vector along the height `b` (global).
    pub v_hat: [Field; 3],
    /// Inward unit normal, points into the domain (global).
    pub w_hat: [Field; 3],
    /// Cross-section width.
    pub a: Field,
    /// Cross-section height.
    pub b: Field,
    /// `TE` mode indices `(m, n)`.
    ///
    /// The `(0, 0)` sentinel is **internal-only**: it carries a uniform
    /// transverse field and was the old lumped-port mode, which has been
    /// removed from the public time-domain API (a uniform delta-gap
    /// profile cannot represent a concentrated quasi-TEM line). It now
    /// survives solely so [`FloquetPort::from_face`] can borrow
    /// [`RectPort::from_face`]'s frame-fitting; no user-facing port
    /// resolves to `(0, 0)`.
    pub mode: (usize, usize),
    /// Reference impedance for the `(0, 0)` sentinel mode, in the
    /// operator's normalised units (`Z = 1` is free-space 377 ohm).
    /// Vestigial, only the internal `(0, 0)` frame-fit path touches it;
    /// `TE_mn` modes ignore it (their impedance is dispersive, set by
    /// the cutoff). Kept to avoid churn until the wave-port work lands.
    pub z0: Field,
}

impl RectPort {
    /// Local `(u, v)` coordinates of a global point on the port plane.
    fn local(&self, x: [Field; 3]) -> (Field, Field) {
        let d = [
            x[0] - self.origin[0],
            x[1] - self.origin[1],
            x[2] - self.origin[2],
        ];
        (dot(d, self.u_hat), dot(d, self.v_hat))
    }

    /// Transverse electric-field profile of the mode at a global point on
    /// the port face, in global coordinates and normalised so the dominant
    /// component peaks at unit amplitude.
    ///
    /// `TE_mn`: `E_u ∝ (n/b)·cos(mπu/a)·sin(nπv/b)`,
    /// `E_v ∝ −(m/a)·sin(mπu/a)·cos(nπv/b)`. The sentinel mode `(0, 0)` is
    /// a **lumped / TEM port**, a uniform transverse field along `v_hat`,
    /// with zero cutoff and a flat (non-dispersive) `Z = 1` impedance.
    pub fn e_profile(&self, x: [Field; 3]) -> [Field; 3] {
        if self.mode == (0, 0) {
            return self.v_hat;
        }
        let (u, v) = self.local(x);
        let (m, n) = (self.mode.0 as Field, self.mode.1 as Field);
        let mu = m * PI / self.a;
        let nv = n * PI / self.b;
        let eu = (n / self.b) * (mu * u).cos() * (nv * v).sin();
        let ev = -(m / self.a) * (mu * u).sin() * (nv * v).cos();
        let scale = (m / self.a).max(n / self.b).max(Field::MIN_POSITIVE);
        let (eu, ev) = (eu / scale, ev / scale);
        [
            eu * self.u_hat[0] + ev * self.v_hat[0],
            eu * self.u_hat[1] + ev * self.v_hat[1],
            eu * self.u_hat[2] + ev * self.v_hat[2],
        ]
    }

    /// Transverse magnetic-field profile for a mode propagating along the
    /// inward normal. For `TE_mn` modes (m, n != 0, 0) this is just
    /// `h_t = ŵ × e_t` at the free-space impedance the operator's
    /// normalisation uses. For the lumped `(0, 0)` mode the result is
    /// scaled by `1 / z0` so the port carries a wave at the user's
    /// reference impedance: `|E| / |H| = z0`. Global coordinates.
    pub fn h_profile(&self, x: [Field; 3]) -> [Field; 3] {
        let h = cross(self.w_hat, self.e_profile(x));
        if self.mode == (0, 0) && self.z0 > 0.0 {
            [h[0] / self.z0, h[1] / self.z0, h[2] / self.z0]
        } else {
            h
        }
    }

    /// Cutoff angular frequency `ω_c = π·√((m/a)² + (n/b)²)` (`c = 1`).
    /// Content below `ω_c` is evanescent and does not propagate.
    pub fn cutoff(&self) -> Field {
        let (m, n) = (self.mode.0 as Field, self.mode.1 as Field);
        PI * ((m / self.a).powi(2) + (n / self.b).powi(2)).sqrt()
    }

    /// Fit a `RectPort` to an axis-aligned port face from its mesh node
    /// coordinates and the inward normal (pointing into the domain).
    ///
    /// The wider transverse dimension becomes the width `a` (`u_hat`), the
    /// narrower the height `b` (`v_hat`); the frame is made right-handed
    /// (`û × v̂ = ŵ`). The `TE_mn` mode then has `m` indexing the wide
    /// dimension, so `TE₁₀` is the dominant mode regardless of orientation.
    ///
    /// `field_axis` overrides the auto-fit transverse axis `v̂`: a lumped
    /// port's voltage-integration direction is projected into the port
    /// plane and used as `v̂` (with `û` rebuilt to keep the frame
    /// right-handed). `None` keeps the auto-fit. A direction parallel to
    /// the normal has no in-plane part and is ignored.
    pub fn from_face(
        nodes: &[[Field; 3]],
        inward_normal: [Field; 3],
        mode: (usize, usize),
        field_axis: Option<[Field; 3]>,
    ) -> RectPort {
        // The inward normal is ±eₖ, the constant (out-of-plane) axis.
        let k = (0..3)
            .max_by(|&i, &j| {
                inward_normal[i]
                    .abs()
                    .partial_cmp(&inward_normal[j].abs())
                    .unwrap()
            })
            .unwrap();
        let s = if inward_normal[k] >= 0.0 { 1.0 } else { -1.0 };
        let others: Vec<usize> = (0..3).filter(|&x| x != k).collect();
        let extent = |ax: usize| -> (Field, Field) {
            let lo = nodes.iter().map(|p| p[ax]).fold(Field::MAX, Field::min);
            let hi = nodes.iter().map(|p| p[ax]).fold(Field::MIN, Field::max);
            (lo, hi - lo)
        };
        let (lo0, ext0) = extent(others[0]);
        let (lo1, ext1) = extent(others[1]);
        // Wide axis → width a, narrow → height b.
        let (wide, narrow, a, b, lo_w, lo_n) = if ext0 >= ext1 {
            (others[0], others[1], ext0, ext1, lo0, lo1)
        } else {
            (others[1], others[0], ext1, ext0, lo1, lo0)
        };
        let mut u_hat = [0.0; 3];
        u_hat[wide] = 1.0;
        let mut v_hat = [0.0; 3];
        v_hat[narrow] = 1.0;
        let mut w_hat = [0.0; 3];
        w_hat[k] = s;
        let mut origin = [0.0; 3];
        origin[wide] = lo_w;
        origin[k] = nodes[0][k];
        // Make (u, v, w) right-handed; flipping v̂ moves its origin corner.
        if dot(cross(u_hat, v_hat), w_hat) >= 0.0 {
            origin[narrow] = lo_n;
        } else {
            v_hat[narrow] = -1.0;
            origin[narrow] = lo_n + b;
        }
        // An explicit field axis (a lumped port's voltage-integration
        // direction) overrides the auto-fit transverse axis: project it
        // into the port plane and rebuild a right-handed (û, v̂, ŵ) frame.
        if let Some(d) = field_axis {
            let dn = dot(d, w_hat);
            let proj =
                [d[0] - dn * w_hat[0], d[1] - dn * w_hat[1], d[2] - dn * w_hat[2]];
            let len = dot(proj, proj).sqrt();
            if len > 1e-9 {
                v_hat = [proj[0] / len, proj[1] / len, proj[2] / len];
                u_hat = cross(v_hat, w_hat); // û = v̂×ŵ ⇒ û×v̂ = ŵ
            }
        }
        RectPort {
            origin,
            u_hat,
            v_hat,
            w_hat,
            a,
            b,
            mode,
            z0: 1.0,
        }
    }

    /// `TE`-mode wave impedance at angular frequency `omega`, in the
    /// solver's normalised units (`Z₀ = 1`): `Z_TE = 1/√(1 − (ω_c/ω)²)`.
    ///
    /// This is the ratio `|E_t|/|H_t|` of the propagating mode. The
    /// forward/backward modal split uses it, and because it is
    /// frequency-dependent the split must be done per frequency. Valid
    /// for `omega > cutoff`.
    pub fn te_impedance(&self, omega: Field) -> Field {
        // Lumped (0,0) is dispersionless and carries the user-set Z0;
        // a true TE_mn mode follows the standard dispersive formula
        // with free-space impedance Z = 1 in operator units.
        if self.mode == (0, 0) {
            return self.z0;
        }
        let r = self.cutoff() / omega;
        1.0 / (1.0 - r * r).sqrt()
    }
}

/// A coaxial-line port carrying the transverse-electromagnetic (TEM) mode.
///
/// Between an inner conductor of radius `r_i` and an outer conductor of
/// radius `r_o` a coaxial line supports a dispersionless TEM mode: a purely
/// radial transverse electric field `E ∝ ρ̂/ρ` and an azimuthal magnetic
/// field `H = ŵ × E`. The mode has no cutoff and travels at exactly `c`,
/// so its modal impedance is flat, exactly the role the `(0, 0)` sentinel
/// of [`RectPort`] plays for a lumped port.
///
/// The port plane is described by its `w_hat` inward normal (pointing into
/// the simulation domain) and the coax `center` lying on it; `ρ` is the
/// in-plane distance from that center.
#[derive(Clone, Debug)]
pub struct CoaxPort {
    /// Coax-axis center on the port plane (global coords).
    pub center: [Field; 3],
    /// Inward unit normal, points into the domain (global). Also the
    /// coax propagation axis.
    pub w_hat: [Field; 3],
    /// Inner-conductor radius.
    pub r_inner: Field,
    /// Outer-conductor radius.
    pub r_outer: Field,
}

impl CoaxPort {
    /// In-plane radial vector `x − center` with its `w_hat` component
    /// removed, the part of the displacement that lies in the port plane.
    fn in_plane(&self, x: [Field; 3]) -> [Field; 3] {
        let d = [
            x[0] - self.center[0],
            x[1] - self.center[1],
            x[2] - self.center[2],
        ];
        let dn = dot(d, self.w_hat);
        [
            d[0] - dn * self.w_hat[0],
            d[1] - dn * self.w_hat[1],
            d[2] - dn * self.w_hat[2],
        ]
    }

    /// Transverse electric-field profile of the TEM mode at a global point
    /// on the port face, in global coordinates.
    ///
    /// The coax TEM field is `E_ρ ∝ 1/ρ`; this returns `ρ̂·(r_i/ρ)`, so the
    /// dominant value is unit magnitude at the inner radius, the same
    /// order-unity normalisation [`RectPort::e_profile`] uses. On the
    /// degenerate axis (`ρ → 0`, off the meshed annulus) the field is zero.
    pub fn e_profile(&self, x: [Field; 3]) -> [Field; 3] {
        let rho_vec = self.in_plane(x);
        let rho = dot(rho_vec, rho_vec).sqrt();
        if rho < COAX_RADIUS_FLOOR {
            return [0.0; 3];
        }
        // ρ̂·(r_i/ρ): unit magnitude at ρ = r_inner, the 1/ρ TEM decay
        // outward, expressed as the in-plane vector scaled by r_i/ρ².
        let scale = self.r_inner / (rho * rho);
        [rho_vec[0] * scale, rho_vec[1] * scale, rho_vec[2] * scale]
    }

    /// Transverse magnetic-field profile for the TEM mode propagating along
    /// the inward normal, `h_t = ŵ × e_t`. Global coordinates.
    pub fn h_profile(&self, x: [Field; 3]) -> [Field; 3] {
        cross(self.w_hat, self.e_profile(x))
    }

    /// Cutoff angular frequency, always `0` for a TEM mode, which
    /// propagates at every frequency down to DC.
    pub fn cutoff(&self) -> Field {
        0.0
    }

    /// Modal wave impedance, flat `Z = 1` in the solver's normalised
    /// units. The TEM mode is non-dispersive, so the forward/backward modal
    /// split needs no per-frequency rescaling, exactly like the `(0, 0)`
    /// lumped sentinel of [`RectPort`].
    pub fn te_impedance(&self, _omega: Field) -> Field {
        1.0
    }

    /// Fit a `CoaxPort` to a coaxial annular port face from its mesh node
    /// coordinates and the inward normal (pointing into the domain).
    ///
    /// The coax center is the centroid of the face nodes projected onto the
    /// port plane, unless `center_override` supplies an explicit axis point.
    /// The inner radius is the smallest in-plane node distance to that
    /// center, the outer radius the largest, the annulus the mesh spans.
    pub fn from_face(
        nodes: &[[Field; 3]],
        inward_normal: [Field; 3],
        center_override: Option<[Field; 3]>,
    ) -> CoaxPort {
        let nl = dot(inward_normal, inward_normal).sqrt();
        let w_hat = [
            inward_normal[0] / nl,
            inward_normal[1] / nl,
            inward_normal[2] / nl,
        ];
        // Center: an explicit axis point, or the centroid of the face nodes.
        let raw_center = center_override.unwrap_or_else(|| {
            let mut c = [0.0; 3];
            for p in nodes {
                for k in 0..3 {
                    c[k] += p[k];
                }
            }
            let inv = 1.0 / nodes.len() as Field;
            [c[0] * inv, c[1] * inv, c[2] * inv]
        });
        let mut port = CoaxPort {
            center: raw_center,
            w_hat,
            r_inner: 0.0,
            r_outer: 0.0,
        };
        // Inner / outer radii, the extreme in-plane node distances.
        let (mut lo, mut hi) = (Field::MAX, 0.0_f64);
        for &p in nodes {
            let rv = port.in_plane(p);
            let r = dot(rv, rv).sqrt();
            lo = lo.min(r);
            hi = hi.max(r);
        }
        port.r_inner = lo;
        port.r_outer = hi;
        port
    }
}

/// Polarisation mode of a [`FloquetPort`].
///
/// TE (`s`-polarised): the electric field is perpendicular to the plane of
/// incidence. TM (`p`-polarised): the electric field lies in the plane of
/// incidence. The plane of incidence is spanned by the port inward normal
/// `ŵ` and the in-plane scan direction
/// `(cosφ·û + sinφ·v̂)`; at normal incidence (`scan_theta = 0`) the plane
/// of incidence collapses to a line and the TE polarisation falls along
/// the in-plane perpendicular `(−sinφ·û + cosφ·v̂)`, the TM along
/// `(cosφ·û + sinφ·v̂)` (both purely transverse).
///
/// The same convention as the FD `FloquetPort` (`mode_nr = 1` → TE,
/// `mode_nr = 2` → TM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloquetPolarisation {
    /// `s`-polarised, E perpendicular to the plane of incidence.
    Te,
    /// `p`-polarised, E in the plane of incidence.
    Tm,
}

/// A Floquet plane-wave port: a uniform transverse plane wave on the port
/// face of a periodic unit cell.
///
/// `u_hat`, `v_hat`, `w_hat` form a right-handed frame (`û × v̂ = ŵ`) with
/// `w_hat` the **inward** normal, pointing into the simulation domain. At
/// normal incidence the polarisation lies entirely in the port plane:
/// `Te` picks the in-plane azimuth perpendicular,
/// `Tm` picks the in-plane azimuth along the scan direction.
///
/// **Oblique-scan approximation.** A real Floquet mode carries a transverse
/// phase factor `e^{-j·k_t·r_t}` across the port face, which makes the
/// field complex-valued. The time-domain backend's port machinery is
/// real-valued (the [`PortMode`] returns real `e_profile` / `h_profile`),
/// so at oblique incidence (`scan_theta ≠ 0`) the transverse phase is
/// **dropped**, the polarisation vector still rotates with `scan_theta`
/// (the TM out-of-plane lift, the TE in-plane orientation) but the field
/// stays uniform across the face. This matches the FD `FloquetPort`'s
/// documented approach. Normal incidence (`scan_theta = 0`) is the
/// validated, exact case; oblique angles are a documented approximation.
#[derive(Clone, Debug)]
pub struct FloquetPort {
    /// A corner of the port rectangle (global coords), the `(u,v)=(0,0)` point.
    pub origin: [Field; 3],
    /// Unit vector along the width `a` (global).
    pub u_hat: [Field; 3],
    /// Unit vector along the height `b` (global).
    pub v_hat: [Field; 3],
    /// Inward unit normal, points into the domain (global).
    pub w_hat: [Field; 3],
    /// Cross-section width.
    pub a: Field,
    /// Cross-section height.
    pub b: Field,
    /// TE / TM polarisation choice.
    pub polarisation: FloquetPolarisation,
    /// Elevation scan angle `θ` from the port normal, in radians.
    pub scan_theta: Field,
    /// Azimuth scan angle `φ` in the port plane, in radians.
    pub scan_phi: Field,
    /// Optional explicit in-plane polarisation override. If `Some`, this
    /// (already-projected and normalised) in-plane unit vector is the
    /// transverse electric-field direction, the `polarisation`, `scan_*`
    /// derivation is bypassed entirely. The dominant use is hooking up an
    /// experiment that pins a specific linear polarisation independent of
    /// the φ-azimuth convention.
    pub e_override: Option<[Field; 3]>,
}

impl FloquetPort {
    /// Resolved unit polarisation vector in global coordinates.
    ///
    /// `e_override` short-circuits the derivation when present. Otherwise
    /// the TE vector is the in-plane perpendicular to the scan azimuth
    /// `(−sinφ·û + cosφ·v̂)`; the TM vector at normal incidence is the
    /// in-plane scan direction `(cosφ·û + sinφ·v̂)`, tilted by `scan_theta`
    /// out of the port plane along `−ŵ` for `θ > 0` (so its E·E stays unit
    /// length). The transverse phase factor `e^{-j·k_t·r_t}` is **dropped**
    ///, see the struct doc.
    fn polarisation_vec(&self) -> [Field; 3] {
        if let Some(p) = self.e_override {
            return p;
        }
        let (cos_p, sin_p) = (self.scan_phi.cos(), self.scan_phi.sin());
        let (cos_t, sin_t) = (self.scan_theta.cos(), self.scan_theta.sin());
        match self.polarisation {
            FloquetPolarisation::Te => {
                // E ⟂ plane of incidence, in-plane perpendicular to (cosφ û + sinφ v̂).
                let c = [-sin_p, cos_p];
                [
                    c[0] * self.u_hat[0] + c[1] * self.v_hat[0],
                    c[0] * self.u_hat[1] + c[1] * self.v_hat[1],
                    c[0] * self.u_hat[2] + c[1] * self.v_hat[2],
                ]
            }
            FloquetPolarisation::Tm => {
                // E in plane of incidence, at θ = 0 along (cosφ û + sinφ v̂);
                // for θ > 0 it tilts out of the port plane along −ŵ so that
                // E·E = 1 (cos²θ + sin²θ).
                let inplane = (cos_t * cos_p, cos_t * sin_p);
                [
                    inplane.0 * self.u_hat[0]
                        + inplane.1 * self.v_hat[0]
                        - sin_t * self.w_hat[0],
                    inplane.0 * self.u_hat[1]
                        + inplane.1 * self.v_hat[1]
                        - sin_t * self.w_hat[1],
                    inplane.0 * self.u_hat[2]
                        + inplane.1 * self.v_hat[2]
                        - sin_t * self.w_hat[2],
                ]
            }
        }
    }

    /// Transverse electric-field profile of the Floquet mode at a global
    /// point on the port face, uniform across the face at the polarisation
    /// vector derived from `polarisation`, `scan_theta`, `scan_phi`
    /// (or `e_override` when supplied). The transverse phase factor
    /// `e^{-j·k_t·r_t}` is dropped at oblique scan; see the struct doc.
    pub fn e_profile(&self, _x: [Field; 3]) -> [Field; 3] {
        self.polarisation_vec()
    }

    /// Transverse magnetic-field profile, `h_t = ŵ × e_t`, exactly as for
    /// the rectangular and coax modes. Free-space impedance is `1` in the
    /// solver's normalised units, so `|H| = |E|` for the propagating wave.
    pub fn h_profile(&self, x: [Field; 3]) -> [Field; 3] {
        cross(self.w_hat, self.e_profile(x))
    }

    /// Cutoff angular frequency, always `0`. A plane wave propagates at
    /// every frequency down to DC; the unit-cell periodicity sets the
    /// transverse momentum, the longitudinal `k_z` carries the rest.
    pub fn cutoff(&self) -> Field {
        0.0
    }

    /// Modal wave impedance, flat `Z = 1` in the solver's normalised
    /// units (free space, non-dispersive). The forward / backward modal
    /// split therefore needs no per-frequency rescaling.
    pub fn te_impedance(&self, _omega: Field) -> Field {
        1.0
    }

    /// Fit a `FloquetPort` to an axis-aligned rectangular unit-cell face
    /// from its mesh node coordinates and the inward normal.
    ///
    /// The frame fitting reuses [`RectPort::from_face`]'s axis-aligned
    /// logic: the wider transverse dimension becomes `a` (`u_hat`), the
    /// narrower `b` (`v_hat`); the frame is made right-handed. The
    /// polarisation is then built from the TE / TM choice and the scan
    /// angles. `polarisation_override`, if `Some`, supplies an explicit
    /// in-plane polarisation direction (projected back into the port
    /// plane and normalised). A direction parallel to the normal has no
    /// in-plane part and is rejected by returning the auto-derived
    /// polarisation (with `e_override = None`).
    pub fn from_face(
        nodes: &[[Field; 3]],
        inward_normal: [Field; 3],
        polarisation: FloquetPolarisation,
        scan_theta: Field,
        scan_phi: Field,
        polarisation_override: Option<[Field; 3]>,
    ) -> FloquetPort {
        // Borrow the rectangular port's frame fitter, same axis-aligned
        // (u, v, w) layout and right-handed convention; mode is irrelevant
        // here so use the lumped sentinel.
        let rect = RectPort::from_face(nodes, inward_normal, (0, 0), None);
        let mut port = FloquetPort {
            origin: rect.origin,
            u_hat: rect.u_hat,
            v_hat: rect.v_hat,
            w_hat: rect.w_hat,
            a: rect.a,
            b: rect.b,
            polarisation,
            scan_theta,
            scan_phi,
            e_override: None,
        };
        if let Some(d) = polarisation_override {
            // Project into the port plane and normalise.
            let dn = dot(d, port.w_hat);
            let proj = [
                d[0] - dn * port.w_hat[0],
                d[1] - dn * port.w_hat[1],
                d[2] - dn * port.w_hat[2],
            ];
            let len = dot(proj, proj).sqrt();
            if len > 1e-9 {
                port.e_override =
                    Some([proj[0] / len, proj[1] / len, proj[2] / len]);
            }
        }
        port
    }
}

/// A port's waveguide mode, the mode-specific data the port machinery
/// consumes, dispatched by variant.
///
/// The port flux, the injection source and the modal extraction are all
/// mode-agnostic: they touch a mode only through `e_profile`, `h_profile`,
/// `cutoff` and `te_impedance`. This enum is the single point where a
/// rectangular `TE_mn` mode, a coaxial TEM mode and a Floquet plane-wave
/// mode differ.
#[derive(Clone, Debug)]
pub enum PortMode {
    /// A rectangular-waveguide `TE_mn` mode (or the `(0,0)` lumped sentinel).
    Rect(RectPort),
    /// A coaxial-line TEM mode.
    Coax(CoaxPort),
    /// A Floquet plane-wave mode on a periodic unit-cell face.
    Floquet(FloquetPort),
    /// A numerically-solved mode from a 2D cross-section eigensolve, the
    /// wave port for an arbitrary (ridged, circular, ...) cross-section
    /// whose profile has no closed form. See
    /// [`rapidfem_core::port_eigen`].
    Numerical(rapidfem_core::port_eigen::NumericalMode),
}

impl PortMode {
    /// Transverse electric-field profile at a global point on the port face.
    pub fn e_profile(&self, x: [Field; 3]) -> [Field; 3] {
        match self {
            PortMode::Rect(p) => p.e_profile(x),
            PortMode::Coax(p) => p.e_profile(x),
            PortMode::Floquet(p) => p.e_profile(x),
            PortMode::Numerical(p) => p.e_profile(x),
        }
    }

    /// Transverse magnetic-field profile at a global point on the port face.
    pub fn h_profile(&self, x: [Field; 3]) -> [Field; 3] {
        match self {
            PortMode::Rect(p) => p.h_profile(x),
            PortMode::Coax(p) => p.h_profile(x),
            PortMode::Floquet(p) => p.h_profile(x),
            PortMode::Numerical(p) => p.h_profile(x),
        }
    }

    /// Cutoff angular frequency of the mode (`0` for a TEM / plane-wave mode).
    pub fn cutoff(&self) -> Field {
        match self {
            PortMode::Rect(p) => p.cutoff(),
            PortMode::Coax(p) => p.cutoff(),
            PortMode::Floquet(p) => p.cutoff(),
            PortMode::Numerical(p) => p.cutoff(),
        }
    }

    /// Modal wave impedance at angular frequency `omega`.
    pub fn te_impedance(&self, omega: Field) -> Field {
        match self {
            PortMode::Rect(p) => p.te_impedance(omega),
            PortMode::Coax(p) => p.te_impedance(omega),
            PortMode::Floquet(p) => p.te_impedance(omega),
            PortMode::Numerical(p) => p.te_impedance(omega),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned port on the z-plane: width along x, height along y,
    /// inward normal +z.
    fn z_port(a: f64, b: f64, mode: (usize, usize)) -> RectPort {
        RectPort {
            origin: [0.0, 0.0, 0.0],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, 1.0],
            a,
            b,
            mode,
            z0: 1.0,
        }
    }

    #[test]
    fn te10_profile_matches_the_analytic_field() {
        // TE₁₀: E = v̂·sin(πx/a), peaks mid-width, vanishes at the side
        // walls, is purely transverse-vertical.
        let p = z_port(0.5, 0.25, (1, 0));
        // Side walls u = 0, a → zero.
        for &u in &[0.0, 0.5] {
            let e = p.e_profile([u, 0.1, 0.0]);
            assert!(e.iter().all(|c| c.abs() < 1e-12), "wall E = {e:?}");
        }
        // Mid-width u = a/2 → unit peak, along v̂ (y).
        let e = p.e_profile([0.25, 0.1, 0.0]);
        assert!((e[1].abs() - 1.0).abs() < 1e-12, "peak |E| = {}", e[1].abs());
        assert!(e[0].abs() < 1e-12 && e[2].abs() < 1e-12, "E not along v̂");
        // Sine shape across the width.
        let e_q = p.e_profile([0.125, 0.1, 0.0]); // u = a/4
        assert!(
            (e_q[1].abs() - (PI * 0.25).sin()).abs() < 1e-12,
            "E(a/4) = {}",
            e_q[1].abs(),
        );
    }

    #[test]
    fn h_profile_is_an_inward_propagating_partner() {
        // h_t = ŵ × e_t ⇒ E, H, and the propagation direction are mutually
        // orthogonal and E × H points inward (ŵ).
        let p = z_port(0.5, 0.25, (1, 0));
        let x = [0.25, 0.1, 0.0];
        let e = p.e_profile(x);
        let h = p.h_profile(x);
        assert!(dot(e, h).abs() < 1e-12, "E·H = {}", dot(e, h));
        let poynting = cross(e, h);
        assert!(
            dot(poynting, p.w_hat) > 0.0,
            "E×H must point inward, got {poynting:?}",
        );
        // |h| = |e| for a transverse profile crossed with the unit normal.
        let (ne, nh) = (dot(e, e).sqrt(), dot(h, h).sqrt());
        assert!((ne - nh).abs() < 1e-12, "|E| {ne} vs |H| {nh}");
    }

    #[test]
    fn from_face_fits_an_axis_aligned_port() {
        // A z = 3 face spanning [0,2]×[0,1], inward normal −ẑ.
        let nodes = [
            [0.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],
            [2.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
            [1.0, 0.5, 3.0],
        ];
        let p = RectPort::from_face(&nodes, [0.0, 0.0, -1.0], (1, 0), None);
        // Width = the larger span (x), height = the smaller (y).
        assert!((p.a - 2.0).abs() < 1e-12, "a = {}", p.a);
        assert!((p.b - 1.0).abs() < 1e-12, "b = {}", p.b);
        assert!((p.w_hat[2] + 1.0).abs() < 1e-12, "inward normal");
        // Right-handed frame.
        let uxv = cross(p.u_hat, p.v_hat);
        assert!(dot(uxv, p.w_hat) > 0.999, "frame not right-handed");
        // The TE₁₀ profile peaks mid-width and vanishes at the side walls.
        let mid = p.e_profile([1.0, 0.5, 3.0]);
        assert!(mid.iter().map(|c| c * c).sum::<f64>().sqrt() > 0.99);
        let wall = p.e_profile([0.0, 0.5, 3.0]);
        assert!(wall.iter().all(|c| c.abs() < 1e-9), "side-wall E ≠ 0");
    }

    #[test]
    fn from_face_honours_an_explicit_field_axis() {
        // A z = 0 face spanning [0,2]×[0,1], auto-fit makes v̂ the narrow
        // (y) axis. An explicit field axis must override that.
        let nodes = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.5, 0.0],
        ];
        let n = [0.0, 0.0, 1.0]; // inward +z
        // Auto-fit picks the narrow (y) axis for v̂.
        let auto = RectPort::from_face(&nodes, n, (0, 0), None);
        assert!((auto.v_hat[1].abs() - 1.0).abs() < 1e-12, "auto v̂ ≠ ±ŷ");
        // An explicit x axis (the wide axis) overrides the auto-fit.
        let p =
            RectPort::from_face(&nodes, n, (0, 0), Some([1.0, 0.0, 0.0]));
        assert!(
            (p.v_hat[0] - 1.0).abs() < 1e-12
                && p.v_hat[1].abs() < 1e-12
                && p.v_hat[2].abs() < 1e-12,
            "v̂ not set to x̂: {:?}",
            p.v_hat,
        );
        // The (0,0) field profile follows the explicit axis.
        let e = p.e_profile([1.0, 0.5, 0.0]);
        assert!((e[0] - 1.0).abs() < 1e-12 && e[1].abs() < 1e-12);
        // The frame stays right-handed.
        let uxv = cross(p.u_hat, p.v_hat);
        assert!(dot(uxv, p.w_hat) > 0.999, "frame not right-handed");
        // A direction with an out-of-plane component is projected back
        // into the port plane.
        let q =
            RectPort::from_face(&nodes, n, (0, 0), Some([0.0, 3.0, 9.0]));
        assert!(
            (q.v_hat[1] - 1.0).abs() < 1e-12 && q.v_hat[2].abs() < 1e-12,
            "out-of-plane direction not projected: {:?}",
            q.v_hat,
        );
    }

    #[test]
    fn lumped_port_is_a_uniform_zero_cutoff_mode() {
        // The (0,0) sentinel mode, a lumped / TEM port: uniform transverse
        // field, no cutoff, flat Z = 1 impedance.
        let p = z_port(0.5, 0.25, (0, 0));
        // Uniform field along v̂ everywhere on the face.
        for &(u, v) in &[(0.1, 0.05), (0.25, 0.1), (0.49, 0.2)] {
            let e = p.e_profile([u, v, 0.0]);
            assert!((e[1] - 1.0).abs() < 1e-12, "not uniform: {e:?}");
            assert!(e[0].abs() < 1e-12 && e[2].abs() < 1e-12);
        }
        // No cutoff, and the impedance is flat (non-dispersive).
        assert!(p.cutoff().abs() < 1e-12, "lumped port has a cutoff");
        for &omega in &[0.3, 1.0, 5.0] {
            assert!((p.te_impedance(omega) - 1.0).abs() < 1e-12);
        }
        // E × H still points inward (a forward-propagating partner).
        let x = [0.25, 0.1, 0.0];
        let poynting = cross(p.e_profile(x), p.h_profile(x));
        assert!(dot(poynting, p.w_hat) > 0.99);
    }

    #[test]
    fn cutoff_matches_the_analytic_frequency() {
        // TE₁₀ of an a = 0.5 guide: ω_c = π/a = 2π.
        let p = z_port(0.5, 0.25, (1, 0));
        assert!((p.cutoff() - 2.0 * PI).abs() < 1e-12);
        // TE₁₁: ω_c = π·√((1/a)² + (1/b)²).
        let p11 = z_port(0.5, 0.25, (1, 1));
        let want = PI * ((1.0 / 0.5_f64).powi(2) + (1.0 / 0.25_f64).powi(2)).sqrt();
        assert!((p11.cutoff() - want).abs() < 1e-12);
    }

    /// A coax port on the z = 0 plane, centred on the origin, inward +z.
    fn z_coax(r_inner: f64, r_outer: f64) -> CoaxPort {
        CoaxPort {
            center: [0.0, 0.0, 0.0],
            w_hat: [0.0, 0.0, 1.0],
            r_inner,
            r_outer,
        }
    }

    #[test]
    fn coax_profile_is_radial_and_decays_as_one_over_rho() {
        // The TEM coax field is purely radial and falls off as 1/ρ, with
        // unit magnitude at the inner radius.
        let p = z_coax(0.2, 0.6);
        // At the inner radius, along +x: |E| = 1, purely radial (along x̂).
        let e = p.e_profile([0.2, 0.0, 0.0]);
        assert!((e[0] - 1.0).abs() < 1e-12, "inner |E| = {}", e[0]);
        assert!(e[1].abs() < 1e-12 && e[2].abs() < 1e-12, "E not radial");
        // At the outer radius: |E| = r_inner/r_outer.
        let eo = p.e_profile([0.0, 0.6, 0.0]);
        let mag = (eo[0] * eo[0] + eo[1] * eo[1] + eo[2] * eo[2]).sqrt();
        assert!((mag - 0.2 / 0.6).abs() < 1e-12, "outer |E| = {mag}");
        assert!(eo[2].abs() < 1e-12, "E has an out-of-plane part");
        // Radial direction everywhere: E ∥ (x − center) in the plane.
        let ed = p.e_profile([0.3, 0.4, 0.0]); // ρ = 0.5
        let rho = 0.5_f64;
        let mag_d = (ed[0] * ed[0] + ed[1] * ed[1]).sqrt();
        assert!((mag_d - 0.2 / rho).abs() < 1e-12, "1/ρ decay broken");
        // ρ̂ = (0.6, 0.8); E must be parallel to it.
        let cross_z = ed[0] * 0.8 - ed[1] * 0.6;
        assert!(cross_z.abs() < 1e-12, "E not along ρ̂");
        // On the (un-meshed) axis the profile is zero, not divergent.
        let e_axis = p.e_profile([0.0, 0.0, 0.0]);
        assert!(e_axis.iter().all(|c| c.abs() < 1e-12), "axis E ≠ 0");
    }

    #[test]
    fn coax_h_profile_is_an_inward_propagating_partner() {
        // h_t = ŵ × e_t, E, H and the propagation direction are mutually
        // orthogonal and E × H points inward (ŵ), like a forward wave.
        let p = z_coax(0.2, 0.6);
        let x = [0.3, 0.4, 0.0];
        let e = p.e_profile(x);
        let h = p.h_profile(x);
        assert!(dot(e, h).abs() < 1e-12, "E·H = {}", dot(e, h));
        let poynting = cross(e, h);
        assert!(
            dot(poynting, p.w_hat) > 0.0,
            "E×H must point inward, got {poynting:?}",
        );
        // |h| = |e| for a transverse profile crossed with the unit normal.
        let (ne, nh) = (dot(e, e).sqrt(), dot(h, h).sqrt());
        assert!((ne - nh).abs() < 1e-12, "|E| {ne} vs |H| {nh}");
        // The TEM mode is azimuthal in H: H ⟂ the radial direction.
        let rho_hat = [0.6, 0.8, 0.0];
        assert!(dot(h, rho_hat).abs() < 1e-12, "H not azimuthal");
    }

    #[test]
    fn coax_port_has_zero_cutoff_and_flat_impedance() {
        // The TEM mode propagates at every frequency: no cutoff, and a flat
        // (non-dispersive) Z = 1 in the solver's normalised units.
        let p = z_coax(0.2, 0.6);
        assert!(p.cutoff().abs() < 1e-12, "TEM mode has a cutoff");
        for &omega in &[0.1, 1.0, 7.0] {
            assert!((p.te_impedance(omega) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn coax_from_face_fits_the_annulus() {
        // Nodes sampled on two concentric rings around (0,0) on the z = 3
        // plane: from_face must recover the center, radii and inward normal.
        let (ri, ro) = (0.25, 0.75);
        let mut nodes = Vec::new();
        for k in 0..8 {
            let ang = k as f64 * std::f64::consts::FRAC_PI_4;
            nodes.push([ri * ang.cos(), ri * ang.sin(), 3.0]);
            nodes.push([ro * ang.cos(), ro * ang.sin(), 3.0]);
        }
        let p = CoaxPort::from_face(&nodes, [0.0, 0.0, -1.0], None);
        assert!((p.r_inner - ri).abs() < 1e-12, "r_inner = {}", p.r_inner);
        assert!((p.r_outer - ro).abs() < 1e-12, "r_outer = {}", p.r_outer);
        assert!((p.w_hat[2] + 1.0).abs() < 1e-12, "inward normal");
        // The centroid of a symmetric ring set is the axis origin (the
        // z = 3 ring sits at z = 3, the in-plane centroid is the origin).
        assert!(p.center[0].abs() < 1e-12, "center off-axis: {:?}", p.center);
        assert!(p.center[1].abs() < 1e-12, "center off-axis: {:?}", p.center);
        assert!((p.center[2] - 3.0).abs() < 1e-12, "center off-plane");
        // An explicit center override is honoured.
        let q = CoaxPort::from_face(
            &nodes,
            [0.0, 0.0, -1.0],
            Some([0.0, 0.0, 3.0]),
        );
        assert!((q.center[2] - 3.0).abs() < 1e-12, "override ignored");
    }

    #[test]
    fn port_mode_dispatches_to_the_variant() {
        // PortMode forwards every query to the wrapped mode unchanged.
        let rect = z_port(0.5, 0.25, (1, 0));
        let mr = PortMode::Rect(rect.clone());
        let x = [0.25, 0.1, 0.0];
        assert_eq!(mr.e_profile(x), rect.e_profile(x));
        assert_eq!(mr.h_profile(x), rect.h_profile(x));
        assert_eq!(mr.cutoff(), rect.cutoff());
        assert_eq!(mr.te_impedance(3.0 * PI), rect.te_impedance(3.0 * PI));

        let coax = z_coax(0.2, 0.6);
        let mc = PortMode::Coax(coax.clone());
        let xc = [0.3, 0.4, 0.0];
        assert_eq!(mc.e_profile(xc), coax.e_profile(xc));
        assert_eq!(mc.h_profile(xc), coax.h_profile(xc));
        assert_eq!(mc.cutoff(), coax.cutoff());
        assert_eq!(mc.te_impedance(1.0), 1.0);

        // A normal-incidence Floquet port also dispatches through.
        let fp = FloquetPort {
            origin: [0.0; 3],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, 1.0],
            a: 1.0,
            b: 1.0,
            polarisation: FloquetPolarisation::Te,
            scan_theta: 0.0,
            scan_phi: 0.0,
            e_override: None,
        };
        let mf = PortMode::Floquet(fp.clone());
        let xf = [0.5, 0.5, 0.0];
        assert_eq!(mf.e_profile(xf), fp.e_profile(xf));
        assert_eq!(mf.h_profile(xf), fp.h_profile(xf));
        assert_eq!(mf.cutoff(), 0.0);
        assert_eq!(mf.te_impedance(2.0 * PI), 1.0);
    }

    #[test]
    fn floquet_port_normal_incidence_te_polarisation() {
        // TE at φ = 0: E is the in-plane perpendicular to (cosφ û + sinφ v̂),
        // so along +v̂. Uniform across the face. |H| = |E|, E·H = 0,
        // E×H = ŵ (inward Poynting).
        let p = FloquetPort {
            origin: [0.0; 3],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, 1.0],
            a: 1.0,
            b: 1.0,
            polarisation: FloquetPolarisation::Te,
            scan_theta: 0.0,
            scan_phi: 0.0,
            e_override: None,
        };
        let e0 = p.e_profile([0.1, 0.1, 0.0]);
        let e1 = p.e_profile([0.7, 0.4, 0.0]);
        assert!((e0[0]).abs() < 1e-12 && (e0[1] - 1.0).abs() < 1e-12);
        for k in 0..3 {
            assert!((e0[k] - e1[k]).abs() < 1e-12, "TE field is not uniform");
        }
        let h = p.h_profile([0.5, 0.5, 0.0]);
        // For E = (0, 1, 0) and ŵ = +ẑ: h = ŵ × E = (-1, 0, 0).
        assert!((h[0] + 1.0).abs() < 1e-12 && h[1].abs() < 1e-12 && h[2].abs() < 1e-12);
        let s = cross(e0, h);
        assert!(dot(s, p.w_hat) > 0.999, "Poynting not inward, got {s:?}");
        // No cutoff, flat impedance.
        assert!(p.cutoff().abs() < 1e-12);
        assert!((p.te_impedance(3.0).abs() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn floquet_port_normal_incidence_tm_polarisation() {
        // TM at φ = 0 and θ = 0: E along +û. Cross-orthogonal to TE.
        let p = FloquetPort {
            origin: [0.0; 3],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, 1.0],
            a: 1.0,
            b: 1.0,
            polarisation: FloquetPolarisation::Tm,
            scan_theta: 0.0,
            scan_phi: 0.0,
            e_override: None,
        };
        let e = p.e_profile([0.5, 0.5, 0.0]);
        assert!((e[0] - 1.0).abs() < 1e-12 && e[1].abs() < 1e-12 && e[2].abs() < 1e-12);
        // For φ = π/2 the TM polarisation rotates to +v̂.
        let p_phi = FloquetPort {
            scan_phi: PI / 2.0,
            ..p.clone()
        };
        let e2 = p_phi.e_profile([0.5, 0.5, 0.0]);
        assert!(e2[0].abs() < 1e-12 && (e2[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn floquet_port_oblique_tm_carries_out_of_plane_component() {
        // At θ > 0 the TM polarisation tilts out of the port plane along
        // −ŵ (so |E| stays 1). The transverse phase factor is dropped.
        let p = FloquetPort {
            origin: [0.0; 3],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, 1.0],
            a: 1.0,
            b: 1.0,
            polarisation: FloquetPolarisation::Tm,
            scan_theta: PI / 6.0, // 30 degrees
            scan_phi: 0.0,
            e_override: None,
        };
        let e = p.e_profile([0.5, 0.5, 0.0]);
        let cos_t = (PI / 6.0).cos();
        let sin_t = (PI / 6.0).sin();
        assert!((e[0] - cos_t).abs() < 1e-12, "E_u != cosθ: {}", e[0]);
        assert!(e[1].abs() < 1e-12);
        assert!((e[2] + sin_t).abs() < 1e-12, "E_w != -sinθ: {}", e[2]);
        // Unit norm.
        let nrm = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        assert!((nrm - 1.0).abs() < 1e-12);
        // And the field is still uniform across the face, the transverse
        // phase is intentionally dropped (real-valued port API).
        let e2 = p.e_profile([0.9, 0.1, 0.0]);
        for k in 0..3 {
            assert!((e[k] - e2[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn floquet_port_from_face_fits_an_axis_aligned_unit_cell() {
        // A z = 0 face spanning [0, 2] x [0, 1], inward normal +z. Auto-fit
        // makes (u, v) = (x, y) with u the wide axis. TE at φ = 0 gives a
        // uniform field along v̂ = +ŷ.
        let nodes = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.5, 0.0],
        ];
        let p = FloquetPort::from_face(
            &nodes,
            [0.0, 0.0, 1.0],
            FloquetPolarisation::Te,
            0.0,
            0.0,
            None,
        );
        assert!((p.a - 2.0).abs() < 1e-12 && (p.b - 1.0).abs() < 1e-12);
        assert!((p.w_hat[2] - 1.0).abs() < 1e-12);
        // Right-handed frame.
        let uxv = cross(p.u_hat, p.v_hat);
        assert!(dot(uxv, p.w_hat) > 0.999);
        let e = p.e_profile([1.0, 0.5, 0.0]);
        // TE φ=0 gives E along +v̂; for this face v̂ = ŷ.
        assert!((e[1].abs() - 1.0).abs() < 1e-12);
        // Explicit polarisation override.
        let q = FloquetPort::from_face(
            &nodes,
            [0.0, 0.0, 1.0],
            FloquetPolarisation::Te,
            0.0,
            0.0,
            Some([1.0, 0.0, 0.0]),
        );
        let eq = q.e_profile([1.0, 0.5, 0.0]);
        assert!((eq[0] - 1.0).abs() < 1e-12, "override ignored: {eq:?}");
    }
}
