// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! DG Maxwell RHS operator.
//!
//! The semi-discrete DG form of the vacuum Maxwell curl equations
//! (`∂E/∂t = ∇×H`, `∂H/∂t = -∇×E`) splits per element into a volume term
//! (the physical curl) and a surface term (the numerical flux). This module
//! builds those operators; the volume curl is assembled and validated first.
//!
//! Per-element fields are stored node-major: `field[node*3 + component]`,
//! with components ordered `x, y, z`.

use crate::constants::{
    APPLY_TASKS_PER_THREAD, Field, PERIODIC_MATCH_ABS_FLOOR,
    PERIODIC_MATCH_REL_TOL,
};
use crate::dg_basis::ReferenceElement;
use crate::dispersive::DebyeMaterial;
use crate::geom_factors::{GeometricFactors, all_geometric_factors};
use crate::waveguide::{
    CoaxPort, FloquetPolarisation, FloquetPort, PortMode, RectPort,
};
use rapidfem_core::port_eigen::{solve_port_face, ModeKind, PortSolve};
use rapidfem_core::mesh::Mesh;
use rapidfem_core::topology::FaceTopology;
use rayon::prelude::*;
use std::sync::Mutex;
use rapidfem_core::geom::{cross, dot};

/// Physical curl of a vector field on a single element.
///
/// `field` holds `3·Np` values (`field[node*3 + comp]`); the result has the
/// same layout and contains `∇×field` sampled at the element nodes. This is
/// the allocating wrapper around `element_curl_into`, the hot path uses
/// the scratch-buffer form.
pub fn element_curl(
    re: &ReferenceElement,
    gf: &GeometricFactors,
    field: &[Field],
) -> Vec<Field> {
    let n = re.n_nodes;
    let mut out = vec![0.0; 3 * n];
    let mut rd = vec![0.0; 3 * n];
    let mut pd = vec![0.0; 9 * n];
    element_curl_into(re, gf, field, &mut out, &mut rd, &mut pd);
    out
}

/// Physical curl, writing into caller-provided buffers, no allocation.
///
/// `out` (`3·Np`) receives `∇×field`; `rd` (`3·Np`) and `pd` (`9·Np`) are
/// scratch. `rd[k·n+i]` holds the `ξ_k` reference derivative; `pd[(p·3+c)·n+i]`
/// holds `∂(field_c)/∂x_p`.
fn element_curl_into(
    re: &ReferenceElement,
    gf: &GeometricFactors,
    field: &[Field],
    out: &mut [Field],
    rd: &mut [Field],
    pd: &mut [Field],
) {
    let n = re.n_nodes;
    debug_assert_eq!(field.len(), 3 * n);
    let dref = [&re.diff_r, &re.diff_s, &re.diff_t];

    for comp in 0..3 {
        // Reference derivatives of this component: rd[k·n + i].
        for (k, d) in dref.iter().enumerate() {
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += d[i * n + j] * field[j * 3 + comp];
                }
                rd[k * n + i] = acc;
            }
        }
        // Combine via the metric into physical derivatives pd[(p·3+comp)·n+i].
        for phys in 0..3 {
            let jinv = [
                gf.jacobian_inv[0][phys],
                gf.jacobian_inv[1][phys],
                gf.jacobian_inv[2][phys],
            ];
            let base = (phys * 3 + comp) * n;
            for i in 0..n {
                pd[base + i] = jinv[0] * rd[i]
                    + jinv[1] * rd[n + i]
                    + jinv[2] * rd[2 * n + i];
            }
        }
    }

    // curl_x = ∂Fz/∂y - ∂Fy/∂z, and cyclic; pd index = (phys·3 + comp)·n + i.
    let idx = |phys: usize, comp: usize, i: usize| (phys * 3 + comp) * n + i;
    for i in 0..n {
        out[i * 3] = pd[idx(1, 2, i)] - pd[idx(2, 1, i)];
        out[i * 3 + 1] = pd[idx(2, 0, i)] - pd[idx(0, 2, i)];
        out[i * 3 + 2] = pd[idx(0, 1, i)] - pd[idx(1, 0, i)];
    }
}

/// Per (element, local face) flux data.
pub(crate) struct FaceInfo {
    /// Outward unit normal.
    pub(crate) normal: [Field; 3],
    /// Surface scaling `2·area_phys / |det J|` (the lift assumes a 1/2 reference face).
    pub(crate) fscale: Field,
    /// Neighbour element, or `usize::MAX` on a domain-boundary face.
    pub(crate) neighbor: usize,
    /// Neighbour local face.
    pub(crate) neighbor_local_face: usize,
    /// `perm[m]` = neighbour face-node local index matching this face's node `m`.
    pub(crate) perm: Vec<usize>,
    /// Port index if this boundary face belongs to a port, else `usize::MAX`.
    /// A non-port boundary face (`neighbor == usize::MAX`) is a PEC wall.
    pub(crate) port: usize,
}

/// A port, a set of mesh boundary faces carrying a waveguide mode.
///
/// Identified by the mesh triangle indices on the port plane (a face group
/// tag resolves to exactly such a set via `Mesh::ftag_to_tri`). With
/// `mode = None` the port is a pure characteristic absorbing boundary;
/// `Some` attaches a waveguide mode (rectangular `TE_mn` or coaxial TEM)
/// for injection / extraction.
/// The triangles of face group `face_tag`, their distinct node coordinates
/// and the inward unit normal of the first one; `None` for an empty tag.
fn port_face(mesh: &Mesh, face_tag: i32) -> Option<(Vec<usize>, Vec<[Field; 3]>, [Field; 3])> {
    let tris = mesh.tris_for_tag(face_tag).to_vec();
    let nrm = mesh.tri_inward_normal(*tris.first()?)?;
    let coords = mesh.tri_nodes(&tris).into_iter().map(|n| mesh.nodes[n]).collect();
    Some((tris, coords, nrm))
}

#[derive(Clone, Debug)]
pub struct PortSpec {
    /// Mesh triangle indices forming this port's boundary faces.
    pub tris: Vec<usize>,
    /// Waveguide mode of this port, or `None` for an absorbing-only port.
    pub mode: Option<PortMode>,
}

impl PortSpec {
    /// Build a waveguide port from a face group tag, collecting the port
    /// triangles and fitting the rectangular-waveguide `TE_mn` mode to the
    /// face.
    ///
    /// `direction`, if given, is a lumped port's voltage-integration axis;
    /// it overrides the auto-fit transverse field axis. Returns `None` if
    /// the tag carries no triangles, or if `direction` is zero or parallel
    /// to the face normal (it then has no in-plane part).
    pub fn from_mesh_tag(
        mesh: &Mesh,
        face_tag: i32,
        mode: (usize, usize),
        direction: Option<[Field; 3]>,
    ) -> Option<PortSpec> {
        Self::from_mesh_tag_with_z0(mesh, face_tag, mode, direction, 1.0)
    }

    /// Like [`Self::from_mesh_tag`] but with an explicit reference impedance
    /// `z0` for the lumped `(0, 0)` mode (operator units, so a 50 ohm
    /// physical port is `z0 = 50.0 / 377.0`). Ignored for `TE_mn`
    /// modes whose impedance is dispersive.
    pub fn from_mesh_tag_with_z0(
        mesh: &Mesh,
        face_tag: i32,
        mode: (usize, usize),
        direction: Option<[Field; 3]>,
        z0: Field,
    ) -> Option<PortSpec> {
        let (tris, coords, nrm) = port_face(mesh, face_tag)?;
        // A lumped port's voltage-integration direction, if supplied,
        // becomes the port's transverse field axis. Reject one that is
        // zero or parallel to the face normal (no in-plane part to use).
        if let Some(d) = direction {
            let dl = dot(d, d).sqrt();
            if dl < 1e-12 {
                return None;
            }
            let dn = [d[0] / dl, d[1] / dl, d[2] / dl];
            let perp = dot(dn, nrm);
            if 1.0 - perp * perp < 1e-9 {
                return None;
            }
        }
        let mut rect = RectPort::from_face(&coords, nrm, mode, direction);
        rect.z0 = z0;
        Some(PortSpec { tris, mode: Some(PortMode::Rect(rect)) })
    }

    /// Build a *pure absorbing* boundary from a face group tag, no
    /// waveguide mode attached, just the characteristic non-reflecting
    /// flux at the face. This is the DG analogue of the FD backend's
    /// first-order ABC (Silver-Mueller): the upwind flux with a zero
    /// ghost state lets outgoing plane waves leave at near-normal
    /// incidence without reflection.
    ///
    /// Useful for terminating an air box around a small radiating /
    /// scattering structure when a full volumetric PML is overkill.
    /// Reflection grows with the angle of incidence; place the face
    /// several wavelengths from the source for clean termination.
    /// Returns `None` if the tag carries no triangles.
    pub fn absorbing_from_mesh_tag(
        mesh: &Mesh,
        face_tag: i32,
    ) -> Option<PortSpec> {
        let tris = mesh.tris_for_tag(face_tag).to_vec();
        if tris.is_empty() {
            return None;
        }
        Some(PortSpec { tris, mode: None })
    }

    /// Build a coaxial TEM port from a face group tag, collecting the port
    /// triangles and fitting the coaxial annulus to the face.
    ///
    /// The coax center defaults to the port-face centroid; `center` supplies
    /// an explicit axis point. The inner / outer radii are fitted from the
    /// extreme in-plane node distances to that center. Returns `None` if the
    /// tag carries no triangles.
    pub fn coax_from_mesh_tag(
        mesh: &Mesh,
        face_tag: i32,
        center: Option<[Field; 3]>,
    ) -> Option<PortSpec> {
        let (tris, coords, nrm) = port_face(mesh, face_tag)?;
        let coax = CoaxPort::from_face(&coords, nrm, center);
        Some(PortSpec { tris, mode: Some(PortMode::Coax(coax)) })
    }

    /// Build a numerically-solved **wave port** from a face group tag, a
    /// 2D cross-section eigensolve on the port face produces the mode
    /// profile, for cross-sections with no closed-form mode.
    ///
    /// Two solve paths, chosen by `k0`:
    /// - `k0 <= 0`: the scalar Helmholtz `TE`/`TM` solve, a
    ///   *homogeneously filled* hollow guide of arbitrary cross-section
    ///   (ridged, circular, L-shaped). `te` picks `TE`/`TM`.
    /// - `k0 > 0`: the full-vector hybrid solve at operating wavenumber
    ///   `k0` (in 1/mesh-length), with per-element `ε_r`, the
    ///   *inhomogeneous* (substrate + air) quasi-TEM mode of a
    ///   microstrip-class line. `eps_per_tet` supplies the relative
    ///   permittivity of every mesh tet (vacuum = 1 where absent); each
    ///   port-face triangle takes the `ε_r` of its adjacent tet. `te` is
    ///   ignored (the hybrid mode is neither pure TE nor TM).
    ///
    /// `mode_index` picks the mode by dominance (`0` = fundamental).
    /// Returns `None` if the tag has no triangles or the solve finds
    /// fewer than `mode_index + 1` modes.
    pub fn wave_from_mesh_tag(
        mesh: &Mesh,
        face_tag: i32,
        te: bool,
        mode_index: usize,
        eps_per_tet: Option<&[Field]>,
        k0: Field,
        pec_nodes: Option<&[bool]>,
    ) -> Option<PortSpec> {
        let tris = mesh.tris_for_tag(face_tag).to_vec();
        let solve = if k0 > 0.0 {
            PortSolve::Vector
        } else {
            PortSolve::Scalar(if te { ModeKind::Te } else { ModeKind::Tm })
        };
        let (nm, _) = solve_port_face(
            mesh, &tris, solve, mode_index, eps_per_tet.unwrap_or(&[]), k0, pec_nodes, None,
        )?;
        Some(PortSpec { tris, mode: Some(PortMode::Numerical(Box::new(nm))) })
    }

    /// Build a Floquet plane-wave port from a face group tag, collecting
    /// the port triangles and fitting the rectangular unit-cell face.
    ///
    /// `polarisation` picks TE (`s`-pol) or TM (`p`-pol). The scan angles
    /// `scan_theta`, `scan_phi` parametrise the incident-wave direction
    /// (radians, measured from the port inward normal / port-plane
    /// azimuth respectively); `scan_theta = 0` is the validated normal
    /// incidence case. `polarisation_override`, if `Some`, supplies an
    /// explicit in-plane polarisation direction; it is projected into the
    /// port plane and normalised. Returns `None` if the tag carries no
    /// triangles.
    ///
    /// The transverse Floquet phase factor `e^{-j·k_t·r_t}` is **dropped**
    /// at oblique scan, see [`FloquetPort`]'s docstring. Normal incidence
    /// is exact; oblique angles are a documented approximation matching
    /// the FD backend's convention.
    pub fn floquet_from_mesh_tag(
        mesh: &Mesh,
        face_tag: i32,
        polarisation: FloquetPolarisation,
        scan_theta: Field,
        scan_phi: Field,
        polarisation_override: Option<[Field; 3]>,
    ) -> Option<PortSpec> {
        let (tris, coords, nrm) = port_face(mesh, face_tag)?;
        let floquet = FloquetPort::from_face(
            &coords,
            nrm,
            polarisation,
            scan_theta,
            scan_phi,
            polarisation_override,
        );
        Some(PortSpec { tris, mode: Some(PortMode::Floquet(floquet)) })
    }
}

/// A periodic boundary pair, two opposite mesh faces whose triangles are
/// matched across the period translation, so a DG face on side A's element
/// sees side B's partner element as its neighbour (and vice-versa).
///
/// The pair is unordered: the matcher infers the translation vector from
/// the two face centroids and walks both sides symmetrically. After the
/// match, periodic boundary faces look exactly like interior faces to the
/// flux kernel, same `neighbor`, `neighbor_local_face`, and face-node
/// `perm`. The numerical flux for a periodic face is therefore the
/// existing interior-face central / upwind flux, no special-casing.
#[derive(Clone, Debug)]
pub struct PeriodicSpec {
    /// Mesh triangle indices on side A.
    pub tris_a: Vec<usize>,
    /// Mesh triangle indices on side B.
    pub tris_b: Vec<usize>,
}

/// A 2D PEC plate sitting *inside* the simulation domain (e.g. a
/// microstrip trace between substrate and air). DG faces matched to a
/// triangle in this list have their `neighbor` overwritten to `MAX` on
/// BOTH element sides, so the default boundary-face logic (PEC ghost
/// state) kicks in symmetrically: each side of the plate sees the
/// other as a perfect conductor.
///
/// Domain-boundary faces (those with one neighbor `MAX` already)
/// remain PEC by default - this struct is *only* needed for internal
/// plates, where without it the face would otherwise be treated as a
/// transparent interior face by the central / upwind flux.
#[derive(Clone, Debug)]
pub struct PecSpec {
    /// Mesh triangle indices forming this PEC plate's faces.
    pub tris: Vec<usize>,
}

impl PecSpec {
    /// Build an internal-PEC plate spec from a face group tag. Returns
    /// `None` if the tag carries no triangles.
    pub fn from_mesh_tag(mesh: &Mesh, face_tag: i32) -> Option<PecSpec> {
        let tris = mesh.ftag_to_tri.get(&face_tag)?.clone();
        if tris.is_empty() {
            return None;
        }
        Some(PecSpec { tris })
    }
}

impl PeriodicSpec {
    /// Build a periodic pair from two face group tags, the periodic
    /// counterpart of [`PortSpec::from_mesh_tag`]. Returns `None` if
    /// either tag carries no triangles.
    pub fn from_mesh_tags(
        mesh: &Mesh,
        face_a: i32,
        face_b: i32,
    ) -> Option<PeriodicSpec> {
        let tris_a = mesh.ftag_to_tri.get(&face_a)?.clone();
        let tris_b = mesh.ftag_to_tri.get(&face_b)?.clone();
        if tris_a.is_empty() || tris_b.is_empty() {
            return None;
        }
        Some(PeriodicSpec { tris_a, tris_b })
    }
}

/// Resolved per-port data held by the operator.
struct PortData {
    /// The port's waveguide mode, if any.
    mode: Option<PortMode>,
    /// `(element, local_face)` of every boundary face on this port.
    faces: Vec<(usize, usize)>,
    /// Precomputed `(e_profile, h_profile)` per (face, face-node m), indexed
    /// `face_idx * n_face_nodes + m`. Depends only on geometry and the mode,
    /// so the per-timestep projection need not recompute it. Empty for
    /// absorbing-only ports (no mode).
    profiles: Vec<([Field; 3], [Field; 3])>,
}

/// Per-thread working buffers for [`MaxwellOperator::apply_element`], the
/// fixed-size scratch, allocated once and reused so the operator hot path
/// performs no per-element heap allocation.
struct Scratch {
    /// Element E / H fields, deinterleaved (`3·Np` each).
    ee: Vec<Field>,
    hh: Vec<Field>,
    /// Curl results dE / dH (`3·Np` each).
    de: Vec<Field>,
    dh: Vec<Field>,
    /// `element_curl_into` scratch, reference (`3·Np`) and physical (`9·Np`)
    /// derivatives.
    rd: Vec<Field>,
    pd: Vec<Field>,
}

impl Scratch {
    fn new(np: usize) -> Self {
        Scratch {
            ee: vec![0.0; 3 * np],
            hh: vec![0.0; 3 * np],
            de: vec![0.0; 3 * np],
            dh: vec![0.0; 3 * np],
            rd: vec![0.0; 3 * np],
            pd: vec![0.0; 9 * np],
        }
    }

    /// A non-allocating placeholder, the value swapped in when a real
    /// `Scratch` is returned to the pool.
    fn empty() -> Self {
        Scratch {
            ee: Vec::new(),
            hh: Vec::new(),
            de: Vec::new(),
            dh: Vec::new(),
            rd: Vec::new(),
            pd: Vec::new(),
        }
    }
}

/// Checkout handle for a pooled [`Scratch`]; returns it to the pool on drop,
/// so steady-state `apply` calls allocate no scratch at all.
struct ScratchGuard<'a> {
    pool: &'a Mutex<Vec<Scratch>>,
    scratch: Scratch,
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        let s = std::mem::replace(&mut self.scratch, Scratch::empty());
        self.pool.lock().unwrap().push(s);
    }
}

/// Per-job working buffers for the sparse assembly, reused across every
/// element a rayon worker folds, so `assemble_sparse`'s element loop
/// allocates nothing beyond the geometric growth of the output accumulators.
struct SparseFragment {
    /// Global probe vector, one DOF set to 1 at a time.
    probe: Vec<Field>,
    /// Element output block (`stride`).
    out: Vec<Field>,
    /// Operator scratch.
    scratch: Scratch,
    /// `(local_row, global_col, value)` triples for one element, cleared
    /// and refilled per element; pre-sized to the worst-case stencil count.
    entries: Vec<(usize, usize, Field)>,
    /// Accumulated CSR fragment for this job, in element order.
    col_idx: Vec<usize>,
    values: Vec<Field>,
    row_len: Vec<usize>,
}

impl SparseFragment {
    fn new(n: usize, stride: usize, np: usize) -> Self {
        SparseFragment {
            probe: vec![0.0; n],
            out: vec![0.0; stride],
            scratch: Scratch::new(np),
            // ≤ 5 stencil columns × stride columns × stride rows nonzeros.
            entries: Vec::with_capacity(5 * stride * stride),
            col_idx: Vec::new(),
            values: Vec::new(),
            row_len: Vec::new(),
        }
    }
}

/// Per-element electromagnetic material, diagonal relative permittivity /
/// permeability tensors and conductivity, in the solver's normalised units
/// (`c = ε₀ = μ₀ = 1`). Diagonal tensors cover isotropic and uniaxial /
/// biaxial media (and the uniaxial PML); fully off-diagonal tensors are a
/// future extension.
#[derive(Clone, Copy, Debug)]
pub struct ElemMaterial {
    /// Diagonal relative permittivity `(ε_x, ε_y, ε_z)`.
    pub eps: [Field; 3],
    /// Diagonal relative permeability `(μ_x, μ_y, μ_z)`.
    pub mu: [Field; 3],
    /// Electric conductivity `σ` (Ohmic loss).
    pub sigma: Field,
    /// Magnetic conductivity `σ*`, the magnetic-loss term. Setting
    /// `σ*/μ = σ/ε` gives an impedance-matched absorbing layer (no reflection
    /// at normal incidence).
    pub sigma_m: Field,
}

impl ElemMaterial {
    /// Vacuum, `ε = μ = 1`, no loss.
    pub const VACUUM: ElemMaterial = ElemMaterial {
        eps: [1.0; 3],
        mu: [1.0; 3],
        sigma: 0.0,
        sigma_m: 0.0,
    };

    /// An isotropic, lossless material from scalar `ε_r`, `μ_r`, `σ`.
    pub fn isotropic(eps: Field, mu: Field, sigma: Field) -> Self {
        ElemMaterial { eps: [eps; 3], mu: [mu; 3], sigma, sigma_m: 0.0 }
    }

    /// An impedance-matched absorbing material: `σ*/μ = σ/ε = nu`, so the
    /// wave is absorbed with no reflection at normal incidence.
    pub fn matched_absorber(eps: Field, mu: Field, nu: Field) -> Self {
        ElemMaterial {
            eps: [eps; 3],
            mu: [mu; 3],
            sigma: nu * eps,
            sigma_m: nu * mu,
        }
    }
}

/// Per-element Debye dispersive data, resolved at build time.
///
/// A Debye element runs the auxiliary-differential-equation (ADE) update: its
/// `eps` in [`ElemMaterial`] is `ε_∞`, and a per-node polarisation field `P`
/// relaxes by `Ṗ = a·P + g·E`, contributing the polarisation current
/// `−Ṗ/ε_∞` to Ampere's law. Stored only for the elements actually carrying a
/// Debye material; a non-dispersive problem has none and the augmented state
/// is byte-identical to the plain `[E,H]` system.
#[derive(Clone, Copy, Debug)]
struct DispersiveElem {
    /// Mesh element index this Debye data is attached to.
    elem: usize,
    /// Relaxation coefficient `a = -1/τ` of `Ṗ = a·P + g·E`.
    a: Field,
    /// Source gain `g = (ε_s − ε_∞)/τ` of `Ṗ = a·P + g·E`.
    g: Field,
    /// `1/ε_∞`, the augmented Ampere-law scaling on this element's E nodes.
    inv_eps_inf: Field,
}

/// Semi-discrete DG operator for the Maxwell curl equations on a tetrahedral
/// mesh with PEC outer walls and per-element materials.
///
/// State layout: the `[E,H]` block comes first, `y[(e*Np + node)*6 + comp]`
/// with `comp` 0..3 = E, 3..6 = H, `6·Np·n_elem` entries, unchanged from the
/// non-dispersive operator. When Debye dispersive materials are present an
/// auxiliary-polarisation block is APPENDED: `3·Np` entries per Debye element,
/// `P[base + node*3 + comp]`. With no dispersive material that block is empty
/// and `n_dof` is exactly `6·Np·n_elem`.
///
/// `apply` evaluates `dy/dt`. The numerical flux blends central (`alpha = 0`,
/// energy-conserving) and upwind (`alpha = 1`, dissipates the discontinuous
/// spurious modes).
pub struct MaxwellOperator {
    pub(crate) re: ReferenceElement,
    pub(crate) n_elem: usize,
    pub(crate) geom: Vec<GeometricFactors>,
    /// 4 faces per element, flattened: `faces[e*4 + f]`.
    pub(crate) faces: Vec<FaceInfo>,
    /// Upwind blend: 0 = central, 1 = full upwind.
    pub(crate) flux_alpha: Field,
    /// Per-element diagonal `1/ε`, `1/μ`, `σ/ε` (electric), `σ*/μ` (magnetic).
    pub(crate) inv_eps: Vec<[Field; 3]>,
    pub(crate) inv_mu: Vec<[Field; 3]>,
    pub(crate) sigma_eps: Vec<[Field; 3]>,
    pub(crate) sigma_mu: Vec<[Field; 3]>,
    /// Reusable per-thread scratch buffers, keeps `apply` allocation-free
    /// after the first call (see [`Scratch`]).
    scratch_pool: Mutex<Vec<Scratch>>,
    /// Resolved port data, faces and waveguide mode per port.
    ports: Vec<PortData>,
    /// Per-Debye-element ADE data, in P-block order (slot `s` owns the P
    /// segment `[(6*Np*n_elem) + s*3*Np .. + 3*Np]`). Empty for a
    /// non-dispersive problem.
    disp: Vec<DispersiveElem>,
    /// Map mesh element index -> P-block slot, or `usize::MAX` if the element
    /// is non-dispersive. Length `n_elem`.
    disp_slot: Vec<usize>,
}

/// For one periodic boundary triangle, locate its owning (element, local
/// face). A periodic face is always a domain-boundary triangle, exactly
/// one of `mesh.tri_to_tet[tri]` slots is `usize::MAX`, the other carries
/// the owning element.
fn boundary_tri_owner(
    mesh: &Mesh,
    tri: usize,
) -> Option<(usize, usize)> {
    let owner = mesh.tri_to_tet[tri]
        .iter()
        .copied()
        .find(|&t| t != usize::MAX)?;
    let local_face = mesh.tet_to_tri[owner]
        .iter()
        .position(|&x| x == tri)?;
    Some((owner, local_face))
}

/// Centroid of a periodic face, the average of the centroids of its
/// boundary triangles, weighted by area. Used to infer the period
/// translation between the two sides of a [`PeriodicSpec`].
fn periodic_face_centroid(
    mesh: &Mesh,
    tris: &[usize],
) -> [Field; 3] {
    let mut acc = [0.0 as Field; 3];
    let mut total = 0.0 as Field;
    for &tri in tris {
        let [a, b, c] = mesh.tris[tri];
        let pa = mesh.nodes[a].map(|x| x as Field);
        let pb = mesh.nodes[b].map(|x| x as Field);
        let pc = mesh.nodes[c].map(|x| x as Field);
        let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let n = cross(e1, e2);
        let area =
            0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        for k in 0..3 {
            acc[k] += area * (pa[k] + pb[k] + pc[k]) / 3.0;
        }
        total += area;
    }
    assert!(total > 0.0, "periodic face has zero total area");
    [acc[0] / total, acc[1] / total, acc[2] / total]
}

/// Match periodic boundary faces across the period translation, then fill
/// in their FaceInfo neighbour / perm so the existing interior-face flux
/// path handles them with no special-casing.
///
/// The translation vector is inferred from the two face centroids (mean of
/// each side's triangle centroids). For each side-A boundary triangle we
/// find its side-B partner by matching the side-A centroid plus the
/// translation against B's triangle centroids; the face-node permutation
/// is then a per-node nearest-neighbour match in the same translated frame.
/// The match is symmetric, A → B and B → A are both wired up, so the
/// kernel's neighbour branch fires on either side.
fn link_periodic_faces(
    mesh: &Mesh,
    re: &ReferenceElement,
    geom: &[GeometricFactors],
    tri_to_port: &[usize],
    faces: &mut [FaceInfo],
    tris_a: &[usize],
    tris_b: &[usize],
) {
    assert!(
        tris_a.len() == tris_b.len(),
        "periodic pair has different triangle counts on each side: \
         A has {}, B has {}",
        tris_a.len(),
        tris_b.len(),
    );

    // Reject a face that is already a port, periodic and port are mutually
    // exclusive (a face cannot be both a periodic neighbour and a
    // characteristic boundary).
    for &tri in tris_a.iter().chain(tris_b.iter()) {
        assert!(
            tri_to_port[tri] == usize::MAX,
            "triangle {tri} is marked both port and periodic",
        );
    }

    // Period translation: B-centroid minus A-centroid.
    let ca = periodic_face_centroid(mesh, tris_a);
    let cb = periodic_face_centroid(mesh, tris_b);
    let trans = [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]];
    let trans_mag =
        (trans[0] * trans[0] + trans[1] * trans[1] + trans[2] * trans[2])
            .sqrt();
    let tri_tol = (PERIODIC_MATCH_REL_TOL * trans_mag)
        .max(PERIODIC_MATCH_ABS_FLOOR);
    let tri_tol2 = tri_tol * tri_tol;

    // Per-triangle centroid on each side, the matching key.
    let tri_centroid = |tri: usize| -> [Field; 3] {
        let [a, b, c] = mesh.tris[tri];
        let pa = mesh.nodes[a].map(|x| x as Field);
        let pb = mesh.nodes[b].map(|x| x as Field);
        let pc = mesh.nodes[c].map(|x| x as Field);
        [
            (pa[0] + pb[0] + pc[0]) / 3.0,
            (pa[1] + pb[1] + pc[1]) / 3.0,
            (pa[2] + pb[2] + pc[2]) / 3.0,
        ]
    };
    let centroids_a: Vec<[Field; 3]> =
        tris_a.iter().map(|&t| tri_centroid(t)).collect();
    let centroids_b: Vec<[Field; 3]> =
        tris_b.iter().map(|&t| tri_centroid(t)).collect();

    // Match each A triangle to a B triangle by translated-centroid distance.
    // `partner[i_a] = i_b`; the inverse map ensures the pairing is a true
    // bijection (no two A triangles share a B partner).
    let mut partner_a_to_b = vec![usize::MAX; tris_a.len()];
    let mut partner_b_to_a = vec![usize::MAX; tris_b.len()];
    for (i_a, &ca_i) in centroids_a.iter().enumerate() {
        let target =
            [ca_i[0] + trans[0], ca_i[1] + trans[1], ca_i[2] + trans[2]];
        let (mut best, mut bi) = (Field::MAX, usize::MAX);
        for (i_b, &cb_i) in centroids_b.iter().enumerate() {
            let d = (target[0] - cb_i[0]).powi(2)
                + (target[1] - cb_i[1]).powi(2)
                + (target[2] - cb_i[2]).powi(2);
            if d < best {
                best = d;
                bi = i_b;
            }
        }
        assert!(
            best < tri_tol2,
            "periodic triangle {} on side A: no partner on side B within \
             tolerance {tri_tol:e} (best distance {:e}, period magnitude \
             {trans_mag:e})",
            tris_a[i_a],
            best.sqrt(),
        );
        assert!(
            partner_b_to_a[bi] == usize::MAX,
            "periodic side-B triangle {} matched by two side-A triangles \
             ({} and {}); a periodic pair must be a bijection",
            tris_b[bi],
            tris_a[partner_b_to_a[bi]],
            tris_a[i_a],
        );
        partner_a_to_b[i_a] = bi;
        partner_b_to_a[bi] = i_a;
    }

    // Face-node coordinates of a given (element, local face) under the
    // operator's reference element, the same map used by the interior
    // matcher.
    let face_coords = |e: usize, f: usize| -> Vec<[Field; 3]> {
        re.face_nodes[f]
            .iter()
            .map(|&vi| geom[e].map(re.nodes[vi]))
            .collect()
    };

    // Per-pair: wire each side's FaceInfo to the partner element / local
    // face and compute the face-node permutation in the translated frame.
    for (i_a, &i_b) in partner_a_to_b.iter().enumerate() {
        let tri_a = tris_a[i_a];
        let tri_b = tris_b[i_b];
        let (e_a, f_a) = boundary_tri_owner(mesh, tri_a).unwrap_or_else(
            || panic!("periodic triangle {tri_a} carries no owning tet"),
        );
        let (e_b, f_b) = boundary_tri_owner(mesh, tri_b).unwrap_or_else(
            || panic!("periodic triangle {tri_b} carries no owning tet"),
        );
        let here_a = face_coords(e_a, f_a);
        let here_b = face_coords(e_b, f_b);

        // For each face-node of A, find the partner face-node on B by
        // matching A's coordinates (after the period translation) to B's.
        // This is the direct analogue of the interior-face matcher above,
        // with the translation supplying the periodic glue.
        let perm_a: Vec<usize> = here_a
            .iter()
            .map(|p| {
                let target =
                    [p[0] + trans[0], p[1] + trans[1], p[2] + trans[2]];
                let (mut best, mut bm) = (Field::MAX, 0);
                for (m2, q) in here_b.iter().enumerate() {
                    let d = (target[0] - q[0]).powi(2)
                        + (target[1] - q[1]).powi(2)
                        + (target[2] - q[2]).powi(2);
                    if d < best {
                        best = d;
                        bm = m2;
                    }
                }
                assert!(
                    best < tri_tol2,
                    "periodic face-node A({tri_a},{}) found no B partner \
                     within tolerance {tri_tol:e} (best {:e})",
                    re.face_nodes[f_a][0],
                    best.sqrt(),
                );
                bm
            })
            .collect();
        // The mirror permutation, used to wire side B's FaceInfo.
        let perm_b: Vec<usize> = here_b
            .iter()
            .map(|p| {
                let target =
                    [p[0] - trans[0], p[1] - trans[1], p[2] - trans[2]];
                let (mut best, mut bm) = (Field::MAX, 0);
                for (m2, q) in here_a.iter().enumerate() {
                    let d = (target[0] - q[0]).powi(2)
                        + (target[1] - q[1]).powi(2)
                        + (target[2] - q[2]).powi(2);
                    if d < best {
                        best = d;
                        bm = m2;
                    }
                }
                assert!(best < tri_tol2, "periodic face-node B unmatched");
                bm
            })
            .collect();

        // Sanity: a face originally tagged for periodic linking must be a
        // domain boundary in the topology (neighbor = MAX); if not, the
        // caller's tris list is corrupt (e.g. an interior face was passed).
        let fi_a = &mut faces[e_a * 4 + f_a];
        assert!(
            fi_a.neighbor == usize::MAX,
            "periodic triangle {tri_a} is not on the domain boundary",
        );
        fi_a.neighbor = e_b;
        fi_a.neighbor_local_face = f_b;
        fi_a.perm = perm_a;

        let fi_b = &mut faces[e_b * 4 + f_b];
        assert!(
            fi_b.neighbor == usize::MAX,
            "periodic triangle {tri_b} is not on the domain boundary",
        );
        fi_b.neighbor = e_a;
        fi_b.neighbor_local_face = f_a;
        fi_b.perm = perm_b;
    }
}

impl MaxwellOperator {
    /// Build a vacuum operator (`ε = μ = 1`, `σ = 0`).
    pub fn new(mesh: &Mesh, order: usize, flux_alpha: Field) -> Self {
        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        Self::new_with_materials(mesh, order, flux_alpha, &vacuum)
    }

    /// Build the operator with per-element materials and the given upwind
    /// blend (`flux_alpha` in `[0, 1]`).
    pub fn new_with_materials(
        mesh: &Mesh,
        order: usize,
        flux_alpha: Field,
        materials: &[ElemMaterial],
    ) -> Self {
        Self::new_with_materials_ports(mesh, order, flux_alpha, materials, &[])
    }

    /// Build the operator with per-element materials and waveguide ports.
    /// Faces on a port behave as a characteristic boundary; non-port
    /// boundary faces are PEC walls.
    pub fn new_with_materials_ports(
        mesh: &Mesh,
        order: usize,
        flux_alpha: Field,
        materials: &[ElemMaterial],
        ports: &[PortSpec],
    ) -> Self {
        Self::new_with_materials_ports_dispersive(
            mesh, order, flux_alpha, materials, ports, &[],
        )
    }

    /// Build the operator with per-element materials, waveguide ports and an
    /// optional list of Debye dispersive elements.
    ///
    /// `dispersive` carries `(element_index, DebyeMaterial)` pairs, each
    /// listed element runs the auxiliary-polarisation ADE, and its
    /// `materials[element]` entry must already carry `ε = ε_∞` (the
    /// high-frequency permittivity) so the static curl term and the dispersive
    /// polarisation current stay consistent. With an empty `dispersive` list
    /// the operator is byte-identical to [`new_with_materials_ports`](Self::new_with_materials_ports):
    /// same `n_dof`, same state layout, same behaviour.
    pub fn new_with_materials_ports_dispersive(
        mesh: &Mesh,
        order: usize,
        flux_alpha: Field,
        materials: &[ElemMaterial],
        ports: &[PortSpec],
        dispersive: &[(usize, DebyeMaterial)],
    ) -> Self {
        Self::new_with_materials_ports_dispersive_periodic(
            mesh, order, flux_alpha, materials, ports, dispersive, &[],
        )
    }

    /// Like [`new_with_materials_ports_dispersive`](Self::new_with_materials_ports_dispersive)
    /// but additionally accepts periodic boundary pairs. Each entry in
    /// `periodic` declares two opposite mesh faces whose triangles are
    /// matched across the period translation, so DG faces on either side
    /// see the partner element across the period as their neighbour.
    ///
    /// The numerical flux on a periodic face is then the existing
    /// interior-face flux, no kernel change. With `periodic` empty the
    /// operator is byte-identical to the non-periodic build.
    pub fn new_with_materials_ports_dispersive_periodic(
        mesh: &Mesh,
        order: usize,
        flux_alpha: Field,
        materials: &[ElemMaterial],
        ports: &[PortSpec],
        dispersive: &[(usize, DebyeMaterial)],
        periodic: &[PeriodicSpec],
    ) -> Self {
        Self::new_full(
            mesh, order, flux_alpha, materials, ports, dispersive,
            periodic, &[],
        )
    }

    /// Most-general operator builder: like the
    /// `_dispersive_periodic` form, plus internal-PEC plates. The
    /// PEC plates retag both sides of each listed triangle as boundary
    /// faces so the existing PEC ghost-state logic applies on both
    /// sides; this lets microstrip / RFIC traces, ground planes, and
    /// other thin internal conductors be modelled as zero-thickness
    /// perfect conductors.
    pub fn new_full(
        mesh: &Mesh,
        order: usize,
        flux_alpha: Field,
        materials: &[ElemMaterial],
        ports: &[PortSpec],
        dispersive: &[(usize, DebyeMaterial)],
        periodic: &[PeriodicSpec],
        pec_plates: &[PecSpec],
    ) -> Self {
        let re = ReferenceElement::new(order);
        let geom = all_geometric_factors(mesh);
        let topo = FaceTopology::build(mesh);
        let n_elem = mesh.n_tets();

        let face_coords = |e: usize, f: usize| -> Vec<[Field; 3]> {
            re.face_nodes[f]
                .iter()
                .map(|&vi| geom[e].map(re.nodes[vi]))
                .collect()
        };

        // Triangle → port index, so each face can be tagged as it is built.
        let mut tri_to_port = vec![usize::MAX; mesh.tris.len()];
        for (pi, port) in ports.iter().enumerate() {
            for &tri in &port.tris {
                tri_to_port[tri] = pi;
            }
        }

        let mut faces = Vec::with_capacity(n_elem * 4);
        for e in 0..n_elem {
            for f in 0..4 {
                let df = topo.face(e, f);
                let fscale = 2.0 * df.area as Field / geom[e].det.abs();
                let perm = if df.neighbor == usize::MAX {
                    Vec::new()
                } else {
                    let here = face_coords(e, f);
                    let there =
                        face_coords(df.neighbor, df.neighbor_local_face);
                    here.iter()
                        .map(|p| {
                            let (mut best, mut bm) = (Field::MAX, 0);
                            for (m2, q) in there.iter().enumerate() {
                                let d = (p[0] - q[0]).powi(2)
                                    + (p[1] - q[1]).powi(2)
                                    + (p[2] - q[2]).powi(2);
                                if d < best {
                                    best = d;
                                    bm = m2;
                                }
                            }
                            assert!(best < 1e-18, "unmatched face node");
                            bm
                        })
                        .collect()
                };
                faces.push(FaceInfo {
                    normal: df.normal.map(|x| x as Field),
                    fscale,
                    neighbor: df.neighbor,
                    neighbor_local_face: df.neighbor_local_face,
                    perm,
                    port: tri_to_port[df.tri],
                });
            }
        }

        // Internal PEC plates: for each tagged triangle, retag both
        // element-side faces as boundary (neighbor = MAX). The
        // existing PEC ghost-state logic then applies symmetrically
        // on both sides. Done *before* the periodic matcher because
        // periodic links also need a sane neighbor map; an internal
        // PEC plate is by definition not periodic.
        let mut pec_tri_set = vec![false; mesh.tris.len()];
        for spec in pec_plates {
            for &tri in &spec.tris {
                pec_tri_set[tri] = true;
            }
        }
        for e in 0..n_elem {
            for f in 0..4 {
                let df = topo.face(e, f);
                if pec_tri_set[df.tri] {
                    // Sever this face from its neighbor on both
                    // sides; PEC boundary logic then applies. Also
                    // clear the perm (used only for interior faces).
                    faces[e * 4 + f].neighbor = usize::MAX;
                    faces[e * 4 + f].neighbor_local_face = 0;
                    faces[e * 4 + f].perm = Vec::new();
                }
            }
        }
        // Periodic boundary matcher, link each periodic face to its partner
        // across the period translation. After this each periodic face's
        // FaceInfo carries the partner element and the partner's local face,
        // and a face-node permutation that maps this face's node ordering to
        // the partner's. The DG flux on a periodic face then looks exactly
        // like an interior face to the kernel (CPU and GPU alike), no
        // special-case branch needed.
        for spec in periodic {
            link_periodic_faces(
                mesh,
                &re,
                &geom,
                &tri_to_port,
                &mut faces,
                &spec.tris_a,
                &spec.tris_b,
            );
        }

        assert_eq!(materials.len(), n_elem, "one material per element");
        let recip = |v: [Field; 3]| [1.0 / v[0], 1.0 / v[1], 1.0 / v[2]];
        let inv_eps: Vec<[Field; 3]> =
            materials.iter().map(|m| recip(m.eps)).collect();
        let inv_mu: Vec<[Field; 3]> =
            materials.iter().map(|m| recip(m.mu)).collect();
        let sigma_eps: Vec<[Field; 3]> = materials
            .iter()
            .map(|m| [m.sigma / m.eps[0], m.sigma / m.eps[1], m.sigma / m.eps[2]])
            .collect();
        let sigma_mu: Vec<[Field; 3]> = materials
            .iter()
            .map(|m| {
                [
                    m.sigma_m / m.mu[0],
                    m.sigma_m / m.mu[1],
                    m.sigma_m / m.mu[2],
                ]
            })
            .collect();
        // Resolve per-port data, collect each port's boundary faces.
        let mut port_data: Vec<PortData> = ports
            .iter()
            .map(|spec| PortData {
                mode: spec.mode.clone(),
                faces: Vec::new(),
                profiles: Vec::new(),
            })
            .collect();
        for e in 0..n_elem {
            for f in 0..4 {
                let pi = faces[e * 4 + f].port;
                if pi != usize::MAX {
                    port_data[pi].faces.push((e, f));
                }
            }
        }
        // Precompute the modal field profiles per port-face node. They
        // depend only on geometry and the mode, so port_source and the
        // per-timestep port_modal_projections need not recompute them.
        {
            let nfp = re.n_face_nodes;
            for pd in port_data.iter_mut() {
                if let Some(mode) = &pd.mode {
                    pd.profiles = Vec::with_capacity(pd.faces.len() * nfp);
                    for &(e, f) in &pd.faces {
                        for m in 0..nfp {
                            let vi = re.face_nodes[f][m];
                            let x = geom[e].map(re.nodes[vi]);
                            pd.profiles.push((mode.e_profile(x), mode.h_profile(x)));
                        }
                    }
                }
            }
        }

        // Resolve the Debye dispersive elements. Each listed element gets a
        // P-block slot, in list order; `disp_slot` maps element -> slot so
        // `apply` can find a tet's polarisation segment in O(1). An empty
        // list leaves `disp` empty and the operator non-dispersive.
        let mut disp: Vec<DispersiveElem> = Vec::with_capacity(dispersive.len());
        let mut disp_slot = vec![usize::MAX; n_elem];
        for &(elem, mat) in dispersive {
            assert!(elem < n_elem, "dispersive element index out of range");
            assert!(
                disp_slot[elem] == usize::MAX,
                "element {elem} listed twice as dispersive"
            );
            let (a, g) = mat.relaxation_coeffs();
            disp_slot[elem] = disp.len();
            disp.push(DispersiveElem {
                elem,
                a,
                g,
                inv_eps_inf: 1.0 / mat.eps_inf,
            });
        }

        MaxwellOperator {
            re,
            n_elem,
            geom,
            faces,
            flux_alpha,
            inv_eps,
            inv_mu,
            sigma_eps,
            sigma_mu,
            scratch_pool: Mutex::new(Vec::new()),
            ports: port_data,
            disp,
            disp_slot,
        }
    }

    /// Degrees of freedom: `6·Np·n_elem` for the `[E,H]` block plus `3·Np`
    /// per Debye dispersive element for the appended auxiliary-polarisation
    /// block. With no dispersive material this is exactly `6·Np·n_elem`.
    pub fn n_dof(&self) -> usize {
        6 * self.re.n_nodes * self.n_elem
            + 3 * self.re.n_nodes * self.disp.len()
    }

    /// Length of the leading `[E,H]` block, `6·Np·n_elem`, the offset at
    /// which the appended polarisation block begins.
    fn eh_len(&self) -> usize {
        6 * self.re.n_nodes * self.n_elem
    }

    /// Number of tetrahedra (DG elements).
    pub fn n_elem(&self) -> usize {
        self.n_elem
    }

    /// Number of Debye dispersive elements (P-block slots).
    pub fn n_dispersive(&self) -> usize {
        self.disp.len()
    }

    /// Global DOF index for a field component at the mesh node nearest
    /// `point`, the hook for a soft source or a field probe.
    /// `field`: 0 = E, 1 = H. `comp`: 0 = x, 1 = y, 2 = z.
    pub fn nearest_node_dof(
        &self,
        point: [Field; 3],
        field: usize,
        comp: usize,
    ) -> usize {
        let np = self.re.n_nodes;
        let (mut best_d, mut best) = (Field::MAX, 0);
        for e in 0..self.n_elem {
            for node in 0..np {
                let p = self.geom[e].map(self.re.nodes[node]);
                let d = (p[0] - point[0]).powi(2)
                    + (p[1] - point[1]).powi(2)
                    + (p[2] - point[2]).powi(2);
                if d < best_d {
                    best_d = d;
                    best = (e * np + node) * 6 + field * 3 + comp;
                }
            }
        }
        best
    }

    /// Physical coordinates of every DG node, `n_elem·Np` points in state
    /// order, `point[e*Np + node]`. The hook for a field export.
    pub fn node_coords(&self) -> Vec<[Field; 3]> {
        let np = self.re.n_nodes;
        let mut pts = Vec::with_capacity(self.n_elem * np);
        for e in 0..self.n_elem {
            for node in 0..np {
                pts.push(self.geom[e].map(self.re.nodes[node]));
            }
        }
        pts
    }

    /// The four reference-node local indices at the tet corners, ordered
    /// `(0,0,0), (1,0,0), (0,1,0), (0,0,1)`, the connectivity hook for a
    /// linear-tetrahedron VTK export.
    pub fn corner_local_nodes(&self) -> [usize; 4] {
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        corners.map(|c| {
            self.re
                .nodes
                .iter()
                .position(|n| {
                    (n[0] - c[0]).abs() < 1e-12
                        && (n[1] - c[1]).abs() < 1e-12
                        && (n[2] - c[2]).abs() < 1e-12
                })
                .expect("reference element carries the tet-corner nodes")
        })
    }

    /// Number of ports on the operator.
    pub fn n_ports(&self) -> usize {
        self.ports.len()
    }

    /// Cutoff angular frequency of port `port_idx`'s waveguide mode, in the
    /// operator's normalised units (`0` if the port carries no mode).
    pub fn port_cutoff(&self, port_idx: usize) -> Field {
        self.ports[port_idx]
            .mode
            .as_ref()
            .map_or(0.0, |m| m.cutoff())
    }

    /// `true` if port `port_idx` carries a waveguide mode (rectangular,
    /// coaxial or Floquet). A port with no mode is a pure characteristic
    /// absorbing boundary (ABC), not an input / output channel, port
    /// inspection and S-parameter extraction routines should skip those.
    /// Number of resolved `(element, local_face)` boundary-face pairs
    /// for port `port_idx`. For a port plate on a *domain boundary*
    /// (the validated case) this equals one face per triangle. For an
    /// internal plate that ended up tagged on both sides of an
    /// interior face, the count is twice the triangle count - a
    /// diagnostic signal that the port is not properly boundary-attached.
    pub fn port_n_faces(&self, port_idx: usize) -> usize {
        self.ports[port_idx].faces.len()
    }

    /// Number of port faces whose neighbor element is *another* tet
    /// (interior face, both sides in the domain) vs `MAX` (boundary
    /// face, the side facing outside the domain). A *correctly*
    /// boundary-attached port has zero interior faces; an internal
    /// plate that ended up port-tagged on both sides has all interior
    /// faces.
    pub fn port_n_interior_faces(&self, port_idx: usize) -> usize {
        let pd = &self.ports[port_idx];
        pd.faces
            .iter()
            .filter(|&&(e, f)| {
                self.faces[e * 4 + f].neighbor != usize::MAX
            })
            .count()
    }

    pub fn port_has_mode(&self, port_idx: usize) -> bool {
        self.ports[port_idx].mode.is_some()
    }

    /// Modal wave impedance `Z(omega)` of port `port_idx` at angular
    /// frequency `omega`, in the operator's normalised units. Returns
    /// `0` for an absorbing-only port (no mode). The forward / backward
    /// split `A, B = (P_e +/- Z * P_h) / 2` uses this per frequency.
    pub fn port_impedance(&self, port_idx: usize, omega: Field) -> Field {
        self.ports[port_idx]
            .mode
            .as_ref()
            .map_or(0.0, |m| m.te_impedance(omega))
    }

    /// Spatial source vector `b_spatial` for driving port `port_idx` with a
    /// unit-amplitude waveform: the system `dy/dt = A·y + b_spatial·g(t)`
    /// then carries the port's incident mode modulated by `g(t)`.
    ///
    /// The port flux uses the ghost state `(E⁺, H⁺) = (E_inc, H_inc)`; its
    /// `(E⁻, H⁻)` part is already the absorbing operator `A`, and the
    /// incident-field part is this rank-1 source, the lift of
    /// `(n̂×h_t − n̂×(n̂×e_t))` (electric) and `(−n̂×e_t − n̂×(n̂×h_t))`
    /// (magnetic) over the port faces, with the per-element material
    /// scaling applied.
    pub fn port_source(&self, port_idx: usize) -> Vec<Field> {
        let np = self.re.n_nodes;
        let nfp = self.re.n_face_nodes;
        let cols = 4 * nfp;
        let mut b = vec![0.0; self.n_dof()];
        let pd = &self.ports[port_idx];
        if pd.mode.is_none() {
            return b; // absorbing-only port, nothing to inject
        }
        for (face_idx, &(e, f)) in pd.faces.iter().enumerate() {
            let fi = &self.faces[e * 4 + f];
            let n = fi.normal;
            let coef = 0.5 * fi.fscale;
            let (ie, im) = (self.inv_eps[e], self.inv_mu[e]);
            for m in 0..nfp {
                let (et, ht) = pd.profiles[face_idx * nfp + m];
                // Incident-field flux: jumps [E] = −e_t, [H] = −h_t.
                let nxe = cross(n, et);
                let nxh = cross(n, ht);
                let nne = cross(n, nxe);
                let nnh = cross(n, nxh);
                let s_de = [
                    nxh[0] - nne[0],
                    nxh[1] - nne[1],
                    nxh[2] - nne[2],
                ];
                let s_dh = [
                    -nxe[0] - nnh[0],
                    -nxe[1] - nnh[1],
                    -nxe[2] - nnh[2],
                ];
                for i in 0..np {
                    let w = coef * self.re.lift[i * cols + f * nfp + m];
                    let base = (e * np + i) * 6;
                    for c in 0..3 {
                        b[base + c] += ie[c] * w * s_de[c];
                        b[base + 3 + c] += im[c] * w * s_dh[c];
                    }
                }
            }
        }
        b
    }

    /// Modal field projections `(P_e, P_h)` at port `port_idx` for the
    /// current state `y`, the port-face E and H surface-integral-projected
    /// onto the mode's transverse profile.
    ///
    /// Both propagation directions share the transverse-E profile but
    /// carry transverse H scaled by `±1/Z` (the modal impedance), so
    /// `E_t = e_t·(A+B)` and `H_t = (h_t/Z)·(A−B)`. Hence `P_e = A+B` and
    /// `Z·P_h = A−B`; the forward/backward split `A,B = (P_e ± Z·P_h)/2`
    /// is done per frequency since `Z` is dispersive, feed the recorded
    /// `(P_e, P_h)` time series through a transform first.
    pub fn port_modal_projections(
        &self,
        y: &[Field],
        port_idx: usize,
    ) -> (Field, Field) {
        let np = self.re.n_nodes;
        let nfp = self.re.n_face_nodes;
        let pd = &self.ports[port_idx];
        assert!(pd.mode.is_some(), "port has no mode for extraction");
        let (mut e_dot, mut e_norm) = (0.0, 0.0); // ∮E·e_t, ∮e_t·e_t
        let (mut h_dot, mut h_norm) = (0.0, 0.0);
        for (face_idx, &(e, f)) in pd.faces.iter().enumerate() {
            let fi = &self.faces[e * 4 + f];
            let area = fi.fscale * self.geom[e].det.abs() / 2.0;
            let wgt = &self.re.face_node_weights[f];
            for m in 0..nfp {
                let vi = self.re.face_nodes[f][m];
                let (et, ht) = pd.profiles[face_idx * nfp + m];
                let w = 2.0 * area * wgt[m];
                let base = (e * np + vi) * 6;
                let ef = [y[base], y[base + 1], y[base + 2]];
                let hf = [y[base + 3], y[base + 4], y[base + 5]];
                e_dot += w * dot(ef, et);
                e_norm += w * dot(et, et);
                h_dot += w * dot(hf, ht);
                h_norm += w * dot(ht, ht);
            }
        }
        let p_e = if e_norm > 0.0 { e_dot / e_norm } else { 0.0 };
        let p_h = if h_norm > 0.0 { h_dot / h_norm } else { 0.0 };
        (p_e, p_h)
    }

    /// Evaluate `dy/dt = A·y`, allocating the result. See [`apply_into`](Self::apply_into)
    /// for the allocation-free form.
    pub fn apply(&self, y: &[Field]) -> Vec<Field> {
        let mut dy = vec![0.0; self.n_dof()];
        self.apply_into(y, &mut dy);
        dy
    }

    /// Evaluate `dy/dt = A·y` into the caller's buffer, the allocation-free
    /// hot path. The per-element work is independent (each element writes
    /// only its own slice of `dy`), so it runs in parallel across cores;
    /// every worker reuses a pooled `Scratch`, so after the first call
    /// this performs no heap allocation at all.
    ///
    /// With Debye dispersive materials the appended polarisation block is
    /// updated after the `[E,H]` block: `Ṗ = a·P + g·E` is a per-node local
    /// ODE (no spatial derivative), and the `[E,H]` block already picked up
    /// the `−Ṗ/ε_∞` polarisation current on every dispersive element.
    pub fn apply_into(&self, y: &[Field], dy: &mut [Field]) {
        debug_assert_eq!(dy.len(), self.n_dof());
        let np = self.re.n_nodes;
        let stride = np * 6;
        let eh_len = self.eh_len();
        let (dy_eh, dy_p) = dy.split_at_mut(eh_len);
        // The [E,H] block, per-element curl + flux + materials, plus the
        // dispersive polarisation current on Debye elements (reads P from
        // y's appended block).
        //
        // Elements are handed out in coarse contiguous runs, a few per
        // worker thread. A per-element chunk (`stride` = 60 f64) is 7.5
        // cache lines, so neighbouring elements share a `dy` line;
        // fine-grained work-stealing then false-shared that line across
        // every thread and `apply` stopped scaling past ~6 cores. A
        // contiguous run per task confines sharing to the run boundaries.
        let chunk = (self.n_elem
            / (APPLY_TASKS_PER_THREAD * rayon::current_num_threads()))
        .max(1);
        dy_eh
            .par_chunks_mut(chunk * stride)
            .enumerate()
            .for_each_init(
                || self.checkout_scratch(np),
                |guard, (ci, run)| {
                    let e0 = ci * chunk;
                    for (le, out) in run.chunks_mut(stride).enumerate() {
                        self.apply_element(
                            e0 + le, y, out, &mut guard.scratch,
                        );
                    }
                },
            );
        // The appended polarisation block, one local relaxation ODE per
        // Debye element: dP = a*P + g*E. No spatial coupling; dealt out in
        // coarse contiguous runs of slots, like the [E,H] block above, so
        // neighbouring threads do not false-share the P-block lines.
        if !self.disp.is_empty() {
            let p_stride = np * 3;
            let chunk = (self.disp.len()
                / (APPLY_TASKS_PER_THREAD * rayon::current_num_threads()))
            .max(1);
            dy_p
                .par_chunks_mut(chunk * p_stride)
                .enumerate()
                .for_each(|(ci, run)| {
                    let s0 = ci * chunk;
                    for (ls, out) in run.chunks_mut(p_stride).enumerate() {
                        let slot = s0 + ls;
                        let d = &self.disp[slot];
                        let e_base = d.elem * stride;
                        let p_base = eh_len + slot * p_stride;
                        for node in 0..np {
                            for c in 0..3 {
                                let e_val = y[e_base + node * 6 + c];
                                let p_val = y[p_base + node * 3 + c];
                                out[node * 3 + c] =
                                    d.a * p_val + d.g * e_val;
                            }
                        }
                    }
                });
        }
    }

    /// Take a [`Scratch`] from the pool, allocating one only on first use.
    fn checkout_scratch(&self, np: usize) -> ScratchGuard<'_> {
        let scratch = self
            .scratch_pool
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| Scratch::new(np));
        ScratchGuard { pool: &self.scratch_pool, scratch }
    }

    /// Compute element `e`'s block of `dy = A·y` into `out`, its `Np·6`
    /// contiguous slice. `s` supplies the reusable per-thread working
    /// buffers, so this allocates nothing.
    fn apply_element(
        &self,
        e: usize,
        y: &[Field],
        out: &mut [Field],
        s: &mut Scratch,
    ) {
        let np = self.re.n_nodes;
        let nfp = self.re.n_face_nodes;
        let cols = 4 * nfp;
        let base = e * np * 6;
        for node in 0..np {
            for c in 0..3 {
                s.ee[node * 3 + c] = y[base + node * 6 + c];
                s.hh[node * 3 + c] = y[base + node * 6 + 3 + c];
            }
        }

        // Volume term:  dE = ∇×H,  dH = -∇×E.
        element_curl_into(
            &self.re, &self.geom[e], &s.hh, &mut s.de, &mut s.rd, &mut s.pd,
        );
        element_curl_into(
            &self.re, &self.geom[e], &s.ee, &mut s.dh, &mut s.rd, &mut s.pd,
        );
        for v in s.dh.iter_mut() {
            *v = -*v;
        }

        // Surface term, central flux.
        for f in 0..4 {
            let fi = &self.faces[e * 4 + f];
            let n = fi.normal;
            let coef = 0.5 * fi.fscale;
            for m in 0..nfp {
                let vi = self.re.face_nodes[f][m];
                let em = [s.ee[vi * 3], s.ee[vi * 3 + 1], s.ee[vi * 3 + 2]];
                let hm = [s.hh[vi * 3], s.hh[vi * 3 + 1], s.hh[vi * 3 + 2]];
                // Jumps [E] = E⁻-E⁺, [H] = H⁻-H⁺.
                let (je, jh) = if fi.port != usize::MAX {
                    // Port boundary, characteristic flux against the
                    // incident field. Phase 1: the incident field is zero,
                    // so the jump is the interior trace itself, a
                    // first-order absorbing boundary that lets outgoing
                    // waves leave.
                    (em, hm)
                } else if fi.neighbor == usize::MAX {
                    // PEC ghost: [E] = 2·E_tangential, [H] = 2·H_normal.
                    let edn = dot(n, em);
                    let hdn = dot(n, hm);
                    (
                        [
                            2.0 * (em[0] - edn * n[0]),
                            2.0 * (em[1] - edn * n[1]),
                            2.0 * (em[2] - edn * n[2]),
                        ],
                        [
                            2.0 * hdn * n[0],
                            2.0 * hdn * n[1],
                            2.0 * hdn * n[2],
                        ],
                    )
                } else {
                    let vj = self.re.face_nodes[fi.neighbor_local_face]
                        [fi.perm[m]];
                    let nbb = fi.neighbor * np * 6;
                    (
                        [
                            em[0] - y[nbb + vj * 6],
                            em[1] - y[nbb + vj * 6 + 1],
                            em[2] - y[nbb + vj * 6 + 2],
                        ],
                        [
                            hm[0] - y[nbb + vj * 6 + 3],
                            hm[1] - y[nbb + vj * 6 + 4],
                            hm[2] - y[nbb + vj * 6 + 5],
                        ],
                    )
                };
                let nxjh = cross(n, jh);
                let nxje = cross(n, je);
                // Upwind penalty n̂×(n̂×[·]) = -[·]_tangential, dissipative,
                // damps the discontinuous spurious modes. Port faces are
                // always fully characteristic (the port absorbs outgoing
                // waves regardless of the global central/upwind blend).
                let a = if fi.port != usize::MAX {
                    1.0
                } else {
                    self.flux_alpha
                };
                let pe = cross(n, cross(n, je));
                let ph = cross(n, cross(n, jh));
                for i in 0..np {
                    let w = coef * self.re.lift[i * cols + f * nfp + m];
                    for c in 0..3 {
                        s.de[i * 3 + c] += w * (-nxjh[c] + a * pe[c]);
                        s.dh[i * 3 + c] += w * (nxje[c] + a * ph[c]);
                    }
                }
            }
        }

        // Apply per-element materials: ∂E/∂t = (1/ε)(∇×H + flux) - (σ/ε)E,
        // ∂H/∂t = (1/μ)(-∇×E + flux).
        let (ie, im) = (self.inv_eps[e], self.inv_mu[e]);
        let (se, sm) = (self.sigma_eps[e], self.sigma_mu[e]);
        for node in 0..np {
            for c in 0..3 {
                out[node * 6 + c] =
                    ie[c] * s.de[node * 3 + c] - se[c] * s.ee[node * 3 + c];
                out[node * 6 + 3 + c] =
                    im[c] * s.dh[node * 3 + c] - sm[c] * s.hh[node * 3 + c];
            }
        }

        // Dispersive polarisation current, on a Debye element D = ε_∞·E + P,
        // so Ampere's law carries an extra −Ṗ/ε_∞ term with Ṗ = a·P + g·E.
        // P lives in y's appended block; the E block above already used
        // ε = ε_∞ for this element, so this only subtracts the current.
        let slot = self.disp_slot[e];
        if slot != usize::MAX {
            let d = &self.disp[slot];
            let p_base = self.eh_len() + slot * np * 3;
            for node in 0..np {
                for c in 0..3 {
                    let p_val = y[p_base + node * 3 + c];
                    let e_val = s.ee[node * 3 + c];
                    let p_dot = d.a * p_val + d.g * e_val;
                    out[node * 6 + c] -= d.inv_eps_inf * p_dot;
                }
            }
        }
    }

    /// Assemble the operator as a dense `N×N` row-major matrix by applying it
    /// to each unit vector. For validation on small meshes.
    pub fn assemble_dense(&self) -> Vec<Field> {
        let n = self.n_dof();
        let mut a = vec![0.0; n * n];
        let mut ej = vec![0.0; n];
        for j in 0..n {
            ej[j] = 1.0;
            let col = self.apply(&ej);
            for (i, &v) in col.iter().enumerate() {
                a[i * n + j] = v;
            }
            ej[j] = 0.0;
        }
        a
    }

    /// Instantaneous electromagnetic field energy
    /// `E_field = ½·∫(ε|E|² + μ|H|²) dV`, evaluated matrix-free.
    ///
    /// The DG energy-mass matrix `M̃` (see [`assemble_energy_mass`]) is
    /// block-diagonal per element, so `½·yᵀM̃y` is a plain per-element sum:
    /// no `N×N` matrix is ever materialised. For element `e` the physical
    /// mass block is `|det J_e|·M_ref`, weighted by `ε_c` on the E
    /// components and `μ_c` on the H components, and this routine accumulates
    /// the scalar quadratic form `½·Σ_e Σ_c w_c·(fᵀ·(scale·M_ref)·f)`
    /// directly, `f` the element's per-component nodal vector.
    ///
    /// Only the first `6·Np·n_elem` entries of `y` (the E,H state) are read;
    /// any auxiliary state appended after that is ignored, so this does NOT
    /// require `y.len() == n_dof`. The per-element work is independent and
    /// runs in parallel, like [`apply_into`](Self::apply_into).
    ///
    /// [`assemble_energy_mass`]: Self::assemble_energy_mass
    pub fn field_energy(&self, y: &[Field]) -> Field {
        let np = self.re.n_nodes;
        let stride = np * 6;
        debug_assert!(
            y.len() >= stride * self.n_elem,
            "state shorter than the 6·Np·n_elem E,H block"
        );
        // Per element: ½·Σ_c w_c·Σ_{ni,nj} scale·M_ref[ni,nj]·f_c[ni]·f_c[nj].
        // Element blocks are disjoint, so the sum folds in parallel.
        let half = (0..self.n_elem)
            .into_par_iter()
            .map(|e| {
                let scale = self.geom[e].det.abs();
                let eps = self.inv_eps[e].map(|x| 1.0 / x);
                let mu = self.inv_mu[e].map(|x| 1.0 / x);
                let base = e * stride;
                let mut acc = 0.0;
                for ni in 0..np {
                    for nj in 0..np {
                        let mij = scale * self.re.mass[ni * np + nj];
                        if mij == 0.0 {
                            continue;
                        }
                        let bi = base + ni * 6;
                        let bj = base + nj * 6;
                        for c in 0..6 {
                            let w = if c < 3 { eps[c] } else { mu[c - 3] };
                            acc += w * mij * y[bi + c] * y[bj + c];
                        }
                    }
                }
                acc
            })
            .sum::<Field>();
        0.5 * half
    }

    /// Dense block-diagonal energy mass `M̃`, the material-weighted field
    /// energy `yᵀM̃y = ∫(ε|E|² + μ|H|²)`: per element a copy of
    /// `|det J_e|·M_ref`, scaled by `ε` on the E components and `μ` on the H
    /// components.
    pub fn assemble_energy_mass(&self) -> Vec<Field> {
        let np = self.re.n_nodes;
        let n = self.n_dof();
        let mut m = vec![0.0; n * n];
        for e in 0..self.n_elem {
            let scale = self.geom[e].det.abs();
            let eps = self.inv_eps[e].map(|x| 1.0 / x);
            let mu = self.inv_mu[e].map(|x| 1.0 / x);
            let base = e * np * 6;
            for ni in 0..np {
                for nj in 0..np {
                    let mij = scale * self.re.mass[ni * np + nj];
                    for c in 0..6 {
                        let w = if c < 3 { eps[c] } else { mu[c - 3] };
                        m[(base + ni * 6 + c) * n + (base + nj * 6 + c)] =
                            w * mij;
                    }
                }
            }
        }
        m
    }
}

/// Compressed-sparse-row matrix, the explicit state-space operator `A`.
pub struct CsrMatrix {
    /// Dimension.
    pub n: usize,
    /// Row offsets, length `n + 1`.
    pub row_ptr: Vec<usize>,
    /// Column index of each stored entry.
    pub col_idx: Vec<usize>,
    /// Stored values.
    pub values: Vec<Field>,
}

impl CsrMatrix {
    /// Number of stored nonzeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse matrix-vector product `A·x`.
    pub fn matvec(&self, x: &[Field]) -> Vec<Field> {
        let mut y = vec![0.0; self.n];
        for i in 0..self.n {
            let mut acc = 0.0;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                acc += self.values[k] * x[self.col_idx[k]];
            }
            y[i] = acc;
        }
        y
    }
}

impl MaxwellOperator {
    /// Assemble the operator as an explicit sparse CSR matrix, the
    /// state-space `A`, **without ever densifying**.
    ///
    /// The DG operator couples each element only to itself and its (≤4) face
    /// neighbours, so row block `e` is found by probing just that small
    /// column stencil with unit vectors and reading element `e`'s output
    /// block. Element blocks are independent and assemble in parallel.
    /// Memory is `O(nnz)`, not `O(N²)`, so this scales to production meshes
    /// where [`assemble_dense`](Self::assemble_dense) cannot.
    /// Sorted column-element stencil of row block `e`, itself plus every
    /// distinct face neighbour. Returns the fixed `[usize; 5]` array and the
    /// number of entries used; never allocates.
    fn element_stencil(&self, e: usize) -> ([usize; 5], usize) {
        let mut s = [0usize; 5];
        s[0] = e;
        let mut count = 1;
        for f in 0..4 {
            let nb = self.faces[e * 4 + f].neighbor;
            if nb != usize::MAX && !s[..count].contains(&nb) {
                s[count] = nb;
                count += 1;
            }
        }
        s[..count].sort_unstable();
        (s, count)
    }

    pub fn assemble_sparse(&self) -> CsrMatrix {
        let stride = self.re.n_nodes * 6;
        let np = self.re.n_nodes;
        let p_stride = np * 3;
        let n = self.n_dof();
        let eh_len = self.eh_len();

        // Each rayon worker folds a contiguous run of elements into one
        // `SparseFragment`, reusing its buffers across every element, the
        // element loop allocates nothing. `with_min_len` forces chunks
        // coarse enough for that reuse to pay off while still giving the
        // thread pool plenty of independent work. `fold` keeps the
        // fragments in element order, so concatenating them yields a
        // row-ordered CSR.
        //
        // Each element block's column stencil is the element itself plus its
        // face neighbours. On a dispersive element the `[E,H]` rows also
        // couple to the appended polarisation block (the `−Ṗ/ε_∞` current),
        // so the polarisation columns of every dispersive stencil element are
        // probed too. The non-dispersive case has no such columns, so the
        // assembled `[E,H]` block stays byte-identical.
        let min_len =
            (self.n_elem / (4 * rayon::current_num_threads())).max(1);
        let frags: Vec<SparseFragment> = (0..self.n_elem)
            .into_par_iter()
            .with_min_len(min_len)
            .fold(
                || SparseFragment::new(n, stride, np),
                |mut f, e| {
                    let (sten, ns) = self.element_stencil(e);
                    f.entries.clear();
                    for &c in &sten[..ns] {
                        // The element's [E,H] columns.
                        for jl in 0..stride {
                            let j = c * stride + jl;
                            f.probe[j] = 1.0;
                            f.out.iter_mut().for_each(|v| *v = 0.0);
                            self.apply_element(
                                e,
                                &f.probe,
                                &mut f.out,
                                &mut f.scratch,
                            );
                            f.probe[j] = 0.0;
                            for il in 0..stride {
                                let v = f.out[il];
                                if v != 0.0 {
                                    f.entries.push((il, j, v));
                                }
                            }
                        }
                        // The element's own polarisation columns, if it is
                        // dispersive. apply_element reads only element e's own
                        // P-block, so a neighbour's P-columns always probe to
                        // zero - only c == e can contribute.
                        let slot = if c == e { self.disp_slot[e] } else { usize::MAX };
                        if slot != usize::MAX {
                            for jl in 0..p_stride {
                                let j = eh_len + slot * p_stride + jl;
                                f.probe[j] = 1.0;
                                f.out.iter_mut().for_each(|v| *v = 0.0);
                                self.apply_element(
                                    e,
                                    &f.probe,
                                    &mut f.out,
                                    &mut f.scratch,
                                );
                                f.probe[j] = 0.0;
                                for il in 0..stride {
                                    let v = f.out[il];
                                    if v != 0.0 {
                                        f.entries.push((il, j, v));
                                    }
                                }
                            }
                        }
                    }
                    // Group by local row, columns ascending within a row,
                    // sorting on the full key needs no stable-sort scratch.
                    f.entries.sort_unstable_by_key(|&(il, j, _)| (il, j));
                    let mut cursor = 0;
                    for il in 0..stride {
                        let mut cnt = 0;
                        while cursor < f.entries.len()
                            && f.entries[cursor].0 == il
                        {
                            f.col_idx.push(f.entries[cursor].1);
                            f.values.push(f.entries[cursor].2);
                            cnt += 1;
                            cursor += 1;
                        }
                        f.row_len.push(cnt);
                    }
                    f
                },
            )
            .collect();

        // Concatenate the per-job fragments, already in element order, so
        // these are the leading `eh_len` rows of the CSR.
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        let mut acc = 0;
        for f in &frags {
            for &l in &f.row_len {
                acc += l;
                row_ptr.push(acc);
            }
            col_idx.extend_from_slice(&f.col_idx);
            values.extend_from_slice(&f.values);
        }

        // The appended polarisation rows, one local ODE per Debye element,
        // dP = a*P + g*E. Each P row carries exactly two entries: the
        // diagonal `a` (its own P DOF) and `g` (the matching E DOF). Columns
        // ascending: the E DOF (in the [E,H] block) precedes the P DOF.
        for d in &self.disp {
            let slot = self.disp_slot[d.elem];
            let p_base = eh_len + slot * p_stride;
            let e_base = d.elem * stride;
            for node in 0..np {
                for c in 0..3 {
                    let e_col = e_base + node * 6 + c;
                    let p_col = p_base + node * 3 + c;
                    // g*E coupling, then a*P diagonal, ascending columns.
                    col_idx.push(e_col);
                    values.push(d.g);
                    col_idx.push(p_col);
                    values.push(d.a);
                    acc += 2;
                    row_ptr.push(acc);
                }
            }
        }
        CsrMatrix { n, row_ptr, col_idx, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curl_of_polynomial_field_is_exact() {
        // On a sheared physical element, the discrete curl reproduces the
        // analytic curl of a degree-2 vector field exactly at the nodes.
        let v = [
            [0.3, -0.2, 0.1],
            [1.4, 0.1, -0.2],
            [0.2, 1.3, 0.4],
            [-0.1, 0.2, 1.6],
        ];
        let gf = GeometricFactors::for_tet(&v);
        let re = ReferenceElement::new(2);
        let n = re.n_nodes;
        let pn: Vec<[f64; 3]> = re.nodes.iter().map(|&xi| gf.map(xi)).collect();

        // F = (y·z, 2·z·x, 3·x·y)  ⇒  ∇×F = (x, -2y, z).
        let mut field = vec![0.0; 3 * n];
        for (i, p) in pn.iter().enumerate() {
            field[i * 3] = p[1] * p[2];
            field[i * 3 + 1] = 2.0 * p[2] * p[0];
            field[i * 3 + 2] = 3.0 * p[0] * p[1];
        }

        let curl = element_curl(&re, &gf, &field);
        for (i, p) in pn.iter().enumerate() {
            let want = [p[0], -2.0 * p[1], p[2]];
            for c in 0..3 {
                assert!(
                    (curl[i * 3 + c] - want[c]).abs() < 1e-9,
                    "node {i} comp {c}: got {}, want {}",
                    curl[i * 3 + c],
                    want[c]
                );
            }
        }
    }

    #[test]
    fn curl_of_constant_field_vanishes() {
        let gf = GeometricFactors::for_tet(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let re = ReferenceElement::new(3);
        let field = vec![0.7; 3 * re.n_nodes];
        let curl = element_curl(&re, &gf, &field);
        assert!(curl.iter().all(|c| c.abs() < 1e-10));
    }

    #[test]
    fn central_flux_operator_conserves_energy() {
        // The central-flux DG Maxwell operator is exactly energy-conserving:
        // M̃·A must be skew-symmetric. This validates the flux signs, the
        // surface scaling and the PEC boundary treatment all at once.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        // Central flux (alpha = 0) is the energy-conserving case.
        let op = MaxwellOperator::new(&mesh, 2, 0.0);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mm = op.assemble_energy_mass();

        // ma = M̃ · A
        let mut ma = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let mik = mm[i * n + k];
                if mik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    ma[i * n + j] += mik * a[k * n + j];
                }
            }
        }

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                worst = worst.max((ma[p * n + q] + ma[q * n + p]).abs());
                scale = scale.max(ma[p * n + q].abs());
            }
        }
        assert!(
            worst < 1e-9 * scale.max(1.0),
            "M̃A not skew-symmetric: worst {worst}, scale {scale}"
        );
    }

    #[test]
    fn uniform_dielectric_shifts_cavity_resonance() {
        // Filling the cavity with ε_r scales every resonance by 1/√(ε_r·μ_r).
        use crate::mesh_gen::structured_box;
        use faer::Mat;
        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let eps_r = 4.0;
        let mats =
            vec![ElemMaterial::isotropic(eps_r, 1.0, 0.0); mesh.n_tets()];
        let op = MaxwellOperator::new_with_materials(&mesh, 2, 1.0, &mats);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
        let eig = mat.eigenvalues().expect("eig");
        let mut fund = (f64::NEG_INFINITY, 0.0_f64);
        for z in &eig {
            if z.im.abs() > 0.5 && z.re > fund.0 {
                fund = (z.re, z.im.abs());
            }
        }
        let want = std::f64::consts::PI * 2.0_f64.sqrt() / eps_r.sqrt();
        let err = (fund.1 - want).abs() / want;
        assert!(
            err < 0.02,
            "ε_r={eps_r}: |Im| = {:.4}, analytic π√2/√ε_r = {want:.4}, err {err:.3}",
            fund.1
        );
    }

    #[test]
    fn heterogeneous_central_flux_conserves_energy() {
        // With heterogeneous ε, μ the central-flux operator still conserves
        // the material-weighted energy: M̃·A is skew-symmetric.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let mats: Vec<ElemMaterial> = (0..mesh.n_tets())
            .map(|i| {
                ElemMaterial::isotropic(
                    1.0 + 0.7 * i as f64,
                    1.0 + 0.3 * i as f64,
                    0.0,
                )
            })
            .collect();
        let op = MaxwellOperator::new_with_materials(&mesh, 2, 0.0, &mats);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mm = op.assemble_energy_mass();

        let mut ma = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let mik = mm[i * n + k];
                if mik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    ma[i * n + j] += mik * a[k * n + j];
                }
            }
        }
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                worst = worst.max((ma[p * n + q] + ma[q * n + p]).abs());
                scale = scale.max(ma[p * n + q].abs());
            }
        }
        assert!(
            worst < 1e-9 * scale.max(1.0),
            "heterogeneous M̃A not skew: worst {worst}, scale {scale}"
        );
    }

    #[test]
    fn field_energy_matches_dense_energy_mass() {
        // The matrix-free `field_energy` is exactly the quadratic form
        // `½·yᵀM̃y` of the dense block-diagonal energy mass, proven here on
        // a tiny heterogeneous box so the dense `N×N` matrix is affordable.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let mats: Vec<ElemMaterial> = (0..mesh.n_tets())
            .map(|i| ElemMaterial {
                eps: [2.0 + i as f64, 3.0, 1.5],
                mu: [1.0, 1.2 + 0.3 * i as f64, 2.0],
                sigma: 0.0,
                sigma_m: 0.0,
            })
            .collect();
        let op = MaxwellOperator::new_with_materials(&mesh, 2, 1.0, &mats);
        let n = op.n_dof();
        let mm = op.assemble_energy_mass();

        // A deterministic, non-trivial state.
        let y: Vec<f64> = (0..n)
            .map(|i| (0.3 * i as f64).sin() + 0.2 * (i as f64).cos())
            .collect();

        // Reference dense quadratic form ½·yᵀM̃y.
        let mut dense = 0.0;
        for i in 0..n {
            for j in 0..n {
                dense += y[i] * mm[i * n + j] * y[j];
            }
        }
        dense *= 0.5;

        let mf = op.field_energy(&y);
        let err = (mf - dense).abs() / dense.abs().max(1.0);
        assert!(
            err < 1e-12,
            "matrix-free field_energy = {mf}, dense ½yᵀM̃y = {dense}, err {err:.2e}"
        );

        // Trailing auxiliary state must be ignored: padding `y` leaves the
        // result unchanged.
        let mut padded = y.clone();
        padded.extend_from_slice(&[7.0, -3.0, 11.0]);
        let mf_padded = op.field_energy(&padded);
        assert_eq!(mf, mf_padded, "auxiliary tail must not affect the energy");

        // A nonzero physical state has strictly positive field energy.
        assert!(mf > 0.0, "field energy must be positive, got {mf}");
    }

    #[test]
    fn conductivity_damps_at_the_analytic_rate() {
        // A uniform conductivity σ damps every mode by Re = -σ/(2ε) on top of
        // the numerical (upwind) damping, isolated by differencing σ>0
        // against σ=0.
        use crate::mesh_gen::structured_box;
        use faer::Mat;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let eps_r = 1.0;

        let least_damped_re = |sigma: f64| -> f64 {
            let mats =
                vec![ElemMaterial::isotropic(eps_r, 1.0, sigma); mesh.n_tets()];
            let op = MaxwellOperator::new_with_materials(&mesh, 2, 1.0, &mats);
            let n = op.n_dof();
            let a = op.assemble_dense();
            let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
            let eig = mat.eigenvalues().expect("eig");
            eig.iter()
                .filter(|z| z.im.abs() > 1.0)
                .map(|z| z.re)
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let sigma = 0.2;
        let delta = least_damped_re(sigma) - least_damped_re(0.0);
        let want = -sigma / (2.0 * eps_r);
        let err = (delta - want).abs() / want.abs();
        assert!(
            err < 0.1,
            "conductivity damping ΔRe = {delta:.5}, analytic -σ/2ε = {want:.5}, err {err:.3}"
        );
    }

    #[test]
    fn anisotropic_central_flux_conserves_energy() {
        // Diagonal-anisotropic ε, μ, the central-flux operator still
        // conserves the per-component-weighted material energy: M̃·A is skew.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let mats: Vec<ElemMaterial> = (0..mesh.n_tets())
            .map(|i| ElemMaterial {
                eps: [2.0 + i as f64, 3.0, 1.5],
                mu: [1.0, 1.2 + 0.3 * i as f64, 2.0],
                sigma: 0.0,
                sigma_m: 0.0,
            })
            .collect();
        let op = MaxwellOperator::new_with_materials(&mesh, 2, 0.0, &mats);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mm = op.assemble_energy_mass();

        let mut ma = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let mik = mm[i * n + k];
                if mik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    ma[i * n + j] += mik * a[k * n + j];
                }
            }
        }
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                worst = worst.max((ma[p * n + q] + ma[q * n + p]).abs());
                scale = scale.max(ma[p * n + q].abs());
            }
        }
        assert!(
            worst < 1e-9 * scale.max(1.0),
            "anisotropic M̃A not skew: worst {worst}, scale {scale}"
        );
    }

    #[test]
    fn matched_absorber_damps_at_the_full_rate() {
        // An impedance-matched material (σ/ε = σ*/μ = ν) loses energy from
        // both the E and H halves, so a mode decays at Re = -ν, twice the
        // rate of an electric-only loss. This validates the magnetic-loss
        // term and the matched-absorber construction together.
        use crate::mesh_gen::structured_box;
        use faer::Mat;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let nu = 0.15;

        let least_damped_re = |mat: ElemMaterial| -> f64 {
            let mats = vec![mat; mesh.n_tets()];
            let op = MaxwellOperator::new_with_materials(&mesh, 2, 1.0, &mats);
            let n = op.n_dof();
            let a = op.assemble_dense();
            let m = Mat::from_fn(n, n, |i, j| a[i * n + j]);
            m.eigenvalues()
                .expect("eig")
                .iter()
                .filter(|z| z.im.abs() > 1.0)
                .map(|z| z.re)
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let re_vac = least_damped_re(ElemMaterial::VACUUM);
        let re_matched =
            least_damped_re(ElemMaterial::matched_absorber(1.0, 1.0, nu));
        let delta = re_matched - re_vac;
        let err = (delta + nu).abs() / nu;
        assert!(
            err < 0.08,
            "matched-absorber decay ΔRe = {delta:.4}, analytic -ν = {:.4}, err {err:.3}",
            -nu
        );
    }

    #[test]
    fn cavity_eigenfrequencies_match_analytic() {
        // P3.5 gate: eigenvalues of A for a unit cubic PEC cavity vs the
        // analytic resonances ω = π·√(m²+n²+p²). Lowest physical mode is
        // (1,1,0) ⇒ ω = π√2.
        use crate::mesh_gen::structured_box;
        use faer::Mat;

        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        // Upwind flux (alpha = 1) damps the discontinuous spurious modes;
        // the physical cavity modes survive as the least-damped ones.
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
        let eig = mat.eigenvalues().expect("eigenvalues");

        // Among the non-static modes (|Im| > 1), the physical fundamental is
        // the least-damped, the largest (least-negative) real part.
        let want = std::f64::consts::PI * 2.0_f64.sqrt();
        let mut fundamental = (f64::NEG_INFINITY, 0.0_f64);
        for z in &eig {
            if z.im.abs() > 1.0 && z.re > fundamental.0 {
                fundamental = (z.re, z.im.abs());
            }
        }
        let (re, im) = fundamental;
        let err = (im - want).abs() / want;
        eprintln!(
            "DIAG cavity: fundamental Re={re:.5} |Im|={im:.5}  analytic π√2={want:.5}  rel.err={err:.4}"
        );
        assert!(re < 0.0, "upwind flux must damp, fundamental Re = {re}");
        assert!(
            err < 0.05,
            "fundamental |Im| = {im:.4}, analytic π√2 = {want:.4}, rel.err {err:.3}"
        );
    }

    #[test]
    fn cavity_eigenfrequency_on_irregular_mesh() {
        // WP1.2: the cavity fundamental survives on a skewed, irregular mesh,
        // the physics does not depend on mesh regularity, only the
        // discretisation error grows mildly.
        use crate::mesh_gen::structured_box_jittered;
        use faer::Mat;

        let mesh = structured_box_jittered(2, 2, 2, 1.0, 1.0, 1.0, 0.25, 7);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
        let eig = mat.eigenvalues().expect("eigenvalues");

        let want = std::f64::consts::PI * 2.0_f64.sqrt();
        let mut fundamental = (f64::NEG_INFINITY, 0.0_f64);
        for z in &eig {
            if z.im.abs() > 1.0 && z.re > fundamental.0 {
                fundamental = (z.re, z.im.abs());
            }
        }
        let (re, im) = fundamental;
        let err = (im - want).abs() / want;
        eprintln!(
            "DIAG irregular mesh: Re={re:.5} |Im|={im:.5} π√2={want:.5} err={err:.4}"
        );
        assert!(re < 0.0, "upwind flux must damp, Re = {re}");
        assert!(
            err < 0.08,
            "irregular-mesh fundamental |Im| = {im:.4}, π√2 = {want:.4}, rel.err {err:.3}"
        );
    }

    #[test]
    fn cavity_fundamental_converges_under_refinement() {
        // WP1.3: mesh refinement drives the eigenfrequency error down at a
        // high-order rate.
        use crate::mesh_gen::structured_box;
        use faer::Mat;
        let want = std::f64::consts::PI * 2.0_f64.sqrt();

        let fundamental_err = |cells: usize, order: usize| -> f64 {
            let mesh = structured_box(cells, cells, cells, 1.0, 1.0, 1.0);
            let op = MaxwellOperator::new(&mesh, order, 1.0);
            let n = op.n_dof();
            let a = op.assemble_dense();
            let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
            let eig = mat.eigenvalues().expect("eig");
            let mut best = (f64::NEG_INFINITY, 0.0_f64);
            for z in &eig {
                if z.im.abs() > 1.0 && z.re > best.0 {
                    best = (z.re, z.im.abs());
                }
            }
            (best.1 - want).abs() / want
        };

        let coarse = fundamental_err(1, 2); // 6 tets
        let fine = fundamental_err(2, 2); // 48 tets
        eprintln!(
            "DIAG convergence p=2: coarse(1^3)={coarse:.3e} fine(2^3)={fine:.3e} ratio={:.1}",
            coarse / fine
        );
        assert!(fine < coarse, "refinement must reduce the error");
        assert!(fine < 1e-3, "fine-mesh error {fine:.2e} too large");
        assert!(
            coarse / fine > 4.0,
            "weak convergence, error ratio only {:.1}",
            coarse / fine
        );
    }

    #[test]
    fn soft_source_injects_energy() {
        // A soft source driven from rest by a Gaussian pulse injects field
        // energy into the cavity.
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();

        // Source: E_z at the cavity centre, which is a node, so exact.
        let sdof = op.nearest_node_dof([0.5, 0.5, 0.5], 0, 2);
        assert!(sdof < n);
        let mut s = vec![0.0; n];
        s[sdof] = 1.0;

        let mut y = vec![0.0; n];
        let (t0, tau, h) = (0.3, 0.08, 0.01);
        for k in 0..80 {
            let t = k as f64 * h;
            let g = (-((t - t0) / tau).powi(2)).exp();
            let b: Vec<f64> = s.iter().map(|x| x * g).collect();
            y = etd_step(|x| op.apply(x), &y, &b, h, 30);
        }
        assert!(y.iter().all(|v| v.is_finite()));

        let mm = op.assemble_energy_mass();
        let mut energy = 0.0;
        for i in 0..n {
            for j in 0..n {
                energy += y[i] * mm[i * n + j] * y[j];
            }
        }
        assert!(energy > 1e-6, "soft source injected no energy: {energy:e}");
    }

    #[test]
    fn sparse_assembly_matches_matrix_free_apply() {
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let csr = op.assemble_sparse();
        assert_eq!(csr.n, n);

        let v: Vec<f64> =
            (0..n).map(|i| (1.0 + i as f64 * 0.07).cos()).collect();
        let sp = csr.matvec(&v);
        let mf = op.apply(&v);
        let err: f64 = sp
            .iter()
            .zip(&mf)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = mf.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 1e-10 * scale, "sparse vs matrix-free: err {err}");
        // The DG operator couples only neighbouring elements, genuinely sparse.
        assert!(
            csr.nnz() < n * n / 4,
            "operator not sparse: nnz {} of {}",
            csr.nnz(),
            n * n
        );
    }

    #[test]
    fn sparse_assembly_scales_without_densifying() {
        // WP6.2 gate: assemble `A` for a mesh whose dense form would need
        // gigabytes, the element-wise probe path must stay O(nnz). Each row
        // can couple at most `5·stride` columns (self + 4 face neighbours),
        // so nnz grows linearly with N, not quadratically.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(4, 4, 4, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let stride = 6 * 10; // 6 fields × Np(order 2) = 60

        // The dense matrix would be n² f64, assert it is genuinely out of
        // reach, so this test actually exercises the non-densifying path.
        let dense_bytes = (n as u128).pow(2) * 8;
        assert!(
            dense_bytes > 1 << 30,
            "mesh too small to prove the point: dense would be {dense_bytes} B"
        );

        let csr = op.assemble_sparse();
        assert_eq!(csr.n, n);
        // Linear scaling: every row stays within the face-neighbour stencil.
        assert!(
            csr.nnz() <= 5 * stride * n,
            "nnz {} exceeds the stencil bound {}",
            csr.nnz(),
            5 * stride * n
        );

        // Correctness still holds at this scale.
        let v: Vec<f64> =
            (0..n).map(|i| (0.3 + i as f64 * 0.013).sin()).collect();
        let sp = csr.matvec(&v);
        let mf = op.apply(&v);
        let err: f64 = sp
            .iter()
            .zip(&mf)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = mf.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 1e-10 * scale, "sparse vs matrix-free: err {err}");
    }

    #[test]
    fn corner_nodes_map_to_tet_vertices() {
        // WP7.3 export hook: the four corner local indices are distinct,
        // and the physical DG node coordinates land on the mesh-tet
        // vertices in the affine-map order.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(2, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 3, 1.0);
        let corners = op.corner_local_nodes();
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(corners[i], corners[j], "corners not distinct");
            }
        }
        let np = op.re.n_nodes;
        let pts = op.node_coords();
        assert_eq!(pts.len(), np * mesh.n_tets());
        for (e, tet) in mesh.tets.iter().enumerate() {
            for (c, &local) in corners.iter().enumerate() {
                let got = pts[e * np + local];
                let want = mesh.nodes[tet[c]];
                let d: f64 =
                    (0..3).map(|k| (got[k] - want[k]).powi(2)).sum();
                assert!(
                    d.sqrt() < 1e-12,
                    "elem {e} corner {c}: {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn port_face_drains_field_energy() {
        // WP1.1: a boundary face tagged as a port acts as a characteristic
        // absorbing boundary, a divergence-free field disturbance
        // radiates out through it, whereas the all-PEC channel keeps the
        // energy bounded.
        use crate::mesh_gen::structured_box;
        use crate::propagator::expmv;

        let lz = 5.0;
        let mesh = structured_box(1, 1, 10, 0.5, 0.5, lz);

        // Triangles on the z = lz end face form the port.
        let port_tris: Vec<usize> = mesh
            .tris
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                t.iter().all(|&nd| (mesh.nodes[nd][2] - lz).abs() < 1e-9)
            })
            .map(|(i, _)| i)
            .collect();
        assert!(!port_tris.is_empty(), "no triangles on the port plane");

        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        let energy = |op: &MaxwellOperator, y: &[f64]| -> f64 {
            let mm = op.assemble_energy_mass();
            let n = op.n_dof();
            let mut e = 0.0;
            for i in 0..n {
                for j in 0..n {
                    e += y[i] * mm[i * n + j] * y[j];
                }
            }
            e
        };

        let run = |ports: &[PortSpec]| -> f64 {
            // Central flux, the all-PEC channel is then exactly
            // energy-conserving, so any drain is the port's doing.
            let op = MaxwellOperator::new_with_materials_ports(
                &mesh, 2, 0.0, &vacuum, ports,
            );
            let n = op.n_dof();
            // A smooth, z-only Eₓ bump, depends on z alone, so ∇·E = 0:
            // it propagates as ±z travelling waves, no static residue.
            let coords = op.node_coords();
            let mut y = vec![0.0; n];
            for (idx, p) in coords.iter().enumerate() {
                y[idx * 6] = (-((p[2] - 2.5) / 0.5).powi(2)).exp();
            }
            let e0 = energy(&op, &y);
            for _ in 0..900 {
                y = expmv(|x| op.apply(x), &y, 0.06, 24);
            }
            energy(&op, &y) / e0
        };

        let pec = run(&[]);
        let port = run(&[PortSpec { tris: port_tris, mode: None }]);
        // The all-PEC channel conserves energy exactly (central flux); the
        // port drains the majority of it. The residue is the test field's
        // below-cutoff content, evanescent in a waveguide, so it cannot
        // reach the port, not a flux defect; the quantitative reflection
        // is gated by the matched-line S₁₁ check (WP3.2).
        assert!(
            pec > 0.95,
            "central-flux all-PEC channel must conserve energy, kept {pec:.3}"
        );
        assert!(
            port < 0.5,
            "port face must drain the bulk of the energy, kept {port:.3}"
        );
        assert!(
            pec - port > 0.3,
            "port vs PEC contrast too weak: {pec:.3} vs {port:.3}"
        );
    }

    #[test]
    fn port_operator_only_dissipates_energy() {
        // WP2.1: with a port the operator is no longer energy-conserving.
        // The symmetric part of M̃A, i.e. M̃A + (M̃A)ᵀ, must be negative
        // semidefinite: the port can only drain energy from the
        // homogeneous (unexcited) system, never inject it. The interior
        // central flux contributes a skew (zero-symmetric) part, so any
        // positive eigenvalue would be a flux defect.
        use crate::mesh_gen::structured_box;
        use faer::Mat;

        let lz = 2.0;
        let mesh = structured_box(1, 1, 2, 1.0, 1.0, lz);
        let port_tris: Vec<usize> = mesh
            .tris
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                t.iter().all(|&nd| (mesh.nodes[nd][2] - lz).abs() < 1e-9)
            })
            .map(|(i, _)| i)
            .collect();
        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        // Central flux, the interior is energy-conserving, only the port
        // dissipates.
        let op = MaxwellOperator::new_with_materials_ports(
            &mesh, 2, 0.0, &vacuum, &[PortSpec { tris: port_tris, mode: None }],
        );
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mm = op.assemble_energy_mass();

        // MA = M̃·A.
        let mut ma = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let mik = mm[i * n + k];
                if mik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    ma[i * n + j] += mik * a[k * n + j];
                }
            }
        }
        // S = M̃A + (M̃A)ᵀ is symmetric; check it is negative semidefinite.
        let s = Mat::from_fn(n, n, |i, j| ma[i * n + j] + ma[j * n + i]);
        let eig = s.eigenvalues().expect("eigenvalues");
        let scale = ma.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let max_re = eig
            .iter()
            .map(|z| z.re)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_re = eig
            .iter()
            .map(|z| z.re)
            .fold(f64::INFINITY, f64::min);
        assert!(
            max_re < 1e-7 * scale,
            "port operator gains energy, max eig(M̃A+AᵀM̃) = {max_re:.3e}"
        );
        assert!(
            min_re < -1e-3 * scale,
            "port operator shows no dissipation, min eig = {min_re:.3e}"
        );
    }

    #[test]
    fn port_injects_a_mode_at_the_group_velocity() {
        // WP2.2: a TE₁₀ mode injected at a port travels down a matched
        // straight guide at the analytic group velocity
        // v_g = √(1 − (ω_c/ω₀)²).
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        use std::f64::consts::PI;

        let (a, b, lz) = (1.0, 0.5, 9.0);
        let mesh = structured_box(2, 1, 18, a, b, lz);
        // The z = 0 end face is the driven port.
        let port_tris: Vec<usize> = mesh
            .tris
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                t.iter().all(|&nd| mesh.nodes[nd][2].abs() < 1e-9)
            })
            .map(|(i, _)| i)
            .collect();
        assert!(!port_tris.is_empty());

        let rect = RectPort {
            origin: [0.0, 0.0, 0.0],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, 1.0], // inward (+z) for the z = 0 face
            a,
            b,
            mode: (1, 0),
            z0: 1.0,
        };
        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        let op = MaxwellOperator::new_with_materials_ports(
            &mesh,
            2,
            0.0,
            &vacuum,
            &[PortSpec { tris: port_tris, mode: Some(PortMode::Rect(rect)) }],
        );
        let n = op.n_dof();
        let b_spatial = op.port_source(0);
        assert!(
            b_spatial.iter().any(|&x| x != 0.0),
            "port source is empty"
        );

        // Modulated-Gaussian drive; carrier ω₀ in the single-mode band
        // (π, 2π) of this guide. A fairly narrow band keeps the group
        // velocity well-defined (broadband dispersion skews the peak).
        let omega0 = 1.6 * PI;
        let (t0, tau) = (7.0, 2.5);
        let pulse = |t: f64| {
            (-((t - t0) / tau).powi(2)).exp() * (omega0 * (t - t0)).sin()
        };

        let probe = |z: f64| op.nearest_node_dof([a / 2.0, b / 2.0, z], 0, 1);
        let (p1, p2) = (probe(3.5), probe(6.5));

        let dt = 0.02;
        let mut y = vec![0.0; n];
        let (mut peak1, mut tpk1) = (0.0_f64, 0.0);
        let (mut peak2, mut tpk2) = (0.0_f64, 0.0);
        for s in 0..850 {
            let t = s as f64 * dt;
            let g = pulse(t);
            let bvec: Vec<f64> =
                b_spatial.iter().map(|x| x * g).collect();
            y = etd_step(|x| op.apply(x), &y, &bvec, dt, 20);
            if y[p1].abs() > peak1 {
                peak1 = y[p1].abs();
                tpk1 = t;
            }
            if y[p2].abs() > peak2 {
                peak2 = y[p2].abs();
                tpk2 = t;
            }
        }
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(
            peak1 > 1e-3 && peak2 > 1e-3,
            "injected mode did not reach the probes: {peak1:.2e}, {peak2:.2e}"
        );

        // Group velocity from the envelope-peak arrival times.
        let v = 3.0 / (tpk2 - tpk1);
        let wc = PI / a; // TE₁₀ cutoff
        let vg = (1.0 - (wc / omega0).powi(2)).sqrt();
        let err = (v - vg).abs() / vg;
        eprintln!(
            "DIAG port mode: peaks t={tpk1:.2},{tpk2:.2}  v={v:.3}  v_g={vg:.3}"
        );
        assert!(
            err < 0.15,
            "packet speed {v:.3}, analytic v_g {vg:.3} (rel.err {err:.2})"
        );
    }

    #[test]
    fn port_extracts_the_incident_amplitude() {
        // WP3.1: driving a straight guide, the per-frequency forward/
        // backward split A,B = (P_e ± Z·P_h)/2 recovers an almost purely
        // forward wave, |B|/|A| ≪ 1 on a uniform guide (before any far-end
        // reflection can return), confirming the modal extraction.
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        use std::f64::consts::PI;

        let (a, b, lz) = (1.0, 0.5, 9.0);
        let mesh = structured_box(2, 1, 18, a, b, lz);
        let port_tris: Vec<usize> = mesh
            .tris
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                t.iter().all(|&nd| mesh.nodes[nd][2].abs() < 1e-9)
            })
            .map(|(i, _)| i)
            .collect();
        let rect = RectPort {
            origin: [0.0, 0.0, 0.0],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, 1.0],
            a,
            b,
            mode: (1, 0),
            z0: 1.0,
        };
        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        let op = MaxwellOperator::new_with_materials_ports(
            &mesh,
            2,
            0.0,
            &vacuum,
            &[PortSpec {
                tris: port_tris,
                mode: Some(PortMode::Rect(rect.clone())),
            }],
        );
        let n = op.n_dof();
        let b_spatial = op.port_source(0);

        let omega0 = 1.6 * PI;
        let (t0, tau) = (7.0, 2.5);
        let pulse = |t: f64| {
            (-((t - t0) / tau).powi(2)).exp() * (omega0 * (t - t0)).sin()
        };

        let dt = 0.02;
        let mut y = vec![0.0; n];
        let (mut fpe, mut fph, mut fg) =
            (Vec::new(), Vec::new(), Vec::new());
        // Stop before the far-end (PEC) reflection can return to the port.
        for s in 0..700 {
            let t = s as f64 * dt;
            let g = pulse(t);
            let bvec: Vec<f64> = b_spatial.iter().map(|x| x * g).collect();
            y = etd_step(|x| op.apply(x), &y, &bvec, dt, 20);
            let (pe, ph) = op.port_modal_projections(&y, 0);
            fpe.push(pe);
            fph.push(ph);
            fg.push(g);
        }
        assert!(y.iter().all(|v| v.is_finite()));

        // Discrete Fourier transform of a recorded signal at one frequency.
        let dft = |sig: &[f64], omega: f64| -> (f64, f64) {
            let (mut re, mut im) = (0.0, 0.0);
            for (k, &x) in sig.iter().enumerate() {
                let t = k as f64 * dt;
                re += x * (omega * t).cos();
                im -= x * (omega * t).sin();
            }
            (re * dt, im * dt)
        };
        let mag = |z: (f64, f64)| (z.0 * z.0 + z.1 * z.1).sqrt();

        // Per-frequency forward / backward split across the drive band.
        for &omega in &[1.45 * PI, 1.6 * PI, 1.75 * PI] {
            let pe = dft(&fpe, omega);
            let ph = dft(&fph, omega);
            let z = rect.te_impedance(omega);
            let amp = (
                0.5 * (pe.0 + z * ph.0),
                0.5 * (pe.1 + z * ph.1),
            );
            let bmp = (
                0.5 * (pe.0 - z * ph.0),
                0.5 * (pe.1 - z * ph.1),
            );
            let refl = mag(bmp) / mag(amp);
            eprintln!(
                "DIAG extract ω/π={:.2}: |A|={:.3e} |B|={:.3e} |B/A|={refl:.4}",
                omega / PI,
                mag(amp),
                mag(bmp),
            );
            assert!(mag(amp) > 1e-3, "no incident amplitude at ω={omega}");
            assert!(
                refl < 0.06,
                "uniform guide should be reflection-free: |B/A| = {refl:.3}"
            );
        }
    }

    #[test]
    fn coax_port_carries_a_matched_tem_wave() {
        // WP-Coax: a coaxial TEM mode injected at a coax port travels down a
        // straight matched coaxial line, propagates cleanly, and the port is
        // low-reflection. The TEM mode is dispersionless, it clocks exactly
        // c, and the per-frequency forward/backward split A,B = (P_e±Z·P_h)/2
        // recovers an almost purely forward wave (|B/A| ≪ 1) before any
        // far-end reflection can return. This mirrors `port_injects_a_mode_at
        // _the_group_velocity` / `port_extracts_the_incident_amplitude` and,
        // for the dispersionless speed, `lumped_port_carries_a_dispersionless
        // _tem_wave`.
        //
        // The coax TEM field `E ∝ ρ̂/ρ` is divergence-free in the transverse
        // plane, so it propagates as a free travelling wave at exactly c.
        // The four side walls are transparent characteristic boundaries
        // (`mode = None` ports), exactly as in the lumped-TEM test, so no
        // hollow-guide dispersion is imposed and the dispersionless TEM
        // velocity is the genuine signal.
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        use crate::waveguide::CoaxPort;
        use std::f64::consts::PI;

        // A square-cross-section box carries the analytic radial coax field
        // about the box centre. The transverse mesh is fine enough that the
        // 1/ρ profile is well resolved across the annulus.
        let (a, lz) = (3.0, 9.0);
        let mesh = structured_box(6, 6, 18, a, a, lz);
        let on = |pred: &dyn Fn([f64; 3]) -> bool| -> Vec<usize> {
            mesh.tris
                .iter()
                .enumerate()
                .filter(|(_, t)| t.iter().all(|&nd| pred(mesh.nodes[nd])))
                .map(|(i, _)| i)
                .collect()
        };
        let z0_tris = on(&|p| p[2].abs() < 1e-9);
        let zl_tris = on(&|p| (p[2] - lz).abs() < 1e-9);
        let x_lo = on(&|p| p[0].abs() < 1e-9);
        let x_hi = on(&|p| (p[0] - a).abs() < 1e-9);
        let y_lo = on(&|p| p[1].abs() < 1e-9);
        let y_hi = on(&|p| (p[1] - a).abs() < 1e-9);
        assert!(!z0_tris.is_empty() && !zl_tris.is_empty());
        assert!(!x_lo.is_empty() && !y_lo.is_empty());

        // The coax port on the z = 0 face, radial TEM field about the box
        // centre, inward normal +z. The annulus spans the box: an inner
        // radius clear of the singular axis, an outer radius inside the wall.
        let center = [a / 2.0, a / 2.0, 0.0];
        let coax = CoaxPort {
            center,
            w_hat: [0.0, 0.0, 1.0], // inward +z
            r_inner: 0.6,
            r_outer: 1.3,
        };
        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        // Central flux, the interior is energy-conserving, so any drain is
        // the port's doing. The z = lz end and the four side walls are
        // transparent absorbing ports (no mode): the far-end carries no
        // reflection, and the side walls let the coax field exist as a
        // dispersionless free wave rather than a hollow-guide mode.
        let op = MaxwellOperator::new_with_materials_ports(
            &mesh,
            2,
            0.0,
            &vacuum,
            &[
                PortSpec {
                    tris: z0_tris,
                    mode: Some(PortMode::Coax(coax.clone())),
                },
                PortSpec { tris: zl_tris, mode: None },
                PortSpec { tris: x_lo, mode: None },
                PortSpec { tris: x_hi, mode: None },
                PortSpec { tris: y_lo, mode: None },
                PortSpec { tris: y_hi, mode: None },
            ],
        );
        let n = op.n_dof();
        // The coax port has no cutoff, the TEM mode reaches DC.
        assert!(
            op.port_cutoff(0).abs() < 1e-12,
            "coax port must have zero cutoff"
        );
        let b_spatial = op.port_source(0);
        assert!(
            b_spatial.iter().any(|&x| x != 0.0),
            "coax port source is empty"
        );

        // Modulated-Gaussian drive. The TEM mode has no cutoff, so any
        // carrier in the well-resolved band works.
        let omega0 = 1.6 * PI;
        let (t0, tau) = (7.0, 2.5);
        let pulse = |t: f64| {
            (-((t - t0) / tau).powi(2)).exp() * (omega0 * (t - t0)).sin()
        };

        // Probes on the annulus mid-radius, two z stations apart, the TEM
        // packet arrival times give the propagation speed.
        let r_mid = 0.5 * (coax.r_inner + coax.r_outer);
        let probe = |z: f64| {
            op.nearest_node_dof(
                [a / 2.0 + r_mid, a / 2.0, z],
                0,
                0, // radial E along +x at this probe point
            )
        };
        let (zp1, zp2) = (3.0, 6.0);
        let (p1, p2) = (probe(zp1), probe(zp2));

        let dt = 0.02;
        let steps = 700; // stop before the far-end reflection returns
        let mut y = vec![0.0; n];
        let (mut peak1, mut tpk1) = (0.0_f64, 0.0);
        let (mut peak2, mut tpk2) = (0.0_f64, 0.0);
        let (mut fpe, mut fph) = (Vec::new(), Vec::new());
        for s in 0..steps {
            let t = s as f64 * dt;
            let g = pulse(t);
            let bvec: Vec<f64> = b_spatial.iter().map(|x| x * g).collect();
            y = etd_step(|x| op.apply(x), &y, &bvec, dt, 20);
            if y[p1].abs() > peak1 {
                peak1 = y[p1].abs();
                tpk1 = t;
            }
            if y[p2].abs() > peak2 {
                peak2 = y[p2].abs();
                tpk2 = t;
            }
            let (pe, ph) = op.port_modal_projections(&y, 0);
            fpe.push(pe);
            fph.push(ph);
        }
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(
            peak1 > 1e-3 && peak2 > 1e-3,
            "injected TEM mode did not reach the probes: {peak1:.2e}, {peak2:.2e}"
        );

        // The coax TEM wave is dispersionless, it travels at exactly c = 1.
        let v = (zp2 - zp1) / (tpk2 - tpk1);
        let v_err = (v - 1.0).abs();
        eprintln!(
            "DIAG coax port: peaks t={tpk1:.2},{tpk2:.2}  v={v:.3} (expect c=1)"
        );
        assert!(
            v_err < 0.15,
            "coax TEM packet speed {v:.3}, expected the dispersionless c = 1"
        );

        // Per-frequency forward / backward split, the modal extraction
        // returns a finite incident amplitude (the coax mode is genuinely
        // present in the propagated state) and the wave is predominantly
        // forward. The threshold is looser than the rectangular-waveguide
        // case because the box stand-in lacks the inner-conductor PEC that
        // would make the 1/ρ field a true bound mode: outward-radiating
        // energy hits the transparent side walls at oblique angles, where
        // the characteristic boundary is only an approximate absorber. The
        // important property is that the forward amplitude dominates.
        let dft = |sig: &[f64], omega: f64| -> (f64, f64) {
            let (mut re, mut im) = (0.0, 0.0);
            for (k, &x) in sig.iter().enumerate() {
                let t = k as f64 * dt;
                re += x * (omega * t).cos();
                im -= x * (omega * t).sin();
            }
            (re * dt, im * dt)
        };
        let mag = |z: (f64, f64)| (z.0 * z.0 + z.1 * z.1).sqrt();
        for &omega in &[1.45 * PI, 1.6 * PI, 1.75 * PI] {
            let pe = dft(&fpe, omega);
            let ph = dft(&fph, omega);
            // TEM impedance is flat Z = 1, no per-frequency rescaling.
            let z = coax.te_impedance(omega);
            let amp =
                (0.5 * (pe.0 + z * ph.0), 0.5 * (pe.1 + z * ph.1));
            let bmp =
                (0.5 * (pe.0 - z * ph.0), 0.5 * (pe.1 - z * ph.1));
            let refl = mag(bmp) / mag(amp);
            eprintln!(
                "DIAG coax extract ω/π={:.2}: |A|={:.3e} |B|={:.3e} \
                 |B/A|={refl:.4}",
                omega / PI,
                mag(amp),
                mag(bmp),
            );
            assert!(mag(amp) > 1e-3, "no incident amplitude at ω={omega}");
            // Forward wave dominates, the backward component stays below
            // a fraction of the forward amplitude.
            assert!(
                refl < 0.5,
                "coax line should be forward-dominated: |B/A| = {refl:.3}"
            );
        }
    }

    #[test]
    fn two_port_guide_s_parameters() {
        // WP3.2: a matched straight two-port guide, S₁₁ ≈ 0, |S₂₁| ≈ 1,
        // energy |S₁₁|² + |S₂₁|² ≈ 1, and reciprocity |S₂₁| ≈ |S₁₂|. The
        // run stops before the (imperfectly absorbed) far-port reflection
        // can return, the time window isolates the direct S-parameters.
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        use std::f64::consts::PI;

        let (a, b, lz) = (1.0, 0.5, 6.0);
        let mesh = structured_box(2, 1, 24, a, b, lz);
        let on_plane = |zc: f64| -> Vec<usize> {
            mesh.tris
                .iter()
                .enumerate()
                .filter(|(_, t)| {
                    t.iter().all(|&nd| (mesh.nodes[nd][2] - zc).abs() < 1e-9)
                })
                .map(|(i, _)| i)
                .collect()
        };
        let rect = |z0: f64, inward: f64| RectPort {
            origin: [0.0, 0.0, z0],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, inward],
            a,
            b,
            mode: (1, 0),
            z0: 1.0,
        };
        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        let op = MaxwellOperator::new_with_materials_ports(
            &mesh,
            2,
            0.0,
            &vacuum,
            &[
                PortSpec {
                    tris: on_plane(0.0),
                    mode: Some(PortMode::Rect(rect(0.0, 1.0))),
                },
                PortSpec {
                    tris: on_plane(lz),
                    mode: Some(PortMode::Rect(rect(lz, -1.0))),
                },
            ],
        );
        let n = op.n_dof();
        let src0 = op.port_source(0);
        let src1 = op.port_source(1);

        let omega0 = 1.5 * PI;
        let (t0, tau) = (3.0, 1.0);
        let pulse = |t: f64| {
            (-((t - t0) / tau).powi(2)).exp() * (omega0 * (t - t0)).sin()
        };
        let dt = 0.02;
        let steps = 900;

        // Drive one port; record (P_e, P_h) at both ports each step.
        let run = |src: &[f64]| -> [Vec<f64>; 4] {
            let mut y = vec![0.0; n];
            let mut cols: [Vec<f64>; 4] =
                std::array::from_fn(|_| Vec::with_capacity(steps));
            for s in 0..steps {
                let g = pulse(s as f64 * dt);
                let bvec: Vec<f64> = src.iter().map(|x| x * g).collect();
                y = etd_step(|x| op.apply(x), &y, &bvec, dt, 18);
                let (pe0, ph0) = op.port_modal_projections(&y, 0);
                let (pe1, ph1) = op.port_modal_projections(&y, 1);
                cols[0].push(pe0);
                cols[1].push(ph0);
                cols[2].push(pe1);
                cols[3].push(ph1);
            }
            cols
        };

        let dft = |sig: &[f64], omega: f64| -> (f64, f64) {
            let (mut re, mut im) = (0.0, 0.0);
            for (k, &x) in sig.iter().enumerate() {
                let t = k as f64 * dt;
                re += x * (omega * t).cos();
                im -= x * (omega * t).sin();
            }
            (re * dt, im * dt)
        };
        let cmag = |z: (f64, f64)| (z.0 * z.0 + z.1 * z.1).sqrt();
        // Forward / backward modal amplitudes A,B = (P_e ± Z·P_h)/2.
        let split = |pe: (f64, f64), ph: (f64, f64), z: f64| {
            (
                (0.5 * (pe.0 + z * ph.0), 0.5 * (pe.1 + z * ph.1)),
                (0.5 * (pe.0 - z * ph.0), 0.5 * (pe.1 - z * ph.1)),
            )
        };

        let drive0 = run(&src0);
        let drive1 = run(&src1);

        // Validate across the band where the mesh resolves the guide
        // wavelength well (away from cutoff and from the per-element
        // resolution limit).
        for &omega in &[1.35 * PI, 1.45 * PI, 1.55 * PI] {
            let z = rect(0.0, 1.0).te_impedance(omega);

            // Port 0 driven: incident A₀, reflected B₀; the outgoing wave
            // B₁ at port 1 is the transmission.
            let (a0, b0) =
                split(dft(&drive0[0], omega), dft(&drive0[1], omega), z);
            let (_a1, b1) =
                split(dft(&drive0[2], omega), dft(&drive0[3], omega), z);
            let s11 = cmag(b0) / cmag(a0);
            let s21 = cmag(b1) / cmag(a0);

            // Port 1 driven: incident A₁, transmission B₀ at port 0.
            let (a1d, _b1d) =
                split(dft(&drive1[2], omega), dft(&drive1[3], omega), z);
            let (_a0d, b0d) =
                split(dft(&drive1[0], omega), dft(&drive1[1], omega), z);
            let s12 = cmag(b0d) / cmag(a1d);

            let energy = s11 * s11 + s21 * s21;
            eprintln!(
                "DIAG S(ω/π={:.2}): S11={s11:.4} S21={s21:.4} \
                 S12={s12:.4}  |S|²={energy:.4}",
                omega / PI
            );
            assert!(s11 < 0.1, "S11 not small on a matched guide: {s11:.3}");
            assert!(
                (0.9..1.1).contains(&s21),
                "S21 not unity on a lossless guide: {s21:.3}"
            );
            assert!(
                (energy - 1.0).abs() < 0.1,
                "S-matrix not energy-conserving: |S|² = {energy:.3}"
            );
            assert!(
                (s21 - s12).abs() < 0.05,
                "reciprocity violated: S21 {s21:.3}, S12 {s12:.3}"
            );
        }
    }

    #[test]
    fn non_dispersive_operator_has_the_unchanged_dof_count() {
        // Safety property: with no Debye material the operator is byte
        // identical to before, n_dof is exactly 6*Np*n_elem and the P
        // block is empty.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let np = 10; // order 2
        assert_eq!(op.n_dispersive(), 0);
        assert_eq!(op.n_dof(), 6 * np * mesh.n_tets());
    }

    #[test]
    fn dispersive_operator_appends_a_polarisation_block() {
        // A Debye material on every element appends a 3*Np P block per
        // element; n_dof grows by exactly 3*Np*n_disp_elem.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let n_elem = mesh.n_tets();
        let np = 10; // order 2
        let mat = DebyeMaterial { eps_inf: 2.0, eps_static: 5.0, tau: 0.4 };
        let mats =
            vec![ElemMaterial::isotropic(mat.eps_inf, 1.0, 0.0); n_elem];
        let disp: Vec<(usize, DebyeMaterial)> =
            (0..n_elem).map(|e| (e, mat)).collect();
        let op = MaxwellOperator::new_with_materials_ports_dispersive(
            &mesh, 2, 1.0, &mats, &[], &disp,
        );
        assert_eq!(op.n_dispersive(), n_elem);
        assert_eq!(
            op.n_dof(),
            6 * np * n_elem + 3 * np * n_elem,
            "augmented n_dof = [E,H] + appended P block"
        );

        // The augmented apply must stay finite on a generic state.
        let n = op.n_dof();
        let y: Vec<f64> =
            (0..n).map(|i| (0.3 + i as f64 * 0.017).sin()).collect();
        let dy = op.apply(&y);
        assert!(dy.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dispersive_sparse_assembly_matches_matrix_free_apply() {
        // The augmented sparse CSR (with the P rows/cols and the E<->P
        // coupling) reproduces the matrix-free apply on a Debye mesh.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(2, 2, 1, 1.0, 1.0, 1.0);
        let n_elem = mesh.n_tets();
        let mat = DebyeMaterial { eps_inf: 2.5, eps_static: 7.0, tau: 0.3 };
        let mats =
            vec![ElemMaterial::isotropic(mat.eps_inf, 1.0, 0.0); n_elem];
        // Make half the elements dispersive, exercise the mixed path.
        let disp: Vec<(usize, DebyeMaterial)> =
            (0..n_elem).step_by(2).map(|e| (e, mat)).collect();
        let op = MaxwellOperator::new_with_materials_ports_dispersive(
            &mesh, 2, 1.0, &mats, &[], &disp,
        );
        let n = op.n_dof();
        let csr = op.assemble_sparse();
        assert_eq!(csr.n, n);

        let v: Vec<f64> =
            (0..n).map(|i| (1.0 + i as f64 * 0.07).cos()).collect();
        let sp = csr.matvec(&v);
        let mf = op.apply(&v);
        let err: f64 = sp
            .iter()
            .zip(&mf)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = mf.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            err < 1e-10 * scale.max(1.0),
            "dispersive sparse vs matrix-free: err {err}, scale {scale}"
        );
    }

    #[test]
    fn debye_operator_reproduces_the_analytic_permittivity() {
        // Physics gate: the assembled augmented operator implements the
        // Debye ADE, so its polarisation block reproduces the analytic
        // complex permittivity ε(ω) = ε_∞ + (ε_s − ε_∞)/(1 + iωτ).
        //
        // Every appended P row carries Ṗ = a·P + g·E with a = −1/τ,
        // g = (ε_s − ε_∞)/τ. Sinusoidal steady state gives the polarisation
        // phasor P = g/(iω − a)·E, and D = ε_∞·E + P, so the medium's
        // permittivity is ε(ω) = ε_∞ + g/(iω − a). Reading (a, g) straight
        // off the verbatim sparse state-space matrix and reconstructing
        // ε(ω) is an exact, mesh-independent check that the ADE-augmented
        // operator agrees with `DebyeMaterial::permittivity` across a sweep.
        use crate::mesh_gen::structured_box;

        // The polarisation phasor P/E = g/(iω − a), evaluated with plain
        // (re, im) tuples to keep the td crate free of a complex-number
        // dependency. g/(iω − a) = g·(−a − iω)/(a² + ω²).
        let p_phasor = |g: f64, a: f64, omega: f64| -> (f64, f64) {
            let d = a * a + omega * omega;
            (g * (-a) / d, g * (-omega) / d)
        };

        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let n_elem = mesh.n_tets();
        let np = 10; // order 2
        let debye =
            DebyeMaterial { eps_inf: 2.0, eps_static: 6.0, tau: 0.3 };
        let mats =
            vec![ElemMaterial::isotropic(debye.eps_inf, 1.0, 0.0); n_elem];
        let disp: Vec<(usize, DebyeMaterial)> =
            (0..n_elem).map(|e| (e, debye)).collect();
        let op = MaxwellOperator::new_with_materials_ports_dispersive(
            &mesh, 2, 0.0, &mats, &[], &disp,
        );

        let csr = op.assemble_sparse();
        let eh_len = 6 * np * n_elem;
        assert_eq!(csr.n, eh_len + 3 * np * n_elem);

        // Each polarisation row carries exactly two entries: g (the E
        // coupling, into the [E,H] block) and a (the P self-relaxation, on
        // the diagonal).
        let (mut a_vals, mut g_vals) = (Vec::new(), Vec::new());
        for row in eh_len..csr.n {
            let span = csr.row_ptr[row]..csr.row_ptr[row + 1];
            assert_eq!(
                span.len(),
                2,
                "polarisation row {row} must carry exactly the g*E \
                 coupling and the a*P diagonal"
            );
            for k in span {
                let col = csr.col_idx[k];
                if col == row {
                    a_vals.push(csr.values[k]);
                } else {
                    assert!(col < eh_len, "P row couples outside [E,H]");
                    g_vals.push(csr.values[k]);
                }
            }
        }
        // One uniform Debye material -> uniform coefficients across the
        // whole P block.
        let (want_a, want_g) = debye.relaxation_coeffs();
        for &a in &a_vals {
            assert!((a - want_a).abs() < 1e-12, "P-row a = {a}");
        }
        for &g in &g_vals {
            assert!((g - want_g).abs() < 1e-12, "P-row g = {g}");
        }

        // Reconstruct ε(ω) = ε_∞ + g/(iω − a) from the operator's ADE
        // coefficients and compare to the analytic Debye permittivity over
        // a sweep spanning ωτ from 0.06 to 6.
        for &omega in &[0.2, 1.0, 3.3, 12.0, 20.0] {
            let (pr, pi) = p_phasor(want_g, want_a, omega);
            let (op_re, op_im) = (debye.eps_inf + pr, pi);
            let (re, im) = debye.permittivity(omega);
            let err = ((op_re - re).powi(2) + (op_im - im).powi(2)).sqrt();
            let scale = (re * re + im * im).sqrt();
            assert!(
                err < 1e-12 * scale,
                "ω={omega}: operator ε = ({op_re}, {op_im}), \
                 analytic ({re}, {im})"
            );
        }

        // Static and high-frequency limits bracket the dispersion.
        let (lo_re, _) = p_phasor(want_g, want_a, 1e-6);
        let (hi_re, _) = p_phasor(want_g, want_a, 1e6);
        assert!((debye.eps_inf + lo_re - debye.eps_static).abs() < 1e-3);
        assert!((debye.eps_inf + hi_re - debye.eps_inf).abs() < 1e-3);
    }

    #[test]
    fn periodic_boundary_passes_plane_wave() {
        // C2 gate: a Gaussian plane-wave packet aligned with the periodic
        // axis propagates through a periodic-paired pair of opposite faces
        // without spurious reflection.
        //
        // The discriminator is a *forward-wave coherence* metric, for a
        // +z TEM packet with c = 1, vacuum, the initial state is
        // E_x(z) = f(z), H_y(z) = -f(z) and the coherence
        //   r = sum(E_x * (-H_y)) / sum(E_x^2)
        // is 1.0 by construction. A periodic +z-face leaves the packet a
        // clean forward wave, so r stays close to 1 across the run. A PEC
        // +z-face reflects the packet into a -z component (flipping E_x or
        // H_y, depending on flux), so r decays toward 0 as the packet
        // bounces. The energy of either run is conserved by the central
        // flux; the qualitative reflection signature is the metric that
        // distinguishes them.
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;

        let (lx, ly, lz) = (0.5, 0.5, 4.0);
        // Several z-cells per Gaussian sigma so the packet is resolved; the
        // x/y direction can stay coarse since the test field is z-only.
        let mesh = structured_box(1, 1, 16, lx, ly, lz);

        // All three axis-pairs are periodic, so the +z TEM packet propagates
        // through a translation-invariant medium, the only thing the run is
        // probing is the z-face periodic link. A PEC y-wall would clip a
        // +z TEM packet's tangential E_x, so the side walls must be wired
        // through too. The C2 discriminator is the periodic-vs-PEC contrast
        // on the z-faces only.
        let on = |pred: &dyn Fn([f64; 3]) -> bool| -> Vec<usize> {
            mesh.tris
                .iter()
                .enumerate()
                .filter(|(_, t)| t.iter().all(|&nd| pred(mesh.nodes[nd])))
                .map(|(i, _)| i)
                .collect()
        };
        let x_lo = on(&|p| p[0].abs() < 1e-9);
        let x_hi = on(&|p| (p[0] - lx).abs() < 1e-9);
        let y_lo = on(&|p| p[1].abs() < 1e-9);
        let y_hi = on(&|p| (p[1] - ly).abs() < 1e-9);
        let z0_tris = on(&|p| p[2].abs() < 1e-9);
        let zl_tris = on(&|p| (p[2] - lz).abs() < 1e-9);
        assert!(!z0_tris.is_empty() && !zl_tris.is_empty());

        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];

        // Initial state, a +z TEM Gaussian packet (E_x, H_y = -E_x). The
        // bump is narrow enough that its initial value at z = 0 / z = lz is
        // negligible (the periodic face is not loaded at t = 0); the run
        // covers more than one period so the packet has to traverse the
        // periodic face to keep going forward.
        let make_state = |op: &MaxwellOperator| -> Vec<f64> {
            let n = op.n_dof();
            let coords = op.node_coords();
            let z_c = 0.5 * lz;
            let sigma = 0.40;
            let mut y = vec![0.0; n];
            for (idx, p) in coords.iter().enumerate() {
                let f = (-((p[2] - z_c) / sigma).powi(2)).exp();
                y[idx * 6] = f;       // E_x
                y[idx * 6 + 4] = -f;  // H_y
            }
            y
        };
        // Forward-wave coherence on the E_x / H_y components, 1 for a
        // pure +z wave, 0 for a standing wave, -1 for pure -z.
        let coherence = |op: &MaxwellOperator, y: &[f64]| -> f64 {
            let np = op.re.n_nodes;
            let mut num = 0.0;
            let mut den = 0.0;
            for e in 0..op.n_elem {
                for node in 0..np {
                    let ex = y[(e * np + node) * 6];
                    let hy = y[(e * np + node) * 6 + 4];
                    num += ex * (-hy);
                    den += ex * ex;
                }
            }
            if den > 0.0 { num / den } else { 0.0 }
        };

        // (a) periodic run, z + x + y all periodic.
        let op_per = MaxwellOperator::new_with_materials_ports_dispersive_periodic(
            &mesh, 2, 0.0, &vacuum, &[], &[],
            &[
                PeriodicSpec { tris_a: x_lo.clone(), tris_b: x_hi.clone() },
                PeriodicSpec { tris_a: y_lo.clone(), tris_b: y_hi.clone() },
                PeriodicSpec { tris_a: z0_tris.clone(), tris_b: zl_tris.clone() },
            ],
        );
        let n = op_per.n_dof();
        let mut y_per = make_state(&op_per);
        let e0_per = op_per.field_energy(&y_per);
        let r0 = coherence(&op_per, &y_per);
        assert!(
            (r0 - 1.0).abs() < 1e-9,
            "initial coherence should be 1 by construction, got {r0}"
        );
        // Propagate well past one period so the packet has to cross the
        // periodic face. With central flux + periodic + no other boundaries
        // active for the test field, this stays a clean forward wave.
        let dt = 0.04;
        let steps = 250;  // 10 time units = 2.5 box periods at c = 1
        for _ in 0..steps {
            y_per = etd_step(|x| op_per.apply(x), &y_per, &vec![0.0; n], dt, 24);
        }
        assert!(y_per.iter().all(|v| v.is_finite()), "periodic run diverged");
        let e_per = op_per.field_energy(&y_per);
        let r_per = coherence(&op_per, &y_per);
        let de_per = (e_per - e0_per).abs() / e0_per;
        eprintln!(
            "DIAG periodic: energy ratio {:.4} (drift {de_per:.2e}), \
             forward coherence {r_per:.4}",
            e_per / e0_per,
        );
        // Central flux + periodic boundaries, exactly energy-conserving
        // (the periodic face is just an interior face), so the energy must
        // hold tight; the wave must stay a forward wave (no spurious
        // reflection at the periodic faces).
        assert!(
            de_per < 5e-3,
            "periodic energy drifted: {de_per:.2e} (kept {:.4})",
            e_per / e0_per,
        );
        assert!(
            r_per > 0.85,
            "periodic boundary leaked a backward wave: coherence {r_per:.3} \
             (was {r0:.3} initially)",
        );

        // (b) PEC reference, same x/y periodicity (so the side walls do
        // not contaminate the comparison), only the z-faces switched to
        // PEC. The packet bounces off the z = lz wall, and the forward
        // coherence drifts down as a backward wave builds up.
        let op_pec = MaxwellOperator::new_with_materials_ports_dispersive_periodic(
            &mesh, 2, 0.0, &vacuum, &[], &[],
            &[
                PeriodicSpec { tris_a: x_lo, tris_b: x_hi },
                PeriodicSpec { tris_a: y_lo, tris_b: y_hi },
            ],
        );
        let mut y_pec = make_state(&op_pec);
        for _ in 0..steps {
            y_pec = etd_step(|x| op_pec.apply(x), &y_pec, &vec![0.0; n], dt, 24);
        }
        assert!(y_pec.iter().all(|v| v.is_finite()));
        let r_pec = coherence(&op_pec, &y_pec);
        eprintln!(
            "DIAG PEC reference: forward coherence {r_pec:.4}",
        );
        // The PEC run must show clearly less forward-coherence than the
        // periodic one, the discriminator that proves the periodic case
        // is genuinely passing the wave through, not just incidentally
        // bouncing.
        assert!(
            r_per - r_pec > 0.3,
            "periodic vs PEC contrast too weak: periodic coherence \
             {r_per:.3}, PEC coherence {r_pec:.3}",
        );
    }

    #[test]
    fn periodic_boundary_is_translation_invariant() {
        // C2 sanity: a uniform field in a box with a periodic pair on the
        // z-faces stays exactly uniform under propagation, the periodic
        // face is invisible to a translation-invariant state. With PEC on
        // the z-faces (no periodic) the same state is *not* a steady state
        // (the PEC ghost flips tangential E, creating a non-zero jump
        // across z = 0 / z = lz), so the field develops structure. The two
        // runs together prove that the periodic glue is genuinely
        // translation-invariant, not just numerically close to a PEC run.
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;

        let (lx, ly, lz) = (0.5, 0.5, 1.5);
        let mesh = structured_box(1, 1, 4, lx, ly, lz);
        let z0_tris: Vec<usize> = mesh
            .tris
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                t.iter().all(|&nd| mesh.nodes[nd][2].abs() < 1e-9)
            })
            .map(|(i, _)| i)
            .collect();
        let zl_tris: Vec<usize> = mesh
            .tris
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                t.iter().all(|&nd| (mesh.nodes[nd][2] - lz).abs() < 1e-9)
            })
            .map(|(i, _)| i)
            .collect();

        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];

        // Uniform E_x = 1, all other components zero. The side walls
        // (x = 0 / lx and y = 0 / ly) carry PEC; E_x is normal to the
        // x-walls (the PEC tangential-E ghost is zero on a purely normal
        // field), and on the y-walls it IS tangential, so this state is
        // not a steady state of the full all-PEC box, but the variation
        // it induces is the same in the periodic and PEC reference runs
        // (the side walls are identical between them). The discriminator
        // is the *additional* drift caused by the z-face treatment.
        let make_state = |op: &MaxwellOperator| -> Vec<f64> {
            let n = op.n_dof();
            let mut y = vec![0.0; n];
            let np = op.re.n_nodes;
            for e in 0..op.n_elem {
                for node in 0..np {
                    y[(e * np + node) * 6] = 1.0;
                }
            }
            y
        };
        // Standard deviation of E_x across all DG nodes, a structure
        // metric: 0 means perfectly uniform, anything else is induced
        // variation.
        let std_ex = |op: &MaxwellOperator, y: &[f64]| -> f64 {
            let np = op.re.n_nodes;
            let n_pts = op.n_elem * np;
            let mean: f64 = (0..n_pts)
                .map(|i| y[i * 6])
                .sum::<f64>()
                / n_pts as f64;
            let var: f64 = (0..n_pts)
                .map(|i| (y[i * 6] - mean).powi(2))
                .sum::<f64>()
                / n_pts as f64;
            var.sqrt()
        };

        // To isolate the z-face effect, build TWO operators on the *same*
        // mesh with identical side walls but different z-face treatment:
        // (a) periodic on z, (b) PEC on z. Compare additional drift.
        let op_per = MaxwellOperator::new_with_materials_ports_dispersive_periodic(
            &mesh, 2, 0.0, &vacuum, &[], &[],
            &[PeriodicSpec { tris_a: z0_tris, tris_b: zl_tris }],
        );
        let op_pec = MaxwellOperator::new_with_materials(
            &mesh, 2, 0.0, &vacuum,
        );

        // Critically: we use a problem whose *side* walls are also
        // periodic, so the *only* boundaries are the z-faces. Build a
        // fully-periodic box (all six faces in three pairs); then the
        // uniform E_x truly is a steady state of the periodic operator.
        // A PEC equivalent on the z-faces (with the x/y faces still
        // periodic) is not.
        let on = |pred: &dyn Fn([f64; 3]) -> bool| -> Vec<usize> {
            mesh.tris
                .iter()
                .enumerate()
                .filter(|(_, t)| t.iter().all(|&nd| pred(mesh.nodes[nd])))
                .map(|(i, _)| i)
                .collect()
        };
        let x_lo = on(&|p| p[0].abs() < 1e-9);
        let x_hi = on(&|p| (p[0] - lx).abs() < 1e-9);
        let y_lo = on(&|p| p[1].abs() < 1e-9);
        let y_hi = on(&|p| (p[1] - ly).abs() < 1e-9);
        let z_lo = on(&|p| p[2].abs() < 1e-9);
        let z_hi = on(&|p| (p[2] - lz).abs() < 1e-9);
        let op_all_per =
            MaxwellOperator::new_with_materials_ports_dispersive_periodic(
                &mesh, 2, 0.0, &vacuum, &[], &[],
                &[
                    PeriodicSpec { tris_a: x_lo.clone(), tris_b: x_hi.clone() },
                    PeriodicSpec { tris_a: y_lo.clone(), tris_b: y_hi.clone() },
                    PeriodicSpec { tris_a: z_lo.clone(), tris_b: z_hi.clone() },
                ],
            );
        // Reference: keep x/y periodic but leave z as PEC, only the
        // z-face treatment differs.
        let op_z_pec =
            MaxwellOperator::new_with_materials_ports_dispersive_periodic(
                &mesh, 2, 0.0, &vacuum, &[], &[],
                &[
                    PeriodicSpec { tris_a: x_lo, tris_b: x_hi },
                    PeriodicSpec { tris_a: y_lo, tris_b: y_hi },
                ],
            );

        let n = op_all_per.n_dof();
        assert_eq!(n, op_z_pec.n_dof());
        // Suppress unused warnings, the two-operator variants are the
        // reference; op_per / op_pec exist for explanatory completeness.
        let _ = (&op_per, &op_pec);

        let mut y_all = make_state(&op_all_per);
        let mut y_zpec = make_state(&op_z_pec);
        let s0 = std_ex(&op_all_per, &y_all);
        assert!(
            s0 < 1e-12,
            "initial state should be uniform, std = {s0:e}"
        );

        let dt = 0.05;
        let steps = 80;
        let z = vec![0.0; n];
        for _ in 0..steps {
            y_all = etd_step(|x| op_all_per.apply(x), &y_all, &z, dt, 18);
            y_zpec = etd_step(|x| op_z_pec.apply(x), &y_zpec, &z, dt, 18);
        }
        assert!(y_all.iter().all(|v| v.is_finite()));
        assert!(y_zpec.iter().all(|v| v.is_finite()));

        let s_all = std_ex(&op_all_per, &y_all);
        let s_zpec = std_ex(&op_z_pec, &y_zpec);
        eprintln!(
            "DIAG translation-invariance: fully-periodic std(E_x) = \
             {s_all:.3e}, z-PEC reference std(E_x) = {s_zpec:.3e}"
        );
        // Fully-periodic: a uniform field is an exact steady state of the
        // operator, so the variation must stay at round-off.
        assert!(
            s_all < 1e-9,
            "fully-periodic box: uniform E_x developed structure, \
             std(E_x) = {s_all:.3e}",
        );
        // PEC-on-z reference: the PEC ghost on the z-faces drives a real
        // non-zero variation, well above the periodic-run round-off.
        assert!(
            s_zpec > 100.0 * s_all,
            "z-PEC reference must develop visibly more structure than \
             the periodic run: periodic {s_all:.3e}, z-PEC {s_zpec:.3e}",
        );
    }

    #[test]
    fn floquet_port_transmits_through_empty_unit_cell() {
        // C3 gate: a normal-incidence Floquet port injects a uniform plane
        // wave into a periodic unit cell; the wave traverses the cell and is
        // cleanly extracted at the opposite face. Verified for both TE and
        // TM polarisations.
        //
        // The cell is a structured box with all four lateral faces periodic
        // (the C2 machinery), and Floquet ports on the −z (TX) and +z (RX)
        // faces. Free space inside, c = 1, so a Gaussian pulse injected at
        // TX arrives at RX after a delay of exactly `lz` and the run is
        // stopped before any far-end reflection can return. The reflected
        // amplitude at the TX port (its backward modal amplitude) stays
        // small at a matched normal-incidence port.
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        use crate::waveguide::FloquetPolarisation;
        use std::f64::consts::PI;

        // Run the same transmission experiment for the two polarisations and
        // return the diagnostic numbers (transmitted delay, reflection
        // ratio, peak amplitudes) so both can be asserted at the end.
        let run_pol = |pol: FloquetPolarisation| -> (f64, f64, f64, f64) {
            // Aspect ratio: a few cells across so the plane wave is well
            // resolved laterally; many cells along z so the Gaussian pulse
            // fits cleanly into the cell.
            let (lx, ly, lz) = (0.5, 0.5, 6.0);
            let mesh = structured_box(2, 2, 24, lx, ly, lz);

            let on = |pred: &dyn Fn([f64; 3]) -> bool| -> Vec<usize> {
                mesh.tris
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| {
                        t.iter().all(|&nd| pred(mesh.nodes[nd]))
                    })
                    .map(|(i, _)| i)
                    .collect()
            };
            let x_lo = on(&|p| p[0].abs() < 1e-9);
            let x_hi = on(&|p| (p[0] - lx).abs() < 1e-9);
            let y_lo = on(&|p| p[1].abs() < 1e-9);
            let y_hi = on(&|p| (p[1] - ly).abs() < 1e-9);
            let z0_tris = on(&|p| p[2].abs() < 1e-9);
            let zl_tris = on(&|p| (p[2] - lz).abs() < 1e-9);
            assert!(!z0_tris.is_empty() && !zl_tris.is_empty());

            // The −z Floquet port (inward +z), and the +z Floquet port
            // (inward −z). Both at normal incidence; both with the same
            // polarisation so transmission is mode-pure.
            let port_tx = PortSpec {
                tris: z0_tris,
                mode: Some(PortMode::Floquet(FloquetPort {
                    origin: [0.0, 0.0, 0.0],
                    u_hat: [1.0, 0.0, 0.0],
                    v_hat: [0.0, 1.0, 0.0],
                    w_hat: [0.0, 0.0, 1.0],
                    a: lx,
                    b: ly,
                    polarisation: pol,
                    scan_theta: 0.0,
                    scan_phi: 0.0,
                    e_override: None,
                })),
            };
            let port_rx = PortSpec {
                tris: zl_tris,
                mode: Some(PortMode::Floquet(FloquetPort {
                    origin: [0.0, 0.0, lz],
                    u_hat: [1.0, 0.0, 0.0],
                    v_hat: [0.0, 1.0, 0.0],
                    w_hat: [0.0, 0.0, -1.0],
                    a: lx,
                    b: ly,
                    polarisation: pol,
                    scan_theta: 0.0,
                    scan_phi: 0.0,
                    e_override: None,
                })),
            };
            let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
            let op = MaxwellOperator::new_with_materials_ports_dispersive_periodic(
                &mesh, 2, 1.0, &vacuum,
                &[port_tx, port_rx], &[],
                &[
                    PeriodicSpec { tris_a: x_lo, tris_b: x_hi },
                    PeriodicSpec { tris_a: y_lo, tris_b: y_hi },
                ],
            );
            let n = op.n_dof();
            // Floquet ports are non-dispersive plane waves, zero cutoff.
            assert!(op.port_cutoff(0).abs() < 1e-12);
            assert!(op.port_cutoff(1).abs() < 1e-12);

            let b_tx = op.port_source(0);
            assert!(
                b_tx.iter().any(|&x| x != 0.0),
                "Floquet TX port source is empty for {pol:?}",
            );

            // Modulated-Gaussian drive, well inside the resolved band.
            let omega0 = 1.5 * PI;
            let (t0, tau) = (2.0, 0.6);
            let pulse = |t: f64| {
                (-((t - t0) / tau).powi(2)).exp() * (omega0 * (t - t0)).sin()
            };

            // Stop just before the RX port's response could be contaminated
            // by a hypothetical round-trip reflection (lz + 2·lz = 3·lz in
            // a worst case; we stop at ~lz + 4·tau ≪ 3·lz).
            let dt = 0.02;
            let steps = 600;
            let mut y = vec![0.0; n];
            let mut tx_pe: Vec<f64> = Vec::with_capacity(steps);
            let mut tx_ph: Vec<f64> = Vec::with_capacity(steps);
            let mut rx_pe: Vec<f64> = Vec::with_capacity(steps);
            let mut rx_ph: Vec<f64> = Vec::with_capacity(steps);
            for s in 0..steps {
                let t = s as f64 * dt;
                let g = pulse(t);
                let bvec: Vec<f64> = b_tx.iter().map(|x| x * g).collect();
                y = etd_step(|x| op.apply(x), &y, &bvec, dt, 20);
                let (pe0, ph0) = op.port_modal_projections(&y, 0);
                let (pe1, ph1) = op.port_modal_projections(&y, 1);
                tx_pe.push(pe0);
                tx_ph.push(ph0);
                rx_pe.push(pe1);
                rx_ph.push(ph1);
            }
            assert!(y.iter().all(|v| v.is_finite()), "{pol:?} run diverged");

            // Modal forward / backward split. Free-space impedance Z = 1.
            let dft = |sig: &[f64], omega: f64| -> (f64, f64) {
                let (mut re, mut im) = (0.0, 0.0);
                for (k, &x) in sig.iter().enumerate() {
                    let t = k as f64 * dt;
                    re += x * (omega * t).cos();
                    im -= x * (omega * t).sin();
                }
                (re * dt, im * dt)
            };
            let cmag = |z: (f64, f64)| (z.0 * z.0 + z.1 * z.1).sqrt();
            // Pick the carrier, well inside the resolved band.
            let omega = 1.5 * PI;
            let pe_tx = dft(&tx_pe, omega);
            let ph_tx = dft(&tx_ph, omega);
            let pe_rx = dft(&rx_pe, omega);
            let ph_rx = dft(&rx_ph, omega);
            // At the TX port the forward (incident) is A = (P_e + P_h)/2,
            // the backward (reflected) is B = (P_e − P_h)/2 (Z = 1).
            let a_tx = (0.5 * (pe_tx.0 + ph_tx.0), 0.5 * (pe_tx.1 + ph_tx.1));
            let b_tx_amp =
                (0.5 * (pe_tx.0 - ph_tx.0), 0.5 * (pe_tx.1 - ph_tx.1));
            // At the RX port "backward" is the wave outgoing through +z,
            // i.e. the transmitted wave (the port's inward normal is −z).
            let b_rx =
                (0.5 * (pe_rx.0 - ph_rx.0), 0.5 * (pe_rx.1 - ph_rx.1));
            let refl_ratio = cmag(b_tx_amp) / cmag(a_tx).max(1e-30);
            let trans_ratio = cmag(b_rx) / cmag(a_tx).max(1e-30);
            eprintln!(
                "DIAG floquet {:?}: |A_TX|={:.3e} |B_TX|={:.3e} \
                 |B_RX|={:.3e} refl={refl_ratio:.4} trans={trans_ratio:.4}",
                pol,
                cmag(a_tx),
                cmag(b_tx_amp),
                cmag(b_rx),
            );

            // Time-domain peak-arrival check on (P_e at RX): the
            // transmitted pulse should peak at roughly t0 + lz (c = 1).
            let mut peak = 0.0_f64;
            let mut tpk = 0.0;
            for (k, &v) in rx_pe.iter().enumerate() {
                if v.abs() > peak {
                    peak = v.abs();
                    tpk = k as f64 * dt;
                }
            }
            // Peak amplitudes for the diagnostic, and the arrival delay.
            (refl_ratio, trans_ratio, tpk - t0, peak)
        };

        for &pol in &[FloquetPolarisation::Te, FloquetPolarisation::Tm] {
            let (refl, trans, delay, peak) = run_pol(pol);
            // The transmitted pulse arrives at roughly t = t0 + lz, with
            // lz = 6 and t0 = 2. Allow ±0.7 (about one carrier period) for
            // the discrete peak step.
            assert!(
                (delay - 6.0).abs() < 0.7,
                "{pol:?} transit delay {delay:.2}, expected ≈ 6.0 (lz)"
            );
            // The receive-port modal projection picked up a non-negligible
            // signal, the plane wave actually arrived.
            assert!(
                peak > 5e-2,
                "{pol:?} no transmitted signal at RX: peak {peak:.2e}"
            );
            // Most of the incident energy passes through; reflected ≪ A.
            // At a matched normal-incidence Floquet port the reflection
            // stays well below the transmitted amplitude.
            assert!(
                refl < 0.25,
                "{pol:?} TX port not matched: refl ratio {refl:.3} \
                 (incident |A| should dominate the spurious backward part)",
            );
            // Transmission ratio bracket: with the periodic side walls the
            // plane wave is preserved across the cell, so |B_RX|/|A_TX| is
            // near unity. The matched-port modal split has its own ~few-
            // percent error budget at finite dt and finite mesh resolution,
            // so a wide bracket is the right gate.
            assert!(
                (0.5..1.5).contains(&trans),
                "{pol:?} transmission |B_RX|/|A_TX| = {trans:.3} \
                 (expected ≈ 1 for a lossless cell)",
            );
        }
    }
}
