// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! 2D wave-port mode solver: transverse eigenmodes at a port cross-section.
//!
//! A modal port injects and extracts a known transverse field profile
//! `(e_t, h_t)`. For a rectangular waveguide or a coaxial line that
//! profile is analytic ([`crate::waveguide`]); for an *arbitrary*
//! cross-section (a ridged guide, an L-shaped duct, a microstrip or
//! coplanar line) the profile has no closed form and is computed here by
//! a 2D eigensolve on the port-face triangulation. Backend-agnostic,
//! the time-domain and frequency-domain solvers share it.
//!
//! Two solvers, both on the extracted [`PortMesh2D`] cross-section:
//!
//! - [`solve_modes`], the **scalar Helmholtz** eigenproblem
//!   `∇_t² ψ + k_c² ψ = 0` (P1 nodal), giving the `TE`/`TM` modes and
//!   cutoffs `k_c` of a *homogeneously filled* hollow guide. `TM` is
//!   Dirichlet (`E_z = 0` on PEC), `TE` Neumann.
//! - [`solve_vector_modes`], the **full-vector hybrid** eigenproblem
//!   (mixed Nédélec-edge `E_t` + Lagrange-nodal `E_z`, eigenvalue
//!   `λ = β²/k0² = n_eff²`) with per-triangle `ε_r`, so it resolves the
//!   quasi-TEM mode of an *inhomogeneous* (substrate + air) line. PEC
//!   walls, the outer boundary and any internal conductor (a microstrip
//!   trace, via [`PortMesh2D::on_pec`]), impose `tangential E = 0`.
//!
//! A solved mode becomes a [`NumericalMode`], which samples `(e_t, h_t)`
//! at arbitrary points for the port machinery: the scalar path
//! barycentric-interpolates a nodal `E_t`, the vector path evaluates the
//! Whitney edge field directly. Validation against analytic cutoffs and
//! effective indices lives in the tests.



/// A port face flattened to its 2D cross-section: nodes in the port
/// plane's local `(u, v)` coordinates, the triangles connecting them,
/// and which nodes lie on the boundary (PEC wall) of the cross-section.
#[derive(Clone, Debug)]
pub struct PortMesh2D {
    /// Local 2D coordinates of each distinct cross-section node.
    pub nodes: Vec<[f64; 2]>,
    /// Triangles as triples of indices into `nodes`.
    pub tris: Vec<[usize; 3]>,
    /// `true` for a node on the outer boundary of the cross-section,
    /// a boundary edge is one used by exactly one triangle. These get
    /// the Dirichlet condition for `TM` modes.
    pub on_boundary: Vec<bool>,
    /// `true` for a node lying on an *internal* PEC conductor (e.g. a
    /// microstrip trace cutting through the cross-section). These carry
    /// the same `tangential E = 0` condition as the outer wall; an edge
    /// with both endpoints on the conductor is a PEC edge. Empty / all
    /// false when the cross-section has no internal conductor.
    pub on_pec: Vec<bool>,
    /// The local frame `(û, v̂)` and origin, so a solved mode profile
    /// can be lifted back to 3D global coordinates.
    pub u_hat: [f64; 3],
    pub v_hat: [f64; 3],
    pub origin: [f64; 3],
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl PortMesh2D {
    /// Flatten a set of port-face triangles (given by global node
    /// coordinates and connectivity) into a 2D cross-section mesh.
    ///
    /// `face_tris` lists the triangles as triples of indices into the
    /// *global* `global_nodes`. `inward_normal` is the port's inward
    /// normal (any nonzero vector along the face normal); it fixes the
    /// out-of-plane axis. An in-plane orthonormal frame `(û, v̂)` is
    /// built from the first triangle's edge, and every node is projected
    /// onto it. Boundary nodes are detected from edges used by a single
    /// triangle.
    ///
    /// `pec_global`, if given, is a per-global-node mask marking nodes
    /// that lie on an internal PEC conductor (a microstrip trace); the
    /// resulting cross-section nodes inherit it as [`on_pec`](Self::on_pec).
    pub fn from_face(
        global_nodes: &[[f64; 3]],
        face_tris: &[[usize; 3]],
        inward_normal: [f64; 3],
        pec_global: Option<&[bool]>,
    ) -> PortMesh2D {
        // Normalise the out-of-plane axis.
        let nl = dot3(inward_normal, inward_normal).sqrt();
        let w_hat = [
            inward_normal[0] / nl,
            inward_normal[1] / nl,
            inward_normal[2] / nl,
        ];
        // Build an in-plane û from the first triangle's first edge,
        // orthogonalised against ŵ; v̂ = ŵ × û completes the frame.
        let t0 = face_tris[0];
        let p0 = global_nodes[t0[0]];
        let p1 = global_nodes[t0[1]];
        let mut e = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let edn = dot3(e, w_hat);
        e = [
            e[0] - edn * w_hat[0],
            e[1] - edn * w_hat[1],
            e[2] - edn * w_hat[2],
        ];
        let el = dot3(e, e).sqrt();
        let u_hat = [e[0] / el, e[1] / el, e[2] / el];
        let v_hat = cross3(w_hat, u_hat);
        let origin = p0;

        // Collect distinct nodes (remap global → local index) and project.
        let mut remap: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut nodes: Vec<[f64; 2]> = Vec::new();
        let mut local_to_global: Vec<usize> = Vec::new();
        let mut tris: Vec<[usize; 3]> = Vec::with_capacity(face_tris.len());
        let project = |g: usize| -> [f64; 2] {
            let p = global_nodes[g];
            let d = [
                p[0] - origin[0],
                p[1] - origin[1],
                p[2] - origin[2],
            ];
            [dot3(d, u_hat), dot3(d, v_hat)]
        };
        for &t in face_tris {
            let mut local = [0usize; 3];
            for (k, &g) in t.iter().enumerate() {
                let idx = *remap.entry(g).or_insert_with(|| {
                    nodes.push(project(g));
                    local_to_global.push(g);
                    nodes.len() - 1
                });
                local[k] = idx;
            }
            tris.push(local);
        }
        // Internal-PEC mask: a cross-section node inherits the PEC flag of
        // its global mesh node (e.g. on a microstrip trace surface).
        let on_pec: Vec<bool> = match pec_global {
            Some(mask) => {
                local_to_global.iter().map(|&g| mask[g]).collect()
            }
            None => vec![false; nodes.len()],
        };

        // Boundary edges are used by exactly one triangle. Tally each
        // undirected edge; mark the endpoints of singly-used edges.
        let mut edge_count: std::collections::HashMap<(usize, usize), u32> =
            std::collections::HashMap::new();
        let key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
        for t in &tris {
            for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edge_count.entry(key(a, b)).or_insert(0) += 1;
            }
        }
        let mut on_boundary = vec![false; nodes.len()];
        for (&(a, b), &c) in &edge_count {
            if c == 1 {
                on_boundary[a] = true;
                on_boundary[b] = true;
            }
        }

        PortMesh2D { nodes, tris, on_boundary, on_pec, u_hat, v_hat, origin }
    }

    /// Number of cross-section nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Per-triangle area and the three constant P1 gradients
    /// `∇λ_i` (2D), for one triangle. Returns `(area, [g0, g1, g2])`.
    fn tri_geom(&self, t: [usize; 3]) -> (f64, [[f64; 2]; 3]) {
        let p = [self.nodes[t[0]], self.nodes[t[1]], self.nodes[t[2]]];
        // Edge vectors; signed area via the cross product of two edges.
        let (x0, y0) = (p[0][0], p[0][1]);
        let (x1, y1) = (p[1][0], p[1][1]);
        let (x2, y2) = (p[2][0], p[2][1]);
        let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        let area = 0.5 * det.abs();
        // P1 gradients: ∇λ_i = (1/2A)·[y_{j}-y_{k}, x_{k}-x_{j}] (cyclic).
        // Use the signed det so the sign is consistent across the formula.
        let inv = 1.0 / det;
        let g0 = [(y1 - y2) * inv, (x2 - x1) * inv];
        let g1 = [(y2 - y0) * inv, (x0 - x2) * inv];
        let g2 = [(y0 - y1) * inv, (x1 - x0) * inv];
        (area, [g0, g1, g2])
    }

    /// Assemble the dense stiffness `S` and lumped (diagonal) mass `m`
    /// for the P1 scalar problem. `S` is `n×n` row-major; `m` is the
    /// length-`n` diagonal (row-sum-lumped consistent mass), which makes
    /// the generalized eigenproblem `S ψ = k_c² diag(m) ψ` reducible to
    /// a symmetric standard problem without a Cholesky factorisation.
    pub fn assemble(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_nodes();
        let mut s = vec![0.0; n * n];
        let mut m = vec![0.0; n];
        for &t in &self.tris {
            let (area, g) = self.tri_geom(t);
            // Stiffness: S_ij += area · (∇λ_i · ∇λ_j).
            for a in 0..3 {
                for b in 0..3 {
                    let val = area * (g[a][0] * g[b][0] + g[a][1] * g[b][1]);
                    s[t[a] * n + t[b]] += val;
                }
            }
            // Lumped mass: each node gets area/3 (row-sum of the
            // consistent element mass area/12·[[2,1,1],…] = area/3).
            for a in 0..3 {
                m[t[a]] += area / 3.0;
            }
        }
        (s, m)
    }
}

/// One transverse eigenmode of the cross-section.
#[derive(Clone, Debug)]
pub struct PortEigenmode {
    /// Cutoff wavenumber `k_c` (operator units; `ω_c = c·k_c`, and with
    /// `c = 1` the cutoff angular frequency equals `k_c`).
    pub k_c: f64,
    /// The scalar modal field `ψ` at every cross-section node
    /// (`E_z` for a `TM` mode, `H_z` for a `TE` mode).
    pub psi: Vec<f64>,
}

/// Boundary condition for the scalar mode solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeKind {
    /// `TM`: `E_z = 0` (Dirichlet) on the PEC boundary.
    Tm,
    /// `TE`: `∂H_z/∂n = 0` (natural Neumann), no constraint applied.
    Te,
}

/// Solve the scalar Helmholtz eigenproblem `S ψ = k_c² diag(m) ψ` on the
/// cross-section, returning the `n_modes` lowest-cutoff propagating modes
/// (smallest positive `k_c²`), sorted ascending.
///
/// `TM` modes pin the boundary nodes to zero (Dirichlet); `TE` modes
/// leave them free. The generalized problem is reduced to the symmetric
/// standard problem `B φ = k_c² φ` with `B = D^{-1/2} S D^{-1/2}` and
/// `D = diag(m)`, then `ψ = D^{-1/2} φ`. Dense, intended for the modest
/// node counts of a single port face.
pub fn solve_modes(
    mesh: &PortMesh2D,
    kind: ModeKind,
    n_modes: usize,
) -> Vec<PortEigenmode> {
    let n_full = mesh.n_nodes();
    let (s_full, m_full) = mesh.assemble();

    // PEC nodes (outer wall + any internal conductor) carry Dirichlet for
    // TM (E_z = 0) and Neumann for TE; an internal conductor still pins
    // E_z, so it is constrained in both, but for the scalar TE Neumann
    // problem only the internal conductor (not the outer wall) is pinned.
    let pec = |i: usize| {
        mesh.on_boundary[i] || mesh.on_pec.get(i).copied().unwrap_or(false)
    };
    let keep: Vec<usize> = match kind {
        ModeKind::Tm => (0..n_full).filter(|&i| !pec(i)).collect(),
        ModeKind::Te => (0..n_full)
            .filter(|&i| !mesh.on_pec.get(i).copied().unwrap_or(false))
            .collect(),
    };
    let n = keep.len();
    if n == 0 {
        return Vec::new();
    }

    // Symmetric reduced problem B = D^{-1/2} S D^{-1/2}.
    let d_inv_sqrt: Vec<f64> =
        keep.iter().map(|&i| 1.0 / m_full[i].sqrt()).collect();
    let b = faer::Mat::<f64>::from_fn(n, n, |i, j| {
        s_full[keep[i] * n_full + keep[j]] * d_inv_sqrt[i] * d_inv_sqrt[j]
    });

    let eig = match b.eigen() {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let evals = eig.S().column_vector();
    let evecs = eig.U();

    // Collect (k_c², column) for positive eigenvalues, then sort.
    let mut idx: Vec<(f64, usize)> = (0..n)
        .filter_map(|k| {
            let lam = evals[k].re;
            // Drop the (near-)zero TE constant mode and any numerical
            // negatives; a propagating mode has k_c² > 0.
            if lam > 1e-9 {
                Some((lam, k))
            } else {
                None
            }
        })
        .collect();
    idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    idx.into_iter()
        .take(n_modes)
        .map(|(lam, k)| {
            // ψ = D^{-1/2} φ, scattered back to full node indexing.
            let mut psi = vec![0.0; n_full];
            for li in 0..n {
                psi[keep[li]] = evecs[(li, k)].re * d_inv_sqrt[li];
            }
            PortEigenmode { k_c: lam.sqrt(), psi }
        })
        .collect()
}

/// A solved numerical port mode, ready to be sampled as a transverse
/// `(e_t, h_t)` profile at arbitrary points on the port face, the
/// drop-in replacement for an analytic [`crate::waveguide::RectPort`]
/// profile when the cross-section has no closed-form mode.
///
/// The transverse fields follow from the scalar potential `ψ` by the
/// standard waveguide relations. With `ψ` piecewise-linear (P1) its
/// transverse gradient `∇_t ψ` is constant per triangle, so:
/// - **TM** (`ψ = E_z`): `E_t ∝ ∇_t ψ`,
/// - **TE** (`ψ = H_z`): `E_t ∝ ẑ × ∇_t ψ`.
///
/// `h_t = ŵ × e_t` (the inward normal `ŵ` plays the role of `ẑ`), and
/// the modal impedance enters the forward/backward split via
/// [`te_impedance`](Self::te_impedance). The profile is normalised so
/// its peak transverse magnitude over the cross-section is unity, to
/// match the analytic profiles' order-unity convention.
#[derive(Clone, Debug)]
pub struct NumericalMode {
    mesh: PortMesh2D,
    /// Transverse electric field `E_t` at each cross-section node, in the
    /// port-plane `(u, v)` components, the **scalar**-path profile
    /// (`from_scalar`); `e_profile` barycentric-interpolates it. Empty
    /// when the Ned-2 representation is used.
    e_uv_node: Vec<[f64; 2]>,
    /// The **vector**-path profile (`from_vector`): the full Nédélec-2
    /// second-kind edge + face coefficient set plus the per-triangle edge
    /// data, so `e_profile` evaluates the degree-2 vector field *directly*
    /// at each query point. This matches the basis the 3-D `Nedelec2Basis`
    /// carries on a port face, so the mode projection in `sparam_waveport`
    /// is exact (modulo Galerkin error) rather than the `O(h)` lossy
    /// projection a Nédélec-1 / P1 hybrid would produce. `None` for scalar.
    ned2: Option<Ned2ModeData>,
    /// Inverse peak `|E_t|` over the cross-section, the unit-peak
    /// normalisation.
    inv_peak: f64,
    /// Inward normal `ŵ = û × v̂` (global), the mode propagation axis.
    w_hat: [f64; 3],
    /// Cutoff wavenumber (`0` for a quasi-TEM numerical mode).
    cutoff: f64,
    /// Modal-impedance model for the forward/backward split.
    z_model: ImpedanceModel,
}

/// Per-mode Ned-2 coefficient bundle and per-triangle edge data needed for
/// direct basis-function evaluation in `e_profile`.
#[derive(Clone, Debug)]
struct Ned2ModeData {
    /// Ned-2 edge coefficients (2 per global edge).
    e_edge: Vec<[f64; 2]>,
    /// Ned-2 face (bubble) coefficients (2 per triangle, interior DOFs).
    e_face: Vec<[f64; 2]>,
    /// Per-triangle edge orientation / sign / length data.
    tri_edges: Vec<TriEdges>,
    /// Longitudinal `E_z` P2 coefficients on the vertex DOFs (per global node)
    /// and edge-midpoint DOFs (per global edge). A quasi-TEM mode in an
    /// inhomogeneous cross-section carries `E_z ≠ 0`; `e_profile` adds it along
    /// the propagation axis, so the reconstructed field is `E_t + ẑ·E_z`.
    e_z_node: Vec<f64>,
    e_z_edge: Vec<f64>,
}

/// How a numerical mode's wave impedance varies with frequency.
#[derive(Clone, Copy, Debug)]
enum ImpedanceModel {
    /// `TE`: `Z = 1/√(1−(k_c/ω)²)`.
    Te { k_c: f64 },
    /// `TM`: `Z = √(1−(k_c/ω)²)`.
    Tm { k_c: f64 },
    /// Flat `Z = z`, a quasi-TEM / hybrid vector mode, with `z = 1/n_eff`
    /// (the TEM wave impedance `η√(μ/ε_eff)` in `η = 1` units).
    Flat { z: f64 },
}

impl NumericalMode {
    /// Build a numerical mode from a scalar `TE`/`TM` eigenpair. The
    /// transverse field follows from `ψ`: `TM` → `E_t ∝ ∇_t ψ`, `TE` →
    /// `E_t ∝ ẑ × ∇_t ψ`. The per-triangle constant gradient is averaged
    /// (area-weighted) to the nodes for the unified nodal representation.
    pub fn from_scalar(
        mesh: PortMesh2D,
        mode: &PortEigenmode,
        kind: ModeKind,
    ) -> NumericalMode {
        let n_node = mesh.n_nodes();
        let mut acc = vec![[0.0f64; 2]; n_node];
        let mut wsum = vec![0.0f64; n_node];
        for &t in &mesh.tris {
            let (area, g) = mesh.tri_geom(t);
            // ∇_t ψ (constant on the triangle).
            let mut gp = [0.0; 2];
            for k in 0..3 {
                gp[0] += mode.psi[t[k]] * g[k][0];
                gp[1] += mode.psi[t[k]] * g[k][1];
            }
            // E_t direction: TM → ∇ψ, TE → ẑ×∇ψ = (−∂v, ∂u).
            let et = match kind {
                ModeKind::Tm => [gp[0], gp[1]],
                ModeKind::Te => [-gp[1], gp[0]],
            };
            for k in 0..3 {
                acc[t[k]][0] += area * et[0];
                acc[t[k]][1] += area * et[1];
                wsum[t[k]] += area;
            }
        }
        let e_uv_node: Vec<[f64; 2]> = (0..n_node)
            .map(|i| {
                if wsum[i] > 0.0 {
                    [acc[i][0] / wsum[i], acc[i][1] / wsum[i]]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();
        let z_model = match kind {
            ModeKind::Te => ImpedanceModel::Te { k_c: mode.k_c },
            ModeKind::Tm => ImpedanceModel::Tm { k_c: mode.k_c },
        };
        let w_hat = cross3(mesh.u_hat, mesh.v_hat);
        let peak = e_uv_node
            .iter()
            .map(|e| (e[0] * e[0] + e[1] * e[1]).sqrt())
            .fold(0.0_f64, f64::max);
        let inv_peak = if peak > 0.0 { 1.0 / peak } else { 0.0 };
        NumericalMode {
            mesh,
            e_uv_node,
            ned2: None,
            inv_peak,
            w_hat,
            cutoff: mode.k_c,
            z_model,
        }
    }

    /// Build a numerical mode from a full-vector hybrid solve. Stores the
    /// Ned-2 second-kind edge + face coefficients and evaluates the
    /// degree-2 vector field **directly** at query points (no nodal
    /// averaging), so the mode projection in the FD `sparam_waveport`
    /// matches the 3-D `Nedelec2Basis` representation exactly. Zero
    /// cutoff, flat impedance `1/n_eff`.
    pub fn from_vector(mesh: PortMesh2D, mode: &VectorMode) -> NumericalMode {
        let z = if mode.n_eff > 0.0 { 1.0 / mode.n_eff } else { 1.0 };
        let (_n_edge, tri_edges, _use) = build_edges(&mesh);
        let e_edge = mode.e_edge_ned2.clone();
        let e_face = mode.e_face_ned2.clone();
        let e_z_node = mode.e_z_node.clone();
        let e_z_edge = mode.e_z_edge.clone();
        // Unit-peak normalisation: sample |E_t| at every triangle's three
        // vertices via the full Ned-2 evaluation and take the max. Two
        // edges of the triangle meet at each corner so this catches the
        // dominant edge-driven peaks; bubble (face) DOFs vanish there but
        // also produce no extra peak above the nodal sum.
        let mut peak = 0.0_f64;
        for (ti, &t) in mesh.tris.iter().enumerate() {
            let (_a, g) = mesh.tri_geom(t);
            for vloc in 0..3 {
                let mut l = [0.0; 3];
                l[vloc] = 1.0;
                let e = ned2_et_at(
                    &tri_edges[ti], &g, &e_edge, &e_face[ti], ti, l,
                );
                peak = peak.max((e[0] * e[0] + e[1] * e[1]).sqrt());
            }
        }
        let inv_peak = if peak > 0.0 { 1.0 / peak } else { 0.0 };
        let w_hat = cross3(mesh.u_hat, mesh.v_hat);
        NumericalMode {
            mesh,
            e_uv_node: Vec::new(),
            ned2: Some(Ned2ModeData {
                e_edge, e_face, tri_edges, e_z_node, e_z_edge,
            }),
            inv_peak,
            w_hat,
            cutoff: 0.0,
            z_model: ImpedanceModel::Flat { z },
        }
    }

    /// Cutoff angular frequency (`= k_c`; `0` for a quasi-TEM mode).
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Modal wave impedance `Z(omega)` in operator units (`η = 1`).
    pub fn te_impedance(&self, omega: f64) -> f64 {
        match self.z_model {
            ImpedanceModel::Flat { z } => z,
            ImpedanceModel::Te { k_c } => {
                let s = (1.0 - (k_c / omega).powi(2)).max(0.0).sqrt();
                if s > 0.0 { 1.0 / s } else { f64::INFINITY }
            }
            ImpedanceModel::Tm { k_c } => {
                (1.0 - (k_c / omega).powi(2)).max(0.0).sqrt()
            }
        }
    }

    /// Locate the triangle containing the in-plane point `(u, v)` and
    /// return its index plus the barycentric coordinates there. Falls back
    /// to the nearest triangle if the point sits just outside the mesh.
    fn locate(&self, uv: [f64; 2]) -> (usize, [f64; 3]) {
        let mut best = (0usize, [1.0, 0.0, 0.0]);
        let mut best_slack = f64::NEG_INFINITY;
        for (ti, &t) in self.mesh.tris.iter().enumerate() {
            let a = self.mesh.nodes[t[0]];
            let b = self.mesh.nodes[t[1]];
            let c = self.mesh.nodes[t[2]];
            let d =
                (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
            if d.abs() < 1e-30 {
                continue;
            }
            let l0 = ((b[1] - c[1]) * (uv[0] - c[0])
                + (c[0] - b[0]) * (uv[1] - c[1]))
                / d;
            let l1 = ((c[1] - a[1]) * (uv[0] - c[0])
                + (a[0] - c[0]) * (uv[1] - c[1]))
                / d;
            let l2 = 1.0 - l0 - l1;
            let slack = l0.min(l1).min(l2);
            if slack >= -1e-9 {
                return (ti, [l0, l1, l2]);
            }
            if slack > best_slack {
                best_slack = slack;
                best = (ti, [l0, l1, l2]);
            }
        }
        best
    }

    /// In-plane `(u, v)` coordinates of a global point on the port face.
    fn to_uv(&self, x: [f64; 3]) -> [f64; 2] {
        let o = self.mesh.origin;
        let d = [x[0] - o[0], x[1] - o[1], x[2] - o[2]];
        [dot3(d, self.mesh.u_hat), dot3(d, self.mesh.v_hat)]
    }

    /// Transverse electric-field profile at a global point on the port
    /// face, in global coordinates, unit-peak normalised. The vector path
    /// evaluates the Whitney edge field directly (sharp); the scalar path
    /// barycentric-interpolates the nodal `E_t`.
    pub fn e_profile(&self, x: [f64; 3]) -> [f64; 3] {
        let uv = self.to_uv(x);
        let (ti, l) = self.locate(uv);
        let mut ez = 0.0f64;
        let e: [f64; 2] = match &self.ned2 {
            Some(nd) => {
                let t = self.mesh.tris[ti];
                let (_a, g) = self.mesh.tri_geom(t);
                // Longitudinal E_z: P2 over the 3 vertices + 3 edge midpoints.
                let te = &nd.tri_edges[ti];
                for v in 0..3 {
                    ez += nd.e_z_node[t[v]] * p2_basis(v, l);
                }
                for el in 0..3 {
                    ez += nd.e_z_edge[te.gidx[el]] * p2_basis(3 + el, l);
                }
                ned2_et_at(te, &g, &nd.e_edge, &nd.e_face[ti], ti, l)
            }
            None => {
                let t = self.mesh.tris[ti];
                let mut e = [0.0f64; 2];
                for k in 0..3 {
                    e[0] += l[k] * self.e_uv_node[t[k]][0];
                    e[1] += l[k] * self.e_uv_node[t[k]][1];
                }
                e
            }
        };
        let (eu, ev, ew) = (e[0] * self.inv_peak, e[1] * self.inv_peak,
                            ez * self.inv_peak);
        [
            eu * self.mesh.u_hat[0] + ev * self.mesh.v_hat[0] + ew * self.w_hat[0],
            eu * self.mesh.u_hat[1] + ev * self.mesh.v_hat[1] + ew * self.w_hat[1],
            eu * self.mesh.u_hat[2] + ev * self.mesh.v_hat[2] + ew * self.w_hat[2],
        ]
    }

    /// Transverse magnetic-field profile `h_t = ŵ × e_t` at a global
    /// point, the inward-propagating partner of `e_t`. Global coords.
    pub fn h_profile(&self, x: [f64; 3]) -> [f64; 3] {
        cross3(self.w_hat, self.e_profile(x))
    }
}

/// One full-vector hybrid mode of an (optionally inhomogeneous) cross
/// section, solved at a fixed operating wavenumber `k0`.
#[derive(Clone, Debug)]
pub struct VectorMode {
    /// Effective index `n_eff = β / k0` (`n_eff² = λ`, the eigenvalue).
    pub n_eff: f64,
    /// Operating free-space wavenumber the solve was run at.
    pub k0: f64,
    /// Transverse electric field `E_t` at each cross-section node, in the
    /// port-plane `(u, v)` components, recovered (area-averaged) from the
    /// edge-element solution. Convenience for inspection; the sharp profile
    /// uses the Ned-2 / face / P2 coefficient arrays directly.
    pub e_uv_node: Vec<[f64; 2]>,
    /// Nédélec-2 second-kind edge solution, 2 coefficients per global
    /// cross-section edge `[mode0, mode1]`. Mode 0 is "λ_a-weighted", mode
    /// 1 is "λ_b-weighted" where (a, b) is the edge's canonical endpoint
    /// ordering. Zero on PEC-constrained edges.
    pub e_edge_ned2: Vec<[f64; 2]>,
    /// Nédélec-2 face (bubble) coefficients, 2 per triangle. These are
    /// interior degrees of freedom not shared with neighbours.
    pub e_face_ned2: Vec<[f64; 2]>,
    /// Longitudinal field `E_z` coefficients on the P2 vertex DOFs (one
    /// per global cross-section node). Zero on PEC nodes.
    pub e_z_node: Vec<f64>,
    /// Longitudinal `E_z` coefficients on the P2 edge-midpoint DOFs (one
    /// per global edge). Zero on PEC edges.
    pub e_z_edge: Vec<f64>,
}

/// Per-triangle edge data for the Whitney (Nédélec) assembly: the global
/// edge index, orientation sign and length, for each of the triangle's
/// three edges (local edge `e` joins local nodes `((e+1)%3, (e+2)%3)`).
#[derive(Clone, Debug)]
struct TriEdges {
    gidx: [usize; 3],
    sign: [f64; 3],
    len: [f64; 3],
}

// =====================================================================
// Nédélec-2 (second-kind) 2-D basis on a triangle, basis evaluators.
//
// Per-triangle local DOF layout (8 transverse + 6 P2-nodal for E_z = 14):
//   [0..6]   = edge DOFs (3 edges x 2 modes per edge)
//              local index 2*e + m where e in 0..3, m in 0..2
//              edge e joins local vertices ((e+1)%3, (e+2)%3); mode 0 is
//              "λ_a-weighted", mode 1 is "λ_b-weighted".
//   [6..8]   = face DOFs (2 interior bubble modes; per-triangle, not shared)
//   [8..11]  = P2 vertex DOFs for E_z (one per local vertex 0,1,2)
//   [11..14] = P2 edge-midpoint DOFs for E_z (one per local edge 0,1,2)
//
// Edge sign and length come from `TriEdges`. Face DOFs are private to the
// triangle (no orientation/sign tracking needed). P2 vertex DOFs alias the
// PortMesh2D node index; P2 edge midpoints alias the global edge index.
//
// The Ned-2 transverse basis functions are degree-2 vector polynomials:
//   edge mode 0:   φ_e^(0)(l) = sign·len · l[a] · W_e(l)
//   edge mode 1:   φ_e^(1)(l) = sign·len · l[b] · W_e(l)
//   face mode 0:   φ_f^(0)(l) = |edge_1| · l[1] · W_(edge_1)(l)
//   face mode 1:   φ_f^(1)(l) = |edge_2| · l[2] · W_(edge_2)(l)
// where W_e(l) = l[a] · g[b] - l[b] · g[a] is the unsigned Whitney basis
// for local edge e (endpoints a, b), g[k] = ∇λ_k.
//
// The curl in 2D is the scalar z-component, ∇×φ_z = ∂φ_y/∂x − ∂φ_x/∂y:
//   ∇×W_e (z)            = 2·c_ab               (constant, c_ab = (g[a]×g[b])_z)
//   ∇×(λ_a · W_e) (z)   = 3·λ_a · c_ab          (linear in λ_a)
//   ∇×(λ_b · W_e) (z)   = 3·λ_b · c_ab          (linear in λ_b)
// =====================================================================

/// 2-D wedge (z-component of the 3-D cross product) of two in-plane
/// vectors `a` and `b`, i.e. `a_x · b_y − a_y · b_x`.
#[inline]
fn wedge2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// Evaluate the Ned-2 second-kind transverse edge basis function for one of
/// the triangle's 6 edge DOFs at barycentric coordinates `l`. For edge `e`
/// with local
/// endpoints `a=(e+1)%3`, `b=(e+2)%3` and Whitney field
/// `W = λ_a∇λ_b − λ_b∇λ_a`,
///   mode 0 (`ne1`) = W,                 (orientation-odd → carries `sign`)
///   mode 1 (`ne2`) = (λ_a − λ_b)·W.     (orientation-even → no `sign`)
/// `span{W, (λ_a−λ_b)W}` is the genuine second-kind Nédélec edge space, NOT
/// `span{λ_a W, λ_b W}`, which is a different (wrong) space.
#[inline]
fn ned2_edge_basis(
    e: usize,
    mode: usize,
    te: &TriEdges,
    g: &[[f64; 2]; 3],
    l: [f64; 3],
) -> [f64; 2] {
    let a = (e + 1) % 3;
    let b = (e + 2) % 3;
    let w = [l[a] * g[b][0] - l[b] * g[a][0], l[a] * g[b][1] - l[b] * g[a][1]];
    // mode 0: W flips sign under edge reversal → apply orientation sign.
    // mode 1: (λ_a−λ_b)·W is reversal-invariant → no sign.
    let s = if mode == 0 { te.sign[e] } else { l[a] - l[b] };
    [s * w[0], s * w[1]]
}

/// Scalar curl (z-component) of the Ned-2 edge basis function. `curl(W) =
/// 2·c_ab` (constant) and `curl((λ_a−λ_b)W) = 3·(λ_a−λ_b)·c_ab` (linear),
/// where `c_ab = (∇λ_a × ∇λ_b)_z`.
#[inline]
fn ned2_edge_curl(
    e: usize,
    mode: usize,
    te: &TriEdges,
    g: &[[f64; 2]; 3],
    l: [f64; 3],
) -> f64 {
    let a = (e + 1) % 3;
    let b = (e + 2) % 3;
    let c_ab = wedge2(g[a], g[b]);
    if mode == 0 {
        te.sign[e] * 2.0 * c_ab
    } else {
        3.0 * (l[a] - l[b]) * c_ab
    }
}

/// Evaluate one of the 2 interior face (bubble) basis functions
/// (using all three vertices in local order):
///   nf1 = −λ_1(λ_2∇λ_0 − λ_0∇λ_2) − λ_0(λ_2∇λ_1 − λ_1∇λ_2),
///   nf2 =  λ_2(λ_0∇λ_1 − λ_1∇λ_0) + λ_1(λ_0∇λ_2 − λ_2∇λ_0).
/// Interior DOFs (per-triangle, not shared) → no orientation sign.
#[inline]
fn ned2_face_basis(
    mode: usize,
    _te: &TriEdges,
    g: &[[f64; 2]; 3],
    l: [f64; 3],
) -> [f64; 2] {
    if mode == 0 {
        // -λ1·(λ2 g0 − λ0 g2) − λ0·(λ2 g1 − λ1 g2)
        [
            -l[1] * (l[2] * g[0][0] - l[0] * g[2][0])
                - l[0] * (l[2] * g[1][0] - l[1] * g[2][0]),
            -l[1] * (l[2] * g[0][1] - l[0] * g[2][1])
                - l[0] * (l[2] * g[1][1] - l[1] * g[2][1]),
        ]
    } else {
        // λ2·(λ0 g1 − λ1 g0) + λ1·(λ0 g2 − λ2 g0)
        [
            l[2] * (l[0] * g[1][0] - l[1] * g[0][0])
                + l[1] * (l[0] * g[2][0] - l[2] * g[0][0]),
            l[2] * (l[0] * g[1][1] - l[1] * g[0][1])
                + l[1] * (l[0] * g[2][1] - l[2] * g[0][1]),
        ]
    }
}

/// Scalar curl (z-component) of the face basis functions, derived from the
/// `nf1`/`nf2` forms above:
///   curl(nf1) = 3·(λ_0·c_12 + λ_1·c_02),
///   curl(nf2) = 3·(λ_1·c_02 + λ_2·c_01),   c_ij = (∇λ_i × ∇λ_j)_z.
#[inline]
fn ned2_face_curl(
    mode: usize,
    _te: &TriEdges,
    g: &[[f64; 2]; 3],
    l: [f64; 3],
) -> f64 {
    let c01 = wedge2(g[0], g[1]);
    let c02 = wedge2(g[0], g[2]);
    let c12 = wedge2(g[1], g[2]);
    if mode == 0 {
        3.0 * (l[0] * c12 + l[1] * c02)
    } else {
        3.0 * (l[1] * c02 + l[2] * c01)
    }
}

// =====================================================================
// P2 Lagrange nodal basis for the longitudinal E_z component.
//
// Per-triangle local DOF layout (6 nodal DOFs):
//   [0..3] = vertex values at local vertices 0, 1, 2:
//              N_v(l) = λ_v · (2λ_v − 1)
//   [3..6] = edge-midpoint values at midpoints of local edges 0, 1, 2:
//              N_e(l) = 4 · λ_a · λ_b   where (a,b) are edge e's endpoints
//
// Gradients are linear in λ, so the curl-like contribution stays in the
// degree-2 mass / degree-1 stiffness range that a 5-point Gauss quadrature
// integrates exactly.
// =====================================================================

/// Evaluate one of the 6 P2 Lagrange basis functions at barycentric `l`.
/// Index 0..3 = vertex DOFs (vertex 0, 1, 2). Index 3..6 = edge-midpoint
/// DOFs (edge 0, 1, 2).
#[inline]
fn p2_basis(dof: usize, l: [f64; 3]) -> f64 {
    if dof < 3 {
        let v = dof;
        l[v] * (2.0 * l[v] - 1.0)
    } else {
        let e = dof - 3;
        let a = (e + 1) % 3;
        let b = (e + 2) % 3;
        4.0 * l[a] * l[b]
    }
}

/// Gradient of the P2 Lagrange basis function with respect to (x, y).
#[inline]
fn p2_grad(dof: usize, g: &[[f64; 2]; 3], l: [f64; 3]) -> [f64; 2] {
    if dof < 3 {
        // ∇(λ_v · (2λ_v − 1)) = (4λ_v − 1) · ∇λ_v
        let v = dof;
        let s = 4.0 * l[v] - 1.0;
        [s * g[v][0], s * g[v][1]]
    } else {
        // ∇(4 λ_a λ_b) = 4 (λ_a · ∇λ_b + λ_b · ∇λ_a)
        let e = dof - 3;
        let a = (e + 1) % 3;
        let b = (e + 2) % 3;
        [
            4.0 * (l[a] * g[b][0] + l[b] * g[a][0]),
            4.0 * (l[a] * g[b][1] + l[b] * g[a][1]),
        ]
    }
}

/// Strang's 7-point Gauss-Dunavant quadrature on the reference triangle,
/// exact for polynomials up to degree 5. Each entry is `(weight, l0, l1, l2)`
/// with weights summing to 1 (multiply integrand by triangle area to get the
/// actual surface integral). Used by the Ned-2 + P2 element-matrix assembly.
const NED2_QPTS_DEG5: [(f64, f64, f64, f64); 7] = [
    (0.225,                  1.0/3.0,            1.0/3.0,            1.0/3.0),
    (0.13239415278850618,    0.05971587178976982, 0.4701420641051151,  0.4701420641051151),
    (0.13239415278850618,    0.4701420641051151,  0.05971587178976982, 0.4701420641051151),
    (0.13239415278850618,    0.4701420641051151,  0.4701420641051151,  0.05971587178976982),
    (0.12593918054482717,    0.7974269853530873,  0.10128650732345633, 0.10128650732345633),
    (0.12593918054482717,    0.10128650732345633, 0.7974269853530873,  0.10128650732345633),
    (0.12593918054482717,    0.10128650732345633, 0.10128650732345633, 0.7974269853530873),
];

/// Enumerate unique mesh edges and, per triangle, the global index,
/// orientation sign and length of its three local edges. Also returns the
/// per-global-edge triangle-use count (1 = boundary edge).
fn build_edges(mesh: &PortMesh2D) -> (usize, Vec<TriEdges>, Vec<u32>) {
    use std::collections::HashMap;
    let mut emap: HashMap<(usize, usize), usize> = HashMap::new();
    let mut use_count: Vec<u32> = Vec::new();
    let mut per_tri: Vec<TriEdges> = Vec::with_capacity(mesh.tris.len());
    for &t in &mesh.tris {
        let mut te =
            TriEdges { gidx: [0; 3], sign: [0.0; 3], len: [0.0; 3] };
        for e in 0..3 {
            let a = t[(e + 1) % 3];
            let b = t[(e + 2) % 3];
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            let gi = *emap.entry((lo, hi)).or_insert_with(|| {
                use_count.push(0);
                use_count.len() - 1
            });
            use_count[gi] += 1;
            te.gidx[e] = gi;
            te.sign[e] = if a == lo { 1.0 } else { -1.0 };
            let pa = mesh.nodes[a];
            let pb = mesh.nodes[b];
            te.len[e] =
                ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2)).sqrt();
        }
        per_tri.push(te);
    }
    (emap.len(), per_tri, use_count)
}

/// Mark the PEC-constrained nodes and edges of a port cross-section.
///
/// A node is constrained on the outer wall or an internal conductor. An edge
/// is constrained if it is an outer-boundary edge (used by a single triangle)
/// or if both endpoints lie on an internal conductor, in which case the edge
/// runs along the trace surface where tangential E must vanish. The internal
/// rule keys on `on_pec` (not the node mask) so two outer-boundary nodes
/// joined by an interior chord are not spuriously constrained.
fn vector_mode_pec_masks(
    mesh: &PortMesh2D,
    tri_edges: &[TriEdges],
    edge_use: &[u32],
) -> (Vec<bool>, Vec<bool>) {
    let n_node = mesh.n_nodes();
    let on_pec = |i: usize| mesh.on_pec.get(i).copied().unwrap_or(false);
    let node_pec: Vec<bool> =
        (0..n_node).map(|i| mesh.on_boundary[i] || on_pec(i)).collect();
    let mut edge_pec = edge_use.iter().map(|&c| c == 1).collect::<Vec<_>>();
    for (ti, &t) in mesh.tris.iter().enumerate() {
        let te = &tri_edges[ti];
        for e in 0..3 {
            let a = t[(e + 1) % 3];
            let b = t[(e + 2) % 3];
            if on_pec(a) && on_pec(b) {
                edge_pec[te.gidx[e]] = true;
            }
        }
    }
    (node_pec, edge_pec)
}

/// Whether the cross-section supports a (quasi-)TEM mode: `true` iff there are
/// at least two electrically-isolated PEC conductors (e.g. a microstrip trace
/// plus its enclosing ground). Found via union-find over PEC nodes joined by
/// PEC edges. A single conductor (hollow guide, or a septum touching the wall)
/// supports no TEM wave, so its curl-free modes are all spurious; with two or
/// more, the genuinely curl-free TEM mode must survive the spurious filter.
fn vector_mode_tem_supported(
    mesh: &PortMesh2D,
    tri_edges: &[TriEdges],
    edge_pec: &[bool],
    node_pec: &[bool],
) -> bool {
    let n_node = mesh.n_nodes();
    let mut uf: Vec<usize> = (0..n_node).collect();
    let find = |uf: &Vec<usize>, mut x: usize| -> usize {
        while uf[x] != x { x = uf[x]; }
        x
    };
    for (ti, &t) in mesh.tris.iter().enumerate() {
        let te = &tri_edges[ti];
        for e in 0..3 {
            if edge_pec[te.gidx[e]] {
                let a = t[(e + 1) % 3];
                let b = t[(e + 2) % 3];
                let ra = find(&uf, a);
                let rb = find(&uf, b);
                if ra != rb { uf[ra] = rb; }
            }
        }
    }
    let mut pec_roots = std::collections::HashSet::new();
    for i in 0..n_node {
        if node_pec[i] { pec_roots.insert(find(&uf, i)); }
    }
    pec_roots.len() >= 2
}

/// Solve the full-vector hybrid mode eigenproblem at a fixed operating
/// wavenumber `k0`, returning up to `n_modes` modes by descending
/// effective index (most-confined first).
///
/// `eps_r` is the per-triangle relative permittivity (length
/// `mesh.tris.len()`), an inhomogeneous fill (substrate + air), exactly
/// what a microstrip-class line needs and what the scalar
/// [`solve_modes`] cannot represent. `μ_r = 1`.
///
/// Mixed Nédélec-edge (`E_t`) / Lagrange-nodal (`E_z`) discretisation,
/// eigenvalue `λ = β²/k0² = n_eff²`. PEC walls (cross-section boundary)
/// impose `tangential E = 0`: boundary edges and nodes are constrained
/// out. The singular generalized problem `A x = λ B x` is solved by
/// shift-invert near `σ` (just above the largest `ε_r`): eigenvalues `ν`
/// of `(A − σB)⁻¹ B`, then `λ = σ + 1/ν`. Dense, sized for one face.
/// Port characteristic length ℓ for the non-dimensionalization wrapper:
/// √(total cross-section area), i.e. the PORT size (not the mesh resolution).
/// This keeps the electrical size κ = k0·ℓ faithful — a µm RFIC port maps to
/// O(1) coordinates with its true (tiny) κ, while an already-O(1) cross-section
/// stays in its natural regime. `n_eff` is ℓ-invariant, so any O(port-size)
/// length is valid; this one preserves the conditioning across scales.
fn port_characteristic_length(mesh: &PortMesh2D) -> f64 {
    let mut area = 0.0_f64;
    for &t in &mesh.tris {
        let (a, _g) = mesh.tri_geom(t);
        area += a;
    }
    area.sqrt()
}

/// Robust quasi-TEM / vector wave-port mode solve.
///
/// Port-local non-dimensionalization (derivations/wave_port/): the 2-D
/// eigenproblem is assembled in a frame scaled by the port's own characteristic
/// length ℓ, so the matrix entries stay O(1) and the conditioning is
/// independent of how small the port is relative to the 3-D mesh (RFIC µm
/// cross-sections otherwise trip the absolute, k0²-scaled tolerances and the
/// ill-conditioned shift-invert LU, returning 0 modes). `n_eff = β/k0` is
/// scale-invariant; the recovered field coefficients are mapped back to the
/// physical frame (transverse Eₜ ∝ ∇L scales as 1/ℓ, longitudinal E_z does not,
/// so Eₜ coefficients are multiplied by ℓ) so downstream profile evaluation on
/// the original mesh stays consistent.
pub fn solve_vector_modes(
    mesh: &PortMesh2D,
    eps_r: &[f64],
    k0: f64,
    n_modes: usize,
) -> Vec<VectorMode> {
    let ell = port_characteristic_length(mesh);
    if !(ell.is_finite() && ell > 0.0) {
        return solve_vector_modes_core(mesh, eps_r, k0, n_modes);
    }
    // Assemble in the port-local O(1) frame: x̃ = x/ℓ, κ = k0·ℓ.
    let scaled = PortMesh2D {
        nodes: mesh.nodes.iter().map(|n| [n[0] / ell, n[1] / ell]).collect(),
        tris: mesh.tris.clone(),
        on_boundary: mesh.on_boundary.clone(),
        on_pec: mesh.on_pec.clone(),
        u_hat: mesh.u_hat,
        v_hat: mesh.v_hat,
        origin: mesh.origin,
    };
    let mut modes = solve_vector_modes_core(&scaled, eps_r, k0 * ell, n_modes);
    for m in modes.iter_mut() {
        // n_eff is ℓ-invariant; report the physical k0. Map Eₜ coefficients
        // back to the physical frame (×ℓ); E_z and the eigenvalue are unchanged.
        m.k0 = k0;
        for c in m.e_edge_ned2.iter_mut() { c[0] *= ell; c[1] *= ell; }
        for c in m.e_face_ned2.iter_mut() { c[0] *= ell; c[1] *= ell; }
        for c in m.e_uv_node.iter_mut() { c[0] *= ell; c[1] *= ell; }
    }
    modes
}

fn solve_vector_modes_core(
    mesh: &PortMesh2D,
    eps_r: &[f64],
    k0: f64,
    n_modes: usize,
) -> Vec<VectorMode> {
    use faer::Mat;
    let n_node = mesh.n_nodes();
    let (n_edge, tri_edges, edge_use) = build_edges(mesh);

    // PEC masks (outer wall + internal conductors) and whether the section
    // carries a (quasi-)TEM mode that the spurious filter must spare.
    let (node_pec, edge_pec) = vector_mode_pec_masks(mesh, &tri_edges, &edge_use);
    let tem_supported =
        vector_mode_tem_supported(mesh, &tri_edges, &edge_pec, &node_pec);

    // DOF numbering for the Ned-2 + P2 hybrid eigenproblem.
    //
    // Block 1, transverse Et (Ned-2 second-kind, degree 2):
    //   2 coefs per cross-section edge (PEC-constrained edges drop both)
    //   2 coefs per triangle (interior face bubbles, never PEC)
    // Block 2, longitudinal Ez (P2 Lagrange):
    //   1 coef per cross-section node (PEC nodes drop)
    //   1 coef per edge midpoint (PEC edges drop)
    //
    // Reduced numbering compresses the free DOFs into a contiguous range
    // 0..ndof; constrained DOFs map to `usize::MAX`.

    let edge_free: Vec<usize> =
        (0..n_edge).filter(|&e| !edge_pec[e]).collect();
    let node_free: Vec<usize> =
        (0..n_node).filter(|&i| !node_pec[i]).collect();
    let ne = edge_free.len();
    let nn = node_free.len();
    let n_tri = mesh.tris.len();
    // Block sizes in the reduced system.
    let n_et_edge = 2 * ne;             // Et edge DOFs
    let n_et_face = 2 * n_tri;          // Et face bubble DOFs
    let n_ez_node = nn;                 // Ez vertex DOFs
    let n_ez_edge = ne;                 // Ez edge-midpoint DOFs
    let n_et = n_et_edge + n_et_face;
    let n_ez = n_ez_node + n_ez_edge;
    let ndof = n_et + n_ez;
    if ndof == 0 {
        return Vec::new();
    }
    // Offset helpers for the reduced index of each DOF kind.
    let off_et_edge = 0usize;
    let off_et_face = off_et_edge + n_et_edge;
    let off_ez_node = off_et_face + n_et_face;
    let off_ez_edge = off_ez_node + n_ez_node;
    let mut edge_red = vec![usize::MAX; n_edge];
    for (r, &g) in edge_free.iter().enumerate() {
        edge_red[g] = r;
    }
    let mut node_red = vec![usize::MAX; n_node];
    for (r, &g) in node_free.iter().enumerate() {
        node_red[g] = r;
    }

    // Per-triangle 14-DOF local layout (matches the basis-evaluator comment
    // above): [0..6]=edge·mode, [6..8]=face·mode, [8..11]=P2 vertex,
    // [11..14]=P2 edge-midpoint. The reduced global DOF index for each local
    // DOF is resolved inline per element in the assembly loop below.

    // Sparse assembly: accumulate COO triplets (faer sums duplicate (i,j)).
    // The port face can carry tens of thousands of DOFs at production
    // refinement, where a dense ndof×ndof factorisation is intractable.
    let mut a_trip: Vec<(usize, usize, f64)> = Vec::new();
    let mut b_trip: Vec<(usize, usize, f64)> = Vec::new();
    // Symmetric eigenproblem A·x = λ·B·x.
    //
    //   A = [ Att − k0²·Btt          0         ]
    //       [        0                0         ]
    //
    //   B = [ Dtt              Dzt^T            ]
    //       [ Dzt        Dzz1 − k0²·Dzz2       ]
    //
    // with
    //   Att[i,j] = ∫ (∇×φ_i)(∇×φ_j) dA              (curl-curl, μ=1)
    //   Btt[i,j] = ∫ ε_r · φ_i · φ_j dA              (ε-weighted Et mass)
    //   Dtt[i,j] = ∫ φ_i · φ_j dA                    (Et plain mass)
    //   Dzt[i,j] = ∫ ∇N_i · φ_j dA                   (Et-Ez coupling)
    //   Dzz1[i,j]= ∫ ∇N_i · ∇N_j dA                  (P2 Laplacian)
    //   Dzz2[i,j]= ∫ ε_r · N_i · N_j dA              (P2 mass)
    //
    // The block-diagonal A naturally pushes the discrete gradient null space
    // (Et = ∇φ in P2) to λ = −k0² < 0; the `neff² > 0` filter then rejects
    // those modes automatically. The earlier "asymmetric A with Ez-Ez
    // ε·k0² block" formulation had the same gradient modes at λ ≈ ε·k0²
    // and polluted the spectrum at the upper end of the physical range.

    for (ti, &t) in mesh.tris.iter().enumerate() {
        let (area, g) = mesh.tri_geom(t);
        let er = eps_r[ti];
        let te = &tri_edges[ti];

        // Resolve all 14 local DOFs once per element.
        let mut dofs = [usize::MAX; 14];
        for k in 0..6 {
            // edges
            let e_loc = k / 2;
            let m = k % 2;
            let r = edge_red[te.gidx[e_loc]];
            dofs[k] = if r == usize::MAX { usize::MAX } else { off_et_edge + 2 * r + m };
        }
        for m in 0..2 {
            dofs[6 + m] = off_et_face + 2 * ti + m;
        }
        for v_loc in 0..3 {
            let r = node_red[t[v_loc]];
            dofs[8 + v_loc] = if r == usize::MAX { usize::MAX } else { off_ez_node + r };
        }
        for e_loc in 0..3 {
            let r = edge_red[te.gidx[e_loc]];
            dofs[11 + e_loc] = if r == usize::MAX { usize::MAX } else { off_ez_edge + r };
        }

        // Precompute basis function values at all 7 quadrature points.
        let nqp = NED2_QPTS_DEG5.len();
        let mut tan_qp = vec![[[0.0f64; 2]; 8]; nqp];
        let mut curl_qp = vec![[0.0f64; 8]; nqp];
        let mut p2_qp = vec![[0.0f64; 6]; nqp];
        let mut p2grad_qp = vec![[[0.0f64; 2]; 6]; nqp];
        for (qi, &(_w, l1, l2, l3)) in NED2_QPTS_DEG5.iter().enumerate() {
            let l = [l1, l2, l3];
            // Et basis: 6 edge + 2 face = 8 functions.
            for e in 0..3 {
                tan_qp[qi][2 * e]     = ned2_edge_basis(e, 0, te, &g, l);
                tan_qp[qi][2 * e + 1] = ned2_edge_basis(e, 1, te, &g, l);
                curl_qp[qi][2 * e]     = ned2_edge_curl(e, 0, te, &g, l);
                curl_qp[qi][2 * e + 1] = ned2_edge_curl(e, 1, te, &g, l);
            }
            tan_qp[qi][6] = ned2_face_basis(0, te, &g, l);
            tan_qp[qi][7] = ned2_face_basis(1, te, &g, l);
            curl_qp[qi][6] = ned2_face_curl(0, te, &g, l);
            curl_qp[qi][7] = ned2_face_curl(1, te, &g, l);
            // E_z basis: 6 P2 functions.
            for k in 0..6 {
                p2_qp[qi][k]     = p2_basis(k, l);
                p2grad_qp[qi][k] = p2_grad(k, &g, l);
            }
        }

        // Et-Et block: A[di,dj] += Att − k0²·Btt,  B[di,dj] += Dtt.
        for i in 0..8 {
            let di = dofs[i];
            if di == usize::MAX { continue; }
            for j in 0..8 {
                let dj = dofs[j];
                if dj == usize::MAX { continue; }
                let mut mass = 0.0;
                let mut stiff = 0.0;
                for (qi, &(w, _l1, _l2, _l3)) in NED2_QPTS_DEG5.iter().enumerate() {
                    let wi = tan_qp[qi][i];
                    let wj = tan_qp[qi][j];
                    mass  += w * (wi[0] * wj[0] + wi[1] * wj[1]);
                    stiff += w * curl_qp[qi][i] * curl_qp[qi][j];
                }
                mass  *= area;
                stiff *= area;
                // Sign convention: A[:8,:8] = Att − k0²·Btt, assembled
                // WITHOUT negation. The
                // generalized eigenvalue is then λ = −β² (the propagating band
                // is λ < 0); β is recovered as √(−λ) in the solve below. The
                // discrete gradient null space lands at λ ≈ 0 and is rejected
                // by the |λ| = β² floor, NOT by an ad-hoc A negation.
                a_trip.push((di, dj, stiff - er * k0 * k0 * mass)); // Att − k0²·Btt
                b_trip.push((di, dj, mass));                         // Dtt
            }
        }

        // Et-Ez cross blocks in B (symmetric):
        //   B[Et_i, Ez_j] = Dzt^T[i,j] = ∫ φ_i · ∇N_j dA
        //   B[Ez_i, Et_j] = Dzt[i,j]   = ∫ ∇N_i · φ_j dA
        // (These are the same scalar integral, written in transposed slots.)
        for i in 0..8 {
            let di_et = dofs[i];
            if di_et == usize::MAX { continue; }
            for jn in 0..6 {
                let dj_ez = dofs[8 + jn];
                if dj_ez == usize::MAX { continue; }
                let mut val = 0.0;
                for (qi, &(w, _l1, _l2, _l3)) in NED2_QPTS_DEG5.iter().enumerate() {
                    let wi = tan_qp[qi][i];
                    let pg = p2grad_qp[qi][jn];
                    val += w * (wi[0] * pg[0] + wi[1] * pg[1]);
                }
                val *= area;
                b_trip.push((di_et, dj_ez, val));
                b_trip.push((dj_ez, di_et, val));
            }
        }

        // Ez-Ez block: B[di,dj] += Dzz1 − k0²·Dzz2 with
        //   Dzz1 = ∫ ∇N_i · ∇N_j dA           (P2 Laplacian)
        //   Dzz2 = ∫ ε_r · N_i · N_j dA       (P2 mass with ε_r)
        for i in 0..6 {
            let di = dofs[8 + i];
            if di == usize::MAX { continue; }
            for j in 0..6 {
                let dj = dofs[8 + j];
                if dj == usize::MAX { continue; }
                let mut grad_ij = 0.0;
                let mut mass_ij = 0.0;
                for (qi, &(w, _l1, _l2, _l3)) in NED2_QPTS_DEG5.iter().enumerate() {
                    let gi = p2grad_qp[qi][i];
                    let gj = p2grad_qp[qi][j];
                    grad_ij += w * (gi[0] * gj[0] + gi[1] * gj[1]);
                    mass_ij += w * p2_qp[qi][i] * p2_qp[qi][j];
                }
                grad_ij *= area;
                mass_ij *= area;
                b_trip.push((di, dj, grad_ij - er * k0 * k0 * mass_ij));
            }
        }
    }

    // ============================================================== SOLVE
    // Convention: the generalized eigenvalue is λ = −β². The physical
    // propagating band is λ ∈ [−ε_max·k0², 0); n_eff² = β²/k0² = −λ/k0², with
    // 0 < n_eff² ≤ ε_max for a confined mode.
    //
    // The Ned-2 + P2 basis carries a large discrete gradient null space whose
    // curl-free modes spread across n_eff² ∈ [ε_min, ε_max] (in a homogeneous
    // region they pile into one degenerate cluster at exactly ε_max). A DENSE
    // all-at-once eigensolve cannot separate that cluster from the genuine
    // guided mode, the eigenvectors mix. A shift-invert ITERATIVE solve
    // (scipy `eigs`) that resolves only the few isolated modes nearest a
    // shift σ keeps the cluster to at most one representative, so the genuine
    // mode converges cleanly. We achieve the same with a dense
    // shift-invert ARNOLDI (Krylov dim ≪ ndof) on M = (A − σB)⁻¹B, swept over
    // σ across the band, then keep the curl-bearing modes (k_t² above floor)
    // and return the fundamental (largest n_eff), the curl-free spurious /
    // gradient modes are rejected by the transverse curl energy
    //   k_t² = ∫|∇×E_t|² dA / ∫|E_t|² dA.
    let eps_max = eps_r.iter().cloned().fold(1.0_f64, f64::max);
    let eps_min = eps_r.iter().cloned().fold(f64::INFINITY, f64::min);
    let k0sq = k0 * k0;
    let beta2_floor = 1e-3 * k0sq;   // reject β ≈ 0 static modes
    let kt2_floor = 1e-3 * k0sq;     // reject curl-free spurious / gradient modes
    // A pure-TEM line mode is curl-free, sits at n_eff² = ε (= ε_max), and is
    // physical only when the fill is homogeneous AND ≥ 2 conductors support a
    // TEM wave (a stripline / coax). In that single case the curl-free mode is
    // the fundamental and must be kept; everywhere else a curl-free mode is a
    // spurious plane-wave / gradient mode. (An inhomogeneous quasi-TEM line,
    // microstrip, has a slightly curl-bearing fundamental BELOW ε_max, so the
    // curl filter still applies and correctly rejects the ε_max spurious.)
    let homogeneous = (eps_max - eps_min) < 1e-9 * eps_max;
    let allow_curl_free = tem_supported && homogeneous;
    let debug = std::env::var("PORT_EIGEN_DEBUG").is_ok();

    // Sparse matvec y = B·v straight from the COO triplets (faer would also
    // do this, but iterating the triplets avoids materialising B separately).
    let b_matvec = |v: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; ndof];
        for &(i, j, val) in &b_trip {
            y[i] += val * v[j];
        }
        y
    };

    // Transverse curl energy ratio k_t² = ∫|∇×E_t|² / ∫|E_t|² for a reduced
    // eigenvector `x`, evaluated directly from its Ned-2 E_t coefficients.
    let curl_kt2 = |x: &[f64]| -> f64 {
        let mut curl2 = 0.0;
        let mut etmass = 0.0;
        for (ti, &t) in mesh.tris.iter().enumerate() {
            let (area, g) = mesh.tri_geom(t);
            let te = &tri_edges[ti];
            let mut coef = [0.0f64; 8];
            for kk in 0..6 {
                let e_loc = kk / 2;
                let m = kk % 2;
                let r = edge_red[te.gidx[e_loc]];
                if r != usize::MAX {
                    coef[kk] = x[off_et_edge + 2 * r + m];
                }
            }
            for m in 0..2 {
                coef[6 + m] = x[off_et_face + 2 * ti + m];
            }
            for &(w, l1, l2, l3) in NED2_QPTS_DEG5.iter() {
                let l = [l1, l2, l3];
                let mut et = [0.0f64; 2];
                let mut cz = 0.0f64;
                for kk in 0..6 {
                    let e_loc = kk / 2;
                    let m = kk % 2;
                    let phi = ned2_edge_basis(e_loc, m, te, &g, l);
                    et[0] += coef[kk] * phi[0];
                    et[1] += coef[kk] * phi[1];
                    cz += coef[kk] * ned2_edge_curl(e_loc, m, te, &g, l);
                }
                for m in 0..2 {
                    let phi = ned2_face_basis(m, te, &g, l);
                    et[0] += coef[6 + m] * phi[0];
                    et[1] += coef[6 + m] * phi[1];
                    cz += coef[6 + m] * ned2_face_curl(m, te, &g, l);
                }
                curl2 += w * area * cz * cz;
                etmass += w * area * (et[0] * et[0] + et[1] * et[1]);
            }
        }
        if etmass > 0.0 { curl2 / etmass } else { 0.0 }
    };

    // Sparse shift-invert Arnoldi at shift σ: build a Krylov subspace of
    // M = (A − σB)⁻¹B (full Gram-Schmidt → upper-Hessenberg H = Vᵀ M V),
    // eigendecompose the small H, and return the real Ritz pairs as
    // (λ = σ + 1/μ, reduced eigenvector). Resolves the modes nearest σ. The
    // shift-invert linear solves go through faer's pure-Rust sparse LU, so the
    // cost scales with the (sparse) factorisation, not ndof³.
    let m_kry = ndof.min(80);
    let arnoldi = |sigma: f64| -> Vec<(f64, Vec<f64>)> {
        use faer::sparse::{SparseColMat, Triplet};
        use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
        use faer::linalg::solvers::SolveCore;
        // C = A − σB as sparse triplets (union pattern of A and B).
        let trips: Vec<Triplet<usize, usize, f64>> = a_trip
            .iter()
            .map(|&(r, c, v)| Triplet { row: r, col: c, val: v })
            .chain(b_trip.iter().map(|&(r, c, v)| Triplet { row: r, col: c, val: -sigma * v }))
            .collect();
        let cmat = match SparseColMat::<usize, f64>::try_new_from_triplets(ndof, ndof, &trips) {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let sym = match SymbolicLu::try_new(cmat.as_ref().symbolic()) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let lu = match Lu::try_new_with_symbolic(sym, cmat.as_ref()) {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };
        // x ← C⁻¹ x (in place) for a length-ndof RHS.
        let solve = |bv: &[f64]| -> Vec<f64> {
            let mut x = Mat::<f64>::from_fn(ndof, 1, |i, _| bv[i]);
            lu.solve_in_place_with_conj(faer::Conj::No, x.as_mut());
            (0..ndof).map(|i| x[(i, 0)]).collect()
        };
        // Deterministic start vector (no RNG: keeps resume/CI reproducible).
        let mut v0: Vec<f64> =
            (0..ndof).map(|i| (((i * 7 + 13) % 97) as f64 / 97.0) - 0.5).collect();
        let n0 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n0 == 0.0 { return Vec::new(); }
        for x in &mut v0 { *x /= n0; }
        let mut vs: Vec<Vec<f64>> = vec![v0];
        let mut hmat = vec![vec![0.0f64; m_kry]; m_kry + 1];
        let mut m_act = m_kry;
        for j in 0..m_kry {
            let bv = b_matvec(&vs[j]);
            let mut w: Vec<f64> = solve(&bv);
            // Full reorthogonalisation against all previous Arnoldi vectors.
            for i in 0..=j {
                let hij: f64 =
                    vs[i].iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                hmat[i][j] = hij;
                for k in 0..ndof { w[k] -= hij * vs[i][k]; }
            }
            let wn = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            hmat[j + 1][j] = wn;
            if wn < 1e-12 { m_act = j + 1; break; }
            for x in &mut w { *x /= wn; }
            vs.push(w);
        }
        // Final Arnoldi residual norm β = h_{m+1,m}: the Ritz pair (μ_k, y_k)
        // has residual ‖M x_k − μ_k x_k‖ = β·|eₘᵀ y_k| (last component of the
        // Hessenberg eigenvector). Only well-converged Ritz pairs are trusted;
        // unconverged ones produce unphysical λ (e.g. n_eff² > ε_max) and must
        // be discarded. β = 0 means an invariant subspace (all converged).
        let beta_resid = if m_act >= 1 { hmat[m_act][m_act - 1] } else { 0.0 };
        let hm = Mat::<f64>::from_fn(m_act, m_act, |i, j| hmat[i][j]);
        let eig = match hm.eigen() { Ok(e) => e, Err(_) => return Vec::new() };
        let evals = eig.S().column_vector();
        let evecs = eig.U();
        let mut out = Vec::new();
        for k in 0..m_act {
            let mu_re = evals[k].re;
            let mu_im = evals[k].im;
            if mu_im.abs() > 1e-6 * (mu_re.abs() + 1e-30) { continue; }
            if mu_re.abs() < 1e-30 { continue; }
            // Ritz convergence: reject pairs whose residual is not ≪ |μ|.
            let y_last = (evecs[(m_act - 1, k)].re.powi(2)
                + evecs[(m_act - 1, k)].im.powi(2)).sqrt();
            let resid = beta_resid * y_last;
            let lambda = sigma + 1.0 / mu_re;
            if resid > 1e-4 * mu_re.abs() { continue; }
            // Ritz vector x = V · y_k (real part).
            let mut x = vec![0.0f64; ndof];
            for jj in 0..m_act {
                let yk = evecs[(jj, k)].re;
                let vj = &vs[jj];
                for i in 0..ndof { x[i] += yk * vj[i]; }
            }
            out.push((lambda, x));
        }
        out
    };

    // Sweep σ across the band (targets in n_eff², kept clear of the ε_max
    // spurious cluster) and accumulate the distinct genuine (curl-bearing)
    // modes. One σ near the fundamental resolves it cleanly; the sweep makes
    // the search robust to where the mode actually sits.
    let mut genuine: Vec<(f64, Vec<f64>)> = Vec::new();
    let sweep_fracs = [0.90, 0.80, 0.68, 0.56, 0.45, 0.35, 0.27];
    for &frac in sweep_fracs.iter() {
        let neff2_t = (frac * eps_max).max(0.05);
        let sigma = -neff2_t * k0sq;
        for (lambda, x) in arnoldi(sigma) {
            let beta2 = -lambda;
            let neff2 = beta2 / k0sq;
            if beta2 <= beta2_floor || neff2 > eps_max + 1e-3 { continue; }
            let kt2 = curl_kt2(&x);
            if debug {
                eprintln!(
                    "  σ_neff²={:.3}  β²={:.4}  n_eff²={:.4}  k_t²={:.4e}",
                    neff2_t, beta2, neff2, kt2
                );
            }
            // Reject curl-free spurious / gradient modes, UNLESS this is a
            // homogeneous multi-conductor TEM line, whose curl-free
            // fundamental at n_eff² = ε_max is physical.
            if !allow_curl_free && kt2 < kt2_floor { continue; }
            // De-duplicate against modes already found (relative n_eff²).
            if genuine.iter().any(|(n, _)| (n - neff2).abs() < 1e-3 * neff2.max(1.0)) {
                continue;
            }
            genuine.push((neff2, x));
        }
    }
    genuine.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    genuine
        .into_iter()
        .take(n_modes)
        .map(|(neff2, x)| {
            // Scatter the reduced eigenvector into per-DOF-kind arrays.
            // Constrained DOFs stay at zero (PEC condition).
            let mut e_edge_ned2 = vec![[0.0f64; 2]; n_edge];
            for (gi, &r) in edge_red.iter().enumerate() {
                if r != usize::MAX {
                    e_edge_ned2[gi][0] = x[off_et_edge + 2 * r];
                    e_edge_ned2[gi][1] = x[off_et_edge + 2 * r + 1];
                }
            }
            let mut e_face_ned2 = vec![[0.0f64; 2]; n_tri];
            for ti in 0..n_tri {
                e_face_ned2[ti][0] = x[off_et_face + 2 * ti];
                e_face_ned2[ti][1] = x[off_et_face + 2 * ti + 1];
            }
            let mut e_z_node = vec![0.0f64; n_node];
            for (gi, &r) in node_red.iter().enumerate() {
                if r != usize::MAX {
                    e_z_node[gi] = x[off_ez_node + r];
                }
            }
            let mut e_z_edge = vec![0.0f64; n_edge];
            for (gi, &r) in edge_red.iter().enumerate() {
                if r != usize::MAX {
                    e_z_edge[gi] = x[off_ez_edge + r];
                }
            }

            // Convenience nodal Et profile (area-weighted average of the
            // Ned-2 evaluation at each triangle's corners), kept for the
            // diagnostics/inspection path used by some tests.
            let mut acc = vec![[0.0f64; 2]; n_node];
            let mut wsum = vec![0.0f64; n_node];
            for (ti, &t) in mesh.tris.iter().enumerate() {
                let (area, g) = mesh.tri_geom(t);
                let te = &tri_edges[ti];
                for vloc in 0..3 {
                    let mut l = [0.0; 3];
                    l[vloc] = 1.0;
                    let et = ned2_et_at(
                        te, &g, &e_edge_ned2, &e_face_ned2[ti], ti, l,
                    );
                    acc[t[vloc]][0] += area * et[0];
                    acc[t[vloc]][1] += area * et[1];
                    wsum[t[vloc]] += area;
                }
            }
            let e_uv_node: Vec<[f64; 2]> = (0..n_node)
                .map(|i| {
                    if wsum[i] > 0.0 {
                        [acc[i][0] / wsum[i], acc[i][1] / wsum[i]]
                    } else {
                        [0.0, 0.0]
                    }
                })
                .collect();

            VectorMode {
                n_eff: neff2.sqrt(),
                k0,
                e_uv_node,
                e_edge_ned2,
                e_face_ned2,
                e_z_node,
                e_z_edge,
            }
        })
        .collect()
}

/// Evaluate the full Ned-2 transverse field `E_t` at barycentric `l` inside
/// triangle `ti`. Sums all 6 edge + 2 face contributions using the per-edge
/// and per-face coefficient arrays.
#[inline]
fn ned2_et_at(
    te: &TriEdges,
    g: &[[f64; 2]; 3],
    e_edge_ned2: &[[f64; 2]],
    e_face_ned2_tri: &[f64; 2],
    _ti: usize,
    l: [f64; 3],
) -> [f64; 2] {
    let mut e = [0.0f64; 2];
    for el in 0..3 {
        let coef = &e_edge_ned2[te.gidx[el]];
        for m in 0..2 {
            let phi = ned2_edge_basis(el, m, te, g, l);
            e[0] += coef[m] * phi[0];
            e[1] += coef[m] * phi[1];
        }
    }
    for m in 0..2 {
        let phi = ned2_face_basis(m, te, g, l);
        e[0] += e_face_ned2_tri[m] * phi[0];
        e[1] += e_face_ned2_tri[m] * phi[1];
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build a structured right-triangle mesh of an `a × b` rectangle
    /// with `nx × ny` cells (each split into two triangles). Returns the
    /// global 3D nodes (on the z = 0 plane) and the triangle connectivity.
    fn rect_mesh(
        a: f64,
        b: f64,
        nx: usize,
        ny: usize,
    ) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let mut nodes = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([
                    a * i as f64 / nx as f64,
                    b * j as f64 / ny as f64,
                    0.0,
                ]);
            }
        }
        let id = |i: usize, j: usize| j * (nx + 1) + i;
        let mut tris = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let (n00, n10, n01, n11) =
                    (id(i, j), id(i + 1, j), id(i, j + 1), id(i + 1, j + 1));
                tris.push([n00, n10, n11]);
                tris.push([n00, n11, n01]);
            }
        }
        (nodes, tris)
    }

    #[test]
    fn extracts_a_rectangular_cross_section() {
        let (nodes, tris) = rect_mesh(2.0, 1.0, 4, 2);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        assert_eq!(pm.n_nodes(), 5 * 3);
        // The 4 rectangle sides are boundary; interior nodes are not.
        let n_boundary = pm.on_boundary.iter().filter(|&&x| x).count();
        // Perimeter nodes of a 5×3 grid: 2*(5+3) - 4 = 12.
        assert_eq!(n_boundary, 12, "boundary node count");
        // Total area recovered = a·b.
        let area: f64 =
            pm.tris.iter().map(|&t| pm.tri_geom(t).0).sum();
        assert!((area - 2.0).abs() < 1e-12, "area = {area}");
    }

    #[test]
    fn tm_modes_of_a_rectangular_guide_match_analytic() {
        // TM_mn cutoff of an a×b guide: k_c = π·√((m/a)² + (n/b)²).
        // Lowest TM mode is TM₁₁. Use a square 1×1 → k_c = π√2.
        let (nodes, tris) = rect_mesh(1.0, 1.0, 16, 16);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let modes = solve_modes(&pm, ModeKind::Tm, 1);
        assert!(!modes.is_empty(), "no TM mode found");
        let kc = modes[0].k_c;
        let want = PI * (2.0_f64).sqrt();
        // P1 on a 16×16 mesh: a couple percent high (FEM over-estimates
        // eigenvalues, lumped mass shifts a touch the other way).
        let rel = (kc - want).abs() / want;
        assert!(rel < 0.04, "TM₁₁ k_c = {kc:.4}, want {want:.4} (rel {rel:.3})");
    }

    #[test]
    fn vector_solver_recovers_te10_neff_homogeneous() {
        // Full-vector solve on a homogeneous (ε=1) 2×1 rectangular metallic
        // guide at k0 above the TE₁₀ cutoff. The dominant mode's effective
        // index must match the analytic TE₁₀: n_eff² = 1 − (k_c/k0)² with
        // k_c = π/a = π/2.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 20, 10);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let eps = vec![1.0; pm.tris.len()];
        let k0 = 3.0; // > π/2 ≈ 1.571, so TE₁₀ propagates
        let modes = solve_vector_modes(&pm, &eps, k0, 1);
        assert!(!modes.is_empty(), "vector solve found no propagating mode");
        let kc = PI / 2.0;
        let want = (1.0 - (kc / k0).powi(2)).sqrt(); // analytic n_eff
        let got = modes[0].n_eff;
        let rel = (got - want).abs() / want;
        assert!(
            rel < 0.05,
            "vector TE₁₀ n_eff = {got:.4}, want {want:.4} (rel {rel:.3})",
        );
        // E_t of TE₁₀ is along v̂ = ŷ and peaks mid-width: check the node
        // nearest (a/2, b/2) has a dominant v-component.
        let mut best = 0usize;
        let mut bestd = f64::INFINITY;
        for (i, n) in pm.nodes.iter().enumerate() {
            let d = (n[0] - 1.0).powi(2) + (n[1] - 0.5).powi(2);
            if d < bestd {
                bestd = d;
                best = i;
            }
        }
        let e = modes[0].e_uv_node[best];
        assert!(
            e[1].abs() > 2.0 * e[0].abs().max(1e-12),
            "TE₁₀ E_t not dominantly along v̂ at mid-width: {e:?}",
        );
    }

    #[test]
    fn internal_pec_septum_raises_vector_cutoff() {
        // A 2×1 guide with a vertical PEC septum at x = 1 splits into two
        // 1×1 half-guides. The dominant mode's cutoff jumps from the open
        // guide's k_c = π/2 (width 2) to k_c = π (width 1). In the vector
        // solver a PEC is `tangential E = 0`, applied to the internal
        // septum edges + nodes. At k0 = 5 (above both cutoffs) the
        // effective index drops accordingly: n_eff² = 1 − (k_c/k0)² goes
        // from ≈ 0.901 (open) to ≈ 0.605 (septum). This is the clean
        // signature that the internal-PEC constraint splits the guide.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 20, 10);
        let pec_global: Vec<bool> =
            nodes.iter().map(|p| (p[0] - 1.0).abs() < 1e-9).collect();
        let eps = vec![1.0; tris.len()];
        let k0 = 5.0;

        let pm_open =
            PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let open = solve_vector_modes(&pm_open, &eps, k0, 1);
        assert!(!open.is_empty(), "no open-guide mode");
        let neff2_open = open[0].n_eff.powi(2);

        let pm_sept = PortMesh2D::from_face(
            &nodes, &tris, [0.0, 0.0, 1.0], Some(&pec_global),
        );
        assert!(pm_sept.on_pec.iter().any(|&x| x), "no septum PEC nodes");
        let sept = solve_vector_modes(&pm_sept, &eps, k0, 1);
        assert!(!sept.is_empty(), "no septum mode (check cutoff < k0)");
        let neff2_sept = sept[0].n_eff.powi(2);

        let want_open = 1.0 - (PI / 2.0 / k0).powi(2); // ≈ 0.901
        let want_sept = 1.0 - (PI / k0).powi(2); // ≈ 0.605
        assert!(
            (neff2_open - want_open).abs() < 0.05,
            "open n_eff² = {neff2_open:.3}, want {want_open:.3}",
        );
        assert!(
            (neff2_sept - want_sept).abs() < 0.06,
            "septum n_eff² = {neff2_sept:.3}, want {want_sept:.3}, \
             internal PEC not splitting the guide",
        );
        assert!(
            neff2_sept < neff2_open - 0.1,
            "septum did not raise the cutoff: {neff2_sept:.3} vs {neff2_open:.3}",
        );
    }

    #[test]
    fn vector_solver_dielectric_filled_neff() {
        // A 2×1 guide fully filled with ε_r = 4: the TE₁₀ dispersion
        // becomes β² = ε_r k0² − k_c², so n_eff² = ε_r − (k_c/k0)² with
        // k_c = π/a = π/2. This validates the ε_r coupling in the
        // assembly (mass + longitudinal terms).
        let (nodes, tris) = rect_mesh(2.0, 1.0, 20, 10);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let eps = vec![4.0; pm.tris.len()];
        let k0 = 3.0;
        let modes = solve_vector_modes(&pm, &eps, k0, 1);
        assert!(!modes.is_empty(), "no propagating mode in ε=4 guide");
        let kc = PI / 2.0;
        let want = (4.0 - (kc / k0).powi(2)).sqrt();
        let got = modes[0].n_eff;
        let rel = (got - want).abs() / want;
        assert!(
            rel < 0.05,
            "ε=4 TE₁₀ n_eff = {got:.4}, want {want:.4} (rel {rel:.3})",
        );
    }

    #[test]
    fn vector_solver_inhomogeneous_neff_is_bracketed() {
        // Half-air (ε=1) / half-dielectric (ε=4) 2×1 guide: the dominant
        // mode's effective index must lie strictly between the all-air and
        // all-dielectric results (a partially-filled guide's n_eff is
        // bracketed by its homogeneous limits), the qualitative signature
        // of a correct inhomogeneous (microstrip-class) hybrid solve.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 20, 10);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        // ε = 4 in the lower half (v < 0.5), air above.
        let eps: Vec<f64> = pm
            .tris
            .iter()
            .map(|&t| {
                let vc = (pm.nodes[t[0]][1]
                    + pm.nodes[t[1]][1]
                    + pm.nodes[t[2]][1])
                    / 3.0;
                if vc < 0.5 { 4.0 } else { 1.0 }
            })
            .collect();
        let k0 = 3.0;
        let modes = solve_vector_modes(&pm, &eps, k0, 1);
        assert!(!modes.is_empty(), "no propagating mode in half-filled guide");
        let neff2 = modes[0].n_eff.powi(2);
        let kc = PI / 2.0;
        let lo = 1.0 - (kc / k0).powi(2); // all-air n_eff²
        let hi = 4.0 - (kc / k0).powi(2); // all-dielectric n_eff²
        assert!(
            neff2 > lo && neff2 < hi,
            "half-filled n_eff² = {neff2:.3} not bracketed by ({lo:.3}, {hi:.3})",
        );
    }

    #[test]
    fn numerical_te10_profile_matches_the_analytic_shape() {
        // The numerical TE₁₀ mode of a 2×1 guide must reproduce the
        // analytic transverse-E shape E_y ∝ sin(πx/a): a peak at
        // mid-width, vanishing at the side walls, purely along v̂ = ŷ.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 24, 12);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let modes = solve_modes(&pm, ModeKind::Te, 1);
        let nm = NumericalMode::from_scalar(pm, &modes[0], ModeKind::Te);
        // Mid-width (x = a/2 = 1): |e_t| near the unit peak, along ŷ.
        let mid = nm.e_profile([1.0, 0.5, 0.0]);
        let mag = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
        assert!(mag > 0.9, "mid-width |e_t| = {mag:.3}, expected ≈ 1");
        assert!(
            mid[1].abs() > 0.9 && mid[0].abs() < 0.15,
            "TE₁₀ e_t not along ŷ: {mid:?}",
        );
        // Near a side wall (x ≈ 0.05): the field should be small.
        let wall = nm.e_profile([0.05, 0.5, 0.0]);
        let wmag =
            (wall[0] * wall[0] + wall[1] * wall[1] + wall[2] * wall[2]).sqrt();
        assert!(wmag < 0.3, "near-wall |e_t| = {wmag:.3}, expected ≪ 1");
        // h_t ⟂ e_t and the cutoff matches.
        let h = nm.h_profile([1.0, 0.5, 0.0]);
        let ehdot = mid[0] * h[0] + mid[1] * h[1] + mid[2] * h[2];
        assert!(ehdot.abs() < 1e-9, "e_t·h_t = {ehdot:.3e}");
        assert!((nm.cutoff() - PI / 2.0).abs() / (PI / 2.0) < 0.04);
    }

    #[test]
    fn microstrip_quasi_tem_natural_scale() {
        // Dimensionless microstrip-like cross-section: width 2, height 1.
        // Substrate band y < 0.1 with eps=3.55, air above with eps=1.
        // Trace as internal-PEC line at y=0.1, x in [0.9, 1.1]. k0=3, well
        // below the homogeneous TE10 cutoff (k_c = pi/2 = 1.571), so the
        // ONLY propagating mode is the conductor-supported quasi-TEM.
        //
        // The Ned-2 + P2 element is second-order, so a 20×10 mesh resolves the
        // quasi-TEM mode as well as the old Whitney/P1 path did on 60×40, and
        // keeps the DENSE eigensolve tractable (~2.9k DOF). Fine production
        // port faces need the sparse shift-invert path (Stage B); the dense
        // solver here is O(ndof³) and would blow up past a few thousand DOF.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 20, 10);
        let mut pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let eps: Vec<f64> = pm.tris.iter().map(|&t| {
            let yc = (pm.nodes[t[0]][1] + pm.nodes[t[1]][1] + pm.nodes[t[2]][1]) / 3.0;
            if yc < 0.1 { 3.55 } else { 1.0 }
        }).collect();
        // Mark trace nodes on internal PEC.
        let mut on_pec = vec![false; pm.nodes.len()];
        for (i, n) in pm.nodes.iter().enumerate() {
            if (n[1] - 0.1).abs() < 1e-6 && (n[0] - 1.0).abs() < 0.1 + 1e-6 {
                on_pec[i] = true;
            }
        }
        pm.on_pec = on_pec;
        let n_pec = pm.on_pec.iter().filter(|&&b| b).count();
        assert!(n_pec >= 2, "trace PEC nodes = {n_pec}, expected at least 2");

        let k0 = 3.0;
        let modes = solve_vector_modes(&pm, &eps, k0, 3);
        eprintln!("[microstrip natural-scale] k0={k0}, eps_max=3.55, {} modes:", modes.len());
        for (i, m) in modes.iter().enumerate() {
            eprintln!("  mode[{i}] n_eff = {:.4}", m.n_eff);
        }
        assert!(!modes.is_empty(), "no microstrip quasi-TEM mode");
        // Quasi-TEM n_eff is bracketed by sqrt(eff_eps) where eff_eps lies
        // between 1 (no substrate) and 3.55 (full substrate fill).
        let n_eff = modes[0].n_eff;
        assert!(n_eff > 1.0 && n_eff < 3.55_f64.sqrt(),
            "n_eff = {n_eff:.3} not bracketed (1, 1.884)");
    }

    /// Build the microstrip cross-section at an arbitrary geometric scale `s`
    /// (coords ×s, k0 ×1/s, so the electrical size κ = k0·ℓ is INVARIANT) and
    /// return the quasi-TEM n_eff. Same physics at every `s`; n_eff must match.
    fn microstrip_neff_at_scale(s: f64) -> f64 {
        // Coarse mesh: the natural-vs-µm comparison is mesh-accuracy-independent
        // (same resolution at both scales), so this only needs to be consistent.
        let (nodes, tris) = rect_mesh(2.0 * s, 1.0 * s, 12, 6);
        let mut pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let eps: Vec<f64> = pm.tris.iter().map(|&t| {
            let yc = (pm.nodes[t[0]][1] + pm.nodes[t[1]][1] + pm.nodes[t[2]][1]) / 3.0;
            if yc < 0.1 * s { 3.55 } else { 1.0 }
        }).collect();
        let mut on_pec = vec![false; pm.nodes.len()];
        for (i, n) in pm.nodes.iter().enumerate() {
            if (n[1] - 0.1 * s).abs() < 1e-6 * s && (n[0] - 1.0 * s).abs() < (0.1 + 1e-6) * s {
                on_pec[i] = true;
            }
        }
        pm.on_pec = on_pec;
        let k0 = 3.0 / s;
        let modes = solve_vector_modes(&pm, &eps, k0, 3);
        assert!(!modes.is_empty(), "no quasi-TEM mode at scale s={s:.0e}");
        modes[0].n_eff
    }

    #[test]
    fn vector_solver_is_scale_invariant_micron_port() {
        // The reported failure: an RFIC port at µm coordinates returns 0 modes,
        // even though the physics is identical to the O(1) "natural" scale. The
        // port-local non-dimensionalization makes the solve scale-invariant.
        // Asserting natural == µm at the SAME (coarse) mesh is what matters and
        // is mesh-accuracy-independent — the absolute n_eff need not be
        // converged here; the well-resolved value is pinned by the
        // `microstrip_quasi_tem_natural_scale` test above.
        let n_natural = microstrip_neff_at_scale(1.0);          // O(1) coords
        let n_micron = microstrip_neff_at_scale(1.0e-6);        // µm coords, GHz·1e6
        eprintln!("[scale-invariance] n_eff: natural={n_natural:.5} µm={n_micron:.5}");
        let rel = (n_micron - n_natural).abs() / n_natural;
        assert!(rel < 1e-3, "n_eff drifted with scale: µm {n_micron:.5} vs natural \
            {n_natural:.5} ({:.3}% — solve is not scale-invariant)", 100.0 * rel);
    }

    /// Reference triangle for basis-evaluator tests: v0=(0,0), v1=(1,0),
    /// v2=(0,1). Returns the per-edge TriEdges (all `sign = 1`, edge lengths
    /// √2, 1, 1 in the (0,1,2) local-edge ordering) and the P1 gradients.
    fn ref_tri() -> (TriEdges, [[f64; 2]; 3]) {
        let te = TriEdges {
            gidx: [0, 1, 2],
            sign: [1.0, 1.0, 1.0],
            len:  [(2.0_f64).sqrt(), 1.0, 1.0],
        };
        // ∇λ_0 = (−1,−1), ∇λ_1 = (1, 0), ∇λ_2 = (0, 1)
        let g = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
        (te, g)
    }

    #[test]
    fn ned2_edge_basis_matches_analytic() {
        let (te, g) = ref_tri();
        // Barycentre l = (1/3, 1/3, 1/3) → physical point (1/3, 1/3).
        let l = [1.0 / 3.0; 3];
        // Edge 0 (a=1, b=2). Second-kind Ned-2 mode 0 = W (plain Whitney):
        //   W_0(l) = l[1]·∇λ_2 − l[2]·∇λ_1 = (1/3)·(0,1) − (1/3)·(1,0)
        //          = (−1/3, 1/3); sign = +1, no length factor.
        let want_e0m0 = [-1.0 / 3.0, 1.0 / 3.0];
        let got = ned2_edge_basis(0, 0, &te, &g, l);
        assert!((got[0] - want_e0m0[0]).abs() < 1e-12 && (got[1] - want_e0m0[1]).abs() < 1e-12,
            "edge0 mode0: got {got:?}, want {want_e0m0:?}");
        // curl(W) = 2·c_12, c_12 = (∇λ_1 × ∇λ_2)_z = 1. So curl = 2.
        let curl = ned2_edge_curl(0, 0, &te, &g, l);
        assert!((curl - 2.0).abs() < 1e-12,
            "edge0 mode0 curl: got {curl}, want 2");

        // Mode 1 = (λ_a − λ_b)·W. At the barycentre λ_a = λ_b = 1/3 so it
        // vanishes; check off-centre l = (0.2, 0.5, 0.3): λ_a−λ_b = 0.5−0.3 = 0.2,
        //   W = l[1]·∇λ_2 − l[2]·∇λ_1 = 0.5·(0,1) − 0.3·(1,0) = (−0.3, 0.5),
        //   mode1 = 0.2·(−0.3, 0.5) = (−0.06, 0.1).
        let l2 = [0.2, 0.5, 0.3];
        let got1 = ned2_edge_basis(0, 1, &te, &g, l2);
        let want_e0m1 = [-0.06, 0.1];
        assert!((got1[0] - want_e0m1[0]).abs() < 1e-12 && (got1[1] - want_e0m1[1]).abs() < 1e-12,
            "edge0 mode1: got {got1:?}, want {want_e0m1:?}");
        // curl(mode1) = 3·(λ_a−λ_b)·c_12 = 3·0.2·1 = 0.6.
        let curl1 = ned2_edge_curl(0, 1, &te, &g, l2);
        assert!((curl1 - 0.6).abs() < 1e-12, "edge0 mode1 curl: got {curl1}, want 0.6");
    }

    #[test]
    fn p2_basis_is_kronecker_at_nodes() {
        // P2 vertex basis at vertex 0 (l = (1,0,0)) should be 1; at other
        // vertices and edge midpoints, 0.
        let v0 = [1.0, 0.0, 0.0];
        let v1 = [0.0, 1.0, 0.0];
        let v2 = [0.0, 0.0, 1.0];
        let m0 = [0.0, 0.5, 0.5]; // midpoint of edge 0 (between v1, v2)
        let m1 = [0.5, 0.0, 0.5]; // midpoint of edge 1 (between v2, v0)
        let m2 = [0.5, 0.5, 0.0]; // midpoint of edge 2 (between v0, v1)
        let nodes = [v0, v1, v2, m0, m1, m2];
        for i in 0..6 {
            for (j, l) in nodes.iter().enumerate() {
                let v = p2_basis(i, *l);
                if i == j {
                    assert!((v - 1.0).abs() < 1e-12, "N_{i}({j}) = {v}, want 1");
                } else {
                    assert!(v.abs() < 1e-12, "N_{i}({j}) = {v}, want 0");
                }
            }
        }
    }

    #[test]
    fn p2_partition_of_unity() {
        // Σ_i N_i(l) ≡ 1 everywhere on the triangle (degree-2 polynomial
        // identity). Check at the barycentre and at a handful of random pts.
        let pts = [
            [1.0 / 3.0; 3],
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.25, 0.25, 0.5],
        ];
        for l in &pts {
            let s: f64 = (0..6).map(|i| p2_basis(i, *l)).sum();
            assert!((s - 1.0).abs() < 1e-12, "Σ N_i({l:?}) = {s}, want 1");
        }
    }

    #[test]
    fn ned2_quad_rule_sums_to_one() {
        let sum: f64 = NED2_QPTS_DEG5.iter().map(|q| q.0).sum();
        assert!((sum - 1.0).abs() < 1e-12, "quad weights sum to {sum}");
        // Each quadrature point is a valid barycentric triple.
        for q in &NED2_QPTS_DEG5 {
            let s = q.1 + q.2 + q.3;
            assert!((s - 1.0).abs() < 1e-12, "barycentric sum = {s}");
            assert!(q.1 >= 0.0 && q.2 >= 0.0 && q.3 >= 0.0,
                "negative bary at {q:?}");
        }
    }

    #[test]
    fn te_modes_of_a_rectangular_guide_match_analytic() {
        // TE_mn cutoff of an a×b guide: k_c = π·√((m/a)² + (n/b)²).
        // Dominant TE₁₀ of a 2×1 guide → k_c = π/2.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 24, 12);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let modes = solve_modes(&pm, ModeKind::Te, 1);
        assert!(!modes.is_empty(), "no TE mode found");
        let kc = modes[0].k_c;
        let want = PI / 2.0;
        let rel = (kc - want).abs() / want;
        assert!(rel < 0.04, "TE₁₀ k_c = {kc:.4}, want {want:.4} (rel {rel:.3})");
    }
}
