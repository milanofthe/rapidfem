//! The parameters of a mesh: sizes, tolerances, budgets and periodic pairs.

use rapidmesh_geom::TaggedPlc;

/// Sizing parameters of a mesh.
#[derive(Debug, Clone)]
pub struct MeshParams {
    /// Target edge length, in the sense gmsh gives its mesh size: volume
    /// edges have a median near `maxh` and spread around it, curve samples
    /// are at most `maxh` apart. Curvature, feature and local sources refine
    /// below it.
    pub maxh: f64,
    /// Per-region target edge length, overriding maxh inside that region
    /// (Maxwell FEM sizes regions by local wavelength, h ~ lambda/sqrt(eps)).
    /// Interfaces and creases follow the finer adjacent region; transitions
    /// into coarser regions grade naturally.
    pub region_maxh: Vec<(u32, f64)>,
    /// The volume refinement adds at most this many points to each region.
    pub max_points: usize,
    /// Size grading: the target edge length may grow by at most this factor
    /// per unit distance from finer features (h(x) is Lipschitz with this
    /// constant). 0.5 grows neighbor elements by roughly 1.5x; INFINITY
    /// disables grading (sizes jump at region interfaces).
    pub grading: f64,
    /// Per-face-tag target edge length, overriding the adjacent regions'
    /// targets on those patches.
    pub face_maxh: Vec<(u32, f64)>,
    /// Per-solid SURFACE target edge length, keyed by the owner solid index
    /// in [TaggedPlc::surface_owners] (scene insertion order, voids
    /// included): refines the solid's boundary patches and grades into the
    /// surrounding volume. The only sizing handle that reaches a void's
    /// walls (a coax inner conductor has no region and no face tag).
    pub surface_maxh: Vec<(u32, f64)>,
    /// Point size sources `(position, h)`: the target shrinks to `h` at the
    /// point and recovers along the Lipschitz grading away from it
    /// (the hook for error-driven adaptive refinement).
    pub size_points: Vec<([f64; 3], f64)>,
    /// Relative chord (sagitta) tolerance for curved EDGES: a curve of radius `R`
    /// is sampled at `h = R*sqrt(8*tol_edge)`, so the chord deviates by at most
    /// `tol_edge * R`. Scale-invariant (constant segments per arc). Default 1e-2.
    pub tol_edge: f64,
    /// Relative chord (sagitta) tolerance for curved SURFACES, the 2D analogue of
    /// [`MeshParams::tol_edge`]: a facet on a surface of principal radius `R` is
    /// sized `h = R*sqrt(8*tol_surf)`. Default 1e-2. (There is no volume
    /// tolerance: the volume size follows from the surface.)
    pub tol_surf: f64,
    /// Relative geometric error the elements may make (`0` = off, the chord
    /// tolerances apply): the volume of every region and the area of every
    /// sheet within this share of the true ones, measured on elements of
    /// [`MeshParams::order`]. Turned into a gap each curved boundary may
    /// leave, by the thickness of what it bounds (see
    /// [`crate::sizing::CurvatureLaw`]); an explicit per-entity tolerance
    /// still wins.
    pub geom_error: f64,
    /// The order of the elements the geometric error is measured on: 1 flat
    /// (chords), 2 quadratic (mid-edge nodes on the true geometry, see the
    /// second-order mesh), which follows a curve with far fewer elements.
    pub order: u8,
    /// Maximum element edge length on EDGES (1-cells), combined with the global
    /// [`MeshParams::maxh`] as `min(maxh, cap_edge)`. `INFINITY` = no extra cap.
    pub cap_edge: f64,
    /// Maximum element edge length on SURFACES (2-cells); `min(maxh, cap_surf)`.
    pub cap_surf: f64,
    /// Maximum element edge length in the VOLUME (3-cells); `min(maxh, cap_vol)`.
    /// Per-region overrides come from [`MeshParams::region_maxh`].
    pub cap_vol: f64,
    /// Per-EDGE size override `(brep edge id, maxh)`: the hierarchical
    /// `g.region(..).surf(..).edge(..).maxh` resolves to entries here, overriding
    /// the global [`MeshParams::cap_edge`] on that specific edge.
    pub edge_maxh: Vec<(u32, f64)>,
    /// Per-EDGE deflection override `(brep edge id, tol)`, overriding
    /// [`MeshParams::tol_edge`] on that edge.
    pub edge_tol: Vec<(u32, f64)>,
    /// Per-FACE size override `(brep face id, maxh)`, overriding
    /// [`MeshParams::cap_surf`] on that face.
    pub surf_maxh: Vec<(u32, f64)>,
    /// Per-FACE deflection override `(brep face id, tol)`, overriding
    /// [`MeshParams::tol_surf`] on that face.
    pub surf_tol: Vec<(u32, f64)>,
    /// Minimum element edge length on SURFACES (2-cells) and their edges: a hard
    /// floor, the field is never refined below it (and the element budget cannot
    /// go under it). `0` = off.
    pub min_h_surf: f64,
    /// Smallest triangle angle (degrees) the refinement of each face mesh
    /// aims for; `0` takes 28.
    pub surf_min_angle: f64,
    /// Triangle budget of a surface mesh: the sizes coarsen by one global
    /// factor over a few remeshes until the count is at most a little over
    /// it (see [`crate::surface_mesh`]). `0` = none.
    pub surf_target_count: usize,
    /// Pairs of B-rep faces meshed with the same triangles: face `b` is
    /// face `a` shifted.
    pub periodic: Vec<PeriodicPair>,
    /// Elements across the thickness of each region: inside a region of
    /// thickness `t` (`2 V / S`, see
    /// [`TaggedPlc::region_thickness`](rapidmesh_geom::TaggedPlc::region_thickness))
    /// the size is at most `t / cells_across`, as if given per region, so a
    /// thin plate or wire gets proper tets through it. `0` = off: a layer far
    /// thinner than the size then gets flat tets through it.
    pub cells_across: f64,
}

impl Default for MeshParams {
    fn default() -> Self {
        MeshParams {
            maxh: f64::INFINITY,
            region_maxh: Vec::new(),
            max_points: 100_000,
            grading: 0.5,
            face_maxh: Vec::new(),
            surface_maxh: Vec::new(),
            size_points: Vec::new(),
            tol_edge: 1e-2,
            tol_surf: 1e-2,
            geom_error: 0.0,
            order: 1,
            cap_edge: f64::INFINITY,
            cap_surf: f64::INFINITY,
            cap_vol: f64::INFINITY,
            edge_maxh: Vec::new(),
            edge_tol: Vec::new(),
            surf_maxh: Vec::new(),
            surf_tol: Vec::new(),
            min_h_surf: 0.0,
            surf_min_angle: 0.0,
            surf_target_count: 0,
            periodic: Vec::new(),
            cells_across: 0.0,
        }
    }
}

fn lookup(table: &[(u32, f64)], id: usize) -> Option<f64> {
    table
        .iter()
        .find(|&&(i, _)| i as usize == id)
        .map(|&(_, v)| v)
}

impl MeshParams {
    /// A copy with every size target scaled by `s`: lengths by `s`, chord
    /// tolerances by `s^2` (size is proportional to sqrt(tol)). The element-budget
    /// loop retunes the global scale with this while preserving the relative
    /// refinement; the cap methods read the scaled fields, so they scale too.
    pub fn scaled(&self, s: f64) -> MeshParams {
        let sv = |v: &[(u32, f64)], e: f64| -> Vec<(u32, f64)> {
            v.iter().map(|&(t, h)| (t, h * e)).collect()
        };
        MeshParams {
            maxh: self.maxh * s,
            region_maxh: sv(&self.region_maxh, s),
            max_points: self.max_points,
            grading: self.grading,
            face_maxh: sv(&self.face_maxh, s),
            surface_maxh: sv(&self.surface_maxh, s),
            size_points: self.size_points.iter().map(|&(p, h)| (p, h * s)).collect(),
            tol_edge: self.tol_edge * s * s,
            tol_surf: self.tol_surf * s * s,
            // The size grows as the square root of the gap for flat elements,
            // as its fourth root for quadratic ones.
            geom_error: self.geom_error * s.powi(if self.order >= 2 { 4 } else { 2 }),
            order: self.order,
            cap_edge: self.cap_edge * s,
            cap_surf: self.cap_surf * s,
            cap_vol: self.cap_vol * s,
            edge_maxh: sv(&self.edge_maxh, s),
            edge_tol: sv(&self.edge_tol, s * s),
            surf_maxh: sv(&self.surf_maxh, s),
            surf_tol: sv(&self.surf_tol, s * s),
            // An absolute floor: the budget does not move it.
            min_h_surf: self.min_h_surf,
            surf_min_angle: self.surf_min_angle,
            surf_target_count: self.surf_target_count,
            periodic: self.periodic.clone(),
            cells_across: self.cells_across,
        }
    }

    /// These parameters with the thickness bound of every region of `plc`
    /// (see [`MeshParams::cells_across`]) in the per-region sizes, the
    /// finer of the two winning. Applying it twice changes nothing.
    pub fn with_thickness_caps(&self, plc: &TaggedPlc) -> MeshParams {
        let mut p = self.clone();
        if !(self.cells_across > 0.0) {
            return p;
        }
        for (r, t) in plc.region_thickness() {
            let cap = t / self.cells_across;
            match p.region_maxh.iter_mut().find(|(rr, _)| *rr == r) {
                Some(e) => e.1 = e.1.min(cap),
                None => p.region_maxh.push((r, cap)),
            }
        }
        p
    }

    /// Effective edge-length cap on 1-cells: the global cap tightened by the
    /// per-dimension edge cap.
    pub fn edge_cap(&self) -> f64 {
        self.maxh.min(self.cap_edge)
    }

    /// Effective edge-length cap on 2-cells (surfaces).
    pub fn surf_cap(&self) -> f64 {
        self.maxh.min(self.cap_surf)
    }

    /// Effective edge-length cap in the 3-cell (volume).
    pub fn vol_cap(&self) -> f64 {
        self.maxh.min(self.cap_vol)
    }

    /// The size floor of the refinement for a model of extent `extent`:
    /// clamps curvature-driven runaway at an eighth of the reference size,
    /// while explicit user targets (per edge, face, region or point) may be
    /// finer and win.
    pub fn h_floor(&self, extent: f64) -> f64 {
        let h_ref = if self.maxh.is_finite() {
            self.maxh
        } else {
            extent / crate::constants::DEFAULT_SUBDIV
        };
        let user_min = self
            .edge_maxh
            .iter()
            .chain(&self.surf_maxh)
            .chain(&self.face_maxh)
            .chain(&self.surface_maxh)
            .chain(&self.region_maxh)
            .map(|&(_, h)| h)
            .chain(self.size_points.iter().map(|&(_, h)| h))
            .fold(f64::INFINITY, f64::min);
        self.min_h_surf.max((h_ref / 8.0).min(user_min)).max(1e-12)
    }

    /// Size cap for brep edge `id`: its per-edge override, else the edge cap.
    pub fn edge_maxh_for(&self, id: usize) -> f64 {
        lookup(&self.edge_maxh, id)
            .unwrap_or(f64::INFINITY)
            .min(self.edge_cap())
    }

    /// Deflection for brep edge `id`: its per-edge override, else `tol_edge`.
    pub fn edge_tol_for(&self, id: usize) -> f64 {
        lookup(&self.edge_tol, id).unwrap_or(self.tol_edge)
    }

    /// Size cap for brep face `id`: its per-face override, else the surface cap.
    pub fn surf_maxh_for(&self, id: usize) -> f64 {
        lookup(&self.surf_maxh, id)
            .unwrap_or(f64::INFINITY)
            .min(self.surf_cap())
    }

    /// Deflection for brep face `id`: its per-face override, else `tol_surf`.
    pub fn surf_tol_for(&self, id: usize) -> f64 {
        lookup(&self.surf_tol, id).unwrap_or(self.tol_surf)
    }
}

/// Patch `b` is patch `a` moved by `shift`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicPair {
    pub a: u32,
    pub b: u32,
    pub shift: [f64; 3],
}
