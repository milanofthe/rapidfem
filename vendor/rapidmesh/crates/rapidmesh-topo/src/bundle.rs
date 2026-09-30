//! THE two embedding endpoints: one for 2D, one for 3D.
//!
//! Each returns a complete bundle — mesh + topology + element geometry — so a
//! consumer (a FEM or MoM solver) gets *everything* from a single call and never
//! has to choose between overlapping accessors. This is the canonical Rust API
//! for embedding rapidmesh:
//!
//! ```ignore
//! // 2D / MoM — the production 2D path (surf2d, the gmsh-grade mesher the wasm
//! // landing uses). Raw tagged 2D polygons in; a planar mesh bundle out.
//! let m = rapidmesh_topo::mesh_2d(&regions, |_p| 0.05, &Default::default());
//! m.points; m.tris; m.tri_tags; m.topo; m.geom; m.rwg_candidate_edges();
//!
//! // 3D / FEM — the volume path.
//! let v = rapidmesh_topo::mesh_3d(&plc, &params);
//! v.mesh; v.topo; v.geom; v.exact_face_normals();
//! ```
//!
//! The 2D path is *the* 2D path: this endpoint, the wasm landing, and the 3D
//! surface stage's planar patches all run the same `surf2d` core. The basis-aware
//! layer (RWG/Nédélec DOFs, quadrature, assembly) lives in the solver.

use crate::source::Tris;
use crate::{TetGeometry, TetTopology, TriGeometry, TriTopology};
use rapidmesh_geom::TaggedPlc;
use rapidmesh_tet::gradefield::GradedField;
use rapidmesh_tet::surf2d::{mesh_polygon_with_chains, PolyMeshParams};
use rapidmesh_tet::{mesh_plc_with, MeshParams, TetMesh};

// ============================== 2D / surface (MoM) ==========================

/// A tagged 2D region: an outer loop with optional holes, all in the xy plane.
/// The `tag` flows to every triangle of this region (the conductor / layer id a
/// MoM build reads for same-tag RWG edges).
#[derive(Clone, Debug)]
pub struct Region2D {
    /// Outer boundary loop (closed; CCW).
    pub outer: Vec<[f64; 2]>,
    /// Hole loops (closed; CW), nested inside `outer`.
    pub holes: Vec<Vec<[f64; 2]>>,
    /// Conductor / layer tag carried by this region's triangles.
    pub tag: i64,
    /// CONSTRAINT CHAINS (open polylines, world coordinates) inside this region's layer: their
    /// segments become element edges, without changing the region's extent. The cross-layer
    /// alignment hands a layer the outlines of the conductors on the neighbouring layers,
    /// clipped to it, so the charge edges those conductors induce fall on element edges. A
    /// chain is attached to the patch that contains it; a chain end on the patch outline is
    /// made an outline vertex.
    pub constraints: Vec<Vec<[f64; 2]>>,
}

impl Region2D {
    /// A region with no holes.
    pub fn new(outer: Vec<[f64; 2]>, tag: i64) -> Self {
        Region2D {
            outer,
            holes: Vec::new(),
            tag,
            constraints: Vec::new(),
        }
    }
}

/// How the diagonals of the edge band's cells lean along a boundary loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BandDiagonals {
    /// Alternating from cell to cell (no preferred direction).
    #[default]
    Alternate,
    /// All one way along the loop, turning nowhere.
    Along,
}

impl std::str::FromStr for BandDiagonals {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "alternate" => Ok(BandDiagonals::Alternate),
            "along" => Ok(BandDiagonals::Along),
            other => Err(format!(
                "unknown band diagonals {other:?} (expected \"alternate\" or \"along\")"
            )),
        }
    }
}

/// Tuning for the production 2D path. `Default` matches the wasm landing.
#[derive(Debug, Clone, Copy)]
pub struct Mesh2DOptions {
    /// Ruppert minimum-angle bound (degrees).
    pub min_angle_deg: f64,
    /// CVT (Lloyd) seed-relaxation iterations.
    pub cvt_iters: usize,
    /// Maximum Ruppert refinement passes.
    pub max_passes: usize,
    /// Triangle BUDGET (`0` = field-driven, the default). When `> 0` the provided sizing field is
    /// scaled by ONE global factor so the mesh lands ~`target_count` triangles — the field sets WHERE
    /// the elements go, the budget HOW MANY (shared across every patch of every group).
    pub target_count: usize,
    /// HARD minimum element size (`0` = none). Applied AFTER the budget scaling, so a refined /
    /// AMR field cannot drive a hotspot (the conductor-edge singularity) below it and swallow the
    /// whole budget — the floor caps the local density and lets the budget spread to the other
    /// important regions. Essential for budgeted AMR.
    pub minh: f64,
    /// HARD maximum element size (`0` = none). Applied AFTER the budget scaling — the gmsh
    /// `MeshSizeMax` analogue: caps the COARSEST element so low-error regions still resolve the
    /// geometry, bounding the fine↔coarse contrast to `maxh/minh` (no extreme size jumps).
    pub maxh: f64,
    /// Sizing-field GRADING (Lipschitz slope, `0` = none): the size field is gradient-limited so it
    /// changes no faster than this per unit distance. Smooths sharp fine→coarse transitions (e.g. an
    /// AMR indicator field) into a graded mesh — essential for element quality.
    pub grading: f64,
    /// How the diagonals of the edge band's cells lean along the outline.
    /// Which reads a conductor's loss closer depends on the solver's basis:
    /// alternating was closer on a thick spiral, one way on a straight
    /// graded strip (#155).
    pub band_diagonals: BandDiagonals,
    /// The size at most this share of the local width of the region (the
    /// width of a trace at the nearest point of its outline; `0` = off):
    /// `1.0` gives cells about as long as the trace is wide.
    pub width_size: f64,
    /// Constraint chains snapped together: a chain point closer than this
    /// share of the size to another chain or to the outline moves onto it,
    /// and chain pieces that then run along another are dropped (`0` =
    /// off). Nearly coincident outlines of stacked layers otherwise force
    /// a band of slivers between them.
    pub snap: f64,
}

impl Default for Mesh2DOptions {
    fn default() -> Self {
        Mesh2DOptions {
            min_angle_deg: 28.0,
            cvt_iters: 4,
            max_passes: 12,
            target_count: 0,
            minh: 0.0,
            maxh: 0.0,
            grading: 0.0,
            band_diagonals: BandDiagonals::Alternate,
            width_size: 0.0,
            snap: 0.0,
        }
    }
}

/// Everything about a 2D mesh (the MoM target): the gmsh-grade planar
/// triangulation from the production 2D path, its derived topology, and its
/// element geometry. Coordinates are 2D.
#[derive(Clone)]
pub struct Mesh2D {
    /// Vertices (2D).
    pub points: Vec<[f64; 2]>,
    /// Triangles (CCW).
    pub tris: Vec<[u32; 3]>,
    /// Conductor / layer tag per triangle (parallel to `tris`).
    pub tri_tags: Vec<i64>,
    /// Edges, incidence, per-edge tags, vertex stars.
    pub topo: TriTopology,
    /// Areas, centroids, second moments, edge lengths/midpoints (planar).
    pub geom: TriGeometry,
    /// The input regions (polygons), retained so the mesh can re-mesh ITSELF with a
    /// new sizing field — the basis for adaptive (AMR) refinement. See [`Mesh2D::remesh`].
    pub regions: Vec<Region2D>,
    /// The meshing options this mesh was built with (retained for `remesh`).
    pub opts: Mesh2DOptions,
}

impl Mesh2D {
    /// Re-mesh the SAME regions with a new sizing field `target` (e.g. an AMR-refined
    /// size field that shrinks `h` near marked elements). Returns a fresh, equally
    /// self-remeshable `Mesh2D`. The regions and meshing options are this mesh's own.
    pub fn remesh(&self, target: impl Fn([f64; 2]) -> f64) -> Mesh2D {
        mesh_2d(&self.regions, target, &self.opts)
    }

    /// RWG-eligible edges `[v0, v1, tri_plus, tri_minus]` (interior, same-tag).
    /// The canonical query lives on [`TriTopology`] — shared with the 3D-surface
    /// endpoint; this forwards so the 2D bundle exposes everything in one place.
    pub fn rwg_candidate_edges(&self) -> Vec<[u32; 4]> {
        self.topo.rwg_candidate_edges()
    }

    /// Conductor outline `[v0, v1, tri]` (free side or tag change). Forwards to
    /// [`TriTopology::boundary_edges`].
    pub fn boundary_edges(&self) -> Vec<[u32; 3]> {
        self.topo.boundary_edges()
    }

    /// Port helper: boundary edges whose both endpoints lie on `{axis = value}`
    /// (`axis` 0/1) with the other coordinate in `[lo, hi]`. Returns vertex pairs.
    pub fn edges_on_line(
        &self,
        axis: usize,
        value: f64,
        lo: f64,
        hi: f64,
        tol: f64,
    ) -> Vec<[u32; 2]> {
        let other = if axis == 0 { 1 } else { 0 };
        let on = |vi: u32| {
            let p = self.points[vi as usize];
            (p[axis] - value).abs() <= tol && p[other] >= lo - tol && p[other] <= hi + tol
        };
        self.topo
            .boundary_edges()
            .iter()
            .filter(|e| on(e[0]) && on(e[1]))
            .map(|e| [e[0], e[1]])
            .collect()
    }
}

/// THE 2D endpoint: mesh ONE layer's tagged polygons through the production 2D path
/// (`surf2d`). A convenience over [`mesh_layers`] for the single-group case (the AMR
/// remesh, the tests, every existing caller). `target` is the sizing field `h(x)`.
pub fn mesh_2d(
    regions: &[Region2D],
    target: impl Fn([f64; 2]) -> f64,
    opts: &Mesh2DOptions,
) -> Mesh2D {
    let groups = [regions.to_vec()];
    mesh_layers(&groups, target, opts)
        .into_iter()
        .next()
        .unwrap_or_else(|| assemble_mesh2d(Vec::new(), Vec::new(), Vec::new(), regions, opts))
}

/// THE grouped 2D endpoint: mesh ALL tagged polygons of ALL layers in ONE shot under ONE
/// global triangle budget. Each `group` is a layer's regions. WITHIN a group, overlapping /
/// abutting regions are unioned (i_overlay) into connected, conformally meshed patches — so a
/// winding handed in as many abutting arcs becomes one RWG-connected component (no open
/// winding); regions of DIFFERENT groups never merge (different metal layers, free to overlap
/// in the plane).
///
/// The merged patches of every group are packed side by side into ONE virtual canvas (disjoint
/// bins, each separated by a gap ≥ the largest patch extent, so no patch's refinement can see or
/// bridge to another) and meshed in a SINGLE constrained triangulation. With `opts.target_count
/// > 0` the Ruppert budget is therefore GLOBAL: the worst (most over-sized / angle-violating)
/// triangles across ALL patches of ALL groups draw from one shared count, so the budget flows to
/// where the geometry needs it — an emergent distribution, not an a-priori per-layer / per-area
/// split. The mesh is then un-translated back to world coordinates and split per group; tags are
/// assigned per ORIGINAL region. Returns one [`Mesh2D`] per input group, in input order.
pub fn mesh_layers(
    groups: &[Vec<Region2D>],
    target: impl Fn([f64; 2]) -> f64,
    opts: &Mesh2DOptions,
) -> Vec<Mesh2D> {
    // 1. Per group: union into merged, pairwise-disjoint patches (record the owning group).
    struct Patch {
        group: usize,
        outer: Vec<[f64; 2]>,
        holes: Vec<Vec<[f64; 2]>>,
        chains: Vec<Vec<[f64; 2]>>,
    }
    let mut patches: Vec<Patch> = Vec::new();
    for (g, regions) in groups.iter().enumerate() {
        let first_patch = patches.len();
        let merged: Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> = if regions.len() > 1 {
            union_regions(regions)
        } else {
            regions
                .iter()
                .filter(|r| r.outer.len() >= 3)
                .map(|r| (r.outer.clone(), r.holes.clone()))
                .collect()
        };
        for (outer, holes) in merged {
            if outer.len() >= 3 {
                patches.push(Patch {
                    group: g,
                    outer,
                    holes,
                    chains: Vec::new(),
                });
            }
        }
        // The group's constraint chains go to the patch that contains them; a chain end on
        // the patch outline becomes an outline vertex (a T-junction is then a shared vertex).
        for ch in regions.iter().flat_map(|r| r.constraints.iter()) {
            if ch.len() < 2 {
                continue;
            }
            let mid = chain_midpoint(ch);
            let Some(pi) = (first_patch..patches.len()).find(|&pi| {
                let p = &patches[pi];
                point_in_ring(mid, &p.outer) && !p.holes.iter().any(|h| point_in_ring(mid, h))
            }) else {
                continue;
            };
            let patch = &mut patches[pi];
            for &end in [ch[0], ch[ch.len() - 1]].iter() {
                insert_on_loop(&mut patch.outer, end);
                for h in &mut patch.holes {
                    insert_on_loop(h, end);
                }
            }
            patch.chains.push(ch.clone());
        }
    }
    if patches.is_empty() {
        return groups
            .iter()
            .map(|r| assemble_mesh2d(Vec::new(), Vec::new(), Vec::new(), r, opts))
            .collect();
    }

    // 1b. SAMPLE the sizing field onto a grid (world coords) and GRADIENT-LIMIT it once: an expensive
    //     field (an AMR indicator that locates a triangle per query, a distance field) is then
    //     evaluated in O(1), and — crucially — sharp fine→coarse jumps are Lipschitz-smoothed to slope
    //     `opts.grading` so the mesh grades cleanly (element quality). A flat field is unaffected.
    //     This is THE canonical way to feed rapidmesh a graded / adaptive sizing field.
    if opts.snap > 0.0 {
        for p in &mut patches {
            snap_chains(&mut p.outer, &mut p.holes, &mut p.chains, &|q| {
                opts.snap * target(q)
            });
        }
    }
    // The size at most a share of the local width, where asked for: cells
    // as long as the trace is wide along it, the band rows across.
    let width = (opts.width_size > 0.0).then(|| {
        let rings: Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> = patches
            .iter()
            .map(|p| (p.outer.clone(), p.holes.clone()))
            .collect();
        crate::offset::WidthField::new(&rings)
    });
    let target = |p: [f64; 2]| -> f64 {
        let t = target(p);
        match &width {
            Some(w) => t.min(opts.width_size * w.at(p)),
            None => t,
        }
    };
    let (mut wlo, mut whi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    let mut h_seed = f64::INFINITY;
    for p in &patches {
        for &q in &p.outer {
            wlo[0] = wlo[0].min(q[0]);
            wlo[1] = wlo[1].min(q[1]);
            whi[0] = whi[0].max(q[0]);
            whi[1] = whi[1].max(q[1]);
        }
        h_seed = h_seed.min(target(centroid2(&p.outer)).max(1e-12));
    }
    let finest = if opts.minh > 0.0 {
        h_seed.min(opts.minh)
    } else {
        h_seed
    };
    let grid_cell = (0.5 * finest).max(1e-12); // GradedField caps the grid at MAX_N cells/axis
    let gf = GradedField::from_fn(wlo, whi, grid_cell, opts.grading, |p| target(p).max(1e-12));
    let tfield = |p: [f64; 2]| -> f64 { gf.eval(p) };

    // 2. Effective sizing. The mesh is graded to the field `f(x)`; with a triangle BUDGET the SAME
    //    field SHAPE is scaled by ONE global factor so the mesh lands ~`budget` triangles. A flat
    //    field then gives a uniform budget mesh (the scaling sweep); a graded / AMR field distributes
    //    the budget where it is fine (more elements at the marked edges) — the sizing field controls
    //    WHERE, the budget controls HOW MANY. The scale is closed-form from the field's natural count
    //    integral `N = ∫ K/f² dA` (the triangle count scales as `1/scale²` under `f → scale·f`), so
    //    it is single-pass — NOT an iterative size retune. `budget == 0` is the plain field-driven
    //    path (`eff_scale = 1`).
    let budget = opts.target_count;
    let minh = opts.minh.max(0.0);
    let maxh = if opts.maxh > 0.0 {
        opts.maxh
    } else {
        f64::INFINITY
    };
    // Sample the field over the patches — `(value, cell area)` — for the budget scaling. The final
    // size is `h(p) = max(s · f(p), minh)`; the scale `s` is found by bisecting the FLOORED count
    // `K · ∫ 1/h² dA = budget` (closed-form, single pass, no re-meshing). The `minh` floor caps the
    // local density, so a hotspot can't draw the whole budget — it spreads to the other regions.
    let mut samples: Vec<(f64, f64)> = Vec::new();
    if budget > 0 {
        const NDIV: usize = 96;
        for p in &patches {
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for &q in &p.outer {
                lo[0] = lo[0].min(q[0]);
                lo[1] = lo[1].min(q[1]);
                hi[0] = hi[0].max(q[0]);
                hi[1] = hi[1].max(q[1]);
            }
            let (w, h) = ((hi[0] - lo[0]).max(1e-30), (hi[1] - lo[1]).max(1e-30));
            let (nx, ny) = if w >= h {
                (NDIV, (NDIV as f64 * h / w).ceil().max(1.0) as usize)
            } else {
                ((NDIV as f64 * w / h).ceil().max(1.0) as usize, NDIV)
            };
            let (dx, dy) = (w / nx as f64, h / ny as f64);
            let cell_a = dx * dy;
            for i in 0..nx {
                for j in 0..ny {
                    let q = [lo[0] + (i as f64 + 0.5) * dx, lo[1] + (j as f64 + 0.5) * dy];
                    if point_in_ring(q, &p.outer) && !p.holes.iter().any(|hl| point_in_ring(q, hl))
                    {
                        samples.push((tfield(q).max(1e-12), cell_a));
                    }
                }
            }
        }
    }
    // Empirical triangle-count constant for the field-limited CVT+Ruppert.
    const COUNT_K: f64 = 4.0;
    let count_of = |s: f64| -> f64 {
        COUNT_K
            * samples
                .iter()
                .map(|&(t, a)| {
                    let hh = (s * t).clamp(minh, maxh);
                    a / (hh * hh)
                })
                .sum::<f64>()
    };
    let s_scale: f64 = if budget > 0 && !samples.is_empty() {
        // count(s) decreases in s; geometric bisection to count(s) = budget.
        let (mut lo, mut hi) = (1e-12f64, 1e12f64);
        for _ in 0..64 {
            let m = (lo * hi).sqrt();
            if count_of(m) > budget as f64 {
                lo = m
            } else {
                hi = m
            }
        }
        (lo * hi).sqrt()
    } else {
        1.0
    };
    let field = |p: [f64; 2]| -> f64 { (s_scale * tfield(p)).clamp(minh, maxh) };
    // Finest FINAL size (sets the field-limited CVT seed step under a budget).
    let f_min = samples
        .iter()
        .map(|&(t, _)| (s_scale * t).clamp(minh, maxh))
        .fold(f64::INFINITY, f64::min);

    // 3. Resample each patch's boundary onto `field` (world coords) — the protected core meshes
    //    against this — and measure its bbox + the finest field sample (the field-driven CVT seed).
    let mut rs_loops: Vec<Vec<Vec<[f64; 2]>>> = Vec::with_capacity(patches.len());
    let mut rs_chains: Vec<Vec<Vec<[f64; 2]>>> = Vec::with_capacity(patches.len());
    let mut bbmin: Vec<[f64; 2]> = Vec::with_capacity(patches.len());
    let mut extent = 0.0f64;
    let mut field_step = f64::INFINITY;
    for p in &patches {
        let mut loops: Vec<Vec<[f64; 2]>> = Vec::with_capacity(1 + p.holes.len());
        loops.push(resample_loop(&p.outer, &field));
        for h in &p.holes {
            loops.push(resample_loop(h, &field));
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for lp in &loops {
            for &q in lp {
                lo[0] = lo[0].min(q[0]);
                lo[1] = lo[1].min(q[1]);
                hi[0] = hi[0].max(q[0]);
                hi[1] = hi[1].max(q[1]);
            }
        }
        extent = extent.max(hi[0] - lo[0]).max(hi[1] - lo[1]);
        field_step = field_step.min(field(centroid2(&p.outer)).max(1e-12));
        rs_chains.push(
            p.chains
                .iter()
                .map(|c| align_chain(c, &loops, &field))
                .collect(),
        );
        rs_loops.push(loops);
        bbmin.push(lo);
    }
    let step = if budget > 0 && f_min.is_finite() {
        // field-limited: seed at the finest FINAL size so the graded field drives the local density.
        (0.7 * f_min).max(1e-12)
    } else if field_step.is_finite() && field_step > 0.0 {
        field_step
    } else {
        (extent * 0.05).max(1e-9)
    };
    let cell = extent + extent.max(step * 4.0); // bin pitch: patch extent + a ≥extent gap

    // 3. Pack the patches into a square-ish grid of `cell`-sized bins. `offset[i]` shifts patch i's
    //    bbox-min onto its bin corner; `grid` maps a bin (col,row) back to its patch index.
    let ncols = (patches.len() as f64).sqrt().ceil().max(1.0) as usize;
    let nrows = patches.len().div_ceil(ncols);
    let mut offset: Vec<[f64; 2]> = Vec::with_capacity(patches.len());
    let mut grid: Vec<i64> = vec![-1; ncols * nrows];
    let mut all_loops: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut all_chains: Vec<Vec<[f64; 2]>> = Vec::new();
    for (i, loops) in rs_loops.iter().enumerate() {
        let (col, row) = (i % ncols, i / ncols);
        let off = [
            col as f64 * cell - bbmin[i][0],
            row as f64 * cell - bbmin[i][1],
        ];
        offset.push(off);
        grid[row * ncols + col] = i as i64;
        for lp in loops {
            all_loops.push(lp.iter().map(|q| [q[0] + off[0], q[1] + off[1]]).collect());
        }
        for ch in &rs_chains[i] {
            all_chains.push(ch.iter().map(|q| [q[0] + off[0], q[1] + off[1]]).collect());
        }
    }
    // Bin lookup: a canvas point's patch (vertices live in their patch's bin; gap points snap to a
    // neighbour — only their field value is read, and gap triangles are filtered by `inside`).
    // Quarter-cell nudge before flooring: `off = k·cell − bbmin` puts a patch's bbox-min boundary
    // vertices EXACTLY on the bin edge, so `p + off` can land one fp-ulp BELOW `k·cell` and a plain
    // floor assigns the vertex to the neighbouring bin — the read-back then un-translates it with
    // the wrong offset, displacing it by a whole bin pitch (an opamp met2 serpentine).
    // Patch content spans `[k·cell, k·cell + extent]` with `extent ≤ cell/2`, so `+cell/4` keeps
    // every content point strictly inside its bin while absorbing fp noise on the low edge.
    let bin_at = |p: [f64; 2]| -> usize {
        let col = ((p[0] + 0.25 * cell) / cell)
            .floor()
            .clamp(0.0, (ncols - 1) as f64) as usize;
        let row = ((p[1] + 0.25 * cell) / cell)
            .floor()
            .clamp(0.0, (nrows - 1) as f64) as usize;
        let g = grid[row * ncols + col];
        if g >= 0 {
            g as usize
        } else {
            0
        }
    };

    // Field on the canvas: un-translate each query to its patch's world frame, then the scaled field.
    let canvas_target = |p: [f64; 2]| -> f64 {
        let i = bin_at(p);
        field([p[0] - offset[i][0], p[1] - offset[i][1]])
    };

    // ONE constrained mesh over the whole canvas — the GLOBAL budget lives here.
    let params = PolyMeshParams {
        step,
        min_angle_deg: opts.min_angle_deg,
        target_count: opts.target_count,
        cvt_iters: opts.cvt_iters,
        max_passes: opts.max_passes,
    };
    let (cpoints, mut ctris) =
        mesh_polygon_with_chains(&all_loops, &all_chains, canvas_target, &params, |_, _| {});
    orient_band_diagonals(
        &cpoints,
        &mut ctris,
        &all_loops,
        &all_chains,
        opts.band_diagonals,
    );

    // 6. Read back: un-translate every vertex, then split the triangles per group. Each emitted
    //    triangle lies wholly within one patch (cross-gap triangles were filtered by `inside`), so
    //    a triangle's group is the group of its first vertex's patch. Reindex per group.
    let pt_patch: Vec<usize> = cpoints.iter().map(|&p| bin_at(p)).collect();
    let world: Vec<[f64; 2]> = cpoints
        .iter()
        .enumerate()
        .map(|(k, &p)| {
            let o = offset[pt_patch[k]];
            [p[0] - o[0], p[1] - o[1]]
        })
        .collect();

    let mut out: Vec<Mesh2D> = Vec::with_capacity(groups.len());
    for (g, regions) in groups.iter().enumerate() {
        let mut remap: Vec<i32> = vec![-1; world.len()];
        let mut points: Vec<[f64; 2]> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for t in &ctris {
            if patches[pt_patch[t[0]]].group != g {
                continue;
            }
            let mut nt = [0u32; 3];
            for (k, &gi) in t.iter().enumerate() {
                if remap[gi] < 0 {
                    remap[gi] = points.len() as i32;
                    points.push(world[gi]);
                }
                nt[k] = remap[gi] as u32;
            }
            tris.push(nt);
        }
        // Tags per ORIGINAL region: one region → its tag directly; several → containment.
        let tri_tags: Vec<i64> = if regions.len() == 1 {
            vec![regions[0].tag; tris.len()]
        } else {
            tris.iter()
                .map(|t| {
                    let (a, b, c) = (
                        points[t[0] as usize],
                        points[t[1] as usize],
                        points[t[2] as usize],
                    );
                    tag_at(
                        [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0],
                        regions,
                    )
                })
                .collect()
        };
        out.push(assemble_mesh2d(points, tris, tri_tags, regions, opts));
    }
    out
}

/// Build the full [`Mesh2D`] bundle (topology + element geometry) from a planar triangulation.
fn assemble_mesh2d(
    points: Vec<[f64; 2]>,
    tris: Vec<[u32; 3]>,
    tri_tags: Vec<i64>,
    regions: &[Region2D],
    opts: &Mesh2DOptions,
) -> Mesh2D {
    let topo = TriTopology::build(&Tris {
        tris: &tris,
        tags: &tri_tags,
        n_verts: points.len(),
    });
    let geom = TriGeometry::build_2d(&topo, &points);
    Mesh2D {
        points,
        tris,
        tri_tags,
        topo,
        geom,
        regions: regions.to_vec(),
        opts: *opts,
    }
}

/// Robust 2D union of all region polygons into connected shapes (i_overlay, MIT). Overlapping or
/// abutting regions merge into one shape; separate ones stay separate. Outers are forced CCW and
/// holes CW so the non-zero fill rule unions correctly regardless of the input orientation. Each
/// output is `(outer, holes)` in i_overlay's canonical orientation (outer CCW, holes CW).
pub fn union_regions(regions: &[Region2D]) -> Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> {
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::float::simplify::SimplifyShape;
    let mut contours: Vec<Vec<[f64; 2]>> = Vec::new();
    for r in regions {
        if r.outer.len() >= 3 {
            contours.push(oriented(&r.outer, true));
            for h in &r.holes {
                if h.len() >= 3 {
                    contours.push(oriented(h, false));
                }
            }
        }
    }
    contours
        .simplify_shape(FillRule::NonZero, 0.0)
        .into_iter()
        .map(|shape| {
            let mut it = shape.into_iter();
            let outer = it.next().unwrap_or_default();
            (outer, it.collect())
        })
        .collect()
}

/// Boolean overlay of two region sets (i_overlay, non-zero fill): `Intersect`, `Difference`
/// (subject minus clip), `Union`. Outers CCW, holes CW.
pub fn overlay_regions(
    subj: &[Region2D],
    clip: &[Region2D],
    rule: i_overlay::core::overlay_rule::OverlayRule,
) -> Vec<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> {
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::float::single::SingleFloatOverlay;
    let shapes = |rs: &[Region2D]| -> Vec<Vec<Vec<[f64; 2]>>> {
        rs.iter()
            .filter(|r| r.outer.len() >= 3)
            .map(|r| {
                let mut sh = vec![oriented(&r.outer, true)];
                sh.extend(
                    r.holes
                        .iter()
                        .filter(|h| h.len() >= 3)
                        .map(|h| oriented(h, false)),
                );
                sh
            })
            .collect()
    };
    let (a, b) = (shapes(subj), shapes(clip));
    a.overlay(&b, rule, FillRule::NonZero)
        .into_iter()
        .map(|shape| {
            let mut it = shape.into_iter();
            let outer = it.next().unwrap_or_default();
            (outer, it.collect())
        })
        .collect()
}

/// A point on a chain: the midpoint of its middle segment (inside the region for a chain
/// that lies inside it).
fn chain_midpoint(ch: &[[f64; 2]]) -> [f64; 2] {
    let k = (ch.len() - 1) / 2;
    let (a, b) = (ch[k], ch[k + 1]);
    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
}

/// Insert `q` as a vertex of the loop if it lies on one of its edges (and is not a vertex yet).
/// Snaps the constraint chains of a patch together: each chain point closer
/// than `tol` at it to the outline, a hole or another chain moves onto the
/// nearest point there, which that loop or chain takes as a point of its
/// own; then the chain pieces that run along a loop or an earlier chain are
/// dropped (the chain splits there).
fn snap_chains(
    outer: &mut Vec<[f64; 2]>,
    holes: &mut [Vec<[f64; 2]>],
    chains: &mut Vec<Vec<[f64; 2]>>,
    tol: &dyn Fn([f64; 2]) -> f64,
) {
    let nearest_on = |pts: &[[f64; 2]], closed: bool, q: [f64; 2]| -> Option<(f64, [f64; 2])> {
        let n = pts.len();
        let segs = if closed { n } else { n.saturating_sub(1) };
        let mut best: Option<(f64, [f64; 2])> = None;
        for i in 0..segs {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let l2 = dx * dx + dy * dy;
            let t = if l2 > 0.0 {
                (((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let x = [a[0] + t * dx, a[1] + t * dy];
            let d = (x[0] - q[0]).hypot(x[1] - q[1]);
            if best.is_none_or(|b| d < b.0) {
                best = Some((d, x));
            }
        }
        best
    };
    // Where snapped points go: 0 the outer loop, 1 + k hole k, and
    // chains by index after the holes.
    let mut into: Vec<(usize, [f64; 2])> = Vec::new();
    for ci in 0..chains.len() {
        for vi in 0..chains[ci].len() {
            let q = chains[ci][vi];
            let reach = tol(q);
            let mut best: Option<(f64, [f64; 2], usize)> = None;
            let mut consider = |hit: Option<(f64, [f64; 2])>, owner: usize| {
                if let Some((d, x)) = hit {
                    if d > 0.0 && d <= reach && best.is_none_or(|b| d < b.0) {
                        best = Some((d, x, owner));
                    }
                }
            };
            consider(nearest_on(outer, true, q), 0);
            for (k, h) in holes.iter().enumerate() {
                consider(nearest_on(h, true, q), 1 + k);
            }
            for (cj, other) in chains.iter().enumerate() {
                if cj != ci {
                    consider(nearest_on(other, false, q), 1 + holes.len() + cj);
                }
            }
            if let Some((_, x, owner)) = best {
                chains[ci][vi] = x;
                into.push((owner, x));
            }
        }
    }
    for (owner, x) in into {
        match owner {
            0 => insert_on_loop(outer, x),
            k if k <= holes.len() => insert_on_loop(&mut holes[k - 1], x),
            k => insert_on_chain(&mut chains[k - 1 - holes.len()], x),
        }
    }
    // Pieces along a loop or an earlier chain go.
    let bits = |p: [f64; 2]| (p[0].to_bits(), p[1].to_bits());
    let key = |a: [f64; 2], b: [f64; 2]| {
        let (x, y) = (bits(a), bits(b));
        if x <= y {
            (x, y)
        } else {
            (y, x)
        }
    };
    let mut taken: std::collections::HashSet<((u64, u64), (u64, u64))> = Default::default();
    for lp in std::iter::once(&*outer).chain(holes.iter()) {
        for i in 0..lp.len() {
            taken.insert(key(lp[i], lp[(i + 1) % lp.len()]));
        }
    }
    let mut out: Vec<Vec<[f64; 2]>> = Vec::new();
    for ch in chains.iter() {
        let mut ch: Vec<[f64; 2]> = ch.clone();
        ch.dedup_by(|a, b| bits(*a) == bits(*b));
        let mut piece: Vec<[f64; 2]> = Vec::new();
        for w in ch.windows(2) {
            if taken.contains(&key(w[0], w[1])) {
                if piece.len() >= 2 {
                    out.push(std::mem::take(&mut piece));
                }
                piece.clear();
                continue;
            }
            if piece.is_empty() {
                piece.push(w[0]);
            }
            piece.push(w[1]);
        }
        if piece.len() >= 2 {
            out.push(piece);
        }
        for w in ch.windows(2) {
            taken.insert(key(w[0], w[1]));
        }
    }
    *chains = out;
}

/// Puts `q` into the open chain `ch` on the segment it lies on.
fn insert_on_chain(ch: &mut Vec<[f64; 2]>, q: [f64; 2]) {
    let mut scale = 0.0f64;
    for p in ch.iter() {
        scale = scale.max(p[0].abs()).max(p[1].abs());
    }
    let eps = 1e-9 * scale.max(1e-30);
    if ch
        .iter()
        .any(|p| (p[0] - q[0]).abs() <= eps && (p[1] - q[1]).abs() <= eps)
    {
        return;
    }
    for i in 0..ch.len().saturating_sub(1) {
        let (a, b) = (ch[i], ch[i + 1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l2 = dx * dx + dy * dy;
        if l2 <= 0.0 {
            continue;
        }
        let t = ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / l2;
        if !(0.0..=1.0).contains(&t) {
            continue;
        }
        if (q[0] - a[0] - t * dx).hypot(q[1] - a[1] - t * dy) <= eps {
            ch.insert(i + 1, q);
            return;
        }
    }
}

fn insert_on_loop(lp: &mut Vec<[f64; 2]>, q: [f64; 2]) {
    let n = lp.len();
    if n < 2 {
        return;
    }
    let mut scale = 0.0f64;
    for p in lp.iter() {
        scale = scale.max(p[0].abs()).max(p[1].abs());
    }
    let eps = 1e-9 * scale.max(1e-30);
    if lp
        .iter()
        .any(|p| (p[0] - q[0]).abs() <= eps && (p[1] - q[1]).abs() <= eps)
    {
        return;
    }
    for i in 0..n {
        let (a, b) = (lp[i], lp[(i + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l2 = dx * dx + dy * dy;
        if l2 <= 0.0 {
            continue;
        }
        let t = ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / l2;
        if !(0.0..=1.0).contains(&t) {
            continue;
        }
        let d = (q[0] - a[0] - t * dx).hypot(q[1] - a[1] - t * dy);
        if d <= eps {
            lp.insert(i + 1, q);
            return;
        }
    }
}

/// A constraint chain resampled on the nodes of the loops it runs along. A chain next to an
/// outline (an edge band row) would otherwise be split to the field on its own: its segments
/// are shorter than the outline's (their ends are mitred), so its nodes drift against the
/// outline's from both ends and the drift changes sign in the middle, where the band's cell
/// diagonals then turn over (a node column through the band; on a thick conductor a local
/// loss excess, #155). Here every loop node whose perpendicular foot lies inside
/// a chain segment, within half the local size, puts a chain node at that foot; a segment no
/// loop node projects onto is resampled to the field as before.
fn align_chain(
    ch: &[[f64; 2]],
    loops: &[Vec<[f64; 2]>],
    target: &impl Fn([f64; 2]) -> f64,
) -> Vec<[f64; 2]> {
    let n = ch.len();
    if n < 2 {
        return ch.to_vec();
    }
    // loop nodes in a uniform grid, so a segment scans only the cells around it
    let pts: Vec<[f64; 2]> = loops.iter().flatten().copied().collect();
    let cell = ch
        .windows(2)
        .map(|w| target(w[0]))
        .fold(f64::INFINITY, f64::min)
        .max(1e-12);
    let key = |p: [f64; 2]| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64);
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = Default::default();
    for (i, &p) in pts.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i);
    }
    let mut out = Vec::with_capacity(2 * n);
    for w in ch.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l2 = dx * dx + dy * dy;
        out.push(a);
        if l2 <= 0.0 {
            continue;
        }
        let len = l2.sqrt();
        let at = |t: f64| [a[0] + t * dx, a[1] + t * dy];
        let reach = 0.5 * target(a).max(target(b));
        let (k0, k1) = (
            key([a[0].min(b[0]) - reach, a[1].min(b[1]) - reach]),
            key([a[0].max(b[0]) + reach, a[1].max(b[1]) + reach]),
        );
        let mut ts: Vec<f64> = Vec::new();
        for kx in k0.0..=k1.0 {
            for ky in k0.1..=k1.1 {
                for &i in grid.get(&(kx, ky)).into_iter().flatten() {
                    let p = pts[i];
                    let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2;
                    if t <= 0.0 || t >= 1.0 {
                        continue;
                    }
                    let f = at(t);
                    if (p[0] - f[0]).hypot(p[1] - f[1]) <= 0.5 * target(f) {
                        ts.push(t);
                    }
                }
            }
        }
        if ts.is_empty() {
            let seg = resample_chain(&[a, b], target);
            out.extend_from_slice(&seg[1..seg.len() - 1]);
            continue;
        }
        ts.sort_by(f64::total_cmp);
        // one node per foot, none crowding another or the segment's ends
        let mut last = 0.0;
        for t in ts {
            let h = target(at(t));
            if (t - last) * len >= 0.25 * h && (1.0 - t) * len >= 0.25 * h {
                out.push(at(t));
                last = t;
            }
        }
    }
    out.push(ch[n - 1]);
    out
}

/// The diagonals of the band between each loop and a chain along it, alternating from cell to
/// cell along the loop. For a loop edge `(p, q)` whose triangle has its apex `x` on a chain, the
/// band cell behind it is the quad `p, q, y, x` with `y` the chain node across `q`; its diagonal
/// runs from `p` to `y` on every even loop edge and from `q` to the node across `p` on every
/// odd one, set by flipping the cell's inner edge where the cell is convex. On an aligned band
/// (`align_chain`) the cells are rectangles, whose Delaunay diagonal is a tie that round-off
/// decides, turning over at random along a run. The alternating pattern has no preferred
/// direction: measured on a thick spiral against a 3D reference it read the loss 1 to 1.5
/// points closer than one diagonal orientation along the whole loop (#155).
fn orient_band_diagonals(
    points: &[[f64; 2]],
    tris: &mut [[usize; 3]],
    loops: &[Vec<[f64; 2]>],
    chains: &[Vec<[f64; 2]>],
    lean: BandDiagonals,
) {
    let bits = |p: [f64; 2]| (p[0].to_bits(), p[1].to_bits());
    let index: std::collections::HashMap<(u64, u64), usize> = points
        .iter()
        .enumerate()
        .map(|(i, &p)| (bits(p), i))
        .collect();
    let on_chain: std::collections::HashSet<usize> = chains
        .iter()
        .flatten()
        .filter_map(|&p| index.get(&bits(p)).copied())
        .collect();
    let is_constraint = {
        let mut s: std::collections::HashSet<(usize, usize)> = Default::default();
        let mut add = |u: usize, v: usize| {
            s.insert((u.min(v), u.max(v)));
        };
        for lp in loops {
            for k in 0..lp.len() {
                if let (Some(&u), Some(&v)) = (
                    index.get(&bits(lp[k])),
                    index.get(&bits(lp[(k + 1) % lp.len()])),
                ) {
                    add(u, v);
                }
            }
        }
        for ch in chains {
            for w in ch.windows(2) {
                if let (Some(&u), Some(&v)) = (index.get(&bits(w[0])), index.get(&bits(w[1]))) {
                    add(u, v);
                }
            }
        }
        s
    };
    // triangles by edge
    let mut by_edge: std::collections::HashMap<(usize, usize), Vec<usize>> = Default::default();
    for (t, tri) in tris.iter().enumerate() {
        for k in 0..3 {
            let (u, v) = (tri[k], tri[(k + 1) % 3]);
            by_edge.entry((u.min(v), u.max(v))).or_default().push(t);
        }
    }
    let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let apex =
        |tri: &[usize; 3], u: usize, v: usize| *tri.iter().find(|&&w| w != u && w != v).unwrap();
    let dist =
        |i: usize, j: usize| (points[i][0] - points[j][0]).hypot(points[i][1] - points[j][1]);
    for lp in loops {
        for k in 0..lp.len() {
            let (Some(&p), Some(&q)) = (
                index.get(&bits(lp[k])),
                index.get(&bits(lp[(k + 1) % lp.len()])),
            ) else {
                continue;
            };
            // odd loop edges lean the other way (the same rule with the edge's ends swapped),
            // unless every cell leans one way
            let (p, q) = if k % 2 == 1 && lean == BandDiagonals::Alternate {
                (q, p)
            } else {
                (p, q)
            };
            let Some(&t1) = by_edge.get(&(p.min(q), p.max(q))).and_then(|v| v.first()) else {
                continue;
            };
            let x = apex(&tris[t1], p, q);
            // only a band cell whose apex sits by `p`: its diagonal (q, x) runs backward
            if !on_chain.contains(&x) || dist(x, p) >= dist(x, q) {
                continue;
            }
            let e = (q.min(x), q.max(x));
            if is_constraint.contains(&e) {
                continue;
            }
            let Some(&t2) = by_edge.get(&e).and_then(|v| v.iter().find(|&&t| t != t1)) else {
                continue;
            };
            let y = apex(&tris[t2], q, x);
            if !on_chain.contains(&y) || y == p {
                continue;
            }
            // the flipped pair (p, q, y) and (p, y, x) must keep the orientation of the old one
            let s = orient(points[p], points[q], points[x]).signum();
            let (n1, n2) = ([p, q, y], [p, y, x]);
            let ok = |t: [usize; 3]| orient(points[t[0]], points[t[1]], points[t[2]]) * s > 0.0;
            if !ok(n1) || !ok(n2) {
                continue;
            }
            let fix = |t: [usize; 3]| if s > 0.0 { t } else { [t[0], t[2], t[1]] };
            let (old1, old2) = (tris[t1], tris[t2]);
            tris[t1] = fix(n1);
            tris[t2] = fix(n2);
            // keep the edge map true for the next loop edges
            for (t, old, new) in [(t1, old1, tris[t1]), (t2, old2, tris[t2])] {
                for k in 0..3 {
                    let (u, v) = (old[k], old[(k + 1) % 3]);
                    if let Some(l) = by_edge.get_mut(&(u.min(v), u.max(v))) {
                        l.retain(|&w| w != t);
                    }
                }
                for k in 0..3 {
                    let (u, v) = (new[k], new[(k + 1) % 3]);
                    by_edge.entry((u.min(v), u.max(v))).or_default().push(t);
                }
            }
        }
    }
}

/// [`resample_loop`] for an OPEN chain: corners kept, edges graded by the field, canonical
/// per edge, the last point kept.
fn resample_chain(ch: &[[f64; 2]], target: &impl Fn([f64; 2]) -> f64) -> Vec<[f64; 2]> {
    let n = ch.len();
    if n < 2 {
        return ch.to_vec();
    }
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n - 1 {
        let closed = resample_loop(&[ch[i], ch[i + 1]], target);
        // A two-point "loop" resamples edge a→b then b→a; keep the first half (a .. before b).
        let half = closed.len() / 2;
        out.extend_from_slice(&closed[..half.max(1)]);
    }
    out.push(ch[n - 1]);
    out
}

/// Signed area (CCW positive).
fn signed_area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += p[i][0] * p[j][1] - p[j][0] * p[i][1];
    }
    0.5 * a
}

/// `p` re-oriented to CCW (`ccw=true`) or CW (`ccw=false`).
fn oriented(p: &[[f64; 2]], ccw: bool) -> Vec<[f64; 2]> {
    let mut v = p.to_vec();
    if (signed_area(p) > 0.0) != ccw {
        v.reverse();
    }
    v
}

/// Even-odd ray cast: is `p` strictly inside the ring?
fn point_in_ring(p: [f64; 2], ring: &[[f64; 2]]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (ring[i], ring[j]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Tag of the original region containing `p` (outer minus holes); falls back to the nearest
/// region's tag (for a centroid nudged just outside by the union's coordinate snapping).
fn tag_at(p: [f64; 2], regions: &[Region2D]) -> i64 {
    for r in regions {
        if point_in_ring(p, &r.outer) && !r.holes.iter().any(|h| point_in_ring(p, h)) {
            return r.tag;
        }
    }
    regions
        .iter()
        .min_by(|x, y| {
            let d = |r: &Region2D| {
                let c = centroid2(&r.outer);
                (c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2)
            };
            d(x).partial_cmp(&d(y)).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.tag)
        .unwrap_or(0)
}

/// Split each edge of a closed loop to the sizing field, so the boundary the protected core
/// meshes against is fine and **graded** — never a pinned coarse edge, and never an abrupt
/// short-next-to-long jump at a corner (which pins a sliver the Ruppert core cannot fix, since
/// it may not move boundary vertices).
///
/// The grading is the key: `h` is sampled at the two EDGE ENDPOINTS (not the midpoint), so two
/// edges sharing a corner agree on the spacing there — the last segment of one edge and the first
/// of the next are both ≈ `h(corner)`. Within an edge the segment length is graded geometrically
/// from `h(a)` to `h(b)`, so adjacent segment lengths differ by only the size-field ratio, never a
/// jump. Corners are preserved.
fn resample_loop(lp: &[[f64; 2]], target: &impl Fn([f64; 2]) -> f64) -> Vec<[f64; 2]> {
    let n = lp.len();
    if n < 2 {
        return lp.to_vec();
    }
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let a = lp[i];
        let b = lp[(i + 1) % n];
        out.push(a); // keep the corner
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            continue;
        }
        // CANONICAL orientation (lexicographic lo→hi): a boundary edge SHARED by two abutting
        // regions is resampled to the IDENTICAL point set from either side (same lo/hi/grading),
        // so the separately-meshed regions weld conformally at the abutment (see `mesh_2d`).
        let rev = (a[0], a[1]) > (b[0], b[1]);
        let (lo, hi) = if rev { (b, a) } else { (a, b) };
        let ha = target(lo).max(1e-9);
        let hb = target(hi).max(1e-9);
        // Segment count = ∫₀ᴸ ds/h(s) with h linear in arc length (= the graded element count).
        let uniform = (ha - hb).abs() <= 1e-9 * ha.max(hb);
        let segs = if uniform {
            (len / ha).ceil().max(1.0) as usize
        } else {
            (len / (hb - ha) * (hb / ha).ln()).abs().ceil().max(1.0) as usize
        };
        // Canonical interior point at fraction k/segs along lo→hi (geometric grading h_k =
        // ha·(hb/ha)^frac ⇒ arc position t; first segment ≈ ha, last ≈ hb).
        let pt = |k: usize| -> [f64; 2] {
            let frac = k as f64 / segs as f64;
            let t = if uniform {
                frac
            } else {
                (ha * (hb / ha).powf(frac) - ha) / (hb - ha)
            };
            [lo[0] + (hi[0] - lo[0]) * t, lo[1] + (hi[1] - lo[1]) * t]
        };
        // Emit the canonical points in the a→b traversal order (reversed if the edge runs hi→lo).
        if rev {
            for k in (1..segs).rev() {
                out.push(pt(k));
            }
        } else {
            for k in 1..segs {
                out.push(pt(k));
            }
        }
    }
    out
}

fn centroid2(pts: &[[f64; 2]]) -> [f64; 2] {
    if pts.is_empty() {
        return [0.0, 0.0];
    }
    let (mut x, mut y) = (0.0, 0.0);
    for p in pts {
        x += p[0];
        y += p[1];
    }
    let n = pts.len() as f64;
    [x / n, y / n]
}

// ============================== 3D / volume (FEM) ===========================

/// Everything about a 3D / volume mesh (the FEM target): the meshed volume, its
/// derived topology (with orientation signs), and its element geometry.
pub struct Mesh3D {
    /// The tet mesh: points, tets, regions, tagged boundary faces, surfaces.
    pub mesh: TetMesh,
    /// Edges, faces, incidence + orientation signs, vertex stars.
    pub topo: TetTopology,
    /// Volumes, ∇λ_i, face areas/normals/centroids, edge lengths.
    pub geom: TetGeometry,
    /// The B-rep face or edge of every face and edge, the regions beside
    /// every face.
    pub class: crate::classes::Classification,
}

impl Mesh3D {
    /// Bundle an existing tet mesh; topology + geometry are derived once.
    pub fn build(mesh: TetMesh) -> Self {
        let topo = TetTopology::build(&mesh);
        let geom = TetGeometry::build(&topo, &mesh.points);
        let class = crate::classes::Classification::build(&mesh, &topo);
        Mesh3D {
            mesh,
            topo,
            geom,
            class,
        }
    }

    /// Exact analytic outward normals per boundary face (`None` where planar / no
    /// closed form). Parallel to `mesh.faces`.
    pub fn exact_face_normals(&self) -> Vec<Option<[f64; 3]>> {
        crate::mesher::exact_face_normals(&self.mesh.points, &self.mesh.faces, &self.mesh.surfaces)
    }
}

/// THE 3D endpoint: mesh a PLC into a complete volume bundle, through the
/// restricted-Delaunay refinement core (the same engine the Python binding
/// runs). For an element budget, mesh with [`rapidmesh_tet::mesh_budgeted`]
/// then [`Mesh3D::build`].
pub fn mesh_3d(plc: &TaggedPlc, params: &MeshParams) -> Mesh3D {
    Mesh3D::build(mesh_plc_with(plc, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convention::NONE;
    use rapidmesh_geom::RegionTag;

    fn min_angle_deg(p: &[[f64; 2]], tris: &[[u32; 3]]) -> f64 {
        let ang = |u: [f64; 2], v: [f64; 2], w: [f64; 2]| {
            let e1 = [v[0] - u[0], v[1] - u[1]];
            let e2 = [w[0] - u[0], w[1] - u[1]];
            let d = (e1[0] * e2[0] + e1[1] * e2[1])
                / ((e1[0] * e1[0] + e1[1] * e1[1]).sqrt() * (e2[0] * e2[0] + e2[1] * e2[1]).sqrt()
                    + 1e-30);
            d.clamp(-1.0, 1.0).acos().to_degrees()
        };
        tris.iter()
            .map(|t| {
                let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
                ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b))
            })
            .fold(180.0, f64::min)
    }

    /// An edge band (a ring chain a sixteenth of the width inside a strip) is laid on the
    /// outline's nodes and its cell diagonals alternate along the whole outline: every chain node
    /// on a long side sits at the perpendicular foot of an outline node, and along each long
    /// side the apex of the triangle on an outline edge switches between the edge's two ends from
    /// one cell to the next (no run of one orientation that turns over midway, #155).
    #[test]
    /// Two constraint chains 0.05 apart at size 1 force slivers between
    /// them; snapped, they are one and the triangles keep their angles.
    fn nearly_coincident_chains_snap_together() {
        let mut r = Region2D::new(vec![[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]], 1);
        r.constraints = vec![
            vec![[2.0, 3.0], [10.0, 3.0], [18.0, 3.0]],
            vec![[2.0, 3.05], [18.0, 3.05]],
        ];
        let worst = |snap: f64| {
            let opts = Mesh2DOptions {
                snap,
                ..Default::default()
            };
            let m = mesh_2d(std::slice::from_ref(&r), |_p| 1.0, &opts);
            m.geom
                .min_angle
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        };
        let (apart, snapped) = (worst(0.0), worst(0.25));
        assert!(apart < 10.0, "chains apart: {apart} deg");
        assert!(snapped > 20.0, "snapped: {snapped} deg");
    }

    #[test]
    /// With `width_size`, a narrow strip meshed at a size far above its
    /// width gets cells about as long as it is wide.
    fn width_size_follows_the_trace() {
        let (len, w) = (200.0, 4.0);
        let strip = Region2D::new(vec![[0.0, 0.0], [len, 0.0], [len, w], [0.0, w]], 1);
        let mean_edge = |opts: &Mesh2DOptions| {
            let m = mesh_2d(std::slice::from_ref(&strip), |_p| 50.0, opts);
            let (mut sum, mut n) = (0.0, 0usize);
            for t in &m.tris {
                for k in 0..3 {
                    let (a, b) = (m.points[t[k] as usize], m.points[t[(k + 1) % 3] as usize]);
                    sum += (a[0] - b[0]).hypot(a[1] - b[1]);
                    n += 1;
                }
            }
            sum / n as f64
        };
        let coarse = mean_edge(&Mesh2DOptions::default());
        let fine = mean_edge(&Mesh2DOptions {
            width_size: 1.0,
            ..Default::default()
        });
        assert!(fine < 0.5 * coarse, "mean edge {fine} against {coarse}");
        assert!(
            fine > 0.3 * w && fine < 1.5 * w,
            "mean edge {fine} for width {w}"
        );
    }

    #[test]
    /// With the diagonals along, every band cell on a side leans the same
    /// way: no change of direction along the outline.
    fn edge_band_diagonals_along_turn_nowhere() {
        let (len, w) = (100.0, 10.0);
        let d = w / 16.0;
        let mut strip = Region2D::new(vec![[0.0, 0.0], [len, 0.0], [len, w], [0.0, w]], 1);
        strip.constraints = vec![vec![
            [d, d],
            [len - d, d],
            [len - d, w - d],
            [d, w - d],
            [d, d],
        ]];
        let opts = Mesh2DOptions {
            band_diagonals: BandDiagonals::Along,
            ..Default::default()
        };
        let m = mesh_2d(&[strip], |_p| w / 2.0, &opts);
        let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
        let on_row = |p: [f64; 2], y: f64| near(p[1], y) && p[0] > 5.0 && p[0] < len - 5.0;
        for (edge, row) in [(0.0, d), (w, w - d)] {
            let mut leans: Vec<bool> = Vec::new();
            for t in &m.tris {
                let pts = t.map(|i| m.points[i as usize]);
                let on_edge: Vec<[f64; 2]> =
                    pts.iter().copied().filter(|p| on_row(*p, edge)).collect();
                let apex: Vec<[f64; 2]> = pts.iter().copied().filter(|p| on_row(*p, row)).collect();
                if on_edge.len() == 2 && apex.len() == 1 {
                    leans.push(near(apex[0][0], on_edge[0][0].max(on_edge[1][0])));
                }
            }
            assert!(leans.len() >= 15, "only {} band cells", leans.len());
            assert!(
                leans.iter().all(|&l| l == leans[0]),
                "the diagonals turn on the side y = {edge}: {leans:?}"
            );
        }
    }

    #[test]
    fn edge_band_is_aligned_with_alternating_diagonals() {
        let (len, w) = (100.0, 10.0);
        let d = w / 16.0;
        let mut strip = Region2D::new(vec![[0.0, 0.0], [len, 0.0], [len, w], [0.0, w]], 1);
        strip.constraints = vec![vec![
            [d, d],
            [len - d, d],
            [len - d, w - d],
            [d, w - d],
            [d, d],
        ]];
        let m = mesh_2d(&[strip], |_p| w / 2.0, &Mesh2DOptions::default());
        let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
        let on_row = |p: [f64; 2], y: f64| near(p[1], y) && p[0] > 5.0 && p[0] < len - 5.0;
        for &q in &m.points {
            for (row, edge) in [(d, 0.0), (w - d, w)] {
                if on_row(q, row) {
                    assert!(
                        m.points
                            .iter()
                            .any(|&p| near(p[1], edge) && near(p[0], q[0])),
                        "chain node {q:?} has no outline node below it"
                    );
                }
            }
        }
        for (edge, row) in [(0.0, d), (w, w - d)] {
            // per band cell along this side: (edge midpoint, apex over the edge's right end)
            let mut cells: Vec<(f64, bool)> = Vec::new();
            for t in &m.tris {
                let pts = t.map(|i| m.points[i as usize]);
                let on_edge: Vec<[f64; 2]> =
                    pts.iter().copied().filter(|p| on_row(*p, edge)).collect();
                let apex: Vec<[f64; 2]> = pts.iter().copied().filter(|p| on_row(*p, row)).collect();
                if on_edge.len() == 2 && apex.len() == 1 {
                    let right = on_edge[0][0].max(on_edge[1][0]);
                    let left = on_edge[0][0].min(on_edge[1][0]);
                    assert!(
                        near(apex[0][0], right) || near(apex[0][0], left),
                        "apex {apex:?} not over the edge {on_edge:?}"
                    );
                    cells.push((0.5 * (left + right), near(apex[0][0], right)));
                }
            }
            cells.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert!(
                cells.len() >= 15,
                "only {} band cells on the side y = {edge}",
                cells.len()
            );
            for pair in cells.windows(2) {
                assert!(
                    pair[0].1 != pair[1].1,
                    "same diagonal twice in a row at {:?}",
                    pair
                );
            }
        }
    }

    #[test]
    fn mesh2d_meshes_a_tagged_square() {
        let sq = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 7);
        // Coarse input edges (length 1) with h = 0.34: only correct if the
        // endpoint resamples the boundary -- a pinned coarse edge gives slivers.
        let m = mesh_2d(&[sq], |_p| 0.34, &Mesh2DOptions::default());
        assert!(!m.tris.is_empty());
        // every triangle carries the region tag.
        assert!(m.tri_tags.iter().all(|&t| t == 7));
        // the triangulation tiles the square exactly: areas sum to 1.
        let area: f64 = m.geom.area.iter().sum();
        assert!((area - 1.0).abs() < 1e-6, "area sum {area}");
        // QUALITY: the boundary was resampled to h, so no pinned-coarse-edge
        // slivers -- the min angle clears a healthy bound.
        let mn = min_angle_deg(&m.points, &m.tris);
        assert!(
            mn > 20.0,
            "min angle {mn} too low (boundary not resampled?)"
        );
        // topology + RWG queries are present and consistent.
        assert!(!m.topo.edges.is_empty());
        assert!(m
            .rwg_candidate_edges()
            .iter()
            .all(|e| e[2] != NONE && e[3] != NONE));
        assert!(!m.boundary_edges().is_empty());
    }

    /// RWG-connected components: triangles linked by an inner edge (shared by two triangles).
    fn n_components(m: &Mesh2D) -> usize {
        let nt = m.tris.len();
        let mut parent: Vec<usize> = (0..nt).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for inc in &m.topo.edge_tris {
            if inc[1] != NONE {
                let (a, b) = (
                    find(&mut parent, inc[0] as usize),
                    find(&mut parent, inc[1] as usize),
                );
                if a != b {
                    parent[a] = b;
                }
            }
        }
        (0..nt)
            .map(|i| find(&mut parent, i))
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// THE FIX: two conductor regions sharing a full edge must mesh as ONE connected component
    /// (continuous RWG graph) while keeping BOTH conductor tags. Before the per-region union this
    /// produced two disjoint components — an open, non-conducting winding.
    #[test]
    fn abutting_regions_weld_into_one_component() {
        let a = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 1);
        let b = Region2D::new(vec![[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]], 2);
        let m = mesh_2d(&[a, b], |_p| 0.34, &Mesh2DOptions::default());
        assert_eq!(
            n_components(&m),
            1,
            "abutting regions must weld into one component"
        );
        assert!(
            m.tri_tags.contains(&1) && m.tri_tags.contains(&2),
            "both tags kept"
        );
        let area: f64 = m.geom.area.iter().sum();
        assert!((area - 2.0).abs() < 1e-6, "area {area}");
    }

    /// CONSTRAINT CHAINS: an inner square given as a closed chain and an open chain ending on
    /// the outline are element edges of the one triangulation — corners and chain ends are
    /// mesh vertices, sampled points on the chains lie on mesh edges, the region stays whole.
    #[test]
    fn constraint_chains_are_element_edges() {
        let inner = vec![[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.3, 0.3]];
        let open = vec![[0.0, 0.5], [0.3, 0.5]];
        let mut r = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 1);
        r.constraints = vec![inner.clone(), open.clone()];
        let m = mesh_2d(&[r], |_p| 0.15, &Mesh2DOptions::default());
        assert_eq!(n_components(&m), 1);
        let area: f64 = m.geom.area.iter().sum();
        assert!((area - 1.0).abs() < 1e-6, "area {area}");
        let has_vertex = |q: [f64; 2]| {
            m.points
                .iter()
                .any(|p| (p[0] - q[0]).abs() < 1e-9 && (p[1] - q[1]).abs() < 1e-9)
        };
        for q in inner.iter().chain(open.iter()) {
            assert!(has_vertex(*q), "chain vertex {q:?} missing");
        }
        let on_edge = |q: [f64; 2]| -> bool {
            m.tris.iter().any(|t| {
                (0..3).any(|k| {
                    let (a, b) = (m.points[t[k] as usize], m.points[t[(k + 1) % 3] as usize]);
                    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                    let l2 = dx * dx + dy * dy;
                    let s = ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / l2;
                    (-1e-9..=1.0 + 1e-9).contains(&s)
                        && ((q[0] - a[0]) - s * dx).hypot((q[1] - a[1]) - s * dy) < 1e-9
                })
            })
        };
        for i in 0..10 {
            let t = 0.3 + 0.4 * (i as f64 + 0.5) / 10.0;
            assert!(on_edge([t, 0.3]), "chain point ({t}, 0.3) inside a cell");
            assert!(on_edge([0.7, t]), "chain point (0.7, {t}) inside a cell");
            assert!(
                on_edge([0.03 * i as f64, 0.5]),
                "open chain point inside a cell"
            );
        }
        // The chains do not change membership: no boundary edge strictly inside.
        for e in m.topo.boundary_edges() {
            let (a, b) = (m.points[e[0] as usize], m.points[e[1] as usize]);
            let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
            assert!(
                !(mid[0] > 1e-6 && mid[0] < 1.0 - 1e-6 && mid[1] > 1e-6 && mid[1] < 1.0 - 1e-6),
                "boundary edge inside the region at {mid:?}"
            );
        }
    }

    /// Partial-edge abutment (B meets only the middle of A's edge) — the case the old vertex weld
    /// could not handle — must also weld via the union.
    #[test]
    fn partial_edge_abut_welds() {
        let a = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 1);
        let b = Region2D::new(vec![[1.0, 0.25], [2.0, 0.25], [2.0, 0.75], [1.0, 0.75]], 2);
        let m = mesh_2d(&[a, b], |_p| 0.2, &Mesh2DOptions::default());
        assert_eq!(n_components(&m), 1, "partial-edge abutment must weld");
    }

    /// Non-touching regions stay separate (two components, two tags).
    #[test]
    fn separate_regions_stay_disconnected() {
        let a = Region2D::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], 1);
        let b = Region2D::new(vec![[3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0]], 2);
        let m = mesh_2d(&[a, b], |_p| 0.34, &Mesh2DOptions::default());
        assert_eq!(n_components(&m), 2, "non-touching regions stay separate");
    }

    /// Count-driven meshing: a triangle budget lands the mesh near the requested count, regardless
    /// of the (here coarse) sizing field.
    #[test]
    fn target_count_budgets_the_mesh() {
        let sq = Region2D::new(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], 1);
        for target in [200usize, 800] {
            let opts = Mesh2DOptions {
                target_count: target,
                ..Default::default()
            };
            let m = mesh_2d(std::slice::from_ref(&sq), |_p| 5.0, &opts); // coarse field; the budget drives it
            let n = m.tris.len();
            assert!(
                (n as f64) > 0.6 * target as f64 && (n as f64) < 1.6 * target as f64,
                "budget {target}: got {n} triangles"
            );
            let area: f64 = m.geom.area.iter().sum();
            assert!((area - 100.0).abs() < 1e-6, "area {area}");
        }
    }

    /// GLOBAL budget across GROUPS (layers): the total triangle count lands near the requested
    /// budget, every group is meshed (its own points/tris, correctly tagged), and groups on
    /// overlapping planes never merge — the count-driven `mesh_layers` contract.
    #[test]
    fn mesh_layers_shares_one_global_budget() {
        // Two layers, geometrically OVERLAPPING in the plane (different metals): a big square on
        // group 0, a smaller square at the same xy on group 1. They must NOT merge.
        let big = Region2D::new(vec![[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]], 1);
        let small = Region2D::new(vec![[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], 2);
        for &budget in &[2000usize, 8000] {
            let opts = Mesh2DOptions {
                target_count: budget,
                ..Default::default()
            };
            let ms = mesh_layers(&[vec![big.clone()], vec![small.clone()]], |_q| 1.0, &opts);
            assert_eq!(ms.len(), 2, "one mesh per group");
            let n: usize = ms.iter().map(|m| m.tris.len()).sum();
            assert!(
                (n as f64) > 0.7 * budget as f64 && (n as f64) < 1.4 * budget as f64,
                "budget {budget}: got {n}"
            );
            assert!(
                ms[0].tri_tags.iter().all(|&t| t == 1) && !ms[0].tris.is_empty(),
                "group 0 tagged 1"
            );
            assert!(
                ms[1].tri_tags.iter().all(|&t| t == 2) && !ms[1].tris.is_empty(),
                "group 1 tagged 2"
            );
            // areas are preserved per group (400 and 100), proving the un-translation round-trips.
            let a0: f64 = ms[0].geom.area.iter().sum();
            let a1: f64 = ms[1].geom.area.iter().sum();
            assert!(
                (a0 - 400.0).abs() < 1e-3 && (a1 - 100.0).abs() < 1e-3,
                "areas {a0} {a1}"
            );
            // bigger layer draws more of the shared budget (area-proportional emergence).
            assert!(
                ms[0].tris.len() > ms[1].tris.len(),
                "big layer takes more budget"
            );
        }
    }

    /// Read-back bin assignment regression (a sky130 opamp met2 serpentine): the union of
    /// abutting rectangles puts resampled boundary vertices EXACTLY on their packing-bin edge, and
    /// an fp-ulp of rounding used to tip them into the neighbouring bin — un-translated with the
    /// wrong offset, they came back displaced by a whole bin pitch (stray nodes ~45 um outside the
    /// layout, giant sliver triangles). Every output vertex must stay inside the input bbox.
    #[test]
    fn mesh_layers_readback_keeps_vertices_in_input_bbox() {
        // 38 abutting met2 rectangles (sky130 opamp serpentine power rail, um) — the minimal
        // real-layout set (ddmin) that reproduced the displacement at h = 0.5.
        let rects: [[f64; 4]; 38] = [
            [-20.405, 28.955, 11.995, 28.960],
            [12.265, 28.450, 13.415, 29.955],
            [-20.405, 27.755, 13.415, 28.450],
            [12.265, 26.250, 13.415, 27.755],
            [-20.405, 25.555, 13.415, 26.250],
            [12.265, 24.075, 13.415, 25.555],
            [-20.405, 23.380, 13.415, 24.075],
            [12.265, 21.950, 13.415, 23.380],
            [-20.405, 21.255, 13.415, 21.950],
            [12.265, 20.010, 13.415, 21.255],
            [12.350, 19.870, 13.415, 20.010],
            [12.265, 18.850, 13.415, 19.870],
            [-20.405, 18.155, 13.415, 18.850],
            [12.265, 16.675, 13.415, 18.155],
            [-20.405, 15.980, 13.415, 16.675],
            [12.265, 14.500, 13.415, 15.980],
            [-20.405, 13.805, 13.415, 14.500],
            [12.265, 12.325, 13.415, 13.805],
            [-20.405, 11.630, 13.415, 12.325],
            [-21.810, 9.150, -20.700, 10.630],
            [12.265, 10.150, 13.415, 11.630],
            [-20.405, 9.455, 13.415, 10.150],
            [-3.915, 3.185, 4.105, 4.235],
            [4.885, 3.260, 5.410, 5.885],
            [12.265, 3.165, 13.415, 9.455],
            [-17.225, 2.180, -10.485, 2.455],
            [-6.885, 2.170, 9.875, 2.445],
            [-13.975, 1.865, -13.210, 1.875],
            [3.585, -1.215, 4.025, 0.140],
            [5.140, -1.650, 5.575, 0.140],
            [7.290, -1.685, 7.645, 0.140],
            [12.265, -4.015, 13.495, 3.165],
            [-20.460, -4.710, 13.495, -4.015],
            [12.185, -6.105, 13.495, -4.710],
            [-20.405, -6.800, 13.495, -6.105],
            [12.185, -8.195, 13.495, -6.800],
            [-21.960, -12.360, -20.605, -9.890],
            [7.865, -17.455, 13.525, -16.100],
        ];
        // SI metres — the units a solver feeds in; the fp tipping is
        // scale-dependent (it did NOT reproduce on the same coordinates in um).
        const UM: f64 = 1e-6;
        let regions: Vec<Region2D> = rects
            .iter()
            .enumerate()
            .map(|(i, &[x0, y0, x1, y1])| {
                Region2D::new(
                    vec![
                        [x0 * UM, y0 * UM],
                        [x1 * UM, y0 * UM],
                        [x1 * UM, y1 * UM],
                        [x0 * UM, y1 * UM],
                    ],
                    i as i64,
                )
            })
            .collect();
        let m = mesh_2d(&regions, |_p| 0.5 * UM, &Mesh2DOptions::default());
        assert!(!m.tris.is_empty());
        let (lo, hi) = ([-21.960 * UM, -17.455 * UM], [13.525 * UM, 29.955 * UM]);
        let tol = 1e-6 * UM;
        let stray: Vec<&[f64; 2]> = m
            .points
            .iter()
            .filter(|p| {
                p[0] < lo[0] - tol || p[0] > hi[0] + tol || p[1] < lo[1] - tol || p[1] > hi[1] + tol
            })
            .collect();
        assert!(
            stray.is_empty(),
            "{} vertices outside the input bbox, e.g. {:?}",
            stray.len(),
            stray.first()
        );
    }

    #[test]
    fn mesh3d_bundles_everything() {
        let mesh = TetMesh {
            points: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            tets: vec![[0, 1, 2, 3]],
            tet_regions: vec![RegionTag(1)],
            faces: Vec::new(),
            surfaces: Vec::new(),
            surface_owners: Vec::new(),
            plc_points: 4,
            point_size: Vec::new(),
            point_class: vec![rapidmesh_tet::PointClass::Interior; 4],
            curve_edges: Vec::new(),
            periodic_points: Vec::new(),
            contact_faces: Vec::new(),
        };
        let v = Mesh3D::build(mesh);
        assert_eq!(v.topo.edges.len(), 6);
        assert_eq!(v.topo.faces.len(), 4);
        assert_eq!(v.geom.volume.len(), 1);
        assert!((v.geom.volume[0] - 1.0 / 6.0).abs() < 1e-12);
    }
}
