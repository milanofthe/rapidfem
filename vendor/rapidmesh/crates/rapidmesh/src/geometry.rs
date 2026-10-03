//! The geometry builder: solids and sheets, their labels, the sizing
//! hierarchy, named faces and edges and periodic face pairs, and the two
//! meshing calls.
//!
//! Solids overlap by priority: a solid added later carves its region out of
//! the earlier ones. Sheets are zero-thickness faces embedded into the
//! volume mesh with an integer tag (PEC traces, ports).

mod ops;

pub use ops::{Object, SheetRef, Transform};

use crate::features::{EdgeCut, EdgePick};
use crate::mesh::{Labels, Mesh, Run, SolidInfo, SurfaceMesh};
use crate::shapes::{Shape, Sheet};
use crate::{Error, Result};
use rapidmesh_brep::{EdgeFilter, FaceFilter, Model, Topology};
use rapidmesh_exact::clock::Instant;
use rapidmesh_geom::{FaceTag, RegionTag, Scene};
use rapidmesh_tet::mesh3::PeriodicPair;
use rapidmesh_tet::{mesh_budgeted, quality_stats, MeshParams};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

type P3 = [f64; 3];

/// A solid added to a [`Geometry`]: `region` tags its tets, `index` (the
/// insertion order, voids included) its surfaces. Voids share region 0 but
/// keep their own index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Solid {
    pub region: u32,
    pub index: u32,
}

/// The dimension a [`Scope`] selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Region,
    Surf,
    Edge,
}

/// A selection of regions, geometric faces or geometric edges, narrowed
/// from region to face to edge:
///
/// ```
/// use rapidmesh::{EdgeFilter, FaceFilter, Scope};
/// // the edges near the origin of the top faces of region 2
/// let s = Scope::region(Some(2))
///     .surfs(Some(FaceFilter::normal([0.0, 0.0, 1.0])))
///     .edges(Some(EdgeFilter::near([0.0; 3])));
/// ```
///
/// A scope without any filter at its level and above is unfiltered: sizing
/// it sets the default of its dimension instead of per-entity values.
#[derive(Clone, Debug)]
pub struct Scope {
    pub level: Level,
    pub region: Option<u32>,
    pub face: Option<FaceFilter>,
    pub edge: Option<EdgeFilter>,
}

impl Scope {
    /// Region `tag`, or every region.
    pub fn region(tag: Option<u32>) -> Scope {
        Scope {
            level: Level::Region,
            region: tag,
            face: None,
            edge: None,
        }
    }

    /// The faces `filter` selects, or every face.
    pub fn surf(filter: Option<FaceFilter>) -> Scope {
        Scope::region(None).surfs(filter)
    }

    /// The edges `filter` selects, or every edge.
    pub fn edge(filter: Option<EdgeFilter>) -> Scope {
        Scope::region(None).edges(filter)
    }

    /// The faces of this scope's regions that `filter` selects.
    pub fn surfs(self, filter: Option<FaceFilter>) -> Scope {
        Scope {
            level: Level::Surf,
            face: filter,
            ..self
        }
    }

    /// The edges of this scope's faces that `filter` selects.
    pub fn edges(self, filter: Option<EdgeFilter>) -> Scope {
        Scope {
            level: Level::Edge,
            edge: filter,
            ..self
        }
    }

    fn unfiltered(&self) -> bool {
        self.region.is_none() && self.face.is_none() && self.edge.is_none()
    }
}

/// The sizing hierarchy: per-dimension defaults and per-entity values.
#[derive(Clone, Debug)]
struct Sizing {
    tol_edge: f64,
    tol_surf: f64,
    maxh_edge: f64,
    maxh_surf: f64,
    maxh_vol: f64,
    /// Per-entity values as given: the selection, what is set and the
    /// value, resolved on the model when a mesh is made (a later one wins).
    scoped: Vec<(Scope, Knob, f64)>,
}

/// What a scoped value sets.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Knob {
    Maxh,
    Tol,
}

/// The per-entity values of the sizing, resolved on one model.
#[derive(Default)]
struct Resolved {
    edge_maxh: BTreeMap<u32, f64>,
    edge_tol: BTreeMap<u32, f64>,
    surf_maxh: BTreeMap<u32, f64>,
    surf_tol: BTreeMap<u32, f64>,
    region_maxh: BTreeMap<u32, f64>,
}

impl Default for Sizing {
    fn default() -> Sizing {
        // A chord of a curved edge or surface deviates by at most this
        // share of its radius: `h = R sqrt(8 tol)`, about ten segments round
        // a circle (the density meshes of wires and vias have been built
        // to; 1e-2 asks for twice the segments and four times the facets).
        Sizing {
            tol_edge: 5e-2,
            tol_surf: 5e-2,
            maxh_edge: f64::INFINITY,
            maxh_surf: f64::INFINITY,
            maxh_vol: f64::INFINITY,
            scoped: Vec::new(),
        }
    }
}

/// Elements across the thickness of a region by default.
pub const DEFAULT_CELLS_ACROSS: f64 = 1.0;

/// Options of [`Geometry::mesh`]; a `None` falls back to the geometry.
#[derive(Clone, Debug)]
pub struct MeshOptions {
    /// Target edge length (the geometry's if `None`, unbounded if neither).
    pub maxh: Option<f64>,
    /// Delaunay quality bound, circumradius over shortest edge.
    pub radius_edge: f64,
    /// Refinement point budget.
    pub max_points: usize,
    /// Size grading: the target grows by at most this much per unit
    /// distance from finer features.
    pub grading: Option<f64>,
    /// Elements across the thickness of each region (see
    /// [`MeshParams::cells_across`]): the size inside a region is at most
    /// its thickness over this, so a thin plate or wire gets proper tets
    /// through it. `None` (the default) leaves it off for the bottom-up
    /// mesher (stacks of layers far thinner than the size take flat tets
    /// through each layer) and takes [`DEFAULT_CELLS_ACROSS`] for the
    /// restricted Delaunay one; `Some(0.0)` turns it off for both.
    pub cells_across: Option<f64>,
    /// The mesher: `None` (the default) meshes bottom-up (each face alone
    /// on the shared samples of its edges, then each region by its
    /// constrained Delaunay tetrahedralization, refined and improved) and
    /// falls back to the restricted Delaunay mesher where bottom-up fails; `Some(true)` meshes
    /// bottom-up only, `Some(false)` with the restricted Delaunay mesher.
    pub bottom_up: Option<bool>,
    /// Relative chord tolerances of curved edges and surfaces.
    pub tol_edge: Option<f64>,
    pub tol_surf: Option<f64>,
    /// Largest element size per dimension, each with `maxh` as the minimum.
    pub maxh_edge: Option<f64>,
    pub maxh_surf: Option<f64>,
    pub maxh_vol: Option<f64>,
    /// Tet budget: the global size is scaled over a few remeshes to land
    /// near it.
    pub target_elements: Option<usize>,
    /// Smallest element size on surfaces and in the volume (0 off).
    pub min_h_surf: f64,
    pub min_h_vol: f64,
}

impl Default for MeshOptions {
    fn default() -> MeshOptions {
        MeshOptions {
            maxh: None,
            radius_edge: 2.0,
            max_points: 500_000,
            grading: None,
            cells_across: None,
            bottom_up: None,
            tol_edge: None,
            tol_surf: None,
            maxh_edge: None,
            maxh_surf: None,
            maxh_vol: None,
            target_elements: None,
            min_h_surf: 0.0,
            min_h_vol: 0.0,
        }
    }
}

/// Options of [`Geometry::surface_mesh`] (see [`MeshOptions`]).
#[derive(Clone, Debug, Default)]
pub struct SurfaceOptions {
    pub maxh: Option<f64>,
    pub grading: Option<f64>,
    pub tol_edge: Option<f64>,
    pub tol_surf: Option<f64>,
    pub maxh_edge: Option<f64>,
    pub maxh_surf: Option<f64>,
    pub maxh_vol: Option<f64>,
    /// Triangle budget: the sizes are coarsened by one factor until the
    /// count is at most a little over it.
    pub target_triangles: Option<usize>,
}

/// The geometry builder.
///
/// ```no_run
/// use rapidmesh::shapes::{Cuboid, Sheet};
/// use rapidmesh::{Geometry, MeshOptions};
/// let mut g = Geometry::new(Some(0.9));
/// let _air = g.add(Cuboid::new([4.0, 4.0, 4.0]))?;
/// let diel = g.add_solid(Cuboid::new([2.0, 2.0, 1.0]).at([1.0, 1.0, 1.0]), Some(0.45), false)?;
/// g.label_solid(diel, "substrate");
/// g.add_sheet(&Sheet::xy(1.0, 1.0, [1.5, 1.5, 2.0]), 7, None)?;
/// let mesh = g.mesh(&MeshOptions::default())?;
/// mesh.write_msh("cell.msh")?;
/// # Ok::<(), rapidmesh::Error>(())
/// ```
pub struct Geometry {
    scene: Scene,
    /// The model of `scene`, built on first use and shared by the
    /// selectors and both meshing calls, so an entity id means the same
    /// everywhere. Every change of the scene drops it.
    model: OnceLock<std::result::Result<Arc<Model>, String>>,
    maxh: Option<f64>,
    grading: f64,
    /// Target size per region, as given with the solid.
    solid_maxh: Vec<(u32, f64)>,
    /// Target size per sheet tag.
    face_maxh: BTreeMap<u32, f64>,
    /// Target size on the surfaces of a solid, per solid index.
    surface_maxh: BTreeMap<u32, f64>,
    size_points: Vec<(P3, f64)>,
    sizing: Sizing,
    /// Periodic face pairs as given: master and slave selections and the
    /// shift, paired on the model when a mesh is made.
    periodic: Vec<(Scope, Scope, P3)>,
    /// Named faces and edges as given, resolved when a mesh is made.
    named: Vec<(String, Scope)>,
    labels: Labels,
    /// Per sheet, parallel to the scene's: how it was given and the
    /// transforms applied to it since, in order.
    sheets: Vec<(Sheet, Vec<Transform>)>,
}

impl Default for Geometry {
    fn default() -> Geometry {
        Geometry::new(None)
    }
}

impl Geometry {
    /// An empty geometry with global target size `maxh` and grading 0.5.
    pub fn new(maxh: Option<f64>) -> Geometry {
        Geometry {
            scene: Scene::new(),
            model: OnceLock::new(),
            maxh,
            grading: 0.5,
            solid_maxh: Vec::new(),
            face_maxh: BTreeMap::new(),
            surface_maxh: BTreeMap::new(),
            size_points: Vec::new(),
            sizing: Sizing::default(),
            periodic: Vec::new(),
            named: Vec::new(),
            labels: Labels::default(),
            sheets: Vec::new(),
        }
    }

    /// The scene, for a change: the model built from it is stale after.
    fn scene_mut(&mut self) -> &mut Scene {
        self.model = OnceLock::new();
        &mut self.scene
    }

    /// The model of the current scene (PLC and B-rep), built on first use,
    /// or the input the scene cannot be assembled from.
    pub fn model(&self) -> Result<Arc<Model>> {
        self.model
            .get_or_init(|| {
                Model::try_of_scene(&self.scene)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            })
            .clone()
            .map_err(Error::Invalid)
    }

    /// The regions, faces and edges of the model, by the ids the scopes,
    /// named sets and periodic pairs use.
    pub fn topology(&self) -> Result<Topology> {
        Ok(self.model()?.topology())
    }

    pub fn maxh(&self) -> Option<f64> {
        self.maxh
    }

    pub fn set_maxh(&mut self, h: f64) {
        self.maxh = Some(h);
    }

    pub fn grading(&self) -> f64 {
        self.grading
    }

    pub fn set_grading(&mut self, g: f64) {
        self.grading = g;
    }

    /// The chord tolerance of curved edges and surfaces.
    pub fn set_tol(&mut self, tol: f64) {
        self.sizing.tol_edge = tol;
        self.sizing.tol_surf = tol;
    }

    /// The solids with their labels, sheet labels and named entities, the
    /// names resolved on the current model.
    pub fn labels(&self) -> Result<Labels> {
        let mut labels = self.labels.clone();
        for (name, scope) in &self.named {
            let ids = self.resolve(scope)?;
            let list = match scope.level {
                Level::Surf => &mut labels.face_names,
                _ => &mut labels.edge_names,
            };
            let i = match list.iter().position(|(n, _)| n == name) {
                Some(i) => i,
                None => {
                    list.push((name.clone(), Vec::new()));
                    list.len() - 1
                }
            };
            for id in ids {
                if !list[i].1.contains(&id) {
                    list[i].1.push(id);
                }
            }
        }
        Ok(labels)
    }

    // ---- solids and sheets ----------------------------------------------

    /// Adds `shape` as a material solid with the global target size.
    pub fn add(&mut self, shape: impl Into<Shape>) -> Result<Solid> {
        self.add_solid(shape, None, false)
    }

    /// Cuts `shape` out of every solid added before it: its walls become
    /// boundary faces.
    pub fn cut(&mut self, shape: impl Into<Shape>) -> Result<Solid> {
        self.add_solid(shape, None, true)
    }

    /// Adds `shape` with target size `maxh` in its region, as material or
    /// (`void`) cut out.
    pub fn add_solid(
        &mut self,
        shape: impl Into<Shape>,
        maxh: Option<f64>,
        void: bool,
    ) -> Result<Solid> {
        let shape: Shape = shape.into();
        let roles = shape.role_names();
        let f = shape.faceted(maxh.or(self.maxh))?;
        Ok(self.add_faceted(f, roles, maxh, void))
    }

    /// The solids of the STEP file at `path` (AP203/AP214), each with its
    /// faces on their true surfaces (planes, quadrics, tori, B-splines), with
    /// target size `maxh` in their regions. Coordinates stay in the file's
    /// unit.
    pub fn import_step(
        &mut self,
        path: impl AsRef<std::path::Path>,
        maxh: Option<f64>,
    ) -> Result<Vec<Solid>> {
        let p = path.as_ref();
        let text = std::fs::read_to_string(p)
            .map_err(|e| Error::Invalid(format!("{}: {e}", p.display())))?;
        let step = rapidmesh_step::read(&text, rapidmesh_step::Tolerance::default())
            .map_err(|e| Error::Invalid(format!("{}: {e}", p.display())))?;
        Ok(step
            .bodies
            .into_iter()
            .map(|b| self.add_faceted(b.solid, Vec::new(), maxh, false))
            .collect())
    }

    /// Adds the solid `f` with the names of its face roles.
    fn add_faceted(
        &mut self,
        f: rapidmesh_geom::Faceted,
        roles: Vec<String>,
        maxh: Option<f64>,
        void: bool,
    ) -> Solid {
        let region = if void {
            self.scene_mut().add_void(f);
            0
        } else {
            let r = self.scene_mut().add_solid(f).0;
            if let Some(h) = maxh {
                self.solid_maxh.push((r, h));
            }
            r
        };
        let index = self.labels.solids.len() as u32;
        self.labels.solids.push(SolidInfo {
            region,
            label: None,
            roles,
        });
        Solid { region, index }
    }

    /// The names of `solid`'s faces by role (empty where a role has none).
    pub fn roles(&self, solid: Solid) -> &[String] {
        self.labels
            .solids
            .get(solid.index as usize)
            .map_or(&[], |s| s.roles.as_slice())
    }

    /// The role of `solid`'s face called `name`.
    pub fn role(&self, solid: Solid, name: &str) -> Result<u32> {
        let roles = self.roles(solid);
        roles
            .iter()
            .position(|r| r == name)
            .map(|i| i as u32)
            .ok_or_else(|| Error::Invalid(format!("role {name:?} is not one of {roles:?}")))
    }

    /// Chamfers or fillets (`cut`) the `edges` of `solid`. The material
    /// comes off that solid alone, and what lies under it fills the cut;
    /// the new faces (plane or cone, cylinder or torus) become faces of the
    /// solid, with roles after its own. With `void` the solid stays and the
    /// material is carved out as voids instead, which leaves the cut empty
    /// (a countersink or a round in the rim of a hole); the new faces are
    /// then the voids'. Straight edges between planes and circles between a
    /// plane square to their axis, a cylinder or a cone take one. Returns
    /// the origin (solid, role) of every new face, one per edge cut. The new
    /// faces are named after the cut, `chamfer0`, `chamfer1`, ... (numbered
    /// on over later cuts), on a void `chamfer` (or `fillet`).
    pub fn cut_edges(
        &mut self,
        solid: Solid,
        edges: &[EdgePick],
        cut: EdgeCut,
        void: bool,
    ) -> Result<Vec<(Solid, u32)>> {
        let i = solid.index as usize;
        let f = self
            .scene
            .solid(i)
            .ok_or_else(|| Error::Invalid(format!("no solid {i}")))?
            .clone();
        if solid.region == 0 {
            return Err(Error::Invalid("a void has no edges to cut".into()));
        }
        let maxh = self
            .solid_maxh
            .iter()
            .find(|(r, _)| *r == solid.region)
            .map(|&(_, h)| h)
            .or(self.maxh);
        let model = self.model()?;
        let done =
            crate::features::cut_edges(&model, solid.index, solid.region, &f, edges, cut, maxh)?;
        if void {
            // The solid stays; the material it would lose is carved out as
            // a void, which also empties what lies under it (a second copy
            // of the new faces on the solid would only coincide with the
            // void's).
            // A piece is the solid's shape with one cutter's surfaces after
            // it: the new face follows the solid's own surfaces.
            let role = f.surfaces.len() as u32;
            let mut faces = Vec::with_capacity(done.removed.len());
            for piece in done.removed {
                let carved = Solid {
                    region: 0,
                    index: self.labels.solids.len() as u32,
                };
                self.scene_mut().add_void(piece);
                // The piece has the solid's faces, then the new one.
                let mut roles = self.roles(solid).to_vec();
                name_role(&mut roles, role, cut.name());
                self.labels.solids.push(SolidInfo {
                    region: 0,
                    label: None,
                    roles,
                });
                faces.push((carved, role));
            }
            return Ok(faces);
        }
        self.scene_mut().replace_solid(i, done.shape);
        let mut roles = self.roles(solid).to_vec();
        let first = roles.iter().filter(|r| r.starts_with(cut.name())).count();
        for (k, &r) in done.roles.iter().enumerate() {
            name_role(&mut roles, r, &format!("{}{}", cut.name(), first + k));
        }
        self.labels.solids[i].roles = roles;
        Ok(done.roles.into_iter().map(|r| (solid, r)).collect())
    }

    /// [`Geometry::cut_edges`] with a chamfer `distance` into both faces.
    pub fn chamfer(
        &mut self,
        solid: Solid,
        edges: &[EdgePick],
        distance: f64,
        void: bool,
    ) -> Result<Vec<(Solid, u32)>> {
        self.cut_edges(solid, edges, EdgeCut::Chamfer(distance), void)
    }

    /// [`Geometry::cut_edges`] with a fillet of `radius`.
    pub fn fillet(
        &mut self,
        solid: Solid,
        edges: &[EdgePick],
        radius: f64,
        void: bool,
    ) -> Result<Vec<(Solid, u32)>> {
        self.cut_edges(solid, edges, EdgeCut::Fillet(radius), void)
    }

    /// Embeds `sheet` with face tag `tag` and (the smallest given) target
    /// size `maxh` on the tag.
    pub fn add_sheet(&mut self, sheet: &Sheet, tag: u32, maxh: Option<f64>) -> Result<SheetRef> {
        let f = sheet.faceted()?;
        if let Some(h) = maxh {
            let e = self.face_maxh.entry(tag).or_insert(h);
            *e = e.min(h);
        }
        self.scene_mut().add_sheet(f, FaceTag(tag));
        self.sheets.push((sheet.clone(), Vec::new()));
        Ok(SheetRef {
            index: self.sheets.len() as u32 - 1,
            tag,
        })
    }

    /// Fuses overlapping solids into one material: the faces between them
    /// go, the outer union surface stays. Returns the first solid, now the
    /// merged region.
    pub fn union(&mut self, solids: &[Solid]) -> Result<Solid> {
        let (&keep, rest) = solids
            .split_first()
            .ok_or_else(|| Error::Invalid("union needs at least one solid".into()))?;
        for s in rest {
            self.scene_mut()
                .merge_region(RegionTag(keep.region), RegionTag(s.region));
            for (r, _) in &mut self.solid_maxh {
                if *r == s.region {
                    *r = keep.region;
                }
            }
            for info in &mut self.labels.solids {
                if info.region == s.region {
                    info.region = keep.region;
                }
            }
        }
        Ok(keep)
    }

    /// Names a solid: its cells form the set and physical group `name`
    /// (solids of one name are one group).
    pub fn label_solid(&mut self, solid: Solid, name: &str) {
        if let Some(info) = self.labels.solids.get_mut(solid.index as usize) {
            info.label = Some(name.to_string());
        }
    }

    /// Names the faces of sheet tag `tag`.
    pub fn label_tag(&mut self, tag: u32, name: &str) {
        self.labels.tag_labels.insert(tag, name.to_string());
    }

    // ---- sizing ------------------------------------------------------------

    /// Target size `h` on every surface of `solid` (the only handle on a
    /// void's walls), recovering along the grading.
    pub fn refine_surface(&mut self, solid: Solid, h: f64) {
        let e = self.surface_maxh.entry(solid.index).or_insert(h);
        *e = e.min(h);
    }

    /// Target size `h` at point `p`, recovering along the grading.
    pub fn add_size_point(&mut self, p: P3, h: f64) {
        self.size_points.push((p, h));
    }

    /// [`Geometry::add_size_point`] for each point with its own size.
    pub fn add_size_points(&mut self, points: &[P3], hs: &[f64]) -> Result<()> {
        if points.len() != hs.len() {
            return Err(Error::Invalid(format!(
                "{} points but {} sizes",
                points.len(),
                hs.len()
            )));
        }
        self.size_points
            .extend(points.iter().copied().zip(hs.iter().copied()));
        Ok(())
    }

    /// The MARK -> REFINE half of an adaptive loop: Dörfler-marks the
    /// triangles of `mesh` by their indicators `eta` and adds each marked one
    /// as a size point at its centroid, its local size over `d.factor`.
    /// Returns the marked triangles; mesh again to refine.
    pub fn mark_dorfler(
        &mut self,
        mesh: &rapidmesh_tet::SurfaceMesh,
        eta: &[f64],
        d: &rapidmesh_tet::Dorfler,
    ) -> Result<Vec<u32>> {
        if eta.len() != mesh.faces.len() {
            return Err(Error::Invalid(format!(
                "{} indicators for {} triangles",
                eta.len(),
                mesh.faces.len()
            )));
        }
        let (marked, points, hs) = mesh.dorfler_size_points(eta, d.theta, d.factor, d.h_min);
        self.add_size_points(&points, &hs)?;
        Ok(marked)
    }

    /// The ids `scope` selects in the current model.
    pub fn resolve(&self, scope: &Scope) -> Result<Vec<u32>> {
        let topo = self.topology()?;
        Ok(match scope.level {
            Level::Region => topo.resolve_regions(scope.region),
            Level::Surf => topo.resolve_faces(
                scope.region,
                scope.face.as_ref().unwrap_or(&FaceFilter::default()),
            ),
            Level::Edge => topo.resolve_edges(
                scope.region,
                scope.face.as_ref(),
                scope.edge.as_ref().unwrap_or(&EdgeFilter::default()),
            ),
        })
    }

    /// Target size on what `scope` selects; unfiltered, the default of its
    /// dimension. A per-entity value beats the default.
    pub fn set_maxh_on(&mut self, scope: &Scope, h: f64) -> Result<()> {
        let s = &mut self.sizing;
        if scope.unfiltered() {
            match scope.level {
                Level::Edge => s.maxh_edge = h,
                Level::Surf => s.maxh_surf = h,
                Level::Region => s.maxh_vol = h,
            }
            return Ok(());
        }
        self.sizing.scoped.push((scope.clone(), Knob::Maxh, h));
        Ok(())
    }

    /// Chord tolerance on what `scope` selects (regions have none: the
    /// volume follows the surface).
    pub fn set_tol_on(&mut self, scope: &Scope, tol: f64) -> Result<()> {
        if scope.level == Level::Region {
            return Err(Error::Invalid(
                "regions have no tolerance (volume follows the surface)".into(),
            ));
        }
        if scope.unfiltered() {
            match scope.level {
                Level::Edge => self.sizing.tol_edge = tol,
                _ => self.sizing.tol_surf = tol,
            }
            return Ok(());
        }
        self.sizing.scoped.push((scope.clone(), Knob::Tol, tol));
        Ok(())
    }

    /// Names the faces or edges `scope` selects (a port, a boundary
    /// condition): a set of the mesh and a physical group of the MSH file.
    /// The selection is kept and resolved when a mesh is made; it must
    /// match something now.
    pub fn name(&mut self, scope: &Scope, name: &str) -> Result<()> {
        let names = match scope.level {
            Level::Region => {
                return Err(Error::Invalid(
                    "name regions with label_solid(solid, name)".into(),
                ))
            }
            Level::Surf => "surf",
            Level::Edge => "edge",
        };
        if self.resolve(scope)?.is_empty() {
            return Err(Error::Invalid(format!(
                "the selection for {name:?} matches no {names}"
            )));
        }
        self.named.push((name.to_string(), scope.clone()));
        Ok(())
    }

    /// Meshes the faces `slave` selects with the same triangles as the
    /// faces `master` selects: every slave face is a master face moved by
    /// `shift` (default: the difference of the area-weighted centroids).
    /// Once per direction of a unit cell, on a complete geometry. Returns
    /// the shift.
    pub fn periodic(&mut self, master: &Scope, slave: &Scope, shift: Option<P3>) -> Result<P3> {
        let (_, t) = self.pair_faces(master, slave, shift)?;
        self.periodic.push((master.clone(), slave.clone(), t));
        Ok(t)
    }

    /// The faces `master` and `slave` select on the current model, paired
    /// by `shift` (default: the difference of their area-weighted
    /// centroids), and the shift.
    fn pair_faces(
        &self,
        master: &Scope,
        slave: &Scope,
        shift: Option<P3>,
    ) -> Result<(Vec<PeriodicPair>, P3)> {
        if master.level != Level::Surf || slave.level != Level::Surf {
            return Err(Error::Invalid(
                "periodic takes two face selections (surf scopes)".into(),
            ));
        }
        let (a, b) = (self.resolve(master)?, self.resolve(slave)?);
        if a.is_empty() || b.is_empty() {
            return Err(Error::Invalid(
                "a periodic selection matches no face".into(),
            ));
        }
        let faces = self.topology()?.faces;
        let cen = |i: u32| faces[i as usize].centroid;
        let area = |i: u32| faces[i as usize].area;
        let centroid = |ids: &[u32]| {
            let w: f64 = ids.iter().map(|&i| area(i)).sum();
            let mut c = [0.0; 3];
            for &i in ids {
                for k in 0..3 {
                    c[k] += area(i) * cen(i)[k];
                }
            }
            c.map(|x| x / w)
        };
        let t = shift.unwrap_or_else(|| {
            let (ca, cb) = (centroid(&a), centroid(&b));
            [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]]
        });
        let span = (0..3)
            .map(|k| {
                let (lo, hi) = faces.iter().fold((f64::MAX, f64::MIN), |(lo, hi), f| {
                    (lo.min(f.centroid[k]), hi.max(f.centroid[k]))
                });
                hi - lo
            })
            .fold(1.0, f64::max);
        let tol = 1e-6 * span;
        let off = |i: u32, j: u32| {
            let (ci, cj) = (cen(i), cen(j));
            let d = [
                ci[0] + t[0] - cj[0],
                ci[1] + t[1] - cj[1],
                ci[2] + t[2] - cj[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        // The image of a face by centroid and area: faces of one side can
        // share a centroid (a side with a hole and the disc in it).
        let miss = |i: u32, j: u32| off(i, j) + (area(i) - area(j)).abs() / span;
        let mut pairs = Vec::with_capacity(a.len());
        for &i in &a {
            let j = b
                .iter()
                .copied()
                .fold(None, |best: Option<u32>, j| match best {
                    Some(k) if miss(i, k) <= miss(i, j) => Some(k),
                    _ => Some(j),
                })
                .expect("b is not empty");
            if off(i, j) > tol || (area(i) - area(j)).abs() > tol * span {
                return Err(Error::Invalid(format!(
                    "face {i} has no image among the slave faces"
                )));
            }
            pairs.push(PeriodicPair {
                a: i,
                b: j,
                shift: t,
            });
        }
        let mut hit: Vec<u32> = pairs.iter().map(|p| p.b).collect();
        hit.sort_unstable();
        hit.dedup();
        if hit.len() != b.len() {
            return Err(Error::Invalid(
                "the slave selection has faces that are no image of a master face".into(),
            ));
        }
        Ok((pairs, t))
    }

    /// The periodic face pairs, paired on the current model.
    pub fn periodic_pairs(&self) -> Result<Vec<PeriodicPair>> {
        let mut out = Vec::new();
        for (m, sl, t) in &self.periodic {
            out.extend(self.pair_faces(m, sl, Some(*t))?.0);
        }
        Ok(out)
    }

    // ---- meshing -------------------------------------------------------------

    /// The per-entity sizes and tolerances on the current model, in the
    /// order they were given (a later one wins).
    fn resolve_sizing(&self) -> Result<Resolved> {
        let mut r = Resolved::default();
        for (scope, knob, v) in &self.sizing.scoped {
            let map = match (knob, scope.level) {
                (Knob::Maxh, Level::Edge) => &mut r.edge_maxh,
                (Knob::Maxh, Level::Surf) => &mut r.surf_maxh,
                (Knob::Maxh, Level::Region) => &mut r.region_maxh,
                (Knob::Tol, Level::Edge) => &mut r.edge_tol,
                (Knob::Tol, _) => &mut r.surf_tol,
            };
            map.extend(self.resolve(scope)?.into_iter().map(|i| (i, *v)));
        }
        Ok(r)
    }

    /// The per-region sizes: the ones given with the solids, overridden by
    /// the region scopes.
    fn region_maxh(&self, r: &Resolved) -> Vec<(u32, f64)> {
        let mut out = self.solid_maxh.clone();
        for (&r, &h) in &r.region_maxh {
            match out.iter_mut().find(|(rr, _)| *rr == r) {
                Some(e) => e.1 = h,
                None => out.push((r, h)),
            }
        }
        out
    }

    /// The mesher parameters for a global size `maxh`, grading and the
    /// per-call overrides of the sizing defaults.
    #[allow(clippy::too_many_arguments)]
    fn params(
        &self,
        maxh: Option<f64>,
        grading: Option<f64>,
        tol: [Option<f64>; 2],
        caps: [Option<f64>; 3],
    ) -> Result<MeshParams> {
        let s = &self.sizing;
        let r = self.resolve_sizing()?;
        let list = |m: &BTreeMap<u32, f64>| m.iter().map(|(&k, &v)| (k, v)).collect();
        Ok(MeshParams {
            maxh: maxh.or(self.maxh).unwrap_or(f64::INFINITY),
            region_maxh: self.region_maxh(&r),
            radius_edge_bound: 0.0,
            min_h_surf: 0.0,
            min_h_vol: 0.0,
            surf_min_angle: 0.0,
            surf_target_count: 0,
            max_points: usize::MAX,
            grading: grading.unwrap_or(self.grading),
            face_maxh: list(&self.face_maxh),
            surface_maxh: list(&self.surface_maxh),
            size_points: self.size_points.clone(),
            tol_edge: tol[0].unwrap_or(s.tol_edge),
            tol_surf: tol[1].unwrap_or(s.tol_surf),
            cap_edge: caps[0].unwrap_or(s.maxh_edge),
            cap_surf: caps[1].unwrap_or(s.maxh_surf),
            cap_vol: caps[2].unwrap_or(s.maxh_vol),
            edge_maxh: list(&r.edge_maxh),
            edge_tol: list(&r.edge_tol),
            surf_maxh: list(&r.surf_maxh),
            surf_tol: list(&r.surf_tol),
            periodic: Vec::new(),
            cells_across: 0.0,
        })
    }

    /// Assembles every solid and sheet exactly, meshes the arrangement with
    /// tetrahedra and improves them.
    pub fn mesh(&self, opts: &MeshOptions) -> Result<Mesh> {
        let t0 = Instant::now();
        rapidmesh_exact::log::clear();
        let ta = Instant::now();
        let model = self.model()?;
        let t_assemble = ta.elapsed();
        rapidmesh_exact::log::stage("assemble.total", t_assemble.as_secs_f64());
        let params = MeshParams {
            radius_edge_bound: opts.radius_edge,
            min_h_surf: opts.min_h_surf,
            min_h_vol: opts.min_h_vol,
            max_points: opts.max_points,
            cells_across: opts.cells_across.unwrap_or(0.0),
            periodic: self.periodic_pairs()?,
            ..self.params(
                opts.maxh,
                opts.grading,
                [opts.tol_edge, opts.tol_surf],
                [opts.maxh_edge, opts.maxh_surf, opts.maxh_vol],
            )?
        };
        let tm = Instant::now();
        let legacy = || {
            let params = MeshParams {
                cells_across: opts.cells_across.unwrap_or(DEFAULT_CELLS_ACROSS),
                ..params.clone()
            };
            mesh_budgeted(&model, &params, opts.target_elements).0
        };
        let bottom_up = || {
            rapidmesh_tet::budgeted(&model, &params, opts.target_elements, &|p| {
                rapidmesh_tet::bottomup::mesh_scene(&self.scene, &model, p)
            })
            .map(|x| x.0)
        };
        let mesh = match opts.bottom_up {
            Some(false) => legacy(),
            Some(true) => {
                bottom_up().map_err(|e| Error::Invalid(format!("bottom-up mesh: {e}")))?
            }
            None => match bottom_up_or_why(bottom_up) {
                Ok(m) => m,
                Err(why) => {
                    rapidmesh_exact::log::warn(
                        "mesh.mesher",
                        format!("bottom-up failed ({why}); restricted Delaunay instead"),
                    );
                    rapidmesh_exact::log::stat("mesh.fallback", 1.0);
                    legacy()
                }
            },
        };
        let t_mesh = tm.elapsed();
        rapidmesh_exact::log::stage("mesh.total", t_mesh.as_secs_f64());
        let quality = quality_stats(&mesh);
        rapidmesh_tet::log_metrics(&quality, mesh.points.len());
        Ok(Mesh::new(
            mesh,
            quality,
            self.labels()?,
            Run::finish(t0),
            Some(model),
        ))
    }

    /// Meshes only the surfaces of the arrangement (interfaces, outer
    /// boundary, sheets) with the full sizing hierarchy; no tets.
    pub fn surface_mesh(&self, opts: &SurfaceOptions) -> Result<SurfaceMesh> {
        let t0 = Instant::now();
        rapidmesh_exact::log::clear();
        let model = self.model()?;
        let params = MeshParams {
            surf_min_angle: 20.0,
            surf_target_count: opts.target_triangles.unwrap_or(0),
            ..self.params(
                opts.maxh,
                opts.grading,
                [opts.tol_edge, opts.tol_surf],
                [opts.maxh_edge, opts.maxh_surf, opts.maxh_vol],
            )?
        };
        let mesh = rapidmesh_tet::bottomup::surface_mesh(&model, &params)
            .map_err(|e| Error::Invalid(format!("surface mesh: {e}")))?;
        rapidmesh_tet::log_surface_metrics(&mesh);
        Ok(SurfaceMesh::new(mesh, self.labels()?, Run::finish(t0)))
    }
}

/// The bottom-up mesh, or why there is none: its error, or the message of a
/// panic inside it (so the caller can fall back rather than abort).
fn bottom_up_or_why<T, E: std::fmt::Display>(
    f: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(m)) => Ok(m),
        Ok(Err(e)) => Err(e.to_string()),
        Err(p) => Err(p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "panic".into())),
    }
}

/// Names role `at` in `roles`, with empty names for the roles before it
/// that had none.
fn name_role(roles: &mut Vec<String>, at: u32, name: &str) {
    let at = at as usize;
    if roles.len() <= at {
        roles.resize(at + 1, String::new());
    }
    roles[at] = name.to_string();
}
