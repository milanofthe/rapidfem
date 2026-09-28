// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The rapidfem geometry: a declarative scene of solids and sheets, realised
//! into a `rapidmesh::Geometry` whenever its topology or a mesh is needed.
//!
//! Keeping the scene on this side makes every edit a change of a list entry
//! (an extrusion replaces its profile sheet by a prism, a cut turns a solid
//! into a void) and leaves the realised model a pure function of the list.
//! Solids overlap by priority, as in rapidmesh: a later solid carves its
//! region out of the earlier ones, and the mesh is always conformal.
//!
//! Faces are named by their origin, [`FaceOrigin`]: the object they come from
//! and, for a solid, the role of the surface in its primitive (a box: -z, +z,
//! -y, +y, -x, +x). An origin survives later changes of the scene, which may
//! split the face or renumber the model; it resolves to the current B-rep
//! face ids on demand.

use std::sync::OnceLock;

use rapidmesh::shapes::{Shape, Sheet};
use rapidmesh::{EdgeCut, EdgePick, FaceFilter, MeshOptions, Scope, Solid, Topology};

/// Index of an object in its [`Geometry`].
pub type ObjId = usize;

/// What an object is.
#[derive(Clone, Debug)]
pub enum Item {
    Solid(Shape),
    Sheet(Sheet),
}

/// A solid or sheet of the scene with its attributes.
#[derive(Clone, Debug)]
pub struct Object {
    pub item: Item,
    /// A void solid is cut out of every solid added before it.
    pub void: bool,
    pub maxh: Option<f64>,
    pub name: Option<String>,
    /// Removed objects keep their slot so ids stay valid.
    pub alive: bool,
}

/// A chamfer or fillet on edges of a solid object, applied in order after
/// the scene is assembled. The new faces become faces of the object with
/// roles after its own (with `void` they belong to the carved voids).
#[derive(Clone, Debug)]
pub struct EdgeOp {
    pub object: ObjId,
    pub edges: Vec<EdgePick>,
    pub cut: EdgeCut,
    pub void: bool,
}

/// Where a B-rep face comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FaceOrigin {
    /// Surface `role` of solid `object`.
    Solid { object: ObjId, role: u32 },
    /// The sheet `object`.
    Sheet { object: ObjId },
}

/// A face selection: the pieces of a face origin, optionally only those
/// bounding the solid `side` (a face split by a later solid keeps its origin
/// on every piece; `side` keeps the pieces that face the solid it was
/// selected through, as a gmsh fragment did).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FaceSel {
    pub origin: FaceOrigin,
    pub side: Option<ObjId>,
}

/// A B-rep face of the realised model.
#[derive(Clone, Debug)]
pub struct Face {
    pub id: u32,
    pub origin: FaceOrigin,
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub area: f64,
    /// Regions on the front and back (0: outside, or a void).
    pub regions: [u32; 2],
    /// Axis-aligned bounding box `[xmin, ymin, zmin, xmax, ymax, zmax]`.
    pub bbox: [f64; 6],
}

/// The scene realised in rapidmesh.
pub struct Realized {
    pub geometry: rapidmesh::Geometry,
    /// The rapidmesh solid of each object (None for sheets and removed ones).
    pub solids: Vec<Option<Solid>>,
    pub topology: Topology,
    pub faces: Vec<Face>,
}

/// The rapidfem geometry, see the module docs.
pub struct Geometry {
    objects: Vec<Object>,
    edge_ops: Vec<EdgeOp>,
    /// Solid objects fused into one region (the first keeps its region).
    unions: Vec<Vec<ObjId>>,
    /// Target sizes on faces, by origin.
    face_maxh: Vec<(Vec<FaceSel>, f64)>,
    maxh: Option<f64>,
    size_points: Vec<([f64; 3], f64)>,
    realized: OnceLock<Result<Realized, String>>,
}

/// Whether face `f` is a piece of selection `s`.
fn matches(solids: &[Option<Solid>], objects: &[Object], s: &FaceSel, f: &Face) -> bool {
    if f.origin != s.origin {
        return false;
    }
    match s.side {
        None => true,
        Some(o) => match solids.get(o).copied().flatten() {
            Some(sol) if !objects[o].void => f.regions.contains(&sol.region),
            _ => true,
        },
    }
}

/// Moves a solid or sheet description by `d`.
fn translate_item(item: &mut Item, d: [f64; 3]) -> Result<(), String> {
    let add = |p: &mut [f64; 3]| {
        for k in 0..3 {
            p[k] += d[k];
        }
    };
    match item {
        Item::Solid(s) => match s {
            Shape::Cuboid(x) => add(&mut x.position),
            Shape::Cylinder(x) => add(&mut x.position),
            Shape::Sphere(x) => add(&mut x.position),
            Shape::Icosphere(x) => add(&mut x.position),
            Shape::Naca0012(x) => add(&mut x.position),
            Shape::Cone(x) => add(&mut x.position),
            Shape::Prism(x) => add(&mut x.position),
            Shape::Torus(x) => add(&mut x.position),
            Shape::Wedge(x) => add(&mut x.position),
            Shape::Helix(x) => add(&mut x.position),
            Shape::Revolve(x) => add(&mut x.position),
            Shape::Sweep(x) => x.path.iter_mut().for_each(add),
            Shape::Loft(x) => {
                x.profile_a.iter_mut().for_each(add);
                x.profile_b.iter_mut().for_each(add);
            }
            Shape::Triangles(x) => x.verts.iter_mut().for_each(add),
            Shape::Import(_) => return Err("translate: an imported solid cannot move yet".into()),
        },
        Item::Sheet(s) => match s {
            Sheet::Rect { corner, .. } => add(corner),
            Sheet::Disc { center, .. } => add(center),
            Sheet::Polygon { position, .. } => add(position),
            Sheet::Nurbs { .. } => return Err("translate: a NURBS sheet cannot move yet".into()),
        },
    }
    Ok(())
}

/// Sheet face tags are the object index shifted by one (0 means untagged).
fn sheet_tag(object: ObjId) -> u32 {
    object as u32 + 1
}

impl Geometry {
    /// An empty geometry with global target size `maxh`.
    pub fn new(maxh: Option<f64>) -> Self {
        Geometry {
            objects: Vec::new(),
            edge_ops: Vec::new(),
            unions: Vec::new(),
            face_maxh: Vec::new(),
            maxh,
            size_points: Vec::new(),
            realized: OnceLock::new(),
        }
    }

    fn changed(&mut self) {
        self.realized = OnceLock::new();
    }

    pub fn maxh(&self) -> Option<f64> {
        self.maxh
    }

    pub fn set_maxh(&mut self, h: Option<f64>) {
        self.maxh = h;
        self.changed();
    }

    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub fn object(&self, id: ObjId) -> &Object {
        &self.objects[id]
    }

    /// Adds a solid (or a void) and returns its id.
    pub fn add_solid(&mut self, shape: impl Into<Shape>, maxh: Option<f64>, void: bool) -> ObjId {
        self.push(Item::Solid(shape.into()), maxh, void)
    }

    /// Adds a sheet and returns its id.
    pub fn add_sheet(&mut self, sheet: Sheet, maxh: Option<f64>) -> ObjId {
        self.push(Item::Sheet(sheet), maxh, false)
    }

    fn push(&mut self, item: Item, maxh: Option<f64>, void: bool) -> ObjId {
        self.objects.push(Object { item, void, maxh, name: None, alive: true });
        self.changed();
        self.objects.len() - 1
    }

    /// Replaces the item of an object in place (an extrusion turns its
    /// profile sheet into a solid): selections of the object stay valid.
    pub fn replace(&mut self, id: ObjId, item: Item) {
        self.objects[id].item = item;
        self.changed();
    }

    /// Turns solids into voids: they are cut out of every solid added
    /// before them and their walls become boundary faces.
    pub fn make_void(&mut self, ids: &[ObjId]) {
        for &i in ids {
            self.objects[i].void = true;
        }
        self.changed();
    }

    /// Removes an object from the scene.
    pub fn remove(&mut self, id: ObjId) {
        self.objects[id].alive = false;
        self.changed();
    }

    pub fn set_name(&mut self, id: ObjId, name: Option<String>) {
        self.objects[id].name = name;
    }

    pub fn set_object_maxh(&mut self, id: ObjId, maxh: Option<f64>) {
        self.objects[id].maxh = maxh;
        self.changed();
    }

    /// Chamfers (`EdgeCut::Chamfer`) or fillets edges of a solid object.
    pub fn cut_edges(&mut self, op: EdgeOp) {
        self.edge_ops.push(op);
        self.changed();
    }

    /// Fuses solid objects into one material region: the faces between
    /// them go, the first object's region takes them all.
    pub fn fuse(&mut self, ids: Vec<ObjId>) {
        self.unions.push(ids);
        self.changed();
    }

    /// A target size `h` on the faces of `origins`.
    pub fn set_face_maxh(&mut self, origins: Vec<FaceSel>, h: f64) {
        self.face_maxh.push((origins, h));
        self.changed();
    }

    /// Moves an object by `d`.
    pub fn translate(&mut self, id: ObjId, d: [f64; 3]) -> Result<(), String> {
        translate_item(&mut self.objects[id].item, d)?;
        self.changed();
        Ok(())
    }

    /// A target size `h` at the point `p`, recovering along the grading.
    pub fn add_size_point(&mut self, p: [f64; 3], h: f64) {
        self.size_points.push((p, h));
        self.changed();
    }

    /// The scene realised in rapidmesh, built on first use after a change.
    pub fn realized(&self) -> Result<&Realized, String> {
        self.realized.get_or_init(|| self.realize()).as_ref().map_err(Clone::clone)
    }

    fn realize(&self) -> Result<Realized, String> {
        let mut g = rapidmesh::Geometry::new(self.maxh);
        let mut solids = vec![None; self.objects.len()];
        for (i, o) in self.objects.iter().enumerate() {
            if !o.alive {
                continue;
            }
            match &o.item {
                Item::Solid(shape) => {
                    let s = g.add_solid(shape.clone(), o.maxh, o.void).map_err(|e| e.to_string())?;
                    solids[i] = Some(s);
                }
                Item::Sheet(sheet) => {
                    g.add_sheet(sheet, sheet_tag(i), o.maxh).map_err(|e| e.to_string())?;
                }
            }
        }
        for op in &self.edge_ops {
            let solid = solids
                .get(op.object)
                .copied()
                .flatten()
                .ok_or_else(|| format!("edge cut on object {}, which is not a solid", op.object))?;
            g.cut_edges(solid, &op.edges, op.cut, op.void).map_err(|e| e.to_string())?;
        }
        for ids in &self.unions {
            // voids have no material region to merge
            let s: Vec<Solid> = ids
                .iter()
                .filter(|&&i| !self.objects[i].void)
                .filter_map(|&i| solids.get(i).copied().flatten())
                .collect();
            if s.len() > 1 {
                let keep = g.union(&s).map_err(|e| e.to_string())?;
                for &i in ids {
                    if let Some(slot) = solids.get_mut(i).and_then(|o| o.as_mut()) {
                        slot.region = keep.region;
                    }
                }
            }
        }
        for &(p, h) in &self.size_points {
            g.add_size_point(p, h);
        }
        let topology = g.topology().map_err(|e| e.to_string())?;
        let model = g.model().map_err(|e| e.to_string())?;
        // Face bounding boxes from the PLC facets of each B-rep face
        // (topology face i is B-rep face i).
        let bbox_of = |id: usize| {
            let mut b = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
            for &ti in &model.brep.faces[id].facets {
                for &v in &model.plc.triangles[ti as usize] {
                    let p = model.plc.vertices[v as usize];
                    for k in 0..3 {
                        b[k] = b[k].min(p[k]);
                        b[k + 3] = b[k + 3].max(p[k]);
                    }
                }
            }
            b
        };

        // solid index (insertion order, voids included) -> object
        let mut object_of_index = std::collections::HashMap::new();
        for (i, s) in solids.iter().enumerate() {
            if let Some(s) = s {
                object_of_index.insert(s.index, i);
            }
        }
        let faces = topology
            .faces
            .iter()
            .enumerate()
            .filter_map(|(id, f)| {
                let origin = if f.face_tag != 0 {
                    FaceOrigin::Sheet { object: (f.face_tag - 1) as ObjId }
                } else {
                    let &object = object_of_index.get(&f.owner)?;
                    FaceOrigin::Solid { object, role: f.role }
                };
                Some(Face {
                    id: id as u32,
                    origin,
                    centroid: f.centroid,
                    normal: f.normal,
                    area: f.area,
                    regions: f.regions,
                    bbox: bbox_of(id),
                })
            })
            .collect();
        let mut realized = Realized { geometry: g, solids, topology, faces };
        for (sels, h) in &self.face_maxh {
            let ids: Vec<u32> = realized
                .faces
                .iter()
                .filter(|f| sels.iter().any(|s| matches(&realized.solids, &self.objects, s, f)))
                .map(|f| f.id)
                .collect();
            for id in ids {
                let filter = FaceFilter { id: Some(id), ..FaceFilter::default() };
                realized.geometry.set_maxh_on(&Scope::surf(Some(filter)), *h).map_err(|e| e.to_string())?;
            }
        }
        Ok(realized)
    }

    /// The region of a solid object in the realised model.
    pub fn region(&self, id: ObjId) -> Result<Option<u32>, String> {
        Ok(self.realized()?.solids[id].filter(|_| !self.objects[id].void).map(|s| s.region))
    }

    /// The faces bounding an object: for a solid every face with its region
    /// on one side (interfaces with later solids included), for a void its
    /// walls, for a sheet the sheet itself. Solid faces carry the object as
    /// their side.
    pub fn faces_of(&self, id: ObjId) -> Result<Vec<FaceSel>, String> {
        let r = self.realized()?;
        let o = &self.objects[id];
        let mut out: Vec<FaceSel> = match (&o.item, r.solids[id]) {
            (Item::Sheet(_), _) => vec![FaceSel { origin: FaceOrigin::Sheet { object: id }, side: None }],
            (Item::Solid(_), Some(s)) if !o.void => r
                .faces
                .iter()
                .filter(|f| f.regions.contains(&s.region))
                .map(|f| FaceSel { origin: f.origin, side: Some(id) })
                .collect(),
            (Item::Solid(_), Some(_)) => r
                .faces
                .iter()
                .filter(|f| matches!(f.origin, FaceOrigin::Solid { object, .. } if object == id))
                .map(|f| FaceSel { origin: f.origin, side: None })
                .collect(),
            (Item::Solid(_), None) => Vec::new(),
        };
        out.sort();
        out.dedup();
        Ok(out)
    }

    /// The edges of a solid object, each named by the roles of the two of
    /// its faces meeting there (the `EdgePick::Between` selector), with the
    /// edge midpoint.
    pub fn edges_of(&self, id: ObjId) -> Result<Vec<([u32; 2], [f64; 3])>, String> {
        let r = self.realized()?;
        let role_of = |face: u32| match r.faces.iter().find(|f| f.id == face)?.origin {
            FaceOrigin::Solid { object, role } if object == id => Some(role),
            _ => None,
        };
        let mut out = Vec::new();
        for e in &r.topology.edges {
            let roles: Vec<u32> = e.faces.iter().filter_map(|&f| role_of(f)).collect();
            if let [a, b] = roles[..] {
                out.push(([a.min(b), a.max(b)], e.midpoint));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out.dedup_by(|a, b| a.0 == b.0);
        Ok(out)
    }

    /// The current B-rep faces of a set of selections.
    pub fn resolve(&self, sels: &[FaceSel]) -> Result<Vec<&Face>, String> {
        let r = self.realized()?;
        Ok(r.faces.iter().filter(|f| sels.iter().any(|s| matches(&r.solids, &self.objects, s, f))).collect())
    }

    /// Meshes the realised scene.
    pub fn mesh(&self, opts: &MeshOptions) -> Result<rapidmesh::Mesh, String> {
        self.realized()?.geometry.mesh(opts).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh::shapes::Cuboid;

    fn substrate_in_air() -> (Geometry, ObjId, ObjId) {
        let mut g = Geometry::new(Some(1.0));
        let air = g.add_solid(Cuboid::new([4.0, 4.0, 3.0]), None, false);
        let sub = g.add_solid(Cuboid::new([4.0, 4.0, 1.0]), None, false);
        (g, air, sub)
    }

    #[test]
    fn a_later_solid_carves_the_earlier_one() {
        let (g, air, sub) = substrate_in_air();
        let (ra, rs) = (g.region(air).unwrap().unwrap(), g.region(sub).unwrap().unwrap());
        assert_ne!(ra, rs);
        // the substrate/air interface bounds both
        let fa = g.faces_of(air).unwrap();
        let fs = g.faces_of(sub).unwrap();
        assert!(fa.iter().any(|a| fs.iter().any(|s| s.origin == a.origin)), "no shared interface face");
        // the air's floor lies under the substrate, which carves it: selected
        // through the air, the floor has no piece left
        let floor = FaceSel { origin: FaceOrigin::Solid { object: air, role: 0 }, side: Some(air) };
        assert!(g.resolve(&[floor]).unwrap().is_empty());
    }

    #[test]
    fn origins_survive_a_later_change() {
        let (mut g, air, _) = substrate_in_air();
        let top = FaceSel { origin: FaceOrigin::Solid { object: air, role: 1 }, side: None }; // +z of the air box
        let before: Vec<u32> = g.resolve(&[top]).unwrap().iter().map(|f| f.id).collect();
        assert_eq!(before.len(), 1);
        // a sheet added later renumbers the model; the origin still finds +z
        g.add_sheet(Sheet::xy(1.0, 1.0, [1.0, 1.0, 2.0]), None);
        let after = g.resolve(&[top]).unwrap();
        assert_eq!(after.len(), 1);
        assert!((after[0].centroid[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_void_leaves_its_walls() {
        let mut g = Geometry::new(Some(1.0));
        let air = g.add_solid(Cuboid::new([4.0, 4.0, 4.0]), None, false);
        let hole = g.add_solid(Cuboid::new([1.0, 1.0, 1.0]).at([1.5, 1.5, 1.5]), None, false);
        g.make_void(&[hole]);
        assert_eq!(g.region(hole).unwrap(), None);
        let walls = g.faces_of(hole).unwrap();
        assert_eq!(walls.len(), 6);
        let ra = g.region(air).unwrap().unwrap();
        for f in g.resolve(&walls).unwrap() {
            assert!(f.regions.contains(&ra) && f.regions.contains(&0), "{:?}", f.regions);
        }
    }

    #[test]
    fn a_fillet_adds_a_face_to_the_solid() {
        let mut g = Geometry::new(Some(0.5));
        let b = g.add_solid(Cuboid::new([2.0, 2.0, 2.0]), None, false);
        let before = g.faces_of(b).unwrap().len();
        // the edge between +z (role 1) and +x (role 5)
        let edges = g.edges_of(b).unwrap();
        assert!(edges.iter().any(|(r, _)| *r == [1, 5]), "{edges:?}");
        g.cut_edges(EdgeOp {
            object: b,
            edges: vec![EdgePick::Between(1, 5)],
            cut: EdgeCut::Fillet(0.3),
            void: false,
        });
        assert_eq!(g.faces_of(b).unwrap().len(), before + 1);
    }
}
