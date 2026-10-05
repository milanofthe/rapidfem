// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The rapidfem geometry: a declarative scene of solids and sheets, realised
//! into a `rapidmesh::Geometry` whenever its topology or a mesh is needed.
//!
//! Keeping the scene on this side makes every edit a change of a list entry
//! (an extrusion replaces its profile sheet by a prism, a cut turns a solid
//! into a hole) and leaves the realised model a pure function of the list.
//!
//! A hole is meshed as a region of its own and dropped from the solver mesh
//! ([`crate::fem_mesh`]): its walls are then simply the faces bounding its
//! region. (A rapidmesh void would own only the walls no other solid claims;
//! walls lying on another solid's faces, the norm in a layer stack, would be
//! lost.)
//! Solids overlap by priority, as in rapidmesh: a later solid carves its
//! region out of the earlier ones, and the mesh is always conformal.
//!
//! Faces are named by their origin, [`FaceOrigin`]: the object they come from
//! and, for a solid, the role of the surface in its primitive (a box: -z, +z,
//! -y, +y, -x, +x). An origin survives later changes of the scene, which may
//! split the face or renumber the model; it resolves to the current B-rep
//! face ids on demand.

use std::sync::OnceLock;

use rapidfem_core::geom::{cross, dot, norm, scale, sub};
use rapidmesh::shapes::{Loft, Prism, Revolve, Shape, Sheet, Sweep};
use std::collections::BTreeMap;

use crate::fem_mesh::Group;
use crate::path::spline;
use crate::sheet_ops::{self, SheetOp};
use rapidmesh::{BoolOp, EdgeCut, EdgePick, FaceFilter, MeshOptions, SurfaceOptions, Object as RmObject, Scope, SheetRef, Solid, Topology, Transform};

/// Index of an object in its [`Geometry`].
pub type ObjId = usize;

/// What an object is.
#[derive(Clone, Debug)]
pub enum Item {
    Solid(Shape),
    Sheet(Sheet),
    /// Several pieces of one sheet sharing the object's tag: a sheet union
    /// (rapidmesh merges overlapping pieces) or a sheet boolean that falls
    /// apart.
    Sheets(Vec<Piece>),
    /// The solid one sheet or several pieces sweep along `vector`, the
    /// pieces' solids fused; the sheets stay as its bottom face. The
    /// object's first `placed` transforms move the sheets before the sweep,
    /// the rest move both after it.
    Extrusion { pieces: Vec<Piece>, vector: [f64; 3], placed: usize },
    /// Body `body` of the STEP file at `path`, in the file's unit.
    Step { path: std::path::PathBuf, body: usize },
}

/// One planar piece of a sheet object.
#[derive(Clone, Debug)]
pub enum Piece {
    /// A sheet and its own transforms, applied before its object's.
    Sheet(Sheet, Vec<Transform>),
    /// rapidmesh's exact boolean `op` of the pieces `target` with each tool
    /// (a tool's pieces merged first) in their common plane, then moved by
    /// `after`.
    Boolean { op: SheetOp, target: Vec<Piece>, tools: Vec<Vec<Piece>>, after: Vec<Transform> },
}

impl Piece {
    /// The piece moved by `more` after its own transforms.
    fn then(&self, more: &[Transform]) -> Piece {
        let mut p = self.clone();
        match &mut p {
            Piece::Sheet(_, own) | Piece::Boolean { after: own, .. } => own.extend_from_slice(more),
        }
        p
    }

    /// The piece added to `g` as one sheet with face tag `tag`.
    fn add(&self, g: &mut rapidmesh::Geometry, tag: u32, maxh: Option<f64>) -> rapidmesh::Result<SheetRef> {
        let (r, moves) = match self {
            Piece::Sheet(sheet, own) => (g.add_sheet(sheet, tag, maxh)?, own),
            Piece::Boolean { op, target, tools, after } => {
                let t = Piece::merged(target, g, tag, maxh)?;
                let tools = tools.iter().map(|t| Piece::merged(t, g, tag, maxh)).collect::<Result<Vec<_>, _>>()?;
                let op = match op {
                    SheetOp::Union => BoolOp::Union,
                    SheetOp::Difference => BoolOp::Difference,
                    SheetOp::Intersection => BoolOp::Intersection,
                };
                (g.sheet_boolean(op, t, &tools)?, after)
            }
        };
        for &tr in moves {
            g.transform(r, tr)?;
        }
        Ok(r)
    }

    /// `pieces` added to `g` and merged into one sheet.
    fn merged(pieces: &[Piece], g: &mut rapidmesh::Geometry, tag: u32, maxh: Option<f64>) -> rapidmesh::Result<SheetRef> {
        let refs = pieces.iter().map(|p| p.add(g, tag, maxh)).collect::<Result<Vec<_>, _>>()?;
        match refs.split_first() {
            Some((&first, [])) => Ok(first),
            Some((&first, rest)) => g.sheet_boolean(BoolOp::Union, first, rest),
            None => Err(rapidmesh::Error::Invalid("an empty sheet operand".into())),
        }
    }
}

/// A solid or sheet of the scene with its attributes.
#[derive(Clone, Debug)]
pub struct Object {
    pub item: Item,
    /// A hole: meshed as its own region, then removed from the solver mesh,
    /// so its walls become boundary faces.
    pub void: bool,
    /// The object's own target size.
    pub maxh: Option<f64>,
    /// The target size of its material, where it has no size of its own.
    pub material_maxh: Option<f64>,
    pub name: Option<String>,
    /// Removed objects keep their slot so ids stay valid.
    pub alive: bool,
    /// Assembly order: a solid of higher priority (or equal priority and
    /// added later) carves its region out of the others.
    pub priority: u64,
    /// Placement changes, applied in order after the object is added.
    pub transforms: Vec<Transform>,
}

impl Object {
    /// The target size the mesher gets: the object's own, else its
    /// material's.
    pub fn size(&self) -> Option<f64> {
        self.maxh.or(self.material_maxh)
    }

    /// Whether the object is a solid (not a sheet).
    pub fn is_solid(&self) -> bool {
        !matches!(self.item, Item::Sheet(_) | Item::Sheets(_))
    }
}

/// Where a selection lies: its area-weighted centroid (a solid's: the centre
/// of its bounding box), its area (0 for a solid) and its bounding box
/// `[xmin, ymin, zmin, xmax, ymax, zmax]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extent {
    pub centroid: [f64; 3],
    pub area: f64,
    pub bbox: [f64; 6],
}

impl Extent {
    /// An empty box, for growing.
    const EMPTY: [f64; 6] = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];

    fn grow(b: &mut [f64; 6], other: &[f64; 6]) {
        for k in 0..3 {
            b[k] = b[k].min(other[k]);
            b[k + 3] = b[k + 3].max(other[k + 3]);
        }
    }

    /// The box around every extent of `items` (an empty box for none).
    pub fn union<'a>(items: impl IntoIterator<Item = &'a Extent>) -> [f64; 6] {
        let mut b = Extent::EMPTY;
        for e in items {
            Extent::grow(&mut b, &e.bbox);
        }
        b
    }

    /// The largest edge of a box (at least 1e-300): the length the relative
    /// selection tolerances refer to.
    pub fn size(b: &[f64; 6]) -> f64 {
        (b[3] - b[0]).max(b[4] - b[1]).max(b[5] - b[2]).max(1e-300)
    }

    /// The positions of the items whose centroid lies at the minimum (or
    /// the maximum) along `axis`, within `tol`.
    pub fn extreme(items: &[Extent], axis: usize, max: bool, tol: f64) -> Vec<usize> {
        let c: Vec<f64> = items.iter().map(|e| e.centroid[axis]).collect();
        let m = if max {
            c.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        } else {
            c.iter().copied().fold(f64::INFINITY, f64::min)
        };
        (0..c.len()).filter(|&i| (c[i] - m).abs() <= tol).collect()
    }

    /// The positions of the items lying flat (thinner than `tol` across)
    /// on one of the six planes of box `b`.
    pub fn on_box(items: &[Extent], b: &[f64; 6], tol: f64) -> Vec<usize> {
        let flat_on = |e: &[f64; 6], k: usize| {
            (e[k + 3] - e[k]).abs() < tol && ((e[k] - b[k]).abs() < tol || (e[k] - b[k + 3]).abs() < tol)
        };
        (0..items.len()).filter(|&i| (0..3).any(|k| flat_on(&items[i].bbox, k))).collect()
    }
}

/// Two centroids closer than this, relative to the model size, are at the
/// same place for [`Extent::extreme`]; a face thinner than this across an
/// axis is flat in it for [`Extent::on_box`].
pub const SELECT_TOL: f64 = 1e-9;

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

/// What lies across a face piece from the solid it was selected through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Across {
    /// Outside the domain or a void.
    Outside,
    /// The region of this object.
    Object(ObjId),
}

/// A face selection: the pieces of a face origin, optionally only those
/// bounding the solid `side`, and of those only the ones with `across` on
/// the other side. A face split by a later solid keeps its origin on every
/// piece; `side` and `across` single out the pieces the way a gmsh fragment
/// made them separate faces (the throat interface of a step, apart from the
/// shoulders around it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FaceSel {
    pub origin: FaceOrigin,
    pub side: Option<ObjId>,
    pub across: Option<Across>,
}

impl FaceSel {
    /// The selection as a flat key `(object, role, side, across)`: role -1
    /// for a sheet, side -1 for none, across -1 for any and -2 for outside.
    pub fn key(&self) -> (ObjId, i64, i64, i64) {
        let side = self.side.map_or(-1, |o| o as i64);
        let across = match self.across {
            None => -1,
            Some(Across::Outside) => -2,
            Some(Across::Object(o)) => o as i64,
        };
        match self.origin {
            FaceOrigin::Solid { object, role } => (object, role as i64, side, across),
            FaceOrigin::Sheet { object } => (object, -1, side, across),
        }
    }

    /// The selection of a key of [`FaceSel::key`].
    pub fn from_key((object, role, side, across): (ObjId, i64, i64, i64)) -> FaceSel {
        let origin = if role < 0 {
            FaceOrigin::Sheet { object }
        } else {
            FaceOrigin::Solid { object, role: role as u32 }
        };
        let across = match across {
            -1 => None,
            -2 => Some(Across::Outside),
            o => Some(Across::Object(o as usize)),
        };
        FaceSel { origin, side: (side >= 0).then_some(side as usize), across }
    }

    /// The sheet itself, selected through no solid.
    pub fn sheet(object: ObjId) -> FaceSel {
        FaceSel { origin: FaceOrigin::Sheet { object }, side: None, across: None }
    }
}

/// `a` divided by its length.
pub fn normalized(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// The sheet of a planar polygon with holes, its vertices in 3D, in the
/// plane they span: an xy polygon at its height when every vertex has the
/// same z, the plane's own axes when it is square to x or y, otherwise a
/// frame in the tilted plane. `None` when the vertices are not in one plane.
pub fn polygon_sheet(points: &[[f64; 3]], holes: &[Vec<[f64; 3]>]) -> Option<Sheet> {
    let first = *points.first()?;
    let all = || points.iter().chain(holes.iter().flatten());
    let size = all().fold(0.0_f64, |m, p| m.max(norm(sub(*p, first))));
    if size == 0.0 {
        return None;
    }
    let mut n = [0.0; 3];
    for k in 0..points.len() {
        let (a, b) = (points[k], points[(k + 1) % points.len()]);
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    if norm(n) == 0.0 {
        return None;
    }
    let n = normalized(n);
    if all().any(|p| dot(sub(*p, first), n).abs() > 1e-9 * size) {
        return None;
    }
    // an axis plane keeps its coordinates exact: the two other axes in turn
    let axis = (0..3).find(|&k| all().all(|p| (p[k] - first[k]).abs() <= 1e-12 * size));
    let (origin, u, v) = match axis {
        Some(k) => {
            let unit_along = |i: usize| {
                let mut e = [0.0; 3];
                e[i] = 1.0;
                e
            };
            let mut origin = [0.0; 3];
            origin[k] = first[k];
            (origin, unit_along((k + 1) % 3), unit_along((k + 2) % 3))
        }
        None => {
            let far = all().fold(first, |m, p| if norm(sub(*p, first)) > norm(sub(m, first)) { *p } else { m });
            let u = normalized(sub(far, first));
            (first, u, cross(n, u))
        }
    };
    let flat = |pts: &[[f64; 3]]| pts.iter().map(|p| {
        let d = sub(*p, origin);
        [dot(d, u), dot(d, v)]
    }).collect::<Vec<_>>();
    Some(Sheet::Polygon { points: flat(points), holes: holes.iter().map(|h| flat(h)).collect(), position: origin, u, v })
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
    /// Priority handed out by the last `bring_to_front`.
    top_priority: u64,
    edge_ops: Vec<EdgeOp>,
    /// Targets reduced to what they share with their tools (used up).
    intersects: Vec<(ObjId, Vec<ObjId>)>,
    /// Solid objects fused into one region (the first keeps its region).
    unions: Vec<Vec<ObjId>>,
    /// Target sizes on faces, by origin.
    face_maxh: Vec<(Vec<FaceSel>, f64)>,
    /// Names given to face selections.
    face_names: BTreeMap<FaceSel, String>,
    maxh: Option<f64>,
    size_points: Vec<([f64; 3], f64)>,
    /// Factor on every target size (global, object, material, face and
    /// size point) when the scene is realized: above 1 coarsens, below
    /// refines, the size relations stay.
    size_scale: f64,
    realized: OnceLock<Result<Realized, String>>,
}

/// Whether face `f` is a piece of selection `s`.
fn matches(solids: &[Option<Solid>], s: &FaceSel, f: &Face) -> bool {
    if f.origin != s.origin {
        return false;
    }
    let region = |o: ObjId| solids.get(o).copied().flatten().map(|sol| sol.region);
    let Some(side) = s.side.and_then(region) else { return true };
    if !f.regions.contains(&side) {
        return false;
    }
    let other = if f.regions[0] == side { f.regions[1] } else { f.regions[0] };
    match s.across {
        None => true,
        Some(Across::Outside) => other == 0 || other == side,
        Some(Across::Object(o)) => region(o) == Some(other),
    }
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
            top_priority: 0,
            edge_ops: Vec::new(),
            intersects: Vec::new(),
            unions: Vec::new(),
            face_maxh: Vec::new(),
            face_names: BTreeMap::new(),
            maxh,
            size_points: Vec::new(),
            size_scale: 1.0,
            realized: OnceLock::new(),
        }
    }

    /// The factor on every target size, see [`Geometry::set_size_scale`].
    pub fn size_scale(&self) -> f64 {
        self.size_scale
    }

    /// Multiplies every target size (global, object, material, face and
    /// size point, and the size a `mesh` call passes) by `scale`.
    pub fn set_size_scale(&mut self, scale: f64) {
        if scale != self.size_scale {
            self.size_scale = scale;
            self.changed();
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
        self.objects.push(Object {
            item,
            void,
            maxh,
            material_maxh: None,
            name: None,
            alive: true,
            priority: self.top_priority,
            transforms: Vec::new(),
        });
        self.changed();
        self.objects.len() - 1
    }

    /// Replaces the item of an object in place (an extrusion turns its
    /// profile sheet into a solid): selections of the object stay valid.
    pub fn replace(&mut self, id: ObjId, item: Item) {
        self.objects[id].item = item;
        self.changed();
    }

    /// Turns solids into holes: they carve their region as before, and the
    /// region is removed from the solver mesh, leaving its walls as boundary.
    pub fn make_void(&mut self, ids: &[ObjId]) {
        for &i in ids {
            self.objects[i].void = true;
        }
        self.changed();
    }

    /// Puts objects above every other one, in the given order: each carves
    /// its region out of everything below it, the last one wins.
    pub fn bring_to_front(&mut self, ids: &[ObjId]) {
        for &i in ids {
            self.top_priority += 1;
            self.objects[i].priority = self.top_priority;
        }
        self.changed();
    }

    /// Combines the sheet `target` with the sheets `tools`: a union keeps
    /// every piece under the target's tag and leaves the merge to rapidmesh,
    /// a difference or intersection becomes rapidmesh's exact boolean in the
    /// common plane, tried once here so a bad operand fails now. The target
    /// becomes the result, the tools are used up.
    pub fn sheet_boolean(&mut self, op: SheetOp, target: ObjId, tools: &[ObjId]) -> Result<(), String> {
        let mut pieces = self.pieces(target)?;
        let others = tools.iter().map(|&i| self.pieces(i)).collect::<Result<Vec<_>, _>>()?;
        let item = if op == SheetOp::Union {
            pieces.extend(others.into_iter().flatten());
            Item::Sheets(pieces)
        } else {
            let piece = Piece::Boolean { op, target: pieces, tools: others, after: Vec::new() };
            piece.add(&mut rapidmesh::Geometry::new(None), 1, None).map_err(|e| e.to_string())?;
            Item::Sheets(vec![piece])
        };
        self.objects[target].item = item;
        self.objects[target].transforms = Vec::new();
        for &i in tools {
            self.objects[i].alive = false;
        }
        self.changed();
        Ok(())
    }

    /// The pieces of a sheet object, each with the object's transforms
    /// after its own.
    fn pieces(&self, id: ObjId) -> Result<Vec<Piece>, String> {
        let o = &self.objects[id];
        match &o.item {
            Item::Sheet(s) => Ok(vec![Piece::Sheet(s.clone(), o.transforms.clone())]),
            Item::Sheets(v) => Ok(v.iter().map(|p| p.then(&o.transforms)).collect()),
            _ => Err(format!("a sheet combines with sheets only, object {id} is a solid")),
        }
    }

    /// Removes an object from the scene.
    pub fn remove(&mut self, id: ObjId) {
        self.objects[id].alive = false;
        self.changed();
    }

    pub fn set_name(&mut self, id: ObjId, name: Option<String>) {
        self.objects[id].name = name;
    }

    /// Names a face selection (`None` clears the name).
    pub fn set_face_name(&mut self, sel: FaceSel, name: Option<String>) {
        match name {
            Some(n) => self.face_names.insert(sel, n),
            None => self.face_names.remove(&sel),
        };
    }

    pub fn face_name(&self, sel: &FaceSel) -> Option<&str> {
        self.face_names.get(sel).map(String::as_str)
    }

    pub fn set_object_maxh(&mut self, id: ObjId, maxh: Option<f64>) {
        self.objects[id].maxh = maxh;
        self.changed();
    }

    /// The target size an object takes from its material.
    pub fn set_material_maxh(&mut self, id: ObjId, maxh: Option<f64>) {
        if self.objects[id].material_maxh != maxh {
            self.objects[id].material_maxh = maxh;
            self.changed();
        }
    }

    /// The last target size set on the faces of `sel`.
    pub fn face_maxh(&self, sel: &FaceSel) -> Option<f64> {
        self.face_maxh.iter().rev().find(|(sels, _)| sels.contains(sel)).map(|&(_, h)| h)
    }

    /// Errors unless every object of `ids` is a solid.
    pub fn check_solids(&self, ids: &[ObjId]) -> Result<(), String> {
        match ids.iter().find(|&&i| !self.objects[i].is_solid()) {
            Some(i) => Err(format!("object {i} is not a solid")),
            None => Ok(()),
        }
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

    /// The solids of the STEP file at `path`, one object each, in the
    /// file's unit, and the length of that unit in metres.
    pub fn add_step(&mut self, path: &std::path::Path, maxh: Option<f64>) -> Result<(Vec<ObjId>, f64), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let step = rapidmesh_step::read(&text, rapidmesh_step::Tolerance::default())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let ids = (0..step.bodies.len())
            .map(|body| self.push(Item::Step { path: path.to_path_buf(), body }, maxh, false))
            .collect();
        Ok((ids, step.metres_per_unit))
    }

    /// Moves, turns, mirrors or stretches an object; its faces keep their
    /// origin.
    pub fn transform(&mut self, id: ObjId, t: Transform) {
        self.objects[id].transforms.push(t);
        self.changed();
    }

    /// A copy of an object in the same place, with its own region.
    pub fn copy(&mut self, id: ObjId) -> ObjId {
        let mut o = self.objects[id].clone();
        o.name = None;
        o.priority = self.top_priority;
        self.objects.push(o);
        self.changed();
        self.objects.len() - 1
    }

    /// `target` becomes what it shares with every tool; the tools are used
    /// up.
    pub fn intersect(&mut self, target: ObjId, tools: Vec<ObjId>) {
        self.intersects.push((target, tools));
        self.changed();
    }

    /// Turns a sheet into the solid it sweeps along `vector`, in place: the
    /// object keeps its id and the sheet becomes its bottom face (role 0, top
    /// 1, then the walls). An unmoved xy polygon or rectangle swept along z
    /// becomes a prism.
    pub fn extrude(&mut self, id: ObjId, vector: [f64; 3]) -> Result<(), String> {
        let [vx, vy, h] = vector;
        let o = &self.objects[id];
        let prism = match &o.item {
            _ if vx != 0.0 || vy != 0.0 || !o.transforms.is_empty() => None,
            Item::Sheet(Sheet::Polygon { points, holes, position, u, v }) if *u == [1.0, 0.0, 0.0] && *v == [0.0, 1.0, 0.0] => Some(Prism {
                points: points.clone(),
                holes: holes.clone(),
                height: h,
                position: *position,
            }),
            Item::Sheet(Sheet::Rect { corner, u, v }) if u[2] == 0.0 && v[2] == 0.0 => Some(Prism {
                points: vec![[0.0, 0.0], [u[0], u[1]], [u[0] + v[0], u[1] + v[1]], [v[0], v[1]]],
                holes: Vec::new(),
                height: h,
                position: *corner,
            }),
            _ => None,
        };
        if let Some(p) = prism {
            // a negative height extrudes downwards: the same prism, shifted
            let p = if h < 0.0 {
                let q = p.position;
                Prism { height: -h, position: [q[0], q[1], q[2] + h], ..p }
            } else {
                p
            };
            self.replace(id, Item::Solid(p.into()));
            return Ok(());
        }
        let pieces = match &self.objects[id].item {
            Item::Sheet(sheet) => vec![Piece::Sheet(sheet.clone(), Vec::new())],
            Item::Sheets(pieces) => pieces.clone(),
            _ => return Err(format!("object {id} is not a sheet")),
        };
        let placed = self.objects[id].transforms.len();
        self.replace(id, Item::Extrusion { pieces, vector, placed });
        Ok(())
    }

    /// The outer vertex loop of a sheet in 3D, where its placement puts it:
    /// the profile of a loft or a revolution.
    pub fn outline(&self, id: ObjId) -> Result<Vec<[f64; 3]>, String> {
        let o = &self.objects[id];
        match &o.item {
            Item::Sheet(s) if o.alive => sheet_ops::outline(s, &o.transforms),
            Item::Sheets(_) => Err(format!("sheet {id} has several pieces, it is no profile")),
            _ => Err(format!("object {id} is not a sheet profile")),
        }
    }

    /// The ruled solid between two sheet profiles with the same vertex
    /// count, vertex `i` of one joined to vertex `i` of the other; the
    /// profiles are used up.
    pub fn loft(&mut self, a: ObjId, b: ObjId, maxh: Option<f64>) -> Result<ObjId, String> {
        let (profile_a, profile_b) = (self.outline(a)?, self.outline(b)?);
        if profile_a.len() != profile_b.len() {
            return Err("both profiles need the same vertex count".into());
        }
        let id = self.add_solid(Loft { profile_a, profile_b }, maxh, false);
        self.remove(a);
        self.remove(b);
        Ok(id)
    }

    /// The solid a sheet profile sweeps turning by `angle` radians about the
    /// axis along `axis` through `point`; the profile is used up.
    pub fn revolve(&mut self, id: ObjId, point: [f64; 3], axis: [f64; 3], angle: f64, maxh: Option<f64>) -> Result<ObjId, String> {
        let a = normalized(axis);
        let rz = self
            .outline(id)?
            .into_iter()
            .map(|p| {
                let d = sub(p, point);
                let z = dot(d, a);
                [norm(sub(d, scale(a, z))), z]
            })
            .collect();
        let mut r = Revolve::new(rz);
        r.position = point;
        r.axis = a;
        r.angle = angle * (180.0 / std::f64::consts::PI);
        let solid = self.add_solid(r, maxh, false);
        self.remove(id);
        Ok(solid)
    }

    /// A round tube along the Catmull-Rom spline through `points` (eight
    /// points per span, a 16-gon section), its radius that of the unmoved
    /// disc `profile`, which is used up.
    pub fn sweep(&mut self, profile: ObjId, points: &[[f64; 3]], maxh: Option<f64>) -> Result<ObjId, String> {
        let o = &self.objects[profile];
        let radius = match &o.item {
            Item::Sheet(Sheet::Disc { radius, .. }) if o.transforms.is_empty() && o.alive => *radius,
            _ => {
                return Err("the profile must be a disc from Geometry.disc, not moved after \
                            (rapidmesh sweeps round tubes)"
                    .into())
            }
        };
        if points.len() < 2 {
            return Err("a sweep needs at least two path points".into());
        }
        let id = self.add_solid(Sweep::new(spline(points, 8), radius), maxh, false);
        self.remove(profile);
        Ok(id)
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
        let scaled = |h: Option<f64>| h.map(|h| h * self.size_scale);
        let mut g = rapidmesh::Geometry::new(scaled(self.maxh));
        let mut solids = vec![None; self.objects.len()];
        // the solids of each STEP file, imported all at once
        let mut steps: std::collections::HashMap<&std::path::Path, Vec<Solid>> = Default::default();
        let mut order: Vec<ObjId> = (0..self.objects.len()).collect();
        order.sort_by_key(|&i| (self.objects[i].priority, i));
        for i in order {
            let o = &self.objects[i];
            if !o.alive {
                continue;
            }
            let e = |e: rapidmesh::Error| e.to_string();
            let (placed, done): (Vec<RmObject>, usize) = match &o.item {
                Item::Solid(shape) => {
                    let s = g.add_solid(shape.clone(), scaled(o.size()), false).map_err(e)?;
                    solids[i] = Some(s);
                    (vec![s.into()], 0)
                }
                Item::Sheet(sheet) => (vec![g.add_sheet(sheet, sheet_tag(i), scaled(o.size())).map_err(e)?.into()], 0),
                Item::Sheets(pieces) => {
                    let mut placed: Vec<RmObject> = Vec::new();
                    for piece in pieces {
                        placed.push(piece.add(&mut g, sheet_tag(i), scaled(o.size())).map_err(e)?.into());
                    }
                    (placed, 0)
                }
                Item::Extrusion { pieces, vector, placed } => {
                    let (mut sheets, mut swept): (Vec<RmObject>, _) = (Vec::new(), Vec::new());
                    for piece in pieces {
                        let r = piece.then(&o.transforms[..*placed]).add(&mut g, sheet_tag(i), scaled(o.size())).map_err(e)?;
                        swept.push(g.extrude(r, *vector, scaled(o.size())).map_err(e)?);
                        sheets.push(r.into());
                    }
                    // the pieces' solids are one material, as their sheets are one sheet
                    let s = g.union(&swept).map_err(e)?;
                    solids[i] = Some(s);
                    sheets.extend(swept.into_iter().map(|s| -> RmObject { s.into() }));
                    (sheets, *placed)
                }
                Item::Step { path, body } => {
                    if !steps.contains_key(path.as_path()) {
                        let removed = self.objects.iter().any(|o| {
                            !o.alive && matches!(&o.item, Item::Step { path: p, .. } if p == path)
                        });
                        if removed {
                            return Err(format!("{}: a STEP file's solids go together, remove all or none", path.display()));
                        }
                        steps.insert(path.as_path(), g.import_step(path, None).map_err(e)?);
                    }
                    let s = *steps[path.as_path()]
                        .get(*body)
                        .ok_or_else(|| format!("{}: no solid {body}", path.display()))?;
                    if let Some(h) = scaled(o.size()) {
                        g.set_maxh_on(&Scope::region(Some(s.region)), h).map_err(e)?;
                    }
                    solids[i] = Some(s);
                    (vec![s.into()], 0)
                }
            };
            for &tr in &o.transforms[done..] {
                for &p in &placed {
                    g.transform(p, tr).map_err(e)?;
                }
            }
        }
        for (target, tools) in &self.intersects {
            let (Some(t), tools) = (
                solids.get(*target).copied().flatten(),
                tools.iter().filter_map(|&i| solids.get(i).copied().flatten()).collect::<Vec<_>>(),
            ) else {
                return Err(format!("intersect: object {target} is not a solid"));
            };
            g.intersect(t, &tools).map_err(|e| e.to_string())?;
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
            // holes merge too: fused pieces of one conductor are one hole
            let s: Vec<Solid> = ids.iter().filter_map(|&i| solids.get(i).copied().flatten()).collect();
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
            g.add_size_point(p, h * self.size_scale);
        }
        let topology = g.topology().map_err(|e| e.to_string())?;
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
                    let object = (f.face_tag - 1) as ObjId;
                    match self.objects[object].item {
                        // the sheet under an extrusion is the solid's bottom
                        Item::Extrusion { .. } => FaceOrigin::Solid { object, role: 0 },
                        _ => FaceOrigin::Sheet { object },
                    }
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
                    bbox: [f.bbox[0][0], f.bbox[0][1], f.bbox[0][2], f.bbox[1][0], f.bbox[1][1], f.bbox[1][2]],
                })
            })
            .collect();
        let mut realized = Realized { geometry: g, solids, topology, faces };
        for (sels, h) in &self.face_maxh {
            let ids: Vec<u32> = realized
                .faces
                .iter()
                .filter(|f| sels.iter().any(|s| matches(&realized.solids, s, f)))
                .map(|f| f.id)
                .collect();
            for id in ids {
                let filter = FaceFilter { id: Some(id), ..FaceFilter::default() };
                realized.geometry.set_maxh_on(&Scope::surf(Some(filter)), *h * self.size_scale).map_err(|e| e.to_string())?;
            }
        }
        Ok(realized)
    }

    /// The region of a solid object in the realised model.
    pub fn region(&self, id: ObjId) -> Result<Option<u32>, String> {
        Ok(self.realized()?.solids[id].filter(|_| !self.objects[id].void).map(|s| s.region))
    }

    /// The regions of the holes, to be removed from the solver mesh.
    pub fn hole_regions(&self) -> Result<Vec<u32>, String> {
        let r = self.realized()?;
        Ok((0..self.objects.len())
            .filter(|&i| self.objects[i].void && self.objects[i].alive)
            .filter_map(|i| r.solids[i].map(|s| s.region))
            .collect())
    }

    /// The faces bounding an object: for a solid every face with its region
    /// on one side (interfaces with later solids included; for a hole, its
    /// walls), for a sheet the sheet itself. Solid faces carry the object as
    /// their side.
    pub fn faces_of(&self, id: ObjId) -> Result<Vec<FaceSel>, String> {
        let r = self.realized()?;
        let o = &self.objects[id];
        let mut out: Vec<FaceSel> = match (&o.item, r.solids[id]) {
            (Item::Sheet(_) | Item::Sheets(_), _) => {
                vec![FaceSel { origin: FaceOrigin::Sheet { object: id }, side: None, across: None }]
            }
            (Item::Solid(_) | Item::Extrusion { .. } | Item::Step { .. }, Some(s)) => r
                .faces
                .iter()
                .filter(|f| f.regions.contains(&s.region))
                .map(|f| {
                    let other = if f.regions[0] == s.region { f.regions[1] } else { f.regions[0] };
                    let across = if other == 0 || other == s.region {
                        Across::Outside
                    } else {
                        match self.object_of_region(other) {
                            Some(o) => Across::Object(o),
                            None => Across::Outside,
                        }
                    };
                    FaceSel { origin: f.origin, side: Some(id), across: Some(across) }
                })
                .collect(),
            (Item::Solid(_) | Item::Extrusion { .. } | Item::Step { .. }, None) => Vec::new(),
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
        out.sort_by_key(|a| a.0);
        out.dedup_by(|a, b| a.0 == b.0);
        Ok(out)
    }

    /// The volume enclosed by a solid object's own shape (before other
    /// solids carve it), for the primitives with a closed form, scaled by
    /// its stretches.
    pub fn solid_volume(&self, id: ObjId) -> Result<f64, String> {
        use std::f64::consts::PI;
        let area2 = |pts: &[[f64; 2]]| -> f64 {
            let n = pts.len();
            0.5 * (0..n)
                .map(|i| pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1])
                .sum::<f64>()
                .abs()
        };
        let o = &self.objects[id];
        let Item::Solid(shape) = &o.item else {
            return Err(format!("object {id}: no closed-form volume (not a primitive solid)"));
        };
        if self.intersects.iter().any(|(t, _)| *t == id) {
            return Err(format!("object {id}: no closed-form volume (intersected)"));
        }
        let scale: f64 = o
            .transforms
            .iter()
            .map(|t| match t {
                Transform::Stretch { factors: f, .. } => (f[0] * f[1] * f[2]).abs(),
                _ => 1.0,
            })
            .product();
        Ok(scale * match shape {
            Shape::Cuboid(c) => c.size[0] * c.size[1] * c.size[2],
            Shape::Prism(p) => {
                (area2(&p.points) - p.holes.iter().map(|h| area2(h)).sum::<f64>()) * p.height
            }
            Shape::Cylinder(c) => PI * c.radius * c.radius * c.height,
            Shape::Cone(c) => PI * c.height * (c.r1 * c.r1 + c.r1 * c.r2 + c.r2 * c.r2) / 3.0,
            Shape::Sphere(s) => 4.0 / 3.0 * PI * s.radius.powi(3),
            Shape::Torus(t) => 2.0 * PI * PI * t.major_radius * t.minor_radius * t.minor_radius,
            Shape::Wedge(w) => 0.5 * (w.size[0] + w.top_x) * w.size[1] * w.size[2],
            _ => return Err(format!("object {id}: no closed-form volume for this shape")),
        })
    }

    /// The first object whose solid holds `region`.
    fn object_of_region(&self, region: u32) -> Option<ObjId> {
        let r = self.realized().ok()?;
        (0..self.objects.len()).find(|&i| {
            r.solids.get(i).copied().flatten().is_some_and(|s| s.region == region)
        })
    }

    /// The current B-rep faces of a set of selections.
    pub fn resolve(&self, sels: &[FaceSel]) -> Result<Vec<&Face>, String> {
        let r = self.realized()?;
        Ok(r.faces.iter().filter(|f| sels.iter().any(|s| matches(&r.solids, s, f))).collect())
    }

    /// Where a face selection lies, over the faces it currently resolves to.
    pub fn face_extent(&self, sel: &FaceSel) -> Result<Extent, String> {
        let faces = self.resolve(std::slice::from_ref(sel))?;
        let area: f64 = faces.iter().map(|f| f.area).sum();
        let mut centroid = [0.0; 3];
        let mut bbox = Extent::EMPTY;
        for f in &faces {
            Extent::grow(&mut bbox, &f.bbox);
            for k in 0..3 {
                centroid[k] += f.centroid[k] * f.area / area.max(f64::MIN_POSITIVE);
            }
        }
        Ok(Extent { centroid, area, bbox })
    }

    /// Where an object lies: the box around its faces and the box's centre
    /// (all zero for an object without faces).
    pub fn object_extent(&self, id: ObjId) -> Result<Extent, String> {
        let sels = self.faces_of(id)?;
        if sels.is_empty() {
            return Ok(Extent { centroid: [0.0; 3], area: 0.0, bbox: [0.0; 6] });
        }
        let parts = sels.iter().map(|s| self.face_extent(s)).collect::<Result<Vec<_>, _>>()?;
        let bbox = Extent::union(&parts);
        let centroid = [0, 1, 2].map(|k| (bbox[k] + bbox[k + 3]) / 2.0);
        Ok(Extent { centroid, area: 0.0, bbox })
    }

    /// Turns every solid named `name` into a hole, its walls boundary faces
    /// that carry a surface condition, and returns the walls with the
    /// volume-to-surface thickness `2V/S` in metres (for a long trace of
    /// width w and thickness t, `w t / (w + t)`: the thickness at which a
    /// two-sided surface impedance on every wall reproduces the DC
    /// resistance `1/(sigma w t)`). `None` when no solid has the name.
    pub fn hollow(&mut self, name: &str) -> Result<Option<(Vec<FaceSel>, f64)>, String> {
        let ids: Vec<ObjId> = (0..self.objects.len())
            .filter(|&i| {
                let o = &self.objects[i];
                o.alive && o.is_solid() && o.name.as_deref() == Some(name)
            })
            .collect();
        if ids.is_empty() {
            return Ok(None);
        }
        self.make_void(&ids);
        let mut walls = Vec::new();
        for &i in &ids {
            walls.extend(self.faces_of(i)?);
        }
        walls.sort_by_key(FaceSel::key);
        walls.dedup();
        let mut area = 0.0;
        for s in &walls {
            area += self.face_extent(s)?.area;
        }
        // Volume from the shapes themselves: walls shared with another hole
        // (a contact) are not in the model, so the walls do not close.
        let mut volume = 0.0;
        for &i in &ids {
            volume += self.solid_volume(i)?;
        }
        if volume <= 0.0 || area <= 0.0 {
            return Err(format!("'{name}' encloses no volume"));
        }
        Ok(Some((walls, 2.0 * volume / area)))
    }

    /// Sizes every object thinner than `base_maxh` that has no size of its
    /// own or from its material: `d / resolution`, at least `min_maxh`, `d`
    /// its smallest nonzero bounding-box extent. Returns the objects sized,
    /// by their name (else their kind and centre in mm), with the size.
    pub fn auto_refine(&mut self, base_maxh: f64, resolution: f64, min_maxh: Option<f64>) -> Result<Vec<(String, f64)>, String> {
        let mut out = Vec::new();
        for id in 0..self.objects.len() {
            let o = &self.objects[id];
            if !o.alive || o.maxh.is_some() || o.material_maxh.is_some() {
                continue;
            }
            let e = self.object_extent(id)?;
            let b = e.bbox;
            let thinnest = [b[3] - b[0], b[4] - b[1], b[5] - b[2]].into_iter().filter(|&d| d > 0.0).reduce(f64::min);
            let Some(d) = thinnest else { continue };
            if d >= base_maxh {
                continue;
            }
            let h = match min_maxh {
                Some(m) => (d / resolution).max(m),
                None => d / resolution,
            };
            let o = &self.objects[id];
            let what = match o.name.as_deref() {
                Some(n) if !n.is_empty() => n.to_string(),
                _ => {
                    let kind = if o.is_solid() { "vol" } else { "sheet" };
                    let c = e.centroid.map(|v| v * 1e3);
                    format!("{kind}@({:.1},{:.1},{:.1})mm", c[0], c[1], c[2])
                }
            };
            self.set_object_maxh(id, Some(h));
            out.push((what, h));
        }
        Ok(out)
    }

    /// Whether `sels` hold every face of a solid that is still meshed (not
    /// a hole), so that a surface condition on them would sit inside the
    /// mesh.
    pub fn covers_solid_shell(&self, sels: &[FaceSel]) -> bool {
        (0..self.objects.len()).any(|id| {
            let o = &self.objects[id];
            if !o.alive || !o.is_solid() || o.void {
                return false;
            }
            match self.faces_of(id) {
                Ok(shell) => !shell.is_empty() && shell.iter().all(|s| sels.contains(s)),
                Err(_) => false,
            }
        })
    }

    /// Tagged face groups (selections) and volume groups (objects) as the
    /// B-rep faces and regions they currently hold.
    ///
    /// A region holds one material: where objects of several of the
    /// `materials` tags share a region (fused solids, each filled on its
    /// own), it goes to the one in front (higher priority, or equal and
    /// added later). Other volume groups (a PML) keep every region of their
    /// objects.
    pub fn groups(
        &self,
        faces: &[(i32, Vec<FaceSel>)],
        volumes: &[(i32, Vec<ObjId>)],
        materials: &[i32],
    ) -> Result<(Vec<Group>, Vec<Group>), String> {
        let mut fg = Vec::with_capacity(faces.len());
        for (tag, sels) in faces {
            fg.push(Group { tag: *tag, ids: self.resolve(sels)?.iter().map(|f| f.id).collect() });
        }
        // region -> (priority, object, material tag) of the object in front
        let mut owner: BTreeMap<u32, (u64, ObjId, i32)> = BTreeMap::new();
        for (tag, objs) in volumes.iter().filter(|(t, _)| materials.contains(t)) {
            for &o in objs {
                let key = (self.objects[o].priority, o, *tag);
                if let Some(r) = self.region(o)? {
                    let e = owner.entry(r).or_insert(key);
                    if (key.0, key.1) > (e.0, e.1) {
                        *e = key;
                    }
                }
            }
        }
        let mut vg = Vec::with_capacity(volumes.len());
        for (tag, objs) in volumes {
            let mut ids = Vec::new();
            for &o in objs {
                if let Some(r) = self.region(o)? {
                    let mine = !materials.contains(tag) || owner.get(&r).is_some_and(|e| e.2 == *tag);
                    if mine && !ids.contains(&r) {
                        ids.push(r);
                    }
                }
            }
            vg.push(Group { tag: *tag, ids });
        }
        Ok((fg, vg))
    }

    /// The bounding box `(xmin, ymin, zmin, xmax, ymax, zmax)` of every
    /// face of the realised scene.
    pub fn bbox(&self) -> Result<[f64; 6], String> {
        let r = self.realized()?;
        let mut b = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for f in &r.faces {
            for k in 0..3 {
                b[k] = b[k].min(f.bbox[k]);
                b[k + 3] = b[k + 3].max(f.bbox[k + 3]);
            }
        }
        Ok(b)
    }

    /// A coarse triangulation of every B-rep face, for previews: per face
    /// id its triangles' corners (nine values each) and flat normals (one
    /// per corner), about `target_triangles` in all.
    pub fn preview(&self, target_triangles: usize) -> Result<BTreeMap<u32, (Vec<f64>, Vec<f64>)>, String> {
        let b = self.bbox()?;
        let diag = ((b[3] - b[0]).powi(2) + (b[4] - b[1]).powi(2) + (b[5] - b[2]).powi(2)).sqrt();
        let opts = SurfaceOptions {
            maxh: Some((diag / 10.0).max(1e-9)),
            target_triangles: Some(target_triangles),
            ..SurfaceOptions::default()
        };
        let m = self.realized()?.geometry.surface_mesh(&opts).map_err(|e| e.to_string())?;
        let mut out: BTreeMap<u32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for f in m.faces.iter().filter(|f| f.patch != rapidmesh::NONE) {
            let [a, b, c] = f.tri.map(|v| m.points[v]);
            let n = rapidfem_core::geom::tri_area_vector(a, b, c);
            let l = rapidfem_core::geom::norm(n).max(f64::MIN_POSITIVE);
            let (pos, nor) = out.entry(f.patch).or_default();
            for p in [a, b, c] {
                pos.extend(p);
                nor.extend(n.map(|x| x / l));
            }
        }
        Ok(out)
    }

    /// Meshes the realised scene.
    pub fn mesh(&self, opts: &MeshOptions) -> Result<rapidmesh::Mesh, String> {
        let scaled = |h: Option<f64>| h.map(|h| h * self.size_scale);
        let opts = MeshOptions {
            maxh: scaled(opts.maxh),
            maxh_edge: scaled(opts.maxh_edge),
            maxh_surf: scaled(opts.maxh_surf),
            maxh_vol: scaled(opts.maxh_vol),
            min_h_surf: opts.min_h_surf * self.size_scale,
            ..opts.clone()
        };
        self.realized()?.geometry.mesh(&opts).map_err(|e| e.to_string())
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
        let floor = FaceSel { origin: FaceOrigin::Solid { object: air, role: 0 }, side: Some(air), across: None };
        assert!(g.resolve(&[floor]).unwrap().is_empty());
    }

    #[test]
    fn origins_survive_a_later_change() {
        let (mut g, air, _) = substrate_in_air();
        let top = FaceSel { origin: FaceOrigin::Solid { object: air, role: 1 }, side: None, across: None }; // +z of the air box
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
        let rh = g.hole_regions().unwrap();
        assert_eq!(rh.len(), 1);
        let walls = g.faces_of(hole).unwrap();
        assert_eq!(walls.len(), 6);
        let ra = g.region(air).unwrap().unwrap();
        for f in g.resolve(&walls).unwrap() {
            assert!(f.regions.contains(&ra) && f.regions.contains(&rh[0]), "{:?}", f.regions);
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

    #[test]
    fn bring_to_front_reverses_the_carving() {
        let mut g = Geometry::new(Some(1.0));
        let inner = g.add_solid(Cuboid::new([1.0, 1.0, 1.0]).at([1.0, 1.0, 1.0]), None, false);
        let outer = g.add_solid(Cuboid::new([3.0, 3.0, 3.0]), None, false);
        // added later, the outer box swallows the inner one
        assert!(g.faces_of(inner).unwrap().is_empty());
        g.bring_to_front(&[inner]);
        assert_eq!(g.faces_of(inner).unwrap().len(), 6);
        assert!(g.region(outer).unwrap().is_some());
    }

    #[test]
    fn transforms_move_an_object_and_keep_its_origins() {
        let mut g = Geometry::new(Some(1.0));
        let air = g.add_solid(Cuboid::new([6.0, 6.0, 6.0]), None, false);
        let b = g.add_solid(Cuboid::new([1.0, 1.0, 1.0]), None, false);
        g.transform(b, Transform::Translate([2.0, 2.0, 2.0]));
        g.transform(b, Transform::Rotate { angle: std::f64::consts::FRAC_PI_2, axis: [0.0, 0.0, 1.0], center: [2.5, 2.5, 2.5] });
        let top = FaceSel { origin: FaceOrigin::Solid { object: b, role: 1 }, side: None, across: None };
        let f = g.resolve(&[top]).unwrap();
        assert_eq!(f.len(), 1);
        assert!((f[0].centroid[2] - 3.0).abs() < 1e-9 && (f[0].area - 1.0).abs() < 1e-9);
        assert_eq!(f[0].bbox, [2.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
        assert_ne!(g.region(air).unwrap(), g.region(b).unwrap());
    }

    #[test]
    fn a_copy_gets_its_own_region() {
        let mut g = Geometry::new(Some(1.0));
        g.add_solid(Cuboid::new([6.0, 2.0, 2.0]), None, false);
        let b = g.add_solid(Cuboid::new([1.0, 1.0, 1.0]).at([0.5, 0.5, 0.5]), None, false);
        let c = g.copy(b);
        g.transform(c, Transform::Translate([3.0, 0.0, 0.0]));
        let (rb, rc) = (g.region(b).unwrap().unwrap(), g.region(c).unwrap().unwrap());
        assert_ne!(rb, rc);
        assert_eq!(g.faces_of(c).unwrap().len(), 6);
    }

    #[test]
    fn intersect_keeps_the_common_part() {
        let mut g = Geometry::new(Some(0.5));
        let a = g.add_solid(Cuboid::new([2.0, 2.0, 2.0]), None, false);
        let b = g.add_solid(Cuboid::new([2.0, 2.0, 2.0]).at([1.0, 1.0, 1.0]), None, false);
        g.intersect(a, vec![b]);
        let faces = g.resolve(&g.faces_of(a).unwrap()).unwrap();
        let area: f64 = faces.iter().map(|f| f.area).sum();
        assert!((area - 6.0).abs() < 1e-9, "area {area}");
    }

    #[test]
    fn a_sheet_extrudes_along_a_vector() {
        let mut g = Geometry::new(Some(0.5));
        g.add_solid(Cuboid::new([4.0, 4.0, 4.0]), None, false);
        let s = g.add_sheet(Sheet::xy(1.0, 1.0, [1.0, 1.0, 1.0]), None);
        g.extrude(s, [0.5, 0.0, 1.0]).unwrap();
        let faces = g.resolve(&g.faces_of(s).unwrap()).unwrap();
        assert_eq!(faces.len(), 6);
        let bottom = FaceSel { origin: FaceOrigin::Solid { object: s, role: 0 }, side: None, across: None };
        assert_eq!(g.resolve(&[bottom]).unwrap().len(), 1);
        let top = FaceSel { origin: FaceOrigin::Solid { object: s, role: 1 }, side: None, across: None };
        let t = g.resolve(&[top]).unwrap();
        assert_eq!(t.len(), 1);
        assert!((t[0].centroid[0] - 2.0).abs() < 1e-9 && (t[0].centroid[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_sheet_moved_before_extrusion_sweeps_in_world_axes() {
        let mut g = Geometry::new(Some(0.5));
        g.add_solid(Cuboid::new([4.0, 4.0, 4.0]), None, false);
        let s = g.add_sheet(Sheet::xy(1.0, 1.0, [1.0, 1.0, 1.0]), None);
        // stood up into the xz-plane, then swept along +y
        g.transform(s, Transform::Rotate { angle: std::f64::consts::FRAC_PI_2, axis: [1.0, 0.0, 0.0], center: [1.0, 1.0, 1.0] });
        g.extrude(s, [0.0, 1.0, 0.0]).unwrap();
        g.transform(s, Transform::Translate([0.0, 0.0, 0.5]));
        let top = FaceSel { origin: FaceOrigin::Solid { object: s, role: 1 }, side: None, across: None };
        let t = g.resolve(&[top]).unwrap();
        assert_eq!(t.len(), 1);
        assert!((t[0].normal[1].abs() - 1.0).abs() < 1e-9, "{:?}", t[0].normal);
        assert_eq!(t[0].bbox, [1.0, 2.0, 1.5, 2.0, 2.0, 2.5]);
    }

    #[test]
    fn a_hollowed_solid_reports_its_volume_to_surface_thickness() {
        let mut g = Geometry::new(Some(1.0));
        g.add_solid(Cuboid::new([6.0, 6.0, 6.0]), None, false);
        let b = g.add_solid(Cuboid::new([4.0, 1.0, 0.5]).at([1.0, 1.0, 1.0]), None, false);
        g.set_name(b, Some("metal".into()));
        assert!(g.hollow("copper").unwrap().is_none());
        let (walls, t) = g.hollow("metal").unwrap().unwrap();
        assert!(g.object(b).void);
        assert_eq!(walls.len(), 6);
        // 2V/S = 2 * 2 / (2 * (4 * 1 + 4 * 0.5 + 1 * 0.5))
        assert!((t - 4.0 / 13.0).abs() < 1e-9, "{t}");
    }

    #[test]
    fn sheet_cuts_need_one_plane_and_keep_a_round_hole() {
        let mut g = Geometry::new(Some(0.5));
        let plate = g.add_sheet(Sheet::xy(4.0, 4.0, [0.0, 0.0, 1.5]), None);
        let above = g.add_sheet(Sheet::xy(1.0, 1.0, [0.0, 0.0, 2.0]), None);
        let err = g.sheet_boolean(SheetOp::Difference, plate, &[above]).unwrap_err();
        assert!(err.contains("one plane"), "{err}");
        let disc = g.add_sheet(Sheet::disc(1.0, [2.0, 2.0, 1.5], [0.0, 0.0, 1.0]), None);
        g.sheet_boolean(SheetOp::Difference, plate, &[disc]).unwrap();
        g.remove(above);
        assert!(g.realized().is_ok());
    }

    #[test]
    fn polygons_in_any_plane_loft() {
        let square = |x: f64, s: f64| vec![[x, -s, -s], [x, s, -s], [x, s, s], [x, -s, s]];
        let a = polygon_sheet(&square(0.0, 1.0), &[]).unwrap();
        assert!(matches!(a, Sheet::Polygon { u: [0.0, 1.0, 0.0], v: [0.0, 0.0, 1.0], .. }));
        assert!(polygon_sheet(&[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 2.0, 0.5]], &[]).is_none());
        let flat = polygon_sheet(&[[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], &[]).unwrap();
        assert!(matches!(flat, Sheet::Polygon { position: [_, _, z], .. } if z == 0.5));
        let mut g = Geometry::new(Some(0.5));
        let pa = g.add_sheet(a, None);
        let pb = g.add_sheet(polygon_sheet(&square(2.0, 2.0), &[]).unwrap(), None);
        assert_eq!(g.outline(pb).unwrap()[2], [2.0, 2.0, 2.0]);
        let horn = g.loft(pa, pb, None).unwrap();
        assert!(g.object(horn).is_solid());
        assert!(!g.object(pa).alive && !g.object(pb).alive);
        let tri = g.add_sheet(flat, None);
        assert!(g.loft(tri, horn, None).is_err());
    }

    #[test]
    fn selections_pick_extremes_and_faces_on_a_box() {
        let e = |c: [f64; 3], b: [f64; 6]| Extent { centroid: c, area: 0.0, bbox: b };
        let items = [
            e([0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
            e([0.5, 0.5, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
            e([0.5, 0.5, 0.5], [0.2, 0.2, 0.5, 0.8, 0.8, 0.5]),
        ];
        assert_eq!(Extent::extreme(&items, 2, false, 1e-9), vec![0]);
        assert_eq!(Extent::extreme(&items, 2, true, 1e-9), vec![1]);
        let b = Extent::union(&items);
        assert_eq!(b, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(Extent::on_box(&items, &b, 1e-9), vec![0, 1]);
    }
}
