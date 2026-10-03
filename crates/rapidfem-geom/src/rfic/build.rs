// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! One-call RFIC model: a GDS layout and a process stack in, a solve-ready
//! scene out, in the gds2palace modelling conventions the SG13G2
//! measurement validation pinned down:
//!
//! - the background dielectric slabs span the layout plus a margin
//! - an air shell wraps the dielectric stack on all six sides, an absorbing
//!   boundary (or one PML per box) on its outside, optionally a PEC floor
//! - metals become surface impedances on the walls of their holes, vias
//!   homogenised anisotropic volume conductors, LOWLOSS layers PEC
//! - ports are vertical lumped plates, placed explicitly or read from GDS
//!   marker layers
//!
//! The scene is not meshed.

use std::collections::BTreeMap;
use std::path::Path;

use rapidfem_core::model::{FaceSpec, PmlSpec};
use rapidmesh::shapes::{Cuboid, Sheet};

use super::gds::{self, Layout};
use super::region::{self, Region};
use super::stack::{PdkLayer, Stack, StackMaterial};
use super::Scene;
use crate::geometry::{Extent, FaceSel, ObjId, SELECT_TOL};
use crate::setup::{Condition, Physics, Target};

/// Gds2palace homogenised via arrays: full conductivity vertically, one
/// tenth laterally (the array conducts through its posts).
pub const VIA_LATERAL_FACTOR: f64 = 0.1;

/// Thickness-to-skin-depth window in which a surface impedance misses the
/// strip resistance by more than a few percent (against a 2D quasi-static
/// reference, issues #48 and #56; the edge-corrected impedance holds from 4
/// skin depths up): "auto" meshes the conductor there.
pub const SIBC_RATIO_LOW: f64 = 1.5;
pub const SIBC_RATIO_HIGH: f64 = 4.0;

const MU0: f64 = 4e-7 * std::f64::consts::PI;

fn skin_depth(f: f64, sigma: f64) -> f64 {
    1.0 / (std::f64::consts::PI * f * MU0 * sigma).sqrt()
}

/// Mesh sizing of [`build`], lengths in metres. `scale` multiplies every
/// size; `slabs` overrides the size of a background slab by name; `graded`
/// splits a slab into stacked zones, listed top-down as (thickness, h), the
/// last one padded to the slab's remaining thickness.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "rapidfem.rfic", get_all, set_all, from_py_object))]
pub struct MeshSpec {
    pub scale: f64,
    pub conductor: f64,
    pub port: f64,
    pub global_h: f64,
    /// The slab size without an override (`global_h` when None).
    pub slab_default: Option<f64>,
    pub slabs: BTreeMap<String, f64>,
    pub graded: BTreeMap<String, Vec<(f64, f64)>>,
}

impl Default for MeshSpec {
    fn default() -> Self {
        MeshSpec { scale: 1.0, conductor: 5e-6, port: 3e-6, global_h: 40e-6, slab_default: None, slabs: BTreeMap::new(), graded: BTreeMap::new() }
    }
}

impl MeshSpec {
    pub fn h(&self, value: f64) -> f64 {
        self.scale * value
    }

    pub fn slab_h(&self, name: &str) -> f64 {
        let base = self.slabs.get(name).copied().unwrap_or(self.slab_default.unwrap_or(self.global_h));
        self.scale * base
    }

    /// The sizes from the stack and the layers drawn in the layout:
    ///
    /// - `conductor` 4x the thinnest drawn layer (finer buys nothing: on the
    ///   2 nH octagon halving it doubles the DOFs at equal mesh quality, the
    ///   worst elements sit in the passivation shell)
    /// - `port` half of that, a couple of elements across the gap
    /// - `global_h` 8x `conductor`, the filler; slabs default to it
    /// - a semiconductor slab thicker than 1.5 `global_h` is graded: its top
    ///   third at `global_h`, the rest at twice that
    ///
    /// `preset` scales everything: "fast" 2, "balanced" 1.5, "accurate" 1.
    pub fn derive(stack: &Stack, drawn: &[String], preset: &str) -> Result<MeshSpec, String> {
        let scale = match preset {
            "fast" => 2.0,
            "balanced" => 1.5,
            "accurate" => 1.0,
            _ => return Err(format!("preset must be fast|balanced|accurate, got {preset:?}")),
        };
        let t_min = stack
            .layers
            .iter()
            .filter(|l| drawn.contains(&l.name) && l.thickness > 0.0)
            .map(|l| l.thickness)
            .reduce(f64::min)
            .ok_or("no drawn layers to derive a mesh policy from")?;
        let conductor = 4.0 * t_min;
        let global_h = 8.0 * conductor;
        let mut graded = BTreeMap::new();
        for d in &stack.dielectrics {
            if stack.slab_material(d).kind == "semiconductor" && d.thickness > 1.5 * global_h {
                graded.insert(d.name.clone(), vec![(d.thickness / 3.0, global_h), (d.thickness, 2.0 * global_h)]);
            }
        }
        Ok(MeshSpec { scale, conductor, port: conductor / 2.0, global_h, graded, ..MeshSpec::default() })
    }
}

/// One end of a [`ViaPort`]: a height, or a layer whose top (lower end) or
/// bottom (upper end) it is.
#[derive(Clone, Debug, PartialEq)]
pub enum ZBound {
    Height(f64),
    Layer(String),
}

/// A vertical lumped-port plate between two stack heights, placed by
/// `span` along `axis` and `at` across it, or by the rectangles on GDS
/// layer `marker` (their wide side the plate width).
#[derive(Clone, Debug, PartialEq)]
pub struct ViaPort {
    pub z: (ZBound, ZBound),
    pub span: Option<(f64, f64)>,
    pub at: Option<f64>,
    /// 'x' or 'y'.
    pub axis: char,
    pub marker: Option<i32>,
    pub z0: f64,
}

/// The mesh policy of [`build`]: a preset derived from the stack, or given.
#[derive(Clone, Debug)]
pub enum Mesh {
    Preset(String),
    Spec(MeshSpec),
}

/// The options of [`build`] (see the Python `rfic.build` for each).
#[derive(Clone, Debug)]
pub struct Options {
    pub top_cell: Option<String>,
    pub ports: Vec<ViaPort>,
    pub margin: f64,
    pub air: f64,
    pub air_top: Option<f64>,
    pub pec_floor: bool,
    pub conductor_model: BTreeMap<String, String>,
    pub band: Option<(f64, f64)>,
    pub mesh: Mesh,
    /// "planar", "conformal" or "none".
    pub passivation: String,
    pub pass_t_side: f64,
    pub pass_t_top: Option<f64>,
    pub conformal_over: Option<String>,
    /// "abc" or "pml".
    pub boundary: String,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            top_cell: None,
            ports: Vec::new(),
            margin: 150e-6,
            air: 50e-6,
            air_top: None,
            pec_floor: false,
            conductor_model: BTreeMap::new(),
            band: None,
            mesh: Mesh::Preset("balanced".into()),
            passivation: "planar".into(),
            pass_t_side: 0.6e-6,
            pass_t_top: None,
            conformal_over: None,
            boundary: "abc".into(),
        }
    }
}

/// What [`build`] made, by object.
#[derive(Clone, Debug, Default)]
pub struct Built {
    /// Layer name and its conductor objects, bottom-up in stack order.
    pub conductors: Vec<(String, Vec<ObjId>)>,
    /// Background slab name and its boxes or prisms.
    pub slabs: Vec<(String, Vec<ObjId>)>,
    /// The six air-shell boxes: top, bottom, -x, +x, -y, +y.
    pub air_shell: Vec<ObjId>,
    /// The port plates (each carries a lumped port).
    pub ports: Vec<ObjId>,
    /// (x0, y0, x1, y1), the layout bounding box plus margin.
    pub footprint: [f64; 4],
    /// The global size the mesh should take unless given another.
    pub maxh: f64,
    /// Things the caller should hear about.
    pub warnings: Vec<String>,
}

/// The treatment of a conductor layer: "pec", "sibc", "volume"
/// (anisotropic via array) or "volume_iso" (meshed at the skin depth).
fn treatment(layer: &PdkLayer, over: Option<&str>, band: Option<(f64, f64)>, warnings: &mut Vec<String>) -> String {
    let t = match over {
        Some(t) => t,
        None if layer.is_pec() => return "pec".into(),
        None if layer.r#type == "via" => return "volume".into(),
        None => "auto",
    };
    if t != "auto" {
        return t.into();
    }
    let Some((lo, hi)) = band else {
        warnings.push(format!(
            "rfic.build: no band given, layer {:?} uses a surface impedance; pass band=(f_min, f_max) so \
             layers between {SIBC_RATIO_LOW} and {SIBC_RATIO_HIGH} skin depths are meshed as volume conductors",
            layer.name
        ));
        return "sibc".into();
    };
    let r_lo = layer.thickness / skin_depth(lo, layer.sigma);
    let r_hi = layer.thickness / skin_depth(hi, layer.sigma);
    if r_hi < SIBC_RATIO_LOW || r_lo > SIBC_RATIO_HIGH { "sibc".into() } else { "volume_iso".into() }
}

/// The rapidfem material of a stack material: air, a lossy semiconductor
/// or a dielectric with its loss tangent.
fn material_for(mat: &StackMaterial, maxh: Option<f64>) -> super::Mat {
    if mat.is_air() {
        super::Mat { maxh, ..super::Mat::air() }
    } else if mat.sigma > 0.0 && mat.kind == "semiconductor" {
        super::Mat { conductivity: mat.sigma, maxh, ..super::Mat::dielectric(mat.er) }
    } else {
        super::Mat { tand: mat.tand, maxh, ..super::Mat::dielectric(mat.er) }
    }
}

/// Every stack layer of the layout as prisms at its height (sheets at its
/// bottom with `thin_conductors` for metals), named after the layer; with
/// `merge` the prisms of one layer fuse into one conductor. Returns the
/// layers in order of appearance with their objects.
pub fn extrude_layout(
    scene: &mut Scene,
    layout: &Layout,
    stack: &Stack,
    crop: Option<[f64; 4]>,
    merge: bool,
    thin_conductors: bool,
) -> Result<Vec<(String, Vec<ObjId>)>, String> {
    let mut per_layer: Vec<(&PdkLayer, Vec<&[[f64; 2]]>)> = Vec::new();
    for p in &layout.polygons {
        let Some(layer) = stack.by_gds(p.layer, p.datatype) else { continue };
        if let Some([x0, y0, x1, y1]) = crop {
            let b = region::bbox(&vec![vec![p.points.clone()]]);
            if b[2] < x0 || b[0] > x1 || b[3] < y0 || b[1] > y1 {
                continue;
            }
        }
        match per_layer.iter_mut().find(|(l, _)| l.name == layer.name) {
            Some((_, polys)) => polys.push(&p.points),
            None => per_layer.push((layer, vec![&p.points])),
        }
    }
    let mut out = Vec::new();
    for (layer, polys) in per_layer {
        let sheet = thin_conductors && layer.r#type == "metal";
        let mut objs = Vec::new();
        for pts in polys {
            if pts.len() < 3 {
                return Err(format!("a polygon on layer {:?} collapsed to {} unique vertices", layer.name, pts.len()));
            }
            let id = if sheet {
                scene.prism(&[pts.to_vec()], layer.z, None, None, None)?
            } else {
                scene.prism(&[pts.to_vec()], layer.z, Some(layer.thickness), None, None)?
            };
            scene.geo.set_name(id, Some(layer.name.clone()));
            objs.push(id);
        }
        if merge && !sheet && objs.len() >= 2 {
            scene.geo.fuse(objs.clone());
        }
        out.push((layer.name.clone(), objs));
    }
    Ok(out)
}

/// The rectangles on a marker layer as (a0, a1, position, axis): the wide
/// side of each is the plate width.
fn markers(layout: &Layout, layer: i32) -> Vec<(f64, f64, f64, char)> {
    layout
        .on(layer, None)
        .map(|p| {
            let [x0, y0, x1, y1] = region::bbox(&vec![vec![p.points.clone()]]);
            if x1 - x0 >= y1 - y0 { (x0, x1, (y0 + y1) / 2.0, 'x') } else { (y0, y1, (x0 + x1) / 2.0, 'y') }
        })
        .collect()
}

/// The faces of object `id` lying at its minimum (or maximum) along `axis`.
fn extreme_faces(scene: &Scene, id: ObjId, axis: usize, max: bool) -> Result<Vec<FaceSel>, String> {
    let faces = scene.geo.faces_of(id)?;
    let items = faces.iter().map(|f| scene.geo.face_extent(f)).collect::<Result<Vec<Extent>, _>>()?;
    let tol = SELECT_TOL * Extent::size(&scene.geo.bbox()?);
    Ok(Extent::extreme(&items, axis, max, tol).into_iter().map(|i| faces[i]).collect())
}

/// Builds the model of the layout at `gds` in `stack` into `scene`.
pub fn build(scene: &mut Scene, gds_path: &Path, stack: &Stack, o: &Options) -> Result<Built, String> {
    if stack.dielectrics.is_empty() {
        return Err("stack has no background dielectrics; build() needs the full vertical cross-section \
                    (Stack.from_xml or a preset with dielectrics)"
            .into());
    }
    if !["planar", "conformal", "none"].contains(&o.passivation.as_str()) {
        return Err(format!("passivation must be planar|conformal|none, got {:?}", o.passivation));
    }
    if !["abc", "pml"].contains(&o.boundary.as_str()) {
        return Err(format!("boundary must be abc|pml, got {:?}", o.boundary));
    }
    if o.boundary == "pml" && o.pec_floor {
        return Err("pec_floor is an ABC-mode option".into());
    }
    let mut built = Built::default();

    // The passivation sheet: the topmost non-air dielectric slab.
    let pass_slab = if o.passivation != "planar" {
        let cands: Vec<usize> = (0..stack.dielectrics.len())
            .filter(|&i| {
                let m = stack.slab_material(&stack.dielectrics[i]);
                !m.is_air() && m.kind == "dielectric"
            })
            .collect();
        Some(*cands.last().ok_or("stack has no passivation slab to modify")?)
    } else {
        None
    };
    // Conformal: the metal the passivation drapes over.
    let mut pass_t_top = o.pass_t_top;
    let conf_layer = if o.passivation == "conformal" {
        let l = match &o.conformal_over {
            Some(name) => stack.by_name(name)?.clone(),
            None => stack
                .layers
                .iter()
                .filter(|l| l.r#type == "metal")
                .reduce(|a, b| if b.z_top() > a.z_top() { b } else { a })
                .ok_or("stack has no metal for the conformal passivation")?
                .clone(),
        };
        if pass_t_top.is_none() {
            pass_t_top = Some(stack.dielectrics[pass_slab.unwrap()].thickness);
        }
        Some(l)
    } else {
        None
    };

    // ── conductors from the GDS ─────────────────────────────────────────
    let layout = gds::read(gds_path, o.top_cell.as_deref())?;
    let extruded = extrude_layout(scene, &layout, stack, None, true, false)?;
    // in stack order, bottom-up
    let mut conductors: Vec<(String, Vec<ObjId>)> = Vec::new();
    for l in &stack.layers {
        if let Some((_, ids)) = extruded.iter().find(|(n, _)| *n == l.name) {
            conductors.push((l.name.clone(), ids.clone()));
        }
    }
    if conductors.is_empty() {
        return Err(format!("no stack layers found in {:?}", gds_path.display()));
    }
    let mesh = match &o.mesh {
        Mesh::Spec(s) => s.clone(),
        Mesh::Preset(p) => {
            let drawn: Vec<String> = conductors.iter().map(|(n, _)| n.clone()).collect();
            MeshSpec::derive(stack, &drawn, p)?
        }
    };

    // Layout bounding box from the conductors.
    let mut bb = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for id in conductors.iter().flat_map(|(_, ids)| ids) {
        let b = scene.geo.object_extent(*id)?.bbox;
        bb = [bb[0].min(b[0]), bb[1].min(b[1]), bb[2].max(b[3]), bb[3].max(b[4])];
    }
    let (x0, y0) = (bb[0] - o.margin, bb[1] - o.margin);
    let (x1, y1) = (bb[2] + o.margin, bb[3] + o.margin);
    let (wx, wy) = (x1 - x0, y1 - y0);
    built.footprint = [x0, y0, x1, y1];

    // ── background dielectric slabs ─────────────────────────────────────
    let z_bot = stack.dielectrics[0].z;
    let mut z_top = stack.dielectrics[stack.dielectrics.len() - 1].z_top();
    let last = stack.dielectrics.len() - 1;
    for (i, d) in stack.dielectrics.iter().enumerate() {
        let mat = stack.slab_material(d);
        let mut thickness = d.thickness;
        let mut z_lo = d.z;
        if Some(i) == pass_slab {
            // the sheet is replaced (conformal) or dropped (none)
            continue;
        }
        if let Some(cl) = &conf_layer
            && d.z <= cl.z
            && cl.z < d.z_top()
        {
            // the oxide stops at the exposed metal's bottom
            thickness = cl.z - d.z;
        }
        if i == last && mat.is_air() {
            // the topmost air slab, capped at air_top (the XML often carries
            // a generous 200 um the boundary does not need)
            thickness = o.air_top.unwrap_or(d.thickness);
            if o.passivation == "conformal" {
                // built as polygon prisms below, only its cap stays a box
                z_top = d.z + thickness;
                continue;
            }
            if o.passivation == "none"
                && let Some(p) = pass_slab
            {
                // extend down over the dropped sheet
                z_lo = stack.dielectrics[p].z;
                thickness += stack.dielectrics[p].thickness;
            }
            z_top = z_lo + thickness;
        }
        let mut boxes = Vec::new();
        match mesh.graded.get(&d.name) {
            Some(zones) if !zones.is_empty() => {
                // top-down zones, the last padded
                let mut z_hi = z_lo + thickness;
                let mut remaining = thickness;
                for (k, &(t_zone, h_zone)) in zones.iter().enumerate() {
                    let t = if k < zones.len() - 1 { t_zone.min(remaining) } else { remaining };
                    if t <= 0.0 {
                        break;
                    }
                    let id = scene.geo.add_solid(Cuboid::new([wx, wy, t]).at([x0, y0, z_hi - t]), None, false);
                    scene.fill(&[id], material_for(&mat, Some(mesh.h(h_zone))));
                    boxes.push(id);
                    z_hi -= t;
                    remaining -= t;
                }
            }
            _ => {
                let id = scene.geo.add_solid(Cuboid::new([wx, wy, thickness]).at([x0, y0, z_lo]), None, false);
                scene.fill(&[id], material_for(&mat, Some(mesh.slab_h(&d.name))));
                boxes.push(id);
            }
        }
        built.slabs.push((d.name.clone(), boxes));
    }

    // ── conformal passivation shell and the air prisms over it ──────────
    if let (Some(cl), Some(p)) = (&conf_layer, pass_slab) {
        let pass = &stack.dielectrics[p];
        let pass_mat = stack.slab_material(pass);
        let air_slab = &stack.dielectrics[last];
        let h_pass = mesh.slab_h(&pass.name);
        let h_air = mesh.h(mesh.global_h);
        let t_top = pass_t_top.unwrap();
        let (zm_lo, zm_hi) = (cl.z, cl.z_top());
        let metal: Region = region::union(
            &layout.on(cl.gds, Some(cl.datatype)).map(|q| vec![q.points.clone()]).collect(),
        );
        if metal.is_empty() {
            return Err(format!("no polygons on GDS layer {}/{} for the conformal passivation", cl.gds, cl.datatype));
        }
        let expanded = region::offset(&metal, o.pass_t_side);
        let foot: Region = vec![vec![vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]];
        let field = region::difference(&foot, &expanded);
        let ring = region::difference(&expanded, &metal);
        let mut shell = Vec::new();
        // the field sheet, everywhere but the expanded metal footprint
        shell.extend(scene.prisms(&field, zm_lo, t_top, Some(material_for(&pass_mat, None)), Some(h_pass))?);
        // the sidewall ring, expanded minus metal
        shell.extend(scene.prisms(&ring, zm_lo, (zm_hi - zm_lo) + t_top, Some(material_for(&pass_mat, None)), Some(h_pass))?);
        // the cap on the metal top
        shell.extend(scene.prisms(&metal, zm_hi, t_top, Some(material_for(&pass_mat, None)), Some(h_pass))?);
        // Air: only the step between the field sheet and the shell top
        // follows the metal outline, everything above is one plain box (the
        // outline stamped through the whole air would force the trace width
        // over its full height).
        let z_shell_top = zm_hi + t_top;
        let mut air_low = scene.prisms(&field, zm_lo + t_top, z_shell_top - (zm_lo + t_top), Some(super::Mat::air()), Some(h_air))?;
        let cap = scene.geo.add_solid(Cuboid::new([wx, wy, z_top - z_shell_top]).at([x0, y0, z_shell_top]), Some(h_air), false);
        scene.fill(&[cap], super::Mat::air());
        air_low.push(cap);
        built.slabs.push((pass.name.clone(), shell));
        built.slabs.push((air_slab.name.clone(), air_low));
    }

    // ── air shell around the dielectric stack (6 disjoint boxes) ────────
    let (a, stack_h) = (o.air, z_top - z_bot);
    let shell_boxes = [
        ([wx + 2.0 * a, wy + 2.0 * a, a], [x0 - a, y0 - a, z_top]),
        ([wx + 2.0 * a, wy + 2.0 * a, a], [x0 - a, y0 - a, z_bot - a]),
        ([a, wy + 2.0 * a, stack_h], [x0 - a, y0 - a, z_bot]),
        ([a, wy + 2.0 * a, stack_h], [x1, y0 - a, z_bot]),
        ([wx, a, stack_h], [x0, y0 - a, z_bot]),
        ([wx, a, stack_h], [x0, y1, z_bot]),
    ];
    for (size, at) in shell_boxes {
        built.air_shell.push(scene.geo.add_solid(Cuboid::new(size).at(at), None, false));
    }
    scene.fill(&built.air_shell.clone(), super::Mat::air());

    // ── conductor treatment ──────────────────────────────────────────────
    // SIBC and PEC conductors become holes, their walls carry the condition;
    // volume conductors carry their bulk conductivity.
    let mut treat = BTreeMap::new();
    for (name, _) in &conductors {
        let layer = stack.by_name(name)?;
        treat.insert(name.clone(), treatment(layer, o.conductor_model.get(name).map(String::as_str), o.band, &mut built.warnings));
    }
    for (name, ids) in &conductors {
        let layer = stack.by_name(name)?;
        let t = treat[name].as_str();
        let mut h = mesh.h(mesh.conductor);
        if t == "volume_iso"
            && let Some((_, f_hi)) = o.band
        {
            h = h.min(layer.thickness / 3.0).min(skin_depth(f_hi, layer.sigma));
        }
        for &id in ids {
            let m = match t {
                "volume" => {
                    let s = layer.sigma;
                    super::Mat { cond_diag: Some([VIA_LATERAL_FACTOR * s, VIA_LATERAL_FACTOR * s, s]), ..super::Mat::dielectric(1.0) }
                }
                "volume_iso" => super::Mat { conductivity: layer.sigma, ..super::Mat::dielectric(1.0) },
                _ => {
                    // a hole later: the background around the layer
                    let bg = stack
                        .dielectric_at(layer.z + layer.thickness / 2.0)
                        .map_or_else(|| StackMaterial::new("air", "dielectric"), |d| stack.slab_material(d));
                    super::Mat { tand: bg.tand, ..super::Mat::dielectric(bg.er) }
                }
            };
            scene.fill(&[id], m);
            scene.geo.set_object_maxh(id, Some(h));
        }
    }

    // ── port plates ───────────────────────────────────────────────────────
    let z_of = |b: &ZBound, lower: bool| -> Result<f64, String> {
        Ok(match b {
            ZBound::Height(z) => *z,
            ZBound::Layer(name) => {
                let l = stack.by_name(name)?;
                if lower { l.z_top() } else { l.z }
            }
        })
    };
    let mut plates = Vec::new();
    for p in &o.ports {
        let placements = match p.marker {
            Some(m) => {
                let found = markers(&layout, m);
                if found.is_empty() {
                    return Err(format!("no rectangles on marker layer {m}"));
                }
                found
            }
            None => match (p.span, p.at) {
                (Some((a0, a1)), Some(at)) => vec![(a0, a1, at, p.axis)],
                _ => return Err("ViaPort needs either marker or span+at".into()),
            },
        };
        let (z_lo, z_hi) = (z_of(&p.z.0, true)?, z_of(&p.z.1, false)?);
        for (a0, a1, pos, axis) in placements {
            let (corner, width) = if axis == 'x' { ([a0, pos, z_lo], [a1 - a0, 0.0, 0.0]) } else { ([pos, a0, z_lo], [0.0, a1 - a0, 0.0]) };
            let id = scene.geo.add_sheet(Sheet::plate(corner, width, [0.0, 0.0, z_hi - z_lo]), Some(mesh.h(mesh.port)));
            plates.push((id, p.z0));
        }
    }

    // ── one conformal assembly, the later objects on top ─────────────────
    let mut front: Vec<ObjId> = built.slabs.iter().flat_map(|(_, ids)| ids.iter().copied()).skip(1).collect();
    front.extend(&built.air_shell);
    front.extend(conductors.iter().flat_map(|(_, ids)| ids.iter().copied()));
    front.extend(plates.iter().map(|(id, _)| *id));
    scene.geo.bring_to_front(&front);

    // ── physics: conductor walls, ports, outer boundary ──────────────────
    let faces = |sels: Vec<FaceSel>| sels.into_iter().map(Target::Face).collect::<Vec<_>>();
    for (name, _) in &conductors {
        let t = treat[name].as_str();
        if t != "pec" && t != "sibc" {
            continue;
        }
        let Some((walls, t_eff)) = scene.geo.hollow(name)? else { continue };
        let condition = if t == "pec" {
            Condition::Pec
        } else {
            let sigma = stack.by_name(name)?.sigma;
            // the wall thickness 2V/S, to three digits as one surface per layer
            let t_eff: f64 = format!("{t_eff:.2e}").parse().unwrap();
            Condition::Face(FaceSpec::SurfaceImpedance {
                tag: 0,
                conductivity: sigma,
                mur: 1.0,
                er: 1.0,
                thickness: Some(t_eff),
                two_sided: true,
                sheet: false,
                zs: None,
            })
        };
        scene.setup.add(Physics::new(condition, faces(walls)));
    }
    for &(id, z0) in &plates {
        let spec = FaceSpec::Lumped { tag: 0, z0, l: 0.0, c: None, direction: [0.0, 0.0, 1.0], width: 0.0, height: 0.0, power: 1.0 };
        scene.setup.add(Physics::new(Condition::Face(spec), vec![Target::Face(FaceSel::sheet(id))]));
    }
    let [top, bot, cxmin, cxmax, cymin, cymax] = built.air_shell[..] else { unreachable!() };
    if o.boundary == "pml" {
        // each air-shell box one single-direction slab, disjoint by construction
        let pml = [
            (top, [0.0, 0.0, 1.0], z_top),
            (bot, [0.0, 0.0, -1.0], z_bot),
            (cxmin, [-1.0, 0.0, 0.0], x0),
            (cxmax, [1.0, 0.0, 0.0], x1),
            (cymin, [0.0, -1.0, 0.0], y0),
            (cymax, [0.0, 1.0, 0.0], y1),
        ];
        for (id, direction, inner_face) in pml {
            let spec = PmlSpec { volume_tag: 0, direction, inner_face, thickness: a, er_base: 1.0, ur_base: 1.0, exponent: 1.5, delta_max: 8.0 };
            scene.setup.add(Physics::new(Condition::Pml(spec), vec![Target::Object(id)]));
        }
    } else {
        let (x, y, z) = (0, 1, 2);
        let mut outer = Vec::new();
        for (id, axis, max) in [
            (top, x, false), (top, x, true), (top, y, false), (top, y, true), (top, z, true),
            (bot, x, false), (bot, x, true), (bot, y, false), (bot, y, true),
            (cxmin, x, false), (cxmin, y, false), (cxmin, y, true),
            (cxmax, x, true), (cxmax, y, false), (cxmax, y, true),
            (cymin, y, false),
            (cymax, y, true),
        ] {
            outer.extend(extreme_faces(scene, id, axis, max)?);
        }
        let floor = extreme_faces(scene, bot, z, false)?;
        if o.pec_floor {
            scene.setup.add(Physics::new(Condition::Pec, faces(floor)));
        } else {
            outer.extend(floor);
        }
        scene.setup.add(Physics::new(Condition::Face(FaceSpec::Abc { tag: 0 }), faces(outer)));
    }

    built.conductors = conductors;
    built.ports = plates.iter().map(|(id, _)| *id).collect();
    built.maxh = mesh.h(mesh.global_h);
    Ok(built)
}
