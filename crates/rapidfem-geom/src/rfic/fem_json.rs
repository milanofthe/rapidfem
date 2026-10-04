// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The rapidpassives `exportForFEM()` JSON as a solve-ready scene: a
//! substrate, oxide and air box around the layout, every conductor
//! extruded to its layer and cut out as a PEC-walled hole (as in
//! [`super::build`]), a vertical lumped-port plate per JSON port and an
//! absorbing boundary on the outside of the air.
//!
//! Each port plate is inset from the layout edge toward the layout centre,
//! so its top edge lands on the bottom face of the port's metal, and drops
//! to the topmost lower metal under it that is not the same net (a via
//! inside both polygons); a port with no such metal drops to the lowest
//! metal onto a ground patch, shared by ports closer than four tab widths.

use serde_json::Value;

use rapidfem_core::model::FaceSpec;
use rapidmesh::shapes::{Cuboid, Sheet};

use super::region::P2;
use super::stack::Stack;
use super::{Mat, Scene};
use crate::geometry::{Extent, FaceSel, ObjId, SELECT_TOL};
use crate::setup::{Condition, Physics, Target};

/// The schema versions understood.
pub const SCHEMA_VERSIONS: [i64; 1] = [1];

/// Consecutive vertices closer than this (microns) are merge slivers of
/// rapidpassives' mergeLayers, far below any layout feature.
const VERTEX_TOL_UM: f64 = 0.01;

const UM: f64 = 1e-6;

/// The options of [`from_fem_json`] (see the Python `rfic.from_fem_json`).
#[derive(Clone, Debug)]
pub struct Options {
    /// "merged" (one prism per via array) or "cells" (every via cell).
    pub via_mode: String,
    pub footprint_margin: f64,
    pub air_height_um: f64,
    pub conductor_maxh_um: f64,
    pub port_maxh_um: f64,
    pub port_tab_um: f64,
    pub port_inset_um: Option<f64>,
    pub port_z0: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            via_mode: "merged".into(),
            footprint_margin: 0.3,
            air_height_um: 60.0,
            conductor_maxh_um: 3.0,
            port_maxh_um: 3.0,
            port_tab_um: 8.0,
            port_inset_um: None,
            port_z0: 50.0,
        }
    }
}

/// What [`from_fem_json`] made.
#[derive(Clone, Debug, Default)]
pub struct FemLayout {
    /// Stack-layer id and its conductor holes.
    pub conductors: Vec<(String, Vec<ObjId>)>,
    /// Port name and its plate.
    pub ports: Vec<(String, ObjId)>,
    pub ground_patches: Vec<ObjId>,
    pub substrate: ObjId,
    pub oxide: ObjId,
    pub air: ObjId,
    /// The global mesh size: a fifth of the smaller footprint side.
    pub maxh: f64,
}

/// Drops consecutive vertices closer than [`VERTEX_TOL_UM`] and a closing
/// copy of the first vertex.
fn clean(poly: &[P2]) -> Vec<P2> {
    let tol2 = VERTEX_TOL_UM * VERTEX_TOL_UM;
    let d2 = |a: P2, b: P2| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
    let mut out: Vec<P2> = Vec::new();
    for &p in poly {
        if out.last().is_none_or(|&q| d2(p, q) >= tol2) {
            out.push(p);
        }
    }
    if out.len() >= 2 && d2(out[0], out[out.len() - 1]) < tol2 {
        out.pop();
    }
    out
}

fn points(v: &Value) -> Result<Vec<P2>, String> {
    v.as_array()
        .ok_or("FEM JSON: a polygon is not a list")?
        .iter()
        .map(|p| match p.as_array().map(|a| a.as_slice()) {
            Some([x, y, ..]) => Ok([x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN)]),
            _ => Err("FEM JSON: a vertex is not [x, y]".to_string()),
        })
        .collect()
}

fn num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).ok_or_else(|| format!("FEM JSON: {key} missing or not a number"))
}

fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key).and_then(Value::as_str).ok_or_else(|| format!("FEM JSON: {key} missing or not a string"))
}

/// The polygon bounding box (x0, x1, y0, y1) in metres.
fn bbox_m(poly: &[P2]) -> [f64; 4] {
    let mut b = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
    for p in poly {
        let (x, y) = (p[0] * UM, p[1] * UM);
        b = [b[0].min(x), b[1].max(x), b[2].min(y), b[3].max(y)];
    }
    b
}

/// Builds the layout of the FEM JSON `doc` into `scene`, the background
/// from the JSON or, with `stack`, from its single-slab summary.
pub fn from_fem_json(scene: &mut Scene, doc: &Value, stack: Option<&Stack>, o: &Options) -> Result<FemLayout, String> {
    let sv = doc.get("schema_version").and_then(Value::as_i64).unwrap_or(1);
    if !SCHEMA_VERSIONS.contains(&sv) {
        return Err(format!("unsupported FEM JSON schema_version {sv}; supported: {SCHEMA_VERSIONS:?}"));
    }
    let stack_doc = doc.get("stack").ok_or("FEM JSON: no stack")?;
    let layers = stack_doc.get("layers").and_then(Value::as_array).ok_or("FEM JSON: no stack layers")?;
    let conductors = doc.get("conductors").and_then(Value::as_array).ok_or("FEM JSON: no conductors")?;
    let ports = doc.get("ports").and_then(Value::as_array).ok_or("FEM JSON: no ports")?;
    let layer = |id: &str| -> Result<&Value, String> {
        layers.iter().find(|l| l.get("id").and_then(Value::as_str) == Some(id)).ok_or_else(|| {
            let mut ids: Vec<&str> = layers.iter().filter_map(|l| l.get("id").and_then(Value::as_str)).collect();
            ids.sort();
            format!(
                "unknown stack layer {id:?} (an export from an older rapidpassives that wrote \
                 generator-internal names such as 'm3' needs a re-export); known stack layers: {ids:?}"
            )
        })
    };

    // Substrate and oxide constants: the JSON, unless a stack is passed.
    let null = Value::Null;
    let sub = stack_doc.get("substrate").unwrap_or(&null);
    let ox = stack_doc.get("oxide").unwrap_or(&null);
    let pos = |v: &Value, key: &str, default: f64| v.get(key).and_then(Value::as_f64).filter(|x| *x != 0.0).unwrap_or(default);
    let sub_thickness_um = pos(sub, "thickness_um", 300.0);
    let mut sub_er = sub.get("er").and_then(Value::as_f64).unwrap_or(11.7);
    // σ [S/m] = 1 / (ρ [Ω·cm] · 0.01)
    let mut sub_sigma = 100.0 / pos(sub, "rho_ohm_cm", 10.0);
    let mut ox_er = ox.get("er").and_then(Value::as_f64).unwrap_or(4.2);
    let mut ox_tand = ox.get("tand").and_then(Value::as_f64).unwrap_or(0.0);
    if let Some(s) = stack {
        let slab = s.slab_summary();
        let get = |part: &str, key: &str| slab.get(part).and_then(|p| p.get(key)).and_then(Value::as_f64);
        sub_er = get("substrate", "er").ok_or("the stack has no semiconductor substrate slab")?;
        sub_sigma = get("substrate", "sigma").unwrap_or(0.0);
        ox_er = get("oxide", "er").ok_or("the stack has no oxide slab")?;
        ox_tand = get("oxide", "tand").unwrap_or(0.0);
    }

    // Footprint: the conductor bbox plus the margin on each side.
    let polys: Vec<Vec<P2>> = conductors.iter().map(|c| points(c.get("polygon").unwrap_or(&null))).collect::<Result<_, _>>()?;
    let all: Vec<P2> = polys.iter().flatten().copied().collect();
    if all.is_empty() {
        return Err("no conductor polygons in FEM JSON".into());
    }
    let (mut x_lo, mut x_hi, mut y_lo, mut y_hi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in &all {
        (x_lo, x_hi, y_lo, y_hi) = (x_lo.min(p[0]), x_hi.max(p[0]), y_lo.min(p[1]), y_hi.max(p[1]));
    }
    let (span_x, span_y) = ((x_hi - x_lo).max(1.0), (y_hi - y_lo).max(1.0));
    let foot_w = (span_x + 2.0 * o.footprint_margin * span_x) * UM;
    let foot_h = (span_y + 2.0 * o.footprint_margin * span_y) * UM;
    let (cx, cy) = ((x_lo + x_hi) / 2.0 * UM, (y_lo + y_hi) / 2.0 * UM);

    // The stack's z range: bottom of the lowest layer, top of the highest.
    let mut by_z: Vec<&Value> = layers.iter().collect();
    by_z.sort_by(|a, b| num(a, "z_um").unwrap_or(0.0).partial_cmp(&num(b, "z_um").unwrap_or(0.0)).unwrap());
    let z_bottom_um = num(by_z[0], "z_um")?;
    let last = by_z[by_z.len() - 1];
    let z_top_um = num(last, "z_um")? + num(last, "thickness_um")?;
    let z_top_m = z_top_um * UM;

    // a mesh study on the bundled layouts: a fifth of the footprint and
    // 3 um on the conductors stay within 0.01 of half these sizes in |S|
    let maxh = foot_w.min(foot_h) / 5.0;
    scene.geo.set_maxh(Some(maxh));
    let corner = [cx - foot_w / 2.0, cy - foot_h / 2.0];
    let substrate = scene.geo.add_solid(
        Cuboid::new([foot_w, foot_h, sub_thickness_um * UM]).at([corner[0], corner[1], z_bottom_um * UM - sub_thickness_um * UM]),
        None,
        false,
    );
    scene.fill(&[substrate], Mat { conductivity: sub_sigma, ..Mat::dielectric(sub_er) });
    let oxide = scene.geo.add_solid(Cuboid::new([foot_w, foot_h, (z_top_um - z_bottom_um) * UM]).at([corner[0], corner[1], z_bottom_um * UM]), None, false);
    let sio2 = Mat { tand: ox_tand, ..Mat::dielectric(ox_er) };
    scene.fill(&[oxide], sio2.clone());
    let air = scene.geo.add_solid(Cuboid::new([foot_w, foot_h, o.air_height_um * UM]).at([corner[0], corner[1], z_top_m]), None, false);
    scene.fill(&[air], Mat::air());

    // Every conductor polygon extruded to its layer.
    let cond_maxh = o.conductor_maxh_um * UM;
    let mut out = FemLayout { substrate, oxide, air, maxh, ..FemLayout::default() };
    let mut extruded: Vec<ObjId> = Vec::new();
    for (c, poly) in conductors.iter().zip(&polys) {
        let id = text(c, "layer")?;
        let l = layer(id)?;
        let (z_lo, thick) = (num(l, "z_um")? * UM, num(l, "thickness_um")? * UM);
        if thick <= 0.0 {
            continue;
        }
        let cells = c.get("polygon_cells").and_then(Value::as_array).filter(|cs| o.via_mode == "cells" && !cs.is_empty());
        // only the primary polygon carries holes (annular rings); via cells
        // are solid pieces
        let (pieces, holes): (Vec<Vec<P2>>, Vec<Vec<P2>>) = match cells {
            Some(cs) => (cs.iter().map(points).collect::<Result<_, _>>()?, Vec::new()),
            None => {
                let holes = match c.get("holes").and_then(Value::as_array) {
                    Some(hs) => hs.iter().map(points).collect::<Result<Vec<_>, _>>()?,
                    None => Vec::new(),
                };
                (vec![poly.clone()], holes)
            }
        };
        let holes: Vec<Vec<P2>> = holes.iter().map(|h| clean(h)).filter(|h| h.len() >= 3).collect();
        for piece in pieces {
            let piece = clean(&piece);
            if piece.len() < 3 {
                continue;
            }
            let mut contours = vec![piece.iter().map(|p| [p[0] * UM, p[1] * UM]).collect::<Vec<P2>>()];
            contours.extend(holes.iter().map(|h| h.iter().map(|p| [p[0] * UM, p[1] * UM]).collect()));
            // no material: a hole below, its walls PEC
            let v = scene.prism(&contours, z_lo, Some(thick), None, Some(cond_maxh))?;
            scene.geo.set_name(v, Some(id.to_string()));
            match out.conductors.iter_mut().find(|(n, _)| n == id) {
                Some((_, vs)) => vs.push(v),
                None => out.conductors.push((id.to_string(), vec![v])),
            }
            extruded.push(v);
        }
    }

    // Ports: the per-port ground below each, then plates and patches.
    let tab = o.port_tab_um * UM;
    let inset = o.port_inset_um.unwrap_or(o.port_tab_um / 2.0) * UM;
    let port_maxh = o.port_maxh_um * UM;
    let mut metals: Vec<&Value> = layers.iter().filter(|l| l.get("type").and_then(Value::as_str) == Some("metal")).collect();
    metals.sort_by(|a, b| num(a, "z_um").unwrap_or(0.0).partial_cmp(&num(b, "z_um").unwrap_or(0.0)).unwrap());
    let lowest = metals.first().ok_or("FEM JSON: the stack has no metal layer")?;
    let polygon_at = |layer_id: &str, px: f64, py: f64| -> Option<[f64; 4]> {
        conductors.iter().zip(&polys).find_map(|(c, poly)| {
            if c.get("layer").and_then(Value::as_str) != Some(layer_id) {
                return None;
            }
            let b = bbox_m(poly);
            (b[0] <= px && px <= b[1] && b[2] <= py && py <= b[3]).then_some(b)
        })
    };
    // a via polygon between the two metals inside both boxes joins them
    let same_net = |top_id: &str, top: Option<[f64; 4]>, bot_id: &str, bot: Option<[f64; 4]>| -> Result<bool, String> {
        let (Some(t), Some(b)) = (top, bot) else { return Ok(false) };
        let (z_lo, z_hi) = (num(layer(bot_id)?, "z_um")?, num(layer(top_id)?, "z_um")?);
        for vl in layers {
            let z = num(vl, "z_um")?;
            if vl.get("type").and_then(Value::as_str) != Some("via") || !(z_lo < z && z < z_hi) {
                continue;
            }
            for (c, poly) in conductors.iter().zip(&polys) {
                if c.get("layer") != vl.get("id") {
                    continue;
                }
                let v = bbox_m(poly);
                let (vx, vy) = (0.5 * (v[0] + v[1]), 0.5 * (v[2] + v[3]));
                let inside = |r: [f64; 4]| r[0] <= vx && vx <= r[1] && r[2] <= vy && vy <= r[3];
                if inside(t) && inside(b) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    };
    // (name, x, y, top z, ground z, needs a patch)
    let mut resolved: Vec<(String, f64, f64, f64, f64, bool)> = Vec::new();
    for p in ports {
        let lay = text(p, "layer")?;
        let port_layer = layer(lay)?;
        let z_top = num(port_layer, "z_um")? * UM;
        let (px0, py0) = (num(p, "x_um")? * UM, num(p, "y_um")? * UM);
        let (dx, dy) = (px0 - cx, py0 - cy);
        let n = dx.hypot(dy);
        let (px, py) = if n > 1e-15 { (px0 - inset * dx / n, py0 - inset * dy / n) } else { (px0, py0) };
        let mut gnd_z = (num(lowest, "z_um")? + num(lowest, "thickness_um")?) * UM;
        let mut needs_patch = true;
        let port_z = num(port_layer, "z_um")?;
        let port_box = polygon_at(lay, px, py);
        for cand in metals.iter().rev() {
            if num(cand, "z_um")? >= port_z {
                continue;
            }
            let cand_id = text(cand, "id")?;
            let cand_box = polygon_at(cand_id, px, py);
            if cand_box.is_none() || same_net(lay, port_box, cand_id, cand_box)? {
                continue;
            }
            gnd_z = (num(cand, "z_um")? + num(cand, "thickness_um")?) * UM;
            needs_patch = false;
            break;
        }
        resolved.push((text(p, "name")?.to_string(), px, py, z_top, gnd_z, needs_patch));
    }

    // Plates tangent to the layout edge, from each port's ground up.
    for (name, px, py, z_top, gnd_z, _) in &resolved {
        let (rx, ry) = (px - cx, py - cy);
        let r = rx.hypot(ry);
        let (tx, ty) = if r > 1e-15 { (-ry / r, rx / r) } else { (0.0, 1.0) };
        let p0 = [px - tx * tab / 2.0, py - ty * tab / 2.0, *gnd_z];
        let plate = scene.geo.add_sheet(Sheet::plate(p0, [tx * tab, ty * tab, 0.0], [0.0, 0.0, z_top - gnd_z]), Some(port_maxh));
        scene.geo.set_name(plate, Some(name.clone()));
        out.ports.push((name.clone(), plate));
    }

    // Ground patches for the ports without a metal below, shared by ports
    // closer than four tab widths (union-find over the needy ones).
    let needy: Vec<usize> = (0..resolved.len()).filter(|&i| resolved[i].5).collect();
    let mut parent: Vec<usize> = (0..resolved.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for (a, &i) in needy.iter().enumerate() {
        for &j in &needy[a + 1..] {
            if (resolved[i].1 - resolved[j].1).hypot(resolved[i].2 - resolved[j].2) < 4.0 * tab {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }
    let mut clusters: Vec<(usize, Vec<usize>)> = Vec::new();
    for &i in &needy {
        let root = find(&mut parent, i);
        match clusters.iter_mut().find(|(r, _)| *r == root) {
            Some((_, members)) => members.push(i),
            None => clusters.push((root, vec![i])),
        }
    }
    let pad = tab.max(4e-6);
    for (_, members) in clusters {
        let xs = members.iter().map(|&i| resolved[i].1);
        let ys = members.iter().map(|&i| resolved[i].2);
        let (gx0, gx1) = (xs.clone().fold(f64::INFINITY, f64::min) - pad, xs.fold(f64::NEG_INFINITY, f64::max) + pad);
        let (gy0, gy1) = (ys.clone().fold(f64::INFINITY, f64::min) - pad, ys.fold(f64::NEG_INFINITY, f64::max) + pad);
        let gnd_z = resolved[members[0]].4;
        let patch = scene.geo.add_sheet(Sheet::plate([gx0, gy0, gnd_z], [gx1 - gx0, 0.0, 0.0], [0.0, gy1 - gy0, 0.0]), Some(port_maxh));
        let names: Vec<&str> = members.iter().map(|&i| resolved[i].0.as_str()).collect();
        scene.geo.set_name(patch, Some(format!("gnd_{}", names.join("_"))));
        out.ground_patches.push(patch);
    }

    // One conformal assembly, then the physics: conductor holes and ground
    // patches PEC, the ports lumped, the outside of the air absorbing.
    let mut front = vec![substrate];
    front.extend(&extruded);
    front.extend(&out.ground_patches);
    front.extend(out.ports.iter().map(|(_, id)| *id));
    front.push(air);
    scene.geo.bring_to_front(&front);
    let mut pec: Vec<Target> = Vec::new();
    for (id, _) in out.conductors.clone() {
        if let Some((walls, _)) = scene.geo.hollow(&id)? {
            pec.extend(walls.into_iter().map(Target::Face));
        }
    }
    pec.extend(out.ground_patches.iter().map(|&p| Target::Face(FaceSel::sheet(p))));
    scene.setup.add(Physics::new(Condition::Pec, pec));
    for (_, plate) in &out.ports {
        let spec = FaceSpec::Lumped { tag: 0, z0: o.port_z0, l: 0.0, c: None, direction: [0.0, 0.0, 1.0], width: 0.0, height: 0.0, power: 1.0 };
        scene.setup.add(Physics::new(Condition::Face(spec), vec![Target::Face(FaceSel::sheet(*plate))]));
    }
    let faces = scene.geo.faces_of(air)?;
    let items = faces.iter().map(|f| scene.geo.face_extent(f)).collect::<Result<Vec<Extent>, _>>()?;
    let model = scene.geo.bbox()?;
    let outer: Vec<Target> = Extent::on_box(&items, &model, SELECT_TOL * Extent::size(&model)).into_iter().map(|i| Target::Face(faces[i])).collect();
    scene.setup.add(Physics::new(Condition::Face(FaceSpec::Abc { tag: 0 }), outer));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_drops_slivers_and_the_closing_vertex() {
        let c = clean(&[[0.0, 0.0], [10.0, 0.0], [10.001, 0.0], [10.0, 10.0], [0.001, 0.0]]);
        assert_eq!(c, vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]);
    }
}
