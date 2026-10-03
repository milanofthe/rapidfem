// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! GDSII reading by gds21 (Dan Fritchman's Layout21): the polygons of one
//! top cell with every structure and array reference resolved, in metres.
//!
//! Boundaries and boxes count (a box's boxtype as its datatype); paths and
//! texts do not, as in the gdstk import this replaces.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use gds21::{GdsElement, GdsLibrary, GdsPoint, GdsStrans, GdsStruct};

use super::region::P2;

/// One polygon of the flattened cell: its GDS layer, datatype and closed
/// outline (no repeated closing vertex), in metres.
#[derive(Clone, Debug)]
pub struct Polygon {
    pub layer: i32,
    pub datatype: i32,
    pub points: Vec<P2>,
}

/// The polygons of a layout's top cell.
#[derive(Clone, Debug)]
pub struct Layout {
    pub cell: String,
    pub polygons: Vec<Polygon>,
}

/// An affine map `p -> [a b; c d] p + t`.
#[derive(Clone, Copy)]
struct Affine {
    m: [[f64; 2]; 2],
    t: P2,
}

impl Affine {
    #[cfg(test)]
    const ID: Affine = Affine { m: [[1.0, 0.0], [0.0, 1.0]], t: [0.0, 0.0] };

    fn apply(&self, p: P2) -> P2 {
        [self.m[0][0] * p[0] + self.m[0][1] * p[1] + self.t[0], self.m[1][0] * p[0] + self.m[1][1] * p[1] + self.t[1]]
    }

    /// `self` after `inner`.
    fn then(&self, inner: &Affine) -> Affine {
        let (a, b) = (self.m, inner.m);
        let m = [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ];
        Affine { m, t: self.apply(inner.t) }
    }

    /// A reference placed at `origin`: reflection about x first, then
    /// magnification and rotation (GDS `STRANS`).
    fn placement(origin: P2, strans: Option<&GdsStrans>) -> Affine {
        let (reflect, mag, angle) = strans.map_or((false, 1.0, 0.0), |s| (s.reflected, s.mag.unwrap_or(1.0), s.angle.unwrap_or(0.0)));
        let (sin, cos) = angle.to_radians().sin_cos();
        let r = if reflect { -1.0 } else { 1.0 };
        Affine { m: [[mag * cos, -mag * sin * r], [mag * sin, mag * cos * r]], t: origin }
    }
}

fn point(p: &GdsPoint) -> P2 {
    [p.x as f64, p.y as f64]
}

/// Drops the closing duplicate and repeated vertices of a GDS outline.
fn clean(points: Vec<P2>) -> Vec<P2> {
    let mut out: Vec<P2> = Vec::with_capacity(points.len());
    for p in points {
        if out.last().is_none_or(|q| (p[0] - q[0]).hypot(p[1] - q[1]) > 0.0) {
            out.push(p);
        }
    }
    while out.len() > 1 && out[0] == out[out.len() - 1] {
        out.pop();
    }
    out
}

/// Reads `path` and flattens `top_cell` (or the one top-level cell).
pub fn read(path: &Path, top_cell: Option<&str>) -> Result<Layout, String> {
    let lib = GdsLibrary::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let unit = lib.units.db_unit();
    let by_name: HashMap<&str, &GdsStruct> = lib.structs.iter().map(|s| (s.name.as_str(), s)).collect();
    let cell = match top_cell {
        Some(name) => *by_name.get(name).ok_or_else(|| {
            let names: Vec<&str> = lib.structs.iter().map(|s| s.name.as_str()).collect();
            format!("top_cell {name:?} not in GDS; available: {}", names.join(", "))
        })?,
        None => {
            let referenced: BTreeSet<&str> = lib
                .structs
                .iter()
                .flat_map(|s| &s.elems)
                .filter_map(|e| match e {
                    GdsElement::GdsStructRef(r) => Some(r.name.as_str()),
                    GdsElement::GdsArrayRef(r) => Some(r.name.as_str()),
                    _ => None,
                })
                .collect();
            let tops: Vec<&GdsStruct> = lib.structs.iter().filter(|s| !referenced.contains(s.name.as_str())).collect();
            match tops.as_slice() {
                [one] => *one,
                _ => {
                    let names: Vec<&str> = tops.iter().map(|s| s.name.as_str()).collect();
                    return Err(format!("GDS has {} top-level cells ({}); specify top_cell=...", tops.len(), names.join(", ")));
                }
            }
        }
    };
    let mut polygons = Vec::new();
    let scale = Affine { m: [[unit, 0.0], [0.0, unit]], t: [0.0, 0.0] };
    flatten(cell, &scale, &by_name, &mut polygons, 0)?;
    Ok(Layout { cell: cell.name.clone(), polygons })
}

fn flatten(cell: &GdsStruct, to_world: &Affine, cells: &HashMap<&str, &GdsStruct>, out: &mut Vec<Polygon>, depth: usize) -> Result<(), String> {
    if depth > 64 {
        return Err(format!("GDS: references nest deeper than 64 levels at cell {:?} (a cycle?)", cell.name));
    }
    let child = |name: &str| cells.get(name).copied().ok_or_else(|| format!("GDS: reference to missing cell {name:?}"));
    for e in &cell.elems {
        match e {
            GdsElement::GdsBoundary(b) => {
                let points = clean(b.xy.iter().map(|p| to_world.apply(point(p))).collect());
                out.push(Polygon { layer: b.layer as i32, datatype: b.datatype as i32, points });
            }
            GdsElement::GdsBox(b) => {
                let points = clean(b.xy.iter().map(|p| to_world.apply(point(p))).collect());
                out.push(Polygon { layer: b.layer as i32, datatype: b.boxtype as i32, points });
            }
            GdsElement::GdsStructRef(r) => {
                let place = Affine::placement(point(&r.xy), r.strans.as_ref());
                flatten(child(&r.name)?, &to_world.then(&place), cells, out, depth + 1)?;
            }
            GdsElement::GdsArrayRef(r) => {
                // xy: the origin, the origin displaced by `cols` column
                // pitches and by `rows` row pitches (in the parent frame)
                let o = point(&r.xy[0]);
                let (cols, rows) = (r.cols.max(1) as f64, r.rows.max(1) as f64);
                let dc = [(r.xy[1].x as f64 - o[0]) / cols, (r.xy[1].y as f64 - o[1]) / cols];
                let dr = [(r.xy[2].x as f64 - o[0]) / rows, (r.xy[2].y as f64 - o[1]) / rows];
                let target = child(&r.name)?;
                for i in 0..r.cols.max(1) {
                    for j in 0..r.rows.max(1) {
                        let (fi, fj) = (i as f64, j as f64);
                        let at = [o[0] + fi * dc[0] + fj * dr[0], o[1] + fi * dc[1] + fj * dr[1]];
                        let place = Affine::placement(at, r.strans.as_ref());
                        flatten(target, &to_world.then(&place), cells, out, depth + 1)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

impl Layout {
    /// The polygons on GDS `layer` (any datatype with `None`).
    pub fn on(&self, layer: i32, datatype: Option<i32>) -> impl Iterator<Item = &Polygon> {
        self.polygons.iter().filter(move |p| p.layer == layer && datatype.is_none_or(|d| p.datatype == d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_reflects_before_rotating() {
        let s = GdsStrans { reflected: true, abs_mag: false, abs_angle: false, mag: None, angle: Some(90.0) };
        let a = Affine::placement([10.0, 0.0], Some(&s));
        // (1, 2) -> reflect (1, -2) -> rotate 90 (2, 1) -> shift (12, 1)
        let p = a.apply([1.0, 2.0]);
        assert!((p[0] - 12.0).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12, "{p:?}");
        let b = Affine::ID.then(&a);
        assert_eq!(b.apply([1.0, 2.0]), p);
    }

    #[test]
    fn clean_drops_the_closing_vertex() {
        let c = clean(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]);
        assert_eq!(c, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
    }
}
