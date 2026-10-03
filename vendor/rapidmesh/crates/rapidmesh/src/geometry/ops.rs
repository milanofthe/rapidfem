//! Operations on what a geometry holds: moving, copying and arraying solids
//! and sheets, and intersecting solids.

use super::{Geometry, Solid};
use crate::mesh::SolidInfo;
use crate::shapes::Sheet;
use crate::{Error, Result};
use rapidmesh_geom::{extrude_sheet, FaceTag, Faceted, SurfaceKind};

type P3 = [f64; 3];

/// A sheet added to a [`Geometry`]: its index among the sheets (insertion
/// order) and its face tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SheetRef {
    pub index: u32,
    pub tag: u32,
}

/// A solid or a sheet of a [`Geometry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Object {
    Solid(Solid),
    Sheet(SheetRef),
}

impl From<Solid> for Object {
    fn from(s: Solid) -> Object {
        Object::Solid(s)
    }
}

impl From<SheetRef> for Object {
    fn from(s: SheetRef) -> Object {
        Object::Sheet(s)
    }
}

/// A change of placement. Moves, turns and mirrors keep every analytic
/// carrier; a stretch keeps them only when uniform (unequal factors keep
/// planes, discrete patches and NURBS; the other curved faces become
/// faceted).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform {
    Translate(P3),
    /// By `angle` radians about the axis along `axis` through `center`
    /// (right-handed).
    Rotate {
        angle: f64,
        axis: P3,
        center: P3,
    },
    /// Across the plane through `point` with normal `normal`.
    Mirror {
        normal: P3,
        point: P3,
    },
    /// By `factors` along x, y and z about `center`.
    Stretch {
        factors: P3,
        center: P3,
    },
}

impl Transform {
    /// `f` transformed, or why it cannot be.
    pub(crate) fn apply(&self, f: &Faceted) -> Result<Faceted> {
        let nonzero = |v: P3, what: &str| {
            if v.iter().all(|c| c.is_finite()) && v.iter().any(|&c| c != 0.0) {
                Ok(())
            } else {
                Err(Error::Invalid(format!("{what} must be a nonzero vector")))
            }
        };
        Ok(match *self {
            Transform::Translate(v) => f.translated(v),
            Transform::Rotate {
                angle,
                axis,
                center,
            } => {
                nonzero(axis, "a rotation axis")?;
                f.rotated(center, axis, angle)
            }
            Transform::Mirror { normal, point } => {
                nonzero(normal, "a mirror normal")?;
                f.mirrored(normal, point)
            }
            Transform::Stretch { factors, center } => {
                if factors.iter().any(|&k| !(k.is_finite() && k != 0.0)) {
                    return Err(Error::Invalid(format!(
                        "stretch factors {factors:?} must be finite and nonzero"
                    )));
                }
                f.scaled(factors, center)
            }
        })
    }

    /// This step taken `k` times, in one go (no drift from repeating it).
    pub fn times(&self, k: u32) -> Transform {
        let kf = k as f64;
        match *self {
            Transform::Translate(v) => Transform::Translate(v.map(|c| c * kf)),
            Transform::Rotate {
                angle,
                axis,
                center,
            } => Transform::Rotate {
                angle: angle * kf,
                axis,
                center,
            },
            Transform::Mirror { .. } if k.is_multiple_of(2) => Transform::Translate([0.0; 3]),
            Transform::Mirror { .. } => *self,
            Transform::Stretch { factors, center } => Transform::Stretch {
                factors: factors.map(|c| c.powi(k as i32)),
                center,
            },
        }
    }

    /// The linear part (a direction maps by it).
    pub(crate) fn linear(&self) -> [[f64; 3]; 3] {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let probe = |d: P3| -> P3 {
            // Map the direction as the difference of two mapped points.
            let at = |p: P3| self.point(p);
            let (a, b) = (at([0.0; 3]), at(d));
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
        };
        let cols = id.map(probe);
        std::array::from_fn(|i| std::array::from_fn(|j| cols[j][i]))
    }

    /// Where the point `p` goes.
    pub(crate) fn point(&self, p: P3) -> P3 {
        let sub = |a: P3, b: P3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let add = |a: P3, b: P3| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        let dot = |a: P3, b: P3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        match *self {
            Transform::Translate(v) => add(p, v),
            Transform::Rotate {
                angle,
                axis,
                center,
            } => {
                let l = dot(axis, axis).sqrt();
                let u = axis.map(|c| c / l);
                let d = sub(p, center);
                let (s, c) = angle.sin_cos();
                let along = dot(u, d);
                let cr = [
                    u[1] * d[2] - u[2] * d[1],
                    u[2] * d[0] - u[0] * d[2],
                    u[0] * d[1] - u[1] * d[0],
                ];
                let r: P3 =
                    std::array::from_fn(|k| d[k] * c + cr[k] * s + u[k] * along * (1.0 - c));
                add(center, r)
            }
            Transform::Mirror { normal, point } => {
                let l = dot(normal, normal).sqrt();
                let n = normal.map(|c| c / l);
                let off = dot(sub(p, point), n);
                sub(p, n.map(|c| 2.0 * off * c))
            }
            Transform::Stretch { factors, center } => {
                std::array::from_fn(|k| center[k] + factors[k] * (p[k] - center[k]))
            }
        }
    }
}

impl Geometry {
    /// Moves, turns, mirrors or stretches `object` in place; its faces keep
    /// their origin, so selections by role still find them.
    pub fn transform(&mut self, object: impl Into<Object>, t: Transform) -> Result<()> {
        match object.into() {
            Object::Solid(s) => {
                let i = s.index as usize;
                let f = self
                    .scene
                    .solid(i)
                    .ok_or_else(|| Error::Invalid(format!("no solid {i}")))?;
                let moved = t.apply(f)?;
                self.scene_mut().replace_solid(i, moved);
            }
            Object::Sheet(s) => {
                let i = s.index as usize;
                let f = self
                    .scene
                    .sheet(i)
                    .ok_or_else(|| Error::Invalid(format!("no sheet {i}")))?;
                let moved = t.apply(f)?;
                self.scene_mut().replace_sheet(i, moved);
                self.sheets[i].1.push(t);
            }
        }
        Ok(())
    }

    /// A copy of `object` in the same place: a solid in a region of its own
    /// with the same target size (a void stays a void), a sheet with the
    /// same face tag.
    pub fn copy(&mut self, object: impl Into<Object>) -> Result<Object> {
        match object.into() {
            Object::Solid(s) => {
                let i = s.index as usize;
                let f = self
                    .scene
                    .solid(i)
                    .ok_or_else(|| Error::Invalid(format!("no solid {i}")))?
                    .clone();
                let region = if s.region == 0 {
                    self.scene_mut().add_void(f);
                    0
                } else {
                    let r = self.scene_mut().add_solid(f).0;
                    let maxh = self
                        .solid_maxh
                        .iter()
                        .find(|(g, _)| *g == s.region)
                        .map(|&(_, h)| h);
                    if let Some(h) = maxh {
                        self.solid_maxh.push((r, h));
                    }
                    r
                };
                let index = self.labels.solids.len() as u32;
                let roles = self.roles(s).to_vec();
                self.labels.solids.push(SolidInfo {
                    region,
                    label: None,
                    roles,
                });
                Ok(Object::Solid(Solid { region, index }))
            }
            Object::Sheet(s) => {
                let i = s.index as usize;
                let f = self
                    .scene
                    .sheet(i)
                    .ok_or_else(|| Error::Invalid(format!("no sheet {i}")))?
                    .clone();
                self.scene_mut().add_sheet(f, FaceTag(s.tag));
                self.sheets.push(self.sheets[i].clone());
                Ok(Object::Sheet(SheetRef {
                    index: self.sheets.len() as u32 - 1,
                    tag: s.tag,
                }))
            }
        }
    }

    /// `count` objects: `object` first, then copies moved by `step` taken
    /// once, twice, ... from it (a row by a translation, a ring by a
    /// rotation).
    pub fn array(
        &mut self,
        object: impl Into<Object>,
        count: u32,
        step: Transform,
    ) -> Result<Vec<Object>> {
        let object = object.into();
        let mut out = vec![object];
        for k in 1..count {
            let c = self.copy(object)?;
            self.transform(c, step.times(k))?;
            out.push(c);
        }
        Ok(out)
    }

    /// `target` becomes what it has in common with every tool (the exact
    /// boolean); the tools are used up (they hold nothing after). The
    /// target's faces keep their roles; a face that came from a tool gets a
    /// role after them.
    pub fn intersect(&mut self, target: Solid, tools: &[Solid]) -> Result<Solid> {
        let i = target.index as usize;
        let mut f = self
            .scene
            .solid(i)
            .ok_or_else(|| Error::Invalid(format!("no solid {i}")))?
            .clone();
        for t in tools {
            if t.index == target.index {
                return Err(Error::Invalid("a solid cannot be its own tool".into()));
            }
            let g = self
                .scene
                .solid(t.index as usize)
                .ok_or_else(|| Error::Invalid(format!("no solid {}", t.index)))?;
            f = f
                .common(g)
                .map_err(|e| Error::Invalid(format!("intersect: {e}")))?;
        }
        self.scene_mut().replace_solid(i, f);
        for t in tools {
            self.scene_mut()
                .replace_solid(t.index as usize, Faceted::new());
        }
        Ok(target)
    }

    /// The solid `sheet` sweeps along `vector` (not in its plane), in a
    /// region of its own with target size `maxh`. The sheet stays as it is,
    /// the solid's bottom face on it. Surfaces: bottom, top, then the walls
    /// (for a disc one cylinder, which needs the vector along its axis); the
    /// first two named `bottom` and `top`.
    pub fn extrude(&mut self, sheet: SheetRef, vector: P3, maxh: Option<f64>) -> Result<Solid> {
        let i = sheet.index as usize;
        let f = self
            .scene
            .sheet(i)
            .ok_or_else(|| Error::Invalid(format!("no sheet {i}")))?
            .clone();
        let (desc, ops) = self.sheets[i].clone();
        let rim = match desc {
            Sheet::Disc {
                radius,
                center,
                axis,
                ..
            } => Some(disc_rim(radius, center, axis, &ops, vector)?),
            Sheet::Nurbs { .. } => {
                return Err(Error::Invalid(
                    "a NURBS sheet does not extrude (it is not flat)".into(),
                ))
            }
            _ => None,
        };
        let solid = extrude_sheet(&f, vector, rim).map_err(Error::Invalid)?;
        let region = self.scene_mut().add_solid(solid).0;
        if let Some(h) = maxh {
            self.solid_maxh.push((region, h));
        }
        let index = self.labels.solids.len() as u32;
        self.labels.solids.push(SolidInfo {
            region,
            label: None,
            roles: crate::shapes::CAP_ROLES.map(String::from).to_vec(),
        });
        Ok(Solid { region, index })
    }
}

/// The cylinder under a disc of `radius` about `center` square to `axis`,
/// taken where `ops` moved it and swept along `vector`.
fn disc_rim(
    radius: f64,
    center: P3,
    axis: P3,
    ops: &[Transform],
    vector: P3,
) -> Result<SurfaceKind> {
    let unit = |v: P3| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        v.map(|c| c / l)
    };
    let (mut c, mut a, mut r) = (center, unit(axis), radius);
    for t in ops {
        if let Transform::Stretch { factors, .. } = t {
            let s = factors[0].abs();
            if factors.iter().any(|k| k.abs() != s) {
                return Err(Error::Invalid(
                    "a disc stretched unevenly is an ellipse; it does not extrude".into(),
                ));
            }
            r *= s;
        }
        let l = t.linear();
        c = t.point(c);
        a = unit(std::array::from_fn(|i| {
            (0..3).map(|j| l[i][j] * a[j]).sum()
        }));
    }
    let v = unit(vector);
    let cross = [
        a[1] * v[2] - a[2] * v[1],
        a[2] * v[0] - a[0] * v[2],
        a[0] * v[1] - a[1] * v[0],
    ];
    if cross.iter().map(|x| x * x).sum::<f64>().sqrt() > 1e-9 {
        return Err(Error::Invalid(
            "a disc extrudes along its axis only (an oblique sweep is an elliptic cylinder)".into(),
        ));
    }
    Ok(SurfaceKind::Cylinder {
        center: c,
        axis: a,
        radius: r,
    })
}
