//! Operations on what a geometry holds: moving, copying and arraying solids
//! and sheets, intersecting solids and the booleans of sheets.

use super::{Geometry, Solid};
use crate::mesh::SolidInfo;
use crate::{Error, Result};
use rapidmesh_exact::vector::{Affine, V3};
use rapidmesh_geom::{extrude_sheet, FaceTag, Faceted};

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
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Transform {
    Translate(V3),
    /// By `angle` radians about the axis along `axis` through `center`
    /// (right-handed).
    Rotate {
        angle: f64,
        axis: V3,
        center: V3,
    },
    /// Across the plane through `point` with normal `normal`.
    Mirror {
        normal: V3,
        point: V3,
    },
    /// By `factors` along x, y and z about `center`.
    Stretch {
        factors: V3,
        center: V3,
    },
}

impl Transform {
    /// The affine map of this step, or why it is none (a zero axis or
    /// normal, a stretch factor that is zero or not finite).
    pub fn affine(&self) -> Result<Affine> {
        let nonzero = |v: V3, what: &str| {
            Error::Invalid(format!("{what} {v:?} must be a finite nonzero vector"))
        };
        let finite = |v: V3| v.iter().all(|c| c.is_finite());
        match *self {
            Transform::Translate(v) => Ok(Affine::translation(v)),
            Transform::Rotate {
                angle,
                axis,
                center,
            } => Affine::rotation(center, axis, angle)
                .filter(|_| finite(axis))
                .ok_or_else(|| nonzero(axis, "a rotation axis")),
            Transform::Mirror { normal, point } => Affine::mirror(point, normal)
                .filter(|_| finite(normal))
                .ok_or_else(|| nonzero(normal, "a mirror normal")),
            Transform::Stretch { factors, center } => {
                if factors.iter().any(|&k| !(k.is_finite() && k != 0.0)) {
                    return Err(Error::Invalid(format!(
                        "stretch factors {factors:?} must be finite and nonzero"
                    )));
                }
                Ok(Affine::stretch(center, factors))
            }
        }
    }

    /// `f` transformed, or why it cannot be.
    pub(crate) fn apply(&self, f: &Faceted) -> Result<Faceted> {
        Ok(f.transformed(&self.affine()?))
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
                Ok(Object::Sheet(SheetRef {
                    index: self.scene.sheet_count() as u32 - 1,
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

    /// `target` becomes its exact boolean `op` with every tool in turn
    /// (`Union`, `Difference` or `Intersection`); all lie in one plane. The
    /// target keeps its tag and its input points bit for bit where the plane
    /// is square to an axis; a round rim that is left (a disc's, a hole one
    /// cut) stays a circle, and extrudes into a cylinder. The tools are used
    /// up (they hold nothing after).
    pub fn sheet_boolean(
        &mut self,
        op: rapidmesh_geom::BoolOp,
        target: SheetRef,
        tools: &[SheetRef],
    ) -> Result<SheetRef> {
        let sheet = |g: &Geometry, i: u32| {
            g.scene
                .sheet(i as usize)
                .cloned()
                .ok_or_else(|| Error::Invalid(format!("no sheet {i}")))
        };
        let mut f = sheet(self, target.index)?;
        for t in tools {
            if t.index == target.index {
                return Err(Error::Invalid("a sheet cannot be its own tool".into()));
            }
            f = rapidmesh_geom::sheet_boolean(&f, &sheet(self, t.index)?, op)
                .map_err(|e| Error::Invalid(format!("sheet boolean: {e}")))?;
        }
        self.scene_mut().replace_sheet(target.index as usize, f);
        for t in tools {
            self.scene_mut()
                .replace_sheet(t.index as usize, Faceted::new());
        }
        Ok(target)
    }

    /// The solid `sheet` sweeps along `vector` (not in its plane), in a
    /// region of its own with target size `maxh`. The sheet stays as it is,
    /// the solid's bottom face on it. Surfaces: bottom, top, then the walls
    /// (along a circle the sheet declares, a disc's rim or a round hole cut
    /// by one, the cylinder, which needs the vector along its axis); the
    /// first two named `bottom` and `top`.
    pub fn extrude(&mut self, sheet: SheetRef, vector: V3, maxh: Option<f64>) -> Result<Solid> {
        let i = sheet.index as usize;
        let f = self
            .scene
            .sheet(i)
            .ok_or_else(|| Error::Invalid(format!("no sheet {i}")))?
            .clone();
        let solid = extrude_sheet(&f, vector).map_err(Error::Invalid)?;
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
