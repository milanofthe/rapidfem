// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! What sits on a geometry: the material of each solid, the ports and
//! boundary conditions on its faces and solids, and the mesh tags they get.
//!
//! The tags are handed out from the setup alone, so they are known before
//! the mesh: a material gets one tag for every solid it fills, the materials
//! in the order of the first solid they fill; then each physics object the
//! next tag, a periodic pair two, in the order they were added. A tag names
//! a face or volume group of the solver mesh and the entries of the
//! [`Model`] placed on it, and a group name for viewers and mesh files: the
//! material's or the condition's kind with a counter per kind, every driven
//! port in one shared `port_<n>` namespace.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rapidfem_core::model::{FaceSpec, MaterialSpec, Model, PmlSpec};

use crate::geometry::{FaceSel, ObjId};

/// What a material or a physics object sits on.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Target {
    /// A solid object.
    Object(ObjId),
    /// A face selection.
    Face(FaceSel),
    /// Group `index` (in file order) of a loaded mesh, named `name`, of
    /// dimension `dim`.
    Group { index: usize, name: String, dim: u8 },
}

impl Target {
    /// 3 for a volume, 2 for faces.
    pub fn dim(&self) -> u8 {
        match self {
            Target::Object(_) => 3,
            Target::Face(_) => 2,
            Target::Group { dim, .. } => *dim,
        }
    }
}

/// A physics object without its tag.
#[derive(Clone, Debug)]
pub enum Condition {
    /// A port or a face boundary condition of [`Model::faces`].
    Face(FaceSpec),
    Pec,
    /// The near-to-far-field surface.
    FarField,
    Pml(PmlSpec),
    /// A periodic pair: side a the targets, side b [`Physics::pair`].
    Periodic,
}

/// A physics object on its targets.
#[derive(Clone, Debug)]
pub struct Physics {
    pub condition: Condition,
    pub targets: Vec<Target>,
    /// Side b of a periodic pair.
    pub pair: Vec<Target>,
    /// Physics objects (by index) whose faces a wave port marks as internal
    /// conductors of its cross-section.
    pub pec: Vec<usize>,
}

impl Physics {
    pub fn new(condition: Condition, targets: Vec<Target>) -> Physics {
        Physics { condition, targets, pair: Vec::new(), pec: Vec::new() }
    }

    /// The stem of the group name.
    fn kind(&self) -> &'static str {
        match &self.condition {
            Condition::Face(f) if f.is_port() => "port",
            Condition::Face(FaceSpec::Abc { .. }) => "abc",
            Condition::Face(FaceSpec::Pmc { .. }) => "pmc",
            Condition::Face(FaceSpec::LumpedElement { .. }) => "lumpedelement",
            Condition::Face(_) => "surfaceimpedance",
            Condition::Pec => "pec",
            Condition::FarField => "farfieldsurface",
            Condition::Pml(_) => "pml",
            Condition::Periodic => "periodicboundary",
        }
    }
}

/// The tags of a [`Setup`] and the groups they name.
#[derive(Clone, Debug, Default)]
pub struct Tagging {
    /// The tag of each material (`None` for one that fills nothing).
    pub materials: Vec<Option<i32>>,
    /// The tag of each physics object (a periodic pair: side a, side b the
    /// next one).
    pub physics: Vec<i32>,
    /// Face groups `(tag, targets)`.
    pub faces: Vec<(i32, Vec<Target>)>,
    /// Volume groups `(tag, targets)`.
    pub volumes: Vec<(i32, Vec<Target>)>,
    /// The group name of every tag.
    pub names: BTreeMap<i32, String>,
}

impl Tagging {
    /// Whether no group holds anything to tag.
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty() && self.volumes.is_empty()
    }
}

/// The materials and physics placed on a geometry.
#[derive(Clone, Debug, Default)]
pub struct Setup {
    /// The material (index) of each solid or volume group.
    materials: BTreeMap<Target, usize>,
    /// The group name stem of each material ("air", "dielectric", ...).
    kinds: Vec<String>,
    physics: Vec<Physics>,
    /// Counts the changes, so a model can tell it is newer than its mesh.
    revision: u64,
}

impl Setup {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// A new material whose groups are named `kind_<n>`; returns its index.
    pub fn add_material(&mut self, kind: String) -> usize {
        self.kinds.push(kind);
        self.kinds.len() - 1
    }

    /// Fills a volume target with material `material` (`None` empties it);
    /// a face target takes no material and is skipped.
    pub fn set_material(&mut self, target: Target, material: Option<usize>) {
        if target.dim() != 3 {
            return;
        }
        match material {
            Some(m) => self.materials.insert(target, m),
            None => self.materials.remove(&target),
        };
        self.revision += 1;
    }

    pub fn material(&self, target: &Target) -> Option<usize> {
        self.materials.get(target).copied()
    }

    /// Adds a physics object; returns its index.
    pub fn add(&mut self, physics: Physics) -> usize {
        self.physics.push(physics);
        self.revision += 1;
        self.physics.len() - 1
    }

    pub fn physics(&self) -> &[Physics] {
        &self.physics
    }

    /// Whether a physics object sits on `target`.
    pub fn is_assigned(&self, target: &Target) -> bool {
        self.physics.iter().any(|p| p.targets.contains(target) || p.pair.contains(target))
    }

    /// The tags and groups, see the module docs.
    pub fn tagging(&self) -> Tagging {
        let mut t = Tagging { materials: vec![None; self.kinds.len()], ..Tagging::default() };
        // one counter per name stem, shared by materials and physics
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut name = |stem: &str| {
            let n = counts.entry(stem.to_string()).or_insert(0);
            *n += 1;
            format!("{stem}_{n}")
        };
        let split = |targets: &[Target], dim: u8| targets.iter().filter(|x| x.dim() == dim).cloned().collect::<Vec<_>>();
        let mut next = 1;
        // the materials in the order of the first solid they fill
        let mut order: Vec<(usize, Vec<Target>)> = Vec::new();
        for (target, &m) in &self.materials {
            match order.iter_mut().find(|(o, _)| *o == m) {
                Some((_, members)) => members.push(target.clone()),
                None => order.push((m, vec![target.clone()])),
            }
        }
        for (m, members) in order {
            t.materials[m] = Some(next);
            t.names.insert(next, name(&self.kinds[m]));
            t.volumes.push((next, members));
            next += 1;
        }
        for p in &self.physics {
            t.physics.push(next);
            if let Condition::Periodic = p.condition {
                let base = name(p.kind());
                t.faces.push((next, split(&p.targets, 2)));
                t.faces.push((next + 1, split(&p.pair, 2)));
                t.names.insert(next, format!("{base}_a"));
                t.names.insert(next + 1, format!("{base}_b"));
                next += 2;
                continue;
            }
            let (faces, volumes) = (split(&p.targets, 2), split(&p.targets, 3));
            if !volumes.is_empty() {
                t.volumes.push((next, volumes));
            }
            if !faces.is_empty() {
                t.faces.push((next, faces));
            }
            t.names.insert(next, name(p.kind()));
            next += 1;
        }
        t
    }

    /// The model under the tags of `tagging`, the material of index `m`
    /// described by `materials[m]` (its volume tag is set here). A solid
    /// that a PML sits on takes no material entry: the layer carries its own
    /// base permittivity and permeability.
    pub fn model(&self, tagging: &Tagging, materials: &[MaterialSpec]) -> Result<Model, String> {
        let mut model = Model::default();
        let pml: BTreeSet<&Target> = self
            .physics
            .iter()
            .filter(|p| matches!(p.condition, Condition::Pml(_)))
            .flat_map(|p| &p.targets)
            .collect();
        let mut placed = vec![false; materials.len()];
        for (target, &m) in &self.materials {
            if pml.contains(target) || placed[m] {
                continue;
            }
            placed[m] = true;
            let volume_tag = tagging.materials[m].ok_or_else(|| format!("material {m} has no tag"))?;
            model.materials.push(MaterialSpec { volume_tag, ..materials[m].clone() });
        }
        for (i, p) in self.physics.iter().enumerate() {
            let tag = tagging.physics[i];
            match &p.condition {
                Condition::Face(spec) => {
                    let mut spec = spec.clone();
                    spec.set_tag(tag);
                    if let FaceSpec::WaveNumerical { pec_tags, .. } = &mut spec {
                        *pec_tags = p
                            .pec
                            .iter()
                            .filter(|&&j| matches!(self.physics.get(j), Some(q) if !matches!(q.condition, Condition::Periodic)))
                            .map(|&j| tagging.physics[j])
                            .collect();
                    }
                    model.faces.push(spec);
                }
                Condition::Pec => model.pec_tags.push(tag),
                Condition::FarField => {
                    if model.far_field_tag.is_some() {
                        return Err("multiple FarFieldSurface objects, but only one near-field-to-far-field \
                                    surface is supported. Pass every face to a single FarFieldSurface(...) \
                                    call (e.g. rf.FarFieldSurface(*air.faces.hull))."
                            .into());
                    }
                    model.far_field_tag = Some(tag);
                }
                Condition::Pml(spec) => model.pml.push(PmlSpec { volume_tag: tag, ..spec.clone() }),
                Condition::Periodic => model.periodic.push((tag, tag + 1)),
            }
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(object: ObjId) -> Target {
        Target::Face(FaceSel::sheet(object))
    }

    #[test]
    fn materials_then_physics_take_tags_in_order() {
        let mut s = Setup::default();
        let air = s.add_material("air".into());
        let diel = s.add_material("dielectric".into());
        // the dielectric fills the earlier solid, so it is tagged first
        s.set_material(Target::Object(3), Some(air));
        s.set_material(Target::Object(1), Some(diel));
        s.set_material(Target::Object(5), Some(air));
        let pec = s.add(Physics::new(Condition::Pec, vec![face(7)]));
        let abc = FaceSpec::Abc { tag: 0 };
        s.add(Physics::new(Condition::Face(abc), vec![face(8)]));
        let mut pair = Physics::new(Condition::Periodic, vec![face(9)]);
        pair.pair = vec![face(10)];
        s.add(pair);
        let t = s.tagging();
        assert_eq!(t.materials, vec![Some(2), Some(1)]);
        assert_eq!(t.volumes[1], (2, vec![Target::Object(3), Target::Object(5)]));
        assert_eq!(t.physics, vec![3, 4, 5]);
        assert_eq!(t.names[&1], "dielectric_1");
        assert_eq!(t.names[&3], "pec_1");
        assert_eq!(t.names[&5], "periodicboundary_1_a");
        assert_eq!(t.names[&6], "periodicboundary_1_b");
        assert_eq!(pec, 0);
        assert!(s.is_assigned(&face(10)) && !s.is_assigned(&face(11)));
        let m = s.model(&t, &[MaterialSpec::vacuum(0), MaterialSpec::vacuum(0)]).unwrap();
        assert_eq!(m.materials.len(), 2);
        assert_eq!(m.pec_tags, vec![3]);
        assert_eq!(m.faces[0].tag(), 4);
        assert_eq!(m.periodic, vec![(5, 6)]);
    }
}
