// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The process stack: patterned layers, background dielectrics and the
//! materials they reference.
//!
//! - patterned layers ([`PdkLayer`]): anything drawn in the GDS, metals,
//!   vias, patterned dielectric cuts
//! - background dielectrics ([`DielectricLayer`]): the unpatterned slabs the
//!   patterned layers are embedded in, substrate, EPI, oxide, passivation,
//!   air
//! - materials ([`StackMaterial`]): a shared name to properties table
//!
//! Ingestion: the gds2palace / ADS stackup XML ([`Stack::from_xml`]), the
//! presets ([`Stack::from_pdk`]; SG13G2 is the bundled IHP XML) and the
//! rapidpassives `Pdk` JSON ([`Stack::from_json`] / [`Stack::to_json`]).
//! All lengths in metres.

use serde_json::{json, Map, Value};

/// The first child element of `n` named `tag`.
fn child<'a, 'i>(n: roxmltree::Node<'a, 'i>, tag: &str) -> Option<roxmltree::Node<'a, 'i>> {
    n.children().find(|c| c.has_tag_name(tag))
}

/// The required `key` of a JSON object.
fn req<'v>(v: &'v Value, key: &str) -> Result<&'v Value, String> {
    v.get(key).ok_or_else(|| format!("stack JSON: missing {key:?}"))
}

/// At or above this conductivity (S/m) a conductor is the gds2palace
/// "LOWLOSS" idealisation: model it as PEC.
pub const PEC_SIGMA: f64 = 1e10;

const UM: f64 = 1e-6;

/// The IHP SG13G2 stackup XML of the public gds2palace repository (200 um
/// thinned substrate with backside metal and LBE layers).
const SG13G2_XML: &str = include_str!("../../data/ihp_sg13g2_200um.xml");

/// One named material of the process.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "rapidfem.rfic", get_all, set_all, from_py_object))]
pub struct StackMaterial {
    pub name: String,
    /// "conductor", "dielectric" or "semiconductor".
    pub kind: String,
    pub er: f64,
    pub tand: f64,
    /// S/m.
    pub sigma: f64,
    pub color: String,
}

impl StackMaterial {
    pub fn new(name: impl Into<String>, kind: &str) -> StackMaterial {
        StackMaterial { name: name.into(), kind: kind.into(), er: 1.0, tand: 0.0, sigma: 0.0, color: "#888".into() }
    }

    /// Air: no conductor, εr = 1, no loss.
    pub fn is_air(&self) -> bool {
        self.kind != "conductor" && self.er == 1.0 && self.sigma == 0.0
    }
}

/// One unpatterned background slab, `z` its bottom.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "rapidfem.rfic", get_all, set_all, from_py_object))]
pub struct DielectricLayer {
    pub name: String,
    /// Name in [`Stack::materials`].
    pub material: String,
    pub z: f64,
    pub thickness: f64,
}

impl DielectricLayer {
    pub fn z_top(&self) -> f64 {
        self.z + self.thickness
    }
}

/// One patterned (GDS-drawn) layer, `z` its bottom. Mirrors the
/// rapidpassives `PdkLayer`, so one JSON describes a stack on both sides.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "rapidfem.rfic", get_all, set_all, from_py_object))]
pub struct PdkLayer {
    pub name: String,
    pub gds: i32,
    pub datatype: i32,
    pub z: f64,
    pub thickness: f64,
    pub color: String,
    /// "metal", "via", "poly", "diffusion", "substrate", "oxide",
    /// "dielectric" or "other".
    pub r#type: String,
    pub er: f64,
    pub ur: f64,
    pub tand: f64,
    /// Bulk conductivity (S/m); [`PEC_SIGMA`] and above is PEC.
    pub sigma: f64,
    /// Optional name in [`Stack::materials`].
    pub material: Option<String>,
}

impl PdkLayer {
    /// A layer with the material defaults (εr = 1, no loss).
    pub fn new(name: &str, gds: i32, datatype: i32, z: f64, thickness: f64, color: &str, r#type: &str) -> PdkLayer {
        PdkLayer {
            name: name.into(),
            gds,
            datatype,
            z,
            thickness,
            color: color.into(),
            r#type: r#type.into(),
            er: 1.0,
            ur: 1.0,
            tand: 0.0,
            sigma: 0.0,
            material: None,
        }
    }

    pub fn z_top(&self) -> f64 {
        self.z + self.thickness
    }

    /// Idealised lossless conductor (gds2palace LOWLOSS convention).
    pub fn is_pec(&self) -> bool {
        self.sigma >= PEC_SIGMA
    }
}

/// A complete process stack: patterned layers and background dielectrics,
/// both sorted bottom to top, and the materials table in file order.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "rapidfem.rfic", from_py_object))]
pub struct Stack {
    pub name: String,
    pub layers: Vec<PdkLayer>,
    pub dielectrics: Vec<DielectricLayer>,
    pub materials: Vec<StackMaterial>,
}

impl Stack {
    /// A stack, layers sorted by (z, thickness) and dielectrics by z.
    pub fn new(name: String, mut layers: Vec<PdkLayer>, mut dielectrics: Vec<DielectricLayer>, materials: Vec<StackMaterial>) -> Stack {
        layers.sort_by(|a, b| (a.z, a.thickness).partial_cmp(&(b.z, b.thickness)).unwrap());
        dielectrics.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap());
        Stack { name, layers, dielectrics, materials }
    }

    /// A preset by PDK name: "sky130" or "sg13g2" (case, '-' and '_'
    /// ignored).
    pub fn from_pdk(name: &str) -> Result<Stack, String> {
        let n: String = name.to_lowercase().chars().filter(|c| *c != '-' && *c != '_').collect();
        match n.as_str() {
            "sky130" | "skywater130" => Ok(Stack::sky130()),
            "sg13g2" | "ihpsg13g2" => Ok(Stack::sg13g2()),
            _ => Err(format!("unknown PDK {name:?}; available: sky130, sg13g2")),
        }
    }

    /// IHP SG13G2, from the bundled stackup XML.
    pub fn sg13g2() -> Stack {
        Stack::from_xml(SG13G2_XML, "SG13G2").expect("the bundled SG13G2 stackup parses")
    }

    /// SkyWater SKY130; layer numbering and heights as in the rapidpassives
    /// `pdk.ts` (z = 0 at the bottom of li1, polysilicon below).
    pub fn sky130() -> Stack {
        let l = |name: &str, gds, dt, z: f64, t: f64, color: &str, ty: &str, er: f64, sigma: f64| {
            let mut p = PdkLayer::new(name, gds, dt, z * UM, t * UM, color, ty);
            p.er = er;
            p.sigma = sigma;
            p
        };
        let layers = vec![
            l("poly", 66, 20, -0.18, 0.18, "#c4725e", "poly", 4.2, 0.0),
            l("licon1", 66, 44, -0.10, 0.10, "#5a5a62", "via", 1.0, 4.1e7),
            l("li1", 67, 20, 0.00, 0.10, "#7b5e8a", "metal", 1.0, 4.1e7),
            l("mcon", 67, 44, 0.10, 0.27, "#5a5a62", "via", 1.0, 4.1e7),
            l("met1", 68, 20, 0.37, 0.36, "#6bbf8a", "metal", 1.0, 4.1e7),
            l("via", 68, 44, 0.73, 0.27, "#5a5a62", "via", 1.0, 4.1e7),
            l("met2", 69, 20, 1.00, 0.36, "#4a9ec2", "metal", 1.0, 4.1e7),
            l("via2", 69, 44, 1.36, 0.42, "#6e6e78", "via", 1.0, 4.1e7),
            l("met3", 70, 20, 1.78, 0.845, "#5aad78", "metal", 1.0, 4.1e7),
            l("via3", 70, 44, 2.625, 0.39, "#6e6e78", "via", 1.0, 4.1e7),
            l("met4", 71, 20, 3.015, 0.845, "#d9513c", "metal", 1.0, 4.1e7),
            l("via4", 71, 44, 3.86, 0.505, "#7a7a84", "via", 1.0, 4.1e7),
            l("met5", 72, 20, 4.365, 1.26, "#e8944a", "metal", 1.0, 4.1e7),
        ];
        Stack::single_slab("SKY130".into(), layers, 300.0 * UM, 11.9, 10.0, 4.2, 0.0)
    }

    /// A stack whose background is one substrate slab below the lowest layer
    /// and one oxide slab up to the highest (the rapidpassives shape).
    fn single_slab(
        name: String,
        layers: Vec<PdkLayer>,
        substrate_thickness: f64,
        substrate_er: f64,
        substrate_sigma: f64,
        oxide_er: f64,
        oxide_tand: f64,
    ) -> Stack {
        let bottom = layers.iter().map(|l| l.z).fold(f64::INFINITY, f64::min);
        let top = layers.iter().map(|l| l.z_top()).fold(f64::NEG_INFINITY, f64::max);
        let mut substrate = StackMaterial::new("substrate", "semiconductor");
        substrate.er = substrate_er;
        substrate.sigma = substrate_sigma;
        let mut oxide = StackMaterial::new("oxide", "dielectric");
        oxide.er = oxide_er;
        oxide.tand = oxide_tand;
        let dielectrics = vec![
            DielectricLayer { name: "substrate".into(), material: "substrate".into(), z: bottom - substrate_thickness, thickness: substrate_thickness },
            DielectricLayer { name: "oxide".into(), material: "oxide".into(), z: bottom, thickness: top - bottom },
        ];
        Stack::new(name, layers, dielectrics, vec![substrate, oxide])
    }

    /// Parses a gds2palace / ADS stackup XML (`<Stackup schemaVersion="2.0">`):
    /// a `<Materials>` table, the top-down `<Dielectrics>` anchored by
    /// `<Substrate Offset=...>` (the depth of the stack bottom below z = 0)
    /// and the `<Layers>` with GDS number, z range and material.
    pub fn from_xml(text: &str, name: &str) -> Result<Stack, String> {
        let doc = roxmltree::Document::parse(text).map_err(|e| format!("stackup XML: {e}"))?;
        let root = doc.root_element();
        if root.tag_name().name() != "Stackup" {
            return Err(format!("not a stackup XML: root element is {:?}", root.tag_name().name()));
        }
        let num = |n: roxmltree::Node, attr: &str, default: f64| -> Result<f64, String> {
            n.attribute(attr).map_or(Ok(default), |v| v.trim().parse().map_err(|_| format!("stackup XML: {attr}={v:?} is not a number")))
        };

        let mut materials = Vec::new();
        for m in root.descendants().filter(|n| n.has_tag_name("Material")) {
            let mut mat = StackMaterial::new(m.attribute("Name").unwrap_or_default(), &m.attribute("Type").unwrap_or("dielectric").to_lowercase());
            mat.er = num(m, "Permittivity", 1.0)?;
            mat.tand = num(m, "DielectricLossTangent", 0.0)?;
            mat.sigma = num(m, "Conductivity", 0.0)?;
            mat.color = format!("#{}", m.attribute("Color").unwrap_or("888888").trim_start_matches('#'));
            materials.push(mat);
        }

        let elayers = child(root, "ELayers").ok_or("stackup XML has no <ELayers> element")?;
        let unit = match elayers.attribute("LengthUnit").unwrap_or("um").to_lowercase().as_str() {
            "m" => 1.0,
            "mm" => 1e-3,
            "um" => 1e-6,
            "nm" => 1e-9,
            u => return Err(format!("stackup XML: unknown LengthUnit {u:?}")),
        };

        // <Substrate Offset="d"/> puts the bottom of the lowest slab at z = -d;
        // the slabs, listed top-down, stack upward in reverse file order.
        let layers_el = child(elayers, "Layers");
        let offset = match layers_el.and_then(|l| child(l, "Substrate")) {
            Some(s) => num(s, "Offset", 0.0)?,
            None => 0.0,
        };
        let mut dielectrics = Vec::new();
        if let Some(d_el) = child(elayers, "Dielectrics") {
            let mut z = -offset * unit;
            let slabs: Vec<_> = d_el.descendants().filter(|n| n.has_tag_name("Dielectric")).collect();
            for d in slabs.into_iter().rev() {
                let t = num(d, "Thickness", 0.0)? * unit;
                dielectrics.push(DielectricLayer {
                    name: d.attribute("Name").unwrap_or_default().into(),
                    material: d.attribute("Material").unwrap_or_default().into(),
                    z,
                    thickness: t,
                });
                z += t;
            }
        }

        let mut layers = Vec::new();
        if let Some(l_el) = layers_el {
            for l in l_el.descendants().filter(|n| n.has_tag_name("Layer")) {
                let z_min = num(l, "Zmin", f64::NAN)? * unit;
                let z_max = num(l, "Zmax", f64::NAN)? * unit;
                if z_min.is_nan() || z_max.is_nan() {
                    return Err("stackup XML: a <Layer> without Zmin / Zmax".into());
                }
                let mname = l.attribute("Material");
                let fallback = StackMaterial::new(mname.unwrap_or("unknown"), "dielectric");
                let mat = mname.and_then(|n| materials.iter().find(|m: &&StackMaterial| m.name == n)).unwrap_or(&fallback);
                let ty = match l.attribute("Type").unwrap_or("conductor").to_lowercase().as_str() {
                    "conductor" => "metal",
                    "via" => "via",
                    "dielectric" => "dielectric",
                    _ => "other",
                };
                let gds = l.attribute("Layer").and_then(|v| v.trim().parse().ok()).ok_or("stackup XML: a <Layer> without a numeric Layer")?;
                let datatype = l.attribute("Datatype").map_or(Ok(0), |v| v.trim().parse()).map_err(|_| "stackup XML: Datatype is not an integer")?;
                let mut p = PdkLayer::new(l.attribute("Name").unwrap_or_default(), gds, datatype, z_min, z_max - z_min, &mat.color, ty);
                p.er = mat.er;
                p.tand = mat.tand;
                p.sigma = mat.sigma;
                p.material = mname.map(str::to_string);
                layers.push(p);
            }
        }
        Ok(Stack::new(name.into(), layers, dielectrics, materials))
    }

    pub fn by_name(&self, name: &str) -> Result<&PdkLayer, String> {
        self.layers.iter().find(|l| l.name == name).ok_or_else(|| {
            let names: Vec<&str> = self.layers.iter().map(|l| l.name.as_str()).collect();
            format!("layer {name:?} not in stack {:?}; available: {names:?}", self.name)
        })
    }

    /// The layer on GDS (number, datatype).
    pub fn by_gds(&self, gds: i32, datatype: i32) -> Option<&PdkLayer> {
        self.layers.iter().find(|l| l.gds == gds && l.datatype == datatype)
    }

    pub fn material(&self, name: &str) -> Option<&StackMaterial> {
        self.materials.iter().find(|m| m.name == name)
    }

    /// A layer's material record, synthesised from the layer's own scalars
    /// when the stack's table lacks it.
    pub fn layer_material(&self, layer: &PdkLayer) -> StackMaterial {
        if let Some(m) = layer.material.as_deref().and_then(|n| self.material(n)) {
            return m.clone();
        }
        let kind = if layer.r#type == "metal" || layer.r#type == "via" { "conductor" } else { "dielectric" };
        StackMaterial {
            name: layer.material.clone().unwrap_or_else(|| layer.name.clone()),
            kind: kind.into(),
            er: layer.er,
            tand: layer.tand,
            sigma: layer.sigma,
            color: layer.color.clone(),
        }
    }

    /// A background slab's material record (vacuum when the table lacks it).
    pub fn slab_material(&self, d: &DielectricLayer) -> StackMaterial {
        self.material(&d.material).cloned().unwrap_or_else(|| {
            StackMaterial::new(if d.material.is_empty() { &d.name } else { &d.material }, "dielectric")
        })
    }

    /// The background slab containing height `z` (bottom inclusive).
    pub fn dielectric_at(&self, z: f64) -> Option<&DielectricLayer> {
        self.dielectrics.iter().find(|d| d.z <= z && z < d.z_top())
    }

    pub fn top_z(&self) -> f64 {
        self.layers.iter().map(PdkLayer::z_top).reduce(f64::max).unwrap_or(0.0)
    }

    /// z of the lowest patterned layer.
    pub fn bottom_z(&self) -> f64 {
        self.layers.iter().map(|l| l.z).reduce(f64::min).unwrap_or(0.0)
    }

    /// The rapidpassives single-slab view of the background: the
    /// semiconductor slabs as one substrate (total thickness, εr and σ of the
    /// thickest), the thickest non-air dielectric as the oxide.
    pub fn slab_summary(&self) -> Map<String, Value> {
        let thickest = |kind: &str, dielectric: bool| {
            self.dielectrics
                .iter()
                .filter(|d| {
                    let m = self.slab_material(d);
                    m.kind == kind && (!dielectric || m.er > 1.0)
                })
                .collect::<Vec<_>>()
        };
        let pick = |ds: &[&DielectricLayer]| {
            let mut best = ds[0];
            for d in ds {
                if d.thickness > best.thickness {
                    best = d;
                }
            }
            self.slab_material(best)
        };
        let mut out = Map::new();
        let semis = thickest("semiconductor", false);
        if !semis.is_empty() {
            let main = pick(&semis);
            let total: f64 = semis.iter().map(|d| d.thickness).sum();
            out.insert("substrate".into(), json!({"thickness_um": total / UM, "er": main.er, "sigma": main.sigma}));
        }
        let oxides = thickest("dielectric", true);
        if !oxides.is_empty() {
            let main = pick(&oxides);
            out.insert("oxide".into(), json!({"er": main.er, "tand": main.tand}));
        }
        out
    }

    /// The rapidpassives `Pdk` JSON (lengths in microns), plus the full
    /// background (`dielectrics`, `materials`) rapidfem round-trips.
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.name.to_lowercase()));
        out.insert("name".into(), json!(self.name));
        out.insert("description".into(), json!(format!("rapidfem stack: {}", self.name)));
        out.extend(self.slab_summary());
        let layers: Vec<Value> = self
            .layers
            .iter()
            .map(|l| {
                let mut m = json!({
                    "name": l.name, "gds": l.gds, "datatype": l.datatype,
                    "z_um": l.z / UM, "thickness_um": l.thickness / UM,
                    "color": l.color, "type": l.r#type,
                    "er": l.er, "ur": l.ur, "tand": l.tand, "sigma": l.sigma,
                });
                if let Some(mat) = &l.material {
                    m["material"] = json!(mat);
                }
                m
            })
            .collect();
        out.insert("layers".into(), Value::Array(layers));
        let dielectrics: Vec<Value> = self
            .dielectrics
            .iter()
            .map(|d| json!({"name": d.name, "material": d.material, "z_um": d.z / UM, "thickness_um": d.thickness / UM}))
            .collect();
        out.insert("dielectrics".into(), Value::Array(dielectrics));
        let mut mats = Map::new();
        for m in &self.materials {
            mats.insert(m.name.clone(), json!({"kind": m.kind, "er": m.er, "tand": m.tand, "sigma": m.sigma, "color": m.color}));
        }
        out.insert("materials".into(), Value::Object(mats));
        Value::Object(out)
    }

    /// The inverse of [`Stack::to_json`]; a plain rapidpassives `Pdk`
    /// (no `dielectrics`) gets its one substrate and one oxide slab.
    pub fn from_json(d: &Value) -> Result<Stack, String> {
        let f = |v: &Value, key: &str, default: f64| v.get(key).and_then(Value::as_f64).unwrap_or(default);
        let s = |v: &Value, key: &str, default: &str| v.get(key).and_then(Value::as_str).unwrap_or(default).to_string();
        let name = req(d, "name")?.as_str().ok_or("stack JSON: name is not a string")?.to_string();
        let mut layers = Vec::new();
        for l in req(d, "layers")?.as_array().ok_or("stack JSON: layers is not a list")? {
            let int = |key: &str| req(l, key).and_then(|v| v.as_i64().map(|i| i as i32).ok_or_else(|| format!("stack JSON: {key} is not an integer")));
            let num = |key: &str| req(l, key).and_then(|v| v.as_f64().ok_or_else(|| format!("stack JSON: {key} is not a number")));
            let mut p = PdkLayer::new(&s(l, "name", ""), int("gds")?, int("datatype")?, num("z_um")? * UM, num("thickness_um")? * UM, &s(l, "color", "#888"), &s(l, "type", "metal"));
            p.er = f(l, "er", 1.0);
            p.ur = f(l, "ur", 1.0);
            p.tand = f(l, "tand", 0.0);
            p.sigma = f(l, "sigma", 0.0);
            p.material = l.get("material").and_then(Value::as_str).map(str::to_string);
            layers.push(p);
        }
        let dielectrics: Vec<DielectricLayer> = d
            .get("dielectrics")
            .and_then(Value::as_array)
            .map(|ds| {
                ds.iter()
                    .map(|dd| DielectricLayer {
                        name: s(dd, "name", ""),
                        material: s(dd, "material", ""),
                        z: f(dd, "z_um", 0.0) * UM,
                        thickness: f(dd, "thickness_um", 0.0) * UM,
                    })
                    .collect()
            })
            .unwrap_or_default();
        if dielectrics.is_empty() {
            let null = Value::Null;
            let sub = d.get("substrate").unwrap_or(&null);
            let ox = d.get("oxide").unwrap_or(&null);
            return Ok(Stack::single_slab(name, layers, f(sub, "thickness_um", 300.0) * UM, f(sub, "er", 11.9), f(sub, "sigma", 10.0), f(ox, "er", 4.2), f(ox, "tand", 0.0)));
        }
        let materials = d
            .get("materials")
            .and_then(Value::as_object)
            .map(|ms| {
                ms.iter()
                    .map(|(n, mm)| StackMaterial {
                        name: n.clone(),
                        kind: s(mm, "kind", "dielectric"),
                        er: f(mm, "er", 1.0),
                        tand: f(mm, "tand", 0.0),
                        sigma: f(mm, "sigma", 0.0),
                        color: s(mm, "color", "#888"),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Stack::new(name, layers, dielectrics, materials))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sg13g2_parses_with_its_background() {
        let s = Stack::sg13g2();
        assert_eq!(s.name, "SG13G2");
        assert!(s.by_name("SUBGND").unwrap().is_pec());
        assert_eq!(s.by_name("TopVia2").unwrap().sigma, 3.143e6);
        assert_eq!(s.dielectric_at(-UM).unwrap().name, "EPI");
        let names: Vec<&str> = s.dielectrics.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["Substrate", "EPI", "SiO2", "Passive", "AIR"]);
    }

    #[test]
    fn json_round_trip_is_exact() {
        for s in [Stack::sky130(), Stack::sg13g2()] {
            let back = Stack::from_json(&s.to_json()).unwrap();
            assert_eq!(back.to_json(), s.to_json());
        }
    }
}
