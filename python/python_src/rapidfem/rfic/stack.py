# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Process-stack data model: `PdkLayer` + `DielectricLayer` + `Stack`.

The `Stack` describes the complete vertical cross-section of a process:

- **patterned layers** (`PdkLayer`): anything drawn in the GDS, metals,
  vias, patterned dielectric cuts (e.g. backside etch)
- **background dielectrics** (`DielectricLayer`): the unpatterned slabs the
  patterned layers are embedded in, substrate, EPI, oxide zones,
  passivation, air
- **materials** (`StackMaterial`): a shared name → properties table, so
  layers reference materials the way process specs do

Ingestion paths:

- ``Stack.from_xml("SG13G2.xml")`` parses the gds2palace / ADS-style
  stackup XML (the format IHP ships for SG13G2)
- ``Stack.from_pdk("sky130" | "sg13g2")`` returns built-in presets;
  sg13g2 is backed by the bundled IHP XML
- ``Stack.from_dict`` / ``stack.to_dict()`` round-trips the rapidpassives
  `Pdk` JSON shape (viewer interop)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rapidfem.geometry import Geometry, GeoObject


LayerType = Literal["metal", "via", "poly", "diffusion", "substrate", "oxide",
                    "dielectric", "other"]

MaterialKind = Literal["conductor", "dielectric", "semiconductor"]


@dataclass
class StackMaterial:
    """One named material of the process, referenced by layers.

    ``sigma >= _PEC_SIGMA`` (the gds2palace "LOWLOSS" convention, 1e10 S/m)
    marks an idealised lossless conductor; builders should treat it as PEC.
    """
    name: str
    kind: MaterialKind = "dielectric"
    er: float = 1.0
    tand: float = 0.0
    sigma: float = 0.0       # S/m
    color: str = "#888"


_PEC_SIGMA = 1e10   # at/above this a conductor is "lossless" -> model as PEC


@dataclass
class DielectricLayer:
    """One unpatterned background slab (substrate, oxide, passivation, air).

    Ordered bottom-to-top in ``Stack.dielectrics``; z is the slab's BOTTOM
    in meters, the top is at z + thickness.
    """
    name: str
    material: str            # name in Stack.materials
    z: float                 # bottom z, meters
    thickness: float         # meters

    @property
    def z_top(self) -> float:
        return self.z + self.thickness


@dataclass
class PdkLayer:
    """One patterned (GDS-drawn) layer in a process stack.

    Mirrors ``rapidpassives.web.lib.stack.pdk.PdkLayer`` so the same JSON
    describes a stack on both rapidfem (FEM-side) and rapidpassives
    (viewer-side).

    Coordinate convention: z is the BOTTOM of the layer (lower face). The top
    is at z + thickness. All distances in meters (rapidpassives uses microns;
    converters below help round-trip).
    """
    name: str
    gds: int
    datatype: int
    z: float                # bottom z, in meters
    thickness: float        # in meters
    color: str = "#888"
    type: LayerType = "metal"
    # Material defaults (FEM-relevant; rapidpassives ignores these)
    er: float = 1.0
    ur: float = 1.0
    tand: float = 0.0
    sigma: float = 0.0       # bulk conductivity (S/m); 1e10 and above is PEC (is_pec)
    material: str | None = None   # optional name in Stack.materials

    @property
    def z_top(self) -> float:
        return self.z + self.thickness

    @property
    def gds_key(self) -> tuple[int, int]:
        return (self.gds, self.datatype)

    @property
    def is_pec(self) -> bool:
        """Idealised lossless conductor (gds2palace LOWLOSS convention)."""
        return self.sigma >= _PEC_SIGMA


@dataclass
class Stack:
    """A complete process stack: patterned layers, background dielectrics,
    and a shared materials table.

    Use ``Stack.from_xml(...)`` for gds2palace/ADS stackup files,
    ``Stack.sky130()`` / ``Stack.sg13g2()`` for built-in presets, or
    construct manually from a list of `PdkLayer`s.

    The background ``dielectrics`` (bottom to top, materials in
    ``materials``) are the whole vertical cross-section around the
    patterned layers.
    """
    name: str
    layers: list[PdkLayer]
    # Background dielectric slabs, bottom-to-top.
    dielectrics: list[DielectricLayer] = field(default_factory=list)
    materials: dict[str, StackMaterial] = field(default_factory=dict)

    # ── Construction helpers ───────────────────────────────────────────────

    def __post_init__(self):
        # Sort layers bottom-to-top by z for deterministic iteration
        self.layers = sorted(self.layers, key=lambda l: (l.z, l.thickness))
        self.dielectrics = sorted(self.dielectrics, key=lambda d: d.z)

    def _single_slab(self) -> dict:
        """The rapidpassives single-slab view of the background: the
        semiconductor slabs as one substrate (total thickness, er and sigma
        of the thickest), the thickest non-air dielectric as the oxide."""
        um = 1e-6
        semis = [d for d in self.dielectrics if self.material_of(d).kind == "semiconductor"]
        oxides = [d for d in self.dielectrics
                  if self.material_of(d).kind == "dielectric" and self.material_of(d).er > 1.0]
        out: dict = {}
        if semis:
            main = self.material_of(max(semis, key=lambda d: d.thickness))
            out["substrate"] = {"thickness_um": sum(d.thickness for d in semis) / um,
                                "er": main.er, "sigma": main.sigma}
        if oxides:
            main = self.material_of(max(oxides, key=lambda d: d.thickness))
            out["oxide"] = {"er": main.er, "tand": main.tand}
        return out

    @staticmethod
    def from_pdk(name: str) -> "Stack":
        """Convenience: dispatch by lowercased PDK name."""
        normalized = name.lower().replace("-", "").replace("_", "")
        if normalized in ("sky130", "skywater130"):
            return Stack.sky130()
        if normalized in ("sg13g2", "ihpsg13g2"):
            return Stack.sg13g2()
        raise ValueError(f"unknown PDK {name!r}; available: sky130, sg13g2")

    @staticmethod
    def from_xml(source, *, name: str | None = None) -> "Stack":
        """Parse a gds2palace / ADS-style stackup XML into a full Stack.

        The format (``<Stackup schemaVersion="2.0">``) carries a
        ``<Materials>`` table, a top-down ``<Dielectrics>`` background
        stack anchored by ``<Substrate Offset=...>`` (depth of the stack
        bottom below z=0), and ``<Layers>`` with GDS number, z-range and a
        material reference. This is the stackup file IHP publishes for
        SG13G2 and the input gds2palace consumes, so a gds2palace user's
        XML drops in unchanged.

        Parameters
        ----------
        source : str | Path
            Path to the XML file (or a string containing XML markup,
            detected by a leading ``<``).
        name : str, optional
            Stack name; defaults to the XML file stem or "stack".
        """
        import xml.etree.ElementTree as ET
        from pathlib import Path

        if isinstance(source, str) and source.lstrip().startswith("<"):
            root = ET.fromstring(source)
            stack_name = name or "stack"
        else:
            root = ET.parse(source).getroot()
            stack_name = name or Path(source).stem

        if root.tag != "Stackup":
            raise ValueError(f"not a stackup XML: root element is {root.tag!r}")

        # ── materials table ────────────────────────────────────────────────
        materials: dict[str, StackMaterial] = {}
        for m in root.iter("Material"):
            mname = m.get("Name")
            materials[mname] = StackMaterial(
                name=mname,
                kind=m.get("Type", "dielectric").lower(),
                er=float(m.get("Permittivity", 1.0)),
                tand=float(m.get("DielectricLossTangent", 0.0)),
                sigma=float(m.get("Conductivity", 0.0)),
                color="#" + m.get("Color", "888888").lstrip("#"),
            )

        elayers = root.find("ELayers")
        if elayers is None:
            raise ValueError("stackup XML has no <ELayers> element")
        unit = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}[
            elayers.get("LengthUnit", "um").lower()]

        # ── background dielectrics, listed top-down, anchored by Offset ────
        # <Substrate Offset="d"/> puts the bottom of the lowest slab at
        # z = -d; slabs then stack upward in reverse file order.
        layers_el = elayers.find("Layers")
        offset = 0.0
        if layers_el is not None:
            sub_el = layers_el.find("Substrate")
            if sub_el is not None:
                offset = float(sub_el.get("Offset", 0.0))

        dielectrics: list[DielectricLayer] = []
        dielectrics_el = elayers.find("Dielectrics")
        if dielectrics_el is not None:
            z = -offset * unit
            for d in reversed(list(dielectrics_el.iter("Dielectric"))):
                t = float(d.get("Thickness", 0.0)) * unit
                dielectrics.append(DielectricLayer(
                    name=d.get("Name"), material=d.get("Material"),
                    z=z, thickness=t,
                ))
                z += t

        # ── patterned layers ───────────────────────────────────────────────
        type_map = {"conductor": "metal", "via": "via", "dielectric": "dielectric"}
        layers: list[PdkLayer] = []
        if layers_el is not None:
            for l in layers_el.iter("Layer"):
                z_min = float(l.get("Zmin")) * unit
                z_max = float(l.get("Zmax")) * unit
                mname = l.get("Material")
                mat = materials.get(mname, StackMaterial(mname or "unknown"))
                layers.append(PdkLayer(
                    name=l.get("Name"),
                    gds=int(l.get("Layer")),
                    datatype=int(l.get("Datatype", 0)),
                    z=z_min, thickness=z_max - z_min,
                    color=mat.color,
                    type=type_map.get(l.get("Type", "conductor").lower(), "other"),
                    er=mat.er, tand=mat.tand, sigma=mat.sigma,
                    material=mname,
                ))

        return Stack(name=stack_name, layers=layers,
                     dielectrics=dielectrics, materials=materials)

    @staticmethod
    def sky130() -> "Stack":
        """SkyWater SKY130 open-source PDK process stack.

        Layer numbering and z-positions match the rapidpassives `pdk.ts` file
        (z=0 at the bottom of li1, polysilicon below at z=-0.18 μm).
        """
        um = 1e-6
        layers = [
            # name        gds  dt   z(μm)  t(μm)   color    type      er    sigma
            PdkLayer("poly",   66, 20, -0.18 * um, 0.18 * um, "#c4725e", "poly",     er=4.2),
            PdkLayer("licon1", 66, 44, -0.10 * um, 0.10 * um, "#5a5a62", "via",      sigma=4.1e7),
            PdkLayer("li1",    67, 20,  0.00 * um, 0.10 * um, "#7b5e8a", "metal",    sigma=4.1e7),
            PdkLayer("mcon",   67, 44,  0.10 * um, 0.27 * um, "#5a5a62", "via",      sigma=4.1e7),
            PdkLayer("met1",   68, 20,  0.37 * um, 0.36 * um, "#6bbf8a", "metal",    sigma=4.1e7),
            PdkLayer("via",    68, 44,  0.73 * um, 0.27 * um, "#5a5a62", "via",      sigma=4.1e7),
            PdkLayer("met2",   69, 20,  1.00 * um, 0.36 * um, "#4a9ec2", "metal",    sigma=4.1e7),
            PdkLayer("via2",   69, 44,  1.36 * um, 0.42 * um, "#6e6e78", "via",      sigma=4.1e7),
            PdkLayer("met3",   70, 20,  1.78 * um, 0.845 * um, "#5aad78", "metal",   sigma=4.1e7),
            PdkLayer("via3",   70, 44,  2.625 * um, 0.39 * um, "#6e6e78", "via",     sigma=4.1e7),
            PdkLayer("met4",   71, 20,  3.015 * um, 0.845 * um, "#d9513c", "metal",  sigma=4.1e7),
            PdkLayer("via4",   71, 44,  3.86 * um, 0.505 * um, "#7a7a84", "via",     sigma=4.1e7),
            PdkLayer("met5",   72, 20,  4.365 * um, 1.26 * um, "#e8944a", "metal",   sigma=4.1e7),
        ]
        return Stack._with_single_slab(
            "SKY130", layers, substrate_thickness=300 * um, substrate_er=11.9,
            substrate_sigma=10.0, oxide_er=4.2, oxide_tand=0.0)

    @staticmethod
    def _with_single_slab(name: str, layers: list[PdkLayer], *, substrate_thickness: float,
                          substrate_er: float, substrate_sigma: float, oxide_er: float,
                          oxide_tand: float) -> "Stack":
        """A stack whose background is one substrate slab below the lowest
        layer and one oxide slab up to the highest (the rapidpassives shape)."""
        bottom = min(l.z for l in layers)
        top = max(l.z_top for l in layers)
        materials = {
            "substrate": StackMaterial("substrate", "semiconductor", er=substrate_er,
                                       sigma=substrate_sigma),
            "oxide": StackMaterial("oxide", "dielectric", er=oxide_er, tand=oxide_tand),
        }
        dielectrics = [
            DielectricLayer("substrate", "substrate", z=bottom - substrate_thickness,
                            thickness=substrate_thickness),
            DielectricLayer("oxide", "oxide", z=bottom, thickness=top - bottom),
        ]
        return Stack(name=name, layers=layers, dielectrics=dielectrics, materials=materials)

    @staticmethod
    def sg13g2() -> "Stack":
        """IHP SG13G2 open-source 130 nm SiGe BiCMOS PDK process stack.

        Backed by the bundled ``ihp_sg13g2_200um.xml`` (the stackup file
        from the public gds2palace repository), so layer z-positions and
        per-layer conductivities are the process-spec values, not generic
        placeholders. 200 µm thinned-substrate variant with backside metal
        and LBE layers.
        """
        from importlib.resources import files
        xml_path = files("rapidfem.rfic") / "data" / "ihp_sg13g2_200um.xml"
        return Stack.from_xml(xml_path.read_text(), name="SG13G2")

    # ── Lookups ────────────────────────────────────────────────────────────

    def by_name(self, name: str) -> PdkLayer:
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError(f"layer {name!r} not in stack {self.name!r}; available: "
                       f"{[l.name for l in self.layers]}")

    def by_gds(self, gds: int, datatype: int = 0) -> PdkLayer | None:
        """Find the layer matching a GDS (number, datatype) tuple. None if absent."""
        for l in self.layers:
            if l.gds == gds and l.datatype == datatype:
                return l
        return None

    def metals(self) -> list[PdkLayer]:
        return [l for l in self.layers if l.type == "metal"]

    def vias(self) -> list[PdkLayer]:
        return [l for l in self.layers if l.type == "via"]

    def material_of(self, layer: "PdkLayer | DielectricLayer") -> StackMaterial:
        """Resolve a layer's material record; synthesises one from the
        layer's own scalars when the stack has no materials table."""
        mname = layer.material if layer.material else None
        if mname and mname in self.materials:
            return self.materials[mname]
        if isinstance(layer, PdkLayer):
            kind = "conductor" if layer.type in ("metal", "via") else "dielectric"
            return StackMaterial(name=mname or layer.name, kind=kind,
                                 er=layer.er, tand=layer.tand,
                                 sigma=layer.sigma, color=layer.color)
        return StackMaterial(name=mname or layer.name)

    def dielectric_at(self, z: float) -> DielectricLayer | None:
        """The background slab containing height z (bottom-inclusive)."""
        for d in self.dielectrics:
            if d.z <= z < d.z_top:
                return d
        return None

    @property
    def top_z(self) -> float:
        return max((l.z_top for l in self.layers), default=0.0)

    @property
    def bottom_z(self) -> float:
        """z of the lowest patterned layer."""
        return min((l.z for l in self.layers), default=0.0)

    # ── JSON interop with rapidpassives ────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to the rapidpassives `Pdk` JSON shape (lengths in microns)."""
        um = 1e-6
        return {
            "id": self.name.lower(),
            "name": self.name,
            "description": f"rapidfem stack: {self.name}",
            **self._single_slab(),
            "layers": [
                {
                    "name": l.name, "gds": l.gds, "datatype": l.datatype,
                    "z_um": l.z / um, "thickness_um": l.thickness / um,
                    "color": l.color, "type": l.type,
                    "er": l.er, "ur": l.ur, "tand": l.tand, "sigma": l.sigma,
                    **({"material": l.material} if l.material else {}),
                }
                for l in self.layers
            ],
            # Additive keys; rapidpassives ignores them, rapidfem round-trips
            # the full dielectric background through them.
            "dielectrics": [
                {
                    "name": d.name, "material": d.material,
                    "z_um": d.z / um, "thickness_um": d.thickness / um,
                }
                for d in self.dielectrics
            ],
            "materials": {
                m.name: {
                    "kind": m.kind, "er": m.er, "tand": m.tand,
                    "sigma": m.sigma, "color": m.color,
                }
                for m in self.materials.values()
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "Stack":
        um = 1e-6
        layers = [
            PdkLayer(
                name=l["name"], gds=l["gds"], datatype=l["datatype"],
                z=l["z_um"] * um, thickness=l["thickness_um"] * um,
                color=l.get("color", "#888"), type=l.get("type", "metal"),
                er=l.get("er", 1.0), ur=l.get("ur", 1.0),
                tand=l.get("tand", 0.0), sigma=l.get("sigma", 0.0),
                material=l.get("material"),
            )
            for l in d["layers"]
        ]
        dielectrics = [
            DielectricLayer(
                name=dd["name"], material=dd["material"],
                z=dd["z_um"] * um, thickness=dd["thickness_um"] * um,
            )
            for dd in d.get("dielectrics", [])
        ]
        materials = {
            mname: StackMaterial(
                name=mname, kind=mm.get("kind", "dielectric"),
                er=mm.get("er", 1.0), tand=mm.get("tand", 0.0),
                sigma=mm.get("sigma", 0.0), color=mm.get("color", "#888"),
            )
            for mname, mm in d.get("materials", {}).items()
        }
        if not dielectrics:
            # a rapidpassives Pdk: one substrate and one oxide slab
            sub, ox = d.get("substrate", {}), d.get("oxide", {})
            return Stack._with_single_slab(
                d["name"], layers,
                substrate_thickness=sub.get("thickness_um", 300) * um,
                substrate_er=sub.get("er", 11.9), substrate_sigma=sub.get("sigma", 10.0),
                oxide_er=ox.get("er", 4.2), oxide_tand=ox.get("tand", 0.0))
        return Stack(name=d["name"], layers=layers,
                     dielectrics=dielectrics, materials=materials)


__all__ = ["Stack", "PdkLayer", "DielectricLayer", "StackMaterial",
           "LayerType", "MaterialKind"]
