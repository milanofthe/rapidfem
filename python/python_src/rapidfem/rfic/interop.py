# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""FEM-JSON bridge, consume rapidpassives' exportForFEM() JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rapidfem import _native
from rapidfem._native import FEM_JSON_SCHEMA_VERSIONS

if TYPE_CHECKING:
    from rapidfem.geometry import Geometry, GeoObject


@dataclass
class FemLayoutResult:
    """Output of :func:`from_fem_json`, a solve-ready geometry plus the
    objects it was built from.

    Typical usage::

        from rapidfem import rfic, ProblemFD
        layout = rfic.from_fem_json("spiral.fem.json")
        layout.geometry.mesh()
        result = ProblemFD(layout.geometry).sweep([1e9, 10e9, 50e9])
    """
    geometry: "Geometry"
    conductors: dict[str, list["GeoObject"]]   # stack-layer id -> conductor holes
    ports: dict[str, "GeoObject"]               # port name -> 2-D port plate
    ground_patches: list                        # local grounds of ports without one
    substrate: "GeoObject"
    oxide: "GeoObject"
    air: "GeoObject"
    doc: dict                                   # the parsed FEM-JSON (metadata + sim)


def from_fem_json(
    source,
    *,
    stack=None,
    via_mode: Literal["merged", "cells"] = "merged",
    footprint_margin: float = 0.3,
    air_height_um: float = 60.0,
    conductor_maxh_um: float = 3.0,
    port_maxh_um: float = 3.0,
    port_tab_um: float = 8.0,
    port_inset_um: float | None = None,
    port_z0: float = 50.0,
) -> FemLayoutResult:
    """Build a solve-ready FEM geometry from a rapidpassives ``exportForFEM`` JSON.

    Conductors (metals AND vias) are extruded to their stack-layer thickness
    and cut out of the mesh, their walls PEC, as :func:`rapidfem.rfic.build`
    does. Each JSON port becomes a vertical lumped port (``port_z0``) inset
    from its nominal location toward the layout centre, so the plate top edge
    lands on the conductor's bottom face, and reaching down to the topmost
    lower metal under it that is not the same net; a port with none gets a
    PEC ground patch on the lowest metal, shared by ports closer than four
    tab widths. The outer faces of the air box are absorbing.

    Parameters
    ----------
    source : str | pathlib.Path | dict
        Path to a ``.fem.json`` file or an already-parsed dict.
    stack : rfic.Stack, optional
        If given, replaces the substrate/oxide constants from the JSON's
        ``stack.substrate`` / ``stack.oxide`` block. The JSON's layer z-stack
        is always trusted (it carries the GDS-derived geometry).
    via_mode : {"merged", "cells"}
        "merged" (default) extrudes the merged bounding box of each via
        array, one conductor volume per array (fast). "cells" extrudes every
        individual via cell if the JSON provides ``polygon_cells``; falls
        back to the merged polygon when cells aren't present.
    footprint_margin : float
        Substrate/oxide/air enclosure margin as a fraction of the conductor
        bbox span. 0.3 = 30% on each side.
    air_height_um : float
        Air-box height above the stack top.
    conductor_maxh_um : float
        Per-volume mesh-size cap for every extruded conductor.
    port_maxh_um : float
        Per-face mesh-size cap for the port plates and ground patches.
    port_tab_um : float
        Port plate width (extent perpendicular to the integration line).
    port_inset_um : float, optional
        Distance to move each port plate inward from the JSON's port
        location (toward layout centre). Default ``port_tab_um / 2``.
    port_z0 : float
        Reference impedance of every port, in ohms.

    Returns
    -------
    FemLayoutResult
    """
    from rapidfem.geometry import Geometry

    if isinstance(source, (str, Path)):
        with open(source) as f:
            doc = json.load(f)
    elif isinstance(source, dict):
        doc = source
    else:
        raise TypeError(f"source must be str/Path/dict, got {type(source).__name__}")
    native, b = _native.rfic_from_fem_json(
        doc, stack=stack, via_mode=via_mode, footprint_margin=footprint_margin,
        air_height_um=air_height_um, conductor_maxh_um=conductor_maxh_um,
        port_maxh_um=port_maxh_um, port_tab_um=port_tab_um,
        port_inset_um=port_inset_um, port_z0=port_z0)
    g = Geometry._adopt(native)
    objs = g._objects
    return FemLayoutResult(
        geometry=g,
        conductors={k: objs(v) for k, v in b["conductors"].items()},
        ports={k: objs([v])[0] for k, v in b["ports"].items()},
        ground_patches=objs(b["ground_patches"]),
        substrate=objs([b["substrate"]])[0], oxide=objs([b["oxide"]])[0],
        air=objs([b["air"]])[0], doc=doc)


__all__ = ["from_fem_json", "FemLayoutResult", "FEM_JSON_SCHEMA_VERSIONS"]
