# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""
RFIC builder for rapidfem, PDK-grade stack definitions, GDS-driven extrusion
helpers, and hand-coded primitives (microstrip, via, GSG port).

Submodules:

- :mod:`rapidfem.rfic.stack` — `Stack` / `PdkLayer` process-stack model,
  mirrors the rapidpassives `Pdk` JSON schema
- :mod:`rapidfem.rfic.primitives` — hand-coded layout primitives
  (``microstrip``, ``via``, ``gsg_port``, ``differential_port``)
- :mod:`rapidfem.rfic.interop` — ``from_fem_json`` bridge consuming
  rapidpassives' ``exportForFEM()`` JSON

Typical workflow::

    import rapidfem as rf
    import rapidfem.rfic as rfic

    stack = rfic.Stack.sky130()                       # PDK preset
    g = rf.Geometry.from_gds(                         # GDS-driven extrusion
        "inductor.gds", stack=stack, top_cell="ind_3turn",
    )
    subs = stack.create_substrate(g, footprint=(400e-6, 400e-6))
    air = g.box(400e-6, 400e-6, 200e-6,               # ABC enclosure
                position=(-200e-6, -200e-6, stack.top_z),
                material=rf.Air())
    rf.PEC(...)                                       # trace + ground BCs
    rf.LumpedPort(...)
    g.mesh()
    result = rf.Problem(g).sweep([1e9, 5e9, 10e9])
"""
from .stack import (
    Stack, PdkLayer, DielectricLayer, StackMaterial, LayerType, MaterialKind,
)
from .primitives import (
    microstrip, via, trace_port, gsg_port, differential_port,
    TracePort, GsgPort, DifferentialPort,
)
from .interop import from_fem_json, FemLayoutResult, FEM_JSON_SCHEMA_VERSIONS
from .build import build, BuiltModel, MeshSpec, ViaPort

__all__ = [
    "Stack", "PdkLayer", "DielectricLayer", "StackMaterial",
    "LayerType", "MaterialKind",
    "build", "BuiltModel", "MeshSpec", "ViaPort",
    "microstrip", "via", "trace_port", "gsg_port", "differential_port",
    "TracePort", "GsgPort", "DifferentialPort",
    "from_fem_json", "FemLayoutResult",
]
