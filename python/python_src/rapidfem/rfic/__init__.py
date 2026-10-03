# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""
RFIC builder for rapidfem: PDK-grade stack definitions, the GDS-driven
model builder and the rapidpassives bridge.

Submodules:

- :mod:`rapidfem.rfic.stack`: `Stack` / `PdkLayer` process-stack model,
  mirrors the rapidpassives `Pdk` JSON schema
- :mod:`rapidfem.rfic.build`: ``build``, GDS + stack + ports to a
  solve-ready model
- :mod:`rapidfem.rfic.interop`: ``from_fem_json`` bridge consuming
  rapidpassives' ``exportForFEM()`` JSON

Typical workflow::

    import rapidfem as rf
    import rapidfem.rfic as rfic

    stack = rfic.Stack.sg13g2()                       # PDK preset
    model = rfic.build("inductor.gds", stack,         # GDS, stack and ports in,
                       ports=[...], band=(1e9, 20e9)) # solve-ready model out
    model.geometry.mesh()
    result = rf.ProblemFD(model.geometry).sweep([1e9, 5e9, 10e9])
"""
from .stack import (
    Stack, PdkLayer, DielectricLayer, StackMaterial, LayerType, MaterialKind,
)
from .interop import from_fem_json, FemLayoutResult, FEM_JSON_SCHEMA_VERSIONS
from .build import build, BuiltModel, MeshSpec, ViaPort

__all__ = [
    "Stack", "PdkLayer", "DielectricLayer", "StackMaterial",
    "LayerType", "MaterialKind",
    "build", "BuiltModel", "MeshSpec", "ViaPort",
    "from_fem_json", "FemLayoutResult",
]
