# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Process-stack model, native: `PdkLayer` (patterned layers), `DielectricLayer`
(background slabs), `StackMaterial` (the shared materials table) and `Stack`.

Ingestion: ``Stack.from_xml`` (the gds2palace / ADS stackup XML IHP ships for
SG13G2), ``Stack.from_pdk("sky130" | "sg13g2")`` / ``Stack.sky130()`` /
``Stack.sg13g2()`` (presets, SG13G2 from the bundled IHP XML) and
``Stack.from_dict`` / ``stack.to_dict()`` (the rapidpassives `Pdk` JSON).
All lengths in metres; a layer's ``z`` is its bottom.
"""
from typing import Literal

from rapidfem._native import DielectricLayer, PdkLayer, Stack, StackMaterial

LayerType = Literal["metal", "via", "poly", "diffusion", "substrate", "oxide",
                    "dielectric", "other"]
MaterialKind = Literal["conductor", "dielectric", "semiconductor"]

__all__ = ["Stack", "PdkLayer", "DielectricLayer", "StackMaterial",
           "LayerType", "MaterialKind"]
