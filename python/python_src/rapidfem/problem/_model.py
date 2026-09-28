# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""The native model of a meshed geometry, shared by both backends."""
from __future__ import annotations

from .._native import Model
from ..materials import Material
from ..physics import PML


def build_model(geometry) -> Model:
    """Place every material and physics object of ``geometry`` on a native
    :class:`rapidfem._native.Model` under the mesh tags ``Geometry.mesh()``
    assigned.

    A volume targeted by a :class:`rapidfem.PML` gets no material entry: the
    layer carries its own ``er_base`` / ``ur_base``.
    """
    model = Model()
    pml_volumes = {id(e) for p in geometry._physics if isinstance(p, PML)
                   for e in p._entities}
    seen = set()
    for ent in geometry._entities:
        mat = ent.material
        if not isinstance(mat, Material) or ent.dim != 3 or id(ent) in pml_volumes:
            continue
        if id(mat) in seen:
            continue
        seen.add(id(mat))
        tag = geometry._material_tags.get(id(mat))
        if tag is None:
            raise RuntimeError(
                f"material {mat!r} has no tag, re-run g.mesh() after attaching it")
        mat._add_to(model, tag)
    for phys in geometry._physics:
        tag = geometry._physics_tags.get(id(phys))
        if tag is None:
            raise RuntimeError(
                f"physics object {phys!r} has no tag, re-run g.mesh() after "
                f"constructing it")
        phys._add_to(model, tag)
    return model
