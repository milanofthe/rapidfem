# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Unit tests for the SurfaceImpedance face topologies (no solve).

Pins what reaches the native model: boundary faces carry neither flag
(coth(γt)), conductor walls carry ``two_sided`` (coth(γt/2)), embedded sheets
carry ``sheet`` (coth(γt/2)/2). Also pins the warning for a BC on the complete
shell of a solid that is still meshed (issue #46).
"""
import warnings

import pytest

import rapidfem as rf
from rapidfem._native import Model


def _sibc(**kwargs):
    g = rf.Geometry()
    plate = g.xy_plate(1e-3, 1e-3, position=(0, 0, 0))
    return rf.SurfaceImpedance(plate, **kwargs)


def _native(sibc, tag=42) -> str:
    """The model entry of `sibc` under `tag`, as the native repr."""
    model = Model()
    sibc._add_to(model, tag)
    return repr(model)


def test_default_is_one_sided():
    entry = _native(_sibc(conductivity=5.8e7, thickness=3e-6), tag=42)
    assert "SurfaceImpedance" in entry
    assert "thickness: Some(" in entry
    assert "two_sided: false" in entry and "sheet: false" in entry


def test_two_sided_emits_flag():
    entry = _native(_sibc(conductivity=5.8e7, thickness=3e-6, two_sided=True), tag=42)
    assert "thickness: Some(" in entry
    assert "two_sided: true" in entry


def test_semi_infinite_has_no_thickness_terms():
    entry = _native(_sibc(conductivity=5.8e7), tag=7)
    assert "thickness: None" in entry
    assert "two_sided: false" in entry


def test_sheet_emits_flag():
    entry = _native(_sibc(conductivity=3e7, thickness=1e-6, sheet=True), tag=42)
    assert "sheet: true" in entry
    assert "two_sided: false" in entry


def test_sheet_and_two_sided_exclude_each_other():
    with pytest.raises(ValueError, match="exclude"):
        _sibc(conductivity=3e7, thickness=1e-6, sheet=True, two_sided=True)


@pytest.mark.parametrize("choice", [None, True, False])
def test_shell_of_meshed_solid_warns(choice):
    g = rf.Geometry()
    trace = g.box(1e-3, 1e-4, 3e-6, position=(0, 0, 0))
    with pytest.warns(UserWarning, match="still meshed"):
        rf.SurfaceImpedance(trace.faces, conductivity=3e7, thickness=3e-6,
                            two_sided=choice)


def test_single_boundary_face_is_silent():
    g = rf.Geometry()
    sub = g.box(1e-3, 1e-3, 1e-4, position=(0, 0, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rf.SurfaceImpedance(sub.faces.min(axis="z"), conductivity=2e7,
                            thickness=4.2e-7)
