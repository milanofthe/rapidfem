# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Unit tests for the SurfaceImpedance ``two_sided`` semantics (no solve).

Pins the finite-thickness config boundary: one-sided sheets (ground planes)
must not emit the flag, shell sheets must, so the Rust side applies
coth(γt) vs coth(γt/2) accordingly. Also pins the shell heuristic that
warns when ``thickness`` is set on a complete solid shell without an
explicit ``two_sided`` choice.
"""
import warnings

import pytest

import rapidfem as rf


def _sibc(**kwargs):
    g = rf.Geometry()
    plate = g.xy_plate(1e-3, 1e-3, position=(0, 0, 0))
    return rf.SurfaceImpedance(plate, **kwargs)


def test_default_is_one_sided():
    toml = _sibc(conductivity=5.8e7, thickness=3e-6)._to_toml(tag=42)
    assert 'type = "surface_impedance"' in toml
    assert "thickness = " in toml
    assert "two_sided" not in toml


def test_two_sided_emits_flag():
    toml = _sibc(conductivity=5.8e7, thickness=3e-6, two_sided=True)._to_toml(tag=42)
    assert "thickness = " in toml
    assert "two_sided = true" in toml


def test_semi_infinite_has_no_thickness_terms():
    toml = _sibc(conductivity=5.8e7)._to_toml(tag=7)
    assert "thickness" not in toml
    assert "two_sided" not in toml


def test_shell_without_choice_warns():
    g = rf.Geometry()
    trace = g.box(1e-3, 1e-4, 3e-6, position=(0, 0, 0))
    with pytest.warns(UserWarning, match="two_sided"):
        rf.SurfaceImpedance(trace.faces, conductivity=3e7, thickness=3e-6)


@pytest.mark.parametrize("choice", [True, False])
def test_shell_with_explicit_choice_is_silent(choice):
    g = rf.Geometry()
    trace = g.box(1e-3, 1e-4, 3e-6, position=(0, 0, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rf.SurfaceImpedance(trace.faces, conductivity=3e7, thickness=3e-6,
                            two_sided=choice)


def test_single_boundary_face_is_silent():
    g = rf.Geometry()
    sub = g.box(1e-3, 1e-3, 1e-4, position=(0, 0, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rf.SurfaceImpedance(sub.faces.min(axis="z"), conductivity=2e7,
                            thickness=4.2e-7)
