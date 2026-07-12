# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Unit tests for the SurfaceImpedance TOML emission (geometry build, no solve).

Pins the ``two_sided`` finite-thickness semantics at the config boundary:
one-sided sheets (ground planes) must not emit the flag, shell sheets must,
so the Rust side applies coth(γt) vs coth(γt/2) accordingly.
"""
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
