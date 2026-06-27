# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Unit tests for S-parameter renormalization (pure math, no solver).

Pins `rapidfem.io.renormalize_sparams` to the textbook reference-impedance
transform: identity when the reference is unchanged, the exact 1-port load
reflection under a reference change, and preservation of reciprocity.
"""
import numpy as np

from rapidfem import io


def test_identity_when_reference_unchanged():
    s = np.array([[[0.2 + 0.1j, 0.9 - 0.05j],
                   [0.9 - 0.05j, 0.1 - 0.2j]]])
    out = io.renormalize_sparams(s, np.array([[75.0, 75.0]]), 75.0)
    assert np.allclose(out, s, atol=1e-12)


def test_one_port_load_reflection_renormalizes_exactly():
    # A physical load Z_L sets Γ = (Z_L − Zref)/(Z_L + Zref). Renormalizing the
    # reflection from z_old to z_new must reproduce the closed-form Γ_new.
    z_l = 100.0
    z_old, z_new = 30.0, 50.0
    g_old = (z_l - z_old) / (z_l + z_old)
    s = np.array([[[g_old + 0j]]])
    out = io.renormalize_sparams(s, np.array([[z_old]]), z_new)
    g_new = (z_l - z_new) / (z_l + z_new)
    assert abs(out[0, 0, 0] - g_new) < 1e-12


def test_lumped_reference_is_noop():
    # Lumped ports already sit at a fixed z0; renormalizing 50 → 50 is identity.
    s = np.array([[[0.33 + 0.0j, 0.94 + 0.0j],
                   [0.94 + 0.0j, 0.33 + 0.0j]]])
    out = io.renormalize_sparams(s, np.array([[50.0, 50.0]]), 50.0)
    assert np.allclose(out, s, atol=1e-12)


def test_reciprocity_preserved():
    # A reciprocal S (S = S^T) referenced to modal impedances stays reciprocal
    # after renormalization to a common reference.
    s = np.array([[[0.2 + 0.3j, 0.7 - 0.1j],
                   [0.7 - 0.1j, -0.1 + 0.25j]]])
    out = io.renormalize_sparams(s, np.array([[377.0, 377.0]]), 50.0)
    assert np.allclose(out[0], out[0].T, atol=1e-12)


def test_modal_match_becomes_mismatch_against_50():
    # A 1-port "matched to its modal impedance" (S11=0 at z_old=Z_mode) is in
    # fact a Z_mode load; against 50 Ω it must show the Z_mode/50 mismatch.
    z_mode = 400.0
    s = np.array([[[0.0 + 0.0j]]])  # matched to the modal reference
    out = io.renormalize_sparams(s, np.array([[z_mode]]), 50.0)
    expected = (z_mode - 50.0) / (z_mode + 50.0)
    assert abs(out[0, 0, 0] - expected) < 1e-12
