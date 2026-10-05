# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Adaptive sweep (issue #67) against the full sweep, and the dispersive
materials it and the full sweep take as frequency-scaled mass matrices.

The geometry is a WR-90 section with a Debye slab between two air sections,
fed by TE10 ports: dispersion, a wave port and a frequency-dependent Robin
term in one small model."""
import numpy as np
import pytest

import rapidfem as rf

A, B = 22.86e-3, 10.16e-3
L_FEED, L_SLAB = 8e-3, 10e-3
DEBYE = dict(er_inf=2.0, er_static=6.0, tau_s=2e-11)
FREQS = np.linspace(8e9, 12e9, 41)


def _guide(slab_material):
    g = rf.Geometry(maxh=4e-3)
    inp = g.box(A, B, L_FEED, position=(-A / 2, -B / 2, 0.0), material=rf.Air())
    slab = g.box(A, B, L_SLAB, position=(-A / 2, -B / 2, L_FEED), material=slab_material)
    out = g.box(A, B, L_FEED, position=(-A / 2, -B / 2, L_FEED + L_SLAB), material=rf.Air())
    g.fragment(slab, inp, out)
    rf.RectWaveguidePort(inp.faces.min(axis="z"))
    rf.RectWaveguidePort(out.faces.max(axis="z"))
    g.mesh()
    return g


def _debye_eps(f):
    w = 2 * np.pi * f
    return DEBYE["er_inf"] + (DEBYE["er_static"] - DEBYE["er_inf"]) / (1 + 1j * w * DEBYE["tau_s"])


def test_dispersive_material_equals_its_constant_value_at_each_frequency():
    """A Debye slab swept at three frequencies gives the S-parameters of a
    constant dielectric with the slab's εr(f) at each one: the per-frequency
    scaling of its mass matrix is the material evaluated there."""
    fs = np.array([8e9, 10e9, 12e9])
    res = rf.ProblemFD(_guide(rf.Material(debye=rf.Debye(**DEBYE)))).sweep(fs)
    for i, f in enumerate(fs):
        eps = _debye_eps(f)
        const = rf.Dielectric(er=eps.real, tand=-eps.imag / eps.real)
        ref = rf.ProblemFD(_guide(const)).sweep([f])
        np.testing.assert_allclose(res.sparams[i], ref.sparams[0], atol=1e-9)


@pytest.mark.parametrize("dispersive", [False, True])
def test_adaptive_sweep_matches_the_full_sweep(dispersive):
    material = rf.Material(debye=rf.Debye(**DEBYE)) if dispersive else rf.Dielectric(er=4.0, tand=0.01)
    g = _guide(material)
    full = rf.ProblemFD(g).sweep(FREQS)
    adaptive = rf.ProblemFD(g).sweep(FREQS, adaptive_tol=1e-6)
    d = np.abs(np.asarray(full.sparams) - np.asarray(adaptive.sparams)).max()
    assert d < 1e-4, d
    np.testing.assert_array_equal(full.full_solve_frequencies, FREQS)
    samples = adaptive.full_solve_frequencies
    assert samples[0] == FREQS[0] and samples[1] == FREQS[-1]
    assert 2 < len(samples) < len(FREQS) // 2


def test_adaptive_sweep_settings_are_checked():
    g = _guide(rf.Dielectric(er=4.0))
    with pytest.raises(RuntimeError, match="tol > 0"):
        rf.ProblemFD(g).sweep(FREQS, adaptive_tol=-1.0)
