# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Bulk conductivity across a frequency sweep, sweep equals single solves.

εr*(ω) carries −j·σ/(ω·ε₀), so a material with bulk σ is frequency-dependent
even without Debye/Drude dispersion. The cached sweep path splits the σ term
into its own mass matrix scaled per frequency; this test pins that a
multi-frequency sweep returns the SAME S-parameters as solving each
frequency in its own one-point sweep (where the assembly frequency is
trivially correct). Regression for the frozen-σ bug where the whole sweep
reused εr*(f₀): there S11 at the top frequency was off by >10%.

The geometry is a small oxide-over-conductive-silicon parallel plate (the
Maxwell-Wagner relaxation case where the frozen σ was first observed).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

um = 1e-6
T_OX, ER_OX = 8.0 * um, 4.1
T_SI, SIGMA_SI = 12.0 * um, 5.0
ER_SI = 11.9
A = 30.0 * um

FREQS = np.array([1.0e9, 8.0e9, 30.0e9])


def _build(g):
    ox = g.box(A, A, T_OX, position=(0, 0, 0),
               material=rf.Dielectric(er=ER_OX, maxh=4 * um))
    si = g.box(A, A, T_SI, position=(0, 0, -T_SI),
               material=rf.Dielectric(er=ER_SI, conductivity=SIGMA_SI, maxh=4 * um))
    port = g.plate(p0=(10 * um, 15 * um, -T_SI),
                   width=(10 * um, 0, 0), height=(0, 0, T_SI + T_OX), maxh=3 * um)
    g.fragment(ox, si, port)
    rf.PEC(ox.faces.max(axis="z"), si.faces.min(axis="z"))
    rf.LumpedPort(port, direction=(0, 0, 1), z0=50.0)


@pytest.mark.slow
@case.phenomenon
def test_sigma_sweep_matches_single_frequency_solves():
    """S11 of a 3-point sweep matches three 1-point sweeps to solver noise."""
    g = case.geometry(maxh=6 * um)
    _build(g)
    _, res = case.sweep(g, FREQS)
    s11_sweep = res.sparams[:, 0, 0]

    s11_single = []
    for f in FREQS:
        g1 = case.geometry(maxh=6 * um)
        _build(g1)
        _, r1 = case.sweep(g1, np.array([f]))
        s11_single.append(r1.sparams[0, 0, 0])
    s11_single = np.array(s11_single)

    err = np.abs(s11_sweep - s11_single)
    assert float(err.max()) < 1e-6, (
        f"sweep vs single-frequency S11 mismatch: {err} "
        f"(sweep {s11_sweep}, single {s11_single})")
