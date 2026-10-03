# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Wire antennas with closed-form far fields: the near-to-far transform.

PHENOMENON: a short dipole radiates sin²θ with a directivity of 1.5
(1.76 dBi), a half-wave dipole cos²(π/2·cosθ)/sin²θ with 1.64 (2.15 dBi), a
quarter-wave monopole on an infinite ground plane the half-wave dipole's
upper half with twice its directivity (5.16 dBi). All three radiate the power
they accept, 1 - |S11|² of the 1 W incident. Each is a strip fed across
a lumped gap in an air box with a first-order ABC; the monopole's box rests on
its ground plane (the PEC box floor), which the transform takes as an
infinite image plane.

Reference: Balanis, *Antenna Theory*, ch. 4.
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

F = 1e9
LAM = 299_792_458.0 / F
MM = 1e-3
W = GAP = 4 * MM


def _strip(g, z0, length, maxh=W):
    return g.plate(p0=(-W / 2, 0, z0), width=(W, 0, 0), height=(0, 0, length), maxh=maxh)


def _solve(g):
    prob, res = case.sweep(g, np.array([F]))
    pat = prob.farfield(res, freq_idx=0, port_idx=0, n_theta=91, n_phi=36)
    return np.asarray(pat.theta_rad), np.asarray(pat.directivity_dbi), pat


def _dipole(length):
    g = case.geometry(maxh=LAM / 8)
    box = LAM
    air = g.box(box, box, 1.3 * box, position=(-box / 2, -box / 2, -0.65 * box),
                material=rf.Air())
    arm = (length - GAP) / 2
    rf.PEC(_strip(g, GAP / 2, arm), _strip(g, -GAP / 2 - arm, arm))
    rf.LumpedPort(_strip(g, -GAP / 2, GAP, maxh=W / 2), direction=(0, 0, 1), z0=73.0)
    rf.ABC(*air.faces.outer)
    return _solve(g)


def _pattern_error(theta, d_dbi, shape):
    """RMS difference of the φ-averaged linear pattern and `shape`, both
    scaled to peak 1."""
    d = (10 ** (d_dbi / 10)).mean(axis=0)
    ref = shape(theta)
    return float(np.sqrt(np.mean((d / d.max() - ref / ref.max()) ** 2)))


@pytest.mark.slow
@case.phenomenon
def test_short_dipole_directivity_and_pattern():
    theta, d, pat = _dipole(0.1 * LAM)
    assert pat.peak_directivity_dbi == pytest.approx(1.76, abs=0.2)
    assert d[:, 45].std() < 0.1, "not symmetric about the dipole axis"
    assert d[:, 0].max() < -30, "no null along the axis"
    assert _pattern_error(theta, d, lambda t: np.sin(t) ** 2) < 0.02


@pytest.mark.slow
@case.phenomenon
def test_half_wave_dipole_directivity_and_pattern():
    theta, d, pat = _dipole(0.47 * LAM)
    assert pat.peak_directivity_dbi == pytest.approx(2.15, abs=0.2)

    def dipole_of_length(t, kl2=np.pi * 0.47):
        # sinusoidal current on a dipole of length L, kL/2 = π L/λ
        s = np.where(np.sin(t) == 0, 1.0, np.sin(t))
        return np.where(np.sin(t) == 0, 0.0, ((np.cos(kl2 * np.cos(t)) - np.cos(kl2)) / s) ** 2)
    assert _pattern_error(theta, d, dipole_of_length) < 0.02
    # a lossless antenna radiates the power it accepts (1 W incident)
    accepted = 10 ** ((pat.peak_gain_dbi - pat.peak_directivity_dbi) / 10)
    assert pat.radiated_power == pytest.approx(accepted, rel=0.03)


@pytest.mark.slow
@case.phenomenon
def test_monopole_on_ground_plane():
    g = case.geometry(maxh=LAM / 8)
    box = LAM
    air = g.box(box, box, 0.7 * box, position=(-box / 2, -box / 2, 0), material=rf.Air())
    rf.PEC(_strip(g, GAP, 0.235 * LAM - GAP))
    rf.LumpedPort(_strip(g, 0, GAP, maxh=W / 2), direction=(0, 0, 1), z0=36.5)
    rf.ABC(air.faces.max(axis="z"), air.faces.min(axis="x"), air.faces.max(axis="x"),
           air.faces.min(axis="y"), air.faces.max(axis="y"))
    theta, d, pat = _solve(g)
    assert pat.peak_directivity_dbi == pytest.approx(5.16, abs=0.2)
    below = theta > np.pi / 2 + 1e-9
    assert np.all(d[:, below] < -90), "field below the ground plane"
    assert d[:, 45].mean() == pytest.approx(pat.peak_directivity_dbi, abs=0.3), \
        "the peak is not on the ground plane (horizon)"
    accepted = 10 ** ((pat.peak_gain_dbi - pat.peak_directivity_dbi) / 10)
    assert pat.radiated_power == pytest.approx(accepted, rel=0.03)
