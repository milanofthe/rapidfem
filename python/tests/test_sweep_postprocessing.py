# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Native post-processing of a sweep on a coarse WR-90 section.

The per-sweep port impedances are the analytic TE10 wave impedance, the
renormalization is the identity against them and agrees with the free
function, the Touchstone file carries the two-port column-major order, and
the element error indicator comes with its tet centroids.
"""
import numpy as np
import pytest

import rapidfem as rf

MM = 1e-3
A, B, L = 22.86 * MM, 10.16 * MM, 30 * MM
F = 10e9


@pytest.fixture(scope="module")
def solved():
    g = rf.Geometry(maxh=4 * MM)
    air = g.box(A, B, L, material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    prob = rf.ProblemFD(g)
    return prob, prob.sweep([F])


def test_port_impedances_are_the_te10_wave_impedance(solved):
    _, res = solved
    z = np.asarray(res.port_impedances)
    assert z.shape == (1, 2)
    eta0 = 376.730313
    fc = 299_792_458.0 / (2 * A)
    assert np.allclose(z, eta0 / np.sqrt(1 - (fc / F) ** 2), rtol=1e-3)


def test_renormalize(solved):
    _, res = solved
    s = np.asarray(res.sparams)
    z = np.asarray(res.port_impedances)
    assert np.allclose(res.renormalize(z[0]), s, atol=1e-10)
    s50 = res.renormalize(50.0)
    assert np.allclose(s50, rf.io.renormalize_sparams(s, z, 50.0), atol=1e-12)
    assert abs(s50[0, 0, 0]) > 0.5, "a ~500 ohm guide is mismatched against 50 ohm"


def test_touchstone_two_port(solved, tmp_path):
    _, res = solved
    path = tmp_path / "wr90.s2p"
    res.to_touchstone(path, z0=50.0)
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("!")]
    assert lines[0] == "# HZ S RI R 50"
    row = np.array([float(v) for v in lines[1].split()])
    s = np.asarray(res.sparams)[0]
    pairs = row[1::2] + 1j * row[2::2]
    assert row[0] == pytest.approx(F)
    assert np.allclose(pairs, [s[0, 0], s[1, 0], s[0, 1], s[1, 1]], rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError, match="fmt"):
        res.to_touchstone(path, fmt="xy")


def test_element_errors_carry_centroids(solved):
    prob, res = solved
    errs = prob.element_errors(res, theta=0.3)
    assert isinstance(errs, rf.ErrorIndicator)
    c = np.asarray(errs.tet_centroids)
    assert c.shape == (prob.n_tets, 3)
    nodes = np.asarray(prob.mesh_nodes)
    assert np.all(c >= nodes.min(axis=0) - 1e-12) and np.all(c <= nodes.max(axis=0) + 1e-12)
    assert len(errs.marked) > 0 and errs.freq_hz == F
    with pytest.raises(IndexError):
        prob.element_errors(res, freq_idx=5)
