# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""The time-domain operator built from the native model (no FD solve).

A coarse WR-90 section with two TE10 ports and PEC walls: the operator is
built from ``build_model``, a driven transient stays finite and carries the
injected pulse, and the frequency-domain-only lumped port is refused.
"""
import numpy as np
import pytest

import rapidfem as rf

A, B, L = 22.86e-3, 10.16e-3, 30.0e-3


def _guide(port=rf.RectWaveguidePort):
    g = rf.Geometry(maxh=6e-3)
    air = g.box(A, B, L, position=(-A / 2, -B / 2, 0), material=rf.Air())
    if port is rf.LumpedPort:
        p_in = rf.LumpedPort(air.faces.min(axis="z"), direction=(0, 1, 0))
    else:
        p_in = port(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    return g, p_in


def test_driven_transient_from_the_model():
    g, p_in = _guide()
    ptd = rf.ProblemTD(g, order=1, flux="upwind")
    assert ptd.n_dofs > 0
    pulse = rf.GaussianPulse(t0=90e-12, tau=22e-12, f0=10e9)
    traj = ptd.transient(port=p_in, waveform=pulse, dt=3e-12, steps=40,
                         method="explicit", device="cpu", verbose=False)
    amp = np.linalg.norm(traj, axis=1)
    assert np.all(np.isfinite(traj))
    assert amp[0] == 0.0 and amp.max() > 0.0, "the port must inject the pulse"


def test_lumped_port_is_refused():
    g, _ = _guide(port=rf.LumpedPort)
    with pytest.raises(RuntimeError, match="LumpedPort is not supported"):
        rf.ProblemTD(g, order=1)
