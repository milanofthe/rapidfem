# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""The time-domain operator built from the native model (no FD solve).

A coarse WR-90 section with two TE10 ports and PEC walls: the operator is
built from the geometry's native model, a driven transient stays finite and
carries the injected pulse, the native Gaussian pulse drives it exactly like
a Python callable, ports are addressed by their face tag, and the
frequency-domain-only lumped port is refused.
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


def test_native_pulse_matches_a_python_callable():
    # The native GaussianPulse is sampled in Rust; wrapping it in a Python
    # callable takes the per-step call path and must drive the same run.
    g, p_in = _guide()
    ptd = rf.ProblemTD(g, order=1, flux="upwind")
    pulse = rf.GaussianPulse(t0=90e-12, tau=22e-12, f0=10e9)
    t = np.linspace(0.0, 200e-12, 9)
    ref = np.exp(-((t - 90e-12) / 22e-12) ** 2) * np.cos(2 * np.pi * 10e9 * (t - 90e-12))
    assert np.allclose(pulse(t), ref, rtol=0.0, atol=1e-15)
    assert pulse(90e-12) == 1.0
    run = dict(port=p_in, dt=3e-12, steps=20, method="explicit", verbose=False)
    native = ptd.transient(waveform=pulse, **run)
    python = ptd.transient(waveform=lambda s: pulse(s), **run)
    assert np.array_equal(native, python)
    rows = ptd.port_signals(native, [p_in]).responses
    assert rows.shape == (1, 21) and np.abs(rows).max() > 0.0


def test_port_of_another_geometry_is_refused():
    g, _ = _guide()
    _, foreign = _guide()
    ptd = rf.ProblemTD(g, order=1)
    with pytest.raises(ValueError, match="not a port"):
        ptd.transient(port=foreign, waveform=rf.GaussianPulse(t0=0.0, tau=1e-11),
                      dt=3e-12, steps=1, verbose=False)
