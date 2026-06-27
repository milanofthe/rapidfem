# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Driven RLC lumped port — the reactive port termination reflects on cue.

A short 50 Ω stripline is driven by ``rf.LumpedPort(z0=50)`` at one end and
terminated by a SECOND lumped port carrying a series reactance,
``rf.LumpedPort(z0=50, l=L)``. The second port's Robin BC loads the line with
its termination impedance ``Z(ω) = R + jωL`` (R = 50 Ω), so looking into the
driven port across the electrically-short matched section the input impedance is
``Z(ω)`` and the reflection is the textbook

    Γ = (Z(ω) − R) / (Z(ω) + R) = jωL / (2R + jωL).

This exercises the frequency-dependent RLC termination of the driven lumped
port (kernel: LumpedPort.impedance / get_gamma), distinct from the passive
``rf.LumpedElement`` load. The pure-R limit (l = 0) is the standard port,
covered by the matched-load tests. Derivation: derivations/lumped_port/.
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

ER = 4.0
SUB_H = 1.0e-3
LINE_W = 0.5e-3        # w → Z0 ≈ 50 Ω shielded stripline
LINE_L = 5.0e-3
SUB_W = 5.0e-3
GAP_IN = 0.4e-3
ZREF = 50.0
L_TERM = 2.0e-9       # series inductance on the terminating port


def _gamma_rlc(freq, r=ZREF, l=L_TERM):
    """Γ = (Z−R)/(Z+R), Z = R + jωL  →  jωL/(2R + jωL)."""
    jwl = 1j * 2 * np.pi * freq * l
    return jwl / (2 * r + jwl)


def _build(g):
    hb = SUB_H / 2.0
    diel = rf.Dielectric(er=ER, tand=0.0, maxh=SUB_H / 3.0)
    lower = g.box(SUB_W, LINE_L, hb, position=(-SUB_W / 2, 0.0, 0.0), material=diel)
    upper = g.box(SUB_W, LINE_L, hb, position=(-SUB_W / 2, 0.0, hb),
                  material=rf.Dielectric(er=ER, tand=0.0, maxh=SUB_H / 3.0))
    trace_l = LINE_L - 2.0 * GAP_IN
    trace = g.xy_plate(LINE_W, trace_l, position=(-LINE_W / 2, GAP_IN, hb))
    feed = g.xz_plate(LINE_W, hb, position=(-LINE_W / 2, GAP_IN, 0.0))
    term = g.xz_plate(LINE_W, hb, position=(-LINE_W / 2, LINE_L - GAP_IN, 0.0))
    g.fragment(lower, upper, trace, feed, term)
    rf.PEC(trace)
    rf.LumpedPort(feed, direction=(0, 0, 1), z0=ZREF)               # driven
    rf.LumpedPort(term, direction=(0, 0, 1), z0=ZREF, l=L_TERM)     # reactive termination
    return feed, term


FREQS = np.linspace(2.0e9, 4.0e9, 4)


@pytest.mark.slow
@case.phenomenon
def test_inductive_port_termination_reflection():
    """Port 2 = 50 Ω + 2 nH terminates the line; |S11| follows jωL/(2R+jωL)."""
    g = case.geometry(maxh=rf.lambda_maxh(f_max=float(np.max(FREQS)), er_max=ER))
    _build(g)
    prob, res = case.sweep(g, FREQS, z0=ZREF)
    s11 = np.abs(res.sparams[:, 0, 0])
    analytic = np.abs([_gamma_rlc(f) for f in FREQS])

    # reflection must grow with frequency (inductive) and track the analytic curve
    assert s11[-1] > s11[0], (
        f"|S11| should rise with f for an inductive termination: {np.round(s11, 3)}")
    rel = np.abs(s11 - analytic) / analytic
    assert float(np.median(rel)) < 0.25, (
        f"RLC-port reflection off analytic jωL/(2R+jωL): FEM |S11|={np.round(s11, 3)} "
        f"vs analytic {np.round(analytic, 3)} (median {100*np.median(rel):.1f}%), "
        f"DOF={prob.n_dofs}")
