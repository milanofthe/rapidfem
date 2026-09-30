# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""The magnetic field's sign: power flows away from the driven port.

With the e^{jωt} time dependence ∇×E = -jωμH; the wrong sign turns the
Poynting vector ½ Re(E × H*) around, and with it every quantity built on H
(the far field's equivalent currents among them).
"""
import numpy as np

import rapidfem as rf

MM = 1e-3


def test_power_flows_from_the_driven_port():
    g = rf.Geometry(maxh=4 * MM)
    air = g.box(22.86 * MM, 10.16 * MM, 40 * MM, material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    prob = rf.Problem(g)
    res = prob.sweep([10e9])
    z = np.asarray(prob.mesh_nodes)[:, 2]
    mid = (z > 10 * MM) & (z < 30 * MM)
    for port, sign in ((0, 1.0), (1, -1.0)):
        e = np.asarray(prob.field_at_nodes(res, 0, port))
        h = np.asarray(prob.h_field_at_nodes(res, 0, port))
        sz = 0.5 * np.real(np.cross(e, np.conj(h)))[mid, 2]
        assert sign * sz.mean() > 0, f"port {port + 1}: power flows the wrong way"
        # a matched guide: the flow is uniform along it
        assert np.all(sign * sz > -0.05 * abs(sz).max())
