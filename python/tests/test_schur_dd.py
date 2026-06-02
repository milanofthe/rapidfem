"""Schur-complement domain-decomposition validation (GH issue #12).

For each geometry, the FD sweep solved by Schur DD (``n_subdomains=k``) must
reproduce the monolithic sweep's S-parameters. S-parameters are the physical
observable and are robust to the field-level ill-conditioning sensitivity of
the curl-curl system, so they agree far more tightly than the raw fields: the
gate is max|S_mono - S_schur| < 5e-3 (measured ~1e-4 on the matched cases).

Local developer tool — never CI (FEM solves do not belong in runners).
"""
from __future__ import annotations

import numpy as np
import pytest

import rapidfem as rf

C = 299_792_458.0
MM = 1e-3
slow = pytest.mark.slow

# S-parameter agreement gate: Schur DD vs monolithic, max over band and ports.
SPARAM_TOL = 5e-3
N_SUB = 4


def _compare(g, freqs, k=N_SUB):
    s_mono = np.asarray(rf.ProblemFD(g).sweep(freqs).sparams)
    s_schur = np.asarray(rf.ProblemFD(g, n_subdomains=k).sweep(freqs).sparams)
    return float(np.max(np.abs(s_mono - s_schur))), s_mono, s_schur


def _build_wr90():
    a, b, L = 22.86 * MM, 10.16 * MM, 30.0 * MM
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12.0e9))
    air = g.box(a, b, L, position=(-a / 2, -b / 2, 0), material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    return g, np.linspace(8.0e9, 12.0e9, 5)


def _build_coax():
    ri, ro, L = 1.50 * MM, 3.45 * MM, 20.0 * MM
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=10.0e9))
    air = g.cylinder(radius=ro, height=L, position=(0, 0, 0), material=rf.Air())
    inner = g.cylinder(radius=ri, height=L, position=(0, 0, 0), material=rf.Air())
    g.fragment(air, inner)
    rf.CoaxPort(air.faces.min(axis="z"), ri=ri, ro=ro, origin=(0, 0, 0))
    rf.CoaxPort(air.faces.max(axis="z"), ri=ri, ro=ro, origin=(0, 0, L))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    return g, np.linspace(2.0e9, 10.0e9, 5)


def _build_iris():
    A, B = 22.86 * MM, 10.16 * MM
    apertures = [10.0 * MM, 8.0 * MM, 10.0 * MM]
    spacing, iris_t, in_len, out_len = 15.0 * MM, 1.0 * MM, 12.0 * MM, 12.0 * MM
    L = in_len + (len(apertures) - 1) * spacing + 2 * iris_t + out_len
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12.4e9))
    air = g.box(A, B, L, position=(-A / 2, -B / 2, 0), material=rf.Air())
    z_centers = [in_len + iris_t / 2 + k * spacing for k in range(len(apertures))]
    iris_vols = []
    for zc, w in zip(z_centers, apertures):
        strip_w = (A - w) / 2
        for side in (-1, +1):
            x0 = -A / 2 if side < 0 else w / 2
            iris_vols.append(g.box(strip_w, B, iris_t,
                                   position=(x0, -B / 2, zc - iris_t / 2), material=rf.Air()))
    g.fragment(air, *iris_vols)
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    return g, np.linspace(9.0e9, 11.0e9, 3)


def _build_microstrip():
    # Coarser than the production example on purpose: this gate validates the
    # inhomogeneous wave-port + open-ABC *code path* under Schur DD, not mesh
    # convergence. The full-resolution microstrip (~400k DOFs) is a phase-1
    # scaling exercise — phase-0's explicit interface matrix holds every
    # subdomain factor at once, so the big case needs the matrix-free interface
    # solve (issue #12) before it fits memory comfortably.
    sub_h, er_sub, tand = 0.508 * MM, 3.55, 0.0027
    line_w, line_l, sub_w, air_h = 1.13 * MM, 30.0 * MM, 20.0 * MM, 10.0 * MM
    freqs = np.linspace(2.85e9, 3.30e9, 3)
    maxh = rf.lambda_maxh(f_max=3.3e9, er_max=er_sub) * 1.8
    g = rf.Geometry(maxh=maxh)
    fr4 = rf.Dielectric(er=er_sub, tand=tand, maxh=sub_h / 2)
    sub = g.box(sub_w, line_l, sub_h, position=(-sub_w / 2, 0, 0), material=fr4)
    air = g.box(sub_w, line_l, air_h, position=(-sub_w / 2, 0, sub_h), material=rf.Air())
    trace = g.xy_plate(line_w, line_l, position=(-line_w / 2, 0, sub_h))
    g.fragment(sub, air, trace)
    pec_strip = rf.PEC(trace, sub.faces.min(axis="z"))
    f0 = 0.5 * (freqs[0] + freqs[-1])
    rf.WavePort(sub.faces.min(axis="y"), air.faces.min(axis="y"), f0=f0, mode_kind="auto", pec=[pec_strip])
    rf.WavePort(sub.faces.max(axis="y"), air.faces.max(axis="y"), f0=f0, mode_kind="auto", pec=[pec_strip])
    rf.ABC(sub.faces.min(axis="x"), sub.faces.max(axis="x"),
           air.faces.min(axis="x"), air.faces.max(axis="x"), air.faces.max(axis="z"))
    g.auto_refine_features(base_maxh=maxh)
    g.mesh()
    return g, freqs


# Tier 1 (fast): hollow modal port + TEM coax port.

def test_schur_wr90_matches_monolithic():
    g, freqs = _build_wr90()
    d, _, _ = _compare(g, freqs)
    print(f"  WR-90 max|dS| = {d:.3e}")
    assert d < SPARAM_TOL, f"WR-90 Schur vs monolithic |dS| {d:.3e} > {SPARAM_TOL}"


def test_schur_coax_matches_monolithic():
    g, freqs = _build_coax()
    d, _, _ = _compare(g, freqs)
    print(f"  coax max|dS| = {d:.3e}")
    assert d < SPARAM_TOL, f"coax Schur vs monolithic |dS| {d:.3e} > {SPARAM_TOL}"


# Tier 2 / 4 (slow): internal-PEC reflective, and inhomogeneous wave port + open ABC.

@slow
def test_schur_iris_matches_monolithic():
    g, freqs = _build_iris()
    d, _, _ = _compare(g, freqs)
    print(f"  iris max|dS| = {d:.3e}")
    assert d < SPARAM_TOL, f"iris Schur vs monolithic |dS| {d:.3e} > {SPARAM_TOL}"


@slow
def test_schur_microstrip_matches_monolithic():
    g, freqs = _build_microstrip()
    d, _, _ = _compare(g, freqs)
    print(f"  microstrip max|dS| = {d:.3e}")
    assert d < SPARAM_TOL, f"microstrip Schur vs monolithic |dS| {d:.3e} > {SPARAM_TOL}"
