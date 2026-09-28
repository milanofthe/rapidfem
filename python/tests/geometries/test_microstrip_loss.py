# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Microstrip conductor + dielectric loss vs frequency — Hammerstad-Jensen.

A shielded (PEC-boxed) 50-ohm microstrip on RO4003C carries two loss
mechanisms: a copper Leontovich ``rf.SurfaceImpedance`` on the trace and
ground plane (conductor loss) and the substrate loss tangent (dielectric
loss). Each frequency is solved with its OWN exact wave-port mode (``f0=f``),
so no single-f0 dispersion approximation contaminates the band.

A lossless twin (PEC conductors, tanδ = 0, *identical* mesh) is solved at
every frequency and its residual attenuation is SUBTRACTED. That twin captures
the solver's numerical floor (discretisation dispersion + imperfect port
de-embedding, ~2-5 dB/m here); the difference isolates the physical loss:

    αc(f) = α[SIBC, tanδ=0] − α[PEC, tanδ=0]      (conductor only)
    αtot(f) = α[SIBC, tanδ] − α[PEC, tanδ=0]      (conductor + dielectric)

extracted from |S21| of a matched line, α = −ln|S21|/L. Two things are
checked that a *wrong basis* (the pre-June-2026 ported element) would fail:

  * MAGNITUDE — αtot within a generous band of the Hammerstad-Jensen /
    Pozar closed forms. HJ is itself ~10-20 % accurate and a zero-thickness
    trace carries a singular edge current (loss ~1.3x high, mesh-converged),
    so the band is wide by design: it catches order-of-magnitude / sign
    failures, not HJ decimals.
  * SCALING — the conductor part must follow the √f skin-effect signature
    (Rs ∝ √f); the fitted exponent must sit near 0.5, not flat or erratic.

Cross-check (independent of the |S21| extraction): the power-balance
dissipation 1 − |S11|² − |S21|² must give the same α — a guard on the
post-processing chain.

References
----------
E. Hammerstad and O. Jensen, "Accurate Models for Microstrip Computer-Aided
Design," IEEE MTT-S Digest, 1980 — Z0(u,εr), εeff(u,εr).
D. M. Pozar, "Microwave Engineering", 4th ed., §3.8 — microstrip attenuation:
αc = Rs/(Z0·W); αd = k0·εr·(εeff−1)·tanδ / (2·√εeff·(εr−1)).
"""
import math

import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

mm = 1e-3
NP2DB = 20.0 / math.log(10.0)      # Np/m -> dB/m

# RO4003C-like, ~50 ohm (Hammerstad-Jensen Z0 = 50.2 ohm, eeff = 2.79).
ER = 3.55
TAND = 0.0027
SUB_H = 0.508 * mm
LINE_W = 1.13 * mm
SIGMA = 5.8e7                       # copper, S/m

# Shielded box: walls far enough that the quasi-TEM mode is unperturbed and no
# in-band box resonance (verified resonance-free 2-20 GHz), yet < 100 k DOF at
# these frequencies. Line length gives a resolvable αL with |S11| small.
LINE_L = 15.0 * mm
SUB_W = 5.0 * mm
AIR_H = 3.0 * mm
FREQS = np.array([2.0e9, 3.0e9, 4.0e9])   # 2x span -> clean √f exponent, < 50 k DOF


# ── Hammerstad-Jensen closed forms (local; only this test needs them) ───────
def _hj_eeff(w, h, er):
    u = w / h
    a = (1.0
         + (1.0 / 49.0) * math.log((u**4 + (u / 52.0) ** 2) / (u**4 + 0.432))
         + (1.0 / 18.7) * math.log(1.0 + (u / 18.1) ** 3))
    b = 0.564 * ((er - 0.9) / (er + 3.0)) ** 0.053
    return (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 + 10.0 / u) ** (-a * b)


def _hj_z0(w, h, er):
    u = w / h
    fu = 6.0 + (2.0 * math.pi - 6.0) * math.exp(-((30.666 / u) ** 0.7528))
    z0_air = (ref.ETA0 / (2.0 * math.pi)) * math.log(fu / u + math.sqrt(1.0 + (2.0 / u) ** 2))
    return z0_air / math.sqrt(_hj_eeff(w, h, er))


def _hj_alpha_c(f):
    """Pozar thin-strip conductor attenuation, Np/m."""
    return ref.surface_resistance(f, SIGMA) / (_hj_z0(LINE_W, SUB_H, ER) * LINE_W)


def _hj_alpha_d(f):
    """Pozar microstrip dielectric attenuation, Np/m."""
    eeff = _hj_eeff(LINE_W, SUB_H, ER)
    return ref.k0(f) * ER * (eeff - 1.0) * TAND / (2.0 * math.sqrt(eeff) * (ER - 1.0))


# ── one lossy-or-lossless microstrip solve at an exact modal frequency ──────
def _solve(freq, *, metal, tand):
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=freq, er_max=ER, per_lambda=8))
    # One element across the substrate in the volume, the resolution the loss
    # bands below are calibrated on; the port cross-sections get finer below.
    fr4 = rf.Dielectric(er=ER, tand=tand, maxh=SUB_H)
    sub = g.box(SUB_W, LINE_L, SUB_H, position=(-SUB_W / 2, 0, 0), material=fr4)
    air = g.box(SUB_W, LINE_L, AIR_H, position=(-SUB_W / 2, 0, SUB_H), material=rf.Air())
    trace = g.xy_plate(LINE_W, LINE_L, position=(-LINE_W / 2, 0, SUB_H))
    g.fragment(sub, air, trace)
    ground = sub.faces.min(axis="z")

    # Copper Leontovich BC (loss) or PEC (lossless twin) on trace + ground.
    cond = (rf.SurfaceImpedance(trace, ground, conductivity=SIGMA) if metal == "sibc"
            else rf.PEC(trace, ground))
    # The conductor faces are marked as the internal quasi-TEM conductor for the
    # (lossless) 2-D port mode via pec=[cond]; in 3-D they stay lossy SIBC (they
    # are NOT forced to 3-D PEC — build_pec_tris only excludes them from the
    # implicit exterior-PEC fill).
    ports = [(sub.faces.min(axis="y"), air.faces.min(axis="y")),
             (sub.faces.max(axis="y"), air.faces.max(axis="y"))]
    # The 2-D port mode needs a finer cross-section than the volume.
    for faces in ports:
        for f in faces:
            f.maxh = SUB_H / 3.0
    for faces in ports:
        rf.WavePort(*faces, f0=freq, mode_kind="auto", pec=[cond])
    # Closed metal shield: the y-ends are the ports, the other five faces PEC.
    rf.PEC(sub.faces.min(axis="x"), sub.faces.max(axis="x"),
           air.faces.min(axis="x"), air.faces.max(axis="x"), air.faces.max(axis="z"))

    g.mesh()
    prob = rf.ProblemFD(g)
    res = prob.sweep(np.array([freq]), z0=50.0)
    assert prob.n_dofs < case.DOF_BUDGET, f"{prob.n_dofs} DOF >= budget"
    s11 = abs(res.sparams[0, 0, 0]); s21 = abs(res.sparams[0, 1, 0])
    pdiss = 1.0 - s11 ** 2 - s21 ** 2
    a_s21 = -math.log(max(s21, 1e-12)) / LINE_L               # Np/m
    a_pb = -0.5 * math.log(max(1.0 - pdiss, 1e-12)) / LINE_L  # Np/m
    return dict(s11=s11, s21=s21, a_s21=a_s21, a_pb=a_pb)


@pytest.mark.slow
@case.phenomenon
def test_microstrip_loss_vs_frequency():
    # Anchor: the chosen geometry really is a ~50 ohm quasi-TEM microstrip.
    assert 45.0 < _hj_z0(LINE_W, SUB_H, ER) < 55.0
    assert 1.0 < _hj_eeff(LINE_W, SUB_H, ER) < ER

    a_tot, a_tot_pb, a_c, s11_max = [], [], [], 0.0
    for f in FREQS:
        twin = _solve(f, metal="pec", tand=0.0)      # numerical floor
        lossy = _solve(f, metal="sibc", tand=TAND)   # conductor + dielectric
        cond = _solve(f, metal="sibc", tand=0.0)     # conductor only
        a_tot.append(lossy["a_s21"] - twin["a_s21"])
        a_tot_pb.append(lossy["a_pb"] - twin["a_pb"])
        a_c.append(cond["a_s21"] - twin["a_s21"])
        s11_max = max(s11_max, lossy["s11"], twin["s11"], cond["s11"])
    a_tot = np.array(a_tot); a_tot_pb = np.array(a_tot_pb); a_c = np.array(a_c)
    hj_tot = np.array([(_hj_alpha_c(f) + _hj_alpha_d(f)) for f in FREQS])
    hj_c = np.array([_hj_alpha_c(f) for f in FREQS])
    ratio = a_tot / hj_tot

    diag = (
        "  f[GHz]  a_tot   a_pb   a_c  | HJ_tot HJ_c | ratio\n" +
        "\n".join(
            f"  {f/1e9:5.1f}  {at*NP2DB:6.2f} {ap*NP2DB:6.2f} {ac*NP2DB:6.2f} | "
            f"{ht*NP2DB:6.2f} {hc*NP2DB:5.2f} | {r:.2f}"
            for f, at, ap, ac, ht, hc, r in
            zip(FREQS, a_tot, a_tot_pb, a_c, hj_tot, hj_c, ratio)))

    # De-embedding is clean: the matched line reflects almost nothing, so the
    # raw −ln|S21|/L extraction needs no port correction.
    assert s11_max < 0.12, f"|S11| rose to {s11_max:.3f}\n{diag}"

    # Not garbage: every attenuation is strictly positive and rises with f.
    assert np.all(a_tot > 0), f"non-positive loss (garbage basis?)\n{diag}"
    assert np.all(np.diff(a_tot) > 0), f"loss not monotone in f\n{diag}"
    assert np.all(a_c > 0), f"non-positive conductor loss\n{diag}"

    # MAGNITUDE: total loss within a generous band of Hammerstad-Jensen. The
    # band is wide on purpose (HJ ~10-20 % accurate; zero-thickness trace ~1.3x
    # high) — it catches order-of-magnitude / scaling failure, not HJ decimals.
    assert np.all((ratio > 0.5) & (ratio < 2.0)), (
        f"αtot/HJ out of [0.5, 2.0] band\n{diag}")

    # SCALING: the conductor part follows the √f skin-effect law (Rs ∝ √f).
    # A wrong basis produced flat / erratic loss; the exponent would miss 0.5.
    exponent = float(np.polyfit(np.log(FREQS), np.log(a_c), 1)[0])
    assert 0.35 < exponent < 0.65, (
        f"conductor-loss exponent {exponent:.3f} not ~0.5 (√f skin effect)\n{diag}")

    # CROSS-CHECK: the independent power-balance extraction (1−|S11|²−|S21|²)
    # must agree with the |S21| attenuation — a guard on the post-processing.
    pb_err = float(np.max(np.abs(a_tot_pb - a_tot) / a_tot))
    assert pb_err < 0.10, (
        f"power-balance vs |S21| disagree by {pb_err*100:.1f}%\n{diag}")
