# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Below-cutoff waveguide attenuator — evanescent decay rate α.

A wave squeezed through a guide section whose TE10 cutoff lies *above* the
operating frequency does not propagate: it tunnels, with the field amplitude
falling off as exp(−α·z), α = sqrt(k_c² − k0²). This is the rectangular-guide
"below-cutoff attenuator".

To keep the driving ports well defined we never drive a below-cutoff port.
Instead a WR-90 cross-section (cutoff ≈ 6.56 GHz) feeds and collects the wave at
both ends — both ports propagate cleanly at 9–10 GHz — while a NARROW central
section (a_narrow = 11 mm, cutoff ≈ 13.6 GHz) is the evanescent region. The
through-transmission is |S21| = |T_step| · exp(−α·L_c) with the two step
junctions contributing a length-independent coupling |T_step|. Measuring two
central lengths cancels that coupling and the port normalization entirely:

    |S21(L_short)| / |S21(L_long)| = exp(+α · (L_long − L_short)),

so the *ratio* isolates the pure evanescent decay rate α of the narrow guide,
which is asserted against the closed form within 15 %.

Reference: Pozar, *Microwave Engineering*, §3.3 (rectangular waveguide, the
below-cutoff / evanescent regime).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# WR-90 feed/collect cross-section (TE10 cutoff ≈ 6.56 GHz → propagates 9–10 GHz)
A_WIDE, B = 22.86e-3, 10.16e-3
# Narrow central section: TE10 cutoff ≈ 13.6 GHz → evanescent across the band.
A_NARROW = 11.0e-3
L_WIDE = 12.0e-3                 # feed/collect length (lets step modes settle)
LC_SHORT, LC_LONG = 8.0e-3, 14.0e-3
FREQS = np.linspace(9.0e9, 10.0e9, 3)
F_MAX = 10.5e9


def _alpha_below_cutoff(f: float, a: float) -> float:
    """Evanescent TE10 decay constant of an a-wide guide: sqrt(k_c² − k0²)."""
    kc = np.pi / a
    k = ref.k0(f)
    arg = kc * kc - k * k
    assert arg > 0.0, "section is NOT below cutoff at this frequency"
    return float(np.sqrt(arg))


def _is_throat_interface(_cog, bbox) -> bool:
    """True for the two conformal wide↔narrow interface faces (left open).

    After `g.fragment` the wide top face is split into a central strip that is
    shared with the narrow throat plus two metal step shoulders. The shared
    interface must stay un-tagged (the solver carries field continuity across
    it); every other face is a metal wall. The interface is the only z-normal
    face whose footprint matches the throat (x≈A_NARROW, y≈B). gmsh inflates a
    degenerate face's bbox by ~1e-7 m, hence the 1e-6 m flatness tolerance.
    """
    x_ext, y_ext, z_ext = bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]
    return (z_ext < 1e-6                                   # z-normal (flat in z)
            and A_NARROW * 0.9 < x_ext < A_NARROW * 1.05   # throat width in x
            and y_ext > B * 0.9)                           # full height in y


def _build_and_solve(lc: float):
    """Wide → narrow(L_c) → wide guide; return (prob, |S21| over FREQS)."""
    g = case.geometry(maxh=rf.lambda_maxh(f_max=F_MAX))
    narrow_maxh = 1.2e-3        # resolve the ~5 mm decay length in the throat

    inp = g.box(A_WIDE, B, L_WIDE, position=(-A_WIDE / 2, -B / 2, 0.0),
                material=rf.Air())
    nar = g.box(A_NARROW, B, lc, position=(-A_NARROW / 2, -B / 2, L_WIDE),
                material=rf.Air(), maxh=narrow_maxh)
    out = g.box(A_WIDE, B, L_WIDE, position=(-A_WIDE / 2, -B / 2, L_WIDE + lc),
                material=rf.Air())
    # The narrow throat is the connector that touches both wide sections.
    g.fragment(nar, inp, out)

    rf.RectWaveguidePort(inp.faces.min(axis="z"))
    rf.RectWaveguidePort(out.faces.max(axis="z"))
    # Metal everywhere except the ports and the two open throat interfaces:
    # the four guide side walls, plus the two step shoulders. `.outer` is no use
    # here (it keeps only faces on the *model* bbox, dropping the throat walls at
    # an interior x and the shoulders at an interior z), so select explicitly.
    walls = [f for box in (inp, nar, out)
             for f in box.faces.unassigned.where(
                 lambda c, b: not _is_throat_interface(c, b))]
    rf.PEC(*walls)

    prob, res = case.sweep(g, FREQS)
    return prob, np.abs(res.sparams[:, 1, 0])


@pytest.mark.slow
@case.phenomenon
def test_below_cutoff_evanescent_decay_rate():
    # Anchor: the narrow throat is genuinely below cutoff across the whole band.
    fc_narrow = ref.rect_cutoff_freq(A_NARROW, B, 1, 0)
    assert fc_narrow > FREQS.max() + 1e9, f"throat cutoff {fc_narrow/1e9:.2f} GHz"
    # ...while the wide feed/collect guides propagate (well above cutoff).
    assert ref.rect_cutoff_freq(A_WIDE, B, 1, 0) < FREQS.min()

    prob_s, s21_short = _build_and_solve(LC_SHORT)
    prob_l, s21_long = _build_and_solve(LC_LONG)

    # Longer evanescent path ⇒ strictly more attenuation at every frequency.
    assert np.all(s21_short > s21_long), (
        f"|S21| should fall with length: short={s21_short}, long={s21_long}"
    )
    # Stay clear of the numerical noise floor so the ratio is meaningful.
    assert s21_long.min() > 1e-4, f"|S21(long)| at floor: {s21_long}"

    d_lc = LC_LONG - LC_SHORT
    measured = s21_short / s21_long
    analytic = np.array([np.exp(_alpha_below_cutoff(f, A_NARROW) * d_lc)
                         for f in FREQS])

    # The measured length-ratio must match exp(+α·ΔL_c) — the pure evanescent
    # decay of the narrow guide — within 15 % across the band.
    err = np.abs(measured - analytic) / analytic
    assert err.max() < 0.15, (
        f"evanescent decay mismatch: measured ratio {measured}, "
        f"analytic {analytic}, rel-err {err}"
    )

    # Recovered α from the measurement, for the record.
    alpha_meas = np.log(measured) / d_lc
    alpha_ana = np.array([_alpha_below_cutoff(f, A_NARROW) for f in FREQS])
    print(f"\nn_dofs short={prob_s.n_dofs} long={prob_l.n_dofs}")
    print(f"alpha measured (Np/m): {alpha_meas}")
    print(f"alpha analytic (Np/m): {alpha_ana}")
