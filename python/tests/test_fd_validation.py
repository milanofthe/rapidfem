"""FD backend (ProblemFD) production-validation suite.

Quantitative pass/fail tests for the frequency-domain S-parameter
solver. Each test builds a small, fast geometry from scratch, runs a
driven `ProblemFD.sweep`, and asserts the result against analytic
microwave theory (or a stored EMerge reference). Every gated quantity
is mirrored into the `report` fixture so the HTML report carries the
same numbers the asserts check.

Mark slow because every test meshes and runs a multi-frequency FD
sweep. Run the fast subset with `pytest -m "not slow"` and the full
suite with `pytest`. Meshes are deliberately coarse (lambda/12 air,
a few thousand tets) so each test finishes in well under a minute.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

import rapidfem as rf

# Touchstone / CSV loaders used as an optional second reference series.
from importlib.machinery import SourceFileLoader

C = 299_792_458.0
MM = 1e-3

# WR-90 X-band rectangular waveguide cross-section.
WR90_A, WR90_B = 22.86 * MM, 10.16 * MM

slow = pytest.mark.slow

# Stored EMerge reference (9-11 GHz, RI format) lives in the repo's
# top-level validation folder. Resolve it relative to this file so the
# test runs from any working directory; load compare.py via its path so
# we reuse the project's own Touchstone/CSV loaders (source of truth).
_VALIDATION_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tests", "validation")
)
_EMERGE_WR90_CSV = os.path.join(_VALIDATION_DIR, "emerge_wr90.csv")


def _load_compare():
    """Import the repo's validation/compare.py (load_csv / load_touchstone)."""
    path = os.path.join(_VALIDATION_DIR, "compare.py")
    return SourceFileLoader("rf_validation_compare", path).load_module()


def _build_wr90(length: float, maxh: float):
    """Straight WR-90 air guide of the given length with TE10 ports both ends."""
    g = rf.Geometry(maxh=maxh)
    air = g.box(WR90_A, WR90_B, length,
                position=(-WR90_A / 2, -WR90_B / 2, 0),
                material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    return g


# -----------------------------------------------------------------------------
# 1. WR-90 single-mode transmission: matched TE10 line, |S21|~1, |S11|~0.
# -----------------------------------------------------------------------------

@slow
def test_wr90_te10_transmission(report):
    """Straight 30 mm WR-90 air guide driven by TE10 modal ports at both
    ends, swept across the 8-12 GHz single-mode band. A lossless matched
    guide must pass essentially all power (|S21| ~ 1, |S11| ~ 0) and stay
    passive. The stored EMerge reference (9-11 GHz) is overlaid as a
    cross-check.
    """
    report.note(
        "Straight 30 mm WR-90 air-filled waveguide, TE10 modal ports at "
        "both ends, swept 8-12 GHz (single-mode band, above the "
        "6.56 GHz TE10 cutoff). A lossless matched guide passes all "
        "power: |S21| stays near 1, |S11| near 0, and the S-matrix stays "
        "passive (|S11|^2 + |S21|^2 <= 1.05). EMerge reference overlaid "
        "over its 9-11 GHz support."
    )
    freqs = np.linspace(8.0e9, 12.0e9, 9)
    g = _build_wr90(length=30.0 * MM, maxh=rf.lambda_maxh(f_max=12.0e9))
    result = rf.ProblemFD(g).sweep(freqs)
    s = np.asarray(result.sparams)

    s11 = np.abs(s[:, 0, 0])
    s21 = np.abs(s[:, 1, 0])
    s11_max = float(s11.max())
    s21_min = float(s21.min())
    # Passivity proxy for a lossless reciprocal 2-port at each frequency.
    passivity = s11 ** 2 + s21 ** 2
    pass_max = float(passivity.max())

    print(f"  |S21| min over band: {s21_min:.4f}")
    print(f"  |S11| max over band: {s11_max:.4f}")
    print(f"  max(|S11|^2+|S21|^2): {pass_max:.4f}")

    # Gates: matched lossless guide.
    assert s21_min > 0.97, f"|S21| dropped to {s21_min:.4f} in the passband"
    assert s11_max < 0.05, f"|S11| rose to {s11_max:.4f} in the passband"
    assert pass_max <= 1.05, f"passivity {pass_max:.4f} above 1.05"

    report.metric("min |S21| (passband)", s21_min, lower=0.97,
                  detail="lossless matched TE10 guide -> ~1")
    report.metric("max |S11| (passband)", s11_max, bound=0.05,
                  detail="matched TE10 ports -> ~0")
    report.metric("max |S11|^2+|S21|^2", pass_max, bound=1.05,
                  detail="passivity proxy, lossless 2-port")

    # Overlay EMerge reference, interpolated onto our sweep grid over the
    # 9-11 GHz overlap region (plot_sparams plots all series vs one axis).
    series = {"rapidfem": s}
    try:
        cmp = _load_compare()
        f_ref, s_ref = cmp.load_csv(_EMERGE_WR90_CSV)
        band = (freqs >= f_ref.min() - 1e6) & (freqs <= f_ref.max() + 1e6)
        # Build an EMerge series on `freqs` (NaN outside support so the plot
        # just doesn't draw there), interpolating each S entry.
        s_emerge = np.full_like(s, np.nan, dtype=complex)
        for i in range(2):
            for j in range(2):
                re_i = np.interp(freqs[band], f_ref, s_ref[:, i, j].real)
                im_i = np.interp(freqs[band], f_ref, s_ref[:, i, j].imag)
                s_emerge[band, i, j] = re_i + 1j * im_i
        series["EMerge"] = s_emerge
        s21_ref = np.abs(s_emerge[band, 1, 0])
        dev = float(np.max(np.abs(s21[band] - s21_ref)))
        print(f"  max |S21| dev vs EMerge (9-11 GHz): {dev:.4f}")
        assert dev < 0.05, f"|S21| deviates {dev:.4f} from EMerge"
        report.metric("max |S21| dev vs EMerge", dev, bound=0.05,
                      detail="9-11 GHz overlap region")
    except (FileNotFoundError, OSError):
        report.note("EMerge reference CSV not found; skipped overlay.")

    report.plot_sparams(
        freqs, series,
        entries=[(1, 1), (2, 1)],
        title="WR-90 TE10 S-parameters",
        caption="solid = rapidfem; dashed = EMerge (9-11 GHz support)",
    )
    report.table(
        "passband summary",
        ["quantity", "value", "gate"],
        [
            ["min |S21|", s21_min, ">= 0.97"],
            ["max |S11|", s11_max, "<= 0.05"],
            ["max |S11|^2+|S21|^2", pass_max, "<= 1.05"],
        ],
    )


# -----------------------------------------------------------------------------
# 2. WR-90 TE10 dispersion: S21 phase slope matches analytic beta(f).
# -----------------------------------------------------------------------------

@slow
def test_wr90_te10_dispersion_matches_analytic(report):
    """The transmission phase of a known-length WR-90 guide encodes the
    TE10 propagation constant beta(f). The per-step phase increment of
    S21 must track the closed-form dispersion relation
    beta = (2 pi f / c) sqrt(1 - (f_c/f)^2), with f_c = c/(2a). This is
    the cleanest analytic gate on the FD modal port: the absolute phase
    carries an arbitrary reference-plane offset, but the increment
    (group-delay slope) is offset-free.
    """
    a = WR90_A
    f_c = C / (2.0 * a)
    L = 30.0 * MM
    report.note(
        "WR-90 TE10 dispersion gate. The S21 phase increment between "
        f"adjacent sweep points must match the analytic beta(f) of a "
        f"{L/MM:.0f} mm guide, beta = (2 pi f/c) sqrt(1-(f_c/f)^2) with "
        f"f_c = c/(2a) = {f_c/1e9:.3f} GHz. Absolute phase carries a "
        "reference-plane offset, so we gate the offset-free phase "
        "increment (and the total accumulated phase span) instead."
    )
    freqs = np.linspace(8.0e9, 12.0e9, 9)
    g = _build_wr90(length=L, maxh=rf.lambda_maxh(f_max=12.0e9))
    result = rf.ProblemFD(g).sweep(freqs)
    s21 = np.asarray(result.sparams)[:, 1, 0]

    phase_fd = np.unwrap(np.angle(s21))
    beta_an = (2.0 * np.pi * freqs / C) * np.sqrt(1.0 - (f_c / freqs) ** 2)
    phase_an = -beta_an * L

    d_fd = np.diff(phase_fd)
    d_an = np.diff(phase_an)
    step_err_deg = float(np.rad2deg(np.max(np.abs(d_fd - d_an))))

    span_fd = float(phase_fd[-1] - phase_fd[0])
    span_an = float(phase_an[-1] - phase_an[0])
    span_rel_err = abs(span_fd - span_an) / abs(span_an)

    # Analytic cutoff sanity: f_c is the standard WR-90 TE10 cutoff.
    f_c_ref = 6.557e9

    print(f"  f_c = {f_c/1e9:.4f} GHz")
    print(f"  max per-step phase error: {step_err_deg:.4f} deg")
    print(f"  total phase span fd/an: {np.rad2deg(span_fd):.2f} / "
          f"{np.rad2deg(span_an):.2f} deg  (rel err {span_rel_err:.2e})")

    # Gates against analytic dispersion.
    assert step_err_deg < 2.0, f"per-step phase error {step_err_deg:.3f} deg"
    assert span_rel_err < 0.02, f"phase-span rel error {span_rel_err:.3%}"

    report.metric("TE10 cutoff f_c", f_c, ref=f_c_ref, tol=0.01, unit="Hz",
                  detail="c/(2a), WR-90 a = 22.86 mm")
    report.metric("max per-step phase err", step_err_deg, bound=2.0,
                  unit="deg", detail="|d_arg(S21)_fd - d_arg(S21)_an|")
    report.metric("phase-span rel error", span_rel_err, bound=0.02,
                  detail="accumulated S21 phase over 8-12 GHz vs analytic")

    report.plot_xy(
        freqs / 1e9,
        {"S21 phase FD [deg]": np.rad2deg(phase_fd),
         "S21 phase analytic [deg]": np.rad2deg(phase_an)},
        xlabel="frequency  [GHz]",
        ylabel="unwrapped arg(S21)  [deg]",
        title="WR-90 TE10 dispersion (S21 phase)",
        caption="curves differ by a constant reference-plane offset; "
                "the offset-free slope (increment) is what is gated",
    )
    report.table(
        "S21 phase increment: FD vs analytic",
        ["f_lo [GHz]", "f_hi [GHz]", "d_arg FD [deg]", "d_arg analytic [deg]"],
        [
            [freqs[k] / 1e9, freqs[k + 1] / 1e9,
             float(np.rad2deg(d_fd[k])), float(np.rad2deg(d_an[k]))]
            for k in range(len(d_fd))
        ],
    )


# -----------------------------------------------------------------------------
# 3. Air-coax TEM line: matched 50 ohm, |S11|~0, characteristic Z0 analytic.
# -----------------------------------------------------------------------------

@slow
def test_coax_tem_matched_50ohm(report):
    """Uniform air-filled coaxial line with coax (TEM) ports at both ends.
    With inner/outer radii chosen for ~50 ohm, the line is matched to the
    port reference impedance, so |S11| ~ 0 and |S21| ~ 1 broadband. The
    analytic air-coax characteristic impedance Z0 = 60 ln(ro/ri) ohm is
    used as the gate (the solver also reports its own port Z0).
    """
    ri, ro = 1.50 * MM, 3.45 * MM
    L = 20.0 * MM
    z0_analytic = 60.0 * math.log(ro / ri)  # air coax, er = 1
    report.note(
        "Uniform 20 mm air-filled coaxial line, TEM coax ports at both "
        f"ends (ri = {ri/MM:.2f} mm, ro = {ro/MM:.2f} mm). The geometry "
        f"is sized for Z0 = 60 ln(ro/ri) = {z0_analytic:.2f} ohm, close "
        "to the 50 ohm port reference, so the line is matched: |S11| ~ 0 "
        "and |S21| ~ 1 with no frequency dependence (a TEM line has no "
        "cutoff)."
    )
    freqs = np.linspace(2.0e9, 10.0e9, 9)

    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=10.0e9))
    air = g.cylinder(radius=ro, height=L, position=(0, 0, 0),
                     material=rf.Air())
    inner = g.cylinder(radius=ri, height=L, position=(0, 0, 0),
                       material=rf.Air())
    g.fragment(air, inner)
    rf.CoaxPort(air.faces.min(axis="z"), ri=ri, ro=ro, origin=(0, 0, 0))
    rf.CoaxPort(air.faces.max(axis="z"), ri=ri, ro=ro, origin=(0, 0, L))
    rf.PEC(*air.faces.unassigned)
    g.mesh()

    result = rf.ProblemFD(g).sweep(freqs)
    s = np.asarray(result.sparams)
    s11 = np.abs(s[:, 0, 0])
    s21 = np.abs(s[:, 1, 0])
    s11_max = float(s11.max())
    s21_min = float(s21.min())
    passivity = s11 ** 2 + s21 ** 2
    pass_max = float(passivity.max())

    print(f"  analytic Z0: {z0_analytic:.3f} ohm")
    print(f"  |S11| max: {s11_max:.4f}   |S21| min: {s21_min:.4f}")
    print(f"  max(|S11|^2+|S21|^2): {pass_max:.4f}")

    # Gates: matched lossless TEM line near 50 ohm.
    assert z0_analytic == pytest.approx(50.0, abs=2.0), \
        f"geometry Z0 {z0_analytic:.2f} ohm not near 50"
    assert s11_max < 0.05, f"|S11| up to {s11_max:.4f} (expected matched)"
    assert s21_min > 0.95, f"|S21| down to {s21_min:.4f} (expected ~1)"
    assert pass_max <= 1.05, f"passivity {pass_max:.4f} above 1.05"

    report.metric("analytic Z0", z0_analytic, ref=50.0, atol=2.0, unit="ohm",
                  detail="60 ln(ro/ri), air coax")
    report.metric("max |S11| (matched)", s11_max, bound=0.05,
                  detail="50 ohm coax into 50 ohm port reference")
    report.metric("min |S21| (TEM)", s21_min, lower=0.95,
                  detail="lossless TEM line, no cutoff")
    report.metric("max |S11|^2+|S21|^2", pass_max, bound=1.05,
                  detail="passivity proxy, lossless 2-port")

    report.plot_sparams(
        freqs, {"rapidfem": s},
        entries=[(1, 1), (2, 1)],
        title="Air-coax 50 ohm TEM S-parameters",
        caption="matched line: |S11| floor near 0 dB-free, |S21| ~ 0 dB",
    )
    report.table(
        "coax TEM summary",
        ["quantity", "value", "gate"],
        [
            ["analytic Z0 [ohm]", z0_analytic, "50 +/- 2"],
            ["max |S11|", s11_max, "<= 0.05"],
            ["min |S21|", s21_min, ">= 0.95"],
            ["max |S11|^2+|S21|^2", pass_max, "<= 1.05"],
        ],
    )
