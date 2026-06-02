"""Self-test of the rapidfem report harness.

Fast, solver-free: it exercises every ``report`` fixture verb on synthetic
data so the plotting / rendering plumbing has a regression guard
independent of the (slow) physics suite. Not marked ``slow``.
"""
import numpy as np


def test_report_sparams_and_metrics(report):
    """Synthetic TD-vs-FD S-parameters: a one-pole transmission line, to
    exercise the S-parameter plot and a pass/fail metric pair."""
    f = np.linspace(8e9, 12e9, 201)
    w = 2 * np.pi * f
    pole = 2 * np.pi * 10e9
    s21 = 1.0 / (1.0 + 1j * (w - pole) / (0.3 * pole))
    s11 = 1.0 - np.abs(s21)
    s_td = np.zeros((f.size, 2, 2), dtype=complex)
    s_td[:, 0, 0] = s11
    s_td[:, 1, 0] = s21
    s_td[:, 0, 1] = s21
    s_td[:, 1, 1] = s11
    s_fd = s_td * (1.0 + 0.01)  # a 1% "reference" offset

    dev = float(np.max(np.abs(np.abs(s_td[:, 1, 0]) - np.abs(s_fd[:, 1, 0]))))
    report.note("Synthetic one-pole line — harness self-test, no solver.")
    report.metric("|S21| max TD-FD dev", dev, bound=0.05)
    report.metric("centre frequency", 10.0e9, ref=10.0e9, tol=0.01, unit="Hz")
    report.plot_sparams(
        f, {"TD": s_td, "FD": s_fd}, entries=[(1, 1), (2, 1)],
        title="S-parameters (synthetic)", caption="solid = TD, dashed = FD",
    )
    assert dev < 0.05


def test_report_xy_and_complex_plane(report):
    """Energy decay (xy) and a pole scatter (complex plane)."""
    t = np.linspace(0, 5, 200)
    report.plot_xy(
        t, {"vacuum": np.exp(-0.05 * t), "absorber": np.exp(-1.2 * t)},
        xlabel="time [ns]", ylabel="field energy", title="energy decay",
        logy=True,
    )
    eig = np.array([-0.02 + 1j * k for k in (1.0, 2.0, 3.0, -1.0, -2.0)])
    report.plot_complex_plane(
        {"reduced eigenvalues": eig}, title="state-space spectrum",
    )
    report.table("a small table", ["mode", "f [GHz]", "Q"],
                 [[1, 10.1, 320.0], [2, 14.3, 280.0]])
    assert np.all(np.isfinite(eig))
