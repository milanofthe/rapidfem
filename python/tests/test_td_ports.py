"""TD port verification suite for the C-series ports (Coax, Floquet,
Periodic). These cover the Python wiring on top of the Rust gates,
which validate the operator-level physics already.

Each test runs a small transient and asserts a quantitative pass
criterion (low reflection, near-unity transmission, finite-energy
preservation).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import rapidfem as rf

C = 299_792_458.0
MM = 1e-3

slow = pytest.mark.slow


# -----------------------------------------------------------------------------
# 1. CoaxPort - straight matched coaxial line, TEM transmission.
# -----------------------------------------------------------------------------

@slow
def test_coax_line_transmits_tem(report):
    """A straight matched 50 ohm coaxial air-line driven at port 0 with
    a matched coax port at port 1. Inner-conductor and outer-shield
    surfaces are PEC; end caps are coax ports. The TEM mode is
    dispersionless, so transmission must be close to unity and
    reflection close to zero across a broad band. Mirrors the
    `fd_coax_step.py` geometry pattern; covers the Python coax wiring
    on top of the Rust C1 gate.
    """
    report.note(
        "Straight matched 50 ohm air-filled coax line (ri=1.50 mm, "
        "ro=3.45 mm, length 20 mm) with coax ports on both end caps and "
        "PEC inner/outer conductors. The TEM mode is dispersionless, so "
        "across 3-7 GHz the reflection |S11| must stay near zero and the "
        "transmission |S21| near unity. Covers the Python coax wiring on "
        "top of the Rust C1 gate."
    )
    r_inner = 1.50 * MM
    r_outer = 3.45 * MM            # ~50 ohm air coax
    length = 20.0 * MM

    g = rf.Geometry(maxh=3.0 * MM)
    air = g.cylinder(radius=r_outer, height=length, position=(0, 0, 0),
                     material=rf.Air())
    inner = g.cylinder(radius=r_inner, height=length, position=(0, 0, 0),
                       material=rf.Air())
    g.fragment(air, inner)
    # End caps carry the two coax ports.
    rf.CoaxPort(air.faces.min(axis="z"), ri=r_inner, ro=r_outer,
                origin=(0, 0, 0))
    rf.CoaxPort(air.faces.max(axis="z"), ri=r_inner, ro=r_outer,
                origin=(0, 0, length))
    # Everything else (inner-conductor surface + outer shield) is PEC.
    rf.PEC(*air.faces.unassigned)
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    # Coax has no cutoff: a narrow sweep around 5 GHz is enough to
    # demonstrate TEM transmission.
    freqs = np.linspace(3.0e9, 7.0e9, 5)
    res = ptd.sparams(freqs, dt=2.0e-12, steps=400, verbose=False)
    s = res.sparams

    s11 = np.abs(s[:, 0, 0])
    s21 = np.abs(s[:, 1, 0])
    print(f"  coax |S11|: max {s11.max():.3f}")
    print(f"  coax |S21|: min {s21.min():.3f} max {s21.max():.3f}")
    assert s11.max() < 0.2, f"coax reflection too high: {s11.max():.3f}"
    assert s21.min() > 0.7, f"coax transmission too low: {s21.min():.3f}"

    report.plot_sparams(
        freqs,
        {"TD": s},
        entries=[(1, 1), (2, 1)],
        title="Coax line S-parameters",
        caption="matched 50 ohm air coax, TEM transmission",
    )
    report.metric("max |S11|", float(s11.max()), bound=0.2,
                  detail="coax reflection across 3-7 GHz")
    report.metric("min |S21|", float(s21.min()), lower=0.7,
                  detail="coax transmission across 3-7 GHz")
    report.table(
        "coax |S| over band",
        ["f (GHz)", "|S11|", "|S21|"],
        [[f / 1e9, s11[k], s21[k]] for k, f in enumerate(freqs)],
    )


# -----------------------------------------------------------------------------
# 2. PeriodicBoundary - field continuity across a periodic pair.
# -----------------------------------------------------------------------------

@slow
@pytest.mark.skip(
    reason="gmsh `setPeriodic` is not wired through Python "
    "`PeriodicBoundary`; opposite faces mesh with different triangle "
    "counts and the Rust matcher rejects the pair. Rust C2 gate covers "
    "the operator-level physics on a structured_box mesh. Wiring "
    "setPeriodic into Geometry.mesh() is a separate task."
)
def test_periodic_boundary_pair_runs_end_to_end():
    """A periodic unit cell driven by a localised pulse. With opposite
    faces tied periodically the energy can recirculate; without (just
    PEC boundaries) the energy stays bounded the same way. The
    operator must build and propagate finitely; energy must not blow
    up. Mirrors the Rust C2 gate's energy-drift check at the Python
    level.
    """
    side = 30.0 * MM
    g = rf.Geometry(maxh=side / 4)
    box = g.box(side, side, side, material=rf.Air())
    # Pair the x = 0 and x = side faces.
    rf.PeriodicBoundary(
        box.faces.min(axis="x"),
        box.faces.max(axis="x"),
    )
    # All other faces stay PEC by default.
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    y0 = np.zeros(ptd.n_dof)
    y0[ptd.probe_dof(
        (side * 0.5, side * 0.5, side * 0.5), field="E", component="z"
    )] = 1.0
    traj = ptd.transient(y0, dt=3e-12, steps=150, device="cpu")
    e0 = ptd.field_energy(traj[0])
    e_max = max(ptd.field_energy(traj[k]) for k in range(traj.shape[0]))
    e_end = ptd.field_energy(traj[-1])
    print(f"  periodic energy E(0)={e0:.4g}, max={e_max:.4g}, end={e_end:.4g}")
    assert np.all(np.isfinite(traj)), "transient must stay finite"
    # Central-flux with periodic + PEC is lossless; max-energy must
    # equal start within a few percent (central flux conserves energy
    # to machine precision; the drift is a per-step accumulator).
    drift = abs(e_max - e0) / e0
    assert drift < 0.1, (
        f"periodic energy drift {drift:.2%} (expected lossless)"
    )


# -----------------------------------------------------------------------------
# 3. FloquetPort - plane wave through a periodic unit cell.
# -----------------------------------------------------------------------------

@slow
@pytest.mark.skip(
    reason="depends on PeriodicBoundary on the four side faces, which "
    "needs gmsh `setPeriodic` wiring (see above). Rust C3 gate covers "
    "the Floquet-port operator physics on a structured mesh."
)
def test_floquet_unit_cell_transmits_plane_wave():
    """Normal-incidence plane wave through a thin air slab. The unit
    cell has Floquet ports on top + bottom and periodic boundaries on
    the four side faces. With no scatterer in the cell, transmission
    must be near unity and reflection near zero. Mirrors the Rust C3
    gate (transmission 0.996, reflection ~ machine eps).
    """
    side = 10.0 * MM
    thick = 15.0 * MM
    g = rf.Geometry(maxh=4.0 * MM)
    cell = g.box(side, side, thick, material=rf.Air())
    # Periodic on the four side faces.
    rf.PeriodicBoundary(
        cell.faces.min(axis="x"), cell.faces.max(axis="x"),
    )
    rf.PeriodicBoundary(
        cell.faces.min(axis="y"), cell.faces.max(axis="y"),
    )
    # Floquet ports on the two z-faces.
    rf.FloquetPort(
        cell.faces.min(axis="z"),
        scan_theta_deg=0.0, scan_phi_deg=0.0, mode_nr=1,
    )
    rf.FloquetPort(
        cell.faces.max(axis="z"),
        scan_theta_deg=0.0, scan_phi_deg=0.0, mode_nr=1,
    )
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    freqs = np.linspace(8e9, 12e9, 5)
    res = ptd.sparams(freqs, dt=1.5e-12, steps=400, verbose=False)
    s = res.sparams

    s11 = np.abs(s[:, 0, 0])
    s21 = np.abs(s[:, 1, 0])
    print(f"  Floquet |S11|: max {s11.max():.3f}")
    print(f"  Floquet |S21|: min {s21.min():.3f} max {s21.max():.3f}")
    assert s11.max() < 0.2, (
        f"Floquet empty-cell reflection too high: {s11.max():.3f}"
    )
    assert s21.min() > 0.7, (
        f"Floquet empty-cell transmission too low: {s21.min():.3f}"
    )


# -----------------------------------------------------------------------------
# 4. WavePort - numerically-solved cross-section mode vs the analytic TE10 port.
# -----------------------------------------------------------------------------

@slow
def test_wave_port_matches_analytic_te10(report):
    """A rectangular WR-90-style guide driven by a numerically-solved
    WavePort must reproduce the analytic RectWaveguidePort TE10 result:
    same cutoff and matching S-parameters above cutoff. This validates
    the 2D cross-section eigensolve end to end (eigenmode -> sampled
    (e_t, h_t) profile -> injection / extraction)."""
    report.note(
        "WR-90-style hollow guide (a=22.86 mm, b=10.16 mm, length 40 mm) "
        "driven by a numerically-solved scalar WavePort vs the analytic "
        "RectWaveguidePort. Validates the 2D cross-section eigensolve end "
        "to end: the eigenmode cutoff must land on the analytic TE10 "
        "f_c = c/(2a), and |S21|/|S11| must track the analytic port "
        "across 8-11 GHz (above the 6.56 GHz cutoff)."
    )
    a, b, length = 22.86 * MM, 10.16 * MM, 40.0 * MM
    freqs = np.linspace(8e9, 11e9, 4)  # above the 6.56 GHz TE10 cutoff

    def build(use_wave):
        g = rf.Geometry(maxh=5 * MM)
        air = g.box(a, b, length, material=rf.Air())
        if use_wave:
            rf.WavePort(air.faces.min(axis="z"), te=True)
            rf.WavePort(air.faces.max(axis="z"), te=True)
        else:
            rf.RectWaveguidePort(air.faces.min(axis="z"))
            rf.RectWaveguidePort(air.faces.max(axis="z"))
        g.mesh()
        return g

    # Cutoff: the eigensolve must land on the analytic TE10 f_c = c/(2a).
    ptd_w = rf.ProblemTD(build(True), order=2, flux="upwind")
    fc_analytic = C / (2.0 * a)
    fc_wave = C * ptd_w._op.port_cutoff(0) / (2.0 * np.pi)
    print(f"  WavePort f_c = {fc_wave/1e9:.3f} GHz, analytic {fc_analytic/1e9:.3f} GHz")
    assert abs(fc_wave - fc_analytic) / fc_analytic < 0.03, (
        f"WavePort cutoff {fc_wave/1e9:.3f} GHz off from "
        f"analytic {fc_analytic/1e9:.3f} GHz"
    )
    report.metric("WavePort f_c", fc_wave, unit="Hz",
                  ref=fc_analytic, tol=0.03,
                  detail="eigensolve cutoff vs analytic TE10 c/(2a)")

    s_w = ptd_w.sparams(freqs, dt=1.0e-12, steps=600, verbose=False).sparams
    s_a = rf.ProblemTD(build(False), order=2, flux="upwind").sparams(
        freqs, dt=1.0e-12, steps=600, verbose=False
    ).sparams

    s11_w, s21_w = np.abs(s_w[:, 0, 0]), np.abs(s_w[:, 1, 0])
    s11_a, s21_a = np.abs(s_a[:, 0, 0]), np.abs(s_a[:, 1, 0])
    for k, f in enumerate(freqs):
        print(f"  f={f/1e9:4.1f}GHz  |S21| wave={s21_w[k]:.3f} analytic={s21_a[k]:.3f}  "
              f"|S11| wave={s11_w[k]:.3f} analytic={s11_a[k]:.3f}")

    # The numerical mode must track the analytic mode: |S21| within 0.08
    # and |S11| within 0.05 across the band (a few percent is the finite
    # mesh / transient budget, identical for both ports).
    d21 = float(np.max(np.abs(s21_w - s21_a)))
    d11 = float(np.max(np.abs(s11_w - s11_a)))
    print(f"  max |S21| dev wave vs analytic: {d21:.3f}")
    print(f"  max |S11| dev wave vs analytic: {d11:.3f}")
    assert d21 < 0.08, f"|S21| wave vs analytic deviates {d21:.3f}"
    assert d11 < 0.05, f"|S11| wave vs analytic deviates {d11:.3f}"

    report.plot_sparams(
        freqs,
        {"WavePort": s_w, "analytic TE10": s_a},
        entries=[(1, 1), (2, 1)],
        title="Scalar WavePort vs analytic TE10",
        caption="solid = numerical WavePort, dashed = analytic RectWaveguidePort",
    )
    report.metric("max |S21| dev wave vs analytic", d21, bound=0.08,
                  detail="max over band of ||S21_wave| - |S21_analytic||")
    report.metric("max |S11| dev wave vs analytic", d11, bound=0.05,
                  detail="max over band of ||S11_wave| - |S11_analytic||")
    report.table(
        "scalar WavePort vs analytic |S| over band",
        ["f (GHz)", "|S21| wave", "|S21| analytic",
         "|S11| wave", "|S11| analytic"],
        [[f / 1e9, s21_w[k], s21_a[k], s11_w[k], s11_a[k]]
         for k, f in enumerate(freqs)],
    )


# -----------------------------------------------------------------------------
# 5. WavePort vector path - inhomogeneous-capable hybrid solve, validated on a
#    hollow guide where the analytic TE10 is the known answer.
# -----------------------------------------------------------------------------

@slow
def test_wave_port_vector_path_matches_analytic_te10(report):
    """The full-vector hybrid wave-port solve (WavePort(f0=...), the path
    that also handles inhomogeneous microstrip-class cross-sections) on a
    hollow WR-90 guide must reproduce the analytic TE10 transmission. The
    transmission |S21| (the Issue #10 metric) tracks tightly; |S11| carries
    a wider budget for the edge-element profile recovery (area-averaged to
    nodes), which is less sharp than the scalar gradient profile."""
    report.note(
        "Full-vector hybrid WavePort solve (WavePort(f0=...), the path "
        "that also handles inhomogeneous microstrip-class cross-sections) "
        "on a hollow WR-90 guide vs the analytic RectWaveguidePort. The "
        "transmission |S21| (the Issue #10 metric) must track the analytic "
        "mode tightly (within 0.05) across 8.5-9.5 GHz; reflection |S11| "
        "carries a wider budget for the edge-element profile recovery."
    )
    a, b, length = 22.86 * MM, 10.16 * MM, 40.0 * MM
    f0 = 9e9
    freqs = np.linspace(8.5e9, 9.5e9, 3)

    def build(vector):
        g = rf.Geometry(maxh=5 * MM)
        air = g.box(a, b, length, material=rf.Air())
        if vector:
            rf.WavePort(air.faces.min(axis="z"), f0=f0)
            rf.WavePort(air.faces.max(axis="z"), f0=f0)
        else:
            rf.RectWaveguidePort(air.faces.min(axis="z"))
            rf.RectWaveguidePort(air.faces.max(axis="z"))
        g.mesh()
        return g

    s_v = rf.ProblemTD(build(True), order=2, flux="upwind").sparams(
        freqs, dt=1.0e-12, steps=600, verbose=False
    ).sparams
    s_a = rf.ProblemTD(build(False), order=2, flux="upwind").sparams(
        freqs, dt=1.0e-12, steps=600, verbose=False
    ).sparams

    s21_v, s21_a = np.abs(s_v[:, 1, 0]), np.abs(s_a[:, 1, 0])
    s11_v, s11_a = np.abs(s_v[:, 0, 0]), np.abs(s_a[:, 0, 0])
    for k, f in enumerate(freqs):
        print(f"  f={f/1e9:4.1f}G  |S21| vec={s21_v[k]:.3f} analytic={s21_a[k]:.3f}  "
              f"|S11| vec={s11_v[k]:.3f} analytic={s11_a[k]:.3f}")
    d21 = float(np.max(np.abs(s21_v - s21_a)))
    print(f"  max |S21| dev vector vs analytic: {d21:.3f}")
    # Transmission (the Issue #10 metric) tracks the analytic mode tightly.
    assert d21 < 0.05, f"vector |S21| deviates {d21:.3f} from analytic"
    # Reflection is in the same ballpark (edge-element profile-recovery
    # budget); both are low-reflection.
    assert s11_v.max() < 0.30, f"vector |S11| too high: {s11_v.max():.3f}"

    report.plot_sparams(
        freqs,
        {"vector WavePort": s_v, "analytic TE10": s_a},
        entries=[(1, 1), (2, 1)],
        title="Vector hybrid WavePort vs analytic TE10",
        caption="solid = vector WavePort, dashed = analytic RectWaveguidePort",
    )
    report.metric("max |S21| dev vector vs analytic", d21, bound=0.05,
                  detail="Issue #10 metric: max ||S21_vec| - |S21_analytic||")
    report.metric("max |S11| vector", float(s11_v.max()), bound=0.30,
                  detail="vector reflection (edge-element profile-recovery budget)")
    report.table(
        "vector WavePort vs analytic |S| over band",
        ["f (GHz)", "|S21| vec", "|S21| analytic",
         "|S11| vec", "|S11| analytic"],
        [[f / 1e9, s21_v[k], s21_a[k], s11_v[k], s11_a[k]]
         for k, f in enumerate(freqs)],
    )
