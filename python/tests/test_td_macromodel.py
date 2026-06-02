"""Physical validation of the multipole (multi-shift rational-Krylov) ROM.

`ProblemTD.macromodel(...)` compiles the matrix-free DGTD operator into a
compact MIMO state-space `(A_hat, B_hat, C_hat)` by projecting the
port-seeded Krylov subspace. The *plain* block-Krylov basis
`span{B, A B, A^2 B, ...}` is biased toward the operator's spectral
radius (the DG eigenvalues reach hundreds of GHz on a refined mesh), so
it underflows on the in-band modes and the reduced S-matrix comes out
frequency-flat with |S21| ~ 0 (see `test_multipole_beats_plain` below).

The multipole remedy is the multi-shift rational-Krylov build
(`shift_freqs_hz=[...]`): several shifts `sigma_k = j*omega_k` placed
across the design band, a short shift-invert sequence at each, and the
union of the per-shift bases. `(sigma_k I - A)^-1` amplifies exactly the
eigenmodes near `sigma_k`, so the union spans the in-band physics that
plain Krylov misses.

These tests validate that the multipole ROM physically reproduces the
in-band S-parameters of an analytically-clean WR-90 rectangular
waveguide against the frequency-domain solver (`ProblemFD.sweep`), and
that the `state_space()` export is a lossless realisation of
`evaluate()`.

SPECTRAL RADIUS vs the GMRES cap
--------------------------------
The shift-invert inside `build_multi_shift` solves `(j*omega I - A) x = b`
with a matrix-free complex GMRES (single cycle, no restart, no
preconditioner). For a skew-symmetric DG operator the shift `j*omega`
sits in the middle of the imaginary spectrum, so GMRES needs a Krylov
basis large enough to resolve it; that basis size is the iteration cap
`gmres_max_iter`, now reachable from Python
(`macromodel(..., gmres_max_iter=...)`). Its default
(`GMRES_SHIFT_MAX_ITER`) suits a modest spectral radius; as `rho(A)`
grows the cap must grow too:

- order=1, coarse/short guide (rho(A) modest): the default cap converges;
  the multipole ROM tracks FD |S21| to ~2% across 9-12 GHz. These tests
  run here for speed.
- order=2, same geometry (n_dof ~ 240k, rho(A) ~3x larger): the default
  cap (60) does NOT converge and |S21| collapses to ~0; raising the cap
  (~250) restores convergence and |S21| returns to ~0.99
  (`test_multipole_rom_order2_needs_larger_gmres_cap`).

The |S21| gate is honest (~3%); the |S11| gate is intentionally loose
because the ROM carries a spurious reflection floor of ~0.1-0.2 that FD
(|S11| ~ 1e-3) does not -- documented in the metrics rather than
tolerance-tuned away. A tighter |S11| gate awaits a passivity-preserving
(SPRIM) projection of the multi-shift basis.
"""
from __future__ import annotations

import numpy as np
import pytest

import rapidfem as rf

MM = 1e-3
C = 299_792_458.0

slow = pytest.mark.slow

# WR-90 single-mode X-band geometry. Short guide + coarse mesh + order=1
# keeps rho(A) in the regime where the native 60-iteration shift-invert
# GMRES converges (see the module docstring). TE10 cutoff of a 22.86 mm
# guide is c/(2a) ~ 6.56 GHz, so 9-12 GHz is clean single mode.
A_WG, B_WG = 22.86 * MM, 10.16 * MM
LENGTH = 20.0 * MM
MAXH = 9.0 * MM
ORDER = 1
BAND = np.linspace(9.0e9, 12.0e9, 13)

# Multipole shift placement: four shifts spanning the band, two
# shift-invert applications per shift per port. Explored against FD; this
# is the configuration that lands |S21| within ~2-3% of FD.
SHIFTS_HZ = [9.0e9, 10.0e9, 11.0e9, 12.0e9]
N_SHIFT_STEPS = 3
R_REQUEST = 300            # generous; the build deflates to r ~ 48

# Honest gates (see module docstring for why |S11| is loose).
S21_DEV_BOUND = 0.04       # max ||S21|_rom - |S21|_fd| over the band
S21_PASSBAND_LOWER = 0.90  # the guide is matched: |S21| ~ 1 in band
S11_DEV_BOUND = 0.25       # ROM reflection floor; informative gate
SIGMA_MAX_BOUND = 1.20     # upwind non-SPRIM passivity overshoot


def _build_geometry():
    """The analytically-clean WR-90 guide shared by FD and TD."""
    g = rf.Geometry(maxh=MAXH)
    air = g.box(A_WG, B_WG, LENGTH, material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
    )
    g.mesh()
    return g


@slow
def test_multipole_rom_matches_fd_sparams(report):
    """Multi-shift rational-Krylov ROM vs the FD solver across the band.

    Builds the multipole ROM, sweeps its S-matrix over 9-12 GHz, and
    asserts |S21| tracks the FD reference to a few percent with a
    passband |S21| ~ 1, the matched-guide signature. |S11| is recorded
    against FD but only loosely gated (the ROM carries a reflection floor
    FD does not -- see the module docstring).
    """
    report.note(
        "Multipole ROM (multi-shift rational Krylov) vs ProblemFD on a "
        "short WR-90 X-band guide (order=1, matched TE10 ports both ends). "
        "Four shifts at 9/10/11/12 GHz, n_shift_steps=3. Gate: max |S21| "
        "deviation ROM-vs-FD under 4%, passband |S21| >= 0.90. |S11| is "
        "recorded but loosely gated -- the ROM carries a ~0.1-0.2 "
        "reflection floor the FD reference (|S11| ~ 1e-3) does not."
    )

    g = _build_geometry()
    s_fd = rf.ProblemFD(g).sweep(BAND).sparams

    ptd = rf.ProblemTD(g, order=ORDER, flux="upwind")
    f_cutoff = ptd.c / (2.0 * A_WG)
    mm = ptd.macromodel(
        r=R_REQUEST, shift_freqs_hz=SHIFTS_HZ, n_shift_steps=N_SHIFT_STEPS
    )
    s_rom = mm.sweep(BAND)

    s21_rom, s21_fd = np.abs(s_rom[:, 1, 0]), np.abs(s_fd[:, 1, 0])
    s11_rom, s11_fd = np.abs(s_rom[:, 0, 0]), np.abs(s_fd[:, 0, 0])
    d21 = float(np.max(np.abs(s21_rom - s21_fd)))
    d11 = float(np.max(np.abs(s11_rom - s11_fd)))
    mean_d21 = float(np.mean(np.abs(s21_rom - s21_fd)))
    s21_min = float(s21_rom.min())

    print(f"  n_dof={ptd.n_dof}, reduced r={mm.r}, "
          f"TE10 cutoff {f_cutoff/1e9:.2f} GHz")
    print(f"  FD  |S21|: {np.round(s21_fd, 3)}")
    print(f"  ROM |S21|: {np.round(s21_rom, 3)}")
    print(f"  max |S21| dev ROM vs FD: {d21:.3f} (mean {mean_d21:.3f})")
    print(f"  max |S11| dev ROM vs FD: {d11:.3f}")

    report.plot_sparams(
        BAND, {"ROM (multi-shift)": s_rom, "FD": s_fd},
        entries=[(1, 1), (2, 1)],
        caption="solid = multipole ROM, dashed = FD",
        title="WR-90 multipole ROM vs FD",
    )
    report.metric("reduced order r", mm.r, unit="states",
                  detail=f"requested {R_REQUEST}, deflated to realised r")
    report.metric("max |S21| dev ROM vs FD", d21, bound=S21_DEV_BOUND,
                  detail="max over band of ||S21|_rom - |S21|_fd|")
    report.metric("mean |S21| dev ROM vs FD", mean_d21,
                  detail="mean over band")
    report.metric("min |S21| in band", s21_min, lower=S21_PASSBAND_LOWER,
                  detail="matched guide: passband transmission ~ 1")
    report.metric("max |S11| dev ROM vs FD", d11, bound=S11_DEV_BOUND,
                  detail="ROM reflection floor; FD |S11| ~ 1e-3 (loose gate)")
    report.table(
        "ROM vs FD per frequency",
        ["f [GHz]", "|S21| ROM", "|S21| FD", "|S11| ROM", "|S11| FD"],
        [[BAND[k] / 1e9, s21_rom[k], s21_fd[k], s11_rom[k], s11_fd[k]]
         for k in range(len(BAND))],
    )

    assert d21 < S21_DEV_BOUND, (
        f"|S21| ROM-vs-FD deviation {d21:.3f} exceeds {S21_DEV_BOUND}"
    )
    assert s21_min >= S21_PASSBAND_LOWER, (
        f"passband |S21| dipped to {s21_min:.3f} < {S21_PASSBAND_LOWER}"
    )
    assert d11 < S11_DEV_BOUND, (
        f"|S11| ROM-vs-FD deviation {d11:.3f} exceeds {S11_DEV_BOUND}"
    )


@slow
def test_multipole_beats_plain(report):
    """The multipole's reason to exist: plain block-Krylov misses the
    in-band transmission, multi-shift hits it.

    Plain `span{B, A B, ...}` is dominated by the DG spectral radius and
    never reaches the 10 GHz physics, so its |S21| stays near 0;
    multi-shift's per-band shift-invert basis recovers |S21| ~ 1.
    """
    report.note(
        "Plain block-Krylov vs multi-shift on the same WR-90 guide. Plain "
        "Krylov is biased toward the DG spectral radius and never reaches "
        "the in-band 9-12 GHz transmission (|S21| ~ 0); the multipole's "
        "per-band shift-invert basis recovers it. This is the physical "
        "justification for the multi-shift path."
    )

    g = _build_geometry()
    s_fd = rf.ProblemFD(g).sweep(BAND).sparams
    ptd = rf.ProblemTD(g, order=ORDER, flux="upwind")

    mm_plain = ptd.macromodel(r=120)
    mm_multi = ptd.macromodel(
        r=R_REQUEST, shift_freqs_hz=SHIFTS_HZ, n_shift_steps=N_SHIFT_STEPS
    )
    s21_plain = np.abs(mm_plain.sweep(BAND)[:, 1, 0])
    s21_multi = np.abs(mm_multi.sweep(BAND)[:, 1, 0])
    s21_fd = np.abs(s_fd[:, 1, 0])

    plain_max = float(s21_plain.max())
    multi_min = float(s21_multi.min())
    print(f"  plain |S21| max over band: {plain_max:.3f} (FD ~ 1)")
    print(f"  multi |S21| min over band: {multi_min:.3f}")

    report.plot_xy(
        BAND / 1e9,
        {"plain block-Krylov": s21_plain,
         "multi-shift (multipole)": s21_multi,
         "FD": s21_fd},
        xlabel="frequency  [GHz]", ylabel="|S21|",
        title="|S21|: plain vs multipole vs FD",
        caption="plain Krylov misses the in-band physics; multipole recovers it",
    )
    report.metric("plain |S21| max", plain_max, bound=0.2,
                  detail="plain Krylov fails to reach in-band transmission")
    report.metric("multi-shift |S21| min", multi_min, lower=0.9,
                  detail="multipole reproduces the matched passband")

    # The multipole must clearly beat plain: in-band transmission is
    # recovered (>0.9) where plain stays near zero (<0.2).
    assert plain_max < 0.2, (
        f"plain Krylov |S21| reached {plain_max:.3f}; expected it to miss "
        f"the in-band physics (< 0.2)"
    )
    assert multi_min >= 0.9 > plain_max, (
        f"multipole |S21| min {multi_min:.3f} did not clearly beat plain "
        f"max {plain_max:.3f}"
    )


@slow
def test_state_space_export_consistent_with_evaluate(report, tmp_path):
    """The `state_space()` export is a lossless realisation of `evaluate()`.

    Checks: real/finite matrices with the documented shapes; the
    eigenvalues of A as physical angular frequencies; .npz/.mat
    round-trip; and -- the key consistency proof -- the S-matrix rebuilt
    by hand from (A, B, C_e, C_h, port_impedances) per the
    forward/backward modal split equals `mm.evaluate()` across the band.
    """
    report.note(
        "state_space() consistency: reconstruct S(f) by hand from the "
        "exported (A, B, C_e, C_h) and the per-port impedances "
        "[a = (z_E + Z z_H)/2, b = (z_E - Z z_H)/2, S = b/a, "
        "H = C (jwI - A)^-1 B] and compare to mm.evaluate() over the band. "
        "Also checks shapes/finiteness, A's eigenvalues as physical "
        "resonances, and a .npz/.mat round-trip."
    )

    g = _build_geometry()
    ptd = rf.ProblemTD(g, order=ORDER, flux="upwind")
    mm = ptd.macromodel(
        r=R_REQUEST, shift_freqs_hz=SHIFTS_HZ, n_shift_steps=N_SHIFT_STEPS
    )
    ss = mm.state_space()  # physical=True
    r, n = ss.r, ss.n_ports

    # 1. Shapes, realness, finiteness.
    assert ss.A.shape == (r, r)
    assert ss.B.shape == (r, n)
    assert ss.C_e.shape == (n, r)
    assert ss.C_h.shape == (n, r)
    assert ss.D.shape == (n, n)
    for name, M in (("A", ss.A), ("B", ss.B), ("C_e", ss.C_e),
                    ("C_h", ss.C_h), ("D", ss.D)):
        assert np.isrealobj(M), f"{name} must be real"
        assert np.all(np.isfinite(M)), f"{name} must be finite"
    assert np.allclose(ss.D, 0.0), "impulse-Krylov model is strictly proper"

    # 2. Eigenvalues of A as physical resonances. The matched guide is
    #    non-resonant, so we do not require a discrete in-band pole; we
    #    assert the spectrum is physical (damped, with structure
    #    bracketing the band) and report what it contains.
    ev = np.linalg.eigvals(ss.A)
    f_im_ghz = np.abs(ev.imag) / (2.0 * np.pi) / 1e9
    assert np.all(ev.real <= 1e-3 * np.abs(ev.imag).max() + 1.0), (
        "physical (upwind) modes must be non-growing (Re lambda <~ 0)"
    )
    near_band = f_im_ghz[(f_im_ghz > 6.0) & (f_im_ghz < 15.0)]
    print(f"  r={r}, n_ports={n}")
    print(f"  A eigenvalues |Im|/2pi near band [6,15] GHz: "
          f"{np.round(np.sort(near_band), 2)}")
    report.metric("reduced order r", r, unit="states")
    report.metric("A eigenmodes near band [6,15] GHz", int(near_band.size),
                  detail="|Im(lambda)|/2pi in 6-15 GHz; non-resonant guide")
    report.plot_complex_plane(
        {"A eigenvalues (rad/s)": ev},
        title="state-space A eigenvalues (physical units)",
        xlabel="Re  [1/s]", ylabel="Im  [rad/s]",
    )

    # 3. .npz / .mat round-trip.
    from scipy.io import loadmat

    npz_path = tmp_path / "rom.npz"
    mat_path = tmp_path / "rom.mat"
    ss.to_npz(npz_path)
    ss.to_mat(mat_path)
    z = np.load(npz_path)
    mt = loadmat(str(mat_path))
    for k in ("A", "B", "C_e", "C_h", "D"):
        assert np.allclose(z[k], getattr(ss, k)), f"npz {k} round-trip"
        assert np.allclose(mt[k], getattr(ss, k)), f"mat {k} round-trip"
    print("  npz/.mat round-trip: OK")

    # 4. Manual S from the export vs evaluate() across the band. This is
    #    macromodel.rs::evaluate reproduced in Python:
    #      X = (jw I - A)^-1 B          (per driven port j)
    #      z_E = C_e X,  z_H = C_h X
    #      a_j = (z_E[j,j] + Z_j z_H[j,j]) / 2   (incident at driven port)
    #      b_i = (z_E[i,j] - Z_i z_H[i,j]) / 2   (scattered at port i)
    #      S[i,j] = b_i / a_j
    def manual_S(f_hz):
        w = 2.0 * np.pi * f_hz
        Z = mm.port_impedances(f_hz)
        X = np.linalg.solve(1j * w * np.eye(r) - ss.A, ss.B)  # r x n
        z_E = ss.C_e @ X                                       # n x n
        z_H = ss.C_h @ X
        S = np.zeros((n, n), dtype=complex)
        for j in range(n):
            a_j = 0.5 * (z_E[j, j] + Z[j] * z_H[j, j])
            for i in range(n):
                b_i = 0.5 * (z_E[i, j] - Z[i] * z_H[i, j])
                S[i, j] = b_i / a_j
        return S

    max_dev = 0.0
    for f in BAND:
        S_manual = manual_S(f)
        S_eval = mm.evaluate(f)
        max_dev = max(max_dev, float(np.max(np.abs(S_manual - S_eval))))
    print(f"  max |S_manual - S_evaluate| over band: {max_dev:.2e}")
    report.metric("max |S_manual - S_evaluate|", max_dev, bound=1e-9,
                  detail="hand-built S from state_space vs mm.evaluate "
                         "(export is lossless)")
    assert max_dev < 1e-9, (
        f"state_space export inconsistent with evaluate: {max_dev:.2e}"
    )


@slow
def test_multipole_rom_passivity(report):
    """Passivity proxy: sigma_max(S) over the band.

    For a lossless reciprocal 2-port sigma_max(S) <= 1. The plain /
    multi-shift builds are not structure-preserving (SPRIM), so a mild
    overshoot is expected and reported rather than asserted away.
    """
    report.note(
        "Passivity proxy sigma_max(S(f)) over the band for the multipole "
        "ROM. A lossless reciprocal 2-port satisfies sigma_max <= 1; the "
        "non-SPRIM upwind build shows a mild overshoot, gated loosely at "
        f"{SIGMA_MAX_BOUND} and reported honestly."
    )

    g = _build_geometry()
    ptd = rf.ProblemTD(g, order=ORDER, flux="upwind")
    mm = ptd.macromodel(
        r=R_REQUEST, shift_freqs_hz=SHIFTS_HZ, n_shift_steps=N_SHIFT_STEPS
    )
    s_rom = mm.sweep(BAND)
    sigma_max = np.array(
        [np.linalg.svd(s_rom[k], compute_uv=False).max()
         for k in range(len(BAND))]
    )
    smax = float(sigma_max.max())
    print(f"  max sigma_max(S) over band: {smax:.4f}")

    report.plot_passivity(
        BAND, sigma_max,
        title="multipole ROM passivity  sigma_max(S)",
        caption="passive bound at 1.0; non-SPRIM upwind overshoot reported",
    )
    report.metric("max sigma_max(S)", smax, bound=SIGMA_MAX_BOUND,
                  detail="passivity proxy; <= 1 ideal, mild overshoot "
                         "expected for non-SPRIM upwind build")
    assert np.all(np.isfinite(sigma_max))
    assert smax < SIGMA_MAX_BOUND, (
        f"sigma_max(S) overshoot {smax:.4f} exceeds {SIGMA_MAX_BOUND}"
    )


@slow
def test_multipole_rom_order2_needs_larger_gmres_cap(report):
    """At order=2 the larger spectral radius needs a bigger GMRES basis.

    Regression gate for the exposed `gmres_max_iter` knob: on the same
    WR-90 guide at order=2 the default shift-invert GMRES cap (60) fails to
    converge and the multipole ROM's in-band |S21| collapses to ~0;
    raising the cap restores the physics (|S21| ~ 1). This is the order=2
    path that the module docstring promises the knob unlocks.
    """
    report.note(
        "Order=2 WR-90: the multipole shift-invert GMRES needs a Krylov "
        "basis large enough to resolve the mid-spectrum shift. With the "
        "default cap (60) the in-band |S21| collapses to ~0; raising "
        "gmres_max_iter to 250 recovers TE10 transmission (|S21| ~ 1). "
        "Demonstrates the gmres_max_iter knob."
    )

    g = _build_geometry()
    ptd = rf.ProblemTD(g, order=2, flux="upwind")
    report.metric("order-2 n_dof", ptd.n_dof, detail="state DOFs at order 2")

    mm_lo = ptd.macromodel(
        r=R_REQUEST, shift_freqs_hz=SHIFTS_HZ, n_shift_steps=N_SHIFT_STEPS,
        gmres_max_iter=60,
    )
    s21_lo = np.abs(mm_lo.sweep(BAND)[:, 1, 0])

    mm_hi = ptd.macromodel(
        r=R_REQUEST, shift_freqs_hz=SHIFTS_HZ, n_shift_steps=N_SHIFT_STEPS,
        gmres_max_iter=250,
    )
    s_hi = mm_hi.sweep(BAND)
    s21_hi = np.abs(s_hi[:, 1, 0])
    print(f"  order=2 |S21| in-band: cap60 max={s21_lo.max():.3f}  "
          f"cap250 min={s21_hi.min():.3f}")

    report.plot_xy(
        BAND / 1e9,
        {"cap=60 (default)": s21_lo, "cap=250": s21_hi},
        xlabel="frequency [GHz]", ylabel="|S21|",
        title="order=2 multipole ROM: GMRES cap vs in-band |S21|",
        caption="default cap collapses to ~0; the raised cap recovers |S21|~1",
    )
    report.metric("|S21| max (cap=60)", float(s21_lo.max()), bound=0.2,
                  detail="default cap fails to converge at order 2")
    report.metric("|S21| min in-band (cap=250)", float(s21_hi.min()),
                  lower=0.90,
                  detail="raised cap recovers physical TE10 transmission")

    assert s21_lo.max() < 0.2, (
        f"order=2 default-cap |S21| unexpectedly non-zero: {s21_lo.max():.3f}"
    )
    assert s21_hi.min() > 0.90, (
        f"order=2 raised-cap |S21| did not recover: min {s21_hi.min():.3f}"
    )
