"""Time-domain DGTD: a TE10 pulse launched into a WR-90 rectangular guide.

The input `rf.RectWaveguidePort` injects the analytic TE10 mode of a WR-90
guide as a band-limited pulse; it travels down the guide and leaves through
the matched output port with no reflection. The 3-D animation shows the
half-sine TE10 field profile propagating along the guide in time.

This is the modal-port injection path: `transient(port=...)` drives the
operator with the port's own mode pattern (`dy/dt = A·y + b·g(t)`)
rather than a point source. Both
guide ends are characteristic (absorbing) ports, so the pulse enters and
exits cleanly; PEC walls are the TD operator's default on every non-port
face.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

A, B, L = 22.86e-3, 10.16e-3, 60.0e-3   # WR-90 width, height, guide length [m]
F0 = 10.0e9                             # drive centre frequency (single-mode band)
MAXH = rf.lambda_maxh(f_max=12.0e9)     # ~2.1 mm, air λ/12 at the band edge

# %% Geometry: an air-filled WR-90 guide, modal ports on the two ends
g = rf.Geometry(maxh=MAXH)
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0), material=rf.Air())
p_in = rf.RectWaveguidePort(air.faces.min(axis="z"))
p_out = rf.RectWaveguidePort(air.faces.max(axis="z"))
rf.PEC(*air.faces.unassigned)
g.mesh()
rf.show(g)

# %% Build the time-domain problem
ptd = rf.ProblemTD(g, order=2, flux="upwind")
fc = ptd.c / (2.0 * A)
print(f"DGTD WR-90 guide - {ptd.n_tets} tets, {ptd.n_dofs} state DOFs, "
      f"TE10 cutoff {fc / 1e9:.2f} GHz, drive {F0 / 1e9:.1f} GHz")

# %% Drive the TE10 mode at the input port and watch the pulse propagate
pulse = rf.GaussianPulse(t0=90e-12, tau=22e-12, f0=F0)
traj = ptd.transient(
    port=p_in, waveform=pulse, dt=3e-12, steps=320,
    method="explicit", device="gpu",
)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"TE10 transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")

# %% Visualise: the TE10 mode travelling down the guide to the matched port
rf.show(traj)

# %% Port modal amplitudes over time (incident at input, transmitted at output)
rf.show(ptd.port_signals(traj, [p_in, p_out], dt=3e-12,
                         labels=["input port", "output port"]))
