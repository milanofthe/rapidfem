"""Time-domain DGTD: a waveguide T-junction power divider.

A TE10 pulse launched into the stem of an H-plane waveguide tee travels up
to the junction and splits into the two crossbar arms. All three guide
ends are characteristic `rf.RectWaveguidePort`s, so the pulse enters at the
stem and the two halves leave cleanly through the arm ports. The 3-D
animation shows the split in time; the port-amplitude plot shows one input
pulse and two equal (3 dB down) output pulses.

H-plane tee: all three guides lie in the xy-plane with the TE10 field
along z, so the mode orientation is consistent through the junction. The
tee is tiled from face-adjacent boxes that `g.fragment` stitches
conformally; the box interfaces at the hub stay interior (the DG flux
carries them), and the outer walls are PEC by the operator's default.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

mm = 1e-3
W = 20.0 * mm            # in-plane guide width (sets the TE10 cutoff)
H = 10.0 * mm            # guide height (z); H < W keeps TE10 dominant
L_ARM = 40.0 * mm        # each crossbar output arm
L_STEM = 40.0 * mm       # the input stem
F0 = 10.0e9              # drive frequency (above the ~7.5 GHz TE10 cutoff)

# %% Geometry: an H-plane tee tiled from face-adjacent boxes
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12.0e9))
air = rf.Air()
hub = g.box(W, W, H, position=(-W / 2, -W / 2, -H / 2), material=air)
stem = g.box(W, L_STEM, H, position=(-W / 2, -W / 2 - L_STEM, -H / 2),
             material=air)
left = g.box(L_ARM, W, H, position=(-W / 2 - L_ARM, -W / 2, -H / 2),
             material=air)
right = g.box(L_ARM, W, H, position=(W / 2, -W / 2, -H / 2), material=air)
g.fragment(hub, stem, left, right)

p_in = rf.RectWaveguidePort(stem.faces.min(axis="y"))
p_left = rf.RectWaveguidePort(left.faces.min(axis="x"))
p_right = rf.RectWaveguidePort(right.faces.max(axis="x"))
# Outer walls are PEC by default; the hub box interfaces stay interior.
g.mesh()
rf.show(g)

# %% Build the time-domain problem
ptd = rf.ProblemTD(g, order=2, flux="upwind")
fc = ptd.c / (2.0 * W)
print(f"DGTD H-plane tee - {ptd.n_tets} tets, {ptd.n_dofs} state DOFs, "
      f"TE10 cutoff {fc / 1e9:.2f} GHz, drive {F0 / 1e9:.1f} GHz")

# %% Drive the stem port and watch the pulse split into the two arms
pulse = rf.GaussianPulse(t0=90e-12, tau=22e-12, f0=F0)
traj = ptd.transient(
    port=p_in, waveform=pulse, dt=3e-12, steps=360,
    method="explicit", device="gpu",
)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"power-divider transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")

# %% Visualise: the pulse dividing at the junction into the two arms
rf.show(traj)

# %% Port modal amplitudes: one input pulse, two equal (3 dB) output pulses
rf.show(ptd.port_signals(traj, [p_in, p_left, p_right], dt=3e-12,
                         labels=["stem (input)", "left arm", "right arm"]))
