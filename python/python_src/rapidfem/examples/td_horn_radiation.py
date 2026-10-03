"""Time-domain DGTD: a pyramidal horn radiating a pulse into free space.

A WR-90 feed launches a TE10 pulse into a flared pyramidal horn; the pulse
reaches the aperture and radiates out as a beam into the surrounding air,
absorbed by a PML slab in the main-beam direction and first-order ABCs on
the side walls. The 3-D animation shows the wavefront leaving the aperture
and forming the directive beam; the port-amplitude plot shows the incident
pulse and the small aperture reflection that returns to the feed.

Open-boundary radiation: the modal `rf.RectWaveguidePort` feeds the horn,
`g.loft` builds the flare, and `rf.PML` + `rf.ABC` terminate the air box so
the radiated field leaves without ringing back into the domain.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

mm = 1e-3
WGA, WGB = 22.86 * mm, 10.16 * mm   # WR-90 feed cross-section
LFEED = 15.0 * mm                   # feed waveguide length
LHORN = 50.0 * mm                   # horn flare length
WH, HH = 30.0 * mm, 22.0 * mm       # aperture width / height
LPAD_BEAM = 88.0 * mm               # +x pad: roomy for the diverging beam to reach the PML cleanly
LPAD_SIDE = 30.0 * mm               # y/z pad: tighter, the side ABC just terminates near-field leakage
PML_T = 15.0 * mm                   # PML slab thickness (beam direction)
F0 = 10.0e9                         # drive frequency

MAXH = rf.lambda_maxh(f_max=11.0e9, per_lambda=8)        # horn / feed metals
MAXH_AIR = rf.lambda_maxh(f_max=11.0e9, per_lambda=3)    # outer air padding

# %% Geometry: feed + lofted horn flare in a PML/ABC-terminated air box
AIR_X0, AIR_X1 = -LFEED, LHORN + LPAD_BEAM
AIR_Y0, AIR_Y1 = -WH / 2 - LPAD_SIDE, WH / 2 + LPAD_SIDE
AIR_Z0, AIR_Z1 = -HH / 2 - LPAD_SIDE, HH / 2 + LPAD_SIDE

# Global cap is the *coarse* outer-air size; the horn flare and feed are
# refined locally below. The big +x pad (88 mm) gives the diverging beam
# room to settle before hitting the PML; the y/z pad (30 mm ≈ 1λ at 10
# GHz) is the standard ABC stand-off, no benefit from going bigger.
g = rf.Geometry(maxh=MAXH_AIR)
air = g.box(AIR_X1 - AIR_X0, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
            position=(AIR_X0, AIR_Y0, AIR_Z0), material=rf.Air())
pml_xp = g.box(PML_T, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
               position=(AIR_X1, AIR_Y0, AIR_Z0), material=rf.Air(),
               maxh=2 * MAXH)
feed = g.box(LFEED, WGA, WGB, position=(-LFEED, -WGA / 2, -WGB / 2),
             material=rf.Air(), maxh=WGB / 3)
throat = g.polygon([
    (0, -WGA / 2, -WGB / 2), (0, WGA / 2, -WGB / 2),
    (0, WGA / 2, WGB / 2), (0, -WGA / 2, WGB / 2),
])
aperture = g.polygon([
    (LHORN, -WH / 2, -HH / 2), (LHORN, WH / 2, -HH / 2),
    (LHORN, WH / 2, HH / 2), (LHORN, -WH / 2, HH / 2),
])
horn = g.loft(throat, aperture, material=rf.Air(), maxh=MAXH)
g.fragment(air, feed, horn, pml_xp)

p_in = rf.RectWaveguidePort(feed.faces.min(axis="x"), mode=(1, 0))
rf.PEC(feed.faces.min(axis="y"), feed.faces.max(axis="y"),
       feed.faces.min(axis="z"), feed.faces.max(axis="z"))
rf.PEC(*horn.faces.where(lambda c, b: 1e-6 < c[0] < LHORN - 1e-6))
rf.PML(pml_xp, direction=(1, 0, 0), inner_face=LHORN + LPAD_BEAM, thickness=PML_T)
rf.PEC(*pml_xp.faces.outer)
rf.ABC(*air.faces.outer.unassigned)
g.mesh()
rf.show(g)

# %% Build the time-domain problem
ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DGTD pyramidal horn - {ptd.n_tets} tets, {ptd.n_dofs} state DOFs, "
      f"drive {F0 / 1e9:.1f} GHz")

# %% Drive the feed and watch the pulse radiate out of the aperture
pulse = rf.GaussianPulse(t0=80e-12, tau=20e-12, f0=F0)
traj = ptd.transient(
    port=p_in, waveform=pulse, dt=2e-12, steps=340,
    method="explicit", device="gpu",
)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"horn transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")

# %% Visualise: the wavefront leaving the aperture into the PML/ABC box
rf.show(traj)

# %% Port modal amplitude: incident pulse and the small aperture reflection
rf.show(ptd.port_signals(traj, [p_in], dt=2e-12, labels=["feed port"]))
