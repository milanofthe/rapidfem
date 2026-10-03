"""Printed inverted-F antenna (IFA) on FR-4 with adaptive mesh refinement.

Antenna trace and ground plane are coplanar on the same copper layer of a
1 mm FR-4 board. Adapted from EMerge ``demo21`` and TI App Note SWRA117D.

The interesting twist here: the initial mesh is **deliberately coarse**,
trace plates inherit the global wavelength-based size cap and the
substrate is only pinned by its thickness. At ~8 mm global ``maxh`` the
0.5-0.9 mm trace widths are barely a fraction of one tet, so the first
sweep gets the resonance frequency wrong by a couple of hundred MHz.

Two AMR passes fix that without the user having to hand-tune
``TRACE_MAXH``: :meth:`rapidfem.ProblemFD.element_errors` returns the
Monk residual indicator at the resonance frequency, the top-η tet
centroids drive :meth:`rapidfem.Geometry.refine_near_points`, and the
mesh is regenerated with a finer h locally around the trace edges. The
loop is user-driven (no wrapper) so the convergence story is visible
print-by-print.
"""

# %% Parameters
import numpy as np
import rapidfem as rf

mm = 1e-3

L1 = 3.94 * mm
L2 = 2.47 * mm
L3 = 4.76 * mm
L4 = 2.64 * mm
L5 = 1.77 * mm
L6 = 4.90 * mm
W1 = 0.90 * mm
W2 = 0.50 * mm
D1 = 0.50 * mm
D4 = 0.50 * mm
D5 = 0.65 * mm

TRACE_W = D1 + L3 + L5 + L2 + L5 + L2 + W2

SUB_H = 1.0 * mm
ER_SUB = 4.4
GROUND_L = 30.0 * mm

PAD = 35.0 * mm

FREQUENCIES = np.linspace(2.0e9, 3.0e9, 21)    # 50 MHz grid around the
                                               # designed 2.45 GHz resonance
F0 = 2.45e9

MAXH = rf.lambda_maxh(f_max=3.0e9)
TRACE_MAXH_COARSE = 1.5 * mm   # rough hint so the trace exists, but is still
                               # under-resolved by ~3x vs the production size

# AMR knobs.
N_AMR_ITERATIONS = 2
AMR_THETA = 0.6            # Doerfler-marking fraction (aggressive, the
                           # eta distribution on the meander is very spiky)
AMR_REFINE_RATIO = 0.5     # new tet size = marked h_k * AMR_REFINE_RATIO


# %% Geometry + Materials (no per-trace TRACE_MAXH, no auto_refine_features,
#    the AMR loop below will catch the under-resolved trace edges instead)
SUB_X0, SUB_X1 = -D1, TRACE_W - D1
SUB_Y0, SUB_Y1 = -GROUND_L, L6 - D4 + W2
SUB_DX = SUB_X1 - SUB_X0
SUB_DY = SUB_Y1 - SUB_Y0

g = rf.Geometry(maxh=MAXH)

# Substrate refinement is set by thickness, not wavelength, at 3 GHz in FR-4
# lambda is ~50 mm but SUB_H = 1 mm, so the box would be ~1 cell thick at the
# wavelength cap. Pin the dielectric at ~1.5x thickness.
fr4 = rf.Dielectric(er=ER_SUB, maxh=1.5 * SUB_H)

sub = g.box(SUB_DX, SUB_DY, SUB_H, position=(SUB_X0, SUB_Y0, -SUB_H),
            material=fr4)

AIR_X0, AIR_X1 = SUB_X0 - PAD, SUB_X1 + PAD
AIR_Y0, AIR_Y1 = SUB_Y0 - PAD, SUB_Y1 + PAD
AIR_Z0, AIR_Z1 = -SUB_H - PAD, PAD
air = g.box(AIR_X1 - AIR_X0, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
            position=(AIR_X0, AIR_Y0, AIR_Z0),
            material=rf.Air())

# 12 trace segments of the meandered F.
ant_segs_xy = [
    (0,                          0,                W1, L6 - D4),
    (0,                          L6 - D4,          L3, W2),
    (W1 + D5,                    0.5 * mm,         W2, L6 - D4 - 0.5 * mm),
    (L3 - W2,                    L6 - D4 - L4,     W2, L4),
    (L3,                         L6 - D4 - L4,     L5, W2),
    (L3 + L5,                    L6 - D4 - L4,     W2, L4),
    (L3 + L5,                    L6 - D4,          L2, W2),
    (L3 + L5 + L2 - W2,          L6 - D4 - L4,     W2, L4),
    (L3 + L5 + L2,               L6 - D4 - L4,     L5, W2),
    (L3 + L5 + L2 + L5,          L6 - D4 - L4,     W2, L4),
    (L3 + L5 + L2 + L5,          L6 - D4,          L2, W2),
    (L3 + L5 + L2 + L5 + L2 - W2, L6 - D4 - L1,    W2, L1),
]
ant_plates = [g.xy_plate(w, h, position=(x, y, 0), maxh=TRACE_MAXH_COARSE)
              for x, y, w, h in ant_segs_xy]

port = g.xy_plate(W2, 0.5 * mm, position=(W1 + D5, 0, 0),
                  maxh=TRACE_MAXH_COARSE)
ground = g.xy_plate(SUB_DX, GROUND_L, position=(SUB_X0, SUB_Y0, 0))

g.fragment(air, sub, ground, port, *ant_plates)

# Physics
rf.LumpedPort(port, direction=(0, 1, 0), z0=50.0)
rf.PEC(*ant_plates, ground)
rf.ABC(*air.faces.outer)

rf.show(g)


# %% Initial mesh, coarse on purpose; trace edges are barely resolved.
g.mesh()
rf.show(g)


# %% AMR loop, print iter, n_tets, |S11|_min, f_res, deltas

print(f"\nAMR loop: {N_AMR_ITERATIONS + 1} sweeps, theta={AMR_THETA}, "
      f"refine_ratio={AMR_REFINE_RATIO}")

prev_s11min = None
prev_f_res = None
prob = None
result = None

for it in range(N_AMR_ITERATIONS + 1):
    prob = rf.ProblemFD(g)
    result = prob.sweep(FREQUENCIES)

    mags = np.array([abs(result.sparams[i, 0, 0])
                     for i in range(len(FREQUENCIES))])
    i_res = int(mags.argmin())
    f_res = FREQUENCIES[i_res]
    s11_min = mags[i_res]

    if prev_s11min is not None:
        ds = s11_min - prev_s11min
        df = (f_res - prev_f_res) / 1e6
        delta_str = f"  d|S11|={ds:+.4f}  df_res={df:+.1f} MHz"
    else:
        delta_str = "  (initial coarse)"

    print(f"iter {it}: tets={prob.n_tets:>6}  DOFs={prob.n_dofs:>6}  "
          f"|S11|_min={s11_min:.4f} @ {f_res / 1e9:.3f} GHz{delta_str}")

    if it == N_AMR_ITERATIONS:
        break  # last iteration just reports

    # Mark high-residual tets at the resonance frequency.
    errs = prob.element_errors(result, freq_idx=i_res, theta=AMR_THETA)
    if len(errs.marked) == 0:
        print("  no tets marked, stopping AMR")
        break
    hot = errs.tet_centroids[errs.marked]
    target_h = float(errs.h_k[errs.marked].mean() * AMR_REFINE_RATIO)
    print(f"  refining {len(errs.marked)} tets, target h = "
          f"{target_h * 1e3:.3f} mm")
    g.refine_near_points(hot, h=target_h)
    g.mesh()

    prev_s11min = s11_min
    prev_f_res = f_res


# %% Final result + far-field at F0
rf.show(prob)
rf.show(result)

print(f"\nfinal mesh: DOFs={prob.n_dofs}, tets={prob.n_tets}")

mags = [abs(result.sparams[i, 0, 0]) for i in range(len(FREQUENCIES))]
fi_min = int(min(range(len(mags)), key=lambda i: mags[i]))
print(f"|S11| min: {mags[fi_min]:.4f} @ {FREQUENCIES[fi_min] / 1e9:.3f} GHz")

fi0 = int(min(range(len(FREQUENCIES)), key=lambda i: abs(FREQUENCIES[i] - F0)))
pattern = prob.farfield(result, freq_idx=fi0, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"Far-field @ {FREQUENCIES[fi0] / 1e9:.2f} GHz: "
          f"D = {pattern.peak_directivity_dbi:.2f} dBi, "
          f"G = {pattern.peak_gain_dbi:.2f} dBi")
