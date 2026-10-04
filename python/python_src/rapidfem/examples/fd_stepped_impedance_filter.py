"""Microstrip stepped-impedance low-pass filter, wave-port driven.

A 7-section trace of alternating wide / narrow strips on a thin substrate
acts as a classic Richards-style LPF. Cutoff around 2 GHz. Adapted from
EMerge's ``demo1_stepped_imp_filter.py``.

Both ends are excited by a full-vector wave port, so the quasi-TEM mode is
de-embedded directly. The single-mode projection is only energy-conserving
while the lowest transverse box resonance of the *port cross-section* stays
above the sweep top. Rather than shrink the ground plane to push that
resonance up (which would distort the filter), the port here is a boxed
aperture: a small ``PORT_W`` x ``PORT_H`` window over the trace at each
end, cut into the otherwise full-width end face by a fragmented plate. The
aperture is narrow enough to keep its transverse resonance above the 8 GHz
stop-band while the ground plane itself stays full-width, and the end face
outside the window stays PEC (a shielding wall with a port window).
"""

# %% Parameters
import math
import numpy as np
import rapidfem as rf

mm = 1e-3
mil = 0.0254 * mm

LENGTHS_MIL = [400, 660, 660, 660, 660, 660, 400]
WIDTHS_MIL  = [ 50, 128,   8, 224,   8, 128,  50]

SUB_H = 62 * mil
ER_SUB = 2.2
AIR_H = 15 * mm
PAD_Y = 12 * mm

# Boxed wave-port aperture. The ports do not span the full ground-plane
# width: a small rectangle over the trace at each end keeps the aperture's
# lowest transverse box resonance above the 8 GHz sweep top, while the
# ground plane itself stays full-width. PORT_W must hold the quasi-TEM mode
# (a few substrate heights of air each side of the widest strip) yet stay
# narrow enough that c / (2 * PORT_W * sqrt(ER_SUB)) sits above the band.
PORT_W = 12.0 * mm
PORT_H = 6.0 * SUB_H         # aperture height: substrate + fringing air above

FREQUENCIES = np.linspace(0.2e9, 8.0e9, 41)
MAXH = rf.lambda_maxh(f_max=8.0e9)


# %% Geometry + Materials
LENGTHS = [L * mil for L in LENGTHS_MIL]
WIDTHS  = [W * mil for W in WIDTHS_MIL]
total_L = sum(LENGTHS)
sub_W   = max(WIDTHS) + 2 * PAD_Y

g = rf.Geometry(maxh=MAXH)

# Substrate is RO-style 62-mil. The narrow 8-mil trace section drives the
# surface refinement separately via auto_refine_features, which is where the
# accuracy sits (coarser traces move |S| by up to 0.06); the substrate
# volume at 2/3 of its thickness keeps the mesh at a fifth of the 1/3 one
# (the full thickness moves |S| by 0.04) and still lets the wave-port
# eigensolve resolve the quasi-TEM mode.
ro = rf.Dielectric(er=ER_SUB, maxh=2 * SUB_H / 3)

sub = g.box(total_L, sub_W, SUB_H, position=(-total_L / 2, -sub_W / 2, 0),
            material=ro)
air = g.box(total_L, sub_W, AIR_H, position=(-total_L / 2, -sub_W / 2, SUB_H),
            material=rf.Air())

x_cursor = -total_L / 2
trace_plates: list = []
for L_seg, W_seg in zip(LENGTHS, WIDTHS):
    plate = g.xy_plate(L_seg, W_seg, position=(x_cursor, -W_seg / 2, SUB_H))
    trace_plates.append(plate)
    x_cursor += L_seg

# Vertical cutting plate at each x-end, centred on the line and spanning the
# substrate plus PORT_H of air above it. Fragmenting it into the end faces
# imprints the aperture outline, splitting the full-width end face into the
# narrow aperture window plus the side strips that flank it.
x_lo, x_hi = -total_L / 2, total_L / 2
port_cut_in = g.plate(p0=(x_lo, -PORT_W / 2, 0),
                      width=(0, PORT_W, 0), height=(0, 0, PORT_H))
port_cut_out = g.plate(p0=(x_hi, -PORT_W / 2, 0),
                       width=(0, PORT_W, 0), height=(0, 0, PORT_H))

g.fragment(sub, air, *trace_plates, port_cut_in, port_cut_out)

# Trace sections + ground plane on one PEC object so the wave-port
# eigensolve marks the internal conductor nodes via pec=[pec_strip].
pec_strip = rf.PEC(sub.faces.min(axis="z"), *trace_plates)


def _aperture(obj, x_plane):
    """The boxed-port sub-face of ``obj`` at ``x_plane``: degenerate in x,
    on the end plane, and bounded by the aperture in y (the flanking side
    strips run out to +-sub_W/2, so they fail the y-bounds test)."""
    # 1e-6 m tolerance, the same slack rf.ABC(*faces.outer) relies on.
    return obj.faces.where(
        lambda c, b: abs(b[3] - b[0]) < 1e-6
        and abs(b[0] - x_plane) < 1e-6
        and b[1] >= -PORT_W / 2 - 1e-6
        and b[4] <= PORT_W / 2 + 1e-6)


# Wave ports on the boxed apertures: substrate + air aperture sub-faces at
# each end. f0 at band centre fixes beta = n_eff(f0) * k0 for the sweep.
F0 = 0.5 * (FREQUENCIES[0] + FREQUENCIES[-1])
rf.WavePort(_aperture(sub, x_lo), _aperture(air, x_lo),
            f0=F0, mode_kind="auto", pec=[pec_strip])
rf.WavePort(_aperture(sub, x_hi), _aperture(air, x_hi),
            f0=F0, mode_kind="auto", pec=[pec_strip])

# Open the enclosure: first-order ABC on the lateral y-walls (substrate and
# air) and the air top. The x-end faces outside the apertures stay PEC (a
# shielding end wall with a port window, the classic wave-port backing).
rf.ABC(sub.faces.min(axis="y"), sub.faces.max(axis="y"),
       air.faces.min(axis="y"), air.faces.max(axis="y"),
       air.faces.max(axis="z"))

rf.show(g)


# %% Mesh
g.auto_refine_features(base_maxh=MAXH, min_maxh=0.3e-3)
g.mesh()
rf.show(g)


# %% Problem + Sweep
prob = rf.ProblemFD(g)
result = prob.sweep(FREQUENCIES)
rf.show(prob)
rf.show(result)

s21_db = [20 * math.log10(max(abs(result.sparams[i, 1, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
cutoff_idx = next((i for i, db in enumerate(s21_db) if db < -3), len(FREQUENCIES) - 1)
print(f"|S21| 3-dB cutoff near {FREQUENCIES[cutoff_idx]/1e9:.2f} GHz "
      f"(stop-band floor {min(s21_db):.1f} dB)")
