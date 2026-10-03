"""Coupled-line bandpass filter on FR4-grade substrate, ported from
``EMerge/examples/demo3_coupled_line_filter.py``.

Six cascaded parallel-coupled microstrip sections form a bandpass around
5.7 GHz on a 0.508 mm substrate (er=3.55). The layout is the classic
half-wave coupled-line meander: each galvanically connected strand carries
the bottom edge of one section and the top edge of the next, and adjacent
strands are edge-coupled across their gap.

Geometry is built entirely on the native rapidfem API: every horizontal
strip is an ``xy_plate``, the strips of one strand are merged into a single
conductor with ``g.fuse`` (the native union, so each strand reaches the
mesher as one watertight PEC face), and ``g.fragment`` makes the trace,
substrate, and air conformal.

Wave ports at the substrate's x-min / x-max cross-sections drive both ends
with the microstrip's quasi-TEM mode (``mode_kind="auto"``). A first-order
ABC on the lateral y-walls and the air top opens the enclosure so the
structure radiates instead of trapping energy in a lossless cavity (see the
boundary-condition note below for why that matters). The sweep covers 5.2 - 6.2 GHz, 21 points and reproduces the
EMerge pass-band shape centred near 5.7 GHz; the open lateral walls sit
close to the trace, so the modelled insertion loss (~-5 to -6 dB) runs
above EMerge's -1.1 dB reference by the extra lateral-radiation loss
(widen the y-pad to trade tet count for less of it).
"""

# %% Parameters
import math

import numpy as np

import rapidfem as rf

mm = 1e-3
mil = 0.0254 * mm

# Substrate (EMerge demo3 uses th = 20 mil ≈ 0.508 mm, er = 3.55)
SUB_H  = 20 * mil
ER_SUB = 3.55
TAND   = 0.001

# Filter geometry (EMerge demo3 values, in mil), symmetric 6-section bandpass
w0 = 37.0 * mil               # input/output line width
l0 = 100.0 * mil              # input/output line length
l1 = 314.22 * mil             # outer-section coupled length
l2 = 301.658 * mil            # inner-section coupled length
l3 = 300.589 * mil            # mid-section coupled length

WS = [v * mil for v in (18.8, 43.484, 44.331, 44.331, 43.484, 18.8)]   # section widths
GS = [v * mil for v in (9.63, 24.84, 41.499, 41.499, 24.84, 9.63)]     # section gaps

FREQUENCIES = np.linspace(5.2e9, 6.2e9, 21)
MAXH = rf.lambda_maxh(f_max=6.2e9, er_max=ER_SUB)

# Air/substrate envelope: no x-pad, the substrate's x-min and x-max faces
# ARE the wave-port cross-sections at the trace entry / exit (the trace
# starts exactly at x = 0 and ends at the rightmost segment). Tight y-pad
# (40 mil ≈ 1 mm) keeps the substrate width narrow enough that the lowest
# transverse-resonance mode of the 20 mm × 0.5 mm slab sits above the
# 6.2 GHz top of the sweep, without that, the wave-port single-mode
# projection misses energy carried by higher-order transverse modes and
# |S11|² + |S21|² runs over unity inside the pass-band.
PAD_X_MIL = 0
PAD_Y_MIL = 40
AIR_H_FACTOR = 4   # height = 4 * substrate thickness above and below


# %% Trace layout, replays the EMerge .straight()/.jump() chain
class _Trace:
    """Builds the demo3 coupled-line meander as axis-aligned rectangles.

    ``straight(L, w, dy)`` extends the current strand (one galvanically
    connected conductor) by ``L`` in +x, optionally switching width to ``w``
    and offsetting the centerline by ``dy``. ``jump(gap, side, reverse)``
    ends the current strand and starts a new one offset sideways by
    ``(w_old + w_new)/2 + gap`` and ``reverse`` backward, that pair of
    strips is one coupled-line section. ``strands`` is the result: a list of
    strands, each a list of ``(x_lo, y_lo, x_hi, y_hi)`` rectangles.
    """

    def __init__(self, x: float, y: float, width: float,
                 direction: tuple[float, float] = (1.0, 0.0)):
        self.x, self.y = x, y
        self.dx, self.dy = direction
        self.width = width
        self.strands: list[list[tuple[float, float, float, float]]] = [[]]

    def straight(self, length: float, width: float | None = None,
                 dy: float = 0.0) -> "_Trace":
        if width is not None:
            self.width = width
        # dy offsets the centerline perpendicular to travel (demo3 runs in +x,
        # so this is a pure y-shift that persists into the strand's new state).
        self.y += dy
        x0, y0 = self.x, self.y
        self.x += length * self.dx
        self.y += length * self.dy
        x_lo, x_hi = sorted((x0, self.x))
        self.strands[-1].append(
            (x_lo, y0 - self.width / 2, x_hi, y0 + self.width / 2))
        return self

    def jump(self, gap: float, side: str, reverse: float,
             width: float | None = None) -> "_Trace":
        new_w = self.width if width is None else width
        # Right-hand unit for direction (dx, dy) is (dy, -dx).
        right_x, right_y = self.dy, -self.dx
        q = -1.0 if side == "left" else 1.0
        offset = (new_w / 2 + self.width / 2 + gap) * q
        self.x = self.x - reverse * self.dx + right_x * offset
        self.y = self.y - reverse * self.dy + right_y * offset
        self.width = new_w
        self.strands.append([])   # break galvanic connection: new conductor
        return self


# Build the coupled-line trace following EMerge demo3 1:1. The dy convention:
# demo3 edge-aligns the outer four transitions (w0↔w1, w1↔w2, w5↔w6, w6↔w0)
# and uses dy=0 for the inner ones; the w5↔w6 transition is intentionally
# given the same dy as w1↔w2 in the original, kept here for parity.
dy_01 = abs(WS[0] - w0) / 2
dy_12 = abs(WS[1] - WS[0]) / 2

tr = _Trace(0.0, 140 * mil, w0)
(tr.straight(l0)                                      # input feedline
   .straight(l1 * 0.8)                                # transition stub
   .straight(l1, WS[0], dy=dy_01)                     # section 1 top
   .jump(gap=GS[0], side="left", reverse=l1)
   .straight(l1, WS[0])                               # section 1 bot
   .straight(l2, WS[1], dy=dy_12)                     # section 2 top
   .jump(gap=GS[1], side="left", reverse=l2)
   .straight(l2, WS[1])                               # section 2 bot
   .straight(l3, WS[2])                               # section 3 top (no dy)
   .jump(gap=GS[2], side="left", reverse=l2)
   .straight(l2, WS[2])                               # section 3 bot
   .straight(l3, WS[3])                               # section 4 top (no dy)
   .jump(gap=GS[3], side="left", reverse=l2)
   .straight(l2, WS[3])                               # section 4 bot
   .straight(l2, WS[4])                               # section 5 top (no dy)
   .jump(gap=GS[4], side="left", reverse=l2)
   .straight(l2, WS[4])                               # section 5 bot
   .straight(l1, WS[5], dy=dy_12)                     # section 6 top (parity dy)
   .jump(gap=GS[5], side="left", reverse=l1)
   .straight(l1, WS[5])                               # section 6 bot
   .straight(l1 * 0.8, w0, dy=dy_01)                  # output transition
   .straight(l0, w0))                                 # output feedline

strands = [s for s in tr.strands if s]
# Layout bbox for the substrate / air enclosure.
all_rects = [r for s in strands for r in s]
x_min = min(r[0] for r in all_rects)
x_max = max(r[2] for r in all_rects)
y_min = min(r[1] for r in all_rects)
y_max = max(r[3] for r in all_rects)


# %% Geometry + Materials
g = rf.Geometry(maxh=MAXH)

# Substrate maxh must be a fraction of SUB_H, at the wave-port cross-section,
# a 1-element-thick FR4 slab is too coarse for the vector eigensolve to see
# the inhomogeneous quasi-TEM mode (returns 0 propagating modes).
fr4 = rf.Dielectric(er=ER_SUB, tand=TAND, maxh=SUB_H / 3)

pad_x = PAD_X_MIL * mil
pad_y = PAD_Y_MIL * mil
sub_w = (x_max - x_min) + 2 * pad_x
sub_h = (y_max - y_min) + 2 * pad_y
sub_origin_x = x_min - pad_x
sub_origin_y = y_min - pad_y

sub = g.box(sub_w, sub_h, SUB_H,
            position=(sub_origin_x, sub_origin_y, 0), material=fr4)

air_h = AIR_H_FACTOR * SUB_H
air = g.box(sub_w, sub_h, air_h,
            position=(sub_origin_x, sub_origin_y, SUB_H),
            material=rf.Air())

# Per-entity mesh size for the trace plates. Match the smallest coupled gap
# so the FEM at least sees the coupling region as one element high; finer
# than this on 70 mm-long traces explodes the tet count.
TRACE_MAXH = min(GS)

# One conductor per strand: emit each horizontal strip as an xy_plate, then
# fuse the strips of a strand into a single watertight face. Touching strips
# only fragmented across a 1-D seam would be topologically shared but can
# still leak across that seam in the mesh; the native union removes it.
trace_plates = []
for strand in strands:
    plates = [
        g.xy_plate(x_hi - x_lo, y_hi - y_lo,
                   position=(x_lo, y_lo, SUB_H), maxh=TRACE_MAXH)
        for (x_lo, y_lo, x_hi, y_hi) in strand
    ]
    if len(plates) > 1:
        g.fuse(plates[0], *plates[1:])
    trace_plates.append(plates[0])

g.fragment(sub, air, *trace_plates)

# The trace + ground plane are the cross-section's internal conductor +
# outer PEC; we capture both on one PEC object so the wave-port
# eigensolve can mark the right nodes via `pec=[pec_microstrip]`.
pec_microstrip = rf.PEC(*trace_plates, sub.faces.min(axis="z"))

# Wave ports at x = 0 (trace entry) and x = sub_w (trace exit). The
# port face is the union of the substrate's and air's x-extreme faces,
# i.e. the full microstrip cross-section at that end. f0 picked at the
# pass-band centre so the eigenmode's β = n_eff · k0 stays accurate.
F0 = 0.5 * (FREQUENCIES[0] + FREQUENCIES[-1])
rf.WavePort(sub.faces.min(axis="x"), air.faces.min(axis="x"),
            f0=F0, mode_kind="auto", pec=[pec_microstrip])
rf.WavePort(sub.faces.max(axis="x"), air.faces.max(axis="x"),
            f0=F0, mode_kind="auto", pec=[pec_microstrip])
# Open the enclosure with an ABC on every free outer wall: the four
# lateral y-side faces of substrate and air, plus the air top (the x-side
# faces are the wave ports). A real filter radiates from its open edges;
# modelling the sides as PEC instead seals
# the structure into a *lossless* cavity, and the half-wave coupled
# resonators then sit on near-undamped box resonances right in the
# pass-band. The driven FEM system goes near-singular there and the direct
# solve returns a non-physical, energy-creating field: |S| reads several dB
# over 0 across the pass-band (worst where the resonators store the most
# energy). The first-order ABC (a plain matched-impedance sheet, dissipative
# by construction) supplies the missing radiation-loss path, damps those
# modes, and restores |S11|² + |S21|² ≤ 1.
rf.ABC(sub.faces.min(axis="y"), sub.faces.max(axis="y"),
       air.faces.min(axis="y"), air.faces.max(axis="y"),
       air.faces.max(axis="z"))

rf.show(g)


# %% Mesh
# The per-plate maxh on the traces carries the small-feature constraint.
g.mesh()
rf.show(g)


# %% Problem + Sweep
prob = rf.ProblemFD(g)
result = prob.sweep(FREQUENCIES)
rf.show(prob)
rf.show(result)

s11_db = np.array([20 * math.log10(max(abs(result.sparams[i, 0, 0]), 1e-12))
                   for i in range(len(FREQUENCIES))])
s21_db = np.array([20 * math.log10(max(abs(result.sparams[i, 1, 0]), 1e-12))
                   for i in range(len(FREQUENCIES))])
print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
print(f"|S11| span: {s11_db.min():.1f} to {s11_db.max():.1f} dB")
print(f"|S21| span: {s21_db.min():.2f} to {s21_db.max():.2f} dB")
peak_idx = int(np.argmax(s21_db))
print(f"|S21| peak: {s21_db[peak_idx]:.2f} dB @ {FREQUENCIES[peak_idx] * 1e-9:.2f} GHz")
