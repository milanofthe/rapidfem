"""Slotted Vivaldi (exponentially-tapered slot) antenna, open-region with PML.

A microstrip line on the top copper of a thin FR-4 board crosses an
exponentially-tapered slot etched into the bottom ground plane. The slot
flares from a narrow ``g`` gap at the feed to a wide ``W`` aperture, giving
the travelling-wave radiation of a Vivaldi (tapered-slot) antenna. The slot
line is terminated by a circular open-circuit cavity cut into the ground; the
microstrip is terminated by a radial stub on top.

This is the rigorous reference setup (geometry ported from EMerge demo19):

* **Parametric taper.** rapidfem has no ``parametric`` curve helper, so the
  taper ``fy(t) = (g/2)·Kᵗ + (W − g·K)/(2 − 2K)·(1 − Kᵗ)``, ``t ∈ [0,1]`` is
  discretised into a point list for :meth:`Geometry.polygon`.
* **Open-region truncation by PML, not ABC.** A first-order ABC is only
  accurate near normal incidence and must sit ≳ λ/4 from the radiator (≈ 25 mm
  at the 3 GHz band edge), which balloons the air box. A PML absorbs oblique
  incidence and can sit close, so a modest air pad + a PML shell keeps the
  mesh small *and* the far field trustworthy. The Vivaldi radiates endfire and
  through the slot to both sides, so all six faces get a PML.
* **Mesh budget.** The bulk air is meshed at λ/5 of the top frequency; the PML
  is 2× coarser (its accuracy comes from the stretch profile, not cell count);
  only the feed / stub / port / taper get a fine local size. That keeps the
  tet count in check despite the padded, six-sided enclosure.

Wideband (3-8 GHz); the sweep is the slow part. Drop ``N_FREQ`` for a quick
look, or raise ``PER_LAMBDA`` / shrink ``PAD`` only if you understand the
accuracy trade.
"""

# %% Parameters
import math

import numpy as np
import rapidfem as rf

mm = 1e-3

G_GAP = 0.3 * mm      # narrow taper gap at the feed
L_SLOT = 70.0 * mm    # taper / slot length (along x)
W_AP = 55.0 * mm      # aperture width (full, along y)
K = 200.0             # exponential growth factor
RADIUS = 20.0 * mm    # open-circuit disc diameter source (radius = RADIUS/2)
TH = 0.5 * mm         # PCB thickness
ER_SUB = 4.4          # FR-4
L_STUB = 7.0 * mm     # radial stub length
STUB_ANG = 80.0       # radial stub opening angle [deg]
STUB_ANG_OFF = 20.0   # stub rotation off the feed axis [deg]

# PCB plan bounds (the taper aperture sticks out past the dielectric edge).
PCB_XMIN, PCB_XMAX = -25.0 * mm, 70.0 * mm
PCB_YMIN, PCB_YMAX = -30.0 * mm, 30.0 * mm

# Open-region truncation. PAD is the near-field buffer between the antenna and
# the PML; PML_T is the absorber thickness. With a PML, PAD can stay small
# (the absorber, not the distance, kills the outgoing wave) which is what keeps
# the mesh affordable.
PAD = 12.0 * mm
PML_T = 18.0 * mm


# 50 Ω microstrip width on FR-4, th = 0.5 mm (Hammerstad synthesis).
def _ms_width_50(z0: float, er: float, h: float) -> float:
    b = 377.0 * math.pi / (2.0 * z0 * math.sqrt(er))
    woh = (2.0 / math.pi) * (
        b - 1.0 - math.log(2.0 * b - 1.0)
        + (er - 1.0) / (2.0 * er) * (math.log(b - 1.0) + 0.39 - 0.61 / er)
    )
    return woh * h


W0 = _ms_width_50(50.0, ER_SUB, TH)   # ≈ 0.95 mm

N_FREQ = 6
FREQUENCIES = np.linspace(3.0e9, 8.0e9, N_FREQ)
F_FF = 6.0e9                          # far-field frequency
PER_LAMBDA = 5                        # air cells per wavelength at f_max
MAXH = rf.lambda_maxh(f_max=8.0e9, per_lambda=PER_LAMBDA)
TAPER_N = 48                          # parametric-curve sampling


# %% Parametric taper curve
# fy(t): half-width of the slot; fy(0)=g/2 (gap), fy(1)=W/2 (aperture).
def fy(t):
    return (G_GAP / 2.0) * K**t + (W_AP - G_GAP * K) / (2.0 - 2.0 * K) * (1.0 - K**t)


def taper_points(x0=0.0):
    """CCW outline of the tapered slot: top edge t:0→1, bottom edge t:1→0."""
    ts = np.linspace(0.0, 1.0, TAPER_N)
    top = [(x0 + t * L_SLOT, fy(t)) for t in ts]
    bot = [(x0 + t * L_SLOT, -fy(t)) for t in reversed(ts)]
    return top + bot


# %% Geometry + Materials
g = rf.Geometry(maxh=MAXH)

# One material per region for size control: substrate (fine), bulk near-field
# air (λ/5), and a 2× coarser air inside the PML (the stretch profile sets the
# absorber's accuracy, so a fine PML mesh is wasted).
fr4 = rf.Dielectric(er=ER_SUB, tand=0.02, maxh=4.0 * TH)
bulk_air = rf.Air()
pml_air = rf.Air(maxh=2.0 * MAXH)

# Bulk air box: PCB bounds padded by PAD on every side.
X0, X1 = PCB_XMIN - PAD, PCB_XMAX + PAD
Y0, Y1 = PCB_YMIN - PAD, PCB_YMAX + PAD
Z0, Z1 = -TH - PAD, PAD
AW, AL, AH = X1 - X0, Y1 - Y0, Z1 - Z0
T = PML_T

air = g.box(AW, AL, AH, position=(X0, Y0, Z0), material=bulk_air)

# Six non-overlapping PML slabs tiling the shell. The ±z slabs span the full
# padded footprint (covering the top/bottom corners); the ±x slabs cover the
# y-edges (extended in y); the ±y slabs fill the remaining mid-band. Every
# outgoing ray exits through exactly one PML.
pml_zp = g.box(AW + 2 * T, AL + 2 * T, T, position=(X0 - T, Y0 - T, Z1), material=pml_air)
pml_zm = g.box(AW + 2 * T, AL + 2 * T, T, position=(X0 - T, Y0 - T, Z0 - T), material=pml_air)
pml_xp = g.box(T, AL + 2 * T, AH, position=(X1, Y0 - T, Z0), material=pml_air)
pml_xm = g.box(T, AL + 2 * T, AH, position=(X0 - T, Y0 - T, Z0), material=pml_air)
pml_yp = g.box(AW, T, AH, position=(X0, Y1, Z0), material=pml_air)
pml_ym = g.box(AW, T, AH, position=(X0, Y0 - T, Z0), material=pml_air)

# Substrate slab z ∈ [-TH, 0].
sub = g.box(PCB_XMAX - PCB_XMIN, PCB_YMAX - PCB_YMIN, TH,
            position=(PCB_XMIN, PCB_YMIN, -TH), material=fr4)

# Ground plane (bottom copper, z = -TH) with the taper slot + open-circuit disc
# cut out. (The EMerge demo also corrugates the ground edges to widen the band;
# dropped here, the thin boolean slivers wreck tet quality.)
ground = g.xy_plate(PCB_XMAX - PCB_XMIN, PCB_YMAX - PCB_YMIN,
                    position=(PCB_XMIN, PCB_YMIN, -TH))
taper = g.polygon([(x, y, -TH) for (x, y) in taper_points()], maxh=4.0 * mm)
disc = g.disc(RADIUS / 2.0, position=(-RADIUS / 2.0 + 1.0 * mm, 0.0, -TH))
g.cut(ground, taper, disc)

# Feed: 50 Ω microstrip on the top copper (z = 0), running +y across the slot.
FEED_X, FEED_Y0 = 2.0 * mm, -10.0 * mm
FEED_LEN = 10.5 * mm
feed = g.xy_plate(W0, FEED_LEN, position=(FEED_X - W0 / 2.0, FEED_Y0, 0.0),
                  maxh=1.0 * mm)

# Radial (sector) stub at the line end for the slot-line transition.
stub_cx, stub_cy = FEED_X, FEED_Y0 + FEED_LEN - 0.2 * mm
base = math.radians(90.0 + STUB_ANG_OFF)
half = math.radians(STUB_ANG / 2.0)
arc = [(stub_cx, stub_cy, 0.0)]
for a in np.linspace(base - half, base + half, 24):
    arc.append((stub_cx + L_STUB * math.cos(a), stub_cy + L_STUB * math.sin(a), 0.0))
stub = g.polygon(arc, maxh=1.2 * mm)

# Lumped feed: a vertical sheet bridging the ground plane (z = -TH) to the
# microstrip trace (z = 0) through the substrate.
port = g.plate(p0=(FEED_X - W0 / 2.0, FEED_Y0, -TH),
               width=(W0, 0.0, 0.0), height=(0.0, 0.0, TH), maxh=0.8 * mm)

g.fragment(air, pml_zp, pml_zm, pml_xp, pml_xm, pml_yp, pml_ym,
           sub, ground, feed, stub, port)

# %% Physics
rf.LumpedPort(port, direction=(0, 0, 1), z0=50.0)
rf.PEC(ground, feed, stub)

rf.PML(pml_zp, direction=(0, 0, 1), inner_face=Z1, thickness=T)
rf.PML(pml_zm, direction=(0, 0, -1), inner_face=Z0, thickness=T)
rf.PML(pml_xp, direction=(1, 0, 0), inner_face=X1, thickness=T)
rf.PML(pml_xm, direction=(-1, 0, 0), inner_face=X0, thickness=T)
rf.PML(pml_yp, direction=(0, 1, 0), inner_face=Y1, thickness=T)
rf.PML(pml_ym, direction=(0, -1, 0), inner_face=Y0, thickness=T)

# Terminate every PML with PEC on its outer hull.
rf.PEC(*pml_zp.faces.outer, *pml_zm.faces.outer,
       *pml_xp.faces.outer, *pml_xm.faces.outer,
       *pml_yp.faces.outer, *pml_ym.faces.outer)

# Near-field-to-far-field surface. A PML domain has no ABC Huygens surface
# for the solver to auto-detect, so mark the bulk-air / PML interface (a closed
# box enclosing the antenna, just inside the absorber) for the NFFT integral.
# (.hull, not .outer: once air is wrapped in PML on all six sides none of its
# faces touch the model bbox, so .outer is empty; .hull keys off air's own box.)
rf.FarFieldSurface(*air.faces.hull)

rf.show(g)


# %% Mesh
# NB: no auto_refine_features here. The feed / stub / port / taper already carry
# explicit local sizes and the substrate gets its size from the FR-4 material;
# auto-refining the 0.5 mm-thin substrate on top of that (and grading it into
# the padded air box) blows the tet count up ~4× for no accuracy gain.
g.mesh()
rf.show(g)


# %% Problem + sweep
prob = rf.ProblemFD(g)
result = prob.sweep(FREQUENCIES)
rf.show(result)

mags = [abs(result.sparams[i, 0, 0]) for i in range(N_FREQ)]
print(f"\nVivaldi: DOFs={prob.n_dofs}, tets={prob.n_tets}")
for f, s in zip(FREQUENCIES, mags):
    print(f"  f={f/1e9:5.2f} GHz   |S11| = {20*np.log10(s):6.2f} dB")

# %% Far-field at 6 GHz
fi = int(min(range(N_FREQ), key=lambda i: abs(FREQUENCIES[i] - F_FF)))
pattern = prob.farfield(result, freq_idx=fi, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"\nFar-field @ {FREQUENCIES[fi]/1e9:.2f} GHz: "
          f"D = {pattern.peak_directivity_dbi:.2f} dBi, "
          f"G = {pattern.peak_gain_dbi:.2f} dBi")
