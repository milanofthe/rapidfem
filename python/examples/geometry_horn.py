"""
Pyramidal horn antenna: polygon profiles, loft, PML volume, farfield.

Demonstrates a WR-90-fed pyramidal horn built from two polygon cross-sections
joined by g.loft(). A one-sided PML volume absorbs forward radiation; ABC
absorbs from the remaining open boundaries. prob.farfield() computes the
radiation pattern at the centre frequency.

Campaign references: peak directivity 11.2 to 11.4 dBi at 10 GHz.
"""
import sys

import numpy as np
import rapidfem as rf


def main() -> int:
    mm = 1e-3
    wga, wgb = 22.86 * mm, 10.16 * mm
    Lfeed, Lhorn = 15.0 * mm, 50.0 * mm
    WH, HH = 30.0 * mm, 22.0 * mm
    LPAD_BEAM, LPAD_SIDE = 88.0 * mm, 30.0 * mm
    PML_T = 15.0 * mm
    FREQUENCIES = np.linspace(8.0e9, 12.0e9, 5)
    F0 = 10.0e9
    MAXH = rf.lambda_maxh(f_max=11.0e9, per_lambda=8)
    MAXH_AIR = rf.lambda_maxh(f_max=11.0e9, per_lambda=3)
    AIR_X0, AIR_X1 = -Lfeed, Lhorn + LPAD_BEAM
    AIR_Y0, AIR_Y1 = -WH / 2 - LPAD_SIDE, WH / 2 + LPAD_SIDE
    AIR_Z0, AIR_Z1 = -HH / 2 - LPAD_SIDE, HH / 2 + LPAD_SIDE

    g = rf.Geometry(maxh=MAXH_AIR)
    air = g.box(AIR_X1 - AIR_X0, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
                position=(AIR_X0, AIR_Y0, AIR_Z0), material=rf.Air())
    pml_xp = g.box(PML_T, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
                   position=(AIR_X1, AIR_Y0, AIR_Z0), maxh=2 * MAXH)
    feed = g.box(Lfeed, wga, wgb, position=(-Lfeed, -wga / 2, -wgb / 2),
                 material=rf.Air(), maxh=wgb / 3)
    throat = g.polygon(
        [(0, -wga / 2, -wgb / 2), (0, wga / 2, -wgb / 2),
         (0, wga / 2, wgb / 2), (0, -wga / 2, wgb / 2)])
    aperture = g.polygon(
        [(Lhorn, -WH / 2, -HH / 2), (Lhorn, WH / 2, -HH / 2),
         (Lhorn, WH / 2, HH / 2), (Lhorn, -WH / 2, HH / 2)])
    horn = g.loft(throat, aperture, material=rf.Air(), maxh=MAXH)

    rf.RectWaveguidePort(feed.faces.min(axis="x"), mode=(1, 0), power=1.0)
    rf.PEC(feed.faces.min(axis="y"), feed.faces.max(axis="y"),
           feed.faces.min(axis="z"), feed.faces.max(axis="z"))
    rf.PEC(*horn.faces.where(lambda c, b: 1e-6 < c[0] < Lhorn - 1e-6))
    rf.PML(pml_xp, direction=(1, 0, 0), inner_face=Lhorn + LPAD_BEAM, thickness=PML_T)
    rf.PEC(*pml_xp.faces.outer)
    rf.ABC(*air.faces.outer.unassigned)
    g.mesh()
    print(f"meshed: {g.n_tets} tets")

    prob = rf.Problem(g)
    res = prob.sweep(FREQUENCIES)
    s_rm = np.asarray(res.sparams)
    s11 = np.abs(s_rm[:, 0, 0])
    print(f"|S11| range = [{s11.min():.3f}, {s11.max():.3f}]")

    fi0 = int(np.argmin(np.abs(FREQUENCIES - F0)))
    pat = prob.farfield(res, freq_idx=fi0, port_idx=0, n_theta=91, n_phi=72)
    print(f"peak directivity = {pat.peak_directivity_dbi:.2f} dBi  "
          f"(campaign references 11.2 to 11.4 dBi)")
    print(f"peak gain = {pat.peak_gain_dbi:.2f} dBi")

    if s11.max() >= 0.35:
        print(f"FAIL: max |S11| {s11.max():.3f} exceeds 0.35")
        return 1
    # campaign references peak directivity 11.2 to 11.4 dBi; gate is 10.5 to 12.0 dBi
    if not (10.5 <= pat.peak_directivity_dbi <= 12.0):
        print(f"FAIL: peak directivity {pat.peak_directivity_dbi:.2f} dBi "
              f"outside [10.5, 12.0]")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
