"""
Inductive iris bandpass filter: voids via cut(), unassigned PEC faces.

Demonstrates a 3-resonator H-plane iris filter built by cutting metal
strip voids out of a WR-90 waveguide box. RectWaveguidePort on each
end; remaining faces collected via .unassigned.

Campaign reference (gmsh sweep): 3-dB passband 10.51 to 11.46 GHz.
"""
import sys

import numpy as np
import rapidfem as rf


def main() -> int:
    A, B = 22.86e-3, 10.16e-3
    APERTURES = [10.0e-3, 8.0e-3, 10.0e-3]
    SPACING = 15.0e-3
    IRIS_T = 1.0e-3
    INPUT_LEN = OUTPUT_LEN = 12.0e-3
    LF = INPUT_LEN + (len(APERTURES) - 1) * SPACING + 2 * IRIS_T + OUTPUT_LEN
    FREQS = np.linspace(8.2e9, 12.4e9, 22)
    z_centers = [INPUT_LEN + IRIS_T / 2 + k * SPACING for k in range(len(APERTURES))]

    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12.4e9))
    box = g.box(A, B, LF, position=(-A / 2, -B / 2, 0), material=rf.Air())
    strips = []
    for zc, w in zip(z_centers, APERTURES):
        strip_w = (A - w) / 2
        for side in (-1, +1):
            x0 = -A / 2 if side < 0 else w / 2
            strips.append(g.box(strip_w, B, IRIS_T, position=(x0, -B / 2, zc - IRIS_T / 2)))
    g.cut(box, *strips)
    rf.RectWaveguidePort(box.faces.min(axis="z"))
    rf.RectWaveguidePort(box.faces.max(axis="z"))
    rf.PEC(*box.faces.unassigned)
    g.mesh()
    print(f"meshed: {g.n_tets} tets")

    res = rf.Problem(g).sweep(FREQS)
    s = np.asarray(res.sparams)
    db = 20 * np.log10(np.maximum(np.abs(s[:, 1, 0]), 1e-12))
    sel = db > -3
    fband = FREQS[sel] / 1e9
    print(f"3-dB band: {fband[0]:.2f} to {fband[-1]:.2f} GHz  "
          f"(campaign reference 10.51 to 11.46)")

    # campaign reference: 3-dB passband 10.51 to 11.46 GHz (tolerance +-0.25 GHz)
    if not (abs(fband[0] - 10.51) < 0.25 and abs(fband[-1] - 11.46) < 0.25):
        print(f"FAIL: band edges {fband[0]:.2f} / {fband[-1]:.2f} GHz outside tolerance")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
