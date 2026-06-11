"""
Coaxial impedance step: voids via cut(), CoaxPort, unassigned selector.

Demonstrates how a coaxial line geometry is built by subtracting inner
conductor cylinders from an outer conductor via g.cut(), assigning
CoaxPort boundary conditions, and collecting remaining PEC faces via
.unassigned.

The 50-ohm to 75-ohm step yields |Gamma| = 25/125 = 0.199 (frequency-flat).
"""
import sys

import numpy as np
import rapidfem as rf


def main() -> int:
    RI_1, RO_1 = 1.50e-3, 3.45e-3
    RI_2, RO_2 = 0.99e-3, 3.45e-3
    L1, L2 = 15.0e-3, 15.0e-3
    L = L1 + L2
    FREQS = np.linspace(1.0e9, 10.0e9, 11)

    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=10.0e9))
    air = g.cylinder(RO_1, L, material=rf.Air())
    inner_a = g.cylinder(RI_1, L1)
    inner_b = g.cylinder(RI_2, L2, position=(0, 0, L1))
    g.cut(air, inner_a, inner_b)
    rf.CoaxPort(air.faces.min(axis="z"), ri=RI_1, ro=RO_1, origin=(0, 0, 0))
    rf.CoaxPort(air.faces.max(axis="z"), ri=RI_2, ro=RO_2, origin=(0, 0, L))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    print(f"meshed: {g.n_tets} tets")

    res = rf.Problem(g).sweep(FREQS)
    s = np.asarray(res.sparams)
    p = np.abs(s[:, 0, 0]) ** 2 + np.abs(s[:, 1, 0]) ** 2
    s11 = np.abs(s[:, 0, 0])
    print(f"|S11| range = [{s11.min():.4f}, {s11.max():.4f}]")
    print(f"passivity range = [{p.min():.4f}, {p.max():.4f}]")

    if not (p.min() > 0.95 and p.max() < 1.005):
        print(f"FAIL: passivity {p.min():.4f} .. {p.max():.4f} outside (0.95, 1.005)")
        return 1
    # 50 ohm -> 75 ohm step: |Gamma| = 25/125 = 0.199 (frequency-flat)
    if not (0.17 < s11.min() and s11.max() < 0.23):
        print(f"FAIL: |S11| {s11.min():.4f} .. {s11.max():.4f} outside (0.17, 0.23)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
