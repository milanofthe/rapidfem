"""
WR-90 straight waveguide on the declarative geometry API.

No raw mesh handling, no TOML strings, no integer tags: primitives,
face selectors and physics objects only. The rapidmesh kernel meshes the
scene and rapidfem.Problem consumes the arrays directly.
"""
import sys

import numpy as np
import rapidfem as rf


def main() -> int:
    a = 22.86e-3   # WR-90 broad wall
    b = 10.16e-3   # narrow wall
    L = 30e-3      # length

    g = rf.Geometry(maxh=3e-3)
    box = g.box(a, b, L, material=rf.Air())
    rf.RectWaveguidePort(box.faces.min(axis="z"))
    rf.RectWaveguidePort(box.faces.max(axis="z"))
    rf.PEC(*box.faces.unassigned)
    g.mesh()
    print(f"meshed: {g.n_tets} tets")

    result = rf.Problem(g).sweep(np.linspace(9.0e9, 11.0e9, 11))

    s11_max = float(np.abs(result.sparams[:, 0, 0]).max())
    s21_min = float(np.abs(result.sparams[:, 1, 0]).min())
    s21_max = float(np.abs(result.sparams[:, 1, 0]).max())
    print(f"max |S11| = {s11_max:.5f}  (expected << 1)")
    print(f"|S21| range = [{s21_min:.5f}, {s21_max:.5f}]  (expected ~1)")

    if s11_max < 0.01 and abs(s21_min - 1.0) < 0.01 and abs(s21_max - 1.0) < 0.01:
        print("OK")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
