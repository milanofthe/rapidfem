# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Scan the in-repo .msh fixtures for sliver tetrahedra.

For each tet: q = 6V / h_mean^3 (normalized volume; ~0.6 regular, ->0 sliver)
and the radius ratio rr = 3*r_in/r_out = (3 * 3V/A_sum) / r_circ-proxy.
Reports the worst tets per mesh so we know whether the conditioning levers
have anything to act on in our actual test set.
"""
from __future__ import annotations

import glob
import os

import numpy as np

try:
    import gmsh
except Exception as e:  # pragma: no cover
    raise SystemExit(f"gmsh python module required: {e}")


def tet_metrics(P):
    """P: (n,4,3) tet vertex arrays -> (q, radius_ratio) per tet."""
    a, b, c, d = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    # signed 6V
    v6 = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)))
    V = v6 / 6.0
    # edges
    import itertools
    elen = []
    for i, j in itertools.combinations(range(4), 2):
        elen.append(np.linalg.norm(P[:, i] - P[:, j], axis=1))
    elen = np.array(elen)            # (6, n)
    h_mean = elen.mean(axis=0)
    h_max = elen.max(axis=0)
    q = v6 / np.maximum(h_mean ** 3, 1e-300)
    # face areas (4 faces) -> inscribed radius r_in = 3V / A_sum
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    Asum = np.zeros(len(P))
    for (i, j, k) in faces:
        Asum += 0.5 * np.linalg.norm(
            np.cross(P[:, j] - P[:, i], P[:, k] - P[:, i]), axis=1)
    r_in = 3.0 * V / np.maximum(Asum, 1e-300)
    # radius ratio proxy: 3*r_in / r_out, r_out ~ h_max/sqrt(3..) -> use h_max
    rr = 3.0 * r_in / np.maximum(h_max, 1e-300)
    return q, rr, V, h_mean


def scan(path):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(path)
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        coord = {int(t): ncoords[3*i:3*i+3] for i, t in enumerate(ntags)}
        etypes, etags, enodes = gmsh.model.mesh.getElements()
        tets = None
        for et, en in zip(etypes, enodes):
            if et == 4:  # 4-node tetra
                tets = np.array(en, dtype=np.int64).reshape(-1, 4)
        if tets is None or len(tets) == 0:
            return None
        P = np.array([[coord[int(n)] for n in tet] for tet in tets])
        return tet_metrics(P)
    finally:
        gmsh.finalize()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    meshes = sorted(glob.glob(os.path.join(repo, "tests", "meshes", "*.msh")))
    print(f"{'mesh':<22}{'#tets':>8}{'min q':>10}{'p01 q':>10}{'min rr':>9}{'#q<1e-6':>9}{'#q<1e-9':>9}")
    print("-" * 77)
    for m in meshes:
        res = scan(m)
        name = os.path.basename(m)
        if res is None:
            print(f"{name:<22}{'(no tets)':>8}")
            continue
        q, rr, V, h = res
        n = len(q)
        print(f"{name:<22}{n:>8}{q.min():>10.2e}{np.percentile(q,1):>10.2e}"
              f"{rr.min():>9.3f}{int((q<1e-6).sum()):>9}{int((q<1e-9).sum()):>9}")


if __name__ == "__main__":
    main()
