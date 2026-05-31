"""Isolate where the new mesher hangs on fd_microstrip_line.

Meshes the microstrip geometry four ways:
    1. delaunay, no optimize  (baseline)
    2. hxt,      no optimize
    3. delaunay, optimize
    4. hxt,      optimize     (the new default)

Times each call and prints tet counts. If one combination hangs we know
which kernel to blame.
"""
from __future__ import annotations

import math
import signal
import sys
import time

import rapidfem as rf


def build_geometry():
    mm = 1e-3
    SUB_H = 0.508 * mm
    ER_SUB = 3.55
    LINE_W = 1.13 * mm
    LINE_L = 30.0 * mm
    SUB_W = 20.0 * mm
    AIR_H = 10.0 * mm
    MAXH = rf.lambda_maxh(f_max=3.3e9, er_max=ER_SUB)

    g = rf.Geometry(maxh=MAXH)
    fr4 = rf.Dielectric(er=ER_SUB, tand=0.0027, maxh=1.5 * SUB_H)
    sub = g.box(SUB_W, LINE_L, SUB_H, position=(-SUB_W / 2, 0, 0), material=fr4)
    air = g.box(SUB_W, LINE_L, AIR_H, position=(-SUB_W / 2, 0, SUB_H),
                material=rf.Air())
    trace = g.xy_plate(LINE_W, LINE_L, position=(-LINE_W / 2, 0, SUB_H))
    port_in = g.plate(
        p0=(-LINE_W / 2, 0, 0), width=(LINE_W, 0, 0), height=(0, 0, SUB_H),
    )
    port_out = g.plate(
        p0=(-LINE_W / 2, LINE_L, 0), width=(LINE_W, 0, 0),
        height=(0, 0, SUB_H),
    )
    g.fragment(sub, air, trace, port_in, port_out)
    rf.LumpedPort(port_in, direction=(0, 0, 1), z0=50.0)
    rf.LumpedPort(port_out, direction=(0, 0, 1), z0=50.0)
    rf.PEC(trace, sub.faces.min(axis="z"))
    rf.ABC(*air.faces.outer, order=2)
    g.auto_refine_features(base_maxh=MAXH)
    return g


def probe(algorithm, optimize, label):
    import gmsh
    print(f"\n=== {label} ===", flush=True)
    print(f"  algorithm={algorithm!r}, optimize={optimize}", flush=True)
    g = build_geometry()
    t0 = time.time()
    try:
        g.mesh(algorithm=algorithm, optimize=optimize)
    except Exception as e:
        print(f"  FAILED after {time.time() - t0:.1f}s: {e}", flush=True)
        return
    elapsed = time.time() - t0
    n_tets = 0
    try:
        _, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
        if elem_node_tags:
            n_tets = len(elem_node_tags[0]) // 4
    except Exception:
        pass
    print(f"  done in {elapsed:.1f}s, {n_tets:,} tets", flush=True)


if __name__ == "__main__":
    probe("delaunay", False, "1. delaunay, no optimize (baseline)")
    probe("hxt",      False, "2. hxt, no optimize")
    probe("delaunay", True,  "3. delaunay, optimize")
    probe("hxt",      True,  "4. hxt, optimize (new default)")
    sys.exit(0)
