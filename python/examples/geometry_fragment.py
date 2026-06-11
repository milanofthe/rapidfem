"""
Selector stress test: box + coincident xy_plate (the patch-antenna scenario).
Conformity is automatic on the rapidmesh kernel (fragment is a no-op); this
verifies that face selectors and a plate object resolve to distinct solver
tags on the meshed scene.
"""
import sys

import rapidfem as rf


def main() -> int:
    g = rf.Geometry(maxh=2.0)
    sub = g.box(10, 10, 1, position=(0, 0, 0), material=rf.Dielectric(er=4.4))
    plate = g.xy_plate(6, 6, position=(2, 2, 1))
    g.fragment(sub, plate)  # no-op, kept for familiarity

    ground = rf.PEC(sub.faces.min(axis="z"))
    wall_lo = rf.ABC(sub.faces.where(lambda c, _: abs(c[0] - 0) < 1e-9))
    wall_hi = rf.ABC(sub.faces.where(lambda c, _: abs(c[0] - 10) < 1e-9))
    patch = rf.PEC(plate)
    g.mesh()

    tags = {
        "ground": g._physics_tags[id(ground)],
        "wall_x_low": g._physics_tags[id(wall_lo)],
        "wall_x_high": g._physics_tags[id(wall_hi)],
        "patch": g._physics_tags[id(patch)],
    }
    print(f"meshed: {g.n_tets} tets")
    for n, t in sorted(tags.items()):
        print(f"  {n!r}: tag {t}")

    nodes, tets, tet_tags, tris, tri_tags = g._mesh_arrays
    if len(set(tags.values())) != len(tags):
        print(f"FAIL: tag collisions in {tags}")
        return 1
    for n, t in tags.items():
        if not (tri_tags == t).any():
            print(f"FAIL: selection {n!r} tagged no faces")
            return 1
    print("OK: all 4 selections resolved to distinct tagged face sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
