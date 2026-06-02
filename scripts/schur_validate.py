"""Validate Schur-DD FD solve against the monolithic solve across a geometry
ladder: for each case, run ProblemFD.sweep with n_subdomains=1 (monolithic) and
n_subdomains=4 (Schur DD) and compare S-parameters.

The honest criterion is that the S-parameters agree closely: the field can
differ by the ill-conditioning sensitivity (~1e-3), but the projected
S-parameters are robust and must match to a few 1e-3. Run locally, never CI.
"""
import math
import time
import numpy as np
import rapidfem as rf

C = 299_792_458.0
MM = 1e-3


def build_wr90():
    a, b, L = 22.86 * MM, 10.16 * MM, 30.0 * MM
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12.0e9))
    air = g.box(a, b, L, position=(-a / 2, -b / 2, 0), material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    return g, np.linspace(8.0e9, 12.0e9, 5)


def build_coax():
    ri, ro, L = 1.50 * MM, 3.45 * MM, 20.0 * MM
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=10.0e9))
    air = g.cylinder(radius=ro, height=L, position=(0, 0, 0), material=rf.Air())
    inner = g.cylinder(radius=ri, height=L, position=(0, 0, 0), material=rf.Air())
    g.fragment(air, inner)
    rf.CoaxPort(air.faces.min(axis="z"), ri=ri, ro=ro, origin=(0, 0, 0))
    rf.CoaxPort(air.faces.max(axis="z"), ri=ri, ro=ro, origin=(0, 0, L))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    return g, np.linspace(2.0e9, 10.0e9, 5)


def build_iris():
    """WR-90 with 3 inductive PEC irises (Tier-2: internal PEC, reflective)."""
    A, B = 22.86 * MM, 10.16 * MM
    apertures = [10.0 * MM, 8.0 * MM, 10.0 * MM]
    spacing, iris_t, in_len, out_len = 15.0 * MM, 1.0 * MM, 12.0 * MM, 12.0 * MM
    L = in_len + (len(apertures) - 1) * spacing + 2 * iris_t + out_len
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12.4e9))
    air = g.box(A, B, L, position=(-A / 2, -B / 2, 0), material=rf.Air())
    z_centers = [in_len + iris_t / 2 + k * spacing for k in range(len(apertures))]
    iris_vols = []
    for zc, w in zip(z_centers, apertures):
        strip_w = (A - w) / 2
        for side in (-1, +1):
            x0 = -A / 2 if side < 0 else w / 2
            iris_vols.append(g.box(strip_w, B, iris_t,
                                   position=(x0, -B / 2, zc - iris_t / 2), material=rf.Air()))
    g.fragment(air, *iris_vols)
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    return g, np.linspace(9.0e9, 11.0e9, 3)


def build_microstrip():
    """50 ohm microstrip, hybrid wave ports + open ABC walls (Tier-2 inhomogeneous + Tier-4 open)."""
    sub_h, er_sub, tand = 0.508 * MM, 3.55, 0.0027
    line_w, line_l, sub_w, air_h = 1.13 * MM, 30.0 * MM, 20.0 * MM, 10.0 * MM
    freqs = np.linspace(2.85e9, 3.30e9, 3)
    maxh = rf.lambda_maxh(f_max=3.3e9, er_max=er_sub)
    g = rf.Geometry(maxh=maxh)
    fr4 = rf.Dielectric(er=er_sub, tand=tand, maxh=sub_h / 3)
    sub = g.box(sub_w, line_l, sub_h, position=(-sub_w / 2, 0, 0), material=fr4)
    air = g.box(sub_w, line_l, air_h, position=(-sub_w / 2, 0, sub_h), material=rf.Air())
    trace = g.xy_plate(line_w, line_l, position=(-line_w / 2, 0, sub_h))
    g.fragment(sub, air, trace)
    pec_strip = rf.PEC(trace, sub.faces.min(axis="z"))
    f0 = 0.5 * (freqs[0] + freqs[-1])
    rf.WavePort(sub.faces.min(axis="y"), air.faces.min(axis="y"), f0=f0, mode_kind="auto", pec=[pec_strip])
    rf.WavePort(sub.faces.max(axis="y"), air.faces.max(axis="y"), f0=f0, mode_kind="auto", pec=[pec_strip])
    rf.ABC(sub.faces.min(axis="x"), sub.faces.max(axis="x"),
           air.faces.min(axis="x"), air.faces.max(axis="x"), air.faces.max(axis="z"))
    g.auto_refine_features(base_maxh=maxh)
    g.mesh()
    return g, freqs


CASES = {"wr90": build_wr90, "coax": build_coax, "iris": build_iris, "microstrip": build_microstrip}


def build_wr90_scaled(length_mm, maxh_mm):
    """WR-90 of a given length / mesh size — a controllable knob to scale DOFs."""
    a, b, L = 22.86 * MM, 10.16 * MM, length_mm * MM
    g = rf.Geometry(maxh=maxh_mm * MM)
    air = g.box(a, b, L, position=(-a / 2, -b / 2, 0), material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()
    return g, np.linspace(9.0e9, 11.0e9, 2)


def build_patch(maxh_scale=1.0):
    """Edge-fed microstrip patch on FR-4 with a full PML enclosure — a genuinely
    BULKY 3D radiation problem (large air box, 5 PML slabs), where the global
    factor fill-in is severe and DD is meaningful. 1 lumped port -> S11."""
    SUB_W, SUB_L, SUB_H, ER = 60 * MM, 60 * MM, 1.6 * MM, 4.4
    PATCH_W, PATCH_L = 38 * MM, 29 * MM
    FEED_W = 1.5 * MM
    PAD_XY, PAD_Z, PML_T = 25 * MM, 60 * MM, 20 * MM
    maxh = rf.lambda_maxh(f_max=2.8e9) * maxh_scale
    tw, tl, air_top = SUB_W + 2 * PAD_XY, SUB_L + 2 * PAD_XY, SUB_H + PAD_Z
    XO, YO = tw / 2, tl / 2
    g = rf.Geometry(maxh=maxh)
    fr4 = rf.Dielectric(er=ER, maxh=1.5 * SUB_H)
    pml_air = rf.Air(maxh=2 * maxh)
    air = g.box(tw, tl, air_top, position=(-XO, -YO, 0), material=rf.Air())
    pxp = g.box(PML_T, tl + 2 * PML_T, air_top, position=(XO, -YO - PML_T, 0), material=pml_air)
    pxm = g.box(PML_T, tl + 2 * PML_T, air_top, position=(-XO - PML_T, -YO - PML_T, 0), material=pml_air)
    pyp = g.box(tw, PML_T, air_top, position=(-XO, YO, 0), material=pml_air)
    pym = g.box(tw, PML_T, air_top, position=(-XO, -YO - PML_T, 0), material=pml_air)
    pzp = g.box(tw + 2 * PML_T, tl + 2 * PML_T, PML_T, position=(-XO - PML_T, -YO - PML_T, air_top), material=pml_air)
    sub = g.box(SUB_W, SUB_L, SUB_H, position=(-SUB_W / 2, -SUB_L / 2, 0), material=fr4)
    patch = g.xy_plate(PATCH_W, PATCH_L, position=(-PATCH_W / 2, -PATCH_L / 2, SUB_H))
    feed = g.plate(p0=(-FEED_W / 2, -PATCH_L / 2, 0), width=(FEED_W, 0, 0), height=(0, 0, SUB_H))
    g.fragment(air, pxp, pxm, pyp, pym, pzp, sub, patch, feed)
    rf.LumpedPort(feed, direction=(0, 0, 1), z0=50.0)
    rf.PEC(patch, sub.faces.min(axis="z"))
    rf.PML(pxp, direction=(1, 0, 0), inner_face=XO, thickness=PML_T)
    rf.PML(pxm, direction=(-1, 0, 0), inner_face=-XO, thickness=PML_T)
    rf.PML(pyp, direction=(0, 1, 0), inner_face=YO, thickness=PML_T)
    rf.PML(pym, direction=(0, -1, 0), inner_face=-YO, thickness=PML_T)
    rf.PML(pzp, direction=(0, 0, 1), inner_face=air_top, thickness=PML_T)
    rf.PEC(*pxp.faces.outer, *pxm.faces.outer, *pyp.faces.outer, *pym.faces.outer, *pzp.faces.outer)
    g.auto_refine_features(base_maxh=maxh)
    g.mesh(optimize=False)
    return g, np.linspace(2.2e9, 2.6e9, 2)


CASES.update({"patch": lambda: build_patch(1.0), "patch_fine": lambda: build_patch(0.7)})


# Larger parametric cases — run ONE at a time, foreground.
CASES.update({
    "wr90_a": lambda: build_wr90_scaled(120.0, 4.0),
    "wr90_b": lambda: build_wr90_scaled(220.0, 3.4),
    "wr90_c": lambda: build_wr90_scaled(320.0, 3.0),
})


def run(name, builder, k=8):
    import time
    g, freqs = builder()
    t0 = time.perf_counter()
    pm = rf.ProblemFD(g)
    s_mono = np.asarray(pm.sweep(freqs).sparams)
    t_mono = time.perf_counter() - t0
    n_dof = pm.n_dofs
    t1 = time.perf_counter()
    s_schur = np.asarray(rf.ProblemFD(g, n_subdomains=k).sweep(freqs).sparams)
    t_schur = time.perf_counter() - t1
    dmax = float(np.max(np.abs(s_mono - s_schur)))
    print(f"[{name}] n_dof={n_dof} k={k} max|dS|={dmax:.3e} "
          f"t_mono={t_mono:.1f}s t_schur={t_schur:.1f}s")
    return dmax, n_dof


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("k=")]
    kopt = next((int(a[2:]) for a in sys.argv[1:] if a.startswith("k=")), 8)
    sel = args or ["wr90", "coax"]
    worst = 0.0
    for name in sel:
        d, _ = run(name, CASES[name], k=kopt)
        worst = max(worst, d)
    print(f"=== worst max|dS| across selected cases = {worst:.3e} ===")
