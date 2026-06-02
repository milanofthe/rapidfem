"""Validate Schur-DD FD solve against the monolithic solve across a geometry
ladder: for each case, run ProblemFD.sweep with n_subdomains=1 (monolithic) and
n_subdomains=4 (Schur DD) and compare S-parameters.

The honest criterion is that the S-parameters agree closely: the field can
differ by the ill-conditioning sensitivity (~1e-3), but the projected
S-parameters are robust and must match to a few 1e-3. Run locally, never CI.
"""
import math
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


def run(name, builder, k=4):
    g, freqs = builder()
    s_mono = np.asarray(rf.ProblemFD(g).sweep(freqs).sparams)
    s_schur = np.asarray(rf.ProblemFD(g, n_subdomains=k).sweep(freqs).sparams)
    dmax = float(np.max(np.abs(s_mono - s_schur)))
    print(f"[{name}] max|S_mono - S_schur(k={k})| = {dmax:.3e}")
    # also print |S11|,|S21| both ways at mid band for a sanity look
    mid = len(freqs) // 2
    print(f"   mid f={freqs[mid]/1e9:.2f}GHz  mono |S11|={abs(s_mono[mid,0,0]):.4f} "
          f"|S21|={abs(s_mono[mid,1,0]):.4f}  schur |S11|={abs(s_schur[mid,0,0]):.4f} "
          f"|S21|={abs(s_schur[mid,1,0]):.4f}")
    return dmax


if __name__ == "__main__":
    import sys
    sel = sys.argv[1:] or list(CASES)
    worst = 0.0
    for name in sel:
        d = run(name, CASES[name])
        worst = max(worst, d)
    print(f"=== worst max|dS| across cases = {worst:.3e} ===")
