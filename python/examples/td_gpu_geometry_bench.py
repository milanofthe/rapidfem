"""GPU vs CPU time-domain DGTD benchmark on real example geometries.

Where ``td_benchmark.py`` profiles the DGTD hot path on a single
geometry, this benchmark runs the *whole* time-domain solver on three
representative geometries, each at three mesh resolutions, and reports
the GPU LSERK4 transient against the CPU LSERK4 transient:

1. Ring resonator - a high-permittivity
   ceramic torus in an air-filled PEC cavity, lit by a homogeneous
   transient: an impulse initial state evolves freely.
2. Power divider - a waveguide T-junction
   with PML-terminated arms, lit by a driven transient: a Gaussian
   soft-pulse injected into the stem.
3. Cavity - a PEC air-cube cavity, lit by
   a driven broadband run through ``driven_transient`` with a probe.

The GPU path is the explicit LSERK4 transient, state device-resident;
for a fair comparison the CPU run uses the same explicit integrator
(``method="explicit"``), which the GPU path mirrors. Meshing is the
practical limiter, so the resolutions are sized so the finest mesh
lands in the 1-4M state-DOF range, not beyond.

Run after ``maturin develop --release``:

    python python/examples/td_gpu_geometry_bench.py
"""

# %% Parameters
import time

import numpy as np

import rapidfem as rf

mm = 1e-3

# Transient step counts. A modest count is enough to time meaningfully
# once the GPU is warmed; the explicit integrator substeps internally to
# respect the CFL limit, so the real work per step is several LSERK4
# substeps.
BENCH_STEPS = 60         # timed transient length, per device
WARMUP_STEPS = 3         # short GPU run to warm the OpenCL context / JIT

# Ring resonator - dielectric torus in a PEC air cavity.
RING_R_MAJ = 11.0 * mm   # ring radius, tube-centre to torus axis
RING_R_MIN = 2.6 * mm    # ring tube (cross-section) radius
RING_ER = 10.0           # high-permittivity ceramic ring
RING_BOX = 38.0 * mm     # cubic air-cavity edge
RING_DT = 4e-12          # transient step of the ring-resonator example

# Power divider - waveguide T-junction, PML-terminated on all arms.
PD_W = 16.0 * mm         # square guide cross-section
PD_L_ARM = 44.0 * mm     # each crossbar output arm
PD_L_STEM = 44.0 * mm    # the input stem
PD_PML_T = 20.0 * mm     # matched-absorber slab thickness
PD_DT = 3e-12            # transient step of the power-divider example

# Cavity - PEC air-cube, broadband driven run.
CAV_L = 40.0 * mm        # cubic cavity edge
CAV_DT = 8e-12           # transient step of the transfer-function example


# %% Ring resonator - geometry build and excitation
# A torus embedded in an
# air-filled PEC box, fragmented conformally; only the six axis-aligned
# cavity walls are PEC, the air-ring interface is left un-walled.
def build_ring(maxh_air, maxh_ring):
    """A meshed ProblemTD for the ring resonator at the given resolution."""
    g = rf.Geometry(maxh=maxh_air)
    air = g.box(RING_BOX, RING_BOX, RING_BOX,
                position=(-RING_BOX / 2, -RING_BOX / 2, -RING_BOX / 2),
                material=rf.Air())
    ring = g.torus(RING_R_MAJ, RING_R_MIN,
                   material=rf.Dielectric(er=RING_ER), maxh=maxh_ring)
    g.fragment(air, ring)
    rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
           air.faces.min(axis="y"), air.faces.max(axis="y"),
           air.faces.min(axis="z"), air.faces.max(axis="z"))
    g.mesh()
    return rf.ProblemTD(g, order=2, flux="upwind")


def run_ring(ptd, device, steps):
    """Homogeneous transient: an impulse initial state evolves freely."""
    y0 = np.zeros(ptd.n_dofs)
    y0[ptd.probe_dof((RING_R_MAJ, 0.0, 0.0), field="E", component="z")] = 1.0
    return ptd.transient(y0, dt=RING_DT, steps=steps, method="explicit",
                         device=device, verbose=False)


# %% Power divider - geometry build and excitation
# A T-shaped air guide tiled by
# face-adjacent boxes, three PML slabs terminating the arms; fragment
# stitches it all conformally, PEC walls are the TD operator's default.
def build_power_divider(maxh):
    """A meshed ProblemTD for the power divider at the given resolution."""
    g = rf.Geometry(maxh=maxh)
    air = rf.Air()
    left = g.box(PD_L_ARM, PD_W, PD_W,
                 position=(-PD_W / 2 - PD_L_ARM, -PD_W / 2, -PD_W / 2),
                 material=air)
    hub = g.box(PD_W, PD_W, PD_W,
                position=(-PD_W / 2, -PD_W / 2, -PD_W / 2), material=air)
    right = g.box(PD_L_ARM, PD_W, PD_W,
                  position=(PD_W / 2, -PD_W / 2, -PD_W / 2), material=air)
    stem = g.box(PD_W, PD_L_STEM, PD_W,
                 position=(-PD_W / 2, -PD_W / 2 - PD_L_STEM, -PD_W / 2),
                 material=air)

    x_in = PD_W / 2 + PD_L_ARM           # crossbar arm-end coordinate
    y_in = -PD_W / 2 - PD_L_STEM         # stem-end coordinate
    pml_xm = g.box(PD_PML_T, PD_W, PD_W,
                   position=(-x_in - PD_PML_T, -PD_W / 2, -PD_W / 2),
                   material=air)
    pml_xp = g.box(PD_PML_T, PD_W, PD_W,
                   position=(x_in, -PD_W / 2, -PD_W / 2), material=air)
    pml_ys = g.box(PD_W, PD_PML_T, PD_W,
                   position=(-PD_W / 2, y_in - PD_PML_T, -PD_W / 2),
                   material=air)
    g.fragment(left, hub, right, stem, pml_xm, pml_xp, pml_ys)

    rf.PML(pml_xm, direction=(-1, 0, 0), inner_face=-x_in, thickness=PD_PML_T)
    rf.PML(pml_xp, direction=(1, 0, 0), inner_face=x_in, thickness=PD_PML_T)
    rf.PML(pml_ys, direction=(0, -1, 0), inner_face=y_in, thickness=PD_PML_T)
    g.mesh()
    return rf.ProblemTD(g, order=2, flux="upwind")


def run_power_divider(ptd, device, steps):
    """Driven transient: a Gaussian soft-pulse injected into the stem."""
    y_in = -PD_W / 2 - PD_L_STEM
    pulse = rf.GaussianPulse(t0=100e-12, tau=26e-12, f0=14e9)
    return ptd.transient(
        source=((0.0, y_in + 8 * mm, 0.0), "E", "z"),
        waveform=pulse, dt=PD_DT, steps=steps, method="explicit",
        device=device, verbose=False,
    )


# %% Cavity - geometry build and excitation
# A closed PEC air-cube,
# driven broadband through driven_transient with one field probe.
def build_cavity(maxh):
    """A meshed ProblemTD for the PEC air-cube cavity at the resolution."""
    g = rf.Geometry(maxh=maxh)
    air = g.box(CAV_L, CAV_L, CAV_L, material=rf.Air())
    rf.PEC(*air.faces.unassigned)        # closed cavity, six PEC walls
    g.mesh()
    return rf.ProblemTD(g, order=2, flux="upwind")


def run_cavity(ptd, device, steps):
    """Driven broadband run: driven_transient with a Gaussian pulse and
    one field probe, exactly the transfer-function example's excitation."""
    pulse = rf.GaussianPulse(t0=160e-12, tau=40e-12, f0=8e9)
    source = ((10 * mm, 10 * mm, 10 * mm), "E", "z")
    probe = ((27 * mm, 31 * mm, 18 * mm), "E", "z")
    return ptd.driven_transient(
        source=source, waveform=pulse, probes=[probe],
        dt=CAV_DT, steps=steps, device=device, verbose=False,
    )


# %% Resolution sweeps
# Three maxh values per geometry, coarse to fine, sized so the finest
# mesh lands in the 1-4M state-DOF range. Meshing is the limiter, so
# these are deliberately short of the 10M-DOF regime where complex-
# geometry meshing turns prohibitive. The actual n_dof reached is
# printed in the table below.
RING_RES = [
    ("coarse", dict(maxh_air=6.0 * mm, maxh_ring=3.0 * mm)),
    ("medium", dict(maxh_air=3.0 * mm, maxh_ring=1.6 * mm)),
    ("fine", dict(maxh_air=2.2 * mm, maxh_ring=1.2 * mm)),
]
PD_RES = [
    ("coarse", dict(maxh=PD_W / 5)),
    ("medium", dict(maxh=PD_W / 7)),
    ("fine", dict(maxh=PD_W / 10)),
]
CAV_RES = [
    ("coarse", dict(maxh=CAV_L / 8)),
    ("medium", dict(maxh=CAV_L / 15)),
    ("fine", dict(maxh=CAV_L / 22)),
]

GEOMETRIES = [
    ("ring resonator", build_ring, run_ring, RING_RES),
    ("power divider", build_power_divider, run_power_divider, PD_RES),
    ("cavity", build_cavity, run_cavity, CAV_RES),
]


# %% Benchmark driver
def bench_one(name, build, run, label, kwargs):
    """Build one meshed geometry and time its CPU and GPU transient.

    Returns a result dict, or a dict with an ``error`` key if the build
    or a run failed - one failure must not kill the whole sweep.
    """
    try:
        t = time.perf_counter()
        ptd = build(**kwargs)
        build_s = time.perf_counter() - t
    except Exception as exc:                 # noqa: BLE001 - report, continue
        return dict(name=name, label=label, error=f"mesh/build failed: {exc}")

    n = ptd.n_dofs
    try:
        # CPU explicit transient - the same LSERK4 integrator the GPU
        # path runs, so the timings compare like for like.
        t = time.perf_counter()
        run(ptd, "cpu", BENCH_STEPS)
        cpu_s = time.perf_counter() - t

        # Warm the GPU: a short run pays the OpenCL context / kernel JIT
        # cost so it does not land on the timed run.
        gpu_s = None
        if ptd._op.gpu_available():
            run(ptd, "gpu", WARMUP_STEPS)
            t = time.perf_counter()
            run(ptd, "gpu", BENCH_STEPS)
            gpu_s = time.perf_counter() - t
    except Exception as exc:                 # noqa: BLE001 - report, continue
        return dict(name=name, label=label, n=n, error=f"run failed: {exc}")

    return dict(name=name, label=label, n=n, build_s=build_s,
                cpu_s=cpu_s, gpu_s=gpu_s)


def main():
    print("rapidfem-td GPU vs CPU benchmark on real example geometries")
    print(f"explicit LSERK4 transient, {BENCH_STEPS} steps per device")
    print("geometries: ring resonator, power divider, cavity\n")

    rows = []
    for name, build, run, resolutions in GEOMETRIES:
        for label, kwargs in resolutions:
            print(f"  building {name} ({label}) ...", flush=True)
            r = bench_one(name, build, run, label, kwargs)
            if "error" in r:
                ndof = r.get("n", "?")
                print(f"    SKIPPED {name} ({label}) "
                      f"n_dof={ndof}: {r['error']}")
            else:
                gpu_txt = ("no GPU" if r["gpu_s"] is None
                           else f"{r['gpu_s'] * 1e3:.0f} ms")
                print(f"    done: n_dof={r['n']}  build {r['build_s']:.1f}s  "
                      f"CPU {r['cpu_s'] * 1e3:.0f} ms  GPU {gpu_txt}")
            rows.append(r)

    # --- the benchmark table ---------------------------------------------
    print(f"\nGPU vs CPU explicit transient ({BENCH_STEPS} steps)")
    print(f"{'geometry':>16} {'res':>8} {'n_dof':>12} "
          f"{'CPU [ms]':>12} {'GPU [ms]':>12} {'speedup':>10}")
    for r in rows:
        if "error" in r and "n" not in r:
            print(f"{r['name']:>16} {r['label']:>8} {'-':>12} "
                  f"{'-':>12} {'-':>12} {'(build failed)':>10}")
        elif "error" in r:
            print(f"{r['name']:>16} {r['label']:>8} {r['n']:>12} "
                  f"{'-':>12} {'-':>12} {'(run failed)':>10}")
        elif r["gpu_s"] is None:
            print(f"{r['name']:>16} {r['label']:>8} {r['n']:>12} "
                  f"{r['cpu_s'] * 1e3:>12.1f} {'no GPU':>12} {'-':>10}")
        else:
            speedup = r["cpu_s"] / r["gpu_s"]
            print(f"{r['name']:>16} {r['label']:>8} {r['n']:>12} "
                  f"{r['cpu_s'] * 1e3:>12.1f} {r['gpu_s'] * 1e3:>12.1f} "
                  f"{speedup:>9.2f}x")

    failed = [r for r in rows if "error" in r]
    if failed:
        print(f"\n{len(failed)} resolution(s) did not complete:")
        for r in failed:
            print(f"  {r['name']} ({r['label']}): {r['error']}")


if __name__ == "__main__":
    main()
