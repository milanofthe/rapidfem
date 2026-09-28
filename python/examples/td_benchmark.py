"""Time-domain DGTD performance benchmark on a real unstructured mesh.

Where the Rust `bench` example measures the DGTD hot path on a structured
box, this measures it on the geometry a production run actually sees: the
dielectric ring resonator of ``td_ring_resonator.py``, a torus embedded
in an air cavity, meshed unstructured. That matters for the flux
term's neighbour gather: on a structured box neighbouring elements sit
close in memory, on an unstructured mesh the element numbering scatters them,
which is the access pattern the operator hits in practice.

Two questions are measured, per mesh resolution:

1. Where the exponential step's time goes: ``rhs`` matvec throughput, the
   adaptive (``KRYLOV_TOL``) and fixed-dimension (``tol=0``) step costs,
   and the matvec / orthogonalisation split.
2. Exponential vs explicit integrator: an exponential step is unbounded
   in size but expensive; an explicit LSERK4 step is cheap but CFL-bound.
   The fair measure is wall-time per unit of simulated time, so this
   estimates the spectral radius, finds the explicit CFL limit, and
   compares the two integrators on that basis.

Run after ``maturin develop --release``:

    python python/examples/td_benchmark.py
"""

import time

import numpy as np

import rapidfem as rf

mm = 1e-3

# Ring-resonator geometry, identical to examples/td_ring_resonator.py.
R_MAJ = 11.0 * mm        # ring radius, tube-centre to torus axis
R_MIN = 2.6 * mm         # ring tube (cross-section) radius
ER = 10.0                # high-permittivity ceramic ring
BOX = 38.0 * mm          # cubic air-cavity edge

KRYLOV_DIM = 40          # Krylov-dimension cap, the solver default
DT = 4e-12               # transient step of the ring-resonator example
PROPAGATE_STEPS = 10     # steps run to reach a representative mid-transient
                         # state (a smooth field, not a raw delta pulse)
POWER_ITERS = 40         # power-iteration count for the spectral radius
CFL_PROBE_STEPS = 20     # explicit steps run per CFL stability probe


def build(maxh_air, maxh_ring):
    """A meshed ProblemTD for the ring resonator at the given resolution."""
    g = rf.Geometry(maxh=maxh_air)
    air = g.box(BOX, BOX, BOX, position=(-BOX / 2, -BOX / 2, -BOX / 2),
                material=rf.Air())
    ring = g.torus(R_MAJ, R_MIN, material=rf.Dielectric(er=ER),
                   maxh=maxh_ring)
    g.fragment(air, ring)
    rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
           air.faces.min(axis="y"), air.faces.max(axis="y"),
           air.faces.min(axis="z"), air.faces.max(axis="z"))
    g.mesh()
    return rf.ProblemTD(g, order=2, flux="upwind")


def median(reps, f):
    """Median wall-clock seconds of `reps` runs of `f`."""
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def representative_state(ptd):
    """A mid-transient field state: a probe-point pulse propagated a few
    steps, so the Krylov benchmark sees a smooth field rather than the
    delta pulse the adaptive subspace would find artificially hard."""
    y = np.zeros(ptd.n_dof)
    y[ptd.probe_dof((R_MAJ, 0.0, 0.0), field="E", component="z")] = 1.0
    for _ in range(PROPAGATE_STEPS):
        y = ptd.step(y, DT, KRYLOV_DIM)
    return y


def spectral_radius(ptd):
    """Largest |eigenvalue| of A (solver units), by power iteration: the
    magnitude ratio ||A.v|| / ||v|| approaches rho as v aligns with the
    largest-magnitude eigenvector."""
    rng = np.random.default_rng(1)
    v = rng.standard_normal(ptd.n_dof)
    v /= np.linalg.norm(v)
    rho = 0.0
    for _ in range(POWER_ITERS):
        av = ptd.rhs(v)
        rho = np.linalg.norm(av)          # ||v|| = 1 each iteration
        v = av / rho
    return rho


def cfl_limit(ptd, rho):
    """Largest physical step that keeps LSERK4 bounded. Probes the
    dimensionless product z = h_solver * rho across a bracket and returns
    (z, h_physical) for the largest stable z, with a safety margin."""
    y0 = np.zeros(ptd.n_dof)
    y0[ptd.probe_dof((R_MAJ, 0.0, 0.0), field="E", component="z")] = 1.0
    stable = None
    for z in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        h = z / (ptd.c * rho)            # h_solver = c * h, want h_solver*rho = z
        y = y0.copy()
        for _ in range(CFL_PROBE_STEPS):
            y = ptd.step_explicit(y, h)
        if np.all(np.isfinite(y)) and np.linalg.norm(y) < 1e4:
            stable = z
        else:
            break
    if stable is None:
        return None, None
    z_safe = 0.9 * stable                # back off from the stability edge
    return z_safe, z_safe / (ptd.c * rho)


def main():
    print("rapidfem-td performance benchmark on a real unstructured mesh")
    print("geometry: dielectric ring resonator (torus in a PEC air cavity)")
    print(f"krylov-dim cap {KRYLOV_DIM}, exponential dt {DT:g} s\n")

    # Coarse to fine, spanning the 1e5 to 1e6 state-DOF production regime.
    resolutions = [
        (6.0 * mm, 3.0 * mm),
        (4.5 * mm, 2.3 * mm),
        (3.4 * mm, 1.8 * mm),
        (2.7 * mm, 1.5 * mm),
    ]

    rows = []
    for maxh_air, maxh_ring in resolutions:
        ptd = build(maxh_air, maxh_ring)
        n = ptd.n_dof
        y = representative_state(ptd)

        ptd.rhs(y)  # warm
        t_rhs = median(50, lambda: ptd.rhs(y))

        # Adaptive step (production, KRYLOV_TOL) and the fixed-dimension
        # worst case (tol=0, the full KRYLOV_DIM matvecs).
        ptd.step(y, DT, KRYLOV_DIM)
        t_adaptive = median(10, lambda: ptd.step(y, DT, KRYLOV_DIM))
        ptd.step(y, DT, KRYLOV_DIM, tol=0.0)
        t_fixed = median(10, lambda: ptd.step(y, DT, KRYLOV_DIM, tol=0.0))

        # Explicit integrator: spectral radius, CFL limit, step cost.
        rho = spectral_radius(ptd)
        z_cfl, h_cfl = cfl_limit(ptd, rho)
        ptd.step_explicit(y, h_cfl)  # warm
        t_explicit = median(20, lambda: ptd.step_explicit(y, h_cfl))

        # GPU vs CPU explicit transient on this real unstructured mesh.
        tr_steps = 40
        t = time.perf_counter()
        ptd.transient(y, dt=DT, steps=tr_steps, method="explicit",
                      device="cpu", verbose=False)
        cpu_tr = time.perf_counter() - t
        gpu_tr = None
        if ptd._op.gpu_available():
            ptd.transient(y, dt=DT, steps=3, device="gpu", verbose=False)
            t = time.perf_counter()
            ptd.transient(y, dt=DT, steps=tr_steps, device="gpu",
                          verbose=False)
            gpu_tr = time.perf_counter() - t

        rows.append(dict(n=n, t_rhs=t_rhs, t_adaptive=t_adaptive,
                         t_fixed=t_fixed, rho=rho, z_cfl=z_cfl,
                         h_cfl=h_cfl, t_explicit=t_explicit,
                         cpu_tr=cpu_tr, gpu_tr=gpu_tr, tr_steps=tr_steps))

    # --- where the exponential step's time goes ---------------------------
    print("exponential step breakdown")
    print(f"{'tets':>8} {'n_dof':>10} {'rhs [ms]':>10} {'Mdof/s':>9} "
          f"{'adaptive':>11} {'fixed [ms]':>11} {'matvec':>9} "
          f"{'ortho':>9} {'ortho%':>7}")
    for r in rows:
        matvec = r["t_rhs"] * KRYLOV_DIM
        ortho = max(r["t_fixed"] - matvec, 0.0)
        print(f"{r['n'] // 60:>8} {r['n']:>10} {r['t_rhs'] * 1e3:>10.3f} "
              f"{r['n'] / r['t_rhs'] / 1e6:>9.1f} "
              f"{r['t_adaptive'] * 1e3:>10.2f} {r['t_fixed'] * 1e3:>11.2f} "
              f"{matvec * 1e3:>9.2f} {ortho * 1e3:>9.2f} "
              f"{100 * ortho / r['t_fixed']:>6.0f}%")

    # --- exponential vs explicit, per unit of simulated time --------------
    # An exponential step covers DT; an explicit LSERK4 step covers only
    # h_cfl. Cost per simulated second is t_step / step_size; the cheaper
    # integrator is the one with the smaller ratio.
    print("\nintegrator comparison (wall-seconds per nanosecond simulated)")
    print(f"{'n_dof':>10} {'h_cfl [s]':>12} {'DT/h_cfl':>9} "
          f"{'expl [ms]':>10} {'exp s/ns':>10} {'expl s/ns':>10} "
          f"{'winner':>16}")
    for r in rows:
        exp_cost = r["t_adaptive"] / DT * 1e-9
        expl_cost = r["t_explicit"] / r["h_cfl"] * 1e-9
        if expl_cost < exp_cost:
            verdict = f"explicit {exp_cost / expl_cost:.2f}x"
        else:
            verdict = f"exponential {expl_cost / exp_cost:.2f}x"
        print(f"{r['n']:>10} {r['h_cfl']:>12.3e} {DT / r['h_cfl']:>9.1f} "
              f"{r['t_explicit'] * 1e3:>10.2f} {exp_cost:>10.2f} "
              f"{expl_cost:>10.2f} {verdict:>16}")

    # --- GPU vs CPU explicit transient on the real unstructured mesh ------
    tr_steps = rows[0]["tr_steps"]
    print(f"\nGPU vs CPU explicit transient ({tr_steps} steps)")
    print(f"{'n_dof':>10} {'CPU [ms]':>12} {'GPU [ms]':>12} {'speedup':>10}")
    for r in rows:
        if r["gpu_tr"] is None:
            print(f"{r['n']:>10} {r['cpu_tr'] * 1e3:>12.1f} "
                  f"{'no GPU':>12} {'-':>10}")
        else:
            print(f"{r['n']:>10} {r['cpu_tr'] * 1e3:>12.1f} "
                  f"{r['gpu_tr'] * 1e3:>12.1f} "
                  f"{r['cpu_tr'] / r['gpu_tr']:>9.2f}x")


if __name__ == "__main__":
    main()
