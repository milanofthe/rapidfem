[![PySimHub](https://pysimhub.io/badge.svg)](https://pysimhub.io/projects/rapidfem)

# RapidFEM

Electromagnetic FEM solver written in Rust, distributed as a Python package on PyPI. Two backends sit
behind one geometry / material / physics API: a frequency-domain solver (Nédélec first-kind
curl-conforming elements, complex-symmetric sparse linear algebra) and a time-domain DGTD solver
(nodal discontinuous Galerkin, Krylov/ETD exponential time integration, model-order reduction). The
geometry is non-dimensionalised before assembly, so sub-micron RFIC passives and metre-scale
structures use the same numerical path. An optional Flask-based local UI provides a code editor and a
live viewer.

## Install

```bash
pip install rapidfem            # solver only
pip install rapidfem[ui]        # solver + local UI
```

Wheels for Windows, Linux, and macOS are built in CI; the Rust core is compiled ahead of time, so no
Rust toolchain is needed on the user's machine. Geometry and meshing run on the built-in
[rapidmesh](https://github.com/milanofthe/rapidmesh) mesher, the sparse solves on the built-in
rslab solver; there are no other native dependencies.

## Quick start (Python API)

```python
import numpy as np
import rapidfem as rf

# Build geometry; attach materials + physics directly to entities
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
air = g.box(22.86e-3, 10.16e-3, 30e-3, position=(-11.43e-3, -5.08e-3, 0),
            material=rf.Air())

rf.RectWaveguidePort(air.faces.min(axis="z"))
rf.RectWaveguidePort(air.faces.max(axis="z"))
rf.PEC(*air.faces.unassigned)

g.mesh()

# Define the problem once, run any number of analyses on it
prob = rf.Problem(g)                      # Problem is the frequency-domain ProblemFD
result = prob.sweep(np.linspace(8e9, 12e9, 21))
print(result.frequencies.shape, result.sparams.shape)

# The same Problem can also drive an eigenmode solve or a far-field pattern:
# modes   = prob.eigenmode(target_frequency=10e9, n_modes=6)
# pattern = prob.farfield(result, freq_idx=10, port_idx=0)
```

`python_src/rapidfem/examples/` has end-to-end runs: microstrip and coupled lines, iris and
stepped-impedance filters, patch / Vivaldi / inverted-F antennas (PML + far-field), pyramidal horns,
dielectric resonators, and the `fd_rfic_*` on-chip passives. RFIC geometry comes from a process stack
and layout via `rapidfem.rfic` (`rfic.Stack`, `Geometry.from_gds`).

## Importing external geometry and meshes

`g.load(path)` brings external geometry into the scene; the action is chosen from the file extension.

```python
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=20e9))

# A closed STL or OBJ surface becomes a solid, split into smooth faces at its
# creases: a normal GeoObject, so boolean it, transform it, select its faces,
# attach materials and physics, exactly like a g.box(...). unit= reads the
# file's coordinates, rotation= and position= place the part.
part = g.load("horn.stl", unit="MM", material=rf.Air(),
              rotation=(math.pi, (0, 0, 1)), position=(0, 0, 5e-3))
post = g.cylinder(radius=0.5e-3, height=5e-3)
g.cut(part, post)                                # compose with primitives
g.rotate(part, math.pi / 2, axis=(0, 1, 0))      # full transform API applies
rf.RectWaveguidePort(part.faces.max(axis="z"))
rf.PEC(*part.faces.unassigned)
g.mesh()

# A pre-built .msh volume mesh (gmsh MSH 4.1 or 2.2) is already tessellated, so
# loading one switches the geometry into mesh mode: its named physical groups
# become selectable handles for materials and physics. g.mesh() bakes the
# bindings (no remeshing) and the usual Problem/sweep pipeline runs unchanged.
# g.save_mesh(path) writes such a file, the groups named as the solver sees them.
g = rf.Geometry()
scene = g.load("waveguide.msh")
scene.group("air").material = rf.Air()
rf.RectWaveguidePort(scene.group("port_in"))
rf.RectWaveguidePort(scene.group("port_out"))
rf.PEC(scene.group("walls"))
g.mesh()
result = rf.Problem(g).sweep(np.linspace(8e9, 12e9, 21))
```

STEP, IGES and BREP import is not available yet (milanofthe/rapidmesh-dev#37).
`examples/fd_stl_import.py` is a full STL-driven sweep.

## Local UI

```bash
rapidfem serve ./my_project/
```

Opens a browser window with a CodeMirror Python editor, a 3D geometry / mesh / field viewer (raw
WebGL2), and S-parameter plots. The geometry view updates on save (`Ctrl+S`); mesh and solver runs
are explicit. Results stream in as the solve runs; fields are fetched on demand as you scrub
frequency and port. `rapidfem.show(g)` sends a geometry to the viewer.

## Features

- Geometry builder: OpenCASCADE primitives with boolean ops, transforms, and fillet/chamfer.
  Ready-made RF structures in `rf.structures` (coax, microstrip, CPW, stripline, waveguides, helix)
  build geometry and ports in one call.
- External CAD / mesh import: `g.load(path)` pulls in STEP / IGES / BREP solids as composable
  primitives, heals STL surfaces into meshable solids, or loads a pre-built `.msh` and exposes its
  named physical groups for material / physics binding.
- RFIC / GDS: `rapidfem.rfic` process stacks and `Geometry.from_gds` build on-chip passives, solved
  after non-dimensionalisation down to sub-micron features.
- Element: Nédélec first-kind curl-conforming tetrahedral elements, order 2 (20 DOFs per cell), in a
  hierarchical basis. Opt-in per-cell mixed order (1-2) via `[element] order_policy = "adaptive"`;
  the default is uniform order 2.
- Excitations: rectangular waveguide ports (arbitrary TE modes), lumped ports (TEM, multi-line
  voltage integral), coax and wave ports, Floquet plane-wave port (normal incidence), first-order
  absorbing boundary.
- PML: anisotropic stretched-coordinate perfectly matched layer.
- Lossy materials: complex permittivity with loss tangent and conductivity, surface impedance for
  metals, Debye dispersion; cached across sweeps.
- Sparse solvers: pure-Rust [`rslab`](https://github.com/milanofthe/rslab) real and complex-symmetric
  LDLᵀ (Bunch-Kaufman) with a-priori memory estimates, vendored, no native dependencies; driven
  sweeps, eigenmodes and port modes all factor through it.
- Frequency sweep: assembles E/B once, analyses the sparsity pattern once, and solves neighbouring
  points by COCG preconditioned with an earlier point's factorisation, refactoring only when the
  iteration gets expensive (2.6x faster on a 681k-DOF 21-point sweep).
- Eigenmode solver: shift-invert Lanczos in the B inner product, with a per-mode residual check.
- Error estimation: residual error indicator with Dörfler marking (`ProblemFD.element_errors`).
- Output: Touchstone (.s1p/.s2p/.snp), scikit-rf, far-field NFFT.
- Parallel assembly: rayon-based element matrix evaluation.

## Time-domain backend (DGTD)

`ProblemTD`, behind the same API, compiles a structure into an explicit linear ODE `dy/dt = A·y` and
exposes it as a model at every level:

- DGTD: nodal discontinuous Galerkin on tetrahedra, upwind or energy-conserving central flux.
- Exponential time integration: matrix-free Krylov/ETD propagator, exact for the linear system at any
  step size (no CFL limit).
- Model export / reduction: the RHS, the verbatim sparse operator `A`, an exponential stepper, or
  Krylov-projected reduced models.
- Materials: heterogeneous, lossy, anisotropic, and Debye-dispersive media; matched absorbing layers;
  periodic boundaries.
- Output: field probes, RFT transfer function, VTK field-animation export.

```python
import rapidfem as rf

ptd  = rf.ProblemTD.box(size=(1, 1, 1), cells=(2, 2, 2), order=2)
traj = ptd.transient(y0, dt=0.02, steps=200)   # turnkey transient
rom  = ptd.reduce(y0, dim=60)                   # model-order reduction
A    = ptd.state_space()                        # the verbatim operator
```

Method notes and the `ProblemTD` API are in [issue #31](https://github.com/milanofthe/rapidfem/issues/31).

## Sparse solver

Every factorisation runs on [`rslab`](https://github.com/milanofthe/rslab), vendored in
`vendor/rslab`: a pure-Rust complex-symmetric LDLᵀ (Bunch-Kaufman) with numeric-only
refactorisation across sweeps and an a-priori RAM gate.

## Verification

`cargo test --release` pins the element and solver kernels against independent symbolic derivations
to machine precision: the Nédélec element matrices, the barycentric integration coefficients, the
surface-element trace, the hierarchical-basis and exact-sequence properties, the global assembly, and
the eigensolver against a densely computed cavity spectrum. The derivations, completeness proofs, and
entrywise cross-checks live in [`derivations/`](derivations/).

`python/tests/` runs end-to-end S-parameter checks on microstrip, stripline, CPW, waveguide, filter,
and antenna geometries against closed-form values. Those run real solves, so they are run locally
(`cd python && pytest -m slow`), not in CI.

## Acknowledgments

rapidfem began as a Rust port of [EMerge](https://github.com/FennisRobert/EMerge)
([PyPI](https://pypi.org/project/emerge/)), Robert Fennis' open-source Python electromagnetic FEM
solver. Its script-first design shaped the geometry / material / physics API, and the early releases
ported its Python implementation to Rust. Thanks to Robert Fennis for building and sharing it.

The solver kernels have since been independently re-derived from primary sources: Nédélec's
edge-element construction, standard microwave theory, and the barycentric integration identities. The
derivations and cross-checks are in [`derivations/`](derivations/).

## License

AGPL-3.0-only; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright (C) Milan
Rother and rapidfem contributors; commercial terms available.
