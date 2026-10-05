# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""rapidfem, frequency- and time-domain electromagnetic FEM solver in Rust.

Quick start::

    import rapidfem as rf
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
    ...                                   # build geometry, assign physics
    g.mesh()
    result = rf.ProblemFD(g).sweep(freqs)
    print(result.frequencies.shape, result.sparams.shape)

Every sparse solve runs on the vendored pure-Rust rslab complex-symmetric
LDLᵀ solver; there are no native dependencies.
"""
from rapidfem._native import SweepResult, Eigenmode, RadiationPattern
from rapidfem.geometry import Geometry, GeoObject, EntityCollection, FaceCollection, EdgeCollection, MeshStats
from rapidfem.materials import (
    Material, Air, Dielectric, Conductor, Anisotropic, Debye, Drude,
)
from rapidfem.physics import (
    RectWaveguidePort, LumpedPort, CoaxPort, WavePort, UserDefinedPort,
    FloquetPort,
    PEC, PMC, ABC, SurfaceImpedance, LumpedElement, PML, PeriodicBoundary,
    FarFieldSurface,
)
from rapidfem.problem import ProblemFD, ProblemTD, ErrorIndicator
from rapidfem.excitation import GaussianPulse
from rapidfem import io  # registers .to_network/.to_hdf5 on SweepResult
from rapidfem import rfic  # RFIC: process stacks, the GDS model builder, rapidpassives bridge
from rapidfem import structures  # general RF structure builders (coax, microstrip, ...)
from rapidfem.ui import capture as _show_capture


_C0 = 299_792_458.0


def lambda_maxh(*, f_max: float, er_max: float = 1.0,
                mu_max: float = 1.0, per_lambda: int = 12) -> float:
    """Wavelength-based mesh size cap.

    Picks the largest tet edge length that still resolves the shortest
    wavelength in the model with ``per_lambda`` elements per wavelength,

    .. math::

        \\mathrm{maxh} = \\frac{c_0}{
            \\sqrt{\\varepsilon_{r,\\max}\\, \\mu_{r,\\max}}\\,
            f_{\\max}\\,
            n_{\\lambda}
        }

    where the smallest local wavelength lives in the highest-εᵣ
    material, that's what bounds the global cap.


    Note
    ----
    For rapidfem's element (Nédélec first kind, order 2) ``per_lambda = 8-12``
    is the usual range. Raise to 15 for stringent accuracy near a
    feature, drop to 6-8 for fast preview meshes.


    Example
    -------
    .. code-block:: python

        # Patch antenna on FR-4 at 2.8 GHz
        maxh = rf.lambda_maxh(f_max=2.8e9, er_max=4.4)
        g = rf.Geometry(maxh=maxh)


    Parameters
    ----------
    f_max : float
        highest frequency in the planned sweep, in Hz
    er_max : float
        largest relative permittivity in the model (defaults to 1)
    mu_max : float
        largest relative permeability (rarely > 1 in microwave work)
    per_lambda : int
        target elements per wavelength

    Returns
    -------
    float
        mesh size cap in metres, ready to pass to
        :class:`rapidfem.Geometry` or :meth:`Geometry.mesh`
    """
    if f_max <= 0:
        raise ValueError(f"f_max must be positive, got {f_max}")
    if er_max <= 0 or mu_max <= 0:
        raise ValueError(f"er_max and mu_max must be positive, got {er_max}, {mu_max}")
    if per_lambda <= 0:
        raise ValueError(f"per_lambda must be positive, got {per_lambda}")
    return _C0 / ((er_max * mu_max) ** 0.5 * f_max * per_lambda)


def show(obj, name: str = "default"):
    """Hand an object to the rapidfem viewer (no-op outside ``rapidfem serve``).

    In a plain Python run, ``show`` prints a one-line summary and
    returns ``obj`` unchanged, scripts behave the same on the command
    line. Under ``rapidfem serve`` (or during a static-demo bake), the
    kernel activates a capture slot; ``show`` forwards the object to
    the live 3-D viewer / S-parameter plot.


    Note
    ----
    Composes with assignment, ``show`` returns its argument unchanged,
    so the typical pattern is ``result = rf.show(prob.sweep(freqs))``.


    Example
    -------
    .. code-block:: python

        rf.show(g)                  # face preview pre-mesh, tet mesh post-mesh
        rf.show(prob)               # E-field point cloud (after .sweep())
        rf.show(result)             # |S-params| plot
        rf.show(ptd.transient(...)) # 3-D time-domain field animation


    Parameters
    ----------
    obj : Geometry, ProblemFD, SweepResult, list[Eigenmode], or a \
        time-domain result
        anything renderable by the UI; pre-mesh geometries render a
        coarse face preview, post-mesh ones render the FEM tet
        mesh; ProblemFD + SweepResult render :math:`|\\mathbf{E}(t, r)|^2`
        point clouds plus an S-parameter plot. The :class:`ProblemTD`
        verb results render too, a :meth:`~rapidfem.ProblemTD.transient`
        trajectory as a 3-D field animation, and
        :meth:`~rapidfem.ProblemTD.driven_transient` /
        :meth:`~rapidfem.ProblemTD.transfer_function` as time-series plots
    name : str
        display slot name; repeated ``show`` calls with the same name
        overwrite earlier outputs, different names allocate separate
        viewers

    Returns
    -------
    obj
        the same object, unchanged
    """
    kind = _show_capture.classify(obj)
    if _show_capture.is_capturing():
        _show_capture.capture(name=name, obj=obj, kind=kind)
    else:
        print(f"rapidfem.show({name}={type(obj).__name__}) [kind={kind}], run via `rapidfem serve` to see it in the UI.")
    return obj


__all__ = [
    "SweepResult", "Eigenmode", "RadiationPattern",
    "Geometry", "GeoObject", "EntityCollection", "FaceCollection", "EdgeCollection", "MeshStats",
    "Material", "Air", "Dielectric", "Conductor", "Anisotropic", "Debye", "Drude",
    "RectWaveguidePort", "LumpedPort", "CoaxPort", "WavePort",
    "UserDefinedPort", "FloquetPort",
    "PEC", "PMC", "ABC", "SurfaceImpedance", "LumpedElement", "PML",
    "PeriodicBoundary", "FarFieldSurface",
    "ProblemFD", "ProblemTD", "ErrorIndicator", "GaussianPulse",
    "io", "rfic", "structures", "show", "lambda_maxh",
]
__version__ = "0.23.0"
