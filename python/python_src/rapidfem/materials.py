# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

#########################################################################################
##
##                                  MATERIALS
##                                (materials.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from __future__ import annotations

from typing import Sequence



# HELPERS ===============================================================================


# DISPERSIVE MODELS =====================================================================

class Debye:
    """First-order Debye dispersion model for polar dielectrics.

    The complex relative permittivity follows

    .. math::

        \\varepsilon_r(\\omega) = \\varepsilon_\\infty
            + \\frac{\\varepsilon_s - \\varepsilon_\\infty}
                   {1 + j \\omega \\tau}

    where :math:`\\varepsilon_\\infty` is the high-frequency limit,
    :math:`\\varepsilon_s` is the static permittivity, and :math:`\\tau`
    is the relaxation time. Suitable for water-like dipolar relaxation
    in the microwave band.


    Example
    -------
    Pass to a :class:`Dielectric`-like volume material wrapper, or
    directly to the generic :class:`Material` constructor:

    .. code-block:: python

        water = rf.Material(debye=rf.Debye(
            er_inf=4.5,
            er_static=80.1,
            tau_s=8.27e-12,
        ))


    Parameters
    ----------
    er_inf : float
        high-frequency permittivity :math:`\\varepsilon_\\infty`
    er_static : float
        static / DC permittivity :math:`\\varepsilon_s`
    tau_s : float
        relaxation time in seconds
    """

    def __init__(self, *, er_inf: float, er_static: float, tau_s: float):
        self.er_inf = float(er_inf)
        self.er_static = float(er_static)
        self.tau_s = float(tau_s)


class Drude:
    """Drude dispersion model for metals and free-electron media.

    The complex relative permittivity follows

    .. math::

        \\varepsilon_r(\\omega) = \\varepsilon_\\infty
            - \\frac{\\omega_p^2}{\\omega (\\omega + j \\gamma)}

    where :math:`\\omega_p` is the plasma frequency and :math:`\\gamma`
    is the damping rate. At optical and infrared frequencies gold,
    silver, and aluminium are well described by this model.


    Example
    -------
    A Drude fit for gold near 200 THz:

    .. code-block:: python

        gold = rf.Material(drude=rf.Drude(
            er_inf=1.0,
            plasma_freq_hz=2.18e15,
            damping_freq_hz=6.46e12,
        ))


    Parameters
    ----------
    plasma_freq_hz : float
        plasma frequency :math:`\\omega_p / 2 \\pi` in Hz
    damping_freq_hz : float
        damping rate :math:`\\gamma / 2 \\pi` in Hz
    er_inf : float
        high-frequency permittivity (defaults to 1)
    """

    def __init__(self, *,
                 plasma_freq_hz: float,
                 damping_freq_hz: float,
                 er_inf: float = 1.0):
        self.plasma_freq_hz = float(plasma_freq_hz)
        self.damping_freq_hz = float(damping_freq_hz)
        self.er_inf = float(er_inf)



# BULK MATERIAL =========================================================================

class Material:
    """Generic linear material assigned to a volume in the geometry.

    Captures isotropic and diagonally anisotropic linear media. The
    constitutive relation seen by the Maxwell solver is

    .. math::

        \\mathbf{D} = \\varepsilon_0 \\varepsilon_r^\\ast \\mathbf{E},
        \\qquad
        \\mathbf{B} = \\mu_0 \\mu_r \\mathbf{H}

    with the complex relative permittivity

    .. math::

        \\varepsilon_r^\\ast = \\varepsilon_r (1 - j \\tan\\delta)
            - j \\frac{\\sigma}{\\omega \\varepsilon_0}

    Use the named subclasses (:class:`Air`, :class:`Dielectric`,
    :class:`Conductor`, :class:`Anisotropic`) for readability; this
    base class is the union of every parameter the solver supports.


    Note
    ----
    Materials are attached to volumes at construction time via the
    primitive's ``material=`` keyword. Multiple volumes can share one
    ``Material`` instance, they then share one material tag of the
    solver mesh.


    Example
    -------
    Direct construction with a custom mix of properties:

    .. code-block:: python

        lossy_pcb = rf.Material(
            er=4.4,
            tand=0.02,
            conductivity=0.0,
        )
        sub = g.box(W, H, T, material=lossy_pcb)


    Parameters
    ----------
    er : float
        relative permittivity :math:`\\varepsilon_r`
    ur : float
        relative permeability :math:`\\mu_r`
    tand : float
        electric loss tangent :math:`\\tan\\delta`
    conductivity : float
        bulk conductivity :math:`\\sigma` in S/m
    cond_diag : Sequence[float], optional
        diagonal anisotropic conductivity ``(σxx, σyy, σzz)`` in S/m,
        overrides scalar ``conductivity`` when set (e.g. homogenised
        via arrays: strong vertical, weak lateral conduction)
    er_diag : Sequence[float], optional
        diagonal anisotropic permittivity ``(εxx, εyy, εzz)``,
        overrides scalar ``er`` when set
    ur_diag : Sequence[float], optional
        diagonal anisotropic permeability ``(μxx, μyy, μzz)``,
        overrides scalar ``ur`` when set
    debye : Debye, optional
        first-order dispersive component
    drude : Drude, optional
        Drude-model dispersive component
    maxh : float, optional
        max tet edge length (m) applied to every volume carrying this
        material. Acts as a per-material refinement floor, useful for
        forcing finer cells inside thin conductors, narrow dielectric
        slabs, or any structure where the global ``maxh`` would
        under-resolve the field. An explicit ``maxh=`` on the primitive
        (``g.box(..., maxh=...)``) still wins; the material's value is
        the fallback when the primitive doesn't carry its own.


    Attributes
    ----------
    er, ur, tand, conductivity : float
        scalar isotropic parameters
    er_diag, ur_diag : tuple[float, float, float] or None
        anisotropic overrides
    debye, drude : Debye or Drude or None
        dispersive components
    maxh : float or None
        per-material mesh-size refinement floor (m)
    """

    def __init__(self, *,
                 er: float = 1.0,
                 ur: float = 1.0,
                 tand: float = 0.0,
                 conductivity: float = 0.0,
                 cond_diag: Sequence[float] | None = None,
                 er_diag: Sequence[float] | None = None,
                 ur_diag: Sequence[float] | None = None,
                 debye: Debye | None = None,
                 drude: Drude | None = None,
                 maxh: float | None = None):
        self.er = float(er)
        self.ur = float(ur)
        self.tand = float(tand)
        self.conductivity = float(conductivity)
        self.cond_diag = tuple(float(v) for v in cond_diag) if cond_diag is not None else None
        self.er_diag = tuple(float(v) for v in er_diag) if er_diag is not None else None
        self.ur_diag = tuple(float(v) for v in ur_diag) if ur_diag is not None else None
        self.debye = debye
        self.drude = drude
        self.maxh = float(maxh) if maxh is not None else None

    def _add_to(self, model, volume_tag: int) -> None:
        """place this material on the native model under a volume tag

        Parameters
        ----------
        model : rapidfem._native.Model
            the model being built
        volume_tag : int
            mesh tag of the volume this material is attached to
        """
        debye = drude = None
        if self.debye is not None:
            d = self.debye
            debye = (d.er_inf, d.er_static, d.tau_s)
        if self.drude is not None:
            d = self.drude
            drude = (d.er_inf, d.plasma_freq_hz, d.damping_freq_hz)
        vec = lambda v: None if v is None else [float(x) for x in v]
        model.add_material(
            volume_tag, er=self.er, ur=self.ur, tand=self.tand,
            conductivity=self.conductivity, cond_diag=vec(self.cond_diag),
            er_diag=vec(self.er_diag), ur_diag=vec(self.ur_diag),
            debye=debye, drude=drude)



# NAMED PRESETS =========================================================================

class Air(Material):
    """Vacuum or lossless air, :math:`\\varepsilon_r = \\mu_r = 1`,
    :math:`\\tan\\delta = \\sigma = 0`.

    The simplest material; sets every permittivity, permeability, and
    loss term to its free-space default. Convenient as a placeholder
    for air boxes, padding regions, and inside :class:`PML` slabs
    (the PML's coordinate stretch overrides the bulk permittivity, so
    the material here only fills the volume role).


    Example
    -------
    Air-filled inner region of a waveguide:

    .. code-block:: python

        air = g.box(A, B, L, material=rf.Air())


    Parameters
    ----------
    maxh : float, optional
        per-material mesh-size refinement floor (m); see :class:`Material`
    """

    def __init__(self, *, maxh: float | None = None):
        super().__init__(maxh=maxh)


class Dielectric(Material):
    """Isotropic lossy dielectric.

    Models a homogeneous insulator with relative permittivity
    :math:`\\varepsilon_r` and an optional loss tangent
    :math:`\\tan\\delta` or bulk conductivity. The effective
    complex permittivity is

    .. math::

        \\varepsilon_r^\\ast = \\varepsilon_r (1 - j \\tan\\delta)
            - j \\frac{\\sigma}{\\omega \\varepsilon_0}


    Example
    -------
    FR-4 PCB substrate:

    .. code-block:: python

        sub = g.box(W, L, H, material=rf.Dielectric(
            er=4.4,
            tand=0.02,
        ))


    Parameters
    ----------
    er : float
        relative permittivity (required, must be > 0)
    tand : float
        electric loss tangent (defaults to lossless)
    ur : float
        relative permeability (defaults to 1; rarely non-unit in
        microwave work)
    conductivity : float
        bulk conductivity in S/m, added on top of ``tand``
    cond_diag : Sequence[float], optional
        diagonal anisotropic conductivity ``(σxx, σyy, σzz)`` in S/m,
        overrides scalar ``conductivity`` (homogenised via arrays)
    maxh : float, optional
        per-material mesh-size refinement floor (m); see :class:`Material`
    """

    def __init__(self, er: float, *,
                 tand: float = 0.0,
                 ur: float = 1.0,
                 conductivity: float = 0.0,
                 cond_diag: Sequence[float] | None = None,
                 maxh: float | None = None):
        super().__init__(er=er, ur=ur, tand=tand, conductivity=conductivity,
                         cond_diag=cond_diag, maxh=maxh)


class Conductor(Material):
    """Bulk lossy conductor.

    Models thick metal where the wave penetrates a non-negligible
    fraction of the skin depth, the bulk conductivity :math:`\\sigma`
    drives Ohmic loss via the volume current
    :math:`\\mathbf{J} = \\sigma \\mathbf{E}`.


    Note
    ----
    For thin metal sheets where the skin depth is much smaller than
    the conductor thickness, use :class:`SurfaceImpedance` on the face
    instead, it captures the loss in a 2-D boundary condition without
    paying for volumetric mesh inside the metal.


    Example
    -------
    Lossy copper plug filling a coaxial section:

    .. code-block:: python

        copper = g.cylinder(radius=r, height=h,
                            material=rf.Conductor(conductivity=5.8e7))


    Parameters
    ----------
    conductivity : float
        bulk conductivity :math:`\\sigma` in S/m
    ur : float
        relative permeability (defaults to 1)
    er : float
        relative permittivity (defaults to 1; metal-bulk runs are
        dominated by the conductivity term so this rarely matters)
    maxh : float, optional
        per-material mesh-size refinement floor (m); see :class:`Material`
    """

    def __init__(self, *, conductivity: float,
                 ur: float = 1.0, er: float = 1.0,
                 maxh: float | None = None):
        super().__init__(er=er, ur=ur, conductivity=conductivity, maxh=maxh)


class Anisotropic(Material):
    """Diagonally anisotropic linear material.

    Permittivity and/or permeability are diagonal tensors with
    independent ``(xx, yy, zz)`` components,

    .. math::

        \\varepsilon_r = \\operatorname{diag}
            (\\varepsilon_{xx}, \\varepsilon_{yy}, \\varepsilon_{zz}),
        \\qquad
        \\mu_r = \\operatorname{diag}
            (\\mu_{xx}, \\mu_{yy}, \\mu_{zz})

    Useful for uniaxial composites, ferrite biasing, and engineered
    metamaterial substrates.


    Example
    -------
    Uniaxial substrate with εzz different from εxx = εyy:

    .. code-block:: python

        sub = g.box(W, L, H, material=rf.Anisotropic(
            er_diag=(3.0, 3.0, 3.4),
        ))


    Parameters
    ----------
    er_diag : Sequence[float], optional
        diagonal permittivity ``(εxx, εyy, εzz)``
    ur_diag : Sequence[float], optional
        diagonal permeability ``(μxx, μyy, μzz)``
    tand : float
        electric loss tangent applied on top of ``er_diag``
    conductivity : float
        bulk conductivity in S/m
    maxh : float, optional
        per-material mesh-size refinement floor (m); see :class:`Material`
    """

    def __init__(self, *,
                 er_diag: Sequence[float] | None = None,
                 ur_diag: Sequence[float] | None = None,
                 tand: float = 0.0,
                 conductivity: float = 0.0,
                 maxh: float | None = None):
        super().__init__(tand=tand, conductivity=conductivity,
                         er_diag=er_diag, ur_diag=ur_diag, maxh=maxh)


__all__ = [
    "Material", "Air", "Dielectric", "Conductor", "Anisotropic",
    "Debye", "Drude",
]
