# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Analytical reference solutions for the phenomenon test suite.

Every geometry test asserts the solver against a *closed form* or a
conservation law — never against another solver. This module collects the
shared closed forms (transmission-line, waveguide, cavity, loss, dispersion).
Niche one-off formulas can live in the test module that uses them; put a
formula here only once a second test needs it, to keep parallel edits
conflict-free.

All SI units. Frequencies in Hz, lengths in m, conductivity in S/m.
"""
from __future__ import annotations

import numpy as np

# ── Physical constants (CODATA, matching crates/.../constants.rs) ──────────
C0 = 299_792_458.0                 # speed of light, m/s
MU0 = 4.0e-7 * np.pi               # vacuum permeability, H/m
EPS0 = 1.0 / (MU0 * C0 * C0)       # vacuum permittivity, F/m
ETA0 = MU0 * C0                    # vacuum wave impedance, ≈376.730 Ω


def k0(f: float) -> float:
    """Free-space wavenumber 2πf/c."""
    return 2.0 * np.pi * f / C0


# ── Rectangular waveguide (TE_mn / TM_mn) ──────────────────────────────────
def rect_cutoff_freq(a: float, b: float, m: int = 1, n: int = 0, er: float = 1.0) -> float:
    """Cutoff frequency of mode (m,n) in an a×b guide filled with εr."""
    return C0 / (2.0 * np.sqrt(er)) * np.hypot(m / a, n / b)


def rect_beta(f: float, a: float, b: float | None = None, m: int = 1, n: int = 0,
              er: float = 1.0) -> float:
    """Propagation constant β of mode (m,n). Real above cutoff, else 0.

    β = sqrt(εr·k0² − k_c²),  k_c = π·hypot(m/a, n/b).
    """
    kc = np.pi * np.hypot(m / a, 0.0 if b is None else n / b)
    k = k0(f) * np.sqrt(er)
    arg = k * k - kc * kc
    return float(np.sqrt(arg)) if arg > 0 else 0.0


def rect_te_impedance(f: float, a: float, b: float | None = None, m: int = 1,
                      n: int = 0, er: float = 1.0) -> float:
    """Wave impedance of a TE_mn mode: Z_TE = ωμ0/β."""
    beta = rect_beta(f, a, b, m, n, er)
    return 2.0 * np.pi * f * MU0 / beta


# ── Rectangular cavity (a×b×d, PEC walls) ──────────────────────────────────
def rect_cavity_freqs(a: float, b: float, d: float, er: float = 1.0,
                      n_modes: int = 8) -> list[float]:
    """Lowest `n_modes` resonant frequencies of a PEC box a×b×d filled with εr.

    f_mnp = c/(2√εr) · sqrt((m/a)² + (n/b)² + (p/d)²), with at most one index
    zero (a valid mode needs ≥2 non-zero indices for TE/TM in a box).
    """
    out = []
    for m in range(0, 4):
        for n in range(0, 4):
            for p in range(0, 4):
                if (m == 0) + (n == 0) + (p == 0) >= 2:
                    continue
                out.append(C0 / (2.0 * np.sqrt(er))
                           * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2))
    return sorted(out)[:n_modes]


# ── TEM transmission line ──────────────────────────────────────────────────
def tem_phase(f: float, length: float, er: float = 1.0) -> float:
    """Insertion phase of a matched TEM line of given length: −β·L."""
    return -k0(f) * np.sqrt(er) * length


def coax_z0(ri: float, ro: float, er: float = 1.0) -> float:
    """Characteristic impedance of a coaxial line: (η0/2π)·ln(ro/ri)/√εr."""
    return ETA0 / (2.0 * np.pi) * np.log(ro / ri) / np.sqrt(er)


def stripline_z0(w: float, b: float, er: float = 1.0) -> float:
    """Stripline Z0 (Cohn/Wheeler approx). w = strip width, b = plate spacing."""
    we = w  # thin-strip limit; callers needing the t>0 correction extend locally
    return 30.0 * np.pi / np.sqrt(er) * b / (we + 0.441 * b)


def microstrip_eeff(w: float, h: float, er: float) -> float:
    """Quasi-static effective permittivity of a microstrip (Hammerstad)."""
    u = w / h
    return (er + 1) / 2 + (er - 1) / 2 * (1 + 12 / u) ** -0.5


# ── Conductor / skin effect ────────────────────────────────────────────────
def skin_depth(f: float, sigma: float, mu_r: float = 1.0) -> float:
    """Skin depth δ = 1/√(π f μ σ)."""
    return 1.0 / np.sqrt(np.pi * f * mu_r * MU0 * sigma)


def surface_resistance(f: float, sigma: float, mu_r: float = 1.0) -> float:
    """Surface resistance R_s = √(π f μ / σ) = 1/(σ δ)."""
    return np.sqrt(np.pi * f * mu_r * MU0 / sigma)


def wr_te10_wall_attenuation(f: float, a: float, b: float, sigma: float,
                             er: float = 1.0) -> float:
    """TE10 conductor attenuation αc (Np/m) of an a×b guide (Pozar eq. 3.96).

    αc = R_s / (b·η·√(1−(fc/f)²)) · (1 + (2b/a)·(fc/f)²).
    """
    rs = surface_resistance(f, sigma)
    eta = ETA0 / np.sqrt(er)
    r = (rect_cutoff_freq(a, b, 1, 0, er) / f) ** 2
    return rs / (b * eta * np.sqrt(1.0 - r)) * (1.0 + (2.0 * b / a) * r)


# ── Dispersion ─────────────────────────────────────────────────────────────
def debye_eps(f: float, er_inf: float, er_static: float, tau: float) -> complex:
    """Complex relative permittivity of a Debye medium at frequency f."""
    w = 2.0 * np.pi * f
    return er_inf + (er_static - er_inf) / (1.0 + 1j * w * tau)


def drude_eps(f: float, plasma_hz: float, damping_hz: float,
              er_inf: float = 1.0) -> complex:
    """Complex relative permittivity of a Drude (free-electron) medium."""
    w = 2.0 * np.pi * f
    wp = 2.0 * np.pi * plasma_hz
    g = 2.0 * np.pi * damping_hz
    return er_inf - wp * wp / (w * (w + 1j * g))
