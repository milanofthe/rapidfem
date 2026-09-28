# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Time-domain excitation waveforms.

An excitation is any callable ``g(t) -> float``; :class:`GaussianPulse` is the
common broadband / modulated choice for transient port drives.
"""
import numpy as np

__all__ = ["GaussianPulse"]


class GaussianPulse:
    """A Gaussian pulse, optionally modulated by a sinusoidal carrier.

    ``g(t) = exp(-((t-t0)/tau)^2) · cos(2π·f0·(t-t0))``

    With ``f0 = None`` the bare Gaussian is a smooth broadband pulse; with a
    carrier it is a band-limited pulse centred on ``f0``.

    Parameters
    ----------
    t0 : float
        Pulse-centre time, in the simulation's time units (seconds in SI).
    tau : float
        Gaussian width (the `1/e` half-width), in the same time units.
    f0 : float, optional
        Carrier frequency in the reciprocal units (Hz in SI); omit for a
        bare Gaussian.
    """

    def __init__(self, *, t0, tau, f0=None):
        self.t0 = float(t0)
        self.tau = float(tau)
        self.f0 = None if f0 is None else float(f0)

    def __call__(self, t):
        env = np.exp(-((np.asarray(t) - self.t0) / self.tau) ** 2)
        if self.f0 is None:
            return env
        return env * np.cos(2.0 * np.pi * self.f0 * (np.asarray(t) - self.t0))
