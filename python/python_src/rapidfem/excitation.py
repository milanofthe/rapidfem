# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Time-domain excitation waveforms.

An excitation is any callable ``g(t) -> float``. :class:`GaussianPulse`, the
common broadband / modulated choice for transient port drives, is native: a
run samples it in Rust instead of calling into Python every step.
"""
from ._native import GaussianPulse

__all__ = ["GaussianPulse"]
