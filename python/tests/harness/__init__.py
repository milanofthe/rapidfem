# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Phenomenon-test harness: build/solve helpers + analytical references.

See `harness.case` for the build→mesh→solve interface (with the DOF budget)
and `harness.references` for the closed-form solutions tests assert against.
"""
from . import case, references

__all__ = ["case", "references"]
