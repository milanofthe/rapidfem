# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""pytest configuration for the rapidfem test suite.

Layout:
  tests/harness/      build→solve helpers + analytical references
  tests/geometries/   one module per physics phenomenon (the breadth suite)
  tests/kernel/       sympy/analytical kernel checks driven from Python (if any)

Markers (also declared in pyproject [tool.pytest.ini_options]):
  slow         full FD sweep or TD transient; opt out with -m "not slow"
  phenomenon   a physics-phenomenon geometry test

`tests/` is on sys.path so test modules can `from harness import case`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
