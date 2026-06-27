# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Physics-phenomenon geometry tests — one module per phenomenon.

Each module builds a parametric geometry, solves via `harness.case`, and
asserts an extracted quantity (S-params, fields, modes, loss, conservation)
against a closed form in `harness.references`. All under the < 100 000 DOF
budget enforced by the harness.
"""
