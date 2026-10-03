# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""RapidFEM problem layer.

`ProblemFD` drives the frequency-domain (Nédélec-FEM) solver; `ProblemTD`
drives the time-domain DGTD solver.
"""
from .fd import ErrorIndicator, ProblemFD
from .td import ProblemTD

__all__ = ["ProblemFD", "ProblemTD", "ErrorIndicator"]
