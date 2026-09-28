# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""RapidFEM problem layer.

`ProblemFD` drives the frequency-domain (Nédélec-FEM) solver; `ProblemTD`
drives the time-domain DGTD solver. `Problem` is retained as an alias of
`ProblemFD` so existing code keeps working unchanged.
"""
from .fd import Adaptive, ErrorIndicator, ProblemFD
from .td import ProblemTD

# Backward-compatible alias, `rf.Problem` continues to mean the
# frequency-domain problem.
Problem = ProblemFD

__all__ = ["Problem", "ProblemFD", "ProblemTD", "Adaptive", "ErrorIndicator"]
