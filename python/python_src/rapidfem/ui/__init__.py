# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""rapidfem.ui, the local UI: the ``rapidfem`` CLI, the capture slot of
:func:`rapidfem.show`, the Flask backend and the bundled SvelteKit frontend.

Importing it is free of the UI's dependencies; serving needs the ``ui``
extra::

    pip install rapidfem[ui]
    rapidfem serve ./my_project/
"""
