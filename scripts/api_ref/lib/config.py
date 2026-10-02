# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""
Configuration for the RapidFEM docs build system.

Single package. API documentation is extracted statically (griffe,
allow_inspection=False) from the RapidFEM Python source — RapidFEM itself
does not need to be installed.
"""

from pathlib import Path

# Directory paths
SCRIPT_DIR = Path(__file__).parent.parent          # scripts/api_ref
REPO_ROOT = SCRIPT_DIR.parent.parent               # rapidfem
FRONTEND = REPO_ROOT / "python" / "python_src" / "rapidfem" / "ui" / "frontend-src"
STATIC_DIR = FRONTEND / "static"
SRC_API_DIR = FRONTEND / "src" / "lib" / "docs" / "api"

# The single documented package.
PACKAGE = {
    "id": "rapidfem",
    "display_name": "RapidFEM",
    # Python source location relative to the repo root.
    "source_subdir": Path("python") / "python_src",
    "root_modules": ["rapidfem"],
    "github_repo": "milanofthe/rapidfem",
}

# Only tags at or above this version are extracted. Earlier tags predate
# the current Python package layout.
MIN_SUPPORTED_VERSION = (0, 5, 0)

# Substring patterns; a module is skipped if any pattern occurs in its name.
# Excludes the CLI, native extension glue, the local UI, and example scripts.
SKIP_PATTERNS = [
    "_native",
    "_version",
    "bridge",
    "examples",
    ".ui",
    "__pycache__",
]
