# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Give every rapidfem Rust and Python source file the SPDX license header.

    // SPDX-License-Identifier: AGPL-3.0-only
    //
    // Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

(``#`` comments in Python). An existing header block at the top of a file
(the SPDX line and the copyright notice lines after it) is replaced, so
the script also moves old headers to the current terms. A shebang line
stays first. Vendored code and the examples are never touched.

Idempotent: a file that already carries the current header is skipped.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YEARS = "2024-2026"
SPDX = "AGPL-3.0-only"

# Source trees of rapidfem's own code.
TREES = ["crates", "python/src", "python/python_src", "python/tests", "scripts", "derivations"]
# Examples are read as notebooks by the UI; a header would show as a cell.
SKIP_DIRS = {"examples", "frontend-src", "node_modules", "target", "__pycache__", ".venv"}


def header(prefix: str) -> list[str]:
    return [f"{prefix} SPDX-License-Identifier: {SPDX}", prefix,
            f"{prefix} Copyright (C) {YEARS} Milan Rother and rapidfem contributors"]


def files() -> list[Path]:
    out = []
    for tree in TREES:
        for ext in ("*.rs", "*.py"):
            for f in (ROOT / tree).rglob(ext):
                if not SKIP_DIRS & set(f.relative_to(ROOT).parts):
                    out.append(f)
    return sorted(out)


# The lines an old header held after its SPDX line.
_OLD = ("Copyright (C)", "This file is part of rapidfem", "the Gmsh additional permission")


def _header_line(line: str, prefix: str) -> bool:
    return line.rstrip() == prefix or (line.startswith(prefix + " ") and any(s in line for s in _OLD))


def patch(path: Path) -> str:
    prefix = "//" if path.suffix == ".rs" else "#"
    lines = path.read_text(encoding="utf-8").split("\n")
    shebang = [lines.pop(0)] if lines and lines[0].startswith("#!") else []
    new = header(prefix)
    if lines[:3] == new:
        return "skip"
    # Drop an old header: the SPDX line and the comment lines after it.
    if lines and re.match(rf"{re.escape(prefix)} SPDX-License-Identifier:", lines[0]):
        i = 1
        while i < len(lines) and _header_line(lines[i], prefix):
            i += 1
        lines = lines[i:]
        while lines and lines[0].strip() == "":
            lines.pop(0)
        status = "updated"
    else:
        status = "added"
    path.write_text("\n".join(shebang + new + [""] + lines), encoding="utf-8")
    return status


def main() -> int:
    counts: dict[str, int] = {}
    for f in files():
        s = patch(f)
        counts[s] = counts.get(s, 0) + 1
        if s != "skip":
            print(f"  {s:7s} {f.relative_to(ROOT).as_posix()}")
    print(", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
