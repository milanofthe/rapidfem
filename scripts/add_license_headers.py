"""Add SPDX license headers to every .rs source file in rapidfem.

Every Rust source file gets the rapidfem header pointing at LICENSE for the
full GPL-3.0+ terms with the Gmsh additional permission.

Idempotent: skips files that already start with `// SPDX-License-Identifier:`.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADER_RAPIDFEM = """\
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

"""

SKIP_MARKER = "// SPDX-License-Identifier:"


def patch(path: Path, header: str) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith(SKIP_MARKER):
        return "skip"
    path.write_text(header + text, encoding="utf-8")
    return "patched"


def main() -> int:
    rs_files: list[Path] = []
    for crate in ("crates/rapidfem-core", "crates/rapidfem-fd", "crates/rapidfem-td", "python"):
        src = ROOT / crate / "src"
        if not src.exists():
            continue
        rs_files.extend(src.glob("**/*.rs"))

    n_rf = n_skip = 0
    for f in sorted(rs_files):
        rel = f.relative_to(ROOT).as_posix()
        status = patch(f, HEADER_RAPIDFEM)
        print(f"  {status:7s} {rel}")
        if status == "skip":
            n_skip += 1
        else:
            n_rf += 1
    print(f"\nPatched {n_rf} files; skipped {n_skip} already-headered files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
