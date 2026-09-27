"""Vendor the rapidmesh library crates into vendor/rapidmesh.

Usage::

    python scripts/vendor_rapidmesh.py <path to a rapidmesh checkout> [git ref]

The ref defaults to ``origin/master``; the working tree of the checkout is
never touched (the files come from ``git archive``). Only the crates behind
the ``rapidmesh`` facade are copied, without tests, benches, examples, binaries
and dev-dependencies. The upstream workspace manifest is trimmed to those
crates and kept as the vendored tree's own ``[workspace]``, so it stays out of
the rapidfem workspace (like vendor/rslab). Rerun to resync.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "vendor" / "rapidmesh"
CRATES = [
    "rapidmesh",
    "rapidmesh-exact",
    "rapidmesh-geom",
    "rapidmesh-csg",
    "rapidmesh-brep",
    "rapidmesh-tet",
    "rapidmesh-topo",
]
DROP_DIRS = ("tests", "benches", "examples", "src/bin")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True).stdout


def strip_sections(manifest: str, names: tuple[str, ...]) -> str:
    """Drop the TOML tables `[name]` / `[[name]]` (up to the next table)."""
    out, skip = [], False
    for line in manifest.splitlines(keepends=True):
        header = re.match(r"^\[\[?([^\]]+)\]\]?\s*$", line)
        if header:
            skip = header.group(1).strip() in names
        if not skip:
            out.append(line)
    return "".join(out)


def workspace_manifest(upstream: str, commit: str) -> str:
    text = strip_sections(upstream, ("profile.bench",))
    members = ",\n".join(f'    "crates/{c}"' for c in CRATES)
    text = re.sub(r"members = \[.*?\]", f"members = [\n{members},\n]", text, flags=re.S)
    # workspace dependencies of crates that are not vendored
    text = "".join(line for line in text.splitlines(keepends=True)
                   if not re.match(r"^rapidmesh-(testutil|wasm)\s*=", line))
    header = (
        f"# Vendored copy of rapidmesh (milanofthe/rapidmesh-dev) at commit\n"
        f"# {commit}, written by rapidfem's scripts/vendor_rapidmesh.py: the\n"
        f"# library crates behind the `rapidmesh` facade, without tests,\n"
        f"# benches, examples, binaries or dev-dependencies. The own [workspace]\n"
        f"# keeps the vendored tree out of the consuming workspace.\n"
    )
    return header + text


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    ref = sys.argv[2] if len(sys.argv) > 2 else "origin/master"
    commit = git(repo, "rev-parse", "--short", ref).decode().strip()

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    paths = [f"crates/{c}" for c in CRATES] + ["LICENSE", "Cargo.lock"]
    tar = tarfile.open(fileobj=io.BytesIO(git(repo, "archive", ref, *paths)))
    tar.extractall(DEST, filter="data")

    for crate in CRATES:
        base = DEST / "crates" / crate
        for d in DROP_DIRS:
            shutil.rmtree(base / d, ignore_errors=True)
        manifest = base / "Cargo.toml"
        text = strip_sections(manifest.read_text(),
                              ("dev-dependencies", "bench", "example", "bin", "test"))
        manifest.write_text(text)

    upstream = git(repo, "show", f"{ref}:Cargo.toml").decode()
    (DEST / "Cargo.toml").write_text(workspace_manifest(upstream, commit))
    print(f"vendored rapidmesh {commit} into {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
