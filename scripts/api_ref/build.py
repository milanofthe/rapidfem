#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""
RapidFEM docs, API extraction build.

For every released git tag (>= MIN_SUPPORTED_VERSION) this extracts the
RapidFEM Python API via griffe static analysis and writes:

    static/api/<tag>.json     full API data for that version
    static/api/latest.json    copy of the newest version
    static/api/versions.json  manifest (latest tag + ordered version list)
    src/lib/docs/api/versions.json  same manifest, importable by the app

Each tag is materialised in a detached git worktree so the working tree
is never touched. RapidFEM does not need to be installed.

Usage:
    python scripts/api_ref/build.py            # extract all supported tags
    python scripts/api_ref/build.py --head     # also extract the current worktree
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib.api import extract_api
from lib.config import (
    MIN_SUPPORTED_VERSION,
    PACKAGE,
    REPO_ROOT,
    SRC_API_DIR,
    STATIC_DIR,
)


def run_git(*args: str, cwd: Path = REPO_ROOT) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def parse_version(tag: str) -> tuple[int, int, int] | None:
    """Parse a 'vMAJOR.MINOR.PATCH' tag into a comparable tuple."""
    parts = tag.lstrip("v").split(".")
    try:
        nums = [int(p) for p in parts[:3]]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def list_tags() -> list[str]:
    """Return supported version tags, oldest first."""
    tags: list[tuple[str, tuple[int, int, int]]] = []
    for line in run_git("tag").splitlines():
        line = line.strip()
        version = parse_version(line)
        if version and version >= MIN_SUPPORTED_VERSION:
            tags.append((line, version))
    tags.sort(key=lambda t: t[1])
    return [t[0] for t in tags]


def extract_ref(ref: str) -> dict:
    """Extract the API for a git ref via a throwaway detached worktree."""
    with tempfile.TemporaryDirectory(prefix="rapidfem-docs-") as tmp:
        worktree = Path(tmp) / "tree"
        run_git("worktree", "add", "--detach", str(worktree), ref)
        try:
            source_path = worktree / PACKAGE["source_subdir"]
            return extract_api(PACKAGE["id"], source_path, PACKAGE["root_modules"])
        finally:
            run_git("worktree", "remove", "--force", str(worktree))


def extract_head() -> dict:
    """Extract the API from the current working tree (no worktree)."""
    source_path = REPO_ROOT / PACKAGE["source_subdir"]
    return extract_api(PACKAGE["id"], source_path, PACKAGE["root_modules"])


def write_api(api: dict, version: str, api_dir: Path) -> int:
    """Annotate and write one API JSON file. Returns the module count."""
    api["display_name"] = PACKAGE["display_name"]
    api["version"] = version
    n_modules = len(api.get("modules", {}))
    if n_modules == 0:
        return 0
    out = api_dir / f"{version}.json"
    out.write_text(json.dumps(api, indent=1), encoding="utf-8")
    return n_modules


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract RapidFEM API docs")
    parser.add_argument(
        "--head",
        action="store_true",
        help="also extract the current working tree as version 'dev'",
    )
    args = parser.parse_args()

    api_dir = STATIC_DIR / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    SRC_API_DIR.mkdir(parents=True, exist_ok=True)

    tags = list_tags()
    if not tags:
        print("No supported tags found.", file=sys.stderr)
        return 1
    print(f"Supported tags: {', '.join(tags)}")

    extracted: list[str] = []
    dates: dict[str, str] = {}

    for tag in tags:
        print(f"  {tag} ...", end=" ", flush=True)
        api = extract_ref(tag)
        n_modules = write_api(api, tag, api_dir)
        if n_modules == 0:
            print("no API, skipped")
            continue
        dates[tag] = run_git("log", "-1", "--format=%cs", tag)
        extracted.append(tag)
        print(f"{n_modules} modules")

    if not extracted:
        print("Nothing extracted.", file=sys.stderr)
        return 1

    latest = extracted[-1]

    if args.head:
        print("  dev (working tree) ...", end=" ", flush=True)
        api = extract_head()
        n_modules = write_api(api, "dev", api_dir)
        print(f"{n_modules} modules" if n_modules else "no API, skipped")
        if n_modules:
            dates["dev"] = run_git("log", "-1", "--format=%cs")

    # 'latest' is an alias for the newest released version.
    shutil.copyfile(api_dir / f"{latest}.json", api_dir / "latest.json")

    ordered = list(reversed(extracted))
    if args.head and (api_dir / "dev.json").exists():
        ordered = ["dev"] + ordered

    manifest = {
        "latest": latest,
        "versions": [{"tag": t, "date": dates.get(t, "")} for t in ordered],
    }
    manifest_json = json.dumps(manifest, indent=1)
    (api_dir / "versions.json").write_text(manifest_json, encoding="utf-8")
    (SRC_API_DIR / "versions.json").write_text(manifest_json, encoding="utf-8")

    print(f"Done. Latest = {latest}, {len(extracted)} versions extracted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
