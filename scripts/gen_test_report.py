"""Generate the unified rapidfem test report (Rust + Python).

LOCAL DEVELOPER TOOL — rapidfem testing runs exclusively on the developer
machine, never in CI. FEM benchmarks in CI runners are explicitly out of
scope; nothing here touches .github/workflows.

What it does
------------
1. (default on) Runs the Rust bridge (``report.rust_bridge``) which executes
   ``cargo test`` for the rapidfem crates and writes Rust sections to
   ``_report/rust-sections.json``. Skip with ``--no-rust``; restrict the
   crate set with ``--rust-crates``.
2. Runs pytest with the report plugin, merging the Rust sections in via
   ``--report-merge`` so Python and Rust land in one HTML report.

Extra arguments after the known flags are forwarded verbatim to pytest, e.g.::

    python scripts/gen_test_report.py -m slow -k waveguide
    python scripts/gen_test_report.py --no-rust python/tests/test_foo.py

The same Python interpreter (``sys.executable``) is used for the pytest run.
"""
from __future__ import annotations

import argparse
import os
import sys

# Repo root: scripts/gen_test_report.py -> parent.
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TESTS_DIR = os.path.join("python", "tests")

DEFAULT_HTML = os.path.join("_report", "rapidfem-test-report.html")
DEFAULT_JSON = os.path.join("_report", "rapidfem-test-report.json")
DEFAULT_RUST_JSON = os.path.join("_report", "rust-sections.json")


def _has_path(pytest_args) -> bool:
    """True if any forwarded arg looks like a test path (a file/dir, a
    nodeid with ``::``, or an existing path), as opposed to a flag/filter."""
    skip_next = False
    for a in pytest_args:
        if skip_next:
            skip_next = False
            continue
        if a in ("-k", "-m", "-p", "-o", "--deselect", "--ignore"):
            skip_next = True   # these take a separate value, not a path
            continue
        if a.startswith("-"):
            continue
        if "::" in a or a.endswith(".py") or os.path.exists(
            os.path.join(REPO_ROOT, a)
        ) or os.path.exists(a):
            return True
    return False


def _run_rust_bridge(crates, out_path, timeout) -> int:
    """Invoke the Rust bridge module via the report package."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "python", "tests"))
    from report import rust_bridge

    sections = rust_bridge.run(crates, timeout=timeout, cwd=REPO_ROOT)
    rust_bridge.write_sections(sections, out_path)
    n_fail = sum(1 for s in sections if s.status in ("failed", "error"))
    print(f"[gen_test_report] rust: {len(sections)} sections "
          f"({n_fail} failed/error) -> {out_path}")
    return 0


def _run_pytest(pytest_args, html, js, merge) -> int:
    import subprocess

    cmd = [
        sys.executable, "-m", "pytest",
        *pytest_args,
        "--report",
        "--report-html", html,
        "--report-json", js,
    ]
    if merge:
        cmd += ["--report-merge", merge]
    print(f"[gen_test_report] pytest: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the unified rapidfem HTML test report "
                    "(LOCAL tool, never CI).",
    )
    parser.add_argument("--no-rust", action="store_true",
                        help="skip the cargo-test Rust bridge")
    parser.add_argument("--rust-crates", nargs="+", default=None,
                        help="restrict the Rust bridge to these crates")
    parser.add_argument("--rust-timeout", type=int, default=None,
                        help="per-crate cargo timeout in seconds")
    parser.add_argument("--html", default=DEFAULT_HTML,
                        help=f"HTML output path (default: {DEFAULT_HTML})")
    parser.add_argument("--json", default=DEFAULT_JSON,
                        help=f"report JSON output path (default: {DEFAULT_JSON})")
    parser.add_argument("--rust-json", default=DEFAULT_RUST_JSON,
                        help="Rust-sections merge JSON path "
                             f"(default: {DEFAULT_RUST_JSON})")
    args, pytest_args = parser.parse_known_args(argv)

    # The report plugin is wired in via python/tests/conftest.py, so pytest
    # must collect that directory or the --report options are unknown. If the
    # forwarded args carry no path (only flags / -k / -m), append the tests
    # dir; filters like -k then apply within it.
    if not _has_path(pytest_args):
        pytest_args = [*pytest_args, TESTS_DIR]

    merge = None
    if not args.no_rust:
        sys.path.insert(0, os.path.join(REPO_ROOT, "python", "tests"))
        from report import rust_bridge

        crates = args.rust_crates or rust_bridge.DEFAULT_CRATES
        timeout = args.rust_timeout or rust_bridge.DEFAULT_TIMEOUT
        _run_rust_bridge(crates, args.rust_json, timeout)
        merge = args.rust_json

    rc = _run_pytest(pytest_args, args.html, args.json, merge)

    html_abs = os.path.normpath(os.path.join(REPO_ROOT, args.html))
    print(f"[gen_test_report] report -> {html_abs}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
