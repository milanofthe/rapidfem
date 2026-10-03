# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""The rapidfem CLI, `rapidfem serve` and `rapidfem demo`.

Entry point registered via pyproject.toml [project.scripts].
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from rapidfem import __version__


def _default_workdir() -> Path:
    """Default workdir: the directory ``rapidfem serve`` was launched from.

    Using the launch cwd means the UI's file browser and the cell-runner
    both operate on the folder the user is standing in, so relative paths
    in a script (e.g. ``rf.Geometry.from_gds("data/foo.gds")``) resolve
    against that same folder. A self-contained project directory just
    works when you run ``rapidfem serve`` inside it. Pass an explicit
    workdir argument to override.
    """
    return Path.cwd()


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from rapidfem.ui.server import create_app, run
    except ImportError as e:
        print(
            "error: rapidfem was installed without the UI extra.\n"
            "       Install with:  pip install 'rapidfem[ui]'\n"
            f"       (import failure: {e})",
            file=sys.stderr,
        )
        return 2

    if args.workdir is None:
        workdir = _default_workdir()
    else:
        workdir = Path(args.workdir).resolve()
        if not workdir.exists():
            print(f"error: workdir does not exist: {workdir}", file=sys.stderr)
            return 2
        if not workdir.is_dir():
            print(f"error: workdir is not a directory: {workdir}", file=sys.stderr)
            return 2

    # Examples are NOT copied into the workdir: they stay browsable read-only
    # in the UI's "Examples" section and open as unsaved buffers, so the
    # workdir is only ever populated by what the user explicitly saves.

    app = create_app(workdir=workdir, debug=args.debug)
    run(app, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def _find_frontend_src() -> Path | None:
    """Locate the SvelteKit source tree shipped with editable installs.

    Returns the absolute path to ``frontend-src/`` if it exists and looks
    like an npm package (`package.json` present). The folder is **not**
    included in published wheels, only the prebuilt ``frontend/dist/`` is,
    so `rapidfem demo` is a dev-only command.
    """
    candidate = Path(__file__).resolve().parent / "frontend-src"
    if (candidate / "package.json").is_file():
        return candidate
    return None


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run the SvelteKit dev server in static-demo mode.

    Mirrors what the GH-Pages build does (`VITE_STATIC_MODE=1`) so the
    landing page + bundled `<fem-viewer>` cards are visible locally for
    UI work, without touching the production `frontend/dist/` bundle
    that `rapidfem serve` ships.
    """
    frontend_src = _find_frontend_src()
    if frontend_src is None:
        print(
            "error: 'rapidfem demo' needs a source checkout (`pip install -e .`).\n"
            "       No frontend-src found next to the installed package.",
            file=sys.stderr,
        )
        return 2

    if not (frontend_src / "node_modules").is_dir():
        print(
            f"rapidfem demo, installing npm dependencies in {frontend_src}",
            file=sys.stderr,
        )
        rc = subprocess.call(["npm", "install"], cwd=frontend_src)
        if rc != 0:
            print("error: npm install failed", file=sys.stderr)
            return rc

    env = {**os.environ, "VITE_STATIC_MODE": "1"}
    cmd = ["npm", "run", "dev", "--", "--port", str(args.port), "--strictPort"]
    if not args.no_browser:
        cmd.append("--open")
    print(
        f"rapidfem demo, Vite dev server (static-demo mode) "
        f"on http://127.0.0.1:{args.port}/  · Ctrl+C to stop",
        file=sys.stderr,
    )
    try:
        return subprocess.call(cmd, cwd=frontend_src, env=env)
    except FileNotFoundError:
        print(
            "error: `npm` not found on PATH. Install Node.js to use `rapidfem demo`.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rapidfem",
        description="rapidfem, frequency- and time-domain EM FEM solver.",
    )
    p.add_argument("--version", action="version", version=f"rapidfem {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    s = sub.add_parser(
        "serve",
        help="launch the local UI (Flask + bundled SvelteKit frontend)",
        description="Start the rapidfem UI on a local web server.",
    )
    s.add_argument(
        "workdir",
        nargs="?",
        default=None,
        help="project working directory (default: current directory)",
    )
    s.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    s.add_argument("--port", type=int, default=5174, help="bind port (default: 5174)")
    s.add_argument("--debug", action="store_true", help="Flask debug mode + hot reload")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    s.set_defaults(func=_cmd_serve)

    d = sub.add_parser(
        "demo",
        help="run the SvelteKit dev server in static-demo mode (dev-only)",
        description=(
            "Spin up the Vite dev server with VITE_STATIC_MODE=1 so the "
            "landing page + <fem-viewer> demo cards are visible locally. "
            "Requires a source checkout and Node.js, only useful for UI "
            "development. Does not touch the bundled `frontend/dist/` that "
            "`rapidfem serve` ships."
        ),
    )
    d.add_argument("--port", type=int, default=5173, help="Vite dev port (default: 5173)")
    d.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    d.set_defaults(func=_cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
