# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Flask server for `rapidfem serve`.

Exposes the bundled SvelteKit frontend on `/` and a small JSON API on
`/api/*`. All endpoints are local-only by convention, there is no
authentication.

Cell execution runs in a per-file subprocess worker (see
``rapidfem.ui.runner``); the server itself never exec's user code.
Streaming events flow via HTTP long-polling on ``/api/cell/poll``.

The actual endpoints are registered in:
  - ``rapidfem.ui.runner``  → /api/cell/*, /api/kernel, /api/field
  - ``rapidfem.ui.api``     → /api/files/*, /api/examples/*
  - this module             → /api/health and the static frontend
"""
from __future__ import annotations

import atexit
import os
import threading
import webbrowser
from importlib import resources
from pathlib import Path

from flask import Flask, jsonify, send_from_directory


_FRONTEND_PACKAGE = "rapidfem.ui.frontend"


def _frontend_dist_path() -> Path | None:
    """Locate the bundled frontend's dist/ directory.

    Returns None when the wheel was built without a frontend bundle (e.g.
    a from-source install that did not run `scripts/build_frontend`).
    Callers should surface a helpful message in that case.
    """
    try:
        root = resources.files(_FRONTEND_PACKAGE)
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    dist = root / "dist"
    try:
        if dist.is_dir():  # type: ignore[union-attr]
            return Path(str(dist))
    except (AttributeError, NotADirectoryError):
        pass
    return None


def create_app(workdir: Path, debug: bool = False) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["RAPIDFEM_WORKDIR"] = Path(workdir).resolve()
    app.config["RAPIDFEM_DEBUG"] = debug

    dist = _frontend_dist_path()
    app.config["RAPIDFEM_FRONTEND_DIST"] = dist

    # CORS for the localhost dev case where the SvelteKit dev server (vite,
    # port 5173) talks to this Flask process (port 5174). Strict-localhost
    # check rather than wildcard.
    @app.after_request
    def _cors(resp):
        from flask import request
        req_origin = request.headers.get("Origin", "")
        if req_origin.startswith("http://127.0.0.1") or req_origin.startswith("http://localhost"):
            resp.headers["Access-Control-Allow-Origin"] = req_origin
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.get("/api/health")
    def health():
        return jsonify({
            "ok": True,
            "workdir": str(app.config["RAPIDFEM_WORKDIR"]),
            "frontend_bundled": dist is not None,
        })

    # Cell runner (subprocess pool + /api/cell/* + /api/kernel).
    from rapidfem.ui import runner
    runner.register(app)
    atexit.register(runner.shutdown_all)

    # File, example and field endpoints.
    try:
        from rapidfem.ui import api  # noqa: F401
        api.register(app)
    except ImportError:
        pass

    # Frontend static serving, only registered when dist/ is present.
    if dist is not None:
        @app.get("/", defaults={"path": ""})
        @app.get("/<path:path>")
        def _spa(path: str):
            # /api/* never falls through to the static handler.
            if path.startswith(("api/",)) or path == "api":
                from flask import abort
                abort(404)
            target = dist / path
            if path and target.exists() and target.is_file():
                return send_from_directory(dist, path)
            # `/` is the prerendered landing, its asset URLs are relative
            # (`./_app/...`), correct only at the root path.
            if not path:
                return send_from_directory(dist, "index.html")
            # Every other client-side route (notebook, embed/test) uses
            # `404.html` as the SPA fallback. adapter-static produces it
            # with **absolute** asset URLs (`/_app/...`), so it hydrates
            # correctly from any URL depth. Without this, deep routes got
            # `index.html` whose relative paths rebased to e.g.
            # `/notebook/_app/...` and Flask re-served HTML for every JS
            # import, silently breaking the whole SvelteKit boot.
            return send_from_directory(dist, "404.html")
    else:
        @app.get("/")
        def _no_frontend():
            return (
                "<h1>rapidfem ui</h1>"
                "<p>Frontend bundle not found. Build it with "
                "<code>scripts/build_frontend.{ps1,sh}</code> and reinstall, "
                "or install a release wheel.</p>"
            ), 503

    return app


def run(app: Flask, host: str = "127.0.0.1", port: int = 5174, open_browser: bool = True) -> None:
    url = f"http://{host}:{port}/"
    print(f"rapidfem serve, workdir: {app.config['RAPIDFEM_WORKDIR']}")
    print(f"rapidfem serve, listening on {url}")
    if app.config["RAPIDFEM_FRONTEND_DIST"] is None:
        print("rapidfem serve, WARNING: no frontend bundle found (run scripts/build_frontend).")

    if open_browser and not os.environ.get("RAPIDFEM_NO_BROWSER"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    # threaded=True so the long-poll endpoint doesn't block other requests
    # while it waits on the worker's stdout queue.
    app.run(
        host=host, port=port,
        debug=app.config["RAPIDFEM_DEBUG"],
        use_reloader=False,
        threaded=True,
    )
