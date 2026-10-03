# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""JSON API endpoints for the rapidfem UI.

Registered onto the Flask app by ``rapidfem.ui.server.create_app``.
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


def _format_exception(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


_WIN = sys.platform == "win32"
if _WIN:
    import ctypes
    import msvcrt
    _STD_OUTPUT_HANDLE = -11
    _STD_ERROR_HANDLE = -12
    _GetStdHandle = ctypes.windll.kernel32.GetStdHandle
    _GetStdHandle.restype = ctypes.c_void_p
    _SetStdHandle = ctypes.windll.kernel32.SetStdHandle
    _SetStdHandle.argtypes = [ctypes.c_int, ctypes.c_void_p]


@contextmanager
def _capture_streams(on_line):
    """OS-level fd capture so Rust eprintln! output reaches the UI.

    ``on_line(kind, text)`` is called per line as soon as the pipe delivers
    it. Used by the notebook
    worker and the demo baker to fold native stdout/stderr into a cell run.
    """
    sys.stdout.flush(); sys.stderr.flush()
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    os.dup2(out_w, 1)
    os.dup2(err_w, 2)

    saved_win_out = saved_win_err = None
    if _WIN:
        saved_win_out = _GetStdHandle(_STD_OUTPUT_HANDLE)
        saved_win_err = _GetStdHandle(_STD_ERROR_HANDLE)
        _SetStdHandle(_STD_OUTPUT_HANDLE, msvcrt.get_osfhandle(1))
        _SetStdHandle(_STD_ERROR_HANDLE, msvcrt.get_osfhandle(2))

    os.close(out_w)
    os.close(err_w)

    lines_out: list[str] = []
    lines_err: list[str] = []

    def reader(fd: int, kind: str, accum: list[str]) -> None:
        buf = b""
        try:
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, _, buf = buf.partition(b"\n")
                    s = raw.rstrip(b"\r").decode("utf-8", errors="replace")
                    if not s:
                        continue
                    accum.append(s)
                    try:
                        on_line(kind, s)
                    except Exception:
                        pass
            if buf:
                tail = buf.decode("utf-8", errors="replace").rstrip()
                if tail:
                    accum.append(tail)
                    try:
                        on_line(kind, tail)
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    t_out = threading.Thread(target=reader, args=(out_r, "stdout", lines_out), daemon=True)
    t_err = threading.Thread(target=reader, args=(err_r, "stderr", lines_err), daemon=True)
    t_out.start()
    t_err.start()
    # sys.stdout/stderr in Python wrap the C fd via a TextIOWrapper with a
    # locale-derived encoding (cp1252 on Windows). User code printing a non-
    # ASCII char (e.g. subscript) would crash on encode. Force UTF-8 for the
    # duration of the cell so prints with Unicode work.
    prior_out_enc = getattr(sys.stdout, "encoding", None)
    prior_err_enc = getattr(sys.stderr, "encoding", None)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", write_through=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", write_through=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        yield lines_out, lines_err
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        try:
            if prior_out_enc:
                sys.stdout.reconfigure(encoding=prior_out_enc)  # type: ignore[attr-defined]
            if prior_err_enc:
                sys.stderr.reconfigure(encoding=prior_err_enc)  # type: ignore[attr-defined]
        except Exception:
            pass
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        if _WIN and saved_win_out is not None and saved_win_err is not None:
            _SetStdHandle(_STD_OUTPUT_HANDLE, saved_win_out)
            _SetStdHandle(_STD_ERROR_HANDLE, saved_win_err)
        os.close(saved_out)
        os.close(saved_err)
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)


# Capture kinds whose display payload is built from a single item, with no
# sim+result pairing, so they can be streamed the instant show() runs.
_STREAMABLE_KINDS = frozenset({
    "geometry", "td_timeseries", "td_transfer", "td_trajectory",
})


def _serialize_streamable(item) -> dict[str, Any] | None:
    """Serialise a single self-contained capture into one display event.

    Handles the kinds in :data:`_STREAMABLE_KINDS` (geometry / mesh preview
    and the time-domain wrappers), which need no cross-item pairing and can
    therefore be emitted mid-cell. Returns the event dict (or an ``error``
    event on failure), or ``None`` for kinds deferred to
    :func:`_serialize_paired`. Never raises.
    """
    from rapidfem.ui.serialize import (
        geometry_to_payload, td_timeseries_payload, td_trajectory_payload)

    if item.kind == "geometry":
        try:
            p = geometry_to_payload(item.obj)
        except Exception as e:  # noqa: BLE001
            return {"kind": "error", "name": item.name, "error": _format_exception(e)}
        kind = "mesh" if p.get("kind") == "mesh" else "geometry"
        return {"kind": kind, "name": item.name, "payload": p}

    if item.kind in ("td_timeseries", "td_transfer", "td_trajectory"):
        # A transfer function reuses the time-series payload builder (it sets
        # domain="freq" itself).
        _td_builder = {
            "td_timeseries": td_timeseries_payload,
            "td_transfer": td_timeseries_payload,
            "td_trajectory": td_trajectory_payload,
        }[item.kind]
        try:
            return {"kind": item.kind, "name": item.name,
                    "payload": _td_builder(item.obj)}
        except Exception as e:  # noqa: BLE001
            return {"kind": "error", "name": item.name, "error": _format_exception(e)}

    return None


def _serialize_paired(captures: list, *, eager_fields: bool = False) -> list[dict[str, Any]]:
    """Serialise the deferred sim+result pairing into mesh + field/result
    displays. Must run after the whole cell, the ``result`` payload needs the
    ``Problem``'s native handle (n_dofs, field_at_nodes). Geometry / td_* are
    handled per-item by :func:`_serialize_streamable`; here we only scan them
    to track ``last_geo`` for the mesh fallback.

    ``eager_fields`` controls how a driven-sweep ``result`` carries its E/J/H
    fields. The live ``rapidfem serve`` path leaves them out (``None``) and the
    viewer pulls the one field it shows from ``/api/field`` on demand. The
    offline bake (``scripts/bake_demo.py``) has no worker to answer that, so it
    passes ``eager_fields=True`` to embed every field inline; ``binpack`` then
    lifts them into ``<name>.field.bin`` for the static web demo.

    ``kind="simulation"`` covers :class:`rapidfem.ProblemFD`; we extract its
    ``.native`` here once, so the rest of the function works with the
    Rust-side accessors directly (``mesh_nodes``, ``field_at_nodes``, ...).
    """
    from rapidfem.ui.serialize import mesh_to_payload
    import numpy as np

    out: list[dict[str, Any]] = []
    last_sim = None
    last_result = None
    last_geo = None
    last_modes = None  # list[Eigenmode] from rapidfem.show(modes_list)

    for item in captures:
        if item.kind == "geometry":
            last_geo = item.obj
        elif item.kind == "simulation":
            # Captured object is a rapidfem.ProblemFD; reach into its native
            # solver for mesh + field accessors. If the user show()ed the
            # Problem before running any analysis, .native raises, surface
            # that as a display-level error rather than crashing the bake.
            try:
                last_sim = item.obj.native
            except (AttributeError, RuntimeError) as e:
                out.append({"kind": "error", "name": item.name,
                            "error": _format_exception(e)})
        elif item.kind == "result":
            last_result = item.obj
        elif item.kind == "eigenmodes":
            last_modes = item.obj
        elif item.kind == "eigenmode":
            # Single mode → wrap in a list so the downstream serialiser
            # always sees a uniform shape.
            last_modes = [item.obj]

    # Pair sim+result into mesh+field displays.
    if last_sim is not None:
        try:
            out.append({"kind": "mesh", "name": "simulation",
                        "payload": mesh_to_payload(last_geo, maxh=0.0)})
        except Exception as e:  # noqa: BLE001
            out.append({"kind": "error", "name": "simulation", "error": _format_exception(e)})

    # Eigenmode result: one "frequency" per mode, no port dimension. We
    # reuse the existing `result` payload shape, frontends already know
    # how to drive the field viewer + frequency slider; for eigenmodes
    # the slider becomes a mode index, and S-params are empty.
    if last_sim is not None and last_modes is not None and last_result is None:
        try:
            import math
            modes = last_modes
            n_mode = len(modes)
            freqs_per_mode = [float(m.frequency_hz) for m in modes]
            # JSON doesn't have an `Infinity` literal, Python's json.dumps
            # writes the non-standard `Infinity` token, which browsers and
            # spec-compliant parsers reject. Lossless modes have Q = inf, so
            # map those to `null`; the frontend's `isFinite()` check renders
            # them as ∞.
            q_factors = [
                float(m.q_factor) if math.isfinite(m.q_factor) else None
                for m in modes
            ]
            # n_driven = 1 (the "port axis" collapses for eigenmodes)
            fields_payload = [[_list(last_sim.mode_field_abc(m))] for m in modes]
            sparams_payload = [[[]] for _ in range(n_mode)]
            out.append({
                "kind": "result", "name": "eigenmodes",
                "payload": {
                    "frequencies": freqs_per_mode,
                    "sparams": sparams_payload,
                    "n_driven": 1, "n_freq": n_mode,
                    "n_dofs": last_sim.n_dofs, "n_tets": last_sim.n_tets,
                    "solve_time_s": 0.0,
                    "fields": fields_payload,
                    "eigenmode": True,
                    "q_factors": q_factors,
                },
            })
        except Exception as e:  # noqa: BLE001
            out.append({"kind": "error", "name": "eigenmodes", "error": _format_exception(e)})

    if last_sim is not None and last_result is not None:
        try:
            s = last_result.sparams
            n_freq, n_p, _ = s.shape
            sparams_payload = np.stack([s.real, s.imag], axis=-1).tolist()
            payload = {
                "frequencies": last_result.frequencies.tolist(),
                "sparams": sparams_payload,
                "n_driven": n_p, "n_freq": n_freq,
                "n_dofs": last_sim.n_dofs, "n_tets": last_sim.n_tets,
                "solve_time_s": last_result.solve_time_s,
            }
            if eager_fields:
                # Offline bake: embed every field so the static demo (no
                # worker) can show them; binpack packs these into field.bin.
                ch = _build_channel_payloads(last_sim, last_result, n_freq, n_p)
                payload["fields"] = ch["E"]
                payload["fields_j"] = ch["J"]
                payload["fields_h"] = ch["H"]
                payload["field_meta"] = None
            else:
                # Live serve: fields are NOT inlined (they were tens of MB of
                # JSON). The viewer fetches the one field it shows on demand
                # via GET /api/field (binary); the worker keeps this result
                # alive to serve those queries.
                payload["fields"] = None
                payload["fields_j"] = None
                payload["fields_h"] = None
                payload["field_meta"] = {
                    "n_freq": n_freq,
                    "n_port": n_p,
                    "channels": _available_field_channels(last_sim, last_result),
                    "lazy": True,
                }
            out.append({"kind": "result", "name": "result", "payload": payload})
        except Exception as e:  # noqa: BLE001
            out.append({"kind": "error", "name": "result", "error": _format_exception(e)})

    return out


def _serialize_captures_for_protocol(
        captures: list, *, eager_fields: bool = False) -> list[dict[str, Any]]:
    """Render a whole batch of captures into display events (`{kind, payload,
    name}`) for the kernel protocol.

    Combines the per-item streamable events (geometry / td_*) with the
    deferred sim+result pairing. Used by callers that serialise a complete
    capture list at once; the streaming worker path instead calls
    :func:`_serialize_streamable` per item (mid-cell) and
    :func:`_serialize_paired` once at cell end.

    ``eager_fields`` is forwarded to :func:`_serialize_paired`; the offline
    bake sets it so driven-sweep fields are embedded for the static demo.
    """
    out: list[dict[str, Any]] = []
    for item in captures:
        evt = _serialize_streamable(item)
        if evt is not None:
            out.append(evt)
    out.extend(_serialize_paired(captures, eager_fields=eager_fields))
    return out


def _available_field_channels(sim, result) -> list[str]:
    """Which of E / J / H the backend can actually produce for this result.

    E is always present for a solved sweep; J (current density) and H come
    back as None for configurations the native solver doesn't derive them
    for. Probe each once at (freq 0, port 0) so the viewer only offers
    channels that will render something instead of blanking on selection.
    """
    chans = ["E"]
    for name in ("J", "H"):
        try:
            if sim.field_abc(result, 0, 0, name) is not None:
                chans.append(name)
        except Exception:  # noqa: BLE001
            pass
    return chans


def _list(abc) -> list[float] | None:
    """A native ABC-phasor buffer as a JSON list (None stays None)."""
    return None if abc is None else abc.tolist()


def _build_channel_payloads(sim, result, n_freq: int, n_p: int) -> dict[str, list]:
    """Per-channel ``[freq][port][flat_abc]`` payloads for E, J, H.

    Each channel is the (A, B, C) phasor encoding of ``field_abc``; the
    frontend's non-lazy field path feeds the flat array straight into the
    splat sampler. Used by the offline bake (``scripts/bake_demo.py``), which
    has no live worker to answer ``/api/field``, so the static web demo needs
    the fields embedded and packed into ``<name>.field.bin``.
    """
    return {ch: [[_list(sim.field_abc(result, fi, pi, ch)) for pi in range(n_p)]
                 for fi in range(n_freq)]
            for ch in ("E", "J", "H")}


def register(app: Flask) -> None:
    workdir: Path = app.config["RAPIDFEM_WORKDIR"]

    # ── File endpoints ────────────────────────────────────────────────────────

    def _safe_path(rel: str) -> Path | None:
        """Resolve `rel` inside workdir; reject path traversal."""
        if not rel or "\x00" in rel:
            return None
        try:
            target = (workdir / rel).resolve()
        except (OSError, ValueError):
            return None
        try:
            target.relative_to(workdir)
        except ValueError:
            return None
        return target

    @app.get("/api/files")
    def api_files_list():
        out: list[dict[str, Any]] = []
        for p in sorted(workdir.rglob("*.py")):
            if any(part.startswith(".") or part in {"__pycache__", "node_modules", "target"} for part in p.relative_to(workdir).parts):
                continue
            try:
                rel = p.relative_to(workdir).as_posix()
                st = p.stat()
            except OSError:
                continue
            out.append({"path": rel, "size": st.st_size, "mtime": st.st_mtime})
        return jsonify({"workdir": str(workdir), "files": out})

    @app.get("/api/files/<path:rel>")
    def api_files_get(rel: str):
        target = _safe_path(rel)
        if target is None or not target.is_file():
            return jsonify({"ok": False, "error": "not found"}), 404
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "path": rel, "content": content})

    # ── Examples (shipped with the package) ───────────────────────────────
    # NB: /api/cell/run and /api/cell/reset moved to rapidfem.ui.runner, the
    # subprocess-based runner exposes them with streaming via /api/cell/poll.

    @app.get("/api/examples")
    def api_examples_list():
        from importlib import resources
        try:
            root = resources.files("rapidfem.examples")
        except (ModuleNotFoundError, FileNotFoundError):
            return jsonify({"examples": []})
        items: list[dict[str, Any]] = []
        for entry in root.iterdir():  # type: ignore[attr-defined]
            if not entry.is_file():
                continue
            name = entry.name
            if not name.endswith(".py") or name.startswith("_"):
                continue
            items.append({"name": name})
        items.sort(key=lambda i: i["name"])
        return jsonify({"examples": items})

    @app.get("/api/examples/<name>")
    def api_examples_get(name: str):
        if not name.endswith(".py") or "/" in name or "\\" in name or ".." in name:
            return jsonify({"ok": False, "error": "invalid"}), 400
        from importlib import resources
        try:
            content = (resources.files("rapidfem.examples") / name).read_text(encoding="utf-8")
        except Exception:
            return jsonify({"ok": False, "error": "not found"}), 404
        return jsonify({"ok": True, "name": name, "content": content})

    @app.put("/api/files/<path:rel>")
    def api_files_put(rel: str):
        target = _safe_path(rel)
        if target is None:
            return jsonify({"ok": False, "error": "invalid path"}), 400
        body = request.get_json(silent=True) or {}
        content = body.get("content", "")
        if not isinstance(content, str):
            return jsonify({"ok": False, "error": "content must be string"}), 400
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "path": rel, "size": target.stat().st_size})

    @app.delete("/api/files/<path:rel>")
    def api_files_delete(rel: str):
        target = _safe_path(rel)
        if target is None or not target.is_file():
            return jsonify({"ok": False, "error": "not found"}), 404
        try:
            target.unlink()
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        # Drop the kernel so a future file at the same path starts fresh.
        from rapidfem.ui import runner
        runner._remove(str(target))
        return jsonify({"ok": True, "path": rel})

    @app.post("/api/files/rename")
    def api_files_rename():
        body = request.get_json(silent=True) or {}
        old_rel = body.get("from", "")
        new_rel = body.get("to", "")
        old = _safe_path(old_rel)
        new = _safe_path(new_rel)
        if old is None or new is None or not old.is_file():
            return jsonify({"ok": False, "error": "invalid path"}), 400
        if new.exists():
            return jsonify({"ok": False, "error": "destination exists"}), 409
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "from": old_rel, "to": new_rel})
