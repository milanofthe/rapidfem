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
def _capture_streams(on_line, stage: str = "cell"):
    """OS-level fd capture so Rust eprintln! output reaches the UI.

    ``on_line(kind, text)`` is called per line as soon as the pipe delivers
    it. ``stage`` is accepted for call-site clarity. Used by the notebook
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


def _td_timeseries_payload(obj) -> dict[str, Any]:
    """``TdResponse`` / ``TdTransfer`` → a line-plot payload.

    A response carries real probe samples on a time axis; a transfer
    function carries a complex ``H`` on a frequency axis. ``domain``
    tells the frontend which it is.
    """
    import numpy as np

    cls = type(obj).__name__
    if cls == "TdResponse":
        x = np.asarray(obj.times, dtype=float).ravel()
        resp = np.asarray(obj.responses, dtype=float)
        labels = list(obj.probe_labels) or [
            f"probe {k}" for k in range(resp.shape[0])
        ]
        series = [
            {"label": labels[k], "y": resp[k].astype(float).tolist()}
            for k in range(resp.shape[0])
        ]
        return {
            "domain": "time",
            "x_label": "Time",
            "x": x.tolist(),
            "series": series,
            "source_label": obj.source_label,
        }
    # TdTransfer, complex frequency response
    x = np.asarray(obj.frequencies, dtype=float).ravel()
    H = np.asarray(obj.H)
    return {
        "domain": "freq",
        "x_label": "Frequency (Hz)",
        "x": x.tolist(),
        "series": [{
            "label": f"H · {obj.probe_label}",
            "y_re": np.real(H).astype(float).tolist(),
            "y_im": np.imag(H).astype(float).tolist(),
        }],
        "source_label": obj.source_label,
    }


def _td_trajectory_payload(
    traj, *, max_frames: int = 180,
) -> dict[str, Any]:
    """``TdTrajectory`` → a self-contained DG-corner mesh + per-node field.

    Emits the DG element corners deduplicated into unique nodes plus the
    tet connectivity, and, per kept frame, per unique node, the ``|E|``
    and ``|H|`` field magnitude. The frontend samples a point cloud from
    that mesh *at runtime* (energy-weighted, like the frequency-domain
    ``viz.ts`` sampler) at whatever density the user picks, then evaluates
    the per-frame field at the fixed sample points. This mirrors the FD
    field viz and lets the baked payload stay small (a few thousand nodes,
    not a fixed 16k-point cloud).

    Per-frame magnitudes are quantised to integers ``0…1000`` of the
    global per-channel maximum (``field_max``), ample for an additive
    colour ramp and an order of magnitude smaller on the wire than raw
    floats. The viewer rescales by ``field_max`` and holds that colour
    scale fixed across the animation. Snapshots are decimated to at most
    ``max_frames``.
    """
    import numpy as np

    p = getattr(traj, "_problem", None)
    if p is None:
        raise RuntimeError(
            "trajectory carries no ProblemTD reference, it must come "
            "straight from ProblemTD.transient()"
        )
    states = np.ascontiguousarray(traj, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    n_snap, n_dof = states.shape

    o = int(p.order)
    np_ = (o + 1) * (o + 2) * (o + 3) // 6
    n_elem = n_dof // (6 * np_)
    corners = np.asarray(p._op.corner_local_nodes(), dtype=np.int64)
    coords = np.asarray(p._op.node_coords(), dtype=float).reshape(n_elem, np_, 3)
    corner_xyz = coords[:, corners, :]                     # [n_elem, 4, 3]

    # Decimate snapshots to a bounded frame count.
    if n_snap > max_frames:
        idx = np.unique(np.linspace(0, n_snap - 1, max_frames).round()
                        .astype(int))
    else:
        idx = np.arange(n_snap)
    # Corner field of every kept frame, [n_frame, n_elem, 4, 6].
    cstates = states[idx].reshape(len(idx), n_elem, np_, 6)[:, :, corners, :]

    # ── Deduplicate the n_elem·4 corner coordinates into unique nodes ───
    # DG elements duplicate every shared corner; rounding to a tolerance
    # then np.unique by row collapses them so the runtime sampler sees a
    # continuous mesh. `inverse` maps each (elem, corner) flat index to a
    # unique-node index → the tet connectivity.
    flat_xyz = corner_xyz.reshape(-1, 3)                   # [n_elem*4, 3]
    span = float(np.ptp(flat_xyz)) or 1.0
    quant = np.round(flat_xyz / (span * 1e-7)).astype(np.int64)
    _, first, inverse = np.unique(
        quant, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).ravel()
    nodes = flat_xyz[first]                                # [n_node, 3]
    n_node = int(nodes.shape[0])
    tets = inverse.reshape(n_elem, 4).astype(np.int32)     # [n_elem, 4]

    # ── Per-node field magnitude per kept frame ─────────────────────────
    # |E|/|H| at every (elem, corner) corner, then averaged over all
    # corners that map to the same unique node. counts[node] is the number
    # of contributing (elem, corner) pairs.
    evec = cstates[..., 0:3]                               # [n_frame,n_elem,4,3]
    hvec = cstates[..., 3:6]
    em = np.linalg.norm(evec, axis=-1).reshape(len(idx), -1)  # [n_frame,n_elem*4]
    hm = np.linalg.norm(hvec, axis=-1).reshape(len(idx), -1)
    counts = np.bincount(inverse, minlength=n_node).astype(float)
    counts[counts == 0] = 1.0
    node_e = np.zeros((len(idx), n_node))
    node_h = np.zeros((len(idx), n_node))
    for fi in range(len(idx)):
        node_e[fi] = np.bincount(inverse, weights=em[fi], minlength=n_node) / counts
        node_h[fi] = np.bincount(inverse, weights=hm[fi], minlength=n_node) / counts

    e_max = max(float(node_e.max()), 1e-30)
    h_max = max(float(node_h.max()), 1e-30)
    qe = np.clip(np.round(node_e / e_max * 1000.0), 0, 1000).astype(np.int16)
    qh = np.clip(np.round(node_h / h_max * 1000.0), 0, 1000).astype(np.int16)

    dt = getattr(traj, "_dt", None)
    times = (np.asarray(idx, dtype=float) * dt).tolist() if dt else \
        np.asarray(idx, dtype=float).tolist()

    return {
        "nodes": nodes.astype(np.float32).ravel().tolist(),
        "tets": tets.ravel().tolist(),
        "n_node": n_node,
        "n_elem": int(n_elem),
        "bbox": _bbox_for_nodes(nodes),
        "n_snapshots": len(idx),
        "times": times,
        "field_max": {"E": e_max, "H": h_max},
        "frames_e": [row.tolist() for row in qe],
        "frames_h": [row.tolist() for row in qh],
    }


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
    from rapidfem.ui.serialize import geometry_to_payload

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
            "td_timeseries": _td_timeseries_payload,
            "td_transfer": _td_timeseries_payload,
            "td_trajectory": _td_trajectory_payload,
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

    ``kind="simulation"`` covers :class:`rapidfem.Problem`; we extract its
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
            # Captured object is a rapidfem.Problem; reach into its native
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
            mesh_payload = mesh_to_payload(last_geo, maxh=0.0)
        except Exception:
            try:
                nodes_np = np.asarray(last_sim.mesh_nodes)
                mesh_payload = {
                    "kind": "mesh",
                    "nodes": nodes_np.ravel().tolist(),
                    "tets": np.asarray(last_sim.mesh_tets).ravel().tolist(),
                    "tris": [], "tri_phys": [], "tet_phys": [1] * int(last_sim.mesh_tets.shape[0]),
                    "phys_names": {"1": "mesh"}, "phys_dim": {"1": 3}, "name_to_tag": {"mesh": 1},
                    "bbox": _bbox_for_nodes(nodes_np),
                    "stats": {"n_nodes": int(nodes_np.shape[0]),
                              "n_tets": int(last_sim.mesh_tets.shape[0]),
                              "n_tris": 0, "mesh_time_s": 0.0, "msh_bytes": 0},
                }
            except Exception:
                mesh_payload = None
        if mesh_payload is not None:
            out.append({"kind": "mesh", "name": "simulation", "payload": mesh_payload})

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
            fields_payload = []
            for m in modes:
                # n_driven = 1 (the "port axis" collapses for eigenmodes)
                E = last_sim.mode_field_at_nodes(m)
                if E is None:
                    fields_payload.append([None])
                    continue
                re = np.asarray(E.real); im = np.asarray(E.imag)
                A = np.sum(re * re, axis=1)
                B = np.sum(im * im, axis=1)
                C = np.sum(re * im, axis=1)
                bin_vals = np.stack([A, B, C], axis=1).astype(np.float32).ravel().tolist()
                fields_payload.append([bin_vals])
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
            sparams_payload = []
            for fi in range(n_freq):
                f_mat = []
                for r in range(n_p):
                    row = []
                    for c in range(n_p):
                        v = s[fi, r, c]
                        row.append([float(v.real), float(v.imag)])
                    f_mat.append(row)
                sparams_payload.append(f_mat)
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


def _bbox_for_nodes(nodes_np) -> dict[str, list[float]]:
    """Bounding box from a (n_nodes, 3) array."""
    if nodes_np is None or len(nodes_np) == 0:
        return {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    mn = nodes_np.min(axis=0).tolist()
    mx = nodes_np.max(axis=0).tolist()
    return {"min": mn, "max": mx}


def _available_field_channels(sim, result) -> list[str]:
    """Which of E / J / H the backend can actually produce for this result.

    E is always present for a solved sweep; J (current density) and H come
    back as None for configurations the native solver doesn't derive them
    for. Probe each once at (freq 0, port 0) so the viewer only offers
    channels that will render something instead of blanking on selection.
    """
    chans = ["E"]
    for name, getter in (("J", sim.current_density_at_nodes),
                          ("H", sim.h_field_at_nodes)):
        try:
            if getter(result, 0, 0) is not None:
                chans.append(name)
        except Exception:  # noqa: BLE001
            pass
    return chans


def _abc_phasor(vec_complex) -> list[float] | None:
    """(n_nodes, 3) complex → flat [A, B, C, ...] per node.

    Encodes |E(t)|² = A·cos²(ωt) + B·sin²(ωt) − 2·C·sin·cos with
    A = |Re|², B = |Im|², C = Re·Im, the same animation-friendly form the
    splat shader composites against a phase uniform. Returns `None` if the
    backend produced no field for this (freq, port).
    """
    if vec_complex is None:
        return None
    import numpy as np
    re = np.asarray(vec_complex.real)
    im = np.asarray(vec_complex.imag)
    A = np.sum(re * re, axis=1)
    B = np.sum(im * im, axis=1)
    C = np.sum(re * im, axis=1)
    return np.stack([A, B, C], axis=1).astype(np.float32).ravel().tolist()


def _build_channel_payloads(sim, result, n_freq: int, n_p: int) -> dict[str, list]:
    """Build per-channel ``[freq][port][flat_abc]`` payloads for E, J, H.

    Each channel uses the same (A, B, C) phasor encoding; the frontend's
    non-lazy field path feeds the flat array straight into the splat sampler.
    Used by the offline bake (``scripts/bake_demo.py``), which has no live
    worker to answer ``/api/field`` on demand, so the static web demo needs
    the fields embedded eagerly and packed into ``<name>.field.bin``. The
    live ``rapidfem serve`` path keeps them lazy instead.
    """
    out: dict[str, list] = {"E": [], "J": [], "H": []}
    for fi in range(n_freq):
        e_freq: list = []
        j_freq: list = []
        h_freq: list = []
        for pi in range(n_p):
            e_freq.append(_abc_phasor(sim.field_at_nodes(result, fi, pi)))
            j_freq.append(_abc_phasor(sim.current_density_at_nodes(result, fi, pi)))
            h_freq.append(_abc_phasor(sim.h_field_at_nodes(result, fi, pi)))
        out["E"].append(e_freq)
        out["J"].append(j_freq)
        out["H"].append(h_freq)
    return out


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
        # Tolerate either the new runner module or absence (legacy single
        # in-process kernel was removed).
        try:
            from rapidfem.ui import runner
            runner._remove(str(target))
        except Exception:
            pass
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
