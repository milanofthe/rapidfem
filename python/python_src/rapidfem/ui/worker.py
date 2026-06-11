"""rapidfem worker subprocess.

Per-file kernel that runs notebook cells in a clean, isolated Python process.
Communicates with the parent server via JSON-line messages on stdin/stdout,
no fd dup2, no WebSocket framing, no GIL gymnastics in the server thread.

Protocol (one JSON object per line):

Server → Worker:
    {"type": "init"}
    {"type": "cell-run", "id": str, "code": str}
    {"type": "reset"}

Worker → Server:
    {"type": "ready"}                                                    # after init
    {"type": "stream", "stream": "stdout"|"stderr", "value": str}       # captured prints / native log
    {"type": "display", "kind": str, "name": str, "payload": dict}      # rapidfem.show() outputs
    {"type": "done", "id": str, "ok": bool}                              # cell finished
    {"type": "error", "id": str, "error": str, "traceback": str}        # cell raised
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import traceback
from typing import Any


# ── Protocol I/O ────────────────────────────────────────────────────────────

# Save the real stdout *before* we replace sys.stdout, protocol writes go
# to the real handle, user prints get rerouted via _ProtocolWriter.
_real_stdout = sys.stdout
_stdout_lock = threading.Lock()


def send(msg: dict[str, Any]) -> None:
    """Write one JSON line to the parent. Thread-safe."""
    with _stdout_lock:
        _real_stdout.write(json.dumps(msg, default=str) + "\n")
        _real_stdout.flush()


def read_message() -> dict | None:
    """Read one JSON line from stdin. Returns None on EOF."""
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # Native libs may leak non-JSON to stdout before we replace it;
            # skip those lines silently.
            continue


class _ProtocolWriter:
    """File-like wrapper that ships writes as ``{"type":"stream"}`` events.

    Installed on ``sys.stdout`` / ``sys.stderr`` permanently, anything the
    user prints, plus output from any logging handler that captured these
    streams at module load, flows through ``send()`` to the parent.
    """

    def __init__(self, stream_name: str):
        self.stream_name = stream_name
        self._in_write = False

    def write(self, text: str) -> int:
        # Re-entry guard: if ``send`` itself prints (it shouldn't, but logging
        # configs sometimes do), don't recurse forever.
        if text and not self._in_write:
            self._in_write = True
            try:
                send({"type": "stream", "stream": self.stream_name, "value": text})
            except Exception:
                pass
            finally:
                self._in_write = False
        return len(text) if text else 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


# ── Namespace + capture ─────────────────────────────────────────────────────

_namespace: dict[str, Any] = {}
_initialized = False

# Last driven-sweep solver handle + result, kept alive so the UI can fetch
# field data on demand (binary, per viewed freq/port/channel) via `field-query`
# without re-solving and without inlining megabytes of field arrays in the
# result payload. Populated after each cell that show()s a Problem + result.
_field_store: dict[str, Any] = {"sim": None, "result": None}


def _reset_namespace() -> None:
    """Wipe the worker's Python namespace."""
    global _namespace
    import rapidfem
    _namespace = {
        "__name__": "__rapidfem_kernel__",
        "__builtins__": __builtins__,
        "rapidfem": rapidfem,
    }
    _field_store["sim"] = None
    _field_store["result"] = None


def initialize() -> None:
    """Bootstrap the worker: replace stdio, import rapidfem, send ready."""
    global _initialized
    if _initialized:
        send({"type": "ready"})
        return

    # Replace stdio FIRST so any import-time prints from rapidfem (e.g. MKL
    # detection logs) also route through the protocol.
    sys.stdout = _ProtocolWriter("stdout")
    sys.stderr = _ProtocolWriter("stderr")

    try:
        import rapidfem  # noqa: F401
    except Exception as e:
        send({
            "type": "error",
            "error": f"Failed to import rapidfem: {e}",
            "traceback": traceback.format_exc(),
        })
        return

    _reset_namespace()

    # Make sure SIGINT raises KeyboardInterrupt inside the current cell so
    # the parent can interrupt a long-running solve.
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, OSError):
        # `signal.signal` is main-thread only; we're in the main thread but
        # guard for unexpected sandbox restrictions.
        pass

    _initialized = True
    send({"type": "ready"})


# ── Cell execution ──────────────────────────────────────────────────────────

def _repr_display(item) -> None:
    """Last-resort fallback: announce a capture by repr when rich
    serialisation is unavailable."""
    send({
        "type": "display",
        "kind": item.kind,
        "name": item.name,
        "payload": {"kind": item.kind, "repr": repr(item.obj)[:200]},
    })


def _stream_display(item) -> None:
    """Capture callback: serialise and send a self-contained display the
    instant ``rapidfem.show()`` runs (geometry / mesh preview, time-domain
    results). Kinds that need sim+result pairing return None here and are
    emitted at cell end by :func:`_emit_paired_displays`.

    Must never raise, it runs inside the user's ``show()`` call; failures are
    reported as display-level error events instead of crashing the cell.
    """
    try:
        from rapidfem.ui.api import _serialize_streamable
        evt = _serialize_streamable(item)
    except ImportError:
        # No rich serialisation available, announce by repr so the user at
        # least sees that something was shown.
        _repr_display(item)
        return
    except Exception as e:  # noqa: BLE001
        send({"type": "display", "kind": "error", "name": item.name,
              "error": f"display serialisation failed: {e}"})
        return
    if evt is not None:
        send({"type": "display", **evt})


def _emit_paired_displays(captured) -> None:
    """Emit the deferred sim+result displays (mesh + S-params / fields) after
    the cell finishes. The streamable kinds were already sent live via
    :func:`_stream_display`, so only the paired ones are produced here.
    """
    try:
        from rapidfem.ui.api import _serialize_paired
    except ImportError:
        # Fall back: announce the un-streamed (sim/result/eigenmode) captures.
        for item in captured:
            from rapidfem.ui.api import _STREAMABLE_KINDS  # may also fail below
            if item.kind not in _STREAMABLE_KINDS:
                _repr_display(item)
        return

    try:
        events = _serialize_paired(captured)
    except Exception as e:
        send({
            "type": "error",
            "error": f"display serialisation failed: {e}",
            "traceback": traceback.format_exc(),
        })
        return

    for evt in events:
        send({"type": "display", **evt})


def _make_sweep_progress():
    """Build a per-frequency sweep callback that streams one lightweight
    ``sweep_point`` per frequency: just that frequency's S-matrix. The frontend
    appends it (no cumulative payload, no field push). Self-contained: never
    raises into the solver.
    """
    def cb(fi, freq, s):
        try:
            n = int(s.shape[0])
            mat = [[[float(s[r, c].real), float(s[r, c].imag)] for c in range(n)]
                   for r in range(n)]
            send({"type": "display", "kind": "sweep_point", "name": "result",
                  "payload": {"freq_idx": int(fi), "freq": float(freq), "s": mat}})
        except Exception:
            pass  # streaming is best-effort; never disturb the solve

    return cb


def _stash_field_sources(captured) -> None:
    """Remember the Problem's native solver + SweepResult from a cell's
    show() captures so `field-query` can interpolate fields on demand. Only
    updates when both are present (a driven sweep was shown), so a later
    geometry-only cell does not wipe a still-viewable result."""
    sim = None
    result = None
    for item in captured:
        if item.kind == "simulation":
            try:
                sim = item.obj.native
            except Exception:
                sim = None
        elif item.kind == "result":
            result = item.obj
    if sim is not None and result is not None:
        _field_store["sim"] = sim
        _field_store["result"] = result


def handle_field_query(msg: dict) -> None:
    """Compute one (freq, port, channel) field from the stashed result and
    reply with the ABC-phasor buffer as base64. The backend decodes it and
    serves raw binary to the viewer (only the field actually being shown is
    ever transferred)."""
    qid = msg.get("qid")
    sim = _field_store.get("sim")
    result = _field_store.get("result")
    if sim is None or result is None:
        send({"type": "field-result", "qid": qid, "ok": False,
              "error": "no field data (run a sweep first)"})
        return
    try:
        import base64
        import numpy as np
        fi = int(msg.get("freq", 0))
        pi = int(msg.get("port", 0))
        channel = str(msg.get("channel", "E"))
        if channel in ("J", "j"):
            arr = sim.current_density_at_nodes(result, fi, pi)
        elif channel in ("H", "h"):
            arr = sim.h_field_at_nodes(result, fi, pi)
        else:
            arr = sim.field_at_nodes(result, fi, pi)
        if arr is None:
            send({"type": "field-result", "qid": qid, "ok": True, "data": ""})
            return
        # ABC phasor (A=Σre², B=Σim², C=Σre·im per node), packed f32 → base64.
        # The backend serves the decoded buffer raw; the viewer reads its
        # byteLength, so no element count needs to ride along.
        re = np.asarray(arr.real)
        im = np.asarray(arr.imag)
        a = np.sum(re * re, axis=1)
        b = np.sum(im * im, axis=1)
        c = np.sum(re * im, axis=1)
        buf = np.stack([a, b, c], axis=1).astype(np.float32).tobytes()
        send({"type": "field-result", "qid": qid, "ok": True,
              "data": base64.b64encode(buf).decode("ascii")})
    except Exception as e:  # noqa: BLE001
        send({"type": "field-result", "qid": qid, "ok": False, "error": str(e)})


def run_cell(msg_id: str, code: str) -> None:
    """Exec a cell in the persistent namespace, emit displays + done/error.

    Self-contained displays (geometry / mesh / time-domain) stream live as
    each ``show()`` runs (via the ``_stream_display`` capture callback) so the
    frontend unlocks their tabs the instant the first data lands; partial
    S-parameters stream per frequency during a sweep (``_make_sweep_progress``);
    the sim+result pairing (with fields) is emitted once at cell end.
    """
    if not _initialized:
        send({"type": "error", "id": msg_id, "error": "Worker not initialized"})
        return

    from rapidfem import _show_capture
    _show_capture.start_capture(on_item=_stream_display,
                                sweep_cb=_make_sweep_progress())
    try:
        try:
            exec(compile(code, "<cell>", "exec"), _namespace)
        except BaseException as e:  # SystemExit and KeyboardInterrupt too
            send({
                "type": "error",
                "id": msg_id,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })
            return
    finally:
        captured = _show_capture.stop_capture()

    _emit_paired_displays(captured)
    _stash_field_sources(captured)
    send({"type": "done", "id": msg_id, "ok": True})


# ── Main loop ───────────────────────────────────────────────────────────────

def main() -> None:
    while True:
        try:
            msg = read_message()
            if msg is None:
                return  # parent closed stdin → exit
            t = msg.get("type")
            if t == "init":
                initialize()
            elif t == "cell-run":
                run_cell(msg.get("id", ""), msg.get("code", ""))
            elif t == "reset":
                _reset_namespace()
                send({"type": "reset-ack"})
            elif t == "field-query":
                handle_field_query(msg)
            else:
                send({"type": "error", "error": f"Unknown message type: {t!r}"})
        except KeyboardInterrupt:
            # SIGINT outside the cell-run exec, nothing to stop. Ignore and
            # carry on; inside cell-run the exception is caught by run_cell.
            continue
        except Exception as e:
            send({
                "type": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    main()
