"""rapidfem worker subprocess.

Per-file kernel that runs notebook cells in a clean, isolated Python process.
Communicates with the parent server via JSON-line messages on stdin/stdout —
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

# Save the real stdout *before* we replace sys.stdout — protocol writes go
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

    Installed on ``sys.stdout`` / ``sys.stderr`` permanently — anything the
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


def _reset_namespace() -> None:
    """Wipe the worker's Python namespace and any gmsh model state."""
    global _namespace
    import rapidfem
    _namespace = {
        "__name__": "__rapidfem_kernel__",
        "__builtins__": __builtins__,
        "rapidfem": rapidfem,
    }
    try:
        import gmsh
        if gmsh.isInitialized():
            gmsh.clear()
    except Exception:
        pass


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

    # gmsh installs a SIGINT handler that only works on the main thread;
    # eager init here keeps later Geometry() calls quick.
    try:
        import gmsh
        if not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
    except Exception:
        pass

    # Restore Python's default SIGINT handler — gmsh.initialize() installs
    # one that suppresses the signal, but we want SIGINT to raise
    # KeyboardInterrupt inside the current cell so the parent can interrupt
    # a long-running solve.
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, OSError):
        # `signal.signal` is main-thread only; we're in the main thread but
        # guard for unexpected sandbox restrictions.
        pass

    _initialized = True
    send({"type": "ready"})


# ── Cell execution ──────────────────────────────────────────────────────────

def _emit_displays(captured) -> None:
    """Serialise each rapidfem.show() capture and send as a `display` event.

    Heavy lifting (Geometry → wireframe payload, SweepResult → fields +
    sparams etc.) reuses the existing api._serialize_captures_for_protocol
    helper, which already knows the wire format the frontend expects.
    """
    try:
        # Lazy import — keeps worker startup fast and decouples from server.
        from rapidfem.ui.api import _serialize_captures_for_protocol
    except Exception:
        # Fall back: no rich serialisation, just announce what was captured.
        for item in captured:
            send({
                "type": "display",
                "kind": item.kind,
                "name": item.name,
                "payload": {"kind": item.kind, "repr": repr(item.obj)[:200]},
            })
        return

    try:
        events = _serialize_captures_for_protocol(captured)
    except Exception as e:
        send({
            "type": "error",
            "error": f"display serialisation failed: {e}",
            "traceback": traceback.format_exc(),
        })
        return

    for evt in events:
        send({"type": "display", **evt})


def run_cell(msg_id: str, code: str) -> None:
    """Exec a cell in the persistent namespace, emit displays + done/error."""
    if not _initialized:
        send({"type": "error", "id": msg_id, "error": "Worker not initialized"})
        return

    from rapidfem import _show_capture
    _show_capture.start_capture()
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

    _emit_displays(captured)
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
            else:
                send({"type": "error", "error": f"Unknown message type: {t!r}"})
        except KeyboardInterrupt:
            # SIGINT outside the cell-run exec — nothing to stop. Ignore and
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
