"""Cell-runner backend for rapidfem serve — one worker subprocess per file.

Replaces the old in-process kernel + WebSocket protocol (which suffered
from os.dup2 / Werkzeug-WS / wsproto-deflate / sendall races). Each open
notebook file gets a long-lived worker subprocess that owns its own
Python namespace and gmsh state. The Flask server brokers JSON messages
in and queue-buffered events out, exposed via plain HTTP endpoints.

HTTP API:

    POST /api/cell/run    {"file": str, "code": str, "reset": bool}
                          -> {"cell_id": str, "ok": true}
                          Kicks off a cell. Returns immediately.

    POST /api/cell/poll   {"file": str}
                          -> {"messages": [event, ...], "done": bool}
                          Long-polls (~100 ms) for stream/display/done/error.

    POST /api/cell/reset  {"file": str}
                          -> {"ok": true}
                          Wipe namespace + gmsh state (sync).

    DELETE /api/kernel    {"file": str}
                          -> {"ok": true}
                          Kill the worker subprocess (recreated lazily).
"""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


# Per-cell stream poll long-poll timeout. 100 ms keeps the UI responsive
# without burning CPU on empty polls.
POLL_TIMEOUT_S = 0.1

# Worker init must complete within this — covers rapidfem import + gmsh init.
INIT_TIMEOUT_S = 30.0


def _worker_script() -> str:
    return str(Path(__file__).parent / "worker.py")


class Session:
    """One worker subprocess + its event queue + the file path it serves."""

    def __init__(self, file_key: str):
        self.file_key = file_key
        self.lock = threading.Lock()
        # `cell_run` blocks on this so two concurrent runs on the same file
        # don't interleave their messages in the queue.
        self.run_lock = threading.Lock()
        self.last_active = time.time()

        self.process = subprocess.Popen(
            [sys.executable, "-u", _worker_script()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        self._queue: queue.Queue[dict] = queue.Queue()
        self._reader_alive = True
        self._reader_thread = threading.Thread(
            target=self._read_stdout, daemon=True,
        )
        self._reader_thread.start()
        # A separate thread to drain stderr — anything the worker writes there
        # is library noise (gmsh, native panics); surface it as a stream event.
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True,
        )
        self._stderr_thread.start()

        self._initialized = False

    # ── I/O ───────────────────────────────────────────────────────────────

    def _read_stdout(self) -> None:
        """Worker stdout → JSON messages → per-session queue."""
        try:
            while self._reader_alive:
                line = self.process.stdout.readline()
                if not line:
                    self._queue.put({"type": "worker-exit"})
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    self._queue.put(json.loads(line))
                except json.JSONDecodeError:
                    # Native libs occasionally bypass _ProtocolWriter and write
                    # raw bytes to fd 1 — wrap them as stream events so the
                    # user still sees the output.
                    self._queue.put({
                        "type": "stream", "stream": "stdout", "value": line + "\n",
                    })
        except Exception:
            self._queue.put({"type": "worker-exit"})

    def _read_stderr(self) -> None:
        """Forward worker stderr (native panics, gmsh log) as stream events."""
        try:
            while self._reader_alive:
                line = self.process.stderr.readline()
                if not line:
                    return
                self._queue.put({
                    "type": "stream", "stream": "stderr", "value": line,
                })
        except Exception:
            pass

    def send(self, msg: dict) -> None:
        """Write one JSON message to the worker's stdin."""
        self.last_active = time.time()
        data = json.dumps(msg, default=str) + "\n"
        try:
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self._queue.put({"type": "worker-exit"})

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def ensure_initialized(self) -> None:
        """Send `init` and block until `ready` (or error/timeout)."""
        if self._initialized:
            return
        self.send({"type": "init"})
        deadline = time.time() + INIT_TIMEOUT_S
        while time.time() < deadline:
            try:
                msg = self._queue.get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                continue
            t = msg.get("type")
            if t == "ready":
                self._initialized = True
                return
            if t == "error":
                raise RuntimeError(
                    f"Worker init failed: {msg.get('error')}\n"
                    f"{msg.get('traceback', '')}"
                )
            if t == "worker-exit":
                raise RuntimeError("Worker process died during init")
            # stdout/stream messages during init — surface them but keep waiting
        raise TimeoutError(f"Worker init timed out after {INIT_TIMEOUT_S:.0f}s")

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def kill(self) -> None:
        self._reader_alive = False
        try:
            self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.kill()
            self.process.wait(timeout=5)
        except Exception:
            pass

    def interrupt(self) -> bool:
        """Raise `KeyboardInterrupt` inside the worker's running cell.

        Sends SIGINT to the subprocess — the worker's main-thread default
        handler converts that into a `KeyboardInterrupt`, which propagates
        out of `exec()` and is caught by the cell-run error path, emitting
        an `error` event to the client.

        Returns False on Windows (no clean equivalent for non-console
        children) or if the process is already dead.
        """
        if not self.is_alive():
            return False
        # Windows: subprocess.CREATE_NEW_PROCESS_GROUP + CTRL_BREAK_EVENT is the
        # usual workaround but our worker isn't a console process — skip for
        # now and let the caller fall back to `kill()` if it really needs to
        # stop a runaway cell.
        if os.name == "nt":
            return False
        try:
            self.process.send_signal(signal.SIGINT)
            return True
        except (ProcessLookupError, OSError):
            return False

    # ── Event queue ───────────────────────────────────────────────────────

    def poll(self, timeout: float) -> list[dict]:
        """Drain pending events. Long-polls up to ``timeout`` if empty."""
        messages: list[dict] = []
        if self._queue.empty() and timeout > 0:
            try:
                messages.append(self._queue.get(timeout=timeout))
            except queue.Empty:
                return messages
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return messages


# ── Global session table ────────────────────────────────────────────────────

_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def _get_or_create(file_key: str) -> Session:
    with _sessions_lock:
        s = _sessions.get(file_key)
        if s and not s.is_alive():
            _sessions.pop(file_key, None)
            s = None
        if s is None:
            s = Session(file_key)
            _sessions[file_key] = s
        return s


def _remove(file_key: str) -> None:
    with _sessions_lock:
        s = _sessions.pop(file_key, None)
    if s:
        s.kill()


def shutdown_all() -> None:
    """Kill every worker — used by atexit so subprocesses don't linger."""
    with _sessions_lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for s in sessions:
        s.kill()


# ── Flask endpoints ─────────────────────────────────────────────────────────

def register(app: Flask) -> None:
    """Register /api/cell/* + /api/kernel endpoints on ``app``."""

    @app.post("/api/cell/run")
    def api_cell_run():
        body = request.get_json(silent=True) or {}
        file_key = body.get("file", "<unnamed>")
        code = body.get("code", "")
        reset_first = bool(body.get("reset", False))
        if not isinstance(code, str):
            return jsonify({"ok": False, "error": "code must be a string"}), 400

        cell_id = body.get("cell_id") or uuid.uuid4().hex[:12]

        try:
            session = _get_or_create(file_key)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        # Block other concurrent runs on the same file so their events don't
        # interleave in the queue. Stays blocked until the worker emits
        # `done` (handled by the poll loop on the client).
        def _run() -> None:
            with session.run_lock:
                try:
                    session.ensure_initialized()
                    if reset_first:
                        session.send({"type": "reset"})
                        # Swallow the reset-ack so it doesn't clutter the
                        # cell's poll stream.
                        _drain_until_ack(session, "reset-ack", timeout=5.0)
                    session.send({
                        "type": "cell-run", "id": cell_id, "code": code,
                    })
                except Exception as e:
                    session._queue.put({
                        "type": "error", "id": cell_id, "error": str(e),
                    })

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"ok": True, "cell_id": cell_id})

    @app.post("/api/cell/poll")
    def api_cell_poll():
        body = request.get_json(silent=True) or {}
        file_key = body.get("file", "<unnamed>")
        with _sessions_lock:
            session = _sessions.get(file_key)
        if session is None:
            return jsonify({"messages": [], "done": True})
        messages = session.poll(timeout=POLL_TIMEOUT_S)
        done = any(m.get("type") in ("done", "error", "worker-exit") for m in messages)
        return jsonify({"messages": messages, "done": done})

    @app.post("/api/cell/reset")
    def api_cell_reset():
        body = request.get_json(silent=True) or {}
        file_key = body.get("file", "<unnamed>")
        try:
            session = _get_or_create(file_key)
            session.ensure_initialized()
            session.send({"type": "reset"})
            _drain_until_ack(session, "reset-ack", timeout=5.0)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.post("/api/cell/interrupt")
    def api_cell_interrupt():
        body = request.get_json(silent=True) or {}
        file_key = body.get("file", "<unnamed>")
        with _sessions_lock:
            session = _sessions.get(file_key)
        if session is None:
            return jsonify({"ok": False, "error": "no active kernel"}), 404
        ok = session.interrupt()
        return jsonify({"ok": bool(ok)})

    @app.delete("/api/kernel")
    def api_kernel_delete():
        body = request.get_json(silent=True) or {}
        file_key = body.get("file", "<unnamed>")
        _remove(file_key)
        return jsonify({"ok": True})


def _drain_until_ack(session: Session, ack_type: str, timeout: float) -> None:
    """Consume queue messages until we see ``ack_type`` (or time out).

    Used after reset so the ack doesn't pollute the next cell-run's poll
    stream. Anything else read here is dropped — at reset time we don't
    care about lingering displays.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        try:
            msg = session._queue.get(timeout=min(0.1, remaining) or 0.01)
        except queue.Empty:
            continue
        if msg.get("type") == ack_type:
            return
    # If we timed out, no big deal — caller proceeds without the ack.
