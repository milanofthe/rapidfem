"""Cell-runner backend for rapidfem serve, one worker subprocess per file.

Replaces the old in-process kernel + WebSocket protocol (which suffered
from os.dup2 / Werkzeug-WS / wsproto-deflate / sendall races). Each open
notebook file gets a long-lived worker subprocess that owns its own
Python namespace. The Flask server brokers JSON messages
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
                          Wipe the namespace (sync).

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

from flask import Flask, Response, jsonify, request


# Per-cell stream poll long-poll timeout. 100 ms keeps the UI responsive
# without burning CPU on empty polls.
POLL_TIMEOUT_S = 0.1

# Worker init must complete within this, covers the rapidfem import.
INIT_TIMEOUT_S = 30.0

# The UI workdir every worker runs in, so a cell's relative paths (GDS,
# meshes, data files) resolve against the folder the file browser shows.
# Set by ``register()`` from ``app.config["RAPIDFEM_WORKDIR"]``; None means
# "inherit the server process cwd" (the pre-workdir behaviour).
_WORKDIR: str | None = None


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
            cwd=_WORKDIR,  # run user cells in the UI workdir (None = inherit)
        )

        self._queue: queue.Queue[dict] = queue.Queue()
        # Request/response channel: messages carrying a `qid` we issued are
        # routed to the matching waiter instead of the poll queue. Used for
        # synchronous worker queries (on-demand field fetch, future
        # introspection) that must not leak into the cell event stream.
        self._pending: dict[str, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._reader_alive = True
        self._reader_thread = threading.Thread(
            target=self._read_stdout, daemon=True,
        )
        self._reader_thread.start()
        # A separate thread to drain stderr, anything the worker writes there
        # is library noise (native panics); surface it as a stream event.
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
                    m = json.loads(line)
                    qid = m.get("qid")
                    if qid is not None:
                        with self._pending_lock:
                            waiter = self._pending.get(qid)
                        if waiter is not None:
                            waiter.put(m)
                            continue
                    self._queue.put(m)
                except json.JSONDecodeError:
                    # Native libs occasionally bypass _ProtocolWriter and write
                    # raw bytes to fd 1, wrap them as stream events so the
                    # user still sees the output.
                    self._queue.put({
                        "type": "stream", "stream": "stdout", "value": line + "\n",
                    })
        except Exception:
            self._queue.put({"type": "worker-exit"})

    def _read_stderr(self) -> None:
        """Forward worker stderr (native panics) as stream events."""
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

    def query(self, msg: dict, timeout: float = 30.0) -> dict:
        """Send a request tagged with a fresh `qid` and block until the worker
        replies with the matching `qid`. The reply is routed past the poll
        queue by `_read_stdout`, so it never appears in the cell stream.

        Note: the worker handles this between cells (its main loop reads the
        next stdin message only after a cell finishes), so a query issued
        while a long cell runs waits until that cell completes.
        """
        qid = uuid.uuid4().hex
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[qid] = waiter
        try:
            self.send({**msg, "qid": qid})
            try:
                return waiter.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError("worker query timed out")
        finally:
            with self._pending_lock:
                self._pending.pop(qid, None)

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
            # stdout/stream messages during init, surface them but keep waiting
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
        # Join the reader threads so their pipe FDs are released before the
        # Session is dropped; process.kill() closes the pipes, which unblocks
        # the readline() calls they are parked on.
        self._reader_thread.join(timeout=1.0)
        self._stderr_thread.join(timeout=1.0)

    def interrupt(self) -> bool:
        """Stop the worker's running cell.

        On POSIX, sends `SIGINT` so the worker's default handler raises
        `KeyboardInterrupt` out of `exec()`, graceful, preserves kernel state
        (works for pure-Python loops; a native solve only checks the signal
        after the call returns, so it isn't interrupted mid-solve there
        either). On Windows there is no reliable signal path to a non-console
        child, so we hard-stop the worker. Either way a long native solve
        (rslab, rapidmesh) can only be stopped by terminating the process, so the
        Windows path is also the robust "stop a runaway solve" path: the worker
        dies, the running cell ends (`worker-exit`), and `_get_or_create`
        spawns a fresh kernel on the next run (state is reset, like Restart).

        Returns False only if the process is already dead.
        """
        if not self.is_alive():
            return False
        if os.name == "nt":
            self._queue.put({
                "type": "stream", "stream": "stderr",
                "value": "\n[kernel interrupted, worker stopped; state reset]\n",
            })
            self.kill()
            return True
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


def _rename(old_key: str, new_key: str) -> bool:
    """Re-key a live session so its kernel state (namespace + the stashed
    sweep result that backs /api/field) survives a Save As of an unsaved
    buffer. No-op (returns False) if no session exists under ``old_key``.
    If ``new_key`` already has a session, that one is killed and replaced
    (the buffer being saved owns the name now).
    """
    if old_key == new_key:
        return True
    with _sessions_lock:
        s = _sessions.get(old_key)
        if s is None:
            return False
        existing = _sessions.pop(new_key, None)
        _sessions.pop(old_key, None)
        s.file_key = new_key
        _sessions[new_key] = s
    if existing:
        existing.kill()
    return True


def shutdown_all() -> None:
    """Kill every worker, used by atexit so subprocesses don't linger."""
    with _sessions_lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for s in sessions:
        s.kill()


# ── Flask endpoints ─────────────────────────────────────────────────────────

def register(app: Flask) -> None:
    """Register /api/cell/* + /api/kernel endpoints on ``app``."""
    global _WORKDIR
    wd = app.config.get("RAPIDFEM_WORKDIR")
    _WORKDIR = str(wd) if wd else None

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

    @app.post("/api/kernel/rename")
    def api_kernel_rename():
        """Re-key a session on Save As so its kernel state (and the stashed
        sweep result behind /api/field) follows the buffer to its new name."""
        body = request.get_json(silent=True) or {}
        old_key = body.get("old", "<unnamed>")
        new_key = body.get("new")
        if not isinstance(new_key, str) or not new_key:
            return jsonify({"ok": False, "error": "new key required"}), 400
        renamed = _rename(old_key, new_key)
        return jsonify({"ok": True, "renamed": renamed})

    @app.get("/api/field")
    def api_field():
        """On-demand field fetch: compute one (freq, port, channel) field in
        the worker (from the last sweep's stashed result) and return it as a
        raw f32 ABC buffer. Only the field currently shown is ever fetched."""
        import base64
        file_key = request.args.get("file", "<unnamed>")
        with _sessions_lock:
            session = _sessions.get(file_key)
        if session is None or not session.is_alive():
            return jsonify({"ok": False, "error": "no active kernel"}), 404
        try:
            resp = session.query({
                "type": "field-query",
                "freq": int(request.args.get("freq", 0)),
                "port": int(request.args.get("port", 0)),
                "channel": request.args.get("channel", "E"),
            }, timeout=30.0)
        except TimeoutError:
            return jsonify({"ok": False, "error": "field query timed out"}), 504
        if not resp.get("ok"):
            return jsonify({"ok": False, "error": resp.get("error", "field error")}), 404
        raw = base64.b64decode(resp.get("data", "") or "")
        return Response(raw, mimetype="application/octet-stream")


def _drain_until_ack(session: Session, ack_type: str, timeout: float) -> None:
    """Consume queue messages until we see ``ack_type`` (or time out).

    Used after reset so the ack doesn't pollute the next cell-run's poll
    stream. Anything else read here is dropped, at reset time we don't
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
    # If we timed out, no big deal, caller proceeds without the ack.
