# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Pre-bake bundled examples for the static web demo.

Reads each `python/python_src/rapidfem/examples/*.py`, splits it into
notebook cells along `# %%` markers, runs each cell through the same
kernel pipeline the live server uses, captures every `rapidfem.show()`
display payload + stdout/stderr, and writes the recordings to
``python/python_src/rapidfem/ui/frontend-src/static/demo/``.

Run from the repo root:

    python scripts/bake_demo.py

Reused machinery (single source of truth for runtime behaviour):
- capture lifecycle           → rapidfem.ui.worker
- show() collector            → rapidfem.ui.capture
- serializer (display events) → rapidfem.ui.serialize / api._serialize_captures_for_protocol
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Each example is baked in its own subprocess (see `_bake_subprocess`) so a
# hang or crash takes down only its own attempt, not the whole bake.
# Per-example overrides live in <name>.bake.json (see `_bake_config`).
BAKE_TIMEOUT_S = 600   # kill + retry an example that runs longer than this
BAKE_ATTEMPTS = 3      # attempts per example before it is skipped

# Bumped whenever the meta.json schema changes, older meta files are
# treated as missing and force a re-bake.
META_SCHEMA = 2

# Manifest is a *derived* view rebuilt after every bake operation from the
# per-example meta files. Version is independent of the meta schema.
MANIFEST_SCHEMA = 2

# Bake subprocesses pin OpenMP and BLAS to one thread; the Rust mesher and
# solvers use their own rayon pools, so they stay parallel.
_BAKE_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

# The curated web-demo set, only these examples are baked into the static
# demo. Every other script under `examples/` stays a package example
# (runnable, listed in the live `rapidfem serve` notebook) but is left out
# of the web demo: numbers-only validation/benchmark scripts and near-
# duplicate geometries do not earn a card on the landing page.
DEMO_EXAMPLES = frozenset({
    # frequency-domain
    "fd_wr90",
    "fd_coax_step",
    "fd_microstrip_line",
    "fd_iris_filter",
    "fd_patch_antenna",
    "fd_inverted_f_antenna",
    "fd_pyramidal_horn",
    "fd_dielectric_resonator",
    # RFIC layout import
    "fd_rfic_spiral_from_json",
    # time-domain, driven modal-port families + complex structures, each a
    # field animation plus a transient port/probe plot
    "td_rect_waveguide",
    "td_coax_line",
    "td_wave_port",
    "td_dielectric_step",
    "td_power_divider",
    "td_horn_radiation",
    "td_coax_open",
})

# Windows console defaults to cp1252, print() falls over on the unicode
# box-drawing chars we use in summaries. Force UTF-8 on the std streams.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# Matches a line that, after optional whitespace, starts with `# %%`.
# Same predicate as Notebook.svelte's `parse()` (regex /^\s*#\s*%%/).
_MARKER_RE = re.compile(r"^\s*#\s*%%")


@dataclass
class Cell:
    """One notebook cell, body text plus its marker line (if any)."""

    text: str
    marker: str | None  # the `# %%` line that introduces this cell, or None
                       # for the implicit first cell when the file does not
                       # start with a marker.

    @property
    def has_code(self) -> bool:
        """True if the cell body has anything other than whitespace."""
        return self.text.strip() != ""


def split_cells(source: str) -> list[Cell]:
    """Split Python source into cells along `# %%` marker lines.

    Mirrors the parsing logic in ``Notebook.svelte:parse`` so cell
    boundaries in the bake step match what users see in the live UI:

    - A marker is any line matching ``^\\s*#\\s*%%``.
    - Text before the first marker becomes an implicit first cell with
      ``marker=None``. If that cell is empty (file starts with a marker),
      it is dropped.
    - The marker line itself is stored on the cell that follows it, NOT
      included in ``cell.text``, so executing ``cell.text`` is safe and
      matches what the live editor sends to the kernel.
    """
    lines = source.split("\n")
    out: list[Cell] = []
    buf: list[str] = []
    marker: str | None = None

    def flush() -> None:
        out.append(Cell(text="\n".join(buf), marker=marker))

    for line in lines:
        if _MARKER_RE.match(line):
            flush()
            marker = line
            buf = []
        else:
            buf.append(line)
    flush()

    # Drop leading empty cell if the file starts with a marker.
    if len(out) > 1 and out[0].marker is None and out[0].text.strip() == "":
        out.pop(0)

    return out


def serialize_cells(cells: Iterable[Cell]) -> str:
    """Inverse of split_cells, round-trip-safe within a tolerance.

    Reassembles the source from cells; intended as a sanity check, not
    as a canonical formatter. Reproduces the same shape Notebook.svelte's
    ``serialize()`` produces.
    """
    parts: list[str] = []
    for c in cells:
        if c.marker is not None:
            parts.append(c.marker)
        parts.append(c.text)
    return "\n".join(parts)


# ── Per-cell exec via the live kernel pipeline ───────────────────────────


def _bake_cell(cell: Cell, kernel) -> dict:
    """Run a single cell through the same pipeline the WS server uses and
    return a JSON-serialisable record of what happened.

    Reuses the production helpers, ``rapidfem.ui.capture`` for ``rapidfem.show()``
    payloads, ``_capture_streams`` for native fd-level stdout/stderr,
    ``_serialize_captures_for_protocol`` for the display-event shape, so
    a baked run cannot diverge from a live run by accident.
    """
    # Imported lazily so a `python scripts/bake_demo.py` that only runs the
    # cell-splitter smoke test does not require the [ui] extra (Flask, etc.).
    from rapidfem.ui import capture as _show_capture
    from rapidfem.ui.api import (
        _capture_streams,
        _format_exception,
        _serialize_captures_for_protocol,
    )

    stream_lines: list[dict] = []

    def on_line(stream: str, line: str) -> None:
        stream_lines.append({"stream": stream, "line": line})

    err: BaseException | None = None
    _show_capture.start_capture()
    try:
        with _capture_streams(on_line=on_line, stage="cell"):
            try:
                exec(compile(cell.text, kernel.file_path or "<cell>", "exec"),
                     kernel.namespace)
            except BaseException as e:  # noqa: BLE001, same shape as the UI worker
                err = e
    finally:
        captured = _show_capture.stop_capture()

    record: dict = {
        "marker": cell.marker,
        "code": cell.text,
        "stream_lines": stream_lines,
    }

    if err is not None:
        record["status"] = "error"
        record["error"] = _format_exception(err)
        record["display_events"] = []
        return record

    try:
        # eager_fields=True: the static demo has no worker to answer
        # /api/field, so driven-sweep E/J/H must be embedded here and packed
        # into <name>.field.bin (the live serve path keeps them lazy).
        display_events = _serialize_captures_for_protocol(captured, eager_fields=True)
    except Exception as e:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = _format_exception(e)
        record["display_events"] = []
        return record

    record["status"] = "ok"
    record["display_events"] = display_events
    return record


# ── Bulk-payload binarisation ────────────────────────────────────────────


def _extract_payloads_to_bin(record: dict, bin_dir: Path) -> list[Path]:
    """Lift every bulk numeric array out of a baked record into the two
    binary sidecars, ``<name>.geo.bin`` (mesh / geometry) and
    ``<name>.field.bin`` (field / trajectory data), replacing each array
    in the record with a ``$bin`` reference.

    Uses :mod:`rapidfem.ui.binpack`. Returns the bin files actually written; a
    buffer holding nothing but its 8-byte header (an example with no mesh,
    or no field data) is skipped.
    """
    from rapidfem.ui import binpack

    events = [
        ev
        for cell in record.get("cells", [])
        for ev in cell.get("display_events", [])
    ]
    geo, field = binpack.pack(events)

    name = record["name"]
    written: list[Path] = []
    for blob, suffix in ((geo, "geo"), (field, "field")):
        if len(blob) > 8:  # more than the bare header
            path = bin_dir / f"{name}.{suffix}.bin"
            path.write_bytes(blob)
            written.append(path)
    return written


def bake_example(path: Path) -> dict:
    """Bake one example file end-to-end.

    Returns ``{name, source, cells: [<cell records>]}``. Cells run
    sequentially against a single persistent namespace, the same model
    the live notebook uses, so variables defined in cell N are visible
    in cell N+1.
    """
    # The bake runs in-process, we don't need the subprocess kernel from
    # rapidfem.ui.runner, just a tiny namespace + the show-capture +
    # serialize pipeline. Live serving uses a real worker subprocess.
    import rapidfem

    source = path.read_text(encoding="utf-8")
    cells = split_cells(source)


    class _Kernel:
        file_path = path.name
        namespace = {
            "__name__": "__rapidfem_bake__",
            "__file__": path.name,
            "rapidfem": rapidfem,
        }

    kernel = _Kernel()

    cell_records: list[dict] = []
    for c in cells:
        if not c.has_code:
            # Pure-whitespace cell (e.g. the implicit one ahead of a
            # docstring-only file). Preserve the marker but skip exec.
            cell_records.append({
                "marker": c.marker,
                "code": c.text,
                "status": "ok",
                "stream_lines": [],
                "display_events": [],
            })
            continue
        cell_records.append(_bake_cell(c, kernel))

    return {
        "name": path.stem,
        "filename": path.name,
        "source": source,
        "cells": cell_records,
    }


# ── Smoke test / dev entry point ─────────────────────────────────────────
def _smoke() -> int:
    """Walk the bundled examples and print a cell summary for each."""
    here = Path(__file__).resolve().parent.parent
    examples_dir = here / "python" / "python_src" / "rapidfem" / "examples"
    files = sorted(p for p in examples_dir.glob("*.py") if not p.name.startswith("_"))
    if not files:
        print(f"no examples found under {examples_dir}")
        return 1

    for path in files:
        src = path.read_text(encoding="utf-8")
        cells = split_cells(src)
        print(f"\n── {path.name}  ({len(cells)} cells)")
        for i, c in enumerate(cells):
            head = (c.marker or "").strip() or "(implicit first cell)"
            n_lines = c.text.count("\n") + (1 if c.text else 0)
            print(f"  [{i}] {head}  · {n_lines} line(s), {len(c.text)} chars")
        # Round-trip check
        if serialize_cells(cells) != src:
            # Tolerated only for trailing-newline edge cases.
            a = serialize_cells(cells).rstrip("\n")
            b = src.rstrip("\n")
            if a != b:
                print(f"  WARNING: round-trip diverges for {path.name}")
    return 0


def _output_dir() -> Path:
    """Where baked artefacts land: the SvelteKit static dir, copied to
    ``dist/demo/`` at build time."""
    here = Path(__file__).resolve().parent.parent
    return here / "python" / "python_src" / "rapidfem" / "ui" / "frontend-src" / "static" / "demo"


def _examples_dir() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / "python" / "python_src" / "rapidfem" / "examples"


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` via tempfile + replace, so a crash mid-write
    never leaves a half-written file that the next run would mistake for a
    valid record. Same-directory tempfile so rename stays atomic on Windows.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_source_file(path: Path) -> str:
    """SHA256 of a Python source file, with CRLF normalised to LF so a
    Windows checkout doesn't see a different hash from a Linux checkout
    of the same content. Returns an empty string for unreadable files."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return _hash_text(text.replace("\r\n", "\n"))


_PACKAGE_HASH_CACHE: str | None = None


def _package_source_hash() -> str:
    """Stable hash over every `.py` under `rapidfem/` (excluding examples/)
    plus this bake script. Detects shared-code edits, a fix in
    `problem/fd.py` invalidates every FD example, a tweak to
    `geometry.py` invalidates all examples that touch meshing, without
    cascading on per-example edits, which `examples/` is carved out for
    (per-example `src_hash` already handles those).

    Compiled extensions (`.pyd`/`.so`) are deliberately NOT hashed: the
    binary content depends on the build host, so hashing them would
    invalidate caches on every editable rebuild. The user-facing version
    bump in `pyproject.toml` (which touches `__init__.py` etc.) is the
    natural cache-invalidation signal for ABI changes.
    """
    global _PACKAGE_HASH_CACHE
    if _PACKAGE_HASH_CACHE is not None:
        return _PACKAGE_HASH_CACHE
    here = Path(__file__).resolve().parent.parent
    pkg = here / "python" / "python_src" / "rapidfem"
    examples_dir = pkg / "examples"
    parts: list[tuple[str, str]] = [("<bake_demo>",
                                     _hash_source_file(Path(__file__)))]
    for f in sorted(pkg.rglob("*.py")):
        if examples_dir in f.parents:
            continue
        rel = f.relative_to(pkg).as_posix()
        parts.append((rel, _hash_source_file(f)))
    blob = "\n".join(f"{rel}={h}" for rel, h in parts)
    _PACKAGE_HASH_CACHE = _hash_text(blob)
    return _PACKAGE_HASH_CACHE


def _meta_path(name: str, out_dir: Path) -> Path:
    return out_dir / f"{name}.meta.json"


def _read_meta(name: str, out_dir: Path) -> dict | None:
    p = _meta_path(name, out_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_meta(name: str, out_dir: Path, meta: dict) -> None:
    _atomic_write_text(_meta_path(name, out_dir),
                       json.dumps(meta, indent=2))


def _bake_config(name: str) -> dict:
    """Per-example orchestrator config from `<name>.bake.json` next to the
    example source. JSON (not TOML) so we stay stdlib-only on Python 3.10.
    Controls *orchestrator behaviour* only, retries, timeout. Runtime
    behaviour of the example (mesh kwargs, solver kwargs) stays in the
    example file so it remains runnable as a plain Python script.

    Schema (all optional):
        {"retries": int, "timeout": int}
    """
    here = Path(__file__).resolve().parent.parent
    cfg_path = (here / "python" / "python_src" / "rapidfem" / "examples"
                / f"{name}.bake.json")
    cfg = {"retries": BAKE_ATTEMPTS, "timeout": BAKE_TIMEOUT_S}
    if cfg_path.is_file():
        try:
            user = json.loads(cfg_path.read_text(encoding="utf-8"))
            for k in ("retries", "timeout"):
                if k in user:
                    cfg[k] = int(user[k])
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"warning: failed to parse {cfg_path.name}: {e}",
                  file=sys.stderr)
    return cfg


def _is_fresh_hash(meta: dict | None, src_path: Path,
                   package_hash: str) -> bool:
    """Hash-based freshness check. An example is fresh iff:
      * `meta.json` exists with the current schema
      * the last bake succeeded (`status == "ok"`)
      * the source `.py` content hash matches
      * the package-wide source hash matches

    Failed bakes are NOT considered fresh: the user might change a flag in
    `<name>.bake.json` to recover, and we want to re-run rather than
    persist a stale failure. They're skipped without `--force` only when
    *both* the source AND the failure mode look identical (handled in
    `bake_all` via the meta record).
    """
    if meta is None:
        return False
    if meta.get("schema") != META_SCHEMA:
        return False
    if meta.get("status") != "ok":
        return False
    if meta.get("src_hash") != _hash_source_file(src_path):
        return False
    if meta.get("package_hash") != package_hash:
        return False
    return True


def _rebuild_manifest(out_dir: Path) -> dict:
    """Rebuild `manifest.json` as a derived view over every
    `<name>.meta.json` in `out_dir`. The manifest is *never* edited
    directly anywhere else, so a partial bake (one example via
    `--bake-one`, or a crash mid-orchestrator) can't leave the UI looking
    at a stale index. Atomic write via tempfile + rename.
    """
    entries: list[dict] = []
    for meta_file in sorted(out_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("schema") != META_SCHEMA:
            continue
        # UI consumers only require these five keys; the rest is for human
        # inspection and for the bake script's own freshness logic.
        entries.append({
            "name": meta["name"],
            "filename": meta["filename"],
            "json": f"{meta['name']}.json",
            "bin_files": meta.get("bin_files", []),
            "n_cells": meta.get("n_cells", 0),
            "cells": meta.get("cells", []),
            "json_bytes": meta.get("json_bytes", 0),
            "bin_bytes": meta.get("bin_bytes", 0),
            "bake_seconds": meta.get("bake_seconds", 0),
            "status": meta.get("status", "unknown"),
        })
    manifest = {
        "version": MANIFEST_SCHEMA,
        "baked_at": int(time.time()),
        "examples": entries,
    }
    _atomic_write_text(out_dir / "manifest.json",
                       json.dumps(manifest, indent=2))
    return manifest


def _cell_summaries(record: dict) -> list[dict]:
    """Compact per-cell status for the manifest (lets the UI surface error
    markers without parsing the full JSON eagerly)."""
    return [
        {
            "marker": c.get("marker"),
            "status": c.get("status"),
            "kinds": [d.get("kind") for d in c.get("display_events", [])],
            "stream_lines": len(c.get("stream_lines", [])),
        }
        for c in record.get("cells", [])
    ]


@dataclass
class BakeOutcome:
    """Verdict of a `_bake_subprocess` run. `success` carries the wall-clock
    seconds on the successful attempt; `failure` carries the last attempt's
    exit code + a short detail string so the caller can record it in the
    meta file (we want failures to be observable, not just absent)."""
    success_seconds: float | None = None
    last_exit_code: int | None = None
    last_detail: str = ""
    attempts: int = 0


def _bake_subprocess(name: str, log: Path) -> BakeOutcome:
    """Bake one example in a fresh ``--bake-one`` subprocess. Each attempt
    gets its own subprocess so a hang or crash takes down only its own
    attempt.

    Retries and timeout come from the per-example config in
    `<name>.bake.json` if present, falling back to `BAKE_ATTEMPTS` and
    `BAKE_TIMEOUT_S`. Setting `retries=1` for examples with known
    deterministic failures (a crash in native code) avoids burning
    three attempts on something that will fail the same way every time.
    """
    cfg = _bake_config(name)
    max_attempts: int = cfg["retries"]
    timeout_s: int = cfg["timeout"]

    cmd = [sys.executable, str(Path(__file__).resolve()), "--bake-one", name]
    out = BakeOutcome()
    for attempt in range(1, max_attempts + 1):
        out.attempts = attempt
        t0 = time.perf_counter()
        outcome: str
        detail = ""
        exit_code: int | None = None
        try:
            proc = subprocess.run(
                cmd, timeout=timeout_s, capture_output=True, text=True,
                env=_BAKE_ENV,
            )
            dt = time.perf_counter() - t0
            if proc.returncode == 0:
                print(f"   baked {name} in {dt:.1f}s (attempt {attempt})",
                      file=sys.stderr)
                _append_log(log, name, attempt, "ok", proc.stdout, proc.stderr)
                out.success_seconds = dt
                return out
            outcome = f"exit {proc.returncode} after {dt:.1f}s"
            detail = (proc.stderr or "")
            exit_code = proc.returncode
            _append_log(log, name, attempt, outcome, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as e:
            dt = time.perf_counter() - t0
            outcome = f"TIMED OUT after {timeout_s}s, killed"
            detail = (e.stderr or b"").decode("utf-8", "replace") \
                if isinstance(e.stderr, bytes) else (e.stderr or "")
            exit_code = -1  # sentinel: process killed by timeout
            _append_log(log, name, attempt, outcome, "", detail)
        out.last_exit_code = exit_code
        out.last_detail = outcome
        print(f"   attempt {attempt}/{max_attempts} for {name}: {outcome}",
              file=sys.stderr)
        for line in detail.strip().splitlines()[-6:]:
            print(f"     | {line}", file=sys.stderr)
    return out


def _append_log(log: Path, name: str, attempt: int, outcome: str,
                out: str, err: str) -> None:
    """Append one bake attempt's full output to the diagnostics log."""
    try:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n{'='*70}\n{name}  attempt {attempt}  -> {outcome}\n")
            if out.strip():
                fh.write(f"--- stdout ---\n{out}\n")
            if err.strip():
                fh.write(f"--- stderr ---\n{err}\n")
    except OSError:
        pass


def bake_all(force: bool = False) -> dict:
    """Bake every example in :data:`DEMO_EXAMPLES` into ``static/demo/``.

    Lazy by default: an example is reused iff its `meta.json` shows
      * the current schema,
      * `status == "ok"`,
      * `src_hash` matches the current `.py`,
      * `package_hash` matches the current rapidfem package.

    Source-hashing replaces the old global-mtime fingerprint: editing
    `examples/foo.py` no longer invalidates every other example via a
    cascading mtime bump, and a `pip install -e` rebuild only causes a
    re-bake when the compiled extension actually changed package source
    (the hash is content-based).

    The `manifest.json` is rebuilt at the end from on-disk meta files, so
    a single example baked via `--bake-one` produces an equivalent
    manifest entry without an orchestrator run.

    `force=True` ignores freshness and bakes everything.
    """
    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    examples_dir = _examples_dir()
    py_files = sorted(p for p in examples_dir.glob("*.py")
                      if p.stem in DEMO_EXAMPLES)
    if not py_files:
        raise FileNotFoundError(
            f"no DEMO_EXAMPLES .py files under {examples_dir}")
    missing = DEMO_EXAMPLES - {p.stem for p in py_files}
    if missing:
        print(f"warning: DEMO_EXAMPLES not found on disk: "
              f"{', '.join(sorted(missing))}", file=sys.stderr)

    expected_names = {p.stem for p in py_files}

    # Prune orphans (example renamed/removed): json, bin, meta. Files not
    # matching the output naming pattern are left alone.
    for old in list(out_dir.glob("*.json")):
        if old.name == "manifest.json":
            continue
        stem = old.stem
        if stem.endswith(".meta"):
            stem = stem[:-5]
        if stem not in expected_names:
            try:
                old.unlink()
            except OSError:
                pass
    for old in list(out_dir.glob("*.bin")):
        prefix = old.name.rsplit(".", 2)[0]
        if prefix not in expected_names:
            try:
                old.unlink()
            except OSError:
                pass

    bake_log = out_dir / "_bake.log"
    bake_log.write_text(
        f"bake_demo run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )

    package_hash = _package_source_hash()
    baked = 0
    reused = 0
    skipped: list[str] = []

    for path in py_files:
        meta = _read_meta(path.stem, out_dir)
        fresh = (not force) and _is_fresh_hash(meta, path, package_hash)

        if fresh:
            json_bytes = (meta or {}).get("json_bytes", 0)
            bin_bytes = (meta or {}).get("bin_bytes", 0)
            print(f"── reusing {path.name}  ({json_bytes:,} B json"
                  f" + {bin_bytes:,} B bin)", file=sys.stderr)
            reused += 1
            continue

        # Skip-with-record: same source already failed deterministically
        # last time AND no `--force`. Avoids reburning attempts on a known-
        # broken example (typical for heap-corruption cases). User can
        # adjust `<name>.bake.json` or fix the source to invalidate the
        # hash and try again.
        if (not force) and meta is not None \
                and meta.get("schema") == META_SCHEMA \
                and meta.get("status") == "failed" \
                and meta.get("src_hash") == _hash_source_file(path) \
                and meta.get("package_hash") == package_hash:
            print(f"── skipping {path.name}, same source as last failed "
                  f"attempt (status={meta.get('exit_code')!r}; "
                  f"edit src or run with --force)", file=sys.stderr)
            skipped.append(path.name)
            continue

        print(f"\n── baking {path.name}", file=sys.stderr)
        outcome = _bake_subprocess(path.stem, bake_log)
        if outcome.success_seconds is None:
            print(f"!! FAILED {path.name}, {outcome.attempts} attempt(s); "
                  f"recording failure in meta (see {bake_log.name})",
                  file=sys.stderr)
            _write_meta(path.stem, out_dir, {
                "schema": META_SCHEMA,
                "name": path.stem,
                "filename": path.name,
                "src_hash": _hash_source_file(path),
                "package_hash": package_hash,
                "baked_at": int(time.time()),
                "bake_seconds": 0.0,
                "status": "failed",
                "exit_code": outcome.last_exit_code,
                "failure_detail": outcome.last_detail,
                "n_cells": 0,
                "cells": [],
                "bin_files": [],
                "json_bytes": 0,
                "bin_bytes": 0,
            })
            skipped.append(path.name)
            continue

        # Success, the subprocess wrote meta + bins + json; nothing more
        # to do here per-example. Print a one-line size summary.
        post_meta = _read_meta(path.stem, out_dir) or {}
        print(
            f"   {post_meta.get('json_bytes', 0):>9,} B json  +  "
            f"{post_meta.get('bin_bytes', 0):>9,} B bin"
            f"   ({len(post_meta.get('bin_files', []))} bin)",
            file=sys.stderr,
        )
        baked += 1

    # One final manifest rebuild: derives the index from whatever meta
    # files actually landed on disk, even if the loop bailed early on some.
    manifest = _rebuild_manifest(out_dir)
    total_json = sum(e["json_bytes"] for e in manifest["examples"])
    total_bin = sum(e["bin_bytes"] for e in manifest["examples"])
    print(
        f"\nwrote manifest.json with {len(manifest['examples'])} example(s)"
        f" ({baked} baked, {reused} reused, {len(skipped)} skipped);"
        f" total {total_json:,} B json + {total_bin:,} B bin"
        f" = {total_json + total_bin:,} B",
        file=sys.stderr,
    )
    if skipped:
        print(f"!! {len(skipped)} example(s) skipped: {', '.join(skipped)}",
              file=sys.stderr)
        print(f"   per-attempt diagnostics: {bake_log}", file=sys.stderr)
    return manifest


def _bake_one(name: str) -> int:
    """Bake a single example end-to-end. Writes:

      * ``<name>.json``      cell records (with bin-refs in display events)
      * ``<name>.*.bin``     packed mesh / field buffers
      * ``<name>.meta.json`` fingerprint + per-cell summary (the freshness
                             oracle for the next run)
      * ``manifest.json``    derived from all `*.meta.json`, atomic

    Crucially, this path is **self-sufficient**: the manifest gets rebuilt
    here, not deferred to an orchestrator run. A bare `--bake-one fd_foo`
    is immediately visible in the UI.
    """
    here = Path(__file__).resolve().parent.parent
    target = here / "python" / "python_src" / "rapidfem" / "examples" / name
    if not target.exists() and target.suffix != ".py":
        target = target.with_suffix(".py")
    if not target.exists():
        print(f"example not found: {target}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    print(f"baking {target.name}...", file=sys.stderr)
    record = bake_example(target)
    dt = time.perf_counter() - t0

    for i, c in enumerate(record["cells"]):
        head = (c["marker"] or "").strip() or "(implicit)"
        ds = c.get("display_events", [])
        kinds = ",".join(d.get("kind", "?") for d in ds) or ", "
        sl = len(c.get("stream_lines", []))
        status = c["status"]
        print(f"  [{i}] {head:30s}  status={status:5s}  display=[{kinds}]  stream={sl}",
              file=sys.stderr)

    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_json_size = len(json.dumps(record))

    bin_files = _extract_payloads_to_bin(record, out_dir)
    json_path = out_dir / f"{record['name']}.json"
    _atomic_write_text(json_path, json.dumps(record))

    post_json_size = json_path.stat().st_size
    bin_total = sum(p.stat().st_size for p in bin_files)
    print(
        f"\nwrote {json_path.name} ({post_json_size:,} B)"
        f" + {len(bin_files)} bin file(s) totalling {bin_total:,} B",
        file=sys.stderr,
    )
    print(
        f"  JSON savings: {raw_json_size:,} -> {post_json_size:,} B  "
        f"({100 * (1 - post_json_size / raw_json_size):.1f}% reduction)",
        file=sys.stderr,
    )
    print(
        f"  combined on-disk: {post_json_size + bin_total:,} B "
        f"({(post_json_size + bin_total) / raw_json_size * 100:.1f}% of raw JSON)",
        file=sys.stderr,
    )

    meta = {
        "schema": META_SCHEMA,
        "name": record["name"],
        "filename": record["filename"],
        "src_hash": _hash_source_file(target),
        "package_hash": _package_source_hash(),
        "baked_at": int(time.time()),
        "bake_seconds": round(dt, 2),
        "status": "ok",
        "exit_code": 0,
        "n_cells": len(record["cells"]),
        "cells": _cell_summaries(record),
        "bin_files": [p.name for p in bin_files],
        "json_bytes": post_json_size,
        "bin_bytes": bin_total,
    }
    _write_meta(record["name"], out_dir, meta)
    _rebuild_manifest(out_dir)
    return 0


def _bootstrap_meta() -> int:
    """One-shot migration: synthesise `<name>.meta.json` files for every
    existing baked artefact under `static/demo/`, using the current source
    hashes. Lets the hash-based freshness check see pre-existing artefacts
    as fresh on the first run after the meta-schema refactor, without
    forcing a 30-minute re-bake of everything.
    """
    out_dir = _output_dir()
    examples_dir = _examples_dir()
    if not out_dir.is_dir():
        print(f"no static/demo dir at {out_dir}", file=sys.stderr)
        return 1
    package_hash = _package_source_hash()
    written = 0
    for json_path in sorted(out_dir.glob("*.json")):
        if json_path.name in ("manifest.json",):
            continue
        if json_path.name.endswith(".meta.json"):
            continue
        name = json_path.stem
        if name not in DEMO_EXAMPLES:
            continue
        src_path = examples_dir / f"{name}.py"
        if not src_path.is_file():
            print(f"skip {name}: no source .py", file=sys.stderr)
            continue
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {name}: failed to read record ({e})", file=sys.stderr)
            continue
        bin_files = sorted(out_dir.glob(f"{name}.*.bin"))
        meta = {
            "schema": META_SCHEMA,
            "name": name,
            "filename": src_path.name,
            "src_hash": _hash_source_file(src_path),
            "package_hash": package_hash,
            "baked_at": int(json_path.stat().st_mtime),
            "bake_seconds": 0.0,
            "status": "ok",
            "exit_code": 0,
            "n_cells": len(record.get("cells", [])),
            "cells": _cell_summaries(record),
            "bin_files": [p.name for p in bin_files],
            "json_bytes": json_path.stat().st_size,
            "bin_bytes": sum(p.stat().st_size for p in bin_files),
        }
        _write_meta(name, out_dir, meta)
        written += 1
        print(f"  wrote {name}.meta.json", file=sys.stderr)
    _rebuild_manifest(out_dir)
    print(f"\nbootstrapped {written} meta file(s); rebuilt manifest.json",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bake-one":
        if len(sys.argv) < 3:
            print("usage: bake_demo.py --bake-one <example_name>", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(_bake_one(sys.argv[2]))
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        raise SystemExit(_smoke())
    if len(sys.argv) > 1 and sys.argv[1] == "--bootstrap-meta":
        raise SystemExit(_bootstrap_meta())
    # Default: lazy bake (only stale examples). --force re-bakes everything.
    force = "--force" in sys.argv[1:]
    bake_all(force=force)
    raise SystemExit(0)
