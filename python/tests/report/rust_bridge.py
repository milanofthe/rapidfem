"""Local Rust-test bridge: feed ``cargo test`` into the rapidfem report.

LOCAL DEVELOPER TOOL — testing in rapidfem runs exclusively on the
developer machine, never in CI. This module runs ``cargo test`` for the
rapidfem crates, parses the textual output into report :class:`Section`
objects (``origin="rust"``), and writes them as a merge JSON that the
pytest plugin picks up via ``--report-merge``.

How the parse works
-------------------
``cargo test`` is run serially with value output::

    cargo test --release -p <crate> -- --nocapture --test-threads=1

With ``--test-threads=1`` each test's ``println!``/``eprintln!`` output is
emitted *before* its ``test <path::name> ... ok`` result line. The parser
therefore buffers every non-result line and, when it hits a result line,
attributes the buffer accumulated since the previous result line to that
test. From the buffer it extracts ``key = value`` numeric pairs as
informational :class:`Metric` items and attaches the raw block as a
:class:`Log`.

CLI::

    python -m report.rust_bridge [--crates rapidfem-core ...] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from .model import Log, Metric, Note, Section, to_jsonable

# Crates whose `cargo test` gates feed the report by default.
DEFAULT_CRATES = ["rapidfem-core", "rapidfem-fd", "rapidfem-td"]

# Generous default timeout per crate (seconds) — local runs may be long.
DEFAULT_TIMEOUT = 3600

# Repo root: python/tests/report/rust_bridge.py -> up three levels.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))

# A cargo test result line, e.g.
#   test mesh::tests::tet_volume ... ok
#   test foo::bar ... FAILED
#   test foo::baz ... ignored
_RESULT_RE = re.compile(
    r"^test\s+(?P<path>[\w:<>{}\- ]+?)\s+\.\.\.\s+"
    r"(?P<outcome>ok|FAILED|ignored)\b.*$"
)

# A `key = value` numeric pair, value in int / float / scientific form.
#   beta = 1.582e+02
#   ||Bmat||_F = 1.287e-02
#   err = 2.3e-12
_KV_RE = re.compile(
    r"^\s*(?P<key>[^=\n]+?)\s*=\s*"
    r"(?P<val>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)

_OUTCOME_STATUS = {"ok": "passed", "FAILED": "failed", "ignored": "skipped"}


def _parse_metrics(block: str) -> list[Metric]:
    """Pull ``key = value`` numeric pairs out of a buffered output block."""
    metrics: list[Metric] = []
    for line in block.splitlines():
        m = _KV_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("val"))
        except ValueError:
            continue
        metrics.append(
            Metric(label=m.group("key").strip(), value=value, cmp="info")
        )
    return metrics


def parse_cargo_output(text: str, group: str = "rust") -> list[Section]:
    """Parse combined ``cargo test`` stdout+stderr into Sections.

    Buffers non-result lines and assigns the block accumulated since the
    previous result line to the test on the next result line. ``group`` is
    used as the section group (typically the crate name).
    """
    sections: list[Section] = []
    buffer: list[str] = []
    # For FAILED tests, cargo prints a `---- <path> stdout ----` capture
    # block in the failures section at the end; map path -> section to
    # backfill failure detail.
    by_path: dict[str, Section] = {}

    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        rm = _RESULT_RE.match(line)
        if rm:
            path = rm.group("path").strip()
            outcome = rm.group("outcome")
            status = _OUTCOME_STATUS.get(outcome, "error")
            name = path.split("::")[-1] if "::" in path else path
            sec_group = group
            # Prefer the module path (everything but the last component) as
            # a finer group when present.
            if "::" in path:
                sec_group = f"{group} :: " + "::".join(path.split("::")[:-1])
            sec = Section(
                nodeid=f"{group}::{path}",
                name=name,
                group=sec_group,
                status=status,
                origin="rust",
            )
            block = "\n".join(buffer).strip()
            if block:
                for met in _parse_metrics(block):
                    sec.items.append(met)
                sec.items.append(Log(title="cargo output", text=block))
            if status == "failed":
                sec.message = block or "test FAILED (see cargo output)"
            sections.append(sec)
            by_path[path] = sec
            buffer = []
            i += 1
            continue

        # Failure capture blocks at the end of a run:
        #   ---- <path> stdout ----
        #   <captured output / panic message>
        fm = re.match(r"^----\s+(?P<path>\S+)\s+stdout\s+----\s*$", line)
        if fm:
            path = fm.group("path").strip()
            j = i + 1
            cap: list[str] = []
            while j < n:
                nxt = lines[j]
                if re.match(r"^----\s+\S+\s+stdout\s+----\s*$", nxt):
                    break
                if re.match(r"^(failures|test result):", nxt.strip()):
                    break
                cap.append(nxt)
                j += 1
            captext = "\n".join(cap).strip()
            sec = by_path.get(path)
            if sec is not None and captext:
                sec.items.append(Log(title="failure detail", text=captext))
                # The captured stdout/panic block is the authoritative
                # failure reason; let it replace any buffered preamble.
                sec.message = captext
            i = j
            continue

        buffer.append(line)
        i += 1

    return sections


def _summary_note(crate: str, sections: list[Section]) -> Section:
    """A small per-crate summary section (ok / failed / ignored counts)."""
    ok = sum(1 for s in sections if s.status == "passed")
    failed = sum(1 for s in sections if s.status == "failed")
    ignored = sum(1 for s in sections if s.status == "skipped")
    status = "failed" if failed else "passed"
    sec = Section(
        nodeid=f"{crate}::__summary__",
        name=f"{crate} summary",
        group=crate,
        status=status,
        origin="rust",
    )
    sec.items.append(
        Note(text=f"cargo test -p {crate}: "
                  f"{ok} ok, {failed} failed, {ignored} ignored.")
    )
    return sec


def run_crate(crate: str, *, timeout: int = DEFAULT_TIMEOUT,
              cwd: str = REPO_ROOT) -> list[Section]:
    """Run ``cargo test`` for one crate and return its parsed Sections.

    On a missing cargo binary, a build failure, or a timeout, a single
    ``status="error"`` section is returned carrying the captured output
    rather than raising.
    """
    cmd = [
        "cargo", "test", "--release", "-p", crate,
        "--", "--nocapture", "--test-threads=1",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return [_error_section(crate, "cargo not found on PATH",
                               "Install the Rust toolchain (cargo).")]
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + "\n" + (exc.stderr or "")
        return [_error_section(crate, f"cargo test timed out after {timeout}s",
                               out)]

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    sections = parse_cargo_output(combined, group=crate)

    # Build / compile failure: no result lines parsed and non-zero exit.
    if not sections and proc.returncode != 0:
        return [_error_section(
            crate, f"cargo test failed (exit {proc.returncode}, no tests ran)",
            combined,
        )]

    sections.append(_summary_note(crate, sections))
    return sections


def _error_section(crate: str, message: str, log_text: str) -> Section:
    sec = Section(
        nodeid=f"{crate}::__error__",
        name=f"{crate} (cargo error)",
        group=crate,
        status="error",
        origin="rust",
        message=message,
    )
    if log_text and log_text.strip():
        sec.items.append(Log(title="cargo output", text=log_text.strip()))
    return sec


def run(crates: list[str], *, timeout: int = DEFAULT_TIMEOUT,
        cwd: str = REPO_ROOT) -> list[Section]:
    """Run every crate serially; return the concatenated Sections."""
    sections: list[Section] = []
    for crate in crates:
        sections.extend(run_crate(crate, timeout=timeout, cwd=cwd))
    return sections


def write_sections(sections: list[Section], out_path: str) -> str:
    """Write sections as ``{"sections": [...]}`` (render.load_sections format)."""
    payload = {"sections": [to_jsonable(s) for s in sections]}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run cargo test and emit rapidfem report sections "
                    "(LOCAL developer tool, never CI).",
    )
    parser.add_argument(
        "--crates", nargs="+", default=DEFAULT_CRATES,
        help=f"crates to test (default: {' '.join(DEFAULT_CRATES)})",
    )
    parser.add_argument(
        "--out", default=os.path.join("_report", "rust-sections.json"),
        help="output JSON path (default: _report/rust-sections.json)",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"per-crate timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--cwd", default=REPO_ROOT,
        help="working directory for cargo (default: repo root)",
    )
    args = parser.parse_args(argv)

    sections = run(args.crates, timeout=args.timeout, cwd=args.cwd)
    out = write_sections(sections, args.out)
    n_fail = sum(1 for s in sections if s.status in ("failed", "error"))
    print(f"rust bridge: {len(sections)} sections "
          f"({n_fail} failed/error) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
