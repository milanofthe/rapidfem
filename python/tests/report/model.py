"""Data model for the rapidfem test report.

Plain dataclasses shared by every producer (the pytest plugin, the Rust
bridge) and the single consumer (:mod:`render`). Everything is JSON-
serialisable via :func:`to_jsonable` so a Python run and a Rust run can be
merged into one report.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

# Test outcome, in severity order (worst first when aggregating).
STATUS_ORDER = ["error", "failed", "xfailed", "skipped", "xpassed", "passed"]

# Item kinds a test can attach to its section.
KIND_METRIC = "metric"
KIND_TABLE = "table"
KIND_PLOT = "plot"
KIND_NOTE = "note"
KIND_LOG = "log"


@dataclass
class Metric:
    """A single checked number.

    ``cmp`` selects how ``value`` is judged against ``ref``:
    ``"approx"`` (|value-ref| within tolerance), ``"le"`` (value ≤ ref,
    an upper bound), ``"ge"`` (value ≥ ref, a lower bound) or ``"info"``
    (no check). ``tol`` is the relative tolerance shown for ``approx``.
    """

    label: str
    value: float | str
    ref: float | None = None
    tol: float | None = None          # relative tolerance (approx only)
    cmp: str = "approx"               # approx | le | ge | info
    unit: str = ""
    passed: bool | None = None        # None → informational only
    detail: str = ""
    kind: str = field(default=KIND_METRIC, init=False)


@dataclass
class Table:
    title: str
    headers: list[str]
    rows: list[list[Any]]
    kind: str = field(default=KIND_TABLE, init=False)


@dataclass
class Plot:
    """An inline SVG figure plus caption."""

    title: str
    svg: str
    caption: str = ""
    kind: str = field(default=KIND_PLOT, init=False)


@dataclass
class Note:
    text: str                          # markdown-ish; rendered as paragraphs
    kind: str = field(default=KIND_NOTE, init=False)


@dataclass
class Log:
    title: str
    text: str
    kind: str = field(default=KIND_LOG, init=False)


@dataclass
class Section:
    """One test (or one Rust gate): its outcome plus attached artefacts."""

    nodeid: str
    name: str                          # short display name
    group: str                         # grouping key (module / crate)
    status: str = "passed"
    duration: float = 0.0              # seconds
    message: str = ""                  # failure / skip reason
    doc: str = ""                      # test docstring, shown as intro
    items: list[Any] = field(default_factory=list)
    origin: str = "python"             # "python" | "rust"


@dataclass
class Report:
    title: str = "rapidfem test report"
    generated: str = ""                # ISO timestamp (injected, not stamped here)
    subtitle: str = ""
    meta: dict[str, str] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)

    def status_counts(self) -> dict[str, int]:
        counts = {s: 0 for s in STATUS_ORDER}
        for sec in self.sections:
            counts[sec.status] = counts.get(sec.status, 0) + 1
        return counts


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / containers to JSON-safe types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def section_from_dict(d: dict) -> Section:
    """Rebuild a :class:`Section` (with typed items) from a plain dict —
    the inverse of :func:`to_jsonable` for the merge step."""
    item_types = {
        KIND_METRIC: Metric,
        KIND_TABLE: Table,
        KIND_PLOT: Plot,
        KIND_NOTE: Note,
        KIND_LOG: Log,
    }
    items = []
    for it in d.get("items", []):
        cls = item_types.get(it.get("kind"))
        if cls is None:
            continue
        payload = {k: v for k, v in it.items() if k != "kind"}
        items.append(cls(**payload))
    sec = Section(
        nodeid=d["nodeid"],
        name=d["name"],
        group=d["group"],
        status=d.get("status", "passed"),
        duration=d.get("duration", 0.0),
        message=d.get("message", ""),
        doc=d.get("doc", ""),
        origin=d.get("origin", "python"),
    )
    sec.items = items
    return sec
