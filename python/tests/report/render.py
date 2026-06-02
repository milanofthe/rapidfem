"""Render a :class:`~report.model.Report` to a single self-contained HTML
file, plus JSON (de)serialisation for merging a Python run with a Rust run.
"""
from __future__ import annotations

import json
import math
import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import style
from .model import Report, Section, section_from_dict, to_jsonable

_HERE = os.path.dirname(__file__)
_TEMPLATES = os.path.join(_HERE, "templates")
_FAVICON = os.path.normpath(
    os.path.join(
        _HERE, "..", "..", "python_src", "rapidfem", "ui",
        "frontend-src", "static", "favicon.svg",
    )
)


def _fmt(v) -> str:
    """Human-friendly number formatting for the report tables."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0:
        return "0"
    if float(x).is_integer() and abs(x) < 1e6:
        return str(int(x))
    a = abs(x)
    if a >= 1e5 or a < 1e-3:
        return f"{x:.4e}"
    return f"{x:.4g}"


def _load_logo() -> str:
    try:
        with open(_FAVICON, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["fmt"] = _fmt
    return env


def _grouped(report: Report):
    """Sections grouped by ``group``, preserving first-seen group order."""
    order: list[str] = []
    buckets: dict[str, list[Section]] = {}
    for sec in report.sections:
        if sec.group not in buckets:
            buckets[sec.group] = []
            order.append(sec.group)
        buckets[sec.group].append(sec)
    return [(g, buckets[g]) for g in order]


def render_html(report: Report, out_path: str, *, producer: str = "pytest") -> str:
    """Write ``report`` as a self-contained HTML file; return the path."""
    env = _env()
    tpl = env.get_template("report.html.j2")
    total_duration = sum(s.duration for s in report.sections)
    html = tpl.render(
        report=report,
        t=style.TOKENS,
        grouped=_grouped(report),
        total_duration=total_duration,
        logo_svg=_load_logo(),
        producer=producer,
        fmt=_fmt,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def dump_json(report: Report, out_path: str) -> str:
    """Serialise a report to JSON (for the Python/Rust merge step)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, indent=1)
    return out_path


def load_sections(json_path: str) -> list[Section]:
    """Load the sections from a report JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [section_from_dict(d) for d in data.get("sections", [])]
