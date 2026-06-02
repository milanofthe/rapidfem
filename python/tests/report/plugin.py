"""pytest plugin: collect outcomes + ``report``-fixture artefacts and emit
the rapidfem HTML report.

Wired in from ``python/tests/conftest.py``. Enable with ``--report``;
control the output paths with ``--report-html`` / ``--report-json``.
Every collected test gets a section whether or not it used the fixture, so
the report is a faithful picture of the run.
"""
from __future__ import annotations

import datetime
import platform
import subprocess

import pytest

from . import render
from .model import Report, Section
from .recorder import Recorder

# Severity rank: a worse phase outcome wins when reducing across phases.
_RANK = {"passed": 0, "xpassed": 1, "skipped": 2, "xfailed": 3,
         "failed": 4, "error": 5}


def pytest_addoption(parser):
    grp = parser.getgroup("rapidfem-report")
    grp.addoption("--report", action="store_true", default=False,
                  help="emit the rapidfem HTML test report")
    grp.addoption("--report-html", default="_report/rapidfem-test-report.html",
                  help="output path for the HTML report")
    grp.addoption("--report-json", default="_report/rapidfem-test-report.json",
                  help="output path for the report JSON (Python/Rust merge)")
    grp.addoption("--report-title", default="rapidfem test report",
                  help="report title")
    grp.addoption("--report-merge", default=None,
                  help="path to a Rust-bridge JSON whose sections are merged in")


def _git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "?"
    except Exception:
        return "?"


def _rapidfem_version() -> str:
    try:
        import rapidfem

        return getattr(rapidfem, "__version__", "?")
    except Exception:
        return "?"


def pytest_configure(config):
    if not config.getoption("--report"):
        return
    report = Report(title=config.getoption("--report-title"))
    report.subtitle = "DGTD / Nédélec-FEM validation suite"
    report.meta = {
        "branch": _git_branch(),
        "rapidfem": _rapidfem_version(),
        "python": platform.python_version(),
        "platform": platform.system(),
    }
    config._rapidfem_report = report
    # nodeid -> Section, shared by the fixture and the outcome hook.
    config._rapidfem_sections = {}


def _get_section(config, item_or_nodeid, name="", group="", doc="") -> Section:
    sections = config._rapidfem_sections
    if hasattr(item_or_nodeid, "nodeid"):
        nodeid = item_or_nodeid.nodeid
    else:
        nodeid = item_or_nodeid
    sec = sections.get(nodeid)
    if sec is None:
        sec = Section(nodeid=nodeid, name=name or nodeid, group=group or "tests")
        sec.doc = doc
        sections[nodeid] = sec
        config._rapidfem_report.sections.append(sec)
    return sec


def _meta_from_item(item):
    name = item.name
    group = item.module.__name__.split(".")[-1] if item.module else "tests"
    doc = ""
    func = getattr(item, "obj", None)
    if func is not None and func.__doc__:
        doc = " ".join(line.strip() for line in func.__doc__.strip().splitlines())
    return name, group, doc


@pytest.fixture
def report(request):
    """Per-test recorder. Attach metrics / tables / plots; they land in the
    test's section of the rapidfem HTML report."""
    config = request.config
    if not config.getoption("--report"):
        # Reporting off: hand back a recorder onto a throwaway section so
        # tests calling report.* stay valid and cheap.
        return Recorder(Section(nodeid="(disabled)", name="(disabled)",
                                group="(disabled)"))
    name, group, doc = _meta_from_item(request.node)
    sec = _get_section(config, request.node, name=name, group=group, doc=doc)
    return Recorder(sec)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    config = item.config
    if not getattr(config, "_rapidfem_report", None):
        return

    name, group, doc = _meta_from_item(item)
    sec = _get_section(config, item, name=name, group=group, doc=doc)
    if not sec.doc:
        sec.doc = doc
    sec.duration += getattr(rep, "duration", 0.0)

    # Reduce this phase's outcome into the section status.
    if rep.skipped:
        status = "xfailed" if getattr(rep, "wasxfail", False) else "skipped"
    elif rep.failed:
        status = "error" if rep.when in ("setup", "teardown") else "failed"
    else:
        status = "xpassed" if getattr(rep, "wasxfail", False) else "passed"
    if _RANK[status] >= _RANK.get(sec.status, 0):
        sec.status = status

    if rep.failed and rep.longrepr is not None:
        sec.message = str(rep.longrepr)
    elif rep.skipped and isinstance(rep.longrepr, tuple):
        # (path, lineno, reason)
        sec.message = rep.longrepr[2]


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    report = getattr(config, "_rapidfem_report", None)
    if report is None:
        return

    # Optionally merge a Rust-bridge JSON produced out of band.
    merge_path = config.getoption("--report-merge")
    if merge_path:
        try:
            report.sections.extend(render.load_sections(merge_path))
        except OSError:
            pass

    report.generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = config.getoption("--report-html")
    js = config.getoption("--report-json")
    render.dump_json(report, js)
    render.render_html(report, html, producer="pytest")
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"rapidfem report → {html}", bold=True)
