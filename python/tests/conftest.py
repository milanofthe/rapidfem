"""pytest configuration for the rapidfem Python test suite.

Marker convention:
- `slow`: runs a full transient or sweep. Default opt-out via
  `pytest -m "not slow"`; explicit opt-in via `pytest -m slow`.

HTML report: pass `--report` to emit a self-contained rapidfem-styled HTML
report (with plots) of the run. Tests opt into rich content via the
`report` fixture (see `tests/report/recorder.py`). The report plugin is
registered below.
"""
import os
import sys

# Make the local `report` package importable regardless of invocation dir,
# so `pytest_plugins` can load the report plugin.
sys.path.insert(0, os.path.dirname(__file__))

pytest_plugins = ("report.plugin",)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that run a full TD transient or FD sweep (minutes)",
    )
