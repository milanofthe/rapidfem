"""rapidfem test-report harness.

A pytest plugin (:mod:`report.plugin`) collects every test's outcome plus
any artefacts it attaches through the ``report`` fixture
(:class:`report.recorder.Recorder`), and renders one self-contained HTML
file in the rapidfem style (:mod:`report.render`). A Rust bridge
(:mod:`report.rust_bridge`) feeds ``cargo test`` gates into the same model
so Python and Rust land in a single report.
"""
from .model import Report, Section
from .recorder import Recorder

__all__ = ["Report", "Section", "Recorder"]
