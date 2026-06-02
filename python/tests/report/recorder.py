"""The ``report`` fixture API.

A test receives a :class:`Recorder` bound to its own report section and
attaches artefacts as it runs::

    def test_wr90_sparams(report):
        report.note("WR-90 guide, TD vs FD over the 9-12 GHz band.")
        report.metric("|S21| max dev", dev, ref=0.0, tol=0.04)
        report.plot_sparams(freqs, {"TD": s_td, "FD": s_fd},
                            entries=[(1, 1), (2, 1)],
                            caption="solid = TD, dashed = FD")

The pytest plugin owns the :class:`~report.model.Section`; the recorder
only appends to ``section.items`` so the plugin can stamp the outcome
afterwards.
"""
from __future__ import annotations

import numpy as np

from . import plots
from .model import Log, Metric, Note, Plot, Section, Table


class Recorder:
    """Per-test handle that accumulates report items onto a section."""

    def __init__(self, section: Section):
        self._section = section

    # -- text / tabular ----------------------------------------------------
    def note(self, text: str) -> None:
        """Attach a short prose note (rendered as paragraphs)."""
        self._section.items.append(Note(text=str(text).strip()))

    def table(self, title: str, headers, rows) -> None:
        """Attach a table. ``rows`` is a sequence of row sequences;
        numbers are formatted by the renderer."""
        self._section.items.append(
            Table(
                title=str(title),
                headers=[str(h) for h in headers],
                rows=[list(r) for r in rows],
            )
        )

    def log(self, title: str, text: str) -> None:
        """Attach a monospaced log/console block."""
        self._section.items.append(Log(title=str(title), text=str(text)))

    # -- metrics -----------------------------------------------------------
    def metric(
        self,
        label: str,
        value,
        *,
        ref=None,
        tol=None,
        atol=None,
        bound=None,
        lower=None,
        unit: str = "",
        detail: str = "",
        passed: bool | None = None,
    ) -> Metric:
        """Record a checked or informational number.

        Pick the check that matches the assertion:

        * ``ref`` + ``tol`` (and/or ``atol``) — *approximately equal*:
          pass if ``|value-ref| <= atol + tol*|ref|``. If ``ref`` is 0 and
          only ``tol`` is given, ``tol`` is treated as an absolute bound
          (relative tolerance is meaningless at zero).
        * ``bound`` — *upper bound* ``value <= bound`` (e.g. a max error,
          ``|S| <= 1.15``).
        * ``lower`` — *lower bound* ``value >= lower`` (e.g.
          ``|S21| >= 0.9``).
        * none of the above — informational (no verdict).

        ``passed`` forces the verdict. Returns the :class:`Metric` so a
        test can read ``.passed``.
        """
        v = float(value) if _isnum(value) else value
        cmp = "info"
        ref_out = None
        tol_out = None
        if passed is None:
            if bound is not None:
                cmp, ref_out = "le", float(bound)
                passed = float(v) <= float(bound)
            elif lower is not None:
                cmp, ref_out = "ge", float(lower)
                passed = float(v) >= float(lower)
            elif ref is not None:
                cmp, ref_out, tol_out = "approx", float(ref), tol
                rt = float(tol) if tol is not None else 0.0
                at = float(atol) if atol is not None else 0.0
                thresh = at + rt * abs(float(ref))
                if thresh == 0.0 and tol is not None:
                    thresh = float(tol)   # ref==0 → tol is absolute
                passed = abs(float(v) - float(ref)) <= thresh
        else:
            if bound is not None:
                cmp, ref_out = "le", float(bound)
            elif lower is not None:
                cmp, ref_out = "ge", float(lower)
            elif ref is not None:
                cmp, ref_out, tol_out = "approx", float(ref), tol

        m = Metric(
            label=str(label),
            value=float(v) if _isnum(v) else str(v),
            ref=ref_out,
            tol=None if tol_out is None else float(tol_out),
            cmp=cmp,
            unit=str(unit),
            passed=passed,
            detail=str(detail),
        )
        self._section.items.append(m)
        return m

    # -- plots -------------------------------------------------------------
    def figure(self, fig, *, title: str = "", caption: str = "") -> None:
        """Attach a raw matplotlib figure (serialised to inline SVG and
        closed)."""
        self._section.items.append(
            Plot(title=str(title), svg=plots.fig_to_svg(fig),
                 caption=str(caption))
        )

    def plot_sparams(
        self,
        freqs_hz,
        series: dict,
        *,
        entries=None,
        db: bool = True,
        title: str = "S-parameters",
        caption: str = "",
    ) -> None:
        svg = plots.plot_sparams(
            freqs_hz, series, entries=entries, db=db, title=title
        )
        self._section.items.append(Plot(title=title, svg=svg, caption=caption))

    def plot_xy(
        self,
        x,
        series: dict,
        *,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        logx: bool = False,
        logy: bool = False,
        markers: bool = False,
        caption: str = "",
    ) -> None:
        svg = plots.plot_xy(
            x, series, xlabel=xlabel, ylabel=ylabel, title=title,
            logx=logx, logy=logy, markers=markers,
        )
        self._section.items.append(Plot(title=title, svg=svg, caption=caption))

    def plot_complex_plane(
        self,
        points: dict,
        *,
        title: str = "",
        xlabel: str = "Re",
        ylabel: str = "Im",
        caption: str = "",
    ) -> None:
        svg = plots.plot_complex_plane(
            points, title=title, xlabel=xlabel, ylabel=ylabel
        )
        self._section.items.append(Plot(title=title, svg=svg, caption=caption))

    def plot_passivity(
        self, freqs_hz, sigma_max, *, title: str = "passivity  σ_max(S)",
        caption: str = "",
    ) -> None:
        svg = plots.plot_passivity(freqs_hz, sigma_max, title=title)
        self._section.items.append(Plot(title=title, svg=svg, caption=caption))


def _isnum(v) -> bool:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
