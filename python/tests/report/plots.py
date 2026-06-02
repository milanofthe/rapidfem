"""Plot helpers that return inline-SVG strings in the rapidfem style.

Every helper applies :func:`style.use_style`, draws onto an Agg figure and
returns a self-contained ``<svg>`` string (no XML prolog) ready to embed in
the report HTML. Figures are always closed, so a long test run leaks no
canvases.
"""
from __future__ import annotations

import io
import re

import numpy as np

from . import style

_SVG_PROLOG = re.compile(r"^<\?xml[^>]*\?>\s*<!DOCTYPE[^>]*>\s*", re.DOTALL)


def fig_to_svg(fig, *, close: bool = True) -> str:
    """Serialise a matplotlib figure to a bare inline ``<svg>`` string."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    svg = buf.getvalue()
    # Drop the XML prolog / doctype so the SVG embeds cleanly inline.
    svg = _SVG_PROLOG.sub("", svg)
    return svg.strip()


def _new_axes(figsize=None):
    style.use_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _to_db(mag: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
    mag = np.abs(np.asarray(mag))
    out = 20.0 * np.log10(np.maximum(mag, 10.0 ** (floor_db / 20.0)))
    return out


def _entry_curve(arr: np.ndarray, entry):
    """Pull the (i,j) 1-based entry out of an S array that is either
    ``(nfreq,)`` or ``(nfreq, N, N)``."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    i, j = entry
    return arr[:, i - 1, j - 1]


def plot_sparams(
    freqs_hz,
    series: dict,
    *,
    entries=None,
    db: bool = True,
    title: str = "",
    figsize=(7.2, 4.0),
) -> str:
    """S-parameter magnitude over frequency.

    Parameters
    ----------
    freqs_hz : array_like
        Frequency axis in Hz (plotted in GHz).
    series : dict[str, ndarray]
        Label → complex S array, each ``(nfreq,)`` or ``(nfreq, N, N)``.
    entries : list[tuple[int, int]] | None
        1-based ``(i, j)`` port pairs to draw from N×N arrays. Defaults to
        ``[(1, 1), (2, 1)]`` clipped to the matrix size; ignored for 1-D
        arrays.
    db : bool
        dB magnitude if True, else linear.
    """
    fig, ax = _new_axes(figsize)
    f_ghz = np.asarray(freqs_hz, dtype=float) / 1e9
    # Solid for the first series, dashed for the second, etc. — keeps a
    # TD-vs-FD overlay legible in one colour family per entry.
    dashes = ["-", "--", ":", "-."]
    for s_idx, (label, arr) in enumerate(series.items()):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            ent_list = [None]
        else:
            n = arr.shape[-1]
            ent_list = entries or [(1, 1), (2, 1)]
            ent_list = [(i, j) for (i, j) in ent_list if i <= n and j <= n]
        for e_idx, entry in enumerate(ent_list):
            y = _entry_curve(arr, entry)
            y = _to_db(y) if db else np.abs(y)
            color = list(style.SERIES.values())[e_idx % len(style.SERIES)]
            name = label if entry is None else f"{label} S{entry[0]}{entry[1]}"
            ax.plot(f_ghz, y, dashes[s_idx % len(dashes)], color=color,
                    label=name)
    ax.set_xlabel("frequency  [GHz]")
    ax.set_ylabel("|S|  [dB]" if db else "|S|")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    ax.margins(x=0.01)
    return fig_to_svg(fig)


def plot_xy(
    x,
    series: dict,
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    logx: bool = False,
    logy: bool = False,
    markers: bool = False,
    figsize=(7.2, 4.0),
) -> str:
    """Generic line plot — energy decay, convergence, |S| linear, etc.

    ``series`` is a dict of label → y (each the same length as ``x``).
    """
    fig, ax = _new_axes(figsize)
    x = np.asarray(x, dtype=float)
    style_kw = {"marker": "o"} if markers else {}
    for label, y in series.items():
        ax.plot(x, np.asarray(y, dtype=float), label=label, **style_kw)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if series:
        ax.legend(loc="best")
    return fig_to_svg(fig)


def plot_complex_plane(
    points: dict,
    *,
    title: str = "",
    xlabel: str = "Re",
    ylabel: str = "Im",
    figsize=(5.2, 5.0),
) -> str:
    """Scatter in the complex plane — state-space eigenvalues / poles.

    ``points`` is a dict of label → complex array.
    """
    fig, ax = _new_axes(figsize)
    for label, z in points.items():
        z = np.asarray(z, dtype=complex)
        ax.scatter(z.real, z.imag, label=label, s=22, alpha=0.85)
    ax.axhline(0.0, color=style.TOKENS["border"], lw=0.8, zorder=0)
    ax.axvline(0.0, color=style.TOKENS["border"], lw=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    return fig_to_svg(fig)


def plot_passivity(
    freqs_hz,
    sigma_max,
    *,
    title: str = "passivity  σ_max(S)",
    figsize=(7.2, 3.4),
) -> str:
    """Largest singular value of S over frequency, with the unit-circle
    passivity bound marked."""
    fig, ax = _new_axes(figsize)
    f_ghz = np.asarray(freqs_hz, dtype=float) / 1e9
    ax.plot(f_ghz, np.asarray(sigma_max, dtype=float),
            color=style.SERIES["accent"], label="σ_max(S)")
    ax.axhline(1.0, color=style.SERIES["success"], ls="--", lw=1.2,
               label="passive bound")
    ax.set_xlabel("frequency  [GHz]")
    ax.set_ylabel("σ_max(S)")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig_to_svg(fig)
