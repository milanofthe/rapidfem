"""rapidfem report design tokens and matplotlib styling.

A single source of truth for the colours / fonts the HTML report and its
plots share, mirrored from the frontend theme
(``ui/frontend-src/src/lib/docs/docs.css``). The HTML renderer reads
:data:`TOKENS` to build the inline ``:root`` CSS; the plot helpers call
:func:`use_style` so every figure matches.
"""
from __future__ import annotations

import os

import matplotlib

# Headless: the report is generated in batch, never shown interactively.
matplotlib.use("Agg")

_MPLSTYLE = os.path.join(os.path.dirname(__file__), "rapidfem.mplstyle")

# Design tokens — the exact frontend values. Hex without the leading '#'
# for the few places matplotlib wants it bare; the '#'-prefixed forms are
# what the CSS uses.
TOKENS = {
    "surface": "#1c1c21",
    "surface_raised": "#232329",
    "surface_panel": "#2a2a31",
    "surface_inset": "#131316",
    "text": "#e2ddd5",
    "text_muted": "#9a96a0",
    "text_disabled": "#8a8790",
    "border": "#35353d",
    "border_subtle": "#2d2d34",
    "accent": "#d9513c",        # lava red — the rapidfem brand colour
    "accent_hover": "#e5634f",
    "accent_secondary": "#e8944a",  # warm orange
    "accent_purple": "#a78bd9",
    "blue": "#4a90d9",          # contrast series (e.g. FD vs TD)
    "success": "#6bbf8a",
    "warning": "#e8944a",
    "error": "#d9513c",
    "font_body": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    "font_mono": "'JetBrains Mono', ui-monospace, monospace",
}

# Plot series colours in cycle order — the same list the mplstyle uses,
# exposed so plot helpers can pick a specific colour by name.
SERIES = {
    "accent": TOKENS["accent"],
    "blue": TOKENS["blue"],
    "success": TOKENS["success"],
    "orange": TOKENS["accent_secondary"],
    "purple": TOKENS["accent_purple"],
    "text": TOKENS["text"],
}

_STYLE_APPLIED = False


def use_style() -> None:
    """Apply the rapidfem matplotlib style (idempotent)."""
    global _STYLE_APPLIED
    if not _STYLE_APPLIED:
        import matplotlib.pyplot as plt

        plt.style.use(_MPLSTYLE)
        _STYLE_APPLIED = True
