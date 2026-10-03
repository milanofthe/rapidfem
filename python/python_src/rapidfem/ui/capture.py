# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Capture-slot for rapidfem.show().

When the UI server runs user code, it calls `start_capture()` first, exec's
the user script, and then collects whatever was passed to `rapidfem.show()`
via `get_captured()`.

Outside UI mode the capture slot is inactive and `show()` is a print-only
no-op, so scripts behave the same on the command line and in the UI.
"""
from __future__ import annotations

import threading
from typing import Any, NamedTuple


class CapturedItem(NamedTuple):
    name: str
    obj: Any
    kind: str  # what classify() returns: "geometry" | "simulation" | "result"
               # | "eigenmode" | "eigenmodes" | "td_timeseries" | "td_transfer"
               # | "td_trajectory" | "unknown"


_state = threading.local()


def start_capture(on_item=None, sweep_cb=None) -> None:
    """Begin capturing rapidfem.show() calls.

    ``on_item`` (optional) is a callback invoked with each
    :class:`CapturedItem` the moment it is captured, letting the UI worker
    stream a display the instant ``show()`` runs instead of waiting for the
    whole cell to finish. The item is still appended to the batch list, so
    :func:`stop_capture` returns the full set as before (the worker uses it
    for the deferred sim+result pairing). The callback must be self-contained
    (never raise) so it cannot break user code.

    ``sweep_cb`` (optional) is a per-frequency callback ``(freq_idx, freq_hz,
    s_matrix)`` that :meth:`ProblemFD.sweep` forwards to the native solver so
    the UI can stream partial S-parameters during a sweep. Retrieved via
    :func:`active_sweep_callback`. Must also be self-contained.
    """
    _state.active = True
    _state.items = []
    _state.on_item = on_item
    _state.sweep_cb = sweep_cb


def stop_capture() -> list[CapturedItem]:
    items: list[CapturedItem] = list(getattr(_state, "items", []))
    _state.active = False
    _state.items = []
    _state.on_item = None
    _state.sweep_cb = None
    return items


def active_sweep_callback():
    """Return the registered per-frequency sweep callback while capturing, else
    None. ``ProblemFD.sweep`` uses this to stream partial results to the UI
    without the user wiring anything up."""
    if not is_capturing():
        return None
    return getattr(_state, "sweep_cb", None)


def get_captured() -> list[CapturedItem]:
    return list(getattr(_state, "items", []))


def is_capturing() -> bool:
    return bool(getattr(_state, "active", False))


def capture(name: str, obj: Any, kind: str) -> None:
    if not is_capturing():
        return
    item = CapturedItem(name=name, obj=obj, kind=kind)
    _state.items.append(item)
    on_item = getattr(_state, "on_item", None)
    if on_item is not None:
        on_item(item)


def classify(obj: Any) -> str:
    cls = type(obj).__name__
    mod = getattr(type(obj), "__module__", "") or ""
    if cls == "Geometry" and mod.startswith("rapidfem"):
        return "geometry"
    # The time-domain `ProblemTD` is not a UI "simulation": its results
    # render through the td_* wrappers instead.
    if cls == "ProblemFD" and mod.startswith("rapidfem"):
        return "simulation"
    if cls == "Simulation":
        return "simulation"
    if cls == "SweepResult":
        return "result"
    # Time-domain result wrappers (rapidfem.problem.td), thin objects the
    # ProblemTD verbs hand back so show() can route them to a UI panel.
    if mod.startswith("rapidfem"):
        if cls == "TdResponse":
            return "td_timeseries"
        if cls == "TdTransfer":
            return "td_transfer"
        if cls == "TdTrajectory":
            return "td_trajectory"
    if cls == "Eigenmode":
        return "eigenmode"
    # `run_eigenmode()` returns a list, accept that as the typical user
    # show() target so they don't have to unpack per-mode.
    if isinstance(obj, list) and obj and type(obj[0]).__name__ == "Eigenmode":
        return "eigenmodes"
    return "unknown"
