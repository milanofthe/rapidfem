# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Binary packing for display-event payloads.

A baked example, or a live display event, carries bulk numeric arrays:
mesh nodes / tris / tets, the geometry-preview triangulation, frequency-
domain field data, time-domain trajectory frames. Serialised as JSON
those are millions of numbers as text.

:func:`pack` walks a list of display events and lifts every bulk array
into one of **two** byte buffers, ``geo`` (mesh / geometry, needed as
soon as the 3-D view opens) and ``field`` (field and trajectory data,
fetched only when the field viewer is shown), replacing each array with
a compact ``$bin`` reference. The JSON that remains is pure structure.

The static-demo bake (``scripts/bake_demo.py``) writes the returned
buffers as sidecar files next to the JSON.

``$bin`` reference shapes
-------------------------
- plain array:   ``{"$bin": <buf>, "dtype": <dt>, "off": <byte>, "n": <count>}``
- field block:   ``{"$bin": "field", "kind": "fields", "dtype": "f32",
                     "off", "n", "n_freq", "n_port", "stride", "mask": [...]}``
- frame block:   ``{"$bin": "field", "kind": "frames", "dtype": "u16",
                     "off", "n", "n_snap", "n_points"}``
                  (``n_points`` is the per-frame row length, for a
                  trajectory it is the unique-node count ``n_node``)

``buf`` is ``"geo"`` or ``"field"``; ``off`` is a byte offset, 4-byte
aligned so a typed-array view can be taken directly.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Format magic + version, written into the buffer header so a reader can
# reject a stale or mismatched blob loudly.
BIN_MAGIC = 0x52464250  # "RFBP"
BIN_VERSION = 2

_NP_DTYPE = {
    "f32": np.float32,
    "i32": np.int32,
    "u16": np.uint16,
    "u8": np.uint8,
}


class _Buffer:
    """A growable little-endian byte buffer; every entry is 4-byte aligned
    so the frontend can take a `Float32Array`/`Int32Array`/`Uint16Array`
    view straight onto it."""

    def __init__(self) -> None:
        # 8-byte header: magic + version, so a slice view still starts aligned.
        self.data = bytearray()
        self.data.extend(int(BIN_MAGIC).to_bytes(4, "little"))
        self.data.extend(int(BIN_VERSION).to_bytes(4, "little"))

    def put(self, values: Any, dtype: str) -> tuple[int, int]:
        """Append `values` as `dtype`; return ``(byte_offset, count)``."""
        pad = (-len(self.data)) % 4
        if pad:
            self.data.extend(b"\x00" * pad)
        off = len(self.data)
        arr = np.ascontiguousarray(values, dtype=_NP_DTYPE[dtype])
        self.data.extend(arr.tobytes())
        return off, int(arr.size)

    def bytes(self) -> bytes:
        return bytes(self.data)


def _ref(buf: str, dtype: str, off: int, n: int) -> dict[str, Any]:
    return {"$bin": buf, "dtype": dtype, "off": off, "n": n}


def _pack_array(buffer: _Buffer, buf_name: str, values: Any, dtype: str) -> dict:
    off, n = buffer.put(values, dtype)
    return _ref(buf_name, dtype, off, n)


def _pack_mesh(geo: _Buffer, payload: dict) -> None:
    """Mesh payload, node coordinates and tet/tri connectivity."""
    if isinstance(payload.get("nodes"), list):
        payload["nodes"] = _pack_array(geo, "geo", payload["nodes"], "f32")
    for key in ("tris", "tri_phys", "tets", "tet_phys"):
        v = payload.get(key)
        if isinstance(v, list) and v:
            payload[key] = _pack_array(geo, "geo", v, "i32")


def _pack_geometry(geo: _Buffer, payload: dict) -> None:
    """Geometry-preview payload, per-entity triangulation / wireframe."""
    for ent in payload.get("entities", []):
        if not isinstance(ent, dict):
            continue
        for key in ("positions", "normals"):
            v = ent.get(key)
            if isinstance(v, list) and v:
                ent[key] = _pack_array(geo, "geo", v, "f32")


def _pack_fields(field: _Buffer, fields: Any) -> Any:
    """A ``[n_freq][n_port]`` nest of per-node arrays (some entries
    ``None``). The present payloads are concatenated into the field
    buffer; the small presence mask stays inline."""
    if not isinstance(fields, list) or not fields:
        return fields
    n_freq = len(fields)
    if not isinstance(fields[0], list):
        return fields
    n_port = len(fields[0])
    mask: list[int] = []
    flat: list[float] = []
    stride: int | None = None
    for row in fields:
        for x in row:
            if x is None:
                mask.append(0)
            else:
                mask.append(1)
                if stride is None:
                    stride = len(x)
                flat.extend(x)
    if stride is None:
        return fields  # all-null, nothing to pack, leave inline
    off, n = field.put(flat, "f32")
    return {
        "$bin": "field", "kind": "fields", "dtype": "f32",
        "off": off, "n": n, "n_freq": n_freq, "n_port": n_port,
        "stride": stride, "mask": mask,
    }


def _pack_trajectory(field: _Buffer, payload: dict) -> None:
    """Time-domain trajectory, the self-contained DG-corner mesh
    (``nodes`` / ``tets``) and its per-node per-frame quantised
    magnitudes. The mesh is small but logically geo-like; it rides in the
    field buffer so the whole trajectory is one unit fetched together."""
    if isinstance(payload.get("nodes"), list):
        payload["nodes"] = _pack_array(field, "field", payload["nodes"], "f32")
    if isinstance(payload.get("tets"), list):
        payload["tets"] = _pack_array(field, "field", payload["tets"], "i32")
    for key in ("frames_e", "frames_h"):
        frames = payload.get(key)
        if isinstance(frames, list) and frames and isinstance(frames[0], list):
            n_snap = len(frames)
            n_points = len(frames[0])
            flat = [v for row in frames for v in row]
            off, n = field.put(flat, "u16")
            payload[key] = {
                "$bin": "field", "kind": "frames", "dtype": "u16",
                "off": off, "n": n, "n_snap": n_snap, "n_points": n_points,
            }


def pack(events: list[dict]) -> tuple[bytes, bytes]:
    """Lift the bulk arrays out of `events` into two binary buffers.

    Mutates each event's ``payload`` in place, bulk arrays become
    ``$bin`` references. Returns ``(geo_bytes, field_bytes)``; either may
    be header-only (16 bytes) when an event set carries no such data.
    """
    geo = _Buffer()
    field = _Buffer()
    for ev in events:
        payload = ev.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = ev.get("kind")
        if kind == "mesh":
            _pack_mesh(geo, payload)
        elif kind == "geometry":
            _pack_geometry(geo, payload)
        elif kind == "result":
            for key in ("fields", "fields_j", "fields_h"):
                f = payload.get(key)
                if f is not None:
                    payload[key] = _pack_fields(field, f)
        elif kind == "td_trajectory":
            _pack_trajectory(field, payload)
    return geo.bytes(), field.bytes()
