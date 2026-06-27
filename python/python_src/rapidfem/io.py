"""
Optional Python-side exporters for `SweepResult`. Each function lazily imports
its backend dep so users without it still get `import rapidfem` for free.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rapidfem import SweepResult


def renormalize_sparams(sparams, z_old, z_new):
    """Renormalize an S-matrix from per-port reference impedances to a new one.

    Modal ports report S-parameters referenced to their own (frequency-
    dependent) modal impedance, so a matched modal line reads |S11|≈0 against
    that self-reference regardless of its characteristic impedance. This
    re-references the S-matrix to a fixed `z_new` (e.g. 50 Ω) via the standard
    transform S → normalized-Z → S', so mismatches against the chosen reference
    appear.

    Parameters
    ----------
    sparams : array (n_freq, n, n) complex
        S-matrix referenced to `z_old`.
    z_old : array (n_freq, n)
        Per-port, per-frequency reference impedances (e.g. the ports' modal
        ``z_mode(f)``).
    z_new : float or array (n,)
        Target reference impedance(s).

    Returns
    -------
    array (n_freq, n, n) complex
        S-matrix renormalized to `z_new`. With ``z_new == z_old`` this is the
        identity; lumped ports (already at a fixed z0) are a no-op.
    """
    import numpy as np

    s = np.asarray(sparams, dtype=complex)
    nf, n, _ = s.shape
    z_old = np.asarray(z_old, dtype=complex)
    z_new = np.broadcast_to(np.asarray(z_new, dtype=complex), (n,))
    eye = np.eye(n)
    sq_new = np.sqrt(z_new)
    out = np.empty_like(s)
    for fi in range(nf):
        sq_old = np.sqrt(z_old[fi])
        # S (ref z_old) -> normalized impedance  zbar = (I+S)(I-S)^-1
        zbar = (eye + s[fi]) @ np.linalg.inv(eye - s[fi])
        # de-normalize to physical Z, then re-normalize to z_new
        z_phys = (sq_old[:, None] * zbar) * sq_old[None, :]
        zbar_new = (z_phys / sq_new[:, None]) / sq_new[None, :]
        # normalized-Z -> S' (ref z_new)
        out[fi] = (zbar_new - eye) @ np.linalg.inv(zbar_new + eye)
    return out


def to_network(result: "SweepResult", z0: float = 50.0, name: str | None = None) -> Any:
    """Convert a SweepResult to a `skrf.Network` for downstream RF analysis.

    Unlocks the scikit-rf ecosystem: Smith charts, plot_s_db, network cascading,
    de-embedding, time-domain conversion, etc. Requires ``pip install scikit-rf``.
    """
    try:
        import skrf
    except ImportError as e:
        raise ImportError(
            "scikit-rf is required for to_network(). Install with: pip install scikit-rf"
        ) from e
    freq = skrf.Frequency.from_f(result.frequencies, unit="Hz")
    return skrf.Network(frequency=freq, s=result.sparams, z0=z0, name=name)


def to_touchstone(
    result: "SweepResult",
    path: str,
    z0: float = 50.0,
    name: str | None = None,
    fmt: str = "ri",
) -> None:
    """Write a SweepResult to a Touchstone file (.s2p / .s4p / .snp).

    If scikit-rf is installed, delegates to its writer (richer formatting).
    Otherwise emits RI-format text directly using only numpy.
    """
    try:
        import skrf  # noqa: F401
        nw = to_network(result, z0=z0, name=name)
        nw.write_touchstone(path)
        return
    except ImportError:
        pass

    import numpy as np
    n_freq = len(result.frequencies)
    n = result.n_driven
    fmt_map = {"ri": ("RI", lambda c: (c.real, c.imag)),
               "ma": ("MA", lambda c: (abs(c), np.angle(c, deg=True))),
               "db": ("DB", lambda c: (20 * np.log10(max(abs(c), 1e-30)), np.angle(c, deg=True)))}
    if fmt.lower() not in fmt_map:
        raise ValueError(f"fmt must be 'ri', 'ma', or 'db', got {fmt!r}")
    fmt_label, conv = fmt_map[fmt.lower()]
    s = result.sparams
    with open(path, "w") as f:
        f.write(f"! Touchstone file written by rapidfem (Python)\n")
        f.write(f"! n_ports = {n}, n_freq = {n_freq}\n")
        f.write(f"# HZ S {fmt_label} R {z0:g}\n")
        for k in range(n_freq):
            row = [f"{result.frequencies[k]:.6e}"]
            # Touchstone S2P+ ordering: row-major over (i, j)
            for i in range(n):
                for j in range(n):
                    a, b = conv(s[k, i, j])
                    row.append(f"{a:.6e}")
                    row.append(f"{b:.6e}")
            f.write(" ".join(row) + "\n")


def to_pyvista(simulation: Any, result: "SweepResult", freq_idx: int = 0, port_idx: int = 0) -> Any:
    """Build a `pyvista.UnstructuredGrid` for one (freq, port) combination.

    Per-node E-field is attached as ``E_real``, ``E_imag``, ``E_magnitude`` arrays.
    Use the returned grid for ``.plot()``, ``.save("field.vtu")``, slicing, etc.
    Requires ``pip install pyvista``.
    """
    try:
        import numpy as np
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            "pyvista is required for to_pyvista(). Install with: pip install pyvista"
        ) from e

    nodes = simulation.mesh_nodes                          # (n_nodes, 3) float64
    tets = simulation.mesh_tets                            # (n_tets, 4) int64
    field = simulation.field_at_nodes(result, freq_idx, port_idx)
    if field is None:
        raise ValueError(f"field_at_nodes returned None for freq_idx={freq_idx}, port_idx={port_idx}")

    # pyvista cell-array format: [4, v0, v1, v2, v3, 4, v0, v1, v2, v3, ...]
    n_tets = tets.shape[0]
    cells = np.empty(n_tets * 5, dtype=np.int64)
    cells[0::5] = 4
    cells[1::5] = tets[:, 0]
    cells[2::5] = tets[:, 1]
    cells[3::5] = tets[:, 2]
    cells[4::5] = tets[:, 3]
    cell_types = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    grid["E_real"] = field.real.astype(np.float64)
    grid["E_imag"] = field.imag.astype(np.float64)
    grid["E_magnitude"] = np.linalg.norm(np.abs(field), axis=1)
    return grid


def to_hdf5(result: "SweepResult", path: str, group: str = "/") -> None:
    """Save a SweepResult to HDF5: frequencies, S-params, metadata."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for to_hdf5(). Install with: pip install h5py"
        ) from e
    with h5py.File(path, "w") as f:
        g = f.require_group(group)
        g.create_dataset("frequencies_hz", data=result.frequencies)
        g.create_dataset("sparams", data=result.sparams)
        g.attrs["n_driven"] = result.n_driven
        g.attrs["solve_time_s"] = result.solve_time_s
        g.attrs["units"] = "Hz; S complex; reference impedance external"


# ── Bind these as methods on SweepResult so users can write `result.to_*` ──
def _bind_methods() -> None:
    try:
        from rapidfem._native import SweepResult
        # PyO3 classes accept attribute assignment in pyo3 0.22+
        SweepResult.to_network = to_network            # type: ignore[attr-defined]
        SweepResult.to_touchstone = to_touchstone      # type: ignore[attr-defined]
        SweepResult.to_hdf5 = to_hdf5                  # type: ignore[attr-defined]
    except (ImportError, AttributeError, TypeError):
        # If PyO3 classes don't accept setattr in this pyo3 version, users still
        # get the free functions via `rapidfem.io.*`.
        pass


_bind_methods()


__all__ = ["to_network", "to_touchstone", "to_hdf5", "to_pyvista"]
