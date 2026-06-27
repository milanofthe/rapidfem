# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Emit the Rust golden test for the material-tensor closed forms.

Two analytic golden groups, both pinned to the EXACT formulas the Rust kernel
implements in `rapidfem_core::materials` (e^{+jωt} time convention, hence
non-positive imaginary parts for loss):

  (A) Uniaxial PML coordinate stretch — `PmlRegion::stretch`:
        axis, sign  <- direction (largest |component| wins, sign = signum)
        u = max(0, sign·(coord_axis - inner_face) / thickness)
        s_d = 1 - j·(u^n · δmax),   s = 1 in the two orthogonal directions
      u is clamped to 0 OUTSIDE the layer, so a point before the inner face
      gives s = (1, 1, 1).

  (B) Lossy-dielectric complex permittivity — `build_material_tensors`:
        εr*(ω) = εr·(1 - j·tanδ) - j·σ/(ω·ε₀),   ω = 2π f

Both forms are algebraic, so the ground truth is exact (no numerics). We mirror
the SAME ε₀ constant the Rust uses so the conductivity term matches to f64.
"""
from __future__ import annotations

import math
import os

# Mirror the EXACT ε₀ used by rapidfem-core (crates/rapidfem-core/src/constants.rs)
# so the σ/(ω·ε₀) term agrees to f64 — do NOT substitute a CODATA value here.
EPS0 = 8.854187818814e-12

TWO_PI = 2.0 * math.pi


def f64(v: float) -> str:
    return f"{float(v):.17e}"


# ---------------------------------------------------------------------------
# Group A: PML stretch
# ---------------------------------------------------------------------------

# (name, direction, inner_face, thickness, exponent, delta_max, sample_points)
# sample_points are (x, y, z); chosen to exercise: a point BEFORE the inner
# face (u clamped -> s = 1), a point exactly AT the inner face (u = 0), and
# several depths into the layer (u in (0, 1]) plus past the outer face (u > 1).
PML_CASES = [
    (
        "plus_x", [1.0, 0.0, 0.0], 0.040, 0.010, 2.0, 8.0,
        [
            (0.030, 0.5, -0.3),   # before inner face -> u clamped to 0
            (0.040, 0.0, 0.0),    # exactly at inner face -> u = 0
            (0.0425, 1.0, 2.0),   # quarter depth
            (0.045, -1.0, 0.5),   # half depth
            (0.050, 0.0, 0.0),    # outer face -> u = 1
            (0.055, 0.0, 0.0),    # past the outer face -> u = 1.5
        ],
    ),
    (
        "minus_z", [0.0, 0.0, -1.0], -0.020, 0.008, 3.0, 6.0,
        [
            (0.1, 0.2, -0.010),   # before inner face (coord > inner_face, sign -) -> u<0 clamp
            (0.0, 0.0, -0.020),   # at inner face
            (1.0, -1.0, -0.022),  # quarter depth
            (0.0, 0.0, -0.026),   # 3/4 depth
            (0.0, 0.0, -0.028),   # outer face -> u = 1
        ],
    ),
    (
        "plus_y", [0.0, 1.0, 0.0], 0.000, 0.005, 1.5, 10.0,
        [
            (0.3, -0.001, 0.7),   # before inner face -> clamp
            (0.0, 0.000, 0.0),    # at inner face
            (-1.0, 0.001, 2.0),   # depth u = 0.2
            (0.0, 0.0025, 0.0),   # half depth
            (0.0, 0.005, 0.0),    # outer face -> u = 1
        ],
    ),
    (
        "minus_x", [-1.0, 0.0, 0.0], -0.015, 0.006, 2.5, 7.5,
        [
            (-0.010, 0.0, 0.0),   # before inner face (coord > inner_face) -> clamp
            (-0.015, 1.0, -1.0),  # at inner face
            (-0.018, 0.0, 0.0),   # half depth
            (-0.021, 0.0, 0.0),   # outer face -> u = 1
        ],
    ),
]


def pml_axis_sign(direction):
    """Replicate PmlRegion::stretch axis/sign selection EXACTLY."""
    d = direction
    a0, a1, a2 = abs(d[0]), abs(d[1]), abs(d[2])
    if a0 > a1 and a0 > a2:
        axis = 0
    elif a1 > a2:
        axis = 1
    else:
        axis = 2
    sign = math.copysign(1.0, d[axis])  # f64::signum semantics for nonzero
    return axis, sign


def pml_stretch(direction, inner_face, thickness, exponent, delta_max, x, y, z):
    axis, sign = pml_axis_sign(direction)
    coord = [x, y, z][axis]
    u_raw = sign * (coord - inner_face) / thickness
    u = max(u_raw, 0.0)
    s_d = complex(1.0, 0.0) - 1j * (u ** exponent * delta_max)
    s = [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0)]
    s[axis] = s_d
    return s


# ---------------------------------------------------------------------------
# Group B: lossy dielectric
# ---------------------------------------------------------------------------

# (name, er, tand, sigma, freq_hz)
DIEL_CASES = [
    ("fr4", 4.4, 0.02, 0.0, 2.4e9),
    ("seawater", 81.0, 0.0, 4.0, 1.0e9),
    ("lossy_combo", 10.2, 0.0035, 0.01, 6.0e9),
    ("air", 1.0, 0.0, 0.0, 1.0e10),
    ("low_freq_cond", 3.0, 0.01, 0.5, 1.0e6),
]


def lossy_eps(er, tand, sigma, freq):
    w0 = TWO_PI * freq
    return er * (1.0 - 1j * tand) - 1j * sigma / (w0 * EPS0)


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

HEADER = """\
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// GENERATED by derivations/materials/emit_material_golden.py — do not edit.
// Golden material tensors for the frequency-domain assembly.
//   (A) Uniaxial PML stretch  s_d = 1 - j·(u^n·δmax),  u = max(0, sign·(coord-inner)/t)
//   (B) Lossy dielectric      εr* = εr·(1 - j·tanδ) - j·σ/(ω·ε₀)
// Both pinned to the exact closed forms the Rust implements (e^{+jωt}).

use rapidfem_core::materials::{build_material_tensors, Dispersion, Material, PmlRegion};

fn rel_err(got: f64, want: f64) -> f64 {
    let scale = want.abs().max(1e-300);
    (got - want).abs() / scale
}
"""


def emit_pml_case(name, direction, inner_face, thickness, exponent, delta_max, points):
    out = f"\n// ===== PML {name}: dir={direction}, inner={inner_face}, t={thickness}, n={exponent}, dmax={delta_max} =====\n"
    out += "#[test]\n"
    out += f"fn pml_stretch_matches_closed_form_{name}() {{\n"
    out += "    let region = PmlRegion {\n"
    out += "        tet_indices: vec![],\n"
    out += "        er_base: 1.0,\n"
    out += "        ur_base: 1.0,\n"
    out += f"        direction: [{f64(direction[0])}, {f64(direction[1])}, {f64(direction[2])}],\n"
    out += f"        inner_face: {f64(inner_face)},\n"
    out += f"        thickness: {f64(thickness)},\n"
    out += f"        exponent: {f64(exponent)},\n"
    out += f"        delta_max: {f64(delta_max)},\n"
    out += "    };\n"
    out += "    // (x, y, z, sx_re, sx_im, sy_re, sy_im, sz_re, sz_im)\n"
    out += "    let table: &[(f64, f64, f64, f64, f64, f64, f64, f64, f64)] = &[\n"
    for (x, y, z) in points:
        s = pml_stretch(direction, inner_face, thickness, exponent, delta_max, x, y, z)
        out += (
            f"        ({f64(x)}, {f64(y)}, {f64(z)}, "
            f"{f64(s[0].real)}, {f64(s[0].imag)}, "
            f"{f64(s[1].real)}, {f64(s[1].imag)}, "
            f"{f64(s[2].real)}, {f64(s[2].imag)}),\n"
        )
    out += "    ];\n"
    out += "    let mut max_err = 0.0_f64;\n"
    out += "    for &(x, y, z, sx_re, sx_im, sy_re, sy_im, sz_re, sz_im) in table {\n"
    out += "        let s = region.stretch(x, y, z);\n"
    out += "        let want = [(sx_re, sx_im), (sy_re, sy_im), (sz_re, sz_im)];\n"
    out += "        for k in 0..3 {\n"
    out += "            let er = rel_err(s[k].re, want[k].0);\n"
    out += "            let ei = rel_err(s[k].im, want[k].1);\n"
    out += "            max_err = max_err.max(er).max(ei);\n"
    out += f'            assert!(er < 1e-12, "{name} s[{{}}] re mismatch @ ({{:.4e}},{{:.4e}},{{:.4e}}): got {{:.17e}} want {{:.17e}} (rel {{:.2e}})", k, x, y, z, s[k].re, want[k].0, er);\n'
    out += f'            assert!(ei < 1e-12, "{name} s[{{}}] im mismatch @ ({{:.4e}},{{:.4e}},{{:.4e}}): got {{:.17e}} want {{:.17e}} (rel {{:.2e}})", k, x, y, z, s[k].im, want[k].1, ei);\n'
    out += "        }\n"
    out += "    }\n"
    out += f'    eprintln!("pml {name}: max rel err {{:.2e}}", max_err);\n'
    out += "}\n"
    return out


def emit_diel_case(name, er, tand, sigma, freq):
    z = lossy_eps(er, tand, sigma, freq)
    out = f"\n// ===== dielectric {name}: er={er}, tand={tand}, sigma={sigma}, f={freq} =====\n"
    out += "#[test]\n"
    out += f"fn lossy_dielectric_matches_closed_form_{name}() {{\n"
    out += "    let mat = Material {\n"
    out += f"        er: {f64(er)},\n"
    out += "        ur: 1.0,\n"
    out += f"        tand: {f64(tand)},\n"
    out += f"        cond: {f64(sigma)},\n"
    out += "        tet_indices: vec![0],\n"
    out += "        er_diag: None,\n"
    out += "        ur_diag: None,\n"
    out += "        dispersion: Dispersion::None,\n"
    out += "    };\n"
    out += f"    let (er_t, _ur_t) = build_material_tensors(1, &[mat], {f64(freq)});\n"
    out += f"    let want_re = {f64(z.real)};\n"
    out += f"    let want_im = {f64(z.imag)};\n"
    out += "    let mut max_err = 0.0_f64;\n"
    out += "    // Isotropic: all three diagonal entries carry the same complex εr*.\n"
    out += "    for k in 0..3 {\n"
    out += "        let v = er_t[0][k][k];\n"
    out += "        let er = rel_err(v.re, want_re);\n"
    out += "        let ei = rel_err(v.im, want_im);\n"
    out += "        max_err = max_err.max(er).max(ei);\n"
    out += f'        assert!(er < 1e-12, "{name} εr* re mismatch [{{}}]: got {{:.17e}} want {{:.17e}} (rel {{:.2e}})", k, v.re, want_re, er);\n'
    out += f'        assert!(ei < 1e-12, "{name} εr* im mismatch [{{}}]: got {{:.17e}} want {{:.17e}} (rel {{:.2e}})", k, v.im, want_im, ei);\n'
    out += "    }\n"
    out += f'    eprintln!("dielectric {name}: max rel err {{:.2e}}", max_err);\n'
    out += "}\n"
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    out_path = os.path.join(repo, "crates", "rapidfem-core", "tests", "material_tensor_golden_test.rs")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    body = HEADER
    for case in PML_CASES:
        body += emit_pml_case(*case)
        print(f"emitted PML case {case[0]}")
    for case in DIEL_CASES:
        body += emit_diel_case(*case)
        print(f"emitted dielectric case {case[0]}")

    with open(out_path, "w") as fh:
        fh.write(body)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
