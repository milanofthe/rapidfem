# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Emit the Rust golden test for the RectWaveguide analytic modal quantities.

Pins the clean-room Rust `RectWaveguide` (crates/rapidfem-fd/src/waveguide.rs)
against closed-form rectangular-waveguide modal theory (Pozar, *Microwave
Engineering*, §3.3) for several (a, b, mode, frequency) cases:

  * propagation constant   β = sqrt(εr·k0² − (mπ/a)² − (nπ/b)²)        (Pozar 3.119)
  * TE wave impedance       Z_TE = ω μ0 / β                            (Pozar 3.22)
  * the transverse modal E-field profile, sampled at cross-section points.

The field uses the *exact* Rust convention: a port-LOCAL coordinate frame whose
origin sits at the face centre, and components

  E_v = pol·amp·cos(mπ x_l/a)·cos(nπ y_l/b)   -> local E_y
  E_h = pol·amp·sin(mπ x_l/a)·sin(nπ y_l/b)   -> local E_x
  E_z = 0

all scaled by qmode = sqrt(Z_TE/Z0), then rotated back to global components.
With the centred local frame (x_l = x − a/2) the TE10 vertical component reduces
to the textbook  E_y = q·pol·amp·sin(π x/a)  — verified symbolically below.

Constants are taken byte-for-byte from rapidfem-core/src/constants.rs so the
golden matches the Rust f64 arithmetic to rounding.
"""
from __future__ import annotations

import math
import os

import sympy as sp

# ---- constants, identical to rapidfem-core/src/constants.rs ----
C0 = 299_792_458.0
Z0 = 376.73031366857
EPS0 = 8.854187818814e-12
MU0 = 1.0 / (C0 * C0 * EPS0)
PI = math.pi  # IEEE-754 double nearest to pi == std::f64::consts::PI


# ---------------------------------------------------------------------------
# Symbolic anchor (Pozar 3.119 / 3.22) and a self-check that the exact Rust
# field chain reduces to the textbook half-sine for TE10.
# ---------------------------------------------------------------------------
def _symbolic_selfcheck() -> None:
    x, a, amp, q, pol = sp.symbols("x a amp q pol", positive=True)
    # Rust local field for TE10 (m=1, n=0) with centred local coord x_l = x - a/2.
    x_l = x - a / 2
    ey_local = q * pol * amp * sp.cos(sp.pi * x_l / a) * sp.cos(0)
    ex_local = q * pol * amp * sp.sin(sp.pi * x_l / a) * sp.sin(0)
    textbook = q * pol * amp * sp.sin(sp.pi * x / a)
    assert sp.simplify(ey_local - textbook) == 0, "TE10 E_y must equal q·pol·amp·sin(pi x/a)"
    assert sp.simplify(ex_local) == 0, "TE10 E_x must vanish"


# ---------------------------------------------------------------------------
# Numeric reference: replicate the EXACT Rust call chain in Python f64.
# ---------------------------------------------------------------------------
def k0_of(freq: float) -> float:
    # Excitation::new(freq, 1.0): k0 = (2*pi*freq)/C0 * 1.0
    return (2.0 * PI * freq) / C0


def omega_scaled_of(freq: float) -> float:
    # exc.omega_scaled() = k0 * C0 ; for l0 = 1 this is the physical omega = 2*pi*f.
    return k0_of(freq) * C0


def beta_of(a, b, m, n, er, freq) -> float:
    k0 = k0_of(freq)
    return math.sqrt(er * k0 * k0 - (PI * m / a) ** 2 - (PI * n / b) ** 2)


def z_mode_of(a, b, m, n, er, freq) -> float:
    return omega_scaled_of(freq) * MU0 / beta_of(a, b, m, n, er, freq)


def qmode_of(a, b, m, n, er, freq) -> float:
    return math.sqrt(z_mode_of(a, b, m, n, er, freq) / Z0)


def amplitude_of(power, a, b) -> float:
    # get_amplitude: sqrt(power * 4 * Z0 / (a*b)); Z0 (not Z_TE), polarization excluded.
    return math.sqrt(power * 4.0 * Z0 / (a * b))


def field_global(case, x, y, z) -> tuple[float, float, float]:
    """Reproduce port_mode_3d_global with an identity-axis, centred-origin CS."""
    a, b = case["a"], case["b"]
    m, n = case["m"], case["n"]
    ox, oy = case["ox"], case["oy"]
    amp = amplitude_of(case["power"], a, b)
    q = qmode_of(a, b, m, n, case["er"], case["freq"])
    pol = case["pol"]
    # in_local_cs with identity axes: subtract origin.
    xl, yl = x - ox, y - oy
    ev = pol * amp * math.cos(PI * m * xl / a) * math.cos(PI * n * yl / b)
    eh = pol * amp * math.sin(PI * m * xl / a) * math.sin(PI * n * yl / b)
    ex, ey, ez = eh, ev, 0.0
    # in_global_basis with identity axes: unchanged.
    return (q * ex, q * ey, q * ez)


# (name, a, b, m, n, er, freq, pol, power). Origin sits at the face centre.
CASES = [
    ("wr90_te10_10ghz", 22.86e-3, 10.16e-3, 1, 0, 1.0, 10e9, 1.0, 1.0),
    ("wr90_te10_12ghz", 22.86e-3, 10.16e-3, 1, 0, 1.0, 12e9, 1.0, 1.0),
    ("wr90_te20_20ghz", 22.86e-3, 10.16e-3, 2, 0, 1.0, 20e9, 1.0, 1.0),
    ("diel_te10_8ghz", 22.86e-3, 10.16e-3, 1, 0, 2.2, 8e9, -1.0, 0.5),
]


def f64(v) -> str:
    return f"{float(v):.17e}"


def make_case(rec):
    name, a, b, m, n, er, freq, pol, power = rec
    case = dict(a=a, b=b, m=m, n=n, er=er, freq=freq, pol=pol, power=power,
                ox=a / 2.0, oy=b / 2.0, name=name)
    # Sample cross-section points (global coords); y at mid-height.
    xs = [a / 8.0, a / 4.0, 3.0 * a / 8.0, a / 2.0, 5.0 * a / 8.0, 3.0 * a / 4.0]
    case["sx"] = xs
    case["sy"] = [b / 2.0] * len(xs)
    case["sz"] = [0.0] * len(xs)
    case["beta"] = beta_of(a, b, m, n, er, freq)
    case["zmode"] = z_mode_of(a, b, m, n, er, freq)
    case["qmode"] = qmode_of(a, b, m, n, er, freq)
    case["amp"] = amplitude_of(power, a, b)
    fields = [field_global(case, x, y, z)
              for x, y, z in zip(case["sx"], case["sy"], case["sz"])]
    case["ex"] = [f[0] for f in fields]
    case["ey"] = [f[1] for f in fields]
    case["ez"] = [f[2] for f in fields]
    return case


def emit_arr(name, vals):
    body = ", ".join(f64(v) for v in vals)
    return f"const {name}: [f64; {len(vals)}] = [{body}];\n"


def emit_case(case):
    n = case["name"]
    o = f"\n// ===== case {n}: a={case['a']*1e3:g}mm b={case['b']*1e3:g}mm " \
        f"TE{case['m']}{case['n']} f={case['freq']/1e9:g}GHz er={case['er']:g} " \
        f"pol={case['pol']:g} =====\n"
    o += emit_arr(f"SX_{n}", case["sx"])
    o += emit_arr(f"SY_{n}", case["sy"])
    o += emit_arr(f"SZ_{n}", case["sz"])
    o += emit_arr(f"EX_{n}", case["ex"])
    o += emit_arr(f"EY_{n}", case["ey"])
    o += emit_arr(f"EZ_{n}", case["ez"])
    o += f"""
#[test]
fn rect_waveguide_matches_pozar_{n}() {{
    let wg = RectWaveguide {{
        port_number: 1,
        power: {f64(case['power'])},
        mode: ({case['m']}, {case['n']}),
        er: {f64(case['er'])},
        polarization: {f64(case['pol'])},
        dims: ({f64(case['a'])}, {f64(case['b'])}),
        cs: CoordinateSystem::new(
            [{f64(case['ox'])}, {f64(case['oy'])}, 0.0],
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    }};
    let exc = Excitation::new({f64(case['freq'])}, 1.0);

    // β  (Pozar 3.119)
    let beta_err = rel(wg.get_beta(&exc), {f64(case['beta'])});
    // Z_TE = ω μ0 / β  (Pozar 3.22)
    let z_err = rel(wg.z_mode(&exc), {f64(case['zmode'])});
    // qmode = sqrt(Z_TE/Z0)
    let q_err = rel(wg.qmode(&exc), {f64(case['qmode'])});
    // amplitude = sqrt(P·4·Z0/(a·b))
    let a_err = rel(wg.get_amplitude(&exc), {f64(case['amp'])});
    assert!(beta_err < 1e-9, "{n} beta rel err {{:.2e}}", beta_err);
    assert!(z_err < 1e-9, "{n} Z_TE rel err {{:.2e}}", z_err);
    assert!(q_err < 1e-9, "{n} qmode rel err {{:.2e}}", q_err);
    assert!(a_err < 1e-9, "{n} amplitude rel err {{:.2e}}", a_err);

    // Transverse modal E-field at sample cross-section points.
    let mut scale = 1e-300_f64;
    let mut fmax = 0.0_f64;
    for i in 0..SX_{n}.len() {{
        let (ex, ey, ez) = wg.port_mode_3d_global(SX_{n}[i], SY_{n}[i], SZ_{n}[i], &exc);
        scale = scale.max(EX_{n}[i].abs()).max(EY_{n}[i].abs()).max(EZ_{n}[i].abs());
        fmax = fmax.max((ex - EX_{n}[i]).abs());
        fmax = fmax.max((ey - EY_{n}[i]).abs());
        fmax = fmax.max((ez - EZ_{n}[i]).abs());
    }}
    let field_err = fmax / scale;
    eprintln!("{n}: beta {{:.2e}} Z {{:.2e}} q {{:.2e}} amp {{:.2e}} field {{:.2e}}",
        beta_err, z_err, q_err, a_err, field_err);
    assert!(field_err < 1e-6, "{n} field rel err {{:.2e}}", field_err);
}}
"""
    return o


HEADER = """\
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// GENERATED by derivations/waveguide/emit_waveguide_golden.py — do not edit.
// Golden modal quantities for the analytic RectWaveguide port, pinned against
// closed-form rectangular-waveguide theory (Pozar, Microwave Engineering §3.3):
//   beta = sqrt(er*k0^2 - (m*pi/a)^2 - (n*pi/b)^2)   (3.119)
//   Z_TE = omega*mu0/beta                            (3.22)
//   transverse TE_mn E-field profile at sample points.
// Excitation is built as the solver does, Excitation::new(freq, 1.0) (physical
// units, l0 = 1 so k0 is the physical free-space wavenumber).
#![allow(non_upper_case_globals)]

use rapidfem_fd::waveguide::{CoordinateSystem, RectWaveguide};
use rapidfem_fd::excitation::Excitation;

fn rel(got: f64, exp: f64) -> f64 {
    (got - exp).abs() / exp.abs().max(1e-300)
}
"""


def main():
    _symbolic_selfcheck()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    out_path = os.path.join(repo, "crates", "rapidfem-fd", "tests",
                            "waveguide_modal_golden_test.rs")
    body = HEADER
    for rec in CASES:
        case = make_case(rec)
        body += emit_case(case)
        print(f"emitted case {case['name']}: "
              f"beta={case['beta']:.6e} Z_TE={case['zmode']:.6e} qmode={case['qmode']:.6f}")
    with open(out_path, "w") as fh:
        fh.write(body)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
