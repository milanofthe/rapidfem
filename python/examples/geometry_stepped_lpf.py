"""
Stepped-impedance microstrip low-pass filter: WavePort, Dielectric, ABC,
auto_refine_features recipe.

Demonstrates a 7-section stepped-impedance LPF on a Rogers RO4003-class
substrate (er=2.2, h=62 mil). WavePort boundary conditions are set on
port-window plates with a shared PEC strip; ABC absorbs radiation from
the open sides and top. g.auto_refine_features() applies the validated
recipe (substrate thickness / 1.5 mesh size at dielectric-air interfaces).

Campaign-validated 3-dB cutoff: 1.76 GHz.
"""
import sys

import numpy as np
import rapidfem as rf


def main() -> int:
    mm = 1e-3
    mil = 0.0254 * mm
    LENGTHS = [x * mil for x in [400, 660, 660, 660, 660, 660, 400]]
    WIDTHS = [x * mil for x in [50, 128, 8, 224, 8, 128, 50]]
    SUB_H = 62 * mil
    ER_SUB = 2.2
    AIR_H = 15 * mm
    PAD_Y = 12 * mm
    PORT_W = 12.0 * mm
    PORT_H = 6.0 * SUB_H
    # Reduced frequency count so the example finishes in a few minutes.
    # Campaign used 21 points; 9 points is sufficient for a cutoff check.
    FREQUENCIES = np.linspace(0.2e9, 8.0e9, 9)
    MAXH = rf.lambda_maxh(f_max=8.0e9)
    total_L = sum(LENGTHS)
    sub_W = max(WIDTHS) + 2 * PAD_Y
    x_lo, x_hi = -total_L / 2, total_L / 2
    F0 = 0.5 * (FREQUENCIES[0] + FREQUENCIES[-1])

    g = rf.Geometry(maxh=MAXH)
    sub = g.box(total_L, sub_W, SUB_H, position=(x_lo, -sub_W / 2, 0),
                material=rf.Dielectric(er=ER_SUB))
    air = g.box(total_L, sub_W, AIR_H, position=(x_lo, -sub_W / 2, SUB_H),
                material=rf.Air())
    x_cursor = x_lo
    trace_plates = []
    for L_seg, W_seg in zip(LENGTHS, WIDTHS):
        trace_plates.append(
            g.xy_plate(L_seg, W_seg, position=(x_cursor, -W_seg / 2, SUB_H), maxh=0.4 * mm))
        x_cursor += L_seg
    port_in = g.plate(p0=(x_lo, -PORT_W / 2, 0), width=(0, PORT_W, 0), height=(0, 0, PORT_H))
    port_out = g.plate(p0=(x_hi, -PORT_W / 2, 0), width=(0, PORT_W, 0), height=(0, 0, PORT_H))

    pec_strip = rf.PEC(sub.faces.min(axis="z"), *trace_plates)
    rf.WavePort(port_in, f0=F0, mode_kind="auto", pec=[pec_strip])
    rf.WavePort(port_out, f0=F0, mode_kind="auto", pec=[pec_strip])
    rf.ABC(
        sub.faces.min(axis="y"), sub.faces.max(axis="y"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
        air.faces.max(axis="z"),
    )
    rf.PEC(sub.faces.where(lambda c, b: abs(b[3] - b[0]) < 1e-9).unassigned,
           air.faces.where(lambda c, b: abs(b[3] - b[0]) < 1e-9).unassigned)
    g.auto_refine_features()  # substrate -> thickness/1.5 (validated recipe)
    g.mesh()
    print(f"meshed: {g.n_tets} tets")

    res = rf.Problem(g).sweep(FREQUENCIES)
    s = np.asarray(res.sparams)
    db = 20 * np.log10(np.maximum(np.abs(s[:, 1, 0]), 1e-12))
    idx = next((i for i, d in enumerate(db) if d < -3), len(db) - 1)
    cutoff_ghz = float(FREQUENCIES[idx] / 1e9)
    floor_db = float(db.min())
    print(f"3-dB cutoff: {cutoff_ghz:.2f} GHz  (campaign-validated: 1.76 GHz)")
    print(f"stopband floor: {floor_db:.1f} dB")

    # campaign-validated 3-dB cutoff is 1.76 GHz; tolerance = one frequency step
    freq_step_ghz = float((FREQUENCIES[-1] - FREQUENCIES[0]) / (len(FREQUENCIES) - 1) / 1e9)
    if abs(cutoff_ghz - 1.76) > freq_step_ghz:
        print(f"FAIL: cutoff {cutoff_ghz:.2f} GHz deviates from 1.76 GHz by more than "
              f"{freq_step_ghz:.3f} GHz")
        return 1
    if floor_db > -12.0:
        print(f"FAIL: stopband floor {floor_db:.1f} dB is above -12 dB")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
