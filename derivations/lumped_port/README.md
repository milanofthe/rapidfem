# Lumped port (RLC), frequency domain

Clean-room derivation of the driven RLC lumped port and its S-parameter
extraction, independent of any other solver. Run `python lumped_port.py`.

The physics is textbook EM:

- **Impedance (Leontovich) BC** in the curl-curl weak form (Jin, *The FEM in
  Electromagnetics*) gives the Robin surface term `gamma = i*kappa*eta0/Zs`.
- **Lumped element -> sheet impedance**: `Zs = Z(omega)*(w/l)`, with `Z` the
  series `R + iwL + 1/(iwC)` or parallel `(1/R + 1/(iwL) + iwC)^-1`.
- **Kurokawa power waves** (Pozar, *Microwave Engineering*) give the
  S-parameter `S = V/V_inc` (`-1` on the driven port).

The script re-derives every constant with sympy and **proves** the full chain
(`gamma`, the scattered-field source `u_inc = -2*gamma*E_inc`, `V_inc =
sqrt(2*P*Z0)`, the mode voltage `V = (1/w) INT E.l̂ dS`) reproduces the analytic
reflection coefficient `(Zload - Z0)/(Zload + Z0)`, and that the factor `2` in
the source is load-bearing.

The single change vs the previous (R-only, line-integral) lumped port is the
voltage extraction: an **area-averaged mode projection** assembled as a boundary
linear form and dotted with the solved field, which is well-defined for tall /
non-TEM ports (e.g. a 184 µm RFIC feed) where a few discrete line integrals
through a non-uniform field degenerate.

Conventions match the kernel: `e^{+i*omega*t}`, `K = E - k0^2 B`,
coordinates scaled by `L0`, `kappa = k0*L0`
(see `../basis_nondim/nondimensionalize.py`).
