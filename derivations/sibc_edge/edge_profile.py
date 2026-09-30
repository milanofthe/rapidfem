# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Edge correction of the surface impedance: the profile and its check in 2D.

The exact 2D magneto-quasistatic solution of a rectangular conductor in a PEC
shield (python/tests/harness/strip2d.py) gives, along the conductor's
perimeter, the ratio of the tangential E to the surface current: the
effective surface impedance. Over Zs = (1+j)/(sigma delta) and against the
distance r to the nearest corner in skin depths it is one curve g(r/delta)
for any rectangle a few skin depths across (the table printed first, the
source of G_TABLE in crates/rapidfem-fd/src/sibc_edge.rs).

The second part solves only the field outside the conductor, with the
surface impedance Zs g(r/delta) on its perimeter, and compares R' with the
reference: within about 1 % on a grid that resolves delta at the corners,
where the plain Leontovich impedance (g = 1) is 13 to 25 % low.

Run: python derivations/sibc_edge/edge_profile.py   (a few minutes)
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "tests"))
from harness.strip2d import MU0, ShieldedStrip  # noqa: E402

SIG = 5.8e7
UM = 1e-6


def delta(f):
    return 1 / np.sqrt(np.pi * f * MU0 * SIG)


def solve(s, f):
    w = 2 * np.pi * f
    sg = s.sigma * s.metal
    K = s._lap + sp.diags(1j * w * MU0 * sg * s.area)
    return spla.spsolve(K.tocsc(), MU0 * sg * s.area)


def perimeter_zeff(s, f, x0, y0, wd, t):
    """Walk the conductor boundary faces: (s coordinate, distance to nearest corner, Zeff/Zs)."""
    A = solve(s, f); w = 2*np.pi*f
    nx, ny = len(s.dx), len(s.dy)
    A = A.reshape(nx, ny); metal = s.metal.reshape(nx, ny)
    xc = 0.5*(s.xe[1:]+s.xe[:-1]); yc = 0.5*(s.ye[1:]+s.ye[:-1])
    zs = (1+1j)*np.sqrt(w*MU0/(2*SIG))
    out = []
    # bottom and top faces (normal y), left and right (normal x)
    for i in range(nx):
        for j in range(ny-1):
            a, b = metal[i,j], metal[i,j+1]
            if a != b:
                im, io = (j, j+1) if a else (j+1, j)   # metal, outside
                d = abs(yc[io]-yc[im])
                dAdn = (A[i,io]-A[i,im])/d          # outward from metal
                K = -dAdn/MU0
                E = 1.0 - 1j*w*0.5*(A[i,im]+A[i,io])   # E at the surface (average)
                r = min(xc[i]-x0, x0+wd-xc[i])
                out.append((r, E/K/zs, s.dx[i], 'y'))
    for j in range(ny):
        for i in range(nx-1):
            a, b = metal[i,j], metal[i+1,j]
            if a != b:
                im, io = (i, i+1) if a else (i+1, i)
                d = abs(xc[io]-xc[im])
                dAdn = (A[io,j]-A[im,j])/d
                K = -dAdn/MU0
                E = 1.0 - 1j*w*0.5*(A[im,j]+A[io,j])
                r = min(yc[j]-y0, y0+t-yc[j])
                out.append((r, E/K/zs, s.dy[j], 'x'))
    return out


# The universal profile, r/delta -> g (the part-1 table at t/delta = 50).
U = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 1e9])
G = np.array([2.2 - 0.05j, 2.168 - 0.072j, 2.063 - 0.125j, 1.915 - 0.165j, 1.752 - 0.241j,
              1.620 - 0.268j, 1.428 - 0.269j, 1.282 - 0.261j, 1.180 - 0.234j, 1.058 - 0.172j,
              0.999 - 0.119j, 0.974 - 0.034j, 0.988 - 0.005j, 1.001 - 0.002j, 1.0])


def g_at(u):
    return np.interp(u, U, G.real) + 1j * np.interp(u, U, G.imag)


def g_face(u0, u1, weighted, n=200):
    """g averaged over a face spanning corner distances [u0, u1]."""
    u = np.linspace(u0, u1, n + 1)
    u = 0.5 * (u[1:] + u[:-1])
    if not weighted or u0 > 4:
        return g_at(u).mean()
    wgt = np.maximum(u, 1e-6) ** (-2 / 3)
    return (g_at(u) * wgt).sum() / wgt.sum()


def surface_z(s, f, x0, y0, wd, t, mode="g", weighted=True):
    w = 2*np.pi*f; dl = delta(f)
    zs = (1+1j)*np.sqrt(w*MU0/(2*SIG))
    nx, ny = len(s.dx), len(s.dy)
    idx = np.arange(nx*ny).reshape(nx, ny)
    metal = s.metal.reshape(nx, ny)
    L = s._lap.tolil()
    # drop couplings into metal cells: rebuild from scratch is simpler
    xc = 0.5*(s.xe[1:]+s.xe[:-1]); yc = 0.5*(s.ye[1:]+s.ye[:-1])
    rows, cols, vals = [], [], []; diag = np.zeros(nx*ny, complex); rhs = np.zeros(nx*ny, complex)
    I_terms = []
    def add(a, b, c):
        rows.extend([a,b,a,b]); cols.extend([a,b,b,a]); vals.extend([c,c,-c,-c])
    def metal_face(c, flen, d, r0, r1):
        g = g_at(0) if False else (g_face(r0/dl, r1/dl, weighted) if mode == "g" else 1.0)
        alpha = zs*g/MU0
        den = 1j*w + alpha/d
        diag[c] += flen/d*1j*w/den
        rhs[c] += flen/d*1.0/den
        I_terms.append((c, flen, d, alpha, den))
    for i in range(nx):
        for j in range(ny):
            if metal[i,j]: diag[idx[i,j]] = 1.0; continue
            c = idx[i,j]
            for (ii, jj, flen, dist, axis) in ((i+1,j,s.dy[j],None,'x'),(i,j+1,s.dx[i],None,'y')):
                if ii >= nx or jj >= ny: continue
                if axis == 'x': dist = 0.5*(s.dx[i]+s.dx[ii])
                else: dist = 0.5*(s.dy[j]+s.dy[jj])
                if metal[ii,jj] or metal[i,j]: continue
                add(c, idx[ii,jj], flen/dist)
    # metal faces
    for i in range(nx):
        for j in range(ny):
            if metal[i,j]: continue
            c = idx[i,j]
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ii, jj = i+di, j+dj
                if not (0 <= ii < nx and 0 <= jj < ny) or not metal[ii,jj]: continue
                if di: 
                    flen, d = s.dy[j], 0.5*s.dx[i]
                    r0 = min(s.ye[j]-y0, y0+t-s.ye[j+1]); r1 = min(s.ye[j+1]-y0, y0+t-s.ye[j])
                    lo, hi = s.ye[j]-y0, s.ye[j+1]-y0
                    r0, r1 = (lo, hi) if hi <= t/2 else (t-hi, t-lo)
                else:
                    flen, d = s.dx[i], 0.5*s.dy[j]
                    lo, hi = s.xe[i]-x0, s.xe[i+1]-x0
                    r0, r1 = (lo, hi) if hi <= wd/2 else (wd-hi, wd-lo)
                metal_face(c, flen, d, r0, r1)
    # shield walls
    diag[idx[0]] += s.dy/(0.5*s.dx[0]); diag[idx[-1]] += s.dy/(0.5*s.dx[-1])
    diag[idx[:,0]] += s.dx/(0.5*s.dy[0]); diag[idx[:,-1]] += s.dx/(0.5*s.dy[-1])
    n = nx*ny
    K = sp.csr_matrix((vals,(rows,cols)), shape=(n,n)) + sp.diags(diag)
    A = spla.spsolve(K.tocsc(), rhs)
    I = 0
    for c, flen, d, alpha, den in I_terms:
        As = (1.0 + alpha*A[c]/d)/den
        I += -(1/MU0)*(A[c]-As)/d*flen
    return 1.0/I


def main():
    print("1. g(r/delta) along the perimeter (Re, Im)")
    grid = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6])
    for wd, t, f in ((10 * UM, 3 * UM, 300e9), (20 * UM, 6 * UM, 300e9), (10 * UM, 3 * UM, 100e9)):
        dl = delta(f)
        W, H = 60 * UM, 40 * UM
        x0, y0 = (W - wd) / 2, 10 * UM
        s = ShieldedStrip(W=W, H=H, x0=x0, y0=y0, w=wd, t=t, sigma=SIG, h0=dl / 60, rate=1.08)
        data = perimeter_zeff(s, f, x0, y0, wd, t)
        r = np.array([d[0] for d in data]) / dl
        g = np.array([d[1] for d in data])
        o = np.argsort(r)
        gi = np.interp(grid, r[o], g[o].real) + 1j * np.interp(grid, r[o], g[o].imag)
        print(f"  w={wd / UM:g} t={t / UM:g} um, t/delta={t / dl:.0f}:",
              " ".join(f"{u}:{v.real:.3f}{v.imag:+.3f}j" for u, v in zip(grid, gi)))

    print("2. R'/reference: Leontovich | g at face centres | g weighted r^-2/3 (fine grid)")
    for wd, t in ((10 * UM, 3 * UM), (2 * UM, 3 * UM)):
        for f in (30e9, 100e9):
            dl = delta(f)
            W, H = 60 * UM, 40 * UM
            x0, y0 = (W - wd) / 2, 10 * UM
            ref = ShieldedStrip(W=W, H=H, x0=x0, y0=y0, w=wd, t=t, sigma=SIG,
                                h0=min(wd, t, dl) / 40, rate=1.15)
            zr = ref.series_impedance(f)
            ext = ShieldedStrip(W=W, H=H, x0=x0, y0=y0, w=wd, t=t, sigma=SIG, h0=dl / 10, rate=1.15)
            zl = surface_z(ext, f, x0, y0, wd, t, mode="leo")
            zp = surface_z(ext, f, x0, y0, wd, t, mode="g", weighted=False)
            zw = surface_z(ext, f, x0, y0, wd, t, mode="g", weighted=True)
            print(f"  w={wd / UM:g} t={t / UM:g} um, t/delta={t / dl:.1f}: "
                  f"{zl.real / zr.real:.3f} | {zp.real / zr.real:.3f} | {zw.real / zr.real:.3f}")


if __name__ == "__main__":
    main()
