// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Network post-processing of a sweep: renormalisation of the S-parameters
//! to a new reference impedance and the Touchstone writer.
//!
//! Every driven port reports its S-parameters against its own reference
//! (lumped ports their `z0`, modal ports their frequency-dependent mode
//! impedance, recorded per frequency in [`SweepResult::port_impedances`]).
//! [`renormalize`] re-references them through the normalised impedance
//! matrix: `S -> Zbar = (I+S)(I-S)^-1 -> Z = D_old·Zbar·D_old ->
//! Zbar' = D_new^-1·Z·D_new^-1 -> S' = (Zbar'-I)(Zbar'+I)^-1`, with
//! `D = diag(sqrt(z))`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

use num_complex::Complex64 as C64;

use crate::simulation::SweepResult;

/// A dense square matrix, row-major.
type Matrix = Vec<Vec<C64>>;

fn identity(n: usize) -> Matrix {
    (0..n)
        .map(|i| (0..n).map(|j| C64::new(if i == j { 1.0 } else { 0.0 }, 0.0)).collect())
        .collect()
}

fn add(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter().zip(b).map(|(ra, rb)| ra.iter().zip(rb).map(|(&x, &y)| x + y).collect()).collect()
}

fn sub(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter().zip(b).map(|(ra, rb)| ra.iter().zip(rb).map(|(&x, &y)| x - y).collect()).collect()
}

fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    (0..n)
        .map(|i| (0..n).map(|j| (0..n).map(|k| a[i][k] * b[k][j]).sum::<C64>()).collect())
        .collect()
}

/// The inverse by Gauss-Jordan elimination with partial pivoting; `None`
/// for a singular matrix.
fn inverse(a: &Matrix) -> Option<Matrix> {
    let n = a.len();
    let zero = C64::new(0.0, 0.0);
    let mut m = a.clone();
    let mut inv = identity(n);
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i][col].norm().total_cmp(&m[j][col].norm()))?;
        if m[piv][col] == zero || !m[piv][col].is_finite() {
            return None;
        }
        m.swap(col, piv);
        inv.swap(col, piv);
        let p = m[col][col];
        for j in 0..n {
            m[col][j] /= p;
            inv[col][j] /= p;
        }
        for i in 0..n {
            let f = m[i][col];
            if i == col || f == zero {
                continue;
            }
            for j in 0..n {
                let (mc, ic) = (m[col][j], inv[col][j]);
                m[i][j] -= f * mc;
                inv[i][j] -= f * ic;
            }
        }
    }
    Some(inv)
}

fn singular(fi: usize) -> String {
    format!("frequency {fi}: singular matrix, the S-parameters cannot be renormalized")
}

/// S-parameters `s` (`[freq][obs][exc]`) referenced to the per-frequency,
/// per-port impedances `z_old` (`[freq][port]`), re-referenced to `z_new`:
/// one impedance per port, or a single one for every port. With
/// `z_new == z_old` this is the identity; a lumped port already at `z_new`
/// is a no-op.
pub fn renormalize(s: &[Matrix], z_old: &[Vec<f64>], z_new: &[f64]) -> Result<Vec<Matrix>, String> {
    if z_old.len() != s.len() {
        return Err(format!(
            "{} frequencies of S-parameters but {} of reference impedances",
            s.len(),
            z_old.len()
        ));
    }
    let mut out = Vec::with_capacity(s.len());
    for (fi, (sf, zf)) in s.iter().zip(z_old).enumerate() {
        let n = sf.len();
        let z_ref: Vec<f64> = if z_new.len() == 1 { vec![z_new[0]; n] } else { z_new.to_vec() };
        if zf.len() != n || z_ref.len() != n || sf.iter().any(|row| row.len() != n) {
            return Err(format!(
                "frequency {fi}: a {n}-port S-matrix needs {n} old and {n} new reference \
                 impedances, got {} and {}",
                zf.len(),
                z_ref.len()
            ));
        }
        let sq_old: Vec<C64> = zf.iter().map(|&z| C64::from(z).sqrt()).collect();
        let sq_new: Vec<C64> = z_ref.iter().map(|&z| C64::from(z).sqrt()).collect();
        let eye = identity(n);
        // S (ref z_old) -> normalised impedance Zbar = (I+S)(I-S)^-1
        let zbar = matmul(&add(&eye, sf), &inverse(&sub(&eye, sf)).ok_or_else(|| singular(fi))?);
        // de-normalise to the physical Z, then normalise to z_new
        let zbar_new: Matrix = (0..n)
            .map(|i| {
                (0..n).map(|j| sq_old[i] * zbar[i][j] * sq_old[j] / sq_new[i] / sq_new[j]).collect()
            })
            .collect();
        // normalised Z -> S' (ref z_new)
        let s_new = matmul(&sub(&zbar_new, &eye), &inverse(&add(&zbar_new, &eye)).ok_or_else(|| singular(fi))?);
        out.push(s_new);
    }
    Ok(out)
}

/// The number format of a Touchstone file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TouchstoneFormat {
    /// Real and imaginary part.
    Ri,
    /// Magnitude and angle in degrees.
    Ma,
    /// Magnitude in dB and angle in degrees.
    Db,
}

impl FromStr for TouchstoneFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "ri" => Ok(TouchstoneFormat::Ri),
            "ma" => Ok(TouchstoneFormat::Ma),
            "db" => Ok(TouchstoneFormat::Db),
            _ => Err(format!("fmt must be 'ri', 'ma', or 'db', got {s:?}")),
        }
    }
}

impl TouchstoneFormat {
    fn label(self) -> &'static str {
        match self {
            TouchstoneFormat::Ri => "RI",
            TouchstoneFormat::Ma => "MA",
            TouchstoneFormat::Db => "DB",
        }
    }

    fn pair(self, c: C64) -> (f64, f64) {
        match self {
            TouchstoneFormat::Ri => (c.re, c.im),
            TouchstoneFormat::Ma => (c.norm(), c.arg().to_degrees()),
            TouchstoneFormat::Db => (20.0 * c.norm().max(1e-30).log10(), c.arg().to_degrees()),
        }
    }
}

/// Writes S-parameters (`[freq][obs][exc]`) as Touchstone 1.0 text with
/// the reference `z0` in the option line. Two-port data is column-major
/// (`S11 S21 S12 S22`); from three ports on every matrix row starts a new
/// line, at most four pairs per line.
pub fn write_touchstone(
    w: &mut impl Write,
    frequencies: &[f64],
    s: &[Matrix],
    z0: f64,
    format: TouchstoneFormat,
) -> io::Result<()> {
    let n = s.first().map_or(0, |m| m.len());
    writeln!(w, "! Touchstone file written by rapidfem")?;
    writeln!(w, "! n_ports = {n}, n_freq = {}", frequencies.len())?;
    writeln!(w, "# HZ S {} R {z0}", format.label())?;
    for (f, m) in frequencies.iter().zip(s) {
        let rows: Matrix = if n == 2 { vec![vec![m[0][0], m[1][0], m[0][1], m[1][1]]] } else { m.clone() };
        write!(w, "{f:.6e}")?;
        for (i, row) in rows.iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                if (i > 0 || j > 0) && j % 4 == 0 {
                    writeln!(w)?;
                }
                let (a, b) = format.pair(c);
                write!(w, " {a:.6e} {b:.6e}")?;
            }
        }
        writeln!(w)?;
    }
    Ok(())
}

impl SweepResult {
    /// The S-parameters re-referenced from [`Self::port_impedances`] to
    /// `z_new` (one impedance per driven port, or one for all), see
    /// [`renormalize`].
    pub fn renormalized(&self, z_new: &[f64]) -> Result<Vec<Matrix>, String> {
        renormalize(&self.sparams, &self.port_impedances, z_new)
    }

    /// Writes the S-parameters to a Touchstone file at `path`, see
    /// [`write_touchstone`].
    pub fn write_touchstone(&self, path: &Path, z0: f64, format: TouchstoneFormat) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        write_touchstone(&mut w, &self.frequencies, &self.sparams, z0, format)?;
        w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    #[test]
    fn renormalize_is_identity_for_an_unchanged_reference() {
        let s = vec![vec![vec![c(0.2, 0.1), c(0.9, -0.05)], vec![c(0.9, -0.05), c(0.1, -0.2)]]];
        let out = renormalize(&s, &[vec![75.0, 75.0]], &[75.0]).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((out[0][i][j] - s[0][i][j]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn renormalize_reproduces_a_load_reflection() {
        // A load Z_L reflects (Z_L - Z)/(Z_L + Z) against the reference Z.
        let (z_l, z_old, z_new) = (100.0, 30.0, 50.0);
        let s = vec![vec![vec![c((z_l - z_old) / (z_l + z_old), 0.0)]]];
        let out = renormalize(&s, &[vec![z_old]], &[z_new]).unwrap();
        assert!((out[0][0][0] - c((z_l - z_new) / (z_l + z_new), 0.0)).norm() < 1e-12);
    }

    #[test]
    fn touchstone_two_port_is_column_major() {
        let s = vec![vec![vec![c(1.0, 0.0), c(2.0, 0.0)], vec![c(3.0, 0.0), c(4.0, 0.0)]]];
        let mut buf = Vec::new();
        write_touchstone(&mut buf, &[1e9], &s, 50.0, TouchstoneFormat::Ri).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[2], "# HZ S RI R 50");
        let re: Vec<f64> = lines[3].split_whitespace().map(|v| v.parse().unwrap()).collect();
        assert_eq!(re, vec![1e9, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0]);
    }
}
