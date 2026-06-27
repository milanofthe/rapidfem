// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! MKL PARDISO sparse direct solver via dynamic loading.
//!
//! Loads `mkl_rt.dll` (or `mkl_rt.2.dll`) at runtime. If MKL is not installed,
//! `PardisoSolver::try_new()` returns `None` and the caller falls back to faer.
//!
//! Uses mtype=6 (complex symmetric indefinite) with 0-based CSR indexing.
//! Upper triangle only, exploits A = Aᵀ for 2× memory/speed savings.
//!
//! When the `pardiso` feature is disabled (e.g., WASM builds), this module exposes
//! API-compatible stubs: `try_new` returns None and the caller path that uses
//! PARDISO is statically dead.

#[cfg(not(feature = "pardiso"))]
pub use stub::*;
#[cfg(feature = "pardiso")]
pub use inner::*;

#[cfg(not(feature = "pardiso"))]
mod stub {
    use num_complex::Complex64 as C64;
    pub struct PardisoSolver;
    impl PardisoSolver {
        pub fn try_new() -> Option<Self> { None }
        pub fn analyze_and_factorize(&mut self, _n: i32, _ia: &[i32], _ja: &[i32], _a: &[C64]) -> Result<(), String> { unreachable!() }
        pub fn factorize(&mut self, _n: i32, _ia: &[i32], _ja: &[i32], _a: &[C64]) -> Result<(), String> { unreachable!() }
        pub fn solve(&mut self, _n: i32, _ia: &[i32], _ja: &[i32], _a: &[C64], _b: &[C64]) -> Result<Vec<C64>, String> { unreachable!() }
    }
    pub fn build_upper_csr(
        _n: usize, _r: &[usize], _c: &[usize], _v: &[C64],
    ) -> (Vec<i32>, Vec<i32>, Vec<C64>) { unreachable!() }
}

#[cfg(feature = "pardiso")]
mod inner {
use std::ffi::c_void;
use num_complex::Complex64 as C64;

/// PARDISO function signature (Fortran calling convention: all args by pointer)
type PardisoFn = unsafe extern "C" fn(
    pt: *mut i64,       // [64] internal handle
    maxfct: *const i32,
    mnum: *const i32,
    mtype: *const i32,
    phase: *const i32,
    n: *const i32,
    a: *const c_void,   // complex values
    ia: *const i32,     // row pointers
    ja: *const i32,     // column indices
    perm: *mut i32,
    nrhs: *const i32,
    iparm: *mut i32,    // [64] control
    msglvl: *const i32,
    b: *mut c_void,     // RHS
    x: *mut c_void,     // solution
    error: *mut i32,
);

pub struct PardisoSolver {
    _lib: libloading::Library,
    pardiso_fn: PardisoFn,
    pt: [i64; 64],
    iparm: [i32; 64],
    mtype: i32,
    analyzed: bool,
}

impl PardisoSolver {
    /// Try to load MKL and create a PARDISO solver.
    /// Returns None if MKL is not found on the system.
    pub fn try_new() -> Option<Self> {
        let lib = unsafe {
            libloading::Library::new("mkl_rt.2.dll")
                .or_else(|_| libloading::Library::new("mkl_rt.dll"))
                .or_else(|_| libloading::Library::new("libmkl_rt.so"))
                .or_else(|_| libloading::Library::new("libmkl_rt.dylib"))
                .ok()?
        };

        let pardiso_fn: PardisoFn = unsafe {
            let sym: libloading::Symbol<PardisoFn> = lib.get(b"pardiso")
                .or_else(|_| lib.get(b"pardiso_"))
                .ok()?;
            *sym
        };

        let pt = [0i64; 64];
        let mut iparm = [0i32; 64];

        // PARDISO iparm configuration (complex-symmetric, mtype=6)
        iparm[0] = 1;    // Don't use default values, we set them
        iparm[1] = 3;    // Permutation: METIS-style minimum-degree (parallel nested dissection)
        iparm[2] = 4;    // Number of threads
        iparm[7] = 0;    // No iterative refinement
        iparm[9] = 13;   // Pivot perturbation magnitude (1e-13). Critical for ill-conditioned
                         // matrices like PML's anisotropic-complex stretched tensors.
        iparm[12] = 2;   // Improved weighted matching, needed for complex-symmetric non-PD.
        iparm[34] = 1;   // 0-based (C-style) indexing

        eprintln!("  PARDISO: MKL loaded successfully");

        Some(PardisoSolver {
            _lib: lib,
            pardiso_fn,
            pt,
            iparm,
            mtype: 6, // complex symmetric indefinite
            analyzed: false,
        })
    }

    /// Phase 22: Numerical factorization (reuses the symbolic factorisation).
    pub fn factorize(&mut self, n: i32, ia: &[i32], ja: &[i32], a: &[C64]) -> Result<(), String> {
        let mut error = 0i32;
        let maxfct = 1i32;
        let mnum = 1i32;
        let phase = 22i32;
        let nrhs = 1i32;
        let msglvl = 0i32;
        let mut perm = vec![0i32; n as usize];
        let mut dummy_b: Vec<C64> = vec![C64::new(0.0, 0.0); n as usize];
        let mut dummy_x: Vec<C64> = vec![C64::new(0.0, 0.0); n as usize];

        unsafe {
            (self.pardiso_fn)(
                self.pt.as_mut_ptr(),
                &maxfct, &mnum, &self.mtype, &phase, &n,
                a.as_ptr() as *const c_void,
                ia.as_ptr(), ja.as_ptr(),
                perm.as_mut_ptr(), &nrhs,
                self.iparm.as_mut_ptr(), &msglvl,
                dummy_b.as_mut_ptr() as *mut c_void,
                dummy_x.as_mut_ptr() as *mut c_void,
                &mut error,
            );
        }

        if error != 0 {
            return Err(format!("PARDISO factorize (phase 22) error: {}", error));
        }
        Ok(())
    }

    /// Phase 33: Solve Ax = b using the existing factorization.
    pub fn solve(&mut self, n: i32, ia: &[i32], ja: &[i32], a: &[C64], b: &[C64]) -> Result<Vec<C64>, String> {
        let mut error = 0i32;
        let maxfct = 1i32;
        let mnum = 1i32;
        let phase = 33i32;
        let nrhs = 1i32;
        let msglvl = 0i32;
        let mut perm = vec![0i32; n as usize];
        let mut b_copy = b.to_vec();
        let mut x = vec![C64::new(0.0, 0.0); n as usize];

        unsafe {
            (self.pardiso_fn)(
                self.pt.as_mut_ptr(),
                &maxfct, &mnum, &self.mtype, &phase, &n,
                a.as_ptr() as *const c_void,
                ia.as_ptr(), ja.as_ptr(),
                perm.as_mut_ptr(), &nrhs,
                self.iparm.as_mut_ptr(), &msglvl,
                b_copy.as_mut_ptr() as *mut c_void,
                x.as_mut_ptr() as *mut c_void,
                &mut error,
            );
        }

        if error != 0 {
            return Err(format!("PARDISO solve (phase 33) error: {}", error));
        }
        Ok(x)
    }

    /// Combined phase 12: analyze + factorize in one call.
    pub fn analyze_and_factorize(&mut self, n: i32, ia: &[i32], ja: &[i32], a: &[C64]) -> Result<(), String> {
        let mut error = 0i32;
        let maxfct = 1i32;
        let mnum = 1i32;
        let phase = 12i32;
        let nrhs = 1i32;
        let msglvl = 0i32;
        let mut perm = vec![0i32; n as usize];
        let mut dummy_b: Vec<C64> = vec![C64::new(0.0, 0.0); n as usize];
        let mut dummy_x: Vec<C64> = vec![C64::new(0.0, 0.0); n as usize];

        unsafe {
            (self.pardiso_fn)(
                self.pt.as_mut_ptr(),
                &maxfct, &mnum, &self.mtype, &phase, &n,
                a.as_ptr() as *const c_void,
                ia.as_ptr(), ja.as_ptr(),
                perm.as_mut_ptr(), &nrhs,
                self.iparm.as_mut_ptr(), &msglvl,
                dummy_b.as_mut_ptr() as *mut c_void,
                dummy_x.as_mut_ptr() as *mut c_void,
                &mut error,
            );
        }

        if error != 0 {
            return Err(format!("PARDISO analyze+factorize (phase 12) error: {}", error));
        }
        self.analyzed = true;
        Ok(())
    }

    /// Check if MKL PARDISO is available on this system.
    pub fn is_available() -> bool {
        Self::try_new().is_some()
    }
}

impl Drop for PardisoSolver {
    fn drop(&mut self) {
        if !self.analyzed { return; }
        let mut error = 0i32;
        let maxfct = 1i32;
        let mnum = 1i32;
        let phase = -1i32; // release
        let n = 0i32;
        let nrhs = 0i32;
        let msglvl = 0i32;
        let dummy: i32 = 0;

        unsafe {
            (self.pardiso_fn)(
                self.pt.as_mut_ptr(),
                &maxfct, &mnum, &self.mtype, &phase, &n,
                std::ptr::null(),
                &dummy, &dummy,
                std::ptr::null_mut(), &nrhs,
                self.iparm.as_mut_ptr(), &msglvl,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut error,
            );
        }
    }
}

/// Build upper-triangle CSR from COO triplets for PARDISO mtype=6.
/// Filters to keep only entries where row <= col, builds 0-indexed CSR.
/// Duplicate entries at the same (row, col) are summed.
pub fn build_upper_csr(
    n: usize,
    triplet_rows: &[usize],
    triplet_cols: &[usize],
    triplet_vals: &[C64],
) -> (Vec<i32>, Vec<i32>, Vec<C64>) {
    // Filter upper triangle (r <= c), sort by (row, col), merge duplicates.
    let mut entries: Vec<(usize, usize, C64)> = Vec::with_capacity(triplet_rows.len());
    for i in 0..triplet_rows.len() {
        let (r, c) = (triplet_rows[i], triplet_cols[i]);
        if r <= c {
            entries.push((r, c, triplet_vals[i]));
        }
    }
    entries.sort_unstable_by_key(|&(r, c, _)| (r, c));

    // Merge duplicates and build CSR in one pass
    let mut ia = vec![0i32; n + 1];
    let mut ja: Vec<i32> = Vec::with_capacity(entries.len());
    let mut a: Vec<C64> = Vec::with_capacity(entries.len());

    let mut prev_row = 0usize;

    let mut i = 0;
    while i < entries.len() {
        let (r, c, mut v) = entries[i];
        i += 1;
        // Sum duplicates at same (r, c)
        while i < entries.len() && entries[i].0 == r && entries[i].1 == c {
            v += entries[i].2;
            i += 1;
        }
        // Fill row pointers up to row r
        while prev_row <= r {
            ia[prev_row] = ja.len() as i32;
            prev_row += 1;
        }
        ja.push(c as i32);
        a.push(v);
    }
    while prev_row <= n {
        ia[prev_row] = ja.len() as i32;
        prev_row += 1;
    }

    (ia, ja, a)
}

}  // end mod inner

