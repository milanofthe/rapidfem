use crate::dense::matrix::SymmetricMatrix;
use crate::error::RslabError;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use num_complex::Complex;
use std::path::Path;

/// A sparse symmetric matrix in coordinate (COO) format, as read from a Matrix
/// Market file. Entries are 0-indexed, lower triangle only (i >= j). Generic
/// over the scalar field: [`f64`] (`real symmetric`, PARDISO `mtype 2`) or
/// [`Complex<f64>`] (`complex symmetric`, PARDISO `mtype 6`).
#[derive(Debug, Clone)]
pub struct MtxMatrix<T = f64> {
    pub n: usize,
    pub entries: Vec<(usize, usize, T)>,
}

impl<T: Scalar> MtxMatrix<T> {
    /// Convert to a dense symmetric matrix.
    ///
    /// X2 (REG-4): duplicate coordinates are **summed**, matching `to_csc`
    /// (which sums via `CscMatrix::from_triplets`) and the Matrix Market /
    /// COO convention used by scipy and MATLAB.
    pub fn to_dense(&self) -> SymmetricMatrix<T> {
        let mut mat = SymmetricMatrix::zeros(self.n);
        for &(i, j, v) in &self.entries {
            let prev = mat.get(i, j);
            mat.set(i, j, prev + v);
        }
        mat
    }

    /// Convert to a CSC sparse matrix (lower triangle).
    pub fn to_csc(&self) -> Result<CscMatrix<T>, RslabError> {
        let rows: Vec<usize> = self.entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = self.entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<T> = self.entries.iter().map(|&(_, _, v)| v).collect();
        CscMatrix::from_triplets(self.n, &rows, &cols, &vals)
    }

    /// Convert to a general (full, unsymmetric) CSC matrix, keeping every entry
    /// as given - the form the unsymmetric LU path / a `LinearOperator` uses.
    pub fn to_general_csc(&self) -> Result<crate::sparse::general::GeneralCsc<T>, RslabError> {
        let rows: Vec<usize> = self.entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = self.entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<T> = self.entries.iter().map(|&(_, _, v)| v).collect();
        crate::sparse::general::GeneralCsc::from_triplets(self.n, &rows, &cols, &vals)
    }
}

/// Read a **real** symmetric Matrix Market file (`mtype 2`).
///
/// Accepts only `%%MatrixMarket matrix coordinate real symmetric`. Indices are
/// converted from 1-based (MTX) to 0-based; upper-triangle entries are
/// transposed to the lower triangle.
pub fn read_mtx(path: &Path) -> Result<MtxMatrix<f64>, RslabError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| RslabError::IoError(format!("{}: {}", path.display(), e)))?;
    parse_mtx(&contents, path.to_string_lossy().as_ref())
}

/// Read a **complex** symmetric Matrix Market file (`mtype 6`).
///
/// Accepts `%%MatrixMarket matrix coordinate complex symmetric`. Each data line
/// is `i j re im`. Indices 1-based → 0-based; upper triangle → lower triangle.
pub fn read_mtx_complex(path: &Path) -> Result<MtxMatrix<Complex<f64>>, RslabError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| RslabError::IoError(format!("{}: {}", path.display(), e)))?;
    parse_mtx_complex(&contents, path.to_string_lossy().as_ref())
}

/// A Matrix Market file loaded with its structure tag, values widened to
/// `Complex<f64>` (a real input gets a zero imaginary part). Auto-detected by
/// [`read_mtx_any`] so a heterogeneous corpus (SuiteSparse) routes each matrix to
/// the right solver path.
pub enum MtxLoaded {
    /// `symmetric` file: lower triangle, for the LDLᵀ path.
    Symmetric(CscMatrix<Complex<f64>>),
    /// `general` file: full pattern, for the LU path.
    General(crate::sparse::general::GeneralCsc<Complex<f64>>),
}

/// Read any `real`/`complex`/`integer` × `symmetric`/`general` Matrix Market
/// coordinate file, widening values to `Complex<f64>`. The header is auto-detected;
/// `pattern`, `hermitian`, and `skew-symmetric` are rejected (not the
/// complex-symmetric / general forms the solvers take).
pub fn read_mtx_any(path: &Path) -> Result<MtxLoaded, RslabError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| RslabError::IoError(format!("{}: {}", path.display(), e)))?;
    let source = path.to_string_lossy();
    let header = contents.lines().next().unwrap_or("");
    let t: Vec<String> = header
        .split_whitespace()
        .map(|s| s.to_ascii_lowercase())
        .collect();
    if t.len() != 5 || t[0] != "%%matrixmarket" || t[1] != "matrix" || t[2] != "coordinate" {
        return Err(RslabError::IoError(format!(
            "{source}: unsupported header '{}'",
            header.trim()
        )));
    }
    let (field, symmetry) = (t[3].clone(), t[4].clone());
    if field != "real" && field != "complex" && field != "integer" {
        return Err(RslabError::IoError(format!(
            "{source}: unsupported field '{field}' (real/complex/integer only)"
        )));
    }
    let is_complex = field == "complex";
    let nvt = if is_complex { 2 } else { 1 };
    let to_c = move |toks: &[&str]| -> Option<Complex<f64>> {
        let re = toks.first()?.parse::<f64>().ok()?;
        let im = if is_complex {
            toks.get(1)?.parse::<f64>().ok()?
        } else {
            0.0
        };
        Some(Complex::new(re, im))
    };
    match symmetry.as_str() {
        "symmetric" => {
            let m = parse_mtx_with(&contents, &source, &field, "symmetric", nvt, to_c)?;
            Ok(MtxLoaded::Symmetric(m.to_csc()?))
        }
        "general" => {
            let m = parse_mtx_with(&contents, &source, &field, "general", nvt, to_c)?;
            Ok(MtxLoaded::General(m.to_general_csc()?))
        }
        other => Err(RslabError::IoError(format!(
            "{source}: unsupported symmetry '{other}' (symmetric/general only)"
        ))),
    }
}

/// Parse a **real** symmetric Matrix Market string (`mtype 2`).
pub fn parse_mtx(contents: &str, source: &str) -> Result<MtxMatrix<f64>, RslabError> {
    parse_mtx_with(contents, source, "real", "symmetric", 1, |toks| {
        toks[0].parse::<f64>().ok()
    })
}

/// Parse a **complex** symmetric Matrix Market string (`mtype 6`). Data lines
/// carry two value tokens, the real and imaginary parts: `i j re im`.
pub fn parse_mtx_complex(
    contents: &str,
    source: &str,
) -> Result<MtxMatrix<Complex<f64>>, RslabError> {
    parse_mtx_with(contents, source, "complex", "symmetric", 2, |toks| {
        let re = toks[0].parse::<f64>().ok()?;
        let im = toks[1].parse::<f64>().ok()?;
        Some(Complex::new(re, im))
    })
}

/// Parse a **complex general** Matrix Market string (both triangles stored, e.g.
/// MoM near-field matrices written as `general`). Entries are kept as given
/// (0-indexed, no lower-triangle folding); the caller decides how to treat
/// symmetry (e.g. extract the lower triangle for the symmetric solver, or feed
/// the full pattern to a general LU). Data lines are `i j re im`.
pub fn parse_mtx_complex_general(
    contents: &str,
    source: &str,
) -> Result<MtxMatrix<Complex<f64>>, RslabError> {
    parse_mtx_with(contents, source, "complex", "general", 2, |toks| {
        let re = toks[0].parse::<f64>().ok()?;
        let im = toks[1].parse::<f64>().ok()?;
        Some(Complex::new(re, im))
    })
}

/// Generic Matrix Market coordinate parser. `field` is the expected header
/// field token (`"real"` / `"complex"`); `symmetry` is the structure token
/// (`"symmetric"` folds upper-triangle entries to the lower triangle;
/// `"general"` keeps every entry as given). `n_value_tokens` is how many
/// whitespace tokens the value spans (1 real, 2 complex); `parse_val` builds the
/// scalar. All the hardening (banner tokenization, untrusted `nnz` clamp +
/// validation, bounds, non-finite rejection) is shared.
fn parse_mtx_with<T: Scalar>(
    contents: &str,
    source: &str,
    field: &str,
    symmetry: &str,
    n_value_tokens: usize,
    parse_val: impl Fn(&[&str]) -> Option<T>,
) -> Result<MtxMatrix<T>, RslabError> {
    let fold_to_lower = symmetry == "symmetric";
    let mut lines = contents.lines().enumerate();

    // Header line
    let (_, header) = lines
        .next()
        .ok_or_else(|| RslabError::IoError(format!("{}: empty file", source)))?;
    // X11: compare the banner token by token (case-insensitive) rather than
    // against one exact single-space string. The Matrix Market spec and the
    // reference NIST `mmio` reader tokenize the banner on arbitrary
    // whitespace, so a legal banner whose fields are separated by multiple
    // spaces or tabs must be accepted, not rejected as "unsupported header".
    let banner: [&str; 5] = ["%%matrixmarket", "matrix", "coordinate", field, symmetry];
    let mut header_tokens = header.split_whitespace();
    let banner_ok = banner.iter().all(|expected| {
        header_tokens
            .next()
            .is_some_and(|tok| tok.eq_ignore_ascii_case(expected))
    }) && header_tokens.next().is_none();
    if !banner_ok {
        return Err(RslabError::IoError(format!(
            "{}: unsupported header '{}' (expected: %%MatrixMarket matrix coordinate {} {})",
            source,
            header.trim(),
            field,
            symmetry
        )));
    }

    // Skip comment lines (start with %)
    let mut size_line: Option<(usize, String)> = None;
    for (line_no, line) in &mut lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        size_line = Some((line_no, trimmed.to_string()));
        break;
    }

    let (size_line_no, size_text) =
        size_line.ok_or_else(|| RslabError::IoError(format!("{}: missing size line", source)))?;

    // Parse "rows cols nnz"
    let parts: Vec<&str> = size_text.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(RslabError::IoError(format!(
            "{}: line {}: expected 'rows cols nnz', got '{}'",
            source,
            size_line_no + 1,
            size_text
        )));
    }
    let rows: usize = parts[0].parse().map_err(|_| {
        RslabError::IoError(format!(
            "{}: line {}: invalid row count '{}'",
            source,
            size_line_no + 1,
            parts[0]
        ))
    })?;
    let cols: usize = parts[1].parse().map_err(|_| {
        RslabError::IoError(format!(
            "{}: line {}: invalid col count '{}'",
            source,
            size_line_no + 1,
            parts[1]
        ))
    })?;
    let nnz: usize = parts[2].parse().map_err(|_| {
        RslabError::IoError(format!(
            "{}: line {}: invalid nnz '{}'",
            source,
            size_line_no + 1,
            parts[2]
        ))
    })?;

    if rows != cols {
        return Err(RslabError::IoError(format!(
            "{}: symmetric matrix must be square, got {}x{}",
            source, rows, cols
        )));
    }
    let n = rows;

    // Parse entries.
    //
    // X10: do not trust the header's `nnz` for the up-front allocation. A
    // corrupt header (e.g. nnz = 10^17) would make this a multi-exabyte
    // request; the allocator returns null and `handle_alloc_error` aborts
    // the process instead of returning an `Err`. `nnz` is only a hint here
    // (never validated against the actual entry count), so clamp the
    // reservation to the source byte length - each entry occupies at least
    // one byte in the file, so that is a hard upper bound on the true
    // count. Valid files (where `nnz <= contents.len()` always holds) get
    // the same exact reservation as before.
    let mut entries = Vec::with_capacity(nnz.min(contents.len()));
    for (line_no, line) in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let expected_tokens = 2 + n_value_tokens;
        if parts.len() != expected_tokens {
            return Err(RslabError::IoError(format!(
                "{}: line {}: expected 'i j {}', got '{}'",
                source,
                line_no + 1,
                if n_value_tokens == 2 {
                    "re im"
                } else {
                    "value"
                },
                trimmed
            )));
        }
        let i: usize = parts[0].parse().map_err(|_| {
            RslabError::IoError(format!(
                "{}: line {}: invalid row index '{}'",
                source,
                line_no + 1,
                parts[0]
            ))
        })?;
        let j: usize = parts[1].parse().map_err(|_| {
            RslabError::IoError(format!(
                "{}: line {}: invalid col index '{}'",
                source,
                line_no + 1,
                parts[1]
            ))
        })?;
        let v: T = parse_val(&parts[2..]).ok_or_else(|| {
            RslabError::IoError(format!(
                "{}: line {}: invalid value '{}'",
                source,
                line_no + 1,
                parts[2..].join(" ")
            ))
        })?;
        // f64::from_str silently accepts "nan", "inf", "-inf"; such a value
        // would build an MtxMatrix carrying a non-finite entry that poisons any
        // downstream factorization. Reject it (for complex, either part).
        if !v.is_finite() {
            return Err(RslabError::IoError(format!(
                "{}: line {}: non-finite value '{}'",
                source,
                line_no + 1,
                parts[2..].join(" ")
            )));
        }

        // Validate bounds (1-indexed in MTX)
        if i == 0 || j == 0 || i > n || j > n {
            return Err(RslabError::IoError(format!(
                "{}: line {}: index ({}, {}) out of bounds for {}x{} matrix",
                source,
                line_no + 1,
                i,
                j,
                n,
                n
            )));
        }

        // Convert to 0-indexed. For `symmetric` files fold the upper triangle
        // down to the lower (i >= j); for `general` keep the entry as given.
        let (i0, j0) = (i - 1, j - 1);
        if fold_to_lower && i0 < j0 {
            entries.push((j0, i0, v));
        } else {
            entries.push((i0, j0, v));
        }
    }

    // X2 (REG-4, repo-review-2026-06-09-verification.md): the size line's
    // `nnz` field was parsed but used only as an allocation hint (clamped
    // by X10) and never validated against the actual entry count. The
    // Matrix Market spec defines that field as the number of entries that
    // follow, so a body with more or fewer data lines than the header
    // declares is a malformed file - previously it parsed silently into a
    // matrix that did not match its own declaration (a truncated file read
    // as a valid smaller matrix). Validate it now. This also turns the
    // bogus-huge-nnz case (X10) from an `Ok` carrying the unvalidated
    // entries into a recoverable count-mismatch `Err`; the no-abort
    // property X10 protects still holds (the reservation above stays
    // clamped to the source byte length, so no multi-exabyte allocation).
    if entries.len() != nnz {
        return Err(RslabError::IoError(format!(
            "{}: declared nnz {} does not match the {} entries in the file",
            source,
            nnz,
            entries.len()
        )));
    }

    Ok(MtxMatrix { n, entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_symmetric_3x3() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 4
1 1 2.0
2 1 -1.0
2 2 3.0
3 3 1.5
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.n, 3);
        assert_eq!(m.entries.len(), 4);

        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), 2.0);
        assert_eq!(dense.get(1, 0), -1.0);
        assert_eq!(dense.get(0, 1), -1.0); // symmetric
        assert_eq!(dense.get(1, 1), 3.0);
        assert_eq!(dense.get(2, 2), 1.5);
        assert_eq!(dense.get(2, 0), 0.0); // not set
    }

    /// X10 (dev/research/repo-review-2026-06-09.md): the entries Vec was
    /// reserved with `Vec::with_capacity(nnz)` straight from the untrusted
    /// MTX size line. A corrupt header declaring an enormous nnz turns that
    /// into a multi-exabyte allocation request; the allocator returns null
    /// and Rust's `handle_alloc_error` ABORTS the process - a hard crash,
    /// not a recoverable `RslabError`, on malformed input a library caller
    /// cannot guard against. The reservation is clamped to the source byte
    /// length - a hard upper bound on the true entry count - so the parse
    /// runs to completion instead of aborting.
    ///
    /// Revised for X2 (REG-4): the declared nnz is now also *validated*
    /// against the real entry count, so this file (10^17 declared, two
    /// real) returns a recoverable count-mismatch `Err` rather than an `Ok`
    /// carrying the unvalidated entries. The property X10 protects - a
    /// bogus huge nnz must not *abort the process* - still holds: the call
    /// RETURNS an `Err`, which it could only do by surviving the clamped
    /// `with_capacity` (an unclamped reservation would abort here before any
    /// count check could run).
    #[test]
    fn huge_nnz_header_does_not_abort() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 100000000000000000
1 1 2.0
2 1 -1.0
";
        let err = parse_mtx(mtx, "test")
            .expect_err("a bogus huge nnz must return a recoverable Err, not abort the process");
        match err {
            RslabError::IoError(msg) => assert!(
                msg.contains("declared nnz") && msg.contains("does not match"),
                "expected an nnz-count-mismatch error, got: {msg}"
            ),
            other => panic!("expected RslabError::IoError, got {other:?}"),
        }
    }

    /// X2 / REG-4 (repo-review-2026-06-09-verification.md): the size line's
    /// declared nnz was parsed but never checked against the actual number
    /// of data lines, so a truncated or corrupt file parsed silently into a
    /// matrix that did not match its own declaration. The count is now
    /// validated. Oracle: the Matrix Market spec - the size line's third
    /// field is the number of entries that follow. Here the header declares
    /// 4 entries but the body has 2; pre-fix this returned `Ok` with
    /// `entries.len() == 2`, post-fix it is rejected.
    #[test]
    fn declared_nnz_must_match_entry_count() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 4
1 1 2.0
2 2 3.0
";
        let err = parse_mtx(mtx, "test").expect_err(
            "a declared nnz that disagrees with the actual entry count must be rejected (X2)",
        );
        match err {
            RslabError::IoError(msg) => assert!(
                msg.contains("declared nnz") && msg.contains("does not match"),
                "expected an nnz-count-mismatch error, got: {msg}"
            ),
            other => panic!("expected RslabError::IoError, got {other:?}"),
        }
    }

    /// X2 / REG-4: duplicate coordinates must be summed by BOTH conversion
    /// paths. `to_csc` summed them (via `from_triplets`) while `to_dense`
    /// overwrote them (last-wins), so the same file produced two different
    /// matrices. Oracle: the Matrix Market / COO duplicate-summing
    /// convention (scipy, MATLAB). Here (1,1) is listed as 2.0 then 1.0;
    /// pre-fix `to_dense.get(0,0)` was 1.0 while `to_csc` summed to 3.0,
    /// post-fix both are 3.0.
    #[test]
    fn duplicate_entries_summed_consistently_by_both_paths() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 3
1 1 2.0
1 1 1.0
2 2 5.0
";
        let m = parse_mtx(mtx, "test").expect("valid file with a duplicate coordinate");

        let dense = m.to_dense();
        assert_eq!(
            dense.get(0, 0),
            3.0,
            "to_dense must sum duplicate (1,1) = 2.0 + 1.0"
        );
        assert_eq!(dense.get(1, 1), 5.0);

        // to_csc collapses the duplicate to a single summed entry too.
        let csc = m.to_csc().expect("to_csc");
        // Column 0 holds (row 0, col 0); find its value.
        let mut v00 = None;
        for k in csc.col_ptr[0]..csc.col_ptr[1] {
            if csc.row_idx[k] == 0 {
                v00 = Some(csc.values[k]);
            }
        }
        assert_eq!(
            v00,
            Some(3.0),
            "to_csc must sum duplicate (1,1); both paths must agree"
        );
    }

    /// X11 (dev/research/repo-review-2026-06-09.md): the banner was compared
    /// against the exact single-space string. A legal MTX banner separates
    /// its five fields with arbitrary whitespace; the NIST `mmio` reference
    /// tokenizes it. Pre-fix this multi-space banner failed the exact-string
    /// compare and returned "unsupported header"; post-fix the token-by-token
    /// compare accepts it.
    #[test]
    fn multispace_banner_is_accepted() {
        let mtx = "\
%%MatrixMarket   matrix    coordinate real   symmetric
2 2 2
1 1 4.0
2 2 5.0
";
        let m = parse_mtx(mtx, "test")
            .expect("a banner separated by multiple spaces is legal and must parse (X11)");
        assert_eq!(m.n, 2);
        assert_eq!(m.entries.len(), 2);
    }

    /// X11: a banner whose fields are separated by tabs is equally legal.
    /// Pre-fix the exact-string compare rejected it; post-fix it parses.
    #[test]
    fn tab_separated_banner_is_accepted() {
        let mtx = "%%MatrixMarket\tmatrix\tcoordinate\treal\tsymmetric\n2 2 1\n1 1 7.0\n";
        let m =
            parse_mtx(mtx, "test").expect("a tab-separated banner is legal and must parse (X11)");
        assert_eq!(m.n, 2);
        assert_eq!(m.entries.len(), 1);
    }

    /// X11: `f64::from_str` accepts "nan", so pre-fix this file parsed to an
    /// `Ok(MtxMatrix)` carrying a NaN entry that silently poisons any
    /// downstream factorization. Post-fix a non-finite value is rejected.
    #[test]
    fn nan_value_is_rejected() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 1 nan
";
        let err = parse_mtx(mtx, "test")
            .expect_err("a NaN entry value must be rejected, not read through (X11)");
        match err {
            RslabError::IoError(msg) => assert!(
                msg.contains("non-finite"),
                "expected a non-finite error, got: {msg}"
            ),
            other => panic!("expected RslabError::IoError, got {other:?}"),
        }
    }

    /// X11: likewise "inf" is accepted by `f64::from_str`. Pre-fix it built an
    /// `MtxMatrix` carrying +inf; post-fix it is rejected.
    #[test]
    fn inf_value_is_rejected() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 1 inf
";
        let err = parse_mtx(mtx, "test")
            .expect_err("an inf entry value must be rejected, not read through (X11)");
        match err {
            RslabError::IoError(msg) => assert!(
                msg.contains("non-finite"),
                "expected a non-finite error, got: {msg}"
            ),
            other => panic!("expected RslabError::IoError, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_with_comments() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
% This is a comment
% Another comment
2 2 1
1 1 5.0
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.n, 2);
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0], (0, 0, 5.0));
    }

    #[test]
    fn test_parse_scientific_notation() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 2
1 1 1.23456789012345678e+02
2 1 -9.87654321098765432e-03
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.entries.len(), 2);
        assert!((m.entries[0].2 - 123.456_789_012_345_68).abs() < 1e-10);
        assert!((m.entries[1].2 - (-0.009_876_543_210_987_654)).abs() < 1e-16);
    }

    #[test]
    fn test_upper_triangle_normalized() {
        // Entry (1,2) with 1 < 2 should be flipped to (1,0) in 0-indexed
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 2 7.0
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0], (1, 0, 7.0)); // normalized to lower triangle
    }

    #[test]
    fn test_reject_general_format() {
        let mtx = "\
%%MatrixMarket matrix coordinate real general
2 2 1
1 1 1.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported header"), "got: {}", msg);
    }

    #[test]
    fn test_real_parser_rejects_complex_header() {
        // The real parser must still reject a complex file (use parse_mtx_complex).
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
2 2 1
1 1 1.0 0.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported header"), "got: {}", msg);
    }

    #[test]
    fn complex_parser_reads_complex_symmetric() {
        // mtype 6: `i j re im`. Upper-triangle entry normalized to lower.
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
3 3 4
1 1 2.0 1.0
2 1 -1.0 0.3
2 2 3.0 -0.5
1 3 0.5 0.2
";
        let m = parse_mtx_complex(mtx, "test").unwrap();
        assert_eq!(m.n, 3);
        assert_eq!(m.entries.len(), 4);
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), Complex::new(2.0, 1.0));
        assert_eq!(dense.get(1, 0), Complex::new(-1.0, 0.3));
        assert_eq!(dense.get(0, 1), Complex::new(-1.0, 0.3)); // symmetric (no conj)
                                                              // (1,3) → normalized to lower (2,0).
        assert_eq!(dense.get(2, 0), Complex::new(0.5, 0.2));
    }

    #[test]
    fn complex_parser_rejects_real_header() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
1 1 1.0
";
        let err = parse_mtx_complex(mtx, "test").unwrap_err();
        assert!(format!("{}", err).contains("unsupported header"));
    }

    #[test]
    fn complex_parser_rejects_wrong_token_count() {
        // A complex line needs two value tokens; one is malformed.
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
2 2 1
1 1 1.0
";
        let err = parse_mtx_complex(mtx, "test").unwrap_err();
        assert!(format!("{}", err).contains("expected 'i j re im'"));
    }

    #[test]
    fn complex_parser_rejects_non_finite() {
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
2 2 1
1 1 1.0 nan
";
        let err = parse_mtx_complex(mtx, "test").unwrap_err();
        assert!(format!("{}", err).contains("non-finite"));
    }

    #[test]
    fn complex_mtx_round_trips_through_solver() {
        use crate::LdltSolver;
        // A small diagonally-dominant complex-symmetric system read from MTX,
        // factored and solved - the end-to-end PARDISO-style path.
        let mtx = "\
%%MatrixMarket matrix coordinate complex symmetric
3 3 5
1 1 4.0 1.0
2 1 -1.0 0.2
2 2 4.0 1.0
3 2 -1.0 0.2
3 3 4.0 1.0
";
        let a = parse_mtx_complex(mtx, "test").unwrap().to_csc().unwrap();
        let b = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.5),
        ];
        let solver = LdltSolver::factor(&a).unwrap();
        let x = solver.solve(&b).unwrap();
        let mut ax = vec![Complex::new(0.0, 0.0); 3];
        a.symv(&x, &mut ax);
        let res = (0..3).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-12, "residual {}", res);
    }

    #[test]
    fn test_reject_array_format() {
        let mtx = "\
%%MatrixMarket matrix array real symmetric
2 2
1.0
2.0
3.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported header"), "got: {}", msg);
    }

    #[test]
    fn test_reject_nonsquare() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 4 1
1 1 1.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("square"), "got: {}", msg);
    }

    #[test]
    fn test_reject_out_of_bounds() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 1
3 1 1.0
";
        let err = parse_mtx(mtx, "test").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("out of bounds"), "got: {}", msg);
    }

    #[test]
    fn test_empty_matrix() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 0
";
        let m = parse_mtx(mtx, "test").unwrap();
        assert_eq!(m.n, 3);
        assert_eq!(m.entries.len(), 0);

        let dense = m.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(dense.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_diagonal_only() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 3
1 1 1.0
2 2 2.0
3 3 3.0
";
        let m = parse_mtx(mtx, "test").unwrap();
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), 1.0);
        assert_eq!(dense.get(1, 1), 2.0);
        assert_eq!(dense.get(2, 2), 3.0);
        assert_eq!(dense.get(1, 0), 0.0);
    }

    #[test]
    fn test_negative_values() {
        let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 3
1 1 -1.0
2 1 -0.0
2 2 -3.5
";
        let m = parse_mtx(mtx, "test").unwrap();
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), -1.0);
        assert_eq!(dense.get(1, 1), -3.5);
    }
}
