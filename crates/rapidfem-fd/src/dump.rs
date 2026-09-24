//! Opt-in dump of the linear systems a frequency sweep factors, for solver
//! benchmarks.
//!
//! With `RAPIDFEM_DUMP_DIR=<dir>` set, every frequency of a sweep writes
//! `<dir>/<tag>_f<NN>.sparta`, where `<tag>` is `RAPIDFEM_DUMP_TAG` (default
//! `rapidfem`) and `NN` the frequency index. The files are SPARTA benchmark
//! containers (`SPARTA02`): the lower triangle of the equilibrated
//! complex-symmetric system exactly as the solver receives it, the port
//! right-hand sides, and `k0`. All files of one sweep share their pattern.
//!
//! `RAPIDFEM_DUMP_LIMIT=<k>` ends the process after the first `k` systems are
//! written, before any of them is factored: for collecting the systems of a
//! model too large to sweep in full.
use num_complex::Complex64 as C64;
use std::io::Write;
use std::path::PathBuf;

/// Directory and file tag, or None when dumping is off.
pub(crate) fn target() -> Option<(PathBuf, String)> {
    let dir = std::env::var_os("RAPIDFEM_DUMP_DIR")?;
    let tag = std::env::var("RAPIDFEM_DUMP_TAG").unwrap_or_else(|_| "rapidfem".to_string());
    Some((PathBuf::from(dir), tag))
}

/// Write one system of a sweep.
///
/// - `index`: frequency index within the sweep, used in the file name.
/// - `k0`: free-space wavenumber in rad/m.
/// - `n`, `rows`, `cols`, `vals`: the full `n x n` matrix as COO triplets,
///   both triangles, duplicates allowed (they are summed).
/// - `rhs`: right-hand sides, each of length `n`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_system(
    (dir, tag): &(PathBuf, String),
    index: usize,
    k0: f64,
    n: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[C64],
    rhs: &[Vec<C64>],
) -> Result<(), String> {
    // Lower triangle, column by column, duplicates summed.
    let mut lower: Vec<(usize, usize, C64)> = rows
        .iter()
        .zip(cols)
        .zip(vals)
        .filter(|((r, c), _)| r >= c)
        .map(|((&r, &c), &v)| (c, r, v))
        .collect();
    lower.sort_unstable_by_key(|&(c, r, _)| (c, r));
    let mut indptr = vec![0u64; n + 1];
    let mut indices: Vec<u64> = Vec::with_capacity(lower.len());
    let mut data: Vec<C64> = Vec::with_capacity(lower.len());
    let mut last = None;
    for (c, r, v) in lower {
        if last == Some((c, r)) {
            *data.last_mut().unwrap() += v;
            continue;
        }
        last = Some((c, r));
        indptr[c + 1] += 1;
        indices.push(r as u64);
        data.push(v);
    }
    for c in 0..n {
        indptr[c + 1] += indptr[c];
    }

    std::fs::create_dir_all(dir).map_err(|e| format!("dump dir {}: {e}", dir.display()))?;
    let path = dir.join(format!("{tag}_f{index:02}.sparta"));
    let file = std::fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = std::io::BufWriter::with_capacity(1 << 22, file);
    let complex = |out: &mut std::io::BufWriter<std::fs::File>, v: &C64| -> std::io::Result<()> {
        out.write_all(&v.re.to_le_bytes())?;
        out.write_all(&v.im.to_le_bytes())
    };
    let mut write = || -> std::io::Result<()> {
        out.write_all(b"SPARTA02")?;
        // n, nnz, number of right-hand sides, flags (0: lower triangle, no solution).
        for word in [n as u64, data.len() as u64, rhs.len() as u64, 0] {
            out.write_all(&word.to_le_bytes())?;
        }
        out.write_all(&k0.to_le_bytes())?;
        for word in indptr.iter().chain(&indices) {
            out.write_all(&word.to_le_bytes())?;
        }
        for v in data.iter().chain(rhs.iter().flatten()) {
            complex(&mut out, v)?;
        }
        out.flush()
    };
    write().map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("  dumped {} (n={n}, nnz(lower)={})", path.display(), data.len());
    let limit = std::env::var("RAPIDFEM_DUMP_LIMIT").ok().and_then(|v| v.parse::<usize>().ok());
    if limit.is_some_and(|k| index + 1 >= k) {
        eprintln!("  RAPIDFEM_DUMP_LIMIT reached, stopping");
        std::process::exit(0);
    }
    Ok(())
}
