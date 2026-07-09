//! SuiteSparse / Matrix Market downloader (feature `matgen-download`).
//!
//! Fetches a matrix archive from the SuiteSparse Matrix Collection
//! (`https://sparse.tamu.edu/MM/<group>/<name>.tar.gz`), extracts the `.mtx`, and
//! caches it locally - so benchmarks can sweep **real application matrices**
//! alongside the generated ones (the realism gold standard). Pure Rust:
//! `ureq` + `rustls` (no OpenSSL), `flate2`/`miniz_oxide`, `tar`.
//!
//! Parse the returned file with the crate's Matrix Market readers
//! ([`crate::read_mtx`], [`crate::read_mtx_complex`], …).

use std::path::PathBuf;

/// Local cache directory: `$RLA_MATGEN_CACHE` if set, else a `rla-matgen` folder
/// under the system temp dir.
pub fn cache_dir() -> PathBuf {
    match std::env::var_os("RLA_MATGEN_CACHE") {
        Some(d) => PathBuf::from(d),
        None => std::env::temp_dir().join("rla-matgen"),
    }
}

/// Download `<group>/<name>` from the SuiteSparse collection (or return the cached
/// copy) and return the path to the extracted `<name>.mtx`. Idempotent: a cached
/// file is reused without hitting the network.
pub fn fetch(group: &str, name: &str) -> Result<PathBuf, String> {
    let dir = cache_dir();
    let mtx_path = dir.join(format!("{name}.mtx"));
    if mtx_path.exists() {
        return Ok(mtx_path);
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("create cache dir: {e}"))?;

    let url = format!("https://sparse.tamu.edu/MM/{group}/{name}.tar.gz");
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| format!("download {url}: {e}"))?;
    let gz = flate2::read::GzDecoder::new(resp.into_reader());
    let mut ar = tar::Archive::new(gz);

    let entries = ar.entries().map_err(|e| format!("read archive: {e}"))?;
    for entry in entries {
        let mut e = entry.map_err(|e| format!("archive entry: {e}"))?;
        let path = e
            .path()
            .map_err(|e| format!("entry path: {e}"))?
            .into_owned();
        // The main matrix is `<name>/<name>.mtx`; skip rhs/coord/solution siblings.
        let is_main = path
            .file_name()
            .map(|f| f.to_string_lossy() == format!("{name}.mtx"))
            .unwrap_or(false);
        if is_main {
            let mut out =
                std::fs::File::create(&mtx_path).map_err(|e| format!("write cache: {e}"))?;
            std::io::copy(&mut e, &mut out).map_err(|e| format!("extract mtx: {e}"))?;
            return Ok(mtx_path);
        }
    }
    Err(format!("no {name}.mtx found in archive {url}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// End-to-end network test (run explicitly: `--ignored`). Fetches a small real
    /// SPD matrix and checks it is a valid Matrix Market file.
    #[test]
    #[ignore = "requires network access to sparse.tamu.edu"]
    fn fetch_small_real_matrix() {
        let p = fetch("HB", "bcsstk14").expect("download bcsstk14");
        assert!(p.exists());
        let s = std::fs::read_to_string(&p).expect("read cached mtx");
        assert!(
            s.starts_with("%%MatrixMarket"),
            "valid Matrix Market header"
        );
    }
}

/// A small curated set of real SuiteSparse matrices spanning the benchmark axes
/// (size / symmetry / conditioning / density) - convenient starting points for a
/// "real matrices" sweep. `(group, name, note)`.
pub fn suggested() -> &'static [(&'static str, &'static str, &'static str)] {
    &[
        ("HB", "bcsstk14", "SPD structural, small, moderate κ"),
        ("HB", "bcsstk18", "SPD structural, medium, ill-conditioned"),
        ("Boeing", "bcsstk39", "SPD structural, larger"),
        ("FIDAP", "ex11", "unsymmetric CFD, medium"),
        ("Bai", "qc2534", "complex unsymmetric (H2+ model), QM"),
        (
            "Schenk_ISEI",
            "barrier2-1",
            "unsymmetric semiconductor, large",
        ),
        (
            "GHS_indef",
            "cont-300",
            "symmetric indefinite (KKT/optimization)",
        ),
        ("Williams", "cant", "SPD FEM, large, dense-ish"),
    ]
}
