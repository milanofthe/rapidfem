//! Phase 2.13b - per-stage symbolic profiler.
//!
//! Mirrors the numeric `Profiler` (`src/numeric/factorize.rs`) but
//! records per-stage timings inside `symbolic_factorize_with_method`
//! instead of per-supernode timings. The diagnostic question this
//! exists to answer is: when symbolic dominates `factor_us` on
//! small-n matrices (KIRBY2_0007: 924 µs / 1159 µs total), which
//! stage carries the constant?
//!
//! Attached to `SupernodeParams::symbolic_profiler` as
//! `Some(Arc<Mutex<SymbolicProfiler>>)`. When `None` the symbolic
//! driver does no timing work - zero overhead.
//!
//! See `dev/research/phase-2.13b-symbolic-profiler.md`.
use std::sync::{Arc, Mutex};

/// One stage timing record. The stage name is `&'static str` because
/// the call sites are fixed at compile time.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StageTiming {
    pub name: &'static str,
    pub us: u64,
}

/// Per-invocation symbolic profiler. Use:
///
/// ```ignore
/// let prof = Arc::new(Mutex::new(SymbolicProfiler::new()));
/// let params = SupernodeParams {
///     symbolic_profiler: Some(prof.clone()),
///     ..Default::default()
/// };
/// let _ = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amd)?;
/// let report = prof.lock().unwrap().report();
/// ```
///
/// A poisoned mutex (only possible if a panic happened while holding
/// the lock) is silently ignored on the recording path: the affected
/// stage sample is dropped and `report()` validation surfaces the gap.
#[derive(Debug, Clone, Default)]
pub struct SymbolicProfiler {
    stages: Vec<StageTiming>,
    total_us: u64,
}

impl SymbolicProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, name: &'static str, us: u64) {
        self.stages.push(StageTiming { name, us });
    }

    pub fn set_total(&mut self, us: u64) {
        self.total_us = us;
    }

    pub fn stages(&self) -> &[StageTiming] {
        &self.stages
    }

    pub fn total_us(&self) -> u64 {
        self.total_us
    }

    pub fn report(&self) -> SymbolicProfileReport {
        let accounted_us: u64 = self.stages.iter().map(|s| s.us).sum();
        let mut warnings: Vec<String> = Vec::new();
        if self.total_us > 0 && accounted_us > self.total_us {
            warnings.push(format!(
                "stage sum ({}) exceeds total ({})",
                accounted_us, self.total_us
            ));
        }
        let stages: Vec<StagePct> = self
            .stages
            .iter()
            .map(|s| StagePct {
                name: s.name,
                us: s.us,
                pct_of_total: if self.total_us > 0 {
                    (s.us as f64) * 100.0 / (self.total_us as f64)
                } else {
                    0.0
                },
            })
            .collect();
        let overhead_pct = if self.total_us > 0 {
            ((self.total_us.saturating_sub(accounted_us)) as f64) * 100.0 / (self.total_us as f64)
        } else {
            0.0
        };
        SymbolicProfileReport {
            total_us: self.total_us,
            accounted_us,
            overhead_pct,
            stages,
            validation_warnings: warnings,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StagePct {
    pub name: &'static str,
    pub us: u64,
    pub pct_of_total: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SymbolicProfileReport {
    pub total_us: u64,
    pub accounted_us: u64,
    /// Fraction of `total_us` not attributed to any stage (struct
    /// build, return-path, etc.). Should be small when all major
    /// stages are instrumented.
    pub overhead_pct: f64,
    pub stages: Vec<StagePct>,
    pub validation_warnings: Vec<String>,
}

/// Convenience: record `stage_name`'s elapsed time into `profiler` if
/// it is `Some`. Used at every stage boundary in
/// `symbolic_factorize_with_method`.
pub fn record_stage(
    profiler: Option<&Arc<Mutex<SymbolicProfiler>>>,
    name: &'static str,
    start: std::time::Instant,
) {
    if let Some(arc) = profiler {
        let us = start.elapsed().as_micros() as u64;
        if let Ok(mut p) = arc.lock() {
            p.record(name, us);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report_is_safe() {
        let p = SymbolicProfiler::new();
        let r = p.report();
        assert_eq!(r.total_us, 0);
        assert_eq!(r.accounted_us, 0);
        assert!(r.stages.is_empty());
        assert!(r.validation_warnings.is_empty());
    }

    #[test]
    fn pct_sums_to_unity_minus_overhead() {
        let mut p = SymbolicProfiler::new();
        p.record("a", 30);
        p.record("b", 50);
        p.set_total(100);
        let r = p.report();
        assert_eq!(r.accounted_us, 80);
        assert!((r.overhead_pct - 20.0).abs() < 1e-9);
        let pct_sum: f64 = r.stages.iter().map(|s| s.pct_of_total).sum();
        assert!((pct_sum - 80.0).abs() < 1e-9);
        assert!(r.validation_warnings.is_empty());
    }

    #[test]
    fn warning_when_stages_exceed_total() {
        let mut p = SymbolicProfiler::new();
        p.record("a", 200);
        p.set_total(100);
        let r = p.report();
        assert!(!r.validation_warnings.is_empty());
    }
}
