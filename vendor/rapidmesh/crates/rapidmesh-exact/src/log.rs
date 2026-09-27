//! Thread-local meshing log: stage timings, statistics, and leveled events.
//!
//! The meshing pipeline (scene assembly -> PLC -> tet mesh -> optimize) runs
//! its stages sequentially on one thread (rayon fan-out happens inside a stage,
//! whose total is recorded on the calling thread). Each stage records its
//! wall-clock duration, key counts, and human-readable events here; the Python
//! binding clears the collector before a mesh and takes the ordered records
//! after, exposing them as `mesh.timings` / `mesh.stats` / `mesh.log`.
//!
//! Events are also printed live to stderr (with an elapsed-time prefix) when the
//! log level is at or below their severity. The level is a fastsim-style
//! threshold (`Debug < Info < Warn < Error`): set `RAPIDMESH_LOG` to
//! `debug`/`info`/`warn`/`error` (or `1`/`true` = info, unset/`0`/`off` = silent),
//! or call [`set_level`] / [`set_verbose`] from the host. So a user can watch the
//! mesher's stages, metrics, and warnings as they happen -- not just a summary
//! after it finishes.

use crate::clock::Instant;
use std::cell::RefCell;

/// Severity of a log event, ordered `Debug < Info < Warn < Error` so a single
/// threshold filters the live output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Fine-grained detail (per-pass counts, inner-loop traces).
    Debug,
    /// Normal progress (stages, metrics).
    Info,
    /// Something unexpected but recovered (divergence backstop, budget cap).
    Warn,
    /// A failure (recorded before a panic, so the log explains the abort).
    Error,
}

impl Level {
    /// Uppercase tag for the live console line.
    pub fn tag(self) -> &'static str {
        match self {
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
        }
    }

    /// Lowercase tag for the Python API.
    pub fn lower(self) -> &'static str {
        match self {
            Level::Debug => "debug",
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
        }
    }

    /// Parses a level name (case-insensitive). `1`/`true`/`on` map to `Info`.
    pub fn parse(s: &str) -> Option<Level> {
        match s.trim().to_ascii_lowercase().as_str() {
            "debug" | "trace" => Some(Level::Debug),
            "info" | "1" | "true" | "on" | "yes" => Some(Level::Info),
            "warn" | "warning" => Some(Level::Warn),
            "error" => Some(Level::Error),
            _ => None,
        }
    }
}

/// One leveled log event: severity, stage, message, and seconds since the run
/// started.
#[derive(Debug, Clone)]
pub struct Event {
    /// Severity.
    pub level: Level,
    /// Stage / subsystem the event belongs to (e.g. "mesh.faces").
    pub stage: String,
    /// Human-readable message.
    pub message: String,
    /// Seconds since [`clear`] (the run start).
    pub at: f64,
}

thread_local! {
    static TIMINGS: RefCell<Vec<(String, f64)>> = const { RefCell::new(Vec::new()) };
    static STATS: RefCell<Vec<(String, f64)>> = const { RefCell::new(Vec::new()) };
    static EVENTS: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
    static START: RefCell<Option<Instant>> = const { RefCell::new(None) };
    /// Minimum level to print live; `None` = silent (records still collected).
    static THRESHOLD: RefCell<Option<Level>> = const { RefCell::new(None) };
}

/// Clears all collectors and (re)starts the run clock. The live-print threshold
/// is taken from `RAPIDMESH_LOG` (unless already set higher via [`set_level`]).
pub fn clear() {
    TIMINGS.with(|t| t.borrow_mut().clear());
    STATS.with(|s| s.borrow_mut().clear());
    EVENTS.with(|e| e.borrow_mut().clear());
    START.with(|s| *s.borrow_mut() = Some(Instant::now()));
    if let Some(v) = std::env::var_os("RAPIDMESH_LOG") {
        set_level(Level::parse(&v.to_string_lossy()));
    }
}

/// Sets the live-print threshold: `Some(level)` prints events at or above
/// `level`; `None` is silent.
pub fn set_level(level: Option<Level>) {
    THRESHOLD.with(|t| *t.borrow_mut() = level);
}

/// The current live-print threshold (`None` = silent).
pub fn level() -> Option<Level> {
    THRESHOLD.with(|t| *t.borrow())
}

/// Back-compat toggle: `true` = [`Level::Info`], `false` = silent.
pub fn set_verbose(on: bool) {
    set_level(on.then_some(Level::Info));
}

/// True if any live printing is on.
pub fn is_verbose() -> bool {
    level().is_some()
}

fn elapsed() -> f64 {
    START.with(|s| s.borrow().map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0))
}

/// Records a leveled event (and prints it live when at/above the threshold).
pub fn event(level: Level, stage: &str, message: impl Into<String>) {
    let message = message.into();
    let at = elapsed();
    if THRESHOLD.with(|t| t.borrow().is_some_and(|thr| level >= thr)) {
        eprintln!("[{at:8.3}s {:>5} {stage}] {message}", level.tag());
    }
    EVENTS.with(|e| {
        e.borrow_mut().push(Event {
            level,
            stage: stage.to_string(),
            message,
            at,
        })
    });
}

/// Debug-level event (fine detail; shown only at the `debug` threshold).
pub fn debug(stage: &str, message: impl Into<String>) {
    event(Level::Debug, stage, message);
}

/// Info-level event.
pub fn info(stage: &str, message: impl Into<String>) {
    event(Level::Info, stage, message);
}

/// Warning-level event.
pub fn warn(stage: &str, message: impl Into<String>) {
    event(Level::Warn, stage, message);
}

/// Error-level event (record before a panic so the log explains the abort).
pub fn error(stage: &str, message: impl Into<String>) {
    event(Level::Error, stage, message);
}

/// Records a stage's wall-clock duration in seconds, in call order, and emits
/// an info event so the timing is visible live.
pub fn stage(name: &str, seconds: f64) {
    TIMINGS.with(|t| t.borrow_mut().push((name.to_string(), seconds)));
    info(name, format!("{seconds:.3}s"));
}

/// Records a named statistic (count, size, etc.), in call order.
pub fn stat(name: &str, value: f64) {
    STATS.with(|s| s.borrow_mut().push((name.to_string(), value)));
}

/// Records a mesh metric BOTH as a machine-readable stat (`stage.name`) and a
/// human-readable info line (`name = value unit`), so the important numbers show
/// up live and in `mesh.stats`.
pub fn metric(stage_name: &str, name: &str, value: f64, unit: &str) {
    stat(&format!("{stage_name}.{name}"), value);
    if unit.is_empty() {
        info(stage_name, format!("{name} = {value:.4}"));
    } else {
        info(stage_name, format!("{name} = {value:.4} {unit}"));
    }
}

/// Drains and returns the collected (timings, stats, events), each ordered by
/// record time.
pub fn take() -> (Vec<(String, f64)>, Vec<(String, f64)>, Vec<Event>) {
    let timings = TIMINGS.with(|t| std::mem::take(&mut *t.borrow_mut()));
    let stats = STATS.with(|s| std::mem::take(&mut *s.borrow_mut()));
    let events = EVENTS.with(|e| std::mem::take(&mut *e.borrow_mut()));
    (timings, stats, events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_order_and_parse() {
        assert!(Level::Debug < Level::Info);
        assert!(Level::Info < Level::Warn);
        assert!(Level::Warn < Level::Error);
        assert_eq!(Level::parse("debug"), Some(Level::Debug));
        assert_eq!(Level::parse("1"), Some(Level::Info));
        assert_eq!(Level::parse("WARN"), Some(Level::Warn));
        assert_eq!(Level::parse("off"), None);
        assert_eq!(Level::Warn.tag(), "WARN");
        assert_eq!(Level::Warn.lower(), "warn");
    }

    #[test]
    fn collects_and_filters() {
        clear();
        set_level(Some(Level::Warn));
        stage("mesh.x", 0.5);
        metric("metrics", "tets", 1234.0, "");
        warn("metrics", "a sliver survived");
        let (timings, stats, events) = take();
        assert_eq!(timings, vec![("mesh.x".to_string(), 0.5)]);
        assert_eq!(stats, vec![("metrics.tets".to_string(), 1234.0)]);
        // All three events are collected regardless of the print threshold.
        assert_eq!(events.len(), 3);
        assert_eq!(events[2].level, Level::Warn);
    }
}
