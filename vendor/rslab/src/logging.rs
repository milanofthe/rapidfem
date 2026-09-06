//! Logging: one sink, one level, no dependencies. Mirrors the fastsim logger.
//!
//! The library is quiet by default (level `Warning`): a setting that the chosen
//! factor path cannot honour, a fallback the solver took on its own, an error
//! it recovered from. `Info` adds one line per analysis, factorization and
//! refactorization with the numbers a solver-in-the-loop user watches (the
//! [`Diagnostics`](crate::Diagnostics) summary: ordering picked, fill, threads,
//! wall time), `Debug` adds the per-solve and per-stage detail.
//!
//! Every record routes through a single [`LogSink`]; the default prints
//! `Info`/`Warning` to stdout and `Error` to stderr, and a host (Python, a UI,
//! a test) installs its own with [`set_sink`] to capture everything in one
//! place. The level comes from `RLA_LOG` (`debug`, `info`, `warning`, `error`,
//! `off`) on first use and from [`set_level`] at run time.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{OnceLock, RwLock};

/// Log levels, ordered; a record is emitted when its level is at or above the
/// active one. Numeric values follow the Python `logging` module.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum LogLevel {
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
    /// Nothing is emitted, errors included.
    Off = 100,
}

impl LogLevel {
    /// Parse a level name (`debug`, `info`, `warning`/`warn`, `error`, `off`/
    /// `none`/`0`), case-insensitively.
    pub fn parse(s: &str) -> Option<LogLevel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "debug" => Some(LogLevel::Debug),
            "info" => Some(LogLevel::Info),
            "warning" | "warn" => Some(LogLevel::Warning),
            "error" => Some(LogLevel::Error),
            "off" | "none" | "0" | "disabled" => Some(LogLevel::Off),
            _ => None,
        }
    }

    /// Tag of the level (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"OFF"`).
    pub fn label(self) -> &'static str {
        match self {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARNING",
            LogLevel::Error => "ERROR",
            LogLevel::Off => "OFF",
        }
    }

    fn from_u8(v: u8) -> LogLevel {
        match v {
            10 => LogLevel::Debug,
            20 => LogLevel::Info,
            30 => LogLevel::Warning,
            40 => LogLevel::Error,
            _ => LogLevel::Off,
        }
    }
}

/// A destination for log records. Install one with [`set_sink`]. The sink
/// receives the level and the bare message; formatting (timestamp, level
/// label) stays with the sink, so a host re-emitting into its own logger does
/// not carry a second timestamp.
pub trait LogSink: Send + Sync {
    fn emit(&self, level: LogLevel, msg: &str);
}

/// The default sink: `HH:MM:SS - LEVEL - msg`, `Info` and `Warning` to stdout
/// (progress, not errors, so a host capturing stderr does not render them as
/// failures), `Error` to stderr.
pub struct DefaultSink;

impl LogSink for DefaultSink {
    fn emit(&self, level: LogLevel, msg: &str) {
        let line = format!("{} - {} - {}", timestamp(), level.label(), msg);
        if level >= LogLevel::Error {
            eprintln!("{line}");
        } else {
            println!("{line}");
        }
    }
}

static SINK: RwLock<Option<Box<dyn LogSink>>> = RwLock::new(None);
/// `0` = not yet initialised from the environment.
static LEVEL: AtomicU8 = AtomicU8::new(0);
static ENV_LEVEL: OnceLock<LogLevel> = OnceLock::new();

fn env_level() -> LogLevel {
    *ENV_LEVEL.get_or_init(|| {
        std::env::var("RLA_LOG")
            .ok()
            .and_then(|s| LogLevel::parse(&s))
            .unwrap_or(LogLevel::Warning)
    })
}

/// The active level (initialised from `RLA_LOG` on first use, default
/// `Warning`).
pub fn level() -> LogLevel {
    match LEVEL.load(Ordering::Relaxed) {
        0 => env_level(),
        v => LogLevel::from_u8(v),
    }
}

/// Set the active level for the whole process.
pub fn set_level(level: LogLevel) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Is a record at `level` currently emitted? Cheap; guard the formatting of
/// expensive messages with it.
#[inline]
pub fn enabled(level: LogLevel) -> bool {
    level != LogLevel::Off && level >= self::level()
}

/// Install a sink for all subsequent records.
pub fn set_sink(sink: Box<dyn LogSink>) {
    *SINK.write().unwrap_or_else(|e| e.into_inner()) = Some(sink);
}

/// Restore the default print sink.
pub fn reset_sink() {
    *SINK.write().unwrap_or_else(|e| e.into_inner()) = None;
}

fn timestamp() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() % 86400;
    format!(
        "{:02}:{:02}:{:02}",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

/// Route a record through the installed sink (or the default one) if its
/// level is active.
pub fn emit(level: LogLevel, msg: &str) {
    if !enabled(level) {
        return;
    }
    let guard = SINK.read().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some(sink) => sink.emit(level, msg),
        None => DefaultSink.emit(level, msg),
    }
}

pub fn debug(msg: &str) {
    emit(LogLevel::Debug, msg);
}
pub fn info(msg: &str) {
    emit(LogLevel::Info, msg);
}
pub fn warn(msg: &str) {
    emit(LogLevel::Warning, msg);
}
pub fn error(msg: &str) {
    emit(LogLevel::Error, msg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct Capture(Arc<Mutex<Vec<(LogLevel, String)>>>);
    impl LogSink for Capture {
        fn emit(&self, level: LogLevel, msg: &str) {
            self.0.lock().unwrap().push((level, msg.to_string()));
        }
    }

    #[test]
    fn parse_levels() {
        assert_eq!(LogLevel::parse("DEBUG"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::parse(" warn "), Some(LogLevel::Warning));
        assert_eq!(LogLevel::parse("off"), Some(LogLevel::Off));
        assert_eq!(LogLevel::parse("loud"), None);
    }

    #[test]
    fn level_gates_and_sink_captures() {
        // Process-global state: this test owns it for its duration.
        let got = Arc::new(Mutex::new(Vec::new()));
        set_sink(Box::new(Capture(got.clone())));
        set_level(LogLevel::Info);
        debug("hidden");
        info("shown");
        warn("also shown");
        set_level(LogLevel::Off);
        error("silenced");
        let v = got.lock().unwrap().clone();
        reset_sink();
        set_level(LogLevel::Warning);
        assert_eq!(v.len(), 2, "{v:?}");
        assert_eq!(v[0].1, "shown");
        assert_eq!(v[1].0, LogLevel::Warning);
    }
}
