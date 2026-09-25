//! Monotonic clock for the ordering crates' stats timing, portable to targets
//! without an OS clock (wasm32-unknown-unknown: `std::time::Instant::now()`
//! panics with "time not implemented on this platform", which would trap
//! every symbolic analysis in the browser). Native builds re-export
//! [`std::time::Instant`] unchanged; on wasm32 the instant is inert and every
//! duration reads zero, so `OrderingStats` timings become no-ops instead of
//! trapping.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

/// Inert stand-in for [`std::time::Instant`] on wasm32: no OS clock exists,
/// so every duration reads zero instead of trapping.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    /// The (inert) current instant.
    pub fn now() -> Self {
        Instant
    }
    /// Always [`std::time::Duration::ZERO`] on wasm32.
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
    /// Always [`std::time::Duration::ZERO`] on wasm32.
    pub fn duration_since(&self, _earlier: Instant) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
