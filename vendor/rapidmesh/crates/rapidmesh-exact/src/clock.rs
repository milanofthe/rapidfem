//! The wall clock of the stage timings. On `wasm32-unknown-unknown` there
//! is no clock (`std::time::Instant::now` panics there), so every duration
//! is zero; everywhere else it is `std::time::Instant`.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub use nowasm::Instant;

#[cfg(target_arch = "wasm32")]
mod nowasm {
    use std::time::Duration;

    /// A point in time that has no clock behind it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Instant;

    impl Instant {
        pub fn now() -> Instant {
            Instant
        }

        pub fn elapsed(&self) -> Duration {
            Duration::ZERO
        }
    }
}
