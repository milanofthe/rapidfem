//! Conservative interval arithmetic used as the sign filter.
//!
//! Rust offers no portable access to FPU rounding modes, so instead of
//! directed rounding every inexact operation widens its result outward by one
//! ulp via `f64::next_down`/`next_up`. Default rounding is to-nearest, so the
//! true result lies within half an ulp of the computed bound -- one ulp outward
//! is strictly conservative. Intervals are therefore slightly wider than with
//! directed rounding, costing only a few extra (correct) fallbacks to exact
//! arithmetic.

use crate::ring::Ring;
use crate::Sign;

/// A closed interval [lo, hi] guaranteed to contain the true real value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    lo: f64,
    hi: f64,
}

// add/sub/mul/neg intentionally mirror the Ring trait instead of std::ops,
// matching Expansion (the generic geometric code calls Ring methods).
#[allow(clippy::should_implement_trait)]
impl Interval {
    /// The exact point interval [v, v].
    pub fn point(v: f64) -> Interval {
        Interval { lo: v, hi: v }
    }

    /// Lower bound.
    pub fn lo(&self) -> f64 {
        self.lo
    }

    /// Upper bound.
    pub fn hi(&self) -> f64 {
        self.hi
    }

    /// Sign of the contained value, or `None` if the interval straddles zero.
    pub fn sign(&self) -> Option<Sign> {
        if self.lo > 0.0 {
            Some(Sign::Positive)
        } else if self.hi < 0.0 {
            Some(Sign::Negative)
        } else if self.lo == 0.0 && self.hi == 0.0 {
            Some(Sign::Zero)
        } else {
            None
        }
    }

    /// Conservative sum.
    pub fn add(self, other: Interval) -> Interval {
        Interval {
            lo: (self.lo + other.lo).next_down(),
            hi: (self.hi + other.hi).next_up(),
        }
    }

    /// Exact negation.
    pub fn neg(self) -> Interval {
        Interval {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    /// Conservative difference.
    pub fn sub(self, other: Interval) -> Interval {
        self.add(other.neg())
    }

    /// Conservative product.
    pub fn mul(self, other: Interval) -> Interval {
        let c = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        let mut lo = c[0];
        let mut hi = c[0];
        for &v in &c[1..] {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        Interval {
            lo: lo.next_down(),
            hi: hi.next_up(),
        }
    }
}

impl Interval {
    /// Conservative quotient, `None` when the divisor contains zero.
    pub fn checked_div(self, other: Interval) -> Option<Interval> {
        if !(other.lo > 0.0 || other.hi < 0.0) {
            return None;
        }
        let c = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ];
        let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(Interval {
            lo: lo.next_down(),
            hi: hi.next_up(),
        })
    }
}

impl Ring for Interval {
    fn from_f64(v: f64) -> Self {
        Interval::point(v)
    }
    fn add(&self, other: &Self) -> Self {
        Interval::add(*self, *other)
    }
    fn sub(&self, other: &Self) -> Self {
        Interval::sub(*self, *other)
    }
    fn mul(&self, other: &Self) -> Self {
        Interval::mul(*self, *other)
    }
    fn neg(&self) -> Self {
        Interval::neg(*self)
    }
}
