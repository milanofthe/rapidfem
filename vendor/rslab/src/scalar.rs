//! The scalar field over which RLA factorizations operate.
//!
//! [`Scalar`] is implemented for [`f64`] (real symmetric LDLᵀ) and
//! [`num_complex::Complex<f64>`] (complex *symmetric* LDLᵀ, PARDISO `mtype 6`).
//!
//! Design notes:
//!
//! * Magnitudes are always returned as `f64`. There is deliberately no
//!   associated `Real` type: every pivot threshold and scaling factor in the
//!   solver is an `f64`, so keeping magnitudes concrete keeps the pivot logic
//!   uniform across the real and complex paths. A future `Complex<f32>` path,
//!   if ever wanted, would revisit this.
//! * The complex path is *symmetric* (A = Aᵀ), **not** Hermitian. On that path
//!   no value is ever conjugated. [`Scalar::conj`] is provided as the identity
//!   for `f64` and as the genuine conjugate for `Complex<f64>` so a future
//!   Hermitian (LDLᴴ) path can reuse the same trait; the current
//!   complex-symmetric kernels simply never call it.

use num_complex::Complex;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A scalar field element supporting the operations the dense and multifrontal
/// numeric kernels require.
pub trait Scalar:
    'static
    + Copy
    + PartialEq
    + Debug
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The additive identity `0`.
    fn zero() -> Self;

    /// The multiplicative identity `1`.
    fn one() -> Self;

    /// Embed a real number into the field (e.g. an `f64` scaling factor).
    fn from_real(r: f64) -> Self;

    /// Euclidean magnitude `|z|`. For `f64` this is the absolute value; for
    /// `Complex<f64>` the modulus `sqrt(re² + im²)`.
    fn magnitude(self) -> f64;

    /// Squared magnitude `|z|²`. Avoids the `sqrt` in [`magnitude`](Self::magnitude)
    /// when only comparisons are needed (the hot path in pivot selection).
    fn magnitude_sq(self) -> f64;

    /// The real part as an `f64`. For real fields this is the value itself; for
    /// complex fields it is `re`. Used for pivot-sign / inertia classification,
    /// which is exact for real symmetric matrices and advisory for complex.
    fn real(self) -> f64;

    /// Complex conjugate. The identity for `f64` and for the
    /// complex-*symmetric* factorization path; meaningful only for a future
    /// Hermitian path.
    fn conj(self) -> Self;

    /// The reciprocal `1/z`. The caller must guarantee `self != 0`.
    fn recip(self) -> Self;

    /// `self * a + b`, using a fused multiply-add where the hardware offers one.
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Whether every component is finite (no `NaN`/`inf`) - used by pivot
    /// health checks.
    fn is_finite(self) -> bool;
}

impl Scalar for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn from_real(r: f64) -> Self {
        r
    }

    #[inline]
    fn magnitude(self) -> f64 {
        self.abs()
    }

    #[inline]
    fn magnitude_sq(self) -> f64 {
        self * self
    }

    #[inline]
    fn real(self) -> f64 {
        self
    }

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }

    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

impl Scalar for Complex<f64> {
    #[inline]
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    #[inline]
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }

    #[inline]
    fn from_real(r: f64) -> Self {
        Complex::new(r, 0.0)
    }

    #[inline]
    fn magnitude(self) -> f64 {
        // `norm` is the Euclidean modulus sqrt(re² + im²), computed via
        // `hypot` to avoid spurious overflow.
        self.norm()
    }

    #[inline]
    fn magnitude_sq(self) -> f64 {
        self.norm_sqr()
    }

    #[inline]
    fn real(self) -> f64 {
        self.re
    }

    #[inline]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }

    #[inline]
    fn recip(self) -> Self {
        // True algebraic reciprocal 1/z = conj(z) / |z|². This is the value
        // used to eliminate with a complex-symmetric pivot; it is NOT a
        // Hermitian operation.
        Complex::inv(&self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // (self·a + b) lowered to four real FMAs - uses the hardware FMA pipes
        // and halves the rounding versus `self * a + b` on `num_complex`.
        let re = f64::mul_add(self.re, a.re, f64::mul_add(-self.im, a.im, b.re));
        let im = f64::mul_add(self.re, a.im, f64::mul_add(self.im, a.re, b.im));
        Complex::new(re, im)
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl Scalar for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }

    #[inline]
    fn from_real(r: f64) -> Self {
        r as f32
    }

    #[inline]
    fn magnitude(self) -> f64 {
        (self as f64).abs()
    }

    #[inline]
    fn magnitude_sq(self) -> f64 {
        let v = self as f64;
        v * v
    }

    #[inline]
    fn real(self) -> f64 {
        self as f64
    }

    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }

    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl Scalar for Complex<f32> {
    #[inline]
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    #[inline]
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }

    #[inline]
    fn from_real(r: f64) -> Self {
        Complex::new(r as f32, 0.0)
    }

    #[inline]
    fn magnitude(self) -> f64 {
        // Compute the modulus in f64 to keep pivot thresholds accurate and
        // avoid f32 overflow on extreme inputs.
        (self.re as f64).hypot(self.im as f64)
    }

    #[inline]
    fn magnitude_sq(self) -> f64 {
        let re = self.re as f64;
        let im = self.im as f64;
        re * re + im * im
    }

    #[inline]
    fn real(self) -> f64 {
        self.re as f64
    }

    #[inline]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }

    #[inline]
    fn recip(self) -> Self {
        Complex::inv(&self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let re = f32::mul_add(self.re, a.re, f32::mul_add(-self.im, a.im, b.re));
        let im = f32::mul_add(self.re, a.im, f32::mul_add(self.im, a.re, b.im));
        Complex::new(re, im)
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracles below are hand-computed: |3+4i| = 5, 1/(1+i) = (1-i)/2, etc.

    #[test]
    fn f64_magnitude_and_recip() {
        assert_eq!((-3.0_f64).magnitude(), 3.0);
        assert_eq!((-3.0_f64).magnitude_sq(), 9.0);
        assert_eq!(4.0_f64.recip(), 0.25);
        assert_eq!(<f64 as Scalar>::from_real(5.0), 5.0);
        assert_eq!(2.0_f64.conj(), 2.0);
        assert_eq!(2.0_f64.mul_add(3.0, 1.0), 7.0);
        assert!(<f64 as Scalar>::one().is_finite());
        assert!(!(1.0_f64 / 0.0).is_finite());
    }

    #[test]
    fn complex_magnitude() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.magnitude(), 5.0);
        assert_eq!(z.magnitude_sq(), 25.0);
    }

    #[test]
    fn complex_recip_is_algebraic_inverse() {
        let z: Complex<f64> = Complex::new(1.0, 1.0);
        // 1/(1+i) = (1-i)/2 = 0.5 - 0.5i
        let r = z.recip();
        assert!((r.re - 0.5).abs() < 1e-15);
        assert!((r.im + 0.5).abs() < 1e-15);
        // z * (1/z) == 1
        let prod = z * r;
        assert!((prod.re - 1.0).abs() < 1e-15);
        assert!(prod.im.abs() < 1e-15);
    }

    #[test]
    fn complex_conj_and_identities() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.conj(), Complex::new(3.0, -4.0));
        assert_eq!(<Complex<f64> as Scalar>::zero(), Complex::new(0.0, 0.0));
        assert_eq!(<Complex<f64> as Scalar>::one(), Complex::new(1.0, 0.0));
        assert_eq!(
            <Complex<f64> as Scalar>::from_real(7.0),
            Complex::new(7.0, 0.0)
        );
    }

    #[test]
    fn complex_mul_add() {
        let z: Complex<f64> = Complex::new(1.0, 1.0);
        // (1+i)*(1+i) + 1 = (1 + 2i - 1) + 1 = 1 + 2i
        let r = z.mul_add(Complex::new(1.0, 1.0), Complex::new(1.0, 0.0));
        assert!((r.re - 1.0).abs() < 1e-15);
        assert!((r.im - 2.0).abs() < 1e-15);
    }

    #[test]
    fn complex_is_finite() {
        assert!(Complex::new(1.0, 2.0).is_finite());
        assert!(!Complex::new(f64::NAN, 0.0).is_finite());
        assert!(!Complex::new(0.0, f64::INFINITY).is_finite());
    }
}
