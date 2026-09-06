//! The scalar field over which RLA factorizations operate.
//!
//! [`Scalar`] is implemented for [`f64`] (real symmetric LDL^T) and
//! [`num_complex::Complex<f64>`] (complex *symmetric* LDL^T, PARDISO `mtype 6`).
//!
//! Design notes:
//!
//! * Magnitudes are always returned as `f64`. There is deliberately no
//!   associated `Real` type: every pivot threshold and scaling factor in the
//!   solver is an `f64`, so keeping magnitudes concrete keeps the pivot logic
//!   uniform across the real and complex paths. A future `Complex<f32>` path,
//!   if ever wanted, would revisit this.
//! * The complex path is *symmetric* (A = A^T), **not** Hermitian. On that path
//!   no value is ever conjugated. [`Scalar::conj`] is provided as the identity
//!   for `f64` and as the genuine conjugate for `Complex<f64>` so a future
//!   Hermitian (LDL^H) path can reuse the same trait; the current
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
    /// `Complex<f64>` the modulus `sqrt(re^2 + im^2)`.
    fn magnitude(self) -> f64;

    /// Squared magnitude `|z|^2`. Avoids the `sqrt` in [`magnitude`](Self::magnitude)
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

    /// Low-precision storage partner (`f64 -> f32`, `c64 -> c32`; the
    /// identity for the already-single types). Powers adaptive-precision
    /// low-rank storage (issue #19): tail vectors whose contribution is
    /// below the compression tolerance are stored in `Lo` at half the
    /// bytes and promoted on read.
    type Lo: Copy + Send + Sync + std::fmt::Debug + 'static;
    /// Whether `Lo` actually halves the storage (`false` for the identity).
    const LO_SHRINKS: bool;
    /// Unit roundoff of `Lo` (the storage-rounding noise level).
    const EPS_LO: f64;
    /// Cast to the low-precision partner.
    fn demote(self) -> Self::Lo;
    /// Cast back up.
    fn promote(lo: Self::Lo) -> Self;

    /// The reciprocal `1/z`. The caller must guarantee `self != 0`.
    fn recip(self) -> Self;

    /// `self * a + b`, using a fused multiply-add where the hardware offers
    /// one. Do **not** call this directly in hot loops: without the `fma`
    /// target feature it lowers to a slow libm software-fma call; go through
    /// [`fmadd`] instead, which guards on the build's target features.
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Whether every component is finite (no `NaN`/`inf`) - used by pivot
    /// health checks.
    fn is_finite(self) -> bool;
}

/// `a*b + c` through the hardware FMA **when the build enables it**
/// (`-C target-cpu=native` or `-C target-feature=+fma`; see
/// `.cargo/config.toml`), else as a plain multiply-add. The guard is
/// load-bearing: a bare [`Scalar::mul_add`] on a baseline x86-64 build lowers
/// to a libm software-fma call, bit-exact but far slower than mul+add, so
/// the hot scalar kernels (triangular solves, Gilbert-Peierls updates) must
/// only ever reach `mul_add` through this switch. `cfg!` resolves at compile
/// time; the untaken branch folds away.
///
/// Numerical note: with FMA the product is not rounded before the add, so
/// results differ from the plain path in the last ulp. Determinism within a
/// build is unaffected (every run takes the same branch); cross-build
/// bit-identity was never guaranteed.
#[inline(always)]
pub(crate) fn fmadd<T: Scalar>(a: T, b: T, c: T) -> T {
    if cfg!(target_feature = "fma") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

impl Scalar for f64 {
    type Lo = f32;
    const LO_SHRINKS: bool = true;
    const EPS_LO: f64 = f32::EPSILON as f64;
    #[inline]
    fn demote(self) -> f32 {
        self as f32
    }
    #[inline]
    fn promote(lo: f32) -> f64 {
        lo as f64
    }

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
    type Lo = Complex<f32>;
    const LO_SHRINKS: bool = true;
    const EPS_LO: f64 = f32::EPSILON as f64;
    #[inline]
    fn demote(self) -> Complex<f32> {
        Complex::new(self.re as f32, self.im as f32)
    }
    #[inline]
    fn promote(lo: Complex<f32>) -> Complex<f64> {
        Complex::new(lo.re as f64, lo.im as f64)
    }

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
        // `norm` is the Euclidean modulus sqrt(re^2 + im^2), computed via
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
        // True algebraic reciprocal 1/z = conj(z) / |z|^2. This is the value
        // used to eliminate with a complex-symmetric pivot; it is NOT a
        // Hermitian operation.
        Complex::inv(&self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // (self*a + b) lowered to four real FMAs - uses the hardware FMA pipes
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
    type Lo = f32;
    const LO_SHRINKS: bool = false;
    const EPS_LO: f64 = f32::EPSILON as f64;
    #[inline]
    fn demote(self) -> f32 {
        self
    }
    #[inline]
    fn promote(lo: f32) -> f32 {
        lo
    }

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
    type Lo = Complex<f32>;
    const LO_SHRINKS: bool = false;
    const EPS_LO: f64 = f32::EPSILON as f64;
    #[inline]
    fn demote(self) -> Complex<f32> {
        self
    }
    #[inline]
    fn promote(lo: Complex<f32>) -> Complex<f32> {
        lo
    }

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
