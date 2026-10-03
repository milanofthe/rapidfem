//! Shewchuk-style floating-point expansion arithmetic.
//!
//! An expansion represents a real number exactly as a sum of f64 components
//! that are nonoverlapping and ordered by increasing magnitude. All operations
//! here are exact (no rounding error in the represented value), which makes
//! sign computation exact.
//!
//! Components live inline up to [`INLINE`] and spill to the heap beyond, so
//! the short expansions of a typical exact predicate never allocate while
//! arbitrary-degree expressions (homogeneous TPI coordinates of degree 7
//! inside a 4x4 determinant) still work.

use crate::ring::Ring;
use crate::Sign;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use smallvec::SmallVec;

/// Components stored inline before an expansion allocates.
const INLINE: usize = 8;
type Comps = SmallVec<[f64; INLINE]>;

/// 2^27 + 1, Shewchuk's splitter for exact two-product.
const SPLITTER: f64 = 134_217_729.0;

/// Exact sum of two f64: returns (approximate sum, roundoff term).
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virt = x - a;
    let a_virt = x - b_virt;
    let b_round = b - b_virt;
    let a_round = a - a_virt;
    (x, a_round + b_round)
}

/// Exact sum of two f64 with |a| >= |b|: returns (approximate sum, roundoff).
#[inline]
pub fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virt = x - a;
    (x, b - b_virt)
}

/// Splits an f64 into high and low halves for exact multiplication.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let a_big = c - a;
    let a_hi = c - a_big;
    (a_hi, a - a_hi)
}

/// Exact product where b is already split.
#[inline]
fn two_product_presplit(a: f64, b: f64, b_hi: f64, b_lo: f64) -> (f64, f64) {
    let x = a * b;
    let (a_hi, a_lo) = split(a);
    let err1 = x - a_hi * b_hi;
    let err2 = err1 - a_lo * b_hi;
    let err3 = err2 - a_hi * b_lo;
    (x, a_lo * b_lo - err3)
}

/// Sums two expansions into `h`, eliminating zero components.
///
/// Inputs must be nonoverlapping expansions sorted by increasing magnitude
/// (the invariant maintained by every routine in this module). The output
/// satisfies the same invariant and is never empty (a zero result is `[0.0]`).
fn fast_expansion_sum_zeroelim(e: &[f64], f: &[f64], h: &mut Comps) {
    h.clear();
    let (elen, flen) = (e.len(), f.len());
    let mut eindex = 0usize;
    let mut findex = 0usize;
    let mut enow = e[0];
    let mut fnow = f[0];
    // Pick the smaller-magnitude head as the initial accumulator.
    let mut q;
    if (fnow > enow) == (fnow > -enow) {
        q = enow;
        eindex += 1;
        if eindex < elen {
            enow = e[eindex];
        }
    } else {
        q = fnow;
        findex += 1;
        if findex < flen {
            fnow = f[findex];
        }
    }
    if eindex < elen && findex < flen {
        // First merge step may use fast_two_sum (new head dominates q).
        let (qnew, hh);
        if (fnow > enow) == (fnow > -enow) {
            let r = fast_two_sum(enow, q);
            qnew = r.0;
            hh = r.1;
            eindex += 1;
            if eindex < elen {
                enow = e[eindex];
            }
        } else {
            let r = fast_two_sum(fnow, q);
            qnew = r.0;
            hh = r.1;
            findex += 1;
            if findex < flen {
                fnow = f[findex];
            }
        }
        q = qnew;
        if hh != 0.0 {
            h.push(hh);
        }
        while eindex < elen && findex < flen {
            let (qnew, hh);
            if (fnow > enow) == (fnow > -enow) {
                let r = two_sum(q, enow);
                qnew = r.0;
                hh = r.1;
                eindex += 1;
                if eindex < elen {
                    enow = e[eindex];
                }
            } else {
                let r = two_sum(q, fnow);
                qnew = r.0;
                hh = r.1;
                findex += 1;
                if findex < flen {
                    fnow = f[findex];
                }
            }
            q = qnew;
            if hh != 0.0 {
                h.push(hh);
            }
        }
    }
    while eindex < elen {
        let r = two_sum(q, e[eindex]);
        q = r.0;
        if r.1 != 0.0 {
            h.push(r.1);
        }
        eindex += 1;
    }
    while findex < flen {
        let r = two_sum(q, f[findex]);
        q = r.0;
        if r.1 != 0.0 {
            h.push(r.1);
        }
        findex += 1;
    }
    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
}

/// Multiplies expansion `e` by scalar `b` into `h`, eliminating zeros.
fn scale_expansion_zeroelim(e: &[f64], b: f64, h: &mut Comps) {
    h.clear();
    let (b_hi, b_lo) = split(b);
    let (mut q, hh) = two_product_presplit(e[0], b, b_hi, b_lo);
    if hh != 0.0 {
        h.push(hh);
    }
    for &ei in &e[1..] {
        let (p1, p0) = two_product_presplit(ei, b, b_hi, b_lo);
        let (sum, hh1) = two_sum(q, p0);
        if hh1 != 0.0 {
            h.push(hh1);
        }
        let (qn, hh2) = fast_two_sum(p1, sum);
        q = qn;
        if hh2 != 0.0 {
            h.push(hh2);
        }
    }
    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
}

/// An exact real number: a nonoverlapping sum of f64 components, or a
/// big dyadic number where f64 components would no longer be exact.
///
/// The component arithmetic is exact only while every product and its
/// roundoff term stay normal floats: implicit points of a high degree,
/// built from coordinates like 1.5e-16 (sin(pi) rounded), reach products
/// far below the subnormal range, where Shewchuk's two-product silently
/// drops bits (#90). Every operation checks its operands' extreme
/// components first (O(1), they are sorted by magnitude) and continues in
/// [`Big`] when a product could leave the safe exponent range, so signs
/// stay exact for every finite input at the cost of big-integer arithmetic
/// in those rare cases only.
///
/// Invariants of the component form: nonoverlapping, sorted by increasing
/// magnitude, never empty, and only a zero expansion contains a zero
/// component (exactly `[0.0]`).
#[derive(Debug, Clone)]
pub struct Expansion(Repr);

#[derive(Debug, Clone)]
enum Repr {
    Comps(Comps),
    Big(Big),
}

/// Components at most this large keep Shewchuk's split and every sum
/// finite.
const SAFE_MAX: f64 = 1.0e290; // < 2^996
/// Products at least this large (in magnitude) keep their roundoff terms,
/// about 2^-106 of them, above the normal range (2^-860 ~ 1.4e-259).
const SAFE_PRODUCT_MIN: f64 = 1.0e-250;
/// Products at most this large stay finite.
const SAFE_PRODUCT_MAX: f64 = 1.0e290;

/// True if the component product of `a` and `b` is exact: every pairwise
/// product within the safe range. Components are sorted by magnitude, so
/// the extremes are the first and last ones.
fn product_safe(a: &[f64], b: &[f64]) -> bool {
    let (alo, ahi) = (a[0].abs(), a[a.len() - 1].abs());
    let (blo, bhi) = (b[0].abs(), b[b.len() - 1].abs());
    ahi <= SAFE_MAX
        && bhi <= SAFE_MAX
        && alo * blo >= SAFE_PRODUCT_MIN
        && ahi * bhi <= SAFE_PRODUCT_MAX
}

// add/sub/mul/neg intentionally mirror the Ring trait instead of std::ops:
// the generic geometric code calls Ring methods, and by-value std operators
// on a heap-spilling type would invite accidental clones.
#[allow(clippy::should_implement_trait)]
impl Expansion {
    /// The exact value `v` (finite).
    pub fn from_f64(v: f64) -> Expansion {
        if v.abs() > SAFE_MAX {
            return Expansion(Repr::Big(Big::from_f64(v)));
        }
        let mut c = Comps::new();
        c.push(v);
        Expansion(Repr::Comps(c))
    }

    fn big(&self) -> Big {
        match &self.0 {
            Repr::Comps(c) => c
                .iter()
                .fold(Big::zero(), |acc, &x| acc.add(&Big::from_f64(x))),
            Repr::Big(b) => b.clone(),
        }
    }

    /// The exact value as `(m, e)` with value `m * 2^e`.
    pub fn to_dyadic(&self) -> (BigInt, i64) {
        let b = self.big();
        (b.m, b.e)
    }

    /// True if the represented value is exactly zero.
    pub fn is_zero(&self) -> bool {
        match &self.0 {
            Repr::Comps(c) => c.len() == 1 && c[0] == 0.0,
            Repr::Big(b) => b.m.is_zero(),
        }
    }

    /// Exact sign of the represented value (sign of the largest component).
    pub fn sign(&self) -> Sign {
        match &self.0 {
            Repr::Comps(c) => Sign::of_f64(*c.last().expect("expansion is never empty")),
            Repr::Big(b) => match b.m.sign() {
                num_bigint::Sign::Minus => Sign::Negative,
                num_bigint::Sign::NoSign => Sign::Zero,
                num_bigint::Sign::Plus => Sign::Positive,
            },
        }
    }

    /// Approximate f64 value (sum of components, smallest first).
    pub fn approx(&self) -> f64 {
        match &self.0 {
            Repr::Comps(c) => c.iter().sum(),
            Repr::Big(b) => b.approx(),
        }
    }

    /// Approximate quotient `self / w` (`w` nonzero), also where both lie
    /// outside the float range.
    pub fn ratio_approx(&self, w: &Expansion) -> f64 {
        if let (Repr::Comps(_), Repr::Comps(_)) = (&self.0, &w.0) {
            let (a, b) = (self.approx(), w.approx());
            let q = a / b;
            if b.is_normal() && (q.is_normal() || a == 0.0) {
                return q;
            }
        }
        let (a, b) = (self.big(), w.big());
        let (ma, ea) = a.top();
        let (mb, eb) = b.top();
        ldexp(ma / mb, ea - eb)
    }

    /// Exact sum.
    pub fn add(&self, other: &Expansion) -> Expansion {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        match (&self.0, &other.0) {
            (Repr::Comps(e), Repr::Comps(f)) => {
                let mut h = Comps::with_capacity(e.len() + f.len());
                fast_expansion_sum_zeroelim(e, f, &mut h);
                Expansion(Repr::Comps(h))
            }
            _ => Expansion(Repr::Big(self.big().add(&other.big()))),
        }
    }

    /// Exact difference.
    pub fn sub(&self, other: &Expansion) -> Expansion {
        self.add(&other.neg())
    }

    /// Exact product with a scalar.
    pub fn scale(&self, b: f64) -> Expansion {
        if self.is_zero() || b == 0.0 {
            return Expansion::from_f64(0.0);
        }
        match &self.0 {
            Repr::Comps(e) if product_safe(e, &[b]) => {
                let mut h = Comps::with_capacity(2 * e.len());
                scale_expansion_zeroelim(e, b, &mut h);
                Expansion(Repr::Comps(h))
            }
            _ => Expansion(Repr::Big(self.big().mul(&Big::from_f64(b)))),
        }
    }

    /// Exact product (distributes `other`'s components over `self`).
    pub fn mul(&self, other: &Expansion) -> Expansion {
        if self.is_zero() || other.is_zero() {
            return Expansion::from_f64(0.0);
        }
        let (a, b) = match (&self.0, &other.0) {
            (Repr::Comps(a), Repr::Comps(b)) if product_safe(a, b) => (a, b),
            _ => return Expansion(Repr::Big(self.big().mul(&other.big()))),
        };
        // Scale the longer operand by each component of the shorter one,
        // accumulating through two reused buffers.
        let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let mut acc = Comps::new();
        scale_expansion_zeroelim(long, short[0], &mut acc);
        let (mut term, mut next) = (Comps::new(), Comps::new());
        for &fi in &short[1..] {
            scale_expansion_zeroelim(long, fi, &mut term);
            fast_expansion_sum_zeroelim(&acc, &term, &mut next);
            std::mem::swap(&mut acc, &mut next);
        }
        Expansion(Repr::Comps(acc))
    }

    /// Exact negation.
    pub fn neg(&self) -> Expansion {
        match &self.0 {
            Repr::Comps(c) => Expansion(Repr::Comps(c.iter().map(|c| -c).collect())),
            Repr::Big(b) => Expansion(Repr::Big(Big {
                m: -b.m.clone(),
                e: b.e,
            })),
        }
    }
}

impl PartialEq for Expansion {
    /// Equal values (whatever the representation).
    fn eq(&self, other: &Expansion) -> bool {
        self.sub(other).is_zero()
    }
}

/// An exact dyadic number `m * 2^e`, `m` odd unless zero (then `e` = 0):
/// no exponent range, no rounding.
#[derive(Debug, Clone)]
struct Big {
    m: BigInt,
    e: i64,
}

impl Big {
    fn zero() -> Big {
        Big {
            m: BigInt::zero(),
            e: 0,
        }
    }

    fn norm(m: BigInt, e: i64) -> Big {
        match m.trailing_zeros() {
            None => Big::zero(),
            Some(z) => Big {
                m: m >> z,
                e: e + z as i64,
            },
        }
    }

    /// The exact value of a finite f64.
    fn from_f64(v: f64) -> Big {
        let bits = v.to_bits();
        let exp = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & ((1u64 << 52) - 1);
        let (mag, e) = if exp == 0 {
            (frac, -1074)
        } else {
            (frac | (1u64 << 52), exp - 1075)
        };
        let m = BigInt::from(mag);
        Big::norm(if v < 0.0 { -m } else { m }, e)
    }

    fn add(&self, other: &Big) -> Big {
        if self.m.is_zero() {
            return other.clone();
        }
        if other.m.is_zero() {
            return self.clone();
        }
        let e = self.e.min(other.e);
        let a = &self.m << (self.e - e) as usize;
        let b = &other.m << (other.e - e) as usize;
        Big::norm(a + b, e)
    }

    fn mul(&self, other: &Big) -> Big {
        Big::norm(&self.m * &other.m, self.e + other.e)
    }

    /// The value as `t * 2^k` with `t` the top 63 bits of the mantissa.
    fn top(&self) -> (f64, i64) {
        let bits = self.m.bits() as i64;
        let shift = (bits - 63).max(0);
        let t = (&self.m >> shift as usize).to_f64().unwrap_or(0.0);
        (t, self.e + shift)
    }

    /// The nearest-ish f64 (top 63 bits of the mantissa, scaled).
    fn approx(&self) -> f64 {
        let (t, k) = self.top();
        ldexp(t, k)
    }
}

/// `x * 2^k` without intermediate overflow of the power.
fn ldexp(mut x: f64, mut k: i64) -> f64 {
    while k > 1000 {
        x *= 2f64.powi(1000);
        k -= 1000;
    }
    while k < -1000 {
        x *= 2f64.powi(-1000);
        k += 1000;
    }
    x * 2f64.powi(k as i32)
}

/// Correctly rounded quotient of two exact expansions (`w` nonzero): the
/// f64 nearest to the exact value x/w, ties to even. Faithful float division
/// is NOT enough where geometry relies on coordinates landing exactly on
/// values they exactly equal (a constructed point on the plane z = 0.015
/// must approximate to 0.015 bit-exactly, or planarity shatters downstream).
pub fn div_round(x: &Expansion, w: &Expansion) -> f64 {
    debug_assert!(w.sign() != Sign::Zero, "division by zero expansion");
    // Newton refinement with exact residuals: q converges to within an ulp.
    let mut q = x.ratio_approx(w);
    if !q.is_finite() {
        q = 0.0;
    }
    for _ in 0..3 {
        let r = x.add(&w.scale(q).neg());
        let dq = r.ratio_approx(w);
        if dq == 0.0 {
            break;
        }
        let q2 = q + dq;
        if q2 == q || !q2.is_finite() {
            break;
        }
        q = q2;
    }
    // Exact placement: walk to the neighbor while the exact value lies
    // beyond the midpoint between q and that neighbor.
    let v_minus_q_sign = |q: f64| -> Sign {
        // sign(x - q w) * sign(w) = sign(v - q)
        let r = x.add(&w.scale(q).neg());
        match (r.sign(), w.sign()) {
            (Sign::Zero, _) => Sign::Zero,
            (a, b) if a == b => Sign::Positive,
            _ => Sign::Negative,
        }
    };
    let beyond_mid = |a: f64, b: f64| -> Sign {
        // sign(2v - (a + b)) relative: sign(2x - (a+b)w) * sign(w),
        // computed without forming a+b in f64.
        let m = x.add(x).add(&w.scale(a).neg()).add(&w.scale(b).neg());
        match (m.sign(), w.sign()) {
            (Sign::Zero, _) => Sign::Zero,
            (s, t) if s == t => Sign::Positive,
            _ => Sign::Negative,
        }
    };
    loop {
        match v_minus_q_sign(q) {
            Sign::Zero => return q + 0.0, // normalize -0.0 to +0.0
            Sign::Positive => {
                let n = q.next_up();
                match beyond_mid(q, n) {
                    Sign::Positive => q = n,
                    Sign::Negative => return q + 0.0,
                    Sign::Zero => {
                        return if (q.to_bits() & 1) == 0 {
                            q + 0.0
                        } else {
                            n + 0.0
                        }
                    }
                }
            }
            Sign::Negative => {
                let n = q.next_down();
                match beyond_mid(n, q) {
                    Sign::Negative => q = n,
                    Sign::Positive => return q + 0.0,
                    Sign::Zero => {
                        return if (q.to_bits() & 1) == 0 {
                            q + 0.0
                        } else {
                            n + 0.0
                        }
                    }
                }
            }
        }
    }
}

impl Ring for Expansion {
    fn from_f64(v: f64) -> Self {
        Expansion::from_f64(v)
    }
    fn add(&self, other: &Self) -> Self {
        Expansion::add(self, other)
    }
    fn sub(&self, other: &Self) -> Self {
        Expansion::sub(self, other)
    }
    fn mul(&self, other: &Self) -> Self {
        Expansion::mul(self, other)
    }
    fn neg(&self) -> Self {
        Expansion::neg(self)
    }
}
