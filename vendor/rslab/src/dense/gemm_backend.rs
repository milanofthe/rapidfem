//! The one GEMM entry of the numeric kernels.
//!
//! Every trailing update goes through [`gemm`], which has the signature and
//! semantics of the `gemm` crate (`dst := alpha * dst + beta * lhs * rhs`,
//! arbitrary strides): the pure-Rust SIMD kernels, the same on every
//! platform, so a factorization is bit-identical wherever it runs.
//!
//! Sequential complex products do not run on the crate's complex kernel:
//! its interleaved complex microkernel reaches about 80% of the real
//! kernel's flop rate, so a complex product is split into real products of
//! the real and imaginary planes in the Gauss form ([`complex_gemm_3m`]:
//! three real products of the same size, 6/8 of the direct flop count; the
//! textbook four-product form [`complex_gemm_4m`] is kept for reference).
//! The split trades flops for memory traffic (the planes are copied), which
//! pays on a compute-bound core and not on a bandwidth-bound machine, so
//! products that run in parallel take the direct kernel; on a 102k-DOF
//! complex FEM factorization the split is worth about 10% single-core and
//! nothing at 8 workers. The split is deterministic (fixed kernels, fixed
//! association), so results stay bit-identical across thread counts. Thin
//! products (see [`GemmMode::split_min_ratio`]) and conjugated operands take the direct
//! kernel, and the split runs per tile, so its scratch stays bounded.

use std::cell::RefCell;

use num_complex::Complex;

use crate::scalar::Scalar;

/// See the module docs; the arguments are those of `gemm::gemm`.
///
/// # Safety
/// The pointers and strides must describe valid, non-overlapping matrices
/// of the given sizes (the contract of `gemm::gemm`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn gemm<T: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    mode: GemmMode,
) {
    T::gemm(
        m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs, alpha,
        beta, conj_dst, conj_lhs, conj_rhs, mode,
    )
}

/// How one product runs: the `gemm` crate's parallelism, and the complex
/// split gate and tile of [`KernelSettings`](crate::KernelSettings).
#[derive(Debug, Clone, Copy)]
pub struct GemmMode {
    pub parallelism: gemm::Parallelism,
    /// A product splits only when its flops `m n k` are at least this many
    /// times its plane copies `m k + k n + m n`. The saving is a quarter of
    /// the kernel time and the copies are memory traffic, so a thin product
    /// (one short dimension: the rank-`k` panel updates, or the narrow
    /// updates of a left-looking LU) loses: on a Ryzen 9900X a MoM LU
    /// factorization took 20% longer at 8 workers with every product above 8k
    /// flops split, and the same as direct at the default `64`.
    pub split_min_ratio: usize,
    /// Tile edge of the split: the planes of one tile of the product
    /// (`3 (2 tile k + tile^2)` reals) are the whole scratch.
    pub split_tile: usize,
}

impl GemmMode {
    pub fn new(parallelism: gemm::Parallelism, k: &crate::KernelSettings) -> Self {
        Self {
            parallelism,
            split_min_ratio: k.complex_split_min_ratio,
            split_tile: k.complex_split_tile.max(1),
        }
    }
}

/// Whether a complex product of this shape goes through the real kernels.
#[inline]
pub fn split_worthwhile(m: usize, n: usize, k: usize, conj: bool, min_ratio: usize) -> bool {
    !conj && m * n * k >= min_ratio * (m * k + k * n + m * n)
}

/// Real plane scratch of one thread (grows to the largest tile seen).
struct Planes<R> {
    buf: Vec<R>,
}

thread_local! {
    static PLANES_F64: RefCell<Planes<f64>> = const { RefCell::new(Planes { buf: Vec::new() }) };
    static PLANES_F32: RefCell<Planes<f32>> = const { RefCell::new(Planes { buf: Vec::new() }) };
}

/// The real field of a complex scalar for which the split is provided.
pub trait SplitReal:
    Copy
    + 'static
    + Send
    + Sync
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn with_planes<F: FnOnce(&mut Vec<Self>) -> Ret, Ret>(f: F) -> Ret;
}
/// Run `f` on the thread's plane buffer. The buffer is taken out of the
/// thread-local for the duration (not borrowed): a parallel real product
/// inside `f` lets rayon's work stealing run another split on this same
/// thread, which then gets an empty buffer of its own instead of a
/// re-entrant borrow. The larger of the two is kept afterwards.
fn with_taken<R: SplitReal, Ret>(
    cell: &'static std::thread::LocalKey<RefCell<Planes<R>>>,
    f: impl FnOnce(&mut Vec<R>) -> Ret,
) -> Ret {
    let mut buf = cell.with(|p| std::mem::take(&mut p.borrow_mut().buf));
    let out = f(&mut buf);
    cell.with(|p| {
        let mut planes = p.borrow_mut();
        if planes.buf.capacity() < buf.capacity() {
            planes.buf = buf;
        }
    });
    out
}
impl SplitReal for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn with_planes<F: FnOnce(&mut Vec<Self>) -> Ret, Ret>(f: F) -> Ret {
        with_taken(&PLANES_F64, f)
    }
}
impl SplitReal for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn with_planes<F: FnOnce(&mut Vec<Self>) -> Ret, Ret>(f: F) -> Ret {
        with_taken(&PLANES_F32, f)
    }
}

/// `a * b` for complex numbers over a [`SplitReal`] (no trait bound on the
/// complex type needed).
#[inline]
fn cmul<R: SplitReal>(a: Complex<R>, b: Complex<R>) -> Complex<R> {
    Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// Copy the real and imaginary parts of a strided `rows x cols` complex
/// matrix into two column-major planes with leading dimension `rows`.
///
/// # Safety
/// `src` with the strides must be a valid `rows x cols` matrix.
unsafe fn split_planes<R: SplitReal>(
    src: *const Complex<R>,
    cs: isize,
    rs: isize,
    rows: usize,
    cols: usize,
    re: &mut [R],
    im: &mut [R],
) {
    for j in 0..cols {
        let col = src.offset(j as isize * cs);
        let (re_col, im_col) = (
            &mut re[j * rows..(j + 1) * rows],
            &mut im[j * rows..(j + 1) * rows],
        );
        for i in 0..rows {
            let v = *col.offset(i as isize * rs);
            re_col[i] = v.re;
            im_col[i] = v.im;
        }
    }
}

/// Real product `dst := (read_dst ? alpha * dst : 0) + beta * lhs * rhs` on
/// column-major planes.
#[allow(clippy::too_many_arguments)]
unsafe fn real_gemm<R: SplitReal>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut R,
    read_dst: bool,
    lhs: *const R,
    rhs: *const R,
    alpha: R,
    beta: R,
    parallelism: gemm::Parallelism,
) {
    gemm::gemm(
        m,
        n,
        k,
        dst,
        m as isize,
        1,
        read_dst,
        lhs,
        m as isize,
        1,
        rhs,
        k as isize,
        1,
        alpha,
        beta,
        false,
        false,
        false,
        parallelism,
    )
}

/// Write the product planes into the strided complex destination:
/// `dst := (read_dst ? alpha * dst : 0) + beta * (pr + i pi)`.
///
/// # Safety
/// `dst` with the strides must be a valid `m x n` matrix.
#[allow(clippy::too_many_arguments)]
unsafe fn combine<R: SplitReal>(
    m: usize,
    n: usize,
    dst: *mut Complex<R>,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    alpha: Complex<R>,
    beta: Complex<R>,
    prod: impl Fn(usize) -> Complex<R>,
) {
    for j in 0..n {
        let col = dst.offset(j as isize * dst_cs);
        for i in 0..m {
            let p = col.offset(i as isize * dst_rs);
            let v = cmul(beta, prod(i + j * m));
            *p = if read_dst {
                let d = cmul(alpha, *p);
                Complex::new(d.re + v.re, d.im + v.im)
            } else {
                v
            };
        }
    }
}

/// The four-product form: `Cr = Ar Br - Ai Bi`, `Ci = Ar Bi + Ai Br`.
///
/// # Safety
/// As `gemm::gemm`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub unsafe fn complex_gemm_4m<R: SplitReal>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut Complex<R>,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const Complex<R>,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const Complex<R>,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: Complex<R>,
    beta: Complex<R>,
    parallelism: gemm::Parallelism,
) {
    R::with_planes(|buf| {
        let (mk, kn, mn) = (m * k, k * n, m * n);
        buf.clear();
        buf.resize(2 * mk + 2 * kn + 2 * mn, R::zero());
        let (ar, rest) = buf.split_at_mut(mk);
        let (ai, rest) = rest.split_at_mut(mk);
        let (br, rest) = rest.split_at_mut(kn);
        let (bi, rest) = rest.split_at_mut(kn);
        let (cr, ci) = rest.split_at_mut(mn);
        split_planes(lhs, lhs_cs, lhs_rs, m, k, ar, ai);
        split_planes(rhs, rhs_cs, rhs_rs, k, n, br, bi);
        let (one, neg) = (R::one(), -R::one());
        // Cr = Ar Br - Ai Bi
        real_gemm(
            m,
            n,
            k,
            cr.as_mut_ptr(),
            false,
            ar.as_ptr(),
            br.as_ptr(),
            one,
            one,
            parallelism,
        );
        real_gemm(
            m,
            n,
            k,
            cr.as_mut_ptr(),
            true,
            ai.as_ptr(),
            bi.as_ptr(),
            one,
            neg,
            parallelism,
        );
        // Ci = Ar Bi + Ai Br
        real_gemm(
            m,
            n,
            k,
            ci.as_mut_ptr(),
            false,
            ar.as_ptr(),
            bi.as_ptr(),
            one,
            one,
            parallelism,
        );
        real_gemm(
            m,
            n,
            k,
            ci.as_mut_ptr(),
            true,
            ai.as_ptr(),
            br.as_ptr(),
            one,
            one,
            parallelism,
        );
        combine(m, n, dst, dst_cs, dst_rs, read_dst, alpha, beta, |e| {
            Complex::new(cr[e], ci[e])
        });
    })
}

/// The three-product (Gauss) form: `T1 = Ar Br`, `T2 = Ai Bi`,
/// `T3 = (Ar + Ai)(Br + Bi)`, `Cr = T1 - T2`, `Ci = T3 - T1 - T2`, per
/// `split_tile` block of the product (the inner dimension whole, so every
/// entry sums as in one product).
///
/// # Safety
/// As `gemm::gemm`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn complex_gemm_3m<R: SplitReal>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut Complex<R>,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const Complex<R>,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const Complex<R>,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: Complex<R>,
    beta: Complex<R>,
    parallelism: gemm::Parallelism,
    tile: usize,
) {
    R::with_planes(|buf| {
        let (mt, nt) = (m.min(tile), n.min(tile));
        let need = 3 * (mt * k + k * nt + mt * nt);
        // every plane is written in full before it is read: no clearing
        if buf.len() < need {
            buf.resize(need, R::zero());
        }
        let (a_planes, rest) = buf.split_at_mut(3 * mt * k);
        let (b_planes, c_planes) = rest.split_at_mut(3 * k * nt);
        for j0 in (0..n).step_by(tile) {
            let nb = (n - j0).min(tile);
            let (br, rest) = b_planes.split_at_mut(k * nb);
            let (bi, rest) = rest.split_at_mut(k * nb);
            let bsum = &mut rest[..k * nb];
            split_planes(
                rhs.offset(j0 as isize * rhs_cs),
                rhs_cs,
                rhs_rs,
                k,
                nb,
                br,
                bi,
            );
            for e in 0..k * nb {
                bsum[e] = br[e] + bi[e];
            }
            for i0 in (0..m).step_by(tile) {
                let mb = (m - i0).min(tile);
                let (ar, rest) = a_planes.split_at_mut(mb * k);
                let (ai, rest) = rest.split_at_mut(mb * k);
                let asum = &mut rest[..mb * k];
                split_planes(
                    lhs.offset(i0 as isize * lhs_rs),
                    lhs_cs,
                    lhs_rs,
                    mb,
                    k,
                    ar,
                    ai,
                );
                for e in 0..mb * k {
                    asum[e] = ar[e] + ai[e];
                }
                let (t1, rest) = c_planes.split_at_mut(mb * nb);
                let (t2, rest) = rest.split_at_mut(mb * nb);
                let t3 = &mut rest[..mb * nb];
                let one = R::one();
                for (t, a, b) in [
                    (&mut *t1, &*ar, &*br),
                    (&mut *t2, &*ai, &*bi),
                    (&mut *t3, &*asum, &*bsum),
                ] {
                    real_gemm(
                        mb,
                        nb,
                        k,
                        t.as_mut_ptr(),
                        false,
                        a.as_ptr(),
                        b.as_ptr(),
                        one,
                        one,
                        parallelism,
                    );
                }
                let tile = dst.offset(i0 as isize * dst_rs + j0 as isize * dst_cs);
                combine(mb, nb, tile, dst_cs, dst_rs, read_dst, alpha, beta, |e| {
                    Complex::new(t1[e] - t2[e], t3[e] - t1[e] - t2[e])
                });
            }
        }
    })
}

/// The complex product of the numeric kernels: the split form unless the
/// product is tiny or conjugated.
///
/// # Safety
/// As `gemm::gemm`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn complex_gemm<R: SplitReal>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut Complex<R>,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const Complex<R>,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const Complex<R>,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: Complex<R>,
    beta: Complex<R>,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    mode: GemmMode,
) where
    Complex<R>: 'static,
{
    let parallelism = mode.parallelism;
    let sequential = matches!(parallelism, gemm::Parallelism::None);
    let conj = conj_dst || conj_lhs || conj_rhs;
    if sequential && split_worthwhile(m, n, k, conj, mode.split_min_ratio) {
        return complex_gemm_3m(
            m,
            n,
            k,
            dst,
            dst_cs,
            dst_rs,
            read_dst,
            lhs,
            lhs_cs,
            lhs_rs,
            rhs,
            rhs_cs,
            rhs_rs,
            alpha,
            beta,
            parallelism,
            mode.split_tile,
        );
    }
    gemm::gemm(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        read_dst,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        conj_dst,
        conj_lhs,
        conj_rhs,
        parallelism,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % 2_000_000) as f64 / 1_000_000.0 - 1.0
        }
        fn c(&mut self) -> Complex64 {
            Complex64::new(self.next(), self.next())
        }
    }

    fn run(
        m: usize,
        n: usize,
        k: usize,
        read_dst: bool,
        alpha: Complex64,
        beta: Complex64,
        transposed_lhs: bool,
    ) {
        let mut rng = Rng(0x9E37_79B9 ^ (m * 131 + n * 17 + k) as u64 | 1);
        let lhs: Vec<Complex64> = (0..m * k).map(|_| rng.c()).collect();
        let rhs: Vec<Complex64> = (0..k * n).map(|_| rng.c()).collect();
        let dst0: Vec<Complex64> = (0..m * n).map(|_| rng.c()).collect();
        // lhs strides: column-major (cs = m, rs = 1) or row-major (cs = 1, rs = k)
        let (lcs, lrs) = if transposed_lhs {
            (1, k as isize)
        } else {
            (m as isize, 1)
        };
        let mut direct = dst0.clone();
        let mut d3 = dst0.clone();
        let mut d4 = dst0.clone();
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                direct.as_mut_ptr(),
                m as isize,
                1,
                read_dst,
                lhs.as_ptr(),
                lcs,
                lrs,
                rhs.as_ptr(),
                k as isize,
                1,
                alpha,
                beta,
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
            complex_gemm_3m(
                m,
                n,
                k,
                d3.as_mut_ptr(),
                m as isize,
                1,
                read_dst,
                lhs.as_ptr(),
                lcs,
                lrs,
                rhs.as_ptr(),
                k as isize,
                1,
                alpha,
                beta,
                gemm::Parallelism::None,
                256,
            );
            complex_gemm_4m(
                m,
                n,
                k,
                d4.as_mut_ptr(),
                m as isize,
                1,
                read_dst,
                lhs.as_ptr(),
                lcs,
                lrs,
                rhs.as_ptr(),
                k as isize,
                1,
                alpha,
                beta,
                gemm::Parallelism::None,
            );
        }
        let norm = direct.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        for (name, got) in [("3m", &d3), ("4m", &d4)] {
            let err = got
                .iter()
                .zip(&direct)
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(
                err <= 1e-13 * (norm + 1.0),
                "{name} m={m} n={n} k={k} read={read_dst} err {err:.3e} norm {norm:.3e}"
            );
        }
    }

    #[test]
    fn split_products_match_the_direct_kernel() {
        let one = Complex64::new(1.0, 0.0);
        let neg = Complex64::new(-1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let odd = Complex64::new(0.3, -0.7);
        for &(m, n, k) in &[
            (1, 1, 4),
            (7, 5, 3),
            (64, 64, 8),
            (200, 64, 16),
            (129, 33, 65),
            (33, 129, 17),
            (300, 300, 300),
            (513, 257, 7),
        ] {
            run(m, n, k, false, zero, one, false);
            run(m, n, k, true, one, neg, false);
            run(m, n, k, true, odd, odd, false);
            run(m, n, k, true, one, neg, true);
        }
    }

    #[test]
    fn split_is_deterministic() {
        let mut rng = Rng(7);
        let (m, n, k) = (100, 40, 20);
        let lhs: Vec<Complex64> = (0..m * k).map(|_| rng.c()).collect();
        let rhs: Vec<Complex64> = (0..k * n).map(|_| rng.c()).collect();
        let mut a = vec![Complex64::new(0.0, 0.0); m * n];
        let mut b = a.clone();
        unsafe {
            complex_gemm_3m(
                m,
                n,
                k,
                a.as_mut_ptr(),
                m as isize,
                1,
                false,
                lhs.as_ptr(),
                m as isize,
                1,
                rhs.as_ptr(),
                k as isize,
                1,
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                gemm::Parallelism::None,
                256,
            );
            complex_gemm_3m(
                m,
                n,
                k,
                b.as_mut_ptr(),
                m as isize,
                1,
                false,
                lhs.as_ptr(),
                m as isize,
                1,
                rhs.as_ptr(),
                k as isize,
                1,
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                gemm::Parallelism::None,
                256,
            );
        }
        assert!(a
            .iter()
            .zip(&b)
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()));
    }
}
