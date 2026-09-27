//! Generic geometric expressions over the [`Ring`] trait.
//!
//! Written once, evaluated with intervals (filter), expansions (exact), or
//! rationals (test oracle). All functions are polynomial in the inputs — no
//! division ever happens; implicit points stay homogeneous.

use crate::ring::Ring;

/// Lifts an f64 point into the ring.
pub fn lift<T: Ring>(p: [f64; 3]) -> [T; 3] {
    std::array::from_fn(|i| T::from_f64(p[i]))
}

/// Component-wise difference a - b.
pub fn sub3<T: Ring>(a: &[T; 3], b: &[T; 3]) -> [T; 3] {
    std::array::from_fn(|i| a[i].sub(&b[i]))
}

/// Cross product.
pub fn cross<T: Ring>(a: &[T; 3], b: &[T; 3]) -> [T; 3] {
    [
        a[1].mul(&b[2]).sub(&a[2].mul(&b[1])),
        a[2].mul(&b[0]).sub(&a[0].mul(&b[2])),
        a[0].mul(&b[1]).sub(&a[1].mul(&b[0])),
    ]
}

/// Dot product.
pub fn dot<T: Ring>(a: &[T; 3], b: &[T; 3]) -> T {
    a[0].mul(&b[0]).add(&a[1].mul(&b[1])).add(&a[2].mul(&b[2]))
}

/// Determinant of a 3x3 matrix (rows).
pub fn det3<T: Ring>(m: &[[T; 3]; 3]) -> T {
    let c0 = m[1][1].mul(&m[2][2]).sub(&m[1][2].mul(&m[2][1]));
    let c1 = m[1][0].mul(&m[2][2]).sub(&m[1][2].mul(&m[2][0]));
    let c2 = m[1][0].mul(&m[2][1]).sub(&m[1][1].mul(&m[2][0]));
    m[0][0]
        .mul(&c0)
        .sub(&m[0][1].mul(&c1))
        .add(&m[0][2].mul(&c2))
}

/// Determinant of a 4x4 matrix (rows), Laplace expansion along the first row.
pub fn det4<T: Ring>(m: &[[T; 4]; 4]) -> T {
    let minor = |col: usize| -> [[T; 3]; 3] {
        std::array::from_fn(|i| {
            let row = &m[i + 1];
            let mut it = (0..4).filter(|&j| j != col);
            std::array::from_fn(|_| row[it.next().expect("3 columns remain")].clone())
        })
    };
    let t0 = m[0][0].mul(&det3(&minor(0)));
    let t1 = m[0][1].mul(&det3(&minor(1)));
    let t2 = m[0][2].mul(&det3(&minor(2)));
    let t3 = m[0][3].mul(&det3(&minor(3)));
    t0.sub(&t1).add(&t2).sub(&t3)
}

/// Determinant of a 4x4 matrix (rows) by Laplace expansion along the last
/// column. For lifted rows `(x, y, z, lift)` each lift then enters exactly
/// once, times a 3x3 minor of plain coordinates: fewer operations, smaller
/// expansions and tighter intervals than expanding along the first row.
pub fn det4_lift<T: Ring>(m: &[[T; 4]; 4]) -> T {
    let minor = |row: usize| -> [[T; 3]; 3] {
        let mut it = (0..4).filter(|&i| i != row);
        std::array::from_fn(|_| {
            let r = &m[it.next().expect("3 rows remain")];
            [r[0].clone(), r[1].clone(), r[2].clone()]
        })
    };
    let t0 = m[0][3].mul(&det3(&minor(0)));
    let t1 = m[1][3].mul(&det3(&minor(1)));
    let t2 = m[2][3].mul(&det3(&minor(2)));
    let t3 = m[3][3].mul(&det3(&minor(3)));
    t1.sub(&t0).add(&t3).sub(&t2)
}

/// Determinant of a 5x5 matrix (rows), Laplace expansion along the first row
/// (the homogeneous in-sphere lift needs it).
pub fn det5<T: Ring>(m: &[[T; 5]; 5]) -> T {
    let minor = |col: usize| -> [[T; 4]; 4] {
        std::array::from_fn(|i| {
            let row = &m[i + 1];
            let mut it = (0..5).filter(|&j| j != col);
            std::array::from_fn(|_| row[it.next().expect("4 columns remain")].clone())
        })
    };
    let mut acc = m[0][0].mul(&det4(&minor(0)));
    for (col, entry) in m[0].iter().enumerate().skip(1) {
        let term = entry.mul(&det4(&minor(col)));
        acc = if col % 2 == 1 {
            acc.sub(&term)
        } else {
            acc.add(&term)
        };
    }
    acc
}

/// Homogeneous coordinates (x, y, z, w) of the intersection of the line
/// through `p`, `q` with the plane through `r`, `s`, `t`.
///
/// w is the plane-normal dot line-direction; w == 0 means line parallel to
/// plane (the point is invalid). Coordinate polynomials have degree 4 in the
/// inputs, w has degree 3.
pub fn lpi_hom<T: Ring>(p: [f64; 3], q: [f64; 3], r: [f64; 3], s: [f64; 3], t: [f64; 3]) -> [T; 4] {
    let (p, q, r, s, t) = (
        lift::<T>(p),
        lift::<T>(q),
        lift::<T>(r),
        lift::<T>(s),
        lift::<T>(t),
    );
    let n = cross(&sub3(&s, &r), &sub3(&t, &r));
    let dir = sub3(&q, &p);
    let w = dot(&n, &dir);
    let num = dot(&n, &sub3(&r, &p));
    // point = p + (num/w) * dir, homogenized with w
    let coord = |i: usize| p[i].mul(&w).add(&num.mul(&dir[i]));
    [coord(0), coord(1), coord(2), w]
}

/// Homogeneous coordinates (x, y, z, w) of the intersection of three planes,
/// each given by three points `planes[k] = [p, q, r]`.
///
/// Cramer's rule on N·x = c with N rows the plane normals and c the plane
/// offsets; w = det(N) == 0 means the planes do not meet in a single point.
/// Coordinate polynomials have degree 7 in the inputs, w has degree 6.
pub fn tpi_hom<T: Ring>(planes: &[[[f64; 3]; 3]; 3]) -> [T; 4] {
    let mut n: Vec<[T; 3]> = Vec::with_capacity(3);
    let mut c: Vec<T> = Vec::with_capacity(3);
    for plane in planes {
        let p = lift::<T>(plane[0]);
        let q = lift::<T>(plane[1]);
        let r = lift::<T>(plane[2]);
        let nk = cross(&sub3(&q, &p), &sub3(&r, &p));
        c.push(dot(&nk, &p));
        n.push(nk);
    }
    let col = |replace: Option<usize>| -> [[T; 3]; 3] {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| match replace {
                Some(rj) if rj == j => c[i].clone(),
                _ => n[i][j].clone(),
            })
        })
    };
    let w = det3(&col(None));
    let x = det3(&col(Some(0)));
    let y = det3(&col(Some(1)));
    let z = det3(&col(Some(2)));
    [x, y, z, w]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expansion::Expansion;

    /// Expanding along the lift column gives exactly the first-row value.
    #[test]
    fn det4_lift_equals_det4() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2001) as f64 * 1e-3 - 1.0
        };
        for _ in 0..500 {
            let m: [[Expansion; 4]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| Expansion::from_f64(rnd())));
            let (a, b) = (det4(&m), det4_lift(&m));
            assert!(a.sub(&b).is_zero(), "{:?} vs {:?}", a.approx(), b.approx());
        }
    }
}
