//! Exact predicates with a symbolic perturbation that removes every
//! cospherical degeneracy.
//!
//! The lifted height `|p|^2` of point `i` is raised by `eps_i`, with
//! `eps_i` of larger points (lexicographically, by their coordinates)
//! dominating. Expanding the 5x5 insphere determinant along the lifted
//! column, the perturbed sign is the sign of the determinant when it is not
//! zero, else the sign of the cofactor of the largest point whose cofactor
//! is not zero; a cofactor is an orientation of the other four points. The
//! largest point of a test against a positive tet always has one, so the
//! perturbed test never ties (Edelsbrunner and Muecke; Devillers and
//! Teillaud).

use geometry_predicates::{insphere, orient3d};

pub type P3 = [f64; 3];

/// The sign of `orient3d`: positive when `d` lies on the side of the plane
/// `a b c` from which `a b c` turn clockwise (Shewchuk's convention: a
/// tet `a b c d` with positive orientation).
pub fn orient(a: P3, b: P3, c: P3, d: P3) -> i8 {
    // Four points on one plane of constant coordinate (the faces of layer
    // stacks hold many) make a column of the determinant zero: exactly
    // zero, without the exact stage the adaptive test would reach.
    if (0..3).any(|k| a[k] == b[k] && a[k] == c[k] && a[k] == d[k]) {
        return 0;
    }
    sign(orient3d(a, b, c, d))
}

/// Whether `e` lies inside the sphere of the positive tet `t`, under the
/// perturbation: never zero. The five points (distinct) are ordered by
/// their coordinates, lexicographically, the larger carrying the dominant
/// perturbation: the answer is a property of the points alone, whatever
/// their ids, so a tetrahedralization made point by point is the one made
/// at once.
pub fn inside(t: [P3; 4], e: P3) -> bool {
    let s = sign(insphere(t[0], t[1], t[2], t[3], e));
    if s != 0 {
        return s > 0;
    }
    let p = [t[0], t[1], t[2], t[3], e];
    let lex = |a: P3, b: P3| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    };
    let mut order = [0usize, 1, 2, 3, 4];
    order.sort_unstable_by(|&i, &j| lex(p[j], p[i]));
    for i in order {
        // The cofactor of row i on the lifted column: (-1)^(i+3) times the
        // orientation of the other four in their order. Raising the lift of
        // a point by eps changes the determinant by eps times it.
        let o: Vec<P3> = (0..5).filter(|&j| j != i).map(|j| p[j]).collect();
        let m = orient(o[0], o[1], o[2], o[3]);
        if m != 0 {
            let cof = if (i + 3) % 2 == 0 { m } else { -m };
            return cof > 0;
        }
    }
    unreachable!("a positive tet has a nonzero cofactor")
}

fn sign(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positive(mut q: [P3; 4]) -> [P3; 4] {
        if orient(q[0], q[1], q[2], q[3]) < 0 {
            q.swap(2, 3);
        }
        q
    }

    /// The unit square turned by `k` quarter turns about its center, its
    /// corners in turn: each turn hands the lexicographic lead to another
    /// corner.
    fn square(k: usize) -> [P3; 4] {
        let base = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        std::array::from_fn(|i| {
            let [x, y] = base[(i + k) % 4];
            [x, y, 0.0]
        })
    }

    /// Off a sphere the exact answer stands; on it, the four corners of a
    /// cospherical square under an apex get exactly one Delaunay diagonal,
    /// whichever corner leads.
    #[test]
    fn cospherical_ties_are_broken_consistently() {
        let apex = [0.5, 0.5, 3.0];
        let [a, b, d, _] = square(0);
        let t = positive([a, b, d, apex]);
        assert!(inside(t, [0.5, 0.5, 0.5]));
        assert!(!inside(t, [3.0, 3.0, 3.0]));
        for k in 0..4 {
            let [a, b, d, e] = square(k);
            // Diagonal a d: the tet a b d apex against e; diagonal b e: the
            // tet a b e apex against d.
            let in1 = inside(positive([a, b, d, apex]), e);
            let in2 = inside(positive([a, b, e, apex]), d);
            assert_ne!(in1, in2, "turn {k}");
        }
    }

    /// The perturbed answer is a property of the points: the same for
    /// every positive ordering of the tet and for an apex on either side of
    /// a cospherical square, whichever corner leads.
    #[test]
    fn the_perturbed_answer_is_invariant() {
        let apexes = [[0.3, 0.6, 2.0], [0.3, 0.6, -2.0], [0.9, 0.1, 0.7]];
        let perms: [[usize; 3]; 6] = [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [1, 0, 2],
            [0, 2, 1],
            [2, 1, 0],
        ];
        for k in 0..4 {
            let sq = square(k);
            // d = corner 3 against the circle of corners 0 1 2.
            let mut answers = Vec::new();
            for m in apexes {
                for p in perms {
                    let t = positive([sq[p[0]], sq[p[1]], sq[p[2]], m]);
                    answers.push(inside(t, sq[3]));
                }
            }
            assert!(
                answers.iter().all(|&x| x == answers[0]),
                "turn {k}: {answers:?}"
            );
        }
    }
}
