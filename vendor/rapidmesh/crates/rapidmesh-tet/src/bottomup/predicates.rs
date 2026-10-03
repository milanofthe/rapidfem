//! Exact predicates with a symbolic perturbation that removes every
//! cospherical degeneracy.
//!
//! The lifted height `|p|^2` of point `i` is raised by `eps_i`, with
//! `eps_i` of larger points (by their key) dominating. Expanding the 5x5
//! insphere determinant along the lifted column, the perturbed sign is the
//! sign of the determinant when it is not zero, else the sign of the
//! cofactor of the largest point whose cofactor is not zero; a cofactor is
//! an orientation of the other four points. The largest point of a test
//! against a positive tet always has one, so the perturbed test never ties
//! (Edelsbrunner and Muecke; Devillers and Teillaud).

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
/// perturbation: never zero. `keys` order the five points (`t` then `e`);
/// larger keys carry the dominant perturbation. The five keys must be
/// distinct.
pub fn inside(t: [P3; 4], e: P3, keys: [u32; 5]) -> bool {
    let s = sign(insphere(t[0], t[1], t[2], t[3], e));
    if s != 0 {
        return s > 0;
    }
    let p = [t[0], t[1], t[2], t[3], e];
    let mut order = [0usize, 1, 2, 3, 4];
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(keys[i]));
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

    fn positive(mut q: [(P3, u32); 4]) -> ([P3; 4], [u32; 4]) {
        if orient(q[0].0, q[1].0, q[2].0, q[3].0) < 0 {
            q.swap(2, 3);
        }
        (q.map(|x| x.0), q.map(|x| x.1))
    }

    /// Off a sphere the exact answer stands; on it, the four corners of a
    /// cospherical square under an apex get exactly one Delaunay diagonal,
    /// whatever the keys.
    #[test]
    fn cospherical_ties_are_broken_consistently() {
        let (a, b, d, e) = (
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        let apex = [0.5, 0.5, 3.0];
        let (t, k) = positive([(a, 0), (b, 1), (d, 3), (apex, 9)]);
        assert!(inside(t, [0.5, 0.5, 0.5], [k[0], k[1], k[2], k[3], 20]));
        assert!(!inside(t, [3.0, 3.0, 3.0], [k[0], k[1], k[2], k[3], 20]));
        for keys in [
            [0, 1, 3, 2, 9],
            [5, 1, 3, 2, 0],
            [2, 7, 1, 4, 3],
            [4, 3, 2, 1, 0],
        ] {
            let [ka, kb, kd, ke, kx] = keys;
            // Diagonal a d: the tet a b d apex against e; diagonal b e: the
            // tet a b e apex against d.
            let (t1, k1) = positive([(a, ka), (b, kb), (d, kd), (apex, kx)]);
            let (t2, k2) = positive([(a, ka), (b, kb), (e, ke), (apex, kx)]);
            let in1 = inside(t1, e, [k1[0], k1[1], k1[2], k1[3], ke]);
            let in2 = inside(t2, d, [k2[0], k2[1], k2[2], k2[3], kd]);
            assert_ne!(in1, in2, "keys {keys:?}");
        }
    }

    /// The perturbed answer is a property of the points: the same for
    /// every positive ordering of the tet and for an apex on either side of
    /// a cospherical square, for every assignment of keys.
    #[test]
    fn the_perturbed_answer_is_invariant() {
        let sq = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let apexes = [[0.3, 0.6, 2.0], [0.3, 0.6, -2.0], [0.9, 0.1, 0.7]];
        let perms: [[usize; 3]; 6] = [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [1, 0, 2],
            [0, 2, 1],
            [2, 1, 0],
        ];
        let key_sets: [[u32; 4]; 6] = [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [2, 4, 1, 3],
            [3, 1, 4, 2],
            [1, 3, 2, 4],
            [4, 1, 3, 2],
        ];
        for keys in key_sets {
            // d = corner 3 against the circle of corners 0 1 2.
            let mut answers = Vec::new();
            for m in apexes {
                for p in perms {
                    let tri = p.map(|i| (sq[i], keys[i]));
                    let (t, k) = positive([tri[0], tri[1], tri[2], (m, 0)]);
                    if orient(t[0], t[1], t[2], t[3]) != 1 {
                        continue;
                    }
                    answers.push(inside(t, sq[3], [k[0], k[1], k[2], k[3], keys[3]]));
                }
            }
            assert!(
                answers.iter().all(|&x| x == answers[0]),
                "keys {keys:?}: {answers:?}"
            );
        }
    }
}
