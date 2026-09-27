//! Moving boundary vertices onto the true shape.
//!
//! The refinement runs on an exact discrete model (a PLC); the analytic
//! carriers and curves it approximates come back here. Each curve and patch
//! vertex moves to its projection onto its [`Shape`], curves before patches,
//! as far as every incident tet stays positively oriented and none drops
//! below `FLOOR_DEG` (or below where its star already was): the full
//! move, else the largest of a few halvings, else none. Labels, faces and
//! connectivity never change, so every invariant but orientation carries
//! over, and orientation is checked exactly. The moves run on the index of
//! [`super::improve`], which snaps again what stayed short once it has
//! reshaped the star ([`super::improve::finish`]).
//!
//! A volume vertex that ended up on faces of one patch only (a cell point
//! inserted so close to the surface that the surface points meant to
//! replace it were rejected as its duplicates) is a surface sample in all
//! but name: it becomes a vertex of that patch first, and snaps with the
//! rest.

use super::oracle::P3;
use super::{Complex, VertexKind};

/// The true shape behind a discrete model.
pub trait Shape: Sync {
    /// The point of the shape a vertex of `kind` at `p` belongs at: on its
    /// patch's carrier or its curve. `None` leaves the vertex where it is.
    fn project(&self, kind: VertexKind, p: P3) -> Option<P3>;
}

/// What [`snap`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SnapStats {
    /// Vertices moved all the way.
    pub full: usize,
    /// Vertices moved part of the way (a full move would invert a tet).
    pub partial: usize,
    /// Vertices left in place (every tried move would invert a tet).
    pub blocked: usize,
    /// Largest remaining distance to the shape.
    pub max_residual: f64,
    /// Volume vertices on faces of one patch, moved to that patch.
    pub adopted: usize,
}

/// Halvings tried before a vertex is left in place.
pub(crate) const HALVINGS: usize = 4;
/// A snap step may not take a tet of the star below this dihedral (degrees)
/// unless the star was there already: fidelity to the shape is not worth a
/// sliver.
pub(crate) const FLOOR_DEG: f64 = 10.0;

/// Moves the boundary vertices of `c` onto `shape`.
pub fn snap(c: &mut Complex, shape: &dyn Shape) -> SnapStats {
    super::improve::snap_complex(c, shape)
}

/// Gives every volume vertex that lies on faces of exactly one patch that
/// patch's kind. Returns how many changed.
pub(crate) fn adopt_face_vertices(c: &mut Complex) -> usize {
    const NONE: u32 = u32::MAX;
    const MIXED: u32 = u32::MAX - 1;
    let mut patch = vec![NONE; c.points.len()];
    for f in &c.faces {
        for &v in &f.tri {
            let p = &mut patch[v as usize];
            *p = if f.patch >= MIXED {
                MIXED
            } else if *p == NONE || *p == f.patch {
                f.patch
            } else {
                MIXED
            };
        }
    }
    let mut adopted = 0;
    for (v, &p) in patch.iter().enumerate() {
        if p < MIXED && c.kinds[v] == VertexKind::Volume {
            c.kinds[v] = VertexKind::Patch(p);
            adopted += 1;
        }
    }
    adopted
}

#[cfg(test)]
mod tests {
    use super::super::oracle::domains::Balls;
    use super::super::oracle::Uniform;
    use super::super::refine::{mesh, Params};
    use super::super::verify::check;
    use super::*;

    /// A sphere shape of radius `r`.
    struct Sphere(f64);

    impl Shape for Sphere {
        fn project(&self, kind: VertexKind, p: P3) -> Option<P3> {
            match kind {
                VertexKind::Patch(_) => {
                    let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    Some(p.map(|x| x * self.0 / l))
                }
                _ => None,
            }
        }
    }

    #[test]
    fn snapping_keeps_the_complex_valid() {
        // Mesh a ball, then snap onto a slightly larger sphere: the moves are
        // real, and orientation must survive.
        let d = Balls::new([0.0; 3], &[1.0]);
        let (mut c, _) = mesh(&d, &Uniform(0.25), &Params::default());
        let st = snap(&mut c, &Sphere(1.02));
        let r = check(&c);
        assert!(r.ok(), "{r:?} {st:?}");
        assert!(st.full > 0, "{st:?}");
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 1.02f64.powi(3);
        assert!(
            (r.volume(1) - exact).abs() < 0.05 * exact,
            "{}",
            r.volume(1)
        );
    }

    #[test]
    fn a_move_that_inverts_is_refused() {
        // Snap far inward: surface vertices would pass through the interior
        // ones, so some moves must be cut short or refused.
        let d = Balls::new([0.0; 3], &[1.0]);
        let (mut c, _) = mesh(&d, &Uniform(0.25), &Params::default());
        let st = snap(&mut c, &Sphere(0.6));
        let r = check(&c);
        assert_eq!(r.inverted, 0, "{r:?}");
        assert!(st.partial + st.blocked > 0, "{st:?}");
    }
}
