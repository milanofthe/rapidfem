//! Exact arithmetic foundation: expansions, interval filters, indirect predicates;
//! and the plain float vector algebra every crate shares ([`vector`]).
//!
//! The exact mesh CSG (arrangements with implicitly represented intersection
//! points), the planar triangulations and the inside/outside classification
//! stand on this crate. The shared mechanism is staged sign evaluation of
//! polynomial expressions over f64 inputs:
//!
//! 1. **Interval filter** ([`Interval`]): conservative interval arithmetic with
//!    one-ulp outward widening per operation. Fast; resolves the sign in the
//!    vast majority of non-degenerate cases.
//! 2. **Exact fallback** ([`Expansion`]): Shewchuk-style floating-point
//!    expansion arithmetic, continued in big dyadic numbers outside the
//!    exponent range where its products stay exact. Slow but exact; resolves
//!    every case.
//!
//! Geometric expressions (determinants, implicit-point coordinates) are written
//! once, generically over the [`Ring`] trait, and evaluated with either number
//! type -- or with a rational type in tests, which serves as the correctness
//! oracle.
//!
//! Implicit points ([`Point3::Lpi`], [`Point3::Tpi`]) are represented by their
//! defining primitives (line/plane, three planes) and evaluated lazily as
//! homogeneous coordinates whose entries are polynomials in the input
//! coordinates -- no constructed (rounded) coordinates ever enter a predicate.
//!
//! # Input domain
//!
//! Every finite input. Expansions whose products would leave the exact
//! exponent range (implicit points of a high degree over tiny or huge
//! coordinates) continue as big dyadic numbers, see [`Expansion`]. The fast
//! path for fully explicit predicates (`geometry-predicates`) assumes
//! coordinates inside ~[1e-142, 1e201], which the unit-box normalization
//! upstream satisfies with an enormous margin.
//!
//! # Provenance (licensing)
//!
//! Everything here is implemented from publicly described methods: Shewchuk's
//! expansion arithmetic and predicates (1997 paper, public-domain reference
//! technique), and the indirect/implicit-point predicate concept as described
//! in the papers of Attene (2020) and Cherchi et al. (2022). No GPL/AGPL code
//! was consulted. The `geometry-predicates` dependency (MIT) provides the fast
//! adaptive path for fully explicit predicates.

pub mod clock;
pub mod expansion;
pub mod geom;
pub mod interval;
pub mod log;
pub mod mem;
pub mod order;
pub mod orient;
pub mod point;
pub mod proj;
pub mod ring;
pub mod vector;

pub use expansion::Expansion;
pub use interval::Interval;
pub use order::{cmp_along, collinear, lex_cmp, strictly_between, within_closed};
pub use orient::{incircle2d, orient2d, orient3d, orient3d_explicit, Prepared3};
pub use point::Point3;
pub use ring::Ring;

/// A coordinate axis, used to select axis-aligned 2D projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// The x axis.
    X,
    /// The y axis.
    Y,
    /// The z axis.
    Z,
}

impl Axis {
    /// Coordinate index of the axis.
    pub fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }

    /// The two coordinates a projection along the axis keeps, paired
    /// cyclically (dropping x keeps (y, z), y keeps (z, x), z keeps
    /// (x, y)): a triangle's projected orientation is then the sign of the
    /// dropped component of its normal.
    pub fn kept(self) -> [usize; 2] {
        match self {
            Axis::X => [1, 2],
            Axis::Y => [2, 0],
            Axis::Z => [0, 1],
        }
    }

    /// `p` projected along the axis (see [`Axis::kept`]).
    pub fn project(self, p: [f64; 3]) -> [f64; 2] {
        self.kept().map(|k| p[k])
    }
}

/// Sign of an exactly evaluated quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    /// Strictly negative.
    Negative,
    /// Exactly zero.
    Zero,
    /// Strictly positive.
    Positive,
}

impl Sign {
    /// Sign of a (finite) f64.
    pub fn of_f64(v: f64) -> Sign {
        if v > 0.0 {
            Sign::Positive
        } else if v < 0.0 {
            Sign::Negative
        } else {
            Sign::Zero
        }
    }

    /// Sign of a product: combines two signs multiplicatively.
    pub fn combine(self, other: Sign) -> Sign {
        match (self, other) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (a, b) if a == b => Sign::Positive,
            _ => Sign::Negative,
        }
    }

    /// The opposite sign.
    pub fn flip(self) -> Sign {
        match self {
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
            Sign::Positive => Sign::Negative,
        }
    }
}
