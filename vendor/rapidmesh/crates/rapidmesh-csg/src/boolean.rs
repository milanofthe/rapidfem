//! Boolean operations on closed solids via the exact arrangement.
//!
//! Both operands are arranged together, every sub-triangle is classified
//! against the other solid by its exact barycenter, and a per-operation keep
//! table selects (and possibly flips) the surviving sub-triangles. Output
//! vertices stay exact ([`Point3`], possibly implicit); rounding to f64 is a
//! separate downstream decision.
//!
//! NOTE: the meshing pipeline itself does NOT go through here -- scene assembly
//! (`rapidmesh_geom::Scene::assemble`) runs `arrange_facets` + `classify`
//! directly, keeping regions instead of a single solid. This module is the
//! standalone mesh-boolean API (`Faceted::to_solid` feeds it) and doubles as
//! the end-to-end watertightness exercise of the arrangement machinery.

use crate::arrange::{arrange, ArrangeError};
use crate::classify::{Classifier, Placement, Sample};
use crate::pool::VertexPool;
use crate::tri::Tri;
use rapidmesh_exact::Point3;

/// A closed, outward-oriented triangle mesh.
#[derive(Debug, Clone)]
pub struct Solid {
    /// The triangles. Watertightness and outward orientation are the
    /// caller's responsibility (builders guarantee it for primitives).
    pub tris: Vec<Tri>,
}

/// A regularized boolean operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    /// A union B.
    Union,
    /// A intersect B.
    Intersection,
    /// A minus B.
    Difference,
}

/// The boundary surface of a boolean result, with exact vertices.
#[derive(Debug)]
pub struct BooleanResult {
    /// Globally deduplicated exact vertices.
    pub vertices: Vec<Point3>,
    /// Output triangles (outward-oriented).
    pub triangles: Vec<[usize; 3]>,
    /// Per output triangle: index of the input facet it came from
    /// (0..a.tris.len() for A, then B) -- carries tags downstream.
    pub source_facet: Vec<usize>,
}

/// Keep table: `Some(flip)` if a sub-triangle with this placement survives.
///
/// Coplanar-coincident regions are kept once (from the A side) when their
/// orientation matches the result's boundary there: same-normal for union
/// and intersection, opposite-normal for difference (B touching A from
/// outside leaves A's wall intact; coincident interiors cancel).
fn keep(op: BoolOp, from_a: bool, placement: Placement) -> Option<bool> {
    match (op, placement) {
        (BoolOp::Union, Placement::Outside) => Some(false),
        (BoolOp::Union, Placement::Boundary { same_normal: true }) if from_a => Some(false),
        (BoolOp::Intersection, Placement::Inside) => Some(false),
        (BoolOp::Intersection, Placement::Boundary { same_normal: true }) if from_a => Some(false),
        (BoolOp::Difference, Placement::Outside) if from_a => Some(false),
        (BoolOp::Difference, Placement::Inside) if !from_a => Some(true),
        (BoolOp::Difference, Placement::Boundary { same_normal: false }) if from_a => Some(false),
        _ => None,
    }
}

/// Regularized boolean of two closed solids. The result boundary is exact;
/// shared/coincident surface regions are handled by the keep table. A facet
/// the arrangement cannot triangulate is an error naming it (`a`'s facets
/// first, then `b`'s).
pub fn boolean(a: &Solid, b: &Solid, op: BoolOp) -> Result<BooleanResult, ArrangeError> {
    let mut all: Vec<Tri> = a.tris.clone();
    all.extend(b.tris.iter().cloned());
    let na = a.tris.len();
    let arr = arrange(&all)?;

    let solids = Classifier::new(vec![&a.tris, &b.tris]);
    let mut pool = VertexPool::default();
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    let mut source_facet: Vec<usize> = Vec::new();
    for (fi, ft) in arr.facets.iter().enumerate() {
        let from_a = fi < na;
        let other = if from_a { 1 } else { 0 };
        for sub in &ft.triangles {
            let (p0, p1, p2) = (
                &ft.vertices[sub[0]],
                &ft.vertices[sub[1]],
                &ft.vertices[sub[2]],
            );
            let sample = Sample::of([p0, p1, p2], &all[fi]);
            let placement = solids.place(other, &sample, &all[fi]);
            let Some(flip) = keep(op, from_a, placement) else {
                continue;
            };
            let i0 = pool.insert(p0.clone());
            let i1 = pool.insert(p1.clone());
            let i2 = pool.insert(p2.clone());
            triangles.push(if flip { [i0, i2, i1] } else { [i0, i1, i2] });
            source_facet.push(fi);
        }
    }
    Ok(BooleanResult {
        vertices: pool.verts,
        triangles,
        source_facet,
    })
}
