//! Faceted shapes: tessellated triangle meshes with analytic surface
//! back-references.

use crate::curve::Curve;
use crate::surface::Surface;
use rapidmesh_csg::{PlanarFacet, Solid, Tri};
use rapidmesh_exact::vector::{Affine, V3};

/// A flat (planar) face carried both as its helper triangulation (the range of
/// `Faceted::tris` that tiles it) and as its first-class boundary polygon. The
/// conformal arrangement intersects the helper triangles but accumulates the
/// resulting constraints at the polygon level, so a flat face has no interior
/// vertices of its own and curves piercing it land on their own vertices.
/// Curved faces (barrels, spheres,
/// tori) carry no `FlatFacet`; their triangles stay first-class.
#[derive(Debug, Clone)]
pub struct FlatFacet {
    /// Boundary loops of the face, in this shape's coordinates.
    pub facet: PlanarFacet,
    /// Index into `Faceted::surfaces` (shared with the helper triangles).
    pub surface: u32,
    /// The helper triangles of this face: `Faceted::tris[tris]`.
    pub tris: std::ops::Range<usize>,
}

/// A tessellated shape: triangles plus, per triangle, the analytic surface
/// it approximates. Used for both closed solids and open sheets; closedness
/// and orientation are invariants of the builder that produced it.
#[derive(Debug, Clone)]
pub struct Faceted {
    /// Where the shape was built: the map from its own frame to the scene
    /// (identity for a new shape, followed by every transform). Entities
    /// that nothing else tells apart are ordered by where they lie in it,
    /// which a turn or a move of the shape leaves as it is.
    pub frame: Affine,
    /// Points of its boundary the shape declares as corners (the vertices
    /// of a polygon sheet): its B-rep edges end there even where nothing
    /// else would tell (one face, one boundary).
    pub corners: Vec<[f64; 3]>,
    /// The triangles.
    pub tris: Vec<Tri>,
    /// Per-triangle index into `surfaces`.
    pub face_surface: Vec<u32>,
    /// The distinct analytic surfaces of this shape.
    pub surfaces: Vec<Option<Surface>>,
    /// Flat faces as first-class boundary polygons (each backed by a range of
    /// `tris`). Empty until a builder registers them; curved faces never do.
    pub flats: Vec<FlatFacet>,
    /// Feature segments along triangle edges that no change of surface marks:
    /// creases inside one smooth region of an import (an open crease a region
    /// wraps around). They become B-rep edges inside their face.
    pub features: Vec<[[f64; 3]; 2]>,
    /// Exact edge curves the shape declares (the B-spline edges of a CAD
    /// file): the B-rep edges along them take them as their carriers.
    pub curves: Vec<EdgeCurve>,
}

/// An exact edge curve of a shape: its carrier and the points of it, in
/// order, that the shape's triangles have along the edge.
#[derive(Debug, Clone)]
pub struct EdgeCurve {
    pub curve: Curve<3>,
    pub points: Vec<[f64; 3]>,
}

impl Faceted {
    /// Empty shape.
    pub fn new() -> Faceted {
        Faceted {
            frame: Affine::IDENTITY,
            corners: Vec::new(),
            tris: Vec::new(),
            face_surface: Vec::new(),
            surfaces: Vec::new(),
            flats: Vec::new(),
            features: Vec::new(),
            curves: Vec::new(),
        }
    }

    /// Registers a surface and returns its index.
    pub fn add_surface(&mut self, s: impl Into<Option<Surface>>) -> u32 {
        self.surfaces.push(s.into());
        (self.surfaces.len() - 1) as u32
    }

    /// Adds a triangle on the given surface.
    pub fn push_tri(&mut self, t: Tri, surface: u32) {
        self.tris.push(t);
        self.face_surface.push(surface);
    }

    /// Adds a flat polygonal face: pushes its helper triangulation `tris` (all
    /// on `surface`, and which must exactly tile the polygon) and records the
    /// boundary `facet` as a first-class planar facet for the conformal
    /// arrangement, its loops wound about the triangles' normal whichever
    /// way the caller gave them.
    pub fn push_flat(&mut self, facet: PlanarFacet, tris: &[Tri], surface: u32) {
        let mut n = [0.0; 3];
        for t in tris {
            let [a, b, c] = t.v;
            let u: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
            let w: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
            n[0] += u[1] * w[2] - u[2] * w[1];
            n[1] += u[2] * w[0] - u[0] * w[2];
            n[2] += u[0] * w[1] - u[1] * w[0];
        }
        let facet = facet.wound_about(n);
        let start = self.tris.len();
        for t in tris {
            self.push_tri(*t, surface);
        }
        self.flats.push(FlatFacet {
            facet,
            surface,
            tris: start..self.tris.len(),
        });
    }

    /// This shape with `cutter` taken away, by the exact boolean.
    pub fn minus(&self, cutter: &Faceted) -> Result<Faceted, rapidmesh_csg::ArrangeError> {
        self.boolean(cutter, rapidmesh_csg::BoolOp::Difference)
    }

    /// What this shape and `other` have in common, by the exact boolean.
    pub fn common(&self, other: &Faceted) -> Result<Faceted, rapidmesh_csg::ArrangeError> {
        self.boolean(other, rapidmesh_csg::BoolOp::Intersection)
    }

    /// The exact boolean `op` of this shape and `other`. Every face keeps
    /// its carrier: this shape's faces theirs, `other`'s faces theirs,
    /// appended after this shape's surfaces (so a role past the shape's own
    /// names a face that came from `other`). The result is given as single
    /// triangles, with this shape's frame, corners and features.
    pub fn boolean(
        &self,
        other: &Faceted,
        op: rapidmesh_csg::BoolOp,
    ) -> Result<Faceted, rapidmesh_csg::ArrangeError> {
        let cutter = other;
        let out = rapidmesh_csg::boolean(&self.to_solid(), &cutter.to_solid(), op)?;
        let pts: Vec<[f64; 3]> = out
            .vertices
            .iter()
            .map(|p| p.approx().expect("boolean vertices are valid points"))
            .collect();
        let own = self.tris.len();
        let base = self.surfaces.len() as u32;
        let mut f = Faceted {
            frame: self.frame,
            corners: self.corners.clone(),
            tris: Vec::new(),
            face_surface: Vec::new(),
            surfaces: self
                .surfaces
                .iter()
                .chain(&cutter.surfaces)
                .cloned()
                .collect(),
            flats: Vec::new(),
            features: self.features.clone(),
            curves: self.curves.clone(),
        };
        for (t, &src) in out.triangles.iter().zip(&out.source_facet) {
            let v = t.map(|i| pts[i]);
            let tri = Tri::new(v[0], v[1], v[2]);
            if tri.is_degenerate() {
                continue; // collapsed to a line by the rounding
            }
            let surface = if src < own {
                self.face_surface[src]
            } else {
                base + cutter.face_surface[src - own]
            };
            f.push_tri(tri, surface);
        }
        Ok(f)
    }

    /// The bare triangle soup as a CSG solid operand. Only meaningful for
    /// shapes built as closed, outward-oriented solids.
    pub fn to_solid(&self) -> Solid {
        Solid {
            tris: self.tris.clone(),
        }
    }

    /// Every point the shape is given by: triangle corners, flat boundary
    /// loops and feature segments.
    pub fn for_each_point(&self, mut f: impl FnMut([f64; 3])) {
        for t in &self.tris {
            t.v.iter().for_each(|&p| f(p));
        }
        for fl in &self.flats {
            fl.facet.outer.iter().for_each(|&p| f(p));
            fl.facet.holes.iter().flatten().for_each(|&p| f(p));
        }
        for s in &self.features {
            s.iter().for_each(|&p| f(p));
        }
    }

    /// The shape with every point moved by `map` (a tiny move: the surface
    /// metadata stays). Triangles whose corners come to coincide are
    /// dropped, flat faces whose loops lose a corner keep the rest of it.
    pub fn with_points(&self, map: impl Fn([f64; 3]) -> [f64; 3]) -> Faceted {
        let mut out = Faceted {
            frame: self.frame,
            corners: self.corners.iter().map(|&p| map(p)).collect(),
            tris: Vec::with_capacity(self.tris.len()),
            face_surface: Vec::with_capacity(self.tris.len()),
            surfaces: self.surfaces.clone(),
            flats: Vec::with_capacity(self.flats.len()),
            features: self.features.iter().map(|s| s.map(&map)).collect(),
            curves: self
                .curves
                .iter()
                .map(|c| EdgeCurve {
                    curve: c.curve.clone(),
                    points: c.points.iter().map(|&p| map(p)).collect(),
                })
                .collect(),
        };
        // Old triangle index -> new, to carry the helper ranges of the flats.
        let mut new_index = vec![usize::MAX; self.tris.len()];
        for (i, t) in self.tris.iter().enumerate() {
            let v = t.v.map(&map);
            let tri = Tri::new(v[0], v[1], v[2]);
            if tri.is_degenerate() {
                continue; // collapsed to a line by the map
            }
            new_index[i] = out.tris.len();
            out.tris.push(tri);
            out.face_surface.push(self.face_surface[i]);
        }
        let dedup = |l: &[[f64; 3]]| -> Vec<[f64; 3]> {
            let mut o: Vec<[f64; 3]> = Vec::with_capacity(l.len());
            for &p in l {
                let q = map(p);
                if o.last() != Some(&q) {
                    o.push(q);
                }
            }
            while o.len() > 1 && o.first() == o.last() {
                o.pop();
            }
            o
        };
        for fl in &self.flats {
            let kept: Vec<usize> = fl
                .tris
                .clone()
                .map(|i| new_index[i])
                .filter(|&i| i != usize::MAX)
                .collect();
            let outer = dedup(&fl.facet.outer);
            // A flat whose helpers were dropped is out of order in the new
            // triangle list or degenerate: its triangles stay as curved ones.
            let contiguous = kept.windows(2).all(|w| w[1] == w[0] + 1);
            if kept.len() != fl.tris.len() || !contiguous || outer.len() < 3 {
                continue;
            }
            out.flats.push(FlatFacet {
                facet: PlanarFacet {
                    outer,
                    holes: fl
                        .facet
                        .holes
                        .iter()
                        .map(|h| dedup(h))
                        .filter(|h| h.len() >= 3)
                        .collect(),
                },
                surface: fl.surface,
                tris: kept[0]..kept[kept.len() - 1] + 1,
            });
        }
        out
    }

    /// The shape carried by the affine map `m`: its points, its curves and
    /// its carriers (see [`Surface::mapped`]); a map that mirrors (a
    /// negative determinant) turns every triangle back outward.
    pub fn transformed(&self, m: &Affine) -> Faceted {
        let map = |p: V3| m.point(p);
        let mut out = Faceted {
            frame: self.frame.then(m),
            corners: self.corners.iter().map(|&p| map(p)).collect(),
            tris: self
                .tris
                .iter()
                .map(|t| Tri::new(map(t.v[0]), map(t.v[1]), map(t.v[2])))
                .collect(),
            face_surface: self.face_surface.clone(),
            features: self.features.iter().map(|f| f.map(map)).collect(),
            curves: self
                .curves
                .iter()
                .map(|c| EdgeCurve {
                    curve: c.curve.mapped(m),
                    points: c.points.iter().map(|&p| map(p)).collect(),
                })
                .collect(),
            flats: self
                .flats
                .iter()
                .map(|fl| FlatFacet {
                    facet: fl.facet.map_points(map),
                    surface: fl.surface,
                    tris: fl.tris.clone(),
                })
                .collect(),
            surfaces: self
                .surfaces
                .iter()
                .map(|s| s.as_ref().and_then(|s| s.mapped(m)))
                .collect(),
        };
        if m.det() < 0.0 {
            for t in &mut out.tris {
                t.v.swap(1, 2);
            }
            for fl in &mut out.flats {
                fl.facet = fl.facet.reversed();
            }
        }
        out
    }
}

impl Default for Faceted {
    fn default() -> Self {
        Faceted::new()
    }
}
