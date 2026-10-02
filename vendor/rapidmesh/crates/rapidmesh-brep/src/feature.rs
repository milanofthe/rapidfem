//! The local feature size of a B-rep: how far each corner and each point of
//! an edge lies from the nearest feature (corner, edge, face) that does not
//! touch it. Protecting balls (#217) are sized by it: a ball round a corner,
//! or along an edge, that stays clear of everything it does not touch keeps
//! the features apart however sharp the angles between them.
//!
//! A point of an edge is measured against what touches neither the edge nor
//! its corners: near a corner the corner's own size governs, as its ball
//! covers the edge's end.

use crate::index::FacetBvh;
use crate::Model;
use rapidmesh_csg::Tri;
use rapidmesh_geom::vec3::V3;

/// What a triangle of the index stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Feature {
    Corner(u32),
    Edge(u32),
    Face(u32),
}

/// The local feature size of a model's B-rep (see the module).
pub struct FeatureSize {
    index: FacetBvh,
    of: Vec<Feature>,
    /// Per corner, the edges and the faces it touches, sorted.
    corner_edges: Vec<Vec<u32>>,
    corner_faces: Vec<Vec<u32>>,
    /// Per edge, the faces it lies on, sorted.
    edge_faces: Vec<Vec<u32>>,
}

impl FeatureSize {
    /// The index of `model`'s corners, edges and faces.
    pub fn new(model: &Model) -> FeatureSize {
        let (plc, brep) = (&model.plc, &model.brep);
        let mut tris: Vec<Tri> = Vec::new();
        let mut of: Vec<Feature> = Vec::new();
        for (v, vertex) in brep.vertices.iter().enumerate() {
            tris.push(Tri::new(vertex.pos, vertex.pos, vertex.pos));
            of.push(Feature::Corner(v as u32));
        }
        for (e, edge) in brep.edges.iter().enumerate() {
            for w in edge.chain.windows(2) {
                tris.push(Tri::new(w[0], w[1], w[1]));
                of.push(Feature::Edge(e as u32));
            }
        }
        for (f, face) in brep.faces.iter().enumerate() {
            for &t in &face.facets {
                let p = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
                tris.push(Tri::new(p[0], p[1], p[2]));
                of.push(Feature::Face(f as u32));
            }
        }
        let sorted = |mut v: Vec<u32>| {
            v.sort_unstable();
            v.dedup();
            v
        };
        let mut corner_edges: Vec<Vec<u32>> = vec![Vec::new(); brep.vertices.len()];
        for (e, edge) in brep.edges.iter().enumerate() {
            for v in edge.ends {
                corner_edges[v.0 as usize].push(e as u32);
            }
        }
        let corner_edges = corner_edges.into_iter().map(sorted).collect();
        let corner_faces = brep
            .vertices
            .iter()
            .map(|v| sorted(v.faces.iter().map(|f| f.0).collect()))
            .collect();
        let edge_faces = brep
            .edges
            .iter()
            .map(|e| sorted(e.coedges.iter().map(|c| brep.coedge(*c).face.0).collect()))
            .collect();
        FeatureSize {
            index: FacetBvh::build(&tris),
            of,
            corner_edges,
            corner_faces,
            edge_faces,
        }
    }

    /// How far corner `v` lies from the nearest corner, edge or face it does
    /// not touch (infinite where there is none).
    pub fn at_corner(&self, brep: &crate::Brep, v: u32) -> f64 {
        let (edges, faces) = (
            &self.corner_edges[v as usize],
            &self.corner_faces[v as usize],
        );
        let keep = |i: u32| match self.of[i as usize] {
            Feature::Corner(w) => w != v,
            Feature::Edge(e) => edges.binary_search(&e).is_err(),
            Feature::Face(f) => faces.binary_search(&f).is_err(),
        };
        let p = brep.vertices[v as usize].pos;
        self.index
            .nearest_where(p, &keep)
            .map_or(f64::INFINITY, |x| x.1)
    }

    /// How far point `p` of edge `e` lies from the nearest feature that
    /// touches neither the edge nor its corners (infinite where there is
    /// none).
    pub fn at_edge(&self, brep: &crate::Brep, e: u32, p: V3) -> f64 {
        let ends = brep.edges[e as usize].ends.map(|v| v.0 as usize);
        let faces = &self.edge_faces[e as usize];
        let touches_end = |x: &dyn Fn(usize) -> bool| x(ends[0]) || x(ends[1]);
        let keep = |i: u32| match self.of[i as usize] {
            Feature::Corner(w) => w as usize != ends[0] && w as usize != ends[1],
            Feature::Edge(d) => {
                d != e && !touches_end(&|v| self.corner_edges[v].binary_search(&d).is_ok())
            }
            Feature::Face(f) => {
                faces.binary_search(&f).is_err()
                    && !touches_end(&|v| self.corner_faces[v].binary_search(&f).is_ok())
            }
        };
        self.index
            .nearest_where(p, &keep)
            .map_or(f64::INFINITY, |x| x.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{solid_box, Scene};

    fn model(boxes: &[([f64; 3], [f64; 3])]) -> Model {
        let mut scene = Scene::new();
        for &(lo, hi) in boxes {
            scene.add_solid(solid_box(lo, hi));
        }
        Model::of_scene(&scene)
    }

    /// A box's corners and edges lie its shortest side from what they do
    /// not touch.
    #[test]
    fn a_box_has_its_shortest_side_as_its_feature_size() {
        let m = model(&[([0.0; 3], [2.0, 3.0, 4.0])]);
        let fs = FeatureSize::new(&m);
        for v in 0..m.brep.vertices.len() as u32 {
            assert!((fs.at_corner(&m.brep, v) - 2.0).abs() < 1e-9);
        }
        for (e, edge) in m.brep.edges.iter().enumerate() {
            let (a, b) = (edge.chain[0], edge.chain[edge.chain.len() - 1]);
            let mid = std::array::from_fn(|k| 0.5 * (a[k] + b[k]));
            let d = fs.at_edge(&m.brep, e as u32, mid);
            assert!((2.0 - 1e-9..=3.0 + 1e-9).contains(&d), "edge {e}: {d}");
        }
    }

    /// A plate far thinner than the box it lies on: its upper corners lie
    /// its thickness from the box's top, which they do not touch.
    #[test]
    fn a_thin_plate_has_its_thickness_as_its_feature_size() {
        let m = model(&[
            ([0.0; 3], [10.0, 10.0, 10.0]),
            ([2.0, 2.0, 10.0], [6.0, 6.0, 10.1]),
        ]);
        let fs = FeatureSize::new(&m);
        let least = (0..m.brep.vertices.len() as u32)
            .map(|v| fs.at_corner(&m.brep, v))
            .fold(f64::INFINITY, f64::min);
        assert!((least - 0.1).abs() < 1e-9, "{least}");
    }
}
