//! Tagged piecewise-linear complex: the central intermediate representation.

use crate::faceted::SurfaceKind;
use crate::vec3::len;

/// Identifies the analytic surface a PLC facet originated from, so the
/// mesher and the second-order nodes follow the exact geometry instead of the
/// linear facet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceRef(pub u32);

/// Region (material) tag carried through CSG into the volume mesh. Every output
/// tet lies in exactly one region -- conformal material interfaces are a hard
/// requirement for Maxwell FEM. `RegionTag(0)` is the background (outside all
/// scene solids).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct RegionTag(pub u32);

/// Boundary/face tag for ports, PEC surfaces, ABC/PML interfaces.
/// `FaceTag(0)` means untagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FaceTag(pub u32);

/// Watertight tagged triangle surface complex.
///
/// Coordinates are expected normalized to a unit box by the builder: the
/// Shewchuk-style robust predicates underneath do not handle exponent
/// overflow (inputs outside ~[1e-142, 1e201] lose their guarantee).
#[derive(Debug, Default, Clone)]
pub struct TaggedPlc {
    /// Vertex coordinates, xyz interleaved.
    pub vertices: Vec<[f64; 3]>,
    /// Triangle vertex indices.
    pub triangles: Vec<[u32; 3]>,
    /// Per-triangle face tag.
    pub face_tags: Vec<FaceTag>,
    /// Per-triangle back-reference to the originating analytic surface.
    pub surface_refs: Vec<SurfaceRef>,
    /// Per-triangle region tags on both sides (front, back) of the facet.
    /// Front is the side the triangle normal points into.
    pub region_tags: Vec<[RegionTag; 2]>,
    /// The analytic surfaces referenced by `surface_refs`.
    pub surfaces: Vec<SurfaceKind>,
    /// Per-surface owner: the index of the scene solid (insertion order,
    /// voids included) whose facets produced the surface, or
    /// [SHEET_OWNER] for sheet surfaces. Parallel to `surfaces`.
    pub surface_owners: Vec<u32>,
    /// Per-surface role: its index among the surfaces of the shape it came
    /// from, in the order each primitive documents (a box: -z, +z, -y, +y,
    /// -x, +x). With the owner it names a surface by its origin, whatever
    /// else the scene holds. Parallel to `surfaces`; empty where unknown.
    pub surface_roles: Vec<u32>,
    /// Per scene solid (the owner index): the frame it was built in (see
    /// [`crate::Frame`]). Empty where unknown.
    pub owner_frames: Vec<crate::Frame>,
    /// Triangle edges (vertex pairs, lower index first) that are features
    /// without a change of surface: the input shapes' explicit feature
    /// segments, split where the arrangement split them.
    pub features: Vec<[u32; 2]>,
    /// Vertices the input shapes declare as corners (see
    /// `Faceted::corners`), sorted.
    pub corners: Vec<u32>,
    /// The exact edge curves the input shapes declare (see
    /// `Faceted::curves`); a B-rep edge whose chain lies on one takes it.
    pub curves: Vec<crate::EdgeCurve>,
}

/// Owner value in [TaggedPlc::surface_owners] for surfaces that belong to an
/// embedded sheet rather than a solid (sheets are addressed by face tag).
pub const SHEET_OWNER: u32 = u32::MAX;

impl TaggedPlc {
    /// The thickness of each region, `2 V / S` from its volume `V` and its
    /// boundary area `S`: the thickness of a plate, the radius of a wire,
    /// whatever their orientation. Sheets inside a region bound nothing
    /// and count for neither. Regions in ascending order, the outside (0)
    /// left out.
    pub fn region_thickness(&self) -> Vec<(u32, f64)> {
        use std::collections::BTreeMap;
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &self.vertices {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        // Volumes about the box center, which keeps the sums well scaled.
        let o: [f64; 3] = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
        let mut vs: BTreeMap<u32, (f64, f64)> = BTreeMap::new();
        for (t, r) in self.triangles.iter().zip(&self.region_tags) {
            let [front, back] = [r[0].0, r[1].0];
            if front == back {
                continue;
            }
            let p = t.map(|v| {
                let q = self.vertices[v as usize];
                [q[0] - o[0], q[1] - o[1], q[2] - o[2]]
            });
            let n = [
                (p[1][1] - p[0][1]) * (p[2][2] - p[0][2])
                    - (p[1][2] - p[0][2]) * (p[2][1] - p[0][1]),
                (p[1][2] - p[0][2]) * (p[2][0] - p[0][0])
                    - (p[1][0] - p[0][0]) * (p[2][2] - p[0][2]),
                (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                    - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]),
            ];
            let area = 0.5 * len(n);
            // The normal points out of the back region, into the front one.
            let vol = (p[0][0] * n[0] + p[0][1] * n[1] + p[0][2] * n[2]) / 6.0;
            for (reg, sign) in [(back, 1.0), (front, -1.0)] {
                if reg != 0 {
                    let e = vs.entry(reg).or_insert((0.0, 0.0));
                    e.0 += sign * vol;
                    e.1 += area;
                }
            }
        }
        vs.into_iter()
            .filter(|(_, (v, s))| *v > 0.0 && *s > 0.0)
            .map(|(r, (v, s))| (r, 2.0 * v / s))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{solid_box, Scene};

    /// A plate's thickness is about its thickness, a cube's a third of its
    /// side, each region its own.
    #[test]
    fn region_thickness_of_a_plate_on_a_cube() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
        scene.add_solid(solid_box([0.0, 0.0, 1.0], [4.0, 4.0, 1.05]));
        let th = scene.assemble().region_thickness();
        // The plate: V = 16 * 0.05, S = 2 * 16 + 16 * 0.05.
        let (cube, plate) = (1.0 / 3.0, 2.0 * 0.8 / 32.8);
        assert_eq!(th.len(), 2, "{th:?}");
        for want in [cube, plate] {
            assert!(th.iter().any(|&(_, t)| (t - want).abs() < 1e-9), "{th:?}");
        }
    }
}
