//! The canonical comparison geometries: EM-representative CSG scenes used by
//! the viewer export, the benchmark harness, and (rebuilt analytically) the
//! gmsh reference script. One definition, every consumer.

use rapidmesh_geom::{cylinder, extrude_polygon, sheet_rect, solid_box, sphere, FaceTag, Scene};

/// One named comparison geometry with its meshing sizes.
pub struct BenchScene {
    /// Geometry name (viewer JSON file stem).
    pub name: &'static str,
    /// The scene.
    pub scene: Scene,
    /// Global target edge length.
    pub maxh: f64,
    /// Per-region target edge length overrides.
    pub region_maxh: Vec<(u32, f64)>,
}

/// Builds all comparison scenes in canonical order.
pub fn comparison_scenes() -> Vec<BenchScene> {
    let mut out = Vec::new();

    // 1. The EM test scene: air box + dielectric block + PEC patch + sheet.
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
        let diel = scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
        scene.add_sheet(
            sheet_rect([1.5, 1.5, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            FaceTag(7),
        );
        scene.add_sheet(
            sheet_rect([0.5, 0.5, 3.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            FaceTag(7),
        );
        out.push(BenchScene {
            name: "em_scene",
            scene,
            maxh: 0.9,
            region_maxh: vec![(diel.0, 0.45)],
        });
    }

    // 2. Cylindrical via through a dielectric block.
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]));
        let via = scene.add_solid(cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.75, 12));
        out.push(BenchScene {
            name: "via",
            scene,
            maxh: 0.6,
            region_maxh: vec![(via.0, 0.3)],
        });
    }

    // 3. Microstrip-like: substrate + air, PEC strip on the interface.
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [6.0, 3.0, 3.0]));
        let subst = scene.add_solid(solid_box([0.0, 0.0, 0.0], [6.0, 3.0, 0.5]));
        scene.add_sheet(
            sheet_rect([0.0, 1.25, 0.5], [6.0, 0.0, 0.0], [0.0, 0.5, 0.0]),
            FaceTag(7),
        );
        out.push(BenchScene {
            name: "microstrip",
            scene,
            maxh: 0.8,
            region_maxh: vec![(subst.0, 0.35)],
        });
    }

    // 4. Sphere in a box (curved geometry).
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]));
        let ball = scene.add_solid(sphere([0.0, 0.0, 0.0], 1.0, 16, 8));
        out.push(BenchScene {
            name: "sphere",
            scene,
            maxh: 0.8,
            region_maxh: vec![(ball.0, 0.4)],
        });
    }

    // 5. L-shaped extruded resonator cavity inside air.
    {
        let l_shape = vec![
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let mut scene = Scene::new();
        scene.add_solid(solid_box([-1.0, -1.0, -1.0], [4.0, 3.0, 2.0]));
        scene.add_solid(extrude_polygon(
            &l_shape,
            &[],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ));
        out.push(BenchScene {
            name: "l_prism",
            scene,
            maxh: 0.5,
            region_maxh: Vec::new(),
        });
    }

    // 6. Density transition: fine dielectric inside coarse air.
    {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]));
        let diel = scene.add_solid(solid_box([1.0, 1.0, 1.0], [3.0, 3.0, 2.0]));
        out.push(BenchScene {
            name: "density_transition",
            scene,
            maxh: 1.4,
            region_maxh: vec![(diel.0, 0.45)],
        });
    }

    out
}
