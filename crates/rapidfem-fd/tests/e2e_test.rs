/// End-to-end test: straight WR-90 waveguide.
/// EMerge reference: S11=0.001496, S21=0.999824.

use num_complex::Complex64 as C64;
use rapidfem_fd::mesh_io::load_mesh;
use rapidfem_fd::basis::Nedelec2Basis;
use rapidfem_fd::waveguide::{RectWaveguide, CoordinateSystem};
use rapidfem_fd::assembly::assemble_and_solve;
use rapidfem_fd::sparam::sparam_waveport;
use rapidfem_fd::interp;
use rapidfem_fd::constants::*;

// WR-90 EMerge fixture at the repo-root `tests/meshes/` (cargo sets the test
// binary cwd to the crate dir, so anchor on CARGO_MANIFEST_DIR).
const WR90_MESH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"), "/../../tests/meshes/wr90_straight.msh");

// Full assemble -> solve -> S-parameter-extraction gate on the EMerge WR-90
// mesh. EMerge reference: |S11|=0.001496, |S21|=0.999824. On this identical
// mesh rapidfem measures |S11|=0.000653, |S21|=0.999958 (matched straight
// guide: |S11| sits in the numeric reflection floor, |S21|~1). The gates
// below are set ~10-15x above the measured EMerge deviation so a real
// regression (wrong port Z0, broken assembly/extraction) breaks them while
// honest mesh/extraction differences pass.
#[test]
fn test_straight_waveguide_sparams() {
    let mesh = load_mesh(WR90_MESH).expect("Load mesh");
    let basis = Nedelec2Basis::new(&mesh);

    let freq = 10.0e9;
    let k0 = 2.0 * PI * freq / C0;

    let port1_tris = mesh.tris_for_tag(3).to_vec();
    let port2_tris = mesh.tris_for_tag(4).to_vec();
    let pec_tris = mesh.tris_for_tag(1).to_vec();

    // Port CS from EMerge reference
    let cs1 = CoordinateSystem::new(
        [0.01143, 0.0, 0.00508], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0],
    );
    let cs2 = CoordinateSystem::new(
        [0.01143, 0.03, 0.00508], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0],
    );

    let port1 = RectWaveguide {
        port_number: 1, power: 1.0, mode: (1, 0), er: 1.0,
        polarization: 1.0, dims: (22.86e-3, 10.16e-3), cs: cs1,
    };
    let port2 = RectWaveguide {
        port_number: 2, power: 1.0, mode: (1, 0), er: 1.0,
        polarization: 1.0, dims: (22.86e-3, 10.16e-3), cs: cs2,
    };

    let ports: Vec<&dyn rapidfem_fd::port::Port> = vec![&port1, &port2];
    let port_tris: Vec<&[usize]> = vec![&port1_tris, &port2_tris];

    let result = assemble_and_solve(&mesh, &basis, &ports, &port_tris, &pec_tris, freq, None)
        .expect("assemble_and_solve failed");

    // S-param extraction: exact port of EMerge's approach
    // EMerge: fieldf = self.basis.interpolate_Ef(solution, tetids=port_tets)
    // which calls MATHLIB.ned2_tet_interp — 3D tet interpolation searching all tets.
    let sol0 = &result.solutions[0];

    // Build field evaluator using find_containing_tet (same as EMerge's ned2_tet_interp)
    let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
        match interp::find_containing_tet(&mesh, x, y, z) {
            Some(tet) => interp::eval_field_in_tet(&mesh, &basis, sol0, tet, x, y, z),
            None => (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)),
        }
    };

    // Test: check if find_containing_tet works for port face points
    let qpts = rapidfem_fd::quadrature::gaus_quad_tri(4);
    let test_tri = mesh.tris[port1_tris[0]];
    let v0 = mesh.nodes[test_tri[0]];
    let v1 = mesh.nodes[test_tri[1]];
    let v2 = mesh.nodes[test_tri[2]];
    let mut found = 0;
    let mut missed = 0;
    for qp in &qpts {
        let x = v0[0]*qp[1] + v1[0]*qp[2] + v2[0]*qp[3];
        let y = v0[1]*qp[1] + v1[1]*qp[2] + v2[1]*qp[3];
        let z = v0[2]*qp[1] + v1[2]*qp[2] + v2[2]*qp[3];
        if interp::find_containing_tet(&mesh, x, y, z).is_some() {
            found += 1;
        } else {
            missed += 1;
        }
    }
    eprintln!("  Port1 tri[0] quad points: found={}, missed={}", found, missed);

    // Check local edge mapping for tet 0
    {
        let tet = &mesh.tets[0];
        let tet_edges = &mesh.tet_to_edge[0];
        let global_edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[tet_edges[i]]);
        let l_edge = rapidfem_fd::basis::local_mapping(tet, &global_edge_nodes);
        eprintln!("  Tet 0: nodes={:?}", tet);
        eprintln!("  local_edge_map = {:?}", l_edge);
        eprintln!("  EMerge ref: [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]");
    }

    // Check a few DOF values
    eprintln!("  sol0[0..5] = {:?}", &sol0[0..5].iter().map(|x| format!("{:.4e}", x)).collect::<Vec<_>>());

    // Check field at test point
    let test_x = 0.01;
    let test_y = 0.001;
    let test_z = 0.005;
    if let Some(tet) = interp::find_containing_tet(&mesh, test_x, test_y, test_z) {
        let (ex, ey, ez) = interp::eval_field_in_tet(&mesh, &basis, sol0, tet, test_x, test_y, test_z);
        eprintln!("  Field at ({},{},{}): Ex={:.6e}, Ey={:.6e}, Ez={:.6e}", test_x, test_y, test_z, ex, ey, ez);
        eprintln!("  EMerge ref: Ex=9.39e0-4.59ej, Ey=-1.22e0-1.81ej, Ez=2.82e3-4.51e2j");
    } else {
        eprintln!("  FAILED: could not find tet for test point");
    }

    // S-param extraction
    let port1_tri_verts: Vec<[usize; 3]> = port1_tris.iter().map(|&ti| mesh.tris[ti]).collect();
    let port2_tri_verts: Vec<[usize; 3]> = port2_tris.iter().map(|&ti| mesh.tris[ti]).collect();

    let p1_ref: &dyn rapidfem_fd::port::Port = &port1;
    let p2_ref: &dyn rapidfem_fd::port::Port = &port2;
    let s11 = sparam_waveport(&mesh.nodes, &port1_tri_verts, p1_ref, k0, true, &fieldf, &(|_x: f64, _y: f64, _z: f64| 1.0), 4);
    let s21 = sparam_waveport(&mesh.nodes, &port2_tri_verts, p2_ref, k0, false, &fieldf, &(|_x: f64, _y: f64, _z: f64| 1.0), 4);

    eprintln!("\n=== S-parameters ===");
    eprintln!("  |S11| = {:.6} ({:.1} dB)", s11.norm(), 20.0 * s11.norm().max(1e-10).log10());
    eprintln!("  |S21| = {:.6} ({:.1} dB)", s21.norm(), 20.0 * s21.norm().max(1e-10).log10());
    eprintln!("  |S11|²+|S21|² = {:.6}", s11.norm_sqr() + s21.norm_sqr());
    eprintln!("  EMerge ref: |S11|=0.001496, |S21|=0.999824");

    // EMerge reference on the identical mesh.
    const EMERGE_S11: f64 = 0.001496;
    const EMERGE_S21: f64 = 0.999824;

    let s11m = s11.norm();
    let s21m = s21.norm();
    let passivity = s11.norm_sqr() + s21.norm_sqr();

    // Transmission must track the EMerge reference closely (measured dev
    // ~1.3e-4; gate at 2e-3, ~15x margin).
    assert!(
        (s21m - EMERGE_S21).abs() < 2.0e-3,
        "|S21| = {:.6} deviates {:.2e} from EMerge {:.6} (> 2e-3)",
        s21m, (s21m - EMERGE_S21).abs(), EMERGE_S21,
    );
    // Matched line: reflection stays in the deep floor (ours 6.5e-4, EMerge
    // 1.5e-3). A wrong port impedance or assembly bug lifts this well above.
    assert!(
        s11m < 5.0e-3,
        "|S11| = {:.6} above the 5e-3 matched-line floor (EMerge ref {:.6})",
        s11m, EMERGE_S11,
    );
    // Passive lossless 2-port: |S11|^2 + |S21|^2 <= 1 (mild numeric overshoot).
    assert!(
        passivity <= 1.01,
        "passivity |S11|^2+|S21|^2 = {:.6} above 1.01", passivity,
    );
    eprintln!("  e2e WR-90 S-parameter gate: PASS");
}
