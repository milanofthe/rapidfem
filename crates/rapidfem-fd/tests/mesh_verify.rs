/// Verify mesh connectivity matches EMerge exactly.
use rapidfem_fd::mesh_io::load_mesh;
use rapidfem_fd::basis::Nedelec2Basis;

// The WR-90 EMerge fixture lives at the repo-root `tests/meshes/`. cargo runs
// a test binary with its cwd set to the crate dir, so a path relative to the
// crate would miss it; anchor on CARGO_MANIFEST_DIR instead.
const WR90_MESH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"), "/../../tests/meshes/wr90_straight.msh");

#[test]
fn test_mesh_counts_vs_emerge() {
    let mesh = load_mesh(WR90_MESH).expect("Load mesh");
    let basis = Nedelec2Basis::new(&mesh);

    eprintln!("Our mesh: {} nodes, {} edges, {} tris, {} tets",
        mesh.n_nodes(), mesh.n_edges(), mesh.n_tris(), mesh.n_tets());
    eprintln!("n_field = {}", basis.n_field);

    // PEC DOFs (tag 1)
    let pec_tris = mesh.tris_for_tag(1);
    let mut pec_ids = std::collections::HashSet::new();
    for &ti in pec_tris {
        let edges = &mesh.tri_to_edge[ti];
        for &ei in edges {
            for &d in &basis.edge_to_field[ei] { pec_ids.insert(d); }
        }
        for &d in &basis.tri_to_field[ti] { pec_ids.insert(d); }
    }
    eprintln!("PEC DOFs: {}", pec_ids.len());
    eprintln!("Free DOFs: {}", basis.n_field - pec_ids.len());

    let port1_tris = mesh.tris_for_tag(3);
    let port2_tris = mesh.tris_for_tag(4);
    eprintln!("Port1: {} tris, Port2: {} tris, PEC: {} tris",
        port1_tris.len(), port2_tris.len(), pec_tris.len());

    // EMerge reference: 222 nodes, 1025 edges, 1404 tris, 600 tets
    assert_eq!(mesh.n_nodes(), 222, "Node count mismatch");
    assert_eq!(mesh.n_tets(), 600, "Tet count mismatch");
    assert_eq!(mesh.n_edges(), 1025, "Edge count mismatch: got {}, expected 1025", mesh.n_edges());
    assert_eq!(mesh.n_tris(), 1404, "Tri count mismatch: got {}, expected 1404", mesh.n_tris());
    assert_eq!(basis.n_field, 4858, "n_field mismatch");
    assert_eq!(pec_ids.len(), 1636, "PEC DOF count mismatch");
    assert_eq!(port1_tris.len(), 44, "Port1 tri count mismatch");
    assert_eq!(port2_tris.len(), 44, "Port2 tri count mismatch");

    eprintln!("Mesh connectivity: PASS — all counts match EMerge");
}
