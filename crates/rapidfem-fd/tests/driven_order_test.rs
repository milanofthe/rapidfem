// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.
//
// The DRIVEN path, at mixed order.
//
// `cavity_spectrum_test` proves the mixed-order SPACE is conforming and that the
// policy picks the right cells, but it does so on the eigenproblem. The driven
// solve adds three things the eigenproblem does not exercise: the Robin/port
// surface term, the excitation RHS, and S-parameter extraction. This drives a real
// two-port waveguide solve through `assemble_and_solve` and checks that a mixed
// order gives the same S-matrix as uniform order 2.
//
// The reduction here is HAND-PICKED, not taken from the policy: cells near the side
// walls, where the TE10 field ~ sin(πx/a) is close to zero, so dropping them to
// order 1 must barely move the S-parameters. That isolates the question this test
// exists for — does the driven assembly handle a mixed-order map, including the
// reduced-order port-face DOFs the minimum rule produces — from the separate
// question of whether the policy chooses well (which cavity_spectrum settles).

mod common;

use common::{boundary_tris, box_mesh};
use num_complex::Complex64 as C64;
use rapidfem_fd::assembly::assemble_and_solve;
use rapidfem_fd::basis::NedelecBasis;
use rapidfem_fd::excitation::Excitation;
use rapidfem_fd::interp::{eval_field_in_tet, TetGrid};
use rapidfem_fd::mesh::Mesh;
use rapidfem_fd::order::OrderMap;
use rapidfem_fd::port::Port;
use rapidfem_fd::sparam::sparam_waveport;
use rapidfem_fd::waveguide::{CoordinateSystem, RectWaveguide};

// WR-90, a short section.
const A: f64 = 0.02286;
const B: f64 = 0.01016;
const L: f64 = 0.020;
const FREQ: f64 = 1.0e10; // 10 GHz, TE10 propagating (cutoff ~6.56 GHz)

fn on_plane(mesh: &Mesh, tri: usize, axis: usize, value: f64) -> bool {
    mesh.tris[tri]
        .iter()
        .all(|&n| (mesh.nodes[n][axis] - value).abs() < 1e-9)
}

/// The boundary triangles split into the two port faces (z=0, z=L) and the PEC
/// walls (everything else).
fn faces(mesh: &Mesh) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let (mut p1, mut p2, mut pec) = (Vec::new(), Vec::new(), Vec::new());
    for t in boundary_tris(mesh) {
        if on_plane(mesh, t, 2, 0.0) {
            p1.push(t);
        } else if on_plane(mesh, t, 2, L) {
            p2.push(t);
        } else {
            pec.push(t);
        }
    }
    (p1, p2, pec)
}

/// A TE10 RectWaveguide port on a z = const face, propagating along `zsign`·ẑ.
fn te10_port(port_number: usize, z: f64, zsign: f64) -> RectWaveguide {
    RectWaveguide {
        port_number,
        power: 1.0,
        mode: (1, 0),
        er: 1.0,
        polarization: 1.0,
        dims: (A, B),
        cs: CoordinateSystem::new(
            [A / 2.0, B / 2.0, z],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, zsign],
        ),
    }
}

/// Solve the two-port guide with the given order map and return the 2x2 S-matrix.
fn solve_s(mesh: &Mesh, orders: OrderMap) -> [[C64; 2]; 2] {
    let basis = NedelecBasis::with_orders(mesh, orders);
    let (pt1, pt2, pec) = faces(mesh);

    let port1 = te10_port(1, 0.0, 1.0);
    let port2 = te10_port(2, L, -1.0);
    let ports: [&dyn Port; 2] = [&port1, &port2];
    let port_tris: [&[usize]; 2] = [&pt1, &pt2];

    let res = assemble_and_solve(mesh, &basis, &ports, &port_tris, &pec, FREQ, None)
        .expect("the driven solve must succeed");
    assert_eq!(res.solutions.len(), 2, "two driven ports -> two excitation solves");

    let exc = Excitation::new(FREQ, mesh.l0);
    let grid = TetGrid::new(mesh);
    let weight = |_x: f64, _y: f64, _z: f64| 1.0; // air

    let mut s = [[C64::new(0.0, 0.0); 2]; 2];
    for (exc_idx, sol) in res.solutions.iter().enumerate() {
        let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
            match grid.find_containing_tet(mesh, x, y, z) {
                Some(tet) => eval_field_in_tet(mesh, &basis, sol, tet, x, y, z),
                None => (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)),
            }
        };
        for (obs_idx, (port, tris)) in [(&port1, &pt1), (&port2, &pt2)].iter().enumerate() {
            let obs_tris: Vec<[usize; 3]> = tris.iter().map(|&ti| mesh.tris[ti]).collect();
            let active = obs_idx == exc_idx;
            s[obs_idx][exc_idx] =
                sparam_waveport(&mesh.nodes, &obs_tris, *port as &dyn Port, &exc, active, &fieldf, &weight, 4);
        }
    }
    s
}

#[test]
fn mixed_order_gives_the_same_s_matrix_as_uniform_order_2() {
    // Fine enough that TE10 is resolved; coarse enough to solve quickly.
    let mesh = box_mesh(A, B, L, 6, 3, 5);
    let (_, _, pec) = faces(&mesh);
    assert!(!pec.is_empty(), "the guide has no PEC walls: face tagging is wrong");

    let s_uniform = solve_s(&mesh, OrderMap::uniform(&mesh, 2));

    // Reduce, to order 1, the cells whose centroid is within 18% of a side wall in
    // x, where sin(πx/a) < 0.55 and the TE10 field carries little energy. That is a
    // genuine order interface (the minimum rule pulls the shared entities of the
    // p2 neighbours down too, including some on the port faces).
    let orders: Vec<u8> = (0..mesh.n_tets())
        .map(|t| {
            let xc: f64 = mesh.tets[t].iter().map(|&n| mesh.nodes[n][0]).sum::<f64>() / 4.0;
            if xc < 0.18 * A || xc > 0.82 * A {
                1
            } else {
                2
            }
        })
        .collect();
    let n1 = orders.iter().filter(|&&p| p == 1).count();
    assert!(n1 > 0 && n1 < mesh.n_tets(), "the test needs a genuine order interface");

    let bm = NedelecBasis::with_orders(&mesh, OrderMap::from_cells(&mesh, orders.clone()));
    let full = NedelecBasis::with_orders(&mesh, OrderMap::uniform(&mesh, 2));
    eprintln!(
        "  {} of {} cells at order 1; DOFs: mixed {}, uniform {} ({:.0}% saved)",
        n1,
        mesh.n_tets(),
        bm.n_field,
        full.n_field,
        100.0 * (1.0 - bm.n_field as f64 / full.n_field as f64)
    );
    assert!(bm.n_field < full.n_field, "the mixed space must have fewer DOFs");

    let s_mixed = solve_s(&mesh, OrderMap::from_cells(&mesh, orders));

    // Compare the whole 2x2. The reduced cells sit where the field is small, so the
    // S-matrix must barely move — but this is a coarse mesh and order 1 vs 2 near a
    // wall is a real change, so allow a few percent rather than machine precision.
    // A broken driven mixed-order path (wrong Robin block size, a mis-scattered
    // reduced port face) would move S by O(1), not O(1e-2).
    let mut worst = 0.0_f64;
    for i in 0..2 {
        for j in 0..2 {
            let d = (s_uniform[i][j] - s_mixed[i][j]).norm();
            worst = worst.max(d);
            eprintln!(
                "  S[{i}][{j}]: uniform {:.4}∠{:+.1}°  mixed {:.4}∠{:+.1}°  |Δ| {:.2e}",
                s_uniform[i][j].norm(),
                s_uniform[i][j].arg().to_degrees(),
                s_mixed[i][j].norm(),
                s_mixed[i][j].arg().to_degrees(),
                d
            );
        }
    }
    assert!(
        worst < 3e-2,
        "the mixed-order S-matrix differs from uniform order 2 by {worst:.3e}: the driven \
         path does not handle a mixed order correctly"
    );

    // And a sanity floor: the driven solve actually produced a wave, not zeros.
    let through = s_uniform[1][0].norm();
    assert!(through > 0.5, "S21 = {through:.3}: the guide is not transmitting, the setup is wrong");
}
