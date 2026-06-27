// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! rapidfem CLI: thin TOML-driven wrapper around `rapidfem_fd::simulation::Simulation`.
//! All simulation logic lives in the library (lib.rs). This file is just I/O glue:
//! parse args → load mesh+config → run sim → write outputs.

use num_complex::Complex64 as C64;
use rapidfem_fd::config;
use rapidfem_fd::constants::PI;
use rapidfem_fd::mesh_io::load_mesh;
use rapidfem_fd::simulation::Simulation;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rapidfem <config.toml>");
        std::process::exit(1);
    }

    let config = config::load_config(&args[1]).expect("Failed to load config");
    let mesh = load_mesh(&config.mesh.file).expect("Failed to load mesh");
    let frequencies = config.frequency.frequencies();
    eprintln!("RapidFEM - {} frequencies", frequencies.len());

    let sim = Simulation::new(mesh, config);

    // ── Eigenmode analysis (if configured) ────────────────────────────────
    if sim.config.eigenmode.is_some() {
        eprintln!("\n--- Eigenmode Analysis ---");
        let modes = sim.run_eigenmode().expect("eigenmode analysis failed");
        eprintln!("\n  {} modes found:", modes.len());
        for (i, mode) in modes.iter().enumerate() {
            let f_ghz = mode.frequency.re / 1e9;
            let q_str = if mode.q_factor.is_finite() {
                format!("{:.1}", mode.q_factor)
            } else {
                "∞".to_string()
            };
            eprintln!("    Mode {}: f = {:.6} GHz, Q = {}", i + 1, f_ghz, q_str);
        }
        if let Some(ref vtk_path) = sim.config.output.vtk {
            for (i, mode) in modes.iter().enumerate() {
                let mode_path = vtk_path
                    .replace(".vtk", &format!("_mode{}.vtk", i + 1))
                    .replace(".vtu", &format!("_mode{}.vtu", i + 1));
                let label = format!("Mode {} f={:.4}GHz", i + 1, mode.frequency.re / 1e9);
                rapidfem_fd::vtk_export::write_vtk(&mode_path, &sim.mesh, &sim.basis, &mode.field, &label)
                    .expect("Failed to write eigenmode VTK");
                eprintln!("    Wrote {}", mode_path);
            }
        }
        if sim.config.frequency.values.is_empty() && sim.config.frequency.range.is_empty() {
            return; // eigenmode-only run
        }
    }

    // ── Frequency sweep ─────────────────────────────────────────────────────
    let result = sim.run_sweep(None).expect("frequency sweep failed");
    eprintln!("\nTotal: {:.3}s for {} frequency points",
        result.solve_time_s, result.frequencies.len());

    for (fi, &freq) in result.frequencies.iter().enumerate() {
        let s = &result.sparams[fi];
        eprint!("  f={:.4e}: ", freq);
        for i in 0..result.n_driven {
            for j in 0..result.n_driven {
                eprint!("|S{}{}|={:.4} ", i + 1, j + 1, s[i][j].norm());
            }
        }
        eprintln!();
    }

    // ── Error estimator (if adaptive config present) ────────────────────────
    if let Some(ref adaptive) = sim.config.adaptive {
        if let (Some(first_freq_sols), Some(&first_freq)) =
            (result.solutions.first(), result.frequencies.first())
        {
            if let Some(first_sol) = first_freq_sols.first() {
                let k0 = rapidfem_fd::excitation::Excitation::new(first_freq, sim.mesh.l0).k0;
                let n_tets = sim.mesh.n_tets();
                let materials_opt = if sim.materials.is_empty() {
                    None
                } else {
                    Some(sim.materials.as_slice())
                };
                let (er_tensors, _) = if let Some(mats) = materials_opt {
                    rapidfem_fd::materials::build_material_tensors(n_tets, mats, first_freq)
                } else {
                    let id: [[C64; 3]; 3] = [
                        [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
                        [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                        [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
                    ];
                    (vec![id; n_tets], vec![id; n_tets])
                };
                let estimate = rapidfem_fd::error_estimator::estimate_error(
                    &sim.mesh, &sim.basis, first_sol, k0, &er_tensors, adaptive.theta,
                );
                let error_path = sim.config.output.vtk.as_deref()
                    .unwrap_or("error.vtk")
                    .replace(".vtk", "_error.vtk");
                rapidfem_fd::vtk_export::write_vtk_error(&error_path, &sim.mesh, &estimate)
                    .expect("Failed to write error VTK");
                eprintln!("Wrote error VTK: {}", error_path);
                let size_path = sim.config.mesh.file.replace(".msh", "_size.pos");
                rapidfem_fd::error_estimator::write_size_field(
                    &size_path, &sim.mesh, &estimate, adaptive.refinement_ratio,
                ).expect("Failed to write size field");
                eprintln!("Wrote size field: {}", size_path);
            }
        }
    }

    // ── VTK export (first frequency, first port) ────────────────────────────
    if let Some(ref path) = sim.config.output.vtk {
        if let (Some(first_freq_sols), Some(&first_freq)) =
            (result.solutions.first(), result.frequencies.first())
        {
            if let Some(first_sol) = first_freq_sols.first() {
                let label = format!("f={:.4e}Hz", first_freq);
                rapidfem_fd::vtk_export::write_vtk(path, &sim.mesh, &sim.basis, first_sol, &label)
                    .expect("Failed to write VTK file");
                eprintln!("Wrote VTK: {}", path);
            }
        }
    }

    // ── Touchstone export ──────────────────────────────────────────────────
    if let Some(ref path) = sim.config.output.touchstone {
        rapidfem_fd::touchstone::write_touchstone(
            path, &result.frequencies, &result.sparams, result.n_driven, sim.config.output.z0,
        ).expect("Failed to write Touchstone file");
        eprintln!("Wrote {}", path);
    }

    // ── Group delay export ─────────────────────────────────────────────────
    if let Some(ref path) = sim.config.output.group_delay {
        write_group_delay(path, &result).expect("Failed to write group delay CSV");
        eprintln!("Wrote group delay: {}", path);
    }

    // ── Far-field radiation pattern (first frequency, first excitation) ─────
    if let Some(ref ff_path) = sim.config.output.farfield {
        if let Some(pattern) = sim.compute_farfield(&result, 0, 0, 91, 72) {
            rapidfem_fd::farfield::write_pattern_csv(ff_path, &pattern)
                .expect("Failed to write far-field CSV");
            eprintln!("  Wrote far-field: {}", ff_path);
            let cuts_path = ff_path.replace(".csv", "_cuts.csv");
            rapidfem_fd::farfield::write_plane_cuts_csv(&cuts_path, &pattern)
                .expect("Failed to write plane cuts CSV");
            eprintln!("  Wrote plane cuts: {}", cuts_path);
        }
    }
}

fn write_group_delay(
    path: &str,
    result: &rapidfem_fd::simulation::SweepResult,
) -> Result<(), Box<dyn std::error::Error>> {
    if result.frequencies.len() < 2 || result.n_driven < 1 {
        return Ok(());
    }
    let nf = result.frequencies.len();
    let n_driven = result.n_driven;
    let mut file = std::fs::File::create(path)?;
    use std::io::Write;
    write!(file, "freq_hz")?;
    for i in 0..n_driven {
        for j in 0..n_driven {
            write!(file, ",tau_g_{}{}_s", i + 1, j + 1)?;
        }
    }
    writeln!(file)?;

    let mut phase: Vec<Vec<f64>> = vec![vec![0.0; nf]; n_driven * n_driven];
    for i in 0..n_driven {
        for j in 0..n_driven {
            let idx = i * n_driven + j;
            let mut prev = result.sparams[0][i][j].arg();
            phase[idx][0] = prev;
            for fi in 1..nf {
                let raw = result.sparams[fi][i][j].arg();
                let diff = raw - prev;
                // Unwrap across an arbitrary number of 2*pi jumps, not just
                // a single one - a coarse sweep can wrap more than once.
                let cur = raw - 2.0 * PI * (diff / (2.0 * PI)).round();
                phase[idx][fi] = cur;
                prev = cur;
            }
        }
    }

    for fi in 0..nf {
        write!(file, "{:.6e}", result.frequencies[fi])?;
        for i in 0..n_driven {
            for j in 0..n_driven {
                let idx = i * n_driven + j;
                let tau = if fi == 0 {
                    -(phase[idx][1] - phase[idx][0]) / (2.0 * PI * (result.frequencies[1] - result.frequencies[0]))
                } else if fi == nf - 1 {
                    -(phase[idx][nf - 1] - phase[idx][nf - 2])
                        / (2.0 * PI * (result.frequencies[nf - 1] - result.frequencies[nf - 2]))
                } else {
                    -(phase[idx][fi + 1] - phase[idx][fi - 1])
                        / (2.0 * PI * (result.frequencies[fi + 1] - result.frequencies[fi - 1]))
                };
                write!(file, ",{:.6e}", tau)?;
            }
        }
        writeln!(file)?;
    }
    Ok(())
}
