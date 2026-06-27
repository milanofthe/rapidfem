// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Load .msh files via mshio and build a Mesh.
//! Mirrors the gmsh extraction in mesh3d._pre_update().

use crate::mesh::Mesh;
use mshio::mshfile::ElementType;
use std::collections::HashMap;

/// Load a gmsh .msh file from disk and build a Mesh.
/// Convenience wrapper around `parse_mesh_bytes`.
pub fn load_mesh(path: &str) -> Result<Mesh, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Cannot read {}: {}", path, e))?;
    parse_mesh_bytes(&bytes).map_err(|e| format!("{}: {}", path, e))
}

/// Parse a gmsh .msh file from in-memory bytes and build a Mesh with full connectivity.
/// This is the core loader; `load_mesh` is a thin file-system wrapper around it.
/// Used by Python (numpy bytes) and WASM (Uint8Array) bindings.
pub fn parse_mesh_bytes(bytes: &[u8]) -> Result<Mesh, String> {
    let msh = mshio::parse_msh_bytes(bytes)
        .map_err(|e| format!("Cannot parse mesh bytes: {:?}", e))?;

    // Extract nodes, build a global node list and tag-to-index map
    let msh_nodes = msh.data.nodes
        .as_ref().ok_or("No nodes in mesh")?;

    // Node tags in gmsh are 1-based. We need to map tag → contiguous index.
    let total_nodes = msh_nodes.num_nodes as usize;
    let mut nodes = Vec::with_capacity(total_nodes);
    let mut tag_to_idx: HashMap<u64, usize> = HashMap::with_capacity(total_nodes);

    for block in &msh_nodes.node_blocks {
        // `node_tags` maps tag -> block-local index. We need the inverse
        // (local index -> tag). Invert it once per block into a flat Vec
        // rather than scanning the whole map for every node, which was
        // O(N^2) on a block with sparse tags.
        let block_tags: Option<Vec<u64>> = block.node_tags.as_ref().map(|tag_map| {
            let mut tags = vec![0u64; block.nodes.len()];
            for (&t, &local_i) in tag_map.iter() {
                if local_i < tags.len() {
                    tags[local_i] = t;
                }
            }
            tags
        });
        for (i, node) in block.nodes.iter().enumerate() {
            let idx = nodes.len();
            // Sparse tags: looked up in the inverted block list. Otherwise
            // tags are sequential, implicit from position (min_node_tag + i).
            let tag = match &block_tags {
                Some(tags) => tags[i],
                None => msh_nodes.min_node_tag + idx as u64,
            };
            tag_to_idx.insert(tag, idx);
            nodes.push([node.x, node.y, node.z]);
        }
    }

    // Extract elements
    let msh_elements = msh.data.elements
        .as_ref().ok_or("No elements in mesh")?;

    let mut tets: Vec<[usize; 4]> = Vec::new();
    let mut tet_entity_tags: Vec<i32> = Vec::new();
    let mut surface_tris: Vec<(i32, [usize; 3])> = Vec::new();

    for block in &msh_elements.element_blocks {
        let etype = block.element_type;
        let entity_tag = block.entity_tag;

        match etype {
            ElementType::Tet4 => {
                for el in &block.elements {
                    let n: Vec<usize> = el.nodes.iter()
                        .map(|&t| *tag_to_idx.get(&t).expect("Unknown node tag in tet"))
                        .collect();
                    let tet = [n[0], n[1], n[2], n[3]];
                    tets.push(tet);
                    tet_entity_tags.push(entity_tag);
                }
            }
            ElementType::Tri3 => {
                for el in &block.elements {
                    let n: Vec<usize> = el.nodes.iter()
                        .map(|&t| *tag_to_idx.get(&t).expect("Unknown node tag in tri"))
                        .collect();
                    let mut tri = [n[0], n[1], n[2]];
                    tri.sort();
                    surface_tris.push((entity_tag, tri));
                }
            }
            _ => {} // Skip other element types (lines, points, etc.)
        }
    }

    if tets.is_empty() {
        return Err("No tetrahedra found in mesh".to_string());
    }

    // Build mesh with connectivity
    let mut mesh = Mesh::from_tets(nodes, tets);

    // Map entity tags to ALL associated physical group tags. A single gmsh
    // entity may belong to multiple physical groups (e.g., a substrate volume
    // can be both `substrate` and `_mat_substrate`). Each physical group must
    // see the entity's tets / tris, so we duplicate per group.
    let mut entity_to_physical_2d: HashMap<i32, Vec<i32>> = HashMap::new();
    let mut entity_to_physical_3d: HashMap<i32, Vec<i32>> = HashMap::new();

    if let Some(ref entities) = msh.data.entities {
        for surf in &entities.surfaces {
            if !surf.physical_tags.is_empty() {
                entity_to_physical_2d.insert(
                    surf.tag,
                    surf.physical_tags.iter().map(|&t| t as i32).collect(),
                );
            }
        }
        for vol in &entities.volumes {
            if !vol.physical_tags.is_empty() {
                entity_to_physical_3d.insert(
                    vol.tag,
                    vol.physical_tags.iter().map(|&t| t as i32).collect(),
                );
            }
        }
    }

    // Build ftag_to_tri: physical tag → mesh triangle indices.
    // If entity has no physical group, fall back to its entity tag.
    for (entity_tag, tri_nodes) in &surface_tris {
        let key = (tri_nodes[0], tri_nodes[1], tri_nodes[2]);
        if let Some(&tri_idx) = mesh.inv_tris.get(&key) {
            match entity_to_physical_2d.get(entity_tag) {
                Some(tags) => {
                    for &phys in tags {
                        mesh.ftag_to_tri.entry(phys).or_default().push(tri_idx);
                    }
                }
                None => {
                    mesh.ftag_to_tri.entry(*entity_tag).or_default().push(tri_idx);
                }
            }
        }
    }

    // Build vtag_to_tet (same multi-group handling).
    for (ti, &entity_tag) in tet_entity_tags.iter().enumerate() {
        match entity_to_physical_3d.get(&entity_tag) {
            Some(tags) => {
                for &phys in tags {
                    mesh.vtag_to_tet.entry(phys).or_default().push(ti);
                }
            }
            None => {
                mesh.vtag_to_tet.entry(entity_tag).or_default().push(ti);
            }
        }
    }

    eprintln!("Mesh: {} nodes, {} edges, {} tris, {} tets",
        mesh.n_nodes(), mesh.n_edges(), mesh.n_tris(), mesh.n_tets());

    // Conditioning gate: flag sliver tets before they reach the solver.
    crate::quality::assess(&mesh).warn_if_poor();

    Ok(mesh)
}
