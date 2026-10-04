//! STEP (ISO 10303-21, AP203/AP214) onto the model: each solid of a file as
//! a closed faceted solid whose facets carry their face's surface (planes,
//! quadrics, tori, B-splines), so the arrangement, the B-rep and the mesher
//! take it like any other shape and measure against the true surfaces.

pub mod entities;
pub mod geometry;
pub mod part21;
pub mod tessellate;

pub use entities::{Model, StepError};
pub use tessellate::{Body, Tolerance};

/// A STEP file read: its solids and the length of its unit in metres.
pub struct Step {
    pub bodies: Vec<Body>,
    pub metres_per_unit: f64,
}

/// Reads the solids of the STEP file `text`.
pub fn read(text: &str, tol: Tolerance) -> Result<Step, String> {
    let x = part21::parse(text).map_err(|e| e.to_string())?;
    let m = entities::decode(&x).map_err(|e| e.to_string())?;
    if m.solids.is_empty() {
        return Err("no solid (MANIFOLD_SOLID_BREP) in the file".into());
    }
    let bodies = tessellate::bodies(&m, tol).map_err(|e| e.to_string())?;
    Ok(Step {
        bodies,
        metres_per_unit: m.metres_per_unit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> String {
        let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "fixtures", name]
            .iter()
            .collect();
        std::fs::read_to_string(p).unwrap()
    }

    /// Every fixture reads into closed solids: each edge of the facets is
    /// used once in each direction.
    #[test]
    fn fixtures_read_into_closed_solids() {
        for (name, solids) in [
            ("bracket.step", 1),
            ("loft.step", 1),
            ("assembly.step", 3),
            ("flange.step", 5),
            // A surface of revolution, a solid with a void, a whole sphere.
            ("turned_part.step", 1),
            // Bounds that fold where a pad's rim touches a hole's.
            ("castellated_module.step", 1),
        ] {
            let s = read(&fixture(name), Tolerance::default())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(s.bodies.len(), solids, "{name}");
            assert_eq!(s.metres_per_unit, 1e-3);
            for b in &s.bodies {
                let mut edges: rustc_hash::FxHashMap<[[u64; 3]; 2], i32> = Default::default();
                for t in &b.solid.tris {
                    for k in 0..3 {
                        let (p, q) = (t.v[k].map(f64::to_bits), t.v[(k + 1) % 3].map(f64::to_bits));
                        *edges.entry([p.min(q), p.max(q)]).or_default() +=
                            if p < q { 1 } else { -1 };
                    }
                }
                let open = edges.values().filter(|&&c| c != 0).count();
                assert_eq!(open, 0, "{name}: {} open edges in {}", open, b.name);
            }
        }
    }
}
