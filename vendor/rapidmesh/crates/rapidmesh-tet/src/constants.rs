//! Central tuning constants for the tet mesher.
//!
//! Every numeric knob that shapes mesher BEHAVIOUR (sampling densities, relaxation
//! iteration caps, quality thresholds, spatial-structure capacities, the sizing
//! field's factors) lives here, grouped and documented, so the meshing recipe is
//! tunable from one place rather than scattered across modules. Pure structural
//! data shared by several modules (the tet face table) is here too, to remove
//! duplication. Algorithm-internal sentinels (`NONE`, bit masks, the FP splitter)
//! stay with their algorithms; constants of the lower crates (`rapidmesh-exact`,
//! `-geom`, `-csg`) stay there -- the dependency direction forbids one file across
//! crates, and they are not mesher tuning knobs.

// ---- seeding / domain bounds ----------------------------------------------
/// Fallback base subdivision of the bbox diagonal when no finite size cap exists.
pub(crate) const DEFAULT_SUBDIV: f64 = 8.0;
// ---- quality diagnostics --------------------------------------------------
/// A tet whose smallest dihedral angle is below this (degrees) is a sliver.
pub const SLIVER_DEG: f64 = 10.0;

// ---- spatial structures ----------------------------------------------------
/// Domain octree max refinement depth (`domain.rs`).
pub(crate) const DOMAIN_MAX_DEPTH: u32 = 18;

// ---- topology --------------------------------------------------------------
/// The four faces of a tet as local vertex-index triples (opposite vertex 0..3).
pub(crate) const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

// ---- boundary fidelity (fidelity.rs) ---------------------------------------
/// A mesh interface and the PLC count as matching where they are closer than
/// this fraction of the local mesh size (the straddler/bridge ratio: the chord
/// sagitta of a face spanning up to a diameter stays below it).
pub const FIDELITY_REL: f64 = 0.25;
/// A PLC edge is sharp when its facets bend by more than this (degrees).
pub const FIDELITY_SHARP_DEG: f64 = 30.0;
/// A mesh edge counts as sharp from this bend on (degrees): half the PLC
/// threshold, so a crease the mesh reproduces with a slightly smaller angle
/// still matches.
pub const FIDELITY_MESH_SHARP_DEG: f64 = 15.0;
/// PLC samples per mesh interface face at most (the sampling budget).
pub const FIDELITY_SAMPLES_PER_FACE: usize = 8;

// ---- layered path (mesh3/layered.rs) ----------------------------------------
/// Model tolerance of the layered path, relative to the finest target size:
/// levels, walls and plan points closer than this are one.
pub(crate) const LAYERED_SNAP: f64 = 1e-3;
/// The layered path takes a stack with a layer thinner than this fraction
/// of the (median) size over it. Refinement meshes thicker layers with better
/// tets at up to this factor more density across them; thinner ones blow it
/// up (the passive stacks, at 0.05, by five to ten times).
pub(crate) const LAYERED_THIN: f64 = 0.1;
