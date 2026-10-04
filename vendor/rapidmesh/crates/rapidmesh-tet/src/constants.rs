//! Tuning constants shared by several modules of the mesher, or that
//! readers look for in one place (the sliver angle, the fidelity
//! thresholds). A knob only one algorithm reads stays next to it.

// ---- seeding / domain bounds ----------------------------------------------
/// Fallback base subdivision of the bbox diagonal when no finite size cap exists.
pub(crate) const DEFAULT_SUBDIV: f64 = 8.0;
// ---- quality diagnostics --------------------------------------------------
/// A tet whose smallest dihedral angle is below this (degrees) is a sliver.
pub const SLIVER_DEG: f64 = 10.0;

// ---- spatial structures ----------------------------------------------------
/// Sizing tree max refinement depth (`sizing/tree.rs`).
pub(crate) const DOMAIN_MAX_DEPTH: u32 = 18;

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
