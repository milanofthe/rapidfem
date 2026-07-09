//! Graph type used throughout the SCOTCH pipeline.
//!
//! Re-exported from [`rslab_metis::internals::graph`] so that
//! rslab-scotch and rslab-metis share a single CSR graph layout
//! (see `dev/decisions.md` re ordering-crate sharing). This avoids
//! a converter at every call into rslab-metis's coarsening / FM /
//! initial-bisection kernels in S5.
//!
//! The S1 graph compression operates on `Graph` values built from
//! a `CscPattern` via [`Graph::from_csc_pattern`].

pub(crate) use rslab_metis::internals::graph::Graph;
