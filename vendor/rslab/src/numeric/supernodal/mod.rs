//! The supernodal machinery shared by the LDL^T and LU factorizations: the
//! left-looking schedule and its dependency-count scheduler, the per-node
//! cmod plan, the panel storage the factors live in, and the tree-parallel
//! triangular solves.

pub(crate) mod analysis;
mod forest;
mod input;
mod node;
pub(crate) mod panel;
mod schedule;
pub(crate) mod solve;
mod store;

pub(crate) use forest::ll_forest;
pub(crate) use input::{Input, InputProgram};
pub(crate) use node::{perturb_pivot, CmodPlan, Gloc, Span};
pub(crate) use schedule::{emit_refcount_offsets, Li, LlSchedule};
pub(crate) use store::{Cells, PanelPtr, SlotStore};
