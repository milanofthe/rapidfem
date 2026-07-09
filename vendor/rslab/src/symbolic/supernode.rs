use crate::ordering::elimination_tree::EliminationTree;
use crate::symbolic::profiler::SymbolicProfiler;
use crate::symbolic::small_leaf::SmallLeafParams;
use std::sync::{Arc, Mutex};

/// Parameters controlling supernode amalgamation.
///
/// Parameters for the **symbolic** phase only (ordering, supernode
/// amalgamation, small-leaf grouping). Scaling and other numeric-time concerns
/// live on the numeric factor options, not here.
#[derive(Clone)]
pub struct SupernodeParams {
    /// Minimum number of eliminated columns in a supernode. Nodes with
    /// fewer eliminations are candidates for merging with their parent.
    /// Default: 32 (matching SSIDS). MUMPS uses 5.
    /// Setting nemin=1 effectively disables amalgamation.
    pub nemin: usize,

    /// Opt-in ordering preprocessing. Default `None`.
    ///
    /// Set to `OrderingPreprocess::LdltCompress` to run MC64 symmetric
    /// matching and collapse each matched pair into one super-variable
    /// before handing the graph to AMD/METIS/SCOTCH. Matches MUMPS's
    /// `ICNTL(12) = 2` for SYM=2. Opt-in while the corpus bench is
    /// collected; see `dev/plans/phase-2.6.5-ldlt-compressed-graph.md`.
    pub preprocess: OrderingPreprocess,

    /// Small-leaf-subtree grouping parameters (Phase 2.9). Controls
    /// which true-leaf supernodes are packed into batch groups for
    /// the numeric small-leaf fast path. The detection runs
    /// unconditionally at symbolic time; the numeric phase chooses whether to
    /// use the groups. See `dev/research/phase-2.9-small-leaf-subtree.md`.
    pub small_leaf: SmallLeafParams,

    /// Phase 2.12 amalgamation strategy: controls whether
    /// `find_supernodes`'s adjacency check is enforced naturally by
    /// the existing postorder (`Adjacency`) or by an SSIDS-style
    /// column renumbering that emits a merge-biased postorder
    /// (`Renumber`, default since Phase 2.12).
    ///
    /// `Adjacency` rejects all sibling-merges that would require the
    /// merged supernode to span non-adjacent columns. On bushy
    /// IPM-KKT trees this leaves dozens of small leaves un-amalgamated
    /// (see `dev/research/phase-2.12-column-renumbering.md`).
    ///
    /// `Renumber` runs a merge prediction pass, then re-postorders
    /// the etree to place desired-merge children adjacent to their
    /// parents. The downstream `find_supernodes` adjacency check
    /// then succeeds naturally for every desired merge. SSIDS
    /// `core_analyse.f90:644-685` is the reference.
    pub amalgamation_strategy: AmalgamationStrategy,

    /// Phase 2.13b per-stage symbolic profiler. When `Some`, the
    /// `symbolic_factorize_with_method` driver records elapsed time
    /// per stage (ordering, etree, postorder, col_counts, renumber,
    /// find_supernodes, etc.). When `None`, every timer is bypassed
    /// (zero overhead). See
    /// `dev/research/phase-2.13b-symbolic-profiler.md`.
    pub symbolic_profiler: Option<Arc<Mutex<SymbolicProfiler>>>,

    /// Relaxed/fill-tolerant amalgamation - the multifrontal-throughput lever.
    /// When `Some`, an adjacent child is merged into its parent even when it is
    /// neither a trivial chain nor size-based, as long as the merged supernode
    /// stays within `max_width` columns and the merge introduces at most
    /// `max_extra_rows` explicit-zero rows. This trades a little fill for much
    /// wider dense fronts (higher-rank Schur GEMMs), the standard sparse-direct
    /// lever (PARDISO/MUMPS apply it universally) that closes most of the
    /// per-front-overhead gap on any matrix whose fundamental supernodes are
    /// narrow. Implies the `Renumber`
    /// merge order so bushy multi-child trees can actually merge. Applied only
    /// above [`RELAX_MIN_N`] unknowns so small problems are unaffected. `None`
    /// (default) = structural / size-based merges only.
    pub relax: Option<RelaxAmalgamation>,
}

/// Relaxed (fill-tolerant) amalgamation thresholds. See
/// [`SupernodeParams::relax`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelaxAmalgamation {
    /// Cap on the merged supernode width (eliminated columns).
    pub max_width: usize,
    /// Maximum explicit-zero rows a single relaxed merge may introduce.
    pub max_extra_rows: usize,
}

/// Relaxed amalgamation is applied only to problems with at least this many
/// unknowns; below it the structural/size-based merges already suffice and the
/// extra fill is not worth it (and keeps small-matrix supernode tests stable).
pub const RELAX_MIN_N: usize = 1024;

/// Phase 2.12 amalgamation strategy selector. See
/// [`SupernodeParams::amalgamation_strategy`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AmalgamationStrategy {
    /// Reject merges whose merged supernode would span non-adjacent
    /// columns. Default before Phase 2.12; kept for parity tests
    /// and as the explicit escape hatch when the merge prediction
    /// pass would over-merge a path-like etree (e.g. MUONSINE).
    Adjacency,
    /// Re-postorder the etree to place desired-merge children
    /// adjacent to their parents, then run the standard
    /// adjacency-checked merge. SSIDS-style column renumbering.
    /// Default behavior of [`AmalgamationStrategy::Auto`] on bushy
    /// IPM-KKT etrees: cuts factor time 30-67% on
    /// ACOPR30/CRESC100/LAKES/NELSON/SWOPF.
    /// See `dev/decisions.md` (Phase 2.12 entries).
    Renumber,
    /// Phase 2.13a: shape-dispatched. A cheap O(n) etree predicate
    /// picks `Adjacency` for path / near-path elimination trees and
    /// `Renumber` for bushy ones, eliminating Renumber's
    /// over-merging regression on path-like trees (MUONSINE_0000:
    /// 5.5× → 1.4× MUMPS) while keeping the IPM-KKT tail wins.
    /// Default since Phase 2.13a. See
    /// `dev/research/phase-2.13a-amalgamation-auto.md`.
    #[default]
    Auto,
}

/// Ordering-stage preprocessing flag.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OrderingPreprocess {
    /// No preprocessing. The fill-reducing ordering runs directly on
    /// the symmetric pattern.
    None,
    /// Duff-Pralet symmetric matching + quotient-graph compression.
    /// See `crate::symbolic::ldlt_compress`.
    LdltCompress,
    /// Shape-dispatched: run `LdltCompress` when cheap shape predicates
    /// predict a benefit, else run `None`. See
    /// `crate::symbolic::pick_ordering_preprocess` for the rule.
    ///
    /// Parallels `ScalingStrategy::Auto`. Default since Phase 2.4.4.
    #[default]
    Auto,
}

impl Default for SupernodeParams {
    fn default() -> Self {
        Self {
            nemin: 16,
            preprocess: OrderingPreprocess::Auto,
            small_leaf: SmallLeafParams::default(),
            amalgamation_strategy: AmalgamationStrategy::default(),
            symbolic_profiler: None,
            relax: None,
        }
    }
}

/// A supernode in the assembly tree.
#[derive(Debug, Clone)]
pub struct Supernode {
    /// Range of columns eliminated in this supernode (in the postordered numbering).
    /// The number of eliminated columns is `cols.end - cols.start`.
    pub first_col: usize,
    pub ncol: usize,
    /// Total number of rows in the frontal matrix (nrow >= ncol).
    pub nrow: usize,
    /// Row indices of the frontal matrix (length nrow).
    /// The first `ncol` entries are the eliminated columns themselves;
    /// the remaining `nrow - ncol` are the non-eliminated rows that form
    /// the contribution block.
    pub row_indices: Vec<usize>,
    /// Children supernode indices.
    pub children: Vec<usize>,
    /// Issue #55 Phase B1: column-count budget for *incoming* delayed
    /// pivots from descendants at numeric time. The numeric phase
    /// enforces `n_delayed_in <= delayed_capacity` per supernode; on
    /// overflow it can return `RslabError::DelayBudgetExceeded` (B3)
    /// or - under the rewired CB trigger (B5) - fall back to
    /// MUMPS-style static perturbation as a last-resort recovery.
    ///
    /// `usize::MAX` is the sentinel for "unbounded" - equivalent to
    /// the pre-Phase-B behavior where the numeric phase grew its
    /// frontal in place to accommodate any number of delays. The
    /// symbolic phase replaces this with a finite estimate during
    /// `find_supernodes` (Phase B2); any code path that constructs
    /// a `Supernode` directly (tests, ad-hoc binary entrypoints) can
    /// leave it at `usize::MAX` and observe pre-Phase-B behavior.
    pub delayed_capacity: usize,
}

impl Supernode {
    /// Number of eliminated columns.
    #[inline]
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    /// Number of rows in the contribution block.
    #[inline]
    pub fn contrib_nrow(&self) -> usize {
        self.nrow - self.ncol
    }

    /// Size of the contribution block in f64 entries.
    #[inline]
    pub fn contrib_size(&self) -> usize {
        let cn = self.contrib_nrow();
        cn * cn
    }
}

/// Phase 2.13a - shape predicate threshold.
///
/// `multi_child_frac < THRESH` ⇒ path / near-path tree, dispatch to
/// `Adjacency`. Otherwise dispatch to `Renumber`. Threshold chosen
/// from the etree-shape probe (`src/bin/diag_etree_shape.rs`) on
/// the 7 known-answer matrices. MUONSINE (the only Renumber-loses
/// case in the probe) sits at 0.002; the next-lowest matrix
/// (SWOPF, Renumber-wins) sits at 0.20. 0.05 is comfortably in the
/// gap. See `dev/research/phase-2.13a-amalgamation-auto.md`.
pub const AUTO_MULTI_CHILD_FRAC_THRESHOLD: f64 = 0.05;

/// Phase 2.13a - pick `Adjacency` vs `Renumber` from the etree
/// shape. O(n) on the etree; called once per
/// `symbolic_factorize_with_method` invocation.
///
/// Returns `Adjacency` when the etree is path / near-path
/// (Renumber would over-merge), otherwise `Renumber`. Never returns
/// `Auto`; this function is the resolver for the `Auto` variant.
pub fn pick_amalgamation_strategy(etree: &EliminationTree) -> AmalgamationStrategy {
    let n = etree.n;
    if n == 0 {
        return AmalgamationStrategy::Adjacency;
    }
    let mut child_count = vec![0usize; n];
    for &p in &etree.parent {
        if let Some(par) = p {
            child_count[par] += 1;
        }
    }
    let n_leaves = child_count.iter().filter(|&&c| c == 0).count();
    let n_internal = n - n_leaves;
    if n_internal == 0 {
        // Forest of isolated nodes - no amalgamation opportunities
        // either way. `Adjacency` is the cheap default.
        return AmalgamationStrategy::Adjacency;
    }
    let n_multi_child = child_count.iter().filter(|&&c| c >= 2).count();
    let multi_child_frac = n_multi_child as f64 / n_internal as f64;
    if multi_child_frac < AUTO_MULTI_CHILD_FRAC_THRESHOLD {
        AmalgamationStrategy::Adjacency
    } else {
        AmalgamationStrategy::Renumber
    }
}

/// Detect fundamental supernodes and apply amalgamation.
///
/// A fundamental supernode is a maximal set of consecutive columns j, j+1, ..., j+k
/// where each column's row structure is identical (the same set of row indices,
/// minus the column being eliminated). This is detected by checking that:
/// 1. Column j+1 has exactly one more nonzero than column j (the new diagonal).
/// 2. The parent of j in the elimination tree is j+1.
///
/// After detecting fundamental supernodes, amalgamation merges small nodes
/// using the SSIDS merge rule:
/// 1. Trivial chain: parent has exactly 1 column AND parent nrow == child nrow - child ncol + parent ncol.
///    (i.e., same row structure minus the eliminated columns)
/// 2. Size-based: both parent AND child have < nemin columns.
///
/// `col_row_indices` provides the actual row indices for each column of L
/// (used to build correct frontal row index sets).
///
/// Returns supernodes in postorder (children before parents).
pub fn find_supernodes(
    etree: &EliminationTree,
    col_counts: &[usize],
    params: &SupernodeParams,
) -> Vec<Supernode> {
    let n = etree.n;
    if n == 0 {
        return Vec::new();
    }

    // Relaxed/fill-tolerant amalgamation (the MoM/FEM throughput lever), applied
    // only at scale (`n >= RELAX_MIN_N`) so small problems - and their supernode
    // structure tests - are unaffected. When active it widens supernodes and
    // implies the Renumber merge order (bushy multi-child trees only merge with
    // the renumbered postorder).
    let relax = if n >= RELAX_MIN_N { params.relax } else { None };
    let relax_width = relax.map(|r| r.max_width);
    let relax_rows = relax.map_or(0, |r| r.max_extra_rows);
    let force_renumber = relax.is_some();

    // Step 1: Find fundamental supernodes (shared with predict_merges)
    let fund = find_fundamental_supernodes(etree, col_counts);
    let snode_starts = fund.snode_starts;
    let mut snode_ncols = fund.snode_ncols;
    let snode_parent = fund.snode_parent;
    let n_snodes = snode_starts.len();

    // Step 2: Amalgamation
    // Track which supernodes are merged (absorbed into parent)
    let mut merged_into = vec![None::<usize>; n_snodes];
    // Track the actual first column of each supernode (may change during merging)
    let mut snode_first_col: Vec<usize> = snode_starts.clone();

    // Iteration order: forward (legacy / `Adjacency` strategy) is the
    // historical behavior - children processed in increasing postorder
    // index. On a multi-child parent only the highest-index child is
    // adjacent to the parent, so only one child merges per multi-child
    // parent (this is what `dev/research/phase-2.12-column-renumbering.md`
    // §2 documents).
    //
    // Reverse iteration (`Renumber` strategy) processes the parent
    // first, then descends to children in decreasing index order.
    // Each merge shrinks the parent's effective `first_col` to the
    // newly-merged child's first_col, opening adjacency for the
    // next-lower-index child. Combined with the merge-biased
    // postorder (which places desired-merge children adjacent to
    // their parent in the column numbering), every desired merge
    // succeeds.
    let reverse =
        force_renumber || matches!(params.amalgamation_strategy, AmalgamationStrategy::Renumber);
    let order: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..n_snodes).rev())
    } else {
        Box::new(0..n_snodes)
    };

    for s in order {
        let sp = snode_parent[s];
        if let Some(p) = sp {
            if find_root(s, &merged_into) != s {
                continue; // already merged into another node
            }

            let root_s = find_root(s, &merged_into);
            let root_p = find_root(p, &merged_into);
            if root_s == root_p {
                continue;
            }

            // Adjacency check: merging is only valid when the child's
            // effective column range [s_first, s_first+s_ncol) is
            // immediately followed by the parent's column range
            // [p_first, p_first+p_ncol). Otherwise the merged
            // supernode's `first_col..first_col+ncol` would no longer
            // be a contiguous block of the column numbering, and
            // downstream code (row-index construction, A-assembly, L
            // storage, solve gather/scatter) would silently claim
            // columns that belong to *other* supernodes.
            //
            // In a postorder-column-numbered elimination tree every
            // parent's columns come after all its descendants', so in
            // a multi-child parent at most one child is adjacent -
            // the one whose last column is parent_first - 1. Merging
            // any other child breaks contiguity. The arrow matrix
            // (variables 0..n-2 all parented by variable n-1) is the
            // archetype: only child n-2 is adjacent to parent n-1.
            //
            // SSIDS side-steps this by emitting a permutation that
            // renumbers columns so merged supernodes are contiguous
            // by construction (`core_analyse.f90:644-685`). That's a
            // strictly better amalgamation policy (merges more
            // children, reduces fill on arrow-like trees) but is a
            // larger refactor. For now the adjacency check is the
            // minimal correctness fix; see
            // `dev/research/phase-2.2.3-plateau.md` for the full
            // analysis.
            let s_first = snode_first_col[root_s];
            let s_ncol = snode_ncols[root_s];
            let p_first = snode_first_col[root_p];
            if s_first + s_ncol != p_first {
                continue;
            }

            let child_ncol = snode_ncols[root_s];
            let parent_ncol = snode_ncols[root_p];

            // SSIDS merge rule:
            // 1. Trivial chain: parent has exactly 1 col AND parent's column
            //    count == child's last column count - 1 (same row structure
            //    minus one eliminated column)
            let trivial_chain = parent_ncol == 1 && {
                let child_last = s_first + s_ncol - 1;
                col_counts[p_first] + 1 == col_counts[child_last]
            };

            // 2. Size-based: both have < nemin columns
            let size_based = child_ncol < params.nemin && parent_ncol < params.nemin;

            // Phase B4 (issue #55): defensive root-supernode width cap.
            // On IPM-KKT matrices with a wide top-level Schur complement
            // (e.g. nql180, pinene_3200), unrestricted amalgamation can
            // grow the root supernode to many thousands of columns. The
            // root frontal is then effectively dense AND receives all
            // delayed-pivot catchment from the subtree - the worst
            // possible combination for memory.
            //
            // The cap applies only above `ROOT_CAP_MIN_N` (small
            // problems can amalgamate freely; the wide-front pathology
            // only manifests at scale and the existing `nemin` logic
            // is the right constraint for small trees). Above the
            // threshold the merged root is capped at
            // `min(0.05 * n, 2048)` columns - loose enough not to
            // disturb non-pathological problems, tight enough that
            // nql180-class KKTs cannot grow back to a dense root.
            const ROOT_CAP_MIN_N: usize = 1024;
            let parent_is_root = snode_parent[root_p].is_none();
            let merged_ncol = child_ncol + parent_ncol;
            let root_cap = if n >= ROOT_CAP_MIN_N {
                (n / 20).min(2048)
            } else {
                usize::MAX
            };
            let root_cap_exceeded = parent_is_root && merged_ncol > root_cap;

            // EXPERIMENTAL relaxed/fill-tolerant merge: merge an adjacent child
            // even when it is not size-based, as long as the merged supernode
            // stays under `relax_width` and the extra explicit-zero fill (the
            // gap between the child's post-elimination structure and the
            // parent's) is within `relax_rows`. This is the PARDISO/MUMPS
            // relaxed-amalgamation lever: trade a little fill for much wider
            // dense fronts (higher-rank Schur GEMMs).
            let relaxed = relax_width.is_some_and(|w| {
                let child_last = s_first + s_ncol - 1;
                let extra = col_counts[child_last].saturating_sub(1 + col_counts[p_first]);
                merged_ncol <= w && extra <= relax_rows
            });

            if (trivial_chain || size_based || relaxed) && !root_cap_exceeded {
                merged_into[root_s] = Some(root_p);
                // Transfer columns to parent and update first column.
                // Adjacency invariant guarantees s_first < p_first,
                // so the merged range is [s_first, p_first+p_ncol).
                snode_ncols[root_p] = merged_ncol;
                snode_first_col[root_p] = s_first;
            }
        }
    }

    // Step 3: Build final supernode list
    // Collect non-merged supernodes
    let mut final_snodes: Vec<Supernode> = Vec::new();
    let mut new_snode_id = vec![0usize; n_snodes]; // old → new supernode index

    for s in 0..n_snodes {
        if merged_into[s].is_some() {
            continue;
        }

        let first_col = snode_first_col[s];
        let ncol = snode_ncols[s];
        // nrow = col_counts[first_col]: number of rows in L for the first
        // column of this supernode, which gives the frontal matrix height
        let nrow = col_counts[first_col].max(ncol);

        // Row indices: the first_col..first_col+ncol are the eliminated columns,
        // plus the remaining rows from col_counts
        // For now, store just the column range - actual row indices are
        // determined during symbolic factorization with the full pattern
        let row_indices = (first_col..first_col + nrow).collect();

        new_snode_id[s] = final_snodes.len();

        final_snodes.push(Supernode {
            first_col,
            ncol,
            nrow,
            row_indices,
            children: Vec::new(),
            // B1: pre-populate with the unbounded sentinel; B2 patches
            // this with a real estimate after the supernode list is
            // built and the tree relationships are wired.
            delayed_capacity: usize::MAX,
        });
    }

    // Set children relationships
    for s in 0..n_snodes {
        if merged_into[s].is_some() {
            continue;
        }
        if let Some(p) = snode_parent[s] {
            let root_p = find_root(p, &merged_into);
            if root_p != s {
                let new_child = new_snode_id[s];
                let new_parent = new_snode_id[root_p];
                final_snodes[new_parent].children.push(new_child);
            }
        }
    }

    final_snodes
}

/// Issue #55 Phase B2: multiplier on `own_ncol` that bounds the
/// per-supernode incoming-delay budget.
///
/// `delayed_capacity(s) = min(subtree_ncol - own_ncol, K * own_ncol)`.
/// With `K = 4` the frontal matrix at supernode `s` can grow by at
/// most `(1 + K) = 5x` its own width before the budget trips at
/// numeric time. This is the loose-but-defensible starting value
/// for the cascade-victim corpus; A2.5 instrumentation (not yet
/// run on the full corpus) would inform a tighter value. See
/// `dev/research/symbolic-delay-budget-2026-05-27.md`.
pub const DELAY_CAPACITY_MULTIPLIER: usize = 4;

/// Issue #55 Phase B2: minimum capacity floor for small supernodes.
///
/// `DELAY_CAPACITY_MULTIPLIER * own_ncol` is too tight when `own_ncol`
/// is very small (e.g. `nemin=1` stress runs that produce
/// single-column supernodes), because a single-column supernode would
/// otherwise have `capacity = 4`, which routinely under-shoots even
/// modest non-pathological delay catchment. The floor ensures every
/// supernode can absorb at least 16 incoming delays before the
/// budget trips - generous for small supernodes, irrelevant for
/// wide supernodes where `K * own_ncol >> 16`. The worst-case bound
/// (`subtree_ncol - own_ncol`) still caps capacity for leaves and
/// near-leaves, so the floor cannot create artificially-large
/// budgets on shallow trees.
pub const DELAY_CAPACITY_MIN_FLOOR: usize = 16;

/// Issue #55 Phase B2: assign per-supernode incoming-delay budget
/// (`Supernode::delayed_capacity`) to a freshly-built supernode
/// list.
///
/// For each supernode `s`, sets
/// `delayed_capacity(s) = min(subtree_ncol(s) - own_ncol(s),
///                            DELAY_CAPACITY_MULTIPLIER * own_ncol(s))`
/// where `subtree_ncol(s)` is the total `ncol` summed across `s`
/// and all its descendants.
///
/// Rationale:
/// - The first term is the loose worst-case upper bound: at most
///   one delay per eliminable column anywhere below `s` can reach
///   `s` (since delays are 1-for-1 fully-summed columns that
///   children failed to eliminate).
/// - The second term tightens that to a constant multiple of the
///   supernode's own width, which is the quantity that drives
///   frontal-matrix memory at numeric time (the frontal is sized
///   as `(own_ncol + n_delayed_in) × nrow`).
///
/// The `min` of the two is the cap actually enforced. For leaves
/// (no children, `subtree_ncol == own_ncol`) the first term is 0,
/// so leaves always get `delayed_capacity == 0` - which is
/// trivially correct (leaves have no children that could send
/// delays). For interior nodes the K-bound is usually tighter
/// than the worst-case bound; for tall thin chains the
/// worst-case bound can be tighter.
///
/// Pre-condition: `snodes` is in postorder (children before parents),
/// per [`find_supernodes`]'s contract.
///
/// Cost: two linear passes over `snodes`. O(n_snodes + sum of
/// children-list lengths) = O(n_snodes) since children lists are a
/// disjoint partition of non-root supernodes.
pub fn assign_delayed_capacities(snodes: &mut [Supernode]) {
    let n = snodes.len();
    // Bottom-up subtree ncol sum. snodes[s].children are all strictly
    // less than s in postorder, so one forward pass suffices.
    let mut subtree_ncol: Vec<usize> = vec![0; n];
    for s in 0..n {
        let mut sum = snodes[s].ncol;
        for &c in &snodes[s].children {
            sum += subtree_ncol[c];
        }
        subtree_ncol[s] = sum;
    }
    for s in 0..n {
        let own = snodes[s].ncol;
        let worst = subtree_ncol[s].saturating_sub(own);
        let tight = DELAY_CAPACITY_MULTIPLIER
            .saturating_mul(own)
            .max(DELAY_CAPACITY_MIN_FLOOR);
        snodes[s].delayed_capacity = worst.min(tight);
    }
}

/// Find the root of the merge chain for supernode s.
fn find_root(s: usize, merged_into: &[Option<usize>]) -> usize {
    let mut node = s;
    while let Some(parent) = merged_into[node] {
        node = parent;
    }
    node
}

/// Output of `find_fundamental_supernodes`.
pub(crate) struct FundamentalSupernodes {
    /// First column of each fundamental supernode.
    pub(crate) snode_starts: Vec<usize>,
    /// Number of columns in each fundamental supernode.
    pub(crate) snode_ncols: Vec<usize>,
    /// Parent fundamental supernode of each fundamental supernode
    /// (the supernode containing the etree-parent of its last column),
    /// or `None` for roots.
    pub(crate) snode_parent: Vec<Option<usize>>,
}

/// Detect *fundamental* supernodes only (no amalgamation, no merging).
///
/// A fundamental supernode is a maximal set of consecutive columns
/// j, j+1, ..., j+k where each column has the same row structure
/// minus the eliminated columns. This is the structural Step 1 of
/// `find_supernodes`, factored out so `predict_merges` can reuse it.
///
/// Conditions for `j` to extend the supernode of `j-1`:
///   1. `parent[j-1] == j` in the etree
///   2. `col_counts[j] + 1 == col_counts[j-1]`
///   3. `j` has exactly one child in the etree (= `j-1`)
pub(crate) fn find_fundamental_supernodes(
    etree: &EliminationTree,
    col_counts: &[usize],
) -> FundamentalSupernodes {
    let n = etree.n;
    if n == 0 {
        return FundamentalSupernodes {
            snode_starts: Vec::new(),
            snode_ncols: Vec::new(),
            snode_parent: Vec::new(),
        };
    }

    let mut snode_id = vec![0usize; n];
    let mut snode_starts: Vec<usize> = Vec::new();

    let mut n_children = vec![0usize; n];
    for j in 0..n {
        if let Some(p) = etree.parent[j] {
            n_children[p] += 1;
        }
    }

    snode_starts.push(0);
    snode_id[0] = 0;
    for j in 1..n {
        let same_snode = etree.parent[j - 1] == Some(j)
            && col_counts[j] + 1 == col_counts[j - 1]
            && n_children[j] == 1;
        if same_snode {
            snode_id[j] = snode_id[j - 1];
        } else {
            snode_id[j] = snode_starts.len();
            snode_starts.push(j);
        }
    }

    let n_snodes = snode_starts.len();
    let mut snode_ncols = vec![0usize; n_snodes];
    let mut snode_parent: Vec<Option<usize>> = vec![None; n_snodes];
    for j in 0..n {
        snode_ncols[snode_id[j]] += 1;
    }
    for s in 0..n_snodes {
        let last_col = snode_starts[s] + snode_ncols[s] - 1;
        if let Some(p) = etree.parent[last_col] {
            snode_parent[s] = Some(snode_id[p]);
        }
    }

    FundamentalSupernodes {
        snode_starts,
        snode_ncols,
        snode_parent,
    }
}

/// Predict desired merges (Phase 2.12) for the SSIDS-style column
/// renumbering. Runs the same fundamental-supernode detection and
/// SSIDS size rule as `find_supernodes`, but **does not enforce
/// adjacency** - the caller uses the merge predictions to drive a
/// merge-biased postorder that *makes* the merges adjacent in the
/// re-postordered numbering.
///
/// Returns a vector `desired_merges` of length `n` where, for each
/// column `c`, `desired_merges[c] = Some(parent_first_col)` indicates
/// that the fundamental supernode containing `c` should be merged
/// into its parent fundamental supernode (whose first column is
/// `parent_first_col`). Columns belonging to non-merging
/// supernodes - and the parent supernode of every merge - get `None`.
///
/// The encoding is per-column (not per-supernode) so the caller can
/// drive a per-node bias on the etree directly.
pub(crate) fn predict_merges(
    etree: &EliminationTree,
    col_counts: &[usize],
    params: &SupernodeParams,
) -> Vec<bool> {
    let n = etree.n;
    let mut bias = vec![false; n];
    if n == 0 {
        return bias;
    }
    let fund = find_fundamental_supernodes(etree, col_counts);
    let n_snodes = fund.snode_starts.len();

    // For each fundamental supernode, decide whether it would merge
    // into its parent under the SSIDS size rule.
    for s in 0..n_snodes {
        let p = match fund.snode_parent[s] {
            Some(p) => p,
            None => continue,
        };
        let child_ncol = fund.snode_ncols[s];
        let parent_ncol = fund.snode_ncols[p];

        // SSIDS rule (mirrors find_supernodes Step 2):
        // 1. Trivial chain: parent has 1 col, parent.col_count + 1 == child.last_col_count
        let s_first = fund.snode_starts[s];
        let p_first = fund.snode_starts[p];
        let child_last = s_first + child_ncol - 1;
        let trivial_chain = parent_ncol == 1 && col_counts[p_first] + 1 == col_counts[child_last];
        // 2. Size-based: both < nemin
        let size_based = child_ncol < params.nemin && parent_ncol < params.nemin;

        if trivial_chain || size_based {
            // Mark every column of this child supernode as "biased
            // late" - its subtree should be emitted adjacent to its
            // parent in the merge-biased postorder.
            for b in bias.iter_mut().skip(s_first).take(child_ncol) {
                *b = true;
            }
        }
    }

    bias
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;
    use crate::symbolic::column_counts::column_counts;

    #[test]
    fn test_supernodes_tridiagonal() {
        // Tridiagonal 4x4: col_counts = [2, 2, 2, 1]
        // Columns 2,3 form a fundamental supernode (parent[2]=3, counts[3]+1=counts[2])
        // Columns 0 and 1 are singletons
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        // With nemin=1, we get 3 supernodes: {0}, {1}, {2,3}
        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);
        assert_eq!(snodes.len(), 3);

        let total_cols: usize = snodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 4);
    }

    #[test]
    fn test_supernodes_tridiagonal_amalgamated() {
        // With large nemin, all singletons should be amalgamated into one
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 32,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // All 4 columns should be amalgamated into 1 supernode
        let total_cols: usize = snodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 4);
        assert_eq!(snodes.len(), 1);
    }

    #[test]
    fn test_supernodes_dense() {
        // Dense 3x3: col_counts = [3, 2, 1]
        // Fundamental: column 1 chains into column 0 (parent[0]=1, counts[1]=counts[0]-1)
        // Column 2 chains into column 1 (parent[1]=2, counts[2]=counts[1]-1)
        // So all 3 columns form one fundamental supernode
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // Should be 1 supernode with 3 columns (fundamental)
        assert_eq!(snodes.len(), 1);
        assert_eq!(snodes[0].ncol(), 3);
        assert_eq!(snodes[0].nrow, 3);
        assert_eq!(snodes[0].contrib_size(), 0); // no contribution block
    }

    #[test]
    fn test_supernodes_block_diagonal() {
        // Two 2x2 dense blocks: two independent supernodes
        let m = CscMatrix::from_triplets(4, &[0, 1, 1, 2, 3, 3], &[0, 0, 1, 2, 2, 3], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // Two fundamental supernodes of size 2
        assert_eq!(snodes.len(), 2);
        assert_eq!(snodes[0].ncol(), 2);
        assert_eq!(snodes[1].ncol(), 2);
    }

    #[test]
    fn test_supernodes_diagonal_no_amalg() {
        // Diagonal 4x4 with nemin=1: 4 singletons, no merging possible
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        // Each column is independent (no parents), so 4 supernodes
        assert_eq!(snodes.len(), 4);
    }

    #[test]
    fn test_supernodes_total_columns() {
        // For any matrix, the total columns across all supernodes should equal n
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        for nemin in [1, 5, 32] {
            let params = SupernodeParams {
                nemin,
                ..Default::default()
            };
            let snodes = find_supernodes(&etree, &counts, &params);
            let total: usize = snodes.iter().map(|s| s.ncol()).sum();
            assert_eq!(total, 5, "nemin={}: total columns {} != 5", nemin, total);
        }
    }

    #[test]
    fn test_supernode_children_valid() {
        // Verify all child indices are valid
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        for (i, s) in snodes.iter().enumerate() {
            for &child in &s.children {
                assert!(child < snodes.len(), "invalid child index");
                assert!(
                    child < i,
                    "child {} should come before parent {} in postorder",
                    child,
                    i
                );
            }
        }
    }
}
