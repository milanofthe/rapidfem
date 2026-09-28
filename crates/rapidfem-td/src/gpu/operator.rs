// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! GPU representation of the DG Maxwell operator.
//!
//! [`GpuOperator`] uploads the CPU operator's data (reference matrices,
//! geometric factors, materials, face topology) to device buffers once,
//! then evaluates `dy/dt = A.y` with the `apply` kernel. The field state
//! buffers are owned and reused, so a later step loop keeps the state
//! resident on the device.
//!
//! Phase P1: non-dispersive operators (the `[E,H]` block). The dispersive
//! polarisation block is a later phase.

use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::Buffer;
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_double, cl_float, cl_int};

use super::GpuContext;
use crate::constants::{
    Field, KCL_A, KCL_B, KCL_BHAT, KCL_STAGES, KRYLOV_CHUNK,
    LSERK4_A, LSERK4_B, LSERK4_STAGES,
};
use crate::propagator::expm;
use crate::rhs::MaxwellOperator;

/// Apply-kernel source, with `NP` / `NFP` / `COLS` prepended at build time.
const APPLY_SRC: &str = include_str!("apply.cl");

/// LSERK4 stage-update kernel source.
const LSERK_SRC: &str = include_str!("lserk.cl");

/// Krylov exponential-propagator kernel source.
const EXPMV_SRC: &str = include_str!("expmv.cl");

/// KCL RK4(3)5[2R+]C adaptive stage / error-reduction kernel source.
const KCL_SRC: &str = include_str!("kcl.cl");

/// Source-injection mode for the GPU KCL adaptive transient. Free runs
/// homogeneously; the two driven variants mirror the LSERK4 GPU paths
/// (single-DOF point hold and full-vector hold), zeroth-order over each
/// output frame.
enum DrivenMode<'a> {
    Free,
    Point { source_dof: usize, g_values: &'a [f32] },
    Vector { g_values: &'a [f32] },
}

/// Target work-group size for the `apply` kernel. A work-group processes
/// one block of `EPG = APPLY_TARGET_WG / NP` elements, its work-items the
/// `NP` DG nodes of each, so the group holds `EPG * NP` work-items, close
/// to this target.
const APPLY_TARGET_WG: usize = 128;

/// Work-group size for the flat per-DOF LSERK4 stage loop.
const DOF_WORK_GROUP: usize = 256;

/// Work-group size for the norm reduction (a power of two for the
/// local-memory tree reduction).
const NORM_WORK_GROUP: usize = 256;

/// The DG Maxwell operator resident on the GPU.
pub struct GpuOperator {
    n_elem: usize,
    /// State-vector length, `6*Np*n_elem` (non-dispersive).
    n_dof: usize,
    /// Elements per `apply` work-group.
    apply_epg: usize,
    /// `apply` work-group size, `apply_epg * Np`.
    apply_wg: usize,
    flux_alpha: f32,
    _program: Program,
    kernel: Kernel,
    // Operator data, uploaded once.
    diff_r: Buffer<cl_float>,
    diff_s: Buffer<cl_float>,
    diff_t: Buffer<cl_float>,
    lift: Buffer<cl_float>,
    face_nodes: Buffer<cl_int>,
    jinv: Buffer<cl_float>,
    inv_eps: Buffer<cl_float>,
    inv_mu: Buffer<cl_float>,
    sigma_eps: Buffer<cl_float>,
    sigma_mu: Buffer<cl_float>,
    face_normal: Buffer<cl_float>,
    face_fscale: Buffer<cl_float>,
    face_neighbor: Buffer<cl_int>,
    face_nbr_local: Buffer<cl_int>,
    face_port: Buffer<cl_int>,
    face_perm: Buffer<cl_int>,
    // State buffers, reused across calls.
    y: Buffer<cl_float>,
    dy: Buffer<cl_float>,
    /// LSERK4 residual register, device-resident across a transient.
    p: Buffer<cl_float>,
    _lserk_program: Program,
    lserk_kernel: Kernel,
    source_kernel: Kernel,
    /// Full-vector held source `k[i] += g*src[i]` (modal-port injection).
    source_vec_kernel: Kernel,
    /// Device-resident source spatial pattern `b`, uploaded once per drive.
    src: Buffer<cl_float>,
    // Krylov exponential propagator (P3). The f64 kernels are created
    // lazily on the first `expmv`: a device without `cl_khr_fp64` (Apple
    // Silicon, many integrated GPUs) cannot instantiate them, but the f32
    // explicit (LSERK4) path does not need them, building them eagerly in
    // `new` would wrongly make the whole operator unconstructible there.
    _expmv_program: Program,
    /// Lazily-built f64 Krylov kernels; `None` until the first `expmv`.
    expmv_kernels: Option<ExpmvKernels>,
    /// Lazily-allocated f64 Krylov buffers, sized on the first `expmv`.
    krylov: Option<Krylov>,
    // KCL RK4(3)5[2R+]C adaptive stepper, f32 like the LSERK4 path,
    // stage updates and the embedded-error vector device-resident, with
    // the host running the PI controller on top of a per-substep
    // `weighted_err_sq` reduction. Built eagerly: no fp64 needed.
    _kcl_program: Program,
    kcl_stage0_kernel: Kernel,
    kcl_build_stage_kernel: Kernel,
    kcl_stage_accum_kernel: Kernel,
    weighted_err_sq_kernel: Kernel,
    copy_vec_kernel: Kernel,
    /// Stage state register `Y_i`, built fresh each interior stage.
    kcl_stage_buf: Buffer<cl_float>,
    /// Carries `dt·F_{i-1}` between stages (and into the next stage's
    /// `kcl_build_stage` evaluation point).
    kcl_dtf: Buffer<cl_float>,
    /// Embedded-error accumulator `e = Σ_i (b̂_i - b_i)·dt·F_i`.
    kcl_err: Buffer<cl_float>,
    /// Snapshot of `y` taken before each adaptive substep, the rollback
    /// source on a rejected step. Restoring it costs one `copy_vec`.
    kcl_ybackup: Buffer<cl_float>,
    /// Per-workgroup partial sums of `(err/scale)²`, summed host-side into
    /// the scalar `err_norm` the controller compares against 1.0. Sized
    /// `n_dof.div_ceil(NORM_WORK_GROUP)`; allocated lazily on first
    /// adaptive run.
    kcl_err_partial: Option<Buffer<cl_float>>,
}

/// The f64 Krylov kernels for the exponential propagator, built lazily so
/// the f32 explicit path stays available on fp64-less devices.
struct ExpmvKernels {
    cast_d2f: Kernel,
    cast_f2d: Kernel,
    dot_rows: Kernel,
    axpy_basis: Kernel,
    norm2: Kernel,
    store_h_col: Kernel,
    finish_norm: Kernel,
    scale_recip: Kernel,
    lincomb: Kernel,
    /// `w[i] += g*src[i]`: the source-column axpy of the augmented driven
    /// Arnoldi (ETD step). See [`GpuOperator::etd_step`].
    axpy_src: Kernel,
}

/// f64 Arnoldi buffers for the Krylov exponential propagator. The basis
/// stays device-resident; the matvec drops to f32 through the `apply`
/// kernel, the orthogonalisation stays f64.
struct Krylov {
    /// Largest Krylov dimension the buffers are sized for.
    cap_dim: usize,
    /// Arnoldi basis, flat, `(cap_dim+1)` vectors of length `n`.
    basis: Buffer<cl_double>,
    /// Arnoldi working vector.
    w: Buffer<cl_double>,
    /// CGS2 projection coefficients.
    proj: Buffer<cl_double>,
    /// Result vector.
    out: Buffer<cl_double>,
    /// Per-work-group partial sums for the norm reduction.
    partials: Buffer<cl_double>,
    /// Hessenberg `H`, device-resident `cap_dim*cap_dim`, so the Arnoldi
    /// loop never round-trips it to the host.
    h: Buffer<cl_double>,
    /// One-element scalar holding the current Arnoldi `hnext = ||w||`.
    hnext: Buffer<cl_double>,
    /// f64 source pattern `b` for the augmented driven Arnoldi (ETD step);
    /// length `n`, uploaded once per [`GpuOperator::etd_step`] call.
    src64: Buffer<cl_double>,
    /// Number of work-groups in the norm reduction.
    n_groups: usize,
}

/// `f64` slice to an `f32` vector.
fn f32v(s: &[Field]) -> Vec<f32> {
    s.iter().map(|&x| x as f32).collect()
}

impl GpuOperator {
    /// Upload a non-dispersive CPU operator to the device.
    pub fn new(
        gpu: &GpuContext,
        op: &MaxwellOperator,
    ) -> Result<Self, String> {
        assert_eq!(
            op.n_dispersive(),
            0,
            "GpuOperator: dispersive materials are a later phase",
        );
        let np = op.re.n_nodes;
        let nfp = op.re.n_face_nodes;
        let cols = 4 * nfp;
        let n_elem = op.n_elem;
        let n_dof = 6 * np * n_elem;

        // Reference element.
        let diff_r = gpu.upload(&f32v(&op.re.diff_r))?;
        let diff_s = gpu.upload(&f32v(&op.re.diff_s))?;
        let diff_t = gpu.upload(&f32v(&op.re.diff_t))?;
        let lift = gpu.upload(&f32v(&op.re.lift))?;
        let mut fn_flat = Vec::with_capacity(4 * nfp);
        for f in 0..4 {
            for m in 0..nfp {
                fn_flat.push(op.re.face_nodes[f][m] as i32);
            }
        }
        let face_nodes = gpu.upload_i32(&fn_flat)?;

        // Per-element geometric factors and materials.
        let mut jinv = Vec::with_capacity(n_elem * 9);
        for g in &op.geom {
            for i in 0..3 {
                for k in 0..3 {
                    jinv.push(g.jacobian_inv[i][k] as f32);
                }
            }
        }
        let jinv = gpu.upload(&jinv)?;
        let flat3 = |v: &[[Field; 3]]| -> Vec<f32> {
            v.iter()
                .flat_map(|a| [a[0] as f32, a[1] as f32, a[2] as f32])
                .collect()
        };
        let inv_eps = gpu.upload(&flat3(&op.inv_eps))?;
        let inv_mu = gpu.upload(&flat3(&op.inv_mu))?;
        let sigma_eps = gpu.upload(&flat3(&op.sigma_eps))?;
        let sigma_mu = gpu.upload(&flat3(&op.sigma_mu))?;

        // Face topology, flattened over `faces[e*4 + f]`.
        let nf = 4 * n_elem;
        let mut normal = Vec::with_capacity(nf * 3);
        let mut fscale = Vec::with_capacity(nf);
        let mut neighbor = Vec::with_capacity(nf);
        let mut nbr_local = Vec::with_capacity(nf);
        let mut port = Vec::with_capacity(nf);
        let mut perm = Vec::with_capacity(nf * nfp);
        for fi in &op.faces {
            normal.extend([
                fi.normal[0] as f32,
                fi.normal[1] as f32,
                fi.normal[2] as f32,
            ]);
            fscale.push(fi.fscale as f32);
            neighbor.push(if fi.neighbor == usize::MAX {
                -1
            } else {
                fi.neighbor as i32
            });
            nbr_local.push(fi.neighbor_local_face as i32);
            port.push(if fi.port == usize::MAX {
                -1
            } else {
                fi.port as i32
            });
            for m in 0..nfp {
                // `perm` is empty on a boundary face; the kernel only reads
                // it on the neighbour branch, so the pad value is inert.
                perm.push(fi.perm.get(m).map_or(0, |&p| p as i32));
            }
        }
        let face_normal = gpu.upload(&normal)?;
        let face_fscale = gpu.upload(&fscale)?;
        let face_neighbor = gpu.upload_i32(&neighbor)?;
        let face_nbr_local = gpu.upload_i32(&nbr_local)?;
        let face_port = gpu.upload_i32(&port)?;
        let face_perm = gpu.upload_i32(&perm)?;

        // Build the kernel with the element dimensions baked in. EPG (the
        // elements per work-group) is chosen so the group sits near the
        // target work-group size.
        let epg = (APPLY_TARGET_WG / np).max(1);
        let apply_wg = epg * np;
        let src = format!(
            "#define NP {np}\n#define NFP {nfp}\n#define COLS {cols}\n\
             #define EPG {epg}\n{APPLY_SRC}"
        );
        let program = gpu.build_program(&src)?;
        let kernel = Kernel::create(&program, "apply")
            .map_err(|e| format!("kernel create failed: {e}"))?;

        let lserk_program = gpu.build_program(LSERK_SRC)?;
        let lserk_kernel = Kernel::create(&lserk_program, "lserk_stage")
            .map_err(|e| format!("lserk kernel create failed: {e}"))?;
        let source_kernel = Kernel::create(&lserk_program, "add_source")
            .map_err(|e| format!("source kernel create failed: {e}"))?;
        let source_vec_kernel =
            Kernel::create(&lserk_program, "add_source_vec")
                .map_err(|e| format!("source_vec kernel create failed: {e}"))?;

        // Build the program (this succeeds even without fp64), but defer
        // creating the f64 kernels themselves to the first `expmv`, see
        // the `expmv_kernels` field note.
        let expmv_program = gpu.build_program(EXPMV_SRC)?;

        // KCL adaptive: f32 program + kernels build eagerly, since they
        // do not depend on fp64. The error-reduction partial buffer is
        // allocated lazily on the first call.
        let kcl_program = gpu.build_program(KCL_SRC)?;
        let kcl_stage0_kernel = Kernel::create(&kcl_program, "kcl_stage0")
            .map_err(|e| format!("kcl_stage0 kernel create failed: {e}"))?;
        let kcl_build_stage_kernel =
            Kernel::create(&kcl_program, "kcl_build_stage")
                .map_err(|e| format!("kcl_build_stage create failed: {e}"))?;
        let kcl_stage_accum_kernel =
            Kernel::create(&kcl_program, "kcl_stage_accum")
                .map_err(|e| format!("kcl_stage_accum create failed: {e}"))?;
        let weighted_err_sq_kernel =
            Kernel::create(&kcl_program, "weighted_err_sq").map_err(|e| {
                format!("weighted_err_sq kernel create failed: {e}")
            })?;
        let copy_vec_kernel = Kernel::create(&kcl_program, "copy_vec")
            .map_err(|e| format!("copy_vec kernel create failed: {e}"))?;

        let y = gpu.alloc(n_dof)?;
        let dy = gpu.alloc(n_dof)?;
        let p = gpu.alloc(n_dof)?;
        let src = gpu.alloc(n_dof)?;
        let kcl_stage_buf = gpu.alloc(n_dof)?;
        let kcl_dtf = gpu.alloc(n_dof)?;
        let kcl_err = gpu.alloc(n_dof)?;
        let kcl_ybackup = gpu.alloc(n_dof)?;

        Ok(GpuOperator {
            n_elem,
            n_dof,
            apply_epg: epg,
            apply_wg,
            flux_alpha: op.flux_alpha as f32,
            _program: program,
            kernel,
            diff_r,
            diff_s,
            diff_t,
            lift,
            face_nodes,
            jinv,
            inv_eps,
            inv_mu,
            sigma_eps,
            sigma_mu,
            face_normal,
            face_fscale,
            face_neighbor,
            face_nbr_local,
            face_port,
            face_perm,
            y,
            dy,
            p,
            _lserk_program: lserk_program,
            lserk_kernel,
            source_kernel,
            source_vec_kernel,
            src,
            _expmv_program: expmv_program,
            expmv_kernels: None,
            krylov: None,
            _kcl_program: kcl_program,
            kcl_stage0_kernel,
            kcl_build_stage_kernel,
            kcl_stage_accum_kernel,
            weighted_err_sq_kernel,
            copy_vec_kernel,
            kcl_stage_buf,
            kcl_dtf,
            kcl_err,
            kcl_ybackup,
            kcl_err_partial: None,
        })
    }

    /// Build the f64 Krylov kernels on first use. Fails with a clear error
    /// on a device without `cl_khr_fp64` (e.g. Apple Silicon), where the
    /// exponential propagator is unavailable but the explicit path is not.
    fn ensure_expmv_kernels(&mut self) -> Result<(), String> {
        if self.expmv_kernels.is_some() {
            return Ok(());
        }
        let kern = |name: &str| {
            Kernel::create(&self._expmv_program, name).map_err(|e| {
                format!(
                    "{name} kernel create failed: {e}, the GPU exponential \
                     propagator needs f64 (cl_khr_fp64), unavailable on this \
                     device; use the explicit stepper or the CPU path"
                )
            })
        };
        self.expmv_kernels = Some(ExpmvKernels {
            cast_d2f: kern("cast_d2f")?,
            cast_f2d: kern("cast_f2d")?,
            dot_rows: kern("dot_rows")?,
            axpy_basis: kern("axpy_basis")?,
            norm2: kern("partial_norm2")?,
            store_h_col: kern("store_h_col")?,
            finish_norm: kern("finish_norm")?,
            scale_recip: kern("scale_recip")?,
            lincomb: kern("lincomb")?,
            axpy_src: kern("axpy_src")?,
        });
        Ok(())
    }

    /// State-vector length the operator expects.
    pub fn n_dof(&self) -> usize {
        self.n_dof
    }

    /// Enqueue the `apply` kernel: `dy = A.y` on the resident state. No
    /// host transfer; the in-order queue serialises it with later work.
    fn enqueue_apply(&self, gpu: &GpuContext) -> Result<(), String> {
        self.enqueue_apply_from(gpu, &self.y)
    }

    /// Enqueue the `apply` kernel reading from an arbitrary device state
    /// buffer `input` (still writing into `self.dy`). The KCL adaptive
    /// stencil evaluates the matvec at the stage-state buffer Y_i, not at
    /// the running S2 accumulator `self.y`, so the input buffer cannot be
    /// hard-coded as it is in the LSERK4 path.
    fn enqueue_apply_from(
        &self,
        gpu: &GpuContext,
        input: &Buffer<cl_float>,
    ) -> Result<(), String> {
        let n_groups = self.n_elem.div_ceil(self.apply_epg);
        let global = n_groups * self.apply_wg;
        let n_elem = self.n_elem as cl_int;
        unsafe {
            ExecuteKernel::new(&self.kernel)
                .set_arg(input)
                .set_arg(&self.dy)
                .set_arg(&self.diff_r)
                .set_arg(&self.diff_s)
                .set_arg(&self.diff_t)
                .set_arg(&self.lift)
                .set_arg(&self.face_nodes)
                .set_arg(&self.jinv)
                .set_arg(&self.inv_eps)
                .set_arg(&self.inv_mu)
                .set_arg(&self.sigma_eps)
                .set_arg(&self.sigma_mu)
                .set_arg(&self.face_normal)
                .set_arg(&self.face_fscale)
                .set_arg(&self.face_neighbor)
                .set_arg(&self.face_nbr_local)
                .set_arg(&self.face_port)
                .set_arg(&self.face_perm)
                .set_arg(&self.flux_alpha)
                .set_arg(&n_elem)
                .set_global_work_size(global)
                .set_local_work_size(self.apply_wg)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("apply kernel launch failed: {e}"))?;
        Ok(())
    }

    /// Enqueue one LSERK4 stage: `p = a*p + dt*k; y += b*p`, with `k` the
    /// `dy` written by the preceding [`enqueue_apply`](Self::enqueue_apply).
    fn enqueue_lserk(
        &self,
        gpu: &GpuContext,
        a: f32,
        b: f32,
        dt: f32,
    ) -> Result<(), String> {
        let global = self.n_dof.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let n = self.n_dof as cl_int;
        unsafe {
            ExecuteKernel::new(&self.lserk_kernel)
                .set_arg(&self.p)
                .set_arg(&self.dy)
                .set_arg(&self.y)
                .set_arg(&a)
                .set_arg(&b)
                .set_arg(&dt)
                .set_arg(&n)
                .set_global_work_size(global)
                .set_local_work_size(DOF_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("lserk kernel launch failed: {e}"))?;
        Ok(())
    }

    /// Evaluate `dy/dt = A.y`: upload `y_host`, run the apply kernel,
    /// download the result. The single-shot form, for validation.
    pub fn apply(
        &mut self,
        gpu: &GpuContext,
        y_host: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y_host.len(), self.n_dof, "state length mismatch");
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y_host, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        self.enqueue_apply(gpu)?;
        // The blocking download serialises behind the kernel.
        gpu.download(&self.dy, self.n_dof)
    }

    /// `f64` host wrapper around [`Self::apply`] for the macromodel
    /// build: cast `f64 -> f32`, run the device matvec, cast back
    /// `f32 -> f64`. The mixed-precision drift is bounded by
    /// [`crate::constants::GPU_REL_TOL`] per matvec; the block-Krylov
    /// build calls this once per basis vector and the projection
    /// onto the orthonormal `V` averages the rounding noise to that
    /// scale across the macromodel.
    ///
    /// Used by the `apply_fn` closure that
    /// [`crate::macromodel::MacroModel::build_with_apply_fn`] takes,
    /// so the CPU and GPU build paths share the same block-CGS2
    /// orthogonalisation and Hessenberg loop. The GPU does the
    /// `n_dof`-sized work; the CPU does the small dot products.
    pub fn apply_f64(
        &mut self,
        gpu: &GpuContext,
        y_host: &[f64],
    ) -> Result<Vec<f64>, String> {
        let y_f32: Vec<f32> = y_host.iter().map(|&v| v as f32).collect();
        let dy_f32 = self.apply(gpu, &y_f32)?;
        Ok(dy_f32.into_iter().map(|v| v as f64).collect())
    }

    /// Enqueue the soft-source add: `dy[source_dof] += val`.
    fn enqueue_add_source(
        &self,
        gpu: &GpuContext,
        dof: cl_int,
        val: f32,
    ) -> Result<(), String> {
        unsafe {
            ExecuteKernel::new(&self.source_kernel)
                .set_arg(&self.dy)
                .set_arg(&dof)
                .set_arg(&val)
                .set_global_work_size(1)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("source kernel launch failed: {e}"))?;
        Ok(())
    }

    /// Enqueue the full-vector held source add: `dy[i] += g * src[i]` over
    /// every DOF. The device-resident `self.src` buffer holds the spatial
    /// pattern `b`, uploaded once per drive; `g` is the per-step waveform.
    fn enqueue_add_source_vec(
        &self,
        gpu: &GpuContext,
        g: f32,
    ) -> Result<(), String> {
        let n = self.n_dof as cl_int;
        unsafe {
            ExecuteKernel::new(&self.source_vec_kernel)
                .set_arg(&self.dy)
                .set_arg(&self.src)
                .set_arg(&g)
                .set_arg(&n)
                .set_global_work_size(self.n_dof)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("source_vec kernel launch failed: {e}"))?;
        Ok(())
    }

    // ── KCL RK4(3)5[2R+]C adaptive stepper ────────────────────────────────

    /// Snapshot `self.y` into `self.kcl_ybackup`, the rollback source for
    /// the next attempted substep. One copy_vec dispatch.
    fn enqueue_kcl_backup(&self, gpu: &GpuContext) -> Result<(), String> {
        self.enqueue_copy(gpu, &self.kcl_ybackup, &self.y)
    }

    /// Restore `self.y` from `self.kcl_ybackup`, used when the controller
    /// rejects the most recent substep. One copy_vec dispatch.
    fn enqueue_kcl_restore(&self, gpu: &GpuContext) -> Result<(), String> {
        self.enqueue_copy(gpu, &self.y, &self.kcl_ybackup)
    }

    /// Generic `dst[i] = src[i]` over `n_dof`, used both for backup and
    /// restore.
    fn enqueue_copy(
        &self,
        gpu: &GpuContext,
        dst: &Buffer<cl_float>,
        src: &Buffer<cl_float>,
    ) -> Result<(), String> {
        let global = self.n_dof.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let n = self.n_dof as cl_int;
        unsafe {
            ExecuteKernel::new(&self.copy_vec_kernel)
                .set_arg(dst)
                .set_arg(src)
                .set_arg(&n)
                .set_global_work_size(global)
                .set_local_work_size(DOF_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("copy_vec launch failed: {e}"))?;
        Ok(())
    }

    /// Stage-0 update: `dtF = dt·k`, `y += b0·dtF`, `e = e0·dtF`, `k` is
    /// the matvec result `A·y_n` from the preceding [`Self::enqueue_apply`].
    fn enqueue_kcl_stage0(
        &self,
        gpu: &GpuContext,
        dt: f32,
        b0: f32,
        e0: f32,
    ) -> Result<(), String> {
        let global = self.n_dof.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let n = self.n_dof as cl_int;
        unsafe {
            ExecuteKernel::new(&self.kcl_stage0_kernel)
                .set_arg(&self.y)
                .set_arg(&self.kcl_dtf)
                .set_arg(&self.kcl_err)
                .set_arg(&self.dy)
                .set_arg(&dt)
                .set_arg(&b0)
                .set_arg(&e0)
                .set_arg(&n)
                .set_global_work_size(global)
                .set_local_work_size(DOF_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("kcl_stage0 launch failed: {e}"))?;
        Ok(())
    }

    /// Build the stage state `Y_i = S2 + (A_sub - b_prev)·dt·F_{i-1}`
    /// into `self.kcl_stage_buf`, the matvec input for stage `i`.
    fn enqueue_kcl_build_stage(
        &self,
        gpu: &GpuContext,
        amb: f32,
    ) -> Result<(), String> {
        let global = self.n_dof.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let n = self.n_dof as cl_int;
        unsafe {
            ExecuteKernel::new(&self.kcl_build_stage_kernel)
                .set_arg(&self.y)
                .set_arg(&self.kcl_dtf)
                .set_arg(&self.kcl_stage_buf)
                .set_arg(&amb)
                .set_arg(&n)
                .set_global_work_size(global)
                .set_local_work_size(DOF_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("kcl_build_stage launch failed: {e}"))?;
        Ok(())
    }

    /// Stage-`i`-accumulation (i >= 1): `dtF = dt·k`, `y += b·dtF`, `e +=
    /// eweight·dtF`. `k` is the matvec result `A·Y_i` written by the
    /// preceding [`Self::enqueue_apply_from`].
    fn enqueue_kcl_stage_accum(
        &self,
        gpu: &GpuContext,
        dt: f32,
        b: f32,
        eweight: f32,
    ) -> Result<(), String> {
        let global = self.n_dof.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let n = self.n_dof as cl_int;
        unsafe {
            ExecuteKernel::new(&self.kcl_stage_accum_kernel)
                .set_arg(&self.y)
                .set_arg(&self.kcl_dtf)
                .set_arg(&self.kcl_err)
                .set_arg(&self.dy)
                .set_arg(&dt)
                .set_arg(&b)
                .set_arg(&eweight)
                .set_arg(&n)
                .set_global_work_size(global)
                .set_local_work_size(DOF_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("kcl_stage_accum launch failed: {e}"))?;
        Ok(())
    }

    /// Allocate the per-workgroup partial-sum buffer for the error
    /// reduction, sized to the device matvec dimensions. Reused across
    /// substeps; lazy because the LSERK4 path doesn't need it.
    fn ensure_kcl_err_partial(
        &mut self,
        gpu: &GpuContext,
    ) -> Result<(), String> {
        if self.kcl_err_partial.is_some() {
            return Ok(());
        }
        let n_groups = self.n_dof.div_ceil(NORM_WORK_GROUP);
        self.kcl_err_partial = Some(gpu.alloc(n_groups)?);
        Ok(())
    }

    /// Run a complete five-stage KCL step on the device-resident state.
    /// Caller must have backed up `y` first (so a controller reject can
    /// roll back). After this returns, `y` holds the candidate `y_{n+1}`
    /// and `kcl_err` holds the per-DOF embedded-error vector.
    fn enqueue_kcl_step(
        &self,
        gpu: &GpuContext,
        h: f32,
    ) -> Result<(), String> {
        // Stage 0: matvec at y_n, the kcl_stage0 kernel initialises
        // dtF/err and advances y to S2 after stage 0.
        self.enqueue_apply(gpu)?;
        self.enqueue_kcl_stage0(
            gpu, h, KCL_B[0] as f32, (KCL_BHAT[0] - KCL_B[0]) as f32,
        )?;
        // Stages 1..s, build stage state, matvec, accumulate.
        for stage in 1..KCL_STAGES {
            let amb = (KCL_A[stage - 1] - KCL_B[stage - 1]) as f32;
            self.enqueue_kcl_build_stage(gpu, amb)?;
            self.enqueue_apply_from(gpu, &self.kcl_stage_buf)?;
            let b = KCL_B[stage] as f32;
            let eweight = (KCL_BHAT[stage] - KCL_B[stage]) as f32;
            self.enqueue_kcl_stage_accum(gpu, h, b, eweight)?;
        }
        Ok(())
    }

    /// Same as [`Self::enqueue_kcl_step`] but for `dy/dt = A·y + b·g(t)`
    /// with a full source-vector hold (modal-port injection). `g_val`
    /// is the waveform value held across the step.
    fn enqueue_kcl_step_driven_vec(
        &self,
        gpu: &GpuContext,
        h: f32,
        g_val: f32,
    ) -> Result<(), String> {
        self.enqueue_apply(gpu)?;
        self.enqueue_add_source_vec(gpu, g_val)?;
        self.enqueue_kcl_stage0(
            gpu, h, KCL_B[0] as f32, (KCL_BHAT[0] - KCL_B[0]) as f32,
        )?;
        for stage in 1..KCL_STAGES {
            let amb = (KCL_A[stage - 1] - KCL_B[stage - 1]) as f32;
            self.enqueue_kcl_build_stage(gpu, amb)?;
            self.enqueue_apply_from(gpu, &self.kcl_stage_buf)?;
            self.enqueue_add_source_vec(gpu, g_val)?;
            let b = KCL_B[stage] as f32;
            let eweight = (KCL_BHAT[stage] - KCL_B[stage]) as f32;
            self.enqueue_kcl_stage_accum(gpu, h, b, eweight)?;
        }
        Ok(())
    }

    /// Same as [`Self::enqueue_kcl_step`] but for `dy/dt = A·y + e_dof·g(t)`
    /// with a single-DOF source hold. `g_val` is the held source value.
    fn enqueue_kcl_step_driven_point(
        &self,
        gpu: &GpuContext,
        h: f32,
        source_dof: cl_int,
        g_val: f32,
    ) -> Result<(), String> {
        self.enqueue_apply(gpu)?;
        self.enqueue_add_source(gpu, source_dof, g_val)?;
        self.enqueue_kcl_stage0(
            gpu, h, KCL_B[0] as f32, (KCL_BHAT[0] - KCL_B[0]) as f32,
        )?;
        for stage in 1..KCL_STAGES {
            let amb = (KCL_A[stage - 1] - KCL_B[stage - 1]) as f32;
            self.enqueue_kcl_build_stage(gpu, amb)?;
            self.enqueue_apply_from(gpu, &self.kcl_stage_buf)?;
            self.enqueue_add_source(gpu, source_dof, g_val)?;
            let b = KCL_B[stage] as f32;
            let eweight = (KCL_BHAT[stage] - KCL_B[stage]) as f32;
            self.enqueue_kcl_stage_accum(gpu, h, b, eweight)?;
        }
        Ok(())
    }

    /// Read `err_norm = sqrt(mean((err / (atol + rtol·max|y|))²))` off the
    /// device. Launches the per-DOF reduction, downloads the per-workgroup
    /// partial sums, finishes the sum host-side. Compares `y_backup` (the
    /// pre-step state) against `y` (the candidate post-step).
    fn read_kcl_err_norm(
        &self,
        gpu: &GpuContext,
        atol: f32,
        rtol: f32,
    ) -> Result<f32, String> {
        let partials = self
            .kcl_err_partial
            .as_ref()
            .expect("kcl_err_partial allocated before read");
        let n = self.n_dof;
        let n_groups = n.div_ceil(NORM_WORK_GROUP);
        let global = n_groups * NORM_WORK_GROUP;
        let n_i = n as cl_int;
        unsafe {
            ExecuteKernel::new(&self.weighted_err_sq_kernel)
                .set_arg(&self.kcl_ybackup)
                .set_arg(&self.y)
                .set_arg(&self.kcl_err)
                .set_arg(partials)
                .set_arg_local_buffer(NORM_WORK_GROUP * 4)
                .set_arg(&atol)
                .set_arg(&rtol)
                .set_arg(&n_i)
                .set_global_work_size(global)
                .set_local_work_size(NORM_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("weighted_err_sq launch failed: {e}"))?;
        let host_partials = gpu.download(partials, n_groups)?;
        let total: f64 = host_partials.iter().map(|&v| v as f64).sum();
        Ok((total / n as f64).sqrt() as f32)
    }

    /// Step-size update factor from the PI controller. On a rejected step
    /// the previous-error blend is dropped (I-only); on acceptance the
    /// PI smoothing avoids step-size oscillations across many frames.
    fn kcl_step_factor(
        err_norm: f32,
        prev_err_norm: f32,
        safety: f32,
        growth_limit: f32,
        shrink_limit: f32,
        pi_alpha: f32,
        pi_beta: f32,
        reject: bool,
    ) -> f32 {
        let f = if reject || prev_err_norm <= 0.0 {
            safety * err_norm.powf(-pi_alpha)
        } else {
            safety
                * err_norm.powf(-pi_alpha)
                * prev_err_norm.powf(pi_beta)
        };
        if reject {
            f.max(shrink_limit)
        } else {
            f.max(shrink_limit).min(growth_limit)
        }
    }

    /// KCL adaptive transient (free system `dy/dt = A·y`), state device-
    /// resident, state-vector trajectory returned flat `[(steps+1) * n_dof]`
    /// with row 0 the initial state. The Rust-side PI controller decides
    /// per-substep accept/reject from a device-computed `err_norm`; only
    /// the scalar `err_norm` (per substep) and the state snapshot (per
    /// accepted frame) cross the bus.
    ///
    /// Returns `(traj, n_accepted, n_rejected, h_min, h_max)`.
    #[allow(clippy::too_many_arguments)]
    pub fn transient_kcl_traj(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        atol: f32,
        rtol: f32,
        safety: f32,
        growth_limit: f32,
        shrink_limit: f32,
        pi_alpha: f32,
        pi_beta: f32,
        min_step_factor: f32,
    ) -> Result<(Vec<f32>, usize, usize, f32, f32), String> {
        self.transient_kcl_traj_impl(
            gpu, y0, dt, steps, atol, rtol, safety, growth_limit,
            shrink_limit, pi_alpha, pi_beta, min_step_factor,
            DrivenMode::Free,
        )
    }

    /// KCL adaptive transient with full-vector source `dy/dt = A·y + b·g(t)`,
    /// the modal-port injection path. `b` is uploaded once into the device-
    /// resident source buffer; `g_values[k]` is the waveform sampled at
    /// `k*dt` and held across the substeps inside output frame `k`.
    #[allow(clippy::too_many_arguments)]
    pub fn transient_kcl_traj_driven_vec(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        b_src: &[f32],
        g_values: &[f32],
        atol: f32,
        rtol: f32,
        safety: f32,
        growth_limit: f32,
        shrink_limit: f32,
        pi_alpha: f32,
        pi_beta: f32,
        min_step_factor: f32,
    ) -> Result<(Vec<f32>, usize, usize, f32, f32), String> {
        assert_eq!(b_src.len(), self.n_dof, "source length mismatch");
        assert_eq!(g_values.len(), steps, "g_values must have `steps` entries");
        unsafe {
            gpu.queue().enqueue_write_buffer(
                &mut self.src, CL_BLOCKING, 0, b_src, &[],
            )
        }
        .map_err(|e| format!("source upload failed: {e}"))?;
        self.transient_kcl_traj_impl(
            gpu, y0, dt, steps, atol, rtol, safety, growth_limit,
            shrink_limit, pi_alpha, pi_beta, min_step_factor,
            DrivenMode::Vector { g_values },
        )
    }

    /// KCL adaptive transient with point source `dy/dt = A·y + e_dof·g(t)`.
    #[allow(clippy::too_many_arguments)]
    pub fn transient_kcl_traj_driven(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        source_dof: usize,
        g_values: &[f32],
        atol: f32,
        rtol: f32,
        safety: f32,
        growth_limit: f32,
        shrink_limit: f32,
        pi_alpha: f32,
        pi_beta: f32,
        min_step_factor: f32,
    ) -> Result<(Vec<f32>, usize, usize, f32, f32), String> {
        assert!(source_dof < self.n_dof, "source_dof out of range");
        assert_eq!(g_values.len(), steps, "g_values must have `steps` entries");
        self.transient_kcl_traj_impl(
            gpu, y0, dt, steps, atol, rtol, safety, growth_limit,
            shrink_limit, pi_alpha, pi_beta, min_step_factor,
            DrivenMode::Point { source_dof, g_values },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn transient_kcl_traj_impl(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        atol: f32,
        rtol: f32,
        safety: f32,
        growth_limit: f32,
        shrink_limit: f32,
        pi_alpha: f32,
        pi_beta: f32,
        min_step_factor: f32,
        mode: DrivenMode<'_>,
    ) -> Result<(Vec<f32>, usize, usize, f32, f32), String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        self.ensure_kcl_err_partial(gpu)?;
        let n = self.n_dof;
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;

        let mut traj = Vec::with_capacity((steps + 1) * n);
        traj.extend_from_slice(y0);

        // Controller state, carried across output frames.
        let mut h = dt;
        let mut prev_err = 0.0_f32;
        let h_min = min_step_factor * dt;
        let mut n_acc = 0_usize;
        let mut n_rej = 0_usize;
        let mut h_min_log = f32::INFINITY;
        let mut h_max_log = 0.0_f32;

        for k in 0..steps {
            let mut t_rel = 0.0_f32;
            // Per-frame source-hold value (zeroth-order hold across the
            // frame): for the point/vector driven cases, the controller's
            // substep convention matches the LSERK4 driven GPU path,
            // waveform sampled once per output cadence.
            let g_val = match &mode {
                DrivenMode::Free => 0.0_f32,
                DrivenMode::Point { g_values, .. } => g_values[k],
                DrivenMode::Vector { g_values } => g_values[k],
            };
            while t_rel < dt {
                let h_try = h.min(dt - t_rel);
                // Backup, attempt, evaluate err, decide.
                self.enqueue_kcl_backup(gpu)?;
                match &mode {
                    DrivenMode::Free => self.enqueue_kcl_step(gpu, h_try)?,
                    DrivenMode::Point { source_dof, .. } => self
                        .enqueue_kcl_step_driven_point(
                            gpu, h_try, *source_dof as cl_int, g_val,
                        )?,
                    DrivenMode::Vector { .. } => self
                        .enqueue_kcl_step_driven_vec(gpu, h_try, g_val)?,
                }
                let err_norm = self.read_kcl_err_norm(gpu, atol, rtol)?;
                let accept = err_norm.is_finite() && err_norm <= 1.0;
                if accept {
                    t_rel += h_try;
                    n_acc += 1;
                    let factor = Self::kcl_step_factor(
                        err_norm.max(1e-12), prev_err, safety, growth_limit,
                        shrink_limit, pi_alpha, pi_beta, false,
                    );
                    h = h_try * factor;
                    prev_err = err_norm.max(1e-12);
                    h_min_log = h_min_log.min(h);
                    h_max_log = h_max_log.max(h);
                } else {
                    n_rej += 1;
                    self.enqueue_kcl_restore(gpu)?;
                    let probe = if err_norm.is_finite() {
                        err_norm
                    } else {
                        10.0
                    };
                    let factor = Self::kcl_step_factor(
                        probe, prev_err, safety, growth_limit, shrink_limit,
                        pi_alpha, pi_beta, true,
                    );
                    h = h_try * factor;
                }
                if h < h_min {
                    return Err(format!(
                        "GPU KCL: step size collapsed below \
                         {min_step_factor:e}·dt after {n_acc} accepted, \
                         {n_rej} rejected substeps"
                    ));
                }
            }
            let row = gpu.download(&self.y, n)?;
            traj.extend_from_slice(&row);
        }
        Ok((traj, n_acc, n_rej, h_min_log, h_max_log))
    }

    /// Propagate `y0` for `steps` LSERK4 steps of size `dt`, fully on the
    /// device: the state stays resident, only `y0` (up) and the final
    /// state (down) cross the bus.
    pub fn transient(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        // Zero the residual register once; stage 0 (a = 0) keeps it reset
        // every step thereafter.
        let zeros = vec![0.0_f32; self.n_dof];
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.p, CL_BLOCKING, 0, &zeros, &[])
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        for _ in 0..steps {
            for stage in 0..LSERK4_STAGES {
                self.enqueue_apply(gpu)?;
                self.enqueue_lserk(
                    gpu,
                    LSERK4_A[stage] as f32,
                    LSERK4_B[stage] as f32,
                    dt,
                )?;
            }
        }
        gpu.queue()
            .finish()
            .map_err(|e| format!("transient sync failed: {e}"))?;
        gpu.download(&self.y, self.n_dof)
    }

    /// Like [`transient`](Self::transient) but returns the full field
    /// trajectory, flat `[(steps+1) * n_dof]` with row 0 the initial
    /// state. `dt` is the output cadence; the explicit integrator takes
    /// `substeps` LSERK4 steps of `dt/substeps` between snapshots, so the
    /// caller can keep the substep within the CFL limit while sampling at
    /// any cadence. One snapshot is downloaded per output step; the state
    /// itself steps device-resident.
    pub fn transient_traj(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        substeps: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        let n = self.n_dof;
        let substeps = substeps.max(1);
        let h = dt / substeps as f32;
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        let zeros = vec![0.0_f32; n];
        unsafe {
            gpu.queue().enqueue_write_buffer(
                &mut self.p, CL_BLOCKING, 0, &zeros, &[],
            )
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        let mut traj = Vec::with_capacity((steps + 1) * n);
        traj.extend_from_slice(y0);
        for _ in 0..steps {
            for _ in 0..substeps {
                for stage in 0..LSERK4_STAGES {
                    self.enqueue_apply(gpu)?;
                    self.enqueue_lserk(
                        gpu,
                        LSERK4_A[stage] as f32,
                        LSERK4_B[stage] as f32,
                        h,
                    )?;
                }
            }
            let row = gpu.download(&self.y, n)?;
            traj.extend_from_slice(&row);
        }
        Ok(traj)
    }

    /// Driven transient: `dy/dt = A.y + b`, with `b` a single-DOF soft
    /// source held constant across each step (the zeroth-order hold the
    /// CPU `step_driven` uses). `source_values[k]` is the source amplitude
    /// for step `k`, and its length sets the step count.
    pub fn transient_driven(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        source_dof: usize,
        source_values: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        assert!(source_dof < self.n_dof, "source_dof out of range");
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        let zeros = vec![0.0_f32; self.n_dof];
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.p, CL_BLOCKING, 0, &zeros, &[])
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        let dof = source_dof as cl_int;
        for &g in source_values {
            for stage in 0..LSERK4_STAGES {
                self.enqueue_apply(gpu)?;
                self.enqueue_add_source(gpu, dof, g)?;
                self.enqueue_lserk(
                    gpu,
                    LSERK4_A[stage] as f32,
                    LSERK4_B[stage] as f32,
                    dt,
                )?;
            }
        }
        gpu.queue()
            .finish()
            .map_err(|e| format!("driven transient sync failed: {e}"))?;
        gpu.download(&self.y, self.n_dof)
    }

    /// Driven transient returning the full trajectory, flat
    /// `[(steps+1) * n_dof]` with row 0 the initial state. `dt` is the
    /// output cadence; the integrator takes `substeps` LSERK4 steps of
    /// `dt/substeps` between snapshots. `source_values` holds one source
    /// amplitude per substep (length `steps * substeps`), so the caller
    /// re-samples the waveform per substep.
    pub fn transient_driven_traj(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        substeps: usize,
        source_dof: usize,
        source_values: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        assert!(source_dof < self.n_dof, "source_dof out of range");
        let substeps = substeps.max(1);
        assert_eq!(
            source_values.len(),
            steps * substeps,
            "source values must have steps*substeps entries",
        );
        let n = self.n_dof;
        let h = dt / substeps as f32;
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        let zeros = vec![0.0_f32; n];
        unsafe {
            gpu.queue().enqueue_write_buffer(
                &mut self.p, CL_BLOCKING, 0, &zeros, &[],
            )
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        let dof = source_dof as cl_int;
        let mut traj = Vec::with_capacity((steps + 1) * n);
        traj.extend_from_slice(y0);
        for k in 0..steps {
            for j in 0..substeps {
                let g = source_values[k * substeps + j];
                for stage in 0..LSERK4_STAGES {
                    self.enqueue_apply(gpu)?;
                    self.enqueue_add_source(gpu, dof, g)?;
                    self.enqueue_lserk(
                        gpu,
                        LSERK4_A[stage] as f32,
                        LSERK4_B[stage] as f32,
                        h,
                    )?;
                }
            }
            let row = gpu.download(&self.y, n)?;
            traj.extend_from_slice(&row);
        }
        Ok(traj)
    }

    /// Upload the source spatial pattern `b` to the device-resident `src`
    /// buffer. Call once before a vector-source drive; the per-step
    /// waveform is then applied by [`enqueue_add_source_vec`].
    fn upload_source(
        &mut self,
        gpu: &GpuContext,
        source: &[f32],
    ) -> Result<(), String> {
        assert_eq!(source.len(), self.n_dof, "source length mismatch");
        unsafe {
            gpu.queue().enqueue_write_buffer(
                &mut self.src, CL_BLOCKING, 0, source, &[],
            )
        }
        .map_err(|e| format!("source upload failed: {e}"))?;
        Ok(())
    }

    /// Vector-source driven transient: `dy/dt = A.y + b·g(t)`, with `b` the
    /// full spatial source pattern (modal-port injection) held constant
    /// across each step. `source` is `b` (length `n_dof`); `source_values`
    /// holds one waveform amplitude `g` per step. Returns the final state.
    pub fn transient_driven_vec(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        source: &[f32],
        source_values: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        self.upload_source(gpu, source)?;
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        let zeros = vec![0.0_f32; self.n_dof];
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.p, CL_BLOCKING, 0, &zeros, &[])
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        for &g in source_values {
            for stage in 0..LSERK4_STAGES {
                self.enqueue_apply(gpu)?;
                self.enqueue_add_source_vec(gpu, g)?;
                self.enqueue_lserk(
                    gpu,
                    LSERK4_A[stage] as f32,
                    LSERK4_B[stage] as f32,
                    dt,
                )?;
            }
        }
        gpu.queue()
            .finish()
            .map_err(|e| format!("driven transient sync failed: {e}"))?;
        gpu.download(&self.y, self.n_dof)
    }

    /// Vector-source driven transient returning the full trajectory, flat
    /// `[(steps+1) * n_dof]` with row 0 the initial state. The GPU
    /// counterpart of the CPU
    /// [`LserkWorkspace::step_with_source_into`](crate::explicit::LserkWorkspace::step_with_source_into)
    /// loop. `source` is the spatial pattern `b`; `source_values` holds one
    /// amplitude per substep (length `steps * substeps`).
    pub fn transient_driven_vec_traj(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        substeps: usize,
        source: &[f32],
        source_values: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        let substeps = substeps.max(1);
        assert_eq!(
            source_values.len(),
            steps * substeps,
            "source values must have steps*substeps entries",
        );
        let n = self.n_dof;
        let h = dt / substeps as f32;
        self.upload_source(gpu, source)?;
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        let zeros = vec![0.0_f32; n];
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.p, CL_BLOCKING, 0, &zeros, &[])
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        let mut traj = Vec::with_capacity((steps + 1) * n);
        traj.extend_from_slice(y0);
        for k in 0..steps {
            for j in 0..substeps {
                let g = source_values[k * substeps + j];
                for stage in 0..LSERK4_STAGES {
                    self.enqueue_apply(gpu)?;
                    self.enqueue_add_source_vec(gpu, g)?;
                    self.enqueue_lserk(
                        gpu,
                        LSERK4_A[stage] as f32,
                        LSERK4_B[stage] as f32,
                        h,
                    )?;
                }
            }
            let row = gpu.download(&self.y, n)?;
            traj.extend_from_slice(&row);
        }
        Ok(traj)
    }

    /// Ensure the f64 Krylov buffers are allocated for dimension `m`.
    fn ensure_krylov(
        &mut self,
        gpu: &GpuContext,
        m: usize,
    ) -> Result<(), String> {
        let n = self.n_dof;
        let big_enough =
            matches!(&self.krylov, Some(k) if k.cap_dim >= m);
        if !big_enough {
            let n_groups = n.div_ceil(NORM_WORK_GROUP);
            self.krylov = Some(Krylov {
                cap_dim: m,
                basis: gpu.alloc_f64((m + 1) * n)?,
                w: gpu.alloc_f64(n)?,
                proj: gpu.alloc_f64(m + 1)?,
                out: gpu.alloc_f64(n)?,
                partials: gpu.alloc_f64(n_groups)?,
                h: gpu.alloc_f64(m * m)?,
                hnext: gpu.alloc_f64(1)?,
                src64: gpu.alloc_f64(n)?,
                n_groups,
            });
        }
        Ok(())
    }

    /// Run an `m`-step Arnoldi process from `basis[0]` (unit-norm in slot
    /// 0). `h_dev` must be a zeroed `m*m` device buffer. The loop runs
    /// entirely device-side, projections accumulate straight into the
    /// device Hessenberg, the norm is finished on the device, so the
    /// Hessenberg is the *only* thing the host reads back, once, at the
    /// end. Fixed dimension `m` (no breakdown check).
    #[allow(clippy::too_many_arguments)]
    fn arnoldi(
        &self,
        gpu: &GpuContext,
        basis: &Buffer<cl_double>,
        w: &Buffer<cl_double>,
        proj: &Buffer<cl_double>,
        partials: &Buffer<cl_double>,
        h_dev: &Buffer<cl_double>,
        hnext_buf: &Buffer<cl_double>,
        n_groups: usize,
        m: usize,
    ) -> Result<(Vec<f64>, usize), String> {
        let n = self.n_dof;
        let n_i = n as cl_int;
        let m_i = m as cl_int;
        let ng_i = n_groups as cl_int;
        let elem_global = n.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let norm_global =
            n.div_ceil(NORM_WORK_GROUP) * NORM_WORK_GROUP;
        let ek = self
            .expmv_kernels
            .as_ref()
            .expect("expmv kernels built before arnoldi");

        for j in 0..m {
            // matvec w = A*basis[j]: cast basis[j] down, apply, cast up.
            let off = (j * n) as cl_int;
            unsafe {
                ExecuteKernel::new(&ek.cast_d2f)
                    .set_arg(basis)
                    .set_arg(&off)
                    .set_arg(&self.y)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("cast_d2f launch failed: {e}"))?;
            self.enqueue_apply(gpu)?;
            unsafe {
                ExecuteKernel::new(&ek.cast_f2d)
                    .set_arg(&self.dy)
                    .set_arg(w)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("cast_f2d launch failed: {e}"))?;

            // CGS2: two passes; each pass's projections accumulate into
            // column j of the device Hessenberg.
            let cols = j + 1;
            let cols_i = cols as cl_int;
            let col_i = j as cl_int;
            let proj_global =
                cols.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
            for _pass in 0..2 {
                unsafe {
                    ExecuteKernel::new(&ek.dot_rows)
                        .set_arg(basis)
                        .set_arg(w)
                        .set_arg(proj)
                        .set_arg(&n_i)
                        .set_arg(&cols_i)
                        .set_arg_local_buffer(DOF_WORK_GROUP * 8)
                        .set_global_work_size(cols * DOF_WORK_GROUP)
                        .set_local_work_size(DOF_WORK_GROUP)
                        .enqueue_nd_range(gpu.queue())
                }
                .map_err(|e| format!("dot_rows launch failed: {e}"))?;
                unsafe {
                    ExecuteKernel::new(&ek.store_h_col)
                        .set_arg(h_dev)
                        .set_arg(proj)
                        .set_arg(&col_i)
                        .set_arg(&m_i)
                        .set_arg(&cols_i)
                        .set_global_work_size(proj_global)
                        .set_local_work_size(DOF_WORK_GROUP)
                        .enqueue_nd_range(gpu.queue())
                }
                .map_err(|e| format!("store_h_col launch failed: {e}"))?;
                unsafe {
                    ExecuteKernel::new(&ek.axpy_basis)
                        .set_arg(w)
                        .set_arg(basis)
                        .set_arg(proj)
                        .set_arg(&n_i)
                        .set_arg(&cols_i)
                        .set_global_work_size(elem_global)
                        .set_local_work_size(DOF_WORK_GROUP)
                        .enqueue_nd_range(gpu.queue())
                }
                .map_err(|e| format!("axpy_basis launch failed: {e}"))?;
            }

            // The last basis vector needs no successor.
            if j + 1 == m {
                break;
            }

            // hnext = ||w||, finished device-side into `hnext_buf` and the
            // subdiagonal H[(j+1), j]; then basis[j+1] = w / hnext.
            unsafe {
                ExecuteKernel::new(&ek.norm2)
                    .set_arg(w)
                    .set_arg(partials)
                    .set_arg(&n_i)
                    .set_arg_local_buffer(NORM_WORK_GROUP * 8)
                    .set_global_work_size(norm_global)
                    .set_local_work_size(NORM_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("partial_norm2 launch failed: {e}"))?;
            unsafe {
                ExecuteKernel::new(&ek.finish_norm)
                    .set_arg(partials)
                    .set_arg(&ng_i)
                    .set_arg(hnext_buf)
                    .set_arg(h_dev)
                    .set_arg(&col_i)
                    .set_arg(&m_i)
                    .set_global_work_size(1)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("finish_norm launch failed: {e}"))?;
            let dst_off = ((j + 1) * n) as cl_int;
            unsafe {
                ExecuteKernel::new(&ek.scale_recip)
                    .set_arg(w)
                    .set_arg(basis)
                    .set_arg(&dst_off)
                    .set_arg(hnext_buf)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("scale_recip launch failed: {e}"))?;
        }
        // The Hessenberg is the only host round-trip, downloaded once.
        let h = gpu.download_f64(h_dev, m * m)?;
        Ok((h, m))
    }

    /// Krylov linear combination `out = sum_i coef[i] * basis[i]`.
    fn lincomb(
        &self,
        gpu: &GpuContext,
        basis: &Buffer<cl_double>,
        coef: &[f64],
        out: &Buffer<cl_double>,
    ) -> Result<(), String> {
        let n_i = self.n_dof as cl_int;
        let dim_i = coef.len() as cl_int;
        let elem_global =
            self.n_dof.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let coef_buf = gpu.upload_f64(coef)?;
        let ek = self
            .expmv_kernels
            .as_ref()
            .expect("expmv kernels built before lincomb");
        unsafe {
            ExecuteKernel::new(&ek.lincomb)
                .set_arg(basis)
                .set_arg(&coef_buf)
                .set_arg(out)
                .set_arg(&n_i)
                .set_arg(&dim_i)
                .set_global_work_size(elem_global)
                .set_local_work_size(DOF_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("lincomb launch failed: {e}"))?;
        Ok(())
    }

    /// Matrix-free `exp(t*A)*v` via an `m`-step Krylov projection, the GPU
    /// counterpart of [`crate::propagator::expmv`].
    ///
    /// For `m` above [`KRYLOV_CHUNK`] the propagation is **sub-stepped**:
    /// `exp(t*A) = exp((t/k)*A)^k` is exact, so `k` sub-steps each with a
    /// small Krylov space give the same result as one large space, but the
    /// device-resident Arnoldi basis is capped at `~KRYLOV_CHUNK * n_dof`
    /// rather than `~m * n_dof`. Sub-steps round-trip the state through the
    /// host; that transfer is small next to the Arnoldi itself.
    pub fn expmv(
        &mut self,
        gpu: &GpuContext,
        v: &[f64],
        t: f64,
        m: usize,
    ) -> Result<Vec<f64>, String> {
        assert_eq!(v.len(), self.n_dof, "state length mismatch");
        assert!(m >= 1, "Krylov dimension must be >= 1");
        self.ensure_expmv_kernels()?;
        let k = m.div_ceil(KRYLOV_CHUNK).max(1);
        let chunk = m.div_ceil(k);
        if k == 1 {
            return self.expmv_chunk(gpu, v, t, chunk);
        }
        // Sub-step: each piece covers t/k with a `chunk`-dimensional space.
        let tau = t / k as f64;
        let mut state = v.to_vec();
        for _ in 0..k {
            state = self.expmv_chunk(gpu, &state, tau, chunk)?;
        }
        Ok(state)
    }

    /// One Krylov sub-step: `exp(t*A)*v` from a single `m`-dimensional
    /// Arnoldi space. The basis is device-resident in f64 and the CGS2
    /// orthogonalisation runs in f64; the matvec drops through the f32
    /// `apply` kernel. The dense `exp(t*H)` of the small Hessenberg is done
    /// on the host.
    fn expmv_chunk(
        &mut self,
        gpu: &GpuContext,
        v: &[f64],
        t: f64,
        m: usize,
    ) -> Result<Vec<f64>, String> {
        let n = self.n_dof;
        assert_eq!(v.len(), n, "state length mismatch");
        assert!(m >= 1, "Krylov dimension must be >= 1");
        let beta: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if beta == 0.0 {
            return Ok(vec![0.0; n]);
        }
        self.ensure_krylov(gpu, m)?;
        let mut kry = self.krylov.take().expect("krylov allocated");

        // basis[0] = v / beta; zero the device Hessenberg.
        let b0: Vec<f64> = v.iter().map(|x| x / beta).collect();
        gpu.write_f64(&mut kry.basis, &b0)?;
        gpu.write_f64(&mut kry.h, &vec![0.0_f64; m * m])?;
        let (h, dim) = self.arnoldi(
            gpu, &kry.basis, &kry.w, &kry.proj, &kry.partials, &kry.h,
            &kry.hnext, kry.n_groups, m,
        )?;

        // Dense exp(t*H) on the host; out = beta * sum_i basis[i] * exp[i,0].
        let mut th = vec![0.0_f64; dim * dim];
        for a in 0..dim {
            for b in 0..dim {
                th[a * dim + b] = t * h[a * m + b];
            }
        }
        let exp_th = expm(&th, dim);
        let coef: Vec<f64> =
            (0..dim).map(|i| beta * exp_th[i * dim]).collect();
        self.lincomb(gpu, &kry.basis, &coef, &kry.out)?;
        gpu.queue()
            .finish()
            .map_err(|e| format!("expmv sync failed: {e}"))?;
        let result = gpu.download_f64(&kry.out, n)?;
        self.krylov = Some(kry);
        Ok(result)
    }

    /// One exponential-time-differencing step of `dy/dt = A·y + b` on the
    /// GPU, with the source `b` held constant across the step:
    /// `y ← exp(hA)·y + h·φ₁(hA)·b`. The GPU counterpart of
    /// [`crate::propagator::etd_step`].
    ///
    /// Uses the same augmented-matrix identity
    /// `exp(h·[[A, b],[0, 0]])·[y; 1] = [exp(hA)y + h·φ₁(hA)b ; 1]`: the
    /// `n`-vector Arnoldi basis stays device-resident, while the `(n+1)`th
    /// augmented ξ-component is carried host-side as a per-basis scalar
    /// (`s`) and the source column `b` is injected into the working vector
    /// by the f64 `axpy_src` device kernel. `b` may be a single-DOF point
    /// source (`b = e_dof·val`) or a full spatial pattern (modal-port
    /// injection), the path is identical.
    ///
    /// `m` caps the Krylov dimension; above [`KRYLOV_CHUNK`] the augmented
    /// propagation is sub-stepped just as [`expmv`](Self::expmv), the
    /// augmented scalar reset to 1 between sub-steps (the augmented
    /// operator's zero last row preserves it).
    pub fn etd_step(
        &mut self,
        gpu: &GpuContext,
        y: &[f64],
        b: &[f64],
        h: f64,
        m: usize,
    ) -> Result<Vec<f64>, String> {
        assert_eq!(y.len(), self.n_dof, "state length mismatch");
        assert_eq!(b.len(), self.n_dof, "source length mismatch");
        assert!(m >= 1, "Krylov dimension must be >= 1");
        self.ensure_expmv_kernels()?;
        self.ensure_krylov(gpu, m)?;
        // Upload the source pattern once; every sub-step reuses it.
        {
            let mut kry = self.krylov.take().expect("krylov allocated");
            gpu.write_f64(&mut kry.src64, b)?;
            self.krylov = Some(kry);
        }
        let k = m.div_ceil(KRYLOV_CHUNK).max(1);
        let chunk = m.div_ceil(k);
        let tau = h / k as f64;
        let mut state = y.to_vec();
        for _ in 0..k {
            state = self.etd_expmv_chunk(gpu, &state, tau, chunk)?;
        }
        Ok(state)
    }

    /// One augmented-Arnoldi sub-step of the ETD propagator: returns the
    /// first `n` components of `exp(tau·[[A,b],[0,0]])·[v; 1]`, i.e.
    /// `exp(tau·A)·v + tau·φ₁(tau·A)·b`. The source `b` is the
    /// device-resident `krylov.src64`, uploaded by [`etd_step`](Self::etd_step).
    ///
    /// Unlike the unforced [`arnoldi`](Self::arnoldi), this cannot run
    /// purely device-side: the augmented inner product carries the scalar
    /// term `s[i]·w_scalar`, so each CGS2 projection is corrected and each
    /// subtraction updates `w_scalar` host-side. The Hessenberg therefore
    /// also lives host-side here (the projections already round-trip).
    fn etd_expmv_chunk(
        &mut self,
        gpu: &GpuContext,
        v: &[f64],
        tau: f64,
        m: usize,
    ) -> Result<Vec<f64>, String> {
        let n = self.n_dof;
        assert_eq!(v.len(), n, "state length mismatch");
        assert!(m >= 1, "Krylov dimension must be >= 1");

        // Augmented start z0 = [v; 1]; beta = ||z0|| includes the unit scalar.
        let beta = (v.iter().map(|x| x * x).sum::<f64>() + 1.0).sqrt();
        let inv_beta = 1.0 / beta;

        let mut kry = self.krylov.take().expect("krylov allocated");
        // basis[0].vec = v/beta on the device; the augmented scalar parts
        // live host-side in `s`, with s[0] = (1)/beta.
        let b0: Vec<f64> = v.iter().map(|x| x * inv_beta).collect();
        gpu.write_f64(&mut kry.basis, &b0)?;
        let mut s = vec![0.0_f64; m + 1];
        s[0] = inv_beta;
        let mut h_host = vec![0.0_f64; m * m];

        let n_i = n as cl_int;
        let elem_global = n.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let norm_global = n.div_ceil(NORM_WORK_GROUP) * NORM_WORK_GROUP;
        let ek = self
            .expmv_kernels
            .as_ref()
            .expect("expmv kernels built before etd_expmv_chunk");

        for j in 0..m {
            // Augmented matvec: w_vec = A·basis_vec[j] + s[j]·b; w_scalar = 0.
            let off = (j * n) as cl_int;
            unsafe {
                ExecuteKernel::new(&ek.cast_d2f)
                    .set_arg(&kry.basis)
                    .set_arg(&off)
                    .set_arg(&self.y)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("cast_d2f launch failed: {e}"))?;
            self.enqueue_apply(gpu)?;
            unsafe {
                ExecuteKernel::new(&ek.cast_f2d)
                    .set_arg(&self.dy)
                    .set_arg(&kry.w)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("cast_f2d launch failed: {e}"))?;
            let sj = s[j];
            unsafe {
                ExecuteKernel::new(&ek.axpy_src)
                    .set_arg(&kry.w)
                    .set_arg(&kry.src64)
                    .set_arg(&sj)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("axpy_src launch failed: {e}"))?;
            let mut w_scalar = 0.0_f64;

            // CGS2: two passes. Each pass's projections take the augmented
            // scalar term s[i]·w_scalar (host correction); the subtraction
            // updates w_scalar -= Σ proj[i]·s[i]. H column j accumulates
            // both passes, exactly as the CPU `expmv_into`.
            let cols = j + 1;
            let cols_i = cols as cl_int;
            for _pass in 0..2 {
                unsafe {
                    ExecuteKernel::new(&ek.dot_rows)
                        .set_arg(&kry.basis)
                        .set_arg(&kry.w)
                        .set_arg(&kry.proj)
                        .set_arg(&n_i)
                        .set_arg(&cols_i)
                        .set_arg_local_buffer(DOF_WORK_GROUP * 8)
                        .set_global_work_size(cols * DOF_WORK_GROUP)
                        .set_local_work_size(DOF_WORK_GROUP)
                        .enqueue_nd_range(gpu.queue())
                }
                .map_err(|e| format!("dot_rows launch failed: {e}"))?;
                // Blocking read serialises behind dot_rows on the in-order
                // queue.
                let mut proj_host = gpu.download_f64(&kry.proj, cols)?;
                for i in 0..cols {
                    proj_host[i] += s[i] * w_scalar;
                    h_host[i * m + j] += proj_host[i];
                }
                let dscalar: f64 =
                    (0..cols).map(|i| proj_host[i] * s[i]).sum();
                gpu.write_f64(&mut kry.proj, &proj_host)?;
                unsafe {
                    ExecuteKernel::new(&ek.axpy_basis)
                        .set_arg(&kry.w)
                        .set_arg(&kry.basis)
                        .set_arg(&kry.proj)
                        .set_arg(&n_i)
                        .set_arg(&cols_i)
                        .set_global_work_size(elem_global)
                        .set_local_work_size(DOF_WORK_GROUP)
                        .enqueue_nd_range(gpu.queue())
                }
                .map_err(|e| format!("axpy_basis launch failed: {e}"))?;
                w_scalar -= dscalar;
            }

            // The last basis vector needs no successor.
            if j + 1 == m {
                break;
            }

            // hnext = ||w_aug|| = sqrt(||w_vec||² + w_scalar²): the vector
            // norm is reduced on the device, the scalar added host-side.
            unsafe {
                ExecuteKernel::new(&ek.norm2)
                    .set_arg(&kry.w)
                    .set_arg(&kry.partials)
                    .set_arg(&n_i)
                    .set_arg_local_buffer(NORM_WORK_GROUP * 8)
                    .set_global_work_size(norm_global)
                    .set_local_work_size(NORM_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("partial_norm2 launch failed: {e}"))?;
            let partials = gpu.download_f64(&kry.partials, kry.n_groups)?;
            let wvec_n2: f64 = partials.iter().sum();
            let hnext = (wvec_n2 + w_scalar * w_scalar).sqrt();
            h_host[(j + 1) * m + j] = hnext;

            // basis[j+1].vec = w_vec / hnext (device); s[j+1] = w_scalar / hnext.
            gpu.write_f64(&mut kry.hnext, &[hnext])?;
            let dst_off = ((j + 1) * n) as cl_int;
            unsafe {
                ExecuteKernel::new(&ek.scale_recip)
                    .set_arg(&kry.w)
                    .set_arg(&kry.basis)
                    .set_arg(&dst_off)
                    .set_arg(&kry.hnext)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("scale_recip launch failed: {e}"))?;
            s[j + 1] = w_scalar / hnext;
        }

        // exp(tau·H) on the host; out_vec = beta·Σ_i basis_vec[i]·exp[i,0]
        //, the first n components of the augmented result. The augmented
        // scalar component is preserved at 1 and discarded.
        let dim = m;
        let mut th = vec![0.0_f64; dim * dim];
        for a in 0..dim {
            for b in 0..dim {
                th[a * dim + b] = tau * h_host[a * m + b];
            }
        }
        let exp_th = expm(&th, dim);
        let coef: Vec<f64> =
            (0..dim).map(|i| beta * exp_th[i * dim]).collect();
        self.lincomb(gpu, &kry.basis, &coef, &kry.out)?;
        gpu.queue()
            .finish()
            .map_err(|e| format!("etd expmv sync failed: {e}"))?;
        let result = gpu.download_f64(&kry.out, n)?;
        self.krylov = Some(kry);
        Ok(result)
    }

    /// Exponential-warmup hybrid transient: the first `warmup` steps use
    /// the exact exponential propagator, the rest the cheaper explicit
    /// LSERK4 stepper. The exact integrator carries the opening transient,
    /// then hands the smooth state to the explicit stepper.
    pub fn transient_hybrid(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        warmup: usize,
        krylov_dim: usize,
    ) -> Result<Vec<f32>, String> {
        let warmup = warmup.min(steps);
        // Warmup: exact exponential steps in f64.
        let mut y: Vec<f64> = y0.iter().map(|&v| v as f64).collect();
        for _ in 0..warmup {
            y = self.expmv(gpu, &y, dt as f64, krylov_dim)?;
        }
        // Remainder: device-resident explicit LSERK4.
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        self.transient(gpu, &y32, dt, steps - warmup)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GPU_REL_TOL;
    use crate::mesh_gen::structured_box;

    /// True if the error reports the device lacks f64 (`cl_khr_fp64`), so
    /// the exponential-propagator tests skip on Apple Silicon and similar
    /// fp64-less GPUs rather than fail, the explicit f32 path is unaffected.
    fn fp64_unavailable(err: &str) -> bool {
        err.contains("fp64") || err.contains("f64")
    }

    /// Relative L2 error of the GPU result against the CPU f64 reference.
    fn rel_l2(gpu: &[f32], cpu: &[Field]) -> f64 {
        let err: f64 = cpu
            .iter()
            .zip(gpu)
            .map(|(&c, &g)| (c as f64 - g as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 =
            cpu.iter().map(|&c| (c as f64).powi(2)).sum::<f64>().sqrt();
        err / scale
    }

    #[test]
    fn gpu_apply_matches_cpu() {
        // P1 gate: the GPU apply matches the CPU f64 apply within the
        // mixed-precision budget GPU_REL_TOL, for a vacuum cavity (trivial
        // materials) and a dielectric fill (inv_eps != 1).
        use crate::rhs::ElemMaterial;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let vacuum = MaxwellOperator::new(&mesh, 2, 1.0);
        let dielectric = MaxwellOperator::new_with_materials(
            &mesh,
            2,
            1.0,
            &vec![ElemMaterial::isotropic(4.0, 1.0, 0.0); mesh.n_tets()],
        );

        for (label, op) in
            [("vacuum", &vacuum), ("dielectric er=4", &dielectric)]
        {
            let n = op.n_dof();
            let y: Vec<Field> =
                (0..n).map(|i| (0.3 + i as Field * 0.017).sin()).collect();
            let cpu_dy = op.apply(&y);

            let mut gop = GpuOperator::new(&gpu, op).expect("GpuOperator");
            assert_eq!(gop.n_dof(), n);
            let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let gpu_dy = gop.apply(&gpu, &y32).expect("gpu apply");

            let rel = rel_l2(&gpu_dy, &cpu_dy);
            eprintln!(
                "GPU apply vs CPU f64 [{label}]: rel L2 = {rel:.3e} \
                 (GPU_REL_TOL {GPU_REL_TOL:.1e})"
            );
            assert!(
                rel < GPU_REL_TOL,
                "GPU apply [{label}] rel.err {rel:.3e} exceeds GPU_REL_TOL",
            );
        }
    }

    #[test]
    fn gpu_kcl_adaptive_matches_cpu() {
        // KCL adaptive on the GPU: the device-resident PI controller
        // produces a trajectory that tracks the CPU adaptive run frame by
        // frame, within the f32 GPU_REL_TOL budget. Both controllers start
        // from the same atol/rtol/seed so the substep paths line up.
        use crate::explicit_adaptive::KclWorkspace;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (0.2 + i as Field * 0.011).sin()).collect();

        // Sub-CFL frame cadence, the controller will mostly run at h≈dt.
        let mut v = y0.clone();
        let mut rho = 1.0;
        for _ in 0..30 {
            let av = op.apply(&v);
            rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
            let inv = 1.0 / rho;
            for (vi, &a) in v.iter_mut().zip(&av) {
                *vi = a * inv;
            }
        }
        let dt = 1.0 / rho;
        let steps = 60;

        // Controller params shared between CPU and GPU. atol/rtol loose
        // enough that f32 GPU noise doesn't make the two paths diverge.
        let atol = 1e-6_f32;
        let rtol = 1e-3_f32;
        let safety = 0.9_f32;
        let growth = 5.0_f32;
        let shrink = 0.2_f32;
        let alpha = 0.7 / 4.0_f32;
        let beta = 0.4 / 4.0_f32;
        let min_step = 1e-10_f32;

        // CPU reference using the same controller logic (mirrored locally
        // since the CPU controller lives in Python). The trajectory only
        // needs to match at output cadence, not substep-by-substep.
        let mut y_cpu = y0.clone();
        let mut ws = KclWorkspace::new();
        let mut err = vec![0.0; n];
        let mut h_cpu = dt;
        let mut prev_err = 0.0_f64;
        let mut traj_cpu = Vec::with_capacity((steps + 1) * n);
        traj_cpu.extend_from_slice(&y_cpu);
        for _ in 0..steps {
            let mut t_rel = 0.0_f64;
            while t_rel < dt {
                let h_try = h_cpu.min(dt - t_rel);
                let y_pre = y_cpu.clone();
                ws.step_into(
                    |x, ax| op.apply_into(x, ax),
                    &mut y_cpu, &mut err, h_try,
                );
                let mut s2 = 0.0_f64;
                for i in 0..n {
                    let scale = atol as f64
                        + rtol as f64
                            * y_pre[i].abs().max(y_cpu[i].abs());
                    let r = err[i] / scale;
                    s2 += r * r;
                }
                let err_norm = (s2 / n as f64).sqrt();
                if err_norm <= 1.0 && err_norm.is_finite() {
                    t_rel += h_try;
                    let f = if prev_err <= 0.0 {
                        safety as f64
                            * err_norm.max(1e-12).powf(-(alpha as f64))
                    } else {
                        safety as f64
                            * err_norm.max(1e-12).powf(-(alpha as f64))
                            * prev_err.powf(beta as f64)
                    };
                    let f = f.max(shrink as f64).min(growth as f64);
                    h_cpu = h_try * f;
                    prev_err = err_norm.max(1e-12);
                } else {
                    y_cpu = y_pre;
                    let probe = if err_norm.is_finite() {
                        err_norm
                    } else {
                        10.0
                    };
                    let f = (safety as f64
                        * probe.max(1e-12).powf(-(alpha as f64)))
                        .max(shrink as f64);
                    h_cpu = h_try * f;
                }
            }
            traj_cpu.extend_from_slice(&y_cpu);
        }

        // GPU run with matched parameters.
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let (traj_gpu, n_acc, n_rej, h_min, h_max) = gop
            .transient_kcl_traj(
                &gpu, &y0_32, dt as f32, steps, atol, rtol, safety,
                growth, shrink, alpha, beta, min_step,
            )
            .expect("transient_kcl_traj");

        eprintln!(
            "GPU KCL adaptive: {n_acc} accepted, {n_rej} rejected; \
             h ∈ [{h_min:.3e}, {h_max:.3e}]"
        );
        // Compare the final accepted snapshot, same controller, same
        // tolerances, so the two paths should match within f32 noise.
        let last_gpu = &traj_gpu[steps * n..(steps + 1) * n];
        let last_cpu = &traj_cpu[steps * n..(steps + 1) * n];
        let rel = rel_l2(last_gpu, last_cpu);
        eprintln!(
            "GPU KCL vs CPU KCL [{steps} frames]: rel L2 = {rel:.3e} \
             (GPU_REL_TOL {GPU_REL_TOL:.1e})"
        );
        assert!(
            rel < 10.0 * GPU_REL_TOL,
            "GPU KCL rel.err {rel:.3e} exceeds 10·GPU_REL_TOL, the f32 \
             controller takes a different substep path",
        );
    }

    #[test]
    fn gpu_transient_matches_cpu() {
        // P2 gate: the device-resident GPU LSERK4 transient matches the
        // CPU LSERK4 transient within GPU_REL_TOL.
        use crate::explicit::LserkWorkspace;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (0.2 + i as Field * 0.011).sin()).collect();

        // Spectral radius by power iteration, for a sub-CFL step.
        let mut v = y0.clone();
        let mut rho = 1.0;
        for _ in 0..30 {
            let av = op.apply(&v);
            rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
            let inv = 1.0 / rho;
            for (vi, &a) in v.iter_mut().zip(&av) {
                *vi = a * inv;
            }
        }
        let dt = 1.0 / rho;
        let steps = 200;

        // CPU reference.
        let mut y_cpu = y0.clone();
        let mut ws = LserkWorkspace::new();
        for _ in 0..steps {
            ws.step_into(|x, ax| op.apply_into(x, ax), &mut y_cpu, dt);
        }

        // GPU, device-resident.
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let y_gpu = gop
            .transient(&gpu, &y0_32, dt as f32, steps)
            .expect("transient");

        let rel = rel_l2(&y_gpu, &y_cpu);
        eprintln!(
            "GPU transient vs CPU [{steps} steps]: rel L2 = {rel:.3e} \
             (GPU_REL_TOL {GPU_REL_TOL:.1e})"
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU transient rel.err {rel:.3e} exceeds GPU_REL_TOL",
        );
    }

    #[test]
    fn gpu_driven_transient_matches_cpu() {
        // P2.3 gate: the GPU driven transient (soft source) matches the
        // CPU driven LSERK4 within GPU_REL_TOL.
        use crate::explicit::LserkWorkspace;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();

        let mut v: Vec<Field> =
            (0..n).map(|i| (0.1 + i as Field * 0.007).sin()).collect();
        let mut rho = 1.0;
        for _ in 0..30 {
            let av = op.apply(&v);
            rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
            let inv = 1.0 / rho;
            for (vi, &a) in v.iter_mut().zip(&av) {
                *vi = a * inv;
            }
        }
        let dt = 1.0 / rho;
        let steps = 150;
        let sdof = n / 3;
        let src: Vec<Field> =
            (0..steps).map(|k| (0.3 * k as Field).sin()).collect();

        // CPU reference, driven from rest.
        let mut y_cpu = vec![0.0; n];
        let mut ws = LserkWorkspace::new();
        for &g in &src {
            ws.step_driven_into(
                |x, ax| op.apply_into(x, ax),
                &mut y_cpu,
                dt,
                sdof,
                g,
            );
        }

        // GPU.
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0 = vec![0.0_f32; n];
        let src32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        let y_gpu = gop
            .transient_driven(&gpu, &y0, dt as f32, sdof, &src32)
            .expect("driven transient");

        let rel = rel_l2(&y_gpu, &y_cpu);
        eprintln!(
            "GPU driven transient vs CPU [{steps} steps]: rel L2 = {rel:.3e}"
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU driven transient rel.err {rel:.3e} exceeds GPU_REL_TOL",
        );
    }

    #[test]
    fn gpu_vector_source_transient_matches_cpu() {
        // Vector-source parity: the GPU full-vector driven transient (the
        // modal-port injection path) matches the CPU LSERK4
        // step_with_source_into within GPU_REL_TOL. The source spreads over
        // many DOFs, unlike the single-DOF point case above.
        use crate::explicit::LserkWorkspace;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();

        let mut v: Vec<Field> =
            (0..n).map(|i| (0.1 + i as Field * 0.007).sin()).collect();
        let mut rho = 1.0;
        for _ in 0..30 {
            let av = op.apply(&v);
            rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
            let inv = 1.0 / rho;
            for (vi, &a) in v.iter_mut().zip(&av) {
                *vi = a * inv;
            }
        }
        let dt = 1.0 / rho;
        let steps = 150;
        // A spread-out spatial source pattern, nonzero on many DOFs.
        let b: Vec<Field> =
            (0..n).map(|i| 0.5 * (0.09 * i as Field).cos()).collect();
        let gvals: Vec<Field> =
            (0..steps).map(|k| (0.3 * k as Field).sin()).collect();

        // CPU reference, driven from rest with the full vector source.
        let mut y_cpu = vec![0.0; n];
        let mut ws = LserkWorkspace::new();
        for &g in &gvals {
            let bg: Vec<Field> = b.iter().map(|&bi| bi * g).collect();
            ws.step_with_source_into(
                |x, ax| op.apply_into(x, ax),
                &mut y_cpu,
                dt,
                &bg,
            );
        }

        // GPU, one substep per step, so the loops line up one-to-one.
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0 = vec![0.0_f32; n];
        let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let g32: Vec<f32> = gvals.iter().map(|&v| v as f32).collect();
        let y_gpu = gop
            .transient_driven_vec(&gpu, &y0, dt as f32, &b32, &g32)
            .expect("vector-source transient");

        let rel = rel_l2(&y_gpu, &y_cpu);
        eprintln!(
            "GPU vector-source transient vs CPU [{steps} steps]: \
             rel L2 = {rel:.3e}"
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU vector-source transient rel.err {rel:.3e} exceeds \
             GPU_REL_TOL",
        );
    }

    #[test]
    fn gpu_expmv_matches_cpu() {
        // P3 gate: the GPU Krylov exponential propagator matches the CPU
        // expmv within GPU_REL_TOL (the f32 matvec caps the accuracy).
        use crate::propagator::expmv;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let v: Vec<Field> =
            (0..n).map(|i| (0.3 + i as Field * 0.013).sin()).collect();
        let t = 0.02;
        let m = 40;

        let cpu = expmv(|x| op.apply(x), &v, t, m);

        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let gpu_out = match gop.expmv(&gpu, &v, t, m) {
            Ok(o) => o,
            Err(e) if fp64_unavailable(&e) => {
                eprintln!("skipping GPU expmv test (no fp64): {e}");
                return;
            }
            Err(e) => panic!("gpu expmv: {e}"),
        };

        let err: f64 = cpu
            .iter()
            .zip(&gpu_out)
            .map(|(&c, &g)| (c - g).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = cpu.iter().map(|&c| c * c).sum::<f64>().sqrt();
        let rel = err / scale;
        eprintln!(
            "GPU expmv vs CPU [m={m}]: rel L2 = {rel:.3e} \
             (GPU_REL_TOL {GPU_REL_TOL:.1e})"
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU expmv rel.err {rel:.3e} exceeds GPU_REL_TOL",
        );
    }

    #[test]
    fn gpu_hybrid_transient_matches_cpu() {
        // P2.4 gate: the GPU exponential-warmup hybrid matches a CPU hybrid
        // (warmup expmv steps, then LSERK4) within GPU_REL_TOL.
        use crate::explicit::LserkWorkspace;
        use crate::propagator::expmv;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (0.15 + i as Field * 0.009).sin()).collect();

        let mut v = y0.clone();
        let mut rho = 1.0;
        for _ in 0..30 {
            let av = op.apply(&v);
            rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
            let inv = 1.0 / rho;
            for (vi, &a) in v.iter_mut().zip(&av) {
                *vi = a * inv;
            }
        }
        let dt = 1.0 / rho;
        let (steps, warmup, m) = (120, 10, 40);

        // CPU hybrid: warmup exponential steps, then LSERK4.
        let mut y_cpu = y0.clone();
        for _ in 0..warmup {
            y_cpu = expmv(|x| op.apply(x), &y_cpu, dt, m);
        }
        let mut ws = LserkWorkspace::new();
        for _ in 0..(steps - warmup) {
            ws.step_into(|x, ax| op.apply_into(x, ax), &mut y_cpu, dt);
        }

        // GPU hybrid.
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let y_gpu = match gop
            .transient_hybrid(&gpu, &y0_32, dt as f32, steps, warmup, m)
        {
            Ok(o) => o,
            Err(e) if fp64_unavailable(&e) => {
                eprintln!("skipping GPU hybrid test (no fp64): {e}");
                return;
            }
            Err(e) => panic!("hybrid transient: {e}"),
        };

        let rel = rel_l2(&y_gpu, &y_cpu);
        eprintln!(
            "GPU hybrid transient vs CPU [{warmup}+{} steps]: rel L2 = {rel:.3e}",
            steps - warmup
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU hybrid rel.err {rel:.3e} exceeds GPU_REL_TOL",
        );
    }

    #[test]
    fn gpu_etd_step_matches_cpu() {
        // WP3 gate: the GPU augmented-Arnoldi ETD step (driven exponential
        // propagator) matches the CPU `etd_step` within GPU_REL_TOL, for
        // both a single-DOF point source (b = e_dof·val) and a spread-out
        // vector source (modal-port injection). Skips on fp64-less devices.
        use crate::propagator::etd_step;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let y: Vec<Field> =
            (0..n).map(|i| (0.25 + i as Field * 0.013).sin()).collect();
        let h = 0.02;
        let m = 40;

        // Point source: a single DOF, the b = e_dof·val special case.
        let mut b_point = vec![0.0_f64; n];
        b_point[n / 3] = 1.7;
        // Vector source: a spread spatial pattern over many DOFs.
        let b_vec: Vec<f64> =
            (0..n).map(|i| 0.4 * (0.07 * i as Field).cos()).collect();

        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");

        for (label, b) in [("point", &b_point), ("vector", &b_vec)] {
            let cpu = etd_step(|x| op.apply(x), &y, b, h, m);
            let gpu_out = match gop.etd_step(&gpu, &y, b, h, m) {
                Ok(o) => o,
                Err(e) if fp64_unavailable(&e) => {
                    eprintln!("skipping GPU etd_step test (no fp64): {e}");
                    return;
                }
                Err(e) => panic!("gpu etd_step [{label}]: {e}"),
            };
            let err: f64 = cpu
                .iter()
                .zip(&gpu_out)
                .map(|(&c, &g)| (c - g).powi(2))
                .sum::<f64>()
                .sqrt();
            let scale: f64 =
                cpu.iter().map(|&c| c * c).sum::<f64>().sqrt();
            let rel = err / scale;
            eprintln!(
                "GPU etd_step vs CPU [{label}, m={m}]: rel L2 = {rel:.3e} \
                 (GPU_REL_TOL {GPU_REL_TOL:.1e})"
            );
            assert!(
                rel < GPU_REL_TOL,
                "GPU etd_step [{label}] rel.err {rel:.3e} exceeds GPU_REL_TOL",
            );
        }
    }
}
