// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! OpenCL GPU host layer for the time-domain backend.
//!
//! Optional, behind the `gpu` feature. OpenCL is loaded at runtime through
//! the ICD loader, so building this adds no toolkit dependency; a machine
//! with no GPU or no OpenCL runtime simply never constructs a
//! [`GpuContext`] and the CPU path runs unchanged.
//!
//! This module is the foundation (plan phase P0.1): device discovery, a
//! context and command queue, buffer up/download, and program build. The
//! DG operator kernels land on top of it in later phases.

use std::ptr;

use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_double, cl_float, cl_int};

mod operator;
pub use operator::GpuOperator;

/// A GPU device with its OpenCL context and command queue.
///
/// Construction discovers the first available GPU device. A `None`-style
/// failure (no device, no runtime) is reported as `Err` so the caller can
/// fall back to the CPU path.
pub struct GpuContext {
    /// Human-readable device name, for logging.
    pub device_name: String,
    /// Whether the device has double precision (`cl_khr_fp64`), which the
    /// exponential propagator needs; the explicit paths run in f32.
    pub fp64: bool,
    context: Context,
    queue: CommandQueue,
}

impl GpuContext {
    /// Discover the first OpenCL GPU device and set up a context and queue.
    pub fn new() -> Result<Self, String> {
        let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|e| format!("OpenCL device query failed: {e}"))?;
        let device_id = *device_ids
            .first()
            .ok_or_else(|| "no OpenCL GPU device found".to_string())?;
        let device = Device::new(device_id);
        let device_name =
            device.name().map_err(|e| format!("device name: {e}"))?;
        let fp64 = device
            .extensions()
            .is_ok_and(|ext| ext.split_whitespace().any(|e| e == "cl_khr_fp64"));
        let context = Context::from_device(&device)
            .map_err(|e| format!("context creation failed: {e}"))?;
        let queue = CommandQueue::create_default(
            &context,
            CL_QUEUE_PROFILING_ENABLE,
        )
        .map_err(|e| format!("command queue creation failed: {e}"))?;
        Ok(GpuContext { device_name, fp64, context, queue })
    }

    /// Build an OpenCL program from kernel source. The `Err` carries the
    /// build log.
    pub fn build_program(&self, source: &str) -> Result<Program, String> {
        Program::create_and_build_from_source(&self.context, source, "")
            .map_err(|log| format!("kernel build failed:\n{log}"))
    }

    /// Upload a host slice into a fresh read-only device buffer.
    pub fn upload(&self, data: &[f32]) -> Result<Buffer<cl_float>, String> {
        let mut buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                data.len(),
                ptr::null_mut(),
            )
        }
        .map_err(|e| format!("buffer creation failed: {e}"))?;
        unsafe {
            self.queue.enqueue_write_buffer(
                &mut buf,
                CL_BLOCKING,
                0,
                data,
                &[],
            )
        }
        .map_err(|e| format!("buffer write failed: {e}"))?;
        Ok(buf)
    }

    /// Upload a host slice into a fresh read-only device buffer of ints.
    pub fn upload_i32(&self, data: &[i32]) -> Result<Buffer<cl_int>, String> {
        let mut buf = unsafe {
            Buffer::<cl_int>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                data.len(),
                ptr::null_mut(),
            )
        }
        .map_err(|e| format!("int buffer creation failed: {e}"))?;
        unsafe {
            self.queue.enqueue_write_buffer(
                &mut buf,
                CL_BLOCKING,
                0,
                data,
                &[],
            )
        }
        .map_err(|e| format!("int buffer write failed: {e}"))?;
        Ok(buf)
    }

    /// Allocate a read-write device buffer of `len` floats, uninitialised.
    pub fn alloc(&self, len: usize) -> Result<Buffer<cl_float>, String> {
        unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                len,
                ptr::null_mut(),
            )
        }
        .map_err(|e| format!("buffer allocation failed: {e}"))
    }

    /// Download `len` floats from a device buffer into a host vector.
    pub fn download(
        &self,
        buf: &Buffer<cl_float>,
        len: usize,
    ) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0_f32; len];
        unsafe {
            self.queue.enqueue_read_buffer(
                buf,
                CL_BLOCKING,
                0,
                &mut out,
                &[],
            )
        }
        .map_err(|e| format!("buffer read failed: {e}"))?;
        Ok(out)
    }

    /// Upload an `f64` host slice into a fresh read-only device buffer.
    pub fn upload_f64(
        &self,
        data: &[f64],
    ) -> Result<Buffer<cl_double>, String> {
        let mut buf = unsafe {
            Buffer::<cl_double>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                data.len(),
                ptr::null_mut(),
            )
        }
        .map_err(|e| format!("f64 buffer creation failed: {e}"))?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, data, &[])
        }
        .map_err(|e| format!("f64 buffer write failed: {e}"))?;
        Ok(buf)
    }

    /// Allocate a read-write `f64` device buffer of `len` elements.
    pub fn alloc_f64(
        &self,
        len: usize,
    ) -> Result<Buffer<cl_double>, String> {
        unsafe {
            Buffer::<cl_double>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                len,
                ptr::null_mut(),
            )
        }
        .map_err(|e| format!("f64 buffer allocation failed: {e}"))
    }

    /// Download `len` `f64` elements from a device buffer.
    pub fn download_f64(
        &self,
        buf: &Buffer<cl_double>,
        len: usize,
    ) -> Result<Vec<f64>, String> {
        let mut out = vec![0.0_f64; len];
        unsafe {
            self.queue
                .enqueue_read_buffer(buf, CL_BLOCKING, 0, &mut out, &[])
        }
        .map_err(|e| format!("f64 buffer read failed: {e}"))?;
        Ok(out)
    }

    /// The command queue, for kernel enqueue in later phases.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Write an `f64` host slice into an existing device buffer.
    pub fn write_f64(
        &self,
        buf: &mut Buffer<cl_double>,
        data: &[f64],
    ) -> Result<(), String> {
        unsafe {
            self.queue
                .enqueue_write_buffer(buf, CL_BLOCKING, 0, data, &[])
        }
        .map_err(|e| format!("f64 buffer write failed: {e}"))?;
        Ok(())
    }
}

/// A trivial elementwise-add kernel, the P0.1 smoke test that the host
/// layer can build and run a kernel end to end on the device.
const VADD_SOURCE: &str = r#"
kernel void vadd(global const float* a,
                 global const float* b,
                 global float* c) {
    const size_t i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"#;

/// Run `c = a + b` on the GPU, the smoke test for [`GpuContext`].
pub fn vector_add(
    gpu: &GpuContext,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, String> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let program = gpu.build_program(VADD_SOURCE)?;
    let kernel = Kernel::create(&program, "vadd")
        .map_err(|e| format!("kernel create failed: {e}"))?;
    let a_buf = gpu.upload(a)?;
    let b_buf = gpu.upload(b)?;
    let c_buf = gpu.alloc(n)?;
    let event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&a_buf)
            .set_arg(&b_buf)
            .set_arg(&c_buf)
            .set_global_work_size(n)
            .enqueue_nd_range(&gpu.queue)
    }
    .map_err(|e| format!("kernel launch failed: {e}"))?;
    event.wait().map_err(|e| format!("kernel wait failed: {e}"))?;
    gpu.download(&c_buf, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_add_runs_on_the_gpu() {
        // P0.1 gate: the host layer builds and runs a kernel on the device.
        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                // No GPU on this machine, skip rather than fail.
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        eprintln!("GPU device: {}", gpu.device_name);

        let n = 4096;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 0.25).collect();
        let c = vector_add(&gpu, &a, &b).expect("vector_add");

        for i in 0..n {
            let want = a[i] + b[i];
            assert!(
                (c[i] - want).abs() <= 1e-6 * want.abs().max(1.0),
                "index {i}: got {}, want {want}",
                c[i],
            );
        }
    }
}
