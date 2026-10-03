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
//! [`GpuContext`] holds device discovery, the context and command queue,
//! buffer up/download and program build; [`GpuOperator`] runs the DG
//! operator kernels on top of it.

use std::ptr;

use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;

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
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer<T>, String> {
        let mut buf = unsafe {
            Buffer::<T>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                data.len(),
                ptr::null_mut(),
            )
        }
        .map_err(|e| format!("buffer creation failed: {e}"))?;
        self.write(&mut buf, data)?;
        Ok(buf)
    }

    /// Allocate a read-write device buffer of `len` elements, uninitialised.
    pub fn alloc<T: Copy>(&self, len: usize) -> Result<Buffer<T>, String> {
        unsafe {
            Buffer::<T>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                len,
                ptr::null_mut(),
            )
        }
        .map_err(|e| format!("buffer allocation failed: {e}"))
    }

    /// Download `len` elements from a device buffer into a host vector.
    pub fn download<T: Copy + Default>(
        &self,
        buf: &Buffer<T>,
        len: usize,
    ) -> Result<Vec<T>, String> {
        let mut out = vec![T::default(); len];
        unsafe {
            self.queue
                .enqueue_read_buffer(buf, CL_BLOCKING, 0, &mut out, &[])
        }
        .map_err(|e| format!("buffer read failed: {e}"))?;
        Ok(out)
    }

    /// Write a host slice into an existing device buffer.
    pub fn write<T: Copy>(
        &self,
        buf: &mut Buffer<T>,
        data: &[T],
    ) -> Result<(), String> {
        unsafe {
            self.queue
                .enqueue_write_buffer(buf, CL_BLOCKING, 0, data, &[])
        }
        .map_err(|e| format!("buffer write failed: {e}"))?;
        Ok(())
    }

    /// The command queue kernels are enqueued on.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }
}
