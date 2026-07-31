//! Native Apple-GPU backend — peak GEMM via a Metal `simdgroup_matrix` kernel.
//!
//! macOS-only, pure Rust (no Python, no `cmake`): `metal-rs` drives the GPU and
//! we compile a small MSL kernel at runtime that uses `simdgroup_matrix` — the
//! Apple GPU's matrix units (its "tensor cores"). That reaches far above the
//! portable WGSL ALU path in [`crate::wgpu_backend`]. Bandwidth is a blit copy.

use std::ffi::c_void;
use std::time::Instant;

use anyhow::Context;
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

use crate::backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, Precision};

/// fp16-input/fp32-accumulate and fp32 GEMM, each simdgroup computing one 8×8
/// output tile. `N` must be a multiple of 8.
const GEMM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant uint&      N [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.y * 8, col = tgid.x * 8;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < N; k += 8) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, A + row * N + k, N);
        simdgroup_load(b, B + k * N + col, N);
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    simdgroup_store(acc, C + row * N + col, N);
}

kernel void gemm_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float*      C [[buffer(2)]],
    constant uint&     N [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.y * 8, col = tgid.x * 8;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < N; k += 8) {
        simdgroup_half8x8 a, b;
        simdgroup_load(a, A + row * N + k, N);
        simdgroup_load(b, B + k * N + col, N);
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    simdgroup_store(acc, C + row * N + col, N);
}
"#;

pub struct MetalBackend {
    device: Device,
    queue: metal::CommandQueue,
    library: metal::Library,
    name: String,
}

impl MetalBackend {
    pub fn new() -> anyhow::Result<Self> {
        let device = Device::system_default().context("no Metal device found")?;
        let queue = device.new_command_queue();
        let library = device
            .new_library_with_source(GEMM_MSL, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("compiling MSL kernel: {e}"))?;
        let name = device.name().to_string();
        Ok(Self {
            device,
            queue,
            library,
            name,
        })
    }

    fn pipeline(&self, func: &str) -> anyhow::Result<metal::ComputePipelineState> {
        let function = self
            .library
            .get_function(func, None)
            .map_err(|e| anyhow::anyhow!("loading kernel {func}: {e}"))?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow::anyhow!("building pipeline {func}: {e}"))
    }

    fn shared_buffer<T>(&self, data: &[T]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Encode `count` dispatches of `pipeline` over an N×N grid into one command
    /// buffer, submit, and wait — amortizing CPU submit overhead.
    fn run_gemm(
        &self,
        pipeline: &metal::ComputePipelineState,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        n: u32,
        count: u32,
    ) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        enc.set_bytes(3, 4, (&n as *const u32).cast());
        let grid = MTLSize::new((n / 8) as u64, (n / 8) as u64, 1);
        let threadgroup = MTLSize::new(32, 1, 1); // one simdgroup
        for _ in 0..count {
            enc.dispatch_thread_groups(grid, threadgroup);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

impl Backend for MetalBackend {
    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.name.clone(),
            backend: "Metal".to_string(),
            device_type: "IntegratedGpu".to_string(),
        }
    }

    fn gemm(
        &self,
        n: u32,
        precision: Precision,
        warmup: u32,
        iterations: u32,
    ) -> anyhow::Result<GemmResult> {
        anyhow::ensure!(
            n >= 8 && n % 8 == 0,
            "the Metal backend needs n to be a positive multiple of 8 (got {n})"
        );
        let len = (n as usize) * (n as usize);

        let (pipeline, a, b) = match precision {
            Precision::F32 => (
                self.pipeline("gemm_f32")?,
                self.shared_buffer(&vec![1.0f32; len]),
                self.shared_buffer(&vec![1.0f32; len]),
            ),
            Precision::F16 => (
                self.pipeline("gemm_f16")?,
                self.shared_buffer(&vec![half::f16::ONE; len]),
                self.shared_buffer(&vec![half::f16::ONE; len]),
            ),
        };
        // C is always fp32 (the accumulator type).
        let c = self.device.new_buffer(
            (len * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        if warmup > 0 {
            self.run_gemm(&pipeline, &a, &b, &c, n, warmup);
        }
        let start = Instant::now();
        self.run_gemm(&pipeline, &a, &b, &c, n, iterations);
        let seconds = start.elapsed().as_secs_f64();

        // A, B all-ones → every C element is n.
        let first = unsafe { *(c.contents() as *const f32) };
        let verified = (first - n as f32).abs() < (n as f32 * 1e-3).max(1.0);

        let tflops = 2.0 * (n as f64).powi(3) * iterations as f64 / seconds / 1e12;
        Ok(GemmResult {
            device: self.name.clone(),
            backend: "Metal".to_string(),
            precision,
            n,
            iterations,
            seconds,
            tflops,
            verified,
        })
    }

    fn bandwidth(
        &self,
        elements: u64,
        warmup: u32,
        iterations: u32,
    ) -> anyhow::Result<BandwidthResult> {
        let bytes = elements * std::mem::size_of::<f32>() as u64;
        let src = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModePrivate);
        let dst = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModePrivate);

        let run = |count: u32| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_blit_command_encoder();
            for _ in 0..count {
                enc.copy_from_buffer(&src, 0, &dst, 0, bytes);
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        };

        if warmup > 0 {
            run(warmup);
        }
        let start = Instant::now();
        run(iterations);
        let seconds = start.elapsed().as_secs_f64();

        // A blit copy is one read + one write.
        let bytes_per_iter = bytes * 2;
        let gb_per_s = (bytes_per_iter as f64 * iterations as f64) / seconds / 1e9;
        Ok(BandwidthResult {
            device: self.name.clone(),
            backend: "Metal".to_string(),
            bytes_per_iter,
            iterations,
            seconds,
            gb_per_s,
        })
    }
}
