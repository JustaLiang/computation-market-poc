//! Native Apple-GPU backend — peak GEMM via a Metal `simdgroup_matrix` kernel.
//!
//! macOS-only, pure Rust (no Python, no `cmake`): `objc2-metal` drives the GPU
//! and we compile a small MSL kernel at runtime that uses `simdgroup_matrix` —
//! the Apple GPU's matrix units (its "tensor cores"). That reaches far above the
//! portable WGSL ALU path in [`crate::wgpu_backend`]. Bandwidth is a blit copy.
//!
//! Binding: `objc2-metal` (actively maintained; the older `metal`/metal-rs crate
//! is deprecated). The protocol methods that take raw pointers — buffer bytes,
//! `setBytes`, the blit copy — are `unsafe`; those are the only `unsafe` blocks.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::time::Instant;

use anyhow::Context;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};

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

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct MetalBackend {
    device: Device,
    queue: Queue,
    library: Library,
    name: String,
}

impl MetalBackend {
    pub fn new() -> anyhow::Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device found")?;
        let queue = device
            .newCommandQueue()
            .context("creating a Metal command queue")?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(GEMM_MSL), None)
            .map_err(|e| anyhow::anyhow!("compiling MSL kernel: {e}"))?;
        let name = device.name().to_string();
        Ok(Self {
            device,
            queue,
            library,
            name,
        })
    }

    fn pipeline(&self, func: &str) -> anyhow::Result<Pipeline> {
        let function = self
            .library
            .newFunctionWithName(&NSString::from_str(func))
            .with_context(|| format!("kernel {func} not found in library"))?;
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| anyhow::anyhow!("building pipeline {func}: {e}"))
    }

    fn shared_buffer<T>(&self, data: &[T]) -> Buffer {
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).expect("buffer data is non-null");
        // SAFETY: `ptr` covers `size_of_val(data)` initialized bytes and stays
        // valid for the call; Metal copies them into shared storage immediately.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                ptr,
                std::mem::size_of_val(data),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("buffer allocation failed")
    }

    /// Encode `count` dispatches of `pipeline` over an N×N grid into one command
    /// buffer, submit, and wait — amortizing CPU submit overhead.
    fn run_gemm(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        a: &ProtocolObject<dyn MTLBuffer>,
        b: &ProtocolObject<dyn MTLBuffer>,
        c: &ProtocolObject<dyn MTLBuffer>,
        n: u32,
        count: u32,
    ) {
        let cmd = self.queue.commandBuffer().expect("command buffer");
        let enc = cmd.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(pipeline);
        let nbytes = NonNull::new(&n as *const u32 as *mut c_void).expect("stack ptr is non-null");
        // SAFETY: a/b/c outlive the encoder; `nbytes` points at `n` (4 bytes),
        // valid for the call; indices match the kernel's [[buffer(k)]] slots.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a), 0, 0);
            enc.setBuffer_offset_atIndex(Some(b), 0, 1);
            enc.setBuffer_offset_atIndex(Some(c), 0, 2);
            enc.setBytes_length_atIndex(nbytes, 4, 3);
        }
        let g = (n / 8) as usize;
        let grid = MTLSize {
            width: g,
            height: g,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        }; // one simdgroup
        for _ in 0..count {
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
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
        let c = self
            .device
            .newBufferWithLength_options(
                len * std::mem::size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .context("allocating the output buffer")?;

        if warmup > 0 {
            self.run_gemm(&pipeline, &a, &b, &c, n, warmup);
        }
        let start = Instant::now();
        self.run_gemm(&pipeline, &a, &b, &c, n, iterations);
        let seconds = start.elapsed().as_secs_f64();

        // A, B all-ones → every C element is n.
        // SAFETY: shared-storage buffer is CPU-visible; the GEMM has completed
        // (we waited), and the first element is a valid fp32.
        let first = unsafe { *(c.contents().as_ptr() as *const f32) };
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
        let bytes = elements as usize * std::mem::size_of::<f32>();
        let src = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModePrivate)
            .context("allocating the source buffer")?;
        let dst = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModePrivate)
            .context("allocating the destination buffer")?;

        let run = |count: u32| {
            let cmd = self.queue.commandBuffer().expect("command buffer");
            let enc = cmd.blitCommandEncoder().expect("blit encoder");
            // SAFETY: src/dst outlive the encoder and the range [0, bytes) is
            // within both buffers.
            unsafe {
                for _ in 0..count {
                    enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &src, 0, &dst, 0, bytes,
                    );
                }
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        };

        if warmup > 0 {
            run(warmup);
        }
        let start = Instant::now();
        run(iterations);
        let seconds = start.elapsed().as_secs_f64();

        // A blit copy is one read + one write.
        let bytes_per_iter = bytes as u64 * 2;
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
