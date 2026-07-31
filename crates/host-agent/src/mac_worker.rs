//! Native Metal GEMM worker — what a macOS rental runs. **Rust-only, no Python.**
//!
//! The agent re-invokes its own binary as `vgpu-agent __mac-worker <n> <port>`.
//! This runs a continuous fp16 GEMM on the Apple GPU via a Metal
//! `simdgroup_matrix` kernel (the matrix units) and serves a tiny JSON status
//! endpoint on `<port>` that the tenant can poll for live throughput. The agent
//! kills the process on `stop_rental`.
//!
//! Binding: `objc2-metal` (the older `metal`/metal-rs crate is deprecated). All
//! Metal objects are created and used inside [`gemm_loop`] on a single thread —
//! nothing Metal crosses the thread boundary — so there is no `Send` concern.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Context;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

/// fp16-input / fp32-accumulate GEMM: each simdgroup computes one 8×8 tile.
const GEMM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

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

/// Shared job stats: (iterations, TFLOP/s).
type Stats = Arc<Mutex<(u64, f64)>>;

/// Entry point for the `__mac-worker` subcommand. Never returns until killed.
pub fn run(n: u32, port: u16) -> anyhow::Result<()> {
    anyhow::ensure!(n >= 8 && n % 8 == 0, "n must be a multiple of 8 (got {n})");
    let stats: Stats = Arc::new(Mutex::new((0, 0.0)));

    // GEMM loop on its own thread; the HTTP server runs on this one.
    let worker = stats.clone();
    std::thread::spawn(move || {
        if let Err(e) = gemm_loop(n, &worker) {
            eprintln!("[mac-worker] GEMM loop failed: {e:#}");
        }
    });

    serve_status(port, n, &stats)
}

fn gemm_loop(n: u32, stats: &Stats) -> anyhow::Result<()> {
    let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let queue = device
        .newCommandQueue()
        .context("creating a command queue")?;
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(GEMM_MSL), None)
        .map_err(|e| anyhow::anyhow!("compiling MSL: {e}"))?;
    let function = library
        .newFunctionWithName(&NSString::from_str("gemm_f16"))
        .context("kernel gemm_f16 not found")?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| anyhow::anyhow!("building pipeline: {e}"))?;

    let len = (n as usize) * (n as usize);
    let ones = vec![half::f16::ONE; len];
    let ones_ptr = NonNull::new(ones.as_ptr() as *mut c_void).expect("buffer data is non-null");
    let bytes16 = len * std::mem::size_of::<half::f16>();
    // SAFETY: `ones_ptr` covers `bytes16` initialized bytes valid for the call;
    // Metal copies them into shared storage. A and B are identical all-ones.
    let a = unsafe {
        device.newBufferWithBytes_length_options(
            ones_ptr,
            bytes16,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .context("allocating A")?;
    let b = unsafe {
        device.newBufferWithBytes_length_options(
            ones_ptr,
            bytes16,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .context("allocating B")?;
    let c = device
        .newBufferWithLength_options(
            len * std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .context("allocating C")?;

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
    const BATCH: u32 = 20;

    let start = Instant::now();
    let mut iters: u64 = 0;
    loop {
        let cmd = queue.commandBuffer().context("command buffer")?;
        let enc = cmd.computeCommandEncoder().context("compute encoder")?;
        enc.setComputePipelineState(&pipeline);
        let nbytes = NonNull::new(&n as *const u32 as *mut c_void).expect("stack ptr is non-null");
        // SAFETY: a/b/c outlive the encoder; `nbytes` points at `n` (4 bytes),
        // valid for the call; indices match the kernel's [[buffer(k)]] slots.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&c), 0, 2);
            enc.setBytes_length_atIndex(nbytes, 4, 3);
        }
        for _ in 0..BATCH {
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        iters += BATCH as u64;
        let elapsed = start.elapsed().as_secs_f64();
        let tflops = 2.0 * (n as f64).powi(3) * iters as f64 / elapsed / 1e12;
        *stats.lock().expect("stats lock") = (iters, tflops);
    }
}

fn serve_status(port: u16, n: u32, stats: &Stats) -> anyhow::Result<()> {
    let listener = TcpListener::bind(("0.0.0.0", port))
        .with_context(|| format!("binding status port {port}"))?;
    println!("[mac-worker] serving status on :{port} (task=gemm-fp16 n={n})");
    for stream in listener.incoming() {
        let Ok(mut s) = stream else { continue };
        let _ = s.read(&mut [0u8; 512]); // drain the request line; we ignore it
        let (iters, tflops) = *stats.lock().expect("stats lock");
        let body =
            format!(r#"{{"task":"gemm-fp16","n":{n},"iters":{iters},"tflops":{tflops:.3}}}"#);
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = s.write_all(response.as_bytes());
    }
    Ok(())
}
