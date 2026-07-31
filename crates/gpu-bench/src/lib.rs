//! `gpu-bench` — a vendor-agnostic GPU compute benchmark.
//!
//! Measures GEMM (matrix-multiply) throughput and memory bandwidth through a
//! [`Backend`] abstraction. The portable [`WgpuBackend`] runs on Metal, Vulkan,
//! and DX12; native peak-throughput backends (cuBLAS, MLX, rocBLAS) implement
//! the same trait behind Cargo features.
//!
//! This crate is deliberately standalone — it has no knowledge of the rental
//! marketplace. The host-agent can later derive `dlperf` from a GEMM result
//! (e.g. `normalize(result.tflops)`), but that adapter lives on the marketplace
//! side; nothing here depends on it.

pub mod backend;
pub mod network;
pub mod suite;
pub mod wgpu_backend;

/// Apple/MLX backend (peak numbers via Python MLX). Always compiled; needs a
/// Python with `mlx` at runtime (`VGPU_MLX_PYTHON`).
pub mod mlx_backend;

/// cuBLAS backend, behind the `cuda` feature — runs on an NVIDIA host.
#[cfg(feature = "cuda")]
pub mod cuda_backend;

pub use backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, NetworkResult, Precision};
pub use mlx_backend::MlxBackend;
pub use network::{probe as probe_network, NetConfig};
pub use suite::{run_suite, Report, SuiteConfig};
pub use wgpu_backend::WgpuBackend;

#[cfg(feature = "cuda")]
pub use cuda_backend::CudaBackend;
