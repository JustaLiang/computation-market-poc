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
pub mod suite;
pub mod wgpu_backend;

/// Apple/MLX backend, behind the `mlx` feature. Placeholder until `mlx-rs` is
/// wired up (needs `cmake`); see [`mlx_backend`].
#[cfg(feature = "mlx")]
pub mod mlx_backend;

pub use backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, NetworkResult, Precision};
pub use suite::{run_suite, ComputeIndex, Report, SuiteConfig};
pub use wgpu_backend::WgpuBackend;

#[cfg(feature = "mlx")]
pub use mlx_backend::MlxBackend;
