//! `gpu-bench` — a vendor-agnostic GPU compute benchmark.
//!
//! Measures GEMM (matrix-multiply) throughput and memory bandwidth through a
//! [`Backend`] abstraction. The portable [`WgpuBackend`] runs on Metal, Vulkan,
//! and DX12; native peak-throughput backends — Apple [`MetalBackend`] and cuBLAS
//! (`feature = "cuda"`) — implement the same trait.
//!
//! This crate is deliberately standalone — it has no knowledge of the rental
//! marketplace. The host-agent can later derive `dlperf` from a GEMM result
//! (e.g. `normalize(result.tflops)`), but that adapter lives on the marketplace
//! side; nothing here depends on it.

pub mod backend;
pub mod network;
pub mod suite;
pub mod wgpu_backend;

/// Native Apple-GPU backend (Metal + `simdgroup_matrix`) — peak, macOS-only.
#[cfg(target_os = "macos")]
pub mod metal_backend;

/// cuBLAS backend, behind the `cuda` feature — runs on an NVIDIA host.
#[cfg(feature = "cuda")]
pub mod cuda_backend;

pub use backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, NetworkResult, Precision};
pub use network::{probe as probe_network, NetConfig};
pub use suite::{run_suite, Report, SuiteConfig};
pub use wgpu_backend::WgpuBackend;

#[cfg(target_os = "macos")]
pub use metal_backend::MetalBackend;

#[cfg(feature = "cuda")]
pub use cuda_backend::CudaBackend;
