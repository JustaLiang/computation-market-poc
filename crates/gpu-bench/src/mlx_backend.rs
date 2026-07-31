//! MLX backend (Apple Silicon) — reserved placeholder.
//!
//! MLX is Apple's array/ML framework; its matmul runs through optimized Metal
//! kernels that reach near the GPU's true fp16 peak, unlike the portable WGSL
//! kernel in [`crate::wgpu_backend`] (which can't touch the matrix units). This
//! is the "near-peak Apple number" backend.
//!
//! It is **not implemented yet**: wiring the `mlx-rs` crate pulls in Apple's MLX
//! C++ library, which builds from source and needs `cmake`. This module exists
//! so the `Backend` slot and the `mlx` feature flag are in place. To finish it:
//!   1. add `mlx-rs` as an optional dependency; set `mlx = ["dep:mlx-rs"]`,
//!   2. build f16 arrays and loop `matmul` + `eval` (MLX is lazy) to time GEMM,
//!   3. return the same [`GemmResult`]/[`BandwidthResult`] as the wgpu backend,
//!      so the suite consumes it unchanged.

use crate::backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, Precision};

const UNIMPLEMENTED: &str =
    "the MLX backend is not implemented yet — it needs `cmake` and the `mlx-rs` \
     dependency (see crates/gpu-bench/src/mlx_backend.rs)";

/// Placeholder for the Apple/MLX backend. Cannot be constructed until MLX is
/// wired up; [`MlxBackend::new`] returns an explanatory error.
pub struct MlxBackend;

impl MlxBackend {
    pub fn new() -> anyhow::Result<Self> {
        anyhow::bail!(UNIMPLEMENTED)
    }
}

impl Backend for MlxBackend {
    fn device_info(&self) -> DeviceInfo {
        // Unreachable: no `MlxBackend` value can exist while `new` always errors.
        unreachable!("{UNIMPLEMENTED}")
    }

    fn gemm(
        &self,
        _n: u32,
        _p: Precision,
        _warmup: u32,
        _iters: u32,
    ) -> anyhow::Result<GemmResult> {
        anyhow::bail!(UNIMPLEMENTED)
    }

    fn bandwidth(
        &self,
        _elements: u64,
        _warmup: u32,
        _iters: u32,
    ) -> anyhow::Result<BandwidthResult> {
        anyhow::bail!(UNIMPLEMENTED)
    }
}
