//! The vendor-agnostic benchmark interface.
//!
//! A [`Backend`] is one way of running compute on one device: the portable
//! `wgpu` backend (Metal/Vulkan/DX12) implements it today; native cuBLAS, MLX,
//! and rocBLAS backends will implement the same trait behind Cargo features.
//! Everything above this trait — the CLI, the report, a future `dlperf` adapter
//! — is written against the trait and never against a specific vendor.

use serde::Serialize;

/// Numeric precision of a GEMM run. `dlperf` cares about `F16`; `F32` is the
/// universally supported baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    F16,
}

impl Precision {
    /// Bytes per scalar element on the wire.
    pub fn size(self) -> u64 {
        match self {
            Precision::F32 => 4,
            Precision::F16 => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F32 => "fp32",
            Precision::F16 => "fp16",
        }
    }
}

/// A compute device a backend can target.
#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    /// Backend/API used to reach it, e.g. "Metal", "Vulkan", "CUDA".
    pub backend: String,
    /// "discrete-gpu", "integrated-gpu", "cpu", etc.
    pub device_type: String,
}

/// Result of a GEMM (matrix-multiply) throughput run.
#[derive(Debug, Clone, Serialize)]
pub struct GemmResult {
    pub device: String,
    pub backend: String,
    pub precision: Precision,
    /// Square matrix dimension: an `n × n` by `n × n` multiply.
    pub n: u32,
    pub iterations: u32,
    pub seconds: f64,
    /// Sustained throughput in teraFLOP/s: `2·n³·iterations / seconds / 1e12`.
    pub tflops: f64,
    /// Whether the computed matrix matched the analytic reference — a false here
    /// means the throughput number is meaningless, so callers must check it.
    pub verified: bool,
}

/// Result of a streaming memory-bandwidth run.
#[derive(Debug, Clone, Serialize)]
pub struct BandwidthResult {
    pub device: String,
    pub backend: String,
    /// Bytes moved per iteration (reads + writes).
    pub bytes_per_iter: u64,
    pub iterations: u32,
    pub seconds: f64,
    pub gb_per_s: f64,
}

/// One benchmarkable device.
pub trait Backend {
    fn device_info(&self) -> DeviceInfo;

    /// Multiply two `n × n` matrices `iterations` times (after `warmup`
    /// untimed iterations) and report sustained throughput.
    fn gemm(
        &self,
        n: u32,
        precision: Precision,
        warmup: u32,
        iterations: u32,
    ) -> anyhow::Result<GemmResult>;

    /// Stream `elements` f32 values through a read-read-write kernel and report
    /// sustained bandwidth.
    fn bandwidth(
        &self,
        elements: u64,
        warmup: u32,
        iterations: u32,
    ) -> anyhow::Result<BandwidthResult>;
}
