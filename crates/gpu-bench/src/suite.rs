//! Runs a *suite* of measurements against one [`Backend`]: GEMM throughput
//! (fp32, and fp16 where supported), memory bandwidth, and a network probe.
//!
//! It reports the **raw numbers**. There is deliberately no single "compute
//! index" — the weighting/formula is undecided, so we don't publish a score.

use serde::Serialize;

use crate::backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, NetworkResult, Precision};

/// What to run. Defaults are a quick, laptop-friendly pass. The network probe is
/// best-effort (offline is not fatal) and on by default; `include_network`
/// lets a multi-backend caller run it just once instead of per backend.
#[derive(Debug, Clone)]
pub struct SuiteConfig {
    /// Square matrix dimension for GEMM.
    pub n: u32,
    pub iters: u32,
    pub warmup: u32,
    /// Elements per buffer for the bandwidth test.
    pub bandwidth_elems: u64,
    /// Run the fp16 GEMM in addition to fp32 (skipped if unsupported).
    pub include_fp16: bool,
    /// Run the network probe. The probe is backend-independent, so a caller that
    /// benchmarks several backends in one go probes once (primary suite) and sets
    /// this `false` on the others to avoid redundant network round-trips.
    pub include_network: bool,
}

impl Default for SuiteConfig {
    fn default() -> Self {
        Self {
            n: 2048,
            iters: 50,
            warmup: 10,
            bandwidth_elems: 32_000_000,
            include_fp16: true,
            include_network: true,
        }
    }
}

/// All measurements from one suite run.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub device: DeviceInfo,
    pub gemm_fp32: Option<GemmResult>,
    pub gemm_fp16: Option<GemmResult>,
    pub bandwidth: Option<BandwidthResult>,
    /// Network down/up/latency (best-effort; `None` if the probe failed).
    pub network: Option<NetworkResult>,
}

/// Run fp32 GEMM, optionally fp16 GEMM, memory bandwidth, and a network probe.
///
/// fp32 GEMM is required (an error fails the suite). fp16, bandwidth, and the
/// network probe are best-effort: on failure they are recorded as `None` and
/// noted, not fatal (e.g. the cuBLAS backend measures GEMM only).
///
/// `on_phase` is called with a short label before each (blocking) phase, so a
/// caller can drive a spinner. Pass `|_| {}` if you don't care.
pub fn run_suite(
    backend: &dyn Backend,
    cfg: &SuiteConfig,
    mut on_phase: impl FnMut(&str),
) -> anyhow::Result<Report> {
    let device = backend.device_info();

    on_phase("fp32 GEMM");
    let gemm_fp32 = Some(backend.gemm(cfg.n, Precision::F32, cfg.warmup, cfg.iters)?);

    let gemm_fp16 = if cfg.include_fp16 {
        on_phase("fp16 GEMM");
        match backend.gemm(cfg.n, Precision::F16, cfg.warmup, cfg.iters) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("note: fp16 GEMM skipped ({e})");
                None
            }
        }
    } else {
        None
    };

    on_phase("memory bandwidth");
    let bandwidth = match backend.bandwidth(cfg.bandwidth_elems, cfg.warmup, cfg.iters) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("note: bandwidth skipped ({e})");
            None
        }
    };

    // Network probe: best-effort (offline is not fatal), and skippable so a
    // multi-backend caller probes only once.
    let network = if cfg.include_network {
        on_phase("network probe");
        match crate::network::probe(&crate::network::NetConfig::default()) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("note: network probe skipped ({e})");
                None
            }
        }
    } else {
        None
    };

    Ok(Report {
        device,
        gemm_fp32,
        gemm_fp16,
        bandwidth,
        network,
    })
}
