//! Runs a *suite* of measurements against one [`Backend`] and blends them into
//! a single provisional **compute index**.
//!
//! The index is the seed of the SPEC's "computation index": today it combines
//! GEMM throughput and memory bandwidth; the network slot is reserved and
//! weighted zero until a network probe exists. The reference constants below are
//! **placeholders, not calibrated** — the number is for relative comparison and
//! to exercise the shape, not an authoritative score yet.

use serde::Serialize;

use crate::backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, NetworkResult, Precision};

/// What to run. Defaults are a quick, laptop-friendly pass.
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
}

impl Default for SuiteConfig {
    fn default() -> Self {
        Self {
            n: 2048,
            iters: 50,
            warmup: 10,
            bandwidth_elems: 32_000_000,
            include_fp16: true,
        }
    }
}

/// Every measurement plus the derived index.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub device: DeviceInfo,
    pub gemm_fp32: Option<GemmResult>,
    pub gemm_fp16: Option<GemmResult>,
    pub bandwidth: Option<BandwidthResult>,
    /// Reserved; always `None` until a network probe is added.
    pub network: Option<NetworkResult>,
    pub index: ComputeIndex,
}

/// A provisional, transparent blend of the measurements.
#[derive(Debug, Clone, Serialize)]
pub struct ComputeIndex {
    /// Weighted score (see module docs). Higher is faster.
    pub score: f64,
    /// GEMM component, as a percent of the reference throughput.
    pub compute_component: f64,
    /// Bandwidth component, as a percent of the reference bandwidth.
    pub memory_component: f64,
    /// Reserved network component; `None` until measured.
    pub network_component: Option<f64>,
    pub note: &'static str,
}

// Provisional reference points — placeholders, NOT calibrated to real hardware.
const REF_TFLOPS: f64 = 100.0; // stand-in for a strong accelerator's portable GEMM
const REF_GBPS: f64 = 1000.0; // stand-in for high-end HBM bandwidth
const W_COMPUTE: f64 = 0.7;
const W_MEMORY: f64 = 0.3;

/// Run fp32 GEMM, optionally fp16 GEMM, and bandwidth; fold into a [`Report`].
///
/// fp32 GEMM and bandwidth are required (an error fails the suite). fp16 is
/// best-effort: if the backend can't do it, it's recorded as `None` and noted,
/// not fatal.
pub fn run_suite(backend: &dyn Backend, cfg: &SuiteConfig) -> anyhow::Result<Report> {
    let device = backend.device_info();

    let gemm_fp32 = Some(backend.gemm(cfg.n, Precision::F32, cfg.warmup, cfg.iters)?);

    let gemm_fp16 = if cfg.include_fp16 {
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

    let bandwidth = Some(backend.bandwidth(cfg.bandwidth_elems, cfg.warmup, cfg.iters)?);

    let index = compute_index(
        gemm_fp16.as_ref().or(gemm_fp32.as_ref()),
        bandwidth.as_ref(),
    );

    Ok(Report {
        device,
        gemm_fp32,
        gemm_fp16,
        bandwidth,
        network: None,
        index,
    })
}

/// Blend the best available GEMM and bandwidth into the provisional index.
fn compute_index(gemm: Option<&GemmResult>, bandwidth: Option<&BandwidthResult>) -> ComputeIndex {
    let compute_component = gemm.map_or(0.0, |g| 100.0 * g.tflops / REF_TFLOPS);
    let memory_component = bandwidth.map_or(0.0, |b| 100.0 * b.gb_per_s / REF_GBPS);
    let score = W_COMPUTE * compute_component + W_MEMORY * memory_component;
    ComputeIndex {
        score,
        compute_component,
        memory_component,
        network_component: None,
        note: "provisional v0: placeholder references, uncalibrated; network reserved",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemm(tflops: f64) -> GemmResult {
        GemmResult {
            device: "test".into(),
            backend: "test".into(),
            precision: Precision::F16,
            n: 2048,
            iterations: 1,
            seconds: 1.0,
            tflops,
            verified: true,
        }
    }

    fn bw(gb_per_s: f64) -> BandwidthResult {
        BandwidthResult {
            device: "test".into(),
            backend: "test".into(),
            bytes_per_iter: 1,
            iterations: 1,
            seconds: 1.0,
            gb_per_s,
        }
    }

    #[test]
    fn index_blends_components_by_weight() {
        // 100 TFLOPS == 100% of ref; 1000 GB/s == 100% of ref → score 100.
        let idx = compute_index(Some(&gemm(100.0)), Some(&bw(1000.0)));
        assert!((idx.compute_component - 100.0).abs() < 1e-9);
        assert!((idx.memory_component - 100.0).abs() < 1e-9);
        assert!((idx.score - 100.0).abs() < 1e-9);
    }

    #[test]
    fn index_weights_compute_more_than_memory() {
        // Pure compute at ref vs pure memory at ref: compute must score higher.
        let compute_only = compute_index(Some(&gemm(100.0)), Some(&bw(0.0)));
        let memory_only = compute_index(Some(&gemm(0.0)), Some(&bw(1000.0)));
        assert!(compute_only.score > memory_only.score);
        assert!((compute_only.score - 70.0).abs() < 1e-9);
        assert!((memory_only.score - 30.0).abs() < 1e-9);
    }

    #[test]
    fn index_handles_missing_measurements() {
        let idx = compute_index(None, None);
        assert_eq!(idx.score, 0.0);
        assert!(idx.network_component.is_none());
    }
}
