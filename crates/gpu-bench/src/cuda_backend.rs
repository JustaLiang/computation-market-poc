//! cuBLAS backend (NVIDIA) — peak fp16/fp32 GEMM, behind `feature = "cuda"`.
//!
//! Built via `cudarc`. Its build script needs the CUDA toolkit headers, so this
//! feature compiles **only on a machine with CUDA installed** (an NVIDIA host);
//! `dynamic-linking` then loads the CUDA libraries at runtime. Like MLX, cuBLAS
//! reaches the GPU's real tensor-core throughput, far above the portable WGSL
//! path. GEMM only — the suite treats bandwidth as best-effort, so this backend
//! leaves the bandwidth number to the portable backend.
//!
//! NOTE: written against the cudarc/cuBLAS API but **not compile-checked here**
//! (no CUDA hardware) — validate on an NVIDIA host.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::CudaDevice;

use crate::backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, Precision};

pub struct CudaBackend {
    dev: Arc<CudaDevice>,
    blas: CudaBlas,
    name: String,
}

impl CudaBackend {
    pub fn new() -> anyhow::Result<Self> {
        let dev = CudaDevice::new(0).context("opening CUDA device 0")?;
        let blas = CudaBlas::new(dev.clone()).context("creating a cuBLAS handle")?;
        let name = format!("NVIDIA GPU (cuda:{})", dev.ordinal());
        Ok(Self { dev, blas, name })
    }

    /// Column-major square GEMM config: C(n×n) = A·B with no transpose.
    fn config<T: Copy>(n: i32, alpha: T, beta: T) -> GemmConfig<T> {
        GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n,
            n,
            k: n,
            alpha,
            lda: n,
            ldb: n,
            beta,
            ldc: n,
        }
    }

    fn gemm_f32(&self, n: u32, warmup: u32, iters: u32) -> anyhow::Result<(f64, bool)> {
        let len = (n as usize) * (n as usize);
        let a = self.dev.htod_copy(vec![1.0f32; len])?;
        let b = self.dev.htod_copy(vec![1.0f32; len])?;
        let mut c = self.dev.alloc_zeros::<f32>(len)?;
        let cfg = Self::config(n as i32, 1.0f32, 0.0f32);

        for _ in 0..warmup {
            unsafe { self.blas.gemm(cfg, &a, &b, &mut c)? };
        }
        self.dev.synchronize()?;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe { self.blas.gemm(cfg, &a, &b, &mut c)? };
        }
        self.dev.synchronize()?;
        let secs = start.elapsed().as_secs_f64();

        let host = self.dev.dtoh_sync_copy(&c)?;
        let verified = (host[0] - n as f32).abs() < (n as f32 * 1e-3).max(1.0);
        Ok((secs, verified))
    }

    fn gemm_f16(&self, n: u32, warmup: u32, iters: u32) -> anyhow::Result<(f64, bool)> {
        use half::f16;
        let len = (n as usize) * (n as usize);
        let a = self.dev.htod_copy(vec![f16::ONE; len])?;
        let b = self.dev.htod_copy(vec![f16::ONE; len])?;
        let mut c = self.dev.alloc_zeros::<f16>(len)?;
        let cfg = Self::config(n as i32, f16::ONE, f16::ZERO);

        for _ in 0..warmup {
            unsafe { self.blas.gemm(cfg, &a, &b, &mut c)? };
        }
        self.dev.synchronize()?;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe { self.blas.gemm(cfg, &a, &b, &mut c)? };
        }
        self.dev.synchronize()?;
        let secs = start.elapsed().as_secs_f64();

        let host = self.dev.dtoh_sync_copy(&c)?;
        let verified = (host[0].to_f32() - n as f32).abs() < (n as f32 * 1e-3).max(1.0);
        Ok((secs, verified))
    }
}

impl Backend for CudaBackend {
    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.name.clone(),
            backend: "cuBLAS".to_string(),
            device_type: "DiscreteGpu".to_string(),
        }
    }

    fn gemm(
        &self,
        n: u32,
        precision: Precision,
        warmup: u32,
        iterations: u32,
    ) -> anyhow::Result<GemmResult> {
        anyhow::ensure!(n > 0, "n must be > 0");
        let (seconds, verified) = match precision {
            Precision::F32 => self.gemm_f32(n, warmup, iterations)?,
            Precision::F16 => self.gemm_f16(n, warmup, iterations)?,
        };
        let tflops = 2.0 * (n as f64).powi(3) * iterations as f64 / seconds / 1e12;
        Ok(GemmResult {
            device: self.name.clone(),
            backend: "cuBLAS".to_string(),
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
        _elements: u64,
        _warmup: u32,
        _iterations: u32,
    ) -> anyhow::Result<BandwidthResult> {
        anyhow::bail!("cuBLAS backend measures GEMM only; use the portable backend for bandwidth")
    }
}
