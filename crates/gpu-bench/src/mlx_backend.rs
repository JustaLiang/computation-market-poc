//! MLX backend (Apple Silicon) — peak numbers via Apple's matrix hardware.
//!
//! MLX's matmul reaches near the GPU's true fp16 peak, unlike the portable WGSL
//! kernel in [`crate::wgpu_backend`]. We drive it through a bundled Python script
//! run by a configured interpreter (`VGPU_MLX_PYTHON`, default `python3`) that
//! must have `mlx` installed. Shelling out avoids the native `mlx-rs` binding
//! (which builds MLX from source and needs `cmake`); the script is the only code
//! that runs, and it prints one JSON line per measurement.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;

use crate::backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, Precision};

/// The bundled MLX micro-benchmark. `gemm`/`bandwidth`/`device` modes, JSON out.
const MLX_BENCH: &str = r#"
import sys, json, time
import mlx.core as mx

def gemm(n, precision, warmup, iters):
    dt = mx.float16 if precision == "fp16" else mx.float32
    a = mx.ones((n, n), dtype=dt)
    b = mx.ones((n, n), dtype=dt)
    for _ in range(warmup):
        mx.eval(a @ b)
    start = time.time()
    c = a @ b
    for _ in range(iters):
        c = a @ b
        mx.eval(c)
    secs = time.time() - start
    val = float(c[0, 0].item())           # all-ones N×N → every entry == n
    verified = abs(val - n) < max(1.0, n * 1e-3)
    return {"tflops": 2 * n**3 * iters / secs / 1e12, "seconds": secs, "verified": verified}

def bandwidth(elems, warmup, iters):
    a = mx.ones((elems,), dtype=mx.float32)
    b = mx.full((elems,), 2.0, dtype=mx.float32)
    for _ in range(warmup):
        mx.eval(a + b)
    start = time.time()
    for _ in range(iters):
        mx.eval(a + b)
    secs = time.time() - start
    return {"gb_per_s": elems * 4 * 3 * iters / secs / 1e9, "seconds": secs}  # 2 read + 1 write

mode = sys.argv[1]
if mode == "gemm":
    print(json.dumps(gemm(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))))
elif mode == "bandwidth":
    print(json.dumps(bandwidth(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))))
"#;

pub struct MlxBackend {
    python: String,
    script: PathBuf,
    device_name: String,
}

impl MlxBackend {
    pub fn new() -> anyhow::Result<Self> {
        let python = std::env::var("VGPU_MLX_PYTHON").unwrap_or_else(|_| "python3".to_string());

        // Fail early with a clear message if MLX isn't importable.
        let ok = Command::new(&python)
            .args(["-c", "import mlx.core"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        anyhow::ensure!(
            ok,
            "MLX not available: `import mlx.core` failed for {python}. Install it \
             (`pip install mlx`) or point VGPU_MLX_PYTHON at a Python that has it."
        );

        let script = std::env::temp_dir().join("gpu-bench-mlx.py");
        std::fs::File::create(&script)
            .with_context(|| format!("writing MLX script to {}", script.display()))?
            .write_all(MLX_BENCH.as_bytes())?;

        // Chip name (macOS); harmless fallback elsewhere.
        let device_name = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Apple GPU".to_string());

        Ok(Self {
            python,
            script,
            device_name,
        })
    }

    /// Run the script in one mode and parse its JSON line.
    fn run(&self, args: &[&str]) -> anyhow::Result<serde_json::Value> {
        let out = Command::new(&self.python)
            .arg(&self.script)
            .args(args)
            .output()
            .with_context(|| format!("running MLX script via {}", self.python))?;
        anyhow::ensure!(
            out.status.success(),
            "MLX script failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
        serde_json::from_slice(&out.stdout).with_context(|| {
            format!(
                "parsing MLX output: {}",
                String::from_utf8_lossy(&out.stdout)
            )
        })
    }
}

impl Backend for MlxBackend {
    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device_name.clone(),
            backend: "MLX".to_string(),
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
        let v = self.run(&[
            "gemm",
            &n.to_string(),
            precision.as_str(),
            &warmup.to_string(),
            &iterations.to_string(),
        ])?;
        Ok(GemmResult {
            device: self.device_name.clone(),
            backend: "MLX".to_string(),
            precision,
            n,
            iterations,
            seconds: v["seconds"].as_f64().unwrap_or(0.0),
            tflops: v["tflops"].as_f64().unwrap_or(0.0),
            verified: v["verified"].as_bool().unwrap_or(false),
        })
    }

    fn bandwidth(
        &self,
        elements: u64,
        warmup: u32,
        iterations: u32,
    ) -> anyhow::Result<BandwidthResult> {
        let v = self.run(&[
            "bandwidth",
            &elements.to_string(),
            &warmup.to_string(),
            &iterations.to_string(),
        ])?;
        Ok(BandwidthResult {
            device: self.device_name.clone(),
            backend: "MLX".to_string(),
            bytes_per_iter: elements * 4 * 3,
            iterations,
            seconds: v["seconds"].as_f64().unwrap_or(0.0),
            gb_per_s: v["gb_per_s"].as_f64().unwrap_or(0.0),
        })
    }
}
