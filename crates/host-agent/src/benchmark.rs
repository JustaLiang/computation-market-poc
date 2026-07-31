//! GPU inventory, `dlperf` score, and hardware fingerprint (SPEC §6).
//!
//! Inventory comes from **NVML** (`nvml-wrapper`), not from parsing
//! `nvidia-smi`: NVML hands us name, VRAM, UUID, and PCI bus id directly.
//!
//! `dlperf` is a scalar performance estimate normalized so a single RTX 4090
//! lands near 42, scaled linearly by GPU count. The intended measurement is
//! fp16 GEMM throughput (8192³, 3 warmup + 20 timed iterations); when no CUDA
//! runtime is available we fall back to a GPU-name lookup table, exactly as SPEC
//! §6 permits. Its purpose is not precision — it is to make hardware
//! misrepresentation expensive (BACKGROUND §4.2).

use sha2::{Digest, Sha256};

/// One physical GPU as reported by NVML.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    pub name: String,
    pub uuid: String,
    pub pci_bus_id: String,
    /// Per-GPU VRAM in megabytes.
    pub vram_mb: i64,
}

/// The result of probing the host's accelerators.
#[derive(Debug, Clone)]
pub struct Inventory {
    pub gpus: Vec<GpuInfo>,
    pub dlperf: f64,
    pub hw_fingerprint: String,
}

impl Inventory {
    pub fn gpu_count(&self) -> i32 {
        self.gpus.len() as i32
    }

    /// Model name used for the offer listing. Assumes a homogeneous host — a
    /// mixed-GPU box is out of scope for the POC register schema (SPEC §3).
    pub fn gpu_name(&self) -> &str {
        self.gpus
            .first()
            .map(|g| g.name.as_str())
            .unwrap_or("unknown")
    }

    /// Per-GPU VRAM in MB (the `machines.vram_mb` column is per GPU).
    pub fn vram_mb(&self) -> i64 {
        self.gpus.first().map(|g| g.vram_mb).unwrap_or(0)
    }
}

/// Probe the host: enumerate GPUs, score them, and fingerprint them.
///
/// Errors when no NVIDIA GPU is visible — a machine with nothing to rent must
/// not register as an offer.
pub fn probe() -> anyhow::Result<Inventory> {
    let gpus = inventory()?;
    if gpus.is_empty() {
        anyhow::bail!("no NVIDIA GPU detected — nothing to offer");
    }
    let hw_fingerprint = fingerprint(&gpus);
    let dlperf = dlperf(&gpus);
    Ok(Inventory {
        gpus,
        dlperf,
        hw_fingerprint,
    })
}

#[cfg(target_os = "linux")]
fn inventory() -> anyhow::Result<Vec<GpuInfo>> {
    use nvml_wrapper::Nvml;

    let nvml = Nvml::init()
        .map_err(|e| anyhow::anyhow!("NVML init failed (is the NVIDIA driver installed?): {e}"))?;
    let count = nvml.device_count()?;
    let mut gpus = Vec::with_capacity(count as usize);
    for i in 0..count {
        let device = nvml.device_by_index(i)?;
        let mem = device.memory_info()?;
        gpus.push(GpuInfo {
            name: device.name()?,
            uuid: device.uuid()?,
            pci_bus_id: device.pci_info()?.bus_id,
            vram_mb: (mem.total / (1024 * 1024)) as i64,
        });
    }
    Ok(gpus)
}

/// macOS (Apple Silicon): inventory via `system_profiler` (no NVML). Apple GPUs
/// have no NVML UUID/PCI id, so we synthesize a stable, swap-sensitive
/// fingerprint id from the machine serial + GPU core count, and report unified
/// memory as the VRAM figure.
#[cfg(target_os = "macos")]
fn inventory() -> anyhow::Result<Vec<GpuInfo>> {
    let out = std::process::Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .map_err(|e| anyhow::anyhow!("running system_profiler: {e}"))?;
    anyhow::ensure!(
        out.status.success(),
        "system_profiler SPDisplaysDataType failed"
    );

    let value: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let gpu = value["SPDisplaysDataType"]
        .as_array()
        .and_then(|a| a.first())
        .ok_or_else(|| anyhow::anyhow!("no GPU reported by system_profiler"))?;

    let name = gpu["_name"].as_str().unwrap_or("Apple GPU").to_string();
    let cores = gpu["sppci_cores"]
        .as_str()
        .and_then(|s| s.trim().parse::<i64>().ok());
    let id = mac_serial().unwrap_or_else(|| name.clone());

    Ok(vec![GpuInfo {
        name,
        uuid: format!("apple-gpu-{id}"),
        pci_bus_id: cores.map(|c| format!("{c}-core")).unwrap_or_default(),
        vram_mb: apple_unified_memory_mb(),
    }])
}

#[cfg(target_os = "macos")]
fn mac_serial() -> Option<String> {
    let out = std::process::Command::new("system_profiler")
        .args(["SPHardwareDataType", "-json"])
        .output()
        .ok()?;
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    value["SPHardwareDataType"][0]["serial_number"]
        .as_str()
        .map(str::to_string)
}

/// Apple Silicon has unified memory; report total system RAM as the VRAM proxy.
#[cfg(target_os = "macos")]
fn apple_unified_memory_mb() -> i64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    (sys.total_memory() / (1024 * 1024)) as i64
}

/// Off-host fallback so the agent still builds on other dev platforms.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn inventory() -> anyhow::Result<Vec<GpuInfo>> {
    anyhow::bail!("GPU inventory is available only on Linux (NVML) and macOS hosts");
}

/// `hw_fingerprint`: SHA-256 over the **sorted** set of
/// `(gpu_name, gpu_uuid, pci_bus_id)` tuples, hex-encoded. Sorting makes it
/// independent of NVML enumeration order; it changes if hardware is swapped
/// (SPEC §6, roadmap item 1 acts on it).
fn fingerprint(gpus: &[GpuInfo]) -> String {
    let mut tuples: Vec<String> = gpus
        .iter()
        .map(|g| format!("{}|{}|{}", g.name, g.uuid, g.pci_bus_id))
        .collect();
    tuples.sort();

    let mut hasher = Sha256::new();
    for t in &tuples {
        hasher.update(t.as_bytes());
        hasher.update([0u8]); // unambiguous separator between records
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Aggregate `dlperf`: measured fp16 GEMM if a CUDA runtime is available,
/// otherwise the sum of per-GPU table lookups (linear in GPU count).
fn dlperf(gpus: &[GpuInfo]) -> f64 {
    if let Some(measured) = gemm_score() {
        return measured;
    }
    gpus.iter().map(|g| lookup_dlperf(&g.name)).sum()
}

/// Measured fp16 GEMM throughput, normalized to the 4090≈42 scale.
///
/// Returns `None` when no CUDA runtime is available — which is every build
/// target here, since lighting this up requires a CUDA GEMM (e.g. `cudarc` +
/// cuBLAS) plus real hardware. This is the single integration point deferred to
/// a CUDA-capable host; SPEC §6 explicitly blesses the lookup fallback below in
/// its absence. (A single measurement at registration does not meet §6's goal
/// anyway; randomized re-benchmarking is roadmap item 1.)
fn gemm_score() -> Option<f64> {
    None
}

/// Per-GPU `dlperf` from a name lookup. Rough estimates anchored at RTX 4090 =
/// 42 and loosely proportional to dense fp16 tensor throughput — enough to rank
/// hardware and to make a fake expensive, not to be exact. Most specific
/// substrings first (e.g. "3090 ti" before "3090").
fn lookup_dlperf(name: &str) -> f64 {
    const TABLE: &[(&str, f64)] = &[
        // Datacenter
        ("h200", 320.0),
        ("h100", 300.0),
        ("a100", 160.0),
        ("l40s", 95.0),
        ("l40", 90.0),
        ("a40", 74.0),
        ("a30", 55.0),
        ("a10g", 32.0),
        ("a10", 30.0),
        ("l4", 24.0),
        ("v100", 28.0),
        ("t4", 16.0),
        // Workstation
        ("rtx 6000 ada", 90.0),
        ("rtx a6000", 78.0),
        ("rtx a5000", 55.0),
        ("rtx a4000", 40.0),
        // GeForce (specific before generic)
        ("4090", 42.0),
        ("4080", 30.0),
        ("4070 ti", 26.0),
        ("4070", 20.0),
        ("3090 ti", 31.0),
        ("3090", 27.0),
        ("3080", 24.0),
        ("3070", 17.0),
        ("2080 ti", 16.0),
    ];

    let lname = name.to_lowercase();
    // Apple Silicon uses its own table — and this guard avoids false matches
    // with old NVIDIA Tesla "M40"/"M2090" names.
    if lname.contains("apple") {
        return apple_dlperf(&lname);
    }
    for (needle, score) in TABLE {
        if lname.contains(needle) {
            return *score;
        }
    }
    // Unknown card: a conservative non-zero estimate so it still ranks, but low
    // enough that misrepresenting it as premium hardware is unattractive.
    tracing::warn!(gpu = %name, "unknown GPU model, using fallback dlperf");
    18.0
}

/// Apple Silicon `dlperf`, on the same 4090≈42 scale — integrated GPUs land well
/// below datacenter parts. Rough estimates; most specific variant first (so
/// "m4 max" is not captured by "m4"). Only reached for names containing "apple".
fn apple_dlperf(lname: &str) -> f64 {
    const TABLE: &[(&str, f64)] = &[
        ("m3 ultra", 13.0),
        ("m2 ultra", 11.0),
        ("m1 ultra", 10.0),
        ("m4 max", 9.0),
        ("m3 max", 8.0),
        ("m2 max", 7.0),
        ("m1 max", 6.0),
        ("m4 pro", 6.0),
        ("m3 pro", 5.0),
        ("m2 pro", 4.5),
        ("m1 pro", 4.0),
        ("m4", 4.0),
        ("m3", 3.0),
        ("m2", 2.5),
        ("m1", 2.0),
    ];
    for (needle, score) in TABLE {
        if lname.contains(needle) {
            return *score;
        }
    }
    3.0 // unknown Apple GPU
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, uuid: &str, pci: &str) -> GpuInfo {
        GpuInfo {
            name: name.to_string(),
            uuid: uuid.to_string(),
            pci_bus_id: pci.to_string(),
            vram_mb: 24564,
        }
    }

    #[test]
    fn fingerprint_is_order_independent() {
        let a = gpu("RTX 4090", "GPU-aaa", "00000000:01:00.0");
        let b = gpu("RTX 4090", "GPU-bbb", "00000000:02:00.0");
        assert_eq!(fingerprint(&[a.clone(), b.clone()]), fingerprint(&[b, a]));
    }

    #[test]
    fn fingerprint_changes_when_hardware_swapped() {
        let base = [gpu("RTX 4090", "GPU-aaa", "00000000:01:00.0")];
        let swapped = [gpu("RTX 4090", "GPU-ccc", "00000000:01:00.0")];
        assert_ne!(fingerprint(&base), fingerprint(&swapped));
    }

    #[test]
    fn single_4090_anchors_near_42() {
        let one = [gpu("NVIDIA GeForce RTX 4090", "GPU-1", "00000000:01:00.0")];
        assert_eq!(dlperf(&one), 42.0);
    }

    #[test]
    fn dlperf_scales_linearly_with_count() {
        let two = [
            gpu("NVIDIA GeForce RTX 4090", "GPU-1", "00000000:01:00.0"),
            gpu("NVIDIA GeForce RTX 4090", "GPU-2", "00000000:02:00.0"),
        ];
        assert_eq!(dlperf(&two), 84.0);
    }

    #[test]
    fn h100_outranks_4090() {
        assert!(lookup_dlperf("NVIDIA H100 80GB HBM3") > lookup_dlperf("NVIDIA GeForce RTX 4090"));
    }

    #[test]
    fn specific_substring_wins_over_generic() {
        // "3090 ti" must not be captured by the "3090" row.
        assert_eq!(lookup_dlperf("NVIDIA GeForce RTX 3090 Ti"), 31.0);
        assert_eq!(lookup_dlperf("NVIDIA GeForce RTX 3090"), 27.0);
    }

    #[test]
    fn unknown_gpu_gets_conservative_fallback() {
        assert_eq!(lookup_dlperf("Some Future GPU 9999"), 18.0);
    }

    #[test]
    fn apple_silicon_uses_its_own_table() {
        // "M4 Max" must not be captured by the bare "m4" row.
        assert_eq!(lookup_dlperf("Apple M4 Max"), 9.0);
        assert_eq!(lookup_dlperf("Apple M4"), 4.0);
        // An Apple GPU ranks below a 4090 and never false-matches the NVIDIA table.
        assert!(lookup_dlperf("Apple M1") < lookup_dlperf("NVIDIA GeForce RTX 4090"));
    }

    #[test]
    fn gemm_measurement_absent_without_cuda() {
        assert_eq!(gemm_score(), None);
    }
}
