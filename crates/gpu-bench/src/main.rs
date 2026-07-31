//! `gpu-bench` CLI.
//!
//! ```text
//! gpu-bench list                     # enumerate visible GPUs
//! gpu-bench run                      # auto: native peak + portable wgpu, side by side
//! gpu-bench run --backend wgpu       # force portable wgpu only (apples-to-apples)
//! gpu-bench run --n 4096 --json      # bigger matmul, machine-readable output
//! gpu-bench run --skip-fp16          # fp32 GEMM + bandwidth only
//! ```
//!
//! `run` defaults to `--backend auto`: the best native backend for the box —
//! cuBLAS if built with `--features cuda`, else native Metal on macOS, else
//! portable wgpu. When auto lands on a native backend it *also* measures the
//! portable wgpu path and prints both columns, since the portable number is the
//! cross-vendor ranking metric. An explicit `--backend X` measures only X.

use std::time::Duration;

use anyhow::Context;
use clap::{Args, Parser, Subcommand, ValueEnum};
use comfy_table::{Cell, Table};
use indicatif::{ProgressBar, ProgressStyle};

use gpu_bench::{
    probe_network, run_suite, Backend, BandwidthResult, GemmResult, NetConfig, NetworkResult,
    Report, SuiteConfig, WgpuBackend,
};

#[derive(Parser)]
#[command(name = "gpu-bench", version, about = "Vendor-agnostic GPU benchmark")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List every GPU adapter visible to the portable backend.
    List {
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Run the measurement suite on the highest-performance GPU.
    Run(RunArgs),
    /// Probe network down/up/latency against a public speedtest peer.
    ///
    /// Makes outbound HTTPS requests to speed.cloudflare.com.
    Net(NetArgs),
}

#[derive(Args)]
struct RunArgs {
    /// Square matrix dimension for GEMM (an n×n · n×n multiply).
    #[arg(long, default_value_t = 2048)]
    n: u32,
    /// Timed iterations.
    #[arg(long, default_value_t = 50)]
    iters: u32,
    /// Untimed warmup iterations.
    #[arg(long, default_value_t = 10)]
    warmup: u32,
    /// Bandwidth test size, in millions of f32 elements per buffer.
    #[arg(long, default_value_t = 32)]
    bandwidth_m: u64,
    /// Which backend to benchmark.
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    backend: BackendChoice,
    /// Skip the fp16 GEMM (run fp32 + bandwidth only).
    #[arg(long)]
    skip_fp16: bool,
    /// Emit JSON instead of tables.
    #[arg(long)]
    json: bool,
    /// Print backend-resolution details and the explanatory note.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendChoice {
    /// Best backend compiled into this build: cuBLAS if built with
    /// `--features cuda`, else native Metal on macOS, else portable wgpu.
    /// Falls back to wgpu if the native backend fails to initialize.
    Auto,
    /// Portable wgpu (Metal/Vulkan/DX12) — ALU throughput, works everywhere.
    Wgpu,
    /// Native Metal `simdgroup_matrix` — peak Apple-GPU numbers (macOS only).
    Metal,
    /// NVIDIA cuBLAS — peak on an NVIDIA host (build with `--features cuda`).
    Cuda,
}

/// Build a backend, returning it alongside the *concrete* choice that was
/// realized (never `Auto`) so the caller can label the result accurately.
/// `verbose` gates the auto-resolution announcement (see [`make_auto_backend`]).
fn make_backend(
    choice: BackendChoice,
    verbose: bool,
) -> anyhow::Result<(Box<dyn Backend>, BackendChoice)> {
    match choice {
        BackendChoice::Auto => make_auto_backend(verbose),
        BackendChoice::Wgpu => Ok((Box::new(WgpuBackend::new()?), BackendChoice::Wgpu)),
        BackendChoice::Metal => {
            #[cfg(target_os = "macos")]
            {
                Ok((
                    Box::new(gpu_bench::MetalBackend::new()?),
                    BackendChoice::Metal,
                ))
            }
            #[cfg(not(target_os = "macos"))]
            {
                anyhow::bail!("the metal backend is macOS-only")
            }
        }
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok((
                    Box::new(gpu_bench::CudaBackend::new()?),
                    BackendChoice::Cuda,
                ))
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!(
                    "cuda backend not built — rebuild with `--features cuda` on an NVIDIA host"
                )
            }
        }
    }
}

/// Resolve `--backend auto` to the best backend compiled into this build,
/// degrading to portable wgpu if the native one won't initialize. The resolved
/// choice is announced on stderr only under `verbose` (the labeled table columns
/// already show which backend ran); a *fallback* is always announced, since a
/// silent degradation from native to wgpu would otherwise be invisible. stderr
/// keeps `--json` on stdout clean.
fn make_auto_backend(verbose: bool) -> anyhow::Result<(Box<dyn Backend>, BackendChoice)> {
    #[cfg(feature = "cuda")]
    {
        match gpu_bench::CudaBackend::new() {
            Ok(b) => {
                if verbose {
                    eprintln!("auto backend: cuBLAS (native NVIDIA)");
                }
                return Ok((Box::new(b), BackendChoice::Cuda));
            }
            Err(e) => {
                eprintln!("auto backend: cuBLAS unavailable ({e:#}); falling back to wgpu");
            }
        }
    }
    #[cfg(all(target_os = "macos", not(feature = "cuda")))]
    {
        match gpu_bench::MetalBackend::new() {
            Ok(b) => {
                if verbose {
                    eprintln!("auto backend: native Metal (Apple GPU)");
                }
                return Ok((Box::new(b), BackendChoice::Metal));
            }
            Err(e) => {
                eprintln!("auto backend: Metal unavailable ({e:#}); falling back to wgpu");
            }
        }
    }
    if verbose {
        eprintln!("auto backend: portable wgpu");
    }
    Ok((Box::new(WgpuBackend::new()?), BackendChoice::Wgpu))
}

#[derive(Args)]
struct NetArgs {
    /// Download probe size, in megabytes.
    #[arg(long, default_value_t = 25)]
    down_mb: u64,
    /// Upload probe size, in megabytes.
    #[arg(long, default_value_t = 10)]
    up_mb: u64,
    /// Number of latency samples (best is reported).
    #[arg(long, default_value_t = 5)]
    latency_samples: u32,
    /// Emit JSON instead of a table.
    #[arg(long)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::List { json } => list(json),
        Cmd::Run(args) => run(args),
        Cmd::Net(args) => net(args),
    }
}

fn net(args: NetArgs) -> anyhow::Result<()> {
    let cfg = NetConfig {
        down_bytes: args.down_mb * 1_000_000,
        up_bytes: args.up_mb * 1_000_000,
        latency_samples: args.latency_samples,
    };
    let result = probe_network(&cfg).context("network probe")?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    let mut table = Table::new();
    table.set_header(vec![Cell::new("Metric"), Cell::new("Value")]);
    table.add_row(vec![
        Cell::new("Download"),
        Cell::new(format!("{:.1} Mbps", result.down_mbps)),
    ]);
    table.add_row(vec![
        Cell::new("Upload"),
        Cell::new(format!("{:.1} Mbps", result.up_mbps)),
    ]);
    table.add_row(vec![
        Cell::new("Latency"),
        Cell::new(format!("{:.1} ms", result.latency_ms)),
    ]);
    println!("{table}");
    println!("\nPeer: speed.cloudflare.com");
    Ok(())
}

fn list(json: bool) -> anyhow::Result<()> {
    let devices = WgpuBackend::list();
    if json {
        println!("{}", serde_json::to_string_pretty(&devices)?);
        return Ok(());
    }
    if devices.is_empty() {
        println!("No GPU adapters found.");
        return Ok(());
    }
    let mut table = Table::new();
    table.set_header(vec!["Device", "Backend", "Type"]);
    for d in devices {
        table.add_row(vec![d.name, d.backend, d.device_type]);
    }
    println!("{table}");
    Ok(())
}

fn run(args: RunArgs) -> anyhow::Result<()> {
    let (backend, resolved_backend) =
        make_backend(args.backend, args.verbose).context("initializing backend")?;
    // In auto mode, when we land on a native (peak) backend, also measure the
    // portable wgpu path and show both. wgpu is a real measurement in its own
    // right — the cross-vendor ranking number — so the auto default surfaces it
    // next to the peak number instead of hiding it. An explicit `--backend X`
    // means "just that one", so no comparison pass runs there.
    let compare_portable = matches!(args.backend, BackendChoice::Auto)
        && !matches!(resolved_backend, BackendChoice::Wgpu);

    // The primary (native) suite owns the (backend-independent) network probe;
    // the portable comparison suite skips it, so the network is probed once.
    let cfg = SuiteConfig {
        n: args.n,
        iters: args.iters,
        warmup: args.warmup,
        bandwidth_elems: args.bandwidth_m * 1_000_000,
        include_fp16: !args.skip_fp16,
        include_network: true,
    };
    // Animated spinner on stderr while the (multi-second) suite runs; the label
    // tracks the current phase. Cleared before results, so it never pollutes the
    // table or `--json` on stdout.
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(ProgressStyle::with_template("{spinner:.cyan} {msg}").unwrap());
    spinner.enable_steady_tick(Duration::from_millis(80));
    spinner.set_message("benchmarking…");

    // In comparison mode, measure portable wgpu *first*, then the native backend,
    // so run order matches the table's column order (portable, then native).
    let portable = if compare_portable {
        let (wgpu_backend, _) = make_backend(BackendChoice::Wgpu, args.verbose)?;
        let wgpu_cfg = SuiteConfig {
            include_network: false,
            ..cfg.clone()
        };
        Some(
            run_suite(wgpu_backend.as_ref(), &wgpu_cfg, |phase| {
                spinner.set_message(format!("benchmarking (portable wgpu): {phase}…"))
            })
            .context("running portable comparison suite")?,
        )
    } else {
        None
    };

    let native_label = if compare_portable {
        "benchmarking (native)"
    } else {
        "benchmarking"
    };
    let report = run_suite(backend.as_ref(), &cfg, |phase| {
        spinner.set_message(format!("{native_label}: {phase}…"))
    })
    .context("running benchmark suite")?;
    spinner.finish_and_clear();

    if args.json {
        match &portable {
            Some(p) => {
                let out = serde_json::json!({ "native": &report, "portable": p });
                println!("{}", serde_json::to_string_pretty(&out)?);
            }
            None => println!("{}", serde_json::to_string_pretty(&report)?),
        }
        return Ok(());
    }

    match &portable {
        Some(p) => print_comparison(&report, resolved_backend, p, &args),
        None => print_single(&report, resolved_backend, &args),
    }
    Ok(())
}

/// One backend: measurement / value / detail table, plus the note.
fn print_single(report: &Report, resolved: BackendChoice, args: &RunArgs) {
    println!(
        "Device:  {} ({}, {})",
        report.device.name, report.device.backend, report.device.device_type
    );

    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Measurement"),
        Cell::new("Value"),
        Cell::new("Detail"),
    ]);

    // GEMM rows (fp32 always, fp16 when run).
    for gemm in [report.gemm_fp32.as_ref(), report.gemm_fp16.as_ref()]
        .into_iter()
        .flatten()
    {
        table.add_row(vec![
            Cell::new(format!("GEMM {}", gemm.precision.as_str())),
            Cell::new(format!("{:.2} TFLOP/s", gemm.tflops)),
            Cell::new(format!(
                "n={}, {} iters, {:.3}s{}",
                gemm.n,
                gemm.iterations,
                gemm.seconds,
                if gemm.verified {
                    ", verified"
                } else {
                    " — UNVERIFIED"
                }
            )),
        ]);
    }
    if report.gemm_fp16.is_none() && !args.skip_fp16 {
        table.add_row(vec![
            Cell::new("GEMM fp16"),
            Cell::new("n/a"),
            Cell::new("not supported on this device"),
        ]);
    }

    if let Some(bw) = &report.bandwidth {
        table.add_row(vec![
            Cell::new("Memory bandwidth"),
            Cell::new(format!("{:.1} GB/s", bw.gb_per_s)),
            Cell::new(format!(
                "{:.0} MB/iter, {} iters, {:.3}s",
                bw.bytes_per_iter as f64 / 1e6,
                bw.iterations,
                bw.seconds
            )),
        ]);
    }

    match &report.network {
        Some(nw) => table.add_row(vec![
            Cell::new("Network"),
            Cell::new(format!("{:.0}↓ / {:.0}↑ Mbps", nw.down_mbps, nw.up_mbps)),
            Cell::new(format!(
                "latency {:.1} ms (speed.cloudflare.com)",
                nw.latency_ms
            )),
        ]),
        None => table.add_row(vec![
            Cell::new("Network"),
            Cell::new("n/a"),
            Cell::new("probe failed (offline?)"),
        ]),
    };

    println!("{table}");
    println!("\nNote: {}", backend_note(resolved));
}

/// Auto mode: portable wgpu and the native (peak) backend side by side. GEMM and
/// bandwidth are per-backend columns; the network probe is backend-independent,
/// so it prints on its own line below the table rather than in a column.
fn print_comparison(native: &Report, resolved: BackendChoice, portable: &Report, args: &RunArgs) {
    println!(
        "Device:  {} ({}, {})",
        native.device.name, native.device.backend, native.device.device_type
    );

    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Measurement"),
        Cell::new(backend_label(BackendChoice::Wgpu)),
        Cell::new(backend_label(resolved)),
    ]);
    table.add_row(vec![
        Cell::new("GEMM fp32"),
        Cell::new(tflops_cell(portable.gemm_fp32.as_ref())),
        Cell::new(tflops_cell(native.gemm_fp32.as_ref())),
    ]);
    if !args.skip_fp16 {
        table.add_row(vec![
            Cell::new("GEMM fp16"),
            Cell::new(tflops_cell(portable.gemm_fp16.as_ref())),
            Cell::new(tflops_cell(native.gemm_fp16.as_ref())),
        ]);
    }
    table.add_row(vec![
        Cell::new("Memory bandwidth"),
        Cell::new(bandwidth_cell(portable.bandwidth.as_ref())),
        Cell::new(bandwidth_cell(native.bandwidth.as_ref())),
    ]);

    println!("{table}");
    // Network is machine-wide, not per-backend, so it stands on its own — a
    // single-cell table below, framed like the one above but with no columns.
    let mut net = Table::new();
    net.add_row(vec![Cell::new(network_summary(native.network.as_ref()))]);
    println!("{net}");
    if args.verbose {
        println!(
            "\nNote: {} = cross-vendor ALU ranking; {} = {}. n={}, {} iters.",
            backend_label(BackendChoice::Wgpu),
            backend_label(resolved),
            native_peak_desc(resolved),
            args.n,
            args.iters,
        );
    }
}

/// Column/label for a concrete backend (never `Auto`).
fn backend_label(c: BackendChoice) -> &'static str {
    match c {
        BackendChoice::Metal => "Native (Metal)",
        BackendChoice::Cuda => "Native (cuBLAS)",
        BackendChoice::Wgpu => "Portable (wgpu)",
        BackendChoice::Auto => "auto",
    }
}

/// How the native backend reaches its peak — for the comparison note.
fn native_peak_desc(c: BackendChoice) -> &'static str {
    match c {
        BackendChoice::Metal => "peak via Apple matrix units (simdgroup_matrix)",
        BackendChoice::Cuda => "peak via NVIDIA tensor cores (cuBLAS)",
        _ => "native peak",
    }
}

/// One-line note for a single-backend run.
fn backend_note(resolved: BackendChoice) -> &'static str {
    match resolved {
        BackendChoice::Wgpu => {
            "portable ALU throughput (no tensor cores) — good for ranking, below vendor peak."
        }
        BackendChoice::Metal => {
            "native Metal simdgroup_matrix (Apple matrix units) — a basic kernel: \
             above the portable path, below a fully-tuned library."
        }
        BackendChoice::Cuda => "cuBLAS on NVIDIA tensor cores.",
        // make_backend never returns Auto — it resolves to a concrete backend.
        BackendChoice::Auto => unreachable!("resolved backend is never Auto"),
    }
}

fn tflops_cell(g: Option<&GemmResult>) -> String {
    match g {
        Some(g) if g.verified => format!("{:.2} TFLOP/s", g.tflops),
        Some(g) => format!("{:.2} TFLOP/s (unverified)", g.tflops),
        None => "n/a".to_string(),
    }
}

fn bandwidth_cell(b: Option<&BandwidthResult>) -> String {
    match b {
        Some(b) => format!("{:.1} GB/s", b.gb_per_s),
        None => "n/a".to_string(),
    }
}

/// Machine-wide network summary as one line, for the comparison table's single
/// (non-split) network cell.
fn network_summary(nw: Option<&NetworkResult>) -> String {
    match nw {
        Some(nw) => format!(
            "Network: {:.0}↓ / {:.0}↑ Mbps, latency {:.1} ms",
            nw.down_mbps, nw.up_mbps, nw.latency_ms
        ),
        None => "Network: probe failed (offline?)".to_string(),
    }
}
