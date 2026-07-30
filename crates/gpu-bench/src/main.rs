//! `gpu-bench` CLI.
//!
//! ```text
//! gpu-bench list                     # enumerate visible GPUs
//! gpu-bench run                      # full suite (fp32 + fp16 GEMM + bandwidth)
//! gpu-bench run --n 4096 --json      # bigger matmul, machine-readable output
//! gpu-bench run --skip-fp16          # fp32 GEMM + bandwidth only
//! ```

use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use comfy_table::{Cell, Table};

use gpu_bench::{run_suite, SuiteConfig, WgpuBackend};

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
    /// Skip the fp16 GEMM (run fp32 + bandwidth only).
    #[arg(long)]
    skip_fp16: bool,
    /// Emit JSON instead of tables.
    #[arg(long)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::List { json } => list(json),
        Cmd::Run(args) => run(args),
    }
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
    let backend = WgpuBackend::new().context("initializing GPU backend")?;
    let cfg = SuiteConfig {
        n: args.n,
        iters: args.iters,
        warmup: args.warmup,
        bandwidth_elems: args.bandwidth_m * 1_000_000,
        include_fp16: !args.skip_fp16,
    };
    let report = run_suite(&backend, &cfg).context("running benchmark suite")?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

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

    table.add_row(vec![
        Cell::new("Network"),
        Cell::new("reserved"),
        Cell::new("not measured (needs a peer); reserved for the index"),
    ]);

    table.add_row(vec![
        Cell::new("Compute index"),
        Cell::new(format!("{:.1}", report.index.score)),
        Cell::new(format!(
            "compute {:.1} · memory {:.1} ({})",
            report.index.compute_component, report.index.memory_component, report.index.note
        )),
    ]);

    println!("{table}");
    println!(
        "\nNote: portable ALU throughput (no tensor cores) — good for ranking, below vendor peak.\
         \n      Compute index is a provisional v0; references are placeholders, not calibrated."
    );
    Ok(())
}
