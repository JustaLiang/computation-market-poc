//! `gpu-bench` CLI.
//!
//! ```text
//! gpu-bench list                     # enumerate visible GPUs
//! gpu-bench run                      # GEMM + bandwidth on the best GPU
//! gpu-bench run --n 4096 --json      # bigger matmul, machine-readable output
//! ```

use anyhow::Context;
use clap::{Args, Parser, Subcommand, ValueEnum};
use comfy_table::{Cell, Table};
use serde::Serialize;

use gpu_bench::{Backend, Precision, WgpuBackend};

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
    /// Run the benchmark suite on the highest-performance GPU.
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
    /// GEMM precision.
    #[arg(long, value_enum, default_value_t = Prec::F32)]
    precision: Prec,
    /// Bandwidth test size, in millions of f32 elements per buffer.
    #[arg(long, default_value_t = 32)]
    bandwidth_m: u64,
    /// Emit JSON instead of tables.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Prec {
    F32,
    F16,
}

impl From<Prec> for Precision {
    fn from(p: Prec) -> Self {
        match p {
            Prec::F32 => Precision::F32,
            Prec::F16 => Precision::F16,
        }
    }
}

/// Combined machine-readable result for `run --json`.
#[derive(Serialize)]
struct RunReport {
    gemm: gpu_bench::GemmResult,
    bandwidth: gpu_bench::BandwidthResult,
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
    let info = backend.device_info();

    let gemm = backend
        .gemm(args.n, args.precision.into(), args.warmup, args.iters)
        .context("GEMM benchmark")?;
    let bandwidth = backend
        .bandwidth(args.bandwidth_m * 1_000_000, args.warmup, args.iters)
        .context("bandwidth benchmark")?;

    if args.json {
        let report = RunReport { gemm, bandwidth };
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!(
        "Device:  {} ({}, {})",
        info.name, info.backend, info.device_type
    );
    if !gemm.verified {
        println!("WARNING: GEMM result failed verification — throughput is unreliable.");
    }

    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Metric"),
        Cell::new("Value"),
        Cell::new("Detail"),
    ]);
    table.add_row(vec![
        Cell::new(format!("GEMM {}", gemm.precision.as_str())),
        Cell::new(format!("{:.2} TFLOP/s", gemm.tflops)),
        Cell::new(format!(
            "n={}, {} iters, {:.3}s{}",
            gemm.n,
            gemm.iterations,
            gemm.seconds,
            if gemm.verified { ", verified" } else { "" }
        )),
    ]);
    table.add_row(vec![
        Cell::new("Memory bandwidth"),
        Cell::new(format!("{:.1} GB/s", bandwidth.gb_per_s)),
        Cell::new(format!(
            "{:.0} MB/iter, {} iters, {:.3}s",
            bandwidth.bytes_per_iter as f64 / 1e6,
            bandwidth.iterations,
            bandwidth.seconds
        )),
    ]);
    println!("{table}");
    println!(
        "\nNote: portable ALU throughput (no tensor cores) — good for ranking, below vendor peak."
    );
    Ok(())
}
