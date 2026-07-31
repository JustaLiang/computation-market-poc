//! `vgpu` — the tenant CLI ("virtual GPU": rent GPU compute and use it as if it
//! were your own box).
//!
//! A thin client over the control plane's tenant API. Per CLAUDE.md this is the
//! one place BTC-denominated formatting lives; everywhere else money is bare
//! integer satoshis.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context as _;
use clap::{Args, Parser, Subcommand};
use comfy_table::{Cell, Table};
use serde::de::DeserializeOwned;
use serde::Serialize;
use vgpu_core::api::{
    AccountResponse, CreateAccountResponse, CreateRentalRequest, CreateRentalResponse,
    DepositRequest, DepositResponse, HealthResponse, OffersResponse, RentalResponse,
};
use vgpu_core::model::is_http_status_image;
use vgpu_core::model::RentalKind;

#[derive(Parser)]
#[command(
    name = "vgpu",
    version,
    about = "Tenant CLI for the GPU rental marketplace"
)]
struct Cli {
    /// Control plane base URL.
    #[arg(
        long,
        env = "VGPU_CONTROL_PLANE",
        default_value = "http://127.0.0.1:8080",
        global = true
    )]
    control_plane: String,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Check the control plane is up.
    Health,
    /// List available machines.
    Offers(OffersArgs),
    /// Create a new tenant account.
    CreateAccount,
    /// Show an account's balance and runway.
    Account { account_id: String },
    /// Get a Lightning invoice to credit an account.
    Deposit { account_id: String, sats: i64 },
    /// Rent a machine.
    Rent(RentArgs),
    /// Show a rental (status, ssh command).
    Rental { rental_id: i64 },
    /// Stop (evict) a rental.
    Stop { rental_id: i64 },
}

#[derive(Args)]
struct OffersArgs {
    /// Filter by GPU name substring.
    #[arg(long)]
    gpu: Option<String>,
    #[arg(long)]
    min_vram_mb: Option<i64>,
    #[arg(long)]
    min_gpu_count: Option<i64>,
    #[arg(long)]
    max_rate: Option<i64>,
    /// Sort order.
    #[arg(long, value_parser = ["value", "rate", "dlperf"])]
    sort: Option<String>,
    #[arg(long)]
    limit: Option<i64>,
}

#[derive(Args)]
struct RentArgs {
    #[arg(long)]
    machine_id: i64,
    #[arg(long)]
    account_id: String,
    #[arg(long)]
    image: String,
    /// SSH public key literal (mutually exclusive with --ssh-key-file).
    #[arg(long, conflicts_with = "ssh_key_file")]
    ssh_key: Option<String>,
    /// Path to an SSH public key file.
    #[arg(long, conflicts_with = "ssh_key")]
    ssh_key_file: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let client = Client::new(cli.control_plane)?;

    match cli.cmd {
        Cmd::Health => health(&client),
        Cmd::Offers(args) => offers(&client, args),
        Cmd::CreateAccount => create_account(&client),
        Cmd::Account { account_id } => account(&client, &account_id),
        Cmd::Deposit { account_id, sats } => deposit(&client, &account_id, sats),
        Cmd::Rent(args) => rent(&client, args),
        Cmd::Rental { rental_id } => rental(&client, rental_id),
        Cmd::Stop { rental_id } => stop(&client, rental_id),
    }
}

// --- HTTP client ----------------------------------------------------------

struct Client {
    http: reqwest::blocking::Client,
    base: String,
}

impl Client {
    fn new(base: String) -> anyhow::Result<Self> {
        Ok(Self {
            http: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()?,
            base: base.trim_end_matches('/').to_string(),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    fn get<T: DeserializeOwned>(&self, path: &str) -> anyhow::Result<T> {
        parse(self.http.get(self.url(path)).send()?)
    }

    fn post<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: Option<&B>,
    ) -> anyhow::Result<T> {
        let mut req = self.http.post(self.url(path));
        if let Some(b) = body {
            req = req.json(b);
        }
        parse(req.send()?)
    }

    fn delete<T: DeserializeOwned>(&self, path: &str) -> anyhow::Result<T> {
        parse(self.http.delete(self.url(path)).send()?)
    }
}

/// Decode a success body, or surface the control plane's `{"error": ...}`.
fn parse<T: DeserializeOwned>(resp: reqwest::blocking::Response) -> anyhow::Result<T> {
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if status.is_success() {
        serde_json::from_str(&text).with_context(|| format!("decoding response: {text}"))
    } else {
        let msg = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v["error"].as_str().map(str::to_string))
            .unwrap_or(text);
        anyhow::bail!("control plane returned {status}: {msg}");
    }
}

// --- money formatting (BTC/USD lives only here) ---------------------------

/// Satoshis with a BTC and rough-USD hint (SPEC §2 anchors at ~$100k/BTC).
fn fmt_sats(sats: i64) -> String {
    let btc = sats as f64 / 1e8;
    let usd = btc * 100_000.0;
    format!("{sats} sats  (≈ {btc:.8} BTC ≈ ${usd:.2})")
}

/// sats per unit of dlperf — the value ranking a buyer cares about.
fn value(rate_sats_per_min: i64, dlperf: f64) -> f64 {
    rate_sats_per_min as f64 / dlperf.max(0.01)
}

/// The connection hint for a rental, by kind — `(label, command)` — or `None`
/// until the host and port are known.
fn endpoint_hint(
    kind: RentalKind,
    host: Option<&str>,
    port: Option<i32>,
) -> Option<(&'static str, String)> {
    let (host, port) = (host?, port?);
    Some(match kind {
        RentalKind::Ssh => ("SSH:", format!("ssh -p {port} root@{host}")),
        RentalKind::HttpStatus => ("Status:", format!("curl http://{host}:{port}/")),
        RentalKind::HttpOpenai => (
            "LLM API:",
            format!(
                "curl http://{host}:{port}/v1/chat/completions -H content-type:application/json \
                 -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":64}}'"
            ),
        ),
    })
}

// --- command handlers -----------------------------------------------------

fn health(client: &Client) -> anyhow::Result<()> {
    let h: HealthResponse = client.get("/health")?;
    println!(
        "Control plane {}  (lightning backend: {})",
        if h.ok { "OK" } else { "DEGRADED" },
        h.ln_backend
    );
    Ok(())
}

fn offers(client: &Client, args: OffersArgs) -> anyhow::Result<()> {
    let mut params: Vec<(&str, String)> = Vec::new();
    if let Some(g) = &args.gpu {
        params.push(("gpu_name", g.clone()));
    }
    if let Some(v) = args.min_vram_mb {
        params.push(("min_vram_mb", v.to_string()));
    }
    if let Some(c) = args.min_gpu_count {
        params.push(("min_gpu_count", c.to_string()));
    }
    if let Some(r) = args.max_rate {
        params.push(("max_rate_sats_per_min", r.to_string()));
    }
    if let Some(s) = &args.sort {
        params.push(("sort", s.clone()));
    }
    if let Some(l) = args.limit {
        params.push(("limit", l.to_string()));
    }

    let resp: OffersResponse = parse(self_get_query(client, "/offers", &params)?)?;

    if resp.offers.is_empty() {
        println!("No machines available.");
        return Ok(());
    }

    let mut table = Table::new();
    table.set_header(vec![
        "ID", "GPU", "#", "VRAM", "dlperf", "sats/min", "sats/hr", "value", "loc",
    ]);
    for o in &resp.offers {
        table.add_row(vec![
            Cell::new(o.machine_id),
            Cell::new(&o.gpu_name),
            Cell::new(o.gpu_count),
            Cell::new(format!("{} GB", o.vram_mb / 1024)),
            Cell::new(format!("{:.0}", o.dlperf)),
            Cell::new(o.rate_sats_per_min),
            Cell::new(o.rate_sats_per_hour),
            Cell::new(format!("{:.2}", value(o.rate_sats_per_min, o.dlperf))),
            Cell::new(o.country.as_deref().unwrap_or("-")),
        ]);
    }
    println!("{table}");
    println!("\nvalue = sats/min per unit of dlperf (lower is better). ~$100k/BTC.");
    Ok(())
}

/// `GET` with query params (kept out of `Client` since only offers needs it).
fn self_get_query(
    client: &Client,
    path: &str,
    params: &[(&str, String)],
) -> anyhow::Result<reqwest::blocking::Response> {
    Ok(client.http.get(client.url(path)).query(params).send()?)
}

fn create_account(client: &Client) -> anyhow::Result<()> {
    let resp: CreateAccountResponse = client.post::<(), _>("/accounts", None)?;
    println!("Created account: {}", resp.account_id);
    println!("Fund it with:  vgpu deposit {} <sats>", resp.account_id);
    Ok(())
}

fn account(client: &Client, id: &str) -> anyhow::Result<()> {
    let a: AccountResponse = client.get(&format!("/accounts/{id}"))?;
    println!("Account {}", a.account_id);
    println!("  Balance: {}", fmt_sats(a.balance_sats));
    println!("  Burn:    {} sats/min", a.burn_sats_per_min);
    match a.runway_minutes {
        Some(m) => println!("  Runway:  {m} minutes"),
        None => println!("  Runway:  — (no running rentals)"),
    }
    Ok(())
}

fn deposit(client: &Client, account_id: &str, sats: i64) -> anyhow::Result<()> {
    let resp: DepositResponse = client.post(
        &format!("/accounts/{account_id}/deposit"),
        Some(&DepositRequest { sats }),
    )?;
    println!("Deposit invoice for {}", fmt_sats(resp.sats));
    println!("  BOLT11:       {}", resp.bolt11);
    println!("  Payment hash: {}", resp.payment_hash);
    println!("\nPay the invoice to credit your balance (the mock backend settles automatically).");
    Ok(())
}

fn rent(client: &Client, args: RentArgs) -> anyhow::Result<()> {
    let ssh_pubkey = match (args.ssh_key, args.ssh_key_file) {
        (Some(k), _) => Some(k),
        (None, Some(path)) => Some(
            std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?
                .trim()
                .to_string(),
        ),
        // metal: images run a host-controlled job that ignores the key, so it's
        // optional there; every other tier is an SSH box and requires one.
        (None, None) if is_http_status_image(&args.image) => None,
        (None, None) => {
            anyhow::bail!("provide --ssh-key or --ssh-key-file (only metal: images may omit it)")
        }
    };

    let resp: CreateRentalResponse = client.post(
        "/rentals",
        Some(&CreateRentalRequest {
            machine_id: args.machine_id,
            account_id: args.account_id,
            image: args.image,
            ssh_pubkey,
        }),
    )?;
    println!(
        "Rental {} created  [{}]  at {} sats/min",
        resp.rental_id,
        resp.status.as_str(),
        resp.rate_sats_per_min
    );
    println!("Track it with:  vgpu rental {}", resp.rental_id);
    Ok(())
}

fn rental(client: &Client, id: i64) -> anyhow::Result<()> {
    let r: RentalResponse = client.get(&format!("/rentals/{id}"))?;
    println!("Rental {}  [{}]", r.id, r.status.as_str());
    println!("  GPU:     {} x{}", r.gpu_name, r.gpu_count);
    println!("  Image:   {}", r.image);
    match endpoint_hint(r.kind, r.ssh_host.as_deref(), r.ssh_port) {
        Some((label, hint)) => println!("  {label:<8} {hint}"),
        None => println!("  {:<8} (pending — not running yet)", "Endpoint:"),
    }
    println!(
        "  Billed:  {} sats over {} min",
        r.sats_charged, r.minutes_billed
    );
    if let Some(err) = &r.error {
        println!("  Error:   {err}");
    }
    Ok(())
}

fn stop(client: &Client, id: i64) -> anyhow::Result<()> {
    let resp: serde_json::Value = client.delete(&format!("/rentals/{id}"))?;
    let status = resp["status"].as_str().unwrap_or("evicting");
    println!("Rental {id} -> {status} (stop queued)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_ranks_h100_above_t4() {
        // Same price, more performance → lower sats-per-dlperf is better.
        let h100 = value(60, 300.0);
        let t4 = value(60, 16.0);
        assert!(h100 < t4);
    }

    #[test]
    fn fmt_sats_shows_denominations() {
        // 60,000 sats ≈ 0.0006 BTC ≈ $60 at the $100k anchor.
        let s = fmt_sats(60_000);
        assert!(s.contains("60000 sats"));
        assert!(s.contains("0.00060000 BTC"));
        assert!(s.contains("$60.00"));
    }
}
