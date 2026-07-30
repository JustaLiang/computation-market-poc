//! Acceptance test — the full lifecycle from SPEC §10, with no GPU, no container
//! runtime, and no Lightning node: a mock LN backend that settles immediately, a
//! manual clock so time advances deterministically, and `BILL_PERIOD` compressed
//! to 2s. The router is driven with `oneshot` — no real port, no flakiness.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;

use control_plane::db;
use control_plane::lightning::mock::MockBackend;
use control_plane::state::{AppState, BillingConfig, Clock};

async fn call(
    app: &Router,
    method: Method,
    uri: &str,
    body: Option<Value>,
    bearer: Option<&str>,
) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(token) = bearer {
        builder = builder.header(AUTHORIZATION, format!("Bearer {token}"));
    }
    let request = match body {
        Some(v) => builder
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(request).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, value)
}

#[tokio::test]
async fn full_lifecycle() {
    const RATE: i64 = 10; // sats/min
    const DEPOSIT: i64 = 30; // exactly three periods

    let pool = db::connect("sqlite::memory:", 1).await.unwrap();
    let state = AppState {
        pool,
        ln: Arc::new(MockBackend::new(Duration::ZERO)), // settles immediately
        billing: BillingConfig {
            bill_period: Duration::from_secs(2), // compressed
            tick: Duration::from_secs(1),
            heartbeat_timeout: Duration::from_secs(90),
        },
        clock: Clock::manual(1_000_000),
        ln_backend_name: "mock".to_string(),
    };
    let app = control_plane::build_router(state.clone());

    // 1. Host registers.
    let (st, body) = call(
        &app,
        Method::POST,
        "/agent/register",
        Some(json!({
            "host_id": "host-1",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "vram_mb": 24564,
            "cpu_name": "AMD Ryzen 9",
            "cpu_cores": 16,
            "ram_mb": 64000,
            "disk_gb": 1000,
            "disk_type": "nvme",
            "public_ip": "203.0.113.5",
            "port_start": 40000,
            "port_end": 40099,
            "dlperf": 42.0,
            "rate_sats_per_min": RATE,
            "hw_fingerprint": "fp-abc"
        })),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "register: {body}");
    let machine_id = body["machine_id"].as_i64().unwrap();
    let token = body["agent_token"].as_str().unwrap().to_string();

    // 2. Offer index lists it with correct rate_sats_per_hour.
    let (st, body) = call(&app, Method::GET, "/offers", None, None).await;
    assert_eq!(st, StatusCode::OK);
    let offers = body["offers"].as_array().unwrap();
    assert_eq!(offers.len(), 1);
    assert_eq!(offers[0]["machine_id"].as_i64().unwrap(), machine_id);
    assert_eq!(offers[0]["rate_sats_per_hour"].as_i64().unwrap(), RATE * 60);

    // 3. Account created, deposit invoice issued, auto-settles, balance credited.
    let (st, body) = call(&app, Method::POST, "/accounts", None, None).await;
    assert_eq!(st, StatusCode::CREATED);
    let account_id = body["account_id"].as_str().unwrap().to_string();

    let (st, body) = call(
        &app,
        Method::POST,
        &format!("/accounts/{account_id}/deposit"),
        Some(json!({ "sats": DEPOSIT })),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(body["sats"].as_i64().unwrap(), DEPOSIT);
    assert!(body["payment_hash"].is_string());

    // Ticker settles the invoice and credits the balance.
    control_plane::billing::tick(&state).await.unwrap();
    let (_, body) = call(
        &app,
        Method::GET,
        &format!("/accounts/{account_id}"),
        None,
        None,
    )
    .await;
    assert_eq!(body["balance_sats"].as_i64().unwrap(), DEPOSIT);

    // 4. Rental created → first minute charged up front, before any container.
    let (st, body) = call(
        &app,
        Method::POST,
        "/rentals",
        Some(json!({
            "machine_id": machine_id,
            "account_id": account_id,
            "image": "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
            "ssh_pubkey": "ssh-ed25519 AAAAC3NzaC1lZDI1"
        })),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::CREATED, "create rental: {body}");
    let rental_id = body["rental_id"].as_i64().unwrap();
    assert_eq!(body["status"].as_str().unwrap(), "provisioning");
    assert_eq!(body["rate_sats_per_min"].as_i64().unwrap(), RATE);

    let (_, acct) = call(
        &app,
        Method::GET,
        &format!("/accounts/{account_id}"),
        None,
        None,
    )
    .await;
    assert_eq!(acct["balance_sats"].as_i64().unwrap(), DEPOSIT - RATE);

    let (_, rental) = call(
        &app,
        Method::GET,
        &format!("/rentals/{rental_id}"),
        None,
        None,
    )
    .await;
    assert_eq!(rental["sats_charged"].as_i64().unwrap(), RATE);
    assert_eq!(rental["minutes_billed"].as_i64().unwrap(), 1);
    assert!(
        rental.get("ssh_pubkey").is_none(),
        "ssh_pubkey must never leave the API"
    );

    // 5. Heartbeat delivers start_rental; agent reports running; ssh_command correct.
    let (st, hb) = call(
        &app,
        Method::POST,
        "/agent/heartbeat",
        Some(json!({ "online": true })),
        Some(&token),
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    let cmds = hb["commands"].as_array().unwrap();
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0]["cmd"].as_str().unwrap(), "start_rental");
    assert_eq!(cmds[0]["rental_id"].as_i64().unwrap(), rental_id);

    // Commands are delivered at most once: a second heartbeat returns nothing.
    let (_, hb2) = call(
        &app,
        Method::POST,
        "/agent/heartbeat",
        Some(json!({ "online": true })),
        Some(&token),
    )
    .await;
    assert_eq!(hb2["commands"].as_array().unwrap().len(), 0);

    let (st, _) = call(
        &app,
        Method::POST,
        "/agent/report",
        Some(json!({
            "rental_id": rental_id,
            "status": "running",
            "ssh_port": 40000,
            "container_id": "deadbeefcafe"
        })),
        Some(&token),
    )
    .await;
    assert_eq!(st, StatusCode::OK);

    let (_, rental) = call(
        &app,
        Method::GET,
        &format!("/rentals/{rental_id}"),
        None,
        None,
    )
    .await;
    assert_eq!(rental["status"].as_str().unwrap(), "running");
    assert_eq!(
        rental["ssh_command"].as_str().unwrap(),
        "ssh -p 40000 root@203.0.113.5"
    );
    assert!(rental.get("ssh_pubkey").is_none());

    // 6. Machine no longer appears in the offer index.
    let (_, body) = call(&app, Method::GET, "/offers", None, None).await;
    assert_eq!(body["offers"].as_array().unwrap().len(), 0);

    // 7. Ticker drains the balance one period at a time; evicts at zero;
    //    balance never goes negative; sats_charged equals the deposit.
    let mut expected_balance = DEPOSIT - RATE; // 20 after the up-front charge
    for _ in 0..2 {
        state.clock.advance(2); // reach the next paid_through
        control_plane::billing::tick(&state).await.unwrap();
        expected_balance -= RATE;
        let (_, acct) = call(
            &app,
            Method::GET,
            &format!("/accounts/{account_id}"),
            None,
            None,
        )
        .await;
        assert_eq!(acct["balance_sats"].as_i64().unwrap(), expected_balance);
        assert!(
            acct["balance_sats"].as_i64().unwrap() >= 0,
            "balance never negative"
        );
    }
    // Balance is now 0; the next period cannot be paid → eviction.
    state.clock.advance(2);
    control_plane::billing::tick(&state).await.unwrap();

    let (_, rental) = call(
        &app,
        Method::GET,
        &format!("/rentals/{rental_id}"),
        None,
        None,
    )
    .await;
    assert_eq!(rental["status"].as_str().unwrap(), "evicting");
    assert_eq!(rental["sats_charged"].as_i64().unwrap(), DEPOSIT);
    let (_, acct) = call(
        &app,
        Method::GET,
        &format!("/accounts/{account_id}"),
        None,
        None,
    )
    .await;
    assert_eq!(acct["balance_sats"].as_i64().unwrap(), 0);

    // 8. stop_rental delivered; agent reports stopped; machine re-listed.
    let (_, hb) = call(
        &app,
        Method::POST,
        "/agent/heartbeat",
        Some(json!({ "online": true })),
        Some(&token),
    )
    .await;
    let cmds = hb["commands"].as_array().unwrap();
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0]["cmd"].as_str().unwrap(), "stop_rental");
    assert_eq!(cmds[0]["rental_id"].as_i64().unwrap(), rental_id);

    let (st, _) = call(
        &app,
        Method::POST,
        "/agent/report",
        Some(json!({ "rental_id": rental_id, "status": "stopped" })),
        Some(&token),
    )
    .await;
    assert_eq!(st, StatusCode::OK);

    let (_, body) = call(&app, Method::GET, "/offers", None, None).await;
    assert_eq!(
        body["offers"].as_array().unwrap().len(),
        1,
        "machine is idle again and should be re-listed"
    );

    // 9. Host payout succeeds and zeroes payout_balance.
    let (st, body) = call(
        &app,
        Method::POST,
        "/agent/payout",
        Some(json!({ "bolt11": "lnbcmock-host-invoice" })),
        Some(&token),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "payout: {body}");
    assert_eq!(body["paid_sats"].as_i64().unwrap(), DEPOSIT);

    let payout_balance: i64 = sqlx::query_scalar("SELECT payout_balance FROM machines WHERE id=?")
        .bind(machine_id)
        .fetch_one(&state.pool)
        .await
        .unwrap();
    assert_eq!(payout_balance, 0);

    // 10. The important one: the ledger reconciles to zero.
    let ledger_sum: i64 = sqlx::query_scalar("SELECT COALESCE(SUM(delta_sats), 0) FROM ledger")
        .fetch_one(&state.pool)
        .await
        .unwrap();
    assert_eq!(ledger_sum, 0, "SUM(ledger.delta_sats) must be 0");
}
