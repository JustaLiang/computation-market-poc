-- Schema for the GPU rental marketplace control plane (SPEC §3).
-- Integer satoshis throughout; timestamps are Unix seconds (i64).
-- Booleans are stored as INTEGER 0/1.

CREATE TABLE machines (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_token       TEXT    NOT NULL UNIQUE,
    host_id           TEXT    NOT NULL UNIQUE,
    gpu_name          TEXT    NOT NULL,
    gpu_count         INTEGER NOT NULL,
    vram_mb           INTEGER NOT NULL,
    cpu_name          TEXT    NOT NULL,
    cpu_cores         INTEGER NOT NULL,
    ram_mb            INTEGER NOT NULL,
    disk_gb           INTEGER NOT NULL,
    disk_type         TEXT    NOT NULL,
    public_ip         TEXT    NOT NULL,
    port_start        INTEGER NOT NULL,
    port_end          INTEGER NOT NULL,
    inet_down_mbps    REAL,
    inet_up_mbps      REAL,
    country           TEXT,
    dlperf            REAL    NOT NULL,
    rate_sats_per_min INTEGER NOT NULL,
    payout_balance    INTEGER NOT NULL DEFAULT 0,
    online            INTEGER NOT NULL DEFAULT 0,
    last_heartbeat    INTEGER NOT NULL DEFAULT 0,
    hw_fingerprint    TEXT    NOT NULL,
    created_at        INTEGER NOT NULL
);

CREATE TABLE accounts (
    id           TEXT    PRIMARY KEY,
    balance_sats INTEGER NOT NULL DEFAULT 0 CHECK (balance_sats >= 0),
    created_at   INTEGER NOT NULL
);

CREATE TABLE rentals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id        INTEGER NOT NULL REFERENCES machines(id),
    account_id        TEXT    NOT NULL REFERENCES accounts(id),
    image             TEXT    NOT NULL,
    ssh_pubkey        TEXT    NOT NULL,
    status            TEXT    NOT NULL,
    ssh_host          TEXT,
    ssh_port          INTEGER,
    container_id      TEXT,
    error             TEXT,
    rate_sats_per_min INTEGER NOT NULL,   -- snapshotted at creation (SPEC §3)
    sats_charged      INTEGER NOT NULL DEFAULT 0,
    minutes_billed    INTEGER NOT NULL DEFAULT 0,
    paid_through      INTEGER NOT NULL DEFAULT 0,
    created_at        INTEGER NOT NULL,
    ended_at          INTEGER
);

CREATE TABLE invoices (
    payment_hash TEXT    PRIMARY KEY,
    account_id   TEXT    NOT NULL REFERENCES accounts(id),
    sats         INTEGER NOT NULL,
    bolt11       TEXT    NOT NULL,
    settled      INTEGER NOT NULL DEFAULT 0,
    created_at   INTEGER NOT NULL
);

-- Append-only double-entry audit trail. Never updated, never deleted.
-- Invariant: SUM(delta_sats) == 0 across the whole table.
CREATE TABLE ledger (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         INTEGER NOT NULL,
    account_id TEXT,
    machine_id INTEGER,
    rental_id  INTEGER,
    delta_sats INTEGER NOT NULL,
    kind       TEXT    NOT NULL,
    note       TEXT
);

-- Queue of instructions awaiting heartbeat delivery (at-most-once).
CREATE TABLE commands (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL REFERENCES machines(id),
    payload    TEXT    NOT NULL,
    delivered  INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_machines_online_dlperf   ON machines(online, dlperf);
CREATE INDEX idx_rentals_status           ON rentals(status);
CREATE INDEX idx_commands_machine_deliver ON commands(machine_id, delivered);
