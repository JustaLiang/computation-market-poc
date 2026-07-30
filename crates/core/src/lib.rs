//! Shared types for the GPU rental marketplace.
//!
//! This crate carries the vocabulary spoken by the control plane, the host
//! agent, and the tenant CLI: the [`money::Sats`] newtype, the domain enums in
//! [`model`], and the request/response DTOs in [`api`]. It performs **no I/O**
//! and depends on neither `axum` nor `sqlx` query machinery — see `CLAUDE.md`.

pub mod api;
pub mod model;
pub mod money;

pub use money::Sats;
