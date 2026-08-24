//! Crate-level error type for the public Qwen3.5 engine boundary.
//!
//! The engine internals propagate [`anyhow::Error`] because it carries the rich
//! context that makes cross-module failures readable during bring-up. This enum
//! is the single, stable error type that the crate's public API surfaces, so
//! external consumers match on one `pegainfer_qwen35::Error` instead of a
//! downstream `anyhow::Error`. Internal `anyhow` errors are folded in at the
//! public boundary via [`Error::from`] (`?`).

use thiserror::Error as ThisError;

/// Errors surfaced by the public `pegainfer_qwen35` engine API.
#[derive(Debug, ThisError)]
pub enum Error {
    /// A failure folded in from the `anyhow`-based engine internals.
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
