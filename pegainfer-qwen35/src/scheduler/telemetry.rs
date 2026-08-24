//! Qwen3.5 scheduler: `telemetry` cluster, split out of scheduler.rs.
//! Reaches the rest of the scheduler via `use super::*;`.

use super::*;

pub(super) fn itl_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PEGAINFER_ITL_DEBUG").is_some())
}

/// Monotonic microseconds since the first ITL step, so `ITL_STEP` timestamps
/// are correlatable within one process run (paired with wall-clock epoch us).
pub(super) fn itl_debug_mono_us() -> u128 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_micros()
}

pub(super) fn log_itl_step(
    step_start: Option<Instant>,
    plan: &str,
    prefill_tokens: usize,
    prefill_reqs: usize,
    decode_n: usize,
) {
    let Some(step_start) = step_start else {
        return;
    };
    let dur_us = step_start.elapsed().as_micros();
    let epoch_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_micros());
    info!(
        "ITL_STEP mono_us={} epoch_us={} plan={} prefill_tok={} prefill_reqs={} decode_n={} dur_us={}",
        itl_debug_mono_us(),
        epoch_us,
        plan,
        prefill_tokens,
        prefill_reqs,
        decode_n,
        dur_us
    );
}
