//! Qwen3.5 的 [`ModelLine`] 实现。

use std::collections::BTreeSet;

use clap::Args as ClapArgs;
use clap::FromArgMatches;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::CliDecodeOverlap;
use pegainfer_frontend::model_line::CliError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

use crate::Qwen35DecodeOverlap;
use crate::Qwen35LaunchOptions;
use crate::Qwen35SchedulerPolicy;
use crate::ServingCapacity;

pub static MODEL_LINE: Qwen35Line = Qwen35Line;

pub struct Qwen35Line;

// Qwen3.5 专属 CLI 参数：将单步 decode 宽度与最大驻留数分开命名。
#[derive(ClapArgs)]
struct Qwen35Cli {
    /// 单步 decode 宽度，1..=64，默认 64。
    #[arg(long)]
    decode_width: Option<usize>,

    /// 最大驻留请求数：单 GPU 默认 2 × decode_width（128，A/B 双 cohort），
    /// TP 默认与 decode_width 一致（64，单 cohort）；超出范围直接报错。
    #[arg(long)]
    max_resident: Option<usize>,

    /// Qwen3.5 prefill/decode scheduler policy. Defaults to `off`; `auto` is
    /// opt-in and currently single-GPU only.
    #[arg(long, value_enum, default_value_t = CliQwen35SchedulerPolicy::Off)]
    qwen35_scheduler_policy: CliQwen35SchedulerPolicy,
}

/// CLI selector for the Qwen3.5 adaptive scheduler policy.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum CliQwen35SchedulerPolicy {
    /// Fixed chunked-prefill behavior.
    #[default]
    Off,
    /// Runtime-state adaptive policy.
    Auto,
}

impl CliQwen35SchedulerPolicy {
    fn resolve(self) -> Qwen35SchedulerPolicy {
        match self {
            Self::Off => Qwen35SchedulerPolicy::Off,
            Self::Auto => Qwen35SchedulerPolicy::Auto,
        }
    }
}

fn cli(ctx: &LaunchContext<'_>) -> Qwen35Cli {
    Qwen35Cli::from_arg_matches(ctx.matches).expect("Qwen35Cli parses from the merged command")
}

fn resolve_capacity_args(cli: &Qwen35Cli, tp_size: usize) -> (usize, usize) {
    let decode_width = cli.decode_width.unwrap_or(crate::DEFAULT_DECODE_WIDTH);
    let max_resident = cli
        .max_resident
        .unwrap_or_else(|| ServingCapacity::default_max_resident(decode_width, tp_size));
    (decode_width, max_resident)
}

fn resolve_decode_overlap(overlap: CliDecodeOverlap) -> Result<Qwen35DecodeOverlap, CliError> {
    match overlap {
        CliDecodeOverlap::Off => Ok(Qwen35DecodeOverlap::Off),
        CliDecodeOverlap::Stream => Ok(Qwen35DecodeOverlap::SharedSm),
        CliDecodeOverlap::GreenCtx => Err(CliError::rule(
            "Qwen3.5 supports --decode-overlap=off|stream only",
        )),
    }
}

impl ModelLine for Qwen35Line {
    fn name(&self) -> &'static str {
        "Qwen3.5"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
        let text_model_type = config
            .get("text_config")
            .and_then(|text| text.get("model_type"))
            .and_then(serde_json::Value::as_str);
        if model_type != Some("qwen3_5") && text_model_type != Some("qwen3_5_text") {
            return Err(format!("model_type {model_type:?} is not \"qwen3_5\""));
        }
        crate::probe_config_json(config).map_err(|error| error.to_string())
    }

    fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
        Qwen35Cli::augment_args(cmd)
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &[
            "device_ordinal",
            "tp_size",
            "cuda_graph",
            "max_prefill_tokens",
            "decode_overlap",
            "decode_sm_pct",
        ]
    }

    fn validate(
        &self,
        ctx: &LaunchContext<'_>,
        _provided: &BTreeSet<String>,
    ) -> Result<(), CliError> {
        let cli = cli(ctx);
        let decode_overlap = resolve_decode_overlap(ctx.shared.decode_overlap)?;
        let (decode_width, max_resident) = resolve_capacity_args(&cli, ctx.shared.tp_size);
        ServingCapacity::validate_request(decode_width, max_resident, ctx.shared.tp_size)
            .map_err(|err| CliError::rule(err.to_string()))?;

        if decode_overlap != Qwen35DecodeOverlap::Off
            && decode_width > crate::MAX_SHARED_SM_DECODE_BATCH
        {
            return Err(CliError::rule(format!(
                "Qwen3.5 --decode-overlap=stream requires --decode-width <= {}; got {decode_width}",
                crate::MAX_SHARED_SM_DECODE_BATCH
            )));
        }
        if decode_overlap != Qwen35DecodeOverlap::Off
            && matches!(cli.qwen35_scheduler_policy, CliQwen35SchedulerPolicy::Auto)
        {
            return Err(CliError::rule(
                "Qwen3.5 --decode-overlap=stream requires --qwen35-scheduler-policy=off",
            ));
        }
        if ctx.shared.tp_size > 1 && decode_overlap != Qwen35DecodeOverlap::Off {
            return Err(CliError::rule(
                "--decode-overlap is single-GPU only; tp_size>1 has no prefill/decode overlap",
            ));
        }
        if ctx.shared.tp_size > 1
            && matches!(cli.qwen35_scheduler_policy, CliQwen35SchedulerPolicy::Auto)
        {
            return Err(CliError::rule(
                "--qwen35-scheduler-policy=auto is single-GPU only; Qwen3.5 TP uses the fixed off policy",
            ));
        }
        Ok(())
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        let cli = cli(ctx);
        let (decode_width, max_resident) = resolve_capacity_args(&cli, ctx.shared.tp_size);
        crate::launch_with_options_policy_and_overlap(
            ctx.model_path,
            Qwen35LaunchOptions {
                device_ordinal: ctx.shared.device_ordinal,
                tp_size: ctx.shared.tp_size,
                cuda_graph: ctx.shared.cuda_graph,
                decode_width,
                max_resident,
                max_prefill_tokens: ctx
                    .shared
                    .max_prefill_tokens
                    .unwrap_or(crate::DEFAULT_MAX_PREFILL_TOKENS),
            },
            cli.qwen35_scheduler_policy.resolve(),
            resolve_decode_overlap(ctx.shared.decode_overlap)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
        )
        .map(LaunchedEngine::Handle)
    }
}

#[cfg(test)]
mod tests {
    use pegainfer_frontend::model_line::parse_for_line;

    use super::*;

    fn validate_argv(argv: &[&str]) -> Result<(), CliError> {
        let (shared, matches, provided) =
            parse_for_line(&MODEL_LINE, argv).map_err(|error| CliError::rule(error.to_string()))?;
        let config = serde_json::json!({});
        let ctx = LaunchContext {
            model_path: std::path::Path::new("unused"),
            config: &config,
            shared: &shared,
            matches: &matches,
        };
        MODEL_LINE.validate(&ctx, &provided)
    }

    #[test]
    fn probe_accepts_qwen35_identity() {
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_text"}
        });
        MODEL_LINE
            .probe(&json)
            .expect("qwen3_5 config should probe");
    }

    #[test]
    fn accepts_tp_size() {
        validate_argv(&["pegainfer", "--tp-size", "2", "--cuda-graph=false"])
            .expect("Qwen3.5 should accept --tp-size for eager TP startup");
    }

    #[test]
    fn rejects_tp_dual_cohort_max_resident() {
        let error = validate_argv(&[
            "pegainfer",
            "--tp-size",
            "2",
            "--cuda-graph=false",
            "--max-resident",
            "128",
        ])
        .expect_err("Qwen3.5 TP should reject max_resident > decode_width");
        assert!(error.to_string().contains("single cohort"));
    }

    #[test]
    fn accepts_scheduler_policy_off() {
        validate_argv(&["pegainfer", "--qwen35-scheduler-policy", "off"])
            .expect("Qwen3.5 should accept explicit scheduler-policy off");
    }

    #[test]
    fn accepts_shared_stream_overlap() {
        let decode_width = crate::MAX_SHARED_SM_DECODE_BATCH.to_string();
        validate_argv(&[
            "pegainfer",
            "--decode-overlap",
            "stream",
            "--decode-width",
            &decode_width,
        ])
        .expect("Qwen3.5 should accept shared-stream overlap");
    }

    #[test]
    fn rejects_stream_overlap_default_decode_width() {
        let error = validate_argv(&["pegainfer", "--decode-overlap", "stream"])
            .expect_err("Qwen3.5 should reject stream overlap at the default decode_width");
        assert!(error.to_string().contains("decode-width"));
        assert!(
            error
                .to_string()
                .contains(&crate::MAX_SHARED_SM_DECODE_BATCH.to_string())
        );
    }

    #[test]
    fn rejects_green_context_overlap() {
        let error = validate_argv(&["pegainfer", "--decode-overlap", "green-ctx"])
            .expect_err("Qwen3.5 should reject green-context overlap");
        assert!(error.to_string().contains("off|stream"));
    }

    #[test]
    fn rejects_auto_policy_with_overlap() {
        let error = validate_argv(&[
            "pegainfer",
            "--decode-overlap",
            "stream",
            "--decode-width",
            "32",
            "--qwen35-scheduler-policy",
            "auto",
        ])
        .expect_err("Qwen3.5 should reject auto policy with overlap");
        assert!(error.to_string().contains("scheduler-policy=off"));
    }

    #[test]
    fn rejects_tp_auto_scheduler_policy() {
        let error = validate_argv(&[
            "pegainfer",
            "--tp-size",
            "2",
            "--cuda-graph=false",
            "--qwen35-scheduler-policy",
            "auto",
        ])
        .expect_err("Qwen3.5 TP should reject auto scheduler-policy");
        assert!(error.to_string().contains("single-GPU only"));
    }

    #[test]
    fn accepts_explicit_decode_width_and_max_resident() {
        validate_argv(&["pegainfer", "--decode-width", "64", "--max-resident", "128"])
            .expect("Qwen3.5 should accept 64/128 on single GPU");
        validate_argv(&["pegainfer", "--decode-width", "5", "--max-resident", "5"])
            .expect("Qwen3.5 should accept sub-bucket decode_width/max_resident");
    }

    #[test]
    fn rejects_legacy_max_batch_flag() {
        let error = validate_argv(&["pegainfer", "--max-batch", "64"])
            .expect_err("--max-batch should no longer be accepted");
        assert!(error.to_string().contains("max-batch"));
    }

    #[test]
    fn rejects_zero_decode_width_and_over_resident() {
        let error = validate_argv(&["pegainfer", "--decode-width", "0"])
            .expect_err("Qwen3.5 should reject zero decode_width");
        assert!(error.to_string().contains("decode_width must be in 1..="));

        let error = validate_argv(&["pegainfer", "--decode-width", "64", "--max-resident", "129"])
            .expect_err("Qwen3.5 should reject max_resident > 2 * decode_width");
        assert!(error.to_string().contains("max_resident"));
    }
}
