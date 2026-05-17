# Simulated Inference Engine

**Created**: 2026-05-16
**Status**: ready for PR review
**TL;DR**: `pegainfer-sim` is a CPU-only simulated model crate that serves through the existing vLLM/OpenAI frontend with configurable TTFT/TPOT. It is a benchmark and frontend validation harness, not a real-model performance path.

## Scope

Issue #125 needs a server path that can run `vllm bench serve` without GPU or model weights while still exercising the same HTTP frontend used by real pegainfer models.

This PR keeps that boundary narrow:

- Add `pegainfer-engine` for the lightweight `EngineHandle`, `GenerateRequest`, `TokenEvent`, and `SamplingParams` contract.
- Re-export that contract from `pegainfer-core` so existing model crates keep their current imports.
- Move the vLLM bridge into `pegainfer-vllm-frontend`, leaving `pegainfer-server/src/vllm_frontend.rs` as a compatibility re-export.
- Add `pegainfer-sim` as an independently maintained model crate with a thin CLI binary.

Out of scope:

- No CUDA, kernel, KV-cache, or real model execution changes.
- No claim about real model serving throughput.
- No jitter, tail-latency distribution, or batching realism beyond fixed TTFT/TPOT timing.

## Behavior

`pegainfer-sim` exposes CLI knobs for model id, port, max model length, base TTFT, prefill throughput, TPOT, and fallback token id.

The timing model is intentionally simple: TTFT is `base_ttft_ms + prompt_len / prefill_tokens_per_ms`, and TPOT is a fixed delay between generated tokens. Output token ids cycle through the prompt tokens, using the fallback id for empty prompts.

The frontend still needs tokenizer/model metadata, but the simulator never loads model weights.

## Evidence

- Format/metadata gates pass: `git diff --check`, `cargo fmt --check`, and `cargo metadata --no-deps --format-version 1`.
- Dependency gate passes: `cargo tree -p pegainfer-sim --edges normal | rg "pegainfer-(core|kernels|qwen|deepseek)|cudarc|cuda"` has no matches.
- Rust gates pass: `cargo test --release -p pegainfer-engine`, `cargo test --release -p pegainfer-vllm-frontend`, `cargo test --release -p pegainfer-sim`, and focused clippy for the frontend and sim crates.
- Local HTTP smoke passes for `/v1/models`, non-streaming `/v1/completions`, and streaming `/v1/completions`.
- Remote `vllm bench serve` gate: 50 successful requests, 0 failed requests, concurrency 4, mean TTFT 22.01 ms, mean TPOT 13.12 ms, output throughput 291.66 tok/s.

## Follow-Ups

If reviewers want richer simulation, add jitter, tail distributions, and batching behavior in follow-up PRs after this crate boundary lands.
