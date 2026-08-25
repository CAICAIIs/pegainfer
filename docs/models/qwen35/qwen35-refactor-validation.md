# Qwen3.5 refactor + validation + serving performance

> **TL;DR** Structurally refactored `pegainfer-qwen35` (god modules split into concern-grouped submodules) plus idiomatic hardening, behavior unchanged. Correctness is fully validated, and the Qwen3.5-4B serving performance was benchmarked on a single A100 and compared head-to-head against vLLM 0.27.0 (same model, config, seed).
>
> **Last touched:** 2026-08

## 1. Structure refactor (behavior unchanged)

- `scheduler.rs` 2624→969 → `scheduler/{backend, tp, steps, telemetry, emit, plan, tests}`
- `tp_executor.rs` 2520→723 → `tp_executor/{executor, worker, responses}`
- `config.rs` 683→285 → `config/{model, tokenizer, tp}`
- `weights.rs` 921→802 + `weights/layers.rs`
- `forward/` — the 5 forward-pass modules regrouped under a single owner
- Idiomatic: crate-level `thiserror` `Error`; production `unwrap/expect` → `Result`; params Bundles; `debug_assert!` → release `ensure!`; removed `phase1`/`P2A`/issue-numbered naming.
- Flat convention: `foo.rs` entry + sibling `foo/` submodules, no `mod.rs`.

## 2. Correctness (all green)

| gate | result |
|---|---|
| `cargo test --lib` | 95/95 |
| `tp2_*` unit tests (2 CUDA + NCCL + weights) | 7/7 |
| `hf_golden_gate` (logits vs HF goldens) | 4/4 (2 single-GPU + 2 tp2) |
| `e2e_scheduler` (scheduler liveness / request flow) | 3/3 (2 single-GPU + 1 tp2) |
| `chunked_prefill` | 1/1 |
| `sampling_behavior` | 1/1 |
| `cargo clippy --features qwen35` | 0 warnings |
| `cargo fmt --check` | clean |

> Behavior is preserved through the refactor; correctness (including the TP-path accuracy gates) is intact.

## 3. PegaInfer serving performance (single A100, vllm-bench)

`input 1024 / output 128 / request-rate 4 / num-prompts 40 / seed 42`:

- Mean / Median / P99 **TTFT**: 96.07 / 84.10 / 185.93 ms
- Mean / Median / P99 **TPOT**: 16.53 / 15.66 / 22.72 ms
- **Output** 445.18 tok/s · **Total** 3856.84 tok/s · peak concurrency 22 · duration 11.50 s
- Smaller run (64/32, rate 1): TTFT 23.6 ms / TPOT 9.6 ms.

## 4. PegaInfer vs vLLM 0.27.0 (direct A/B, same config/seed/GPU)

| | PegaInfer | vLLM |
|---|---|---|
| TTFT mean | **96.07 ms** | 114.29 ms |
| TPOT mean | 16.53 ms | **11.70 ms** |
| Output tok/s | 445.18 | **455.02** |
| Total tok/s | 3856.84 | **4095.15** |

> vLLM decodes faster (lower TPOT / marginally higher throughput); PegaInfer prefills faster (lower TTFT); the two are close on this host/workload. This differs from the repo's documented RTX 5090 + 1024/256 finding where PegaInfer trailed vLLM more clearly — on this A100 host with this workload the gap is small.

## 5. Host setup fixes needed to run vLLM here (for reproduction)

1. Install clean vllm 0.27.0 into `/mnt/data/models/vllm-clean-venv`.
2. Use `--tokenizer /mnt/data/models/Qwen3.5-4B` and `HF_ENDPOINT=https://hf-mirror.com` (this host cannot reach Hugging Face directly).
3. `vllm bench serve --base-url http://127.0.0.1:<port>` (omit a trailing `/v1` so it does not build `/v1/v1` → 404).
4. Add `from __future__ import annotations` to `flashinfer/comm/fd_exchange.py` (fixes a `tuple[...]`/`array.array[int]` TypeError on Python 3.10).
5. Serve with `CUDA_HOME=/usr/local/cuda-13.1` + its `bin` on `PATH` + `CC=gcc-12 CXX=g++-12` (vLLM's JIT kernels need CUDA >= 12 and a newer GCC; the host default nvcc is 11.5 and GCC is 11).

## 6. Result JSON

- PegaInfer: `/tmp/q35b5/qwen35-1024x128-r4.json`
- vLLM: `/tmp/q35v/vllm-1024x128-r4.json`
