# Qwen3.5-4B Optimization

> **TL;DR:** Hybrid 24 linear + 8 full-attention. Current headline: the decode-tuning refresh fuses MLP gate/up and tunes decode cuBLASLt buckets, improving direct TPOT by `2.1–3.2%`; vLLM still leads 1024/256 HTTP decode and high-concurrency throughput, and the next gap search is serving/scheduler/event-sync overhead. The current pegainfer-vs-vLLM comparison lives in [Qwen3.5 serving: pegainfer vs vLLM on RTX 5090](../../benchmarks/qwen35-4b-serving-vllm-rtx5090.md). The one correctness note that matters: the chunk-wise GDR prefill had a `v_new` writeback bug that is fixed, and the old exact-text e2e baseline is retired in favor of the HF logits gate.
>
> **Last touched:** 2026-08. Qwen3.5 runtime code lives in top-level `pegainfer-qwen35`. Accuracy coverage: `PEGAINFER_TEST_MODEL_PATH=<absolute Qwen3.5-4B path> cargo test --release -p pegainfer-qwen35 --test hf_golden_gate -- --nocapture`; run `e2e_scheduler` when scheduler request-flow behavior changes. The old exact-text e2e/regen baseline was retired by the HF logits gate in `docs/models/qwen35/accuracy.md`. The in-process `bench_serving` bin referenced in the historical log below is retired as of 2026-08 — use HTTP-level benching (`scripts/bench_http_serving.py` / vllm-bench, see [profiling-guide](../../playbooks/profiling-guide.md)) for serving numbers.

## Goal

Close Qwen3.5's decode and serving gap against the current vLLM baseline on the
same GPU/workload. The current refresh is a measured incremental step: direct
PegaInfer decode gets a few-percent TPOT improvement, but HTTP 1024/256 and
high-concurrency serving still trail vLLM and remain active optimization work.

## Current Decode Refresh

Changes in the decode-tuning refresh:

- MLP gate/up weights are stacked at load time, so runtime uses one gate-up GEMM
  plus fused SiLU\*mul.
- Decode cuBLASLt buckets are tuned before CUDA Graph capture.
- Token selection stays on the shared batched sampler path.

Same-host direct A/B:

| Workload | Metric | upstream/main | tuned branch | Delta |
| --- | --- | ---: | ---: | ---: |
| 1 input / 256 output | steady TPOT avg | 6.524 ms | 6.386 ms | -2.1% |
| 1 input / 512 output | steady TPOT avg | 6.603 ms | 6.397 ms | -3.1% |
| 1024 input / 256 output | steady TPOT avg | 7.338 ms | 7.100 ms | -3.2% |
| 2048 input / 1 output | TTFT avg | 97.978 ms | 95.855 ms | -2.2% |

Current vLLM boundary:

- Prompt-len-1 HTTP decode is close: 1/256 TPOT mean `6.282 ms` vs vLLM
  `6.214 ms`, and 1/512 TPOT mean `6.381 ms` vs vLLM `6.221 ms`.
- 1024/256 HTTP decode still trails: TPOT mean `7.110 ms` vs vLLM `6.346 ms`
  at concurrency 1, widening to `15.566 ms` vs `9.823 ms` at concurrency 16.
- 2048/1 and 1024/256 TTFT rows are fixed-client timings, not token-normalized
  prefill throughput, because the servers report different prompt-token totals.
- Nsight Systems measured the direct 1024/256 concurrency-16 path at `9.320 ms`
  steady TPOT avg; the HTTP row is `15.566 ms`. The next gap search is
  serving/scheduler/event-sync overhead.

## Correctness caveat: GDR chunk-state writeback

The major chunk-wise correctness blocker was one bug in
`gdr_chunk_state_qwen35_kernel`: it wrote `v_new` after multiplying by
`exp(g_last - g_t)`. The correct semantics are to write the ungated `v_new` to
memory and use only the gated form for the recurrent `h += k @ v_new_gated`
update. After the fix, the chunk-wise path matches FLA — `v_new` stage diff
`max ~1.95e-3`, `chunk_o` output diff `max ~1.22e-4`, and the final recurrent
state stays exact after layout alignment.

## Architecture

- Layers: 32 (24 linear attention + 8 full attention at indices 3, 7, 11, 15,
  19, 23, 27, 31).
- `hidden_dim` 2560, MLP `intermediate_size` 9216, RMSNorm (1+weight) offset,
  eps 1e-6. 4B ties embeddings (embed_tokens doubles as LM head); 9B/27B are
  untied with a top-level `lm_head.weight`. Vocab 248,320.

Full attention (8 layers): 16 q heads, 4 kv heads (GQA 4), `head_dim` 256,
partial RoPE (`rotary_dim=64`, `theta=1e7`), q projection includes an output
gate `[8192, 2560]`, QK norm per head.

Linear attention (24 layers): 16 q heads / 16 k heads (k_dim 128), 32 v heads
(v_dim 128); Conv1d kernel_dim 4 on QKV (dim 8192); gated delta rule with
recurrent state `[32, 128, 128]` f32 per layer; output gating via Z projection
`[4096, 2560]` → SiLU gate on RMSNorm'd output; `A_log [32]`, `dt_bias [32]`,
`norm_weight [128]`.

Prefill pipeline (full attention): RNMSNorm → batched Q/K/V GEMMs →
prep (QK norm, partial RoPE, KV write) → batched Triton attention →
output gate → O GEMM → residual+norm → fused MLP. Linear attention: RNMSNorm →
QKV/Z/B/A GEMMs → conv1d_prefill → 7 chunk-wise Triton GDR stages
(prepare/cumsum/kkt/solve/recompute/state/output) → gated RMSNorm → O GEMM →
residual+norm → fused MLP.

Decode is fully CUDA Graph'd with zero GPU allocation after the first token.
conv1d and GDR are single-token operations — no per-token loop penalty.
Full-attention prefill is no longer a meaningful TTFT bottleneck; the remaining
prefill cost is the linear-attention (GDR) compute.

## Where the time is (measured conclusions)

- **Decode is bandwidth-bound.** GEMV + fused MLP are about `84–91%` of a decode
  step, and both sit at the memory roofline (DRAM `>80%`, SM compute `11–19%`,
  occupancy `75–97%`). The LM-head GEMV (`248320x2560`) is a pure DRAM-streaming
  case at ~1.5 ms; the `Q/QKV` (`8192x2560`) shape is memory-bound with a mild
  partial-wave tail. A `ROWS_PER_BLOCK` sweep on `Q/QKV` (4→6→8) was worse, so
  the lever for the medium shapes is data movement, not launch geometry.
- **The recurrent path is visible but not dominant.** `gated_delta_rule` is
  `1.10 ms/step` (8.7%) versus `5.64 ms/step` for the fused MLP pair. The 24
  linear-attention layers add extra decode projection pressure (QKV, Z, B, A,
  out_proj) — that is the main reason Qwen3.5 sits above the Qwen3-4B `~10.6ms`
  TPOT reference on the same GPU.
- **Prefill's bottleneck moved.** The old host-side per-token loop was the
  problem (67% CPU launch overhead at seq=128). After batching full-attention
  attention (Triton), then replacing per-token GDR launches with a single
  chunk-wise Triton pipeline per layer, the launch/copy/allocation churn
  collapsed (kernel launches `~295k → 278`, DtoD copies `~492k → 2`, alloc/free
  `~690k → 1.5k`), and prefill-heavy TTFT went from a `16.8s` baseline to `~222ms`
  at `(2048,1)` — effectively parity on the fixture. The remaining cost is
  `gated_delta_rule_prefill` plus batched GEMM.
- **A simple Triton GEMV is not a replacement for the handwritten kernel** on
  the real decode shapes (2–2.6× slower), even autotuned, so Triton is worth
  keeping as a reference/autotune surface, not a drop-in.

## Optimization history (conclusions)

The optimization log below is ordered newest-first. Each entry is the
optimization, the problem it fixed, and the measured result. The heavy per-step
nsys tables that produced these conclusions are folded away; only the durable
takeaways and key A/B numbers remain.

| # | Optimization | Problem fixed | Result |
| --- | --- | --- | --- |
| #9 | Restore the dedicated j-loop-parallel GDR decode kernel | The accuracy-parity refactor routed single-token decode through the 7-stage chunk-wise Triton pipeline, adding ~0.42 ms/step of launch overhead | Decode TPOT back to `11.78ms` (from `12.28ms`, −3.8%), matching the pre-refactor baseline |
| #8 | J-loop parallelism + state layout transpose + pass fusion in the GDR decode kernel | Kernel launched 32 blocks × 4 warps/block — too few warps to hide DRAM latency (latency-bound) | GDR decode microbench `37.1 → 14.8µs` (−60%); decode TPOT `12.53 → 11.77ms` (−6.1%) |
| #7 | Chunk-wise Triton GDR prefill | Per-token GDR launches + per-token orchestrating bookkeeping dominated prefill | `(2048,1)` TTFT `378 → 222ms`; fixed the `v_new` writeback bug; recurrent state exact vs FLA |
| #6 | Triton fused-recurrent GDR prefill (one kernel per layer) | Host-side per-token token loop for linear-attention prefill | First profile where prefill was not orchestration-bound; TTFT `~378ms`, `1.7×` vLLM |
| #3 | Batched Triton full-attention prefill | Legacy single-token HD256 attention loop (80% of GPU time at baseline) | seq128/512/2048 attention microbench `50.7µs / 365µs / 1.64ms`; full-attention prefill no longer the bottleneck |
| #1 | Batched `rms_norm_batched_offset_kernel` (`<<<seq_len, 256>>>`) | 8,193 per-token RMSNorm launches at seq=128 | RMSNorm kernel time at seq=2048 `38.7ms → 17.7µs`; `(2048,1)` TTFT `16.8s → 14.5s` |

Baseline at `#0` (2026-03-14): prefill `(2048,1)` was `16.8s` vs vLLM `222ms`
(76× slower), decode `(1,128)` was `12.55ms` vs vLLM `11.64ms` (+8%). The two
independent prefill problems were (1) CPU launch overhead — ~20k tiny kernel
launches in a per-token loop, and (2) no batched kernels for attention/recurrent
ops — even with zero launch overhead, 2048 tokens × 32 layers of single-token
kernels would have cost ~3.5s. Both are now addressed.
