# Qwen3.5 decode kernel attribution vs vLLM 0.27 (A100)

> **TL;DR:** nsys kernel-level attribution of the remaining serving gap (1×A100-40GB, upstream/main+`#1033`/`#1034` lineage, vLLM 0.27.0, 1024-token prompts, `--cuda-graph-trace=node`, steady-decode session capture). Four findings: (1) at c16 our average decode batch is **~8 vs vLLM's 16** — the #469 admission waves are still there and halve batch efficiency; (2) at bs1 our step is **3.9 ms kernels + 5.9 ms host gap** (vLLM: 8.2 + 0.1) — the bs1 loss is CPU-side, not kernel-side; (3) per layer-step our GDN decode kernel is 97.5 µs vs vLLM's FLA `fused_recurrent` 42.9 µs, and FlashInfer `BatchDecodeWithPagedKVCache` 169 µs vs `flash_fwd_splitkv` 64.6 µs — at half the batch, so per-token ~4× worse; (4) two unattributed per-step GEMM shapes (`cutlass_75_tensorop align1` 2.17 ms and `gemv2T` 964 µs) cost ~1.5 ms/step and need caller identification.
>
> **Last touched:** 2026-09

## Contract

- GPU 1× A100-SXM4-40GB (sm_80); model `/mnt/data/models/Qwen3.5-4B`; greedy, seed 42, random dataset (1024-token prompts).
- PegaInfer at `00f0088` + overlap-aware budget (`fb16fc15` lineage), default serve flags (serial, policy off — kernel composition is the subject, not scheduling).
- vLLM 0.27.0 (vllm-omni venv, FLASH_ATTN + FLA GDN + FlashInfer sampler, piecewise CUDA graphs).
- Capture: server under `nsys launch`/`start`/`stop` sessions with `--cuda-graph-trace=node`; capture armed after readiness + warmup, stopped after the bench; `--export=sqlite`; `nsys stats --report cuda_gpu_kern_sum`.
- Absolute times are node-trace inflated — used for **composition and cross-engine ratios at the same workload**, not as TPOT claims. Wall TPOT from the HTTP runs: PegaInfer bs1@1024 `9.82 ms`, vLLM `8.32`; c16 `14.26` vs `11.00`.

## Findings

### 1. Admission waves: average decode batch ~8 vs 16 (scheduler-side)

Same 16×1024/256 workload: aggregate decode steps implied by the per-layer-per-step GDN kernel — PegaInfer `12,576 / 24 layers = 524` steps vs vLLM `6,120 / 24 = 255`. With 4,096 decode tokens that is an average batch of **~7.8 vs 16.0**. Our scheduler admits in waves and lets the batch drain before refilling; vLLM's continuous admission keeps all 16 slots decoding. This is the surviving #469 finding and it multiplies the cost of every other inefficiency.

### 2. bs1: the step is 60% host gap (CPU-side)

bs1 @1024 ctx: PegaInfer kernel sum `3.92 ms` vs wall TPOT `9.82 ms` → **~5.9 ms/step host gap**; vLLM kernel sum `8.23 ms` vs wall `8.32 ms` → `0.09 ms` gap. Our bs1 kernels are *faster* than vLLM's (GEMM 2.6 vs 7.1 ms — ours is near the bandwidth floor; theirs is not), so the entire bs1 TPOT deficit is inter-kernel gap: launch structure, per-step sync, or scheduler wakeups. Next probe: `cuda_api_sum` + gap analysis on the bs1 trace; check what the decode graph does not cover at bs1 (sampling/logprobs D2H, rope cache, KV commit).

### 3. Decode attention + GDN kernels: 2.3–2.6× per layer-step at half batch

| kernel family | PegaInfer (batch ~8) | vLLM (batch 16) | per layer-step |
| --- | --- | --- | --- |
| GDN decode | `gated_delta_rule_decode_batch_kernel` 97.5 µs ×24/step | `fused_recurrent_gated_delta_rule_packed` 42.9 µs ×24/step | **2.3×** (per-token ~4.6×) |
| full-attn decode | FlashInfer `BatchDecodeWithPagedKVCacheKernel` 169 µs ×8/step | `flash::flash_fwd_splitkv_kernel` 64.6 µs ×8/step | **2.6×** (per-token ~5×) |

Half the batch explains ~2×; the rest is kernel efficiency on sm_80 — neither kernel has an A100 tuning pass (decode tuning history is sm_120/RTX 5090).

### 4. Unattributed per-step GEMM shapes (~1.5 ms/step, PegaInfer-only)

- `cutlass_75_tensorop_bf16_s1688gemm_bf16_128x128_tn_align1`: **2.17 ms × 239 instances** — an sm_75-era kernel at alignment 1; one instance is ~15% of a decode step. Caller unknown (suspect output projection / lm_head path over the padded selection vocab).
- `gemv2T_kernel_val<...128...>`: **964 µs × 256 instances** — a matrix-vector shape recurring once per aggregate step while the batch is >1.
- vLLM shows one comparable ~0.92 ms/step kernel (their lm_head), so the delta to chase is the align1 kernel plus the GEMV-vs-GEMM shape choice.

### 5. What is NOT the problem

- bs1 full-attn paged decode: ours ≈ vLLM at 1024 ctx (0.20 vs 0.21 ms/step family time). The bs1 attention path is fine; the ctx-dependent TPOT growth measured earlier (8.74 → 9.82 ms from 1 → 1024 tokens) is dominated by the host gap, not attention.
- Prefill: not captured in this pass (decode attribution only).

## Improvement queue (ordered by expected value)

1. Fix admission waves (keep the decode batch full) — pure scheduler; multiplies the value of everything below.
2. Identify and re-shape the `align1` 2.17 ms kernel and the per-step `gemv2T` (likely lm_head / logits path) — possibly a pad/align fix.
3. Close the bs1 5.9 ms host gap (trace `cuda_api_sum`, find the per-step sync).
4. Retune decode GEMM buckets + GDN/full-attn decode kernels on sm_80 (largest effort; compare against FLA `fused_recurrent` and `flash_fwd_splitkv` as references).

## Claim boundary

Single runs per capture, one GPU, node-trace-inflated absolute times (composition/ratio claims only), kernel-family grouping with template args stripped; the polluted first `p_bs1_ctx1` capture was discarded — the ctx1-vs-1024 comparison awaits a clean recapture. Aggregate-step counts are inferred from kernel instance counts (24 GDN layers/step, 8 full-attn layers/step).
