# Qwen3.5-4B serving vs vLLM 0.27.0 on A100-40GB

> **TL;DR:** First-pass gap map against the **latest** vLLM (0.27.0) on 1×A100-40GB, upstream PegaInfer `70a600b7`. Zero failed requests on both engines. PegaInfer wins 1024-token bs1 TTFT (`78 vs 118 ms`) but loses everywhere else: bs1 decode TPOT grows with context on PegaInfer (`8.74→9.82 ms` 1→1024 tokens) while vLLM stays flat (`8.31→8.32`); batch TPOT is +24–35% at c4–c16; the worst single gap is ITL p99 at 1024/256 c8 (`65 vs 12 ms`) — prefill invading decode; sustained open-loop TTFT collapses (`867 vs 218 ms` at QPS16). Single run per cell — attribution evidence, not a retained record.
>
> **Last touched:** 2026-09

## Setup

| Field | Value |
| --- | --- |
| Date | 2026-09-06 |
| GPU | 1× NVIDIA A100-SXM4-40GB (sm_80), CUDA_VISIBLE_DEVICES=1 |
| PegaInfer | upstream/main `70a600b7` (`refactor(qwen35): split TP-shard helpers into scheduler/tp.rs (#968)`), release build, `PEGAINFER_CUDA_SM=80`, default serve flags |
| vLLM | `0.27.0` (vllm-omni venv; torch 2.13.0, FlashInfer 0.6.16.post3, flash-linear-attention 0.5.2), JIT via nvcc CUDA 13.0 |
| Model | `/mnt/data/models/Qwen3.5-4B` (same files as the #469 5090 run) |
| vLLM serve | `--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 --no-enable-prefix-caching`; V1 engine, chunked prefill on (default), FLASH_ATTN full-attn + Triton/FLA GDN prefill kernel, FlashInfer sampler, piecewise CUDA graphs to 512 |
| PegaInfer serve | `target/release/pegainfer --model-path … --served-model-name Qwen3.5-4B --port 8000` (defaults) |
| Client | `vllm bench serve --backend openai --endpoint /v1/completions --dataset-name random --random-range-ratio 0 --temperature 0 --ignore-eos --seed 42`, `--num-prompts = --max-concurrency` per fixed cell |
| Artifacts | `~/ab_pega/`, `~/ab_vllm027/` on the bench host (one result JSON + client log per cell) |

## Results (single run per cell)

| Cell | TPOT OI / vLLM ms | Δ | out tok/s OI / vLLM | Δ | TTFT OI / vLLM ms | ITL p99 OI / vLLM ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1/256 c1 | 8.74 / 8.31 | +5% | 114 / 120 | −5% | **16.3** / 20.0 | 10.0 / 9.0 |
| 1/256 c16 | 12.13 / 9.75 | +24% | 1236 / 1598 | −23% | 218.9 / 70.9 | 13.2 / 11.5 |
| 1024/256 c1 | 9.82 / 8.32 | +18% | 99 / 114 | −13% | **77.6** / 118.2 | 10.8 / 9.5 |
| 1024/256 c4 | 11.29 / 9.08 | +24% | 328 / 403 | −19% | 222.9 / 220.2 | 11.7 / 10.6 |
| 1024/256 c8 | 12.23 / 9.88 | +24% | 580 / 713 | −19% | 370.9 / 337.7 | **65.5** / 12.0 |
| 1024/256 c16 | 14.87 / 11.00 | +35% | 901 / 1212 | −26% | 668.6 / 539.4 | 81.4 / 83.3 |
| QPS8 (1024/128) | 22.30 / 13.87 | +61% | 673 / 761 | −12% | 154.9 / 131.2 | 85.3 / 69.3 |
| QPS16 | 36.74 / 23.60 | +56% | 1024 / 1404 | −27% | **867.2** / 218.2 | 101.3 / 93.4 |

Both engines completed every cell with zero failures. PegaInfer observed avg input ≈974–985 vs requested 1024 (known random-dataset tokenizer artifact from #469).

## Where Qwen3.5 is weak, ranked

1. **ITL p99 stall at moderate concurrency (1024/256 c8: 65 vs 12 ms).** Prefill work invades the decode batch; vLLM's default chunked prefill + overlap scheduling keeps decode ITL flat until saturation. This is the #470 pattern re-confirmed against the newest baseline. Lever: make the #715 unified prefill-overlap lane default-eligible for qwen35, add admission coalescing (the gemma4 `PEGAINFER_ADMIT_COALESCE_MS` pattern), and shrink mixed-step chunk size while decode is active.
2. **Decode TPOT is context-dependent at bs1 (+12% from 1→1024 tokens; vLLM +0.1%).** vLLM's FLA GDN decode + FLASH_ATTN paged decode keep step time flat with context; ours grows ~1.1 ms/step at 1024 ctx. Lever: nsys direct bs1 A/B at 1 vs 4096 ctx, kernel-table diff over the 8 full-attn layers (decode attention tuning history is sm_120/5090-centric, not sm_80) and the GDN decode kernel vs FLA.
3. **Batch decode slope (+24% TPOT at c4/c8, +35% at c16).** Ours and vLLM both pay for batch, ours faster. Combined GPU-step + host contribution is unattributed on this stack — run the #469 follow-up per-step trace (queue wait, step type, batch size, send overhead) before kernel work. Also re-run decode cublasLt bucket tuning on sm_80.
4. **Sustained open-loop TTFT collapse (QPS16: 867 vs 218 ms).** Same admission/chunk policy family as (1); needs the #470 matrix rerun with overlap enabled.
5. **Missing product surfaces vs this baseline** (not exercised by the numbers above): hybrid prefix caching is vLLM default-on and our #257 joint KV/recurrent/conv snapshot is still open; vLLM carries MTP/speculative decode for the hybrid family while our DFlash (#434) is still opt-in and default-off.

Where we win: 1024-token bs1 TTFT is 34% better (`77.6 vs 118.2 ms`) and 1-token bs1 TTFT is 19% better — the direct prefill path and cold-start admission are genuinely fast; protect these while fixing the batch path.

## Claim Boundary

- Single run per cell (not median-of-3), one GPU, one host, one seed: this is a **gap map / attribution evidence**, not a retained record or parity claim. The retained record remains `qwen35-4b-serving-vllm-rtx5090-2026-07.md`.
- vLLM ran in a vllm-omni-patched venv (allocator/inductor patches; core V1 serving path). Prefix caching disabled on both sides; spec decode off on both sides.
- Different GPU generation than the #469 record (sm_80 vs sm_120); TPOT/TTFT numbers are not comparable across the two docs — only the gap shape is.
- No direct (in-process) diagnostic was run in this pass; the c16 host-vs-GPU split from #469 is unverified on this stack.

## Follow-Up

1. Repeat the three worst cells (1024/256 c8/c16, QPS16) as median-of-3 to promote the doc to a retained record.
2. Per-step serving trace fields (roadmap "Attribute the #469 HTTP gap") on 1024/256 c16 to split host vs GPU before touching kernels.
3. nsys direct bs1 decode at 1 vs 4096 ctx for the context-dependent TPOT (weakness 2).
