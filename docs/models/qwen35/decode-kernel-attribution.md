# Qwen3.5 decode kernel attribution vs vLLM 0.27 (A100)

> **TL;DR:** nsys kernel-level attribution of the remaining serving gap (1×A100-40GB, upstream/main `8f455a18`, vLLM 0.27.0, 1024-token prompts, `--cuda-graph-trace=node`, steady-decode session capture, both engines re-captured on the same day). Five findings. (1) The GDN decode deficit is closed by #1054: `gated_delta_rule_decode_batch_kernel` now runs 48.5 µs per layer-step against vLLM's FLA `fused_recurrent` 42.7 µs, a 1.14× ratio. (2) The decode GEMM family is at per-kernel parity with vLLM on every shape — `gate_up` 82.5 vs 82.4 µs, the M=2560 family 33.5 vs 33.7 µs, `lm_head` 915 vs 921 µs, the prefill family 478 vs 454 ms — and the residual gap is vLLM's projection fusion, which issues one M=12288 GEMM per linear layer (fused qkv+z) and one M=10240 GEMM per full-attention layer (fused q+k+v) where we issue five separate projections. (3) Full-attention paged decode is the largest single block: 154.1 µs per layer-step against vLLM's `flash_fwd_splitkv` 64.9 µs plus a 6.9 µs combine kernel, because FlashInfer's batch-decode grid is fixed at `(batch, kv_heads)` — 4 CTAs at bs1, 64 at c16 — while FA2 splits the KV range into a `(m_blocks, splits, batch × kv_heads)` grid of 192 CTAs at c16. (4) A split-KV HD256 decode kernel now serves every decode bucket up to 16 and improves all three of bs1, c8 and c16 over two runs a side with zero failed requests (bs1 9.41/9.36 → 8.57/8.57 ms, c8 11.25/11.27 → 10.65/10.67, c16 13.06/13.08 → 12.80/12.82 mean TPOT; qps16 unchanged); a load-only control puts the c16 memory floor for the same access pattern at 54 µs per layer-step against the 154.1 µs the FlashInfer kernel spends and the 71.8 µs vLLM spends, so the rest of the c16 gap is arithmetic and needs a tensor-core kernel. (5) The decode GEMM alignment audit is clean: all eight shapes carry M and K on multiples of 128, so the #1046 class of defect has no second instance.
>
> **Last touched:** 2026-09

## Contract

- GPU 1× A100-SXM4-40GB (sm_80); model Qwen3.5-4B (`/mnt/data/models/Qwen3.5-4B`, hidden 2560, 32 layers = 24 linear + 8 full attention, 16 qo heads, 4 kv heads, head_dim 256, GQA 4, vocab 248320, intermediate 9216); greedy, seed 42, random dataset.
- PegaInfer upstream/main `8f455a18`, default serve flags (serial, policy off — kernel composition is the subject, scheduling is not).
- vLLM 0.27.0: FLASH_ATTN full attention + FLA Triton GDN + FlashInfer sampler, piecewise CUDA graphs.
- Capture: server under `nsys launch`/`start`/`stop` with `--cuda-graph-trace=node`; capture armed after readiness + warmup, stopped after the bench; `--export=sqlite`; `nsys stats --report cuda_gpu_kern_sum` and `cuda_gpu_kern_gb_sum`.
- Client for both engines: `vllm bench serve --backend openai --dataset-name random --random-range-ratio 0 --temperature 0 --ignore-eos --seed 42 --num-prompts 16 --random-input-len 1024 --random-output-len 256 --max-concurrency 16`, 0 failed requests on both sides.
- Absolute kernel times are node-trace inflated and are used for **composition and cross-engine ratios at the same workload**, never as TPOT claims. The captures predate the split-KV change in finding 4; wall TPOT on the captured tree was PegaInfer c16 `13.07 ms`, bs1 `9.41 ms`, and the current state is in the table below.

## Wall-clock state (default flags, 0 failed requests)

| cell | TPOT | ITL p99 | TTFT | output tok/s |
| --- | --- | --- | --- | --- |
| bs1 @1024 | 8.57 | — | 87 | — |
| c8 | 10.65 | 65.6 | 381 | 654 |
| c16 | 12.80 | 79.2 | 673 | 1021 |
| qps16 | 32.11 | 97.2 | 772 | 1134 |

These are the split-KV side of the finding-4 differential, so they are the state of the tree once that change lands; the same cells read 9.41 / 11.25 / 13.06 / 32.11 ms on the FlashInfer path.

## Findings

### 1. The GDN decode deficit is closed

| kernel | PegaInfer | vLLM 0.27 |
| --- | --- | --- |
| GDN decode per layer-step | `gated_delta_rule_decode_batch_kernel` 48.5 µs | `fused_recurrent_gated_delta_rule_packed_decode_kernel` 42.7 µs |
| instances per step | 24 | 24 |

#1054 (`0db99381`, merged 2026-09-18) carried the state slice in registers across the `kv_mem` reduction and took this kernel from 97.0 µs to 48.5 µs per layer-step. The remaining 1.14× ratio is worth about +0.14 ms/step at c16 and is no longer a first-tier lever.

### 2. The decode GEMM family is at per-kernel parity; the residual is projection fusion

The production `gemm_lt_tune` path covers eight decode shapes at every bucket in `[1,2,4,8,16,32]`:

| M | K | projection | launches/step |
| --- | --- | --- | --- |
| 32 | 2560 | `in_proj_b`, `in_proj_a` | 48 |
| 1024 | 2560 | full-attention `k_proj`, `v_proj` | 16 |
| 2560 | 4096 | linear `out_proj`, full `o_proj` | 32 |
| 2560 | 9216 | MLP `down_proj` | 32 |
| 4096 | 2560 | linear `in_proj_z` | 24 |
| 8192 | 2560 | linear `in_proj_qkv`, full `q_proj` | 32 |
| 18432 | 2560 | MLP `gate_up_proj` | 32 |
| 248192 | 2560 | `lm_head` over the padded selection width | 1 |

Every M and K is a multiple of 128, and both operands and the output carry a leading dimension equal to their row length, so the alignment class of defect fixed by #1046 has no second instance. N is the decode bucket and takes the values in `[1,2,4,8,16,32]`, which no alignment rule constrains. The cublasLt plan is consulted for every one of these 217 launches per decode step.

Head to head on the kernels both engines run, matched by grid shape:

| kernel (tile, grid) | shape | PegaInfer | vLLM 0.27 |
| --- | --- | --- | --- |
| `64x64_sliced1x2_64x5` `(288,1,1)` | gate_up M=18432 | 8576 × 82.52 µs | 8032 × 82.41 µs |
| `64x64_sliced1x2_64x5` `(128,1,1)` | M=8192 | 8640 × 38.21 µs | — |
| `64x64_sliced1x2_64x5` `(192,1,1)` | M=12288 (qkv+z fused) | — | 6120 × 51.85 µs |
| `64x64_sliced1x2_64x5` `(160,1,1)` | M=10240 (q+k+v fused) | — | 2040 × 43.74 µs |
| `64x64_sliced1x2_64x5` `(3878,1,1)` / `(3880,1,1)` | lm_head | 286 × 914.7 µs | 3 × 893.4 µs |
| `128x64_64x4` `(20,1,4)` | M=2560 family | 16768 × 33.46 µs | 16192 × 33.66 µs |
| `128x64_64x3` `(1940,1,1)` | lm_head (vLLM's separate pick) | — | 261 × 921.0 µs |
| `256x128_64x3` | prefill family | 478 ms | 454 ms |

`gate_up` is within 0.2% of vLLM, the M=2560 family within 0.6%, `lm_head` within 0.7%, and the prefill family within 5%. Within this family the two engines diverge on the class of kernel, not on the kernel: the totals are 1302 ms against 1071 ms over the window, and the grid table shows why. vLLM issues 24 fused qkv+z GEMMs and 8 fused q+k+v GEMMs per step where we issue 24 qkv + 24 z + 8 q + 8 k + 8 v. Fusing those two groups removes 40 launches per decode step and folds 16 small M=1024 GEMMs into a full-width one.

### 3. Full-attention paged decode is the largest single block, and the cause is grid shape

| engine | kernel | grid | block | µs per layer-step |
| --- | --- | --- | --- | --- |
| PegaInfer | FlashInfer `BatchDecodeWithPagedKVCacheKernel<kNone,2,1,8,32,4,1>` | `(16,4,1)` | `(32,4,1)` | 154.1 |
| vLLM 0.27 | `flash::flash_fwd_splitkv_kernel<256,64,64,4,…>` | `(1,3,64)` | `(128,1,1)` | 64.9 |
| vLLM 0.27 | `flash::flash_fwd_splitkv_combine_kernel<…,4,2,true>` | `(64,1,1)` | `(128,1,1)` | 6.9 |

`BatchDecodeWithPagedKVCacheDispatched` launches `nblks(padded_batch_size, num_kv_heads)` and never grows the grid with KV length, so the device sees 4 CTAs at bs1 and 64 at c16. FA2 instead splits the KV range and launches `(num_m_blocks, num_splits, batch × kv_heads)`. At c16 the KV traffic per layer-step is 67 MB, a 48 µs floor at the 1389 GB/s this card reaches on the `lm_head` GEMM; vLLM lands at 71.8 µs across its two kernels (67% of that floor) and we land at 154.1 µs (31%). At bs1 the whole layer-step is 4.2 MB of KV and the kernel is purely latency-bound: FlashInfer's 4 CTAs cost about 156 µs where FA2's split grid cost about 26 µs in the earlier bs1 capture.

### 4. A split-KV HD256 decode kernel now serves every decode bucket up to 16

FlashInfer's grid is fixed at `(batch, kv_heads)`, so the cheapest large win was to stop using its launch shape at the batch sizes where it starves the device. A register-streaming split-KV kernel replaces it for decode buckets at or below 16: one CTA per (KV split, kv head, request), each warp owning one query head of the GQA group and walking its share of the range one position at a time straight into registers, with no shared memory and no barrier. A second kernel merges the partials, one CTA per (request, query head).

Measured with the same client, same tree, two runs per side, zero failed requests:

| cell, mean TPOT | FlashInfer batch decode | split-KV decode | delta |
| --- | --- | --- | --- |
| bs1 @1024, out 256 | 9.41 / 9.36 ms | **8.57 / 8.57 ms** | −8.6% |
| c8 @1024, out 256 | 11.25 / 11.27 ms | **10.65 / 10.67 ms** | −5.4% |
| c16 @1024, out 256 | 13.06 / 13.08 ms | **12.80 / 12.82 ms** | −2.0% |
| qps16 | 32.11 / 32.16 ms | 32.11 / 32.10 ms | unchanged |

Each cell keeps the same output throughput ordering (c8 620 → 654 tok/s, c16 999 → 1021). bs1 moves from 12.6% behind vLLM 0.27 (8.32 ms) to 2.9% behind. `hf_golden_gate` passes on both fixtures (the standard one and `long`) with single-threaded test execution; the per-block deltas are unchanged from the FlashInfer path (mean 0.026–0.028 nats, max 0.13–0.17).

The gain shrinks as the batch grows because the kernel approaches its own arithmetic limit. A load-only control isolates that limit: with the attention arithmetic removed but the same grid, the same K/V addresses and the same bytes, the kernel reads the layer's 67 MB of KV in **54 µs at 1235 GB/s**, close to the 1389 GB/s this card reaches on the `lm_head` GEMM. The full kernel needs 124 µs at the same geometry, and vLLM's FA2 pair does it in 72 µs. So the c16 ceiling for any replacement is about 54 µs per layer-step, and the last 70 µs between the current kernel and that ceiling is arithmetic and per-position overhead.

Three structural attempts at it all landed at 124 µs or worse: four positions per lane to break the 8-deep FFMA chain and the five dependent shuffles (169–190 µs, so the extra registers cost more than the added parallelism buys), hoisting the page-table load out of the position loop (126 µs, so the compiler already had it), and staging the tiles through 64 KB of shared memory (117 µs, rejected separately for a softmax defect). Reaching vLLM's number needs tensor cores — `mma.m16n8k16` with the batch dimension packed into M so that a full 16-row tile of (request, query head) pairs shares one K tile.

One invariant is worth stating because it is silent when broken: an empty split must still write its partial for **every** query head. A warp owns exactly one head of the GQA group and the partial pointers are already offset to it, so a neutral write guarded by `warp == 0` leaves the other three heads holding whatever the previous layer left in the buffer. The symptom is a small per-position error that grows with the number of empty splits, and the gate catches it only because short contexts produce empty splits at all.

### 5. What is NOT the problem

- bs1 GEMM/GEMV: 7.42 ms/step of kernel busy against vLLM's ~7.1, both near the weight-read bandwidth floor for 8.41 GB of BF16 weights.
- Host/launch overhead: the captures show 91–92% GPU busy with only ~0.75 ms/step of inter-kernel gaps.
- Admission/batching shape at c16: `PEGAINFER_ITL_DEBUG=1` shows `decode_n` at 16/16 for the whole decode, and the captures agree.
- Decode-GEMM alignment: all eight shapes carry M and K on multiples of 128 (finding 2).
- Prefill: the `256x128_64x3` family and the `gdr_chunk_*` kernels are within 5% of vLLM. The window covers roughly 15 prefill steps for both engines, which is why per-step figures must be read against the whole-window totals.

## Open question

The PegaInfer window holds 270 steps where vLLM's holds 255 for the same 256-token outputs, at 24 GDN and 8 paged-decode launches per step on both sides. Dividing a window total by its own step count therefore flatters us by about 6% on every per-step row, and the prefill ramp (16 chunked-prefill steps against vLLM's shorter ramp) is the likely source. Per-step comparisons in this document are read against whole-window totals that are not exposed to it: `64x64_sliced1x2` 1302 ms against 1071 ms, M=2560 574 against 556, prefill 478 against 454, GDN 314 against 261, paged decode 333 against 145 (vLLM's split-KV kernel 131 plus its combine kernel 14).

## Improvement queue (ordered by expected value)

1. **A tensor-core split-KV paged decode for HD256.** Worth about 0.7 ms/step at c16 (154.1 µs against vLLM's 71.8 µs across its pair of kernels, times 8 layers) and it would extend the split kernel's wins to the buckets above 16; the register-streaming form already serving those buckets plateaus at 124 µs against a 54 µs memory floor. Either write the m16n8k16 kernel with the batch packed into M, or vendor flash-attention's sm80 hdim256 sources, which is the kernel vLLM actually runs (`Flash_fwd_kernel_traits<256, 64, 64, 4>`).
2. **Fuse the decode projections the way vLLM does.** One M=12288 GEMM per linear layer (qkv+z) and one M=10240 GEMM per full-attention layer (q+k+v) replaces 5 launches with 2 and removes 40 launches per decode step. Expected about 0.2–0.4 ms/step at c16; while bs1 is bandwidth-bound it should be close to neutral there.
3. **Exhaustive cublasLt candidate search.** `gemm_lt_tune` times at most the 16 algos `cublasLtMatmulAlgoGetHeuristic` returns; `cublasLtMatmulAlgoGetIds` plus `cublasLtMatmulAlgoInit`/`AlgoConfigSetAttribute` enumerates every tile, stage, split-K and swizzle the library can build. The per-kernel parity in finding 2 suggests little headroom in the shapes tuned today, so treat this as a sweep of the untuned prefill shapes rather than of the decode set.
4. **QPS16 TTFT and throughput.** These follow from total GPU step time; #1034 already took the scheduling side as far as it goes.

## Rejected leads

- **FlashInfer split-KV decode for HD256.** Wired through the tree's existing `decode_split_kv_launch<256>` template (64-token chunks, cap 64, buckets ≤ 8) it measured 10.03 ms against 9.77 ms for the same tree at bs1 @1024. The standalone work in finding 4 shows the direction is right and FlashInfer's own partitioning is what loses.
- **The split-KV kernel above bucket 16.** It wins at bs1, c8 and c16 and the margin shrinks with the batch as its own arithmetic cost takes over; the tensor-core item below is what would carry it to the larger buckets.
- **Four positions per lane at c16.** Intended to break the 8-deep FFMA chain and the five dependent shuffles per position, it measured 169–190 µs against 124.3 µs for one position per lane: the added registers and per-group bookkeeping cost more than the added instruction-level parallelism buys. Register pressure is not the cause either — the build reports 8 bytes of spill.
- **Hoisting the page-table load out of the position loop.** One lookup per page instead of one per position moved c16 from 124.3 µs to 126.2 µs, so the compiler had already taken that dependency off the critical path.
- **Admission waves at c16 and a 5.9 ms/step bs1 host gap.** Both were readings from polluted or partial captures and were retracted during review; the clean captures in this document supersede them.

## Claim boundary

Single runs per capture, two runs per side per HTTP cell, one GPU, node-trace-inflated absolute times used for composition and cross-engine ratios only. Kernel-family groupings strip template arguments; aggregate step counts are inferred from kernel instance counts (24 GDN and 8 paged-decode launches per step). The vLLM and PegaInfer captures were taken the same day on the same client command against their own default flags, so the comparison holds for kernel composition and per-kernel ratios, not for a wall-clock parity claim. The split-KV results behind finding 4 are same-tree differentials with the split threshold as the only variable, two runs a side; those rows are serving measurements, while the kernel-level numbers behind them (the split sweep, the load-only floor, the per-variant timings) are direct kernel-path measurements from a standalone harness against the same paged layout, not serving measurements.
