# Qwen3.5 Tensor Parallelism

> **TL;DR:** Qwen3.5 tensor parallelism (TP) reuses Qwen3's controller/worker runtime rather than building a second parallel stack. Dense full-attention + MLP are sharded first; linear-attention/GDR weights and state are replicated at that stage, then sharded as a separate follow-on so failures stay attributable. TP is eager-only: `tp_size > 1` with CUDA Graph enabled fails closed before serving. `TP=2` is the first validated degree and now passes short/long HF logits gates, scheduler e2e, and a real multi-turn serving gate. TP never all-reduces recurrent or conv state.
>
> **Last touched:** 2026-08

## Goal

Add tensor-parallel support for `Qwen3.5-4B` by reusing the Qwen3 TP runtime. `TP=2` is the first validation target, not an architectural limit; the implementation is degree-parametric where the model dimensions divide cleanly. Unsupported or indivisible degrees must fail closed before model load.

## Reused Qwen3 TP shape

- controller/worker broadcast execution model
- `RequestId` request identity and the coarse-grained prefill/decode/unified/drop step protocol
- rank-local worker-owned model state, CUDA context, cuBLAS, and NCCL resources
- hidden all-reduce after row-parallel projections
- replicated embedding / tied `lm_head` as the first-pass simplification

Qwen3.5-specific design work stays on model geometry and state ownership: hybrid layer layout, gated q projection, linear-attention conv state, and GDR recurrent state.

## Boundaries

Out of scope: multi-node TP, data parallelism, pipeline parallelism, vocab-parallel embedding/`lm_head`, and TP-aware prefix-cache / recurrent-state snapshots. TP CUDA Graph capture/replay is also a separate follow-up (see below).

## Why dense first, linear attention (GDR) second

Qwen3.5 has two separable TP problems.

- **Dense** is already proven by Qwen3: full-attention head sharding, local KV heads, MLP intermediate sharding, all-reduce after row-parallel projections, worker-thread CUDA/NCCL execution.
- **Linear attention** is Qwen3.5-specific: conv state and GDR recurrent state are long-lived request state, the current GDR AOT kernels are built for the global value-head shape, and `DropRequest` cleanup plus re-admission must preserve rank-local recurrent-state boundaries.

If dense TP and GDR TP land together, failures are hard to attribute. Dense TP first narrows correctness debugging to runtime + dense sharding; GDR TP then isolates the recurrent-state contract. This makes no speedup promise by itself — report a matched dense-TP TP2 versus sharded-GDR TP2 A/B before making any performance claim.

## Model geometry

Qwen3.5-4B: 32 layers (24 linear attention + 8 full attention at indices `3, 7, 11, 15, 19, 23, 27, 31`), `hidden_size = 2560`, `intermediate_size = 9216`, tied embedding/`lm_head`, `vocab_size = 248320`.

Full attention: 16 q heads, 4 kv heads, `head_dim = 256`, `q_dim = 4096`, `kv_dim = 1024`. The q projection includes an output gate, so gated q output dim is `2 * q_dim = 8192`.

Linear attention: 16 key heads / `linear_key_head_dim = 128`, 32 value heads / `linear_value_head_dim = 128`; `linear_qkv_dim = 8192`, `linear_z_dim = 4096`. Recurrent state per linear layer is `[32, 128, 128]` f32; conv state is `8192 * (kernel_dim - 1)` bf16.

## Partition contract

For any candidate `tp`, require `num_attention_heads % tp == 0`, `num_key_value_heads % tp == 0`, and `intermediate_size % tp == 0`. GDR sharding additionally requires `linear_num_key_heads % tp == 0` and `linear_num_value_heads % tp == 0`.

Local dims are the global dims divided by `tp` (local q/kv heads, local `q_dim`/`kv_dim`, `local_gated_q_dim = 2 * local_q_dim`, `local_intermediate`, local linear key/value heads and their dims, local recurrent/conv state).

**Gated `q_proj` is the one non-obvious slicing rule.** Qwen3.5's gated q projection must be sharded by head-local q/gate pairs: each rank owns a contiguous query-head range and receives both that head's q rows and its gate rows. Do not reuse a naive contiguous row shard if the physical layout can split q rows from their gate rows. MLP gate/up row sharding and down column sharding need explicit reconstruction/layout tests.

## Dense TP: replicate linear attention

Shard full-attention `q_proj`/`k_proj`/`v_proj`/`o_proj`, full-attention KV over local KV heads, and MLP `gate_proj`/`up_proj`/`down_proj`. Replicate embedding/tied `lm_head`, all linear-attention weights, conv state, GDR recurrent state, and the existing GDR kernels/scratch shapes.

Execution: full-attention runs local q/k/v + local attention + local `o_proj`, then all-reduces the hidden; MLP runs local gate/up + activation + local `down_proj`, then all-reduces the hidden; linear attention runs the full layer on every rank and updates a full local recurrent-state copy — never all-reduce replicated linear-attention output.

State ownership: the scheduler owns request admission, identity, logical page allocation, streaming handles, sampling params, generation counters, and finish bookkeeping; rank workers own rank-local model shards, physical KV, decode buffers, and recurrent/conv state. Rank 0 is not special for state mutation; it follows the same worker command protocol and returns artifacts for scheduler-side result resolution. All workers observe the same ordered `RunPrefillChunks` / `RunDecodeStep` / `DropRequest` / `Shutdown` commands.

## CUDA Graph is excluded under TP

`TP > 1` is eager-only. Enabling CUDA Graph with `tp_size > 1` must fail closed with an explicit startup/configuration error before serving requests. TP graph capture is a follow-up because Qwen3.5 graph state includes recurrent slots, slot compaction, padding slots, and NCCL ordering questions.

## Eager unified prefill + decode

When active decode and scheduled prefill coexist, TP uses the same `plan::build_next_plan` decision as the single-GPU scheduler and emits one start-gated `RunUnifiedStep` to every rank. The canonical plan carries ordered prefill and decode items plus separate seeds: the scheduler selects the decode seed first and the prefill seed second to preserve prior RNG order, and workers execute prefill first and decode second on every rank. Rank 0 returns `TpUnifiedResult`; every non-primary rank returns `Ack`. The order is the collective-order contract.

State and artifacts are keyed by `RequestId`:

- worker-local KV/conv/recurrent state is found, created, promoted, and released by `RequestId`;
- the primary worker returns prefill/decode artifacts carrying their `RequestId`, and the scheduler resolves them by ID, rejecting unknown, duplicate, or missing results instead of trusting returned row position;
- finish, explicit drop, cancellation, and client disconnect broadcast the same `DropRequest(RequestId)` lifecycle command to every rank;
- each worker returns `DropAck { existed }`, and every controller-side drop carries a `DropExpectation::{MustBeAbsent, MustExist}` derived from scheduler-owned lifecycle state, not from a separate `worker_state_materialized` flag;
- cancellation before the first successful prefill dispatch requires `MustBeAbsent` and an exact-rank all-false `DropAck` set; partially prefetched, active, disconnected, and completion-candidate requests require `MustExist` and an exact-rank all-true set. Uniformity alone is insufficient: all-false for `MustExist`, all-true for `MustBeAbsent`, or mixed true/false proves lifecycle divergence and poisons the whole TP executor even if the drop leaves every rank absent.

Controller response collection validates an exact rank set, not just a message count: every response rank must be in `0..world_size` and appear exactly once. Ping requires `Ack` from every rank; drop requires `DropAck` values matching the controller's `DropExpectation`; prefill/decode/unified require exactly one matching typed result from rank 0 plus `Ack` from every non-primary rank. Missing, duplicate, out-of-range, wrong-variant, or mixed drop-existence responses are protocol failures; if workers may already have mutated state, such a failure poisons the complete TP replica.

Successful request completion has a fail-closed commit boundary. After computing a terminal artifact, the scheduler keeps the request unresolved in a local completion candidate and withholds its user-visible terminal events (an EOS candidate buffers only `Finished`; a length-limited candidate buffers the final `Token` plus `Finished`; completion on the first prefill token, including `max_tokens <= 1`, uses the same rule). It first requires a valid `MustExist` all-rank all-true `DropAck`, then removes the logical request, and only then publishes the buffered events in order — so a client that observes `Finished` can rely on consistent rank-local cleanup. If completion drop fails, the scheduler publishes neither the buffered token nor `Finished`, poisons the replica, and keeps the candidate unresolved for terminal error fan-out. This boundary makes only successful termination atomic, not the whole streamed response transactional.

Cancellation cleanup is a scheduler-tick boundary (`drain -> prune -> publish load -> admission -> plan`), not a late planning cleanup. Cancelled residents no longer count as running or hold capacity, and a cancellation racing past the prune check is still retired by the existing token-send failure path.

Artifact alignment follows the ordered plan rather than returned row position: prefill expects artifacts only for `finish_prefill == true` IDs and preserves explicit absence for non-final rows; decode expects one artifact for every row. Unknown, duplicate, or missing IDs are fatal after execution. This TP adapter contract does not require the TP1/single-GPU logits-and-sampling path to adopt sparse artifacts.

Controller-provable structural errors are rejected before dispatch without poisoning. Worker-local existence/phase/materialization/capacity mismatches are not protected by an all-rank validation barrier and are replica-fatal, as is any CUDA/NCCL, response-set, artifact, or lifecycle failure after execution is released. On fatal failure the scheduler closes and drains submissions, emits exactly one terminal `Error` for every unresolved request, publishes an idle load snapshot, and begins whole-executor teardown; it does not retry per-request drops or claim rank-local cleanup after poison. The scheduler allocates a TP `RequestId` before the first prefill command materializes worker state, so cancellation in that interval legitimately requires `MustBeAbsent`.

The fixed 16-device Triton AOT handle table remains. Before model loading or worker launch, TP startup validates every requested logical CUDA ordinal against that supported range; dynamic handle allocation is not a prerequisite. `tp_size > 1` with CUDA Graph requested continues to fail closed.

## Sharded linear-attention / GDR

A separate follow-on shards the 24 linear-attention layers to true TP and additionally requires `linear_num_key_heads % tp == 0` and `linear_num_value_heads % tp == 0`; unsupported degrees and local kernel shapes fail before model loading. Shard every head-indexed surface by the local key/value-head ranges: `in_proj_qkv` (preserving local q/k/value channel layout), `in_proj_z`, `in_proj_b`, `in_proj_a`, `conv1d_weight`, `dt_bias`, `A_log`, `out_proj` input columns, conv state, recurrent state, GDR scratch, and intermediate buffers.

The `norm_weight` stays deliberately replicated because it is shared by every local value head — it is not a head-indexed state or collective surface. Embedding and tied `lm_head` also stay replicated.

Each rank runs local projections, convolution, GDR prefill/decode kernels, gated RMSNorm/output-gate work, and local `out_proj` against local dimensions. The only linear-attention collective is the hidden all-reduce after `out_proj`.

**Non-negotiable invariant:** never all-reduce GDR recurrent state or conv state. These are owned by rank-local request state for their full lifetime.

Validation for GDR sharding must include loader reconstruction/layout tests, local AOT-kernel shape validation, rank-local allocation checks, the dense-TP and unified-step regression gates, short and long TP2 HF replay, cleanup without stale request state, and a matched dense-TP TP2 versus sharded-GDR TP2 HBM/latency/throughput report (evidence, not a speedup threshold).

## vLLM reference contract

Use vLLM's `Qwen3NextForCausalLM` / `QwenGatedDeltaNetAttention` as the reference, not code to copy mechanically: GDN state shape depends on `tp_size`; q/k/v/z are tensor-parallel column projections; `out_proj` is row-parallel and reduces back to full hidden; `dt_bias` and `A_log` are sharded over local value heads; b/a projections are local-value-head aware (some quantized paths may replicate small projections and slice locally); GDR prefill/decode kernels consume local head/state shapes. PegaInfer-specific work remains: worker-owned rank-local recurrent state, `RequestId` lifecycle, request-state removal and re-admission, `DropRequest` cleanup, and fail-closed kernel-shape validation.

## Key fixes and gotchas

These defects were found during the first TP2 bring-up and are worth remembering for any multi-GPU use of the Qwen3.5 path.

### Gated q projection layout

The original assumption was that `q_proj.weight` rows were physically `[all q rows][all gate rows]`. The real Qwen3.5 kernel contract is per-head interleaved: `[head0 q][head0 gate][head1 q][head1 gate]...`. For TP2 the fixed loader preserves contiguous head-interleaved ranges — rank 0 loads rows `0..4096`, rank 1 loads `4096..8192`. The old loader gathered local q rows and gate rows separately, then rebuilt a `[q][gate]` fused matrix, which corrupted the first full-attention contribution and failed the TP2 HF gate from prefill position `0`.

### Per-device Triton AOT handles

With two CUDA devices, the generated GDR Triton AOT C stubs could not cache `CUmodule` / `CUfunction` in process-global state — the rank that loaded a GDR kernel first could leave the other rank with an invalid function handle. The generated stubs now cache module/function handles per CUDA device ordinal, and fail closed before indexing the fixed per-device handle tables if `cuCtxGetDevice` returns an ordinal outside the table size (guards against out-of-bounds writes on high CUDA ordinals) while preserving the static-table implementation.

### Worker-local NCCL setup

NCCL comms are initialized inside rank worker threads after each worker binds its CUDA context and initializes thread-local cuBLAS. Creating comms on the controller thread and moving them into workers caused invalid-handle symptoms and hangs. This matches the design contract: TP workers own rank-local CUDA/NCCL execution resources.

### Current-main API compatibility

Rebasing onto current `main` required preserving the TP execution boundary while adopting newer shared contracts:

- Hybrid batch decode now builds `Vec<&mut RecurrentState>` from graph-owned slots before entering the common linear-attention helper, keeping request state in place.
- `pegainfer_sample::select_batch` now requires request-local sampling steps. TP still samples one row at a time and has no request-local sampling counter, so it passes step `0` and retains its existing per-row `sample_seed` offset. Do not substitute batch row indices for request-local steps — that would make seeded output depend on batch composition.
- Qwen3.5 launch/tests use the current `EngineLoadOptions` surface (the removed `enable_prefill_profile` field is no longer supplied).
- TP scheduler tests set `GenerateRequest::data_parallel_rank` to `None` because TP is TP-only, not DP.
- Synthetic TP config/loader fixtures include `tie_word_embeddings`, matching the current `Config35` contract.
- TP2 short/long HF gates use `Golden::load_for(model_path, long)` and pass the complete `Golden` to metadata validation.

## Validation evidence

TP2 correctness and lifecycle coverage (real weights, two CUDA devices):

- **Short HF logits gate** — sequential eager: `108` positions, mean `0.0258`, p99 `0.0801`; batched eager: `72` positions, mean `0.0257`, p99 `0.0809`.
- **Long HF logits gate** — prompts `4097` and `8192`, sequential eager: `18` positions, mean `0.0232`, p99 `0.0792`.
- **Scheduler e2e** — context-window rejection, greedy/logprobs paths, sequential, repeated, and concurrent mixed greedy/sampling requests, consumer drop, post-drop scheduler health.
- **HTTP serving smoke** — `/v1/models`, non-streaming and streaming `/v1/completions`, concurrent completions, finite logprobs, chunked prefill forced with `max_prefill_tokens=1`, and TP+CUDA Graph fail-closed startup.
- **Real multi-turn serving gate** — a 2x RTX 3090 TP2 server completed `12/12` dependent conversations and `44/44` turns at client concurrency 4 over server capacity 2, then admitted and completed another `4/4` conversations and `8/8` turns without restart, with clean shutdown; full pins, commands, and raw JSON are in [docs/benchmarks/qwen35-tp2-phase2a-multiturn.md](../../benchmarks/qwen35-tp2-phase2a-multiturn.md).
- TP1 regression gates pass after the TP2 additions.

Test knobs: `PEGAINFER_TEST_TP_DEVICES` (comma-separated TP2 CUDA ordinals, default `0,1`; requires two distinct ordinals), `PEGAINFER_TEST_MODEL_PATH` (real Qwen3.5 weights for HF/scheduler/serving tests), and `PEGAINFER_TEST_FRONTEND_MODEL_PATH` (optional tokenizer/config metadata path, defaults to the model path). TP2 tests remain ignored by default because they require two CUDA devices, NCCL, and real weights; long TP2 HF replay is memory-sensitive.

## Serving scope: prompt echo is unsupported

Qwen3.5 serving does not support `echo=true` — both the legacy and stepped vLLM bridges submit `echo: false`, so it has never been part of the HTTP serving contract. Direct engine callers can still construct the shared request type with `echo=true`, but the Qwen3.5 scheduler rejects those requests immediately after queue drain and cancellation pruning, before load accounting, capacity admission, backend-state allocation, or TP command dispatch. Qwen3.5 no longer emits `TokenEvent::PromptTokens` after prefill; the variant remains in `pegainfer-frontend` for other model lines but is sealed off from the Qwen3.5 scheduler path.

## Tracked follow-up

- Shard linear-attention/GDR state (see "Sharded linear-attention / GDR") without weakening the unified-step lifecycle and ID contracts.
- Promote any stable contract change discovered here back into this doc.
- Decide whether the server CLI should accept arbitrary TP device ordinals instead of only `0..tp_size`.
- TP CUDA Graph capture/replay, TP-aware prefix caching, and recurrent-state snapshots remain separate follow-up RFCs.

Note: `tp_executor.rs` was refactored into `tp_executor/{executor,worker,responses}.rs` — see [refactor-crate-structure.md](refactor-crate-structure.md).

## References

- `docs/models/qwen3/tp-design.md`
- `pegainfer-qwen3/src/config.rs`, `pegainfer-qwen3/src/executor.rs`
- `pegainfer-qwen35/src/config.rs`, `weights.rs`, `recurrent_state.rs`, `batch_decode.rs`
- vLLM `Qwen3NextForCausalLM`, `QwenGatedDeltaNetAttention`
