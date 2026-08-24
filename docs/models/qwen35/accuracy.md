# Qwen3.5 Accuracy

> **TL;DR:** Qwen3.5 correctness is guarded by short and long HF-backed logits goldens (`tests/hf_golden_gate.rs`, `test_data/qwen35-{size}-hf-golden.safetensors` / `-hf-long-golden.safetensors`). The HF fixtures use `AutoModelForCausalLM` with `use_cache=True` / `past_key_values`, so they match pegainfer's prefill + decode shape. The gate is size-portable: it derives a `0.8b`/`2b`/`4b`/`9b`/`27b` fixture key from the pointed model's config content, and all five sizes ship committed short + long fixtures. Full GSM8K 8-shot matches the HF baseline within `0.15` percentage points. The older exact-text `test_data/Qwen3.5-4B.json` and its regeneration test are retired; `e2e_scheduler` stays a scheduler liveness/integration check that also gates model-wide collapse.
>
> **Last touched:** 2026-08. Current accuracy command (crate-local, needs an absolute `PEGAINFER_TEST_MODEL_PATH`): `cargo test --release -p pegainfer-qwen35 --test hf_golden_gate -- --nocapture`. Run `e2e_scheduler` only when scheduler request-flow behavior changes.

## Goal

- External truth source: Hugging Face Transformers Qwen3.5 in `eval()` mode with
  `use_cache=True` / `past_key_values` for the logits gate.
- Short-term success: the short and long HF logits gates stay green under the
  calibrated regret/mean/p99 tolerances, and GSM8K 8-shot stays within the HF
  baseline band.
- Debugging success: any future prompt-level drift is either eliminated or
  explained by a recorded numeric tolerance.

## Current state

- The reusable debugging method lives in [../../playbooks/accuracy-parity-playbook.md](../../playbooks/accuracy-parity-playbook.md).
- `pegainfer-qwen35/tests/hf_golden_gate.rs` checks pegainfer logits against pinned HF bf16 `past_key_values` oracles, a short + long pair per committed size (`0.8b`, `2b`, `4b`, `9b`, `27b`).
- `pegainfer-qwen35/tests/e2e.rs`, `tests/regen_test_data.rs`, and `test_data/Qwen3.5-4B.json` are retired — they were exact-text PegaInfer self-baselines, not HF accuracy gates.
- `pegainfer-qwen35/tests/e2e_scheduler.rs` still loads the model and exercises sequential, repeated, concurrent, and consumer-drop scheduler paths, but no longer reads an exact-text fixture; it also folds in a model-wide collapse net (see below).
- A broader PegaInfer-owned rand/hash corpus (#186) is deferred until cross-architecture exact-token drift is handled by policy (`sm_80`/`sm_90`/`sm_120`).

## HF logits golden (short)

- Fixture: `test_data/qwen35-4b-hf-golden.safetensors` (~59 KiB); dumper `tools/accuracy/dump_qwen35_hf_golden.py`.
- Oracle path: HF `AutoModelForCausalLM.eval()` in bf16, prompt prefill with `use_cache=True`, then one-token teacher-forced decode through `past_key_values`.
- Model snapshot: `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; config hash `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.
- Shape: 12 seed-fixed prompt sequences, prompt length 1–128, 8 teacher-forced decode tokens, 108 scored positions, top-64 HF logprobs per position.
- Tolerances: regret `0.20`, mean head-delta `0.06`, p99 head-delta `0.20`; max is printed but not asserted (coverage-unstable).
- Replay surface: sequential bs=1 through the graph decode path, batched graph passes at 5→8 and 3→4 bucket straddles, and slot-compaction replay after a mid-batch request drop. Qwen3.5 currently has no eager batched decode path.

Run (RTX 5090 `sm_120`, Triton 3.4.0 build-time AOT):

```bash
PEGAINFER_CUDA_SM=120 \
PEGAINFER_TRITON_PYTHON=$TRITON_PYTHON \
PEGAINFER_TEST_MODEL_PATH=$MODEL_PATH \
PEGAINFER_TEST_MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
cargo test --release -p pegainfer-qwen35 --test hf_golden_gate -- --nocapture
```

Observed floor (RTX 5090 `sm_120`):

| Pass | positions | mean | p50 | p99 |
| --- | ---: | ---: | ---: | ---: |
| sequential bs=1 graph | 108 | 0.0248 | 0.0175 | 0.0862 |
| batched graph (5 padded) | 45 | 0.0256 | 0.0199 | 0.0757 |
| batched graph (3 padded) | 27 | 0.0260 | 0.0179 | 0.1007 |
| slot-compaction graph | 38 | 0.0285 | 0.0219 | 0.1031 |

## HF logits golden (long)

- Fixture: `test_data/qwen35-4b-hf-long-golden.safetensors` (~58 KiB); dumper with `--prompt-lens 4097,8192 --decode-tokens 8`.
- Shape: 2 seed-fixed sequences, prompts of 4097 and 8192 tokens, 8 teacher-forced decode tokens, 18 scored positions, top-64 logprobs each.
- Purpose: protect the RoPE cache boundary and the long prefill-to-decode logits path. This logits-level gate is paired with the full GSM8K 8-shot run below for task-score evidence.

Observed long floor (RTX 5090 `sm_120`):

| Pass | positions | mean | p50 | p99 |
| --- | ---: | ---: | ---: | ---: |
| long sequential bs=1 graph | 18 | 0.0216 | 0.0238 | 0.0700 |

## GSM8K 8-shot task score

The correctness fix was checked through the serving path with `lm-eval==0.4.11` and `local-completions` pointed at `/v1/completions`, all 1,319 GSM8K examples, `batch_size=1`:

| Filter | exact_match | stderr | Delta vs HF 79.45% |
| --- | ---: | ---: | ---: |
| strict-match | 79.38% | 1.11% | -0.07 pp |
| flexible-extract | 79.30% | 1.12% | -0.15 pp |

This establishes GSM8K 8-shot parity for the measured serving path. It makes no
claim about MMLU, HellaSwag, ARC, long-context admission, non-greedy sampling,
or `batch_size > 1` task scores.

## Size-portable fixture selection

- `hf_golden_gate.rs` derives the fixture key from config content, never the directory name: `text_config.hidden_size` / `num_hidden_layers` of `(1024, 24)` → `0.8b`, `(2048, 24)` → `2b`, `(2560, 32)` → `4b`, `(4096, 32)` → `9b`, `(5120, 64)` → `27b`. The mapping lives in `fixture_size_name` and must stay in sync with `SIZE_NAMES` in `tools/accuracy/dump_qwen35_hf_golden.py`.
- Default fixture paths are `test_data/qwen35-{size}-hf-golden.safetensors` and `-hf-long-golden.safetensors`; `PEGAINFER_QWEN35_HF_GOLDEN` / `PEGAINFER_QWEN35_HF_LONG_GOLDEN` override them.
- Failure semantics: an unreadable/malformed config or a missing committed fixture panics; a recognized size with no committed fixture skips and prints the expected path; an env override pointing at a missing file panics. Every mapped geometry now has a fixture.
- Tolerances are shared across sizes from the 4B calibration; the 0.8B/2B/9B/27B floors all sit well inside them. 9B confirms the untied `lm_head` fix; 27B covers the group-6 full-attention decode reroute; 0.8B/2B are the first checkpoints with GDN expansion factor 1 (`linear_num_value_heads == linear_num_key_heads`), validated on GH200 `sm_90` (floors mean `0.023–0.030`, p99 `≤ 0.115`).
- The model-wide collapse net folds into `tests/e2e_scheduler.rs`: its free-running completions fail when at least half collapse into token loops (distinct-token ratio, same-token run, or exact repeated tail period). This is the size-independent safety net under the fixture gate.

## Correctness findings (resolved)

These are the durable conclusions from the layer-by-layer debugging that
preceded the HF goldens. They explain what the gate protects and are kept as
decision context, not as a changelog.

- **HF full-prefill logits are not a reliable truth source for later generated tokens.** For token `t > 0`, the correct reference is HF's real incremental `past_key_values` decode trace, not a fresh full-prefill of the reconstructed prefix. HF's own incremental path also drifts from reconstructed full-prefill by the last layer, so "HF full-prefill on the generated prefix" is doubly unsafe.
- **`conv1d` pre-`SiLU` bf16 rounding.** HF fallback executes `Conv1d` on bf16, materializes a bf16 conv result, then applies `SiLU`; the kernel previously accumulated in fp32 and applied `SiLU` on that fp32 sum. Rounding the conv sum to bf16 before `SiLU` (in both decode and prefill kernels) zeroed the layer-0 `conv1d_out` diff vs HF.
- **`argmax` tie-break across CUDA threads.** Equal-valued logits could select a larger token id because its owning thread had a smaller `tid`. Fixed to prefer the smallest token id on exact ties (host-side `argmax` semantics); guarded by `test_argmax_tie_prefers_smallest_index[_across_thread_strides]`.
- **`conv1d` prefill state handoff when `seq_len < kernel_size - 1`.** Repeated `seq_len=1` prefill calls only wrote the newest token into the tail slot, leaving earlier slots stale; the kernel now snapshots the old per-channel state and rebuilds the final `(kernel_size - 1)` window from `[old_state, x_seq]`.
- **The residual generation drift is cumulative, not catastrophic.** After the discrete decode-state bugs were fixed, what remains is smaller bf16 prefill accumulation distributed across layers; some prompt-level divergence is tie-sensitive (competing top logits separated by only `0.125`–`0.25`). A rejected lead: forcing extra bf16 rounding inside `rms_norm_gated_kernel` made the real layer-0 HF comparison worse and was reverted.

## Deferred rand/hash corpus

#186 also discussed a larger PegaInfer-owned rand/hash regression corpus once the
HF gate is trusted. That idea is still useful, but checked-in exact token/hash
data may depend on GPU architecture and CUDA stack. Do not land it as a normal
regression gate until the policy says whether it is per-arch,
tolerance-adjudicated through HF, or a local diagnostic only.

## Retired historical accuracy tooling

The layer-`0`/decode dump and comparison tools that produced the findings above
are not present in the current tree after the model-crate split: `qwen35_dump_layer0`,
`qwen35_dump_decode_layer_ids`, `qwen35_check_incremental`, and friends
(`tools/accuracy/hf_dump_*`, `compare_qwen35_dump.py`, etc.). Any new
accuracy work should use the HF logits gate rather than reconstructing a
layer-by-layer pipeline.
