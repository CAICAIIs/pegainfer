# Qwen3.5 refactor + validation + serving performance

> **TL;DR** 结构重构（God 模块拆分成关注点目录）+ 惯用法收尾，行为保持不变；正确性全部通过；在单卡 A100#1 上测了 PegaInfer serving 性能，并与 vLLM 0.27.0（同模型同配置同 seed）做了直接 A/B。
>
> **Last touched:** 2026-08

## 1. 结构重构（behavior unchanged）
- `scheduler.rs` 2624→969 → `scheduler/{backend,tp,steps,telemetry,emit,plan,tests}`
- `tp_executor.rs` 2520→723 → `tp_executor/{executor,worker,responses}`
- `config.rs` 683→285 → `config/{model,tokenizer,tp}`
- `weights.rs` 921→802 + `weights/layers.rs`
- `forward/` 前向 5 模块归位成组
- 惯用法：crate 级 `thiserror` `Error`、prod 去 `unwrap/expect`、参数 Bundle、`debug_assert!→ensure!`、去 phase1/P2A/issue 行话。
- 扁平约定：`foo.rs` 入口 + 同名 `foo/`，无 `mod.rs`。

## 2. 正确性（all green）
| gate | result |
|---|---|
| `cargo test --lib` | 95/95 |
| tp2 单测（双卡+NCCL+权重） | 7/7 |
| `hf_golden_gate`（logits vs HF golden） | 4/4（单卡 2 + tp2 2）|
| `e2e_scheduler`（调度存活/请求流） | 3/3（单卡 2 + tp2 1）|
| `chunked_prefill` | 1/1 |
| `sampling_behavior` | 1/1 |
| `cargo clippy --features qwen35` | 0 warnings |
| `cargo fmt --check` | clean |

> 重构后行为保持、正确性达标（含 TP 路径准确率 gate）。

## 3. PegaInfer serving 性能（单 A100#1，vllm-bench 规范跑测）
`input 1024 / output 128 / request-rate 4 / num-prompts 40 / seed 42`：
- Mean/Median/P99 **TTFT**: 96.07 / 84.10 / 185.93 ms
- Mean/Median/P99 **TPOT**: 16.53 / 15.66 / 22.72 ms
- **Output** 445.18 tok/s · **Total** 3856.84 tok/s · 峰值并发 22 · 用时 11.50s
- 小负载（64/32/rate1）：TTFT 23.6ms / TPOT 9.6ms。

## 4. PegaInfer vs vLLM 0.27.0（直接 A/B，同配置同 seed 同 GPU）
| | PegaInfer | vLLM |
|---|---|---|
| TTFT mean | **96.07 ms** | 114.29 ms |
| TPOT mean | 16.53 ms | **11.70 ms** |
| Output tok/s | 445.18 | **455.02** |
| Total tok/s | 3856.84 | **4095.15** |

> 结论：**vLLM 解码更快（TPOT/吞吐略优）；PegaInfer prefill/TTFT 更快；总体接近**。与仓库文档里 RTX 5090 + 1024/256 的"PegaInfer 显著落后 vLLM"不同——在这台 A100 + 本配置下两者档次接近。

## 5. 让 vLLM 在本机跑起来的修复（复现用）
1. 装干净 vllm 0.27.0 到 `/mnt/data/models/vllm-clean-venv`。
2. `--tokenizer /mnt/data/models/Qwen3.5-4B` + `HF_ENDPOINT=https://hf-mirror.com`（本机直连 HF 被墙）。
3. `vllm bench serve --base-url http://127.0.0.1:<port>`（不带 `/v1`，避免拼 `/v1/v1` → 404）。
4. flashinfer `fd_exchange.py` 加 `from __future__ import annotations`（修 `array.array[int]` 在 Py3.10 的 TypeError）。
5. vLLM serve 用 `CUDA_HOME=/usr/local/cuda-13.1` + `PATH` 指向其 `bin` + `CC=gcc-12 CXX=g++-12`（JIT kernel 需 CUDA≥12 + 较新 GCC；主机默认 nvcc=11.5、GCC=11）。

## 6. 关键结果 JSON
- PegaInfer: `/tmp/q35b5/qwen35-1024x128-r4.json`
- vLLM: `/tmp/q35v/vllm-1024x128-r4.json`
