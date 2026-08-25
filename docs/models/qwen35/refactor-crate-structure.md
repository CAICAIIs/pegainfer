# Qwen3.5 crate refactor — record

> **TL;DR** Refactored `pegainfer-qwen35` from monolithic "god" modules into clean concern-grouped flat submodules (repo convention: `foo.rs` entry + `foo/` submodules, no `mod.rs`), and brought the crate to idiomatic, warning-free clippy. Behavior unchanged — all validation green.
>
> **Last touched:** 2026-08

## Result (branch `refactor/qwen35-scheduler-backend-split`)

| Module | Before | After | Split into |
|---|---|---|---|
| `scheduler.rs` | 2624 | 969 | `scheduler/{backend, tp, steps, telemetry, emit}` + `plan`/`tests` |
| `tp_executor.rs` | 2520 | 723 | `tp_executor/{executor, worker, responses}` |
| `config.rs` | 683 | 285 | `config/{model, tokenizer, tp}` |
| `weights.rs` | 921 | 802 | `weights/layers.rs` |
| forward | 5 files at crate root | 1 group | `forward/{prefill, batch_decode, batch_decode_graph, unified_forward, recurrent}` |

> All validation green: `cargo clippy --features qwen35` 0 warnings, `cargo test --lib` 95/95, `tp2_*` 7/7 (weights + 2 CUDA devices + NCCL). See [`qwen35-refactor-validation.md`](qwen35-refactor-validation.md) for the correctness + serving-performance report.

## Key technique

- **Split a concern cluster whole**, and keep shared state types in the parent module.
- **Visibility**: `use self::child::*` re-exports child `pub(super)` items; mark `pub(super)` on the items/functions/traits the parent references (else E0425/E0405). Struct fields need `pub(super)` too when read cross-module (else E0616/E0624); trait/`Drop` impl methods cannot take a visibility qualifier (E0449). Public API (`Qwen35TpExecutor`, step items) keeps `pub`/`pub(crate)` and is re-exported.
- **Tooling**: content-marker extraction; `rustfmt --check` for a parser pass before `cargo check`; carry `#[derive(...)]`/doc comments along.
- **Gotchas**: `#[derive]/doc` comments can be dropped by extraction (re-attach them); a blank line after a doc comment trips `empty_line_after_doc_comments` (run `cargo clippy --fix` for the removals); `#![allow(clippy::wildcard_imports)]` documents the submodule glob convention.

## Remaining work

- `weights.rs` holds the model's single large `impl Qwen35Model` (load/inference orchestration) — kept together for cohesion.
- Group-internal files such as `forward/recurrent.rs` (982) and `scheduler/plan.rs` (907) are in the same size range as the sibling model crates (qwen3 / deepseek-v2-lite / kimi-k2) and are intentionally not split further (splitting to finer grain would diverge from the repo's accepted granularity).
