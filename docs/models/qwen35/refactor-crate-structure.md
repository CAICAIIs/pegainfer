# Qwen3.5 (pegainfer-qwen35) 大重构 —— 落地记录

> **TL;DR** 目标：上游 `pegainfer-qwen35`（`openinfer-project/openinfer` @`029d3a95`）。用「拆分 God 模块 + 归位到清晰属主 + 惯用法」为主、**重命名最小化**；遵守 repo **扁平约定**（`foo.rs + foo/`，无 `mod.rs`）。每个切片在 GPU 机 `cargo check + fmt + test --lib`（及补权重+双卡后 `tp2`）验证，行为保持。
>
> **Last touched:** 2026-08

## 结果（主线分支 `refactor/qwen35-scheduler-backend-split`）
| 模块 | 改动前 | 改动后 | 拆分 |
|---|---|---|---|
| `scheduler.rs` | 2624 | 978 | `scheduler/{backend,tp,steps,telemetry,emit}` + `plan`/`tests` |
| `tp_executor.rs` | 2520 | 723 | `tp_executor/{executor,worker,responses}` |
| `config.rs` | 683 | 290 | `config/{model,tokenizer,tp}`（分支 `refactor/qwen35-config-split`）|
| `forward/` | 5 平铺 | 成组 | `forward/{prefill,batch_decode,batch_decode_graph,unified_forward,recurrent}` |

> 全部通过 `cargo clippy --features qwen35`（0 告警）+ `cargo test --lib` 95/95 + `tp2` 7/7（权重 `/mnt/data/models/Qwen3.5-4B` + 双卡 `CUDA_VISIBLE_DEVICES=1,2` + NCCL）。

## 切片提交（主线）
`1d77e6f`(backend) · `e218049`(tp) · `4e80daf`(steps) · `4c7f418`(telemetry) · `775b7ed`(emit) · `9f0d2d3`(doc 回补) · `47bd225`(forward) · `eaf7b0d`(worker) · `293c0be`(responses) · `e0e1c2b`(executor) · `1171cb7`(clippy 0 告警)

## 关键手法与教训
- **拆法（最稳）**：**整簇搬 + 状态类型留父模块**；父用 `use self::child::*` 带回，子用 `use super::*` 够到父与兄弟。
- **可见性**：`use self::child::*` glob **能**带出子模块 `pub(super)` 项；只需给「父模块要调用的函数/引用的 trait / 对外类型」标 `pub(super)`（否则 E0425/E0405）。**跨模块共享的结构体，字段/方法也要 `pub(super)`**（否则 E0616/E0624）；trait/`Drop` impl 方法不能加可见性（E0449）。`Qwen35TpExecutor`(pub)/step items(pub(crate)) 保留原可见性，用 `pub use`/`pub(crate) use` 重导出。
- **工具**：内容标记（头行）+ 精确行号 + `#[derive]`/doc 感知；边界因 cargo fmt 漂移 → 先 `rustfmt --check` 解析层、再 `cargo check` 类型层；搬移前先 `git checkout` 干净版、用 `cat -n` 核对端点。
- **坑**：`#[derive(...)]`/doc 注释会被提取遗漏（需回补/归位）；`doc` 后多空行触发 `empty_line_after_doc_comments`；crate 级 `#![allow(clippy::wildcard_imports)]` 声明子模块 glob 惯例；`tp2_*` 是 `#[ignore]` + 需权重/NCCL/双卡。

## 下一步
- **`weights.rs`(921) 拆分**：这次因 `impl Qwen35Model` 交织、脚本括号 bug 未落地，需按内容标记重做；`scheduler/tests.rs`(980) 下沉。
- **惯用法收尾**：crate 级 `thiserror` `Error`、prod 去 `unwrap`/`expect`、参数 Bundle、`ensure!`。
- 其余大文件：`forward/recurrent.rs`(982)、`scheduler/plan.rs`(907)、`scheduler/tests.rs`(980)。
