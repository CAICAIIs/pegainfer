# DeepSeek V4 HTTP Serving Benchmark Gate

**Created**: 2026-05-14
**Status**: active
**Canonical task**: task #18

## Purpose

This gate measures the OpenAI-compatible HTTP endpoint under concurrent load. It
does not use the in-process `bench_serving request` path as serving evidence.

The benchmark client sends streaming `/v1/completions` requests and records:

- QPS and completed/failed/timeout counts.
- TTFT from request send to first streamed text chunk.
- ITL and TPOT from streamed text chunks.
- End-to-end latency percentiles.
- Per-request output hashes plus a combined output hash for reproducibility.

## Reproducible Commands

Build the server on the target GPU host:

```bash
cd /path/to/pegainfer
export PATH=/usr/local/cuda-13.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.1
export PEGAINFER_TILELANG_PYTHON=/path/to/venv/bin/python
export PEGAINFER_TRITON_PYTHON=/path/to/venv/bin/python
export PEGAINFER_NVCC_JOBS=8
export CARGO_TARGET_DIR=/path/to/pegainfer-target

cargo build --release -p pegainfer-server --features deepseek-v4 --bin pegainfer
```

Start the OpenAI-compatible HTTP endpoint:

```bash
$CARGO_TARGET_DIR/release/pegainfer \
  --model-path /data/DeepSeek-V4-Flash \
  --port 18118
```

Verify the model endpoint:

```bash
curl -sS http://127.0.0.1:18118/v1/models
```

Run the HTTP serving benchmark:

```bash
python3 scripts/bench_http_serving.py \
  --base-url http://127.0.0.1:18118 \
  --model /data/DeepSeek-V4-Flash \
  --warmup 2 \
  --num-requests 8 \
  --concurrency 2 \
  --prompt-words 16 \
  --max-tokens 16 \
  --timeout 240 \
  --out /tmp/dsv4_http_bench_task18.json
```

The script is intentionally model-server agnostic at the HTTP layer. It only
requires an OpenAI-compatible `/v1/completions` endpoint that supports streaming
responses.

## Current Evidence

Evidence below was collected on the internal 8-GPU DeepSeek-V4-Flash validation
host. It describes only this commit, machine, endpoint, and harness.

| Field | Value |
| --- | --- |
| Commit | PR body records the validated head; tracked docs avoid self-referential commit hashes. |
| Endpoint | OpenAI-compatible `/v1/completions`, streaming |
| Model | `/data/DeepSeek-V4-Flash` |
| Workload | warmup `2`, measured requests `8`, concurrency `2`, prompt words `16`, max tokens `16`, temperature `0`, ignore EOS `true`, timeout `240s` |
| Result | completed `8`, failed `0`, timeout `0`, error rate `0.0` |
| QPS | `1.6869` completed requests/s |
| Latency | avg `1112.19ms`, p50 `1179.70ms`, p95 `1207.23ms`, p99 `1207.23ms` |
| TTFT | avg `680.38ms`, p50 `746.66ms`, p95 `775.06ms`, p99 `775.06ms` |
| TPOT | avg `28.78ms`, p50 `28.81ms`, p95 `28.88ms`, p99 `28.88ms` |
| ITL | avg `28.78ms`, p50 `28.28ms`, p95 `30.57ms`, p99 `30.74ms` |
| Output stability | output chunks `128`, unique output hashes `8`, combined output hash `22706877075acde0` |

## Boundary

This PR establishes a benchmark gate and one real HTTP run. It does not claim
vLLM parity, production serving stability, larger batch scalability, paged or
prefix KV, or P/D handoff behavior.

`bench_serving request` remains the in-process direct regression path. It is not
used as a substitute for HTTP serving metrics in this document.
