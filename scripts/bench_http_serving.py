#!/usr/bin/env python3
"""OpenAI-compatible HTTP serving benchmark for pegainfer.

The harness intentionally talks to /v1/completions over HTTP instead of using
the in-process bench_serving binary. It records streaming TTFT/ITL/TPOT,
request latency, QPS, error rate, timeout rate, and deterministic output hashes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import socket
import statistics
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


@dataclass
class RequestResult:
    index: int
    ok: bool
    status: int | None
    error: str | None
    timed_out: bool
    start_s: float
    first_token_s: float | None
    end_s: float
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    itl_ms: list[float]
    output_chunks: int
    output_chars: int
    output_hash: str
    text_prefix: str


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = round((pct / 100.0) * (len(sorted_values) - 1))
    return sorted_values[idx]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "samples": 0,
        }
    sorted_values = sorted(values)
    return {
        "avg_ms": statistics.fmean(sorted_values),
        "p50_ms": percentile(sorted_values, 50),
        "p95_ms": percentile(sorted_values, 95),
        "p99_ms": percentile(sorted_values, 99),
        "max_ms": sorted_values[-1],
        "samples": len(sorted_values),
    }


def make_prompt(index: int, prompt_words: int) -> str:
    words = [
        DEFAULT_PROMPT_WORDS[(index + offset) % len(DEFAULT_PROMPT_WORDS)]
        for offset in range(prompt_words)
    ]
    return " ".join(words)


def parse_sse_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


def request_once(
    index: int,
    url: urllib.parse.ParseResult,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    ignore_eos: bool,
) -> RequestResult:
    start = time.perf_counter()
    first_token: float | None = None
    last_token: float | None = None
    inter_token_ms: list[float] = []
    chunks: list[str] = []
    status: int | None = None

    try:
        conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        port = url.port
        conn = conn_cls(url.hostname, port=port, timeout=timeout)
        path = (url.path.rstrip("/") or "") + "/v1/completions"
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "ignore_eos": ignore_eos,
        }
        conn.request(
            "POST",
            path,
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        status = response.status
        if status != 200:
            error_body = response.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {status}: {error_body}")

        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            text = parse_sse_text(payload)
            if not text:
                continue
            now = time.perf_counter()
            if first_token is None:
                first_token = now
            if last_token is not None:
                inter_token_ms.append((now - last_token) * 1000.0)
            last_token = now
            chunks.append(text)
        conn.close()
        if max_tokens > 0 and not chunks:
            raise RuntimeError("stream completed without streamed text chunks")
        end = time.perf_counter()
        text = "".join(chunks)
        output_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        latency_ms = (end - start) * 1000.0
        ttft_ms = None if first_token is None else (first_token - start) * 1000.0
        tpot_ms = None
        if first_token is not None and last_token is not None and len(chunks) > 1:
            tpot_ms = (last_token - first_token) * 1000.0 / (len(chunks) - 1)
        return RequestResult(
            index=index,
            ok=True,
            status=status,
            error=None,
            timed_out=False,
            start_s=start,
            first_token_s=first_token,
            end_s=end,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            itl_ms=inter_token_ms,
            output_chunks=len(chunks),
            output_chars=len(text),
            output_hash=output_hash,
            text_prefix=text[:80],
        )
    except (TimeoutError, socket.timeout) as exc:
        end = time.perf_counter()
        return failed_result(index, status, start, end, str(exc), timed_out=True)
    except Exception as exc:  # noqa: BLE001 - benchmark reports the error string.
        end = time.perf_counter()
        return failed_result(index, status, start, end, str(exc), timed_out=False)


def failed_result(
    index: int,
    status: int | None,
    start: float,
    end: float,
    error: str,
    timed_out: bool,
) -> RequestResult:
    return RequestResult(
        index=index,
        ok=False,
        status=status,
        error=error,
        timed_out=timed_out,
        start_s=start,
        first_token_s=None,
        end_s=end,
        latency_ms=(end - start) * 1000.0,
        ttft_ms=None,
        tpot_ms=None,
        itl_ms=[],
        output_chunks=0,
        output_chars=0,
        output_hash="",
        text_prefix="",
    )


def run_batch(args: argparse.Namespace, measured: bool) -> tuple[list[RequestResult], float]:
    url = urllib.parse.urlparse(args.base_url)
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise SystemExit(f"invalid --base-url: {args.base_url}")

    offset = args.warmup if measured else 0
    count = args.num_requests if measured else args.warmup
    prompts = [make_prompt(offset + idx, args.prompt_words) for idx in range(count)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request_once,
                idx,
                url,
                args.model,
                prompt,
                args.max_tokens,
                args.temperature,
                args.timeout,
                args.ignore_eos,
            )
            for idx, prompt in enumerate(prompts)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    ended = time.perf_counter()
    results.sort(key=lambda result: result.index)
    return results, ended - started


def build_report(args: argparse.Namespace, measured: list[RequestResult], wall_s: float) -> dict[str, Any]:
    successes = [result for result in measured if result.ok]
    failures = [result for result in measured if not result.ok]
    latencies = [result.latency_ms for result in successes]
    ttfts = [result.ttft_ms for result in successes if result.ttft_ms is not None]
    tpots = [result.tpot_ms for result in successes if result.tpot_ms is not None]
    itls: list[float] = []
    output_chunks = [result.output_chunks for result in successes]
    output_chars = [result.output_chars for result in successes]
    hashes = [result.output_hash for result in successes]

    for result in successes:
        itls.extend(result.itl_ms)

    return {
        "schema_version": 1,
        "kind": "openai_http_completions_stream_benchmark",
        "base_url": args.base_url,
        "model": args.model,
        "workload": {
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "prompt_words": args.prompt_words,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "timeout_s": args.timeout,
        },
        "summary": {
            "wall_s": wall_s,
            "completed": len(successes),
            "failed": len(failures),
            "timeouts": sum(1 for result in failures if result.timed_out),
            "qps": len(successes) / wall_s if wall_s > 0 else 0.0,
            "error_rate": len(failures) / args.num_requests if args.num_requests else 0.0,
            "timeout_rate": (
                sum(1 for result in failures if result.timed_out) / args.num_requests
                if args.num_requests
                else 0.0
            ),
            "output_chunks_total": sum(output_chunks),
            "output_chars_total": sum(output_chars),
            "unique_output_hashes": len(set(hashes)),
            "combined_output_hash": hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()[:16],
        },
        "metrics": {
            "latency": summarize(latencies),
            "ttft": summarize(ttfts),
            "tpot": summarize(tpots),
            "itl": summarize(itls),
        },
        "requests": [asdict(result) for result in measured],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt-words", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.num_requests <= 0:
        raise SystemExit("--num-requests must be positive")
    if args.warmup > 0:
        warmup_results, _ = run_batch(args, measured=False)
        failed = [result for result in warmup_results if not result.ok]
        if failed:
            raise SystemExit(f"warmup failed: {failed[0].error}")

    measured, wall_s = run_batch(args, measured=True)
    report = build_report(args, measured, wall_s)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    if report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
