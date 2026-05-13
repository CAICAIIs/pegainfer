#!/usr/bin/env python3
"""Regression tests for scripts/bench_http_serving.py."""

from __future__ import annotations

import importlib.util
import sys
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench_http_serving.py"
SPEC = importlib.util.spec_from_file_location("bench_http_serving", SCRIPT_PATH)
assert SPEC and SPEC.loader
bench_http_serving = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_http_serving
SPEC.loader.exec_module(bench_http_serving)


class DoneOnlyHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format: str, *args: object) -> None:
        return


class BenchHttpServingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), DoneOnlyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.url = urllib.parse.urlparse(f"http://{host}:{port}")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_done_only_stream_fails_when_tokens_requested(self) -> None:
        result = bench_http_serving.request_once(
            index=0,
            url=self.url,
            model="fake-model",
            prompt="hello",
            max_tokens=1,
            temperature=0.0,
            timeout=5,
            ignore_eos=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 200)
        self.assertIn("without streamed text chunks", result.error or "")
        self.assertEqual(result.output_chunks, 0)


if __name__ == "__main__":
    unittest.main()
