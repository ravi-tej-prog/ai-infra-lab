from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from qwen_platform.benchmark_run import BenchmarkPlan, execute_plan
from qwen_platform.client import ClientSettings, chat
from qwen_platform.sse import StreamAccumulator, consume_sse_line


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/v1/models":
            self.send_error(404)
            return
        body = b'{"object":"list","data":[{"id":"mock-model"}]}'
        self._send(200, "application/json", body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode())
        if payload.get("fail"):
            self._send(500, "application/json", b'{"error":"boom"}')
            return
        if payload.get("stream"):
            self._stream()
            return
        body = json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        ).encode()
        self._send(200, "application/json", body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        self.wfile.flush()
        time.sleep(0.2)
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n')
        self.wfile.flush()
        time.sleep(0.2)
        self.wfile.write(b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n')
        self.wfile.flush()
        usage = json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
            }
        )
        self.wfile.write(f"data: {usage}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server_url() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_role_chunk_does_not_count_as_the_first_token() -> None:
    accumulator = StreamAccumulator()
    done = consume_sse_line(
        accumulator,
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        0.1,
    )
    assert done is False
    assert accumulator.content_arrival_offsets == []
    consume_sse_line(
        accumulator,
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        0.4,
    )
    assert accumulator.content_arrival_offsets == [0.4]
    assert accumulator.text == "Hi"


def test_streaming_client_separates_ttft_from_end_to_end(server_url: str) -> None:
    settings = ClientSettings(base_url=server_url, model="mock-model", timeout_seconds=10)
    result = _run(
        chat(settings, "hello", max_output_tokens=8, stream=True)
    )
    assert result.success is True
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 2
    assert result.ttft_seconds is not None
    assert result.e2e_seconds is not None
    assert result.ttft_seconds < result.e2e_seconds
    assert result.ttft_seconds > 0.05
    assert result.tpot_seconds == pytest.approx(result.e2e_seconds - result.ttft_seconds)
    assert result.text == "Hello there"
    assert result.content_chunk_count == 2


def test_unary_client_has_no_ttft(server_url: str) -> None:
    settings = ClientSettings(base_url=server_url, model="mock-model", timeout_seconds=10)
    result = _run(chat(settings, "hello", max_output_tokens=8, stream=False))
    assert result.success is True
    assert result.ttft_seconds is None
    assert result.tpot_seconds is None
    assert result.e2e_seconds is not None
    assert result.completion_tokens == 1


def test_benchmark_against_mock_server_records_closed_loop_counts(server_url: str) -> None:
    plan = BenchmarkPlan(
        base_url=server_url,
        model="mock-model",
        engine="other",
        concurrency=2,
        num_requests=4,
        warmup_requests=0,
        max_output_tokens=8,
        temperature=0.0,
        timeout_seconds=10,
        prompt_profile="short",
    )
    document = _run(execute_plan(plan))
    assert document["status"] == "completed"
    assert document["verification"] == "CLIENT_ONLY"
    assert document["vllm_gpu_baseline"] is False
    assert document["summary"]["success_count"] == 4
    assert document["summary"]["failure_count"] == 0
    assert document["summary"]["ttft_seconds"]["count"] == 4
    assert document["summary"]["aggregate_output_tokens_per_second"] is not None
    assert len(document["requests"]) == 4


def test_unreachable_endpoint_is_not_run_and_has_no_fake_latency() -> None:
    port = _closed_port()
    plan = BenchmarkPlan(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="Qwen/Qwen2.5-14B-Instruct",
        engine="vllm",
        concurrency=1,
        num_requests=4,
        warmup_requests=1,
        max_output_tokens=32,
        temperature=0.0,
        timeout_seconds=2,
        target_input_tokens=32,
    )
    document = _run(execute_plan(plan))
    assert document["status"] == "not_run"
    assert document["verification"] == "REQUIRES_NVIDIA_GPU"
    assert document["vllm_gpu_baseline"] is False
    assert document["requests"] == []
    assert document["summary"]["success_count"] == 0
    assert document["summary"]["ttft_seconds"]["mean"] is None
    assert document["wall_seconds"] is None


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run(coro: object) -> object:
    import asyncio

    return asyncio.run(coro)  # type: ignore[arg-type]
