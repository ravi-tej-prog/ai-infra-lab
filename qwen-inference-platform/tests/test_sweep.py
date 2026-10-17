from __future__ import annotations

import asyncio
from pathlib import Path

from qwen_platform.benchmark_run import BenchmarkPlan, _request_prompt
from qwen_platform.sweep import SweepConfig, load_sweep_rows, render_markdown, run_sweep

from tests.test_client_and_benchmark import server_url  # noqa: F401  (pytest fixture)


def _plan(unique: bool) -> BenchmarkPlan:
    return BenchmarkPlan(
        base_url="http://unused/v1",
        model="mock-model",
        engine="other",
        concurrency=1,
        num_requests=1,
        warmup_requests=0,
        max_output_tokens=8,
        temperature=0.0,
        timeout_seconds=1,
        unique_prompts=unique,
    )


def test_unique_prompts_differ_at_the_first_token() -> None:
    first = _request_prompt(_plan(True), "same body")
    second = _request_prompt(_plan(True), "same body")
    assert first != second
    assert first.endswith("same body")
    assert first.split()[1] != second.split()[1]


def test_shared_prompt_is_unchanged_by_default() -> None:
    assert _request_prompt(_plan(False), "same body") == "same body"


def test_sweep_writes_one_run_per_cell_and_a_table(server_url: str, tmp_path: Path) -> None:  # noqa: F811
    config = SweepConfig(
        base_url=server_url,
        model="mock-model",
        engine="other",
        input_tokens=[16, 64],
        concurrency=[1, 2],
        max_output_tokens=8,
        temperature=0.0,
        unique_prompts=True,
        warmup_requests=0,
        min_requests=2,
        requests_per_slot=1,
        timeout_seconds=10,
    )
    rows = asyncio.run(run_sweep(config, tmp_path))
    assert [(row["target_input_tokens"], row["concurrency"]) for row in rows] == [(16, 1), (16, 2), (64, 1), (64, 2)]
    assert all(row["status"] == "completed" for row in rows)
    assert len(list(tmp_path.glob("in*-c*.json"))) == 4

    reloaded = load_sweep_rows(tmp_path)
    assert reloaded == rows
    table = render_markdown(reloaded, "mock")
    assert table.count("\n| ") == 5


def test_sweep_stops_when_the_server_is_down(tmp_path: Path) -> None:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = SweepConfig(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="mock-model",
        engine="ollama",
        input_tokens=[16, 64],
        concurrency=[1, 2],
        max_output_tokens=8,
        temperature=0.0,
        unique_prompts=True,
        warmup_requests=0,
        min_requests=1,
        requests_per_slot=1,
        timeout_seconds=2,
    )
    rows = asyncio.run(run_sweep(config, tmp_path))
    assert len(rows) == 1
    assert rows[0]["status"] == "not_run"
