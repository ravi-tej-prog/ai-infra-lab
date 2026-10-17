"""Closed-loop benchmark runner for an OpenAI-compatible endpoint.

Concurrency N means at most N requests are in flight. The runner sends
`num_requests` measured calls after optional warmup calls. Warmup is excluded
from the summary. Wall-clock throughput uses only the measured window.
"""

from __future__ import annotations

import asyncio
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from qwen_platform.client import ChatResult, ClientSettings, InferenceClientError, chat, preflight
from qwen_platform.prompts import prompt_for_target_tokens, prompt_from_profile
from qwen_platform.reporting import summarize_requests


class RequestObserver(Protocol):
    """Optional hooks around measured requests, e.g. a Prometheus exporter."""

    def cell_started(self) -> None: ...

    def cell_finished(self) -> None: ...

    def request_started(self) -> None: ...

    def request_finished(self, result: ChatResult) -> None: ...


@dataclass(frozen=True)
class BenchmarkPlan:
    base_url: str
    model: str
    engine: str
    concurrency: int
    num_requests: int
    warmup_requests: int
    max_output_tokens: int
    temperature: float
    timeout_seconds: float
    target_input_tokens: int | None = None
    prompt_profile: str | None = None
    api_key: str | None = None
    mark_nvidia_baseline: bool = False
    unique_prompts: bool = False

    def settings(self) -> ClientSettings:
        return ClientSettings(
            base_url=self.base_url,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            api_key=self.api_key,
        )


async def execute_plan(plan: BenchmarkPlan, observer: RequestObserver | None = None) -> dict[str, Any]:
    if plan.concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if plan.num_requests < 1:
        raise ValueError("num_requests must be >= 1")
    if plan.warmup_requests < 0:
        raise ValueError("warmup_requests must be >= 0")
    prompt, prompt_builder = _select_prompt(plan)
    document = _base_document(plan, prompt_builder)
    settings = plan.settings()
    timeout = httpx.Timeout(plan.timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as http:
        try:
            await preflight(settings, http)
        except InferenceClientError as exc:
            document["status"] = "not_run"
            document["notes"].append(str(exc))
            document["notes"].append(
                "No measured requests were sent. This file is a not-run record, not a latency sample."
            )
            document["summary"] = summarize_requests([], None)
            return document

        for index in range(plan.warmup_requests):
            await chat(
                settings,
                _request_prompt(plan, prompt),
                max_output_tokens=plan.max_output_tokens,
                temperature=plan.temperature,
                stream=True,
                client=http,
            )
            document["notes"].append(f"Warmup request {index + 1} finished and was excluded from the summary.")

        wall_started = asyncio.get_running_loop().time()
        semaphore = asyncio.Semaphore(plan.concurrency)

        async def one(index: int) -> dict[str, Any]:
            async with semaphore:
                if observer:
                    observer.request_started()
                result = await chat(
                    settings,
                    _request_prompt(plan, prompt),
                    max_output_tokens=plan.max_output_tokens,
                    temperature=plan.temperature,
                    stream=True,
                    client=http,
                )
                if observer:
                    observer.request_finished(result)
            return _request_row(index, plan, result)

        if observer:
            observer.cell_started()
        try:
            rows = await asyncio.gather(*(one(index) for index in range(plan.num_requests)))
        finally:
            if observer:
                observer.cell_finished()
        wall_seconds = asyncio.get_running_loop().time() - wall_started

    ordered = sorted(rows, key=lambda row: int(row["request_index"]))
    document["requests"] = ordered
    document["wall_seconds"] = round(wall_seconds, 6)
    document["summary"] = summarize_requests(ordered, wall_seconds)
    success_count = document["summary"]["success_count"]
    failure_count = document["summary"]["failure_count"]
    if success_count == 0:
        document["status"] = "failed"
    elif failure_count:
        document["status"] = "completed_with_failures"
    else:
        document["status"] = "completed"
    return document


def _request_prompt(plan: BenchmarkPlan, prompt: str) -> str:
    """Put a random tag at the start so no two requests share a prompt prefix.

    Ollama and vLLM reuse the KV cache for a repeated prefix. With one fixed
    prompt, every request after the first skips most of prefill and TTFT stops
    growing with input length. The tag sits at the front because a cache match
    stops at the first differing token.
    """

    if not plan.unique_prompts:
        return prompt
    return f"[request {uuid.uuid4().hex[:12]}] {prompt}"


def _select_prompt(plan: BenchmarkPlan) -> tuple[str, str]:
    if plan.prompt_profile:
        return prompt_from_profile(plan.prompt_profile), f"profile:{plan.prompt_profile}"
    if plan.target_input_tokens is None:
        raise ValueError("Set target_input_tokens or prompt_profile")
    return (
        prompt_for_target_tokens(plan.target_input_tokens),
        "repeat-sentence-v1-char-estimate",
    )


def _base_document(plan: BenchmarkPlan, prompt_builder: str) -> dict[str, Any]:
    verification, baseline, notes = _verification(plan)
    return {
        "schema_version": 1,
        "status": "not_run",
        "verification": verification,
        "vllm_gpu_baseline": baseline,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_platform": platform.system(),
        "client_machine": platform.machine(),
        "engine": plan.engine,
        "model": plan.model,
        "endpoint": plan.base_url,
        "concurrency": plan.concurrency,
        "num_requests": plan.num_requests,
        "warmup_requests": plan.warmup_requests,
        "target_input_tokens": plan.target_input_tokens,
        "prompt_profile": plan.prompt_profile,
        "prompt_builder": prompt_builder,
        "unique_prompts": plan.unique_prompts,
        "max_output_tokens": plan.max_output_tokens,
        "temperature": plan.temperature,
        "wall_seconds": None,
        "measurement": "client_observed",
        "formulas": {
            "ttft_seconds": "first_content_chunk_time - request_start",
            "e2e_seconds": "stream_end_time - request_start",
            "tpot_seconds": "(e2e_seconds - ttft_seconds) / (output_tokens - 1)",
            "output_tokens_per_second": "output_tokens / e2e_seconds",
            "decode_tokens_per_second": "(output_tokens - 1) / (e2e_seconds - ttft_seconds)",
            "aggregate_output_tokens_per_second": "sum(output_tokens) / measured_wall_seconds",
            "inter_chunk_latency_seconds": "gaps between content-bearing SSE chunks",
        },
        "notes": notes + _prefix_cache_note(plan),
        "summary": None,
        "requests": [],
    }


def _verification(plan: BenchmarkPlan) -> tuple[str, bool, list[str]]:
    notes = [
        "Times are measured on the client. They include network delay and queue wait.",
        "TTFT is the arrival of the first non-empty content chunk, not the end of the request.",
        "TPOT uses API completion_tokens. Inter-chunk gaps are recorded separately because one SSE event can carry more than one token.",
    ]
    if plan.engine == "ollama":
        notes.append(
            "This run uses Ollama. It checks the client and the result schema on Apple Silicon. "
            "It is a GGUF comparison, and the vLLM NVIDIA baseline remains unmeasured."
        )
        return "LOCAL_VERIFIED", False, notes
    if plan.engine == "vllm":
        notes.append(
            "A vLLM GPU baseline requires a Linux host with NVIDIA GPUs. "
            "This Mac can only be the client. vllm_gpu_baseline stays false unless "
            "--mark-nvidia-baseline is passed after that server is actually running."
        )
        if plan.mark_nvidia_baseline:
            notes.append("Operator marked this endpoint as a real NVIDIA vLLM server.")
            return "REQUIRES_NVIDIA_GPU", True, notes
        return "REQUIRES_NVIDIA_GPU", False, notes
    notes.append("Engine label is not vllm or ollama. Treat the numbers as a client check only.")
    return "CLIENT_ONLY", False, notes


def _prefix_cache_note(plan: BenchmarkPlan) -> list[str]:
    if plan.unique_prompts:
        return ["Each request starts with a random tag, so the server cannot reuse a cached prompt prefix."]
    return ["Every request sends the same prompt. A server with prefix caching may skip prefill after the first request."]


def _request_row(index: int, plan: BenchmarkPlan, result: ChatResult) -> dict[str, Any]:
    row = result.as_dict()
    preview = row.pop("text")
    row["output_preview"] = preview[:80]
    row["request_index"] = index
    row["input_tokens"] = row.pop("prompt_tokens")
    row["output_tokens"] = row.pop("completion_tokens")
    row["input_token_source"] = "api_usage" if row["input_tokens"] is not None else None
    row["output_token_source"] = "api_usage" if row["output_tokens"] is not None else None
    row["concurrency"] = plan.concurrency
    row["max_output_tokens"] = plan.max_output_tokens
    return row
