"""Minimal OpenAI-compatible chat client with client-side timing.

Streaming is required for TTFT. A non-streaming response only has an end time,
so TTFT stays empty on that path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from qwen_platform.metrics import (
    decode_tokens_per_second,
    end_to_end_seconds,
    output_tokens_per_second,
    time_per_output_token_seconds,
    time_to_first_token_seconds,
)
from qwen_platform.sse import StreamAccumulator, consume_sse_line


class InferenceClientError(Exception):
    def __init__(self, message: str, *, error_type: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.http_status = http_status


@dataclass(frozen=True)
class ChatResult:
    success: bool
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_seconds: float | None
    tpot_seconds: float | None
    e2e_seconds: float | None
    output_tokens_per_second: float | None
    decode_tokens_per_second: float | None
    inter_chunk_latency_seconds: tuple[float, ...]
    content_chunk_count: int
    streamed: bool
    error_type: str | None = None
    error_message: str | None = None
    http_status: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_seconds": _round(self.ttft_seconds),
            "tpot_seconds": _round(self.tpot_seconds),
            "e2e_seconds": _round(self.e2e_seconds),
            "output_tokens_per_second": _round(self.output_tokens_per_second),
            "decode_tokens_per_second": _round(self.decode_tokens_per_second),
            "inter_chunk_latency_seconds": [_round(item) for item in self.inter_chunk_latency_seconds],
            "content_chunk_count": self.content_chunk_count,
            "streamed": self.streamed,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "http_status": self.http_status,
        }


@dataclass(frozen=True)
class ClientSettings:
    base_url: str
    model: str
    timeout_seconds: float = 180.0
    api_key: str | None = None

    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/models"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def build_chat_payload(
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    stream: bool,
) -> dict[str, Any]:
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be >= 1")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


async def preflight(settings: ClientSettings, client: httpx.AsyncClient | None = None) -> None:
    """Confirm the OpenAI-compatible server answers `/models`.

    A connection failure becomes an InferenceClientError so the benchmark can
    record `not_run` instead of a file full of repeated connection errors.
    """

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
    try:
        response = await http.get(settings.models_url(), headers=settings.headers())
    except httpx.HTTPError as exc:
        raise InferenceClientError(
            f"Endpoint is not reachable at {settings.models_url()}: {exc}",
            error_type=type(exc).__name__,
        ) from exc
    finally:
        if owns_client:
            await http.aclose()
    if response.status_code >= 400:
        raise InferenceClientError(
            f"{settings.models_url()} returned HTTP {response.status_code}",
            error_type="HTTPStatusError",
            http_status=response.status_code,
        )


async def chat(
    settings: ClientSettings,
    prompt: str,
    *,
    max_output_tokens: int,
    temperature: float = 0.0,
    stream: bool = True,
    client: httpx.AsyncClient | None = None,
) -> ChatResult:
    payload = build_chat_payload(
        model=settings.model,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        stream=stream,
    )
    timeout = httpx.Timeout(settings.timeout_seconds, connect=10.0)
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    started = time.perf_counter()
    try:
        if stream:
            return await _chat_stream(http, settings, payload, started)
        return await _chat_unary(http, settings, payload, started)
    except InferenceClientError as exc:
        return _failure(exc, streamed=stream, e2e_seconds=time.perf_counter() - started)
    except httpx.HTTPError as exc:
        wrapped = InferenceClientError(str(exc), error_type=type(exc).__name__)
        return _failure(wrapped, streamed=stream, e2e_seconds=time.perf_counter() - started)
    finally:
        if owns_client:
            await http.aclose()


async def _chat_stream(
    http: httpx.AsyncClient,
    settings: ClientSettings,
    payload: dict[str, Any],
    started: float,
) -> ChatResult:
    accumulator = StreamAccumulator()
    async with http.stream(
        "POST",
        settings.chat_url(),
        json=payload,
        headers=settings.headers(),
    ) as response:
        if response.status_code >= 400:
            body = (await response.aread()).decode("utf-8", errors="replace")[:500]
            raise InferenceClientError(
                f"HTTP {response.status_code}: {body}",
                error_type="HTTPStatusError",
                http_status=response.status_code,
            )
        async for line in response.aiter_lines():
            offset = time.perf_counter() - started
            if consume_sse_line(accumulator, line, offset):
                break
    finished = time.perf_counter()
    return _success_from_times(
        text=accumulator.text,
        finish_reason=accumulator.finish_reason,
        prompt_tokens=accumulator.prompt_tokens,
        completion_tokens=accumulator.completion_tokens,
        started=started,
        first_token_at=_first_token_at(started, accumulator),
        finished=finished,
        content_offsets=accumulator.content_arrival_offsets,
        streamed=True,
    )


async def _chat_unary(
    http: httpx.AsyncClient,
    settings: ClientSettings,
    payload: dict[str, Any],
    started: float,
) -> ChatResult:
    response = await http.post(settings.chat_url(), json=payload, headers=settings.headers())
    finished = time.perf_counter()
    if response.status_code >= 400:
        raise InferenceClientError(
            f"HTTP {response.status_code}: {response.text[:500]}",
            error_type="HTTPStatusError",
            http_status=response.status_code,
        )
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise InferenceClientError("Response was not JSON", error_type="InvalidJSON") from exc
    choices = body.get("choices") or []
    text = ""
    finish_reason = None
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            text = content
        finish = choices[0].get("finish_reason")
        if isinstance(finish, str):
            finish_reason = finish
    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None
    )
    return _success_from_times(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        started=started,
        first_token_at=None,
        finished=finished,
        content_offsets=[],
        streamed=False,
    )


def _first_token_at(started: float, accumulator: StreamAccumulator) -> float | None:
    if not accumulator.content_arrival_offsets:
        return None
    return started + accumulator.content_arrival_offsets[0]


def _success_from_times(
    *,
    text: str,
    finish_reason: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    started: float,
    first_token_at: float | None,
    finished: float,
    content_offsets: list[float],
    streamed: bool,
) -> ChatResult:
    ttft = time_to_first_token_seconds(started, first_token_at)
    e2e = end_to_end_seconds(started, finished)
    tpot = time_per_output_token_seconds(e2e, ttft, completion_tokens)
    gaps = tuple(
        content_offsets[index] - content_offsets[index - 1] for index in range(1, len(content_offsets))
    )
    return ChatResult(
        success=True,
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        e2e_seconds=e2e,
        output_tokens_per_second=output_tokens_per_second(completion_tokens, e2e),
        decode_tokens_per_second=decode_tokens_per_second(completion_tokens, e2e, ttft),
        inter_chunk_latency_seconds=gaps,
        content_chunk_count=len(content_offsets),
        streamed=streamed,
    )


def _failure(exc: InferenceClientError, *, streamed: bool, e2e_seconds: float) -> ChatResult:
    return ChatResult(
        success=False,
        text="",
        finish_reason=None,
        prompt_tokens=None,
        completion_tokens=None,
        ttft_seconds=None,
        tpot_seconds=None,
        e2e_seconds=e2e_seconds,
        output_tokens_per_second=None,
        decode_tokens_per_second=None,
        inter_chunk_latency_seconds=(),
        content_chunk_count=0,
        streamed=streamed,
        error_type=exc.error_type,
        error_message=str(exc)[:500],
        http_status=exc.http_status,
    )


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)
