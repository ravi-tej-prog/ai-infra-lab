"""Parse OpenAI-compatible server-sent events.

One SSE chunk is not guaranteed to contain exactly one token. Callers keep
the API `usage.completion_tokens` value as the token count and treat chunk
gaps as inter-chunk latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamAccumulator:
    text_parts: list[str] = field(default_factory=list)
    content_arrival_offsets: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw_events: int = 0

    @property
    def text(self) -> str:
        return "".join(self.text_parts)


def parse_sse_data(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def apply_sse_payload(accumulator: StreamAccumulator, payload: dict[str, Any], offset_seconds: float) -> None:
    accumulator.raw_events += 1
    usage = payload.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int):
            accumulator.prompt_tokens = prompt
        if isinstance(completion, int):
            accumulator.completion_tokens = completion

    choices = payload.get("choices") or []
    if not choices:
        return
    choice = choices[0]
    finish = choice.get("finish_reason")
    if isinstance(finish, str):
        accumulator.finish_reason = finish

    delta = choice.get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str) and content:
        accumulator.text_parts.append(content)
        accumulator.content_arrival_offsets.append(offset_seconds)


def consume_sse_line(accumulator: StreamAccumulator, line: str, offset_seconds: float) -> bool:
    """Apply one line. Return True when the stream sends the terminal marker."""

    data = parse_sse_data(line)
    if data is None:
        return False
    if data == "[DONE]":
        return True
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SSE data was not JSON: {data[:180]}") from exc
    if not isinstance(payload, dict):
        raise ValueError("SSE JSON payload must be an object")
    apply_sse_payload(accumulator, payload, offset_seconds)
    return False
