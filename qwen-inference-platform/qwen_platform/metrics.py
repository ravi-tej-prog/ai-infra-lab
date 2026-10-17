"""Latency definitions used by the benchmark client.

These match the per-request formulas in vLLM's serving benchmark
(`benchmark_serving.py`): TTFT is the wait until the first output token,
and TPOT divides the remaining time by every output token except the first.

All times here are client-observed. They include network delay and any time
the request spent waiting before the server started the model. Server-side
histograms such as `vllm:time_to_first_token_seconds` are a later phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def time_to_first_token_seconds(
    request_start_seconds: float,
    first_token_seconds: float | None,
) -> float | None:
    if first_token_seconds is None:
        return None
    return first_token_seconds - request_start_seconds


def end_to_end_seconds(request_start_seconds: float, stream_end_seconds: float) -> float:
    return stream_end_seconds - request_start_seconds


def time_per_output_token_seconds(
    e2e_seconds: float,
    ttft_seconds: float | None,
    output_tokens: int | None,
) -> float | None:
    """TPOT = (E2E - TTFT) / (output_tokens - 1), when more than one token exists.

    The first token's wait is TTFT, which is dominated by prefill. Each later
    token is a decode step. With a single output token there is no decode
    interval, so TPOT is undefined.
    """

    if ttft_seconds is None or output_tokens is None or output_tokens <= 1:
        return None
    decode_seconds = e2e_seconds - ttft_seconds
    if decode_seconds < 0:
        return None
    return decode_seconds / (output_tokens - 1)


def output_tokens_per_second(
    output_tokens: int | None,
    e2e_seconds: float,
) -> float | None:
    """Tokens produced divided by the full request time, prefill included."""

    if output_tokens is None or e2e_seconds <= 0:
        return None
    return output_tokens / e2e_seconds


def decode_tokens_per_second(
    output_tokens: int | None,
    e2e_seconds: float,
    ttft_seconds: float | None,
) -> float | None:
    """Decode throughput. This is 1 / TPOT when TPOT is defined."""

    tpot = time_per_output_token_seconds(e2e_seconds, ttft_seconds, output_tokens)
    if tpot is None or tpot <= 0:
        return None
    return 1.0 / tpot


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: list[float], percent: float) -> float | None:
    """Linear percentile, matching NumPy's default interpolation.

    `percent` is in the range 0 to 100. An empty sample has no percentile.
    """

    if not values:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


@dataclass(frozen=True)
class Distribution:
    count: int
    mean: float | None
    p50: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": _round(self.mean),
            "p50": _round(self.p50),
            "p95": _round(self.p95),
            "min": _round(self.minimum),
            "max": _round(self.maximum),
        }


def distribution(values: list[float]) -> Distribution:
    return Distribution(
        count=len(values),
        mean=mean(values),
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        minimum=min(values) if values else None,
        maximum=max(values) if values else None,
    )


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)
