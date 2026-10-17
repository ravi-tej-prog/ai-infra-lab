"""Prometheus exporter for client-observed benchmark metrics.

Ollama has no /metrics endpoint, so the benchmark client exports what it
measures. On a GPU node the same Prometheus also scrapes vLLM's own /metrics
and the DCGM exporter. Client TTFT minus vLLM's server TTFT is the time spent
outside the engine: network, HTTP, and any queue in front of vLLM.

Histograms give Prometheus approximate percentiles from bucket counts. The
run JSON keeps the exact per-request values. Bucket edges are chosen around
the latencies a 3B-14B model shows on one Mac or one GPU.

prometheus_client is imported only when metrics are turned on.
"""

from __future__ import annotations

import time
from typing import Any

from qwen_platform.client import ChatResult

LABELS = ("run_label", "model", "engine", "input_tokens", "concurrency")

TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)
TPOT_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1)
E2E_BUCKETS = (0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


class BenchMetrics:
    """One exporter per process. Each benchmark cell gets its own label set."""

    def __init__(self, port: int, run_label: str) -> None:
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server
        except ImportError as exc:
            raise ValueError(
                "--metrics-port needs prometheus-client: .venv/bin/python -m pip install -e '.[dev]'"
            ) from exc

        self.port = port
        self.run_label = run_label
        self.registry = CollectorRegistry()
        self.ttft = Histogram(
            "llm_client_ttft_seconds",
            "Client-observed time to first content token",
            LABELS,
            buckets=TTFT_BUCKETS,
            registry=self.registry,
        )
        self.tpot = Histogram(
            "llm_client_tpot_seconds",
            "Client-observed time per output token after the first",
            LABELS,
            buckets=TPOT_BUCKETS,
            registry=self.registry,
        )
        self.e2e = Histogram(
            "llm_client_e2e_seconds",
            "Client-observed request time from send to end of stream",
            LABELS,
            buckets=E2E_BUCKETS,
            registry=self.registry,
        )
        self.requests = Counter(
            "llm_client_requests",
            "Measured requests by outcome",
            (*LABELS, "status"),
            registry=self.registry,
        )
        self.output_tokens = Counter(
            "llm_client_output_tokens",
            "Output tokens reported by the API",
            LABELS,
            registry=self.registry,
        )
        self.prompt_tokens = Counter(
            "llm_client_prompt_tokens",
            "Prompt tokens reported by the API, chat template included",
            LABELS,
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "llm_client_in_flight_requests",
            "Measured requests sent and not yet finished",
            LABELS,
            registry=self.registry,
        )
        self.target_concurrency = Gauge(
            "llm_client_target_concurrency",
            "Concurrency level of the cell that is running now, 0 between cells",
            ("run_label", "model"),
            registry=self.registry,
        )
        start_http_server(port, registry=self.registry)

    def cell(self, *, model: str, engine: str, input_tokens: int | None, concurrency: int) -> CellObserver:
        labels = {
            "run_label": self.run_label,
            "model": model,
            "engine": engine,
            "input_tokens": str(input_tokens) if input_tokens is not None else "profile",
            "concurrency": str(concurrency),
        }
        return CellObserver(self, labels)

    def linger(self, seconds: float) -> None:
        """Keep /metrics up so Prometheus scrapes the last cell before the process exits."""

        if seconds > 0:
            print(f"metrics: holding :{self.port}/metrics for {seconds:.0f}s for a final scrape", flush=True)
            time.sleep(seconds)


class CellObserver:
    """Hooks the benchmark runner calls around each measured request."""

    def __init__(self, metrics: BenchMetrics, labels: dict[str, str]) -> None:
        self.metrics = metrics
        self.labels = labels

    def cell_started(self) -> None:
        self.metrics.target_concurrency.labels(
            run_label=self.labels["run_label"], model=self.labels["model"]
        ).set(float(self.labels["concurrency"]))

    def cell_finished(self) -> None:
        self.metrics.target_concurrency.labels(
            run_label=self.labels["run_label"], model=self.labels["model"]
        ).set(0)

    def request_started(self) -> None:
        self.metrics.in_flight.labels(**self.labels).inc()

    def request_finished(self, result: ChatResult) -> None:
        metrics = self.metrics
        metrics.in_flight.labels(**self.labels).dec()
        row: dict[str, Any] = result.as_dict()
        metrics.requests.labels(**self.labels, status="success" if row["success"] else "error").inc()
        if not row["success"]:
            return
        _observe(metrics.ttft, self.labels, row.get("ttft_seconds"))
        _observe(metrics.tpot, self.labels, row.get("tpot_seconds"))
        _observe(metrics.e2e, self.labels, row.get("e2e_seconds"))
        if isinstance(row.get("completion_tokens"), int):
            metrics.output_tokens.labels(**self.labels).inc(row["completion_tokens"])
        if isinstance(row.get("prompt_tokens"), int):
            metrics.prompt_tokens.labels(**self.labels).inc(row["prompt_tokens"])


def _observe(histogram: Any, labels: dict[str, str], value: object) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        histogram.labels(**labels).observe(float(value))
