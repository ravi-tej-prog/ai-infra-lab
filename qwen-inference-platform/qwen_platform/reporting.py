"""Turn benchmark rows into JSON, CSV, and a text summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from qwen_platform.metrics import distribution, mean


def summarize_requests(requests: list[dict[str, Any]], wall_seconds: float | None) -> dict[str, Any]:
    successful = [row for row in requests if row.get("success")]
    failures = [row for row in requests if not row.get("success")]

    def column(name: str) -> list[float]:
        values: list[float] = []
        for row in successful:
            value = row.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        return values

    output_tokens = [
        int(row["output_tokens"])
        for row in successful
        if isinstance(row.get("output_tokens"), int)
    ]
    input_tokens = [
        int(row["input_tokens"])
        for row in successful
        if isinstance(row.get("input_tokens"), int)
    ]
    total_output = sum(output_tokens)
    aggregate_output_tps = None
    if wall_seconds is not None and wall_seconds > 0 and output_tokens:
        aggregate_output_tps = round(total_output / wall_seconds, 6)
    request_throughput = None
    if wall_seconds is not None and wall_seconds > 0:
        request_throughput = round(len(successful) / wall_seconds, 6)

    itl_values: list[float] = []
    for row in successful:
        for gap in row.get("inter_chunk_latency_seconds") or []:
            if isinstance(gap, (int, float)) and not isinstance(gap, bool):
                itl_values.append(float(gap))

    return {
        "success_count": len(successful),
        "failure_count": len(failures),
        "input_tokens": distribution(input_tokens).as_dict() if input_tokens else _empty_distribution(),
        "output_tokens": distribution([float(item) for item in output_tokens]).as_dict()
        if output_tokens
        else _empty_distribution(),
        "ttft_seconds": distribution(column("ttft_seconds")).as_dict(),
        "tpot_seconds": distribution(column("tpot_seconds")).as_dict(),
        "e2e_seconds": distribution(column("e2e_seconds")).as_dict(),
        "output_tokens_per_second_per_request": distribution(column("output_tokens_per_second")).as_dict(),
        "mean_inter_chunk_latency_seconds": _round(mean(itl_values)),
        "inter_chunk_latency_seconds": distribution(itl_values).as_dict(),
        "total_output_tokens": total_output,
        "aggregate_output_tokens_per_second": aggregate_output_tps,
        "request_throughput_per_second": request_throughput,
        "errors": [
            {
                "request_index": row.get("request_index"),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
                "http_status": row.get("http_status"),
            }
            for row in failures
        ],
    }


def render_summary_text(document: dict[str, Any]) -> str:
    summary = document.get("summary") or {}
    lines = [
        "Benchmark summary",
        f"status: {document.get('status')}",
        f"verification: {document.get('verification')}",
        f"vllm_gpu_baseline: {document.get('vllm_gpu_baseline')}",
        f"engine: {document.get('engine')}",
        f"model: {document.get('model')}",
        f"endpoint: {document.get('endpoint')}",
        f"timestamp: {document.get('timestamp')}",
        f"concurrency: {document.get('concurrency')}",
        f"measured_requests: {document.get('num_requests')}",
        f"success_count: {summary.get('success_count')}",
        f"failure_count: {summary.get('failure_count')}",
        f"wall_seconds: {document.get('wall_seconds')}",
        _metric_line("TTFT seconds", summary.get("ttft_seconds")),
        _metric_line("TPOT seconds", summary.get("tpot_seconds")),
        _metric_line("E2E seconds", summary.get("e2e_seconds")),
        (
            "aggregate output tokens/second: "
            f"{summary.get('aggregate_output_tokens_per_second')}"
        ),
        (
            "request throughput per second: "
            f"{summary.get('request_throughput_per_second')}"
        ),
        (
            "mean inter-chunk latency seconds: "
            f"{summary.get('mean_inter_chunk_latency_seconds')}"
        ),
    ]
    notes = document.get("notes") or []
    if notes:
        lines.append("notes:")
        lines.extend(f"- {note}" for note in notes)
    errors = summary.get("errors") or []
    if errors:
        lines.append("errors:")
        for error in errors:
            lines.append(
                f"- request {error.get('request_index')}: "
                f"{error.get('error_type')}: {error.get('error_message')}"
            )
    return "\n".join(lines) + "\n"


def write_run_files(document: dict[str, Any], json_path: Path) -> dict[str, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    csv_path = json_path.with_suffix(".csv")
    text_path = json_path.with_suffix(".summary.txt")
    _write_csv(document, csv_path)
    text_path.write_text(render_summary_text(document), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "summary": text_path}


def _write_csv(document: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "timestamp",
        "model",
        "endpoint",
        "engine",
        "concurrency",
        "request_index",
        "input_tokens",
        "output_tokens",
        "ttft_seconds",
        "tpot_seconds",
        "e2e_seconds",
        "output_tokens_per_second",
        "success",
        "error_type",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in document.get("requests") or []:
            writer.writerow(
                {
                    "timestamp": document.get("timestamp"),
                    "model": document.get("model"),
                    "endpoint": document.get("endpoint"),
                    "engine": document.get("engine"),
                    "concurrency": document.get("concurrency"),
                    "request_index": row.get("request_index"),
                    "input_tokens": row.get("input_tokens"),
                    "output_tokens": row.get("output_tokens"),
                    "ttft_seconds": row.get("ttft_seconds"),
                    "tpot_seconds": row.get("tpot_seconds"),
                    "e2e_seconds": row.get("e2e_seconds"),
                    "output_tokens_per_second": row.get("output_tokens_per_second"),
                    "success": row.get("success"),
                    "error_type": row.get("error_type"),
                    "error_message": row.get("error_message"),
                }
            )


def _metric_line(label: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        return f"{label}: unavailable"
    return (
        f"{label}: count={payload.get('count')} mean={payload.get('mean')} "
        f"p50={payload.get('p50')} p95={payload.get('p95')} "
        f"min={payload.get('min')} max={payload.get('max')}"
    )


def _empty_distribution() -> dict[str, float | int | None]:
    return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)
