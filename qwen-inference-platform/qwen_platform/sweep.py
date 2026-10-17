"""Grid sweep over input length and concurrency, plus a table across the runs.

One sweep is one server setting: one model, one engine configuration. Each
cell of the grid is an ordinary benchmark run written with `write_run_files`,
so a single cell can be read on its own. The table reads every run file in a
sweep directory and puts one row per cell.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qwen_platform.benchmark_run import BenchmarkPlan, execute_plan
from qwen_platform.prom import BenchMetrics
from qwen_platform.reporting import write_run_files


@dataclass(frozen=True)
class SweepConfig:
    base_url: str
    model: str
    engine: str
    input_tokens: list[int]
    concurrency: list[int]
    max_output_tokens: int
    temperature: float
    unique_prompts: bool
    warmup_requests: int
    min_requests: int
    requests_per_slot: int
    timeout_seconds: float

    def requests_for(self, concurrency: int) -> int:
        """Enough requests that every slot is refilled a few times at this level."""

        return max(self.min_requests, self.requests_per_slot * concurrency)


def load_sweep_config(path: Path, *, base_url: str | None = None, model: str | None = None) -> SweepConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    endpoint = raw.get("endpoint") or {}
    workload = raw.get("workload") or {}
    sweep = raw.get("sweep") or {}
    input_tokens = [int(value) for value in workload.get("input_tokens") or []]
    concurrency = [int(value) for value in sweep.get("concurrency") or []]
    if not input_tokens:
        raise ValueError(f"{path}: workload.input_tokens must list at least one length")
    if not concurrency:
        raise ValueError(f"{path}: sweep.concurrency must list at least one level")
    return SweepConfig(
        base_url=str(base_url or endpoint.get("base_url") or "http://127.0.0.1:11434/v1"),
        model=str(model or endpoint.get("model") or "qwen2.5:14b"),
        engine=str(endpoint.get("engine") or "ollama"),
        input_tokens=input_tokens,
        concurrency=concurrency,
        max_output_tokens=int(workload.get("max_output_tokens", 128)),
        temperature=float(workload.get("temperature", 0.0)),
        unique_prompts=bool(workload.get("unique_prompts", True)),
        warmup_requests=int(sweep.get("warmup_requests", 1)),
        min_requests=int(sweep.get("min_requests", 8)),
        requests_per_slot=int(sweep.get("requests_per_slot", 2)),
        timeout_seconds=float(sweep.get("timeout_seconds", 300)),
    )


async def run_sweep(
    config: SweepConfig,
    out_dir: Path,
    *,
    api_key: str | None = None,
    metrics: BenchMetrics | None = None,
    mark_nvidia_baseline: bool = False,
) -> list[dict[str, Any]]:
    """Run every cell.

    A server that is down stops the whole sweep. A length whose run fails
    outright skips its higher concurrency levels, which would only fail slower.
    """

    rows: list[dict[str, Any]] = []
    for input_tokens in config.input_tokens:
        for concurrency in config.concurrency:
            plan = BenchmarkPlan(
                base_url=config.base_url,
                model=config.model,
                engine=config.engine,
                concurrency=concurrency,
                num_requests=config.requests_for(concurrency),
                warmup_requests=config.warmup_requests,
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
                timeout_seconds=config.timeout_seconds,
                target_input_tokens=input_tokens,
                api_key=api_key,
                unique_prompts=config.unique_prompts,
                mark_nvidia_baseline=mark_nvidia_baseline,
            )
            observer = None
            if metrics:
                observer = metrics.cell(
                    model=config.model,
                    engine=config.engine,
                    input_tokens=input_tokens,
                    concurrency=concurrency,
                )
            document = await execute_plan(plan, observer)
            write_run_files(document, out_dir / f"in{input_tokens}-c{concurrency}.json")
            row = table_row(document)
            rows.append(row)
            print(format_progress(row), flush=True)
            if document["status"] == "not_run":
                return rows
            if document["status"] == "failed":
                break
    return rows


def table_row(document: dict[str, Any]) -> dict[str, Any]:
    summary = document.get("summary") or {}

    def stat(metric: str, key: str) -> float | None:
        payload = summary.get(metric) or {}
        return payload.get(key)

    return {
        "model": document.get("model"),
        "status": document.get("status"),
        "target_input_tokens": document.get("target_input_tokens"),
        "input_tokens": stat("input_tokens", "mean"),
        "concurrency": document.get("concurrency"),
        "requests": document.get("num_requests"),
        "failures": summary.get("failure_count"),
        "ttft_p50": stat("ttft_seconds", "p50"),
        "ttft_p95": stat("ttft_seconds", "p95"),
        "tpot_p50": stat("tpot_seconds", "p50"),
        "tpot_p95": stat("tpot_seconds", "p95"),
        "e2e_p50": stat("e2e_seconds", "p50"),
        "e2e_p95": stat("e2e_seconds", "p95"),
        "aggregate_output_tps": summary.get("aggregate_output_tokens_per_second"),
        "requests_per_second": summary.get("request_throughput_per_second"),
    }


def load_sweep_rows(sweep_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sweep_dir.glob("in*-c*.json"):
        rows.append(table_row(json.loads(path.read_text(encoding="utf-8"))))
    rows.sort(key=lambda row: (row["target_input_tokens"] or 0, row["concurrency"] or 0))
    return rows


COLUMNS = [
    ("input_tokens", "input tok", 0),
    ("concurrency", "conc", 0),
    ("requests", "reqs", 0),
    ("failures", "fail", 0),
    ("ttft_p50", "TTFT p50 s", 3),
    ("ttft_p95", "TTFT p95 s", 3),
    ("tpot_p50", "TPOT p50 ms", "ms"),
    ("tpot_p95", "TPOT p95 ms", "ms"),
    ("e2e_p50", "E2E p50 s", 2),
    ("e2e_p95", "E2E p95 s", 2),
    ("aggregate_output_tps", "agg tok/s", 1),
    ("requests_per_second", "req/s", 2),
]


def render_markdown(rows: list[dict[str, Any]], title: str) -> str:
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(label for _, label, _ in COLUMNS) + " |")
    lines.append("|" + "---:|" * len(COLUMNS))
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(key), digits) for key, _, digits in COLUMNS) + " |")
    lines.append("")
    lines.append(
        "TTFT and E2E are client-observed and include queue wait. TPOT is "
        "(E2E - TTFT) / (output tokens - 1). Input tokens is the API prompt count, chat template included."
    )
    return "\n".join(lines) + "\n"


def write_sweep_table(rows: list[dict[str, Any]], sweep_dir: Path, title: str) -> dict[str, Path]:
    markdown_path = sweep_dir / "sweep.md"
    csv_path = sweep_dir / "sweep.csv"
    markdown_path.write_text(render_markdown(rows, title), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    return {"markdown": markdown_path, "csv": csv_path}


def format_progress(row: dict[str, Any]) -> str:
    return (
        f"in={row['target_input_tokens']:>5} c={row['concurrency']:>2} {row['status']:<24} "
        f"ttft_p95={_cell(row['ttft_p95'], 3)}s tpot_p50={_cell(row['tpot_p50'], 'ms')}ms "
        f"e2e_p95={_cell(row['e2e_p95'], 2)}s agg={_cell(row['aggregate_output_tps'], 1)} tok/s"
    )


def _cell(value: Any, digits: int | str) -> str:
    if value is None:
        return "-"
    if digits == "ms":
        return f"{float(value) * 1000:.1f}"
    return f"{float(value):.{digits}f}"
