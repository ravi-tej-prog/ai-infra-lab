"""Command line for the phase 1 client, benchmark, and vLLM command renderer."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from qwen_platform.benchmark_run import BenchmarkPlan, execute_plan
from qwen_platform.client import ClientSettings, chat
from qwen_platform.model_math import memory_report
from qwen_platform.prom import BenchMetrics
from qwen_platform.prompts import PROFILES, prompt_from_profile
from qwen_platform.reporting import render_summary_text, write_run_files
from qwen_platform.serving import cuda_launch_block_reason, load_profile, render_shell
from qwen_platform.sweep import load_sweep_config, load_sweep_rows, run_sweep, write_sweep_table

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVE_CONFIG = ROOT / "model-serving" / "configs" / "qwen2.5-14b-instruct.baseline.yaml"
DEFAULT_BENCH_CONFIG = ROOT / "benchmark" / "configs" / "methodology.yaml"
DEFAULT_SWEEP_CONFIG = ROOT / "benchmark" / "configs" / "ollama-sweep.yaml"
SWEEPS_DIR = ROOT / "benchmark" / "results" / "sweeps"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen-inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render-serve", help="Print the vLLM serve command for a profile")
    render.add_argument("--config", type=Path, default=DEFAULT_SERVE_CONFIG)
    render.set_defaults(handler=_render_serve)

    preflight = subparsers.add_parser("preflight", help="Print what this machine can and cannot run")
    preflight.set_defaults(handler=_preflight)

    math = subparsers.add_parser("model-math", help="Print the calculated Qwen2.5-14B memory model")
    math.set_defaults(handler=_model_math)

    infer = subparsers.add_parser("infer", help="Send one chat request and print timing")
    infer.add_argument("--base-url", default=os.environ.get("INFERENCE_BASE_URL", "http://127.0.0.1:8000/v1"))
    infer.add_argument("--model", default=os.environ.get("INFERENCE_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
    infer.add_argument("--engine", default=os.environ.get("INFERENCE_ENGINE", "vllm"))
    infer.add_argument("--profile", choices=sorted(PROFILES), default="short")
    infer.add_argument("--prompt", default=None)
    infer.add_argument("--max-tokens", type=int, default=32)
    infer.add_argument("--temperature", type=float, default=0.0)
    infer.add_argument("--timeout-seconds", type=float, default=180.0)
    infer.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    infer.set_defaults(handler=_infer)

    bench = subparsers.add_parser("bench", help="Run the closed-loop benchmark, or record not-run")
    bench.add_argument("--config", type=Path, default=None)
    bench.add_argument("--base-url", default=None)
    bench.add_argument("--model", default=None)
    bench.add_argument("--engine", default=None)
    bench.add_argument("--concurrency", type=int, default=None)
    bench.add_argument("--num-requests", type=int, default=None)
    bench.add_argument("--warmup-requests", type=int, default=None)
    bench.add_argument("--target-input-tokens", type=int, default=None)
    bench.add_argument("--prompt-profile", choices=sorted(PROFILES), default=None)
    bench.add_argument("--max-tokens", type=int, default=None)
    bench.add_argument("--temperature", type=float, default=None)
    bench.add_argument("--timeout-seconds", type=float, default=None)
    bench.add_argument("--output", type=Path, default=None)
    bench.add_argument(
        "--unique-prompts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefix each request with a random tag so the server cannot reuse a cached prompt",
    )
    bench.add_argument(
        "--mark-nvidia-baseline",
        action="store_true",
        help="Set only when the endpoint is a real NVIDIA vLLM server",
    )
    bench.set_defaults(handler=_bench)

    sweep = subparsers.add_parser("sweep", help="Run every input length at every concurrency level")
    sweep.add_argument("--config", type=Path, default=DEFAULT_SWEEP_CONFIG)
    sweep.add_argument("--label", required=True, help="Name of this server setting, e.g. 14b-parallel-default")
    sweep.add_argument("--base-url", default=None)
    sweep.add_argument("--model", default=None)
    sweep.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Serve Prometheus metrics on this port while the sweep runs, e.g. 9101",
    )
    sweep.add_argument(
        "--metrics-linger",
        type=float,
        default=15.0,
        help="Seconds to keep /metrics up after the last cell so Prometheus scrapes it",
    )
    sweep.add_argument(
        "--mark-nvidia-baseline",
        action="store_true",
        help="Set only when the endpoint is a real NVIDIA vLLM server",
    )
    sweep.set_defaults(handler=_sweep)

    report = subparsers.add_parser("sweep-report", help="Rebuild the table for one or more sweep directories")
    report.add_argument("dirs", type=Path, nargs="+")
    report.set_defaults(handler=_sweep_report)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _render_serve(args: argparse.Namespace) -> int:
    profile = load_profile(args.config)
    block = cuda_launch_block_reason()
    print(f"# profile: {profile.profile}")
    print(f"# target vLLM version: {profile.vllm_version}")
    print("# command below is rendered only; this process does not start vLLM")
    if block:
        print(f"# {block}")
    print(render_shell(profile))
    return 0


def _preflight(_args: argparse.Namespace) -> int:
    block = cuda_launch_block_reason()
    report = {
        "client_platform": platform.system(),
        "client_machine": platform.machine(),
        "python": platform.python_version(),
        "vllm_importable": _vllm_importable(),
        "cuda_launch": "blocked" if block else "not_blocked_by_os_check",
        "cuda_launch_reason": block,
        "local_checks": [
            "serving profile renders a vllm serve command",
            "metric formulas",
            "OpenAI-compatible client against a mock server or Ollama",
        ],
        "nvidia_required": [
            "vllm serve Qwen/Qwen2.5-14B-Instruct",
            "BF16 GPU memory residency",
            "KV cache utilization",
            "vLLM concurrency baseline",
        ],
    }
    print(json.dumps(report, indent=2))
    return 0


def _model_math(_args: argparse.Namespace) -> int:
    print(json.dumps(memory_report().as_dict(), indent=2))
    return 0


def _infer(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else prompt_from_profile(args.profile)
    settings = ClientSettings(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        api_key=_api_key(),
    )
    result = asyncio.run(
        chat(
            settings,
            prompt,
            max_output_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream,
        )
    )
    payload = result.as_dict()
    payload["engine"] = args.engine
    payload["base_url"] = args.base_url
    payload["model"] = args.model
    payload["prompt_profile"] = None if args.prompt is not None else args.profile
    payload["max_output_tokens"] = args.max_tokens
    if args.engine == "vllm":
        payload["verification"] = "REQUIRES_NVIDIA_GPU"
        payload["vllm_gpu_baseline"] = False
    elif args.engine == "ollama":
        payload["verification"] = "LOCAL_VERIFIED"
        payload["vllm_gpu_baseline"] = False
    else:
        payload["verification"] = "CLIENT_ONLY"
        payload["vllm_gpu_baseline"] = False
    print(json.dumps(payload, indent=2))
    return 0 if result.success else 2


def _bench(args: argparse.Namespace) -> int:
    config_path = args.config or DEFAULT_BENCH_CONFIG
    file_values = _load_bench_file(config_path)
    plan = BenchmarkPlan(
        base_url=str(
            _pick(
                args.base_url,
                os.environ.get("INFERENCE_BASE_URL"),
                file_values.get("base_url"),
                "http://127.0.0.1:8000/v1",
            )
        ),
        model=str(
            _pick(
                args.model,
                os.environ.get("INFERENCE_MODEL"),
                file_values.get("model"),
                "Qwen/Qwen2.5-14B-Instruct",
            )
        ),
        engine=str(_pick(args.engine, os.environ.get("INFERENCE_ENGINE"), file_values.get("engine"), "vllm")),
        concurrency=int(_pick(args.concurrency, None, file_values.get("concurrency"), 1)),
        num_requests=int(_pick(args.num_requests, None, file_values.get("num_requests"), 4)),
        warmup_requests=int(_pick(args.warmup_requests, None, file_values.get("warmup_requests"), 0)),
        max_output_tokens=int(_pick(args.max_tokens, None, file_values.get("max_output_tokens"), 64)),
        temperature=float(_pick(args.temperature, None, file_values.get("temperature"), 0.0)),
        timeout_seconds=float(_pick(args.timeout_seconds, None, file_values.get("timeout_seconds"), 300.0)),
        target_input_tokens=_optional_int(
            _pick(args.target_input_tokens, None, file_values.get("target_input_tokens"), None)
        ),
        prompt_profile=_pick(args.prompt_profile, None, file_values.get("prompt_profile"), None),
        api_key=_api_key(),
        mark_nvidia_baseline=bool(args.mark_nvidia_baseline),
        unique_prompts=bool(_pick(args.unique_prompts, None, file_values.get("unique_prompts"), False)),
    )
    if isinstance(plan.prompt_profile, str) and plan.prompt_profile.lower() == "null":
        plan = _replace(plan, prompt_profile=None)
    if plan.prompt_profile is None and plan.target_input_tokens is None:
        plan = _replace(plan, target_input_tokens=128)
    document = asyncio.run(execute_plan(plan))
    pending = file_values.get("pending_concurrency_levels") or []
    if pending and args.concurrency is None:
        document["notes"].append(
            "The methodology lists later concurrency levels "
            f"{pending}. This command measured one level only."
        )
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "benchmark" / "results" / "runs" / f"{stamp}-{plan.engine}-c{plan.concurrency}.json"
    paths = write_run_files(document, output)
    sys.stdout.write(render_summary_text(document))
    print(f"json: {paths['json']}")
    print(f"csv: {paths['csv']}")
    print(f"summary: {paths['summary']}")
    if document["status"] == "not_run":
        return 2
    if document["status"] == "failed":
        return 2
    return 0


def _sweep(args: argparse.Namespace) -> int:
    config = load_sweep_config(args.config, base_url=args.base_url, model=args.model)
    out_dir = SWEEPS_DIR / args.label
    print(
        f"sweep {args.label}: model={config.model} inputs={config.input_tokens} "
        f"concurrency={config.concurrency} max_output={config.max_output_tokens} "
        f"unique_prompts={config.unique_prompts}",
        flush=True,
    )
    metrics = None
    if args.metrics_port is not None:
        metrics = BenchMetrics(args.metrics_port, args.label)
        print(f"metrics: http://127.0.0.1:{args.metrics_port}/metrics", flush=True)
    rows = asyncio.run(
        run_sweep(
            config,
            out_dir,
            api_key=_api_key(),
            metrics=metrics,
            mark_nvidia_baseline=bool(args.mark_nvidia_baseline),
        )
    )
    paths = write_sweep_table(rows, out_dir, f"{args.label} ({config.model})")
    print(paths["markdown"].read_text(encoding="utf-8"))
    print(f"table: {paths['markdown']}")
    if metrics:
        metrics.linger(args.metrics_linger)
    if rows and rows[-1]["status"] == "not_run":
        return 2
    return 0


def _sweep_report(args: argparse.Namespace) -> int:
    for sweep_dir in args.dirs:
        rows = load_sweep_rows(sweep_dir)
        if not rows:
            raise FileNotFoundError(f"No run files in {sweep_dir}")
        model = rows[0]["model"]
        paths = write_sweep_table(rows, sweep_dir, f"{sweep_dir.name} ({model})")
        print(paths["markdown"].read_text(encoding="utf-8"))
    return 0


def _load_bench_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    endpoint = raw.get("endpoint") or {}
    workload = raw.get("workload") or {}
    sweep = raw.get("sweep") or {}
    if not isinstance(endpoint, dict) or not isinstance(workload, dict) or not isinstance(sweep, dict):
        raise ValueError(f"{path} has an unexpected shape")
    concurrency = sweep.get("concurrency")
    pending_levels: list[int] = []
    if isinstance(concurrency, list):
        if not concurrency:
            raise ValueError("sweep.concurrency must not be empty")
        pending_levels = [int(level) for level in concurrency[1:]]
        concurrency = int(concurrency[0])
    values: dict[str, object] = {
        "base_url": endpoint.get("base_url"),
        "model": endpoint.get("model"),
        "engine": endpoint.get("engine"),
        "concurrency": concurrency,
        "num_requests": sweep.get("measured_requests"),
        "warmup_requests": sweep.get("warmup_requests"),
        "target_input_tokens": workload.get("target_input_tokens"),
        "prompt_profile": workload.get("prompt_profile"),
        "max_output_tokens": workload.get("max_output_tokens"),
        "temperature": workload.get("temperature"),
        "unique_prompts": workload.get("unique_prompts"),
        "timeout_seconds": sweep.get("timeout_seconds"),
        "pending_concurrency_levels": pending_levels,
    }
    return values


def _pick(cli_value: object, env_value: object, file_value: object, default: object) -> object:
    """First non-null wins: CLI, then environment, then the YAML file, then the default."""

    for candidate in (cli_value, env_value, file_value, default):
        if candidate is not None:
            return candidate
    return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _replace(plan: BenchmarkPlan, **changes: object) -> BenchmarkPlan:
    data = plan.__dict__.copy()
    data.update(changes)
    return BenchmarkPlan(**data)


def _api_key() -> str | None:
    value = os.environ.get("INFERENCE_API_KEY", "").strip()
    if not value or value == "EMPTY":
        return None
    return value


def _vllm_importable() -> bool:
    try:
        import vllm  # type: ignore[import-not-found]
    except Exception:
        return False
    return bool(getattr(vllm, "__version__", True))
