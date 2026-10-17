"""Load a serving profile and render a `vllm serve` command.

The flag names are limited to arguments checked on the vLLM 0.31.0 engine
argument page (docs.vllm.ai/en/latest/configuration/engine_args/, 2026-10-05)
plus `--host` and `--port`, which the Qwen vLLM deployment guide documents as
frontend arguments of `vllm serve`.

Rendering does not install or start vLLM. Launching is refused on macOS.
"""

from __future__ import annotations

import json
import platform
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TARGET_VLLM_VERSION = "0.31.0"

# Verified on the engine arguments page. Values of null are omitted.
ENGINE_FLAGS: dict[str, str] = {
    "dtype": "--dtype",
    "quantization": "--quantization",
    "max_model_len": "--max-model-len",
    "tensor_parallel_size": "--tensor-parallel-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "kv_cache_dtype": "--kv-cache-dtype",
    "generation_config": "--generation-config",
    "override_generation_config": "--override-generation-config",
}

FRONTEND_FLAGS: dict[str, str] = {
    "host": "--host",
    "port": "--port",
}

DTYPE_CHOICES = {"auto", "bfloat16", "float", "float16", "float32", "half"}

# The engine arguments page lists these `kv_cache_dtype` choices.
KV_CACHE_DTYPE_CHOICES = {
    "auto",
    "bfloat16",
    "float16",
    "fp8",
    "fp8_ds_mla",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_per_token_head",
    "int4_per_token_head",
    "int8_per_token_head",
    "nvfp4",
    "nvfp4_4over6",
    "turboquant_3bit_nc",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_k8v4",
}


@dataclass(frozen=True)
class ServeProfile:
    profile: str
    vllm_version: str
    model: str
    host: str
    port: int
    engine: dict[str, Any]


def load_profile(path: Path) -> ServeProfile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    _require_keys(raw, {"schema_version", "profile", "vllm_version", "model", "server", "engine"})
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    server = raw["server"]
    engine = raw["engine"]
    if not isinstance(server, dict) or not isinstance(engine, dict):
        raise ValueError("server and engine must be mappings")
    unknown_server = set(server) - set(FRONTEND_FLAGS)
    if unknown_server:
        raise ValueError(f"Unknown server keys: {sorted(unknown_server)}")
    unknown_engine = set(engine) - set(ENGINE_FLAGS)
    if unknown_engine:
        raise ValueError(
            "Unknown engine keys: "
            f"{sorted(unknown_engine)}. The renderer only emits flags checked for vLLM {TARGET_VLLM_VERSION}."
        )
    model = _string(raw["model"], "model")
    host = _string(server.get("host"), "server.host")
    port = _int(server.get("port"), "server.port")
    if not 1 <= port <= 65535:
        raise ValueError("server.port must be between 1 and 65535")
    _validate_engine(engine)
    version = _string(raw["vllm_version"], "vllm_version")
    return ServeProfile(
        profile=_string(raw["profile"], "profile"),
        vllm_version=version,
        model=model,
        host=host,
        port=port,
        engine=dict(engine),
    )


def render_argv(profile: ServeProfile) -> list[str]:
    argv = ["vllm", "serve", profile.model, "--host", profile.host, "--port", str(profile.port)]
    for key, flag in ENGINE_FLAGS.items():
        value = profile.engine.get(key)
        if value is None:
            continue
        if key == "override_generation_config":
            argv.extend([flag, json.dumps(value, separators=(",", ":"))])
            continue
        if isinstance(value, bool):
            raise ValueError(f"{key} must not be a boolean in this profile")
        argv.extend([flag, str(value)])
    return argv


def render_shell(profile: ServeProfile) -> str:
    return " ".join(shlex.quote(part) for part in render_argv(profile))


def cuda_launch_block_reason(system_name: str | None = None) -> str | None:
    """Return the reason this machine must not launch vLLM, or None if it might.

    `None` does not mean CUDA is present. It means the OS is not macOS, so the
    operator still has to be on a Linux host with an NVIDIA driver.
    """

    system = system_name if system_name is not None else platform.system()
    if system == "Darwin":
        return (
            "This host is macOS. vLLM 0.31.0 on PyPI ships Linux wheels "
            "(manylinux x86_64 and aarch64) and a source tarball, not a macOS CUDA build. "
            "Apple Silicon has no NVIDIA GPU and no CUDA driver, so vLLM cannot run on CUDA here. "
            "Render the command on this Mac, and run it later on a Linux machine with NVIDIA GPUs."
        )
    return None


def _validate_engine(engine: dict[str, Any]) -> None:
    dtype = engine.get("dtype")
    if dtype is not None and dtype not in DTYPE_CHOICES:
        raise ValueError(f"dtype must be one of {sorted(DTYPE_CHOICES)}")
    quantization = engine.get("quantization")
    if quantization is not None and (not isinstance(quantization, str) or not quantization.strip()):
        raise ValueError("quantization must be a non-empty string or null")
    _optional_positive_int(engine.get("max_model_len"), "max_model_len")
    tp = engine.get("tensor_parallel_size")
    if tp is not None:
        parsed = _int(tp, "tensor_parallel_size")
        if parsed < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
    util = engine.get("gpu_memory_utilization")
    if util is not None:
        if isinstance(util, bool) or not isinstance(util, (int, float)):
            raise ValueError("gpu_memory_utilization must be a number")
        if not 0 < float(util) <= 1:
            raise ValueError("gpu_memory_utilization must be in the range (0, 1]")
    _optional_positive_int(engine.get("max_num_seqs"), "max_num_seqs")
    _optional_positive_int(engine.get("max_num_batched_tokens"), "max_num_batched_tokens")
    kv_dtype = engine.get("kv_cache_dtype")
    if kv_dtype is not None and kv_dtype not in KV_CACHE_DTYPE_CHOICES:
        raise ValueError("kv_cache_dtype is not in the vLLM 0.31 choice list")
    generation_config = engine.get("generation_config")
    if generation_config is not None and (
        not isinstance(generation_config, str) or not generation_config.strip()
    ):
        raise ValueError("generation_config must be a non-empty string or null")
    override = engine.get("override_generation_config")
    if override is not None and not isinstance(override, dict):
        raise ValueError("override_generation_config must be a mapping or null")


def _optional_positive_int(value: Any, name: str) -> None:
    if value is None:
        return
    parsed = _int(value, name)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")


def _require_keys(raw: dict[str, Any], keys: set[str]) -> None:
    missing = keys - set(raw)
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value
