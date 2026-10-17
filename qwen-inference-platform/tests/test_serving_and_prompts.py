from pathlib import Path

import pytest

from qwen_platform.cli import main
from qwen_platform.prompts import CharEstimateCounter, prompt_for_target_tokens, prompt_from_profile
from qwen_platform.serving import cuda_launch_block_reason, load_profile, render_argv

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "model-serving" / "configs" / "qwen2.5-14b-instruct.baseline.yaml"
AWQ = ROOT / "model-serving" / "configs" / "qwen2.5-14b-instruct.awq.yaml"


def test_baseline_command_uses_checked_flags_and_skips_nulls() -> None:
    argv = render_argv(load_profile(BASELINE))
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen2.5-14B-Instruct"]
    assert "--dtype" in argv
    assert "auto" in argv
    assert "--max-model-len" in argv
    assert "8192" in argv
    assert "--tensor-parallel-size" in argv
    assert "--gpu-memory-utilization" in argv
    assert "--kv-cache-dtype" in argv
    assert "--generation-config" in argv
    assert "vllm" in argv
    assert "--host" in argv
    assert "--port" in argv
    assert "--quantization" not in argv
    assert "--max-num-seqs" not in argv
    assert "--max-num-batched-tokens" not in argv
    assert "--rope-scaling" not in argv


def test_awq_profile_changes_the_checkpoint_and_not_the_flag_set() -> None:
    argv = render_argv(load_profile(AWQ))
    assert "Qwen/Qwen2.5-14B-Instruct-AWQ" in argv
    assert "--quantization" not in argv


def test_unknown_engine_key_is_rejected(tmp_path: Path) -> None:
    text = BASELINE.read_text(encoding="utf-8")
    broken = text.replace("dtype: auto", "dtype: auto\n  invented_flag: 1")
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown engine keys"):
        load_profile(path)


def test_render_serve_prints_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["render-serve", "--config", str(BASELINE)]) == 0
    output = capsys.readouterr().out
    assert "vllm serve Qwen/Qwen2.5-14B-Instruct" in output
    if cuda_launch_block_reason() is not None:
        assert "macOS" in output


def test_macos_cannot_launch_cuda_vllm() -> None:
    reason = cuda_launch_block_reason("Darwin")
    assert reason is not None
    assert "macOS" in reason
    assert cuda_launch_block_reason("Linux") is None


def test_prompt_profiles_grow() -> None:
    short = prompt_from_profile("short")
    medium = prompt_from_profile("medium")
    long = prompt_from_profile("long")
    assert len(short) < len(medium) < len(long)


def test_target_prompt_meets_the_estimate() -> None:
    counter = CharEstimateCounter()
    prompt = prompt_for_target_tokens(200, counter)
    assert counter(prompt) >= 200
