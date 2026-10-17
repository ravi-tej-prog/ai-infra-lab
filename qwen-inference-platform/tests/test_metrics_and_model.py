import pytest

from qwen_platform.metrics import (
    decode_tokens_per_second,
    distribution,
    output_tokens_per_second,
    percentile,
    time_per_output_token_seconds,
)
from qwen_platform.model_math import (
    PUBLISHED_BF16_PARAMETER_COUNT,
    kv_bytes_for_tokens,
    kv_bytes_per_token,
    memory_report,
    parameter_count,
    tensor_parallel_candidates,
)


def test_parameter_count_matches_huggingface_safetensors_header() -> None:
    assert parameter_count() == PUBLISHED_BF16_PARAMETER_COUNT == 14_770_033_664


def test_kv_cache_bytes_follow_grouped_query_attention() -> None:
    assert kv_bytes_per_token(2) == 196_608
    assert kv_bytes_for_tokens(8192) == 1_610_612_736
    assert kv_bytes_for_tokens(32_768) == 4 * kv_bytes_for_tokens(8192)
    assert kv_bytes_for_tokens(131_072) == 16 * kv_bytes_for_tokens(8192)


def test_tensor_parallel_candidates_divide_both_head_counts() -> None:
    assert tensor_parallel_candidates() == [1, 2, 4, 8]


def test_memory_report_marks_the_published_match() -> None:
    report = memory_report()
    assert report.parameter_count_matches_published is True
    assert report.bf16_weight_gib == 27.511
    assert report.kv_gib_at_8192_tokens == 1.5
    assert report.kv_gib_at_32768_tokens == 6.0
    assert report.kv_gib_at_131072_tokens == 24.0


def test_tpot_excludes_the_first_token() -> None:
    e2e = 1.0
    ttft = 0.4
    assert time_per_output_token_seconds(e2e, ttft, 4) == pytest.approx(0.2)
    assert output_tokens_per_second(4, e2e) == pytest.approx(4.0)
    assert decode_tokens_per_second(4, e2e, ttft) == pytest.approx(5.0)


def test_single_output_token_has_no_tpot() -> None:
    assert time_per_output_token_seconds(0.5, 0.5, 1) is None
    assert time_per_output_token_seconds(0.5, None, 4) is None
    assert decode_tokens_per_second(1, 0.5, 0.5) is None


def test_percentile_interpolates_and_empty_input_is_empty() -> None:
    assert percentile([], 95) is None
    assert percentile([10.0], 95) == 10.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    summary = distribution([1.0, 2.0, 3.0, 4.0])
    assert summary.count == 4
    assert summary.p50 == 2.5
    assert summary.minimum == 1.0
    assert summary.maximum == 4.0
