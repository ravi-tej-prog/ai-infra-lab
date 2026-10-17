"""Arithmetic memory model for Qwen2.5-14B-Instruct.

The shapes come from the checkpoint `config.json`. The parameter total is
recomputed from those shapes and checked against the safetensors header
published on the Hugging Face model API (14,770,033,664 BF16 parameters).

Nothing in this module is a measured GPU residency. Runtime memory also
includes activations, CUDA graphs, allocator fragmentation, and the KV pool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Hugging Face config.json for Qwen/Qwen2.5-14B-Instruct.
HIDDEN_SIZE = 5120
NUM_LAYERS = 48
NUM_ATTENTION_HEADS = 40
NUM_KEY_VALUE_HEADS = 8
INTERMEDIATE_SIZE = 13824
VOCAB_SIZE = 152064
MAX_POSITION_EMBEDDINGS = 32768
YARN_FULL_CONTEXT_TOKENS = 131072
TIE_WORD_EMBEDDINGS = False

# Hugging Face API `usedStorage` values fetched 2026-10-05.
# These are repository file bytes, including safetensors framing.
BF16_FILE_BYTES = 29_540_134_000
AWQ_FILE_BYTES = 9_980_170_584
GPTQ_INT4_FILE_BYTES = 9_988_759_504

PUBLISHED_BF16_PARAMETER_COUNT = 14_770_033_664


def head_dim() -> int:
    if HIDDEN_SIZE % NUM_ATTENTION_HEADS != 0:
        raise ValueError("hidden size must divide evenly into attention heads")
    return HIDDEN_SIZE // NUM_ATTENTION_HEADS


def parameter_count() -> int:
    """Count weights from the architecture, including QKV bias and untied logits."""

    head = head_dim()
    q_out = NUM_ATTENTION_HEADS * head
    kv_out = NUM_KEY_VALUE_HEADS * head
    attention = (
        HIDDEN_SIZE * q_out
        + HIDDEN_SIZE * kv_out
        + HIDDEN_SIZE * kv_out
        + q_out * HIDDEN_SIZE
        + q_out
        + kv_out
        + kv_out
    )
    mlp = 3 * HIDDEN_SIZE * INTERMEDIATE_SIZE
    norms_per_layer = 2 * HIDDEN_SIZE
    per_layer = attention + mlp + norms_per_layer
    final_norm = HIDDEN_SIZE
    embedding_matrices = 1 if TIE_WORD_EMBEDDINGS else 2
    embeddings = embedding_matrices * VOCAB_SIZE * HIDDEN_SIZE
    return per_layer * NUM_LAYERS + final_norm + embeddings


def weight_bytes(bytes_per_parameter: int) -> int:
    return parameter_count() * bytes_per_parameter


def kv_bytes_per_token(bytes_per_element: int = 2) -> int:
    """BF16/FP16 KV: 2 (K and V) * layers * kv_heads * head_dim * element size.

    Grouped-query attention stores KV for `NUM_KEY_VALUE_HEADS`, not for every
    query head. That is why the cache is smaller than a 40-head multi-head model.
    """

    return 2 * NUM_LAYERS * NUM_KEY_VALUE_HEADS * head_dim() * bytes_per_element


def kv_bytes_for_tokens(num_tokens: int, bytes_per_element: int = 2) -> int:
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0")
    return num_tokens * kv_bytes_per_token(bytes_per_element)


def tensor_parallel_candidates() -> list[int]:
    """TP sizes that divide both the query heads and the KV heads.

    Confirm on the GPU host. Some vLLM versions reject other sizes, and this
    phase does not run tensor-parallel serving.
    """

    candidates = []
    for size in range(1, NUM_KEY_VALUE_HEADS + 1):
        if NUM_ATTENTION_HEADS % size == 0 and NUM_KEY_VALUE_HEADS % size == 0:
            candidates.append(size)
    return candidates


def gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def decimal_gb(num_bytes: int) -> float:
    return num_bytes / 1_000_000_000


@dataclass(frozen=True)
class MemoryReport:
    parameter_count: int
    published_parameter_count: int
    parameter_count_matches_published: bool
    bf16_weight_bytes: int
    bf16_weight_gib: float
    bf16_file_bytes: int
    bf16_file_gib: float
    awq_file_bytes: int
    awq_file_gib: float
    gptq_int4_file_bytes: int
    gptq_int4_file_gib: float
    kv_bytes_per_token_bf16: int
    kv_gib_at_8192_tokens: float
    kv_gib_at_32768_tokens: float
    kv_gib_at_131072_tokens: float
    tensor_parallel_candidates: list[int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def memory_report() -> MemoryReport:
    params = parameter_count()
    bf16 = weight_bytes(2)
    return MemoryReport(
        parameter_count=params,
        published_parameter_count=PUBLISHED_BF16_PARAMETER_COUNT,
        parameter_count_matches_published=params == PUBLISHED_BF16_PARAMETER_COUNT,
        bf16_weight_bytes=bf16,
        bf16_weight_gib=round(gib(bf16), 3),
        bf16_file_bytes=BF16_FILE_BYTES,
        bf16_file_gib=round(gib(BF16_FILE_BYTES), 3),
        awq_file_bytes=AWQ_FILE_BYTES,
        awq_file_gib=round(gib(AWQ_FILE_BYTES), 3),
        gptq_int4_file_bytes=GPTQ_INT4_FILE_BYTES,
        gptq_int4_file_gib=round(gib(GPTQ_INT4_FILE_BYTES), 3),
        kv_bytes_per_token_bf16=kv_bytes_per_token(2),
        kv_gib_at_8192_tokens=round(gib(kv_bytes_for_tokens(8192)), 3),
        kv_gib_at_32768_tokens=round(gib(kv_bytes_for_tokens(MAX_POSITION_EMBEDDINGS)), 3),
        kv_gib_at_131072_tokens=round(gib(kv_bytes_for_tokens(YARN_FULL_CONTEXT_TOKENS)), 3),
        tensor_parallel_candidates=tensor_parallel_candidates(),
    )
