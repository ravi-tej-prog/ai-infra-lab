"""Fixed prompts for the inference client and the benchmark harness.

Token targets are approximate until a response reports `usage.prompt_tokens`.
That API count includes the chat template, so it is the prefill length that
matters. A 4-character estimate is only a way to grow the user text.
"""

from __future__ import annotations

from collections.abc import Callable

TokenCounter = Callable[[str], int]

SENTENCE = (
    "Capacity planning for one inference replica starts from a fixed prompt "
    "and a fixed output length, then raises concurrency one step at a time. "
)

PROFILES: dict[str, int] = {
    "short": 1,
    "medium": 20,
    "long": 80,
}


class CharEstimateCounter:
    """Stand-in counter: about four English characters per token.

    This is not the Qwen tokenizer. Record API usage as the real token count.
    """

    def __call__(self, text: str) -> int:
        return max(1, len(text) // 4)


def prompt_from_profile(profile: str) -> str:
    try:
        repeats = PROFILES[profile]
    except KeyError as exc:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown prompt profile '{profile}'. Known profiles: {known}") from exc
    return SENTENCE * repeats


def prompt_for_target_tokens(target_tokens: int, counter: TokenCounter | None = None) -> str:
    if target_tokens < 1:
        raise ValueError("target_input_tokens must be >= 1")
    count = counter or CharEstimateCounter()
    if count(SENTENCE) >= target_tokens:
        return SENTENCE

    low = 1
    high = 1
    while count(SENTENCE * high) < target_tokens:
        high *= 2
        if high > 100_000:
            raise ValueError("target_input_tokens is too large for the prompt builder")

    best = SENTENCE * high
    while low <= high:
        mid = (low + high) // 2
        candidate = SENTENCE * mid
        if count(candidate) >= target_tokens:
            best = candidate
            high = mid - 1
        else:
            low = mid + 1
    return best
