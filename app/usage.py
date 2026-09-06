from __future__ import annotations

from .config import Settings


def estimate_request_tokens(system_prompt: str, user_prompt: str) -> int:
    """Conservative local gate: roughly two Unicode chars/token plus output reserve."""
    input_estimate = (len(system_prompt) + len(user_prompt) + 1) // 2
    return max(1, input_estimate) + 256


def estimate_batch_request_tokens(
    system_prompt: str, user_prompt: str, document_count: int
) -> int:
    """Conservative local batch estimate, not a provider-enforced output cap.

    The relay exposes no supported completion-token limit.  Grow the reserve
    with both batch width and input size so admission does not pretend that a
    fixed reserve is a hard safety guarantee.
    """
    input_chars = len(system_prompt) + len(user_prompt)
    input_estimate = (input_chars + 1) // 2
    # Batch records are verbose JSON objects. Reserve roughly 2K tokens per
    # document and at least half of the user payload, so wider or denser input
    # becomes harder to admit even though the relay cannot enforce output size.
    output_reserve = max(1024, document_count * 2048, (len(user_prompt) + 1) // 2)
    return max(1, input_estimate) + output_reserve


def estimate_repair_request_tokens(system_prompt: str, user_prompt: str) -> int:
    """Conservative admission estimate for the one-shot semantic-label repair."""
    input_estimate = (len(system_prompt) + len(user_prompt) + 1) // 2
    return max(1, input_estimate) + 256


def estimate_review_agent_request_tokens(system_prompt: str, user_prompt: str) -> int:
    """Conservative admission estimate for one bounded Agent decision.

    The output reserve is intentionally larger than the label-only repair
    reserve because one decision may request several read or patch actions.
    """
    input_estimate = (len(system_prompt) + len(user_prompt) + 1) // 2
    return max(1, input_estimate) + 768


def configured_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    settings: Settings,
) -> float | None:
    input_price = settings.model_input_price_per_million
    output_price = settings.model_output_price_per_million
    if input_price is None or output_price is None:
        return None
    return round(
        prompt_tokens * input_price / 1_000_000
        + completion_tokens * output_price / 1_000_000,
        8,
    )
