from __future__ import annotations

from .config import Settings


def estimate_request_tokens(system_prompt: str, user_prompt: str) -> int:
    """Conservative local gate: roughly two Unicode chars/token plus output reserve."""
    input_estimate = (len(system_prompt) + len(user_prompt) + 1) // 2
    return max(1, input_estimate) + 256


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
