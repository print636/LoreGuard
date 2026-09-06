from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./loreguard.db"
    redis_url: str = "redis://localhost:6379/0"
    use_celery: bool = False
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    enable_model_extraction: bool = False
    provider_timeout_seconds: float = 30
    provider_max_attempts: int = 2
    # Generic OpenAI-compatible behavior is capability-neutral by default.
    # Relays that implement this extension may opt in explicitly.
    provider_thinking_mode: Literal["disabled", "enabled"] | None = None
    provider_total_deadline_seconds: float | None = Field(default=None, gt=0)
    provider_max_completion_tokens: int | None = Field(default=None, gt=0)
    # Bound successful upstream response bodies independently of model token
    # controls, which OpenAI-compatible relays may ignore or not support.
    provider_max_response_bytes: int | None = Field(
        default=2 * 1024 * 1024, ge=1
    )
    worker_lease_seconds: float = Field(default=120, gt=0)
    worker_lease_heartbeat_seconds: float = Field(default=20, gt=0)
    model_circuit_breaker_failed_documents: int = 1
    model_chunk_max_chars: int = 6000
    model_chunk_overlap_lines: int = 1
    model_max_chunks_per_document: int = 24
    model_batch_max_chars: int = Field(default=18_000, ge=1)
    model_batch_max_estimated_tokens: int = Field(default=12_000, ge=1)
    semantic_repair_timeout_seconds: float = Field(default=5.0, gt=0)
    semantic_repair_total_deadline_seconds: float = Field(default=8.0, gt=0)
    # Some OpenAI-compatible gateways reject completion-cap capability fields.
    # Keep the repair call capability-neutral unless the deployment explicitly
    # configures a base or repair-specific cap.
    semantic_repair_max_completion_tokens: int | None = Field(default=None, ge=64)
    semantic_repair_max_response_bytes: int = Field(default=64_000, ge=1)
    # The evidence-repair Agent is opt-in while its frozen evaluation suite is
    # being built. These are defense-in-depth ceilings, not tuning knobs that
    # can be raised without bound by a deployment environment.
    enable_review_agent: bool = False
    review_agent_max_decision_rounds: int = Field(default=2, ge=1, le=2)
    review_agent_max_tool_calls: int = Field(default=6, ge=1, le=6)
    review_agent_max_span_chars: int = Field(default=4_000, ge=1, le=8_000)
    review_agent_max_span_reads: int = Field(default=40, ge=1, le=40)
    review_agent_max_read_requests_per_action: int = Field(default=40, ge=1, le=40)
    review_agent_max_read_lines: int = Field(default=12, ge=1, le=20)
    review_agent_context_radius_lines: int = Field(default=20, ge=0, le=50)
    review_agent_token_budget: int = Field(default=8_000, ge=256, le=12_000)
    review_agent_timeout_seconds: float = Field(default=15.0, gt=0, le=30.0)
    review_agent_total_deadline_seconds: float = Field(default=30.0, gt=0, le=60.0)
    # Remain capability-neutral for relays that reject max_tokens.
    review_agent_max_completion_tokens: int | None = Field(default=None, ge=64)
    review_agent_max_response_bytes: int = Field(default=64_000, ge=1, le=128_000)
    per_run_token_budget: int = 20_000
    daily_token_budget: int = 100_000
    model_input_price_per_million: float | None = None
    model_output_price_per_million: float | None = None
    max_upload_bytes: int = 10 * 1024 * 1024
    diff_max_lines_per_version: int = 20_000
    diff_max_chars_per_version: int = 2_000_000
    diff_max_output_lines: int = 4_000
    rate_limit_per_minute: int = 30
    rate_limit_window_seconds: float = 60

    @field_validator("provider_thinking_mode", mode="before")
    @classmethod
    def normalize_empty_provider_thinking_mode(cls, value):
        """Compose's unset interpolation is an empty string, meaning no opt-in."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("review_agent_max_completion_tokens", mode="before")
    @classmethod
    def normalize_empty_review_agent_completion_limit(cls, value):
        """An unset Compose interpolation keeps the relay-neutral None default."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
