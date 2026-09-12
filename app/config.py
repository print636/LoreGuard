from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
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
    # Evidence embeddings are an independent, opt-in capability.  Do not fall
    # back to the chat/extraction credential or model: deployments commonly
    # route the two APIs to different providers and trust boundaries.
    enable_embeddings: bool = False
    embedding_api_key: str = Field(default="", max_length=4_096)
    embedding_base_url: str = Field(default="", max_length=2_048)
    embedding_model: str = Field(default="", max_length=255)
    embedding_model_revision: str = Field(default="unspecified", max_length=120)
    embedding_deployment_fingerprint: str = Field(
        default="unspecified", max_length=160
    )
    embedding_profile_namespace: str = Field(default="default", max_length=80)
    embedding_dimensions: int | None = Field(default=None, ge=1, le=16_000)
    embedding_allow_insecure_http: bool = False
    embedding_timeout_seconds: float = Field(default=10.0, gt=0, le=30.0)
    embedding_max_attempts: int = Field(default=2, ge=1, le=4)
    embedding_total_deadline_seconds: float = Field(default=20.0, gt=0, le=60.0)
    embedding_max_response_bytes: int = Field(
        default=4 * 1024 * 1024, ge=1, le=16 * 1024 * 1024
    )
    embedding_batch_max_items: int = Field(default=64, ge=1, le=256)
    embedding_batch_max_chars: int = Field(default=120_000, ge=1, le=500_000)
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
    # Optional post-rule evidence review.  This is deliberately separate from
    # the extraction repair Agent: it may annotate a finding, but it can never
    # rewrite or remove the deterministic finding itself.
    enable_issue_evidence_review: bool = False
    issue_evidence_review_max_issues: int = Field(default=8, ge=1, le=8)
    issue_evidence_review_top_k: int = Field(default=6, ge=1, le=6)
    issue_evidence_review_batch_size: int = Field(default=4, ge=1, le=4)
    issue_evidence_review_max_evidence_chars: int = Field(
        default=6_000, ge=256, le=6_000
    )
    issue_evidence_review_token_budget: int = Field(
        default=6_000, ge=256, le=6_000
    )
    issue_evidence_review_timeout_seconds: float = Field(
        default=20.0, gt=0, le=20.0
    )
    issue_evidence_review_total_deadline_seconds: float = Field(
        default=45.0, gt=0, le=45.0
    )
    issue_evidence_review_max_completion_tokens: int = Field(
        default=1_400, ge=64, le=1_400
    )
    issue_evidence_review_max_response_bytes: int = Field(
        default=64_000, ge=1, le=64_000
    )
    issue_evidence_review_require_hybrid: bool = True
    # Evidence Investigator is an independent, default-off capability.  These
    # values are server-owned safety ceilings, not model-selected tuning knobs.
    # The token budget is an internal admission/quota budget; it is not a hard
    # upper bound on provider billing when a relay under-reports or over-runs.
    enable_evidence_investigator: bool = False
    evidence_investigator_max_seeds: int = Field(default=1, ge=1, le=8)
    evidence_investigator_max_decision_rounds: int = Field(
        default=6, ge=3, le=32
    )
    evidence_investigator_max_tool_calls: int = Field(default=6, ge=3, le=32)
    evidence_investigator_max_searches: int = Field(default=2, ge=1, le=8)
    evidence_investigator_max_reads: int = Field(default=2, ge=1, le=16)
    evidence_investigator_max_results: int = Field(default=12, ge=1, le=48)
    evidence_investigator_max_read_lines: int = Field(default=12, ge=1, le=20)
    evidence_investigator_max_span_chars: int = Field(
        default=12_000, ge=256, le=24_000
    )
    evidence_investigator_token_budget: int = Field(
        default=8_000, ge=1_024, le=20_000
    )
    evidence_investigator_max_prompt_bytes: int = Field(
        default=128 * 1_024, ge=4 * 1_024, le=256 * 1_024
    )
    evidence_investigator_timeout_seconds: float = Field(
        default=15.0, gt=0, le=30.0
    )
    evidence_investigator_total_deadline_seconds: float = Field(
        default=45.0, gt=0, le=60.0
    )
    evidence_investigator_max_completion_tokens: int = Field(
        default=768, ge=64, le=2_048
    )
    evidence_investigator_max_response_bytes: int = Field(
        default=64_000, ge=1_024, le=128_000
    )
    evidence_investigator_top_k: int = Field(default=6, ge=1, le=12)
    evidence_investigator_branch_limit: int = Field(default=30, ge=1, le=50)
    evidence_investigator_require_hybrid: bool = True
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

    @field_validator("embedding_dimensions", mode="before")
    @classmethod
    def normalize_empty_embedding_dimensions(cls, value):
        """An unset dimension keeps the embedding capability unconfigured."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_evidence_investigator_limits(self):
        """Reject internally inconsistent Investigator safety ceilings."""

        minimum_rounds = self.evidence_investigator_max_seeds * 3
        if (
            self.evidence_investigator_max_decision_rounds < minimum_rounds
            or self.evidence_investigator_max_tool_calls < minimum_rounds
            or self.evidence_investigator_max_tool_calls
            < self.evidence_investigator_max_decision_rounds
        ):
            raise ValueError(
                "evidence investigator decision/tool limits cannot cover seeds"
            )
        if (
            self.evidence_investigator_max_searches
            < self.evidence_investigator_max_seeds
            or self.evidence_investigator_max_reads
            < self.evidence_investigator_max_seeds
        ):
            raise ValueError(
                "evidence investigator search/read limits cannot cover seeds"
            )
        minimum_results = (
            self.evidence_investigator_max_seeds
            * self.evidence_investigator_top_k
        )
        if (
            self.evidence_investigator_max_results < minimum_results
            or self.evidence_investigator_branch_limit
            < self.evidence_investigator_top_k
        ):
            raise ValueError("evidence investigator retrieval limits are inconsistent")
        if (
            self.evidence_investigator_total_deadline_seconds
            < self.evidence_investigator_timeout_seconds
        ):
            raise ValueError("evidence investigator deadline is shorter than timeout")
        minimum_completion_reservation = (
            minimum_rounds * self.evidence_investigator_max_completion_tokens
        )
        if self.evidence_investigator_token_budget < minimum_completion_reservation:
            raise ValueError(
                "evidence investigator token budget cannot reserve minimum outputs"
            )
        if (
            self.enable_evidence_investigator
            and self.evidence_investigator_require_hybrid
            and not self.enable_embeddings
        ):
            raise ValueError(
                "hybrid evidence investigator requires the embeddings capability"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
