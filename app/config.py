from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


_SENSITIVE_SETTING_INPUTS = frozenset(
    {
        "auth_secret_key",
        "openai_api_key",
        "embedding_api_key",
        "account_model_keyring_json",
        # Connection URLs can contain userinfo even though production
        # documentation recommends secret indirection.
        "database_url",
        "redis_url",
    }
)


def _redact_sensitive_settings_input(value):
    if isinstance(value, dict):
        return {
            key: (
                None
                if str(key).lower() in _SENSITIVE_SETTING_INPUTS
                else _redact_sensitive_settings_input(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_settings_input(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_settings_input(item) for item in value)
    return value


class Settings(BaseSettings):
    # Settings validation can fail before structured logging and redaction are
    # available (for example, while importing the API or Celery worker).  Keep
    # Pydantic from embedding raw environment values in the exception text:
    # several fields below intentionally carry credentials or key material.
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    def __init__(self, **values):
        """Redact secret inputs even from explicit ``ValidationError.errors``.

        ``hide_input_in_errors`` secures the normal string/repr logging path,
        but Pydantic deliberately retains raw values in ``errors()``. Some
        startup wrappers serialize that structure, so rebuild failures with
        secret fields removed while preserving locations and error types.
        """

        try:
            super().__init__(**values)
        except ValidationError as exc:
            sanitized = []
            for raw in exc.errors():
                item = dict(raw)
                location = item.get("loc") or ()
                first = str(location[0]).lower() if location else ""
                item["input"] = (
                    None
                    if first in _SENSITIVE_SETTING_INPUTS
                    else _redact_sensitive_settings_input(item.get("input"))
                )
                sanitized.append(item)
            raise ValidationError.from_exception_data(
                exc.title,
                sanitized,
                hide_input=True,
            ) from None

    # Optional immutable revision embedded in API/worker runtime provenance.
    # Real evaluation gates require the full commit SHA and fail closed when it
    # is absent; ordinary development remains usable without it.
    loreguard_build_revision: str = Field(
        default="", max_length=64, pattern=r"^(?:|[a-f0-9]{40,64})$"
    )
    database_url: str = "sqlite:///./loreguard.db"
    redis_url: str = "redis://localhost:6379/0"
    use_celery: bool = False
    deployment_environment: Literal["local", "production"] = "local"
    auth_mode: Literal["anonymous", "required"] = "anonymous"
    # Server-only HMAC key for opaque session and CSRF token digests. It is
    # deliberately independent from provider/API credentials.
    auth_secret_key: str = Field(default="", repr=False, max_length=4_096)
    auth_session_ttl_seconds: int = Field(default=14 * 24 * 60 * 60, ge=300, le=90 * 24 * 60 * 60)
    auth_cookie_secure: bool = False
    auth_cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    cors_allowed_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8000,http://127.0.0.1:8000,"
        "http://localhost:8080,http://127.0.0.1:8080"
    )
    openai_api_key: str = Field(default="", repr=False, max_length=4_096)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    # Account-owned chat credentials use an encryption trust root independent
    # from AUTH_SECRET_KEY.  Production requires an explicit keyring; local
    # deployments may generate the dedicated key file on first use.
    account_model_active_key_id: str = Field(default="local-v1", max_length=64)
    account_model_keyring_json: str = Field(default="", repr=False, max_length=32_768)
    account_model_keyring_file: str = Field(default="", max_length=2_048)
    account_model_local_key_path: str = Field(
        default="data/account-model-master.key", max_length=2_048
    )
    # Comma-separated exact HTTPS origins. The normalized OPENAI_BASE_URL
    # origin is also admitted for backwards-compatible local deployments.
    account_model_allowed_origins: str = Field(default="", max_length=16_384)
    # Internal, request-edge snapshot populated when an account credential is
    # resolved.  It prevents a run-local OPENAI_BASE_URL from implicitly
    # adding its own origin to the administrator's allowlist.  This value is
    # deliberately excluded from serialized settings and is not a secret.
    provider_transport_allowed_origins: str = Field(
        default="", max_length=16_384, exclude=True, repr=False
    )
    # Opaque, non-secret worker guard context. Account resolution populates
    # these fields so every actual chat request can re-check revocation after
    # the initial decrypt. They are excluded from settings serialization.
    provider_runtime_config_id: str = Field(
        default="", max_length=128, exclude=True, repr=False
    )
    provider_runtime_user_id: str = Field(
        default="", max_length=128, exclude=True, repr=False
    )
    provider_runtime_revision: int = Field(default=0, ge=0, exclude=True, repr=False)
    provider_runtime_endpoint_sha256: str = Field(
        default="", max_length=64, exclude=True, repr=False
    )
    enable_model_extraction: bool = False
    # Dedicated hard ceilings for explicit document-context suggestions. The
    # bounded call is user-triggered and never confirms narrative authority.
    narrative_context_inference_timeout_seconds: float = Field(
        default=20.0, gt=0, le=30.0
    )
    narrative_context_inference_total_deadline_seconds: float = Field(
        default=25.0, gt=0, le=45.0
    )
    narrative_context_inference_max_completion_tokens: int = Field(
        default=1_200, ge=256, le=2_000
    )
    narrative_context_inference_max_response_bytes: int = Field(
        default=32_000, ge=1_024, le=64_000
    )
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
    # Tightly bound successful upstream response bodies independently of model
    # token controls, which compatible relays may ignore. ``None`` disables
    # only this configurable cap; the provider retains a hard safety ceiling.
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
    # Character consistency is an independent, default-off capability.  Its
    # extraction and review calls share one feature gate but retain dedicated
    # resource ceilings, so enabling it cannot silently borrow the evidence
    # reviewer's budget or change that reviewer's behavior.
    enable_character_consistency: bool = False
    character_consistency_sensitivity: Literal[
        "conservative", "balanced", "exploratory"
    ] = "balanced"
    # One shared admission budget covers every extraction and drift-review
    # call in the optional stage.  The per-call budgets below can only tighten
    # this ceiling; they are not additive entitlements.
    character_consistency_stage_token_budget: int = Field(
        default=60_000, ge=256, le=150_000
    )
    character_consistency_max_chunks_per_run: int = Field(
        default=24, ge=1, le=128
    )
    character_consistency_max_candidates_per_run: int = Field(
        default=64, ge=1, le=256
    )
    character_signal_max_chunk_chars: int = Field(
        default=8_000, ge=256, le=12_000
    )
    character_signal_max_records: int = Field(default=48, ge=1, le=64)
    # Experimental prompt A/B for the primary character signal extractor.
    # Keep disabled until frozen DEV results justify changing the default.
    character_signal_full_line_prompt_v2: bool = False
    # Scope experiment depends on the full-line evidence protocol.
    character_signal_core_scope_prompt_v3: bool = False
    # A draft chunk can receive bounded, one-trait-at-a-time recall calls for
    # undercovered confirmed traits. Keep the selected set no larger than the
    # content-free context boundary; the shared stage budget remains final.
    character_signal_targeted_max_targets_per_chunk: int = Field(
        default=12, ge=1, le=12
    )
    character_signal_timeout_seconds: float = Field(
        default=30.0, gt=0, le=30.0
    )
    character_signal_max_attempts: int = Field(default=2, ge=1, le=4)
    # Logical full-package generations are separate from transport retries.
    # A second call is allowed only after local structure/evidence validation
    # rejects the first complete response.
    character_signal_package_max_attempts: int = Field(default=2, ge=1, le=2)
    character_signal_total_deadline_seconds: float = Field(
        default=60.0, gt=0, le=60.0
    )
    # One logical signal extraction may generate at most two complete packages.
    # This budget spans that whole regeneration cycle; it is independent from
    # the per-response completion cap and remains subordinate to the stage cap.
    character_signal_token_budget: int = Field(
        default=22_000, ge=256, le=40_000
    )
    character_signal_max_completion_tokens: int = Field(
        default=4_096, ge=64, le=8_192
    )
    character_signal_max_response_bytes: int = Field(
        default=64_000, ge=1_024, le=128_000
    )
    character_drift_max_observations: int = Field(default=12, ge=1, le=24)
    character_drift_max_support_evidence: int = Field(default=8, ge=0, le=16)
    character_drift_max_evidence_chars: int = Field(
        default=8_000, ge=256, le=16_000
    )
    character_drift_token_budget: int = Field(
        default=4_000, ge=256, le=8_000
    )
    character_drift_timeout_seconds: float = Field(
        default=30.0, gt=0, le=30.0
    )
    character_drift_max_attempts: int = Field(default=2, ge=1, le=4)
    character_drift_total_deadline_seconds: float = Field(
        default=60.0, gt=0, le=60.0
    )
    character_drift_max_completion_tokens: int = Field(
        default=1_000, ge=64, le=1_500
    )
    character_drift_max_response_bytes: int = Field(
        default=32_000, ge=1_024, le=64_000
    )
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
        default=16_000, ge=1_024, le=20_000
    )
    evidence_investigator_max_prompt_bytes: int = Field(
        default=128 * 1_024, ge=4 * 1_024, le=256 * 1_024
    )
    # Defaults remain production-oriented.  The higher ceilings only permit
    # explicit slow-provider diagnostics; they are not performance targets.
    evidence_investigator_timeout_seconds: float = Field(
        default=30.0, gt=0, le=120.0
    )
    evidence_investigator_total_deadline_seconds: float = Field(
        default=60.0, gt=0, le=600.0
    )
    evidence_investigator_max_completion_tokens: int = Field(
        default=768, ge=64, le=2_048
    )
    evidence_investigator_max_response_bytes: int = Field(
        default=64_000, ge=1_024, le=128_000
    )
    evidence_investigator_top_k: int = Field(default=6, ge=1, le=12)
    evidence_investigator_branch_limit: int = Field(default=30, ge=1, le=50)
    # Per-run embedding input resource quota. This is deliberately measured in
    # characters and is neither a chat-token budget nor an API billing claim.
    evidence_investigator_embedding_max_input_chars: int = Field(
        default=250_000, ge=1, le=1_000_000
    )
    # Embeddings are always required when the Investigator is enabled. False
    # permits a clearly diagnosed lexical fallback only after a transient
    # index/query-vector failure; it is not a no-embedding operating mode.
    evidence_investigator_require_hybrid: bool = True
    # Must leave enough headroom for the 60K optional character stage after
    # deterministic/model extraction has already consumed part of the run.
    per_run_token_budget: int = 100_000
    daily_token_budget: int = 100_000
    model_input_price_per_million: float | None = None
    model_output_price_per_million: float | None = None
    max_upload_bytes: int = 10 * 1024 * 1024
    diff_max_lines_per_version: int = 20_000
    diff_max_chars_per_version: int = 2_000_000
    diff_max_output_lines: int = 4_000
    rate_limit_per_minute: int = 30
    rate_limit_window_seconds: float = 60

    @field_validator("openai_api_key", "embedding_api_key")
    @classmethod
    def validate_authorization_header_secret(cls, value: str) -> str:
        """Accept only visible ASCII that is safe in an HTTP header value.

        httpx includes the rejected header value in the representation of some
        encoding exceptions. Rejecting Unicode, whitespace, and controls while
        settings are loaded keeps credentials out of that failure path.
        """

        if value and any(not 0x21 <= ord(character) <= 0x7E for character in value):
            raise ValueError("provider API key must use header-safe ASCII")
        return value

    @field_validator("provider_thinking_mode", mode="before")
    @classmethod
    def normalize_empty_provider_thinking_mode(cls, value):
        """Compose's unset interpolation is an empty string, meaning no opt-in."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "provider_total_deadline_seconds",
        "provider_max_completion_tokens",
        "semantic_repair_max_completion_tokens",
        mode="before",
    )
    @classmethod
    def normalize_empty_optional_provider_limits(cls, value):
        """Allow Compose to pass an explicit empty optional limit."""

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

        # SEARCH -> READ -> SUBMIT/ABSTAIN is the normal three-tool path.  The
        # loop additionally permits one content-free correction per seed, so
        # a valid configured run must reserve four provider decisions for each
        # selected seed.  Rejected calls do not consume the tool counter, but
        # retaining tool >= decision is an intentionally conservative ceiling.
        minimum_rounds = self.evidence_investigator_max_seeds * 4
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
        if self.enable_evidence_investigator and not self.enable_embeddings:
            raise ValueError(
                "evidence investigator requires the embeddings capability"
            )
        if self.enable_evidence_investigator:
            revision = self.embedding_model_revision.strip()
            deployment = self.embedding_deployment_fingerprint.strip()
            try:
                endpoint = urlsplit(self.embedding_base_url.strip())
                valid_endpoint = (
                    endpoint.scheme in {"http", "https"}
                    and bool(endpoint.hostname)
                    and endpoint.username is None
                    and endpoint.password is None
                    and not endpoint.query
                    and not endpoint.fragment
                    and (
                        endpoint.scheme == "https"
                        or self.embedding_allow_insecure_http
                    )
                )
                # Accessing ``port`` also rejects malformed/out-of-range ports.
                _ = endpoint.port
            except (TypeError, ValueError):
                valid_endpoint = False
            if (
                not self.embedding_model.strip()
                or not revision
                or revision.lower() == "unspecified"
                or not deployment
                or deployment.lower() == "unspecified"
                or self.embedding_dimensions is None
                or not valid_endpoint
            ):
                raise ValueError(
                    "evidence investigator requires a complete embedding profile"
                )
        return self

    @model_validator(mode="after")
    def validate_character_consistency_limits(self):
        """Keep the two model stages independently bounded and admissible."""

        if (
            self.character_signal_core_scope_prompt_v3
            and not self.character_signal_full_line_prompt_v2
        ):
            raise ValueError("character signal core scope v3 requires full line v2")
        if (
            self.character_signal_total_deadline_seconds
            < self.character_signal_timeout_seconds
        ):
            raise ValueError("character signal deadline is shorter than timeout")
        if (
            self.character_drift_total_deadline_seconds
            < self.character_drift_timeout_seconds
        ):
            raise ValueError("character drift deadline is shorter than timeout")
        if (
            self.character_signal_token_budget
            < self.character_signal_max_completion_tokens
        ):
            raise ValueError("character signal budget cannot reserve model output")
        if (
            self.character_drift_token_budget
            < self.character_drift_max_completion_tokens
        ):
            raise ValueError("character drift budget cannot reserve model output")
        if self.character_consistency_stage_token_budget < max(
            self.character_signal_max_completion_tokens,
            self.character_drift_max_completion_tokens,
        ):
            raise ValueError(
                "character consistency stage budget cannot reserve a model output"
            )
        return self

    @model_validator(mode="after")
    def validate_auth_security_boundary(self):
        """Fail closed when authentication is enabled without its trust root."""

        if self.auth_mode == "required" and len(self.auth_secret_key) < 32:
            raise ValueError(
                "AUTH_MODE=required requires AUTH_SECRET_KEY with at least 32 characters"
            )
        if self.auth_cookie_samesite == "none" and not self.auth_cookie_secure:
            raise ValueError("SameSite=None authentication cookies must be Secure")
        if self.deployment_environment == "production":
            if self.auth_mode != "required":
                raise ValueError("production requires AUTH_MODE=required")
            if not self.auth_cookie_secure:
                raise ValueError("production authentication cookies must be Secure")
            try:
                database_scheme = make_url(self.database_url).drivername
            except ArgumentError as exc:
                raise ValueError(
                    "production DATABASE_URL must be a valid PostgreSQL SQLAlchemy URL"
                ) from exc
            if database_scheme != "postgresql+psycopg":
                raise ValueError(
                    "production DATABASE_URL must use postgresql+psycopg"
                )
            if not self.account_model_active_key_id.strip() or not (
                self.account_model_keyring_json.strip()
                or self.account_model_keyring_file.strip()
            ):
                raise ValueError(
                    "production account-model credentials require an explicit "
                    "active key id and keyring"
                )
        # Parse eagerly so a typo cannot silently broaden or break browser
        # credential handling after the process has started.
        self.parsed_cors_origins()
        return self

    def parsed_cors_origins(self) -> list[str]:
        origins = [item.strip() for item in self.cors_allowed_origins.split(",")]
        if not origins or any(not item for item in origins):
            raise ValueError("CORS_ALLOWED_ORIGINS must contain exact origins")
        for origin in origins:
            try:
                value = urlsplit(origin)
                valid = (
                    value.scheme in {"http", "https"}
                    and bool(value.hostname)
                    and value.username is None
                    and value.password is None
                    and value.path in {"", "/"}
                    and not value.query
                    and not value.fragment
                    and origin != "*"
                )
                _ = value.port
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"invalid exact CORS origin: {origin!r}")
            if self.deployment_environment == "production" and value.scheme != "https":
                raise ValueError("production CORS origins must use https")
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()
