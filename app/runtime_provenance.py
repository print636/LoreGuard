from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .config import Settings
from .embeddings import EmbeddingNotConfiguredError, OpenAICompatibleEmbeddingProvider
from .evidence_chunks import EvidenceChunker
from .provider import safe_thinking_configuration


RUNTIME_PROVENANCE_SCHEMA = "loreguard-runtime-provenance-v1"
_MAX_BUNDLE_FILES = 512
_MAX_BUNDLE_FILE_BYTES = 4 * 1024 * 1024


def safe_runtime_provenance(settings: Settings) -> dict[str, Any]:
    """Return content-free runtime identity shared by API and worker.

    The payload deliberately records only a relay hostname and a hash of the
    normalized endpoint. It never contains a credential, URL query, response,
    source document, or complete base URL.
    """

    endpoint = _safe_endpoint_identity(settings.openai_base_url)
    thinking = safe_thinking_configuration(settings)
    profile_fingerprint = _embedding_profile_fingerprint(settings)
    chunker_fingerprint = _fingerprint(EvidenceChunker().version)
    effective_deadline = min(
        value
        for value in (
            float(settings.evidence_investigator_total_deadline_seconds),
            (
                float(settings.provider_total_deadline_seconds)
                if settings.provider_total_deadline_seconds is not None
                else None
            ),
        )
        if value is not None
    )
    effective_completion = min(
        value
        for value in (
            settings.evidence_investigator_max_completion_tokens,
            settings.provider_max_completion_tokens,
        )
        if value is not None
    )
    effective_response_bytes = min(
        value
        for value in (
            settings.evidence_investigator_max_response_bytes,
            settings.provider_max_response_bytes,
        )
        if value is not None
    )
    model_alias = _safe_label(settings.openai_model, maximum=255)
    revision = settings.loreguard_build_revision.strip() or None

    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA,
        "build": {
            "git_revision": revision,
            "service_artifact_sha256": service_artifact_sha256(),
        },
        "chat_provider": {
            "model_alias": model_alias,
            "relay_hostname": endpoint[0] if endpoint is not None else None,
            "relay_configuration_sha256": (
                endpoint[1] if endpoint is not None else None
            ),
            "temperature": 0,
            "thinking_configured": thinking["configured"],
            "thinking_mode": thinking["mode"],
        },
        "capabilities": {
            "model_extraction": settings.enable_model_extraction,
            "issue_evidence_review": settings.enable_issue_evidence_review,
            "record_repair_agent": settings.enable_review_agent,
            "evidence_investigator": settings.enable_evidence_investigator,
            "embeddings": settings.enable_embeddings,
        },
        "investigator_limits": {
            "max_seeds": settings.evidence_investigator_max_seeds,
            "max_decision_rounds": (
                settings.evidence_investigator_max_decision_rounds
            ),
            "max_tool_calls": settings.evidence_investigator_max_tool_calls,
            "max_searches": settings.evidence_investigator_max_searches,
            "max_reads": settings.evidence_investigator_max_reads,
            "max_results": settings.evidence_investigator_max_results,
            "max_read_lines": settings.evidence_investigator_max_read_lines,
            "max_span_chars": settings.evidence_investigator_max_span_chars,
            "token_budget": settings.evidence_investigator_token_budget,
            "max_agent_input_bytes": settings.evidence_investigator_max_prompt_bytes,
            "provider_call_timeout_seconds": min(
                float(settings.provider_timeout_seconds),
                float(settings.evidence_investigator_timeout_seconds),
                effective_deadline,
            ),
            "total_deadline_seconds": effective_deadline,
            "max_completion_tokens": effective_completion,
            "max_response_bytes": effective_response_bytes,
            "provider_attempts_per_decision": 1,
            "top_k": settings.evidence_investigator_top_k,
            "branch_limit": settings.evidence_investigator_branch_limit,
            "embedding_max_input_chars": (
                settings.evidence_investigator_embedding_max_input_chars
            ),
            "require_hybrid": settings.evidence_investigator_require_hybrid,
            "daily_token_budget": settings.daily_token_budget,
        },
        "rag": {
            "strategy": "keyword+vector+entity-rrf",
            "profile_fingerprint": profile_fingerprint,
            "chunker_fingerprint": chunker_fingerprint,
            "require_hybrid": settings.evidence_investigator_require_hybrid,
            "top_k": settings.evidence_investigator_top_k,
            "branch_limit": settings.evidence_investigator_branch_limit,
        },
    }


@lru_cache(maxsize=1)
def service_artifact_sha256() -> str | None:
    """Fingerprint executable service sources copied into the image.

    This is not presented as an OCI registry digest. It is a deterministic
    proof that the API and worker loaded the same checked-in Python bundle and
    dependency lock input, even when the evaluator cannot access Docker.
    """

    try:
        root = Path(__file__).resolve().parents[1]
        app_root = root / "app"
        files = sorted(
            (path for path in app_root.rglob("*.py") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        requirements = root / "requirements.txt"
        if requirements.is_file():
            files.append(requirements)
        if not 1 <= len(files) <= _MAX_BUNDLE_FILES:
            return None
        digest = hashlib.sha256()
        for path in files:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            size = path.stat().st_size
            if size < 0 or size > _MAX_BUNDLE_FILE_BYTES:
                return None
            payload = path.read_bytes()
            if len(payload) != size:
                return None
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def _embedding_profile_fingerprint(settings: Settings) -> str | None:
    try:
        profile = OpenAICompatibleEmbeddingProvider(settings).profile
    except (EmbeddingNotConfiguredError, TypeError, ValueError):
        return None
    return _fingerprint(profile.profile_id)


def _safe_endpoint_identity(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str) or value != value.strip() or len(value) > 2_048:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if (
            parsed.scheme not in {"http", "https"}
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            return None
        ascii_hostname = hostname.encode("idna").decode("ascii").lower()
        if not 1 <= len(ascii_hostname) <= 253:
            return None
        port = parsed.port
        netloc = ascii_hostname if port is None else f"{ascii_hostname}:{port}"
        path = parsed.path.rstrip("/")
        normalized = urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))
    except (UnicodeError, ValueError):
        return None
    return ascii_hostname, _fingerprint(normalized)


def _safe_label(value: object, *, maximum: int) -> str | None:
    if not isinstance(value, str) or value != value.strip() or not value:
        return None
    if len(value) > maximum or any(ord(character) < 32 for character in value):
        return None
    return value


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
