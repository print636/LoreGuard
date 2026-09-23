from __future__ import annotations

import hmac
import json
import unicodedata
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from .auth import AuthContext, get_auth_context, require_csrf
from .config import Settings, get_settings
from .db import (
    AccountProviderBindingRow,
    AccountProviderConfigRow,
    AnalysisRunRow,
    SessionLocal,
    new_id,
)
from .provider import (
    OpenAICompatibleProvider,
    ProviderError,
    RetryPolicy,
)
from .provider_credentials import (
    CredentialAAD,
    EncryptedCredential,
    ProviderCredentialError,
    ProviderEndpointRejected,
    allowed_provider_origins,
    load_provider_keyring,
    normalize_provider_base_url,
    provider_endpoint_sha256,
    safe_provider_identity,
    validate_provider_request_url,
)
from .rate_limit import SlidingWindowLimiter
from .time_utils import utc_now_naive


NO_STORE_HEADERS = {"Cache-Control": "no-store"}
FIXED_KEY_MASK = "••••••••••••"
MAX_PROVIDER_BODY_BYTES = 16 * 1024
_provider_test_limiter = SlidingWindowLimiter(limit=5, window_seconds=60)


class AccountProviderWriteIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expected_revision: int = Field(ge=0)
    base_url: str = Field(min_length=1, max_length=2_048)
    model: str = Field(min_length=1, max_length=255)
    api_key: str | None = Field(default=None, min_length=1, max_length=4_096, repr=False)

    @field_validator("base_url", "model")
    @classmethod
    def reject_ambiguous_text(cls, value: str) -> str:
        if value != value.strip() or any(
            unicodedata.category(character).startswith("C") for character in value
        ):
            raise ValueError("invalid text")
        return value

    @field_validator("api_key")
    @classmethod
    def reject_ambiguous_secret(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(not 0x21 <= ord(character) <= 0x7E for character in value):
            raise ValueError("invalid secret")
        return value


class ExpectedRevisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)


def _sanitize_provider_identity(value: object) -> dict[str, Any]:
    source = "deterministic_only"
    configured = False
    result: dict[str, Any] = {"schema_version": 1}
    if isinstance(value, dict):
        candidate_source = value.get("source")
        if candidate_source in {
            "account_byok",
            "service_default",
            "service_default_legacy",
            "deterministic_only",
        }:
            source = candidate_source
        configured = value.get("configured") is True
        revision = value.get("config_revision")
        if type(revision) is int and revision > 0:
            result["config_revision"] = revision
        config_id = value.get("config_id")
        if isinstance(config_id, str) and 1 <= len(config_id) <= 128:
            result["config_id"] = config_id
        model = value.get("model_alias")
        if isinstance(model, str) and 1 <= len(model) <= 255:
            result["model_alias"] = model
        endpoint_hash = value.get("endpoint_configuration_sha256")
        if (
            isinstance(endpoint_hash, str)
            and len(endpoint_hash) == 64
            and all(character in "0123456789abcdef" for character in endpoint_hash)
        ):
            result["endpoint_configuration_sha256"] = endpoint_hash
    result["source"] = source
    result["configured"] = configured
    return result


class ProviderRuntime:
    """Run-local settings plus the frozen non-secret provider identity."""

    def __init__(
        self,
        settings: Settings,
        identity: dict[str, Any],
        *,
        config_id: str | None,
        available: bool,
        unavailable_reason: str | None = None,
    ) -> None:
        self.settings = settings
        self.identity = _sanitize_provider_identity(identity)
        self.config_id = config_id
        self.available = available
        self.unavailable_reason = unavailable_reason


def _safe_http(status_code: int, code: str, message: str, **detail: Any) -> HTTPException:
    return HTTPException(
        status_code,
        detail={"code": code, "message": message, **detail},
        headers=NO_STORE_HEADERS,
    )


async def _safe_payload(request: Request, model_type):
    """Parse a secret-bearing body without FastAPI echoing invalid input."""

    body = await request.body()
    if not body or len(body) > MAX_PROVIDER_BODY_BYTES:
        raise _safe_http(422, "invalid_provider_payload", "模型配置请求格式无效")
    try:
        raw = json.loads(body)
        if not isinstance(raw, dict):
            raise ValueError
        return model_type.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError, TypeError):
        raise _safe_http(422, "invalid_provider_payload", "模型配置请求格式无效") from None


def _clean_model_name(value: str) -> str:
    if not value or value != value.strip() or len(value) > 255:
        raise _safe_http(422, "invalid_provider_model", "模型名称格式无效")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise _safe_http(422, "invalid_provider_model", "模型名称格式无效")
    return value


def _aad(row: AccountProviderConfigRow) -> CredentialAAD:
    return CredentialAAD(
        user_id=row.user_id,
        config_id=row.id,
        revision=row.revision,
        endpoint_sha256=row.endpoint_sha256,
        model_name=row.model_name,
    )


def _envelope(row: AccountProviderConfigRow) -> EncryptedCredential:
    if (
        row.secret_ciphertext is None
        or row.secret_nonce is None
        or row.encryption_key_id is None
    ):
        raise ProviderCredentialError("provider credential is unavailable")
    return EncryptedCredential(
        ciphertext=bytes(row.secret_ciphertext),
        nonce=bytes(row.secret_nonce),
        key_id=row.encryption_key_id,
        schema_version=row.secret_schema_version,
    )


def _decrypt_row(row: AccountProviderConfigRow, settings: Settings) -> str:
    if row.state not in {"usable", "superseded"}:
        raise ProviderCredentialError("provider credential is revoked")
    # The authenticated AAD contains the persisted endpoint fingerprint.  It
    # must also be tied back to the current canonical URL before decrypting;
    # otherwise an attacker with database write access could change only
    # ``base_url`` to a second allowlisted host while retaining a valid
    # ciphertext/AAD pair.
    try:
        normalized = normalize_provider_base_url(
            row.base_url,
            allowed_origins=allowed_provider_origins(settings),
        )
        current_hash = provider_endpoint_sha256(normalized)
    except ProviderEndpointRejected as exc:
        raise ProviderCredentialError("provider endpoint is unavailable") from exc
    if not hmac.compare_digest(current_hash, row.endpoint_sha256):
        raise ProviderCredentialError("provider endpoint is unavailable")
    secret = load_provider_keyring(settings).decrypt(_envelope(row), _aad(row))
    # Recheck legacy/imported ciphertext at the trust boundary as well as on
    # new writes. This prevents a malformed decrypted value from reaching an
    # HTTP library exception that could include the Authorization header.
    if not secret or any(not 0x21 <= ord(character) <= 0x7E for character in secret):
        raise ProviderCredentialError("provider credential is unavailable")
    return secret


def _identity_for_row(row: AccountProviderConfigRow) -> dict[str, Any]:
    return {
        **safe_provider_identity(
            row.id,
            row.revision,
            row.model_name,
            row.endpoint_sha256,
        ),
        "source": "account_byok",
        "configured": True,
    }


def _run_local_settings(
    settings: Settings,
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    config_id: str = "",
    user_id: str = "",
    revision: int = 0,
    endpoint_sha256: str = "",
) -> Settings:
    """Freeze the deployment allowlist before replacing OPENAI_BASE_URL.

    ``allowed_provider_origins`` intentionally admits the service's configured
    OPENAI_BASE_URL for backwards compatibility.  Calling it after replacing
    that URL with an account value would make the request-edge check
    self-authorizing.  This snapshot is therefore computed from the untouched
    deployment settings and inherited by every provider fork in the run.
    """

    origins = allowed_provider_origins(settings)
    return settings.model_copy(
        update={
            "openai_api_key": api_key,
            "openai_base_url": base_url,
            "openai_model": model_name,
            "provider_transport_allowed_origins": ",".join(origins),
            "provider_runtime_config_id": config_id,
            "provider_runtime_user_id": user_id,
            "provider_runtime_revision": revision,
            "provider_runtime_endpoint_sha256": endpoint_sha256,
        }
    )


def _current_config(db, user_id: str, *, lock: bool = False):
    statement = select(AccountProviderBindingRow).where(
        AccountProviderBindingRow.user_id == user_id
    )
    if lock:
        statement = statement.with_for_update()
    binding = db.scalar(statement)
    row = (
        db.get(AccountProviderConfigRow, binding.current_config_id)
        if binding is not None and binding.current_config_id is not None
        else None
    )
    if row is not None and (row.user_id != user_id or row.state != "usable"):
        raise ProviderCredentialError("current provider binding is invalid")
    return binding, row


def account_provider_runtime(
    db,
    *,
    user_id: str,
    base_settings: Settings | None = None,
) -> ProviderRuntime:
    """Resolve the current account config without ever borrowing another user."""

    settings = base_settings or get_settings()
    binding, row = _current_config(db, user_id)
    if row is not None:
        identity = _identity_for_row(row)
        try:
            secret = _decrypt_row(row, settings)
            validate_provider_request_url(
                row.base_url,
                allowed_origins=allowed_provider_origins(settings),
            )
        except (ProviderCredentialError, ProviderEndpointRejected):
            return ProviderRuntime(
                settings.model_copy(update={"openai_api_key": ""}),
                identity,
                config_id=row.id,
                available=False,
                unavailable_reason="credential_unavailable",
            )
        local = _run_local_settings(
            settings,
            api_key=secret,
            base_url=row.base_url,
            model_name=row.model_name,
            config_id=row.id,
            user_id=row.user_id,
            revision=row.revision,
            endpoint_sha256=row.endpoint_sha256,
        )
        return ProviderRuntime(
            local,
            identity,
            config_id=row.id,
            available=True,
        )

    if settings.auth_mode == "anonymous" and settings.openai_api_key.strip():
        try:
            normalized = normalize_provider_base_url(
                settings.openai_base_url,
                allowed_origins=allowed_provider_origins(settings),
            )
            endpoint_hash = provider_endpoint_sha256(normalized)
        except ProviderEndpointRejected:
            return ProviderRuntime(
                settings.model_copy(update={"openai_api_key": ""}),
                {
                    "schema_version": 1,
                    "source": "service_default",
                    "configured": False,
                    "reason": "endpoint_rejected",
                },
                config_id=None,
                available=False,
                unavailable_reason="endpoint_rejected",
            )
        return ProviderRuntime(
            _run_local_settings(
                settings,
                api_key=settings.openai_api_key,
                base_url=normalized,
                model_name=settings.openai_model,
            ),
            {
                "schema_version": 1,
                "source": "service_default",
                "configured": True,
                "model_alias": settings.openai_model,
                "endpoint_configuration_sha256": endpoint_hash,
            },
            config_id=None,
            available=True,
        )

    return ProviderRuntime(
        settings.model_copy(update={"openai_api_key": ""}),
        {
            "schema_version": 1,
            "source": "deterministic_only",
            "configured": False,
        },
        config_id=None,
        available=False,
        unavailable_reason="not_configured",
    )


def bind_account_provider_to_run(
    db,
    run: AnalysisRunRow,
    *,
    user_id: str,
    base_settings: Settings | None = None,
) -> None:
    """Freeze only an opaque config reference and a non-secret identity."""

    runtime = account_provider_runtime(
        db, user_id=user_id, base_settings=base_settings
    )
    run.provider_config_id = runtime.config_id
    run.provider_identity = runtime.identity


def runtime_for_analysis_run(
    db,
    run: AnalysisRunRow,
    *,
    base_settings: Settings | None = None,
) -> ProviderRuntime:
    """Resolve exactly the provider revision frozen on a run."""

    settings = base_settings or get_settings()
    identity = run.provider_identity if isinstance(run.provider_identity, dict) else None
    source = identity.get("source") if identity else None
    if run.provider_config_id is not None:
        row = db.get(AccountProviderConfigRow, run.provider_config_id)
        if (
            row is None
            or run.requested_by_user_id is None
            or row.user_id != run.requested_by_user_id
            or identity != _identity_for_row(row)
        ):
            return ProviderRuntime(
                settings.model_copy(update={"openai_api_key": ""}),
                identity or {"source": "account_byok", "configured": False},
                config_id=run.provider_config_id,
                available=False,
                unavailable_reason="identity_mismatch",
            )
        try:
            secret = _decrypt_row(row, settings)
            validate_provider_request_url(
                row.base_url,
                allowed_origins=allowed_provider_origins(settings),
            )
        except (ProviderCredentialError, ProviderEndpointRejected):
            return ProviderRuntime(
                settings.model_copy(update={"openai_api_key": ""}),
                identity,
                config_id=row.id,
                available=False,
                unavailable_reason="credential_revoked",
            )
        return ProviderRuntime(
            _run_local_settings(
                settings,
                api_key=secret,
                base_url=row.base_url,
                model_name=row.model_name,
                config_id=row.id,
                user_id=row.user_id,
                revision=row.revision,
                endpoint_sha256=row.endpoint_sha256,
            ),
            identity,
            config_id=row.id,
            available=True,
        )

    if source == "service_default" and settings.auth_mode == "anonymous":
        try:
            normalized = normalize_provider_base_url(
                settings.openai_base_url,
                allowed_origins=allowed_provider_origins(settings),
            )
            current_hash = provider_endpoint_sha256(normalized)
        except ProviderEndpointRejected:
            current_hash = None
            normalized = settings.openai_base_url
        if (
            not settings.openai_api_key.strip()
            or not identity
            or identity.get("configured") is not True
            or identity.get("model_alias") != settings.openai_model
            or identity.get("endpoint_configuration_sha256") != current_hash
        ):
            return ProviderRuntime(
                settings.model_copy(update={"openai_api_key": ""}),
                identity or {"source": "service_default", "configured": False},
                config_id=None,
                available=False,
                unavailable_reason="identity_mismatch",
            )
        return ProviderRuntime(
            _run_local_settings(
                settings,
                api_key=settings.openai_api_key,
                base_url=normalized,
                model_name=settings.openai_model,
            ),
            identity,
            config_id=None,
            available=True,
        )
    if source in {"deterministic_only", "account_byok"} or settings.auth_mode == "required":
        return ProviderRuntime(
            settings.model_copy(update={"openai_api_key": ""}),
            identity or {"source": "deterministic_only", "configured": False},
            config_id=None,
            available=False,
            unavailable_reason="not_configured",
        )
    # Legacy anonymous runs predate provider snapshots and retain local-server
    # compatibility. Required-auth runs never enter this fallback.
    return ProviderRuntime(
        settings,
        {
            "schema_version": 1,
            "source": "service_default_legacy",
            "configured": bool(settings.openai_api_key.strip()),
        },
        config_id=None,
        available=bool(settings.openai_api_key.strip()),
    )


def _serialize(binding, row) -> dict[str, Any]:
    revision = binding.lock_version if binding is not None else 0
    current_settings = get_settings()
    service_default_available = (
        current_settings.auth_mode == "anonymous"
        and bool(current_settings.openai_api_key.strip())
    )
    if row is None:
        return {
            "configured": False,
            "revision": revision,
            "api_key": {"state": "absent", "masked": None},
            "base_url": None,
            "model": None,
            "updated_at": binding.updated_at if binding is not None else None,
            "last_test": None,
            "service_default_available": service_default_available,
        }
    last_test_payload = _safe_last_test_payload(row.last_test_payload)
    return {
        "configured": True,
        "revision": revision,
        "api_key": {"state": "set", "masked": FIXED_KEY_MASK},
        "base_url": row.base_url,
        "model": row.model_name,
        "updated_at": binding.updated_at,
        "last_test": (
            {
                "profile_revision": row.revision,
                "status": (
                    row.last_test_status
                    if row.last_test_status in {"passed", "failed"}
                    else "failed"
                ),
                "category": (
                    row.last_test_category
                    if row.last_test_category
                    in {
                        "success",
                        "invalid_response",
                        "unauthorized",
                        "forbidden",
                        "rate_limit",
                        "upstream_5xx",
                        "connect_timeout",
                        "read_timeout",
                        "transport",
                        "response_too_large",
                        "unsupported_content_encoding",
                        "response_decompression",
                        "nonretry_http",
                        "body_json",
                        "response_shape",
                        "empty_content",
                        "usage_shape",
                        "truncated",
                        "content_json",
                        "endpoint_rejected",
                        "credential_invalid",
                        "credential_revoked",
                        "credential_reflected",
                        "provider",
                    }
                    else "provider"
                ),
                "tested_at": row.last_tested_at,
                **last_test_payload,
            }
            if row.last_tested_at is not None
            else None
        ),
        "service_default_available": service_default_available,
    }


def _bounded_counter(value: object, *, maximum: int = 1_000_000_000) -> int | None:
    if type(value) is int and 0 <= value <= maximum:
        return value
    return None


def _safe_last_test_payload(value: object) -> dict[str, Any]:
    """Allowlist persisted probe metadata before returning it to a client.

    This remains content-free even if a future writer or direct database edit
    adds upstream text, request headers, or credential-looking unknown keys.
    """

    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for name in ("json_contract_ok", "reachable", "authorized"):
        candidate = value.get(name)
        if candidate is None or type(candidate) is bool:
            result[name] = candidate
    latency = _bounded_counter(value.get("latency_ms"), maximum=3_600_000)
    if latency is not None:
        result["latency_ms"] = latency
    usage = value.get("token_usage")
    if isinstance(usage, dict):
        prompt = _bounded_counter(usage.get("prompt"))
        completion = _bounded_counter(usage.get("completion"))
        if prompt is not None and completion is not None:
            result["token_usage"] = {
                "prompt": prompt,
                "completion": completion,
                # Recompute rather than trust a potentially polluted total.
                "total": prompt + completion,
            }
    return result


def _test_result(provider: OpenAICompatibleProvider) -> dict[str, Any]:
    try:
        result = provider.complete("Return only valid JSON.", 'Return {"status":"ok"}.')
    except ProviderError as exc:
        telemetry = exc.telemetry
        return {
            "status": "failed",
            "category": exc.category,
            "json_contract_ok": None,
            "reachable": exc.http_status is not None,
            "authorized": False if exc.category in {"unauthorized", "forbidden"} else None,
            "latency_ms": getattr(telemetry, "elapsed_ms", exc.elapsed_ms),
            "token_usage": {
                "prompt": getattr(telemetry, "prompt_tokens", 0),
                "completion": getattr(telemetry, "completion_tokens", 0),
                "total": getattr(telemetry, "prompt_tokens", 0)
                + getattr(telemetry, "completion_tokens", 0),
            },
        }
    try:
        contract_ok = json.loads(result.text) == {"status": "ok"}
    except (json.JSONDecodeError, TypeError):
        contract_ok = False
    telemetry = result.telemetry
    return {
        "status": "passed" if contract_ok else "failed",
        "category": "success" if contract_ok else "invalid_response",
        "json_contract_ok": contract_ok,
        "reachable": True,
        "authorized": True,
        "latency_ms": getattr(telemetry, "elapsed_ms", None),
        "token_usage": {
            "prompt": result.prompt_tokens,
            "completion": result.completion_tokens,
            "total": result.prompt_tokens + result.completion_tokens,
        },
    }


router = APIRouter(prefix="/api/v1/account/model-provider", tags=["account-provider"])


@router.get("")
def get_account_provider(
    response: Response,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    response.headers.update(NO_STORE_HEADERS)
    with SessionLocal() as db:
        binding, row = _current_config(db, context.user_id)
        return _serialize(binding, row)


@router.put("")
async def put_account_provider(
    request: Request,
    response: Response,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    payload = await _safe_payload(request, AccountProviderWriteIn)
    settings = get_settings()
    try:
        normalized_url = normalize_provider_base_url(
            payload.base_url,
            allowed_origins=allowed_provider_origins(settings),
        )
        endpoint_hash = provider_endpoint_sha256(normalized_url)
        keyring = load_provider_keyring(settings)
    except ProviderEndpointRejected:
        raise _safe_http(
            422,
            "provider_endpoint_not_allowed",
            "模型服务地址不在部署允许的 HTTPS 地址范围内",
        ) from None
    except ProviderCredentialError:
        raise _safe_http(
            503,
            "provider_credential_store_unavailable",
            "账户模型密钥存储暂不可用",
        ) from None
    model_name = _clean_model_name(payload.model)
    now = utc_now_naive()
    with SessionLocal() as db:
        binding, old = _current_config(db, context.user_id, lock=True)
        actual = binding.lock_version if binding is not None else 0
        if payload.expected_revision != actual:
            raise _safe_http(
                409,
                "provider_config_revision_conflict",
                "模型配置已在其他页面更新，请刷新后重试",
                actual_revision=actual,
            )
        if payload.api_key is None:
            if old is None:
                raise _safe_http(
                    422,
                    "provider_api_key_required",
                    "首次保存模型配置时必须填写 API Key",
                )
            try:
                secret = _decrypt_row(old, settings)
            except ProviderCredentialError:
                raise _safe_http(
                    409,
                    "provider_api_key_unavailable",
                    "原 API Key 已不可用，请输入新密钥",
                ) from None
        else:
            secret = payload.api_key

        revision = actual + 1
        config_id = new_id()
        aad = CredentialAAD(
            user_id=context.user_id,
            config_id=config_id,
            revision=revision,
            endpoint_sha256=endpoint_hash,
            model_name=model_name,
        )
        encrypted = keyring.encrypt(secret, aad)
        row = AccountProviderConfigRow(
            id=config_id,
            user_id=context.user_id,
            revision=revision,
            base_url=normalized_url,
            model_name=model_name,
            endpoint_sha256=endpoint_hash,
            secret_ciphertext=encrypted.ciphertext,
            secret_nonce=encrypted.nonce,
            encryption_key_id=encrypted.key_id,
            secret_schema_version=encrypted.schema_version,
            state="usable",
            created_by_user_id=context.user_id,
            created_at=now,
        )
        try:
            db.add(row)
            # A concurrent first save has no binding row for FOR UPDATE to
            # lock. The unique revision constraint can therefore reject at
            # flush, so the flush belongs inside the optimistic-lock handler.
            db.flush()
            if old is not None:
                old.state = "superseded"
                old.superseded_at = now
            if binding is None:
                binding = AccountProviderBindingRow(
                    user_id=context.user_id,
                    current_config_id=row.id,
                    lock_version=revision,
                    created_at=now,
                    updated_at=now,
                )
                db.add(binding)
            else:
                binding.current_config_id = row.id
                binding.lock_version = revision
                binding.updated_at = now
            db.commit()
        except IntegrityError:
            # PostgreSQL row locks serialize replacements. SQLite ignores
            # SELECT FOR UPDATE, so the unique revision constraint remains the
            # final authority and is translated into the same optimistic-lock
            # contract instead of leaking a 500 response.
            db.rollback()
            latest_binding, _ = _current_config(db, context.user_id)
            latest_revision = (
                latest_binding.lock_version if latest_binding is not None else 0
            )
            raise _safe_http(
                409,
                "provider_config_revision_conflict",
                "模型配置已在其他页面更新，请刷新后重试",
                actual_revision=latest_revision,
            ) from None
        result = _serialize(binding, row)
    response.headers.update(NO_STORE_HEADERS)
    return result


@router.delete("")
async def delete_account_provider(
    request: Request,
    response: Response,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    payload = await _safe_payload(request, ExpectedRevisionIn)
    now = utc_now_naive()
    with SessionLocal() as db:
        binding, row = _current_config(db, context.user_id, lock=True)
        actual = binding.lock_version if binding is not None else 0
        if payload.expected_revision != actual:
            raise _safe_http(
                409,
                "provider_config_revision_conflict",
                "模型配置已在其他页面更新，请刷新后重试",
                actual_revision=actual,
            )
        if binding is None or row is None:
            raise _safe_http(404, "provider_config_not_found", "尚未保存账户模型配置")
        row.state = "scrubbed"
        row.revoked_at = now
        row.secret_ciphertext = None
        row.secret_nonce = None
        row.encryption_key_id = None
        binding.current_config_id = None
        binding.lock_version = actual + 1
        binding.updated_at = now
        # A delete is a cryptographic revocation, not merely removal of the
        # current pointer. Scrub every retained historical envelope so queued
        # runs cannot continue with an older account secret.
        for historical in db.scalars(
            select(AccountProviderConfigRow).where(
                AccountProviderConfigRow.user_id == context.user_id,
                AccountProviderConfigRow.secret_ciphertext.is_not(None),
            )
        ).all():
            historical.state = "scrubbed"
            historical.revoked_at = historical.revoked_at or now
            historical.secret_ciphertext = None
            historical.secret_nonce = None
            historical.encryption_key_id = None
        db.commit()
        result = _serialize(binding, None)
    response.headers.update(NO_STORE_HEADERS)
    return result


@router.post("/test")
async def test_account_provider(
    request: Request,
    response: Response,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    payload = await _safe_payload(request, ExpectedRevisionIn)
    allowed, retry_after = _provider_test_limiter.check(context.user_id)
    if not allowed:
        raise HTTPException(
            429,
            detail={
                "code": "provider_test_rate_limited",
                "message": "模型连接测试过于频繁，请稍后重试",
            },
            headers={**NO_STORE_HEADERS, "Retry-After": str(retry_after)},
        )
    settings = get_settings()
    with SessionLocal() as db:
        binding, row = _current_config(db, context.user_id)
        actual = binding.lock_version if binding is not None else 0
        if payload.expected_revision != actual:
            raise _safe_http(
                409,
                "provider_config_revision_conflict",
                "模型配置已在其他页面更新，请刷新后重试",
                actual_revision=actual,
            )
        if row is None:
            raise _safe_http(404, "provider_config_not_found", "尚未保存账户模型配置")
        config_id = row.id
        try:
            secret = _decrypt_row(row, settings)
            normalized = validate_provider_request_url(
                row.base_url,
                allowed_origins=allowed_provider_origins(settings),
            )
        except (ProviderCredentialError, ProviderEndpointRejected):
            raise _safe_http(
                409,
                "provider_config_unavailable",
                "账户模型配置已失效，请重新保存",
            ) from None
        resolved = _run_local_settings(
            settings,
            api_key=secret,
            base_url=normalized,
            model_name=row.model_name,
            config_id=row.id,
            user_id=row.user_id,
            revision=row.revision,
            endpoint_sha256=row.endpoint_sha256,
        )
        bounded = resolved.model_copy(
            update={
                "enable_model_extraction": True,
                "provider_timeout_seconds": min(settings.provider_timeout_seconds, 10.0),
                "provider_total_deadline_seconds": min(
                    settings.provider_total_deadline_seconds or 15.0, 15.0
                ),
                "provider_max_attempts": 1,
                "provider_max_completion_tokens": min(
                    settings.provider_max_completion_tokens or 64, 64
                ),
                "provider_max_response_bytes": min(
                    settings.provider_max_response_bytes or 16_384, 16_384
                ),
            }
        )
    provider = OpenAICompatibleProvider(
        bounded,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0, jitter_ratio=0),
    )
    # The gateway is intentionally synchronous because workers use it from
    # ordinary threads. Offload this explicit HTTP endpoint so a slow upstream
    # cannot block the Uvicorn event loop for the bounded probe deadline.
    result = await run_in_threadpool(_test_result, provider)
    now = utc_now_naive()
    safe_payload = {
        "json_contract_ok": result["json_contract_ok"],
        "reachable": result["reachable"],
        "authorized": result["authorized"],
        "latency_ms": result["latency_ms"],
        "token_usage": result["token_usage"],
    }
    with SessionLocal() as db:
        binding, current = _current_config(db, context.user_id, lock=True)
        actual = binding.lock_version if binding is not None else 0
        if (
            payload.expected_revision != actual
            or current is None
            or current.id != config_id
        ):
            raise _safe_http(
                409,
                "provider_config_revision_conflict",
                "测试期间模型配置已更新，本次结果未保存",
                actual_revision=actual,
            )
        current.last_tested_at = now
        current.last_test_status = result["status"]
        current.last_test_category = result["category"]
        current.last_test_payload = safe_payload
        db.commit()
    response.headers.update(NO_STORE_HEADERS)
    return {
        "profile_revision": payload.expected_revision,
        "tested_at": now,
        **result,
    }
