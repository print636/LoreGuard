from __future__ import annotations

import base64
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.account_provider import (
    account_provider_runtime,
    bind_account_provider_to_run,
    runtime_for_analysis_run,
)
from app.db import (
    AccountProviderConfigRow,
    AnalysisRunRow,
    DocumentNarrativeContextRevisionRow,
    DocumentRow,
    ProjectRow,
    SessionLocal,
    WorkspaceMemberRow,
)
from app.main import _safe_run_provider_execution, app, settings, write_limiter
from app.provider import OpenAICompatibleProvider, ProviderError, RetryPolicy
from app.service import capture_run_inputs, execute_analysis
from app.time_utils import utc_now_naive


CANARY_A = "sk-account-a-never-echo-7ef8d9"
CANARY_B = "sk-account-b-never-echo-0a1b2c"
ALLOWED_ORIGINS = "https://api.example.com,https://relay.example.com"


def test_failed_run_without_provider_evidence_is_not_reported_as_provider_failure():
    run = AnalysisRunRow(
        status="failed",
        prompt_tokens=0,
        completion_tokens=0,
        provider_identity={
            "source": "account_byok",
            "configured": True,
            "config_revision": 1,
            "model_alias": "safe-model",
            "endpoint_configuration_sha256": "a" * 64,
        },
    )

    execution = _safe_run_provider_execution(run)

    assert execution["status"] == "not_used"
    assert execution["planned_source"] == "account_byok"


def test_completed_baseline_fallback_reports_verified_provider_failure():
    run = AnalysisRunRow(
        status="completed",
        prompt_tokens=0,
        completion_tokens=0,
        provider_identity={
            "source": "account_byok",
            "configured": True,
            "config_revision": 1,
            "model_alias": "safe-model",
            "endpoint_configuration_sha256": "a" * 64,
        },
    )
    diagnostic = {
        "model": {
            "attempted_chunks": 1,
            "succeeded_chunks": 0,
            "failed_chunks": 1,
            "provider_calls": [
                {"status": "failure", "category": "transport"},
            ],
        },
        "untrusted_nested_payload": {
            "provider_calls": [{"status": "success"}],
        },
    }

    execution = _safe_run_provider_execution(run, diagnostic)

    assert execution["status"] == "provider_failed"


def test_zero_token_success_diagnostic_proves_provider_use():
    run = AnalysisRunRow(
        status="completed",
        prompt_tokens=0,
        completion_tokens=0,
        provider_identity={"source": "service_default", "configured": True},
    )

    execution = _safe_run_provider_execution(
        run,
        {
            "usage_accounting": {
                "provider_calls": [{"status": "success", "category": "success"}],
            }
        },
    )

    assert execution["status"] == "used"


@contextmanager
def provider_test_settings(*, auth_mode: str = "required"):
    names = (
        "auth_mode",
        "auth_secret_key",
        "auth_cookie_secure",
        "auth_cookie_samesite",
        "account_model_active_key_id",
        "account_model_keyring_json",
        "account_model_keyring_file",
        "account_model_allowed_origins",
        "provider_transport_allowed_origins",
        "openai_api_key",
        "openai_base_url",
        "openai_model",
        "enable_model_extraction",
    )
    previous = {name: getattr(settings, name) for name in names}
    settings.auth_mode = auth_mode
    settings.auth_secret_key = "test-only-auth-secret-key-32-bytes-minimum"
    settings.auth_cookie_secure = False
    settings.auth_cookie_samesite = "lax"
    settings.account_model_active_key_id = "test-v1"
    settings.account_model_keyring_json = json.dumps(
        {"version": 1, "keys": {"test-v1": base64.b64encode(b"K" * 32).decode()}}
    )
    settings.account_model_keyring_file = ""
    settings.account_model_allowed_origins = ALLOWED_ORIGINS
    settings.provider_transport_allowed_origins = ""
    settings.openai_api_key = "server-global-key-must-not-leak"
    settings.openai_base_url = "https://api.openai.com/v1"
    settings.openai_model = "server-model"
    settings.enable_model_extraction = True
    write_limiter.events.clear()
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)
        write_limiter.events.clear()


def register(client: TestClient) -> tuple[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"byok-{uuid4().hex}@example.com",
            "password": "correct horse battery staple",
            "display_name": "BYOK 测试作者",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["user"]["id"], response.headers["X-CSRF-Token"]


def save_provider(
    client: TestClient,
    csrf: str,
    *,
    expected_revision: int,
    api_key: str | None,
    model: str,
    base_url: str = "https://api.example.com/v1",
):
    body = {
        "expected_revision": expected_revision,
        "base_url": base_url,
        "model": model,
    }
    if api_key is not None:
        body["api_key"] = api_key
    return client.put(
        "/api/v1/account/model-provider",
        headers={"X-CSRF-Token": csrf},
        json=body,
    )


def test_provider_api_never_echoes_secret_and_revisioned_delete_scrubs_history():
    with provider_test_settings(), TestClient(app) as client:
        user_id, csrf = register(client)
        created = save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="story-model-v1",
        )
        assert created.status_code == 200, created.text
        assert created.headers["cache-control"] == "no-store"
        assert created.json()["revision"] == 1
        assert created.json()["api_key"] == {
            "state": "set",
            "masked": "••••••••••••",
        }
        assert CANARY_A not in created.text

        retained = save_provider(
            client,
            csrf,
            expected_revision=1,
            api_key=None,
            model="story-model-v2",
            base_url="https://relay.example.com/openai/v1",
        )
        assert retained.status_code == 200, retained.text
        assert retained.json()["revision"] == 2
        assert CANARY_A not in retained.text

        stale = save_provider(
            client,
            csrf,
            expected_revision=1,
            api_key=CANARY_B,
            model="stale-write",
        )
        assert stale.status_code == 409
        assert stale.json()["detail"]["actual_revision"] == 2
        assert CANARY_B not in stale.text

        with SessionLocal() as db:
            rows = db.scalars(
                select(AccountProviderConfigRow)
                .where(AccountProviderConfigRow.user_id == user_id)
                .order_by(AccountProviderConfigRow.revision)
            ).all()
            assert [row.state for row in rows] == ["superseded", "usable"]
            assert all(row.secret_ciphertext for row in rows)
            assert all(CANARY_A.encode() not in bytes(row.secret_ciphertext) for row in rows)
            assert CANARY_A not in repr(rows)
            current = rows[-1]
            current.last_tested_at = utc_now_naive()
            current.last_test_status = "failed"
            current.last_test_category = "rate_limit"
            current.last_test_payload = {
                "json_contract_ok": True,
                "reachable": True,
                "authorized": True,
                "latency_ms": 42,
                "token_usage": {"prompt": 2, "completion": 3, "total": 999},
                "upstream_body": CANARY_A,
                "api_key": CANARY_A,
            }
            db.commit()

        sanitized = client.get("/api/v1/account/model-provider")
        assert sanitized.status_code == 200
        assert CANARY_A not in sanitized.text
        assert sanitized.json()["last_test"] == {
            "profile_revision": 2,
            "status": "failed",
            "category": "rate_limit",
            "tested_at": sanitized.json()["last_test"]["tested_at"],
            "json_contract_ok": True,
            "reachable": True,
            "authorized": True,
            "latency_ms": 42,
            "token_usage": {"prompt": 2, "completion": 3, "total": 5},
        }

        deleted = client.request(
            "DELETE",
            "/api/v1/account/model-provider",
            headers={"X-CSRF-Token": csrf},
            json={"expected_revision": 2},
        )
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["revision"] == 3
        assert deleted.json()["configured"] is False
        with SessionLocal() as db:
            rows = db.scalars(
                select(AccountProviderConfigRow).where(
                    AccountProviderConfigRow.user_id == user_id
                )
            ).all()
            assert rows
            assert all(row.state == "scrubbed" for row in rows)
            assert all(row.secret_ciphertext is None for row in rows)
            assert all(row.secret_nonce is None for row in rows)
            assert all(row.encryption_key_id is None for row in rows)


def test_secret_bearing_validation_is_generic_csrf_protected_and_no_store():
    with provider_test_settings(), TestClient(app) as client:
        _, csrf = register(client)
        missing_csrf = client.put(
            "/api/v1/account/model-provider",
            json={
                "expected_revision": 0,
                "base_url": "https://api.example.com/v1",
                "model": "model",
                "api_key": CANARY_A,
            },
        )
        assert missing_csrf.status_code == 403
        assert missing_csrf.headers["cache-control"] == "no-store"
        assert CANARY_A not in missing_csrf.text

        invalid = client.put(
            "/api/v1/account/model-provider",
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
            content=json.dumps(
                {
                    "expected_revision": 0,
                    "base_url": "https://api.example.com/v1",
                    "model": "model",
                    "api_key": f"{CANARY_A} bad whitespace",
                }
            ),
        )
        assert invalid.status_code == 422
        assert invalid.json()["detail"]["code"] == "invalid_provider_payload"
        assert CANARY_A not in invalid.text


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.com/v1",
        "https://localhost/v1",
        "https://127.0.0.1/v1",
        "https://user@api.example.com/v1",
        "https://api.example.com/v1?key=secret",
        "https://api.example.com/v1#fragment",
        "https://not-allowlisted.example/v1",
    ],
)
def test_provider_api_rejects_ssrf_and_non_allowlisted_endpoints(base_url: str):
    with provider_test_settings(), TestClient(app) as client:
        _, csrf = register(client)
        response = save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="story-model",
            base_url=base_url,
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "provider_endpoint_not_allowed"
        assert CANARY_A not in response.text


def test_provider_api_rejects_unicode_key_without_echo_or_transport():
    unicode_canary = "sk-不可回显-credential"
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    with provider_test_settings(), TestClient(app) as client:
        _, csrf = register(client)
        response = save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=unicode_canary,
            model="story-model",
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "invalid_provider_payload"
        assert unicode_canary not in response.text

        compromised = settings.model_copy(
            update={
                "openai_api_key": unicode_canary,
                "openai_base_url": "https://api.example.com/v1",
                "openai_model": "story-model",
                "enable_model_extraction": True,
                "provider_transport_allowed_origins": "https://api.example.com",
            }
        )
        provider = OpenAICompatibleProvider(
            compromised,
            transport=httpx.MockTransport(handle),
        )
        with pytest.raises(ProviderError) as raised:
            provider.complete("system", "user")
        assert raised.value.category == "credential_invalid"
        assert unicode_canary not in repr(raised.value)
        assert calls == []


def test_two_accounts_are_isolated_and_required_mode_never_borrows_global_key():
    with provider_test_settings(), TestClient(app) as first, TestClient(app) as second:
        first_id, first_csrf = register(first)
        second_id, second_csrf = register(second)
        assert save_provider(
            first,
            first_csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="first-model",
        ).status_code == 200
        assert save_provider(
            second,
            second_csrf,
            expected_revision=0,
            api_key=CANARY_B,
            model="second-model",
        ).status_code == 200

        assert first.get("/api/v1/account/model-provider").json()["model"] == "first-model"
        assert second.get("/api/v1/account/model-provider").json()["model"] == "second-model"
        with SessionLocal() as db:
            first_runtime = account_provider_runtime(db, user_id=first_id)
            second_runtime = account_provider_runtime(db, user_id=second_id)
            no_config = account_provider_runtime(db, user_id=f"missing-{uuid4().hex}")
        assert first_runtime.settings.openai_api_key == CANARY_A
        assert second_runtime.settings.openai_api_key == CANARY_B
        assert first_runtime.identity["model_alias"] == "first-model"
        assert second_runtime.identity["model_alias"] == "second-model"
        assert no_config.available is False
        assert no_config.settings.openai_api_key == ""
        assert no_config.identity["source"] == "deterministic_only"


def test_frozen_run_keeps_revision_on_replace_and_delete_revokes_queued_use():
    with provider_test_settings(), TestClient(app) as client:
        user_id, csrf = register(client)
        assert save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="model-v1",
        ).status_code == 200
        run = AnalysisRunRow(requested_by_user_id=user_id)
        with SessionLocal() as db:
            bind_account_provider_to_run(db, run, user_id=user_id)
        frozen_config_id = run.provider_config_id
        frozen_identity = dict(run.provider_identity)

        assert save_provider(
            client,
            csrf,
            expected_revision=1,
            api_key=CANARY_B,
            model="model-v2",
        ).status_code == 200
        run.provider_config_id = frozen_config_id
        run.provider_identity = frozen_identity
        with SessionLocal() as db:
            frozen_runtime = runtime_for_analysis_run(db, run)
        assert frozen_runtime.available is True
        assert frozen_runtime.settings.openai_api_key == CANARY_A
        assert frozen_runtime.settings.openai_model == "model-v1"

        deleted = client.request(
            "DELETE",
            "/api/v1/account/model-provider",
            headers={"X-CSRF-Token": csrf},
            json={"expected_revision": 2},
        )
        assert deleted.status_code == 200
        with SessionLocal() as db:
            revoked_runtime = runtime_for_analysis_run(db, run)
        assert revoked_runtime.available is False
        assert revoked_runtime.settings.openai_api_key == ""
        assert revoked_runtime.unavailable_reason == "credential_revoked"


def test_transport_edge_uses_frozen_deployment_allowlist_not_mutated_base_url():
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    compromised = settings.model_copy(
        update={
            "openai_api_key": CANARY_A,
            "openai_base_url": "https://attacker.example/v1",
            "openai_model": "model",
            "enable_model_extraction": True,
            "provider_transport_allowed_origins": "https://api.example.com",
        }
    )
    provider = OpenAICompatibleProvider(
        compromised,
        transport=httpx.MockTransport(handle),
    )
    with pytest.raises(ProviderError) as raised:
        provider.complete("system", "user")
    assert raised.value.category == "endpoint_rejected"
    assert calls == []


def test_tampered_allowlisted_base_url_cannot_reuse_bound_ciphertext_or_send_key():
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    with provider_test_settings(), TestClient(app) as client:
        user_id, csrf = register(client)
        assert save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="endpoint-bound-model",
            base_url="https://api.example.com/v1",
        ).status_code == 200
        with SessionLocal() as db:
            resolved_before_tamper = account_provider_runtime(db, user_id=user_id)
            row = db.get(AccountProviderConfigRow, resolved_before_tamper.config_id)
            assert row is not None
            # Keep the authenticated endpoint hash/ciphertext untouched while
            # switching only the URL to a different administrator-allowed
            # endpoint. Neither decrypt nor final transport may accept it.
            row.base_url = "https://relay.example.com/v1"
            db.commit()
        with SessionLocal() as db:
            resolved_after_tamper = account_provider_runtime(db, user_id=user_id)
        assert resolved_after_tamper.available is False
        assert resolved_after_tamper.settings.openai_api_key == ""

        provider = OpenAICompatibleProvider(
            resolved_before_tamper.settings,
            transport=httpx.MockTransport(handle),
        )
        with pytest.raises(ProviderError) as raised:
            provider.complete("system", "user")
        assert raised.value.category == "credential_revoked"
        assert calls == []


def test_unicode_model_alias_remains_usable_at_request_guard():
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with provider_test_settings(), TestClient(app) as client:
        user_id, csrf = register(client)
        assert save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="叙事-模型-v1",
        ).status_code == 200
        with SessionLocal() as db:
            runtime = account_provider_runtime(db, user_id=user_id)
        provider = OpenAICompatibleProvider(
            runtime.settings,
            transport=httpx.MockTransport(handle),
        )

        result = provider.complete("system", "user")

        assert result.text == '{"ok":true}'
        assert len(calls) == 1


def test_anonymous_service_default_identity_is_frozen_against_config_drift():
    with provider_test_settings(auth_mode="anonymous"):
        with SessionLocal() as db:
            initial = account_provider_runtime(db, user_id=f"unbound-{uuid4().hex}")
        run = AnalysisRunRow(
            requested_by_user_id=f"unbound-{uuid4().hex}",
            provider_config_id=None,
            provider_identity=initial.identity,
        )
        drifted = settings.model_copy(update={"openai_model": "changed-after-queue"})
        with SessionLocal() as db:
            runtime = runtime_for_analysis_run(db, run, base_settings=drifted)
        assert runtime.available is False
        assert runtime.settings.openai_api_key == ""
        assert runtime.unavailable_reason == "identity_mismatch"


def test_worker_resolves_only_the_provider_revision_frozen_on_the_run():
    captured: dict[str, object] = {}

    class CapturingPipeline:
        accepts_run_local_settings = True

        def __init__(self, *, settings):
            captured["api_key"] = settings.openai_api_key
            captured["base_url"] = settings.openai_base_url
            captured["model"] = settings.openai_model
            captured["transport_origins"] = settings.provider_transport_allowed_origins

        def run(self, documents, on_stage, checkpoint):
            checkpoint()
            on_stage("check", 75, "规则检查完成")
            return SimpleNamespace(
                directives=[],
                issues=[],
                warnings=[],
                prompt_tokens=0,
                completion_tokens=0,
                model_used=False,
                diagnostics={
                    "model": {
                        "enabled": True,
                        "configured": True,
                        "succeeded_chunks": 0,
                        "failed_chunks": 0,
                        "skipped_chunks": 0,
                        "invalid_records": 0,
                        "empty_response_chunks": 0,
                    },
                    "provenance": {
                        "schema_version": 1,
                        "directives": [],
                        "issues": [],
                    },
                },
            )

        def conservative_run_token_debit(self, result):
            return result.prompt_tokens + result.completion_tokens

    with provider_test_settings(), TestClient(app) as client:
        user_id, csrf = register(client)
        assert save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="worker-model",
            base_url="https://api.example.com/v1",
        ).status_code == 200
        with SessionLocal() as db:
            workspace_id = db.scalar(
                select(WorkspaceMemberRow.workspace_id).where(
                    WorkspaceMemberRow.user_id == user_id
                )
            )
            project = ProjectRow(workspace_id=workspace_id, name="worker-byok")
            db.add(project)
            db.flush()
            document = DocumentRow(
                project_id=project.id,
                name="chapter.md",
                content="林澈走进车站。",
                version=1,
            )
            run = AnalysisRunRow(
                project_id=project.id,
                requested_by_user_id=user_id,
            )
            db.add_all([document, run])
            db.flush()
            capture_run_inputs(db, run, [document])
            bind_account_provider_to_run(db, run, user_id=user_id)
            db.commit()
            run_id = run.id

        with patch("app.service.AnalysisPipeline", CapturingPipeline):
            execute_analysis(run_id, raise_on_failure=True)
        assert captured == {
            "api_key": CANARY_A,
            "base_url": "https://api.example.com/v1",
            "model": "worker-model",
            "transport_origins": (
                "https://api.example.com,https://relay.example.com"
            ),
        }
        with SessionLocal() as db:
            assert db.get(AnalysisRunRow, run_id).status == "completed"


def test_context_inference_uses_account_provider_and_persists_safe_identity():
    captured: dict[str, str] = {}
    model_payload = {
        "document_role": "canon",
        "publication_status": "published",
        "scope": {
            "schema_version": 1,
            "timeline_key": "main",
            "release": {"key": "v3", "ordinal": 3},
        },
        "confidence": 0.91,
        "reasoning": "原文明确给出资料类型、发布状态与版本。",
        "evidence": [
            {
                "line_start": 1,
                "line_end": 1,
                "text": "资料类型：世界设定",
                "supported_fields": ["document_role"],
            },
            {
                "line_start": 2,
                "line_end": 2,
                "text": "状态：已发布",
                "supported_fields": ["publication_status"],
            },
            {
                "line_start": 3,
                "line_end": 3,
                "text": "版本：v3",
                "supported_fields": ["release"],
            },
        ],
    }

    def provider_factory(provider_settings):
        captured["api_key"] = provider_settings.openai_api_key
        captured["base_url"] = provider_settings.openai_base_url
        captured["model"] = provider_settings.openai_model

        def respond(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    model_payload, ensure_ascii=False
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                },
            )

        bounded = provider_settings.model_copy(
            update={
                "enable_model_extraction": True,
                "provider_max_attempts": 1,
                "provider_total_deadline_seconds": 2.0,
            }
        )
        return OpenAICompatibleProvider(
            bounded,
            transport=httpx.MockTransport(respond),
            retry_policy=RetryPolicy(
                max_attempts=1, base_delay_seconds=0, jitter_ratio=0
            ),
            sleep=lambda _: None,
        )

    with provider_test_settings(), TestClient(app) as client:
        _, csrf = register(client)
        saved = save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="context-model",
            base_url="https://api.example.com/v1",
        )
        assert saved.status_code == 200, saved.text
        project = client.post(
            "/api/v1/projects",
            headers={"X-CSRF-Token": csrf},
            json={"name": f"context-byok-{uuid4().hex}"},
        )
        assert project.status_code == 201, project.text
        document = client.post(
            f"/api/v1/projects/{project.json()['id']}/documents/text",
            headers={"X-CSRF-Token": csrf},
            json={
                "name": "资料.md",
                "content": "资料类型：世界设定\n状态：已发布\n版本：v3",
                "document_role": "reference",
                "narrative_context": {
                    "resolution_state": "unresolved",
                    "publication_status": "unknown",
                    "scope": {"timeline_key": "main"},
                },
            },
        )
        assert document.status_code == 201, document.text
        path = (
            f"/api/v1/projects/{project.json()['id']}/documents/"
            f"{document.json()['id']}/narrative-context/inference"
        )
        with patch(
            "app.main._narrative_context_inference_provider",
            side_effect=provider_factory,
        ):
            response = client.post(
                path,
                headers={"X-CSRF-Token": csrf},
                json={"expected_revision": 1},
            )
        assert response.status_code == 201, response.text
        assert captured == {
            "api_key": CANARY_A,
            "base_url": "https://api.example.com/v1",
            "model": "context-model",
        }
        assert CANARY_A not in response.text
        assert "https://api.example.com" not in response.text
        with SessionLocal() as db:
            revision = db.scalar(
                select(DocumentNarrativeContextRevisionRow)
                .where(
                    DocumentNarrativeContextRevisionRow.document_id
                    == document.json()["id"]
                )
                .order_by(DocumentNarrativeContextRevisionRow.revision.desc())
            )
            assert revision.inference_provider_config_id is not None
            assert revision.inference_provider_identity["source"] == "account_byok"
            assert revision.inference_provider_identity["model_alias"] == "context-model"
            serialized_identity = json.dumps(revision.inference_provider_identity)
            assert CANARY_A not in serialized_identity
            assert "https://api.example.com" not in serialized_identity


def test_first_save_flush_conflict_is_a_safe_optimistic_lock_response():
    with provider_test_settings(), TestClient(app) as client:
        _, csrf = register(client)
        db = SessionLocal()
        original_flush = db.flush
        flush_calls = 0

        def conflict_once(*args, **kwargs):
            nonlocal flush_calls
            flush_calls += 1
            # The initial SELECT FOR UPDATE performs a harmless autoflush;
            # fail the explicit insert flush that follows it.
            if flush_calls == 2:
                raise IntegrityError("unique conflict", {}, RuntimeError())
            return original_flush(*args, **kwargs)

        with (
            patch.object(db, "flush", side_effect=conflict_once),
            patch("app.account_provider.SessionLocal", return_value=db),
        ):
            response = save_provider(
                client,
                csrf,
                expected_revision=0,
                api_key=CANARY_A,
                model="conflicting-model",
            )
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "provider_config_revision_conflict"
        assert CANARY_A not in response.text


def test_early_origin_and_rate_limit_rejections_are_never_cacheable():
    with provider_test_settings(), TestClient(app) as client:
        _, csrf = register(client)
        origin_rejected = client.put(
            "/api/v1/account/model-provider",
            headers={"X-CSRF-Token": csrf, "Origin": "https://evil.example"},
            json={
                "expected_revision": 0,
                "base_url": "https://api.example.com/v1",
                "model": "model",
                "api_key": CANARY_A,
            },
        )
        assert origin_rejected.status_code == 403
        assert origin_rejected.headers["cache-control"] == "no-store"
        previous_limit = write_limiter.limit
        write_limiter.limit = 0
        try:
            rate_limited = client.put(
                "/api/v1/account/model-provider",
                headers={"X-CSRF-Token": csrf},
                json={
                    "expected_revision": 0,
                    "base_url": "https://api.example.com/v1",
                    "model": "model",
                    "api_key": CANARY_A,
                },
            )
        finally:
            write_limiter.limit = previous_limit
            write_limiter.events.clear()
        assert rate_limited.status_code == 429
        assert rate_limited.headers["cache-control"] == "no-store"
        assert CANARY_A not in origin_rejected.text + rate_limited.text


def test_slow_connection_probe_is_offloaded_from_the_async_request_handler():
    observed: dict[str, object] = {}

    async def fake_threadpool(function, *args):
        observed["function"] = function
        observed["provider"] = args[0]
        return {
            "status": "passed",
            "category": "success",
            "json_contract_ok": True,
            "reachable": True,
            "authorized": True,
            "latency_ms": 7,
            "token_usage": {"prompt": 1, "completion": 1, "total": 2},
        }

    with provider_test_settings(), TestClient(app) as client:
        _, csrf = register(client)
        assert save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="probe-model",
        ).status_code == 200
        with patch(
            "app.account_provider.run_in_threadpool", side_effect=fake_threadpool
        ):
            response = client.post(
                "/api/v1/account/model-provider/test",
                headers={"X-CSRF-Token": csrf},
                json={"expected_revision": 1},
            )
        assert response.status_code == 200, response.text
        assert observed["function"].__name__ == "_test_result"
        assert isinstance(observed["provider"], OpenAICompatibleProvider)


def test_delete_blocks_the_next_request_even_after_worker_decrypted_the_key():
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    with provider_test_settings(), TestClient(app) as client:
        user_id, csrf = register(client)
        assert save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="revocable-model",
        ).status_code == 200
        with SessionLocal() as db:
            runtime = account_provider_runtime(db, user_id=user_id)
        deleted = client.request(
            "DELETE",
            "/api/v1/account/model-provider",
            headers={"X-CSRF-Token": csrf},
            json={"expected_revision": 1},
        )
        assert deleted.status_code == 200
        provider = OpenAICompatibleProvider(
            runtime.settings,
            transport=httpx.MockTransport(respond),
        )
        with pytest.raises(ProviderError) as raised:
            provider.complete("system", "user")
        assert raised.value.category == "credential_revoked"
        assert requests == []


def test_revocation_during_retry_backoff_blocks_the_second_http_attempt():
    requests: list[httpx.Request] = []

    with provider_test_settings(), TestClient(app) as client:
        user_id, csrf = register(client)
        assert save_provider(
            client,
            csrf,
            expected_revision=0,
            api_key=CANARY_A,
            model="retry-revocation-model",
        ).status_code == 200
        with SessionLocal() as db:
            runtime = account_provider_runtime(db, user_id=user_id)

        def first_attempt_then_revoke(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            with SessionLocal() as db:
                row = db.get(AccountProviderConfigRow, runtime.config_id)
                row.state = "scrubbed"
                row.secret_ciphertext = None
                row.secret_nonce = None
                row.encryption_key_id = None
                db.commit()
            return httpx.Response(429)

        provider = OpenAICompatibleProvider(
            runtime.settings,
            transport=httpx.MockTransport(first_attempt_then_revoke),
            retry_policy=RetryPolicy(
                max_attempts=2, base_delay_seconds=0, jitter_ratio=0
            ),
            sleep=lambda _: None,
        )
        with pytest.raises(ProviderError) as raised:
            provider.complete("system", "user")
        assert raised.value.category == "credential_revoked"
        assert len(requests) == 1


def test_docker_build_context_excludes_local_account_master_keys():
    repository = Path(__file__).resolve().parents[1]
    patterns = {
        line.strip()
        for line in (repository / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "data/account-model-master.key" in patterns
    assert "data/account-model-keys/" in patterns
    git_patterns = {
        line.strip()
        for line in (repository / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "data/account-model-master.key" in git_patterns
    assert "data/account-model-keys/" in git_patterns
