from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.character_consistency_stage import (
    CharacterConsistencyStage,
    _FrozenDocument,
    _explicit_support_kind,
    _find_support_evidence,
    _safe_case_trace,
    _safe_baseline_hint,
    _safe_server_context,
    _select_authoritative_baselines,
    _targeted_completion_reserve,
    _trait_applies_to_release,
)
from app.character_drift import ConfirmedTraitSnapshot
from app.character_trait_extraction import (
    MAX_CHARACTER_SIGNAL_BASELINE_HINT_CHARS,
    MAX_CHARACTER_SIGNAL_SERVER_CONTEXT_CHARS,
    CharacterSignalChunk,
)
from app.config import Settings
from app.db import (
    AnalysisRunCharacterTraitInputRow,
    AnalysisRunRow,
    CharacterTraitCandidateRow,
    SessionLocal,
)
from app.main import app, settings as app_settings, write_limiter
from app.domain import EvidenceSpan
from app.narrative_context import NarrativeScopeV1, payload_sha256
from app.pipeline import DocumentInput
from app.service import _load_verified_snapshot, execute_analysis


class QueueProvider:
    def __init__(self, *responses: str | Exception):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("unexpected provider call")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(text=value, prompt_tokens=17, completion_tokens=9)


class CapturingQueueProvider(QueueProvider):
    def __init__(self, *responses: str | Exception):
        super().__init__(*responses)
        self.completion_caps: list[int] = []

    def fork_for_character_consistency(
        self, *, settings, stage, remaining_deadline_seconds=None
    ):
        if stage == "signal":
            self.completion_caps.append(settings.character_signal_max_completion_tokens)
        return self


@pytest.fixture(autouse=True)
def _clear_write_rate_limiter():
    write_limiter.events.clear()
    yield
    write_limiter.events.clear()


def _settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "stage-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "provider_max_attempts": 1,
        "character_consistency_stage_token_budget": 20_000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_targeted_reserve_reclaims_short_response_capacity_without_lifting_cap():
    chunk = CharacterSignalChunk(
        document_id="doc-1",
        document_name="draft.md",
        content="甲在晨会上主动发言。\n乙听完后离开。\n甲在晚会上主动攀谈。",
        global_line_start=7,
        source_kind="draft",
    )
    settings = _settings(character_signal_max_completion_tokens=4_096)

    assert _targeted_completion_reserve(settings, chunk) == 2_048
    assert _targeted_completion_reserve(
        settings, chunk, candidate_ranges=((7, 7),)
    ) == 2_048
    assert _targeted_completion_reserve(
        _settings(character_signal_max_completion_tokens=1_024), chunk
    ) == 1_024

    long_chunk = chunk.__class__(
        document_id="doc-2",
        document_name="long.md",
        content="甲" + "讲述过往" * 400,
        global_line_start=1,
        source_kind="draft",
    )
    assert _targeted_completion_reserve(settings, long_chunk) == 4_096

    paired_chunk = chunk.__class__(
        document_id="doc-3",
        document_name="paired.md",
        content="甲" + "准备" * 250 + "。\n她" + "记录" * 250 + "。",
        global_line_start=7,
        source_kind="draft",
    )
    assert 2_048 < _targeted_completion_reserve(
        settings, paired_chunk, candidate_ranges=((7, 8),)
    ) <= 4_096


def test_targeted_reserve_reaches_provider_fork_for_short_draft():
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="祁雾走进会议室。",
            narrative_context=_context(publication="draft"),
        )
        provider = CapturingQueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence="祁雾说话直来直往，这是他的核心性格。",
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            ),
            _response(),
            _response(),
            _response(),
        )
        _run_stage(_new_run(client, project["id"]), provider)

    assert provider.completion_caps[:2] == [4_096, 4_096]
    assert provider.completion_caps[2:] == [2_048, 2_048]


def test_empty_first_targeted_pass_recovers_safe_adjacent_pronoun_in_verification():
    antecedent = "这里只有祁雾。"
    observation = "她用奉承话术迂回交流。"
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content=f"{antecedent}\n{observation}",
            narrative_context=_context(publication="draft"),
        )
        pronoun_record = _record(
            character="祁雾",
            evidence=f"{antecedent}\n{observation}",
            polarity="negative",
            kind="action",
            dimension="core_personality",
            trait_key="directness",
            statement="祁雾用奉承话术迂回交流",
            line=1,
        )
        pronoun_record["source_line_end"] = 2
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence="祁雾说话直来直往，这是他的核心性格。",
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            ),
            _response(),
            _response(),
            _response(pronoun_record),
        )
        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_max_completion_tokens=512,
        )

    counts = result.diagnostics["counts"]
    assert counts["targeted_verification_scheduled_count"] == 1
    assert counts["targeted_verification_candidate_line_count"] == 2
    assert counts["targeted_verification_signal_added_count"] == 1
    assert counts["draft_observation_count"] == 1
    assert result.diagnostics["material_coverage"] == "complete"
    verification_prompt = provider.calls[3][1]
    assert "candidate_lines_only" in verification_prompt
    assert f"1: {antecedent}" in verification_prompt
    assert f"2: {observation}" in verification_prompt


def _context(
    *,
    publication: str,
    branch: str | None = None,
    exclusive_group: str | None = None,
) -> dict:
    scope: dict = {"timeline_key": "main"}
    if branch is not None:
        scope["branch"] = {
            "path": ["main", branch],
            "exclusive_group": exclusive_group,
        }
    return {
        "resolution_state": "confirmed",
        "publication_status": publication,
        "scope": scope,
    }


def _create_document(
    client: TestClient,
    project_id: str,
    *,
    name: str,
    role: str,
    content: str,
    narrative_context: dict,
) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/documents/text",
        json={
            "name": name,
            "document_role": role,
            "content": content,
            "narrative_context": narrative_context,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _new_run(client: TestClient, project_id: str) -> str:
    with patch("app.main.dispatch_analysis"):
        response = client.post(f"/api/v1/projects/{project_id}/analysis-runs")
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    with SessionLocal() as db:
        run = db.get(AnalysisRunRow, run_id)
        assert run is not None
        run.status = "running"
        db.commit()
    return run_id


def _run_stage(
    run_id: str,
    provider: QueueProvider,
    *,
    checkpoint=None,
    **setting_overrides,
):
    with SessionLocal() as db:
        run = db.get(AnalysisRunRow, run_id)
        assert run is not None
        documents, metadata = _load_verified_snapshot(db, run_id)
        result = CharacterConsistencyStage(
            settings=_settings(**setting_overrides),
            provider=provider,
            checkpoint=checkpoint,
        ).run(
            db,
            run_id=run_id,
            project_id=run.project_id,
            documents=documents,
            metadata=metadata,
            remaining_run_tokens=20_000,
        )
        run.status = "completed"
        db.commit()
        return result


def _record(
    *,
    evidence: str,
    polarity: str,
    kind: str,
    character: str = "林澈",
    dimension: str = "preference",
    trait_key: str = "食物偏好:蜜瓜",
    line: int = 1,
    statement: str | None = None,
) -> dict:
    return {
        "character": character,
        "dimension": dimension,
        "trait_key": trait_key,
        "statement": statement or evidence.rstrip("。"),
        "polarity": polarity,
        "stability": "core" if dimension == "core_personality" else "stable",
        "observation_kind": kind,
        "context": "",
        "key_object": "蜜瓜" if dimension == "preference" else "",
        "source_line_start": line,
        "source_line_end": line,
        "evidence": evidence,
    }


def _response(*records: dict) -> str:
    return json.dumps({"records": list(records)}, ensure_ascii=False)


def _confirm_only_candidate(client: TestClient, project_id: str, run_id: str) -> str:
    with SessionLocal() as db:
        rows = list(
            db.scalars(
                select(CharacterTraitCandidateRow).where(
                    CharacterTraitCandidateRow.source_run_id == run_id
                )
            ).all()
        )
    assert len(rows) == 1
    candidate = rows[0]
    response = client.post(
        (
            f"/api/v1/projects/{project_id}/characters/"
            f"{quote(candidate.character_key, safe='')}/profile-candidates/"
            f"{candidate.id}/decisions"
        ),
        json={"decision": "confirm", "expected_revision": 0},
        headers={"Idempotency-Key": f"stage-confirm-{uuid4().hex}"},
    )
    assert response.status_code == 201, response.text
    return candidate.id


def _confirm_all_candidates(
    client: TestClient, project_id: str, run_id: str
) -> list[str]:
    with SessionLocal() as db:
        rows = list(
            db.scalars(
                select(CharacterTraitCandidateRow)
                .where(CharacterTraitCandidateRow.source_run_id == run_id)
                .order_by(CharacterTraitCandidateRow.character_key)
            ).all()
        )
    identifiers: list[str] = []
    for candidate in rows:
        response = client.post(
            (
                f"/api/v1/projects/{project_id}/characters/"
                f"{quote(candidate.character_key, safe='')}/profile-candidates/"
                f"{candidate.id}/decisions"
            ),
            json={"decision": "confirm", "expected_revision": 0},
            headers={"Idempotency-Key": f"stage-confirm-all-{uuid4().hex}"},
        )
        assert response.status_code == 201, response.text
        identifiers.append(candidate.id)
    return identifiers


def _confirmed_directness_project(client: TestClient) -> dict:
    project = client.post(
        "/api/v1/projects", json={"name": f"定向补抽-{uuid4().hex}"}
    ).json()
    profile_line = "祁雾说话直来直往，这是他的核心性格。"
    _create_document(
        client,
        project["id"],
        name="profile.md",
        role="character_profile",
        content=profile_line,
        narrative_context=_context(publication="published"),
    )
    seed = _new_run(client, project["id"])
    seed_result = _run_stage(
        seed,
        QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence=profile_line,
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            )
        ),
    )
    assert seed_result.diagnostics["counts"]["targeted_pass_scheduled_count"] == 0
    _confirm_only_candidate(client, project["id"], seed)
    return project


def _confirmed_speech_project(client: TestClient) -> dict:
    project = client.post(
        "/api/v1/projects", json={"name": f"说话方式补抽-{uuid4().hex}"}
    ).json()
    profile_line = "祁雾说话始终简短，这是她稳定的说话方式。"
    _create_document(
        client,
        project["id"],
        name="profile.md",
        role="character_profile",
        content=profile_line,
        narrative_context=_context(publication="published"),
    )
    seed = _new_run(client, project["id"])
    _run_stage(
        seed,
        QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence=profile_line,
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="speech_pattern",
                    trait_key="concise_speech",
                    statement="说话始终简短",
                )
            )
        ),
    )
    _confirm_only_candidate(client, project["id"], seed)
    return project


def _confirmed_two_target_project(client: TestClient) -> tuple[dict, list[dict]]:
    project = client.post(
        "/api/v1/projects", json={"name": f"单目标补抽-{uuid4().hex}"}
    ).json()
    profile_lines = [
        "祁雾重视同伴互动，这是他的核心性格。",
        "祁雾说话直来直往，这也是他的核心性格。",
    ]
    profile_records = [
        _record(
            character="祁雾",
            evidence=profile_lines[0],
            polarity="positive",
            kind="explicit_declaration",
            dimension="core_personality",
            trait_key="a_companion_interaction",
            statement="重视同伴互动",
            line=1,
        ),
        _record(
            character="祁雾",
            evidence=profile_lines[1],
            polarity="positive",
            kind="explicit_declaration",
            dimension="core_personality",
            trait_key="z_directness",
            statement="说话直来直往",
            line=2,
        ),
    ]
    _create_document(
        client,
        project["id"],
        name="profile.md",
        role="character_profile",
        content="\n".join(profile_lines),
        narrative_context=_context(publication="published"),
    )
    seed = _new_run(client, project["id"])
    _run_stage(seed, QueueProvider(_response(*profile_records)))
    assert len(_confirm_all_candidates(client, project["id"], seed)) == 2
    return project, profile_records


def test_positive_state_does_not_suppress_targeted_recall_of_opposed_speech():
    first_line = "祁雾一口气说了很长一段话。"
    same_direction_state = "祁雾说明自己通常会简短回答。"
    second_line = "祁雾随后又连续讲了很久。"
    profile_line = "祁雾说话始终简短，这是她稳定的说话方式。"
    with TestClient(app) as client:
        project = _confirmed_speech_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content=f"{first_line}\n{same_direction_state}\n{second_line}",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence=profile_line,
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="speech_pattern",
                    trait_key="concise_speech",
                    statement="说话始终简短",
                )
            ),
            _response(
                _record(
                    character="祁雾",
                    evidence=first_line,
                    polarity="negative",
                    kind="speech_sample",
                    dimension="speech_pattern",
                    trait_key="concise_speech",
                    line=1,
                ),
                _record(
                    character="祁雾",
                    evidence=same_direction_state,
                    polarity="positive",
                    kind="state_description",
                    dimension="speech_pattern",
                    trait_key="concise_speech",
                    line=2,
                ),
            ),
            _response(),
            _response(
                _record(
                    character="祁雾",
                    evidence=second_line,
                    polarity="negative",
                    kind="speech_sample",
                    dimension="speech_pattern",
                    trait_key="concise_speech",
                    line=3,
                )
            ),
            json.dumps(
                {
                    "verdict": "needs_confirmation",
                    "explanation": "两次反向说话表现需要进一步确认。",
                    "citations": ["B01", "C01", "C02"],
                },
                ensure_ascii=False,
            ),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_max_completion_tokens=512,
        )

        counts = result.diagnostics["counts"]
        assert counts["targeted_pass_scheduled_count"] == 2
        assert counts["targeted_signal_added_count"] == 1
        assert counts["targeted_verification_scheduled_count"] == 1
        assert counts["targeted_verification_completed_count"] == 1
        assert counts["targeted_verification_signal_added_count"] == 1
        assert counts["draft_observation_count"] == 3
        assert result.diagnostics["outcome"] == "completed"
        assert len(result.issues) == 1
        assert len(provider.calls) == 5
        targeted_prompt = provider.calls[2][1]
        assert '"requested_polarity":"negative"' in targeted_prompt
        assert '"line_start":1,"line_end":1' in targeted_prompt
        assert '"line_start":2,"line_end":2' not in targeted_prompt
        assert '"baseline_polarity"' not in targeted_prompt
        assert '"baseline_hint":"说话始终简短"' in targeted_prompt
        verification_prompt = provider.calls[3][1]
        assert "检索视图：candidate_lines_only" in verification_prompt
        assert f"3: {second_line}" in verification_prompt
        assert f"1: {first_line}" not in verification_prompt


def test_two_independent_primary_speech_samples_suppress_targeted_recall():
    first_line = "祁雾一口气说了很长一段话。"
    second_line = "祁雾随后又连续讲了很久。"
    profile_line = "祁雾说话始终简短，这是她稳定的说话方式。"
    with TestClient(app) as client:
        project = _confirmed_speech_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content=f"{first_line}\n{second_line}",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence=profile_line,
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="speech_pattern",
                    trait_key="concise_speech",
                    statement="说话始终简短",
                )
            ),
            _response(
                _record(
                    character="祁雾",
                    evidence=first_line,
                    polarity="negative",
                    kind="speech_sample",
                    dimension="speech_pattern",
                    trait_key="concise_speech",
                    line=1,
                ),
                _record(
                    character="祁雾",
                    evidence=second_line,
                    polarity="negative",
                    kind="speech_sample",
                    dimension="speech_pattern",
                    trait_key="concise_speech",
                    line=2,
                ),
            ),
            json.dumps(
                {
                    "verdict": "needs_confirmation",
                    "explanation": "两次反向说话表现需要进一步确认。",
                    "citations": ["B01", "C01", "C02"],
                },
                ensure_ascii=False,
            ),
        )

        result = _run_stage(_new_run(client, project["id"]), provider)

        counts = result.diagnostics["counts"]
        assert counts["targeted_pass_scheduled_count"] == 0
        assert counts["draft_observation_count"] == 2
        assert result.diagnostics["outcome"] == "completed"
        assert len(result.issues) == 1
        assert len(provider.calls) == 3


def test_run1_pending_confirm_run2_detects_explicit_preference_conflict():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"角色阶段-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content="林澈喜欢蜜瓜。",
            narrative_context=_context(publication="published"),
        )
        first = _new_run(client, project["id"])
        first_result = _run_stage(
            first,
            QueueProvider(
                _response(
                    _record(
                        evidence="林澈喜欢蜜瓜。",
                        polarity="positive",
                        kind="explicit_declaration",
                    )
                )
            ),
        )
        assert not first_result.issues
        candidate_id = _confirm_only_candidate(client, project["id"], first)

        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="林澈明确说自己讨厌蜜瓜。",
            narrative_context=_context(publication="draft"),
        )
        second = _new_run(client, project["id"])
        second_result = _run_stage(
            second,
            QueueProvider(
                _response(
                    _record(
                        evidence="林澈喜欢蜜瓜。",
                        polarity="positive",
                        kind="explicit_declaration",
                        trait_key="喜欢的食物",
                        statement="喜欢蜜瓜",
                    )
                ),
                _response(
                    _record(
                        evidence="林澈明确说自己讨厌蜜瓜。",
                        polarity="negative",
                        kind="explicit_declaration",
                        trait_key="蜜瓜偏好",
                    )
                ),
                json.dumps(
                    {
                        "verdict": "contradicts",
                        "explanation": "当前明确偏好与已确认偏好相反。",
                        "citations": ["B01", "C01"],
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        assert (
            second_result.diagnostics["counts"]["targeted_pass_scheduled_count"]
            == 0
        )
        assert len(second_result.issues) == 1
        issue = second_result.issues[0]
        assert issue.category.value == "character_drift"
        assert issue.severity.value == "high"
        assert issue.metadata["confirmed_candidate_id"] == candidate_id
        assert issue.metadata["character_key"] == "林澈"
        assert len(issue.evidence) >= 2
        assert issue.suggestion == "请核对是否存在尚未记录的成长、伪装或情境依据"
        with SessionLocal() as db:
            candidates = list(
                db.scalars(
                    select(CharacterTraitCandidateRow).where(
                        CharacterTraitCandidateRow.project_id == project["id"]
                    )
                ).all()
            )
        assert len(candidates) == 1
        assert second_result.diagnostics["counts"]["persisted_reused"] == 1
        assert second_result.diagnostics["case_trace"] == [
            {
                "character_key": "林澈",
                "dimension": "preference",
                "comparison_key": "preference:蜜瓜",
                "matched_observation_count": 1,
                    "prepare_reason": "reported_opposed_preference",
                "review_outcome": "completed",
                "review_verdict": "contradicts",
                "citation_roles": ["B", "C"],
                "final_outcome": "conflict",
                "visible": True,
                "promote_reason": "model_contradicts",
            }
        ]


def test_nonempty_preference_behavior_gets_one_excluding_verification_pass():
    profile_line = "林澈喜欢蜜瓜。"
    first_line = "林澈拒绝吃蜜瓜。"
    second_line = "林澈明确说自己讨厌蜜瓜。"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"偏好非空补查-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content=profile_line,
            narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(
            seed,
            QueueProvider(
                _response(
                    _record(
                        evidence=profile_line,
                        polarity="positive",
                        kind="explicit_declaration",
                        trait_key="melon_preference",
                        statement="喜欢蜜瓜",
                    )
                )
            ),
        )
        _confirm_only_candidate(client, project["id"], seed)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content=f"{first_line}\n{second_line}",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                _record(
                    evidence=profile_line,
                    polarity="positive",
                    kind="explicit_declaration",
                    trait_key="melon_preference",
                    statement="喜欢蜜瓜",
                )
            ),
            _response(),
            _response(
                _record(
                    evidence=first_line,
                    polarity="negative",
                    kind="preference_expression",
                    trait_key="melon_preference",
                    statement="拒绝吃蜜瓜",
                    line=1,
                )
            ),
            _response(
                _record(
                    evidence=second_line,
                    polarity="negative",
                    kind="action",
                    trait_key="melon_preference",
                    statement="讨厌蜜瓜",
                    line=2,
                )
            ),
            json.dumps(
                {
                    "verdict": "contradicts",
                    "explanation": "直接偏好表达与已确认偏好相反。",
                    "citations": ["B01", "C01"],
                },
                ensure_ascii=False,
            ),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_max_completion_tokens=512,
        )

        counts = result.diagnostics["counts"]
        assert counts["targeted_pass_scheduled_count"] == 2
        assert counts["targeted_verification_scheduled_count"] == 1
        assert counts["targeted_verification_signal_added_count"] == 1
        assert counts["draft_observation_count"] == 2
        assert len(result.issues) == 1
        verification_prompt = provider.calls[3][1]
        assert f"1: {first_line}" not in verification_prompt
        assert f"2: {second_line}" in verification_prompt


def test_draft_extractor_receives_exact_confirmed_key_and_enters_drift_comparison():
    exact_key = "observant_before_speaking"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"比较键上下文-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content="林澈总是先观察现场，然后主动开口。",
            narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(
            seed,
            QueueProvider(
                _response(
                    _record(
                        evidence="林澈总是先观察现场，然后主动开口。",
                        polarity="positive",
                        kind="explicit_declaration",
                        dimension="core_personality",
                        trait_key=exact_key,
                    )
                )
            ),
        )
        _confirm_only_candidate(client, project["id"], seed)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="林澈从不观察现场，立刻抢先发言。",
            narrative_context=_context(publication="draft"),
        )

        provider = QueueProvider(
            _response(
                _record(
                    evidence="林澈总是先观察现场，然后主动开口。",
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key=exact_key,
                )
            ),
            _response(
                _record(
                    evidence="林澈从不观察现场，立刻抢先发言。",
                    polarity="negative",
                    kind="action",
                    dimension="core_personality",
                    trait_key=exact_key,
                )
            ),
            _response(),
        )
        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_max_completion_tokens=2_048,
        )

        assert result.diagnostics["counts"]["drift_considered"] == 1
        assert result.diagnostics["counts"]["context_included_trait_count"] == 1
        assert len(provider.calls) == 3
        draft_prompt = provider.calls[1][1]
        context_line = next(
            line for line in draft_prompt.splitlines() if line.startswith("服务端上下文：")
        )
        context = json.loads(context_line.removeprefix("服务端上下文："))
        assert context["confirmed_traits"] == [
            {
                "character": "林澈",
                "comparison_key": "core_personality:observantbeforespeaking",
                "dimension": "core_personality",
                "trait_key": exact_key,
            }
        ]
        assert context["confirmed_traits_coverage"]["state"] == "complete"
        assert "林澈总是先观察现场" not in draft_prompt
        counts = result.diagnostics["counts"]
        assert counts["targeted_pass_scheduled_count"] == 1
        assert counts["targeted_empty_pass_count"] == 1
        assert counts["targeted_verification_first_empty_target_count"] == 1
        assert counts["targeted_verification_no_candidate_target_count"] == 1
        assert counts["targeted_verification_scheduled_count"] == 0
        assert counts["draft_observation_count"] == 1
        assert result.diagnostics["outcome"] == "completed"
        assert result.diagnostics["material_coverage"] == "complete"
        assert result.issues == ()
        targeted_prompt = provider.calls[2][1]
        assert '"requested_polarity":"negative"' in targeted_prompt
        assert '"exclude_evidence_ranges":[{"line_start":1,"line_end":1}]' in targeted_prompt
        assert '"baseline_polarity"' not in targeted_prompt
        assert '"baseline_hint":"林澈总是先观察现场' in targeted_prompt


def test_primary_invalid_packages_fail_closed_and_mark_material_partial():
    profile_line = "林澈喜欢蜜瓜。"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"主抽取拒绝-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content=profile_line,
            narrative_context=_context(publication="published"),
        )
        response = _response(
            _record(
                evidence=profile_line,
                polarity="positive",
                kind="explicit_declaration",
            ),
            {},
        )

        result = _run_stage(
            _new_run(client, project["id"]), QueueProvider(response, response)
        )

        assert result.diagnostics["outcome"] == "degraded"
        assert result.diagnostics["material_coverage"] == "partial"
        assert result.diagnostics["reason_counts"]["schema_validation"] == 2
        assert result.diagnostics["usage"]["attempted_calls"] == 2
        assert result.diagnostics["counts"]["pending_candidate_count"] == 0


def test_primary_ignored_duplicate_is_auditable_without_marking_stage_partial():
    profile_line = "林澈喜欢蜜瓜。"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"主抽取去重-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content=profile_line,
            narrative_context=_context(publication="published"),
        )
        valid = _record(
            evidence=profile_line,
            polarity="positive",
            kind="explicit_declaration",
        )
        unsupported_duplicate = {
            **valid,
            "statement": "林澈愿意持续选择蜜瓜",
        }

        result = _run_stage(
            _new_run(client, project["id"]),
            QueueProvider(
                _response(unsupported_duplicate, valid),
                _response(valid),
            ),
        )

        assert result.diagnostics["outcome"] == "completed"
        assert result.diagnostics["material_coverage"] == "complete"
        assert (
            result.diagnostics["reason_counts"][
                "regenerated_from_statement_support"
            ]
            == 1
        )
        assert result.diagnostics["counts"]["signal_ignored_duplicate_count"] == 0
        assert result.diagnostics["counts"]["pending_candidate_count"] == 1


def test_zero_primary_observations_trigger_one_targeted_batch_and_recover_signal():
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        draft_line = "祁雾用奉承话术迂回交流。"
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content=draft_line,
            narrative_context=_context(publication="draft"),
        )
        targeted_valid = _record(
            character="祁雾",
            evidence=draft_line,
            polarity="negative",
            kind="action",
            dimension="core_personality",
            trait_key="directness",
            statement="用奉承话术迂回交流",
        )
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence="祁雾说话直来直往，这是他的核心性格。",
                    polarity="negative",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            ),
            _response(),
            _response(
                _record(
                    character="祁雾",
                    evidence=draft_line,
                    polarity="positive",
                    kind="action",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="拒绝直说计划",
                ),
                targeted_valid,
            ),
            _response(targeted_valid),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_max_completion_tokens=2_048,
        )

        counts = result.diagnostics["counts"]
        assert counts["targeted_eligible_target_count"] == 1
        assert counts["targeted_pass_scheduled_count"] == 1
        assert counts["targeted_pass_attempted_count"] == 2
        assert counts["targeted_pass_completed_count"] == 1
        assert counts["targeted_signal_added_count"] == 1
        assert counts["targeted_record_ignored_duplicate_count"] == 0
        assert counts["targeted_reviewer_reserve_tokens"] == 4_000
        assert counts["draft_observation_count"] == 1
        assert len(provider.calls) == 4
        assert sum("角色草稿覆盖复核器" in system for system, _ in provider.calls) == 2
        targeted_prompt = provider.calls[2][1]
        assert '"character":"祁雾"' in targeted_prompt
        assert '"trait_key":"directness"' in targeted_prompt
        assert "说话直来直往，这是他的核心性格" not in targeted_prompt


def test_completed_empty_targeted_pass_does_not_manufacture_observation():
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="祁雾走进会议室并关上门。",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence="祁雾说话直来直往，这是他的核心性格。",
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            ),
            _response(),
            _response(),
            _response(),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_max_completion_tokens=2_048,
        )

        counts = result.diagnostics["counts"]
        assert counts["targeted_empty_pass_count"] == 2
        assert counts["targeted_verification_scheduled_count"] == 1
        assert counts["targeted_verification_completed_count"] == 1
        assert counts["targeted_verification_empty_count"] == 1
        assert counts["targeted_signal_added_count"] == 0
        assert counts["draft_observation_count"] == 0
        assert result.diagnostics["material_coverage"] == "complete"
        assert result.diagnostics["reason_counts"][
            "targeted_no_supported_observation"
        ] == 1
        assert result.diagnostics["case_trace"][0]["final_outcome"] == "unverifiable"
        assert sum("角色草稿覆盖复核器" in system for system, _ in provider.calls) == 2


def test_targeted_pass_rejects_wrong_line_and_marks_material_partial():
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        draft_line = "祁雾直接说明了计划。"
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content=draft_line,
            narrative_context=_context(publication="draft"),
        )
        wrong_line = _record(
            character="祁雾",
            evidence=draft_line,
            polarity="negative",
            kind="action",
            dimension="core_personality",
            trait_key="directness",
            statement="直接说明了计划",
            line=99,
        )
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence="祁雾说话直来直往，这是他的核心性格。",
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            ),
            _response(),
            _response(wrong_line),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_max_completion_tokens=2_048,
        )

        counts = result.diagnostics["counts"]
        assert counts["targeted_record_rejected_count"] == 1
        assert counts["targeted_signal_added_count"] == 0
        assert result.diagnostics["material_coverage"] == "partial"
        assert result.diagnostics["reason_counts"][
            "targeted_pass_evidence_range"
        ] == 1


def test_targeted_pass_budget_rejection_is_partial_without_provider_call():
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="祁雾直接说明了计划。",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence="祁雾说话直来直往，这是他的核心性格。",
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            ),
            _response(),
        )
        with patch(
            "app.character_trait_extraction.estimate_issue_evidence_review_tokens",
            side_effect=[300, 300, 900],
        ):
            result = _run_stage(
                _new_run(client, project["id"]),
                provider,
                character_consistency_stage_token_budget=1_000,
                character_signal_token_budget=1_000,
                character_signal_max_completion_tokens=64,
            )

        counts = result.diagnostics["counts"]
        assert counts["targeted_pass_scheduled_count"] == 1
        assert counts["targeted_pass_attempted_count"] == 0
        assert counts["targeted_signal_added_count"] == 0
        assert result.diagnostics["material_coverage"] == "partial"
        assert result.diagnostics["reason_counts"][
            "targeted_reviewer_budget_reserve"
        ] == 1
        assert len(provider.calls) == 2


def test_targeted_target_capacity_truncation_is_partial_and_still_one_batch():
    profile_lines = [
        "祁雾说话直来直往，这是他的核心性格。",
        "祁雾重视同伴互动，这也是他的核心性格。",
    ]
    profile_records = [
        _record(
            character="祁雾",
            evidence=profile_lines[0],
            polarity="positive",
            kind="explicit_declaration",
            dimension="core_personality",
            trait_key="directness",
            statement="说话直来直往",
            line=1,
        ),
        _record(
            character="祁雾",
            evidence=profile_lines[1],
            polarity="positive",
            kind="explicit_declaration",
            dimension="core_personality",
            trait_key="companion_interaction_value",
            statement="重视同伴互动",
            line=2,
        ),
    ]
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"补抽容量-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content="\n".join(profile_lines),
            narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(seed, QueueProvider(_response(*profile_records)))
        assert len(_confirm_all_candidates(client, project["id"], seed)) == 2
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="祁雾走进会议室并关上门。",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(*profile_records),
            _response(),
            _response(),
            _response(),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_targeted_max_targets_per_chunk=1,
            character_signal_max_completion_tokens=2_048,
        )

        counts = result.diagnostics["counts"]
        assert counts["targeted_eligible_target_count"] == 2
        assert counts["targeted_selected_target_count"] == 1
        assert counts["targeted_truncated_target_count"] == 1
        assert counts["targeted_pass_scheduled_count"] == 2
        assert counts["targeted_verification_scheduled_count"] == 1
        assert counts["targeted_verification_empty_count"] == 1
        assert result.diagnostics["material_coverage"] == "partial"
        assert result.diagnostics["reason_counts"]["targeted_target_limit"] == 1
        assert sum("角色草稿覆盖复核器" in system for system, _ in provider.calls) == 2


def test_focused_target_empty_does_not_suppress_another_legal_target():
    draft_line = "祁雾用奉承话术迂回交流。"
    with TestClient(app) as client:
        project, profile_records = _confirmed_two_target_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content=draft_line,
            narrative_context=_context(publication="draft"),
        )
        recovered = _record(
            character="祁雾",
            evidence=draft_line,
            polarity="negative",
            kind="action",
            dimension="core_personality",
            trait_key="z_directness",
            statement="用奉承话术迂回交流",
        )
        provider = QueueProvider(
            _response(*profile_records),
            _response(),
            _response(),
            _response(recovered),
            _response(),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_max_completion_tokens=512,
        )

        counts = result.diagnostics["counts"]
        assert counts["targeted_selected_target_count"] == 2
        assert counts["targeted_pass_scheduled_count"] == 3
        assert counts["targeted_pass_attempted_count"] == 3
        assert counts["targeted_pass_completed_count"] == 3
        assert counts["targeted_empty_pass_count"] == 2
        assert counts["targeted_verification_scheduled_count"] == 1
        assert counts["targeted_verification_completed_count"] == 1
        assert counts["targeted_verification_empty_count"] == 1
        assert counts["targeted_signal_added_count"] == 1
        assert counts["configured_stage_token_budget"] == 20_000
        assert counts["remaining_run_tokens_at_stage_start"] == 20_000
        assert counts["stage_token_budget"] == 20_000
        assert result.diagnostics["material_coverage"] == "complete"
        targeted_calls = [
            user
            for system, user in provider.calls
            if "角色草稿覆盖复核器" in system
        ]
        assert len(targeted_calls) == 3
        target_payloads = [
            json.loads(
                next(
                    line for line in prompt.splitlines() if line.startswith("targets：")
                ).removeprefix("targets：")
            )
            for prompt in targeted_calls
        ]
        assert [len(payload) for payload in target_payloads] == [1, 1, 1]
        assert [payload[0]["trait_key"] for payload in target_payloads] == [
            "a_companion_interaction",
            "z_directness",
            "a_companion_interaction",
        ]


def test_focused_target_failure_is_partial_and_does_not_fabricate_signal():
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="祁雾走进会议室。",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence="祁雾说话直来直往，这是他的核心性格。",
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            ),
            _response(),
            RuntimeError("targeted provider failed"),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            character_signal_max_completion_tokens=2_048,
        )

        counts = result.diagnostics["counts"]
        assert counts["targeted_pass_scheduled_count"] == 1
        assert counts["targeted_pass_attempted_count"] == 1
        assert counts["targeted_pass_completed_count"] == 0
        assert counts["targeted_signal_added_count"] == 0
        assert result.diagnostics["outcome"] == "partial"
        assert result.diagnostics["material_coverage"] == "partial"
        assert result.diagnostics["reason_counts"][
            "targeted_pass_provider_error"
        ] == 1


def test_focused_target_budget_exhaustion_leaves_remaining_target_partial():
    with TestClient(app) as client:
        project, profile_records = _confirmed_two_target_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="祁雾走进会议室。",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(*profile_records),
            _response(),
            _response(),
        )
        with patch(
            "app.character_trait_extraction.estimate_issue_evidence_review_tokens",
            side_effect=[100, 100, 700],
        ):
            result = _run_stage(
                _new_run(client, project["id"]),
                provider,
                character_consistency_stage_token_budget=2_000,
                character_signal_max_completion_tokens=64,
                character_drift_token_budget=1_000,
                character_drift_max_completion_tokens=64,
            )

        counts = result.diagnostics["counts"]
        assert counts["targeted_pass_scheduled_count"] == 2
        assert counts["targeted_pass_attempted_count"] == 1
        assert counts["targeted_pass_completed_count"] == 1
        assert counts["targeted_budget_exhausted_target_count"] == 1
        assert counts["targeted_signal_added_count"] == 0
        assert result.diagnostics["material_coverage"] == "partial"
        assert result.diagnostics["reason_counts"][
            "targeted_reviewer_budget_reserve"
        ] == 1
        assert len(provider.calls) == 3


def test_empty_verification_budget_exhaustion_is_explicitly_partial():
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="祁雾走进会议室。",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence="祁雾说话直来直往，这是他的核心性格。",
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            ),
            _response(),
            _response(),
        )
        with patch(
            "app.character_trait_extraction.estimate_issue_evidence_review_tokens",
            side_effect=[100, 100, 700],
        ):
            result = _run_stage(
                _new_run(client, project["id"]),
                provider,
                character_consistency_stage_token_budget=2_000,
                character_signal_token_budget=1_000,
                character_signal_max_completion_tokens=64,
                character_drift_token_budget=1_000,
                character_drift_max_completion_tokens=64,
            )

        counts = result.diagnostics["counts"]
        assert counts["targeted_pass_scheduled_count"] == 2
        assert counts["targeted_pass_attempted_count"] == 1
        assert counts["targeted_verification_scheduled_count"] == 1
        assert counts["targeted_verification_attempted_count"] == 0
        assert counts["targeted_verification_budget_exhausted_count"] == 1
        assert result.diagnostics["material_coverage"] == "partial"
        assert result.diagnostics["reason_counts"][
            "targeted_verification_reviewer_budget_reserve"
        ] == 1
        assert len(provider.calls) == 3


def test_cancellation_after_initial_targets_stops_before_empty_verification():
    class TargetedRecallCancelled(RuntimeError):
        pass

    with TestClient(app) as client:
        project, profile_records = _confirmed_two_target_project(client)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="祁雾走进会议室。",
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(*profile_records),
            _response(),
            _response(),
            _response(),
        )

        def checkpoint():
            completed_target_calls = sum(
                "角色草稿覆盖复核器" in system for system, _ in provider.calls
            )
            if completed_target_calls >= 2:
                raise TargetedRecallCancelled("cancel before verification")

        with pytest.raises(TargetedRecallCancelled, match="before verification"):
            _run_stage(
                _new_run(client, project["id"]),
                provider,
                checkpoint=checkpoint,
                character_signal_max_completion_tokens=2_048,
            )

        assert sum(
            "角色草稿覆盖复核器" in system for system, _ in provider.calls
        ) == 2


def test_non_draft_chunks_never_schedule_targeted_pass():
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        provider = QueueProvider(
            _response(
                _record(
                    character="祁雾",
                    evidence="祁雾说话直来直往，这是他的核心性格。",
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                )
            )
        )

        result = _run_stage(_new_run(client, project["id"]), provider)

        counts = result.diagnostics["counts"]
        assert counts["targeted_eligible_target_count"] == 0
        assert counts["targeted_pass_scheduled_count"] == 0
        assert len(provider.calls) == 1


def test_confirmed_key_context_limit_degrades_stage_coverage_to_partial():
    characters = [f"角色{index:02}" for index in range(13)]
    profile_lines = [f"{character}喜欢蜜瓜。" for character in characters]
    profile_records = [
        _record(
            character=character,
            evidence=profile_lines[index],
            polarity="positive",
            kind="explicit_declaration",
            line=index + 1,
        )
        for index, character in enumerate(characters)
    ]
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"上下文上限-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profiles.md",
            role="character_profile",
            content="\n".join(profile_lines),
            narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(seed, QueueProvider(_response(*profile_records)))
        assert len(_confirm_all_candidates(client, project["id"], seed)) == 13
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="角色00仍然喜欢蜜瓜。",
            narrative_context=_context(publication="draft"),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            QueueProvider(
                _response(*profile_records),
                _response(
                    _record(
                        character="角色00",
                        evidence="角色00仍然喜欢蜜瓜。",
                        polarity="positive",
                        kind="preference_expression",
                    )
                ),
            ),
        )

        assert result.diagnostics["outcome"] == "partial"
        assert result.diagnostics["material_coverage"] == "partial"
        assert (
            result.diagnostics["reason_counts"][
                "confirmed_trait_context_truncated"
            ]
            == 1
        )
        counts = result.diagnostics["counts"]
        assert counts["context_eligible_trait_count"] == 13
        assert counts["context_included_trait_count"] == 12
        assert counts["context_truncated_document_count"] == 1


def test_single_personality_behavior_is_hidden_but_two_behaviors_can_need_confirmation():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"人格阶段-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content="林澈在陌生人面前从不主动交谈。",
            narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(
            seed,
            QueueProvider(
                _response(
                    _record(
                        evidence="林澈在陌生人面前从不主动交谈。",
                        polarity="negative",
                        kind="explicit_declaration",
                        dimension="core_personality",
                        trait_key="社交主动性",
                    )
                )
            ),
        )
        _confirm_only_candidate(client, project["id"], seed)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="林澈主动向陌生人问候。\n林澈主动邀请陌生人同行。",
            narrative_context=_context(publication="draft"),
        )

        one_run = _new_run(client, project["id"])
        one = _run_stage(
            one_run,
            QueueProvider(
                _response(
                    _record(
                        evidence="林澈在陌生人面前从不主动交谈。",
                        polarity="negative",
                        kind="explicit_declaration",
                        dimension="core_personality",
                        trait_key="社交主动性",
                    )
                ),
                _response(
                    _record(
                        evidence="林澈主动向陌生人问候。",
                        polarity="positive",
                        kind="action",
                        dimension="core_personality",
                        trait_key="社交主动性",
                        line=1,
                    )
                ),
            ),
        )
        assert not one.issues

        two_run = _new_run(client, project["id"])
        two = _run_stage(
            two_run,
            QueueProvider(
                _response(
                    _record(
                        evidence="林澈在陌生人面前从不主动交谈。",
                        polarity="negative",
                        kind="explicit_declaration",
                        dimension="core_personality",
                        trait_key="社交主动性",
                    )
                ),
                _response(
                    _record(
                        evidence="林澈主动向陌生人问候。",
                        polarity="positive",
                        kind="action",
                        dimension="core_personality",
                        trait_key="社交主动性",
                        line=1,
                    ),
                    _record(
                        evidence="林澈主动邀请陌生人同行。",
                        polarity="positive",
                        kind="action",
                        dimension="core_personality",
                        trait_key="社交主动性",
                        line=2,
                    ),
                ),
                json.dumps(
                    {
                        "verdict": "needs_confirmation",
                        "explanation": "两次反向表现已达到复核门槛，但材料不足以确认人格永久改变。",
                        "citations": ["B01", "C01", "C02"],
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        assert len(two.issues) == 1
        assert two.issues[0].severity.value == "medium"
        assert two.issues[0].metadata["judgement"] == "needs_confirmation"


def test_support_keyword_requires_affirmative_clause():
    assert (
        _explicit_support_kind(
            "正文没有提到林澈经历任务伪装或促使林澈改变的经历。"
        )
        is None
    )
    assert _explicit_support_kind("任务中林澈一直假装讨厌蜜瓜。") == "exception"
    assert _explicit_support_kind("训练后林澈逐渐改变了待人方式。") == "causal_bridge"


@pytest.mark.parametrize(
    "line,expected",
    [
        (
            "林澈若在潜入任务中启动镜面身份，就会假装开朗。",
            None,
        ),
        ("除非林澈完成训练，否则不能改变待人方式。", None),
        ("苏弦在临时舞台上遭遇临时停电。", None),
        ("祁雾为潜入宴会，假装成侍者并故意表现得健谈。", "exception"),
        ("苏弦受伤后暂时失忆。", "exception"),
        ("规则规定祁雾可以伪装成侍者。", None),
    ],
)
def test_support_classifier_rejects_conditions_and_non_character_temporary_nouns(
    line: str, expected: str | None
):
    assert _explicit_support_kind(line) == expected


def test_support_search_never_reuses_baseline_evidence_line():
    scope = NarrativeScopeV1()
    baseline = _confirmed_trait(dimension="core_personality", trait_key="社交主动性").model_copy(
        update={
            "evidence": (
                EvidenceSpan(
                    document_id="profile",
                    document_name="profile.md",
                    line_start=1,
                    line_end=1,
                    text="林澈在开始训练前从不主动与陌生人交谈。",
                ),
            )
        }
    )
    source = _FrozenDocument(
        input_id="input-profile",
        document=DocumentInput(
            id="profile",
            name="profile.md",
            content="林澈在开始训练前从不主动与陌生人交谈。",
            role="character_profile",
        ),
        document_version=1,
        content_sha256="0" * 64,
        ordinal=0,
        source_kind="formal_character_profile",
        source_reason="formal_character_profile",
        scope=scope,
        resolution_state="confirmed",
        publication_status="published",
        authority_tier="formal_record",
    )

    assert _find_support_evidence(
        baseline=baseline,
        baseline_scope=scope,
        draft_scopes=(scope,),
        documents=[source],
        limit=8,
    ) == ()


def test_case_trace_covers_every_bounded_baseline_without_source_text_leakage():
    profile_lines = [
        "机密原文甲：林澈从不主动与陌生人交谈。",
        "机密原文乙：苏弦喜欢蜜瓜。",
        "机密原文丙：祁雾总是保持冷静。",
    ]
    profile_records = (
        _record(
            character="林澈",
            evidence=profile_lines[0],
            statement=profile_lines[0].rstrip("。"),
            polarity="negative",
            kind="explicit_declaration",
            dimension="core_personality",
            trait_key="社交主动性",
            line=1,
        ),
        _record(
            character="苏弦",
            evidence=profile_lines[1],
            statement=profile_lines[1].rstrip("。"),
            polarity="positive",
            kind="explicit_declaration",
            dimension="preference",
            trait_key="食物偏好:蜜瓜",
            line=2,
        ),
        _record(
            character="祁雾",
            evidence=profile_lines[2],
            statement=profile_lines[2].rstrip("。"),
            polarity="positive",
            kind="explicit_declaration",
            dimension="core_personality",
            trait_key="情绪稳定性",
            line=3,
        ),
    )
    draft_lines = [
        "机密草稿丁：林澈主动向陌生人问候。",
        "机密草稿戊：苏弦仍然喜欢蜜瓜。",
    ]
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"诊断轨迹-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profiles.md",
            role="character_profile",
            content="\n".join(profile_lines),
            narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(seed, QueueProvider(_response(*profile_records)))
        assert len(_confirm_all_candidates(client, project["id"], seed)) == 3
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="\n".join(draft_lines),
            narrative_context=_context(publication="draft"),
        )

        result = _run_stage(
            _new_run(client, project["id"]),
            QueueProvider(
                _response(*profile_records),
                _response(
                    _record(
                        character="林澈",
                        evidence=draft_lines[0],
                        statement=draft_lines[0].rstrip("。"),
                        polarity="positive",
                        kind="action",
                        dimension="core_personality",
                        trait_key="社交主动性",
                        line=1,
                    ),
                    _record(
                        character="苏弦",
                        evidence=draft_lines[1],
                        statement=draft_lines[1].rstrip("。"),
                        polarity="positive",
                        kind="preference_expression",
                        dimension="preference",
                        trait_key="食物偏好:蜜瓜",
                        line=2,
                    ),
                ),
            ),
        )

        trace = result.diagnostics["case_trace"]
        assert len(trace) == 3
        by_character = {row["character_key"]: row for row in trace}
        assert by_character["林澈"] == {
            "character_key": "林澈",
            "dimension": "core_personality",
            "comparison_key": "core_personality:社交主动性",
            "matched_observation_count": 1,
            "prepare_reason": "single_behavior_is_not_drift",
            "review_outcome": "not_run",
            "review_verdict": None,
            "citation_roles": [],
            "final_outcome": "needs_confirmation",
            "visible": False,
            "promote_reason": "single_behavior_is_not_drift",
        }
        assert by_character["苏弦"]["prepare_reason"] == "no_opposition"
        assert by_character["苏弦"]["final_outcome"] == "no_issue"
        assert by_character["祁雾"]["matched_observation_count"] == 0
        assert by_character["祁雾"]["prepare_reason"] == "no_matching_observation"
        assert by_character["祁雾"]["final_outcome"] == "unverifiable"
        serialized = json.dumps(trace, ensure_ascii=False)
        assert "evidence" not in serialized
        assert "statement" not in serialized
        assert "document" not in serialized
        for marker in ("机密原文甲", "机密原文乙", "机密原文丙", "机密草稿丁", "机密草稿戊"):
            assert marker not in serialized


def test_case_trace_redacts_url_and_credential_shaped_identifiers():
    secret = "sk-1234567890abcdef"
    trace = _safe_case_trace(
        character_key="https://private.invalid/character",
        baseline=_confirmed_trait().model_copy(
            update={"trait_key": f"api_key:{secret}"}
        ),
        matched_observation_count=0,
        prepare_reason="no_matching_observation",
        review=None,
        final_outcome="unverifiable",
        visible=False,
        promote_reason="no_matching_observation",
    )

    serialized = json.dumps(trace, ensure_ascii=False)
    assert "https://" not in serialized
    assert secret not in serialized
    assert trace["character_key"].startswith("redacted_")
    assert trace["comparison_key"].startswith("redacted_")


def _confirmed_trait(
    *,
    authority: str = "formal_record",
    valid_from: int | None = None,
    valid_until: int | None = None,
    character: str = "林澈",
    dimension: str = "preference",
    trait_key: str = "食物偏好:蜜瓜",
) -> ConfirmedTraitSnapshot:
    return ConfirmedTraitSnapshot(
        id=f"ct_{authority}_{sum(ord(char) for char in trait_key)}",
        character=character,
        dimension=dimension,
        trait_key=trait_key,
        statement="林澈喜欢蜜瓜",
        polarity="positive",
        stability="stable",
        origin="explicit_setting",
        authority_tier=authority,
        valid_from_release_ordinal=valid_from,
        valid_until_release_ordinal=valid_until,
        evidence=(
            EvidenceSpan(
                document_id="profile",
                document_name="profile.md",
                line_start=1,
                line_end=1,
                text="林澈喜欢蜜瓜。",
            ),
        ),
    )


def test_safe_server_context_is_bounded_content_free_and_marks_truncation():
    scope = NarrativeScopeV1()
    source = _FrozenDocument(
        input_id="input-draft",
        document=DocumentInput(
            id="draft",
            name="draft.md",
            content="角色00改变了行为。",
            role="chapter",
        ),
        document_version=1,
        content_sha256="0" * 64,
        ordinal=0,
        source_kind="draft",
        source_reason="draft",
        scope=scope,
        resolution_state="confirmed",
        publication_status="draft",
        authority_tier="draft",
    )
    entries = [
        (
            SimpleNamespace(candidate_id=f"candidate-{index:02}"),
            _confirmed_trait(
                character=f"角色{index:02}",
                trait_key=f"trait_{index:02}",
            ),
            scope,
            f"角色{index:02}",
        )
        for index in range(13)
    ]
    entries.append(
        (
            SimpleNamespace(candidate_id="candidate-secret"),
            _confirmed_trait(
                character="泄露测试",
                trait_key="https://example.invalid/?api_key=credential-placeholder",
            ),
            scope,
            "泄露测试",
        )
    )

    result = _safe_server_context(source, baselines=entries)
    payload = json.loads(result.payload)

    assert len(result.payload) <= MAX_CHARACTER_SIGNAL_SERVER_CONTEXT_CHARS
    assert result.eligible_traits == 14
    assert result.included_traits == 12
    assert result.truncated
    assert payload["confirmed_traits_coverage"] == {
        "state": "partial",
        "included": 12,
        "eligible": 14,
    }
    assert len(payload["confirmed_traits"]) == 12
    assert "https://" not in result.payload
    assert "api_key" not in result.payload
    assert "credential-placeholder" not in result.payload
    assert "林澈喜欢蜜瓜" not in result.payload
    assert "baseline_hint" not in result.payload
    assert "profile.md" not in result.payload
    assert len(result.targets) == 12
    assert all(target.baseline_hint == "林澈喜欢蜜瓜" for target in result.targets)


def test_baseline_hint_sanitizes_controls_and_marks_bounded_truncation():
    raw = "林澈喜欢\n蜜瓜\x00" + "很" * 500
    hint = _safe_baseline_hint(raw)

    assert len(hint) <= MAX_CHARACTER_SIGNAL_BASELINE_HINT_CHARS
    assert "\n" not in hint
    assert "\x00" not in hint
    assert hint.endswith("[语义提示已截断]")


def test_baseline_hint_sanitizes_unicode_separators_and_surrogates():
    hint = _safe_baseline_hint("前半\u2028后半\ud800结尾")

    assert hint == "前半 后半 结尾"
    assert hint.encode("utf-8")


def test_safe_server_context_only_exposes_scope_and_release_valid_keys():
    target_scope = NarrativeScopeV1.model_validate(
        {
            "timeline_key": "main",
            "branch": {"path": ["main", "route-a"], "exclusive_group": "route"},
            "release": {"key": "v5", "ordinal": 5},
        }
    )
    sibling_scope = NarrativeScopeV1.model_validate(
        {
            "timeline_key": "main",
            "branch": {"path": ["main", "route-b"], "exclusive_group": "route"},
            "release": {"key": "v5", "ordinal": 5},
        }
    )
    source = _FrozenDocument(
        input_id="input-draft",
        document=DocumentInput("draft", "draft.md", "林澈改变了行为。", "chapter"),
        document_version=1,
        content_sha256="0" * 64,
        ordinal=0,
        source_kind="draft",
        source_reason="draft",
        scope=target_scope,
        resolution_state="confirmed",
        publication_status="draft",
        authority_tier="draft",
    )
    entries = [
        (
            SimpleNamespace(candidate_id="active"),
            _confirmed_trait(
                trait_key="active_key",
                valid_from=1,
                valid_until=10,
            ),
            target_scope,
            "林澈",
        ),
        (
            SimpleNamespace(candidate_id="expired"),
            _confirmed_trait(
                trait_key="expired_key",
                valid_from=1,
                valid_until=4,
            ),
            target_scope,
            "林澈",
        ),
        (
            SimpleNamespace(candidate_id="sibling"),
            _confirmed_trait(
                trait_key="sibling_key",
                valid_from=1,
                valid_until=10,
            ),
            sibling_scope,
            "林澈",
        ),
    ]

    payload = json.loads(_safe_server_context(source, baselines=entries).payload)

    assert [row["trait_key"] for row in payload["confirmed_traits"]] == [
        "active_key"
    ]
    assert payload["confirmed_traits_coverage"]["state"] == "complete"


def test_release_bounds_and_authority_are_enforced_before_drift_review():
    bounded = _confirmed_trait(valid_from=12, valid_until=20)
    assert _trait_applies_to_release(
        bounded,
        NarrativeScopeV1.model_validate(
            {"timeline_key": "main", "release": {"key": "v1.1", "ordinal": 11}}
        ),
    ) is False
    assert _trait_applies_to_release(
        bounded,
        NarrativeScopeV1.model_validate(
            {"timeline_key": "main", "release": {"key": "v1.2", "ordinal": 12}}
        ),
    ) is True
    assert _trait_applies_to_release(bounded, NarrativeScopeV1()) is None

    scope = NarrativeScopeV1()
    formal = (
        SimpleNamespace(candidate_id="formal"),
        _confirmed_trait(authority="formal_record"),
        scope,
        "林澈",
    )
    canon = (
        SimpleNamespace(candidate_id="canon"),
        _confirmed_trait(authority="core_canon"),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([formal, canon])
    assert shadowed == 1
    assert [row[1].authority_tier for row in selected] == ["core_canon"]


def test_snapshot_uuid_and_history_origin_are_explicitly_adapted_and_unknown_fails_closed():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"快照映射-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content="林澈喜欢蜜瓜。",
            narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(
            seed,
            QueueProvider(
                _response(
                    _record(
                        evidence="林澈喜欢蜜瓜。",
                        polarity="positive",
                        kind="explicit_declaration",
                    )
                )
            ),
        )
        candidate_id = _confirm_only_candidate(client, project["id"], seed)
        frozen = _new_run(client, project["id"])
        with SessionLocal() as db:
            row = db.scalar(
                select(AnalysisRunCharacterTraitInputRow).where(
                    AnalysisRunCharacterTraitInputRow.run_id == frozen
                )
            )
            assert row is not None
            payload = dict(row.payload)
            payload["origin"] = "history_inference"
            row.payload = payload
            row.payload_sha256 = payload_sha256(payload)
            db.commit()

        with SessionLocal() as db:
            stage = CharacterConsistencyStage(settings=_settings())
            reasons: Counter[str] = Counter()
            mapped = stage._load_confirmed_traits(
                db,
                run_id=frozen,
                project_id=project["id"],
                reasons=reasons,
            )
            assert len(mapped) == 1
            _, baseline, _, _ = mapped[0]
            assert baseline.id == f"ct_{candidate_id}"
            assert baseline.character == "林澈"
            assert baseline.dimension == "preference"
            assert baseline.statement == "林澈喜欢蜜瓜"
            assert baseline.origin == "confirmed_history_inference"

            row = db.scalar(
                select(AnalysisRunCharacterTraitInputRow).where(
                    AnalysisRunCharacterTraitInputRow.run_id == frozen
                )
            )
            payload = dict(row.payload)
            payload["origin"] = "model_claimed_global"
            row.payload = payload
            row.payload_sha256 = payload_sha256(payload)
            db.commit()

        with SessionLocal() as db:
            reasons = Counter()
            assert stage._load_confirmed_traits(
                db,
                run_id=frozen,
                project_id=project["id"],
                reasons=reasons,
            ) == []
            assert reasons["invalid_confirmed_trait_snapshot"] == 1


def test_sibling_branch_scope_does_not_emit_character_issue():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"分支隔离-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content="林澈喜欢蜜瓜。",
            narrative_context=_context(
                publication="published",
                branch="route-a",
                exclusive_group="route",
            ),
        )
        seed = _new_run(client, project["id"])
        _run_stage(
            seed,
            QueueProvider(
                _response(
                    _record(
                        evidence="林澈喜欢蜜瓜。",
                        polarity="positive",
                        kind="explicit_declaration",
                    )
                )
            ),
        )
        _confirm_only_candidate(client, project["id"], seed)
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="林澈明确说自己讨厌蜜瓜。",
            narrative_context=_context(
                publication="draft",
                branch="route-b",
                exclusive_group="route",
            ),
        )
        run_id = _new_run(client, project["id"])
        result = _run_stage(
            run_id,
            QueueProvider(
                _response(
                    _record(
                        evidence="林澈喜欢蜜瓜。",
                        polarity="positive",
                        kind="explicit_declaration",
                    )
                ),
                _response(
                    _record(
                        evidence="林澈明确说自己讨厌蜜瓜。",
                        polarity="negative",
                        kind="explicit_declaration",
                    )
                ),
            ),
        )
        assert not result.issues
        assert result.diagnostics["reason_counts"]["drift_scope_incompatible"] == 1


def test_provider_failure_degrades_optional_stage_without_candidates_or_issues():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"模型降级-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content="林澈喜欢蜜瓜。",
            narrative_context=_context(publication="published"),
        )
        run_id = _new_run(client, project["id"])
        result = _run_stage(run_id, QueueProvider(RuntimeError("upstream failed")))
        assert result.issues == ()
        assert result.diagnostics["outcome"] == "degraded"
        assert result.diagnostics["reason_code"] == "model_stage_unavailable"
        with SessionLocal() as db:
            assert not list(
                db.scalars(
                    select(CharacterTraitCandidateRow).where(
                        CharacterTraitCandidateRow.source_run_id == run_id
                    )
                ).all()
            )


def test_empty_model_extraction_is_explicitly_partial_not_clean():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"空抽取-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="林澈在市集与陌生人交谈。",
            narrative_context=_context(publication="draft"),
        )
        run_id = _new_run(client, project["id"])
        result = _run_stage(run_id, QueueProvider(_response()))
        assert result.issues == ()
        assert result.diagnostics["outcome"] == "partial"
        assert result.diagnostics["material_coverage"] == "partial"
        assert result.diagnostics["reason_counts"]["no_draft_signals"] == 1


def test_service_internal_character_stage_failure_still_completes_baseline_run():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"服务降级-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content="普通章节正文。",
            narrative_context=_context(publication="draft"),
        )
        with patch("app.main.dispatch_analysis"):
            response = client.post(
                f"/api/v1/projects/{project['id']}/analysis-runs"
            )
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]
        original = app_settings.enable_character_consistency
        app_settings.enable_character_consistency = True
        try:
            with patch(
                "app.service.CharacterConsistencyStage.run",
                side_effect=RuntimeError("stage integration defect"),
            ):
                execute_analysis(run_id, raise_on_failure=True)
        finally:
            app_settings.enable_character_consistency = original
        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            assert run is not None
            assert run.status == "completed"
        diagnostics = client.get(
            f"/api/v1/analysis-runs/{run_id}/diagnostics"
        ).json()
        assert diagnostics["character_consistency"]["outcome"] == "degraded"
        assert (
            diagnostics["character_consistency"]["reason_code"]
            == "internal_failure"
        )
