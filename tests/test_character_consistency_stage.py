from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.character_consistency_stage import (
    CharacterConsistencyStage,
    failed_character_consistency_stage,
    _FrozenDocument,
    _classify_frozen_source,
    _baseline_shadowed_at_scope,
    _observation_matches_baseline,
    _explicit_support_kind,
    _find_support_evidence,
    _safe_case_trace,
    _safe_accepted_draft_observation_refs,
    _safe_baseline_hint,
    _safe_server_context,
    _select_authoritative_baselines,
    _signal_matches_target,
    _target_has_sufficient_recall_evidence,
    _target_with_existing_evidence_ranges,
    _targeted_completion_reserve,
    _trusted_axis_binding,
    _trait_applies_to_release,
)
from app.character_drift import (
    CHARACTER_REVIEW_SYSTEM_PROMPT,
    CharacterDriftCase,
    CharacterReviewDiagnostics,
    CharacterReviewResult,
    ConfirmedTraitSnapshot,
    ModelDriftDecision,
    SupportEvidence,
    prepare_character_drift,
)
from app.character_trait_extraction import (
    MAX_CHARACTER_SIGNAL_BASELINE_HINT_CHARS,
    MAX_CHARACTER_SIGNAL_SERVER_CONTEXT_CHARS,
    CharacterSignal,
    CharacterSignalChunk,
    CharacterSignalDiagnostics,
    CharacterSignalExtractionResult,
    CharacterSignalTarget,
    SupportTraceAttemptV1,
    SupportTraceEventV1,
    SupportTraceV1,
    _targeted_chunk_prompt,
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


def test_empty_character_stage_has_empty_mismatch_diagnostics():
    diagnostics = failed_character_consistency_stage().diagnostics
    assert diagnostics["evidence_mismatch_counts"] == {}
    assert diagnostics["evidence_mismatch_chunks"] == []
    assert diagnostics["evidence_mismatch_chunks_omitted_count"] == 0
    assert diagnostics["core_label_scope_counts"] == {}
    assert diagnostics["accepted_model_core_without_literal_label_count"] == 0


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


def test_targeted_mismatch_diagnostics_include_recall_and_verification_phases():
    antecedent = "这里只有祁雾。"
    observation = "她用奉承话术迂回交流。"
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        _create_document(
            client, project["id"], name="private-draft.md", role="chapter",
            content=f"{antecedent}\n{observation}",
            narrative_context=_context(publication="draft"),
        )
        profile_line = "祁雾说话直来直往，这是他的核心性格。"
        profile_record = _record(
            character="祁雾", evidence=profile_line, polarity="positive",
            kind="explicit_declaration", dimension="core_personality",
            trait_key="directness", statement="说话直来直往",
        )
        mismatch = _record(
            character="祁雾", evidence=antecedent, polarity="negative",
            kind="action", dimension="core_personality",
            trait_key="directness", statement="祁雾用奉承话术迂回交流",
        )
        mismatch["source_line_end"] = 2
        format_mismatch = {
            **mismatch,
            "evidence": f"{antecedent}\n{observation.replace('。', '！')}",
        }
        provider = QueueProvider(
            _response(profile_record), _response(),
            _response(mismatch), _response(),
            _response(format_mismatch), _response(),
        )
        result = _run_stage(
            _new_run(client, project["id"]), provider,
            character_signal_max_completion_tokens=512,
        )

    diagnostics = result.diagnostics
    assert diagnostics["counts"]["targeted_verification_scheduled_count"] == 1
    assert diagnostics["evidence_mismatch_counts"] == {
        "presentation_difference": 1, "multiline_omission": 1,
    }
    events = diagnostics["evidence_mismatch_chunks"]
    assert len(events) == 2
    assert [event["phase"] for event in events] == [
        "targeted_recall", "targeted_verification",
    ]
    assert [event["counts"] for event in events] == [
        {"multiline_omission": 1}, {"presentation_difference": 1},
    ]
    assert all(event["source_document_ordinal"] == 1 for event in events)
    assert all(event["document_chunk_ordinal"] == 1 for event in events)
    assert all(event["stage_chunk_ordinal"] == 2 for event in events)
    assert all(event["target_ordinal"] == 1 for event in events)
    assert all(event["document_role"] == "chapter" for event in events)
    assert all(event["source_kind"] == "draft" for event in events)
    assert all(event["outcome"] == "completed" for event in events)
    assert diagnostics["evidence_mismatch_chunks_omitted_count"] == 0
    assert "private-draft.md" not in json.dumps(events, ensure_ascii=False)


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


@pytest.mark.parametrize(
    ("role", "publication", "resolution", "expected_kind", "expected_reason"),
    [
        ("chapter", "draft", "confirmed", "draft", "draft"),
        ("chapter", "in_review", "confirmed", "draft", "draft"),
        ("chapter", "published", "confirmed", "published_history", "history"),
        ("chapter", "retired", "confirmed", "published_history", "history"),
        ("chapter", "unknown", "confirmed", None, "reference"),
        ("reference", "in_review", "confirmed", None, "reference"),
        ("chapter", "in_review", "inferred", None, "inferred"),
    ],
)
def test_frozen_source_classification_includes_reviewing_chapters_only_at_confirmed_scope(
    role: str,
    publication: str,
    resolution: str,
    expected_kind: str | None,
    expected_reason: str,
):
    context = _context(publication=publication, branch="route-a", exclusive_group="routes")
    context["resolution_state"] = resolution
    kind, reason, scope, actual_resolution, actual_publication, _ = (
        _classify_frozen_source({"document_role": role}, context)
    )
    assert (kind, reason, actual_resolution, actual_publication) == (
        expected_kind, expected_reason, resolution, publication,
    )
    assert scope is not None
    assert scope.branch is not None
    assert scope.branch.path == ["main", "route-a"]


def test_in_review_target_survives_api_selection_and_frozen_stage_binding_without_model():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"审阅中文稿目标-{uuid4().hex}"},
        ).json()
        target = _create_document(
            client, project["id"], name="reviewing.md", role="chapter",
            content="林澈在演讲前突然主动和陌生人攀谈。",
            narrative_context=_context(
                publication="in_review", branch="route-a", exclusive_group="routes"
            ),
        )
        incompatible_history = _create_document(
            client, project["id"], name="other-route.md", role="chapter",
            content="另一条分支的历史章节。",
            narrative_context=_context(
                publication="published", branch="route-b", exclusive_group="routes"
            ),
        )
        with patch("app.main.dispatch_analysis"):
            created = client.post(
                f"/api/v1/projects/{project['id']}/analysis-runs",
                json={"mode": "draft_review", "target_document_ids": [target["id"]]},
            )
        assert created.status_code == 202, created.text
        with SessionLocal() as db:
            documents, metadata = _load_verified_snapshot(db, created.json()["id"])
            assert [row.id for row in documents] == [target["id"]]
            assert incompatible_history["id"] not in [row.id for row in documents]
            reasons: Counter[str] = Counter()
            frozen = CharacterConsistencyStage(settings=_settings())._bind_frozen_documents(
                db,
                run_id=created.json()["id"],
                documents=documents,
                metadata=metadata,
                reasons=reasons,
            )
        assert len(frozen) == 1
        assert frozen[0].source_kind == "draft"
        assert frozen[0].publication_status == "in_review"
        assert reasons["source_draft"] == 1


def _run_stage(
    run_id: str,
    provider: QueueProvider,
    *,
    checkpoint=None,
    remaining_run_tokens: int = 20_000,
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
            remaining_run_tokens=remaining_run_tokens,
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


@pytest.mark.parametrize("recover", (False, True))
def test_stage_core_label_scope_counts_survive_safe_serialization(recover: bool):
    line = "甲的核心性格是谨慎核对。甲喜欢热茶。"
    unrelated = _record(
        character="甲", evidence=line, polarity="positive",
        kind="explicit_declaration", dimension="core_personality",
        trait_key="tea_preference", statement="甲喜欢热茶",
    )
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"作用域诊断-{uuid4().hex}"}
        ).json()
        _create_document(
            client, project["id"], name="profile.md", role="character_profile",
            content=line, narrative_context=_context(publication="published"),
        )
        result = _run_stage(
            _new_run(client, project["id"]),
            QueueProvider(
                _response(unrelated),
                _response() if recover else _response(unrelated),
            ),
        )

    diagnostics = json.loads(json.dumps(result.diagnostics, ensure_ascii=False))
    assert diagnostics["reason_counts"]["source_formal"] == 1
    assert diagnostics["reason_counts"].get("core_label_scope", 0) == (
        0 if recover else 2
    )
    assert diagnostics["reason_counts"].get(
        "regenerated_from_core_label_scope", 0
    ) == (1 if recover else 0)
    assert diagnostics["core_label_scope_counts"] == {
        "selected_other_assertion": 1 if recover else 2
    }
    assert sum(diagnostics["core_label_scope_counts"].values()) == sum(
        diagnostics["reason_counts"].get(key, 0)
        for key in ("core_label_scope", "regenerated_from_core_label_scope")
    )
    assert diagnostics["accepted_model_core_without_literal_label_count"] == 0


def test_stage_unlabeled_core_observation_uses_final_deduplicated_signals():
    line = "甲始终谨慎核对记录。"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"去重诊断-{uuid4().hex}"}
        ).json()
        first = _create_document(
            client, project["id"], name="profile-a.md", role="character_profile",
            content=line, narrative_context=_context(publication="published"),
        )
        _create_document(
            client, project["id"], name="profile-b.md", role="character_profile",
            content=line, narrative_context=_context(publication="published"),
        )
        signal = CharacterSignal(
            id="cs_" + "a" * 32, character="甲",
            dimension="core_personality", trait_key="record_verification",
            statement="甲始终谨慎核对记录", polarity="positive",
            stability="core", observation_kind="explicit_declaration",
            source_kind="formal_character_profile",
            evidence=EvidenceSpan(
                document_id=first["id"], document_name="profile-a.md",
                line_start=1, line_end=1, text=line,
            ),
        )
        per_chunk = CharacterSignalExtractionResult(
            signals=(signal,),
            diagnostics=CharacterSignalDiagnostics(
                outcome="completed", attempted_calls=1,
                raw_records=1, accepted_records=1, rejected_records=0,
                accepted_model_core_without_literal_label_count=1,
            ),
        )
        with patch(
            "app.character_consistency_stage.CharacterSignalExtractor.extract",
            return_value=per_chunk,
        ) as mock_extract:
            result = _run_stage(
                _new_run(client, project["id"]), QueueProvider(),
            )

    assert mock_extract.call_count == 2
    assert result.diagnostics["counts"]["signal_count"] == 1
    assert result.diagnostics["accepted_model_core_without_literal_label_count"] == 1
    assert result.diagnostics["accepted_signal_histogram"] == [{
        "source_kind": "formal_character_profile", "stability": "core",
        "dimension": "core_personality", "count": 1,
    }]


@pytest.mark.parametrize("event_limit", (128, 1))
def test_mismatch_stage_diagnostics_keep_role_order_and_bound_details(
    monkeypatch, event_limit: int,
):
    monkeypatch.setattr(
        "app.character_consistency_stage._MAX_EVIDENCE_MISMATCH_CHUNKS",
        event_limit,
    )
    canon_line = "林澈一直喜欢蜜瓜。"
    history_line = "林澈每周都买一颗蜜瓜。"
    canon_rejected = _record(
        evidence="一直喜欢蜜瓜", polarity="positive",
        kind="explicit_declaration", statement="林澈一直喜欢蜜瓜",
    )
    history_valid = _record(
        evidence=history_line, polarity="positive", kind="action",
        statement="林澈每周都买一颗蜜瓜",
    )
    history_rejected = {
        **history_valid,
        "evidence": history_line.replace("。", "！"),
    }
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"形态诊断-{uuid4().hex}"}
        ).json()
        canon = _create_document(
            client, project["id"], name="private-canon.md", role="canon",
            content=canon_line, narrative_context=_context(publication="published"),
        )
        history = _create_document(
            client, project["id"], name="private-history.md", role="chapter",
            content=history_line, narrative_context=_context(publication="published"),
        )
        result = _run_stage(
            _new_run(client, project["id"]),
            QueueProvider(
                _response(canon_rejected), _response(canon_rejected),
                _response(history_rejected), _response(history_valid),
            ),
            character_signal_max_completion_tokens=1_024,
        )

    diagnostics = result.diagnostics
    assert diagnostics["outcome"] == "partial"
    assert diagnostics["reason_counts"]["evidence_mismatch"] == 2
    assert diagnostics["reason_counts"]["regenerated_from_evidence_mismatch"] == 1
    assert diagnostics["evidence_mismatch_counts"] == {
        "presentation_difference": 1, "source_excerpt": 2,
    }
    assert diagnostics["evidence_mismatch_chunks_omitted_count"] == 2 - min(event_limit, 2)
    events = diagnostics["evidence_mismatch_chunks"]
    assert len(events) == min(event_limit, 2)
    assert events[0] == {
        "source_document_ordinal": 0,
        "document_chunk_ordinal": 1,
        "stage_chunk_ordinal": 1,
        "document_role": "canon",
        "source_kind": "formal_character_profile",
        "phase": "primary_extraction",
        "target_ordinal": None,
        "outcome": "degraded",
        "counts": {"source_excerpt": 2},
    }
    if event_limit > 1:
        assert events[1] == {
            "source_document_ordinal": 1,
            "document_chunk_ordinal": 1,
            "stage_chunk_ordinal": 2,
            "document_role": "chapter",
            "source_kind": "published_history",
            "phase": "primary_extraction",
            "target_ordinal": None,
            "outcome": "completed",
            "counts": {"presentation_difference": 1},
        }
    serialized = json.dumps(events, ensure_ascii=False)
    for private in (canon_line, history_line, canon["id"], history["id"],
                    "private-canon.md", "private-history.md"):
        assert private not in serialized


def test_same_line_distinct_preference_objects_reach_stage_candidates():
    profile_line = "林澈喜欢蜜瓜，也喜欢葡萄。"
    melon = _record(
        evidence=profile_line,
        polarity="positive",
        kind="explicit_declaration",
        trait_key="food_preference",
        statement="林澈喜欢蜜瓜",
    )
    grape = {
        **_record(
            evidence=profile_line,
            polarity="positive",
            kind="explicit_declaration",
            trait_key="food_preference",
            statement="林澈喜欢葡萄",
        ),
        "key_object": "葡萄",
    }
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"同线不同对象-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content=profile_line,
            narrative_context=_context(publication="published"),
        )
        run_id = _new_run(client, project["id"])
        result = _run_stage(run_id, QueueProvider(_response(melon, grape)))

        with SessionLocal() as db:
            candidates = list(
                db.scalars(
                    select(CharacterTraitCandidateRow).where(
                        CharacterTraitCandidateRow.source_run_id == run_id
                    )
                ).all()
            )

    assert result.diagnostics["outcome"] == "completed"
    assert result.diagnostics["counts"]["signal_count"] == 2
    assert result.diagnostics["counts"]["pending_candidate_count"] == 2
    assert {candidate.comparison_key for candidate in candidates} == {
        "preference:蜜瓜",
        "preference:葡萄",
    }


def test_v4_same_line_support_ids_survive_stage_without_double_counting_line_evidence():
    profile_line = "林澈一直喜欢蜜瓜，林澈长期喜欢蜜瓜。"
    first = {
        **_record(
            evidence=profile_line, polarity="positive", kind="explicit_declaration",
            trait_key="melon_preference", statement="林澈一直喜欢蜜瓜",
        ),
        "support_id": "L1:A1",
    }
    second = {
        **_record(
            evidence=profile_line, polarity="positive", kind="explicit_declaration",
            trait_key="melon_preference", statement="林澈长期喜欢蜜瓜",
        ),
        "support_id": "L1:A2",
    }
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"子句身份-{uuid4().hex}"}
        ).json()
        _create_document(
            client, project["id"], name="profile.md", role="character_profile",
            content=profile_line,
            narrative_context=_context(publication="published"),
        )
        run_id = _new_run(client, project["id"])
        result = _run_stage(
            run_id,
            QueueProvider(_response(first, second)),
            character_signal_full_line_prompt_v2=True,
            character_signal_support_id_v4=True,
        )
        with SessionLocal() as db:
            candidates = list(db.scalars(
                select(CharacterTraitCandidateRow).where(
                    CharacterTraitCandidateRow.source_run_id == run_id
                )
            ).all())

    assert result.diagnostics["outcome"] == "completed"
    assert result.diagnostics["counts"]["signal_count"] == 2
    assert result.diagnostics["counts"]["pending_candidate_count"] == 1
    assert len(candidates) == 1
    assert len(candidates[0].evidence) == 1


@pytest.mark.parametrize("trace_enabled", (False, True))
def test_stage_support_trace_marks_missing_formal_trace_unavailable(
    monkeypatch, trace_enabled: bool,
):
    monkeypatch.setattr(
        "app.character_consistency_stage._MAX_SUPPORT_TRACE_CHUNKS", 1
    )
    private_lines = ("甲始终喜欢热茶。", "乙始终喜欢梨汤。")
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"匿名断言诊断-{uuid4().hex}"}
        ).json()
        for index, line in enumerate(private_lines):
            _create_document(
                client, project["id"], name=f"private-{index}.md",
                role="character_profile", content=line,
                narrative_context=_context(publication="published"),
            )
        diagnostic = CharacterSignalDiagnostics(
            outcome="skipped", attempted_calls=0, raw_records=0,
            accepted_records=0, rejected_records=0,
        )
        mocked_diagnostic = SimpleNamespace(
            **diagnostic.model_dump(), support_trace=None,
        )
        mocked_extraction = SimpleNamespace(
            signals=(), diagnostics=mocked_diagnostic,
        )
        with patch(
            "app.character_consistency_stage.CharacterSignalExtractor.extract",
            return_value=mocked_extraction,
        ) as mock_extract:
            result = _run_stage(
                _new_run(client, project["id"]), QueueProvider(),
                character_signal_full_line_prompt_v2=True,
                character_signal_support_id_v4=True,
                character_signal_support_trace_v1=trace_enabled,
            )

    assert mock_extract.call_count == 2
    diagnostics = result.diagnostics
    if not trace_enabled:
        assert "support_trace_chunks" not in diagnostics
        assert "support_trace_chunks_omitted_count" not in diagnostics
        return
    assert diagnostics["support_trace_chunks"] == [
        {
            "stage_chunk_ordinal": 1,
            "outcome": "skipped",
            "availability": "unavailable",
            "trace": None,
        }
    ]
    assert diagnostics["support_trace_chunks_omitted_count"] == 1
    serialized = json.dumps(diagnostics["support_trace_chunks"], ensure_ascii=False)
    assert all(value not in serialized for value in (
        *private_lines, "private-0.md", "private-1.md", "L1:A1",
    ))


def test_stage_support_trace_exports_validated_formal_chunk_only():
    profile_line = "甲长期喜欢热茶，甲始终喜欢梨汤。"
    draft_line = "甲走进房间。"
    trace = SupportTraceV1(
        indexed_support_count=2,
        attempts=(SupportTraceAttemptV1(
            attempt=1, observability="parsed", submitted_slots=(2,),
            events=(SupportTraceEventV1(slot=2, outcome="accepted"),),
            unbound_record_events=0,
        ),),
        final_state="clean", final_accepted_slots=(2,),
    )
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"断言诊断关联-{uuid4().hex}"}
        ).json()
        _create_document(
            client, project["id"], name="private-profile.md",
            role="character_profile", content=profile_line,
            narrative_context=_context(publication="published"),
        )
        _create_document(
            client, project["id"], name="private-draft.md", role="chapter",
            content=draft_line,
            narrative_context=_context(publication="draft"),
        )
        diagnostics = CharacterSignalDiagnostics(
            outcome="completed", attempted_calls=1, raw_records=1,
            accepted_records=1, rejected_records=0, support_trace=trace,
        )
        extraction = CharacterSignalExtractionResult(diagnostics=diagnostics)
        with patch(
            "app.character_consistency_stage.CharacterSignalExtractor.extract",
            return_value=extraction,
        ) as mock_extract:
            result = _run_stage(
                _new_run(client, project["id"]), QueueProvider(),
                character_signal_full_line_prompt_v2=True,
                character_signal_support_id_v4=True,
                character_signal_support_trace_v1=True,
            )

    assert mock_extract.call_count == 2
    assert result.diagnostics["support_trace_chunks"] == [
        {
            "stage_chunk_ordinal": 1,
            "outcome": "completed",
            "availability": "available",
            "trace": trace.model_dump(mode="json"),
        }
    ]
    assert result.diagnostics["support_trace_chunks_omitted_count"] == 0
    serialized = json.dumps(result.diagnostics["support_trace_chunks"], ensure_ascii=False)
    assert all(value not in serialized for value in (
        profile_line, draft_line, "private-profile.md", "private-draft.md",
        "L1:A1", "L1:A2",
    ))


def test_accepted_signal_diagnostics_are_content_free_and_explain_history_singleton():
    profile_line = "林澈长期喜欢蜜瓜。"
    history_line = "林澈每次回城都喝梨汤。"
    profile_record = _record(
        evidence=profile_line,
        polarity="positive",
        kind="explicit_declaration",
        statement="林澈长期喜欢蜜瓜",
    )
    history_record = {
        **_record(
            evidence=history_line,
            polarity="positive",
            kind="action",
            trait_key="drink_preference",
            statement="林澈每次回城都喝梨汤",
        ),
        "key_object": "梨汤",
    }
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"无内容诊断-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content=profile_line,
            narrative_context=_context(publication="published"),
        )
        _create_document(
            client,
            project["id"],
            name="history.md",
            role="chapter",
            content=history_line,
            narrative_context=_context(publication="published"),
        )
        result = _run_stage(
            _new_run(client, project["id"]),
            QueueProvider(_response(profile_record), _response(history_record)),
        )

    diagnostics = result.diagnostics
    assert diagnostics["counts"]["signal_count"] == 2
    assert diagnostics["counts"]["pending_candidate_count"] == 1
    histogram = diagnostics["accepted_signal_histogram"]
    assert histogram == [
        {
            "source_kind": "formal_character_profile",
            "stability": "stable",
            "dimension": "preference",
            "count": 1,
        },
        {
            "source_kind": "published_history",
            "stability": "stable",
            "dimension": "preference",
            "count": 1,
        },
    ]
    assert sum(bucket["count"] for bucket in histogram) == diagnostics["counts"][
        "signal_count"
    ]
    assert diagnostics["candidate_eligibility"] == {
        "stable_or_core_formal_signals": 1,
        "stable_or_core_history_signals": 1,
        "prelimit_candidates": 1,
    }
    aggregate = json.dumps(
        {
            "accepted_signal_histogram": histogram,
            "candidate_eligibility": diagnostics["candidate_eligibility"],
        },
        ensure_ascii=False,
    )
    assert all(
        content not in aggregate
        for content in ("林澈", "蜜瓜", "梨汤", "drink_preference", profile_line, history_line)
    )


def test_empty_failure_diagnostics_keep_content_free_signal_fields():
    diagnostics = failed_character_consistency_stage().diagnostics
    assert diagnostics["accepted_signal_histogram"] == []
    assert diagnostics["candidate_eligibility"] == {
        "stable_or_core_formal_signals": 0,
        "stable_or_core_history_signals": 0,
        "prelimit_candidates": 0,
    }


def test_case_trace_uses_frozen_objects_for_shared_preference_label():
    profile_line = "林澈喜欢蜜瓜，也喜欢葡萄。"
    draft_line = "林澈仍然喜欢蜜瓜，也仍然喜欢葡萄。"
    melon = _record(
        evidence=profile_line,
        polarity="positive",
        kind="explicit_declaration",
        trait_key="food_preference",
        statement="林澈喜欢蜜瓜",
    )
    grape = {**melon, "statement": "林澈喜欢葡萄", "key_object": "葡萄"}
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"同标签诊断对象-{uuid4().hex}"}
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
        _run_stage(seed, QueueProvider(_response(melon, grape)))
        confirmed_candidate_ids = _confirm_all_candidates(client, project["id"], seed)
        assert len(confirmed_candidate_ids) == 2
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content=draft_line,
            narrative_context=_context(publication="draft"),
        )
        result = _run_stage(
            _new_run(client, project["id"]),
            QueueProvider(
                _response(melon, grape),
                _response(
                    {**melon, "evidence": draft_line},
                    {**grape, "evidence": draft_line},
                ),
            ),
        )

    trace = result.diagnostics["case_trace"]
    assert len(trace) == 2
    assert {row["comparison_key"] for row in trace} == {
        "preference:蜜瓜",
        "preference:葡萄",
    }
    assert {row["confirmed_candidate_id_sha256"] for row in trace} == {
        hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        for candidate_id in confirmed_candidate_ids
    }
    assert not any(candidate_id in repr(trace) for candidate_id in confirmed_candidate_ids)
    assert all(row["matched_observation_count"] == 1 for row in trace)
    assert {
        row["comparison_key"]: row["matched_observation_refs"][0][
            "key_object_sha256"
        ]
        for row in trace
    } == {
        f"preference:{name}": hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in ("蜜瓜", "葡萄")
    }


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
                "confirmed_candidate_id_sha256": hashlib.sha256(
                    candidate_id.encode("utf-8")
                ).hexdigest(),
                "matched_observation_count": 1,
                "matched_observation_refs": [{
                    "document_name": "draft.md",
                    "line_start": 1,
                    "line_end": 1,
                    "observation_kind": "preference_expression",
                    "polarity": "negative",
                    "key_object_sha256": hashlib.sha256(
                        "蜜瓜".encode("utf-8")
                    ).hexdigest(),
                }],
                "matched_observation_refs_truncated": False,
                "prepare_reason": "reported_opposed_preference",
                "review_outcome": "completed",
                "review_verdict": "contradicts",
                "citation_roles": ["B", "C"],
                "citation_refs": [
                    {"handle": "B01", "role": "B", "document_name": "profile.md", "line_start": 1, "line_end": 1},
                    {"handle": "C01", "role": "C", "document_name": "draft.md", "line_start": 1, "line_end": 1},
                ],
                "citation_refs_incomplete": False,
                "final_outcome": "conflict",
                "visible": True,
                "promote_reason": "model_contradicts",
            }
        ]


def test_qualified_preference_from_confirmed_profile_reaches_review_without_identity_rewrite():
    profile_line = "林澈一直喜欢冰镇蜜瓜，这是他的稳定偏好。"
    draft_line = "林澈当着众人的面说：“我一直最讨厌蜜瓜，闻到味道就想离开。”"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"限定对象补桥-{uuid4().hex}"}
        ).json()
        _create_document(
            client, project["id"], name="profile.md", role="character_profile",
            content=profile_line, narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(
            seed,
            QueueProvider(
                _response(
                    {
                        **_record(
                            evidence=profile_line,
                            polarity="positive",
                            kind="explicit_declaration",
                            trait_key="melon_preference",
                        ),
                        "key_object": "冰镇蜜瓜",
                    }
                )
            ),
        )
        _confirm_only_candidate(client, project["id"], seed)
        _create_document(
            client, project["id"], name="draft.md", role="chapter",
            content=draft_line, narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                {
                    **_record(
                        evidence=profile_line,
                        polarity="positive",
                        kind="explicit_declaration",
                        trait_key="melon_preference",
                    ),
                    "key_object": "冰镇蜜瓜",
                }
            ),
            _response(
                _record(
                    evidence=draft_line,
                    polarity="negative",
                    kind="preference_expression",
                    trait_key="melon_preference",
                )
            ),
            json.dumps(
                {
                    "verdict": "contradicts",
                    "explanation": "草稿对蜜瓜作出普遍反向偏好声明，覆盖冰镇蜜瓜。",
                    "citations": ["B01", "C01"],
                },
                ensure_ascii=False,
            ),
        )

        result = _run_stage(_new_run(client, project["id"]), provider)

    assert result.diagnostics["outcome"] == "completed"
    assert result.diagnostics["counts"]["targeted_pass_scheduled_count"] == 0
    assert len(provider.calls) == 3
    assert "局部体验、不同食品或范围不清" in CHARACTER_REVIEW_SYSTEM_PROMPT
    trace = result.diagnostics["case_trace"][0]
    assert '"comparison_key":"preference:冰镇蜜瓜"' in provider.calls[1][1]
    assert trace["matched_observation_count"] == 1
    assert trace["review_verdict"] == "contradicts"
    assert trace["final_outcome"] == "conflict"
    assert len(result.issues) == 1


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
        assert result.diagnostics["accepted_signal_histogram"] == []
        assert result.diagnostics["candidate_eligibility"] == {
            "stable_or_core_formal_signals": 0,
            "stable_or_core_history_signals": 0,
            "prelimit_candidates": 0,
        }


def test_isolated_character_profile_keeps_other_character_after_failed_section():
    profile = (
        "# 角色档案\n"
        "## 林澈\n"
        "林澈喜欢蜜瓜。\n"
        "## 祁雾\n"
        "祁雾说话直来直往，这是她的核心性格。"
    )
    good_line = "祁雾说话直来直往，这是她的核心性格。"
    malformed = _response({})
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"角色档案隔离-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content=profile,
            narrative_context=_context(publication="published"),
        )
        provider = QueueProvider(
            malformed,
            malformed,
            _response(
                _record(
                    character="祁雾",
                    evidence=good_line,
                    polarity="positive",
                    kind="explicit_declaration",
                    dimension="core_personality",
                    trait_key="directness",
                    statement="说话直来直往",
                    line=5,
                )
            ),
        )
        run_id = _new_run(client, project["id"])
        result = _run_stage(run_id, provider)

        assert len(provider.calls) == 3
        assert "林澈喜欢蜜瓜" in provider.calls[0][1]
        assert "祁雾说话直来直往" not in provider.calls[0][1]
        assert "祁雾说话直来直往" in provider.calls[2][1]
        assert result.diagnostics["material_coverage"] == "partial"
        assert result.diagnostics["counts"]["pending_candidate_count"] == 1
        with SessionLocal() as db:
            candidates = list(
                db.scalars(
                    select(CharacterTraitCandidateRow).where(
                        CharacterTraitCandidateRow.source_run_id
                        == run_id
                    )
                ).all()
            )
        assert len(candidates) == 1
        assert candidates[0].character_display_name == "祁雾"


def test_many_profile_sections_reserve_one_chunk_for_draft_extraction():
    profile = "# 角色档案\n" + "\n".join(
        f"## 角色{index:02}\n角色{index:02}喜欢蜜瓜。"
        for index in range(25)
    )
    draft_line = "角色00讨厌蜜瓜。"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"长档案保留草稿-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content=profile,
            narrative_context=_context(publication="published"),
        )
        _create_document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            content=draft_line,
            narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                _record(
                    character="角色00",
                    evidence=draft_line,
                    polarity="negative",
                    kind="explicit_declaration",
                )
            ),
            *([_response()] * 23),
        )
        result = _run_stage(
            _new_run(client, project["id"]),
            provider,
            remaining_run_tokens=100_000,
            character_consistency_stage_token_budget=100_000,
            character_signal_max_completion_tokens=64,
        )

    assert len(provider.calls) == 24
    assert draft_line in provider.calls[0][1]
    assert result.diagnostics["counts"]["planned_chunks"] == 26
    assert result.diagnostics["counts"]["processed_chunks"] == 24
    assert result.diagnostics["counts"]["draft_observation_count"] == 1
    assert result.diagnostics["reason_counts"]["chunk_limit"] == 2
    assert result.diagnostics["material_coverage"] == "partial"


def test_chunk_cap_reserves_each_draft_and_uncapped_run_keeps_source_order():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"多草稿块调度-{uuid4().hex}"}
        ).json()
        _create_document(
            client,
            project["id"],
            name="profile.md",
            role="character_profile",
            content="林澈喜欢蜜瓜。",
            narrative_context=_context(publication="published"),
        )
        for name, content in (
            ("draft-east.md", "林澈进入东线。"),
            ("draft-west.md", "林澈进入西线。"),
        ):
            _create_document(
                client,
                project["id"],
                name=name,
                role="chapter",
                content=content,
                narrative_context=_context(publication="draft"),
            )

        capped_provider = QueueProvider(_response(), _response())
        capped = _run_stage(
            _new_run(client, project["id"]),
            capped_provider,
            character_consistency_max_chunks_per_run=2,
        )
        assert "林澈进入东线" in capped_provider.calls[0][1]
        assert "林澈进入西线" in capped_provider.calls[1][1]
        assert capped.diagnostics["reason_counts"]["chunk_limit"] == 1
        assert capped.diagnostics["material_coverage"] == "partial"

        uncapped_provider = QueueProvider(_response(), _response(), _response())
        uncapped = _run_stage(
            _new_run(client, project["id"]), uncapped_provider
        )
        assert "林澈喜欢蜜瓜" in uncapped_provider.calls[0][1]
        assert "林澈进入东线" in uncapped_provider.calls[1][1]
        assert "林澈进入西线" in uncapped_provider.calls[2][1]
        assert uncapped.diagnostics["counts"]["processed_chunks"] == 3
        assert "chunk_limit" not in uncapped.diagnostics["reason_counts"]


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


def test_verification_admission_diagnostic_separates_no_call_from_model_rejection():
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
            side_effect=[100, 100, 100, 1_100],
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

        assert len(provider.calls) == 3
        assert result.diagnostics["reason_counts"]["targeted_verification_token_budget"] == 1
        assert result.diagnostics["counts"]["token_admission_omitted_count"] == 0
        assert result.diagnostics["token_admission_events"] == [
            {
                "stage_phase": "targeted_verification",
                "signal_phase": "initial",
                "chunk_ordinal": 2,
                "target_ordinal": 1,
                "estimated_tokens": 1_100,
                "available_tokens": 700,
                "stage_remaining_before": 1_700,
                "reviewer_reserve_tokens": 1_000,
                "model_calls_before_failure": 0,
            }
        ]
        assert result.diagnostics["material_coverage"] == "partial"


def test_second_target_admission_location_is_numeric_and_stable():
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
        with patch(
            "app.character_trait_extraction.estimate_issue_evidence_review_tokens",
            side_effect=[100, 100, 700, 1_200, 100],
        ):
            result = _run_stage(
                _new_run(client, project["id"]),
                provider,
                character_consistency_stage_token_budget=3_000,
                character_signal_max_completion_tokens=64,
                character_drift_token_budget=1_000,
                character_drift_max_completion_tokens=64,
            )

        events = result.diagnostics["token_admission_events"]
        assert len(events) == 1
        assert events[0] == {
            "stage_phase": "targeted_recall",
            "signal_phase": "initial",
            "chunk_ordinal": 2,
            "target_ordinal": 2,
            "estimated_tokens": 1_200,
            "available_tokens": 1_100,
            "stage_remaining_before": 2_100,
            "reviewer_reserve_tokens": 1_000,
            "model_calls_before_failure": 0,
        }
        assert len(provider.calls) == 4
        assert result.diagnostics["material_coverage"] == "partial"


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


@pytest.mark.parametrize(
    "line,expected",
    [
        ("甲在场，乙完成六周训练后已经克服恐惧。", None),
        ("甲看见乙暂时伪装成守卫。", None),
        ("甲目睹乙逐渐改变。", None),
        ("甲要求乙暂时伪装成守卫。", None),
        ("乙要求甲暂时伪装成守卫。", None),
        ("甲的同伴乙完成六周训练后已经克服恐惧。", None),
        ("甲和乙一起暂时伪装成守卫。", None),
        ("甲在场。乙完成六周训练后已经克服恐惧。", None),
        ("甲知道乙完成六周训练后已经克服恐惧。", None),
        ("甲说乙暂时伪装成守卫。", None),
        ("甲、乙暂时伪装成守卫。", None),
        ("甲完成六周训练后已经克服恐惧。", "causal_bridge"),
        ("甲暂时伪装成守卫。", "exception"),
        ("甲受伤后暂时失忆。", "exception"),
        ("训练后甲逐渐改变了待人方式。", "causal_bridge"),
        ("甲为潜入宴会，假装成侍者并故意表现得健谈。", "exception"),
    ],
)
def test_generic_support_requires_target_as_event_agent(
    line: str, expected: str | None,
):
    assert _explicit_support_kind(line, character="甲") == expected


def test_support_search_never_exposes_other_actor_generic_g_or_x_handles():
    scope = NarrativeScopeV1()
    baseline = _confirmed_trait(character="甲", dimension="core_personality")
    source = _FrozenDocument(
        input_id="history-input",
        document=DocumentInput(
            id="history", name="history.md", role="chapter",
            content=(
                "甲在场，乙完成六周训练后已经克服恐惧。\n"
                "甲看见乙暂时伪装成守卫。\n"
                "甲完成六周训练后已经克服恐惧。\n"
                "甲暂时伪装成守卫。"
            ),
        ),
        document_version=1, content_sha256="0" * 64, ordinal=0,
        source_kind="published_history", source_reason="published_history",
        scope=scope, resolution_state="confirmed",
        publication_status="published", authority_tier="formal_record",
    )
    support = _find_support_evidence(
        baseline=baseline, baseline_scope=scope,
        draft_scopes=(scope,), draft_ordinals=(1,),
        draft_document_ids=("draft",), documents=[source], limit=8,
    )
    assert [(row.evidence.line_start, row.kind) for row in support] == [
        (3, "causal_bridge"), (4, "exception"),
    ]


def test_published_training_and_medical_support_remain_actor_bound():
    root = Path(__file__).resolve().parents[1] / "data"
    for version, suite, character, expected in (
        ("v1", "dev", "沈禾", {3: "causal_bridge", 4: "causal_bridge"}),
        ("v2", "dev", "许箬", {3: "causal_bridge", 4: "causal_bridge"}),
        ("v2", "dev", "荀木", {5: "exception"}),
        ("v2", "transfer", "温弦", {11: "causal_bridge"}),
        ("v2", "transfer", "童画", {13: "exception"}),
    ):
        rows = (
            root / f"character-axis-challenge-{version}" / suite
            / "03-published-history-v1.0.md"
        ).read_text(encoding="utf-8").splitlines()
        for line_number, kind in expected.items():
            assert _explicit_support_kind(
                rows[line_number - 1], character=character,
            ) == kind

    transfer_rows = (
        root / "character-axis-challenge-v2" / "transfer"
        / "03-published-history-v1.0.md"
    ).read_text(encoding="utf-8").splitlines()
    for character in ("许砚灯", "曲霁", "罗月", "杭泊"):
        assert _explicit_support_kind(
            transfer_rows[2], character=character,
        ) is None


def test_transfer_published_harm_is_g_only_with_same_actor_public_review():
    history = (
        Path(__file__).resolve().parents[1]
        / "data/character-axis-challenge-v2/transfer/03-published-history-v1.0.md"
    ).read_text(encoding="utf-8")
    scope = NarrativeScopeV1()
    source = _FrozenDocument(
        input_id="history-input",
        document=DocumentInput(
            id="history", name="03-published-history-v1.0.md",
            role="chapter", content=history,
        ),
        document_version=1, content_sha256="0" * 64, ordinal=0,
        source_kind="published_history", source_reason="published_history",
        scope=scope, resolution_state="confirmed",
        publication_status="published", authority_tier="formal_record",
    )
    support = _find_support_evidence(
        baseline=_confirmed_trait(character="温弦", dimension="core_personality"),
        baseline_scope=scope, draft_scopes=(scope,), draft_ordinals=(1,),
        draft_document_ids=("draft",), documents=[source], limit=8,
    )
    assert [(row.evidence.line_start, row.kind) for row in support] == [
        (9, "causal_bridge"), (11, "causal_bridge"),
    ]


@pytest.mark.parametrize(
    "line,expected",
    [
        (
            "去年夜航时，荀木曾因高烧被医师当面要求当日暂停所有热饮，"
            "先喝常温水。荀木照做，退烧后又恢复了喝热饮的习惯。",
            "exception",
        ),
        (
            "荀木高烧，医生当面要求荀木当天暂停热饮。荀木照做，退烧后恢复。",
            "exception",
        ),
        ("如果荀木高烧被医师要求当日暂停热饮，荀木就会照做。", None),
        ("荀木高烧被医师要求当日暂停热饮，荀木可能照做。", None),
        ("荀木高烧被医师要求当日暂停热饮，荀木没有照做。", None),
        ("荀木高烧被医师要求当日暂停热饮，荀木照做的说法并未证实。", None),
        ("荀木高烧被医师要求当日暂停热饮，尚未说明荀木是否照做。", None),
        ("医嘱要求荀木当日暂停热饮，荀木照做。", None),
        ("规则规定荀木高烧时应该暂停热饮。", None),
        ("荀木高烧被医师要求当日暂停热饮。", None),
        ("荀木高烧，医师要求苏弦当日暂停热饮，荀木照做。", None),
        ("戏本台词：「荀木高烧被医师要求当日暂停热饮，荀木照做。」", None),
    ],
)
def test_medical_support_requires_same_actor_real_order_and_compliance(
    line: str, expected: str | None,
):
    assert _explicit_support_kind(line, character="荀木") == expected


def test_medical_support_accepts_prior_spicy_food_treatment_only_after_compliance():
    treatment = (
        "童画在事故救援中被热灰灼伤咽喉。医师在治疗记录里明确写下："
        "从当天起三天内，童画要避开热和辛辣食物，三天后复查；"
    )
    assert _explicit_support_kind(
        treatment + "童画照做，治疗期间暂不吃热椒饼。记录没有说她不再喜欢热椒饼。",
        character="童画",
    ) == "exception"
    assert _explicit_support_kind(
        treatment + "记录没有说她不再喜欢热椒饼。",
        character="童画",
    ) is None


def test_medical_support_accepts_actual_published_transfer_compliance_form():
    history = (
        "童画在事故救援中被热灰灼伤咽喉。医师在治疗记录里明确写下："
        "从当天起三天内，童画要避开热和辛辣食物，三天后复查；"
        "童画当天就依医嘱把刚出炉的热椒饼放回摊位，只喝温水；"
        "记录没有说她不再喜欢热椒饼。市集草稿发生在治疗后的第二天。"
    )
    assert _explicit_support_kind(history, character="童画") == "exception"
    assert _explicit_support_kind(
        history.replace("童画当天就依医嘱把刚出炉的热椒饼放回摊位，只喝温水；", ""),
        character="童画",
    ) is None


def test_public_retrospective_statement_is_actual_speech_not_future_action():
    event = (
        "顾衡为了尽快恢复贸易路口，命令温弦先开一处未经检查的次级幕门。"
        "温弦当时照办，结果粉尘涌入避难廊，十余名居民被迫转移；"
        "她亲眼看见一个孩子因呼吸困难被担架抬走。"
    )
    statement = (
        "事故复盘会上，温弦当着顾衡和居民的面说：「我以前不敢公开反对师父；"
        "这次服从错误命令伤到了人。以后如果命令会把居民送进灰潮，"
        "我会在现场直接提出反对，哪怕是顾衡下的命令。」"
        "顾衡没有把这句话解释成玩笑，也没有要求她收回。"
    )
    # Harmful compliance by itself is background, not evidence of growth.
    assert _explicit_support_kind(event, character="温弦") is None
    assert _explicit_support_kind(statement, character="温弦") == "causal_bridge"
    assert _explicit_support_kind(statement, character="顾衡") is None
    for unsupported in (
        statement.replace("事故复盘会上", "排练会上"),
        statement.replace("温弦当着顾衡和居民的面说", "温弦打算当着顾衡和居民的面说"),
        statement.replace("温弦当着顾衡和居民的面说", "温弦并未当着顾衡和居民的面说"),
        statement.replace("这次服从错误命令伤到了人。", "如果服从错误命令可能伤到人。"),
        statement.replace("我以前不敢", "我以前并非不敢"),
        statement.replace("这次服从错误命令伤到了人。", "这次服从错误命令没有伤到人。"),
        statement.replace("我会在现场直接提出反对", "我会在现场不反对"),
        statement.replace("事故复盘会上，温弦当着顾衡和居民的面说：", "温弦在私下说："),
    ):
        assert _explicit_support_kind(unsupported, character="温弦") is None


def test_harmful_compliance_support_requires_later_same_source_public_statement():
    event = (
        "总监命令温弦开启未经检查的幕门。温弦当时照办，"
        "结果十余名居民被迫转移。"
    )
    statement = (
        "事故复盘会上，温弦当着总监和居民的面说：「我以前不敢公开反对上级；"
        "这次服从错误命令伤到了人。以后如果命令会伤人，我会当面提出反对。」"
    )
    scope = NarrativeScopeV1.model_validate({
        "release": {"key": "v1.0", "ordinal": 10}
    })
    draft_scope = NarrativeScopeV1.model_validate({
        "release": {"key": "v1.1", "ordinal": 11}
    })
    baseline = _confirmed_trait(character="温弦", dimension="core_personality")

    def found(content: str) -> list[int]:
        source = _FrozenDocument(
            input_id="history-input",
            document=DocumentInput(
                id="history", name="history.md", content=content, role="chapter"
            ),
            document_version=1, content_sha256="0" * 64, ordinal=1,
            source_kind="published_history", source_reason="published_history",
            scope=scope, resolution_state="confirmed",
            publication_status="published", authority_tier="formal_record",
        )
        support = _find_support_evidence(
            baseline=baseline, baseline_scope=scope,
            draft_scopes=(draft_scope,), draft_ordinals=(2,),
            draft_document_ids=("draft-document",), documents=[source], limit=8,
        )
        assert all(row.kind == "causal_bridge" for row in support)
        return [row.evidence.line_start for row in support]

    assert found(event + "\ninterlude\n" + statement) == [1, 3]
    assert found(event) == []
    assert found(event + "\n" + statement.replace("温弦当着", "另一人当着")) == []
    assert found(event + "\n" + statement.replace("事故复盘会上", "排练会上")) == []
    assert found(event.replace("温弦当时照办", "温弦没有照办") + "\n" + statement) == [2]
    assert found(event.replace("结果十余名居民被迫转移", "结果十余名居民没有被迫转移") + "\n" + statement) == [2]
    assert found(event.replace("命令温弦", "命令另一人") + "\n" + statement) == [2]
    assert found(statement + "\n" + event) == [1]


def test_support_search_accepts_only_prior_published_authoritative_sources():
    history_line = (
        "去年夜航时，荀木曾因高烧被医师当面要求当日暂停所有热饮。"
        "荀木照做，退烧后恢复了喝热饮的习惯。"
    )
    baseline = _confirmed_trait(character="荀木", trait_key="hot_drink_preference")
    old_scope = NarrativeScopeV1.model_validate({
        "release": {"key": "v1.0", "ordinal": 10}
    })
    draft_scope = NarrativeScopeV1.model_validate({
        "release": {"key": "v1.1", "ordinal": 11}
    })
    future_scope = NarrativeScopeV1.model_validate({
        "release": {"key": "v1.2", "ordinal": 12}
    })

    def source(
        name: str, *, kind: str, publication: str, authority: str,
        ordinal: int, scope: NarrativeScopeV1 = old_scope,
        content: str = history_line,
    ) -> _FrozenDocument:
        return _FrozenDocument(
            input_id=f"input-{name}",
            document=DocumentInput(
                id=name, name=f"{name}.md", content=content,
                role="chapter" if kind in {"draft", "published_history"} else "canon",
            ),
            document_version=1, content_sha256="0" * 64, ordinal=ordinal,
            source_kind=kind, source_reason=kind, scope=scope,
            resolution_state="confirmed", publication_status=publication,
            authority_tier=authority,
        )

    sources = [
        source("growth", kind="formal_character_profile", publication="published",
               authority="core_canon", ordinal=0,
               content="训练后荀木逐渐改变了待人方式。"),
        source("medical", kind="published_history", publication="published",
               authority="formal_record", ordinal=1),
        source("self_draft", kind="draft", publication="draft",
               authority="draft", ordinal=2),
        source("mislabelled_draft", kind="draft", publication="published",
               authority="formal_record", ordinal=0),
        source("in_review", kind="formal_character_profile", publication="in_review",
               authority="formal_record", ordinal=0),
        source("retired", kind="published_history", publication="retired",
               authority="formal_record", ordinal=0),
        source("low_authority", kind="formal_character_profile", publication="published",
               authority="reference", ordinal=0),
        source("future_input", kind="published_history", publication="published",
               authority="formal_record", ordinal=3),
        source("future_release", kind="published_history", publication="published",
               authority="formal_record", ordinal=0, scope=future_scope),
    ]
    support = _find_support_evidence(
        baseline=baseline, baseline_scope=old_scope,
        draft_scopes=(draft_scope,), draft_ordinals=(2,),
        draft_document_ids=("draft-document",),
        documents=sources, limit=12,
    )
    assert {(row.evidence.document_id, row.kind) for row in support} == {
        ("growth", "causal_bridge"), ("medical", "exception"),
    }
    assert all(row.eligible_draft_document_ids == ("draft-document",) for row in support)
    assert all(row.publication_status == "published" for row in support)


@pytest.mark.parametrize("include_published_history", [False, True])
def test_api_stage_never_promotes_same_draft_medical_defense_to_support(
    monkeypatch, include_published_history: bool,
):
    profile_line = "荀木喜欢热姜梅露。"
    medical_line = (
        "去年夜航时，荀木曾因高烧被医师当面要求当日暂停所有热饮。"
        "荀木照做，退烧后恢复了喝热姜梅露的习惯。"
    )
    draft_line = "荀木说讨厌热姜梅露。"
    profile_record = {
        **_record(
            character="荀木", evidence=profile_line, polarity="positive",
            kind="explicit_declaration", trait_key="hot_drink_preference",
        ),
        "key_object": "热姜梅露",
    }
    draft_record = {
        **_record(
            character="荀木", evidence=draft_line, polarity="negative",
            kind="preference_expression", trait_key="hot_drink_preference",
        ),
        "key_object": "热姜梅露",
    }
    captured_support: list[tuple[SupportEvidence, ...]] = []
    original_prepare = prepare_character_drift

    def capture_prepare(case):
        captured_support.append(case.support_evidence)
        return original_prepare(case)

    monkeypatch.setattr(
        "app.character_consistency_stage.prepare_character_drift", capture_prepare
    )
    monkeypatch.setattr(
        "app.character_consistency_stage.CharacterConsistencyReviewer.review",
        lambda _self, _prepared: _trace_review(
            ("B01", "C01", "X01") if include_published_history else ("B01", "C01"),
            verdict="explained" if include_published_history else "needs_confirmation",
        ),
    )
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"医疗来源门-{uuid4().hex}"}
        ).json()
        _create_document(
            client, project["id"], name="profile.md", role="character_profile",
            content=profile_line, narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(seed, QueueProvider(_response(profile_record)))
        _confirm_only_candidate(client, project["id"], seed)
        if include_published_history:
            _create_document(
                client, project["id"], name="medical-history.md", role="chapter",
                content=medical_line, narrative_context=_context(publication="published"),
            )
        draft_document = _create_document(
            client, project["id"], name="draft.md", role="chapter",
            content=f"{draft_line}\n{medical_line}",
            narrative_context=_context(publication="draft"),
        )
        responses = [_response(profile_record)]
        if include_published_history:
            responses.append(_response())
        responses.append(_response(draft_record))
        result = _run_stage(_new_run(client, project["id"]), QueueProvider(*responses))

    assert captured_support
    supports = captured_support[-1]
    if include_published_history:
        assert [(row.evidence.document_name, row.kind) for row in supports] == [
            ("medical-history.md", "exception")
        ]
        assert supports[0].eligible_draft_document_ids == (draft_document["id"],)
    else:
        assert supports == ()
    trace = result.diagnostics["case_trace"][0]
    if include_published_history:
        assert trace["review_verdict"] == "explained"
        assert [row for row in trace["citation_refs"] if row["role"] == "X"] == [{
            "handle": "X01", "role": "X", "document_name": "medical-history.md",
            "line_start": 1, "line_end": 1,
        }]
    else:
        assert trace["review_verdict"] == "needs_confirmation"
        assert all(row["role"] != "X" for row in trace["citation_refs"])


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
        draft_ordinals=(1,),
        draft_document_ids=("draft",),
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
        confirmed_candidate_ids = _confirm_all_candidates(client, project["id"], seed)
        assert len(confirmed_candidate_ids) == 3
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
        assert {row["confirmed_candidate_id_sha256"] for row in trace} == {
            hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
            for candidate_id in confirmed_candidate_ids
        }
        assert by_character["林澈"] == {
            "character_key": "林澈",
            "dimension": "core_personality",
            "comparison_key": "core_personality:社交主动性",
            "confirmed_candidate_id_sha256": by_character["林澈"][
                "confirmed_candidate_id_sha256"
            ],
            "matched_observation_count": 1,
            "matched_observation_refs": [{
                "document_name": "draft.md",
                "line_start": 1,
                "line_end": 1,
                "observation_kind": "action",
                "polarity": "positive",
                "key_object_sha256": None,
            }],
            "matched_observation_refs_truncated": False,
            "prepare_reason": "single_behavior_is_not_drift",
            "review_outcome": "not_run",
            "review_verdict": None,
            "citation_roles": [],
            "citation_refs": [],
            "citation_refs_incomplete": False,
            "final_outcome": "needs_confirmation",
            "visible": False,
            "promote_reason": "single_behavior_is_not_drift",
        }
        assert by_character["苏弦"]["prepare_reason"] == "no_opposition"
        assert by_character["苏弦"]["final_outcome"] == "no_issue"
        assert by_character["苏弦"]["matched_observation_refs"] == [{
            "document_name": "draft.md",
            "line_start": 2,
            "line_end": 2,
            "observation_kind": "preference_expression",
            "polarity": "positive",
            "key_object_sha256": hashlib.sha256(
                "蜜瓜".encode("utf-8")
            ).hexdigest(),
        }]
        assert by_character["祁雾"]["matched_observation_count"] == 0
        assert by_character["祁雾"]["matched_observation_refs"] == []
        assert by_character["祁雾"]["prepare_reason"] == "no_matching_observation"
        assert by_character["祁雾"]["final_outcome"] == "unverifiable"
        serialized = json.dumps(trace, ensure_ascii=False)
        assert "evidence" not in serialized
        assert "statement" not in serialized
        assert "document_id" not in serialized
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


@pytest.mark.parametrize(
    "frozen_key",
    (None, "preference:https://private.invalid/object", "preference:wrong:axis"),
)
def test_case_trace_legacy_fallback_rejects_missing_or_unsafe_frozen_key(frozen_key):
    baseline = _confirmed_trait(trait_key="food_preference")
    entry = (
        _snapshot_stub("legacy", frozen_key),
        baseline,
        NarrativeScopeV1(),
        "林澈",
    )
    trace = _safe_case_trace(
        character_key="林澈",
        baseline=baseline,
        baseline_entry=entry,
        matched_observation_count=0,
        prepare_reason="no_matching_observation",
        review=None,
        final_outcome="unverifiable",
        visible=False,
        promote_reason="no_matching_observation",
    )

    assert trace["comparison_key"] == "preference:foodpreference"
    assert trace["confirmed_candidate_id_sha256"] is None
    assert "private.invalid" not in json.dumps(trace, ensure_ascii=False)


def test_case_trace_observation_provenance_is_bounded_and_content_free():
    observations = tuple(
        CharacterSignal(
            id=f"cs_{index:032x}",
            character="林澈",
            dimension="preference",
            trait_key="melon_preference",
            statement="机密模型陈述不可出现在诊断中",
            polarity="negative",
            stability="stable",
            observation_kind="preference_expression",
            key_object="蜜瓜",
            source_kind="draft",
            evidence=EvidenceSpan(
                document_id=f"secret-input-{index}",
                document_name=(
                    "sk-1234567890abcdef.md" if index == 0 else "draft.md"
                ),
                line_start=index + 1,
                line_end=index + 1,
                text="机密证据原文不可出现在诊断中",
            ),
        )
        for index in range(13)
    )
    trace = _safe_case_trace(
        character_key="林澈",
        baseline=_confirmed_trait(),
        matched_observation_count=len(observations),
        matched_observations=observations,
        prepare_reason="reported_opposed_preference",
        review=None,
        final_outcome="needs_confirmation",
        visible=False,
        promote_reason="review_unavailable",
    )

    refs = trace["matched_observation_refs"]
    assert trace["matched_observation_count"] == 13
    assert len(refs) == 12
    assert trace["matched_observation_refs_truncated"] is True
    assert refs[0] == {
        "document_name": refs[0]["document_name"],
        "line_start": 1,
        "line_end": 1,
        "observation_kind": "preference_expression",
        "polarity": "negative",
        "key_object_sha256": hashlib.sha256("蜜瓜".encode("utf-8")).hexdigest(),
    }
    assert refs[0]["document_name"].startswith("redacted_")
    assert refs[1]["document_name"] == "draft.md"
    serialized = json.dumps(trace, ensure_ascii=False)
    for marker in (
        "sk-1234567890abcdef", "机密模型陈述", "机密证据原文",
        "secret-input", '"key_object":',
    ):
        assert marker not in serialized


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


def _citation_trace_case(
    *,
    observation_count: int = 1,
    bridge_line: int = 7,
    bridge_document_name: str = "bridge.md",
):
    observations = tuple(
        CharacterSignal(
            id=f"cs_{index:032x}",
            character="林澈",
            dimension="preference",
            trait_key="食物偏好:蜜瓜",
            statement="机密模型陈述：林澈厌恶蜜瓜",
            polarity="negative",
            stability="stable",
            observation_kind="preference_expression",
            key_object="蜜瓜",
            source_kind="draft",
            evidence=EvidenceSpan(
                document_id=f"draft-{index}",
                document_name="draft.md",
                line_start=index + 1,
                line_end=index + 1,
                text="机密草稿原文：林澈说讨厌蜜瓜。",
            ),
        )
        for index in range(observation_count)
    )
    support = (
        SupportEvidence(
            id="se_trace_bridge",
            kind="causal_bridge",
            summary="机密成长摘要",
            explicit=True,
            evidence=EvidenceSpan(
                document_id="bridge-document",
                document_name=bridge_document_name,
                line_start=bridge_line,
                line_end=bridge_line,
                text="机密成长原文",
            ),
        ),
        SupportEvidence(
            id="se_trace_exception",
            kind="exception",
            summary="机密例外摘要",
            explicit=True,
            evidence=EvidenceSpan(
                document_id="exception-document",
                document_name="exception.md",
                line_start=11,
                line_end=11,
                text="机密例外原文",
            ),
        ),
    )
    return prepare_character_drift(
        CharacterDriftCase(
            id="cdc_trace_citations",
            baseline=_confirmed_trait(),
            observations=observations,
            support_evidence=support,
            scope_compatibility="compatible",
            material_coverage="complete",
        )
    )


def _trace_review(citations: tuple[str, ...], *, verdict: str = "explained"):
    return CharacterReviewResult(
        decision=ModelDriftDecision(
            verdict=verdict,
            explanation="有足够的历史证据解释当前表现。",
            citations=citations,
        ),
        diagnostics=CharacterReviewDiagnostics(outcome="completed", reason="completed"),
    )


def _citation_case_trace(prepared, review):
    return _safe_case_trace(
        character_key="林澈",
        baseline=prepared.case.baseline,
        matched_observation_count=len(prepared.matching_observations),
        matched_observations=prepared.matching_observations,
        prepared=prepared,
        prepare_reason=prepared.reason,
        review=review,
        final_outcome="no_issue",
        visible=False,
        promote_reason="model_explained" if review is not None else "review_unavailable",
    )


def test_case_trace_explained_citations_resolve_frozen_bridge_and_exception_coordinates():
    prepared = _citation_trace_case()
    trace = _citation_case_trace(
        prepared, _trace_review(("B01", "C01", "G01", "X01"))
    )
    assert trace["review_verdict"] == "explained"
    assert trace["citation_roles"] == ["B", "C", "G", "X"]
    assert trace["citation_refs"] == [
        {"handle": "B01", "role": "B", "document_name": "profile.md", "line_start": 1, "line_end": 1},
        {"handle": "C01", "role": "C", "document_name": "draft.md", "line_start": 1, "line_end": 1},
        {"handle": "G01", "role": "G", "document_name": "bridge.md", "line_start": 7, "line_end": 7},
        {"handle": "X01", "role": "X", "document_name": "exception.md", "line_start": 11, "line_end": 11},
    ]
    assert trace["citation_refs_incomplete"] is False
    serialized = json.dumps(trace, ensure_ascii=False)
    assert "机密" not in serialized
    assert '"text"' not in serialized
    assert '"summary"' not in serialized


def test_explained_review_without_visible_issue_retains_frozen_growth_citation_ref():
    profile_line = "林澈喜欢蜜瓜。"
    growth_line = "训练后林澈逐渐改变了待人方式。"
    draft_line = "林澈明确说自己讨厌蜜瓜。"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"解释引用坐标-{uuid4().hex}"},
        ).json()
        _create_document(
            client, project["id"], name="profile.md", role="character_profile",
            content=profile_line, narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(
            seed,
            QueueProvider(
                _response(
                    _record(
                        evidence=profile_line, polarity="positive",
                        kind="explicit_declaration", trait_key="melon_preference",
                    )
                )
            ),
        )
        _confirm_only_candidate(client, project["id"], seed)
        _create_document(
            client, project["id"], name="growth.md", role="chapter",
            content=growth_line, narrative_context=_context(publication="published"),
        )
        _create_document(
            client, project["id"], name="draft.md", role="chapter",
            content=draft_line, narrative_context=_context(publication="draft"),
        )
        provider = QueueProvider(
            _response(
                _record(
                    evidence=profile_line, polarity="positive",
                    kind="explicit_declaration", trait_key="melon_preference",
                )
            ),
            _response(),
            _response(
                _record(
                    evidence=draft_line, polarity="negative",
                    kind="preference_expression", trait_key="melon_preference",
                )
            ),
            json.dumps(
                {
                    "verdict": "explained",
                    "explanation": "已发布的历史经历解释了当前表现。",
                    "citations": ["B01", "C01", "G01"],
                },
                ensure_ascii=False,
            ),
        )
        result = _run_stage(_new_run(client, project["id"]), provider)

    assert result.issues == ()
    trace = result.diagnostics["case_trace"][0]
    assert trace["review_verdict"] == "explained"
    assert trace["final_outcome"] == "no_issue"
    assert trace["citation_refs_incomplete"] is False
    assert {item["handle"]: item for item in trace["citation_refs"]}["G01"] == {
        "handle": "G01", "role": "G", "document_name": "growth.md",
        "line_start": 1, "line_end": 1,
    }
    assert growth_line not in json.dumps(trace, ensure_ascii=False)


@pytest.mark.parametrize(
    "citations",
    [
        ("B01", "C01", "G99"),
        ("B01", "C01", "Q01"),
        ("B1", "C01", "G01"),
        ("B01", "B01", "C01", "G01"),
    ],
)
def test_case_trace_bad_or_duplicate_citation_handle_fails_closed(citations):
    trace = _citation_case_trace(_citation_trace_case(), _trace_review(citations))
    assert trace["citation_refs"] == []
    assert trace["citation_refs_incomplete"] is True


@pytest.mark.parametrize(
    ("bridge_line", "bridge_document_name"),
    [
        (0, "bridge.md"),
        (10_000_001, "bridge.md"),
        (7, "sk-1234567890abcdef.md"),
    ],
)
def test_case_trace_citation_coordinate_or_name_out_of_bounds_fails_closed(
    bridge_line: int, bridge_document_name: str,
):
    trace = _citation_case_trace(
        _citation_trace_case(
            bridge_line=bridge_line,
            bridge_document_name=bridge_document_name,
        ),
        _trace_review(("B01", "C01", "G01")),
    )
    assert trace["citation_refs"] == []
    assert trace["citation_refs_incomplete"] is True
    assert "sk-1234567890abcdef" not in json.dumps(trace, ensure_ascii=False)


def test_case_trace_citation_refs_are_bounded_and_marked_incomplete_if_truncated():
    prepared = _citation_trace_case(observation_count=7)
    # The validated provider contract permits at most eight citations. This
    # synthetic over-limit result still must not grow persisted diagnostics.
    oversized = ModelDriftDecision.model_construct(
        verdict="explained",
        explanation="有足够的历史证据解释当前表现。",
        citations=("B01", *tuple(f"C{index:02d}" for index in range(1, 8)), "G01"),
    )
    review = CharacterReviewResult.model_construct(
        decision=oversized,
        diagnostics=CharacterReviewDiagnostics(outcome="completed", reason="completed"),
    )
    trace = _citation_case_trace(prepared, review)
    assert len(trace["citation_refs"]) == 8
    assert [item["handle"] for item in trace["citation_refs"]] == [
        "B01", "C01", "C02", "C03", "C04", "C05", "C06", "C07",
    ]
    assert trace["citation_refs_incomplete"] is True


def test_case_trace_without_review_has_no_citation_refs_and_no_false_incomplete():
    trace = _citation_case_trace(_citation_trace_case(), None)
    assert trace["review_outcome"] == "not_run"
    assert trace["citation_roles"] == []
    assert trace["citation_refs"] == []
    assert trace["citation_refs_incomplete"] is False


def test_case_trace_invalid_model_response_marks_citation_refs_incomplete_without_guessing():
    degraded = CharacterReviewResult(
        decision=None,
        diagnostics=CharacterReviewDiagnostics(
            outcome="degraded", reason="invalid_model_response",
            attempted_calls=1,
        ),
    )
    trace = _citation_case_trace(_citation_trace_case(), degraded)
    assert trace["review_outcome"] == "degraded"
    assert trace["citation_refs"] == []
    assert trace["citation_refs_incomplete"] is True


def test_accepted_draft_refs_keep_unmatched_actor_but_exclude_rejected_false_actor():
    profile_line = "林澈喜欢蜜瓜。"
    draft_line = "苏弦喜欢蜜瓜。"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"错归属诊断-{uuid4().hex}"},
        ).json()
        _create_document(
            client, project["id"], name="profile.md", role="character_profile",
            content=profile_line, narrative_context=_context(publication="published"),
        )
        seed = _new_run(client, project["id"])
        _run_stage(
            seed,
            QueueProvider(
                _response(
                    _record(
                        evidence=profile_line, polarity="positive",
                        kind="explicit_declaration", trait_key="melon_preference",
                    )
                )
            ),
        )
        _confirm_only_candidate(client, project["id"], seed)
        _create_document(
            client, project["id"], name="draft.md", role="chapter",
            content=draft_line, narrative_context=_context(publication="draft"),
        )
        result = _run_stage(
            _new_run(client, project["id"]),
            QueueProvider(
                _response(
                    _record(
                        evidence=profile_line, polarity="positive",
                        kind="explicit_declaration", trait_key="melon_preference",
                    )
                ),
                _response(
                    _record(
                        character="苏弦", evidence=draft_line,
                        polarity="positive", kind="preference_expression",
                        trait_key="melon_preference",
                    ),
                    _record(
                        character="祁雾", evidence=draft_line,
                        polarity="positive", kind="preference_expression",
                        trait_key="melon_preference",
                    ),
                ),
                _response(
                    _record(
                        character="苏弦", evidence=draft_line,
                        polarity="positive", kind="preference_expression",
                        trait_key="melon_preference",
                    ),
                ),
            ),
        )

    diagnostics = result.diagnostics
    assert diagnostics["accepted_draft_observation_total"] == 1
    assert diagnostics["accepted_draft_observation_refs_truncated"] is False
    assert diagnostics["accepted_draft_observation_refs"] == [{
        "character_key": "苏弦",
        "dimension": "preference",
        "polarity": "positive",
        "observation_kind": "preference_expression",
        "document_name": "draft.md",
        "line_start": 1,
        "line_end": 1,
    }]
    assert diagnostics["reason_counts"]["regenerated_from_character_support"] == 1
    assert diagnostics["case_trace"][0]["matched_observation_refs"] == []
    assert diagnostics["case_trace"][0]["review_outcome"] == "not_run"
    serialized = json.dumps(diagnostics["accepted_draft_observation_refs"], ensure_ascii=False)
    assert draft_line not in serialized
    assert "祁雾" not in serialized
    assert "trait_key" not in serialized
    assert "key_object" not in serialized


def test_accepted_draft_refs_are_bounded_and_omit_unsafe_actor_or_filename():
    signals = tuple(
        CharacterSignal(
            id=f"cs_{index:032x}",
            character=("sk-1234567890abcdef" if index == 64 else "苏弦"),
            dimension="preference",
            trait_key="机密模型标签",
            statement="机密模型陈述",
            polarity="positive",
            stability="stable",
            observation_kind="preference_expression",
            key_object="机密对象",
            source_kind="draft",
            evidence=EvidenceSpan(
                document_id=f"draft-{index}",
                document_name="draft.md",
                line_start=index + 1,
                line_end=index + 1,
                text="机密证据原文",
            ),
        )
        for index in range(65)
    )
    refs, total, truncated = _safe_accepted_draft_observation_refs(signals)
    assert total == 65 and len(refs) == 64 and truncated is True
    assert refs[0]["line_start"] == 1 and refs[-1]["line_start"] == 64
    serialized = json.dumps(refs, ensure_ascii=False)
    for secret in ("sk-1234567890abcdef", "机密模型标签", "机密模型陈述", "机密对象", "机密证据原文"):
        assert secret not in serialized
    unsafe_document = signals[0].model_copy(
        update={
            "evidence": signals[0].evidence.model_copy(
                update={"document_name": "https://private.invalid/draft.md"}
            )
        }
    )
    refs, total, truncated = _safe_accepted_draft_observation_refs(
        (signals[0], unsafe_document)
    )
    assert total == 2 and len(refs) == 1 and truncated is True
    assert "private.invalid" not in json.dumps(refs, ensure_ascii=False)


def _approved_core_baseline(
    *,
    trait_key: str,
    axis_id: str = "11111111-1111-4111-8111-111111111111",
    authority: str = "formal_record",
) -> ConfirmedTraitSnapshot:
    definition = "涉及同伴安全的路线决策是否征询当值伙伴"
    base = _confirmed_trait(
        character="林澈",
        dimension="core_personality",
        trait_key=trait_key,
        authority=authority,
    )
    return ConfirmedTraitSnapshot.model_validate(
        {
            **base.model_dump(),
            "statement": "林澈作出撤离决策前会先征询同伴",
            "approved_axis_id": axis_id,
            "approved_axis_version": 1,
            "approved_axis_display_name": "同伴协商",
            "approved_axis_definition": definition,
            "approved_axis_definition_sha256": hashlib.sha256(
                definition.encode("utf-8")
            ).hexdigest(),
        }
    )


def test_approved_axis_deduplicates_different_raw_labels_without_exposing_id():
    scope = NarrativeScopeV1()
    first = _approved_core_baseline(trait_key="partner_consultation")
    second = _approved_core_baseline(trait_key="collaborative_route_choice")
    entries = [
        (_snapshot_stub("first"), first, scope, "林澈"),
        (_snapshot_stub("second"), second, scope, "林澈"),
    ]
    source = _FrozenDocument(
        input_id="draft-input",
        document=DocumentInput(
            id="draft-axis", name="draft.md", content="林澈独自决定撤离路线。", role="chapter"
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
    context = _safe_server_context(source, baselines=entries)
    assert context.eligible_traits == 1
    assert context.included_traits == 1
    assert len(context.targets) == 1
    assert context.targets[0].approved_axis_identity == first.approved_axis_identity
    assert first.approved_axis_id not in context.payload
    assert first.approved_axis_definition not in context.payload
    prompt = _targeted_chunk_prompt(
        CharacterSignalChunk(
            "draft-axis", "draft.md", "林澈独自决定撤离路线。", 1, "draft"
        ),
        context.targets,
    )
    assert first.approved_axis_definition in prompt
    assert first.approved_axis_id not in prompt
    assert '"approved_axis_id"' not in prompt


def test_approved_axis_requires_server_binding_and_preserves_observation_label():
    base = _approved_core_baseline(trait_key="partner_consultation")
    entry = (_snapshot_stub("first"), base, NarrativeScopeV1(), "林澈")
    alternate_label_baseline = _approved_core_baseline(
        trait_key="collaborative_route_choice"
    )
    alternate_entry = (
        _snapshot_stub("second"), alternate_label_baseline,
        NarrativeScopeV1(), "林澈",
    )
    observation = CharacterSignal(
        id="cs_" + "a" * 32,
        character="林澈",
        dimension="core_personality",
        trait_key="new_model_label",
        statement="林澈独自决定撤离路线",
        polarity="negative",
        stability="core",
        observation_kind="decision",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft-axis",
            document_name="draft.md",
            line_start=1,
            line_end=1,
            text="林澈独自决定撤离路线。",
        ),
    )
    assert _observation_matches_baseline(entry, observation) is False
    assert _observation_matches_baseline(
        entry, observation, approved_axis_binding=base.approved_axis_identity
    ) is True
    assert _observation_matches_baseline(
        alternate_entry,
        observation,
        approved_axis_binding=base.approved_axis_identity,
    ) is True
    unbound = CharacterDriftCase(
        id="cdc_unbound",
        baseline=base,
        observations=(observation,),
        scope_compatibility="compatible",
    )
    assert prepare_character_drift(unbound).matching_observations == ()
    bound = unbound.model_copy(
        update={"approved_axis_bound_observation_ids": (observation.id,)}
    )
    assert prepare_character_drift(bound).matching_observations == (observation,)
    assert prepare_character_drift(bound).matching_observations[0].trait_key == (
        "new_model_label"
    )
    alternate_case = bound.model_copy(update={"baseline": alternate_label_baseline})
    assert prepare_character_drift(alternate_case).matching_observations == (
        observation,
    )


def test_approved_axis_shadowing_never_merges_different_or_legacy_axes():
    scope = NarrativeScopeV1()
    formal = (
        _snapshot_stub("formal"),
        _approved_core_baseline(trait_key="first_label"),
        scope,
        "林澈",
    )
    canon = (
        _snapshot_stub("canon"),
        _approved_core_baseline(trait_key="different_label", authority="core_canon"),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([formal, canon])
    assert shadowed == 1
    assert selected == [canon]
    different = (
        _snapshot_stub("other"),
        _approved_core_baseline(
            trait_key="first_label",
            authority="core_canon",
            axis_id="22222222-2222-4222-8222-222222222222",
        ),
        scope,
        "林澈",
    )
    assert _select_authoritative_baselines([formal, different])[1] == 0
    legacy = (
        _snapshot_stub("legacy"),
        _confirmed_trait(
            character="林澈", dimension="core_personality", trait_key="first_label"
        ),
        scope,
        "林澈",
    )
    assert _select_authoritative_baselines([legacy, canon])[1] == 0


def test_same_actor_source_line_bound_to_two_axes_is_ambiguous_even_with_other_text():
    first = _approved_core_baseline(trait_key="first_label")
    second = _approved_core_baseline(
        trait_key="second_label",
        axis_id="22222222-2222-4222-8222-222222222222",
    )
    first_signal = CharacterSignal(
        id="cs_" + "a" * 32,
        character="林澈",
        dimension="core_personality",
        trait_key="first_label",
        statement="林澈独自决定路线",
        polarity="negative",
        stability="temporary",
        observation_kind="decision",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft-axis", document_name="draft.md",
            line_start=7, line_end=7, text="林澈独自决定路线。"
        ),
    )
    second_signal = first_signal.model_copy(
        update={
            "id": "cs_" + "b" * 32,
            "trait_key": "second_label",
            "statement": "林澈没有征询伙伴",
            "evidence": first_signal.evidence.model_copy(
                update={"text": "没有征询伙伴。"}
            ),
        }
    )
    axes_by_signal = {
        first_signal.id: {first.approved_axis_identity},
        second_signal.id: {second.approved_axis_identity},
    }
    axes_by_line = {
        ("林澈", "draft-axis", 7): {
            first.approved_axis_identity,
            second.approved_axis_identity,
        }
    }
    for signal in (first_signal, second_signal):
        assert _trusted_axis_binding(
            signal,
            axis_bindings_by_signal=axes_by_signal,
            axis_bindings_by_line=axes_by_line,
        ) is None


def test_approved_axis_primary_key_match_still_requires_clean_targeted_binding():
    with TestClient(app) as client:
        project = _confirmed_directness_project(client)
        lines = (
            "祁雾用奉承话术迂回交流。",
            "祁雾再次用奉承话术迂回交流。",
        )
        _create_document(
            client, project["id"], name="draft.md", role="chapter",
            content="\n".join(lines),
            narrative_context=_context(publication="draft"),
        )
        run_id = _new_run(client, project["id"])
        definition = "是否在交流中直接表达真实意见"
        axis_id = "11111111-1111-4111-8111-111111111111"
        with SessionLocal() as db:
            row = db.scalar(
                select(AnalysisRunCharacterTraitInputRow).where(
                    AnalysisRunCharacterTraitInputRow.run_id == run_id
                )
            )
            assert row is not None
            payload = {
                **row.payload,
                "approved_axis_id": axis_id,
                "approved_axis_version": 1,
                "approved_axis_display_name": "直接表达",
                "approved_axis_definition": definition,
                "approved_axis_definition_sha256": hashlib.sha256(
                    definition.encode("utf-8")
                ).hexdigest(),
            }
            row.payload = payload
            row.payload_sha256 = payload_sha256(payload)
            db.commit()
        records = tuple(
            _record(
                character="祁雾",
                evidence=line,
                polarity="negative",
                kind="action",
                dimension="core_personality",
                trait_key="directness",
                line=index,
                statement=line.rstrip("。"),
            )
            for index, line in enumerate(lines, start=1)
        )
        provider = QueueProvider(
            _response(),
            _response(*records),  # raw labels already match; still unapproved
            _response(*records),  # clean one-target call binds the axis
            json.dumps(
                {
                    "verdict": "contradicts",
                    "explanation": "连续两次用奉承话术回避直说，与基线相反。",
                    "citations": ["B1", "C1", "C2"],
                },
                ensure_ascii=False,
            ),
        )
        result = _run_stage(run_id, provider)
        assert result.diagnostics["counts"]["targeted_pass_scheduled_count"] == 1
        assert result.diagnostics["counts"]["targeted_pass_completed_count"] == 1
        assert result.diagnostics["counts"]["drift_reviewed"] == 1
        assert result.diagnostics["case_trace"][0]["matched_observation_count"] == 2
        assert result.diagnostics["case_trace"][0]["observation_axis_binding"] == (
            "server_targeted_evidence"
        )
        assert any(definition in prompt for _, prompt in provider.calls)
        assert all(axis_id not in prompt for _, prompt in provider.calls)


def _snapshot_stub(candidate_id: str, comparison_key: str | None = None):
    return SimpleNamespace(
        candidate_id=candidate_id,
        payload=(
            {"comparison_key": comparison_key}
            if comparison_key is not None
            else {}
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
            _snapshot_stub(
                f"candidate-{index:02}", f"preference:trait_{index:02}"
            ),
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
            _snapshot_stub("candidate-secret"),
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
            _snapshot_stub("active", "preference:active_key"),
            _confirmed_trait(
                trait_key="active_key",
                valid_from=1,
                valid_until=10,
            ),
            target_scope,
            "林澈",
        ),
        (
            _snapshot_stub("expired", "preference:expired_key"),
            _confirmed_trait(
                trait_key="expired_key",
                valid_from=1,
                valid_until=4,
            ),
            target_scope,
            "林澈",
        ),
        (
            _snapshot_stub("sibling", "preference:sibling_key"),
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
        _snapshot_stub("formal", "preference:蜜瓜"),
        _confirmed_trait(authority="formal_record"),
        scope,
        "林澈",
    )
    canon = (
        _snapshot_stub("canon", "preference:蜜瓜"),
        _confirmed_trait(authority="core_canon"),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([formal, canon])
    assert shadowed == 1
    assert [row[1].authority_tier for row in selected] == ["core_canon"]


def test_explicit_character_profile_shadows_conflicting_published_history():
    scope = NarrativeScopeV1()
    profile = (
        _snapshot_stub("profile", "preference:蜜瓜"),
        _confirmed_trait().model_copy(
            update={"id": "ct_profile", "origin": "explicit_setting"}
        ),
        scope,
        "林澈",
    )
    history = (
        _snapshot_stub("history", "preference:蜜瓜"),
        _confirmed_trait().model_copy(
            update={
                "id": "ct_history",
                "origin": "confirmed_history_inference",
                "statement": "林澈讨厌蜜瓜",
                "polarity": "negative",
            }
        ),
        scope,
        "林澈",
    )

    selected, shadowed = _select_authoritative_baselines([history, profile])

    assert [row[0].candidate_id for row in selected] == ["profile"]
    assert shadowed == 1
    # Two explicit settings need author resolution; one cannot silently
    # replace the other merely because they have the same authority tier.
    second_profile = (
        _snapshot_stub("other-profile", "preference:蜜瓜"),
        profile[1].model_copy(update={"id": "ct_other_profile"}),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([profile, second_profile])
    assert [row[0].candidate_id for row in selected] == [
        "profile",
        "other-profile",
    ]
    assert shadowed == 0


def test_object_bearing_baselines_require_an_explicit_same_object_axis():
    scope = NarrativeScopeV1()
    history = (
        _snapshot_stub("history", "preference:葡萄"),
        _confirmed_trait(trait_key="food_preference").model_copy(
            update={
                "origin": "confirmed_history_inference",
                "statement": "林澈喜欢葡萄",
                "evidence": (
                    EvidenceSpan(
                        document_id="history", document_name="history.md",
                        line_start=1, line_end=1, text="林澈喜欢葡萄。"
                    ),
                ),
            }
        ),
        scope,
        "林澈",
    )
    profile = (
        _snapshot_stub("profile", "preference:蜜瓜"),
        _confirmed_trait(trait_key="food_preference"),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([history, profile])
    assert selected == [history, profile]
    assert shadowed == 0
    assert not _baseline_shadowed_at_scope(history, selected, scope)

    explicit_history = (
        history[0],
        history[1].model_copy(update={"trait_key": "食物偏好:葡萄"}),
        scope,
        "林澈",
    )
    explicit_profile = (
        profile[0],
        profile[1].model_copy(update={"trait_key": "食物偏好:蜜瓜"}),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines(
        [explicit_history, explicit_profile]
    )
    assert selected == [explicit_history, explicit_profile]
    assert shadowed == 0
    legacy_history = (
        _snapshot_stub("legacy-history"),
        history[1].model_copy(update={"trait_key": "食物偏好:蜜瓜"}),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines(
        [legacy_history, explicit_profile]
    )
    assert selected == [legacy_history, explicit_profile]
    assert shadowed == 0

    def observation(key_object: str, signal_id: str) -> CharacterSignal:
        return CharacterSignal(
            id=f"cs_{signal_id * 32}",
            character="林澈",
            dimension="preference",
            trait_key="food_preference",
            statement=f"林澈喜欢{key_object}",
            polarity="positive",
            stability="stable",
            observation_kind="explicit_declaration",
            key_object=key_object,
            source_kind="draft",
            evidence=EvidenceSpan(
                document_id="draft", document_name="draft.md",
                line_start=1, line_end=1, text=f"林澈喜欢{key_object}。"
            ),
        )

    grape = observation("葡萄", "a")
    melon = observation("蜜瓜", "b")
    target = CharacterSignalTarget(
        character="林澈",
        dimension="preference",
        trait_key="food_preference",
        comparison_key="preference:蜜瓜",
        baseline_polarity="negative",
        requested_polarity="positive",
        baseline_hint="喜欢蜜瓜",
    )
    assert not _signal_matches_target(grape, target)
    assert _signal_matches_target(melon, target)
    assert _observation_matches_baseline(profile, grape) is False
    assert _observation_matches_baseline(profile, melon) is True
    assert _observation_matches_baseline(legacy_history, melon) is None


def test_qualified_preference_bridge_is_only_for_opposed_draft_review_gates():
    scope = NarrativeScopeV1()
    frozen = (
        _snapshot_stub("qualified-profile", "preference:冰镇蜜瓜"),
        _confirmed_trait(trait_key="melon_preference"),
        scope,
        "林澈",
    )
    legacy = (
        _snapshot_stub("legacy-profile"),
        frozen[1],
        scope,
        "林澈",
    )
    target = CharacterSignalTarget(
        character="林澈",
        dimension="preference",
        trait_key="melon_preference",
        comparison_key="preference:冰镇蜜瓜",
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="林澈喜欢冰镇蜜瓜",
    )
    direct = CharacterSignal(
        id="cs_" + "b" * 32,
        character="林澈",
        dimension="preference",
        trait_key="melon_preference",
        statement="林澈一直讨厌蜜瓜",
        polarity="negative",
        stability="temporary",
        observation_kind="preference_expression",
        key_object="蜜瓜",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft", document_name="draft.md",
            line_start=8, line_end=8, text="林澈一直讨厌蜜瓜。",
        ),
    )
    grape = direct.model_copy(
        update={
            "key_object": "葡萄",
            "evidence": direct.evidence.model_copy(update={"text": "林澈一直讨厌葡萄。"}),
        }
    )
    flavored_candy = direct.model_copy(
        update={
            "evidence": direct.evidence.model_copy(update={"text": "林澈一直讨厌蜜瓜味糖。"}),
        }
    )
    one_off = direct.model_copy(
        update={
            "observation_kind": "action",
            "evidence": direct.evidence.model_copy(update={"text": "林澈拒绝吃蜜瓜。"}),
        }
    )

    assert _signal_matches_target(direct, target)
    assert _target_has_sufficient_recall_evidence(target, (direct,))
    assert _target_with_existing_evidence_ranges(
        target, (direct,)
    ).existing_evidence_ranges == ((8, 8),)
    assert _observation_matches_baseline(frozen, direct) is True
    assert _observation_matches_baseline(legacy, direct) is None
    for unrelated in (grape, flavored_candy, one_off):
        assert not _signal_matches_target(unrelated, target)
        assert not _target_has_sufficient_recall_evidence(target, (unrelated,))
        assert _observation_matches_baseline(frozen, unrelated) is False


@pytest.mark.parametrize("dimension", ("value", "behavior_boundary", "current_state"))
def test_same_object_different_nonpreference_axes_keep_both_hints_and_never_cross_review(
    dimension: str,
):
    scope = NarrativeScopeV1()
    trust = (
        _snapshot_stub("trust", f"{dimension}:同伴"),
        _confirmed_trait(
            authority="formal_record", dimension=dimension, trait_key="companion_trust"
        ).model_copy(update={"statement": "林澈信任同伴"}),
        scope,
        "林澈",
    )
    protect = (
        _snapshot_stub("protect", f"{dimension}:同伴"),
        _confirmed_trait(
            authority="core_canon", dimension=dimension, trait_key="companion_protection"
        ).model_copy(update={"statement": "林澈会保护同伴"}),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([trust, protect])
    assert selected == [trust, protect]
    assert shadowed == 0

    draft = _FrozenDocument(
        input_id="draft-input",
        document=DocumentInput("draft", "draft.md", "林澈不再保护同伴。", "chapter"),
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
    hint = _safe_server_context(draft, baselines=selected)
    assert hint.eligible_traits == hint.included_traits == 2
    assert {target.trait_key for target in hint.targets} == {
        "companion_trust",
        "companion_protection",
    }

    observation = CharacterSignal(
        id="cs_" + "f" * 32,
        character="林澈",
        dimension=dimension,
        trait_key="companion_protection",
        statement="林澈不再保护同伴",
        polarity="negative",
        stability="stable",
        observation_kind="explicit_declaration",
        key_object="同伴",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft", document_name="draft.md",
            line_start=1, line_end=1, text="林澈不再保护同伴。"
        ),
    )
    assert _observation_matches_baseline(trust, observation) is False
    assert _observation_matches_baseline(protect, observation) is True
    assert not _signal_matches_target(
        observation, next(target for target in hint.targets if target.trait_key == "companion_trust")
    )
    assert _signal_matches_target(
        observation, next(target for target in hint.targets if target.trait_key == "companion_protection")
    )


def test_oversized_frozen_object_key_degrades_hint_without_stage_exception():
    scope = NarrativeScopeV1()
    frozen_key = "preference:" + "x" * (161 - len("preference:"))
    assert len(frozen_key) == 161
    entry = (
        _snapshot_stub("profile", frozen_key),
        _confirmed_trait(),
        scope,
        "林澈",
    )
    draft = _FrozenDocument(
        input_id="draft-input",
        document=DocumentInput(
            id="draft", name="draft.md", content="林澈喜欢蜜瓜。", role="chapter"
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
    hint = _safe_server_context(draft, baselines=[entry])
    assert hint.targets == ()
    assert hint.ambiguous_traits == 1
    assert json.loads(hint.payload)["confirmed_traits_coverage"]["state"] == "partial"


def test_same_comparison_axis_opposite_polarity_disables_targeted_hint():
    scope = NarrativeScopeV1()
    first = (
        SimpleNamespace(candidate_id="first"),
        _confirmed_trait(dimension="core_personality", trait_key="公开讲话"),
        scope,
        "林澈",
    )
    second = (
        SimpleNamespace(candidate_id="second"),
        _confirmed_trait(dimension="core_personality", trait_key="公开讲话能力").model_copy(
            update={"id": "ct_second", "polarity": "negative"}
        ),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([first, second])
    assert shadowed == 0
    assert selected == [first, second]
    draft = _FrozenDocument(
        input_id="draft-input",
        document=DocumentInput(
            id="draft", name="draft.md", content="林澈公开讲话。", role="chapter"
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
    hint = _safe_server_context(draft, baselines=selected)
    assert hint.targets == ()
    assert hint.ambiguous_traits == 1
    assert json.loads(hint.payload)["confirmed_traits_coverage"]["state"] == "partial"

    same_polarity = (
        second[0],
        second[1].model_copy(update={"polarity": "positive"}),
        scope,
        "林澈",
    )
    repeated = _safe_server_context(draft, baselines=[first, same_polarity])
    assert repeated.ambiguous_traits == 0
    assert len(repeated.targets) == 1


def test_profile_shadow_requires_overlapping_release_compatible_scope_and_axis():
    scope = NarrativeScopeV1.model_validate(
        {
            "timeline_key": "main",
            "branch": {"path": ["east"], "exclusive_group": "route"},
        }
    )
    sibling_scope = NarrativeScopeV1.model_validate(
        {
            "timeline_key": "main",
            "branch": {"path": ["west"], "exclusive_group": "route"},
        }
    )
    profile = (
        _snapshot_stub("profile", "preference:蜜瓜"),
        _confirmed_trait(valid_from=5, valid_until=10),
        scope,
        "林澈",
    )
    history = _confirmed_trait().model_copy(
        update={"origin": "confirmed_history_inference"}
    )
    entries = [
        profile,
        (
            _snapshot_stub("earlier-release", "preference:蜜瓜"),
            history.model_copy(
                update={
                    "id": "ct_earlier_release",
                    "valid_from_release_ordinal": 1,
                    "valid_until_release_ordinal": 4,
                }
            ),
            scope,
            "林澈",
        ),
        (
            _snapshot_stub("sibling-branch", "preference:蜜瓜"),
            history.model_copy(update={"id": "ct_sibling_branch"}),
            sibling_scope,
            "林澈",
        ),
        (
            _snapshot_stub("different-axis", "preference:航路抉择"),
            history.model_copy(
                update={"id": "ct_different_axis", "trait_key": "航路抉择"}
            ),
            scope,
            "林澈",
        ),
    ]

    selected, shadowed = _select_authoritative_baselines(entries)

    assert [row[0].candidate_id for row in selected] == [
        "profile",
        "earlier-release",
        "sibling-branch",
        "different-axis",
    ]
    assert shadowed == 0


def test_profile_shadow_uses_draft_release_without_erasing_earlier_history():
    scope = NarrativeScopeV1()
    history = (
        _snapshot_stub("history", "preference:蜜瓜"),
        _confirmed_trait(valid_from=1, valid_until=10).model_copy(
            update={"origin": "confirmed_history_inference", "polarity": "negative"}
        ),
        scope,
        "林澈",
    )
    profile = (
        _snapshot_stub("profile", "preference:蜜瓜"),
        _confirmed_trait(valid_from=5, valid_until=10),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([history, profile])
    assert selected == [history, profile]
    assert shadowed == 0

    early = NarrativeScopeV1.model_validate(
        {"release": {"key": "v3", "ordinal": 3}}
    )
    later = NarrativeScopeV1.model_validate(
        {"release": {"key": "v7", "ordinal": 7}}
    )
    assert not _baseline_shadowed_at_scope(history, selected, early)
    assert _baseline_shadowed_at_scope(history, selected, later)
    assert not _baseline_shadowed_at_scope(history, selected, scope)

    def hint_at(target: NarrativeScopeV1) -> list[str]:
        source = _FrozenDocument(
            input_id="draft-input",
            document=DocumentInput(
                id="draft", name="draft.md", content="林澈吃了蜜瓜。", role="chapter"
            ),
            document_version=1,
            content_sha256="0" * 64,
            ordinal=0,
            source_kind="draft",
            source_reason="draft",
            scope=target,
            resolution_state="confirmed",
            publication_status="draft",
            authority_tier="draft",
        )
        return [
            item.baseline_polarity
            for item in _safe_server_context(source, baselines=selected).targets
        ]

    assert hint_at(early) == ["negative"]
    assert hint_at(later) == ["positive"]
    assert hint_at(scope) == []


def test_profile_shadow_only_covers_its_branch_and_activity():
    global_scope = NarrativeScopeV1()
    east_scope = NarrativeScopeV1.model_validate(
        {"branch": {"path": ["east"], "exclusive_group": "route"}}
    )
    west_scope = NarrativeScopeV1.model_validate(
        {"branch": {"path": ["west"], "exclusive_group": "route"}}
    )
    event_scope = NarrativeScopeV1.model_validate(
        {"branch": {"path": ["east"], "exclusive_group": "route"},
         "activity_key": "festival"}
    )
    history = (
        _snapshot_stub("history", "preference:蜜瓜"),
        _confirmed_trait().model_copy(update={"origin": "confirmed_history_inference"}),
        global_scope,
        "林澈",
    )
    profile = (
        _snapshot_stub("profile", "preference:蜜瓜"),
        _confirmed_trait(),
        event_scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([history, profile])
    assert selected == [history, profile]
    assert shadowed == 0
    assert _baseline_shadowed_at_scope(history, selected, event_scope)
    assert not _baseline_shadowed_at_scope(history, selected, east_scope)
    assert not _baseline_shadowed_at_scope(history, selected, west_scope)
    assert not _baseline_shadowed_at_scope(history, selected, global_scope)


def test_contextual_profile_does_not_erase_history_in_another_context():
    scope = NarrativeScopeV1()
    history = (
        SimpleNamespace(candidate_id="history"),
        _confirmed_trait(dimension="contextual_behavior", trait_key="公开讲话").model_copy(
            update={"origin": "confirmed_history_inference", "contexts": ("公开场合",)}
        ),
        scope,
        "林澈",
    )
    profile = (
        SimpleNamespace(candidate_id="profile"),
        _confirmed_trait(dimension="contextual_behavior", trait_key="公开讲话").model_copy(
            update={"contexts": ("私下谈话",)}
        ),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines([history, profile])
    assert selected == [history, profile]
    assert shadowed == 0
    assert not _baseline_shadowed_at_scope(
        history, selected, scope, observation_context="公开场合"
    )
    assert not _baseline_shadowed_at_scope(
        history, selected, scope, observation_context="私下谈话"
    )
    draft = _FrozenDocument(
        input_id="draft-input",
        document=DocumentInput(
            id="draft", name="draft.md", content="林澈向众人说话。", role="chapter"
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
    hint = _safe_server_context(draft, baselines=selected)
    assert hint.targets == ()
    assert hint.ambiguous_traits == 1
    assert not hint.truncated
    assert json.loads(hint.payload)["confirmed_traits_coverage"] == {
        "state": "partial",
        "included": 0,
        "eligible": 1,
    }
    covering_profile = (
        profile[0],
        profile[1].model_copy(update={"contexts": ("公开场合", "私下谈话")}),
        scope,
        "林澈",
    )
    assert _baseline_shadowed_at_scope(
        history, [history, covering_profile], scope, observation_context="公开场合"
    )
    mixed_history = (
        history[0],
        history[1].model_copy(update={"contexts": ("公开场合", "私下谈话")}),
        scope,
        "林澈",
    )
    public_profile = (
        profile[0],
        profile[1].model_copy(update={"contexts": ("公开场合",)}),
        scope,
        "林澈",
    )
    selected, shadowed = _select_authoritative_baselines(
        [mixed_history, public_profile]
    )
    assert shadowed == 0
    assert _baseline_shadowed_at_scope(
        mixed_history, selected, scope, observation_context="公开场合"
    )
    assert not _baseline_shadowed_at_scope(
        mixed_history, selected, scope, observation_context="私下谈话"
    )


def test_shadowed_history_trait_still_supplies_published_growth_evidence():
    scope = NarrativeScopeV1()
    profile = _confirmed_trait(
        dimension="core_personality", trait_key="公开讲解意愿"
    )
    history = profile.model_copy(
        update={"id": "ct_history_growth", "origin": "confirmed_history_inference"}
    )
    selected, shadowed = _select_authoritative_baselines(
        [
            (SimpleNamespace(candidate_id="profile"), profile, scope, "林澈"),
            (SimpleNamespace(candidate_id="history"), history, scope, "林澈"),
        ]
    )
    assert [row[0].candidate_id for row in selected] == ["profile"]
    assert shadowed == 1

    published_history = _FrozenDocument(
        input_id="input-history",
        document=DocumentInput(
            id="history",
            name="history.md",
            content="训练后林澈逐渐改变了待人方式。",
            role="chapter",
        ),
        document_version=1,
        content_sha256="0" * 64,
        ordinal=1,
        source_kind="published_history",
        source_reason="history",
        scope=scope,
        resolution_state="confirmed",
        publication_status="published",
        authority_tier="formal_record",
    )
    support = _find_support_evidence(
        baseline=profile,
        baseline_scope=scope,
        draft_scopes=(scope,),
        draft_ordinals=(2,),
        draft_document_ids=("draft",),
        documents=[published_history],
        limit=8,
    )
    assert len(support) == 1
    assert support[0].kind == "causal_bridge"
    assert support[0].evidence.document_id == "history"


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
