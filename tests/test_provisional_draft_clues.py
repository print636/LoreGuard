from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.character_trait_extraction import (
    CharacterSignalChunk,
    CharacterSignalExtractor,
    CharacterSignal,
    _SignalValidationFailure,
    _ValidatedSignalPackage,
    _failed_package_result,
)
from app.character_consistency_stage import (
    PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY,
    ProvisionalCluesUnavailable,
    _FrozenDocument,
    _safe_provisional_draft_clues,
    project_provisional_draft_clues,
)
from app.auth import AuthContext, get_auth_context
from app.config import Settings
from app.db import (
    AnalysisDiagnosticRow, AnalysisRunInputRow, AnalysisRunRow, DocumentRow,
    LOCAL_USER_ID, SessionLocal,
)
from app.main import app
from app.domain import EvidenceSpan
from app.pipeline import DocumentInput


LINES = ("林澈一直喜欢蜜瓜。", "周尧一直讨厌青椒。")


class QueueProvider:
    def __init__(self, *responses: str | Exception):
        self.responses = list(responses)

    def complete(self, _system: str, _user: str):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response, prompt_tokens=17, completion_tokens=9)


def settings(**overrides: object) -> Settings:
    values = {
        "openai_api_key": "test-key",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "provider_max_attempts": 1,
        "character_signal_draft_trace_v1": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def chunk(lines: tuple[str, ...] = LINES) -> CharacterSignalChunk:
    return CharacterSignalChunk("doc-frozen", "draft.md", "\n".join(lines), 10, "draft")


def record(
    line: int = 10, *, character: str = "林澈", object_name: str = "蜜瓜",
    polarity: str = "positive", evidence: str | None = None,
) -> dict[str, object]:
    source = LINES[line - 10]
    return {
        "character": character,
        "dimension": "preference",
        "trait_key": f"食物偏好:{object_name}",
        "statement": source.rstrip("。"),
        "polarity": polarity,
        "stability": "stable",
        "observation_kind": "explicit_declaration",
        "context": "",
        "key_object": object_name,
        "source_line_start": line,
        "source_line_end": line,
        "evidence": source if evidence is None else evidence,
    }


def response(*records: dict[str, object]) -> str:
    return json.dumps({"records": records}, ensure_ascii=False)


def test_one_safe_partial_package_survives_failed_retry_without_formal_admission():
    valid = record()
    rejected = record(11, character="周尧", object_name="青椒",
                      polarity="negative", evidence="错误回显")
    result = CharacterSignalExtractor(
        QueueProvider(response(valid, rejected), response(rejected)),
        settings=settings(),
    ).extract(chunk())

    assert result.signals == result.draft_observations == result.pending_candidates == ()
    assert len(result.provisional_draft_clues) == 1
    clue = result.provisional_draft_clues[0]
    assert (clue.character, clue.evidence.line_start, clue.evidence.text) == (
        "林澈", 10, LINES[0]
    )
    assert "provisional_draft_clues" not in result.model_dump()
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.accepted_records == 0
    trace = result.diagnostics.draft_trace
    assert trace is not None
    assert trace.final_state == "no_clean_package"
    assert trace.final_accepted_record_ordinals == ()


def test_later_clean_package_has_priority_over_partial_clues():
    valid = record()
    rejected = record(11, character="周尧", object_name="青椒",
                      polarity="negative", evidence="错误回显")
    result = CharacterSignalExtractor(
        QueueProvider(response(valid, rejected), response(valid)),
        settings=settings(),
    ).extract(chunk())

    assert result.diagnostics.outcome == "completed"
    assert result.provisional_draft_clues == ()
    assert len(result.draft_observations) == 1
    assert result.diagnostics.draft_trace.final_state == "clean"


@pytest.mark.parametrize("later_package", [
    '{"records":[',
    response({"source_line_start": 11, "source_line_end": 11}),
    response({**record(11), "source_line_start": 999_999,
              "source_line_end": 999_999}),
])
def test_later_global_or_unknown_failure_clears_earlier_partial_clue(
    later_package: str,
):
    rejected = record(11, character="周尧", object_name="青椒",
                      polarity="negative", evidence="错误回显")
    result = CharacterSignalExtractor(
        QueueProvider(response(record(), rejected), later_package),
        settings=settings(),
    ).extract(chunk())
    assert result.provisional_draft_clues == ()
    assert result.signals == result.draft_observations == ()


def test_retry_without_model_response_keeps_single_safe_parsed_package():
    rejected = record(11, character="周尧", object_name="青椒",
                      polarity="negative", evidence="错误回显")
    result = CharacterSignalExtractor(
        QueueProvider(response(record(), rejected), RuntimeError("network failure")),
        settings=settings(),
    ).extract(chunk())
    assert len(result.provisional_draft_clues) == 1
    assert result.signals == result.draft_observations == ()
    assert result.diagnostics.draft_trace.final_state == "no_clean_package"


def test_other_parsed_attempt_only_vetoes_locally_bound_opposite_same_axis():
    rejected = record(11, character="周尧", object_name="青椒",
                      polarity="negative", evidence="错误回显")
    first_clue = CharacterSignalExtractor(
        QueueProvider(response(record(), rejected)),
        settings=settings(character_signal_package_max_attempts=1),
    ).extract(chunk()).provisional_draft_clues[0]
    unrelated = CharacterSignalExtractor(
        QueueProvider(response(record(11, character="周尧", object_name="青椒",
                                      polarity="negative"))),
        settings=settings(),
    ).extract(chunk()).signals[0]
    local_failure = (_SignalValidationFailure(1, "evidence_mismatch"),)
    first = _ValidatedSignalPackage(
        parsed=True, signals=(first_clue,),
        provisional_draft_clues=(first_clue,),
        reason_counts={"evidence_mismatch": 1}, failures=local_failure,
    )
    unrelated_second = _ValidatedSignalPackage(
        parsed=True, signals=(unrelated,),
        reason_counts={"evidence_mismatch": 1}, failures=local_failure,
    )
    kept = _failed_package_result(
        [first, unrelated_second], attempted_calls=2,
        prompt_tokens=0, completion_tokens=0, charged_tokens=0,
    )
    assert kept.provisional_draft_clues == (first_clue,)

    # The second signal stands for a separately bound opposing proposal from
    # another parsed attempt; an unbound rejected record above has no veto.
    opposite = first_clue.model_copy(update={
        "id": "cs_" + "f" * 32, "polarity": "negative",
    })
    conflicting_second = _ValidatedSignalPackage(
        parsed=True, signals=(opposite,),
        reason_counts={"evidence_mismatch": 1}, failures=local_failure,
    )
    removed = _failed_package_result(
        [first, conflicting_second], attempted_calls=2,
        prompt_tokens=0, completion_tokens=0, charged_tokens=0,
    )
    assert removed.provisional_draft_clues == ()


@pytest.mark.parametrize("bad_record", [
    {**record(), "evidence": "错误回显"},
    {**record(11), "source_line_start": 999_999,
     "source_line_end": 999_999},
    {key: value for key, value in record(11).items() if key != "character"},
    {**record(11), "requested_polarity": "positive"},
])
def test_overlap_unknown_coordinates_or_format_failure_cannot_retain_clue(
    bad_record: dict[str, object],
):
    result = CharacterSignalExtractor(
        QueueProvider(response(record(), bad_record)),
        settings=settings(character_signal_package_max_attempts=1),
    ).extract(chunk())
    assert result.provisional_draft_clues == ()
    assert result.signals == result.draft_observations == ()


def test_same_actor_same_axis_on_another_line_blocks_provisional_clue():
    lines = (LINES[0], "林澈今天仍然喜欢蜜瓜。")
    bad = {**record(), "source_line_start": 11, "source_line_end": 11,
           "statement": "林澈今天仍然喜欢蜜瓜", "evidence": "错误回显"}
    result = CharacterSignalExtractor(
        QueueProvider(response(record(), bad)),
        settings=settings(character_signal_package_max_attempts=1),
    ).extract(chunk(lines))
    assert result.provisional_draft_clues == ()


@pytest.mark.parametrize("response_kind", ["invalid_json", "oversize"])
def test_unparsed_or_oversize_package_is_closed(response_kind: str):
    response_text = (
        '{"records":[' if response_kind == "invalid_json" else "x" * 100_000
    )
    result = CharacterSignalExtractor(
        QueueProvider(response_text),
        settings=settings(character_signal_package_max_attempts=1),
    ).extract(chunk())
    assert result.provisional_draft_clues == ()
    assert result.diagnostics.draft_trace.final_state == "no_clean_package"


def frozen_source() -> _FrozenDocument:
    content = "\n".join(["前言。"] * 9 + list(LINES))
    return _FrozenDocument(
        input_id="input-frozen",
        document=DocumentInput("doc-frozen", "draft.md", content, role="chapter"),
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        ordinal=1,
        source_kind="draft",
        source_reason="draft",
        scope=None,
        resolution_state="confirmed",
        publication_status="draft",
        authority_tier="draft",
    )


def test_stage_projection_is_source_bound_and_suppresses_accepted_same_line_axis():
    source = frozen_source()
    rejected = record(11, character="周尧", object_name="青椒",
                      polarity="negative", evidence="错误回显")
    provisional = CharacterSignalExtractor(
        QueueProvider(response(record(), rejected)),
        settings=settings(character_signal_package_max_attempts=1),
    ).extract(chunk()).provisional_draft_clues[0]
    unrelated = CharacterSignalExtractor(
        QueueProvider(response(record(11, character="周尧", object_name="青椒",
                                      polarity="negative"))),
        settings=settings(),
    ).extract(chunk()).signals[0]
    same_line_axis = CharacterSignalExtractor(
        QueueProvider(response(record())), settings=settings()
    ).extract(chunk()).signals[0]

    clues, truncated = _safe_provisional_draft_clues(
        [(source, provisional)], accepted_signals=(unrelated,)
    )
    assert len(clues) == 1 and truncated is False
    assert clues[0].evidence == LINES[0]
    assert clues[0].proposed_statement == LINES[0].rstrip("。")
    assert _safe_provisional_draft_clues(
        [(source, provisional)], accepted_signals=(same_line_axis,)
    ) == ((), False)
    assert _safe_provisional_draft_clues(
        [(source, provisional.model_copy(update={
            "evidence": provisional.evidence.model_copy(update={"text": "伪造"})
        }))]
    ) == ((), False)


def test_compatible_opposite_axes_only_suppress_overlapping_source_lines():
    first_line = "林澈先主动和陌生人交谈，随后拒绝继续交谈。"
    second_line = "林澈拒绝和陌生人继续交谈。"
    content = "\n".join(["前言。"] * 9 + [first_line, second_line])
    source = replace(
        frozen_source(),
        document=DocumentInput("doc-frozen", "draft.md", content,
                               role="chapter"),
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    first = CharacterSignal(
        id="cs_" + "a" * 32, character="林澈",
        dimension="contextual_behavior", trait_key="social_initiative",
        statement="林澈主动和陌生人交谈", polarity="positive",
        stability="situational", observation_kind="interaction",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="doc-frozen", document_name="draft.md",
            line_start=10, line_end=10, text=first_line,
        ),
    )
    opposite_same_line = first.model_copy(update={
        "id": "cs_" + "b" * 32,
        "trait_key": "social_initiative_pattern",
        "statement": "林澈拒绝继续交谈",
        "polarity": "negative",
    })
    opposite_other_line = opposite_same_line.model_copy(update={
        "evidence": EvidenceSpan(
            document_id="doc-frozen", document_name="draft.md",
            line_start=11, line_end=11, text=second_line,
        ),
    })
    assert _safe_provisional_draft_clues([
        (source, first), (source, opposite_same_line),
    ]) == ((), False)
    distinct, truncated = _safe_provisional_draft_clues([
        (source, first), (source, opposite_other_line),
    ])
    assert [item.line_start for item in distinct] == [10, 11]
    assert truncated is False

    # Final accepted draft observations veto a conflicting provisional axis
    # even when their source spans do not overlap.
    assert _safe_provisional_draft_clues(
        [(source, first)], accepted_signals=(opposite_other_line,)
    ) == ((), False)

    same_direction_other_line = opposite_other_line.model_copy(update={
        "id": "cs_" + "c" * 32,
        "polarity": "positive",
        "statement": "林澈继续主动和陌生人交谈",
    })
    same_direction, truncated = _safe_provisional_draft_clues(
        [(source, first)], accepted_signals=(same_direction_other_line,)
    )
    assert [item.line_start for item in same_direction] == [10]
    assert truncated is False

    for offset, non_opposing in enumerate(("neutral", "unclear"), start=1):
        accepted = opposite_other_line.model_copy(update={
            "id": "cs_" + str(offset) * 32,
            "polarity": non_opposing,
        })
        retained, truncated = _safe_provisional_draft_clues(
            [(source, first)], accepted_signals=(accepted,)
        )
        assert [item.line_start for item in retained] == [10]
        assert truncated is False


def test_private_projection_validates_immutable_input_and_rejects_tampering():
    source = frozen_source()
    rejected = record(11, character="周尧", object_name="青椒",
                      polarity="negative", evidence="错误回显")
    provisional = CharacterSignalExtractor(
        QueueProvider(response(record(), rejected)),
        settings=settings(character_signal_package_max_attempts=1),
    ).extract(chunk()).provisional_draft_clues[0]
    clues, _ = _safe_provisional_draft_clues([(source, provisional)])
    private = {"items": [clues[0].model_dump(mode="json")], "truncated": False}
    frozen_row = AnalysisRunInputRow(
        id=source.input_id, run_id="run-1", document_id=source.document.id,
        document_name=source.document.name,
        document_version=source.document_version,
        content=source.document.content,
        content_sha256=source.content_sha256,
        ordinal=1,
    )
    public = project_provisional_draft_clues(private, [frozen_row])
    assert public["items"][0]["id"].startswith("pc_")
    assert public["items"][0]["evidence"] == LINES[0]
    assert "input_id" not in public["items"][0]
    assert "content_sha256" not in public["items"][0]
    assert project_provisional_draft_clues(
        {"items": [], "truncated": False}, [frozen_row]
    ) == {"items": [], "truncated": False}
    bad = {"items": [{**private["items"][0], "evidence": "伪造"}],
           "truncated": False}
    with pytest.raises(ProvisionalCluesUnavailable):
        project_provisional_draft_clues(bad, [frozen_row])
    with pytest.raises(ProvisionalCluesUnavailable):
        project_provisional_draft_clues(private, [])


def test_provisional_endpoint_is_isolated_from_diagnostics_export_and_live_document():
    content = "\n".join(["前言。"] * 9 + list(LINES))
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"待复核线索-{uuid4().hex}"}
        ).json()
        created = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={
                "name": "draft.md", "document_role": "chapter", "content": content,
                "narrative_context": {
                    "resolution_state": "confirmed",
                    "publication_status": "draft",
                    "scope": {"timeline_key": "main"},
                },
            },
        )
        assert created.status_code == 201, created.text
        doc_id = created.json()["id"]
        with patch("app.main.dispatch_analysis"):
            created_run = client.post(
                f"/api/v1/projects/{project['id']}/analysis-runs"
            )
        assert created_run.status_code == 202, created_run.text
        run_id = created_run.json()["id"]
        endpoint = f"/api/v1/analysis-runs/{run_id}/provisional-clues"
        assert client.get(endpoint).status_code == 409
        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            run.status = "completed"
            db.commit()
        assert client.get(endpoint).json() == {"items": [], "truncated": False}

        actual_chunk = CharacterSignalChunk(
            doc_id, "draft.md", "\n".join(LINES), 10, "draft"
        )
        rejected = record(11, character="周尧", object_name="青椒",
                          polarity="negative", evidence="错误回显")
        provisional = CharacterSignalExtractor(
            QueueProvider(response(record(), rejected)),
            settings=settings(character_signal_package_max_attempts=1),
        ).extract(actual_chunk).provisional_draft_clues[0]
        with SessionLocal() as db:
            frozen = db.query(AnalysisRunInputRow).filter_by(run_id=run_id).one()
            source = _FrozenDocument(
                input_id=frozen.id,
                document=DocumentInput(doc_id, frozen.document_name, frozen.content,
                                       role="chapter"),
                document_version=frozen.document_version,
                content_sha256=frozen.content_sha256,
                ordinal=frozen.ordinal,
                source_kind="draft", source_reason="draft", scope=None,
                resolution_state="confirmed", publication_status="draft",
                authority_tier="draft",
            )
            clues, truncated = _safe_provisional_draft_clues([(source, provisional)])
            assert len(clues) == 1 and not truncated
            db.add(AnalysisDiagnosticRow(
                run_id=run_id,
                payload={
                    "character_consistency": {
                        "outcome": "partial", "material_coverage": "partial"
                    },
                    PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY: {
                        "items": [clues[0].model_dump(mode="json")],
                        "truncated": False,
                    },
                },
            ))
            db.commit()

        shown = client.get(endpoint)
        assert shown.status_code == 200, shown.text
        item = shown.json()["items"][0]
        assert item["evidence"] == LINES[0]
        assert item["proposed_statement"] == LINES[0].rstrip("。")
        assert item["reason"] == "partial_model_package"
        assert not {"issue", "severity", "confidence", "category", "feedback"} & set(item)
        assert shown.headers["cache-control"] == "private, no-store"
        diagnostics = client.get(f"/api/v1/analysis-runs/{run_id}/diagnostics")
        assert diagnostics.status_code == 200
        assert PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY not in diagnostics.json()
        assert LINES[0] not in diagnostics.text
        assert item["proposed_statement"] not in diagnostics.text
        exported = client.get(f"/api/v1/analysis-runs/{run_id}/export.md")
        assert exported.status_code == 200, exported.text
        assert LINES[0] not in exported.text
        assert item["proposed_statement"] not in exported.text

        with SessionLocal() as db:
            live = db.get(DocumentRow, doc_id)
            live.content = "活文档已经修改。"
            live.version += 1
            db.commit()
        assert client.get(endpoint).json() == shown.json()

        app.dependency_overrides[get_auth_context] = lambda: AuthContext(
            user_id=LOCAL_USER_ID, workspace_id=str(uuid4()),
            role="owner", anonymous=True,
        )
        try:
            assert client.get(endpoint).status_code == 404
        finally:
            app.dependency_overrides.pop(get_auth_context, None)

        with SessionLocal() as db:
            diagnostic = db.get(AnalysisDiagnosticRow, run_id)
            damaged = dict(diagnostic.payload)
            damaged[PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY] = {
                "items": [{**clues[0].model_dump(mode="json"), "evidence": "伪造"}],
                "truncated": False,
            }
            diagnostic.payload = damaged
            db.commit()
        assert client.get(endpoint).status_code == 409

        with SessionLocal() as db:
            diagnostic = db.get(AnalysisDiagnosticRow, run_id)
            restored = dict(diagnostic.payload)
            restored[PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY] = {
                "items": [clues[0].model_dump(mode="json")], "truncated": False,
            }
            diagnostic.payload = restored
            frozen = db.query(AnalysisRunInputRow).filter_by(run_id=run_id).one()
            frozen.content = "冻结输入被篡改。"
            db.commit()
        assert client.get(endpoint).status_code == 409
