from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_scope_review_provider import SCOPE_REVIEW_USER_PREFIX
from app.character_consistency_stage import CharacterConsistencyStage, _FrozenDocument
from app.character_trait_extraction import (
    CharacterSignalChunk,
    CharacterSignalExtractor,
    build_pending_trait_candidates,
)
from app.config import Settings
from app.db import AnalysisRunRow, CharacterTraitCandidateRow, SessionLocal
from app.main import app
from app.pipeline import DocumentInput
from app.service import _load_verified_snapshot


def _settings(**updates) -> Settings:
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "provider_max_attempts": 1,
        "character_signal_package_max_attempts": 2,
        "character_signal_full_line_prompt_v2": True,
        "character_signal_support_id_v4": True,
        "character_signal_semantic_scope_v5": True,
        "character_signal_scope_review_v1": True,
        "character_signal_support_trace_v1": True,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def _record(
    line: str,
    *,
    support_id: str,
    statement: str,
    actor_anchor_id: str = "",
    label_anchor_id: str = "",
    scope_relation: str = "local",
    character: str = "桑衍",
    dimension: str = "preference",
    trait_key: str = "melon_preference",
    key_object: str = "蜜瓜",
    stability: str = "stable",
) -> dict:
    return {
        "character": character,
        "dimension": dimension,
        "trait_key": trait_key,
        "statement": statement,
        "polarity": "positive",
        "stability": stability,
        "observation_kind": "explicit_declaration",
        "context": "",
        "key_object": key_object,
        "source_line_start": 2,
        "source_line_end": 2,
        "evidence": line,
        "support_id": support_id,
        "actor_anchor_id": actor_anchor_id,
        "label_anchor_id": label_anchor_id,
        "scope_relation": scope_relation,
    }


class ReviewingProvider:
    def __init__(self, records: list[dict], verdicts: list[dict] | None = None, *, raw_review: str | None = None):
        self.records = records
        self.verdicts = verdicts or []
        self.raw_review = raw_review
        self.calls: list[tuple[str, str]] = []
        self.review_payload: dict | None = None

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        if not user.startswith(SCOPE_REVIEW_USER_PREFIX):
            return SimpleNamespace(
                text=json.dumps({"records": self.records}, ensure_ascii=False),
                prompt_tokens=17,
                completion_tokens=9,
            )
        self.review_payload = json.loads(user[len(SCOPE_REVIEW_USER_PREFIX):])
        if self.raw_review is not None:
            response = self.raw_review
        else:
            request = self.review_payload["request"]
            items = []
            for ordinal, proposal in enumerate(request["proposals"]):
                line = next(
                    row for row in request["lines"]
                    if proposal["support_id"] in {
                        clause["support_id"] for clause in row["clauses"]
                    }
                )
                clauses = {clause["support_id"]: clause for clause in line["clauses"]}
                target = clauses[proposal["support_id"]]
                anchor_ids = (
                    proposal.get("actor_anchor_id"), proposal.get("label_anchor_id")
                )
                first_offset = min(
                    [target["start_offset"]]
                    + [clauses[anchor_id]["start_offset"] for anchor_id in anchor_ids if anchor_id]
                )
                exact_basis = [
                    clause["support_id"] for clause in line["clauses"]
                    if first_offset <= clause["start_offset"] <= target["start_offset"]
                ]
                item = {
                    "proposal_id": proposal["proposal_id"],
                    "support_id": proposal["support_id"],
                    "verdict": "supported",
                    "actor": "proposed",
                    "actuality": "asserted",
                    "label_relation": "same_axis" if proposal["label_anchor_id"] else "none",
                    "object_relation": "same" if proposal["key_object"] else "not_applicable",
                    "polarity_relation": "same",
                    "statement_relation": "supported",
                    "level_supported": "yes",
                    "basis_ids": exact_basis,
                }
                if ordinal < len(self.verdicts):
                    item.update(self.verdicts[ordinal])
                items.append(item)
            response = json.dumps({
                "schema_version": request["schema_version"],
                "request_digest": self.review_payload["request_digest"],
                "items": items,
            }, ensure_ascii=False)
        return SimpleNamespace(text=response, prompt_tokens=13, completion_tokens=11)


def _extract(line: str, provider: ReviewingProvider, *, settings: Settings | None = None):
    frozen = "## 桑衍\n" + line + "\n"
    source = ScopeReviewSourceIdentity(
        run_input_id="frozen-input-1",
        document_id="document-1",
        document_version=3,
        content_sha256=hashlib.sha256(frozen.encode("utf-8")).hexdigest(),
    )
    chunk = CharacterSignalChunk("document-1", "profile.md", line, 2, "formal_character_profile")
    result = CharacterSignalExtractor(provider, settings=settings or _settings()).extract(
        chunk, source_identity=source, frozen_content=frozen,
    )
    return result, provider


def test_nonliteral_same_axis_continuation_reaches_review_and_can_be_pending():
    line = (
        "桑衍的核心性格是让搭档事先知道风险，"
        "在日常航路调整中，她先向搭档说明可能危及航船的情况。"
    )
    record = _record(
        line, support_id="L2:A3", actor_anchor_id="L2:A1",
        label_anchor_id="L2:A1", scope_relation="labelled_elaboration",
        statement="桑衍先向搭档说明可能危及航船的情况",
        dimension="core_personality", stability="core", key_object="航船",
        trait_key="risk_disclosure",
    )
    result, provider = _extract(line, ReviewingProvider([record]))

    assert result.diagnostics.outcome == "completed"
    assert result.diagnostics.attempted_calls == 2
    assert len(result.signals) == len(result.pending_candidates) == 1
    assert result.diagnostics.support_trace.final_accepted_slots == (3,)
    request = provider.review_payload["request"]
    assert request["run_input_id"] == "frozen-input-1"
    assert request["document_version"] == 3
    assert request["block_line_start"] == 2
    assert request["lines"][0]["text"] == line
    assert [row["support_id"] for row in request["lines"][0]["clauses"]] == [
        "L2:A1", "L2:A2", "L2:A3",
    ]
    assert request["lines"][0]["clauses"][2]["text"] == "她先向搭档说明可能危及航船的情况"


@pytest.mark.parametrize("line, record, conflict", [
    (
        "桑衍相信周尧喜欢蜜瓜。",
        {"support_id": "L2:A1", "statement": "桑衍喜欢蜜瓜"},
        {"actor": "other"},
    ),
    (
        "桑衍长期喜欢热茶，周尧喜欢蜜瓜，她常带一块蜜瓜。",
        {
            "support_id": "L2:A3", "actor_anchor_id": "L2:A1",
            "scope_relation": "same_actor_continuation",
            "statement": "桑衍常带一块蜜瓜",
        },
        {"actor": "other"},
    ),
    (
        "桑衍的核心性格是告知搭档风险，她会把桌面擦干净。",
        {
            "support_id": "L2:A2", "actor_anchor_id": "L2:A1",
            "label_anchor_id": "L2:A1", "scope_relation": "labelled_elaboration",
            "statement": "桑衍会把桌面擦干净", "dimension": "core_personality",
            "stability": "core", "key_object": "桌面", "trait_key": "desk_cleaning",
        },
        {"label_relation": "different_axis"},
    ),
    (
        "桑衍长期稳定偏好热茶，她也喜欢蜜瓜。",
        {
            "support_id": "L2:A2", "actor_anchor_id": "L2:A1",
            "label_anchor_id": "L2:A1", "scope_relation": "labelled_elaboration",
            "statement": "桑衍也喜欢蜜瓜",
        },
        {"label_relation": "different_axis"},
    ),
])
def test_four_p1_semantic_conflicts_do_not_reach_pending(line, record, conflict):
    proposed = _record(line, **record)
    result, provider = _extract(
        line, ReviewingProvider([proposed], [{"verdict": "rejected", **conflict}])
    )

    assert len(provider.calls) == 2  # Semantic disagreement cannot regenerate.
    assert result.diagnostics.outcome == "partial"
    assert result.signals == result.pending_candidates == ()
    assert result.diagnostics.reason_counts == {"scope_review_reviewer_rejected": 1}
    assert result.diagnostics.support_trace.final_accepted_slots == ()


def test_mixed_package_retains_only_supported_and_tracks_uncertain():
    line = "桑衍喜欢热茶，她也喜欢蜜瓜。"
    first = _record(
        line, support_id="L2:A1", statement="桑衍喜欢热茶",
        key_object="热茶", trait_key="tea_preference",
    )
    second = _record(
        line, support_id="L2:A2", actor_anchor_id="L2:A1",
        scope_relation="same_actor_continuation", statement="桑衍喜欢蜜瓜",
    )
    result, provider = _extract(
        line, ReviewingProvider([first, second], [{}, {"verdict": "uncertain", "actor": "ambiguous"}])
    )

    assert len(provider.calls) == 2
    assert result.diagnostics.outcome == "partial"
    assert [signal.key_object for signal in result.signals] == ["热茶"]
    assert len(result.pending_candidates) == 1
    assert result.diagnostics.reason_counts == {"scope_review_reviewer_uncertain": 1}
    assert result.diagnostics.support_trace.final_accepted_slots == (1,)


def test_contradicted_statement_cannot_pass_even_with_correct_actor_and_object():
    line = "桑衍喜欢蜜瓜。"
    proposed = _record(line, support_id="L2:A1", statement="桑衍讨厌蜜瓜")
    result, provider = _extract(
        line,
        ReviewingProvider(
            [proposed],
            [{"verdict": "rejected", "statement_relation": "contradicted"}],
        ),
    )
    assert len(provider.calls) == 2
    assert result.diagnostics.outcome == "partial"
    assert result.signals == result.pending_candidates == ()
    assert result.diagnostics.reason_counts == {"scope_review_reviewer_rejected": 1}


def test_invalid_review_and_frozen_source_mismatch_fail_closed():
    line = "桑衍喜欢蜜瓜。"
    record = _record(line, support_id="L2:A1", statement="桑衍喜欢蜜瓜")
    invalid, provider = _extract(line, ReviewingProvider([record], raw_review='{"items":[]}'))
    assert invalid.signals == invalid.pending_candidates == ()
    assert invalid.diagnostics.outcome == "partial"
    assert invalid.diagnostics.charged_tokens > 0
    assert invalid.diagnostics.reason_counts == {"scope_review_response_invalid": 1}
    assert len(provider.calls) == 2

    mismatched_source = ScopeReviewSourceIdentity(
        run_input_id="frozen-input-1", document_id="document-1",
        document_version=1,
        content_sha256=hashlib.sha256("different".encode()).hexdigest(),
    )
    provider = ReviewingProvider([record])
    skipped = CharacterSignalExtractor(provider, settings=_settings()).extract(
        CharacterSignalChunk("document-1", "profile.md", line, 2, "formal_character_profile"),
        source_identity=mismatched_source,
        frozen_content="## 桑衍\n" + line + "\n",
    )
    assert skipped.diagnostics.outcome == "skipped"
    assert skipped.diagnostics.reason_counts == {"scope_review_source_mismatch": 1}
    assert provider.calls == []


def test_review_reserve_blocks_extraction_before_spending_it():
    line = "桑衍喜欢蜜瓜。"
    provider = ReviewingProvider([_record(line, support_id="L2:A1", statement="桑衍喜欢蜜瓜")])
    result, provider = _extract(
        line, provider,
        settings=_settings(character_signal_token_budget=5_000),
    )
    assert result.diagnostics.outcome == "skipped"
    assert result.diagnostics.reason_counts == {"token_budget": 1}
    assert result.diagnostics.token_admission.available_tokens == 0
    assert provider.calls == []


def test_structural_regeneration_finishes_before_single_batch_review():
    line = "桑衍喜欢蜜瓜。"
    record = _record(line, support_id="L2:A1", statement="桑衍喜欢蜜瓜")

    class FirstInvalid(ReviewingProvider):
        initial = True

        def complete(self, system: str, user: str):
            if self.initial and not user.startswith(SCOPE_REVIEW_USER_PREFIX):
                self.initial = False
                self.calls.append((system, user))
                return SimpleNamespace(
                    text='{"records":[{"wrong":"schema"}]}',
                    prompt_tokens=17, completion_tokens=9,
                )
            return super().complete(system, user)

    result, provider = _extract(line, FirstInvalid([record]))
    assert len(provider.calls) == 3
    assert provider.calls[2][1].startswith(SCOPE_REVIEW_USER_PREFIX)
    assert result.diagnostics.attempted_calls == 3
    assert result.diagnostics.outcome == "completed"
    assert result.diagnostics.reason_counts == {"regenerated_from_schema_validation": 1}


def test_provider_failure_charges_review_estimate_and_abstains():
    line = "桑衍喜欢蜜瓜。"
    record = _record(line, support_id="L2:A1", statement="桑衍喜欢蜜瓜")

    class FailingReview(ReviewingProvider):
        def complete(self, system: str, user: str):
            if user.startswith(SCOPE_REVIEW_USER_PREFIX):
                self.calls.append((system, user))
                raise RuntimeError("review unavailable")
            return super().complete(system, user)

    result, provider = _extract(line, FailingReview([record]))
    assert len(provider.calls) == 2
    assert result.diagnostics.attempted_calls == 2
    assert result.diagnostics.outcome == "partial"
    assert result.signals == result.pending_candidates == ()
    assert result.diagnostics.reason_counts == {"scope_review_provider_error": 1}
    assert result.diagnostics.charged_tokens > result.diagnostics.prompt_tokens + result.diagnostics.completion_tokens


def test_flag_dependencies_and_optional_extended_deadline():
    with pytest.raises(ValueError, match="requires semantic scope v5"):
        _settings(character_signal_semantic_scope_v5=False)
    with pytest.raises(ValueError, match="shorter than scope review timeout"):
        _settings(character_signal_timeout_seconds=5, character_signal_total_deadline_seconds=20)
    assert _settings(character_signal_total_deadline_seconds=120).character_signal_total_deadline_seconds == 120


def test_stage_passes_frozen_run_identity_and_keeps_rejected_local_out_of_candidates():
    line = "桑衍相信周尧喜欢蜜瓜。"
    frozen = "## 桑衍\n" + line + "\n"
    document = DocumentInput("document-1", "profile.md", frozen, role="character_profile")
    provider = ReviewingProvider(
        [_record(line, support_id="L2:A1", statement="桑衍喜欢蜜瓜")],
        [{"verdict": "rejected", "actor": "other"}],
    )
    stage = CharacterConsistencyStage(settings=_settings(), provider=provider)
    source = _FrozenDocument(
        input_id="server-frozen-row-9",
        document=document,
        document_version=7,
        content_sha256=hashlib.sha256(frozen.encode("utf-8")).hexdigest(),
        ordinal=1,
        source_kind="formal_character_profile",
        source_reason="formal_character_profile",
        scope=None,
        resolution_state="confirmed",
        publication_status="published",
        authority_tier="formal_character_profile",
    )
    stage._bind_frozen_documents = lambda *args, **kwargs: [source]
    stage._load_confirmed_traits = lambda *args, **kwargs: []

    result = stage.run(
        None, run_id="run-1", project_id="project-1", documents=[document],
        metadata=[], remaining_run_tokens=30_000,
    )

    assert result.diagnostics["outcome"] == "partial"
    assert result.diagnostics["counts"]["pending_candidate_count"] == 0
    assert result.diagnostics["reason_counts"]["scope_review_reviewer_rejected"] == 1
    assert result.diagnostics["usage"]["attempted_calls"] == 2
    assert provider.review_payload["request"]["run_input_id"] == "server-frozen-row-9"
    assert provider.review_payload["request"]["document_version"] == 7


@pytest.mark.parametrize("strip_refs", (False, True))
def test_stage_persists_same_line_reviewed_targets_as_separate_formal_candidates(
    strip_refs: bool,
):
    line = "桑衍喜欢蜜瓜，桑衍也喜欢蜜瓜。"
    first = _record(line, support_id="L2:A1", statement="桑衍喜欢蜜瓜")
    second = _record(line, support_id="L2:A2", statement="桑衍也喜欢蜜瓜")
    provider = ReviewingProvider([first, second])
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/projects", json={"name": f"分句审阅-{uuid4().hex}"}
        )
        assert created.status_code == 201, created.text
        project_id = created.json()["id"]
        document = client.post(
            f"/api/v1/projects/{project_id}/documents/text",
            json={
                "name": "profile.md",
                "content": "## 桑衍\n" + line + "\n",
                "document_role": "character_profile",
                "narrative_context": {
                    "resolution_state": "confirmed",
                    "publication_status": "published",
                    "scope": {"timeline_key": "main"},
                },
            },
        )
        assert document.status_code == 201, document.text
        with patch("app.main.dispatch_analysis"):
            created_run = client.post(f"/api/v1/projects/{project_id}/analysis-runs")
        assert created_run.status_code == 202, created_run.text
        run_id = created_run.json()["id"]
        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            assert run is not None
            run.status = "running"
            db.commit()
            documents, metadata = _load_verified_snapshot(db, run_id)
            builder_override = (
                patch(
                    "app.character_consistency_stage.build_pending_trait_candidates",
                    side_effect=lambda signals: tuple(
                        row.model_copy(update={"support_refs": ()})
                        for row in build_pending_trait_candidates(signals)
                    ),
                ) if strip_refs else nullcontext()
            )
            with builder_override:
                result = CharacterConsistencyStage(
                    settings=_settings(), provider=provider
                ).run(
                    db,
                    run_id=run_id,
                    project_id=project_id,
                    documents=documents,
                    metadata=metadata,
                    remaining_run_tokens=30_000,
                )
            db.commit()
            rows = list(db.scalars(select(CharacterTraitCandidateRow).where(
                CharacterTraitCandidateRow.source_run_id == run_id
            )).all())

    assert result.diagnostics["counts"]["signal_count"] == 2
    assert result.diagnostics["counts"]["pending_candidate_count"] == 2
    if strip_refs:
        assert result.diagnostics["outcome"] == "partial"
        assert result.diagnostics["counts"]["persisted_created"] == 0
        assert result.diagnostics["counts"]["persisted_failed"] == 2
        assert result.diagnostics["reason_counts"]["candidate_persistence"] == 2
        assert rows == []
        return
    assert result.diagnostics["counts"]["persisted_created"] == 2
    assert len(rows) == 2
    assert len({row.candidate_fingerprint for row in rows}) == 2
    assert {row.support_bindings_v1["bindings"][0]["support_id"] for row in rows} == {
        "L2:A1", "L2:A2",
    }
    assert all(row.support_binding_mode == "required_v1" for row in rows)
