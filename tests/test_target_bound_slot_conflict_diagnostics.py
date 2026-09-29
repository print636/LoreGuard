from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.character_trait_extraction as extraction_module
from app.character_draft_actor_review import (
    TargetBoundDraftReviewBatchEvaluation,
    TargetBoundDraftReviewDecision,
    TargetBoundDraftReviewEvaluation,
    target_bound_draft_review_batch_digest,
    target_bound_draft_review_request_digest,
)
from app.character_draft_actor_review_provider import TargetBoundDraftReviewRun
from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_trait_extraction import (
    CharacterSignalChunk,
    CharacterSignalDiagnostics,
    CharacterSignalExtractor,
    CharacterSignalTarget,
    stable_trait_identity,
)
from app.config import Settings


class _EmptySignalProvider:
    def complete(self, _system: str, _user: str):
        return SimpleNamespace(
            text=json.dumps({"records": []}),
            prompt_tokens=4,
            completion_tokens=2,
        )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="slot-conflict-test",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock",
        enable_character_consistency=True,
        character_signal_full_line_prompt_v2=True,
        character_draft_actor_review_v1=True,
        character_target_bound_draft_review_v3=True,
        character_signal_package_max_attempts=1,
        character_signal_max_attempts=1,
        provider_max_attempts=1,
    )


def test_target_bound_evaluation_slot_conflicts_reach_internal_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
):
    content = "沈砚明确表示自己讨厌栗子糕。"
    source = ScopeReviewSourceIdentity(
        run_input_id="private-run-input",
        document_id="private-draft",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    target = CharacterSignalTarget(
        character="沈砚",
        dimension="preference",
        trait_key="food_preference",
        comparison_key=stable_trait_identity(
            "preference", "food_preference", "甜食"
        ),
        key_object="甜食",
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="沈砚长期喜欢甜食",
    )

    def fake_review(entries, **_kwargs):
        requests = tuple(entry.request for entry in entries)
        evaluations = tuple(
            TargetBoundDraftReviewEvaluation(
                request_digest=target_bound_draft_review_request_digest(request),
                decisions=tuple(
                    TargetBoundDraftReviewDecision(
                        proposal_id=proposal.proposal_id,
                        verdict="uncertain",
                        reason="slot_conflict",
                        slot_conflicts=("actor", "object_span"),
                    )
                    for proposal in request.proposals
                ),
            )
            for request in requests
        )
        return TargetBoundDraftReviewRun(
            evaluation=TargetBoundDraftReviewBatchEvaluation(
                batch_digest=target_bound_draft_review_batch_digest(requests),
                evaluations=evaluations,
            ),
            estimated_tokens=20,
            attempted_calls=1,
            prompt_tokens=8,
            completion_tokens=4,
            charged_tokens=20,
            failure_reason=None,
        )

    monkeypatch.setattr(
        extraction_module, "run_target_bound_draft_review", fake_review
    )
    result = CharacterSignalExtractor(
        _EmptySignalProvider(), settings=_settings()
    ).extract_targeted(
        CharacterSignalChunk(
            document_id="private-draft",
            document_name="private.md",
            content=content,
            global_line_start=1,
            source_kind="draft",
        ),
        (target,),
        target_ordinal=1,
        source_identity=source,
        frozen_content=content,
    )

    assert result.diagnostics.outcome == "partial"
    assert result.diagnostics.reason_counts["target_bound_slot_conflict"] == 1
    assert result.diagnostics.target_bound_slot_conflict_counts == {
        "actor": 1,
        "object_span": 1,
    }
    serialized = result.diagnostics.model_dump(mode="json")
    assert "target_bound_slot_conflict_counts" not in serialized
    assert content not in repr(
        result.diagnostics.target_bound_slot_conflict_counts
    )


def test_target_bound_diagnostic_keys_are_the_closed_evaluator_enum():
    with pytest.raises(ValidationError):
        CharacterSignalDiagnostics(
            outcome="partial",
            attempted_calls=1,
            raw_records=0,
            accepted_records=0,
            rejected_records=0,
            target_bound_slot_conflict_counts={"private story": 1},
        )
