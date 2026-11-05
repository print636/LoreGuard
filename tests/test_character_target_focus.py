"""Synthetic contract/source tests, not a claim about real model accuracy."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.character_drift import (
    CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT, CHARACTER_REVIEW_POLICY_VERSION,
    CharacterConsistencyReviewer, CharacterDriftCase, ConfirmedTraitSnapshot,
    prepare_character_drift, promote_character_drift, target_focus_review_receipt,
    with_target_focus,
)
from app.character_target_focus import (
    TARGET_FOCUS_POLICY_V1, TARGET_FOCUS_REVIEW_KEY, TARGET_FOCUS_SIGNAL_INSTRUCTIONS,
)
from app.character_trait_extraction import CharacterSignal
from app.config import Settings
from app.domain import EvidenceSpan
from tests.test_character_single_event_policy import event, payload


PROFILE = "陶宁接客前总会先核对预约，从不透露委托内容；这是长期习惯。"
STATEMENT = "陶宁接客前总会先核对预约"
OTHER_STATEMENT = "陶宁从不透露委托内容"


def focused(text="陶宁向来客透露了委托内容。", *, count=1, bound=True):
    span = EvidenceSpan(document_id="profile", document_name="profile.md",
                        line_start=1, line_end=1, text=PROFILE)
    bindings = {
        "schema_version": "character-support-bindings-v1", "index_version": "assertion-index-v1",
        "bindings": [{"evidence_index": 0, "support_id": "L1:A1",
                      "target": {"support_id": "L1:A1", "start_offset": 0,
                                 "end_offset": len(STATEMENT), "role": "target"}}],
    }
    baseline = ConfirmedTraitSnapshot(
        id="ct_synthetic_focus", character="陶宁", dimension="core_personality",
        trait_key="preparation", statement=STATEMENT, polarity="positive", stability="core",
        contexts=("接客前",), origin="explicit_setting", evidence=(span,),
        target_support_bindings_v1=bindings if bound else None,
    )
    observations = tuple(CharacterSignal(
        id=f"cs_{i:032x}", character="陶宁", dimension="core_personality", trait_key="preparation",
        statement=text, polarity="negative", stability="temporary", observation_kind="action",
        source_kind="draft", evidence=EvidenceSpan(document_id="draft", document_name="draft.md",
                                                   line_start=i, line_end=i, text=text),
    ) for i in range(1, count + 1))
    candidate = prepare_character_drift(CharacterDriftCase(
        id="cdc_synthetic_focus", baseline=baseline, observations=observations,
        scope_compatibility="compatible", material_coverage="complete", explanation_coverage="complete",
    ), review_policy=CHARACTER_REVIEW_POLICY_VERSION)
    return with_target_focus(candidate, TARGET_FOCUS_POLICY_V1)


def focused_response(request, *, verdict="contradicts", relation="contradicts", basis="explicit"):
    """A fixture responder used by the native service/policy tests too."""
    current = [row for row in request["evidence"] if row["id"].startswith("C")]
    events = []
    for row in current:
        check = {
            "target_relation": relation, "opposition_basis": basis,
            "counterevidence": ([{"citation": row["id"], "quote": row["text"][:240]}]
                                if relation == "contradicts" else []),
        }
        events.append(event(row["id"], verdict=verdict, target_focus=check))
    return payload(*events, verdict=verdict,
                   citations=("B01", *(row["id"] for row in current)),
                   target_focus_sha256=request["target_focus_sha256"])


class FocusProvider:
    def __init__(self, transform=lambda response: response, **response_options):
        self.calls = []
        self.transform, self.options = transform, response_options

    def complete(self, system, user):
        request = json.loads(user)
        self.calls.append((system, request))
        response = self.transform(focused_response(request, **self.options))
        return SimpleNamespace(text=json.dumps(response, ensure_ascii=False), prompt_tokens=40, completion_tokens=80)


def run_review(candidate, provider):
    settings = Settings(_env_file=None, enable_character_consistency=True,
                        character_drift_review_max_generations=1)
    settings._character_target_focus_policy = TARGET_FOCUS_POLICY_V1
    return CharacterConsistencyReviewer(provider, settings=settings).review(candidate)


def test_same_line_neighbor_is_context_not_target_and_omission_is_not_formal():
    candidate = focused()
    provider = FocusProvider(verdict="needs_confirmation", relation="not_established", basis="description_absence")
    result = run_review(candidate, provider)
    system, request = provider.calls[0]
    assert system == CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT
    assert request["target_focus"]["statement"] == STATEMENT
    assert [x["text"] for x in request["target_focus"]["target_assertions"]] == [STATEMENT]
    assert "从不透露委托内容" in [x["text"] for x in request["target_focus"]["surrounding_assertions"]]
    assert request["evidence"][0]["text"] == PROFILE
    assert "否认" in system and "合理语义推断" in system
    assert promote_character_drift(candidate, result).outcome == "needs_confirmation"
    receipt = target_focus_review_receipt(candidate, result)
    assert receipt["events"][0]["opposition_basis"] == "description_absence"
    assert PROFILE not in json.dumps(receipt, ensure_ascii=False)
    assert len(provider.calls) == 1


@pytest.mark.parametrize(("text", "basis"), [
    ("陶宁没有核对预约就接待了来客。", "explicit"),
    ("陶宁尚不清楚来客是否约好时间，便开始了接待。", "supported_inference"),
])
def test_clear_counterexample_and_semantic_paraphrase_can_still_be_formal(text, basis):
    candidate = focused(text)
    provider = FocusProvider(basis=basis)
    result = run_review(candidate, provider)
    assert promote_character_drift(candidate, result).outcome == "conflict"
    receipt = target_focus_review_receipt(candidate, result)
    ref = receipt["events"][0]["counterevidence_refs"][0]
    assert text[ref["start_offset"]:ref["end_offset"]] == text
    assert len(provider.calls) == 1


@pytest.mark.parametrize("bad", ["neighbor", "omission", "borrowed_source", "invented_quote", "wrong_target"])
def test_new_contract_cannot_promote_without_target_and_current_source_alignment(bad):
    candidate = focused("陶宁没有核对预约就接待了来客。")

    def transform(value):
        check = value["events"][0]["target_focus"]
        if bad == "neighbor":
            check.update(target_relation="different", opposition_basis="none", counterevidence=[])
        elif bad == "omission":
            check["opposition_basis"] = "description_absence"
        elif bad == "borrowed_source":
            check["counterevidence"][0]["citation"] = "C02"
        elif bad == "invented_quote":
            check["counterevidence"][0]["quote"] = "原文中不存在的行为"
        else:
            value["target_focus_sha256"] = "f" * 64
        return value

    provider = FocusProvider(transform)
    result = run_review(candidate, provider)
    assert result.decision is None
    assert promote_character_drift(candidate, result).outcome != "conflict"
    assert target_focus_review_receipt(candidate, result) is None
    assert len(provider.calls) == 1


def test_legacy_candidate_keeps_exact_confirmed_statement_and_full_context_without_guessed_spans():
    candidate = focused("陶宁没有核对预约就接待了来客。", bound=False)
    provider = FocusProvider()
    result = run_review(candidate, provider)
    focus = provider.calls[0][1]["target_focus"]
    assert focus["binding_status"] == "statement_only" and focus["target_assertions"] == []
    assert focus["statement"] == STATEMENT
    assert provider.calls[0][1]["evidence"][0]["text"] == PROFILE
    assert promote_character_drift(candidate, result).outcome == "conflict"


def test_target_coordinates_are_rebuilt_not_trusted_from_frozen_json_alone():
    candidate = focused()
    binding = copy.deepcopy(candidate.case.baseline.target_support_bindings_v1)
    binding["bindings"][0]["target"]["end_offset"] += 1
    changed = candidate.model_copy(update={"case": candidate.case.model_copy(update={
        "baseline": candidate.case.baseline.model_copy(update={"target_support_bindings_v1": binding})})})
    with pytest.raises(ValueError, match="offsets"):
        with_target_focus(changed, TARGET_FOCUS_POLICY_V1)


def test_old_policy_accepts_old_response_and_rejects_new_fields():
    from tests.test_character_single_event_policy import prepared, review
    old = prepared()
    result, _ = review(old, payload(event("C01")))
    assert promote_character_drift(old, result).outcome == "conflict"
    response = payload(event("C01", target_focus={"target_relation": "contradicts",
        "opposition_basis": "explicit", "counterevidence": []}), target_focus_sha256="f" * 64)
    result, _ = review(old, response, character_drift_review_max_generations=1)
    assert result.decision is None and target_focus_review_receipt(old, result) is None


def test_focus_response_reserve_honors_explicit_smaller_cap_before_provider_call():
    candidate = focused("陶宁没有核对预约就接待了来客。")
    provider = FocusProvider()
    settings = Settings(_env_file=None, enable_character_consistency=True,
                        character_drift_max_completion_tokens=900)
    settings._character_target_focus_policy = TARGET_FOCUS_POLICY_V1
    result = CharacterConsistencyReviewer(provider, settings=settings).review(candidate)
    assert result.diagnostics.reason == "response_contract_budget"
    assert provider.calls == []


def test_native_stage_uses_confirmed_statement_in_primary_review_and_public_explanation():
    from app.main import app, write_limiter
    from app.db import SessionLocal, AnalysisRunRow
    from app.service import _load_verified_snapshot
    from app.character_consistency_stage import CharacterConsistencyStage
    from tests.test_character_consistency_stage import _settings
    from tests.test_character_single_event_stage import _single_action_run, _RuntimeSingleEventProvider

    class NativeProvider(_RuntimeSingleEventProvider):
        def complete(self, system, user):
            if system == CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT:
                self.calls.append((system, user))
                return SimpleNamespace(text=json.dumps(focused_response(json.loads(user)), ensure_ascii=False),
                                       prompt_tokens=40, completion_tokens=80)
            return super().complete(system, user)

    write_limiter.events.clear()
    with TestClient(app) as client:
        run_id, provider = _single_action_run(client, NativeProvider)
        config = _settings(character_explanation_review_v1=True)
        config._character_target_focus_policy = TARGET_FOCUS_POLICY_V1
        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            documents, metadata = _load_verified_snapshot(db, run_id)
            result = CharacterConsistencyStage(settings=config, provider=provider).run(
                db, run_id=run_id, project_id=run.project_id, documents=documents,
                metadata=metadata, remaining_run_tokens=60_000,
            )
            db.commit()
    assert len(result.issues) == 1 and not result.review_clues
    issue = result.issues[0]
    focus_call = next(json.loads(user) for system, user in provider.calls
                      if system == CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT)
    assert issue.explanation.startswith("本次对照设定：" + focus_call["target_focus"]["statement"])
    receipt = issue.metadata[TARGET_FOCUS_REVIEW_KEY]
    assert receipt == result.diagnostics["case_trace"][0][TARGET_FOCUS_REVIEW_KEY]
    primary = next(user for system, user in provider.calls if system.endswith(TARGET_FOCUS_SIGNAL_INSTRUCTIONS))
    assert "confirmed_target_statement" in primary
    assert focus_call["target_focus"]["statement"] in primary
