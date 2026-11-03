"""Actual model-contract/stage receipt producers through the sealed consumer.

Only original synthetic text and a mock provider are used, without gold,
live databases, model calls, scorer changes or manufactured event receipts.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from app.character_consistency_stage import (
    _FrozenDocument, _safe_case_trace, _target_coverage_certificate, _target_observation_pool, _to_issue,
)
from app.character_drift import (
    CHARACTER_REVIEW_POLICY_VERSION, CharacterConsistencyReviewer,
    CharacterDriftCase, SupportEvidence, prepare_character_drift, promote_character_drift,
)
from app.character_trait_extraction import CharacterSignal
from app.config import Settings
from app.domain import EvidenceSpan
from app.pipeline import DocumentInput
from scripts import ooc_eval_contract as contract
from scripts import run_ooc_sealed_http as runner
from scripts.ooc_sealed_prediction import project_case_prediction
from tests.test_character_consistency_stage import _confirmed_trait
from tests.test_ooc_sealed_http_runner import SuccessfulApi, _sha, _uuid, _write_bundle


def _event(*handles, verdict="contradicts", **overrides):
    value = dict(current_citations=list(handles), verdict=verdict,
                 explanation="该事件原文可定位。", actor_match="same", trait_match="same",
                 actuality="actual", object_match="not_applicable", situation_match="not_applicable")
    value.update(overrides)
    return value


def _fixture(tmp_path, *, count=1, events=None, citations=None, verdict="contradicts",
             explanation_coverage="complete", medical_support=False):
    baseline_text = "林澈一直喜欢蜜瓜。\n"
    target_text = "\n".join(("林澈表示讨厌蜜瓜。", "林澈把蜜瓜推回盘边。", "林澈再次拒绝蜜瓜。")[:count]) + "\n"
    if medical_support:
        target_text = "林澈仍说自己喜欢蜜瓜，但依照今日医嘱暂时把蜜瓜推回盘边。\n"
    bundle, case, setup = _write_bundle(tmp_path / "public-bundle", baseline_text=baseline_text, target_text=target_text)
    selector = setup.candidate_selector.model_copy(update={"trait_type": "preference"})
    setup = setup.model_copy(update={"candidate_selector": selector, "author_axis": None})
    case = case.model_copy(update={"axis": case.axis.model_copy(update={"sha256": runner._canonical_value_sha({
        "candidate_selector": selector.model_dump(mode="json"), "author_axis": None,
    })})})
    api = SuccessfulApi(bundle)
    public_ids = (selector.source_document_id, setup.target_document_id)
    runtime_ids = (api.baseline_runtime_id, api.target_runtime_id)
    source_documents = dict(zip(public_ids, (baseline_text, target_text)))
    runtime_map = dict(zip(public_ids, runtime_ids))
    document_rows = list(zip(public_ids, runtime_ids, ("formal_character_profile", "draft")))
    if medical_support:
        history_id, history_runtime_id = _uuid(8), _uuid(107)
        history = "医嘱规定林澈今日须暂避蜜瓜，治疗结束后可恢复原有饮食。\n"
        history_path = f"cases/{case.group_id}/{history_id}.md"
        (tmp_path / "public-bundle" / history_path).write_text(history, encoding="utf-8", newline="\n")
        case = case.model_copy(update={
            "documents": (*case.documents, contract.DocumentSnapshot(
                document_id=history_id, path=history_path, version=1, sha256=_sha(history))),
            "evidence_catalog": (*case.evidence_catalog, contract.EvidenceAnchor(
                evidence_id=_uuid(997), document_id=history_id, line_start=1, line_end=1)),
        })
        setup = setup.model_copy(update={"documents": (
            setup.documents[0], runner.PublicDocumentSetup(
                document_id=history_id, phase="baseline", document_role="chapter", story_scope="main",
                resolution_state="confirmed", publication_status="published", import_order=2,
            ), setup.documents[1].model_copy(update={"import_order": 3}),
        )})
        source_documents[history_id] = history
        runtime_map[history_id] = history_runtime_id
        document_rows.append((history_id, history_runtime_id, "published_history"))
    sources = {}
    for i, (public_id, runtime_id, kind) in enumerate(document_rows, 1):
        text = source_documents[public_id]
        sources[runtime_id] = _FrozenDocument(
            input_id=_uuid(200 + i), document=DocumentInput(
                id=runtime_id, name=f"{public_id}.md", content=text,
                role="character_profile" if i == 1 else "chapter"),
            document_version=1, content_sha256=_sha(text), ordinal=i,
            source_kind=kind, source_reason=kind, scope=None,
            resolution_state="draft" if kind == "draft" else "confirmed",
            publication_status="draft" if kind == "draft" else "published",
            authority_tier="draft" if kind == "draft" else "formal_record",
        )

    def span(runtime_id, line):
        document = sources[runtime_id].document
        return EvidenceSpan(document_id=runtime_id, document_name=document.name,
                            line_start=line, line_end=line, text=document.content.splitlines()[line - 1])

    baseline = _confirmed_trait().model_copy(update={
        "id": f"ct_{api.candidate_id}", "evidence": (span(runtime_ids[0], 1),),
    })
    observations = tuple(CharacterSignal(
        id=f"cs_{i:032x}", character="林澈", dimension="preference", trait_key=baseline.trait_key,
        statement=span(runtime_ids[1], i).text, key_object="蜜瓜", polarity="negative",
        stability="temporary", observation_kind="action", source_kind="draft",
        evidence=span(runtime_ids[1], i),
    ) for i in range(1, count + 1))
    supports = ()
    if medical_support:
        supports = (SupportEvidence(
            id="se_synthetic_medical", kind="exception", summary="已发布且当前生效的临时饮食医嘱。",
            explicit=True, evidence=span(history_runtime_id, 1), source_kind="published_history",
            publication_status="published", authority_tier="formal_record", resolution_state="confirmed",
            source_ordinal=1, applicable_observation_ids=tuple(row.id for row in observations),
            selection_basis="semantic_relation_v1",
        ),)
    prepared = prepare_character_drift(CharacterDriftCase(
        id="cdc_single_event_public_synthetic", baseline=baseline, observations=observations,
        support_evidence=supports, scope_compatibility="compatible", material_coverage="complete",
        explanation_coverage=explanation_coverage,
    ), review_policy=CHARACTER_REVIEW_POLICY_VERSION)
    response = dict(policy_version=CHARACTER_REVIEW_POLICY_VERSION,
                    verdict=verdict, explanation="依冻结设定与该次行为作出判断。",
                    citations=list(citations or ("B01",) + tuple(f"C{i:02d}" for i in range(1, count + 1))),
                    events=events or [_event(*(f"C{i:02d}" for i in range(1, count + 1)))])

    class Provider:
        def complete(self, system, user):
            return SimpleNamespace(text=json.dumps(response, ensure_ascii=False), prompt_tokens=17, completion_tokens=23)

    review = CharacterConsistencyReviewer(Provider(), settings=Settings(
        _env_file=None, openai_api_key="unit-placeholder", enable_character_consistency=True,
    )).review(prepared)
    assert review.diagnostics.outcome == "completed"
    promoted = promote_character_drift(prepared, review)
    trace = _safe_case_trace(
        character_key=baseline.character, baseline=baseline,
        baseline_entry=(SimpleNamespace(candidate_id=api.candidate_id), baseline, None, baseline.character),
        matched_observation_count=count, matched_observations=prepared.matching_observations,
        prepared=prepared, prepare_reason=prepared.reason, material_coverage="complete",
        review=review, final_outcome=promoted.outcome, visible=promoted.visible,
        promote_reason=promoted.reason, frozen_by_document=sources,
    )
    trace["target_observation_pool"] = _target_observation_pool(observations, sources)
    processing = dict(planned_chunks=len(sources), processed_chunks=len(sources), model_completed_chunks=len(sources),
                      model_incomplete_chunks=0, model_uncalled_chunks=0)
    trace["target_coverage"] = _target_coverage_certificate(
        trace, baseline=baseline, unresolved_alias_observations=(), formal_signals=(),
        frozen_by_document=sources, target_aliases=(), processing=processing,
        pre_alias_partial=False, baseline_limit=False,
        explanation_processing_partial=explanation_coverage != "complete",
    )
    formal, clues = [], []
    if promoted.visible:
        issue = _to_issue(promoted=promoted, prepared=prepared, review=review,
                          confirmed_candidate_id=api.candidate_id, judgement=promoted.outcome,
                          frozen_by_document=sources, comparison_key=trace["comparison_key"])
        row = {**issue.model_dump(mode="json"), "report_class": "formal" if promoted.outcome == "conflict" else "review_clue"}
        (formal if promoted.outcome == "conflict" else clues).append(row)
    diagnostics = {
        "runtime_provenance": {"character_consistency_limits": {
            "drift_review_policy_version": CHARACTER_REVIEW_POLICY_VERSION,
        }},
        "character_consistency": {"outcome": "completed" if explanation_coverage == "complete" else "partial",
                                  "material_coverage": "complete" if explanation_coverage == "complete" else "partial",
                                  "counts": processing, "reason_counts": {}, "case_trace": [trace]},
    }
    return SimpleNamespace(case=case, setup=setup, api=api, trace=trace, diagnostics=diagnostics,
                           formal=formal, clues={"items": clues, "truncated": False, "unavailable_count": 0, "scan_limited": False},
                           source_documents=source_documents, runtime_map=runtime_map, promoted=promoted)


def _consume(fixture):
    return runner._snapshot_from_reports(
        fixture.case, fixture.setup, fixture.api.candidate_id, {"status": "completed"},
        fixture.diagnostics, fixture.formal, fixture.clues, {"items": [], "truncated": False},
        contract.StageFunnel(input_candidates=1, retrieved_candidates=1, grounded_candidates=1,
                             adjudicated_candidates=1, surfaced_candidates=1),
        assessment_policy=contract.EVENT_ASSESSMENT_POLICY,
        source_documents=fixture.source_documents, runtime_document_ids=fixture.runtime_map,
    )


def _prediction(fixture):
    return project_case_prediction(fixture.case, _consume(fixture), assessment_policy=contract.EVENT_ASSESSMENT_POLICY)


def test_one_reviewed_action_reaches_formal_prediction_with_actual_receipt(tmp_path):
    fixture = _fixture(tmp_path)
    snapshot = _consume(fixture)
    assert fixture.promoted.outcome == "conflict"
    assert _prediction(fixture).outcome == "conflict"
    assert [ref.role for ref in snapshot.citations] == ["B", "C"]
    assert snapshot.citations[1].event_group_id == fixture.trace["reviewed_events"]["events"][0]["event_group_id"]
    assert snapshot.citations == snapshot.report_evidence
    assert "evidence_binding" not in snapshot.model_dump(mode="json")


def test_two_fragments_share_server_group_instead_of_manufactured_independent_ids(tmp_path):
    fixture = _fixture(tmp_path, count=2)
    snapshot = _consume(fixture)
    assert _prediction(fixture).outcome == "conflict"
    assert len({ref.event_group_id for ref in snapshot.citations if ref.role == "C"}) == 1
    # The legacy two-event policy still cannot receive new independent events.
    assert project_case_prediction(fixture.case, snapshot).outcome == "unassessed"


def test_only_actual_selected_clear_event_is_projected(tmp_path):
    fixture = _fixture(tmp_path, count=2, events=[_event("C01"), _event("C02", verdict="needs_confirmation", actuality="unclear")],
                       citations=("B01", "C01"))
    assert _prediction(fixture).outcome == "conflict"
    assert fixture.trace["matched_observation_count"] == 2
    assert len(fixture.trace["reviewed_events"]["events"]) == 1


def test_response_order_differs_from_report_storage_order_without_losing_binding(tmp_path):
    fixture = _fixture(tmp_path, citations=("C01", "B01"))
    snapshot = _consume(fixture)
    assert _prediction(fixture).outcome == "conflict"
    assert [ref.role for ref in snapshot.citations] == ["C", "B"]
    assert snapshot.citations == snapshot.report_evidence


def test_fully_reviewed_outside_comparison_can_be_no_issue_without_report(tmp_path):
    fixture = _fixture(tmp_path, verdict="explained", events=[_event("C01", verdict="needs_confirmation", object_match="different")])
    assert fixture.promoted.outcome == "no_issue"
    assert not fixture.formal and not fixture.clues["items"]
    assert _prediction(fixture).outcome == "no_issue"


@pytest.mark.parametrize("mutation", [
    "missing_receipt", "missing_policy", "stripped_new_markers", "bad_group_hash", "missing_member",
    "unclear_actor", "wrong_version", "wrong_text_hash", "wrong_runtime_id", "wrong_report_text",
    "wrong_response_order", "no_public_bytes", "changed_public_bytes", "bad_uuid_hash", "bad_event_enum_type",
])
def test_new_policy_receipt_or_source_tampering_fails_closed(tmp_path, mutation):
    fixture = _fixture(tmp_path)
    receipt = fixture.trace["reviewed_events"]
    metadata = fixture.formal[0]["metadata"]
    if mutation == "missing_receipt":
        del fixture.trace["reviewed_events"]
        del metadata["reviewed_events"]
    elif mutation == "missing_policy":
        del fixture.trace["review_policy"]
        del metadata["review_policy"]
    elif mutation == "stripped_new_markers":
        for record in (fixture.trace, metadata):
            del record["review_policy"]
            del record["reviewed_events"]
    elif mutation == "bad_group_hash":
        receipt["events"][0]["event_group_id"] = "f" * 64
    elif mutation == "missing_member":
        receipt["events"][0]["current_citations"] = []
    elif mutation == "unclear_actor":
        receipt["events"][0]["actor_match"] = "unclear"
    elif mutation == "wrong_version":
        receipt["evidence_refs"][1]["document_version"] = 2
    elif mutation == "wrong_text_hash":
        receipt["evidence_refs"][1]["text_sha256"] = "e" * 64
    elif mutation == "wrong_runtime_id":
        fixture.formal[0]["evidence"][1]["document_id"] = _uuid(555)
    elif mutation == "wrong_report_text":
        fixture.formal[0]["evidence"][1]["text"] = "原文没有这句。"
    elif mutation == "wrong_response_order":
        metadata["review_citation_refs"].reverse()
    elif mutation == "no_public_bytes":
        fixture.source_documents = None
    elif mutation == "changed_public_bytes":
        fixture.source_documents[fixture.setup.target_document_id] += "增加一行。"
    elif mutation == "bad_uuid_hash":
        receipt["evidence_refs"][1]["source_input_id_sha256"] = "not-a-hash"
    elif mutation == "bad_event_enum_type":
        receipt["events"][0]["verdict"] = ["contradicts"]
    if "reviewed_events" in fixture.trace:
        metadata["reviewed_events"] = deepcopy(fixture.trace["reviewed_events"])
    assert _prediction(fixture).outcome == "unassessed"


def test_no_issue_cannot_omit_a_reviewed_current_group(tmp_path):
    fixture = _fixture(tmp_path, count=2, verdict="explained", events=[
        _event("C01", "C02", verdict="needs_confirmation", object_match="different"),
    ])
    receipt = fixture.trace["reviewed_events"]
    receipt["evidence_refs"] = receipt["evidence_refs"][:2]
    event = receipt["events"][0]
    event["current_citations"] = ["C01"]
    event["event_group_id"] = runner._canonical_value_sha({
        "schema_version": receipt["schema_version"],
        "confirmed_candidate_id_sha256": receipt["confirmed_candidate_id_sha256"],
        "comparison_key": receipt["comparison_key"], "members": receipt["evidence_refs"][1:],
    })
    fixture.trace["citation_refs"] = fixture.trace["citation_refs"][:2]
    assert _prediction(fixture).outcome == "unassessed"


def test_in_scope_explained_cannot_reuse_outside_receipt_without_actual_support(tmp_path):
    fixture = _fixture(tmp_path, verdict="explained", events=[
        _event("C01", verdict="needs_confirmation", object_match="different"),
    ])
    event = fixture.trace["reviewed_events"]["events"][0]
    event.update(verdict="explained", object_match="same", situation_match="same")
    assert not {ref["role"] for ref in fixture.trace["reviewed_events"]["evidence_refs"]} & {"G", "X"}
    assert _prediction(fixture).outcome == "unassessed"


@pytest.mark.parametrize("coverage", ["not_run", "partial"])
def test_verified_scope_dependency_preserves_raw_coverage_and_projects_no_issue(tmp_path, coverage):
    fixture = _fixture(tmp_path, verdict="explained", explanation_coverage=coverage, events=[
        _event("C01", verdict="explained", actor_match="unclear", object_match="different"),
    ])
    snapshot = _consume(fixture)
    assert fixture.trace["target_coverage"]["schema_version"] == "character-target-coverage-v2"
    assert fixture.trace["explanation_coverage"] == snapshot.explanation_coverage == coverage
    assert snapshot.stage_status == "completed"
    proof = snapshot.explanation_requirement
    assert proof is not None and proof.state == "not_required_by_scope" and proof.observation_count == 1
    assert proof.reviewed_events_sha256 == runner._canonical_value_sha(fixture.trace["reviewed_events"])
    assert _prediction(fixture).outcome == "no_issue"
    assert fixture.diagnostics["character_consistency"]["outcome"] == "partial"


@pytest.mark.parametrize("mutation", [
    "missing_pool", "pool_hash", "receipt_hash", "extra_observation", "unreviewed",
    "processing", "post_alias_gap", "other_explanation_gap", "unresolved_alias", "downgraded_v1",
])
def test_scope_requirement_cannot_bypass_full_pool_or_other_processing_gaps(tmp_path, mutation):
    fixture = _fixture(tmp_path, verdict="explained", explanation_coverage="not_run", events=[
        _event("C01", verdict="explained", object_match="different"),
    ])
    trace = fixture.trace
    cert = trace["target_coverage"]
    if mutation == "missing_pool":
        del trace["target_observation_pool"]
    elif mutation == "pool_hash":
        cert["explanation_requirement"]["target_observation_pool_sha256"] = "f" * 64
    elif mutation == "receipt_hash":
        cert["explanation_requirement"]["reviewed_events_sha256"] = "f" * 64
    elif mutation == "extra_observation":
        pool = trace["target_observation_pool"]
        pool["observation_refs"].append({**pool["observation_refs"][0], "observation_id_sha256": "f" * 64})
        pool["observation_count"] = 2
        cert["explanation_requirement"]["target_observation_pool_sha256"] = runner._canonical_value_sha(pool)
    elif mutation == "unreviewed":
        trace["unreviewed_observation_count"] = 1
    elif mutation == "processing":
        cert["processing"]["model_incomplete_chunks"] = 1
        fixture.diagnostics["character_consistency"]["counts"]["model_incomplete_chunks"] = 1
    elif mutation == "post_alias_gap":
        cert["non_alias_blockers"] = ["post_alias_processing_gap"]
    elif mutation == "other_explanation_gap":
        cert["non_alias_blockers"] = ["other_case_explanation_incomplete"]
    elif mutation == "unresolved_alias":
        cert["alias_disposition"].update(total=1, unknown=1)
        fixture.diagnostics["character_consistency"]["reason_counts"] = {"unmatched_character_alias": 1}
    elif mutation == "downgraded_v1":
        cert["schema_version"] = "character-target-coverage-v1"
        del cert["explanation_requirement"]
    assert _prediction(fixture).outcome == "unassessed"


@pytest.mark.parametrize("outcome,surface", [("conflict", "formal"), ("needs_confirmation", "review")])
def test_scope_requirement_is_never_a_formal_or_clue_coverage_waiver(tmp_path, outcome, surface):
    fixture = _fixture(tmp_path, verdict="explained", explanation_coverage="not_run", events=[
        _event("C01", verdict="explained", object_match="different"),
    ])
    snapshot = _consume(fixture)
    assert snapshot.explanation_requirement is not None
    tampered = snapshot.model_copy(update={
        "final_outcome": outcome, "formal_issue_count": int(surface == "formal"),
        "review_clue_count": int(surface == "review"), "report_evidence": snapshot.citations,
    })
    assert project_case_prediction(fixture.case, tampered, assessment_policy=contract.EVENT_ASSESSMENT_POLICY).outcome == "unassessed"


def test_medical_support_can_explain_clear_in_scope_event_with_actual_x_source(tmp_path):
    fixture = _fixture(tmp_path, verdict="explained", medical_support=True,
                       events=[_event("C01", verdict="explained")], citations=("B01", "C01", "X01"))
    snapshot = _consume(fixture)
    assert snapshot.explanation_requirement is None
    assert "explanation_requirement" not in snapshot.model_dump(mode="json")
    assert snapshot.explanation_coverage == "complete"
    assert [ref.role for ref in snapshot.citations] == ["B", "C", "X"]
    assert _prediction(fixture).outcome == "no_issue"


def test_real_x_support_cannot_wash_unclear_in_scope_actor_into_no_issue(tmp_path):
    fixture = _fixture(tmp_path, verdict="explained", medical_support=True,
                       events=[_event("C01", verdict="explained")], citations=("B01", "C01", "X01"))
    fixture.trace["reviewed_events"]["events"][0]["actor_match"] = "unclear"
    assert _prediction(fixture).outcome == "unassessed"
