"""Synthetic backend producers exercised through the sealed HTTP consumer.

No gold, model, live service or developer database is used. Certificates come
from the actual backend helpers, including their frozen source verification.
"""
from __future__ import annotations

from types import SimpleNamespace

import httpx

from app.character_consistency_stage import (
    _FrozenDocument,
    _safe_case_trace,
    _target_coverage_certificate,
    _to_issue,
)
from app.character_drift import (
    CharacterDriftCase,
    prepare_character_drift,
    promote_character_drift,
)
from app.character_trait_extraction import CharacterSignal
from app.domain import EvidenceSpan
from app.pipeline import DocumentInput
from scripts import ooc_eval_contract as contract
from scripts import run_ooc_sealed_http as runner
from scripts.ooc_sealed_prediction import project_case_prediction
from tests.test_character_consistency_stage import _confirmed_trait
from tests.test_ooc_sealed_http_runner import (
    SuccessfulApi,
    _config,
    _sha,
    _uuid,
    _write_bundle,
)


def _producer_fixture(
    tmp_path, *, outside_identity_proven=True, no_matching=False,
    post_alias_non_alias_partial=False,
):
    bundle, case, setup = _write_bundle(
        tmp_path / "producer-bundle",
        baseline_text="林澈喜欢蜜瓜。\n周尧喜欢葡萄。\n",
        target_text="林澈把蜜瓜放回桌上。\n周尧喜欢葡萄。\n",
    )
    selector = setup.candidate_selector.model_copy(update={"trait_type": "preference"})
    setup = setup.model_copy(update={"candidate_selector": selector, "author_axis": None})
    case = case.model_copy(update={"axis": case.axis.model_copy(update={
        "sha256": runner._canonical_value_sha({
            "candidate_selector": selector.model_dump(mode="json"), "author_axis": None,
        }),
    })})
    public = bundle.public.model_copy(update={"cases": (case,)})
    setup_bundle = bundle.setup.model_copy(update={
        "cases": (setup,), "public_input_sha256": contract.canonical_sha256(public),
    })
    bundle = bundle.model_copy(update={"public": public, "setup": setup_bundle})
    api = SuccessfulApi(bundle)
    public_ids = (selector.source_document_id, setup.target_document_id)
    runtime_ids = (api.baseline_runtime_id, api.target_runtime_id)
    runtime_map = dict(zip(public_ids, runtime_ids))
    sources = {}
    for ordinal, (public_id, runtime_id, source_kind) in enumerate(zip(
        public_ids, runtime_ids, ("formal_character_profile", "draft"),
    ), 1):
        text = bundle.documents[public_id]
        sources[runtime_id] = _FrozenDocument(
            input_id=_uuid(200 + ordinal),
            document=DocumentInput(
                id=runtime_id, name=f"{public_id}.md", content=text,
                role="character_profile" if ordinal == 1 else "chapter",
            ),
            document_version=1, content_sha256=_sha(text), ordinal=ordinal,
            source_kind=source_kind, source_reason=source_kind, scope=None,
            resolution_state="confirmed" if ordinal == 1 else "draft",
            publication_status="published" if ordinal == 1 else "draft",
            authority_tier="formal_record" if ordinal == 1 else "draft",
        )

    def span(runtime_id, line):
        source = sources[runtime_id]
        return EvidenceSpan(
            document_id=runtime_id, document_name=source.document.name,
            line_start=line, line_end=line,
            text=source.document.content.splitlines()[line - 1],
        )

    baseline = _confirmed_trait().model_copy(update={
        "id": f"ct_{api.candidate_id}", "evidence": (span(runtime_ids[0], 1),),
    })

    def signal(runtime_id, line, character, *, formal=False):
        source = sources[runtime_id]
        return CharacterSignal(
            id=f"cs_{(line + (10 if formal else 20)):032x}",
            character=character, dimension="preference",
            trait_key=baseline.trait_key if character == "林澈" else "食物偏好:葡萄",
            key_object="蜜瓜" if character == "林澈" else "葡萄",
            statement=span(runtime_id, line).text.rstrip("。"),
            polarity="negative" if character == "林澈" else "positive",
            stability="stable" if formal else "temporary",
            observation_kind="explicit_declaration" if formal else "action",
            source_kind=source.source_kind, evidence=span(runtime_id, line),
            source_run_input_id=source.input_id if formal else None,
            support_id=f"L{line}:A1" if formal else None,
            scope_relation="local" if formal else None,
        )

    current = signal(runtime_ids[1], 1, "林澈")
    outside = signal(runtime_ids[1], 2, "周尧")
    formal = signal(runtime_ids[0], 2, "周尧", formal=True)
    prepared = None if no_matching else prepare_character_drift(CharacterDriftCase(
        id="cdc_synthetic_producer_consumer", baseline=baseline,
        observations=(current,),
        scope_compatibility="compatible", material_coverage="complete",
        explanation_coverage="complete",
    ))
    promoted = None if no_matching else promote_character_drift(prepared, None)
    trace = _safe_case_trace(
        character_key=baseline.character, baseline=baseline,
        baseline_entry=(SimpleNamespace(candidate_id=api.candidate_id), baseline, None, baseline.character),
        matched_observation_count=0 if no_matching else len(prepared.matching_observations),
        matched_observations=() if no_matching else prepared.matching_observations,
        prepared=prepared,
        prepare_reason="no_matching_observation" if no_matching else prepared.reason,
        material_coverage="complete", review=None,
        final_outcome="unverifiable" if no_matching else promoted.outcome, visible=not no_matching,
        promote_reason="no_matching_observation" if no_matching else promoted.reason,
        frozen_by_document=sources,
    )
    processing = {
        "planned_chunks": 2, "processed_chunks": 2, "model_completed_chunks": 2,
        "model_incomplete_chunks": 0, "model_uncalled_chunks": 0,
    }
    trace["target_coverage"] = _target_coverage_certificate(
        trace, baseline=baseline,
        unresolved_alias_observations=((outside, ()),),
        formal_signals=(formal,) if outside_identity_proven else (),
        frozen_by_document=sources, target_aliases=(), processing=processing,
        pre_alias_partial=False, baseline_limit=False,
        post_alias_non_alias_partial=post_alias_non_alias_partial,
    )
    items = []
    if not no_matching:
        issue = _to_issue(
            promoted=promoted, prepared=prepared, review=None,
            confirmed_candidate_id=api.candidate_id,
            judgement=promoted.outcome, frozen_by_document=sources,
        )
        items.append({**issue.model_dump(mode="json"), "report_class": "review_clue"})
    review = {"items": items, "truncated": False, "unavailable_count": 0, "scan_limited": False}
    diagnostics = {
        "runtime_provenance": api.runtime,
        "character_consistency": {
            "outcome": "partial", "material_coverage": "partial",
            "counts": {**processing, "model_called_chunks": 2},
            "reason_counts": {"unmatched_character_alias": 1}, "case_trace": [trace],
        },
    }
    return SimpleNamespace(
        bundle=bundle, case=case, setup=setup, api=api, diagnostics=diagnostics,
        review=review, trace=trace, runtime_map=runtime_map,
    )


def _consume(fixture):
    return runner._snapshot_from_reports(
        fixture.case, fixture.setup, fixture.api.candidate_id,
        {"status": "completed"}, fixture.diagnostics, [], fixture.review,
        {"items": [], "truncated": False},
        contract.StageFunnel(input_candidates=1, retrieved_candidates=1,
                            grounded_candidates=1, adjudicated_candidates=1, surfaced_candidates=1),
        source_documents=fixture.bundle.documents, runtime_document_ids=fixture.runtime_map,
    )


def test_backend_source_bound_clue_reaches_consumer_without_global_coverage_rewrite(tmp_path):
    fixture = _producer_fixture(tmp_path)
    certificate = fixture.trace["target_coverage"]
    assert certificate["status"] == "complete"
    assert certificate["alias_disposition"]["proven_outside_target"] == 1
    assert fixture.trace["server_evidence_pair"] == fixture.review["items"][0]["metadata"]["server_evidence_pair"]
    snapshot = _consume(fixture)
    prediction = project_case_prediction(fixture.case, snapshot)
    assert (prediction.outcome, prediction.surface) == ("indeterminate", "review_clue")
    assert snapshot.evidence_binding == "server_evidence_pair_v1"
    assert [ref.role for ref in snapshot.citations] == ["B", "C"]
    assert all(ref.event_group_id is None for ref in snapshot.citations)
    assert fixture.trace["citation_refs"] == []
    assert fixture.trace["review_outcome"] == "not_run"
    assert fixture.diagnostics["character_consistency"]["outcome"] == "partial"
    assert not runner._run_stage_complete({"status": "completed"}, fixture.diagnostics)


def test_backend_unknown_actor_certificate_remains_unassessed_in_consumer(tmp_path):
    fixture = _producer_fixture(tmp_path, outside_identity_proven=False)
    certificate = fixture.trace["target_coverage"]
    assert certificate["status"] == "partial"
    assert certificate["alias_disposition"]["unknown"] == 1
    assert project_case_prediction(fixture.case, _consume(fixture)).outcome == "unassessed"


def test_backend_no_matching_target_cannot_become_no_issue_in_consumer(tmp_path):
    fixture = _producer_fixture(tmp_path, no_matching=True)
    assert fixture.trace["prepare_reason"] == "no_matching_observation"
    assert fixture.trace["explanation_coverage"] == "not_run"
    assert fixture.trace["target_coverage"]["status"] == "partial"
    assert project_case_prediction(fixture.case, _consume(fixture)).outcome == "unassessed"


def test_backend_post_alias_processing_gap_cannot_be_waived_by_outside_actor(tmp_path):
    fixture = _producer_fixture(tmp_path, post_alias_non_alias_partial=True)
    certificate = fixture.trace["target_coverage"]
    assert certificate["alias_disposition"]["proven_outside_target"] == 1
    assert certificate["status"] == "partial"
    assert "post_alias_processing_gap" in certificate["non_alias_blockers"]
    assert project_case_prediction(fixture.case, _consume(fixture)).outcome == "unassessed"


def test_http_case_execution_passes_uploaded_draft_identity_to_backend_pair_consumer(tmp_path):
    fixture = _producer_fixture(tmp_path)

    def transport(request):
        path = request.url.path
        if request.method == "GET" and fixture.api.draft_run_id in path:
            if path.endswith("/diagnostics"):
                return httpx.Response(200, json=fixture.diagnostics)
            if path.endswith("/issues"):
                return httpx.Response(200, json=[])
            if path.endswith("/review-clues"):
                return httpx.Response(200, json=fixture.review)
        response = fixture.api(request)
        if request.method == "GET" and path.endswith("/profile-candidates"):
            payload = response.json()
            payload["items"][0]["evidence"][0]["text"] = fixture.bundle.documents[
                fixture.setup.candidate_selector.source_document_id
            ].splitlines()[0]
            return httpx.Response(200, json=payload)
        return response

    config = _config()
    with httpx.Client(base_url=config.base_url, transport=httpx.MockTransport(transport)) as client:
        snapshot = runner._case_execution(
            client, fixture.bundle, fixture.case, fixture.setup, config,
            runner._canonical_value_sha(fixture.api.runtime), fixture.api.project_id,
            sleeper=lambda _: None, monotonic=lambda: 0.0,
        )
    prediction = project_case_prediction(fixture.case, snapshot)
    assert (prediction.outcome, prediction.surface) == ("indeterminate", "review_clue")
