"""Ordinary synthetic target coverage and server-selected clue contracts."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.character_consistency_stage import (
    _FrozenDocument, _alias_disposition_for_target, _safe_case_trace,
    _server_evidence_pair_binding, _target_coverage_certificate, _to_issue,
)
from app.character_drift import CharacterDriftCase, prepare_character_drift, promote_character_drift
from app.character_trait_extraction import CharacterSignal
from app.domain import EvidenceSpan
from app.pipeline import DocumentInput
from tests.test_character_consistency_stage import _confirmed_trait


def _source(identity, text, *, kind="draft", ordinal=1):
    document = DocumentInput(id=identity, name=f"{identity}.md", content=text,
                             role="chapter" if kind == "draft" else "character_profile")
    return _FrozenDocument(
        input_id=f"input-{identity}", document=document, document_version=1,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(), ordinal=ordinal,
        source_kind=kind, source_reason=kind, scope=None,
        resolution_state="confirmed", publication_status="draft" if kind == "draft" else "published",
        authority_tier="draft" if kind == "draft" else "formal_record",
    )


def _span(source):
    return EvidenceSpan(document_id=source.document.id, document_name=source.document.name,
                        line_start=1, line_end=1, text=source.document.content)


def _signal(source, character, *, formal=False, suffix=1):
    return CharacterSignal(
        id=f"cs_{suffix:032x}", character=character, dimension="preference",
        trait_key="食物偏好", key_object="葡萄", statement=source.document.content.rstrip("。"),
        polarity="positive", stability="stable" if formal else "temporary",
        observation_kind="explicit_declaration", source_kind=source.source_kind,
        evidence=_span(source),
        source_run_input_id=source.input_id if formal else None,
        support_id="L1:A1" if formal else None,
        scope_relation="local" if formal else None,
    )


def _fixture():
    candidate = str(uuid4())
    profile = _source("profile", "林澈喜欢蜜瓜。", kind="formal_character_profile")
    draft = _source("draft", "林澈讨厌蜜瓜。")
    other_profile = _source("other-profile", "周尧喜欢葡萄。", kind="formal_character_profile")
    other_draft = _source("other-draft", "周尧喜欢葡萄。")
    sources = {s.document.id: s for s in (profile, draft, other_profile, other_draft)}
    baseline = _confirmed_trait().model_copy(update={"id":f"ct_{candidate}","evidence":(_span(profile),)})
    current = _signal(draft, "林澈").model_copy(update={
        "trait_key":baseline.trait_key, "key_object":"蜜瓜", "polarity":"negative"})
    prepared = prepare_character_drift(CharacterDriftCase(
        id="cdc_synthetic_coverage", baseline=baseline, observations=(current,),
        scope_compatibility="compatible", material_coverage="complete", explanation_coverage="complete"))
    other = _signal(other_draft, "周尧", suffix=2)
    formal = _signal(other_profile, "周尧", formal=True, suffix=3)
    trace = {
        "confirmed_candidate_id_sha256":hashlib.sha256(candidate.encode()).hexdigest(),
        "character_key":"林澈", "dimension":"preference", "comparison_key":"preference:食物偏好:蜜瓜",
        "material_coverage":"complete", "material_coverage_reasons":[],
        "explanation_coverage":"complete", "matched_observation_count":1,
        "matched_observation_refs_truncated":False, "citation_refs_incomplete":False,
        "prepare_reason":"single_behavior_is_not_drift", "review_required":False,
        "review_outcome":"not_run", "promote_reason":"single_behavior_is_not_drift",
        "final_outcome":"needs_confirmation",
    }
    return candidate, sources, baseline, prepared, other, formal, trace


def _coverage(trace, baseline, sources, other, formal, **overrides):
    values = dict(
        baseline=baseline, unresolved_alias_observations=((other,()),), formal_signals=(formal,),
        frozen_by_document=sources, target_aliases=(),
        processing={"planned_chunks":2,"processed_chunks":2,"model_completed_chunks":2,
                    "model_incomplete_chunks":0,"model_uncalled_chunks":0},
        pre_alias_partial=False, baseline_limit=False,
    )
    values.update(overrides)
    return _target_coverage_certificate(trace, **values)


def test_source_proven_other_character_does_not_invalidate_complete_target():
    _, sources, baseline, _, other, formal, trace = _fixture()
    certificate = _coverage(trace, baseline, sources, other, formal)
    assert certificate["status"] == "complete"
    assert certificate["alias_disposition"] == {
        "total":1,"proven_outside_target":1,"possibly_target":0,"unknown":0,
        "outside_proofs":certificate["alias_disposition"]["outside_proofs"],
    }
    proof = certificate["alias_disposition"]["outside_proofs"][0]
    assert proof["proof_kind"] == "formal_actor_and_draft_binding_v1"
    assert len(proof["source_refs"]) == 2
    assert {ref["document_name"] for ref in proof["source_refs"]} == {"other-profile.md","other-draft.md"}
    assert all("text" not in ref for ref in proof["source_refs"])
    assert certificate["non_alias_blockers"] == []


@pytest.mark.parametrize("mode", ["nickname", "missing_formal_binding", "actor_mismatch", "target_alias"])
def test_unproven_or_target_alias_still_blocks_target_coverage(mode):
    _, sources, baseline, _, other, formal, trace = _fixture()
    aliases = ()
    if mode == "nickname":
        other = other.model_copy(update={"character":"小周"})
    elif mode == "missing_formal_binding":
        formal = formal.model_copy(update={"source_run_input_id":None})
    elif mode == "actor_mismatch":
        other = other.model_copy(update={"statement":"林澈喜欢葡萄"})
    else:
        aliases = ("周尧",)
    certificate = _coverage(trace, baseline, sources, other, formal, target_aliases=aliases)
    assert certificate["status"] == "partial"
    assert certificate["alias_disposition"]["proven_outside_target"] == 0
    assert certificate["alias_disposition"]["unknown"] + certificate["alias_disposition"]["possibly_target"] == 1


@pytest.mark.parametrize("change", [
    {"material_coverage":"partial"}, {"explanation_coverage":"not_run"},
    {"matched_observation_count":0,"prepare_reason":"no_matching_observation"},
    {"review_required":True,"review_outcome":"not_run"},
    {"citation_refs_incomplete":True}, {"matched_observation_refs_truncated":True},
])
def test_target_local_gaps_never_receive_complete_certificate(change):
    _, sources, baseline, _, other, formal, trace = _fixture()
    certificate = _coverage({**trace, **change}, baseline, sources, other, formal)
    assert certificate["status"] == "partial"
    assert certificate["non_alias_blockers"]


def test_source_package_gaps_and_truncation_remain_blocking():
    _, sources, baseline, _, other, formal, trace = _fixture()
    for overrides in [
        {"pre_alias_partial":True}, {"baseline_limit":True},
        {"processing":{"planned_chunks":2,"processed_chunks":2,"model_completed_chunks":1,
                       "model_incomplete_chunks":1,"model_uncalled_chunks":0}},
    ]:
        assert _coverage(trace, baseline, sources, other, formal, **overrides)["status"] == "partial"


def test_later_non_alias_gap_blocks_even_with_proven_other_actor():
    _, sources, baseline, _, other, formal, trace = _fixture()
    certificate = _coverage(
        trace, baseline, sources, other, formal, post_alias_non_alias_partial=True,
    )
    assert certificate["alias_disposition"]["proven_outside_target"] == 1
    assert certificate["status"] == "partial"
    assert "post_alias_processing_gap" in certificate["non_alias_blockers"]


def test_unverifiable_target_cannot_borrow_an_alias_only_coverage_exemption():
    _, sources, baseline, _, other, formal, trace = _fixture()
    certificate = _coverage({**trace, "final_outcome":"unverifiable"}, baseline, sources, other, formal)
    assert certificate["status"] == "partial"
    assert "case_outcome_unverifiable" in certificate["non_alias_blockers"]


def test_server_pair_trace_and_report_share_frozen_non_model_binding():
    candidate, sources, baseline, prepared, _, _, _ = _fixture()
    promoted = promote_character_drift(prepared, None)
    issue = _to_issue(promoted=promoted, prepared=prepared, review=None,
                      confirmed_candidate_id=candidate, judgement="needs_confirmation", frozen_by_document=sources)
    trace = _safe_case_trace(
        character_key="林澈", baseline=baseline,
        baseline_entry=(SimpleNamespace(candidate_id=candidate),baseline,None,"林澈"),
        matched_observation_count=1, matched_observations=prepared.matching_observations,
        prepared=prepared, prepare_reason=prepared.reason, material_coverage="complete", review=None,
        final_outcome=promoted.outcome, visible=True, promote_reason=promoted.reason,
        frozen_by_document=sources,
    )
    binding = issue.metadata["server_evidence_pair"]
    assert trace["server_evidence_pair"] == binding
    assert trace["citation_refs"] == []
    assert binding["report_class"] == "review_clue"
    assert [ref["role"] for ref in binding["evidence_refs"]] == ["B","C"]
    for index, ref in enumerate(binding["evidence_refs"]):
        assert ref["evidence_index"] == index
        assert ref["text_sha256"] == hashlib.sha256(issue.evidence[index].text.encode()).hexdigest()
        assert not {"handle","response_index","event_group_id"} & ref.keys()


def test_server_pair_source_tampering_cannot_create_a_certificate():
    candidate, sources, _, prepared, _, _, _ = _fixture()
    source = sources["draft"]
    sources["draft"] = _source("draft", "周尧讨厌蜜瓜。")
    assert _server_evidence_pair_binding(prepared,candidate_id=candidate,frozen_by_document=sources) is None
    sources["draft"] = source
    assert _server_evidence_pair_binding(prepared,candidate_id="not-a-candidate",frozen_by_document=sources) is None
