"""An independent original-claim check over a previously admitted source path."""
from __future__ import annotations

from dataclasses import dataclass

from .character_draft_actor_review import (
    SelectedOriginalClaim, SelectedObjectOriginalClaim, TargetBoundDraftReviewRequest,
    TARGET_BOUND_DRAFT_REVIEW_PROMPT_V16, TARGET_BOUND_DRAFT_REVIEW_PROMPT_V17, TARGET_BOUND_DRAFT_REVIEW_PROMPT_V18,
    target_bound_draft_review_request_digest, verify_target_bound_draft_review_source,
)
from .character_claim_resolution import claim_resolution_target_sha256, text_sha256
from .config import SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3


@dataclass(frozen=True)
class SelectedActionCertificate:
    request: TargetBoundDraftReviewRequest
    proposal_id: str
    signal: object


def selected_origin_payload(request, proposal) -> dict:
    return dict(request_sha256=target_bound_draft_review_request_digest(request), proposal_id=proposal.proposal_id,
                selection_digest=request.selection_digest, fact_clause_id=proposal.fact_clause_id,
                path_ids=list(proposal.path_ids), window_ids=list(request.selection_window_ids))


def build_selected_claim_review_request(pending, origin, *, source_identity, frozen_content, protocol_version="v17"):
    from .character_trait_extraction import _primary_claim_identity, _target_bound_review_target
    from .character_action_selection import ActionChoice, _selection_binding
    request = origin.request
    raw = pending.original
    original_source = raw.source if pending.retained is not None else raw.index.source
    if (protocol_version not in {"v17", "v18"} or request.prompt_version != TARGET_BOUND_DRAFT_REVIEW_PROMPT_V16
            or protocol_version == "v17" and (raw.record.key_object != "" or pending.retained is not None)
            or original_source != source_identity
            or claim_resolution_target_sha256(request.target) != pending.target_sha256
            or not verify_target_bound_draft_review_source(request, source_identity, frozen_content=frozen_content)):
        raise ValueError("selected_claim_origin_invalid")
    proposal = next(p for p in request.proposals if p.proposal_id == origin.proposal_id)
    record = raw.record
    if (record.source_line_start != proposal.line_start or record.source_line_end != proposal.line_end
            or record.evidence != proposal.evidence
            or origin.signal.selected_action_origin != selected_origin_payload(request, proposal)):
        raise ValueError("selected_claim_source_mismatch")
    original = _primary_claim_identity(raw, pending.package_attempt, source_identity,
        _target_bound_review_target(raw.target, target_ordinal=raw.target_ordinal), object_review=False)
    claim_type = SelectedObjectOriginalClaim if record.key_object else SelectedOriginalClaim
    claim = claim_type(**{**original.model_dump(), "key_object": record.key_object})
    choice = ActionChoice(fact_clause_id=proposal.fact_clause_id, actor_anchor_id=proposal.actor_anchor_id, path_ids=proposal.path_ids)
    data = request.model_dump(mode="python")
    data.update(prompt_version=TARGET_BOUND_DRAFT_REVIEW_PROMPT_V18 if protocol_version == "v18" else TARGET_BOUND_DRAFT_REVIEW_PROMPT_V17, original_claim=claim,
                origin_request_sha256=target_bound_draft_review_request_digest(request), origin_observation_id=origin.signal.id,
                proposals=(proposal,), selection_digest=_selection_binding(source_identity, request.target_digest,
                    request.selection_window_ids, (choice,)))
    return TargetBoundDraftReviewRequest.model_validate(data)


def review_selected_original_claim(pending, origin, *, chunk, source_identity, frozen_content, provider, settings, token_budget):
    from .character_trait_extraction import (
        _TargetBoundDraftOutcome, _bind_target_bound_draft_certificate, _draft_actor_source_matches_chunk,
    )
    from .character_draft_actor_review_provider import TargetBoundDraftReviewBatchEntry, run_target_bound_draft_review
    from .config import HABIT_SCENE_CALL_TIMEOUT_SECONDS
    policy = settings._character_selected_claim_resolution_policy
    if not (policy in {SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3} and settings._character_claim_resolution_v1
            and settings._character_habit_action_selection_v1
            and (pending.retained is None or policy == SELECTED_CLAIM_RESOLUTION_POLICY_V3)):
        return _TargetBoundDraftOutcome(reason_counts=(("selected_claim_resolution_disabled", 1),))
    if not _draft_actor_source_matches_chunk(source_identity, frozen_content, chunk):
        return _TargetBoundDraftOutcome(reason_counts=(("selected_claim_origin_invalid", 1),))
    try:
        request = build_selected_claim_review_request(pending, origin, source_identity=source_identity, frozen_content=frozen_content,
            protocol_version="v17" if policy == SELECTED_CLAIM_RESOLUTION_POLICY_V1 else "v18")
    except (ValueError, TypeError, StopIteration):
        return _TargetBoundDraftOutcome(reason_counts=(("selected_claim_origin_invalid", 1),))
    run = run_target_bound_draft_review(
        (TargetBoundDraftReviewBatchEntry(request=request, expected_source=source_identity, frozen_content=frozen_content,
                                         expected_primary_claim=request.original_claim),),
        provider=provider, token_budget=token_budget, completion_reserve=settings.character_draft_actor_review_completion_tokens,
        timeout_seconds=HABIT_SCENE_CALL_TIMEOUT_SECONDS, remaining_deadline_seconds=HABIT_SCENE_CALL_TIMEOUT_SECONDS,
        max_response_bytes=settings.character_draft_actor_review_max_response_bytes, max_attempts=1, habit_scene_policy=True,
    )
    usage = {key: getattr(run, key) for key in ("attempted_calls", "prompt_tokens", "completion_tokens", "charged_tokens")}
    reasons = (("selected_claim_" + run.failure_reason, 1),) if run.failure_reason else ()
    decision = run.evaluation.evaluations[0].decisions[0]
    if decision.verdict != "supported":
        reason = "object_unverified" if decision.slot_conflicts == ("object_span",) else "record_unverified"
        return _TargetBoundDraftOutcome(reason_counts=(*reasons, ("selected_claim_unresolved", 1)),
            material_reasons=((pending.original.record_index, reason),) if decision.raw_verdict is not None else (), **usage)
    try:
        proposal = request.proposals[0]
        signal = _bind_target_bound_draft_certificate(request=request, proposal=proposal, decision=decision,
            target=pending.original.target, chunk=chunk, source_identity=source_identity, frozen_content=frozen_content,
            expected_primary_claim=request.original_claim)
    except (TypeError, ValueError):
        return _TargetBoundDraftOutcome(reason_counts=(*reasons, ("selected_claim_certificate_invalid", 1)), **usage)
    if signal.id == origin.signal.id:
        signal = signal.model_copy(update={"selected_action_origin": origin.signal.selected_action_origin})
    proof = dict(record_index=pending.original.record_index, source_input_id=request.run_input_id,
        source_document_id=request.document_id, source_content_sha256=request.content_sha256,
        target_sha256=claim_resolution_target_sha256(request.target), review_claim_sha256=request.original_claim.claim_digest,
        request_sha256=target_bound_draft_review_request_digest(request), proposal_id=proposal.proposal_id,
        observation_id=signal.id, line_start=signal.evidence.line_start, line_end=signal.evidence.line_end,
        text_sha256=text_sha256(signal.evidence.text), observation_statement_sha256=text_sha256(signal.statement),
        origin_request_sha256=request.origin_request_sha256, origin_observation_id=request.origin_observation_id,
        fact_clause_id=proposal.fact_clause_id)
    return _TargetBoundDraftOutcome(signals=(signal,), claim_certificates=(proof,),
        resolved_record_indices=frozenset({pending.original.record_index}), reason_counts=reasons, **usage)
