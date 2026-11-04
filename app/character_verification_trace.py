"""Frozen-line diagnostics for the second, candidate-lines-only draft search."""

from .character_scope_review import ScopeReviewSourceIdentity
from .character_trait_extraction import (
    CharacterSignalChunk, CharacterSignalExtractionResult, CharacterSignalTarget,
    DraftSignalTraceAttemptV1, _draft_actor_source_matches_chunk, _target_bound_review_target,
    target_bound_draft_target_digest,
)

MAX_VERIFICATION_TRACE_CHUNKS = 32
_MAX_RECORDS_PER_ATTEMPT = 16


def verification_trace_chunk(
    source: ScopeReviewSourceIdentity | None, frozen_content: str,
    chunk: CharacterSignalChunk, target: CharacterSignalTarget, *,
    stage_chunk_ordinal: int, target_ordinal: int, selected: tuple[tuple[int, int], ...],
    omitted_lines: int, selection_status: str,
    result: CharacterSignalExtractionResult | None,
) -> dict | None:
    """Publish coordinates inside a selected window, keeping claims separate from evidence."""
    if not _draft_actor_source_matches_chunk(source, frozen_content, chunk):
        return None
    try:
        target_digest = target_bound_draft_target_digest(
            _target_bound_review_target(target, target_ordinal=target_ordinal)
        )
    except (TypeError, ValueError):
        return None
    def in_selected_window(span: tuple[int, int]) -> bool:
        return any(start <= span[0] <= span[1] <= end for start, end in selected)
    row = {
        "source_sha256": source.content_sha256, "target_digest": target_digest,
        "stage_chunk_ordinal": stage_chunk_ordinal, "target_ordinal": target_ordinal,
        "selected_ranges": list(selected), "selection_omitted_line_count": omitted_lines,
        "selection_status": selection_status,
        "primary_attempts": None, "bound_review_submitted_ranges": None,
        "accepted_observation_ranges": None,
    }
    if result is None:
        # Selection finished explicitly without sending this target to the extractor.
        row.update(primary_attempts=[], bound_review_submitted_ranges=[], accepted_observation_ranges=[])
        return row

    # Claimed coordinates from the generation model are distinct from the
    # source-checked ranges accepted by the binder. Unparseable, unlocatable,
    # omitted or absent records cannot establish that no behavior was proposed.
    if result.verification_primary_attempts is not None:
        attempts = []
        for raw in result.verification_primary_attempts:
            attempt = DraftSignalTraceAttemptV1.model_validate(raw.model_dump(mode="python"))
            events = []
            for event in attempt.events[:_MAX_RECORDS_PER_ATTEMPT]:
                span = None
                if event.claimed_line_start_ordinal is not None:
                    span = (chunk.global_line_start + event.claimed_line_start_ordinal - 1,
                            chunk.global_line_start + event.claimed_line_end_ordinal - 1)
                events.append({
                    "claimed_range": span if span is not None and in_selected_window(span) else None,
                    "outcome": event.outcome, "reason": event.reason,
                })
            attempts.append({
                "attempt": attempt.attempt, "observability": attempt.observability,
                "submitted_record_count": attempt.submitted_record_count,
                "records": events, "records_omitted": len(attempt.events) - len(events),
            })
        row["primary_attempts"] = attempts
    dispatch = result.bound_review_submitted_ranges
    if dispatch is not None and all(in_selected_window(span) for span in dispatch):
        row["bound_review_submitted_ranges"] = list(dispatch)
    observations = {
        (signal.evidence.line_start, signal.evidence.line_end)
        for signal in result.draft_observations
        if signal.evidence.document_id == chunk.document_id
    }
    if all(in_selected_window(span) for span in observations):
        row["accepted_observation_ranges"] = sorted(observations)
    return row
