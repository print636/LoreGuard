"""Source-bound unresolved object claims; these are never observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .character_scope_review import ScopeReviewSourceIdentity

if TYPE_CHECKING:
    from .character_trait_extraction import CharacterSignalTarget, _RawCharacterSignal


MAX_RETAINED_HABIT_CLAIMS = 32


@dataclass(frozen=True, slots=True)
class RetainedHabitClaim:
    """A typed raw record with verified coordinates, not verified semantics."""
    record_index: int
    record: _RawCharacterSignal
    source: ScopeReviewSourceIdentity
    target: CharacterSignalTarget
    target_ordinal: int


def collect_retained_habit_claims(validation, response_text, *, chunk, targets, source_identity, frozen_content):
    """Retain only object-support failures attached to one authorized habit."""
    from .character_trait_extraction import (
        _ENVELOPE_ADAPTER, _RECORD_ADAPTER, _draft_actor_source_matches_chunk,
        _target_actor_literals, _target_bound_review_target, stable_trait_identity,
    )
    from .character_draft_actor_review import _exact_actor_occurrences

    if not validation.parsed or not _draft_actor_source_matches_chunk(source_identity, frozen_content, chunk):
        return (), 0
    try:
        records = _ENVELOPE_ADAPTER.validate_json(response_text).records
    except (TypeError, ValueError, ValidationError):
        return (), 0
    # Ordinary accepted/deferred records keep their existing route and identity.
    handled = {row.record_index for row in validation.target_bound_draft_deferred}
    handled.update(row.record_index for row in validation.draft_actor_deferred)
    lines = frozen_content.splitlines()
    retained, omitted = [], 0
    for failure in validation.failures:
        index = failure.record_index
        if failure.reason != "key_object_support" or index is None or index in handled or not 0 <= index < len(records):
            continue
        try:
            record = _RECORD_ADAPTER.validate_python(records[index])
        except (TypeError, ValueError, ValidationError):
            continue
        if (record.dimension != "contextual_behavior" or not record.key_object.strip()
                or record.stability not in {"temporary", "situational"}
                or record.observation_kind not in {"action", "interaction"}
                or record.source_line_start != record.source_line_end
                or not chunk.global_line_start <= record.source_line_start <= chunk.global_line_end
                or not 1 <= record.source_line_start <= len(lines)
                or record.evidence != lines[record.source_line_start - 1]):
            continue
        matches = [(ordinal, target) for ordinal, target in enumerate(targets, 1)
            if target.dimension == "contextual_behavior" and target.habit_contexts
            and record.character in _target_actor_literals(target)
            and record.polarity == target.requested_polarity
            and stable_trait_identity(record.dimension, record.trait_key) == target.comparison_key]
        if len(matches) != 1:
            continue
        ordinal, target = matches[0]
        # A named authorized identity occurs in this window. Who performs the
        # claimed action, and whether it happened, still require V16 and V18.
        if not _exact_actor_occurrences(record.evidence, _target_bound_review_target(target, target_ordinal=ordinal)):
            continue
        handled.add(index)
        if len(retained) == MAX_RETAINED_HABIT_CLAIMS:
            omitted += 1
            continue
        retained.append(RetainedHabitClaim(index, record, source_identity, target, ordinal))
    return tuple(retained), omitted
