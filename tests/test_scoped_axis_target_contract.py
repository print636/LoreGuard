"""Approved value/boundary targets keep frozen author scope and exact object."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.character_consistency_stage import (
    CharacterConsistencyStage,
    _FrozenDocument,
    _safe_server_context,
    _signal_matches_target,
    _verified_target_axis_polarity,
)
from app.character_drift import ConfirmedTraitSnapshot
from app.character_trait_extraction import (
    TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
    CharacterSignal,
    CharacterSignalChunk,
    CharacterSignalTarget,
    _matching_target,
    _targeted_chunk_prompt,
)
from app.domain import EvidenceSpan
from app.narrative_context import NarrativeScopeV1, payload_sha256
from app.pipeline import DocumentInput


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _baseline_payload(dimension: str) -> dict:
    definition = "家庭危机时是否保护家人"
    situation = "家庭危机时"
    proposition = "林澈在家庭危机时保护家人"
    return {
        "id": "ct_candidate-scoped",
        "character": "林澈",
        "dimension": dimension,
        "trait_key": "保护家人",
        "statement": proposition,
        "polarity": "positive",
        "stability": "stable",
        "origin": "explicit_setting",
        "evidence": (
            EvidenceSpan(
                document_id="profile", document_name="profile.md",
                line_start=1, line_end=1, text=proposition,
            ),
        ),
        "approved_axis_id": "00000000-0000-4000-8000-000000000001",
        "approved_axis_version": 1,
        "approved_axis_display_name": "家庭保护",
        "approved_axis_definition": definition,
        "approved_axis_definition_sha256": _sha(definition),
        "approved_axis_comparison_key": f"{dimension}:家人",
        "approved_axis_applicability_scope": situation,
        "approved_axis_applicability_scope_sha256": _sha(situation),
        "axis_positive_proposition": proposition,
        "axis_positive_proposition_sha256": _sha(proposition),
        "axis_alignment": "same",
        "axis_polarity": "positive",
    }


def _source() -> _FrozenDocument:
    text = "家庭危机时，林澈抛下家人独自离开。"
    return _FrozenDocument(
        input_id="draft-input",
        document=DocumentInput("draft", "draft.md", text, "chapter"),
        document_version=1,
        content_sha256=_sha(text),
        ordinal=0,
        source_kind="draft",
        source_reason="draft",
        scope=NarrativeScopeV1(),
        resolution_state="confirmed",
        publication_status="draft",
        authority_tier="draft",
    )


def _entry(dimension: str, *, frozen_key: str | None = None):
    baseline = ConfirmedTraitSnapshot.model_validate(_baseline_payload(dimension))
    return (
        SimpleNamespace(
            candidate_id="candidate-scoped",
            payload={
                "comparison_key": frozen_key or f"{dimension}:家人"
            },
        ),
        baseline,
        NarrativeScopeV1(),
        "林澈",
    )


@pytest.mark.parametrize("dimension", ("value", "behavior_boundary"))
def test_scoped_target_keeps_frozen_object_and_untrusted_author_hints(dimension):
    entry = _entry(dimension)
    source = _source()
    context = _safe_server_context(source, baselines=[entry])
    assert context.eligible_traits == context.included_traits == 1
    assert len(context.targets) == 1
    target = context.targets[0]
    assert target.comparison_key == f"{dimension}:家人"
    assert target.approved_axis_comparison_key == target.comparison_key
    assert target.approved_axis_applicability_scope == "家庭危机时"
    assert target.axis_positive_proposition == "林澈在家庭危机时保护家人"
    assert _verified_target_axis_polarity(
        target, source=source, baselines=[entry]
    ) == "positive"
    prompt = _targeted_chunk_prompt(
        CharacterSignalChunk("draft", "draft.md", source.document.content, 1, "draft"),
        (target,),
    )
    assert '"comparison_key":"' + f"{dimension}:家人" + '"' in prompt
    assert '"applicability_scope":"家庭危机时"' in prompt
    assert '"axis_positive_proposition":"林澈在家庭危机时保护家人"' in prompt
    assert target.approved_axis_id not in prompt
    assert "对象或情境无法由可引用的原文支持时" in (
        TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT
    )
    assert json.loads(context.payload)["confirmed_traits"][0][
        "comparison_key"
    ] == target.comparison_key


@pytest.mark.parametrize("dimension", ("value", "behavior_boundary"))
def test_scoped_target_rejects_false_object_even_when_model_reuses_label(dimension):
    target = _safe_server_context(_source(), baselines=[_entry(dimension)]).targets[0]
    signal = CharacterSignal(
        id="cs_" + "a" * 32,
        character="林澈",
        dimension=dimension,
        trait_key=target.trait_key,
        statement="林澈抛下陌生人独自离开",
        polarity="negative",
        stability="situational",
        observation_kind="action",
        key_object="陌生人",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft", document_name="draft.md",
            line_start=1, line_end=1, text="家庭危机时，林澈抛下陌生人独自离开。",
        ),
    )
    assert _matching_target(signal, (target,)) is None
    assert _signal_matches_target(signal, target) is False
    # Generic-label deletion in legacy stable_trait_identity must not turn a
    # distinct literal object into the frozen author object.
    label_like_object = signal.model_copy(update={"key_object": "家人价值取向"})
    assert _matching_target(label_like_object, (target,)) is None
    assert _signal_matches_target(label_like_object, target) is False
    true_object = signal.model_copy(update={"key_object": "家人"})
    assert _matching_target(true_object, (target,)) == target
    assert _signal_matches_target(true_object, target) is True
    # The stage's trusted binding depends on the author object, not on a
    # model-supplied trait label. The extractor still enforces label protocol.
    assert _signal_matches_target(
        true_object.model_copy(update={"trait_key": "另一种标签"}), target
    ) is True


@pytest.mark.parametrize("dimension", ("value", "behavior_boundary"))
def test_scoped_target_requires_exact_hash_bound_author_metadata(dimension):
    target = _safe_server_context(_source(), baselines=[_entry(dimension)]).targets[0]
    payload = target.model_dump()
    for corrupt in (
        {**payload, "approved_axis_comparison_key": f"{dimension}:陌生人"},
        {**payload, "approved_axis_applicability_scope_sha256": "0" * 64},
        {**payload, "axis_positive_proposition_sha256": "0" * 64},
        {**payload, "approved_axis_applicability_scope": 42},
        {**payload, "axis_positive_proposition": None},
        {**payload, "axis_positive_proposition": "\ud800"},
    ):
        with pytest.raises(ValidationError):
            CharacterSignalTarget.model_validate(corrupt)
    # A snapshot row's frozen key must agree with the author's axis object.
    mismatch = _safe_server_context(
        _source(), baselines=[_entry(dimension, frozen_key=f"{dimension}:陌生人")]
    )
    assert mismatch.targets == ()
    assert mismatch.ambiguous_traits == 1


@pytest.mark.parametrize("dimension", ("value", "behavior_boundary"))
def test_scoped_author_metadata_survives_run_snapshot_hydration(dimension):
    baseline = _baseline_payload(dimension)
    payload = {
        "candidate_id": "candidate-scoped",
        "scope": NarrativeScopeV1().model_dump(),
        "evidence": [item.model_dump() for item in baseline["evidence"]],
        "origin": baseline["origin"],
        "character_key": "林澈",
        "character_display_name": baseline["character"],
        "trait_type": dimension,
        "trait_key": baseline["trait_key"],
        "value": baseline["statement"],
        "polarity": baseline["polarity"],
        "stability": baseline["stability"],
        "comparison_key": baseline["approved_axis_comparison_key"],
        **{key: value for key, value in baseline.items() if key.startswith(
            ("approved_axis_", "axis_")
        )},
    }
    row = SimpleNamespace(
        candidate_id="candidate-scoped", project_id="project-one",
        payload=payload, payload_sha256=payload_sha256(payload),
    )
    db = SimpleNamespace(
        scalars=lambda _query: SimpleNamespace(all=lambda: [row])
    )
    entries = CharacterConsistencyStage()._load_confirmed_traits(
        db, run_id="run-one", project_id="project-one", reasons=Counter()
    )
    assert len(entries) == 1
    target = _safe_server_context(_source(), baselines=entries).targets[0]
    assert target.comparison_key == f"{dimension}:家人"
    assert target.approved_axis_applicability_scope == "家庭危机时"
