from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from app.character_support_bindings import (
    TraitSupportRef,
    bind_support_refs,
    support_bindings_sha256,
    verify_stored_support_bindings,
)
from app.character_trait_extraction import CharacterSignal, build_pending_trait_candidates
from app.domain import EvidenceSpan


LINE = "林澈喜欢蜜瓜。林澈喜欢葡萄。"


def _source(line: str = LINE):
    digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
    snapshot = SimpleNamespace(
        id="input-1", document_id="document-1", document_name="profile.md",
        document_version=1, content_sha256=digest, content=line,
    )
    evidence = [{
        "input_id": snapshot.id, "document_id": snapshot.document_id,
        "document_name": snapshot.document_name,
        "document_version": snapshot.document_version,
        "content_sha256": snapshot.content_sha256,
        "line_start": 1, "line_end": 1, "text": line,
    }]
    return snapshot, evidence


def _ref(support_id: str, **changes) -> TraitSupportRef:
    return TraitSupportRef(
        evidence_index=0, support_id=support_id,
        scope_relation="local", **changes,
    )


def _signal(support_id: str, statement: str, ordinal: int) -> CharacterSignal:
    return CharacterSignal(
        id=f"cs_{ordinal:032x}", character="林澈", dimension="preference",
        trait_key="fruit_preference", statement=statement,
        polarity="positive", stability="stable",
        observation_kind="explicit_declaration", key_object="水果",
        source_kind="formal_character_profile",
        evidence=EvidenceSpan(
            document_id="document-1", document_name="profile.md",
            line_start=1, line_end=1, text=LINE,
        ),
        support_id=support_id, scope_relation="local",
        source_run_input_id="input-1",
    )


def test_reviewed_same_line_targets_become_independent_candidates():
    rows = build_pending_trait_candidates((
        _signal("L1:A1", "偏爱一种水果", 1),
        _signal("L1:A2", "也偏爱另一种水果", 2),
    ))
    assert len(rows) == 2
    assert len({row.id for row in rows}) == 2
    assert {row.support_refs[0].support_id for row in rows} == {"L1:A1", "L1:A2"}
    assert all(len(row.support_refs) == 1 and len(row.evidence) == 1 for row in rows)


def test_target_span_is_not_the_whole_line_or_neighbor_clause():
    snapshot, evidence = _source()
    payload = bind_support_refs(
        [_ref("L1:A2")], evidence=evidence, frozen_by_id={snapshot.id: snapshot}
    )
    binding = payload["bindings"][0]
    target = binding["target"]
    assert target["support_id"] == "L1:A2"
    assert LINE[target["start_offset"]:target["end_offset"]] == "林澈喜欢葡萄"
    assert binding["context"] == []
    assert verify_stored_support_bindings(
        payload, support_bindings_sha256(payload),
        evidence=evidence, frozen_by_id={snapshot.id: snapshot},
    ) == payload


def test_context_chain_is_server_recomputed_not_extra_reviewer_basis():
    line = "林澈的核心性格是谨慎。他会先观察。随后才行动。"
    snapshot, evidence = _source(line)
    payload = bind_support_refs([
        TraitSupportRef(
            evidence_index=0, support_id="L1:A3",
            actor_anchor_id="L1:A1", label_anchor_id="L1:A1",
            scope_relation="labelled_elaboration",
        )
    ], evidence=evidence, frozen_by_id={snapshot.id: snapshot})
    binding = payload["bindings"][0]
    assert binding["target"]["support_id"] == "L1:A3"
    assert [part["support_id"] for part in binding["context"]] == [
        "L1:A1", "L1:A2",
    ]
    assert [part["role"] for part in binding["context"]] == [
        "actor_anchor", "bridge",
    ]


def test_offsets_are_unicode_codepoints_after_supplementary_character():
    line = "😀林澈先观察。随后他才行动。"
    snapshot, evidence = _source(line)
    payload = bind_support_refs(
        [_ref("L1:A2")], evidence=evidence, frozen_by_id={snapshot.id: snapshot}
    )
    target = payload["bindings"][0]["target"]
    assert target["start_offset"] == line.index("随后")
    assert line[target["start_offset"]:target["end_offset"]] == "随后他才行动"


@pytest.mark.parametrize("mutation", [
    lambda data: data["bindings"][0]["target"].update(start_offset=0),
    lambda data: data["bindings"][0].update(support_id="L1:A1"),
    lambda data: data["bindings"][0].update(actor_anchor_id="L1:A2"),
])
def test_stored_binding_tampering_fails_even_with_recomputed_digest(mutation):
    snapshot, evidence = _source()
    payload = bind_support_refs(
        [_ref("L1:A2")], evidence=evidence, frozen_by_id={snapshot.id: snapshot}
    )
    mutation(payload)
    with pytest.raises(ValueError):
        verify_stored_support_bindings(
            payload, support_bindings_sha256(payload),
            evidence=evidence, frozen_by_id={snapshot.id: snapshot},
        )


def test_binding_refuses_wrong_frozen_source_and_bad_anchor():
    snapshot, evidence = _source()
    snapshot.content = LINE + "附加内容"
    with pytest.raises(ValueError):
        bind_support_refs([_ref("L1:A2")], evidence=evidence,
                          frozen_by_id={snapshot.id: snapshot})
    snapshot.content = LINE
    with pytest.raises(ValueError):
        bind_support_refs([
            TraitSupportRef(
                evidence_index=0, support_id="L1:A1",
                actor_anchor_id="L1:A2", scope_relation="same_actor_continuation",
            )
        ], evidence=evidence, frozen_by_id={snapshot.id: snapshot})


def test_legacy_null_binding_remains_unavailable():
    snapshot, evidence = _source()
    assert verify_stored_support_bindings(
        None, None, evidence=evidence, frozen_by_id={snapshot.id: snapshot}
    ) is None
