from __future__ import annotations

from dataclasses import replace

from app.character_consistency_stage import (
    _FrozenDocument,
    _explanation_window_specs,
)
from app.character_drift import ConfirmedTraitSnapshot
from app.character_trait_extraction import CharacterSignal
from app.domain import EvidenceSpan
from app.narrative_context import NarrativeBranch, NarrativeScopeV1
from app.pipeline import DocumentInput


def _baseline() -> ConfirmedTraitSnapshot:
    return ConfirmedTraitSnapshot(
        id="ct_explanation_window_coverage",
        character="林澈",
        dimension="core_personality",
        trait_key="社交倾向:内向",
        statement="林澈长期内向，不会主动在陌生人中高谈阔论。",
        polarity="positive",
        stability="stable",
        origin="explicit_setting",
        authority_tier="core_canon",
        evidence=(
            EvidenceSpan(
                document_id="profile",
                document_name="profile.md",
                line_start=1,
                line_end=1,
                text="林澈长期内向，不会主动在陌生人中高谈阔论。",
            ),
        ),
    )


def _observation() -> CharacterSignal:
    return CharacterSignal(
        id="cs_" + "1" * 32,
        character="林澈",
        dimension="core_personality",
        trait_key="社交倾向:内向",
        statement="主动在宴会上与陌生人长谈",
        polarity="negative",
        stability="situational",
        observation_kind="action",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft",
            document_name="draft.md",
            line_start=1,
            line_end=1,
            text="林澈主动在宴会上与陌生人长谈。",
        ),
    )


def _source(
    *,
    document_id: str,
    content: str,
    ordinal: int,
    source_kind: str,
) -> _FrozenDocument:
    published = source_kind != "draft"
    return _FrozenDocument(
        input_id=f"input-{document_id}",
        document=DocumentInput(
            id=document_id,
            name=f"{document_id}.md",
            role="chapter",
            content=content,
        ),
        document_version=1,
        content_sha256="0" * 64,
        ordinal=ordinal,
        source_kind=source_kind,
        source_reason=source_kind,
        scope=NarrativeScopeV1(),
        resolution_state="confirmed",
        publication_status="published" if published else "draft",
        authority_tier="formal_record" if published else "draft",
    )


def _windows(documents: list[_FrozenDocument], *, limit: int = 24):
    scope = NarrativeScopeV1()
    return _explanation_window_specs(
        baseline=_baseline(),
        baseline_scope=scope,
        observations=(_observation(),),
        draft_scopes=(scope,),
        draft_ordinals=(1,),
        draft_document_ids=("draft",),
        documents=documents,
        limit=limit,
    )


def test_complete_segmentation_includes_pronoun_only_history_imported_later():
    history = _source(
        document_id="history",
        ordinal=9,
        source_kind="published_history",
        content=(
            "宴会前，众人要求新来的密探混入商会。\n"
            "她必须表现得健谈，才能避免身份暴露。"
        ),
    )
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content="林澈主动在宴会上与陌生人长谈。",
    )

    candidates, incomplete = _windows([draft, history])

    assert incomplete is False
    assert any(
        row.evidence.document_id == "history"
        and "她必须表现得健谈" in row.evidence.text
        for row in candidates
    )


def test_complete_segmentation_covers_every_non_empty_source_line():
    history = _source(
        document_id="history",
        ordinal=0,
        source_kind="published_history",
        content="\n".join(f"历史资料第{index}行。" for index in range(1, 18)),
    )
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content="林澈主动在宴会上与陌生人长谈。",
    )

    candidates, incomplete = _windows([history, draft])

    assert incomplete is False
    history_rows = [row for row in candidates if row.evidence.document_id == "history"]
    covered = {
        line
        for row in history_rows
        for line in range(row.evidence.line_start, row.evidence.line_end + 1)
    }
    assert covered == set(range(1, 18))


def test_in_review_current_draft_is_exhaustively_segmented():
    draft = replace(
        _source(
            document_id="draft",
            ordinal=1,
            source_kind="draft",
            content=(
                "林澈主动在宴会上与陌生人长谈。\n"
                "因为正在卧底，她必须扮演健谈的商人。"
            ),
        ),
        publication_status="in_review",
    )

    candidates, incomplete = _windows([draft])

    assert incomplete is False
    assert candidates
    assert all(row.publication_status == "in_review" for row in candidates)
    assert any("因为正在卧底" in row.evidence.text for row in candidates)


def test_unknown_formal_profile_is_not_silently_omitted():
    profile = replace(
        _source(
            document_id="legacy-profile",
            ordinal=0,
            source_kind="formal_character_profile",
            content="林澈执行潜伏任务时会伪装成健谈的商人。",
        ),
        publication_status="unknown",
        authority_tier="core_canon",
    )
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content="林澈主动在宴会上与陌生人长谈。",
    )

    candidates, incomplete = _windows([profile, draft])

    assert incomplete is False
    profile_candidates = [
        row for row in candidates if row.evidence.document_id == "legacy-profile"
    ]
    assert profile_candidates
    assert all(row.publication_status == "unknown" for row in profile_candidates)


def test_candidate_cap_or_unbindable_long_line_fails_coverage_closed():
    many = _source(
        document_id="history",
        ordinal=0,
        source_kind="published_history",
        content="\n".join((f"段落{index}。" + "甲" * 1_990) for index in range(4)),
    )
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content="林澈主动在宴会上与陌生人长谈。",
    )
    capped, capped_incomplete = _windows([many, draft], limit=2)
    assert len(capped) == 2
    assert capped_incomplete is True

    overlong = _source(
        document_id="history-long",
        ordinal=0,
        source_kind="published_history",
        content="她" * 16_001,
    )
    _, overlong_incomplete = _windows([overlong, draft])
    assert overlong_incomplete is True


def test_published_source_keeps_only_scope_compatible_observation_documents():
    branch_a = NarrativeScopeV1(
        branch=NarrativeBranch(path=["route-a"], exclusive_group="ending")
    )
    branch_b = NarrativeScopeV1(
        branch=NarrativeBranch(path=["route-b"], exclusive_group="ending")
    )
    history = _source(
        document_id="history-a",
        ordinal=8,
        source_kind="published_history",
        content="她在 A 路线中为潜入宴会而扮演健谈商人。",
    )
    history = replace(history, scope=branch_a)
    draft_a = _source(
        document_id="draft-a",
        ordinal=1,
        source_kind="draft",
        content="林澈主动在宴会上与陌生人长谈。",
    )
    draft_a = replace(draft_a, scope=branch_a)
    draft_b = _source(
        document_id="draft-b",
        ordinal=2,
        source_kind="draft",
        content="林澈在另一场宴会上与陌生人长谈。",
    )
    draft_b = replace(draft_b, scope=branch_b)
    observation_a = _observation().model_copy(
        update={
            "evidence": _observation().evidence.model_copy(
                update={"document_id": "draft-a", "document_name": "draft-a.md"}
            )
        }
    )
    observation_b = _observation().model_copy(
        update={
            "id": "cs_" + "2" * 32,
            "evidence": _observation().evidence.model_copy(
                update={
                    "document_id": "draft-b",
                    "document_name": "draft-b.md",
                    "text": "林澈在另一场宴会上与陌生人长谈。",
                }
            ),
        }
    )

    candidates, incomplete = _explanation_window_specs(
        baseline=_baseline(),
        baseline_scope=NarrativeScopeV1(),
        observations=(observation_a, observation_b),
        draft_scopes=(branch_a, branch_b),
        draft_ordinals=(1, 2),
        draft_document_ids=("draft-a", "draft-b"),
        documents=[history, draft_a, draft_b],
    )

    assert incomplete is False
    history_candidates = [
        row for row in candidates if row.evidence.document_id == "history-a"
    ]
    assert history_candidates
    assert all(
        row.eligible_draft_document_ids == ("draft-a",)
        for row in history_candidates
    )


def test_other_target_draft_without_safe_case_binding_makes_coverage_partial():
    scope = NarrativeScopeV1()
    current = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content="林澈主动在宴会上与陌生人长谈。",
    )
    other = _source(
        document_id="draft-other",
        ordinal=2,
        source_kind="draft",
        content="补记：她当时的健谈表现可能与潜入伪装有关。",
    )
    other = replace(other, publication_status="in_review")

    candidates, incomplete = _explanation_window_specs(
        baseline=_baseline(),
        baseline_scope=scope,
        observations=(_observation(),),
        # Production supplies only documents that produced this case's
        # matching observations. The second target draft has no opposed signal
        # and therefore is intentionally absent from these three tuples.
        draft_scopes=(scope,),
        draft_ordinals=(1,),
        draft_document_ids=("draft",),
        documents=[current, other],
    )

    assert [row.evidence.document_id for row in candidates] == ["draft"]
    assert incomplete is True
