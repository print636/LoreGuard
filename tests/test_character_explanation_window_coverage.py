from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.character_consistency_stage import (
    _FrozenDocument,
    _compound_explanation_debt_is_closed,
    _explanation_window_specs,
    _segment_explanation_source,
    _without_cross_actor_definitive_support,
)
from app.config import Settings
from app.character_drift import ConfirmedTraitSnapshot, SupportEvidence
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


def _windows(
    documents: list[_FrozenDocument],
    *,
    limit: int = 48,
    baseline: ConfirmedTraitSnapshot | None = None,
):
    scope = NarrativeScopeV1()
    return _explanation_window_specs(
        baseline=baseline or _baseline(),
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


def test_markdown_section_windows_do_not_mix_adjacent_character_blocks():
    document = DocumentInput(
        id="profile-sections",
        name="profile.md",
        role="character_profile",
        content=(
            "# 角色档案\n\n"
            "## 白榆\n\n"
            "白榆平常说话简短直接。\n"
            "她从不使用冗长奉承话。\n\n"
            "## 祁雾\n\n"
            "祁雾为了潜入商会启动镜面身份。\n"
            "这层伪装持续到任务结束。"
        ),
    )

    spans, incomplete = _segment_explanation_source(document)

    assert incomplete is False
    assert any("## 白榆" in row.text and "白榆平常" in row.text for row in spans)
    assert any("## 祁雾" in row.text and "祁雾为了" in row.text for row in spans)
    assert all(not ("白榆" in row.text and "祁雾" in row.text) for row in spans)
    covered = {
        line
        for row in spans
        for line in range(row.line_start, row.line_end + 1)
        if document.content.splitlines()[line - 1].strip()
    }
    non_empty = {
        index for index, line in enumerate(document.content.splitlines(), start=1)
        if line.strip()
    }
    assert covered == non_empty


def test_baseline_block_is_not_reintroduced_as_an_explanation_candidate():
    profile = replace(
        _source(
            document_id="profile",
            ordinal=0,
            source_kind="formal_character_profile",
            content=(
                "# 角色设定档案\n\n"
                "## 林澈\n\n"
                "林澈长期内向，不会主动在陌生人中高谈阔论。\n\n"
                "## 旧事\n\n"
                "此前，她曾为调查而短暂扮演健谈商人。"
            ),
        ),
        authority_tier="core_canon",
    )
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content="林澈主动在宴会上与陌生人长谈。",
    )
    baseline = _baseline().model_copy(
        update={
            "evidence": (
                _baseline().evidence[0].model_copy(
                    update={"line_start": 5, "line_end": 5}
                ),
            )
        }
    )

    candidates, incomplete = _windows(
        [profile, draft], baseline=baseline
    )

    assert incomplete is False
    profile_rows = [
        row.evidence for row in candidates
        if row.evidence.document_id == "profile"
    ]
    assert profile_rows
    assert all(
        not (row.line_start <= 5 <= row.line_end)
        and not (row.line_start <= 3 <= row.line_end)
        for row in profile_rows
    )
    assert any(
        row.line_start <= 9 <= row.line_end
        and "短暂扮演健谈商人" in row.text
        for row in profile_rows
    )


def test_current_observation_is_not_reintroduced_but_adjacent_explanation_survives():
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content=(
            "林澈主动在宴会上与陌生人长谈。\n"
            "此前她接受了公开表达训练，并已完成训练。\n"
            "旁人离开了宴会厅。"
        ),
    )

    candidates, incomplete = _windows([draft])

    assert incomplete is False
    spans = [row.evidence for row in candidates]
    assert spans
    assert all(not (row.line_start <= 1 <= row.line_end) for row in spans)
    assert any(
        row.line_start <= 2 <= row.line_end
        and "公开表达训练" in row.text
        for row in spans
    )


def test_current_behaviour_purpose_line_is_removed_without_becoming_explanation():
    observation = _observation().model_copy(
        update={
            "evidence": _observation().evidence.model_copy(
                update={
                    "text": "林澈主动夸赞守门人，以换取通行。",
                }
            ),
            "statement": "林澈主动夸赞守门人",
        }
    )
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content="林澈主动夸赞守门人，以换取通行。",
    )

    candidates, incomplete = _explanation_window_specs(
        baseline=_baseline(),
        baseline_scope=NarrativeScopeV1(),
        observations=(observation,),
        draft_scopes=(NarrativeScopeV1(),),
        draft_ordinals=(1,),
        draft_document_ids=("draft",),
        documents=[draft],
    )

    assert candidates == ()
    assert incomplete is False


def test_same_line_behaviour_and_second_assertion_fails_coverage_closed():
    line = (
        "林澈主动在宴会上与陌生人长谈。"
        "事后档案说明她当时正在执行潜伏任务。"
    )
    observation = _observation().model_copy(
        update={
            "evidence": _observation().evidence.model_copy(
                update={"text": line}
            )
        }
    )
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content=line,
    )

    candidates, incomplete = _explanation_window_specs(
        baseline=_baseline(),
        baseline_scope=NarrativeScopeV1(),
        observations=(observation,),
        draft_scopes=(NarrativeScopeV1(),),
        draft_ordinals=(1,),
        draft_document_ids=("draft",),
        documents=[draft],
    )

    # The exact physical line may be reviewed only through a server-owned
    # possible-only cap; it can never become independent G/X evidence.
    assert len(candidates) == 1
    assert candidates[0].promotion_cap == "possible_only"
    assert candidates[0].evidence.text == line
    assert incomplete is True


def test_mismatched_baseline_snapshot_fails_explanation_coverage_closed():
    profile = replace(
        _source(
            document_id="profile",
            ordinal=0,
            source_kind="formal_character_profile",
            content="林澈后来已经变得健谈。",
        ),
        authority_tier="core_canon",
    )
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content="林澈主动在宴会上与陌生人长谈。",
    )

    candidates, incomplete = _windows([profile, draft])

    assert incomplete is True
    assert any(
        row.evidence.document_id == "profile" for row in candidates
    )


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

    # The current C line is never an E candidate; the unbound other draft is
    # still enough to make explanation coverage partial.
    assert candidates == ()
    assert incomplete is True


def test_ordered_target_drafts_bind_only_strictly_later_compatible_c_sources():
    early_scope = NarrativeScopeV1(
        release={"key": "chapter-3", "ordinal": 3}
    )
    middle_scope = NarrativeScopeV1(
        release={"key": "chapter-5", "ordinal": 5}
    )
    late_scope = NarrativeScopeV1(
        release={"key": "chapter-7", "ordinal": 7}
    )
    after_scope = NarrativeScopeV1(
        release={"key": "chapter-9", "ordinal": 9}
    )
    early = replace(
        _source(
            document_id="draft-early",
            ordinal=30,
            source_kind="draft",
            content="林澈先在小型聚会上主动攀谈。",
        ),
        scope=early_scope,
    )
    middle = replace(
        _source(
            document_id="draft-middle",
            ordinal=1,
            source_kind="draft",
            content="补记：她从第五章开始以健谈商人的身份执行潜伏。",
        ),
        scope=middle_scope,
    )
    late = replace(
        _source(
            document_id="draft-late",
            ordinal=2,
            source_kind="draft",
            content="林澈后来又在宴会上与陌生人长谈。",
        ),
        scope=late_scope,
    )
    after = replace(
        _source(
            document_id="draft-after",
            ordinal=0,
            source_kind="draft",
            content="第九章才发生的训练不能解释之前的行为。",
        ),
        scope=after_scope,
    )
    early_observation = _observation().model_copy(
        update={
            "evidence": _observation().evidence.model_copy(
                update={
                    "document_id": "draft-early",
                    "document_name": "draft-early.md",
                    "text": "林澈先在小型聚会上主动攀谈。",
                }
            )
        }
    )
    late_observation = _observation().model_copy(
        update={
            "id": "cs_" + "2" * 32,
            "evidence": _observation().evidence.model_copy(
                update={
                    "document_id": "draft-late",
                    "document_name": "draft-late.md",
                    "text": "林澈后来又在宴会上与陌生人长谈。",
                }
            ),
        }
    )

    candidates, incomplete = _explanation_window_specs(
        baseline=_baseline(),
        baseline_scope=NarrativeScopeV1(),
        observations=(early_observation, late_observation),
        draft_scopes=(early_scope, late_scope),
        # Deliberately contradict narrative order: import ordinal is not used.
        draft_ordinals=(30, 2),
        draft_document_ids=("draft-early", "draft-late"),
        documents=[after, middle, late, early],
    )

    assert incomplete is False
    middle_rows = [
        row for row in candidates
        if row.evidence.document_id == "draft-middle"
    ]
    assert middle_rows
    assert all(
        row.eligible_draft_document_ids == ("draft-late",)
        for row in middle_rows
    )
    assert not any(
        row.evidence.document_id == "draft-after" for row in candidates
    )


def _semantic_support(
    *,
    support_id: str,
    kind: str,
    text: str,
    applicable: tuple[str, ...],
    document_id: str = "history",
    line: int = 1,
    explicit: bool = True,
) -> SupportEvidence:
    return SupportEvidence(
        id=f"se_{support_id}",
        kind=kind,
        summary=text,
        explicit=explicit,
        evidence=EvidenceSpan(
            document_id=document_id,
            document_name=f"{document_id}.md",
            line_start=line,
            line_end=line,
            text=text,
        ),
        source_kind="published_history",
        publication_status="published",
        authority_tier="formal_record",
        resolution_state="confirmed",
        source_ordinal=0,
        eligible_draft_document_ids=("draft",),
        selection_basis="semantic_relation_v1",
        applicable_observation_ids=applicable,
    )


def test_other_named_character_cannot_supply_target_definitive_explanation():
    observation_id = _observation().id
    other_actor = _semantic_support(
        support_id="other-actor",
        kind="exception",
        text="祁雾为了潜入商会启动镜面身份。",
        applicable=(observation_id,),
    )
    other_actor_possible = other_actor.model_copy(
        update={
            "id": "se_other-actor-possible",
            "kind": "possible_explanation",
            "explicit": False,
        }
    )
    target_and_other = other_actor.model_copy(
        update={
            "id": "se_both-actors",
            "evidence": other_actor.evidence.model_copy(
                update={"text": "白榆看见祁雾启动镜面身份。"}
            ),
        }
    )

    retained, rejected = _without_cross_actor_definitive_support(
        (other_actor, other_actor_possible, target_and_other),
        target_character_key="白榆",
        actor_literals_by_character={
            "白榆": ("白榆",),
            "祁雾": ("祁雾",),
        },
    )

    assert rejected == 1
    assert retained == (other_actor_possible, target_and_other)


def test_compound_line_debt_requires_explicit_nonoverlapping_gx_for_every_c():
    first = _observation()
    second = first.model_copy(
        update={
            "id": "cs_" + "2" * 32,
            "evidence": first.evidence.model_copy(
                update={
                    "line_start": 3,
                    "line_end": 3,
                    "text": "次日，林澈再次主动与陌生人长谈；侍者关门。",
                }
            ),
        }
    )
    growth_first = _semantic_support(
        support_id="growth-first",
        kind="causal_bridge",
        text="林澈已完成持续六周的公开表达训练。",
        applicable=(first.id,),
    )
    growth_second = growth_first.model_copy(
        update={
            "id": "se_growth-second",
            "applicable_observation_ids": (second.id,),
        }
    )

    assert _compound_explanation_debt_is_closed(
        observations=(first, second),
        compound_observation_ids=(first.id, second.id),
        support_evidence=(growth_first, growth_second),
    )
    assert not _compound_explanation_debt_is_closed(
        observations=(first, second),
        compound_observation_ids=(first.id, second.id),
        support_evidence=(growth_first,),
    )
    assert not _compound_explanation_debt_is_closed(
        observations=(first, second),
        compound_observation_ids=(first.id, second.id),
        support_evidence=(
            growth_first.model_copy(
                update={
                    "kind": "possible_explanation",
                    "explicit": False,
                    "applicable_observation_ids": (first.id, second.id),
                }
            ),
        ),
    )
    overlapping = growth_first.model_copy(
        update={
            "evidence": first.evidence.model_copy(deep=True),
            "applicable_observation_ids": (first.id, second.id),
        }
    )
    assert not _compound_explanation_debt_is_closed(
        observations=(first, second),
        compound_observation_ids=(first.id, second.id),
        support_evidence=(overlapping,),
    )


def test_explanation_window_capacity_defaults_to_48_and_rejects_above_64():
    history = _source(
        document_id="history-medium",
        ordinal=0,
        source_kind="published_history",
        content="\n".join(
            f"历史段落{index}。" + "甲" * 1_980 for index in range(30)
        ),
    )
    draft = _source(
        document_id="draft",
        ordinal=1,
        source_kind="draft",
        content="林澈主动在宴会上与陌生人长谈。",
    )

    candidates, incomplete = _windows([history, draft])

    # The 30 history windows remain eligible; the current draft C line does
    # not consume an explanation-candidate slot.
    assert len(candidates) == 30
    assert incomplete is False
    assert Settings(
        _env_file=None
    ).character_explanation_max_windows_per_case == 48
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            character_explanation_max_windows_per_case=65,
        )
