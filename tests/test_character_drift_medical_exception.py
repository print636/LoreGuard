from __future__ import annotations

import pytest

from app.character_drift import (
    CharacterDriftCase,
    CharacterReviewDiagnostics,
    CharacterReviewResult,
    ConfirmedTraitSnapshot,
    ModelDriftDecision,
    SupportEvidence,
    _actual_current_medical_refusal,
    _medical_restriction,
    prepare_character_drift,
    promote_character_drift,
)
from app.character_trait_extraction import CharacterSignal
from app.domain import EvidenceSpan


DRINK_BASELINE = "荀木长期稳定地喜欢热姜梅露，夜航后总会要一杯。"
DRINK_CURRENT = (
    "荀木在灯会当夜高烧，到医所复诊。"
    "医师看过她去年的病历后，当面要求她今天暂停热饮、先喝常温水；"
    "荀木把热姜梅露推回去，说「今天不要热的」，次日退烧后又喝了热姜梅露。"
)
DRINK_HISTORY = (
    "去年夜航时，荀木曾因高烧被医师当面要求当日暂停所有热饮，先喝常温水。"
    "荀木照做，退烧后又恢复了夜航后喝热姜梅露的习惯。"
)
FOOD_BASELINE = "童画长期稳定地喜欢热椒饼，赶集时总会买一块。"
FOOD_CURRENT = (
    "童画在摊位前闻到刚出炉的热椒饼，仍说自己很喜欢这种饼；"
    "但童画记得医师规定的三天禁热禁辣，说明天复查前先不吃热椒饼，"
    "改拿一杯温水。她的拒食只发生在治疗期间。"
)
FOOD_HISTORY = (
    "童画在事故救援中被热灰灼伤咽喉。医师在治疗记录里明确写下："
    "从当天起三天内，童画要避开热和辛辣食物，三天后复查；"
    "童画当天就依医嘱把刚出炉的热椒饼放回摊位，只喝温水。"
)
FOOD_HISTORY_CATEGORY_ONLY = (
    "童画曾被热灰灼伤咽喉，医师明确写下三天内要避开热和辛辣食物；"
    "童画照做，治疗结束后恢复正常进食。"
)


def _span(document_id: str, text: str, line: int = 1) -> EvidenceSpan:
    return EvidenceSpan(
        document_id=document_id,
        document_name=f"{document_id}.md",
        line_start=line,
        line_end=line,
        text=text,
    )


def _case(
    *,
    character: str = "荀木",
    key_object: str = "热姜梅露",
    baseline_text: str = DRINK_BASELINE,
    current_text: str = DRINK_CURRENT,
    history_text: str = DRINK_HISTORY,
    support_overrides: dict | None = None,
    observation_overrides: dict | None = None,
) -> CharacterDriftCase:
    baseline = ConfirmedTraitSnapshot(
        id="ct_food",
        character=character,
        dimension="preference",
        trait_key=key_object,
        statement=f"{character}喜欢{key_object}",
        polarity="positive",
        stability="stable",
        origin="explicit_setting",
        evidence=(_span("profile", baseline_text),),
    )
    observation = CharacterSignal(
        id="cs_" + "a" * 32,
        character=character,
        dimension="preference",
        trait_key=key_object,
        statement=f"{character}暂时不吃或不喝{key_object}",
        polarity="negative",
        stability="situational",
        observation_kind="action",
        key_object=key_object,
        source_kind="draft",
        evidence=_span("current", current_text, 12),
    )
    if observation_overrides:
        observation = observation.model_copy(update=observation_overrides)
    support_payload = {
        "id": "se_medical",
        "kind": "exception",
        "summary": "已发布的医疗临时处置",
        "explicit": True,
        "evidence": _span("history", history_text, 5),
        "source_kind": "published_history",
        "publication_status": "published",
        "authority_tier": "formal_record",
        "resolution_state": "confirmed",
        "source_ordinal": 10,
        "eligible_draft_document_ids": ("current",),
    }
    support_payload.update(support_overrides or {})
    support = SupportEvidence(**support_payload)
    return CharacterDriftCase(
        id="cdc_medical",
        baseline=baseline,
        observations=(observation,),
        support_evidence=(support,),
        scope_compatibility="compatible",
        material_coverage="complete",
    )


def _review(verdict: str, citations: tuple[str, ...]) -> CharacterReviewResult:
    return CharacterReviewResult(
        decision=ModelDriftDecision(
            verdict=verdict,
            explanation="这是已发生的临时医嘱限制。",
            citations=citations,
        ),
        diagnostics=CharacterReviewDiagnostics(outcome="completed", reason="completed"),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "character": "童画",
            "key_object": "热椒饼",
            "baseline_text": FOOD_BASELINE,
            "current_text": FOOD_CURRENT,
            "history_text": FOOD_HISTORY,
        },
        {
            "character": "童画",
            "key_object": "热椒饼",
            "baseline_text": FOOD_BASELINE,
            "current_text": FOOD_CURRENT,
            "history_text": FOOD_HISTORY_CATEGORY_ONLY,
        },
    ],
)
def test_actual_medical_exception_allows_explanation_only(kwargs: dict) -> None:
    prepared = prepare_character_drift(_case(**kwargs))
    assert prepared.reviewer_eligible is True
    assert prepared.candidate_level == "possible"
    assert prepared.reason == "single_medical_exception_review_only"

    explained = promote_character_drift(
        prepared, _review("explained", ("B01", "C01", "X01"))
    )
    assert explained.outcome == "no_issue"
    assert explained.visible is False

    contradicted = promote_character_drift(
        prepared, _review("contradicts", ("B01", "C01"))
    )
    assert contradicted.outcome == "needs_confirmation"
    assert contradicted.outcome != "conflict"


@pytest.mark.parametrize(
    "current_text,history_text,support_overrides,observation_overrides",
    [
        ("荀木把热姜梅露推回去。", DRINK_HISTORY, {}, {}),
        (
            "荀木高烧，医师要求她暂停热饮；叶箫把热姜梅露推回去。",
            DRINK_HISTORY, {}, {},
        ),
        (
            "荀木高烧，医师要求她暂停热饮；荀木看见叶箫把热姜梅露推回去。",
            DRINK_HISTORY, {}, {},
        ),
        (
            "荀木高烧，医师要求她暂停热饮；荀木没有把热姜梅露推回去。",
            DRINK_HISTORY, {}, {},
        ),
        (
            "荀木高烧，医师要求叶箫暂停热饮；荀木把热姜梅露推回去。",
            DRINK_HISTORY, {}, {},
        ),
        (
            "荀木高烧，医师要求她暂停热饮；荀木计划改天不喝热姜梅露。",
            DRINK_HISTORY, {}, {},
        ),
        (DRINK_CURRENT, "如果荀木高烧，医师可要求暂停热饮；荀木照做。", {}, {}),
        (DRINK_CURRENT, "荀木高烧，医师要求暂停热饮；荀木没有照做。", {}, {}),
        (DRINK_CURRENT, "荀木高烧，医师要求暂停热饮；叶箫照做。", {}, {}),
        (DRINK_CURRENT, DRINK_HISTORY, {"eligible_draft_document_ids": ("other-draft",)}, {}),
        (DRINK_CURRENT, DRINK_HISTORY, {"publication_status": None}, {}),
        (DRINK_CURRENT, DRINK_HISTORY, {"authority_tier": None}, {}),
        (DRINK_CURRENT, DRINK_HISTORY, {"source_ordinal": None}, {}),
        (DRINK_CURRENT, DRINK_HISTORY, {"evidence": _span("current", DRINK_HISTORY, 5)}, {}),
        (DRINK_CURRENT, DRINK_HISTORY, {}, {"key_object": "热椒饼"}),
    ],
)
def test_medical_exception_fails_closed_on_missing_or_wrong_evidence(
    current_text: str,
    history_text: str,
    support_overrides: dict,
    observation_overrides: dict,
) -> None:
    prepared = prepare_character_drift(
        _case(
            current_text=current_text,
            history_text=history_text,
            support_overrides=support_overrides,
            observation_overrides=observation_overrides,
        )
    )
    assert prepared.reviewer_eligible is False
    assert prepared.reason == "single_behavior_is_not_drift"
    assert promote_character_drift(prepared, None).outcome == "needs_confirmation"


def test_medical_explanation_must_cite_the_qualifying_x() -> None:
    prepared = prepare_character_drift(_case())
    missing_x = promote_character_drift(
        prepared, _review("explained", ("B01", "C01", "G01"))
    )
    assert missing_x.outcome == "unverifiable"
    assert missing_x.reason == "medical_exception_citation_mismatch"


def test_unrelated_x_cannot_substitute_for_matching_medical_x() -> None:
    case = _case()
    relevant = case.support_evidence[0]
    unrelated = relevant.model_copy(
        update={
            "id": "se_other_actor",
            "evidence": _span(
                "other-history",
                "叶箫去年高烧，医师要求她当日暂停热饮；叶箫照做。",
                3,
            ),
        }
    )
    prepared = prepare_character_drift(
        case.model_copy(update={"support_evidence": (unrelated, relevant)})
    )
    assert prepared.reviewer_eligible is True
    assert promote_character_drift(
        prepared, _review("explained", ("B01", "C01", "X01"))
    ).outcome == "unverifiable"
    assert promote_character_drift(
        prepared, _review("explained", ("B01", "C01", "X02"))
    ).outcome == "no_issue"


def test_ordinary_direct_preference_path_is_unchanged() -> None:
    case = _case(observation_overrides={"observation_kind": "preference_expression"})
    prepared = prepare_character_drift(case)
    assert prepared.reason == "reported_opposed_preference"
    assert prepared.candidate_level == "strong"


@pytest.mark.parametrize(
    "reported_action",
    [
        "医生指示荀木把热姜梅露推回去。",
        "医生要求荀木把热姜梅露推回去。",
        "医生命令荀木把热姜梅露推回去。",
        "叶箫转述医生要求荀木把热姜梅露推回去。",
        "叶箫要求荀木把热姜梅露推回去。",
        "荀木应该把热姜梅露推回去。",
        "荀木打算把热姜梅露推回去。",
        "荀木明天会把热姜梅露推回去。",
        "医生指示荀木先不喝热姜梅露，改拿常温水。",
        "叶箫说荀木先不喝热姜梅露，改拿常温水。",
        "荀木说明天先不喝热姜梅露。",
    ],
)
def test_directive_report_or_future_is_not_an_actual_refusal(reported_action: str) -> None:
    text = "荀木发烧，医生要求今天暂停热饮。" + reported_action
    restriction = _medical_restriction(text, "荀木")
    if restriction is not None:
        assert not _actual_current_medical_refusal(
            text,
            character="荀木",
            key_object="热姜梅露",
            restriction_end=restriction[1],
        )
    assert not prepare_character_drift(_case(current_text=text)).reviewer_eligible


@pytest.mark.parametrize(
    "text",
    [
        "荀木在门外等候，叶箫发烧复诊。医生要求她今天暂停热饮；荀木把热姜梅露推回去，改喝常温水。",
        "荀木发烧，医生要求叶箫今天暂停热饮；荀木把热姜梅露推回去。",
        "荀木发烧，叶箫也发烧。医生要求她今天暂停热饮；荀木把热姜梅露推回去。",
        "荀木看见叶箫发烧被医生要求今天暂停热饮；荀木把热姜梅露推回去。",
        "荀木发烧，叶箫陪同。医生要求她今天暂停热饮；荀木把热姜梅露推回去。",
    ],
)
def test_medical_restriction_does_not_borrow_another_patients_illness(
    text: str,
) -> None:
    assert _medical_restriction(text, "荀木") is None
    assert not prepare_character_drift(_case(current_text=text)).reviewer_eligible
