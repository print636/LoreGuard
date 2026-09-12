from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.candidate_promotion as candidate_promotion_module
from app.candidate_promotion import (
    CandidateEvidenceResolver,
    CandidatePromotionLimits,
    TrustedDocumentContext,
    promote_investigator_candidates,
)
from app.domain import EvidenceSpan, IssueCategory, ParsedDirective
from app.evidence_authority import (
    EvidenceGrantAuthority,
    InvestigationScope,
    ScopedEvidenceDocument,
)
from app.evidence_chunks import EvidenceChunker, SnapshotDocumentKey
from app.evidence_investigator import (
    CandidateRecordSubmission,
    ReadSpanArgs,
    build_investigation_seeds,
)
from app.evidence_investigator_state import UntrustedCandidateEnvelope
from app.evidence_investigator_loop import (
    CandidateShapeDecisionTrace,
    AuthorizedCandidateBinding,
    EvidenceInvestigatorLoopResult,
    ReadSpanDecisionTrace,
    SearchEvidenceDecisionTrace,
    SubmitVerdictDecisionTrace,
    safe_candidate_trace_field_names,
)
from app.rules import detect_issues


CASES = {
    IssueCategory.fact_conflict: {
        "lines": ("岚的发色是银色。", "岚的发色是黑色。"),
        "baseline": ("fact", {"subject": "岚", "predicate": "发色", "value": "银色"}),
        "candidate": ("fact", {"subject": "岚", "predicate": "发色", "value": "黑色"}),
    },
    IssueCategory.location_collision: {
        "lines": (
            "2026-01-01 20:00，岚在北塔。",
            "2026-01-01 20:00，岚在南港。",
        ),
        "baseline": (
            "event",
            {"time": "2026-01-01 20:00", "location": "北塔", "participants": "岚"},
        ),
        "candidate": (
            "event",
            {"time": "2026-01-01 20:00", "location": "南港", "participants": "岚"},
        ),
    },
    IssueCategory.knowledge_without_acquisition: {
        "lines": (
            "2026-01-01 09:00，岚说出潮门口令。",
            "2026-01-01 12:00，岚才得知潮门口令。",
        ),
        "baseline": (
            "claims_knows",
            {"character": "岚", "fact": "潮门口令", "time": "2026-01-01 09:00"},
        ),
        "candidate": (
            "knows",
            {"character": "岚", "fact": "潮门口令", "time": "2026-01-01 12:00"},
        ),
    },
    IssueCategory.item_ownership: {
        "lines": (
            "2026-01-01 08:00，苏弦保管星钥。",
            "2026-01-01 09:00，岚使用星钥开启舱门。",
        ),
        "baseline": (
            "item",
            {"item": "星钥", "owner": "苏弦", "time": "2026-01-01 08:00"},
        ),
        "candidate": (
            "uses",
            {"item": "星钥", "user": "岚", "time": "2026-01-01 09:00"},
        ),
    },
    IssueCategory.world_rule_conflict: {
        "lines": ("在北塔中，跃迁必然失效。", "岚在北塔中发动跃迁。"),
        "baseline": (
            "world_rule",
            {"key": "scope_action:北塔:跃迁", "value": "disabled"},
        ),
        "candidate": (
            "world_assert",
            {"key": "scope_action:北塔:跃迁", "value": "performed", "actor": "岚"},
        ),
    },
}


def _semantics(kind: str) -> dict[str, str]:
    if kind == "claims_knows":
        return {
            "modality": "reported",
            "source_scope": "character_dialogue",
            "certainty": "certain",
        }
    return {
        "modality": "conditional_rule" if kind == "world_rule" else "asserted",
        "source_scope": "world_rule" if kind == "world_rule" else "narrator",
        "certainty": "certain",
    }


def _directive(kind: str, attrs: dict[str, str], line: int, text: str) -> ParsedDirective:
    return ParsedDirective(
        kind=kind,
        attrs={**attrs, **_semantics(kind), "story_scope": "main"},
        evidence=EvidenceSpan(
            document_id="doc-1",
            document_name="chapter.md",
            line_start=line,
            line_end=line,
            text=text,
        ),
        provenance_sources=frozenset({"baseline"}),
    )


def _completed_result(
    envelopes: tuple[UntrustedCandidateEnvelope, ...],
    bindings: tuple[AuthorizedCandidateBinding, ...],
) -> EvidenceInvestigatorLoopResult:
    if not envelopes:
        return EvidenceInvestigatorLoopResult(
            outcome="completed",
            reason_code="completed",
            envelopes=(),
            authorized_candidates=bindings,
        )
    trace = []
    for seed_ordinal, envelope in enumerate(envelopes, start=1):
        candidate = CandidateRecordSubmission.model_validate_json(
            envelope.candidate_payloads[0]
        )
        decision_index = (seed_ordinal - 1) * 3
        trace.extend(
            (
                SearchEvidenceDecisionTrace(
                    provider_decision_index=decision_index + 1,
                    seed_ordinal=seed_ordinal,
                    result_count=1,
                ),
                ReadSpanDecisionTrace(
                    provider_decision_index=decision_index + 2,
                    seed_ordinal=seed_ordinal,
                    document_ref_hash="a" * 64,
                    line_start=candidate.source_line_start,
                    line_end=candidate.source_line_end,
                    selected_result_rank=1,
                    overlaps_anchor=False,
                    covers_entire_result=True,
                ),
                SubmitVerdictDecisionTrace(
                    provider_decision_index=decision_index + 3,
                    seed_ordinal=seed_ordinal,
                    candidate_count=1,
                    candidate_shapes=(
                        CandidateShapeDecisionTrace(
                            kind=candidate.kind,
                            field_names=safe_candidate_trace_field_names(
                                candidate.kind, candidate.fields
                            ),
                            source_line_count=(
                                candidate.source_line_end
                                - candidate.source_line_start
                                + 1
                            ),
                        ),
                    ),
                ),
            )
        )
    return EvidenceInvestigatorLoopResult(
        outcome="completed",
        reason_code="completed",
        envelopes=envelopes,
        authorized_candidates=bindings,
        provider_calls=len(trace),
        completed_seeds=len({row.seed_ref for row in envelopes}),
        executed_tool_calls=len(trace),
        executed_searches=len(envelopes),
        executed_reads=len(envelopes),
        decision_trace=tuple(trace),
    )


def _prepared_case(
    family: IssueCategory,
    *,
    candidate_fields: dict | None = None,
    candidate_line: int = 2,
    candidate_text: str | None = None,
):
    case = CASES[family]
    lines = list(case["lines"])
    if candidate_text is not None:
        lines[1] = candidate_text
    content = "\n".join(lines)
    baseline_kind, baseline_attrs = case["baseline"]
    baseline = _directive(baseline_kind, baseline_attrs, 1, lines[0])
    seed = build_investigation_seeds("run-a", [baseline])[0]
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=7,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    source = ScopedEvidenceDocument(snapshot=snapshot, content=content)
    scope = InvestigationScope.create(
        run_id="run-a", project_id="project-a", documents=(source,)
    )
    counter = iter(f"TOKEN{index:011d}" for index in range(1, 20))
    authority = EvidenceGrantAuthority(
        scope=scope,
        seeds=(seed,),
        token_factory=lambda: next(counter),
    )
    chunk = EvidenceChunker(
        target_chars=max(1, len(content)),
        min_chars=1,
        max_chars=max(1, len(content)),
        overlap_chars=0,
    ).chunk(
        project_id="project-a",
        document_id="doc-1",
        document_version=7,
        content=content,
        content_sha256=snapshot.content_sha256,
    )[0]
    search_grant = authority.issue_search_grants(seed.seed_ref, (chunk,))[0]
    span = authority.read_span(
        ReadSpanArgs(
            seed_ref=seed.seed_ref,
            result_ref=search_grant.result_ref,
            line_start=candidate_line,
            line_end=candidate_line,
        ),
        max_read_lines=2,
    )
    candidate_kind, default_fields = case["candidate"]
    candidate = CandidateRecordSubmission.model_validate(
        {
            "kind": candidate_kind,
            "span_ref": span.span_ref,
            "source_line_start": candidate_line,
            "source_line_end": candidate_line,
            "fields": candidate_fields if candidate_fields is not None else default_fields,
        }
    )
    envelope = UntrustedCandidateEnvelope.create(
        seed_ref=seed.seed_ref,
        candidates=(candidate,),
        authorized_span_hashes=(span.text_sha256,),
    )
    binding = AuthorizedCandidateBinding(
        seed_ref=seed.seed_ref,
        candidate_payload=envelope.candidate_payloads[0],
        span_ref=candidate.span_ref,
        authorized_span_sha256=span.text_sha256,
        snapshot=snapshot,
        line_start=candidate_line,
        line_end=candidate_line,
        char_start=span.char_start,
        char_end=span.char_end,
        text=lines[candidate_line - 1],
        text_sha256=hashlib.sha256(
            lines[candidate_line - 1].encode("utf-8")
        ).hexdigest(),
        authorized_span_char_start=span.char_start,
        authorized_span_char_end=span.char_end,
    )
    resolver = CandidateEvidenceResolver(
        scope=scope,
        documents=(
            TrustedDocumentContext(
                document_id="doc-1",
                document_name="chapter.md",
                story_scope="main",
                document_role="chapter",
            ),
        ),
        investigator_result=_completed_result((envelope,), (binding,)),
    )
    return baseline, seed, envelope, resolver


def _promote(family: IssueCategory, **kwargs):
    baseline, seed, envelope, resolver = _prepared_case(family, **kwargs)
    result = promote_investigator_candidates(
        baseline_directives=(baseline,),
        baseline_issues=detect_issues([baseline]),
        envelopes=(envelope,),
        seeds=(seed,),
        evidence_resolver=resolver,
    )
    return result, baseline, envelope, seed, resolver


@pytest.mark.parametrize("family", list(IssueCategory))
def test_all_five_rule_families_require_and_pass_deterministic_reproduction(family):
    result, _, _, _, _ = _promote(family)

    assert result.accepted_candidates == 1
    assert len(result.promoted_directives) == 1
    assert len(result.added_issues) == 1
    assert result.added_issues[0].category == family
    candidate = result.promoted_directives[0]
    assert candidate.provenance_sources == frozenset({"model"})
    assert candidate.evidence.document_name == "chapter.md"
    assert candidate.attrs["story_scope"] == "main"
    assert candidate.attrs["document_role"] == "chapter"
    assert any(
        span.model_dump() == candidate.evidence.model_dump()
        for span in result.added_issues[0].evidence
    )
    assert result.issues[-1].id == result.added_issues[0].id


@pytest.mark.parametrize(
    "forbidden",
    [
        "owner",
        "modality",
        "certainty",
        "source_scope",
        "document_role",
        "provenance_sources",
    ],
)
def test_per_kind_positive_allowlist_rejects_model_claimed_server_fields(forbidden):
    fields = {"subject": "岚", "predicate": "发色", "value": "黑色", forbidden: "x"}
    result, *_ = _promote(IssueCategory.fact_conflict, candidate_fields=fields)

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_fields_forbidden": 1}


@pytest.mark.parametrize(
    "reserved", ["project_id", "document_version", "title", "severity", "confidence"]
)
def test_transport_itself_rejects_issue_and_snapshot_claims(reserved):
    with pytest.raises(ValidationError):
        CandidateRecordSubmission.model_validate(
            {
                "kind": "fact",
                "span_ref": f"span_{'X' * 16}",
                "source_line_start": 1,
                "source_line_end": 1,
                "fields": {
                    "subject": "岚",
                    "predicate": "发色",
                    "value": "黑色",
                    reserved: "untrusted",
                },
            }
        )


def test_nested_non_string_and_byte_limits_fail_closed():
    nested, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_fields={"subject": "岚", "predicate": "发色", "value": {"nested": "黑色"}},
    )
    oversized, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_fields={"subject": "岚", "predicate": "发色", "value": "黑" * 180},
    )

    assert dict(nested.rejection_counts) == {"candidate_shape_invalid": 1}
    assert dict(oversized.rejection_counts) == {"candidate_shape_invalid": 1}


def test_ungrounded_value_and_join_mismatch_cannot_reach_rules():
    ungrounded, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_fields={"subject": "岚", "predicate": "发色", "value": "紫色"},
    )
    wrong_join, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_fields={"subject": "苏弦", "predicate": "发色", "value": "黑色"},
    )

    assert dict(ungrounded.rejection_counts) == {"candidate_not_grounded": 1}
    assert dict(wrong_join.rejection_counts) == {"candidate_join_mismatch": 1}


@pytest.mark.parametrize("perception", ["听到", "看到", "读到"])
def test_explicit_perception_of_specific_knowledge_is_grounded(perception):
    result, *_ = _promote(
        IssueCategory.knowledge_without_acquisition,
        candidate_text=f"2026-01-01 12:00，岚{perception}潮门口令。",
    )

    assert result.accepted_candidates == 1
    assert len(result.added_issues) == 1


@pytest.mark.parametrize(
    "candidate_text",
    [
        "2026-01-01 12:00，岚从苏弦处得知潮门口令。",
        "2026-01-01 12:00，岚查阅档案，才得知潮门口令。",
    ],
)
def test_explicit_source_or_reading_bridge_grounds_knowledge(candidate_text):
    result, *_ = _promote(
        IssueCategory.knowledge_without_acquisition,
        candidate_text=candidate_text,
    )

    assert result.accepted_candidates == 1
    assert len(result.added_issues) == 1


@pytest.mark.parametrize(
    "candidate_text",
    [
        "2026-01-01 12:00，小岚得知潮门口令。",
        "2026-01-01 12:00，岚得知潮门口令失效。",
    ],
)
def test_character_or_fact_prefix_does_not_ground_knowledge(candidate_text):
    result, *_ = _promote(
        IssueCategory.knowledge_without_acquisition,
        candidate_text=candidate_text,
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_not_grounded": 1}


@pytest.mark.parametrize(
    "candidate_text",
    [
        "2026-01-01 12:00，岚只接触了写有潮门口令的铜片。",
        "2026-01-01 12:00，岚与苏弦站在潮门口令旁；她拆封了信息载体。",
        "2026-01-01 12:00，岚站在潮门口令旁；她得知了潮门口令。",
    ],
)
def test_carrier_contact_or_pronoun_relation_does_not_ground_knowledge(
    candidate_text,
):
    result, *_ = _promote(
        IssueCategory.knowledge_without_acquisition,
        candidate_text=candidate_text,
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_not_grounded": 1}


@pytest.mark.parametrize(
    "candidate_fields",
    [
        {"character": "苏弦", "fact": "潮门口令", "time": "2026-01-01 12:00"},
        {"character": "岚", "fact": "潮门认证信息", "time": "2026-01-01 12:00"},
    ],
)
def test_related_character_or_broader_knowledge_cannot_replace_anchor_join(
    candidate_fields,
):
    result, *_ = _promote(
        IssueCategory.knowledge_without_acquisition,
        candidate_fields=candidate_fields,
        candidate_text=(
            f"2026-01-01 12:00，{candidate_fields['character']}得知"
            f"{candidate_fields['fact']}。"
        ),
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_join_mismatch": 1}


@pytest.mark.parametrize(
    ("family", "candidate_text", "candidate_fields", "expected_reason"),
    [
        (
            IssueCategory.fact_conflict,
            "岚说赤羽的发色是黑色。",
            {"subject": "岚", "predicate": "发色", "value": "黑色"},
            "candidate_not_grounded",
        ),
        (
            IssueCategory.fact_conflict,
            "岚说赤羽的发色是黑色。",
            {"subject": "赤羽", "predicate": "发色", "value": "黑色"},
            "candidate_join_mismatch",
        ),
        (
            IssueCategory.location_collision,
            "2026-01-01 20:00，苏弦说岚在南港。",
            {
                "time": "2026-01-01 20:00",
                "location": "南港",
                "participants": "岚",
            },
            "candidate_not_grounded",
        ),
        (
            IssueCategory.location_collision,
            "2026-01-01 20:00，岚不在南港。",
            {
                "time": "2026-01-01 20:00",
                "location": "南港",
                "participants": "岚",
            },
            "candidate_not_grounded",
        ),
        (
            IssueCategory.item_ownership,
            "2026-01-01 09:00，苏弦警告岚不要使用星钥。",
            {"item": "星钥", "user": "岚", "time": "2026-01-01 09:00"},
            "candidate_not_grounded",
        ),
        (
            IssueCategory.item_ownership,
            "2026-01-01 09:00，苏弦允许岚使用星钥开启舱门。",
            {"item": "星钥", "user": "岚", "time": "2026-01-01 09:00"},
            "candidate_not_grounded",
        ),
    ],
)
def test_field_cooccurrence_without_direct_relation_cannot_manufacture_conflict(
    family, candidate_text, candidate_fields, expected_reason
):
    result, *_ = _promote(
        family,
        candidate_text=candidate_text,
        candidate_fields=candidate_fields,
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {expected_reason: 1}


@pytest.mark.parametrize(
    ("family", "candidate_text", "candidate_fields"),
    [
        (
            IssueCategory.knowledge_without_acquisition,
            "清晨，岚才得知潮门口令。",
            {"character": "岚", "fact": "潮门口令", "time": "清晨"},
        ),
        (
            IssueCategory.item_ownership,
            "正午，岚使用星钥开启舱门。",
            {"item": "星钥", "user": "岚", "time": "正午"},
        ),
        (
            IssueCategory.world_rule_conflict,
            "清晨，岚在北塔中发动跃迁。",
            {
                "key": "scope_action:北塔:跃迁",
                "value": "performed",
                "actor": "岚",
                "time": "清晨",
            },
        ),
        (
            IssueCategory.fact_conflict,
            "2026-02-31 09:00，岚的发色是黑色。",
            {
                "subject": "岚",
                "predicate": "发色",
                "value": "黑色",
                "time": "2026-02-31 09:00",
            },
        ),
    ],
)
def test_only_valid_sortable_timestamps_can_enter_deterministic_rules(
    family, candidate_text, candidate_fields
):
    result, *_ = _promote(
        family,
        candidate_text=candidate_text,
        candidate_fields=candidate_fields,
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_shape_invalid": 1}


def test_explicit_permission_then_completed_use_remains_a_grounded_action():
    result, *_ = _promote(
        IssueCategory.item_ownership,
        candidate_text="2026-01-01 09:00，岚获准后使用了星钥开启舱门。",
        candidate_fields={
            "item": "星钥",
            "user": "岚",
            "time": "2026-01-01 09:00",
        },
    )

    assert result.accepted_candidates == 1
    assert len(result.added_issues) == 1


@pytest.mark.parametrize(
    "candidate_text",
    [
        "2026-01-01 09:00，岚在修复室操作了星钥。",
        "2026-01-01 09:00，岚取得授权后实际使用了星钥。",
        "2026-01-01 09:00，星钥被岚启用。",
        "2026-01-01 09:00，岚对星钥进行了操作。",
    ],
)
def test_bound_actual_use_variants_remain_promotable(candidate_text):
    result, *_ = _promote(
        IssueCategory.item_ownership,
        candidate_text=candidate_text,
        candidate_fields={
            "item": "星钥",
            "user": "岚",
            "time": "2026-01-01 09:00",
        },
    )

    assert result.accepted_candidates == 1
    assert len(result.added_issues) == 1


@pytest.mark.parametrize(
    "candidate_text",
    [
        "2026-01-01 09:00，苏岚操作了星钥。",
        "2026-01-01 09:00，岚操作了星钥匙。",
        "2026-01-01 09:00，岚操作了星钥外壳。",
        "2026-01-01 09:00，岚使用星钥启动器后离开。",
        "2026-01-01 09:00，岚使用星钥打开器时停电。",
        "2026-01-01 09:00，用户指南：岚使用星钥时应先登记。",
        "2026-01-01 09:00，系统允许岚操作星钥。",
        "2026-01-01 09:00，管理员授权岚使用星钥。",
        "2026-01-01 09:00，计划如下：岚使用星钥。",
        "2026-01-01 09:00，执行方案如下：岚操作星钥。",
        "2026-01-01 09:00，核验计划：岚使用星钥。",
        "2026-01-01 09:00，撤离方案：岚操作星钥。",
        "2026-01-01 09:00，安全规范：岚使用星钥。",
        "2026-01-01 09:00，设备使用规范：岚操作星钥。",
        "2026-01-01 09:00，战斗预案：岚使用星钥。",
        "2026-01-01 09:00，巡检流程：岚操作星钥。",
        "2026-01-01 09:00，编辑安排：岚使用星钥。",
    ],
)
def test_unbound_or_instructional_use_cannot_be_promoted(candidate_text):
    result, *_ = _promote(
        IssueCategory.item_ownership,
        candidate_text=candidate_text,
        candidate_fields={
            "item": "星钥",
            "user": "岚",
            "time": "2026-01-01 09:00",
        },
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_not_grounded": 1}


@pytest.mark.parametrize(
    "candidate_text",
    [
        "管理员计划将岚的发色设置为黑色。",
        "管理员要求将岚的发色调整为黑色。",
        "核验计划：岚的发色设为黑色。",
        "撤离方案：岚的发色设为黑色。",
        "安全规范：岚的发色设为黑色。",
        "设备使用规范：岚的发色设为黑色。",
        "战斗预案：岚的发色设为黑色。",
        "巡检流程：岚的发色设为黑色。",
        "编辑安排：岚的发色设为黑色。",
        "岚的发色变成黑色长发。",
        "陆岚的发色是黑色。",
    ],
)
def test_planned_transition_or_value_prefix_cannot_ground_fact(candidate_text):
    result, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_text=candidate_text,
        candidate_fields={"subject": "岚", "predicate": "发色", "value": "黑色"},
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_not_grounded": 1}


@pytest.mark.parametrize(
    "candidate_text",
    [
        "管理员最终将岚的发色调整为黑色。",
        "岚的发色，现已调整为黑色。",
        "岚的发色不是银色而是黑色。",
        "岚的发色变成黑色了。",
        "岚的发色是黑色的。",
    ],
)
def test_completed_fact_transition_variants_remain_promotable(candidate_text):
    result, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_text=candidate_text,
        candidate_fields={"subject": "岚", "predicate": "发色", "value": "黑色"},
    )

    assert result.accepted_candidates == 1
    assert len(result.added_issues) == 1


@pytest.mark.parametrize(
    "speech_act",
    [
        "准许",
        "准予",
        "容许",
        "批准",
        "答应",
        "授意",
        "指示",
        "吩咐",
        "催促",
    ],
)
def test_third_party_permission_or_instruction_is_not_an_actual_use(speech_act):
    result, *_ = _promote(
        IssueCategory.item_ownership,
        candidate_text=f"2026-01-01 09:00，苏弦{speech_act}岚使用星钥。",
        candidate_fields={
            "item": "星钥",
            "user": "岚",
            "time": "2026-01-01 09:00",
        },
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_not_grounded": 1}


def test_scope_action_assertion_rejects_free_form_value_cooccurrence():
    result, *_ = _promote(
        IssueCategory.world_rule_conflict,
        candidate_text="岚在北塔发动跃迁，跃迁呈现蓝色。",
        candidate_fields={
            "key": "scope_action:北塔:跃迁",
            "value": "蓝色",
            "actor": "岚",
        },
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_shape_invalid": 1}


@pytest.mark.parametrize(
    "candidate_text",
    [
        "北塔明确拒绝岚的跃迁授权。",
        "岚没有获得在北塔跃迁的许可。",
        "北塔禁止向岚授予跃迁许可。",
    ],
)
def test_denied_scope_action_cannot_be_promoted_as_a_conflicting_world_assert(
    candidate_text,
):
    result, *_ = _promote(
        IssueCategory.world_rule_conflict,
        candidate_text=candidate_text,
        candidate_fields={
            "key": "scope_action:北塔:跃迁",
            "value": "denied",
            "actor": "岚",
        },
    )

    assert result.accepted_candidates == 0
    assert not result.added_issues
    assert not result.issues
    assert dict(result.rejection_counts) == {"candidate_shape_invalid": 1}


def test_authorized_span_hash_and_full_source_line_are_rechecked():
    baseline, seed, envelope, resolver = _prepared_case(IssueCategory.fact_conflict)
    forged = replace(envelope, authorized_span_hashes=("0" * 64,))

    with pytest.raises(ValueError, match="not from the loop result"):
        promote_investigator_candidates(
            baseline_directives=(baseline,),
            baseline_issues=(),
            envelopes=(forged,),
            seeds=(seed,),
            evidence_resolver=resolver,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "seed_ref",
        "span_ref",
        "snapshot",
        "line_range",
        "candidate_offsets",
        "authorized_offsets",
        "authorized_hash",
        "text",
        "candidate_payload",
    ],
)
def test_resolver_reconstructs_and_rejects_every_tampered_binding_field(mutation):
    baseline, seed, envelope, resolver = _prepared_case(IssueCategory.fact_conflict)
    internal = next(iter(resolver._bindings.values()))
    snapshot = next(iter(resolver._documents))
    binding = AuthorizedCandidateBinding(
        seed_ref=internal.seed_ref,
        candidate_payload=internal.candidate_payload,
        span_ref=internal.span_ref,
        snapshot=snapshot,
        line_start=internal.line_start,
        line_end=internal.line_end,
        char_start=internal.char_start,
        char_end=internal.char_end,
        text=internal.text,
        text_sha256=internal.text_sha256,
        authorized_span_char_start=internal.authorized_span_char_start,
        authorized_span_char_end=internal.authorized_span_char_end,
        authorized_span_sha256=internal.authorized_span_sha256,
    )
    if mutation == "seed_ref":
        object.__setattr__(binding, "seed_ref", f"seed_{'0' * 32}")
    elif mutation == "span_ref":
        object.__setattr__(binding, "span_ref", f"span_{'Z' * 16}")
    elif mutation == "snapshot":
        object.__setattr__(
            binding,
            "snapshot",
            SnapshotDocumentKey(
                project_id=snapshot.project_id,
                document_id=snapshot.document_id,
                document_version=snapshot.document_version + 1,
                content_sha256=snapshot.content_sha256,
            ),
        )
    elif mutation == "line_range":
        object.__setattr__(binding, "line_start", 1)
    elif mutation == "candidate_offsets":
        object.__setattr__(binding, "char_start", binding.char_start + 1)
    elif mutation == "authorized_offsets":
        object.__setattr__(
            binding,
            "authorized_span_char_start",
            binding.authorized_span_char_start + 1,
        )
    elif mutation == "authorized_hash":
        object.__setattr__(binding, "authorized_span_sha256", "0" * 64)
    elif mutation == "text":
        object.__setattr__(binding, "text", "岚的发色是紫色。")
        object.__setattr__(
            binding,
            "text_sha256",
            hashlib.sha256(binding.text.encode("utf-8")).hexdigest(),
        )
    else:
        object.__setattr__(
            binding,
            "candidate_payload",
            envelope.candidate_payloads[0] + " ",
        )

    with pytest.raises(ValueError):
        CandidateEvidenceResolver(
            scope=resolver._scope,
            documents=tuple(resolver._contexts.values()),
            investigator_result=_completed_result((envelope,), (binding,)),
        )


def test_partial_line_binding_is_verified_by_offsets_instead_of_expanding_the_line():
    first = "岚的发色是银色。"
    second = "无关前缀；岚的发色是黑色。无关后缀。"
    content = f"{first}\n{second}"
    baseline = _directive(
        "fact", {"subject": "岚", "predicate": "发色", "value": "银色"}, 1, first
    )
    seed = build_investigation_seeds("run-partial", [baseline])[0]
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    scope = InvestigationScope.create(
        run_id="run-partial",
        project_id="project-a",
        documents=(ScopedEvidenceDocument(snapshot=snapshot, content=content),),
    )
    partial_text = "岚的发色是黑色。"
    char_start = content.index(partial_text)
    char_end = char_start + len(partial_text)
    candidate = CandidateRecordSubmission.model_validate(
        {
            "kind": "fact",
            "span_ref": f"span_{'P' * 16}",
            "source_line_start": 2,
            "source_line_end": 2,
            "fields": {"subject": "岚", "predicate": "发色", "value": "黑色"},
        }
    )
    envelope = UntrustedCandidateEnvelope.create(
        seed_ref=seed.seed_ref,
        candidates=(candidate,),
        authorized_span_hashes=(hashlib.sha256(partial_text.encode("utf-8")).hexdigest(),),
    )
    binding = AuthorizedCandidateBinding(
        seed_ref=seed.seed_ref,
        candidate_payload=envelope.candidate_payloads[0],
        span_ref=candidate.span_ref,
        snapshot=snapshot,
        line_start=2,
        line_end=2,
        char_start=char_start,
        char_end=char_end,
        text=partial_text,
        text_sha256=hashlib.sha256(partial_text.encode("utf-8")).hexdigest(),
        authorized_span_char_start=char_start,
        authorized_span_char_end=char_end,
        authorized_span_sha256=hashlib.sha256(partial_text.encode("utf-8")).hexdigest(),
    )
    resolver = CandidateEvidenceResolver(
        scope=scope,
        documents=(
            TrustedDocumentContext(
                document_id="doc-1",
                document_name="chapter.md",
                story_scope="main",
                document_role="chapter",
            ),
        ),
        investigator_result=_completed_result((envelope,), (binding,)),
    )

    result = promote_investigator_candidates(
        baseline_directives=(baseline,),
        baseline_issues=(),
        envelopes=(envelope,),
        seeds=(seed,),
        evidence_resolver=resolver,
    )

    assert result.accepted_candidates == 1
    assert result.promoted_directives[0].evidence.text == partial_text
    assert result.promoted_directives[0].evidence.line_start == 2


def test_same_source_line_is_reused_even_when_at_anchor_text_differs_from_binding():
    declared = "苏弦保管星钥，岚使用星钥。"
    content = f"@item item=星钥 owner=苏弦 | {declared}"
    baseline = _directive(
        "item",
        {"item": "星钥", "owner": "苏弦"},
        1,
        declared,
    )
    seed = build_investigation_seeds("run-at-line", (baseline,))[0]
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    scope = InvestigationScope.create(
        run_id="run-at-line",
        project_id="project-a",
        documents=(ScopedEvidenceDocument(snapshot=snapshot, content=content),),
    )
    candidate = CandidateRecordSubmission.model_validate(
        {
            "kind": "uses",
            "span_ref": f"span_{'R' * 16}",
            "source_line_start": 1,
            "source_line_end": 1,
            "fields": {"item": "星钥", "user": "岚"},
        }
    )
    source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    envelope = UntrustedCandidateEnvelope.create(
        seed_ref=seed.seed_ref,
        candidates=(candidate,),
        authorized_span_hashes=(source_hash,),
    )
    binding = AuthorizedCandidateBinding(
        seed_ref=seed.seed_ref,
        candidate_payload=envelope.candidate_payloads[0],
        span_ref=candidate.span_ref,
        snapshot=snapshot,
        line_start=1,
        line_end=1,
        char_start=0,
        char_end=len(content),
        text=content,
        text_sha256=source_hash,
        authorized_span_char_start=0,
        authorized_span_char_end=len(content),
        authorized_span_sha256=source_hash,
    )
    resolver = CandidateEvidenceResolver(
        scope=scope,
        documents=(
            TrustedDocumentContext(
                document_id="doc-1",
                document_name="chapter.md",
                story_scope="main",
                document_role="chapter",
            ),
        ),
        investigator_result=_completed_result((envelope,), (binding,)),
    )

    result = promote_investigator_candidates(
        baseline_directives=(baseline,),
        baseline_issues=(),
        envelopes=(envelope,),
        seeds=(seed,),
        evidence_resolver=resolver,
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_evidence_reused": 1}


def test_same_evidence_and_non_conflicting_candidate_are_not_promoted():
    reused, *_ = _promote(IssueCategory.fact_conflict, candidate_line=1)
    compatible, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_fields={"subject": "岚", "predicate": "发色", "value": "银色"},
        candidate_text="岚的发色是银色。",
    )

    assert dict(reused.rejection_counts) == {"candidate_evidence_reused": 1}
    assert dict(compatible.rejection_counts) == {"candidate_no_rule_conflict": 1}


def test_question_evidence_is_rejected_by_semantic_gate_even_when_fields_are_visible():
    result, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_fields={"subject": "岚", "predicate": "发色", "value": "黑色"},
        candidate_text="岚的发色是否是黑色？",
    )
    assert dict(result.rejection_counts) == {"candidate_semantics_rejected": 1}


@pytest.mark.parametrize("negator", ["不是", "并非", "不为"])
def test_explicit_same_value_fact_negation_is_grounded_with_negative_polarity(
    negator,
):
    result, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_fields={"subject": "岚", "predicate": "发色", "value": "银色"},
        candidate_text=f"岚的发色{negator}银色。",
    )

    assert result.accepted_candidates == 1
    assert len(result.added_issues) == 1
    assert result.promoted_directives[0].attrs["polarity"] == "negative"


@pytest.mark.parametrize(
    "text",
    [
        "岚的发色不是黑色，银色只是制服颜色。",
        "岚的发色并非不是银色。",
    ],
)
def test_unrelated_or_double_negation_does_not_ground_same_value_fact(text):
    result, *_ = _promote(
        IssueCategory.fact_conflict,
        candidate_fields={"subject": "岚", "predicate": "发色", "value": "银色"},
        candidate_text=text,
    )

    assert result.accepted_candidates == 0
    assert dict(result.rejection_counts) == {"candidate_not_grounded": 1}


def test_json_node_limit_is_independent_of_transport_field_count():
    result, baseline, envelope, seed, resolver = _promote(
        IssueCategory.fact_conflict
    )
    assert result.accepted_candidates == 1
    bounded = promote_investigator_candidates(
        baseline_directives=(baseline,),
        baseline_issues=(),
        envelopes=(envelope,),
        seeds=(seed,),
        evidence_resolver=resolver,
        limits=CandidatePromotionLimits(max_json_nodes=6),
    )
    assert dict(bounded.rejection_counts) == {"candidate_shape_invalid": 1}


def test_duplicate_or_replayed_external_envelopes_are_rejected():
    baseline, seed, envelope, resolver = _prepared_case(IssueCategory.fact_conflict)

    with pytest.raises(ValueError, match="not from the loop result"):
        promote_investigator_candidates(
            baseline_directives=(baseline,),
            baseline_issues=(),
            envelopes=(envelope, envelope),
            seeds=(seed,),
            evidence_resolver=resolver,
        )


def test_baseline_issue_prefix_is_byte_equivalent_and_never_deleted_or_replaced():
    target, seed, envelope, resolver = _prepared_case(IssueCategory.fact_conflict)
    other_one = _directive(
        "fact", {"subject": "苏弦", "predicate": "瞳色", "value": "蓝色"}, 1, "苏弦的瞳色是蓝色。"
    )
    other_one.evidence.document_id = "other-a"
    other_one.evidence.document_name = "other.md"
    other_two = _directive(
        "fact", {"subject": "苏弦", "predicate": "瞳色", "value": "金色"}, 2, "苏弦的瞳色是金色。"
    )
    other_two.evidence.document_id = "other-a"
    other_two.evidence.document_name = "other.md"
    baseline = (other_one, other_two, target)
    baseline_issues = tuple(detect_issues(list(baseline)))
    before = json.dumps(
        [row.model_dump(mode="json") for row in baseline_issues],
        ensure_ascii=False,
        sort_keys=True,
    )

    result = promote_investigator_candidates(
        baseline_directives=baseline,
        baseline_issues=baseline_issues,
        envelopes=(envelope,),
        seeds=(seed,),
        evidence_resolver=resolver,
    )
    after = json.dumps(
        [row.model_dump(mode="json") for row in result.issues[: len(baseline_issues)]],
        ensure_ascii=False,
        sort_keys=True,
    )

    assert before == after
    assert len(result.issues) == len(baseline_issues) + 1


def test_trusted_resolver_requires_complete_exact_document_context():
    baseline, seed, envelope, resolver = _prepared_case(IssueCategory.fact_conflict)
    assert baseline and seed and envelope and resolver
    # The concrete resolver—not candidate JSON—owns every document name.
    with pytest.raises(ValueError, match="do not match"):
        CandidateEvidenceResolver(
            scope=resolver._scope,
            documents=(),
            investigator_result=_completed_result((), ()),
        )


def test_resolver_rejects_duck_typed_or_deserialized_loop_result():
    _, _, _, resolver = _prepared_case(IssueCategory.fact_conflict)
    forged = SimpleNamespace(
        outcome="completed",
        reason_code="completed",
        envelopes=resolver.envelopes,
        authorized_candidates=(),
    )

    with pytest.raises(ValueError, match="completed investigator result"):
        CandidateEvidenceResolver(
            scope=resolver._scope,
            documents=tuple(resolver._contexts.values()),
            investigator_result=forged,
        )


def test_result_diagnostics_never_contain_candidate_or_source_text():
    result, *_ = _promote(IssueCategory.fact_conflict)
    rendered = json.dumps(result.safe_dict(), ensure_ascii=False)
    assert "岚" not in rendered
    assert "黑色" not in rendered
    assert result.safe_dict()["accepted_candidates"] == 1


def test_sensitive_promotion_objects_do_not_leak_story_text_through_repr():
    result, _, envelope, seed, resolver = _promote(IssueCategory.fact_conflict)
    candidate = envelope.candidates[0]
    resolved = resolver.resolve(
        seed_ref=seed.seed_ref,
        candidate=candidate,
        expected_span_hash=envelope.authorized_span_hashes[0],
    )
    binding = next(iter(resolver._bindings.values()))

    for value in (resolved, binding, result):
        rendered = repr(value)
        assert "岚" not in rendered
        assert "黑色" not in rendered
        assert "chapter.md" not in rendered


def test_tampered_envelope_is_rejected_before_json_semantics():
    baseline, seed, envelope, resolver = _prepared_case(IssueCategory.fact_conflict)
    object.__setattr__(envelope, "trusted", True)
    with pytest.raises(ValueError, match="not from the loop result"):
        promote_investigator_candidates(
            baseline_directives=(baseline,),
            baseline_issues=(),
            envelopes=(envelope,),
            seeds=(seed,),
            evidence_resolver=resolver,
        )


def test_multi_candidate_envelope_is_invalid_before_promotion_budgeting():
    _, seed, envelope, _ = _prepared_case(IssueCategory.fact_conflict)
    first_candidate = envelope.candidates[0]
    second_candidate = first_candidate.model_copy(
        update={"span_ref": f"span_{'B' * 16}"},
        deep=True,
    )
    with pytest.raises(ValueError, match="untrusted candidate envelope"):
        UntrustedCandidateEnvelope.create(
            seed_ref=seed.seed_ref,
            candidates=(first_candidate, second_candidate),
            authorized_span_hashes=(
                envelope.authorized_span_hashes[0],
                envelope.authorized_span_hashes[0],
            ),
        )


def test_candidate_identity_is_stable_across_fresh_opaque_refs():
    lines = ("岚的发色是银色。", "岚的发色是黑色。")
    content = "\n".join(lines)
    baseline = _directive(
        "fact", {"subject": "岚", "predicate": "发色", "value": "银色"}, 1, lines[0]
    )
    seed = build_investigation_seeds("run-stable", (baseline,))[0]
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    scope = InvestigationScope.create(
        run_id="run-stable",
        project_id="project-a",
        documents=(ScopedEvidenceDocument(snapshot=snapshot, content=content),),
    )
    contexts = (
        TrustedDocumentContext(
            document_id="doc-1",
            document_name="chapter.md",
            story_scope="main",
            document_role="chapter",
        ),
    )

    def run(span_ref):
        line_number = 2
        candidate = CandidateRecordSubmission.model_validate(
            {
                "kind": "fact",
                "span_ref": span_ref,
                "source_line_start": line_number,
                "source_line_end": line_number,
                "fields": {
                    "subject": "岚",
                    "predicate": "发色",
                    "value": "黑色",
                },
            }
        )
        text = lines[line_number - 1]
        span_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        char_start = len(lines[0]) + 1
        envelope = UntrustedCandidateEnvelope.create(
            seed_ref=seed.seed_ref,
            candidates=(candidate,),
            authorized_span_hashes=(span_hash,),
        )
        binding = AuthorizedCandidateBinding(
            seed_ref=seed.seed_ref,
            candidate_payload=envelope.candidate_payloads[0],
            span_ref=candidate.span_ref,
            snapshot=snapshot,
            line_start=line_number,
            line_end=line_number,
            char_start=char_start,
            char_end=char_start + len(text),
            text=text,
            text_sha256=span_hash,
            authorized_span_char_start=char_start,
            authorized_span_char_end=char_start + len(text),
            authorized_span_sha256=span_hash,
        )
        resolver = CandidateEvidenceResolver(
            scope=scope,
            documents=contexts,
            investigator_result=_completed_result(
                (envelope,),
                (binding,),
            )
        )
        return promote_investigator_candidates(
            baseline_directives=(baseline,),
            baseline_issues=(),
            seeds=(seed,),
            evidence_resolver=resolver,
        )

    first = run(f"span_{'C' * 16}")
    second = run(f"span_{'D' * 16}")

    assert first.accepted_candidates == second.accepted_candidates == 1
    assert first.promoted_directives[0].evidence.line_start == 2
    assert second.promoted_directives[0].evidence.line_start == 2
    assert first.added_issues[0].id == second.added_issues[0].id


def test_three_thousand_same_join_rows_use_one_bounded_rule_recheck(monkeypatch):
    baseline, seed, envelope, resolver = _prepared_case(IssueCategory.fact_conflict)
    calls = []
    real_detect = detect_issues

    def bounded_detect(rows):
        calls.append(len(rows))
        return real_detect(rows)

    monkeypatch.setattr(candidate_promotion_module, "detect_issues", bounded_detect)
    started = time.perf_counter()
    result = promote_investigator_candidates(
        baseline_directives=(baseline,) * 3_000,
        baseline_issues=(),
        seeds=(seed,),
        evidence_resolver=resolver,
    )
    elapsed = time.perf_counter() - started

    assert result.accepted_candidates == 1
    assert len(result.directives) == 3_001
    assert calls == [2]
    assert elapsed < 3.0


def test_elapsed_deadline_discards_an_already_reproduced_candidate_atomically():
    baseline, seed, _, resolver = _prepared_case(IssueCategory.fact_conflict)
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls < 5 else 10.0

    result = promote_investigator_candidates(
        baseline_directives=(baseline,),
        baseline_issues=(),
        seeds=(seed,),
        evidence_resolver=resolver,
        limits=CandidatePromotionLimits(max_elapsed_ms=1),
        monotonic=clock,
    )

    assert result.accepted_candidates == 0
    assert result.promoted_directives == ()
    assert result.added_issues == ()
    assert len(result.directives) == 1
    assert result.issues == ()
    assert dict(result.rejection_counts) == {"candidate_budget": 1}


def test_checkpoint_control_flow_is_never_converted_to_candidate_rejection():
    baseline, seed, _, resolver = _prepared_case(IssueCategory.fact_conflict)

    class LeaseLost(RuntimeError):
        pass

    def checkpoint():
        raise LeaseLost("lease lost")

    with pytest.raises(LeaseLost, match="lease lost"):
        promote_investigator_candidates(
            baseline_directives=(baseline,),
            baseline_issues=(),
            seeds=(seed,),
            evidence_resolver=resolver,
            checkpoint=checkpoint,
        )
