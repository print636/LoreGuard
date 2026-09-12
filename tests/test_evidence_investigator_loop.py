import hashlib
import json
import re

import httpx
import pytest

from app.config import Settings
from app.domain import EvidenceSpan, IssueCategory, ParsedDirective
from app.evidence_authority import (
    InvestigationScope,
    ScopedEvidenceDocument,
)
from app.evidence_chunks import EvidenceChunker, SnapshotDocumentKey
from app.evidence_investigator import (
    InvestigatorRejected,
    build_investigation_seeds,
    get_candidate_field_contract,
    get_family_semantic_guidance,
)
from app.evidence_investigator_loop import (
    AbstainDecisionTrace,
    CandidateShapeDecisionTrace,
    EvidenceInvestigatorLoopResult,
    EvidenceInvestigatorToolLoop,
    InvestigatorLoopPolicy,
    ReadSpanDecisionTrace,
    SearchEvidenceDecisionTrace,
    SubmitVerdictDecisionTrace,
    clone_evidence_investigator_loop_result,
)
from app.evidence_investigator_state import InvestigatorLimits
from app.provider import (
    OpenAICompatibleProvider,
    ProviderCallTelemetry,
    ProviderRetryExhausted,
    ProviderToolCall,
    ProviderToolCallError,
    RetryPolicy,
    ToolCallResult,
)
from app.usage import estimate_evidence_investigator_tokens


CONTENT = "岚的发色是银色。\n岚在镜中发现自己的发色已经变成黑色。"


def directive(subject="岚", *, predicate="发色", value="银色", line=1, text=None):
    return ParsedDirective(
        kind="fact",
        attrs={
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        },
        evidence=EvidenceSpan(
            document_id="doc-1",
            document_name="chapter.md",
            line_start=line,
            line_end=line,
            text=text or CONTENT.splitlines()[line - 1],
        ),
        provenance_sources=frozenset({"model"}),
    )


def context(*, directives=None):
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=1,
        content_sha256=hashlib.sha256(CONTENT.encode("utf-8")).hexdigest(),
    )
    document = ScopedEvidenceDocument(snapshot=snapshot, content=CONTENT)
    scope = InvestigationScope.create(
        run_id="run-a",
        project_id="project-a",
        documents=(document,),
    )
    seeds = build_investigation_seeds(
        "run-a", directives or [directive()], limit=8
    )
    chunk = EvidenceChunker(
        target_chars=100,
        min_chars=1,
        max_chars=120,
        overlap_chars=0,
    ).chunk(
        project_id=snapshot.project_id,
        document_id=snapshot.document_id,
        document_version=snapshot.document_version,
        content=CONTENT,
        content_sha256=snapshot.content_sha256,
    )[0]
    return scope, seeds, chunk


class Tokens:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return f"tok{self.value:013d}"


class ScriptedProvider:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []

    def complete_with_tools(self, system, user, *, tools, tool_choice, limits):
        request = {
            "system": system,
            "user": user,
            "tools": tools,
            "tool_choice": tool_choice,
            "limits": limits,
        }
        self.requests.append(request)
        if not self.steps:
            raise AssertionError("unexpected provider call")
        step = self.steps.pop(0)
        if callable(step):
            step = step(request)
        if isinstance(step, Exception):
            raise step
        return step


class FakeRetriever:
    def __init__(self, chunks=(), *, error=None):
        self.chunks = tuple(chunks)
        self.error = error
        self.requests = []

    def search(self, *, seed, query, limit):
        self.requests.append((seed, query, limit))
        if self.error is not None:
            raise self.error
        return self.chunks[:limit]


def result(name, arguments, *, prompt_tokens=10, completion_tokens=2, call_id="call_1"):
    return ToolCallResult(
        tool_calls=(
            ProviderToolCall(id=call_id, name=name, arguments=arguments),
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def search_args(seed_ref):
    return {
        "seed_ref": seed_ref,
        "query": "岚的发色是否发生冲突",
        "entity_terms": ["岚"],
    }


def abstain_for_current(request):
    seed_ref = json.loads(request["user"])["current_seed"]["seed_ref"]
    return result(
        "ABSTAIN",
        {"seed_ref": seed_ref, "reason": "no_relevant_evidence"},
        prompt_tokens=4,
        completion_tokens=1,
    )


def estimated_request_charge(request, *, completion_reserve=768):
    tools_json = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request["tools"]
        ],
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return estimate_evidence_investigator_tokens(
        request["system"],
        request["user"],
        tools_json,
        completion_reserve=completion_reserve,
    )


def test_search_read_submit_uses_contextual_native_tools_and_server_bindings():
    scope, seeds, chunk = context()
    seed_ref = seeds[0].seed_ref
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref), prompt_tokens=11, completion_tokens=3),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 2,
                "line_end": 2,
            },
            prompt_tokens=13,
            completion_tokens=2,
        ),
        result(
            "SUBMIT_VERDICT",
            {
                "seed_ref": seed_ref,
                "verdict": "candidate_conflict",
                "candidates": [
                    {
                        "kind": "fact",
                        "span_ref": "span_tok0000000000002",
                        "source_line_start": 2,
                        "source_line_end": 2,
                        "fields": {
                            "subject": "岚",
                            "predicate": "发色",
                            "value": "黑色",
                        },
                    }
                ],
            },
            prompt_tokens=17,
            completion_tokens=4,
        ),
    )
    retriever = FakeRetriever((chunk,))

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=retriever,
        scope=scope,
        seeds=seeds,
        token_factory=Tokens(),
    ).run()

    assert outcome.outcome == "completed"
    assert outcome.reason_code == "completed"
    assert outcome.provider_calls == 3
    assert outcome.reported_prompt_tokens == 11 + 13 + 17
    assert outcome.reported_completion_tokens == 3 + 2 + 4
    assert outcome.charged_tokens == sum(
        estimated_request_charge(request) for request in provider.requests
    )
    assert outcome.charged_tokens > (
        outcome.reported_prompt_tokens + outcome.reported_completion_tokens
    )
    assert outcome.usage_unavailable_calls == 0
    assert len(outcome.envelopes) == 1
    assert outcome.envelopes[0].trusted is False
    assert len(outcome.authorized_candidates) == 1
    binding = outcome.authorized_candidates[0]
    assert binding.candidate.fields == {
        "subject": "岚",
        "predicate": "发色",
        "value": "黑色",
    }
    assert binding.snapshot == chunk.snapshot
    assert binding.span_ref == "span_tok0000000000002"
    assert (
        binding.authorized_span_sha256
        == outcome.envelopes[0].authorized_span_hashes[0]
    )
    assert binding.snapshot.document_id == "doc-1"
    assert binding.line_start == 2
    assert binding.text == CONTENT.splitlines()[1]
    # The existing promotable case is an untimed, completed transition.  The
    # decision contract must not turn transition wording alone into abstention.
    assert "time" not in binding.candidate.fields

    offered = [[tool.name for tool in row["tools"]] for row in provider.requests]
    assert offered == [
        ["SEARCH_EVIDENCE", "ABSTAIN"],
        ["READ_SPAN", "ABSTAIN"],
        ["SUBMIT_VERDICT", "ABSTAIN"],
    ]
    prompts = [json.loads(row["user"]) for row in provider.requests]
    assert [row["current_phase"] for row in prompts] == [
        "search",
        "read",
        "verdict",
    ]
    assert [row["allowed_next_actions"] for row in prompts] == offered
    assert prompts[0]["current_seed"]["allowed_candidate_kinds"] == ["fact"]
    assert prompts[0]["current_seed"]["candidate_field_contracts"] == {
        "fact": {
            "required_fields": ["subject", "predicate", "value"],
            "optional_fields": ["time"],
            "semantic_guidance": (
                "候选必须与 anchor 的 subject、predicate 相同；冲突须为两条肯定事实的 value 不同，"
                "或同一 value 的一肯定一明确否定。只有双方都提供合法、精确且可排序的 time，"
                "并且 time 不同时，才把变更、恢复、更新或替代后的状态视为阶段演进并必须 "
                "ABSTAIN。任一方没有合法精确 time 或 time 相同，仍须按上述真实冲突规则判断，"
                "不得仅因出现状态转变措辞而放弃。"
            ),
        }
    }
    assert "一肯定一明确否定" in prompts[0]["current_seed"][
        "candidate_field_contracts"
    ]["fact"]["semantic_guidance"]
    assert prompts[0]["current_seed"]["family_semantic_guidance"] == (
        "只调查同一主体同一属性的同时冲突：肯定取值互异，或同一取值一肯定一明确否定。"
        "只有两条记录都有合法、精确且可排序的 time，并且 time 不同，才把明确的状态"
        "变更、恢复、更新或替代视为阶段演进并 ABSTAIN。任一方没有合法精确 time 或 "
        "time 相同，仍按真实冲突规则判断，不得仅因状态转变措辞而放弃。"
    )
    assert prompts[0]["current_seed"]["source_line_guidance"] == (
        "一次 READ 后进入 verdict；选择最直接支持完整非 anchor 新记录的最小充分"
        "范围，不要只读背景，且不得覆盖 anchor 证据行。"
    )
    assert "time" not in prompts[0]["current_seed"]["anchor"]["fields"]
    assert [row["remaining_limits"]["round_no"] for row in prompts] == [1, 2, 3]
    assert prompts[1]["observations"] == [
        {
            "kind": "search_result",
            "rank": 1,
            "result_ref": "result_tok0000000000001",
            "line_start": 1,
            "line_end": 2,
            "overlaps_anchor": True,
        }
    ]
    assert prompts[2]["observations"][-1] == {
        "kind": "read_span",
        "span_ref": "span_tok0000000000002",
        "line_start": 2,
        "line_end": 2,
        "absolute_lines": [
            {
                "line_number": 2,
                "text": "岚在镜中发现自己的发色已经变成黑色。",
            }
        ],
    }
    assert "verdict_checklist" not in prompts[0]
    assert "verdict_checklist" not in prompts[1]
    checklist = prompts[2]["verdict_checklist"]
    assert set(checklist) == {
        "submission_boundary",
        "field_copy_rule",
        "rule_preconditions",
        "uncertainty_policy",
    }
    assert "anchor.fields 已由服务端接受" in checklist["submission_boundary"]
    assert "无需重证" in checklist["submission_boundary"]
    assert "必须只提交这一条" in checklist["submission_boundary"]
    assert "没有否认候选关系已发生的阻断" in checklist[
        "submission_boundary"
    ]
    assert "明确否定型 fact 不属于此类阻断" in checklist[
        "submission_boundary"
    ]
    assert "安全兜底" in checklist["submission_boundary"]
    assert "逐字段核对" in checklist["field_copy_rule"]
    assert "identity/join 字段或字段分量" in checklist["field_copy_rule"]
    assert "current_seed.anchor.fields" in checklist["field_copy_rule"]
    assert "合同或服务规范定义" in checklist["field_copy_rule"]
    assert "absolute_lines" in checklist["field_copy_rule"]
    assert "不要求在 absolute_lines 中逐字出现" in checklist["field_copy_rule"]
    for normalized_value in ("scope_action key", "performed", "body_state"):
        assert normalized_value in checklist["field_copy_rule"]
    assert "除此以外的普通自由文本字段才必须" in checklist["field_copy_rule"]
    assert "绝对行号" in checklist["field_copy_rule"]
    assert "time 与 anchor 兼容" in checklist["rule_preconditions"]
    assert "只核对 candidate 新记录" in checklist["rule_preconditions"]
    assert "不重证 anchor" in checklist["rule_preconditions"]
    assert "仍缺失、矛盾或确有多义" in checklist["uncertainty_policy"]
    assert "臆测未陈述例外" in checklist["uncertainty_policy"]
    assert "不是默认答案" in checklist["uncertainty_policy"]
    verdict_tools = {
        tool.name: tool.description for tool in provider.requests[2]["tools"]
    }
    read_tools = {
        tool.name: tool.description for tool in provider.requests[1]["tools"]
    }
    assert "最直接支持一条完整非 anchor 新记录" in read_tools["READ_SPAN"]
    assert "一条最佳非 anchor 新记录" in verdict_tools["SUBMIT_VERDICT"]
    assert "安全兜底" in verdict_tools["SUBMIT_VERDICT"]
    assert "不得用于试探" in verdict_tools["SUBMIT_VERDICT"]
    assert "正确终局" in verdict_tools["ABSTAIN"]
    assert "不是默认答案" in verdict_tools["ABSTAIN"]
    submit_schema = next(
        tool.parameters
        for tool in provider.requests[2]["tools"]
        if tool.name == "SUBMIT_VERDICT"
    )
    assert submit_schema["properties"]["candidates"]["minItems"] == 1
    assert submit_schema["properties"]["candidates"]["maxItems"] == 1
    assert all(
        "current_seed.anchor.fields 是服务端已接受" in request["system"]
        and "必须只 SUBMIT 这一条" in request["system"]
        and "证据没有否认候选关系已发生" in request["system"]
        and "候选本身为明确否定型 fact 不算" in request["system"]
        and "臆测未陈述例外均不是理由" in request["system"]
        and "ABSTAIN 是正确终局" in request["system"]
        and "不是默认或更安全的答案" in request["system"]
        for request in provider.requests
    )
    assert all("无显式否定、" not in request["system"] for request in provider.requests)
    assert all(row["tool_choice"] == "required" for row in provider.requests)
    assert all(row["limits"].max_calls == 1 for row in provider.requests)
    assert len(retriever.requests) == 1
    retrieved_seed, query, limit = retriever.requests[0]
    assert retrieved_seed.seed_ref == seed_ref
    assert query.text == "岚的发色是否发生冲突"
    assert query.entity_terms == ("岚",)
    assert limit == 6
    # Model-visible schemas contain no project/document/snapshot/evidence fields.
    schemas = json.dumps(
        [
            tool.parameters
            for request in provider.requests
            for tool in request["tools"]
        ],
        ensure_ascii=False,
    )
    for forbidden in (
        '"project_id"',
        '"document_id"',
        '"document_name"',
        '"document_version"',
        '"content_sha256"',
        '"evidence"',
        '"role"',
    ):
        assert forbidden not in schemas

    # Safe diagnostics cannot leak source text, model fields or document ids.
    safe_payload = outcome.safe_dict()
    trace = safe_payload["decision_trace"]
    assert set(trace) == {"schema_version", "complete", "actions"}
    assert trace["schema_version"] == "evidence_investigator_safe_trace_v1"
    assert trace["complete"] is True
    assert trace["actions"] == [
        {
            "provider_decision_index": 1,
            "seed_ordinal": 1,
            "phase": "search",
            "action": "SEARCH_EVIDENCE",
            "result_count": 1,
        },
        {
            "provider_decision_index": 2,
            "seed_ordinal": 1,
            "phase": "read",
            "action": "READ_SPAN",
            "document_ref_hash": trace["actions"][1]["document_ref_hash"],
            "line_start": 2,
            "line_end": 2,
            "selected_result_rank": 1,
            "overlaps_anchor": True,
            "covers_entire_result": False,
        },
        {
            "provider_decision_index": 3,
            "seed_ordinal": 1,
            "phase": "verdict",
            "action": "SUBMIT_VERDICT",
            "candidate_count": 1,
            "candidate_shapes": [
                {
                    "kind": "fact",
                    "field_names": ["predicate", "subject", "value"],
                    "source_line_count": 1,
                }
            ],
        },
    ]
    assert re.fullmatch(r"[a-f0-9]{64}", trace["actions"][1]["document_ref_hash"])
    safe = json.dumps(safe_payload, ensure_ascii=False)
    binding_safe = json.dumps(binding.safe_dict(), ensure_ascii=False)
    for secret in (
        CONTENT,
        "岚的发色是否发生冲突",
        "岚",
        "黑色",
        "doc-1",
        "chapter.md",
        seed_ref,
        "result_tok0000000000001",
        "span_tok0000000000002",
        "call_1",
    ):
        assert secret not in safe
        assert secret not in binding_safe
    for secret in (CONTENT, "黑色", "doc-1", "chapter.md"):
        assert secret not in repr(outcome)
        assert secret not in repr(binding)

    class ActivePayloads:
        touched = False

        def __iter__(self):
            self.touched = True
            raise RuntimeError("CANARY-PRODUCT-ITER")

    envelope = outcome.envelopes[0]
    original_payloads = envelope.candidate_payloads
    active_payloads = ActivePayloads()
    object.__setattr__(envelope, "candidate_payloads", active_payloads)
    assert outcome.safe_dict()["decision_trace"]["complete"] is False
    assert active_payloads.touched is False
    object.__setattr__(envelope, "candidate_payloads", original_payloads)

    mismatched_submit = SubmitVerdictDecisionTrace(
        provider_decision_index=3,
        seed_ordinal=1,
        candidate_count=1,
        candidate_shapes=(
            CandidateShapeDecisionTrace(
                kind="fact",
                field_names=("subject",),
                source_line_count=1,
            ),
        ),
    )
    object.__setattr__(
        outcome,
        "decision_trace",
        (*outcome.decision_trace[:2], mismatched_submit),
    )
    with pytest.raises(ValueError, match="loop result"):
        clone_evidence_investigator_loop_result(outcome)


def test_knowledge_evidence_guidance_is_present_in_model_visible_prompt():
    content = "2026-01-01 09:00，岚说出潮门口令。"
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-knowledge",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    scope = InvestigationScope.create(
        run_id="run-knowledge",
        project_id="project-a",
        documents=(ScopedEvidenceDocument(snapshot=snapshot, content=content),),
    )
    anchor = ParsedDirective(
        kind="claims_knows",
        attrs={
            "character": "岚",
            "fact": "潮门口令",
            "time": "2026-01-01 09:00",
            "modality": "reported",
            "source_scope": "character_dialogue",
            "certainty": "certain",
        },
        evidence=EvidenceSpan(
            document_id="doc-knowledge",
            document_name="chapter.md",
            line_start=1,
            line_end=1,
            text=content,
        ),
        provenance_sources=frozenset({"baseline"}),
    )
    seed = build_investigation_seeds("run-knowledge", (anchor,), limit=1)[0]
    provider = ScriptedProvider(abstain_for_current)

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=(seed,),
    ).run()

    assert outcome.outcome == "completed"
    prompt = json.loads(provider.requests[0]["user"])["current_seed"]
    assert prompt["family_semantic_guidance"] == get_family_semantic_guidance(
        IssueCategory.knowledge_without_acquisition
    )
    for kind in ("knows", "claims_knows"):
        assert prompt["candidate_field_contracts"][kind]["semantic_guidance"] == (
            get_candidate_field_contract(kind).semantic_guidance
        )
    family_guidance = prompt["family_semantic_guidance"]
    knows_guidance = prompt["candidate_field_contracts"]["knows"][
        "semantic_guidance"
    ]
    for required in (
        "同一角色对同一知识",
        "claims_knows",
        "更早的精确可排序时间",
        "较晚的实际获知",
        "应提交 knows",
        "猜测、试探、假口令、错误信息",
    ):
        assert required in family_guidance
    for required in (
        "更晚的精确可排序 time",
        "应提交 knows",
        "猜测、试探、假口令、错误信息",
        "仅接触信息载体",
    ):
        assert required in knows_guidance


def test_decision_trace_models_reject_unreachable_or_unbounded_metadata():
    with pytest.raises(ValueError, match="trace position"):
        SearchEvidenceDecisionTrace(
            provider_decision_index=65,
            seed_ordinal=1,
            result_count=0,
        )
    with pytest.raises(ValueError, match="abstain trace"):
        AbstainDecisionTrace(
            provider_decision_index=1,
            seed_ordinal=1,
            phase="search",
            reason="private provider explanation",
        )
    with pytest.raises(ValueError, match="candidate fields"):
        CandidateShapeDecisionTrace(
            kind="fact",
            field_names=("value", "subject"),
            source_line_count=1,
        )
    with pytest.raises(ValueError, match="read trace"):
        ReadSpanDecisionTrace(
            provider_decision_index=1,
            seed_ordinal=1,
            document_ref_hash="a" * 64,
            line_start=1,
            line_end=1,
            selected_result_rank=1,
            overlaps_anchor=1,
            covers_entire_result=False,
        )
    with pytest.raises(ValueError, match="decision trace"):
        EvidenceInvestigatorLoopResult(
            outcome="completed",
            reason_code="completed",
            provider_calls=1,
            abstained_seeds=1,
            executed_tool_calls=1,
            decision_trace=(),
        )


def test_tampered_decision_trace_is_suppressed_without_touching_foreign_objects():
    action = AbstainDecisionTrace(
        provider_decision_index=1,
        seed_ordinal=1,
        phase="search",
        reason="insufficient_evidence",
    )
    outcome = EvidenceInvestigatorLoopResult(
        outcome="completed",
        reason_code="completed",
        provider_calls=1,
        abstained_seeds=1,
        executed_tool_calls=1,
        decision_trace=(action,),
    )

    class ForeignTraceAction:
        touched = False

        @property
        def action(self):
            self.touched = True
            return "CANARY-RAW-PAYLOAD"

    foreign = ForeignTraceAction()
    object.__setattr__(outcome, "decision_trace", (foreign,))

    trace = outcome.safe_dict()["decision_trace"]
    assert trace == {
        "schema_version": "evidence_investigator_safe_trace_v1",
        "complete": False,
        "actions": [],
    }
    assert foreign.touched is False
    assert "CANARY-RAW-PAYLOAD" not in json.dumps(trace)


def test_tampered_nested_trace_values_fail_closed_without_executing_dunders():
    outcome = EvidenceInvestigatorLoopResult(
        outcome="completed",
        reason_code="completed",
        provider_calls=1,
        abstained_seeds=1,
        executed_tool_calls=1,
        decision_trace=(
            AbstainDecisionTrace(
                provider_decision_index=1,
                seed_ordinal=1,
                phase="search",
                reason="insufficient_evidence",
            ),
        ),
    )

    class ActiveValue:
        def __init__(self):
            self.touched = False

        def __hash__(self):
            self.touched = True
            raise RuntimeError("CANARY-HASH")

        def __ne__(self, other):
            self.touched = True
            raise RuntimeError("CANARY-NE")

        def __iter__(self):
            self.touched = True
            raise RuntimeError("CANARY-ITER")

    phase_canary = ActiveValue()
    action = SearchEvidenceDecisionTrace(
        provider_decision_index=1,
        seed_ordinal=1,
        result_count=0,
    )
    object.__setattr__(action, "phase", phase_canary)
    object.__setattr__(outcome, "decision_trace", (action,))
    assert outcome.safe_dict()["decision_trace"]["complete"] is False
    assert phase_canary.touched is False

    shapes_canary = ActiveValue()
    submit = SubmitVerdictDecisionTrace(
        provider_decision_index=1,
        seed_ordinal=1,
        candidate_count=1,
        candidate_shapes=(
            CandidateShapeDecisionTrace(
                kind="fact",
                field_names=("predicate", "subject", "value"),
                source_line_count=1,
            ),
        ),
    )
    object.__setattr__(submit, "candidate_shapes", shapes_canary)
    object.__setattr__(outcome, "decision_trace", (submit,))
    assert outcome.safe_dict()["decision_trace"]["complete"] is False
    assert shapes_canary.touched is False

    field_names_canary = ActiveValue()
    shape = CandidateShapeDecisionTrace(
        kind="fact",
        field_names=("predicate", "subject", "value"),
        source_line_count=1,
    )
    object.__setattr__(shape, "field_names", field_names_canary)
    object.__setattr__(submit, "candidate_shapes", (shape,))
    object.__setattr__(outcome, "decision_trace", (submit,))
    assert outcome.safe_dict()["decision_trace"]["complete"] is False
    assert field_names_canary.touched is False

    hash_canary = ActiveValue()
    object.__setattr__(outcome, "outcome", hash_canary)
    assert outcome.safe_dict()["decision_trace"]["complete"] is False
    assert hash_canary.touched is False

    object.__setattr__(outcome, "outcome", "completed")
    reason_hash_canary = ActiveValue()
    object.__setattr__(outcome, "reason_code", reason_hash_canary)
    assert outcome.safe_dict()["decision_trace"]["complete"] is False
    assert reason_hash_canary.touched is False


@pytest.mark.parametrize("field_name", ["outcome", "reason_code"])
def test_loop_result_rejects_non_string_enum_without_hashing_it(field_name):
    class ActiveHash:
        touched = False

        def __hash__(self):
            self.touched = True
            raise RuntimeError("CANARY-HASH")

    active = ActiveHash()
    values = {
        "outcome": "degraded",
        "reason_code": "provider_timeout",
    }
    values[field_name] = active
    with pytest.raises(ValueError, match="loop (outcome|reason)"):
        EvidenceInvestigatorLoopResult(**values)
    assert active.touched is False


def test_completed_submit_requires_matching_products_and_read_rank_is_bounded():
    shape = CandidateShapeDecisionTrace(
        kind="fact",
        field_names=("predicate", "subject", "value"),
        source_line_count=1,
    )
    with pytest.raises(ValueError, match="trace products"):
        EvidenceInvestigatorLoopResult(
            outcome="completed",
            reason_code="completed",
            provider_calls=3,
            completed_seeds=1,
            executed_tool_calls=3,
            executed_searches=1,
            executed_reads=1,
            decision_trace=(
                SearchEvidenceDecisionTrace(1, 1, 1),
                ReadSpanDecisionTrace(2, 1, "a" * 64, 1, 1, 1, False, True),
                SubmitVerdictDecisionTrace(3, 1, 1, (shape,)),
            ),
        )

    with pytest.raises(ValueError, match="decision trace"):
        EvidenceInvestigatorLoopResult(
            outcome="degraded",
            reason_code="provider_timeout",
            provider_calls=2,
            executed_tool_calls=2,
            executed_searches=1,
            executed_reads=1,
            decision_trace=(
                SearchEvidenceDecisionTrace(1, 1, 1),
                ReadSpanDecisionTrace(2, 1, "a" * 64, 1, 1, 2, False, True),
            ),
        )


def test_loop_result_clone_rejects_active_container_without_iterating_it():
    outcome = EvidenceInvestigatorLoopResult(
        outcome="degraded",
        reason_code="provider_timeout",
    )

    class ActiveIterable:
        touched = False

        def __iter__(self):
            self.touched = True
            raise RuntimeError("CANARY-ACTIVE-CONTAINER")

    active = ActiveIterable()
    object.__setattr__(outcome, "envelopes", active)

    with pytest.raises(ValueError, match="loop result"):
        clone_evidence_investigator_loop_result(outcome)
    assert active.touched is False


@pytest.mark.parametrize(
    ("kind", "required_boundaries"),
    [
        (
            "fact",
            (
                "双方都提供合法、精确且可排序的 time",
                "time 不同",
                "视为阶段演进并必须 ABSTAIN",
                "任一方没有合法精确 time 或 time 相同",
                "仍须按上述真实冲突规则判断",
                "不得仅因出现状态转变措辞而放弃",
            ),
        ),
        (
            "item",
            (
                "许可、授权、计划或演示安排",
                "不等于交接、领取、持有或保管已经发生",
            ),
        ),
        (
            "uses",
            (
                "许可、授权、计划、准备或演示安排",
                "不等于实际使用",
                "anchor 的 owner 在 candidate time 仍适用",
                "不同 user 明确实际使用该 item",
                "应提交 uses",
                "不得臆测未陈述的交接或例外",
            ),
        ),
    ],
)
def test_candidate_guidance_covers_general_temporal_and_modal_boundaries(
    kind, required_boundaries
):
    guidance = get_candidate_field_contract(kind).semantic_guidance

    for boundary in required_boundaries:
        assert boundary in guidance


def test_family_guidance_keeps_fact_and_item_decisions_conservative():
    fact = get_family_semantic_guidance(IssueCategory.fact_conflict)
    item = get_family_semantic_guidance(IssueCategory.item_ownership)

    assert "两条记录都有合法、精确且可排序的 time" in fact
    assert "time 不同" in fact
    assert "阶段演进并 ABSTAIN" in fact
    assert "任一方没有合法精确 time 或 time 相同" in fact
    assert "仍按真实冲突规则判断" in fact
    assert "不得仅因状态转变措辞而放弃" in fact
    assert "许可、授权、计划、准备或演示安排" in item
    assert "不证明交接或实际使用" in item
    assert "anchor 的 owner 在 candidate time 仍适用" in item
    assert "不同 user 明确实际使用该 item" in item
    assert "应提交 uses" in item
    assert "不得臆测未陈述的交接或例外" in item


@pytest.mark.parametrize(
    ("family", "positive_boundary", "negative_boundary"),
    [
        (
            IssueCategory.fact_conflict,
            "肯定取值互异",
            "阶段演进并 ABSTAIN",
        ),
        (
            IssueCategory.location_collision,
            "同一参与者在同一精确时间",
            "只是相邻时刻时必须 ABSTAIN",
        ),
        (
            IssueCategory.knowledge_without_acquisition,
            "应提交 knows",
            "猜测、试探、假口令、错误信息",
        ),
        (
            IssueCategory.item_ownership,
            "应提交 uses",
            "许可、授权、计划、准备或演示安排",
        ),
        (
            IssueCategory.world_rule_conflict,
            "已经完成的规则相关行为",
            "未完成行为不能作为已执行事实",
        ),
    ],
)
def test_five_family_guides_keep_positive_and_negative_decision_boundaries(
    family, positive_boundary, negative_boundary
):
    guidance = get_family_semantic_guidance(family)

    assert positive_boundary in guidance
    assert negative_boundary in guidance


def test_usage_is_reported_immediately_and_checkpoints_wrap_external_work():
    scope, seeds, chunk = context()
    seed_ref = seeds[0].seed_ref
    checkpoints = []
    usage_rows = []
    provider = ScriptedProvider(
        result(
            "SEARCH_EVIDENCE",
            search_args(seed_ref),
            prompt_tokens=7,
            completion_tokens=2,
        ),
        result(
            "ABSTAIN",
            {"seed_ref": seed_ref, "reason": "insufficient_evidence"},
            prompt_tokens=5,
            completion_tokens=1,
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        checkpoint=lambda: checkpoints.append("checkpoint"),
        usage_callback=lambda *row: usage_rows.append(row),
    ).run()

    assert outcome.outcome == "completed"
    expected_charges = [
        estimated_request_charge(request) for request in provider.requests
    ]
    assert outcome.reported_prompt_tokens == 12
    assert outcome.reported_completion_tokens == 3
    assert outcome.charged_tokens == sum(expected_charges)
    assert len(checkpoints) >= 10
    assert [(row[1], row[2], row[3], row[5]) for row in usage_rows] == [
        (True, 7, 2, "success"),
        (True, 5, 1, "success"),
    ]
    assert [row[4] for row in usage_rows] == expected_charges
    for request, expected in zip(provider.requests, expected_charges, strict=True):
        without_schemas = estimate_evidence_investigator_tokens(
            request["system"],
            request["user"],
            "[]",
            completion_reserve=768,
        )
        assert expected > without_schemas
    assert outcome.safe_dict()["decision_trace"]["actions"][-1] == {
        "provider_decision_index": 2,
        "seed_ordinal": 1,
        "phase": "read",
        "action": "ABSTAIN",
        "reason": "insufficient_evidence",
    }


def test_abstain_after_read_records_only_allowlisted_verdict_metadata():
    scope, seeds, chunk = context()
    seed_ref = seeds[0].seed_ref
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 2,
                "line_end": 2,
            },
        ),
        result(
            "ABSTAIN",
            {
                "seed_ref": seed_ref,
                "reason": "no_rule_validated_conflict",
            },
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        token_factory=Tokens(),
        limits=InvestigatorLimits(max_charged_tokens=20_000),
    ).run()

    assert (outcome.outcome, outcome.abstained_seeds) == ("completed", 1)
    terminal = outcome.safe_dict()["decision_trace"]["actions"][-1]
    assert terminal == {
        "provider_decision_index": 3,
        "seed_ordinal": 1,
        "phase": "verdict",
        "action": "ABSTAIN",
        "reason": "no_rule_validated_conflict",
    }


def test_search_observation_marks_non_anchor_chunk_without_leaking_document_id():
    base_scope, seeds, _ = context()
    other_content = "岚的发色变成黑色。"
    other_snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-2",
        document_version=1,
        content_sha256=hashlib.sha256(other_content.encode("utf-8")).hexdigest(),
    )
    scope = InvestigationScope.create(
        run_id="run-a",
        project_id="project-a",
        documents=(
            base_scope.documents[0],
            ScopedEvidenceDocument(snapshot=other_snapshot, content=other_content),
        ),
    )
    other_chunk = EvidenceChunker(
        target_chars=100,
        min_chars=1,
        max_chars=120,
        overlap_chars=0,
    ).chunk(
        project_id="project-a",
        document_id="doc-2",
        document_version=1,
        content=other_content,
        content_sha256=other_snapshot.content_sha256,
    )[0]
    seed_ref = seeds[0].seed_ref
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        result(
            "ABSTAIN",
            {"seed_ref": seed_ref, "reason": "insufficient_evidence"},
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((other_chunk,)),
        scope=scope,
        seeds=seeds,
        token_factory=Tokens(),
    ).run()

    assert outcome.outcome == "completed"
    visible = json.loads(provider.requests[1]["user"])["observations"][0]
    assert visible["rank"] == 1
    assert visible["overlaps_anchor"] is False
    assert "doc-2" not in json.dumps(visible)


def test_anchor_read_is_content_free_recoverable_then_non_anchor_subrange_succeeds():
    scope, seeds, chunk = context()
    seed_ref = seeds[0].seed_ref
    tokens = Tokens()
    provider = ScriptedProvider(
        result(
            "SEARCH_EVIDENCE",
            search_args(seed_ref),
            prompt_tokens=7,
            completion_tokens=2,
        ),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 1,
                "line_end": 1,
            },
            prompt_tokens=8,
            completion_tokens=2,
        ),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 2,
                "line_end": 2,
            },
            prompt_tokens=9,
            completion_tokens=2,
        ),
        result(
            "SUBMIT_VERDICT",
            {
                "seed_ref": seed_ref,
                "verdict": "candidate_conflict",
                "candidates": [
                    {
                        "kind": "fact",
                        "span_ref": "span_tok0000000000002",
                        "source_line_start": 2,
                        "source_line_end": 2,
                        "fields": {
                            "subject": "岚",
                            "predicate": "发色",
                            "value": "黑色",
                        },
                    }
                ],
            },
            prompt_tokens=10,
            completion_tokens=3,
        ),
    )
    retriever = FakeRetriever((chunk,))

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=retriever,
        scope=scope,
        seeds=seeds,
        token_factory=tokens,
        limits=InvestigatorLimits(
            max_decision_rounds=4,
            max_tool_calls=4,
            max_charged_tokens=20_000,
        ),
    ).run()

    assert outcome.outcome == "completed"
    assert outcome.provider_calls == 4
    assert outcome.executed_tool_calls == 3
    assert outcome.executed_searches == 1
    assert outcome.executed_reads == 1
    assert outcome.recoverable_rejections == 1
    assert outcome.charged_tokens == sum(
        estimated_request_charge(request) for request in provider.requests
    )
    assert len(retriever.requests) == 1
    # One result grant and one successful span grant were minted.  The rejected
    # anchor read never reached the authority read path.
    assert tokens.value == 2
    assert outcome.authorized_candidates[0].line_start == 2

    correction = json.loads(provider.requests[2]["user"])
    assert correction["current_phase"] == "read"
    assert correction["allowed_next_actions"] == ["READ_SPAN", "ABSTAIN"]
    assert correction["observations"][-1] == {
        "kind": "retryable_tool_rejection",
        "reason_code": "anchor_evidence_reused",
        "remaining_corrections": 0,
    }
    assert correction["remaining_limits"]["reads"] == 12
    assert correction["remaining_limits"]["corrections"] == 0
    rejection_payload = json.dumps(
        correction["observations"][-1], ensure_ascii=False, sort_keys=True
    )
    for rejected_detail in (
        "result_tok0000000000001",
        '"line_start"',
        '"line_end"',
        CONTENT.splitlines()[0],
    ):
        assert rejected_detail not in rejection_payload
    safe = json.dumps(outcome.safe_dict(), ensure_ascii=False)
    assert CONTENT not in safe
    assert "result_tok0000000000001" not in safe
    trace_actions = outcome.safe_dict()["decision_trace"]["actions"]
    assert [row["provider_decision_index"] for row in trace_actions] == [1, 3, 4]
    assert [row["action"] for row in trace_actions] == [
        "SEARCH_EVIDENCE",
        "READ_SPAN",
        "SUBMIT_VERDICT",
    ]


def test_two_candidate_submit_is_corrected_once_to_one_final_candidate():
    scope, seeds, chunk = context()
    seed_ref = seeds[0].seed_ref

    def submitted_candidate(value):
        return {
            "kind": "fact",
            "span_ref": "span_tok0000000000002",
            "source_line_start": 2,
            "source_line_end": 2,
            "fields": {
                "subject": "岚",
                "predicate": "发色",
                "value": value,
            },
        }

    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 2,
                "line_end": 2,
            },
        ),
        result(
            "SUBMIT_VERDICT",
            {
                "seed_ref": seed_ref,
                "verdict": "candidate_conflict",
                "candidates": [
                    submitted_candidate("黑色"),
                    submitted_candidate("紫色"),
                ],
            },
        ),
        result(
            "SUBMIT_VERDICT",
            {
                "seed_ref": seed_ref,
                "verdict": "candidate_conflict",
                "candidates": [submitted_candidate("黑色")],
            },
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        token_factory=Tokens(),
        limits=InvestigatorLimits(
            max_decision_rounds=4,
            max_charged_tokens=20_000,
        ),
    ).run()

    assert outcome.outcome == "completed"
    assert outcome.provider_calls == 4
    assert outcome.executed_tool_calls == 3
    assert outcome.recoverable_rejections == 1
    assert len(outcome.envelopes) == 1
    assert len(outcome.envelopes[0].candidate_payloads) == 1
    assert len(outcome.authorized_candidates) == 1
    correction = json.loads(provider.requests[3]["user"])
    assert correction["current_phase"] == "verdict"
    assert correction["allowed_next_actions"] == ["SUBMIT_VERDICT", "ABSTAIN"]
    assert correction["observations"][-1] == {
        "kind": "retryable_tool_rejection",
        "reason_code": "invalid_tool_arguments",
        "remaining_corrections": 0,
    }


def test_anchor_read_without_recovery_degrades_before_span_is_minted():
    scope, seeds, chunk = context()
    seed_ref = seeds[0].seed_ref
    tokens = Tokens()
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 1,
                "line_end": 1,
            },
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        token_factory=tokens,
        policy=InvestigatorLoopPolicy(max_recoverable_rejections_per_seed=0),
        limits=InvestigatorLimits(max_charged_tokens=16_000),
    ).run()

    assert (outcome.outcome, outcome.reason_code) == (
        "degraded",
        "anchor_evidence_reused",
    )
    assert outcome.provider_calls == 2
    assert outcome.executed_tool_calls == 1
    assert outcome.executed_reads == 0
    assert outcome.recoverable_rejections == 0
    assert tokens.value == 1


def test_out_of_grant_anchor_probe_does_not_reveal_document_linkage():
    scope, seeds, _ = context()
    seed_ref = seeds[0].seed_ref
    snapshot = scope.documents[0].snapshot
    chunks = EvidenceChunker(
        target_chars=12,
        min_chars=1,
        max_chars=18,
        overlap_chars=0,
    ).chunk(
        project_id=snapshot.project_id,
        document_id=snapshot.document_id,
        document_version=snapshot.document_version,
        content=CONTENT,
        content_sha256=snapshot.content_sha256,
    )
    non_anchor_chunk = next(
        chunk
        for chunk in chunks
        if chunk.line_start == 2 and chunk.line_end == 2
    )
    tokens = Tokens()
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 1,
                "line_end": 1,
            },
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((non_anchor_chunk,)),
        scope=scope,
        seeds=seeds,
        token_factory=tokens,
        limits=InvestigatorLimits(max_charged_tokens=16_000),
    ).run()

    assert (outcome.outcome, outcome.reason_code) == (
        "degraded",
        "evidence_range",
    )
    assert outcome.provider_calls == 2
    assert outcome.executed_tool_calls == 1
    assert outcome.executed_reads == 0
    assert outcome.recoverable_rejections == 0
    assert tokens.value == 1


def test_partial_line_read_preserves_unambiguous_server_character_offsets():
    content = "岚的发色是银色。\n远处的钟连续响了三次。岚的发色突然变成黑色。"
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    scope = InvestigationScope.create(
        run_id="run-a",
        project_id="project-a",
        documents=(ScopedEvidenceDocument(snapshot=snapshot, content=content),),
    )
    seeds = build_investigation_seeds(
        "run-a",
        [directive(line=1, text=content.splitlines()[0])],
        limit=1,
    )
    chunks = EvidenceChunker(
        target_chars=12,
        min_chars=1,
        max_chars=18,
        overlap_chars=0,
    ).chunk(
        project_id=snapshot.project_id,
        document_id=snapshot.document_id,
        document_version=snapshot.document_version,
        content=content,
        content_sha256=snapshot.content_sha256,
    )
    target = next(chunk for chunk in chunks if "黑色" in chunk.text)
    assert target.char_start > 0
    seed_ref = seeds[0].seed_ref
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 2,
                "line_end": 2,
            },
        ),
        result(
            "SUBMIT_VERDICT",
            {
                "seed_ref": seed_ref,
                "verdict": "candidate_conflict",
                "candidates": [
                    {
                        "kind": "fact",
                        "span_ref": "span_tok0000000000002",
                        "source_line_start": 2,
                        "source_line_end": 2,
                        "fields": {
                            "subject": "岚",
                            "predicate": "发色",
                            "value": "黑色",
                        },
                    }
                ],
            },
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((target,)),
        scope=scope,
        seeds=seeds,
        token_factory=Tokens(),
    ).run()

    assert outcome.outcome == "completed"
    binding = outcome.authorized_candidates[0]
    assert (binding.char_start, binding.char_end) == (
        target.char_start,
        target.char_end,
    )
    assert (binding.authorized_span_char_start, binding.authorized_span_char_end) == (
        target.char_start,
        target.char_end,
    )
    assert content[binding.char_start : binding.char_end] == binding.text
    assert binding.text == target.text


def test_checkpoint_cancellation_propagates_after_usage_was_already_recorded():
    class LeaseLost(RuntimeError):
        pass

    scope, seeds, chunk = context()
    checkpoint_calls = 0
    usage_rows = []

    def checkpoint():
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        # before provider, after immediate usage, before tool, before retrieval
        if checkpoint_calls == 4:
            raise LeaseLost("lease lost")

    with pytest.raises(LeaseLost, match="lease lost"):
        EvidenceInvestigatorToolLoop(
            provider=ScriptedProvider(
                result("SEARCH_EVIDENCE", search_args(seeds[0].seed_ref))
            ),
            retriever=FakeRetriever((chunk,)),
            scope=scope,
            seeds=seeds,
            checkpoint=checkpoint,
            usage_callback=lambda *row: usage_rows.append(row),
        ).run()

    assert len(usage_rows) == 1
    assert usage_rows[0][1] is True
    assert usage_rows[0][4] > 12


def test_server_schedules_each_seed_and_model_can_only_abstain_for_current_seed():
    directives = [
        directive("岚", predicate="发色", value="银色", line=1),
        directive(
            "洛",
            predicate="发色",
            value="黑色",
            line=2,
            text=CONTENT.splitlines()[1],
        ),
    ]
    scope, seeds, _ = context(directives=directives)
    provider = ScriptedProvider(*(abstain_for_current for _ in seeds))

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
    ).run()

    assert outcome.outcome == "completed"
    assert outcome.completed_seeds == 0
    assert outcome.abstained_seeds == len(seeds)
    assert outcome.reported_prompt_tokens == 4 * len(seeds)
    assert outcome.reported_completion_tokens == len(seeds)
    assert outcome.charged_tokens == sum(
        estimated_request_charge(request) for request in provider.requests
    )
    prompted_refs = [
        json.loads(row["user"])["current_seed"]["seed_ref"]
        for row in provider.requests
    ]
    assert prompted_refs == [seed.seed_ref for seed in seeds]


def test_multi_seed_trace_binds_only_submitted_seed_products_in_action_order():
    directives = [
        directive("岚", predicate="发色", value="银色", line=1),
        directive(
            "洛",
            predicate="发色",
            value="黑色",
            line=2,
            text=CONTENT.splitlines()[1],
        ),
    ]
    scope, seeds, chunk = context(directives=directives)
    submitted_seed_ref = seeds[1].seed_ref
    provider = ScriptedProvider(
        abstain_for_current,
        result("SEARCH_EVIDENCE", search_args(submitted_seed_ref)),
        result(
            "READ_SPAN",
            {
                "seed_ref": submitted_seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 1,
                "line_end": 1,
            },
        ),
        result(
            "SUBMIT_VERDICT",
            {
                "seed_ref": submitted_seed_ref,
                "verdict": "candidate_conflict",
                "candidates": [
                    {
                        "kind": "fact",
                        "span_ref": "span_tok0000000000002",
                        "source_line_start": 1,
                        "source_line_end": 1,
                        "fields": {
                            "subject": "洛",
                            "predicate": "发色",
                            "value": "银色",
                        },
                    }
                ],
            },
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        token_factory=Tokens(),
        limits=InvestigatorLimits(max_charged_tokens=20_000),
    ).run()

    assert (
        outcome.outcome,
        outcome.reason_code,
        outcome.completed_seeds,
        outcome.abstained_seeds,
    ) == (
        "completed",
        "completed",
        1,
        1,
    )
    assert len(outcome.envelopes) == len(outcome.authorized_candidates) == 1
    actions = outcome.safe_dict()["decision_trace"]["actions"]
    assert [row["seed_ordinal"] for row in actions] == [1, 2, 2, 2]
    assert [row["action"] for row in actions] == [
        "ABSTAIN",
        "SEARCH_EVIDENCE",
        "READ_SPAN",
        "SUBMIT_VERDICT",
    ]
    assert outcome.safe_dict()["submitted_envelopes"] == 1
    assert outcome.safe_dict()["authorized_candidates"] == 1


def test_budget_preflight_exposes_default_eight_seed_structural_mismatch():
    directives = [
        directive(
            f"角色{index}",
            predicate="身份",
            value=f"身份{index}",
        )
        for index in range(8)
    ]
    scope, seeds, _ = context(directives=directives)
    provider = ScriptedProvider()
    loop = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
    )

    preflight = loop.budget_preflight()

    assert preflight.seed_count == 8
    assert preflight.minimum_required_rounds == 8
    assert preflight.minimum_initial_reservation > preflight.max_charged_tokens
    assert preflight.minimum_path_admissible is False
    assert preflight.maximum_local_run_reservation > preflight.max_charged_tokens
    assert preflight.oversized_initial_prompts == 0
    assert provider.requests == []
    safe = json.dumps(preflight.safe_dict(), ensure_ascii=False)
    assert "角色0" not in safe
    assert "chapter.md" not in safe


def test_budget_preflight_accepts_a_small_default_seed_batch():
    scope, seeds, _ = context()
    loop = EvidenceInvestigatorToolLoop(
        provider=ScriptedProvider(),
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
    )

    preflight = loop.budget_preflight()

    assert preflight.seed_count == 1
    assert preflight.minimum_initial_reservation < preflight.max_charged_tokens
    assert preflight.minimum_path_admissible is True


def test_missing_or_zero_usage_stops_instead_of_creating_a_free_loop():
    scope, seeds, chunk = context()
    provider = ScriptedProvider(
        result(
            "SEARCH_EVIDENCE",
            search_args(seeds[0].seed_ref),
            prompt_tokens=0,
            completion_tokens=0,
        )
    )
    retriever = FakeRetriever((chunk,))
    usage_rows = []

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=retriever,
        scope=scope,
        seeds=seeds,
        usage_callback=lambda *row: usage_rows.append(row),
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == "usage_unavailable"
    assert outcome.reported_prompt_tokens == 0
    assert outcome.reported_completion_tokens == 0
    assert outcome.usage_unavailable_calls == 1
    assert outcome.charged_tokens == estimated_request_charge(provider.requests[0])
    assert outcome.envelopes == ()
    assert outcome.authorized_candidates == ()
    assert retriever.requests == []
    assert len(usage_rows) == 1
    assert usage_rows[0][1:] == (
        False,
        0,
        0,
        outcome.charged_tokens,
        "usage_unavailable",
    )


def test_invalid_argument_shape_without_usage_cannot_enter_correction_turn():
    scope, seeds, _ = context()
    malformed = ToolCallResult(
        tool_calls=(
            ProviderToolCall(
                id="call_1",
                name="SEARCH_EVIDENCE",
                arguments=[],  # type: ignore[arg-type]
            ),
        ),
        prompt_tokens=0,
        completion_tokens=0,
    )
    provider = ScriptedProvider(
        malformed,
        result(
            "ABSTAIN",
            {
                "seed_ref": seeds[0].seed_ref,
                "reason": "insufficient_evidence",
            },
        ),
    )
    usage_rows = []

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
        usage_callback=lambda *row: usage_rows.append(row),
    ).run()

    assert (outcome.outcome, outcome.reason_code) == (
        "degraded",
        "usage_unavailable",
    )
    assert outcome.provider_calls == 1
    assert outcome.executed_tool_calls == 0
    assert outcome.recoverable_rejections == 0
    assert outcome.reported_prompt_tokens == 0
    assert outcome.reported_completion_tokens == 0
    assert outcome.usage_unavailable_calls == 1
    assert outcome.charged_tokens == estimated_request_charge(provider.requests[0])
    assert len(provider.steps) == 1
    assert usage_rows == [
        (
            None,
            False,
            0,
            0,
            outcome.charged_tokens,
            "tool_call_arguments_shape",
        )
    ]
    safe = outcome.safe_dict()
    assert safe["charged_tokens"] == outcome.charged_tokens
    assert safe["usage_unavailable_calls"] == 1


def test_foreign_provider_result_is_rejected_without_reading_active_properties():
    class EvilResult:
        telemetry_touched = False

        @property
        def telemetry(self):
            self.telemetry_touched = True
            raise RuntimeError("must not be read")

    scope, seeds, _ = context()
    response = EvilResult()
    provider = ScriptedProvider(response)
    usage_rows = []

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
        usage_callback=lambda *row: usage_rows.append(row),
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == "provider_contract_invalid"
    assert response.telemetry_touched is False
    assert outcome.usage_unavailable_calls == 1
    assert outcome.charged_tokens == estimated_request_charge(provider.requests[0])
    assert usage_rows == [
        (
            None,
            False,
            0,
            0,
            outcome.charged_tokens,
            "tool_response_shape",
        )
    ]


@pytest.mark.parametrize("deleted_field", ["telemetry", "prompt_tokens", "tool_calls"])
def test_forged_exact_result_with_deleted_slot_fails_closed(deleted_field):
    scope, seeds, _ = context()
    response = result(
        "ABSTAIN",
        {
            "seed_ref": seeds[0].seed_ref,
            "reason": "no_relevant_evidence",
        },
    )
    object.__delattr__(response, deleted_field)
    provider = ScriptedProvider(response)
    usage_rows = []

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
        usage_callback=lambda *row: usage_rows.append(row),
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == "provider_contract_invalid"
    assert outcome.usage_unavailable_calls == 1
    assert len(usage_rows) == 1
    assert usage_rows[0][0] is None
    assert usage_rows[0][1:4] == (False, 0, 0)
    assert usage_rows[0][5] == "tool_response_shape"


def test_exact_result_does_not_expose_untrusted_telemetry_to_usage_callback():
    class TelemetryTrap:
        touched = False

        @property
        def prompt_tokens(self):
            self.touched = True
            raise RuntimeError("must not be read")

    scope, seeds, _ = context()
    telemetry = TelemetryTrap()
    response = result(
        "ABSTAIN",
        {
            "seed_ref": seeds[0].seed_ref,
            "reason": "no_relevant_evidence",
        },
    )
    object.__setattr__(response, "telemetry", telemetry)
    usage_rows = []

    outcome = EvidenceInvestigatorToolLoop(
        provider=ScriptedProvider(response),
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
        usage_callback=lambda *row: usage_rows.append(row),
    ).run()

    assert outcome.outcome == "completed"
    assert telemetry.touched is False
    assert len(usage_rows) == 1
    assert usage_rows[0][0] is None
    assert usage_rows[0][1:] == (
        True,
        10,
        2,
        outcome.charged_tokens,
        "success",
    )


def test_valid_provider_telemetry_is_sanitized_and_cloned_for_accounting():
    scope, seeds, _ = context()
    telemetry = ProviderCallTelemetry(
        input_chars=50,
        response_chars=20,
        received_bytes=40,
        prompt_tokens=10,
        completion_tokens=2,
        elapsed_ms=25,
        category="success",
        attempt_no=1,
        http_status=200,
        request_id="request/valid",
    )
    response = result(
        "ABSTAIN",
        {
            "seed_ref": seeds[0].seed_ref,
            "reason": "no_relevant_evidence",
        },
    )
    object.__setattr__(response, "telemetry", telemetry)
    usage_rows = []

    outcome = EvidenceInvestigatorToolLoop(
        provider=ScriptedProvider(response),
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
        usage_callback=lambda *row: usage_rows.append(row),
    ).run()

    assert outcome.outcome == "completed"
    callback_telemetry = usage_rows[0][0]
    assert type(callback_telemetry) is ProviderCallTelemetry
    assert callback_telemetry is not telemetry
    assert callback_telemetry.model_dump() == telemetry.model_dump()


def test_passthrough_exception_is_propagated_without_reading_active_properties():
    class LeaseLost(RuntimeError):
        category_touched = False
        telemetry_touched = False

        @property
        def category(self):
            self.category_touched = True
            raise ValueError("must not be read")

        @property
        def telemetry(self):
            self.telemetry_touched = True
            raise ValueError("must not be read")

    scope, seeds, _ = context()
    failure = LeaseLost("lease lost")
    provider = ScriptedProvider(failure)
    usage_rows = []

    with pytest.raises(LeaseLost) as caught:
        EvidenceInvestigatorToolLoop(
            provider=provider,
            retriever=FakeRetriever(),
            scope=scope,
            seeds=seeds,
            usage_callback=lambda *row: usage_rows.append(row),
            passthrough_exceptions=(LeaseLost,),
        ).run()

    assert caught.value is failure
    assert failure.category_touched is False
    assert failure.telemetry_touched is False
    assert len(usage_rows) == 1
    assert usage_rows[0][0:4] == (None, False, 0, 0)
    assert usage_rows[0][5] == "provider"


def test_oversized_anchor_prompt_is_rejected_before_a_provider_call():
    content = "岚的发色是银色。" + "背景" * 3_000
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    scope = InvestigationScope.create(
        run_id="run-a",
        project_id="project-a",
        documents=(ScopedEvidenceDocument(snapshot=snapshot, content=content),),
    )
    seeds = build_investigation_seeds(
        "run-a", [directive(text=content)], limit=1
    )
    provider = ScriptedProvider(abstain_for_current)

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
        policy=InvestigatorLoopPolicy(max_prompt_bytes=4 * 1024),
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == "prompt_too_large"
    assert outcome.provider_calls == 0
    assert provider.requests == []


@pytest.mark.parametrize(
    ("tool_result", "reason"),
    [
        (ToolCallResult(tool_calls=(), prompt_tokens=3, completion_tokens=1), "no_tool_call"),
        (
            ToolCallResult(
                tool_calls=(
                    ProviderToolCall("call_1", "ABSTAIN", {}),
                    ProviderToolCall("call_2", "ABSTAIN", {}),
                ),
                prompt_tokens=3,
                completion_tokens=1,
            ),
            "multiple_tool_calls",
        ),
        (result("NOT_ALLOWED", {}, prompt_tokens=3, completion_tokens=1), "unknown_tool"),
    ],
)
def test_missing_multiple_and_unknown_tools_fail_closed(tool_result, reason):
    scope, seeds, _ = context()
    provider = ScriptedProvider(tool_result)
    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == reason
    assert outcome.envelopes == ()
    assert outcome.reported_prompt_tokens == 3
    assert outcome.reported_completion_tokens == 1
    assert outcome.charged_tokens == estimated_request_charge(provider.requests[0])


def test_custom_provider_arguments_are_independently_size_bounded():
    scope, seeds, chunk = context()
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seeds[0].seed_ref))
    )
    retriever = FakeRetriever((chunk,))

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=retriever,
        scope=scope,
        seeds=seeds,
        policy=InvestigatorLoopPolicy(
            max_tool_argument_bytes=1,
            max_recoverable_rejections_per_seed=0,
        ),
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == "invalid_tool_arguments"
    assert retriever.requests == []


def test_custom_provider_cyclic_arguments_fail_closed_without_recursion():
    scope, seeds, _ = context()
    cyclic = {"seed_ref": seeds[0].seed_ref}
    cyclic["cycle"] = cyclic
    provider = ScriptedProvider(
        ToolCallResult(
            tool_calls=(ProviderToolCall("call_1", "ABSTAIN", cyclic),),
            prompt_tokens=3,
            completion_tokens=1,
        )
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
        policy=InvestigatorLoopPolicy(max_recoverable_rejections_per_seed=0),
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == "invalid_tool_arguments"


def test_one_invalid_argument_turn_can_be_corrected_without_executing_it():
    scope, seeds, _ = context()
    seed_ref = seeds[0].seed_ref
    provider = ScriptedProvider(
        result(
            "SEARCH_EVIDENCE",
            {"seed_ref": seed_ref, "query": "", "entity_terms": []},
            prompt_tokens=9,
            completion_tokens=2,
        ),
        result(
            "ABSTAIN",
            {"seed_ref": seed_ref, "reason": "insufficient_evidence"},
            prompt_tokens=11,
            completion_tokens=2,
        ),
    )
    retriever = FakeRetriever()

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=retriever,
        scope=scope,
        seeds=seeds,
        limits=InvestigatorLimits(max_charged_tokens=16_000),
    ).run()

    assert outcome.outcome == "completed"
    assert outcome.provider_calls == 2
    assert outcome.executed_tool_calls == 1
    assert outcome.executed_searches == 0
    assert outcome.recoverable_rejections == 1
    assert outcome.charged_tokens == sum(
        estimated_request_charge(request) for request in provider.requests
    )
    assert retriever.requests == []
    correction_prompt = json.loads(provider.requests[1]["user"])
    assert correction_prompt["observations"] == [
        {
            "kind": "retryable_tool_rejection",
            "reason_code": "invalid_tool_arguments",
            "remaining_corrections": 0,
        }
    ]
    assert correction_prompt["remaining_limits"]["corrections"] == 0


def test_second_invalid_argument_turn_degrades_and_discards_any_candidates():
    scope, seeds, _ = context()
    invalid = {"seed_ref": seeds[0].seed_ref, "query": "", "entity_terms": []}
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", invalid),
        result("SEARCH_EVIDENCE", invalid),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
        limits=InvestigatorLimits(max_charged_tokens=16_000),
    ).run()

    assert (outcome.outcome, outcome.reason_code) == (
        "degraded",
        "invalid_tool_arguments",
    )
    assert outcome.provider_calls == 2
    assert outcome.executed_tool_calls == 0
    assert outcome.recoverable_rejections == 1
    assert outcome.envelopes == ()


def test_repeated_search_is_rejected_before_rag_then_one_correction_can_finish():
    scope, seeds, _ = context()
    seed_ref = seeds[0].seed_ref
    repeated = search_args(seed_ref)
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", repeated),
        result("SEARCH_EVIDENCE", repeated),
        result(
            "ABSTAIN",
            {"seed_ref": seed_ref, "reason": "no_relevant_evidence"},
        ),
    )
    retriever = FakeRetriever()

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=retriever,
        scope=scope,
        seeds=seeds,
        limits=InvestigatorLimits(max_charged_tokens=16_000),
    ).run()

    assert outcome.outcome == "completed"
    assert outcome.provider_calls == 3
    assert outcome.executed_tool_calls == 2
    assert outcome.executed_searches == 1
    assert outcome.recoverable_rejections == 1
    assert len(retriever.requests) == 1
    final_prompt = json.loads(provider.requests[2]["user"])
    assert final_prompt["observations"][-1]["reason_code"] == "repeated_action"


def test_repeated_action_correction_is_strictly_single_use():
    scope, seeds, _ = context()
    repeated = search_args(seeds[0].seed_ref)
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", repeated),
        result("SEARCH_EVIDENCE", repeated),
        result("SEARCH_EVIDENCE", repeated),
    )
    retriever = FakeRetriever()

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=retriever,
        scope=scope,
        seeds=seeds,
        limits=InvestigatorLimits(max_charged_tokens=16_000),
    ).run()

    assert (outcome.outcome, outcome.reason_code) == (
        "degraded",
        "repeated_action",
    )
    assert outcome.provider_calls == 3
    assert outcome.executed_tool_calls == 1
    assert outcome.executed_searches == 1
    assert outcome.recoverable_rejections == 1
    assert len(retriever.requests) == 1


def test_phase_gate_blocks_second_search_after_results_before_rag_dispatch():
    scope, seeds, chunk = context()
    seed_ref = seeds[0].seed_ref
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        result(
            "SEARCH_EVIDENCE",
            {
                "seed_ref": seed_ref,
                "query": "岚 黑色 发色 第二次检索",
                "entity_terms": ["岚", "黑色", "发色"],
            },
        ),
    )
    retriever = FakeRetriever((chunk,))

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=retriever,
        scope=scope,
        seeds=seeds,
        limits=InvestigatorLimits(max_charged_tokens=16_000),
    ).run()

    assert (outcome.outcome, outcome.reason_code) == ("degraded", "unknown_tool")
    assert outcome.provider_calls == 2
    assert outcome.executed_tool_calls == 1
    assert outcome.executed_searches == 1
    assert len(retriever.requests) == 1
    assert [tool.name for tool in provider.requests[1]["tools"]] == [
        "READ_SPAN",
        "ABSTAIN",
    ]


def test_degraded_loop_reports_only_tools_that_finished_before_timeout():
    scope, seeds, chunk = context()
    seed_ref = seeds[0].seed_ref
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        ProviderRetryExhausted(
            "safe timeout",
            category="read_timeout",
            attempt_no=1,
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        limits=InvestigatorLimits(max_charged_tokens=16_000),
    ).run()

    assert (outcome.outcome, outcome.reason_code) == (
        "degraded",
        "provider_timeout",
    )
    assert outcome.provider_calls == 2
    assert outcome.executed_tool_calls == 1
    assert outcome.executed_searches == 1
    assert outcome.executed_reads == 0
    safe = outcome.safe_dict()
    assert safe["executed_tool_calls"] == 1
    assert safe["executed_searches"] == 1
    assert safe["executed_reads"] == 0
    assert safe["decision_trace"]["complete"] is True
    assert [
        row["action"] for row in safe["decision_trace"]["actions"]
    ] == ["SEARCH_EVIDENCE"]


def test_degraded_loop_preserves_content_free_trace_through_completed_read():
    scope, seeds, chunk = context()
    seed_ref = seeds[0].seed_ref
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 2,
                "line_end": 2,
            },
        ),
        ProviderRetryExhausted(
            "safe timeout",
            category="read_timeout",
            attempt_no=1,
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        token_factory=Tokens(),
        limits=InvestigatorLimits(max_charged_tokens=20_000),
    ).run()

    assert (outcome.outcome, outcome.reason_code) == (
        "degraded",
        "provider_timeout",
    )
    assert (outcome.executed_tool_calls, outcome.executed_reads) == (2, 1)
    trace = outcome.safe_dict()["decision_trace"]
    assert trace["complete"] is True
    assert [row["action"] for row in trace["actions"]] == [
        "SEARCH_EVIDENCE",
        "READ_SPAN",
    ]
    serialized = json.dumps(trace, ensure_ascii=False)
    for forbidden in (
        CONTENT,
        "岚",
        "黑色",
        seed_ref,
        "result_tok0000000000001",
        "span_tok0000000000002",
    ):
        assert forbidden not in serialized


def test_document_trace_pseudonym_is_stable_within_run_and_changes_across_runs():
    base_scope, _, chunk = context()

    def read_document_hash(run_id: str) -> str:
        scope = InvestigationScope.create(
            run_id=run_id,
            project_id="project-a",
            documents=base_scope.documents,
        )
        seeds = build_investigation_seeds(run_id, [directive()], limit=8)
        seed_ref = seeds[0].seed_ref
        outcome = EvidenceInvestigatorToolLoop(
            provider=ScriptedProvider(
                result("SEARCH_EVIDENCE", search_args(seed_ref)),
                result(
                    "READ_SPAN",
                    {
                        "seed_ref": seed_ref,
                        "result_ref": "result_tok0000000000001",
                        "line_start": 2,
                        "line_end": 2,
                    },
                ),
                result(
                    "ABSTAIN",
                    {
                        "seed_ref": seed_ref,
                        "reason": "no_rule_validated_conflict",
                    },
                ),
            ),
            retriever=FakeRetriever((chunk,)),
            scope=scope,
            seeds=seeds,
            token_factory=Tokens(),
        ).run()
        assert outcome.outcome == "completed"
        return outcome.safe_dict()["decision_trace"]["actions"][1][
            "document_ref_hash"
        ]

    first = read_document_hash("run-a")
    repeated = read_document_hash("run-a")
    another_run = read_document_hash("run-b")

    assert re.fullmatch(r"[a-f0-9]{64}", first)
    assert first == repeated
    assert first != another_run


@pytest.mark.parametrize("invalid", [-1, 2, True, "1"])
def test_recoverable_rejection_limit_is_a_hard_one_turn_cap(invalid):
    with pytest.raises(ValueError, match="recoverable rejection limit"):
        InvestigatorLoopPolicy(
            max_recoverable_rejections_per_seed=invalid  # type: ignore[arg-type]
        )


def test_scope_arguments_and_cross_seed_attempts_are_rejected_before_retrieval():
    scope, seeds, chunk = context(
        directives=[
            directive("岚", line=1),
            directive(
                "洛", line=2, text=CONTENT.splitlines()[1], value="黑色"
            ),
        ]
    )
    extra_scope = {**search_args(seeds[0].seed_ref), "project_id": "other"}
    for arguments, reason in (
        (extra_scope, "invalid_tool_arguments"),
        (search_args(seeds[1].seed_ref), "cross_seed"),
    ):
        retriever = FakeRetriever((chunk,))
        outcome = EvidenceInvestigatorToolLoop(
            provider=ScriptedProvider(result("SEARCH_EVIDENCE", arguments)),
            retriever=retriever,
            scope=scope,
            seeds=seeds,
            policy=InvestigatorLoopPolicy(
                max_recoverable_rejections_per_seed=(
                    0 if reason == "invalid_tool_arguments" else 1
                )
            ),
        ).run()
        assert outcome.outcome == "degraded"
        assert outcome.reason_code == reason
        assert outcome.envelopes == ()
        assert retriever.requests == []


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (
            ProviderRetryExhausted("safe", category="rate_limit", attempt_no=2),
            "provider_rate_limit",
        ),
        (
            ProviderRetryExhausted("safe", category="read_timeout", attempt_no=2),
            "provider_timeout",
        ),
        (
            ProviderToolCallError("safe", category="tool_response_shape"),
            "provider_contract_invalid",
        ),
    ],
)
def test_provider_failures_map_to_allowlisted_content_free_diagnostics(failure, reason):
    scope, seeds, _ = context()
    provider = ScriptedProvider(failure)
    usage_rows = []
    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
        usage_callback=lambda *row: usage_rows.append(row),
    ).run()
    assert outcome.outcome == "degraded"
    assert outcome.reason_code == reason
    assert outcome.safe_dict()["reason_code"] == reason
    assert outcome.envelopes == ()
    assert outcome.reported_prompt_tokens == 0
    assert outcome.reported_completion_tokens == 0
    assert outcome.usage_unavailable_calls == 1
    assert outcome.charged_tokens == estimated_request_charge(provider.requests[0])
    assert usage_rows[0][1:] == (
        False,
        0,
        0,
        outcome.charged_tokens,
        failure.category,
    )


def test_token_admission_and_round_budgets_fail_closed_with_heuristic_charge():
    scope, seeds, chunk = context()
    token_provider = ScriptedProvider(
        result(
            "SEARCH_EVIDENCE",
            search_args(seeds[0].seed_ref),
            prompt_tokens=6,
            completion_tokens=1,
        )
    )
    token_limited = EvidenceInvestigatorToolLoop(
        provider=token_provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        limits=InvestigatorLimits(max_charged_tokens=5),
    ).run()
    assert token_limited.outcome == "degraded"
    assert token_limited.reason_code == "token_budget"
    assert token_limited.provider_calls == 0
    assert token_limited.charged_tokens == 0
    assert token_provider.requests == []

    provider = ScriptedProvider(
        result(
            "SEARCH_EVIDENCE",
            search_args(seeds[0].seed_ref),
            prompt_tokens=1,
            completion_tokens=1,
        )
    )
    round_limited = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        limits=InvestigatorLimits(max_decision_rounds=1),
    ).run()
    assert round_limited.outcome == "degraded"
    assert round_limited.reason_code == "round_budget"
    assert round_limited.provider_calls == 1
    assert round_limited.reported_prompt_tokens == 1
    assert round_limited.reported_completion_tokens == 1
    assert round_limited.charged_tokens == estimated_request_charge(
        provider.requests[0]
    )
    assert round_limited.envelopes == ()


def test_reported_usage_above_reservation_becomes_the_budget_charge():
    scope, seeds, _ = context()
    provider = ScriptedProvider(
        result(
            "ABSTAIN",
            {
                "seed_ref": seeds[0].seed_ref,
                "reason": "insufficient_evidence",
            },
            prompt_tokens=3_000,
            completion_tokens=101,
        )
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
    ).run()

    assert estimated_request_charge(provider.requests[0]) < 3_101
    assert outcome.outcome == "completed"
    assert outcome.reported_prompt_tokens == 3_000
    assert outcome.reported_completion_tokens == 101
    assert outcome.charged_tokens == 3_101


def test_retrieval_failure_and_out_of_scope_chunks_cannot_mint_capabilities():
    scope, seeds, chunk = context()
    failed = EvidenceInvestigatorToolLoop(
        provider=ScriptedProvider(result("SEARCH_EVIDENCE", search_args(seeds[0].seed_ref))),
        retriever=FakeRetriever(error=RuntimeError("private retrieval detail")),
        scope=scope,
        seeds=seeds,
    ).run()
    assert failed.reason_code == "retrieval_failed"
    assert "private retrieval detail" not in json.dumps(failed.safe_dict())

    forged_snapshot = SnapshotDocumentKey(
        project_id="other-project",
        document_id="doc-1",
        document_version=1,
        content_sha256=chunk.snapshot.content_sha256,
    )
    # Recompute a valid id under the forged snapshot through the real chunker.
    forged_chunk = EvidenceChunker(
        target_chars=100, min_chars=1, max_chars=120, overlap_chars=0
    ).chunk(
        project_id="other-project",
        document_id="doc-1",
        document_version=1,
        content=CONTENT,
        content_sha256=forged_snapshot.content_sha256,
    )[0]
    outside = EvidenceInvestigatorToolLoop(
        provider=ScriptedProvider(result("SEARCH_EVIDENCE", search_args(seeds[0].seed_ref))),
        retriever=FakeRetriever((forged_chunk,)),
        scope=scope,
        seeds=seeds,
    ).run()
    assert outside.outcome == "degraded"
    assert outside.reason_code == "snapshot_mismatch"
    assert outside.envelopes == ()


def test_declared_retriever_control_flow_exception_is_re_raised_unchanged():
    class LeaseLost(RuntimeError):
        pass

    scope, seeds, _ = context()
    lost = LeaseLost("lease no longer owned")
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seeds[0].seed_ref))
    )
    usage_rows = []

    with pytest.raises(LeaseLost) as caught:
        EvidenceInvestigatorToolLoop(
            provider=provider,
            retriever=FakeRetriever(error=lost),
            scope=scope,
            seeds=seeds,
            passthrough_exceptions=(LeaseLost,),
            usage_callback=lambda *row: usage_rows.append(row),
        ).run()

    assert caught.value is lost
    # The provider decision is charged before retrieval begins, even though
    # caller-owned cancellation/lease control flow aborts the loop.
    assert len(usage_rows) == 1
    assert usage_rows[0][1] is True
    assert usage_rows[0][4] == estimated_request_charge(provider.requests[0])


def test_unlisted_retriever_exception_is_still_safely_degraded():
    class LeaseLost(RuntimeError):
        pass

    scope, seeds, _ = context()
    outcome = EvidenceInvestigatorToolLoop(
        provider=ScriptedProvider(
            result("SEARCH_EVIDENCE", search_args(seeds[0].seed_ref))
        ),
        retriever=FakeRetriever(error=LeaseLost("private detail")),
        scope=scope,
        seeds=seeds,
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == "retrieval_failed"
    assert "private detail" not in json.dumps(outcome.safe_dict())


def test_declared_provider_control_flow_is_charged_then_re_raised_unchanged():
    class AnalysisCancelled(RuntimeError):
        pass

    scope, seeds, _ = context()
    cancelled = AnalysisCancelled("cancelled")
    provider = ScriptedProvider(cancelled)
    usage_rows = []

    with pytest.raises(AnalysisCancelled) as caught:
        EvidenceInvestigatorToolLoop(
            provider=provider,
            retriever=FakeRetriever(),
            scope=scope,
            seeds=seeds,
            passthrough_exceptions=(AnalysisCancelled,),
            usage_callback=lambda *row: usage_rows.append(row),
        ).run()

    assert caught.value is cancelled
    assert len(usage_rows) == 1
    assert usage_rows[0][1:] == (
        False,
        0,
        0,
        estimated_request_charge(provider.requests[0]),
        "provider",
    )


@pytest.mark.parametrize(
    "passthrough_exceptions",
    [
        [RuntimeError],
        (RuntimeError,) * 9,
        (BaseException,),
        (Exception,),
        (RuntimeError,),
        (ValueError,),
        (ProviderRetryExhausted,),
        (InvestigatorRejected,),
        ("LeaseLost",),
    ],
)
def test_passthrough_exception_contract_rejects_broad_or_invalid_types(
    passthrough_exceptions,
):
    scope, seeds, _ = context()
    with pytest.raises(TypeError, match="passthrough exceptions"):
        EvidenceInvestigatorToolLoop(
            provider=ScriptedProvider(),
            retriever=FakeRetriever(),
            scope=scope,
            seeds=seeds,
            passthrough_exceptions=passthrough_exceptions,
        )


def test_retriever_chunk_alias_cannot_rebind_authorized_candidate_snapshot():
    base_scope, seeds, chunk = context()
    other_snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-2",
        document_version=1,
        content_sha256=hashlib.sha256(CONTENT.encode("utf-8")).hexdigest(),
    )
    scope = InvestigationScope.create(
        run_id="run-a",
        project_id="project-a",
        documents=(
            base_scope.documents[0],
            ScopedEvidenceDocument(snapshot=other_snapshot, content=CONTENT),
        ),
    )
    seed_ref = seeds[0].seed_ref

    class MutatingTokens(Tokens):
        def __call__(self):
            token = super().__call__()
            if self.value == 1:
                object.__setattr__(chunk, "snapshot", other_snapshot)
            return token

    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(seed_ref)),
        result(
            "READ_SPAN",
            {
                "seed_ref": seed_ref,
                "result_ref": "result_tok0000000000001",
                "line_start": 2,
                "line_end": 2,
            },
        ),
        result(
            "SUBMIT_VERDICT",
            {
                "seed_ref": seed_ref,
                "verdict": "candidate_conflict",
                "candidates": [
                    {
                        "kind": "fact",
                        "span_ref": "span_tok0000000000002",
                        "source_line_start": 2,
                        "source_line_end": 2,
                        "fields": {
                            "subject": "岚",
                            "predicate": "发色",
                            "value": "黑色",
                        },
                    }
                ],
            },
        ),
    )

    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        token_factory=MutatingTokens(),
    ).run()

    assert chunk.snapshot == other_snapshot  # the attack actually ran
    assert outcome.outcome == "completed"
    assert outcome.authorized_candidates[0].snapshot.document_id == "doc-1"


def test_a_later_seed_failure_discards_earlier_authorized_candidates():
    scope, seeds, chunk = context(
        directives=[
            directive("岚", line=1),
            directive(
                "洛", line=2, text=CONTENT.splitlines()[1], value="黑色"
            ),
        ]
    )
    first = seeds[0].seed_ref
    non_anchor_line = 2 if seeds[0].anchor.evidence.line_start == 1 else 1
    provider = ScriptedProvider(
        result("SEARCH_EVIDENCE", search_args(first)),
        result(
            "READ_SPAN",
            {
                "seed_ref": first,
                "result_ref": "result_tok0000000000001",
                "line_start": non_anchor_line,
                "line_end": non_anchor_line,
            },
        ),
        result(
            "SUBMIT_VERDICT",
            {
                "seed_ref": first,
                "verdict": "candidate_conflict",
                "candidates": [
                    {
                        "kind": "fact",
                        "span_ref": "span_tok0000000000002",
                        "source_line_start": non_anchor_line,
                        "source_line_end": non_anchor_line,
                        "fields": {
                            "subject": "洛",
                            "predicate": "发色",
                            "value": "黑色",
                        },
                    }
                ],
            },
        ),
        ToolCallResult(tool_calls=(), prompt_tokens=3, completion_tokens=1),
    )
    outcome = EvidenceInvestigatorToolLoop(
        provider=provider,
        retriever=FakeRetriever((chunk,)),
        scope=scope,
        seeds=seeds,
        token_factory=Tokens(),
        limits=InvestigatorLimits(max_charged_tokens=20_000),
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == "no_tool_call"
    assert outcome.completed_seeds == 1
    assert outcome.envelopes == ()
    assert outcome.authorized_candidates == ()


def openai_provider(handler):
    settings = Settings(
        _env_file=None,
        enable_model_extraction=True,
        openai_api_key="test-only",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock",
        provider_timeout_seconds=1,
        provider_max_attempts=1,
        provider_thinking_mode=None,
        provider_max_completion_tokens=None,
    )
    return OpenAICompatibleProvider(
        settings,
        transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(
            max_attempts=1, base_delay_seconds=0, jitter_ratio=0
        ),
        sleep=lambda _: None,
    )


def test_real_openai_compatible_provider_drives_an_abstain_turn():
    scope, seeds, _ = context()

    def handler(request):
        payload = json.loads(request.content)
        assert payload["tool_choice"] == "required"
        assert [row["function"]["name"] for row in payload["tools"]] == [
            "SEARCH_EVIDENCE",
            "ABSTAIN",
        ]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_safe_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ABSTAIN",
                                        "arguments": json.dumps(
                                            {
                                                "seed_ref": seeds[0].seed_ref,
                                                "reason": "insufficient_evidence",
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    outcome = EvidenceInvestigatorToolLoop(
        provider=openai_provider(handler),
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
    ).run()

    assert outcome.outcome == "completed"
    assert outcome.abstained_seeds == 1
    assert outcome.reported_prompt_tokens == 5
    assert outcome.reported_completion_tokens == 2
    assert outcome.charged_tokens > 7
    assert outcome.safe_dict()["decision_trace"]["actions"] == [
        {
            "provider_decision_index": 1,
            "seed_ordinal": 1,
            "phase": "search",
            "action": "ABSTAIN",
            "reason": "insufficient_evidence",
        }
    ]


@pytest.mark.parametrize(
    ("handler", "reason"),
    [
        (
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": None}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            ),
            "no_tool_call",
        ),
        (lambda request: httpx.Response(429), "provider_rate_limit"),
        (
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [{"message": "invalid", "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            ),
            "provider_contract_invalid",
        ),
        (
            lambda request: (_ for _ in ()).throw(
                httpx.ReadTimeout("timed out", request=request)
            ),
            "provider_timeout",
        ),
    ],
)
def test_real_openai_compatible_transport_failures_are_safely_mapped(handler, reason):
    scope, seeds, _ = context()
    outcome = EvidenceInvestigatorToolLoop(
        provider=openai_provider(handler),
        retriever=FakeRetriever(),
        scope=scope,
        seeds=seeds,
    ).run()

    assert outcome.outcome == "degraded"
    assert outcome.reason_code == reason
    assert outcome.envelopes == ()
