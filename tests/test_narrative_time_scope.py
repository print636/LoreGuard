"""Finite temporal membership and physical-state comparisons; no real models."""
from copy import deepcopy
import json

import httpx
import pytest

from app.config import NARRATIVE_TIME_SCOPE_POLICY_V1, SEMANTIC_STATE_POLICY_V1, NARRATIVE_TIME_KNOWLEDGE_POLICY_V1
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import OpenAICompatibleProvider, RetryPolicy
from app.model_extractor import (ModelEnhancedExtractor, BATCH_SYSTEM_PROMPT_NARRATIVE_V1,
                                 BATCH_SYSTEM_PROMPT_NARRATIVE_TIME_SCOPE_V1)
from app.narrative_relations import compare_time_points, bind_relation, NarrativeTimeExtraction, _digest
from app.narrative_time_scope import (NarrativeTimeScopeExtraction, bind_time_scope, member_matches,
    valid_time_scope, validate_time_scope_sources, scope_comparison, NARRATIVE_TIME_SCOPE_INSTRUCTIONS,
    TimeScopeBindingError)
from app.domain import EvidenceSpan, ParsedDirective
from app.chunking import chunk_document
from app.semantic_state import custody_comparisons, SEMANTIC_STATE_INSTRUCTIONS
from app.candidate_normalizer import DeterministicCandidateNormalizer
from tests.test_semantic_state import record, state_fixture
from tests.test_model_extractor import settings, completion


def scope_wire(quote, members, *, relation="same_day_ordered", line=1, omitted=0):
    return dict(kind="narrative_time_scope", source_line_start=line, source_line_end=line,
        modality="asserted", source_scope="narrator", certainty="certain", basis="supported_inference",
        relation_quote=quote, relation=relation, members=members, omitted_count=omitted)


def member(doc_ref, text, time, clock, *, line=1):
    return dict(doc_ref=doc_ref, line_start=line, line_end=line, event_quote=text, time=time,
                clock=clock, clock_relation="equivalent" if clock else "unclear")


def chain_fixture(*, relation="same_day_ordered", last_time="10:00", omitted=0):
    basis = "两处记录的行动发生在同一天，确实按以下先后进行。"
    first = "09:00，库管把铜钥匙交给许舟。"
    second = "当天九点半，许舟把铜钥匙交到沈澈手里。"
    third = f"{last_time}，沈澈用铜钥匙打开库房。"
    docs = [DocumentInput("a", "a.md", basis + "\n" + first, "chapter", "main"),
            DocumentInput("b", "b.md", second + "\n" + third, "chapter", "main")]
    raw = [[record("item", first, dict(item="铜钥匙", owner="许舟"), time="09:00", state="transfer", sender="库管", line=2)],
           [record("item", second, dict(item="铜钥匙", owner="沈澈"), time="当天九点半", state="transfer", sender="许舟"),
            record("uses", third, dict(item="铜钥匙", user="沈澈"), time=last_time, line=2)]]
    scope = scope_wire(basis, [member("d1", first, "09:00", "09:00", line=2),
        member("d2", second, "当天九点半", "09:30"), member("d2", third, last_time, "10:00" if last_time == "10:00" else None, line=2)],
        relation=relation, omitted=omitted)
    raw[0].append(scope)
    return docs, raw


def run_chain(*, enabled=True, mutate=None, **options):
    docs, raw = chain_fixture(**options)
    if mutate:
        mutate(raw)
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return completion(json.dumps({"documents": [dict(doc_ref=f"d{i+1}", records=rows) for i,rows in enumerate(raw)]}, ensure_ascii=False))
    config = settings(provider_max_attempts=1, enable_review_agent=False)
    config._semantic_state_policy = SEMANTIC_STATE_POLICY_V1
    config._narrative_time_knowledge_policy = NARRATIVE_TIME_KNOWLEDGE_POLICY_V1
    config._narrative_time_scope_policy = NARRATIVE_TIME_SCOPE_POLICY_V1 if enabled else None
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0), sleep=lambda _:None)
    result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider), settings=config).run(docs)
    return result, calls, docs


@pytest.mark.parametrize(("relation", "last_time"), [("same_day", "10:00"), ("same_day_ordered", "10:00"), ("ordered", "随后")])
def test_original_phrases_and_bounded_members_complete_actual_handover(relation, last_time):
    result, calls, docs = run_chain(relation=relation, last_time=last_time)
    assert len(calls) == 1
    system, user = [m["content"] for m in calls[0]["messages"]]
    assert system == BATCH_SYSTEM_PROMPT_NARRATIVE_TIME_SCOPE_V1 + SEMANTIC_STATE_INSTRUCTIONS + NARRATIVE_TIME_SCOPE_INSTRUCTIONS
    assert "相互隔离" not in user
    assert "跨行/跨文档拼接替代词面证据" not in system
    assert "时间关系允许引用本次授权输入中多个 doc_ref" in system
    assert "无源或不明时保留 unclear" in system
    assert "同次输出结束前检查" in system and "omitted_count如实记录" in system
    sent = json.loads(user.split("\n", 1)[1])["documents"]
    assert [x["doc_ref"] for x in sent] == ["d1", "d2"]
    assert all(all(line in x["numbered_lines"] for line in doc.content.splitlines()) for x, doc in zip(sent, docs, strict=True))
    assert not [r for r in result.issues if r.category.value == "item_ownership"]
    assert custody_comparisons([r for r in result.directives if r.semantic_state or r.narrative_time_scope]) == ([], 0)
    assert result.diagnostics["semantic_state_comparison"]["unresolved_pairs"] == 0
    scope = next(r for r in result.directives if r.kind == "narrative_time_scope")
    assert valid_time_scope(scope) and validate_time_scope_sources(scope, {d.id:d.content for d in docs})
    assert "narrative_time_scope" not in scope.model_dump()
    assert any(r.attrs.get("time") == "当天九点半" for r in result.directives)
    assert not any("2026" in r.attrs.get("time", "") for r in result.directives)


@pytest.mark.parametrize("mode", ["old_run", "omitted", "unclear", "missing_member", "wrong_source", "clock_disagrees", "too_many"])
def test_uncovered_conflicting_or_unbound_scopes_do_not_complete_time(mode):
    def mutate(raw):
        scope = raw[0][-1]
        if mode == "unclear": scope["relation"] = "unclear"
        if mode == "missing_member": scope["members"] = scope["members"][1:]
        if mode == "wrong_source": scope["members"][0]["event_quote"] = "源外事件"
        if mode == "clock_disagrees": scope["members"][0]["clock"] = "11:00"
        if mode == "too_many": scope["members"] *= 6
    result, calls, _ = run_chain(enabled=mode != "old_run", omitted=1 if mode == "omitted" else 0, mutate=mutate)
    assert len(calls) == 1
    assert result.diagnostics["semantic_state_comparison"]["unresolved_pairs"] > 0
    assert not [r for r in result.issues if r.category.value == "item_ownership"]
    if mode == "old_run":
        assert not any(r.narrative_time_scope for r in result.directives)
        system, user = [m["content"] for m in calls[0]["messages"]]
        assert system == BATCH_SYSTEM_PROMPT_NARRATIVE_V1 + SEMANTIC_STATE_INSTRUCTIONS
        assert user.startswith("批量抽取以下相互隔离的文档。")
        assert "跨行/跨文档拼接替代词面证据" in system
    if mode in {"omitted", "unclear", "wrong_source", "too_many"}:
        assert result.diagnostics["model"]["unresolved_invalid_records"] > 0
    assert result.model_used  # Valid primitive siblings remain admitted.


def test_scope_cannot_bind_another_action_at_same_time_on_the_same_line():
    first_text = "第一回09:00，许舟用铜钥匙开门。"
    second_text = "第二回09:00，沈澈用银钥匙开箱。"
    text = first_text + second_text
    first = state_fixture(text, record("uses", first_text, dict(item="铜钥匙", user="许舟"), time="第一回09:00"))
    second = state_fixture(text, record("uses", second_text, dict(item="银钥匙", user="沈澈"), time="第二回09:00"))
    from app.semantic_state import supported_semantic_state
    assert supported_semantic_state(first) and supported_semantic_state(second)
    start = first_text.index("09:00")
    scope_member = dict(document_id="d",line_start=1,line_end=1,story_scope="main",text_sha256=_digest(text),
        event_ref=dict(start=0,end=len(first_text),text_sha256=_digest(first_text)),
        time_ref=dict(start=start,end=start+5,text_sha256=_digest("09:00")),clock="09:00",clock_relation="same")
    assert member_matches(scope_member, first)
    assert not member_matches(scope_member, second)
    event = ParsedDirective(kind="event", attrs=dict(time="09:00",location="开箱",participants="沈澈",story_scope="main"), evidence=second.evidence)
    assert not member_matches(scope_member, event)


@pytest.mark.parametrize("conflict", ["opposite_order", "different_day"])
def test_contradictory_source_bound_relations_cannot_choose_a_favorable_scope(conflict):
    def mutate(raw):
        original = raw[0][-1]
        if conflict == "opposite_order":
            opposing = deepcopy(original)
            opposing.update(relation="ordered", members=list(reversed(opposing["members"])))
        else:
            opposing = {k:v for k,v in original.items() if k not in {"members", "omitted_count"}}
            opposing.update(kind="narrative_time_relation", relation="different_day",
                left=dict(doc_ref="d1",line_start=2,line_end=2,time="09:00"),
                right=dict(doc_ref="d2",line_start=1,line_end=1,time="当天九点半"))
        raw[0].append(opposing)
    result, _, _ = run_chain(mutate=mutate)
    assert result.diagnostics["semantic_state_comparison"]["unresolved_pairs"] > 0
    assert not [r for r in result.issues if r.category.value == "item_ownership"]


def test_sourced_same_day_scope_applies_to_ongoing_retention_without_inventing_a_date():
    basis = "两处记载明确属于同一天的上午。"
    first_text = "九点，许舟将铜钥匙贴身保管，此后一直没有交给别人。"
    second_text = "十点，沈澈从袋里取出铜钥匙。"
    docs = [DocumentInput("a", "a.md", basis + "\n" + first_text, "chapter", "main"),
            DocumentInput("b", "b.md", second_text, "chapter", "main")]
    first = state_fixture(docs[0].content, record("item",first_text,dict(item="铜钥匙",owner="许舟"),
        time="九点",state="retained_custody",retention="ongoing",line=2),document_id="a")
    second = state_fixture(second_text, record("item",second_text,dict(item="铜钥匙",owner="沈澈"),
        time="十点",state="observed_possession"),document_id="b")
    raw = scope_wire(basis,[member("d1",first_text,"九点","09:00",line=2),
                            member("d2",second_text,"十点","10:00")],relation="same_day")
    scope = bind_time_scope(NarrativeTimeScopeExtraction.model_validate(raw),
        ParsedDirective(kind="narrative_time_scope",attrs={"story_scope":"main","document_role":"chapter"},
            evidence=EvidenceSpan(document_id="a",document_name="a.md",line_start=1,line_end=1,text=basis)),
        {f"d{i+1}":(doc,chunk_document(doc,12000)[0]) for i,doc in enumerate(docs)})
    assert custody_comparisons([first,second])[1] == 1
    issues, unknown = custody_comparisons([first,second,scope])
    assert len(issues) == 1 and unknown == 0
    assert len(issues[0].evidence) == 3  # State, observed possession, temporal source.


@pytest.mark.parametrize("whole_paragraph", [False, True])
def test_unlocated_baseline_action_is_not_erased_by_sourced_proposition(whole_paragraph):
    text = "10:00，许舟用铜钥匙打开库房。"
    if whole_paragraph:
        text += "许舟后来又用铜钥匙打开侧门。"
    raw = record("uses", text, dict(item="铜钥匙",user="许舟"),time="10:00")
    if whole_paragraph:
        # More specific source quotes disambiguate roles, but do not identify the
        # untimed baseline's action when the proposition spans both operations.
        for role in raw["semantic_relation"]["roles"].values():
            role.update(quote="许舟用铜钥匙打开库房", relation="equivalent")
    row = state_fixture(text, raw)
    baseline = row.model_copy(update={"semantic_state":None,"attrs":{**row.attrs,"time":""},"provenance_sources":frozenset({"baseline"})})
    state = state_fixture(text, record("item",text.split("。",1)[0],dict(item="铜钥匙",owner="许舟"),time="10:00",state="observed_possession"))
    result = DeterministicCandidateNormalizer().enrich([], [baseline, row, state])
    assert len(result.directives) == 3
    assert sum(r.kind == "uses" for r in result.directives) == 2
    assert any(r.semantic_state is None and not r.attrs.get("time") for r in result.directives)
    assert any(r.kind == "item" and r.semantic_state for r in result.directives)


@pytest.mark.parametrize("mode", ["shared_prefix", "local_priority", "ambiguous", "absent", "probable"])
def test_member_time_uses_unique_same_evidence_prefix_and_preserves_uncertainty(mode):
    basis = "这两处行动发生在同一天。"
    action = "许岚在南门值守。"
    first = "16:00，江遥抵达北楼，" + action
    if mode == "local_priority":
        action = "16:00，许岚在南门值守。"
        first = "16:00，江遥抵达北楼；" + action
    elif mode == "ambiguous":
        first = "16:00，江遥抵达北楼；16:00，江遥离开，" + action
    elif mode == "absent":
        first = "江遥抵达北楼，" + action
    second = "17:00，许岚来到东门。"
    doc = DocumentInput("prefix", "prefix.md", "\n".join([basis, first, second]), "chapter", "main")
    span = lambda n: EvidenceSpan(document_id=doc.id, document_name=doc.name,
        line_start=n, line_end=n, text=doc.content.splitlines()[n-1])
    raw = scope_wire(basis, [member("d1", action, "16:00", "16:00", line=2),
                             member("d1", second, "17:00", "17:00", line=3)], relation="same_day")
    if mode == "probable":
        raw["certainty"] = "probable"
    typed = NarrativeTimeScopeExtraction.model_validate(raw)
    row = ParsedDirective(kind="narrative_time_scope", attrs={"story_scope":"main", "document_role":"chapter"}, evidence=span(1))
    sources = {"d1": (doc, chunk_document(doc, 12000)[0])}
    if mode in {"ambiguous", "absent"}:
        with pytest.raises(TimeScopeBindingError, match="source_not_unique"):
            bind_time_scope(typed, row, sources)
        return
    scope = bind_time_scope(typed, row, sources)
    assert validate_time_scope_sources(scope, {doc.id:doc.content})
    source_member = scope.narrative_time_scope["members"][0]
    assert source_member["time_ref"]["start"] == (first.rindex("16:00") if mode == "local_priority" else 0)
    left = ParsedDirective(kind="event", attrs={"time":"16:00", "location":"南门", "participants":"许岚", "story_scope":"main"}, evidence=span(2))
    right = ParsedDirective(kind="event", attrs={"time":"17:00", "location":"东门", "participants":"许岚", "story_scope":"main"}, evidence=span(3))
    assert member_matches(source_member, left)
    assert not member_matches(source_member, left.model_copy(update={"attrs":{**left.attrs,"location":"北楼","participants":"江遥"}}))
    order, evidence = compare_time_points(left, right, [scope])
    from app.semantic_quality import assess_directive
    _, reason = assess_directive(scope)
    if mode == "probable":
        assert order is None and reason == "time_scope_unresolved"
        assert scope.attrs["certainty"] == "probable"
    else:
        assert order == -1 and len(evidence) == 1 and reason is None
    if mode == "local_priority":
        proof = deepcopy(scope.narrative_time_scope)
        proof["members"][0]["time_ref"] = dict(start=0,end=5,text_sha256=_digest("16:00"))
        proof["payload_sha256"] = _digest({k:v for k,v in proof.items() if k != "payload_sha256"})
        altered = scope.model_copy(update={"narrative_time_scope":proof})
        assert not validate_time_scope_sources(altered, {doc.id:doc.content})
        assert not member_matches(proof["members"][0], left)
