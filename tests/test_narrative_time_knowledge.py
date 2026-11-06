"""Ordinary synthetic source/ordering contracts; fake provider, no model calls."""
import json

import httpx
import pytest

from app.config import NARRATIVE_TIME_KNOWLEDGE_POLICY_V1
from app.domain import ParsedDirective
from app.model_extractor import ModelEnhancedExtractor, BATCH_SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT_NARRATIVE_V1
from app.narrative_relations import compare_time_points, valid_relation
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import OpenAICompatibleProvider, RetryPolicy
from app.rules import detect_issues
from tests.test_model_extractor import settings, completion


def labels(**values):
    return dict(modality="asserted", source_scope="narrator", certainty="certain", **values)


def event(location, *, time="12:00", line=1):
    return labels(kind="event", time=time, location=location, participants=["许宁"],
                  source_line_start=line, source_line_end=line)


def constraint(text, *, time="14:00", relation="unknown_before", line=1, **extra):
    row = labels(kind="knowledge_constraint", character="许宁", fact="轮班口令", time=time,
                 relation=relation, knowledge_scope="whole_content", basis="explicit",
                 relation_quote=text, source_line_start=line, source_line_end=line)
    row.update(extra)
    return row


def link(quote, *, left_time="12:00", right_time="12:00", right_line=2, relation="same_day"):
    return labels(kind="narrative_time_relation", relation=relation, basis="explicit", relation_quote=quote,
                  left=dict(doc_ref="d1", line_start=1, line_end=1, time=left_time),
                  right=dict(doc_ref="d2", line_start=right_line, line_end=right_line, time=right_time),
                  source_line_start=1, source_line_end=1)


def run_pair(first, second, first_records, second_records, *, enabled=True, right_scope="main"):
    calls = []
    payload = {"documents": [{"doc_ref": "d1", "records": first_records},
                              {"doc_ref": "d2", "records": second_records}]}

    def handler(request):
        calls.append(json.loads(request.content))
        return completion(json.dumps(payload, ensure_ascii=False))

    config = settings(provider_max_attempts=1, enable_review_agent=False)
    config._narrative_time_knowledge_policy = NARRATIVE_TIME_KNOWLEDGE_POLICY_V1 if enabled else None
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0), sleep=lambda _: None)
    docs = [DocumentInput("source-a", "a.md", first, "chapter", "main"),
            DocumentInput("source-b", "b.md", second, "chapter", right_scope)]
    result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider), settings=config).run(docs)
    return result, calls


def location_fixture(relation="same_day", **options):
    quote = "两处记录属于同一天。" if relation == "same_day" else "两处记录分属不同的日子。"
    first = "12:00，许宁在北楼值守。"
    second = quote + "\n12:00，许宁在南门值守。"
    return run_pair(first, second, [event("北楼")], [event("南门", line=2), link(quote, relation=relation)], **options)


def test_same_day_clock_points_reach_existing_location_category_with_all_sources():
    result, calls = location_fixture()
    issues = [row for row in result.issues if row.category.value == "location_collision"]
    assert len(issues) == 1
    assert len(calls) == 1 and calls[0]["messages"][0]["content"] == BATCH_SYSTEM_PROMPT_NARRATIVE_V1
    assert len(issues[0].evidence) == 3
    relation = next(row for row in result.directives if row.kind == "narrative_time_relation")
    assert valid_relation(relation)
    assert "doc_ref" not in json.dumps(relation.model_dump(mode="json"))
    assert "narrative_relation" not in relation.model_dump(mode="json")
    assert all(row.attrs.get("time", "12:00") == "12:00" for row in result.directives if row.kind == "event")


@pytest.mark.parametrize("mode", ["no_relation", "different_day", "unclear", "missing_source", "cross_scope", "old_policy"])
def test_same_clock_is_not_itself_a_same_day_certificate(mode):
    quote = "两处记录属于同一天。"
    relation = link(quote, relation="different_day" if mode == "different_day" else "same_day")
    if mode == "unclear":
        relation.update(relation="unclear", basis="unclear", certainty="unknown")
    if mode == "missing_source":
        relation["left"]["doc_ref"] = "d9"
    result, calls = run_pair("12:00，许宁在北楼值守。", quote + "\n12:00，许宁在南门值守。",
        [event("北楼")], [event("南门", line=2)] + ([] if mode == "no_relation" else [relation]),
        enabled=mode != "old_policy", right_scope="other" if mode == "cross_scope" else "main")
    assert not [row for row in result.issues if row.category.value == "location_collision"]
    assert len(calls) == 1
    if mode == "unclear":
        assert result.diagnostics["model"]["unresolved_invalid_records"] >= 1
    if mode == "old_policy":
        assert calls[0]["messages"][0]["content"] == BATCH_SYSTEM_PROMPT
        assert not any(row.kind == "narrative_time_relation" for row in result.directives)


@pytest.mark.parametrize(("relation", "positive_time", "expected"), [
    ("unknown_before", "11:30", True), ("unknown_before", "14:00", False),
    ("unknown_through", "14:00", True), ("unknown_through", "14:01", False),
])
def test_knowledge_lower_bound_is_not_an_acquisition_and_before_through_are_distinct(relation, positive_time, expected):
    first = "14:00之前，许宁明确不知道轮班口令。" if relation == "unknown_before" else "截至14:00，许宁仍不知道轮班口令。"
    quote = "两段发生在同一天。"
    positive = f"{positive_time}，许宁准确复述了轮班口令，并凭它通过核验。"
    result, calls = run_pair(first, quote + "\n" + positive, [constraint(first, relation=relation)],
        [constraint(positive, time=positive_time, relation="known_by", line=2),
         link(quote, left_time="14:00", right_time=positive_time)])
    issues = [row for row in result.issues if row.category.value == "knowledge_without_acquisition"]
    assert bool(issues) is expected
    assert not any(row.kind == "knows" for row in result.directives)
    assert len(calls) == 1
    if expected:
        assert len(issues[0].evidence) == 3
        assert "不代表" in issues[0].explanation


@pytest.mark.parametrize("scope", ["channel_only", "unclear"])
def test_one_unavailable_channel_does_not_prove_global_ignorance(scope):
    first = "14:00之前，许宁未从档案官处听说轮班口令。"
    quote = "两段发生在同一天。"
    positive = "11:30，许宁准确说出了轮班口令。"
    result, _ = run_pair(first, quote + "\n" + positive,
        [constraint(first, knowledge_scope=scope)],
        [constraint(positive, time="11:30", relation="known_by", line=2),
         link(quote, left_time="14:00", right_time="11:30")])
    assert not [row for row in result.issues if row.category.value == "knowledge_without_acquisition"]


def test_absolute_times_need_no_invented_day_or_acquisition():
    first = "2026-05-12 14:00之前，许宁明确不知道轮班口令。"
    second = "2026-05-12 11:30，许宁准确说出了轮班口令。"
    result, _ = run_pair(first, second,
        [constraint(first, time="2026-05-12 14:00")],
        [constraint(second, time="2026-05-12 11:30", relation="known_by")])
    assert len([row for row in result.issues if row.category.value == "knowledge_without_acquisition"]) == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_later_acquisition_is_not_an_implicit_first_acquisition_under_new_policy(enabled):
    acquisition = labels(kind="knows", character="许宁", fact="轮班口令",
        time="2026-05-12 14:00", source_line_start=1, source_line_end=1)
    claim = labels(kind="claims_knows", character="许宁", fact="轮班口令",
        time="2026-05-12 11:30", source_line_start=1, source_line_end=1)
    result, calls = run_pair("2026-05-12 14:00，许宁阅读来信后获知轮班口令。",
        "2026-05-12 11:30，许宁准确说出了轮班口令。", [acquisition], [claim], enabled=enabled)
    assert {row.kind for row in result.directives} >= {"knows", "claims_knows"}
    assert bool([row for row in result.issues if row.category.value == "knowledge_without_acquisition"]) is not enabled
    assert len(calls) == 1


def test_legacy_or_tampered_rows_cannot_borrow_relation_authority():
    result, _ = location_fixture()
    raw = [ParsedDirective.model_validate(row.model_dump(mode="json")) for row in result.directives]
    assert not detect_issues(raw)
    relation = next(row for row in result.directives if row.kind == "narrative_time_relation")
    changed = relation.model_copy(update={"attrs": {**relation.attrs, "relation": "before"}})
    assert not valid_relation(changed)


def test_conflicting_direct_relations_remain_unassessed_without_transitive_guessing():
    quote = "两处记录属于同一天。"
    opposite_quote = "两处记录分属不同的日子。"
    opposite = link(opposite_quote, relation="different_day")
    opposite.update(source_line_start=3, source_line_end=3)
    result, _ = run_pair("12:00，许宁在北楼值守。", quote + "\n12:00，许宁在南门值守。\n" + opposite_quote,
        [event("北楼")], [event("南门", line=2), link(quote), opposite])
    assert len([row for row in result.directives if row.kind == "narrative_time_relation"]) == 2
    assert not [row for row in result.issues if row.category.value == "location_collision"]


@pytest.mark.parametrize("prefix", ["假如", "据未经核实的报告，"])
def test_surface_noncanonical_source_cannot_be_laundered_by_optimistic_model_labels(prefix):
    first = prefix + "14:00之前，许宁明确不知道轮班口令。"
    positive = "11:30，许宁准确说出了轮班口令。"
    quote = "两段发生在同一天。"
    result, _ = run_pair(first, quote + "\n" + positive, [constraint(first)],
        [constraint(positive, time="11:30", relation="known_by", line=2),
         link(quote, left_time="14:00", right_time="11:30")])
    assert not [row for row in result.issues if row.category.value == "knowledge_without_acquisition"]
