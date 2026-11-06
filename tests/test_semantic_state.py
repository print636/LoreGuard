"""Public synthetic contracts; all provider responses are controlled, not model quality evidence."""
from copy import deepcopy
import json

import httpx
import pytest

from app.config import SEMANTIC_STATE_POLICY_V1
from app.domain import ParsedDirective
from app.model_extractor import ModelEnhancedExtractor, RECORD_ADAPTER
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import OpenAICompatibleProvider, RetryPolicy
from app.rules import detect_issues
from app.semantic_quality import assess_directive, eligible_for_deterministic_rules
from app.semantic_state import (SEMANTIC_STATE_INSTRUCTIONS, bind_semantic_state,
                                valid_semantic_state, custody_comparisons)
from app.chunking import chunk_document
from tests.test_model_extractor import settings, completion


def record(kind, quote, fields, *, time=None, state=None, retention=None, through=None, sender=None, line=1):
    roles = {key: {"quote": value, "relation": "same"} for key, value in fields.items()}
    raw = dict(kind=kind, **fields, modality="asserted", source_scope="narrator", certainty="certain",
               source_line_start=line, source_line_end=line,
               semantic_relation=dict(quote=quote, roles=roles, relation="supported", actuality="actual",
                   basis="supported_inference", state_kind=state, retention_scope=retention,
                   through_time=through, transfer_from={"quote": sender, "relation": "same"} if sender else None))
    if time:
        raw["time"] = time
    return raw


def state_fixture(text, raw, *, document_id="d", role="chapter", scope="main"):
    document = DocumentInput(document_id, document_id + ".md", text, role, scope)
    chunk = chunk_document(document, 12000)[0]
    typed = RECORD_ADAPTER.validate_python(raw)
    row = ModelEnhancedExtractor._to_directive(document, chunk, typed, require_lexical_support=False)
    bound = bind_semantic_state(typed, row)
    return assess_directive(bound)[0]


def run_source(text, rows, *, enabled=True):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return completion(json.dumps({"records": rows}, ensure_ascii=False))
    config = settings(provider_max_attempts=1, enable_review_agent=False, model_batch_enabled=False)
    config._semantic_state_policy = SEMANTIC_STATE_POLICY_V1 if enabled else None
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0), sleep=lambda _: None)
    result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider), settings=config).run(
        [DocumentInput("source", "source.md", text, "chapter", "main")])
    return result, calls


def fact_fixture():
    text = "北窗敞着。"
    raw = record("fact", text, dict(subject="北窗", predicate="状态", value="开启"))
    raw["semantic_relation"]["roles"].update(predicate=dict(quote="敞着", relation="equivalent"),
                                          value=dict(quote="敞着", relation="equivalent"))
    return text, raw


def test_nonliteral_fact_and_physical_possession_reach_rules_in_existing_single_request():
    text, fact = fact_fixture()
    text += "\n2026-05-01 10:00，许舟从口袋抽出铜钥匙。"
    possession = record("item", text.splitlines()[1], dict(item="铜钥匙", owner="许舟"),
                        time="2026-05-01 10:00", state="observed_possession", line=2)
    result, calls = run_source(text, [fact, possession])
    proved = [r for r in result.directives if r.semantic_state]
    assert {r.kind for r in proved} == {"fact", "item"}
    assert all(valid_semantic_state(r) and eligible_for_deterministic_rules(r) for r in proved)
    assert len(calls) == 1 and calls[0]["messages"][0]["content"].endswith(SEMANTIC_STATE_INSTRUCTIONS)
    assert all("semantic_state" not in r.model_dump() for r in proved)


@pytest.mark.parametrize("mode", ["different_subject", "different_value", "unclear", "denied", "hypothetical", "reported", "bad_quote", "old_run"])
def test_uncertain_wrong_role_and_nonactual_propositions_never_become_canonical(mode):
    text, raw = fact_fixture()
    relation = raw["semantic_relation"]
    if mode.startswith("different_"):
        relation["roles"][mode.removeprefix("different_")]["relation"] = "different"
    elif mode == "unclear":
        relation["relation"] = "unclear"
    elif mode == "denied":
        relation["actuality"] = "negated"
    elif mode == "hypothetical":
        text = "如果北窗敞着，雨水就会飘进来。"
    elif mode == "reported":
        raw.update(source_scope="character_dialogue", modality="reported")
    elif mode == "bad_quote":
        relation["quote"] = "南窗开着。"
    result, calls = run_source(text, [raw], enabled=mode != "old_run")
    assert not any(r.semantic_state and eligible_for_deterministic_rules(r) for r in result.directives)
    assert len(calls) == 1
    if mode == "old_run":
        assert SEMANTIC_STATE_INSTRUCTIONS not in calls[0]["messages"][0]["content"]


def retained_and_observed(*, end=None, time="2026-05-01 10:00", scope="main"):
    first = "2026-05-01 09:00，许舟收好铜钥匙，此后一直贴身保管，没有交给别人。"
    if end:
        first += f"这项保管安排持续至{end}。"
    state = record("item", first, dict(item="铜钥匙", owner="许舟"), time="2026-05-01 09:00",
                   state="retained_custody", retention="bounded" if end else "ongoing", through=end)
    second = f"{time}，沈澈从袋里摸出铜钥匙。"
    observed = record("item", second, dict(item="铜钥匙", owner="沈澈"), time=time, state="observed_possession")
    return [state_fixture(first, state, document_id="past"),
            state_fixture(second, observed, document_id="now", scope=scope)]


def test_new_observed_holder_does_not_erase_explicit_ongoing_retention():
    rows = retained_and_observed()
    assert all(valid_semantic_state(r) for r in rows)
    issues, unresolved = custody_comparisons(rows)
    assert len(issues) == 1 and unresolved == 0
    assert "法律所有权" in issues[0].suggestion
    assert {e.document_id for e in issues[0].evidence} == {"past", "now"}


@pytest.mark.parametrize("mode", ["after_end", "unknown_date", "cross_scope", "no_retention"])
def test_absence_of_handover_record_or_unknown_order_is_not_a_conflict(mode):
    rows = retained_and_observed(end="2026-05-01 09:30" if mode == "after_end" else None,
        time="10:00" if mode == "unknown_date" else "2026-05-01 10:00",
        scope="elsewhere" if mode == "cross_scope" else "main")
    if mode == "no_retention":
        rows[0] = rows[0].model_copy(update={"semantic_state": None,
            "attrs": {k:v for k,v in rows[0].attrs.items() if k not in {"state_kind", "retention_scope"}}})
    issues, unresolved = custody_comparisons(rows)
    assert not issues
    if mode in {"unknown_date", "no_retention"}:
        assert unresolved > 0


def test_single_sourced_transfer_updates_persistent_author_contract_but_observation_does_not():
    from app.parser import parse_document
    baseline = parse_document("canon", "canon.md", "@item item=铜钥匙 owner=许舟 | 铜钥匙由许舟保管").directives[0]
    text = "2026-05-01 09:30，许舟把铜钥匙交到沈澈手中。"
    transfer = state_fixture(text, record("item", text, dict(item="铜钥匙", owner="沈澈"),
        time="2026-05-01 09:30", state="transfer", sender="许舟"), document_id="transfer", scope="global")
    text = "2026-05-01 10:00，沈澈以铜钥匙拧开箱盖。"
    use = state_fixture(text, record("uses", text, dict(item="铜钥匙", user="沈澈"),
        time="2026-05-01 10:00"), document_id="use", scope="global")
    assert eligible_for_deterministic_rules(use)  # not tied to an operation verb list
    assert not custody_comparisons([baseline, transfer, use])[0]
    assert len(custody_comparisons([baseline, use])[0]) == 1
    ongoing = retained_and_observed()[0]
    # A later actual transfer ends an ongoing state, but cannot rewrite an
    # explicit bounded no-transfer assertion covering that same time.
    assert not custody_comparisons([ongoing, transfer, use])[0]
    bounded = retained_and_observed(end="2026-05-01 11:00")[0]
    assert custody_comparisons([bounded, transfer, use])[0]
    # Legacy invocation keeps its original rule path.
    assert len(detect_issues([baseline, use])) == 1


def test_proof_cannot_be_borrowed_after_source_identity_or_field_mutation():
    text, raw = fact_fixture()
    row = state_fixture(text, raw)
    for changed in (
        row.model_copy(update={"attrs": {**row.attrs, "subject": "南窗"}}),
        row.model_copy(update={"evidence": row.evidence.model_copy(update={"document_id": "other"})}),
        row.model_copy(update={"semantic_state": {**row.semantic_state, "raw_reply": "extra"}}),
    ):
        assert not valid_semantic_state(changed)
        assert not eligible_for_deterministic_rules(changed)


@pytest.mark.parametrize("mode", ["ordered_handover", "missing_handover", "unknown_order"])
def test_ordinary_custody_consumes_only_a_sourced_ordered_handover(mode):
    text = "2026-05-01 09:00，铜钥匙由许舟保管。"
    custody = state_fixture(text, record("item", text, dict(item="铜钥匙", owner="许舟"),
        time="2026-05-01 09:00", state="custody"), document_id="custody")
    transfer_time = "09:30" if mode == "unknown_order" else "2026-05-01 09:30"
    text = f"{transfer_time}，许舟把铜钥匙交到沈澈手里。"
    transfer = state_fixture(text, record("item", text, dict(item="铜钥匙", owner="沈澈"),
        time=transfer_time, state="transfer", sender="许舟"), document_id="transfer")
    text = "2026-05-01 10:00，沈澈用铜钥匙打开库房。"
    use = state_fixture(text, record("uses", text, dict(item="铜钥匙", user="沈澈"),
        time="2026-05-01 10:00"), document_id="use")
    rows = [custody, use] if mode == "missing_handover" else [custody, transfer, use]
    assert all(valid_semantic_state(row) and eligible_for_deterministic_rules(row) for row in rows)
    issues, unresolved = custody_comparisons(rows)
    assert not issues
    assert unresolved == 0 if mode == "ordered_handover" else unresolved > 0
