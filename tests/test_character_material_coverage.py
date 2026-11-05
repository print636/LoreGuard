"""Unverified claims retain located evidence without becoming observations."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from app.character_material_coverage import (
    DraftMaterialGap, DraftMaterialTracker, MATERIAL_REVIEW_KEY, MaterialGapLedger,
)
from app.character_trait_extraction import CharacterSignalChunk
from app.narrative_context import payload_sha256
from tests.test_character_draft_actor_stage_wiring import _draft_source, _partial_result, _result
from tests.test_habit_claim_recovery import SOURCE, _source, extract, record


def receipt(source, start=1, end=1):
    return dict(events=[], evidence_refs=[{
        "role": "C", "line_start": start, "line_end": end,
        "source_input_id_sha256": hashlib.sha256(source.input_id.encode()).hexdigest(),
    }])


def project(ledger, source, start=1, end=1, reasons=()):
    return ledger.for_case(
        case_id="cdc_abc", candidate_id="candidate", reviewed_events=receipt(source, start, end),
        target_key=("廖辰", "contextual_behavior"), draft_document_ids={source.document.id},
        case_reasons=set(reasons),
    )


@pytest.mark.parametrize("targeted", [False, True])
def test_only_unresolved_sibling_has_a_material_gap_after_exact_claim_recovery(targeted):
    good = record("廖辰把行李牌装入口袋")
    bad = record("廖辰捏碎珍珠")
    result, _ = extract([good, bad], targeted)
    gap, = result.material_gaps
    assert gap.reason == "statement_unverified"
    assert gap.line_start == gap.line_end == 1
    assert gap.text_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert len(result.draft_observations) == 1 and result.diagnostics.outcome == "partial"
    assert "珍珠" not in json.dumps(gap.model_dump(), ensure_ascii=False)
    recovered, _ = extract([good], targeted)
    assert recovered.material_gaps == () and recovered.material_resolved_claims


def tracker(content):
    return DraftMaterialTracker(_source(content), content,
                                CharacterSignalChunk("draft-v3", "draft.md", content, 1, "draft"))


def capture(value, rows, failures):
    validation = SimpleNamespace(
        parsed=True, failures=[SimpleNamespace(record_index=i, reason=r) for i, r in failures],
        draft_trace_attempt=None,
    )
    value.capture(json.dumps({"records": rows}, ensure_ascii=False), validation)


def test_final_complete_package_discards_historical_failures_and_late_timeout_is_unknown():
    value = tracker(SOURCE)
    capture(value, [record("廖辰捏碎珍珠")], [(0, "statement_support")])
    assert value.finish(_result()).material_gaps == ()
    value.pending_response = True
    gaps = value.finish(_partial_result()).material_gaps
    assert len(gaps) == 2 and gaps[-1].reason == "response_unavailable"
    assert gaps[-1].line_start is None


def test_claimed_range_or_misquoted_text_is_not_source_verified():
    value = tracker(SOURCE)
    bad = dict(record("廖辰捏碎珍珠"), evidence="PRIVATE-RAW-CLAIM")
    capture(value, [bad], [(0, "statement_support")])
    gap, = value.finish(_partial_result()).material_gaps
    assert gap.line_start is None and gap.text_sha256 is None and gap.claim_digest is None


def test_cross_line_source_window_is_located_but_does_not_prove_event_independence():
    content = "廖辰清点装置。\n她提起把手。\n晚间廖辰封好柜门。"
    value = tracker(content)
    raw = dict(record("廖辰捏碎珍珠"), source_line_start=1, source_line_end=2,
               evidence="\n".join(content.splitlines()[:2]))
    capture(value, [raw], [(0, "statement_support")])
    result = value.finish(_partial_result())
    source = _draft_source(content)
    ledger = MaterialGapLedger()
    ledger.add(result, source, "primary_extraction")
    overlapping = project(ledger, source, 2, 3)
    external = project(ledger, source, 3, 3)
    assert overlapping["gaps"][0]["relation"] == "overlap"
    assert external["gaps"][0]["relation"] == "elsewhere"
    assert external["gaps"][0]["reason"] == "statement_unverified"
    assert external["payload_sha256"] == payload_sha256({k: v for k, v in external.items() if k != "payload_sha256"})


def test_a_later_accepted_exact_claim_clears_only_that_debt_and_new_failure_remains():
    source = _draft_source(SOURCE)
    value = tracker(SOURCE)
    capture(value, [record("廖辰捏碎珍珠")], [(0, "statement_support")])
    unresolved = value.finish(_partial_result())
    key = unresolved.material_gaps[0].claim_digest
    ledger = MaterialGapLedger()
    ledger.add(unresolved, source, "primary_extraction")
    ledger.add(_result().model_copy(update={"material_resolved_claims": (key,), "material_gaps": ()}),
               source, "targeted_recall")
    assert not project(ledger, source)["gaps"]
    ledger.add(unresolved, source, "targeted_verification")
    assert len(project(ledger, source)["gaps"]) == 1


def test_gap_grouping_preserves_count_and_truncation_is_visible():
    source = _draft_source("\n".join(f"第{i}行。" for i in range(20)))
    ledger = MaterialGapLedger()
    gaps = tuple(DraftMaterialGap(reason="statement_unverified", line_start=i, line_end=i,
                                 text_sha256="1" * 64) for i in range(1, 16))
    ledger.add(_partial_result().model_copy(update={"material_gaps": (*gaps, gaps[0])}),
               source, "primary_extraction")
    result = project(ledger, source)
    assert len(result["gaps"]) == 12
    assert result["gaps"][0]["count"] == 2
    assert result["total_count"] == 16 and result["omitted_count"] == 3


def test_missing_package_ledger_and_non_record_coverage_remain_unknown():
    source = _draft_source(SOURCE)
    ledger = MaterialGapLedger()
    ledger.add(_partial_result(), source, "primary_extraction")
    result = project(ledger, source, reasons=("case_release_unknown",))
    assert all(gap["location"] == gap["relation"] == "unknown" for gap in result["gaps"])
    assert result["gaps"][-1]["phase"] == "case_coverage"


def test_normal_stage_attaches_ledger_to_the_existing_review_receipt():
    from fastapi.testclient import TestClient
    from app.main import app, write_limiter
    from tests.test_character_single_event_stage import _single_action_run, _run_stage
    write_limiter.events.clear()
    with TestClient(app) as client:
        run_id, provider = _single_action_run(client)
        result = _run_stage(run_id, provider, character_explanation_review_v1=True)
    finding, = result.issues
    ledger = finding.metadata[MATERIAL_REVIEW_KEY]
    assert ledger["reviewed_events_sha256"] == payload_sha256(finding.metadata["reviewed_events"])
    assert not ledger["gaps"] and ledger["total_count"] == 0
    assert result.diagnostics["case_trace"][0]["target_coverage"]["status"] == "complete"


def test_partial_regeneration_keeps_old_unresolved_claim_and_exact_acceptance_clears_it():
    value = tracker(SOURCE)
    original = record("廖辰捏碎珍珠")
    replacement = record("廖辰踢开皮箱")
    capture(value, [original], [(0, "statement_support")])
    key = value.gaps[0].claim_digest
    capture(value, [replacement], [(0, "statement_support")])
    assert len(value.finish(_partial_result()).material_gaps) == 2
    validation = SimpleNamespace(
        parsed=True, failures=[SimpleNamespace(record_index=1, reason="statement_support")],
        draft_trace_attempt=SimpleNamespace(events=[
            SimpleNamespace(record_ordinal=1, outcome="accepted"),
            SimpleNamespace(record_ordinal=2, outcome="rejected"),
        ]),
    )
    value.capture(json.dumps({"records": [original, replacement]}, ensure_ascii=False), validation)
    partial = value.finish(_partial_result())
    assert len(partial.material_gaps) == 1
    assert key in partial.material_resolved_claims
    assert partial.material_gaps[0].claim_digest != key


def test_scoped_resolution_does_not_erase_the_same_raw_claim_for_another_target():
    source = _draft_source(SOURCE)
    value = tracker(SOURCE)
    capture(value, [record("廖辰捏碎珍珠")], [(0, "statement_support")])
    unresolved = value.finish(_partial_result())
    key = unresolved.material_gaps[0].claim_digest
    first_target, second_target = ("廖辰", "first"), ("廖辰", "second")
    ledger = MaterialGapLedger(scoped_resolution=True)
    ledger.add(unresolved, source, "primary_extraction")
    ledger.add(unresolved, source, "targeted_recall", first_target)
    # A later targeted copy must not overwrite the original cross-target debt.
    assert len(ledger.entries) == 2
    ledger.add(_result().model_copy(update={"material_resolved_claims": (key,), "material_gaps": ()}),
               source, "targeted_verification", first_target)
    ledger.resolved_scopes.add((source.input_id, "first-candidate", key))
    def final(candidate, target):
        return ledger.for_case(case_id="case", candidate_id=candidate, reviewed_events=receipt(source),
                               target_key=target, draft_document_ids={source.document.id}, case_reasons=set())
    assert not final("first-candidate", first_target)["gaps"]
    other = final("second-candidate", second_target)
    assert len(other["gaps"]) == 1 and other["gaps"][0]["phase"] == "primary_extraction"
