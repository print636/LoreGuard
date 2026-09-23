from __future__ import annotations

import json

import httpx
import pytest

import scripts.run_character_consistency_live as live_acceptance

from scripts.run_character_consistency_live import (
    FIXTURES,
    _diagnostic_partial_baseline_allowed,
    _matches_selector,
    _model_available_for_session,
    _start_run,
    _validate_oracle_payload,
)


FIXTURE = FIXTURES["ooc-v1"]


def test_ooc_fixture_has_distinct_conflict_growth_and_single_emergency():
    oracle = _validate_oracle_payload(json.loads(
        (FIXTURE / "acceptance-oracle.json").read_text(encoding="utf-8")
    ))
    by_case = {row["case_id"]: row for row in oracle["expected_cases"]}
    assert len(by_case) == len(oracle["confirm_candidates"]) == 3
    assert by_case["nanzhi_unexplained_social_shift"]["final_outcome"] == "conflict"
    assert by_case["yuli_training_bridge"]["final_outcome"] == "no_issue"
    assert by_case["cenzhu_emergency_interrupt"]["final_outcome"] == "needs_confirmation"

    profiles = (FIXTURE / "02-character-profiles.md").read_text(encoding="utf-8").splitlines()
    history = (FIXTURE / "03-published-history-v1.0.md").read_text(encoding="utf-8").splitlines()
    draft = (FIXTURE / "04-draft-event-v1.1.md").read_text(encoding="utf-8").splitlines()
    for selector in oracle["confirm_candidates"]:
        line = profiles[selector["source_line"] - 1]
        assert selector["character_key"] in line
        assert selector["value_contains"] in line
    assert "清晨" in draft[3] and "傍晚" in draft[4]
    assert "连续八周" in history[2]
    assert "第一次厉声打断" in draft[9]
    for case in oracle["expected_cases"]:
        if not case["visible"]:
            continue
        for evidence in case["visible_issue"]["allowed_evidence"]:
            path = FIXTURE / evidence["document_name"]
            lines = path.read_text(encoding="utf-8").splitlines()
            assert 1 <= evidence["line_start"] <= evidence["line_end"] <= len(lines)


def test_ooc_oracle_accepts_only_semantically_paired_trait_polarity():
    oracle = json.loads((FIXTURE / "acceptance-oracle.json").read_text(encoding="utf-8"))
    selector = next(
        row for row in oracle["confirm_candidates"]
        if row["case_id"] == "cenzhu_emergency_interrupt"
    )
    candidate = {
        "character_key": "岑烛",
        "trait_type": "core_personality",
        "origin": "explicit_setting",
        "reviewable": True,
        "value": "几乎不会当众厉声打断队友",
        "stability": "core",
        "evidence": [{
            "document_name": "02-character-profiles.md",
            "line_start": 13,
            "line_end": 13,
        }],
    }
    assert _matches_selector(
        {**candidate, "trait_key": "interruption_restraint", "polarity": "positive"},
        selector,
    )
    assert _matches_selector(
        {**candidate, "trait_key": "interruption_behavior", "polarity": "negative"},
        selector,
    )
    assert not _matches_selector(
        {**candidate, "trait_key": "interruption_restraint", "polarity": "negative"},
        selector,
    )


def test_live_runner_uses_guided_mode_and_account_byok_without_echoing_profile():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/v1/account/model-provider":
            return httpx.Response(200, json={"configured": True, "base_url": "private"})
        return httpx.Response(200, json={"id": "run-1"})

    with httpx.Client(
        base_url="http://example.test", transport=httpx.MockTransport(handler)
    ) as client:
        assert _model_available_for_session(client, {"model": {"configured": False}})
        assert _start_run(client, "project-1", mode="baseline_build") == "run-1"
        assert _start_run(client, "project-1", mode="draft_review") == "run-1"
    assert [request.url.path for request in seen] == [
        "/api/v1/account/model-provider",
        "/api/v1/projects/project-1/analysis-runs",
        "/api/v1/projects/project-1/analysis-runs",
    ]
    assert [json.loads(request.content)["mode"] for request in seen[1:]] == [
        "baseline_build", "draft_review"
    ]
    assert "private" not in repr(seen[1:])


def test_partial_baseline_diagnostic_never_admits_unprocessed_or_failed_run():
    partial = {
        "status": "completed",
        "stage_outcome": "partial",
        "material_coverage": "partial",
        "planned_chunks": 3,
        "processed_chunks": 3,
        "stage_usage": {"attempted_calls": 4},
        "reason_counts": {"key_object_support": 7},
    }
    assert _diagnostic_partial_baseline_allowed(partial)
    assert not _diagnostic_partial_baseline_allowed({**partial, "processed_chunks": 2})
    assert not _diagnostic_partial_baseline_allowed({**partial, "status": "failed"})
    assert not _diagnostic_partial_baseline_allowed({**partial, "reason_counts": {}})
    assert not _diagnostic_partial_baseline_allowed({
        **partial, "reason_counts": {"read_timeout": 1}
    })


def test_live_artifact_is_created_once_and_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(live_acceptance, "ROOT", tmp_path)
    live_acceptance._emit_report({"passed": False}, "probe.json")
    artifact = tmp_path / "artifacts" / "probe.json"
    assert json.loads(artifact.read_text(encoding="utf-8")) == {"passed": False}
    with pytest.raises(FileExistsError):
        live_acceptance._emit_report({"passed": True}, "probe.json")
    assert json.loads(artifact.read_text(encoding="utf-8")) == {"passed": False}
