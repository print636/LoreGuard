from __future__ import annotations

import argparse
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
TRANSFER_FIXTURE = FIXTURES["ooc-transfer-v1"]


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


def test_transfer_fixture_fixes_a_distinct_world_and_three_case_oracle():
    oracle = _validate_oracle_payload(json.loads(
        (TRANSFER_FIXTURE / "acceptance-oracle.json").read_text(encoding="utf-8")
    ), require_frozen_semantic_axis=True)
    cases = {row["case_id"]: row for row in oracle["expected_cases"]}
    assert len(cases) == len(oracle["confirm_candidates"]) == 3
    assert cases["jiqing_unexplained_unilateral_route"]["visible"] is True
    assert cases["wentang_published_training_bridge"]["final_outcome"] == "no_issue"
    assert cases["zhouyao_single_emergency_rebuke"]["matched_observation_count"] == 1

    profiles = (TRANSFER_FIXTURE / "02-character-profiles.md").read_text(
        encoding="utf-8"
    ).splitlines()
    history = (TRANSFER_FIXTURE / "03-published-history-v1.0.md").read_text(
        encoding="utf-8"
    ).splitlines()
    draft = (TRANSFER_FIXTURE / "04-draft-event-v1.1.md").read_text(
        encoding="utf-8"
    ).splitlines()
    for selector in oracle["confirm_candidates"]:
        assert selector["character_key"] in profiles[selector["source_line"] - 1]
    assert "没有警报" in draft[2]
    assert "霁青" in draft[3] and "霁青" in draft[4]
    assert "六周" in history[2]
    assert "第一次厉声呵斥" in draft[9]
    for case in oracle["expected_cases"]:
        if not case["visible"]:
            continue
        for evidence in case["visible_issue"]["allowed_evidence"]:
            lines = (TRANSFER_FIXTURE / evidence["document_name"]).read_text(
                encoding="utf-8"
            ).splitlines()
            assert 1 <= evidence["line_start"] <= evidence["line_end"] <= len(lines)


def test_transfer_oracle_rejects_wrong_axis_direction_value_and_missing_anchors():
    payload = json.loads(
        (TRANSFER_FIXTURE / "acceptance-oracle.json").read_text(encoding="utf-8")
    )
    oracle = _validate_oracle_payload(payload, require_frozen_semantic_axis=True)
    selector = oracle["confirm_candidates"][0]
    first, second = selector["trait_polarity_options"][:2]
    candidate = {
        "character_key": selector["character_key"],
        "trait_type": selector["trait_type"],
        "origin": "explicit_setting",
        "reviewable": True,
        "trait_key": first["trait_key"],
        "polarity": first["polarity"],
        "stability": selector["stability"],
        "value": "先征询当值伙伴意见",
        "evidence": [{
            "document_name": selector["source_document"],
            "line_start": selector["source_line"],
            "line_end": selector["source_line"],
        }],
    }
    assert _matches_selector(candidate, selector)
    assert _matches_selector(
        {**candidate, "trait_key": second["trait_key"], "polarity": second["polarity"]},
        selector,
    )
    assert not _matches_selector({**candidate, "trait_key": "food_preference"}, selector)
    assert not _matches_selector({**candidate, "polarity": "negative"}, selector)
    assert not _matches_selector({**candidate, "value": "喜欢海盐糖"}, selector)

    for missing_field in ("value_contains_any", "trait_polarity_options", "stability"):
        invalid = json.loads(json.dumps(payload, ensure_ascii=False))
        invalid["confirm_candidates"][0].pop(missing_field)
        with pytest.raises(RuntimeError):
            _validate_oracle_payload(invalid, require_frozen_semantic_axis=True)
    invalid = json.loads(json.dumps(payload, ensure_ascii=False))
    invalid.pop("selector_policy")
    with pytest.raises(RuntimeError, match="selector policy is missing"):
        _validate_oracle_payload(invalid, require_frozen_semantic_axis=True)
    for missing_field in (
        "final_outcome", "review_verdict", "visible", "matched_observation_count"
    ):
        invalid = json.loads(json.dumps(payload, ensure_ascii=False))
        case = invalid["expected_cases"][2]
        case.pop(missing_field)
        with pytest.raises(RuntimeError):
            _validate_oracle_payload(invalid, require_frozen_semantic_axis=True)


def test_transfer_loader_cannot_downgrade_frozen_policy(monkeypatch, tmp_path):
    payload = json.loads(
        (TRANSFER_FIXTURE / "acceptance-oracle.json").read_text(encoding="utf-8")
    )
    payload.pop("selector_policy")
    (tmp_path / "acceptance-oracle.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setitem(FIXTURES, "ooc-transfer-v1", tmp_path)
    monkeypatch.setattr(live_acceptance, "DEMO", tmp_path)
    with pytest.raises(RuntimeError, match="selector policy is missing"):
        live_acceptance._load_oracle()


def _source_anchor_candidate(
    selector: dict,
    *,
    candidate_id: str = "11111111-1111-4111-8111-111111111111",
) -> dict:
    return {
        "id": candidate_id,
        "revision": 1,
        "character_key": selector["character_key"],
        "trait_type": selector["trait_type"],
        "origin": "explicit_setting",
        "reviewable": True,
        "trait_key": "unlisted_but_source_anchored_trait",
        "polarity": "negative",
        "stability": "temporary",
        "value": selector.get("value_contains", selector.get("value_contains_any", [""])[0]),
        "evidence": [{
            "document_name": selector["source_document"],
            "line_start": selector["source_line"],
            "line_end": selector["source_line"],
        }],
    }


def _review_with_candidates(selector: dict, candidates: list[dict], *, diagnostic: bool):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/projects/project-1/characters":
            return httpx.Response(200, json={"items": [{"character_key": selector["character_key"]}]})
        if request.url.path.endswith("/profile-candidates"):
            return httpx.Response(200, json={"items": candidates})
        if request.url.path.endswith("/decisions"):
            return httpx.Response(200, json={"status": "confirmed"})
        raise AssertionError(request.url.path)

    with httpx.Client(
        base_url="http://example.test", transport=httpx.MockTransport(handler)
    ) as client:
        return live_acceptance._review_explicit_candidates(
            client, "project-1", [selector],
            diagnostic_source_anchored=diagnostic,
        )


def test_source_anchored_dev_review_is_explicit_and_does_not_relax_strict_review():
    oracle = _validate_oracle_payload(json.loads(
        (TRANSFER_FIXTURE / "acceptance-oracle.json").read_text(encoding="utf-8")
    ), require_frozen_semantic_axis=True)
    selector = oracle["confirm_candidates"][0]
    candidate = _source_anchor_candidate(selector)
    assert not _matches_selector(candidate, selector)
    assert live_acceptance._matches_source_anchored_selector(candidate, selector)
    with pytest.raises(live_acceptance.AcceptanceFailure) as strict_error:
        _review_with_candidates(selector, [candidate], diagnostic=False)
    assert strict_error.value.safe_payload["code"] == "oracle_selector_match_count"

    review = _review_with_candidates(selector, [candidate], diagnostic=True)
    assert review["confirmed"] == 1
    identities = review["diagnostic_relaxed_candidate_identities"]
    assert len(identities) == 1
    assert identities[0]["strict_selector_match"] is False
    assert identities[0]["candidate_id_sha256"] == live_acceptance._sha256_text(
        candidate["id"]
    )
    assert "value" not in identities[0]


@pytest.mark.parametrize("candidates", [[], [
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
]])
def test_source_anchored_dev_review_rejects_missing_or_ambiguous_candidates(candidates):
    oracle = _validate_oracle_payload(json.loads(
        (TRANSFER_FIXTURE / "acceptance-oracle.json").read_text(encoding="utf-8")
    ), require_frozen_semantic_axis=True)
    selector = oracle["confirm_candidates"][0]
    rows = [_source_anchor_candidate(selector, candidate_id=value) for value in candidates]
    with pytest.raises(live_acceptance.AcceptanceFailure) as error:
        _review_with_candidates(selector, rows, diagnostic=True)
    assert error.value.safe_payload["code"] == "diagnostic_source_anchor_match_count"
    assert error.value.safe_payload["details"]["matched_candidates"] == len(rows)


def test_source_anchored_dev_report_forces_strict_gate_false(monkeypatch):
    captured = []
    # run() selects a module-global fixture; restore it for later test modules.
    monkeypatch.setattr(live_acceptance, "DEMO", FIXTURES["demo"])
    monkeypatch.setattr(live_acceptance, "_request", lambda *_args, **_kwargs: {
        "runtime_provenance": {"capabilities": {"character_consistency": True}},
        "model": {"configured": True},
    })
    monkeypatch.setattr(live_acceptance, "_execute_trial", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(live_acceptance, "_evaluate_gates", lambda _trials: {
        "at_least_three_independent_full_workflow_trials": True,
        "other": True,
    })
    monkeypatch.setattr(live_acceptance, "_emit_report", lambda report, _path: captured.append(report))
    args = argparse.Namespace(
        fixture="ooc-transfer-v1", trials=1, base_url="http://example.test",
        run_timeout_seconds=1.0, output_json=None,
        diagnostic_continue_partial_baseline=False,
        diagnostic_review_source_anchored_candidates=True,
    )
    assert live_acceptance.run(args) == 1
    assert captured[0]["passed"] is False
    assert captured[0]["gates"]["strict_candidate_selector_identity"] is False
    assert captured[0]["claims"]["diagnostic_source_anchored_candidate_review"] is True

    for fixture, trials in (("demo", 1), ("ooc-v1", 3)):
        with pytest.raises(live_acceptance.AcceptanceFailure) as error:
            live_acceptance.run(argparse.Namespace(**{**vars(args), "fixture": fixture, "trials": trials}))
        assert error.value.safe_payload["code"] == "diagnostic_source_anchored_scope_invalid"


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
        **partial, "reason_counts": {"source_formal": 2, "source_history": 1}
    })
    assert _diagnostic_partial_baseline_allowed({
        **partial,
        "reason_counts": {"source_formal": 2, "key_object_support": 1},
    })
    assert _diagnostic_partial_baseline_allowed({
        **partial,
        "reason_counts": {
            "source_formal": 2,
            "directional_trait_key": 1,
            "regenerated_from_evidence_mismatch": 2,
        },
    })
    assert not _diagnostic_partial_baseline_allowed({
        **partial, "reason_counts": {"regenerated_from_evidence_mismatch": 2}
    })
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
