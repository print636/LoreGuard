"""Offline contract checks for the developer-visible scoped OOC runner."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from scripts import run_scoped_ooc_dev_v1 as runner


def test_preflight_pins_inputs_without_parsing_oracle(monkeypatch):
    original = runner._json_object

    def guarded(data: bytes, stage: str):
        assert stage != "sealed_assessment"
        return original(data, stage)

    monkeypatch.setattr(runner, "_json_object", guarded)
    args = SimpleNamespace(phase="preflight", output_json=None)
    result, exit_code = runner.run(args)
    assert exit_code == 0
    assert result["cases"] == 10
    assert result["oracle_parsed"] is False
    assert result["aggregate_accuracy"] is None


def test_default_port_fails_before_any_http(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("HTTP must not be opened")

    monkeypatch.setattr(runner.httpx, "Client", forbidden)
    args = SimpleNamespace(
        phase="prepare", output_json="scoped-offline-test.json",
        allow_provider_call=True, base_url="http://127.0.0.1:8000",
        expected_eval_db_id=None, expected_eval_instance_id=None,
        expected_eval_db_path=None,
    )
    result, exit_code = runner.run(args)
    assert exit_code == 1
    assert result["failure"]["code"] == "isolation_base_url_invalid"


@pytest.mark.parametrize(
    ("changes", "missing", "expected_failure"),
    [
        (
            {"signal_support_id_v4": False,
             "signal_support_segmenter_version": None,
             "signal_semantic_scope_v5": False,
             "signal_semantic_scope_version": None,
             "signal_scope_review_v1": False,
             "signal_scope_review_schema_version": None,
             "signal_scope_review_prompt_version": None},
            (), "signal_support_id_v4_not_enabled",
        ),
        ({}, ("signal_support_id_v4",), "signal_support_id_v4_not_enabled"),
        (
            {"signal_support_segmenter_version": "unexpected-version"},
            (), "signal_support_segmenter_version_mismatch",
        ),
        (
            {"signal_semantic_scope_v5": False,
             "signal_semantic_scope_version": None,
             "signal_scope_review_v1": False,
             "signal_scope_review_schema_version": None,
             "signal_scope_review_prompt_version": None},
            (), "signal_semantic_scope_v5_not_enabled",
        ),
        (
            {"signal_semantic_scope_version": "unexpected-version"},
            (), "signal_semantic_scope_version_mismatch",
        ),
        (
            {"signal_scope_review_v1": False,
             "signal_scope_review_schema_version": None,
             "signal_scope_review_prompt_version": None},
            (), "signal_scope_review_v1_not_enabled",
        ),
        (
            {"signal_scope_review_schema_version": "unexpected-version"},
            (), "signal_scope_review_schema_version_mismatch",
        ),
        (
            {"signal_scope_review_prompt_version": "character-scope-review-prompt-v2"},
            (), "signal_scope_review_prompt_version_mismatch",
        ),
    ],
)
def test_prepare_requires_exact_support_binding_health_before_project_create(
    monkeypatch, changes, missing, expected_failure,
):
    fixture = runner.verify_fixture()
    limits = {
        "scoped_axis_drift_v1": True,
        "signal_full_line_echo_v2": True,
        "signal_support_id_v4": True,
        "signal_support_segmenter_version": "assertion-index-v1",
        "signal_semantic_scope_v5": True,
        "signal_semantic_scope_version": "semantic-scope-v6",
        "signal_scope_review_v1": True,
        "signal_scope_review_schema_version": "character-scope-review-v2",
        "signal_scope_review_prompt_version": "character-scope-review-prompt-v4",
    }
    limits.update(changes)
    for field in missing:
        limits.pop(field)
    provenance = {"character_consistency_limits": limits}
    health = {"runtime_provenance": provenance, "model": {"configured": True}}
    monkeypatch.setattr(runner.isolation, "_verify_live_isolation",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner.legacy, "_code_state", lambda: {
        "git_head": "a" * 40, "worktree_clean": True,
    })
    monkeypatch.setattr(runner.legacy, "_local_service_artifact_sha256",
                        lambda _root: "b" * 64)
    monkeypatch.setattr(runner.legacy, "_service_preflight_gate",
                        lambda *_args: "c" * 64)
    monkeypatch.setattr(runner.legacy, "_safe_character_runtime_provenance",
                        lambda value: value)

    def request(_client, method, path, route, **_kwargs):
        assert (method, path, route) == ("GET", "/health", "health"), (
            "preflight must fail before creating a project or calling a model"
        )
        return health

    monkeypatch.setattr(runner.legacy, "_request", request)
    with pytest.raises(runner.legacy.SafeFailure) as failure:
        runner._prepare(
            object(), fixture, args=argparse.Namespace(run_timeout_seconds=1),
            expected={},
        )
    assert failure.value.payload == {
        "code": expected_failure, "stage": "runtime_preflight", "details": {},
    }


def test_exact_source_target_quote_and_unique_candidate(monkeypatch):
    fixture = runner.verify_fixture()
    axis = fixture.plan["proposed_axes"][0]
    line = fixture.files[runner.COMMON[1]].decode("utf-8").splitlines()[2]
    quote_text = axis["source"]["source_quote"]
    start = line.index(quote_text)
    row = {
        "id": "candidate-1", "project_id": "project-1", "source_run_id": "run-1",
        "review_state": "pending", "reviewable": True, "source_verified": True,
        "character_key": axis["character_key"], "trait_type": axis["dimension"],
        "comparison_key": axis["comparison_key"], "origin": "explicit_setting",
        "stability": "stable", "polarity": "positive", "revision": 0,
        "evidence": [{"document_id": "doc-profile", "document_version": 1,
                      "content_sha256": runner.PINNED_SHA256[runner.COMMON[1]],
                      "input_id": "frozen-input", "document_name": runner.COMMON[1], "line_start": 3,
                      "line_end": 3, "text": line}],
    }
    monkeypatch.setattr(
        runner.legacy, "_verified_target_span",
        lambda _row: ("verified", ("L3:A1", start, start + len(quote_text))),
    )
    assert runner._candidate_matches(
        row, axis, fixture, project_id="project-1", baseline_run_id="run-1",
        source_document_id="doc-profile",
    )
    assert runner._inventory(
        [row], axis, fixture, project_id="project-1", baseline_run_id="run-1",
        source_document_id="doc-profile",
    )["unique"]
    assert not runner._inventory(
        [row, {**row, "id": "candidate-2"}], axis, fixture,
        project_id="project-1", baseline_run_id="run-1",
        source_document_id="doc-profile",
    )["unique"]
    monkeypatch.setattr(
        runner.legacy, "_verified_target_span",
        lambda _row: ("verified", ("L3:A1", 0, start)),
    )
    assert not runner._candidate_matches(
        row, axis, fixture, project_id="project-1", baseline_run_id="run-1",
        source_document_id="doc-profile",
    )
    assert not runner._candidate_matches(
        {**row, "comparison_key": "other-object"}, axis, fixture,
        project_id="project-1", baseline_run_id="run-1",
        source_document_id="doc-profile",
    )


def test_author_decision_file_requires_explicit_simulation_and_unique_selection():
    fixture = runner.verify_fixture()
    cases = fixture.manifest["cases"]
    checkpoint = {"cases": [
        {"case_id": case["case_id"], "axis_key": case["axis_key"],
         "baseline_admitted": True,
         "candidate_inventory": {"unique": True, "candidate_revisions": {"cand": 0}}}
        for case in cases
    ]}
    choices = [
        {"case_id": case["case_id"], "axis_key": case["axis_key"],
         "candidate_id": "cand", "decision": "confirm", "axis_alignment": "same"}
        for case in cases
    ]
    envelope = {
        "schema_version": runner.DECISIONS_SCHEMA,
        "fixture_manifest_sha256": runner.PINNED_SHA256["manifest.json"],
        "checkpoint_sha256": "checkpoint-hash",
        "reviewer_declaration": runner.REVIEWER_DECLARATION,
        "choices": choices,
    }
    assert len(runner._parse_decisions(
        json.dumps(envelope).encode(), fixture, checkpoint, "checkpoint-hash"
    )) == 10
    envelope["reviewer_declaration"] = "auto-confirmed"
    with pytest.raises(runner.legacy.SafeFailure):
        runner._parse_decisions(
            json.dumps(envelope).encode(), fixture, checkpoint, "checkpoint-hash"
        )
    envelope["reviewer_declaration"] = runner.REVIEWER_DECLARATION
    checkpoint["cases"][0]["candidate_inventory"]["unique"] = False
    with pytest.raises(runner.legacy.SafeFailure):
        runner._parse_decisions(
            json.dumps(envelope).encode(), fixture, checkpoint, "checkpoint-hash"
        )


def test_prepare_uses_ten_independent_projects_and_no_drafts(monkeypatch):
    fixture = runner.verify_fixture()
    projects: list[str] = []
    uploaded: list[tuple[str, str]] = []
    runs: list[str] = []

    monkeypatch.setattr(runner.isolation, "_verify_live_isolation",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_service_preflight",
                        lambda _client: ({"worktree_clean": True}, "digest", {}))
    monkeypatch.setattr(runner, "_check_runtime", lambda *_args: None)
    monkeypatch.setattr(runner.legacy, "_request", lambda _client, method, path, route,
                        **_kwargs: {"id": f"project-{len(projects) + 1}"}
                        if route == "project_create" else None)

    def fake_upload(_client, project_id, _fixture, row, *, draft=False):
        assert not draft
        uploaded.append((project_id, row["file"]))
        return "document-id"

    def fake_start(_client, project_id, *, mode, target_id=None):
        assert mode == "baseline_build" and target_id is None
        projects.append(project_id)
        run_id = f"run-{len(runs) + 1}"
        runs.append(run_id)
        return run_id

    monkeypatch.setattr(runner, "_upload", fake_upload)
    monkeypatch.setattr(runner.legacy, "_start_run", fake_start)
    monkeypatch.setattr(runner.legacy, "_wait_run",
                        lambda _client, run_id, **_kwargs: {"id": run_id})
    monkeypatch.setattr(runner.legacy, "_run_summary",
                        lambda _client, run, **_kwargs: {"run_id": run["id"]})
    monkeypatch.setattr(runner.legacy, "_baseline_admission",
                        lambda _summary: {"admitted": False, "checks": {}})
    monkeypatch.setattr(runner.legacy, "_list_pending", lambda *_args: [])
    monkeypatch.setattr(runner, "_inventory",
                        lambda *_args, **_kwargs: {
                            "match_count": 0, "unique": False,
                            "candidate_revisions": {},
                        })
    monkeypatch.setattr(runner.legacy, "_public_run", lambda _summary: {})
    report = runner._prepare(
        object(), fixture, args=argparse.Namespace(run_timeout_seconds=1),
        expected={},
    )
    assert len(report["cases"]) == 10
    assert len(set(projects)) == 10
    assert len(uploaded) == 30
    assert {name for _, name in uploaded} == set(runner.COMMON)
    assert report["quality_scored"] is False


def test_confirmation_posts_explicit_scope_and_direction_hashes(monkeypatch):
    fixture = runner.verify_fixture()
    case = fixture.manifest["cases"][0]
    axis = runner._axis_for_case(fixture, case)
    checkpoint_case = {
        "project_id": "project-1", "baseline_run_id": "run-1",
        "document_ids": {name: ("doc-profile" if name == runner.COMMON[1]
                                else "doc-" + name) for name in runner.COMMON},
        "candidate_inventory": {
            "match_count": 1, "unique": True,
            "candidate_revisions": {"candidate-1": 0},
        },
    }
    choice = {"candidate_id": "candidate-1", "axis_alignment": "same"}
    pending = {
        "id": "candidate-1", "polarity": "positive", "revision": 0,
        "review_state": "pending", "decisions": [],
    }
    confirmed = {
        **pending, "revision": 1, "review_state": "confirmed",
        "source_run_id": "run-1", "approved_axis_id": "axis-1",
        "approved_axis_version": 1, "axis_alignment": "same",
        "axis_polarity": "positive",
        "axis_positive_proposition_sha256": runner._digest(axis["positive_proposition"]),
        "decisions": [{"id": "decision-1", "decision": "confirm",
                       "expected_revision": 0}],
    }
    details = iter((pending, confirmed))
    monkeypatch.setattr(runner, "_candidate_detail", lambda *_args: next(details))
    monkeypatch.setattr(runner, "_candidate_matches", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner, "_inventory", lambda *_args, **_kwargs:
                        checkpoint_case["candidate_inventory"])
    monkeypatch.setattr(runner.legacy, "_list_pending", lambda *_args: [pending])
    captured = {}

    def fake_request(_client, method, _path, route, **kwargs):
        if route == "baseline_status":
            return {"id": "run-1", "project_id": "project-1", "status": "completed"}
        if route == "axis_create":
            return {
                "id": "axis-1", "project_id": "project-1", "version": 1,
                "trait_type": axis["dimension"],
                "definition_sha256": runner._digest(axis["definition"]),
                "positive_proposition_sha256": runner._digest(axis["positive_proposition"]),
                "comparison_key": axis["comparison_key"],
                "applicability_scope_sha256": runner._digest(axis["applicability_scope"]),
            }
        if route == "author_decision":
            captured.update(kwargs)
            return {"candidate": confirmed, "decision_id": "decision-1",
                    "deduplicated": False}
        raise AssertionError(route)

    monkeypatch.setattr(runner.legacy, "_request", fake_request)
    result = runner._confirm_candidate(object(), fixture, case, checkpoint_case, choice)
    assert result["confirmed"] is True
    assert captured["json"]["scope_applicability_confirmed"] is True
    assert captured["json"]["expected_axis_applicability_scope_sha256"] == runner._digest(
        axis["applicability_scope"]
    )
    assert captured["json"]["axis_alignment"] == "same"
    assert captured["headers"]["Idempotency-Key"].startswith("scoped-ooc-v1-")


def test_deferred_author_choices_never_upload_drafts_or_run_model(monkeypatch):
    fixture = runner.verify_fixture()
    cases = fixture.manifest["cases"]
    checkpoint = {
        "evaluation_isolation": {}, "code_state": {"worktree_clean": True},
        "runtime_provenance_sha256": "digest",
        "cases": [{
            "case_id": case["case_id"], "axis_key": case["axis_key"],
            "project_id": f"project-{index}",
            "baseline_run_id": f"baseline-{index}",
            "baseline_admitted": True, "baseline_checks": {"run_completed": True},
            "candidate_inventory": {"match_count": 0, "unique": False,
                                    "candidate_revisions": {}},
        } for index, case in enumerate(cases)],
    }
    decisions = {case["case_id"]: {"decision": "defer"} for case in cases}
    monkeypatch.setattr(runner.isolation, "_verify_live_isolation",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_service_preflight", lambda _client: (
        {"worktree_clean": True}, "digest", {},
    ))
    monkeypatch.setattr(runner, "_upload", lambda *_args, **_kwargs:
                        pytest.fail("deferred case uploaded a draft"))
    monkeypatch.setattr(runner.legacy, "_start_run", lambda *_args, **_kwargs:
                        pytest.fail("deferred case called a model"))
    report = runner._review(
        object(), fixture, checkpoint, decisions,
        args=argparse.Namespace(run_timeout_seconds=1), expected={},
    )
    assert len(report["cases"]) == 10
    assert all(row["unassessed_reason"] == "author_confirmation_deferred"
               for row in report["cases"])
    assert report["aggregate_accuracy"] is None


def test_partial_draft_and_uncertain_material_are_not_clean_pass():
    fixture = runner.verify_fixture()
    case = fixture.manifest["cases"][-1]
    gold = {
        "kind": "uncertain_material",
        "expected_when_confirmed_and_complete": {
            "formal_issue_count": 0,
            "outcome": "needs_confirmation_or_unverifiable",
            "clean_pass": False,
        },
        "applicability_if_both_current_observations_bound": None,
        "independent_events": "yes",
    }
    observed = {
        "case_id": case["case_id"], "unassessed_reason": "draft_coverage_or_report_incomplete",
    }
    partial = runner._assess_case(gold, case, observed)
    assert partial["assessment_state"] == "unassessed"
    assert partial["formal_count_matches_oracle"] is None
    complete = runner._assess_case(gold, case, {
        **observed, "unassessed_reason": None,
        "expected_axis_polarity": "positive",
        "draft": {"status": "completed", "stage_outcome": "completed",
                  "material_coverage": "complete"},
        "accepted_draft_observation_refs": [], "formal_issues": [],
        "review_clues": {"count": 0}, "provisional_clues": {"count": 0},
        "author_confirmation": {"candidate_id_sha256": runner._digest("candidate"),
                                "axis_polarity": "positive"},
        "case_trace": [],
    })
    assert complete["assessment_state"] == "observed_developer_visible"
    assert complete["clean_pass"] is False
    assert "uncertain_material_not_marked_for_review" in complete["wrong_reason_flags"]
