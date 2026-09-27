"""Safety and scoring contract for the frozen Return Season HTTP runner."""

from __future__ import annotations

import argparse
import json
from uuid import uuid4

import pytest

from scripts import run_return_season_live as runner


def _args(port: int, *, simulate_author: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        preflight_only=False,
        base_url=f"http://127.0.0.1:{port}",
        expected_eval_db_id=str(uuid4()),
        expected_eval_instance_id=str(uuid4()),
        expected_eval_db_path="C:/isolated-eval.db",
        allow_provider_call=True,
        simulate_author=simulate_author,
        require_draft_trace=False,
        run_timeout_seconds=10,
        output_json=f"return-season-test-{uuid4().hex}.json",
    )


def test_fixture_preflight_hashes_oracle_without_decoding_it(monkeypatch):
    parsed: list[bytes] = []
    original = runner._json_object

    def spy(data, stage):
        parsed.append(data)
        return original(data, stage)

    monkeypatch.setattr(runner, "_json_object", spy)
    fixture = runner.verify_fixture()
    assert len(parsed) == 2
    assert "oracle.json" not in fixture.files
    assert set(fixture.files) == set(runner.PINNED_SHA256) - {"oracle.json"}


@pytest.mark.parametrize("port", [8000, 8080])
def test_default_user_ports_are_rejected_before_http_or_model(monkeypatch, port):
    class NoClient:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("HTTP must not start on a user-facing default port")

    monkeypatch.setattr(runner.httpx, "Client", NoClient)
    report, code = runner.run(_args(port))
    assert code == 1
    assert report["failure"]["code"] == "isolation_base_url_invalid"
    assert "C:/isolated-eval.db" not in json.dumps(report)


def test_upload_uses_only_story_files_and_preserves_manifest_context(monkeypatch):
    fixture = runner.verify_fixture()
    calls = []

    def record(_client, method, path, route, **kwargs):
        calls.append((method, path, route, kwargs))
        return {"id": str(uuid4())}

    monkeypatch.setattr(runner.legacy, "_request", record)
    for row in fixture.manifest["documents_in_import_order"]:
        runner._upload(object(), "project-id", fixture, row)
    assert len(calls) == 5
    names = [item[3]["json"]["name"] for item in calls]
    assert names == [*runner.BASELINE, runner.DRAFT]
    for (_, _, route, kwargs), row in zip(calls, fixture.manifest["documents_in_import_order"]):
        assert route == "upload"
        assert kwargs["json"]["narrative_context"]["scope"] == row["scope"]
        assert kwargs["json"]["content"] == fixture.files[row["file"]].decode("utf-8")
        assert "oracle" not in kwargs["json"]["name"]
        assert "review-plan" not in kwargs["json"]["name"]


def test_partial_and_zero_observation_drafts_are_unassessed():
    fixture = runner.verify_fixture()
    arm = {"baseline_admitted": True, "approved_axis_bound_count": 0}
    partial = {
        "status": "completed", "stage_outcome": "partial",
        "material_coverage": "partial", "planned_chunks": 7,
        "processed_chunks": 7, "accepted_draft_observation_total": 0,
    }
    complete_zero = {
        **partial, "stage_outcome": "completed", "material_coverage": "complete",
    }
    complete_one = {**complete_zero, "accepted_draft_observation_total": 1}
    for summary, expected_reason in (
        (partial, "draft_coverage_incomplete"),
        (complete_zero, "no_accepted_draft_observation"),
        (complete_one, "approved_axis_unbound"),
    ):
        result = runner._score_after_runs(fixture, [(arm.copy(), summary)])[0]
        assert result["assessed_case_count"] == 0
        assert result["unassessed_case_count"] == 11
        assert {case["reason_code"] for case in result["case_assessment"]} == {expected_reason}
        assert {case["status"] for case in result["case_assessment"]} == {"unassessed"}


def test_source_inventory_is_bounded_and_does_not_repeat_candidate_text():
    fixture = runner.verify_fixture()
    axis = fixture.plan["approved_axes_to_review"][0]
    line = fixture.files["02-character-profiles.md"].decode("utf-8").splitlines()[axis["source"]["line"] - 1]
    candidate = {
        "reviewable": True, "character_key": axis["character_key"],
        "trait_type": "value", "origin": "explicit_setting",
        "polarity": "positive", "stability": "stable",
        "value": runner.TARGET_CLAUSES[axis["axis_key"]] + " SECRET_SENTINEL",
        "evidence": [{
            "document_name": axis["source"]["document_name"],
            "line_start": axis["source"]["line"],
            "line_end": axis["source"]["line"], "text": line,
        }],
        "id": str(uuid4()), "revision": 0,
    }
    inventory, chosen = runner._inventory([candidate], fixture)
    assert inventory[axis["axis_key"]]["strict_core_personality_matches"] == 0
    assert inventory[axis["axis_key"]]["source_clause_value_or_boundary_matches"] == 1
    assert chosen[axis["axis_key"]] is candidate
    assert "SECRET_SENTINEL" not in json.dumps(inventory, ensure_ascii=False)


def test_simulated_confirmation_uses_public_decision_api_without_axis_binding(monkeypatch):
    fixture = runner.verify_fixture()
    chosen = {}
    inventory = {}
    for axis in fixture.plan["approved_axes_to_review"]:
        key = axis["axis_key"]
        chosen[key] = {"id": str(uuid4()), "revision": 0}
        inventory[key] = {"strict_core_personality_matches": 0}
    calls = []

    def record(_client, method, path, route, **kwargs):
        calls.append((method, path, route, kwargs))
        candidate_id = path.split("/")[-2]
        return {"candidate": {
            "id": candidate_id, "review_state": "confirmed",
            "approved_axis_id": None,
        }}

    monkeypatch.setattr(runner.legacy, "_request", record)
    count = runner._confirm_simulated(object(), "project-id", fixture, inventory, chosen)
    assert count == 5
    assert len(calls) == 5
    for method, path, route, kwargs in calls:
        assert method == "POST" and route == "simulated_review"
        assert path.endswith("/decisions")
        assert kwargs["json"]["decision"] == "confirm"
        assert "approved_axis_id" not in kwargs["json"]
        assert "positive_proposition" not in json.dumps(kwargs["json"])


def _draft_trace_stage(*, clean: bool, claimed_line: int = 3,
                       claimed_end_line: int | None = None):
    return {
        "draft_trace_chunks": [{
            "source_document_ordinal": 5,
            "document_chunk_ordinal": 1,
            "stage_chunk_ordinal": 7,
            "phase": "primary_extraction",
            "target_ordinal": None,
            "outcome": "partial" if not clean else "completed",
            "availability": "available",
            "trace": {
                "schema_version": "draft-signal-trace-v1",
                "attempts": [{
                    "attempt": 1, "observability": "parsed",
                    "submitted_record_count": 1, "package_reason": None,
                    "events": [{
                        "record_ordinal": 1,
                        "claimed_line_start_ordinal": claimed_line,
                        "claimed_line_end_ordinal": (
                            claimed_end_line if claimed_end_line is not None else claimed_line
                        ),
                        "outcome": "accepted", "reason": None,
                    }],
                }],
                "final_state": "clean" if clean else "no_clean_package",
                "final_accepted_record_ordinals": [1] if clean else [],
                "source_candidate_linkage": "unavailable",
            },
        }],
        "draft_trace_chunks_omitted_count": 0,
    }


def test_dirty_trace_attempt_acceptance_is_not_final_or_case_scored():
    stage = _draft_trace_stage(clean=False)
    trace, omitted = runner._safe_draft_trace_chunks(stage)
    assert omitted == 0
    event = trace[0]["trace"]["attempts"][0]["events"][0]
    assert event["outcome"] == "accepted"
    assert event["final_accepted"] is False
    fixture = runner.verify_fixture()
    public = {"baseline_admitted": True, "draft_trace_chunks": trace}
    partial = {
        "status": "completed", "stage_outcome": "partial",
        "material_coverage": "partial", "planned_chunks": 7,
        "processed_chunks": 7, "accepted_draft_observation_total": 0,
    }
    scored = runner._score_after_runs(
        fixture, [(public, partial)], max_chunk_chars=8000,
    )[0]
    mapped = scored["draft_trace_chunks"][0]["trace"]["attempts"][0]["events"][0]
    assert mapped["claimed_start_Dxx"] == "D01"
    assert mapped["claimed_single_line_Dxx"] == "D01"
    assert mapped["claimed_start_basis"] == "unverified_model_claim"
    assert mapped["claimed_span_status"] == "single_line_claim"
    assert mapped["final_accepted"] is False
    assert scored["assessed_case_count"] == 0
    assert scored["unassessed_case_count"] == 11


def test_trace_schema_rejects_private_fields_and_invalid_claims():
    clean = _draft_trace_stage(clean=True)
    trace, _ = runner._safe_draft_trace_chunks(clean)
    event = trace[0]["trace"]["attempts"][0]["events"][0]
    assert event["final_accepted"] is True
    malicious = json.loads(json.dumps(clean))
    malicious["draft_trace_chunks"][0]["trace"]["attempts"][0]["events"][0][
        "raw_provider_response"
    ] = "PRIVATE_SENTINEL"
    assert runner._safe_draft_trace_chunks(malicious) is None
    malformed = json.loads(json.dumps(clean))
    malformed["draft_trace_chunks"][0]["phase"] = ["private"]
    assert runner._safe_draft_trace_chunks(malformed) is None
    unavailable = _draft_trace_stage(clean=False)
    unavailable["draft_trace_chunks"][0]["trace"]["attempts"][0].update({
        "observability": "unavailable", "submitted_record_count": None,
        "events": [],
    })
    assert runner._safe_draft_trace_chunks(unavailable) is not None


def test_baseline_trace_projects_only_enums_ordinals_and_rejects_private_fields():
    stage = {
        "baseline_trace_chunks": [{
            "source_document_ordinal": 2,
            "document_chunk_ordinal": 3,
            "stage_chunk_ordinal": 5,
            "source_kind": "published_history",
            "outcome": "completed",
            "availability": "available",
            "trace": {
                "schema_version": "baseline-signal-trace-v1",
                "attempts": [
                    {"attempt": 1, "status": "rejected",
                     "submitted_record_count": 1, "package_reason": None,
                     "rejected_records": [
                         {"record_ordinal": 1, "reason": "statement_support"}
                     ]},
                    {"attempt": 2, "status": "completed",
                     "submitted_record_count": 1, "package_reason": None,
                     "rejected_records": []},
                ],
                "final_state": "clean",
            },
        }],
        "baseline_trace_chunks_omitted_count": 0,
    }
    safe, omitted = runner._safe_baseline_trace_chunks(stage)
    assert omitted == 0
    assert safe[0]["source_document_ordinal"] == 2
    assert [row["attempt"] for row in safe[0]["trace"]["attempts"]] == [1, 2]
    secret = "sk-private-provider-credential"
    for mutation in (
        lambda payload: payload["baseline_trace_chunks"][0].update({"name": secret}),
        lambda payload: payload["baseline_trace_chunks"][0]["trace"].update(
            {"statement": secret}
        ),
        lambda payload: payload["baseline_trace_chunks"][0]["trace"]["attempts"][0]
            ["rejected_records"][0].update({"reason": secret}),
        lambda payload: payload["baseline_trace_chunks"][0].update(
            {"source_kind": [secret]}
        ),
        lambda payload: payload["baseline_trace_chunks"][0]["trace"]["attempts"][0]
            ["rejected_records"][0].update({"record_ordinal": 65}),
    ):
        poisoned = json.loads(json.dumps(stage))
        mutation(poisoned)
        assert runner._safe_baseline_trace_chunks(poisoned) is None
    assert secret not in json.dumps(safe, ensure_ascii=False)


def test_partial_baseline_report_keeps_safe_chunk_trace_before_stopping(monkeypatch):
    fixture = runner.verify_fixture()
    stage = {
        "baseline_trace_chunks": [{
            "source_document_ordinal": 1,
            "document_chunk_ordinal": 2,
            "stage_chunk_ordinal": 4,
            "source_kind": "formal_character_profile",
            "outcome": "degraded",
            "availability": "available",
            "trace": {
                "schema_version": "baseline-signal-trace-v1",
                "attempts": [
                    {"attempt": 1, "status": "rejected",
                     "submitted_record_count": 1, "package_reason": None,
                     "rejected_records": [
                         {"record_ordinal": 1, "reason": "statement_support"}
                     ]},
                    {"attempt": 2, "status": "rejected",
                     "submitted_record_count": 1, "package_reason": None,
                     "rejected_records": [
                         {"record_ordinal": 1, "reason": "statement_support"}
                     ]},
                ],
                "final_state": "no_clean_package",
            },
        }],
        "baseline_trace_chunks_omitted_count": 0,
    }

    def request(_client, method, _path, route, **_kwargs):
        if route == "project_create" and method == "POST":
            return {"id": "project-id"}
        if route == "baseline_trace_diagnostics" and method == "GET":
            return {"character_consistency": stage}
        pytest.fail(f"unexpected request route: {route}")

    monkeypatch.setattr(runner.legacy, "_request", request)
    monkeypatch.setattr(runner, "_upload", lambda *_args: None)
    monkeypatch.setattr(runner.legacy, "_start_run", lambda *_args, **_kwargs: "run-id")
    monkeypatch.setattr(
        runner.legacy, "_wait_run", lambda *_args, **_kwargs: {"id": "run-id"}
    )
    monkeypatch.setattr(
        runner.legacy, "_run_summary",
        lambda *_args, **_kwargs: {"stage_outcome": "partial"},
    )
    monkeypatch.setattr(runner, "_check_runtime", lambda *_args: None)
    monkeypatch.setattr(
        runner.legacy, "_baseline_admission", lambda *_args: {"admitted": False}
    )
    monkeypatch.setattr(runner.legacy, "_public_run", lambda *_args: {})
    public, draft = runner._execute_arm(
        object(), fixture, simulated=False, timeout_seconds=5,
        runtime_digest="a" * 64, draft_trace_enabled=False,
    )
    assert draft is None
    assert public["baseline_admitted"] is False
    assert public["arm_stop_reason"] == "baseline_admission_failed"
    assert public["baseline_trace_status"] == "available"
    assert public["baseline_trace_chunks"][0]["trace"]["attempts"][1][
        "rejected_records"
    ] == [{"record_ordinal": 1, "reason": "statement_support"}]


def test_cross_line_model_claim_does_not_bind_one_dxx_case():
    fixture = runner.verify_fixture()
    trace, _ = runner._safe_draft_trace_chunks(
        _draft_trace_stage(clean=False, claimed_line=3, claimed_end_line=4)
    )
    public = {"baseline_admitted": True, "draft_trace_chunks": trace}
    scored = runner._score_after_runs(
        fixture, [(public, None)], max_chunk_chars=8000,
    )[0]
    event = scored["draft_trace_chunks"][0]["trace"]["attempts"][0]["events"][0]
    assert event["claimed_start_Dxx"] == "D01"
    assert event["claimed_single_line_Dxx"] is None
    assert event["claimed_span_status"] == "multi_line_claim"
    assert scored["unassessed_case_count"] == 11


def test_trace_retains_first_128_chunks_when_later_rows_are_omitted():
    stage = _draft_trace_stage(clean=True)
    stage["draft_trace_chunks"] *= 128
    stage["draft_trace_chunks_omitted_count"] = 3
    projected, omitted = runner._safe_draft_trace_chunks(stage)
    assert len(projected) == 128
    assert omitted == 3


def test_oracle_is_rehashed_at_scoring_after_runs(monkeypatch):
    fixture = runner.verify_fixture()
    original = runner.Path.read_bytes

    def changed(self):
        if self == runner.DATASET / "oracle.json":
            return b"changed after preflight"
        return original(self)

    monkeypatch.setattr(runner.Path, "read_bytes", changed)
    with pytest.raises(runner.legacy.SafeFailure) as error:
        runner._score_after_runs(fixture, [])
    assert error.value.payload["code"] == "oracle_hash_mismatch"


def test_required_trace_rejects_disabled_runtime_before_model_call(monkeypatch):
    calls = []

    def request(_client, method, path, route, **_kwargs):
        calls.append((method, route))
        assert method == "GET" and route == "health"
        return {"model": {"configured": True}, "runtime_provenance": {}}

    limits = {"signal_max_chunk_chars": 8000,
              "signal_draft_trace_v1": False,
              "signal_draft_trace_version": None}
    monkeypatch.setattr(runner.legacy, "_request", request)
    monkeypatch.setattr(runner.legacy, "_code_state", lambda: {"git_head": "a" * 40})
    monkeypatch.setattr(runner.legacy, "_local_service_artifact_sha256", lambda _root: "b" * 64)
    monkeypatch.setattr(runner.legacy, "_service_preflight_gate", lambda *_args: "c" * 64)
    monkeypatch.setattr(runner.legacy, "_safe_character_runtime_provenance",
                        lambda _raw: {"character_consistency_limits": limits})
    with pytest.raises(runner.legacy.SafeFailure) as error:
        runner._runtime_preflight(object(), require_draft_trace=True)
    assert error.value.payload["code"] == "draft_trace_required"
    assert calls == [("GET", "health")]
    limits.update({"signal_draft_trace_v1": True,
                   "signal_draft_trace_version": "draft-signal-trace-v1"})
    assert runner._runtime_preflight(object(), require_draft_trace=True) == (
        "c" * 64, 8000, True,
    )


def test_required_trace_failure_stops_before_creating_evaluation_project(monkeypatch):
    class FakeClient:
        def __init__(self, *_args, **kwargs):
            assert kwargs["trust_env"] is False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    args = _args(8765)
    args.require_draft_trace = True
    monkeypatch.setattr(runner.httpx, "Client", FakeClient)
    monkeypatch.setattr(runner.isolation, "_expected_isolation", lambda *_args: {})
    monkeypatch.setattr(runner.isolation, "_verify_live_isolation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner, "_runtime_preflight",
        lambda _client, **_kwargs: runner._fail("draft_trace_required", "runtime_preflight"),
    )
    monkeypatch.setattr(
        runner, "_execute_arm", lambda *_args, **_kwargs: pytest.fail("model arm started")
    )
    report, code = runner.run(args)
    assert code == 1
    assert report["failure"]["code"] == "draft_trace_required"


def test_oracle_is_loaded_only_after_both_arms_finish(monkeypatch):
    fixture = runner.verify_fixture()
    oracle_bytes = (runner.DATASET / "oracle.json").read_bytes()
    calls: list[str] = []
    original = runner._json_object

    def spy(data, stage):
        if data == oracle_bytes:
            assert calls == ["A", "B"]
            calls.append("oracle")
        return original(data, stage)

    class FakeClient:
        def __init__(self, *_args, **kwargs):
            assert kwargs["trust_env"] is False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def arm(_client, _fixture, *, simulated, **_kwargs):
        calls.append("B" if simulated else "A")
        return ({"baseline_admitted": True, "approved_axis_bound_count": 0}, None)

    monkeypatch.setattr(runner, "_json_object", spy)
    monkeypatch.setattr(runner.httpx, "Client", FakeClient)
    monkeypatch.setattr(runner.isolation, "_expected_isolation", lambda *_args: {})
    monkeypatch.setattr(runner.isolation, "_verify_live_isolation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner, "_runtime_preflight",
        lambda _client, **_kwargs: ("a" * 64, 8000, False),
    )
    monkeypatch.setattr(runner, "_execute_arm", arm)
    report, code = runner.run(_args(8765, simulate_author=True))
    assert code == 0
    assert calls == ["A", "B", "oracle"]
    assert len(report["arms"]) == 2
    assert all(row["unassessed_case_count"] == 11 for row in report["arms"])
