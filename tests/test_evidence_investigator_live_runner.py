from __future__ import annotations

import json
import shutil
import urllib.error
from collections import Counter
from pathlib import Path

import pytest

from scripts.run_evidence_investigator_live import (
    DATASET_ROOT,
    GroundTruth,
    HttpApiError,
    LiveEvaluationError,
    RunnerOptions,
    UrllibJsonApi,
    _classify,
    _safe_capability_isolation,
    _safe_diagnostics,
    _service_reported_effective_limits,
    build_parser,
    load_case_plans,
    main,
    run_live_evaluation,
    verify_frozen_dataset,
    write_artifact_exclusive,
)


class FixtureHttpApi:
    """HTTP-shaped service double; it never constructs or calls a provider."""

    def __init__(self, plans):
        self.plans = {plan.case_id: plan for plan in plans}
        self.pending_case_ids = iter(self.plans)
        self.projects: dict[str, str] = {}
        self.runs: dict[str, str] = {}
        self.documents: dict[str, dict[str, str]] = {}
        self.calls: list[tuple[str, str, object]] = []

    def request_json(self, method, path, *, payload=None, timeout):
        assert timeout > 0
        self.calls.append((method, path, payload))
        if method == "GET" and path == "/health":
            return {
                "status": "ok",
                "model": {
                    "configured": True,
                    "thinking": {"configured": True, "mode": "disabled"},
                },
            }
        if method == "POST" and path == "/api/v1/projects":
            assert set(payload) == {"name", "description"}
            assert payload["name"].startswith("EIL isolated ")
            assert not any(case_id in payload["name"] for case_id in self.plans)
            case_id = next(self.pending_case_ids)
            project_id = f"project-{case_id}"
            self.projects[project_id] = case_id
            self.documents[project_id] = {}
            return {"id": project_id}
        if method == "POST" and path.endswith("/documents/text"):
            project_id = path.split("/")[4]
            case_id = self.projects[project_id]
            assert set(payload) == {
                "name",
                "content",
                "document_role",
                "story_scope",
            }
            matching = [
                document
                for document in self.plans[case_id].documents
                if document.filename == payload["name"]
            ]
            assert len(matching) == 1
            document = matching[0]
            assert payload == document.upload_payload()
            actual_id = f"doc-{case_id}-{document.logical_id}"
            self.documents[project_id][document.logical_id] = actual_id
            return {
                "id": actual_id,
                "document_role": document.role,
                "story_scope": document.story_scope,
            }
        if method == "POST" and path.endswith("/analysis-runs"):
            project_id = path.split("/")[4]
            case_id = self.projects[project_id]
            run_id = f"run-{case_id}"
            self.runs[run_id] = case_id
            return {"id": run_id, "status": "queued"}
        if method == "GET" and path.startswith("/api/v1/analysis-runs/"):
            parts = path.split("/")
            run_id = parts[4]
            case_id = self.runs[run_id]
            plan = self.plans[case_id]
            if len(parts) == 5:
                return {
                    "id": run_id,
                    "status": "completed",
                    "prompt_tokens": 29,
                    "completion_tokens": 5,
                    "usage_accounting": {"charged_tokens": 91},
                }
            if parts[5] == "issues":
                if plan.ground_truth.decision == "abstain":
                    return []
                evidence = []
                for logical_id, start, end in plan.ground_truth.allowed_evidence:
                    evidence.append(
                        {
                            "document_id": self.documents[
                                f"project-{case_id}"
                            ][logical_id],
                            "document_name": f"{logical_id}.md",
                            "line_start": start,
                            "line_end": end,
                            "text": "CANARY source body must never reach the artifact",
                        }
                    )
                return [
                    {
                        "id": f"issue-{case_id}",
                        "category": plan.ground_truth.issue_category,
                        "evidence": evidence,
                    }
                ]
            if parts[5] == "diagnostics":
                positive = plan.ground_truth.decision == "added_issue"
                provider_calls = 3 if positive else 1
                return {
                    "model": {
                        "enabled": False,
                        "configured": False,
                        "used": False,
                        "logical_call_count": 0,
                        "provider_calls": [],
                    },
                    "evidence_investigator": {
                        "enabled": True,
                        "outcome": "completed",
                        "reason_code": "completed",
                        "seed_count": 1,
                        "loop": {
                            "outcome": "completed",
                            "reason_code": "completed",
                            "provider_calls": provider_calls,
                            "executed_tool_calls": provider_calls,
                            "executed_searches": 1 if positive else 0,
                            "executed_reads": 1 if positive else 0,
                            "recoverable_rejections": 0,
                            "completed_seeds": 1 if positive else 0,
                            "abstained_seeds": 0 if positive else 1,
                            "submitted_envelopes": 1 if positive else 0,
                            "authorized_candidates": 1 if positive else 0,
                        },
                        "promotion": {
                            "submitted_candidates": 1 if positive else 0,
                            "accepted_candidates": 1 if positive else 0,
                            "added_issues": 1 if positive else 0,
                            "rejection_counts": {},
                        },
                        "budget_preflight": {
                            "seed_count": 1,
                            "minimum_path_admissible": True,
                            "minimum_required_rounds": 1,
                            "minimum_initial_reservation": 100,
                            "maximum_local_round_reservation": 200,
                            "maximum_local_run_reservation": 1200,
                            "max_charged_tokens": 16000,
                            "oversized_initial_prompts": 0,
                        },
                        "usage": {
                            "reported_prompt_tokens": 29,
                            "reported_completion_tokens": 5,
                            "charged_tokens": 91,
                            "provider_calls": [
                                {"category": "success"} for _ in range(provider_calls)
                            ]
                        },
                        "rag": {
                            "index": {
                                "outcome": "complete",
                                "reason": None,
                                "profile_fingerprint": "a" * 64,
                                "chunker_fingerprint": "fixture-chunker-v1",
                            },
                            "retrievals": [
                                {
                                    "mode": "hybrid",
                                    "strategy": "keyword+vector+entity-rrf",
                                }
                            ],
                            "total_retrievals": 1,
                        },
                    }
                }
        raise AssertionError(f"unexpected mock route: {method} {path}")


class NoCallApi:
    def __init__(self):
        self.calls = 0

    def request_json(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("HTTP must not be reached")


def options(tmp_path: Path, *, split="dev", confirm=True) -> RunnerOptions:
    return RunnerOptions(
        split=split,
        base_url="http://service.test:8000",
        artifact_path=tmp_path / f"{split}-result.json",
        case_timeout_seconds=30,
        request_timeout_seconds=2,
        poll_interval_seconds=0.05,
        confirm_live_provider=confirm,
    )


def test_mock_http_runner_scores_dev_without_oracle_or_secret_leak(tmp_path):
    plans, _, _ = load_case_plans("dev")
    api = FixtureHttpApi(plans)

    artifact, digest = run_live_evaluation(options(tmp_path), api=api)

    assert len(digest) == 64
    assert artifact["summary"]["case_count"] == 8
    assert artifact["summary"]["failed"] == 0
    assert artifact["summary"]["classification_counts"] == {
        "active_abstain": 3,
        "positive_added_issue": 5,
    }
    assert artifact["summary"]["capability_isolation_verified"] == 8
    assert artifact["safe_configuration"][
        "service_reported_effective_limits"
    ] == {
        "observed_cases": 8,
        "consistent_across_cases": True,
        "values": {
            "max_charged_tokens": 16000,
            "source": "service_budget_preflight",
        },
        "distinct_snapshot_count": 1,
    }
    assert all(row["passed"] for row in artifact["cases"])
    assert all(
        row["capability_isolation"]["verified"] is True
        for row in artifact["cases"]
    )
    assert all("case_wall_latency_ms" in row for row in artifact["cases"])
    assert all("latency_ms" not in row for row in artifact["cases"])
    assert all(
        row["investigator"]["loop"]["tool_call_count_basis"]
        == "server_reported_executed_tools"
        for row in artifact["cases"]
    )
    upload_payloads = [
        payload for method, path, payload in api.calls if path.endswith("/documents/text")
    ]
    assert len(upload_payloads) == 16
    assert all(
        set(payload) == {"name", "content", "document_role", "story_scope"}
        for payload in upload_payloads
    )
    transmitted = json.dumps(upload_payloads, ensure_ascii=False)
    for forbidden in ("expected", "candidate", "lure_candidate", "difficulty"):
        assert f'"{forbidden}"' not in transmitted

    persisted = (tmp_path / "dev-result.json").read_text(encoding="utf-8")
    assert "http://service.test:8000" not in persisted
    assert "CANARY source body" not in persisted
    assert "sk-" not in persisted
    assert all(document.content not in persisted for plan in plans for document in plan.documents)


def _investigator_diagnostics(*, reason="completed", outcome="completed"):
    return {
        "available": True,
        "enabled": True,
        "outcome": outcome,
        "reason_code": reason,
        "seed_count": 1,
        "loop": {
            "outcome": outcome,
            "reason_code": reason,
            "provider_decision_calls": 1,
            "tool_calls": 1,
            "tool_call_count_basis": "server_reported_executed_tools",
            "completed_seeds": 1 if outcome == "completed" else 0,
            "abstained_seeds": 0,
            "submitted_envelopes": 1 if outcome == "completed" else 0,
            "authorized_candidates": 1 if outcome == "completed" else 0,
        },
        "promotion": {
            "submitted_candidates": 1 if outcome == "completed" else None,
            "accepted_candidates": 1 if outcome == "completed" else None,
            "added_issues": 1 if outcome == "completed" else None,
            "rejection_counts": {},
        },
        "usage": {
            "reported_prompt_tokens": 10,
            "reported_completion_tokens": 2,
            "charged_tokens": 20,
            "provider_category_counts": {"success": 1},
        },
    }


def _isolation(*, verified=True):
    return {"verified": verified}


@pytest.mark.parametrize(
    "protocol_reason",
    ["repeated_action", "invalid_tool_arguments", "multiple_tool_calls"],
)
def test_classification_separates_agent_protocol_and_isolation_failures(
    protocol_reason,
):
    ground_truth = GroundTruth(
        decision="added_issue",
        issue_category="fact_conflict",
        added_issue_count=1,
        allowed_evidence=(("canon", 1, 1), ("chapter", 2, 2)),
    )
    protocol = _investigator_diagnostics(
        reason=protocol_reason, outcome="degraded"
    )

    classification, passed = _classify(
        run_status="completed",
        categories=Counter(),
        diagnostics=protocol,
        ground_truth=ground_truth,
        target_evidence_match=False,
        citation_scope_authorized=True,
        capability_isolation=_isolation(),
    )
    assert (classification, passed) == ("agent_protocol_failure", False)

    classification, passed = _classify(
        run_status="completed",
        categories=Counter({"fact_conflict": 1}),
        diagnostics=_investigator_diagnostics(),
        ground_truth=ground_truth,
        target_evidence_match=True,
        citation_scope_authorized=True,
        capability_isolation=_isolation(verified=False),
    )
    assert (classification, passed) == ("isolation_failure", False)


def test_capability_isolation_requires_both_switches_off_and_matching_usage():
    investigator = _investigator_diagnostics()
    status = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "usage_accounting": {"charged_tokens": 20},
    }
    diagnostics = {
        "model": {
            "enabled": False,
            "configured": False,
            "used": False,
            "logical_call_count": 0,
            "provider_calls": [],
        }
    }

    isolated = _safe_capability_isolation(
        diagnostics, status_payload=status, investigator=investigator
    )
    assert isolated["verified"] is True
    assert isolated["issue_evidence_review"] == {
        "diagnostics_present": False,
        "enabled": False,
        "logical_calls": 0,
        "provider_calls": 0,
        "switch_proof": "stage_diagnostics_absent",
    }

    diagnostics["ai_evidence_review"] = {
        "enabled": True,
        "provider_calls": [],
    }
    contaminated = _safe_capability_isolation(
        diagnostics, status_payload=status, investigator=investigator
    )
    assert contaminated["verified"] is False
    assert "issue_evidence_review_not_disabled" in contaminated["reason_codes"]

    diagnostics.pop("ai_evidence_review")
    diagnostics["model"]["provider_calls"] = [{"purpose": "extract"}]
    diagnostics["model"]["logical_call_count"] = 1
    contaminated = _safe_capability_isolation(
        diagnostics, status_payload=status, investigator=investigator
    )
    assert contaminated["verified"] is False
    assert "main_extraction_calls_nonzero_or_unknown" in contaminated["reason_codes"]

    diagnostics["model"]["provider_calls"] = []
    diagnostics["model"]["logical_call_count"] = 0
    mismatched_status = dict(status, prompt_tokens=11)
    contaminated = _safe_capability_isolation(
        diagnostics,
        status_payload=mismatched_status,
        investigator=investigator,
    )
    assert contaminated["verified"] is False
    assert "aggregate_chat_usage_mismatch_or_unknown" in contaminated["reason_codes"]


def test_safe_diagnostics_keeps_only_safe_budget_and_execution_counters():
    sanitized = _safe_diagnostics(
        {
            "evidence_investigator": {
                "enabled": True,
                "outcome": "degraded",
                "reason_code": "repeated_action",
                "seed_count": 1,
                "loop": {
                    "outcome": "degraded",
                    "reason_code": "repeated_action",
                    "provider_calls": 3,
                    "executed_tool_calls": 2,
                    "executed_searches": 1,
                    "executed_reads": 1,
                    "recoverable_rejections": 1,
                    "reported_prompt_tokens": 10,
                    "reported_completion_tokens": 2,
                    "charged_tokens": 20,
                },
                "budget_preflight": {
                    "seed_count": 1,
                    "minimum_path_admissible": True,
                    "minimum_required_rounds": 1,
                    "minimum_initial_reservation": 100,
                    "maximum_local_round_reservation": 200,
                    "maximum_local_run_reservation": 1200,
                    "max_charged_tokens": 16000,
                    "oversized_initial_prompts": 0,
                    "secret": "must-not-survive",
                },
            }
        }
    )

    assert sanitized["loop"]["tool_calls"] == 2
    assert sanitized["loop"]["tool_call_count_basis"] == (
        "server_reported_executed_tools"
    )
    assert sanitized["loop"]["searches"] == 1
    assert sanitized["loop"]["reads"] == 1
    assert sanitized["loop"]["recoverable_rejections"] == 1
    assert sanitized["usage"] == {
        "reported_prompt_tokens": 10,
        "reported_completion_tokens": 2,
        "charged_tokens": 20,
        "token_count_basis": "legacy_loop_compatibility",
        "provider_category_counts": {},
    }
    assert sanitized["effective_limits"] == {
        "max_charged_tokens": 16000,
        "source": "service_budget_preflight",
    }
    assert sanitized["budget_preflight"]["seed_count"] == 1
    assert "secret" not in sanitized["budget_preflight"]


def test_top_level_usage_survives_missing_loop_and_preserves_failure_reason():
    raw_diagnostics = {
        "model": {
            "enabled": False,
            "configured": False,
            "used": False,
            "logical_call_count": 0,
            "provider_calls": [],
        },
        "evidence_investigator": {
            "enabled": True,
            "outcome": "degraded",
            "reason_code": "deadline",
            "seed_count": 1,
            "loop": None,
            "usage": {
                "reported_prompt_tokens": 10,
                "reported_completion_tokens": 2,
                "charged_tokens": 20,
                "provider_calls": [{"category": "read_timeout"}],
            },
        },
    }
    investigator = _safe_diagnostics(raw_diagnostics)
    assert investigator["loop"]["outcome"] is None
    assert investigator["reason_code"] == "deadline"
    assert investigator["usage"] == {
        "reported_prompt_tokens": 10,
        "reported_completion_tokens": 2,
        "charged_tokens": 20,
        "token_count_basis": "investigator_usage_ledger",
        "provider_category_counts": {"read_timeout": 1},
    }

    isolation = _safe_capability_isolation(
        raw_diagnostics,
        status_payload={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "usage_accounting": {"charged_tokens": 20},
        },
        investigator=investigator,
    )
    assert isolation["verified"] is True

    classification, passed = _classify(
        run_status="completed",
        categories=Counter(),
        diagnostics=investigator,
        ground_truth=GroundTruth(
            decision="abstain",
            issue_category="fact_conflict",
            added_issue_count=0,
            allowed_evidence=(),
        ),
        target_evidence_match=False,
        citation_scope_authorized=True,
        capability_isolation=isolation,
    )
    assert (classification, passed) == ("timeout_or_budget", False)


def test_top_level_deadline_wins_after_completed_loop():
    diagnostics = _investigator_diagnostics(
        reason="deadline", outcome="degraded"
    )
    diagnostics["loop"].update(
        {"outcome": "completed", "reason_code": "completed"}
    )

    classification, passed = _classify(
        run_status="completed",
        categories=Counter(),
        diagnostics=diagnostics,
        ground_truth=GroundTruth(
            decision="abstain",
            issue_category="fact_conflict",
            added_issue_count=0,
            allowed_evidence=(),
        ),
        target_evidence_match=False,
        citation_scope_authorized=True,
        capability_isolation=_isolation(),
    )
    assert (classification, passed) == ("timeout_or_budget", False)


def test_common_effective_limit_requires_every_case_and_exact_agreement():
    first = {
        "investigator": {
            "effective_limits": {
                "max_charged_tokens": 16000,
                "source": "service_budget_preflight",
            }
        }
    }
    second = json.loads(json.dumps(first))

    assert _service_reported_effective_limits([first, second]) == {
        "observed_cases": 2,
        "consistent_across_cases": True,
        "values": {
            "max_charged_tokens": 16000,
            "source": "service_budget_preflight",
        },
        "distinct_snapshot_count": 1,
    }

    second["investigator"]["effective_limits"]["max_charged_tokens"] = 8000
    assert _service_reported_effective_limits([first, second]) == {
        "observed_cases": 2,
        "consistent_across_cases": False,
        "values": None,
        "distinct_snapshot_count": 2,
    }

    assert _service_reported_effective_limits([first, {"investigator": None}]) == {
        "observed_cases": 1,
        "consistent_across_cases": False,
        "values": None,
        "distinct_snapshot_count": 1,
    }


def test_confirmation_gate_precedes_fixture_and_http(tmp_path):
    api = NoCallApi()
    with pytest.raises(LiveEvaluationError, match="confirmation_required"):
        run_live_evaluation(options(tmp_path, confirm=False), api=api)
    assert api.calls == 0
    assert not (tmp_path / "dev-result.json").exists()


def test_holdout_tamper_fails_before_http(tmp_path):
    copied = tmp_path / "fixture"
    shutil.copytree(DATASET_ROOT, copied)
    target = copied / "holdout" / "EIL-H-01" / "chapter.md"
    target.write_text(target.read_text(encoding="utf-8") + "\n篡改。\n", encoding="utf-8")
    api = NoCallApi()

    with pytest.raises(LiveEvaluationError, match="fixture_integrity_failed"):
        run_live_evaluation(
            options(tmp_path, split="holdout"), api=api, dataset_root=copied
        )
    assert api.calls == 0


def test_artifact_writer_refuses_overwrite(tmp_path):
    target = tmp_path / "immutable.json"
    write_artifact_exclusive(target, {"first": True})
    with pytest.raises(LiveEvaluationError, match="artifact_exists"):
        write_artifact_exclusive(target, {"first": False})
    assert json.loads(target.read_text(encoding="utf-8")) == {"first": True}


def test_existing_artifact_refuses_before_http(tmp_path):
    target = tmp_path / "dev-result.json"
    target.write_text("preserve me", encoding="utf-8")
    api = NoCallApi()
    with pytest.raises(LiveEvaluationError, match="artifact_exists"):
        run_live_evaluation(options(tmp_path), api=api)
    assert api.calls == 0
    assert target.read_text(encoding="utf-8") == "preserve me"


def test_cli_help_documents_live_confirmation(capsys):
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--confirm-live-provider" in output
    assert "--split" in output
    assert "--base-url" in output


def test_cli_refuses_without_explicit_live_confirmation(tmp_path, capsys):
    target = tmp_path / "must-not-exist.json"
    exit_code = main(["--split", "dev", "--artifact", str(target)])
    assert exit_code == 2
    assert "confirmation_required" in capsys.readouterr().err
    assert not target.exists()


def test_frozen_fixture_is_valid_without_running_a_model():
    manifest, manifest_hash, freeze_hash = verify_frozen_dataset()
    assert manifest["dataset_id"] == "evidence-investigator-live-v1"
    assert len(manifest_hash) == 64
    assert len(freeze_hash) == 64


def test_http_client_rejects_redirect_without_reading_body(monkeypatch):
    client = UrllibJsonApi("http://service.test")

    def redirect(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "http://secret.invalid/path", 302, "Found", {}, None
        )

    monkeypatch.setattr(client._opener, "open", redirect)
    with pytest.raises(HttpApiError) as raised:
        client.request_json("GET", "/health", timeout=1)
    assert raised.value.code == "http_redirect_rejected"
    assert "secret.invalid" not in str(raised.value)
