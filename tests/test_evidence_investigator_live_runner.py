from __future__ import annotations

import json
import shutil
import urllib.error
from collections import Counter
from pathlib import Path

import pytest
import scripts.run_evidence_investigator_live as live_runner

from scripts.run_evidence_investigator_live import (
    CasePlan,
    DATASET_ROOT,
    GroundTruth,
    HttpApiError,
    LiveEvaluationError,
    RunnerOptions,
    SourceDocument,
    UrllibJsonApi,
    _build_outcome_summary,
    _classify,
    _safe_capability_isolation,
    _safe_diagnostics,
    _safe_runtime_provenance,
    _service_reported_effective_limits,
    _target_evidence_match,
    build_parser,
    compare_dev_artifact_payloads,
    load_case_plans,
    main,
    run_live_evaluation,
    verify_frozen_dataset,
    write_artifact_exclusive,
)


FIXTURE_GIT_REVISION = "b" * 40
FIXTURE_SERVICE_ARTIFACT = "c" * 64
FIXTURE_PROFILE = "a" * 64
FIXTURE_CHUNKER = "d" * 64


def runtime_provenance():
    return {
        "schema_version": "loreguard-runtime-provenance-v1",
        "build": {
            "git_revision": FIXTURE_GIT_REVISION,
            "service_artifact_sha256": FIXTURE_SERVICE_ARTIFACT,
        },
        "chat_provider": {
            "model_alias": "fixture-model",
            "relay_hostname": "relay.example",
            "relay_configuration_sha256": "e" * 64,
            "temperature": 0,
            "thinking_configured": True,
            "thinking_mode": "disabled",
        },
        "capabilities": {
            "model_extraction": False,
            "issue_evidence_review": False,
            "record_repair_agent": False,
            "evidence_investigator": True,
            "embeddings": True,
        },
        "investigator_limits": {
            "max_seeds": 1,
            "max_decision_rounds": 6,
            "max_tool_calls": 6,
            "max_searches": 2,
            "max_reads": 2,
            "max_results": 12,
            "max_read_lines": 12,
            "max_span_chars": 12000,
            "token_budget": 16000,
            "max_agent_input_bytes": 131072,
            "provider_call_timeout_seconds": 30.0,
            "total_deadline_seconds": 60.0,
            "max_completion_tokens": 768,
            "max_response_bytes": 64000,
            "provider_attempts_per_decision": 1,
            "top_k": 6,
            "branch_limit": 30,
            "embedding_max_input_chars": 250000,
            "require_hybrid": True,
            "daily_token_budget": 100000,
        },
        "rag": {
            "strategy": "keyword+vector+entity-rrf",
            "profile_fingerprint": FIXTURE_PROFILE,
            "chunker_fingerprint": FIXTURE_CHUNKER,
            "require_hybrid": True,
            "top_k": 6,
            "branch_limit": 30,
        },
    }


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
                "runtime_provenance": runtime_provenance(),
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
                    "runtime_provenance": runtime_provenance(),
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
                                "profile_fingerprint": FIXTURE_PROFILE,
                                "chunker_fingerprint": FIXTURE_CHUNKER,
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


def test_mock_http_runner_scores_dev_without_oracle_or_secret_leak(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        live_runner,
        "_git_state",
        lambda: {
            "commit": FIXTURE_GIT_REVISION,
            "tracked_worktree_dirty": False,
        },
    )
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
    assert artifact["schema_version"] == "evidence-investigator-live-http-v3"
    assert artifact["provenance_gate"]["passed"] is True
    assert artifact["development_gate"]["passed"] is True
    assert artifact["reproducibility_fingerprint"] == artifact[
        "safe_configuration_fingerprint"
    ]
    assert artifact["summary"]["capability_isolation_verified"] == 8
    expected_outcome_metrics = {
        "final_output_observed_cases": 8,
        "system_safe_cases": 8,
        "system_safety_rate": 1.0,
        "erroneous_added_issue_count": 0,
        "capability_isolation_verified": 8,
        "capability_isolation_rate": 1.0,
        "positive_cases": 5,
        "positive_passed": 5,
        "positive_recall": 1.0,
        "negative_cases": 3,
        "negative_safe_cases": 3,
        "negative_safety_rate": 1.0,
        "negative_active_abstain_cases": 3,
        "agent_active_abstain_rate": 1.0,
        "promotion_accounting_observed_cases": 8,
        "promotion_submitted_candidates": 5,
        "promotion_accepted_candidates": 5,
        "promotion_acceptance_rate": 1.0,
        "normally_terminated_cases": 8,
        "normal_termination_rate": 1.0,
    }
    assert {
        key: artifact["summary"][key] for key in expected_outcome_metrics
    } == expected_outcome_metrics
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
    assert "https://relay.example" not in persisted
    assert all(document.content not in persisted for plan in plans for document in plan.documents)


def test_outcome_summary_separates_safety_agent_judgment_and_liveness():
    def result(
        *,
        expected_decision,
        classification,
        passed,
        counts,
        evidence_match=False,
        authorized=True,
        submitted=0,
        accepted=0,
        isolated=True,
        run_status="completed",
    ):
        return {
            "expected_decision": expected_decision,
            "expected_issue_category": "fact_conflict",
            "classification": classification,
            "passed": passed,
            "run_status": run_status,
            "issue_category_counts": counts,
            "citation_scope_authorized": authorized,
            "target_evidence_match": evidence_match,
            "investigator": {
                "promotion": {
                    "submitted_candidates": submitted,
                    "accepted_candidates": accepted,
                }
            },
            "capability_isolation": (
                {"verified": isolated} if isolated is not None else None
            ),
        }

    results = [
        result(
            expected_decision="added_issue",
            classification="positive_added_issue",
            passed=True,
            counts={"fact_conflict": 1},
            evidence_match=True,
            submitted=1,
            accepted=1,
        ),
        result(
            expected_decision="added_issue",
            classification="wrong_candidate",
            passed=False,
            counts={"world_rule_conflict": 2},
            evidence_match=False,
            submitted=2,
            accepted=1,
        ),
        result(
            expected_decision="abstain",
            classification="active_abstain",
            passed=True,
            counts={},
        ),
        result(
            expected_decision="abstain",
            classification="promotion_rejection",
            passed=True,
            counts={},
            submitted=1,
            accepted=0,
        ),
        result(
            expected_decision="abstain",
            classification="timeout_or_budget",
            passed=False,
            counts={},
            authorized=None,
            isolated=None,
            run_status=None,
        ),
    ]

    assert _build_outcome_summary(results) == {
        "final_output_observed_cases": 4,
        "system_safe_cases": 3,
        "system_safety_rate": 0.6,
        "erroneous_added_issue_count": 2,
        "capability_isolation_verified": 4,
        "capability_isolation_rate": 0.8,
        "positive_cases": 2,
        "positive_passed": 1,
        "positive_recall": 0.5,
        "negative_cases": 3,
        "negative_safe_cases": 2,
        "negative_safety_rate": 0.666667,
        "negative_active_abstain_cases": 1,
        "agent_active_abstain_rate": 0.333333,
        "promotion_accounting_observed_cases": 5,
        "promotion_submitted_candidates": 4,
        "promotion_accepted_candidates": 2,
        "promotion_acceptance_rate": 0.5,
        "normally_terminated_cases": 4,
        "normal_termination_rate": 0.8,
    }


def test_target_evidence_requires_one_issue_to_cover_every_required_span():
    ground_truth = GroundTruth(
        decision="added_issue",
        issue_category="fact_conflict",
        added_issue_count=1,
        allowed_evidence=(("canon", 1, 1), ("chapter", 2, 2)),
    )
    split_across_issues = [
        {"category": "fact_conflict", "spans": (("canon", 1, 1),)},
        {"category": "fact_conflict", "spans": (("chapter", 2, 2),)},
    ]
    one_complete_issue = [
        {
            "category": "fact_conflict",
            "spans": (("canon", 1, 1), ("chapter", 2, 2)),
        }
    ]
    no_evidence_ground_truth = GroundTruth(
        decision="added_issue",
        issue_category="fact_conflict",
        added_issue_count=1,
        allowed_evidence=(),
    )

    assert _target_evidence_match(split_across_issues, ground_truth) is False
    assert _target_evidence_match(one_complete_issue, ground_truth) is True
    assert _target_evidence_match(one_complete_issue, no_evidence_ground_truth) is False

    summary = _build_outcome_summary(
        [
            {
                "expected_decision": "added_issue",
                "expected_issue_category": "fact_conflict",
                "classification": "wrong_candidate",
                "passed": False,
                "run_status": "completed",
                "issue_category_counts": {"fact_conflict": 2},
                "citation_scope_authorized": True,
                "target_evidence_match": False,
                "investigator": {"promotion": {}},
                "capability_isolation": {"verified": True},
            }
        ]
    )
    assert summary["erroneous_added_issue_count"] == 2
    assert summary["system_safe_cases"] == 0


def test_mock_http_e2e_persists_split_outcome_metrics_without_frozen_fixture_read(
    tmp_path, monkeypatch
):
    def document(logical_id, role, content):
        return SourceDocument(
            logical_id=logical_id,
            filename=f"{logical_id}.md",
            content=content,
            role=role,
            story_scope="synthetic",
            line_count=1,
        )

    positive = CasePlan(
        case_id="SYN-P-01",
        split="dev",
        documents=(
            document("canon", "canon", "Synthetic canon."),
            document("chapter", "chapter", "Synthetic contradiction."),
        ),
        ground_truth=GroundTruth(
            decision="added_issue",
            issue_category="fact_conflict",
            added_issue_count=1,
            allowed_evidence=(("canon", 1, 1), ("chapter", 1, 1)),
        ),
    )
    negative = CasePlan(
        case_id="SYN-N-01",
        split="dev",
        documents=(
            document("canon", "canon", "Synthetic canon."),
            document("chapter", "chapter", "Synthetic compatible event."),
        ),
        ground_truth=GroundTruth(
            decision="abstain",
            issue_category="fact_conflict",
            added_issue_count=0,
            allowed_evidence=(),
        ),
    )
    plans = (positive, negative)
    monkeypatch.setattr(
        "scripts.run_evidence_investigator_live.load_case_plans",
        lambda split, dataset_root: (plans, "a" * 64, "b" * 64),
    )

    artifact, _ = run_live_evaluation(options(tmp_path), api=FixtureHttpApi(plans))

    assert artifact["summary"]["passed"] == 2
    assert artifact["summary"]["system_safety_rate"] == 1.0
    assert artifact["summary"]["capability_isolation_verified"] == 2
    assert artifact["summary"]["positive_recall"] == 1.0
    assert artifact["summary"]["negative_safety_rate"] == 1.0
    assert artifact["summary"]["agent_active_abstain_rate"] == 1.0
    assert artifact["summary"]["promotion_acceptance_rate"] == 1.0
    assert artifact["summary"]["normal_termination_rate"] == 1.0
    assert artifact["summary"]["erroneous_added_issue_count"] == 0
    persisted = json.loads((tmp_path / "dev-result.json").read_text(encoding="utf-8"))
    assert persisted["summary"] == artifact["summary"]


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
    [
        "repeated_action",
        "invalid_tool_arguments",
        "multiple_tool_calls",
        "anchor_evidence_reused",
    ],
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


def test_runtime_provenance_parser_is_exact_and_fail_closed():
    value = runtime_provenance()

    assert _safe_runtime_provenance(value) == value

    missing_limit = json.loads(json.dumps(value))
    missing_limit["investigator_limits"].pop("max_reads")
    assert _safe_runtime_provenance(missing_limit) is None

    unexpected_url = json.loads(json.dumps(value))
    unexpected_url["chat_provider"]["base_url"] = "https://relay.example/v1"
    assert _safe_runtime_provenance(unexpected_url) is None


def _qualified_dev_artifact(*, started_at, completed_at, model="fixture-model"):
    provenance = runtime_provenance()
    provenance["chat_provider"]["model_alias"] = model
    git = {
        "commit": FIXTURE_GIT_REVISION,
        "tracked_worktree_dirty": False,
        "stable_during_run": True,
    }
    safe_configuration = {
        "artifact_schema": "evidence-investigator-live-http-v3",
        "dataset_id": "evidence-investigator-live-v1",
        "split": "dev",
        "manifest_sha256": live_runner.PINNED_MANIFEST_SHA256,
        "freeze_sha256": live_runner.PINNED_FREEZE_SHA256,
        "case_timeout_seconds": 180.0,
        "request_timeout_seconds": 30.0,
        "poll_interval_seconds": 1.0,
        "runner_git": git,
        "service_observation": {
            "service_ok": True,
            "model_configured": True,
            "thinking_configured": True,
            "thinking_mode": "disabled",
            "runtime_provenance": provenance,
        },
        "service_reported_effective_limits": {
            "observed_cases": 8,
            "consistent_across_cases": True,
            "values": {},
            "distinct_snapshot_count": 1,
        },
        "worker_runtime_provenance_matches_api_cases": 8,
        "rag_configuration_verified_cases": 8,
    }
    fingerprint = live_runner._sha256_bytes(
        live_runner._canonical_json_bytes(safe_configuration)
    )
    provenance_gate = {
        "passed": True,
        "reason_codes": [],
        "api_runtime_provenance": provenance,
        "worker_runtime_provenance_observed_cases": 8,
        "worker_runtime_provenance_matches_api_cases": 8,
        "rag_configuration_verified_cases": 8,
        "case_count": 8,
    }
    summary = {
        "case_count": 8,
        "passed": 5,
        "positive_cases": 5,
        "positive_passed": 3,
        "negative_cases": 3,
        "negative_safe_cases": 3,
        "negative_active_abstain_cases": 2,
        "normally_terminated_cases": 7,
        "erroneous_added_issue_count": 0,
        "final_output_observed_cases": 8,
        "capability_isolation_verified": 8,
        "promotion_accounting_observed_cases": 8,
    }
    return {
        "schema_version": "evidence-investigator-live-http-v3",
        "dataset_id": "evidence-investigator-live-v1",
        "split": "dev",
        "execution_source": "live_http_service",
        "started_at": started_at,
        "completed_at": completed_at,
        "fixture": {
            "manifest_sha256": live_runner.PINNED_MANIFEST_SHA256,
            "freeze_sha256": live_runner.PINNED_FREEZE_SHA256,
        },
        "git": git,
        "safe_configuration": safe_configuration,
        "safe_configuration_fingerprint": fingerprint,
        "reproducibility_fingerprint": fingerprint,
        "provenance_gate": provenance_gate,
        "development_gate": live_runner._build_development_gate(
            "dev", summary, provenance_gate
        ),
        "summary": summary,
        "cases": [{} for _ in range(8)],
        "privacy_boundary": {
            "source_bodies_persisted": False,
            "provider_payloads_persisted": False,
            "credentials_persisted": False,
            "service_address_persisted": False,
        },
        "diagnostic_boundary": "fixture",
    }


def test_dev_pair_requires_v3_qualified_identical_nonoverlapping_runs():
    first = _qualified_dev_artifact(
        started_at="2026-09-12T10:00:00+00:00",
        completed_at="2026-09-12T10:05:00+00:00",
    )
    second = _qualified_dev_artifact(
        started_at="2026-09-12T10:06:00+00:00",
        completed_at="2026-09-12T10:11:00+00:00",
    )

    passed = compare_dev_artifact_payloads(first, second)
    assert passed["passed"] is True
    assert passed["shared_reproducibility_fingerprint"] is not None

    changed = _qualified_dev_artifact(
        started_at="2026-09-12T10:06:00+00:00",
        completed_at="2026-09-12T10:11:00+00:00",
        model="different-model",
    )
    rejected = compare_dev_artifact_payloads(first, changed)
    assert rejected["passed"] is False
    assert "runtime_configuration_changed_between_dev_runs" in rejected[
        "reason_codes"
    ]


def test_dev_pair_treats_v2_and_missing_fields_as_historical_only():
    historical = {
        "schema_version": "evidence-investigator-live-http-v2",
        "dataset_id": "evidence-investigator-live-v1",
        "split": "dev",
    }
    current = _qualified_dev_artifact(
        started_at="2026-09-12T10:06:00+00:00",
        completed_at="2026-09-12T10:11:00+00:00",
    )

    result = compare_dev_artifact_payloads(historical, current)
    assert result["passed"] is False
    assert "first_artifact_historical_only" in result["reason_codes"]

    missing = dict(current)
    missing.pop("provenance_gate")
    result = compare_dev_artifact_payloads(current, missing)
    assert result["passed"] is False
    assert "second_provenance_gate_failed" in result["reason_codes"]

    incomplete = json.loads(json.dumps(current))
    incomplete["provenance_gate"].pop(
        "worker_runtime_provenance_matches_api_cases"
    )
    result = compare_dev_artifact_payloads(current, incomplete)
    assert result["passed"] is False
    assert "second_provenance_gate_failed" in result["reason_codes"]

    incomplete = json.loads(json.dumps(current))
    incomplete["development_gate"]["checks"].pop(
        "runtime_provenance_and_rag_verified"
    )
    result = compare_dev_artifact_payloads(current, incomplete)
    assert result["passed"] is False
    assert "second_development_gate_failed" in result["reason_codes"]


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


def test_dev_plan_loading_never_reads_holdout_source_bytes(monkeypatch):
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def guarded_read_bytes(path):
        if "holdout" in path.parts:
            raise AssertionError("dev verification opened holdout bytes")
        return original_read_bytes(path)

    def guarded_read_text(path, *args, **kwargs):
        if "holdout" in path.parts:
            raise AssertionError("dev plan loading opened holdout text")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    plans, manifest_hash, freeze_hash = load_case_plans("dev")

    assert len(plans) == 8
    assert len(manifest_hash) == 64
    assert len(freeze_hash) == 64


def test_holdout_tamper_fails_before_http(tmp_path):
    copied = tmp_path / "fixture"
    shutil.copytree(DATASET_ROOT, copied)
    target = copied / "holdout" / "EIL-H-01" / "chapter.md"
    target.write_bytes(b"tampered fixture copy\n")
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


def test_frozen_dev_fixture_is_valid_without_running_a_model():
    manifest, manifest_hash, freeze_hash = verify_frozen_dataset(split="dev")
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
